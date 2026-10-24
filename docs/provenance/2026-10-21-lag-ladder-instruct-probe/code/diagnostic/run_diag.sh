#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# use-gpu task: the no-repetition-penalty diagnostic of the farhan-run Step 1 probe, on whatever node the dispatcher
# gives it (ONLY= restricts that to RTX PRO 4000 Blackwell nodes, the registered run's card). Uses every selected card:
# the node's assigned cards that are free (no compute process, < 500 MiB) and carry no live foreign claim. Caps 256
# and 768 are each split into contiguous prompt shards, one process per card; shards merge back into one folder per
# cap with the package's own summarize(). Claims heartbeat every 30 s and are released at the end. Exit 0 only if
# every shard exited 0 and both merged folders validate.
set -uo pipefail

ROOT=$WORK
PKG=$ROOT/2026-10-21-farhan-run
LL=$ROOT/lag-ladder
PY=$ROOT/probe-venv/bin/python
DIAG=$ROOT/_gpu_run/diag
OUT=$ROOT/diagnostics/no-rep-penalty
CLAIMS=$CLAIMS_DIR
NPROMPTS=200; CAPS=(256 768)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=$HF_HOME
export TOKENIZERS_PARALLELISM=false

NODE=$(hostname -s); JOB=${SLURM_JOB_ID:?run inside an allocation}; CL=${SLURM_CLUSTER_NAME:-cluster-a}
ATTEMPT=$(date +%Y%m%d-%H%M%S)
RUN=$ROOT/_gpu_run/diag-$ATTEMPT; mkdir -p "$RUN/logs"
OWNER_TOKEN="rrhs-diag-norp-$NODE-$$-$ATTEMPT"
CLAIM_FILES=(); PIDS=(); NAMES=(); RCS=(); STATE=STARTING; NOTE=""

iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$RUN/run.log"; }
claim_age() {
  local hb t
  hb=$(sed -n 's/.*"heartbeat": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  [ -n "$hb" ] || hb=$(sed -n 's/.*"claimed": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  t=$(date -d "$hb" +%s 2>/dev/null) || { echo 999999; return; }
  echo $(( $(date +%s) - t ))
}
write_state() {
  local i tasks="" n
  for i in "${!NAMES[@]}"; do
    n=0; [ -f "$RUN/shards/${NAMES[$i]}/records.jsonl" ] && n=$(wc -l < "$RUN/shards/${NAMES[$i]}/records.jsonl")
    tasks+="$([ -n "$tasks" ] && echo ,)\"${NAMES[$i]}\": {\"device\": \"${DEV_UUID[$i]:-}\", \"pid\": \"${PIDS[$i]:-}\", \"rc\": \"${RCS[$i]:-}\", \"records\": $n, \"want\": ${WANT[$i]:-0}}"
  done
  printf '{"state": "%s", "updated": "%s", "attempt": "%s", "host": "%s", "job": "%s", "step": "%s", "note": "%s", "tasks": {%s}}\n' \
    "$STATE" "$(iso)" "$ATTEMPT" "$NODE" "$JOB" "${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}" "$NOTE" "$tasks" > "$RUN/.state.tmp.$$"
  mv -f "$RUN/.state.tmp.$$" "$RUN/state.json"
}
claim_beat() {
  local f now; now=$(iso)
  for f in ${CLAIM_FILES[@]+"${CLAIM_FILES[@]}"}; do
    [ -f "$f" ] && grep -qF "$OWNER_TOKEN" "$f" || continue
    sed "s/\"heartbeat\": *\"[^\"]*\"/\"heartbeat\": \"$now\"/" "$f" > "$CLAIMS/.$(basename "$f").tmp.$$" && mv -f "$CLAIMS/.$(basename "$f").tmp.$$" "$f"
  done
}
claim_release() {
  local f
  for f in ${CLAIM_FILES[@]+"${CLAIM_FILES[@]}"}; do
    [ -f "$f" ] && grep -qF "$OWNER_TOKEN" "$f" && rm -f "$f" && say "released $(basename "$f")"
  done
  CLAIM_FILES=()
}
stop_own() { local p; for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$p" 2>/dev/null; done; sleep 5
             for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -KILL "$p" 2>/dev/null; done; }
on_signal() { say "signal: stopping this run's workers only"; STATE=INTERRUPTED; NOTE=signal; stop_own; write_state; claim_release; exit 130; }
trap on_signal INT TERM
trap claim_release EXIT
fail() { STATE=FAILED; NOTE="$*"; say "FAILED: $*"; write_state; exit 1; }

ln -sfn "$RUN" "$ROOT/_gpu_run/diag-latest"
say "attempt $ATTEMPT on $NODE ($CL) job $JOB step ${SLURM_STEP_ID:-?}; dispatcher N=${N:-unset}; inherited CVD=${CUDA_VISIBLE_DEVICES:-unset}"
[ -x "$PY" ] || fail "probe-venv missing"
[ ! -e "$OUT" ] || fail "$OUT already exists (never overwrite an earlier diagnostic)"

# ---- 1. which cards: map inherited tokens to UUIDs, then keep the free, unclaimed ones ---------------------------
IFS=',' read -r -a TOK_ALL <<< "${CUDA_VISIBLE_DEVICES:?no inherited CUDA_VISIBLE_DEVICES}"
mapfile -t UUID_ALL < <("$PY" -I -c "
import torch
for i in range(torch.cuda.device_count()): print('GPU-' + str(torch.cuda.get_device_properties(i).uuid))" 2>>"$RUN/logs/map.err")
[ "${#UUID_ALL[@]}" = "${#TOK_ALL[@]}" ] || fail "token/UUID count mismatch: ${#TOK_ALL[@]} tokens, ${#UUID_ALL[@]} UUIDs"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits > "$RUN/logs/preprobe.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$RUN/logs/preprobe_apps.csv"
SEL_TOK=(); SEL_UUID=()
for i in "${!TOK_ALL[@]}"; do
  u=${UUID_ALL[$i]}; u8=$(printf '%s' "${u#GPU-}" | cut -c1-8)
  used=$(awk -F', ' -v u="$u" '$2==u {print $4}' "$RUN/logs/preprobe.csv")
  [ -n "$used" ] && [ "$used" -lt 500 ] || { say "skip token ${TOK_ALL[$i]} $u: ${used:-?} MiB used"; continue; }
  grep -qF "$u" "$RUN/logs/preprobe_apps.csv" && { say "skip token ${TOK_ALL[$i]} $u: compute process"; continue; }
  live=""
  for g in "$CLAIMS"/*_"$CL"_"$JOB"_"$NODE"_"$u8"; do
    [ -e "$g" ] || continue; [ -d "$g" ] && g=$g/owner.json
    [ "$(claim_age "$g")" -lt 900 ] && live=$(basename "$g")
  done
  [ -z "$live" ] || { say "skip token ${TOK_ALL[$i]} $u: live claim $live"; continue; }
  f="$CLAIMS/rrhs_${CL}_${JOB}_${NODE}_${u8}"; now=$(iso)
  if ! ( set -o noclobber; printf '{"claimed": "%s", "cluster": "%s", "job": "%s", "node": "%s", "gpu_index": "%s", "selected_gpu_uuids": ["%s"], "owner": "claude session rrhs (farhan-run no-repetition-penalty diagnostic; token %s)", "run": "%s", "task": "rrhs:farhan-diag-norp", "heartbeat": "%s"}\n' \
        "$now" "$CL" "$JOB" "$NODE" "${TOK_ALL[$i]}" "$u" "$OWNER_TOKEN" "$RUN" "$now" > "$f" ) 2>/dev/null; then
    say "skip token ${TOK_ALL[$i]} $u: could not create $f"; continue
  fi
  CLAIM_FILES+=("$f"); SEL_TOK+=("${TOK_ALL[$i]}"); SEL_UUID+=("$u")
done
G=${#SEL_TOK[@]}
say "selected $G card(s): ${SEL_TOK[*]:-none} ($(awk -F', ' '{print $3}' "$RUN/logs/preprobe.csv" | sort -u | tr '\n' ' '))"
[ "$G" -ge 1 ] || fail "no free unclaimed card on $NODE"
grep -q "RTX PRO 4000 Blackwell" "$RUN/logs/preprobe.csv" || say "WARNING: not the registered run's card model"

# ---- 2. plan: cap 768 (slower) gets the extra card when G is odd; G=1 runs both caps on one card in turn ---------
plan_shards() {   # cap count -> NAMES/WANT/STARTS entries
  local cap=$1 k=$2 j s=0 len
  for ((j = 0; j < k; j++)); do
    len=$(( NPROMPTS / k + (j < NPROMPTS % k ? 1 : 0) ))
    NAMES+=("cap${cap}-s$(printf '%02d' "$j")"); WANT+=("$len"); STARTS+=("$s"); SCAP+=("$cap"); s=$((s + len))
  done
}
NAMES=(); WANT=(); STARTS=(); SCAP=(); DEV_TOK=(); DEV_UUID=()
if [ "$G" -ge 2 ]; then
  k256=$(( G / 2 )); k768=$(( G - G / 2 ))
  plan_shards 256 "$k256"; plan_shards 768 "$k768"
  for i in "${!NAMES[@]}"; do DEV_TOK[$i]=${SEL_TOK[$i]}; DEV_UUID[$i]=${SEL_UUID[$i]}; done
else
  plan_shards 256 1; plan_shards 768 1
  DEV_TOK=("${SEL_TOK[0]}" "${SEL_TOK[0]}"); DEV_UUID=("${SEL_UUID[0]}" "${SEL_UUID[0]}")
fi
for i in "${!NAMES[@]}"; do say "plan ${NAMES[$i]}: prompts ${STARTS[$i]}..$(( STARTS[$i] + WANT[$i] - 1 )) on token ${DEV_TOK[$i]} (${DEV_UUID[$i]})"; done
write_state

# ---- 3. launch -------------------------------------------------------------------------------------------------------
mkdir -p "$RUN/shards"
launch() {
  local i=$1
  ( cd "$PKG" && CUDA_VISIBLE_DEVICES=${DEV_TOK[$i]} exec "$PY" -I -B "$DIAG/norp_probe.py" "$PKG" --lag-ladder "$LL" \
      --prompts "$PKG/gsm8k_test_prompts.json" --start "${STARTS[$i]}" --n "${WANT[$i]}" --max-new "${SCAP[$i]}" \
      --out "$RUN/shards/${NAMES[$i]}" ) > "$RUN/logs/${NAMES[$i]}.log" 2>&1 &
  PIDS[$i]=$!; say "launched ${NAMES[$i]} pid ${PIDS[$i]} on token ${DEV_TOK[$i]}"
}
if [ "$G" -ge 2 ]; then for i in "${!NAMES[@]}"; do launch "$i"; done; else launch 0; fi
STATE=RUNNING; write_state

# ---- 4. supervise --------------------------------------------------------------------------------------------------
t0=$(date +%s); snap=0
while :; do
  sleep 30; claim_beat
  for i in "${!NAMES[@]}"; do
    [ -n "${PIDS[$i]:-}" ] && [ -z "${RCS[$i]:-}" ] || continue
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      wait "${PIDS[$i]}"; RCS[$i]=$?; say "${NAMES[$i]} exited rc=${RCS[$i]}"
      if [ "$G" -lt 2 ] && [ "$i" = 0 ]; then launch 1; fi     # single card: second cap after the first
    fi
  done
  write_state
  if [ "$snap" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 60 ]; then
    { echo "== +60s $(iso)"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } > "$RUN/logs/verify60.txt" 2>&1
    snap=1; say "60 s snapshot: $RUN/logs/verify60.txt"
  fi
  done_all=1; for i in "${!NAMES[@]}"; do [ -n "${RCS[$i]:-}" ] || done_all=0; done
  [ "$done_all" = 1 ] && break
done

# ---- 5. merge + validate -----------------------------------------------------------------------------------------
bad=""; for i in "${!NAMES[@]}"; do [ "${RCS[$i]}" = 0 ] || bad+="${NAMES[$i]} rc=${RCS[$i]}; "; done
[ -z "$bad" ] || fail "shard failures: $bad"
mkdir -p "$OUT"
for c in "${CAPS[@]}"; do
  dirs=(); for i in "${!NAMES[@]}"; do [ "${SCAP[$i]}" = "$c" ] && dirs+=("$RUN/shards/${NAMES[$i]}"); done
  m=$("$PY" -I -B "$DIAG/merge_diag.py" "$PKG" "$LL" "$c" "$NPROMPTS" "$OUT/probe-norp-$c" "${dirs[@]}" 2>&1)
  echo "$m" | tee -a "$RUN/run.log"
  grep -q "^MERGE_OK" <<< "$m" || fail "merge cap $c failed"
done
grep -h "^DIAG_GENCFG" "$RUN"/logs/cap*.log | sort | uniq -c | tee -a "$RUN/run.log"
( cd "$OUT" && sha256sum probe-norp-*/* ) > "$OUT/sha256sums.txt"
claim_beat

# ---- 6. which decoding made each 768 record set: raw vs penalized argmax, teacher-forced on card 0 ----------------
STATE=CHECKING; write_state
say "argmax check on token ${DEV_TOK[0]}"
a=$(CUDA_VISIBLE_DEVICES=${DEV_TOK[0]} "$PY" -I -B "$DIAG/argmax_check.py" "$PKG" "$LL" "$OUT/argmax_check.json" \
      "registered768=$PKG/probe-instruct-768/records.jsonl" "norp768=$OUT/probe-norp-768/records.jsonl" 2> "$RUN/logs/argmax_check.err")
arc=$?; echo "$a" | tee -a "$RUN/run.log"
[ "$arc" = 0 ] || say "ARGMAX_CHECK_FAIL rc=$arc (see $RUN/logs/argmax_check.err); diagnostic outputs are still valid"
STATE=SUCCEEDED; NOTE="all shards rc=0, both caps merged; argmax check rc=$arc"; write_state; say "final state $STATE"
claim_release
