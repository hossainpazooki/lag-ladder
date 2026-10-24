#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# Runs ON gpu-node-1 inside cluster-a job JOB-A (launched with fleet-tools/gpus.sh run). One attempt of the farhan-run
# Step 1 probe: claim the two selected cards, build the probe env (the package's setup_probe.sh), pre-fetch the
# pinned model, re-probe the cards, launch one baseline_probe.py per card (cap 256, cap 768), heartbeat the claims,
# record every exit status, validate outputs, release the claims. State goes to $RUN/state.json (atomic writes).
set -uo pipefail

RUN=$WORK/_gpu_run
WORK=$WORK
PKG=$WORK/2026-10-21-farhan-run
CLAIMS=$CLAIMS_DIR
CL=cluster-a; JOB=JOB-A; NODE=gpu-node-1
TOKENS=(0 1)        # inherited CUDA_VISIBLE_DEVICES tokens, PCI order
UUIDS=(GPU-<gpu01> GPU-<gpu02>)
CAPS=(256 768)
NPROMPTS=200
LL_COMMIT=e32d523
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct; MODEL_REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$HOME/.local/bin:$PATH"           # uv lives here; keeps setup_probe.sh off the curl installer
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export HF_HOME=$HF_HOME
export TOKENIZERS_PARALLELISM=false

ATTEMPT=$(date +%Y%m%d-%H%M%S)
LOG=$RUN/logs/attempt-$ATTEMPT; mkdir -p "$LOG"
OWNER_TOKEN="rrhs-farhan-probe-$NODE-$$-$ATTEMPT"
PY=$WORK/probe-venv/bin/python
CLAIM_FILES=(); PIDS=(); RCS=(); STATE=STARTING; NOTE=""

iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$LOG/run.log"; }

write_state() {
  local i caps="" n
  for i in "${!CAPS[@]}"; do
    n=0; [ -f "$PKG/probe-instruct-${CAPS[$i]}/records.jsonl" ] && n=$(wc -l < "$PKG/probe-instruct-${CAPS[$i]}/records.jsonl")
    caps+="$([ -n "$caps" ] && echo ,)\"${CAPS[$i]}\": {\"device\": \"${UUIDS[$i]}\", \"pid\": \"${PIDS[$i]:-}\", \"rc\": \"${RCS[$i]:-}\", \"records\": $n, \"log\": \"$LOG/probe-${CAPS[$i]}.log\"}"
  done
  printf '{"state": "%s", "updated": "%s", "attempt": "%s", "host": "%s", "job": "%s", "step": "%s", "n": %s, "note": "%s", "caps": {%s}}\n' \
    "$STATE" "$(iso)" "$ATTEMPT" "$(hostname -s)" "$JOB" "${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}" "$NPROMPTS" "$NOTE" "$caps" > "$RUN/.state.json.tmp.$$"
  mv -f "$RUN/.state.json.tmp.$$" "$RUN/state.json"
}

claim_json() {  # idx uuid claimed heartbeat
  printf '{"claimed": "%s", "cluster": "%s", "job": "%s", "node": "%s", "gpu_index": "%s", "selected_gpu_uuids": ["%s"], "owner": "claude session rrhs (farhan-run Step 1 baseline probe; token %s)", "run": "%s", "task": "rrhs:farhan-probe:cap%s", "heartbeat": "%s"}\n' \
    "$3" "$CL" "$JOB" "$NODE" "$1" "$2" "$OWNER_TOKEN" "$RUN" "$5" "$4"
}

claim_age() {   # seconds since a claim file's heartbeat (else claim time); large if unreadable
  local hb t
  hb=$(sed -n 's/.*"heartbeat": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  [ -n "$hb" ] || hb=$(sed -n 's/.*"claimed": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  t=$(date -d "$hb" +%s 2>/dev/null) || { echo 999999; return; }
  echo $(( $(date +%s) - t ))
}

claim_take() {
  local i u8 f g now; now=$(iso)
  for i in "${!UUIDS[@]}"; do
    u8=$(printf '%s' "${UUIDS[$i]#GPU-}" | cut -c1-8)
    for g in "$CLAIMS"/*_"$CL"_"$JOB"_"$NODE"_"$u8"; do      # any other campaign's live claim on this card?
      [ -e "$g" ] || continue
      [ -d "$g" ] && g=$g/owner.json
      if [ "$(claim_age "$g")" -lt 900 ] && ! grep -qF "$OWNER_TOKEN" "$g"; then
        say "CLAIM_COLLISION $(basename "$g") heartbeat $(claim_age "$g")s old"; return 3
      fi
    done
    f="$CLAIMS/rrhs_${CL}_${JOB}_${NODE}_${u8}"
    if ! ( set -o noclobber; claim_json "${TOKENS[$i]}" "${UUIDS[$i]}" "$now" "$now" "${CAPS[$i]}" > "$f" ) 2>/dev/null; then
      if [ "$(claim_age "$f")" -lt 900 ]; then say "CLAIM_COLLISION $(basename "$f") (live rrhs claim)"; return 3; fi
      say "replacing stale own claim $(basename "$f")"
      claim_json "${TOKENS[$i]}" "${UUIDS[$i]}" "$now" "$now" "${CAPS[$i]}" > "$CLAIMS/.$(basename "$f").tmp.$$" && mv -f "$CLAIMS/.$(basename "$f").tmp.$$" "$f"
    fi
    CLAIM_FILES+=("$f"); say "claimed $(basename "$f")"
  done
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

stop_own_workers() {
  local p
  for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$p" 2>/dev/null; done
  sleep 5
  for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -KILL "$p" 2>/dev/null; done
}

on_signal() { say "signal received; stopping this run's workers only"; STATE=INTERRUPTED; NOTE="signal"; stop_own_workers; write_state; claim_release; exit 130; }
trap on_signal INT TERM
trap 'claim_release' EXIT

fail() { STATE=FAILED; NOTE="$*"; say "FAILED: $*"; write_state; exit 1; }

# ---- 0. identity ----------------------------------------------------------------------------------------------
[ "$(hostname -s)" = "$NODE" ] && [ "${SLURM_JOB_ID:-}" = "$JOB" ] || { echo "wrong place: $(hostname -s) job ${SLURM_JOB_ID:-none}"; exit 2; }
say "attempt $ATTEMPT on $(hostname -s) job $SLURM_JOB_ID step ${SLURM_STEP_ID:-?} inherited CVD=${CUDA_VISIBLE_DEVICES:-unset}"
write_state

# ---- 1. claim --------------------------------------------------------------------------------------------------
claim_take || fail "claim collision"
write_state

# ---- 2. environment (package's own setup script) ------------------------------------------------------------------
STATE=SETUP; write_state
if [ -x "$PY" ] && grep -q SETUP_PROBE_DONE "$RUN/logs/setup.ok" 2>/dev/null; then
  say "probe-venv already built (setup.ok present); skipping setup_probe.sh"
else
  say "running setup_probe.sh (LL_COMMIT=$LL_COMMIT) on token ${TOKENS[0]}"
  ( while sleep 30; do claim_beat; done ) & BEAT=$!
  ( cd "$PKG" && CUDA_VISIBLE_DEVICES=${TOKENS[0]} LL_COMMIT=$LL_COMMIT bash "$PKG/setup_probe.sh" ) > "$LOG/setup.log" 2>&1
  src=$?; kill "$BEAT" 2>/dev/null; wait "$BEAT" 2>/dev/null
  say "setup_probe.sh rc=$src; verdict lines:"; grep -E "LAG_LADDER_HEAD|VERSIONS|CUDA_FALLBACK|SELFTEST|^918c|SETUP_PROBE_DONE|STOP" "$LOG/setup.log" | tee -a "$LOG/run.log"
  [ "$src" = 0 ] || fail "setup_probe.sh rc=$src (see $LOG/setup.log)"
  grep -q "^918c7e67f051d41f" "$LOG/setup.log" || fail "prompts hash line missing"
  grep -q "SELFTEST_OK" "$LOG/setup.log" || fail "selftest line missing"
  grep -q "^VERSIONS .* True " "$LOG/setup.log" || grep -q "^VERSIONS [^ ]* True" "$LOG/setup.log" || fail "torch does not see CUDA"
  cp "$LOG/setup.log" "$RUN/logs/setup.ok"
fi
[ "$(git -C "$WORK/lag-ladder" rev-parse --short=7 HEAD)" = "$LL_COMMIT" ] || fail "lag-ladder not at $LL_COMMIT"
claim_beat

# ---- 3. pre-fetch the pinned model (one download, not two racing ones) ---------------------------------------------
say "pre-fetching $MODEL_ID@$MODEL_REV"
"$PY" -I -c "from huggingface_hub import snapshot_download as s; print('MODEL_CACHED', s('$MODEL_ID', revision='$MODEL_REV'))" >> "$LOG/run.log" 2>&1 \
  || fail "model pre-fetch failed"
claim_beat

# ---- 4. live re-probe of the selected cards, right before launch --------------------------------------------------
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits > "$LOG/preprobe.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$LOG/preprobe_apps.csv"
for u in "${UUIDS[@]}"; do
  used=$(awk -F', ' -v u="$u" '$2==u {print $3}' "$LOG/preprobe.csv")
  [ -n "$used" ] || fail "selected card $u not visible"
  [ "$used" -lt 500 ] || fail "selected card $u holds ${used} MiB before launch"
  ! grep -qF "$u" "$LOG/preprobe_apps.csv" || fail "selected card $u has a compute process before launch"
done
for i in "${!TOKENS[@]}"; do      # the narrowed mask must name exactly the selected card at runtime
  got=$(CUDA_VISIBLE_DEVICES=${TOKENS[$i]} "$PY" -I -c "
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print('GPU-' + str(torch.cuda.get_device_properties(0).uuid))" 2>>"$LOG/cudacheck.txt" | tail -1)
  echo "token ${TOKENS[$i]} -> $got" >> "$LOG/cudacheck.txt"
  [ "$got" = "${UUIDS[$i]}" ] || fail "token ${TOKENS[$i]} resolves to '$got', expected ${UUIDS[$i]}"
done
say "selected cards free and mapped (MiB used: $(awk -F', ' '{printf "%s:%s ", $1, $3}' "$LOG/preprobe.csv"))"

# ---- 5. launch, one process per card ------------------------------------------------------------------------------
cd "$PKG"
for i in "${!CAPS[@]}"; do
  ( CUDA_VISIBLE_DEVICES=${TOKENS[$i]} exec "$PY" baseline_probe.py --lag-ladder ../lag-ladder \
      --prompts gsm8k_test_prompts.json --n "$NPROMPTS" --max-new "${CAPS[$i]}" --out "probe-instruct-${CAPS[$i]}" ) \
      > "$LOG/probe-${CAPS[$i]}.log" 2>&1 &
  PIDS[$i]=$!
  say "launched cap ${CAPS[$i]} pid ${PIDS[$i]} on token ${TOKENS[$i]} (${UUIDS[$i]})"
done
STATE=RUNNING; write_state

# ---- 6. supervise ---------------------------------------------------------------------------------------------------
t0=$(date +%s); verified=0
while :; do
  alive=0
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 1 ] || break
  sleep 30
  claim_beat; write_state
  if [ "$verified" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 60 ]; then
    { echo "== +60s $(iso)"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } > "$LOG/verify60.txt" 2>&1
    verified=1; say "60 s snapshot written to $LOG/verify60.txt"
  fi
  # a dead worker while its sibling still runs is reported at once (the watcher reads state.json)
  for i in "${!PIDS[@]}"; do
    if [ -z "${RCS[$i]:-}" ] && ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      wait "${PIDS[$i]}"; RCS[$i]=$?; say "cap ${CAPS[$i]} exited rc=${RCS[$i]}"; write_state
    fi
  done
done
for i in "${!PIDS[@]}"; do
  if [ -z "${RCS[$i]:-}" ]; then wait "${PIDS[$i]}"; RCS[$i]=$?; say "cap ${CAPS[$i]} exited rc=${RCS[$i]}"; fi
done

# ---- 7. validate ----------------------------------------------------------------------------------------------------
ok=1; msgs=""
for i in "${!CAPS[@]}"; do
  c=${CAPS[$i]}; d="$PKG/probe-instruct-$c"
  [ "${RCS[$i]}" = 0 ] || { ok=0; msgs+="cap$c rc=${RCS[$i]}; "; continue; }
  v=$("$PY" -I -c "
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
idx = [r['index'] for r in recs]
p = json.load(open(sys.argv[2]))
ok = len(recs) == int(sys.argv[3]) and sorted(idx) == list(range(int(sys.argv[3]))) and p['n'] == len(recs) and p['max_new'] == int(sys.argv[4])
print('VALID' if ok else f'INVALID n={len(recs)} uniq={len(set(idx))} probe_n={p[\"n\"]} max_new={p[\"max_new\"]}')
" "$d/records.jsonl" "$d/probe.json" "$NPROMPTS" "$c" 2>&1 | tail -1)
  grep -q "^PROBE_DONE n $NPROMPTS max_new $c " "$LOG/probe-$c.log" || v="$v; no PROBE_DONE line"
  say "cap $c: $v"
  [ "$v" = VALID ] || { ok=0; msgs+="cap$c $v; "; }
done
if [ "$ok" = 1 ]; then
  ( cd "$PKG" && sha256sum probe-instruct-*/* ) > "$LOG/sha256sums.txt"
  grep -h "^PROBE_DONE" "$LOG"/probe-*.log | tee -a "$LOG/run.log"
  STATE=SUCCEEDED; NOTE="both caps valid"
else
  STATE=FAILED; NOTE="$msgs"
fi
write_state; say "final state $STATE $NOTE"
claim_release
[ "$STATE" = SUCCEEDED ]
