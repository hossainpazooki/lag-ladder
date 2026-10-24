#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# One attempt of the cap-256 argmax provenance check (the 768 check's twin: diag/argmax_check.py, unedited), run
# inside a held allocation by queue_watch.sh. Picks ONE assigned card that is free (no compute process, < 500 MiB)
# and carries no live claim from anyone, claims it, re-probes it, checks CUDA in the probe venv, runs the check,
# validates, and only then copies the result next to the 768 one. Claims heartbeat every 30 s and are released on
# every exit path this script controls.
# Exit: 0 done and valid (or already done) | 75 no usable card on this node | 76 environment unusable here | 1 failed.
set -uo pipefail

ROOT=$WORK
PKG=$ROOT/2026-10-21-farhan-run
LL=$ROOT/lag-ladder
PY=$ROOT/probe-venv/bin/python
DIAG=$ROOT/_gpu_run/diag
BASE=$ROOT/_gpu_run/argmax256
OUT=$ROOT/diagnostics/no-rep-penalty
FINAL=$OUT/argmax_check_256.json
CLAIMS=$CLAIMS_DIR
SETS=("registered256=$PKG/probe-instruct-256/records.jsonl" "norp256=$OUT/probe-norp-256/records.jsonl")

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

NODE=$(hostname -s); JOB=${SLURM_JOB_ID:?run inside an allocation}; CL=${SLURM_CLUSTER_NAME:-cluster-a}
ATTEMPT=$(date +%Y%m%d-%H%M%S)-$NODE
RUN=$BASE/attempt-$ATTEMPT; mkdir -p "$RUN/logs"
OWNER_TOKEN="rrhs-argmax256-$NODE-$$-$ATTEMPT"
CLAIM_FILE=""; TOK=""; UUID=""; PID=""; RC=""; STATE=STARTING; NOTE=""

iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$RUN/run.log"; }
write_state() {
  printf '{"state": "%s", "updated": "%s", "attempt": "%s", "host": "%s", "cluster": "%s", "job": "%s", "step": "%s", "device_token": "%s", "device": "%s", "pid": "%s", "rc": "%s", "note": "%s"}\n' \
    "$STATE" "$(iso)" "$ATTEMPT" "$NODE" "$CL" "$JOB" "${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}" "$TOK" "$UUID" "$PID" "$RC" "$NOTE" > "$RUN/.state.tmp.$$"
  mv -f "$RUN/.state.tmp.$$" "$RUN/state.json"
}
claim_age() {
  local hb t
  hb=$(sed -n 's/.*"heartbeat": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  [ -n "$hb" ] || hb=$(sed -n 's/.*"claimed": *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -1)
  t=$(date -d "$hb" +%s 2>/dev/null) || { echo 999999; return; }
  echo $(( $(date +%s) - t ))
}
live_claim_on() {   # any claim younger than 900 s that names this card (by full UUID or by <node>_<uuid8>), any owner/job,
  local u=$1 u8=$2 mine=${3:-} g f   # other than <mine>
  for g in "$CLAIMS"/*_"$NODE"_"$u8" $(grep -lF "$u" "$CLAIMS"/* "$CLAIMS"/*/owner.json 2>/dev/null); do
    [ -e "$g" ] || continue; [ "$g" = "$mine" ] && continue; f=$g; [ -d "$f" ] && f=$f/owner.json
    [ -f "$f" ] && [ "$(claim_age "$f")" -lt 900 ] && { basename "$g"; return 0; }
  done
  return 1
}
claim_beat() {
  [ -n "$CLAIM_FILE" ] && [ -f "$CLAIM_FILE" ] && grep -qF "$OWNER_TOKEN" "$CLAIM_FILE" || return 0
  sed "s/\"heartbeat\": *\"[^\"]*\"/\"heartbeat\": \"$(iso)\"/" "$CLAIM_FILE" > "$CLAIMS/.$(basename "$CLAIM_FILE").tmp.$$" \
    && mv -f "$CLAIMS/.$(basename "$CLAIM_FILE").tmp.$$" "$CLAIM_FILE"
}
claim_release() {
  [ -n "$CLAIM_FILE" ] && [ -f "$CLAIM_FILE" ] && grep -qF "$OWNER_TOKEN" "$CLAIM_FILE" && rm -f "$CLAIM_FILE" && say "released $(basename "$CLAIM_FILE")"
  CLAIM_FILE=""
}
stop_own() { [ -n "$PID" ] || return 0; kill -TERM "$PID" 2>/dev/null; sleep 5; kill -KILL "$PID" 2>/dev/null; }
finish() { STATE=$1; NOTE=$2; say "$STATE: $NOTE"; write_state; claim_release; exit "$3"; }
on_signal() { say "signal: stopping this attempt's worker only"; stop_own; finish INTERRUPTED signal 130; }
trap on_signal INT TERM
trap claim_release EXIT

say "attempt $ATTEMPT on $NODE ($CL) job $JOB step ${SLURM_STEP_ID:-?}; inherited CVD=${CUDA_VISIBLE_DEVICES:-unset}"
write_state
[ ! -e "$FINAL" ] || finish SKIPPED "$FINAL already exists (never overwrite)" 0
[ -x "$PY" ] || finish UNUSABLE "probe-venv missing" 76
for s in "${SETS[@]}"; do [ -s "${s#*=}" ] || finish FAILED "input missing: ${s#*=}" 1; done

# ---- 1. one free, unclaimed assigned card ---------------------------------------------------------------------------
IFS=',' read -r -a TOK_ALL <<< "${CUDA_VISIBLE_DEVICES:?no inherited CUDA_VISIBLE_DEVICES}"
probe() {
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv,noheader,nounits > "$RUN/logs/$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$RUN/logs/${1}_apps.csv"
}
probe preprobe
# token -> UUID without touching CUDA (a torch mapping would open a context on every assigned card, busy ones too):
# a UUID token is itself; an index token is the nvidia-smi row of that index (PCI order = CUDA_DEVICE_ORDER here).
# The single selected card is re-checked through CUDA before anything is allocated on it.
UUID_ALL=()
for t in "${TOK_ALL[@]}"; do
  if [[ $t == GPU-* ]]; then u=$(awk -F', ' -v u="$t" '$2==u {print $2}' "$RUN/logs/preprobe.csv")
  else u=$(awk -F', ' -v i="$t" '$1==i {print $2}' "$RUN/logs/preprobe.csv"); fi
  [ -n "$u" ] || finish UNUSABLE "token $t does not resolve to a card nvidia-smi lists" 76
  UUID_ALL+=("$u")
done
card_free() {   # <probe name> <uuid>: < 500 MiB used and no compute process
  local used; used=$(awk -F', ' -v u="$2" '$2==u {print $4}' "$RUN/logs/$1.csv")
  [ -n "$used" ] && [ "$used" -lt 500 ] && ! grep -qF "$2" "$RUN/logs/${1}_apps.csv"
}
for i in "${!TOK_ALL[@]}"; do
  u=${UUID_ALL[$i]}; u8=$(printf '%s' "${u#GPU-}" | cut -c1-8)
  card_free preprobe "$u" || { say "skip token ${TOK_ALL[$i]} $u: busy ($(awk -F', ' -v u="$u" '$2==u {print $4 " MiB, " $6 "%"}' "$RUN/logs/preprobe.csv"))"; continue; }
  if c=$(live_claim_on "$u" "$u8"); then say "skip token ${TOK_ALL[$i]} $u: live claim $c"; continue; fi
  f="$CLAIMS/rrhs_${CL}_${JOB}_${NODE}_${u8}"; now=$(iso)
  if ! ( set -o noclobber; printf '{"claimed": "%s", "cluster": "%s", "job": "%s", "node": "%s", "gpu_index": "%s", "selected_gpu_uuids": ["%s"], "owner": "claude session rrhs (farhan-run cap-256 argmax check; token %s)", "run": "%s", "task": "rrhs:farhan-argmax256", "heartbeat": "%s"}\n' \
      "$now" "$CL" "$JOB" "$NODE" "${TOK_ALL[$i]}" "$u" "$OWNER_TOKEN" "$RUN" "$now" > "$f" ) 2>/dev/null; then
    say "skip token ${TOK_ALL[$i]} $u: could not create $(basename "$f")"; continue
  fi
  CLAIM_FILE=$f
  sleep 5; probe postclaim; c=""                                     # re-probe after taking the claim
  if card_free postclaim "$u" && ! c=$(live_claim_on "$u" "$u8" "$f"); then
    TOK=${TOK_ALL[$i]}; UUID=$u; break
  fi
  say "token ${TOK_ALL[$i]} $u taken between probe and claim (${c:-busy}); releasing"; claim_release
done
[ -n "$TOK" ] || finish NO_CARD "no free unclaimed card on $NODE" 75
say "selected token $TOK $UUID ($(awk -F', ' -v u="$UUID" '$2==u {print $3 ", " $5 " MiB, driver " $7}' "$RUN/logs/postclaim.csv"))"
write_state

# ---- 2. CUDA in the probe venv on the selected card ------------------------------------------------------------------
if ! CUDA_VISIBLE_DEVICES=$TOK "$PY" -I -c "
import torch, sys
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'cuda'
seen = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid)
assert seen == sys.argv[1], f'visible {seen} is not the selected {sys.argv[1]}'   # before any allocation
x = torch.ones(1024, 1024, device='cuda'); torch.cuda.synchronize()
print('CUDA_OK', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), seen, float((x @ x).sum()))
" "$UUID" > "$RUN/logs/cudacheck.txt" 2>&1; then
  finish UNUSABLE "CUDA check failed on token $TOK (see $RUN/logs/cudacheck.txt)" 76
fi
grep -qF "$UUID" "$RUN/logs/cudacheck.txt" || finish UNUSABLE "CUDA-visible device is not the selected $UUID" 76
say "$(head -1 "$RUN/logs/cudacheck.txt")"

# ---- 3. run, heartbeat, 60 s check ------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$TOK "$PY" -I -B "$DIAG/argmax_check.py" "$PKG" "$LL" "$RUN/argmax_check_256.json" "${SETS[@]}" \
  > "$RUN/logs/argmax_check.log" 2> "$RUN/logs/argmax_check.err" &
PID=$!; STATE=RUNNING; write_state; say "launched argmax_check.py pid $PID on token $TOK"
t0=$(date +%s); snap=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 10; claim_beat; write_state
  if [ "$snap" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 60 ] && kill -0 "$PID" 2>/dev/null; then
    { echo "== +60s $(iso)"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } > "$RUN/logs/verify60.txt" 2>&1
    snap=1
    if awk -F', ' -v u="$UUID" '$1==u' "$RUN/logs/verify60.txt" | grep -q .; then say "+60 s: worker resident on $UUID"
    else say "WARNING +60 s: no compute process listed on $UUID (see logs/verify60.txt)"; fi
  fi
done
wait "$PID"; RC=$?; say "argmax_check.py exited rc=$RC after $(( $(date +%s) - t0 )) s"
cat "$RUN/logs/argmax_check.log" | tee -a "$RUN/run.log"
[ "$RC" = 0 ] || finish FAILED "argmax_check.py rc=$RC (see $RUN/logs/argmax_check.err)" 1

# ---- 4. validate, then publish next to the 768 result ----------------------------------------------------------------
STATE=VALIDATING; write_state
v=$("$PY" -I -B - "$RUN/argmax_check_256.json" "$UUID" <<'PY' 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
need = {"registered256", "norp256"}
assert set(d["sets"]) == need, f"sets {sorted(d['sets'])}"
for k in sorted(need):
    s = d["sets"][k]
    assert s["records"] == 200, f"{k} records {s['records']}"
    assert s["ids_recovered"] >= 190, f"{k} ids_recovered {s['ids_recovered']}"
    assert s["tokens"] > 0 and 0 <= s["frac_raw"] <= 1 and 0 <= s["frac_penalized"] <= 1, f"{k} fractions"
    print(f"VALID {k} records {s['records']} recovered {s['ids_recovered']} tokens {s['tokens']} "
          f"raw {s['frac_raw']:.4f} penalized {s['frac_penalized']:.4f} all_raw {s['records_all_raw']} all_pen {s['records_all_penalized']}")
print("VALIDATE_OK", d["device"])
PY
)
echo "$v" | tee -a "$RUN/run.log"
grep -q '^VALIDATE_OK' <<< "$v" || finish FAILED "validation failed" 1
( set -o noclobber; cat "$RUN/argmax_check_256.json" > "$FINAL" ) 2>/dev/null || finish FAILED "$FINAL appeared meanwhile; not overwriting" 1
cmp -s "$RUN/argmax_check_256.json" "$FINAL" || finish FAILED "copy to $FINAL differs" 1
( cd "$OUT" && sha256sum argmax_check_256.json ) | tee "$RUN/argmax_check_256.sha256" >> "$RUN/run.log"
finish SUCCEEDED "cap-256 argmax check valid on $NODE token $TOK ($UUID); published $FINAL" 0
