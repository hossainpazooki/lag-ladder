#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# Watcher for the use-gpu diagnostic task (tmux on the login node; holds no GPU). Every 60 s: follows the newest
# diag-* run's state.json, logs progress, flags a stall (no new record for 20 min while RUNNING), and if the task's
# Slurm step vanishes without a terminal state, marks it INTERRUPTED and releases that run's claims.
# Also exits if the dispatcher finishes without the task ever starting. 24 h limit.
set -uo pipefail
ROOT=$WORK/_gpu_run
DISPATCH_LOG=${1:?usage: watch.sh <dispatch.log>}
CLAIMS=$CLAIMS_DIR
LOG=$ROOT/diag-watch.log
iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$LOG"; }
field() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d}))" "$RUN/state.json" "$1" 2>/dev/null; }

say "diag watcher up on $(hostname -s) pid $$"
start=$(date +%s); last_total=-1; last_change=$(date +%s); missing=0; RUN=""
while [ $(( $(date +%s) - start )) -lt 86400 ]; do
  if [ -z "$RUN" ]; then
    l=$(readlink -f "$ROOT/diag-latest" 2>/dev/null)
    if [ -n "$l" ] && [ "$(stat -c %Y "$l")" -ge "$start" ]; then RUN=$l; say "following $RUN"; fi
  fi
  if [ -z "$RUN" ] || [ ! -f "$RUN/state.json" ]; then
    if grep -q "dispatch complete\|GIVING UP" "$DISPATCH_LOG" 2>/dev/null; then say "TERMINAL dispatcher ended before the task started: $(tail -3 "$DISPATCH_LOG" | tr '\n' ' ')"; exit 1; fi
    sleep 60; continue
  fi
  st=$(field "d['state']"); step=$(field "d['step']")
  prog=$(field "' '.join(k + '=' + str(v['records']) + '/' + str(v['want']) + ('' if v['rc'] == '' else ' rc=' + v['rc']) for k, v in sorted(d['tasks'].items()))")
  total=$(field "sum(v['records'] for v in d['tasks'].values())")
  say "STATE $st step $step | $prog"
  case "$st" in SUCCEEDED|FAILED|INTERRUPTED) say "TERMINAL $st note: $(field "d['note']")"; exit 0 ;; esac
  if [ "${total:-0}" != "$last_total" ]; then last_total=$total; last_change=$(date +%s)
  elif [ "$st" = RUNNING ] && [ $(( $(date +%s) - last_change )) -ge 1200 ]; then
    say "STALL no new record for $(( ($(date +%s) - last_change) / 60 )) min"
  fi
  job=${step%%.*}; cl=$(field "d['host']" | grep -q '^<cluster-b-node-prefix>' && echo "-M cluster-b" || echo "")
  if squeue $cl -h -s -j "$job" -o %i 2>/dev/null | grep -qx "$step"; then missing=0; else missing=$((missing + 1)); fi
  if [ "$missing" -ge 2 ]; then
    say "DRIVER_GONE step $step not in squeue twice while state=$st; marking INTERRUPTED"
    python3 - "$RUN/state.json" <<'PY'
import json, sys, os, datetime
p = sys.argv[1]; d = json.load(open(p))
if d["state"] not in ("SUCCEEDED", "FAILED", "INTERRUPTED"):
    d["state"] = "INTERRUPTED"; d["note"] = "watcher: task step vanished without a terminal state"
    d["updated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t = p + ".watch.tmp"; json.dump(d, open(t, "w")); os.replace(t, p)
PY
    for f in "$CLAIMS"/rrhs_*; do [ -f "$f" ] && grep -qF "\"run\": \"$RUN\"" "$f" && rm -f "$f" && say "released $(basename "$f")"; done
    exit 1
  fi
  sleep 60
done
say "watcher reached its 24 h limit"
