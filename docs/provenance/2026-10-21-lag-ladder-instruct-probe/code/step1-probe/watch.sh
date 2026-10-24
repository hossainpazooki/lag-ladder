#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# Experiment watcher for the farhan-run Step 1 probe (runs in tmux on the login node; holds no GPU).
# Every 60 s: reads state.json, logs progress, flags a stall (no new record in 20 min while RUNNING), and if the
# driver's Slurm step is gone without a terminal state, marks the run INTERRUPTED and releases this run's claims.
# Exits on SUCCEEDED / FAILED / INTERRUPTED, or after 24 h.
set -uo pipefail
RUN=$WORK/_gpu_run
CLAIMS=$CLAIMS_DIR
JOB=JOB-A
iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$RUN/watch.log"; }
field() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$RUN/state.json" "$1" 2>/dev/null; }

say "watcher up on $(hostname -s) pid $$"
start=$(date +%s); last_total=-1; last_change=$(date +%s); missing=0
while [ $(( $(date +%s) - start )) -lt 86400 ]; do
  date +%s > "$RUN/watch.heartbeat"
  if [ ! -f "$RUN/state.json" ]; then say "no state.json yet"; sleep 60; continue; fi
  st=$(field "d['state']"); step=$(field "d['step']"); upd=$(field "d['updated']")
  prog=$(field "' '.join(f\"cap{k}={v['records']}/{d['n']} rc={v['rc'] or '-'}\" for k,v in d['caps'].items())")
  total=$(field "sum(v['records'] for v in d['caps'].values())")
  say "STATE $st step $step updated $upd | $prog"
  case "$st" in
    SUCCEEDED|FAILED|INTERRUPTED) say "TERMINAL $st note: $(field "d['note']")"; exit 0 ;;
  esac
  # progress / stall
  if [ "${total:-0}" != "$last_total" ]; then last_total=$total; last_change=$(date +%s)
  elif [ "$st" = RUNNING ] && [ $(( $(date +%s) - last_change )) -ge 1200 ]; then
    say "STALL no new record for $(( ($(date +%s) - last_change) / 60 )) min"
  fi
  # driver step alive?
  if [ -n "$step" ] && [ "${step#*.}" != "" ]; then
    if squeue -h -s -j "$JOB" -o %i 2>/dev/null | grep -qx "$step"; then missing=0; else missing=$((missing + 1)); fi
    if [ "$missing" -ge 2 ]; then
      say "DRIVER_GONE step $step not in squeue twice while state=$st; marking INTERRUPTED"
      python3 - "$RUN/state.json" <<'PY'
import json, sys, os, datetime
p = sys.argv[1]; d = json.load(open(p))
if d["state"] not in ("SUCCEEDED", "FAILED", "INTERRUPTED"):
    d["state"] = "INTERRUPTED"; d["note"] = "watcher: driver step vanished without a terminal state"
    d["updated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t = p + ".watch.tmp"; json.dump(d, open(t, "w")); os.replace(t, p)
PY
      for f in "$CLAIMS"/rrhs_cluster-a_${JOB}_gpu-node-1_*; do
        [ -f "$f" ] && grep -qF "\"run\": \"$RUN\"" "$f" && rm -f "$f" && say "released $(basename "$f")"
      done
      exit 1
    fi
  fi
  sleep 60
done
say "watcher reached its 24 h limit"
