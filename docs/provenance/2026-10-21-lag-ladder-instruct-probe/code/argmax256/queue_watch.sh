#!/usr/bin/env bash
# REDACTED COPY for a public repository (see ../../README.md, Redactions): $WORK, $HF_HOME, $SCRATCH,
# $CLAIMS_DIR, $FLEET_TOOLS, cluster-*, gpu-node-*, JOB-* and GPU-<gpuNN> stand for literal site values.
# Not runnable as is; code/CODE_HASHES.txt holds a keyed hash of the bytes that ran.
# Durable queue watcher for the cap-256 argmax check (task.sh). Runs as an srun --overlap step inside the CPU host job
# gpu-watchers (no GPU); a login-node tmux window only holds the srun client. Every POLL s (30 min, the gpu-run rule
# while nothing is usable) it reads the fleet cache for held allocations with an idle card, and tries them in order
# (the registered card model, RTX PRO 4000 Blackwell, first; then the smallest adequate card). task.sh re-probes and
# claims inside the allocation; this watcher never picks a card itself. While an attempt runs it checks every 60 s.
# Never allocates or releases a Slurm job. Exit 0 when the result is published and valid, 1 after MAX_FAIL real
# failures, 2 at the end of LIFE, 3 if another watcher is alive.
set -uo pipefail

ROOT=$WORK
BASE=$ROOT/_gpu_run/argmax256
CACHE=$FLEET_TOOLS/.gpu_state.tsv
CLAIMS=$CLAIMS_DIR
FINAL=$ROOT/diagnostics/no-rep-penalty/argmax_check_256.json
PY=$ROOT/probe-venv/bin/python
LOG=$BASE/watch.log; Q=$BASE/queue.json; HB=$BASE/watch.heartbeat; ALERTS=$BASE/ALERTS.txt; LOCK=$BASE/watcher.lock
POLL=${POLL:-1800}; LIFE=${LIFE:-252000}; MAX_FAIL=${MAX_FAIL:-2}; MIN_VRAM=${MIN_VRAM:-16}; MIN_LEFT=${MIN_LEFT:-3600}
mkdir -p "$BASE/logs"

STRIP=(); while IFS= read -r v; do STRIP+=(-u "$v"); done < <(env | sed -n 's/^\(SLURM[^=]*\|SRUN[^=]*\)=.*/\1/p')
iso() { date +%Y-%m-%dT%H:%M:%S%:z; }
say() { echo "[$(iso)] $*" | tee -a "$LOG"; }
alert() { echo "[$(iso)] $*" >> "$ALERTS"; say "ALERT $*"; }
beat() { echo "$(iso) $(hostname -s) $$ ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} $STATE" > "$HB.tmp.$$" && mv -f "$HB.tmp.$$" "$HB"; }
qset() {   # state [last_attempt]
  STATE=$1
  "$PY" -I - "$Q" "$STATE" "$(iso)" "${2:-}" "$ATTEMPTS" "$FAILURES" "$(hostname -s) pid $$ step ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}" "${EXCL[*]:-}" <<'PY'
import json, os, sys
p, state, now, last, att, fail, who, excl = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else {}
d.update(state=state, updated=now, attempts=int(att), failures=int(fail), watcher=who, excluded_nodes=excl.split())
if last: d["last_attempt"] = last
d.setdefault("queued_at", now)
t = p + ".tmp"; json.dump(d, open(t, "w"), indent=1); os.replace(t, p)
PY
  beat
}
final_ok() {
  [ -s "$FINAL" ] && "$PY" -I -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['sets']
assert set(s) == {'registered256', 'norp256'} and all(v['records'] == 200 for v in s.values())" "$FINAL" 2>/dev/null
}
tl_secs() { awk -v t="$1" 'BEGIN { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
  n = split(t, b, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + b[i]; print d * 86400 + s }'; }
candidates() {   # cache: cluster node ngpu timeleft gpuname busy utils procs vram peak allocmemMB hostfreeG jobid
  local cl node ngpu tl name busy utils procs vram rest jid
  while IFS=$'\t' read -r cl node ngpu tl name busy utils procs vram peak amem hfree jid; do
    [ -n "${jid:-}" ] && [ "${busy:-0}" -lt "${ngpu:-0}" ] || continue
    [[ $name == *GH200* ]] && continue
    [ "${vram%%.*}" -ge "$MIN_VRAM" ] 2>/dev/null || continue
    [ "$(tl_secs "$tl")" -ge "$MIN_LEFT" ] || continue
    [[ " ${EXCL[*]:-} " == *" $node "* ]] && continue
    pref=1; [[ $name == *"RTX PRO 4000"* ]] && pref=0
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pref" "$vram" "$((ngpu - busy))" "$cl" "$jid" "$node" "$name" "$tl"
  done < "$CACHE" | sort -t$'\t' -k1,1n -k2,2n -k3,3nr
}

# ---- one watcher only --------------------------------------------------------------------------------------------
STATE=QUEUED; ATTEMPTS=0; FAILURES=0; EXCL=()
if ! mkdir "$LOCK" 2>/dev/null; then
  age=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
  if [ "$age" -lt 600 ]; then echo "another watcher is alive (heartbeat ${age}s old: $(cat "$HB"))"; exit 3; fi
  say "taking over a stale lock (heartbeat ${age}s old: $(cat "$HB" 2>/dev/null))"
fi
echo "$(hostname -s) $$ $(iso)" > "$LOCK/owner"
trap 'say "watcher exiting (state $STATE)"; rm -rf "$LOCK"' EXIT
trap 'say "watcher signalled"; exit 130' INT TERM
if [ -f "$Q" ]; then
  ATTEMPTS=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('attempts', 0))" "$Q")
  FAILURES=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('failures', 0))" "$Q")
  read -r -a EXCL <<< "$("$PY" -I -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('excluded_nodes', [])))" "$Q")"
fi
qset QUEUED
QUEUED_AT=$(date -d "$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['queued_at'])" "$Q")" +%s)
say "watcher up on $(hostname -s) pid $$ step ${SLURM_JOB_ID:-?}.${SLURM_STEP_ID:-?}; queued since $(date -d @"$QUEUED_AT" +%FT%T%:z); POLL ${POLL}s LIFE ${LIFE}s; attempts $ATTEMPTS failures $FAILURES"
start=$(date +%s); warned24=0

attempt() {   # cl jid node name -> 0 done | 10 try next candidate | 11 retry next tick | 1 terminal failure
  local cl=$1 jid=$2 node=$3 name=$4 mark rc dir srl m
  mark=$BASE/logs/.mark.$$; touch "$mark"; srl=$BASE/logs/srun-$(date +%Y%m%d-%H%M%S)-$node.log
  m=""; [ "$cl" != cluster-a ] && m="-M $cl"
  ATTEMPTS=$((ATTEMPTS + 1)); qset STARTING
  say "attempt $ATTEMPTS: $cl job $jid $node ($name) -> $srl"
  env "${STRIP[@]}" srun $m --jobid="$jid" --overlap -n1 bash -lc "bash $BASE/task.sh" < /dev/null > "$srl" 2>&1 &
  local spid=$! last=""
  while kill -0 "$spid" 2>/dev/null; do
    sleep 60; beat
    dir=$(find "$BASE" -maxdepth 1 -name "attempt-*-$node" -newer "$mark" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    if [ -n "$dir" ] && [ -f "$dir/state.json" ]; then
      s=$(sed -n 's/.*"state": "\([^"]*\)".*"device": "\([^"]*\)".*"note": "\([^"]*\)".*/\1 \2 \3/p' "$dir/state.json")
      [ "$s" != "$last" ] && { say "  $node: $s"; last=$s; }
      case "$s" in RUNNING*|VALIDATING*) [ "$STATE" = RUNNING ] || qset RUNNING "$dir" ;; esac
    fi
  done
  wait "$spid"; rc=$?
  dir=$(find "$BASE" -maxdepth 1 -name "attempt-*-$node" -newer "$mark" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
  rm -f "$mark"
  say "  $node: srun rc=$rc attempt dir ${dir:-none}; $(tail -1 "${dir:-/dev/null}/run.log" 2>/dev/null)"
  if [ -z "$dir" ]; then ATTEMPTS=$((ATTEMPTS - 1)); say "  launch did not reach task.sh: $(tail -2 "$srl" | tr '\n' ' ')"; qset QUEUED; return 10; fi
  case "$rc" in
    0)  final_ok && { qset SUCCEEDED "$dir"; return 0; }
        FAILURES=$((FAILURES + 1)); alert "attempt $dir exited 0 but $FINAL does not validate" ;;
    75) ATTEMPTS=$((ATTEMPTS - 1)); qset QUEUED "$dir"; return 10 ;;
    76) EXCL+=("$node"); ATTEMPTS=$((ATTEMPTS - 1)); say "  excluding $node: $(tail -1 "$dir/run.log")"; qset QUEUED "$dir"; return 10 ;;
    *)  FAILURES=$((FAILURES + 1)); alert "attempt $dir FAILED rc=$rc on $node: $(tail -1 "$dir/run.log" 2>/dev/null) (logs $dir/logs, srun $srl)" ;;
  esac
  for f in "$CLAIMS"/rrhs_*; do [ -f "$f" ] && grep -qF "\"run\": \"$dir\"" "$f" && rm -f "$f" && say "  released leftover $(basename "$f")"; done
  if [ "$FAILURES" -ge "$MAX_FAIL" ]; then qset FAILED "$dir"; alert "TERMINAL FAILED after $FAILURES failures; needs diagnosis before any relaunch"; return 1; fi
  qset QUEUED "$dir"; return 11
}

while [ $(( $(date +%s) - start )) -lt "$LIFE" ]; do
  if final_ok; then qset SUCCEEDED; say "TERMINAL SUCCEEDED: $FINAL valid"; exit 0; fi
  cache_age=$(( $(date +%s) - $(stat -c %Y "$CACHE" 2>/dev/null || echo 0) ))
  mapfile -t C < <(candidates)
  say "check: ${#C[@]} allocation(s) with an idle card (cache ${cache_age}s old)${EXCL[*]:+; excluded ${EXCL[*]}}"
  for row in "${C[@]}"; do
    IFS=$'\t' read -r _ vram free cl jid node name tl <<< "$row"
    say "  candidate $node $cl/$jid $name ${vram}G, $free idle, $tl left"
    attempt "$cl" "$jid" "$node" "$name"; r=$?
    case $r in 0) say "TERMINAL SUCCEEDED: $FINAL valid"; exit 0 ;; 1) exit 1 ;; 11) break ;; esac
  done
  waited=$(( $(date +%s) - QUEUED_AT ))
  if [ "$warned24" = 0 ] && [ "$waited" -ge 86400 ]; then
    warned24=1; alert "WAITED_24H: queued since $(date -d @"$QUEUED_AT" +%FT%T%:z) with no usable card; the gpu-run rule now allows asking the user about a new allocation (this watcher never submits one)"
  fi
  qset QUEUED
  for ((k = 0; k < POLL / 60; k++)); do sleep 60; beat; final_ok && break; done
done
qset QUEUED; alert "watcher reached its LIFE limit (${LIFE}s) with the task still queued; restart queue_watch.sh to continue (queued_at is kept)"
exit 2
