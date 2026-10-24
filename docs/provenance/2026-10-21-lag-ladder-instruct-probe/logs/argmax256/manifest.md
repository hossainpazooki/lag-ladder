# gpu_run manifest: farhan-run cap-256 argmax provenance check (diagnostic, not the registered probe)

- queued_at: 2026-10-23 ~17:20 MDT (session rrhs, Claude Code); the watcher keeps the original time in `queue.json`.
- why: the no-repetition-penalty diagnostic checked decoding provenance (raw vs penalized argmax, teacher-forced)
  only at cap 768 (`diagnostics/no-rep-penalty/argmax_check.json`). At cap 256 the band change (44.5 -> 51.5,
  flips-verdict -> accuracy-verdict) rests on 3 prompts, and its decoding is inferred from DIAG_GENCFG, not measured.
  This runs the same script on the cap-256 record sets. It is the only remaining experiment that can run now:
  Step 2 (smoke, train, (B), (C), package) waits on the authors' registration (commit, config/ownrun_instruct.toml,
  tokens-keep/, run_instruct.sh, chain_instruct.sh, package.sh), which is not upstream as of main 732c8d4.
- command (per attempt, inside an allocation): `_gpu_run/diag/argmax_check.py` (unedited) with
  `registered256=2026-10-21-farhan-run/probe-instruct-256/records.jsonl` and
  `norp256=diagnostics/no-rep-penalty/probe-norp-256/records.jsonl`; Qwen2.5-1.5B-Instruct@989aa79, float32,
  one forward pass per record, penalty tested 1.1. ~6 GB VRAM, 1 card, a few minutes.
- placement: one free card (no compute process, < 500 MiB) with no live claim from anyone, inside an allocation this
  user already holds. Prefer RTX PRO 4000 Blackwell (the probe's and the 768 check's card); any x86 card >= 16 GB
  otherwise (the card name lands in the JSON's `device`). Never allocates; never shares a busy card.
- at queue time: all 51 held GPUs busy (another-project ray/vLLM, 100 %); hence the queue.
- outputs: `diagnostics/no-rep-penalty/argmax_check_256.json` (written only after validation; never overwritten),
  plus per-attempt `attempt-<ts>-<node>/` here (state.json, run.log, logs/, argmax_check_256.json, .sha256).
- claims: `$CLAIMS_DIR/rrhs_<cluster>_<job>_<node>_<uuid8>`, heartbeat 10 s, removed
  on every exit path task.sh controls; the watcher removes leftovers of a failed attempt.
- watcher: `queue_watch.sh`, an srun --overlap step in CPU host job cluster-a JOB-W (gpu-watchers, cpu-host-1); the srun
  client is held by tmux `rrhs:argmax256` on login-1 (holder.log). Checks capacity every 30 min, an attempt every
  60 s; LIFE 70 h; MAX_FAIL 2 (one automatic retry); logs WAITED_24H to ALERTS.txt at 24 h (never submits a job).
  State: `queue.json` (QUEUED/STARTING/RUNNING/SUCCEEDED/FAILED), `watch.log`, `watch.heartbeat`, `ALERTS.txt`.
- restart after a kill: rerun the holder command below; the lock is taken over once the heartbeat is > 10 min old,
  and queued_at, attempts and failures carry over.
  `env -u SLURM_JOB_ID srun -M cluster-a --jobid=JOB-W --overlap -n1 bash $WORK/_gpu_run/argmax256/queue_watch.sh`
- dry run before queuing: `attempt-20261023-171435-gpu-node-1/` (gpu-node-1 all 8 cards busy -> NO_CARD, rc 75, no claim).
