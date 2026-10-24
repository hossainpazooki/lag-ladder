# gpu_run manifest: farhan-run Step 1 baseline probe

- queued_at: 2026-10-21T21:35-06:00 (session rrhs, Claude Code)
- package: $WORK/2026-10-21-farhan-run (unzipped from 2026-10-21-farhan-run__1_.zip)
- task: README Step 1. setup_probe.sh (LL_COMMIT=e32d523), then baseline_probe.py at --max-new 256 and 768,
  --n 200 (the runbook allows it), float32 (the run's precision), greedy, everything else default.
- Step 2 (setup_run.sh) is NOT run: it needs the registration commit, config/ownrun_instruct.toml and tokens-keep/,
  none of which exist yet (README sec. 2-3).
- placement: cluster-a job JOB-A, node gpu-node-1 (8x RTX PRO 4000 Blackwell 24 GB, driver 595.91.07, 48 CPU, 320 GB).
  Selected: inherited token 0 = GPU-<gpu01> (cap 256),
            inherited token 1 = GPU-<gpu02> (cap 768); CUDA_DEVICE_ORDER=PCI_BUS_ID.
  Probed 21:27: all 8 cards 2 MiB, no compute apps, no service ports; only the holder's sleep infinity.
  No committed claim on gpu-node-1 (only hidden .trc tmp files); another-project fleet stopped by the user 20:39.
- claims: $CLAIMS_DIR/rrhs_cluster-a_JOB-A_gpu-node-1_{<gpu01>,<gpu02>}, heartbeat 30 s.
- workload: Qwen2.5-1.5B-Instruct fp32 (~6 GB), HF generate batch 1; one process per card. Kept as-is (registered
  probe), not moved to vLLM: the numbers must be the script's.
- outputs: <package>/probe-instruct-256/, <package>/probe-instruct-768/ (probe.json, records.jsonl)
- logs/state: this folder (state.json, logs/attempt-*/, watch.log)
