# PipelineRL per-step checkpoints are a config change, not a fork: the "additive fork" in design §6 shrinks to a YAML override

ts: 2026-09-25T04:27:20Z
commit: 04169c6abbb7fb01e1b0818192e8ef11442b6330
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("Holdover ARR design & experiments"), seed for the instrument bridge (first observed 2026-09-24, re-captured at close)
status: verified
fact: `conf/finetune/base.yaml` exposes `save_checkpoint_steps: 100`, `keep_intermediate_checkpoints: True`, `also_save_steps: []` and `use_safetensors: true`. A checkpoint every optimizer step for the own run (design §6, deferred by R8) is `save_checkpoint_steps: 1` (or a sparse list in `also_save_steps`) saved as HF safetensors — no code change. The design doc's "fork whose only changes are additive config" and the handoff brief's "per-step checkpoint saving unverified" are both resolved: no fork is needed unless a step-numbered output layout has to be fixed, and the 4×H100 minimum footprint is the remaining cost.
basis: `gh api repos/ServiceNow/PipelineRL/contents/conf/finetune/base.yaml --jq .content | base64 -d | grep -n -E "save_checkpoint_steps|also_save_steps|use_safetensors|keep_intermediate"` → `69:save_checkpoint_steps: 100`, `71:keep_intermediate_checkpoints: True`, `85:also_save_steps: []`, `86:use_safetensors: true` (captured 2026-09-25T04:27:20Z at producer sha 58d3934).
re-verify: gh api repos/ServiceNow/PipelineRL/contents/conf/finetune/base.yaml --jq .content | base64 -d | grep -n -E "save_checkpoint_steps|also_save_steps"
