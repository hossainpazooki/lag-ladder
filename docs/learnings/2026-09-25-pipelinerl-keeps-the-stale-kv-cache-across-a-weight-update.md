# PipelineRL keeps the stale KV cache across a weight update, in bf16, with fp32 master weights and no lag cap by default

ts: 2026-09-25T04:27:17Z
commit: 04169c6abbb7fb01e1b0818192e8ef11442b6330
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("Holdover ARR design & experiments"), seed for the instrument bridge (first observed 2026-09-24, re-captured at close)
status: verified
fact: The author implementation of in-flight weight updates does what the paper's STALE arm models: on a weight update the vLLM worker pauses with `mode="keep", clear_cache=False`, receives the new tensors, and resumes on the cache the old weights wrote. The inference engine runs bf16 (`vllm_config.dtype: bfloat16`), the trainer keeps fp32 master weights (`param_dtype: fp32`), and `max_lag: null` is the default. So the engine's cache is bf16 and decode-written, which is the precision/origin gap the bridge seed's G1 and G2 measure; and "PipelineRL" in design §3's arm table is a literal production analogue, not a paraphrase.
basis: `gh api repos/ServiceNow/PipelineRL/contents/pipelinerl/vllm1.py --jq .content | base64 -d | grep -n clear_cache` → `160:            await self.engine.pause_generation(mode="keep", clear_cache=False)`; `… conf/base.yaml … | grep -n -E "^\s+dtype:|param_dtype|max_lag"` → `61:    dtype: bfloat16`, `99:  param_dtype: fp32`, `106:max_lag: null`; `gh api repos/ServiceNow/PipelineRL/commits/main --jq .sha` → `58d393458625ad63ed539f2dcd072c85700c557f` (captured 2026-09-25T04:27:17Z; the lag-ladder commit named above is this repo's HEAD at capture, the fact is about the producer at that sha).
re-verify: gh api repos/ServiceNow/PipelineRL/contents/pipelinerl/vllm1.py --jq .content | base64 -d | grep -n clear_cache
