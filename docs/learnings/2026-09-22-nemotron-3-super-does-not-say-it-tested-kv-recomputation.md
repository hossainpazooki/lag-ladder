# Nemotron 3 Super does not say it tested KV-cache recomputation; the survey post's "tested … and found no benefit" is unsupported for it

ts: 2026-09-22T23:09:33Z
commit: e79d3ad4789467cec39ffb5ca733a8b07a0c0173
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("async-rl-holdover-design"), prior-art re-read
status: refuted-assumption
fact: Huang's post "Is Frontier Asynchronous RL Solved?" (2026-05-31) says "Magistral and Nemotron 3 Super both tested recomputing KV caches after weight syncs and found no benefit." The Nemotron 3 Super report (arXiv 2604.12374) contains exactly two sentences on the subject — "We do not recompute the KV cache after updating the model weights on the inference workers" (§3.2.4, with workers held "at most one step behind") and "We did not recompute the KV cache after in-flight weight updates" (Async RL Infrastructure) — and no sentence saying recomputation was tested. Magistral (2506.10910) has one unevidenced sentence: "For performance, we find that recomputing the key-value cache is not necessary, potentially due to off-policy corrections inherent to the loss function." The design doc's 09-19 draft did not make this error; the post does, and a reviewer may repeat it.
basis: `grep -n -i "recomput" nemotron3super.txt | grep -i kv` over `pdftotext -layout` of arXiv 2604.12374 → lines 1452 ("We do not recompute the KV cache after") and 1522 ("recompute the KV cache after in-flight weight updates."); `grep -c -i "tested recomput" nemotron3super.txt` → `0` (re-captured 2026-09-22T23:09:33Z at e79d3ad from the text extracted 2026-09-20T20:04-04:00; the PDF was fetched that day).
re-verify: curl -sL https://arxiv.org/pdf/2604.12374 | pdftotext -layout - - | grep -n -i "recompute the KV cache"
