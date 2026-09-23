# PipelineRL §5.1 / Fig. 7 measures stale-vs-recomputed KV across a lag axis 0–32, not at one operating lag

ts: 2026-09-22T23:09:35Z
commit: e79d3ad4789467cec39ffb5ca733a8b07a0c0173
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("async-rl-holdover-design"), prior-art re-read
status: refuted-assumption
fact: The 09-19 design draft (and the 2026-09-14 learning it leaned on) described PipelineRL's stale-KV finding as "one operating lag". The paper's §5.1 sets g_max = 32 and L = 2048, swaps the behavior policy to the next checkpoint every L/g_max tokens from starting checkpoints 0 / 100 / 190 of a Qwen 2.5 7B run (222 optimizer steps), and Figure 7 plots KL(μ ‖ π_{C+g}) against lag g for three curves — Conventional RL, PipelineRL (stale cache), and "PipelineRL with KV cache recomputed". So "a lag ladder" within 0–32 is prior art; what is not is the tail beyond 32, a second model family, a paired same-token contrast, a task-level read-out, and a null control. Design §2 was rewritten (rev 2) on this basis and novelty item (ii) narrowed to the range.
basis: `grep -n -E "KV cache recomputed|maximum lag gmax is set to 32|stale KV-cache for mixed" pipelinerl.txt` over `pdftotext -layout` of arXiv 2509.19128 → lines 560, 566, 569; page 10 rendered with PyMuPDF and read: three curves per panel, x-axis "Lag (g)" 0–32 (re-captured 2026-09-22T23:09:35Z at e79d3ad from the text extracted 2026-09-20T20:04-04:00).
re-verify: curl -sL https://arxiv.org/pdf/2509.19128 | pdftotext -layout - - | grep -n -E "KV cache recomputed|gmax is set to 32"
