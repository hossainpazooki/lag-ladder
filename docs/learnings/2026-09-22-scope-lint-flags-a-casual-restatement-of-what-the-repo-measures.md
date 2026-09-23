# The scope lint flags a casual restatement of what the repo measures, not only a deliberate paraphrase

ts: 2026-09-22T23:09:29Z
commit: e79d3ad4789467cec39ffb5ca733a8b07a0c0173
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("async-rl-holdover-design"), scaffold step
status: verified
fact: The README opener written on 2026-09-22 — "Measurement repo for Holdover: what a KV cache written by one RL policy checkpoint costs when a later checkpoint reads it, as a function of how many optimizer updates apart the two are." — overlaps the scope sentence's distinctive stems at 0.65 (threshold 0.45) and is flagged as a paraphrase; ledger entry 0001's first sentence was flagged at 0.48 the same way. Both were reworded, not exempted. Any one-line description of the repo drifts toward the scope sentence's vocabulary, so the lint fires on ordinary prose, which is the intended behaviour.
basis: `.venv/Scripts/python.exe -c "from lag_ladder.lint_scope import check_paraphrase, _overlap_score; s='Measurement repo for Holdover: ...'; print(round(_overlap_score(s),2), len(check_paraphrase(s,'README.md')))"` → `0.65 1` (captured 2026-09-22T23:09:29Z at e79d3ad; the original firing during scaffolding printed "word overlap 0.70" for the same sentence with its markdown heading attached, and "0.48" for ledger 0001's opener).
re-verify: .venv/Scripts/python.exe -c "from lag_ladder.lint_scope import _overlap_score; print(round(_overlap_score('Measurement repo for Holdover: what a KV cache written by one RL policy checkpoint costs when a later checkpoint reads it, as a function of how many optimizer updates apart the two are.'),2))"
