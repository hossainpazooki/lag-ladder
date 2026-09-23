# Learnings index — lag-ladder

One fact per dated entry; entries are immutable, a wrong one is superseded by a new entry with `kills:`.
Each entry carries `ts:`, `commit:`, `session:`, `status:`, `fact:`, `basis:`, `re-verify:` (read-only).

| date | entry | status | one line |
|---|---|---|---|
| 2026-09-22 | [scope-lint-flags-a-casual-restatement-of-what-the-repo-measures](2026-09-22-scope-lint-flags-a-casual-restatement-of-what-the-repo-measures.md) | verified | an ordinary one-line description of the repo overlaps the scope sentence at 0.65 and is flagged; cite the sentence, don't restate it |
| 2026-09-22 | [machine-default-python-is-3-14-and-the-project-pins-below-3-13](2026-09-22-machine-default-python-is-3-14-and-the-project-pins-below-3-13.md) | verified | build the venv from `py -3.12`; bare `python -m venv` gives 3.14 and the install refuses |
| 2026-09-22 | [nemotron-3-super-does-not-say-it-tested-kv-recomputation](2026-09-22-nemotron-3-super-does-not-say-it-tested-kv-recomputation.md) | refuted-assumption | the survey post's "tested and found no benefit" has no support in the Nemotron report; Magistral has one unevidenced sentence |
| 2026-09-22 | [pipelinerl-fig-7-has-a-lag-axis-to-32-with-a-recomputed-cache-curve](2026-09-22-pipelinerl-fig-7-has-a-lag-axis-to-32-with-a-recomputed-cache-curve.md) | refuted-assumption | "one operating lag" was wrong: Fig. 7 is KL vs lag 0–32 with a recomputed-cache curve; novelty (ii) narrowed to the tail |
