# lag-ladder Instruct-init run: Step 1 baseline probe, decoding diagnostic, argmax provenance checks

2026-10-24 (UTC) · Co-author delivery by the runner (GitHub `farhanishmam`) for the `hossainpazooki/lag-ladder` authors.

**Not evidence under this repository.** This folder is the runner's return of Step 1 of the lag-ladder runbook
package `2026-10-21-farhan-run` (Instruct-initialized own run, lag-ladder commit `e32d523`). It is in `linear-ceiling`
only because the runner can push here and cannot push to `hossainpazooki/lag-ladder`. It touches no ledger entry,
registered config, seal or `results/` path of this repository. No ledger entry, paper sentence, README figure or review
may cite a number from it. It is not a submission under the PROPOSED co-author admission procedure
([`docs/2026-10-03-co-author-run-admission.md`](../../2026-10-03-co-author-run-admission.md)): it has no registration
in this repository, no `report.json` and no Hub backup (§1, §2.1, §2.2, §2.4). Every figure below is the runner's
claim, keyed to the file and field it comes from. [`verify_bundle.py`](verify_bundle.py) recomputes the Results
tables from the files in this folder.

**Status: placement proposed, not ruled.** The runner does not submit this run to this repository's record. The folder
is on branch `exp/lag-ladder-instruct-baseline-probe` for the lag-ladder authors and the operator. Whether it is kept,
where it goes, and whether it may be merged into `main` are the operator's rulings.

**Terminology.** In this folder, "registered" (the registered (C) loop, answer rule or cap), "band" (`flips-verdict`,
`accuracy-verdict`), "summary line" (what the runbook calls a "verdict line"), "seal" and "ledger entry" are all
lag-ladder's terms, from its ledger, its runbook or `baseline_probe.py`. "Step 1 probe" (or "probe") means the
runbook's Step 1 probe run unedited, which lag-ladder has not registered. The argmax files label it
`registered256`/`registered768`, and `verify_bundle.py` labels it `reg`. None of these terms refers to this
repository's ledger, bands or verdict cells.

## Contents

1. [Summary](#summary)
2. [What ran](#what-ran)
3. [Results](#results)
4. [How the figures were checked](#how-the-figures-were-checked)
5. [Corrections to the runner's original note](#corrections-to-the-runners-original-note)
6. [Environment and chronology](#environment-and-chronology)
7. [Limitations](#limitations)
8. [Not done, and what each item waits on](#not-done-and-what-each-item-waits-on)
9. [Redactions](#redactions)
10. [Layout](#layout)
11. [Verify or reproduce](#verify-or-reproduce)

## Summary

- **Step 1 ran as the runbook specifies, taking its `--n 200` option at both caps.** Qwen2.5-1.5B-Instruct (revision
  `989aa79`), float32, greedy, on the pinned prompts (sha256 `918c7e67…`). Cap 256 scored **44.5 [37.5, 51.5]**
  (89/200), band *flips-verdict*. Cap 768 scored **64.5 [58.0, 71.0]** (129/200), band *accuracy-verdict*.
- **Cap 256 cuts the Instruct model off.** 46 % of the probe's cap-256 completions (45.5 % of the diagnostic's)
  reach the cap. Every completion that ended earlier is identical at cap 768, so the whole 256→768 gain (+40 correct
  for the probe, +41 for the diagnostic) comes from the prompts cut at 256.
- **The probe's decoding is not lag-ladder's registered (C) loop's.** `baseline_probe.py` calls
  `model.generate(do_sample=False)`, which inherits the model revision's `generation_config.json`:
  `repetition_penalty` 1.1 and `eos_token_id` [151645, 151643]. The registered (C) loop (`tools/task_swap.py`
  `_greedy`) is plain argmax that stops on 151645 only. Teacher-forced checks confirm that the probe records were
  decoded under the penalty, at both caps.
- **Under (C)'s decoding the cap-256 band changes.** A diagnostic rerun with plain argmax scored **51.5 [44.5, 58.0]**
  at cap 256, which is *accuracy-verdict*, and **72.0 [65.5, 78.0]** at cap 768. The cap-256 change is narrow:
  103/200 is 3 correct answers above the inclusive 100/200 threshold, so 4 fewer would make it *flips-verdict*. The
  paired gain is marginal: exact McNemar p = 0.0595 at 256 and 0.0444 at 768.
- **Which decoding counts, which cap, and which answer marker are the authors' rulings.** This folder only reports.
- **Step 2 (smoke, train, (B), (C), package) did not run.** The registration it needs has not been delivered. Lag-ladder's
  `sample-rollouts` appears, from reading the code (not tested), to inherit the same decoding settings. See
  [Not done](#not-done-and-what-each-item-waits-on).

## What ran

| # | Experiment | What it is | Status | Outputs |
|---|---|---|---|---|
| 1 | Step 1 setup | `setup_probe.sh` with `LL_COMMIT=e32d523`: one venv, the selftest, the prompts hash | done | `logs/step1-probe/setup.log`, `run.log` |
| 2 | Step 1 probe, cap 256 | `baseline_probe.py --n 200 --max-new 256`, unedited (256 is lag-ladder's registered (C) cap for the base run; no Instruct cap is registered) | done | `step1-probe/probe-instruct-256/` |
| 3 | Step 1 probe, cap 768 | the same at `--max-new 768` (does the cap or the model limit accuracy? see [Cap or model](#cap-or-model)) | done | `step1-probe/probe-instruct-768/` |
| 4 | Diagnostic, cap 256 | the package's `baseline_probe.main()`, unedited, with the loaded generation config set to (C)'s decoding: `repetition_penalty` 1.0, `eos_token_id` 151645 (`code/diagnostic/norp_probe.py`); 4 contiguous shards of 50 prompts, merged with the package's `summarize()` | done | `diagnostic-no-rep-penalty/probe-norp-256/` |
| 5 | Diagnostic, cap 768 | the same at cap 768 | done | `diagnostic-no-rep-penalty/probe-norp-768/` |
| 6 | Argmax provenance, cap 768 | each completion teacher-forced through the pinned model in one forward pass: is each recorded token the raw argmax or the 1.1-penalized argmax? (`code/diagnostic/argmax_check.py`) | done | `diagnostic-no-rep-penalty/argmax_check.json` |
| 7 | Argmax provenance, cap 256 | the same script, unedited, on the cap-256 record sets | done | `diagnostic-no-rep-penalty/argmax_check_256.json` |
| 8 | Step 2 | setup, smoke, 32-step train, (B) reader run and summary, (C) equivalence and cells, package | **not run** | none |

Rows 1–3 are the runbook's Step 1. Rows 4–7 are the runner's diagnostics: they are not in the runbook and are not
registered anywhere.

## Results

### Accuracy and descriptives

Every cell is a field of the set's `probe.json`. "wrote `####`" is `1 - no_hash_marker_rate`. Accuracy uses
lag-ladder's registered answer rule (last `####`, else last number). The `\boxed{}` column uses the package's
descriptive rule, not the registered one. `no_answer_rate` and `first_vs_last_hash_disagree_rate` are 0.0 in all four
sets.

| set | decoding | cap | correct / n | accuracy, 95% CI (points) | band | reached cap | wrote `####` | wrote `\boxed{}` | accuracy by last `\boxed{}` | mean new tokens | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `probe-instruct-256` | penalty 1.1 (inherited) | 256 | 89 / 200 | 44.5 [37.5, 51.5] | flips-verdict | 0.460 | 0.090 | 0.230 | 18.0 | 215.9 | 587.1 |
| `probe-instruct-768` | penalty 1.1 (inherited) | 768 | 129 / 200 | 64.5 [58.0, 71.0] | accuracy-verdict | 0.000 | 0.105 | 0.560 | 41.0 | 261.9 | 723.0 |
| `probe-norp-256` | plain argmax | 256 | 103 / 200 | 51.5 [44.5, 58.0] | accuracy-verdict | 0.455 | 0.025 | 0.250 | 21.0 | 209.7 | 153.4 |
| `probe-norp-768` | plain argmax | 768 | 144 / 200 | 72.0 [65.5, 78.0] | accuracy-verdict | 0.000 | 0.030 | 0.590 | 41.0 | 253.6 | 204.2 |

- The CI is the package's percentile bootstrap: 2,000 resamples, numpy `default_rng(0)`. Two stored bounds are
  `57.99999999999999` in `probe.json`: the lower bound of `probe-instruct-768` and the upper bound of `probe-norp-256`.
  The table, the `PROBE_DONE` line (cap 768) and the `MERGE_OK` line (cap 256) show both as 58.0.
- For the diagnostic, `elapsed_s` is the slowest of the four shards, which ran in parallel on one card each
  (`probe.json` → `diagnostic`).
- The summary lines (the runbook calls them verdict lines) in `logs/step1-probe/run.log` print `PROBE_DONE n 200
  max_new 256 accuracy 44.5 [37.5, 51.5] cap 0.46 no_answer 0.00 hash_marker 0.09 boxed_marker 0.23 acc_boxed 18.0
  loop 0.00 band flips-verdict` and `PROBE_DONE n 200 max_new 768 accuracy 64.5 [58.0, 71.0] cap 0.00 no_answer 0.00
  hash_marker 0.10 boxed_marker 0.56 acc_boxed 41.0 loop 0.00 band accuracy-verdict`. The diagnostic's merge lines,
  `MERGE_OK cap 256 …` and `MERGE_OK cap 768 …`, are in `logs/diagnostic/run.log`.
- The runbook's `--n 100` subset is the first 100 prompts. Correct counts on it: probe 45/100 at cap 256 and 65/100 at
  cap 768; diagnostic 50/100 and 70/100.
- The authors' decision rule, from their runbook, sets the bands: ≥ 50 % accuracy-verdict, 20–50 % flips-verdict,
  < 20 % stop. Under it, the probe as shipped puts cap 256 in the 20–50 % row and cap 768 in the ≥ 50 % row. Under
  (C)'s decoding, both caps are in the ≥ 50 % row. This report applies no ruling.
- **The Instruct model mostly ignores the `####` instruction.** Under the shipped decoding, 9 % (cap 256) and 10.5 %
  (cap 768) of completions wrote `####`; under plain argmax, 2.5 % and 3 %. The model reasons in LaTeX and often
  answers in `\boxed{}`. The registered rule then falls back to the last number in the text. This is the "marker"
  question the runbook leaves to the registration.

### Cap or model

This answers row 3's question, from `probe.json` → `cap_rate`, `correct_and_capped` and
`accuracy_uncapped_only_points`, and from the records paired by `index`:

| decoding | cap 256: reached the cap (correct) | cap 256: ended earlier (correct) | those cut at 256, correct at cap 768 | cap-256 early-ended completions identical at cap 768 |
|---|---|---|---|---|
| shipped (penalty 1.1) | 92 (15) | 108 (74, i.e. 68.5 %) | 15 → 55 | 108 / 108 |
| plain argmax | 91 (15) | 109 (88, i.e. 80.7 %) | 15 → 56 | 109 / 109 |

So the whole 256→768 gain (89 → 129 for the probe, 103 → 144 for the diagnostic) comes from the prompts cut at 256.
At cap 768 no completion reached the cap in either decoding. The "ended earlier" accuracy is accuracy on a
self-selected subset, not an estimate of accuracy without a cap.

### Shipped decoding vs (C)'s decoding, paired by prompt

Computed from the two sets' `records.jsonl`, matched on `index`. `verify_bundle.py` checks each value.

| cap | right→wrong | wrong→right | net | exact McNemar p (two-sided) | paired bootstrap of the difference, seed 0, 2,000 resamples | extracted answer differs (numerically) | completion text differs |
|---|---|---|---|---|---|---|---|
| 256 | 17 | 31 | +14 (+7.0 points) | 0.0595 | [0.0, 13.5] | 119 / 200 | 200 / 200 |
| 768 | 17 | 32 | +15 (+7.5 points) | 0.0444 | [1.0, 14.0] | 80 / 200 | 200 / 200 |

- The direction is the same at both caps, but the gain is marginal on 200 prompts. The two caps are not independent
  tests. Within each decoding, all 200 cap-768 completions begin with the cap-256 text, and 109 (probe) and 110
  (diagnostic) are identical to it.
- The cap-256 band change is narrow: 103/200 is 3 above the inclusive 100/200 threshold, so 4 fewer correct answers
  would make it flips-verdict. The first 100 prompts score exactly 50/100, on the threshold.
- The stop-set half of the override changed nothing measurable. Every probe completion that ended before the cap
  and can be rebuilt as tokens (see the next table) stopped on 151645, never on 151643. The shift comes from the
  penalty.
- A qualitative reading of the texts, from the runner's diagnostic note: with the penalty, the model copies the
  question less, and its answers fail more often on set-up, omitted quantities and arithmetic. This was not
  quantified.

### Which decoding produced which records

From `argmax_check.json` (cap 768) and `argmax_check_256.json` (cap 256). Token ids are rebuilt by re-encoding
`completion_text`, and are used only when they hash to the record's `completion_sha256`. A full-sequence forward pass
rounds differently from token-by-token decoding, so an occasional near-tie disagreement is expected under either rule.

| record set | card | records rebuilt | tokens | equal to raw argmax | equal to 1.1-penalized argmax | records matching at every token |
|---|---|---|---|---|---|---|
| probe, cap 768 | RTX PRO 4000 Blackwell | 198 / 200 | 51,783 | 48,997 (94.62 %) | 51,783 (100.00 %) | 198 penalized, 0 raw |
| diagnostic, cap 768 | RTX PRO 4000 Blackwell | 200 / 200 | 50,712 | 50,712 (100.00 %) | 48,032 (94.72 %) | 200 raw, 0 penalized |
| probe, cap 256 | H200 NVL | 199 / 200 | 42,930 | 40,412 (94.13 %) | 42,929 (99.998 %) | 198 penalized, 0 raw |
| diagnostic, cap 256 | H200 NVL | 200 / 200 | 41,943 | 41,943 (100.00 %) | 39,519 (94.22 %) | 200 raw, 0 penalized |

The probe records were decoded under the penalty at both caps, and the diagnostic records under plain argmax. This is
measured on the records, not only inferred from the generation config. The one cap-256 probe token that is not the
penalized argmax is in a single record. Records that cannot be rebuilt: probe cap 256, index 3; probe cap 768, indices
3 and 170. Their re-encoded text has the right length but a different token sequence (a non-canonical tokenization).
The answers come from the text, so no accuracy figure depends on this.

## How the figures were checked

- **`verify_bundle.py`** checks:
  - every file against `SHA256SUMS`, that `SHA256SUMS` lists every file in the folder, and the two original checksum
    listings shipped with the runs
  - that each set has 200 records with indices 0..199, that `records_sha256` in `probe.json` matches, and the float32,
    model and full prompt-hash pins
  - every `probe.json` summary field, recomputed from the records
  - every value in the Cap-or-model, paired and argmax tables above

  It exits 1 on any mismatch. Two flags extend it:
  - `--lag-ladder <checkout at e32d523>` re-derives every record's `answer`, first-`####` answer and `correct` from
    `completion_text` with lag-ladder's registered functions.
  - `--package <farhan-run folder>` re-runs the package's own `summarize()` and compares it to `probe.json` field by
    field.

  On the runner's side, all three modes print `VERIFY_BUNDLE OK`: 0 records differ, and every field is equal.
- **Not checked by `verify_bundle.py`:** which records cannot be rebuilt as tokens and the stop-token statement. Both
  need the tokenizer: `argmax_check.py` counts the records it rebuilds but does not name the others, so the indices
  come from the same re-encoding and hash test, run separately. Also not checked: `elapsed_s`, the environment
  strings, the chronology and the package hashes, which are keyed to their files.
- **Shards:** the diagnostic's four shard record files per cap concatenate byte for byte to the merged
  `records.jsonl`. This was checked when the folder was assembled; the shard folders are not included.
- **Earlier self-check:** on 2026-10-23, before this folder was assembled, the runner's own Claude Code session
  re-checked rows 1–6 with separate agent passes (four checkers, each with a skeptic, plus a critic). It recomputed
  every number, verified every hash, and checked that each Slurm step exited `0:0` and that no run left a process or
  a GPU claim behind. This is a self-check by the runner, not an independent review, a `docs/reviews/` record or the
  operator's recompute, and its working files are not included. From this folder, the Slurm exits and the claim
  sweep can be read only from the `rc=0`, `released` and `SRUN_EXIT 0` lines in `logs/`. Row 7 ran after that
  check; its exit and claim release are in `logs/argmax256/attempt/run.log`.
- **Upstream drift:** lag-ladder `main` was at `54c4d48` on 2026-10-24, 27 commits past `e32d523`. None of them
  changes `final_number`, `first_hash_number`, `is_correct` or the value of `SYSTEM_LINE`, so the scoring rule these
  records use is still upstream's.

## Corrections to the runner's original note

`diagnostic-no-rep-penalty/README.md` is the runner's note from 2026-10-22, with its cap-256 paragraph added on
2026-10-24 (UTC) after row 7 ran. It is kept as written apart from redaction and a not-evidence note under its title,
which maps its relative paths to this folder.

- "48 of 200 answers change at cap 256 … and 49 at 768" counts correctness flips (17 + 31 and 17 + 32). The extracted
  answers themselves differ on 119 and 80 prompts.
- "No completion in either run contains 151643" holds for every completion whose tokens can be rebuilt. Probe cap
  256, index 3, reached the cap, so it has no stop token. Probe cap 768, indices 3 and 170, both ended before the
  cap, and their stop token cannot be determined (Limitations, item 6).
- Its "independent recheck, 2026-10-22" was a check by the runner's own session, and it left no record in this folder.

## Environment and chronology

| item | value |
|---|---|
| lag-ladder | `e32d523449458c5ab9ca99e55bb98272c61d5cc9` (`LAG_LADDER_HEAD`; `SELFTEST_OK summarize/band/bootstrap/boxed`) |
| model | `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, float32, batch 1, HF `generate` |
| software | torch 2.14.1+cu130 (CUDA 13.0), transformers 5.20.2, Python 3.12.15, Linux x86_64 (`probe.json` → `env`); numpy 2.5.4, tokenizers 0.23.3, safetensors 0.8.0, huggingface-hub 2.2.0 (the venv's install list in `logs/step1-probe/setup.log`; the same venv ran rows 1–7) |
| card, rows 1–6 | NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB, driver 595.91.07; one 8-card node: one card per cap (Step 1), all 8 cards (diagnostic) |
| card, row 7 | NVIDIA H200 NVL, 143,771 MiB, driver 595.91.07; same venv and weights |
| Step 1 | 2026-10-22 03:30:37Z setup → 03:32:13Z both caps launched → cap 256 exit 0 at 03:42:14Z, cap 768 exit 0 at 03:44:44Z |
| diagnostic | 2026-10-22 05:59:16Z 8 shards launched → all exit 0 by 06:03:17Z → merged → cap-768 argmax check 06:03:18Z–06:04:16Z |
| cap-256 argmax check | dry run 2026-10-23 23:14:35Z on the Step 1 node: all 8 cards busy (no free card, rc 75); queued 23:15:46Z; half-hourly checks found no idle card until 2026-10-24 01:45:48Z; an RTX 3090 node then had no free unclaimed card (rc 75); ran on an H200 NVL 01:47:21Z–01:48:11Z, exit 0 |

Logs print local time, UTC−6.

## Limitations

1. **200 of 1,319 prompts.** Every figure covers the first 200 prompts of the pinned file. The runbook's `--n 100`
   figure is the first-100 subset above, not a separate run.
2. **One greedy decode per set.** There is no seed variation to report. A different CUDA build or card can move a
   near-tie token.
3. **The diagnostic is not a (C) run.** It follows (C)'s decoding rule, but it decodes from an unrounded float32 cache,
   which is (C)'s fp32 floor arm, not its float16-rounded FRESH arm. It also uses step-0 weights, not step-g weights,
   and 200 prompts rather than 1,319. No Instruct cap is registered in lag-ladder yet.
4. **The diagnostic ran in 4 shards per cap.** Each prompt is decoded alone (batch 1), so shard boundaries do not
   change outputs; the shards ran on eight cards of the same model. Its `elapsed_s` is the slowest shard, not a
   sequential wall time.
5. **The cap-256 argmax check ran on a different card model** (H200 NVL; rows 1–6 used the RTX PRO 4000 Blackwell).
   - No RTX PRO 4000 was free when the check was queued (a dry run on the Step 1 node found all 8 busy) or when it ran.
   - The watcher first tried an RTX 3090 node (no free unclaimed card, rc 75), then ran on the first idle H200 NVL
     card. `queue.json` counts only attempts that got a card.
   - The check is a teacher-forced forward pass, so a card change can only move near-ties. The single non-matching
     token in 42,930 fits either a card change or the forward-pass rounding.
6. **Rebuilding tokens is incomplete.** The records store `completion_text` and a hash, not token ids. 3 of the 800
   records cannot be rebuilt (see above), so the argmax fractions cover 797 of 800 completions. The stop-token
   statement holds for all probe completions except cap-768 indices 3 and 170. Both ended before the cap, and their
   stop token (151645 or 151643) cannot be determined from the text.
7. **The paired statistics are marginal.** McNemar p is 0.0595 at 256 and 0.0444 at 768, the two caps are correlated,
   and the cap-256 band difference is 3 correct answers above an inclusive threshold (4 fewer would reverse it). Do
   not read the diagnostic as establishing a band, only as showing that the band depends on the decoding.
8. **No Hub backup and no `report.json`.** The evidence bundle of the PROPOSED admission procedure (§2.1, §2.2 and
   §2.4: a `report.json` with `complete: true` and `code_sha256`, pushed to an HF dataset) was not made. Whether that
   procedure applies to a pre-registration probe of another repository is for the operator and the lag-ladder authors
   to decide. The records here are complete and small, so they are committed whole, byte for byte.
9. **Not all of the code that ran is here.**
   - The package's own files (`baseline_probe.py`, `setup_probe.sh`, the prompts file) are listed by sha256 under
     [Package files](#package-files-not-included) and not included. The runbook marks the package a private working
     document, and this repository is public.
   - The decision rule and the Step 2 plan described in this README come from that runbook. Only what is needed to
     read the results is restated, and the authors may ask for it to be removed.
   - The runner's shell drivers are committed redacted. `code/CODE_HASHES.txt` holds an HMAC-SHA256 of the bytes that
     ran; the runner holds the key and the originals and hands them to the authors or the operator on request.
   - The three Python helpers are byte-identical to what ran (plain sha256).
10. **Logs are redacted and trimmed.**
    - Dropped: progress bars, `Loading weights` lines, the Hugging Face Hub "unauthenticated requests" warning, and
      uv's download, resolve and venv-creation lines. The
      installed-package list in `setup.log` is kept.
    - Site identifiers were replaced (see [Redactions](#redactions)).
    - Every summary line (`PROBE_DONE`, `MERGE_OK`, `ARGMAX_CHECK`, `VERSIONS`, `LAG_LADDER_HEAD`, `SETUP_PROBE_DONE`,
      `SRUN_EXIT`) is kept verbatim.
    - Some files are left out; see [Not included](#not-included-and-why).

## Not done, and what each item waits on

The Step 2 pieces below come from the authors' runbook. Only their names are restated.

| item | status | waits on |
|---|---|---|
| return of Step 1 to the authors | done by this branch | the runbook asks for both folders and `sha256sum probe-instruct-*/*`; both are in `step1-probe/` |
| Step 2 setup (`setup_run.sh`) | not run | lag-ladder's registration commit (`LL_COMMIT`), `config/ownrun_instruct.toml` (the runbook's provisional name) and its (C) counterpart, and lag-ladder's seal and ledger entry. None is on lag-ladder `main` as of `54c4d48` (2026-10-24); its newest configs are `ownrun_agentic.toml` and `ownrun_llama.toml`. Also the `tokens-keep/` files (tokenized-prompt arrays, not credentials), which come with the package and have not been received. |
| smoke (2 GRPO steps), step-0 digest pin | not run | `run_instruct.sh`, not yet delivered; then the authors pin the digest the smoke prints |
| train (32 steps), (B) reader run and summary | not run | the above, plus a card larger than the 24 GB Step 1 card, as the runbook's trainer requires. The runner holds larger cards (A100 80 GB, H200 NVL, RTX PRO 6000), shared with other work, and none is reserved for Step 2. None was idle at the half-hourly checks from 2026-10-23 23:15Z to 2026-10-24 01:15Z; at 01:46Z two H200 NVL cards were idle, and row 7 used one of them for 50 s (`logs/argmax256/watch.log`). |
| (C) equivalence and cells | not run | `chain_instruct.sh` and the (C) config; the authors' ruling on decoding, cap and marker |
| package (`package.sh`, tarballs, `PACKAGE_OK`) | not run | `package.sh`, not yet delivered |
| between-steps asks (runbook §2) | partly answered | Recorded: the Step 1 card line (`logs/step1-probe/card_line.txt`: RTX PRO 4000 Blackwell, 24,467 MiB, driver 595.91.07) and the free space on the shared scratch filesystem (132 TiB by `df -h`, taken just after Step 1 finished). Not answered: which larger card the run would use, and whether the machine can hold the run's long single processes, as the runbook asks. The runner answers these. |
| authors' rulings | pending | which decoding counts for the Instruct registration, which cap, and whether `\boxed{}` becomes the marker |
| cap-256 argmax check on the probe's own card model | not done | it ran on an H200 instead (Limitations, item 5) |

### For the authors: the same inherited settings in Step 2 tooling (code reading, not tested)

Lag-ladder's `tools/own_run.py` `sample-rollouts` (line 249 at `e32d523`, line 287 at `54c4d48`) calls
`model.generate(do_sample=True, temperature=…, top_p=1.0, top_k=0, …)` without `repetition_penalty` or
`eos_token_id`. On this Instruct revision it would therefore sample under `repetition_penalty` 1.1 and stop on
[151645, 151643], as the probe did. The authors may want to set both explicitly before building the Instruct token
files. The trainer's own generation was not examined.

## Redactions

`linear-ceiling` is public, and its contributor rules forbid hostnames, addresses, tokens and balances in docs.

**Byte for byte, unchanged:** every `probe.json` and `records.jsonl`, both `argmax_check*.json`, both original
checksum listings, and `code/diagnostic/{norp_probe.py, merge_diag.py, argmax_check.py}`.

**Redacted:** the shell drivers under `code/`, every file under `logs/`, and `diagnostic-no-rep-penalty/README.md`
(the runner's original note). These labels replace literal values:
- site paths: `$WORK`, `$HF_HOME`, `$SCRATCH`, `$CLAIMS_DIR`, `$FLEET_TOOLS`; the fleet-tools folder
  name: `fleet-tools`. The user name occurred only inside these paths.
- node and cluster names: `gpu-node-1`, `gpu-node-2`, `gpu-node-3`, `cpu-host-1`, `login-1`, `cluster-a`,
  `cluster-b`, `<cluster-b-node-prefix>`
- Slurm job ids: `JOB-A`, `JOB-B`, `JOB-C`, `JOB-W`
- GPU UUIDs: `GPU-<gpu01>` … `GPU-<gpu09>`, one label per physical card
- another workload on the same allocations: `another-project`

**Edits beyond the label substitution:**
- the log lines dropped as described in Limitations, item 10
- a "REDACTED COPY" header after the shebang of each redacted shell script
- the not-evidence note under the original note's title
- in `logs/step1-probe/manifest.md`, a quotation from the runbook replaced by "the runbook allows it"
- in `logs/step1-probe/card_line.txt`, the file-system line (file server, mount point, totals) reduced to its
  free-space figure

`logs/REDACTIONS.json` lists the labels used and these edits.

### Package files (not included)

By sha256 of the bytes the runner received:

| file | sha256 |
|---|---|
| `README.md` (the runbook) | `72e27d5a7029ff3bf7c2e6c0d50988c054021d97c76132ecae03afc0658c69e8` |
| `README.pdf` | `0eb7876cc411856f2c37541dc3bc46d64be5c3ba600adc267f4cbcfc7d20b54f` |
| `baseline_probe.py` | `81192494b31eb03d8c8792586badd80935b2ad448c77b97e3ec89bfacdec5664` |
| `setup_probe.sh` | `b3b88958f9c270d7ee2863a7209a4f36b862afdfba2924610f7a3500a7f63b22` |
| `setup_run.sh` | `44b23e5f7a60015044f7a1c8573d5e29eca36b9fcaa09945c8ba60e9c2d3485f` |
| `gsm8k_test_prompts.json` | `918c7e67f051d41f7b153911f0497bc62c82b76cb04ef84f6f5ee0e66ab22450` |

## Layout

```
README.md                         this report
verify_bundle.py                  recomputes the Results tables; exit 1 on any mismatch
SHA256SUMS                        every file in this folder except itself (verify_bundle.py checks the coverage)
.gitattributes                    keeps the committed bytes on CRLF checkouts
step1-probe/                      Step 1 as the runbook asks: probe-instruct-{256,768}/{probe.json,records.jsonl},
                                  probe-instruct-sha256sums.txt (the runbook's "sha256sum probe-instruct-*/*")
diagnostic-no-rep-penalty/        rows 4-7: probe-norp-{256,768}/, sha256sums.txt, argmax_check.json (cap 768),
                                  argmax_check_256.json (cap 256), README.md (the runner's original note, redacted)
code/step1-probe/                 run.sh (claim, setup, launch one card per cap, validate, release), watch.sh
code/diagnostic/                  norp_probe.py, merge_diag.py, argmax_check.py (verbatim); run_diag.sh, watch.sh
code/argmax256/                   task.sh (one free card: claim, CUDA check, run, validate, publish), queue_watch.sh
code/CODE_HASHES.txt              sha256 (verbatim files) or keyed HMAC-SHA256 (redacted files) of the code that ran
logs/step1-probe/                 run.log, srun.log, setup.log, probe-{256,768}.log, card_line.txt, verify60.txt, ...
logs/diagnostic/                  run.log, cap{256,768}-s0{0..3}.log, verify60.txt, state.json, watch.log, dispatch.log
logs/argmax256/                   manifest.md (written at queue time, when lag-ladder main was 732c8d4), queue.json,
                                  watch.log, attempt/{run.log,state.json,cudacheck.txt,argmax_check.log}
logs/REDACTIONS.json              the labels used, the dropped log-line pattern, the file-specific edits
```

### Not included, and why

- The diagnostic shard folders (`cap{256,768}-s0{0..3}/probe.json, records.jsonl`): their records concatenate byte for
  byte to `probe-norp-{256,768}/records.jsonl` (sha256 `8045615d…`, `00a65bb6…`). Each shard's `PROBE_DONE` line is in
  `logs/diagnostic/cap*-s0*.log`.
- The GPU snapshots taken before launch and after claiming (`preprobe*.csv`, `postclaim*.csv`): they list other
  workloads' processes and card UUIDs.
- The dry-run and RTX 3090 no-card attempt folders of the cap-256 check (summarised in `logs/argmax256/watch.log`
  and `manifest.md`).
- `holder.log` (identical to `logs/argmax256/watch.log`), the `srun-*.log` files, and the `*.err` files (weight-loading
  progress bars only, or empty).
- Dispatcher scaffolding and heartbeat files (site tooling).

## Verify or reproduce

Verify, from this folder with Python 3.12 and numpy (no GPU, no model):

```bash
python verify_bundle.py
python verify_bundle.py --lag-ladder <lag-ladder checkout at e32d523>
python verify_bundle.py --package <the 2026-10-21-farhan-run folder>
sha256sum -c SHA256SUMS
```

Reproduce Step 1 with the package, as its runbook says. The runner ran it with `--n 200` and one card per cap:

```bash
LL_COMMIT=e32d523 bash setup_probe.sh
../probe-venv/bin/python baseline_probe.py --lag-ladder ../lag-ladder --prompts gsm8k_test_prompts.json \
    --n 200 --max-new 256 --out probe-instruct-256
../probe-venv/bin/python baseline_probe.py --lag-ladder ../lag-ladder --prompts gsm8k_test_prompts.json \
    --n 200 --max-new 768 --out probe-instruct-768
```

- **The diagnostic:** `code/diagnostic/norp_probe.py <package dir>` with the same arguments plus `--start`, one shard
  per card, then `merge_diag.py`.
- **The argmax check:** `argmax_check.py <package dir> <lag-ladder dir> <out.json> label=records.jsonl ...` on one
  CUDA card.

`code/diagnostic/run_diag.sh` and `code/argmax256/task.sh` show the exact invocations, with site paths redacted.
