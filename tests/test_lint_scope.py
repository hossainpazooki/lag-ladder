"""Adapted from linear-ceiling tests/test_lint_scope.py: same cases, re-anchored on this repo's sentence."""
import io

import pytest

from lag_ladder.lint_scope import SCOPE_SENTENCE, _overlap_score, _ascii_safe, check_paraphrase, check_readme


def test_scope_sentence_is_the_design_text():
    assert SCOPE_SENTENCE == ("The ladder measures what a KV cache written under one policy checkpoint costs "
                              "when read under a later one, as a function of how many optimizer updates apart "
                              "they are; it does not train on stale rollouts and does not measure serving latency.")


def test_readme_needs_exactly_one():
    assert check_readme("no sentence here") == ["README.md: scope sentence appears 0 times, expected exactly 1"]
    assert check_readme(SCOPE_SENTENCE + "\n\n" + SCOPE_SENTENCE) == \
        ["README.md: scope sentence appears 2 times, expected exactly 1"]
    assert check_readme("x\n" + SCOPE_SENTENCE + "\ny") == []


def test_wrapped_verbatim_sentence_counts_as_one():
    wrapped = ("> The ladder measures what a KV cache written under one policy checkpoint costs when read\n"
               "> under a later one, as a function of how many optimizer updates apart they are; it does\n"
               "> not train on stale rollouts and does not measure serving latency.")
    assert check_readme(wrapped) == []


PARAPHRASE = ("The ladder measures what a KV cache written under one checkpoint costs when read under a "
              "later checkpoint, as a function of optimizer updates; serving latency is not measured.")
NEAR_MISS = ("The pilot dumps the cache at every rung of the ladder before any score is read, and the "
             "rollouts it uses were never trained on.")


def test_anchor_scores_sit_on_either_side_of_the_threshold():
    # The threshold (0.45) was calibrated in linear-ceiling on its own sentence; these two anchors
    # re-establish the margin on this repo's sentence. If either drifts across, recalibrate, don't tune.
    assert _overlap_score(PARAPHRASE) >= 0.60
    assert _overlap_score(NEAR_MISS) <= 0.30


def test_paraphrase_is_flagged():
    result = check_paraphrase(PARAPHRASE, "docs/x.md")
    assert len(result) == 1 and "closely paraphrases the scope sentence" in result[0] and PARAPHRASE in result[0]
    assert check_paraphrase(SCOPE_SENTENCE, "docs/x.md") == []


def test_paraphrase_joined_by_arrow_is_also_flagged():
    text = PARAPHRASE.replace("; serving", " → serving")
    result = check_paraphrase(text, "docs/x.md")
    assert len(result) == 1 and text in result[0]


def test_near_miss_that_shares_keywords_stays_unflagged_on_overlap():
    assert check_paraphrase(NEAR_MISS, "ledger/ledger.md") == []


def test_flagged_nonascii_sentence_is_reported_without_raising():
    text = PARAPHRASE.replace("The ladder", "The ladder ρ")
    problems = check_paraphrase(text, "docs/rho.md")
    assert len(problems) == 1 and "ρ" in problems[0]
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        print("SCOPE:", problems[0], file=stream)
    print("SCOPE:", _ascii_safe(problems[0]), file=stream)
    stream.flush()
    buf = stream.buffer.getvalue().decode("cp1252")
    assert "SCOPE:" in buf and "latency" in buf and "\\u03c1" in buf


def test_repo_passes():
    from lag_ladder.lint_scope import main
    assert main() == 0
