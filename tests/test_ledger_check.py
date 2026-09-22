"""Adapted from linear-ceiling tests/test_ledger_check.py: the manifest-citation and frozen-map cases are
dropped with their checks; the chain-required-from-0002 case is new."""
import re

import pytest

from lag_ladder import REPO_ROOT
from lag_ladder.ledger_check import (
    CHAIN_REQUIRED_FROM, REQUIRED_IDS, VERDICTS, chain_hash, check, check_against, ledger_at, parse_ledger,
)
from tests.conftest import commit_all

HEAD_TEXT = """# Ledger
| id | statement | decided by | verdict |
|---|---|---|---|
| H-P1 | s | pilot | unresolved |
| H-P2 | s | pilot | unresolved |
## Entries
### 0001 — 2026-09-22 — first
body
"""


def _append(text: str, num: int, body: str, *, chain: bool = True) -> str:
    """Append entry `num`; from CHAIN_REQUIRED_FROM on it carries the correct prior-entries-sha256."""
    entries_start = re.search(r"^## Entries\s*$", text, re.M).start()
    heading = f"### {num:04d} — 2026-09-22 — t\n"
    line = f"prior-entries-sha256: {chain_hash(text, len(text), entries_start)}\n" if chain and num >= CHAIN_REQUIRED_FROM else ""
    return text + heading + line + body + "\n"


GOOD = _append(HEAD_TEXT, 2, "second")


def test_parse_finds_entries_and_verdicts():
    d = parse_ledger(GOOD)
    assert d["entries"] == [1, 2]
    assert d["hypotheses"] == {"H-P1": "unresolved", "H-P2": "unresolved"}


def test_check_accepts_good():
    assert check(GOOD) == []


def test_check_flags_duplicate_entry_and_bad_verdict():
    bad = GOOD.replace("### 0002", "### 0001").replace("| pilot | unresolved |", "| pilot | probably |", 1)
    problems = check(bad)
    assert any("duplicate" in p for p in problems) and any("verdict" in p for p in problems)


def test_required_ids_start_empty_and_are_checked_when_named():
    assert REQUIRED_IDS == ()
    from lag_ladder import ledger_check
    old = ledger_check.REQUIRED_IDS
    try:
        ledger_check.REQUIRED_IDS = ("H-P9",)
        assert any("H-P9" in p for p in check(GOOD))
    finally:
        ledger_check.REQUIRED_IDS = old


# --- chain (check 2) --------------------------------------------------------------------------

def test_chain_required_from_0002():
    missing = _append(HEAD_TEXT, 2, "second", chain=False)
    problems = check(missing)
    assert any("0002 carries no prior-entries-sha256" in p for p in problems)
    assert check(HEAD_TEXT) == []            # 0001 alone needs no chain line


def test_chain_detects_edit_to_prior_entry():
    tampered = GOOD.replace("### 0001 — 2026-09-22 — first", "### 0001 — 2026-09-22 — first (reworded)")
    assert any("prior-entries-sha256" in p and "does not match" in p for p in check(tampered))


def test_chain_ignores_header_and_table_edits():
    edited = (GOOD.replace("| H-P1 | s | pilot | unresolved |", "| H-P1 | s | pilot | HELD |")
              + "")
    # the cell change itself is a provenance problem (no verdict line), not a chain problem
    assert not any("prior-entries-sha256" in p for p in check(edited))


def test_chain_hashes_crlf_and_lf_identically():
    assert check(GOOD.replace("\n", "\r\n")) == []


# --- verdict-cell provenance (check 4) ----------------------------------------------------------

def test_a_cell_no_entry_sets_is_refused_by_name():
    problems = check(GOOD.replace("| H-P2 | s | pilot | unresolved |", "| H-P2 | s | pilot | HELD |"))
    assert any(p.startswith("H-P2 verdict cell is 'HELD' but no entry sets it says 'unresolved'") for p in problems)


def test_verdict_line_sets_the_cell_and_the_newest_wins():
    held = GOOD.replace("| H-P2 | s | pilot | unresolved |", "| H-P2 | s | pilot | HELD |")
    assert check(_append(held, 3, "verdict: H-P2 = HELD")) == []
    reopened = _append(_append(GOOD, 3, "verdict: H-P2 = HELD"), 4, "verdict: H-P2 = unresolved")
    assert check(reopened) == []
    stale = _append(_append(held, 3, "verdict: H-P2 = HELD"), 4, "verdict: H-P2 = unresolved")
    assert any("entry 0004's `verdict:` line says 'unresolved'" in p for p in check(stale))


def test_verdict_line_must_name_a_registered_id_and_a_known_verdict():
    problems = check(_append(GOOD, 3, "verdict: H-X9 = HELD\nverdict: H-P1 = MAYBE"))
    assert any("names H-X9" in p for p in problems) and any("'MAYBE'" in p for p in problems)


def test_every_vocabulary_verdict_is_settable():
    for v in VERDICTS:
        text = _append(GOOD.replace("| H-P1 | s | pilot | unresolved |", f"| H-P1 | s | pilot | {v} |"),
                       3, f"verdict: H-P1 = {v}")
        assert check(text) == [], v


# --- block diff against a base revision (check 3) ----------------------------------------------

def _commit_ledger(repo, text):
    (repo / "ledger" / "ledger.md").write_text(text, encoding="utf-8")
    commit_all(repo, "ledger")


def test_block_diff_accepts_an_append_and_header_edits(repo):
    _commit_ledger(repo, GOOD)
    base = ledger_at("HEAD", repo)
    assert check_against(_append(GOOD, 3, "appended"), base) == []
    assert check_against(GOOD.replace("# Ledger", "# Ledger (header reworded)"), base) == []
    assert check_against(GOOD.replace("| H-P1 | s | pilot | unresolved |", "| H-P1 | s | pilot | HELD |"), base) == []
    assert check_against(GOOD.replace("\n", "\r\n"), base) == []
    assert check_against(GOOD + "\n\n", base) == []


def test_block_diff_refuses_an_edit_to_the_trailing_entry(repo):
    text = _append(GOOD, 3, "f*(tau_K) median 0.31 UNRESOLVED")
    _commit_ledger(repo, text)
    base = ledger_at("HEAD", repo)
    assert check(text) == [] and check_against(text, base) == []
    edited = text.replace("median 0.31 UNRESOLVED", "median 0.01 HOLDS")
    assert check(edited) == []                                    # the chain alone is blind to the trailing entry
    problems = check_against(edited, base)
    assert len(problems) == 1 and problems[0].startswith("entry 0003 differs from its text at the base revision")
    assert "immutable" in problems[0]


def test_block_diff_refuses_a_removed_entry(repo):
    _commit_ledger(repo, GOOD)
    problems = check_against(HEAD_TEXT, ledger_at("HEAD", repo))
    assert problems == ["entry 0002 is present at the base revision but gone now; registered entries are never removed"]


def test_block_diff_refuses_an_unreadable_base(repo):
    _commit_ledger(repo, GOOD)
    with pytest.raises(ValueError, match="cannot read ledger/ledger.md at 'HEAD~5'"):
        ledger_at("HEAD~5", repo)


# --- the real ledger ----------------------------------------------------------------------------

def test_repo_ledger_is_clean():
    text = (REPO_ROOT / "ledger" / "ledger.md").read_text(encoding="utf-8")
    assert check(text) == []
    d = parse_ledger(text)
    assert 1 in d["entries"]
    assert all(v in VERDICTS for v in d["hypotheses"].values())


def test_repo_ledger_blocks_match_head():
    """The working tree's entry blocks equal HEAD's -- what CI runs against the push base. Red until
    the repo has a first commit; that is the check refusing, not skipping."""
    now = (REPO_ROOT / "ledger" / "ledger.md").read_text(encoding="utf-8")
    assert check_against(now, ledger_at("HEAD")) == []
