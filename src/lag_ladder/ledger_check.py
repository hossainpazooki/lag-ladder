"""Ledger lint. Runs in CI; every check fails closed.

Adapted from linear-ceiling `src/linear_ceiling/ledger_check.py` (UPSTREAM.md provenance). Kept:
structure, chain, block diff, verdict-cell provenance. Dropped: the E7 corpus-manifest citation check
(no corpus here) and the frozen provenance map (every cell in this ledger is set by a `verdict:` line
from entry 0001 on). Tightened: the chain is REQUIRED from entry 0002 on, not merely checked when present.

1. STRUCTURE: entry numbers unique; every id in REQUIRED_IDS has a verdict cell from the allowed
   vocabulary. REQUIRED_IDS grows with each registering entry; it starts empty.

2. CHAIN (entry 0002 on): each entry records the sha256 of everything from the `## Entries` heading
   up to (not including) its own heading, over universal-newline text as UTF-8. A silent edit to any
   entry that HAS a successor fails here; an entry from 0002 on without the line fails here. The
   header prose and the hypotheses table sit above `## Entries` and are outside the chain by design.

3. BLOCK DIFF (`--against <rev>`): every `### NNNN` block present in `ledger/ledger.md` at <rev> must
   be byte-identical now (universal newlines; trailing whitespace of a block ignored). Closes the hole
   the chain leaves: the TRAILING entry has no successor hashing it. Locally <rev> defaults to HEAD; in
   CI it is the base of the push or of the pull request. An unreadable base is REFUSED, never skipped.
   What it cannot see: a squash-merge, and a history rewrite that regenerates chain and blocks together.

4. VERDICT-CELL PROVENANCE: a non-`unresolved` cell must equal the newest `verdict: H-XX = <VERDICT>`
   line among the entries; a cell no line sets must read `unresolved`. Reopening is
   `verdict: H-XX = unresolved`.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from lag_ladder import REPO_ROOT

REQUIRED_IDS: tuple[str, ...] = ()     # grows with each registering entry (the entry names the id it adds)
VERDICTS = ("unresolved", "HELD", "NOT CONFIRMED", "WITHDRAWN", "SUPERSEDED", "SHELVED", "UNESTIMABLE")
CHAIN_REQUIRED_FROM = 2

_ENTRY = re.compile(r"^### (\d{4}) — \d{4}-\d{2}-\d{2} — .+$", re.M)
_ROW = re.compile(r"^\| (H-[A-Za-z0-9]+) \|.*\| ([^|]+) \|\s*$", re.M)
_ENTRIES_HEAD = re.compile(r"^## Entries\s*$", re.M)
_CHAIN = re.compile(r"^prior-entries-sha256: ([0-9a-f]{64})\s*$", re.M)
_VERDICT_LINE = re.compile(r"^verdict: (H-[A-Za-z0-9]+) = (.+?)\s*$", re.M)
LEDGER_REL = "ledger/ledger.md"


def chain_hash(text: str, upto: int, entries_start: int) -> str:
    """sha256 of the entries section from `## Entries` (inclusive) to `upto` (exclusive)."""
    return hashlib.sha256(text[entries_start:upto].encode("utf-8")).hexdigest()


def parse_ledger(text: str) -> dict:
    entries = [int(m.group(1)) for m in _ENTRY.finditer(text)]
    hyps = {m.group(1): m.group(2).strip() for m in _ROW.finditer(text)}
    return {"entries": entries, "hypotheses": hyps}


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _blocks(text: str) -> list[tuple[int, str]]:
    """(entry number, block text) for every `### NNNN` entry, in file order."""
    marks = list(_ENTRY.finditer(text))
    return [(int(m.group(1)), text[m.start():(marks[i + 1].start() if i + 1 < len(marks) else len(text))])
            for i, m in enumerate(marks)]


def _check_chain(text: str) -> list[str]:
    head = _ENTRIES_HEAD.search(text)
    marks = list(_ENTRY.finditer(text))
    problems = []
    for i, m in enumerate(marks):
        num = int(m.group(1))
        block_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        c = _CHAIN.search(text, m.start(), block_end)
        if not c:
            if num >= CHAIN_REQUIRED_FROM:
                problems.append(f"entry {num:04d} carries no prior-entries-sha256 line; every entry from "
                                f"{CHAIN_REQUIRED_FROM:04d} on hashes the entries section above it")
            continue
        if head is None:
            return ["a prior-entries-sha256 line exists but no '## Entries' heading was found"]
        want = chain_hash(text, m.start(), head.start())
        if c.group(1) != want:
            problems.append(
                f"entry {m.group(1)}: prior-entries-sha256 {c.group(1)[:12]}... does not match the "
                f"entries section above it ({want[:12]}...); a registered entry was edited after "
                "the chain recorded it, or the chain line itself is wrong"
            )
    return problems


def _check_provenance(text: str, d: dict) -> list[str]:
    problems: list[str] = []
    lines: dict[str, tuple[int, str]] = {}
    for num, block in _blocks(text):
        for m in _VERDICT_LINE.finditer(block):
            hid, v = m.group(1), m.group(2).strip()
            if hid not in d["hypotheses"]:
                problems.append(f"entry {num:04d}: `verdict:` line names {hid}, which is not in the table")
            elif v not in VERDICTS:
                problems.append(f"entry {num:04d}: `verdict:` line sets {hid} to {v!r}, not one of {VERDICTS}")
            elif hid not in lines or num > lines[hid][0]:
                lines[hid] = (num, v)
    for hid, cell in d["hypotheses"].items():
        if hid in lines:
            want, src = lines[hid][1], f"entry {lines[hid][0]:04d}'s `verdict:` line"
        else:
            want, src = "unresolved", "no entry sets it"
        if cell != want:
            problems.append(f"{hid} verdict cell is {cell!r} but {src} says {want!r}; a cell changes only by a "
                            f"numbered entry carrying `verdict: {hid} = <VERDICT>`")
    return problems


def check(text: str) -> list[str]:
    """Checks 1, 2 and 4 on the text alone (no git)."""
    text = _norm(text)
    d = parse_ledger(text)
    problems = []
    seen = set()
    for e in d["entries"]:
        if e in seen:
            problems.append(f"duplicate entry number {e:04d}")
        seen.add(e)
    for hid in REQUIRED_IDS:
        if hid not in d["hypotheses"]:
            problems.append(f"hypothesis {hid} is not registered in the table")
    for hid, v in d["hypotheses"].items():
        if v not in VERDICTS:
            problems.append(f"{hid} has verdict {v!r}, not one of {VERDICTS}")
    problems.extend(_check_chain(text))
    problems.extend(_check_provenance(text, d))
    return problems


def check_against(now_text: str, base_text: str) -> list[str]:
    """Check 3: every entry block at the base revision is byte-identical now."""
    base = {n: b.rstrip() for n, b in _blocks(_norm(base_text))}
    now = {n: b.rstrip() for n, b in _blocks(_norm(now_text))}
    problems = []
    for num in sorted(base):
        if num not in now:
            problems.append(f"entry {num:04d} is present at the base revision but gone now; registered "
                            "entries are never removed")
            continue
        if now[num] != base[num]:
            a, b = base[num].split("\n"), now[num].split("\n")
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            old = a[k] if k < len(a) else "<end of block>"
            new = b[k] if k < len(b) else "<end of block>"
            problems.append(f"entry {num:04d} differs from its text at the base revision (block line {k + 1}: "
                            f"{old[:80]!r} -> {new[:80]!r}); registered entries are immutable -- append a "
                            "new entry instead")
    return problems


def ledger_at(rev: str, repo_root: Path = REPO_ROOT) -> str:
    """`ledger/ledger.md` as committed at `rev`, or a refusal: an unreadable base is a
    force-push or a broken checkout, and the check must fail rather than skip."""
    r = subprocess.run(["git", "show", f"{rev}:{LEDGER_REL}"], cwd=repo_root, capture_output=True,
                       text=True, encoding="utf-8")
    if r.returncode != 0:
        raise ValueError(f"cannot read {LEDGER_REL} at {rev!r} ({r.stderr.strip()[:120]}); the block-diff "
                         "check refuses rather than skips")
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lag_ladder.ledger_check")
    ap.add_argument("--against", default="HEAD", metavar="REV",
                    help="revision whose entry blocks must be byte-identical now (default HEAD)")
    a = ap.parse_args(argv)
    now = (REPO_ROOT / "ledger" / "ledger.md").read_text(encoding="utf-8")
    problems = check(now)
    try:
        problems += check_against(now, ledger_at(a.against))
    except ValueError as e:
        problems.append(str(e))
    for p in problems:
        print("LEDGER:", p)
    print(f"ledger ok (blocks unchanged vs {a.against})" if not problems else f"ledger: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
