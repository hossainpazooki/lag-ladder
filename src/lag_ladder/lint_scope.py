"""The scope sentence appears exactly once in README.md, verbatim, and no sentence anywhere in
README/ledger/docs paraphrases it. Heuristic for paraphrase: a sentence whose distinctive vocabulary
overlaps heavily with the scope sentence's, and that is not the verbatim sentence itself, is flagged.

Adapted from linear-ceiling `src/linear_ceiling/lint_scope.py` (UPSTREAM.md provenance): the mechanism
is verbatim; SCOPE_SENTENCE and EXEMPT are this repo's. The threshold 0.45 was calibrated there on two
anchor cases; this repo's tests re-anchor it on its own sentence (tests/test_lint_scope.py).
"""
import re
import sys

from lag_ladder import REPO_ROOT

SCOPE_SENTENCE = ("The ladder measures what a KV cache written under one policy checkpoint costs when read "
                  "under a later one, as a function of how many optimizer updates apart they are; it does "
                  "not train on stale rollouts and does not measure serving latency.")

# Verbatim source documents: committed exactly as authored. They must not be edited to dodge this lint,
# so they are exempted by exact filename instead of relying on the paraphrase heuristic to spare them.
EXEMPT = {"docs/2026-09-19-holdover-design.md"}

_SENTENCE_BOUNDARY = r"(?<=[.!?])\s+"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "if", "in", "is", "it",
    "its", "nor", "not", "of", "on", "or", "our", "she", "should", "so", "such", "than",
    "that", "the", "their", "then", "these", "they", "this", "those", "to", "was", "we",
    "were", "what", "when", "which", "while", "who", "whom", "will", "with", "would", "you",
    "your",
}

_STEM_LEN = 6


def _stem(word: str) -> str:
    return word if len(word) <= _STEM_LEN else word[:_STEM_LEN]


def _distinctive_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS}


_SCOPE_STEMS = _distinctive_stems(SCOPE_SENTENCE)
_PARAPHRASE_THRESHOLD = 0.45


def _overlap_score(sentence: str) -> float:
    if not _SCOPE_STEMS:
        return 0.0
    stems = _distinctive_stems(sentence)
    return len(stems & _SCOPE_STEMS) / len(_SCOPE_STEMS)


def _normalize(text: str) -> str:
    # join wrapped lines (and blockquote markers) so a verbatim sentence split over lines still matches
    text = re.sub(r"\n>\s?", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_readme(text: str) -> list[str]:
    n = _normalize(text).count(SCOPE_SENTENCE)
    return [] if n == 1 else [f"README.md: scope sentence appears {n} times, expected exactly 1"]


def check_paraphrase(text: str, label: str) -> list[str]:
    flat = _normalize(text)
    problems = []
    for sent in re.split(_SENTENCE_BOUNDARY, flat):
        s = sent.strip()
        if not s or SCOPE_SENTENCE in s:
            continue
        score = _overlap_score(s)
        if score >= _PARAPHRASE_THRESHOLD:
            problems.append(f"{label}: sentence closely paraphrases the scope sentence "
                            f"(word overlap {score:.2f} >= {_PARAPHRASE_THRESHOLD}) and is not "
                            f"the verbatim scope sentence: '{s}'")
    return problems


def _ascii_safe(s: str) -> str:
    # print() output must be ASCII on the Windows-local cp1252 console; files stay UTF-8.
    return s.encode("ascii", "backslashreplace").decode("ascii")


def main() -> int:
    problems = check_readme((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    files = [REPO_ROOT / "README.md", REPO_ROOT / "ledger" / "ledger.md",
             *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    for f in files:
        if "superpowers" in f.parts:
            continue
        rel = f.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT:
            continue
        problems += check_paraphrase(f.read_text(encoding="utf-8"), rel)
    for p in problems:
        print("SCOPE:", _ascii_safe(p))
    print("scope ok" if not problems else f"scope: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
