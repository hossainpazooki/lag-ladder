"""The prefix cut (ledger 0042): a spans sidecar for the held-out token file, so the reader scores held-out sequence 40 + i
with a shorter prefix and a continuation as long as rollout row i's completion.

The reader driver passes a per-sequence `--prefix-len` and `--length` for any pair that carries a spans file
(`reader.span_args`), indexed by the sequence number. A cache is causal, so the scorer's first prefix_len - 1 rows of a
768-row dump are the cache of the shorter prefix (`score_continuation.py` reads `k[:rows]`): the existing dumps serve
every cut and nothing is re-dumped. Rows outside the held-out range are null, so a job on any other sequence refuses.

    python tools/cut_spans.py --rollout-spans R.spans.json --rollout-spans-sha256 HEX --cut prompt --out OUT.spans.json

Cuts: `prompt` (prefix = rollout row i's prompt length), or a fixed prefix length in tokens (`192`, `384`). In every
cut the continuation is rollout row i's completion length, length_i - prefix_len_i of the rollout file. Bytes are a
pure function of the rollout spans file and the arguments; the registering config pins their sha256. Refuses to
overwrite. Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path

RULE = ("held-out sequence lo + i: prefix_len = the cut (rollout row i's prompt length for `prompt`, else the fixed "
        "length), length = prefix_len + (rollout length_i - rollout prefix_len_i); rows outside [lo, hi) are null")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cut_spans(rollout: dict, cut: str, heldout: tuple[int, int], n_rows: int, seq_len: int) -> dict:
    """The spans object for one cut. Refuses a cut whose continuation would not fit the row or leave no token."""
    lo, hi = heldout
    pl, ln = rollout.get("prefix_len"), rollout.get("length")
    if not isinstance(pl, list) or not isinstance(ln, list) or len(pl) != len(ln):
        raise ValueError("the rollout spans need lists prefix_len and length of one size")
    if not (0 <= lo < hi <= n_rows):
        raise ValueError(f"held-out range [{lo}, {hi}) is not inside {n_rows} rows")
    if hi - lo != len(pl):
        raise ValueError(f"held-out range [{lo}, {hi}) has {hi - lo} sequences; the rollout file has {len(pl)} rows")
    if cut != "prompt" and not (cut.isdigit() and int(cut) > 1):
        raise ValueError(f"cut must be 'prompt' or a prefix length in tokens, not {cut!r}")
    prefix, length = [None] * n_rows, [None] * n_rows
    for i, (p, e) in enumerate(zip(pl, ln)):
        p, e = int(p), int(e)
        c = e - p
        if c < 2:
            raise ValueError(f"rollout row {i}: completion of {c} tokens")
        q = p if cut == "prompt" else int(cut)
        if q + c > seq_len:
            raise ValueError(f"held-out sequence {lo + i}: prefix {q} + continuation {c} > row length {seq_len}")
        prefix[lo + i], length[lo + i] = q, q + c
    return {"prefix_len": prefix, "length": length}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-spans", required=True)
    ap.add_argument("--rollout-spans-sha256", required=True, help="the rollout spans file's registered sha256")
    ap.add_argument("--cut", required=True, help="'prompt' or a prefix length in tokens")
    ap.add_argument("--heldout", type=int, nargs=2, default=[40, 50])
    ap.add_argument("--rows", type=int, default=50, help="rows of the held-out token file")
    ap.add_argument("--seq-len", type=int, default=1024, help="tokens per row of the held-out token file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.rollout_spans)
    raw = src.read_bytes()
    got = sha256_bytes(raw)
    if got != a.rollout_spans_sha256:
        raise SystemExit(f"{src} hashes to {got[:12]}, not the registered {a.rollout_spans_sha256[:12]}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    try:
        spans = cut_spans(json.loads(raw.decode("utf-8")), a.cut, tuple(a.heldout), a.rows, a.seq_len)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    rec = {**spans, "cut": a.cut, "heldout": list(a.heldout), "rule": RULE,
           "rollout_spans_sha256": got, "rows": a.rows, "seq_len": a.seq_len,
           "tool": "lag-ladder tools/cut_spans.py"}
    data = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{out} sha256 {sha256_bytes(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
