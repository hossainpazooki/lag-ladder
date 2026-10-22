"""Statistic (B) text `agentic` (the 2026-10-22 plan, cell 1b): the MLPerf Agentic Inference windows as a held-out
token file for the own run, under the instrument pair's tokenizer.

Input: the rendered trajectories exported by linear-ceiling (`python -m linear_ceiling.e8_text --suites
mlperf-coding mlperf-workflow --out <rendered.jsonl>`: one {suite, traj_id, text} per line). The window rule is
linear-ceiling's entry 0016 §4, restated here so this repository imports nothing from it: one window per
trajectory, the FIRST seq_len tokens; a trajectory shorter than seq_len is skipped, never padded; n windows
stratified equally over the suites, drawn with make_rng(seed) from each suite's eligible trajectories sorted by
traj_id, so the draw is a pure function of seed + rendered file + tokenizer. One departure from linear-ceiling's
row order, stated here because the reader reads a CONTIGUOUS held-out range: the drawn rows are INTERLEAVED by
suite (row 0 coding, row 1 workflow, ...) instead of written suite by suite, so every contiguous range of even
length holds the suites equally and config/ownrun_agentic.toml's heldout = [40, 50) reads 5 + 5, not 10 of one
suite. Output: the int64 [n, seq_len] token file at the path the config names (under the instrument's gitignored
data/tokens/) and a manifest sidecar beside it ({rule, rendered_sha256, tokenizer, shape, sha256, rows}). The
reader takes P = tokens[:prefix_len] and C = the rest, the registered convention; no spans sidecar, nothing padded.

Run with any interpreter that has `tokenizers` (the instrument's does); this file imports lag_ladder.rng only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lag_ladder.rng import make_rng  # noqa: E402

RULE = ("linear-ceiling ledger entry 0016 section 4 (first window per trajectory, equal stratification), restated in "
        "tools/agentic_tokens.py; rows interleaved by suite so every contiguous held-out range is balanced")


def read_rendered(path: Path) -> list[dict]:
    """The exported trajectories; refuses a line without suite, traj_id and text."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict) or any(not isinstance(r.get(k), str) for k in ("suite", "traj_id", "text")):
                raise ValueError(f"{path}:{i}: not a {{suite, traj_id, text}} record")
            items.append(r)
    return items


def sample_windows(items: list[dict], encode, suites: tuple[str, ...], n: int, seq_len: int, seed: int
                   ) -> tuple[np.ndarray, list[dict]]:
    """Eligible = at least seq_len tokens. Equal draw per suite with make_rng(seed); the [n, seq_len] int64
    matrix and the drawn {suite, traj_id} rows, interleaved by suite (module docstring)."""
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("suites must be distinct and non-empty")
    if n % len(suites):
        raise ValueError(f"n={n} does not stratify equally over {len(suites)} suites")
    per = n // len(suites)
    eligible: dict[str, list[tuple[str, list[int]]]] = {s: [] for s in suites}
    for it in sorted(items, key=lambda r: (r["suite"], r["traj_id"])):
        if it["suite"] not in eligible:
            continue
        ids = list(encode(it["text"]))
        if len(ids) >= seq_len:
            eligible[it["suite"]].append((it["traj_id"], ids[:seq_len]))
    rng = make_rng(seed)
    drawn: dict[str, list[tuple[str, list[int]]]] = {}
    for s in suites:
        pool = eligible[s]
        if len(pool) < per:
            raise ValueError(f"suite {s}: only {len(pool)} trajectories have >= {seq_len} tokens; need {per}")
        drawn[s] = [pool[i] for i in sorted(rng.choice(len(pool), size=per, replace=False).tolist())]
    rows, manifest = [], []
    for k in range(per):
        for s in suites:
            tid, ids = drawn[s][k]
            rows.append(ids)
            manifest.append({"suite": s, "traj_id": tid})
    return np.asarray(rows, dtype=np.int64), manifest


def write_tokens(out: Path, tokens: np.ndarray, rows: list[dict], *, rendered_sha256: str, tokenizer,
                 seed: int) -> Path:
    """The .npy and its .manifest.json sidecar; prints nothing."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tokens)
    man = {"rule": RULE, "rendered_sha256": rendered_sha256, "tokenizer": tokenizer, "seed": int(seed),
           "shape": list(tokens.shape), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "rows": rows}
    out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def tokenizer_encoder(tokenizer_json: Path):
    """Token ids from a snapshot's tokenizer.json, no special tokens added (as linear-ceiling's qwen_encoder)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    return lambda text: tok.encode(text, add_special_tokens=False).ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the agentic text's held-out token file (window rule in the module docstring)")
    ap.add_argument("--rendered", required=True, help="linear-ceiling's exported {suite, traj_id, text} JSONL")
    ap.add_argument("--tokenizer-json", required=True, help="the instrument pair's tokenizer.json (a Hub snapshot's)")
    ap.add_argument("--suites", nargs="+", default=["mlperf-coding", "mlperf-workflow"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="the token file at the path config/ownrun_agentic.toml names")
    a = ap.parse_args(argv)
    rendered = Path(a.rendered)
    items = read_rendered(rendered)
    tokens, rows = sample_windows(items, tokenizer_encoder(Path(a.tokenizer_json)), tuple(a.suites), a.n, a.seq_len, a.seed)
    tj = Path(a.tokenizer_json)     # not resolved: a Hub snapshot's tokenizer.json is a symlink into blobs/
    tokenizer = {"snapshot": tj.parent.name if tj.parent.parent.name == "snapshots" else None,
                 "sha256": hashlib.sha256(tj.read_bytes()).hexdigest()}
    out = write_tokens(Path(a.out), tokens, rows, rendered_sha256=hashlib.sha256(rendered.read_bytes()).hexdigest(),
                       tokenizer=tokenizer, seed=a.seed)
    for f in (out, out.with_suffix(".manifest.json")):
        print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
