"""Statistic (B) text `agentic` (the 2026-10-22 plan, cell 1b): the MLPerf Agentic Inference windows as a held-out
token file for the own run, under the instrument pair's tokenizer.

Input: the rendered trajectories exported by linear-ceiling (`python -m linear_ceiling.e8_text --suites
mlperf-coding mlperf-workflow --out <rendered.jsonl>`: one {suite, traj_id, text} per line). The window rule is
linear-ceiling's entry 0016 §4, restated here so this repository imports nothing from it: one window per
trajectory, the FIRST seq_len tokens; a trajectory shorter than seq_len is skipped, never padded; n windows
stratified equally over the suites, drawn with make_rng(seed) from each suite's eligible trajectories sorted by
traj_id, written SUITE BY SUITE in the suites' order (coding rows first, then workflow), so the draw is a pure
function of seed + rendered file + tokenizer and, at the same seed, the SAME rows in the SAME order as
linear-ceiling's data/e8m token file: the two repositories read one set of windows (ruled 2026-10-22; the
builder's --assert-equal checks the ids row by row against that file). Every row is held out -- nothing in the
own run trained on this text -- so config/ownrun_agentic.toml reads all n rows, not a contiguous tail. Output:
the int64 [n, seq_len] token file at the path the config names (under the instrument's gitignored data/tokens/)
and a manifest sidecar beside it ({rule, rendered_sha256, tokenizer, seed, shape, sha256, rows, equal_to}). The
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

RULE = ("linear-ceiling ledger entry 0016 section 4 (first window per trajectory, equal stratification, rows written "
        "suite by suite), restated in tools/agentic_tokens.py; at the same seed the same rows as linear-ceiling's file")


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
    matrix and the drawn {suite, traj_id} rows, suite by suite in the suites' order (module docstring)."""
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
    for s in suites:
        for tid, ids in drawn[s]:
            rows.append(ids)
            manifest.append({"suite": s, "traj_id": tid})
    return np.asarray(rows, dtype=np.int64), manifest


def assert_equal_ids(tokens: np.ndarray, rows: list[dict], other_npy: Path) -> dict:
    """The proof that the two repositories read one set of windows: every row of `tokens` equals the same row of
    linear-ceiling's token file, id by id (its manifest sidecar, if present, must name the same trajectories in the
    same order). Returns {path, sha256, manifest_sha256, rows_compared}; raises AssertionError naming the first
    differing row otherwise."""
    other_npy = Path(other_npy)
    other = np.load(other_npy)
    if other.shape != tokens.shape:
        raise AssertionError(f"shape {list(tokens.shape)} here, {list(other.shape)} in {other_npy.as_posix()}")
    for i in range(tokens.shape[0]):
        if not np.array_equal(tokens[i], other[i]):
            k = int(np.argmax(tokens[i] != other[i]))
            raise AssertionError(f"row {i} differs from {other_npy.as_posix()} at position {k}: "
                                 f"{int(tokens[i, k])} here, {int(other[i, k])} there")
    out = {"path": other_npy.as_posix(), "sha256": hashlib.sha256(other_npy.read_bytes()).hexdigest(),
           "manifest_sha256": None, "rows_compared": int(tokens.shape[0])}
    mp = other_npy.with_suffix(".manifest.json")
    if mp.exists():
        theirs = [{"suite": r["suite"], "traj_id": r["traj_id"]} for r in json.loads(mp.read_text(encoding="utf-8"))["rows"]]
        if theirs != rows:
            first = next(i for i, (a, b) in enumerate(zip(theirs, rows)) if a != b)
            raise AssertionError(f"manifest row {first}: {rows[first]} here, {theirs[first]} in {mp.as_posix()}")
        out["manifest_sha256"] = hashlib.sha256(mp.read_bytes()).hexdigest()
    return out


def write_tokens(out: Path, tokens: np.ndarray, rows: list[dict], *, rendered_sha256: str, tokenizer,
                 seed: int, equal_to: dict | None = None) -> Path:
    """The .npy and its .manifest.json sidecar; prints nothing."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tokens)
    man = {"rule": RULE, "rendered_sha256": rendered_sha256, "tokenizer": tokenizer, "seed": int(seed),
           "shape": list(tokens.shape), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "rows": rows,
           "equal_to": equal_to}
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
    ap.add_argument("--seed", type=int, default=8, help="linear-ceiling's config/e8m.toml seed, so the draws coincide")
    ap.add_argument("--out", required=True, help="the token file at the path config/ownrun_agentic.toml names")
    ap.add_argument("--assert-equal", type=Path, default=None,
                    help="linear-ceiling's token file (data/e8m/agent_n50_len1024_seed8.npy); every row's ids must match")
    a = ap.parse_args(argv)
    rendered = Path(a.rendered)
    items = read_rendered(rendered)
    tokens, rows = sample_windows(items, tokenizer_encoder(Path(a.tokenizer_json)), tuple(a.suites), a.n, a.seq_len, a.seed)
    equal_to = assert_equal_ids(tokens, rows, a.assert_equal) if a.assert_equal else None
    tj = Path(a.tokenizer_json)     # not resolved: a Hub snapshot's tokenizer.json is a symlink into blobs/
    tokenizer = {"snapshot": tj.parent.name if tj.parent.parent.name == "snapshots" else None,
                 "sha256": hashlib.sha256(tj.read_bytes()).hexdigest()}
    out = write_tokens(Path(a.out), tokens, rows, rendered_sha256=hashlib.sha256(rendered.read_bytes()).hexdigest(),
                       tokenizer=tokenizer, seed=a.seed, equal_to=equal_to)
    for f in (out, out.with_suffix(".manifest.json")):
        print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.as_posix()}")
    if equal_to:
        print(f"equal: all {equal_to['rows_compared']} rows' ids match {equal_to['path']} ({equal_to['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
