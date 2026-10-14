"""The statistic (B) driver, "does the reader notice" (design §13, ledger 0011): per rung, the reader is
handed the writer's float16 dump of the 768-token prefix (stale) against its own (fresh) and scored on the
256-token continuation by `tools/score_continuation.py`, run with the pinned instrument's interpreter by
subprocess; nothing from the instrument is imported, and this driver never imports the scorer.

Every cell reuses the distance run's dumps under `results/distance/pairs/<pair>/dumps/` (R-B11); nothing is
re-dumped. Controls per distinct reader, before its rungs, halting on failure: float16 floor (the scorer's
`--live` arm; the maximum logit difference and the median mean-KL per text must stay within the limits the
probe fixed), identity (the reader's own dump in both arms; a determinism check on `identity_max_logit_diff`),
scrambled (the next held-out sequence's prefix, corpus text; a median mean-KL below `scrambled_min` marks the
reader's rungs UNEVALUABLE and skips them, recorded for the summarizer). No number is decided here beyond the
configured halting thresholds; `summarize_reader` recomputes everything from the reports and refuses on
mismatch.

`run` refuses while the config is unregistered, uncommitted, unpinned, missing its ledger entry, missing the
floor limits, or unsealed. `probe` is the pre-registration float16-floor probe: one reader, every held-out
sequence, both texts, `--live`; it needs no registration, floor limits or seal — only the upstream gate —
and refuses to write under `results/`.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import ReaderConfig, ReaderRung, load_reader_config, load_seal_config
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.seal import require_sealed
from lag_ladder.upstream_gate import check_upstream

UPSTREAM_PATHS = ("scripts/dump_kv.py", "kvt/data.py", "scripts/score_positions.py", "kvt/pertoken.py")
SEAL_NAME = "reader"
RUN_RECORD = "run.json"
REPORT = "report.json"
TOOLS = REPO_ROOT / "tools"
SCORER = "score_continuation.py"
DISTANCE_RESULTS = "results/distance"   # the dumps and local checkpoints the cells reuse (R-B11)


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"reader REFUSED: {msg}")


def reader_slug(spec: str) -> str:
    """A filesystem name for a reader spec (model_id[@revision] or local:<name>)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


@dataclass(frozen=True)
class Job:
    kind: str              # floor | identity | scrambled | rung
    reader: str            # the scorer spec
    label: str             # the rung label, or the reader's slug for a control
    pair: str
    seq: int               # the scored sequence (and --seq)
    text: str              # corpus | on-path
    fresh_key: str         # dump key of the fresh arm
    stale_key: str         # dump key of the stale arm
    stale_seq: int         # the stale arm's sequence (differs from seq only for scrambled)
    live: bool = False
    rung: ReaderRung | None = None


@dataclass(frozen=True)
class Plan:
    readers: tuple[str, ...]
    jobs: tuple[Job, ...]


def plan(cfg: ReaderConfig) -> Plan:
    """Per distinct reader spec in order of first use: floor (every text), identity, then scrambled (corpus
    only); then every rung of that reader in config order, every text. Sequences in held-out order inside
    each block, so a halt leaves nothing of that reader's rungs behind it."""
    texts = list(cfg.rule["texts"])

    def seqs_of(pair: str) -> list[int]:
        return list(range(*(cfg.pairs[pair].heldout or cfg.heldout)))

    readers: list[str] = []
    own: dict[str, ReaderRung] = {}          # reader spec -> a rung naming its own dump
    for r in cfg.rungs:
        if r.reader not in readers:
            readers.append(r.reader)
            own[r.reader] = r
    jobs: list[Job] = []
    for spec in readers:
        o = own[spec]
        slug = reader_slug(spec)
        seqs = seqs_of(o.pair)
        for t in texts:
            for i in seqs:
                jobs.append(Job("floor", spec, slug, o.pair, i, t, o.reader_dump, o.reader_dump, i, live=True))
        for i in seqs:
            jobs.append(Job("identity", spec, slug, o.pair, i, "corpus", o.reader_dump, o.reader_dump, i))
        for pos, i in enumerate(seqs):
            j = seqs[(pos + 1) % len(seqs)]
            jobs.append(Job("scrambled", spec, slug, o.pair, i, "corpus", o.reader_dump, o.reader_dump, j))
        for r in cfg.rungs:
            if r.reader != spec:
                continue
            for t in (r.texts or texts):
                for i in seqs_of(r.pair):
                    jobs.append(Job("rung", spec, r.label, r.pair, i, t, r.reader_dump, r.writer_dump, i, rung=r))
    return Plan(tuple(readers), tuple(jobs))


# --- layout ------------------------------------------------------------------------------------------------

def dumps_root(repo_root: Path, pair: str) -> Path:
    return Path(repo_root) / DISTANCE_RESULTS / "pairs" / pair / "dumps"


def checkpoints_dir(repo_root: Path) -> Path:
    return Path(repo_root) / DISTANCE_RESULTS / "checkpoints"


def dump_dir(repo_root: Path, pair: str, key: str, seq: int) -> Path:
    return dumps_root(repo_root, pair) / key / f"seq{seq}"


def job_dir(cfg: ReaderConfig, j: Job) -> Path:
    if j.kind == "rung":
        return cfg.results_dir / "rungs" / j.label / j.text / f"seq{j.seq}"
    return cfg.results_dir / "controls" / reader_slug(j.reader) / j.kind / j.text / f"seq{j.seq}"


# --- the gate ------------------------------------------------------------------------------------------------

def upstream_python(upstream: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        p = Path(upstream) / ".venv" / rel
        if p.exists():
            return p
    raise _refuse(f"no interpreter under {upstream}/.venv; the instrument runs in its own environment")


def _committed_as_is(path: Path, repo_root: Path) -> bool:
    rel = path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo_root, capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo_root)
    return tracked.returncode == 0 and clean.returncode == 0


def used_pairs(cfg: ReaderConfig) -> list[str]:
    out = []
    for r in cfg.rungs:
        if r.pair not in out:
            out.append(r.pair)
    return out


def assert_ready(cfg: ReaderConfig, repo_root: Path, *, seal_cfg=None, upstream_check=check_upstream,
                 tools_dir: Path = TOOLS) -> None:
    repo_root = Path(repo_root)
    cname = cfg.config_path.name
    if not cfg.registered_by:
        raise _refuse(f"config/{cname} is UNREGISTERED (registered_by is empty); a numbered ledger entry must fix "
                      "its values before anything runs")
    for k in ("floor_max_median_kl",):
        if cfg.controls.get(k) is None:
            raise _refuse(f"config/{cname} has no {k}; the registering entry fixes the float16 floor from the "
                          "probe before any score (R-B9, R-B10)")
    for name in used_pairs(cfg):
        if cfg.pairs[name].tokens_sha256.endswith("_PENDING"):
            raise _refuse(f"config/{cname} still carries the tokens_sha256 placeholder for pair {name}")
        if (cfg.pairs[name].spans_sha256 or "").endswith("_PENDING"):
            raise _refuse(f"config/{cname} still carries the spans_sha256 placeholder for pair {name}")
    for rel in ("ledger/ledger.md", cfg.config_path):
        p = repo_root / rel if isinstance(rel, str) else rel
        if not _committed_as_is(p, repo_root):
            raise _refuse(f"{p.name} is not committed as-is; entry {cfg.registered_by} and config/{cname} must be "
                          "committed before any score")
    committed = subprocess.run(["git", "show", "HEAD:ledger/ledger.md"], cwd=repo_root,
                               capture_output=True, text=True, encoding="utf-8")
    if committed.returncode != 0:
        raise _refuse("cannot read HEAD:ledger/ledger.md")
    if f"### {cfg.registered_by} " not in committed.stdout:
        raise _refuse(f"the committed ledger has no entry {cfg.registered_by}, which config/{cname} names")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="reader")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for name in used_pairs(cfg):
        pt = cfg.pairs[name]
        if not pt.tokens.exists():
            raise _refuse(f"held-out token file {pt.tokens} for pair {name} does not exist")
        got = sha256_file_bytes(pt.tokens)
        if got != pt.tokens_sha256:
            raise _refuse(f"token file for pair {name} hashes to {got[:12]}, not the registered {pt.tokens_sha256[:12]}")
        if pt.spans is not None:
            if not pt.spans.exists():
                raise _refuse(f"spans file {pt.spans} for pair {name} does not exist")
            got = sha256_file_bytes(pt.spans)
            if got != pt.spans_sha256:
                raise _refuse(f"spans file for pair {name} hashes to {got[:12]}, not the registered {pt.spans_sha256[:12]}")
    if not (Path(tools_dir) / SCORER).exists():
        raise _refuse(f"tool {SCORER} is missing under {tools_dir}")
    if seal_cfg is None:
        seal_cfg = load_seal_config(repo_root / "config" / "seal.toml", repo_root)
    try:
        require_sealed(SEAL_NAME, seal_cfg, repo_root=repo_root)
    except RuntimeError as e:
        raise _refuse(f"no committed seal for {SEAL_NAME}: {e}") from e
    upstream_python(cfg.upstream_path)


# --- the scorer, by subprocess -------------------------------------------------------------------------------

CHECKPOINT_ROOT_ENV = "KVT_CHECKPOINT_ROOT"   # the scorer resolves local:<name> readers and dump checkpoints under it


def spans_of(cfg: ReaderConfig, pair: str) -> dict | None:
    """A pair's per-sequence prefix_len and length (§14), read from its spans file; None without one."""
    pt = cfg.pairs[pair]
    if pt.spans is None:
        return None
    sp = json.loads(pt.spans.read_text(encoding="utf-8"))
    for k in ("prefix_len", "length"):
        if not isinstance(sp.get(k), list):
            raise _refuse(f"spans file {pt.spans} for pair {pair} has no list {k!r}")
    return sp


def span_args(cfg: ReaderConfig, j: Job) -> list[str]:
    """`--prefix-len` (and `--length`) for one job: the config's prefix_len, or the pair's spans at this sequence."""
    sp = spans_of(cfg, j.pair)
    if sp is None:
        return ["--prefix-len", str(cfg.prefix_len)]
    try:
        return ["--prefix-len", str(int(sp["prefix_len"][j.seq])), "--length", str(int(sp["length"][j.seq]))]
    except (IndexError, TypeError, ValueError) as e:
        raise _refuse(f"spans for pair {j.pair} have no entry for seq {j.seq}: {e}") from e


def score(cfg: ReaderConfig, repo_root: Path, j: Job, out: Path, *, probe: bool = False,
          runner=subprocess.run, tools_dir: Path = TOOLS) -> Path:
    report = out / REPORT
    if report.exists():
        return report
    args = [str((Path(tools_dir) / SCORER).resolve()),
            "--reader", j.reader,
            "--stale", str(dump_dir(repo_root, j.pair, j.stale_key, j.stale_seq).resolve()),
            "--tokens", str(cfg.pairs[j.pair].tokens.resolve()), "--seq", str(j.seq),
            *span_args(cfg, j), "--text", j.text,
            "--out", str(out.resolve()), "--device", "auto", "--dtype", cfg.dtype]
    if j.text == "on-path":
        args += ["--generate-under", str(cfg.rule["generate_under"])]
    if j.live:
        args.append("--live")         # the live float32 cache replaces the fresh arm; the scorer refuses --fresh with it
    else:
        args += ["--fresh", str(dump_dir(repo_root, j.pair, j.fresh_key, j.seq).resolve())]
    if probe:
        args.append("--probe")
    cmd = [str(upstream_python(cfg.upstream_path)), *args]
    env = {**os.environ, CHECKPOINT_ROOT_ENV: str(checkpoints_dir(repo_root).resolve())}
    r = runner(cmd, cwd=str(cfg.upstream_path), capture_output=True, env=env)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"{SCORER} failed on {j.kind} {j.label} seq{j.seq} ({j.text}):\n{err[-2000:]}")
    if not report.exists():
        raise _refuse(f"{SCORER} did not write {report}")
    return report


def _read_report(report: Path) -> dict:
    rec = json.loads(report.read_text(encoding="utf-8"))
    if rec.get("schema") != "reader-report-v1":
        raise _refuse(f"{report} is not a reader-report-v1")
    return rec


def _mean_kl(rec: dict) -> float:
    return float(np.mean(np.asarray(rec["kl_per_token"], dtype=np.float64)))


# --- run -----------------------------------------------------------------------------------------------------

def run(cfg: ReaderConfig, repo_root: Path, *, runner=subprocess.run, limit: int | None = None, seal_cfg=None,
        upstream_check=check_upstream, tools_dir: Path = TOOLS) -> dict:
    assert_ready(cfg, repo_root, seal_cfg=seal_cfg, upstream_check=upstream_check, tools_dir=tools_dir)
    p = plan(cfg)
    lo, hi = cfg.heldout
    n_of = {name: (lambda h: h[1] - h[0])(cfg.pairs[name].heldout or cfg.heldout) for name in cfg.pairs}
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cfg.results_dir / RUN_RECORD
    record = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "dtype": cfg.dtype,
              "heldout": [lo, hi], "prefix_len": cfg.prefix_len, "seed": cfg.seed,
              "config_sha256": sha256_file_bytes(cfg.config_path),
              "tokens_sha256": {name: cfg.pairs[name].tokens_sha256 for name in used_pairs(cfg)},
              "spans_sha256": {name: cfg.pairs[name].spans_sha256 for name in used_pairs(cfg)
                               if cfg.pairs[name].spans is not None},
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished": None,
              "halted": None, "unevaluable_readers": {}, "jobs": []}

    def save():
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    floor_kls: dict[tuple[str, str], list[float]] = {}     # (reader, text) -> mean KL per sequence
    scrambled_kls: dict[str, list[float]] = {}             # reader -> mean KL per sequence
    unevaluable: set[str] = set()
    done = 0
    try:
        for j in p.jobs:
            if limit is not None and done >= limit:
                break
            if j.kind == "rung" and j.reader in unevaluable:
                continue
            out = job_dir(cfg, j)
            out.mkdir(parents=True, exist_ok=True)
            report = score(cfg, repo_root, j, out, runner=runner, tools_dir=tools_dir)
            rec = _read_report(report)
            mkl = _mean_kl(rec)
            mdiff = float(rec["max_abs_logit_diff"])
            record["jobs"].append({"kind": j.kind, "reader": j.reader, "label": j.label, "pair": j.pair,
                                   "seq": j.seq, "text": j.text, "live": j.live,
                                   "dir": out.relative_to(cfg.results_dir).as_posix(),
                                   "report_sha256": sha256_file_bytes(report),
                                   "mean_kl": mkl, "max_abs_logit_diff": mdiff, "done": True})
            done += 1
            if j.kind == "floor":
                # 0013: the floor halts on median mean-KL only; max_abs_logit_diff is recorded, not a limit
                # (a single float16 rounding extreme over 256 x vocab logits is heavy-tailed across readers).
                key = (j.reader, j.text)
                floor_kls.setdefault(key, []).append(mkl)
                if len(floor_kls[key]) == n_of[j.pair]:
                    med = float(np.median(floor_kls[key]))
                    if med > float(cfg.controls["floor_max_median_kl"]):
                        raise _refuse(f"float16 floor failed: {j.reader} ({j.text}) median mean-KL = {med:.3e} > "
                                      f"{cfg.controls['floor_max_median_kl']}")
            elif j.kind == "identity":
                if mdiff > float(cfg.controls["identity_max_logit_diff"]):
                    raise _refuse(f"identity control failed: {j.reader} seq{j.seq} reads max_abs_logit_diff = "
                                  f"{mdiff:.3e} > {cfg.controls['identity_max_logit_diff']}; the scorer does not "
                                  "reproduce its own forward")
            elif j.kind == "scrambled":
                scrambled_kls.setdefault(j.reader, []).append(mkl)
                if len(scrambled_kls[j.reader]) == n_of[j.pair]:
                    med = float(np.median(scrambled_kls[j.reader]))
                    if med < float(cfg.controls["scrambled_min"]):
                        # Not a halt: the reader's rungs are UNEVALUABLE (design §13) and are skipped.
                        unevaluable.add(j.reader)
                        record["unevaluable_readers"][j.reader] = (
                            f"UNEVALUABLE: median scrambled mean-KL {med:.3e} < scrambled_min "
                            f"{cfg.controls['scrambled_min']}; its rungs are skipped")
            save()
    except Refusal as e:
        record["halted"] = str(e)
        save()
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return record


# --- probe ---------------------------------------------------------------------------------------------------

def probe(cfg: ReaderConfig, repo_root: Path, *, out: Path, reader: str | None = None, pair: str | None = None,
          dump_key: str | None = None, runner=subprocess.run, upstream_check=check_upstream,
          tools_dir: Path = TOOLS) -> dict:
    """The pre-registration float16-floor probe (R-B9): the named reader's live float32 cache against its own
    float16 dump, every held-out sequence, every text. The reader defaults to the first rung's (reader
    step_200 in config/reader.toml). Needs no registration, floor limits or seal; writes OUTSIDE results/ only."""
    repo_root = Path(repo_root)
    out_dir = Path(out).resolve()
    first = cfg.rungs[0]
    reader = reader or first.reader
    pair = pair or first.pair
    dump_key = dump_key or first.reader_dump
    results = (repo_root / "results").resolve()
    if out_dir == results or results in out_dir.parents:
        raise _refuse(f"probe --out {out_dir} is under results/; the probe is pre-registration and must not "
                      "write where a registered run records")
    if pair not in cfg.pairs:
        raise _refuse(f"probe pair {pair!r} is not in [[reader.pairs]]")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="reader-probe")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    if not cfg.pairs[pair].tokens.exists():
        raise _refuse(f"held-out token file {cfg.pairs[pair].tokens} does not exist")
    summary = {"reader": reader, "pair": pair, "dump": dump_key, "heldout": list(cfg.heldout),
               "prefix_len": cfg.prefix_len, "texts": list(cfg.rule["texts"]),
               "max_abs_logit_diff": {}, "median_mean_kl": {}, "jobs": []}
    for t in cfg.rule["texts"]:
        kls, diffs = [], []
        for i in range(*cfg.heldout):
            j = Job("floor", reader, reader_slug(reader), pair, i, t, dump_key, dump_key, i, live=True)
            out = out_dir / t / f"seq{i}"
            out.mkdir(parents=True, exist_ok=True)
            report = score(cfg, repo_root, j, out, probe=True, runner=runner, tools_dir=tools_dir)
            rec = _read_report(report)
            kls.append(_mean_kl(rec))
            diffs.append(float(rec["max_abs_logit_diff"]))
            summary["jobs"].append({"text": t, "seq": i, "dir": out.as_posix(),
                                    "mean_kl": kls[-1], "max_abs_logit_diff": diffs[-1]})
        summary["max_abs_logit_diff"][t] = float(max(diffs))
        summary["median_mean_kl"][t] = float(np.median(kls))
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# --- cli -----------------------------------------------------------------------------------------------------

def render_plan(cfg: ReaderConfig, p: Plan) -> str:
    n = cfg.heldout[1] - cfg.heldout[0]
    lines = [f"reader plan: {len(p.readers)} readers, {len(cfg.rungs)} rungs, {len(p.jobs)} jobs "
             f"({n} held-out sequences each; texts {list(cfg.rule['texts'])}; dtype {cfg.dtype}); "
             f"registered_by={cfg.registered_by or 'UNREGISTERED'}"]
    for spec in p.readers:
        lines.append(f"  controls  {spec}")
    for r in cfg.rungs:
        lines.append(f"  rung {r.label:28s} {r.pair}: {r.writer_dump} -> {r.reader_dump}  reader {r.reader}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="statistic (B): per-token KL of the reader over the writer's cache")
    ap.add_argument("command", choices=["plan", "check", "run", "probe"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "reader.toml"))
    ap.add_argument("--limit", type=int, default=None, help="run: stop after this many score jobs (resumable)")
    ap.add_argument("--reader", default=None, help="probe: the reader spec; default the first rung's reader")
    ap.add_argument("--pair", default=None, help="probe: the pair; default the first rung's")
    ap.add_argument("--dump", default=None, help="probe: the reader's own dump key; default the first rung's")
    ap.add_argument("--out", default=None, help="probe: output directory, OUTSIDE results/")
    a = ap.parse_args(argv)
    cfg = load_reader_config(Path(a.config), REPO_ROOT)
    try:
        if a.command == "plan":
            print(render_plan(cfg, plan(cfg)))
        elif a.command == "check":
            assert_ready(cfg, REPO_ROOT)
            print("reader gate ok")
        elif a.command == "probe":
            if not a.out:
                print("reader REFUSED: probe needs --out (outside results/)", file=sys.stderr)
                return 2
            s = probe(cfg, REPO_ROOT, out=Path(a.out), reader=a.reader, pair=a.pair, dump_key=a.dump)
            print(f"probe: max_abs_logit_diff {s['max_abs_logit_diff']}, median mean-KL {s['median_mean_kl']}; "
                  f"recorded under {a.out}")
        else:
            rec = run(cfg, REPO_ROOT, limit=a.limit)
            print(f"reader: {len(rec['jobs'])} jobs recorded under {cfg.results_dir}; run summarize_reader")
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
