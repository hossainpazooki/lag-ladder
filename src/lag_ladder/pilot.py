"""The (A)-only pilot driver: f*(tau_K) on a ladder of published checkpoints (design §9 step 3).

For each anchor a (the writer checkpoint) and each later revision r (the reader), and each held-out
sequence, the pinned instrument dumps the K/V cache of the same tokens under both checkpoints and scores
the writer's cache against the reader's own at every position; `lag_ladder.fstar_record` turns that
record into f*(tau). The instrument is invoked in its own environment, by subprocess, on the bytes the
pin names; nothing from it is imported.

Two controls run before any ladder cell is scored and the run stops on the first failure:
  * identity: the anchor's cache read by the anchor (k = 0) must give f* at most `identity_max` on every
    held-out sequence -- design §3: at k = 0 every arm is the fresh cache, so a nonzero reading is a
    defect of the instrument on this machine, never a result;
  * scrambled: the anchor's cache for one sequence read against the same revision's cache for another
    sequence, position by position, must give a median f* of at least `scrambled_min`, or the instrument
    cannot see the cache and `summarize_pilot` produces no ladder table.

`run` refuses unless the config is registered by a committed ledger entry, committed as-is, pinned to
an upstream commit whose four (A) paths are the invoked bytes, sealed (a prediction for the pair is
committed first, invariant 1), and the held-out token file hashes to the registered value. `plan`
prints the jobs and decides nothing. `check` runs the gate and nothing else.
"""
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import PilotConfig, load_pilot_config, load_seal_config
from lag_ladder.fstar_record import evaluate_job, tau_key
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.seal import require_sealed
from lag_ladder.upstream_gate import check_upstream

# The (A) surface entry 0002 names; a change to any of these after the pin turns the gate red.
UPSTREAM_PATHS = ("scripts/dump_kv.py", "kvt/data.py", "scripts/score_positions.py", "kvt/pertoken.py")
RUN_RECORD = "run.json"
PERTOKEN = "pertoken.npz"
KINDS = ("identity", "scrambled", "ladder")


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"pilot REFUSED: {msg}")


def steps(revision: str) -> int:
    m = re.fullmatch(r"step_(\d+)", revision)
    if not m:
        raise _refuse(f"revision {revision!r} is not of the form step_<int>; the lag is read from the name")
    return int(m.group(1))


@dataclass(frozen=True)
class DumpJob:
    revision: str
    seq: int


@dataclass(frozen=True)
class ScoreJob:
    kind: str              # identity | scrambled | ladder
    anchor: str
    partner: str
    seq: int
    src: DumpJob           # the writer's cache (candidate)
    tgt: DumpJob           # the reader's own cache (reference)
    lag: int


@dataclass(frozen=True)
class Plan:
    dumps: tuple[DumpJob, ...]
    scores: tuple[ScoreJob, ...]


def plan(cfg: PilotConfig) -> Plan:
    """Every dump and score the run performs, in run order: per anchor, identity, then scrambled, then
    the ladder by increasing lag; sequences in held-out order inside each."""
    seqs = list(range(*cfg.heldout))
    if len(seqs) < 2:
        raise _refuse("the scrambled control needs at least two held-out sequences")
    scores: list[ScoreJob] = []
    for a in cfg.anchors:
        sa = steps(a)
        for i in seqs:
            scores.append(ScoreJob("identity", a, a, i, DumpJob(a, i), DumpJob(a, i), 0))
        for pos, i in enumerate(seqs):
            j = seqs[(pos + 1) % len(seqs)]
            scores.append(ScoreJob("scrambled", a, a, i, DumpJob(a, i), DumpJob(a, j), 0))
        partners = sorted((r for r in cfg.revisions if steps(r) > sa), key=steps)
        for r in partners:
            for i in seqs:
                scores.append(ScoreJob("ladder", a, r, i, DumpJob(a, i), DumpJob(r, i), steps(r) - sa))
    seen, dumps = set(), []
    for s in scores:
        for d in (s.src, s.tgt):
            if d not in seen:
                seen.add(d)
                dumps.append(d)
    return Plan(tuple(dumps), tuple(scores))


def pair_dir(cfg: PilotConfig) -> Path:
    return cfg.results_dir / cfg.pair


def dump_dir(cfg: PilotConfig, d: DumpJob) -> Path:
    return pair_dir(cfg) / "dumps" / d.revision / f"seq{d.seq}"


def score_dir(cfg: PilotConfig, s: ScoreJob) -> Path:
    cell = f"{s.anchor}-scrambled" if s.kind == "scrambled" else f"{s.anchor}-to-{s.partner}"
    return pair_dir(cfg) / cell / f"seq{s.seq}"


def tokens_dir(cfg: PilotConfig) -> Path:
    return pair_dir(cfg) / "tokens"


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


def assert_ready(cfg: PilotConfig, repo_root: Path, *, seal_cfg=None, upstream_check=check_upstream) -> None:
    """The gate. Order matters: the cheap, local refusals first; the upstream checkout last."""
    repo_root = Path(repo_root)
    cname = cfg.config_path.name
    if not cfg.registered_by:
        raise _refuse(f"config/{cname} is UNREGISTERED (registered_by is empty); a numbered ledger entry must fix "
                      "its values before anything runs")
    if cfg.tokens_sha256.endswith("_PENDING"):
        raise _refuse(f"config/{cname} still carries the tokens_sha256 placeholder; the registering entry records "
                      "the held-out token file's hash")
    for rel in ("ledger/ledger.md", cfg.config_path):
        p = repo_root / rel if isinstance(rel, str) else rel
        if not _committed_as_is(p, repo_root):
            raise _refuse(f"{p.name} is not committed as-is; entry {cfg.registered_by} and config/{cname} must be "
                          "committed before any dump")
    committed = subprocess.run(["git", "show", "HEAD:ledger/ledger.md"], cwd=repo_root,
                               capture_output=True, text=True, encoding="utf-8")
    if committed.returncode != 0:
        raise _refuse("cannot read HEAD:ledger/ledger.md")
    if f"### {cfg.registered_by} " not in committed.stdout:
        raise _refuse(f"the committed ledger has no entry {cfg.registered_by}, which config/{cname} names as its "
                      "registering entry")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="pilot")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    if not cfg.tokens.exists():
        raise _refuse(f"held-out token file {cfg.tokens} does not exist; the upstream's prepare_tokens writes it")
    got = sha256_file_bytes(cfg.tokens)
    if got != cfg.tokens_sha256:
        raise _refuse(f"held-out token file hashes to {got[:12]}, not the registered {cfg.tokens_sha256[:12]}")
    if seal_cfg is None:
        seal_cfg = load_seal_config(repo_root / "config" / "seal.toml", repo_root)
    try:
        require_sealed(cfg.pair, seal_cfg, repo_root=repo_root)
    except RuntimeError as e:
        raise _refuse(f"no committed seal for {cfg.pair}: {e}") from e
    upstream_python(cfg.upstream_path)


def run_upstream(cfg: PilotConfig, module: str, args: list[str], runner=subprocess.run) -> None:
    cmd = [str(upstream_python(cfg.upstream_path)), "-m", module, *args]
    r = runner(cmd, cwd=str(cfg.upstream_path), capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"upstream command failed ({module}):\n{err[-2000:]}")


def prepare_tokens(cfg: PilotConfig) -> tuple[dict[int, Path], Path, int]:
    """One single-sequence token file per held-out sequence, and the identity pairs file for their
    length. Returns ({seq: path}, pairs path, T)."""
    seqs = np.load(cfg.tokens)
    if seqs.ndim != 2:
        raise _refuse(f"{cfg.tokens.name} is not a [n_seqs, len] array")
    lo, hi = cfg.heldout
    if hi > seqs.shape[0]:
        raise _refuse(f"heldout [{lo}, {hi}) exceeds the {seqs.shape[0]} sequences in {cfg.tokens.name}")
    tdir = tokens_dir(cfg)
    tdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i in range(lo, hi):
        p = tdir / f"seq{i}.npy"
        if not p.exists():
            np.save(p, seqs[i:i + 1])
        files[i] = p
    T = int(seqs.shape[1])
    pairs = tdir / f"pairs_identity_T{T}.npz"
    if not pairs.exists():
        pos = np.arange(T, dtype=np.int64)
        np.savez(pairs, pairs=np.stack([pos, pos], 1))
    return files, pairs, T


def dump(cfg: PilotConfig, d: DumpJob, tokens: Path, runner=subprocess.run) -> Path:
    out = dump_dir(cfg, d)
    if (out / "meta.json").exists():
        return out                                   # resume: the instrument refuses to overwrite anyway
    run_upstream(cfg, "scripts.dump_kv", ["--pair", cfg.pair, "--which", "source", "--revision", d.revision,
                                           "--tokens", str(tokens.resolve()), "--stride", "1",
                                           "--dtype", cfg.dtype, "--out", str(out.resolve())], runner)
    if not (out / "meta.json").exists():
        raise _refuse(f"dump did not produce {out}/meta.json")
    return out


def score(cfg: PilotConfig, s: ScoreJob, pairs: Path, runner=subprocess.run) -> Path:
    out = score_dir(cfg, s)
    report, pt = out / "report.json", out / PERTOKEN
    if report.exists() and pt.exists():
        return out
    run_upstream(cfg, "scripts.score_positions",
                 ["--same-src", str(dump_dir(cfg, s.src).resolve()), "--same-tgt", str(dump_dir(cfg, s.tgt).resolve()),
                  "--pairs", str(pairs.resolve()), "--out", str(report.resolve()), "--per-token", str(pt.resolve())],
                 runner)
    if not (report.exists() and pt.exists()):
        raise _refuse(f"score_positions did not write both {report.name} and {pt.name} under {out}")
    return out


def run(cfg: PilotConfig, repo_root: Path, *, runner=subprocess.run, limit: int | None = None,
        seal_cfg=None, upstream_check=check_upstream) -> dict:
    assert_ready(cfg, repo_root, seal_cfg=seal_cfg, upstream_check=upstream_check)
    p = plan(cfg)
    files, pairs, T = prepare_tokens(cfg)
    taus = [float(t) for t in cfg.rule["tau_ladder"]]
    tk = tau_key(cfg.rule["tau_K"])
    record = {"pair": cfg.pair, "registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha,
              "dtype": cfg.dtype, "heldout": list(cfg.heldout), "seq_len": T,
              "tokens_sha256": cfg.tokens_sha256, "config_sha256": sha256_file_bytes(cfg.config_path),
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished": None,
              "halted": None, "jobs": []}
    pair_dir(cfg).mkdir(parents=True, exist_ok=True)
    rec_path = pair_dir(cfg) / RUN_RECORD

    def save():
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    scrambled: dict[str, list[float]] = {a: [] for a in cfg.anchors}
    done = 0
    try:
        for s in p.scores:
            if limit is not None and done >= limit:
                break
            for d in (s.src, s.tgt):
                dump(cfg, d, files[d.seq], runner)
            out = score(cfg, s, pairs, runner)
            ev = evaluate_job(out, taus, who="pilot")
            job = {"kind": s.kind, "anchor": s.anchor, "partner": s.partner, "seq": s.seq, "lag": s.lag,
                   "dir": out.relative_to(cfg.results_dir).as_posix(), "report_sha256": ev["report_sha256"],
                   "pertoken_sha256": ev["pertoken_sha256"], "fstar": ev["fstar"]}
            record["jobs"].append(job)
            done += 1
            f = ev["fstar"][tk]
            if s.kind == "identity" and f > float(cfg.controls["identity_max"]):
                raise _refuse(f"identity control failed: {s.anchor} seq{s.seq} reads f*(tau_K) = {f:.6f} > "
                              f"{cfg.controls['identity_max']}; the instrument does not reproduce its own cache on "
                              "this machine")
            if s.kind == "scrambled":
                scrambled[s.anchor].append(f)
                n_seqs = cfg.heldout[1] - cfg.heldout[0]
                if len(scrambled[s.anchor]) == n_seqs:
                    med = float(np.median(scrambled[s.anchor]))
                    if med < float(cfg.controls["scrambled_min"]):
                        raise _refuse(f"scrambled control failed: {s.anchor} median f*(tau_K) = {med:.4f} < "
                                      f"{cfg.controls['scrambled_min']}; the instrument cannot see the cache")
            save()
    except Refusal as e:
        record["halted"] = str(e)
        save()
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return record


def render_plan(cfg: PilotConfig, p: Plan) -> str:
    lines = [f"pilot plan for {cfg.pair}: {len(p.dumps)} dumps, {len(p.scores)} scores "
             f"(held-out {cfg.heldout[0]}..{cfg.heldout[1] - 1}, dtype {cfg.dtype}); registered_by="
             f"{cfg.registered_by or 'UNREGISTERED'}"]
    for s in p.scores:
        lines.append(f"  {s.kind:9s} {s.anchor} -> {s.partner} seq{s.seq} lag {s.lag}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the (A)-only pilot: f*(tau_K) over a checkpoint ladder")
    ap.add_argument("command", choices=["plan", "check", "run"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "pilot.toml"))
    ap.add_argument("--limit", type=int, default=None, help="run: stop after this many score jobs (resumable)")
    a = ap.parse_args(argv)
    cfg = load_pilot_config(Path(a.config), REPO_ROOT)
    try:
        if a.command == "plan":
            print(render_plan(cfg, plan(cfg)))
        elif a.command == "check":
            assert_ready(cfg, REPO_ROOT)
            print("pilot gate ok")
        else:
            rec = run(cfg, REPO_ROOT, limit=a.limit)
            print(f"pilot: {len(rec['jobs'])} jobs recorded under {pair_dir(cfg)}; run summarize_pilot")
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
