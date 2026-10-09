"""The weight-distance ladder driver (design §12, ledger 0007): f*(tau) per rung, where a rung is a writer
checkpoint and a reader checkpoint of one pair, and the x-axis is their relative weight distance.

A rung's writer and reader are refs: one side of an upstream pair at a Hub revision, at the Hub default, or at
a local checkpoint directory under `results_dir/checkpoints` (a converted `.bin` checkpoint, or a perturbed
copy of the noise anchor written by `tools/perturb_checkpoint.py` for the backbone). Every dump, score and
distance is computed by the pinned instrument's interpreter in its own tree, by subprocess; nothing from it is
imported. The distance of a rung is recomputed from the two checkpoints as dumped (`tools/weight_distance.py`)
and recorded with the sha256 of every weight file it read, so the summarizer can tie it to the dumps'
checkpoint manifests.

Controls, per distinct writer ref, before any rung that uses it: identity (the writer read by itself) must give
f* at most `identity_max` on every held-out sequence; scrambled (one sequence's cache read against the next
sequence's at the same positions) must give a median f* of at least `scrambled_min`. The run halts on the
first failure. `run` refuses unless the config is registered by a committed entry, committed as-is, pinned to
an upstream commit whose four (A) paths are the invoked bytes, every token file hashes to its registered value,
every real local checkpoint exists with its provenance, the tools exist, and a prediction is sealed under
`weight-distance`. `plan` prints the jobs and decides nothing; `check` runs the gate and nothing else.
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import DistanceConfig, Ref, Rung, load_distance_config, load_seal_config
from lag_ladder.fstar_record import evaluate_job, tau_key
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.seal import require_sealed
from lag_ladder.upstream_gate import check_upstream

UPSTREAM_PATHS = ("scripts/dump_kv.py", "kvt/data.py", "scripts/score_positions.py", "kvt/pertoken.py")
SEAL_NAME = "weight-distance"
RUN_RECORD = "run.json"
PERTOKEN = "pertoken.npz"
DISTANCE_FILE = "distance.json"
TOOLS = REPO_ROOT / "tools"
TOOL_FILES = ("perturb_checkpoint.py", "weight_distance.py")


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"distance REFUSED: {msg}")


@dataclass(frozen=True)
class DumpRef:
    pair: str
    which: str
    revision: str | None
    local_path: str | None

    @classmethod
    def of(cls, pair: str, ref: Ref) -> "DumpRef":
        return cls(pair, ref.which, ref.revision, ref.local_path)

    def key(self) -> str:
        """The dump directory name: the side and what pins it (two sides of a pair at the Hub default are
        two different checkpoints and must never share a dump)."""
        if self.local_path:
            pin = f"local-{self.local_path}"
        elif self.revision:
            pin = self.revision
        else:
            pin = "default"
        return f"{self.which}@{pin}"

    def label(self) -> str:
        return f"{self.pair}:{self.key()}"


@dataclass(frozen=True)
class Job:
    kind: str                 # identity | scrambled | rung
    pair: str
    label: str                # the rung label, or the writer ref's label for a control
    seq: int
    src: DumpRef              # the writer's cache (candidate)
    tgt: DumpRef              # the reader's own cache (reference)
    src_seq: int
    tgt_seq: int
    rung: Rung | None = None


@dataclass(frozen=True)
class Plan:
    writers: tuple[DumpRef, ...]
    jobs: tuple[Job, ...]


def plan(cfg: DistanceConfig) -> Plan:
    """Per distinct writer ref in order of first use: identity, then scrambled; then every rung in config
    order (real rungs, then the noise backbone). Sequences in held-out order inside each."""
    seqs = list(range(*cfg.heldout))
    writers: list[DumpRef] = []
    for r in cfg.rungs:
        w = DumpRef.of(r.pair, r.writer)
        if w not in writers:
            writers.append(w)
    jobs: list[Job] = []
    for w in writers:
        for i in seqs:
            jobs.append(Job("identity", w.pair, w.label(), i, w, w, i, i))
        for pos, i in enumerate(seqs):
            j = seqs[(pos + 1) % len(seqs)]
            jobs.append(Job("scrambled", w.pair, w.label(), i, w, w, i, j))
    for r in cfg.rungs:
        w, rd = DumpRef.of(r.pair, r.writer), DumpRef.of(r.pair, r.reader)
        for i in seqs:
            jobs.append(Job("rung", r.pair, r.label, i, w, rd, i, i, r))
    return Plan(tuple(writers), tuple(jobs))


# --- layout ------------------------------------------------------------------------------------------------

def pair_root(cfg: DistanceConfig, pair: str) -> Path:
    return cfg.results_dir / "pairs" / pair


def dump_dir(cfg: DistanceConfig, ref: DumpRef, seq: int) -> Path:
    return pair_root(cfg, ref.pair) / "dumps" / ref.key() / f"seq{seq}"


def checkpoints_dir(cfg: DistanceConfig) -> Path:
    return cfg.results_dir / "checkpoints"


def rung_dir(cfg: DistanceConfig, label: str) -> Path:
    return cfg.results_dir / "rungs" / label


def job_dir(cfg: DistanceConfig, j: Job) -> Path:
    if j.kind == "rung":
        return rung_dir(cfg, j.label) / f"seq{j.seq}"
    return pair_root(cfg, j.pair) / "controls" / f"{j.src.key()}-{j.kind}" / f"seq{j.seq}"


def tokens_dir(cfg: DistanceConfig, pair: str) -> Path:
    return pair_root(cfg, pair) / "tokens"


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


def used_pairs(cfg: DistanceConfig) -> list[str]:
    out = []
    for r in cfg.rungs:
        if r.pair not in out:
            out.append(r.pair)
    return out


def local_checkpoints(cfg: DistanceConfig) -> list[str]:
    """Local checkpoint names the real rungs need to exist before the run (noise rungs make their own)."""
    out = []
    for r in cfg.rungs:
        if r.kind != "real":
            continue
        for ref in (r.writer, r.reader):
            if ref.local_path and ref.local_path not in out:
                out.append(ref.local_path)
    return out


def assert_ready(cfg: DistanceConfig, repo_root: Path, *, seal_cfg=None, upstream_check=check_upstream,
                 tools_dir: Path = TOOLS) -> None:
    repo_root = Path(repo_root)
    cname = cfg.config_path.name
    if not cfg.registered_by:
        raise _refuse(f"config/{cname} is UNREGISTERED (registered_by is empty); a numbered ledger entry must fix "
                      "its values before anything runs")
    for name in used_pairs(cfg):
        if cfg.pairs[name].tokens_sha256.endswith("_PENDING"):
            raise _refuse(f"config/{cname} still carries the tokens_sha256 placeholder for pair {name}")
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
        raise _refuse(f"the committed ledger has no entry {cfg.registered_by}, which config/{cname} names")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="distance")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for name in used_pairs(cfg):
        pt = cfg.pairs[name]
        if not pt.tokens.exists():
            raise _refuse(f"held-out token file {pt.tokens} for pair {name} does not exist")
        got = sha256_file_bytes(pt.tokens)
        if got != pt.tokens_sha256:
            raise _refuse(f"token file for pair {name} hashes to {got[:12]}, not the registered {pt.tokens_sha256[:12]}")
    for name in local_checkpoints(cfg):
        d = checkpoints_dir(cfg) / name
        if not (d / "model.safetensors").exists() or not (d / "provenance.json").exists():
            raise _refuse(f"local checkpoint {d} is missing model.safetensors or provenance.json; "
                          "tools/convert_bin_checkpoint.py writes both")
    for t in TOOL_FILES:
        if not (Path(tools_dir) / t).exists():
            raise _refuse(f"tool {t} is missing under {tools_dir}")
    if seal_cfg is None:
        seal_cfg = load_seal_config(repo_root / "config" / "seal.toml", repo_root)
    try:
        require_sealed(SEAL_NAME, seal_cfg, repo_root=repo_root)
    except RuntimeError as e:
        raise _refuse(f"no committed seal for {SEAL_NAME}: {e}") from e
    upstream_python(cfg.upstream_path)


# --- the instrument, by subprocess ------------------------------------------------------------------------

CHECKPOINT_ROOT_ENV = "KVT_CHECKPOINT_ROOT"   # the instrument locates a dump's local checkpoint under it to verify the dump


def _run(cfg: DistanceConfig, cmd_tail: list[str], runner, what: str) -> None:
    cmd = [str(upstream_python(cfg.upstream_path)), *cmd_tail]
    # KVDump.load re-verifies a local checkpoint's manifest on every load (dumps from converted or perturbed
    # checkpoints); the instrument finds it as $KVT_CHECKPOINT_ROOT/<local_checkpoint>.
    env = {**os.environ, CHECKPOINT_ROOT_ENV: str(checkpoints_dir(cfg).resolve())}
    r = runner(cmd, cwd=str(cfg.upstream_path), capture_output=True, env=env)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"{what} failed:\n{err[-2000:]}")


def run_upstream(cfg: DistanceConfig, module: str, args: list[str], runner=subprocess.run) -> None:
    _run(cfg, ["-m", module, *args], runner, f"upstream command ({module})")


def run_tool(cfg: DistanceConfig, tool: str, args: list[str], runner=subprocess.run, tools_dir: Path = TOOLS) -> None:
    _run(cfg, [str((Path(tools_dir) / tool).resolve()), *args], runner, f"tool {tool}")


def prepare_tokens(cfg: DistanceConfig, pair: str) -> tuple[dict[int, Path], Path, int]:
    seqs = np.load(cfg.pairs[pair].tokens)
    if seqs.ndim != 2:
        raise _refuse(f"{cfg.pairs[pair].tokens.name} is not a [n_seqs, len] array")
    lo, hi = cfg.heldout
    if hi > seqs.shape[0]:
        raise _refuse(f"heldout [{lo}, {hi}) exceeds the {seqs.shape[0]} sequences in {cfg.pairs[pair].tokens.name}")
    tdir = tokens_dir(cfg, pair)
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


def dump(cfg: DistanceConfig, ref: DumpRef, seq: int, tokens: Path, runner=subprocess.run) -> Path:
    out = dump_dir(cfg, ref, seq)
    if (out / "meta.json").exists():
        return out
    args = ["--pair", ref.pair, "--which", ref.which]
    if ref.revision:
        args += ["--revision", ref.revision]
    elif ref.local_path:
        args += ["--local-path", str((checkpoints_dir(cfg) / ref.local_path).resolve())]
    args += ["--tokens", str(tokens.resolve()), "--stride", "1", "--dtype", cfg.dtype, "--out", str(out.resolve())]
    run_upstream(cfg, "scripts.dump_kv", args, runner)
    if not (out / "meta.json").exists():
        raise _refuse(f"dump did not produce {out}/meta.json")
    return out


def score(cfg: DistanceConfig, j: Job, pairs_npz: Path, runner=subprocess.run) -> Path:
    out = job_dir(cfg, j)
    report, pt = out / "report.json", out / PERTOKEN
    if report.exists() and pt.exists():
        return out
    run_upstream(cfg, "scripts.score_positions",
                 ["--same-src", str(dump_dir(cfg, j.src, j.src_seq).resolve()),
                  "--same-tgt", str(dump_dir(cfg, j.tgt, j.tgt_seq).resolve()),
                  "--pairs", str(pairs_npz.resolve()), "--out", str(report.resolve()), "--per-token", str(pt.resolve())],
                 runner)
    if not (report.exists() and pt.exists()):
        raise _refuse(f"score_positions did not write both {report.name} and {pt.name} under {out}")
    return out


def _checkpoint_block(cfg: DistanceConfig, ref: DumpRef, seq: int) -> dict:
    meta = json.loads((dump_dir(cfg, ref, seq) / "meta.json").read_text(encoding="utf-8"))
    block = meta.get("checkpoint")
    if not block:
        raise _refuse(f"dump {dump_dir(cfg, ref, seq)} carries no checkpoint provenance; the pin predates G3")
    return block


def spec_of(cfg: DistanceConfig, ref: DumpRef, seq: int) -> str:
    """What tools/weight_distance.py reads for this ref: the local directory, or model_id@resolved_commit as
    the dump recorded it (so the distance is over the bytes that were dumped)."""
    if ref.local_path:
        return str((checkpoints_dir(cfg) / ref.local_path).resolve())
    block = _checkpoint_block(cfg, ref, seq)
    commit = block.get("resolved_commit") or ""
    return f"{block['model_id']}@{commit}"


def perturb(cfg: DistanceConfig, rung: Rung, anchor_seq: int, runner=subprocess.run, tools_dir: Path = TOOLS) -> Path:
    out = checkpoints_dir(cfg) / rung.reader.local_path
    if (out / "provenance.json").exists():
        return out
    if out.exists():
        raise _refuse(f"{out} exists without provenance.json; remove it before rerunning")
    block = _checkpoint_block(cfg, DumpRef.of(rung.pair, rung.writer), anchor_seq)
    run_tool(cfg, "perturb_checkpoint.py",
             ["--model-id", block["model_id"], "--revision", block.get("resolved_commit") or rung.writer.revision,
              "--rel-norm", f"{rung.rel_norm:g}", "--seed", str(cfg.seed), "--out", str(out.resolve())],
             runner, tools_dir)
    if not (out / "provenance.json").exists():
        raise _refuse(f"perturb_checkpoint.py did not write {out}/provenance.json")
    return out


def distance(cfg: DistanceConfig, rung: Rung, seq: int, runner=subprocess.run, tools_dir: Path = TOOLS) -> Path:
    out = rung_dir(cfg, rung.label) / DISTANCE_FILE
    if out.exists():
        return out
    w, rd = DumpRef.of(rung.pair, rung.writer), DumpRef.of(rung.pair, rung.reader)
    run_tool(cfg, "weight_distance.py", ["--writer", spec_of(cfg, w, seq), "--reader", spec_of(cfg, rd, seq),
                                         "--out", str(out.resolve())], runner, tools_dir)
    if not out.exists():
        raise _refuse(f"weight_distance.py did not write {out}")
    return out


def run(cfg: DistanceConfig, repo_root: Path, *, runner=subprocess.run, limit: int | None = None, seal_cfg=None,
        upstream_check=check_upstream, tools_dir: Path = TOOLS, prune_noise: bool = True) -> dict:
    assert_ready(cfg, repo_root, seal_cfg=seal_cfg, upstream_check=upstream_check, tools_dir=tools_dir)
    p = plan(cfg)
    taus = [float(t) for t in cfg.rule["tau_ladder"]]
    tk = tau_key(cfg.rule["tau_K"])
    lo, hi = cfg.heldout
    n_seqs = hi - lo
    tokens: dict[str, tuple[dict, Path, int]] = {}
    for name in used_pairs(cfg):
        tokens[name] = prepare_tokens(cfg, name)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cfg.results_dir / RUN_RECORD
    record = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "dtype": cfg.dtype,
              "heldout": [lo, hi], "seed": cfg.seed, "config_sha256": sha256_file_bytes(cfg.config_path),
              "tokens_sha256": {name: cfg.pairs[name].tokens_sha256 for name in used_pairs(cfg)},
              "seq_len": {name: t[2] for name, t in tokens.items()},
              "local_checkpoints": {name: sha256_file_bytes(checkpoints_dir(cfg) / name / "model.safetensors")
                                    for name in local_checkpoints(cfg)},
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished": None,
              "halted": None, "rungs": {}, "jobs": []}

    def save():
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    scrambled: dict[str, list[float]] = {}
    done = 0
    current_noise: str | None = None
    try:
        for j in p.jobs:
            if limit is not None and done >= limit:
                break
            files, pairs_npz, _ = tokens[j.pair]
            if j.kind == "rung" and j.rung.kind == "noise":
                # the anchor's dumps exist (its controls ran first); make the perturbed reader once per rung
                perturb(cfg, j.rung, lo, runner, tools_dir)
            for ref, s in ((j.src, j.src_seq), (j.tgt, j.tgt_seq)):
                dump(cfg, ref, s, files[s], runner)
            if j.kind == "rung" and j.label not in record["rungs"]:
                dfile = distance(cfg, j.rung, lo, runner, tools_dir)
                drec = json.loads(dfile.read_text(encoding="utf-8"))
                entry = {"kind": j.rung.kind, "pair": j.pair, "writer": j.src.label(), "reader": j.tgt.label(),
                         "distance_file": dfile.relative_to(cfg.results_dir).as_posix(),
                         "distance_sha256": sha256_file_bytes(dfile), "rel_delta_all": drec.get("rel_delta_all"),
                         "rel_delta_k_proj": drec.get("rel_delta_k_proj")}
                if j.rung.kind == "noise":
                    prov = json.loads((checkpoints_dir(cfg) / j.tgt.local_path / "provenance.json").read_text(encoding="utf-8"))
                    entry.update({"target_rel_norm": j.rung.rel_norm, "realized_rel_norm": prov.get("realized_rel_norm"),
                                  "perturbed_sha256": prov.get("out_sha256")})
                record["rungs"][j.label] = entry
            out = score(cfg, j, pairs_npz, runner)
            ev = evaluate_job(out, taus, who="distance")
            record["jobs"].append({"kind": j.kind, "pair": j.pair, "label": j.label, "seq": j.seq,
                                   "dir": out.relative_to(cfg.results_dir).as_posix(),
                                   "report_sha256": ev["report_sha256"], "pertoken_sha256": ev["pertoken_sha256"],
                                   "fstar": ev["fstar"]})
            done += 1
            f = ev["fstar"][tk]
            if j.kind == "identity" and f > float(cfg.controls["identity_max"]):
                raise _refuse(f"identity control failed: {j.label} seq{j.seq} reads f*(tau_K) = {f:.6f} > "
                              f"{cfg.controls['identity_max']}; the instrument does not reproduce its own cache")
            if j.kind == "scrambled":
                scrambled.setdefault(j.label, []).append(f)
                if len(scrambled[j.label]) == n_seqs:
                    med = float(np.median(scrambled[j.label]))
                    if med < float(cfg.controls["scrambled_min"]):
                        raise _refuse(f"scrambled control failed: {j.label} median f*(tau_K) = {med:.4f} < "
                                      f"{cfg.controls['scrambled_min']}; the instrument cannot see the cache")
            if j.kind == "rung" and j.rung.kind == "noise" and prune_noise and j.seq == hi - 1:
                # every score of this rung is written and verified; the perturbed weights are not needed again
                weights = checkpoints_dir(cfg) / j.tgt.local_path / "model.safetensors"
                if weights.exists():
                    weights.unlink()
                    record["rungs"][j.label]["pruned"] = True
            save()
    except Refusal as e:
        record["halted"] = str(e)
        save()
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return record


def render_plan(cfg: DistanceConfig, p: Plan) -> str:
    n = cfg.heldout[1] - cfg.heldout[0]
    lines = [f"distance plan: {len(p.writers)} writer refs, {len(cfg.rungs)} rungs, {len(p.jobs)} jobs "
             f"({n} held-out sequences each; dtype {cfg.dtype}); registered_by={cfg.registered_by or 'UNREGISTERED'}"]
    for w in p.writers:
        lines.append(f"  controls  {w.label()}")
    for r in cfg.rungs:
        lines.append(f"  {r.kind:5s} {r.label:28s} {DumpRef.of(r.pair, r.writer).label()} -> {DumpRef.of(r.pair, r.reader).label()}"
                     + (f"  target {r.rel_norm:g}" if r.rel_norm else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the weight-distance ladder: f*(tau) per rung")
    ap.add_argument("command", choices=["plan", "check", "run"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "distance.toml"))
    ap.add_argument("--limit", type=int, default=None, help="run: stop after this many score jobs (resumable)")
    ap.add_argument("--keep-noise-checkpoints", action="store_true", help="run: do not delete perturbed weights after scoring")
    a = ap.parse_args(argv)
    cfg = load_distance_config(Path(a.config), REPO_ROOT)
    try:
        if a.command == "plan":
            print(render_plan(cfg, plan(cfg)))
        elif a.command == "check":
            assert_ready(cfg, REPO_ROOT)
            print("distance gate ok")
        else:
            rec = run(cfg, REPO_ROOT, limit=a.limit, prune_noise=not a.keep_noise_checkpoints)
            print(f"distance: {len(rec['jobs'])} jobs recorded under {cfg.results_dir}; run summarize_distance")
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
