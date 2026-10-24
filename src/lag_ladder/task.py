"""The statistic (C) driver, "does the task notice" (design §15): the reader at step g answers GSM8K test prompts
greedily under its own prompt cache (FRESH), the step-0 writer's (STALE), another prompt's (NEAR) and a web prefix
of the prompt's length (FAR); `tools/task_swap.py` does the generation, run with the pinned instrument's interpreter by
subprocess; nothing from the instrument is imported, and this driver never imports the tool.

Plan: the dump-equivalence control first (the writer's live cache against the step-0 dump of 0017's rollout prompts,
and 0018's rollout (B1) reproduced at one reader through the tool) — a halt before any cell; then one `cells` job per
reader over every prompt, which writes the four arms and, for the first `controls.n_prompts` prompts, the floor arm
(FRESH built in float32, not rounded) and the identity arm (FRESH generated twice). After each cells job: the floor's
answer-flip rate against FRESH over its limit halts; an identity flip halts. NEAR and FAR are judged by the summarizer
(DEGRADES_C or the cell is UNEVALUABLE); no number is decided here beyond the halting thresholds.

`run` refuses while the config is unregistered, uncommitted, unpinned, missing its ledger entry, or unsealed; `check`
is the gate alone; `plan` prints the jobs and decides nothing. `probe` is the pre-registration probe (section 15): one
reader, the first n prompts, every arm and both control arms, the equivalence control first; it needs no registration,
pin or seal - only the upstream gate - refuses to write under results/, and writes probe.json with the numbers the
registering entry fixes (the floor's flip rate, the FRESH baseline, each arm's discordance and the bootstrap half-width
it implies at the registered n).
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lag_ladder import REPO_ROOT
from lag_ladder.config import TaskConfig, load_seal_config, load_task_config
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.reader import UPSTREAM_PATHS, DISTANCE_RESULTS, checkpoints_dir, reader_slug, upstream_python, _committed_as_is
from lag_ladder.seal import require_sealed
from lag_ladder.upstream_gate import check_upstream

TOOL = "task_swap.py"
TOOLS = REPO_ROOT / "tools"
RUN_RECORD = "run.json"
ARMS = ("fresh", "stale", "near", "far")
CONTROL_ARMS = ("fresh32", "fresh2")
CHECKPOINT_ROOT_ENV = "KVT_CHECKPOINT_ROOT"


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"task REFUSED: {msg}")


@dataclass(frozen=True)
class Job:
    kind: str                 # equivalence | cells
    reader: str
    writer: str
    indices: list[int] = field(default_factory=list)
    control_n: int = 0


@dataclass(frozen=True)
class Plan:
    jobs: tuple[Job, ...]


def plan(cfg: TaskConfig) -> Plan:
    jobs = [Job("equivalence", cfg.controls["equivalence_reader"], cfg.writer)]
    idx = list(range(cfg.n_prompts))
    for r in cfg.readers:
        jobs.append(Job("cells", r, cfg.writer, idx, int(cfg.controls["n_prompts"])))
    return Plan(tuple(jobs))


def job_dir(cfg: TaskConfig, j: Job) -> Path:
    if j.kind == "equivalence":
        return cfg.results_dir / "equivalence"
    return cfg.results_dir / "readers" / reader_slug(j.reader)


# --- the gate ----------------------------------------------------------------------------------------------------

def assert_ready(cfg: TaskConfig, repo_root: Path, *, seal_cfg=None, upstream_check=check_upstream,
                 tools_dir: Path = TOOLS) -> None:
    repo_root = Path(repo_root)
    cname = cfg.config_path.name
    if not cfg.registered_by:
        raise _refuse(f"config/{cname} is UNREGISTERED (registered_by is empty); a numbered ledger entry must fix "
                      "its values before anything runs")
    for name, sha in (("prompts_sha256", cfg.prompts_sha256), ("far_tokens_sha256", cfg.far_tokens_sha256),
                      ("rollout_tokens_sha256", cfg.controls["rollout_tokens_sha256"]),
                      ("rollout_spans_sha256", cfg.controls["rollout_spans_sha256"])):
        if sha.endswith("_PENDING"):
            raise _refuse(f"config/{cname} still carries the {name} placeholder")
    for rel in ("ledger/ledger.md", cfg.config_path):
        p = repo_root / rel if isinstance(rel, str) else rel
        if not _committed_as_is(p, repo_root):
            raise _refuse(f"{p.name} is not committed as-is; entry {cfg.registered_by} and config/{cname} must be "
                          "committed before any score")
    committed = subprocess.run(["git", "show", "HEAD:ledger/ledger.md"], cwd=repo_root,
                               capture_output=True, text=True, encoding="utf-8")
    if committed.returncode != 0 or f"### {cfg.registered_by} " not in committed.stdout:
        raise _refuse(f"the committed ledger has no entry {cfg.registered_by}, which config/{cname} names")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="task")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for what, path, sha in (("prompts", cfg.prompts, cfg.prompts_sha256), ("far_tokens", cfg.far_tokens, cfg.far_tokens_sha256),
                            ("rollout_tokens", cfg.controls["rollout_tokens"], cfg.controls["rollout_tokens_sha256"]),
                            ("rollout_spans", cfg.controls["rollout_spans"], cfg.controls["rollout_spans_sha256"])):
        if not Path(path).exists():
            raise _refuse(f"{what} file {path} does not exist")
        got = sha256_file_bytes(path)
        if got != sha:
            raise _refuse(f"{what} file hashes to {got[:12]}, not the registered {sha[:12]}")
    prompts = json.loads(Path(cfg.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) < cfg.n_prompts:
        raise _refuse(f"prompts file holds {len(prompts) if isinstance(prompts, list) else 'no'} rows; n_prompts is {cfg.n_prompts}")
    if not (Path(tools_dir) / TOOL).exists():
        raise _refuse(f"tool {TOOL} is missing under {tools_dir}")
    for spec in (cfg.writer, *cfg.readers, cfg.controls["equivalence_reader"]):
        if not (checkpoints_dir(repo_root) / spec[len("local:"):] / "model.safetensors").exists():
            raise _refuse(f"checkpoint for {spec} is not under {DISTANCE_RESULTS}/checkpoints")
    if seal_cfg is None:
        seal_cfg = load_seal_config(repo_root / "config" / "seal.toml", repo_root)
    try:
        require_sealed(cfg.seal, seal_cfg, repo_root=repo_root)
    except RuntimeError as e:
        raise _refuse(f"no committed seal for {cfg.seal}: {e}") from e
    upstream_python(cfg.upstream_path)


# --- the tool, by subprocess ---------------------------------------------------------------------------------------

def _call(cfg: TaskConfig, repo_root: Path, args: list[str], runner, tools_dir: Path) -> None:
    cmd = [str(upstream_python(cfg.upstream_path)), str((Path(tools_dir) / TOOL).resolve()), *args]
    env = {**os.environ, CHECKPOINT_ROOT_ENV: str(checkpoints_dir(repo_root).resolve())}
    r = runner(cmd, cwd=str(cfg.upstream_path), capture_output=True, env=env)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"{TOOL} failed on {args[0]}:\n{err[-2000:]}")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise _refuse(f"{TOOL} did not write {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r.get("schema") != "task-record-v2":
            raise _refuse(f"{path} holds a record that is not task-record-v2")
    return rows


def _by_index(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        if r["index"] in out:
            raise _refuse(f"index {r['index']} recorded twice")
        out[int(r["index"])] = r
    return out


def resolve_expect_b1(cfg: TaskConfig, repo_root: Path) -> tuple[float, dict | None]:
    """The rollout (B1) the equivalence control must reproduce: the number the config pins (0025: 0018's step-32
    median), or, when the config points at a (B) summary (0035: no earlier entry's number exists for a new
    initialisation), that summary's rung and text median read now — the (B) stage runs before (C) on the same
    machine, and the record carries the summary's sha256 beside the pointer. Refuses when the summary, the rung or the text is
    missing or withheld."""
    eb = cfg.controls["equivalence_expect_b1"]
    if not isinstance(eb, dict):
        return float(eb), None
    f = Path(eb["summary"])
    where = f"expected (B1) for the equivalence control: {eb['rung']} / {eb['text']} in {f.name}"
    if not f.exists():
        raise _refuse(f"{where}: the (B) summary {f} does not exist; the (B) stage must run first")
    s = json.loads(f.read_text(encoding="utf-8"))
    e = next((e for e in s.get("rungs", []) if e.get("label") == eb["rung"]), None)
    if e is None or e.get("withheld"):
        raise _refuse(f"{where}: the rung is missing or withheld in the summary")
    m = e.get("texts", {}).get(eb["text"], {}).get("b1", {}).get("median_kl")
    if m is None or float(m) <= 0:
        raise _refuse(f"{where}: the summary carries no positive median for that text")
    rel = Path(f).resolve().relative_to(Path(repo_root).resolve()).as_posix() if Path(f).resolve().is_relative_to(Path(repo_root).resolve()) else str(f)
    return float(m), {"summary": rel, "rung": eb["rung"], "text": eb["text"], "sha256": sha256_file_bytes(f)}


def judge_equivalence(rec: dict, cfg: TaskConfig) -> dict:
    """The limit is on the absolute gap (R-C21 as restated by 0026: 2^-5, one float16 ulp in the top binade the caches
    occupy); the relative gap and the share of differing elements are descriptives — the live-vs-dump discrepancy is
    float32 forward noise of ~1e-5 absolute, many float16 ulps of a near-zero element."""
    lim = float(cfg.controls["equivalence_max_abs_diff"])
    abs_diff = float(rec["max_abs_diff"])
    b1 = rec.get("b1") or {}
    rel = b1.get("rel_gap")
    rel_lim = float(cfg.controls["equivalence_b1_max_rel"])
    ok = abs_diff <= lim and rel is not None and float(rel) <= rel_lim
    return {"max_abs_diff": abs_diff, "limit": lim, "max_rel_diff": float(rec["max_rel_diff"]),
            "share_differing": rec.get("share_differing"),
            "b1_median": b1.get("median"), "b1_expected": b1.get("expected"),
            "b1_rel_gap": rel, "b1_rel_limit": rel_lim, "passed": ok}


def judge_controls(out: Path, control_n: int, cfg: TaskConfig) -> dict:
    fresh = _by_index(_read_jsonl(out / "fresh.jsonl"))
    f32 = _by_index(_read_jsonl(out / "fresh32.jsonl"))
    f2 = _by_index(_read_jsonl(out / "fresh2.jsonl"))
    if len(f32) != control_n or len(f2) != control_n:
        raise _refuse(f"{out}: control arms hold {len(f32)}/{len(f2)} records, expected {control_n}")
    flips = sum(f32[i]["answer"] != fresh[i]["answer"] for i in f32)
    ident = sum(f2[i]["completion_sha256"] != fresh[i]["completion_sha256"] for i in f2)
    floor_rate = flips / control_n
    return {"n": control_n, "floor_flip_rate": floor_rate, "floor_limit": float(cfg.controls["floor_max_flip_rate"]),
            "floor_passed": floor_rate <= float(cfg.controls["floor_max_flip_rate"]),
            "identity_flips": ident, "identity_passed": ident == 0}


# --- run -----------------------------------------------------------------------------------------------------------

def run(cfg: TaskConfig, repo_root: Path, *, runner=subprocess.run, seal_cfg=None, upstream_check=check_upstream,
        tools_dir: Path = TOOLS) -> dict:
    assert_ready(cfg, repo_root, seal_cfg=seal_cfg, upstream_check=upstream_check, tools_dir=tools_dir)
    p = plan(cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cfg.results_dir / RUN_RECORD
    record = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "seed": cfg.seed,
              "writer": cfg.writer, "readers": list(cfg.readers), "n_prompts": cfg.n_prompts, "max_new": cfg.max_new,
              "config_sha256": sha256_file_bytes(cfg.config_path), "prompts_sha256": cfg.prompts_sha256,
              "far_tokens_sha256": cfg.far_tokens_sha256,
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished": None,
              "halted": None, "equivalence": None, "controls": {}, "jobs": []}

    def save():
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    try:
        for j in p.jobs:
            out = job_dir(cfg, j)
            out.mkdir(parents=True, exist_ok=True)
            if j.kind == "equivalence":
                expect_b1, expect_from = resolve_expect_b1(cfg, repo_root)
                _call(cfg, repo_root, ["equivalence", "--writer", j.writer, "--dump-root", str(Path(cfg.controls["rollout_dump_root"]).resolve()),
                                       "--tokens", str(Path(cfg.controls["rollout_tokens"]).resolve()),
                                       "--spans", str(Path(cfg.controls["rollout_spans"]).resolve()),
                                       "--reader", j.reader, "--expect-b1", repr(expect_b1),
                                       "--out", str(out.resolve())], runner, tools_dir)
                eq_path = out / "equivalence.json"
                if not eq_path.exists():
                    raise _refuse(f"{TOOL} did not write {eq_path}")
                eq = json.loads(eq_path.read_text(encoding="utf-8"))
                record["equivalence"] = judge_equivalence(eq, cfg)
                if expect_from is not None:
                    record["equivalence"]["b1_expected_from"] = expect_from
                record["jobs"].append({"kind": "equivalence", "reader": j.reader, "dir": out.relative_to(cfg.results_dir).as_posix(),
                                       "file_sha256": sha256_file_bytes(eq_path), "done": True})
                if not record["equivalence"]["passed"]:
                    raise _refuse(f"dump equivalence failed: max_abs_diff {record['equivalence']['max_abs_diff']:.3e} "
                                  f"(limit {record['equivalence']['limit']}), (B1) rel gap {record['equivalence']['b1_rel_gap']} "
                                  f"(limit {record['equivalence']['b1_rel_limit']}); the live cache is not the dump's")
            else:
                idx_path = out / "indices.json"
                idx_path.write_text(json.dumps(j.indices), encoding="utf-8")
                _call(cfg, repo_root, ["cells", "--reader", j.reader, "--writer", j.writer,
                                       "--prompts", str(Path(cfg.prompts).resolve()), "--indices", str(idx_path.resolve()),
                                       "--far-tokens", str(Path(cfg.far_tokens).resolve()), "--far-seq", str(cfg.far_seq),
                                       "--max-new", str(cfg.max_new), "--control-n", str(j.control_n),
                                       "--out", str(out.resolve())], runner, tools_dir)
                for arm in ARMS:
                    rows = _by_index(_read_jsonl(out / f"{arm}.jsonl"))
                    if sorted(rows) != sorted(j.indices):
                        raise _refuse(f"{out / (arm + '.jsonl')} holds {len(rows)} records, expected {len(j.indices)}")
                ctl = judge_controls(out, j.control_n, cfg)
                record["controls"][j.reader] = ctl
                record["jobs"].append({"kind": "cells", "reader": j.reader, "dir": out.relative_to(cfg.results_dir).as_posix(),
                                       "files_sha256": {f"{arm}.jsonl": sha256_file_bytes(out / f"{arm}.jsonl") for arm in ARMS + CONTROL_ARMS},
                                       "done": True})
                if not ctl["floor_passed"]:
                    raise _refuse(f"float16 floor failed: {j.reader} answer-flip rate {ctl['floor_flip_rate']:.3f} > "
                                  f"{ctl['floor_limit']} between FRESH through float16 and FRESH in float32")
                if not ctl["identity_passed"]:
                    raise _refuse(f"identity control failed: {j.reader} generated {ctl['identity_flips']} different "
                                  "completions on a second FRESH pass; the tool does not reproduce its own generation")
            save()
    except Refusal as e:
        record["halted"] = str(e)
        save()
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return record


# --- probe ---------------------------------------------------------------------------------------------------------

def probe(cfg: TaskConfig, repo_root: Path, *, out: Path, n: int, reader: str | None = None, max_new: int | None = None,
          runner=subprocess.run, upstream_check=check_upstream, tools_dir: Path = TOOLS) -> dict:
    """The pre-registration probe: equivalence, then one cells job for `reader` (default the last reader) over the
    first n prompts with the control arms on all of them; writes OUTSIDE results/ only; decides nothing. `max_new`
    overrides the config's cap (the probe may read the cap's effect before the registration fixes it)."""
    import math
    from lag_ladder.summarize_task import flip_rate, no_answer_rate, paired_delta
    repo_root = Path(repo_root)
    out_dir = Path(out).resolve()
    results = (repo_root / "results").resolve()
    if out_dir == results or results in out_dir.parents:
        raise _refuse(f"probe --out {out_dir} is under results/; the probe is pre-registration and must not write "
                      "where a registered run records")
    reader = reader or cfg.readers[-1]
    if reader not in cfg.readers:
        raise _refuse(f"probe reader {reader!r} is not one of the config's readers {list(cfg.readers)}")
    if not (0 < n <= cfg.n_prompts):
        raise _refuse(f"probe n must be in (0, n_prompts={cfg.n_prompts}]")
    max_new = cfg.max_new if max_new is None else int(max_new)
    if max_new <= 0:
        raise _refuse(f"probe max_new must be positive, got {max_new}")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="task-probe")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for what, path in (("prompts", cfg.prompts), ("far_tokens", cfg.far_tokens),
                       ("rollout_tokens", cfg.controls["rollout_tokens"]), ("rollout_spans", cfg.controls["rollout_spans"])):
        if not Path(path).exists():
            raise _refuse(f"{what} file {path} does not exist")
    if not (Path(tools_dir) / TOOL).exists():
        raise _refuse(f"tool {TOOL} is missing under {tools_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_dir = out_dir / "equivalence"
    eq_dir.mkdir(exist_ok=True)
    _call(cfg, repo_root, ["equivalence", "--writer", cfg.writer,
                           "--dump-root", str(Path(cfg.controls["rollout_dump_root"]).resolve()),
                           "--tokens", str(Path(cfg.controls["rollout_tokens"]).resolve()),
                           "--spans", str(Path(cfg.controls["rollout_spans"]).resolve()),
                           "--reader", cfg.controls["equivalence_reader"],
                           "--expect-b1", repr(float(cfg.controls["equivalence_expect_b1"])),
                           "--out", str(eq_dir)], runner, tools_dir)
    eq_path = eq_dir / "equivalence.json"
    if not eq_path.exists():
        raise _refuse(f"{TOOL} did not write {eq_path}")
    equivalence = judge_equivalence(json.loads(eq_path.read_text(encoding="utf-8")), cfg)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    idx = list(range(n))
    idx_path = cells_dir / "indices.json"
    idx_path.write_text(json.dumps(idx), encoding="utf-8")
    _call(cfg, repo_root, ["cells", "--reader", reader, "--writer", cfg.writer,
                           "--prompts", str(Path(cfg.prompts).resolve()), "--indices", str(idx_path.resolve()),
                           "--far-tokens", str(Path(cfg.far_tokens).resolve()), "--far-seq", str(cfg.far_seq),
                           "--max-new", str(max_new), "--control-n", str(n), "--out", str(cells_dir)], runner, tools_dir)
    fresh = _by_index(_read_jsonl(cells_dir / "fresh.jsonl"))
    if sorted(fresh) != idx:
        raise _refuse(f"fresh.jsonl holds {len(fresh)} records, expected {n}")
    ctl = judge_controls(cells_dir, n, cfg)
    rec = {"schema": "task-probe-v1", "reader": reader, "writer": cfg.writer, "n": n, "max_new": max_new,
           "prompts_sha256": sha256_file_bytes(cfg.prompts), "far_tokens_sha256": sha256_file_bytes(cfg.far_tokens),
           "equivalence": equivalence, "controls": ctl,
           "acc_fresh": float(100.0 * sum(fresh[i]["correct"] for i in idx) / n), "arms": {},
           "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for arm in ("stale", "near", "far"):
        rows = _by_index(_read_jsonl(cells_dir / f"{arm}.jsonl"))
        if sorted(rows) != idx:
            raise _refuse(f"{arm}.jsonl holds {len(rows)} records, expected {n}")
        d = paired_delta([rows[i]["correct"] for i in idx], [fresh[i]["correct"] for i in idx],
                         reps=int(cfg.rule["bootstrap_reps"]), seed=cfg.seed)
        disc = d["discordant"] / n
        rec["arms"][arm] = {"delta": d, "discordant": d["discordant"], "discordance_rate": disc,
                            "flip_rate": flip_rate([rows[i]["answer"] for i in idx], [fresh[i]["answer"] for i in idx]),
                            "no_answer_rate": no_answer_rate([rows[i]["answer"] for i in idx]),
                            "cap_rate": float(sum(bool(rows[i]["reached_cap"]) for i in idx) / n),
                            # the interval half-width this discordance implies: 1.96 * sqrt(d / m) in points
                            "halfwidth_points_at_discordance": {str(m): 100.0 * 1.96 * math.sqrt(disc / m)
                                                                for m in (n, cfg.n_prompts)}}
    (out_dir / "probe.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lag_ladder.task")
    ap.add_argument("command", choices=["plan", "check", "run", "probe"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "ownrun_task.toml"))
    ap.add_argument("--out", default=None, help="probe: output directory, OUTSIDE results/")
    ap.add_argument("--n", type=int, default=64, help="probe: the first n prompts")
    ap.add_argument("--reader", default=None, help="probe: the reader spec; default the config's last reader")
    ap.add_argument("--max-new", type=int, default=None, help="probe: override the config's max_new")
    a = ap.parse_args(argv)
    cfg = load_task_config(Path(a.config), REPO_ROOT)
    try:
        if a.command == "probe":
            if not a.out:
                raise _refuse("probe needs --out (outside results/)")
            rec = probe(cfg, REPO_ROOT, out=Path(a.out), n=a.n, reader=a.reader, max_new=a.max_new)
            eq, c = rec["equivalence"], rec["controls"]
            print(f"probe {rec['reader']} n {rec['n']} max_new {rec['max_new']}: equivalence {'ok' if eq['passed'] else 'FAILED'} "
                  f"(max_abs_diff {eq['max_abs_diff']:.3e}, max_rel_diff {eq['max_rel_diff']:.3e}, b1 rel gap {eq['b1_rel_gap']}); "
                  f"floor flip {c['floor_flip_rate']:.3f}; identity flips {c['identity_flips']}; acc_fresh {rec['acc_fresh']:.1f}")
            for arm, x in rec["arms"].items():
                d = x["delta"]
                hw = x["halfwidth_points_at_discordance"][str(cfg.n_prompts)]
                print(f"  {arm:6s} delta {d['delta_points']:+6.2f} [{d['ci_low']:+6.2f}, {d['ci_high']:+6.2f}]  "
                      f"discordance {x['discordance_rate']:.3f}  flips {x['flip_rate']:.3f}  no-answer {x['no_answer_rate']:.3f}  "
                      f"cap {x['cap_rate']:.3f}  halfwidth at n={cfg.n_prompts}: {hw:.2f} points")
            print(f"wrote {Path(a.out) / 'probe.json'}")
            return 0
        if a.command == "plan":
            for j in plan(cfg).jobs:
                print(f"{j.kind:12s} reader {j.reader:16s} writer {j.writer}  prompts {len(j.indices)}  controls {j.control_n}")
            return 0
        if a.command == "check":
            assert_ready(cfg, REPO_ROOT)
            print("task gate ok")
            return 0
        rec = run(cfg, REPO_ROOT)
        print(f"finished: {len(rec['jobs'])} jobs; equivalence {rec['equivalence']}")
        return 0
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
