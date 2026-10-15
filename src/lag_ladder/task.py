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
is the gate alone; `plan` prints the jobs and decides nothing.
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
        if r.get("schema") != "task-record-v1":
            raise _refuse(f"{path} holds a record that is not task-record-v1")
    return rows


def _by_index(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        if r["index"] in out:
            raise _refuse(f"index {r['index']} recorded twice")
        out[int(r["index"])] = r
    return out


def judge_equivalence(rec: dict, cfg: TaskConfig) -> dict:
    lim = float(cfg.controls["equivalence_max_abs_diff"])
    diff = float(rec["max_abs_diff"])
    b1 = rec.get("b1") or {}
    rel = b1.get("rel_gap")
    rel_lim = float(cfg.controls["equivalence_b1_max_rel"])
    ok = diff <= lim and rel is not None and float(rel) <= rel_lim
    return {"max_abs_diff": diff, "limit": lim, "b1_median": b1.get("median"), "b1_expected": b1.get("expected"),
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
                _call(cfg, repo_root, ["equivalence", "--writer", j.writer, "--dump-root", str(Path(cfg.controls["rollout_dump_root"]).resolve()),
                                       "--tokens", str(Path(cfg.controls["rollout_tokens"]).resolve()),
                                       "--spans", str(Path(cfg.controls["rollout_spans"]).resolve()),
                                       "--reader", j.reader, "--expect-b1", repr(float(cfg.controls["equivalence_expect_b1"])),
                                       "--out", str(out.resolve())], runner, tools_dir)
                eq_path = out / "equivalence.json"
                if not eq_path.exists():
                    raise _refuse(f"{TOOL} did not write {eq_path}")
                eq = json.loads(eq_path.read_text(encoding="utf-8"))
                record["equivalence"] = judge_equivalence(eq, cfg)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lag_ladder.task")
    ap.add_argument("command", choices=["plan", "check", "run"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "ownrun_task.toml"))
    a = ap.parse_args(argv)
    cfg = load_task_config(Path(a.config), REPO_ROOT)
    try:
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
