"""The reduced statistic (C) on the PipelineRL TIR arm (design 2026-11-09, reduced (C); decisions 3-5): the reader at
step g replays the reserved GSM8K test prompts as greedy multi-turn tool episodes under its own prompt cache (FRESH), the
step-0 writer's cache of the rendered prompt (STALE-prompt) and the writer's cache of the system turn with the tool
schema alone (STALE-shared); `tools/task_swap_tir.py` does the episodes, run under the pinned instrument's interpreter by
subprocess with the sandbox endpoint passed through the environment; nothing from the instrument or from PipelineRL is
imported, and this driver never imports the tool. `task.py` (0025-0035's single-turn driver) is byte-unchanged.

Plan: the driver fidelity gate first (the tool's rendering of every turn of the 50 actor-mode episodes against upstream's
own prompt tokens; a mismatch halts before any cell), then one `cells` job per reader over every reserved prompt, which
writes the three arms and, on the first `n_prompts` prompts, the floor arm (FRESH from an unrounded cache; RC3: all of
them) and, on the first `controls.n_prompts`, the identity arm (FRESH again) and the positive controls NEAR and FAR
(RC3: 400). After each cells job: the floor's answer-flip rate against FRESH over its limit halts; an identity flip
halts. NEAR and FAR are judged by the summarizer (the reader is UNEVALUABLE when either flips fewer than
control_min_flip_rate); no number is decided here beyond the halting thresholds. `plan` prints the per-reader episode
count decision 4's timing rule multiplies by the measured seconds per episode; the comparison against the ceiling is the
operator's, since the ceiling is set after the smoke.

`run` refuses while the config is unregistered, uncommitted, unpinned, missing its ledger entry or unsealed, while
`content_eos` is still the placeholder, while the reserved prompts are not exactly the full set minus the frozen draw,
and unless `SANDBOX_ENDPOINT` is set to a loopback address. `check` is the gate alone; `plan` decides nothing. `probe`
is the pre-registration probe: fidelity, then one reader on the first n reserved prompts with the floor and the controls
on all n; it needs no registration, pin or seal — only the upstream gate — refuses to write under results/, and writes
probe.json with the numbers the registering entry fixes.
"""
import argparse
import ipaddress
import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

from lag_ladder import REPO_ROOT
from lag_ladder.config import load_seal_config, _hex_or_pending, _local_spec, _rel_under_upstream
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.reader import UPSTREAM_PATHS, DISTANCE_RESULTS, checkpoints_dir, reader_slug, upstream_python, _committed_as_is
from lag_ladder.seal import require_sealed
from lag_ladder.upstream_gate import check_upstream

TOOL = "task_swap_tir.py"
RC_BUDGET = 4             # the tool's exit code on a decision-4 budget halt (tools/task_swap_tir.py RC_BUDGET; not imported)
TOOLS = REPO_ROOT / "tools"
RUN_RECORD = "run.json"
RECORD_SCHEMA = "tir-record-v1"
FIDELITY_SCHEMA = "tir-fidelity-v1"
ARMS = ("fresh", "stale_prompt", "stale_shared")
FLOOR_ARM = "fresh32"
CONTROL_ARMS = ("near", "far", "fresh2")
ALL_ARMS = ARMS + (FLOOR_ARM,) + CONTROL_ARMS
CHECKPOINT_ROOT_ENV = "KVT_CHECKPOINT_ROOT"
SANDBOX_ENV = "SANDBOX_ENDPOINT"
CONTENT_EOS = ("keep", "strip")
CONTENT_EOS_PENDING = "CONTENT_EOS_PENDING"


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"task_tir REFUSED: {msg}")


# --- config --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskTirConfig:
    results_dir: Path
    upstream_path: Path
    upstream_sha: str
    registered_by: str
    seal: str
    writer: str                   # local:pltir-step_0
    readers: tuple[str, ...]      # local:pltir-step_<g>
    prompts_all: Path             # every GSM8K test row, [{index, question, gold}] (0035's file)
    prompts_all_sha256: str
    frozen: Path                  # {"seed","n","total","indices"} from the stream tool
    frozen_sha256: str
    prompts: Path                 # the reserved rows, the full set minus the frozen draw, in file order
    prompts_sha256: str
    n_prompts: int
    far_tokens: Path
    far_tokens_sha256: str
    far_seq: int
    max_new: int
    max_turns: int
    max_model_len: int
    min_generation_tokens: int
    system_prompt: Path           # under the repository
    system_prompt_sha256: str
    content_eos: str              # keep | strip | CONTENT_EOS_PENDING
    seed: int
    rule: dict
    controls: dict
    config_path: Path


def _rel_under_repo(s, where: str, root: Path) -> Path:
    if (not isinstance(s, str) or not s.strip() or PurePosixPath(s).is_absolute() or PureWindowsPath(s).is_absolute()
            or ".." in PurePosixPath(s).parts):
        raise ValueError(f"{where} must be a relative path under the repository")
    return root / s


def load_task_tir_config(path: Path, repo_root: Path) -> TaskTirConfig:
    path = Path(path)
    with open(path, "rb") as f:
        c = tomllib.load(f)["task_tir"]
    for key in ("results_dir", "upstream_path", "upstream_sha", "writer", "readers", "prompts_all", "prompts_all_sha256",
                "frozen", "frozen_sha256", "prompts", "prompts_sha256", "n_prompts", "far_tokens", "far_tokens_sha256",
                "far_seq", "max_new", "max_turns", "max_model_len", "min_generation_tokens", "system_prompt",
                "system_prompt_sha256", "content_eos", "seed", "rule", "controls"):
        if key not in c:
            raise ValueError(f"{path.name} [task_tir] is missing {key}")
    root = Path(repo_root)
    n = path.name
    upstream = (root / c["upstream_path"]).resolve()
    writer = _local_spec(c["writer"], f"{n} [task_tir] writer")
    readers = c["readers"]
    if not isinstance(readers, list) or not readers or len(set(readers)) != len(readers):
        raise ValueError(f"{n} [task_tir] readers must be a non-empty list of distinct specs")
    readers = tuple(_local_spec(r, f"{n} [task_tir] readers") for r in readers)
    if writer in readers:
        raise ValueError(f"{n} [task_tir] writer {writer} is also a reader; FRESH vs STALE would be identity")
    np_ = int(c["n_prompts"])
    if np_ <= 0:
        raise ValueError(f"{n} [task_tir] n_prompts must be positive")
    max_new, max_turns, mml, ming = int(c["max_new"]), int(c["max_turns"]), int(c["max_model_len"]), int(c["min_generation_tokens"])
    if max_new <= 0 or int(c["far_seq"]) < 0:
        raise ValueError(f"{n} [task_tir] max_new must be positive and far_seq non-negative")
    if max_turns <= 0:
        raise ValueError(f"{n} [task_tir] max_turns must be positive")
    if not (0 < ming <= max_new < mml):
        raise ValueError(f"{n} [task_tir] needs 0 < min_generation_tokens <= max_new < max_model_len")
    eos = str(c["content_eos"])
    if eos not in CONTENT_EOS and eos != CONTENT_EOS_PENDING:
        raise ValueError(f"{n} [task_tir] content_eos must be keep, strip or {CONTENT_EOS_PENDING}")
    rule = dict(c["rule"])
    for k in ("bootstrap_reps", "control_min_flip_rate"):
        if k not in rule:
            raise ValueError(f"{n} [task_tir.rule] is missing {k}")
    if int(rule["bootstrap_reps"]) < 100:
        raise ValueError(f"{n} [task_tir.rule] bootstrap_reps must be at least 100")
    if not (0 < float(rule["control_min_flip_rate"]) <= 1):
        raise ValueError(f"{n} [task_tir.rule] control_min_flip_rate must be in (0, 1]")
    controls = dict(c["controls"])
    for k in ("n_prompts", "floor_max_flip_rate", "fidelity_episodes", "fidelity_episodes_sha256"):
        if k not in controls:
            raise ValueError(f"{n} [task_tir.controls] is missing {k}")
    if not (0 < int(controls["n_prompts"]) <= np_):
        raise ValueError(f"{n} [task_tir.controls] n_prompts must be in (0, task_tir.n_prompts]")
    if not (0 <= float(controls["floor_max_flip_rate"]) < 1):
        raise ValueError(f"{n} [task_tir.controls] floor_max_flip_rate must be in [0, 1)")
    controls["fidelity_episodes"] = _rel_under_upstream(controls["fidelity_episodes"], f"{n} [task_tir.controls] fidelity_episodes", upstream)
    controls["fidelity_episodes_sha256"] = _hex_or_pending(controls["fidelity_episodes_sha256"], f"{n} fidelity_episodes_sha256")
    return TaskTirConfig(
        results_dir=root / c["results_dir"], upstream_path=upstream, upstream_sha=str(c["upstream_sha"]),
        registered_by=str(c.get("registered_by", "")), seal=str(c.get("seal", "pipelinerl-tir-task")),
        writer=writer, readers=readers,
        prompts_all=_rel_under_upstream(c["prompts_all"], f"{n} [task_tir] prompts_all", upstream),
        prompts_all_sha256=_hex_or_pending(c["prompts_all_sha256"], f"{n} prompts_all_sha256"),
        frozen=_rel_under_upstream(c["frozen"], f"{n} [task_tir] frozen", upstream),
        frozen_sha256=_hex_or_pending(c["frozen_sha256"], f"{n} frozen_sha256"),
        prompts=_rel_under_upstream(c["prompts"], f"{n} [task_tir] prompts", upstream),
        prompts_sha256=_hex_or_pending(c["prompts_sha256"], f"{n} prompts_sha256"), n_prompts=np_,
        far_tokens=_rel_under_upstream(c["far_tokens"], f"{n} [task_tir] far_tokens", upstream),
        far_tokens_sha256=_hex_or_pending(c["far_tokens_sha256"], f"{n} far_tokens_sha256"), far_seq=int(c["far_seq"]),
        max_new=max_new, max_turns=max_turns, max_model_len=mml, min_generation_tokens=ming,
        system_prompt=_rel_under_repo(c["system_prompt"], f"{n} [task_tir] system_prompt", root),
        system_prompt_sha256=_hex_or_pending(c["system_prompt_sha256"], f"{n} system_prompt_sha256"),
        content_eos=eos, seed=int(c["seed"]), rule=rule, controls=controls, config_path=path)


# --- plan ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    kind: str                 # fidelity | cells
    reader: str
    writer: str
    indices: list[int] = field(default_factory=list)
    floor_n: int = 0
    control_n: int = 0


@dataclass(frozen=True)
class Plan:
    jobs: tuple[Job, ...]


def reserved_indices(cfg: TaskTirConfig) -> list[int]:
    """The reserved prompts' GSM8K indices in file order (the tool keys its records by this index)."""
    rows = json.loads(Path(cfg.prompts).read_text(encoding="utf-8"))
    return [int(p["index"]) for p in rows]


def select_readers(cfg: TaskTirConfig, readers: list[str] | None) -> list[str]:
    if readers is None:
        return list(cfg.readers)
    if not readers:
        raise _refuse("the reader filter is empty; omit it to run every reader")
    for r in readers:
        if readers.count(r) > 1:
            raise _refuse(f"the reader filter names {r!r} twice")
        if r not in cfg.readers:
            raise _refuse(f"filter reader {r!r} is not one of the config's readers {list(cfg.readers)}")
    return [r for r in cfg.readers if r in readers]


def plan(cfg: TaskTirConfig, readers: list[str] | None = None) -> Plan:
    """Fidelity (the writer's tokenizer renders the 50 episodes), then the cells jobs of the named readers in the
    config's order; None means every reader. The floor runs on every prompt, the controls on the first
    controls.n_prompts (RC3)."""
    jobs = [Job("fidelity", cfg.writer, cfg.writer)]
    idx = reserved_indices(cfg) if Path(cfg.prompts).exists() else list(range(cfg.n_prompts))
    for r in select_readers(cfg, readers):
        jobs.append(Job("cells", r, cfg.writer, idx, len(idx), int(cfg.controls["n_prompts"])))
    return Plan(tuple(jobs))


def projection(cfg: TaskTirConfig) -> dict:
    """The per-reader episode count decision 4's timing rule multiplies by the measured seconds per episode."""
    p, c = cfg.n_prompts, int(cfg.controls["n_prompts"])
    return {"prompts": p, "arms_per_prompt": len(ALL_ARMS), "episodes_per_reader": p * len(ARMS) + p + c * len(CONTROL_ARMS)}


def job_dir(cfg: TaskTirConfig, j: Job) -> Path:
    if j.kind == "fidelity":
        return cfg.results_dir / "fidelity"
    return cfg.results_dir / "readers" / reader_slug(j.reader)


# --- the gate ----------------------------------------------------------------------------------------------------

def sandbox_endpoint(env) -> str:
    ep = env.get(SANDBOX_ENV, "")
    if not ep:
        raise _refuse(f"{SANDBOX_ENV} is not set; the sandbox must be up on loopback before any cell (decision 4)")
    host = urllib.parse.urlsplit(ep).hostname
    try:
        ok = host is not None and (host == "localhost" or ipaddress.ip_address(host).is_loopback)
    except ValueError:
        ok = False
    if not ok:
        raise _refuse(f"{SANDBOX_ENV} must be a loopback address (127.0.0.1 or localhost), got host {host!r}")
    return ep


def check_partition(cfg: TaskTirConfig) -> list[int]:
    """The reserved file is exactly the full prompt file minus the frozen draw, in the full file's order."""
    allp = json.loads(Path(cfg.prompts_all).read_text(encoding="utf-8"))
    fz = json.loads(Path(cfg.frozen).read_text(encoding="utf-8"))
    res = json.loads(Path(cfg.prompts).read_text(encoding="utf-8"))
    frozen = {int(i) for i in fz["indices"]}
    if int(fz.get("total", -1)) != len(allp) or len(frozen) != int(fz.get("n", -1)):
        raise _refuse(f"the frozen file names total {fz.get('total')} and n {fz.get('n')}; the prompt file holds {len(allp)} rows")
    key = lambda p: (int(p["index"]), str(p["question"]), str(p["gold"]))
    expected = [key(p) for p in allp if int(p["index"]) not in frozen]
    if [key(p) for p in res] != expected:
        raise _refuse(f"the reserved prompts are not the partition (full set minus the frozen draw): {len(res)} rows against {len(expected)} expected")
    if len(res) != cfg.n_prompts:
        raise _refuse(f"the reserved file holds {len(res)} rows; n_prompts is {cfg.n_prompts}")
    return [int(p["index"]) for p in res]


def assert_ready(cfg: TaskTirConfig, repo_root: Path, *, seal_cfg=None, upstream_check=check_upstream,
                 tools_dir: Path = TOOLS, env=os.environ) -> None:
    repo_root = Path(repo_root)
    cname = cfg.config_path.name
    if not cfg.registered_by:
        raise _refuse(f"config/{cname} is UNREGISTERED (registered_by is empty); a numbered ledger entry must fix "
                      "its values before anything runs")
    if cfg.content_eos not in CONTENT_EOS:
        raise _refuse(f"config/{cname} still carries content_eos = {cfg.content_eos}; the fidelity run off the pod pins keep or strip")
    for name, sha in (("prompts_all_sha256", cfg.prompts_all_sha256), ("frozen_sha256", cfg.frozen_sha256),
                      ("prompts_sha256", cfg.prompts_sha256), ("far_tokens_sha256", cfg.far_tokens_sha256),
                      ("system_prompt_sha256", cfg.system_prompt_sha256),
                      ("fidelity_episodes_sha256", cfg.controls["fidelity_episodes_sha256"])):
        if sha.endswith("_PENDING"):
            raise _refuse(f"config/{cname} still carries the {name} placeholder")
    for rel in ("ledger/ledger.md", cfg.config_path, cfg.system_prompt):
        p = repo_root / rel if isinstance(rel, str) else rel
        if not _committed_as_is(p, repo_root):
            raise _refuse(f"{p.name} is not committed as-is; entry {cfg.registered_by} and config/{cname} must be "
                          "committed before any score")
    committed = subprocess.run(["git", "show", "HEAD:ledger/ledger.md"], cwd=repo_root,
                               capture_output=True, text=True, encoding="utf-8")
    if committed.returncode != 0 or f"### {cfg.registered_by} " not in committed.stdout:
        raise _refuse(f"the committed ledger has no entry {cfg.registered_by}, which config/{cname} names")
    # a number alone is not a registration: the same number can head another entry (a ledger that grew under a rebase),
    # so the entry registered_by names must itself name this config
    entry = committed.stdout.split(f"### {cfg.registered_by} ", 1)[1].split("\n### ", 1)[0]
    if f"config/{cname}" not in entry:
        raise _refuse(f"the committed ledger's entry {cfg.registered_by} does not name config/{cname}; registered_by must be "
                      "the entry that registers this config")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="task_tir")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for what, path, sha in (("prompts_all", cfg.prompts_all, cfg.prompts_all_sha256), ("frozen", cfg.frozen, cfg.frozen_sha256),
                            ("prompts", cfg.prompts, cfg.prompts_sha256), ("far_tokens", cfg.far_tokens, cfg.far_tokens_sha256),
                            ("system_prompt", cfg.system_prompt, cfg.system_prompt_sha256),
                            ("fidelity_episodes", cfg.controls["fidelity_episodes"], cfg.controls["fidelity_episodes_sha256"])):
        if not Path(path).exists():
            raise _refuse(f"{what} file {path} does not exist")
        got = sha256_file_bytes(path)
        if got != sha:
            raise _refuse(f"{what} file hashes to {got[:12]}, not the registered {sha[:12]}")
    check_partition(cfg)
    if not (Path(tools_dir) / TOOL).exists():
        raise _refuse(f"tool {TOOL} is missing under {tools_dir}")
    for spec in (cfg.writer, *cfg.readers):
        if not (checkpoints_dir(repo_root) / spec[len("local:"):] / "model.safetensors").exists():
            raise _refuse(f"checkpoint for {spec} is not under {DISTANCE_RESULTS}/checkpoints")
    sandbox_endpoint(env)
    if seal_cfg is None:
        seal_cfg = load_seal_config(repo_root / "config" / "seal.toml", repo_root)
    try:
        require_sealed(cfg.seal, seal_cfg, repo_root=repo_root)
    except RuntimeError as e:
        raise _refuse(f"no committed seal for {cfg.seal}: {e}") from e
    upstream_python(cfg.upstream_path)


# --- the tool, by subprocess ---------------------------------------------------------------------------------------

def _call(cfg: TaskTirConfig, repo_root: Path, args: list[str], runner, tools_dir: Path, env, *, ok_codes=(0,)) -> int:
    cmd = [str(upstream_python(cfg.upstream_path)), str((Path(tools_dir) / TOOL).resolve()), *args]
    run_env = {**os.environ, **dict(env), CHECKPOINT_ROOT_ENV: str(checkpoints_dir(repo_root).resolve())}
    r = runner(cmd, cwd=str(cfg.upstream_path), capture_output=True, env=run_env)
    if r.returncode not in ok_codes:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"{TOOL} failed on {args[0]} (exit {r.returncode}):\n{err[-2000:]}")
    return int(r.returncode)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise _refuse(f"{TOOL} did not write {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    for r in rows:
        if r.get("schema") != RECORD_SCHEMA:
            raise _refuse(f"{path} holds a record that is not {RECORD_SCHEMA}")
    return rows


def _by_index(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        if r["index"] in out:
            raise _refuse(f"index {r['index']} recorded twice")
        out[int(r["index"])] = r
    return out


def judge_fidelity(rep: dict) -> dict:
    if rep.get("schema") != FIDELITY_SCHEMA:
        raise _refuse(f"fidelity.json is not {FIDELITY_SCHEMA}")
    return {"n_episodes": int(rep["n_episodes"]), "n_turns": int(rep["n_turns"]), "turns_matched": int(rep["turns_matched"]),
            "turns_mismatched": int(rep["turns_mismatched"]), "turns_extend_cleanly": int(rep["turns_extend_cleanly"]),
            "content_eos": rep["content_eos"], "passed": bool(rep["passed"]) and int(rep["turns_mismatched"]) == 0 and int(rep["n_turns"]) > 0}


def judge_controls(out: Path, indices: list[int], floor_n: int, control_n: int, cfg: TaskTirConfig) -> dict:
    fresh = _by_index(_read_jsonl(out / "fresh.jsonl"))
    f32 = _by_index(_read_jsonl(out / f"{FLOOR_ARM}.jsonl"))
    f2 = _by_index(_read_jsonl(out / "fresh2.jsonl"))
    if sorted(f32) != sorted(indices[:floor_n]):
        raise _refuse(f"{out}: the floor arm holds {len(f32)} records, expected the first {floor_n} prompts")
    if sorted(f2) != sorted(indices[:control_n]):
        raise _refuse(f"{out}: the identity arm holds {len(f2)} records, expected the first {control_n} prompts")
    flips = sum(f32[i]["answer"] != fresh[i]["answer"] for i in f32)
    ident = sum(f2[i]["completion_sha256"] != fresh[i]["completion_sha256"] for i in f2)
    floor_rate = flips / floor_n
    lim = float(cfg.controls["floor_max_flip_rate"])
    return {"floor_n": floor_n, "floor_flip_rate": floor_rate, "floor_limit": lim, "floor_passed": floor_rate <= lim,
            "control_n": control_n, "identity_flips": ident, "identity_passed": ident == 0}


# --- run -----------------------------------------------------------------------------------------------------------

def _fidelity_args(cfg: TaskTirConfig, out: Path) -> list[str]:
    return ["fidelity", "--episodes", str(Path(cfg.controls["fidelity_episodes"]).resolve()), "--tokenizer", cfg.writer,
            "--system-prompt", str(Path(cfg.system_prompt).resolve()), "--content-eos", cfg.content_eos,
            "--max-model-len", str(cfg.max_model_len), "--out", str(out.resolve())]


def _cells_args(cfg: TaskTirConfig, j: Job, idx_path: Path, out: Path, *, budget_seconds: float | None = None,
                later_episodes: int = 0) -> list[str]:
    args = ["cells", "--reader", j.reader, "--writer", j.writer,
            "--prompts", str(Path(cfg.prompts).resolve()), "--indices", str(idx_path.resolve()),
            "--far-tokens", str(Path(cfg.far_tokens).resolve()), "--far-seq", str(cfg.far_seq),
            "--max-new", str(cfg.max_new), "--max-turns", str(cfg.max_turns), "--max-model-len", str(cfg.max_model_len),
            "--min-generation", str(cfg.min_generation_tokens), "--floor-n", str(j.floor_n), "--control-n", str(j.control_n),
            "--system-prompt", str(Path(cfg.system_prompt).resolve()), "--content-eos", cfg.content_eos,
            "--out", str(out.resolve()), "--later-episodes", str(int(later_episodes))]
    if budget_seconds is not None:
        args += ["--budget-seconds", f"{budget_seconds:.0f}"]
    return args


def episodes_of(j: Job) -> int:
    """The episodes a cells job runs: every prompt on the three arms, the floor on the first floor_n, the three controls
    on the first control_n (decision 4's projection unit)."""
    return len(j.indices) * len(ARMS) + j.floor_n + j.control_n * len(CONTROL_ARMS)


def run(cfg: TaskTirConfig, repo_root: Path, *, runner=subprocess.run, seal_cfg=None, upstream_check=check_upstream,
        tools_dir: Path = TOOLS, readers: list[str] | None = None, env=os.environ, budget_seconds: float | None = None) -> dict:
    """Decision 4's budget: `budget_seconds` is the wall time the cells may still use (the operator computes it from the
    ceiling and the hourly rate outside the repo); each cells job gets what is left of it and the later shards' episode
    counts, and the run halts for a ruling when the tool exits RC_BUDGET. None: the tool prints its projection only."""
    run_t0 = time.monotonic()
    if budget_seconds is not None and budget_seconds <= 0:
        raise _refuse("the budget must be a positive number of seconds")
    run_readers = select_readers(cfg, readers)                     # a bad filter refuses before the gate or any job
    assert_ready(cfg, repo_root, seal_cfg=seal_cfg, upstream_check=upstream_check, tools_dir=tools_dir, env=env)
    p = plan(cfg, readers)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cfg.results_dir / RUN_RECORD
    record = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "seed": cfg.seed,
              "writer": cfg.writer, "readers": list(cfg.readers), "readers_run": run_readers,
              "n_prompts": cfg.n_prompts, "max_new": cfg.max_new, "max_turns": cfg.max_turns, "max_model_len": cfg.max_model_len,
              "min_generation_tokens": cfg.min_generation_tokens, "content_eos": cfg.content_eos,
              "config_sha256": sha256_file_bytes(cfg.config_path), "prompts_sha256": cfg.prompts_sha256,
              "prompts_all_sha256": cfg.prompts_all_sha256, "frozen_sha256": cfg.frozen_sha256,
              "far_tokens_sha256": cfg.far_tokens_sha256, "system_prompt_sha256": cfg.system_prompt_sha256,
              "fidelity_episodes_sha256": cfg.controls["fidelity_episodes_sha256"],
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "finished": None,
              "budget_seconds": budget_seconds, "halted": None, "fidelity": None, "controls": {}, "budget": {}, "jobs": []}
    sandbox = {SANDBOX_ENV: sandbox_endpoint(env)}

    def save():
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    try:
        for k, j in enumerate(p.jobs):
            out = job_dir(cfg, j)
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            if j.kind == "fidelity":
                _call(cfg, repo_root, _fidelity_args(cfg, out), runner, tools_dir, sandbox, ok_codes=(0, 3))
                fid_path = out / "fidelity.json"
                if not fid_path.exists():
                    raise _refuse(f"{TOOL} did not write {fid_path}")
                record["fidelity"] = judge_fidelity(json.loads(fid_path.read_text(encoding="utf-8")))
                record["jobs"].append({"kind": "fidelity", "reader": j.reader, "dir": out.relative_to(cfg.results_dir).as_posix(),
                                       "file_sha256": sha256_file_bytes(fid_path), "seconds": time.monotonic() - t0, "done": True})
                if not record["fidelity"]["passed"]:
                    raise _refuse(f"driver fidelity failed: {record['fidelity']['turns_mismatched']} of {record['fidelity']['n_turns']} "
                                  f"turns render differently from upstream's prompt tokens (content_eos {cfg.content_eos}); no cell runs")
            else:
                idx_path = out / "indices.json"
                idx_path.write_text(json.dumps(j.indices), encoding="utf-8")
                later = sum(episodes_of(jj) for jj in p.jobs[k + 1:] if jj.kind == "cells")
                remaining = None if budget_seconds is None else max(1.0, budget_seconds - (time.monotonic() - run_t0))
                rc = _call(cfg, repo_root, _cells_args(cfg, j, idx_path, out, budget_seconds=remaining, later_episodes=later),
                           runner, tools_dir, sandbox, ok_codes=(0, RC_BUDGET))
                budget_path = out / "budget.json"
                if budget_path.exists():
                    # this call's projection only: the tool overwrites the file with checked:false at its start, so an
                    # earlier sitting's report (possibly over) is never recorded as this run's
                    b = json.loads(budget_path.read_text(encoding="utf-8"))
                    if b.get("checked") is True:
                        record["budget"][j.reader] = b
                if rc == RC_BUDGET:
                    b = record["budget"].get(j.reader) or {}
                    raise _refuse(f"budget: {j.reader} projects {float(b.get('projected_s', float('nan'))):.0f}s of cells against the "
                                  f"{float(b.get('budget_s', float('nan'))):.0f}s remaining (decision 4); halted for a ruling, "
                                  "resume with run after it (done prompts are skipped)")
                for arm in ARMS:
                    rows = _by_index(_read_jsonl(out / f"{arm}.jsonl"))
                    if sorted(rows) != sorted(j.indices):
                        raise _refuse(f"{out / (arm + '.jsonl')} holds {len(rows)} records, expected {len(j.indices)}")
                for arm in ("near", "far"):
                    rows = _by_index(_read_jsonl(out / f"{arm}.jsonl"))
                    if sorted(rows) != sorted(j.indices[:j.control_n]):
                        raise _refuse(f"{out / (arm + '.jsonl')} holds {len(rows)} records, expected the first {j.control_n} prompts")
                ctl = judge_controls(out, j.indices, j.floor_n, j.control_n, cfg)
                record["controls"][j.reader] = ctl
                record["jobs"].append({"kind": "cells", "reader": j.reader, "dir": out.relative_to(cfg.results_dir).as_posix(),
                                       "files_sha256": {f"{arm}.jsonl": sha256_file_bytes(out / f"{arm}.jsonl") for arm in ALL_ARMS},
                                       "seconds": time.monotonic() - t0, "done": True})
                if not ctl["floor_passed"]:
                    raise _refuse(f"float16 floor failed: {j.reader} answer-flip rate {ctl['floor_flip_rate']:.3f} > "
                                  f"{ctl['floor_limit']} between FRESH through float16 and FRESH in float32")
                if not ctl["identity_passed"]:
                    raise _refuse(f"identity control failed: {j.reader} generated {ctl['identity_flips']} different "
                                  "episodes on a second FRESH pass; the tool does not reproduce its own generation")
            save()
    except Refusal as e:
        record["halted"] = str(e)
        save()
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    return record


# --- probe ---------------------------------------------------------------------------------------------------------

def probe(cfg: TaskTirConfig, repo_root: Path, *, out: Path, n: int, reader: str | None = None, runner=subprocess.run,
          upstream_check=check_upstream, tools_dir: Path = TOOLS, env=os.environ) -> dict:
    """The pre-registration probe: fidelity, then one cells job for `reader` (default the last reader) over the first n
    reserved prompts with the floor and the controls on all of them; writes OUTSIDE results/ only; decides nothing."""
    from lag_ladder.summarize_task import flip_rate, no_answer_rate
    repo_root = Path(repo_root)
    out_dir = Path(out).resolve()
    results = (repo_root / "results").resolve()
    if out_dir == results or results in out_dir.parents:
        raise _refuse(f"probe --out {out_dir} is under results/; the probe is pre-registration and must not write "
                      "where a registered run records")
    reader = reader or cfg.readers[-1]
    if reader not in cfg.readers:
        raise _refuse(f"probe reader {reader!r} is not one of the config's readers {list(cfg.readers)}")
    if cfg.content_eos not in CONTENT_EOS:
        raise _refuse(f"probe needs content_eos pinned to keep or strip (run the fidelity gate with both first)")
    try:
        upstream_check(cfg.upstream_path, cfg.upstream_sha, UPSTREAM_PATHS, who="task_tir-probe")
    except RuntimeError as e:
        raise Refusal(str(e)) from e
    for what, path in (("prompts", cfg.prompts), ("far_tokens", cfg.far_tokens), ("system_prompt", cfg.system_prompt),
                       ("fidelity_episodes", cfg.controls["fidelity_episodes"])):
        if not Path(path).exists():
            raise _refuse(f"{what} file {path} does not exist")
    if not (Path(tools_dir) / TOOL).exists():
        raise _refuse(f"tool {TOOL} is missing under {tools_dir}")
    sandbox = {SANDBOX_ENV: sandbox_endpoint(env)}
    idx_all = reserved_indices(cfg)
    if not (0 < n <= len(idx_all)):
        raise _refuse(f"probe n must be in (0, {len(idx_all)}]")
    out_dir.mkdir(parents=True, exist_ok=True)
    fid_dir = out_dir / "fidelity"
    fid_dir.mkdir(exist_ok=True)
    _call(cfg, repo_root, _fidelity_args(cfg, fid_dir), runner, tools_dir, sandbox, ok_codes=(0, 3))
    fid_path = fid_dir / "fidelity.json"
    if not fid_path.exists():
        raise _refuse(f"{TOOL} did not write {fid_path}")
    fidelity = judge_fidelity(json.loads(fid_path.read_text(encoding="utf-8")))
    if not fidelity["passed"]:
        raise _refuse(f"driver fidelity failed in the probe: {fidelity['turns_mismatched']} of {fidelity['n_turns']} turns differ")
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    idx = idx_all[:n]
    idx_path = cells_dir / "indices.json"
    idx_path.write_text(json.dumps(idx), encoding="utf-8")
    j = Job("cells", reader, cfg.writer, idx, n, n)
    t0 = time.monotonic()
    _call(cfg, repo_root, _cells_args(cfg, j, idx_path, cells_dir), runner, tools_dir, sandbox)
    seconds = time.monotonic() - t0
    fresh = _by_index(_read_jsonl(cells_dir / "fresh.jsonl"))
    if sorted(fresh) != sorted(idx):
        raise _refuse(f"fresh.jsonl holds {len(fresh)} records, expected {n}")
    ctl = judge_controls(cells_dir, idx, n, n, cfg)
    rec = {"schema": "tir-probe-v1", "reader": reader, "writer": cfg.writer, "n": n, "indices": idx,
           "max_new": cfg.max_new, "max_turns": cfg.max_turns, "content_eos": cfg.content_eos,
           "prompts_sha256": sha256_file_bytes(cfg.prompts), "far_tokens_sha256": sha256_file_bytes(cfg.far_tokens),
           "fidelity": fidelity, "controls": ctl, "seconds": seconds,
           "seconds_per_episode": seconds / (n * len(ALL_ARMS)),
           "acc_fresh": float(100.0 * sum(fresh[i]["correct"] for i in idx) / n),
           "math_answer_rate_fresh": float(sum(fresh[i]["answered_by"] == "MathAnswer" for i in idx) / n),
           "mean_turns_fresh": float(sum(fresh[i]["n_turns"] for i in idx) / n), "arms": {},
           "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for arm in ("stale_prompt", "stale_shared", "near", "far"):
        rows = _by_index(_read_jsonl(cells_dir / f"{arm}.jsonl"))
        if sorted(rows) != sorted(idx):
            raise _refuse(f"{arm}.jsonl holds {len(rows)} records, expected {n}")
        rec["arms"][arm] = {"flip_rate": flip_rate([rows[i]["answer"] for i in idx], [fresh[i]["answer"] for i in idx]),
                            "tool_sequence_flip_rate": float(sum(rows[i]["tool_sequence_sha256"] != fresh[i]["tool_sequence_sha256"] for i in idx) / n),
                            "no_answer_rate": no_answer_rate([rows[i]["answer"] for i in idx]),
                            "math_answer_rate": float(sum(rows[i]["answered_by"] == "MathAnswer" for i in idx) / n),
                            "acc": float(100.0 * sum(rows[i]["correct"] for i in idx) / n),
                            "cap_rate": float(sum(bool(rows[i]["reached_cap"]) for i in idx) / n),
                            "stale_inside_lcp_rate": float(sum(bool(rows[i]["stale_inside_lcp"]) for i in idx) / n),
                            "mean_sandbox_hits": float(sum(rows[i]["sandbox_hits"] for i in idx) / n)}
    (out_dir / "probe.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lag_ladder.task_tir")
    ap.add_argument("command", choices=["plan", "check", "run", "probe"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "pipelinerl_tir_task.toml"))
    ap.add_argument("--out", default=None, help="probe: output directory, OUTSIDE results/")
    ap.add_argument("--n", type=int, default=20, help="probe: the first n reserved prompts")
    ap.add_argument("--reader", default=None, help="probe: the reader spec; default the config's last reader")
    ap.add_argument("--readers", default=None,
                    help="run: comma-separated reader specs to run (in the config's order) after the fidelity gate; default every reader")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="run: wall seconds the cells may still use (decision 4; computed by the operator from the ceiling, "
                         "never stored in the repo); the run halts for a ruling when a shard's projection exceeds it")
    a = ap.parse_args(argv)
    cfg = load_task_tir_config(Path(a.config), REPO_ROOT)
    try:
        if a.readers is not None and a.command != "run":
            raise _refuse("--readers applies to run only")
        if a.budget_seconds is not None and a.command != "run":
            raise _refuse("--budget-seconds applies to run only")
        readers = None if a.readers is None else [r.strip() for r in a.readers.split(",") if r.strip()]
        if a.command == "probe":
            if not a.out:
                raise _refuse("probe needs --out (outside results/)")
            rec = probe(cfg, REPO_ROOT, out=Path(a.out), n=a.n, reader=a.reader)
            c = rec["controls"]
            print(f"probe {rec['reader']} n {rec['n']}: fidelity {'ok' if rec['fidelity']['passed'] else 'FAILED'} "
                  f"(extend cleanly {rec['fidelity']['turns_extend_cleanly']}/{rec['fidelity']['n_turns']}); floor flip {c['floor_flip_rate']:.3f}; "
                  f"identity flips {c['identity_flips']}; acc_fresh {rec['acc_fresh']:.1f}; MathAnswer {rec['math_answer_rate_fresh']:.3f}; "
                  f"turns {rec['mean_turns_fresh']:.2f}; {rec['seconds_per_episode']:.1f} s/episode")
            for arm, x in rec["arms"].items():
                print(f"  {arm:13s} flips {x['flip_rate']:.3f}  tool-seq flips {x['tool_sequence_flip_rate']:.3f}  acc {x['acc']:.1f}  "
                      f"no-answer {x['no_answer_rate']:.3f}  cap {x['cap_rate']:.3f}  span-in-lcp {x['stale_inside_lcp_rate']:.3f}")
            print(f"wrote {Path(a.out) / 'probe.json'}")
            return 0
        if a.command == "plan":
            for j in plan(cfg).jobs:
                print(f"{j.kind:9s} reader {j.reader:22s} writer {j.writer}  prompts {len(j.indices)}  floor {j.floor_n}  controls {j.control_n}")
            pr = projection(cfg)
            print(f"PROJECTION prompts {pr['prompts']} arms-per-prompt {pr['arms_per_prompt']} episodes-per-reader {pr['episodes_per_reader']}")
            return 0
        if a.command == "check":
            assert_ready(cfg, REPO_ROOT)
            print("task_tir gate ok")
            return 0
        rec = run(cfg, REPO_ROOT, readers=readers, budget_seconds=a.budget_seconds)
        print(f"finished: {len(rec['jobs'])} jobs; readers run {rec['readers_run']}; fidelity {rec['fidelity']}")
        return 0
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
