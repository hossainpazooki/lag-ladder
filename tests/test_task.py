"""The statistic (C) driver (design §15): the config loader's refusals, the plan (equivalence first, then one cells
job per reader with the controls), the gate, and the run's halting rules, with a stand-in for tools/task_swap.py."""
import hashlib
import json
from pathlib import Path

import pytest

from lag_ladder import task
from lag_ladder.config import load_task_config
from lag_ladder.seal import write_prediction
from tests.conftest import commit_all, git
from tests.fake_task_swap import FakeTaskSwap
from tests.test_reader import _setup_rollout

TASK_CONFIG = '''
[task]
results_dir = "results/task"
upstream_path = "../upstream"
upstream_sha = "{sha}"
registered_by = "{reg}"
seal = "ownrun-task"
writer = "local:step_0"
readers = ["local:step_1", "local:step_32"]
prompts = "data/tokens/gsm8k_test.json"
prompts_sha256 = "{psha}"
n_prompts = 12
far_tokens = "data/tokens/qwen.npy"
far_tokens_sha256 = "{qsha}"
far_seq = 1
max_new = 8
seed = 0

[task.rule]
holds_abs_max = 1.0
holds_ci_abs_max = 2.0
degrades_max = -5.0
degrades_ci_max = -2.0
bootstrap_reps = 200

[task.controls]
n_prompts = 8
floor_max_flip_rate = 0.2
equivalence_max_abs_diff = 0.0
equivalence_b1_max_rel = 1e-3
equivalence_reader = "local:step_32"
equivalence_expect_b1 = 0.08914
rollout_tokens = "data/tokens/qwen.npy"
rollout_tokens_sha256 = "{qsha}"
rollout_spans = "data/tokens/qwen.spans.json"
rollout_spans_sha256 = "{ssha}"
rollout_dump_root = "results/distance/pairs/ownrun-rollout/dumps/source@step_200"
'''

LEDGER_0030 = "\n### 0030 — 2026-10-16 — statistic (C) registered (test)\n\nprior-entries-sha256: x\n\nregisters config/task.toml.\n"


def _setup_task(repo, seal_cfg, *, registered="0030", seal=True):
    cfg_r, tools = _setup_rollout(repo, seal_cfg)
    up = repo.parent / "upstream"
    prompts = [{"index": i, "question": f"q{i}", "gold": str(i * 2)} for i in range(12)]
    ppath = up / "data" / "tokens" / "gsm8k_test.json"
    ppath.write_text(json.dumps(prompts), encoding="utf-8")
    psha = hashlib.sha256(ppath.read_bytes()).hexdigest()
    qsha = hashlib.sha256((up / "data" / "tokens" / "qwen.npy").read_bytes()).hexdigest()
    ssha = hashlib.sha256((up / "data" / "tokens" / "qwen.spans.json").read_bytes()).hexdigest()
    for step in (0, 1, 32):
        d = repo / "results" / "distance" / "checkpoints" / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"w" * (step + 1)); (d / "config.json").write_text("{}")
    sha = git(up, "rev-parse", "HEAD").strip()
    cpath = repo / "config" / "task.toml"
    cpath.write_text(TASK_CONFIG.format(sha=sha, reg=registered, psha=psha, qsha=qsha, ssha=ssha), encoding="utf-8")
    lpath = repo / "ledger" / "ledger.md"
    lpath.write_text(lpath.read_text(encoding="utf-8") + LEDGER_0030, encoding="utf-8")
    if seal:
        write_prediction("ownrun-task", {"registering_entry": "0030", "predictions": {}}, seal_cfg)
    (tools / "task_swap.py").write_text("# stand-in\n")
    commit_all(repo, "register task")
    return load_task_config(cpath, repo), tools


def test_task_config_loads_and_refuses_bad_values(repo, seal_cfg):
    cfg, _ = _setup_task(repo, seal_cfg)
    assert cfg.registered_by == "0030" and cfg.readers == ("local:step_1", "local:step_32") and cfg.writer == "local:step_0"
    assert cfg.n_prompts == 12 and cfg.controls["n_prompts"] == 8 and cfg.rule["bootstrap_reps"] == 200
    assert cfg.prompts.name == "gsm8k_test.json" and cfg.controls["rollout_spans"].name == "qwen.spans.json"
    cpath = repo / "config" / "task.toml"
    base = cpath.read_text(encoding="utf-8")
    for old, new, msg in [('readers = ["local:step_1", "local:step_32"]', 'readers = []', "readers"),
                          ('n_prompts = 12', 'n_prompts = 0', "n_prompts"),
                          ('floor_max_flip_rate = 0.2', 'floor_max_flip_rate = 1.5', "floor_max_flip_rate"),
                          ('holds_abs_max = 1.0', 'holds_abs_max = 3.0', "holds_abs_max"),
                          ('writer = "local:step_0"', 'writer = "local:step_1"', "writer")]:
        cpath.write_text(base.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            load_task_config(cpath, repo)


def test_plan_runs_equivalence_first_then_one_cells_job_per_reader(repo, seal_cfg):
    cfg, _ = _setup_task(repo, seal_cfg)
    p = task.plan(cfg)
    assert [j.kind for j in p.jobs] == ["equivalence", "cells", "cells"]
    assert [j.reader for j in p.jobs] == ["local:step_32", "local:step_1", "local:step_32"]
    assert all(j.writer == "local:step_0" for j in p.jobs)
    assert p.jobs[1].indices == list(range(12)) and p.jobs[1].control_n == 8
    assert task.job_dir(cfg, p.jobs[0]).as_posix().endswith("results/task/equivalence")
    assert task.job_dir(cfg, p.jobs[2]).as_posix().endswith("results/task/readers/local_step_32")


def test_run_refuses_unregistered_and_a_bad_prompt_hash(repo, seal_cfg):
    cfg, tools = _setup_task(repo, seal_cfg)
    cpath = repo / "config" / "task.toml"
    cpath.write_text(cpath.read_text(encoding="utf-8").replace('registered_by = "0030"', 'registered_by = ""'), encoding="utf-8")
    with pytest.raises(task.Refusal, match="UNREGISTERED"):
        task.run(load_task_config(cpath, repo), repo, runner=FakeTaskSwap(), seal_cfg=seal_cfg, tools_dir=tools)
    cpath.write_text(cpath.read_text(encoding="utf-8").replace('registered_by = ""', 'registered_by = "0030"'), encoding="utf-8")
    (repo.parent / "upstream" / "data" / "tokens" / "gsm8k_test.json").write_text("[]", encoding="utf-8")
    with pytest.raises(task.Refusal, match="prompts"):
        task.run(cfg, repo, runner=FakeTaskSwap(), seal_cfg=seal_cfg, tools_dir=tools)


def test_run_writes_the_record_and_passes_the_tool_the_contract_arguments(repo, seal_cfg):
    cfg, tools = _setup_task(repo, seal_cfg)
    fake = FakeTaskSwap()
    rec = task.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["halted"] is None and rec["finished"] and len(rec["jobs"]) == 3 and all(j["done"] for j in rec["jobs"])
    eq = fake.calls[0]
    assert eq[2] == "equivalence" and "--expect-b1" in eq and "--reader" in eq and eq[eq.index("--reader") + 1] == "local:step_32"
    cells = fake.calls[1]
    assert cells[2] == "cells" and cells[cells.index("--writer") + 1] == "local:step_0"
    assert cells[cells.index("--far-seq") + 1] == "1" and cells[cells.index("--control-n") + 1] == "8"
    assert cells[cells.index("--max-new") + 1] == "8"
    env = fake.envs[1] if hasattr(fake, "envs") else None
    out = Path(cells[cells.index("--out") + 1])
    assert (out / "stale.jsonl").exists() and len((out / "fresh32.jsonl").read_text().splitlines()) == 8
    assert rec["prompts_sha256"] == cfg.prompts_sha256 and rec["jobs"][0]["kind"] == "equivalence"
    assert rec["equivalence"]["max_abs_diff"] == 0.0 and rec["equivalence"]["passed"] is True


def test_run_halts_when_the_live_cache_differs_from_the_dump(repo, seal_cfg):
    cfg, tools = _setup_task(repo, seal_cfg)
    fake = FakeTaskSwap(equivalence=lambda a: (0.25, float(a["--expect-b1"])))
    with pytest.raises(task.Refusal, match="equivalence"):
        task.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert "equivalence" in rec["halted"] and len(fake.calls) == 1


def test_run_halts_on_a_floor_over_its_limit_or_a_nondeterministic_identity(repo, seal_cfg):
    cfg, tools = _setup_task(repo, seal_cfg)

    def noisy_floor(i, step, arm, gold):
        from tests.fake_task_swap import default_outcome
        if arm == "fresh32":
            return "3", False
        return default_outcome(i, step, arm, gold)

    with pytest.raises(task.Refusal, match="floor"):
        task.run(cfg, repo, runner=FakeTaskSwap(outcome=noisy_floor), seal_cfg=seal_cfg, tools_dir=tools)

    def flaky(i, step, arm, gold):
        from tests.fake_task_swap import default_outcome
        if arm == "fresh2" and i == 4:
            return "77", False
        return default_outcome(i, step, arm, gold)

    with pytest.raises(task.Refusal, match="identity"):
        task.run(cfg, repo, runner=FakeTaskSwap(outcome=flaky), seal_cfg=seal_cfg, tools_dir=tools)
