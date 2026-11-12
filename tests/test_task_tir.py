"""The reduced-(C) driver (design 2026-11-09, reduced (C), decisions 4-5): the config loader's refusals, the plan (fidelity
first, then one cells job per reader with the floor on every prompt and the controls on the first n), the gate (adds the
loopback sandbox endpoint), the run's halting rules and the probe, with a stand-in for tools/task_swap_tir.py."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from lag_ladder import task_tir
from lag_ladder.seal import write_prediction
from tests.conftest import commit_all, git
from tests.fake_task_swap_tir import FakeTaskSwapTir
from tests.test_task import _setup_task

TIR_CONFIG = '''
[task_tir]
results_dir = "results/pipelinerl_tir_task"
upstream_path = "../upstream"
upstream_sha = "{sha}"
registered_by = "{reg}"
seal = "pipelinerl-tir-task"
writer = "local:pltir-step_0"
readers = {readers}
prompts_all = "data/tokens/gsm8k_test.json"
prompts_all_sha256 = "{asha}"
frozen = "data/tokens/pltir_frozen.json"
frozen_sha256 = "{fsha}"
prompts = "data/tokens/gsm8k_test_reserved_pltir.json"
prompts_sha256 = "{psha}"
n_prompts = 9
far_tokens = "data/tokens/qwen.npy"
far_tokens_sha256 = "{qsha}"
far_seq = 1
max_new = 8
max_turns = 3
max_model_len = 4096
min_generation_tokens = 2
system_prompt = "config/pipelinerl_tir_system_prompt.txt"
system_prompt_sha256 = "{ssha}"
content_eos = "{eos}"
seed = 0

[task_tir.rule]
bootstrap_reps = 200
control_min_flip_rate = 0.5

[task_tir.controls]
n_prompts = 4
floor_max_flip_rate = 0.2
fidelity_episodes = "data/tokens/pltir_episodes.jsonl"
fidelity_episodes_sha256 = "{esha}"
'''

LEDGER_0039 = "\n### 0039 — 2026-11-10 — reduced (C) on the PipelineRL TIR arm registered (test)\n\nprior-entries-sha256: x\n\nregisters config/pipelinerl_tir_task.toml.\n"
SYSTEM = "You are a math-focused AI Agent.\n"


def _setup_tir(repo, seal_cfg, *, registered="0039", seal=True, readers=("local:pltir-step_1", "local:pltir-step_32"),
               predictions=None, eos="strip", frozen=(1, 4, 10)):
    """The (C) fixture of test_task (upstream, prompts, qwen tokens, the ownrun-task registration) plus this arm's files:
    the frozen draw, the reserved prompts, the system prompt, the episodes file, the pltir checkpoints, the config, its
    ledger entry and its seal, committed. The tool is a stand-in file; the runner is the fake."""
    _, tools = _setup_task(repo, seal_cfg)
    up = repo.parent / "upstream"
    allp = up / "data" / "tokens" / "gsm8k_test.json"
    prompts = json.loads(allp.read_text(encoding="utf-8"))               # 12 rows, q{i}, gold str(i*2)
    fz = up / "data" / "tokens" / "pltir_frozen.json"
    fz.write_text(json.dumps({"seed": 0, "n": len(frozen), "total": len(prompts), "indices": sorted(frozen)}), encoding="utf-8")
    res = up / "data" / "tokens" / "gsm8k_test_reserved_pltir.json"
    res.write_text(json.dumps([p for p in prompts if p["index"] not in frozen]), encoding="utf-8")
    eps = up / "data" / "tokens" / "pltir_episodes.jsonl"
    eps.write_text("".join(json.dumps({"schema": "pltir-episode-v1", "index": i, "question": f"q{i}", "gold": str(2 * i), "turns": []}) + "\n"
                           for i in sorted(frozen)), encoding="utf-8")
    sp = repo / "config" / "pipelinerl_tir_system_prompt.txt"
    sp.write_text(SYSTEM, encoding="utf-8", newline="\n")
    for step in sorted({0, *(int(r.rsplit("_", 1)[1]) for r in readers)}):
        d = repo / "results" / "distance" / "checkpoints" / f"pltir-step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"p" * (step + 1)); (d / "config.json").write_text("{}")
    sha = git(up, "rev-parse", "HEAD").strip()
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cpath = repo / "config" / "pipelinerl_tir_task.toml"
    cpath.write_text(TIR_CONFIG.format(sha=sha, reg=registered, readers=json.dumps(list(readers)), asha=h(allp), fsha=h(fz),
                                       psha=h(res), qsha=h(up / "data" / "tokens" / "qwen.npy"), ssha=h(sp), eos=eos, esha=h(eps)),
                     encoding="utf-8")
    lpath = repo / "ledger" / "ledger.md"
    lpath.write_text(lpath.read_text(encoding="utf-8") + LEDGER_0039, encoding="utf-8")
    if seal:
        write_prediction("pipelinerl-tir-task", {"registering_entry": "0039", "predictions": predictions or {}}, seal_cfg)
    (tools / "task_swap_tir.py").write_text("# stand-in\n")
    commit_all(repo, "register the tir task")
    return task_tir.load_task_tir_config(cpath, repo), tools


ENV = {"SANDBOX_ENDPOINT": "http://127.0.0.1:8080"}


def test_config_loads_and_refuses_bad_values(repo, seal_cfg):
    cfg, _ = _setup_tir(repo, seal_cfg)
    assert cfg.registered_by == "0039" and cfg.seal == "pipelinerl-tir-task" and cfg.n_prompts == 9
    assert cfg.writer == "local:pltir-step_0" and cfg.readers == ("local:pltir-step_1", "local:pltir-step_32")
    assert cfg.max_turns == 3 and cfg.max_model_len == 4096 and cfg.min_generation_tokens == 2 and cfg.content_eos == "strip"
    assert cfg.system_prompt == repo / "config" / "pipelinerl_tir_system_prompt.txt"
    assert cfg.controls["n_prompts"] == 4 and cfg.controls["fidelity_episodes"].name == "pltir_episodes.jsonl"
    cpath = repo / "config" / "pipelinerl_tir_task.toml"
    text = cpath.read_text(encoding="utf-8")
    for old, new, msg in (('content_eos = "strip"', 'content_eos = "maybe"', "content_eos"),
                          ("max_turns = 3", "max_turns = 0", "max_turns"),
                          ("min_generation_tokens = 2", "min_generation_tokens = 5000", "min_generation_tokens"),
                          ('writer = "local:pltir-step_0"', 'writer = "local:pltir-step_1"', "also a reader"),
                          ("n_prompts = 4\n", "n_prompts = 40\n", "controls"),
                          ('system_prompt = "config/pipelinerl_tir_system_prompt.txt"', 'system_prompt = "../x.txt"', "system_prompt")):
        cpath.write_text(text.replace(old, new, 1), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            task_tir.load_task_tir_config(cpath, repo)
    cpath.write_text(text.replace('content_eos = "strip"', 'content_eos = "CONTENT_EOS_PENDING"'), encoding="utf-8")
    assert task_tir.load_task_tir_config(cpath, repo).content_eos == "CONTENT_EOS_PENDING"


def test_plan_runs_fidelity_first_then_one_cells_job_per_reader_with_rc3_controls(repo, seal_cfg):
    cfg, _ = _setup_tir(repo, seal_cfg)
    p = task_tir.plan(cfg)
    assert [j.kind for j in p.jobs] == ["fidelity", "cells", "cells"]
    assert [j.reader for j in p.jobs] == ["local:pltir-step_0", "local:pltir-step_1", "local:pltir-step_32"]
    assert p.jobs[1].indices == [0, 2, 3, 5, 6, 7, 8, 9, 11]        # the reserved indices in file order, not 0..n-1
    assert p.jobs[1].floor_n == 9 and p.jobs[1].control_n == 4
    assert task_tir.job_dir(cfg, p.jobs[0]).as_posix().endswith("results/pipelinerl_tir_task/fidelity")
    assert task_tir.job_dir(cfg, p.jobs[2]).as_posix().endswith("results/pipelinerl_tir_task/readers/local_pltir-step_32")
    assert task_tir.projection(cfg) == {"prompts": 9, "arms_per_prompt": 7, "episodes_per_reader": 9 * 3 + 9 + 3 * 4}


def test_gate_refuses_unregistered_pending_eos_a_bad_partition_and_a_non_loopback_sandbox(repo, seal_cfg):
    cfg, tools = _setup_tir(repo, seal_cfg)
    cpath = repo / "config" / "pipelinerl_tir_task.toml"
    text = cpath.read_text(encoding="utf-8")
    cpath.write_text(text.replace('registered_by = "0039"', 'registered_by = ""'), encoding="utf-8")
    with pytest.raises(task_tir.Refusal, match="UNREGISTERED"):
        task_tir.run(task_tir.load_task_tir_config(cpath, repo), repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    cpath.write_text(text.replace('content_eos = "strip"', 'content_eos = "CONTENT_EOS_PENDING"'), encoding="utf-8")
    commit_all(repo, "pending eos")
    with pytest.raises(task_tir.Refusal, match="content_eos"):
        task_tir.run(task_tir.load_task_tir_config(cpath, repo), repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    cpath.write_text(text, encoding="utf-8")
    commit_all(repo, "restore")
    with pytest.raises(task_tir.Refusal, match="SANDBOX_ENDPOINT"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env={})
    with pytest.raises(task_tir.Refusal, match="loopback"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env={"SANDBOX_ENDPOINT": "http://10.1.2.3:8080"})
    # the reserved file must be exactly the full set minus the frozen draw (hash-pinned, so the file is rewritten and re-pinned)
    res = repo.parent / "upstream" / "data" / "tokens" / "gsm8k_test_reserved_pltir.json"
    rows = json.loads(res.read_text(encoding="utf-8"))
    res.write_text(json.dumps(rows[:-1]), encoding="utf-8")
    cpath.write_text(text.replace(cfg.prompts_sha256, hashlib.sha256(res.read_bytes()).hexdigest()).replace("n_prompts = 9", "n_prompts = 8"), encoding="utf-8")
    commit_all(repo, "short reserved")
    with pytest.raises(task_tir.Refusal, match="partition"):
        task_tir.run(task_tir.load_task_tir_config(cpath, repo), repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)


def test_run_writes_the_record_and_passes_the_tool_the_contract_arguments(repo, seal_cfg):
    cfg, tools = _setup_tir(repo, seal_cfg)
    fake = FakeTaskSwapTir()
    rec = task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    assert rec["halted"] is None and rec["finished"] and len(rec["jobs"]) == 3 and all(j["done"] for j in rec["jobs"])
    fid = fake.calls[0]
    assert fid[2] == "fidelity" and fid[fid.index("--tokenizer") + 1] == "local:pltir-step_0"
    assert fid[fid.index("--content-eos") + 1] == "strip" and fid[fid.index("--episodes") + 1].endswith("pltir_episodes.jsonl")
    assert fid[fid.index("--system-prompt") + 1].endswith("pipelinerl_tir_system_prompt.txt")
    cells = fake.calls[1]
    assert cells[2] == "cells" and cells[cells.index("--writer") + 1] == "local:pltir-step_0" and cells[cells.index("--reader") + 1] == "local:pltir-step_1"
    for flag, val in (("--far-seq", "1"), ("--max-new", "8"), ("--max-turns", "3"), ("--max-model-len", "4096"), ("--min-generation", "2"),
                      ("--floor-n", "9"), ("--control-n", "4"), ("--content-eos", "strip")):
        assert cells[cells.index(flag) + 1] == val, flag
    assert fake.envs[1]["KVT_CHECKPOINT_ROOT"].endswith("checkpoints") and fake.envs[1]["SANDBOX_ENDPOINT"] == ENV["SANDBOX_ENDPOINT"]
    out = Path(cells[cells.index("--out") + 1])
    assert len((out / "fresh32.jsonl").read_text().splitlines()) == 9 and len((out / "near.jsonl").read_text().splitlines()) == 4
    assert rec["fidelity"]["passed"] is True and rec["jobs"][0]["kind"] == "fidelity"
    assert set(rec["jobs"][1]["files_sha256"]) == {f"{a}.jsonl" for a in ("fresh", "stale_prompt", "stale_shared", "fresh32", "near", "far", "fresh2")}
    assert rec["jobs"][1]["seconds"] >= 0 and rec["content_eos"] == "strip" and rec["system_prompt_sha256"] == cfg.system_prompt_sha256
    assert set(rec["controls"]) == {"local:pltir-step_1", "local:pltir-step_32"}
    assert rec["controls"]["local:pltir-step_1"] == {"floor_n": 9, "floor_flip_rate": pytest.approx(1 / 9), "floor_limit": 0.2, "floor_passed": True,
                                                     "control_n": 4, "identity_flips": 0, "identity_passed": True}


def test_run_halts_on_a_fidelity_mismatch_before_any_cell(repo, seal_cfg):
    cfg, tools = _setup_tir(repo, seal_cfg)
    fake = FakeTaskSwapTir(fidelity=lambda args: (98, 2))
    with pytest.raises(task_tir.Refusal, match="fidelity"):
        task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    assert [c[2] for c in fake.calls] == ["fidelity"]
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert rec["halted"] and "fidelity" in rec["halted"] and rec["fidelity"]["turns_mismatched"] == 2 and rec["fidelity"]["passed"] is False


def test_run_halts_on_a_floor_over_its_limit_or_a_nondeterministic_identity(repo, seal_cfg):
    from tests.fake_task_swap_tir import default_outcome
    cfg, tools = _setup_tir(repo, seal_cfg)

    def bad_floor(i, step, arm, gold):
        if arm == "fresh32":
            return "77", "MathAnswer", 2, False
        return default_outcome(i, step, arm, gold)

    with pytest.raises(task_tir.Refusal, match="floor"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(outcome=bad_floor), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert rec["halted"] and rec["controls"]["local:pltir-step_1"]["floor_passed"] is False

    def flaky(i, step, arm, gold):
        if arm == "fresh2" and i == 2:
            return "3", "MathAnswer", 2, False
        return default_outcome(i, step, arm, gold)

    with pytest.raises(task_tir.Refusal, match="identity"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(outcome=flaky), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)


def test_run_passes_the_remaining_budget_and_the_later_shards_to_cells_and_halts_on_the_tools_projection(repo, seal_cfg):
    cfg, tools = _setup_tir(repo, seal_cfg)
    per_reader = 9 * 3 + 9 + 3 * 4                    # episodes_of one cells job on the fixture (= projection()["episodes_per_reader"])
    assert task_tir.episodes_of(task_tir.plan(cfg).jobs[1]) == per_reader
    fake = FakeTaskSwapTir()
    rec = task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV, budget_seconds=3600.0)
    first, second = fake.calls[1], fake.calls[2]
    assert 0 < float(first[first.index("--budget-seconds") + 1]) <= 3600.0
    assert first[first.index("--later-episodes") + 1] == str(per_reader)      # the second reader's episodes
    assert second[second.index("--later-episodes") + 1] == "0"
    assert rec["budget_seconds"] == 3600.0 and rec["budget"]["local:pltir-step_1"]["over"] is False and rec["halted"] is None
    assert all(b["checked"] is True for b in rec["budget"].values())
    # no budget: the flag is absent (the tool prints its projection only), the later-shard count is still passed. The
    # same results_dir still holds the first run's budget.json per reader; the fake's start-of-call write (checked:false)
    # replaces it, and run() records checked:true reports only, so the earlier sitting's projection is not re-read
    fake = FakeTaskSwapTir()
    rec = task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    assert "--budget-seconds" not in fake.calls[1] and fake.calls[1][fake.calls[1].index("--later-episodes") + 1] == str(per_reader)
    assert rec["budget_seconds"] is None and rec["budget"] == {}
    left = list(cfg.results_dir.rglob("budget.json"))
    assert left and all(json.loads(p.read_text(encoding="utf-8")) == {"schema": "tir-budget-v1", "checked": False} for p in left)
    # over: the first cells job halts the run for a ruling; the second reader is never called
    fake = FakeTaskSwapTir(budget_over=True)
    with pytest.raises(task_tir.Refusal, match="budget"):
        task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV, budget_seconds=10.0)
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert [c[2] for c in fake.calls] == ["fidelity", "cells"] and "budget" in rec["halted"]
    assert rec["budget"]["local:pltir-step_1"]["over"] is True and rec["finished"] is None
    with pytest.raises(task_tir.Refusal, match="positive"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV, budget_seconds=0.0)


def test_run_with_a_reader_filter_runs_the_fidelity_then_only_the_named_readers(repo, seal_cfg):
    cfg, tools = _setup_tir(repo, seal_cfg)
    fake = FakeTaskSwapTir()
    rec = task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV, readers=["local:pltir-step_32"])
    assert [c[2] for c in fake.calls] == ["fidelity", "cells"] and rec["readers_run"] == ["local:pltir-step_32"]
    assert not (cfg.results_dir / "readers" / "local_pltir-step_1").exists()
    with pytest.raises(task_tir.Refusal, match="not one of"):
        task_tir.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools, env=ENV, readers=["local:step_1"])


def test_probe_runs_unregistered_on_the_first_n_prompts_outside_results(repo, seal_cfg, tmp_path):
    cfg, tools = _setup_tir(repo, seal_cfg, registered="", seal=False)
    fake = FakeTaskSwapTir()
    rec = task_tir.probe(cfg, repo, out=tmp_path / "probe", n=5, runner=fake, tools_dir=tools, env=ENV)
    assert rec["schema"] == "tir-probe-v1" and rec["reader"] == "local:pltir-step_32" and rec["n"] == 5
    assert [c[2] for c in fake.calls] == ["fidelity", "cells"]
    cells = fake.calls[1]
    assert cells[cells.index("--floor-n") + 1] == "5" and cells[cells.index("--control-n") + 1] == "5"
    assert json.loads(Path(cells[cells.index("--indices") + 1]).read_text()) == [0, 2, 3, 5, 6]
    assert rec["fidelity"]["passed"] and rec["controls"]["floor_n"] == 5
    assert set(rec["arms"]) == {"stale_prompt", "stale_shared", "near", "far"}
    assert 0 <= rec["arms"]["stale_prompt"]["flip_rate"] <= 1 and "math_answer_rate" in rec["arms"]["stale_prompt"]
    assert (tmp_path / "probe" / "probe.json").exists()
    with pytest.raises(task_tir.Refusal, match="results/"):
        task_tir.probe(cfg, repo, out=repo / "results" / "x", n=5, runner=fake, tools_dir=tools, env=ENV)


def test_cli_plan_prints_the_jobs_and_the_projection(repo, seal_cfg, monkeypatch, capsys):
    cfg, tools = _setup_tir(repo, seal_cfg)
    monkeypatch.setattr(task_tir, "REPO_ROOT", repo)
    monkeypatch.setattr(task_tir, "TOOLS", tools)
    assert task_tir.main(["plan", "--config", str(repo / "config" / "pipelinerl_tir_task.toml")]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "fidelity" in out and "PROJECTION prompts 9 arms-per-prompt 7 episodes-per-reader 48" in out
    assert task_tir.main(["probe", "--config", str(repo / "config" / "pipelinerl_tir_task.toml"), "--n", "3"]) == 2
    assert "needs --out" in capsys.readouterr().err


def test_read_jsonl_splits_on_newline_only(tmp_path):
    # amendment 1 (2026-11-10): str.splitlines() also breaks a line at U+2028 (and U+2029, U+0085), which a model's
    # completion text may carry and json.dumps(ensure_ascii=False) would leave raw inside a string. The Task 4 writer
    # escapes it (ensure_ascii=True); the reader splits on "\n" only, so a record carrying one stays one row whichever
    # way it was written
    for ch in ("\u2028", "\u2029", "\u0085"):
        text = "x = 1" + ch + "y = 2"
        rec = {"schema": task_tir.RECORD_SCHEMA, "index": 0, "completion_texts": [text]}
        escaped = tmp_path / "escaped.jsonl"
        with open(escaped, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        assert escaped.read_bytes().isascii()                                   # the amended writer's file: escape only
        rows = task_tir._read_jsonl(escaped)
        assert len(rows) == 1 and rows[0]["completion_texts"] == [text]
        raw = tmp_path / "raw.jsonl"
        with open(raw, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"schema": "tir-record-v1", "index": 0, "completion_texts": ["x = 1' + ch + 'y = 2"]}\n')
        assert ch.encode("utf-8") in raw.read_bytes()                           # the character is raw in the file
        assert len(raw.read_text(encoding="utf-8").splitlines()) == 2           # the split the reader must not use
        rows = task_tir._read_jsonl(raw)
        assert len(rows) == 1 and rows[0]["completion_texts"] == [text]


def test_gate_refuses_a_ledger_entry_that_does_not_name_this_config(repo, seal_cfg):
    # a number alone is not a registration. The same number can head another entry once the
    # ledger has grown under a rebase, so the entry registered_by names must itself name this config
    cfg, tools = _setup_tir(repo, seal_cfg)
    lpath = repo / "ledger" / "ledger.md"
    text = lpath.read_text(encoding="utf-8")
    assert text.count("registers config/pipelinerl_tir_task.toml.") == 1
    lpath.write_text(text.replace("registers config/pipelinerl_tir_task.toml.", "the outcome of another run of pipelinerl_tir_task.toml."),
                     encoding="utf-8")
    commit_all(repo, "entry 0039 is another entry")
    with pytest.raises(task_tir.Refusal, match="does not name config/pipelinerl_tir_task.toml"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    # the entry is the text under its own heading only: a later entry that names the config does not register it for 0039
    later = "\n### 0040 \u2014 2026-11-11 \u2014 another entry (test)\n\nregisters config/pipelinerl_tir_task.toml.\n"
    lpath.write_text(text.replace("registers config/pipelinerl_tir_task.toml.", "the outcome of another run.") + later, encoding="utf-8")
    commit_all(repo, "a later entry names the config")
    with pytest.raises(task_tir.Refusal, match="does not name config/pipelinerl_tir_task.toml"):
        task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
    lpath.write_text(text, encoding="utf-8")
    commit_all(repo, "the registering entry again")
    task_tir.run(cfg, repo, runner=FakeTaskSwapTir(), seal_cfg=seal_cfg, tools_dir=tools, env=ENV)
