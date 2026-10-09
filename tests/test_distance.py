"""The weight-distance driver: the plan, the gate's refusals, and a run against the fake instrument and tools."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lag_ladder.config import load_distance_config
from lag_ladder.distance import SEAL_NAME, UPSTREAM_PATHS, DumpRef, Refusal, assert_ready, plan, run
from lag_ladder.seal import write_prediction
from tests.conftest import commit_all, git
from tests.fake_instrument import FakeInstrument

CONFIG = '''
[distance]
results_dir = "results/distance"
upstream_path = "../upstream"
upstream_sha = "{sha}"
heldout = [1, 4]
dtype = "float32"
seed = 3
registered_by = "{reg}"
[distance.rule]
statistic = "median f*(tau_K) per rung"
tau_K = 0.3186
tau_ladder = [0.3186, 0.10, 0.03]
holds_max = 0.15
degrades_min = 0.50
[distance.controls]
identity_max = 0.0
scrambled_min = 0.90
[distance.noise]
pair = "olmo2-1b-rlvr1"
which = "source"
anchor_revision = "step_200"
rel_norms = [1e-3, 1e-1]
[[distance.pairs]]
name = "olmo2-1b-rlvr1"
tokens = "data/tokens/olmo.npy"
tokens_sha256 = "{osha}"
[[distance.pairs]]
name = "qwen2.5-1.5b-to-instruct"
tokens = "data/tokens/qwen.npy"
tokens_sha256 = "{qsha}"
[[distance.rungs]]
label = "olmo-sft-to-dpo"
pair = "olmo2-1b-rlvr1"
writer = {{ which = "source", local_path = "olmo2-1b-sft" }}
reader = {{ which = "target", local_path = "olmo2-1b-dpo" }}
[[distance.rungs]]
label = "olmo-dpo-to-step_200"
pair = "olmo2-1b-rlvr1"
writer = {{ which = "source", local_path = "olmo2-1b-dpo" }}
reader = {{ which = "target", revision = "step_200" }}
[[distance.rungs]]
label = "qwen-base-to-instruct"
pair = "qwen2.5-1.5b-to-instruct"
writer = {{ which = "source" }}
reader = {{ which = "target" }}
'''
LEDGER = "# Ledger\n\n## Entries\n\n### 0008 — 2026-10-10 — Distance ladder registered\n\nregisters config/distance.toml.\n"


def _upstream(root: Path) -> tuple[Path, str, dict]:
    up = root / "upstream"
    if (up / ".git").exists():
        return up, git(up, "rev-parse", "HEAD").strip(), _token_shas(up)
    for rel in UPSTREAM_PATHS:
        (up / rel).parent.mkdir(parents=True, exist_ok=True)
        (up / rel).write_text(f"# {rel}\n")
    git(up, "init", "-q", "-b", "main")
    git(up, "add", "-A")
    git(up, "commit", "-q", "-m", "pin")
    (up / "data" / "tokens").mkdir(parents=True)
    np.save(up / "data" / "tokens" / "olmo.npy", np.arange(32, dtype=np.int64).reshape(4, 8))
    np.save(up / "data" / "tokens" / "qwen.npy", np.arange(100, 132, dtype=np.int64).reshape(4, 8))
    (up / ".venv" / "Scripts").mkdir(parents=True)
    (up / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    return up, git(up, "rev-parse", "HEAD").strip(), _token_shas(up)


def _token_shas(up: Path) -> dict:
    return {n: hashlib.sha256((up / "data" / "tokens" / f"{n}.npy").read_bytes()).hexdigest() for n in ("olmo", "qwen")}


def _tools(root: Path) -> Path:
    t = root / "tools"
    t.mkdir(exist_ok=True)
    for name in ("perturb_checkpoint.py", "weight_distance.py"):
        (t / name).write_text("# stand-in\n")
    return t


def _local_checkpoint(repo: Path, name: str) -> None:
    d = repo / "results" / "distance" / "checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(f"weights of {name}".encode())
    (d / "provenance.json").write_text(json.dumps({"tool": "fake convert", "source": name}))


def _setup(repo: Path, seal_cfg, *, registered="0008", pin=None, seal=True, commit=True, checkpoints=True, qsha=None):
    up, sha, shas = _upstream(repo.parent)
    (repo / "config").mkdir(exist_ok=True)
    cpath = repo / "config" / "distance.toml"
    cpath.write_text(CONFIG.format(sha=pin or sha, reg=registered, osha=shas["olmo"], qsha=qsha or shas["qwen"]), encoding="utf-8")
    (repo / "ledger").mkdir(exist_ok=True)
    (repo / "ledger" / "ledger.md").write_text(LEDGER, encoding="utf-8")
    if checkpoints:
        for name in ("olmo2-1b-sft", "olmo2-1b-dpo"):
            _local_checkpoint(repo, name)
    if seal:
        write_prediction(SEAL_NAME, {"knee_rel_distance_at_holds_max": 0.01}, seal_cfg)
    if commit and git(repo, "status", "--porcelain").strip():
        commit_all(repo, "register")
    return load_distance_config(cpath, repo), _tools(repo.parent)


def test_plan_runs_controls_per_writer_then_rungs(repo, seal_cfg):
    cfg, _ = _setup(repo, seal_cfg, commit=False)
    p = plan(cfg)
    assert [w.label() for w in p.writers] == ["olmo2-1b-rlvr1:source@local-olmo2-1b-sft", "olmo2-1b-rlvr1:source@local-olmo2-1b-dpo",
                                              "qwen2.5-1.5b-to-instruct:source@default", "olmo2-1b-rlvr1:source@step_200"]
    kinds = [j.kind for j in p.jobs]
    assert kinds == (["identity"] * 3 + ["scrambled"] * 3) * 4 + ["rung"] * 15
    labels = [j.label for j in p.jobs if j.kind == "rung"]
    assert labels == ["olmo-sft-to-dpo"] * 3 + ["olmo-dpo-to-step_200"] * 3 + ["qwen-base-to-instruct"] * 3 + ["noise-0.001"] * 3 + ["noise-0.1"] * 3
    noise = [j for j in p.jobs if j.label == "noise-0.1"][0]
    assert noise.src == DumpRef("olmo2-1b-rlvr1", "source", "step_200", None) and noise.tgt == DumpRef("olmo2-1b-rlvr1", "source", None, "noise-0.1")


def test_gate_refuses_in_order(repo, seal_cfg):
    cfg, tools = _setup(repo, seal_cfg, registered="", seal=False)
    with pytest.raises(Refusal, match="UNREGISTERED"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    cfg, tools = _setup(repo, seal_cfg, qsha="TOKENS_SHA256_PENDING", seal=False)
    with pytest.raises(Refusal, match="placeholder for pair qwen"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    cfg, tools = _setup(repo, seal_cfg, qsha="1" * 64, seal=False)
    with pytest.raises(Refusal, match="hashes to"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    cfg, tools = _setup(repo, seal_cfg, seal=False, checkpoints=False)
    import shutil
    shutil.rmtree(repo / "results" / "distance" / "checkpoints")
    with pytest.raises(Refusal, match="local checkpoint .* is missing"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    cfg, tools = _setup(repo, seal_cfg, seal=False)
    with pytest.raises(Refusal, match="no committed seal"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    (tools / "weight_distance.py").unlink()
    with pytest.raises(Refusal, match="tool weight_distance.py is missing"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)


def test_run_records_rungs_distances_and_prunes_noise_weights(repo, seal_cfg):
    cfg, tools = _setup(repo, seal_cfg)
    assert_ready(cfg, repo, seal_cfg=seal_cfg, tools_dir=tools)
    fake = FakeInstrument()
    rec = run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["finished"] and rec["halted"] is None and len(rec["jobs"]) == 39
    assert set(rec["rungs"]) == {"olmo-sft-to-dpo", "olmo-dpo-to-step_200", "qwen-base-to-instruct", "noise-0.001", "noise-0.1"}
    assert rec["rungs"]["noise-0.1"]["target_rel_norm"] == 0.1 and rec["rungs"]["noise-0.1"]["realized_rel_norm"] == 0.1
    assert rec["rungs"]["noise-0.1"]["rel_delta_all"] == 0.1 and rec["rungs"]["noise-0.1"]["pruned"] is True
    assert rec["rungs"]["olmo-sft-to-dpo"]["rel_delta_all"] == 0.02 and rec["rungs"]["qwen-base-to-instruct"]["rel_delta_all"] == 0.05
    ck = repo / "results" / "distance" / "checkpoints"
    assert not (ck / "noise-0.1" / "model.safetensors").exists() and (ck / "noise-0.1" / "provenance.json").exists()
    assert (ck / "olmo2-1b-sft" / "model.safetensors").exists(), "real local checkpoints are never pruned"
    tools_calls = [c for c in fake.calls if c[1] != "-m"]
    assert sorted({Path(c[1]).name for c in tools_calls}) == ["perturb_checkpoint.py", "weight_distance.py"]
    perturb = [c for c in tools_calls if c[1].endswith("perturb_checkpoint.py")]
    assert len(perturb) == 2 and all("--seed" in c and c[c.index("--seed") + 1] == "3" for c in perturb)
    assert all(c[c.index("--revision") + 1] == "cstep_200" for c in perturb), "perturbs the resolved commit the anchor dump recorded"
    dumps = [c for c in fake.calls if c[1] == "-m" and c[2] == "scripts.dump_kv"]
    assert len({tuple(c[c.index("--out") + 1:]) for c in dumps}) == len(dumps), "a dump directory was dumped twice"
    assert any("--local-path" in c for c in dumps) and any("--revision" in c for c in dumps)
    by = {(j["label"], j["kind"]): j for j in rec["jobs"]}
    assert by[("noise-0.001", "rung")]["fstar"]["0.3186"] == 0.0           # 20 x 1e-3 = 0.02 < tau: nothing to remove
    assert by[("noise-0.1", "rung")]["fstar"]["0.3186"] > 0.5              # 20 x 0.1 = 2: nearly everything
    assert all(j["fstar"]["0.3186"] == 0.0 for j in rec["jobs"] if j["kind"] == "identity")
    # resumable: a second run adds no jobs and re-dumps nothing
    n = len(fake.calls)
    rec2 = run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert len(rec2["jobs"]) == 39 and not [c for c in fake.calls[n:] if c[1] == "-m" and c[2] == "scripts.dump_kv"]


def test_run_halts_on_a_failed_control_before_any_rung(repo, seal_cfg):
    cfg, tools = _setup(repo, seal_cfg)
    blind = FakeInstrument(deviation=lambda sm, tm, i, j: 0.0)
    with pytest.raises(Refusal, match="scrambled control failed"):
        run(cfg, repo, runner=blind, seal_cfg=seal_cfg, tools_dir=tools)
    rec = json.loads((repo / "results" / "distance" / "run.json").read_text())
    assert rec["halted"] and not rec["rungs"] and all(j["kind"] != "rung" for j in rec["jobs"])
