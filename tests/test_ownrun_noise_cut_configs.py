"""The five configurations ledger 0042 registers: the ten noise readers config/ownrun_noise.toml names, the planner's
2,840 jobs over the five, and the three prefix-cut configurations' spans pins against the bytes tools/cut_spans.py
writes from 0017's rollout spans file (when the instrument checkout is beside this tree)."""
import hashlib
import importlib.util

import pytest

from lag_ladder import REPO_ROOT, reader
from lag_ladder.config import load_reader_config
from lag_ladder.summarize_noise_cut import CONFIGS


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H, R = "qwen2.5-1.5b-to-instruct", "qwen2.5-1.5b-ownrun-rollout"


def test_the_config_names_ten_readers_seed_0_at_every_lag_then_seeds_1_and_2_at_lags_1_and_32():
    cfg = load_reader_config(REPO_ROOT / "config" / "ownrun_noise.toml", REPO_ROOT)
    es = _tool("noise_reader").noise_readers(cfg)
    assert [(e["lag"], e["seed"]) for e in es] == [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (32, 0),
                                                   (1, 1), (32, 1), (1, 2), (32, 2)]
    assert all(e["pairs"] == [H, R] for e in es)
    assert es[0]["name"] == "noise-ownrun-lag1-s0"


def test_the_five_configs_plan_2840_jobs():
    n = {k: len(reader.plan(load_reader_config(REPO_ROOT / "config" / f, REPO_ROOT)).jobs) for k, f in CONFIGS.items()}
    assert n == {"replicate": 660, "noise": 1100, "cut_prompt": 360, "cut_192": 360, "cut_384": 360}


ROLLOUT_FILE = REPO_ROOT.parent / "kv-transfer-replication" / "data" / "tokens" / "qwen2.5-1.5b-ownrun_rollout_n10_seed0.spans.json"


@pytest.mark.skipif(not ROLLOUT_FILE.exists(), reason="0017's rollout spans file is in the instrument checkout only")
@pytest.mark.parametrize("cut", ["prompt", "192", "384"])
def test_the_configs_pin_the_bytes_the_tool_writes(tmp_path, cut):
    cfg = load_reader_config(REPO_ROOT / "config" / f"ownrun_cut_{cut}.toml", REPO_ROOT)
    pt = cfg.pairs["qwen2.5-1.5b-to-instruct"]
    rollout_sha = load_reader_config(REPO_ROOT / "config" / "ownrun_far.toml", REPO_ROOT).pairs[
        "qwen2.5-1.5b-ownrun-rollout"].spans_sha256
    out = tmp_path / pt.spans.name
    assert _tool("cut_spans").main(["--rollout-spans", str(ROLLOUT_FILE), "--rollout-spans-sha256", rollout_sha,
                           "--cut", cut, "--out", str(out)]) == 0
    assert hashlib.sha256(out.read_bytes()).hexdigest() == pt.spans_sha256
