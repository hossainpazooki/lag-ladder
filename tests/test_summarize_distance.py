"""The distance summarizer recomputes every f* and every distance from disk and refuses on any disagreement."""
import json

import pytest

from lag_ladder.distance import run
from lag_ladder.summarize_distance import render, summarize
from tests.fake_instrument import FakeInstrument
from tests.test_distance import _setup


def _ran(repo, seal_cfg):
    cfg, tools = _setup(repo, seal_cfg)
    run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)
    return cfg


def test_summary_sorts_rungs_by_distance_and_bands_them(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    s = summarize(cfg)
    assert all(w["controls_passed"] for w in s["writers"].values()) and len(s["writers"]) == 4
    labels = [e["label"] for e in s["rungs"]]
    # 0.001 (noise), then the two local rungs at the fake's 0.02 in config order, then 0.05, then 0.1
    assert labels == ["noise-0.001", "olmo-sft-to-dpo", "olmo-dpo-to-step_200", "qwen-base-to-instruct", "noise-0.1"]
    by = {e["label"]: e for e in s["rungs"]}
    assert by["noise-0.001"]["band"] == "HOLDS" and by["noise-0.1"]["band"] == "DEGRADES"
    assert by["qwen-base-to-instruct"]["rel_delta_all"] == 0.05 and by["qwen-base-to-instruct"]["band"] in ("UNRESOLVED", "DEGRADES")
    assert by["noise-0.1"]["realized_rel_norm"] == 0.1 and by["noise-0.1"]["kind"] == "noise"
    assert by["olmo-sft-to-dpo"]["median_one_minus_r2"] == pytest.approx(0.4, abs=1e-6)   # 20 x 0.02
    text = render(s)
    assert "oracle lower bound" in text and "noise-0.1" in text and "DEGRADES" in text


def test_summary_refuses_a_distance_over_other_bytes_than_the_dumps(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    dfile = cfg.results_dir / "rungs" / "qwen-base-to-instruct" / "distance.json"
    rec = json.loads(dfile.read_text())
    rec["reader"]["files"]["model.safetensors"] = "f" * 64
    dfile.write_text(json.dumps(rec))
    # the run record names the original file by hash, so the first refusal is the hash mismatch
    with pytest.raises(ValueError, match="not the one run.json names"):
        summarize(cfg)
    run_rec_path = cfg.results_dir / "run.json"
    run_rec = json.loads(run_rec_path.read_text())
    from lag_ladder.hashing import sha256_file_bytes
    run_rec["rungs"]["qwen-base-to-instruct"]["distance_sha256"] = sha256_file_bytes(dfile)
    run_rec_path.write_text(json.dumps(run_rec))
    with pytest.raises(ValueError, match="dump's manifest names"):
        summarize(cfg)


def test_summary_refuses_an_incomplete_run(repo, seal_cfg):
    cfg, tools = _setup(repo, seal_cfg)
    run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools, limit=30)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(cfg)
