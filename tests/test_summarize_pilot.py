"""The summarizer recomputes every f* from disk and refuses on any disagreement."""
import json

import numpy as np
import pytest

from lag_ladder.fstar_record import evaluate_job
from lag_ladder.pilot import run
from lag_ladder.summarize_pilot import render, summarize
from tests.fake_instrument import FakeInstrument
from tests.test_pilot import _setup


def _ran(repo, seal_cfg, deviation=None):
    cfg = _setup(repo, seal_cfg)
    fake = FakeInstrument() if deviation is None else FakeInstrument(deviation=deviation)
    run(cfg, repo, runner=fake, seal_cfg=seal_cfg)
    return cfg


def test_summary_bands_each_lag_and_carries_the_oracle_sentence(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    s = summarize(cfg)
    a = s["anchors"]["step_200"]
    assert a["controls_passed"] and a["identity"]["max"] == 0.0 and a["scrambled"]["median"] >= 0.9
    lags = [c["lag"] for c in a["ladder"]]
    assert lags == [200, 400]
    assert a["ladder"][0]["band"] == "HOLDS" and a["ladder"][0]["median_fstar"]["0.3186"] == 0.0
    assert a["ladder"][1]["band"] in ("UNRESOLVED", "DEGRADES") and a["ladder"][1]["median_fstar"]["0.3186"] > 0.0
    assert a["ladder"][1]["median_fstar"]["0.03"] >= a["ladder"][1]["median_fstar"]["0.1"] >= a["ladder"][1]["median_fstar"]["0.3186"]
    # the descriptive 1 - R^2 per lag follows the fake's deviation rule (lag / 1000) and is reported beside f*
    assert a["ladder"][0]["median_one_minus_r2"] == pytest.approx(0.2, abs=1e-6)
    assert a["ladder"][1]["median_one_minus_r2"] == pytest.approx(0.4, abs=1e-6)
    assert len(a["ladder"][1]["per_seq_one_minus_r2"]) == 3
    text = render(s)
    assert "oracle lower bound" in text and "HOLDS" in text and "400" in text and "median 1-R^2" in text


def test_summary_refuses_a_tampered_per_token_file(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    pt = next((cfg.results_dir / cfg.pair / "step_200-to-step_400").rglob("pertoken.npz"))
    with np.load(pt) as z:
        arrays = {k: z[k] for k in z}
    arrays["same_K"] = arrays["same_K"] * 1.5
    np.savez_compressed(pt, **arrays)
    with pytest.raises(ValueError, match="does not match the sha256"):
        summarize(cfg)


def test_summary_refuses_a_report_whose_squares_do_not_sum(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    job = next((cfg.results_dir / cfg.pair / "step_200-to-step_400").rglob("report.json")).parent
    rec = json.loads((job / "report.json").read_text())
    rec["same"]["K"][0]["sse"][0] *= 2.0
    (job / "report.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="do not sum to the recorded SSE"):
        evaluate_job(job, [0.3186])


def test_summary_refuses_an_incomplete_run(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)
    run(cfg, repo, runner=FakeInstrument(), limit=7, seal_cfg=seal_cfg)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(cfg)


def test_summary_withholds_the_ladder_when_a_control_fails(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)
    weak = lambda sm, tm, i, j: (0.0 if sm["dir"] == tm["dir"] else (0.2 if i != j else 0.5))   # scrambled too small
    with pytest.raises(RuntimeError):
        run(cfg, repo, runner=FakeInstrument(deviation=weak), seal_cfg=seal_cfg)
    # the driver halted before the ladder, so the summary reports the failure and no table
    with pytest.raises(ValueError, match="incomplete"):
        summarize(cfg)
