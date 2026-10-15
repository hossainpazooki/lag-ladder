"""summarize_task's statistics (design §15): the paired accuracy difference with its seeded paired bootstrap, the
(C) bands, the flip and no-answer rates, and the rank correlation with the per-prompt (B1). Pure; no run on disk."""
import json

import numpy as np
import pytest

from lag_ladder import summarize_task as st


def test_paired_bootstrap_interval_is_seeded_and_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    fresh = rng.random(400) < 0.3
    stale = fresh.copy()
    flip = rng.random(400) < 0.1
    stale[flip] = ~stale[flip]
    r = st.paired_delta(stale, fresh, reps=500, seed=0)
    assert r["n"] == 400 and r["delta_points"] == pytest.approx(100 * (stale.mean() - fresh.mean()))
    assert r["ci_low"] <= r["delta_points"] <= r["ci_high"] and r["reps"] == 500
    assert st.paired_delta(stale, fresh, reps=500, seed=0) == r
    assert st.paired_delta(stale, fresh, reps=500, seed=1) != r
    assert r["discordant"] == int((stale != fresh).sum())
    same = st.paired_delta(fresh, fresh, reps=100, seed=0)
    assert same["delta_points"] == 0.0 and same["ci_low"] == 0.0 == same["ci_high"]


def test_bands_follow_the_design_with_the_sign_fixed():
    rule = {"holds_abs_max": 1.0, "holds_ci_abs_max": 2.0, "degrades_max": -5.0, "degrades_ci_max": -2.0}
    assert st.band(0.4, -1.5, 1.9, rule) == "HOLDS_C"
    assert st.band(0.4, -2.5, 1.9, rule) == "UNRESOLVED"          # interval leaves +-2
    assert st.band(-6.0, -9.0, -2.5, rule) == "DEGRADES_C"
    assert st.band(-6.0, -9.0, -1.0, rule) == "UNRESOLVED"        # interval reaches above -2
    assert st.band(-1.5, -3.0, 0.0, rule) == "UNRESOLVED"
    assert st.band(-0.5, -1.0, 0.2, rule) == "HOLDS_C"


def test_location_clause_reads_the_interval_not_the_band():
    assert st.location_ok(ci_low=-1.2, ci_high=0.8, lower_bound=-2.0) is True
    assert st.location_ok(ci_low=0.1, ci_high=1.0, lower_bound=-2.0) is False      # excludes 0
    assert st.location_ok(ci_low=-2.5, ci_high=0.3, lower_bound=-2.0) is False     # lower end under the bound


def test_rates_and_rank_correlation():
    stale_ans = ["1", "2", None, "4", "5"]
    fresh_ans = ["1", "3", None, "4", None]
    assert st.flip_rate(stale_ans, fresh_ans) == pytest.approx(2 / 5)      # 2 vs 3, 5 vs None
    assert st.no_answer_rate(stale_ans) == pytest.approx(1 / 5)
    assert st.rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert st.rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # ties take average ranks: flips {0,1} against a (B1) that orders them perfectly reads the point-biserial rho
    flips = [0, 1, 0, 1, 1, 0]
    b1 = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]
    r0, r1 = 2.0, 5.0                      # average rank of the zeros and of the ones
    rb = [1, 6, 2, 5, 4, 3]
    rf = [r0 if f == 0 else r1 for f in flips]
    expected = np.corrcoef(rf, rb)[0, 1]
    assert st.rank_correlation(flips, b1) == pytest.approx(expected)
    assert st.rank_correlation([0, 0, 0], [1, 2, 3]) is None


# --- the run-level summarizer: recomputed from the JSONL records, withholds and refuses ---------------------------

def _run(repo, seal_cfg, outcome=None):
    from lag_ladder import task
    from tests.fake_task_swap import FakeTaskSwap
    from tests.test_task import _setup_task
    cfg, tools = _setup_task(repo, seal_cfg)
    task.run(cfg, repo, runner=FakeTaskSwap(outcome=outcome) if outcome else FakeTaskSwap(), seal_cfg=seal_cfg, tools_dir=tools)
    return cfg


def test_summary_recomputes_each_lag_from_the_records(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    s = st.summarize(cfg)
    assert s["registered_by"] == "0030" and s["equivalence"]["passed"] and set(s["lags"]) == {"local:step_1", "local:step_32"}
    g32 = s["lags"]["local:step_32"]
    assert g32["n"] == 12 and set(g32["arms"]) == {"stale", "near", "far"}
    stale = g32["arms"]["stale"]
    # the fake: FRESH right on index % 2 == 0 (6 of 12); STALE at step 32 flips index % 2 == 1 to "5"; a flip on an
    # even index never happens, so no correct answer is lost: delta 0, flips 6/12
    assert stale["delta"]["acc_fresh"] == pytest.approx(50.0) and stale["delta"]["delta_points"] == pytest.approx(0.0)
    assert stale["flip_rate"] == pytest.approx(6 / 12) and stale["band"] in ("HOLDS_C", "UNRESOLVED")
    assert stale["no_answer_rate"] == 0.0 and stale["cap_rate"] == pytest.approx(1 / 12)
    assert stale["rank_correlation_b1"] is not None and stale["mean_first_divergence"] is not None
    # NEAR and FAR answer 9999 everywhere: every correct FRESH answer is lost -> delta -50 on 6 of 12 discordant
    assert g32["arms"]["near"]["delta"]["delta_points"] == pytest.approx(-50.0)
    assert g32["arms"]["far"]["delta"]["delta_points"] == pytest.approx(-50.0)
    assert g32["arms"]["near"]["band"] == "DEGRADES_C" == g32["arms"]["far"]["band"]
    assert g32["controls"]["floor_passed"] and g32["controls"]["identity_passed"]
    assert g32["positive_controls_fire"] is True and g32["evaluable"] is True
    assert s["seal_readout"] is None or isinstance(s["seal_readout"], dict)
    assert (cfg.results_dir / "summary.json").exists()


def test_summary_marks_a_lag_unevaluable_when_a_positive_control_does_not_degrade(repo, seal_cfg):
    from tests.fake_task_swap import default_outcome

    def lazy_near(i, step, arm, gold):
        if arm == "near" and step == 1:
            return default_outcome(i, step, "fresh", gold)       # NEAR reads like FRESH: the instrument did not fire
        return default_outcome(i, step, arm, gold)

    cfg = _run(repo, seal_cfg, outcome=lazy_near)
    s = st.summarize(cfg)
    g1, g32 = s["lags"]["local:step_1"], s["lags"]["local:step_32"]
    assert g1["evaluable"] is False and g1["arms"]["stale"]["band"] is None and g1["arms"]["stale"]["delta"] is not None
    assert g32["evaluable"] is True


def test_summary_refuses_a_record_hash_mismatch(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    p = cfg.results_dir / "readers" / "local_step_1" / "stale.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"answer": "5"', '"answer": "6"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        st.summarize(cfg)


def test_seal_readout_reads_the_three_clauses(tmp_path):
    from lag_ladder.summarize_reader import PREDICTIONS
    (tmp_path / PREDICTIONS).mkdir(parents=True)
    (tmp_path / PREDICTIONS / "ownrun-task.json").write_text(json.dumps({"prediction": {"predictions": {
        "i_positive_controls_fire": {"readers": ["local:step_1", "local:step_32"]},
        "ii_location_at_first_lag": {"reader": "local:step_1", "lower_bound": -2.0},
        "iii_flip_rate_at_last_lag": {"reader": "local:step_32", "at_least": 0.10}}}}), encoding="utf-8")
    summary = {"lags": {
        "local:step_1": {"positive_controls_fire": True, "evaluable": True,
                         "arms": {"stale": {"delta": {"ci_low": -1.0, "ci_high": 0.5}, "flip_rate": 0.05}}},
        "local:step_32": {"positive_controls_fire": True, "evaluable": True,
                          "arms": {"stale": {"delta": {"ci_low": -3.0, "ci_high": -0.5}, "flip_rate": 0.2}}}}}
    c = st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]
    assert c["i"]["verdict"] == "HELD" and c["ii"]["verdict"] == "HELD" and c["iii"]["verdict"] == "HELD"
    summary["lags"]["local:step_32"]["positive_controls_fire"] = False
    summary["lags"]["local:step_32"]["evaluable"] = False
    c = st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]
    assert c["i"]["verdict"] == "FALSIFIED" and c["iii"]["verdict"] == "UNREAD"
    summary["lags"]["local:step_1"]["arms"]["stale"]["delta"]["ci_low"] = 0.2
    assert st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]["ii"]["verdict"] == "FALSIFIED"
