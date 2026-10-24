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


def test_flip_rate_interval_is_a_seeded_bootstrap_over_prompts():
    flips = [1] * 55 + [0] * 9
    r = st.flip_rate_interval(flips, reps=500, seed=0)
    assert r["rate"] == pytest.approx(55 / 64) and r["ci_low"] <= r["rate"] <= r["ci_high"] and r["n"] == 64
    assert r["ci_low"] > 0.7 and r["ci_high"] < 1.0
    assert st.flip_rate_interval(flips, reps=500, seed=0) == r and st.flip_rate_interval(flips, reps=500, seed=1) != r
    assert st.flip_rate_interval([0, 0, 0, 0], reps=50, seed=0) == {"n": 4, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "reps": 50, "seed": 0}
    with pytest.raises(ValueError):
        st.flip_rate_interval([], reps=10, seed=0)


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
    assert stale["flip_ci_low"] <= 6 / 12 <= stale["flip_ci_high"] and stale["flip_ci_low"] > 0.0
    assert stale["no_answer_rate"] == 0.0 and stale["cap_rate"] == pytest.approx(1 / 12)
    assert stale["rank_correlation_b1"] is not None and stale["mean_first_divergence"] is not None
    # NEAR and FAR answer 9999 everywhere: every correct FRESH answer is lost -> delta -50 on 6 of 12 discordant
    assert g32["arms"]["near"]["delta"]["delta_points"] == pytest.approx(-50.0)
    assert g32["arms"]["far"]["delta"]["delta_points"] == pytest.approx(-50.0)
    assert g32["arms"]["near"]["band"] == "DEGRADES_C" == g32["arms"]["far"]["band"]
    assert g32["arms"]["near"]["flip_rate"] == 1.0 == g32["arms"]["far"]["flip_rate"]
    assert g32["controls"]["floor_passed"] and g32["controls"]["identity_passed"]
    assert g32["positive_controls_fire"] is True and g32["evaluable"] is True
    assert s["seal_readout"] is None or isinstance(s["seal_readout"], dict)
    assert (cfg.results_dir / "summary.json").exists()


def test_summary_marks_a_lag_unevaluable_when_a_positive_control_flips_too_few_answers(repo, seal_cfg):
    from tests.fake_task_swap import default_outcome

    def lazy_near(i, step, arm, gold):
        if arm == "near" and step == 1:
            # NEAR flips 5 of 12 answers, all of them FRESH's correct ones (even indices), so accuracy falls 41.7 points
            # and the old accuracy band reads DEGRADES_C; under control_min_flip_rate 0.5 the instrument did NOT fire
            return ("9999" if i % 2 == 0 and i % 12 < 10 else default_outcome(i, step, "fresh", gold)[0]), False
        return default_outcome(i, step, arm, gold)

    cfg = _run(repo, seal_cfg, outcome=lazy_near)
    s = st.summarize(cfg)
    g1, g32 = s["lags"]["local:step_1"], s["lags"]["local:step_32"]
    assert g1["arms"]["near"]["flip_rate"] == pytest.approx(5 / 12) and g1["positive_controls_fire"] is False
    # the band would have fired under the withdrawn rule: this is what makes the test bite on the flips rule
    assert g1["arms"]["near"]["band"] == "DEGRADES_C"
    assert g1["evaluable"] is False and g1["arms"]["stale"]["band"] is None and g1["arms"]["stale"]["delta"] is not None
    assert g1["arms"]["stale"]["flip_rate"] is not None                    # the descriptives stay
    assert g32["evaluable"] is True


def test_summary_refuses_a_record_hash_mismatch(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    p = cfg.results_dir / "readers" / "local_step_1" / "stale.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"answer": "5"', '"answer": "6"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        st.summarize(cfg)


def test_seal_readout_reads_the_four_clauses_on_flips(tmp_path):
    from lag_ladder.summarize_reader import PREDICTIONS
    (tmp_path / PREDICTIONS).mkdir(parents=True)
    (tmp_path / PREDICTIONS / "ownrun-task.json").write_text(json.dumps({"prediction": {"predictions": {
        "i_positive_controls_fire": {"readers": ["local:step_1", "local:step_32"]},
        "ii_flip_rate_at_first_lag": {"reader": "local:step_1", "at_least": 0.10},
        "iii_flip_rate_at_last_lag": {"reader": "local:step_32", "at_least": 0.10},
        "iv_flip_growth": {"first": "local:step_1", "last": "local:step_32"}}}}), encoding="utf-8")
    summary = {"lags": {
        "local:step_1": {"positive_controls_fire": True, "evaluable": True,
                         "arms": {"stale": {"flip_rate": 0.30, "flip_ci_low": 0.27, "flip_ci_high": 0.33}}},
        "local:step_32": {"positive_controls_fire": True, "evaluable": True,
                          "arms": {"stale": {"flip_rate": 0.86, "flip_ci_low": 0.84, "flip_ci_high": 0.88}}}}}
    c = st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]
    assert [c[k]["verdict"] for k in ("i", "ii", "iii", "iv")] == ["HELD"] * 4
    assert c["iv"]["intervals"] == {"local:step_1": [0.27, 0.33], "local:step_32": [0.84, 0.88]}
    # (ii) falls when the first lag reproduces the answer; (iv) falls when the intervals touch
    summary["lags"]["local:step_1"]["arms"]["stale"] = {"flip_rate": 0.05, "flip_ci_low": 0.03, "flip_ci_high": 0.85}
    c = st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]
    assert c["ii"]["verdict"] == "FALSIFIED" and c["iv"]["verdict"] == "FALSIFIED" and c["iii"]["verdict"] == "HELD"
    # an unevaluable lag withholds every clause that reads it
    summary["lags"]["local:step_32"]["positive_controls_fire"] = False
    summary["lags"]["local:step_32"]["evaluable"] = False
    c = st.seal_readout(summary, tmp_path, "ownrun-task")["clauses"]
    assert c["i"]["verdict"] == "FALSIFIED" and c["iii"]["verdict"] == "UNREAD" and c["iv"]["verdict"] == "UNREAD"



# --- 0035: two clauses the Instruct-initialised run seals on accuracy, which no earlier (C) seal could carry ---------

def _task_seal(root, predictions):
    (root / st.PREDICTIONS).mkdir(parents=True, exist_ok=True)
    (root / st.PREDICTIONS / "x.json").write_text(json.dumps({"prediction": {"predictions": predictions}}), encoding="utf-8")


def _lag(acc_fresh, delta, lo, hi, band="UNRESOLVED", evaluable=True):
    return {"evaluable": evaluable, "acc_fresh": acc_fresh, "positive_controls_fire": True,
            "arms": {"stale": {"flip_rate": 0.5, "flip_ci_low": 0.45, "flip_ci_high": 0.55,
                               "delta": {"delta_points": delta, "ci_low": lo, "ci_high": hi}, "band": band}}}


def test_fresh_accuracy_floor_clause_reads_acc_fresh_at_every_named_reader(tmp_path):
    """(v): FRESH greedy accuracy (points) at least a floor at every named reader — the Instruct baseline the probe
    read at 72.0 [65.5, 78.0]; UNREAD when a reader is withheld or missing."""
    _task_seal(tmp_path, {"v_fresh_accuracy_at_least": {"readers": ["local:step_1", "local:step_32"], "at_least_points": 50.0}})
    s = {"lags": {"local:step_1": _lag(71.2, -0.5, -2.0, 1.0), "local:step_32": _lag(68.0, -6.0, -8.5, -3.6)}}
    c = st.seal_readout(s, tmp_path, "x")["clauses"]["v"]
    assert c == {"readers": ["local:step_1", "local:step_32"], "at_least_points": 50.0,
                 "acc_fresh": {"local:step_1": 71.2, "local:step_32": 68.0}, "verdict": "HELD"}
    s["lags"]["local:step_32"]["acc_fresh"] = 49.9
    assert st.seal_readout(s, tmp_path, "x")["clauses"]["v"]["verdict"] == "FALSIFIED"
    s["lags"]["local:step_32"]["evaluable"] = False
    assert st.seal_readout(s, tmp_path, "x")["clauses"]["v"]["verdict"] == "UNREAD"
    del s["lags"]["local:step_32"]
    assert st.seal_readout(s, tmp_path, "x")["clauses"]["v"]["verdict"] == "UNREAD"


def test_degrades_at_lag_clause_reads_the_paired_delta_interval_entirely_below_zero(tmp_path):
    """(vi): at the named reader the STALE − FRESH accuracy interval lies entirely below zero (ci_high < 0); the band
    the summarizer assigned is reported beside it, not re-derived."""
    _task_seal(tmp_path, {"vi_degrades_at_lag": {"reader": "local:step_32"}})
    held = {"lags": {"local:step_32": _lag(68.0, -6.0, -8.5, -3.6, band="DEGRADES_C")}}
    c = st.seal_readout(held, tmp_path, "x")["clauses"]["vi"]
    assert c == {"reader": "local:step_32", "delta_points": -6.0, "interval": [-8.5, -3.6], "band": "DEGRADES_C",
                 "verdict": "HELD"}
    straddles = {"lags": {"local:step_32": _lag(68.0, -0.7, -2.0, 0.5)}}
    assert st.seal_readout(straddles, tmp_path, "x")["clauses"]["vi"]["verdict"] == "FALSIFIED"
    unevaluable = {"lags": {"local:step_32": _lag(68.0, -6.0, -8.5, -3.6, evaluable=False)}}
    assert st.seal_readout(unevaluable, tmp_path, "x")["clauses"]["vi"]["verdict"] == "UNREAD"
