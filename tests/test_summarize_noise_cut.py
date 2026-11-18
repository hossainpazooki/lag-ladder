"""summarize_noise_cut (ledger 0042): the read-out of the noise readers and the prefix cut from five summaries — the ratios and their factors,
the floor bounds, withheld rungs and rungs skipped with a failed unit, the seed spread, the cut ladder over 768, the
on-path mark and the replicate beside 0018 — on synthetic summaries in summarize_reader's shape; and which rungs were
skipped, read from run records in reader.run's shape built from the real planner on config/ownrun_cut_192.toml."""
import math

import pytest

from lag_ladder import REPO_ROOT, reader
from lag_ladder.config import load_reader_config
from lag_ladder.summarize_noise_cut import LAGS, ratio, readout, render, skipped_units
from lag_ladder.summarize_reader import _control_key


def _rung(label, texts, *, withheld=False, distance=1e-5, unit="u"):
    return {"label": label, "withheld": withheld, "rel_delta_all": distance, "rel_norm": None, "controls": unit,
            "texts": {t: {"b1": {"median_kl": v}, "at_floor": fl} for t, (v, fl) in texts.items()}}


def _summary(rungs, units=None):
    return {"rungs": rungs, "controls": units or {"u": {"passed": True, "evaluable_under": None,
                                                        "scrambled": {"median_kl": 0.12}, "far": None}}}


def _five(T_r=1e-2, T_h=1e-4, N_r=1e-3, N_h=1e-6, *, nh_floor=False, cut=2e-4, seeds=(1e-3, 2e-3, 4e-3)):
    held = {"corpus": (T_h, False), "on-path": (T_h * 2, False)}
    rep = _summary([_rung(f"ownrun-rollout-lag-{g}", {"rollout": (T_r, False)}) for g in LAGS]
                   + [_rung(f"ownrun-lag-{g}", held) for g in LAGS])
    nrs = []
    for g in LAGS:
        for s in (0, 1, 2):
            if s and g not in (1, 32):
                continue
            nrs.append(_rung(f"ownrun-rollout-noise-lag-{g}-s{s}", {"rollout": (seeds[s] if g in (1, 32) else N_r, False)},
                             distance=1e-5 * (1 + 1e-6)))
            nrs.append(_rung(f"ownrun-noise-lag-{g}-s{s}", {"corpus": (N_h, nh_floor), "on-path": (N_h, nh_floor)}))
    cuts = {c: _summary([_rung(f"ownrun-lag-{g}", {"corpus": (cut, False), "on-path": (cut, False)}) for g in LAGS])
            for c in ("cut_prompt", "cut_192", "cut_384")}
    return {"replicate": rep, "noise": _summary(nrs), **cuts}


def test_the_identity_factors_multiply_to_the_whole_ratio():
    r = readout(_five(seeds=(1e-3, 1e-3, 1e-3)))
    row = r["lags"]["4"]
    assert math.isclose(row["T_rollout_over_N_rollout"]["value"], 10.0)
    h = row["heldout"]["corpus"]
    assert math.isclose(h["N_rollout_over_N_heldout"]["value"], 1e3)
    assert math.isclose(h["T_heldout_over_N_heldout"]["value"], 1e2)
    assert math.isclose(h["T_rollout_over_T_heldout"]["value"], 1e2)
    assert math.isclose(h["factor_product"], h["T_rollout_over_T_heldout"]["value"])
    assert math.isclose(row["distance"]["noise_over_real"], 1 + 1e-6)


def test_a_floor_operand_makes_a_bound_and_a_withheld_one_no_value():
    a, b = {"value": 4.0, "at_floor": False, "withheld": False}, {"value": 2.0, "at_floor": True, "withheld": False}
    assert ratio(a, b) == {"value": 2.0, "bound": "lower", "why": None}
    assert ratio(b, a)["bound"] == "upper"
    assert ratio(b, b)["bound"] == "unread"
    w = {"value": None, "at_floor": None, "withheld": True}
    assert ratio(a, w) == {"value": None, "bound": None, "why": "withheld"}
    r = readout(_five(nh_floor=True))
    assert r["lags"]["1"]["heldout"]["corpus"]["N_rollout_over_N_heldout"]["bound"] == "lower"
    assert r["lags"]["1"]["heldout"]["corpus"]["T_heldout_over_N_heldout"]["bound"] == "lower"
    assert r["lags"]["1"]["T_rollout_over_N_rollout"]["bound"] is None


def test_seed_spread_and_the_cut_ladder():
    r = readout(_five(seeds=(1e-3, 2e-3, 4e-3), cut=2e-4))
    assert math.isclose(r["seeds"]["1"]["rollout"]["max_over_min"], 4.0)
    assert r["seeds"]["32"]["corpus"]["max_over_min"] == 1.0
    c = r["cuts"]["8"]["corpus"]
    assert math.isclose(c["cut_192"]["over_768"]["value"], 2.0)
    assert math.isclose(c["cut_prompt"]["T_rollout_over_cut"]["value"], 50.0)
    assert "prefix cut" in render(r) and "UNREAD" in render(r)


def test_refuses_a_missing_summary_or_rung():
    s = _five()
    del s["cut_384"]
    with pytest.raises(ValueError, match="summaries missing"):
        readout(s)
    s = _five()
    s["noise"]["rungs"] = [e for e in s["noise"]["rungs"] if e["label"] != "ownrun-noise-lag-16-s0"]
    with pytest.raises(ValueError, match="no rung ownrun-noise-lag-16-s0"):
        readout(s)


def test_withheld_rungs_and_failed_units_are_reported():
    s = _five()
    s["noise"]["rungs"][0]["withheld"] = True
    s["noise"]["controls"] = {"u": {"passed": False, "evaluable_under": None, "scrambled": {"median_kl": 0.05}, "far": None}}
    r = readout(s)
    assert r["lags"]["1"]["N_rollout"]["value"] is None
    assert r["lags"]["1"]["T_rollout_over_N_rollout"]["why"] == "withheld"
    assert "units withheld" in render(r)


def _cut_units_with_lag_4_failed(*, passed):
    """cut_192 with one control unit per reader, as on the held-out pair; the unit of lag 4 missed scrambled
    (or, with passed=True, did not) and its rung is dropped from the summary as the driver skips it."""
    s = _five()
    s["cut_192"]["rungs"] = [_rung(f"ownrun-lag-{g}", {"corpus": (2e-4, False), "on-path": (2e-4, False)}, unit=f"r{g}")
                             for g in LAGS if g != 4]
    s["cut_192"]["controls"] = {f"r{g}": {"passed": g != 4 or passed, "evaluable_under": None,
                                          "scrambled": {"median_kl": 0.05 if g == 4 else 0.12}, "far": None} for g in LAGS}
    return s, {"cut_192": {f"ownrun-lag-{g}": f"r{g}" for g in LAGS}}


def test_a_rung_skipped_with_its_failed_unit_reads_withheld_and_the_rest_is_read():
    s, units = _cut_units_with_lag_4_failed(passed=False)
    r = readout(s, units=units)
    c = r["cuts"]["4"]["corpus"]["cut_192"]
    assert c["reading"]["value"] is None and c["reading"]["withheld"] and c["reading"]["skipped"]
    assert c["reading"]["unit"] == "r4" and c["reading"]["scrambled_median"] == 0.05
    assert c["over_768"] == {"value": None, "bound": None, "why": "withheld"}
    assert math.isclose(r["cuts"]["8"]["corpus"]["cut_192"]["over_768"]["value"], 2.0)
    assert r["controls"]["cut_192"]["r4"] == {"passed": False, "evaluable_under": None, "scrambled_median": 0.05,
                                              "far_median": None}
    assert "cut_192: units withheld or evaluable under far: r4 (passed False, scrambled 0.05" in render(r)


def test_an_absent_rung_is_refused_when_its_unit_passed_or_without_the_units():
    s, units = _cut_units_with_lag_4_failed(passed=True)
    with pytest.raises(ValueError, match="no rung ownrun-lag-4 in the summary, though its unit r4 passed"):
        readout(s, units=units)
    s, units = _cut_units_with_lag_4_failed(passed=False)
    with pytest.raises(ValueError, match=r"cut_192: no rung ownrun-lag-4 in the summary$"):
        readout(s)
    del s["cut_192"]["controls"]["r4"]
    with pytest.raises(ValueError, match="its unit r4 has no controls"):
        readout(s, units=units)


def _cut_192():
    cfg = load_reader_config(REPO_ROOT / "config" / "ownrun_cut_192.toml", REPO_ROOT)
    r4 = next(r for r in cfg.rungs if r.label == "ownrun-lag-4")
    return cfg, _control_key(r4.reader, r4.pair, cfg)


def _record(cfg, unev=(), *, upto=None, **over):
    """A run record as reader.run writes it, its jobs from the real planner in plan order, the rungs of the units it
    names UNEVALUABLE skipped as the driver skips them; upto keeps only the first jobs, as a stop partway leaves them."""
    jobs = [{"kind": j.kind, "reader": j.reader, "label": j.label, "pair": j.pair, "seq": j.seq, "text": j.text,
             "done": True} for j in reader.plan(cfg).jobs if not (j.kind == "rung" and reader.control_key(j, cfg) in unev)]
    rec = {"finished": "set at the end", "halted": None, "unevaluable_readers": {u: "UNEVALUABLE: scrambled" for u in unev},
           "jobs": jobs[:upto]}
    rec.update(over)
    return rec


def _last_units_scrambled(cfg, n):
    """The index just past the n-th scrambled job of the plan's last control unit (cut configs: step_32)."""
    plan = reader.plan(cfg).jobs
    last = reader.control_key(plan[-1], cfg)
    idx = [i for i, j in enumerate(plan) if j.kind == "scrambled" and reader.control_key(j, cfg) == last]
    return last, idx[n - 1] + 1


def test_the_skipped_rungs_are_read_from_the_run_record_with_the_drivers_unit_key():
    cfg, u4 = _cut_192()
    assert {reader.control_key(j, cfg) for j in reader.plan(cfg).jobs if j.label == "ownrun-lag-4"} == {u4}
    assert skipped_units(cfg, _record(cfg), name="cut_192") == {}
    units = skipped_units(cfg, _record(cfg, (u4,)), name="cut_192")
    assert units == {"ownrun-lag-4": u4}
    s = _five()                                       # the summary as the driver leaves it: no lag-4 rung, u4 failed
    s["cut_192"]["rungs"] = [e for e in s["cut_192"]["rungs"] if e["label"] != "ownrun-lag-4"]
    s["cut_192"]["controls"][u4] = {"passed": False, "evaluable_under": None, "scrambled": {"median_kl": 0.05}, "far": None}
    c = readout(s, units={"cut_192": units})["cuts"]["4"]["on-path"]["cut_192"]["reading"]
    assert c["skipped"] and c["unit"] == u4 and c["scrambled_median"] == 0.05


def test_a_halted_run_is_refused():
    cfg, _ = _cut_192()
    last, k = _last_units_scrambled(cfg, 1)
    rec = _record(cfg, upto=k - 1, halted="float16 floor failed: local:step_32 (corpus) median mean-KL = 2.000e-04 > 0.0001")
    with pytest.raises(ValueError, match="cut_192: the run halted: float16 floor failed"):
        skipped_units(cfg, rec, name="cut_192")


def test_a_run_that_did_not_finish_is_refused():
    cfg, _ = _cut_192()
    last, k = _last_units_scrambled(cfg, 5)          # a crash in the last unit's scrambled block, record not closed
    with pytest.raises(ValueError, match="cut_192: the run did not finish"):
        skipped_units(cfg, _record(cfg, upto=k, finished=None), name="cut_192")


def test_a_run_cut_short_with_limit_is_refused():
    cfg, _ = _cut_192()
    last, k = _last_units_scrambled(cfg, 5)          # --limit stops the loop and still sets finished
    with pytest.raises(ValueError, match=r"cut_192: the run record's \d+ jobs are not the plan's \d+ jobs"):
        skipped_units(cfg, _record(cfg, upto=k), name="cut_192")


def test_a_record_whose_named_units_do_not_match_its_jobs_is_refused():
    cfg, u4 = _cut_192()
    rec = _record(cfg)                                # named UNEVALUABLE, but its rungs ran
    rec["unevaluable_readers"] = {u4: "UNEVALUABLE: scrambled"}
    with pytest.raises(ValueError, match="jobs are not the plan's"):
        skipped_units(cfg, rec, name="cut_192")
    rec = _record(cfg, (u4,))
    rec["jobs"][-1]["done"] = False
    with pytest.raises(ValueError, match="jobs are not the plan's"):
        skipped_units(cfg, rec, name="cut_192")
    rec = _record(cfg)
    rec["unevaluable_readers"] = {"local:elsewhere @ another-pair": "UNEVALUABLE: scrambled"}
    with pytest.raises(ValueError, match="no configured rung has"):
        skipped_units(cfg, rec, name="cut_192")


def test_the_on_path_rows_carry_the_mark():
    r = readout(_five())
    assert r["lags"]["1"]["heldout"]["corpus"]["shared_tokens"] is True
    assert r["lags"]["1"]["heldout"]["on-path"]["shared_tokens"] is False
    assert [r["seeds"]["1"][t]["shared_tokens"] for t in ("rollout", "corpus", "on-path")] == [True, True, False]
    lines = render(r).splitlines()
    marked = [ln for ln in lines if "tokens may differ" in ln]
    assert len([ln for ln in marked if ln.lstrip().startswith("on-path")]) == len(LAGS)
    seeds = [ln for ln in marked if ln.startswith("seeds at g=")]
    assert len(seeds) == 2 and all(ln.endswith("tokens may differ]") and ln.count("tokens may differ") == 1 for ln in seeds)
    assert len(marked) == len(LAGS) + 2


def test_the_replicate_beside_0018():
    s = _five(T_r=2e-2)
    ref = _summary([_rung(f"ownrun-rollout-lag-{g}", {"rollout": (1e-2, False)}) for g in LAGS]
                   + [_rung(f"ownrun-lag-{g}", {"corpus": (1e-4, False), "on-path": (2e-4, False)}) for g in LAGS])
    for e in ref["rungs"]:
        del e["controls"]                                  # 0018's summary predates the per-rung unit key
    r = readout(s, ref)
    assert math.isclose(r["replicate_vs_0018"]["ownrun-rollout-lag-2/rollout"]["ratio"], 2.0)
    assert math.isclose(r["replicate_vs_0018"]["ownrun-lag-2/corpus"]["ratio"], 1.0)
