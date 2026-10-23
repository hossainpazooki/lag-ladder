"""The reader summarizer (contract interface 4) recomputes every statistic from the report.json arrays and
refuses on any disagreement; it withholds rungs whose reader's controls failed and marks rungs at the float16
floor. The fake scorer's cells are known functions of the stand-in distance, so (B1)-(B4) are checkable."""
import json
import math

import pytest

from lag_ladder import REPO_ROOT, reader
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.summarize_reader import summarize
from tests.fake_instrument import FakeInstrument, default_kl
from tests.test_reader import EXPECTED_KL, _setup_reader

BANDS = ("HOLDS_B", "DEGRADES_B", "UNRESOLVED")


def _ran(repo, seal_cfg, kl=None):
    cfg, tools = _setup_reader(repo, seal_cfg)
    reader.run(cfg, repo, runner=FakeInstrument(kl=kl or default_kl), seal_cfg=seal_cfg, tools_dir=tools)
    return cfg


def _cells(s: dict) -> list[dict]:
    """One flat cell per rung and text: the rung's per-text statistics with its label, text and withheld flag."""
    return [{"label": e["label"], "text": t, "withheld": e["withheld"], **st}
            for e in s["rungs"] for t, st in e["texts"].items()]


def _entries(s: dict) -> dict:
    """(label, text) -> rung cell."""
    return {(c["label"], c["text"]): c for c in _cells(s)}


def _floats(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _floats(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _floats(v)
    elif isinstance(x, float):
        yield x


def _carries(entry: dict, value: float, rel=1e-9) -> bool:
    return any(f == pytest.approx(value, rel=rel) for f in _floats(entry))


def test_summary_bands_per_text_and_recomputes_b1_to_b4(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    s = summarize(cfg)
    by = _entries(s)
    for text in ("corpus", "on-path"):
        assert by[("backbone-1e-3", text)]["band"] == "HOLDS_B"       # 4e-4 <= 1e-3
        assert by[("olmo-sft-to-dpo", text)]["band"] == "DEGRADES_B"  # 0.16 >= 1e-1
        assert by[("backbone-1e-1", text)]["band"] == "DEGRADES_B"
        assert by[("qwen-base-to-instruct", text)]["band"] == "DEGRADES_B"
        for label, kl in EXPECTED_KL.items():
            e = by[(label, text)]
            # (B1): the fake writes a constant kl_per_token, so mean, median, p90 and max all read kl
            assert _carries(e, kl), f"{label}/{text} does not carry the recomputed (B1) {kl}"
            # (B2)/(B3): logprob_fresh - logprob_stale = kl per token; two continuation tokens per sequence
            assert _carries(e, 2 * kl) or kl == 0.0, f"{label}/{text} does not carry the per-sequence log weight"
            # (B4): the fake's argmaxes agree exactly when mean KL <= 1e-3
            assert _carries(e, 1.0 if kl <= 1e-3 else 0.0), f"{label}/{text} does not carry argmax agreement"
    assert (cfg.results_dir / "summary.json").exists()


def test_summary_places_rungs_on_the_distance_axis(repo, seal_cfg):
    """Each rung's position comes from the distance ladder's distance.json (never from run.json); the slope is
    fit over the backbone, every real rung names its nearest backbone rung of the same pair, and the sealed
    clauses are read out (none here: the test seal names no clauses)."""
    cfg = _ran(repo, seal_cfg)
    s = summarize(cfg)
    by = {e["label"]: e for e in s["rungs"]}
    noise = [e for e in s["rungs"] if e["kind"] == "noise"]
    assert noise and all(e["rel_norm"] is not None and e["distance_source"] is not None for e in noise)
    for text in ("corpus", "on-path"):
        fit = s["slope_fit"][text]
        assert fit["slope"] is not None and len(fit["points"]) == len(noise) and fit["sealed_rungs"] is False
    real = [e for e in s["rungs"] if e["kind"] == "real" and e["rel_delta_all"]]
    assert real and all(e["distance_source"].endswith("distance.json") for e in real)
    matched = [e for e in real if e["matched_backbone"]]
    assert matched and all(by[e["matched_backbone"]]["pair"] == e["pair"] for e in matched)
    assert all(e["matched_backbone"] is None for e in real if not any(n["pair"] == e["pair"] for n in noise))
    assert s["seal_readout"] is not None and s["seal_readout"]["clauses"] == {}


def test_summary_refuses_a_tampered_kl_array(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    report = sorted(cfg.results_dir.rglob("report.json"))[0]
    rec = json.loads(report.read_text(encoding="utf-8"))
    rec["kl_per_token"] = [v * 2 for v in rec["kl_per_token"]] or [1.0]
    report.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        summarize(cfg)
    assert not (cfg.results_dir / "summary.json").exists(), "a refusal must not leave a summary behind"


def test_summary_refuses_a_continuation_hash_mismatch(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    report = sorted(cfg.results_dir.rglob("report.json"))[-1]
    rec = json.loads(report.read_text(encoding="utf-8"))
    rec["continuation_sha256"] = "0" * 64
    report.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        summarize(cfg)


def test_summary_refuses_a_missing_report(repo, seal_cfg):
    cfg = _ran(repo, seal_cfg)
    sorted(cfg.results_dir.rglob("report.json"))[0].unlink()
    with pytest.raises((ValueError, RuntimeError)):
        summarize(cfg)


def test_summary_withholds_a_rung_whose_reader_failed_scrambled(repo, seal_cfg):
    def blind_qwen(fm, sm, fs, ss, live):
        if not live and fs != ss and fm["pair"] == "qwen2.5-1.5b-to-instruct":
            return 0.0, 0.0
        return default_kl(fm, sm, fs, ss, live)

    cfg = _ran(repo, seal_cfg, kl=blind_qwen)
    s = summarize(cfg)
    for e in _cells(s):
        if e["label"] == "qwen-base-to-instruct":
            assert e["band"] not in BANDS, "a rung whose reader failed scrambled must be withheld, not banded"
    banded = {e["label"] for e in _cells(s) if e["band"] in BANDS}
    assert "backbone-1e-3" in banded and "qwen-base-to-instruct" not in banded


def test_summary_marks_a_rung_at_the_floor(repo, seal_cfg):
    def near_floor(fm, sm, fs, ss, live):
        kl, mld = default_kl(fm, sm, fs, ss, live)
        if not live and kl == pytest.approx(4e-4):
            return 1.5e-6, 1e-7       # backbone-1e-3, within 2x the measured floor of 1e-6
        return kl, mld

    cfg = _ran(repo, seal_cfg, kl=near_floor)
    s = summarize(cfg)
    by = _entries(s)
    for text in ("corpus", "on-path"):
        e = by[("backbone-1e-3", text)]
        assert e["band"] == "HOLDS_B"
        assert any("floor" in k and v is True for k, v in e.items()), f"backbone-1e-3/{text} is not marked at the floor"
        far = by[("olmo-sft-to-dpo", text)]
        assert not any("floor" in k and v is True for k, v in far.items()), "a rung far above the floor was marked"


def test_summary_resolves_a_run_machine_token_path_by_its_tail(repo, seal_cfg):
    """Reports pulled from a pod name the token file by that machine's absolute path; the summarizer finds
    the file here by its trailing components and still holds it to the recorded sha256."""
    cfg = _ran(repo, seal_cfg)
    corpus = [p for p in sorted(cfg.results_dir.rglob("report.json"))
              if json.loads(p.read_text(encoding="utf-8"))["text"] == "corpus"]
    assert corpus
    for report in corpus:
        rec = json.loads(report.read_text(encoding="utf-8"))
        rec["tokens_file"] = "/workspace/elsewhere/" + "/".join(rec["tokens_file"].split("/")[-3:])
        report.write_text(json.dumps(rec), encoding="utf-8")
    _rename_reports_in_run_record(cfg)
    summarize(cfg)
    rec = json.loads(corpus[0].read_text(encoding="utf-8"))
    rec["inputs_sha256"] = {k: ("0" * 64 if k.endswith(".npy") else v) for k, v in rec["inputs_sha256"].items()}
    corpus[0].write_text(json.dumps(rec), encoding="utf-8")
    _rename_reports_in_run_record(cfg)
    with pytest.raises((ValueError, RuntimeError)):
        summarize(cfg)


def _rename_reports_in_run_record(cfg):
    """Re-point run.json at the rewritten report files (the test edits the path a report names, not its data)."""
    rec_path = cfg.results_dir / "run.json"
    run_rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for j in run_rec["jobs"]:
        f = cfg.results_dir / j["dir"] / "report.json"
        if f.exists():
            j["report_sha256"] = sha256_file_bytes(f)
    rec_path.write_text(json.dumps(run_rec), encoding="utf-8")


def test_summary_reads_a_rollout_rung_on_its_own_text_and_verifies_the_stored_continuation(repo, seal_cfg):
    from tests.test_reader import _setup_rollout
    cfg, tools = _setup_rollout(repo, seal_cfg)
    reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)
    s = summarize(cfg)
    e = {x["label"]: x for x in s["rungs"]}["rollout-1e-3"]
    assert set(e["texts"]) == {"rollout"} and e["texts"]["rollout"]["n_seqs"] == 4
    assert e["texts"]["rollout"]["band"] in BANDS
    # a rollout report whose stored continuation is not the token file's own is refused like a corpus one
    rep = cfg.results_dir / "rungs" / "rollout-1e-3" / "rollout" / "seq0" / "report.json"
    rec = json.loads(rep.read_text(encoding="utf-8"))
    rec["continuation_tokens"] = [t + 1 for t in rec["continuation_tokens"]]
    from lag_ladder.summarize_reader import continuation_sha256
    rec["continuation_sha256"] = continuation_sha256(rec["continuation_tokens"])
    rep.write_text(json.dumps(rec), encoding="utf-8")
    _rename_reports_in_run_record(cfg)
    with pytest.raises((ValueError, RuntimeError), match="not tokens"):
        summarize(cfg)


def test_seal_readout_fits_the_slope_only_over_texts_the_sealed_rungs_carry(tmp_path):
    """A §14-style seal lists corpus, on-path and rollout texts but seals its slope over corpus lag rungs, which
    carry no rollout column. The missing rollout slope is not a withheld input: the clause reads HELD/FALSIFIED
    from the texts those rungs carry (the own-run summary of 2026-10-14 printed UNREAD here)."""
    from lag_ladder.summarize_reader import PREDICTIONS, seal_readout
    (tmp_path / PREDICTIONS).mkdir(parents=True)
    (tmp_path / PREDICTIONS / "x.json").write_text(json.dumps({"prediction": {
        "texts": ["corpus", "on-path", "rollout"],
        "predictions": {"i_backbone_quadratic": {"rungs": ["lag-1", "lag-2"], "slope_in": [1.5, 2.5]}}}}), encoding="utf-8")
    summary = {"rule": {"texts": ["corpus", "on-path", "rollout"], "holds_max": 1e-3},
               "rungs": [{"label": "lag-1", "texts": {"corpus": {}, "on-path": {}}},
                         {"label": "lag-2", "texts": {"corpus": {}, "on-path": {}}},
                         {"label": "rollout-lag-1", "texts": {"rollout": {}}}],
               "slope_fit": {"corpus": {"slope": 2.26}, "on-path": {"slope": 2.19}, "rollout": {"slope": None}}}
    c = seal_readout(summary, tmp_path, "x")["clauses"]["i"]
    assert c["verdict"] == "HELD" and set(c["slope"]) == {"corpus", "on-path"}
    summary["slope_fit"]["on-path"]["slope"] = 1.2
    assert seal_readout(summary, tmp_path, "x")["clauses"]["i"]["verdict"] == "FALSIFIED"
    summary["slope_fit"]["on-path"]["slope"] = None
    assert seal_readout(summary, tmp_path, "x")["clauses"]["i"]["verdict"] == "UNREAD"


def test_summary_judges_a_rung_against_the_controls_of_its_own_pair_under_per_pair(repo, seal_cfg):
    from tests.test_reader import _setup_per_pair
    cfg, tools = _setup_per_pair(repo, seal_cfg)
    reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)
    s = summarize(cfg)
    ctl = s["controls"]
    roll_key = [k for k in ctl if "ownrun-rollout" in k and "local:noise-0.001" in k]
    held_key = [k for k in ctl if "olmo2-1b-rlvr1" in k and "local:noise-0.001" in k]
    assert len(roll_key) == 1 and len(held_key) == 1
    rc = ctl[roll_key[0]]
    assert set(rc["floor"]["median_kl_by_text"]) == {"rollout"} and rc["scrambled"]["passed"] and rc["passed"]
    assert set(ctl[held_key[0]]["floor"]["median_kl_by_text"]) == {"corpus", "on-path"}
    e = {x["label"]: x for x in s["rungs"]}
    assert e["rollout-1e-3"]["controls_passed"] is True and e["rollout-1e-3"]["controls"] == roll_key[0]


def test_summary_judges_the_far_control_and_reads_each_rung_over_its_controls(repo, seal_cfg):
    from tests.test_reader import _setup_far
    cfg, tools = _setup_far(repo, seal_cfg)
    reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)
    s = summarize(cfg)
    roll_key = [k for k in s["controls"] if "ownrun-rollout" in k and "local:noise-0.001" in k][0]
    rc = s["controls"][roll_key]
    assert rc["far"]["passed"] and rc["far"]["median_kl"] == pytest.approx(2.0) and rc["passed"]
    assert rc["far_over_scrambled"] == pytest.approx(rc["far"]["median_kl"] / rc["scrambled"]["median_kl"])
    held_key = [k for k in s["controls"] if "olmo2-1b-rlvr1" in k and "local:noise-0.001" in k][0]
    assert s["controls"][held_key]["far"] is None and s["controls"][held_key]["passed"]
    e = {x["label"]: x for x in s["rungs"]}["rollout-1e-3"]["texts"]["rollout"]
    assert e["over_scrambled"] == pytest.approx(e["b1"]["median_kl"] / rc["scrambled"]["median_kl"])
    assert e["over_far"] == pytest.approx(e["b1"]["median_kl"] / rc["far"]["median_kl"])
    # a rung on a unit without a far control carries no far ratio
    other = [x for x in s["rungs"] if x["pair"] == "olmo2-1b-rlvr1" and not x["withheld"]][0]
    assert other["texts"]["corpus"]["over_far"] is None and other["texts"]["corpus"]["over_scrambled"] is not None


def test_seal_readout_reads_the_far_clauses_from_the_control_units_and_the_rungs(tmp_path):
    """0022: (i) far over scrambled >= a factor on every unit the seal names; (ii) every named rung's median over
    its unit's far median <= a bound, on the named text. Withheld or missing inputs read UNREAD."""
    from lag_ladder.summarize_reader import PREDICTIONS, seal_readout
    (tmp_path / PREDICTIONS).mkdir(parents=True)
    (tmp_path / PREDICTIONS / "x.json").write_text(json.dumps({"prediction": {
        "texts": ["rollout"],
        "predictions": {"i_far_over_scrambled": {"units": ["r1 @ roll", "r2 @ roll"], "at_least": 2.0},
                        "ii_rollout_over_far": {"rungs": ["lag-1", "lag-2"], "text": "rollout", "at_most": 0.5}}}}),
        encoding="utf-8")
    summary = {"rule": {"texts": ["corpus", "on-path"], "holds_max": 1e-3},
               "controls": {"r1 @ roll": {"far_over_scrambled": 3.0}, "r2 @ roll": {"far_over_scrambled": 2.5}},
               "rungs": [{"label": "lag-1", "withheld": False, "texts": {"rollout": {"over_far": 0.1}}},
                         {"label": "lag-2", "withheld": False, "texts": {"rollout": {"over_far": 0.4}}}],
               "slope_fit": {}}
    c = seal_readout(summary, tmp_path, "x")["clauses"]
    assert c["i"]["verdict"] == "HELD" and c["i"]["far_over_scrambled"] == {"r1 @ roll": 3.0, "r2 @ roll": 2.5}
    assert c["ii"]["verdict"] == "HELD" and c["ii"]["over_far"] == {"lag-1": 0.1, "lag-2": 0.4}
    summary["controls"]["r2 @ roll"]["far_over_scrambled"] = 1.5
    summary["rungs"][1]["texts"]["rollout"]["over_far"] = 0.7
    c = seal_readout(summary, tmp_path, "x")["clauses"]
    assert c["i"]["verdict"] == "FALSIFIED" and c["ii"]["verdict"] == "FALSIFIED"
    summary["controls"]["r2 @ roll"]["far_over_scrambled"] = None
    summary["rungs"][1]["withheld"] = True
    c = seal_readout(summary, tmp_path, "x")["clauses"]
    assert c["i"]["verdict"] == "UNREAD" and c["ii"]["verdict"] == "UNREAD"


# --- 0028: the second seed's replication clauses, read against the comparators the seal carries -------------------

def test_summary_marks_a_unit_evaluable_under_far_when_its_scrambled_misses(repo, seal_cfg):
    from tests.test_reader import _setup_evaluable_under_far, near_misses_far_clears
    cfg, tools = _setup_evaluable_under_far(repo, seal_cfg)
    reader.run(cfg, repo, runner=FakeInstrument(kl=near_misses_far_clears), seal_cfg=seal_cfg, tools_dir=tools)
    s = summarize(cfg)
    c = s["controls"]["local:noise-0.001 @ ownrun-rollout"]
    assert c["scrambled"]["passed"] is False and c["far"]["passed"] is True
    assert c["passed"] is True and c["evaluable_under"] == "far"
    e = {x["label"]: x for x in s["rungs"]}["rollout-1e-3"]
    assert e["withheld"] is False and e["texts"]["rollout"]["b1"]["median_kl"] is not None
    held = [k for k in s["controls"] if "olmo2-1b-rlvr1" in k][0]
    assert s["controls"][held]["evaluable_under"] is None and s["controls"][held]["passed"]


def _replication_fixture(tmp_path, reference_at_1=4.0e-3):
    from lag_ladder.summarize_reader import PREDICTIONS
    (tmp_path / PREDICTIONS).mkdir(parents=True, exist_ok=True)
    pred = {"texts": ["corpus", "on-path", "rollout"], "rule": {"holds_max": 1e-3},
            "predictions": {
                "i_heldout_holds_at_every_lag": {"rungs": ["lag-1", "lag-2"], "texts": ["corpus", "on-path"]},
                "ii_rollout_replicates_seed1": {"rungs": {"1": "roll-1", "2": "roll-2"}, "text": "rollout",
                                                "within_factor": 3.0, "reference": {"1": reference_at_1, "2": 1.2e-2},
                                                "and_at_lag_1_at_least": 1e-3},
                "iii_rollout_over_heldout": {"pairs_of_rungs": {"1": ["roll-1", "lag-1"], "2": ["roll-2", "lag-2"]},
                                             "text": "rollout", "heldout_texts": ["corpus", "on-path"], "at_least_factor": 1000.0},
                "iv_controls_as_0020": {"units": ["r1 @ held", "r1 @ roll", "r2 @ held", "r2 @ roll"]}}}
    (tmp_path / PREDICTIONS / "x.json").write_text(json.dumps({"prediction": pred}), encoding="utf-8")

    def held(c, o):
        return {"corpus": {"b1": {"median_kl": c}, "band": "HOLDS_B", "at_floor": False},
                "on-path": {"b1": {"median_kl": o}, "band": "HOLDS_B", "at_floor": False}}

    def roll(m):
        return {"rollout": {"b1": {"median_kl": m}, "band": "UNRESOLVED", "at_floor": False}}

    return {"rule": {"texts": ["corpus", "on-path"], "holds_max": 1e-3},
            "controls": {"r1 @ held": {"passed": True, "evaluable_under": None},
                         "r1 @ roll": {"passed": True, "evaluable_under": "far"},
                         "r2 @ held": {"passed": True, "evaluable_under": None},
                         "r2 @ roll": {"passed": True, "evaluable_under": None}},
            "rungs": [{"label": "lag-1", "withheld": False, "texts": held(2e-7, 1e-7)},
                      {"label": "lag-2", "withheld": False, "texts": held(5e-7, 3e-7)},
                      {"label": "roll-1", "withheld": False, "texts": roll(5e-3)},
                      {"label": "roll-2", "withheld": False, "texts": roll(1.5e-2)}],
            "slope_fit": {}}


def test_seal_readout_reads_the_replication_clauses_against_the_seals_stored_comparators(tmp_path):
    """0028: (i-holds) HOLDS_B and off the floor on the named rungs and texts; (ii) each rollout median within a
    factor of the reference the seal carries, and a floor at one lag; (iii) rollout over each held-out text at
    least a factor; (iv) every named control unit passed, with the far rescue reported."""
    from lag_ladder.summarize_reader import seal_readout
    summary = _replication_fixture(tmp_path)
    c = seal_readout(summary, tmp_path, "x")["clauses"]
    assert {k: v["verdict"] for k, v in c.items()} == {"i-holds": "HELD", "ii": "HELD", "iii": "HELD", "iv": "HELD"}
    assert c["i-holds"]["holds_max"] == 1e-3 and c["i-holds"]["median_kl"]["lag-2"]["on-path"] == 3e-7
    assert c["ii"]["median_kl"] == {"1": 5e-3, "2": 1.5e-2}
    assert c["ii"]["accept"]["1"] == pytest.approx([4e-3 / 3, 4e-3 * 3]) and c["ii"]["at_lag_1_at_least"] == 1e-3
    assert c["iii"]["ratio"]["1"]["corpus"] == pytest.approx(5e-3 / 2e-7) and c["iii"]["at_least_factor"] == 1000.0
    assert c["iv"]["units"]["r1 @ roll"] == {"passed": True, "evaluable_under": "far"}


def test_each_replication_clause_falsifies_on_its_own_input_and_reads_unread_when_withheld(tmp_path):
    import copy
    from lag_ladder.summarize_reader import seal_readout

    def verdicts(s):
        return {k: v["verdict"] for k, v in seal_readout(s, tmp_path, "x")["clauses"].items()}

    base = _replication_fixture(tmp_path)
    s = copy.deepcopy(base); s["rungs"][0]["texts"]["corpus"]["b1"]["median_kl"] = 2e-3
    assert verdicts(s)["i-holds"] == "FALSIFIED"
    s = copy.deepcopy(base); s["rungs"][1]["texts"]["on-path"]["at_floor"] = True
    assert verdicts(s)["i-holds"] == "FALSIFIED"
    s = copy.deepcopy(base); s["rungs"][2]["texts"]["rollout"]["b1"]["median_kl"] = 1.3e-2     # above 3x 4e-3
    v = verdicts(s); assert v["ii"] == "FALSIFIED" and v["iii"] == "HELD"
    s = copy.deepcopy(base); s["rungs"][3]["texts"]["rollout"]["b1"]["median_kl"] = 3.9e-3     # under 1.2e-2 / 3
    assert verdicts(s)["ii"] == "FALSIFIED"
    s = copy.deepcopy(base); s["rungs"][2]["texts"]["rollout"]["b1"]["median_kl"] = 1.9e-4     # 950x corpus at lag 1
    v = verdicts(s); assert v["iii"] == "FALSIFIED"
    s = copy.deepcopy(base); s["controls"]["r2 @ roll"]["passed"] = False
    assert verdicts(s)["iv"] == "FALSIFIED"
    s = copy.deepcopy(base); del s["controls"]["r2 @ held"]
    assert verdicts(s)["iv"] == "UNREAD"
    s = copy.deepcopy(base); s["rungs"][3]["withheld"] = True
    v = verdicts(s); assert v["ii"] == "UNREAD" and v["iii"] == "UNREAD" and v["i-holds"] == "HELD"
    s = copy.deepcopy(base); s["rungs"][1]["withheld"] = True
    assert verdicts(s)["i-holds"] == "UNREAD"
    # the g = 1 floor is its own half: within the band but under 1e-3 falsifies (ii)
    base2 = _replication_fixture(tmp_path, reference_at_1=2.4e-3)
    s = copy.deepcopy(base2); s["rungs"][2]["texts"]["rollout"]["b1"]["median_kl"] = 9e-4          # in [8e-4, 7.2e-3]
    assert verdicts(s)["ii"] == "FALSIFIED"
    s["rungs"][2]["texts"]["rollout"]["b1"]["median_kl"] = 1.1e-3
    assert verdicts(s)["ii"] == "HELD"


# --- 0030: a fourth text's reading placed against two earlier readings; rungs pinned to an earlier placement -----

def _lands_fixture(tmp_path, margin=1.0):
    from lag_ladder.summarize_reader import PREDICTIONS
    (tmp_path / PREDICTIONS).mkdir(parents=True, exist_ok=True)
    pred = {"texts": ["agentic"], "rule": {"holds_max": 1e-3},
            "predictions": {
                "iii_lands_with": {"rungs": {"1": "ag-1", "32": "ag-32"}, "text": "agentic", "margin_log10": margin,
                                   "reference_corpus": {"1": 2e-7, "32": 6.6e-6},
                                   "reference_rollout": {"1": 4.7e-3, "32": 8.9e-2}},
                "distance_must_equal": {"rungs": {"1": "ag-1", "32": "ag-32"},
                                        "reference": {"1": 2.2895683763734974e-05, "32": 1.0494302565318275e-04},
                                        "rel_tol": 1e-9}}}
    (tmp_path / PREDICTIONS / "x.json").write_text(json.dumps({"prediction": pred}), encoding="utf-8")

    def ag(m):
        return {"agentic": {"b1": {"median_kl": m}, "band": "HOLDS_B", "at_floor": False}}

    return {"rule": {"texts": ["corpus", "on-path"], "holds_max": 1e-3}, "controls": {},
            "rungs": [{"label": "ag-1", "withheld": False, "rel_delta_all": 2.2895683763734974e-05, "texts": ag(3e-7)},
                      {"label": "ag-32", "withheld": False, "rel_delta_all": 1.0494302565318275e-04, "texts": ag(9e-6)}],
            "slope_fit": {}}


def test_lands_clause_names_which_reference_each_lag_lands_with_and_judges_the_margin(tmp_path):
    import copy
    from lag_ladder.summarize_reader import seal_readout
    s = _lands_fixture(tmp_path)
    c = seal_readout(s, tmp_path, "x")["clauses"]["lands"]
    assert c["verdict"] == "HELD" and c["lands_with"] == {"1": "corpus", "32": "corpus"}
    assert c["distance"]["1"]["to_corpus_log10"] == pytest.approx(math.log10(3e-7 / 2e-7))
    assert c["distance"]["1"]["to_rollout_log10"] == pytest.approx(math.log10(4.7e-3 / 3e-7))
    # closer to corpus but by under a decade: FALSIFIED, and the side it lands with is still named
    t = copy.deepcopy(s); t["rungs"][1]["texts"]["agentic"]["b1"]["median_kl"] = 1.5e-3      # corpus 2.36 dec, rollout 1.77
    c = seal_readout(t, tmp_path, "x")["clauses"]["lands"]
    assert c["verdict"] == "FALSIFIED" and c["lands_with"]["32"] == "rollout"
    t = copy.deepcopy(s); t["rungs"][1]["texts"]["agentic"]["b1"]["median_kl"] = 4e-4        # corpus 1.78 dec, rollout 2.35
    c = seal_readout(t, tmp_path, "x")["clauses"]["lands"]
    assert c["verdict"] == "FALSIFIED" and c["lands_with"]["32"] == "corpus"
    t = copy.deepcopy(s); t["rungs"][0]["withheld"] = True
    assert seal_readout(t, tmp_path, "x")["clauses"]["lands"]["verdict"] == "UNREAD"


def test_distance_pin_refuses_a_missing_or_moved_rung_and_passes_the_sealed_placement(tmp_path):
    import copy
    from lag_ladder.summarize_reader import assert_distance_pins
    s = _lands_fixture(tmp_path)
    assert_distance_pins(s, tmp_path, "x")                       # equal: no refusal
    assert_distance_pins(s, tmp_path, "nosuchseal")              # no seal: nothing to pin
    t = copy.deepcopy(s); t["rungs"][1]["rel_delta_all"] = 1.0494302565318275e-04 * (1 + 1e-7)
    with pytest.raises(ValueError, match="differs from the sealed reference"):
        assert_distance_pins(t, tmp_path, "x")
    t = copy.deepcopy(s); t["rungs"][1]["rel_delta_all"] = None
    with pytest.raises(ValueError, match="has no position"):
        assert_distance_pins(t, tmp_path, "x")
    t = copy.deepcopy(s); t["rungs"] = t["rungs"][:1]
    with pytest.raises(ValueError, match="is not in this summary"):
        assert_distance_pins(t, tmp_path, "x")


# --- §14.5: the two readouts the families' seals need that no earlier seal had -------------------------------------

def _seal(root, predictions: dict, texts=("corpus", "on-path", "rollout")):
    from lag_ladder.summarize_reader import PREDICTIONS
    (root / PREDICTIONS).mkdir(parents=True, exist_ok=True)
    (root / PREDICTIONS / "x.json").write_text(
        json.dumps({"prediction": {"texts": list(texts), "rule": {"holds_max": 1e-3}, "predictions": predictions}}),
        encoding="utf-8")


def _rung(label, x, **medians):
    return {"label": label, "withheld": False, "rel_delta_all": x,
            "texts": {t: {"b1": {"median_kl": m}, "band": "UNRESOLVED", "at_floor": False} for t, m in medians.items()}}


_BASE = {"rule": {"texts": ["corpus", "on-path"], "holds_max": 1e-3}, "controls": {}, "slope_fit": {}}


def test_lost_at_lag_1_clause_reads_the_rollout_median_at_the_first_lag(tmp_path):
    """0017's sentence as a sealed clause: HOLDS_B is lost at the first optimizer step on the trainer's own text."""
    from lag_ladder.summarize_reader import seal_readout
    _seal(tmp_path, {"ii_rollout_lost_at_lag_1": {"rung": "roll-1", "text": "rollout", "at_least": 1e-3}})
    c = seal_readout({**_BASE, "rungs": [_rung("roll-1", 2e-5, rollout=4.7e-3)]}, tmp_path, "x")["clauses"]["ii-lost"]
    assert c == {"rung": "roll-1", "text": "rollout", "at_least": 1e-3, "median_kl": 4.7e-3, "verdict": "HELD"}
    c = seal_readout({**_BASE, "rungs": [_rung("roll-1", 2e-5, rollout=4e-4)]}, tmp_path, "x")["clauses"]["ii-lost"]
    assert c["verdict"] == "FALSIFIED"
    withheld = {**_rung("roll-1", 2e-5, rollout=4.7e-3), "withheld": True}
    assert seal_readout({**_BASE, "rungs": [withheld]}, tmp_path, "x")["clauses"]["ii-lost"]["verdict"] == "UNREAD"
    assert seal_readout({**_BASE, "rungs": []}, tmp_path, "x")["clauses"]["ii-lost"]["verdict"] == "UNREAD"


def _excess_seal(root, rungs, against_sha=None, against="results/reader/summary.json"):
    from lag_ladder.hashing import sha256_file_bytes
    sha = against_sha or sha256_file_bytes(REPO_ROOT / "results" / "reader" / "summary.json")
    _seal(root, {"iii_rl_excess_over_0015_backbone": {
        "against": against, "against_sha256": sha, "backbone_pair": "olmo2-1b-rlvr1",
        "distance_range": [1e-4, 1e-1], "at_least_factor": 10.0, "rungs": rungs, "texts": ["corpus", "on-path"]}})


def _copy_0015_reference(root):
    ref = REPO_ROOT / "results" / "reader" / "summary.json"
    (root / "results" / "reader").mkdir(parents=True, exist_ok=True)
    (root / "results" / "reader" / "summary.json").write_bytes(ref.read_bytes())


def test_backbone_excess_clause_reproduces_0018s_reading_from_0015s_summary(tmp_path):
    """0017's clause (iii), read by hand in 0018 as 11.1 (corpus) / 7.88 (on-path) at lag 32 against noise-0.0001:
    the readout recomputes it from 0015's summary.json (sha 5d761594...) and 0018's own medians, and FALSIFIES on
    on-path as 0018 did; lag 16 at 9.55e-5 lies outside [1e-4, 1e-1] and is listed, not read."""
    from lag_ladder.hashing import sha256_file_bytes
    from lag_ladder.summarize_reader import seal_readout
    assert sha256_file_bytes(REPO_ROOT / "results" / "reader" / "summary.json").startswith("5d76159485892f1d")
    s18 = json.loads((REPO_ROOT / "results" / "ownrun" / "summary.json").read_text(encoding="utf-8"))
    by = {e["label"]: e for e in s18["rungs"]}
    rungs = [_rung(k, by[k]["rel_delta_all"], **{t: by[k]["texts"][t]["b1"]["median_kl"] for t in ("corpus", "on-path")})
             for k in ("ownrun-lag-16", "ownrun-lag-32")]
    _copy_0015_reference(tmp_path)
    _excess_seal(tmp_path, ["ownrun-lag-16", "ownrun-lag-32"])
    c = seal_readout({**_BASE, "rungs": rungs}, tmp_path, "x")["clauses"]["iii-excess"]
    assert c["against_sha256"].startswith("5d76159485892f1d") and c["backbone_pair"] == "olmo2-1b-rlvr1"
    assert c["outside_range"] == {"ownrun-lag-16": pytest.approx(9.547494899946862e-05)}
    r = c["rungs"]["ownrun-lag-32"]
    assert r["nearest_backbone"] == "noise-0.0001" and r["rel_delta_all"] == pytest.approx(1.0494302565318275e-4)
    assert round(r["ratio"]["corpus"], 1) == 11.1 and round(r["ratio"]["on-path"], 2) == 7.88
    assert c["verdict"] == "FALSIFIED"


def test_backbone_excess_clause_refuses_a_moved_reference_and_reads_unread_with_no_rung_in_range(tmp_path):
    from lag_ladder.summarize_reader import seal_readout
    summary = {**_BASE, "rungs": [_rung("lag-1", 2e-5, corpus=2e-7, **{"on-path": 1e-7})]}
    _copy_0015_reference(tmp_path)
    _excess_seal(tmp_path, ["lag-1"])
    c = seal_readout(summary, tmp_path, "x")["clauses"]["iii-excess"]
    assert c["verdict"] == "UNREAD" and c["rungs"] == {} and list(c["outside_range"]) == ["lag-1"]
    _excess_seal(tmp_path, ["lag-1"], against_sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        seal_readout(summary, tmp_path, "x")


def test_backbone_excess_readout_leaves_0017s_prose_only_seal_alone(tmp_path):
    """ledger/predictions/ownrun.json carries the key as prose (0018 read it by hand); no machine-readable fields,
    no readout, and 0018's summary re-summarizes as written."""
    from lag_ladder.summarize_reader import seal_readout
    _seal(tmp_path, {"iii_rl_excess_over_0015_backbone": {
        "against": "results/reader/summary.json sha256 5d76... (entry 0015)", "statement": "read by hand"}})
    assert seal_readout({**_BASE, "rungs": []}, tmp_path, "x")["clauses"] == {}
