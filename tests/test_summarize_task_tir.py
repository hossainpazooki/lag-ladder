"""The reduced-(C) summarizer (design 2026-11-09, reduced (C)): every number recomputed from the JSONL records the fake
tool wrote through the driver; the controls re-derived and compared with run.json; the positive controls read on the
first n prompts; PC1-PC3 read against a sealed prediction file; refusals on a hash mismatch or a halted run."""
import json
import shutil
from pathlib import Path

import pytest

from lag_ladder import summarize_task_tir as stt
from lag_ladder import task_tir
from lag_ladder.hashing import sha256_file_bytes
from tests.fake_task_swap_tir import FakeTaskSwapTir, default_outcome
from tests.test_task_tir import ENV, _setup_tir

PREDICTIONS = {"pc1_flip_rate_at_first_lag": {"reader": "local:pltir-step_1", "arm": "stale_prompt", "at_least": 0.10},
               "pc2_shared_below_prompt": {"readers": ["local:pltir-step_1", "local:pltir-step_32"]},
               "pc3_flip_growth": {"first": "local:pltir-step_1", "last": "local:pltir-step_32", "arm": "stale_prompt"}}


def _run(repo, seal_cfg, outcome=None, predictions=PREDICTIONS):
    cfg, tools = _setup_tir(repo, seal_cfg, predictions=predictions)
    task_tir.run(cfg, repo, runner=FakeTaskSwapTir(outcome=outcome) if outcome else FakeTaskSwapTir(), seal_cfg=seal_cfg,
                 tools_dir=tools, env=ENV)
    return cfg


def test_summary_recomputes_each_lag_from_the_records(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    s = stt.summarize(cfg)
    assert s["registered_by"] == "0039" and s["fidelity"]["passed"] and set(s["lags"]) == {"local:pltir-step_1", "local:pltir-step_32"}
    lag = s["lags"]["local:pltir-step_1"]
    assert lag["n"] == 9 and lag["controls"]["floor_n"] == 9 and lag["controls"]["control_n"] == 4
    # the fake answers gold on even indices; the expectation is recomputed from the records, not assumed
    fresh = [json.loads(l) for l in (cfg.results_dir / "readers" / "local_pltir-step_1" / "fresh.jsonl").read_text().splitlines()]
    assert lag["acc_fresh"] == pytest.approx(100.0 * sum(r["correct"] for r in fresh) / 9)
    assert lag["math_answer_rate_fresh"] == 1.0 and lag["mean_turns_fresh"] == 2.0 and lag["tool_call_rate_fresh"] == 1.0
    sp = lag["arms"]["stale_prompt"]
    stale = {r["index"]: r for r in (json.loads(l) for l in (cfg.results_dir / "readers" / "local_pltir-step_1" / "stale_prompt.jsonl").read_text().splitlines())}
    fr = {r["index"]: r for r in fresh}
    assert sp["n"] == 9 and sp["flip_rate"] == pytest.approx(sum(stale[i]["answer"] != fr[i]["answer"] for i in fr) / 9)
    assert sp["flip_ci_low"] <= sp["flip_rate"] <= sp["flip_ci_high"] and sp["rank_correlation_b1"] is not None
    assert sp["tool_sequence_flip_rate"] == 0.0 and sp["stale_inside_lcp_rate"] == 1.0 and sp["mean_sandbox_hits"] == 1.0
    assert set(lag["arms"]) == {"stale_prompt", "stale_shared", "near", "far"}
    assert lag["arms"]["near"]["n"] == 4 and lag["arms"]["far"]["flip_rate"] == 1.0 and lag["arms"]["near"]["rank_correlation_b1"] is None
    assert lag["positive_controls_fire"] is True and lag["evaluable"] is True
    assert (cfg.results_dir / "summary.json").exists()
    ro = s["seal_readout"]["clauses"]
    assert set(ro) == {"pc1", "pc2", "pc3"} and all(c["verdict"] in ("HELD", "FALSIFIED") for c in ro.values())
    # growth: the fake flips more at step 32 than at step 1
    assert s["lags"]["local:pltir-step_32"]["arms"]["stale_prompt"]["flip_rate"] >= sp["flip_rate"]
    text = stt.render(s)
    assert text.isascii() and "stale_prompt" in text and "pc1" in text


def test_summary_marks_a_lag_unevaluable_when_a_positive_control_flips_too_few_answers(repo, seal_cfg):
    def weak_far(i, step, arm, gold):
        if arm == "far":
            return default_outcome(i, step, "fresh", gold)       # FAR reproduces FRESH: the instrument did not fire
        return default_outcome(i, step, arm, gold)

    cfg = _run(repo, seal_cfg, outcome=weak_far)
    s = stt.summarize(cfg)
    for lag in s["lags"].values():
        assert lag["positive_controls_fire"] is False and lag["evaluable"] is False
        assert lag["arms"]["stale_prompt"]["flip_rate"] is not None          # descriptives stay
    assert all(c["verdict"] == "UNREAD" for c in s["seal_readout"]["clauses"].values())


def test_summary_refuses_a_record_hash_mismatch_and_a_halted_run(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    p = cfg.results_dir / "readers" / "local_pltir-step_1" / "stale_prompt.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"gold"', '"gold "', 1), encoding="utf-8")   # any byte; the hash is read first
    with pytest.raises(ValueError, match="sha256"):
        stt.summarize(cfg)
    rec_path = cfg.results_dir / "run.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["halted"] = "task_tir REFUSED: x"
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match="halted"):
        stt.summarize(cfg)


def test_summary_recomputed_controls_must_agree_with_run_json(repo, seal_cfg):
    cfg = _run(repo, seal_cfg)
    rec_path = cfg.results_dir / "run.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["controls"]["local:pltir-step_1"]["identity_flips"] = 1
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from run.json"):
        stt.summarize(cfg)


def test_seal_readout_reads_pc1_pc2_pc3(tmp_path):
    (tmp_path / stt.PREDICTIONS).mkdir(parents=True)
    (tmp_path / stt.PREDICTIONS / "pipelinerl-tir-task.json").write_text(json.dumps({"prediction": {"predictions": PREDICTIONS}}), encoding="utf-8")

    def lag(prompt, shared, lo, hi, evaluable=True):
        return {"evaluable": evaluable, "positive_controls_fire": evaluable,
                "arms": {"stale_prompt": {"flip_rate": prompt, "flip_ci_low": lo, "flip_ci_high": hi},
                         "stale_shared": {"flip_rate": shared, "flip_ci_low": lo, "flip_ci_high": hi}}}

    s = {"lags": {"local:pltir-step_1": lag(0.30, 0.20, 0.27, 0.33), "local:pltir-step_32": lag(0.60, 0.40, 0.57, 0.63)}}
    c = stt.seal_readout(s, tmp_path, "pipelinerl-tir-task")["clauses"]
    assert [c[k]["verdict"] for k in ("pc1", "pc2", "pc3")] == ["HELD"] * 3
    assert c["pc1"] == {"reader": "local:pltir-step_1", "arm": "stale_prompt", "at_least": 0.10, "flip_rate": 0.30, "verdict": "HELD"}
    assert c["pc2"]["below"] == {"local:pltir-step_1": True, "local:pltir-step_32": True}
    assert c["pc3"]["flip_rates"] == {"local:pltir-step_1": 0.30, "local:pltir-step_32": 0.60}
    assert c["pc3"]["intervals"] == {"local:pltir-step_1": [0.27, 0.33], "local:pltir-step_32": [0.57, 0.63]}
    # pc1 falls under 0.1; pc2 falls when shared is not below prompt at one reader; pc3 is on the point estimates, so
    # it still holds here (0.05 < 0.60) although the intervals now overlap
    s["lags"]["local:pltir-step_1"]["arms"]["stale_prompt"] = {"flip_rate": 0.05, "flip_ci_low": 0.03, "flip_ci_high": 0.60}
    c = stt.seal_readout(s, tmp_path, "pipelinerl-tir-task")["clauses"]
    assert c["pc1"]["verdict"] == "FALSIFIED" and c["pc2"]["verdict"] == "FALSIFIED" and c["pc3"]["verdict"] == "HELD"
    # pc3 falls when the rate at g=32 is not above the rate at g=1 (equal rates included), whatever the intervals
    s["lags"]["local:pltir-step_1"] = lag(0.30, 0.20, 0.27, 0.33)
    s["lags"]["local:pltir-step_32"]["arms"]["stale_prompt"] = {"flip_rate": 0.30, "flip_ci_low": 0.57, "flip_ci_high": 0.63}
    c = stt.seal_readout(s, tmp_path, "pipelinerl-tir-task")["clauses"]
    assert c["pc3"]["verdict"] == "FALSIFIED" and c["pc1"]["verdict"] == "HELD"
    # an unevaluable lag withholds every clause that reads it
    s["lags"]["local:pltir-step_32"] = lag(0.60, 0.40, 0.57, 0.63, evaluable=False)
    s["lags"]["local:pltir-step_1"] = lag(0.30, 0.20, 0.27, 0.33)
    c = stt.seal_readout(s, tmp_path, "pipelinerl-tir-task")["clauses"]
    assert c["pc1"]["verdict"] == "HELD" and c["pc2"]["verdict"] == "UNREAD" and c["pc3"]["verdict"] == "UNREAD"
    with pytest.raises(ValueError, match="no sealed predictions"):       # a missing seal file is refused, not read as no readout
        stt.seal_readout(s, tmp_path, "no-such-seal")
    # ... and so is a seal file that is there without the three sealed clauses: no payload, no keys, renamed keys, one short
    for payload in ({}, {"prediction": {"predictions": {}}}, {"prediction": {"predictions": {"pc1": PREDICTIONS["pc1_flip_rate_at_first_lag"]}}},
                    {"prediction": {"predictions": {k: v for k, v in PREDICTIONS.items() if k != "pc3_flip_growth"}}},
                    {"prediction": {"predictions": {k: v for k, v in PREDICTIONS.items() if k != "pc1_flip_rate_at_first_lag"}}}):
        (tmp_path / stt.PREDICTIONS / "broken.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="three sealed clauses"):
            stt.seal_readout(s, tmp_path, "broken")


def test_reader_splits_on_newline_only(tmp_path):
    # str.splitlines() also breaks a line at U+2028 (and U+2029, U+0085), which a model's
    # completion text may carry and json.dumps(ensure_ascii=False) would leave raw inside a string. The Task 4 writer
    # escapes it (ensure_ascii=True); the summarizer's reader splits on "\n" only, so a record carrying one stays one row
    # whichever way it was written (the hash the reader checks first is the file's own, so only the split is under test)
    for ch in ("\u2028", "\u2029", "\u0085"):
        text = "x = 1" + ch + "y = 2"
        rec = {"schema": task_tir.RECORD_SCHEMA, "index": 0, "completion_texts": [text]}
        escaped = tmp_path / "escaped.jsonl"
        with open(escaped, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        assert escaped.read_bytes().isascii()                                   # the amended writer's file: escape only
        rows = stt._load_jsonl(escaped, sha256_file_bytes(escaped))
        assert len(rows) == 1 and rows[0]["completion_texts"] == [text]
        raw = tmp_path / "raw.jsonl"
        with open(raw, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"schema": "tir-record-v1", "index": 0, "completion_texts": ["x = 1' + ch + 'y = 2"]}\n')
        assert ch.encode("utf-8") in raw.read_bytes()                           # the character is raw in the file
        assert len(raw.read_text(encoding="utf-8").splitlines()) == 2           # the split the reader must not use
        rows = stt._load_jsonl(raw, sha256_file_bytes(raw))
        assert len(rows) == 1 and rows[0]["completion_texts"] == [text]


def test_summary_refuses_a_run_that_did_not_finish(repo, seal_cfg):
    # task_tir.run saves run.json after every job and sets `finished` only at the end, so a
    # killed sitting (pod loss, OOM) leaves `halted` and `finished` both unset and the later readers' cells jobs missing;
    # that record is refused, not summarized as final
    cfg = _run(repo, seal_cfg)
    rec_path = cfg.results_dir / "run.json"
    good = json.loads(rec_path.read_text(encoding="utf-8"))
    jobs = [j for j in good["jobs"] if not (j["kind"] == "cells" and j["reader"] == "local:pltir-step_32")]
    assert len(jobs) == len(good["jobs"]) - 1 and good["finished"] and good["halted"] is None
    killed = dict(good, finished=None, jobs=jobs)
    rec_path.write_text(json.dumps(killed), encoding="utf-8")
    with pytest.raises(ValueError, match="did not finish"):
        stt.summarize(cfg)
    assert not (cfg.results_dir / "summary.json").exists()


def test_summary_json_is_written_with_lf_newlines(repo, seal_cfg, tmp_path):
    # the file is written with newline="\n": summary.json's bytes, and
    # so any digest of it, are the same on Windows and Linux, at the default target and at --out
    cfg = _run(repo, seal_cfg)
    stt.summarize(cfg)
    data = (cfg.results_dir / "summary.json").read_bytes()
    assert b"\r" not in data and data.count(b"\n") > 1 and json.loads(data.decode("utf-8"))["registered_by"] == "0039"
    out = tmp_path / "elsewhere.json"
    stt.summarize(cfg, out=out)
    assert out.read_bytes() == data


def test_summary_refuses_a_tree_whose_repo_holds_no_seal(repo, seal_cfg, tmp_path, monkeypatch):
    # summarize() takes the repo to be the root's grandparent. A results tree pulled to another
    # place has no sealed predictions there, and the summary of record must not be written without its PC1-PC3 readout
    cfg = _run(repo, seal_cfg)
    pulled = tmp_path / "pulled" / "pipelinerl_tir_task"
    shutil.copytree(cfg.results_dir, pulled)
    with pytest.raises(ValueError, match="no sealed predictions"):
        stt.summarize(cfg, root=pulled)
    assert not (pulled / "summary.json").exists()
    out = tmp_path / "elsewhere.json"
    with pytest.raises(ValueError, match="no sealed predictions"):
        stt.summarize(cfg, root=pulled, out=out)
    assert not out.exists()
    # the same records under <repo>/results/<name>, the default root, read the seal
    assert set(stt.summarize(cfg)["seal_readout"]["clauses"]) == {"pc1", "pc2", "pc3"}
    # ... and so does that root typed as a relative path: the repo is found from the resolved path
    monkeypatch.chdir(cfg.results_dir)
    assert set(stt.summarize(cfg, root=Path("."))["seal_readout"]["clauses"]) == {"pc1", "pc2", "pc3"}
    # a seal file that is there but holds no clauses is refused the same way, before anything is written
    (repo / stt.PREDICTIONS / "pipelinerl-tir-task.json").write_text(json.dumps({"prediction": {"predictions": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="three sealed clauses"):
        stt.summarize(cfg, out=out)
    assert not out.exists()
