"""The reader summarizer (contract interface 4) recomputes every statistic from the report.json arrays and
refuses on any disagreement; it withholds rungs whose reader's controls failed and marks rungs at the float16
floor. The fake scorer's cells are known functions of the stand-in distance, so (B1)-(B4) are checkable."""
import json

import pytest

from lag_ladder import reader
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
