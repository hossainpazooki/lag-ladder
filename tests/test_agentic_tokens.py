"""tools/agentic_tokens.py (the 2026-10-22 plan, cell 1b): the window rule restated from linear-ceiling's entry
0016 §4 over the rendered export, the manifest sidecar, and config/ownrun_agentic.toml (registered by 0030)."""
import importlib.util
import json

import numpy as np
import pytest

from lag_ladder import REPO_ROOT, reader
from lag_ladder.config import load_reader_config

spec = importlib.util.spec_from_file_location("agentic_tokens", REPO_ROOT / "tools" / "agentic_tokens.py")
agentic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentic)

SUITES = ("mlperf-coding", "mlperf-workflow")


def words(text):
    return [hash(w) % 1000 for w in text.split()]


def _items(n_coding=3, n_workflow=3, short=()):
    out = [{"suite": "mlperf-coding", "traj_id": f"deepswe_{i}", "text": "w " * (3 if f"deepswe_{i}" in short else 20)}
           for i in range(n_coding)]
    out += [{"suite": "mlperf-workflow", "traj_id": f"sim_{i}", "text": "w " * (3 if f"sim_{i}" in short else 20)}
            for i in range(n_workflow)]
    return out


def test_draw_is_stratified_skips_short_and_is_a_pure_function_of_seed():
    items = _items(short=("deepswe_0", "sim_2"))
    tok, rows = agentic.sample_windows(items, words, SUITES, n=4, seq_len=8, seed=0)
    assert tok.shape == (4, 8) and tok.dtype == np.int64
    assert [r["suite"] for r in rows] == ["mlperf-coding"] * 2 + ["mlperf-workflow"] * 2     # suite by suite, as linear-ceiling
    assert {r["traj_id"] for r in rows} <= {"deepswe_1", "deepswe_2", "sim_0", "sim_1"}
    assert agentic.sample_windows(list(reversed(items)), words, SUITES, n=4, seq_len=8, seed=0)[1] == rows
    many = _items(n_coding=12, n_workflow=12)      # a pool larger than the share, so the seed can matter
    a = agentic.sample_windows(many, words, SUITES, n=4, seq_len=8, seed=0)[1]
    assert a == agentic.sample_windows(list(reversed(many)), words, SUITES, n=4, seq_len=8, seed=0)[1]
    assert a != agentic.sample_windows(many, words, SUITES, n=4, seq_len=8, seed=1)[1]


def test_refuses_a_suite_that_cannot_fill_its_share_and_unequal_stratification():
    with pytest.raises(ValueError, match="mlperf-coding: only 1 trajectories"):
        agentic.sample_windows(_items(n_coding=1), words, SUITES, n=4, seq_len=8, seed=0)
    with pytest.raises(ValueError, match="stratify equally"):
        agentic.sample_windows(_items(), words, SUITES, n=3, seq_len=8, seed=0)


def test_read_rendered_refuses_a_malformed_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"suite": "mlperf-coding", "traj_id": "a", "text": "x y"}\n\n{"suite": "mlperf-coding"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":3: not a"):
        agentic.read_rendered(p)
    p.write_text('{"suite": "mlperf-coding", "traj_id": "a", "text": "x y"}\n', encoding="utf-8")
    assert agentic.read_rendered(p) == [{"suite": "mlperf-coding", "traj_id": "a", "text": "x y"}]


def test_assert_equal_ids_passes_the_same_file_and_names_the_first_differing_row(tmp_path):
    tok = np.arange(8, dtype=np.int64).reshape(2, 4)
    rows = [{"suite": "s", "traj_id": "a"}, {"suite": "s", "traj_id": "b"}]
    other = tmp_path / "lc" / "agent.npy"
    other.parent.mkdir()
    np.save(other, tok)
    got = agentic.assert_equal_ids(tok, rows, other)
    assert got["rows_compared"] == 2 and got["manifest_sha256"] is None and len(got["sha256"]) == 64
    other.with_suffix(".manifest.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert len(agentic.assert_equal_ids(tok, rows, other)["manifest_sha256"]) == 64
    with pytest.raises(AssertionError, match="manifest row 1"):
        agentic.assert_equal_ids(tok, [rows[0], {"suite": "s", "traj_id": "c"}], other)
    bad = tok.copy()
    bad[1, 2] = 99
    with pytest.raises(AssertionError, match="row 1 differs .* at position 2: 99 here, 6 there"):
        agentic.assert_equal_ids(bad, rows, other)
    with pytest.raises(AssertionError, match="shape"):
        agentic.assert_equal_ids(tok[:1], rows[:1], other)


def test_write_tokens_manifest_pins_the_input_and_the_file(tmp_path):
    tok = np.arange(8, dtype=np.int64).reshape(2, 4)
    out = agentic.write_tokens(tmp_path / "t" / "x.npy", tok, [{"suite": "s", "traj_id": "a"}, {"suite": "s", "traj_id": "b"}],
                               rendered_sha256="ab" * 32, tokenizer="snap/tokenizer.json", seed=0)
    man = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert np.array_equal(np.load(out), tok)
    assert man["rendered_sha256"] == "ab" * 32 and man["shape"] == [2, 4] and len(man["sha256"]) == 64
    assert man["rule"].startswith("linear-ceiling ledger entry 0016") and man["rows"][1]["traj_id"] == "b"
    assert man["equal_to"] is None and man["seed"] == 0


def test_ownrun_agentic_config_loads_names_the_agentic_text_per_rung_and_is_registered_by_0030():
    cfg = load_reader_config(REPO_ROOT / "config" / "ownrun_agentic.toml", REPO_ROOT)
    base = load_reader_config(REPO_ROOT / "config" / "ownrun.toml", REPO_ROOT)
    assert cfg.registered_by == "0030" and cfg.seal == "ownrun-agentic"
    # the one sha: linear-ceiling's data/e8m/agent_n50_len1024_seed8.npy, byte-identical to the file the builder writes
    assert cfg.pairs["qwen2.5-1.5b-mlperf"].tokens_sha256 == "9dce6082d057fb739a17b0c5454ecc57cf2ec0476a902bf1609d563de9221351"
    assert cfg.pairs["qwen2.5-1.5b-mlperf"].spans is None
    assert [r.label for r in cfg.rungs] == [f"ownrun-agentic-lag-{g}" for g in (1, 2, 4, 8, 16, 32)]
    assert tuple(cfg.pairs["qwen2.5-1.5b-mlperf"].heldout or cfg.heldout) == (0, 50)     # every window is held out (ruled 2026-10-22)
    assert all(r.texts == ("agentic",) and r.pair == "qwen2.5-1.5b-mlperf" and r.writer_dump == "source@local-step_0"
               and r.reader == f"local:{r.reader_dump.split('@local-')[1]}" for r in cfg.rungs)
    # the registered convention of 0017, unchanged: prefix, dtype, rule, limits, the instrument pin; only the rows differ
    assert (cfg.prefix_len, cfg.dtype, cfg.rule, cfg.upstream_sha) == (base.prefix_len, base.dtype, base.rule, base.upstream_sha)
    assert tuple(base.heldout) == (40, 50) and tuple(cfg.heldout) == (0, 50)
    assert cfg.controls["per_pair"] is True and cfg.controls["floor_max_median_kl"] == base.controls["floor_max_median_kl"]
    # the plan puts every control of a reader on the agentic text (per_pair), then its rung
    jobs = reader.plan(cfg).jobs
    assert {j.text for j in jobs} == {"agentic"} and {j.kind for j in jobs} >= {"identity", "scrambled", "floor", "rung"}
    assert len(jobs) == 6 * 3 * 50 + 6 * 50     # six readers x (floor, identity, scrambled) x fifty rows, plus six rungs x fifty
    # the scrambled control pairs row i with row i + 1 (wrap 49 -> 0): within suite everywhere but the seam (24 -> 25) and the wrap
    pairs = [(j.seq, j.stale_seq) for j in jobs if j.kind == "scrambled" and j.reader == "local:step_1"]
    assert pairs == [(i, (i + 1) % 50) for i in range(50)]
    suite = lambda i: "coding" if i < 25 else "workflow"
    assert [p for p in pairs if suite(p[0]) != suite(p[1])] == [(24, 25), (49, 0)]


def test_agentic_is_a_per_rung_text_never_a_rule_text(tmp_path):
    src = (REPO_ROOT / "config" / "ownrun_agentic.toml").read_text(encoding="utf-8")
    p = tmp_path / "ownrun_agentic.toml"
    p.write_text(src.replace('texts = ["corpus", "on-path"]', 'texts = ["agentic"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="reader.rule"):
        load_reader_config(p, REPO_ROOT)
