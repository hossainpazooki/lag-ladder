"""The inject path for a hybrid-state cache (the Kimi Linear seed): which state goes where under each stale arm,
and that a dump round-trips with its shapes, dtypes and hashes. Exercised against `tests/fake_hybrid.py`, a
two-layer numpy stand-in with the cache class shape of `KimiDynamicCache`; no torch, no fla, no model."""
import json

import numpy as np
import pytest

from tests.fake_hybrid import (CONV_W, KDA_D, KDA_HEADS, LAYER_TYPES, MLA_DK, MLA_DV, MLA_HEADS, VOCAB,
                               FakeHybrid, FakeHybridCache)
from tests.test_score_continuation import kl_fresh_stale

P, C = 12, 5


def _tokens(seed: int, n: int = P + C) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, VOCAB, size=n)


def _dump(tmp_path, name: str, model: FakeHybrid, prefix: np.ndarray):
    from lag_ladder import hybrid_state as hs
    cache = model.prefill(prefix)
    d = tmp_path / name
    meta = hs.write_dump(d, cache, layer_types=LAYER_TYPES, prefix_len=len(prefix), extra={"model": name})
    return d, meta


def _score(model: FakeHybrid, arm: str, fresh_dir, stale_dir, cont: np.ndarray):
    """Logits of the continuation under `arm`, as the pod driver will compute them: compose, fill, forward."""
    from lag_ladder import hybrid_state as hs
    composed = hs.compose(arm, hs.load_dump(fresh_dir), hs.load_dump(stale_dir))
    cache = FakeHybridCache()
    composed.fill(cache)
    return model.forward(cont, cache), cache


def _one_pass(model: FakeHybrid, tokens: np.ndarray) -> np.ndarray:
    return model.forward(tokens, FakeHybridCache())[P:]


# --- the dump ------------------------------------------------------------------------------------------------

def test_dump_round_trip_keeps_shapes_dtypes_hashes_and_layer_kinds(tmp_path):
    from lag_ladder import hybrid_state as hs
    reader = FakeHybrid(seed=0)
    d, meta = _dump(tmp_path, "reader", reader, _tokens(1)[:P])
    assert meta["schema"] == hs.SCHEMA and meta["prefix_len"] == P
    assert meta["layer_types"] == list(LAYER_TYPES)
    assert meta["mla"] == {"layers": [0], "heads": MLA_HEADS, "k_dim": MLA_DK, "v_dim": MLA_DV, "dtype": "float64"}
    assert meta["kda"] == {"layers": [1], "heads": KDA_HEADS, "k_dim": KDA_D, "v_dim": KDA_D,
                           "state_dtype": "float32", "conv_width": CONV_W, "conv_dtype": "float64"}
    assert json.loads((d / "meta.json").read_text(encoding="utf-8")) == meta
    loaded = hs.load_dump(d)
    K, V = loaded.mla[0]
    assert K.shape == (P, MLA_HEADS, MLA_DK) and V.shape == (P, MLA_HEADS, MLA_DV)
    S, conv = loaded.kda[1]
    assert S.shape == (KDA_HEADS, KDA_D, KDA_D) and S.dtype == np.float32
    assert len(conv) == 3 and all(c.shape == (KDA_HEADS * KDA_D, CONV_W) for c in conv)
    assert set(meta["files"]) == {"mla_layer00.npz", "kda_layer01.npz"}


def test_load_refuses_a_file_whose_hash_changed(tmp_path):
    from lag_ladder import hybrid_state as hs
    d, _ = _dump(tmp_path, "reader", FakeHybrid(0), _tokens(1)[:P])
    f = d / "kda_layer01.npz"
    z = dict(np.load(f))
    z["recurrent"] = z["recurrent"] + np.float32(1e-3)
    np.savez(f, **z)
    with pytest.raises(ValueError, match="sha256"):
        hs.load_dump(d)


def test_load_refuses_a_recurrent_state_that_is_not_float32(tmp_path):
    from lag_ladder import hybrid_state as hs
    d, meta = _dump(tmp_path, "reader", FakeHybrid(0), _tokens(1)[:P])
    f = d / "kda_layer01.npz"
    z = dict(np.load(f))
    z["recurrent"] = z["recurrent"].astype(np.float16)
    np.savez(f, **z)
    meta["files"]["kda_layer01.npz"] = hs.sha256_file(f)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="float32"):
        hs.load_dump(d)


def test_load_refuses_a_shape_that_disagrees_with_meta(tmp_path):
    from lag_ladder import hybrid_state as hs
    d, meta = _dump(tmp_path, "reader", FakeHybrid(0), _tokens(1)[:P])
    meta["mla"]["k_dim"] = MLA_DK + 1
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="k_dim"):
        hs.load_dump(d)


# --- the arms ------------------------------------------------------------------------------------------------

def test_arms_name_the_source_of_every_state_kind(tmp_path):
    from lag_ladder import hybrid_state as hs
    m = FakeHybrid(0)
    fresh, _ = _dump(tmp_path, "reader", m, _tokens(1)[:P])
    stale, _ = _dump(tmp_path, "writer", FakeHybrid(1), _tokens(1)[:P])
    f, s = hs.load_dump(fresh), hs.load_dump(stale)
    assert hs.ARMS == ("FRESH", "FULL-STALE", "MLA-STALE", "KDA-STALE")
    assert hs.compose("FRESH", f, s).sources == {"mla": "fresh", "kda": "fresh"}
    assert hs.compose("FULL-STALE", f, s).sources == {"mla": "stale", "kda": "stale"}
    assert hs.compose("MLA-STALE", f, s).sources == {"mla": "stale", "kda": "fresh"}
    assert hs.compose("KDA-STALE", f, s).sources == {"mla": "fresh", "kda": "stale"}
    with pytest.raises(ValueError, match="arm"):
        hs.compose("STALE", f, s)


def test_every_arm_carries_the_mla_cache_so_the_prefix_length_is_read_from_it(tmp_path):
    from lag_ladder import hybrid_state as hs
    m = FakeHybrid(0)
    fresh, _ = _dump(tmp_path, "reader", m, _tokens(1)[:P])
    stale, _ = _dump(tmp_path, "writer", FakeHybrid(1), _tokens(1)[:P])
    for arm in hs.ARMS:
        _, cache = _score(m, arm, fresh, stale, _tokens(1)[P:])
        assert cache.get_seq_length() == P + C          # P injected + C appended by the forward
        assert cache.recurrent_states[1].dtype == np.float32


def test_fresh_arm_differs_from_one_pass_only_by_the_float32_state_round_trip(tmp_path):
    """The fake's floor: its one pass keeps the recurrent state in float64, the cached path stores it in float32
    at the prefix boundary (as the kernels do), so FRESH-from-cache is not bit-identical to one pass. This is the
    shape of the floor control the probe must measure on the real model (seed §4), here with a known cause."""
    m = FakeHybrid(0)
    toks = _tokens(1)
    fresh, _ = _dump(tmp_path, "reader", m, toks[:P])
    logits, _ = _score(m, "FRESH", fresh, fresh, toks[P:])
    one = _one_pass(m, toks)
    gap = np.abs(logits - one).max()
    assert 0.0 < gap < 1e-7, gap            # float32 eps 1.2e-7 x a state of order 0.1
    assert max(kl_fresh_stale(one[t], logits[t]) for t in range(C)) < 1e-12


def test_full_stale_from_the_readers_own_dump_is_the_identity_control(tmp_path):
    m = FakeHybrid(0)
    toks = _tokens(1)
    fresh, _ = _dump(tmp_path, "reader", m, toks[:P])
    a, _ = _score(m, "FRESH", fresh, fresh, toks[P:])
    b, _ = _score(m, "FULL-STALE", fresh, fresh, toks[P:])
    assert np.abs(a - b).max() == 0.0


def test_fill_copies_so_the_dump_arrays_are_not_mutated_by_the_forward(tmp_path):
    from lag_ladder import hybrid_state as hs
    m = FakeHybrid(0)
    toks = _tokens(1)
    fresh, _ = _dump(tmp_path, "reader", m, toks[:P])
    loaded = hs.load_dump(fresh)
    before = loaded.kda[1][0].copy()
    cache = FakeHybridCache()
    hs.compose("FRESH", loaded, loaded).fill(cache)
    m.forward(toks[P:], cache)
    assert np.array_equal(loaded.kda[1][0], before)
    assert not np.array_equal(cache.recurrent_states[1], before)


def test_compose_refuses_dumps_that_disagree_on_prefix_or_layer_kinds(tmp_path):
    from lag_ladder import hybrid_state as hs
    m = FakeHybrid(0)
    fresh, _ = _dump(tmp_path, "reader", m, _tokens(1)[:P])
    short, _ = _dump(tmp_path, "writer", m, _tokens(1)[:P - 1])
    with pytest.raises(ValueError, match="prefix_len"):
        hs.compose("FULL-STALE", hs.load_dump(fresh), hs.load_dump(short))
    other, meta = _dump(tmp_path, "other", m, _tokens(1)[:P])
    meta["layer_types"] = ["linear_attention", "full_attention"]
    (other / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="layer"):
        hs.load_dump(other)


# --- the decomposition, with a known answer ---------------------------------------------------------------

def _four_arms(tmp_path, reader: FakeHybrid, writer: FakeHybrid):
    toks = _tokens(1)
    fresh, _ = _dump(tmp_path, "reader", reader, toks[:P])
    stale, _ = _dump(tmp_path, "writer", writer, toks[:P])
    out = {arm: _score(reader, arm, fresh, stale, toks[P:])[0] for arm in
           ("FRESH", "FULL-STALE", "MLA-STALE", "KDA-STALE")}
    kl = {arm: float(np.mean([kl_fresh_stale(out["FRESH"][t], out[arm][t]) for t in range(C)])) for arm in out}
    assert kl["FRESH"] == 0.0
    assert kl["FULL-STALE"] > 1e-4, "the fake writer must actually differ"
    return out, kl


def test_a_writer_that_differs_only_in_the_last_kind_injures_only_that_arm(tmp_path):
    """KDA is the fake's last layer, so a writer whose KDA weights differ writes the same MLA K/V as the reader:
    KDA-STALE reproduces FULL-STALE exactly and MLA-STALE is the identity. The clean per-kind attribution."""
    out, kl = _four_arms(tmp_path, FakeHybrid(0), FakeHybrid(0, kda_scale=1.6))
    assert np.abs(out["KDA-STALE"] - out["FULL-STALE"]).max() == 0.0
    assert kl["MLA-STALE"] == 0.0


def test_a_writer_that_differs_upstream_leaks_into_the_downstream_states(tmp_path):
    """A writer whose MLA weights differ feeds different hidden states into its KDA layer, so its recurrent and
    conv states differ too: KDA-STALE is NOT the identity and MLA-STALE is NOT the whole injury. The arms
    attribute the injury to the state kind the reader consumes, not to the layer kind whose weights moved; the
    decomposition clause (seed §4, R-K3) has to be worded for this."""
    out, kl = _four_arms(tmp_path, FakeHybrid(0), FakeHybrid(0, mla_scale=1.6))
    assert kl["KDA-STALE"] > 0.0
    assert np.abs(out["MLA-STALE"] - out["FULL-STALE"]).max() > 0.0
    assert kl["MLA-STALE"] > 0.0 and kl["KDA-STALE"] > 0.0


def test_scrambled_prefix_reads_far_above_the_identity(tmp_path):
    m = FakeHybrid(0)
    toks, other = _tokens(1), _tokens(2)
    fresh, _ = _dump(tmp_path, "reader", m, toks[:P])
    scr, _ = _dump(tmp_path, "scrambled", m, other[:P])
    a, _ = _score(m, "FRESH", fresh, fresh, toks[P:])
    b, _ = _score(m, "FULL-STALE", fresh, scr, toks[P:])
    assert float(np.mean([kl_fresh_stale(a[t], b[t]) for t in range(C)])) > 1e-3


# --- the per-token statistic, as the driver will record it -------------------------------------------------

def test_kl_per_token_follows_the_scorer_contract():
    from lag_ladder import hybrid_state as hs
    r = np.random.default_rng(0)
    lf, ls = r.normal(size=(C, VOCAB)), r.normal(size=(C, VOCAB))
    got = hs.kl_per_token(lf, ls)
    want = np.array([kl_fresh_stale(lf[t], ls[t]) for t in range(C)])
    assert got.shape == (C,) and np.allclose(got, want, rtol=0, atol=1e-12)
    assert hs.kl_per_token(lf, lf).max() == 0.0
