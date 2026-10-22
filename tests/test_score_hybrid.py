"""The torch-free contract of `tools/score_hybrid.py` (design §16, R-K1): the bf16 bit packing the dump layout
uses, the shape check against a Kimi-style config.json, the scoring-input indexing, the arm sources, the dump
layout's compatibility with `lag_ladder.hybrid_state`, the report keys, and the config compatibility assertion.
The tool is imported as a module here; it must not import torch (or fla, transformers) at import time, so these
tests run in the offline venv. Anything that needs a model runs on the pod."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import lag_ladder
from lag_ladder import hybrid_state
from tests.fake_hybrid import FakeHybrid, LAYER_TYPES
from tests.test_score_continuation import REPORT_KEYS

TOOL = lag_ladder.REPO_ROOT / "tools" / "score_hybrid.py"

HYBRID_REPORT_KEYS = REPORT_KEYS | {"arm", "sources", "state_v_first", "dtype", "cached_rows", "seconds"}

# the Kimi Linear 48B-A3B config.json fields the shape check reads (retrieved 2026-10-22; digest a6ac3c2c4b5aa723)
KIMI_CFG = {
    "num_hidden_layers": 27, "num_attention_heads": 32, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
    "v_head_dim": 128, "mla_use_nope": True, "rope_theta": 10000.0, "vocab_size": 163840, "dtype": "bfloat16",
    "linear_attn_config": {"full_attn_layers": [4, 8, 12, 16, 20, 24, 27], "head_dim": 128,
                           "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26],
                           "num_heads": 32, "short_conv_kernel_size": 4},
}


@pytest.fixture(scope="module")
def sh():
    if not TOOL.exists():
        pytest.fail("tools/score_hybrid.py not built")
    for name in ("torch", "fla", "transformers"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("score_hybrid", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_hybrid"] = mod          # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


# --- import discipline --------------------------------------------------------------------------------------

def test_tool_imports_no_torch_and_nothing_from_the_repo_at_import_time(sh):
    src = TOOL.read_text(encoding="utf-8")
    import re
    assert not re.search(r"^(from|import)\s+(torch|fla|transformers|kvt|lag_ladder)\b", src, re.M), \
        "top-level import of a pod-only package or of the repo; import torch inside the functions that need it"
    assert "torch" not in sys.modules


# --- bf16 bit packing ------------------------------------------------------------------------------------------

def test_bf16_bits_round_trip_exact_on_bf16_representable_values(sh):
    # values with at most 8 significant bits are exactly bf16; packing and unpacking must return them bit-exact
    x = np.array([0.0, 1.0, -2.5, 0.15625, 1024.0, 2.0 ** 100, -(2.0 ** -100), 1.0 + 2.0 ** -7], dtype=np.float32)
    bits = sh.f32_to_bf16_bits(x)
    assert bits.dtype == np.uint16 and bits.shape == x.shape
    back = sh.bf16_bits_to_f32(bits)
    assert back.dtype == np.float32
    assert np.array_equal(back, x)


def test_f32_to_bf16_rounds_to_nearest_even(sh):
    # 1 + 2^-8 sits exactly between bf16 neighbours 1.0 and 1 + 2^-7: ties go to even (1.0);
    # 1 + 3*2^-9 is above the midpoint and rounds up to 1 + 2^-7
    x = np.array([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -9], dtype=np.float32)
    back = sh.bf16_bits_to_f32(sh.f32_to_bf16_bits(x))
    assert back.tolist() == [1.0, 1.0 + 2.0 ** -7]


def test_bf16_bits_keep_the_upper_half_of_the_float32_pattern(sh):
    x = np.array([1.0, -1.0], dtype=np.float32)
    assert sh.f32_to_bf16_bits(x).tolist() == [0x3F80, 0xBF80]


# --- expected shapes from config.json ------------------------------------------------------------------------

def test_expected_shapes_from_the_kimi_config(sh):
    e = sh.expected_shapes(KIMI_CFG)
    assert e["layer_types"].count("full_attention") == 7 and len(e["layer_types"]) == 27
    assert [i for i, t in enumerate(e["layer_types"]) if t == "full_attention"] == [3, 7, 11, 15, 19, 23, 26]
    assert e["mla"] == {"heads": 32, "k_dim": 192, "v_dim": 128}
    assert e["kda"] == {"heads": 32, "k_dim": 128, "v_dim": 128, "conv_width": 4, "conv_channels": 4096}


def test_expected_shapes_refuses_a_layer_in_both_lists_or_in_neither(sh):
    bad = json.loads(json.dumps(KIMI_CFG))
    bad["linear_attn_config"]["kda_layers"].remove(2)          # layer 2 in neither list
    with pytest.raises(ValueError):
        sh.expected_shapes(bad)
    bad = json.loads(json.dumps(KIMI_CFG))
    bad["linear_attn_config"]["kda_layers"].append(4)          # layer 4 in both
    with pytest.raises(ValueError):
        sh.expected_shapes(bad)


def test_check_shapes_names_every_mismatch_and_passes_the_expected_cache(sh):
    e = sh.expected_shapes(KIMI_CFG)
    T = 1024
    observed = sh.observed_shapes_from_lists(
        key_cache=[(1, 32, T, 192) if t == "full_attention" else None for t in e["layer_types"]],
        value_cache=[(1, 32, T, 128) if t == "full_attention" else None for t in e["layer_types"]],
        recurrent=[("float32", (1, 32, 128, 128)) if t == "linear_attention" else None for t in e["layer_types"]],
        conv=[[(1, 4096, 4)] * 3 if t == "linear_attention" else None for t in e["layer_types"]],
    )
    assert sh.check_shapes(observed, e, T) == []
    observed["recurrent"][0] = ("bfloat16", (1, 32, 128, 128))
    observed["key_cache"][3] = (1, 32, T, 72)                      # the top-level head_dim, the wrong one
    problems = sh.check_shapes(observed, e, T)
    assert len(problems) == 2
    assert any("layer 0" in p and "float32" in p for p in problems)
    assert any("layer 3" in p and "192" in p for p in problems)


# --- scoring input indexing (the one place the tool resolves the spec) --------------------------------------

def test_scored_input_caches_prefix_minus_one_rows_and_seeds_from_the_last_prefix_token(sh):
    seq = np.arange(100, 112)
    cached_rows, inp, cont = sh.scored_input(seq, prefix_len=8)
    assert cached_rows == 7
    assert cont.tolist() == [108, 109, 110, 111]
    assert inp.tolist() == [107, 108, 109, 110], "logits[t] must produce cont[t]; the last target is never fed"


def test_scored_input_refuses_a_prefix_that_leaves_no_continuation(sh):
    with pytest.raises(ValueError):
        sh.scored_input(np.arange(8), prefix_len=8)
    cached_rows, inp, cont = sh.scored_input(np.arange(8), prefix_len=7)   # one token of continuation is scoreable
    assert cached_rows == 6 and inp.tolist() == [6] and cont.tolist() == [7]


# --- arms ------------------------------------------------------------------------------------------------------

def test_arms_and_sources_agree_with_hybrid_state(sh):
    assert tuple(sh.ARMS) == hybrid_state.ARMS + ("ONE-PASS",)
    for arm in hybrid_state.ARMS:
        assert sh.SOURCES[arm] == hybrid_state._SOURCES[arm]
    assert sh.SOURCES["ONE-PASS"] == {"mla": "none", "kda": "none"}, "the floor arm injects nothing"


# --- the dump layout: the tool's own writer/reader against lag_ladder.hybrid_state ------------------------------

def _fake_cache_lists(P=12, seed=0):
    m = FakeHybrid(seed)
    tokens = np.arange(P) % 11
    cache = m.prefill(tokens)
    # the real cache carries a batch axis and [B, H, T, D] K/V; emulate that from the fake's [T, H, D]
    key = [None if k is None else np.transpose(k, (1, 0, 2))[None].astype(np.float32) for k in cache.key_cache]
    val = [None if v is None else np.transpose(v, (1, 0, 2))[None].astype(np.float32) for v in cache.value_cache]
    rec = [None if s is None else s[None] for s in cache.recurrent_states]
    conv = [None if c is None else tuple(x[None].astype(np.float32) for x in c) for c in cache.conv_states]
    return key, val, rec, conv


def test_tool_dump_loads_through_hybrid_state_with_bf16_bits_for_the_activation_kinds(sh, tmp_path):
    key, val, rec, conv = _fake_cache_lists()
    arrays = sh.arrays_from_lists(key, val, rec, conv, LAYER_TYPES, to_numpy=sh.numpy_from_array,
                                  activation_dtype="bfloat16")
    meta = sh.write_dump(tmp_path / "d", arrays, LAYER_TYPES, cached_rows=12,
                         extra={"state_v_first": False, "semantic_prefix_len": 13})
    d = hybrid_state.load_dump(tmp_path / "d")
    assert d.prefix_len == 12 and d.layer_types == list(LAYER_TYPES)
    K, V = d.mla[0]
    assert K.dtype == np.uint16 and K.shape == (12, 2, 6) and V.shape == (12, 2, 4)
    assert d.meta["mla"]["dtype"] == "bfloat16" and d.meta["kda"]["conv_dtype"] == "bfloat16"
    S, cq = d.kda[1][0], d.kda[1][1][0]
    assert S.dtype == np.float32 and S.shape == (2, 3, 3) and cq.dtype == np.uint16 and cq.shape == (6, 4)
    assert d.meta["extra"] == {"state_v_first": False, "semantic_prefix_len": 13}
    assert meta == d.meta


def test_tool_reader_restores_batch_axis_and_layout_and_round_trips_bf16_exactly(sh, tmp_path):
    key, val, rec, conv = _fake_cache_lists()
    # make the activations bf16-representable first so the round trip is exact, as it is for a bf16 model
    key = [None if k is None else sh.bf16_bits_to_f32(sh.f32_to_bf16_bits(k)) for k in key]
    val = [None if v is None else sh.bf16_bits_to_f32(sh.f32_to_bf16_bits(v)) for v in val]
    conv = [None if c is None else tuple(sh.bf16_bits_to_f32(sh.f32_to_bf16_bits(x)) for x in c) for c in conv]
    arrays = sh.arrays_from_lists(key, val, rec, conv, LAYER_TYPES, to_numpy=sh.numpy_from_array,
                                  activation_dtype="bfloat16")
    sh.write_dump(tmp_path / "d", arrays, LAYER_TYPES, cached_rows=12, extra={})
    dump = sh.read_dump(tmp_path / "d")
    unpack = lambda a, kind: sh.bf16_bits_to_f32(a) if a.dtype == np.uint16 else a   # the torch converter's job
    k2, v2, r2, c2 = sh.lists_from_arrays(dump, from_numpy=unpack)
    assert k2[1] is None and r2[0] is None
    assert k2[0].shape == (1, 2, 12, 6) and np.array_equal(k2[0], key[0])
    assert v2[0].shape == (1, 2, 12, 4) and np.array_equal(v2[0], val[0])
    assert r2[1].shape == (1, 2, 3, 3) and r2[1].dtype == np.float32 and np.array_equal(r2[1], rec[1])
    assert len(c2[1]) == 3 and c2[1][2].shape == (1, 6, 4) and np.array_equal(c2[1][2], conv[1][2])


def test_tool_reader_refuses_a_dump_whose_sha256_moved(sh, tmp_path):
    key, val, rec, conv = _fake_cache_lists()
    arrays = sh.arrays_from_lists(key, val, rec, conv, LAYER_TYPES, to_numpy=sh.numpy_from_array,
                                  activation_dtype="bfloat16")
    sh.write_dump(tmp_path / "d", arrays, LAYER_TYPES, cached_rows=12, extra={})
    p = tmp_path / "d" / "kda_layer01.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        sh.read_dump(tmp_path / "d")


def test_compose_lists_takes_each_kind_from_the_arm_source(sh, tmp_path):
    for name, seed in (("fresh", 0), ("stale", 1)):
        key, val, rec, conv = _fake_cache_lists(seed=seed)
        arrays = sh.arrays_from_lists(key, val, rec, conv, LAYER_TYPES, to_numpy=sh.numpy_from_array,
                                      activation_dtype="bfloat16")
        sh.write_dump(tmp_path / name, arrays, LAYER_TYPES, cached_rows=12, extra={})
    fresh, stale = sh.read_dump(tmp_path / "fresh"), sh.read_dump(tmp_path / "stale")
    ident = lambda a, kind: a
    kf, _, rf, _ = sh.lists_from_arrays(fresh, ident)
    ks, _, rs, _ = sh.lists_from_arrays(stale, ident)
    k, v, r, c = sh.compose_lists("MLA-STALE", fresh, stale, ident)
    assert np.array_equal(k[0], ks[0]) and np.array_equal(r[1], rf[1])
    k, v, r, c = sh.compose_lists("KDA-STALE", fresh, stale, ident)
    assert np.array_equal(k[0], kf[0]) and np.array_equal(r[1], rs[1])
    with pytest.raises(ValueError):
        sh.compose_lists("ONE-PASS", fresh, stale, ident)


# --- the report ------------------------------------------------------------------------------------------------

def test_report_fields_carry_the_contract_keys_and_the_kl_direction(sh):
    rng = np.random.default_rng(0)
    lf, ls = rng.normal(size=(4, 11)), rng.normal(size=(4, 11))
    cont = np.array([1, 5, 9, 2])
    fields = sh.score_fields(lf, ls, cont)
    assert set(fields) == {"continuation_tokens", "continuation_sha256", "logprob_fresh", "logprob_stale",
                           "kl_per_token", "argmax_fresh", "argmax_stale", "max_abs_logit_diff"}
    assert np.allclose(fields["kl_per_token"], hybrid_state.kl_per_token(lf, ls))
    assert fields["argmax_fresh"] == lf.argmax(-1).tolist()
    assert fields["logprob_fresh"][2] == pytest.approx(
        float(lf[2, 9] - np.log(np.exp(lf[2]).sum())))
    assert fields["continuation_sha256"] == __import__("hashlib").sha256(
        np.asarray(cont, dtype=np.int64).tobytes()).hexdigest()


def test_report_keys_are_the_reader_report_keys_plus_the_hybrid_axis(sh):
    assert set(sh.REPORT_KEYS) == HYBRID_REPORT_KEYS
    assert sh.SCHEMA == "hybrid-report-v1"


# --- config compatibility (writer vs reader) ------------------------------------------------------------------

def test_configs_compatible_passes_identical_and_names_the_field_that_differs(sh):
    assert sh.config_mismatches(KIMI_CFG, KIMI_CFG) == []
    other = json.loads(json.dumps(KIMI_CFG))
    other["rope_theta"] = 1e6
    other["linear_attn_config"]["num_heads"] = 16
    problems = sh.config_mismatches(KIMI_CFG, other)
    assert any("rope_theta" in p for p in problems) and any("num_heads" in p for p in problems)


# --- the CLI surface (needs torch; no model) ----------------------------------------------------------------------

def test_cli_names_the_contract_flags():
    pytest.importorskip("torch", reason="the scorer runs with the pod's interpreter; torch is not pinned here")
    import subprocess
    r = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for sub in ("shapes", "dump", "score"):
        assert sub in r.stdout
