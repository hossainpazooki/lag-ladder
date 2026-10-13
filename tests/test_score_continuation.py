"""Pure-math units for the (B) scorer contract (design section 13, interface reader-report-v1): the KL
direction, the RoPE apply/strip algebra, and the report schema. No model is downloaded and no real forward
runs here; anything that needs a model is skipped and runs on the pod."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import lag_ladder

# The exact top-level keys of report.json (contract interface 1); the summarizer and the driver depend on them.
REPORT_KEYS = frozenset({
    "schema", "reader", "fresh_dump", "stale_dump", "tokens_file", "seq", "prefix_len",
    "text", "generate_under", "live",
    "continuation_tokens", "continuation_sha256",
    "logprob_fresh", "logprob_stale", "kl_per_token",
    "argmax_fresh", "argmax_stale", "max_abs_logit_diff",
    "inputs_sha256", "reader_checkpoint",
    "device", "torch_version", "peak_cuda_mem_bytes", "gen_seconds",
})

SCORER_FLAGS = ("--reader", "--fresh", "--stale", "--tokens", "--seq", "--prefix-len",
                "--text", "--generate-under", "--live", "--probe", "--out", "--device", "--dtype")


# --- the contract formulas, as numpy references -------------------------------------------------------------

def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def kl_fresh_stale(logits_fresh, logits_stale) -> float:
    """KL(pi_fresh || pi_stale) = sum_v p_fresh * (logp_fresh - logp_stale), natural log, full vocabulary."""
    lf, ls = _log_softmax(logits_fresh), _log_softmax(logits_stale)
    return float((np.exp(lf) * (lf - ls)).sum())


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _cos_sin(spec: dict, positions):
    inv = np.asarray(spec["inv_freq"], dtype=np.float64)
    freqs = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb), float(spec.get("attention_scaling", 1.0))


def rope_apply(x, spec, positions):
    """The model writes K = m * R(pos) * k_content (transformers scales cos/sin by attention_scaling)."""
    cos, sin, m = _cos_sin(spec, positions)
    return (x * cos + rotate_half(x) * sin) * m


def rope_strip(x, spec, positions):
    """Exact inverse of rope_apply: R^T * x / m (the instrument's strip divides by attention_scaling)."""
    cos, sin, m = _cos_sin(spec, positions)
    return (x * cos - rotate_half(x) * sin) / m


def _spec(theta: float, d_h: int, m: float = 1.0) -> dict:
    inv_freq = 1.0 / (theta ** (np.arange(0, d_h, 2, dtype=np.float64) / d_h))
    return {"inv_freq": inv_freq.tolist(), "attention_scaling": m}


# --- KL ------------------------------------------------------------------------------------------------------

def test_kl_is_zero_on_identical_logits_and_shift_invariant():
    logits = np.array([0.3, -1.2, 2.0, 0.0])
    assert kl_fresh_stale(logits, logits) == pytest.approx(0.0, abs=1e-12)
    assert kl_fresh_stale(logits, logits + 7.5) == pytest.approx(0.0, abs=1e-9)


def test_kl_direction_is_fresh_over_stale():
    p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    expected = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
    got = kl_fresh_stale(np.log(p), np.log(q))
    assert got == pytest.approx(expected, rel=1e-9)
    # KL is asymmetric; a scorer that silently swapped the arms would read the other value
    assert got != pytest.approx(kl_fresh_stale(np.log(q), np.log(p)), rel=1e-3)
    assert got > 0.0


def test_kl_is_nonnegative_on_random_logits():
    rng = np.random.default_rng(0)
    for _ in range(20):
        f, s = rng.normal(size=32), rng.normal(size=32)
        assert kl_fresh_stale(f, s) >= 0.0


# --- RoPE ----------------------------------------------------------------------------------------------------

def test_rotate_half_twice_is_minus_identity():
    x = np.random.default_rng(1).normal(size=(3, 8))
    assert np.allclose(rotate_half(rotate_half(x)), -x)


def test_strip_inverts_apply_at_unit_attention_factor():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 16))
    spec = _spec(1e4, 16)
    pos = np.arange(5)
    assert np.allclose(rope_strip(rope_apply(x, spec, pos), spec, pos), x, atol=1e-12)


def test_strip_inverts_apply_at_scaled_attention_factor():
    # apply = m(x cos + Rx sin); strip divides by m, so the round trip is exact for any m. A strip that
    # multiplies instead (the naive sign flip) would return m^2 x and corrupt a YaRN writer's stale K.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 8))
    spec = _spec(1e4, 8, m=1.3)
    pos = np.arange(4)
    assert np.allclose(rope_strip(rope_apply(x, spec, pos), spec, pos), x, atol=1e-12)
    naive = rope_apply(x, spec, pos)
    cos, sin, m = _cos_sin(spec, pos)
    assert np.allclose((naive * cos - rotate_half(naive) * sin) * m, (1.3 ** 2) * x, atol=1e-12)


def test_cross_spec_rerotation_equals_direct_reader_rotation():
    # K on disk is rope-applied with the WRITER's spec; strip(writer) then apply(reader) must equal
    # applying the reader's rotation to the raw K. This is the qwen-base-to-math path (theta 1e6 vs 1e4).
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(6, 16))
    pos = np.arange(6)
    writer, reader = _spec(1e6, 16), _spec(1e4, 16)
    on_disk = rope_apply(raw, writer, pos)
    rerotated = rope_apply(rope_strip(on_disk, writer, pos), reader, pos)
    assert np.allclose(rerotated, rope_apply(raw, reader, pos), atol=1e-10)
    # same-spec dumps are injected AS IS: a needless strip+reapply is at best a no-op, never relied on
    assert np.allclose(rope_apply(rope_strip(on_disk, writer, pos), writer, pos), on_disk, atol=1e-10)


# --- the report schema, via the stand-in scorer --------------------------------------------------------------

def _dump_dir(root: Path, name: str, seq: int) -> Path:
    d = root / name / f"seq{seq}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "pair": "p", "which": "source", "dir": d.resolve().as_posix(),
        "checkpoint": {"kind": "hf", "model_id": f"fake/{name}", "revision": None, "local_checkpoint": None,
                       "manifest_digest": "d" * 64}}), encoding="utf-8")
    return d


def _run_fake(tmp_path, extra: list[str], out: Path):
    from tests.fake_instrument import FakeInstrument
    fresh = _dump_dir(tmp_path, "reader", 1)
    stale = _dump_dir(tmp_path, "writer", 1)
    tok = tmp_path / "tokens.npy"
    np.save(tok, np.arange(32, dtype=np.int64).reshape(4, 8))
    fake = FakeInstrument()
    cmd = ["python", "tools/score_continuation.py", "--reader", "fake/reader@main",
           "--fresh", str(fresh), "--stale", str(stale), "--tokens", str(tok),
           "--seq", "1", "--prefix-len", "6", "--text", "corpus", "--out", str(out), *extra]
    return fake(cmd), out / "report.json"


def test_standin_report_carries_exactly_the_contract_keys(tmp_path):
    r, report = _run_fake(tmp_path, [], tmp_path / "out")
    assert r.returncode == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert set(rep) == REPORT_KEYS
    assert rep["schema"] == "reader-report-v1" and rep["generate_under"] is None and rep["live"] is False
    assert rep["continuation_tokens"] == list(range(14, 16)), "corpus text is the sequence's own tail"
    assert len(rep["kl_per_token"]) == len(rep["logprob_fresh"]) == len(rep["continuation_tokens"])
    assert rep["reader_checkpoint"] == {"manifest_digest": "d" * 64}


def test_standin_probe_refuses_an_out_under_results(tmp_path):
    out = tmp_path / "results" / "reader" / "x"
    r, report = _run_fake(tmp_path, ["--probe"], out)
    assert r.returncode != 0 and not report.exists()


# --- the real scorer: CLI surface only (needs torch; no model) ----------------------------------------------

def test_scorer_cli_names_the_contract_flags():
    pytest.importorskip("torch", reason="the scorer runs with the instrument's interpreter; torch is not pinned here")
    tool = lag_ladder.REPO_ROOT / "tools" / "score_continuation.py"
    if not tool.exists():
        pytest.skip("tools/score_continuation.py not built yet")
    r = subprocess.run([sys.executable, str(tool), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in SCORER_FLAGS:
        assert flag in r.stdout, f"scorer --help does not name {flag}"


@pytest.mark.skip(reason="needs a real model forward; runs on the pod under the GPU protocol")
def test_injected_cache_reproduces_a_full_forward():
    pass
