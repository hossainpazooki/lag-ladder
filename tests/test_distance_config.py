"""The weight-distance config: refs, rungs, pairs, the noise backbone, and every refusal."""
import pytest

from lag_ladder import REPO_ROOT
from lag_ladder.config import Ref, load_distance_config

DISTANCE = '''
[distance]
results_dir = "results/distance"
upstream_path = "../upstream"
upstream_sha = "UPSTREAM_SHA_PENDING"
heldout = [1, 4]
dtype = "float32"
seed = 0
registered_by = ""
[distance.rule]
statistic = "median f*(tau_K) per rung"
tau_K = 0.3186
tau_ladder = [0.3186, 0.10, 0.03]
holds_max = 0.15
degrades_min = 0.50
[distance.controls]
identity_max = 0.0
scrambled_min = 0.90
[distance.noise]
pair = "olmo2-1b-rlvr1"
which = "source"
anchor_revision = "step_200"
rel_norms = [1e-3, 1e-1]
[[distance.pairs]]
name = "olmo2-1b-rlvr1"
tokens = "data/tokens/olmo.npy"
tokens_sha256 = "TOKENS_SHA256_PENDING"
[[distance.pairs]]
name = "qwen2.5-1.5b-to-instruct"
tokens = "data/tokens/qwen.npy"
tokens_sha256 = "TOKENS_SHA256_PENDING"
[[distance.rungs]]
label = "olmo-sft-to-dpo"
pair = "olmo2-1b-rlvr1"
writer = { which = "source", local_path = "olmo2-1b-sft" }
reader = { which = "target", local_path = "olmo2-1b-dpo" }
[[distance.rungs]]
label = "qwen-base-to-instruct"
pair = "qwen2.5-1.5b-to-instruct"
writer = { which = "source" }
reader = { which = "target" }
'''


def _write(tmp_path, text):
    p = tmp_path / "distance.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_distance_config_loads_rungs_then_noise_rungs(tmp_path):
    cfg = load_distance_config(_write(tmp_path, DISTANCE), tmp_path)
    assert [r.label for r in cfg.rungs] == ["olmo-sft-to-dpo", "qwen-base-to-instruct", "noise-0.001", "noise-0.1"]
    real, noise = cfg.rungs[0], cfg.rungs[-1]
    assert real.kind == "real" and real.writer == Ref("source", None, "olmo2-1b-sft") and real.reader == Ref("target", None, "olmo2-1b-dpo")
    assert noise.kind == "noise" and noise.rel_norm == 0.1 and noise.pair == "olmo2-1b-rlvr1"
    assert noise.writer == Ref("source", "step_200", None) and noise.reader == Ref("source", None, "noise-0.1")
    assert cfg.pairs["qwen2.5-1.5b-to-instruct"].tokens == (tmp_path / "../upstream").resolve() / "data/tokens/qwen.npy"
    assert cfg.heldout == (1, 4) and cfg.registered_by == "" and cfg.noise["rel_norms"] == [0.001, 0.1]


@pytest.mark.parametrize("old, new, msg", [
    ('reader = { which = "target" }', 'reader = { which = "source" }', "same ref"),
    ('writer = { which = "source", local_path = "olmo2-1b-sft" }', 'writer = { which = "source", local_path = "olmo2-1b-sft", revision = "x" }', "not both"),
    ('local_path = "olmo2-1b-dpo"', 'local_path = "../olmo2-1b-dpo"', "plain name"),
    ('label = "qwen-base-to-instruct"', 'label = "olmo-sft-to-dpo"', "unique"),
    ('label = "qwen-base-to-instruct"', 'label = "noise-0.5"', "noise-"),
    ('pair = "qwen2.5-1.5b-to-instruct"\nwriter', 'pair = "nope"\nwriter', "not in"),
    ('rel_norms = [1e-3, 1e-1]', 'rel_norms = [1e-1, 1e-3]', "strictly increasing"),
    ('rel_norms = [1e-3, 1e-1]', 'rel_norms = [1e-3, 1.0]', "in \\(0, 1\\)"),
    ('anchor_revision = "step_200"', 'anchor_revision = "a/b"', "plain name"),
    ('heldout = [1, 4]', 'heldout = [3, 4]', "at least two"),
    ('tokens_sha256 = "TOKENS_SHA256_PENDING"\n[[distance.pairs]]\nname = "qwen', 'tokens_sha256 = "zz"\n[[distance.pairs]]\nname = "qwen', "64-hex"),
    ('registered_by = ""', 'registered_by = "7"', "four-digit"),
])
def test_distance_config_refuses_bad_values(tmp_path, old, new, msg):
    assert DISTANCE.count(old) == 1, old
    with pytest.raises(ValueError, match=msg):
        load_distance_config(_write(tmp_path, DISTANCE.replace(old, new)), tmp_path)


def test_repo_distance_config_loads_unregistered():
    cfg = load_distance_config(REPO_ROOT / "config" / "distance.toml", REPO_ROOT)
    assert cfg.registered_by == "0008", "the distance config's registering entry changed: update this test"
    assert len(cfg.noise["rel_norms"]) == 7 and sum(r.kind == "noise" for r in cfg.rungs) == 7
    assert {r.pair for r in cfg.rungs} <= set(cfg.pairs)
