import pytest

from lag_ladder.config import load_pilot_config, load_seal_config


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_seal_config_expands_upstream_and_resolves_relative_to_root(tmp_path):
    p = _write(tmp_path, "seal.toml", '''
predictions_dir = "ledger/predictions"
upstream_path = "../up"
[[artifact_roots]]
path = "mappers"
pattern = "{pair}/**/k*.safetensors"
[[artifact_roots]]
path = "${upstream}/mappers"
pattern = "{pair}/**/k*.safetensors"
''')
    cfg = load_seal_config(p, repo_root=tmp_path / "repo")
    assert cfg.predictions_dir == (tmp_path / "repo" / "ledger" / "predictions")
    assert cfg.artifact_roots[0].path == tmp_path / "repo" / "mappers"
    assert cfg.artifact_roots[1].path == (tmp_path / "repo" / ".." / "up" / "mappers").resolve()
    assert "{pair}" in cfg.artifact_roots[1].pattern


def test_seal_config_refuses_pattern_without_pair_placeholder(tmp_path):
    p = _write(tmp_path, "seal.toml", '''
predictions_dir = "ledger/predictions"
upstream_path = "../up"
[[artifact_roots]]
path = "mappers"
pattern = "**/k*.safetensors"
''')
    with pytest.raises(ValueError, match="{pair}"):
        load_seal_config(p, repo_root=tmp_path)


PILOT = '''
[pilot]
source = "allenai/OLMo-2-0425-1B-RLVR1"
base = "allenai/OLMo-2-0425-1B-DPO"
revisions = ["step_200", "step_400", "step_600"]
stride_steps = 200
anchors = ["step_200"]
results_dir = "results/fstar"
upstream_path = "../kv-transfer-replication"
upstream_sha = "UPSTREAM_SHA_PENDING"
seed = 7
registered_by = ""
[pilot.rule]
statistic = "median f*(tau_K)"
tau_K = 0.3186
tau_ladder = [0.3186, 0.10, 0.03]
holds_max = 0.15
degrades_min = 0.50
'''


def test_pilot_config_loads_and_records_unregistered(tmp_path):
    cfg = load_pilot_config(_write(tmp_path, "pilot.toml", PILOT), repo_root=tmp_path)
    assert cfg.revisions == ("step_200", "step_400", "step_600") and cfg.anchors == ("step_200",)
    assert cfg.results_dir == tmp_path / "results" / "fstar" and cfg.seed == 7
    assert cfg.registered_by == "" and cfg.rule["tau_K"] == 0.3186


@pytest.mark.parametrize("old, new, msg", [
    ('anchors = ["step_200"]', 'anchors = ["step_999"]', "anchor"),
    ('tau_ladder = [0.3186, 0.10, 0.03]', 'tau_ladder = [0.10, 0.03]', "start at tau_K"),
    ('tau_ladder = [0.3186, 0.10, 0.03]', 'tau_ladder = [0.3186, 0.03, 0.10]', "strictly decreasing"),
    ('holds_max = 0.15', 'holds_max = 0.60', "holds_max < degrades_min"),
    ('registered_by = ""', 'registered_by = "12"', "four-digit"),
    ('seed = 7', 'seed = true', "seed"),
    ('revisions = ["step_200", "step_400", "step_600"]', 'revisions = ["step_200", "step_200"]', "distinct"),
])
def test_pilot_config_refuses_bad_values(tmp_path, old, new, msg):
    assert PILOT.count(old) == 1
    with pytest.raises(ValueError, match=msg):
        load_pilot_config(_write(tmp_path, "pilot.toml", PILOT.replace(old, new)), repo_root=tmp_path)


def test_repo_configs_load():
    from lag_ladder import REPO_ROOT
    load_seal_config(REPO_ROOT / "config" / "seal.toml", REPO_ROOT)
    cfg = load_pilot_config(REPO_ROOT / "config" / "pilot.toml", REPO_ROOT)
    assert cfg.registered_by == "", "the pilot config is registered: update this test with the entry number"
    assert len(cfg.revisions) == 13 and cfg.stride_steps == 200
