import json

import pytest

from lag_ladder.hashing import hash_json_file
from lag_ladder.seal import (SealViolation, find_mapper_artifacts, prediction_path,
                                 require_sealed, sidecar_path, verify_all, write_prediction)
from tests.conftest import commit_all, git

PAIR = "qwen3-0.6b-to-4b"
PAYLOAD = {"readout": {"K": {"k1": 0.5}, "V": {"k1": 0.3}}, "screen_version": "0.0.1"}


def _plant_mapper(root, pair=PAIR, tag=None):
    d = root / "mappers" / pair / (tag or "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "k1.safetensors").write_bytes(b"\0")
    return d / "k1.safetensors"


# --- writer -------------------------------------------------------------------

def test_write_seals_when_no_artifact_exists(seal_cfg):
    digest = write_prediction(PAIR, PAYLOAD, seal_cfg)
    p, s = prediction_path(PAIR, seal_cfg), sidecar_path(PAIR, seal_cfg)
    assert p.exists() and s.read_text().strip() == digest == hash_json_file(p)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["sealed_pre_fit"] is True and rec["prior_artifacts"] == [] and rec["prediction"] == PAYLOAD


def test_writer_refuses_if_local_mapper_artifact_exists(repo, seal_cfg):
    """Invariant 1a."""
    hit = _plant_mapper(repo)
    with pytest.raises(SealViolation, match="k1.safetensors"):
        write_prediction(PAIR, PAYLOAD, seal_cfg)
    assert not prediction_path(PAIR, seal_cfg).exists()


def test_writer_refuses_if_upstream_mapper_artifact_exists(repo, seal_cfg):
    """The upstream tree is part of 'anywhere'. Tagged subdirs (mappers/<pair>/<tag>/) count too."""
    _plant_mapper(repo.parent / "upstream", tag="n200")
    with pytest.raises(SealViolation, match="n200"):
        write_prediction(PAIR, PAYLOAD, seal_cfg)


def test_writer_refuses_if_upstream_r2_report_exists(repo, seal_cfg):
    d = repo.parent / "upstream" / "results" / "mapper" / PAIR
    d.mkdir(parents=True)
    (d / "r2.json").write_text("{}")
    with pytest.raises(SealViolation, match="r2.json"):
        write_prediction(PAIR, PAYLOAD, seal_cfg)


def test_writer_fails_closed_on_missing_root(repo, seal_cfg):
    import shutil
    shutil.rmtree(repo.parent / "upstream" / "mappers")
    with pytest.raises(SealViolation, match="does not exist"):
        write_prediction(PAIR, PAYLOAD, seal_cfg)


def test_writer_refuses_to_overwrite(seal_cfg):
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    with pytest.raises(SealViolation, match="already sealed"):
        write_prediction(PAIR, {"different": 1}, seal_cfg)


def test_other_pairs_artifacts_do_not_block(repo, seal_cfg):
    _plant_mapper(repo, pair="qwen3-0.6b-to-1.7b")
    write_prediction(PAIR, PAYLOAD, seal_cfg)


# --- pair name validation (path escape) -----------------------------------------

@pytest.mark.parametrize("bad_pair", [
    "../../outside/pwned",
    "../escape",
    "sub/dir",
    "sub\\dir",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "..",
    ".",
    "",
])
def test_write_prediction_refuses_path_escaping_pair(seal_cfg, bad_pair):
    with pytest.raises(SealViolation):
        write_prediction(bad_pair, PAYLOAD, seal_cfg)
    # nothing must be created outside (or inside) predictions_dir
    assert list(seal_cfg.predictions_dir.iterdir()) == []


def test_write_prediction_refuses_path_escaping_pair_does_not_write_outside_tree(seal_cfg):
    outside_marker = seal_cfg.predictions_dir.parent.parent / "outside"
    with pytest.raises(SealViolation, match="path separator"):
        write_prediction("../../outside/pwned", PAYLOAD, seal_cfg)
    assert not outside_marker.exists()


# --- post-fit (decision D2) ---------------------------------------------------

def test_post_fit_records_prior_artifacts_and_is_not_pre_fit(repo, seal_cfg):
    hit = _plant_mapper(repo)
    write_prediction(PAIR, PAYLOAD, seal_cfg, post_fit=True)
    rec = json.loads(prediction_path(PAIR, seal_cfg).read_text(encoding="utf-8"))
    assert rec["sealed_pre_fit"] is False
    assert rec["prior_artifacts"] == [hit.resolve().as_posix()]


def test_post_fit_flag_refused_when_pair_is_actually_pre_fit(seal_cfg):
    with pytest.raises(SealViolation, match="post_fit"):
        write_prediction(PAIR, PAYLOAD, seal_cfg, post_fit=True)


def test_post_fit_record_only_consumable_with_allow_post_fit(repo, seal_cfg):
    _plant_mapper(repo)
    write_prediction(PAIR, PAYLOAD, seal_cfg, post_fit=True)
    commit_all(repo)
    with pytest.raises(SealViolation, match="post-fit"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)
    assert require_sealed(PAIR, seal_cfg, repo_root=repo, allow_post_fit=True)["sealed_pre_fit"] is False


# --- runner gate ----------------------------------------------------------------

def test_require_sealed_refuses_without_file(repo, seal_cfg):
    """Invariant 1b."""
    with pytest.raises(SealViolation, match="no sealed prediction"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)


def test_require_sealed_refuses_hash_mismatch(repo, seal_cfg):
    """Invariant 1b: tamper the payload, keep the sidecar."""
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo)
    p = prediction_path(PAIR, seal_cfg)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["prediction"]["readout"]["K"]["k1"] = 0.9
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SealViolation, match="hash"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)


def test_require_sealed_refuses_uncommitted(repo, seal_cfg):
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    with pytest.raises(SealViolation, match="not tracked"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)
    git(repo, "add", "-A")
    with pytest.raises(SealViolation, match="not tracked|differs from HEAD"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)
    git(repo, "commit", "-q", "-m", "seal")
    assert require_sealed(PAIR, seal_cfg, repo_root=repo)["pair"] == PAIR


def test_require_sealed_refuses_history_rewrite_of_a_seal(repo, seal_cfg):
    """A seal re-committed with a matching sidecar is still a broken seal: the file was
    modified after its sealing commit."""
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo, "seal")
    p, s = prediction_path(PAIR, seal_cfg), sidecar_path(PAIR, seal_cfg)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["prediction"]["readout"]["K"]["k1"] = 0.9
    from lag_ladder.hashing import canonical_bytes, hash_json_obj
    p.write_bytes(canonical_bytes(rec)); s.write_text(hash_json_obj(rec) + "\n")
    commit_all(repo, "tamper")
    with pytest.raises(SealViolation, match="modified after sealing"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)


def test_require_sealed_refuses_delete_then_readd(repo, seal_cfg):
    """CRITICAL bypass repro: `git log --diff-filter=M` never sees a D-then-A pair, so a
    prediction (and its sidecar, so the hash still matches and the earlier hash-mismatch
    check can't be what fires) deleted and re-created at the same paths with different
    content must still be refused -- both changed after the commit that sealed them."""
    from lag_ladder.hashing import canonical_bytes, hash_json_obj
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo, "seal")
    p, s = prediction_path(PAIR, seal_cfg), sidecar_path(PAIR, seal_cfg)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rel_p, rel_s = p.relative_to(repo).as_posix(), s.relative_to(repo).as_posix()
    git(repo, "rm", "-q", rel_p, rel_s)
    git(repo, "commit", "-q", "-m", "remove")
    rec["prediction"]["readout"]["K"]["k1"] = 999
    p.parent.mkdir(parents=True, exist_ok=True)  # git rm prunes the now-empty directory
    p.write_bytes(canonical_bytes(rec))
    s.write_text(hash_json_obj(rec) + "\n")
    commit_all(repo, "readd tampered")
    with pytest.raises(SealViolation, match="modified after sealing"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)


def test_require_sealed_refuses_rename_away_and_back(repo, seal_cfg):
    """CRITICAL bypass repro: a `git mv` away and back never shows an M status either, so a
    prediction (and its sidecar, kept in lock-step so the hash still matches) renamed away
    and back with tampered content must still be caught by the same history rule."""
    from lag_ladder.hashing import canonical_bytes, hash_json_obj
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo, "seal")
    p, s = prediction_path(PAIR, seal_cfg), sidecar_path(PAIR, seal_cfg)
    rel_p, rel_s = p.relative_to(repo).as_posix(), s.relative_to(repo).as_posix()
    tmp_p, tmp_s = rel_p + ".tmp", rel_s + ".tmp"
    git(repo, "mv", rel_p, tmp_p)
    git(repo, "mv", rel_s, tmp_s)
    git(repo, "commit", "-q", "-m", "rename away")
    git(repo, "mv", tmp_p, rel_p)
    git(repo, "mv", tmp_s, rel_s)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["prediction"]["readout"]["K"]["k1"] = 999
    p.write_bytes(canonical_bytes(rec))
    s.write_text(hash_json_obj(rec) + "\n")
    commit_all(repo, "rename back tampered")
    with pytest.raises(SealViolation, match="modified after sealing"):
        require_sealed(PAIR, seal_cfg, repo_root=repo)


def test_require_committed_can_be_relaxed_only_explicitly(repo, seal_cfg):
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    assert require_sealed(PAIR, seal_cfg, repo_root=repo, require_committed=False)["pair"] == PAIR


# --- CI verifier ------------------------------------------------------------------

def test_verify_all_ok_and_reports(repo, seal_cfg):
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo)
    out = verify_all(seal_cfg, repo)
    assert len(out) == 1 and out[0].startswith(f"OK {PAIR} ")


def test_verify_all_refuses_orphan_sidecar(repo, seal_cfg):
    sidecar_path(PAIR, seal_cfg).write_text("00" * 32 + "\n")
    commit_all(repo)
    with pytest.raises(SealViolation, match="orphan"):
        verify_all(seal_cfg, repo)


def test_verify_all_refuses_tampered(repo, seal_cfg):
    write_prediction(PAIR, PAYLOAD, seal_cfg)
    commit_all(repo)
    p = prediction_path(PAIR, seal_cfg)
    p.write_text(p.read_text(encoding="utf-8").replace("0.5", "0.6"), encoding="utf-8")
    commit_all(repo, "tamper")
    with pytest.raises(SealViolation, match="hash mismatch"):
        verify_all(seal_cfg, repo)


def test_verify_all_with_nothing_sealed_is_ok_and_says_so(repo, seal_cfg):
    assert verify_all(seal_cfg, repo) == ["OK (no sealed predictions yet)"]
