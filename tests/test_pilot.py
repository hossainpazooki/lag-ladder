"""The pilot driver: the plan, the gate's refusals in order, and a run against the fake instrument."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lag_ladder.config import load_pilot_config
from lag_ladder.pilot import (
    UPSTREAM_PATHS, DumpJob, Refusal, assert_ready, dump_dir, plan, run, score_dir, steps,
)
from lag_ladder.seal import write_prediction
from tests.conftest import commit_all, git
from tests.fake_instrument import FakeInstrument

CONFIG = '''
[pilot]
source = "allenai/OLMo-2-0425-1B-RLVR1"
base = "allenai/OLMo-2-0425-1B-DPO"
revisions = ["step_200", "step_400", "step_600"]
stride_steps = 200
anchors = ["step_200"]
results_dir = "results/fstar"
upstream_path = "../upstream"
upstream_sha = "{sha}"
pair = "olmo2-1b-rlvr1"
tokens = "data/tokens/olmo2-1b-rlvr1_n4_len8_seed0.npy"
tokens_sha256 = "{tsha}"
heldout = [1, 4]
dtype = "float32"
seed = 0
registered_by = "{reg}"
[pilot.rule]
statistic = "median f*(tau_K)"
tau_K = 0.3186
tau_ladder = [0.3186, 0.10, 0.03]
holds_max = 0.15
degrades_min = 0.50
[pilot.controls]
identity_max = 0.0
scrambled_min = 0.90
'''
LEDGER = "# Ledger\n\n## Entries\n\n### 0005 — 2026-10-09 — Pilot registered\n\nregisters config/pilot.toml.\n"


def _upstream(root: Path) -> tuple[Path, str]:
    """A git checkout with the four (A) paths committed, a tokens file and a stand-in interpreter."""
    up = root / "upstream"
    if (up / ".git").exists():                      # a test that sets up twice reuses the checkout
        return up, git(up, "rev-parse", "HEAD").strip()
    for rel in UPSTREAM_PATHS:
        (up / rel).parent.mkdir(parents=True, exist_ok=True)
        (up / rel).write_text(f"# {rel}\n")
    git(up, "init", "-q", "-b", "main")
    git(up, "add", "-A")
    git(up, "commit", "-q", "-m", "pin")
    sha = git(up, "rev-parse", "HEAD").strip()
    tok = up / "data" / "tokens" / "olmo2-1b-rlvr1_n4_len8_seed0.npy"
    tok.parent.mkdir(parents=True)
    np.save(tok, np.arange(32, dtype=np.int64).reshape(4, 8))
    (up / ".venv" / "Scripts").mkdir(parents=True)
    (up / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    return up, sha


def _setup(repo: Path, seal_cfg, *, registered="0005", pin=None, tsha=None, seal=True, commit=True):
    up, sha = _upstream(repo.parent)
    tsha = tsha or hashlib.sha256((up / "data" / "tokens" / "olmo2-1b-rlvr1_n4_len8_seed0.npy").read_bytes()).hexdigest()
    (repo / "config").mkdir(exist_ok=True)
    cpath = repo / "config" / "pilot.toml"
    cpath.write_text(CONFIG.format(sha=pin or sha, tsha=tsha, reg=registered), encoding="utf-8")
    (repo / "ledger").mkdir(exist_ok=True)
    (repo / "ledger" / "ledger.md").write_text(LEDGER, encoding="utf-8")
    if seal:
        write_prediction("olmo2-1b-rlvr1", {"median_fstar_tau_K_at_lag_400": 0.2}, seal_cfg)
    if commit:
        commit_all(repo, "register")
    return load_pilot_config(cpath, repo)


def test_steps_reads_the_lag_from_the_revision_name():
    assert steps("step_200") == 200 and steps("step_2600") == 2600
    with pytest.raises(Refusal, match="step_<int>"):
        steps("main")


def test_plan_orders_controls_before_the_ladder_and_dedups_dumps(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg, commit=False)
    p = plan(cfg)
    kinds = [s.kind for s in p.scores]
    assert kinds == ["identity"] * 3 + ["scrambled"] * 3 + ["ladder"] * 6
    assert [s.lag for s in p.scores if s.kind == "ladder"] == [200, 200, 200, 400, 400, 400]
    scr = [s for s in p.scores if s.kind == "scrambled"]
    assert [(s.src.seq, s.tgt.seq) for s in scr] == [(1, 2), (2, 3), (3, 1)]
    assert set(p.dumps) == {DumpJob(r, i) for r in ("step_200", "step_400", "step_600") for i in (1, 2, 3)}
    assert score_dir(cfg, scr[0]).as_posix().endswith("results/fstar/olmo2-1b-rlvr1/step_200-scrambled/seq1")
    assert dump_dir(cfg, p.dumps[0]).as_posix().endswith("results/fstar/olmo2-1b-rlvr1/dumps/step_200/seq1")


def test_gate_refuses_unregistered_config_first(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg, registered="", seal=False)
    with pytest.raises(Refusal, match="UNREGISTERED"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)


def test_gate_refuses_tokens_placeholder(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg, tsha="TOKENS_SHA256_PENDING", seal=False)
    with pytest.raises(Refusal, match="tokens_sha256 placeholder"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)


def test_gate_refuses_uncommitted_config_and_missing_entry(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg, commit=False)
    with pytest.raises(Refusal, match="not committed as-is"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)
    commit_all(repo, "register")
    cfg = _setup(repo, seal_cfg, registered="0006", seal=False)
    with pytest.raises(Refusal, match="no entry 0006"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)


def test_gate_refuses_a_moved_pin_then_a_wrong_token_hash_then_a_missing_seal(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg, pin="0" * 40, seal=False)
    with pytest.raises(Refusal, match="not an ancestor"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)
    cfg = _setup(repo, seal_cfg, tsha="1" * 64, seal=False)
    with pytest.raises(Refusal, match="hashes to"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)
    cfg = _setup(repo, seal_cfg, seal=False)
    with pytest.raises(Refusal, match="no committed seal"):
        assert_ready(cfg, repo, seal_cfg=seal_cfg)


def test_run_records_every_job_and_resumes(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)
    assert_ready(cfg, repo, seal_cfg=seal_cfg)
    fake = FakeInstrument()
    rec = run(cfg, repo, runner=fake, limit=4, seal_cfg=seal_cfg)
    assert len(rec["jobs"]) == 4 and rec["finished"] and rec["halted"] is None
    n_before = len(fake.calls)
    rec = run(cfg, repo, runner=fake, seal_cfg=seal_cfg)
    assert len(rec["jobs"]) == 12
    dumps = [c for c in fake.calls[n_before:] if c[2] == "scripts.dump_kv"]
    assert all(("--revision" in c and "--dtype" in c and c[c.index("--stride") + 1] == "1") for c in dumps)
    assert len({tuple(c[c.index("--out") + 1:]) for c in dumps}) == len(dumps), "a dump directory was dumped twice"
    kinds = [j["kind"] for j in rec["jobs"]]
    assert kinds == ["identity"] * 3 + ["scrambled"] * 3 + ["ladder"] * 6
    assert all(j["fstar"]["0.3186"] == 0.0 for j in rec["jobs"] if j["kind"] == "identity")
    assert all(j["fstar"]["0.3186"] >= 0.9 for j in rec["jobs"] if j["kind"] == "scrambled")
    lag200 = [j["fstar"]["0.3186"] for j in rec["jobs"] if j["kind"] == "ladder" and j["lag"] == 200]
    lag400 = [j["fstar"]["0.3186"] for j in rec["jobs"] if j["kind"] == "ladder" and j["lag"] == 400]
    assert max(lag200) == 0.0 and min(lag400) > 0.0
    saved = json.loads((cfg.results_dir / cfg.pair / "run.json").read_text())
    assert saved["jobs"] == rec["jobs"] and saved["tokens_sha256"] == cfg.tokens_sha256
    assert (cfg.results_dir / cfg.pair / "tokens" / "pairs_identity_T8.npz").exists()


def test_run_halts_on_the_identity_control(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)
    fake = FakeInstrument(deviation=lambda sm, tm, i, j: 0.5)   # even the anchor against itself deviates
    with pytest.raises(Refusal, match="identity control failed"):
        run(cfg, repo, runner=fake, seal_cfg=seal_cfg)
    saved = json.loads((cfg.results_dir / cfg.pair / "run.json").read_text())
    assert saved["halted"] and "identity" in saved["halted"] and saved["finished"] is None
    assert len(saved["jobs"]) == 1


def test_run_halts_on_the_scrambled_control(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)
    fake = FakeInstrument(deviation=lambda sm, tm, i, j: 0.0)   # nothing ever deviates: the instrument is blind
    with pytest.raises(Refusal, match="scrambled control failed"):
        run(cfg, repo, runner=fake, seal_cfg=seal_cfg)


def test_run_refuses_a_failing_upstream_command(repo, seal_cfg):
    cfg = _setup(repo, seal_cfg)

    def bad(cmd, cwd=None, capture_output=False, **kw):
        return subprocess.CompletedProcess(cmd, 1, b"", b"boom")

    with pytest.raises(Refusal, match="upstream command failed"):
        run(cfg, repo, runner=bad, seal_cfg=seal_cfg)
