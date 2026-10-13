"""The (B) reader driver (contract interface 3): the gate's refusals in order, controls per reader before its
rungs, the halt and UNEVALUABLE paths, and the probe. Everything runs against the fake scorer; the dumps it
reads are the distance run's, as the design prescribes (reuse, never re-dump)."""
import json
from pathlib import Path

import numpy as np
import pytest

from lag_ladder import reader
from lag_ladder.config import load_reader_config
from lag_ladder.distance import run as distance_run
from lag_ladder.seal import write_prediction
from tests.conftest import commit_all, git
from tests.fake_instrument import FakeInstrument, default_kl
from tests.test_distance import _setup as _setup_distance, _token_shas

READER_CONFIG = '''
[reader]
results_dir = "results/reader"
upstream_path = "../upstream"
upstream_sha = "{sha}"
heldout = [1, 4]
prefix_len = 6
dtype = "float32"
seed = 3
registered_by = "{reg}"
[reader.rule]
statistic = "median mean per-token KL(fresh || stale) per rung per text"
holds_max = 1e-3
degrades_min = 1e-1
texts = ["corpus", "on-path"]
generate_under = "stale"
[reader.controls]
identity_max_logit_diff = 1e-5
scrambled_min = 1e-1
{floor}
[[reader.pairs]]
name = "olmo2-1b-rlvr1"
tokens = "data/tokens/olmo.npy"
tokens_sha256 = "{osha}"
[[reader.pairs]]
name = "qwen2.5-1.5b-to-instruct"
tokens = "data/tokens/qwen.npy"
tokens_sha256 = "{qsha}"
[[reader.rungs]]
label = "olmo-sft-to-dpo"
pair = "olmo2-1b-rlvr1"
reader = "local:olmo2-1b-dpo"
writer_dump = "source@local-olmo2-1b-sft"
reader_dump = "target@local-olmo2-1b-dpo"
[[reader.rungs]]
label = "qwen-base-to-instruct"
pair = "qwen2.5-1.5b-to-instruct"
reader = "fake/qwen2.5-1.5b-to-instruct-target"
writer_dump = "source@default"
reader_dump = "target@default"
[[reader.rungs]]
label = "backbone-1e-3"
pair = "olmo2-1b-rlvr1"
reader = "local:noise-0.001"
writer_dump = "source@step_200"
reader_dump = "source@local-noise-0.001"
[[reader.rungs]]
label = "backbone-1e-1"
pair = "olmo2-1b-rlvr1"
reader = "local:noise-0.1"
writer_dump = "source@step_200"
reader_dump = "source@local-noise-0.1"
'''
FLOOR_LINES = "floor_max_median_kl = 1e-4"
LEDGER_0013 = "\n### 0013 — 2026-10-13 — Reader registered\n\nregisters config/reader.toml.\n"

# the fake's known cells: mean KL = (20 x stand-in distance)^2 per text, 2.0 scrambled, 1e-6 at the floor
EXPECTED_KL = {"olmo-sft-to-dpo": 0.16, "qwen-base-to-instruct": 1.0, "backbone-1e-3": 4e-4, "backbone-1e-1": 4.0}


def _upstream_full(root: Path) -> None:
    """The distance fixture's upstream, plus the (B) recipe sources the reader gate may pin (kvt/cache.py,
    kvt/rope.py). Built before _setup so test_distance's _upstream reuses it."""
    up = root / "upstream"
    if (up / ".git").exists():
        return
    for rel in ("scripts/dump_kv.py", "kvt/data.py", "scripts/score_positions.py", "kvt/pertoken.py",
                "kvt/cache.py", "kvt/rope.py"):
        (up / rel).parent.mkdir(parents=True, exist_ok=True)
        (up / rel).write_text(f"# {rel}\n")
    (up / "data" / "tokens").mkdir(parents=True)
    np.save(up / "data" / "tokens" / "olmo.npy", np.arange(32, dtype=np.int64).reshape(4, 8))
    np.save(up / "data" / "tokens" / "qwen.npy", np.arange(100, 132, dtype=np.int64).reshape(4, 8))
    (up / ".venv" / "Scripts").mkdir(parents=True)
    (up / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    git(up, "init", "-q", "-b", "main")
    git(up, "add", "-A")
    git(up, "commit", "-q", "-m", "pin")


def _setup_reader(repo, seal_cfg, *, registered="0013", floor=True, seal=True):
    """Distance setup + run (so the dumps exist, noise weights kept), then the reader registration."""
    _upstream_full(repo.parent)
    dcfg, tools = _setup_distance(repo, seal_cfg)
    distance_run(dcfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools, prune_noise=False)
    (tools / "score_continuation.py").write_text("# stand-in\n")
    up = repo.parent / "upstream"
    shas = _token_shas(up)
    sha = git(up, "rev-parse", "HEAD").strip()
    cpath = repo / "config" / "reader.toml"
    cpath.write_text(READER_CONFIG.format(sha=sha, reg=registered, osha=shas["olmo"], qsha=shas["qwen"],
                                          floor=FLOOR_LINES if floor else ""), encoding="utf-8")
    lpath = repo / "ledger" / "ledger.md"
    lpath.write_text(lpath.read_text(encoding="utf-8") + LEDGER_0013, encoding="utf-8")
    if seal:
        write_prediction(reader.SEAL_NAME, {"floor_median_kl_probe": 1e-6}, seal_cfg)
    if git(repo, "status", "--porcelain").strip():
        commit_all(repo, "register reader")
    return load_reader_config(cpath, repo), tools


# --- reading the fake's call log by the scorer CLI (contract interface 1), not the driver's internals --------

def _parse(rest: list[str]) -> dict:
    args, i = {}, 0
    while i < len(rest):
        if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            args[rest[i]] = rest[i + 1]; i += 2
        else:
            args[rest[i]] = True; i += 1
    return args


def scorer_calls(fake: FakeInstrument) -> list[dict]:
    return [_parse(c[2:]) for c in fake.calls if len(c) > 1 and str(c[1]).endswith("score_continuation.py")]


def kind_of(a: dict) -> str:
    if "--live" in a:
        return "floor"
    fresh, stale = Path(a["--fresh"]), Path(a["--stale"])
    if fresh == stale:
        return "identity"
    if fresh.parent == stale.parent:
        return "scrambled"          # the reader's own ref, another sequence's prefix
    return "rung"


def reader_ref(a: dict) -> Path:
    """The reader's dump ref dir: the fresh arm's parent (its own cache), or the stale arm's under --live."""
    return Path(a["--stale"]).parent if "--live" in a else Path(a["--fresh"]).parent


# --- the gate -------------------------------------------------------------------------------------------------

def test_run_refuses_unregistered_before_anything_else(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg, registered="", floor=False, seal=False)
    with pytest.raises(reader.Refusal, match="UNREGISTERED"):
        reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)


def test_run_refuses_while_the_floor_limits_are_unset(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg, floor=False)
    with pytest.raises(reader.Refusal, match="floor"):
        reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)


def test_run_refuses_without_a_committed_seal(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg, seal=False)
    with pytest.raises(reader.Refusal, match="seal"):
        reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)


# --- the run --------------------------------------------------------------------------------------------------

def test_run_scores_controls_per_reader_before_its_rungs(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg)
    fake = FakeInstrument()
    rec = reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["finished"] and rec["halted"] is None
    for key in ("config_sha256", "upstream_sha", "tokens_sha256", "started", "jobs"):
        assert key in rec, f"run record is missing {key}"
    calls = scorer_calls(fake)
    kinds = [kind_of(a) for a in calls]
    assert {"floor", "identity", "scrambled", "rung"} <= set(kinds)
    # per reader ref: every control before the first rung cell that reads through it
    by_ref: dict[Path, dict[str, list[int]]] = {}
    for i, a in enumerate(calls):
        by_ref.setdefault(reader_ref(a), {}).setdefault(kind_of(a), []).append(i)
    rung_refs = [r for r, k in by_ref.items() if "rung" in k]
    assert rung_refs, "no rung cells were scored"
    for r in rung_refs:
        controls = [i for kind in ("floor", "identity", "scrambled") for i in by_ref[r].get(kind, [])]
        assert controls, f"reader {r.name} has rungs but no controls"
        assert max(controls) < min(by_ref[r]["rung"]), f"a rung of reader {r.name} ran before its controls"
    # both texts per rung ref; on-path cells name the registered arm to generate under
    rung_calls = [a for a in calls if kind_of(a) == "rung"]
    assert {a["--text"] for a in rung_calls} == {"corpus", "on-path"}
    assert all(a["--generate-under"] == "stale" for a in rung_calls if a["--text"] == "on-path")
    assert all(a.get("--prefix-len") == "6" for a in calls), "every cell scores the registered prefix length"
    # scrambled is corpus text only
    assert all(a["--text"] == "corpus" for a in calls if kind_of(a) == "scrambled")
    # resumable: a second run re-scores nothing
    n = len(fake.calls)
    reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert not scorer_calls_since(fake, n), "a finished run was re-scored"


def scorer_calls_since(fake: FakeInstrument, n: int) -> list:
    return [c for c in fake.calls[n:] if len(c) > 1 and str(c[1]).endswith("score_continuation.py")]


def test_run_halts_on_an_identity_failure_before_any_rung(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg)

    def broken(fm, sm, fs, ss, live):
        if not live and fm["dir"] == sm["dir"] and fs == ss:
            return 0.0, 1.0          # a logit difference far over identity_max_logit_diff = 1e-5
        return default_kl(fm, sm, fs, ss, live)

    fake = FakeInstrument(kl=broken)
    with pytest.raises(reader.Refusal, match="identity"):
        reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert not [a for a in scorer_calls(fake) if kind_of(a) == "rung"], "a rung was scored after the halt"
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert rec["halted"]


def test_run_halts_on_a_floor_over_its_limits(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg)

    def hot_floor(fm, sm, fs, ss, live):
        if live:
            return 0.5, 0.5          # the float16 floor reads far over floor_max_median_kl = 1e-4
        return default_kl(fm, sm, fs, ss, live)

    fake = FakeInstrument(kl=hot_floor)
    with pytest.raises(reader.Refusal, match="floor"):
        reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert not [a for a in scorer_calls(fake) if kind_of(a) == "rung"]


def test_scrambled_failure_marks_the_reader_unevaluable_without_halting(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg)

    def blind_qwen(fm, sm, fs, ss, live):
        if not live and fs != ss and fm["pair"] == "qwen2.5-1.5b-to-instruct":
            return 0.0, 0.0          # this reader cannot see the cache: median scrambled KL < scrambled_min
        return default_kl(fm, sm, fs, ss, live)

    fake = FakeInstrument(kl=blind_qwen)
    rec = reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["finished"] and rec["halted"] is None, "a scrambled failure is UNEVALUABLE, not a halt"
    assert "UNEVALUABLE" in json.dumps(rec)
    # the other readers' rungs still ran
    labels = {Path(a["--fresh"]).parent.name for a in scorer_calls(fake) if kind_of(a) == "rung"}
    assert "source@local-noise-0.001" in labels


# --- the probe ------------------------------------------------------------------------------------------------

def test_probe_runs_unregistered_and_only_live_cells(repo, seal_cfg, tmp_path):
    cfg, tools = _setup_reader(repo, seal_cfg, registered="", floor=False, seal=False)
    fake = FakeInstrument()
    out = tmp_path / "probe-out"
    reader.probe(cfg, repo, out=out, runner=fake, tools_dir=tools)
    calls = scorer_calls(fake)
    assert calls, "the probe scored nothing"
    assert all("--live" in a for a in calls), "the probe is the float16-floor measurement: --live only"
    assert out.exists() and any(out.rglob("report.json"))


def test_probe_refuses_an_out_under_results(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg, registered="", floor=False, seal=False)
    with pytest.raises(reader.Refusal, match="results"):
        reader.probe(cfg, repo, out=repo / "results" / "reader" / "probe", runner=FakeInstrument(), tools_dir=tools)
    assert not (repo / "results" / "reader" / "probe").exists()
