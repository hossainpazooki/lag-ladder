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
LEDGER_0014 = "\n### 0014 — 2026-10-13 — Reader registered\n\nregisters config/reader.toml.\n"

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


def _setup_reader(repo, seal_cfg, *, registered="0014", floor=True, seal=True):
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
    lpath.write_text(lpath.read_text(encoding="utf-8") + LEDGER_0014, encoding="utf-8")
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


# --- design S14: per-pair heldout, spans (per-sequence prefix_len/length) and the stored-continuation text ----

ROLLOUT_PAIR = '''
[[reader.pairs]]
name = "ownrun-rollout"
tokens = "data/tokens/qwen.npy"
tokens_sha256 = "{qsha}"
spans = "data/tokens/qwen.spans.json"
spans_sha256 = "{ssha}"
heldout = [0, 4]
'''
ROLLOUT_RUNG = '''
[[reader.rungs]]
label = "rollout-1e-3"
pair = "ownrun-rollout"
reader = "local:noise-0.001"
writer_dump = "source@step_200"
reader_dump = "source@local-noise-0.001"
texts = ["rollout"]
'''
SPANS = {"prefix_len": [2, 3, 2, 3], "length": [8, 8, 7, 6]}


def _setup_rollout(repo, seal_cfg):
    """The reader fixture plus a rollout pair (qwen tokens, spans sidecar, its own held-out range) and one rollout
    rung on an existing reader; its dumps are copies of the noise rung's dumps under the rollout pair's layout."""
    import shutil, hashlib
    cfg, tools = _setup_reader(repo, seal_cfg)
    up = repo.parent / "upstream"
    spans_path = up / "data" / "tokens" / "qwen.spans.json"
    spans_path.write_text(json.dumps(SPANS), encoding="utf-8")
    ssha = hashlib.sha256(spans_path.read_bytes()).hexdigest()
    shas = _token_shas(up)
    cpath = repo / "config" / "reader.toml"
    text = cpath.read_text(encoding="utf-8")
    text = text.replace("[[reader.rungs]]", ROLLOUT_PAIR.format(qsha=shas["qwen"], ssha=ssha) + "[[reader.rungs]]", 1)
    cpath.write_text(text + ROLLOUT_RUNG, encoding="utf-8")
    src = repo / "results" / "distance" / "pairs" / "olmo2-1b-rlvr1" / "dumps"
    dst = repo / "results" / "distance" / "pairs" / "ownrun-rollout" / "dumps"
    for key in ("source@step_200", "source@local-noise-0.001"):
        for i in range(4):
            shutil.copytree(src / key / f"seq{max(i, 1)}", dst / key / f"seq{i}")
    commit_all(repo, "add rollout pair")
    return load_reader_config(cpath, repo), tools


def test_rollout_rung_scores_stored_continuations_with_per_sequence_spans(repo, seal_cfg):
    cfg, tools = _setup_rollout(repo, seal_cfg)
    assert cfg.pairs["ownrun-rollout"].heldout == (0, 4) and cfg.pairs["ownrun-rollout"].spans is not None
    p = reader.plan(cfg)
    roll = [j for j in p.jobs if j.label == "rollout-1e-3"]
    assert [j.seq for j in roll] == [0, 1, 2, 3] and {j.text for j in roll} == {"rollout"}
    # controls for that reader stay on its first rung's pair and the config's held-out range
    ctl = [j for j in p.jobs if j.reader == "local:noise-0.001" and j.kind != "rung"]
    assert {j.pair for j in ctl} == {"olmo2-1b-rlvr1"} and {j.seq for j in ctl} == {1, 2, 3}
    fake = FakeInstrument()
    reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    calls = [_parse(c[2:]) for c in fake.calls if len(c) > 1 and c[1].endswith("score_continuation.py")]
    rc = [a for a in calls if a.get("--text") == "rollout"]
    assert len(rc) == 4
    by_seq = {int(a["--seq"]): a for a in rc}
    assert [int(by_seq[i]["--prefix-len"]) for i in range(4)] == SPANS["prefix_len"]
    assert [int(by_seq[i]["--length"]) for i in range(4)] == SPANS["length"]
    assert all("--generate-under" not in a for a in rc)
    corpus = [a for a in calls if a.get("--text") == "corpus"]
    assert corpus and all("--length" not in a and a["--prefix-len"] == "6" for a in corpus)
    rep = json.loads((cfg.results_dir / "rungs" / "rollout-1e-3" / "rollout" / "seq3" / "report.json").read_text())
    row = np.load(repo.parent / "upstream" / "data" / "tokens" / "qwen.npy")[3]
    assert rep["continuation_tokens"] == [int(t) for t in row[3:6]]
    rec = json.loads((cfg.results_dir / "run.json").read_text(encoding="utf-8"))
    assert rec["spans_sha256"] == {"ownrun-rollout": cfg.pairs["ownrun-rollout"].spans_sha256}


def test_rollout_text_needs_spans_and_is_not_a_rule_text(repo, seal_cfg):
    cfg, _ = _setup_reader(repo, seal_cfg)
    cpath = repo / "config" / "reader.toml"
    base = cpath.read_text(encoding="utf-8")
    cpath.write_text(base + ROLLOUT_RUNG.replace('pair = "ownrun-rollout"', 'pair = "qwen2.5-1.5b-to-instruct"'), encoding="utf-8")
    with pytest.raises(ValueError, match="spans"):
        load_reader_config(cpath, repo)
    cpath.write_text(base.replace('texts = ["corpus", "on-path"]', 'texts = ["corpus", "rollout"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="per-rung"):
        load_reader_config(cpath, repo)


def test_the_gate_requires_the_seal_the_config_names(repo, seal_cfg):
    cfg, tools = _setup_reader(repo, seal_cfg)
    cpath = repo / "config" / "reader.toml"
    cpath.write_text(cpath.read_text(encoding="utf-8").replace('registered_by = "0014"', 'registered_by = "0014"\nseal = "ownrun"'),
                     encoding="utf-8")
    commit_all(repo, "name another seal")
    cfg = load_reader_config(cpath, repo)
    assert cfg.seal == "ownrun"
    with pytest.raises(reader.Refusal, match="ownrun"):
        reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)


def test_a_readers_first_rung_may_not_be_a_rollout_rung(repo, seal_cfg):
    cfg, _ = _setup_rollout(repo, seal_cfg)
    cpath = repo / "config" / "reader.toml"
    text = cpath.read_text(encoding="utf-8")
    # move the rollout rung ahead of every other rung of its reader
    head, roll = text.split("\n[[reader.rungs]]\nlabel = \"rollout-1e-3\"", 1)
    cpath.write_text(head.replace("[[reader.rungs]]", "[[reader.rungs]]\nlabel = \"rollout-1e-3\"" + roll + "\n[[reader.rungs]]", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="first rung"):
        load_reader_config(cpath, repo)


# --- 0020: controls per (reader, pair), on the pair's own text --------------------------------------------------

def _setup_per_pair(repo, seal_cfg, *, rollout_first: bool = False):
    """The rollout fixture with `per_pair = true`; optionally the rollout rung moved ahead of the reader's others."""
    cfg, tools = _setup_rollout(repo, seal_cfg)
    cpath = repo / "config" / "reader.toml"
    text = cpath.read_text(encoding="utf-8").replace("[reader.controls]\n", "[reader.controls]\nper_pair = true\n", 1)
    if rollout_first:
        head, roll = text.split("\n[[reader.rungs]]\nlabel = \"rollout-1e-3\"", 1)
        text = head.replace("[[reader.rungs]]", "[[reader.rungs]]\nlabel = \"rollout-1e-3\"" + roll + "\n[[reader.rungs]]", 1)
    cpath.write_text(text, encoding="utf-8")
    commit_all(repo, "per-pair controls")
    return load_reader_config(cpath, repo), tools


def test_per_pair_controls_run_on_each_pair_of_the_reader_on_that_pairs_own_text(repo, seal_cfg):
    cfg, tools = _setup_per_pair(repo, seal_cfg)
    assert cfg.controls["per_pair"] is True
    p = reader.plan(cfg)
    ctl = [j for j in p.jobs if j.reader == "local:noise-0.001" and j.kind != "rung"]
    held = [j for j in ctl if j.pair == "olmo2-1b-rlvr1"]
    roll = [j for j in ctl if j.pair == "ownrun-rollout"]
    assert {j.seq for j in held} == {1, 2, 3}
    assert {(j.kind, j.text) for j in held} == {("floor", "corpus"), ("floor", "on-path"), ("identity", "corpus"),
                                                ("scrambled", "corpus")}
    assert {j.seq for j in roll} == {0, 1, 2, 3}
    assert {(j.kind, j.text) for j in roll} == {("floor", "rollout"), ("identity", "rollout"), ("scrambled", "rollout")}
    # a pair's controls precede that pair's rungs, and land under their own directory
    order = {id(j): i for i, j in enumerate(p.jobs)}
    rung = [j for j in p.jobs if j.label == "rollout-1e-3"]
    assert max(order[id(j)] for j in roll) < min(order[id(j)] for j in rung)
    assert len({reader.job_dir(cfg, j) for j in ctl}) == len(ctl)
    assert "ownrun-rollout" in reader.job_dir(cfg, roll[0]).as_posix()
    rec = reader.run(cfg, repo, runner=FakeInstrument(), seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["halted"] is None
    scr = [j for j in rec["jobs"] if j["kind"] == "scrambled" and j["pair"] == "ownrun-rollout"]
    assert len(scr) == 4 and {j["text"] for j in scr} == {"rollout"}


def test_per_pair_lets_a_readers_first_rung_be_a_rollout_rung(repo, seal_cfg):
    cfg, _ = _setup_per_pair(repo, seal_cfg, rollout_first=True)
    assert cfg.rungs[0].label == "rollout-1e-3"


def test_per_pair_scrambled_failure_withholds_only_that_pairs_rungs(repo, seal_cfg):
    cfg, tools = _setup_per_pair(repo, seal_cfg)
    # the rollout pair's dumps are copies of the olmo pair's; name the pair in their metadata so the fake can see it
    for meta in (repo / "results" / "distance" / "pairs" / "ownrun-rollout" / "dumps").rglob("meta.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["pair"] = "ownrun-rollout"
        meta.write_text(json.dumps(m), encoding="utf-8")

    def blind_on_rollout(fm, sm, fs, ss, live):
        if not live and fs != ss and fm.get("pair") == "ownrun-rollout":
            return 0.0, 0.0
        return default_kl(fm, sm, fs, ss, live)

    fake = FakeInstrument(kl=blind_on_rollout)
    rec = reader.run(cfg, repo, runner=fake, seal_cfg=seal_cfg, tools_dir=tools)
    assert rec["finished"] and rec["halted"] is None
    ran = {(j["label"], j["pair"]) for j in rec["jobs"] if j["kind"] == "rung"}
    assert ("rollout-1e-3", "ownrun-rollout") not in ran
    assert any(pair == "olmo2-1b-rlvr1" and reader_of == "local:noise-0.001"
               for (reader_of, pair) in {(j["reader"], j["pair"]) for j in rec["jobs"] if j["kind"] == "rung"})
    assert "UNEVALUABLE" in json.dumps(rec["unevaluable_readers"])
