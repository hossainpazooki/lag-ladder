"""tools/noise_reader.py (ledger 0042, the noise readers): their target distances, the norm check, the commands one reader issues and the order of them, and the prune guard. The perturbation, the
distance and the dumps run on the run machine and are not exercised here (a fake runner writes their outputs)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lag_ladder import REPO_ROOT

spec = importlib.util.spec_from_file_location("noise_reader", REPO_ROOT / "tools" / "noise_reader.py")
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)

H, R = "qwen2.5-1.5b-to-instruct", "qwen2.5-1.5b-ownrun-rollout"


def _rung(label, pair, name, reader_dump=None, writer_dump="source@local-step_0"):
    return SimpleNamespace(label=label, pair=pair, reader=f"local:{name}",
                           reader_dump=reader_dump or f"source@local-{name}", writer_dump=writer_dump)


def _cfg(tmp_path, rungs, *, far=True):
    up = tmp_path / "upstream"
    (up / "data").mkdir(parents=True, exist_ok=True)
    np.save(up / "data" / "h.npy", np.arange(6 * 8, dtype=np.int64).reshape(6, 8))
    np.save(up / "data" / "r.npy", np.arange(3 * 8, dtype=np.int64).reshape(3, 8))
    pairs = {H: SimpleNamespace(tokens=up / "data" / "h.npy", heldout=None, far_pair=None),
             R: SimpleNamespace(tokens=up / "data" / "r.npy", heldout=(0, 2), far_pair=H if far else None)}
    return SimpleNamespace(rungs=rungs, pairs=pairs, heldout=(4, 6), upstream_path=up,
                           results_dir=tmp_path / "ll" / "results" / "ownrun_noise",
                           config_path=tmp_path / "ownrun_noise.toml")


def test_noise_readers_adds_the_far_pair_and_refuses_unexpected_dump_keys(tmp_path):
    name = "noise-ownrun-lag4-s1"
    es = nr.noise_readers(_cfg(tmp_path, [_rung("ownrun-rollout-noise-lag-4-s1", R, name)]))
    assert es == [{"name": name, "lag": 4, "seed": 1, "pairs": [R, H]}]
    with pytest.raises(nr.Refusal, match="reader_dump"):
        nr.noise_readers(_cfg(tmp_path, [_rung("x", H, name, reader_dump="source@local-step_4")]))
    with pytest.raises(nr.Refusal, match="writer_dump"):
        nr.noise_readers(_cfg(tmp_path, [_rung("x", H, name, writer_dump="source@local-step_1")]))
    with pytest.raises(nr.Refusal, match="is not noise-ownrun-lag"):
        nr.noise_readers(_cfg(tmp_path, [_rung("x", H, "noise-0.01")]))
    with pytest.raises(nr.Refusal, match="names no noise reader"):
        nr.noise_readers(_cfg(tmp_path, []))


def test_check_norms_passes_one_quantity_and_refuses_either_gap():
    c = nr.check_norms(2.2895683763734974e-05, 2.2895700e-05, 2.2895700e-05)
    assert c["measured_vs_realized_rel"] == 0.0 and c["realized_vs_target_rel"] < 1e-5
    with pytest.raises(nr.Refusal, match="do not measure one quantity"):
        nr.check_norms(1e-4, 1e-4, 1e-4 * (1 + 1e-8))
    with pytest.raises(nr.Refusal, match="misses the target"):
        nr.check_norms(1e-4, 1e-4 * (1 + 1e-3), 1e-4 * (1 + 1e-3))


def _distance(ll: Path, lag: int, writer="step_0", reader=None, v=5.267686242230837e-05):
    f = ll / "results" / "distance" / "ownrun" / f"step_{lag}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    ck = ll / "results" / "distance" / "checkpoints"
    f.write_text(json.dumps({"writer": {"dir": str(ck / writer)}, "reader": {"dir": str(ck / (reader or f"step_{lag}"))},
                             "rel_delta_all": v}), encoding="utf-8")
    return v


def test_target_is_steps_distance_to_step_0(tmp_path):
    ll = tmp_path / "ll"
    with pytest.raises(nr.Refusal, match="is missing"):
        nr.target_of(ll, 4)
    v = _distance(ll, 4)
    got, f, sha = nr.target_of(ll, 4)
    assert got == v and f.name == "step_4.json" and len(sha) == 64
    _distance(ll, 4, writer="step_1")
    with pytest.raises(nr.Refusal, match="not step_0 -> step_4"):
        nr.target_of(ll, 4)


class FakeRunner:
    """Writes what each tool would: the perturbation's provenance, the distance record, a dump's meta.json."""

    def __init__(self, measured_delta=0.0, n_tensors=338):
        self.calls, self.measured_delta, self.n = [], measured_delta, n_tensors

    def __call__(self, cmd, cwd=None, capture_output=True):
        self.calls.append((list(map(str, cmd)), cwd))
        c = list(map(str, cmd))
        if c[1].endswith("perturb_checkpoint.py"):
            out = Path(c[c.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "model.safetensors").write_bytes(b"w")
            t = float(c[c.index("--rel-norm") + 1])
            (out / "provenance.json").write_text(json.dumps({"seed": int(c[c.index("--seed") + 1]), "target_rel_norm": t,
                                                             "realized_rel_norm": t * (1 + 1e-7), "n_float_tensors": self.n,
                                                             "out_sha256": "ab" * 32}), encoding="utf-8")
        elif c[1].endswith("weight_distance.py"):
            out = Path(c[c.index("--out") + 1])
            prov = json.loads((Path(c[c.index("--reader") + 1]) / "provenance.json").read_text(encoding="utf-8"))
            out.write_text(json.dumps({"rel_delta_all": prov["realized_rel_norm"] * (1 + self.measured_delta),
                                       "n_tensors_compared": self.n, "only_writer": [], "only_reader": []}), encoding="utf-8")
        elif "scripts.dump_kv" in c:
            out = Path(c[c.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "meta.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr=b"")


def _tree(tmp_path, lag=4):
    ll = tmp_path / "ll"
    s0 = ll / "results" / "distance" / "checkpoints" / "step_0"
    s0.mkdir(parents=True)
    (s0 / "provenance.json").write_text("{}", encoding="utf-8")
    return ll, _distance(ll, lag)


BIG = lambda _p: SimpleNamespace(free=100 << 30)   # noqa: E731


def test_make_one_perturbs_measures_then_dumps_both_pairs_and_records_it(tmp_path):
    ll, target = _tree(tmp_path)
    name = "noise-ownrun-lag4-s1"
    cfg = _cfg(tmp_path, [_rung("a", H, name), _rung("b", R, name)])
    run = FakeRunner()
    rec = nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("kvt-py"), tool_python=Path("tool-py"),
                      runner=run, disk_usage=BIG)
    cmds = [c for c, _ in run.calls]
    assert cmds[0][:2] == ["tool-py", str(ll / "tools" / "perturb_checkpoint.py")]
    assert cmds[0][cmds[0].index("--rel-norm") + 1] == repr(target) and cmds[0][cmds[0].index("--seed") + 1] == "1"
    assert cmds[1][1].endswith("weight_distance.py")
    dumps = [c for c in cmds[2:]]
    assert all("scripts.dump_kv" in c for c in dumps) and len(dumps) == 2 + 2     # held-out seqs 4, 5; rollout 0, 1
    outs = [Path(c[c.index("--out") + 1]).relative_to(ll).as_posix() for c in dumps]
    assert outs == [f"results/distance/pairs/{H}/dumps/source@local-{name}/seq4",
                    f"results/distance/pairs/{H}/dumps/source@local-{name}/seq5",
                    f"results/distance/pairs/{R}/dumps/source@local-{name}/seq0",
                    f"results/distance/pairs/{R}/dumps/source@local-{name}/seq1"]
    assert all(c[c.index("--local-path") + 1] == str(ll / "results" / "distance" / "checkpoints" / name) for c in dumps)
    assert all(cwd == str(cfg.upstream_path) for _, cwd in run.calls[2:])
    assert rec["complete"] and rec["lag"] == 4 and rec["seed"] == 1 and rec["target_rel_norm"] == target
    assert rec["dumps"] == {H: [4, 6], R: [0, 2]}
    saved = json.loads((ll / "results" / "distance" / "checkpoints" / name / "noise.json").read_text(encoding="utf-8"))
    assert saved == rec
    again = FakeRunner()
    assert nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("k"), tool_python=Path("t"),
                       runner=again, disk_usage=BIG) == rec and again.calls == []


def test_make_one_refuses_a_measured_distance_off_the_realized_norm(tmp_path):
    ll, _ = _tree(tmp_path)
    cfg = _cfg(tmp_path, [_rung("a", H, "noise-ownrun-lag4-s0")])
    run = FakeRunner(measured_delta=1e-6)
    with pytest.raises(nr.Refusal, match="do not measure one quantity"):
        nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("k"), tool_python=Path("t"),
                    runner=run, disk_usage=BIG)
    assert not any("scripts.dump_kv" in c for c, _ in run.calls)          # nothing dumped from a misplaced reader
    assert not (ll / "results" / "distance" / "checkpoints" / "noise-ownrun-lag4-s0" / "noise.json").exists()


def test_make_one_refuses_on_low_disk_and_without_step_0(tmp_path):
    ll, _ = _tree(tmp_path)
    cfg = _cfg(tmp_path, [_rung("a", H, "noise-ownrun-lag4-s0")])
    run = FakeRunner()
    with pytest.raises(nr.Refusal, match="GiB free"):
        nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("k"), tool_python=Path("t"),
                    runner=run, disk_usage=lambda _p: SimpleNamespace(free=1 << 30))
    assert run.calls == []
    (ll / "results" / "distance" / "checkpoints" / "step_0" / "provenance.json").unlink()
    with pytest.raises(nr.Refusal, match="step 0's kept checkpoint"):
        nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("k"), tool_python=Path("t"),
                    runner=run, disk_usage=BIG)


def test_prune_needs_a_finished_run_and_keeps_the_records(tmp_path):
    ll, _ = _tree(tmp_path)
    name = "noise-ownrun-lag4-s0"
    cfg = _cfg(tmp_path, [_rung("a", H, name)])
    nr.make_one(cfg, nr.noise_readers(cfg)[0], ll=ll, kvt_python=Path("k"), tool_python=Path("t"),
                runner=FakeRunner(), disk_usage=BIG)
    with pytest.raises(nr.Refusal, match="are kept until"):
        nr.prune(cfg, ll)
    cfg.results_dir.mkdir(parents=True)
    (cfg.results_dir / "run.json").write_text(json.dumps({"halted": "x", "finished": None}), encoding="utf-8")
    with pytest.raises(nr.Refusal, match="not a finished run"):
        nr.prune(cfg, ll)
    (cfg.results_dir / "run.json").write_text(json.dumps({"halted": None, "finished": "t"}), encoding="utf-8")
    d = ll / "results" / "distance" / "checkpoints" / name
    assert nr.prune(cfg, ll) == [f"results/distance/checkpoints/{name}/model.safetensors"]
    assert sorted(p.name for p in d.iterdir()) == ["noise.json", "provenance.json", "weight_distance.json"]
