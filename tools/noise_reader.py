"""The noise readers of ledger 0042 (`config/ownrun_noise.toml`), made, measured and dumped on the run machine.

A noise reader `local:noise-ownrun-lag<g>-s<s>` is step 0 plus seeded isotropic Gaussian noise scaled so that its
whole-vector relative L2 distance to step 0 equals step g's: the target is `rel_delta_all` of
results/distance/ownrun/step_<g>.json (written in 0017 by `tools/weight_distance.py` from step 0 to step g). Per
reader, `make`:
  1. writes the perturbed checkpoint with `tools/perturb_checkpoint.py` under results/distance/checkpoints/<name>/;
  2. measures it with `tools/weight_distance.py` from step 0 (written beside it as weight_distance.json) and refuses
     unless the measured distance equals the realized norm the perturbation recorded (the same quantity:
     `same_rel_tol`) and the realized norm equals the target (`target_rel_tol`);
  3. dumps it on every pair its rungs use, at the pair's sequences, with the command and layout 0017's save hook used
     (`own_run.dump_command`, `results/distance/pairs/<pair>/dumps/source@local-<name>/seq<i>`);
  4. writes noise.json beside the checkpoint: the target and its file's sha256, the three norms, the checks.
Noise bytes do not travel between machines (LL 0012), so a reader is made where it is read; `realized_rel_norm`
ties it to its lag. `prune` deletes the weight files of every noise reader once the configuration's run has finished,
keeping provenance.json, weight_distance.json and noise.json. Run with the lag-ladder interpreter; the two tools run
with --tool-python (torch), the dumps with the instrument's interpreter.

    python tools/noise_reader.py plan  [--config config/ownrun_noise.toml]
    python tools/noise_reader.py make  [--config ...] --kvt-python PY --tool-python PY [--name NAME]
    python tools/noise_reader.py prune [--config ...]
"""
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lag_ladder import REPO_ROOT  # noqa: E402
from lag_ladder.config import load_reader_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("own_run", Path(__file__).resolve().parent / "own_run.py")
own_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(own_run)

NAME_RE = re.compile(r"noise-ownrun-lag(\d+)-s(\d+)")
SAME_REL_TOL = 1e-9      # weight_distance vs the perturbation's own realized norm: one quantity, float64 sums
TARGET_REL_TOL = 1e-4    # realized vs target: the float32 rounding of theta + s z (about 1e-6 at this size)
MIN_FREE_BYTES = 10 << 30


class Refusal(RuntimeError):
    pass


def _refuse(msg: str) -> Refusal:
    return Refusal(f"noise_reader REFUSED: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def noise_readers(cfg) -> list[dict]:
    """The config's noise readers in order of first use: name, lag, seed and the pairs their rungs use (with the
    sequences and the token file). Refuses a noise rung whose reader dump or writer dump is not the expected key."""
    out: dict[str, dict] = {}
    for r in cfg.rungs:
        if not r.reader.startswith("local:noise-"):
            continue
        name = r.reader[len("local:"):]
        m = NAME_RE.fullmatch(name)
        if not m:
            raise _refuse(f"rung {r.label}: noise reader {name!r} is not noise-ownrun-lag<g>-s<s>")
        if r.reader_dump != f"source@local-{name}":
            raise _refuse(f"rung {r.label}: reader_dump {r.reader_dump!r} is not source@local-{name}")
        if r.writer_dump != "source@local-step_0":
            raise _refuse(f"rung {r.label}: writer_dump {r.writer_dump!r} is not step 0's")
        e = out.setdefault(name, {"name": name, "lag": int(m.group(1)), "seed": int(m.group(2)), "pairs": []})
        if r.pair not in e["pairs"]:
            e["pairs"].append(r.pair)
    for e in out.values():
        # the far control reads the reader's own dump of the far pair: dump that pair too
        for p in list(e["pairs"]):
            far = cfg.pairs[p].far_pair
            if far is not None and far not in e["pairs"]:
                e["pairs"].append(far)
    if not out:
        raise _refuse(f"{cfg.config_path.name} names no noise reader")
    return list(out.values())


def target_of(ll: Path, lag: int) -> tuple[float, Path, str]:
    """Step g's distance to step 0 as 0017's save hook measured it, with the file and its sha256."""
    f = ll / "results" / "distance" / "ownrun" / f"step_{lag}.json"
    if not f.exists():
        raise _refuse(f"{f} is missing; the target distance of lag {lag} is read from it")
    rec = json.loads(f.read_text(encoding="utf-8"))
    w, r = Path(str(rec["writer"]["dir"])).name, Path(str(rec["reader"]["dir"])).name
    if w != "step_0" or r != f"step_{lag}":
        raise _refuse(f"{f} measures {w} -> {r}, not step_0 -> step_{lag}")
    v = float(rec["rel_delta_all"])
    if not (0 < v < 1):
        raise _refuse(f"{f}: rel_delta_all {v!r} is not in (0, 1)")
    return v, f, sha256_file(f)


def check_norms(target: float, realized: float, measured: float, *, same_rel_tol: float = SAME_REL_TOL,
                target_rel_tol: float = TARGET_REL_TOL) -> dict:
    """The norm the perturbation scaled to is the distance the summary places the reader
    at. Refuses on either gap; returns both relative gaps."""
    same = abs(measured - realized) / realized
    to_target = abs(realized - target) / target
    if same > same_rel_tol:
        raise _refuse(f"weight_distance reads {measured!r}, the perturbation realized {realized!r}: relative gap "
                      f"{same:.3e} > {same_rel_tol:g}; the two do not measure one quantity")
    if to_target > target_rel_tol:
        raise _refuse(f"realized norm {realized!r} misses the target {target!r} by {to_target:.3e} relative > "
                      f"{target_rel_tol:g}")
    return {"measured_vs_realized_rel": same, "realized_vs_target_rel": to_target,
            "same_rel_tol": same_rel_tol, "target_rel_tol": target_rel_tol}


def _run(runner, cmd: list[str], cwd: Path | None = None) -> None:
    r = runner(cmd, cwd=str(cwd) if cwd else None, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace") if isinstance(r.stderr, bytes) else str(r.stderr)
        raise _refuse(f"{' '.join(str(c) for c in cmd[:3])} ... rc={r.returncode}\n{err[-3000:]}")


def make_one(cfg, e: dict, *, ll: Path, kvt_python: Path, tool_python: Path, runner=subprocess.run,
             disk_usage=shutil.disk_usage) -> dict:
    ckpt_root = ll / "results" / "distance" / "checkpoints"
    step0, out = ckpt_root / "step_0", ckpt_root / e["name"]
    record = out / "noise.json"
    if record.exists():
        rec = json.loads(record.read_text(encoding="utf-8"))
        if rec.get("complete"):
            return rec
    if not (step0 / "provenance.json").exists():
        raise _refuse(f"{step0} has no provenance.json; step 0's kept checkpoint must be in place")
    target, tfile, tsha = target_of(ll, e["lag"])
    if not out.exists():
        free = disk_usage(str(ckpt_root)).free
        if free < MIN_FREE_BYTES:
            raise _refuse(f"{free >> 30} GiB free under {ckpt_root}; a float32 copy needs about 6")
        _run(runner, [str(tool_python), str(ll / "tools" / "perturb_checkpoint.py"), "--local-path", str(step0),
                      "--rel-norm", repr(target), "--seed", str(e["seed"]), "--out", str(out)])
    prov_f, wd_f = out / "provenance.json", out / "weight_distance.json"
    if not prov_f.exists():
        raise _refuse(f"{out} exists without provenance.json (an interrupted perturbation?); remove it and make again")
    prov = json.loads(prov_f.read_text(encoding="utf-8"))
    if int(prov["seed"]) != e["seed"] or float(prov["target_rel_norm"]) != target:
        raise _refuse(f"{prov_f} was written for seed {prov['seed']} at {prov['target_rel_norm']!r}, "
                      f"not seed {e['seed']} at {target!r}")
    if not wd_f.exists():
        _run(runner, own_run.distance_command(tool_python, ll, step0, out, wd_f))
    wd = json.loads(wd_f.read_text(encoding="utf-8"))
    if wd["only_writer"] or wd["only_reader"] or int(wd["n_tensors_compared"]) != int(prov["n_float_tensors"]):
        raise _refuse(f"{wd_f}: compared {wd['n_tensors_compared']} tensors of {prov['n_float_tensors']} "
                      f"(only writer {wd['only_writer']}, only reader {wd['only_reader']})")
    checks = check_norms(target, float(prov["realized_rel_norm"]), float(wd["rel_delta_all"]))
    dumps = {}
    for pair in e["pairs"]:
        pt = cfg.pairs[pair]
        seqs = range(*(pt.heldout or cfg.heldout))
        per_seq = own_run.write_per_seq_tokens(pt.tokens, ll / "results" / "distance" / "pairs" / pair / "tokens", seqs)
        for i, p in zip(seqs, per_seq):
            d = ll / "results" / "distance" / "pairs" / pair / "dumps" / f"source@local-{e['name']}" / f"seq{i}"
            if not (d / "meta.json").exists():
                _run(runner, own_run.dump_command(kvt_python, cfg.upstream_path, out, p, d), cwd=cfg.upstream_path)
            if not (d / "meta.json").exists():
                raise _refuse(f"the dump of {e['name']} on {pair} seq{i} wrote no meta.json")
        dumps[pair] = [seqs.start, seqs.stop]
    rec = {"tool": "lag-ladder tools/noise_reader.py", "name": e["name"], "lag": e["lag"], "seed": e["seed"],
           "target_rel_norm": target, "target_source": tfile.relative_to(ll).as_posix(), "target_source_sha256": tsha,
           "realized_rel_norm": float(prov["realized_rel_norm"]), "measured_rel_delta_all": float(wd["rel_delta_all"]),
           "checks": checks, "provenance_sha256": sha256_file(prov_f), "weight_distance_sha256": sha256_file(wd_f),
           "weights_sha256": prov.get("out_sha256"), "dumps": dumps, "complete": True}
    record.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def prune(cfg, ll: Path) -> list[str]:
    """Delete the weight files of every noise reader of a configuration whose run finished without a halt."""
    rec_f = cfg.results_dir / "run.json"
    if not rec_f.exists():
        raise _refuse(f"no {rec_f}; the noise readers are kept until their configuration has run")
    rec = json.loads(rec_f.read_text(encoding="utf-8"))
    if rec.get("halted") or not rec.get("finished"):
        raise _refuse(f"{rec_f} is not a finished run (halted {rec.get('halted')!r}); nothing deleted")
    gone = []
    for e in noise_readers(cfg):
        d = ll / "results" / "distance" / "checkpoints" / e["name"]
        if not (d / "noise.json").exists():
            raise _refuse(f"{d} has no noise.json; nothing deleted")
        for f in sorted(d.glob("*.safetensors")):
            f.unlink()
            gone.append(f.relative_to(ll).as_posix())
    return gone


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "make", "prune"])
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "ownrun_noise.toml"))
    ap.add_argument("--kvt-python", default=None, help="make: the instrument's interpreter (dumps)")
    ap.add_argument("--tool-python", default=None, help="make: an interpreter with torch (perturb, distance)")
    ap.add_argument("--name", default=None, help="make: one noise reader; default every one, in config order")
    a = ap.parse_args(argv)
    cfg = load_reader_config(Path(a.config), REPO_ROOT)
    try:
        es = noise_readers(cfg)
        if a.command == "plan":
            for e in es:
                print(f"{e['name']}  lag {e['lag']}  seed {e['seed']}  pairs {', '.join(e['pairs'])}")
            return 0
        if a.command == "prune":
            for f in prune(cfg, REPO_ROOT):
                print(f"deleted {f}")
            return 0
        if not a.kvt_python or not a.tool_python:
            raise _refuse("make needs --kvt-python and --tool-python")
        if a.name is not None:
            es = [e for e in es if e["name"] == a.name]
            if not es:
                raise _refuse(f"{a.name!r} is not a noise reader of {cfg.config_path.name}")
        for e in es:
            rec = make_one(cfg, e, ll=REPO_ROOT, kvt_python=Path(a.kvt_python), tool_python=Path(a.tool_python))
            c = rec["checks"]
            print(f"NOISE {rec['name']} target {rec['target_rel_norm']:.6e} realized {rec['realized_rel_norm']:.6e} "
                  f"measured {rec['measured_rel_delta_all']:.6e} (gaps {c['measured_vs_realized_rel']:.1e}, "
                  f"{c['realized_vs_target_rel']:.1e}) dumps {rec['dumps']}")
        return 0
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
