"""A stand-in for the pinned instrument's commands and for the repo's checkpoint tools, for tests only.

It is called the way the drivers call `subprocess.run`: `[python, "-m", "scripts.dump_kv", ...]`,
`[python, "-m", "scripts.score_positions", ...]`, `[python, <tools>/perturb_checkpoint.py, ...]` and
`[python, <tools>/weight_distance.py, ...]`. It writes the files the real commands write and nothing heavier:

  * a dump directory gets `meta.json` (with the revision / local path it was asked for and a `checkpoint`
    block shaped like the instrument's) and `checkpoint_manifest.json` naming a `model.safetensors` sha256:
    a local checkpoint's real file hash, or a deterministic stand-in for a Hub ref;
  * a score call reads the pairs file for n, builds per-token squares [n, L, H] from the two dump
    directories by a rule the test chooses (`deviation(src_meta, tgt_meta, src_seq, tgt_seq) -> float`, the
    mean centered deviation), writes `report.json` with the per-layer SSE/SST/R^2 the real scorer records
    and the per-token file it names by sha256;
  * `perturb_checkpoint.py` writes a checkpoint directory (a small `model.safetensors`, `config.json`,
    `provenance.json` with the realized norm equal to the target);
  * `weight_distance.py` writes `distance.json` with the files it "read" (hashes consistent with the dumps'
    manifests) and a relative distance from the same rule the scorer uses (`distance_of`).

Default rules: two different sequences of the same checkpoint deviate by 2.0; the same directory deviates by 0;
otherwise the deviation is 20 x the distance, where the distance is the noise rung's target norm, 0.02 for any
other local checkpoint pair, lag / 20000 for two step_N revisions of one model (so the deviation is lag / 1000,
as the pilot tests expect), 0.0005 for two other revisions of one model, and 0.05 between two Hub sides of a pair.

For the (B) build it also stands in for `tools/score_continuation.py`, invoked the same subprocess way
(`[python, <tools>/score_continuation.py, --fresh ... --stale ...]`). It writes a `report.json` with exactly
the reader-report-v1 keys, whose `kl_per_token` is a KNOWN function of the stand-in distance
(`kl(fresh_meta, stale_meta, fresh_seq, stale_seq, live) -> (mean_kl, max_abs_logit_diff)`, default
`default_kl`): 1e-6 in --live (the float16 floor), 0 on identity, 2.0 on a scrambled (cross-sequence) cell,
and (20 x distance)^2 otherwise, so bands, slope-fit inputs and matched-rung lookup are all exercised.
`continuation_sha(tokens)` is the one place the stand-in's continuation hash recipe lives.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

L, H = 2, 2


def _hub_sha(model_id: str, revision) -> str:
    return hashlib.sha256(f"{model_id}@{revision or 'main'}".encode()).hexdigest()


def _meta_spec(meta: dict) -> dict:
    """What a dump 'is', for the rules: {local: name|None, model_id, revision, which, pair}."""
    ck = meta["checkpoint"]
    return {"local": ck.get("local_checkpoint"), "model_id": ck["model_id"], "revision": ck.get("revision"),
            "which": meta.get("which"), "pair": meta.get("pair"), "dir": meta["dir"]}


def default_distance(a: dict, b: dict) -> float:
    if a["local"] and a["local"].startswith("noise-"):
        return float(a["local"].split("-", 1)[1])
    if b["local"] and b["local"].startswith("noise-"):
        return float(b["local"].split("-", 1)[1])
    if a["local"] or b["local"]:
        return 0.02
    if a["model_id"] == b["model_id"]:
        ra, rb = a.get("revision") or "", b.get("revision") or ""
        if ra.startswith("step_") and rb.startswith("step_"):
            return abs(int(ra.split("_")[1]) - int(rb.split("_")[1])) / 20000.0   # deviation 20x: lag / 1000
        return 0.0005
    return 0.05


def default_deviation(src_meta: dict, tgt_meta: dict, src_seq: int, tgt_seq: int) -> float:
    if src_meta["dir"] == tgt_meta["dir"]:
        return 0.0
    if src_seq != tgt_seq:
        return 2.0
    return min(1.5, 20.0 * default_distance(_meta_spec(src_meta), _meta_spec(tgt_meta)))


FLOOR_KL = 1e-6   # the stand-in's float16 floor: what a --live cell reads


def default_kl(fresh_meta: dict, stale_meta: dict, fresh_seq: int, stale_seq: int, live: bool) -> tuple[float, float]:
    """(mean per-token KL in nats, max abs logit diff) of a (B) cell: a known function of the stand-in distance,
    quadratic where the deviation rule is linear, so a log-log slope fit over the backbone reads 2."""
    if live:
        return FLOOR_KL, FLOOR_KL
    if fresh_meta["dir"] == stale_meta["dir"] and fresh_seq == stale_seq:
        return 0.0, 0.0
    if fresh_seq != stale_seq:
        return 2.0, 1.0
    d = default_distance(_meta_spec(fresh_meta), _meta_spec(stale_meta))
    return (20.0 * d) ** 2, 20.0 * d


def continuation_sha(tokens) -> str:
    """The scorer's continuation hash recipe, in one place: sha256 over the token ids as little-endian int64
    bytes (tools/score_continuation.py). The summarizer's recompute must agree with this."""
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


class FakeInstrument:
    def __init__(self, deviation=default_deviation, distance=default_distance, kl=default_kl):
        self.deviation = deviation
        self.distance = distance
        self.kl = kl
        self.calls = []
        self.envs = []

    def __call__(self, cmd, cwd=None, capture_output=False, **kw):
        self.calls.append(list(cmd))
        self.envs.append(kw.get("env"))
        if cmd[1] == "-m":
            module, rest = cmd[2], cmd[3:]
        else:
            module, rest = Path(cmd[1]).name, cmd[2:]
        args, i = {}, 0
        while i < len(rest):
            if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                args[rest[i]] = rest[i + 1]; i += 2
            else:
                args[rest[i]] = True; i += 1
        if module == "scripts.dump_kv":
            return self._dump(cmd, args)
        if module == "scripts.score_positions":
            return self._score(cmd, args)
        if module == "perturb_checkpoint.py":
            return self._perturb(cmd, args)
        if module == "weight_distance.py":
            return self._distance(cmd, args)
        if module == "score_continuation.py":
            return self._score_continuation(cmd, args)
        return subprocess.CompletedProcess(cmd, 2, b"", b"fake instrument: unknown command")

    def _dump(self, cmd, args):
        out = Path(args["--out"])
        out.mkdir(parents=True, exist_ok=True)
        seqs = np.load(args["--tokens"])
        pair, which = args["--pair"], args["--which"]
        model_id = f"fake/{pair}-{which}"
        if "--local-path" in args:
            lp = Path(args["--local-path"])
            weights = lp / "model.safetensors"
            if not weights.exists():
                return subprocess.CompletedProcess(cmd, 1, b"", f"no model.safetensors under {lp}".encode())
            sha, ck = hashlib.sha256(weights.read_bytes()).hexdigest(), {"kind": "local", "model_id": model_id, "revision": None,
                                                                         "resolved_commit": None, "local_checkpoint": lp.name}
        else:
            rev = args.get("--revision")
            sha, ck = _hub_sha(model_id, rev), {"kind": "hf", "model_id": model_id, "revision": rev,
                                                "resolved_commit": f"c{rev or 'main'}", "local_checkpoint": None}
        manifest = {"format": "kvt-checkpoint-manifest/1", "algorithm": "sha256",
                    "files": [{"path": "config.json", "sha256": "0" * 64, "size": 1}, {"path": "model.safetensors", "sha256": sha, "size": 1}]}
        manifest["digest"] = hashlib.sha256(json.dumps(manifest["files"]).encode()).hexdigest()
        ck.update({"manifest_file": "checkpoint_manifest.json", "manifest_digest": manifest["digest"]})
        (out / "checkpoint_manifest.json").write_text(json.dumps(manifest))
        (out / "meta.json").write_text(json.dumps({"n_seqs": int(seqs.shape[0]), "stride": int(args["--stride"]),
                                                   "which": which, "pair": pair, "revision": args.get("--revision"),
                                                   "local_path": args.get("--local-path"), "load_dtype": args.get("--dtype"),
                                                   "model": model_id, "checkpoint": ck, "dir": out.resolve().as_posix()}))
        (out / "meta.npz").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def _score(self, cmd, args):
        src, tgt = Path(args["--same-src"]), Path(args["--same-tgt"])
        sm, tm = json.loads((src / "meta.json").read_text()), json.loads((tgt / "meta.json").read_text())
        n = int(np.load(args["--pairs"])["pairs"].shape[0])
        dev = self.deviation(sm, tm, _seq(src), _seq(tgt))
        rng = np.random.default_rng(n + len(self.calls))
        sst = rng.uniform(50.0, 100.0, size=(L, H))
        w = rng.gamma(4.0, size=(n, L, H)) if dev > 0 else np.zeros((n, L, H))
        if dev > 0:
            w = w / w.mean(0, keepdims=True)
        sq32 = ((dev * (sst / n))[None] * w).astype(np.float32)
        sse = sq32.astype(np.float64).sum(0)
        r2 = 1.0 - sse / sst
        layers = [{"sse": [float(x) for x in sse[l]], "sst": [float(x) for x in sst[l]], "r2_head_mean": float(r2[l].mean())}
                  for l in range(L)]
        rec = {"n_pairs": n, "same": {"K": layers, "V": layers},
               "same_K_r2_layer_mean": float(np.mean([l["r2_head_mean"] for l in layers])),
               "same_V_r2_layer_mean": float(np.mean([l["r2_head_mean"] for l in layers])), "seconds": 0.0}
        pt = Path(args["--per-token"])
        pt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pt, same_K=sq32, same_V=sq32, ref_K=np.ones_like(sq32), ref_V=np.ones_like(sq32))
        rec["per_token"] = {"path": pt.name, "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(), "dtype": "float32",
                            "arrays": ["ref_K", "ref_V", "same_K", "same_V"], "layout": "[n_pairs, n_layers, n_kv]"}
        out = Path(args["--out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def _perturb(self, cmd, args):
        out = Path(args["--out"])
        if out.exists():
            return subprocess.CompletedProcess(cmd, 1, b"", b"refusing to overwrite")
        out.mkdir(parents=True)
        target = float(args["--rel-norm"])
        (out / "model.safetensors").write_bytes(f"perturbed {args['--model-id']}@{args['--revision']} {target:g} seed {args['--seed']}".encode())
        (out / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
        (out / "provenance.json").write_text(json.dumps({"tool": "fake perturb", "seed": int(args["--seed"]), "target_rel_norm": target,
                                                         "realized_rel_norm": target, "out_sha256": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()}))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def _score_continuation(self, cmd, args):
        out = Path(args["--out"])
        if "--probe" in args and "results" in out.resolve().parts:
            return subprocess.CompletedProcess(cmd, 2, b"", f"probe REFUSED: --out {out} is under results/".encode())
        live = "--live" in args
        if live == ("--fresh" in args):
            return subprocess.CompletedProcess(cmd, 2, b"", b"--live replaces --fresh; exactly one is required")
        stale_dir = Path(args["--stale"])
        fresh_dir = Path(args["--fresh"]) if "--fresh" in args else stale_dir
        fm = json.loads((fresh_dir / "meta.json").read_text(encoding="utf-8"))
        sm = json.loads((stale_dir / "meta.json").read_text(encoding="utf-8"))
        text, prefix, seq = args["--text"], int(args["--prefix-len"]), int(args["--seq"])
        tokens_file = Path(args["--tokens"])
        row = np.load(tokens_file)[seq]
        n_cont = int(row.shape[0]) - prefix
        if n_cont <= 0:
            return subprocess.CompletedProcess(cmd, 2, b"", b"prefix-len leaves no continuation")
        mean_kl, mld = self.kl(fm, sm, _seq(fresh_dir), _seq(stale_dir), live)
        if text == "corpus":
            cont, gen_under, gen_seconds = [int(t) for t in row[prefix:]], None, None
        else:
            cont, gen_under, gen_seconds = [int((seq * 31 + i) % 97) for i in range(n_cont)], args["--generate-under"], 0.01
        agree = mean_kl <= 1e-3
        reader_meta = sm if live else fm
        inputs = {p.resolve().as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (tokens_file, fresh_dir / "meta.json", stale_dir / "meta.json")}
        rep = {"schema": "reader-report-v1", "reader": args.get("--reader", ""),
               "fresh_dump": fresh_dir.resolve().as_posix(), "stale_dump": stale_dir.resolve().as_posix(),
               "tokens_file": tokens_file.resolve().as_posix(), "seq": seq, "prefix_len": prefix,
               "text": text, "generate_under": gen_under, "live": live,
               "continuation_tokens": cont, "continuation_sha256": continuation_sha(cont),
               "logprob_fresh": [-1.0] * n_cont, "logprob_stale": [-1.0 - mean_kl] * n_cont,
               "kl_per_token": [mean_kl] * n_cont,
               "argmax_fresh": list(range(n_cont)), "argmax_stale": list(range(n_cont)) if agree else [i + 1 for i in range(n_cont)],
               "max_abs_logit_diff": mld, "inputs_sha256": inputs,
               "reader_checkpoint": {"manifest_digest": reader_meta["checkpoint"]["manifest_digest"]} if reader_meta.get("checkpoint") else None,
               "device": "cpu", "torch_version": "0.0-fake", "peak_cuda_mem_bytes": None, "gen_seconds": gen_seconds}
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def _distance(self, cmd, args):
        def side(spec):
            p = Path(spec)
            if p.is_dir():
                return {"local": p.name, "model_id": None, "revision": None}, {"model.safetensors": hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()}
            model_id, commit = spec.split("@", 1)
            rev = commit[1:] if commit.startswith("c") else commit      # the fake's resolved commit is "c" + revision
            rev = None if rev == "main" else rev
            return {"local": None, "model_id": model_id, "revision": rev}, {"model.safetensors": _hub_sha(model_id, rev)}
        wa, wf = side(args["--writer"]); ra, rf = side(args["--reader"])
        d = self.distance(wa, ra)
        out = Path(args["--out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"tool": "fake weight_distance", "writer": {"spec": args["--writer"], "files": wf},
                                   "reader": {"spec": args["--reader"], "files": rf}, "rel_delta_all": d,
                                   "rel_delta_k_proj": d * 6.0, "n_tensors_compared": 3}))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def _seq(dump_dir: Path) -> int:
    return int(dump_dir.name.replace("seq", ""))
