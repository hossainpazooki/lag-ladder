"""A stand-in for the pinned instrument's two commands, for tests only.

It is called the way `lag_ladder.pilot` calls `subprocess.run`: `[python, "-m", "scripts.dump_kv", ...]`
and `[python, "-m", "scripts.score_positions", ...]`, with `cwd` the upstream checkout. It writes the
files the real commands write and nothing else:

  * a dump directory gets `meta.json` (with the revision it was asked for) and nothing heavier;
  * a score call reads the pairs file for n, builds per-token squares [n, L, H] from the two dump
    directories by a rule the test chooses (`deviation(src_meta, tgt_meta, src_seq, tgt_seq) -> float`,
    the mean centered deviation), writes `report.json` with the per-layer SSE/SST/R^2 the real scorer
    records and the per-token file it names by sha256.

With the default rule the same directory on both sides deviates by exactly 0, two sequences of the same
revision deviate by 2.0 (every token far outside any tau), and two revisions of the same sequence deviate
by lag / 1000 -- so the ladder crosses tau_K = 0.3186 between lags 200 and 400.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

L, H = 2, 2


def default_deviation(src_meta: dict, tgt_meta: dict, src_seq: int, tgt_seq: int) -> float:
    if src_meta["dir"] == tgt_meta["dir"]:
        return 0.0
    if src_seq != tgt_seq:
        return 2.0
    lag = abs(int(tgt_meta["revision"].split("_")[1]) - int(src_meta["revision"].split("_")[1]))
    return lag / 1000.0


class FakeInstrument:
    def __init__(self, deviation=default_deviation):
        self.deviation = deviation
        self.calls = []

    def __call__(self, cmd, cwd=None, capture_output=False, **kw):
        self.calls.append(list(cmd))
        args = {cmd[i]: cmd[i + 1] for i in range(3, len(cmd) - 1, 2) if cmd[i].startswith("--")}
        flags = {c for c in cmd[3:] if c.startswith("--") and c not in args}
        if cmd[2] == "scripts.dump_kv":
            out = Path(args["--out"])
            out.mkdir(parents=True, exist_ok=True)
            seqs = np.load(args["--tokens"])
            (out / "meta.json").write_text(json.dumps({"n_seqs": int(seqs.shape[0]), "stride": int(args["--stride"]),
                                                       "revision": args.get("--revision"), "load_dtype": args.get("--dtype"),
                                                       "dir": out.resolve().as_posix()}))
            (out / "meta.npz").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if cmd[2] == "scripts.score_positions":
            src, tgt = Path(args["--same-src"]), Path(args["--same-tgt"])
            sm, tm = json.loads((src / "meta.json").read_text()), json.loads((tgt / "meta.json").read_text())
            n = int(np.load(args["--pairs"])["pairs"].shape[0])
            dev = self.deviation(sm, tm, _seq(src), _seq(tgt))
            rng = np.random.default_rng(n + len(self.calls))
            sst = rng.uniform(50.0, 100.0, size=(L, H))
            # squares whose centered mean is exactly `dev` per layer-head: sq = dev * (sst / n) * w, mean(w) = 1
            w = rng.gamma(4.0, size=(n, L, H)) if dev > 0 else np.zeros((n, L, H))
            if dev > 0:
                w = w / w.mean(0, keepdims=True)
            sq = (dev * (sst / n))[None] * w
            sq32 = sq.astype(np.float32)
            sse = sq32.astype(np.float64).sum(0)
            r2 = 1.0 - sse / sst
            layers = [{"sse": [float(x) for x in sse[l]], "sst": [float(x) for x in sst[l]],
                       "r2_head_mean": float(r2[l].mean())} for l in range(L)]
            rec = {"n_pairs": n, "same": {"K": layers, "V": layers},
                   "same_K_r2_layer_mean": float(np.mean([l["r2_head_mean"] for l in layers])),
                   "same_V_r2_layer_mean": float(np.mean([l["r2_head_mean"] for l in layers])), "seconds": 0.0}
            pt = Path(args["--per-token"])
            pt.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pt, same_K=sq32, same_V=sq32, ref_K=np.ones_like(sq32), ref_V=np.ones_like(sq32))
            rec["per_token"] = {"path": pt.name, "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
                                "dtype": "float32", "arrays": sorted(["same_K", "same_V", "ref_K", "ref_V"]),
                                "layout": "[n_pairs, n_layers, n_kv]; sse[l][h] == sum over tokens (float64)"}
            out = Path(args["--out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec))
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return subprocess.CompletedProcess(cmd, 2, b"", b"fake instrument: unknown command")


def _seq(dump_dir: Path) -> int:
    return int(dump_dir.name.replace("seq", ""))
