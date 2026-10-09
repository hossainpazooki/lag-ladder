"""Relative L2 distance between two checkpoints as dumped (design §12, R-D1): over the whole parameter vector and
over the attention K projections, float64 accumulation, streamed tensor by tensor. Run with the instrument's
interpreter; imports nothing from the instrument.

    python tools/weight_distance.py --writer DIR_OR_SPEC --reader DIR_OR_SPEC --out distance.json

A spec is a local checkpoint directory, or `model_id@revision` (revision may be empty) resolved through the
Hub cache. The record carries the sha256 of every weight file read on each side, so a summarizer can tie the
distance to the exact bytes the dumps' checkpoint manifests name. Tensors present on one side only are
listed and excluded; a shape mismatch refuses.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(spec: str) -> Path:
    p = Path(spec)
    if p.is_dir():
        return p.resolve()
    if "@" not in spec:
        raise SystemExit(f"{spec}: not a directory and not model_id@revision")
    model_id, rev = spec.split("@", 1)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, revision=(rev or None), allow_patterns=["*.safetensors", "*.json"]))


def tensors(d: Path):
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors under {d}")
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                yield k, sf.get_tensor(k)
    return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    wd, rd = resolve(a.writer), resolve(a.reader)
    writer = {k: v for k, v in tensors(wd) if v.is_floating_point()}
    w_sq = sum(float(v.double().pow(2).sum()) for v in writer.values())
    k_keys = [k for k in writer if "k_proj" in k]
    wk_sq = sum(float(writer[k].double().pow(2).sum()) for k in k_keys)
    d_sq, dk_sq, only_reader, worst = 0.0, 0.0, [], ("", 0.0)
    seen = set()
    for k, v in tensors(rd):
        if not v.is_floating_point():
            continue
        if k not in writer:
            only_reader.append(k)
            continue
        if v.shape != writer[k].shape:
            raise SystemExit(f"{k}: shape {tuple(v.shape)} on the reader vs {tuple(writer[k].shape)} on the writer")
        seen.add(k)
        s = float((v.double() - writer[k].double()).pow(2).sum())
        d_sq += s
        if k in k_keys:
            dk_sq += s
        rel = (s ** 0.5) / (float(writer[k].double().pow(2).sum()) ** 0.5 or 1.0)
        if rel > worst[1]:
            worst = (k, rel)
    only_writer = sorted(set(writer) - seen)
    rec = {"tool": "lag-ladder tools/weight_distance.py",
           "writer": {"spec": a.writer, "dir": str(wd), "files": {f.name: sha256_file(f) for f in sorted(wd.glob("*.safetensors"))}},
           "reader": {"spec": a.reader, "dir": str(rd), "files": {f.name: sha256_file(f) for f in sorted(rd.glob("*.safetensors"))}},
           "n_tensors_compared": len(seen), "only_writer": only_writer, "only_reader": only_reader,
           "writer_norm": w_sq ** 0.5, "rel_delta_all": (d_sq ** 0.5) / (w_sq ** 0.5),
           "rel_delta_k_proj": (dk_sq ** 0.5) / (wk_sq ** 0.5) if wk_sq else None, "n_k_proj": len(k_keys),
           "worst_tensor": worst[0], "worst_tensor_rel": worst[1]}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"wrote {out}: rel delta all {rec['rel_delta_all']:.4e}, k_proj {rec['rel_delta_k_proj']}, "
          f"{len(seen)} tensors, only-writer {len(only_writer)}, only-reader {len(only_reader)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
