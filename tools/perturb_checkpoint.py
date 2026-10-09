"""Write a perturbed copy of a checkpoint: seeded isotropic Gaussian noise scaled to a target relative L2 norm
over the whole parameter vector (design §12, R-D2). Run with the instrument's interpreter (torch, safetensors,
huggingface_hub); imports nothing from the instrument.

    python tools/perturb_checkpoint.py --model-id ID --revision REV --rel-norm 1e-3 --seed 0 --out DIR
    python tools/perturb_checkpoint.py --local-path DIR_IN --rel-norm 1e-3 --seed 0 --out DIR

The output is a float32 `model.safetensors` (bfloat16 cannot carry a 1e-4 relative perturbation: its mantissa
step is ~4e-3), the source's `config.json` with `torch_dtype` set to float32, and `provenance.json` recording
the source files' sha256, the seed, the target and the realized relative norm (over all floating tensors,
float64 accumulation) and the per-tensor noise rule: one standard normal draw per element, scaled by a single
global factor so that ||noise|| / ||theta|| equals the target. Refuses to overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(args) -> Path:
    if args.local_path:
        return Path(args.local_path).resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(args.model_id, revision=args.revision,
                                  allow_patterns=["*.safetensors", "*.json"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--local-path", default=None)
    ap.add_argument("--rel-norm", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.model_id) == bool(a.local_path):
        raise SystemExit("give exactly one of --model-id (with --revision) or --local-path")
    if not (0 < a.rel_norm < 1):
        raise SystemExit("--rel-norm must be in (0, 1)")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    src = resolve(a)
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors under {src}")
    tensors, order = {}, []
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                tensors[k] = sf.get_tensor(k)
                order.append(k)
    float_keys = [k for k in order if tensors[k].is_floating_point()]
    theta_sq = sum(float(tensors[k].double().pow(2).sum()) for k in float_keys)
    gen = torch.Generator(device="cpu").manual_seed(int(a.seed))
    noise = {k: torch.randn(tensors[k].shape, generator=gen, dtype=torch.float32) for k in float_keys}
    noise_sq = sum(float(noise[k].double().pow(2).sum()) for k in float_keys)
    scale = a.rel_norm * (theta_sq ** 0.5) / (noise_sq ** 0.5)
    out_tensors, delta_sq = {}, 0.0
    for k in order:
        t = tensors[k]
        if k in noise:
            t32 = t.float() + noise[k] * scale
            delta_sq += float((t32.double() - t.double()).pow(2).sum())
            out_tensors[k] = t32.contiguous()
        else:
            out_tensors[k] = t.contiguous()
    realized = (delta_sq ** 0.5) / (theta_sq ** 0.5)
    out.mkdir(parents=True)
    save_file(out_tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["torch_dtype"] = "float32"
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt"):
        if (src / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())
    prov = {"tool": "lag-ladder tools/perturb_checkpoint.py", "source": {"model_id": a.model_id, "revision": a.revision,
            "local_path": a.local_path, "resolved_dir": str(src), "files": {f.name: sha256_file(f) for f in files}},
            "seed": int(a.seed), "target_rel_norm": a.rel_norm, "realized_rel_norm": realized,
            "n_float_tensors": len(float_keys), "theta_norm": theta_sq ** 0.5, "out_dtype": "float32",
            "rule": "theta' = theta + s * z, z ~ N(0, I) per element (torch.randn, CPU generator, one seed), "
                    "s chosen so ||s z|| / ||theta|| = target over all floating tensors",
            "out_sha256": sha256_file(out / "model.safetensors")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {out}: target {a.rel_norm:g}, realized {realized:.6g}, {len(float_keys)} float tensors, seed {a.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
