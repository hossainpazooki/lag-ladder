"""Convert a Hub checkpoint shipped as `pytorch_model.bin` into a local safetensors checkpoint the instrument can
hash and load (design §12, R-D3: OLMo-2 1B SFT and DPO). Run with the instrument's interpreter; imports nothing
from the instrument.

    python tools/convert_bin_checkpoint.py --model-id allenai/OLMo-2-0425-1B-SFT --out DIR

Weights are loaded with `torch.load(..., weights_only=True)` and written unchanged in their stored dtype;
`config.json` and the tokenizer files are copied. `provenance.json` records the Hub commit, the .bin's sha256,
every tensor's name/dtype/shape digest, and the output's sha256. Refuses to overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(a.model_id, revision=a.revision, allow_patterns=["*.bin", "*.json", "*.txt", "*.model"]))
    bins = sorted(src.glob("*.bin"))
    if not bins:
        raise SystemExit(f"no .bin weights under {src}")
    state, parts = {}, {}
    for b in bins:
        sd = torch.load(str(b), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k in state:
                raise SystemExit(f"tensor {k} appears in two shards")
            state[k] = v.contiguous()
        parts[b.name] = sha256_file(b)
        del sd
    out.mkdir(parents=True)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    for extra in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"):
        if (src / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())
    prov = {"tool": "lag-ladder tools/convert_bin_checkpoint.py", "source": {"model_id": a.model_id, "revision": a.revision,
            "resolved_dir": str(src), "commit": src.name, "bin_files": parts},
            "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in state.items()},
            "out_sha256": sha256_file(out / "model.safetensors")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(state)} tensors from {', '.join(parts)} at {src.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
