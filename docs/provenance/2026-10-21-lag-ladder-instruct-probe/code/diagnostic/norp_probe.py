"""Diagnostic wrapper: the package's baseline_probe.py, unedited, with decoding made the registered (C) loop's.

baseline_probe.py calls model.generate(do_sample=False, ...) and so inherits the pinned revision's
generation_config.json: repetition_penalty 1.1 (applied by transformers even without sampling) and
eos_token_id [151645, 151643]. The registered (C) loop (lag-ladder tools/task_swap.py _greedy) is plain argmax that
stops only on tok.eos_token_id. This wrapper patches the loaded model's generation_config to repetition_penalty 1.0
and eos_token_id = tok.eos_token_id, prints the effective setting, then runs baseline_probe.main() with the given args.
Nothing else changes (prompts, system line, chat template, answer rule, float32, cap, records format).

    python -I -B norp_probe.py <package dir> [baseline_probe.py args...]
"""
import json
import sys

sys.dont_write_bytecode = True          # never write __pycache__ into the package folder
PKG = sys.argv[1]
sys.path.insert(0, PKG)

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

_orig = AutoModelForCausalLM.from_pretrained


def _patched(model_id, *args, **kwargs):
    model = _orig(model_id, *args, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_id, revision=kwargs.get("revision"))
    gc = model.generation_config
    before = {"repetition_penalty": gc.repetition_penalty, "eos_token_id": gc.eos_token_id}
    gc.repetition_penalty = 1.0
    gc.eos_token_id = tok.eos_token_id
    print("DIAG_GENCFG " + json.dumps({"before": before, "after": {"repetition_penalty": gc.repetition_penalty,
                                                                   "eos_token_id": gc.eos_token_id}}), flush=True)
    return model


AutoModelForCausalLM.from_pretrained = _patched

import baseline_probe   # noqa: E402

raise SystemExit(baseline_probe.main(sys.argv[2:]))
