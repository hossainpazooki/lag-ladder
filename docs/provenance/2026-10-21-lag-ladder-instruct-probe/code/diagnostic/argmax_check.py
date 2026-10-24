"""Which decoding produced a records.jsonl? Teacher-force each completion through the pinned model (float32, one full
forward pass) and count, token by token, whether the recorded token is the argmax of the raw logits (the registered
(C) loop's rule) or of the logits after transformers' 1.1 repetition penalty over every token seen so far (prompt +
completion; logit / 1.1 if positive, * 1.1 if negative). Token ids are recovered by re-encoding completion_text and are
used only when they hash to the record's completion_sha256. Full-sequence logits differ from incremental-decoding logits
by rounding, so a few near-tie disagreements are expected under either rule.

    python -I -B argmax_check.py <package dir> <lag-ladder dir> <out.json> label=records.jsonl [label=records.jsonl ...]
"""
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np   # noqa: E402
import torch   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

pkg, ll, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ll + "/tools")
from own_run import SYSTEM_LINE   # noqa: E402

MODEL, REV, PEN = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 1.1
prompts = {p["index"]: p for p in json.load(open(pkg + "/gsm8k_test_prompts.json", encoding="utf-8"))}
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32).to("cuda").eval()
eos = tok.eos_token_id
result = {"model": f"{MODEL}@{REV}", "penalty_tested": PEN, "device": torch.cuda.get_device_name(0), "sets": {}}

for spec in sys.argv[4:]:
    label, path = spec.split("=", 1)
    s = {"records": 0, "ids_recovered": 0, "tokens": 0, "eq_raw_argmax": 0, "eq_penalized_argmax": 0,
         "records_all_raw": 0, "records_all_penalized": 0, "raw_mismatch_examples": []}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); s["records"] += 1
        q = prompts[r["index"]]["question"]
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        pid = tok(text, add_special_tokens=False)["input_ids"]
        cid = tok(r["completion_text"], add_special_tokens=False)["input_ids"] + ([] if r["reached_cap"] else [eos])
        if len(pid) != r["prompt_len"] or hashlib.sha256(np.asarray(cid, dtype=np.int64).tobytes()).hexdigest() != r["completion_sha256"]:
            continue
        s["ids_recovered"] += 1
        with torch.inference_mode():
            logits = model(torch.tensor([pid + cid], device="cuda")).logits[0].float()
            seen = torch.zeros(logits.shape[-1], dtype=torch.bool, device="cuda")
            seen[torch.tensor(pid, device="cuda")] = True
            L, n_raw, n_pen = len(pid), 0, 0
            for t, tid in enumerate(cid):
                lg = logits[L - 1 + t]
                pen = torch.where(seen, torch.where(lg > 0, lg / PEN, lg * PEN), lg)
                raw_ok, pen_ok = int(lg.argmax()) == tid, int(pen.argmax()) == tid
                n_raw += raw_ok; n_pen += pen_ok
                if not raw_ok and len(s["raw_mismatch_examples"]) < 5:
                    s["raw_mismatch_examples"].append({"index": r["index"], "pos": t, "recorded": tid,
                                                       "raw_argmax": int(lg.argmax())})
                seen[tid] = True
        s["tokens"] += len(cid); s["eq_raw_argmax"] += n_raw; s["eq_penalized_argmax"] += n_pen
        s["records_all_raw"] += n_raw == len(cid); s["records_all_penalized"] += n_pen == len(cid)
    s["frac_raw"] = s["eq_raw_argmax"] / max(1, s["tokens"]); s["frac_penalized"] = s["eq_penalized_argmax"] / max(1, s["tokens"])
    result["sets"][label] = s
    print(f"ARGMAX_CHECK {label}: {s['ids_recovered']}/{s['records']} records, {s['tokens']} tokens, "
          f"raw-argmax {s['frac_raw']:.4f} (all-raw records {s['records_all_raw']}), "
          f"penalized-argmax {s['frac_penalized']:.4f} (all-penalized records {s['records_all_penalized']})", flush=True)

json.dump(result, open(out_json, "w"), indent=1)
