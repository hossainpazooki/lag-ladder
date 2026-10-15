"""A stand-in for tools/task_swap.py, called the way the (C) driver calls `subprocess.run`:
`[python, <tools>/task_swap.py, equivalence|cells, ...]`. It writes the files the real tool writes — `equivalence.json`,
and one JSONL per arm (fresh, fresh32, fresh2, stale, near, far) with the task-record-v1 keys — by rules a test chooses:

  * `equivalence(args) -> (max_abs_diff, b1_median)`; default (0.0, the --expect-b1 value);
  * `outcome(index, step, arm) -> (answer, reached_cap)`; default: FRESH answers gold on prompts with index % 2 == 0,
    STALE flips the answer on prompts with index % max(2, 40 // step) == 1 (more flips at larger steps), NEAR and FAR answer
    a wrong number everywhere, fresh32 == fresh except index 7, fresh2 == fresh.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ARMS = ("fresh", "fresh32", "fresh2", "stale", "near", "far")


def default_outcome(index: int, step: int, arm: str, gold: str):
    fresh = gold if index % 2 == 0 else "0"
    if arm in ("fresh", "fresh2"):
        return fresh, False
    if arm == "fresh32":
        return ("1" if index == 7 else fresh), False
    if arm == "stale":
        period = max(2, 40 // max(step, 1))
        return ("5" if index % period == 1 else fresh), index % 50 == 3
    return "9999", False


def _sha(tokens):
    import numpy as np
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


class FakeTaskSwap:
    def __init__(self, equivalence=None, outcome=default_outcome):
        self.equivalence = equivalence
        self.outcome = outcome
        self.calls = []

    def __call__(self, cmd, cwd=None, capture_output=False, **kw):
        self.calls.append(list(cmd))
        sub = cmd[2]
        rest = cmd[3:]
        args, i = {}, 0
        while i < len(rest):
            if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                args[rest[i]] = rest[i + 1]; i += 2
            else:
                args[rest[i]] = True; i += 1
        if sub == "equivalence":
            out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
            expect = float(args["--expect-b1"]) if "--expect-b1" in args else None
            diff, b1 = self.equivalence(args) if self.equivalence else (0.0, expect)
            rec = {"schema": "task-equivalence-v1", "writer": args["--writer"], "dump_root": args["--dump-root"],
                   "per_seq": [{"seq": s, "max_abs_diff": diff, "layers": 2, "rows": 3} for s in range(4)],
                   "max_abs_diff": diff,
                   "b1": None if expect is None else {"reader": args.get("--reader"), "median": b1, "expected": expect,
                                                      "rel_gap": abs(b1 - expect) / expect}}
            (out / "equivalence.json").write_text(json.dumps(rec), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if sub == "cells":
            out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
            prompts = {int(p["index"]): p for p in json.loads(Path(args["--prompts"]).read_text(encoding="utf-8"))}
            indices = [int(i) for i in json.loads(Path(args["--indices"]).read_text(encoding="utf-8"))]
            step = int(args["--reader"].rsplit("_", 1)[1])
            control_n = int(args.get("--control-n", len(indices)))
            for arm in ARMS:
                with open(out / f"{arm}.jsonl", "w", encoding="utf-8") as f:
                    for k, i in enumerate(indices):
                        if arm in ("fresh32", "fresh2") and k >= control_n:
                            continue
                        gold = prompts[i]["gold"]
                        ans, cap = self.outcome(i, step, arm, gold)
                        fresh_ans, _ = self.outcome(i, step, "fresh", gold)
                        comp = [int(ans or 0) + 1, 2, 3]
                        fresh_comp = [int(fresh_ans or 0) + 1, 2, 3]
                        rec = {"schema": "task-record-v1", "index": i, "prompt_len": 50 + i % 7, "n_tokens": 3,
                               "reached_cap": cap, "answer": ans, "gold": gold, "correct": ans == gold,
                               "completion_sha256": _sha(comp),
                               "first_divergence": None if arm == "fresh" else (None if comp == fresh_comp else 0),
                               "b1_fresh_path": (0.001 * step * (1 + i % 3)) if arm == "stale" else None}
                        f.write(json.dumps(rec) + "\n")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return subprocess.CompletedProcess(cmd, 2, b"", b"fake task_swap: unknown command")
