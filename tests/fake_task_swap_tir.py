"""A stand-in for tools/task_swap_tir.py, called the way the reduced-(C) driver calls `subprocess.run`:
`[python, <tools>/task_swap_tir.py, fidelity|cells, ...]`. It writes what the real tool writes — `fidelity.json`, and one
JSONL per arm (fresh, stale_prompt, stale_shared on every index; fresh32 on the first --floor-n; near, far, fresh2 on the
first --control-n) with the tir-record-v1 keys — by rules a test chooses:

  * `fidelity(args) -> (turns_matched, turns_mismatched)`; default (100, 0), so the gate passes;
  * `outcome(index, step, arm, gold) -> (answer, answered_by, n_turns, reached_cap)`; default: FRESH answers gold on
    prompts with index % 2 == 0 through MathAnswer in 2 turns, STALE-prompt flips the answer on prompts with
    index % max(2, 8 // step) == 1 (more flips at larger steps: one reserved prompt at step 1, every odd one at step 32),
    STALE-shared flips on half of those, NEAR and FAR answer a wrong number everywhere, fresh32 == fresh except index 7,
    fresh2 == fresh.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ARMS = ("fresh", "stale_prompt", "stale_shared")
FLOOR_ARM = "fresh32"
CONTROL_ARMS = ("near", "far", "fresh2")


def default_outcome(index: int, step: int, arm: str, gold: str):
    fresh = gold if index % 2 == 0 else "0"
    if arm in ("fresh", "fresh2"):
        return fresh, "MathAnswer", 2, False
    if arm == "fresh32":
        return ("1" if index == 7 else fresh), "MathAnswer", 2, False
    period = max(2, 8 // max(step, 1))
    if arm == "stale_prompt":
        return ("5" if index % period == 1 else fresh), "MathAnswer", 2, index % 50 == 3
    if arm == "stale_shared":
        return ("5" if index % (2 * period) == 1 else fresh), "MathAnswer", 2, False
    return "9999", "last_content", 1, False


def _sha_tokens(tokens):
    import numpy as np
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class FakeTaskSwapTir:
    def __init__(self, fidelity=None, outcome=default_outcome, budget_over=False):
        self.fidelity = fidelity
        self.outcome = outcome
        self.budget_over = budget_over        # cells with --budget-seconds: write budget.json over budget and exit 4
        self.calls = []
        self.envs = []

    def __call__(self, cmd, cwd=None, capture_output=False, env=None, **kw):
        self.calls.append(list(cmd))
        self.envs.append(dict(env or {}))
        sub = cmd[2]
        rest = cmd[3:]
        args, i = {}, 0
        while i < len(rest):
            if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                args[rest[i]] = rest[i + 1]; i += 2
            else:
                args[rest[i]] = True; i += 1
        if sub == "fidelity":
            out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
            matched, mismatched = self.fidelity(args) if self.fidelity else (100, 0)
            rec = {"schema": "tir-fidelity-v1", "n_episodes": 50, "n_turns": matched + mismatched, "turns_matched": matched,
                   "turns_mismatched": mismatched, "turns_extend_cleanly": matched // 2, "content_eos": args["--content-eos"],
                   "max_model_len": int(args["--max-model-len"]), "per_episode": [], "passed": mismatched == 0}
            (out / "fidelity.json").write_text(json.dumps(rec), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0 if rec["passed"] else 3, b"", b"" if rec["passed"] else b"fidelity FAIL")
        if sub == "cells":
            out = Path(args["--out"]); out.mkdir(parents=True, exist_ok=True)
            # as the tool: budget.json is overwritten with checked:false at the start of EVERY cells call, and only the
            # 100-episode check (here: whenever --budget-seconds is passed) writes checked:true; run() records the latter only
            (out / "budget.json").write_text(json.dumps({"schema": "tir-budget-v1", "checked": False}), encoding="utf-8")
            if "--budget-seconds" in args:
                budget = float(args["--budget-seconds"])
                later = int(args.get("--later-episodes", 0))
                rep = {"schema": "tir-budget-v1", "checked": True, "episodes_ran": 100, "elapsed_s": 300.0, "seconds_per_episode": 3.0,
                       "episodes_projected": 100 + later, "projected_s": budget * (2.0 if self.budget_over else 0.5),
                       "budget_s": budget, "over": self.budget_over}
                (out / "budget.json").write_text(json.dumps(rep), encoding="utf-8")
                if self.budget_over:
                    return subprocess.CompletedProcess(cmd, 4, b"CELLS_HALT budget", b"")
            prompts = {int(p["index"]): p for p in json.loads(Path(args["--prompts"]).read_text(encoding="utf-8"))}
            indices = [int(i) for i in json.loads(Path(args["--indices"]).read_text(encoding="utf-8"))]
            step = int(args["--reader"].rsplit("_", 1)[1])
            floor_n = int(args.get("--floor-n", 0))
            control_n = int(args.get("--control-n", 0))
            for arm in ARMS + (FLOOR_ARM,) + CONTROL_ARMS:
                with open(out / f"{arm}.jsonl", "w", encoding="utf-8") as f:
                    for k, i in enumerate(indices):
                        if arm == FLOOR_ARM and k >= floor_n:
                            continue
                        if arm in CONTROL_ARMS and k >= control_n:
                            continue
                        gold = prompts[i]["gold"]
                        ans, by, turns, cap = self.outcome(i, step, arm, gold)
                        comp = [int(ans or 0) + 1, 2, 3]
                        seq = ["run_python_code", "MathAnswer"] if by == "MathAnswer" else ["run_python_code"]
                        rec = {"schema": "tir-record-v1", "index": i, "gold": gold, "answer": ans,
                               "math_answer_raw": (f"\\boxed{{{ans}}}" if by == "MathAnswer" else None), "answered_by": by,
                               "correct": ans == gold, "n_turns": turns,
                               "turns": [{"prompt_len": 400 + i % 7, "n_gen": 30, "reached_cap": cap, "tool_calls": [s], "lcp": 399 + i % 7,
                                          "prefilled": 1} for s in seq[:turns]],
                               "tool_sequence_sha256": _sha_text(",".join(seq[:turns])), "completion_sha256": _sha_tokens(comp),
                               "completion_texts": [f"#### {ans}"], "sandbox_hits": 0 if arm == "fresh" else 1,
                               "stale_span": None if arm in ("fresh", "fresh32", "fresh2") else (120 if arm == "stale_shared" else 399 + i % 7),
                               "stale_inside_lcp": None if arm in ("fresh", "fresh32", "fresh2") else True,
                               "reached_cap": cap, "b1_fresh_path": (0.001 * step * (1 + i % 3)) if arm == "stale_prompt" else None}
                        f.write(json.dumps(rec) + "\n")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return subprocess.CompletedProcess(cmd, 2, b"", b"fake task_swap_tir: unknown command")
