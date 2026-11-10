"""PipelineRL's actor stream to lag-ladder's inputs (design 2026-11-09 §5.1, §6; upstream pinned at 58d3934). Three
commands, no model:

  prompts    the frozen draw: n GSM8K test indices out of the 1,319 with lag_ladder.rng.make_rng(seed), without
             replacement, sorted; writes the frozen list, the custom JSONL upstream's loader reads from
             datasets/custom/<name>.jsonl ({id, task, answer, dataset}; load_datasets.py:236-272 at the pin) and the
             reserved [{index, question, gold}] for statistic (C).
  episodes   the actor-mode stream (<output_dir>/streams/actor/<instance>/<partition>/<shard>.jsonl, one JSON list of
             per-turn TrainingText dicts per group; streams.py:238-243, 249-277, actor.py:212-218; the TIR domain's
             `finished` means "MathAnswer was submitted", domains/tir/rollouts.py:365) to one episode per frozen
             prompt: turns grouped by (group_id, metadata.rollout_index) and ordered by metadata.step_index, matched to
             the prompt by the first user turn's text, the FIRST complete episode per prompt in stream order kept, the
             rest counted. Writes pltir-episode-v1 JSONL sorted by index and prints the coverage; exit 3 while a frozen
             prompt is missing, so the operator loops on it and stops the actor at exit 0. The actor may be writing
             meanwhile, so a shard's unterminated last line that does not parse is counted as `partial` and skipped.
             The episodes file is ASCII: json escapes the U+2028, U+2029 and U+0085 that model or sandbox text may hold
             and str.splitlines() would break a record on.
  ag-prompt  the first turn of every episode as the `ag-prompt` rollout pair: tokens = the turn's input_ids (the
             rendered system + tools + question prompt followed by the first assistant turn), prefix_len = its
             prompt_tokens, padded to --row-len with --pad-id through own_run.pad_rows; prints the two sha256 the
             registration pins.

Run with the instrument's interpreter (numpy). Imports own_run from this directory and lag_ladder.rng from ../src and
nothing else from lag_ladder; nothing from kvt or PipelineRL. Pure helpers are module level and tested offline on
synthetic stream lines (tests/test_pipelinerl_stream.py). All printing is ASCII.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
from own_run import pad_rows  # noqa: E402  (pure; own_run imports nothing heavy at import)
from lag_ladder.rng import make_rng  # noqa: E402  (the one lag_ladder import a tool may make)

EPISODE_SCHEMA = "pltir-episode-v1"
TURN_KEYS = ("input_ids", "labels", "prompt_tokens", "output_tokens", "finished", "text", "metadata")
_USER_TURN_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


# --- pure helpers ------------------------------------------------------------------------------------------------

def draw_frozen(n_total: int, n: int, seed: int) -> list[int]:
    """n distinct indices in [0, n_total) from make_rng(seed) without replacement, ascending (design §5.1)."""
    if not (0 < n <= n_total):
        raise ValueError(f"cannot draw {n} of {n_total}")
    rng = make_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))


def custom_rows(prompts: list[dict], frozen: list[int], dataset: str) -> list[dict]:
    """The custom-dataset rows in frozen order. `id` is the GSM8K test index: the loader renumbers `id` by row and keeps
    the original under `source_id`; `task` and `answer` are what the TIR rollout reads (domains/tir/rollouts.py:264, 341)."""
    by = {int(p["index"]): p for p in prompts}
    return [{"id": int(i), "task": by[int(i)]["question"], "answer": by[int(i)]["gold"], "dataset": dataset} for i in frozen]


def reserved_rows(prompts: list[dict], frozen: list[int]) -> list[dict]:
    fz = {int(i) for i in frozen}
    return [p for p in prompts if int(p["index"]) not in fz]


def question_of(text: str) -> str | None:
    """The first user turn of a rendered Qwen chat: the question (tool responses render as later user turns)."""
    m = _USER_TURN_RE.search(text)
    return m.group(1) if m else None


def leading_run(labels: list[int]) -> int:
    """The leading run of the mask value upstream wrote over the prompt (labels[0]; token ids are never negative, so the
    run ends at the first generated token). The mask constant is read from the record, not assumed."""
    if not labels:
        return 0
    mask, n = labels[0], 0
    for x in labels:
        if x != mask:
            break
        n += 1
    return n


def check_turn(turn: dict) -> str | None:
    """None when the per-turn record is consistent; else the reason. prompt_tokens (vLLM's count) must equal the masked
    run (the tokenizer's rendering): a difference means vLLM and the tokenizer rendered the prompt differently."""
    for k in TURN_KEYS:
        if k not in turn:
            return f"missing {k}"
    if len(turn["input_ids"]) != len(turn["labels"]):
        return "input_ids and labels differ in length"
    n = leading_run(turn["labels"])
    if n != int(turn["prompt_tokens"]):
        return f"leading masked run {n} != prompt_tokens {turn['prompt_tokens']}"
    return None


def group_episodes(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Per-turn records grouped by (group_id, rollout_index), each group's turns sorted by step_index."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        md = r.get("metadata") or {}
        groups.setdefault((str(r.get("group_id")), int(md.get("rollout_index", 0))), []).append(r)
    for turns in groups.values():
        turns.sort(key=lambda t: int((t.get("metadata") or {}).get("step_index", 0)))
    return groups


def _complete(turns: list[dict]) -> bool:
    return [int((t.get("metadata") or {}).get("step_index", -1)) for t in turns] == list(range(len(turns)))


def episode_from(turns: list[dict], prompts_by_question: dict[str, dict]) -> tuple[dict | None, str]:
    """(episode, reason): the pltir-episode-v1 record for a complete group whose first user turn is a frozen question;
    reasons are "ok", "incomplete", "not a frozen prompt", or the failing turn check."""
    if not turns or not _complete(turns):
        return None, "incomplete"
    for t in turns:
        bad = check_turn(t)
        if bad:
            return None, bad
    q = question_of(turns[0]["text"])
    if q is None:
        return None, "no user turn"
    p = prompts_by_question.get(q)
    if p is None:
        return None, "not a frozen prompt"
    md = turns[0]["metadata"]
    ep = {"schema": EPISODE_SCHEMA, "index": int(p["index"]), "question": p["question"], "gold": p["gold"],
          "group_id": turns[0].get("group_id"), "rollout_index": int(md.get("rollout_index", 0)),
          "model_version": md.get("model_version"),
          "turns": [{"step_index": int(t["metadata"]["step_index"]), "prompt_tokens": int(t["prompt_tokens"]),
                     "output_tokens": int(t["output_tokens"]), "finished": bool(t["finished"]),
                     "input_ids": [int(x) for x in t["input_ids"]], "labels": [int(x) for x in t["labels"]],
                     "text": t["text"]} for t in turns]}
    return ep, "ok"


def stream_files(streams_dir: Path) -> list[Path]:
    return sorted(p for p in Path(streams_dir).rglob("*.jsonl"))


def collect_episodes(files: list[Path], prompts_by_question: dict[str, dict]) -> tuple[dict[int, dict], dict]:
    """Every stream line of every file in order; the first complete episode per frozen prompt is kept.

    The actor may still be writing. Upstream writes each group as one line, then the newline, then flushes
    (streams.py:249-277 at the pin), so a file's last line can be on disk in part, cut anywhere, inside a multi-byte
    character included. Lines are therefore read as bytes: an unterminated last line that does not decode and parse is
    counted as `partial` and skipped (a later pass reads it whole); an unterminated last line that parses is whole (only
    its newline is pending) and is read. A newline-terminated line that does not parse is not a write in progress:
    ValueError names the file and line."""
    kept: dict[int, dict] = {}
    counts = {"lines": 0, "groups": 0, "kept": 0, "duplicates": 0, "incomplete": 0, "not_frozen": 0, "bad_turn": 0,
              "partial": 0}
    for f in files:
        with open(f, "rb") as fh:
            for lineno, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except ValueError as e:   # UnicodeDecodeError and JSONDecodeError
                    if not raw.endswith(b"\n"):
                        counts["partial"] += 1
                        continue
                    raise ValueError(f"{f}:{lineno}: a newline-terminated stream line does not parse ({e})") from e
                counts["lines"] += 1
                records = obj if isinstance(obj, list) else [obj]
                for turns in group_episodes(records).values():
                    counts["groups"] += 1
                    ep, why = episode_from(turns, prompts_by_question)
                    if ep is None:
                        counts["incomplete" if why == "incomplete" else "not_frozen" if why == "not a frozen prompt" else "bad_turn"] += 1
                        continue
                    if ep["index"] in kept:
                        counts["duplicates"] += 1
                        continue
                    kept[ep["index"]] = ep
                    counts["kept"] += 1
    return kept, counts


def ag_prompt_rows(episodes: list[dict]) -> list[tuple[list[int], int]]:
    """(tokens, prefix_len) per episode in index order: the first turn's input_ids and its prompt_tokens."""
    rows = []
    for ep in sorted(episodes, key=lambda e: int(e["index"])):
        t0 = ep["turns"][0]
        rows.append(([int(x) for x in t0["input_ids"]], int(t0["prompt_tokens"])))
    return rows


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_json(p: Path, obj) -> None:
    Path(p).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8", newline="\n")


# --- commands ------------------------------------------------------------------------------------------------------

def cmd_prompts(a) -> int:
    prompts = json.loads(Path(a.gsm8k).read_text(encoding="utf-8"))
    if a.expect_total is not None and len(prompts) != a.expect_total:
        raise SystemExit(f"{a.gsm8k} holds {len(prompts)} rows, expected {a.expect_total}")
    frozen = draw_frozen(len(prompts), a.n, a.seed)
    _write_json(a.frozen_out, {"seed": a.seed, "n": a.n, "total": len(prompts), "indices": frozen})
    with open(a.custom_out, "w", encoding="utf-8", newline="\n") as f:
        for row in custom_rows(prompts, frozen, a.dataset):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(a.reserved_out, reserved_rows(prompts, frozen))
    print(f"FROZEN rows={a.n} seed={a.seed} sha256={_sha(a.frozen_out)} {a.frozen_out}")
    print(f"CUSTOM rows={a.n} sha256={_sha(a.custom_out)} {a.custom_out}")
    print(f"RESERVED rows={len(prompts) - a.n} sha256={_sha(a.reserved_out)} {a.reserved_out}")
    return 0


def cmd_episodes(a) -> int:
    prompts = json.loads(Path(a.gsm8k).read_text(encoding="utf-8"))
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))["indices"]
    fz = {int(i) for i in frozen}
    by_q = {p["question"]: p for p in prompts if int(p["index"]) in fz}
    if len(by_q) != len(fz):
        raise SystemExit("two frozen prompts share a question text; the stream cannot be matched")
    kept, counts = collect_episodes(stream_files(a.streams), by_q)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(kept):
            f.write(json.dumps(kept[i], ensure_ascii=True) + "\n")   # ASCII: no Unicode line break inside a record
    missing = sorted(fz - set(kept))
    print(f"EPISODES kept {len(kept)}/{len(fz)} lines {counts['lines']} groups {counts['groups']} duplicates {counts['duplicates']} "
          f"incomplete {counts['incomplete']} not_frozen {counts['not_frozen']} bad_turn {counts['bad_turn']} "
          f"partial {counts['partial']} missing {missing}")
    return 0 if not missing else 3


def cmd_ag_prompt(a) -> int:
    lines = Path(a.episodes).read_text(encoding="utf-8").split("\n")   # newline only: splitlines() also breaks on U+2028
    episodes = [json.loads(l) for l in lines if l.strip()]
    arr, spans = pad_rows(ag_prompt_rows(episodes), a.row_len, a.pad_id)
    np.save(a.tokens_out, arr)
    _write_json(a.spans_out, spans)
    print(f"TOKENS rows={arr.shape[0]} sha256={_sha(a.tokens_out)} {a.tokens_out}")
    print(f"SPANS sha256={_sha(a.spans_out)} {a.spans_out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--gsm8k", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--n", type=int, required=True)
    p.add_argument("--expect-total", type=int, default=None); p.add_argument("--dataset", required=True)
    p.add_argument("--frozen-out", required=True); p.add_argument("--custom-out", required=True); p.add_argument("--reserved-out", required=True)
    p.set_defaults(fn=cmd_prompts)
    e = sub.add_parser("episodes")
    e.add_argument("--streams", required=True); e.add_argument("--gsm8k", required=True); e.add_argument("--frozen", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_episodes)
    g = sub.add_parser("ag-prompt")
    g.add_argument("--episodes", required=True); g.add_argument("--row-len", type=int, required=True); g.add_argument("--pad-id", type=int, required=True)
    g.add_argument("--tokens-out", required=True); g.add_argument("--spans-out", required=True)
    g.set_defaults(fn=cmd_ag_prompt)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
