"""Statistic (C) on PipelineRL's tool-integrated policy (design 2026-11-09, reduced (C)): the reader at step g replays
GSM8K test prompts as greedy multi-turn tool episodes under an injected prompt cache, and the final answers are compared
across arms. The system prompt, tool schema, Hermes tool-call format, argument parsing, tool messages, blocked-code
patterns and sandbox output format are PipelineRL's at 58d3934 (`pipelinerl/domains/tir/rollouts.py`,
`pipelinerl/rl_tool_parser_plugin.py`), ported here and checked literally by tests/test_task_swap_tir.py; the rendering
is the tokenizer's own chat template with the tools, as upstream renders it for the length check
(`domains/tir/rollouts.py:283-288`: `apply_chat_template(messages, add_special_tokens=True, add_generation_prompt=True,
tools=tools)`) and for the training text's prompt ids (`pipelinerl/async_llm.py:305-310`);
PipelineRL itself is never imported. Caches are built and injected as `task_swap.py` builds them (its helpers are
imported from this directory, with `score_continuation.py` behind them; nothing from `kvt` or `lag_ladder`).

    python tools/task_swap_tir.py fidelity --episodes FILE --tokenizer SPEC --system-prompt FILE
                                           --content-eos keep|strip --max-model-len 8192 --out DIR
    python tools/task_swap_tir.py cells --reader local:pltir-step_g --writer local:pltir-step_0 --prompts FILE
                                        --indices FILE --far-tokens FILE.npy --far-seq 40 --max-new 1024 --max-turns 3
                                        --max-model-len 8192 --min-generation 256 --floor-n N --control-n N
                                        --system-prompt FILE --content-eos keep|strip --out DIR
                                        [--budget-seconds S] [--later-episodes N]

The cache rule (design open item 3). Upstream re-renders every turn from its message list: the assistant turn comes
back as `content` (the parser's whole raw output on the Hermes path) plus `tool_calls` (the template re-emits each as
`<tool_call>{"name": ..., "arguments": <tojson of the JSON string>}</tool_call>`), and tool results come back as a
`user` turn wrapping `<tool_response>`. So turn t+1's prompt need not extend turn t's prompt plus its generated tokens.
The loop therefore keeps the cache up to the longest common prefix of the cached tokens and the new rendering, and
prefills the rest under the reader, in every arm alike; `turns[].lcp` and `turns[].prefilled` record what happened and
the record says whether the stale span lay inside that prefix. `fidelity` is the gate on the rendering: given upstream's
own 50 actor-mode episodes, every turn's prompt rendered here must equal upstream's `prompt_tokens`-long prefix of its
`input_ids` token for token (a mismatch exits 3); it also counts the turns whose prompt extends the previous turn's
prompt + generation cleanly. `--content-eos` says whether the assistant `content` in the message list keeps the
trailing eos string (vLLM returned it with `skip_special_tokens: False`, `include_stop_str_in_output: True`) or not;
the gate is run with both and the registration pins the one that matches.

Arms, per prompt: fresh (the reader's own cache throughout), stale_prompt (the step-0 writer's cache of the rendered
system + tools + question, the ag-prompt span), stale_shared (the writer's cache of system + tools alone, the span an
engine with prefix caching keeps across episodes) on every prompt; fresh32 (the floor: fresh from a cache not rounded
through float16) on the first --floor-n prompts; near (another prompt's cache cut to this prompt's length), far (a
held-out web prefix of this prompt's length) and fresh2 (the identity: fresh generated again) on the first --control-n
prompts (RC3). Every prompt cache is rounded through float16 except fresh32's. As in task_swap, the injected cache
covers the prompt's rows but the last; the last prompt token is prefilled under the reader to yield the first generated
token, so the stale span of stale_prompt, near and far is prompt_len - 1; stale_shared injects the writer's rows of the
shared span only and the reader prefills the question over them (a stale prefix, not a splice of two caches), so its
span is the shared span. Later turns' tokens are always the reader's own. Decision 4's halt: after 100 episodes of a
shard the projection (seconds per episode x (the shard's episodes + --later-episodes); the whole shard, since its first
100 are paid out of the same budget) is printed and written to budget.json with checked:true; budget.json is overwritten
with checked:false before the first episode of every call, so a caller reads this call's projection or none; with
--budget-seconds it exits 4 when over. Decoding
is raw argmax, stopping on the tokenizer's eos, as task_swap._greedy. The sandbox (SandboxFusion's REST
`POST <SANDBOX_ENDPOINT>/run_code`, loopback only) is memoised per prompt across arms so a flip cannot come from a
nondeterministic tool result; hits are recorded. Printing is ASCII; torch is imported inside the commands only.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from own_run import final_number  # noqa: E402
from task_swap import (is_correct, near_pairing, tokens_sha256, _layers_through_float16, _mean_kl_fresh_path,  # noqa: E402
                       _sc, _to_torch, _tokenizer_for)

SCHEMA = "tir-record-v1"
FIDELITY_SCHEMA = "tir-fidelity-v1"
ARMS = ("fresh", "stale_prompt", "stale_shared")          # every prompt
FLOOR_ARM = "fresh32"                                      # the first --floor-n prompts (RC3: all of them)
CONTROL_ARMS = ("near", "far", "fresh2")                   # the first --control-n prompts (RC3: 400)
ALL_ARMS = ARMS + (FLOOR_ARM,) + CONTROL_ARMS
RECORD_KEYS = frozenset({
    "schema", "index", "gold", "answer", "math_answer_raw", "answered_by", "correct", "n_turns", "turns",
    "tool_sequence_sha256", "completion_sha256", "completion_texts", "sandbox_hits", "stale_span", "stale_inside_lcp",
    "reached_cap", "b1_fresh_path",
})
CONTENT_EOS = ("keep", "strip")

# --- the port: pipelinerl/domains/tir/rollouts.py at 58d3934 -----------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python_code",
        "description": "Execute Python code. Print only the final result.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "MathAnswer",
        "description": "Submit the final answer in LaTeX \\boxed{} format.",
        "parameters": {"type": "object",
                       "properties": {"answer": {"type": "string", "description": "The final answer"}},
                       "required": ["answer"]}}},
]

BLOCKED_PATTERNS = [re.compile(p) for p in (
    r"\bsys\.exit\b", r"\bos\._exit\b", r"\bos\.system\b", r"\bsubprocess\b", r"\bos\.popen\b", r"\bos\.exec\w*\b",
    r"\bos\.spawn\w*\b", r"\bos\.kill\b", r"\bshutil\.rmtree\b", r"\bos\.remove\b", r"\bos\.unlink\b")]


def check_code_safety(code: str) -> str | None:
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(code):
            return f"Blocked: code contains forbidden pattern '{pattern.pattern}'"
    return None


def format_sandbox_output(stdout: str, stderr: str, status: str, message: str) -> str:
    """rollouts.py:98-110: stdout, `[stderr]` block, `[execution timed out]` when "timeout" is in the status or the
    message, `[no output]` when nothing else; joined by newlines."""
    is_timeout = "timeout" in (status or "").lower() or "timeout" in (message or "").lower()
    parts = []
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    if is_timeout:
        parts.append("[execution timed out]")
    if not parts:
        parts.append("[no output]")
    return "\n".join(parts)


def parse_tool_arguments(arguments, fallback_key: str | None) -> dict:
    """rollouts.py:134-149."""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    if fallback_key is not None and isinstance(parsed, str):
        return {fallback_key: parsed}
    return {}


# --- the port: pipelinerl/rl_tool_parser_plugin.py at 58d3934 ----------------------------------------------------

_JSON_SCALAR_TYPES = (dict, list, str, int, float, bool)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
_FINAL_RESPONSE_RE = re.compile(r"\[BEGIN FINAL RESPONSE\](.*?)\[END FINAL RESPONSE\]", re.DOTALL)
_TOOL_CALLS_RE = re.compile(r"<tool_calls>\s*(.*?)\s*(?:</tool_calls>|$)", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_START = "<tool_call>"


def _build_tool_call(index: int, parsed: dict, *, force_id: str | None = None) -> dict | None:
    """plugin.py:21-39: arguments re-serialised with json.dumps(ensure_ascii=False); a non-scalar becomes {}."""
    try:
        args_obj = parsed.get("arguments", {})
        if not isinstance(args_obj, _JSON_SCALAR_TYPES):
            args_obj = {}
        call_id = force_id if force_id is not None else parsed.get("id", f"call_{index}")
        return {"id": call_id, "type": "function",
                "function": {"name": str(parsed.get("name", "")), "arguments": json.dumps(args_obj, ensure_ascii=False)}}
    except Exception:
        return None


def parse_tool_calls(model_output: str, tools_declared: bool = True) -> tuple[list[dict], str]:
    """HermesRLToolParser.extract_tool_calls (plugin.py:75-201) as (tool_calls, content). Path (a): a `<tool_calls>`
    aggregate wins; (b) with tools declared, bare JSON with name and arguments; (c) `<tool_call>` blocks, ids
    `call_<i>`, malformed JSON skipped, content = the whole output unless a final-response wrapper is present."""
    final_response_match = None
    try:
        tool_calls_matches = list(_TOOL_CALLS_RE.finditer(model_output))
        if tool_calls_matches:
            tool_calls_json = tool_calls_matches[-1].group(1).strip()
            try:
                parsed_calls = json.loads(tool_calls_json) if tool_calls_json else []
            except Exception:
                parsed_calls = []
            tool_calls = [tc for tc in (_build_tool_call(i, pc) for i, pc in enumerate(parsed_calls)) if tc is not None]
            final_response_match = _FINAL_RESPONSE_RE.search(model_output)
            return tool_calls, (final_response_match.group(1).strip() if final_response_match else "")
        if tools_declared:
            candidates = []
            final_response_match = _FINAL_RESPONSE_RE.search(model_output)
            if final_response_match:
                candidates.append(final_response_match.group(1).strip())
            candidates.append(model_output.strip())
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    continue
                parsed_list = []
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    parsed_list = [parsed]
                elif isinstance(parsed, list) and all(isinstance(it, dict) for it in parsed):
                    parsed_list = [it for it in parsed if "name" in it and "arguments" in it]
                if not parsed_list:
                    continue
                tool_calls = [tc for tc in (_build_tool_call(i, pc) for i, pc in enumerate(parsed_list)) if tc is not None]
                return tool_calls, (final_response_match.group(1).strip() if final_response_match else "")
        content_to_search = model_output
        final_response_match = _FINAL_RESPONSE_RE.search(model_output)
        if final_response_match:
            final_response_content = final_response_match.group(1).strip()
            if _TOOL_CALL_START in final_response_content:
                content_to_search = final_response_content
            elif _TOOL_CALL_START not in model_output:
                return [], final_response_content
        if _TOOL_CALL_START not in content_to_search:
            return [], model_output
        tool_calls = []
        for i, match in enumerate(_TOOL_CALL_RE.findall(content_to_search)):
            json_str = match[0] if match[0] else match[1]
            try:
                parsed_call = json.loads(json_str.strip())
            except Exception:
                continue
            tc = _build_tool_call(i, parsed_call, force_id=f"call_{i}")
            if tc is not None:
                tool_calls.append(tc)
        if tool_calls and final_response_match:
            content = ""
        elif final_response_match:
            content = final_response_match.group(1).strip()
        else:
            content = model_output
        return tool_calls, content
    except Exception:
        if final_response_match:
            return [], final_response_match.group(1).strip()
        return [], model_output


# --- the sandbox client and the handlers -----------------------------------------------------------------------------

def _post_json(req: urllib.request.Request, timeout: float) -> dict:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class SandboxClient:
    """SandboxFusion's REST shape (sandbox-fusion 0.3.7, client.py:85-94, models.py:28-40): POST /run_code with
    {code, language, run_timeout}; the response's status, message and run_result {status, stdout, stderr}. Five
    attempts with a short back-off, as the SDK's tenacity policy; any failure after that is `[execution error: ...]`
    exactly as rollouts.py:116 renders an exception. The endpoint must be loopback (design decision 4)."""

    def __init__(self, endpoint: str, timeout: float, attempts: int = 5):
        host = urllib.parse.urlsplit(endpoint).hostname
        try:
            ok = host is not None and (host == "localhost" or ipaddress.ip_address(host).is_loopback)
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"SANDBOX_ENDPOINT must be loopback (127.0.0.1 or localhost), got host {host!r}")
        self.url = endpoint.rstrip("/") + "/run_code"
        self.timeout, self.attempts = float(timeout), int(attempts)

    def run(self, code: str) -> str:
        body = json.dumps({"code": code, "language": "python", "run_timeout": self.timeout}).encode("utf-8")
        last = None
        for attempt in range(self.attempts):
            req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                resp = _post_json(req, self.timeout + 5.0)
            except Exception as exc:  # noqa: BLE001  (every failure is rendered, never raised: rollouts.py:114-116)
                last = exc
                time.sleep(min(2.0, 0.2 * (attempt + 1)))
                continue
            rr = resp.get("run_result") or {}
            return format_sandbox_output(rr.get("stdout") or "", rr.get("stderr") or "", str(resp.get("status", "")),
                                         resp.get("message") or "")
        return f"[execution error: {last}]"


class SandboxMemo:
    """code -> result per key (one prompt), so every arm of a prompt sees the same tool result for the same code."""

    def __init__(self):
        self._m: dict[tuple, dict[str, str]] = {}

    def get(self, key, code: str) -> str | None:
        return self._m.get(key, {}).get(code)

    def put(self, key, code: str, result: str) -> None:
        self._m.setdefault(key, {})[code] = result


@dataclass
class ToolContext:
    sandbox: object
    memo: SandboxMemo
    memo_key: tuple
    messages: list = field(default_factory=list)
    final_answer: str | None = None
    submitted: bool = False
    num_python_calls: int = 0
    sandbox_hits: int = 0


def handle_tool_call(tc: dict, ctx: ToolContext) -> None:
    """rollouts.py:164-188: MathAnswer sets the answer and `Answer submitted: ...`; run_python_code runs the code (or
    its block message) and appends the result; anything else `Unknown tool: <name>`. The code path is memoised."""
    name = tc["function"]["name"]
    if name == "MathAnswer":
        args = parse_tool_arguments(tc["function"]["arguments"], "answer")
        ctx.final_answer = args.get("answer", "")
        ctx.submitted = True
        ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"Answer submitted: {ctx.final_answer}"})
        return
    if name == "run_python_code":
        args = parse_tool_arguments(tc["function"]["arguments"], "code")
        code = args.get("code") or args.get("python_code", "")
        ctx.num_python_calls += 1
        rejection = check_code_safety(code)
        if rejection is not None:
            result = rejection
        else:
            hit = ctx.memo.get(ctx.memo_key, code)
            if hit is not None:
                ctx.sandbox_hits += 1
                result = hit
            else:
                result = ctx.sandbox.run(code) if ctx.sandbox is not None else "[execution error: no sandbox]"
                ctx.memo.put(ctx.memo_key, code, result)
        ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return
    ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"Unknown tool: {name}"})


# --- rendering ---------------------------------------------------------------------------------------------------------

def episode_prompt_messages(system_prompt: str, question: str) -> list[dict]:
    """rollouts.py:261-264 with tir.yaml's task_template "{task}"."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def render_ids(tok, messages: list[dict], tools: list[dict]) -> list[int]:
    """The prompt as upstream renders it (async_llm.py:305-310: the template with the tools and the generation prompt),
    rendered to text and then tokenized with add_special_tokens=False, as task_swap.cmd_cells does (the Qwen template
    carries its own special tokens; the tokenizer adds none). Rendering through text keeps one code path across the
    transformers versions whose tokenize=True return types differ; the fidelity gate is what certifies the result."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def shared_prefix_len(tok, system_prompt: str, tools: list[dict]) -> int:
    """The ag-shared span: the rendered system turn with the tools (the template puts the tool schema inside the
    system block), which every episode's prompt starts with."""
    text = tok.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False, add_generation_prompt=False, tools=tools)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def strip_content_eos(content: str, eos: str, mode: str) -> str:
    if mode not in CONTENT_EOS:
        raise ValueError(f"content_eos must be one of {CONTENT_EOS}")
    if mode == "strip" and eos and content.endswith(eos):
        return content[: -len(eos)]
    return content


def text_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --- the turn loop over an engine --------------------------------------------------------------------------------------

def run_episode(engine, tok, system_prompt: str, question: str, *, sandbox, memo: SandboxMemo, memo_key: tuple,
                max_new: int, max_turns: int, max_model_len: int, min_generation: int, content_eos: str,
                stale_span: int | None) -> dict:
    """rollouts.py:279-328 against an engine with cache_tokens(), truncate(n), prefill(ids), generate(max_new) ->
    (ids, text). The engine's cache holds the arm's initial prompt cache when this is called (the caller built it and
    prefilled the prompt's tokens into it); each turn keeps the cache to the lcp with the new rendering and prefills
    the rest. Returns the episode dict cell_record() consumes."""
    messages = episode_prompt_messages(system_prompt, question)
    ctx = ToolContext(sandbox=sandbox, memo=memo, memo_key=memo_key, messages=messages)
    turns, completion_ids, completion_texts, tool_sequence = [], [], [], []
    stale_inside = None if stale_span is None else True
    reached_cap = False
    for _turn in range(max_turns):
        prompt_ids = render_ids(tok, messages, TOOLS)
        prompt_len = len(prompt_ids)
        remaining = max_model_len - prompt_len
        if remaining < min_generation:
            break
        cached = engine.cache_tokens()
        n = lcp(cached, prompt_ids)
        if stale_span is not None and n < stale_span:
            stale_inside = False
        if len(cached) > n:
            engine.truncate(n)
        if prompt_len > n:
            engine.prefill(prompt_ids[n:])
        gen_ids, gen_text = engine.generate(min(max_new, remaining))
        cap = len(gen_ids) >= min(max_new, remaining) and (not gen_ids or gen_ids[-1] != tok.eos_token_id)
        reached_cap = reached_cap or cap
        completion_ids.append(list(gen_ids))
        completion_texts.append(gen_text)
        calls, content = parse_tool_calls(gen_text)
        names = [c["function"]["name"] for c in calls]
        tool_sequence.extend(names)
        turns.append({"prompt_len": prompt_len, "n_gen": len(gen_ids), "reached_cap": bool(cap), "tool_calls": names,
                      "lcp": n, "prefilled": prompt_len - n})
        if not calls:
            break
        messages.append({"role": "assistant", "content": strip_content_eos(content or "", tok.eos_token or "", content_eos),
                         "tool_calls": calls})
        for tc in calls:
            handle_tool_call(tc, ctx)
            if ctx.submitted:
                break
        if ctx.submitted:
            break
    if ctx.final_answer is not None:
        answered_by, raw = "MathAnswer", ctx.final_answer
        answer = final_number(ctx.final_answer)
    elif completion_texts:
        answered_by, raw = "last_content", None
        answer = final_number(tok.decode(completion_ids[-1], skip_special_tokens=True))
    else:
        answered_by, raw, answer = "none", None, None
    return {"answer": answer, "math_answer_raw": raw, "answered_by": answered_by, "n_turns": len(turns), "turns": turns,
            "tool_sequence": tool_sequence, "completion_ids": completion_ids, "completion_texts": completion_texts,
            "sandbox_hits": ctx.sandbox_hits, "stale_span": stale_span, "stale_inside_lcp": stale_inside,
            "reached_cap": reached_cap,
            # the two hashes cell_record() writes, computed the same way here so an episode can be read without a record
            "tool_sequence_sha256": text_sha256(",".join(tool_sequence)),
            "completion_sha256": tokens_sha256([t for ids in completion_ids for t in ids])}


def cell_record(*, index: int, gold: str, episode: dict, b1_fresh_path: float | None) -> dict:
    flat = [t for ids in episode["completion_ids"] for t in ids]
    return {"schema": SCHEMA, "index": int(index), "gold": gold, "answer": episode["answer"],
            "math_answer_raw": episode["math_answer_raw"], "answered_by": episode["answered_by"],
            "correct": is_correct(episode["answer"], gold), "n_turns": int(episode["n_turns"]), "turns": episode["turns"],
            "tool_sequence_sha256": text_sha256(",".join(episode["tool_sequence"])),
            "completion_sha256": tokens_sha256(flat), "completion_texts": episode["completion_texts"],
            "sandbox_hits": int(episode["sandbox_hits"]), "stale_span": episode["stale_span"],
            "stale_inside_lcp": episode["stale_inside_lcp"], "reached_cap": bool(episode["reached_cap"]),
            "b1_fresh_path": b1_fresh_path}


# --- the fidelity gate ----------------------------------------------------------------------------------------------

def fidelity_report(episodes: list[dict], tok, system_prompt: str, *, content_eos: str, max_model_len: int) -> dict:
    """For every episode, replay upstream's own assistant texts and tool results through this file's message
    construction and rendering, and compare turn by turn: turn k's prompt here = render(messages_k) must equal the first
    prompt_tokens ids of upstream's turn-k input_ids; messages_{k+1} = messages_k + the assistant turn parsed from
    decode(gen_k) + one tool message per call, whose content is read back from upstream's turn-(k+1) `text` (the
    rendered prompt followed by the completion) between `<tool_response>\\n` and `\\n</tool_response>`, the last
    len(calls) of them in order. A mismatch records the first diverging position. Also counts the turns whose prompt
    extends the previous prompt + generation (clean extension), for open item 3."""
    per, matched, mismatched, clean, n_turns = [], 0, 0, 0, 0
    for ep in episodes:
        messages = episode_prompt_messages(system_prompt, ep["question"])
        rows = []
        prev_full = None
        for k, t in enumerate(ep["turns"]):
            n_turns += 1
            pl = int(t["prompt_tokens"])
            upstream_prompt = [int(x) for x in t["input_ids"][:pl]]
            gen = [int(x) for x in t["input_ids"][pl:]]
            ours = render_ids(tok, messages, TOOLS)
            d = None if ours == upstream_prompt else lcp(ours, upstream_prompt)
            ok = d is None
            matched += ok
            mismatched += not ok
            ext = prev_full is not None and upstream_prompt[: len(prev_full)] == prev_full
            clean += bool(ext)
            rows.append({"step_index": int(t["step_index"]), "prompt_len": pl, "ours_len": len(ours), "match": ok,
                         "first_divergence": d, "extends_previous": bool(ext) if prev_full is not None else None})
            prev_full = upstream_prompt + gen
            gen_text = tok.decode(gen, skip_special_tokens=False)
            calls, content = parse_tool_calls(gen_text)
            if k + 1 >= len(ep["turns"]):
                # upstream writes `finished` = "the answer was submitted" on every turn of an episode (rollouts.py:364-365).
                # At the last recorded turn this port must agree: it parses a MathAnswer there if and only if finished is
                # true. A disagreement is a divergence in how the episode ends, which no rendering shows (there is no later
                # turn); the turn then counts as mismatched, with first_divergence None and ours_len set
                answered = any(c["function"]["name"] == "MathAnswer" for c in calls)
                if ok and answered != bool(t["finished"]):
                    matched -= 1
                    mismatched += 1
                    rows[-1]["match"] = False
                break
            if not calls:
                # upstream recorded another turn, so its loop went on where this port parsed no tool call: a divergence
                # of the parser or of the loop, not the end of the episode. The later turns cannot be rendered from here;
                # each is counted as mismatched (first_divergence None), so the gate fails instead of passing on what it
                # skipped
                for later in ep["turns"][k + 1:]:
                    n_turns += 1
                    mismatched += 1
                    rows.append({"step_index": int(later["step_index"]), "prompt_len": int(later["prompt_tokens"]),
                                 "ours_len": None, "match": False, "first_divergence": None, "extends_previous": None})
                break
            messages.append({"role": "assistant", "content": strip_content_eos(content or "", tok.eos_token or "", content_eos),
                             "tool_calls": calls})
            results = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", ep["turns"][k + 1]["text"], re.DOTALL)
            results = results[-len(calls):] if len(results) >= len(calls) else results
            for tc, res in zip(calls, results):
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
        per.append({"index": int(ep["index"]), "turns": rows})
    return {"schema": FIDELITY_SCHEMA, "n_episodes": len(episodes), "n_turns": n_turns, "turns_matched": matched,
            "turns_mismatched": mismatched, "turns_extend_cleanly": clean, "content_eos": content_eos,
            "max_model_len": int(max_model_len), "per_episode": per, "passed": mismatched == 0 and n_turns > 0}


def cmd_fidelity(a) -> int:
    tok = _tokenizer_for(a.tokenizer)
    system_prompt = Path(a.system_prompt).read_text(encoding="utf-8")
    episodes = [json.loads(l) for l in Path(a.episodes).read_text(encoding="utf-8").split("\n") if l.strip()]
    rep = fidelity_report(episodes, tok, system_prompt, content_eos=a.content_eos, max_model_len=a.max_model_len)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"FIDELITY episodes {rep['n_episodes']} turns {rep['n_turns']} matched {rep['turns_matched']} "
          f"mismatched {rep['turns_mismatched']} extend_cleanly {rep['turns_extend_cleanly']} content_eos {a.content_eos} "
          f"-> {'PASS' if rep['passed'] else 'FAIL'}", flush=True)
    return 0 if rep["passed"] else 3


# --- the GPU engine (torch imported here only) -------------------------------------------------------------------------

class TorchEngine:
    """The engine interface over score_continuation's injection: the cache is a DynamicCache built from float32 (K, V)
    layers plus the ids it covers; prefill runs one forward over the new ids at the right positions; generate is raw
    argmax one token per forward, stopping after eos (task_swap._greedy's recipe). `ids` (what cache_tokens() returns)
    are exactly the rows the cache holds: the eos generate stops on is returned but never forwarded, so it is no row
    and not in `ids`; a next turn whose rendering carries it (--content-eos keep) finds the lcp at the last forwarded
    row and prefills the eos at its own position instead of skipping it over a cache one row short."""

    def __init__(self, model, tok, device):
        import torch
        self.torch, self.model, self.tok, self.device = torch, model, tok, device
        self.sc = _sc()
        self.cache, self.ids = None, []

    def reset(self, layers, ids: list[int]):
        self.cache = self.sc.build_cache(layers, self.device)
        self.ids = list(ids)

    def cache_tokens(self):
        return list(self.ids)

    def _layers(self):
        c = self.cache
        out = []
        for l in range(self.model.config.num_hidden_layers):
            k, v = (c.layers[l].keys, c.layers[l].values) if hasattr(c, "layers") else (c.key_cache[l], c.value_cache[l])
            out.append((k[0].permute(1, 0, 2).float().cpu(), v[0].permute(1, 0, 2).float().cpu()))
        return out

    def truncate(self, n: int):
        layers = [(k[:n], v[:n]) for k, v in self._layers()]
        self.cache = self.sc.build_cache(layers, self.device)
        self.ids = self.ids[:n]

    def prefill(self, ids: list[int]):
        torch = self.torch
        past = len(self.ids)
        with torch.no_grad():
            pos = torch.arange(past, past + len(ids), device=self.device)
            attn = torch.ones(1, past + len(ids), dtype=torch.long, device=self.device)
            out = self.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=self.cache, attention_mask=attn,
                             position_ids=pos[None], cache_position=pos, use_cache=True)
        self.cache = out.past_key_values
        self.ids.extend(int(t) for t in ids)
        self._last_logits = out.logits[0, -1].float()

    def generate(self, max_new: int):
        torch = self.torch
        eos = self.tok.eos_token_id
        gen = []
        with torch.no_grad():
            tok = int(self._last_logits.argmax())
            for _ in range(max_new):
                gen.append(tok)
                if eos is not None and tok == eos:
                    break           # returned, never forwarded: no cache row, so not in self.ids
                past = len(self.ids)
                pos = torch.arange(past, past + 1, device=self.device)
                attn = torch.ones(1, past + 1, dtype=torch.long, device=self.device)
                out = self.model(input_ids=torch.tensor([[tok]], device=self.device), past_key_values=self.cache, attention_mask=attn,
                                 position_ids=pos[None], cache_position=pos, use_cache=True)
                self.cache = out.past_key_values
                self.ids.append(tok)
                self._last_logits = out.logits[0, -1].float()
                tok = int(self._last_logits.argmax())
        return gen, self.tok.decode(gen, skip_special_tokens=False)


def injected_rows(arm: str, pl: int, shared: int) -> tuple[int, int | None]:
    """How many rows of the arm's whole-prompt cache are injected before the first turn, and the stale span recorded
    (None for the fresh arms). Every arm injects the prompt's rows but the last (whose forward under the reader yields
    the first generated token) except stale_shared, which injects the writer's rows of the shared span only: the
    question's rows are then the reader's own, computed over the stale prefix, so the arm is a stale prefix and not a
    splice of two caches computed over different prefixes. Never past the last prompt token."""
    last = pl - 1
    if arm == "stale_shared":
        rows = min(shared, last)
        return rows, rows
    return last, (last if arm in ("stale_prompt", "near", "far") else None)


BUDGET_CHECK_AT = 100     # decision 4: "after 100 episodes per shard"
RC_BUDGET = 4             # the cells exit code when the projection exceeds --budget-seconds


def budget_projection(elapsed_s: float, episodes_ran: int, episodes_projected: int, budget_s: float | None) -> tuple[float, bool]:
    """Decision 4's rule at the shard level: the seconds per episode so far, scaled to the episodes the projection
    covers (this whole shard, the episodes already run included, plus the later shards'; the budget it is compared with
    is the one that stood at the shard's start, which paid for those episodes too); returns (projected seconds, over
    budget). A budget of None never halts; the projection is still printed."""
    if episodes_ran <= 0:
        raise ValueError("budget_projection needs at least one finished episode")
    projected = elapsed_s / episodes_ran * episodes_projected
    return projected, (budget_s is not None and projected > budget_s)


def _prompt_cache(sc, model, ids: list[int], device, rounded: bool):
    """The arm's cache of the whole prompt (every position; the first generated token is read from the prefill of the
    last prompt token, so all prompt rows are injected). Rounded through float16 unless the floor arm."""
    import torch
    layers = sc.live_prefix_layers(model, torch.tensor(ids, dtype=torch.int64), len(ids), device)
    return _to_torch(_layers_through_float16(layers)) if rounded else [(k.float().cpu(), v.float().cpu()) for k, v in layers]


def _b1_fresh_path(reader, fresh_layers, stale_layers, prompt_ids: list[int], cont: list[int], device) -> float | None:
    """(B1) on the fresh path for the first turn: the FRESH first-turn tokens teacher-forced under both caches, through
    task_swap's own routine (the registered instrument's), so the number is 0027's statistic on an agentic first turn."""
    if not cont:
        return None
    rows = len(prompt_ids) - 1
    return float(_mean_kl_fresh_path(reader, [(k[:rows], v[:rows]) for k, v in fresh_layers],
                                     [(k[:rows], v[:rows]) for k, v in stale_layers], prompt_ids[-1], rows, list(cont), device))


def cmd_cells(a) -> int:
    import torch
    sc = _sc()
    device = sc.pick_device(a.device)
    endpoint = os.environ.get("SANDBOX_ENDPOINT", "")
    if not endpoint:
        raise SystemExit("SANDBOX_ENDPOINT is not set; the sandbox must be up on loopback before any cell")
    sandbox = SandboxClient(endpoint, timeout=a.sandbox_timeout)
    system_prompt = Path(a.system_prompt).read_text(encoding="utf-8")
    prompts = {int(p["index"]): p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
    indices = [int(i) for i in json.loads(Path(a.indices).read_text(encoding="utf-8"))]
    tok = _tokenizer_for(a.reader)
    reader = sc.load_reader(a.reader, torch.float32, device)
    writer = sc.load_reader(a.writer, torch.float32, device)
    far_tokens = np.load(a.far_tokens)
    enc = {i: render_ids(tok, episode_prompt_messages(system_prompt, prompts[i]["question"]), TOOLS) for i in indices}
    shared = shared_prefix_len(tok, system_prompt, TOOLS)
    lengths = [len(enc[i]) for i in indices]
    near_of = {indices[k]: indices[v] for k, v in near_pairing(lengths).items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {arm: open(out / f"{arm}.jsonl", "a", encoding="utf-8", newline="\n") for arm in ALL_ARMS}
    done = {arm: set() for arm in ALL_ARMS}
    for arm in ALL_ARMS:
        p = out / f"{arm}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    done[arm].add(json.loads(line)["index"])
    first_path = out / "fresh_first_ids.jsonl"
    fresh_first_ids: dict[int, list[int]] = {}
    if first_path.exists():
        for line in first_path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                fresh_first_ids[int(row["index"])] = [int(t) for t in row["ids"]]
    first_file = open(first_path, "a", encoding="utf-8", newline="\n")
    memo = SandboxMemo()
    engine = TorchEngine(reader, tok, device)

    def arms_of(n: int) -> list[str]:
        return list(ARMS) + ([FLOOR_ARM] if n < a.floor_n else []) + (list(CONTROL_ARMS) if n < a.control_n else [])

    episodes_todo = sum(1 for n, i in enumerate(indices) for arm in arms_of(n) if i not in done[arm])
    episodes_ran, checked = 0, False
    # this call's report or nothing: overwritten with checked:false before the first episode, so a resume that finishes a
    # shard in under BUDGET_CHECK_AT episodes leaves no earlier sitting's projection (possibly over) for run() to read
    (out / "budget.json").write_text(json.dumps({"schema": "tir-budget-v1", "checked": False}, indent=1), encoding="utf-8",
                                     newline="\n")
    t0 = time.monotonic()
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    for n, i in enumerate(indices):
        ids = enc[i]
        pl = len(ids)
        arms = arms_of(n)
        if all(i in done[arm] for arm in arms):
            continue
        fresh_live = sc.live_prefix_layers(reader, torch.tensor(ids, dtype=torch.int64), pl, device)
        fresh = _to_torch(_layers_through_float16(fresh_live))
        stale_full = _prompt_cache(sc, writer, ids, device, rounded=True)
        # the arm's whole-prompt cache; injected_rows() says how many of its rows are injected (the prompt's rows but
        # the last for every arm except stale_shared, which injects the writer's shared-span rows only) and the span
        # recorded. The cache is always taken to cover the prompt's own ids, so run_episode's first turn finds the lcp
        # at the injected rows and prefills the rest under the reader (the last prompt token, or for stale_shared the
        # whole question over the stale prefix). stale_shared is therefore a stale prefix, never a splice of two
        # caches computed over different prefixes.
        caches = {"fresh": fresh, "stale_prompt": stale_full, "stale_shared": stale_full}
        if n < a.floor_n:
            caches[FLOOR_ARM] = [(k.float().cpu(), v.float().cpu()) for k, v in fresh_live]
        if n < a.control_n:
            caches["near"] = _prompt_cache(sc, reader, (enc[near_of[i]] + [pad] * pl)[:pl], device, rounded=True)
            far_row = [int(t) for t in far_tokens[a.far_seq + (n % (far_tokens.shape[0] - a.far_seq))]]
            caches["far"] = _prompt_cache(sc, reader, (far_row + [pad] * pl)[:pl], device, rounded=True)
            caches["fresh2"] = fresh
        eps = {}
        for arm in arms:
            if i in done[arm]:
                continue
            rows, span = injected_rows(arm, pl, shared)
            engine.reset([(k[:rows], v[:rows]) for k, v in caches[arm]], ids[:rows])
            eps[arm] = run_episode(engine, tok, system_prompt, prompts[i]["question"], sandbox=sandbox, memo=memo, memo_key=("prompt", i),
                                   max_new=a.max_new, max_turns=a.max_turns, max_model_len=a.max_model_len,
                                   min_generation=a.min_generation, content_eos=a.content_eos, stale_span=span)
            episodes_ran += 1
        if "fresh" in eps:
            # FRESH's first-turn tokens, kept beside the records so (B1) on a resumed shard (fresh done, stale_prompt
            # not) teacher-forces the same tokens it would have in one sitting
            fresh_first_ids[i] = list(eps["fresh"]["completion_ids"][0]) if eps["fresh"]["completion_ids"] else []
            first_file.write(json.dumps({"index": i, "ids": fresh_first_ids[i]}) + "\n")
            first_file.flush()
        if "stale_prompt" in eps and i not in fresh_first_ids:
            raise SystemExit(f"fresh.jsonl holds prompt {i} but {first_path} does not; the shard cannot be resumed")
        b1 = _b1_fresh_path(reader, fresh, stale_full, ids, fresh_first_ids[i], device) if "stale_prompt" in eps else None
        for arm, ep in eps.items():
            rec = cell_record(index=i, gold=prompts[i]["gold"], episode=ep, b1_fresh_path=b1 if arm == "stale_prompt" else None)
            files[arm].write(json.dumps(rec, ensure_ascii=True) + "\n")
            files[arm].flush()
        if (n + 1) % 10 == 0:
            print(f"CELLS {n + 1}/{len(indices)} {time.monotonic() - t0:.0f}s", flush=True)
        if not checked and episodes_ran >= BUDGET_CHECK_AT:
            # decision 4: after 100 episodes of a shard, the projected cell total against the budget passed at launch
            checked = True
            elapsed = time.monotonic() - t0
            # the whole shard, not its remainder: the episodes already run are paid out of the same S (taken at the
            # shard's start), so projected = elapsed + seconds per episode x the rest, plus the later shards' episodes
            covers = episodes_todo + a.later_episodes
            projected, over = budget_projection(elapsed, episodes_ran, covers, a.budget_seconds)
            report = {"schema": "tir-budget-v1", "checked": True, "episodes_ran": episodes_ran, "elapsed_s": elapsed,
                      "seconds_per_episode": elapsed / episodes_ran, "episodes_projected": covers, "projected_s": projected,
                      "budget_s": a.budget_seconds, "over": over}
            (out / "budget.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
            print(f"CELLS_PROJECTION episodes {episodes_ran} elapsed {elapsed:.0f}s per-episode {elapsed / episodes_ran:.1f}s "
                  f"projected {projected:.0f}s over {covers} episodes budget {a.budget_seconds}", flush=True)
            if over:
                for f in files.values():
                    f.close()
                first_file.close()
                print(f"CELLS_HALT budget: projected {projected:.0f}s exceeds the {a.budget_seconds:.0f}s remaining (decision 4)", flush=True)
                return RC_BUDGET
    for f in files.values():
        f.close()
    first_file.close()
    print(f"CELLS_DONE {len(indices)} prompts", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fidelity")
    f.add_argument("--episodes", required=True); f.add_argument("--tokenizer", required=True)
    f.add_argument("--system-prompt", required=True); f.add_argument("--content-eos", choices=CONTENT_EOS, required=True)
    f.add_argument("--max-model-len", type=int, default=8192); f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fidelity)
    c = sub.add_parser("cells")
    c.add_argument("--reader", required=True); c.add_argument("--writer", required=True)
    c.add_argument("--prompts", required=True); c.add_argument("--indices", required=True)
    c.add_argument("--far-tokens", required=True); c.add_argument("--far-seq", type=int, default=40)
    c.add_argument("--max-new", type=int, default=1024); c.add_argument("--max-turns", type=int, default=3)
    c.add_argument("--max-model-len", type=int, default=8192); c.add_argument("--min-generation", type=int, default=256)
    c.add_argument("--floor-n", type=int, default=0, help="prompts (from the first) that also get the floor arm fresh32")
    c.add_argument("--control-n", type=int, default=0, help="prompts (from the first) that also get the identity arm fresh2")
    c.add_argument("--system-prompt", required=True); c.add_argument("--content-eos", choices=CONTENT_EOS, required=True)
    c.add_argument("--sandbox-timeout", type=float, default=10.0); c.add_argument("--out", required=True)
    c.add_argument("--device", default="auto")
    c.add_argument("--budget-seconds", type=float, default=None,
                   help="decision 4: wall seconds the cells may still use; after 100 episodes the projection is compared and the "
                        "tool exits 4 if it is over (no budget: the projection is printed only)")
    c.add_argument("--later-episodes", type=int, default=0,
                   help="episodes of the shards after this one, added to this whole shard's episodes in the projection")
    c.set_defaults(fn=cmd_cells)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
