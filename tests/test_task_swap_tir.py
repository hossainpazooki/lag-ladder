"""tools/task_swap_tir.py's pure parts (design 2026-11-09, reduced (C)): upstream's tool schema, blocked patterns,
sandbox output format, Hermes parser (three paths and the fallback), argument parsing and handlers ported from
PipelineRL at 58d3934 and checked against literal expectations; the turn loop against a fake engine and a fake sandbox;
the fidelity gate against synthetic episodes rendered with a fake tokenizer. No torch, no model, no network."""
import contextlib
import importlib.util
import json
import sys
import types

import pytest

from lag_ladder import REPO_ROOT

spec = importlib.util.spec_from_file_location("task_swap_tir", REPO_ROOT / "tools" / "task_swap_tir.py")
tt = importlib.util.module_from_spec(spec)
sys.modules["task_swap_tir"] = tt          # dataclasses resolve string annotations through sys.modules
spec.loader.exec_module(tt)

SYSTEM = "You are a math-focused AI Agent.\n"


# --- the port, against literal upstream values -------------------------------------------------------------------

def test_tool_schema_is_upstreams_build_tool_definitions():
    assert [t["function"]["name"] for t in tt.TOOLS] == ["run_python_code", "MathAnswer"]
    assert tt.TOOLS[0] == {"type": "function", "function": {"name": "run_python_code",
                           "description": "Execute Python code. Print only the final result.",
                           "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                                          "required": ["code"]}}}
    assert tt.TOOLS[1]["function"]["description"] == "Submit the final answer in LaTeX \\boxed{} format."
    assert tt.TOOLS[1]["function"]["parameters"]["required"] == ["answer"]


def test_blocked_patterns_and_messages_match_upstream():
    assert [p.pattern for p in tt.BLOCKED_PATTERNS] == [r"\bsys\.exit\b", r"\bos\._exit\b", r"\bos\.system\b", r"\bsubprocess\b",
                                                        r"\bos\.popen\b", r"\bos\.exec\w*\b", r"\bos\.spawn\w*\b", r"\bos\.kill\b",
                                                        r"\bshutil\.rmtree\b", r"\bos\.remove\b", r"\bos\.unlink\b"]
    assert tt.check_code_safety("print(1+1)") is None
    assert tt.check_code_safety("import subprocess\nprint(2)") == "Blocked: code contains forbidden pattern '\\bsubprocess\\b'"
    assert tt.check_code_safety("os.execv('x')") == "Blocked: code contains forbidden pattern '\\bos\\.exec\\w*\\b'"


def test_sandbox_output_format_matches_upstream():
    assert tt.format_sandbox_output("42\n", "", "Success", "") == "42"
    assert tt.format_sandbox_output("", "Traceback\n", "Failed", "") == "[stderr]\nTraceback"
    assert tt.format_sandbox_output("1\n", "warn\n", "Success", "") == "1\n[stderr]\nwarn"
    assert tt.format_sandbox_output("", "", "SandboxError", "run timeout") == "[execution timed out]"
    assert tt.format_sandbox_output("x", "", "Success", "TimeOut hit") == "x\n[execution timed out]"
    assert tt.format_sandbox_output("", "", "Success", "") == "[no output]"


def test_parse_tool_arguments_matches_upstream_fallbacks():
    assert tt.parse_tool_arguments('{"code": "print(1)"}', "code") == {"code": "print(1)"}
    assert tt.parse_tool_arguments('"print(1)"', "code") == {"code": "print(1)"}
    assert tt.parse_tool_arguments("not json", "code") == {}
    assert tt.parse_tool_arguments(None, "code") == {}
    assert tt.parse_tool_arguments("[1, 2]", "code") == {}
    assert tt.parse_tool_arguments("3", None) == {}


def test_parser_path_c_hermes_blocks_ids_arguments_and_content():
    out = 'Plan.\n<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call><|im_end|>'
    calls, content = tt.parse_tool_calls(out)
    assert len(calls) == 1 and calls[0]["id"] == "call_0" and calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "run_python_code"
    assert calls[0]["function"]["arguments"] == '{"code": "print(6*7)"}'     # json.dumps(args, ensure_ascii=False)
    assert content == out                                                    # content = the whole output on path (c)
    # two calls, the second with malformed JSON: skipped, ids stay by position
    out2 = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call><tool_call>\n{bad\n</tool_call>'
    calls2, _ = tt.parse_tool_calls(out2)
    assert [c["function"]["name"] for c in calls2] == ["MathAnswer"] and calls2[0]["id"] == "call_0"
    assert json.loads(calls2[0]["function"]["arguments"]) == {"answer": "\\boxed{42}"}
    # an unterminated block parses through the second alternative
    calls3, _ = tt.parse_tool_calls('<tool_call>{"name": "MathAnswer", "arguments": {"answer": "1"}}')
    assert calls3 and calls3[0]["function"]["name"] == "MathAnswer"
    # no block: no calls, content = the output
    assert tt.parse_tool_calls("The answer is 7.") == ([], "The answer is 7.")
    # a non-scalar arguments value becomes {}
    calls4, _ = tt.parse_tool_calls('<tool_call>{"name": "x", "arguments": null}</tool_call>')
    assert calls4[0]["function"]["arguments"] == "{}"


def test_parser_path_a_and_b_and_final_response_wrapper():
    # (a) <tool_calls> aggregated JSON wins over everything, content = the final response or ""
    out = '<tool_calls>[{"name": "MathAnswer", "arguments": {"answer": "2"}, "id": "abc"}]</tool_calls>'
    calls, content = tt.parse_tool_calls(out)
    assert calls[0]["id"] == "abc" and content == ""
    calls, content = tt.parse_tool_calls("<tool_calls>not json</tool_calls>")
    assert calls == [] and content == ""
    # (b) bare JSON with name and arguments when tools are declared
    calls, content = tt.parse_tool_calls('{"name": "run_python_code", "arguments": {"code": "1"}}')
    assert calls[0]["function"]["name"] == "run_python_code" and calls[0]["id"] == "call_0" and content == ""
    calls, content = tt.parse_tool_calls('{"name": "run_python_code", "arguments": {"code": "1"}}', tools_declared=False)
    assert calls == [] and content == '{"name": "run_python_code", "arguments": {"code": "1"}}'
    # the final-response wrapper on path (c): content is the wrapped text when there are no calls, "" with calls
    wrapped = "[BEGIN FINAL RESPONSE]done[END FINAL RESPONSE]"
    assert tt.parse_tool_calls(wrapped) == ([], "done")
    out = '[BEGIN FINAL RESPONSE]<tool_call>{"name": "MathAnswer", "arguments": {"answer": "5"}}</tool_call>[END FINAL RESPONSE]'
    calls, content = tt.parse_tool_calls(out)
    assert calls[0]["function"]["name"] == "MathAnswer" and content == ""


def test_handlers_build_upstreams_tool_messages_and_memoise_the_sandbox():
    class FakeSandbox:
        def __init__(self):
            self.calls = []

        def run(self, code):
            self.calls.append(code)
            return f"ran:{code}"

    sb = FakeSandbox()
    memo = tt.SandboxMemo()
    ctx = tt.ToolContext(sandbox=sb, memo=memo, memo_key=("prompt", 3), messages=[])
    tt.handle_tool_call({"id": "call_0", "type": "function", "function": {"name": "run_python_code", "arguments": '{"code": "print(1)"}'}}, ctx)
    tt.handle_tool_call({"id": "call_1", "type": "function", "function": {"name": "run_python_code", "arguments": '{"code": "print(1)"}'}}, ctx)
    tt.handle_tool_call({"id": "call_2", "type": "function", "function": {"name": "run_python_code", "arguments": '"print(2)"'}}, ctx)
    tt.handle_tool_call({"id": "call_3", "type": "function", "function": {"name": "nope", "arguments": "{}"}}, ctx)
    assert sb.calls == ["print(1)", "print(2)"] and ctx.sandbox_hits == 1 and ctx.num_python_calls == 3
    assert ctx.messages[0] == {"role": "tool", "tool_call_id": "call_0", "content": "ran:print(1)"}
    assert ctx.messages[3] == {"role": "tool", "tool_call_id": "call_3", "content": "Unknown tool: nope"}
    assert not ctx.submitted
    tt.handle_tool_call({"id": "call_4", "type": "function", "function": {"name": "MathAnswer", "arguments": '{"answer": "\\\\boxed{18}"}'}}, ctx)
    assert ctx.submitted and ctx.final_answer == "\\boxed{18}"
    assert ctx.messages[-1] == {"role": "tool", "tool_call_id": "call_4", "content": "Answer submitted: \\boxed{18}"}
    # blocked code never reaches the sandbox; the memo is per key
    ctx2 = tt.ToolContext(sandbox=sb, memo=memo, memo_key=("prompt", 4), messages=[])
    tt.handle_tool_call({"id": "c", "type": "function", "function": {"name": "run_python_code", "arguments": '{"code": "import subprocess"}'}}, ctx2)
    assert ctx2.messages[0]["content"].startswith("Blocked:") and sb.calls == ["print(1)", "print(2)"]
    tt.handle_tool_call({"id": "d", "type": "function", "function": {"name": "run_python_code", "arguments": '{"code": "print(1)"}'}}, ctx2)
    assert sb.calls == ["print(1)", "print(2)", "print(1)"] and ctx2.sandbox_hits == 0


def test_sandbox_client_refuses_a_non_loopback_endpoint_and_formats_errors(monkeypatch):
    with pytest.raises(ValueError, match="loopback"):
        tt.SandboxClient("http://10.0.0.5:8080", timeout=10.0)
    c = tt.SandboxClient("http://127.0.0.1:8080", timeout=10.0)
    assert c.url == "http://127.0.0.1:8080/run_code"
    calls = []

    def boom(req, timeout):
        calls.append(req)
        raise OSError("connection refused")

    monkeypatch.setattr(tt, "_post_json", boom)
    monkeypatch.setattr(tt.time, "sleep", lambda s: None)
    assert c.run("print(1)") == "[execution error: connection refused]" and len(calls) == 5
    monkeypatch.setattr(tt, "_post_json", lambda req, timeout: {"status": "Success", "message": "",
                                                                  "run_result": {"status": "Finished", "stdout": "3\n", "stderr": ""}})
    assert c.run("print(3)") == "3"
    body = json.loads(calls[0].data.decode("utf-8"))
    assert body == {"code": "print(1)", "language": "python", "run_timeout": 10.0}


# --- rendering and the cache rule ---------------------------------------------------------------------------------

class FakeTok:
    """Renders a chat as upstream's Qwen template would in shape (one token per character after a fixed header); the
    template facts the real gate checks (tojson arguments, tool responses as user turns) are the real tokenizer's job."""
    eos_token = "<|im_end|>"
    eos_token_id = 0
    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kw):
        s = "T" + json.dumps(tools or []) + "\n"
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                s += "<|im_start|>assistant\n" + (m.get("content") or "")
                for tc in m["tool_calls"]:
                    s += '\n<tool_call>\n{"name": "' + tc["function"]["name"] + '", "arguments": ' + json.dumps(tc["function"]["arguments"]) + '}\n</tool_call>'
                s += "<|im_end|>\n"
            elif m["role"] == "tool":
                s += "<|im_start|>user\n<tool_response>\n" + m["content"] + "\n</tool_response><|im_end|>\n"
            else:
                s += "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}          # one token per character; eos id 0 is no character

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i != self.eos_token_id)


def test_lcp_and_render_ids_and_eos_strip():
    assert tt.lcp([1, 2, 3], [1, 2, 4]) == 2 and tt.lcp([], [1]) == 0 and tt.lcp([1, 2], [1, 2]) == 2
    tok = FakeTok()
    msgs = tt.episode_prompt_messages(SYSTEM, "What is 6*7?")
    assert msgs == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is 6*7?"}]
    ids = tt.render_ids(tok, msgs, tt.TOOLS)
    assert ids == tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=tt.TOOLS))["input_ids"]
    assert tt.strip_content_eos("abc<|im_end|>", "<|im_end|>", "strip") == "abc"
    assert tt.strip_content_eos("abc<|im_end|>", "<|im_end|>", "keep") == "abc<|im_end|>"
    assert tt.strip_content_eos("abc", "<|im_end|>", "strip") == "abc"
    assert tt.shared_prefix_len(tok, SYSTEM, tt.TOOLS) == len(tok(tok.apply_chat_template([{"role": "system", "content": SYSTEM}], tokenize=False, tools=tt.TOOLS))["input_ids"])


class FakeEngine:
    """The engine interface the loop drives: a cache of token ids with prefill/truncate/generate; generation is a script
    per call so a test decides the assistant text (the whole script is returned as text even when the ids are cut at
    max_new, so a cap test reads reached_cap, not the text). Records every call for the cache-rule assertions."""

    def __init__(self, tok, scripts):
        self.tok, self.scripts, self.cache, self.log = tok, list(scripts), [], []

    def cache_tokens(self):
        return list(self.cache)

    def truncate(self, n):
        self.log.append(("truncate", n)); self.cache = self.cache[:n]

    def prefill(self, ids):
        self.log.append(("prefill", len(ids))); self.cache.extend(ids)

    def generate(self, max_new):
        text = self.scripts.pop(0)
        ids = self.tok(text)["input_ids"][:max_new]
        if len(ids) < max_new:
            ids = ids + [self.tok.eos_token_id]
        self.log.append(("generate", len(ids))); self.cache.extend(ids)
        return ids, text


def test_run_episode_keeps_the_cache_to_the_lcp_prefills_the_rest_and_stops_after_mathanswer():
    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call>'
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call>'
    eng = FakeEngine(tok, ["Plan.\n" + call, ans])

    class SB:
        def run(self, code):
            return "42"

    rec = tt.run_episode(eng, tok, SYSTEM, "What is 6*7?", sandbox=SB(), memo=tt.SandboxMemo(), memo_key=("p", 0),
                         max_new=256, max_turns=3, max_model_len=4096, min_generation=8, content_eos="strip",
                         stale_span=None)
    assert rec["answered_by"] == "MathAnswer" and rec["math_answer_raw"] == "\\boxed{42}" and rec["answer"] == "42"
    assert rec["n_turns"] == 2 and [t["tool_calls"] for t in rec["turns"]] == [["run_python_code"], ["MathAnswer"]]
    # turn 0: the whole prompt is prefilled; turn 1: the cache is kept to the lcp with the re-rendered prompt (the
    # generated raw block differs from the template's tojson'd call) and the remainder prefilled
    kinds = [k for k, _ in eng.log]
    assert kinds[:2] == ["prefill", "generate"] and kinds[2:4] == ["truncate", "prefill"] and kinds[4] == "generate"
    t1 = rec["turns"][1]
    assert 0 < t1["lcp"] < t1["prompt_len"] and t1["prefilled"] == t1["prompt_len"] - t1["lcp"]
    assert rec["completion_sha256"] and rec["tool_sequence_sha256"] and rec["sandbox_hits"] == 0
    assert rec["completion_texts"] == ["Plan.\n" + call, ans] and rec["reached_cap"] is False


def test_run_episode_without_a_tool_call_reads_the_last_number_and_a_cap_is_recorded():
    tok = FakeTok()
    eng = FakeEngine(tok, ["The answer is #### 12"])
    rec = tt.run_episode(eng, tok, SYSTEM, "q", sandbox=None, memo=tt.SandboxMemo(), memo_key=("p", 1),
                         max_new=256, max_turns=3, max_model_len=4096, min_generation=8, content_eos="strip", stale_span=None)
    assert rec["answered_by"] == "last_content" and rec["answer"] == "12" and rec["n_turns"] == 1 and rec["math_answer_raw"] is None
    eng = FakeEngine(tok, ["x" * 200])
    rec = tt.run_episode(eng, tok, SYSTEM, "q", sandbox=None, memo=tt.SandboxMemo(), memo_key=("p", 2),
                         max_new=16, max_turns=3, max_model_len=4096, min_generation=8, content_eos="strip", stale_span=None)
    assert rec["turns"][0]["reached_cap"] is True and rec["reached_cap"] is True and rec["answer"] is None


def test_run_episode_stops_when_the_context_leaves_too_little_and_marks_the_stale_span():
    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(1)"}}\n</tool_call>'
    eng = FakeEngine(tok, [call, call, call])

    class SB:
        def run(self, code):
            return "1"

    prompt_ids = tt.render_ids(tok, tt.episode_prompt_messages(SYSTEM, "q"), tt.TOOLS)
    eng.cache = prompt_ids[:-1]                    # as cells injects it: every prompt row but the last
    rec = tt.run_episode(eng, tok, SYSTEM, "q", sandbox=SB(), memo=tt.SandboxMemo(), memo_key=("p", 3),
                         max_new=64, max_turns=3, max_model_len=len(prompt_ids) + 100, min_generation=8, content_eos="strip",
                         stale_span=len(prompt_ids) - 1)
    assert 1 <= rec["n_turns"] < 3 and rec["stale_span"] == len(prompt_ids) - 1 and rec["stale_inside_lcp"] is True
    assert rec["turns"][0]["lcp"] == len(prompt_ids) - 1 and rec["turns"][0]["prefilled"] == 1
    assert rec["answered_by"] in ("none", "last_content")
    # a cache that does not reach the span is recorded as such
    eng = FakeEngine(tok, [call])
    rec = tt.run_episode(eng, tok, SYSTEM, "q", sandbox=SB(), memo=tt.SandboxMemo(), memo_key=("p", 4),
                         max_new=256, max_turns=1, max_model_len=4096, min_generation=8, content_eos="strip",
                         stale_span=len(prompt_ids) - 1)
    assert rec["stale_inside_lcp"] is False and rec["turns"][0]["lcp"] == 0


def test_run_episode_over_a_shared_span_cache_prefills_the_question_and_keeps_the_span():
    tok = FakeTok()
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{7}"}}\n</tool_call>'
    eng = FakeEngine(tok, [ans])

    class SB:
        def run(self, code):
            return "7"

    prompt_ids = tt.render_ids(tok, tt.episode_prompt_messages(SYSTEM, "What is 3+4?"), tt.TOOLS)
    shared = tt.shared_prefix_len(tok, SYSTEM, tt.TOOLS)
    assert 0 < shared < len(prompt_ids) - 1
    eng.cache = prompt_ids[:shared]                 # as cells injects stale_shared: the writer's shared-span rows only
    rec = tt.run_episode(eng, tok, SYSTEM, "What is 3+4?", sandbox=SB(), memo=tt.SandboxMemo(), memo_key=("p", 5),
                         max_new=64, max_turns=3, max_model_len=4096, min_generation=8, content_eos="strip", stale_span=shared)
    assert eng.log[0] == ("prefill", len(prompt_ids) - shared)     # the question is the reader's own, over the stale prefix
    assert rec["turns"][0]["lcp"] == shared and rec["turns"][0]["prefilled"] == len(prompt_ids) - shared
    assert rec["stale_span"] == shared and rec["stale_inside_lcp"] is True and rec["answer"] == "7"


# --- the GPU engine's cache bookkeeping, over stand-ins for torch and the model ------------------------------------

class EosTok(FakeTok):
    """FakeTok with the eos as the special token a real tokenizer makes of it: "<|im_end|>" in rendered text is the
    single id 0, and decode renders id 0 as that string unless asked to skip it (TorchEngine decodes with
    skip_special_tokens=False, so under --content-eos keep the assistant content carries it into the next rendering)."""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for k, part in enumerate(text.split(self.eos_token)):
            if k:
                ids.append(self.eos_token_id)
            ids.extend(ord(ch) for ch in part)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        eos = "" if skip_special_tokens else self.eos_token
        return "".join(eos if i == self.eos_token_id else chr(i) for i in ids)


class _RowCache:
    """A DynamicCache stood in by the ids of the rows it holds, in position order."""

    def __init__(self, rows):
        self.rows = [int(t) for t in rows]


class _Predicts:
    """`out.logits[0, -1].float().argmax()` of a forward whose next-token prediction is `tok`."""

    def __init__(self, tok):
        self.tok = tok

    def __getitem__(self, idx):
        return self

    def float(self):
        return self

    def argmax(self):
        return self.tok


class _ScriptedLM:
    """A causal LM without weights: every forward must be fed the cache's next positions (a real DynamicCache appends
    rows at whatever position_ids it is given, so a gap is silent there and an assertion here); it appends the fed ids
    as rows and predicts the next token of the script, one prediction per forward."""

    def __init__(self, script):
        self.script = list(script)
        self.config = types.SimpleNamespace(num_hidden_layers=1)

    def __call__(self, *, input_ids, past_key_values, attention_mask, position_ids, cache_position, use_cache):
        held, fed = past_key_values.rows, [int(t) for t in input_ids[0]]
        want = list(range(len(held), len(held) + len(fed)))
        got = [int(p) for p in cache_position]
        assert got == want and [int(p) for p in position_ids[0]] == want, \
            f"forward fed at positions {got[:1]}.. over a cache of {len(held)} rows"
        assert attention_mask.shape[1] == len(held) + len(fed)
        return types.SimpleNamespace(past_key_values=_RowCache(held + fed), logits=_Predicts(self.script.pop(0)))


def test_torch_engine_ids_are_its_cached_rows_so_a_kept_eos_is_prefilled_at_its_own_position(monkeypatch):
    """TorchEngine returns the eos it stops on but never forwards it, so the eos is no cached row. Under
    --content-eos keep the next turn's rendering carries that eos where the generation ended; if the engine's ids held
    it, the lcp would run through it, nothing would be prefilled for it and every later row would sit one position past
    the cache (silently, on a real DynamicCache). The real TorchEngine and run_episode, over stand-ins for torch, the
    scorer's cache builder and the model: the engine's ids are exactly the rows it has forwarded, the eos is prefilled
    at its own position, and the cache ends holding every id the engine claims."""
    import numpy as np

    tok = EosTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call>'
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call>'
    turn0, turn1 = tok(call)["input_ids"] + [tok.eos_token_id], tok(ans)["input_ids"] + [tok.eos_token_id]
    fake_torch = types.ModuleType("torch")
    fake_torch.long = "long"
    fake_torch.no_grad = contextlib.nullcontext
    fake_torch.arange = lambda a, b, device=None: np.arange(a, b)
    fake_torch.ones = lambda *shape, dtype=None, device=None: np.ones(shape)
    fake_torch.tensor = lambda data, device=None, dtype=None: np.asarray(data)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(tt, "_sc", lambda: types.SimpleNamespace(build_cache=lambda layers, device: _RowCache(layers)))
    lm = _ScriptedLM(turn0 + turn1)            # the prefill of each turn predicts its first token; the eos is never fed
    eng = tt.TorchEngine(lm, tok, "cpu")
    prompt_ids = tt.render_ids(tok, tt.episode_prompt_messages(SYSTEM, "What is 6*7?"), tt.TOOLS)
    eng.reset(prompt_ids[:-1], prompt_ids[:-1])  # as cells injects it; the stand-in cache is built from the rows' ids

    class SB:
        def run(self, code):
            return "42"

    rec = tt.run_episode(eng, tok, SYSTEM, "What is 6*7?", sandbox=SB(), memo=tt.SandboxMemo(), memo_key=("p", 6),
                         max_new=256, max_turns=3, max_model_len=4096, min_generation=8, content_eos="keep", stale_span=None)
    assert rec["answered_by"] == "MathAnswer" and rec["answer"] == "42" and rec["n_turns"] == 2
    assert rec["completion_ids"] == [turn0, turn1] and rec["reached_cap"] is False   # the eos is still returned
    t0, t1 = rec["turns"]
    # turn 1's rendering carries turn 0's eos where the generation ended: the cache is kept through the last forwarded
    # row (the prompt and the generation but its eos), and the eos is prefilled at its own position
    assert t1["lcp"] == t0["prompt_len"] + t0["n_gen"] - 1 and t1["prefilled"] == t1["prompt_len"] - t1["lcp"]
    rows = eng.cache.rows
    assert rows == eng.cache_tokens() and lm.script == []
    assert rows[: t0["prompt_len"]] == prompt_ids and rows[t0["prompt_len"]: t0["prompt_len"] + len(turn0)] == turn0
    assert rows[t1["lcp"]] == tok.eos_token_id and len(rows) == t1["prompt_len"] + len(turn1) - 1


def test_injected_rows_keep_every_prompt_row_but_the_last_except_stale_shared_which_injects_the_shared_span():
    assert tt.injected_rows("fresh", 400, 120) == (399, None)
    assert tt.injected_rows("fresh32", 400, 120) == (399, None) and tt.injected_rows("fresh2", 400, 120) == (399, None)
    assert tt.injected_rows("stale_prompt", 400, 120) == (399, 399)
    assert tt.injected_rows("near", 400, 120) == (399, 399) and tt.injected_rows("far", 400, 120) == (399, 399)
    assert tt.injected_rows("stale_shared", 400, 120) == (120, 120)
    assert tt.injected_rows("stale_shared", 100, 120) == (99, 99)       # never past the last prompt token


def test_budget_projection_scales_the_seconds_per_episode_and_flags_an_overrun():
    assert tt.BUDGET_CHECK_AT == 100 and tt.RC_BUDGET == 4
    assert tt.budget_projection(50.0, 100, 1000, None) == (500.0, False)
    assert tt.budget_projection(50.0, 100, 1000, 500.0) == (500.0, False)
    assert tt.budget_projection(50.0, 100, 1000, 499.0) == (500.0, True)
    with pytest.raises(ValueError):
        tt.budget_projection(1.0, 0, 10, 5.0)


def test_cell_record_has_the_schema_keys_and_the_tool_sequence_hash_is_over_names():
    rec = tt.cell_record(index=3, gold="42", episode={"answer": "42", "math_answer_raw": "\\boxed{42}", "answered_by": "MathAnswer",
                                                      "n_turns": 2, "turns": [], "tool_sequence": ["run_python_code", "MathAnswer"],
                                                      "completion_ids": [[1, 2], [3]], "completion_texts": ["a", "b"], "sandbox_hits": 0,
                                                      "stale_span": None, "stale_inside_lcp": None, "reached_cap": False},
                         b1_fresh_path=None)
    assert set(rec) == tt.RECORD_KEYS and rec["schema"] == "tir-record-v1" and rec["correct"] is True
    assert rec["tool_sequence_sha256"] == tt.text_sha256("run_python_code,MathAnswer")
    assert rec["completion_sha256"] == tt.tokens_sha256([1, 2, 3])


# --- the fidelity gate on synthetic episodes ----------------------------------------------------------------------

def _episode_with(tok, question, assistant_texts, tool_results, index=5):
    """A pltir-episode-v1 record rendered with the fake tokenizer, as upstream would have written it."""
    msgs = tt.episode_prompt_messages(SYSTEM, question)
    turns = []
    for k, text in enumerate(assistant_texts):
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=tt.TOOLS)
        prompt_ids = tt.render_ids(tok, msgs, tt.TOOLS)
        gen = tok(text)["input_ids"] + [tok.eos_token_id]
        # upstream's `text` is the rendered prompt followed by the completion; turn k+1's prompt carries turn k's tool
        # responses, which is where the gate reads them back
        turns.append({"step_index": k, "prompt_tokens": len(prompt_ids), "output_tokens": len(gen), "finished": True,
                      "input_ids": prompt_ids + gen, "labels": [-100] * len(prompt_ids) + gen, "text": prompt_text + text})
        calls, content = tt.parse_tool_calls(text)
        if not calls:
            break
        msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
        for tc, res in zip(calls, tool_results[k]):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
    # upstream writes whether the answer was submitted on every turn of the episode (rollouts.py:364-365)
    fin = any(c["function"]["name"] == "MathAnswer" for c in tt.parse_tool_calls(assistant_texts[len(turns) - 1])[0])
    for t in turns:
        t["finished"] = fin
    return {"schema": "pltir-episode-v1", "index": index, "question": question, "gold": "42", "group_id": "g", "rollout_index": 0,
            "model_version": 0, "turns": turns}


def test_fidelity_matches_a_consistent_episode_and_names_a_mismatch(tmp_path, monkeypatch):
    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call>'
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call>'
    ep = _episode_with(tok, "What is 6*7?", [call, ans], [["42"], ["Answer submitted: \\boxed{42}"]])
    rep = tt.fidelity_report([ep], tok, SYSTEM, content_eos="strip", max_model_len=4096)
    assert rep["passed"] and rep["n_turns"] == 2 and rep["turns_matched"] == 2 and rep["turns_mismatched"] == 0
    assert rep["per_episode"][0]["index"] == 5 and rep["per_episode"][0]["turns"][1]["match"] is True
    assert 0 <= rep["turns_extend_cleanly"] <= 2
    bad = json.loads(json.dumps(ep))
    bad["turns"][1]["input_ids"][3] += 1
    rep = tt.fidelity_report([bad], tok, SYSTEM, content_eos="strip", max_model_len=4096)
    assert not rep["passed"] and rep["turns_mismatched"] == 1 and rep["per_episode"][0]["turns"][1]["first_divergence"] == 3
    # the CLI: exit 3 on the mismatch, 0 on the match, the report written either way
    epf = tmp_path / "eps.jsonl"
    epf.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    monkeypatch.setattr(tt, "_tokenizer_for", lambda spec: tok)
    assert tt.main(["fidelity", "--episodes", str(epf), "--tokenizer", "local:x", "--system-prompt", str(_sys_file(tmp_path)),
                    "--content-eos", "strip", "--max-model-len", "4096", "--out", str(tmp_path / "fid")]) == 3
    assert json.loads((tmp_path / "fid" / "fidelity.json").read_text(encoding="utf-8"))["passed"] is False
    epf.write_text(json.dumps(ep) + "\n", encoding="utf-8")
    assert tt.main(["fidelity", "--episodes", str(epf), "--tokenizer", "local:x", "--system-prompt", str(_sys_file(tmp_path)),
                    "--content-eos", "strip", "--max-model-len", "4096", "--out", str(tmp_path / "fid")]) == 0


def _sys_file(tmp_path):
    p = tmp_path / "system.txt"
    p.write_text(SYSTEM, encoding="utf-8", newline="\n")
    return p


# --- the arm files: ASCII on write, newline-only on a resume -------------------------------------------------------

TOOL_BREAKS = "tool output \u2028 a \u2029 b \u0085 caf\u00e9"   # str.splitlines() breaks on the first three


class _FakeTensor:
    """A cache layer as cells handles it: numpy underneath, with the slicing and the .float()/.cpu()/.numpy() calls
    cells and task_swap's float16 rounding make on a torch tensor."""

    def __init__(self, a):
        self.a = __import__("numpy").asarray(a, dtype="float32")

    def __getitem__(self, s):
        return _FakeTensor(self.a[s])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_cells_writes_ascii_so_unicode_line_breaks_keep_one_record_per_line_and_resume_reads_them_back(tmp_path, monkeypatch, capsys):
    """A sandbox result can hold U+2028, U+2029 or U+0085 (and any non-ASCII text), which str.splitlines() breaks on; a
    model that quotes the tool's output carries them into completion_texts. cells writes the arm files ASCII (json
    escapes them), one newline per record and no CR, and a resume reads them back by splitting on the newline character
    only, so a file left with raw breaks by another writer still resumes. The run is cells end to end over stand-ins
    for torch, the scorer, the tokenizer and the engine; the sandbox is the real client over a faked POST."""
    import sys
    import types

    import numpy as np

    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(1)"}}\n</tool_call>'

    class EchoEngine:
        """Turn 1 calls the sandbox; turn 2 quotes the tool's response back and answers, as a model would."""
        generated = 0

        def __init__(self, model, tok_, device):
            self.tok, self.cache = tok_, []

        def reset(self, layers, ids):
            assert all(k.a.shape[0] == len(ids) for k, _ in layers)      # the injected rows are the ids the cache covers
            self.cache = list(ids)

        def cache_tokens(self):
            return list(self.cache)

        def truncate(self, n):
            self.cache = self.cache[:n]

        def prefill(self, ids):
            self.cache.extend(ids)

        def generate(self, max_new):
            type(self).generated += 1
            seen = self.tok.decode(self.cache)
            if "<tool_response>" not in seen:
                text = call
            else:
                quoted = seen.rsplit("<tool_response>\n", 1)[1].split("\n</tool_response>", 1)[0]
                text = "The tool printed " + quoted + " so #### 42"
            ids = self.tok(text)["input_ids"][:max_new] + [self.tok.eos_token_id]
            self.cache.extend(ids)
            return ids, text

    fake_torch = types.ModuleType("torch")
    fake_torch.float32, fake_torch.int64 = "float32", "int64"
    fake_torch.tensor = lambda x, dtype=None: [int(t) for t in x]
    fake_torch.from_numpy = _FakeTensor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_sc = types.SimpleNamespace(
        pick_device=lambda name: "cpu",
        load_reader=lambda spec, dtype, device: spec,
        live_prefix_layers=lambda model, ids, rows, device: [(_FakeTensor(np.zeros((rows, 1, 2))), _FakeTensor(np.zeros((rows, 1, 2))))])
    monkeypatch.setattr(tt, "_sc", lambda: fake_sc)
    monkeypatch.setattr(tt, "_tokenizer_for", lambda spec: tok)
    monkeypatch.setattr(tt, "TorchEngine", EchoEngine)
    monkeypatch.setattr(tt, "_mean_kl_fresh_path", lambda *a: 0.25)
    monkeypatch.setattr(tt, "_post_json", lambda req, timeout: {"status": "Success", "message": "",
                                                                  "run_result": {"status": "Finished", "stdout": TOOL_BREAKS + "\n", "stderr": ""}})
    monkeypatch.setenv("SANDBOX_ENDPOINT", "http://127.0.0.1:8080")
    prompts, idx, far, out = tmp_path / "prompts.json", tmp_path / "indices.json", tmp_path / "far.npy", tmp_path / "cells"
    prompts.write_text(json.dumps([{"index": 3, "question": "What is 6*7?", "gold": "42"},
                                   {"index": 7, "question": "What is 40+2?", "gold": "42"}]), encoding="utf-8")
    idx.write_text(json.dumps([3, 7]), encoding="utf-8")
    np.save(far, np.arange(16, dtype=np.int64).reshape(2, 8))
    args = ["cells", "--reader", "local:r", "--writer", "local:w", "--prompts", str(prompts), "--indices", str(idx),
            "--far-tokens", str(far), "--far-seq", "1", "--max-new", "256", "--max-turns", "3", "--max-model-len", "8192",
            "--min-generation", "8", "--floor-n", "1", "--control-n", "1", "--system-prompt", str(_sys_file(tmp_path)),
            "--content-eos", "strip", "--out", str(out)]
    assert tt.main(args) == 0
    assert "CELLS_DONE 2 prompts" in capsys.readouterr().out
    expected = {arm: ([3, 7] if arm in tt.ARMS else [3]) for arm in tt.ALL_ARMS}
    assert len(expected) == 7 and EchoEngine.generated == 2 * sum(len(v) for v in expected.values())
    for arm, want in expected.items():
        raw = (out / f"{arm}.jsonl").read_bytes()
        assert raw.isascii() and raw.count(b"\n") == len(want) and b"\r" not in raw, arm
        recs = [json.loads(line) for line in raw.decode("ascii").split("\n") if line]
        assert [r["index"] for r in recs] == want and all(TOOL_BREAKS in r["completion_texts"][1] for r in recs), arm
        assert all(r["answer"] == "42" and r["n_turns"] == 2 for r in recs), arm
    # a resume over that directory: done[arm] holds every index, so no episode runs again and no byte is appended
    before = {arm: (out / f"{arm}.jsonl").read_bytes() for arm in tt.ALL_ARMS}
    ran = EchoEngine.generated
    assert tt.main(args) == 0 and EchoEngine.generated == ran
    assert {arm: (out / f"{arm}.jsonl").read_bytes() for arm in tt.ALL_ARMS} == before
    # the same records left raw by another writer: the breaks are live, and the resume still reads every record back
    for arm, want in expected.items():
        p = out / f"{arm}.jsonl"
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").split("\n") if line]
        p.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8"))
        assert len(p.read_text(encoding="utf-8").splitlines()) > len(want), arm
    assert tt.main(args) == 0 and EchoEngine.generated == ran


def test_fidelity_fails_when_upstream_went_on_past_a_turn_where_the_port_parsed_no_tool_call():
    # upstream recording a later turn means its loop continued. If this port parses no tool call
    # in the turn before, that is a divergence, and every turn after it must count as mismatched, not be skipped.
    # Three turns, so "each later turn" is pinned and not only the first
    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call>'
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call>'
    ep = _episode_with(tok, "What is 6*7?", [call, call, ans], [["42"], ["42"], ["Answer submitted: \\boxed{42}"]])
    assert len(ep["turns"]) == 3
    bad = json.loads(json.dumps(ep))
    pl = bad["turns"][0]["prompt_tokens"]
    bad["turns"][0]["input_ids"] = bad["turns"][0]["input_ids"][:pl] + tok("I will think about it.")["input_ids"] + [tok.eos_token_id]
    rep = tt.fidelity_report([bad], tok, SYSTEM, content_eos="strip", max_model_len=4096)
    assert not rep["passed"] and rep["n_turns"] == 3 and rep["turns_matched"] == 1 and rep["turns_mismatched"] == 2
    rows = rep["per_episode"][0]["turns"]
    assert [r["step_index"] for r in rows] == [0, 1, 2] and rows[0]["match"] is True
    for later, t in zip(rows[1:], bad["turns"][1:]):                  # each later turn, not only the first
        assert later["match"] is False and later["first_divergence"] is None and later["ours_len"] is None
        assert later["prompt_len"] == t["prompt_tokens"]
    # an episode that really ends on a turn without a tool call is still one matched turn
    last = _episode_with(tok, "What is 6*7?", ["The answer is 42."], [[]])
    rep = tt.fidelity_report([last], tok, SYSTEM, content_eos="strip", max_model_len=4096)
    assert rep["passed"] and rep["n_turns"] == 1 and rep["turns_mismatched"] == 0


def test_fidelity_fails_when_the_last_turn_disagrees_with_upstreams_finished():
    # upstream's `finished` says whether the answer was submitted. At the last recorded turn there
    # is no later prompt to render, so a divergence in how the episode ends shows only here
    tok = FakeTok()
    call = '<tool_call>\n{"name": "run_python_code", "arguments": {"code": "print(6*7)"}}\n</tool_call>'
    ans = '<tool_call>\n{"name": "MathAnswer", "arguments": {"answer": "\\\\boxed{42}"}}\n</tool_call>'
    fid = lambda e: tt.fidelity_report([e], tok, SYSTEM, content_eos="strip", max_model_len=4096)
    plain = _episode_with(tok, "What is 6*7?", [call, "The answer is 42."], [["42"], []])       # ends on a turn with no call
    code = _episode_with(tok, "What is 6*7?", [call, call], [["42"], ["42"]])                    # ends on a call that is no answer
    answer = _episode_with(tok, "What is 6*7?", [call, ans], [["42"], ["Answer submitted: \\boxed{42}"]])
    assert [e["turns"][-1]["finished"] for e in (plain, code, answer)] == [False, False, True]
    for e in (plain, code, answer):                                       # as upstream would have written them: all pass
        rep = fid(e)
        assert rep["passed"] and rep["n_turns"] == 2 and rep["turns_matched"] == 2
    for e in (plain, code, answer):                                       # upstream's finished says the opposite: each fails
        bad = json.loads(json.dumps(e))
        for t in bad["turns"]:
            t["finished"] = not t["finished"]
        rep = fid(bad)
        assert not rep["passed"] and rep["n_turns"] == 2 and rep["turns_matched"] == 1 and rep["turns_mismatched"] == 1
        first, last = rep["per_episode"][0]["turns"]
        assert first["match"] is True and last["match"] is False and last["first_divergence"] is None
        assert last["ours_len"] == last["prompt_len"]                     # rendered and equal: the divergence is the ending
    gone = json.loads(json.dumps(answer))                                 # a record without the field is refused, not guessed
    del gone["turns"][-1]["finished"]
    with pytest.raises(KeyError):
        fid(gone)
    both = json.loads(json.dumps(answer))                                 # mis-rendered and disagreeing: counted once
    for t in both["turns"]:
        t["finished"] = False
    both["turns"][-1]["input_ids"][0] += 1
    rep = fid(both)
    assert not rep["passed"] and rep["turns_matched"] == 1 and rep["turns_mismatched"] == 1
    assert rep["per_episode"][0]["turns"][-1]["first_divergence"] == 0
