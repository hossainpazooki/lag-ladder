"""tools/pipelinerl_stream.py (design 2026-11-09 §5.1, §6): the seeded frozen draw, the custom-dataset and reserved-prompt
files, the actor-stream parse (groups by rollout, turns by step_index, the first complete episode per frozen prompt kept),
and the ag-prompt token file with its spans. Synthetic stream lines; no model, no PipelineRL."""
import importlib.util
import json

import numpy as np
import pytest

from lag_ladder import REPO_ROOT
from lag_ladder.rng import make_rng

spec = importlib.util.spec_from_file_location("pipelinerl_stream", REPO_ROOT / "tools" / "pipelinerl_stream.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

PROMPTS = [{"index": i, "question": f"Question number {i}?", "gold": str(3 * i)} for i in range(20)]
MASK = -100


def _turn(question: str, prompt: list[int], gen: list[int], step: int, rollout: int, group: str, prior_blocks=()):
    """One TrainingText dict as upstream writes it: labels mask the prompt; text is the rendered chat."""
    text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n" + question + "<|im_end|>\n"
    for b in prior_blocks:
        text += "<|im_start|>assistant\n...<|im_end|>\n<|im_start|>user\n<tool_response>\n" + b + "\n</tool_response><|im_end|>\n"
    text += "<|im_start|>assistant\n" + " ".join(map(str, gen)) + "<|im_end|>\n"
    return {"input_ids": prompt + gen, "labels": [MASK] * len(prompt) + gen, "prompt_tokens": len(prompt),
            "output_tokens": len(gen), "finished": True, "text": text, "group_id": group, "reward": 0.0,
            "metadata": {"model_version": 0, "rollout_index": rollout, "step_index": step}}


def _episode(i: int, group: str, turns: int = 2, rollout: int = 0):
    q = PROMPTS[i]["question"]
    out, blocks = [], []
    for t in range(turns):
        prompt = list(range(100, 100 + 10 + 7 * t))
        gen = [500 + t, 501 + t, 502 + t]
        out.append(_turn(q, prompt, gen, t, rollout, group, tuple(blocks)))
        blocks.append(f"result {t}")
    return out


def test_draw_frozen_is_seeded_sorted_and_without_replacement():
    f = ps.draw_frozen(1319, 50, 0)
    assert len(f) == 50 == len(set(f)) and f == sorted(f) and all(0 <= i < 1319 for i in f)
    assert f == ps.draw_frozen(1319, 50, 0) != ps.draw_frozen(1319, 50, 1)
    assert f == sorted(int(i) for i in make_rng(0).choice(1319, size=50, replace=False))
    with pytest.raises(ValueError):
        ps.draw_frozen(10, 11, 0)


def test_custom_and_reserved_rows_partition_the_prompts():
    frozen = [1, 5, 7]
    custom = ps.custom_rows(PROMPTS, frozen, "pltir_frozen50")
    assert [c["id"] for c in custom] == frozen and custom[0] == {"id": 1, "task": "Question number 1?", "answer": "3", "dataset": "pltir_frozen50"}
    reserved = ps.reserved_rows(PROMPTS, frozen)
    assert len(reserved) == 17 and {r["index"] for r in reserved}.isdisjoint(frozen)
    assert [r["index"] for r in reserved] == sorted(r["index"] for r in reserved)


def test_question_of_reads_the_first_user_turn_and_turn_checks_use_the_mask_run():
    turns = _episode(3, "g3")
    assert ps.question_of(turns[0]["text"]) == "Question number 3?"
    assert ps.question_of(turns[1]["text"]) == "Question number 3?"      # not the tool-response user turn
    assert ps.question_of("no turns") is None
    assert ps.leading_run([MASK, MASK, 7, 8]) == 2 and ps.leading_run([]) == 0
    assert ps.check_turn(turns[0]) is None
    bad = dict(turns[0], prompt_tokens=3)
    assert "prompt_tokens" in ps.check_turn(bad)
    bad = dict(turns[0], labels=turns[0]["labels"][:-1])
    assert "length" in ps.check_turn(bad)
    assert "missing" in ps.check_turn({k: v for k, v in turns[0].items() if k != "text"})


def test_group_episodes_orders_turns_by_step_index_and_episode_from_names_the_reason():
    a = _episode(2, "ga", turns=3)
    b = _episode(4, "gb", turns=1, rollout=1)
    records = [a[2], b[0], a[0], a[1]]
    groups = ps.group_episodes(records)
    assert set(groups) == {("ga", 0), ("gb", 1)}
    assert [t["metadata"]["step_index"] for t in groups[("ga", 0)]] == [0, 1, 2]
    by_q = {p["question"]: p for p in PROMPTS if p["index"] in (2, 4)}
    ep, why = ps.episode_from(groups[("ga", 0)], by_q)
    assert why == "ok" and ep["schema"] == "pltir-episode-v1" and ep["index"] == 2 and ep["gold"] == "6"
    assert [t["step_index"] for t in ep["turns"]] == [0, 1, 2] and ep["turns"][1]["prompt_tokens"] == 17
    assert ps.episode_from(groups[("ga", 0)][1:], by_q) == (None, "incomplete")
    other = _episode(9, "go")                                           # not frozen
    assert ps.episode_from(other, by_q) == (None, "not a frozen prompt")


def test_collect_episodes_keeps_the_first_complete_episode_per_prompt_and_counts_the_rest(tmp_path):
    d = tmp_path / "streams" / "actor" / "0"
    (d / "0").mkdir(parents=True)
    (d / "1").mkdir(parents=True)
    lines0 = [json.dumps(_episode(2, "g2a")), json.dumps(_episode(4, "g4")), json.dumps(_episode(9, "g9"))]
    lines1 = [json.dumps(_episode(2, "g2b", turns=3)), json.dumps(_episode(6, "g6")[1:])]   # a duplicate; an incomplete one
    (d / "0" / "0.jsonl").write_text("\n".join(lines0) + "\n", encoding="utf-8")
    (d / "1" / "0.jsonl").write_text("\n".join(lines1) + "\n", encoding="utf-8")
    by_q = {p["question"]: p for p in PROMPTS if p["index"] in (2, 4, 6)}
    kept, counts = ps.collect_episodes(ps.stream_files(tmp_path / "streams" / "actor"), by_q)
    assert sorted(kept) == [2, 4] and kept[2]["group_id"] == "g2a" and len(kept[2]["turns"]) == 2
    assert counts == {"lines": 5, "groups": 5, "kept": 2, "duplicates": 1, "incomplete": 1, "not_frozen": 1, "bad_turn": 0,
                      "partial": 0}


def test_collect_episodes_skips_a_half_written_last_line_and_counts_it_as_partial(tmp_path):
    """The actor writes while `episodes` runs: upstream writes each group as one line, then the newline, then flushes
    (pipelinerl/streams.py:249-277 at the pin), so a shard's last line can be on disk in part, cut anywhere, inside a
    multi-byte character included. Such a line is counted as partial and skipped; a whole last line still waiting for its
    newline is read; a newline-terminated line that does not parse is corruption, not a write in progress."""
    root = tmp_path / "streams" / "actor" / "0"
    for p in ("a", "b", "c"):
        (root / p).mkdir(parents=True)
    whole = json.dumps(_episode(2, "g2"))
    (root / "a" / "0.jsonl").write_bytes((whole + "\n" + json.dumps(_episode(4, "g4"))).encode("utf-8"))  # newline pending
    cut = json.dumps(_episode(2, "g2-later"))[:40]                         # the last line cut at 40 characters
    (root / "b" / "0.jsonl").write_bytes((json.dumps(_episode(6, "g6")) + "\n" + cut).encode("utf-8"))
    wide = _episode(8, "g8")
    wide[0]["text"] += "café"
    raw = json.dumps(wide, ensure_ascii=False).encode("utf-8")             # upstream writes non-ASCII as raw UTF-8
    (root / "c" / "0.jsonl").write_bytes(raw[: raw.index("é".encode("utf-8")) + 1])   # cut inside a 2-byte character
    by_q = {p["question"]: p for p in PROMPTS if p["index"] in (2, 4, 6, 8)}
    kept, counts = ps.collect_episodes(ps.stream_files(tmp_path / "streams" / "actor"), by_q)
    assert sorted(kept) == [2, 4, 6] and kept[2]["group_id"] == "g2"
    assert counts == {"lines": 3, "groups": 3, "kept": 3, "duplicates": 0, "incomplete": 0, "not_frozen": 0, "bad_turn": 0,
                      "partial": 2}
    bad = tmp_path / "bad" / "0.jsonl"
    bad.parent.mkdir()
    bad.write_bytes((whole + "\n" + cut + "\n").encode("utf-8"))        # terminated, so not a write in progress
    with pytest.raises(ValueError, match=r"0\.jsonl:2"):
        ps.collect_episodes([bad], by_q)


def test_ag_prompt_rows_take_the_first_turn_and_pad_rows_writes_the_spans():
    eps = [ps.episode_from(_episode(i, f"g{i}"), {p["question"]: p for p in PROMPTS})[0] for i in (4, 2)]
    rows = ps.ag_prompt_rows(eps)
    assert [len(r[0]) for r in rows] == [13, 13] and [r[1] for r in rows] == [10, 10]
    arr, spans = ps.pad_rows(rows, 16, 0)
    assert arr.shape == (2, 16) and arr.dtype == np.int64 and spans == {"prefix_len": [10, 10], "length": [13, 13]}
    assert list(arr[0, :13]) == rows[0][0] and list(arr[0, 13:]) == [0, 0, 0]


def test_cli_prompts_writes_the_three_files_and_episodes_exits_3_until_covered(tmp_path, capsys):
    gsm = tmp_path / "gsm8k.json"
    gsm.write_text(json.dumps(PROMPTS), encoding="utf-8")
    frozen, custom, reserved = tmp_path / "frozen.json", tmp_path / "custom.jsonl", tmp_path / "reserved.json"
    rc = ps.main(["prompts", "--gsm8k", str(gsm), "--seed", "0", "--n", "3", "--expect-total", "20", "--dataset", "d",
                  "--frozen-out", str(frozen), "--custom-out", str(custom), "--reserved-out", str(reserved)])
    assert rc == 0
    fz = json.loads(frozen.read_text(encoding="utf-8"))
    assert fz["indices"] == ps.draw_frozen(20, 3, 0) and fz["seed"] == 0 and fz["n"] == 3 and fz["total"] == 20
    assert len(custom.read_text(encoding="utf-8").splitlines()) == 3 and len(json.loads(reserved.read_text(encoding="utf-8"))) == 17
    out = capsys.readouterr().out
    assert out.count("sha256=") == 3 and out.isascii()
    with pytest.raises(SystemExit):
        ps.main(["prompts", "--gsm8k", str(gsm), "--seed", "0", "--n", "3", "--expect-total", "1319", "--dataset", "d",
                 "--frozen-out", str(frozen), "--custom-out", str(custom), "--reserved-out", str(reserved)])
    # episodes: one of the three frozen prompts missing -> exit 3; all three -> exit 0 and the file is sorted by index
    d = tmp_path / "run" / "streams" / "actor" / "0" / "0"
    d.mkdir(parents=True)
    idx = fz["indices"]
    d.joinpath("0.jsonl").write_text("\n".join(json.dumps(_episode(i, f"g{i}")) for i in idx[:2]) + "\n", encoding="utf-8")
    epf = tmp_path / "episodes.jsonl"
    assert ps.main(["episodes", "--streams", str(tmp_path / "run" / "streams" / "actor"), "--gsm8k", str(gsm),
                    "--frozen", str(frozen), "--out", str(epf)]) == 3
    with open(d / "0.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_episode(idx[2], "glast")) + "\n")
    assert ps.main(["episodes", "--streams", str(tmp_path / "run" / "streams" / "actor"), "--gsm8k", str(gsm),
                    "--frozen", str(frozen), "--out", str(epf)]) == 0
    eps = [json.loads(l) for l in epf.read_text(encoding="utf-8").splitlines()]
    assert [e["index"] for e in eps] == idx
    tok, sp = tmp_path / "ag.npy", tmp_path / "ag.spans.json"
    assert ps.main(["ag-prompt", "--episodes", str(epf), "--row-len", "32", "--pad-id", "0", "--tokens-out", str(tok), "--spans-out", str(sp)]) == 0
    assert np.load(tok).shape == (3, 32) and json.loads(sp.read_text(encoding="utf-8"))["prefix_len"] == [10, 10, 10]


def test_cli_episodes_exits_3_on_a_half_written_last_line_and_reports_it(tmp_path, capsys):
    """The operator loops while the exit code is 3: a line still being written must keep the loop going, not stop it."""
    gsm = tmp_path / "gsm8k.json"
    gsm.write_text(json.dumps(PROMPTS), encoding="utf-8")
    idx = ps.draw_frozen(20, 3, 0)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"seed": 0, "n": 3, "total": 20, "indices": idx}), encoding="utf-8")
    shard = tmp_path / "streams" / "actor" / "0" / "0" / "0.jsonl"
    shard.parent.mkdir(parents=True)
    last = json.dumps(_episode(idx[2], "glast"))
    shard.write_bytes(("\n".join(json.dumps(_episode(i, f"g{i}")) for i in idx[:2]) + "\n" + last[:40]).encode("utf-8"))
    args = ["episodes", "--streams", str(tmp_path / "streams" / "actor"), "--gsm8k", str(gsm), "--frozen", str(frozen),
            "--out", str(tmp_path / "episodes.jsonl")]
    assert ps.main(args) == 3
    out = capsys.readouterr().out
    assert "kept 2/3 lines 2 " in out and f"partial 1 missing [{idx[2]}]" in out and out.isascii()
    with open(shard, "ab") as f:
        f.write((last[40:] + "\n").encode("utf-8"))                          # the rest of the line, then its newline
    assert ps.main(args) == 0
    out = capsys.readouterr().out
    assert "kept 3/3 lines 3 " in out and "partial 0 missing []" in out


UNICODE_BREAKS = "tool output \u2028 a \u2029 b \u0085 c café"   # str.splitlines() breaks on the first three


def test_cli_episodes_writes_ascii_so_unicode_line_breaks_in_turn_text_keep_one_record_per_line(tmp_path, capsys):
    """Upstream writes stream lines with orjson, which leaves non-ASCII raw (pipelinerl/streams.py:275-276 at the pin), and
    model or sandbox text can hold U+2028, U+2029 or U+0085. The episodes file is read back with str.splitlines() (here,
    and by the off-pod fidelity gate that consumes it), which breaks on all three, so the file is written ASCII: json escapes them."""
    gsm = tmp_path / "gsm8k.json"
    gsm.write_text(json.dumps(PROMPTS), encoding="utf-8")
    idx = ps.draw_frozen(20, 3, 0)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"seed": 0, "n": 3, "total": 20, "indices": idx}), encoding="utf-8")
    eps = [_episode(i, f"g{i}") for i in idx]
    eps[0][1]["text"] += UNICODE_BREAKS                                   # a later turn's text, as a tool result
    shard = tmp_path / "streams" / "actor" / "0" / "0" / "0.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(("\n".join(json.dumps(e, ensure_ascii=False) for e in eps) + "\n").encode("utf-8"))  # raw, as orjson
    epf = tmp_path / "episodes.jsonl"
    assert ps.main(["episodes", "--streams", str(tmp_path / "streams" / "actor"), "--gsm8k", str(gsm), "--frozen", str(frozen),
                    "--out", str(epf)]) == 0
    assert "kept 3/3 lines 3 " in capsys.readouterr().out
    raw = epf.read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 3
    lines = epf.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["turns"][1]["text"].endswith(UNICODE_BREAKS)   # the text round-trips
    tok, sp = tmp_path / "ag.npy", tmp_path / "ag.spans.json"
    assert ps.main(["ag-prompt", "--episodes", str(epf), "--row-len", "32", "--pad-id", "0", "--tokens-out", str(tok),
                    "--spans-out", str(sp)]) == 0
    assert np.load(tok).shape == (3, 32)


def test_ag_prompt_splits_the_episodes_file_on_newline_only(tmp_path):
    """An episodes file left non-ASCII by another writer: a raw U+2028, U+2029 or U+0085 sits inside a JSON string and is
    not a record boundary, so ag-prompt splits on newline only."""
    eps = [ps.episode_from(_episode(i, f"g{i}"), {p["question"]: p for p in PROMPTS})[0] for i in (2, 4)]
    eps[0]["turns"][0]["text"] += UNICODE_BREAKS
    epf = tmp_path / "episodes.jsonl"
    epf.write_bytes("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps).encode("utf-8"))
    assert len(epf.read_text(encoding="utf-8").splitlines()) == 5                 # the planted breaks are live
    tok, sp = tmp_path / "ag.npy", tmp_path / "ag.spans.json"
    assert ps.main(["ag-prompt", "--episodes", str(epf), "--row-len", "16", "--pad-id", "0", "--tokens-out", str(tok),
                    "--spans-out", str(sp)]) == 0
    assert np.load(tok).shape == (2, 16) and json.loads(sp.read_text(encoding="utf-8"))["prefix_len"] == [10, 10]
