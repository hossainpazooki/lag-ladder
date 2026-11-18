"""tools/cut_spans.py (ledger 0042, the prefix cut): the held-out spans sidecars for the cut prefixes."""
import hashlib
import importlib.util
import json

import pytest

from lag_ladder import REPO_ROOT

spec = importlib.util.spec_from_file_location("cut_spans", REPO_ROOT / "tools" / "cut_spans.py")
cut_spans = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cut_spans)

ROLLOUT = {"prefix_len": [10, 7, 12], "length": [30, 27, 20]}     # completions of 20, 20 and 8 tokens


def test_prompt_cut_takes_each_rows_prompt_and_completion_lengths():
    s = cut_spans.cut_spans(ROLLOUT, "prompt", (5, 8), 10, 64)
    assert s["prefix_len"] == [None] * 5 + [10, 7, 12] + [None] * 2
    assert s["length"] == [None] * 5 + [30, 27, 20] + [None] * 2


def test_fixed_cut_keeps_the_completion_lengths():
    s = cut_spans.cut_spans(ROLLOUT, "24", (5, 8), 10, 64)
    assert s["prefix_len"][5:8] == [24, 24, 24]
    assert [e - p for p, e in zip(s["prefix_len"][5:8], s["length"][5:8])] == [20, 20, 8]
    assert s["prefix_len"][:5] == [None] * 5 and s["length"][8:] == [None, None]


@pytest.mark.parametrize("cut, heldout, rows, seq_len, msg", [
    ("50", (5, 8), 10, 64, "> row length 64"),
    ("prompt", (5, 7), 10, 64, "has 2 sequences"),
    ("prompt", (8, 11), 10, 64, "is not inside 10 rows"),
    ("1", (5, 8), 10, 64, "cut must be"),
    ("x", (5, 8), 10, 64, "cut must be"),
])
def test_refuses_a_cut_that_does_not_fit(cut, heldout, rows, seq_len, msg):
    with pytest.raises(ValueError, match=msg):
        cut_spans.cut_spans(ROLLOUT, cut, heldout, rows, seq_len)


def test_refuses_a_one_token_completion():
    with pytest.raises(ValueError, match="completion of 1 tokens"):
        cut_spans.cut_spans({"prefix_len": [5], "length": [6]}, "prompt", (0, 1), 1, 64)


def _write_rollout(tmp_path):
    src = tmp_path / "r.spans.json"
    src.write_bytes(json.dumps({**ROLLOUT, "model": "x"}).encode("utf-8"))
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def _cli(src, sha, out, cut="prompt"):
    return cut_spans.main(["--rollout-spans", str(src), "--rollout-spans-sha256", sha, "--cut", cut,
                           "--heldout", "5", "8", "--rows", "10", "--seq-len", "64", "--out", str(out)])


def test_cli_writes_the_same_bytes_twice_and_records_its_source(tmp_path):
    src, sha = _write_rollout(tmp_path)
    assert _cli(src, sha, tmp_path / "a.json") == 0 and _cli(src, sha, tmp_path / "b.json") == 0
    a = (tmp_path / "a.json").read_bytes()
    assert a == (tmp_path / "b.json").read_bytes() and b"\r" not in a
    rec = json.loads(a)
    assert rec["rollout_spans_sha256"] == sha and rec["cut"] == "prompt" and rec["heldout"] == [5, 8]


def test_cli_refuses_a_rollout_file_off_its_pin_and_an_existing_output(tmp_path):
    src, sha = _write_rollout(tmp_path)
    with pytest.raises(SystemExit, match="not the registered"):
        _cli(src, "0" * 64, tmp_path / "a.json")
    (tmp_path / "a.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _cli(src, sha, tmp_path / "a.json")
