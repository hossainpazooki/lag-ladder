"""tools/task_swap.py's offline parts (design §15, statistic (C)): the NEAR pairing, the float16 rounding of a live
cache, the dump-equivalence comparison, the answer extraction it borrows from own_run, first divergence, and the record
schema. Anything that needs a model runs on the pod and is not exercised here."""
import importlib.util
import json

import numpy as np
import pytest

from lag_ladder import REPO_ROOT

spec = importlib.util.spec_from_file_location("task_swap", REPO_ROOT / "tools" / "task_swap.py")
task_swap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_swap)


def test_near_pairing_takes_the_next_longer_prompt_and_wraps_the_longest_to_the_shortest():
    # lengths by index: 0:10, 1:30, 2:20, 3:30 -> ascending order (ties by index): 0(10), 2(20), 1(30), 3(30)
    pair = task_swap.near_pairing([10, 30, 20, 30])
    assert pair == {0: 2, 2: 1, 1: 3, 3: 0}
    # every source is at least as long as its target (so the cut to the target's length is a true cut), except the
    # wrap: the longest takes the shortest's and is padded by the caller
    lengths = [10, 30, 20, 30]
    for i, j in pair.items():
        assert lengths[j] >= lengths[i] or i == 3
    assert task_swap.near_pairing([7]) == {0: 0}


def test_through_float16_rounds_like_the_dump_path():
    x = np.array([1.0000001, 65519.0, 1e-9, -3.14159265], dtype=np.float32)
    y = task_swap.through_float16(x)
    assert y.dtype == np.float32
    assert np.array_equal(y, x.astype(np.float16).astype(np.float32))


def test_compare_layers_reports_the_largest_gap_after_rounding_and_zero_on_equal_caches():
    k = np.random.default_rng(0).normal(size=(5, 2, 4)).astype(np.float32)
    v = np.random.default_rng(1).normal(size=(5, 2, 4)).astype(np.float32)
    live = [(k, v), (k * 2, v * 2)]
    dump = [(k.astype(np.float16), v.astype(np.float16)), ((k * 2).astype(np.float16), (v * 2).astype(np.float16))]
    r = task_swap.compare_layers(live, dump)
    assert r["max_abs_diff"] == 0.0 and r["layers"] == 2 and r["rows"] == 5
    dump[1] = (dump[1][0].copy(), dump[1][1].copy())
    dump[1][1][3, 1, 2] += np.float16(0.5)
    r = task_swap.compare_layers(live, dump)
    assert r["max_abs_diff"] == pytest.approx(0.5, rel=1e-2) and r["worst"] == {"layer": 1, "side": "V"}
    with pytest.raises(ValueError, match="layers"):
        task_swap.compare_layers(live, dump[:1])


def test_answer_extraction_is_own_runs():
    assert task_swap.final_number("so #### 42") == "42" and task_swap.final_number("x = 1,234.5 then 7") == "7"
    assert task_swap.final_number("no digits") is None
    assert task_swap.gsm8k_gold("... #### 1,000") == "1000"
    assert task_swap.is_correct(task_swap.final_number("#### 18"), "18") and not task_swap.is_correct(None, "18")


def test_first_divergence_is_the_first_differing_position_or_none():
    assert task_swap.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert task_swap.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert task_swap.first_divergence([1, 2], [1, 2, 3]) == 2


def test_record_has_the_contract_keys_and_marks_the_cap_and_a_missing_answer():
    rec = task_swap.cell_record(index=7, prompt_len=80, completion=[5, 6, 7], max_new=3, eos_id=99, text="no number here",
                                gold="4", fresh_completion=[5, 6, 8])
    assert set(rec) == task_swap.RECORD_KEYS
    assert rec["index"] == 7 and rec["prompt_len"] == 80 and rec["answer"] is None and rec["correct"] is False
    assert rec["reached_cap"] is True and rec["first_divergence"] == 2 and rec["n_tokens"] == 3
    assert rec["completion_sha256"] == task_swap.tokens_sha256([5, 6, 7])
    rec = task_swap.cell_record(index=1, prompt_len=5, completion=[1, 99], max_new=3, eos_id=99, text="#### 4", gold="4",
                                fresh_completion=None)
    assert rec["reached_cap"] is False and rec["correct"] is True and rec["first_divergence"] is None
    json.dumps(rec)
