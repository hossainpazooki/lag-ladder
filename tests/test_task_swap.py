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


def test_compare_layers_reports_a_relative_gap_of_one_ulp_regardless_of_magnitude_with_a_floor_near_zero():
    # One float16 ulp at |x| in [16, 32) is 2^-6 absolute but <= 2^-10 relative; the same ulp at [8, 16) is 2^-7.
    # An absolute limit calibrated on one binade false-alarms on the next; the relative figure does not.
    big = np.full((2, 1, 2), 20.0, dtype=np.float32)
    live = [(big, big.copy())]
    dump = [(big.astype(np.float16), big.astype(np.float16))]
    dump[0][0][0, 0, 0] = np.nextafter(np.float16(20.0), np.float16(32.0))
    r = task_swap.compare_layers(live, dump)
    assert r["max_abs_diff"] == pytest.approx(2 ** -6)
    assert r["max_rel_diff"] <= 2 ** -10 and r["max_rel_diff"] > 0
    assert r["worst"] == {"layer": 0, "side": "K"}
    # A gap at an element that is zero on one side is bounded by the absolute floor (float16 min normal, 2^-14),
    # not reported as a relative gap of 1.0.
    small = np.zeros((1, 1, 1), dtype=np.float32)
    live = [(small, small)]
    dump = [(np.array([[[2 ** -24]]], dtype=np.float16), small.astype(np.float16))]
    r = task_swap.compare_layers(live, dump)
    assert r["max_rel_diff"] == pytest.approx(2 ** -24 / (2 ** -24 + 2 ** -14))
    assert r["max_rel_diff"] < 1e-3


def test_answer_extraction_is_own_runs():
    assert task_swap.final_number("so #### 42") == "42" and task_swap.final_number("x = 1,234.5 then 7") == "7"
    assert task_swap.final_number("no digits") is None
    assert task_swap.gsm8k_gold("... #### 1,000") == "1000"
    assert task_swap.is_correct(task_swap.final_number("#### 18"), "18") and not task_swap.is_correct(None, "18")


def test_first_divergence_is_the_first_differing_position_or_none():
    assert task_swap.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert task_swap.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert task_swap.first_divergence([1, 2], [1, 2, 3]) == 2


def test_first_hash_number_is_the_number_after_the_first_marker_or_none():
    assert task_swap.first_hash_number("so #### 3 ... later #### 5") == "3"
    assert task_swap.first_hash_number("#### 1,234.5 and #### 7") == "1234.5"
    assert task_swap.first_hash_number("the answer is 42") is None
    # the registered rule reads the LAST marker; the first-marker reading is kept beside it for the loop diagnosis
    assert task_swap.final_number("so #### 3 ... later #### 5") == "5"


def test_record_has_the_contract_keys_and_marks_the_cap_and_a_missing_answer():
    rec = task_swap.cell_record(index=7, prompt_len=80, completion=[5, 6, 7], max_new=3, eos_id=99, text="no number here",
                                gold="4", fresh_completion=[5, 6, 8])
    assert set(rec) == task_swap.RECORD_KEYS and rec["schema"] == "task-record-v2"
    assert rec["completion_text"] == "no number here" and rec["answer_first_hash"] is None
    assert rec["index"] == 7 and rec["prompt_len"] == 80 and rec["answer"] is None and rec["correct"] is False
    assert rec["reached_cap"] is True and rec["first_divergence"] == 2 and rec["n_tokens"] == 3
    assert rec["completion_sha256"] == task_swap.tokens_sha256([5, 6, 7])
    rec = task_swap.cell_record(index=1, prompt_len=5, completion=[1, 99], max_new=3, eos_id=99, text="#### 4", gold="4",
                                fresh_completion=None)
    assert rec["reached_cap"] is False and rec["correct"] is True and rec["first_divergence"] is None
    assert rec["completion_text"] == "#### 4" and rec["answer_first_hash"] == "4"
    rec = task_swap.cell_record(index=2, prompt_len=5, completion=[1, 2, 3], max_new=3, eos_id=99,
                                text="#### 4 wait #### 6", gold="4", fresh_completion=None)
    assert rec["answer"] == "6" and rec["correct"] is False and rec["answer_first_hash"] == "4"
    json.dumps(rec)


def test_greedy_and_fresh_path_kl_run_without_autograd():
    """1024-token generation OOMed on a 44 GB card (2026-10-15): the forwards kept their autograd graphs across steps."""
    torch = pytest.importorskip("torch", reason="the tool runs with the instrument's interpreter; torch is not pinned here")
    import types
    seen = []

    class Out:
        def __init__(self, cache):
            self.past_key_values = cache
            self.logits = torch.zeros(1, 1, 5)

    def model(**kw):
        seen.append(torch.is_grad_enabled())
        return Out(kw["past_key_values"])

    fake_sc = types.SimpleNamespace(build_cache=lambda layers, device: "cache",
                                    forward_logits=lambda m, layers, inp, past, dev: (seen.append(torch.is_grad_enabled()),
                                                                                      torch.zeros(1, inp.shape[1], 5))[1])
    task_swap._sc = lambda: fake_sc
    toks = task_swap._greedy(model, [], seed_token=1, past_len=3, max_new=4, eos_id=None, device="cpu")
    kl = task_swap._mean_kl_fresh_path(model, [], [], seed_token=1, past_len=3, cont=[2, 3], device="cpu")
    assert len(toks) == 4 and kl == 0.0
    assert seen and not any(seen), "every forward must run under torch.no_grad()"
