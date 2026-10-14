"""tools/own_run.py's offline parts (design §14): the keep set, the padded token file with its spans sidecar,
GSM8K answer extraction and rewards, the seeded prompt selection, and the dump/distance commands the save hook
issues. The trainer itself needs a GPU and TRL and is not exercised here."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from lag_ladder import REPO_ROOT

spec = importlib.util.spec_from_file_location("own_run", REPO_ROOT / "tools" / "own_run.py")
own_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(own_run)


def test_keep_set_default_and_parsing():
    assert own_run.keep_set(None) == frozenset({0, 1, 2, 4, 8, 16, 32})
    assert own_run.keep_set("0, 3,3,5") == frozenset({0, 3, 5})
    with pytest.raises(ValueError):
        own_run.keep_set("0,-1")


def test_pad_rows_writes_spans_and_refuses_bad_rows():
    arr, spans = own_run.pad_rows([([5, 6, 7, 8, 9], 2), ([1, 2, 3], 1)], row_len=8, pad_id=0)
    assert arr.shape == (2, 8) and arr.dtype == np.int64
    assert arr[0].tolist() == [5, 6, 7, 8, 9, 0, 0, 0] and arr[1].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    assert spans == {"prefix_len": [2, 1], "length": [5, 3]}
    # the scorer's rule: prefix_len + 1 < length, so at least one scored token after the fed prefix token
    with pytest.raises(ValueError):
        own_run.pad_rows([([1, 2, 3], 2)], row_len=8, pad_id=0)
    with pytest.raises(ValueError):
        own_run.pad_rows([(list(range(9)), 2)], row_len=8, pad_id=0)


def test_gsm8k_gold_and_final_number():
    assert own_run.gsm8k_gold("Janet has 3 eggs.\n#### 1,234") == "1234"
    assert own_run.final_number("so the answer is 17 dollars. #### 17") == "17"
    assert own_run.final_number("we get 3 then 42.") == "42"
    assert own_run.final_number("no digits here") is None
    with pytest.raises(ValueError):
        own_run.gsm8k_gold("no marker")


def test_rewards():
    assert own_run.reward_exact("steps... #### 1234", "1234") == 1.0
    assert own_run.reward_exact("steps... #### 1,234.0", "1234") == 1.0
    assert own_run.reward_exact("steps... 1233", "1234") == 0.0
    assert own_run.reward_format("#### 5") == 0.2 and own_run.reward_format("5") == 0.0


def test_prompt_selection_is_seeded_and_distinct():
    a = own_run.select_prompt_indices(7473, 10, 0)
    assert a == own_run.select_prompt_indices(7473, 10, 0) and len(set(a)) == 10 == len(a)
    assert a != own_run.select_prompt_indices(7473, 10, 1)
    assert all(0 <= i < 7473 for i in a) and a == sorted(a)


def test_commands_name_the_instrument_pair_and_float32(tmp_path):
    cmd = own_run.dump_command(Path("py"), Path("kvt"), Path("ckpt"), Path("seq40.npy"), Path("out"))
    assert cmd[1:3] == ["-m", "scripts.dump_kv"] and "--local-path" in cmd and "--stride" in cmd
    assert cmd[cmd.index("--pair") + 1] == own_run.KVT_PAIR and cmd[cmd.index("--dtype") + 1] == "float32"
    d = own_run.distance_command(Path("py"), Path("ll"), Path("w"), Path("r"), Path("o"))
    assert d[1].endswith("weight_distance.py") and d[d.index("--writer") + 1] == "w"


def test_per_seq_token_files_match_rows(tmp_path):
    arr = np.arange(40, dtype=np.int64).reshape(5, 8)
    f = tmp_path / "tok.npy"
    np.save(f, arr)
    paths = own_run.write_per_seq_tokens(f, tmp_path / "per", range(2, 5))
    assert [p.name for p in paths] == ["seq2.npy", "seq3.npy", "seq4.npy"]
    assert np.array_equal(np.load(paths[1]), arr[3:4])


def test_sha256_dir_safetensors_is_order_independent_of_creation(tmp_path):
    (tmp_path / "b.safetensors").write_bytes(b"2")
    (tmp_path / "a.safetensors").write_bytes(b"1")
    (tmp_path / "config.json").write_text("{}")
    h1 = own_run.sha256_dir_safetensors(tmp_path)
    (tmp_path / "config.json").write_text('{"x": 1}')
    assert own_run.sha256_dir_safetensors(tmp_path) == h1   # only the weights are hashed


def test_labels_by_step_reads_the_config_rungs():
    labels = own_run.labels_by_step(REPO_ROOT / "config" / "ownrun.toml")
    assert sorted(labels) == [1, 2, 4, 8, 16, 32]
    assert labels[32] == ["ownrun-lag-32", "ownrun-rollout-lag-32"]
