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


def test_dump_command_takes_the_instrument_pair():
    cmd = own_run.dump_command(Path("py"), Path("kvt"), Path("ckpt"), Path("s.npy"), Path("o"), kvt_pair="qwen3-0.6b-to-1.7b")
    assert cmd[cmd.index("--pair") + 1] == "qwen3-0.6b-to-1.7b"


# --- §14.5: the prompt format (chat for templated models, plain for template-less bases) --------------------------

class _Tok:
    """A tokenizer stand-in: `chat_template` present or not, and a recognisable rendering."""
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "CHAT:" + msgs[-1]["content"]


def test_render_prompt_plain_is_the_registered_string_and_keeps_special_tokens():
    text, special = own_run.render_prompt(_Tok(None), "What is 2+2?", "plain")
    assert text == f"{own_run.SYSTEM_LINE}\n\nQuestion: What is 2+2?\nAnswer:"
    # TRL 1.15.0 tokenizes a string prompt with processing_class(text=prompts), add_special_tokens left at its
    # default (grpo_trainer.py `_tokenize_prompts`, line 1716), so the rollout prefix must carry the same BOS
    assert special is True


def test_render_prompt_chat_uses_the_template_and_fails_loudly_without_one():
    text, special = own_run.render_prompt(_Tok("{{ messages }}"), "What is 2+2?", "chat")
    assert text == "CHAT:What is 2+2?" and special is False
    with pytest.raises(SystemExit, match="meta-llama/Llama-3.2-3B@13afe.* has no chat template"):
        own_run.render_prompt(_Tok(None), "q", "chat", model="meta-llama/Llama-3.2-3B@13afe5124825")
    with pytest.raises(ValueError, match="prompt format"):
        own_run.render_prompt(_Tok(None), "q", "markdown")


def test_train_prompt_is_role_dicts_under_chat_and_the_plain_string_under_plain():
    assert own_run.train_prompt("q", "chat") == [{"role": "system", "content": own_run.SYSTEM_LINE},
                                                 {"role": "user", "content": "q"}]
    assert own_run.train_prompt("q", "plain") == f"{own_run.SYSTEM_LINE}\n\nQuestion: q\nAnswer:"


def test_count_stats_summarises_prompt_token_counts():
    assert own_run.count_stats([5, 3, 4]) == {"n": 3, "min": 3, "median": 4, "max": 5}
    assert own_run.count_stats([]) == {"n": 0, "min": None, "median": None, "max": None}


def test_both_commands_take_prompt_format_with_chat_as_the_default():
    ap = own_run.build_parser()
    common_s = ["sample-rollouts", "--model", "m", "--tokens-out", "t", "--spans-out", "s", "--excluded-out", "e"]
    common_t = ["train", "--model", "m", "--lag-ladder", "l", "--kvt", "k", "--kvt-python", "p", "--heldout-tokens", "h",
                "--rollout-tokens", "r", "--excluded-indices", "x", "--config", "c", "--out", "o"]
    assert ap.parse_args(common_s).prompt_format == "chat"
    assert ap.parse_args(common_s + ["--prompt-format", "plain"]).prompt_format == "plain"
    assert ap.parse_args(common_t).prompt_format == "chat"
    assert ap.parse_args(common_t + ["--prompt-format", "plain"]).prompt_format == "plain"
    with pytest.raises(SystemExit):
        ap.parse_args(common_t + ["--prompt-format", "markdown"])


class _EosTok:
    """A tokenizer that only knows its eos and pad ids (the decoding spec reads nothing else)."""
    def __init__(self, eos, pad=None):
        self.eos_token_id, self.pad_token_id = eos, pad


def test_decoding_spec_pins_every_generation_config_parameter_and_the_sampler_passes_it():
    """2026-10-24: `generate` fills what the call omits from the checkpoint's generation_config.json (Qwen2.5-1.5B-
    Instruct ships repetition_penalty 1.1 there), so the sampler passes the whole spec and records it."""
    s = own_run.decoding_spec(_EosTok(7, None), do_sample=True, temperature=1.0, max_new_tokens=256)
    assert s == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                 "no_repeat_ngram_size": 0, "num_beams": 1, "eos_token_id": 7, "pad_token_id": 7,
                 "max_new_tokens": 256}
    g = own_run.decoding_spec(_EosTok(7, 0), do_sample=False, temperature=0.3, max_new_tokens=8)
    assert g["do_sample"] is False and g["temperature"] == 1.0 and g["pad_token_id"] == 0   # a pad id of 0 is a pad id
    assert g["eos_token_id"] == 7 and g["repetition_penalty"] == 1.0 and g["max_new_tokens"] == 8
    src = (REPO_ROOT / "tools" / "own_run.py").read_text(encoding="utf-8")
    calls = [line for line in src.splitlines() if ".generate(" in line]
    assert calls and all("**spec" in line for line in calls), calls
