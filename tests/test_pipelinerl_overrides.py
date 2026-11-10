"""The two Hydra override files for the PipelineRL TIR arm (design 2026-11-09 §4, R3, R4, RC2, §5.1): the production
recipe and the three lines that turn it into the 50-prompt actor-mode sample. Values are the design's; the key names are
upstream's at 58d3934. No path, host, token or spend figure may appear in either file."""
import re

from lag_ladder import REPO_ROOT

PRODUCTION = {
    "finetune": "grpo",
    "finetune.rl.policy_loss": "ppo",
    "finetune.attempts": "8",
    "finetune.learning_rate": "1e-6",
    "finetune.lr_scheduler_type": "constant_with_warmup",
    "finetune.num_warmup_steps": "0",
    "finetune.gradient_accumulation_passes": "128",
    "finetune.train_batch_size": "1",
    "finetune.max_train_steps": "96",
    "finetune.save_checkpoint_steps": "1000",
    "finetune.also_save_steps": "[1,2,4,8,16,32,64,65,66,68,72,80,96]",
    "finetune.keep_intermediate_checkpoints": "true",
    "finetune.seq_length": "8192",
    "finetune.seq_parallel": "1",
    "llm.parameters.max_tokens": "1024",
    "test_llm.parameters.max_tokens": "1024",
    "vllm_config.vllm_kwargs.max_model_len": "8192",
    "train_dataset_names": "[gsm8k_train]",
    "test_dataset_names": "[gsm8k_test]",
    "wandb.use_wandb": "false",
    "seed": "42",
}
SAMPLE = {
    "debug.mode": "actor",
    "finetune.attempts": "1",
    "train_dataset_names": "[pltir_frozen50]",
}
FORBIDDEN = re.compile(r"output_dir|model_path|https?://|/home/|/workspace/|[A-Za-z]:\\|\$|hf_[A-Za-z0-9]|\d+\s*USD|\brunpod\b", re.I)


def parse_overrides(text: str) -> dict[str, str]:
    """One `key=value` per non-comment line; the first `=` splits; a key twice is an error."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"not an override: {line!r}"
        k, v = line.split("=", 1)
        assert k not in out, f"key {k} twice"
        out[k] = v
    return out


def _read(name: str) -> str:
    p = REPO_ROOT / "config" / name
    raw = p.read_bytes()
    assert b"\r" not in raw, f"{name} has CRLF line endings"
    return raw.decode("utf-8")


def test_production_overrides_are_exactly_the_designs_values():
    text = _read("pipelinerl_tir_overrides.txt")
    assert parse_overrides(text) == PRODUCTION
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
    # the comment names where the machine values come from and what asserts the resolved config
    assert "output_dir" not in parse_overrides(text) and "--cfg job --resolve" in text


def test_sample_overrides_turn_the_recipe_into_the_actor_mode_sample():
    text = _read("pipelinerl_tir_sample_overrides.txt")
    assert parse_overrides(text) == SAMPLE
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
    # the sample overrides the production file's attempts and train set and nothing it does not name
    merged = {**PRODUCTION, **SAMPLE}
    assert merged["finetune.attempts"] == "1" and merged["train_dataset_names"] == "[pltir_frozen50]"
    assert merged["test_dataset_names"] == "[gsm8k_test]" and merged["finetune.rl.policy_loss"] == "ppo"
    assert "datasets/custom/pltir_frozen50.jsonl" in text
