"""The PipelineRL pin for the agentic arm (design 2026-11-09): UPSTREAM.md cites the pinned sha with the files the
driver's rendering depends on; config/pipelinerl_tir.toml loads UNREGISTERED with the agreed values; the system prompt
file is upstream's YAML block byte for byte (ends with one newline)."""
import hashlib
import re

import lag_ladder
from lag_ladder import REPO_ROOT
from lag_ladder.config import load_reader_config

PIN = "58d393458625ad63ed539f2dcd072c85700c557f"
CITED_FILES = ("conf/tir.yaml", "conf/base.yaml", "conf/finetune/base.yaml", "conf/finetune/grpo.yaml",
               "pipelinerl/domains/tir/rollouts.py", "pipelinerl/async_llm.py", "pipelinerl/rl_tool_parser_plugin.py",
               "pipelinerl/actor.py", "pipelinerl/finetune_loop.py", "pipelinerl/launch.py", "pipelinerl/streams.py",
               "pipelinerl/domains/math/load_datasets.py")


def test_upstream_md_pins_pipelinerl_for_the_agentic_arm_with_the_cited_files():
    text = (REPO_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    assert PIN == lag_ladder.PRODUCER_OBSERVED_SHA
    sec = text.split("PipelineRL — pinned for the agentic arm", 1)
    assert len(sec) == 2, "UPSTREAM.md has no 'PipelineRL — pinned for the agentic arm' subsection"
    body = sec[1].split("\n## 3.", 1)[0]
    assert PIN in body
    for f in CITED_FILES:
        assert f"`{f}`" in body, f"UPSTREAM.md does not cite {f}"
    assert "never imported" in body and "Hydra overrides" in body
    # the set of 40-hex shas is unchanged (tests/test_imports.py's rule): the new section cites the same sha
    shas = set(re.findall(r"\b[0-9a-f]{40}\b", text))
    assert shas == {lag_ladder.INSTRUMENT_SHA, lag_ladder.CHASSIS_SHA, lag_ladder.PRODUCER_OBSERVED_SHA}


def test_system_prompt_file_is_upstreams_yaml_block_with_one_trailing_newline():
    p = REPO_ROOT / "config" / "pipelinerl_tir_system_prompt.txt"
    raw = p.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert text.startswith("You are a math-focused AI Agent. Solve problems by combining clear symbolic reasoning\n")
    assert text.rstrip("\n").endswith("Always verify with run_python_code before invoking MathAnswer.")
    assert text.count("\n") == 20 and len(raw) == 986
    assert "run_python_code" in text and "MathAnswer" in text and "\\boxed{}" in text
    # pinned so a silent edit fails here before it fails the fidelity gate on the pod (the block of conf/tir.yaml:8-28 at
    # the pin, indent stripped, one trailing newline; computed 2026-11-09 from the pinned file)
    assert hashlib.sha256(raw).hexdigest() == "7ee8c06e19444789e29a71dcd483d954baae4cb5c2f4f617c90ca72513c12b41"


def test_pipelinerl_tir_reader_config_loads_unregistered_with_b1min_values():
    cfg = load_reader_config(REPO_ROOT / "config" / "pipelinerl_tir.toml", REPO_ROOT)
    assert cfg.registered_by == ""
    assert cfg.seal == "pipelinerl-tir"
    assert cfg.heldout == (40, 50) and cfg.prefix_len == 768 and cfg.seed == 0
    assert cfg.rule["holds_max"] == 1e-3 and cfg.rule["degrades_min"] == 1e-1
    assert cfg.controls["per_pair"] is True and cfg.controls["floor_max_median_kl"] == 1e-4
    assert cfg.controls["identity_max_logit_diff"] == 1e-5 and cfg.controls["scrambled_min"] == 1e-1
    assert set(cfg.pairs) == {"qwen2.5-1.5b-to-instruct", "qwen2.5-1.5b-instruct-pltir-agprompt"}
    held = cfg.pairs["qwen2.5-1.5b-to-instruct"]
    assert held.tokens_sha256 == "01bd7698f7413ab9a3b6958991487bbba626f5a8cc4f24142cd22091d96af747"
    ag = cfg.pairs["qwen2.5-1.5b-instruct-pltir-agprompt"]
    assert ag.spans is not None and ag.heldout == (0, 50) and ag.far_pair is None
    assert ag.tokens_sha256.endswith("_PENDING") and ag.spans_sha256.endswith("_PENDING")
    labels = [r.label for r in cfg.rungs]
    assert labels == [f"pltir-lag-{g}" for g in (1, 2, 4, 8, 16, 32)] + [f"pltir-agprompt-lag-{g}" for g in (1, 2, 4, 8, 16, 32)]
    for r in cfg.rungs:
        g = int(r.label.rsplit("-", 1)[1])
        assert r.reader == f"local:pltir-step_{g}"
        assert r.writer_dump == "target@local-pltir-step_0" and r.reader_dump == f"target@local-pltir-step_{g}"
    assert all(r.texts == ("rollout",) for r in cfg.rungs if r.label.startswith("pltir-agprompt"))


def test_trainer_table_carries_the_producer_pin_and_checkpoint_names():
    import tomllib
    with open(REPO_ROOT / "config" / "pipelinerl_tir.toml", "rb") as f:
        t = tomllib.load(f)["trainer"]
    assert t["producer_sha"] == PIN and t["config_name"] == "tir"
    assert t["model"] == "Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert t["checkpoint_prefix"] == "pltir-step_" and t["steps"] == 96
    assert t["keep"] == [0, 1, 2, 4, 8, 16, 32, 64, 65, 66, 68, 72, 80, 96]
    assert t["dump_steps"] == [0, 1, 2, 4, 8, 16, 32]
    assert t["kvt_pair"] == "qwen2.5-1.5b-to-instruct" and t["kvt_which"] == "target"
    assert t["frozen_seed"] == 0 and t["frozen_n"] == 50
    assert t["overrides"] == "config/pipelinerl_tir_overrides.txt"
    assert t["sample_overrides"] == "config/pipelinerl_tir_sample_overrides.txt"
    assert t["system_prompt"] == "config/pipelinerl_tir_system_prompt.txt"
