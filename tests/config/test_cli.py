from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_rl.cli import main, validate_smoke_config
from tgvf_rl.compatibility_stack import (
    CONTROL_COMPATIBILITY_STACK,
    TORCH211_CU129_COMPATIBILITY_STACK,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_bounded_fsdp2_config_is_explicit_and_valid() -> None:
    config = validate_smoke_config(REPOSITORY_ROOT / "configs/smoke/fsdp2.toml")
    assert config["stack"]["physical_gpu_ids"] == [2, 3]
    assert config["stack"]["vllm_enable_mm_embeds"] is True
    assert config["run_id"] == "I8H-SC-30-FSDP2-INFRA-20260719"
    assert config["objective"]["production_rl"] is False
    assert config["checkpoint"]["contents"] == ["model", "optimizer", "extra"]


def test_torch211_fsdp2_config_requires_explicit_candidate_stack() -> None:
    path = REPOSITORY_ROOT / "configs/smoke/fsdp2_torch211.toml"
    with pytest.raises(ValueError, match="verl_commit"):
        validate_smoke_config(path)
    config = validate_smoke_config(
        path, stack_selector=TORCH211_CU129_COMPATIBILITY_STACK
    )
    assert config["stack"]["verl_commit"] == (
        "638b8ff84f279e054982f1f4633a546f3c6ced68"
    )


def test_cli_can_print_static_compatibility_contract(capsys) -> None:
    assert main(["compat-info"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "rollout_backend" not in output
    assert "verl_candidate_commit" in output
    assert output["compatibility_stack"]["selector"] == CONTROL_COMPATIBILITY_STACK
    assert output["required_overrides"]["trainer.use_v1"] is False


def test_cli_requires_named_candidate_stack_and_rejects_freeform(capsys) -> None:
    assert (
        main(
            [
                "compat-info",
                "--stack",
                TORCH211_CU129_COMPATIBILITY_STACK,
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert (
        output["compatibility_stack"]["selector"] == TORCH211_CU129_COMPATIBILITY_STACK
    )
    overrides = output["required_overrides"]
    assert overrides["trainer.use_v1"] is True
    assert overrides["actor_rollout_ref.rollout.free_cache_engine"] is False
    assert overrides["actor_rollout_ref.rollout.enable_sleep_mode"] is False
    assert overrides["actor_rollout_ref.rollout.checkpoint_engine.backend"] == "naive"
    assert overrides[
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
    ].endswith("LosslessTransferQueueAgentLoopManager")
    with pytest.raises(SystemExit):
        main(["compat-info", "--stack", "2.11-custom"])
