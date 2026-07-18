from __future__ import annotations

import pytest

from tgvf_rl.config.loader import validate_run_config
from tgvf_rl.config.schema import RunConfig, RunGate, StackConfig
from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ModelIdentity


def test_production_config_does_not_invent_data_reward_or_prompt() -> None:
    config = RunConfig(
        run_id="not-production-yet",
        gate=RunGate.PRODUCTION,
        stack=StackConfig("e003163", "vllm", "fsdp2", True, (2, 3)),
        primary_model=ModelIdentity(
            "qwen3_vl", "fixture", "/fixture", 151669, "0" * 64
        ),
        secondary_model=None,
        representation_artifact=None,
        data_manifest=None,
        prompt_identity=None,
        reward_identity=None,
        objective_identity=None,
        max_tool_calls=None,
    )
    with pytest.raises(ContractUnsetError):
        validate_run_config(config)


def test_gpu_scope_rejects_unapproved_physical_devices() -> None:
    with pytest.raises(ValueError, match="only physical GPUs"):
        StackConfig("e003163", "vllm", "fsdp2", True, (0, 1))
