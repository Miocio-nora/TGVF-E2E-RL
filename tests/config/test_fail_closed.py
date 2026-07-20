from __future__ import annotations

from dataclasses import replace

import pytest

from tgvf_rl.config.loader import config_sha256, validate_run_config
from tgvf_rl.config.schema import RunConfig, RunGate, StackConfig
from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity


def test_production_config_does_not_invent_data_reward_or_prompt() -> None:
    config = RunConfig(
        run_id="not-production-yet",
        gate=RunGate.PRODUCTION,
        stack=StackConfig("e003163", "vllm", "fsdp2", True, (2, 3)),
        primary_model=ModelIdentity(
            "qwen3_vl", "fixture", "/fixture", 151669, "0" * 64
        ),
        secondary_model=None,
        target_conditioning=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        ),
        representation_artifact=None,
        data_manifest=None,
        prompt_identity=None,
        reward_identity=None,
        objective_identity=None,
        max_tool_calls=None,
        enabled_tool_names=None,
    )
    with pytest.raises(ContractUnsetError):
        validate_run_config(config)


def test_gpu_scope_rejects_unapproved_physical_devices() -> None:
    with pytest.raises(ValueError, match="only physical GPUs"):
        StackConfig("e003163", "vllm", "fsdp2", True, (0, 1))


def test_provider_selection_is_part_of_run_identity() -> None:
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, "0" * 64)

    def config(target_conditioning: TargetConditioningConfig) -> RunConfig:
        return RunConfig(
            run_id="provider-identity",
            gate=RunGate.SKELETON,
            stack=StackConfig("e003163", "vllm", "fsdp2", True, ()),
            primary_model=model,
            secondary_model=None,
            target_conditioning=target_conditioning,
            representation_artifact=None,
            data_manifest=None,
            prompt_identity=None,
            reward_identity=None,
            objective_identity=None,
            max_tool_calls=None,
            enabled_tool_names=None,
        )

    contextual = config(
        TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        )
    )
    embedding = config(
        TargetConditioningConfig(
            provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
            embedding_identity="model.embed_tokens",
        )
    )

    assert config_sha256(contextual) != config_sha256(embedding)

    tgvf_only = replace(contextual, enabled_tool_names=("tgvf_focus_tool",))
    fused = replace(
        contextual,
        enabled_tool_names=("tgvf_focus_tool", "image_zoom_in_tool"),
    )
    assert config_sha256(tgvf_only) != config_sha256(fused)


def test_rollout_requires_an_explicit_registered_tool_set() -> None:
    base = RunConfig(
        run_id="tool-selection",
        gate=RunGate.SYNTHETIC_ROLLOUT,
        stack=StackConfig("e003163", "vllm", "fsdp2", True, ()),
        primary_model=ModelIdentity(
            "qwen3_vl", "fixture", "/fixture", 151669, "0" * 64
        ),
        secondary_model=None,
        target_conditioning=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        ),
        representation_artifact=None,
        data_manifest=None,
        prompt_identity=ArtifactIdentity("prompt", "fixture", "v1", "1" * 64),
        reward_identity=None,
        objective_identity=None,
        max_tool_calls=3,
        enabled_tool_names=("tgvf_focus_tool", "image_zoom_in_tool"),
    )
    validate_run_config(base)
    with pytest.raises(ValueError, match="unknown enabled rollout tools"):
        validate_run_config(replace(base, enabled_tool_names=("unknown",)))
