from __future__ import annotations

from dataclasses import replace

import pytest

from tgvf_rl.contracts.errors import ContractUnsetError, IdentityMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity
from tgvf_rl.policy import (
    DecoderLoRAConfig,
    PilotGRPOConfig,
    PilotSamplingConfig,
    PolicyPilotV1Config,
    PolicyVisualToolExperimentConfig,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile


SHA0 = "0" * 64
SHA1 = "1" * 64


def _identity(
    config: PilotSamplingConfig, *, max_tokens: int = 127
) -> SamplingIdentity:
    assert config.min_p is not None
    return SamplingIdentity(
        policy_version=PolicyVersion("pilot", 0, SHA0),
        backend=config.backend,
        backend_version=config.backend_version,
        seed=42,
        rng_state_sha256=SHA1,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        repetition_penalty=config.repetition_penalty,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        logit_processors=config.logit_processors,
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=config.asynchronous_staleness_steps,
        max_tokens=max_tokens,
        do_sample=config.do_sample,
        stop_token_ids=config.stop_token_ids,
        stop_strings=config.stop_strings,
        include_stop_str_in_output=config.include_stop_str_in_output,
        ignore_eos=config.ignore_eos,
    )


def _bound_sampling() -> PilotSamplingConfig:
    return PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(2,),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=False,
        ignore_eos=False,
    )


def test_sampling_is_unbound_until_min_p_is_an_explicit_run_input() -> None:
    sampling = PilotSamplingConfig()
    assert sampling.is_run_bound is False
    with pytest.raises(ContractUnsetError, match="min_p"):
        sampling.as_vllm_parameters(max_tokens=128)

    with pytest.raises(ContractUnsetError, match="stop/EOS"):
        sampling.bind_min_p(0.0).as_vllm_parameters(max_tokens=128)

    bound = _bound_sampling()
    parameters = bound.as_vllm_parameters(max_tokens=127)
    assert parameters == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_token_ids": [2],
        "stop": ["</tool_call>"],
        "include_stop_str_in_output": False,
        "ignore_eos": False,
        "max_tokens": 127,
        "logprobs": True,
    }
    assert "n" not in parameters
    assert bound.remaining_response_tokens(17) == 8175
    bound.validate_sampling_identity(_identity(bound), expected_max_tokens=127)


def test_sampling_identity_and_trajectory_budget_fail_closed() -> None:
    sampling = _bound_sampling()
    with pytest.raises(IdentityMismatchError, match="SamplingIdentity"):
        sampling.validate_sampling_identity(
            replace(_identity(sampling), repetition_penalty=1.1),
            expected_max_tokens=127,
        )
    with pytest.raises(IdentityMismatchError, match="max_tokens"):
        sampling.validate_sampling_identity(
            _identity(sampling, max_tokens=126), expected_max_tokens=127
        )
    with pytest.raises(ValueError, match="exhausted"):
        sampling.remaining_response_tokens(8192)
    with pytest.raises(ValueError, match="remaining trajectory budget"):
        sampling.as_vllm_parameters(max_tokens=8193)


def test_policy_pilot_v1_fixes_only_tgvf_and_exact_optimizer_envelope() -> None:
    pilot = PolicyPilotV1Config(
        sampling=_bound_sampling(),
    )
    assert pilot.enabled_tool_names == ("tgvf_focus_tool",)
    assert pilot.tool_profile is NativeToolCapabilityProfile.TGVF_ONLY
    assert pilot.max_tgvf_call_attempts == 4
    assert pilot.image_max_pixels == 512 * 512
    assert pilot.sampling.trajectories_per_prompt == 8
    assert pilot.sampling.max_response_length == 8192
    assert pilot.lora == DecoderLoRAConfig()
    assert pilot.lora.expected_target_module_count == 36 * 7
    assert pilot.grpo == PilotGRPOConfig()
    assert pilot.grpo.total_training_epochs is None
    assert pilot.grpo.ratio_bounds == (0.8, 1.2)
    assert len(pilot.identity_sha256) == 64

    with pytest.raises(ValueError, match="enabled_tool_names"):
        PolicyPilotV1Config(
            enabled_tool_names=("tgvf_focus_tool", "image_zoom_in_tool")
        )
    with pytest.raises(ValueError, match="tool_profile"):
        PolicyPilotV1Config(tool_profile=NativeToolCapabilityProfile.CROP_ONLY)
    with pytest.raises(ValueError, match="dropout"):
        DecoderLoRAConfig(dropout=0.1)
    assert DecoderLoRAConfig(initial_learning_rate=1.0e-6).initial_learning_rate == 1.0e-6
    assert DecoderLoRAConfig(initial_learning_rate=3.0e-6).initial_learning_rate == 3.0e-6
    assert DecoderLoRAConfig(initial_learning_rate=1.0e-5).initial_learning_rate == 1.0e-5
    with pytest.raises(ValueError, match="initial_learning_rate"):
        DecoderLoRAConfig(initial_learning_rate=2.0e-6)
    with pytest.raises(ValueError, match="filter_groups"):
        PilotGRPOConfig(filter_groups=True)


def test_visual_tool_experiment_accepts_crop_without_relaxing_formal_pilot() -> None:
    crop = PolicyVisualToolExperimentConfig(
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        enabled_tool_names=("image_zoom_in_tool",),
        sampling=_bound_sampling(),
    )

    assert crop.tool_profile is NativeToolCapabilityProfile.CROP_ONLY
    assert crop.enabled_tool_names == ("image_zoom_in_tool",)
    assert crop.image_max_pixels == 512 * 512
    with pytest.raises(ValueError, match="TGVF-only"):
        PolicyVisualToolExperimentConfig(sampling=_bound_sampling())
