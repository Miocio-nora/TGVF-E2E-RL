from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.framework.verl.launcher import (
    build_policy_e2e_smoke_verl_plan,
    compose_upstream_verl_config,
)
from tgvf_rl.framework.verl.launcher import _reward_custom_config
from tgvf_rl.framework.verl.method_matrix_launcher import (
    route_policy_method_matrix_plan,
)
from tgvf_rl.framework.verl.policy_task_runner import (
    _stage3_reward_coefficients,
    _stage3_reward_switches,
)
from tgvf_rl.policy.config import PolicyMethodProfile
from tgvf_rl.policy.deepeyes_official_protocol import VISUAL_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config
from tgvf_rl.protocol import (
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)
from tgvf_rl.rewards.stage3_shaped import (
    Stage3ShapedComponentName,
    Stage3ShapedRewardFacts,
    Stage3ShapedRewardKernel,
)
from tests.policy.test_method_run_config import (
    method_config_factory as _policy_method_config_factory,
)
from tests.policy.test_run_config import _write_minimal_upstream_config_directory


@pytest.fixture(name="method_config_factory")
def _method_config_factory_fixture(tmp_path):
    return _policy_method_config_factory.__wrapped__(tmp_path)


def _tfree_reward(**changes: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "profile": "stage3-shaped-v1",
        "task_kind": "mixed",
        "answer_verifier": "rule_first_qwen25_72b",
        "answer_verifier_sha256": "1" * 64,
        "judge_mode": "qwen25_72b_semantic_fallback",
        "judge_config_path": "/fixture/answer-judge.json",
        "judge_config_sha256": "2" * 64,
        "answer_weight": None,
        "format_weight": None,
        "conditional_tool_weight": None,
        "tool_utility": None,
        "tool_utility_reward_enabled": False,
        "focus_reward_enabled": False,
        "grounding_reward_enabled": False,
        "visual_quality_judge_mode": "disabled",
        "visual_quality_judge_config_path": None,
        "visual_quality_judge_config_sha256": None,
        "visual_quality_judge_identity": None,
        "answer_reward_scale": 2.0,
        "repeated_call_penalty": 0.05,
        "protocol_error_penalty": 2.0,
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_launcher_emits_answer_protocol_reward_without_external_sidecars() -> None:
    custom = _reward_custom_config(SimpleNamespace(reward=_tfree_reward()))

    assert custom["answer_reward_scale"] == 2.0
    assert custom["repeated_call_penalty"] == 0.05
    assert custom["protocol_error_penalty"] == 2.0
    assert custom["tool_utility_reward_enabled"] is False
    assert custom["focus_reward_enabled"] is False
    assert custom["grounding_reward_enabled"] is False
    assert custom["visual_quality_judge_mode"] == "disabled"
    assert not any("tool_utility_sidecar" in key for key in custom)
    assert not any("visual_quality_judge_config" in key for key in custom)


def test_task_runner_reads_all_reward_coefficients_from_binding() -> None:
    config = SimpleNamespace(reward=_tfree_reward())

    assert _stage3_reward_coefficients(config) == (2.0, 0.05, 2.0)
    assert _stage3_reward_switches(config) == (False, False)
    config.reward = _tfree_reward(
        answer_reward_scale=0.0,
        repeated_call_penalty=0.0,
        protocol_error_penalty=0.0,
    )
    assert _stage3_reward_coefficients(config) == (0.0, 0.0, 0.0)


def test_real_typed_tfree_config_builds_base_and_method_plan(
    method_config_factory,
) -> None:
    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
    )
    config = load_policy_e2e_smoke_run_config(path)

    base = build_policy_e2e_smoke_verl_plan(config)
    plan = route_policy_method_matrix_plan(config, base)

    reward = base.overrides["actor_rollout_ref.rollout.custom"]["reward"]
    assert reward["answer_reward_scale"] == 1.25
    assert reward["repeated_call_penalty"] == 0.0
    assert reward["protocol_error_penalty"] == 0.75
    assert reward["tool_utility_reward_enabled"] is False
    assert reward["focus_reward_enabled"] is False
    assert reward["grounding_reward_enabled"] is False
    assert not any("tool_utility_sidecar" in key for key in reward)
    assert not any("visual_quality_judge_config" in key for key in reward)
    assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 345_678
    assert plan.overrides["actor_rollout_ref.rollout.n"] == 3
    assert plan.external_components["method_matrix_profile"] == "crop"


def test_typed_config_reaches_reward_kernel_and_cpu_hydra_composition(
    method_config_factory,
    tmp_path: Path,
) -> None:
    """Close config -> reward -> plan -> upstream composition without a GPU."""

    path, _ = method_config_factory(
        profile=PolicyMethodProfile.CROP,
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
        observation_id=NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        image_max_pixels=456_789,
        trajectories_per_prompt=3,
        max_response_length=1_357,
        maximum_tool_calls=5,
        rollout_seed=91,
        maximum_optimizer_steps=3,
    )
    config = load_policy_e2e_smoke_run_config(path)
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    composed = compose_upstream_verl_config(
        plan,
        config_directory=_write_minimal_upstream_config_directory(tmp_path),
    )
    kernel = Stage3ShapedRewardKernel(
        answer_reward_scale=float(config.reward.answer_reward_scale),
        repeated_call_penalty=float(config.reward.repeated_call_penalty),
        protocol_error_penalty=float(config.reward.protocol_error_penalty),
    )
    reward = kernel.score(
        Stage3ShapedRewardFacts(
            answer_correct=True,
            tool_label=None,
            tool_call_count=2,
            successful_tgvf_observation_count=1,
            quality_rewards_enabled=False,
            label_confidence=None,
            tool_utility_reward_enabled=False,
            protocol_errors=("protocol_invalid",),
        )
    )

    assert config.policy.image_max_pixels == 456_789
    assert config.rollout_rng.master_seed == 91
    assert reward.answer_gated is False
    assert reward.component(Stage3ShapedComponentName.ANSWER).score == 1.25
    assert reward.component(Stage3ShapedComponentName.TOOL).score == 0.0
    assert reward.component(Stage3ShapedComponentName.FOCUS).score == 0.0
    assert reward.component(Stage3ShapedComponentName.GROUNDING).score == 0.0
    assert reward.component(Stage3ShapedComponentName.PROTOCOL).score == -0.75
    assert reward.total == 0.5
    assert composed.data.mm_processor_kwargs.max_pixels == 456_789
    assert composed.actor_rollout_ref.rollout.n == 3
    assert composed.trainer.total_training_steps == 3
