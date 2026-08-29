from __future__ import annotations

from pathlib import Path

from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.config import PolicyCropTGVFPixel512ParityExperimentConfig
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile


_ROOT = Path(__file__).resolve().parents[2]
_FORMAL = _ROOT / (
    "configs/policy/runs/prl_26_e_qwen3_instruct_full_atomic_crop_tgvf_"
    "train512_parity_s32_bs16_n16_teacher25_ws8.toml"
)
_CANARY = _ROOT / (
    "configs/policy/runs/prl_26_e_c0_qwen3_instruct_full_atomic_crop_tgvf_"
    "train512_parity_bs4_n2_teacher25_1step_ws4.toml"
)


def test_atomic_formal_is_fresh_s0_pixel512_s32_teacher25_parity() -> None:
    config = load_policy_e2e_smoke_run_config(_FORMAL.resolve())

    assert (
        config.schema_version
        == POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
    )
    assert isinstance(config.policy, PolicyCropTGVFPixel512ParityExperimentConfig)
    assert config.policy.image_max_pixels == 262_144
    assert config.protocol.tool_profile is NativeToolCapabilityProfile.CROP_TGVF
    assert config.protocol.enabled_tool_names == ("tgvf_crop_tool",)
    assert config.protocol.prompt_sha256 == (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert config.policy.sampling.stop_strings == ("</tool_call>",)
    assert config.policy.sampling.stop_token_ids == (151_645,)
    assert config.policy.sampling.include_stop_str_in_output is True
    assert config.policy.sampling.trajectories_per_prompt == 16
    assert config.policy.sampling.temperature == 1.0
    assert config.rollout_rng.master_seed == 42
    assert config.dataset.runtime_binding.schedule_seed == 42
    assert config.accumulation.global_prompt_batch_size == 16
    assert config.optimizer.learning_rate == 1.0e-6
    assert config.training.maximum_optimizer_steps == 32
    assert config.training.checkpoint_steps == (0, 8, 16, 24, 32)
    assert config.training.permanent_checkpoint_steps == (8, 16, 24, 32)
    assert config.training.resume_mode == "auto"
    assert config.training.resume_from_path is None
    assert config.representation.adapter_update_mode.value == "frozen_adapter"
    assert not config.reward.tool_utility_reward_enabled
    assert not config.reward.focus_reward_enabled
    assert not config.reward.grounding_reward_enabled


def test_atomic_c0_is_separate_bounded_functional_gate() -> None:
    config = load_policy_e2e_smoke_run_config(_CANARY.resolve())

    assert config.policy.image_max_pixels == 262_144
    assert config.policy.sampling.trajectories_per_prompt == 2
    assert config.policy.sampling.max_response_length == 512
    assert config.distributed.physical_gpu_ids == (0, 1, 2, 3)
    assert config.accumulation.global_prompt_batch_size == 4
    assert config.training.maximum_optimizer_steps == 1
    assert config.training.resume_mode == "disable"
    assert config.training.permanent_checkpoint_steps == (1,)


def test_atomic_launcher_selects_atomic_teacher25_dataset_and_boundary() -> None:
    config = load_policy_e2e_smoke_run_config(_FORMAL.resolve())
    plan = build_trainable_tgvf_verl_launch_plan(config, mode="formal", target_step=32)

    assert plan.overrides["data.custom_cls.name"] == "PolicyTeacherQuarterMixDataset"
    assert (
        plan.overrides["actor_rollout_ref.rollout.agent.default_agent_loop"]
        == "prl20_crop_tgvf_deepeyes_matched_visual"
    )
    binding = plan.overrides["data.policy_teacher_quarter_mix"]
    assert binding["schedule_seed"] == 42
    assert binding["tool_profile"] is NativeToolCapabilityProfile.CROP_TGVF
    assert binding["visual_prompt_bundle_sha256"] == config.protocol.prompt_sha256
    custom = plan.overrides["actor_rollout_ref.rollout.custom"]
    assert custom["sampling"]["stop_strings"] == ["</tool_call>"]
    assert custom["sampling"]["include_stop_str_in_output"] is True
    assert custom["protocol"]["maximum_tool_calls"] == 6
