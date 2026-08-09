from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from tgvf_rl.framework.verl.policy_task_runner import (
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
)
from tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from tgvf_rl.framework.verl.trainable_tgvf_engine import TRAINABLE_TGVF_MODEL_TYPE
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    TRAINABLE_TGVF_EXTERNAL_MODULE,
    TrainableTGVFVerlLaunchPlan,
    apply_trainable_tgvf_launch_environment,
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_CONFIG = (
    _ROOT
    / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_matched_8step_ws8.toml"
)
_CONFIG_WS4 = (
    _ROOT
    / "configs/policy/runs/"
    "prl_15_r1_qwen3_instruct_full_rp66_bs16_n16_t1_"
    "crop16_math_equiv_8step_ws4.toml"
)


def _config():
    return load_policy_e2e_smoke_run_config(_CONFIG)


def _config_ws4():
    return load_policy_e2e_smoke_run_config(_CONFIG_WS4)


def test_formal_plan_binds_full_model_matched_rp66_path() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    values = plan.overrides

    assert plan.target_step == values["trainer.total_training_steps"] == 8
    assert values["data.train_files"] == [str(DEEPEYES_TRAIN_SENTINEL)]
    assert values["data.val_files"] == [str(DEEPEYES_PROBE_SENTINEL)]
    assert values["data.custom_cls.name"] == "TGVFDeepEyesMatchedDataset"
    assert values["data.train_batch_size"] == 16
    assert values["actor_rollout_ref.rollout.n"] == 16
    assert values["trainer.n_gpus_per_node"] == 8
    assert values["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 8
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 32
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 32
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is True
    assert values["actor_rollout_ref.model.use_remove_padding"] is True
    assert values["actor_rollout_ref.model.use_fused_kernels"] is True
    assert values["actor_rollout_ref.model.override_config.attn_implementation"] == (
        "sdpa"
    )
    assert values["actor_rollout_ref.actor.fsdp_config.use_torch_compile"] is True
    assert values["actor_rollout_ref.actor.fsdp_config.model_dtype"] == "fp32"
    assert values["actor_rollout_ref.actor.optim.total_training_steps"] == 8
    assert values["actor_rollout_ref.rollout.response_length"] == 20480
    assert "actor_rollout_ref.rollout.repetition_penalty" not in values
    assert values["data.filter_overlong_prompts"] is True
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert values["actor_rollout_ref.model.lora_rank"] == 0
    assert values["actor_rollout_ref.model.lora.rank"] == 0
    assert values["actor_rollout_ref.actor.freeze_vision_tower"] is False
    assert values["actor_rollout_ref.model.external_lib"] == (
        TRAINABLE_TGVF_EXTERNAL_MODULE
    )
    assert values["actor_rollout_ref.model.model_type"] == TRAINABLE_TGVF_MODEL_TYPE
    assert values["actor_rollout_ref.rollout.checkpoint_manager_class"] == (
        TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
    )
    assert values["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
    )
    assert values["actor_rollout_ref.actor.policy_loss.loss_mode"] == (
        NATIVE_DEEPEYES_POLICY_LOSS_MODE
    )
    custom = values["actor_rollout_ref.rollout.custom"]
    assert custom["protocol"]["maximum_tool_calls"] == 6
    assert custom["weight_sync"]["interval_optimizer_steps"] == 1
    assert custom["reward"]["judge_config_sha256"] == (
        plan.config.reward.judge_config_sha256
    )
    assert custom["checkpoint_steps"] == [0, 1, 4, 8]
    assert custom["reference_diagnostic"] == {
        "enabled": False,
        "coefficient": 0.0,
        "worker_route": "disabled_zero_kl_control",
        "observation_source": "not_computed",
    }
    assert plan.environment[POLICY_REFERENCE_DIAGNOSTIC_ENV] == "0"


def test_smoke_changes_horizon_output_and_checkpoint_not_scientific_shape() -> None:
    formal = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    smoke = build_trainable_tgvf_verl_launch_plan(_config(), mode="smoke")

    assert smoke.target_step == 1
    assert formal.overrides["trainer.logger"] == ["console", "wandb"]
    assert smoke.overrides["trainer.logger"] == ["console"]
    assert smoke.overrides["trainer.total_training_steps"] == 1
    assert smoke.overrides["trainer.default_local_dir"].endswith(
        "/smoke/checkpoints"
    )
    assert formal.environment[POLICY_METRICS_PATH_ENV].endswith("/metrics.jsonl")
    assert "/smoke/" not in formal.environment[POLICY_METRICS_PATH_ENV]
    assert smoke.environment[POLICY_METRICS_PATH_ENV].endswith(
        "/smoke/metrics.jsonl"
    )
    assert smoke.overrides["actor_rollout_ref.rollout.custom"][
        "checkpoint_steps"
    ] == [0, 1]
    for key in (
        "data.train_files",
        "data.val_files",
        "data.train_batch_size",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    ):
        assert smoke.overrides[key] == formal.overrides[key]


def test_world4_smoke_preserves_crop16_equal_micro_objective() -> None:
    world8 = build_trainable_tgvf_verl_launch_plan(_config(), mode="smoke")
    world4 = build_trainable_tgvf_verl_launch_plan(_config_ws4(), mode="smoke")

    assert world8.overrides["trainer.n_gpus_per_node"] == 8
    assert world4.overrides["trainer.n_gpus_per_node"] == 4
    assert world4.overrides["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 4
    assert world4.overrides["actor_rollout_ref.ref.fsdp_config.fsdp_size"] == 4
    assert world4.overrides["actor_rollout_ref.rollout.agent.num_workers"] == 4
    assert world4.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    for key in (
        "data.train_batch_size",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    ):
        assert world4.overrides[key] == world8.overrides[key]

    assert world4.overrides["data.train_batch_size"] == 16
    assert world4.overrides["actor_rollout_ref.rollout.n"] == 16
    assert world4.overrides[
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
    ] == 32
    assert world4.overrides[
        "actor_rollout_ref.actor.optim.total_training_steps"
    ] == 8
    actor_batch = world4.overrides["actor_rollout_ref.rollout.custom"][
        "actor_batch_contract"
    ]
    assert actor_batch == {
        "global_prompt_batch_size": 16,
        "rollouts_per_prompt": 16,
        "fsdp_data_parallel_size": 4,
        "prompt_micro_batch_size_per_rank": 2,
        "configured_gradient_accumulation_steps": 2,
        "upstream_ppo_mini_batch_size_prompts": 16,
        "upstream_internal_mini_batch_size_trajectories": 256,
        "upstream_ppo_micro_batch_size_per_gpu_trajectories": 32,
        "upstream_inference_micro_batch_size_per_gpu_trajectories": 32,
        "derived_actor_forward_backward_microbatches": 2,
        "derived_gradient_accumulation_steps": 2,
        "optimizer_steps_per_trainer_step": 1,
    }


def test_labeled_smoke_gets_an_isolated_output_closure() -> None:
    smoke = build_trainable_tgvf_verl_launch_plan(
        _config(), mode="smoke", smoke_id="actor-rollout-only-v1"
    )

    assert smoke.overrides["trainer.default_local_dir"].endswith(
        "/smoke/actor-rollout-only-v1/checkpoints"
    )
    assert smoke.environment[POLICY_METRICS_PATH_ENV].endswith(
        "/smoke/actor-rollout-only-v1/metrics.jsonl"
    )


def test_launch_environment_drops_optional_state_from_an_earlier_run() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config_ws4(), mode="smoke")
    environment = {
        "UNRELATED_PARENT_VALUE": "preserve",
        "TGVF_POLICY_AGENT_LOOP_WORKER_INDEX": "7",
        "TGVF_POLICY_HORIZON_EXTENSION_PATH": "/stale/extension.json",
        "TGVF_POLICY_HORIZON_EXTENSION_SHA256": "f" * 64,
    }

    observed = apply_trainable_tgvf_launch_environment(
        plan,
        environment=environment,
    )

    assert observed is environment
    assert observed["UNRELATED_PARENT_VALUE"] == "preserve"
    assert "TGVF_POLICY_AGENT_LOOP_WORKER_INDEX" not in observed
    assert "TGVF_POLICY_HORIZON_EXTENSION_PATH" not in observed
    assert "TGVF_POLICY_HORIZON_EXTENSION_SHA256" not in observed
    assert observed["TGVF_POLICY_RUN_ID"] == plan.config.run_id


def test_dedicated_plan_rejects_an_enabled_policy_lora() -> None:
    valid = build_trainable_tgvf_verl_launch_plan(_config(), mode="smoke")
    with pytest.raises(ValueError, match="lora_rank"):
        TrainableTGVFVerlLaunchPlan(
            config=valid.config,
            mode=valid.mode,
            target_step=valid.target_step,
            overrides={**valid.overrides, "actor_rollout_ref.model.lora_rank": 8},
            environment=valid.environment,
            external_components=valid.external_components,
        )


def test_every_dotted_override_renders_for_pinned_hydra() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    arguments = plan.hydra_override_args()

    assert len(arguments) == len(plan.overrides)
    assert all(argument.startswith("++") and "=" in argument for argument in arguments)
