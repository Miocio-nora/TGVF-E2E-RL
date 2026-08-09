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
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_CONFIG = (
    _ROOT
    / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_matched_8step_gpu0123.toml"
)


def _config():
    return load_policy_e2e_smoke_run_config(_CONFIG)


def test_formal_plan_binds_full_model_matched_rp66_path() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    values = plan.overrides

    assert plan.target_step == values["trainer.total_training_steps"] == 8
    assert values["data.train_files"] == [str(DEEPEYES_TRAIN_SENTINEL)]
    assert values["data.val_files"] == [str(DEEPEYES_PROBE_SENTINEL)]
    assert values["data.custom_cls.name"] == "TGVFDeepEyesMatchedDataset"
    assert values["data.train_batch_size"] == 16
    assert values["actor_rollout_ref.rollout.n"] == 16
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
