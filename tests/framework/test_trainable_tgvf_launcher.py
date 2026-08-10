from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from tgvf_rl.framework.verl.policy_task_runner import (
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
    POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV,
)
from tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
    TGVF_CHECKPOINT_ENGINE_CONTROL_KEY,
)
from tgvf_rl.framework.verl.trainable_tgvf_engine import TRAINABLE_TGVF_MODEL_TYPE
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    TRAINABLE_TGVF_EXTERNAL_MODULE,
    TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES,
    TrainableTGVFVerlLaunchPlan,
    apply_trainable_tgvf_launch_environment,
    build_trainable_tgvf_verl_launch_plan,
    compose_trainable_tgvf_verl_config,
)
from tgvf_rl.framework.verl.policy_checkpoint_lifecycle import (
    POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
    policy_checkpoint_lifecycle_from_runtime,
)
from tgvf_rl.policy.checkpoint import PilotRunIdentityHashes
from tgvf_rl.policy.run_config import (
    POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
    RP66AdapterUpdateMode,
    load_policy_e2e_smoke_run_config,
)


_ROOT = Path(__file__).parents[2]
_CONFIG = (
    _ROOT / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_matched_8step_ws8.toml"
)
_CONFIG_WS4 = (
    _ROOT / "configs/policy/runs/"
    "prl_15_r1_qwen3_instruct_full_rp66_bs16_n16_t1_"
    "crop16_math_equiv_8step_ws4.toml"
)
_CONFIG_CANARY = (
    _ROOT / "configs/policy/runs/"
    "prl_15_c0_qwen3_instruct_full_rp66_bs4_n2_functional_canary_ws4.toml"
)


def _config():
    return load_policy_e2e_smoke_run_config(_CONFIG)


def _config_ws4():
    return load_policy_e2e_smoke_run_config(_CONFIG_WS4)


def _config_canary():
    return load_policy_e2e_smoke_run_config(_CONFIG_CANARY)


def _v2_config(tmp_path: Path, adapter_update_mode: RP66AdapterUpdateMode):
    source = _CONFIG.read_text(encoding="utf-8")
    source = source.replace(
        (f'schema_version = "{POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA}"'),
        (f'schema_version = "{POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA}"'),
        1,
    )
    source = source.replace(
        "[representation]\n",
        (f'[representation]\nadapter_update_mode = "{adapter_update_mode.value}"\n'),
        1,
    )
    path = tmp_path / f"rp66-control-{adapter_update_mode.value}.toml"
    path.write_text(source, encoding="utf-8")
    return load_policy_e2e_smoke_run_config(path)


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
    assert values["actor_rollout_ref.rollout.cudagraph_capture_sizes"] == list(
        TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES
    )
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
    assert values["actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs"] == {
        TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {"adapter_update_mode": "joint"}
    }
    assert values["actor_rollout_ref.rollout.agent.default_agent_loop"] == (
        TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
    )
    assert values["actor_rollout_ref.actor.policy_loss.loss_mode"] == (
        NATIVE_DEEPEYES_POLICY_LOSS_MODE
    )
    custom = values["actor_rollout_ref.rollout.custom"]
    assert custom["protocol"]["maximum_tool_calls"] == 6
    assert (
        custom["trainable_tgvf"]["adapter_update_mode"]
        == (plan.config.representation.adapter_update_mode.value)
        == "joint"
    )
    assert (
        custom["trainable_tgvf"]["adapter_trainable"]
        is (plan.config.representation.adapter_trainable)
        is True
    )
    assert custom["weight_sync"] == {
        "mode": plan.config.distributed.weight_sync_mode,
        "interval_optimizer_steps": 1,
        "payload": "full_qwen_plus_trainable_rp66",
    }
    assert (
        custom["reward"]["schema_version"]
        == (plan.config.schema_version)
        == POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA
    )
    assert custom["reward"]["judge_config_sha256"] == (
        plan.config.reward.judge_config_sha256
    )
    assert custom["checkpoint_steps"] == list(range(9))
    assert custom["checkpoint_lifecycle"] == {
        "schema_version": POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
        "checkpoint_steps": list(range(9)),
        "every_completed_step": True,
        "rolling_retention_across_restarts": True,
        "rolling_max_checkpoints": 2,
        "permanent_steps": [8],
        "permanent_directory": str(plan.config.output.root / "permanent-checkpoints"),
    }
    assert custom["reference_diagnostic"] == {
        "enabled": False,
        "coefficient": 0.0,
        "worker_route": "disabled_zero_kl_control",
        "observation_source": "not_computed",
    }
    assert plan.environment[POLICY_REFERENCE_DIAGNOSTIC_ENV] == "0"


@pytest.mark.parametrize(
    ("adapter_update_mode", "adapter_trainable", "payload"),
    (
        (
            RP66AdapterUpdateMode.JOINT,
            True,
            "full_qwen_plus_trainable_rp66",
        ),
        (
            RP66AdapterUpdateMode.FROZEN_ADAPTER,
            False,
            "full_qwen_plus_frozen_rp66",
        ),
    ),
)
def test_v2_plan_records_dynamic_adapter_ownership(
    tmp_path: Path,
    adapter_update_mode: RP66AdapterUpdateMode,
    adapter_trainable: bool,
    payload: str,
) -> None:
    config = _v2_config(tmp_path, adapter_update_mode)
    plan = build_trainable_tgvf_verl_launch_plan(config, mode="formal")
    custom = plan.overrides["actor_rollout_ref.rollout.custom"]

    assert plan.config.schema_version == POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA
    assert plan.config.representation.adapter_update_mode is adapter_update_mode
    assert plan.config.representation.adapter_trainable is adapter_trainable
    assert custom["trainable_tgvf"]["adapter_update_mode"] == (
        adapter_update_mode.value
    )
    assert custom["trainable_tgvf"]["adapter_trainable"] is adapter_trainable
    assert plan.overrides[
        "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs"
    ] == {
        TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {
            "adapter_update_mode": adapter_update_mode.value
        }
    }
    assert custom["weight_sync"] == {
        "mode": config.distributed.weight_sync_mode,
        "interval_optimizer_steps": 1,
        "payload": payload,
    }
    assert custom["reward"]["schema_version"] == config.schema_version
    # v2 optimizer ownership must not leave the historical Crop-16 batch path.
    assert plan.overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 32
    assert plan.overrides["actor_rollout_ref.actor.optim.total_training_steps"] == 8
    assert plan.overrides["actor_rollout_ref.rollout.cudagraph_capture_sizes"] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


def test_smoke_changes_horizon_output_and_checkpoint_not_scientific_shape() -> None:
    formal = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    smoke = build_trainable_tgvf_verl_launch_plan(_config(), mode="smoke")

    assert smoke.target_step == 1
    assert formal.overrides["trainer.logger"] == ["console", "wandb"]
    assert smoke.overrides["trainer.logger"] == ["console"]
    assert smoke.overrides["trainer.total_training_steps"] == 1
    assert smoke.overrides["trainer.default_local_dir"].endswith("/smoke/checkpoints")
    assert formal.environment[POLICY_METRICS_PATH_ENV].endswith("/metrics.jsonl")
    assert "/smoke/" not in formal.environment[POLICY_METRICS_PATH_ENV]
    assert smoke.environment[POLICY_METRICS_PATH_ENV].endswith("/smoke/metrics.jsonl")
    assert smoke.overrides["actor_rollout_ref.rollout.custom"]["checkpoint_steps"] == [
        0,
        1,
    ]
    assert smoke.overrides["actor_rollout_ref.rollout.custom"][
        "checkpoint_lifecycle"
    ] == {
        "schema_version": POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
        "checkpoint_steps": [0, 1],
        "every_completed_step": False,
        "rolling_retention_across_restarts": True,
        "rolling_max_checkpoints": 2,
        "permanent_steps": [],
        "permanent_directory": "",
    }
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
    assert (
        world4.overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 32
    )
    assert world4.overrides["actor_rollout_ref.actor.optim.total_training_steps"] == 8
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


def test_functional_canary_is_an_isolated_low_cost_full_path_launch() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config_canary(), mode="canary")
    values = plan.overrides

    assert plan.target_step == values["trainer.total_training_steps"] == 1
    assert values["data.train_files"] == [str(DEEPEYES_SMOKE_SENTINEL)]
    assert values["data.val_files"] == [str(DEEPEYES_SMOKE_SENTINEL)]
    assert values["data.train_batch_size"] == 4
    assert values["data.gen_batch_size"] == 4
    assert plan.config.policy.sampling.max_response_length == 512
    assert values["data.max_response_length"] == 8192
    assert values["actor_rollout_ref.rollout.n"] == 2
    assert values["actor_rollout_ref.rollout.response_length"] == 8192
    assert values["actor_rollout_ref.rollout.response_length"] > (
        plan.config.policy.sampling.max_response_length
    )
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == 4
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 2
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 2
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == 2
    assert values["trainer.n_gpus_per_node"] == 4
    assert values["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 4
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert values["trainer.logger"] == ["console"]
    assert values["trainer.default_local_dir"].endswith("/canary/checkpoints")
    assert plan.environment[POLICY_METRICS_PATH_ENV].endswith("/canary/metrics.jsonl")
    assert plan.environment["TGVF_POLICY_STATE_DIR"].endswith(
        "/canary/runtime-policy-state"
    )
    assert plan.environment[POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV] == "1"
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is True
    assert values["actor_rollout_ref.model.use_remove_padding"] is True
    assert values["actor_rollout_ref.model.use_fused_kernels"] is True

    custom = values["actor_rollout_ref.rollout.custom"]
    assert custom["checkpoint_steps"] == [0, 1]
    assert custom["checkpoint_lifecycle"]["every_completed_step"] is False
    assert custom["checkpoint_lifecycle"]["permanent_steps"] == []
    assert custom["functional_canary"] == {
        "minimum_successful_tgvf_observations": 1,
        "failure_boundary": "before_optimizer_mutation",
        "dataset_split": "smoke",
    }
    assert custom["actor_batch_contract"] == {
        "global_prompt_batch_size": 4,
        "rollouts_per_prompt": 2,
        "fsdp_data_parallel_size": 4,
        "prompt_micro_batch_size_per_rank": 1,
        "configured_gradient_accumulation_steps": 1,
        "upstream_ppo_mini_batch_size_prompts": 4,
        "upstream_internal_mini_batch_size_trajectories": 8,
        "upstream_ppo_micro_batch_size_per_gpu_trajectories": 2,
        "upstream_inference_micro_batch_size_per_gpu_trajectories": 2,
        "derived_actor_forward_backward_microbatches": 1,
        "derived_gradient_accumulation_steps": 1,
        "optimizer_steps_per_trainer_step": 1,
    }

    composed = compose_trainable_tgvf_verl_config(plan)
    assert composed.data.train_batch_size == 4
    assert composed.actor_rollout_ref.rollout.n == 2
    assert composed.trainer.total_training_steps == 1


def test_composed_formal_lifecycle_is_every_step_and_permanently_keeps_step8() -> None:
    plan = build_trainable_tgvf_verl_launch_plan(_config(), mode="formal")
    composed = compose_trainable_tgvf_verl_config(plan)
    identity = PilotRunIdentityHashes.from_hashes(
        plan.config.run_id, {"test": "0" * 64}
    )

    lifecycle = policy_checkpoint_lifecycle_from_runtime(
        composed,
        run_identity=identity,
        world_size=8,
    )

    assert lifecycle is not None
    assert lifecycle.checkpoint_steps == tuple(range(9))
    assert lifecycle.every_completed_step is True
    assert lifecycle.permanent_steps == (8,)


def test_functional_canary_cannot_silently_shrink_a_matched_control_config() -> None:
    with pytest.raises(ValueError, match="functional canary run config differs"):
        build_trainable_tgvf_verl_launch_plan(_config_ws4(), mode="canary")


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
        POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV: "1",
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
    assert POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV not in observed
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
