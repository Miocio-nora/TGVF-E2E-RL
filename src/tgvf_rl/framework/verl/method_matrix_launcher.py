"""Thin method-matrix overlay for the shared Policy veRL plan.

The generic launcher remains the single owner of dataset binding, rollout,
reward, exact-action transport, and process environment.  This module changes
only the scientific treatment surface: full-Qwen engine registration, DeepEyes
actor reduction, config-derived batch geometry, and method-specific checkpoint
publication.  Resolution, horizon, sampling width, seed, and capacity remain
ordinary run-config inputs.  Original Qwen is an evaluation baseline and
intentionally has no training route here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tgvf_rl.policy.config import PolicyMethodProfile
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.protocol import NativeToolCapabilityProfile

from .launcher import UpstreamVerlLaunchPlan
from .full_qwen_checkpoint_manager import (
    FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from .trainable_crop_engine import TRAINABLE_CROP_MODEL_TYPE
from .trainable_tgvf_checkpoint_manager import (
    TGVF_CHECKPOINT_ENGINE_CONTROL_KEY,
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from .trainable_tgvf_engine import TRAINABLE_TGVF_MODEL_TYPE


POLICY_METHOD_MATRIX_SCHEMA = "tgvf.policy-method-matrix.v1"
# Compatibility name for callers written while the first @512 matrix was
# being reconstructed.  The schema itself is deliberately resolution-agnostic.
TEACHER25_METHOD_MATRIX_SCHEMA = POLICY_METHOD_MATRIX_SCHEMA
POLICY_ORIGINAL_EVAL_BASELINE = "original_qwen_eval_only"
TEACHER25_ORIGINAL_EVAL_BASELINE = POLICY_ORIGINAL_EVAL_BASELINE
TRAINABLE_CROP_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_crop_external"
TRAINABLE_TGVF_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_tgvf_external"

_NATIVE_QWEN_METHODS = frozenset(
    {
        PolicyMethodProfile.NO_TOOL,
        PolicyMethodProfile.CROP,
    }
)
_EXPECTED_TOOL_PROFILE = {
    PolicyMethodProfile.NO_TOOL: NativeToolCapabilityProfile.NO_TOOL,
    PolicyMethodProfile.CROP: NativeToolCapabilityProfile.CROP_ONLY,
    PolicyMethodProfile.TGVF_SHORT: NativeToolCapabilityProfile.TGVF_ONLY,
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2: (NativeToolCapabilityProfile.TGVF_ONLY),
    PolicyMethodProfile.ATOMIC: NativeToolCapabilityProfile.CROP_TGVF,
}
_TOOL_RUNTIME = {
    PolicyMethodProfile.NO_TOOL: "direct_only_no_tool",
    PolicyMethodProfile.CROP: "crop_only",
    PolicyMethodProfile.TGVF_SHORT: "tgvf_only",
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2: "tgvf_only",
    PolicyMethodProfile.ATOMIC: "atomic_crop_tgvf",
}


def route_policy_method_matrix_plan(
    config: PolicyE2ESmokeRunConfig,
    base_plan: UpstreamVerlLaunchPlan,
) -> UpstreamVerlLaunchPlan:
    """Select the method overlay only for an explicit method binding."""

    # Legacy callers (and preflight test doubles) predate the optional method
    # binding.  Only an explicitly bound matrix config receives an overlay;
    # every other object keeps the generic plan unchanged.
    if getattr(config, "method", None) is None:
        return base_plan
    return build_policy_method_matrix_plan(
        config,
        base_plan=base_plan,
        method=config.method.profile,
    )


route_teacher25_method_matrix_plan = route_policy_method_matrix_plan


def build_policy_method_matrix_plan(
    config: PolicyE2ESmokeRunConfig,
    *,
    base_plan: UpstreamVerlLaunchPlan,
    method: PolicyMethodProfile | None = None,
) -> UpstreamVerlLaunchPlan:
    """Overlay one NoTool/Crop/TGVF/Atomic arm on the generic veRL plan."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("method matrix requires PolicyE2ESmokeRunConfig")
    if not isinstance(base_plan, UpstreamVerlLaunchPlan):
        raise TypeError("base_plan must be UpstreamVerlLaunchPlan")
    if config.method is None:
        raise ValueError("method matrix requires config.method")
    resolved = config.method.profile
    if method is not None and method is not resolved:
        raise ValueError("explicit method differs from the run-config schema")
    if base_plan.run_identity_sha256 != config.identity_sha256:
        raise ValueError("method-matrix base plan and run identity differ")
    _assert_method_controls(config, method=resolved)

    native_qwen = resolved in _NATIVE_QWEN_METHODS
    external_module = (
        TRAINABLE_CROP_EXTERNAL_MODULE
        if native_qwen
        else TRAINABLE_TGVF_EXTERNAL_MODULE
    )
    model_type = TRAINABLE_CROP_MODEL_TYPE if native_qwen else TRAINABLE_TGVF_MODEL_TYPE
    current_replay_port = _current_replay_port(resolved)
    engine_registrar = _engine_registrar(resolved)
    sampling = config.policy.sampling
    accumulation = config.accumulation
    prompt_micro = accumulation.prompt_micro_batch_size_per_rank
    rollout_prompt_micro = accumulation.rollout_prompt_micro_batch_size_per_engine
    actor_trajectory_micro = prompt_micro * sampling.trajectories_per_prompt
    rollout_trajectory_micro = rollout_prompt_micro * sampling.trajectories_per_prompt
    values = dict(base_plan.overrides)
    values.update(
        {
            "actor_rollout_ref.model.external_lib": external_module,
            "actor_rollout_ref.model.model_type": model_type,
            "actor_rollout_ref.model.lora_rank": 0,
            "actor_rollout_ref.model.lora.rank": 0,
            "actor_rollout_ref.model.lora.freeze_vision_model": False,
            "actor_rollout_ref.model.lora.freeze_vision_projection": False,
            "actor_rollout_ref.model.lora.freeze_language_model": False,
            "actor_rollout_ref.actor.freeze_vision_tower": False,
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.actor.loss_agg_mode": (NATIVE_DEEPEYES_LOSS_AGG_MODE),
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
            "actor_rollout_ref.actor.ppo_epochs": 1,
            "actor_rollout_ref.actor.shuffle": False,
            "actor_rollout_ref.actor.entropy_coeff": 0.0,
            "actor_rollout_ref.actor.use_kl_loss": False,
            "actor_rollout_ref.actor.clip_ratio": 0.2,
            "actor_rollout_ref.actor.clip_ratio_low": 0.2,
            "actor_rollout_ref.actor.clip_ratio_high": 0.2,
            "actor_rollout_ref.actor.clip_ratio_c": 3.0,
            "data.mm_processor_kwargs.max_pixels": config.policy.image_max_pixels,
            "data.train_batch_size": accumulation.global_prompt_batch_size,
            "data.gen_batch_size": accumulation.global_prompt_batch_size,
            "actor_rollout_ref.actor.ppo_mini_batch_size": (
                accumulation.global_prompt_batch_size
            ),
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": (
                actor_trajectory_micro
            ),
            "actor_rollout_ref.rollout.n": sampling.trajectories_per_prompt,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": (
                rollout_trajectory_micro
            ),
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (
                actor_trajectory_micro
            ),
        }
    )
    manager_path = "actor_rollout_ref.rollout.checkpoint_manager_class"
    manager_kwargs_path = "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs"
    adapter_update_mode: str | None = None
    if native_qwen:
        values[manager_path] = FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN
        values.pop(manager_kwargs_path, None)
        checkpoint_manager = FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN
        weight_payload = "full_qwen"
    else:
        adapter_update_mode = config.representation.adapter_update_mode.value
        values[manager_path] = TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
        values[manager_kwargs_path] = {
            TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {
                "adapter_update_mode": adapter_update_mode
            }
        }
        checkpoint_manager = TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
        weight_payload = f"full_qwen_plus_{adapter_update_mode}_rp66"

    values["actor_rollout_ref.rollout.custom"] = _method_custom_record(
        values["actor_rollout_ref.rollout.custom"],
        method=resolved,
        external_module=external_module,
        model_type=model_type,
        checkpoint_manager=checkpoint_manager,
        weight_sync_mode=config.distributed.weight_sync_mode,
        weight_sync_interval_optimizer_steps=(
            config.distributed.weight_sync_interval_optimizer_steps
        ),
        native_deepstack_enabled=config.policy.native_deepstack_enabled,
        weight_payload=weight_payload,
        current_replay_port=current_replay_port,
        engine_registrar=engine_registrar,
        matrix_id=config.method.matrix_id,
        image_max_pixels=config.policy.image_max_pixels,
        global_prompt_batch_size=accumulation.global_prompt_batch_size,
        rollouts_per_prompt=sampling.trajectories_per_prompt,
        world_size=config.distributed.world_size,
        maximum_optimizer_steps=_positive_plan_integer(
            values.get("trainer.total_training_steps"),
            path="trainer.total_training_steps",
        ),
        prompt_micro_batch_size_per_rank=prompt_micro,
        rollout_prompt_micro_batch_size_per_engine=rollout_prompt_micro,
        gradient_accumulation_steps=accumulation.gradient_accumulation_steps,
        adapter_update_mode=adapter_update_mode,
    )
    external_components = dict(base_plan.external_components)
    external_components.update(
        {
            "method_matrix_profile": resolved.value,
            "method_matrix_original_baseline": POLICY_ORIGINAL_EVAL_BASELINE,
            "actor_engine": model_type,
            "actor_external_lib": external_module,
            "actor_loss": NATIVE_DEEPEYES_POLICY_LOSS_MODE,
            "checkpoint_engine_manager": checkpoint_manager,
            "exact_replay_records": "rollout_materialized_exact_bundle",
            "exact_replay_registration": engine_registrar,
            "exact_replay_forward": current_replay_port,
            "adapter_update_mode": adapter_update_mode or "unused",
        }
    )
    return replace(
        base_plan,
        overrides=values,
        external_components=external_components,
    )


# Compatibility facade for the first Teacher25/@512 matrix.  New callers use
# the resolution- and dataset-agnostic name above.
build_teacher25_method_matrix_plan = build_policy_method_matrix_plan


def _assert_method_controls(
    config: PolicyE2ESmokeRunConfig,
    *,
    method: PolicyMethodProfile,
) -> None:
    if config.method is None:
        raise ValueError("method matrix requires config.method")
    expected = {
        "protocol.tool_profile": (
            config.protocol.tool_profile,
            _EXPECTED_TOOL_PROFILE[method],
        ),
        "policy.method": (getattr(config.policy, "method", None), method),
        "distributed.visible_gpu_count": (
            len(config.distributed.physical_gpu_ids),
            config.distributed.world_size,
        ),
    }
    expected_weight_sync = (
        "nccl_full_qwen_v1"
        if method in _NATIVE_QWEN_METHODS
        else "nccl_full_qwen_plus_trainable_rp66_v1"
    )
    expected["distributed.weight_sync_mode"] = (
        config.distributed.weight_sync_mode,
        expected_weight_sync,
    )
    expected["distributed.weight_sync_interval_optimizer_steps"] = (
        config.distributed.weight_sync_interval_optimizer_steps,
        1,
    )
    mismatches = {
        name: (actual, required)
        for name, (actual, required) in expected.items()
        if actual != required
    }
    if mismatches:
        raise ValueError(f"method config differs: mismatches={mismatches!r}")


def _method_custom_record(
    value: object,
    *,
    method: PolicyMethodProfile,
    external_module: str,
    model_type: str,
    checkpoint_manager: str,
    weight_sync_mode: str,
    weight_sync_interval_optimizer_steps: int,
    native_deepstack_enabled: bool,
    weight_payload: str,
    current_replay_port: str,
    engine_registrar: str,
    matrix_id: str,
    image_max_pixels: int,
    global_prompt_batch_size: int,
    rollouts_per_prompt: int,
    world_size: int,
    maximum_optimizer_steps: int,
    prompt_micro_batch_size_per_rank: int,
    rollout_prompt_micro_batch_size_per_engine: int,
    gradient_accumulation_steps: int,
    adapter_update_mode: str | None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("generic Policy plan lost its custom replay record")
    custom = dict(value)
    exact_replay = dict(custom.get("exact_replay", {}))
    exact_replay.update(
        {
            "model_type": model_type,
            "registration_module": external_module,
            "engine_registrar_fqn": engine_registrar,
            "current_forward_port_fqn": current_replay_port,
            "record_source": "rollout_materialized_exact_bundle",
            "sampled_actions": "immutable_behavior_token_ids",
            "current_visual_source": "live_preprocessed_pixels",
            "reference_visual_source": "recorded_features",
        }
    )
    custom.update(
        {
            "method_matrix": {
                "schema_version": POLICY_METHOD_MATRIX_SCHEMA,
                "matrix_id": matrix_id,
                "profile": method.value,
                "original_baseline": POLICY_ORIGINAL_EVAL_BASELINE,
                "training_enabled": True,
                "image_max_pixels": image_max_pixels,
                "global_prompt_batch_size": global_prompt_batch_size,
                "rollouts_per_prompt": rollouts_per_prompt,
                "world_size": world_size,
                "maximum_optimizer_steps": maximum_optimizer_steps,
                "checkpoint_manager": checkpoint_manager,
                "tool_runtime": _TOOL_RUNTIME[method],
                "representation": (
                    "unused" if method in _NATIVE_QWEN_METHODS else adapter_update_mode
                ),
                "full_qwen_trainable": True,
                "native_deepstack_enabled": native_deepstack_enabled,
            },
            "exact_replay": exact_replay,
            "weight_sync": {
                "mode": weight_sync_mode,
                "interval_optimizer_steps": weight_sync_interval_optimizer_steps,
                "payload": weight_payload,
            },
            "actor_batch_contract": {
                "global_prompt_batch_size": global_prompt_batch_size,
                "rollouts_per_prompt": rollouts_per_prompt,
                "fsdp_data_parallel_size": world_size,
                "prompt_micro_batch_size_per_rank": (prompt_micro_batch_size_per_rank),
                "rollout_prompt_micro_batch_size_per_engine": (
                    rollout_prompt_micro_batch_size_per_engine
                ),
                "configured_gradient_accumulation_steps": (gradient_accumulation_steps),
                "upstream_ppo_mini_batch_size_prompts": global_prompt_batch_size,
                "upstream_internal_mini_batch_size_trajectories": (
                    global_prompt_batch_size * rollouts_per_prompt
                ),
                "upstream_ppo_micro_batch_size_per_gpu_trajectories": (
                    prompt_micro_batch_size_per_rank * rollouts_per_prompt
                ),
                "upstream_inference_micro_batch_size_per_gpu_trajectories": (
                    prompt_micro_batch_size_per_rank * rollouts_per_prompt
                ),
                "derived_actor_forward_backward_microbatches": (
                    gradient_accumulation_steps
                ),
                "derived_gradient_accumulation_steps": (gradient_accumulation_steps),
                "optimizer_steps_per_trainer_step": 1,
            },
        }
    )
    return custom


def _current_replay_port(method: PolicyMethodProfile) -> str:
    return (
        "tgvf_rl.policy.trainable_crop_replay.TrainableCropCurrentReplayPort"
        if method in _NATIVE_QWEN_METHODS
        else "tgvf_rl.policy.trainable_tgvf_replay.TrainableTGVFCurrentReplayPort"
    )


def _engine_registrar(method: PolicyMethodProfile) -> str:
    return (
        "tgvf_rl.framework.verl.trainable_crop_engine."
        "register_trainable_crop_fsdp2_engine"
        if method in _NATIVE_QWEN_METHODS
        else "tgvf_rl.framework.verl.trainable_tgvf_engine."
        "register_trainable_tgvf_fsdp2_engine"
    )


def _positive_plan_integer(value: object, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"method base plan requires positive {path}")
    return value


__all__ = [
    "POLICY_METHOD_MATRIX_SCHEMA",
    "POLICY_ORIGINAL_EVAL_BASELINE",
    "TEACHER25_METHOD_MATRIX_SCHEMA",
    "TEACHER25_ORIGINAL_EVAL_BASELINE",
    "TRAINABLE_CROP_EXTERNAL_MODULE",
    "TRAINABLE_TGVF_EXTERNAL_MODULE",
    "build_policy_method_matrix_plan",
    "build_teacher25_method_matrix_plan",
    "route_policy_method_matrix_plan",
    "route_teacher25_method_matrix_plan",
]
