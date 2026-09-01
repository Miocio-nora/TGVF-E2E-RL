from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from tgvf_rl.framework.verl.launcher import (
    _checkpoint_frequency,
    _vllm_rollout_replica_count,
    build_policy_e2e_smoke_verl_plan,
    compose_upstream_verl_config,
)
from tgvf_rl.framework.verl.compatibility import (
    VerlConfigurationError,
    validate_verl_config_mapping,
)
from tgvf_rl.framework.verl.full_qwen_checkpoint_manager import (
    FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from tgvf_rl.framework.verl.dynamic_token_loss_contract import (
    DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE,
    METHOD_MATRIX_BYPASS_LOSS_MODULE,
    METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME,
)
from tgvf_rl.framework.verl.method_matrix_launcher import (
    POLICY_ORIGINAL_EVAL_BASELINE,
    TRAINABLE_CROP_EXTERNAL_MODULE,
    TRAINABLE_TGVF_EXTERNAL_MODULE,
    build_policy_method_matrix_plan,
    route_policy_method_matrix_plan,
)
from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from tgvf_rl.framework.verl.trainable_crop_engine import (
    TRAINABLE_CROP_MODEL_TYPE,
)
from tgvf_rl.framework.verl.trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from tgvf_rl.framework.verl.trainable_tgvf_engine import (
    TRAINABLE_TGVF_MODEL_TYPE,
)
from tgvf_rl.policy import dev_launch
from tgvf_rl.policy.config import PolicyMethodExperimentConfig, PolicyMethodProfile
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.deepeyes_official_protocol import VISUAL_PROMPT_IDENTITY
from tgvf_rl.policy.no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from tgvf_rl.policy.method_matrix_validation import (
    validate_policy_method_matrix,
)
from tgvf_rl.policy.run_config import (
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import (
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)
from tests.policy.test_method_run_config import (
    method_config_factory as _policy_method_config_factory,
)
from tests.policy.test_run_config import (
    _write_config,
    _write_minimal_upstream_config_directory,
)


def test_periodic_checkpoint_plan_uses_ten_step_upstream_gate() -> None:
    assert _checkpoint_frequency((0, 10, 20, 30, 40, 50), maximum_step=50) == 10


def test_nonuniform_checkpoint_plan_uses_exact_schedule_gcd_gate() -> None:
    assert _checkpoint_frequency((0, 10, 20, 45, 80), maximum_step=80) == 5


MethodConfigFactory = Callable[..., tuple[Path, str]]
_METHOD_CASES = (
    (
        PolicyMethodProfile.NO_TOOL,
        NativeToolCapabilityProfile.NO_TOOL,
        NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1,
        True,
    ),
    (
        PolicyMethodProfile.CROP,
        NativeToolCapabilityProfile.CROP_ONLY,
        VISUAL_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1,
        True,
    ),
    (
        PolicyMethodProfile.TGVF_SHORT,
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
        False,
    ),
    (
        PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1,
        False,
    ),
    (
        PolicyMethodProfile.ATOMIC,
        NativeToolCapabilityProfile.CROP_TGVF,
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1,
        False,
    ),
)

_CANONICAL_METHOD_MATRIX_CONFIGS = (
    (
        PolicyMethodProfile.NO_TOOL,
        "prl_28_a_qwen3_instruct_no_tool_pixel512_s32_bs16_n16_teacher25_ws8.toml",
        1,
    ),
    (
        PolicyMethodProfile.CROP,
        "prl_28_b_qwen3_instruct_crop_pixel512_s32_bs16_n16_teacher25_ws8.toml",
        6,
    ),
    (
        PolicyMethodProfile.TGVF_SHORT,
        ("prl_28_c_qwen3_instruct_tgvf_short_pixel512_s32_bs16_n16_teacher25_ws8.toml"),
        6,
    ),
    (
        PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
        (
            "prl_28_d_qwen3_instruct_tgvf_target_guide_v2_pixel512_s32_"
            "bs16_n16_teacher25_ws8.toml"
        ),
        6,
    ),
    (
        PolicyMethodProfile.ATOMIC,
        "prl_28_e_qwen3_instruct_atomic_pixel512_s32_bs16_n16_teacher25_ws8.toml",
        6,
    ),
)


@pytest.fixture(name="method_config_factory")
def _method_config_factory_fixture(tmp_path: Path) -> MethodConfigFactory:
    return _policy_method_config_factory.__wrapped__(tmp_path)


def _load_method_config(
    factory: MethodConfigFactory,
    case_index: int,
    **changes: object,
) -> tuple[PolicyE2ESmokeRunConfig, bool]:
    profile, tool_profile, prompt_sha256, observation_id, native_qwen = _METHOD_CASES[
        case_index
    ]
    path, _ = factory(
        profile=profile,
        tool_profile=tool_profile,
        prompt_sha256=prompt_sha256,
        observation_id=observation_id,
        **changes,
    )
    config = load_policy_e2e_smoke_run_config(path)
    assert config.method is not None
    assert config.method.profile is profile
    assert isinstance(config.policy, PolicyMethodExperimentConfig)
    return config, native_qwen


@pytest.mark.parametrize("case_index", range(len(_METHOD_CASES)))
def test_explicit_method_binding_selects_engine_and_checkpoint_publication(
    method_config_factory: MethodConfigFactory,
    case_index: int,
) -> None:
    config, native_qwen = _load_method_config(method_config_factory, case_index)
    assert config.method is not None
    profile = config.method.profile.value
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    values = plan.overrides

    assert plan.external_components["method_matrix_profile"] == profile
    assert (
        plan.external_components["method_matrix_original_baseline"]
        == POLICY_ORIGINAL_EVAL_BASELINE
    )
    assert values["data.mm_processor_kwargs.max_pixels"] == 345_678
    assert (
        values["data.train_batch_size"] == config.accumulation.global_prompt_batch_size
    )
    assert values["actor_rollout_ref.rollout.n"] == 3
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 3
    assert values["trainer.n_gpus_per_node"] == config.distributed.world_size == 4
    assert values["trainer.total_training_steps"] == 2
    assert (
        values["actor_rollout_ref.actor.policy_loss.loss_mode"]
        == NATIVE_DEEPEYES_POLICY_LOSS_MODE
    )
    assert values["actor_rollout_ref.model.lora_rank"] == 0
    assert values["actor_rollout_ref.actor.freeze_vision_tower"] is False
    custom = values["actor_rollout_ref.rollout.custom"]
    assert custom["method_matrix"] == {
        **custom["method_matrix"],
        "matrix_id": config.method.matrix_id,
        "profile": profile,
        "image_max_pixels": 345_678,
        "rollouts_per_prompt": 3,
        "maximum_optimizer_steps": 2,
    }
    assert custom["method_matrix"]["representation"] == (
        "unused" if native_qwen else "frozen_adapter"
    )
    assert custom["method_matrix"]["full_qwen_trainable"] is True
    assert custom["method_matrix"]["native_deepstack_enabled"] is True
    assert custom["weight_sync"]["interval_optimizer_steps"] == (
        config.distributed.weight_sync_interval_optimizer_steps
    )
    assert custom["exact_replay"]["record_source"] == (
        "rollout_materialized_exact_bundle"
    )
    assert (
        plan.external_components["exact_replay_registration"]
        == custom["exact_replay"]["engine_registrar_fqn"]
    )
    assert (
        plan.external_components["exact_replay_forward"]
        == custom["exact_replay"]["current_forward_port_fqn"]
    )

    manager_path = "actor_rollout_ref.rollout.checkpoint_manager_class"
    manager_kwargs = "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs"
    if native_qwen:
        assert values["actor_rollout_ref.model.external_lib"] == (
            TRAINABLE_CROP_EXTERNAL_MODULE
        )
        assert values["actor_rollout_ref.model.model_type"] == (
            TRAINABLE_CROP_MODEL_TYPE
        )
        assert values[manager_path] == FULL_QWEN_CHECKPOINT_ENGINE_MANAGER_FQN
        assert manager_kwargs not in values
        assert custom["weight_sync"]["payload"] == "full_qwen"
    else:
        assert values["actor_rollout_ref.model.external_lib"] == (
            TRAINABLE_TGVF_EXTERNAL_MODULE
        )
        assert values["actor_rollout_ref.model.model_type"] == (
            TRAINABLE_TGVF_MODEL_TYPE
        )
        assert values[manager_path] == TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
        assert values[manager_kwargs] == {
            "tgvf_control": {"adapter_update_mode": "frozen_adapter"}
        }
        assert custom["weight_sync"]["payload"] == (
            "full_qwen_plus_frozen_adapter_rp66"
        )


@pytest.mark.parametrize("native_deepstack_enabled", (True, False))
def test_method_plan_binds_one_native_deepstack_control_to_actor_and_rollout(
    method_config_factory: MethodConfigFactory,
    native_deepstack_enabled: bool,
) -> None:
    config, _ = _load_method_config(
        method_config_factory,
        2,
        native_deepstack_enabled=native_deepstack_enabled,
    )
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    values = plan.overrides

    assert (
        values["actor_rollout_ref.model.override_config.tgvf_native_deepstack_enabled"]
        is native_deepstack_enabled
    )
    assert (
        values["actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"][
            "tgvf_native_deepstack_enabled"
        ]
        is native_deepstack_enabled
    )
    assert (
        values["actor_rollout_ref.rollout.custom"]["method_matrix"][
            "native_deepstack_enabled"
        ]
        is native_deepstack_enabled
    )


def test_method_v2_propagates_optimized_dynamic_surface_and_composes(
    method_config_factory: MethodConfigFactory,
    tmp_path: Path,
) -> None:
    config, _ = _load_method_config(
        method_config_factory,
        1,
        performance_v2=True,
    )
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    values = plan.overrides

    assert values["actor_rollout_ref.actor.use_dynamic_bsz"] is True
    assert values["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"] is True
    assert values["actor_rollout_ref.ref.log_prob_use_dynamic_bsz"] is True
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] is None
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] is None
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] is None
    assert values["actor_rollout_ref.actor.policy_loss.loss_mode"] == (
        DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE
    )
    assert values["actor_rollout_ref.model.use_remove_padding"] is True
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is True
    assert values["actor_rollout_ref.rollout.enable_prefix_caching"] is True
    assert values["actor_rollout_ref.rollout.enable_chunked_prefill"] is True
    assert values["actor_rollout_ref.rollout.enforce_eager"] is False
    assert values["actor_rollout_ref.rollout.cudagraph_capture_sizes"] == [1, 2, 4, 8]
    assert values["actor_rollout_ref.rollout.tensor_model_parallel_size"] == 1
    assert values["actor_rollout_ref.rollout.data_parallel_size"] == 1
    assert values["actor_rollout_ref.rollout.pipeline_model_parallel_size"] == 1
    assert values["actor_rollout_ref.rollout.disaggregation.enabled"] is False
    assert values["actor_rollout_ref.rollout.calculate_log_probs"] is True
    assert (
        values["actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode"]
        is True
    )
    assert values["algorithm.rollout_correction.bypass_mode"] is True
    assert plan.external_components["actor_loss"] == (
        DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE
    )
    assert plan.external_components["actor_execution_loss"] == (
        METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME
    )
    assert plan.external_components["actor_execution_loss_module"] == (
        METHOD_MATRIX_BYPASS_LOSS_MODULE
    )
    performance = values["actor_rollout_ref.rollout.custom"]["performance"]
    assert performance["source_schema_version"] == config.schema_version
    assert performance["prefix_cache_identity_basis"] == (
        "vllm-0.12-block-hash-mm-feature-identifier"
    )
    assert performance["prefix_cache_invalidation"] == (
        "verl-clear-kv-cache-after-weight-update"
    )
    assert performance["judge_max_concurrency_per_worker"] == 8
    assert performance["reference_replay_mode"] == "off"
    assert (
        values["actor_rollout_ref.rollout.custom"]["reference_diagnostic"]["enabled"]
        is False
    )
    assert performance["judge_concurrency_scope"] == ("agent_loop_worker_process_local")
    assert performance["vllm_data_parallel_size"] == 1
    assert performance["vllm_pipeline_parallel_size"] == 1
    assert performance["vllm_disaggregation_enabled"] is False
    assert performance["derived_vllm_rollout_replicas"] == (
        config.distributed.world_size
        // (
            config.performance.vllm_tensor_parallel_size
            * performance["vllm_data_parallel_size"]
            * performance["vllm_pipeline_parallel_size"]
        )
    )
    drifted_values = dict(values)
    drifted_custom = dict(values["actor_rollout_ref.rollout.custom"])
    drifted_performance = dict(performance)
    drifted_performance["derived_vllm_rollout_replicas"] += 1
    drifted_custom["performance"] = drifted_performance
    drifted_values["actor_rollout_ref.rollout.custom"] = drifted_custom
    with pytest.raises(ValueError, match="rollout replica receipt differs"):
        replace(plan, overrides=drifted_values)
    drifted_topology = dict(values)
    drifted_topology["actor_rollout_ref.rollout.data_parallel_size"] = 2
    with pytest.raises(ValueError, match="performance override drift"):
        replace(plan, overrides=drifted_topology)
    actor_batch = values["actor_rollout_ref.rollout.custom"]["actor_batch_contract"]
    assert actor_batch["dynamic_token_batching"] is True
    assert actor_batch["upstream_ppo_micro_batch_size_per_gpu_trajectories"] is None
    assert values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == (
        config.capacity.actor_ppo_max_token_len_per_gpu
    )
    assert values["actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"] == (
        config.capacity.rollout_log_prob_max_token_len_per_gpu
    )
    assert values["actor_rollout_ref.ref.log_prob_max_token_len_per_gpu"] == (
        config.capacity.reference_log_prob_max_token_len_per_gpu
    )

    composed = compose_upstream_verl_config(
        plan,
        config_directory=_write_minimal_upstream_config_directory(tmp_path),
    )
    assert composed.actor_rollout_ref.actor.use_dynamic_bsz is True
    assert composed.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz is True
    assert composed.actor_rollout_ref.ref.log_prob_use_dynamic_bsz is True
    assert composed.actor_rollout_ref.rollout.cudagraph_capture_sizes == [1, 2, 4, 8]
    assert composed.actor_rollout_ref.rollout.data_parallel_size == 1
    assert composed.actor_rollout_ref.rollout.pipeline_model_parallel_size == 1
    assert composed.actor_rollout_ref.rollout.disaggregation.enabled is False
    validate_verl_config_mapping(
        composed,
        expected_world_size=config.distributed.world_size,
    )
    missing_invalidation = OmegaConf.to_container(composed, resolve=True)
    del missing_invalidation["actor_rollout_ref"]["rollout"]["custom"]["performance"][
        "prefix_cache_invalidation"
    ]
    with pytest.raises(
        VerlConfigurationError,
        match="multimodal hash identity and weight-update invalidation",
    ):
        validate_verl_config_mapping(
            missing_invalidation,
            expected_world_size=config.distributed.world_size,
        )


def test_vllm_rollout_replica_count_derives_ws8_tp1_without_hardcoding() -> None:
    assert (
        _vllm_rollout_replica_count(
            world_size=8,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            disaggregation_enabled=False,
        )
        == 8
    )
    assert (
        _vllm_rollout_replica_count(
            world_size=8,
            tensor_parallel_size=1,
            data_parallel_size=2,
            pipeline_parallel_size=1,
            disaggregation_enabled=False,
        )
        == 4
    )
    with pytest.raises(ValueError, match="footprint must divide world size"):
        _vllm_rollout_replica_count(
            world_size=8,
            tensor_parallel_size=3,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            disaggregation_enabled=False,
        )
    with pytest.raises(ValueError, match="explicit prefill/decode topology"):
        _vllm_rollout_replica_count(
            world_size=8,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            disaggregation_enabled=True,
        )


def test_original_remains_an_eval_baseline_without_a_training_route(
    tmp_path: Path,
) -> None:
    path, _text, _external = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    base = build_policy_e2e_smoke_verl_plan(config)

    assert route_policy_method_matrix_plan(config, base) is base
    assert "method_matrix_profile" not in base.external_components
    with pytest.raises(ValueError, match="requires config.method"):
        build_policy_method_matrix_plan(config, base_plan=base)


@pytest.mark.parametrize("case_index", (1, 4))
def test_method_overlay_cpu_composes_through_existing_plan_interface(
    method_config_factory: MethodConfigFactory,
    tmp_path: Path,
    case_index: int,
) -> None:
    config, native_qwen = _load_method_config(method_config_factory, case_index)
    assert config.method is not None
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    composed = compose_upstream_verl_config(
        plan,
        config_directory=_write_minimal_upstream_config_directory(tmp_path),
    )

    assert composed.actor_rollout_ref.model.model_type == (
        TRAINABLE_CROP_MODEL_TYPE if native_qwen else TRAINABLE_TGVF_MODEL_TYPE
    )
    assert composed.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 3
    assert (
        composed.actor_rollout_ref.rollout.custom.method_matrix.profile
        == config.method.profile.value
    )
    assert (
        composed.actor_rollout_ref.rollout.custom.method_matrix.image_max_pixels
        == 345_678
    )
    assert composed.actor_rollout_ref.rollout.custom.exact_replay.record_source == (
        "rollout_materialized_exact_bundle"
    )


def test_dev_launch_routes_explicit_method_without_new_gate_logic(
    method_config_factory: MethodConfigFactory,
) -> None:
    config, _native_qwen = _load_method_config(method_config_factory, 4)
    assert config.method is not None

    prepared = dev_launch.prepare_policy_dev_launch(
        config,
        python_executable=Path(sys.executable).absolute(),
        host_environment={},
    )

    assert (
        prepared.plan.external_components["method_matrix_profile"]
        == config.method.profile.value
    )
    assert any(
        argument.startswith("++actor_rollout_ref.model.model_type=")
        and TRAINABLE_TGVF_MODEL_TYPE in argument
        for argument in prepared.command
    )
    assert (
        prepared.environment[dev_launch.POLICY_EXECUTION_PROFILE_ENVIRONMENT]
        == dev_launch.POLICY_DEV_EXECUTION_PROFILE
    )


def test_method_parameters_switch_as_one_config_owned_surface(
    method_config_factory: MethodConfigFactory,
) -> None:
    config, _native_qwen = _load_method_config(
        method_config_factory,
        1,
        image_max_pixels=777_777,
        trajectories_per_prompt=5,
        max_response_length=2_345,
        maximum_tool_calls=7,
        rollout_seed=99,
        maximum_optimizer_steps=7,
    )
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    values = plan.overrides
    custom = values["actor_rollout_ref.rollout.custom"]

    assert config.policy.image_max_pixels == 777_777
    assert config.policy.sampling.trajectories_per_prompt == 5
    assert config.policy.sampling.max_response_length == 2_345
    assert config.protocol.maximum_tool_calls == 7
    assert config.rollout_rng.master_seed == 99
    assert config.training.maximum_optimizer_steps == 7
    assert values["data.mm_processor_kwargs.max_pixels"] == 777_777
    assert values["actor_rollout_ref.rollout.n"] == 5
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 5
    assert values["trainer.total_training_steps"] == 7
    assert custom["method_matrix"]["maximum_optimizer_steps"] == 7
    assert custom["method_matrix"]["image_max_pixels"] == 777_777
    assert custom["method_matrix"]["rollouts_per_prompt"] == 5


@pytest.mark.parametrize("case_index", (1, 4))
def test_method_marker_does_not_broaden_checkpoint_invariants(
    method_config_factory: MethodConfigFactory,
    case_index: int,
) -> None:
    config, native_qwen = _load_method_config(method_config_factory, case_index)
    plan = route_policy_method_matrix_plan(
        config,
        build_policy_e2e_smoke_verl_plan(config),
    )
    changed = dict(plan.overrides)
    manager_path = "actor_rollout_ref.rollout.checkpoint_manager_class"
    if native_qwen:
        changed[manager_path] = TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
        error = "behavior checkpoint manager differs"
    else:
        changed.pop(manager_path)
        error = "checkpoint manager differs"

    with pytest.raises(ValueError, match=error):
        replace(plan, overrides=changed)


def test_canonical_five_arm_matrix_loads_plans_composes_and_validates(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[2]
    run_config_root = repository_root / "configs" / "policy" / "runs"
    upstream_config_root = _write_minimal_upstream_config_directory(tmp_path)
    configs: list[PolicyE2ESmokeRunConfig] = []

    for profile, filename, maximum_tool_calls in _CANONICAL_METHOD_MATRIX_CONFIGS:
        config = load_policy_e2e_smoke_run_config(run_config_root / filename)
        assert config.method is not None
        assert config.method.matrix_id == (
            "prl28-config-driven-pixel512-s32-teacher25-v1"
        )
        assert config.method.profile is profile
        assert config.policy.image_max_pixels == 262_144
        assert config.policy.sampling.trajectories_per_prompt == 16
        assert config.policy.sampling.max_response_length == 20_480
        assert config.protocol.maximum_tool_calls == maximum_tool_calls
        assert config.rollout_rng.master_seed == 42
        assert config.reward.answer_reward_scale == 2.0
        assert config.reward.repeated_call_penalty == 0.05
        assert config.reward.protocol_error_penalty == 2.0
        assert config.accumulation.global_prompt_batch_size == 16
        assert config.distributed.world_size == 8
        assert config.training.maximum_optimizer_steps == 32
        assert config.training.checkpoint_steps == (0, 8, 16, 24, 32)

        plan = route_policy_method_matrix_plan(
            config,
            build_policy_e2e_smoke_verl_plan(config),
        )
        assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 262_144
        assert plan.overrides["actor_rollout_ref.rollout.n"] == 16
        assert plan.overrides["trainer.total_training_steps"] == 32
        composed = compose_upstream_verl_config(
            plan,
            config_directory=upstream_config_root,
        )
        assert composed.data.mm_processor_kwargs.max_pixels == 262_144
        assert composed.actor_rollout_ref.rollout.n == 16
        assert composed.trainer.total_training_steps == 32
        assert (
            composed.actor_rollout_ref.rollout.custom.method_matrix.profile
            == profile.value
        )
        configs.append(config)

    receipt = validate_policy_method_matrix(configs)
    assert receipt.matrix_id == "prl28-config-driven-pixel512-s32-teacher25-v1"
    assert len(receipt.shared_fingerprint_sha256) == 64
