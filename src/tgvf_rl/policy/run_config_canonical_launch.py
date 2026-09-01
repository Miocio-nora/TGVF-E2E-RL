"""Canonical launch bindings for a validated policy run configuration.

This module turns already-decoded model and dataset inputs into immutable
protocol, rollout, reward, and training bindings.  It does not parse TOML,
import judge implementations, or import the public run-config facade.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.data import (
    DeepEyes47KRuntimeBinding,
    PolicyT1MixedRuntimeBinding,
    PolicyT1RLRuntimeBinding,
    PolicyTeacherQuarterMixRuntimeBinding,
)
from tgvf_rl.protocol import (
    NativeActionBoundaryProtocolId,
    NativeAssistantDialect,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
    StandardToolError,
    ToolErrorCode,
    validate_success_observation_protocol,
    visual_tool_prompt_identity,
)

from .config import (
    POLICY_PILOT_V1_ACCEPTED_LEARNING_RATES,
    POLICY_PILOT_V1_TOOL_PROFILE,
    POLICY_PILOT_V1_VLLM_VERSION,
    DecoderLoRAConfig,
    PilotGRPOConfig,
    PilotSamplingConfig,
    PolicyMethodExperimentConfig,
    PolicyMethodProfile,
    PolicyMethodSamplingConfig,
    PolicyNoToolMatchedExperimentConfig,
    PolicyPilotV1Config,
    PolicyTGVFStage3ExperimentConfig,
    PolicyVisualToolExperimentConfig,
)
from .crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from .deepeyes_official_protocol import VISUAL_PROMPT_IDENTITY
from .no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from .tgvf_deepeyes_matched_protocol import TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
from .tgvf_target_guide_v2_protocol import TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY
from .run_config_reward import bind_policy_reward
from .run_config_schema import (
    POLICY_E2E_AGENT_LOOP_CONFIG_PATH,
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
    POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN,
    POLICY_E2E_SMOKE_CAP_ERROR_SHA256,
    POLICY_E2E_SMOKE_SEED_DERIVATION_NAME,
    POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256,
    POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256,
    PolicyMethodMatrixBinding,
    SmokeAccumulationBinding,
    SmokeCapacityBinding,
    SmokeDistributedBinding,
    SmokeFrameworkBinding,
    SmokeOptimizerBinding,
    SmokeOutputBinding,
    SmokePerformanceBinding,
    SmokePrecisionBinding,
    SmokeProtocolBinding,
    SmokeRewardBinding,
    SmokeRolloutRNGBinding,
    SmokeSchedulerBinding,
    SmokeTrainingBinding,
)
from .run_config_validation import (
    _absolute_path,
    _boolean,
    _checkpoint_steps,
    _distributed,
    _exact_real,
    _existing_file,
    _fqn,
    _integer,
    _logprob_measurement,
    _nonnegative_int,
    _nonnegative_int_tuple,
    _nonnegative_real,
    _optional_absolute_path,
    _positive_int,
    _positive_real,
    _real,
    _require_exact,
    _require_within,
    _safe_project_name,
    _sha256,
    _sha256_file,
    _table,
    _text,
    _text_tuple,
    _unit_interval,
)


class _DeepEyesControlPort(Protocol):
    def prompt_bundle_sha256(
        self,
        assistant_dialect: NativeAssistantDialect,
    ) -> str: ...


_HISTORICAL_THINKING_PROMPT_BUNDLES = {
    NativeToolCapabilityProfile.TGVF_ONLY: (
        "b44d8a6ff67f3752d9debe6365b0cb9ce4e37a13f117c7fdd87519d57751283f"
    ),
    NativeToolCapabilityProfile.CROP_ONLY: (
        "4bc9d8e814e0c6735d03a671ecff79cff6cffc74811e2693410fe3fe446cb31d"
    ),
    NativeToolCapabilityProfile.CROP_TGVF: (
        "c860c0a348646fc9a06500217709ec62b9bc01c422024641f35db232748da57f"
    ),
}


_TEACHER_QUARTER_PROMPT_BUNDLES = {
    NativeToolCapabilityProfile.NO_TOOL: frozenset(
        {NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256}
    ),
    NativeToolCapabilityProfile.CROP_ONLY: frozenset(
        {VISUAL_PROMPT_IDENTITY.bundle_sha256}
    ),
    NativeToolCapabilityProfile.TGVF_ONLY: frozenset(
        {
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        }
    ),
    NativeToolCapabilityProfile.CROP_TGVF: frozenset(
        {CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256}
    ),
}

_TOOL_PROFILE_BY_METHOD = {
    PolicyMethodProfile.NO_TOOL: NativeToolCapabilityProfile.NO_TOOL,
    PolicyMethodProfile.CROP: NativeToolCapabilityProfile.CROP_ONLY,
    PolicyMethodProfile.TGVF_SHORT: NativeToolCapabilityProfile.TGVF_ONLY,
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2: (NativeToolCapabilityProfile.TGVF_ONLY),
    PolicyMethodProfile.ATOMIC: NativeToolCapabilityProfile.CROP_TGVF,
}


def _method_prompt_sha256(method: PolicyMethodProfile) -> str:
    if method is PolicyMethodProfile.NO_TOOL:
        return NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256
    if method is PolicyMethodProfile.CROP:
        return VISUAL_PROMPT_IDENTITY.bundle_sha256
    if method is PolicyMethodProfile.TGVF_SHORT:
        return TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    if method is PolicyMethodProfile.TGVF_TARGET_GUIDE_V2:
        return TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256
    if method is PolicyMethodProfile.ATOMIC:
        return CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    raise AssertionError(f"unsupported method profile: {method!r}")


_SUCCESS_OBSERVATION_BY_METHOD = {
    PolicyMethodProfile.NO_TOOL: (
        NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1
    ),
    PolicyMethodProfile.CROP: (
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    ),
    PolicyMethodProfile.TGVF_SHORT: (
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1
    ),
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2: (
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1
    ),
    PolicyMethodProfile.ATOMIC: (
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1
    ),
}


def _call_cap_error_sha256(maximum_tool_calls: int) -> str:
    return StandardToolError(
        code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
        message=(
            f"The maximum of {maximum_tool_calls} tool-call attempts has been "
            "reached; this call was not executed."
        ),
        attempt_index=maximum_tool_calls,
        recoverable=True,
        maximum_tool_calls=maximum_tool_calls,
    ).payload_sha256


@dataclass(frozen=True, slots=True)
class _CanonicalLaunchBindings:
    protocol: SmokeProtocolBinding
    sampling: PilotSamplingConfig
    rollout_rng: SmokeRolloutRNGBinding
    reward: SmokeRewardBinding
    optimizer: SmokeOptimizerBinding
    scheduler: SmokeSchedulerBinding
    precision: SmokePrecisionBinding
    accumulation: SmokeAccumulationBinding
    distributed: SmokeDistributedBinding
    capacity: SmokeCapacityBinding
    performance: SmokePerformanceBinding
    framework: SmokeFrameworkBinding
    training: SmokeTrainingBinding
    output: SmokeOutputBinding
    policy: (
        PolicyPilotV1Config
        | PolicyMethodExperimentConfig
        | PolicyNoToolMatchedExperimentConfig
        | PolicyTGVFStage3ExperimentConfig
        | PolicyVisualToolExperimentConfig
    )


def bind_canonical_policy_launch(
    payload: Mapping[str, object],
    *,
    allow_external_agent_loop_config: bool,
    allow_historical_reward_contract: bool,
    assistant_dialect: NativeAssistantDialect,
    deepeyes_control: _DeepEyesControlPort | None,
    deepeyes_scaled_crop_run: bool,
    explicit_observation_run: bool,
    formal_pilot: bool,
    iteration_sha256: str,
    mixed_run: bool,
    model: ModelIdentity,
    model_table: Mapping[str, object],
    method_binding: PolicyMethodMatrixBinding | None,
    runtime_binding: (
        DeepEyes47KRuntimeBinding
        | PolicyT1RLRuntimeBinding
        | PolicyT1MixedRuntimeBinding
        | PolicyTeacherQuarterMixRuntimeBinding
    ),
    stage3_shaped_reward_version: str,
    stage3_shaped_run: bool,
    visual_always_judge: bool,
    pilot_reward_weight_profile_name: Callable[..., Any],
    load_openai_compatible_judge: Callable[..., Any],
    load_tgvf_tool_utility_runtime_binding: Callable[..., Any],
    load_tgvf_visual_quality_judge: Callable[..., Any],
) -> _CanonicalLaunchBindings:
    """Validate and bind every canonical launch-owned configuration section."""

    method_profile = None if method_binding is None else method_binding.profile
    method_run = method_profile is not None
    performance_v2 = (
        payload.get("schema_version") == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA
    )
    performance_table: Mapping[str, object] | None = None
    if performance_v2:
        performance_table = _table(
            payload,
            "performance",
            {
                "dynamic_token_batching",
                "use_remove_padding",
                "enable_gradient_checkpointing",
                "vllm_enable_prefix_caching",
                "vllm_enable_chunked_prefill",
                "vllm_enable_cuda_graph",
                "vllm_cuda_graph_capture_sizes",
                "vllm_tensor_parallel_size",
                "rollout_logprob_bypass",
                "reference_replay_mode",
                "judge_dispatch_mode",
                "judge_max_concurrency_per_worker",
            },
        )

    protocol_fields = {
        "prompt_sha256",
        "cap_error_sha256",
        "tool_profile",
        "tool_schema_sha256",
        "enabled_tool_names",
        "maximum_tool_calls",
    }
    if explicit_observation_run:
        protocol_fields.add("success_observation_protocol_id")
        protocol_fields.add("action_boundary_protocol_id")
    protocol_table = _table(
        payload,
        "protocol",
        protocol_fields,
    )
    enabled_tools = _text_tuple(
        protocol_table["enabled_tool_names"], name="protocol.enabled_tool_names"
    )
    try:
        tool_profile = NativeToolCapabilityProfile(protocol_table["tool_profile"])
    except (TypeError, ValueError) as error:
        raise ValueError("protocol.tool_profile is invalid") from error
    _require_exact(
        enabled_tools,
        tool_profile.tool_names,
        "protocol.enabled_tool_names",
    )
    _require_exact(
        protocol_table["tool_schema_sha256"],
        tool_profile.tool_set_sha256,
        "protocol.tool_schema_sha256",
    )
    one_call_protocol = not method_run and (
        stage3_shaped_run or tool_profile is NativeToolCapabilityProfile.NO_TOOL
    )
    maximum_tool_calls = _positive_int(
        protocol_table["maximum_tool_calls"],
        name="protocol.maximum_tool_calls",
    )
    if not method_run:
        _require_exact(
            maximum_tool_calls,
            1 if one_call_protocol else 4,
            "protocol.maximum_tool_calls",
        )
    cap_error_sha256 = _sha256(
        protocol_table["cap_error_sha256"], name="protocol.cap_error_sha256"
    )
    _require_exact(
        cap_error_sha256,
        _call_cap_error_sha256(maximum_tool_calls)
        if method_run
        else POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256
        if one_call_protocol
        else POLICY_E2E_SMOKE_CAP_ERROR_SHA256,
        "protocol.cap_error_sha256",
    )
    success_observation_protocol_id: NativeSuccessObservationProtocolId | None = None
    action_boundary_protocol_id: NativeActionBoundaryProtocolId | None = None
    if explicit_observation_run:
        success_observation_protocol_id = validate_success_observation_protocol(
            protocol_table["success_observation_protocol_id"],
            tool_profile=tool_profile,
            assistant_dialect=assistant_dialect,
        )
        try:
            action_boundary_protocol_id = NativeActionBoundaryProtocolId(
                protocol_table["action_boundary_protocol_id"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "protocol.action_boundary_protocol_id is invalid"
            ) from error
        if method_profile is not None:
            _require_exact(
                success_observation_protocol_id,
                _SUCCESS_OBSERVATION_BY_METHOD[method_profile],
                "method success-observation protocol",
            )
            _require_exact(
                action_boundary_protocol_id,
                NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2,
                "method action-boundary protocol",
            )
    protocol = SmokeProtocolBinding(
        prompt_sha256=_sha256(
            protocol_table["prompt_sha256"], name="protocol.prompt_sha256"
        ),
        cap_error_sha256=cap_error_sha256,
        tool_profile=tool_profile,
        tool_schema_sha256=protocol_table["tool_schema_sha256"],
        enabled_tool_names=enabled_tools,
        maximum_tool_calls=maximum_tool_calls,
        success_observation_protocol_id=success_observation_protocol_id,
        action_boundary_protocol_id=action_boundary_protocol_id,
    )
    if method_profile is not None:
        _require_exact(
            protocol.tool_profile,
            _TOOL_PROFILE_BY_METHOD[method_profile],
            "method tool profile",
        )
    if deepeyes_control is not None:
        _require_exact(
            protocol.prompt_sha256,
            deepeyes_control.prompt_bundle_sha256(assistant_dialect),
            "protocol.prompt_sha256",
        )
    elif method_profile is not None:
        _require_exact(
            protocol.prompt_sha256,
            _method_prompt_sha256(method_profile),
            "method prompt profile",
        )
    elif mixed_run:
        if isinstance(runtime_binding, PolicyTeacherQuarterMixRuntimeBinding):
            if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_INSTRUCT:
                raise ValueError("Teacher25 requires Qwen3-VL Instruct")
            accepted_prompt_hashes = set(_TEACHER_QUARTER_PROMPT_BUNDLES[tool_profile])
        else:
            accepted_prompt_hashes = {
                visual_tool_prompt_identity(
                    tool_profile,
                    assistant_dialect=assistant_dialect,
                ).bundle_sha256
            }
            if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
                accepted_prompt_hashes.add(
                    _HISTORICAL_THINKING_PROMPT_BUNDLES[tool_profile]
                )
        if protocol.prompt_sha256 not in accepted_prompt_hashes:
            raise ValueError("protocol.prompt_sha256 differs from model dialect")

    sampling_table = _table(
        payload,
        "sampling",
        {
            "trajectories_per_prompt",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_response_length",
            "asynchronous_staleness_steps",
            "do_sample",
            "backend",
            "backend_version",
            "logit_processors",
            "logprob_measurement",
            "stop_token_ids",
            "stop_strings",
            "include_stop_str_in_output",
            "ignore_eos",
            "rollout_master_seed",
            "seed_derivation_name",
            "seed_derivation_sha256",
        },
    )
    sampling_type = PolicyMethodSamplingConfig if method_run else PilotSamplingConfig
    sampling = sampling_type(
        trajectories_per_prompt=_integer(
            sampling_table["trajectories_per_prompt"],
            name="sampling.trajectories_per_prompt",
        ),
        temperature=_real(sampling_table["temperature"], name="sampling.temperature"),
        top_p=_real(sampling_table["top_p"], name="sampling.top_p"),
        top_k=_integer(sampling_table["top_k"], name="sampling.top_k"),
        min_p=_real(sampling_table["min_p"], name="sampling.min_p"),
        repetition_penalty=_real(
            sampling_table["repetition_penalty"], name="sampling.repetition_penalty"
        ),
        presence_penalty=_real(
            sampling_table["presence_penalty"], name="sampling.presence_penalty"
        ),
        frequency_penalty=_real(
            sampling_table["frequency_penalty"], name="sampling.frequency_penalty"
        ),
        max_response_length=_positive_int(
            sampling_table["max_response_length"], name="sampling.max_response_length"
        ),
        asynchronous_staleness_steps=_nonnegative_int(
            sampling_table["asynchronous_staleness_steps"],
            name="sampling.asynchronous_staleness_steps",
        ),
        do_sample=_boolean(sampling_table["do_sample"], name="sampling.do_sample"),
        backend=_text(sampling_table["backend"], name="sampling.backend"),
        backend_version=_text(
            sampling_table["backend_version"], name="sampling.backend_version"
        ),
        logit_processors=_text_tuple(
            sampling_table["logit_processors"], name="sampling.logit_processors"
        ),
        logprob_measurement=_logprob_measurement(sampling_table["logprob_measurement"]),
        stop_token_ids=_nonnegative_int_tuple(
            sampling_table["stop_token_ids"], name="sampling.stop_token_ids"
        ),
        stop_strings=_text_tuple(
            sampling_table["stop_strings"], name="sampling.stop_strings"
        ),
        include_stop_str_in_output=_boolean(
            sampling_table["include_stop_str_in_output"],
            name="sampling.include_stop_str_in_output",
        ),
        ignore_eos=_boolean(sampling_table["ignore_eos"], name="sampling.ignore_eos"),
    )
    if not method_run:
        expected_sampling_scale = (16, 20480) if deepeyes_scaled_crop_run else (8, 8192)
        _require_exact(
            (sampling.trajectories_per_prompt, sampling.max_response_length),
            expected_sampling_scale,
            "sampling DeepEyes-reference scale",
        )
    if (
        "</tool_call>" in (sampling.stop_strings or ())
        and sampling.include_stop_str_in_output is not True
    ):
        raise ValueError(
            "sampling.include_stop_str_in_output must be true when </tool_call> "
            "is a stop string so the complete closing tag remains policy-sampled"
        )
    derivation_name = _text(
        sampling_table["seed_derivation_name"],
        name="sampling.seed_derivation_name",
    )
    derivation_sha256 = _sha256(
        sampling_table["seed_derivation_sha256"],
        name="sampling.seed_derivation_sha256",
    )
    _require_exact(
        derivation_name,
        POLICY_E2E_SMOKE_SEED_DERIVATION_NAME,
        "sampling.seed_derivation_name",
    )
    _require_exact(
        derivation_sha256,
        POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256,
        "sampling.seed_derivation_sha256",
    )
    rollout_rng = SmokeRolloutRNGBinding(
        master_seed=_nonnegative_int(
            sampling_table["rollout_master_seed"],
            name="sampling.rollout_master_seed",
        ),
        derivation_name=derivation_name,
        derivation_sha256=derivation_sha256,
    )
    if method_profile is not None:
        expected_stop_strings = (
            () if method_profile is PolicyMethodProfile.NO_TOOL else ("</tool_call>",)
        )
        _require_exact(
            (
                sampling.stop_token_ids,
                sampling.stop_strings,
                sampling.include_stop_str_in_output,
                sampling.ignore_eos,
            ),
            ((151645,), expected_stop_strings, True, False),
            "method complete stop contract",
        )
    reward = bind_policy_reward(
        payload,
        allow_historical_reward_contract=allow_historical_reward_contract,
        deepeyes_control_present=deepeyes_control is not None,
        formal_pilot=formal_pilot,
        iteration_sha256=iteration_sha256,
        method_binding=method_binding,
        mixed_run=mixed_run,
        runtime_binding=runtime_binding,
        stage3_shaped_reward_version=stage3_shaped_reward_version,
        stage3_shaped_run=stage3_shaped_run,
        tool_profile=tool_profile,
        visual_always_judge=visual_always_judge,
        pilot_reward_weight_profile_name=pilot_reward_weight_profile_name,
        load_openai_compatible_judge=load_openai_compatible_judge,
        load_tgvf_tool_utility_runtime_binding=(load_tgvf_tool_utility_runtime_binding),
        load_tgvf_visual_quality_judge=load_tgvf_visual_quality_judge,
    )

    optimizer_table = _table(
        payload,
        "optimizer",
        {
            "name",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "maximum_gradient_norm",
        },
    )
    _require_exact(optimizer_table["name"], "adamw", "optimizer.name")
    learning_rate = _positive_real(
        optimizer_table["learning_rate"], name="optimizer.learning_rate"
    )
    if learning_rate not in POLICY_PILOT_V1_ACCEPTED_LEARNING_RATES:
        raise ValueError(
            "optimizer.learning_rate must be one of "
            f"{POLICY_PILOT_V1_ACCEPTED_LEARNING_RATES!r}"
        )
    optimizer = SmokeOptimizerBinding(
        name=optimizer_table["name"],
        learning_rate=learning_rate,
        beta1=_unit_interval(optimizer_table["beta1"], name="optimizer.beta1"),
        beta2=_unit_interval(optimizer_table["beta2"], name="optimizer.beta2"),
        epsilon=_positive_real(optimizer_table["epsilon"], name="optimizer.epsilon"),
        weight_decay=_nonnegative_real(
            optimizer_table["weight_decay"], name="optimizer.weight_decay"
        ),
        maximum_gradient_norm=_exact_real(
            optimizer_table["maximum_gradient_norm"],
            1.0,
            "optimizer.maximum_gradient_norm",
        ),
    )

    scheduler_table = _table(
        payload,
        "scheduler",
        {"name", "warmup_steps", "total_steps", "minimum_learning_rate_ratio"},
    )
    scheduler_name = _text(scheduler_table["name"], name="scheduler.name")
    if scheduler_name not in {"constant", "cosine"}:
        raise ValueError("scheduler.name must be constant or cosine")
    scheduler = SmokeSchedulerBinding(
        name=scheduler_name,
        warmup_steps=_nonnegative_int(
            scheduler_table["warmup_steps"], name="scheduler.warmup_steps"
        ),
        total_steps=_positive_int(
            scheduler_table["total_steps"], name="scheduler.total_steps"
        ),
        minimum_learning_rate_ratio=_unit_interval(
            scheduler_table["minimum_learning_rate_ratio"],
            name="scheduler.minimum_learning_rate_ratio",
            inclusive=True,
        ),
    )

    precision_table = _table(
        payload,
        "precision",
        {
            "parameter_dtype",
            "reduce_dtype",
            "optimizer_state_dtype",
            "autocast_dtype",
            "gradient_scaler_enabled",
            "allow_tf32",
        },
    )
    _require_exact(
        precision_table["parameter_dtype"], "bfloat16", "precision.parameter_dtype"
    )
    _require_exact(
        precision_table["optimizer_state_dtype"],
        "float32",
        "precision.optimizer_state_dtype",
    )
    _require_exact(
        precision_table["autocast_dtype"], "bfloat16", "precision.autocast_dtype"
    )
    _require_exact(
        precision_table["gradient_scaler_enabled"],
        False,
        "precision.gradient_scaler_enabled",
    )
    reduce_dtype = _text(precision_table["reduce_dtype"], name="precision.reduce_dtype")
    if reduce_dtype not in {"bfloat16", "float32"}:
        raise ValueError("precision.reduce_dtype must be bfloat16 or float32")
    precision = SmokePrecisionBinding(
        parameter_dtype=precision_table["parameter_dtype"],
        reduce_dtype=reduce_dtype,
        optimizer_state_dtype=precision_table["optimizer_state_dtype"],
        autocast_dtype=precision_table["autocast_dtype"],
        gradient_scaler_enabled=precision_table["gradient_scaler_enabled"],
        allow_tf32=_boolean(precision_table["allow_tf32"], name="precision.allow_tf32"),
    )

    accumulation_table = _table(
        payload,
        "accumulation",
        {
            "global_prompt_batch_size",
            "prompt_micro_batch_size_per_rank",
            "rollout_prompt_micro_batch_size_per_engine",
            "gradient_accumulation_steps",
        },
    )
    accumulation = SmokeAccumulationBinding(
        global_prompt_batch_size=_positive_int(
            accumulation_table["global_prompt_batch_size"],
            name="accumulation.global_prompt_batch_size",
        ),
        prompt_micro_batch_size_per_rank=_positive_int(
            accumulation_table["prompt_micro_batch_size_per_rank"],
            name="accumulation.prompt_micro_batch_size_per_rank",
        ),
        rollout_prompt_micro_batch_size_per_engine=_positive_int(
            accumulation_table["rollout_prompt_micro_batch_size_per_engine"],
            name="accumulation.rollout_prompt_micro_batch_size_per_engine",
        ),
        gradient_accumulation_steps=_positive_int(
            accumulation_table["gradient_accumulation_steps"],
            name="accumulation.gradient_accumulation_steps",
        ),
    )

    distributed_fields = {
        "physical_gpu_ids",
        "logical_gpu_ids",
        "world_size",
        "actor_logical_gpu_ids",
        "rollout_logical_gpu_ids",
        "fsdp_strategy",
        "fsdp_reshard_after_forward",
        "rollout_backend",
        "placement",
        "weight_sync_mode",
        "weight_sync_interval_optimizer_steps",
    }
    if not performance_v2:
        distributed_fields.add("vllm_tensor_parallel_size")
    distributed_table = _table(
        payload,
        "distributed",
        distributed_fields,
    )
    normalized_distributed_table = dict(distributed_table)
    if performance_table is not None:
        normalized_distributed_table["vllm_tensor_parallel_size"] = _positive_int(
            performance_table["vllm_tensor_parallel_size"],
            name="performance.vllm_tensor_parallel_size",
        )
    distributed = _distributed(normalized_distributed_table)
    expected_global_batch = (
        accumulation.prompt_micro_batch_size_per_rank
        * distributed.world_size
        * accumulation.gradient_accumulation_steps
    )
    if accumulation.global_prompt_batch_size != expected_global_batch:
        raise ValueError(
            "accumulation global prompt batch is inconsistent with world size"
        )

    capacity_fields = {
        "max_prompt_length",
        "actor_ppo_max_token_len_per_gpu",
        "rollout_log_prob_max_token_len_per_gpu",
        "reference_log_prob_max_token_len_per_gpu",
        "vllm_gpu_memory_utilization",
        "vllm_max_num_batched_tokens",
        "vllm_max_model_len",
        "vllm_max_num_seqs",
    }
    if not performance_v2:
        capacity_fields.update({"vllm_enable_chunked_prefill", "vllm_enforce_eager"})
    capacity_table = _table(
        payload,
        "capacity",
        capacity_fields,
    )
    normalized_capacity_table = dict(capacity_table)
    if performance_table is not None:
        normalized_capacity_table["vllm_enable_chunked_prefill"] = _boolean(
            performance_table["vllm_enable_chunked_prefill"],
            name="performance.vllm_enable_chunked_prefill",
        )
        normalized_capacity_table["vllm_enforce_eager"] = not _boolean(
            performance_table["vllm_enable_cuda_graph"],
            name="performance.vllm_enable_cuda_graph",
        )
    capacity = SmokeCapacityBinding(
        max_prompt_length=_positive_int(
            normalized_capacity_table["max_prompt_length"],
            name="capacity.max_prompt_length",
        ),
        actor_ppo_max_token_len_per_gpu=_positive_int(
            normalized_capacity_table["actor_ppo_max_token_len_per_gpu"],
            name="capacity.actor_ppo_max_token_len_per_gpu",
        ),
        rollout_log_prob_max_token_len_per_gpu=_positive_int(
            normalized_capacity_table["rollout_log_prob_max_token_len_per_gpu"],
            name="capacity.rollout_log_prob_max_token_len_per_gpu",
        ),
        reference_log_prob_max_token_len_per_gpu=_positive_int(
            normalized_capacity_table["reference_log_prob_max_token_len_per_gpu"],
            name="capacity.reference_log_prob_max_token_len_per_gpu",
        ),
        vllm_gpu_memory_utilization=_unit_interval(
            normalized_capacity_table["vllm_gpu_memory_utilization"],
            name="capacity.vllm_gpu_memory_utilization",
        ),
        vllm_max_num_batched_tokens=_positive_int(
            normalized_capacity_table["vllm_max_num_batched_tokens"],
            name="capacity.vllm_max_num_batched_tokens",
        ),
        vllm_max_model_len=_positive_int(
            normalized_capacity_table["vllm_max_model_len"],
            name="capacity.vllm_max_model_len",
        ),
        vllm_max_num_seqs=_positive_int(
            normalized_capacity_table["vllm_max_num_seqs"],
            name="capacity.vllm_max_num_seqs",
        ),
        vllm_enable_chunked_prefill=_boolean(
            normalized_capacity_table["vllm_enable_chunked_prefill"],
            name="capacity.vllm_enable_chunked_prefill",
        ),
        vllm_enforce_eager=_boolean(
            normalized_capacity_table["vllm_enforce_eager"],
            name="capacity.vllm_enforce_eager",
        ),
    )
    if performance_table is None:
        performance = SmokePerformanceBinding(
            dynamic_token_batching=False,
            use_remove_padding=False,
            enable_gradient_checkpointing=False,
            vllm_enable_prefix_caching=False,
            vllm_enable_chunked_prefill=capacity.vllm_enable_chunked_prefill,
            vllm_enable_cuda_graph=not capacity.vllm_enforce_eager,
            vllm_cuda_graph_capture_sizes=(),
            vllm_tensor_parallel_size=distributed.vllm_tensor_parallel_size,
            rollout_logprob_bypass=True,
            reference_replay_mode="full_diagnostic",
            judge_dispatch_mode="inherit",
            judge_max_concurrency_per_worker=1,
        )
    else:
        capture_sizes = _cuda_graph_capture_sizes(
            performance_table["vllm_cuda_graph_capture_sizes"]
        )
        cuda_graph_enabled = not capacity.vllm_enforce_eager
        if cuda_graph_enabled and not capture_sizes:
            raise ValueError(
                "enabled CUDA graph requires explicit performance."
                "vllm_cuda_graph_capture_sizes"
            )
        if capture_sizes and capture_sizes[-1] > capacity.vllm_max_num_batched_tokens:
            raise ValueError(
                "performance.vllm_cuda_graph_capture_sizes cannot exceed "
                "capacity.vllm_max_num_batched_tokens"
            )
        performance = SmokePerformanceBinding(
            dynamic_token_batching=_boolean(
                performance_table["dynamic_token_batching"],
                name="performance.dynamic_token_batching",
            ),
            use_remove_padding=_boolean(
                performance_table["use_remove_padding"],
                name="performance.use_remove_padding",
            ),
            enable_gradient_checkpointing=_boolean(
                performance_table["enable_gradient_checkpointing"],
                name="performance.enable_gradient_checkpointing",
            ),
            vllm_enable_prefix_caching=_boolean(
                performance_table["vllm_enable_prefix_caching"],
                name="performance.vllm_enable_prefix_caching",
            ),
            vllm_enable_chunked_prefill=capacity.vllm_enable_chunked_prefill,
            vllm_enable_cuda_graph=cuda_graph_enabled,
            vllm_cuda_graph_capture_sizes=capture_sizes,
            vllm_tensor_parallel_size=distributed.vllm_tensor_parallel_size,
            rollout_logprob_bypass=_boolean(
                performance_table["rollout_logprob_bypass"],
                name="performance.rollout_logprob_bypass",
            ),
            reference_replay_mode=_text(
                performance_table["reference_replay_mode"],
                name="performance.reference_replay_mode",
            ),
            judge_dispatch_mode=_text(
                performance_table["judge_dispatch_mode"],
                name="performance.judge_dispatch_mode",
            ),
            judge_max_concurrency_per_worker=_positive_int(
                performance_table["judge_max_concurrency_per_worker"],
                name="performance.judge_max_concurrency_per_worker",
            ),
        )
        if performance.judge_dispatch_mode == "inherit":
            raise ValueError(
                "method-matrix v2 judge dispatch must be inline or "
                "dedicated_thread_pool"
            )
    minimum_context = capacity.max_prompt_length + sampling.max_response_length
    if capacity.vllm_max_model_len < minimum_context:
        raise ValueError(
            "capacity.vllm_max_model_len cannot hold max prompt plus response"
        )
    if (
        capacity.rollout_log_prob_max_token_len_per_gpu
        < capacity.actor_ppo_max_token_len_per_gpu
        or capacity.reference_log_prob_max_token_len_per_gpu
        < capacity.actor_ppo_max_token_len_per_gpu
    ):
        raise ValueError(
            "rollout/reference log-prob token bounds cannot be smaller than the "
            "actor bound"
        )
    if capacity.vllm_max_num_batched_tokens > capacity.vllm_max_model_len:
        raise ValueError(
            "capacity.vllm_max_num_batched_tokens cannot exceed max_model_len"
        )
    if (
        capacity.vllm_max_num_seqs
        < accumulation.rollout_prompt_micro_batch_size_per_engine
        * sampling.trajectories_per_prompt
    ):
        raise ValueError(
            "capacity.vllm_max_num_seqs cannot hold one rollout engine micro-batch"
        )

    framework_table = _table(
        payload,
        "framework",
        {
            "agent_loop_config_path",
            "agent_loop_config_sha256",
            "runtime_invocation_factory_fqn",
            "server_timeout_seconds",
        },
    )
    agent_loop_config_path = _existing_file(
        framework_table["agent_loop_config_path"],
        name="framework.agent_loop_config_path",
    )
    if (
        not allow_external_agent_loop_config
        and agent_loop_config_path != POLICY_E2E_AGENT_LOOP_CONFIG_PATH
    ):
        raise ValueError(
            "framework.agent_loop_config_path differs from the checked-in "
            "Policy Pilot composition"
        )
    agent_loop_config_sha256 = _sha256(
        framework_table["agent_loop_config_sha256"],
        name="framework.agent_loop_config_sha256",
    )
    if _sha256_file(agent_loop_config_path) != agent_loop_config_sha256:
        raise ValueError("framework AgentLoop config SHA256 mismatch")
    runtime_factory_fqn = _fqn(
        framework_table["runtime_invocation_factory_fqn"],
        name="framework.runtime_invocation_factory_fqn",
    )
    _require_exact(
        runtime_factory_fqn,
        POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN,
        "framework.runtime_invocation_factory_fqn",
    )
    framework = SmokeFrameworkBinding(
        agent_loop_config_path=agent_loop_config_path,
        agent_loop_config_sha256=agent_loop_config_sha256,
        runtime_invocation_factory_fqn=runtime_factory_fqn,
        server_timeout_seconds=_positive_real(
            framework_table["server_timeout_seconds"],
            name="framework.server_timeout_seconds",
        ),
    )

    training_table = _table(
        payload,
        "training",
        {
            "total_training_epochs",
            "maximum_optimizer_steps",
            "checkpoint_steps",
            "logger",
            "project_name",
            "validation_before_training",
            "validation_frequency",
            "resume_mode",
            "resume_from_path",
            "maximum_actor_checkpoints_to_keep",
        },
    )
    loggers = _text_tuple(training_table["logger"], name="training.logger")
    if not loggers or len(set(loggers)) != len(loggers):
        raise ValueError("training.logger must be non-empty and unique")
    unsupported_loggers = set(loggers).difference({"console", "wandb"})
    if unsupported_loggers:
        raise ValueError(
            f"training.logger contains unsupported backends: {sorted(unsupported_loggers)!r}"
        )
    validation_frequency = _integer(
        training_table["validation_frequency"],
        name="training.validation_frequency",
    )
    if validation_frequency != -1:
        raise ValueError(
            "bounded Policy E2E smoke requires training.validation_frequency=-1"
        )
    resume_mode = _text(training_table["resume_mode"], name="training.resume_mode")
    if resume_mode not in {"auto", "disable", "resume_path"}:
        raise ValueError("training.resume_mode must be auto, disable, or resume_path")
    resume_from_path = _optional_absolute_path(
        training_table["resume_from_path"],
        name="training.resume_from_path",
    )
    if resume_mode in {"auto", "disable"} and resume_from_path is not None:
        raise ValueError("training.resume_from_path must be empty in auto/disable mode")
    if resume_mode == "resume_path":
        if resume_from_path is None or not resume_from_path.is_dir():
            raise ValueError(
                "training.resume_from_path must be an existing directory in resume_path mode"
            )
    training = SmokeTrainingBinding(
        total_training_epochs=_positive_int(
            training_table["total_training_epochs"],
            name="training.total_training_epochs",
        ),
        maximum_optimizer_steps=_positive_int(
            training_table["maximum_optimizer_steps"],
            name="training.maximum_optimizer_steps",
        ),
        checkpoint_steps=_checkpoint_steps(training_table["checkpoint_steps"]),
        logger=loggers,
        project_name=_safe_project_name(training_table["project_name"]),
        validation_before_training=_boolean(
            training_table["validation_before_training"],
            name="training.validation_before_training",
        ),
        validation_frequency=validation_frequency,
        resume_mode=resume_mode,
        resume_from_path=resume_from_path,
        maximum_actor_checkpoints_to_keep=_positive_int(
            training_table["maximum_actor_checkpoints_to_keep"],
            name="training.maximum_actor_checkpoints_to_keep",
        ),
    )
    if training.validation_before_training:
        raise ValueError(
            "bounded Policy E2E smoke does not own a validation population"
        )
    if training.checkpoint_steps[-1] > training.maximum_optimizer_steps:
        raise ValueError("training checkpoint step exceeds maximum_optimizer_steps")
    if scheduler.total_steps < training.maximum_optimizer_steps:
        raise ValueError(
            "scheduler total_steps is smaller than maximum_optimizer_steps"
        )
    if scheduler.warmup_steps >= scheduler.total_steps:
        raise ValueError("scheduler warmup_steps must be smaller than total_steps")
    output_table = _table(
        payload, "output", {"root", "checkpoint_directory", "metrics_path"}
    )
    output_root = _absolute_path(output_table["root"], name="output.root")
    if output_root == Path("/"):
        raise ValueError("output.root cannot be the filesystem root")
    checkpoint_directory = _absolute_path(
        output_table["checkpoint_directory"], name="output.checkpoint_directory"
    )
    metrics_path = _absolute_path(
        output_table["metrics_path"], name="output.metrics_path"
    )
    _require_within(
        checkpoint_directory, output_root, name="output.checkpoint_directory"
    )
    _require_within(metrics_path, output_root, name="output.metrics_path")
    if resume_from_path is not None:
        _require_within(resume_from_path, output_root, name="training.resume_from_path")
    output = SmokeOutputBinding(output_root, checkpoint_directory, metrics_path)

    if method_profile is not None:
        policy_type = PolicyMethodExperimentConfig
    elif stage3_shaped_run:
        policy_type = PolicyTGVFStage3ExperimentConfig
    elif protocol.tool_profile is NativeToolCapabilityProfile.NO_TOOL:
        policy_type = PolicyNoToolMatchedExperimentConfig
    elif protocol.tool_profile is POLICY_PILOT_V1_TOOL_PROFILE:
        policy_type = PolicyPilotV1Config
    else:
        policy_type = PolicyVisualToolExperimentConfig
    policy_kwargs: dict[str, object] = {}
    if method_profile is not None:
        policy_kwargs["method"] = method_profile
    policy = policy_type(
        model_family=model.family,
        model_path=model.revision_or_path,
        native_deepstack_enabled=model_table["native_deepstack_enabled"],
        tool_profile=protocol.tool_profile,
        enabled_tool_names=protocol.enabled_tool_names,
        max_tgvf_call_attempts=protocol.maximum_tool_calls,
        image_max_pixels=model_table["image_max_pixels"],
        sampling=sampling,
        lora=DecoderLoRAConfig(initial_learning_rate=optimizer.learning_rate),
        grpo=PilotGRPOConfig(total_training_epochs=training.total_training_epochs),
        **policy_kwargs,
    )
    if sampling.backend_version != POLICY_PILOT_V1_VLLM_VERSION:
        raise ValueError("sampling backend version differs from Policy Pilot v1")
    return _CanonicalLaunchBindings(
        protocol=protocol,
        sampling=sampling,
        rollout_rng=rollout_rng,
        reward=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
        accumulation=accumulation,
        distributed=distributed,
        capacity=capacity,
        performance=performance,
        framework=framework,
        training=training,
        output=output,
        policy=policy,
    )


def _cuda_graph_capture_sizes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise ValueError(
            "performance.vllm_cuda_graph_capture_sizes must be an array of "
            "positive integers"
        )
    sizes = tuple(value)
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError(
            "performance.vllm_cuda_graph_capture_sizes must be strictly "
            "increasing and unique"
        )
    return sizes


__all__ = ["bind_canonical_policy_launch"]
