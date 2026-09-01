"""Immutable schema objects shared by policy run-config consumers.

This module owns the stable configuration identities and value objects.  It
deliberately does not parse TOML, load judges, inspect artifacts, or perform
filesystem validation; those responsibilities remain in :mod:`.run_config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.public_api_compat import rebind_public_class
from tgvf_rl.protocol import (
    StandardToolError,
    ToolErrorCode,
)

from .config import PolicyMethodProfile

if TYPE_CHECKING:
    from tgvf_rl.conditioning import TargetConditioningConfig
    from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
    from tgvf_rl.data import (
        DeepEyes47KRuntimeBinding,
        PolicyTeacherQuarterMixRuntimeBinding,
        PolicyT1MixedRuntimeBinding,
        PolicyT1RLRuntimeBinding,
    )
    from tgvf_rl.data.tgvf_tool_utility import TGVFToolUtilityRuntimeBinding
    from tgvf_rl.protocol import (
        NativeActionBoundaryProtocolId,
        NativeSuccessObservationProtocolId,
        NativeToolCapabilityProfile,
    )

    from .config import PolicyPilotV1Config
    from .deepeyes_strict_control import DeepEyesStrictControlBinding


POLICY_E2E_SMOKE_CONFIG_SCHEMA = "policy-e2e-smoke-config-v3"
POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA = "policy-e2e-mixed-run-config-v4"
POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA = "policy-e2e-formal-pilot-config-v1"
POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA = "policy-e2e-stage3-shaped-run-config-v1"
POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA = (
    "policy-e2e-deepeyes-scaled-crop-run-config-v1"
)
POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA = (
    "policy-e2e-deepeyes-strict-control-run-config-v1"
)
POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA = (
    "policy-e2e-explicit-observation-run-config-v1"
)
POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA = "policy-e2e-method-matrix-run-config-v1"
POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA = "policy-e2e-method-matrix-run-config-v2"
POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    "policy-e2e-crop-tgvf-tfree-deepeyes-matched-pixel512-parity-run-config-v1"
)
POLICY_E2E_ATOMIC_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
)
POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    "policy-e2e-crop-tfree-exact-deepeyes-matched-pixel512-parity-run-config-v1"
)
POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    "policy-e2e-no-tool-tfree-deepeyes-matched-pixel512-parity-run-config-v1"
)
POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    "policy-e2e-tgvf-short-tfree-deepeyes-matched-pixel512-parity-run-config-v1"
)
POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA = (
    "policy-e2e-tgvf-target-guide-v2-tfree-deepeyes-matched-"
    "pixel512-parity-run-config-v1"
)
POLICY_E2E_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS = frozenset(
    {
        POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    }
)
POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS = frozenset(
    {
        POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        *POLICY_E2E_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS,
    }
)
POLICY_E2E_METHOD_RUN_CONFIG_SCHEMAS = frozenset(
    {
        POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
        POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
        *POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS,
    }
)
# Stabilization bridge for this historical run-config family. It proves that
# observation and action identities are explicit, but it is not a replacement
# for the newer method-specific NoTool/Crop/TGVF/Atomic @512 schemas.
POLICY_E2E_SMOKE_CODE_REPOSITORY = "Miocio-nora/TGVF-E2E-RL"
POLICY_E2E_SMOKE_JUDGE_MODE = "not_applicable"
POLICY_E2E_SMOKE_REWARD_TASK = "multiple_choice"
POLICY_E2E_SMOKE_ANSWER_VERIFIER = "exact_match"
POLICY_E2E_SMOKE_SEED_DERIVATION_NAME = "content-addressed-vllm-turn-rng-v1"
POLICY_E2E_MIXED_REWARD_TASK = "mixed"
POLICY_E2E_MIXED_ANSWER_VERIFIER = "rule_first_qwen25_72b"
POLICY_E2E_MIXED_JUDGE_MODE = "qwen25_72b_semantic_fallback"
POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER = "rule_first_explicit_alternate"
POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE = "explicit_alternate_semantic_fallback"
POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER = (
    "visual_always_qwen25_72b_thinklite_rule_first"
)
POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256 = (
    "9f8136fc11af71e9debb3ee2eb040592ab5c524678ed8b90a1be22dd02b835e9"
)
POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE = "qwen25_72b_always_visual"
POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_CONFIG_SHA256 = (
    "26b733aa079fa3adc4c0eeddb7e847c15c809a1dcc2affe2bc6947e6e7ac1dee"
)
POLICY_E2E_DEEPEYES_RULE_FIRST_JUDGE_CONFIG_SHA256 = (
    "1ec38f640f943702ad812dc367fc66edf843a663a1c1048ebb39a0d25fac18a9"
)


def _fixed_contract_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256 = _fixed_contract_sha256(
    {
        "schema": POLICY_E2E_SMOKE_SEED_DERIVATION_NAME,
        "rng_state_schema": "tgvf-vllm-turn-rng-v1",
        "rng_state_fields": (
            "master_seed",
            "stream_identity_sha256",
            "behavior_policy.run_id",
            "behavior_policy.optimizer_step",
            "behavior_policy.weights_sha256",
            "prompt_token_ids_sha256",
            "turn_index",
        ),
        "rng_state_digest": "sha256-canonical-json",
        "seed_digest": "sha256(bytes('tgvf-vllm-seed-v1\\0') + rng_state_sha256)",
        "seed_projection": "first-8-big-endian-mod-(2**31-1)",
    }
)

_POLICY_E2E_MCQ_CANDIDATE_CONTRACT_V2 = {
    "schema": "terminal-mcq-decision-v2",
    "decision_order": (
        "final-nonempty-line-canonical-A-through-H",
        "last-explicit-answer-option-choice-range-marker",
    ),
    "candidate_preprocessing": (
        "strip-qwen-im-end-or-endoftext-suffix",
        "unwrap-whole-answer-tag-or-latex-boxed",
        "strip-qwen-im-end-or-endoftext-suffix",
        "strip-markdown-emphasis",
    ),
    "canonical_final_line": (
        "parenthesized-or-bracketed-A-through-H",
        "A-through-H-followed-by-period-or-colon-and-optional-text",
        "bare-A-through-H-only",
    ),
    "explicit_marker_scope": "entire-candidate",
    "arbitrary_prose": (
        "unresolved-without-final-line-canonical-letter-or-explicit-marker"
    ),
    "expected": "same-deterministic-parser",
    "fallback_when_unparsed": "strip-casefold-collapse-whitespace-exact",
    "judge": "disabled",
}

POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256 = _fixed_contract_sha256(
    {
        "schema": "policy-e2e-smoke-mcq-verifier-v3",
        "task": POLICY_E2E_SMOKE_REWARD_TASK,
        "route": "multiple_choice_rule",
        "mcq": _POLICY_E2E_MCQ_CANDIDATE_CONTRACT_V2,
    }
)
POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256 = _fixed_contract_sha256(
    {
        "schema": "policy-e2e-mixed-answer-verifier-v2",
        "routes": {
            "mcq": _POLICY_E2E_MCQ_CANDIDATE_CONTRACT_V2,
            "math": "normalized_exact_then_numeric_then_qwen25_72b",
            "open_vqa": "normalized_exact_then_qwen25_72b",
        },
        "judge_failure": "abort_reward_batch",
        "mcq_judge_calls": "forbidden",
    }
)


def policy_e2e_mixed_alternate_answer_verifier_sha256(
    model_identity: ArtifactIdentity,
) -> str:
    """Bind the rule-first verifier contract to one exact alternate model.

    The default verifier digest explicitly names Qwen2.5-72B in its semantic
    fallback routes.  Reusing that digest for another model would make reward
    receipts scientifically false, so every explicit alternate derives a new
    verifier digest from the complete loaded model identity.
    """

    if not isinstance(model_identity, ArtifactIdentity):
        raise TypeError("alternate verifier requires an ArtifactIdentity")
    identity_payload = {
        "namespace": model_identity.namespace,
        "name": model_identity.name,
        "version": model_identity.version,
        "sha256": model_identity.sha256,
    }
    return _fixed_contract_sha256(
        {
            "schema": "policy-e2e-mixed-answer-verifier-explicit-alternate-v1",
            "routes": {
                "mcq": _POLICY_E2E_MCQ_CANDIDATE_CONTRACT_V2,
                "math": "normalized_exact_then_numeric_then_explicit_alternate",
                "open_vqa": "normalized_exact_then_explicit_alternate",
            },
            "semantic_fallback_model_identity": identity_payload,
            "judge_failure": "abort_reward_batch",
            "mcq_judge_calls": "forbidden",
        }
    )


# Historical verifier identities remain readable only for immutable evaluation
# snapshots.  They are not accepted by the default training/launch loader: the
# implementation behind these identities was superseded, so silently treating
# them as the current verifier would change the reward contract of an old run.
POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256 = (
    "2a3d5fa4b7e594939aabb2d1b1192499deea86040d980374f7bc8af3e9082e1c"
)
POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256 = (
    "661133336fc1db8b4a14a360efa84fc4180f040b7f1be4992f83b4d5cdda8e17"
)
POLICY_E2E_SMOKE_CAP_ERROR_SHA256 = StandardToolError(
    code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
    message=(
        "The maximum of 4 tool-call attempts has been reached; "
        "this call was not executed."
    ),
    attempt_index=4,
    recoverable=True,
    maximum_tool_calls=4,
).payload_sha256
POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256 = StandardToolError(
    code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
    message=(
        "The maximum of 1 tool-call attempts has been reached; "
        "this call was not executed."
    ),
    attempt_index=1,
    recoverable=True,
    maximum_tool_calls=1,
).payload_sha256
POLICY_E2E_PIXEL512_SIX_CALL_CAP_ERROR_SHA256 = StandardToolError(
    code=ToolErrorCode.TOOL_CALL_LIMIT_EXCEEDED.value,
    message=(
        "The maximum of 6 tool-call attempts has been reached; "
        "this call was not executed."
    ),
    attempt_index=6,
    recoverable=True,
    maximum_tool_calls=6,
).payload_sha256
POLICY_E2E_AGENT_LOOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "policy"
    / "agent_loops"
    / "tgvf_native_policy_v1.yaml"
)
POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN = (
    "tgvf_rl.framework.verl.policy_runtime.PolicyE2ERuntimeInvocationFactory"
)


@dataclass(frozen=True, slots=True)
class SmokeSelectedMCQSample:
    sample_id: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str
    data_source: str
    task_kind: str = "mcq"


@dataclass(frozen=True, slots=True)
class SmokeDatasetSelection:
    kind: str
    root: Path
    runtime_binding: (
        DeepEyes47KRuntimeBinding
        | PolicyT1RLRuntimeBinding
        | PolicyT1MixedRuntimeBinding
        | PolicyTeacherQuarterMixRuntimeBinding
    )
    samples_sha256: str
    iteration_identity_sha256: str
    sample_id: str | None
    cursor: int | None
    selected_sample: SmokeSelectedMCQSample | None


_PIXEL512_PARITY_METHOD_BY_SCHEMA = {
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA: (
        PolicyMethodProfile.NO_TOOL
    ),
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA: (
        PolicyMethodProfile.CROP
    ),
    POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA: (
        PolicyMethodProfile.TGVF_SHORT
    ),
    POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA: (
        PolicyMethodProfile.TGVF_TARGET_GUIDE_V2
    ),
    POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA: (
        PolicyMethodProfile.ATOMIC
    ),
}


def pixel512_parity_method_for_schema(
    schema_version: object,
) -> PolicyMethodProfile | None:
    """Classify one run schema without treating unrelated historical schemas alike."""

    if not isinstance(schema_version, str):
        return None
    return _PIXEL512_PARITY_METHOD_BY_SCHEMA.get(schema_version)


# Import compatibility for the unfinished historical launcher overlay. New code
# reads ``PolicyE2ESmokeRunConfig.method.profile`` instead of inferring from a
# resolution-bearing schema name.
PolicyPixel512ParityMethod = PolicyMethodProfile


@dataclass(frozen=True, slots=True)
class PolicyMethodMatrixBinding:
    """Explicit method identity shared by arbitrary-resolution matrix arms."""

    matrix_id: str
    profile: PolicyMethodProfile
    legacy_schema_alias: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.matrix_id, str) or not self.matrix_id.strip():
            raise ValueError("method.matrix_id must be a non-empty string")
        if not isinstance(self.profile, PolicyMethodProfile):
            raise TypeError("method.profile must be PolicyMethodProfile")
        if self.legacy_schema_alias is not None and (
            not isinstance(self.legacy_schema_alias, str)
            or self.legacy_schema_alias
            not in POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS
        ):
            raise ValueError("method legacy schema alias is invalid")


class RP66AdapterUpdateMode(str, Enum):
    """Optimizer ownership of the RP67/RP66-compatible TGVF adapter."""

    JOINT = "joint"
    FROZEN_ADAPTER = "frozen_adapter"


@dataclass(frozen=True, slots=True)
class SmokeRepresentationBinding:
    artifact_path: Path
    artifact_file_sha256: str
    artifact: ArtifactIdentity
    expected_run_id: str
    expected_run_identity_sha256: str
    conditioning: TargetConditioningConfig
    adapter_update_mode: RP66AdapterUpdateMode = RP66AdapterUpdateMode.JOINT

    @property
    def adapter_trainable(self) -> bool:
        return self.adapter_update_mode is RP66AdapterUpdateMode.JOINT


@dataclass(frozen=True, slots=True)
class SmokeProtocolBinding:
    prompt_sha256: str
    cap_error_sha256: str
    tool_profile: NativeToolCapabilityProfile
    tool_schema_sha256: str
    enabled_tool_names: tuple[str, ...]
    maximum_tool_calls: int
    success_observation_protocol_id: NativeSuccessObservationProtocolId | None = None
    action_boundary_protocol_id: NativeActionBoundaryProtocolId | None = None


@dataclass(frozen=True, slots=True)
class SmokeRolloutRNGBinding:
    master_seed: int
    derivation_name: str
    derivation_sha256: str


@dataclass(frozen=True, slots=True)
class SmokeRewardBinding:
    profile: str
    task_kind: str
    answer_verifier: str
    answer_verifier_sha256: str
    judge_mode: str
    judge_reason: str
    answer_weight: float | None
    format_weight: float | None
    conditional_tool_weight: float | None
    protocol_error_penalty: float | None = None
    answer_reward_scale: float | None = None
    repeated_call_penalty: float | None = None
    judge_config_path: Path | None = None
    judge_config_sha256: str | None = None
    tool_utility: TGVFToolUtilityRuntimeBinding | None = None
    tool_utility_reward_enabled: bool | None = None
    focus_reward_enabled: bool | None = None
    grounding_reward_enabled: bool | None = None
    visual_quality_judge_config_path: Path | None = None
    visual_quality_judge_config_sha256: str | None = None
    visual_quality_judge_identity: ArtifactIdentity | None = None
    visual_quality_judge_mode: str | None = None
    judge_model_route: str = "qwen2.5_72b"
    alternate_judge_model_name: str | None = None
    alternate_judge_model_identity: ArtifactIdentity | None = None
    alternate_semantics_acknowledged: bool = False

    def __post_init__(self) -> None:
        if self.judge_model_route not in {"qwen2.5_72b", "explicit_alternate"}:
            raise ValueError("reward judge model route is invalid")
        if type(self.alternate_semantics_acknowledged) is not bool:
            raise TypeError("alternate judge semantic acknowledgement must be bool")
        if self.judge_model_route == "qwen2.5_72b":
            if (
                self.alternate_judge_model_name is not None
                or self.alternate_judge_model_identity is not None
                or self.alternate_semantics_acknowledged
            ):
                raise ValueError(
                    "default Qwen2.5-72B reward route cannot carry alternate binding"
                )
            return
        if (
            not isinstance(self.alternate_judge_model_name, str)
            or not self.alternate_judge_model_name.strip()
        ):
            raise ValueError("explicit alternate judge requires its model name")
        if not isinstance(self.alternate_judge_model_identity, ArtifactIdentity):
            raise ValueError("explicit alternate judge requires its model identity")
        if not self.alternate_semantics_acknowledged:
            raise ValueError(
                "explicit alternate judge requires semantic acknowledgement"
            )


@dataclass(frozen=True, slots=True)
class SmokeOptimizerBinding:
    name: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    maximum_gradient_norm: float


@dataclass(frozen=True, slots=True)
class SmokeSchedulerBinding:
    name: str
    warmup_steps: int
    total_steps: int
    minimum_learning_rate_ratio: float


@dataclass(frozen=True, slots=True)
class SmokePrecisionBinding:
    parameter_dtype: str
    reduce_dtype: str
    optimizer_state_dtype: str
    autocast_dtype: str
    gradient_scaler_enabled: bool
    allow_tf32: bool


@dataclass(frozen=True, slots=True)
class SmokeAccumulationBinding:
    global_prompt_batch_size: int
    prompt_micro_batch_size_per_rank: int
    rollout_prompt_micro_batch_size_per_engine: int
    gradient_accumulation_steps: int


@dataclass(frozen=True, slots=True)
class SmokeDistributedBinding:
    physical_gpu_ids: tuple[int, ...]
    logical_gpu_ids: tuple[int, ...]
    world_size: int
    actor_logical_gpu_ids: tuple[int, ...]
    rollout_logical_gpu_ids: tuple[int, ...]
    fsdp_strategy: str
    fsdp_reshard_after_forward: bool
    rollout_backend: str
    vllm_tensor_parallel_size: int
    placement: str
    weight_sync_mode: str
    weight_sync_interval_optimizer_steps: int


@dataclass(frozen=True, slots=True)
class SmokeCapacityBinding:
    max_prompt_length: int
    actor_ppo_max_token_len_per_gpu: int
    rollout_log_prob_max_token_len_per_gpu: int
    reference_log_prob_max_token_len_per_gpu: int
    vllm_gpu_memory_utilization: float
    vllm_max_num_batched_tokens: int
    vllm_max_model_len: int
    vllm_max_num_seqs: int
    vllm_enable_chunked_prefill: bool
    vllm_enforce_eager: bool

    @property
    def response_transport_length(self) -> int:
        """Maximum response-side width that still fits the bound context."""

        return self.vllm_max_model_len - self.max_prompt_length


@dataclass(frozen=True, slots=True)
class SmokePerformanceBinding:
    """Explicit execution-only performance choices for one Policy run.

    The binding deliberately distinguishes the rollout-logprob *bypass* from
    disabling behavior-logprob collection. Exact replay still records the
    post-transform behavior probabilities; the bypass avoids recomputing the
    old-policy logprobs after rollout. Judge concurrency is explicitly local
    to each AgentLoop worker process, never a run-global request budget.
    """

    dynamic_token_batching: bool
    use_remove_padding: bool
    enable_gradient_checkpointing: bool
    vllm_enable_prefix_caching: bool
    vllm_enable_chunked_prefill: bool
    vllm_enable_cuda_graph: bool
    vllm_cuda_graph_capture_sizes: tuple[int, ...]
    vllm_tensor_parallel_size: int
    rollout_logprob_bypass: bool
    reference_replay_mode: str
    judge_dispatch_mode: str
    judge_max_concurrency_per_worker: int

    def __post_init__(self) -> None:
        boolean_fields = (
            "dynamic_token_batching",
            "use_remove_padding",
            "enable_gradient_checkpointing",
            "vllm_enable_prefix_caching",
            "vllm_enable_chunked_prefill",
            "vllm_enable_cuda_graph",
            "rollout_logprob_bypass",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"performance.{name} must be boolean")
        if (
            type(self.vllm_tensor_parallel_size) is not int
            or self.vllm_tensor_parallel_size <= 0
        ):
            raise ValueError(
                "performance.vllm_tensor_parallel_size must be a positive integer"
            )
        if not isinstance(self.vllm_cuda_graph_capture_sizes, tuple) or any(
            type(value) is not int or value <= 0
            for value in self.vllm_cuda_graph_capture_sizes
        ):
            raise ValueError(
                "performance.vllm_cuda_graph_capture_sizes must contain positive integers"
            )
        if tuple(sorted(set(self.vllm_cuda_graph_capture_sizes))) != (
            self.vllm_cuda_graph_capture_sizes
        ):
            raise ValueError(
                "performance.vllm_cuda_graph_capture_sizes must be strictly "
                "increasing and unique"
            )
        if not self.vllm_enable_cuda_graph and self.vllm_cuda_graph_capture_sizes:
            raise ValueError("disabled CUDA graph requires empty capture sizes")
        if self.rollout_logprob_bypass is not True:
            raise ValueError(
                "exact replay requires performance.rollout_logprob_bypass=true"
            )
        if self.reference_replay_mode not in {"off", "full_diagnostic"}:
            raise ValueError(
                "performance.reference_replay_mode must be off or full_diagnostic"
            )
        if self.judge_dispatch_mode not in {
            "inherit",
            "inline",
            "dedicated_thread_pool",
        }:
            raise ValueError(
                "performance.judge_dispatch_mode must be inherit, inline, or "
                "dedicated_thread_pool"
            )
        if (
            type(self.judge_max_concurrency_per_worker) is not int
            or not 1 <= self.judge_max_concurrency_per_worker <= 256
        ):
            raise ValueError(
                "performance.judge_max_concurrency_per_worker must be in [1, 256]"
            )
        if (
            self.judge_dispatch_mode in {"inherit", "inline"}
            and self.judge_max_concurrency_per_worker != 1
        ):
            raise ValueError(
                "inherit/inline judge dispatch requires maximum concurrency 1"
            )


@dataclass(frozen=True, slots=True)
class SmokeFrameworkBinding:
    agent_loop_config_path: Path
    agent_loop_config_sha256: str
    runtime_invocation_factory_fqn: str
    server_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class SmokeTrainingBinding:
    total_training_epochs: int
    maximum_optimizer_steps: int
    checkpoint_steps: tuple[int, ...]
    logger: tuple[str, ...]
    project_name: str
    validation_before_training: bool
    validation_frequency: int
    resume_mode: str
    resume_from_path: Path | None
    maximum_actor_checkpoints_to_keep: int


@dataclass(frozen=True, slots=True)
class SmokeOutputBinding:
    root: Path
    checkpoint_directory: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class PolicyE2ESmokeRunConfig:
    """One complete non-formal smoke identity with no execution side effects."""

    run_id: str
    code: CodeIdentity
    model: ModelIdentity
    dataset: SmokeDatasetSelection
    representation: SmokeRepresentationBinding
    protocol: SmokeProtocolBinding
    policy: PolicyPilotV1Config
    rollout_rng: SmokeRolloutRNGBinding
    reward: SmokeRewardBinding
    optimizer: SmokeOptimizerBinding
    scheduler: SmokeSchedulerBinding
    precision: SmokePrecisionBinding
    accumulation: SmokeAccumulationBinding
    distributed: SmokeDistributedBinding
    capacity: SmokeCapacityBinding
    framework: SmokeFrameworkBinding
    training: SmokeTrainingBinding
    output: SmokeOutputBinding
    source_path: Path
    source_sha256: str
    canonical_json: str
    performance: SmokePerformanceBinding | None = None
    method: PolicyMethodMatrixBinding | None = None
    deepeyes_control: DeepEyesStrictControlBinding | None = None
    formal_pilot: bool = False
    schema_version: str = POLICY_E2E_SMOKE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        accepted = {
            POLICY_E2E_SMOKE_CONFIG_SCHEMA: False,
            POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA: True,
            POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA: False,
            **{
                schema: False
                for schema in POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS
            },
        }
        if self.schema_version not in accepted:
            raise ValueError("policy E2E run config schema mismatch")
        if self.formal_pilot is not accepted[self.schema_version]:
            raise ValueError("policy E2E run formal_pilot mode differs from schema")
        legacy_profile = pixel512_parity_method_for_schema(self.schema_version)
        if self.schema_version in {
            POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
            POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA,
        }:
            if self.method is None or self.method.legacy_schema_alias is not None:
                raise ValueError(
                    "method-matrix schema requires an explicit method binding"
                )
        elif legacy_profile is not None:
            if (
                self.method is None
                or self.method.profile is not legacy_profile
                or self.method.legacy_schema_alias != self.schema_version
            ):
                raise ValueError("legacy PRL26 schema method binding differs")
        elif self.method is not None:
            raise ValueError("non-method run config cannot carry a method binding")
        if self.performance is not None and not isinstance(
            self.performance, SmokePerformanceBinding
        ):
            raise TypeError("performance must be SmokePerformanceBinding or None")
        if (
            self.schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA
            and self.performance is None
        ):
            raise ValueError(
                "method-matrix v2 requires an explicit performance binding"
            )

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def as_record(self) -> dict[str, Any]:
        record = json.loads(self.canonical_json)
        if not isinstance(record, dict):  # pragma: no cover - construction invariant
            raise RuntimeError("canonical smoke configuration is not an object")
        return record


_PUBLIC_RUN_CONFIG_MODULE = "tgvf_rl.policy.run_config"
_RUN_CONFIG_SCHEMA_TYPES = (
    SmokeSelectedMCQSample,
    SmokeDatasetSelection,
    PolicyMethodMatrixBinding,
    SmokeRepresentationBinding,
    SmokeProtocolBinding,
    SmokeRolloutRNGBinding,
    SmokeRewardBinding,
    SmokeOptimizerBinding,
    SmokeSchedulerBinding,
    SmokePrecisionBinding,
    SmokeAccumulationBinding,
    SmokeDistributedBinding,
    SmokeCapacityBinding,
    SmokePerformanceBinding,
    SmokeFrameworkBinding,
    SmokeTrainingBinding,
    SmokeOutputBinding,
    PolicyE2ESmokeRunConfig,
)

# These objects historically lived in ``policy.run_config``.  Keep that public
# and pickle identity while the implementation moves behind the facade.
for _schema_type in _RUN_CONFIG_SCHEMA_TYPES:
    rebind_public_class(
        _schema_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUN_CONFIG_MODULE,
    )
del _schema_type


__all__ = [
    "POLICY_E2E_AGENT_LOOP_CONFIG_PATH",
    "POLICY_E2E_ATOMIC_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA",
    "POLICY_E2E_MIXED_ALTERNATE_ANSWER_VERIFIER",
    "POLICY_E2E_MIXED_ALTERNATE_JUDGE_MODE",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256",
    "POLICY_E2E_MIXED_JUDGE_MODE",
    "POLICY_E2E_MIXED_REWARD_TASK",
    "POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_V2_SCHEMA",
    "POLICY_E2E_METHOD_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_PIXEL512_SIX_CALL_CAP_ERROR_SHA256",
    "POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN",
    "POLICY_E2E_SMOKE_ANSWER_VERIFIER",
    "POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256",
    "POLICY_E2E_SMOKE_CAP_ERROR_SHA256",
    "POLICY_E2E_SMOKE_CODE_REPOSITORY",
    "POLICY_E2E_SMOKE_CONFIG_SCHEMA",
    "POLICY_E2E_SMOKE_JUDGE_MODE",
    "POLICY_E2E_SMOKE_REWARD_TASK",
    "POLICY_E2E_SMOKE_SEED_DERIVATION_NAME",
    "POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256",
    "POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256",
    "POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS",
    "PolicyE2ESmokeRunConfig",
    "PolicyMethodMatrixBinding",
    "SmokeAccumulationBinding",
    "SmokeCapacityBinding",
    "SmokeDatasetSelection",
    "SmokeDistributedBinding",
    "SmokeFrameworkBinding",
    "SmokeOptimizerBinding",
    "SmokeOutputBinding",
    "SmokePerformanceBinding",
    "SmokePrecisionBinding",
    "SmokeProtocolBinding",
    "SmokeRepresentationBinding",
    "SmokeRewardBinding",
    "SmokeRolloutRNGBinding",
    "SmokeSchedulerBinding",
    "SmokeSelectedMCQSample",
    "SmokeTrainingBinding",
    "policy_e2e_mixed_alternate_answer_verifier_sha256",
    "pixel512_parity_method_for_schema",
]
