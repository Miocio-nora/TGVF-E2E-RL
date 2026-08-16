"""Strict, read-only configuration for one non-formal policy E2E smoke.

The loader binds launch inputs but deliberately performs no launch work: it
does not import a model, initialize CUDA, create an output directory, or write
any file.  Formal Policy Pilot manifests remain a separate contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.identity import ArtifactIdentity, CodeIdentity, ModelIdentity
from tgvf_rl.contracts.tokens import LogProbMeasurement
from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_MANIFEST_FILE,
    DEEPEYES47K_RUNTIME_SCHEMA_VERSION,
    DEEPEYES47K_SAMPLES_FILE,
    DEEPEYES47K_SCHEMA_VERSION,
    DEEPEYES47K_SHUFFLE_ALGORITHM,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyes47KRuntimeBinding,
    POLICY_T1_ARXIVQA_DATASET_KIND,
    POLICY_T1_MIXED_DATASET_KIND,
    PolicyT1DecisionStage,
    PolicyT1MixedRuntimeBinding,
    PolicyT1RLRuntimeBinding,
    policy_t1_mixed_iteration_identity_sha256,
    policy_t1_rl_iteration_identity_sha256,
    verify_policy_t1_mixed_artifact_binding,
    verify_policy_t1_rl_artifact_binding,
)
from tgvf_rl.data.policy_teacher_quarter_mix import (
    POLICY_TEACHER_QUARTER_MIX_DATASET_KIND,
    PolicyTeacherQuarterMixRuntimeBinding,
    policy_teacher_quarter_mix_iteration_identity_sha256,
    verify_policy_teacher_quarter_mix_artifact_binding,
)
from tgvf_rl.data.policy_teacher_ratio_mix import (
    POLICY_TEACHER_RATIO_MIX_DATASET_KIND,
    PolicyTeacherRatioMixRuntimeBinding,
    policy_teacher_ratio_mix_iteration_identity_sha256,
    verify_policy_teacher_ratio_mix_artifact_binding,
)
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityRuntimeBinding,
    load_tgvf_tool_utility_runtime_binding,
)
from tgvf_rl.judges import (
    load_openai_compatible_judge,
    load_tgvf_visual_quality_judge,
)
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeToolCapabilityProfile,
    StandardToolError,
    ToolErrorCode,
    visual_tool_prompt_identity,
)
from tgvf_rl.protocol.native import native_assistant_dialect_for_model
from tgvf_rl.rewards.schema import pilot_reward_weight_profile_name
from tgvf_rl.rewards.stage3_shaped import STAGE3_SHAPED_REWARD_VERSION

from .config import (
    POLICY_PILOT_FUNCTIONAL_CANARY_SAMPLING_SCALE,
    POLICY_PILOT_V1_ACCEPTED_LEARNING_RATES,
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_HISTORICAL_THINKING_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_NAME,
    POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_PATH,
    POLICY_PILOT_V1_MODEL_FAMILY,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_PATH,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
    POLICY_PILOT_V1_TOOL_PROFILE,
    POLICY_PILOT_V1_VLLM_VERSION,
    DecoderLoRAConfig,
    PilotGRPOConfig,
    PilotSamplingConfig,
    PolicyPilotV1Config,
    PolicyTGVFStage3ExperimentConfig,
    PolicyTrainableRP66ExperimentConfig,
    PolicyCropTGVFMatchedExperimentConfig,
    PolicyVisualToolExperimentConfig,
)
from .crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from .tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)


POLICY_E2E_SMOKE_CONFIG_SCHEMA = "policy-e2e-smoke-config-v3"
POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA = "policy-e2e-mixed-run-config-v4"
POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA = "policy-e2e-formal-pilot-config-v1"
POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA = "policy-e2e-stage3-shaped-run-config-v1"
POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA = (
    "policy-e2e-deepeyes-scaled-crop-run-config-v1"
)
POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA = (
    "policy-e2e-trainable-rp66-deepeyes-matched-run-config-v1"
)
POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA = (
    "policy-e2e-rp66-deepeyes-matched-control-run-config-v2"
)
POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA = (
    "policy-e2e-rp66-deepeyes-matched-control-run-config-v3"
)
POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA = (
    "policy-e2e-rp66-shaped-matched-control-run-config-v1"
)
POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA = (
    "policy-e2e-rp66-tfree-matched-control-run-config-v1"
)
POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA = (
    "policy-e2e-crop-tgvf-tfree-deepeyes-matched-run-config-v1"
)
POLICY_E2E_RP66_EXPLICIT_CONTROL_RUN_CONFIG_SCHEMAS = frozenset(
    {
        POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
    }
)
POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS = frozenset(
    {
        POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        *POLICY_E2E_RP66_EXPLICIT_CONTROL_RUN_CONFIG_SCHEMAS,
    }
)
POLICY_E2E_TGVF_BACKED_MATCHED_RUN_CONFIG_SCHEMAS = frozenset(
    {
        *POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }
)
POLICY_E2E_SMOKE_CODE_REPOSITORY = "Miocio-nora/TGVF-E2E-RL"
POLICY_E2E_SMOKE_JUDGE_MODE = "not_applicable"
POLICY_E2E_SMOKE_REWARD_TASK = "multiple_choice"
POLICY_E2E_SMOKE_ANSWER_VERIFIER = "exact_match"
POLICY_E2E_SMOKE_SEED_DERIVATION_NAME = "content-addressed-vllm-turn-rng-v1"
POLICY_E2E_MIXED_REWARD_TASK = "mixed"
POLICY_E2E_MIXED_ANSWER_VERIFIER = "rule_first_qwen25_72b"
POLICY_E2E_MIXED_JUDGE_MODE = "qwen25_72b_semantic_fallback"

_SUPPORTED_POLICY_MODEL_IDENTITIES = frozenset(
    {
        (
            POLICY_PILOT_V1_MODEL_NAME,
            POLICY_PILOT_V1_MODEL_PATH,
            POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
        ),
        (
            POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_NAME,
            POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_PATH,
            POLICY_PILOT_V1_HISTORICAL_THINKING_CHAT_TEMPLATE_SHA256,
        ),
    }
)

# Read compatibility for immutable pre-Instruct run records.  New runs always
# bind the dialect-specific identity returned by visual_tool_prompt_identity().
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
# Historical run configs bind the verifier contract that was active when the
# experiment was launched.  Keeping those known identities readable is
# necessary for post-training evaluation; it does not relax the contract for
# newly launched trainable-RP66 runs.
_LEGACY_POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256S = frozenset(
    {"661133336fc1db8b4a14a360efa84fc4180f040b7f1be4992f83b4d5cdda8e17"}
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
POLICY_E2E_TRAINABLE_RP66_SIX_CALL_CAP_ERROR_SHA256 = StandardToolError(
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
POLICY_E2E_TRAINABLE_RP66_AGENT_LOOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "policy"
    / "agent_loops"
    / "prl15_trainable_rp66_matched.yaml"
)
POLICY_E2E_CROP_TGVF_MATCHED_AGENT_LOOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "policy"
    / "agent_loops"
    / "prl20_crop_tgvf_deepeyes_matched.yaml"
)
POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN = (
    "tgvf_rl.framework.verl.policy_runtime.PolicyE2ERuntimeInvocationFactory"
)

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FQN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MCQ_OPTION_PATTERN = re.compile(r"(?im)(?:^|\n)\s*(?:\(([A-H])\)|([A-H])[.):.])\s+\S")
_MCQ_LETTER_ANSWER = re.compile(r"^\(?[A-H]\)?(?:[.):])?$", re.IGNORECASE)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "formal_pilot",
    "run_id",
    "code",
    "model",
    "dataset",
    "representation",
    "protocol",
    "sampling",
    "reward",
    "optimizer",
    "scheduler",
    "precision",
    "accumulation",
    "distributed",
    "capacity",
    "framework",
    "training",
    "output",
}
_DEEPEYES_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_id",
    "snapshot",
    "fixture",
    "source_files",
    "source_total_rows",
    "sample_count",
    "shuffle",
    "samples",
    "images",
    "content_sha256",
}


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
        | PolicyTeacherRatioMixRuntimeBinding
    )
    samples_sha256: str
    iteration_identity_sha256: str
    sample_id: str | None
    cursor: int | None
    selected_sample: SmokeSelectedMCQSample | None


class RP66AdapterUpdateMode(str, Enum):
    """Optimizer ownership of RP66 inside a matched TGVF policy run."""

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
    judge_config_path: Path | None = None
    judge_config_sha256: str | None = None
    tool_utility: TGVFToolUtilityRuntimeBinding | None = None
    tool_utility_reward_enabled: bool | None = None
    focus_reward_enabled: bool | None = None
    grounding_reward_enabled: bool | None = None
    visual_quality_judge_config_path: Path | None = None
    visual_quality_judge_config_sha256: str | None = None
    visual_quality_judge_identity: ArtifactIdentity | None = None


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
    actor_optimizer_offload: bool


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
    permanent_checkpoint_steps: tuple[int, ...]


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
    formal_pilot: bool = False
    schema_version: str = POLICY_E2E_SMOKE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        accepted = {
            POLICY_E2E_SMOKE_CONFIG_SCHEMA: False,
            POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA: True,
            POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA: False,
            POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA: False,
        }
        if self.schema_version not in accepted:
            raise ValueError("policy E2E run config schema mismatch")
        if self.formal_pilot is not accepted[self.schema_version]:
            raise ValueError("policy E2E run formal_pilot mode differs from schema")

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def as_record(self) -> dict[str, Any]:
        record = json.loads(self.canonical_json)
        if not isinstance(record, dict):  # pragma: no cover - construction invariant
            raise RuntimeError("canonical smoke configuration is not an object")
        return record


def formal_deepeyes47k_iteration_identity_sha256(
    binding: DeepEyes47KRuntimeBinding,
    *,
    samples_sha256: str,
) -> str:
    """Compute the identity emitted by the formal DeepEyes runtime loader."""

    if not isinstance(binding, DeepEyes47KRuntimeBinding) or binding.fixture:
        raise ValueError("iteration identity requires a formal DeepEyes binding")
    samples_digest = _sha256(samples_sha256, name="samples_sha256")
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": DEEPEYES47K_RUNTIME_SCHEMA_VERSION,
                "dataset_id": DEEPEYES47K_DATASET_ID,
                "snapshot": DEEPEYES47K_SNAPSHOT,
                "fixture": False,
                "sample_count": DEEPEYES47K_TOTAL_ROWS,
                "shuffle_algorithm": DEEPEYES47K_SHUFFLE_ALGORITHM,
                "shuffle_seed": binding.shuffle_seed,
                "manifest_file_sha256": binding.manifest_file_sha256,
                "content_sha256": binding.content_sha256,
                "samples_sha256": samples_digest,
            }
        )
    ).hexdigest()


def load_policy_e2e_smoke_run_config(
    path: str | Path,
    *,
    allow_external_agent_loop_config: bool = False,
) -> PolicyE2ESmokeRunConfig:
    """Read and validate a complete smoke TOML without launching anything."""

    if type(allow_external_agent_loop_config) is not bool:
        raise ValueError("allow_external_agent_loop_config must be a bool")
    source_path = _existing_file(path, name="config path")
    if source_path.is_symlink():
        raise ValueError("config path must not be a symlink")
    raw = source_path.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        payload = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("policy E2E smoke config is not strict UTF-8 TOML") from error
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("policy E2E smoke top-level fields differ")
    schema_version = payload["schema_version"]
    if schema_version not in {
        POLICY_E2E_SMOKE_CONFIG_SCHEMA,
        POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
        POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }:
        raise ValueError("policy E2E run config schema mismatch")
    formal_pilot = schema_version == POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA
    if payload["formal_pilot"] is not formal_pilot:
        raise ValueError("policy E2E run formal_pilot mode differs from schema")
    mixed_run = schema_version != POLICY_E2E_SMOKE_CONFIG_SCHEMA
    stage3_shaped_run = schema_version in {
        POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }
    rp66_shaped_run = schema_version in {
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }
    tfree_reward_run = schema_version in {
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }
    deepeyes_scaled_crop_run = (
        schema_version == POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA
    )
    rp66_matched_run = schema_version in POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS
    crop_tgvf_matched_run = (
        schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA
    )
    tgvf_backed_matched_run = (
        schema_version in POLICY_E2E_TGVF_BACKED_MATCHED_RUN_CONFIG_SCHEMAS
    )
    run_id = _safe_run_id(payload["run_id"])

    code_table = _table(payload, "code", {"repository", "commit", "dirty"})
    if code_table["repository"] != POLICY_E2E_SMOKE_CODE_REPOSITORY:
        raise ValueError("code.repository differs from the accepted repository")
    commit = _text(code_table["commit"], name="code.commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("code.commit must be a full lowercase Git commit")
    if code_table["dirty"] is not False:
        raise ValueError("policy E2E smoke requires code.dirty=false")
    code = CodeIdentity(code_table["repository"], commit)

    model_table = _table(
        payload,
        "model",
        {
            "family",
            "name",
            "path",
            "tokenizer_length",
            "chat_template_sha256",
            "native_deepstack_enabled",
            "image_max_pixels",
        },
    )
    _require_exact(model_table["family"], POLICY_PILOT_V1_MODEL_FAMILY, "model.family")
    model_name = _text(model_table["name"], name="model.name")
    model_path = _text(model_table["path"], name="model.path")
    _require_exact(
        model_table["tokenizer_length"],
        POLICY_PILOT_V1_TOKENIZER_LENGTH,
        "model.tokenizer_length",
    )
    chat_template_sha256 = _sha256(
        model_table["chat_template_sha256"],
        name="model.chat_template_sha256",
    )
    if (model_name, model_path, chat_template_sha256) not in (
        _SUPPORTED_POLICY_MODEL_IDENTITIES
    ):
        raise ValueError(
            "model name/path/chat-template identity is not an exact supported "
            "Qwen3-VL 8B edition"
        )
    assistant_dialect = native_assistant_dialect_for_model(model_name)
    _require_exact(
        model_table["native_deepstack_enabled"], True, "model.native_deepstack_enabled"
    )
    expected_image_max_pixels = (
        1_003_520
        if deepeyes_scaled_crop_run or tgvf_backed_matched_run
        else 512 * 512
    )
    _require_exact(
        model_table["image_max_pixels"],
        expected_image_max_pixels,
        "model.image_max_pixels",
    )
    model = ModelIdentity(
        family=model_table["family"],
        model_name=model_name,
        revision_or_path=model_path,
        tokenizer_length=model_table["tokenizer_length"],
        chat_template_sha256=chat_template_sha256,
    )

    raw_dataset = payload.get("dataset")
    policy_t1_arxivqa_dataset = (
        isinstance(raw_dataset, Mapping)
        and raw_dataset.get("kind") == POLICY_T1_ARXIVQA_DATASET_KIND
    )
    policy_t1_mixed_dataset = (
        isinstance(raw_dataset, Mapping)
        and raw_dataset.get("kind") == POLICY_T1_MIXED_DATASET_KIND
    )
    policy_teacher_quarter_mix_dataset = (
        isinstance(raw_dataset, Mapping)
        and raw_dataset.get("kind") == POLICY_TEACHER_QUARTER_MIX_DATASET_KIND
    )
    policy_teacher_ratio_mix_dataset = (
        isinstance(raw_dataset, Mapping)
        and raw_dataset.get("kind") == POLICY_TEACHER_RATIO_MIX_DATASET_KIND
    )
    if (
        policy_t1_arxivqa_dataset
        or policy_t1_mixed_dataset
        or policy_teacher_quarter_mix_dataset
        or policy_teacher_ratio_mix_dataset
    ):
        if not mixed_run:
            raise ValueError("Policy T1 retained data requires a mixed/formal run")
        dataset_fields = {
            "kind",
            "root",
            "decision_stage",
            "sample_count",
            "manifest_file_sha256",
            "content_sha256",
            "samples_sha256",
            "iteration_identity_sha256",
            "shuffle_seed",
        }
        if policy_teacher_ratio_mix_dataset:
            dataset_fields.add("teacher_percentage")
        dataset_table = _table(
            payload,
            "dataset",
            dataset_fields,
        )
        dataset_kind = str(dataset_table["kind"])
        dataset_root = _existing_directory(dataset_table["root"], name="dataset.root")
        manifest_file_sha256 = _sha256(
            dataset_table["manifest_file_sha256"],
            name="dataset.manifest_file_sha256",
        )
        content_sha256 = _sha256(
            dataset_table["content_sha256"], name="dataset.content_sha256"
        )
        shuffle_seed = _nonnegative_int(
            dataset_table["shuffle_seed"], name="dataset.shuffle_seed"
        )
        expected_sample_count = _positive_int(
            dataset_table["sample_count"], name="dataset.sample_count"
        )
        samples_sha256 = _sha256(
            dataset_table["samples_sha256"], name="dataset.samples_sha256"
        )
        iteration_sha256 = _sha256(
            dataset_table["iteration_identity_sha256"],
            name="dataset.iteration_identity_sha256",
        )
        if policy_teacher_ratio_mix_dataset:
            _require_exact(
                dataset_table["decision_stage"],
                "final",
                "dataset.decision_stage",
            )
            teacher_percentage = _positive_int(
                dataset_table["teacher_percentage"],
                name="dataset.teacher_percentage",
            )
            runtime_binding = PolicyTeacherRatioMixRuntimeBinding(
                manifest_file_sha256=manifest_file_sha256,
                content_sha256=content_sha256,
                schedule_seed=shuffle_seed,
                expected_sample_count=expected_sample_count,
                teacher_percentage=teacher_percentage,
            )
            if iteration_sha256 != (
                policy_teacher_ratio_mix_iteration_identity_sha256(
                    runtime_binding, samples_sha256=samples_sha256
                )
            ):
                raise ValueError(
                    "dataset iteration identity differs from its teacher-ratio "
                    "mixture binding"
                )
            verify_policy_teacher_ratio_mix_artifact_binding(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
            )
        elif policy_teacher_quarter_mix_dataset:
            _require_exact(
                dataset_table["decision_stage"],
                "final",
                "dataset.decision_stage",
            )
            runtime_binding = PolicyTeacherQuarterMixRuntimeBinding(
                manifest_file_sha256=manifest_file_sha256,
                content_sha256=content_sha256,
                schedule_seed=shuffle_seed,
                expected_sample_count=expected_sample_count,
            )
            if iteration_sha256 != (
                policy_teacher_quarter_mix_iteration_identity_sha256(
                    runtime_binding, samples_sha256=samples_sha256
                )
            ):
                raise ValueError(
                    "dataset iteration identity differs from its teacher-quarter "
                    "mixture binding"
                )
            verify_policy_teacher_quarter_mix_artifact_binding(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
            )
        elif policy_t1_mixed_dataset:
            _require_exact(
                dataset_table["decision_stage"],
                "final",
                "dataset.decision_stage",
            )
            runtime_binding = PolicyT1MixedRuntimeBinding(
                manifest_file_sha256=manifest_file_sha256,
                content_sha256=content_sha256,
                shuffle_seed=shuffle_seed,
                expected_sample_count=expected_sample_count,
            )
            if iteration_sha256 != policy_t1_mixed_iteration_identity_sha256(
                runtime_binding, samples_sha256=samples_sha256
            ):
                raise ValueError(
                    "dataset iteration identity differs from its mixed T1 binding"
                )
            verify_policy_t1_mixed_artifact_binding(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
            )
        else:
            try:
                decision_stage = PolicyT1DecisionStage(dataset_table["decision_stage"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "dataset.decision_stage must be provisional or final"
                ) from error
            runtime_binding = PolicyT1RLRuntimeBinding(
                manifest_file_sha256=manifest_file_sha256,
                content_sha256=content_sha256,
                shuffle_seed=shuffle_seed,
                decision_stage=decision_stage,
                expected_sample_count=expected_sample_count,
            )
            if iteration_sha256 != policy_t1_rl_iteration_identity_sha256(
                runtime_binding, samples_sha256=samples_sha256
            ):
                raise ValueError(
                    "dataset iteration identity differs from its T1 binding"
                )
            verify_policy_t1_rl_artifact_binding(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
            )
        sample_id = None
        cursor = None
        selected_sample = None
    else:
        if isinstance(raw_dataset, Mapping) and "kind" in raw_dataset:
            raise ValueError("dataset.kind is unsupported")
        dataset_fields = {
            "root",
            "dataset_id",
            "snapshot",
            "sample_count",
            "manifest_file_sha256",
            "content_sha256",
            "samples_sha256",
            "iteration_identity_sha256",
            "shuffle_seed",
        }
        if not mixed_run:
            dataset_fields.update({"sample_id", "cursor"})
        dataset_table = _table(payload, "dataset", dataset_fields)
        dataset_kind = "deepeyes47k"
        _require_exact(
            dataset_table["dataset_id"], DEEPEYES47K_DATASET_ID, "dataset.dataset_id"
        )
        _require_exact(
            dataset_table["snapshot"], DEEPEYES47K_SNAPSHOT, "dataset.snapshot"
        )
        _require_exact(
            dataset_table["sample_count"],
            DEEPEYES47K_TOTAL_ROWS,
            "dataset.sample_count",
        )
        dataset_root = _existing_directory(dataset_table["root"], name="dataset.root")
        runtime_binding = DeepEyes47KRuntimeBinding.formal(
            manifest_file_sha256=_sha256(
                dataset_table["manifest_file_sha256"],
                name="dataset.manifest_file_sha256",
            ),
            content_sha256=_sha256(
                dataset_table["content_sha256"], name="dataset.content_sha256"
            ),
            shuffle_seed=_nonnegative_int(
                dataset_table["shuffle_seed"], name="dataset.shuffle_seed"
            ),
        )
        samples_sha256 = _sha256(
            dataset_table["samples_sha256"], name="dataset.samples_sha256"
        )
        iteration_sha256 = _sha256(
            dataset_table["iteration_identity_sha256"],
            name="dataset.iteration_identity_sha256",
        )
        expected_iteration = formal_deepeyes47k_iteration_identity_sha256(
            runtime_binding, samples_sha256=samples_sha256
        )
        if iteration_sha256 != expected_iteration:
            raise ValueError(
                "dataset iteration identity differs from its formal binding"
            )
        if mixed_run:
            _verify_deepeyes_artifact(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
            )
            sample_id = None
            cursor = None
            selected_sample = None
        else:
            sample_id = _text(dataset_table["sample_id"], name="dataset.sample_id")
            cursor = _nonnegative_int(dataset_table["cursor"], name="dataset.cursor")
            if cursor >= DEEPEYES47K_TOTAL_ROWS:
                raise ValueError("dataset.cursor lies outside DeepEyes-47K")
            selected_sample = _verify_deepeyes_files(
                dataset_root,
                binding=runtime_binding,
                samples_sha256=samples_sha256,
                sample_id=sample_id,
                cursor=cursor,
            )
    dataset = SmokeDatasetSelection(
        kind=dataset_kind,
        root=dataset_root,
        runtime_binding=runtime_binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=iteration_sha256,
        sample_id=sample_id,
        cursor=cursor,
        selected_sample=selected_sample,
    )

    representation_fields = {
        "artifact_path",
        "artifact_file_sha256",
        "artifact_manifest_sha256",
        "artifact_namespace",
        "artifact_name",
        "artifact_version",
        "expected_run_id",
        "expected_run_identity_sha256",
        "conditioning",
    }
    if (
        schema_version in POLICY_E2E_RP66_EXPLICIT_CONTROL_RUN_CONFIG_SCHEMAS
        or crop_tgvf_matched_run
    ):
        representation_fields.add("adapter_update_mode")
    representation_table = _table(
        payload,
        "representation",
        representation_fields,
    )
    artifact_path = _existing_file(
        representation_table["artifact_path"], name="representation.artifact_path"
    )
    if artifact_path.is_symlink():
        raise ValueError("representation artifact must not be a symlink")
    artifact_file_sha256 = _sha256(
        representation_table["artifact_file_sha256"],
        name="representation.artifact_file_sha256",
    )
    if _sha256_file(artifact_path) != artifact_file_sha256:
        raise ValueError("representation artifact file SHA256 mismatch")
    artifact = ArtifactIdentity(
        namespace=_text(
            representation_table["artifact_namespace"],
            name="representation.artifact_namespace",
        ),
        name=_text(
            representation_table["artifact_name"],
            name="representation.artifact_name",
        ),
        version=_text(
            representation_table["artifact_version"],
            name="representation.artifact_version",
        ),
        sha256=_sha256(
            representation_table["artifact_manifest_sha256"],
            name="representation.artifact_manifest_sha256",
        ),
    )
    conditioning = _conditioning(representation_table["conditioning"])
    if (
        schema_version in POLICY_E2E_RP66_EXPLICIT_CONTROL_RUN_CONFIG_SCHEMAS
        or crop_tgvf_matched_run
    ):
        try:
            adapter_update_mode = RP66AdapterUpdateMode(
                representation_table["adapter_update_mode"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError("representation.adapter_update_mode is invalid") from error
    else:
        adapter_update_mode = RP66AdapterUpdateMode.JOINT
    representation = SmokeRepresentationBinding(
        artifact_path=artifact_path,
        artifact_file_sha256=artifact_file_sha256,
        artifact=artifact,
        expected_run_id=_text(
            representation_table["expected_run_id"],
            name="representation.expected_run_id",
        ),
        expected_run_identity_sha256=_sha256(
            representation_table["expected_run_identity_sha256"],
            name="representation.expected_run_identity_sha256",
        ),
        conditioning=conditioning,
        adapter_update_mode=adapter_update_mode,
    )

    protocol_table = _table(
        payload,
        "protocol",
        {
            "prompt_sha256",
            "cap_error_sha256",
            "tool_profile",
            "tool_schema_sha256",
            "enabled_tool_names",
            "maximum_tool_calls",
        },
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
    expected_maximum_tool_calls = (
        6
        if tgvf_backed_matched_run
        else 1
        if stage3_shaped_run
        else 4
    )
    _require_exact(
        protocol_table["maximum_tool_calls"],
        expected_maximum_tool_calls,
        "protocol.maximum_tool_calls",
    )
    cap_error_sha256 = _sha256(
        protocol_table["cap_error_sha256"], name="protocol.cap_error_sha256"
    )
    _require_exact(
        cap_error_sha256,
        (
            POLICY_E2E_TRAINABLE_RP66_SIX_CALL_CAP_ERROR_SHA256
            if tgvf_backed_matched_run
            else POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256
            if stage3_shaped_run
            else POLICY_E2E_SMOKE_CAP_ERROR_SHA256
        ),
        "protocol.cap_error_sha256",
    )
    protocol = SmokeProtocolBinding(
        prompt_sha256=_sha256(
            protocol_table["prompt_sha256"], name="protocol.prompt_sha256"
        ),
        cap_error_sha256=cap_error_sha256,
        tool_profile=tool_profile,
        tool_schema_sha256=protocol_table["tool_schema_sha256"],
        enabled_tool_names=enabled_tools,
        maximum_tool_calls=protocol_table["maximum_tool_calls"],
    )
    if mixed_run:
        accepted_prompt_hashes = {
            visual_tool_prompt_identity(
                tool_profile,
                assistant_dialect=assistant_dialect,
            ).bundle_sha256
        }
        if crop_tgvf_matched_run:
            accepted_prompt_hashes = {
                CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
            }
        elif rp66_matched_run:
            accepted_prompt_hashes = {
                TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
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
    sampling = PilotSamplingConfig(
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
    sampling_scale = (
        sampling.trajectories_per_prompt,
        sampling.max_response_length,
    )
    if tgvf_backed_matched_run:
        accepted_sampling_scales = {
            (16, 20480),
            POLICY_PILOT_FUNCTIONAL_CANARY_SAMPLING_SCALE,
        }
    elif deepeyes_scaled_crop_run:
        accepted_sampling_scales = {(16, 20480)}
    else:
        accepted_sampling_scales = {(8, 8192)}
    if sampling_scale not in accepted_sampling_scales:
        raise ValueError(
            "sampling DeepEyes-reference scale differs from accepted values "
            f"{sorted(accepted_sampling_scales)!r}"
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

    reward_fields = {
        "task_kind",
        "answer_verifier",
        "answer_verifier_sha256",
        "judge_mode",
        "judge_reason",
    }
    if mixed_run:
        reward_fields.update({"judge_config_path", "judge_config_sha256"})
    if stage3_shaped_run:
        reward_fields.add("profile")
        if tfree_reward_run:
            reward_fields.add("tool_utility_reward_enabled")
        else:
            reward_fields.update(
                {
                    "tool_utility_sidecar_path",
                    "tool_utility_sidecar_sha256",
                    "tool_utility_manifest_path",
                    "tool_utility_manifest_sha256",
                }
            )
        if rp66_shaped_run:
            reward_fields.update(
                {
                    "focus_reward_enabled",
                    "grounding_reward_enabled",
                    "visual_quality_judge_mode",
                }
            )
            raw_reward = payload.get("reward")
            if (
                isinstance(raw_reward, Mapping)
                and raw_reward.get("focus_reward_enabled") is True
            ):
                reward_fields.update(
                    {
                        "visual_quality_judge_config_path",
                        "visual_quality_judge_config_sha256",
                    }
                )
        else:
            reward_fields.update(
                {
                    "visual_quality_judge_config_path",
                    "visual_quality_judge_config_sha256",
                }
            )
    else:
        reward_fields.update(
            {"answer_weight", "format_weight", "conditional_tool_weight"}
        )
    reward_table = _table(
        payload,
        "reward",
        reward_fields,
    )
    expected_task = (
        POLICY_E2E_MIXED_REWARD_TASK if mixed_run else POLICY_E2E_SMOKE_REWARD_TASK
    )
    expected_verifier = (
        POLICY_E2E_MIXED_ANSWER_VERIFIER
        if mixed_run
        else POLICY_E2E_SMOKE_ANSWER_VERIFIER
    )
    expected_judge_mode = (
        POLICY_E2E_MIXED_JUDGE_MODE if mixed_run else POLICY_E2E_SMOKE_JUDGE_MODE
    )
    _require_exact(reward_table["task_kind"], expected_task, "reward.task_kind")
    _require_exact(
        reward_table["answer_verifier"],
        expected_verifier,
        "reward.answer_verifier",
    )
    _require_exact(reward_table["judge_mode"], expected_judge_mode, "reward.judge_mode")
    answer_verifier_sha256 = _sha256(
        reward_table["answer_verifier_sha256"],
        name="reward.answer_verifier_sha256",
    )
    expected_answer_verifier_sha256 = (
        POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256
        if mixed_run
        else POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256
    )
    known_historical_answer_verifier = (
        mixed_run
        and not tgvf_backed_matched_run
        and answer_verifier_sha256 in _LEGACY_POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256S
    )
    if (
        answer_verifier_sha256 != expected_answer_verifier_sha256
        and not known_historical_answer_verifier
    ):
        raise ValueError("reward.answer_verifier_sha256 differs")
    if mixed_run:
        judge_config_path = _existing_file(
            reward_table["judge_config_path"], name="reward.judge_config_path"
        )
        if judge_config_path.is_symlink():
            raise ValueError("reward judge config must not be a symlink")
        judge_config_sha256 = _sha256(
            reward_table["judge_config_sha256"],
            name="reward.judge_config_sha256",
        )
        if _sha256_file(judge_config_path) != judge_config_sha256:
            raise ValueError("reward judge config SHA256 mismatch")
        if tgvf_backed_matched_run:
            from tgvf_rl.rewards.deepeyes_verl_reward import (
                load_deepeyes_judge_service_config,
            )

            load_deepeyes_judge_service_config(
                judge_config_path,
                expected_file_sha256=judge_config_sha256,
            )
        else:
            bound_judge = load_openai_compatible_judge(
                judge_config_path,
                expected_file_sha256=judge_config_sha256,
            )
            if formal_pilot and not bound_judge.formal_pilot_accepted:
                raise ValueError("reward judge is not accepted for the formal Pilot")
    else:
        judge_config_path = None
        judge_config_sha256 = None
    if stage3_shaped_run:
        _require_exact(
            reward_table["profile"],
            STAGE3_SHAPED_REWARD_VERSION,
            "reward.profile",
        )
        if not isinstance(
            runtime_binding,
            (
                PolicyT1MixedRuntimeBinding,
                PolicyTeacherQuarterMixRuntimeBinding,
                PolicyTeacherRatioMixRuntimeBinding,
            ),
        ):
            raise ValueError(
                "Stage3-shaped reward requires a retained mixed T1 dataset"
            )
        if (
            isinstance(
                runtime_binding,
                (
                    PolicyTeacherQuarterMixRuntimeBinding,
                    PolicyTeacherRatioMixRuntimeBinding,
                ),
            )
            and not tfree_reward_run
        ):
            raise ValueError(
                "teacher-quarter data requires a T-free reward; the historical "
                "tool-utility sidecar has no teacher labels"
            )
        expected_shaped_profile = (
            NativeToolCapabilityProfile.CROP_TGVF
            if crop_tgvf_matched_run
            else NativeToolCapabilityProfile.TGVF_ONLY
        )
        if tool_profile is not expected_shaped_profile:
            raise ValueError(
                "Stage3-shaped reward tool profile differs from its run schema"
            )
        if tfree_reward_run:
            tool_utility_reward_enabled = _boolean(
                reward_table["tool_utility_reward_enabled"],
                name="reward.tool_utility_reward_enabled",
            )
            _require_exact(
                tool_utility_reward_enabled,
                False,
                "T-free tool-utility reward switch",
            )
            tool_utility = None
        else:
            tool_utility_reward_enabled = True
            sidecar_path = _existing_file(
                reward_table["tool_utility_sidecar_path"],
                name="reward.tool_utility_sidecar_path",
            )
            sidecar_sha256 = _sha256(
                reward_table["tool_utility_sidecar_sha256"],
                name="reward.tool_utility_sidecar_sha256",
            )
            sidecar_manifest_path = _existing_file(
                reward_table["tool_utility_manifest_path"],
                name="reward.tool_utility_manifest_path",
            )
            sidecar_manifest_sha256 = _sha256(
                reward_table["tool_utility_manifest_sha256"],
                name="reward.tool_utility_manifest_sha256",
            )
            tool_utility = load_tgvf_tool_utility_runtime_binding(
                sidecar_path,
                expected_sidecar_sha256=sidecar_sha256,
                manifest_path=sidecar_manifest_path,
                expected_manifest_sha256=sidecar_manifest_sha256,
                expected_dataset_iteration_identity_sha256=iteration_sha256,
            )
        if rp66_shaped_run:
            focus_reward_enabled = _boolean(
                reward_table["focus_reward_enabled"],
                name="reward.focus_reward_enabled",
            )
            grounding_reward_enabled = _boolean(
                reward_table["grounding_reward_enabled"],
                name="reward.grounding_reward_enabled",
            )
            if focus_reward_enabled != grounding_reward_enabled:
                raise ValueError(
                    "RP66 shaped Focus/Grounding reward switches must agree"
                )
            if focus_reward_enabled:
                _require_exact(
                    reward_table["visual_quality_judge_mode"],
                    "api",
                    "RP66 shaped visual-quality judge mode",
                )
                visual_quality_config_path = _existing_file(
                    reward_table["visual_quality_judge_config_path"],
                    name="reward.visual_quality_judge_config_path",
                )
                visual_quality_config_sha256 = _sha256(
                    reward_table["visual_quality_judge_config_sha256"],
                    name="reward.visual_quality_judge_config_sha256",
                )
                if (
                    _sha256_file(visual_quality_config_path)
                    != visual_quality_config_sha256
                ):
                    raise ValueError(
                        "reward visual-quality judge config SHA256 mismatch"
                    )
                bound_visual_quality_judge = load_tgvf_visual_quality_judge(
                    visual_quality_config_path,
                    expected_file_sha256=visual_quality_config_sha256,
                )
                if not bound_visual_quality_judge.formal_pilot_accepted:
                    raise ValueError(
                        "visual-quality API judge is not accepted for Policy RL"
                    )
                visual_quality_judge_identity = (
                    bound_visual_quality_judge.config_identity
                )
            else:
                _require_exact(
                    reward_table["visual_quality_judge_mode"],
                    "disabled",
                    "RP66 shaped visual-quality judge mode",
                )
                visual_quality_config_path = None
                visual_quality_config_sha256 = None
                visual_quality_judge_identity = None
        else:
            # Historical Stage3 configs predate component switches.  Keep the
            # sentinel values absent so their serialized runtime/run identity
            # remains byte-for-byte compatible; ``None`` means the legacy
            # visual Focus/Grounding path stays enabled.
            focus_reward_enabled = None
            grounding_reward_enabled = None
            visual_quality_config_path = _existing_file(
                reward_table["visual_quality_judge_config_path"],
                name="reward.visual_quality_judge_config_path",
            )
            visual_quality_config_sha256 = _sha256(
                reward_table["visual_quality_judge_config_sha256"],
                name="reward.visual_quality_judge_config_sha256",
            )
            if (
                _sha256_file(visual_quality_config_path)
                != visual_quality_config_sha256
            ):
                raise ValueError("reward visual-quality judge config SHA256 mismatch")
            bound_visual_quality_judge = load_tgvf_visual_quality_judge(
                visual_quality_config_path,
                expected_file_sha256=visual_quality_config_sha256,
            )
            visual_quality_judge_identity = (
                bound_visual_quality_judge.config_identity
            )
        reward_profile = STAGE3_SHAPED_REWARD_VERSION
        reward_weights: tuple[float, float, float] | None = None
    else:
        reward_weights = (
            _real(reward_table["answer_weight"], name="reward.answer_weight"),
            _real(reward_table["format_weight"], name="reward.format_weight"),
            _real(
                reward_table["conditional_tool_weight"],
                name="reward.conditional_tool_weight",
            ),
        )
        pilot_reward_weight_profile_name(reward_weights)
        reward_profile = "pilot-v1"
        tool_utility = None
        tool_utility_reward_enabled = None
        focus_reward_enabled = None
        grounding_reward_enabled = None
        visual_quality_config_path = None
        visual_quality_config_sha256 = None
        visual_quality_judge_identity = None
    reward = SmokeRewardBinding(
        profile=reward_profile,
        task_kind=reward_table["task_kind"],
        answer_verifier=reward_table["answer_verifier"],
        answer_verifier_sha256=answer_verifier_sha256,
        judge_mode=reward_table["judge_mode"],
        judge_reason=_text(reward_table["judge_reason"], name="reward.judge_reason"),
        answer_weight=None if reward_weights is None else reward_weights[0],
        format_weight=None if reward_weights is None else reward_weights[1],
        conditional_tool_weight=None if reward_weights is None else reward_weights[2],
        judge_config_path=judge_config_path,
        judge_config_sha256=judge_config_sha256,
        tool_utility=tool_utility,
        tool_utility_reward_enabled=tool_utility_reward_enabled,
        focus_reward_enabled=focus_reward_enabled,
        grounding_reward_enabled=grounding_reward_enabled,
        visual_quality_judge_config_path=visual_quality_config_path,
        visual_quality_judge_config_sha256=visual_quality_config_sha256,
        visual_quality_judge_identity=visual_quality_judge_identity,
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
        "vllm_tensor_parallel_size",
        "placement",
        "weight_sync_mode",
        "weight_sync_interval_optimizer_steps",
    }
    lifecycle_control_run = schema_version in {
        POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    }
    if lifecycle_control_run:
        distributed_fields.add("actor_optimizer_offload")
    distributed_table = _table(
        payload,
        "distributed",
        distributed_fields,
    )
    if tgvf_backed_matched_run:
        trainable_rp66_world_size = _positive_int(
            distributed_table["world_size"], name="distributed.world_size"
        )
        if trainable_rp66_world_size not in {4, 8}:
            raise ValueError(
                "trainable RP66 supports only the matched world4 or world8 topology"
            )
        required_world_size = trainable_rp66_world_size
    else:
        required_world_size = 4
    distributed = _distributed(
        distributed_table,
        required_world_size=required_world_size,
        actor_optimizer_offload=(
            _boolean(
                distributed_table["actor_optimizer_offload"],
                name="distributed.actor_optimizer_offload",
            )
            if lifecycle_control_run
            else True
        ),
    )
    expected_global_batch = (
        accumulation.prompt_micro_batch_size_per_rank
        * distributed.world_size
        * accumulation.gradient_accumulation_steps
    )
    if accumulation.global_prompt_batch_size != expected_global_batch:
        raise ValueError(
            "accumulation global prompt batch is inconsistent with world size"
        )

    capacity_table = _table(
        payload,
        "capacity",
        {
            "max_prompt_length",
            "actor_ppo_max_token_len_per_gpu",
            "rollout_log_prob_max_token_len_per_gpu",
            "reference_log_prob_max_token_len_per_gpu",
            "vllm_gpu_memory_utilization",
            "vllm_max_num_batched_tokens",
            "vllm_max_model_len",
            "vllm_max_num_seqs",
            "vllm_enable_chunked_prefill",
            "vllm_enforce_eager",
        },
    )
    capacity = SmokeCapacityBinding(
        max_prompt_length=_positive_int(
            capacity_table["max_prompt_length"],
            name="capacity.max_prompt_length",
        ),
        actor_ppo_max_token_len_per_gpu=_positive_int(
            capacity_table["actor_ppo_max_token_len_per_gpu"],
            name="capacity.actor_ppo_max_token_len_per_gpu",
        ),
        rollout_log_prob_max_token_len_per_gpu=_positive_int(
            capacity_table["rollout_log_prob_max_token_len_per_gpu"],
            name="capacity.rollout_log_prob_max_token_len_per_gpu",
        ),
        reference_log_prob_max_token_len_per_gpu=_positive_int(
            capacity_table["reference_log_prob_max_token_len_per_gpu"],
            name="capacity.reference_log_prob_max_token_len_per_gpu",
        ),
        vllm_gpu_memory_utilization=_unit_interval(
            capacity_table["vllm_gpu_memory_utilization"],
            name="capacity.vllm_gpu_memory_utilization",
        ),
        vllm_max_num_batched_tokens=_positive_int(
            capacity_table["vllm_max_num_batched_tokens"],
            name="capacity.vllm_max_num_batched_tokens",
        ),
        vllm_max_model_len=_positive_int(
            capacity_table["vllm_max_model_len"],
            name="capacity.vllm_max_model_len",
        ),
        vllm_max_num_seqs=_positive_int(
            capacity_table["vllm_max_num_seqs"],
            name="capacity.vllm_max_num_seqs",
        ),
        vllm_enable_chunked_prefill=_boolean(
            capacity_table["vllm_enable_chunked_prefill"],
            name="capacity.vllm_enable_chunked_prefill",
        ),
        vllm_enforce_eager=_boolean(
            capacity_table["vllm_enforce_eager"],
            name="capacity.vllm_enforce_eager",
        ),
    )
    minimum_context = capacity.max_prompt_length + sampling.max_response_length
    if capacity.vllm_max_model_len < minimum_context:
        raise ValueError(
            "capacity.vllm_max_model_len cannot hold max prompt plus response"
        )
    # PRL14/PRL15 use a fixed actor micro-batch and disable dynamic batching;
    # the *_max_token_len_per_gpu fields are therefore inactive capacity
    # metadata. The generic bound below assumes dynamic token batching and
    # would incorrectly reject Crop-16's proven micro32/16384 configuration.
    if not tgvf_backed_matched_run:
        minimum_actor_tokens = (
            accumulation.prompt_micro_batch_size_per_rank
            * sampling.trajectories_per_prompt
            * minimum_context
        )
        if capacity.actor_ppo_max_token_len_per_gpu < minimum_actor_tokens:
            raise ValueError(
                "capacity.actor_ppo_max_token_len_per_gpu is smaller than one "
                "expanded Policy Pilot micro-batch"
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
    expected_agent_loop_config_path = (
        POLICY_E2E_CROP_TGVF_MATCHED_AGENT_LOOP_CONFIG_PATH
        if crop_tgvf_matched_run
        else POLICY_E2E_TRAINABLE_RP66_AGENT_LOOP_CONFIG_PATH
        if rp66_matched_run
        else POLICY_E2E_AGENT_LOOP_CONFIG_PATH
    )
    if (
        not allow_external_agent_loop_config
        and agent_loop_config_path.name != expected_agent_loop_config_path.name
    ):
        raise ValueError(
            "framework.agent_loop_config_path filename differs from the "
            "schema-selected Policy Pilot composition"
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

    training_fields = {
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
    }
    if lifecycle_control_run:
        training_fields.add("permanent_checkpoint_steps")
    training_table = _table(
        payload,
        "training",
        training_fields,
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
        permanent_checkpoint_steps=(
            _strictly_increasing_positive_int_tuple(
                training_table["permanent_checkpoint_steps"],
                name="training.permanent_checkpoint_steps",
            )
            if lifecycle_control_run
            else ()
        ),
    )
    if training.validation_before_training:
        raise ValueError(
            "bounded Policy E2E smoke does not own a validation population"
        )
    if training.checkpoint_steps[-1] > training.maximum_optimizer_steps:
        raise ValueError("training checkpoint step exceeds maximum_optimizer_steps")
    if any(
        step > training.maximum_optimizer_steps
        for step in training.permanent_checkpoint_steps
    ):
        raise ValueError(
            "training permanent checkpoint step exceeds maximum_optimizer_steps"
        )
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

    if crop_tgvf_matched_run:
        policy_type = PolicyCropTGVFMatchedExperimentConfig
    elif rp66_matched_run:
        policy_type = PolicyTrainableRP66ExperimentConfig
    elif stage3_shaped_run:
        policy_type = PolicyTGVFStage3ExperimentConfig
    elif protocol.tool_profile is POLICY_PILOT_V1_TOOL_PROFILE:
        policy_type = PolicyPilotV1Config
    else:
        policy_type = PolicyVisualToolExperimentConfig
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
    )
    if sampling.backend_version != POLICY_PILOT_V1_VLLM_VERSION:
        raise ValueError("sampling backend version differs from Policy Pilot v1")
    if deepeyes_scaled_crop_run:
        if not isinstance(
            runtime_binding,
            (
                PolicyT1MixedRuntimeBinding,
                PolicyTeacherQuarterMixRuntimeBinding,
                PolicyTeacherRatioMixRuntimeBinding,
            ),
        ):
            raise ValueError(
                "DeepEyes-scaled Crop reference requires the retained mixed-T1 dataset"
            )
        _require_exact(
            model.model_name,
            POLICY_PILOT_V1_MODEL_NAME,
            "DeepEyes-scaled model edition",
        )
        _require_exact(
            model_table["image_max_pixels"],
            1_003_520,
            "DeepEyes reference image pixel cap",
        )
        _require_exact(
            protocol.tool_profile,
            NativeToolCapabilityProfile.CROP_ONLY,
            "DeepEyes-scaled tool profile",
        )
        _require_exact(
            (
                reward.answer_weight,
                reward.format_weight,
                reward.conditional_tool_weight,
            ),
            (0.8, 0.2, 1.2),
            "DeepEyes reward weights",
        )
        _require_exact(
            (
                optimizer.name,
                optimizer.learning_rate,
                optimizer.beta1,
                optimizer.beta2,
                optimizer.epsilon,
                optimizer.weight_decay,
                optimizer.maximum_gradient_norm,
            ),
            ("adamw", 1.0e-6, 0.9, 0.999, 1.0e-8, 0.01, 1.0),
            "DeepEyes actor optimizer contract",
        )
        _require_exact(
            (
                accumulation.global_prompt_batch_size,
                accumulation.prompt_micro_batch_size_per_rank,
                accumulation.rollout_prompt_micro_batch_size_per_engine,
                accumulation.gradient_accumulation_steps,
            ),
            (256, 1, 1, 64),
            "four-GPU accumulated DeepEyes batch contract",
        )
        _require_exact(
            (
                distributed.physical_gpu_ids,
                distributed.logical_gpu_ids,
                distributed.world_size,
            ),
            ((0, 1, 2, 3), (0, 1, 2, 3), 4),
            "four-GPU DeepEyes-scaled placement",
        )
        _require_exact(
            (
                capacity.max_prompt_length,
                capacity.actor_ppo_max_token_len_per_gpu,
                capacity.rollout_log_prob_max_token_len_per_gpu,
                capacity.reference_log_prob_max_token_len_per_gpu,
                capacity.vllm_max_num_batched_tokens,
                capacity.vllm_max_model_len,
                capacity.vllm_max_num_seqs,
                capacity.vllm_enable_chunked_prefill,
                capacity.vllm_gpu_memory_utilization,
                capacity.vllm_enforce_eager,
            ),
            (
                8192,
                524288,
                524288,
                524288,
                32768,
                32768,
                32,
                False,
                0.45,
                False,
            ),
            "four-B200 DeepEyes-scaled capacity adaptation",
        )
        _require_exact(
            (
                scheduler.name,
                scheduler.warmup_steps,
                scheduler.total_steps,
                scheduler.minimum_learning_rate_ratio,
                training.maximum_optimizer_steps,
            ),
            ("constant", 0, 80, 0.0, 20),
            "DeepEyes phase-one optimization horizon",
        )
        _require_exact(
            training.checkpoint_steps,
            (0, 1, 2, 4, 8, 20),
            "DeepEyes phase-one checkpoint plan",
        )
        if "wandb" not in training.logger:
            raise ValueError("DeepEyes-scaled Crop reference requires W&B logging")

    if tgvf_backed_matched_run:
        if not isinstance(
            runtime_binding,
            (
                PolicyT1MixedRuntimeBinding,
                PolicyTeacherQuarterMixRuntimeBinding,
                PolicyTeacherRatioMixRuntimeBinding,
            ),
        ):
            raise ValueError(
                "trainable RP66 pilot requires the retained mixed-T1 dataset"
            )
        _require_exact(
            model.model_name,
            POLICY_PILOT_V1_MODEL_NAME,
            "trainable RP66 model edition",
        )
        _require_exact(
            model_table["image_max_pixels"],
            1_003_520,
            "trainable RP66 image pixel cap",
        )
        _require_exact(
            protocol.tool_profile,
            (
                NativeToolCapabilityProfile.CROP_TGVF
                if crop_tgvf_matched_run
                else NativeToolCapabilityProfile.TGVF_ONLY
            ),
            "matched TGVF-backed tool profile",
        )
        if rp66_shaped_run:
            _require_exact(
                (
                    reward.profile,
                    reward.answer_weight,
                    reward.format_weight,
                    reward.conditional_tool_weight,
                    reward.tool_utility_reward_enabled,
                ),
                (
                    STAGE3_SHAPED_REWARD_VERSION,
                    None,
                    None,
                    None,
                    not tfree_reward_run,
                ),
                "RP66 shaped reward controls",
            )
            if (
                type(reward.focus_reward_enabled) is not bool
                or reward.grounding_reward_enabled
                is not reward.focus_reward_enabled
            ):
                raise ValueError(
                    "RP66 shaped visual-quality reward controls differ"
                )
        else:
            _require_exact(
                (
                    reward.answer_weight,
                    reward.format_weight,
                    reward.conditional_tool_weight,
                ),
                (0.8, 0.2, 1.2),
                "trainable RP66 DeepEyes reward weights",
            )
        _require_exact(
            (
                optimizer.name,
                optimizer.learning_rate,
                optimizer.beta1,
                optimizer.beta2,
                optimizer.epsilon,
                optimizer.weight_decay,
                optimizer.maximum_gradient_norm,
            ),
            ("adamw", 1.0e-6, 0.9, 0.999, 1.0e-8, 0.01, 1.0),
            "trainable RP66 actor optimizer contract",
        )
        functional_canary = (
            sampling_scale == POLICY_PILOT_FUNCTIONAL_CANARY_SAMPLING_SCALE
        )
        if functional_canary:
            _require_exact(
                (
                    accumulation.global_prompt_batch_size,
                    accumulation.prompt_micro_batch_size_per_rank,
                    accumulation.rollout_prompt_micro_batch_size_per_engine,
                    accumulation.gradient_accumulation_steps,
                ),
                (4, 1, 1, 1),
                "trainable RP66 functional-canary accumulation contract",
            )
            _require_exact(
                (
                    scheduler.name,
                    scheduler.warmup_steps,
                    scheduler.total_steps,
                    scheduler.minimum_learning_rate_ratio,
                    training.maximum_optimizer_steps,
                ),
                ("constant", 0, 1, 0.0, 1),
                "trainable RP66 functional-canary optimization horizon",
            )
            _require_exact(
                training.checkpoint_steps,
                (0, 1),
                "trainable RP66 functional-canary checkpoint plan",
            )
        else:
            # Batch, accumulation, horizon and checkpoint endpoints are
            # scientific run variables.  Their structural relationships were
            # validated above; do not collapse every formal run back to the
            # historical BS16/eight-step pilot constants.
            _require_exact(
                (
                    scheduler.name,
                    scheduler.warmup_steps,
                    scheduler.minimum_learning_rate_ratio,
                ),
                ("constant", 0, 0.0),
                "trainable RP66 scheduler shape",
            )
            if scheduler.total_steps != training.maximum_optimizer_steps:
                raise ValueError(
                    "trainable RP66 scheduler horizon differs from the configured "
                    "optimizer horizon"
                )
            if training.checkpoint_steps[-1] != training.maximum_optimizer_steps:
                raise ValueError(
                    "trainable RP66 checkpoint plan must include the configured "
                    "final optimizer step"
                )
        if functional_canary and training.logger != ("console",):
            raise ValueError("trainable RP66 functional canary must be console-only")
        if not functional_canary and "wandb" not in training.logger:
            raise ValueError("trainable RP66 pilot requires W&B logging")

    canonical_json = json.dumps(
        _normalize_json(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PolicyE2ESmokeRunConfig(
        run_id=run_id,
        code=code,
        model=model,
        dataset=dataset,
        representation=representation,
        protocol=protocol,
        policy=policy,
        rollout_rng=rollout_rng,
        reward=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
        accumulation=accumulation,
        distributed=distributed,
        capacity=capacity,
        framework=framework,
        training=training,
        output=output,
        source_path=source_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_json=canonical_json,
        formal_pilot=payload["formal_pilot"],
        schema_version=schema_version,
    )


def _verify_deepeyes_artifact(
    root: Path,
    *,
    binding: DeepEyes47KRuntimeBinding,
    samples_sha256: str,
) -> Path:
    manifest_path = root / DEEPEYES47K_MANIFEST_FILE
    samples_path = root / DEEPEYES47K_SAMPLES_FILE
    for path, name in ((manifest_path, "manifest"), (samples_path, "samples")):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"DeepEyes {name} file is missing or unsafe")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != binding.manifest_file_sha256:
        raise ValueError("DeepEyes manifest-file SHA256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("DeepEyes manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) != _DEEPEYES_MANIFEST_FIELDS:
        raise ValueError("DeepEyes manifest fields differ")
    if manifest_bytes != _canonical_json_bytes(manifest) + b"\n":
        raise ValueError("DeepEyes manifest is not canonical JSON")
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    computed_content = hashlib.sha256(_canonical_json_bytes(descriptor)).hexdigest()
    if (
        manifest["content_sha256"] != binding.content_sha256
        or computed_content != binding.content_sha256
    ):
        raise ValueError("DeepEyes manifest content SHA256 mismatch")
    required = {
        "schema_version": DEEPEYES47K_SCHEMA_VERSION,
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": DEEPEYES47K_SNAPSHOT,
        "fixture": False,
        "sample_count": DEEPEYES47K_TOTAL_ROWS,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("DeepEyes formal manifest identity mismatch")
    if manifest.get("shuffle") != {
        "algorithm": DEEPEYES47K_SHUFFLE_ALGORITHM,
        "seed": binding.shuffle_seed,
    }:
        raise ValueError("DeepEyes manifest shuffle identity mismatch")
    if manifest.get("samples") != {
        "path": DEEPEYES47K_SAMPLES_FILE,
        "rows": DEEPEYES47K_TOTAL_ROWS,
        "sha256": samples_sha256,
    }:
        raise ValueError("DeepEyes manifest samples identity mismatch")
    if _sha256_file(samples_path) != samples_sha256:
        raise ValueError("DeepEyes samples file SHA256 mismatch")
    return samples_path


def _verify_deepeyes_files(
    root: Path,
    *,
    binding: DeepEyes47KRuntimeBinding,
    samples_sha256: str,
    sample_id: str,
    cursor: int,
) -> SmokeSelectedMCQSample:
    samples_path = _verify_deepeyes_artifact(
        root,
        binding=binding,
        samples_sha256=samples_sha256,
    )
    selected: object | None = None
    with samples_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == cursor:
                try:
                    selected = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "selected DeepEyes sample is invalid JSON"
                    ) from error
                break
    if not isinstance(selected, Mapping) or selected.get("sample_id") != sample_id:
        raise ValueError("DeepEyes sample_id differs from the selected cursor")
    return _verify_selected_mcq_sample(selected, root=root)


def _verify_selected_mcq_sample(
    selected: Mapping[str, object], *, root: Path
) -> SmokeSelectedMCQSample:
    """Bind the smoke's exact-match reward route to a genuine MCQ row."""

    if selected.get("task_kind") != "mcq":
        raise ValueError("selected DeepEyes row must have task_kind='mcq'")
    extra_info = selected.get("extra_info")
    reward_model = selected.get("reward_model")
    if not isinstance(extra_info, Mapping) or set(extra_info) != {"question"}:
        raise ValueError("selected DeepEyes MCQ extra_info schema differs")
    if not isinstance(reward_model, Mapping) or set(reward_model) != {"ground_truth"}:
        raise ValueError("selected DeepEyes MCQ reward_model schema differs")
    question = extra_info.get("question")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("selected DeepEyes MCQ question must be non-empty")
    if (
        not isinstance(ground_truth, str)
        or _MCQ_LETTER_ANSWER.fullmatch(ground_truth.strip()) is None
    ):
        raise ValueError("selected DeepEyes MCQ ground truth must be a choice letter")
    labels = {
        next(label for label in match.groups() if label is not None).upper()
        for match in _MCQ_OPTION_PATTERN.finditer(question)
    }
    if len(labels) < 2:
        raise ValueError("selected DeepEyes MCQ question must contain choices")
    image = selected.get("image")
    if not isinstance(image, Mapping) or set(image) != {"path", "sha256"}:
        raise ValueError("selected DeepEyes MCQ image schema differs")
    relative_image = image.get("path")
    if not isinstance(relative_image, str) or not relative_image:
        raise ValueError("selected DeepEyes MCQ image path must be non-empty")
    lexical = Path(relative_image)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise ValueError("selected DeepEyes MCQ image path is unsafe")
    image_path = (root / lexical).resolve(strict=False)
    images_root = (root / "images").resolve(strict=False)
    _require_within(image_path, images_root, name="selected DeepEyes MCQ image")
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError("selected DeepEyes MCQ image must be a regular file")
    image_sha256 = _sha256(image.get("sha256"), name="selected image sha256")
    if _sha256_file(image_path) != image_sha256:
        raise ValueError("selected DeepEyes MCQ image SHA256 mismatch")
    data_source = selected.get("data_source")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("selected DeepEyes MCQ data_source must be non-empty")
    return SmokeSelectedMCQSample(
        sample_id=str(selected["sample_id"]),
        image_path=image_path,
        image_sha256=image_sha256,
        question=question,
        ground_truth=ground_truth,
        data_source=data_source,
    )


def _conditioning(value: object) -> TargetConditioningConfig:
    if not isinstance(value, Mapping) or "provider" not in value:
        raise ValueError("representation.conditioning must bind a provider")
    try:
        provider = TargetConditioningProviderKind(value["provider"])
    except (TypeError, ValueError) as error:
        raise ValueError("representation conditioning provider is invalid") from error
    if provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        if set(value) != {"provider", "hidden_layer"}:
            raise ValueError("contextual conditioning fields differ")
        return TargetConditioningConfig(
            provider=provider,
            hidden_layer=_integer(
                value["hidden_layer"], name="representation.conditioning.hidden_layer"
            ),
        )
    if set(value) != {"provider", "embedding_identity"}:
        raise ValueError("target-token embedding conditioning fields differ")
    return TargetConditioningConfig(
        provider=provider,
        embedding_identity=_text(
            value["embedding_identity"],
            name="representation.conditioning.embedding_identity",
        ),
    )


def _distributed(
    table: Mapping[str, object],
    *,
    required_world_size: int = 4,
    actor_optimizer_offload: bool = True,
) -> SmokeDistributedBinding:
    physical = _nonnegative_int_tuple(
        table["physical_gpu_ids"], name="distributed.physical_gpu_ids"
    )
    logical = _nonnegative_int_tuple(
        table["logical_gpu_ids"], name="distributed.logical_gpu_ids"
    )
    actor = _nonnegative_int_tuple(
        table["actor_logical_gpu_ids"], name="distributed.actor_logical_gpu_ids"
    )
    rollout = _nonnegative_int_tuple(
        table["rollout_logical_gpu_ids"], name="distributed.rollout_logical_gpu_ids"
    )
    for name, values in (
        ("physical_gpu_ids", physical),
        ("logical_gpu_ids", logical),
        ("actor_logical_gpu_ids", actor),
        ("rollout_logical_gpu_ids", rollout),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"distributed.{name} must be non-empty and unique")
    world_size = _positive_int(table["world_size"], name="distributed.world_size")
    if logical != tuple(range(len(logical))) or len(physical) != len(logical):
        raise ValueError("distributed physical/logical GPU mapping is invalid")
    expected_logical = tuple(range(required_world_size))
    if len(physical) != required_world_size or logical != expected_logical:
        raise ValueError(
            f"this Policy E2E run requires {required_world_size} physical GPUs "
            f"mapped to logical GPUs 0-{required_world_size - 1}"
        )
    if actor != logical or world_size != len(actor):
        raise ValueError(
            "this smoke requires every logical GPU in the FSDP2 actor world"
        )
    if world_size != required_world_size:
        raise ValueError(
            f"this Policy E2E run identity requires world_size={required_world_size}"
        )
    placement = _text(table["placement"], name="distributed.placement")
    if placement != "colocated" or rollout != actor:
        raise ValueError("this smoke requires colocated actor/rollout placement")
    _require_exact(table["fsdp_strategy"], "fsdp2", "distributed.fsdp_strategy")
    _require_exact(table["rollout_backend"], "vllm", "distributed.rollout_backend")
    tp = _positive_int(
        table["vllm_tensor_parallel_size"], name="distributed.vllm_tensor_parallel_size"
    )
    if tp != 1:
        raise ValueError("this Policy E2E run requires vLLM TP=1")
    if len(rollout) % tp != 0:
        raise ValueError("vLLM tensor parallel size must divide rollout GPUs")
    return SmokeDistributedBinding(
        physical_gpu_ids=physical,
        logical_gpu_ids=logical,
        world_size=world_size,
        actor_logical_gpu_ids=actor,
        rollout_logical_gpu_ids=rollout,
        fsdp_strategy=table["fsdp_strategy"],
        fsdp_reshard_after_forward=_boolean(
            table["fsdp_reshard_after_forward"],
            name="distributed.fsdp_reshard_after_forward",
        ),
        rollout_backend=table["rollout_backend"],
        vllm_tensor_parallel_size=tp,
        placement=placement,
        weight_sync_mode=_text(
            table["weight_sync_mode"], name="distributed.weight_sync_mode"
        ),
        weight_sync_interval_optimizer_steps=_positive_int(
            table["weight_sync_interval_optimizer_steps"],
            name="distributed.weight_sync_interval_optimizer_steps",
        ),
        actor_optimizer_offload=actor_optimizer_offload,
    )


def _table(
    payload: Mapping[str, object], name: str, fields: set[str]
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"policy E2E smoke [{name}] fields differ")
    return value


def _safe_run_id(value: object) -> str:
    text = _text(value, name="run_id")
    if _SAFE_RUN_ID.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError("run_id is not a safe path-independent identity")
    return text


def _safe_project_name(value: object) -> str:
    text = _text(value, name="training.project_name")
    if _SAFE_PROJECT_NAME.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError("training.project_name is not a safe logger identity")
    return text


def _fqn(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _FQN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a dotted Python symbol")
    return text


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _positive_int(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value: object, *, name: str) -> float:
    result = _real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: object, *, name: str) -> float:
    result = _real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval(value: object, *, name: str, inclusive: bool = False) -> float:
    result = _real(value, name=name)
    valid = 0.0 <= result <= 1.0 if inclusive else 0.0 < result < 1.0
    if not valid:
        raise ValueError(f"{name} lies outside its unit interval")
    return result


def _exact_real(value: object, expected: float, name: str) -> float:
    result = _real(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must equal {expected!r}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _text_tuple(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name=f"{name}[]") for item in _sequence(value, name=name))


def _nonnegative_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(
        _nonnegative_int(item, name=f"{name}[]") for item in _sequence(value, name=name)
    )


def _strictly_increasing_positive_int_tuple(
    value: object, *, name: str
) -> tuple[int, ...]:
    values = tuple(
        _positive_int(item, name=f"{name}[]") for item in _sequence(value, name=name)
    )
    if not values or any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must increase strictly")
    return values


def _checkpoint_steps(value: object) -> tuple[int, ...]:
    steps = _nonnegative_int_tuple(value, name="training.checkpoint_steps")
    if (
        not steps
        or steps[0] != 0
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("training.checkpoint_steps must increase strictly from zero")
    return steps


def _logprob_measurement(value: object) -> LogProbMeasurement:
    try:
        return LogProbMeasurement(_text(value, name="sampling.logprob_measurement"))
    except ValueError as error:
        raise ValueError("sampling.logprob_measurement is invalid") from error


def _existing_file(value: object, *, name: str) -> Path:
    unresolved = Path(value) if isinstance(value, (str, Path)) else None
    if unresolved is not None and unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = _absolute_path(value, name=name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not identify a file")
    return path


def _existing_directory(value: object, *, name: str) -> Path:
    unresolved = Path(_text(value, name=name))
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = _absolute_path(value, name=name)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not identify a directory")
    return path


def _absolute_path(value: object, *, name: str) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, name=name)
    repository_token = "${TGVF_REPOSITORY_ROOT}"
    if raw == repository_token or raw.startswith(repository_token + "/"):
        suffix = raw.removeprefix(repository_token).lstrip("/")
        raw = str(Path(__file__).resolve().parents[3] / suffix)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute normalized path")
    return path.resolve(strict=False)


def _optional_absolute_path(value: object, *, name: str) -> Path | None:
    if value == "":
        return None
    return _absolute_path(value, name=name)


def _require_within(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain under output.root") from error


def _require_exact(actual: object, expected: object, name: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{name} differs from required value {expected!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalize_json(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_json(item) for item in value]
    return value


__all__ = [
    "POLICY_E2E_AGENT_LOOP_CONFIG_PATH",
    "POLICY_E2E_CROP_TGVF_MATCHED_AGENT_LOOP_CONFIG_PATH",
    "POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA",
    "POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256",
    "POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TRAINABLE_RP66_AGENT_LOOP_CONFIG_PATH",
    "POLICY_E2E_TRAINABLE_RP66_SIX_CALL_CAP_ERROR_SHA256",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_MIXED_JUDGE_MODE",
    "POLICY_E2E_MIXED_REWARD_TASK",
    "POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_RP66_EXACT_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_RP66_TFREE_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_RP66_EXPLICIT_CONTROL_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_RP66_MATCHED_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_TGVF_BACKED_MATCHED_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_SMOKE_ANSWER_VERIFIER",
    "POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_SMOKE_CAP_ERROR_SHA256",
    "POLICY_E2E_SMOKE_CODE_REPOSITORY",
    "POLICY_E2E_SMOKE_CONFIG_SCHEMA",
    "POLICY_E2E_SMOKE_JUDGE_MODE",
    "POLICY_E2E_SMOKE_REWARD_TASK",
    "POLICY_E2E_SMOKE_SEED_DERIVATION_NAME",
    "POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256",
    "POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN",
    "PolicyE2ESmokeRunConfig",
    "RP66AdapterUpdateMode",
    "SmokeAccumulationBinding",
    "SmokeCapacityBinding",
    "SmokeDatasetSelection",
    "SmokeSelectedMCQSample",
    "SmokeDistributedBinding",
    "SmokeFrameworkBinding",
    "SmokeOptimizerBinding",
    "SmokeOutputBinding",
    "SmokePrecisionBinding",
    "SmokeProtocolBinding",
    "SmokeRepresentationBinding",
    "SmokeRewardBinding",
    "SmokeRolloutRNGBinding",
    "SmokeSchedulerBinding",
    "SmokeTrainingBinding",
    "formal_deepeyes47k_iteration_identity_sha256",
    "load_policy_e2e_smoke_run_config",
]
