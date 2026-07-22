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
)
from tgvf_rl.judges import load_openai_compatible_judge
from tgvf_rl.protocol import (
    NativeToolCapabilityProfile,
    StandardToolError,
    ToolErrorCode,
    visual_tool_prompt_identity,
)

from .config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
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
    PolicyVisualToolExperimentConfig,
)


POLICY_E2E_SMOKE_CONFIG_SCHEMA = "policy-e2e-smoke-config-v3"
POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA = "policy-e2e-mixed-run-config-v4"
POLICY_E2E_SMOKE_CODE_REPOSITORY = "Miocio-nora/TGVF-E2E-RL"
POLICY_E2E_SMOKE_JUDGE_MODE = "not_applicable"
POLICY_E2E_SMOKE_REWARD_TASK = "multiple_choice"
POLICY_E2E_SMOKE_ANSWER_VERIFIER = "exact_match"
POLICY_E2E_SMOKE_SEED_DERIVATION_NAME = "content-addressed-vllm-turn-rng-v1"
POLICY_E2E_MIXED_REWARD_TASK = "mixed"
POLICY_E2E_MIXED_ANSWER_VERIFIER = "rule_first_qwen25_72b"
POLICY_E2E_MIXED_JUDGE_MODE = "qwen25_72b_semantic_fallback"


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
POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256 = _fixed_contract_sha256(
    {
        "schema": "policy-e2e-smoke-mcq-verifier-v2",
        "task": POLICY_E2E_SMOKE_REWARD_TASK,
        "route": "multiple_choice_rule",
        "candidate": (
            "canonical-A-through-H-or-last-explicit-answer-option-choice-range-marker"
        ),
        "wrappers": (
            "answer-tag",
            "latex-boxed",
            "markdown-emphasis",
            "qwen-im-end-suffix",
        ),
        "forbidden": "arbitrary-prose-leading-letter",
        "expected": "same-deterministic-parser",
        "fallback_when_unparsed": "strip-casefold-collapse-whitespace-exact",
        "judge": "disabled",
    }
)
POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256 = _fixed_contract_sha256(
    {
        "schema": "policy-e2e-mixed-answer-verifier-v1",
        "routes": {
            "mcq": "deterministic_rule_only",
            "math": "normalized_exact_then_numeric_then_qwen25_72b",
            "open_vqa": "normalized_exact_then_qwen25_72b",
        },
        "judge_failure": "abort_reward_batch",
        "mcq_judge_calls": "forbidden",
    }
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
    root: Path
    runtime_binding: DeepEyes47KRuntimeBinding
    samples_sha256: str
    iteration_identity_sha256: str
    sample_id: str | None
    cursor: int | None
    selected_sample: SmokeSelectedMCQSample | None


@dataclass(frozen=True, slots=True)
class SmokeRepresentationBinding:
    artifact_path: Path
    artifact_file_sha256: str
    artifact: ArtifactIdentity
    expected_run_id: str
    expected_run_identity_sha256: str
    conditioning: TargetConditioningConfig


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
    task_kind: str
    answer_verifier: str
    answer_verifier_sha256: str
    judge_mode: str
    judge_reason: str
    answer_weight: float
    format_weight: float
    conditional_tool_weight: float
    judge_config_path: Path | None = None
    judge_config_sha256: str | None = None


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
    formal_pilot: bool = False
    schema_version: str = POLICY_E2E_SMOKE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        accepted = {
            POLICY_E2E_SMOKE_CONFIG_SCHEMA: False,
            POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA: False,
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
) -> PolicyE2ESmokeRunConfig:
    """Read and validate a complete smoke TOML without launching anything."""

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
    }:
        raise ValueError("policy E2E run config schema mismatch")
    if payload["formal_pilot"] is not False:
        raise ValueError("policy E2E integration config must set formal_pilot=false")
    mixed_run = schema_version == POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA
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
    _require_exact(model_table["name"], POLICY_PILOT_V1_MODEL_NAME, "model.name")
    _require_exact(model_table["path"], POLICY_PILOT_V1_MODEL_PATH, "model.path")
    _require_exact(
        model_table["tokenizer_length"],
        POLICY_PILOT_V1_TOKENIZER_LENGTH,
        "model.tokenizer_length",
    )
    _require_exact(
        model_table["chat_template_sha256"],
        POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
        "model.chat_template_sha256",
    )
    _require_exact(
        model_table["native_deepstack_enabled"], True, "model.native_deepstack_enabled"
    )
    _require_exact(model_table["image_max_pixels"], 512 * 512, "model.image_max_pixels")
    model = ModelIdentity(
        family=model_table["family"],
        model_name=model_table["name"],
        revision_or_path=model_table["path"],
        tokenizer_length=model_table["tokenizer_length"],
        chat_template_sha256=model_table["chat_template_sha256"],
    )

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
    dataset_table = _table(
        payload,
        "dataset",
        dataset_fields,
    )
    _require_exact(
        dataset_table["dataset_id"], DEEPEYES47K_DATASET_ID, "dataset.dataset_id"
    )
    _require_exact(dataset_table["snapshot"], DEEPEYES47K_SNAPSHOT, "dataset.snapshot")
    _require_exact(
        dataset_table["sample_count"], DEEPEYES47K_TOTAL_ROWS, "dataset.sample_count"
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
        raise ValueError("dataset iteration identity differs from its formal binding")
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
        root=dataset_root,
        runtime_binding=runtime_binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=iteration_sha256,
        sample_id=sample_id,
        cursor=cursor,
        selected_sample=selected_sample,
    )

    representation_table = _table(
        payload,
        "representation",
        {
            "artifact_path",
            "artifact_file_sha256",
            "artifact_manifest_sha256",
            "artifact_namespace",
            "artifact_name",
            "artifact_version",
            "expected_run_id",
            "expected_run_identity_sha256",
            "conditioning",
        },
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
    _require_exact(
        protocol_table["maximum_tool_calls"], 4, "protocol.maximum_tool_calls"
    )
    cap_error_sha256 = _sha256(
        protocol_table["cap_error_sha256"], name="protocol.cap_error_sha256"
    )
    _require_exact(
        cap_error_sha256,
        POLICY_E2E_SMOKE_CAP_ERROR_SHA256,
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
        _require_exact(
            protocol.prompt_sha256,
            visual_tool_prompt_identity(tool_profile).bundle_sha256,
            "protocol.prompt_sha256",
        )

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
        "answer_weight",
        "format_weight",
        "conditional_tool_weight",
    }
    if mixed_run:
        reward_fields.update({"judge_config_path", "judge_config_sha256"})
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
    _require_exact(
        reward_table["task_kind"], expected_task, "reward.task_kind"
    )
    _require_exact(
        reward_table["answer_verifier"],
        expected_verifier,
        "reward.answer_verifier",
    )
    _require_exact(
        reward_table["judge_mode"], expected_judge_mode, "reward.judge_mode"
    )
    answer_verifier_sha256 = _sha256(
        reward_table["answer_verifier_sha256"],
        name="reward.answer_verifier_sha256",
    )
    _require_exact(
        answer_verifier_sha256,
        (
            POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256
            if mixed_run
            else POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256
        ),
        "reward.answer_verifier_sha256",
    )
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
        load_openai_compatible_judge(
            judge_config_path,
            expected_file_sha256=judge_config_sha256,
        )
    else:
        judge_config_path = None
        judge_config_sha256 = None
    reward = SmokeRewardBinding(
        task_kind=reward_table["task_kind"],
        answer_verifier=reward_table["answer_verifier"],
        answer_verifier_sha256=answer_verifier_sha256,
        judge_mode=reward_table["judge_mode"],
        judge_reason=_text(reward_table["judge_reason"], name="reward.judge_reason"),
        answer_weight=_exact_real(
            reward_table["answer_weight"], 0.8, "reward.answer_weight"
        ),
        format_weight=_exact_real(
            reward_table["format_weight"], 0.2, "reward.format_weight"
        ),
        conditional_tool_weight=_exact_real(
            reward_table["conditional_tool_weight"],
            1.2,
            "reward.conditional_tool_weight",
        ),
        judge_config_path=judge_config_path,
        judge_config_sha256=judge_config_sha256,
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
    optimizer = SmokeOptimizerBinding(
        name=optimizer_table["name"],
        learning_rate=_exact_real(
            optimizer_table["learning_rate"], 1.0e-5, "optimizer.learning_rate"
        ),
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

    distributed_table = _table(
        payload,
        "distributed",
        {
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
        },
    )
    distributed = _distributed(distributed_table)
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
    if agent_loop_config_path != POLICY_E2E_AGENT_LOOP_CONFIG_PATH:
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
    if scheduler.total_steps != training.maximum_optimizer_steps:
        raise ValueError("scheduler total_steps differs from maximum_optimizer_steps")
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

    policy_type = (
        PolicyPilotV1Config
        if protocol.tool_profile is POLICY_PILOT_V1_TOOL_PROFILE
        else PolicyVisualToolExperimentConfig
    )
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


def _distributed(table: Mapping[str, object]) -> SmokeDistributedBinding:
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
    if physical != (0, 1, 2, 3) or logical != (0, 1, 2, 3):
        raise ValueError(
            "this Policy E2E smoke identity requires physical/logical GPUs 0-3"
        )
    if actor != logical or world_size != len(actor):
        raise ValueError(
            "this smoke requires every logical GPU in the FSDP2 actor world"
        )
    if world_size != 4:
        raise ValueError("this Policy E2E smoke identity requires world_size=4")
    placement = _text(table["placement"], name="distributed.placement")
    if placement != "colocated" or rollout != actor:
        raise ValueError("this smoke requires colocated actor/rollout placement")
    _require_exact(table["fsdp_strategy"], "fsdp2", "distributed.fsdp_strategy")
    _require_exact(table["rollout_backend"], "vllm", "distributed.rollout_backend")
    tp = _positive_int(
        table["vllm_tensor_parallel_size"], name="distributed.vllm_tensor_parallel_size"
    )
    if tp != 1:
        raise ValueError("the initial 4-GPU Policy E2E smoke requires vLLM TP=1")
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
    "POLICY_E2E_MIXED_ANSWER_VERIFIER",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_MIXED_JUDGE_MODE",
    "POLICY_E2E_MIXED_REWARD_TASK",
    "POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA",
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
