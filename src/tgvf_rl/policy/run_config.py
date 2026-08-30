"""Strict, read-only configuration for one non-formal policy E2E smoke.

The loader binds launch inputs but deliberately performs no launch work: it
does not import a model, initialize CUDA, create an output directory, or write
any file.  Formal Policy Pilot manifests remain a separate contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import tomllib

from tgvf_rl.artifact_contracts import canonical_json_bytes as _canonical_json_bytes
from tgvf_rl.conditioning import TargetConditioningConfig as TargetConditioningConfig
from tgvf_rl.contracts.identity import ArtifactIdentity, CodeIdentity, ModelIdentity
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
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityRuntimeBinding as TGVFToolUtilityRuntimeBinding,
    load_tgvf_tool_utility_runtime_binding,
)
from tgvf_rl.judges import (
    load_openai_compatible_judge,
    load_tgvf_visual_quality_judge,
)
from tgvf_rl.protocol import (
    NativeActionBoundaryProtocolId,
    NativeAssistantDialect,
    NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
    validate_success_observation_protocol,
    visual_tool_prompt_identity,
)
from tgvf_rl.protocol.native import native_assistant_dialect_for_model
from tgvf_rl.rewards.schema import pilot_reward_weight_profile_name
from tgvf_rl.rewards.stage3_shaped import STAGE3_SHAPED_REWARD_VERSION

from .config import (
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
    PolicyVisualToolExperimentConfig,
)
from .deepeyes_strict_control import (
    DeepEyesStrictControlBinding,
    DeepEyesVisualAnswerVerifierMode,
)
from .run_config_schema import (
    POLICY_E2E_AGENT_LOOP_CONFIG_PATH,
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
    POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE,
    POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA,
    POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
    POLICY_E2E_MIXED_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256,
    POLICY_E2E_MIXED_JUDGE_MODE,
    POLICY_E2E_MIXED_REWARD_TASK,
    POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256,
    POLICY_E2E_SMOKE_CAP_ERROR_SHA256,
    POLICY_E2E_SMOKE_CODE_REPOSITORY,
    POLICY_E2E_SMOKE_CONFIG_SCHEMA,
    POLICY_E2E_SMOKE_JUDGE_MODE,
    POLICY_E2E_SMOKE_REWARD_TASK,
    POLICY_E2E_SMOKE_SEED_DERIVATION_NAME,
    POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256,
    POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256,
    POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    SmokeAccumulationBinding,
    SmokeCapacityBinding,
    SmokeDatasetSelection,
    SmokeDistributedBinding,
    SmokeFrameworkBinding,
    SmokeOptimizerBinding,
    SmokeOutputBinding,
    SmokePrecisionBinding,
    SmokeProtocolBinding,
    SmokeRepresentationBinding,
    SmokeRewardBinding,
    SmokeRolloutRNGBinding,
    SmokeSchedulerBinding,
    SmokeSelectedMCQSample,
    SmokeTrainingBinding,
)
from .run_config_validation import (
    _absolute_path,
    _boolean,
    _checkpoint_steps,
    _conditioning,
    _distributed,
    _exact_real,
    _existing_directory,
    _existing_file,
    _fqn,
    _integer,
    _logprob_measurement,
    _nonnegative_int,
    _nonnegative_int_tuple,
    _nonnegative_real,
    _normalize_json,
    _optional_absolute_path,
    _positive_int,
    _positive_real,
    _real,
    _require_exact,
    _require_within,
    _safe_project_name,
    _safe_run_id,
    _sha256,
    _sha256_file,
    _table,
    _text,
    _text_tuple,
    _unit_interval,
    _validate_deepeyes_strict_judge,
)


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
    allow_historical_reward_contract: bool = False,
    allow_historical_read_only_contract: bool = False,
) -> PolicyE2ESmokeRunConfig:
    """Read and validate a complete smoke TOML without launching anything."""

    if type(allow_external_agent_loop_config) is not bool:
        raise ValueError("allow_external_agent_loop_config must be a bool")
    if type(allow_historical_reward_contract) is not bool:
        raise ValueError("allow_historical_reward_contract must be a bool")
    if type(allow_historical_read_only_contract) is not bool:
        raise ValueError("allow_historical_read_only_contract must be a bool")
    source_path = _existing_file(path, name="config path")
    if source_path.is_symlink():
        raise ValueError("config path must not be a symlink")
    raw = source_path.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        payload = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("policy E2E smoke config is not strict UTF-8 TOML") from error
    if not isinstance(payload, Mapping):
        raise ValueError("policy E2E smoke top-level fields differ")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        POLICY_E2E_SMOKE_CONFIG_SCHEMA,
        POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
        POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA,
        POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA,
    }:
        raise ValueError("policy E2E run config schema mismatch")
    deepeyes_strict_control_run = (
        schema_version == POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA
    )
    expected_top_level_fields = (
        _TOP_LEVEL_FIELDS | {"deepeyes_control"}
        if deepeyes_strict_control_run
        else _TOP_LEVEL_FIELDS
    )
    if set(payload) != expected_top_level_fields:
        raise ValueError("policy E2E smoke top-level fields differ")
    if deepeyes_strict_control_run and not allow_historical_read_only_contract:
        raise ValueError(
            "historical PRL12 strict-control v1 is read-only; explicit historical "
            "contract loading is required"
        )
    formal_pilot = schema_version == POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA
    if payload["formal_pilot"] is not formal_pilot:
        raise ValueError("policy E2E run formal_pilot mode differs from schema")
    mixed_run = schema_version != POLICY_E2E_SMOKE_CONFIG_SCHEMA
    stage3_shaped_run = schema_version == POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA
    deepeyes_scaled_crop_run = schema_version in {
        POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA,
    }
    deepeyes_control = (
        DeepEyesStrictControlBinding.from_mapping(payload["deepeyes_control"])
        if deepeyes_strict_control_run
        else None
    )
    explicit_observation_run = (
        schema_version == POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA
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
    expected_image_max_pixels = 1_003_520 if deepeyes_scaled_crop_run else 512 * 512
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
    if policy_t1_arxivqa_dataset or policy_t1_mixed_dataset:
        if not mixed_run:
            raise ValueError("Policy T1 retained data requires a mixed/formal run")
        dataset_table = _table(
            payload,
            "dataset",
            {
                "kind",
                "root",
                "decision_stage",
                "sample_count",
                "manifest_file_sha256",
                "content_sha256",
                "samples_sha256",
                "iteration_identity_sha256",
                "shuffle_seed",
            },
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
        if policy_t1_mixed_dataset:
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
    expected_maximum_tool_calls = 1 if stage3_shaped_run else 4
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
            POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256
            if stage3_shaped_run
            else POLICY_E2E_SMOKE_CAP_ERROR_SHA256
        ),
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
    protocol = SmokeProtocolBinding(
        prompt_sha256=_sha256(
            protocol_table["prompt_sha256"], name="protocol.prompt_sha256"
        ),
        cap_error_sha256=cap_error_sha256,
        tool_profile=tool_profile,
        tool_schema_sha256=protocol_table["tool_schema_sha256"],
        enabled_tool_names=enabled_tools,
        maximum_tool_calls=protocol_table["maximum_tool_calls"],
        success_observation_protocol_id=success_observation_protocol_id,
        action_boundary_protocol_id=action_boundary_protocol_id,
    )
    if deepeyes_control is not None:
        _require_exact(
            protocol.prompt_sha256,
            deepeyes_control.prompt_bundle_sha256(assistant_dialect),
            "protocol.prompt_sha256",
        )
    elif mixed_run:
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
        reward_fields.update(
            {
                "profile",
                "tool_utility_sidecar_path",
                "tool_utility_sidecar_sha256",
                "tool_utility_manifest_path",
                "tool_utility_manifest_sha256",
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
    visual_always_judge = (
        deepeyes_control is not None
        and deepeyes_control.visual_answer_verifier
        is DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B
    )
    if visual_always_judge:
        expected_verifier = POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER
        expected_judge_mode = POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_MODE
    else:
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
    if visual_always_judge:
        current_answer_verifier_sha256 = (
            POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_ANSWER_VERIFIER_SHA256
        )
    else:
        current_answer_verifier_sha256 = (
            POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256
            if mixed_run
            else POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256
        )
    historical_answer_verifier_sha256 = (
        POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256
        if mixed_run
        else POLICY_E2E_SMOKE_ANSWER_VERIFIER_V2_SHA256
    )
    accepted_answer_verifier_sha256s = {current_answer_verifier_sha256}
    if allow_historical_reward_contract and deepeyes_control is None:
        accepted_answer_verifier_sha256s.add(historical_answer_verifier_sha256)
    if answer_verifier_sha256 not in accepted_answer_verifier_sha256s:
        raise ValueError(
            "reward.answer_verifier_sha256 differs from the current contract"
            + (
                " and the named historical evaluation contract"
                if allow_historical_reward_contract
                else ""
            )
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
        if deepeyes_control is not None:
            _validate_deepeyes_strict_judge(
                judge_config_path,
                judge_config_sha256=judge_config_sha256,
                visual_always=visual_always_judge,
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
        if not isinstance(runtime_binding, PolicyT1MixedRuntimeBinding):
            raise ValueError("Stage3-shaped reward requires the mixed-v2 T1 dataset")
        if tool_profile is not NativeToolCapabilityProfile.TGVF_ONLY:
            raise ValueError("Stage3-shaped reward requires the TGVF-only tool profile")
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
        visual_quality_config_path = _existing_file(
            reward_table["visual_quality_judge_config_path"],
            name="reward.visual_quality_judge_config_path",
        )
        visual_quality_config_sha256 = _sha256(
            reward_table["visual_quality_judge_config_sha256"],
            name="reward.visual_quality_judge_config_sha256",
        )
        if _sha256_file(visual_quality_config_path) != visual_quality_config_sha256:
            raise ValueError("reward visual-quality judge config SHA256 mismatch")
        bound_visual_quality_judge = load_tgvf_visual_quality_judge(
            visual_quality_config_path,
            expected_file_sha256=visual_quality_config_sha256,
        )
        visual_quality_judge_identity = bound_visual_quality_judge.config_identity
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

    if stage3_shaped_run:
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
        if not isinstance(runtime_binding, PolicyT1MixedRuntimeBinding):
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
        deepeyes_control=deepeyes_control,
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


__all__ = [
    "POLICY_E2E_AGENT_LOOP_CONFIG_PATH",
    "POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA",
    "POLICY_E2E_STAGE3_ONE_CALL_CAP_ERROR_SHA256",
    "POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256",
    "POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256",
    "POLICY_E2E_MIXED_JUDGE_MODE",
    "POLICY_E2E_MIXED_REWARD_TASK",
    "POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA",
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
