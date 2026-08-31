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
    POLICY_TEACHER_QUARTER_MIX_DATASET_KIND,
    POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT,
    POLICY_TEACHER_QUARTER_MIX_SEED,
    PolicyT1DecisionStage,
    PolicyT1MixedRuntimeBinding,
    PolicyT1RLRuntimeBinding,
    PolicyTeacherQuarterMixRuntimeBinding,
    policy_t1_mixed_iteration_identity_sha256,
    policy_t1_rl_iteration_identity_sha256,
    policy_teacher_quarter_mix_iteration_identity_sha256,
    verify_policy_t1_mixed_artifact_binding,
    verify_policy_t1_rl_artifact_binding,
    verify_policy_teacher_quarter_mix_artifact_binding,
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
    NativeActionBoundaryProtocolId as NativeActionBoundaryProtocolId,
    NativeSuccessObservationProtocolId as NativeSuccessObservationProtocolId,
    NativeToolCapabilityProfile,
)
from tgvf_rl.protocol.native import native_assistant_dialect_for_model
from tgvf_rl.rewards.schema import pilot_reward_weight_profile_name
from tgvf_rl.rewards.stage3_shaped import STAGE3_SHAPED_REWARD_VERSION

from .config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_HISTORICAL_THINKING_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_NAME,
    POLICY_PILOT_V1_HISTORICAL_THINKING_MODEL_PATH,
    POLICY_PILOT_V1_MODEL_FAMILY,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_PATH,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
    PolicyMethodExperimentConfig as PolicyMethodExperimentConfig,
    PolicyMethodProfile as PolicyMethodProfile,
    PolicyPilotV1Config as PolicyPilotV1Config,
    PolicyTGVFStage3ExperimentConfig as PolicyTGVFStage3ExperimentConfig,
    PolicyVisualToolExperimentConfig as PolicyVisualToolExperimentConfig,
)
from .deepeyes_strict_control import (
    DeepEyesStrictControlBinding,
    DeepEyesVisualAnswerVerifierMode,
)
from .run_config_canonical_launch import bind_canonical_policy_launch
from .run_config_schema import (
    POLICY_E2E_AGENT_LOOP_CONFIG_PATH,
    POLICY_E2E_ATOMIC_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
    POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA,
    POLICY_E2E_EXPLICIT_OBSERVATION_RUN_CONFIG_SCHEMA,
    POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
    POLICY_E2E_MIXED_ANSWER_VERIFIER,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_V1_SHA256,
    POLICY_E2E_MIXED_JUDGE_MODE,
    POLICY_E2E_MIXED_REWARD_TASK,
    POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
    POLICY_E2E_METHOD_RUN_CONFIG_SCHEMAS,
    POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS,
    POLICY_E2E_PIXEL512_SIX_CALL_CAP_ERROR_SHA256,
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
    POLICY_E2E_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS,
    POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    PolicyMethodMatrixBinding,
    RP66AdapterUpdateMode,
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
    pixel512_parity_method_for_schema,
)
from .run_config_validation import (
    _absolute_path as _absolute_path,
    _boolean as _boolean,
    _checkpoint_steps as _checkpoint_steps,
    _conditioning as _conditioning,
    _distributed as _distributed,
    _exact_real as _exact_real,
    _existing_directory as _existing_directory,
    _existing_file as _existing_file,
    _fqn as _fqn,
    _integer as _integer,
    _logprob_measurement as _logprob_measurement,
    _nonnegative_int as _nonnegative_int,
    _nonnegative_int_tuple as _nonnegative_int_tuple,
    _nonnegative_real as _nonnegative_real,
    _normalize_json as _normalize_json,
    _optional_absolute_path as _optional_absolute_path,
    _positive_int as _positive_int,
    _positive_real as _positive_real,
    _real as _real,
    _require_exact as _require_exact,
    _require_within as _require_within,
    _safe_project_name as _safe_project_name,
    _safe_run_id as _safe_run_id,
    _sha256 as _sha256,
    _sha256_file as _sha256_file,
    _table as _table,
    _text as _text,
    _text_tuple as _text_tuple,
    _unit_interval as _unit_interval,
    _validate_deepeyes_strict_judge as _validate_deepeyes_strict_judge,
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
        POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA,
        *POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS,
    }:
        raise ValueError("policy E2E run config schema mismatch")
    deepeyes_strict_control_run = (
        schema_version == POLICY_E2E_DEEPEYES_STRICT_CONTROL_RUN_CONFIG_SCHEMA
    )
    if deepeyes_strict_control_run:
        expected_top_level_fields = _TOP_LEVEL_FIELDS | {"deepeyes_control"}
    elif schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA:
        expected_top_level_fields = _TOP_LEVEL_FIELDS | {"method"}
    else:
        expected_top_level_fields = _TOP_LEVEL_FIELDS
    if set(payload) != expected_top_level_fields:
        raise ValueError("policy E2E smoke top-level fields differ")
    if deepeyes_strict_control_run and not allow_historical_read_only_contract:
        raise ValueError(
            "historical PRL12 strict-control v1 is read-only; explicit historical "
            "contract loading is required"
        )
    legacy_method_profile = pixel512_parity_method_for_schema(schema_version)
    if legacy_method_profile is not None and not allow_historical_read_only_contract:
        raise ValueError(
            "historical resolution-named PRL26 schemas are read-only; use the "
            "method-matrix schema for new runs"
        )
    if schema_version == POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA:
        method_table = _table(payload, "method", {"matrix_id", "profile"})
        try:
            method_profile = PolicyMethodProfile(method_table["profile"])
        except (TypeError, ValueError) as error:
            raise ValueError("method.profile is invalid") from error
        method_binding = PolicyMethodMatrixBinding(
            matrix_id=_safe_run_id(method_table["matrix_id"]),
            profile=method_profile,
        )
    elif legacy_method_profile is not None:
        method_binding = PolicyMethodMatrixBinding(
            matrix_id="prl26-pixel512-parity-legacy",
            profile=legacy_method_profile,
            legacy_schema_alias=schema_version,
        )
    else:
        method_binding = None
    formal_pilot = schema_version == POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA
    if payload["formal_pilot"] is not formal_pilot:
        raise ValueError("policy E2E run formal_pilot mode differs from schema")
    mixed_run = schema_version != POLICY_E2E_SMOKE_CONFIG_SCHEMA
    method_run = method_binding is not None
    stage3_shaped_run = (
        schema_version == POLICY_E2E_STAGE3_SHAPED_RUN_CONFIG_SCHEMA or method_run
    )
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
        or method_run
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
    native_deepstack_enabled = _boolean(
        model_table["native_deepstack_enabled"],
        name="model.native_deepstack_enabled",
    )
    if not method_run:
        _require_exact(
            native_deepstack_enabled,
            True,
            "model.native_deepstack_enabled",
        )
    image_max_pixels = _positive_int(
        model_table["image_max_pixels"], name="model.image_max_pixels"
    )
    if not method_run:
        expected_image_max_pixels = 1_003_520 if deepeyes_scaled_crop_run else 512 * 512
        _require_exact(
            image_max_pixels,
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
    policy_teacher_quarter_dataset = (
        isinstance(raw_dataset, Mapping)
        and raw_dataset.get("kind") == POLICY_TEACHER_QUARTER_MIX_DATASET_KIND
    )
    if (
        policy_t1_arxivqa_dataset
        or policy_t1_mixed_dataset
        or policy_teacher_quarter_dataset
    ):
        if not mixed_run:
            raise ValueError(
                "Policy retained training data requires a mixed/formal run"
            )
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
        if policy_teacher_quarter_dataset:
            _require_exact(
                dataset_table["decision_stage"],
                "final",
                "dataset.decision_stage",
            )
            _require_exact(
                expected_sample_count,
                POLICY_TEACHER_QUARTER_MIX_SAMPLE_COUNT,
                "dataset.sample_count",
            )
            _require_exact(
                shuffle_seed,
                POLICY_TEACHER_QUARTER_MIX_SEED,
                "dataset.shuffle_seed",
            )
            runtime_binding = PolicyTeacherQuarterMixRuntimeBinding(
                manifest_file_sha256=manifest_file_sha256,
                content_sha256=content_sha256,
                schedule_seed=shuffle_seed,
                expected_sample_count=expected_sample_count,
            )
            if iteration_sha256 != policy_teacher_quarter_mix_iteration_identity_sha256(
                runtime_binding, samples_sha256=samples_sha256
            ):
                raise ValueError(
                    "dataset iteration identity differs from its Teacher25 binding"
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
    adapter_backed_method_run = (
        method_binding is not None
        and method_binding.profile
        in {
            PolicyMethodProfile.TGVF_SHORT,
            PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
            PolicyMethodProfile.ATOMIC,
        }
    )
    if adapter_backed_method_run:
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
    if adapter_backed_method_run:
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

    visual_always_judge = (
        deepeyes_control is not None
        and deepeyes_control.visual_answer_verifier
        is DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B
    )
    canonical_launch = bind_canonical_policy_launch(
        payload,
        allow_external_agent_loop_config=allow_external_agent_loop_config,
        allow_historical_reward_contract=allow_historical_reward_contract,
        assistant_dialect=assistant_dialect,
        deepeyes_control=deepeyes_control,
        deepeyes_scaled_crop_run=deepeyes_scaled_crop_run,
        explicit_observation_run=explicit_observation_run,
        formal_pilot=formal_pilot,
        iteration_sha256=iteration_sha256,
        mixed_run=mixed_run,
        model=model,
        model_table=model_table,
        method_binding=method_binding,
        runtime_binding=runtime_binding,
        stage3_shaped_reward_version=STAGE3_SHAPED_REWARD_VERSION,
        stage3_shaped_run=stage3_shaped_run,
        visual_always_judge=visual_always_judge,
        pilot_reward_weight_profile_name=pilot_reward_weight_profile_name,
        load_openai_compatible_judge=load_openai_compatible_judge,
        load_tgvf_tool_utility_runtime_binding=(load_tgvf_tool_utility_runtime_binding),
        load_tgvf_visual_quality_judge=load_tgvf_visual_quality_judge,
    )
    protocol = canonical_launch.protocol
    rollout_rng = canonical_launch.rollout_rng
    reward = canonical_launch.reward
    optimizer = canonical_launch.optimizer
    scheduler = canonical_launch.scheduler
    precision = canonical_launch.precision
    accumulation = canonical_launch.accumulation
    distributed = canonical_launch.distributed
    capacity = canonical_launch.capacity
    framework = canonical_launch.framework
    training = canonical_launch.training
    output = canonical_launch.output
    policy = canonical_launch.policy

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
        method=method_binding,
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
    "POLICY_E2E_ATOMIC_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_CROP_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
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
    "POLICY_E2E_METHOD_MATRIX_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_METHOD_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_NO_TOOL_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS",
    "POLICY_E2E_PIXEL512_SIX_CALL_CAP_ERROR_SHA256",
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
    "POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA",
    "POLICY_E2E_TGVF_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMAS",
    "PolicyE2ESmokeRunConfig",
    "PolicyMethodMatrixBinding",
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
    "pixel512_parity_method_for_schema",
]
