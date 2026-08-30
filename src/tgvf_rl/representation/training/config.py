"""Strict TOML identity for executable Qwen3 representation training.

The public facade preserves the historical configuration API while immutable
schemas, TOML table parsing, and external-file binding live in one-way leaves.
Loading remains read-only: it never initializes CUDA, creates an output
directory, loads model weights, or starts training.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tomllib

from tgvf_rl.conditioning import (
    TargetConditioningConfig as TargetConditioningConfig,
    TargetConditioningProviderKind as TargetConditioningProviderKind,
)
from tgvf_rl.contracts.identity import (
    CodeIdentity as CodeIdentity,
    ModelIdentity as ModelIdentity,
)
from tgvf_rl.representation.adapter import TGVFAdapterVariant

from .checkpoint import (
    L_GEN_GLOBAL_REDUCTION as L_GEN_GLOBAL_REDUCTION,
    MATRIX_CE_GLOBAL_REDUCTION as MATRIX_CE_GLOBAL_REDUCTION,
    RepresentationAccumulationIdentity as RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2 as RepresentationAccumulationIdentityV2,
)
from .config_binding import _verify_external_files as _verify_external_files
from .config_parser import (
    _parse_adapter as _parse_adapter,
    _parse_checkpoint as _parse_checkpoint,
    _parse_code as _parse_code,
    _parse_conditioning as _parse_conditioning,
    _parse_data as _parse_data,
    _parse_data_split as _parse_data_split,
    _parse_execution as _parse_execution,
    _parse_fsdp2 as _parse_fsdp2,
    _parse_initialization as _parse_initialization,
    _parse_model as _parse_model,
    _parse_objective as _parse_objective,
    _parse_optimizer as _parse_optimizer,
    _parse_output as _parse_output,
    _parse_post_training_internal_evaluation as _parse_post_training_internal_evaluation,
    _parse_prompt as _parse_prompt,
    _parse_resume as _parse_resume,
    _parse_scheduler as _parse_scheduler,
    _parse_training as _parse_training,
)
from .config_schema import (
    ACCEPTED_QWEN3_ATTENTION_BACKEND,
    ACCEPTED_QWEN3_MODEL_DTYPE,
    ACCEPTED_QWEN3_MODEL_NAME,
    NO_INITIALIZATION_SOURCE,
    NO_RESUME_CHECKPOINT,
    NO_RESUME_CODE_COMPATIBILITY as NO_RESUME_CODE_COMPATIBILITY,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    REPRESENTATION_TRAINING_SCOPE,
    VALIDATED_NON_TRAINING_CODE_TRANSITION as VALIDATED_NON_TRAINING_CODE_TRANSITION,
    RepresentationAdamWConfig,
    RepresentationCheckpointConfig,
    RepresentationCodeConfig,
    RepresentationDataConfig,
    RepresentationDataConfigV2,
    RepresentationDataSplitConfig,
    RepresentationExecutionConfig,
    RepresentationFSDP2TopologyConfig,
    RepresentationInitializationConfig,
    RepresentationModelConfig,
    RepresentationObjectiveExecutionConfig,
    RepresentationObjectiveExecutionConfigV2,
    RepresentationObjectiveExecutionConfigV3,
    RepresentationOutputConfig,
    RepresentationPostTrainingInternalEvaluationConfig,
    RepresentationResumeConfig,
    RepresentationTrainingLoopConfig,
)
from .config_run_schema import RepresentationTrainingConfig
from .config_values import (
    _absolute_path as _absolute_path,
    _bool as _bool,
    _boolean as _boolean,
    _canonical_mapping_sha256 as _canonical_mapping_sha256,
    _configuration_source_path as _configuration_source_path,
    _exact_fields as _exact_fields,
    _existing_file_path as _existing_file_path,
    _float as _float,
    _float_tuple as _float_tuple,
    _int as _int,
    _int_tuple as _int_tuple,
    _integer as _integer,
    _nearest_existing_parent as _nearest_existing_parent,
    _non_empty_text as _non_empty_text,
    _path as _path,
    _positive_int as _positive_int,
    _safe_filename as _safe_filename,
    _sha256 as _sha256,
    _string as _string,
    _table as _table,
)
from .data import SplitOverlapPolicy as SplitOverlapPolicy
from .distributed_checkpoint import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION as DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2 as DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
)
from .fsdp2 import RepresentationFSDP2Config as RepresentationFSDP2Config
from .losses import MatrixCEScoreMode as MatrixCEScoreMode
from .native_pipeline import (
    REPRESENTATION_PROMPT_IDENTITY as REPRESENTATION_PROMPT_IDENTITY,
    REPRESENTATION_PROMPT_SCHEMA_VERSION as REPRESENTATION_PROMPT_SCHEMA_VERSION,
    RepresentationPromptConfig as RepresentationPromptConfig,
)
from .objective import (
    RepresentationObjectiveConfig as RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2 as RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3 as RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind as RepresentationObjectiveKind,
    resolve_matrix_ce_score_config as resolve_matrix_ce_score_config,
)
from .runtime import (
    ACCEPTED_QWEN3_MODEL_FIXTURES as ACCEPTED_QWEN3_MODEL_FIXTURES,
    qwen3_input_embedding_identity as qwen3_input_embedding_identity,
)
from .trainer import (
    RepresentationOptimizerConfig as RepresentationOptimizerConfig,
    RepresentationPrecision as RepresentationPrecision,
    RepresentationSchedulerConfig as RepresentationSchedulerConfig,
    RepresentationSchedulerKind as RepresentationSchedulerKind,
    RepresentationTrainerConfig as RepresentationTrainerConfig,
)


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "run_id",
        "code",
        "model",
        "conditioning",
        "data",
        "prompt",
        "objective",
        "optimizer",
        "scheduler",
        "execution",
        "initialization",
        "fsdp2",
        "training",
        "output",
        "resume",
        "checkpoint",
    }
)
_POST_TRAINING_INTERNAL_EVALUATION_FIELD = "post_training_internal_evaluation"
_ADAPTER_FIELD = "adapter"


def load_representation_training_config(
    path: str | Path,
    *,
    verify_external_files: bool = True,
    allow_existing_post_training_report: bool = False,
) -> RepresentationTrainingConfig:
    """Parse and validate one complete representation-training TOML identity.

    ``verify_external_files=False`` is reserved for schema/unit fixtures.  The
    CLI and production runner use the default, which verifies the local model
    directory, both exact JSONL byte hashes, output parents, and any requested
    resume checkpoint without loading model weights or touching CUDA.
    """

    if not isinstance(verify_external_files, bool):
        raise TypeError("verify_external_files must be a bool")
    if not isinstance(allow_existing_post_training_report, bool):
        raise TypeError("allow_existing_post_training_report must be a bool")
    if allow_existing_post_training_report and not verify_external_files:
        raise ValueError(
            "allow_existing_post_training_report requires external-file verification"
        )
    source_path = _configuration_source_path(path)
    raw = source_path.read_bytes()
    source_toml_sha256 = sha256(raw).hexdigest()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid representation training TOML: {error}") from error
    if not isinstance(value, dict):  # tomllib currently always returns dict
        raise TypeError("representation training TOML root must be a table")
    schema_version = _string(value, "schema_version", table="root")
    root_fields = _TOP_LEVEL_FIELDS
    if schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    }:
        root_fields = root_fields | {_POST_TRAINING_INTERNAL_EVALUATION_FIELD}
    if schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5:
        root_fields = root_fields | {_ADAPTER_FIELD}
    _exact_fields(value, root_fields, table="root")
    canonical_config_sha256 = _canonical_mapping_sha256(value)

    scope = _string(value, "scope", table="root")
    run_id = _string(value, "run_id", table="root")
    code = _parse_code(_table(value, "code", table="root"))
    model = _parse_model(_table(value, "model", table="root"))
    adapter_variant = (
        _parse_adapter(_table(value, _ADAPTER_FIELD, table="root"))
        if schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5
        else TGVFAdapterVariant.FULL_D_DEEPSTACK
    )
    provider = _parse_conditioning(
        _table(value, "conditioning", table="root"), model.identity
    )
    data = _parse_data(
        _table(value, "data", table="root"),
        schema_version=schema_version,
    )
    prompt = _parse_prompt(
        _table(value, "prompt", table="root"),
        config_schema_version=schema_version,
    )
    objective = _parse_objective(
        _table(value, "objective", table="root"),
        schema_version=schema_version,
    )
    optimizer = _parse_optimizer(_table(value, "optimizer", table="root"))
    scheduler = _parse_scheduler(
        _table(value, "scheduler", table="root"),
        schema_version=schema_version,
    )
    execution = _parse_execution(_table(value, "execution", table="root"))
    initialization = _parse_initialization(
        _table(value, "initialization", table="root")
    )
    fsdp2 = _parse_fsdp2(_table(value, "fsdp2", table="root"))
    training = _parse_training(_table(value, "training", table="root"))
    output = _parse_output(_table(value, "output", table="root"))
    resume = _parse_resume(_table(value, "resume", table="root"))
    checkpoint = _parse_checkpoint(_table(value, "checkpoint", table="root"))
    post_training_internal_evaluation = (
        _parse_post_training_internal_evaluation(
            _table(value, _POST_TRAINING_INTERNAL_EVALUATION_FIELD, table="root")
        )
        if schema_version
        in {
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
        }
        else None
    )

    config = RepresentationTrainingConfig(
        schema_version=schema_version,
        scope=scope,
        run_id=run_id,
        code=code,
        model=model,
        adapter_variant=adapter_variant,
        provider=provider,
        data=data,
        prompt=prompt,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        execution=execution,
        initialization=initialization,
        fsdp2=fsdp2,
        training=training,
        output=output,
        resume=resume,
        checkpoint=checkpoint,
        post_training_internal_evaluation=post_training_internal_evaluation,
        source_path=source_path,
        source_toml_sha256=source_toml_sha256,
        canonical_config_sha256=canonical_config_sha256,
    )
    if verify_external_files:
        _verify_external_files(
            config,
            allow_existing_post_training_report=(allow_existing_post_training_report),
        )
    return config


__all__ = [
    "ACCEPTED_QWEN3_ATTENTION_BACKEND",
    "ACCEPTED_QWEN3_MODEL_DTYPE",
    "ACCEPTED_QWEN3_MODEL_NAME",
    "NO_INITIALIZATION_SOURCE",
    "NO_RESUME_CHECKPOINT",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5",
    "REPRESENTATION_TRAINING_SCOPE",
    "RepresentationAdamWConfig",
    "RepresentationCheckpointConfig",
    "RepresentationCodeConfig",
    "RepresentationDataConfig",
    "RepresentationDataConfigV2",
    "RepresentationDataSplitConfig",
    "RepresentationExecutionConfig",
    "RepresentationFSDP2TopologyConfig",
    "RepresentationInitializationConfig",
    "RepresentationModelConfig",
    "RepresentationObjectiveExecutionConfig",
    "RepresentationObjectiveExecutionConfigV2",
    "RepresentationObjectiveExecutionConfigV3",
    "RepresentationOutputConfig",
    "RepresentationPostTrainingInternalEvaluationConfig",
    "RepresentationResumeConfig",
    "RepresentationTrainingConfig",
    "RepresentationTrainingLoopConfig",
    "load_representation_training_config",
]
