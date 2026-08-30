"""Strict TOML table parsers for representation-training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.representation.adapter import TGVFAdapterVariant

from .config_schema import (
    NO_INITIALIZATION_SOURCE,
    NO_RESUME_CHECKPOINT,
    NO_RESUME_CODE_COMPATIBILITY,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
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
from .config_values import (
    _absolute_path,
    _boolean,
    _exact_fields,
    _float,
    _float_tuple,
    _int,
    _int_tuple,
    _path,
    _string,
    _table,
)
from .data import SplitOverlapPolicy
from .losses import MatrixCEScoreMode
from .native_pipeline import (
    REPRESENTATION_PROMPT_SCHEMA_VERSION,
    RepresentationPromptConfig,
)
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
)
from .runtime import qwen3_input_embedding_identity
from .trainer import (
    RepresentationPrecision,
    RepresentationSchedulerConfig,
    RepresentationSchedulerKind,
)


def _parse_code(value: Mapping[str, Any]) -> RepresentationCodeConfig:
    _exact_fields(
        value,
        {"repository", "commit", "dirty", "dirty_state_sha256"},
        table="code",
    )
    dirty = _boolean(value, "dirty", table="code")
    raw_dirty_sha = _string(value, "dirty_state_sha256", table="code")
    dirty_sha = None if raw_dirty_sha == NO_INITIALIZATION_SOURCE else raw_dirty_sha
    return RepresentationCodeConfig(
        repository=_string(value, "repository", table="code"),
        commit=_string(value, "commit", table="code"),
        dirty=dirty,
        dirty_state_sha256=dirty_sha,
    )


def _parse_adapter(value: Mapping[str, Any]) -> TGVFAdapterVariant:
    _exact_fields(value, {"variant"}, table="adapter")
    raw = _string(value, "variant", table="adapter")
    try:
        return TGVFAdapterVariant(raw)
    except ValueError as error:
        raise ValueError(f"adapter.variant is unsupported: {raw!r}") from error


def _parse_model(value: Mapping[str, Any]) -> RepresentationModelConfig:
    optional_fields = {"image_max_pixels"} if "image_max_pixels" in value else set()
    _exact_fields(
        value,
        {
            "family",
            "model_name",
            "local_path",
            "tokenizer_length",
            "chat_template_sha256",
            "dtype",
            "attention_backend",
            "local_files_only",
            "trust_remote_code",
            "tokenizer_resize",
        }
        | optional_fields,
        table="model",
    )
    return RepresentationModelConfig(
        family=_string(value, "family", table="model"),
        model_name=_string(value, "model_name", table="model"),
        local_path=_path(value, "local_path", table="model", allow_empty=False),
        tokenizer_length=_int(value, "tokenizer_length", table="model"),
        chat_template_sha256=_string(value, "chat_template_sha256", table="model"),
        dtype=_string(value, "dtype", table="model"),
        attention_backend=_string(value, "attention_backend", table="model"),
        local_files_only=_boolean(value, "local_files_only", table="model"),
        trust_remote_code=_boolean(value, "trust_remote_code", table="model"),
        tokenizer_resize=_boolean(value, "tokenizer_resize", table="model"),
        image_max_pixels=(
            _int(value, "image_max_pixels", table="model")
            if "image_max_pixels" in value
            else None
        ),
    )


def _parse_conditioning(
    value: Mapping[str, Any], model_identity: ModelIdentity
) -> TargetConditioningConfig:
    provider_raw = _string(value, "provider", table="conditioning")
    try:
        provider = TargetConditioningProviderKind(provider_raw)
    except ValueError as error:
        raise ValueError(
            f"conditioning.provider is unsupported: {provider_raw!r}"
        ) from error
    if provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        _exact_fields(value, {"provider", "hidden_layer"}, table="conditioning")
        return TargetConditioningConfig(
            provider=provider,
            hidden_layer=_int(value, "hidden_layer", table="conditioning"),
        )
    _exact_fields(value, {"provider", "embedding_identity"}, table="conditioning")
    embedding_identity = _string(value, "embedding_identity", table="conditioning")
    expected = qwen3_input_embedding_identity(model_identity)
    if embedding_identity != expected:
        raise ValueError(
            "conditioning.embedding_identity differs from the canonical Qwen3 input "
            "embedding identity"
        )
    return TargetConditioningConfig(
        provider=provider,
        embedding_identity=embedding_identity,
    )


def _parse_data(
    value: Mapping[str, Any],
    *,
    schema_version: str,
) -> RepresentationDataConfig | RepresentationDataConfigV2:
    if schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    }:
        _exact_fields(
            value,
            {
                "warn_on_target_leakage",
                "split_overlap_policy",
                "expected_overlap_report_sha256",
                "train",
                "validation",
            },
            table="data",
        )
        policy_raw = _string(value, "split_overlap_policy", table="data")
        try:
            policy = SplitOverlapPolicy(policy_raw)
        except ValueError as error:
            raise ValueError(
                f"data.split_overlap_policy is unsupported: {policy_raw!r}"
            ) from error
        return RepresentationDataConfigV2(
            train=_parse_data_split(_table(value, "train", table="data"), name="train"),
            validation=_parse_data_split(
                _table(value, "validation", table="data"), name="validation"
            ),
            warn_on_target_leakage=_boolean(
                value, "warn_on_target_leakage", table="data"
            ),
            split_overlap_policy=policy,
            expected_overlap_report_sha256=_string(
                value,
                "expected_overlap_report_sha256",
                table="data",
            ),
        )
    if schema_version != REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("representation training config schema mismatch")
    _exact_fields(
        value,
        {
            "warn_on_target_leakage",
            "require_disjoint_validation",
            "train",
            "validation",
        },
        table="data",
    )
    return RepresentationDataConfig(
        train=_parse_data_split(_table(value, "train", table="data"), name="train"),
        validation=_parse_data_split(
            _table(value, "validation", table="data"), name="validation"
        ),
        warn_on_target_leakage=_boolean(value, "warn_on_target_leakage", table="data"),
        require_disjoint_validation=_boolean(
            value, "require_disjoint_validation", table="data"
        ),
    )


def _parse_data_split(
    value: Mapping[str, Any], *, name: str
) -> RepresentationDataSplitConfig:
    table = f"data.{name}"
    _exact_fields(
        value,
        {"jsonl_path", "source_sha256", "batch_size", "sampler_seed"},
        table=table,
    )
    return RepresentationDataSplitConfig(
        jsonl_path=_path(value, "jsonl_path", table=table, allow_empty=False),
        source_sha256=_string(value, "source_sha256", table=table),
        batch_size=_int(value, "batch_size", table=table),
        sampler_seed=_int(value, "sampler_seed", table=table),
    )


def _parse_prompt(
    value: Mapping[str, Any],
    *,
    config_schema_version: str,
) -> RepresentationPromptConfig:
    if config_schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    }:
        _exact_fields(
            value,
            {"schema_version", "identity", "template", "sha256"},
            table="prompt",
        )
        prompt_schema_version = _string(value, "schema_version", table="prompt")
    elif config_schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
    }:
        _exact_fields(value, {"identity", "template", "sha256"}, table="prompt")
        prompt_schema_version = REPRESENTATION_PROMPT_SCHEMA_VERSION
    else:
        raise ValueError("representation training config schema mismatch")
    return RepresentationPromptConfig(
        identity=_string(value, "identity", table="prompt"),
        template=_string(value, "template", table="prompt"),
        expected_sha256=_string(value, "sha256", table="prompt"),
        schema_version=prompt_schema_version,
    )


def _parse_objective(
    value: Mapping[str, Any],
    *,
    schema_version: str,
) -> (
    RepresentationObjectiveExecutionConfig
    | RepresentationObjectiveExecutionConfigV2
    | RepresentationObjectiveExecutionConfigV3
):
    if schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    }:
        required = {
            "identity",
            "kind",
            "matrix_ce_weight",
            "l_gen_weight",
            "norm_weight",
            "manifold_enabled",
            "manifold_weight",
        }
        optional = {
            field
            for field in ("matrix_ce_mode", "matrix_ce_temperature")
            if field in value
        }
        _exact_fields(value, required | optional, table="objective")
        kind_raw = _string(value, "kind", table="objective")
        try:
            kind = RepresentationObjectiveKind(kind_raw)
        except ValueError as error:
            raise ValueError(f"objective.kind is unsupported: {kind_raw!r}") from error
        mode = MatrixCEScoreMode.BALANCED
        if "matrix_ce_mode" in value:
            mode_raw = _string(value, "matrix_ce_mode", table="objective")
            try:
                mode = MatrixCEScoreMode(mode_raw)
            except ValueError as error:
                raise ValueError(
                    f"objective.matrix_ce_mode is unsupported: {mode_raw!r}"
                ) from error
        return RepresentationObjectiveExecutionConfigV3(
            objective=RepresentationObjectiveConfigV3(
                identity=_string(value, "identity", table="objective"),
                kind=kind,
                matrix_ce_weight=_float(value, "matrix_ce_weight", table="objective"),
                l_gen_weight=_float(value, "l_gen_weight", table="objective"),
                norm_weight=_float(value, "norm_weight", table="objective"),
                matrix_ce_mode=mode,
                matrix_ce_temperature=(
                    _float(value, "matrix_ce_temperature", table="objective")
                    if "matrix_ce_temperature" in value
                    else 1.0
                ),
            ),
            manifold_enabled=_boolean(value, "manifold_enabled", table="objective"),
            manifold_weight=_float(value, "manifold_weight", table="objective"),
        )
    if schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2:
        _exact_fields(
            value,
            {
                "identity",
                "kind",
                "matrix_ce_weight",
                "l_gen_weight",
                "norm_weight",
                "manifold_enabled",
                "manifold_weight",
            },
            table="objective",
        )
        kind_raw = _string(value, "kind", table="objective")
        try:
            kind = RepresentationObjectiveKind(kind_raw)
        except ValueError as error:
            raise ValueError(f"objective.kind is unsupported: {kind_raw!r}") from error
        return RepresentationObjectiveExecutionConfigV2(
            objective=RepresentationObjectiveConfigV2(
                identity=_string(value, "identity", table="objective"),
                kind=kind,
                matrix_ce_weight=_float(value, "matrix_ce_weight", table="objective"),
                l_gen_weight=_float(value, "l_gen_weight", table="objective"),
                norm_weight=_float(value, "norm_weight", table="objective"),
            ),
            manifold_enabled=_boolean(value, "manifold_enabled", table="objective"),
            manifold_weight=_float(value, "manifold_weight", table="objective"),
        )
    if schema_version != REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("representation training config schema mismatch")
    _exact_fields(
        value,
        {
            "identity",
            "kind",
            "matrix_ce_weight",
            "l_gen_weight",
            "manifold_enabled",
            "manifold_weight",
            "norm_loss",
        },
        table="objective",
    )
    kind_raw = _string(value, "kind", table="objective")
    try:
        kind = RepresentationObjectiveKind(kind_raw)
    except ValueError as error:
        raise ValueError(f"objective.kind is unsupported: {kind_raw!r}") from error
    return RepresentationObjectiveExecutionConfig(
        objective=RepresentationObjectiveConfig(
            identity=_string(value, "identity", table="objective"),
            kind=kind,
            matrix_ce_weight=_float(value, "matrix_ce_weight", table="objective"),
            l_gen_weight=_float(value, "l_gen_weight", table="objective"),
        ),
        manifold_enabled=_boolean(value, "manifold_enabled", table="objective"),
        manifold_weight=_float(value, "manifold_weight", table="objective"),
        norm_loss=_string(value, "norm_loss", table="objective"),
    )


def _parse_optimizer(value: Mapping[str, Any]) -> RepresentationAdamWConfig:
    _exact_fields(
        value,
        {
            "type",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
        },
        table="optimizer",
    )
    betas = _float_tuple(value, "betas", table="optimizer", length=2)
    return RepresentationAdamWConfig(
        optimizer_type=_string(value, "type", table="optimizer"),
        learning_rate=_float(value, "learning_rate", table="optimizer"),
        betas=(betas[0], betas[1]),
        eps=_float(value, "eps", table="optimizer"),
        weight_decay=_float(value, "weight_decay", table="optimizer"),
        amsgrad=_boolean(value, "amsgrad", table="optimizer"),
        maximize=_boolean(value, "maximize", table="optimizer"),
        foreach=_boolean(value, "foreach", table="optimizer"),
        capturable=_boolean(value, "capturable", table="optimizer"),
        differentiable=_boolean(value, "differentiable", table="optimizer"),
        fused=_boolean(value, "fused", table="optimizer"),
        decoupled_weight_decay=_boolean(
            value, "decoupled_weight_decay", table="optimizer"
        ),
    )


def _parse_scheduler(
    value: Mapping[str, Any],
    *,
    schema_version: str,
) -> RepresentationSchedulerConfig:
    fields = {"kind", "total_steps", "warmup_steps"}
    if schema_version in {
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
        REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    }:
        fields.add("min_lr_ratio")
    elif schema_version != REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("representation training config schema mismatch")
    _exact_fields(value, fields, table="scheduler")
    kind_raw = _string(value, "kind", table="scheduler")
    try:
        kind = RepresentationSchedulerKind(kind_raw)
    except ValueError as error:
        raise ValueError(f"scheduler.kind is unsupported: {kind_raw!r}") from error
    return RepresentationSchedulerConfig(
        kind=kind,
        total_steps=_int(value, "total_steps", table="scheduler"),
        warmup_steps=_int(value, "warmup_steps", table="scheduler"),
        min_lr_ratio=(
            _float(value, "min_lr_ratio", table="scheduler")
            if schema_version
            in {
                REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
                REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
                REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
                REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
            }
            else None
        ),
    )


def _parse_execution(value: Mapping[str, Any]) -> RepresentationExecutionConfig:
    _exact_fields(
        value,
        {
            "precision",
            "max_grad_norm",
            "require_all_adapter_gradients",
            "gradient_clip_norm_type",
            "gradient_clip_error_if_nonfinite",
        },
        table="execution",
    )
    precision_raw = _string(value, "precision", table="execution")
    try:
        precision = RepresentationPrecision(precision_raw)
    except ValueError as error:
        raise ValueError(
            f"execution.precision is unsupported: {precision_raw!r}"
        ) from error
    return RepresentationExecutionConfig(
        precision=precision,
        max_grad_norm=_float(value, "max_grad_norm", table="execution"),
        require_all_adapter_gradients=_boolean(
            value, "require_all_adapter_gradients", table="execution"
        ),
        gradient_clip_norm_type=_float(
            value, "gradient_clip_norm_type", table="execution"
        ),
        gradient_clip_error_if_nonfinite=_boolean(
            value, "gradient_clip_error_if_nonfinite", table="execution"
        ),
    )


def _parse_initialization(
    value: Mapping[str, Any],
) -> RepresentationInitializationConfig:
    _exact_fields(
        value,
        {
            "kind",
            "seed",
            "source_artifact_sha256",
            "allow_legacy_checkpoint_initialization",
        },
        table="initialization",
    )
    source = _string(value, "source_artifact_sha256", table="initialization")
    if source != NO_INITIALIZATION_SOURCE:
        raise ValueError(
            "initialization.source_artifact_sha256 must be the explicit 'none' sentinel"
        )
    return RepresentationInitializationConfig(
        kind=_string(value, "kind", table="initialization"),
        seed=_int(value, "seed", table="initialization"),
        source_artifact_sha256=None,
        allow_legacy_checkpoint_initialization=_boolean(
            value,
            "allow_legacy_checkpoint_initialization",
            table="initialization",
        ),
    )


def _parse_fsdp2(value: Mapping[str, Any]) -> RepresentationFSDP2TopologyConfig:
    _exact_fields(
        value,
        {
            "strategy",
            "world_size",
            "physical_gpu_ids",
            "logical_gpu_ids",
            "device_type",
            "mesh_dim_name",
            "mesh_shape",
            "reshard_after_forward",
            "parameter_dtype",
            "reduce_dtype",
            "output_dtype",
            "cast_forward_inputs",
            "offload_policy",
        },
        table="fsdp2",
    )
    return RepresentationFSDP2TopologyConfig(
        strategy=_string(value, "strategy", table="fsdp2"),
        world_size=_int(value, "world_size", table="fsdp2"),
        physical_gpu_ids=_int_tuple(value, "physical_gpu_ids", table="fsdp2"),
        logical_gpu_ids=_int_tuple(value, "logical_gpu_ids", table="fsdp2"),
        device_type=_string(value, "device_type", table="fsdp2"),
        mesh_dim_name=_string(value, "mesh_dim_name", table="fsdp2"),
        mesh_shape=_int_tuple(value, "mesh_shape", table="fsdp2"),
        reshard_after_forward=_boolean(value, "reshard_after_forward", table="fsdp2"),
        parameter_dtype=_string(value, "parameter_dtype", table="fsdp2"),
        reduce_dtype=_string(value, "reduce_dtype", table="fsdp2"),
        output_dtype=_string(value, "output_dtype", table="fsdp2"),
        cast_forward_inputs=_boolean(value, "cast_forward_inputs", table="fsdp2"),
        offload_policy=_string(value, "offload_policy", table="fsdp2"),
    )


def _parse_training(value: Mapping[str, Any]) -> RepresentationTrainingLoopConfig:
    required_fields = {
        "gradient_accumulation_steps",
        "target_optimizer_steps",
        "validation_every_optimizer_steps",
        "log_every_optimizer_steps",
    }
    _exact_fields(
        value,
        required_fields
        | (
            {"groups_per_rank_per_optimizer_step"}
            if "groups_per_rank_per_optimizer_step" in value
            else set()
        ),
        table="training",
    )
    return RepresentationTrainingLoopConfig(
        gradient_accumulation_steps=_int(
            value, "gradient_accumulation_steps", table="training"
        ),
        target_optimizer_steps=_int(value, "target_optimizer_steps", table="training"),
        validation_every_optimizer_steps=_int(
            value, "validation_every_optimizer_steps", table="training"
        ),
        log_every_optimizer_steps=_int(
            value, "log_every_optimizer_steps", table="training"
        ),
        groups_per_rank_per_optimizer_step=(
            _int(value, "groups_per_rank_per_optimizer_step", table="training")
            if "groups_per_rank_per_optimizer_step" in value
            else 1
        ),
    )


def _parse_post_training_internal_evaluation(
    value: Mapping[str, Any],
) -> RepresentationPostTrainingInternalEvaluationConfig:
    enabled = _boolean(value, "enabled", table="post_training_internal_evaluation")
    if not enabled:
        _exact_fields(
            value,
            {"enabled"},
            table="post_training_internal_evaluation",
        )
        return RepresentationPostTrainingInternalEvaluationConfig(enabled=False)
    fields = {
        "enabled",
        "evaluation_id",
        "ordered_group_manifest_path",
        "ordered_group_manifest_sha256",
        "counterfactual_manifest_path",
        "counterfactual_manifest_sha256",
        "report_path",
        "random_seed",
        "max_new_tokens",
        "eos_token_ids",
    }
    grounding_keys = {
        "grounding_manifest_path",
        "grounding_manifest_sha256",
    }
    if grounding_keys & set(value):
        fields.update(grounding_keys)
    _exact_fields(value, fields, table="post_training_internal_evaluation")
    return RepresentationPostTrainingInternalEvaluationConfig(
        enabled=True,
        evaluation_id=_string(
            value, "evaluation_id", table="post_training_internal_evaluation"
        ),
        ordered_group_manifest_path=_path(
            value,
            "ordered_group_manifest_path",
            table="post_training_internal_evaluation",
            allow_empty=False,
        ),
        ordered_group_manifest_sha256=_string(
            value,
            "ordered_group_manifest_sha256",
            table="post_training_internal_evaluation",
        ),
        counterfactual_manifest_path=_path(
            value,
            "counterfactual_manifest_path",
            table="post_training_internal_evaluation",
            allow_empty=False,
        ),
        counterfactual_manifest_sha256=_string(
            value,
            "counterfactual_manifest_sha256",
            table="post_training_internal_evaluation",
        ),
        grounding_manifest_path=(
            _path(
                value,
                "grounding_manifest_path",
                table="post_training_internal_evaluation",
                allow_empty=False,
            )
            if "grounding_manifest_path" in value
            else None
        ),
        grounding_manifest_sha256=(
            _string(
                value,
                "grounding_manifest_sha256",
                table="post_training_internal_evaluation",
            )
            if "grounding_manifest_sha256" in value
            else None
        ),
        report_path=_path(
            value,
            "report_path",
            table="post_training_internal_evaluation",
            allow_empty=False,
        ),
        random_seed=_int(
            value, "random_seed", table="post_training_internal_evaluation"
        ),
        max_new_tokens=_int(
            value, "max_new_tokens", table="post_training_internal_evaluation"
        ),
        eos_token_ids=_int_tuple(
            value, "eos_token_ids", table="post_training_internal_evaluation"
        ),
    )


def _parse_output(value: Mapping[str, Any]) -> RepresentationOutputConfig:
    _exact_fields(
        value,
        {"final_artifact_path", "metrics_jsonl_path", "allow_overwrite"},
        table="output",
    )
    return RepresentationOutputConfig(
        final_artifact_path=_path(
            value, "final_artifact_path", table="output", allow_empty=False
        ),
        metrics_jsonl_path=_path(
            value, "metrics_jsonl_path", table="output", allow_empty=False
        ),
        allow_overwrite=_boolean(value, "allow_overwrite", table="output"),
    )


def _parse_resume(value: Mapping[str, Any]) -> RepresentationResumeConfig:
    compatibility_fields = {
        "code_compatibility",
        "compatible_live_dirty_state_sha256",
    }
    present_compatibility_fields = set(value) & compatibility_fields
    expected_fields = {"enabled", "checkpoint_path", "strict_identity"}
    if present_compatibility_fields:
        expected_fields |= compatibility_fields
    _exact_fields(
        value,
        expected_fields,
        table="resume",
    )
    enabled = _boolean(value, "enabled", table="resume")
    raw_path = _string(value, "checkpoint_path", table="resume")
    checkpoint_path = (
        None
        if raw_path == NO_RESUME_CHECKPOINT
        else _absolute_path(Path(raw_path), field_name="resume.checkpoint_path")
    )
    return RepresentationResumeConfig(
        enabled=enabled,
        checkpoint_path=checkpoint_path,
        strict_identity=_boolean(value, "strict_identity", table="resume"),
        code_compatibility=(
            _string(value, "code_compatibility", table="resume")
            if present_compatibility_fields
            else NO_RESUME_CODE_COMPATIBILITY
        ),
        compatible_live_dirty_state_sha256=(
            None
            if not present_compatibility_fields
            or _string(
                value,
                "compatible_live_dirty_state_sha256",
                table="resume",
            )
            == NO_RESUME_CODE_COMPATIBILITY
            else _string(
                value,
                "compatible_live_dirty_state_sha256",
                table="resume",
            )
        ),
    )


def _parse_checkpoint(value: Mapping[str, Any]) -> RepresentationCheckpointConfig:
    _exact_fields(
        value,
        {
            "directory",
            "filename_prefix",
            "save_every_optimizer_steps",
            "save_final",
            "keep_last",
            "strict_identity",
            "optimizer_boundary_only",
            "format",
        },
        table="checkpoint",
    )
    return RepresentationCheckpointConfig(
        directory=_path(value, "directory", table="checkpoint", allow_empty=False),
        filename_prefix=_string(value, "filename_prefix", table="checkpoint"),
        save_every_optimizer_steps=_int(
            value, "save_every_optimizer_steps", table="checkpoint"
        ),
        save_final=_boolean(value, "save_final", table="checkpoint"),
        keep_last=_int(value, "keep_last", table="checkpoint"),
        strict_identity=_boolean(value, "strict_identity", table="checkpoint"),
        optimizer_boundary_only=_boolean(
            value, "optimizer_boundary_only", table="checkpoint"
        ),
        format=_string(value, "format", table="checkpoint"),
    )
