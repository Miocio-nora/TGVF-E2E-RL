"""Aggregate representation-training run configuration contract.

This leaf composes immutable component schemas and owns only whole-run
cross-field invariants and its validation payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tgvf_rl.conditioning import TargetConditioningConfig
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.public_api_compat import rebind_public_class
from tgvf_rl.representation.adapter import TGVFAdapterVariant

from .checkpoint import RepresentationAccumulationIdentity
from .config_schema import (
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    REPRESENTATION_TRAINING_SCOPE,
    RepresentationAdamWConfig,
    RepresentationCheckpointConfig,
    RepresentationCodeConfig,
    RepresentationDataConfig,
    RepresentationDataConfigV2,
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
from .config_values import _absolute_path, _non_empty_text, _sha256
from .data import SplitOverlapPolicy
from .distributed_checkpoint import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
)
from .native_pipeline import (
    REPRESENTATION_PROMPT_IDENTITY,
    REPRESENTATION_PROMPT_SCHEMA_VERSION,
    RepresentationPromptConfig,
)
from .objective import resolve_matrix_ce_score_config
from .trainer import (
    RepresentationPrecision,
    RepresentationSchedulerConfig,
    RepresentationSchedulerKind,
)


@dataclass(frozen=True, slots=True)
class RepresentationTrainingConfig:
    schema_version: str
    scope: str
    run_id: str
    code: RepresentationCodeConfig
    model: RepresentationModelConfig
    adapter_variant: TGVFAdapterVariant
    provider: TargetConditioningConfig
    data: RepresentationDataConfig | RepresentationDataConfigV2
    prompt: RepresentationPromptConfig
    objective: (
        RepresentationObjectiveExecutionConfig
        | RepresentationObjectiveExecutionConfigV2
        | RepresentationObjectiveExecutionConfigV3
    )
    optimizer: RepresentationAdamWConfig
    scheduler: RepresentationSchedulerConfig
    execution: RepresentationExecutionConfig
    initialization: RepresentationInitializationConfig
    fsdp2: RepresentationFSDP2TopologyConfig
    training: RepresentationTrainingLoopConfig
    output: RepresentationOutputConfig
    resume: RepresentationResumeConfig
    checkpoint: RepresentationCheckpointConfig
    post_training_internal_evaluation: (
        RepresentationPostTrainingInternalEvaluationConfig | None
    )
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version not in {
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
        }:
            raise ValueError("representation training config schema mismatch")
        if self.prompt.schema_version != REPRESENTATION_PROMPT_SCHEMA_VERSION:
            raise ValueError("representation training requires prompt schema v1")
        if self.prompt.identity != REPRESENTATION_PROMPT_IDENTITY:
            raise ValueError(
                "representation training requires the fixed image-question prompt identity"
            )
        if self.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION:
            if not isinstance(self.data, RepresentationDataConfig):
                raise TypeError(
                    "training config v1 requires its historical data contract"
                )
            if not isinstance(self.objective, RepresentationObjectiveExecutionConfig):
                raise TypeError("training config v1 requires its historical objective")
            if self.scheduler.kind is RepresentationSchedulerKind.HISTORICAL_COSINE:
                raise ValueError("training config v1 cannot select historical cosine")
            if self.checkpoint.format != (
                DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION
            ):
                raise ValueError("training config v1 requires DCP schema v1")
        elif self.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2:
            if not isinstance(self.data, RepresentationDataConfigV2):
                raise TypeError(
                    "training config v2 requires an overlap-bound data contract"
                )
            if not isinstance(self.objective, RepresentationObjectiveExecutionConfigV2):
                raise TypeError("training config v2 requires its norm-aware objective")
            if self.scheduler.kind is not RepresentationSchedulerKind.HISTORICAL_COSINE:
                raise ValueError("training config v2 requires historical cosine")
            if self.checkpoint.format != (
                DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
            ):
                raise ValueError("training config v2 requires DCP schema v2")
        else:
            if not isinstance(self.data, RepresentationDataConfigV2):
                raise TypeError(
                    "training config v3 requires an overlap-bound data contract"
                )
            if not isinstance(self.objective, RepresentationObjectiveExecutionConfigV3):
                raise TypeError(
                    "training config v3 requires its mode-aware Matrix-CE objective"
                )
            if self.scheduler.kind is not RepresentationSchedulerKind.HISTORICAL_COSINE:
                raise ValueError("training config v3 requires historical cosine")
            if self.checkpoint.format != (
                DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
            ):
                raise ValueError("training config v3 requires DCP schema v2")
        if not isinstance(self.adapter_variant, TGVFAdapterVariant):
            raise TypeError("representation Adapter variant must be explicit")
        if self.schema_version != REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5:
            if self.adapter_variant is not TGVFAdapterVariant.FULL_D_DEEPSTACK:
                raise ValueError(
                    "non-historical Adapter variants require training config schema v5"
                )
        if self.schema_version in {
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
        }:
            if self.post_training_internal_evaluation is None:
                raise ValueError(
                    "training config v4 requires post-training internal evaluation"
                )
        elif self.post_training_internal_evaluation is not None:
            raise ValueError(
                "training config v1-v3 cannot carry post-training internal evaluation"
            )
        if self.scope != REPRESENTATION_TRAINING_SCOPE:
            raise ValueError("representation training scope mismatch")
        _non_empty_text(self.run_id, field_name="run_id")
        _absolute_path(self.source_path, field_name="configuration source path")
        _sha256(self.source_toml_sha256, field_name="source_toml_sha256")
        _sha256(self.canonical_config_sha256, field_name="canonical_config_sha256")
        if self.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION:
            if self.scheduler.total_steps != self.training.target_optimizer_steps:
                raise ValueError(
                    "scheduler.total_steps must equal training.target_optimizer_steps"
                )
        elif self.training.target_optimizer_steps > self.scheduler.total_steps:
            raise ValueError(
                "training.target_optimizer_steps cannot exceed the historical "
                "scheduler horizon"
            )
        expected_precision_dtype = (
            "bfloat16"
            if self.execution.precision is RepresentationPrecision.BF16
            else "float32"
        )
        if self.fsdp2.parameter_dtype != expected_precision_dtype:
            raise ValueError(
                "FSDP2 parameter dtype must match representation execution precision"
            )
        if self.data.train.batch_size < 2 or self.data.validation.batch_size < 2:
            raise ValueError("both data splits require same-image comparison batches")
        log_every = self.training.log_every_optimizer_steps
        if self.checkpoint.save_every_optimizer_steps % log_every:
            raise ValueError(
                "checkpoint.save_every_optimizer_steps must be divisible by "
                "training.log_every_optimizer_steps so every periodic checkpoint "
                "has a durable train metric"
            )
        if self.training.target_optimizer_steps % log_every:
            raise ValueError(
                "training.target_optimizer_steps must be divisible by "
                "training.log_every_optimizer_steps so the final checkpoint has "
                "a durable train metric"
            )
        if (
            self.training.validation_every_optimizer_steps
            % self.checkpoint.save_every_optimizer_steps
        ):
            raise ValueError(
                "training.validation_every_optimizer_steps must be divisible by "
                "checkpoint.save_every_optimizer_steps so validation always runs "
                "after a durable checkpoint"
            )

    @property
    def code_identity(self) -> CodeIdentity:
        return self.code.identity

    @property
    def model_identity(self) -> ModelIdentity:
        return self.model.identity

    @property
    def accumulation_identity(self) -> RepresentationAccumulationIdentity:
        return self.training.accumulation_identity(world_size=self.fsdp2.world_size)

    def validation_payload(self) -> dict[str, object]:
        """Concise, JSON-safe identity emitted by the validation-only CLI."""

        matrix_ce_mode, matrix_ce_temperature = resolve_matrix_ce_score_config(
            self.objective.objective
        )
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "run_id": self.run_id,
            "source_path": str(self.source_path),
            "source_toml_sha256": self.source_toml_sha256,
            "canonical_config_sha256": self.canonical_config_sha256,
            "code": {
                "repository": self.code.repository,
                "commit": self.code.commit,
                "dirty": self.code.dirty,
                "dirty_state_sha256": self.code.dirty_state_sha256,
            },
            "model": {
                "family": self.model.family,
                "model_name": self.model.model_name,
                "local_path": str(self.model.local_path),
                "tokenizer_length": self.model.tokenizer_length,
                "chat_template_sha256": self.model.chat_template_sha256,
                "dtype": self.model.dtype,
                "attention_backend": self.model.attention_backend,
                "image_max_pixels": self.model.image_max_pixels,
            },
            "adapter_variant": self.adapter_variant.value,
            "conditioning_provider": self.provider.provider.value,
            "prompt_identity": self.prompt.identity,
            "prompt_schema_version": self.prompt.schema_version,
            "prompt_sha256": self.prompt.sha256,
            "objective_identity": self.objective.objective.identity,
            "objective_schema_version": self.objective.objective.schema_version,
            "matrix_ce_mode": matrix_ce_mode.value,
            "matrix_ce_temperature": matrix_ce_temperature,
            "train_source_sha256": self.data.train.source_sha256,
            "validation_source_sha256": self.data.validation.source_sha256,
            "split_overlap_policy": (
                self.data.split_overlap_policy.value
                if isinstance(self.data, RepresentationDataConfigV2)
                else SplitOverlapPolicy.REQUIRE_DISJOINT.value
            ),
            "expected_overlap_report_sha256": (
                self.data.expected_overlap_report_sha256
                if isinstance(self.data, RepresentationDataConfigV2)
                else None
            ),
            "world_size": self.fsdp2.world_size,
            "physical_gpu_ids": list(self.fsdp2.physical_gpu_ids),
            "target_optimizer_steps": self.training.target_optimizer_steps,
            "resume_enabled": self.resume.enabled,
            "post_training_internal_evaluation_enabled": (
                False
                if self.post_training_internal_evaluation is None
                else self.post_training_internal_evaluation.enabled
            ),
            "gpu_work_launched": False,
        }


_PUBLIC_CONFIG_MODULE = "tgvf_rl.representation.training.config"
rebind_public_class(
    RepresentationTrainingConfig,
    implementation_module=__name__,
    public_module=_PUBLIC_CONFIG_MODULE,
)


__all__ = ["RepresentationTrainingConfig"]
