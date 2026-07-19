"""Strict TOML identity for executable Qwen3 representation training.

The representation phase deliberately has no implicit runnable experiment.
Every field that can change the model, native prompt, data population,
optimization, distributed topology, or checkpoint continuation participates
in the canonical configuration digest.  The one accepted optional scientific
field is balanced Matrix-CE temperature, whose registered default is ``1.0``;
its resolved value is still bound by the objective/run identity.

Loading a configuration is read-only.  In particular, it never initializes
CUDA, creates an output directory, loads model weights, or starts training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib
from typing import Any

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity

from .checkpoint import (
    MATRIX_CE_GLOBAL_REDUCTION,
    L_GEN_GLOBAL_REDUCTION,
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
)
from .distributed_checkpoint import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
)
from .data import SplitOverlapPolicy
from .fsdp2 import RepresentationFSDP2Config
from .losses import MatrixCEScoreMode
from .native_pipeline import RepresentationPromptConfig
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
    resolve_matrix_ce_score_config,
)
from .runtime import (
    ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256,
    ACCEPTED_QWEN3_MODEL_PATH,
    ACCEPTED_QWEN3_TOKENIZER_LENGTH,
    qwen3_input_embedding_identity,
)
from .trainer import (
    RepresentationOptimizerConfig,
    RepresentationPrecision,
    RepresentationSchedulerConfig,
    RepresentationSchedulerKind,
    RepresentationTrainerConfig,
)


REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION = "representation-training-config-v1"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2 = "representation-training-config-v2"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3 = "representation-training-config-v3"
REPRESENTATION_TRAINING_SCOPE = "qwen3_native_representation_phase_training"
ACCEPTED_QWEN3_MODEL_NAME = "Qwen3-VL-8B-Thinking"
ACCEPTED_QWEN3_MODEL_DTYPE = "bfloat16"
ACCEPTED_QWEN3_ATTENTION_BACKEND = "sdpa"
NO_INITIALIZATION_SOURCE = "none"
NO_RESUME_CHECKPOINT = "none"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
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


@dataclass(frozen=True, slots=True)
class RepresentationCodeConfig:
    repository: str
    commit: str
    dirty: bool
    dirty_state_sha256: str | None

    def __post_init__(self) -> None:
        _non_empty_text(self.repository, field_name="code.repository")
        _non_empty_text(self.commit, field_name="code.commit")
        _bool(self.dirty, field_name="code.dirty")
        if self.dirty:
            _sha256(self.dirty_state_sha256, field_name="code.dirty_state_sha256")
        elif self.dirty_state_sha256 is not None:
            raise ValueError(
                "code.dirty_state_sha256 must be 'none' when code.dirty is false"
            )

    @property
    def identity(self) -> CodeIdentity:
        return CodeIdentity(
            repository=self.repository,
            commit=self.commit,
            dirty_state_sha256=self.dirty_state_sha256,
        )


@dataclass(frozen=True, slots=True)
class RepresentationModelConfig:
    family: str
    model_name: str
    local_path: Path
    tokenizer_length: int
    chat_template_sha256: str
    dtype: str
    attention_backend: str
    local_files_only: bool
    trust_remote_code: bool
    tokenizer_resize: bool
    image_max_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.family != "qwen3_vl":
            raise ValueError("model.family must be 'qwen3_vl'")
        if self.model_name != ACCEPTED_QWEN3_MODEL_NAME:
            raise ValueError(f"model.model_name must be {ACCEPTED_QWEN3_MODEL_NAME!r}")
        if str(self.local_path) != ACCEPTED_QWEN3_MODEL_PATH:
            raise ValueError("model.local_path must be the accepted stable Qwen3 path")
        if self.tokenizer_length != ACCEPTED_QWEN3_TOKENIZER_LENGTH:
            raise ValueError("model.tokenizer_length differs from the accepted fixture")
        if self.chat_template_sha256 != ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256:
            raise ValueError(
                "model.chat_template_sha256 differs from the accepted fixture"
            )
        if self.dtype != ACCEPTED_QWEN3_MODEL_DTYPE:
            raise ValueError(f"model.dtype must be {ACCEPTED_QWEN3_MODEL_DTYPE!r}")
        if self.attention_backend != ACCEPTED_QWEN3_ATTENTION_BACKEND:
            raise ValueError(
                "model.attention_backend must be the accepted SDPA backend"
            )
        if self.local_files_only is not True:
            raise ValueError("model.local_files_only must be true")
        if self.trust_remote_code is not False:
            raise ValueError("model.trust_remote_code must be false")
        if self.tokenizer_resize is not False:
            raise ValueError("model.tokenizer_resize must be false")
        if self.image_max_pixels is not None:
            _positive_int(
                self.image_max_pixels,
                field_name="model.image_max_pixels",
            )

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            family=self.family,
            model_name=self.model_name,
            revision_or_path=str(self.local_path),
            tokenizer_length=self.tokenizer_length,
            chat_template_sha256=self.chat_template_sha256,
        )


@dataclass(frozen=True, slots=True)
class RepresentationDataSplitConfig:
    jsonl_path: Path
    source_sha256: str
    batch_size: int
    sampler_seed: int

    def __post_init__(self) -> None:
        _absolute_path(self.jsonl_path, field_name="data split jsonl_path")
        _sha256(self.source_sha256, field_name="data split source_sha256")
        _positive_int(self.batch_size, field_name="data split batch_size")
        if self.batch_size < 2:
            raise ValueError("same-image Matrix CE requires batch_size >= 2")
        _integer(self.sampler_seed, field_name="data split sampler_seed")


@dataclass(frozen=True, slots=True)
class RepresentationDataConfig:
    train: RepresentationDataSplitConfig
    validation: RepresentationDataSplitConfig
    warn_on_target_leakage: bool
    require_disjoint_validation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.train, RepresentationDataSplitConfig) or not isinstance(
            self.validation, RepresentationDataSplitConfig
        ):
            raise TypeError("data train/validation must be typed split configs")
        _bool(
            self.warn_on_target_leakage,
            field_name="data.warn_on_target_leakage",
        )
        if self.require_disjoint_validation is not True:
            raise ValueError("data.require_disjoint_validation must be true")
        if self.train.jsonl_path == self.validation.jsonl_path:
            raise ValueError("train and validation JSONL paths must be distinct")
        if self.train.source_sha256 == self.validation.source_sha256:
            raise ValueError("train and validation JSONL snapshots must be distinct")


@dataclass(frozen=True, slots=True)
class RepresentationDataConfigV2:
    """Explicit content-bound split-overlap contract for current training."""

    train: RepresentationDataSplitConfig
    validation: RepresentationDataSplitConfig
    warn_on_target_leakage: bool
    split_overlap_policy: SplitOverlapPolicy
    expected_overlap_report_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.train, RepresentationDataSplitConfig) or not isinstance(
            self.validation, RepresentationDataSplitConfig
        ):
            raise TypeError("data train/validation must be typed split configs")
        _bool(
            self.warn_on_target_leakage,
            field_name="data.warn_on_target_leakage",
        )
        if not isinstance(self.split_overlap_policy, SplitOverlapPolicy):
            raise TypeError("data.split_overlap_policy must be explicit")
        _sha256(
            self.expected_overlap_report_sha256,
            field_name="data.expected_overlap_report_sha256",
        )
        if self.train.jsonl_path == self.validation.jsonl_path:
            raise ValueError("train and validation JSONL paths must be distinct")
        if self.train.source_sha256 == self.validation.source_sha256:
            raise ValueError("train and validation JSONL snapshots must be distinct")


@dataclass(frozen=True, slots=True)
class RepresentationObjectiveExecutionConfig:
    objective: RepresentationObjectiveConfig
    manifold_enabled: bool
    manifold_weight: float
    norm_loss: str

    def __post_init__(self) -> None:
        if not isinstance(self.objective, RepresentationObjectiveConfig):
            raise TypeError("objective must be a RepresentationObjectiveConfig")
        if self.objective.kind is not RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN:
            raise ValueError("the executable baseline requires Matrix CE plus L_gen")
        if self.manifold_enabled is not False or self.manifold_weight != 0.0:
            raise ValueError(
                "manifold optimization must be explicitly disabled at zero"
            )
        if self.norm_loss != "unset_not_implemented":
            raise ValueError("norm_loss must remain explicitly unset_not_implemented")


@dataclass(frozen=True, slots=True)
class RepresentationObjectiveExecutionConfigV2:
    """Executable norm-aware objective with manifold fixed at zero."""

    objective: RepresentationObjectiveConfigV2
    manifold_enabled: bool
    manifold_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.objective, RepresentationObjectiveConfigV2):
            raise TypeError("objective must be a RepresentationObjectiveConfigV2")
        if self.manifold_enabled is not False or self.manifold_weight != 0.0:
            raise ValueError(
                "manifold optimization must be explicitly disabled at zero"
            )


@dataclass(frozen=True, slots=True)
class RepresentationObjectiveExecutionConfigV3:
    """Executable objective with explicit Matrix-CE scoring mathematics."""

    objective: RepresentationObjectiveConfigV3
    manifold_enabled: bool
    manifold_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.objective, RepresentationObjectiveConfigV3):
            raise TypeError("objective must be a RepresentationObjectiveConfigV3")
        if self.manifold_enabled is not False or self.manifold_weight != 0.0:
            raise ValueError(
                "manifold optimization must be explicitly disabled at zero"
            )


@dataclass(frozen=True, slots=True)
class RepresentationAdamWConfig:
    optimizer_type: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: bool
    capturable: bool
    differentiable: bool
    fused: bool
    decoupled_weight_decay: bool

    def __post_init__(self) -> None:
        if self.optimizer_type != "adamw":
            raise ValueError("optimizer.type must be 'adamw'")
        # Reuse the trainer's numerical validation for scientific parameters.
        self.trainer_config
        for field_name in (
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
        ):
            _bool(getattr(self, field_name), field_name=f"optimizer.{field_name}")
        if self.amsgrad or self.maximize or self.capturable or self.differentiable:
            raise ValueError(
                "representation AdamW requires amsgrad/maximize/capturable/"
                "differentiable=false"
            )
        if self.foreach or self.fused:
            raise ValueError(
                "representation AdamW requires explicit foreach=false and fused=false"
            )
        if not self.decoupled_weight_decay:
            raise ValueError("representation AdamW requires decoupled weight decay")

    @property
    def trainer_config(self) -> RepresentationOptimizerConfig:
        return RepresentationOptimizerConfig(
            learning_rate=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    @property
    def torch_options(self) -> dict[str, bool]:
        """Resolved options a runner must pass instead of using torch defaults."""

        return {
            "amsgrad": self.amsgrad,
            "maximize": self.maximize,
            "foreach": self.foreach,
            "capturable": self.capturable,
            "differentiable": self.differentiable,
            "fused": self.fused,
        }


@dataclass(frozen=True, slots=True)
class RepresentationExecutionConfig:
    precision: RepresentationPrecision
    max_grad_norm: float
    require_all_adapter_gradients: bool
    gradient_clip_norm_type: float
    gradient_clip_error_if_nonfinite: bool

    def __post_init__(self) -> None:
        self.trainer_config
        if self.gradient_clip_norm_type != 2.0:
            raise ValueError("execution.gradient_clip_norm_type must be 2.0")
        if self.gradient_clip_error_if_nonfinite is not True:
            raise ValueError("execution.gradient_clip_error_if_nonfinite must be true")

    @property
    def trainer_config(self) -> RepresentationTrainerConfig:
        return RepresentationTrainerConfig(
            precision=self.precision,
            max_grad_norm=self.max_grad_norm,
            require_all_adapter_gradients=self.require_all_adapter_gradients,
        )


@dataclass(frozen=True, slots=True)
class RepresentationInitializationConfig:
    kind: str
    seed: int
    source_artifact_sha256: None
    allow_legacy_checkpoint_initialization: bool

    def __post_init__(self) -> None:
        if self.kind != "fresh_random":
            raise ValueError("initialization.kind must be 'fresh_random'")
        _integer(self.seed, field_name="initialization.seed")
        if self.source_artifact_sha256 is not None:
            raise ValueError("fresh initialization source must be None")
        if self.allow_legacy_checkpoint_initialization is not False:
            raise ValueError(
                "initialization.allow_legacy_checkpoint_initialization must be false"
            )


@dataclass(frozen=True, slots=True)
class RepresentationFSDP2TopologyConfig:
    strategy: str
    world_size: int
    physical_gpu_ids: tuple[int, ...]
    logical_gpu_ids: tuple[int, ...]
    device_type: str
    mesh_dim_name: str
    mesh_shape: tuple[int, ...]
    reshard_after_forward: bool
    parameter_dtype: str
    reduce_dtype: str
    output_dtype: str
    cast_forward_inputs: bool
    offload_policy: str

    def __post_init__(self) -> None:
        if self.strategy != "fsdp2":
            raise ValueError("fsdp2.strategy must be 'fsdp2'")
        if self.world_size != 2:
            raise ValueError("representation FSDP2 world_size must be 2")
        if (
            len(self.physical_gpu_ids) != self.world_size
            or any(
                isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
                for gpu_id in self.physical_gpu_ids
            )
            or len(set(self.physical_gpu_ids)) != self.world_size
        ):
            raise ValueError(
                "fsdp2.physical_gpu_ids must contain two distinct non-negative "
                "physical GPU IDs"
            )
        if self.logical_gpu_ids != (0, 1):
            raise ValueError("CUDA-visible logical GPU IDs must be [0, 1]")
        if self.device_type != "cuda":
            raise ValueError("representation FSDP2 device_type must be 'cuda'")
        if self.mesh_dim_name != "fsdp" or self.mesh_shape != (2,):
            raise ValueError(
                "representation FSDP2 requires one mesh dimension fsdp=[2]"
            )
        _bool(
            self.reshard_after_forward,
            field_name="fsdp2.reshard_after_forward",
        )
        if self.parameter_dtype not in {"bfloat16", "float32"}:
            raise ValueError("fsdp2.parameter_dtype must be bfloat16 or float32")
        if self.reduce_dtype != "float32":
            raise ValueError("fsdp2.reduce_dtype must be float32")
        if self.output_dtype != self.parameter_dtype:
            raise ValueError("fsdp2 output and parameter dtypes must match")
        if self.cast_forward_inputs is not True:
            raise ValueError("fsdp2.cast_forward_inputs must be true")
        if self.offload_policy != "none":
            raise ValueError("the accepted representation FSDP2 path uses no offload")

    @property
    def runtime_config(self) -> RepresentationFSDP2Config:
        return RepresentationFSDP2Config(
            world_size=self.world_size,
            reshard_after_forward=self.reshard_after_forward,
        )


@dataclass(frozen=True, slots=True)
class RepresentationTrainingLoopConfig:
    gradient_accumulation_steps: int
    target_optimizer_steps: int
    validation_every_optimizer_steps: int
    log_every_optimizer_steps: int
    groups_per_rank_per_optimizer_step: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "gradient_accumulation_steps",
            "target_optimizer_steps",
            "validation_every_optimizer_steps",
            "log_every_optimizer_steps",
            "groups_per_rank_per_optimizer_step",
        ):
            _positive_int(
                getattr(self, field_name), field_name=f"training.{field_name}"
            )
        if (
            self.groups_per_rank_per_optimizer_step > 1
            and self.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "training.groups_per_rank_per_optimizer_step > 1 requires "
                "training.gradient_accumulation_steps = 1"
            )

    def accumulation_identity(
        self, *, world_size: int
    ) -> RepresentationAccumulationIdentity:
        if self.groups_per_rank_per_optimizer_step > 1:
            return RepresentationAccumulationIdentityV2(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                data_parallel_world_size=world_size,
                matrix_ce_reduction=MATRIX_CE_GLOBAL_REDUCTION,
                l_gen_reduction=L_GEN_GLOBAL_REDUCTION,
                checkpoint_at_optimizer_boundary=True,
                groups_per_rank_per_optimizer_step=(
                    self.groups_per_rank_per_optimizer_step
                ),
            )
        return RepresentationAccumulationIdentity(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            data_parallel_world_size=world_size,
            matrix_ce_reduction=MATRIX_CE_GLOBAL_REDUCTION,
            l_gen_reduction=L_GEN_GLOBAL_REDUCTION,
            checkpoint_at_optimizer_boundary=True,
        )


@dataclass(frozen=True, slots=True)
class RepresentationOutputConfig:
    final_artifact_path: Path
    metrics_jsonl_path: Path
    allow_overwrite: bool

    def __post_init__(self) -> None:
        _absolute_path(
            self.final_artifact_path, field_name="output.final_artifact_path"
        )
        _absolute_path(self.metrics_jsonl_path, field_name="output.metrics_jsonl_path")
        if self.final_artifact_path == self.metrics_jsonl_path:
            raise ValueError("artifact and metrics output paths must differ")
        if self.allow_overwrite is not False:
            raise ValueError("output.allow_overwrite must be false")


@dataclass(frozen=True, slots=True)
class RepresentationResumeConfig:
    enabled: bool
    checkpoint_path: Path | None
    strict_identity: bool

    def __post_init__(self) -> None:
        _bool(self.enabled, field_name="resume.enabled")
        if self.enabled:
            if self.checkpoint_path is None:
                raise ValueError("enabled resume requires an exact checkpoint path")
            _absolute_path(self.checkpoint_path, field_name="resume.checkpoint_path")
        elif self.checkpoint_path is not None:
            raise ValueError("disabled resume checkpoint_path must be 'none'")
        if self.strict_identity is not True:
            raise ValueError("resume.strict_identity must be true")


@dataclass(frozen=True, slots=True)
class RepresentationCheckpointConfig:
    directory: Path
    filename_prefix: str
    save_every_optimizer_steps: int
    save_final: bool
    keep_last: int
    strict_identity: bool
    optimizer_boundary_only: bool
    format: str

    def __post_init__(self) -> None:
        _absolute_path(self.directory, field_name="checkpoint.directory")
        _safe_filename(self.filename_prefix, field_name="checkpoint.filename_prefix")
        _positive_int(
            self.save_every_optimizer_steps,
            field_name="checkpoint.save_every_optimizer_steps",
        )
        _positive_int(self.keep_last, field_name="checkpoint.keep_last")
        if self.save_final is not True:
            raise ValueError("checkpoint.save_final must be true")
        if self.strict_identity is not True:
            raise ValueError("checkpoint.strict_identity must be true")
        if self.optimizer_boundary_only is not True:
            raise ValueError("checkpoint.optimizer_boundary_only must be true")
        if self.format not in {
            DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
            DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
        }:
            raise ValueError("checkpoint.format differs from the implementation schema")


@dataclass(frozen=True, slots=True)
class RepresentationTrainingConfig:
    schema_version: str
    scope: str
    run_id: str
    code: RepresentationCodeConfig
    model: RepresentationModelConfig
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
    source_path: Path
    source_toml_sha256: str
    canonical_config_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version not in {
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
            REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
        }:
            raise ValueError("representation training config schema mismatch")
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
                    "training config v3 requires its explicit Matrix-CE objective"
                )
            if self.scheduler.kind is not RepresentationSchedulerKind.HISTORICAL_COSINE:
                raise ValueError("training config v3 requires historical cosine")
            if self.checkpoint.format != (
                DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2
            ):
                raise ValueError("training config v3 requires DCP schema v2")
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
            "conditioning_provider": self.provider.provider.value,
            "prompt_identity": self.prompt.identity,
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
            "gpu_work_launched": False,
        }


def load_representation_training_config(
    path: str | Path,
    *,
    verify_external_files: bool = True,
) -> RepresentationTrainingConfig:
    """Parse and validate one complete representation-training TOML identity.

    ``verify_external_files=False`` is reserved for schema/unit fixtures.  The
    CLI and production runner use the default, which verifies the local model
    directory, both exact JSONL byte hashes, output parents, and any requested
    resume checkpoint without loading model weights or touching CUDA.
    """

    if not isinstance(verify_external_files, bool):
        raise TypeError("verify_external_files must be a bool")
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
    _exact_fields(value, _TOP_LEVEL_FIELDS, table="root")
    canonical_config_sha256 = _canonical_mapping_sha256(value)

    schema_version = _string(value, "schema_version", table="root")
    scope = _string(value, "scope", table="root")
    run_id = _string(value, "run_id", table="root")
    code = _parse_code(_table(value, "code", table="root"))
    model = _parse_model(_table(value, "model", table="root"))
    provider = _parse_conditioning(
        _table(value, "conditioning", table="root"), model.identity
    )
    data = _parse_data(
        _table(value, "data", table="root"),
        schema_version=schema_version,
    )
    prompt = _parse_prompt(_table(value, "prompt", table="root"))
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

    config = RepresentationTrainingConfig(
        schema_version=schema_version,
        scope=scope,
        run_id=run_id,
        code=code,
        model=model,
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
        source_path=source_path,
        source_toml_sha256=source_toml_sha256,
        canonical_config_sha256=canonical_config_sha256,
    )
    if verify_external_files:
        _verify_external_files(config)
    return config


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


def _parse_prompt(value: Mapping[str, Any]) -> RepresentationPromptConfig:
    _exact_fields(value, {"identity", "template", "sha256"}, table="prompt")
    return RepresentationPromptConfig(
        identity=_string(value, "identity", table="prompt"),
        template=_string(value, "template", table="prompt"),
        expected_sha256=_string(value, "sha256", table="prompt"),
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
    if schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3:
        required = {
            "identity",
            "kind",
            "matrix_ce_weight",
            "l_gen_weight",
            "norm_weight",
            "matrix_ce_mode",
            "manifold_enabled",
            "manifold_weight",
        }
        optional = (
            {"matrix_ce_temperature"} if "matrix_ce_temperature" in value else set()
        )
        _exact_fields(value, required | optional, table="objective")
        kind_raw = _string(value, "kind", table="objective")
        try:
            kind = RepresentationObjectiveKind(kind_raw)
        except ValueError as error:
            raise ValueError(f"objective.kind is unsupported: {kind_raw!r}") from error
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
    _exact_fields(
        value,
        {"enabled", "checkpoint_path", "strict_identity"},
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


def _verify_external_files(config: RepresentationTrainingConfig) -> None:
    if not config.model.local_path.is_dir():
        raise ValueError(
            f"accepted Qwen3 model directory is unavailable: {config.model.local_path}"
        )
    required_model_files = (
        "config.json",
        "tokenizer.json",
        "chat_template.json",
        "model.safetensors.index.json",
    )
    missing = tuple(
        name
        for name in required_model_files
        if not (config.model.local_path / name).is_file()
    )
    if missing:
        raise ValueError(f"accepted Qwen3 directory is incomplete: {missing}")
    for name, split in (
        ("train", config.data.train),
        ("validation", config.data.validation),
    ):
        path = _existing_file_path(
            split.jsonl_path, field_name=f"data.{name}.jsonl_path"
        )
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != split.source_sha256:
            raise ValueError(
                f"data.{name}.source_sha256 mismatch: expected "
                f"{split.source_sha256}, got {actual}"
            )
    for name, path in (
        ("output.final_artifact_path", config.output.final_artifact_path),
        ("output.metrics_jsonl_path", config.output.metrics_jsonl_path),
        ("checkpoint.directory", config.checkpoint.directory),
    ):
        parent = path if name == "checkpoint.directory" else path.parent
        existing_parent = _nearest_existing_parent(parent)
        if not existing_parent.is_dir():
            raise ValueError(f"{name} has no usable directory ancestor")
    if config.resume.enabled:
        assert config.resume.checkpoint_path is not None
        if not config.resume.checkpoint_path.is_dir():
            raise ValueError(
                "resume.checkpoint_path must be an existing distributed "
                f"checkpoint directory: {config.resume.checkpoint_path}"
            )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, table: str
) -> None:
    actual = set(value)
    missing = sorted(set(expected) - actual)
    unknown = sorted(actual - set(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"[{table}] fields do not match schema: {' '.join(details)}")


def _table(value: Mapping[str, Any], key: str, *, table: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(f"[{table}.{key}] must be a TOML table")
    if any(not isinstance(name, str) for name in item):
        raise TypeError(f"[{table}.{key}] keys must be strings")
    return item


def _string(value: Mapping[str, Any], key: str, *, table: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{table}.{key} must be a non-empty string")
    return item


def _boolean(value: Mapping[str, Any], key: str, *, table: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise TypeError(f"{table}.{key} must be a boolean")
    return item


def _int(value: Mapping[str, Any], key: str, *, table: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{table}.{key} must be an integer")
    return item


def _float(value: Mapping[str, Any], key: str, *, table: str) -> float:
    item = value.get(key)
    if not isinstance(item, float) or not math.isfinite(item):
        raise TypeError(f"{table}.{key} must be an explicit finite TOML float")
    return item


def _int_tuple(value: Mapping[str, Any], key: str, *, table: str) -> tuple[int, ...]:
    item = value.get(key)
    if not isinstance(item, list) or not item:
        raise TypeError(f"{table}.{key} must be a non-empty integer array")
    if any(isinstance(entry, bool) or not isinstance(entry, int) for entry in item):
        raise TypeError(f"{table}.{key} must contain only integers")
    return tuple(item)


def _float_tuple(
    value: Mapping[str, Any], key: str, *, table: str, length: int
) -> tuple[float, ...]:
    item = value.get(key)
    if not isinstance(item, list) or len(item) != length:
        raise TypeError(f"{table}.{key} must be a {length}-float array")
    if any(not isinstance(entry, float) or not math.isfinite(entry) for entry in item):
        raise TypeError(f"{table}.{key} must contain explicit finite TOML floats")
    return tuple(item)


def _path(value: Mapping[str, Any], key: str, *, table: str, allow_empty: bool) -> Path:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise TypeError(f"{table}.{key} must be a path string")
    return _absolute_path(Path(item), field_name=f"{table}.{key}")


def _absolute_path(value: Path, *, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if "\x00" in str(value):
        raise ValueError(f"{field_name} contains a null byte")
    return value


def _existing_file_path(value: str | Path, *, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a path")
    path = Path(value)
    _absolute_path(path, field_name=field_name)
    if not path.is_file():
        raise ValueError(f"{field_name} does not resolve to a file: {path}")
    return path


def _configuration_source_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("configuration path must be a path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            f"configuration path does not resolve to a file: {value}"
        ) from error
    if not path.is_file():
        raise ValueError(f"configuration path is not a file: {path}")
    return path


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _safe_filename(value: str, *, field_name: str) -> None:
    _non_empty_text(value, field_name=field_name)
    if value in {".", ".."} or Path(value).name != value or "\x00" in value:
        raise ValueError(f"{field_name} must be a plain filename prefix")


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def _positive_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


__all__ = [
    "ACCEPTED_QWEN3_ATTENTION_BACKEND",
    "ACCEPTED_QWEN3_MODEL_DTYPE",
    "ACCEPTED_QWEN3_MODEL_NAME",
    "NO_INITIALIZATION_SOURCE",
    "NO_RESUME_CHECKPOINT",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2",
    "REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3",
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
    "RepresentationResumeConfig",
    "RepresentationTrainingConfig",
    "RepresentationTrainingLoopConfig",
    "load_representation_training_config",
]
