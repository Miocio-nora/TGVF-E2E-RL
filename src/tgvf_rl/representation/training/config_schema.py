"""Immutable representation-training configuration value objects.

The schema owns scientific identities and cross-field invariants only.  TOML
parsing, filesystem binding, and execution remain outside this leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.public_api_compat import rebind_public_class

from .checkpoint import (
    L_GEN_GLOBAL_REDUCTION,
    MATRIX_CE_GLOBAL_REDUCTION,
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
)
from .config_values import (
    _absolute_path,
    _bool,
    _integer,
    _non_empty_text,
    _positive_int,
    _safe_filename,
    _sha256,
)
from .data import SplitOverlapPolicy
from .distributed_checkpoint import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION_V2,
)
from .fsdp2 import RepresentationFSDP2Config
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
)
from .runtime import ACCEPTED_QWEN3_MODEL_FIXTURES
from .trainer import (
    RepresentationOptimizerConfig,
    RepresentationPrecision,
    RepresentationTrainerConfig,
)


REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION = "representation-training-config-v1"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2 = "representation-training-config-v2"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3 = "representation-training-config-v3"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4 = "representation-training-config-v4"
REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5 = "representation-training-config-v5"
REPRESENTATION_TRAINING_SCOPE = "qwen3_native_representation_phase_training"
ACCEPTED_QWEN3_MODEL_NAME = "Qwen3-VL-8B-Instruct"
ACCEPTED_QWEN3_MODEL_DTYPE = "bfloat16"
ACCEPTED_QWEN3_ATTENTION_BACKEND = "sdpa"
NO_INITIALIZATION_SOURCE = "none"
NO_RESUME_CHECKPOINT = "none"
NO_RESUME_CODE_COMPATIBILITY = "none"
VALIDATED_NON_TRAINING_CODE_TRANSITION = "validated_non_training_code_transition_v1"


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
        fixture = ACCEPTED_QWEN3_MODEL_FIXTURES.get(self.model_name)
        if fixture is None:
            raise ValueError("model.model_name must identify a pinned Qwen3 edition")
        if str(self.local_path) != fixture["path"]:
            raise ValueError("model.local_path must match the selected Qwen3 edition")
        if self.tokenizer_length != fixture["tokenizer_length"]:
            raise ValueError("model.tokenizer_length differs from the accepted fixture")
        if self.chat_template_sha256 != fixture["chat_template_sha256"]:
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
        if self.world_size not in (2, 4):
            raise ValueError("representation FSDP2 world_size must be 2 or 4")
        if (
            len(self.physical_gpu_ids) != self.world_size
            or any(
                isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
                for gpu_id in self.physical_gpu_ids
            )
            or len(set(self.physical_gpu_ids)) != self.world_size
        ):
            raise ValueError(
                "fsdp2.physical_gpu_ids must contain world_size distinct "
                "non-negative physical GPU IDs"
            )
        expected_logical_gpu_ids = tuple(range(self.world_size))
        if self.logical_gpu_ids != expected_logical_gpu_ids:
            raise ValueError(
                "CUDA-visible logical GPU IDs must be contiguous from zero "
                "through world_size - 1"
            )
        if self.device_type != "cuda":
            raise ValueError("representation FSDP2 device_type must be 'cuda'")
        if self.mesh_dim_name != "fsdp" or self.mesh_shape != (self.world_size,):
            raise ValueError(
                "representation FSDP2 requires one mesh dimension whose size "
                "equals world_size"
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
        if self.groups_per_rank_per_optimizer_step > 1:
            if (
                self.groups_per_rank_per_optimizer_step
                % self.gradient_accumulation_steps
                != 0
            ):
                raise ValueError(
                    "training.groups_per_rank_per_optimizer_step must be evenly "
                    "divisible by training.gradient_accumulation_steps"
                )
            if (
                self.groups_per_rank_per_optimizer_step
                // self.gradient_accumulation_steps
                <= 1
            ):
                raise ValueError(
                    "direct multi-group execution requires more than one group "
                    "per accumulation microstep"
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
class RepresentationPostTrainingInternalEvaluationConfig:
    """Explicit once-after-completion internal-evaluation switch."""

    enabled: bool
    evaluation_id: str | None = None
    ordered_group_manifest_path: Path | None = None
    ordered_group_manifest_sha256: str | None = None
    counterfactual_manifest_path: Path | None = None
    counterfactual_manifest_sha256: str | None = None
    grounding_manifest_path: Path | None = None
    grounding_manifest_sha256: str | None = None
    report_path: Path | None = None
    random_seed: int | None = None
    max_new_tokens: int | None = None
    eos_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _bool(self.enabled, field_name="post_training_internal_evaluation.enabled")
        optional_values = (
            self.evaluation_id,
            self.ordered_group_manifest_path,
            self.ordered_group_manifest_sha256,
            self.counterfactual_manifest_path,
            self.counterfactual_manifest_sha256,
            self.grounding_manifest_path,
            self.grounding_manifest_sha256,
            self.report_path,
            self.random_seed,
            self.max_new_tokens,
            self.eos_token_ids,
        )
        if not self.enabled:
            if any(value is not None for value in optional_values):
                raise ValueError(
                    "disabled post-training internal evaluation cannot carry inputs"
                )
            return
        _non_empty_text(
            self.evaluation_id,
            field_name="post_training_internal_evaluation.evaluation_id",
        )
        for name, path in (
            ("ordered_group_manifest_path", self.ordered_group_manifest_path),
            ("counterfactual_manifest_path", self.counterfactual_manifest_path),
            ("report_path", self.report_path),
        ):
            if not isinstance(path, Path):
                raise TypeError(
                    f"post_training_internal_evaluation.{name} must be a Path"
                )
            _absolute_path(path, field_name=f"post_training_internal_evaluation.{name}")
        for name, digest in (
            ("ordered_group_manifest_sha256", self.ordered_group_manifest_sha256),
            ("counterfactual_manifest_sha256", self.counterfactual_manifest_sha256),
        ):
            _sha256(digest, field_name=f"post_training_internal_evaluation.{name}")
        if (self.grounding_manifest_path is None) != (
            self.grounding_manifest_sha256 is None
        ):
            raise ValueError(
                "grounding manifest path and SHA256 must be configured together"
            )
        if self.grounding_manifest_path is not None:
            _absolute_path(
                self.grounding_manifest_path,
                field_name=(
                    "post_training_internal_evaluation.grounding_manifest_path"
                ),
            )
            _sha256(
                self.grounding_manifest_sha256,
                field_name=(
                    "post_training_internal_evaluation.grounding_manifest_sha256"
                ),
            )
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError(
                "post_training_internal_evaluation.random_seed must be a non-negative integer"
            )
        _positive_int(
            self.max_new_tokens,
            field_name="post_training_internal_evaluation.max_new_tokens",
        )
        if (
            not isinstance(self.eos_token_ids, tuple)
            or not self.eos_token_ids
            or len(set(self.eos_token_ids)) != len(self.eos_token_ids)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in self.eos_token_ids
            )
        ):
            raise ValueError(
                "post_training_internal_evaluation.eos_token_ids must be unique non-negative integers"
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
    code_compatibility: str = NO_RESUME_CODE_COMPATIBILITY
    compatible_live_dirty_state_sha256: str | None = None

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
        if self.code_compatibility == NO_RESUME_CODE_COMPATIBILITY:
            if self.compatible_live_dirty_state_sha256 is not None:
                raise ValueError(
                    "resume compatible live-code SHA requires explicit compatibility"
                )
        elif self.code_compatibility == VALIDATED_NON_TRAINING_CODE_TRANSITION:
            if not self.enabled:
                raise ValueError("resume code compatibility requires enabled resume")
            _sha256(
                self.compatible_live_dirty_state_sha256,
                field_name="resume.compatible_live_dirty_state_sha256",
            )
        else:
            raise ValueError("unsupported resume.code_compatibility")


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


_PUBLIC_CONFIG_MODULE = "tgvf_rl.representation.training.config"
_CONFIG_SCHEMA_TYPES = (
    RepresentationCodeConfig,
    RepresentationModelConfig,
    RepresentationDataSplitConfig,
    RepresentationDataConfig,
    RepresentationDataConfigV2,
    RepresentationObjectiveExecutionConfig,
    RepresentationObjectiveExecutionConfigV2,
    RepresentationObjectiveExecutionConfigV3,
    RepresentationAdamWConfig,
    RepresentationExecutionConfig,
    RepresentationInitializationConfig,
    RepresentationFSDP2TopologyConfig,
    RepresentationTrainingLoopConfig,
    RepresentationPostTrainingInternalEvaluationConfig,
    RepresentationOutputConfig,
    RepresentationResumeConfig,
    RepresentationCheckpointConfig,
)

for _schema_type in _CONFIG_SCHEMA_TYPES:
    rebind_public_class(
        _schema_type,
        implementation_module=__name__,
        public_module=_PUBLIC_CONFIG_MODULE,
    )
del _schema_type


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
    *(_config_type.__name__ for _config_type in _CONFIG_SCHEMA_TYPES),
]
