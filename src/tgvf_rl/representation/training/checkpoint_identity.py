"""Run, optimizer, scheduler, and Adapter identities for checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tgvf_rl.conditioning.base import TargetConditioningConfig
from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.objectives.base import spec_identity_sha256
from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function
from tgvf_rl.representation.adapter import TGVFAdapter, TGVFAdapterVariant

from .checkpoint_integrity import (
    _adapter_state_to_cpu,
    _finite_ratio,
    _integer,
    _non_empty_text,
    _non_negative_finite_float,
    _non_negative_int,
    _positive_finite_float,
    _positive_int,
    _qualified_type,
    _require_adapter,
    _runtime_bool,
    _runtime_float,
    _runtime_optional_bool,
    _sha256,
    _state_digest,
    _strictly_increasing_non_negative_ints,
    _validate_run_identity_contract,
)
from .objective import RepresentationObjectiveConfig
from .sampling import (
    SAMPLER_IDENTITY_SCHEMA_VERSION,
    SAMPLER_STATE_SCHEMA_VERSION,
    SameImageBatchSampler,
)
from .validation_identity import RepresentationValidationDataIdentity


REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION = "representation-run-identity-v2"
REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2 = (
    REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION
)
REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3 = "representation-run-identity-v3"
REPRESENTATION_ACCUMULATION_SCHEMA_VERSION = "representation-accumulation-v1"
REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2 = "representation-accumulation-v2"
REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION = "representation-adapter-contract-v1"
REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2 = "representation-adapter-contract-v2"
REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION = (
    "representation-optimizer-identity-v1"
)
REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION = (
    "representation-scheduler-identity-v1"
)
REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2 = (
    "representation-scheduler-identity-v2"
)
REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION = "representation-trainer-execution-v1"
REPRESENTATION_INITIALIZATION_SCHEMA_VERSION = "representation-initialization-v1"
REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION = "representation-sampler-contract-v1"

MATRIX_CE_GLOBAL_REDUCTION = "global_ce_numerator_over_valid_rows"
L_GEN_GLOBAL_REDUCTION = "global_sum_of_per_sample_mean_nll_over_sample_count"


@dataclass(frozen=True, slots=True)
class RepresentationAccumulationIdentity:
    """Exact logical-batch reduction and optimizer-boundary contract."""

    gradient_accumulation_steps: int
    data_parallel_world_size: int
    matrix_ce_reduction: str = MATRIX_CE_GLOBAL_REDUCTION
    l_gen_reduction: str = L_GEN_GLOBAL_REDUCTION
    checkpoint_at_optimizer_boundary: bool = True
    schema_version: str = REPRESENTATION_ACCUMULATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_common_fields()
        if self.schema_version != REPRESENTATION_ACCUMULATION_SCHEMA_VERSION:
            raise ValueError("representation accumulation schema mismatch")

    def _validate_common_fields(self) -> None:
        _positive_int(
            self.gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        _positive_int(
            self.data_parallel_world_size, field_name="data_parallel_world_size"
        )
        if self.matrix_ce_reduction != MATRIX_CE_GLOBAL_REDUCTION:
            raise ValueError("unsupported Matrix-CE accumulation reduction")
        if self.l_gen_reduction != L_GEN_GLOBAL_REDUCTION:
            raise ValueError("unsupported L_gen accumulation reduction")
        if self.checkpoint_at_optimizer_boundary is not True:
            raise ValueError(
                "representation checkpoints must be taken at optimizer boundaries"
            )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationAccumulationIdentityV2(RepresentationAccumulationIdentity):
    """Exact optimizer batch identity for multiple direct groups per rank."""

    groups_per_rank_per_optimizer_step: int
    schema_version: str = REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        RepresentationAccumulationIdentity._validate_common_fields(self)
        _positive_int(
            self.groups_per_rank_per_optimizer_step,
            field_name="groups_per_rank_per_optimizer_step",
        )
        if self.groups_per_rank_per_optimizer_step <= 1:
            raise ValueError(
                "representation accumulation v2 requires more than one direct "
                "group per rank per optimizer step"
            )
        if (
            self.groups_per_rank_per_optimizer_step % self.gradient_accumulation_steps
            != 0
        ):
            raise ValueError(
                "groups_per_rank_per_optimizer_step must be evenly divisible by "
                "gradient_accumulation_steps"
            )
        if self.groups_per_accumulation_microstep <= 1:
            raise ValueError(
                "representation accumulation v2 requires more than one direct "
                "group per accumulation microstep"
            )
        if self.schema_version != REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2:
            raise ValueError("representation accumulation v2 schema mismatch")

    @property
    def groups_per_accumulation_microstep(self) -> int:
        return (
            self.groups_per_rank_per_optimizer_step // self.gradient_accumulation_steps
        )


def _validate_accumulation_identity(identity: object) -> None:
    if type(identity) is RepresentationAccumulationIdentity:
        expected_schema_version = REPRESENTATION_ACCUMULATION_SCHEMA_VERSION
    elif type(identity) is RepresentationAccumulationIdentityV2:
        expected_schema_version = REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2
    else:
        raise TypeError("unsupported representation accumulation identity type")
    if identity.schema_version != expected_schema_version:
        raise ValueError("representation accumulation identity schema mismatch")
    identity.__post_init__()


@dataclass(frozen=True, slots=True)
class RepresentationOptimizerIdentity:
    """Initial AdamW hyperparameters, including resolved torch options."""

    optimizer_type: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: bool | None
    capturable: bool
    differentiable: bool
    fused: bool | None
    decoupled_weight_decay: bool
    schema_version: str = REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_text(self.optimizer_type, field_name="optimizer_type")
        _positive_finite_float(self.learning_rate, field_name="learning_rate")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(not isinstance(value, float) for value in self.betas)
            or not 0 <= self.betas[0] < 1
            or not 0 <= self.betas[1] < 1
        ):
            raise ValueError("optimizer betas must be an explicit float pair in [0,1)")
        _positive_finite_float(self.eps, field_name="eps")
        _non_negative_finite_float(self.weight_decay, field_name="weight_decay")
        for field_name in (
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "decoupled_weight_decay",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        for field_name in ("foreach", "fused"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be bool or None")
        if not self.decoupled_weight_decay:
            raise ValueError("representation AdamW must use decoupled weight decay")
        if self.schema_version != REPRESENTATION_OPTIMIZER_IDENTITY_SCHEMA_VERSION:
            raise ValueError("representation optimizer identity schema mismatch")

    @classmethod
    def from_optimizer(
        cls, optimizer: torch.optim.Optimizer
    ) -> RepresentationOptimizerIdentity:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("representation optimizer identity requires AdamW")
        defaults = optimizer.defaults
        if len(optimizer.param_groups) != 1:
            raise ValueError(
                "representation optimizer identity v1 requires one parameter group"
            )
        required = {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        }
        if not required <= set(defaults):
            raise ValueError(
                "AdamW defaults do not expose the required hyperparameters"
            )
        betas = defaults["betas"]
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError("AdamW defaults contain malformed betas")
        group = optimizer.param_groups[0]
        for option in required - {"lr"}:
            if group.get(option) != defaults[option]:
                raise ValueError(
                    f"AdamW parameter-group {option} differs from optimizer defaults"
                )
        group_initial_lr = group.get("initial_lr", group.get("lr"))
        if group_initial_lr != defaults["lr"]:
            raise ValueError(
                "AdamW parameter-group initial lr differs from optimizer defaults"
            )
        return cls(
            optimizer_type=_qualified_type(optimizer),
            learning_rate=_runtime_float(defaults["lr"], field_name="lr"),
            betas=(
                _runtime_float(betas[0], field_name="beta1"),
                _runtime_float(betas[1], field_name="beta2"),
            ),
            eps=_runtime_float(defaults["eps"], field_name="eps"),
            weight_decay=_runtime_float(
                defaults["weight_decay"], field_name="weight_decay"
            ),
            amsgrad=_runtime_bool(defaults["amsgrad"], field_name="amsgrad"),
            maximize=_runtime_bool(defaults["maximize"], field_name="maximize"),
            foreach=_runtime_optional_bool(defaults["foreach"], field_name="foreach"),
            capturable=_runtime_bool(defaults["capturable"], field_name="capturable"),
            differentiable=_runtime_bool(
                defaults["differentiable"], field_name="differentiable"
            ),
            fused=_runtime_optional_bool(defaults["fused"], field_name="fused"),
            # Torch versions before this explicit option still implement AdamW
            # with decoupled decay; absence therefore resolves to True.
            decoupled_weight_decay=_runtime_bool(
                defaults.get("decoupled_weight_decay", True),
                field_name="decoupled_weight_decay",
            ),
        )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationSchedulerIdentity:
    """Project LambdaLR construction that cannot be recovered from state_dict."""

    scheduler_type: str
    kind: str
    total_steps: int
    warmup_steps: int
    schema_version: str = REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_text(self.scheduler_type, field_name="scheduler_type")
        if self.kind not in {"constant", "linear_warmup_decay"}:
            raise ValueError("unsupported representation scheduler kind")
        _positive_int(self.total_steps, field_name="total_steps")
        _non_negative_int(self.warmup_steps, field_name="warmup_steps")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("scheduler warmup_steps must be smaller than total_steps")
        if self.kind == "constant" and self.warmup_steps:
            raise ValueError("constant scheduler cannot have warmup steps")
        if self.schema_version != REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION:
            raise ValueError("representation scheduler identity schema mismatch")

    @classmethod
    def project_lambda(
        cls, *, kind: str, total_steps: int, warmup_steps: int
    ) -> RepresentationSchedulerIdentity:
        return cls(
            scheduler_type=("torch.optim.lr_scheduler.LambdaLR"),
            kind=kind,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

    @classmethod
    def from_config(cls, config: object) -> RepresentationSchedulerIdentity:
        kind = getattr(config, "kind", None)
        kind_value = getattr(kind, "value", kind)
        min_lr_ratio = getattr(config, "min_lr_ratio", None)
        if min_lr_ratio is not None or kind_value == "historical_cosine":
            return RepresentationSchedulerIdentityV2.project_lambda(
                kind=kind_value,
                total_steps=getattr(config, "total_steps", None),
                warmup_steps=getattr(config, "warmup_steps", None),
                min_lr_ratio=min_lr_ratio,
            )
        return cls.project_lambda(
            kind=kind_value,
            total_steps=getattr(config, "total_steps", None),
            warmup_steps=getattr(config, "warmup_steps", None),
        )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationSchedulerIdentityV2(RepresentationSchedulerIdentity):
    """Scheduler identity that binds the historical cosine minimum ratio."""

    min_lr_ratio: float
    schema_version: str = REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _non_empty_text(self.scheduler_type, field_name="scheduler_type")
        if self.kind != "historical_cosine":
            raise ValueError("scheduler identity v2 requires historical_cosine")
        _positive_int(self.total_steps, field_name="total_steps")
        _non_negative_int(self.warmup_steps, field_name="warmup_steps")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("scheduler warmup_steps must be smaller than total_steps")
        _finite_ratio(self.min_lr_ratio, field_name="min_lr_ratio")
        if self.schema_version != REPRESENTATION_SCHEDULER_IDENTITY_SCHEMA_VERSION_V2:
            raise ValueError("representation scheduler identity v2 schema mismatch")

    @classmethod
    def project_lambda(
        cls,
        *,
        kind: str,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> RepresentationSchedulerIdentityV2:
        return cls(
            scheduler_type="torch.optim.lr_scheduler.LambdaLR",
            kind=kind,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )


@dataclass(frozen=True, slots=True)
class RepresentationTrainerExecutionIdentity:
    """Precision and gradient-clipping semantics used by every update."""

    precision: str
    max_grad_norm: float
    require_all_adapter_gradients: bool
    gradient_clip_norm_type: float = 2.0
    gradient_clip_error_if_nonfinite: bool = True
    schema_version: str = REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("representation precision must be fp32 or bf16")
        _positive_finite_float(self.max_grad_norm, field_name="max_grad_norm")
        if not isinstance(self.require_all_adapter_gradients, bool):
            raise TypeError("require_all_adapter_gradients must be bool")
        if self.gradient_clip_norm_type != 2.0:
            raise ValueError("representation trainer requires L2 gradient clipping")
        if self.gradient_clip_error_if_nonfinite is not True:
            raise ValueError("gradient clipping must fail on non-finite norms")
        if self.schema_version != REPRESENTATION_TRAINER_EXECUTION_SCHEMA_VERSION:
            raise ValueError("representation trainer execution schema mismatch")

    @classmethod
    def from_config(cls, config: object) -> RepresentationTrainerExecutionIdentity:
        precision = getattr(config, "precision", None)
        precision_value = getattr(precision, "value", precision)
        return cls(
            precision=precision_value,
            max_grad_norm=getattr(config, "max_grad_norm", None),
            require_all_adapter_gradients=getattr(
                config, "require_all_adapter_gradients", None
            ),
        )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationInitializationIdentity:
    """Original Adapter initialization, distinct from a resume checkpoint."""

    kind: str
    seed: int
    initial_adapter_state_sha256: str
    source_artifact_sha256: str | None
    schema_version: str = REPRESENTATION_INITIALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in {"fresh_random", "adapter_artifact"}:
            raise ValueError("unsupported representation initialization kind")
        _integer(self.seed, field_name="initialization seed")
        _sha256(
            self.initial_adapter_state_sha256,
            field_name="initial_adapter_state_sha256",
        )
        if self.kind == "fresh_random":
            if self.source_artifact_sha256 is not None:
                raise ValueError("fresh initialization cannot name a source artifact")
        else:
            _sha256(
                self.source_artifact_sha256,
                field_name="source_artifact_sha256",
            )
        if self.schema_version != REPRESENTATION_INITIALIZATION_SCHEMA_VERSION:
            raise ValueError("representation initialization schema mismatch")

    @classmethod
    def from_adapter(
        cls,
        adapter: TGVFAdapter,
        *,
        kind: str,
        seed: int,
        source_artifact_sha256: str | None,
    ) -> RepresentationInitializationIdentity:
        return cls(
            kind=kind,
            seed=seed,
            initial_adapter_state_sha256=_state_digest(_adapter_state_to_cpu(adapter)),
            source_artifact_sha256=source_artifact_sha256,
        )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationSamplerContractIdentity:
    """Rank-independent same-image grouping and deterministic shuffle contract."""

    batch_size: int
    seed: int
    world_size: int
    data_manifest_sha256: str
    sampler_identity_schema_version: str = SAMPLER_IDENTITY_SCHEMA_VERSION
    sampler_state_schema_version: str = SAMPLER_STATE_SCHEMA_VERSION
    schema_version: str = REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _positive_int(self.batch_size, field_name="sampler batch_size")
        if self.batch_size < 2:
            raise ValueError("same-image Matrix CE requires sampler batch_size >= 2")
        _integer(self.seed, field_name="sampler seed")
        _positive_int(self.world_size, field_name="sampler world_size")
        _sha256(self.data_manifest_sha256, field_name="data_manifest_sha256")
        if self.sampler_identity_schema_version != SAMPLER_IDENTITY_SCHEMA_VERSION:
            raise ValueError("same-image sampler identity schema mismatch")
        if self.sampler_state_schema_version != SAMPLER_STATE_SCHEMA_VERSION:
            raise ValueError("same-image sampler state schema mismatch")
        if self.schema_version != REPRESENTATION_SAMPLER_CONTRACT_SCHEMA_VERSION:
            raise ValueError("representation sampler contract schema mismatch")

    @classmethod
    def from_sampler(
        cls, sampler: SameImageBatchSampler
    ) -> RepresentationSamplerContractIdentity:
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("sampler must be a SameImageBatchSampler")
        return cls(
            batch_size=sampler.batch_size,
            seed=sampler.seed,
            world_size=sampler.world_size,
            data_manifest_sha256=sampler.data_manifest_sha256,
        )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationAdapterContractIdentity:
    """Architecture plus borrowed-Qwen projection identities required to load."""

    d_lm: int
    d_v: int
    attention_dim: int
    spatial_merge_size: int
    deepstack_branch_layers: tuple[int, ...]
    main_projection_identity: str
    deepstack_projection_identities: tuple[str, ...]
    schema_version: str = REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("d_lm", "d_v", "attention_dim", "spatial_merge_size"):
            _positive_int(getattr(self, field_name), field_name=field_name)
        _strictly_increasing_non_negative_ints(
            self.deepstack_branch_layers, field_name="deepstack_branch_layers"
        )
        _non_empty_text(
            self.main_projection_identity, field_name="main_projection_identity"
        )
        if len(self.deepstack_projection_identities) != len(
            self.deepstack_branch_layers
        ):
            raise ValueError(
                "D-DeepStack projection identities must align with branch layers"
            )
        for identity in self.deepstack_projection_identities:
            _non_empty_text(identity, field_name="deepstack_projection_identity")
        all_projection_identities = (
            self.main_projection_identity,
            *self.deepstack_projection_identities,
        )
        if len(set(all_projection_identities)) != len(all_projection_identities):
            raise ValueError(
                "main and D-DeepStack projection identities must be unique"
            )
        if self.schema_version != REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION:
            raise ValueError("representation Adapter contract schema mismatch")

    @classmethod
    def from_adapter(
        cls, adapter: TGVFAdapter
    ) -> RepresentationAdapterContractIdentity:
        _require_adapter(adapter)
        if adapter.variant is not TGVFAdapterVariant.FULL_D_DEEPSTACK:
            raise ValueError(
                "adapter contract v1 can identify only the full D-DeepStack variant"
            )
        return cls(
            d_lm=adapter.d_lm,
            d_v=adapter.d_v,
            attention_dim=adapter.attn_dim,
            spatial_merge_size=adapter.spatial_merge_size,
            deepstack_branch_layers=tuple(adapter.d_deepstack_branch_layers),
            main_projection_identity=adapter.main_output_projection_identity,
            deepstack_projection_identities=(
                adapter.deepstack_output_projection_identities
            ),
        )

    def assert_matches(self, adapter: TGVFAdapter) -> None:
        if self != type(self).from_adapter(adapter):
            raise IdentityMismatchError(
                "TGVF Adapter architecture/projection identity mismatch"
            )

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationAdapterContractIdentityV2(RepresentationAdapterContractIdentity):
    """Adapter contract that content-binds the selected structural variant."""

    variant: str
    schema_version: str = REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        for field_name in ("d_lm", "d_v", "attention_dim", "spatial_merge_size"):
            _positive_int(getattr(self, field_name), field_name=field_name)
        _strictly_increasing_non_negative_ints(
            self.deepstack_branch_layers, field_name="deepstack_branch_layers"
        )
        _non_empty_text(
            self.main_projection_identity, field_name="main_projection_identity"
        )
        if len(self.deepstack_projection_identities) != len(
            self.deepstack_branch_layers
        ):
            raise ValueError(
                "D-DeepStack projection identities must align with branch layers"
            )
        for identity in self.deepstack_projection_identities:
            _non_empty_text(identity, field_name="deepstack_projection_identity")
        all_projection_identities = (
            self.main_projection_identity,
            *self.deepstack_projection_identities,
        )
        if len(set(all_projection_identities)) != len(all_projection_identities):
            raise ValueError(
                "main and D-DeepStack projection identities must be unique"
            )
        if self.variant not in {variant.value for variant in TGVFAdapterVariant}:
            raise ValueError("unknown TGVF Adapter structural variant")
        if self.schema_version != REPRESENTATION_ADAPTER_CONTRACT_SCHEMA_VERSION_V2:
            raise ValueError("representation Adapter contract v2 schema mismatch")

    @classmethod
    def from_adapter(
        cls, adapter: TGVFAdapter
    ) -> RepresentationAdapterContractIdentityV2:
        _require_adapter(adapter)
        return cls(
            d_lm=adapter.d_lm,
            d_v=adapter.d_v,
            attention_dim=adapter.attn_dim,
            spatial_merge_size=adapter.spatial_merge_size,
            deepstack_branch_layers=tuple(adapter.d_deepstack_branch_layers),
            main_projection_identity=adapter.main_output_projection_identity,
            deepstack_projection_identities=(
                adapter.deepstack_output_projection_identities
            ),
            variant=adapter.variant.value,
        )

    def assert_matches(self, adapter: TGVFAdapter) -> None:
        if self != type(self).from_adapter(adapter):
            raise IdentityMismatchError(
                "TGVF Adapter architecture/projection/variant identity mismatch"
            )


def representation_adapter_contract_identity(
    adapter: TGVFAdapter,
) -> RepresentationAdapterContractIdentity:
    """Keep existing full artifacts v1-compatible; use v2 for new variants."""

    _require_adapter(adapter)
    if adapter.variant is TGVFAdapterVariant.FULL_D_DEEPSTACK:
        return RepresentationAdapterContractIdentity.from_adapter(adapter)
    return RepresentationAdapterContractIdentityV2.from_adapter(adapter)


@dataclass(frozen=True, slots=True)
class RepresentationRunIdentity:
    """Complete immutable identity shared by artifact and training checkpoint."""

    run_id: str
    code: CodeIdentity
    model: ModelIdentity
    provider: TargetConditioningConfig
    data_manifest_sha256: str
    prompt_sha256: str
    objective: RepresentationObjectiveConfig
    adapter_contract: RepresentationAdapterContractIdentity
    accumulation: RepresentationAccumulationIdentity
    optimizer: RepresentationOptimizerIdentity
    scheduler: RepresentationSchedulerIdentity | None
    trainer_execution: RepresentationTrainerExecutionIdentity
    initialization: RepresentationInitializationIdentity
    sampler_contract: RepresentationSamplerContractIdentity
    schema_version: str = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_common_fields()
        if self.schema_version != REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2:
            raise ValueError("representation run identity v2 schema mismatch")

    def _validate_common_fields(self) -> None:
        _non_empty_text(self.run_id, field_name="run_id")
        if not isinstance(self.code, CodeIdentity):
            raise TypeError("code must be a CodeIdentity")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        if not isinstance(self.provider, TargetConditioningConfig):
            raise TypeError("provider must be a TargetConditioningConfig")
        _sha256(self.data_manifest_sha256, field_name="data_manifest_sha256")
        _sha256(self.prompt_sha256, field_name="prompt_sha256")
        if not isinstance(self.objective, RepresentationObjectiveConfig):
            raise TypeError("objective must be a RepresentationObjectiveConfig")
        if not isinstance(self.adapter_contract, RepresentationAdapterContractIdentity):
            raise TypeError(
                "adapter_contract must be a RepresentationAdapterContractIdentity"
            )
        _validate_accumulation_identity(self.accumulation)
        if not isinstance(self.optimizer, RepresentationOptimizerIdentity):
            raise TypeError("optimizer must be a RepresentationOptimizerIdentity")
        if self.scheduler is not None and not isinstance(
            self.scheduler, RepresentationSchedulerIdentity
        ):
            raise TypeError(
                "scheduler must be a RepresentationSchedulerIdentity or None"
            )
        if not isinstance(
            self.trainer_execution, RepresentationTrainerExecutionIdentity
        ):
            raise TypeError(
                "trainer_execution must be a RepresentationTrainerExecutionIdentity"
            )
        if not isinstance(self.initialization, RepresentationInitializationIdentity):
            raise TypeError(
                "initialization must be a RepresentationInitializationIdentity"
            )
        if not isinstance(self.sampler_contract, RepresentationSamplerContractIdentity):
            raise TypeError(
                "sampler_contract must be a RepresentationSamplerContractIdentity"
            )
        if self.sampler_contract.data_manifest_sha256 != self.data_manifest_sha256:
            raise ValueError("sampler contract data manifest differs from run identity")
        if (
            self.sampler_contract.world_size
            != self.accumulation.data_parallel_world_size
        ):
            raise ValueError("sampler and accumulation world sizes must match")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)

    @property
    def provider_identity_sha256(self) -> str:
        return spec_identity_sha256(self.provider)

    @property
    def model_identity_sha256(self) -> str:
        return spec_identity_sha256(self.model)

    @property
    def objective_identity_sha256(self) -> str:
        return spec_identity_sha256(self.objective)

    @property
    def scheduler_identity_sha256(self) -> str | None:
        return None if self.scheduler is None else self.scheduler.identity_sha256


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationRunIdentityV3(RepresentationRunIdentity):
    """Run identity that also freezes validation data and the planned horizon."""

    validation_identity: RepresentationValidationDataIdentity
    planned_target_optimizer_steps: int
    schema_version: str = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3

    def __post_init__(self) -> None:
        RepresentationRunIdentity._validate_common_fields(self)
        if not isinstance(
            self.validation_identity,
            RepresentationValidationDataIdentity,
        ):
            raise TypeError(
                "validation_identity must be a RepresentationValidationDataIdentity"
            )
        self.validation_identity.__post_init__()
        if (
            self.validation_identity.train_retained_manifest_sha256
            != self.data_manifest_sha256
        ):
            raise ValueError(
                "validation identity train manifest differs from run identity"
            )
        _positive_int(
            self.planned_target_optimizer_steps,
            field_name="planned_target_optimizer_steps",
        )
        if (
            self.scheduler is not None
            and self.planned_target_optimizer_steps > self.scheduler.total_steps
        ):
            raise ValueError("planned optimizer steps exceed the scheduler horizon")
        if self.schema_version != REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3:
            raise ValueError("representation run identity v3 schema mismatch")


def _validate_run_identity(identity: object) -> None:
    _validate_run_identity_contract(
        identity,
        run_identity_type=RepresentationRunIdentity,
        run_identity_v3_type=RepresentationRunIdentityV3,
        schema_version_v2=REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2,
        schema_version_v3=REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3,
        validate_accumulation=_validate_accumulation_identity,
    )


_CHECKPOINT_IDENTITY_TYPES = (
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
    RepresentationOptimizerIdentity,
    RepresentationSchedulerIdentity,
    RepresentationSchedulerIdentityV2,
    RepresentationTrainerExecutionIdentity,
    RepresentationInitializationIdentity,
    RepresentationSamplerContractIdentity,
    RepresentationAdapterContractIdentity,
    RepresentationAdapterContractIdentityV2,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3,
)

for _identity_type in _CHECKPOINT_IDENTITY_TYPES:
    rebind_public_class(
        _identity_type,
        implementation_module=__name__,
        public_module="tgvf_rl.representation.training.checkpoint",
    )
del _identity_type

for _identity_function in (
    _validate_accumulation_identity,
    representation_adapter_contract_identity,
    _validate_run_identity,
):
    rebind_public_function(
        _identity_function,
        implementation_module=__name__,
        public_module="tgvf_rl.representation.training.checkpoint",
    )
del _identity_function
