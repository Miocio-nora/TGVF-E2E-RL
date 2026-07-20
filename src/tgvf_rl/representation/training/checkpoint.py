"""Strict artifacts and resumable checkpoints for the representation phase.

The deployable artifact contains only tensors owned by :class:`TGVFAdapter`.
Frozen Qwen projection ports are bound by identity but are never serialized.
Training checkpoints are deliberately optimizer-step-boundary checkpoints:
there are no partially accumulated gradients whose identity could be lost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Protocol

import torch

from tgvf_rl.conditioning.base import TargetConditioningConfig
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.objectives.base import spec_identity_sha256
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.representation.adapter import TGVFAdapter, TGVFAdapterVariant

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
REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION = "representation-adapter-artifact-v2"
REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION = (
    "representation-training-checkpoint-v2"
)
REPRESENTATION_RNG_STATE_SCHEMA_VERSION = "representation-rng-state-v2"

MATRIX_CE_GLOBAL_REDUCTION = "global_ce_numerator_over_valid_rows"
L_GEN_GLOBAL_REDUCTION = "global_sum_of_per_sample_mean_nll_over_sample_count"

_BORROWED_QWEN_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)
_HEX = frozenset("0123456789abcdef")


class _Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> object: ...


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
            main_projection_identity=adapter.main_projection.identity,
            deepstack_projection_identities=tuple(
                port.identity for port in adapter.d_deepstack_projections.projections
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
            main_projection_identity=adapter.main_projection.identity,
            deepstack_projection_identities=tuple(
                port.identity for port in adapter.d_deepstack_projections.projections
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


@dataclass(frozen=True, slots=True)
class RepresentationTensorManifestEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        _non_empty_text(self.name, field_name="tensor name")
        if not self.shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ValueError("tensor shape must contain non-negative integer sizes")
        _non_empty_text(self.dtype, field_name="tensor dtype")
        _sha256(self.tensor_sha256, field_name="tensor_sha256")


@dataclass(frozen=True, slots=True)
class RepresentationAdapterArtifactManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    adapter_state_sha256: str
    tensors: tuple[RepresentationTensorManifestEntry, ...]
    schema_version: str = REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("artifact run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _sha256(self.adapter_state_sha256, field_name="adapter_state_sha256")
        _validate_tensor_manifest(self.tensors)
        if self.schema_version != REPRESENTATION_ADAPTER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("representation Adapter artifact schema mismatch")

    @property
    def artifact_identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationAdapterArtifact:
    manifest: RepresentationAdapterArtifactManifest
    adapter_state: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class RepresentationTrainingCheckpointManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    accumulation_microstep: int
    adapter_state_sha256: str
    adapter_tensors: tuple[RepresentationTensorManifestEntry, ...]
    optimizer_type: str
    optimizer_parameter_names_by_group: tuple[tuple[str, ...], ...]
    optimizer_identity_sha256: str
    optimizer_state_sha256: str
    scheduler_type: str | None
    scheduler_identity_sha256: str | None
    scheduler_state_sha256: str | None
    sampler_type: str
    sampler_contract_identity_sha256: str
    sampler_identity_sha256: str
    sampler_state_sha256: str
    trainer_execution_identity_sha256: str
    initialization_identity_sha256: str
    rng_state_sha256: str
    schema_version: str = REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("checkpoint run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        if self.accumulation_microstep != 0:
            raise ValueError(
                "representation checkpoints cannot contain partial accumulation"
            )
        _sha256(self.adapter_state_sha256, field_name="adapter_state_sha256")
        _validate_tensor_manifest(self.adapter_tensors)
        _non_empty_text(self.optimizer_type, field_name="optimizer_type")
        if not self.optimizer_parameter_names_by_group or any(
            not group for group in self.optimizer_parameter_names_by_group
        ):
            raise ValueError("optimizer parameter-name groups must be non-empty")
        flattened = tuple(
            name for group in self.optimizer_parameter_names_by_group for name in group
        )
        if any(not isinstance(name, str) or not name for name in flattened):
            raise ValueError("optimizer parameter names must be non-empty strings")
        if len(flattened) != len(set(flattened)):
            raise ValueError("optimizer parameters must occur exactly once")
        _sha256(self.optimizer_identity_sha256, field_name="optimizer_identity_sha256")
        if (
            self.optimizer_identity_sha256
            != self.run_identity.optimizer.identity_sha256
        ):
            raise ValueError("checkpoint optimizer identity digest mismatch")
        _sha256(self.optimizer_state_sha256, field_name="optimizer_state_sha256")
        scheduler_fields = (
            self.scheduler_type,
            self.scheduler_identity_sha256,
            self.scheduler_state_sha256,
        )
        if any(value is None for value in scheduler_fields) != all(
            value is None for value in scheduler_fields
        ):
            raise ValueError("scheduler type, identity, and state presence must align")
        if self.scheduler_type is not None:
            _non_empty_text(self.scheduler_type, field_name="scheduler_type")
            _sha256(
                self.scheduler_identity_sha256,
                field_name="scheduler_identity_sha256",
            )
            _sha256(
                self.scheduler_state_sha256,
                field_name="scheduler_state_sha256",
            )
        if (
            self.scheduler_identity_sha256
            != self.run_identity.scheduler_identity_sha256
        ):
            raise ValueError("checkpoint scheduler identity digest mismatch")
        _non_empty_text(self.sampler_type, field_name="sampler_type")
        _sha256(
            self.sampler_contract_identity_sha256,
            field_name="sampler_contract_identity_sha256",
        )
        if (
            self.sampler_contract_identity_sha256
            != self.run_identity.sampler_contract.identity_sha256
        ):
            raise ValueError("checkpoint sampler contract identity digest mismatch")
        _sha256(self.sampler_identity_sha256, field_name="sampler_identity_sha256")
        _sha256(self.sampler_state_sha256, field_name="sampler_state_sha256")
        _sha256(
            self.trainer_execution_identity_sha256,
            field_name="trainer_execution_identity_sha256",
        )
        if (
            self.trainer_execution_identity_sha256
            != self.run_identity.trainer_execution.identity_sha256
        ):
            raise ValueError("checkpoint trainer execution identity digest mismatch")
        _sha256(
            self.initialization_identity_sha256,
            field_name="initialization_identity_sha256",
        )
        if (
            self.initialization_identity_sha256
            != self.run_identity.initialization.identity_sha256
        ):
            raise ValueError("checkpoint initialization identity digest mismatch")
        _sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if self.schema_version != REPRESENTATION_TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("representation training checkpoint schema mismatch")

    @property
    def checkpoint_identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class RepresentationTrainingCheckpoint:
    manifest: RepresentationTrainingCheckpointManifest
    adapter_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    scheduler_state: dict[str, object] | None
    sampler_state: dict[str, object]
    rng_state: dict[str, object]


@dataclass(frozen=True, slots=True)
class RepresentationResumeResult:
    global_step: int
    next_global_step: int
    run_identity_sha256: str
    checkpoint_identity_sha256: str
    exact: bool = True


def save_representation_adapter_artifact_atomic(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    run_identity: RepresentationRunIdentity,
    global_step: int,
) -> RepresentationAdapterArtifactManifest:
    """Atomically write a deployable Adapter-only artifact."""

    _validate_runtime_identity(run_identity, adapter=adapter)
    _non_negative_int(global_step, field_name="global_step")
    _validate_initial_state_at_step_zero(run_identity, adapter, global_step)
    adapter_state = _adapter_state_to_cpu(adapter)
    manifest = RepresentationAdapterArtifactManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=global_step,
        adapter_state_sha256=_state_digest(adapter_state),
        tensors=_tensor_manifest(adapter_state),
    )
    artifact = RepresentationAdapterArtifact(manifest, adapter_state)
    _validate_adapter_artifact(artifact)
    _save_atomic(artifact, path)
    return manifest


def load_representation_adapter_artifact(
    path: str | Path,
) -> RepresentationAdapterArtifact:
    value = _torch_load(path)
    if not isinstance(value, RepresentationAdapterArtifact):
        raise ReplayMismatchError("file is not a representation Adapter artifact")
    _validate_adapter_artifact(value)
    return value


def restore_representation_adapter_artifact(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    expected_run_identity: RepresentationRunIdentity,
) -> RepresentationAdapterArtifactManifest:
    """Strictly restore owned tensors while leaving Qwen projection state intact."""

    _validate_runtime_identity(expected_run_identity, adapter=adapter)
    artifact = load_representation_adapter_artifact(path)
    _assert_same_run_identity(artifact.manifest.run_identity, expected_run_identity)
    adapter.load_artifact_state_dict(artifact.adapter_state)
    return artifact.manifest


def save_representation_training_checkpoint_atomic(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
    global_step: int,
) -> RepresentationTrainingCheckpointManifest:
    """Save exact training state at an optimizer-step boundary."""

    _validate_runtime_identity(
        run_identity,
        adapter=adapter,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=accumulation,
        trainer_execution=trainer_execution,
    )
    _non_negative_int(global_step, field_name="global_step")
    _validate_initial_state_at_step_zero(run_identity, adapter, global_step)
    parameter_names = _optimizer_parameter_names_by_group(adapter, optimizer)
    _validate_scheduler(scheduler, optimizer=optimizer)

    adapter_state = _adapter_state_to_cpu(adapter)
    optimizer_state = _plain_cpu_state(optimizer.state_dict())
    scheduler_state = (
        None if scheduler is None else _plain_cpu_state(scheduler.state_dict())
    )
    sampler_state = _plain_cpu_state(sampler.state_dict())
    rng_state = capture_representation_rng_state()
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("optimizer state_dict must be a mapping")
    if scheduler_state is not None and not isinstance(scheduler_state, dict):
        raise RuntimeError("scheduler state_dict must be a mapping")
    if not isinstance(sampler_state, dict):
        raise RuntimeError("sampler state_dict must be a mapping")

    manifest = RepresentationTrainingCheckpointManifest(
        run_identity=run_identity,
        run_identity_sha256=run_identity.identity_sha256,
        global_step=global_step,
        accumulation_microstep=0,
        adapter_state_sha256=_state_digest(adapter_state),
        adapter_tensors=_tensor_manifest(adapter_state),
        optimizer_type=_qualified_type(optimizer),
        optimizer_parameter_names_by_group=parameter_names,
        optimizer_identity_sha256=run_identity.optimizer.identity_sha256,
        optimizer_state_sha256=_state_digest(optimizer_state),
        scheduler_type=None if scheduler is None else _qualified_type(scheduler),
        scheduler_identity_sha256=run_identity.scheduler_identity_sha256,
        scheduler_state_sha256=(
            None if scheduler_state is None else _state_digest(scheduler_state)
        ),
        sampler_type=_qualified_type(sampler),
        sampler_contract_identity_sha256=(
            run_identity.sampler_contract.identity_sha256
        ),
        sampler_identity_sha256=sampler.identity_sha256,
        sampler_state_sha256=_state_digest(sampler_state),
        trainer_execution_identity_sha256=(
            run_identity.trainer_execution.identity_sha256
        ),
        initialization_identity_sha256=run_identity.initialization.identity_sha256,
        rng_state_sha256=_state_digest(rng_state),
    )
    checkpoint = RepresentationTrainingCheckpoint(
        manifest=manifest,
        adapter_state=adapter_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        sampler_state=sampler_state,
        rng_state=rng_state,
    )
    _validate_training_checkpoint(checkpoint)
    _save_atomic(checkpoint, path)
    return manifest


def load_representation_training_checkpoint(
    path: str | Path,
) -> RepresentationTrainingCheckpoint:
    value = _torch_load(path)
    if not isinstance(value, RepresentationTrainingCheckpoint):
        raise ReplayMismatchError("file is not a representation training checkpoint")
    _validate_training_checkpoint(value)
    return value


def restore_representation_training_checkpoint(
    path: str | Path,
    *,
    adapter: TGVFAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: _Stateful | None,
    sampler: SameImageBatchSampler,
    expected_run_identity: RepresentationRunIdentity,
    accumulation: RepresentationAccumulationIdentity,
    trainer_execution: RepresentationTrainerExecutionIdentity,
) -> RepresentationResumeResult:
    """Validate all state before restoring, rolling back on an apply failure."""

    _validate_runtime_identity(
        expected_run_identity,
        adapter=adapter,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=accumulation,
        trainer_execution=trainer_execution,
    )
    checkpoint = load_representation_training_checkpoint(path)
    manifest = checkpoint.manifest
    _assert_same_run_identity(manifest.run_identity, expected_run_identity)
    if manifest.optimizer_type != _qualified_type(optimizer):
        raise IdentityMismatchError("optimizer type differs from checkpoint")
    current_names = _optimizer_parameter_names_by_group(adapter, optimizer)
    if current_names != manifest.optimizer_parameter_names_by_group:
        raise IdentityMismatchError(
            "optimizer parameter-name topology differs from checkpoint"
        )
    _validate_optimizer_group_shape(optimizer, checkpoint.optimizer_state)
    _validate_scheduler(scheduler, optimizer=optimizer)
    actual_scheduler_type = None if scheduler is None else _qualified_type(scheduler)
    if actual_scheduler_type != manifest.scheduler_type:
        raise IdentityMismatchError("scheduler presence/type differs from checkpoint")
    if manifest.sampler_type != _qualified_type(sampler):
        raise IdentityMismatchError("sampler type differs from checkpoint")
    if manifest.sampler_identity_sha256 != sampler.identity_sha256:
        raise IdentityMismatchError("sampler identity differs from checkpoint")

    rollback_adapter = _adapter_state_to_cpu(adapter)
    rollback_optimizer = deepcopy(optimizer.state_dict())
    rollback_scheduler = None if scheduler is None else deepcopy(scheduler.state_dict())
    rollback_sampler = deepcopy(sampler.state_dict())
    rollback_rng = capture_representation_rng_state()
    try:
        adapter.load_artifact_state_dict(checkpoint.adapter_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        if scheduler is not None:
            if checkpoint.scheduler_state is None:
                raise ReplayMismatchError("checkpoint scheduler state is missing")
            scheduler.load_state_dict(checkpoint.scheduler_state)
        sampler.load_state_dict(checkpoint.sampler_state)
        restore_representation_rng_state(checkpoint.rng_state)
    except Exception:
        adapter.load_artifact_state_dict(rollback_adapter)
        optimizer.load_state_dict(rollback_optimizer)
        if scheduler is not None and rollback_scheduler is not None:
            scheduler.load_state_dict(rollback_scheduler)
        sampler.load_state_dict(rollback_sampler)
        restore_representation_rng_state(rollback_rng)
        raise

    return RepresentationResumeResult(
        global_step=manifest.global_step,
        next_global_step=manifest.global_step + 1,
        run_identity_sha256=manifest.run_identity_sha256,
        checkpoint_identity_sha256=manifest.checkpoint_identity_sha256,
    )


def capture_representation_rng_state() -> dict[str, object]:
    """Capture Python, CPU torch, and the current local CUDA generator."""

    state: dict[str, object] = {
        "schema_version": REPRESENTATION_RNG_STATE_SCHEMA_VERSION,
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state().cpu().clone(),
    }
    # A CPU-only save must not initialize CUDA merely to inspect its generators.
    if torch.cuda.is_initialized():
        device_index = torch.cuda.current_device()
        visible_device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= visible_device_count:
            raise RuntimeError(
                "current CUDA device index is outside the visible device topology"
            )
        state["torch_cuda"] = {
            "device_index": device_index,
            "visible_device_count": visible_device_count,
            "state": torch.cuda.get_rng_state(device_index).cpu().clone(),
        }
    return state


def restore_representation_rng_state(state: Mapping[str, object]) -> None:
    """Strictly restore the represented CPU and current local CUDA generators."""

    _validate_rng_state(state)
    python_state = state["python"]
    cpu_state = state["torch_cpu"]
    random.setstate(python_state)  # type: ignore[arg-type]
    torch.random.set_rng_state(cpu_state)  # type: ignore[arg-type]
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        torch.cuda.set_rng_state(
            cuda_state["state"],  # type: ignore[index]
            device=cuda_state["device_index"],  # type: ignore[index]
        )


def _validate_adapter_artifact(artifact: RepresentationAdapterArtifact) -> None:
    if not isinstance(artifact.manifest, RepresentationAdapterArtifactManifest):
        raise ReplayMismatchError("malformed representation artifact manifest")
    _validate_run_identity(artifact.manifest.run_identity)
    artifact.manifest.__post_init__()
    state = _validate_adapter_state(artifact.adapter_state)
    if _state_digest(state) != artifact.manifest.adapter_state_sha256:
        raise ReplayMismatchError("representation artifact state digest mismatch")
    if _tensor_manifest(state) != artifact.manifest.tensors:
        raise ReplayMismatchError("representation artifact tensor manifest mismatch")


def _validate_training_checkpoint(
    checkpoint: RepresentationTrainingCheckpoint,
) -> None:
    manifest = checkpoint.manifest
    if not isinstance(manifest, RepresentationTrainingCheckpointManifest):
        raise ReplayMismatchError("malformed representation checkpoint manifest")
    _validate_run_identity(manifest.run_identity)
    manifest.__post_init__()
    adapter_state = _validate_adapter_state(checkpoint.adapter_state)
    if _state_digest(adapter_state) != manifest.adapter_state_sha256:
        raise ReplayMismatchError("checkpoint Adapter state digest mismatch")
    if _tensor_manifest(adapter_state) != manifest.adapter_tensors:
        raise ReplayMismatchError("checkpoint Adapter tensor manifest mismatch")
    if not isinstance(checkpoint.optimizer_state, Mapping):
        raise ReplayMismatchError("checkpoint optimizer state is not a mapping")
    if _state_digest(checkpoint.optimizer_state) != manifest.optimizer_state_sha256:
        raise ReplayMismatchError("checkpoint optimizer state digest mismatch")
    if (checkpoint.scheduler_state is None) != (manifest.scheduler_type is None):
        raise ReplayMismatchError("checkpoint scheduler presence mismatch")
    if checkpoint.scheduler_state is not None:
        if not isinstance(checkpoint.scheduler_state, Mapping):
            raise ReplayMismatchError("checkpoint scheduler state is not a mapping")
        if _state_digest(checkpoint.scheduler_state) != manifest.scheduler_state_sha256:
            raise ReplayMismatchError("checkpoint scheduler state digest mismatch")
    _validate_global_step_state(
        manifest.global_step,
        optimizer_state=checkpoint.optimizer_state,
        scheduler_state=checkpoint.scheduler_state,
    )
    if not isinstance(checkpoint.sampler_state, Mapping):
        raise ReplayMismatchError("checkpoint sampler state is not a mapping")
    if _state_digest(checkpoint.sampler_state) != manifest.sampler_state_sha256:
        raise ReplayMismatchError("checkpoint sampler state digest mismatch")
    if not isinstance(checkpoint.rng_state, Mapping):
        raise ReplayMismatchError("checkpoint RNG state is not a mapping")
    _validate_rng_state(checkpoint.rng_state)
    if _state_digest(checkpoint.rng_state) != manifest.rng_state_sha256:
        raise ReplayMismatchError("checkpoint RNG state digest mismatch")


def _validate_runtime_identity(
    identity: RepresentationRunIdentity,
    *,
    adapter: TGVFAdapter,
    sampler: SameImageBatchSampler | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: _Stateful | None = None,
    accumulation: RepresentationAccumulationIdentity | None = None,
    trainer_execution: RepresentationTrainerExecutionIdentity | None = None,
) -> None:
    _validate_run_identity(identity)
    identity.adapter_contract.assert_matches(adapter)
    if sampler is not None:
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("sampler must be a SameImageBatchSampler")
        if sampler.data_manifest_sha256 != identity.data_manifest_sha256:
            raise IdentityMismatchError(
                "sampler data manifest differs from representation run identity"
            )
        if sampler.world_size != identity.accumulation.data_parallel_world_size:
            raise IdentityMismatchError(
                "sampler world size differs from accumulation identity"
            )
        if (
            RepresentationSamplerContractIdentity.from_sampler(sampler)
            != identity.sampler_contract
        ):
            raise IdentityMismatchError(
                "same-image sampler contract differs from run identity"
            )
    if optimizer is not None:
        if (
            RepresentationOptimizerIdentity.from_optimizer(optimizer)
            != identity.optimizer
        ):
            raise IdentityMismatchError(
                "optimizer hyperparameters differ from representation run identity"
            )
        _validate_scheduler(scheduler, optimizer=optimizer)
        _validate_scheduler_identity(scheduler, identity.scheduler)
    elif scheduler is not None:
        raise ValueError("scheduler validation requires its optimizer")
    if accumulation is not None:
        _validate_accumulation_identity(accumulation)
        if accumulation != identity.accumulation:
            raise IdentityMismatchError(
                "runtime accumulation differs from representation run identity"
            )
    if trainer_execution is not None:
        if not isinstance(trainer_execution, RepresentationTrainerExecutionIdentity):
            raise TypeError(
                "trainer_execution must be a RepresentationTrainerExecutionIdentity"
            )
        if trainer_execution != identity.trainer_execution:
            raise IdentityMismatchError(
                "precision/gradient clipping differs from representation run identity"
            )


def _validate_run_identity(identity: object) -> None:
    if not isinstance(identity, RepresentationRunIdentity):
        raise TypeError("run identity must be a RepresentationRunIdentity")
    schema_version = getattr(identity, "schema_version", None)
    if type(identity) is RepresentationRunIdentity:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V2
    elif type(identity) is RepresentationRunIdentityV3:
        expected_schema_version = REPRESENTATION_RUN_IDENTITY_SCHEMA_VERSION_V3
    else:
        raise TypeError("unsupported representation run identity type")
    if schema_version != expected_schema_version:
        raise ValueError("representation run identity schema mismatch")
    identity.code.__post_init__()
    identity.model.__post_init__()
    identity.provider.__post_init__()
    identity.objective.__post_init__()
    identity.adapter_contract.__post_init__()
    _validate_accumulation_identity(identity.accumulation)
    identity.optimizer.__post_init__()
    if identity.scheduler is not None:
        identity.scheduler.__post_init__()
    identity.trainer_execution.__post_init__()
    identity.initialization.__post_init__()
    identity.sampler_contract.__post_init__()
    if isinstance(identity, RepresentationRunIdentityV3):
        identity.validation_identity.__post_init__()
    identity.__post_init__()


def _validate_initial_state_at_step_zero(
    identity: RepresentationRunIdentity,
    adapter: TGVFAdapter,
    global_step: int,
) -> None:
    if global_step != 0:
        return
    current = _state_digest(_adapter_state_to_cpu(adapter))
    if current != identity.initialization.initial_adapter_state_sha256:
        raise IdentityMismatchError(
            "global-step-zero Adapter state differs from initialization identity"
        )


def _assert_same_run_identity(
    recorded: RepresentationRunIdentity,
    expected: RepresentationRunIdentity,
) -> None:
    if recorded.identity_sha256 != expected.identity_sha256 or recorded != expected:
        raise IdentityMismatchError(
            "representation checkpoint/artifact run identity mismatch"
        )


def _adapter_state_to_cpu(adapter: TGVFAdapter) -> dict[str, torch.Tensor]:
    _require_adapter(adapter)
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in adapter.artifact_state_dict().items()
    }


def _validate_adapter_state(
    state: object,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ReplayMismatchError("Adapter state must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ReplayMismatchError("Adapter state keys must be non-empty strings")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ReplayMismatchError("Adapter state values must all be tensors")
    if any(
        name.startswith(_BORROWED_QWEN_PREFIXES)  # type: ignore[union-attr]
        for name in state
    ):
        raise ReplayMismatchError(
            "borrowed Qwen projection state is forbidden in Adapter artifacts"
        )
    return state  # type: ignore[return-value]


def _tensor_manifest(
    state: Mapping[str, torch.Tensor],
) -> tuple[RepresentationTensorManifestEntry, ...]:
    return tuple(
        RepresentationTensorManifestEntry(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            tensor_sha256=_tensor_checksum(value),
        )
        for name, value in sorted(state.items())
    )


def _validate_tensor_manifest(
    entries: tuple[RepresentationTensorManifestEntry, ...],
) -> None:
    if not isinstance(entries, tuple) or not entries:
        raise ValueError("Adapter tensor manifest must be a non-empty tuple")
    for entry in entries:
        if not isinstance(entry, RepresentationTensorManifestEntry):
            raise TypeError("Adapter tensor manifest has an invalid entry")
        entry.__post_init__()
    names = tuple(entry.name for entry in entries)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("Adapter tensor manifest names must be unique and sorted")


def _optimizer_parameter_names_by_group(
    adapter: TGVFAdapter, optimizer: torch.optim.Optimizer
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    artifact_keys = set(adapter.artifact_state_dict())
    owned_parameters = {
        id(parameter): name
        for name, parameter in adapter.named_parameters()
        if name in artifact_keys and parameter.requires_grad
    }
    if not owned_parameters:
        raise ValueError("TGVF Adapter has no trainable owned parameters")
    groups: list[tuple[str, ...]] = []
    actual_ids: list[int] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            name = owned_parameters.get(parameter_id)
            if name is None:
                raise ValueError(
                    "optimizer contains a frozen, borrowed-Qwen, or external parameter"
                )
            actual_ids.append(parameter_id)
            names.append(name)
        if not names:
            raise ValueError("optimizer parameter groups cannot be empty")
        groups.append(tuple(names))
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("optimizer contains a duplicate parameter")
    if set(actual_ids) != set(owned_parameters):
        missing = sorted(
            owned_parameters[item] for item in set(owned_parameters) - set(actual_ids)
        )
        raise ValueError(f"optimizer omits trainable Adapter parameters: {missing}")
    return tuple(groups)


def _validate_optimizer_group_shape(
    optimizer: torch.optim.Optimizer, saved_state: Mapping[str, object]
) -> None:
    groups = saved_state.get("param_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ReplayMismatchError("optimizer checkpoint param_groups are malformed")
    if len(groups) != len(optimizer.param_groups):
        raise IdentityMismatchError("optimizer parameter-group count mismatch")
    for current, saved in zip(optimizer.param_groups, groups, strict=True):
        if not isinstance(saved, Mapping) or not isinstance(saved.get("params"), list):
            raise ReplayMismatchError(
                "optimizer checkpoint parameter group is malformed"
            )
        if len(current["params"]) != len(saved["params"]):
            raise IdentityMismatchError("optimizer parameter-group width mismatch")


def _validate_global_step_state(
    global_step: int,
    *,
    optimizer_state: Mapping[str, object],
    scheduler_state: Mapping[str, object] | None,
) -> None:
    state_rows = optimizer_state.get("state")
    if not isinstance(state_rows, Mapping):
        raise ReplayMismatchError("optimizer checkpoint state rows are malformed")
    optimizer_steps: list[int] = []
    for row in state_rows.values():
        if not isinstance(row, Mapping) or "step" not in row:
            raise ReplayMismatchError("AdamW checkpoint row has no step counter")
        optimizer_steps.append(_checkpoint_counter(row["step"], name="AdamW step"))
    if global_step > 0 and not optimizer_steps:
        raise ReplayMismatchError("nonzero global step has empty AdamW state")
    if any(step != global_step for step in optimizer_steps):
        raise ReplayMismatchError("AdamW step counters differ from global_step")
    if scheduler_state is not None:
        last_epoch = _checkpoint_counter(
            scheduler_state.get("last_epoch"), name="scheduler last_epoch"
        )
        step_count = _checkpoint_counter(
            scheduler_state.get("_step_count"), name="scheduler _step_count"
        )
        if last_epoch != global_step or step_count != global_step + 1:
            raise ReplayMismatchError(
                "scheduler counters differ from representation global_step"
            )


def _checkpoint_counter(value: object, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ReplayMismatchError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayMismatchError(f"{name} must be numeric")
    resolved = int(value)
    if resolved < 0 or float(resolved) != float(value):
        raise ReplayMismatchError(f"{name} must be a non-negative integer")
    return resolved


def _validate_scheduler(
    scheduler: _Stateful | None, *, optimizer: torch.optim.Optimizer
) -> None:
    if scheduler is None:
        return
    if not callable(getattr(scheduler, "state_dict", None)) or not callable(
        getattr(scheduler, "load_state_dict", None)
    ):
        raise TypeError("scheduler must provide state_dict/load_state_dict")
    bound_optimizer = getattr(scheduler, "optimizer", None)
    if bound_optimizer is not optimizer:
        raise ValueError("scheduler must be bound to the checkpointed optimizer")


def _validate_scheduler_identity(
    scheduler: _Stateful | None,
    identity: RepresentationSchedulerIdentity | None,
) -> None:
    if scheduler is None:
        if identity is not None:
            raise IdentityMismatchError(
                "run identity requires a scheduler but runtime has none"
            )
        return
    if identity is None:
        raise IdentityMismatchError(
            "runtime scheduler is absent from representation run identity"
        )
    if _qualified_type(scheduler) != identity.scheduler_type:
        raise IdentityMismatchError("scheduler type differs from run identity")
    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if not isinstance(lr_lambdas, list) or not lr_lambdas:
        raise IdentityMismatchError(
            "project scheduler does not expose its LambdaLR construction"
        )
    for multiplier in lr_lambdas:
        closure = getattr(multiplier, "__closure__", None)
        if not closure:
            raise IdentityMismatchError(
                "project scheduler LambdaLR has no configuration closure"
            )
        matched = False
        for cell in closure:
            config = cell.cell_contents
            kind = getattr(config, "kind", None)
            kind_value = getattr(kind, "value", kind)
            if (
                kind_value == identity.kind
                and getattr(config, "total_steps", None) == identity.total_steps
                and getattr(config, "warmup_steps", None) == identity.warmup_steps
                and getattr(config, "min_lr_ratio", None)
                == getattr(identity, "min_lr_ratio", None)
            ):
                matched = True
                break
        if not matched:
            raise IdentityMismatchError(
                "scheduler construction differs from representation run identity"
            )


def _validate_rng_state(state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise ReplayMismatchError("RNG state must be a mapping")
    allowed = {"schema_version", "python", "torch_cpu", "torch_cuda"}
    required = {"schema_version", "python", "torch_cpu"}
    keys = set(state)
    if not required <= keys or not keys <= allowed:
        raise ReplayMismatchError("RNG state fields do not match the v2 schema")
    if state["schema_version"] != REPRESENTATION_RNG_STATE_SCHEMA_VERSION:
        raise ReplayMismatchError("RNG state schema mismatch")
    validator = random.Random()
    try:
        validator.setstate(state["python"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError("Python RNG state is malformed") from error
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise ReplayMismatchError("torch CPU RNG state is malformed")
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not isinstance(cuda_state, Mapping) or set(cuda_state) != {
            "device_index",
            "visible_device_count",
            "state",
        }:
            raise ReplayMismatchError("torch CUDA RNG state is malformed")
        device_index = cuda_state["device_index"]
        visible_device_count = cuda_state["visible_device_count"]
        generator_state = cuda_state["state"]
        if (
            isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
            or isinstance(visible_device_count, bool)
            or not isinstance(visible_device_count, int)
            or visible_device_count <= 0
            or device_index >= visible_device_count
            or not isinstance(generator_state, torch.Tensor)
            or generator_state.device.type != "cpu"
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise ReplayMismatchError("torch CUDA RNG state is malformed")
        if not torch.cuda.is_available():
            raise ReplayMismatchError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if not torch.cuda.is_initialized():
            raise ReplayMismatchError(
                "checkpoint contains CUDA RNG state but CUDA is not initialized"
            )
        if visible_device_count != torch.cuda.device_count():
            raise ReplayMismatchError(
                "visible CUDA device count differs from checkpoint"
            )
        if device_index != torch.cuda.current_device():
            raise ReplayMismatchError(
                "current CUDA device index differs from checkpoint"
            )


def _plain_cpu_state(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _plain_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_cpu_state(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint state value {type(value).__qualname__}")


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _update_state_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(_tensor_checksum(value).encode())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        ordered = sorted(value.items(), key=lambda item: _mapping_key(item[0]))
        for key, item in ordered:
            _update_state_digest(digest, key)
            _update_state_digest(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, Enum):
        digest.update(b"enum\0")
        _update_state_digest(digest, value.value)
    elif isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
    elif isinstance(value, bool):
        digest.update(b"bool\0")
        digest.update(b"1" if value else b"0")
    elif isinstance(value, int):
        digest.update(b"int\0")
        digest.update(str(value).encode())
    elif isinstance(value, float):
        digest.update(b"float\0")
        digest.update(json.dumps(value, allow_nan=False).encode())
    elif value is None:
        digest.update(b"none\0")
    else:
        raise TypeError(
            f"unsupported checkpoint digest value {type(value).__qualname__}"
        )


def _mapping_key(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", "1" if value else "0")
    if isinstance(value, int):
        return ("int", str(value))
    raise TypeError(f"unsupported checkpoint mapping key {type(value).__qualname__}")


def _tensor_checksum(value: torch.Tensor) -> str:
    # The shared checksum helper expects at least one dimension when re-viewing
    # bytes; optimizer state legitimately contains scalar step tensors.
    canonical = value if value.ndim else value.reshape(1)
    return tensor_checksum(canonical)


def _save_atomic(value: object, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _torch_load(path: str | Path) -> object:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise ReplayMismatchError(
            f"cannot load representation checkpoint: {error}"
        ) from error


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _require_adapter(adapter: object) -> None:
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _strictly_increasing_non_negative_ints(values: object, *, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{field_name} must contain integers")
    if any(value < 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError(f"{field_name} must be unique and strictly increasing")


def _positive_finite_float(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be an explicit positive finite float")


def _non_negative_finite_float(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be an explicit non-negative finite float")


def _finite_ratio(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be an explicit finite float in [0,1]")


def _runtime_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"runtime optimizer {field_name} must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"runtime optimizer {field_name} must be finite")
    return resolved


def _runtime_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"runtime optimizer {field_name} must be bool")
    return value


def _runtime_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"runtime optimizer {field_name} must be bool or None")
    return value
