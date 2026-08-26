"""Explicit composition of representation-phase Matrix CE and ``L_gen``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from .losses import (
    EvidenceReadabilityLossTerms,
    HistoricalNormLossTerms,
    MatrixCEScoreMode,
    SameImageMatrixCELossTerms,
)


REPRESENTATION_OBJECTIVE_SCHEMA_VERSION = "representation_objective_v1"
REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V2 = "representation_objective_v2"
REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V3 = "representation_objective_v3"


class RepresentationObjectiveKind(str, Enum):
    """Scientifically distinct representation objective identities."""

    MATRIX_CE_AND_L_GEN = "matrix_ce_and_l_gen"
    MATRIX_CE_L_GEN_AND_NORM = "matrix_ce_l_gen_and_norm"
    MATRIX_CE_ONLY_ABLATION = "matrix_ce_only_ablation"
    L_GEN_AND_NORM_NO_MATRIX_CE_ABLATION = (
        "l_gen_and_norm_no_matrix_ce_ablation"
    )


@dataclass(frozen=True, slots=True)
class RepresentationObjectiveConfig:
    """Required, no-default weights for one named representation objective."""

    identity: str
    kind: RepresentationObjectiveKind
    matrix_ce_weight: float
    l_gen_weight: float
    schema_version: str = REPRESENTATION_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("representation objective identity must be non-empty")
        if not isinstance(self.kind, RepresentationObjectiveKind):
            raise TypeError("representation objective kind must be explicit")
        if self.schema_version != REPRESENTATION_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("representation objective schema mismatch")
        _validate_weight(self.matrix_ce_weight, field_name="matrix_ce_weight")
        _validate_weight(self.l_gen_weight, field_name="l_gen_weight")
        if self.matrix_ce_weight <= 0:
            raise ValueError("Matrix-CE weight must be greater than zero")
        if self.kind is RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN:
            if self.l_gen_weight <= 0:
                raise ValueError("the baseline requires a nonzero L_gen weight")
        elif self.kind is RepresentationObjectiveKind.MATRIX_CE_ONLY_ABLATION:
            if self.l_gen_weight != 0:
                raise ValueError(
                    "the Matrix-CE-only ablation requires L_gen weight zero"
                )
        else:
            raise ValueError("objective v1 does not support the norm-aware v2 kind")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationObjectiveConfigV2(RepresentationObjectiveConfig):
    """Norm-aware objective identity accepted after RPI-20260719-NORM-EVAL.

    The baseline fixes the historical norm weight at ``0.1`` and requires
    nonzero readability.  Historical Matrix-CE-only/no-norm fixtures remain
    represented by objective v1; v2 intentionally exposes no no-norm option.
    """

    norm_weight: float
    schema_version: str = REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("representation objective identity must be non-empty")
        if not isinstance(self.kind, RepresentationObjectiveKind):
            raise TypeError("representation objective kind must be explicit")
        if self.schema_version != REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V2:
            raise ValueError("representation objective v2 schema mismatch")
        _validate_weight(self.matrix_ce_weight, field_name="matrix_ce_weight")
        _validate_weight(self.l_gen_weight, field_name="l_gen_weight")
        _validate_weight(self.norm_weight, field_name="norm_weight")
        if self.matrix_ce_weight <= 0:
            raise ValueError("Matrix-CE weight must be greater than zero")
        if self.kind is not RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM:
            raise ValueError(
                "objective v2 requires the matrix_ce_l_gen_and_norm baseline"
            )
        if self.l_gen_weight <= 0:
            raise ValueError("the v2 baseline requires a nonzero L_gen weight")
        if self.norm_weight != 0.1:
            raise ValueError("the v2 baseline requires historical norm weight 0.1")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationObjectiveConfigV3(RepresentationObjectiveConfigV2):
    """Norm-aware objective with an explicit Matrix-CE cell-score identity.

    V3 also owns the named no-Matrix-CE ablation. The raw Matrix-CE term is
    still evaluated for matched diagnostics, but its exact zero weight removes
    it from both the optimized total and Adapter gradients.
    """

    matrix_ce_mode: MatrixCEScoreMode = MatrixCEScoreMode.BALANCED
    matrix_ce_temperature: float = 1.0
    schema_version: str = REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V3

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("representation objective identity must be non-empty")
        if self.schema_version != REPRESENTATION_OBJECTIVE_SCHEMA_VERSION_V3:
            raise ValueError("representation objective v3 schema mismatch")
        _validate_weight(self.matrix_ce_weight, field_name="matrix_ce_weight")
        _validate_weight(self.l_gen_weight, field_name="l_gen_weight")
        _validate_weight(self.norm_weight, field_name="norm_weight")
        if self.kind is RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM:
            if self.matrix_ce_weight <= 0:
                raise ValueError("Matrix-CE weight must be greater than zero")
        elif self.kind is (
            RepresentationObjectiveKind.L_GEN_AND_NORM_NO_MATRIX_CE_ABLATION
        ):
            if self.matrix_ce_weight != 0.0:
                raise ValueError(
                    "the no-Matrix-CE ablation requires Matrix-CE weight zero"
                )
        else:
            raise ValueError(
                "objective v3 requires either the matrix_ce_l_gen_and_norm "
                "baseline or the named no-Matrix-CE ablation"
            )
        if self.l_gen_weight <= 0:
            raise ValueError("the v3 baseline requires a nonzero L_gen weight")
        if self.norm_weight != 0.1:
            raise ValueError("the v3 baseline requires historical norm weight 0.1")
        _validate_matrix_ce_score_config(
            self.matrix_ce_mode,
            self.matrix_ce_temperature,
        )


RepresentationObjectiveConfigLike = (
    RepresentationObjectiveConfig
    | RepresentationObjectiveConfigV2
    | RepresentationObjectiveConfigV3
)


def resolve_matrix_ce_score_config(
    config: RepresentationObjectiveConfigLike,
) -> tuple[MatrixCEScoreMode, float]:
    """Resolve old objective schemas to their exact historical score contract."""

    if not isinstance(
        config,
        (
            RepresentationObjectiveConfig,
            RepresentationObjectiveConfigV2,
            RepresentationObjectiveConfigV3,
        ),
    ):
        raise TypeError("config must be a representation objective config")
    if isinstance(config, RepresentationObjectiveConfigV3):
        return config.matrix_ce_mode, config.matrix_ce_temperature
    return MatrixCEScoreMode.LEGACY_SUMMED_NLL, 1.0


@dataclass(frozen=True, slots=True)
class RepresentationObjectiveValue:
    """Loss and separately loggable components after global term reduction."""

    config: RepresentationObjectiveConfig
    total_loss: torch.Tensor
    matrix_ce_loss: torch.Tensor
    l_gen_loss: torch.Tensor
    weighted_matrix_ce: torch.Tensor
    weighted_l_gen: torch.Tensor
    matrix_valid_row_count: int
    l_gen_sample_count: int
    norm_loss: torch.Tensor | None = None
    weighted_norm: torch.Tensor | None = None
    norm_sample_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "total_loss",
            "matrix_ce_loss",
            "l_gen_loss",
            "weighted_matrix_ce",
            "weighted_l_gen",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(f"{field_name} must be a scalar tensor")
        if self.matrix_valid_row_count <= 0 or self.l_gen_sample_count <= 0:
            raise ValueError("representation objective counts must be positive")
        if isinstance(self.config, RepresentationObjectiveConfigV2):
            for field_name in ("norm_loss", "weighted_norm"):
                value = getattr(self, field_name)
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise ValueError(f"{field_name} must be a scalar tensor for v2")
            if self.norm_sample_count <= 0:
                raise ValueError("representation objective norm count must be positive")
        elif (
            self.norm_loss is not None
            or self.weighted_norm is not None
            or self.norm_sample_count != 0
        ):
            raise ValueError("representation objective v1 cannot contain norm terms")


def compose_reference_representation_objective(
    matrix_terms: SameImageMatrixCELossTerms,
    l_gen_terms: EvidenceReadabilityLossTerms,
    config: RepresentationObjectiveConfigLike,
    norm_terms: HistoricalNormLossTerms | None = None,
) -> RepresentationObjectiveValue:
    """Compose one single-process logical batch without hidden scaling.

    This is a value/reference primitive, not a DDP/FSDP backward-scaling API.
    Production code must separately freeze reducer semantics, world-size
    scaling, and accumulation before it may compose distributed terms.
    """

    if not isinstance(matrix_terms, SameImageMatrixCELossTerms):
        raise TypeError("matrix_terms must be SameImageMatrixCELossTerms")
    if not isinstance(l_gen_terms, EvidenceReadabilityLossTerms):
        raise TypeError("l_gen_terms must be EvidenceReadabilityLossTerms")
    if not isinstance(
        config, (RepresentationObjectiveConfig, RepresentationObjectiveConfigV2)
    ):
        raise TypeError("config must be a representation objective config")
    if matrix_terms.valid_row_count <= 0:
        raise ValueError("representation objective requires valid Matrix-CE rows")
    if l_gen_terms.sample_count <= 0:
        raise ValueError("representation objective requires L_gen samples")
    if matrix_terms.valid_row_count != l_gen_terms.sample_count:
        raise ValueError(
            "Matrix-CE rows and L_gen samples must come from the same logical batch"
        )
    if isinstance(config, RepresentationObjectiveConfigV2):
        if not isinstance(norm_terms, HistoricalNormLossTerms):
            raise TypeError("objective v2 requires HistoricalNormLossTerms")
        if norm_terms.sample_count != matrix_terms.valid_row_count:
            raise ValueError(
                "Matrix-CE rows and norm samples must come from the same logical batch"
            )
    elif norm_terms is not None:
        raise ValueError("objective v1 cannot compose historical norm terms")
    _validate_scalar_numerator(matrix_terms.numerator, name="Matrix-CE numerator")
    _validate_scalar_numerator(l_gen_terms.numerator, name="L_gen numerator")
    if (
        matrix_terms.numerator.device != l_gen_terms.numerator.device
        or matrix_terms.numerator.dtype != l_gen_terms.numerator.dtype
    ):
        raise ValueError("Matrix-CE and L_gen numerators must share device and dtype")
    if norm_terms is not None:
        _validate_scalar_numerator(norm_terms.numerator, name="norm numerator")
        if (
            norm_terms.numerator.device != matrix_terms.numerator.device
            or norm_terms.numerator.dtype != matrix_terms.numerator.dtype
        ):
            raise ValueError(
                "Matrix-CE, L_gen, and norm numerators must share device and dtype"
            )

    matrix_ce = matrix_terms.numerator / matrix_terms.valid_row_count
    l_gen = l_gen_terms.numerator / l_gen_terms.sample_count
    weighted_matrix = matrix_ce * config.matrix_ce_weight
    weighted_l_gen = l_gen * config.l_gen_weight
    norm_loss = (
        None if norm_terms is None else norm_terms.numerator / norm_terms.sample_count
    )
    weighted_norm = (
        None if norm_loss is None else norm_loss * config.norm_weight  # type: ignore[union-attr]
    )
    total = weighted_matrix + weighted_l_gen
    if weighted_norm is not None:
        total = total + weighted_norm
    return RepresentationObjectiveValue(
        config=config,
        total_loss=total,
        matrix_ce_loss=matrix_ce,
        l_gen_loss=l_gen,
        weighted_matrix_ce=weighted_matrix,
        weighted_l_gen=weighted_l_gen,
        matrix_valid_row_count=matrix_terms.valid_row_count,
        l_gen_sample_count=l_gen_terms.sample_count,
        norm_loss=norm_loss,
        weighted_norm=weighted_norm,
        norm_sample_count=0 if norm_terms is None else norm_terms.sample_count,
    )


def _validate_weight(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{field_name} must be an explicit float")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_matrix_ce_score_config(
    mode: object,
    temperature: object,
) -> None:
    if not isinstance(mode, MatrixCEScoreMode):
        raise TypeError("matrix_ce_mode must be a MatrixCEScoreMode")
    if isinstance(temperature, bool) or not isinstance(temperature, float):
        raise TypeError("matrix_ce_temperature must be an explicit float")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("matrix_ce_temperature must be finite and positive")
    if mode is MatrixCEScoreMode.LEGACY_SUMMED_NLL and temperature != 1.0:
        raise ValueError("legacy_summed_nll requires matrix_ce_temperature 1.0")


def _validate_scalar_numerator(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f"{name} must be finite")
