"""Explicit composition of representation-phase Matrix CE and ``L_gen``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from .losses import EvidenceReadabilityLossTerms, SameImageMatrixCELossTerms


REPRESENTATION_OBJECTIVE_SCHEMA_VERSION = "representation_objective_v1"


class RepresentationObjectiveKind(str, Enum):
    """Scientifically distinct representation objective identities."""

    MATRIX_CE_AND_L_GEN = "matrix_ce_and_l_gen"
    MATRIX_CE_ONLY_ABLATION = "matrix_ce_only_ablation"


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
        elif self.l_gen_weight != 0:
            raise ValueError("the Matrix-CE-only ablation requires L_gen weight zero")


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


def compose_reference_representation_objective(
    matrix_terms: SameImageMatrixCELossTerms,
    l_gen_terms: EvidenceReadabilityLossTerms,
    config: RepresentationObjectiveConfig,
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
    if not isinstance(config, RepresentationObjectiveConfig):
        raise TypeError("config must be RepresentationObjectiveConfig")
    if matrix_terms.valid_row_count <= 0:
        raise ValueError("representation objective requires valid Matrix-CE rows")
    if l_gen_terms.sample_count <= 0:
        raise ValueError("representation objective requires L_gen samples")
    if matrix_terms.valid_row_count != l_gen_terms.sample_count:
        raise ValueError(
            "Matrix-CE rows and L_gen samples must come from the same logical batch"
        )
    _validate_scalar_numerator(matrix_terms.numerator, name="Matrix-CE numerator")
    _validate_scalar_numerator(l_gen_terms.numerator, name="L_gen numerator")
    if (
        matrix_terms.numerator.device != l_gen_terms.numerator.device
        or matrix_terms.numerator.dtype != l_gen_terms.numerator.dtype
    ):
        raise ValueError("Matrix-CE and L_gen numerators must share device and dtype")

    matrix_ce = matrix_terms.numerator / matrix_terms.valid_row_count
    l_gen = l_gen_terms.numerator / l_gen_terms.sample_count
    weighted_matrix = matrix_ce * config.matrix_ce_weight
    weighted_l_gen = l_gen * config.l_gen_weight
    total = weighted_matrix + weighted_l_gen
    return RepresentationObjectiveValue(
        config=config,
        total_loss=total,
        matrix_ce_loss=matrix_ce,
        l_gen_loss=l_gen,
        weighted_matrix_ce=weighted_matrix,
        weighted_l_gen=weighted_l_gen,
        matrix_valid_row_count=matrix_terms.valid_row_count,
        l_gen_sample_count=l_gen_terms.sample_count,
    )


def _validate_weight(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{field_name} must be an explicit float")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_scalar_numerator(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f"{name} must be finite")
