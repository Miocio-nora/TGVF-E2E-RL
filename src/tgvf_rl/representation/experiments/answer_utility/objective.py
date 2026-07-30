"""Pure loss composition for the isolated answer-utility experiment.

All inputs are already-reduced scalar losses.  In particular, the three
answer values are mean token NLLs (including the supervised EOS token), not
summed sequence NLLs.  Keeping token reduction outside this module makes the
objective independent of a particular Qwen forward implementation while the
strict tensor checks prevent accidental broadcasting or mixed-device loss
composition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION = "answer-utility-objective-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerUtilityObjectiveConfig:
    """Explicit weights and comparison scale for one experiment sidecar.

    No loss weight has a default.  A sidecar therefore records every active
    and inactive term instead of silently inheriting an experiment-wide
    setting.  Profiles E0--E4 additionally validate these weights in
    :mod:`.config`.
    """

    answer_weight: float
    correct_vs_zero_weight: float
    correct_vs_wrong_weight: float
    existing_evidence_weight: float
    existing_matrix_weight: float
    norm_weight: float
    comparison_margin: float
    comparison_temperature: float
    schema_version: str = ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("answer-utility objective schema mismatch")
        for field_name in (
            "answer_weight",
            "correct_vs_zero_weight",
            "correct_vs_wrong_weight",
            "existing_evidence_weight",
            "existing_matrix_weight",
            "norm_weight",
            "comparison_margin",
        ):
            _non_negative_explicit_float(
                getattr(self, field_name), field_name=field_name
            )
        _positive_explicit_float(
            self.comparison_temperature,
            field_name="comparison_temperature",
        )
        weights = self.loss_weights
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one answer-utility loss weight must be positive")
        comparison_active = (
            self.correct_vs_zero_weight > 0.0 or self.correct_vs_wrong_weight > 0.0
        )
        if comparison_active and self.comparison_margin <= 0.0:
            raise ValueError("an active comparison requires a positive margin")
        if not comparison_active and self.comparison_margin != 0.0:
            raise ValueError("comparison_margin must be 0.0 when comparisons are off")

    @property
    def loss_weights(self) -> tuple[float, float, float, float, float, float]:
        """Return weights in the stable sidecar/profile comparison order."""

        return (
            self.answer_weight,
            self.correct_vs_zero_weight,
            self.correct_vs_wrong_weight,
            self.existing_evidence_weight,
            self.existing_matrix_weight,
            self.norm_weight,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerUtilityObjectiveTerms:
    """Optional scalar terms produced by the selected experiment views.

    ``correct_answer_nll`` is both the absolute answer-supervision term and
    the positive member of either counterfactual comparison.  A value is
    required exactly when its configured weight path is active.
    """

    correct_answer_nll: torch.Tensor | None = None
    zero_answer_nll: torch.Tensor | None = None
    wrong_answer_nll: torch.Tensor | None = None
    existing_evidence_loss: torch.Tensor | None = None
    existing_matrix_loss: torch.Tensor | None = None
    norm_loss: torch.Tensor | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerUtilityObjectiveValue:
    """Composed loss together with raw and weighted, separately loggable terms."""

    config: AnswerUtilityObjectiveConfig
    total_loss: torch.Tensor
    answer_nll: torch.Tensor | None
    correct_vs_zero_loss: torch.Tensor | None
    correct_vs_wrong_loss: torch.Tensor | None
    existing_evidence_loss: torch.Tensor | None
    existing_matrix_loss: torch.Tensor | None
    norm_loss: torch.Tensor | None
    weighted_answer: torch.Tensor | None
    weighted_correct_vs_zero: torch.Tensor | None
    weighted_correct_vs_wrong: torch.Tensor | None
    weighted_existing_evidence: torch.Tensor | None
    weighted_existing_matrix: torch.Tensor | None
    weighted_norm: torch.Tensor | None

    def __post_init__(self) -> None:
        _scalar_floating_tensor(self.total_loss, field_name="total_loss")
        raw_and_weighted = (
            self.answer_nll,
            self.correct_vs_zero_loss,
            self.correct_vs_wrong_loss,
            self.existing_evidence_loss,
            self.existing_matrix_loss,
            self.norm_loss,
            self.weighted_answer,
            self.weighted_correct_vs_zero,
            self.weighted_correct_vs_wrong,
            self.weighted_existing_evidence,
            self.weighted_existing_matrix,
            self.weighted_norm,
        )
        for value in raw_and_weighted:
            if value is not None:
                _scalar_floating_tensor(value, field_name="objective component")


def compose_answer_utility_objective(
    terms: AnswerUtilityObjectiveTerms,
    config: AnswerUtilityObjectiveConfig,
) -> AnswerUtilityObjectiveValue:
    """Compose absolute answer, counterfactual, and legacy auxiliary terms.

    The smooth pairwise margin is expressed in NLL space.  For a control
    ``c`` it is

    ``temperature * softplus((correct_nll - c_nll + margin) / temperature)``.

    Minimizing it makes the correct-D answer NLL lower than the control-D NLL
    by at least ``margin``.  Multiplication by temperature preserves NLL units
    and approaches a hinge loss as the temperature approaches zero.
    """

    if not isinstance(terms, AnswerUtilityObjectiveTerms):
        raise TypeError("terms must be AnswerUtilityObjectiveTerms")
    if not isinstance(config, AnswerUtilityObjectiveConfig):
        raise TypeError("config must be AnswerUtilityObjectiveConfig")

    requirements = {
        "correct_answer_nll": (
            config.answer_weight > 0.0
            or config.correct_vs_zero_weight > 0.0
            or config.correct_vs_wrong_weight > 0.0
        ),
        "zero_answer_nll": config.correct_vs_zero_weight > 0.0,
        "wrong_answer_nll": config.correct_vs_wrong_weight > 0.0,
        "existing_evidence_loss": config.existing_evidence_weight > 0.0,
        "existing_matrix_loss": config.existing_matrix_weight > 0.0,
        "norm_loss": config.norm_weight > 0.0,
    }
    active: list[tuple[str, torch.Tensor]] = []
    for field_name, required in requirements.items():
        value = getattr(terms, field_name)
        if required:
            if value is None:
                raise ValueError(f"active objective requires {field_name}")
            _scalar_floating_tensor(value, field_name=field_name)
            active.append((field_name, value))
        elif value is not None:
            raise ValueError(f"inactive objective term {field_name} must be None")
    _require_same_device_and_dtype(active)

    correct = terms.correct_answer_nll
    zero = terms.zero_answer_nll
    wrong = terms.wrong_answer_nll
    correct_vs_zero = (
        None
        if config.correct_vs_zero_weight == 0.0
        else _smooth_nll_margin(
            _required_tensor(correct),
            _required_tensor(zero),
            margin=config.comparison_margin,
            temperature=config.comparison_temperature,
        )
    )
    correct_vs_wrong = (
        None
        if config.correct_vs_wrong_weight == 0.0
        else _smooth_nll_margin(
            _required_tensor(correct),
            _required_tensor(wrong),
            margin=config.comparison_margin,
            temperature=config.comparison_temperature,
        )
    )

    weighted_answer = _weighted(correct, config.answer_weight)
    weighted_correct_vs_zero = _weighted(correct_vs_zero, config.correct_vs_zero_weight)
    weighted_correct_vs_wrong = _weighted(
        correct_vs_wrong, config.correct_vs_wrong_weight
    )
    weighted_existing_evidence = _weighted(
        terms.existing_evidence_loss, config.existing_evidence_weight
    )
    weighted_existing_matrix = _weighted(
        terms.existing_matrix_loss, config.existing_matrix_weight
    )
    weighted_norm = _weighted(terms.norm_loss, config.norm_weight)
    weighted_terms = (
        weighted_answer,
        weighted_correct_vs_zero,
        weighted_correct_vs_wrong,
        weighted_existing_evidence,
        weighted_existing_matrix,
        weighted_norm,
    )
    present_weighted_terms = [value for value in weighted_terms if value is not None]
    if not present_weighted_terms:  # guarded by config validation; defensive here
        raise RuntimeError("answer-utility objective has no active terms")
    total = present_weighted_terms[0]
    for value in present_weighted_terms[1:]:
        total = total + value

    return AnswerUtilityObjectiveValue(
        config=config,
        total_loss=total,
        answer_nll=correct if config.answer_weight > 0.0 else None,
        correct_vs_zero_loss=correct_vs_zero,
        correct_vs_wrong_loss=correct_vs_wrong,
        existing_evidence_loss=terms.existing_evidence_loss,
        existing_matrix_loss=terms.existing_matrix_loss,
        norm_loss=terms.norm_loss,
        weighted_answer=weighted_answer,
        weighted_correct_vs_zero=weighted_correct_vs_zero,
        weighted_correct_vs_wrong=weighted_correct_vs_wrong,
        weighted_existing_evidence=weighted_existing_evidence,
        weighted_existing_matrix=weighted_existing_matrix,
        weighted_norm=weighted_norm,
    )


def _smooth_nll_margin(
    correct_nll: torch.Tensor,
    control_nll: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    return F.softplus((correct_nll - control_nll + margin) / temperature) * temperature


def _weighted(value: torch.Tensor | None, weight: float) -> torch.Tensor | None:
    if weight == 0.0:
        return None
    return _required_tensor(value) * weight


def _required_tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:  # requirements are checked before composition
        raise RuntimeError("required objective tensor is missing")
    return value


def _scalar_floating_tensor(value: object, *, field_name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"{field_name} must be a scalar tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{field_name} must use a floating dtype")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f"{field_name} must be finite")


def _require_same_device_and_dtype(
    values: list[tuple[str, torch.Tensor]],
) -> None:
    reference_name, reference = values[0]
    for field_name, value in values[1:]:
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{field_name} must share device and dtype with {reference_name}"
            )


def _non_negative_explicit_float(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{field_name} must be an explicit float")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _positive_explicit_float(value: object, *, field_name: str) -> None:
    _non_negative_explicit_float(value, field_name=field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")
