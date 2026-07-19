"""Typed, pure reductions for representation-phase internal evaluation.

These helpers report evaluation evidence only.  They define neither promotion
thresholds nor a norm-training objective.  Query score matrices always contain
NLL values, so lower values are better.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final, Literal

import torch


NLL_LOWER_IS_BETTER: Final = "nll_lower_is_better"
INDEX_STABLE_TIE_BREAK: Final = "stable_column_index"
NEAR_IDENTICAL_TOKEN_STD_THRESHOLD: Final = 1e-4

ScoreSemantics = Literal["nll_lower_is_better"]
TieBreakSemantics = Literal["stable_column_index"]


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    """Finite-value population statistics with the original count retained."""

    count: int
    finite_count: int
    finite_rate: float
    mean: float | None
    median: float | None
    population_std: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class ReadoutNLLs:
    """Evidence-token NLLs for one sample under matched readout controls."""

    correct_d: float
    target_only: float
    random_d: float
    wrong_same_image_d: float | None = None
    wrong_different_image_d: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("correct_d", "target_only", "random_d"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), name=field_name),
            )
        for field_name in ("wrong_same_image_d", "wrong_different_image_d"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _finite_float(value, name=field_name)
                )


@dataclass(frozen=True, slots=True)
class ReadoutSampleMetrics:
    """Per-sample advantages; positive means correct ``D`` has lower NLL."""

    nlls: ReadoutNLLs
    advantage_vs_target_only: float
    advantage_vs_random_d: float
    advantage_vs_wrong_same_image_d: float | None
    advantage_vs_wrong_different_image_d: float | None
    correct_beats_all_available_controls: bool
    correct_beats_all_controls: bool | None


@dataclass(frozen=True, slots=True)
class ReadoutControlMetrics:
    control: str
    available_count: int
    mean_nll: float | None
    mean_correct_advantage: float | None
    correct_win_rate: float | None


@dataclass(frozen=True, slots=True)
class ReadoutMetrics:
    sample_count: int
    mean_correct_d_nll: float | None
    median_correct_d_nll: float | None
    target_only: ReadoutControlMetrics
    random_d: ReadoutControlMetrics
    wrong_same_image_d: ReadoutControlMetrics
    wrong_different_image_d: ReadoutControlMetrics
    correct_beats_all_available_controls_rate: float | None
    complete_control_sample_count: int
    correct_beats_all_controls_rate: float | None
    score_semantics: ScoreSemantics = NLL_LOWER_IS_BETTER


def readout_sample_metrics(nlls: ReadoutNLLs) -> ReadoutSampleMetrics:
    """Compute strict lower-NLL wins for one readout sample.

    A tie is not a win, matching the historical ``pct_positive`` reduction.
    ``correct_beats_all_controls`` is unavailable unless both optional
    wrong-``D`` controls were evaluated.  The separately named
    ``correct_beats_all_available_controls`` uses every control present on the
    sample and never silently imputes a missing wrong-``D`` score.
    """

    if not isinstance(nlls, ReadoutNLLs):
        raise TypeError("nlls must be a ReadoutNLLs instance")

    optional_controls = (
        nlls.wrong_same_image_d,
        nlls.wrong_different_image_d,
    )
    controls = (nlls.target_only, nlls.random_d) + tuple(
        value for value in optional_controls if value is not None
    )
    complete = all(value is not None for value in optional_controls)
    return ReadoutSampleMetrics(
        nlls=nlls,
        advantage_vs_target_only=nlls.target_only - nlls.correct_d,
        advantage_vs_random_d=nlls.random_d - nlls.correct_d,
        advantage_vs_wrong_same_image_d=_optional_advantage(
            nlls.correct_d, nlls.wrong_same_image_d
        ),
        advantage_vs_wrong_different_image_d=_optional_advantage(
            nlls.correct_d, nlls.wrong_different_image_d
        ),
        correct_beats_all_available_controls=all(
            nlls.correct_d < control for control in controls
        ),
        correct_beats_all_controls=(
            all(nlls.correct_d < control for control in controls) if complete else None
        ),
    )


def summarize_readout(nll_rows: Sequence[ReadoutNLLs]) -> ReadoutMetrics:
    """Reduce readout rows using sample-weighted historical mean semantics."""

    rows = _readout_rows(nll_rows)
    sample_metrics = tuple(readout_sample_metrics(row) for row in rows)
    complete_wins = tuple(
        row.correct_beats_all_controls
        for row in sample_metrics
        if row.correct_beats_all_controls is not None
    )
    return ReadoutMetrics(
        sample_count=len(rows),
        mean_correct_d_nll=_mean(row.correct_d for row in rows),
        median_correct_d_nll=_median(row.correct_d for row in rows),
        target_only=_control_metrics(
            "target_only",
            rows,
            sample_metrics,
            nll_field="target_only",
            advantage_field="advantage_vs_target_only",
        ),
        random_d=_control_metrics(
            "random_d",
            rows,
            sample_metrics,
            nll_field="random_d",
            advantage_field="advantage_vs_random_d",
        ),
        wrong_same_image_d=_control_metrics(
            "wrong_same_image_d",
            rows,
            sample_metrics,
            nll_field="wrong_same_image_d",
            advantage_field="advantage_vs_wrong_same_image_d",
        ),
        wrong_different_image_d=_control_metrics(
            "wrong_different_image_d",
            rows,
            sample_metrics,
            nll_field="wrong_different_image_d",
            advantage_field="advantage_vs_wrong_different_image_d",
        ),
        correct_beats_all_available_controls_rate=_bool_mean(
            row.correct_beats_all_available_controls for row in sample_metrics
        ),
        complete_control_sample_count=len(complete_wins),
        correct_beats_all_controls_rate=_bool_mean(complete_wins),
    )


def grouped_readout_metrics(
    nll_rows: Sequence[ReadoutNLLs],
    group_labels: Sequence[str] | None = None,
) -> dict[str, ReadoutMetrics]:
    """Optionally group readout rows without changing the global reduction."""

    rows = _readout_rows(nll_rows)
    labels = _group_labels(group_labels, expected=len(rows))
    if labels is None:
        return {}
    groups: dict[str, list[ReadoutNLLs]] = defaultdict(list)
    for label, row in zip(labels, rows, strict=True):
        groups[label].append(row)
    return {label: summarize_readout(groups[label]) for label in sorted(groups)}


@dataclass(frozen=True, slots=True)
class QueryRowMetrics:
    row_index: int
    diagonal_nll: float
    best_wrong_nll: float
    diagonal_rank: int
    reciprocal_rank: float
    top1: bool
    top2: bool
    diagonal_gap: float


@dataclass(frozen=True, slots=True)
class QueryScoreMatrixMetrics:
    """Per-image retrieval metrics for one same-image NLL matrix.

    Ties reproduce the exact historical Python stable-sort behavior: columns
    are initially in ascending index order, so equal NLLs favor the lower
    column index.  This is intentionally not a conservative tied-rank metric.
    """

    nll_matrix: tuple[tuple[float, ...], ...]
    rows: tuple[QueryRowMetrics, ...]
    top1_accuracy: float
    top2_accuracy: float
    mean_reciprocal_rank: float
    mean_diagonal_gap: float
    median_diagonal_gap: float
    score_semantics: ScoreSemantics = NLL_LOWER_IS_BETTER
    tie_break_semantics: TieBreakSemantics = INDEX_STABLE_TIE_BREAK

    @property
    def group_size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    group_count: int
    sample_count: int
    retrieval_top1: float | None
    retrieval_top2: float | None
    mean_reciprocal_rank: float | None
    mean_diagonal_gap: float | None
    median_diagonal_gap: float | None
    score_semantics: ScoreSemantics = NLL_LOWER_IS_BETTER
    tie_break_semantics: TieBreakSemantics = INDEX_STABLE_TIE_BREAK


def query_score_matrix_metrics(nll_matrix: torch.Tensor) -> QueryScoreMatrixMetrics:
    """Reduce a square same-image matrix whose entries are lower-is-better NLL."""

    values = _validated_nll_matrix(nll_matrix)
    group_size = len(values)
    rows: list[QueryRowMetrics] = []
    for row_index, row in enumerate(values):
        # Python's sort is stable.  Iterating columns in index order exactly
        # preserves the historical lower-column-index tie break.
        ranked_columns = sorted(range(group_size), key=lambda column: row[column])
        rank = ranked_columns.index(row_index) + 1
        wrong_nll = min(
            value for column, value in enumerate(row) if column != row_index
        )
        diagonal_nll = row[row_index]
        rows.append(
            QueryRowMetrics(
                row_index=row_index,
                diagonal_nll=diagonal_nll,
                best_wrong_nll=wrong_nll,
                diagonal_rank=rank,
                reciprocal_rank=1.0 / rank,
                top1=rank == 1,
                top2=rank <= 2,
                diagonal_gap=wrong_nll - diagonal_nll,
            )
        )
    return QueryScoreMatrixMetrics(
        nll_matrix=values,
        rows=tuple(rows),
        top1_accuracy=_required_bool_mean(row.top1 for row in rows),
        top2_accuracy=_required_bool_mean(row.top2 for row in rows),
        mean_reciprocal_rank=_required_mean(row.reciprocal_rank for row in rows),
        mean_diagonal_gap=_required_mean(row.diagonal_gap for row in rows),
        median_diagonal_gap=_required_median(row.diagonal_gap for row in rows),
    )


def summarize_query_score_matrices(
    nll_matrices: Sequence[torch.Tensor],
) -> QueryMetrics:
    """Reduce query retrieval by rows, not by an unweighted mean of groups."""

    if not isinstance(nll_matrices, Sequence):
        raise TypeError("nll_matrices must be a sequence")
    matrices = tuple(query_score_matrix_metrics(matrix) for matrix in nll_matrices)
    rows = tuple(row for matrix in matrices for row in matrix.rows)
    return _summarize_query_rows(rows, group_count=len(matrices))


def grouped_query_score_metrics(
    nll_matrices: Sequence[torch.Tensor],
    group_labels: Sequence[str] | None = None,
) -> dict[str, QueryMetrics]:
    """Optionally group whole same-image matrices by caller-provided labels."""

    if not isinstance(nll_matrices, Sequence):
        raise TypeError("nll_matrices must be a sequence")
    matrices = tuple(nll_matrices)
    labels = _group_labels(group_labels, expected=len(matrices))
    if labels is None:
        return {}
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for label, matrix in zip(labels, matrices, strict=True):
        groups[label].append(matrix)
    return {
        label: summarize_query_score_matrices(groups[label]) for label in sorted(groups)
    }


def grouped_query_row_metrics(
    nll_matrices: Sequence[torch.Tensor],
    row_group_labels: Sequence[Sequence[str]] | None = None,
) -> dict[str, QueryMetrics]:
    """Optionally group matrix rows by sample-level evidence/source labels.

    ``row_group_labels[matrix_index][row_index]`` labels one query/evidence
    sample.  Reductions remain sample weighted, matching the historical
    evidence-type and source-profile reports.  ``group_count`` reports how many
    distinct same-image matrices contributed at least one row to that label.
    """

    if not isinstance(nll_matrices, Sequence):
        raise TypeError("nll_matrices must be a sequence")
    matrices = tuple(query_score_matrix_metrics(matrix) for matrix in nll_matrices)
    if row_group_labels is None:
        return {}
    if isinstance(row_group_labels, (str, bytes)) or not isinstance(
        row_group_labels, Sequence
    ):
        raise TypeError("row_group_labels must be a nested sequence or None")
    nested_labels = tuple(row_group_labels)
    if len(nested_labels) != len(matrices):
        raise ValueError("row_group_labels must align one-to-one with matrices")

    grouped_rows: dict[str, list[QueryRowMetrics]] = defaultdict(list)
    contributing_matrices: dict[str, set[int]] = defaultdict(set)
    for matrix_index, (matrix, matrix_labels) in enumerate(
        zip(matrices, nested_labels, strict=True)
    ):
        labels = _group_labels(matrix_labels, expected=matrix.group_size)
        if labels is None:  # pragma: no cover - nested entries cannot be None by type
            raise AssertionError("matrix row labels unexpectedly absent")
        for label, row in zip(labels, matrix.rows, strict=True):
            grouped_rows[label].append(row)
            contributing_matrices[label].add(matrix_index)
    return {
        label: _summarize_query_rows(
            tuple(grouped_rows[label]),
            group_count=len(contributing_matrices[label]),
        )
        for label in sorted(grouped_rows)
    }


@dataclass(frozen=True, slots=True)
class TensorDistributionDiagnostics:
    """Health metrics for a tensor; no field contributes an optimization loss."""

    shape: tuple[int, ...]
    element_values: DistributionMetrics
    token_norms: DistributionMetrics
    fully_finite: bool
    token_cosine_to_mean_population_std: float | None
    near_identical_token_collapse: bool | None
    collapse_definition_threshold: float = NEAR_IDENTICAL_TOKEN_STD_THRESHOLD

    @property
    def finite_rate(self) -> float:
        return self.element_values.finite_rate

    @property
    def mean_token_norm(self) -> float | None:
        return self.token_norms.mean


@dataclass(frozen=True, slots=True)
class TensorDiagnosticsSummary:
    tensor_count: int
    fully_finite_tensor_rate: float | None
    mean_element_finite_rate: float | None
    collapse_evaluable_count: int
    near_identical_token_collapse_count: int
    near_identical_token_collapse_rate: float | None
    mean_token_norm_across_tensors: DistributionMetrics


@dataclass(frozen=True, slots=True)
class NormComparisonDiagnostics:
    """D/source norm distributions retained strictly as health diagnostics."""

    d: TensorDistributionDiagnostics
    source_visual: TensorDistributionDiagnostics
    d_to_source_mean_token_norm_ratio: DistributionMetrics
    denominator_epsilon: float


@dataclass(frozen=True, slots=True)
class RepresentationHealthSummary:
    """Pinned paired D/source report semantics, retained as diagnostics only."""

    sample_count: int
    joint_d_source_finite_rate: float
    d_near_identical_token_collapse_rate: float
    collapse_warning: bool
    mean_d_token_norm: DistributionMetrics
    mean_source_visual_token_norm: DistributionMetrics
    d_to_source_mean_token_norm_ratio: DistributionMetrics


def tensor_distribution_diagnostics(
    tensor: torch.Tensor,
) -> TensorDistributionDiagnostics:
    """Report finite values, token norms, and historical collapse diagnostic.

    For rank two or greater, all leading dimensions are treated as token axes
    and the final dimension is the feature axis.  Collapse is evaluated only
    when every token vector is finite.  The historical definition is the
    population standard deviation of token cosine similarity to the mean token
    vector being below ``1e-4``; that diagnostic threshold is not a promotion
    threshold.
    """

    values = _metric_tensor(tensor, name="tensor").float()
    if values.ndim == 0:
        raise ValueError("tensor diagnostics require at least one dimension")
    if values.shape[-1] == 0:
        raise ValueError("tensor diagnostics require a non-empty final dimension")

    element_metrics = _distribution(values)
    tokens = _as_token_matrix(values)
    token_norms = torch.linalg.vector_norm(tokens, dim=-1)
    token_norm_metrics = _distribution(token_norms)
    fully_finite = bool(torch.isfinite(values).all().item())

    cosine_std: float | None = None
    collapse: bool | None = None
    if values.ndim >= 2 and tokens.shape[0] > 0 and fully_finite:
        mean_token = tokens.mean(dim=0, keepdim=True)
        cosine = torch.nn.functional.cosine_similarity(tokens, mean_token, dim=-1)
        cosine_std = float(cosine.std(unbiased=False).item())
        collapse = cosine_std < NEAR_IDENTICAL_TOKEN_STD_THRESHOLD

    return TensorDistributionDiagnostics(
        shape=tuple(values.shape),
        element_values=element_metrics,
        token_norms=token_norm_metrics,
        fully_finite=fully_finite,
        token_cosine_to_mean_population_std=cosine_std,
        near_identical_token_collapse=collapse,
    )


def norm_comparison_diagnostics(
    d: torch.Tensor,
    source_visual: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> NormComparisonDiagnostics:
    """Report D norms relative to mean source norm without defining a loss."""

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    d_values = _metric_tensor(d, name="d").float()
    source_values = _metric_tensor(source_visual, name="source_visual").float()
    d_metrics = tensor_distribution_diagnostics(d_values)
    source_metrics = tensor_distribution_diagnostics(source_values)
    d_norms = torch.linalg.vector_norm(_as_token_matrix(d_values), dim=-1)
    source_norms = torch.linalg.vector_norm(_as_token_matrix(source_values), dim=-1)
    finite_source_norms = source_norms[torch.isfinite(source_norms)]
    if finite_source_norms.numel() == 0:
        ratios = torch.full_like(d_norms, float("nan"))
    else:
        denominator = finite_source_norms.mean().clamp_min(eps)
        ratios = d_norms / denominator
    return NormComparisonDiagnostics(
        d=d_metrics,
        source_visual=source_metrics,
        d_to_source_mean_token_norm_ratio=_distribution(ratios),
        denominator_epsilon=float(eps),
    )


def summarize_representation_health(
    diagnostics: Sequence[NormComparisonDiagnostics],
) -> RepresentationHealthSummary:
    """Reproduce the historical paired finite/collapse warning reduction.

    Historical ``finite_rate`` counts a sample only when both ``D`` and its
    source visual tensor are fully finite.  Collapse is evaluated on ``D``;
    an unevaluable/non-finite ``D`` contributes zero to the collapse numerator
    but remains in the sample denominator.  The warning is exactly
    ``collapse_rate > 0.1 or joint_finite_rate < 1.0``.
    """

    if not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a sequence")
    rows = tuple(diagnostics)
    if not rows:
        raise ValueError("representation health summary requires at least one sample")
    if any(not isinstance(row, NormComparisonDiagnostics) for row in rows):
        raise TypeError("diagnostics must contain NormComparisonDiagnostics")

    joint_finite_rate = sum(
        row.d.fully_finite and row.source_visual.fully_finite for row in rows
    ) / len(rows)
    collapse_rate = sum(
        bool(row.d.near_identical_token_collapse) for row in rows
    ) / len(rows)
    d_mean_norms = torch.tensor(
        [
            row.d.mean_token_norm
            for row in rows
            if row.d.fully_finite and row.d.mean_token_norm is not None
        ],
        dtype=torch.float64,
    )
    source_mean_norms = torch.tensor(
        [
            row.source_visual.mean_token_norm
            for row in rows
            if row.source_visual.fully_finite
            and row.source_visual.mean_token_norm is not None
        ],
        dtype=torch.float64,
    )
    ratios = torch.tensor(
        [
            row.d_to_source_mean_token_norm_ratio.mean
            for row in rows
            if row.d.fully_finite
            and row.source_visual.fully_finite
            and row.d_to_source_mean_token_norm_ratio.mean is not None
        ],
        dtype=torch.float64,
    )
    return RepresentationHealthSummary(
        sample_count=len(rows),
        joint_d_source_finite_rate=joint_finite_rate,
        d_near_identical_token_collapse_rate=collapse_rate,
        collapse_warning=collapse_rate > 0.1 or joint_finite_rate < 1.0,
        mean_d_token_norm=_distribution(d_mean_norms),
        mean_source_visual_token_norm=_distribution(source_mean_norms),
        d_to_source_mean_token_norm_ratio=_distribution(ratios),
    )


def summarize_tensor_diagnostics(
    diagnostics: Sequence[TensorDistributionDiagnostics],
) -> TensorDiagnosticsSummary:
    """Reduce per-tensor health diagnostics without emitting gate warnings."""

    if not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a sequence")
    rows = tuple(diagnostics)
    if any(not isinstance(row, TensorDistributionDiagnostics) for row in rows):
        raise TypeError("diagnostics must contain TensorDistributionDiagnostics")
    collapses = tuple(
        row.near_identical_token_collapse
        for row in rows
        if row.near_identical_token_collapse is not None
    )
    mean_norms = torch.tensor(
        [row.mean_token_norm for row in rows if row.mean_token_norm is not None],
        dtype=torch.float64,
    )
    return TensorDiagnosticsSummary(
        tensor_count=len(rows),
        fully_finite_tensor_rate=_bool_mean(row.fully_finite for row in rows),
        mean_element_finite_rate=_mean(row.finite_rate for row in rows),
        collapse_evaluable_count=len(collapses),
        near_identical_token_collapse_count=sum(collapses),
        near_identical_token_collapse_rate=_bool_mean(collapses),
        mean_token_norm_across_tensors=_distribution(mean_norms),
    )


def grouped_tensor_diagnostics(
    diagnostics: Sequence[TensorDistributionDiagnostics],
    group_labels: Sequence[str] | None = None,
) -> dict[str, TensorDiagnosticsSummary]:
    """Optionally group already-computed tensor diagnostics."""

    if not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a sequence")
    rows = tuple(diagnostics)
    labels = _group_labels(group_labels, expected=len(rows))
    if labels is None:
        return {}
    groups: dict[str, list[TensorDistributionDiagnostics]] = defaultdict(list)
    for label, row in zip(labels, rows, strict=True):
        groups[label].append(row)
    return {
        label: summarize_tensor_diagnostics(groups[label]) for label in sorted(groups)
    }


@dataclass(frozen=True, slots=True)
class AttentionDiagnostics:
    """Historical slot-to-visual attention reductions for one observation."""

    slot_count: int
    visual_token_count: int
    effective_topk: int
    entropy_values: tuple[float, ...]
    top1_mass_values: tuple[float, ...]
    topk_mass_values: tuple[float, ...]
    visual_token_coverage: float


@dataclass(frozen=True, slots=True)
class AttentionDiagnosticsSummary:
    observation_count: int
    slot_count: int
    effective_topk: int | None
    entropy: DistributionMetrics
    top1_mass: DistributionMetrics
    topk_mass: DistributionMetrics
    mean_visual_token_coverage: float | None


def attention_diagnostics(
    *,
    sub_slot_attention_weights: torch.Tensor | None = None,
    attention_weights: torch.Tensor | None = None,
    topk: int = 5,
    eps: float = 1e-12,
) -> AttentionDiagnostics | None:
    """Reduce explicitly supplied slot-to-visual attention weights.

    This reproduces the registered historical selection contract without
    inferring model axes: a rank-3 ``sub_slot_attention_weights`` tensor is
    averaged over dimension one; a rank-2 tensor is used directly; otherwise a
    rank-2 ``attention_weights`` tensor is used as fallback.  Invalid or absent
    ranks return ``None``.  Selected input must be finite and non-negative.
    """

    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    attention = _select_slot_visual_attention(
        sub_slot_attention_weights=sub_slot_attention_weights,
        attention_weights=attention_weights,
    )
    if attention is None or attention.numel() == 0:
        return None
    weights = _metric_tensor(attention, name="selected attention").float()
    if weights.shape[-1] == 0:
        return None
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("selected attention must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError("selected attention must be non-negative")

    normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    safe = normalized.clamp_min(eps)
    entropy = -(safe * safe.log()).sum(dim=-1)
    sorted_weights = normalized.sort(dim=-1, descending=True).values
    effective_topk = min(topk, normalized.shape[-1])
    top1_mass = sorted_weights[:, 0]
    topk_mass = sorted_weights[:, :effective_topk].sum(dim=-1)
    topk_indices = normalized.topk(k=effective_topk, dim=-1).indices
    coverage = topk_indices.unique().numel() / normalized.shape[-1]
    return AttentionDiagnostics(
        slot_count=int(normalized.shape[0]),
        visual_token_count=int(normalized.shape[1]),
        effective_topk=effective_topk,
        entropy_values=_float_tuple(entropy),
        top1_mass_values=_float_tuple(top1_mass),
        topk_mass_values=_float_tuple(topk_mass),
        visual_token_coverage=float(coverage),
    )


def summarize_attention_diagnostics(
    diagnostics: Sequence[AttentionDiagnostics],
) -> AttentionDiagnosticsSummary:
    """Merge historical attention value lists and mean coverage scalars."""

    if not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a sequence")
    rows = tuple(diagnostics)
    if any(not isinstance(row, AttentionDiagnostics) for row in rows):
        raise TypeError("diagnostics must contain AttentionDiagnostics")
    effective_topks = {row.effective_topk for row in rows}
    if len(effective_topks) > 1:
        raise ValueError(
            "attention diagnostics with different effective_topk values must "
            "be summarized separately"
        )
    entropy = tuple(value for row in rows for value in row.entropy_values)
    top1 = tuple(value for row in rows for value in row.top1_mass_values)
    topk_values = tuple(value for row in rows for value in row.topk_mass_values)
    return AttentionDiagnosticsSummary(
        observation_count=len(rows),
        slot_count=sum(row.slot_count for row in rows),
        effective_topk=(None if not effective_topks else next(iter(effective_topks))),
        entropy=_distribution(torch.tensor(entropy, dtype=torch.float64)),
        top1_mass=_distribution(torch.tensor(top1, dtype=torch.float64)),
        topk_mass=_distribution(torch.tensor(topk_values, dtype=torch.float64)),
        mean_visual_token_coverage=_mean(row.visual_token_coverage for row in rows),
    )


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number (bool is not accepted)")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _optional_advantage(correct: float, control: float | None) -> float | None:
    return None if control is None else control - correct


def _readout_rows(nll_rows: Sequence[ReadoutNLLs]) -> tuple[ReadoutNLLs, ...]:
    if not isinstance(nll_rows, Sequence):
        raise TypeError("nll_rows must be a sequence")
    rows = tuple(nll_rows)
    if any(not isinstance(row, ReadoutNLLs) for row in rows):
        raise TypeError("nll_rows must contain ReadoutNLLs instances")
    return rows


def _control_metrics(
    control: str,
    rows: tuple[ReadoutNLLs, ...],
    sample_metrics: tuple[ReadoutSampleMetrics, ...],
    *,
    nll_field: str,
    advantage_field: str,
) -> ReadoutControlMetrics:
    nll_values = tuple(
        value for row in rows if (value := getattr(row, nll_field)) is not None
    )
    advantages = tuple(
        value
        for row in sample_metrics
        if (value := getattr(row, advantage_field)) is not None
    )
    return ReadoutControlMetrics(
        control=control,
        available_count=len(advantages),
        mean_nll=_mean(nll_values),
        mean_correct_advantage=_mean(advantages),
        correct_win_rate=_bool_mean(value > 0 for value in advantages),
    )


def _validated_nll_matrix(
    nll_matrix: torch.Tensor,
) -> tuple[tuple[float, ...], ...]:
    values = _metric_tensor(nll_matrix, name="nll_matrix")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("nll_matrix must be square with shape [K, K]")
    if values.shape[0] < 2:
        raise ValueError("same-image query metrics require at least two targets")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("nll_matrix must contain only finite NLL values")
    return tuple(
        tuple(float(value) for value in row) for row in values.detach().cpu().tolist()
    )


def _summarize_query_rows(
    rows: tuple[QueryRowMetrics, ...], *, group_count: int
) -> QueryMetrics:
    return QueryMetrics(
        group_count=group_count,
        sample_count=len(rows),
        retrieval_top1=_bool_mean(row.top1 for row in rows),
        retrieval_top2=_bool_mean(row.top2 for row in rows),
        mean_reciprocal_rank=_mean(row.reciprocal_rank for row in rows),
        mean_diagonal_gap=_mean(row.diagonal_gap for row in rows),
        median_diagonal_gap=_median(row.diagonal_gap for row in rows),
    )


def _metric_tensor(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    return tensor.detach()


def _as_token_matrix(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values.reshape(-1, values.shape[-1])


def _distribution(values: torch.Tensor) -> DistributionMetrics:
    flat = values.detach().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    count = int(flat.numel())
    finite_count = int(finite.numel())
    if finite_count == 0:
        return DistributionMetrics(
            count=count,
            finite_count=0,
            finite_rate=0.0,
            mean=None,
            median=None,
            population_std=None,
            minimum=None,
            maximum=None,
        )
    finite_cpu = tuple(float(value) for value in finite.cpu().tolist())
    return DistributionMetrics(
        count=count,
        finite_count=finite_count,
        finite_rate=finite_count / count if count else 0.0,
        mean=float(finite.mean().item()),
        median=float(statistics.median(finite_cpu)),
        population_std=float(finite.std(unbiased=False).item()),
        minimum=float(finite.min().item()),
        maximum=float(finite.max().item()),
    )


def _select_slot_visual_attention(
    *,
    sub_slot_attention_weights: torch.Tensor | None,
    attention_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if isinstance(sub_slot_attention_weights, torch.Tensor):
        if sub_slot_attention_weights.ndim == 3:
            if not sub_slot_attention_weights.dtype.is_floating_point:
                raise TypeError("sub-slot attention must use a floating dtype")
            return sub_slot_attention_weights.mean(dim=1)
        if sub_slot_attention_weights.ndim == 2:
            return sub_slot_attention_weights
    if isinstance(attention_weights, torch.Tensor) and attention_weights.ndim == 2:
        return attention_weights
    return None


def _group_labels(
    group_labels: Sequence[str] | None, *, expected: int
) -> tuple[str, ...] | None:
    if group_labels is None:
        return None
    if isinstance(group_labels, (str, bytes)) or not isinstance(group_labels, Sequence):
        raise TypeError("group_labels must be a sequence or None")
    labels = tuple(group_labels)
    if len(labels) != expected:
        raise ValueError("group_labels must align one-to-one with metric inputs")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("group labels must be non-empty strings")
    return labels


def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().flatten().tolist())


def _mean(values: Iterable[float]) -> float | None:
    materialized = tuple(float(value) for value in values)
    return None if not materialized else sum(materialized) / len(materialized)


def _median(values: Iterable[float]) -> float | None:
    materialized = tuple(float(value) for value in values)
    return None if not materialized else float(statistics.median(materialized))


def _bool_mean(values: Iterable[bool]) -> float | None:
    materialized = tuple(bool(value) for value in values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _required_mean(values: Iterable[float]) -> float:
    result = _mean(values)
    if result is None:  # pragma: no cover - guarded by matrix size validation
        raise AssertionError("expected at least one value")
    return result


def _required_median(values: Iterable[float]) -> float:
    result = _median(values)
    if result is None:  # pragma: no cover - guarded by matrix size validation
        raise AssertionError("expected at least one value")
    return result


def _required_bool_mean(values: Iterable[bool]) -> float:
    result = _bool_mean(values)
    if result is None:  # pragma: no cover - guarded by matrix size validation
        raise AssertionError("expected at least one value")
    return result
