"""Pure-tensor losses for representation-phase readability and specificity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch.nn import functional as F


EVIDENCE_IGNORE_INDEX = -100
HISTORICAL_NORM_EPS = 1e-6


class MatrixCEScoreMode(str, Enum):
    """Selectable cell-score mathematics for same-image Matrix CE."""

    LEGACY_SUMMED_NLL = "legacy_summed_nll"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class CausalEvidenceLosses:
    """Per-sample evidence likelihood reductions after a causal shift.

    ``per_sample_token_mean_nll`` is the representation readability loss before
    its independently configured sample reduction and the balanced Matrix-CE
    cell statistic.  ``per_sample_summed_log_likelihood`` is the historical
    negative summed-NLL Matrix-CE statistic.  ``valid_token_counts`` makes the
    different normalizers observable.
    """

    per_sample_token_mean_nll: torch.Tensor
    per_sample_summed_log_likelihood: torch.Tensor
    valid_token_counts: torch.Tensor


def matrix_ce_cell_scores(
    losses: CausalEvidenceLosses,
    *,
    mode: MatrixCEScoreMode,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert per-cell evidence losses into selectable Matrix-CE scores.

    ``legacy_summed_nll`` returns the historical negative summed NLL exactly.
    ``balanced`` removes evidence-length scaling before applying a fixed
    positive temperature: ``-mean_nll / temperature``.
    """

    if not isinstance(losses, CausalEvidenceLosses):
        raise TypeError("losses must be CausalEvidenceLosses")
    if not isinstance(mode, MatrixCEScoreMode):
        raise TypeError("Matrix-CE score mode must be explicit")
    if isinstance(temperature, bool) or not isinstance(temperature, float):
        raise TypeError("Matrix-CE temperature must be an explicit float")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Matrix-CE temperature must be finite and positive")
    if mode is MatrixCEScoreMode.LEGACY_SUMMED_NLL:
        if temperature != 1.0:
            raise ValueError("legacy_summed_nll requires temperature 1.0")
        # Preserve the historical hot path and tensor identity exactly.
        return losses.per_sample_summed_log_likelihood

    values = -losses.per_sample_token_mean_nll / temperature
    if values.ndim != 1 or values.shape != losses.valid_token_counts.shape:
        raise ValueError("Matrix-CE cell scores and evidence counts must align")
    if not values.dtype.is_floating_point:
        raise TypeError("Matrix-CE cell scores must use a floating dtype")
    return values


@dataclass(frozen=True, slots=True)
class EvidenceReadabilityLossTerms:
    """Unnormalized terms for the historical sample-mean readability loss."""

    numerator: torch.Tensor
    sample_count: int

    @property
    def mean(self) -> torch.Tensor:
        if self.sample_count <= 0:
            raise ValueError("evidence readability terms require at least one sample")
        return self.numerator / self.sample_count


@dataclass(frozen=True, slots=True)
class SameImageMatrixCELossTerms:
    """Unnormalized Matrix-CE terms for global DDP/accumulation reduction.

    Training code must sum ``numerator`` and ``valid_row_count`` over the entire
    data-parallel accumulation window before applying the final division.  It
    must not average already-normalized rank-local or microbatch-local means.
    """

    numerator: torch.Tensor
    valid_row_count: int

    @property
    def mean(self) -> torch.Tensor:
        if self.valid_row_count <= 0:
            return self.numerator
        return self.numerator / self.valid_row_count


@dataclass(frozen=True, slots=True)
class HistoricalNormLossTerms:
    """Unnormalized sample terms for the fixed historical norm objective."""

    numerator: torch.Tensor
    sample_count: int

    @property
    def mean(self) -> torch.Tensor:
        if self.sample_count <= 0:
            raise ValueError("historical norm terms require at least one sample")
        return self.numerator / self.sample_count


def causal_evidence_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> CausalEvidenceLosses:
    """Compute causal evidence NLL and summed log-likelihood per sample.

    ``logits[:, position]`` predicts ``labels[:, position + 1]``.  A label of
    ``-100`` is ignored.  Every sample must retain at least one evidence label
    after this shift; otherwise the function fails closed.  The historical
    implementation used ``clamp_min(1)`` and silently reported zero for such a
    sample.  Rejecting it here is an intentional native-pipeline correctness
    repair.  In particular, a non-ignored label only at position zero is not a
    valid causal supervision token.
    """

    _validate_causal_evidence_inputs(logits, labels)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != EVIDENCE_IGNORE_INDEX
    valid_token_counts = valid_mask.sum(dim=-1)
    if bool((valid_token_counts == 0).any().item()):
        raise ValueError(
            "every sample must have at least one non-ignored evidence token "
            "after the causal shift"
        )

    valid_labels = shift_labels[valid_mask]
    if bool(((valid_labels < 0) | (valid_labels >= logits.shape[-1])).any().item()):
        raise ValueError("non-ignored evidence labels must be valid vocabulary ids")

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=EVIDENCE_IGNORE_INDEX,
        reduction="none",
    ).reshape_as(shift_labels)
    summed_nll = per_token_nll.sum(dim=-1)
    token_mean_nll = summed_nll / valid_token_counts.to(dtype=summed_nll.dtype)
    return CausalEvidenceLosses(
        per_sample_token_mean_nll=token_mean_nll,
        per_sample_summed_log_likelihood=-summed_nll,
        valid_token_counts=valid_token_counts,
    )


def evidence_readability_loss_terms(
    losses: CausalEvidenceLosses,
) -> EvidenceReadabilityLossTerms:
    """Return sample-sum/sample-count terms without changing token weighting."""

    if not isinstance(losses, CausalEvidenceLosses):
        raise TypeError("losses must be a CausalEvidenceLosses instance")
    values = losses.per_sample_token_mean_nll
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("per-sample evidence losses must be a non-empty vector")
    return EvidenceReadabilityLossTerms(
        numerator=values.sum(),
        sample_count=int(values.shape[0]),
    )


def historical_visual_token_norm_loss(
    d_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return the one accepted historical norm loss for one visual path.

    The computation deliberately follows the pinned implementation in FP32:
    ``mean(log((||D_t|| + 1e-6) / clamp_min(mean ||V_s||, 1e-6))**2)``.
    The post-merger source is detached inside this primitive, so the only
    gradient-bearing input is ``D``.  There is intentionally no selectable
    norm mode, target statistic, or epsilon.
    """

    _validate_norm_tensor(d_tokens, name="D tokens")
    _validate_norm_tensor(source_tokens, name="source visual tokens")
    if d_tokens.device != source_tokens.device:
        raise ValueError("D and source visual tokens must share a device")
    if d_tokens.shape[-1] != source_tokens.shape[-1]:
        raise ValueError("D and source visual tokens must share hidden width")

    d_norm = torch.linalg.vector_norm(d_tokens.float(), dim=-1)
    source_reference = (
        torch.linalg.vector_norm(source_tokens.detach().float(), dim=-1)
        .mean()
        .clamp_min(HISTORICAL_NORM_EPS)
    )
    log_ratio = torch.log((d_norm + HISTORICAL_NORM_EPS) / source_reference)
    return log_ratio.square().mean()


def historical_sample_norm_loss(
    main_d: torch.Tensor,
    main_source: torch.Tensor,
    deepstack_d: Sequence[torch.Tensor],
    deepstack_source: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Reduce main plus the three corresponding D-DeepStack paths per sample.

    The three branch losses are averaged first.  That branch mean and the main
    loss are then averaged with equal weight, exactly as required by
    ``RPI-20260719-NORM-EVAL``.
    """

    d_branches = tuple(deepstack_d)
    source_branches = tuple(deepstack_source)
    if len(d_branches) != 3 or len(source_branches) != 3:
        raise ValueError("historical sample norm loss requires exactly three branches")
    main_loss = historical_visual_token_norm_loss(main_d, main_source)
    branch_losses = torch.stack(
        tuple(
            historical_visual_token_norm_loss(d_branch, source_branch)
            for d_branch, source_branch in zip(d_branches, source_branches, strict=True)
        )
    )
    return (main_loss + branch_losses.mean()) / 2


def historical_norm_loss_terms(
    per_sample_losses: Sequence[torch.Tensor],
) -> HistoricalNormLossTerms:
    """Return the sample numerator/count without local pre-normalization."""

    values = tuple(per_sample_losses)
    if not values:
        raise ValueError("historical norm loss requires at least one sample")
    first = values[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 0:
        raise ValueError("every historical norm sample loss must be a scalar tensor")
    if not first.dtype.is_floating_point:
        raise TypeError("historical norm sample losses must use floating dtype")
    for value in values:
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError(
                "every historical norm sample loss must be a scalar tensor"
            )
        if value.device != first.device or value.dtype != first.dtype:
            raise ValueError(
                "historical norm sample losses must share device and dtype"
            )
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError("historical norm sample loss must be finite")
    return HistoricalNormLossTerms(
        numerator=torch.stack(values).sum(),
        sample_count=len(values),
    )


def same_image_matrix_ce_loss_terms(
    score_matrices: Sequence[torch.Tensor],
) -> SameImageMatrixCELossTerms:
    """Return the CE numerator and row count without local normalization."""

    matrices, total_rows, zero = _validate_score_matrices(score_matrices)
    total_loss = zero
    for scores in matrices:
        if scores.numel() == 0:
            continue
        labels = torch.arange(scores.shape[0], device=scores.device)
        total_loss = total_loss + F.cross_entropy(scores, labels, reduction="sum")
    return SameImageMatrixCELossTerms(
        numerator=total_loss,
        valid_row_count=total_rows,
    )


def same_image_matrix_ce_loss(score_matrices: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return diagonal-label Matrix CE normalized by rows across all groups.

    Each square matrix represents one same-image group: rows hold fixed target
    context/evidence and columns select the candidate main ``D`` plus all of its
    D-DeepStack branches.  The correct column for row ``i`` is ``i``.  Cross
    entropy is summed within each matrix and divided once by the total number
    of rows across all non-empty matrices.  No temperature is applied.

    Empty input, or input containing only ``[0, 0]`` matrices, returns a
    detached scalar zero and therefore cannot create a false training signal.
    """

    return same_image_matrix_ce_loss_terms(score_matrices).mean


def same_image_matrix_ce_score_gradients(
    score_matrices: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return Matrix-CE value and explicit gradients with respect to scores.

    The returned gradient for every non-empty matrix is
    ``(softmax(scores) - one_hot(diagonal)) / total_rows``.  Score tensors are
    detached for this explicit calculation, so the gradient tensors do not
    retain or mutate the caller's autograd graph.  The loss value itself keeps
    the normal autograd relationship to input scores.  This registered legacy
    helper accepts FP16, BF16, or FP32 and is a local parity/debug primitive; a
    distributed or accumulated trainer must differentiate the unnormalized
    numerator from :func:`same_image_matrix_ce_loss_terms` instead of averaging
    these already locally normalized gradients.
    """

    matrices, total_rows, zero = _validate_score_matrices(score_matrices)
    if matrices and matrices[0].dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(
            "explicit legacy Matrix-CE score gradients require FP16, BF16, or FP32"
        )
    if total_rows == 0:
        return zero, tuple(torch.zeros_like(scores) for scores in matrices)

    loss = same_image_matrix_ce_loss(matrices)
    gradients: list[torch.Tensor] = []
    for scores in matrices:
        if scores.numel() == 0:
            gradients.append(torch.zeros_like(scores))
            continue
        row_indices = torch.arange(scores.shape[0], device=scores.device)
        probabilities = torch.softmax(scores.detach().float(), dim=-1).to(
            dtype=scores.dtype
        )
        gradient = probabilities.clone()
        gradient[row_indices, row_indices] -= 1
        gradients.append(gradient / total_rows)
    return loss, tuple(gradients)


def _validate_causal_evidence_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensor instances")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch and sequence dimensions")
    if logits.shape[0] == 0:
        raise ValueError("causal evidence loss requires a non-empty batch")
    if logits.shape[1] < 2:
        raise ValueError(
            "causal evidence loss requires at least two sequence positions"
        )
    if logits.shape[2] == 0:
        raise ValueError("causal evidence loss requires a non-empty vocabulary")
    if not logits.dtype.is_floating_point:
        raise TypeError("logits must use a floating dtype")
    if labels.dtype != torch.long:
        raise TypeError("labels must have dtype torch.long")
    if logits.device != labels.device:
        raise ValueError("logits and labels must be on the same device")


def _validate_norm_tensor(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 2 or value.shape[-2] == 0 or value.shape[-1] == 0:
        raise ValueError(f"{name} must contain non-empty token and hidden dimensions")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_score_matrices(
    score_matrices: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], int, torch.Tensor]:
    if not isinstance(score_matrices, Sequence):
        raise TypeError("score_matrices must be a sequence of tensors")
    matrices = tuple(score_matrices)
    if not matrices:
        return matrices, 0, torch.zeros(())

    first = matrices[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("score_matrices must contain only torch.Tensor instances")
    if not first.dtype.is_floating_point:
        raise TypeError("score matrices must use a floating dtype")
    reference_device = first.device
    reference_dtype = first.dtype

    total_rows = 0
    for scores in matrices:
        if not isinstance(scores, torch.Tensor):
            raise TypeError("score_matrices must contain only torch.Tensor instances")
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("every score matrix must be square with shape [K, K]")
        if scores.shape[0] == 1:
            raise ValueError(
                "a non-empty Matrix-CE group requires at least two targets"
            )
        if not scores.dtype.is_floating_point:
            raise TypeError("score matrices must use a floating dtype")
        if scores.device != reference_device or scores.dtype != reference_dtype:
            raise ValueError("all score matrices must share device and dtype")
        if scores.numel() != 0:
            total_rows += int(scores.shape[0])
    return matrices, total_rows, first.new_zeros(())
