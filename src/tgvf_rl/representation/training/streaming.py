"""Memory-bounded Matrix-CE/readability execution over one same-image group."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
)
from tgvf_rl.representation.deepstack import build_original_image_key_block_mask

from .losses import (
    CausalEvidenceLosses,
    causal_evidence_losses,
    historical_sample_norm_loss,
)
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
)
from .readout import (
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    assert_frozen_deterministic_readout_model,
)


@dataclass(frozen=True, slots=True)
class StreamingGroupScores:
    """Detached first-pass values used to derive exact Matrix-CE gradients."""

    sample_ids: tuple[str, ...]
    score_matrix: torch.Tensor
    diagonal_l_gen: torch.Tensor
    evidence_token_counts: torch.Tensor
    historical_norm: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.sample_ids)
        if size < 2 or self.score_matrix.shape != (size, size):
            raise ValueError("streaming score matrix must have shape [K,K], K>=2")
        if self.diagonal_l_gen.shape != (size,):
            raise ValueError("streaming L_gen values must have shape [K]")
        if self.evidence_token_counts.shape != (size,):
            raise ValueError("streaming evidence counts must have shape [K]")
        if self.historical_norm.shape != (size,):
            raise ValueError("streaming historical norm values must have shape [K]")
        for tensor in (
            self.score_matrix,
            self.diagonal_l_gen,
            self.evidence_token_counts,
            self.historical_norm,
        ):
            if tensor.requires_grad:
                raise ValueError("streaming first-pass values must be detached")


@dataclass(frozen=True, slots=True)
class StreamingMultiGroupScores:
    """Detached blockwise scores plus the realized cross-group batch schedule."""

    group_sample_ids: tuple[tuple[str, ...], ...]
    group_scores: tuple[StreamingGroupScores, ...]
    qwen_forward_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.group_scores) < 2 or len(self.group_sample_ids) != len(
            self.group_scores
        ):
            raise ValueError("direct multi-group scores require at least two groups")
        if self.group_sample_ids != tuple(
            scores.sample_ids for scores in self.group_scores
        ):
            raise ValueError("multi-group sample identities and score blocks differ")
        flattened = tuple(
            sample_id
            for sample_ids in self.group_sample_ids
            for sample_id in sample_ids
        )
        if len(set(flattened)) != len(flattened):
            raise ValueError("direct multi-group sample identities must be unique")
        if not self.qwen_forward_batch_sizes or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > len(self.group_scores)
            for size in self.qwen_forward_batch_sizes
        ):
            raise ValueError("multi-group Qwen batch sizes must be in [1, group_count]")
        expected_cells = sum(
            len(scores.sample_ids) ** 2 for scores in self.group_scores
        )
        if sum(self.qwen_forward_batch_sizes) != expected_cells:
            raise ValueError("multi-group Qwen batches do not cover every local cell")
        reference = self.group_scores[0].score_matrix
        if any(
            scores.score_matrix.device != reference.device
            or scores.score_matrix.dtype != reference.dtype
            for scores in self.group_scores[1:]
        ):
            raise ValueError("multi-group score blocks must share device and dtype")


@dataclass(frozen=True, slots=True)
class StreamingGlobalNormalization:
    """Global counts for one complete data-parallel accumulation window.

    FSDP/DDP averages synchronized gradients across ranks. Multiplying local
    numerators by ``data_parallel_world_size / global_count`` therefore yields
    the gradient of one global numerator divided by its global denominator.
    """

    matrix_valid_rows: int
    l_gen_samples: int
    data_parallel_world_size: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "matrix_valid_rows",
            "l_gen_samples",
            "data_parallel_world_size",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class StreamingBackwardMetrics:
    """Detached local metrics; trainer aggregation remains explicit."""

    matrix_ce_numerator: torch.Tensor
    l_gen_numerator: torch.Tensor
    norm_numerator: torch.Tensor | None
    local_row_count: int
    local_sample_count: int
    weighted_local_mean: torch.Tensor
    weighted_norm_local_mean: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _StreamingCell:
    group_index: int
    row_index: int
    column_index: int
    source: RepresentationVisualTensorBundle
    row: RepresentationReadoutRow
    candidate: RepresentationVisualTensorBundle
    blocked_attention_mask: torch.Tensor


def score_streaming_same_image_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
) -> StreamingGroupScores:
    """Run the detached score pass without retaining any K-squared graph."""

    _validate_execution_inputs(family_adapter, model, group)
    score_rows: list[torch.Tensor] = []
    diagonal_l_gen: list[torch.Tensor] = []
    evidence_counts: list[torch.Tensor] = []
    norm_losses: list[torch.Tensor] = []
    with torch.no_grad():
        for candidate in group.candidates:
            norm_losses.append(
                historical_sample_norm_loss(
                    candidate.visual.main,
                    group.source_visual.main,
                    candidate.visual.deepstack,
                    group.source_visual.deepstack,
                )
            )
        for row_index, row in enumerate(group.rows):
            blocked_mask = _blocked_evidence_attention_mask(row, group.source_visual)
            row_scores: list[torch.Tensor] = []
            for column_index, candidate in enumerate(group.candidates):
                losses = _forward_cell_losses(
                    family_adapter,
                    model,
                    source=group.source_visual,
                    row=row,
                    candidate=candidate.visual,
                    blocked_attention_mask=blocked_mask,
                )
                row_scores.append(losses.per_sample_summed_log_likelihood[0])
                if row_index == column_index:
                    diagonal_l_gen.append(losses.per_sample_token_mean_nll[0])
                    evidence_counts.append(losses.valid_token_counts[0])
            score_rows.append(torch.stack(row_scores))
    return StreamingGroupScores(
        sample_ids=tuple(row.sample_id for row in group.rows),
        score_matrix=torch.stack(score_rows).detach(),
        diagonal_l_gen=torch.stack(diagonal_l_gen).detach(),
        evidence_token_counts=torch.stack(evidence_counts).detach(),
        historical_norm=torch.stack(norm_losses).detach(),
    )


def score_streaming_same_image_groups(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    groups: Sequence[SameImageReadoutGroup],
) -> StreamingMultiGroupScores:
    """Score independent same-image blocks with cross-group Qwen cell batches.

    A batch wave contains at most one cell from each group. Compatibility
    bucketing may split a wave, but cells from one group are never combined
    into a larger cross-image Matrix-CE block.
    """

    materialized = _validate_multi_group_execution_inputs(
        family_adapter,
        model,
        groups,
    )
    score_cells: list[list[list[torch.Tensor | None]]] = [
        [[None for _ in group.candidates] for _ in group.rows] for group in materialized
    ]
    diagonal_l_gen: list[list[torch.Tensor | None]] = [
        [None for _ in group.rows] for group in materialized
    ]
    evidence_counts: list[list[torch.Tensor | None]] = [
        [None for _ in group.rows] for group in materialized
    ]
    norm_losses: list[list[torch.Tensor]] = [[] for _ in materialized]
    forward_batch_sizes: list[int] = []
    with torch.no_grad():
        for group_index, group in enumerate(materialized):
            for candidate in group.candidates:
                norm_losses[group_index].append(
                    historical_sample_norm_loss(
                        candidate.visual.main,
                        group.source_visual.main,
                        candidate.visual.deepstack,
                        group.source_visual.deepstack,
                    )
                )
        for wave in _multi_group_cell_waves(materialized):
            for compatible_cells in _partition_compatible_cells(wave):
                losses = _forward_cell_batch_losses(
                    family_adapter,
                    model,
                    compatible_cells,
                )
                forward_batch_sizes.append(len(compatible_cells))
                for batch_index, cell in enumerate(compatible_cells):
                    score_cells[cell.group_index][cell.row_index][cell.column_index] = (
                        losses.per_sample_summed_log_likelihood[batch_index]
                    )
                    if cell.row_index == cell.column_index:
                        diagonal_l_gen[cell.group_index][cell.row_index] = (
                            losses.per_sample_token_mean_nll[batch_index]
                        )
                        evidence_counts[cell.group_index][cell.row_index] = (
                            losses.valid_token_counts[batch_index]
                        )

    group_scores = tuple(
        StreamingGroupScores(
            sample_ids=tuple(row.sample_id for row in group.rows),
            score_matrix=_stack_complete_matrix(
                score_cells[group_index],
                name="multi-group score matrix",
            ).detach(),
            diagonal_l_gen=_stack_complete_vector(
                diagonal_l_gen[group_index],
                name="multi-group diagonal L_gen",
            ).detach(),
            evidence_token_counts=_stack_complete_vector(
                evidence_counts[group_index],
                name="multi-group evidence counts",
            ).detach(),
            historical_norm=torch.stack(norm_losses[group_index]).detach(),
        )
        for group_index, group in enumerate(materialized)
    )
    return StreamingMultiGroupScores(
        group_sample_ids=tuple(scores.sample_ids for scores in group_scores),
        group_scores=group_scores,
        qwen_forward_batch_sizes=tuple(forward_batch_sizes),
    )


def backward_streaming_same_image_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
    scores: StreamingGroupScores,
    *,
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> StreamingBackwardMetrics:
    """Recompute one cell at a time and backpropagate the exact global objective.

    The first pass determines the Matrix-CE derivative with respect to each
    scalar score. This second pass recomputes and immediately releases each
    frozen-Qwen cell graph. Cell gradients stop at the candidate main-D and
    DeepStack outputs; they are accumulated there, then each candidate's TGVF
    Adapter graph is traversed exactly once. This is both memory bounded and
    compatible with FSDP2 pre/post-backward hooks.
    """

    _validate_execution_inputs(family_adapter, model, group)
    if not isinstance(scores, StreamingGroupScores):
        raise TypeError("scores must be StreamingGroupScores")
    if not isinstance(
        objective, (RepresentationObjectiveConfig, RepresentationObjectiveConfigV2)
    ):
        raise TypeError("objective must be a representation objective config")
    if not isinstance(normalization, StreamingGlobalNormalization):
        raise TypeError("normalization must be StreamingGlobalNormalization")
    sample_ids = tuple(row.sample_id for row in group.rows)
    if scores.sample_ids != sample_ids:
        raise ValueError("streaming scores belong to a different group/order")
    size = len(sample_ids)
    if normalization.matrix_valid_rows < size or normalization.l_gen_samples < size:
        raise ValueError("global normalization counts cannot be smaller than one group")

    probabilities = torch.softmax(scores.score_matrix.float(), dim=-1).to(
        dtype=scores.score_matrix.dtype
    )
    score_gradients = probabilities.clone()
    diagonal = torch.arange(size, device=score_gradients.device)
    score_gradients[diagonal, diagonal] -= 1
    score_gradients = score_gradients * (
        objective.matrix_ce_weight
        * normalization.data_parallel_world_size
        / normalization.matrix_valid_rows
    )
    l_gen_gradient = (
        objective.l_gen_weight
        * normalization.data_parallel_world_size
        / normalization.l_gen_samples
    )

    candidate_tensors = tuple(
        (candidate.visual.main, *candidate.visual.deepstack)
        for candidate in group.candidates
    )
    for tensors in candidate_tensors:
        if any(not tensor.requires_grad for tensor in tensors):
            raise ValueError(
                "every candidate main-D/DeepStack output must retain its Adapter graph"
            )
    accumulated_candidate_gradients = [
        [torch.zeros_like(tensor) for tensor in tensors]
        for tensors in candidate_tensors
    ]
    if isinstance(objective, RepresentationObjectiveConfigV2):
        for column_index, (candidate, tensors) in enumerate(
            zip(group.candidates, candidate_tensors, strict=True)
        ):
            live_norm = historical_sample_norm_loss(
                candidate.visual.main,
                group.source_visual.main,
                candidate.visual.deepstack,
                group.source_visual.deepstack,
            )
            if not torch.equal(
                live_norm.detach(), scores.historical_norm[column_index]
            ):
                raise RuntimeError(
                    "deterministic streaming recompute changed a norm value"
                )
            if objective.norm_weight:
                norm_surrogate = live_norm * (
                    objective.norm_weight
                    * normalization.data_parallel_world_size
                    / normalization.l_gen_samples
                )
                norm_gradients = torch.autograd.grad(
                    norm_surrogate,
                    tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for accumulator, gradient in zip(
                    accumulated_candidate_gradients[column_index],
                    norm_gradients,
                    strict=True,
                ):
                    accumulator.add_(gradient.detach())
    for row_index, row in enumerate(group.rows):
        blocked_mask = _blocked_evidence_attention_mask(row, group.source_visual)
        for column_index, candidate in enumerate(group.candidates):
            losses = _forward_cell_losses(
                family_adapter,
                model,
                source=group.source_visual,
                row=row,
                candidate=candidate.visual,
                blocked_attention_mask=blocked_mask,
            )
            live_score = losses.per_sample_summed_log_likelihood[0]
            expected_score = scores.score_matrix[row_index, column_index]
            if not torch.equal(live_score.detach(), expected_score):
                raise RuntimeError(
                    "deterministic streaming recompute changed a Matrix-CE score"
                )
            surrogate = live_score * score_gradients[row_index, column_index]
            if row_index == column_index and objective.l_gen_weight:
                live_l_gen = losses.per_sample_token_mean_nll[0]
                if not torch.equal(
                    live_l_gen.detach(), scores.diagonal_l_gen[row_index]
                ):
                    raise RuntimeError(
                        "deterministic streaming recompute changed an L_gen value"
                    )
                surrogate = surrogate + live_l_gen * l_gen_gradient
            gradients = torch.autograd.grad(
                surrogate,
                candidate_tensors[column_index],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            for accumulator, gradient in zip(
                accumulated_candidate_gradients[column_index],
                gradients,
                strict=True,
            ):
                accumulator.add_(gradient.detach())

    for tensors, gradients in zip(
        candidate_tensors,
        accumulated_candidate_gradients,
        strict=True,
    ):
        torch.autograd.backward(
            tensors,
            grad_tensors=tuple(gradients),
            retain_graph=False,
            create_graph=False,
        )
    _backward_collective_padding(group.collective_padding)

    labels = torch.arange(size, device=scores.score_matrix.device)
    matrix_numerator = F.cross_entropy(scores.score_matrix, labels, reduction="sum")
    l_gen_numerator = scores.diagonal_l_gen.sum()
    norm_numerator = (
        scores.historical_norm.sum()
        if isinstance(objective, RepresentationObjectiveConfigV2)
        else None
    )
    weighted_norm_local_mean = (
        None
        if norm_numerator is None
        else norm_numerator / size * objective.norm_weight
    )
    weighted_local_mean = (
        matrix_numerator / size * objective.matrix_ce_weight
        + l_gen_numerator / size * objective.l_gen_weight
    )
    if weighted_norm_local_mean is not None:
        weighted_local_mean = weighted_local_mean + weighted_norm_local_mean
    return StreamingBackwardMetrics(
        matrix_ce_numerator=matrix_numerator,
        l_gen_numerator=l_gen_numerator,
        norm_numerator=norm_numerator,
        local_row_count=size,
        local_sample_count=size,
        weighted_local_mean=weighted_local_mean,
        weighted_norm_local_mean=weighted_norm_local_mean,
    )


def backward_streaming_same_image_groups(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    groups: Sequence[SameImageReadoutGroup],
    scores: StreamingMultiGroupScores,
    *,
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> StreamingBackwardMetrics:
    """Backpropagate one globally normalized objective over separate CE blocks."""

    materialized = _validate_multi_group_execution_inputs(
        family_adapter,
        model,
        groups,
    )
    if not isinstance(scores, StreamingMultiGroupScores):
        raise TypeError("scores must be StreamingMultiGroupScores")
    if not isinstance(
        objective, (RepresentationObjectiveConfig, RepresentationObjectiveConfigV2)
    ):
        raise TypeError("objective must be a representation objective config")
    if not isinstance(normalization, StreamingGlobalNormalization):
        raise TypeError("normalization must be StreamingGlobalNormalization")
    sample_ids = tuple(
        tuple(row.sample_id for row in group.rows) for group in materialized
    )
    if scores.group_sample_ids != sample_ids:
        raise ValueError("multi-group streaming scores belong to another group/order")
    local_rows = sum(len(group.rows) for group in materialized)
    if (
        normalization.matrix_valid_rows < local_rows
        or normalization.l_gen_samples < local_rows
    ):
        raise ValueError(
            "global normalization counts cannot be smaller than all local groups"
        )

    score_gradients: list[torch.Tensor] = []
    for group_scores in scores.group_scores:
        size = len(group_scores.sample_ids)
        probabilities = torch.softmax(group_scores.score_matrix.float(), dim=-1).to(
            dtype=group_scores.score_matrix.dtype
        )
        gradients = probabilities.clone()
        diagonal = torch.arange(size, device=gradients.device)
        gradients[diagonal, diagonal] -= 1
        gradients.mul_(
            objective.matrix_ce_weight
            * normalization.data_parallel_world_size
            / normalization.matrix_valid_rows
        )
        score_gradients.append(gradients)
    l_gen_gradient = (
        objective.l_gen_weight
        * normalization.data_parallel_world_size
        / normalization.l_gen_samples
    )

    candidate_tensors = tuple(
        tuple(
            (candidate.visual.main, *candidate.visual.deepstack)
            for candidate in group.candidates
        )
        for group in materialized
    )
    for group_tensors in candidate_tensors:
        for tensors in group_tensors:
            if any(not tensor.requires_grad for tensor in tensors):
                raise ValueError(
                    "every candidate main-D/DeepStack output must retain its "
                    "Adapter graph"
                )
    accumulated_candidate_gradients = [
        [[torch.zeros_like(tensor) for tensor in tensors] for tensors in group_tensors]
        for group_tensors in candidate_tensors
    ]

    if isinstance(objective, RepresentationObjectiveConfigV2):
        for group_index, (group, group_tensors, group_scores) in enumerate(
            zip(materialized, candidate_tensors, scores.group_scores, strict=True)
        ):
            for column_index, (candidate, tensors) in enumerate(
                zip(group.candidates, group_tensors, strict=True)
            ):
                live_norm = historical_sample_norm_loss(
                    candidate.visual.main,
                    group.source_visual.main,
                    candidate.visual.deepstack,
                    group.source_visual.deepstack,
                )
                if not torch.equal(
                    live_norm.detach(), group_scores.historical_norm[column_index]
                ):
                    raise RuntimeError(
                        "deterministic multi-group recompute changed a norm value"
                    )
                norm_surrogate = live_norm * (
                    objective.norm_weight
                    * normalization.data_parallel_world_size
                    / normalization.l_gen_samples
                )
                norm_gradients = torch.autograd.grad(
                    norm_surrogate,
                    tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for accumulator, gradient in zip(
                    accumulated_candidate_gradients[group_index][column_index],
                    norm_gradients,
                    strict=True,
                ):
                    accumulator.add_(gradient.detach())

    backward_batch_sizes: list[int] = []
    for wave in _multi_group_cell_waves(materialized):
        for compatible_cells in _partition_compatible_cells(wave):
            losses = _forward_cell_batch_losses(
                family_adapter,
                model,
                compatible_cells,
            )
            backward_batch_sizes.append(len(compatible_cells))
            surrogate = losses.per_sample_summed_log_likelihood.new_zeros(())
            flat_candidate_tensors: list[torch.Tensor] = []
            for batch_index, cell in enumerate(compatible_cells):
                group_scores = scores.group_scores[cell.group_index]
                live_score = losses.per_sample_summed_log_likelihood[batch_index]
                expected_score = group_scores.score_matrix[
                    cell.row_index, cell.column_index
                ]
                if not torch.equal(live_score.detach(), expected_score):
                    raise RuntimeError(
                        "deterministic multi-group recompute changed a Matrix-CE score"
                    )
                surrogate = (
                    surrogate
                    + live_score
                    * score_gradients[cell.group_index][
                        cell.row_index, cell.column_index
                    ]
                )
                if cell.row_index == cell.column_index and objective.l_gen_weight:
                    live_l_gen = losses.per_sample_token_mean_nll[batch_index]
                    if not torch.equal(
                        live_l_gen.detach(),
                        group_scores.diagonal_l_gen[cell.row_index],
                    ):
                        raise RuntimeError(
                            "deterministic multi-group recompute changed an L_gen value"
                        )
                    surrogate = surrogate + live_l_gen * l_gen_gradient
                flat_candidate_tensors.extend(
                    candidate_tensors[cell.group_index][cell.column_index]
                )
            gradients = torch.autograd.grad(
                surrogate,
                tuple(flat_candidate_tensors),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            cursor = 0
            for cell in compatible_cells:
                accumulators = accumulated_candidate_gradients[cell.group_index][
                    cell.column_index
                ]
                for accumulator in accumulators:
                    accumulator.add_(gradients[cursor].detach())
                    cursor += 1
            if cursor != len(gradients):
                raise RuntimeError("multi-group candidate gradient partition drifted")
    if tuple(backward_batch_sizes) != scores.qwen_forward_batch_sizes:
        raise RuntimeError("multi-group Qwen compatibility schedule changed")

    adapter_outputs: list[torch.Tensor] = []
    adapter_output_gradients: list[torch.Tensor] = []
    for group_tensors, group_gradients in zip(
        candidate_tensors,
        accumulated_candidate_gradients,
        strict=True,
    ):
        for tensors, gradients in zip(group_tensors, group_gradients, strict=True):
            adapter_outputs.extend(tensors)
            adapter_output_gradients.extend(gradients)
    for group in materialized:
        for padding in group.collective_padding:
            tensors = (padding.main, *padding.deepstack)
            if any(not tensor.requires_grad for tensor in tensors):
                raise ValueError(
                    "training collective padding must retain every Adapter graph"
                )
            adapter_outputs.extend(tensors)
            adapter_output_gradients.extend(
                torch.zeros_like(tensor) for tensor in tensors
            )
    torch.autograd.backward(
        tuple(adapter_outputs),
        grad_tensors=tuple(adapter_output_gradients),
        retain_graph=False,
        create_graph=False,
    )

    matrix_numerators = tuple(
        F.cross_entropy(
            group_scores.score_matrix,
            torch.arange(
                len(group_scores.sample_ids),
                device=group_scores.score_matrix.device,
            ),
            reduction="sum",
        )
        for group_scores in scores.group_scores
    )
    matrix_numerator = torch.stack(matrix_numerators).sum()
    l_gen_numerator = torch.stack(
        tuple(group_scores.diagonal_l_gen.sum() for group_scores in scores.group_scores)
    ).sum()
    norm_numerator = (
        torch.stack(
            tuple(
                group_scores.historical_norm.sum()
                for group_scores in scores.group_scores
            )
        ).sum()
        if isinstance(objective, RepresentationObjectiveConfigV2)
        else None
    )
    weighted_norm_local_mean = (
        None
        if norm_numerator is None
        else norm_numerator / local_rows * objective.norm_weight
    )
    weighted_local_mean = (
        matrix_numerator / local_rows * objective.matrix_ce_weight
        + l_gen_numerator / local_rows * objective.l_gen_weight
    )
    if weighted_norm_local_mean is not None:
        weighted_local_mean = weighted_local_mean + weighted_norm_local_mean
    return StreamingBackwardMetrics(
        matrix_ce_numerator=matrix_numerator,
        l_gen_numerator=l_gen_numerator,
        norm_numerator=norm_numerator,
        local_row_count=local_rows,
        local_sample_count=local_rows,
        weighted_local_mean=weighted_local_mean,
        weighted_norm_local_mean=weighted_norm_local_mean,
    )


def _validate_multi_group_execution_inputs(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    groups: Sequence[SameImageReadoutGroup],
) -> tuple[SameImageReadoutGroup, ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError("groups must be a sequence of same-image readout groups")
    materialized = tuple(groups)
    if len(materialized) < 2:
        raise ValueError("direct multi-group execution requires at least two groups")
    for group in materialized:
        _validate_execution_inputs(family_adapter, model, group)
    sample_ids = tuple(row.sample_id for group in materialized for row in group.rows)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("direct multi-group execution requires unique sample IDs")
    return materialized


def _multi_group_cell_waves(
    groups: tuple[SameImageReadoutGroup, ...],
) -> tuple[tuple[_StreamingCell, ...], ...]:
    maximum_size = max(len(group.rows) for group in groups)
    waves: list[tuple[_StreamingCell, ...]] = []
    for row_index in range(maximum_size):
        for column_index in range(maximum_size):
            cells = []
            for group_index, group in enumerate(groups):
                size = len(group.rows)
                if row_index >= size or column_index >= size:
                    continue
                row = group.rows[row_index]
                cells.append(
                    _StreamingCell(
                        group_index=group_index,
                        row_index=row_index,
                        column_index=column_index,
                        source=group.source_visual,
                        row=row,
                        candidate=group.candidates[column_index].visual,
                        blocked_attention_mask=_blocked_evidence_attention_mask(
                            row,
                            group.source_visual,
                        ),
                    )
                )
            if cells:
                waves.append(tuple(cells))
    return tuple(waves)


def _partition_compatible_cells(
    cells: tuple[_StreamingCell, ...],
) -> tuple[tuple[_StreamingCell, ...], ...]:
    buckets: dict[tuple[object, ...], list[_StreamingCell]] = {}
    for cell in cells:
        request = _cell_request(
            cell.source,
            cell.row,
            cell.candidate,
            cell.blocked_attention_mask,
        )
        buckets.setdefault(_request_batch_key(request), []).append(cell)
    return tuple(tuple(bucket) for bucket in buckets.values())


def _request_batch_key(request: InjectedForwardRequest) -> tuple[object, ...]:
    position_batch_dimension = 0 if request.position_ids.ndim == 2 else 1
    position_shape = tuple(
        dimension
        for index, dimension in enumerate(request.position_ids.shape)
        if index != position_batch_dimension
    )
    block_keys = tuple(
        (
            block.kind,
            block.positions,
            tuple(block.embeddings.shape[1:]),
            block.embeddings.dtype,
            block.embeddings.device,
            tuple(
                (
                    positions,
                    tuple(branch.shape[1:]),
                    branch.dtype,
                    branch.device,
                )
                for branch, positions in zip(
                    block.deepstack,
                    block.deepstack_positions,
                    strict=True,
                )
            ),
        )
        for block in request.visual_blocks
    )
    return (
        tuple(request.input_ids.shape[1:]),
        request.input_ids.dtype,
        request.input_ids.device,
        tuple(request.attention_mask.shape[1:]),
        request.attention_mask.dtype,
        request.attention_mask.device,
        request.position_ids.ndim,
        position_shape,
        request.position_ids.dtype,
        request.position_ids.device,
        block_keys,
    )


def _forward_cell_batch_losses(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    cells: tuple[_StreamingCell, ...],
) -> CausalEvidenceLosses:
    if not cells:
        raise ValueError("a Qwen cell batch cannot be empty")
    requests = tuple(
        _cell_request(
            cell.source,
            cell.row,
            cell.candidate,
            cell.blocked_attention_mask,
        )
        for cell in cells
    )
    reference_key = _request_batch_key(requests[0])
    if any(_request_batch_key(request) != reference_key for request in requests[1:]):
        raise ValueError("Qwen cell batch contains incompatible requests")
    position_batch_dimension = 0 if requests[0].position_ids.ndim == 2 else 1
    visual_blocks = tuple(
        InjectedVisualBlock(
            kind=requests[0].visual_blocks[block_index].kind,
            positions=requests[0].visual_blocks[block_index].positions,
            embeddings=torch.cat(
                tuple(
                    request.visual_blocks[block_index].embeddings
                    for request in requests
                ),
                dim=0,
            ),
            deepstack=tuple(
                torch.cat(
                    tuple(
                        request.visual_blocks[block_index].deepstack[branch_index]
                        for request in requests
                    ),
                    dim=0,
                )
                for branch_index in range(
                    len(requests[0].visual_blocks[block_index].deepstack)
                )
            ),
            deepstack_positions=requests[0]
            .visual_blocks[block_index]
            .deepstack_positions,
        )
        for block_index in range(len(requests[0].visual_blocks))
    )
    batched_request = InjectedForwardRequest(
        input_ids=torch.cat(tuple(request.input_ids for request in requests), dim=0),
        attention_mask=torch.cat(
            tuple(request.attention_mask for request in requests),
            dim=0,
        ),
        position_ids=torch.cat(
            tuple(request.position_ids for request in requests),
            dim=position_batch_dimension,
        ),
        visual_blocks=visual_blocks,
        use_cache=False,
    )
    result = family_adapter.forward_injected(model, batched_request)
    labels = torch.tensor(
        tuple(cell.row.supervision.labels for cell in cells),
        dtype=torch.long,
        device=result.logits.device,
    )
    return causal_evidence_losses(result.logits, labels)


def _cell_request(
    source: RepresentationVisualTensorBundle,
    row: RepresentationReadoutRow,
    candidate: RepresentationVisualTensorBundle,
    blocked_attention_mask: torch.Tensor,
) -> InjectedForwardRequest:
    source_block = InjectedVisualBlock(
        kind="source_image",
        positions=row.source_positions,
        embeddings=source.main,
        deepstack=source.deepstack,
        deepstack_positions=tuple(row.source_positions for _ in source.deepstack),
    )
    candidate_block = InjectedVisualBlock(
        kind="focused_d",
        positions=row.d_positions,
        embeddings=candidate.main,
        deepstack=candidate.deepstack,
        deepstack_positions=tuple(row.d_positions for _ in candidate.deepstack),
    )
    return InjectedForwardRequest(
        input_ids=row.input_ids,
        attention_mask=blocked_attention_mask,
        position_ids=row.position_ids,
        visual_blocks=(source_block, candidate_block),
        use_cache=False,
    )


def _stack_complete_vector(
    values: Sequence[torch.Tensor | None],
    *,
    name: str,
) -> torch.Tensor:
    if not values or any(value is None for value in values):
        raise RuntimeError(f"{name} did not cover every expected cell")
    return torch.stack(tuple(value for value in values if value is not None))


def _stack_complete_matrix(
    rows: Sequence[Sequence[torch.Tensor | None]],
    *,
    name: str,
) -> torch.Tensor:
    if not rows:
        raise RuntimeError(f"{name} is empty")
    return torch.stack(tuple(_stack_complete_vector(row, name=name) for row in rows))


def _backward_collective_padding(
    padding: tuple[RepresentationVisualTensorBundle, ...],
) -> None:
    """Traverse each padding Adapter graph with an exact zero gradient.

    Padding is deliberately absent from every score/loss term.  Its sole role
    is to issue the same number of composable-FSDP backward collectives as the
    rank whose local same-image group has the largest real K.
    """

    for visual in padding:
        tensors = (visual.main, *visual.deepstack)
        if any(not tensor.requires_grad for tensor in tensors):
            raise ValueError(
                "training collective padding must retain every Adapter graph"
            )
        torch.autograd.backward(
            tensors,
            grad_tensors=tuple(torch.zeros_like(tensor) for tensor in tensors),
            retain_graph=False,
            create_graph=False,
        )


def _validate_execution_inputs(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
) -> None:
    if not isinstance(family_adapter, QwenVLMFamilyAdapter):
        raise TypeError("family_adapter must be QwenVLMFamilyAdapter")
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("group must be SameImageReadoutGroup")
    assert_frozen_deterministic_readout_model(model)
    if any(
        row.supervision.family != family_adapter.capabilities.family
        for row in group.rows
    ):
        raise ValueError("readout supervision belongs to a different Qwen family")
    if len(group.source_visual.deepstack) != (
        family_adapter.capabilities.deepstack_branch_count
    ):
        raise ValueError("source DeepStack branches differ from family capability")


def _blocked_evidence_attention_mask(
    row: RepresentationReadoutRow,
    source: RepresentationVisualTensorBundle,
) -> torch.Tensor:
    if len(row.source_positions) != source.main.shape[1]:
        raise ValueError("source visual positions do not match source tokens")
    first_evidence = row.supervision.evidence_token_positions[0]
    final_evidence = row.supervision.evidence_token_positions[-1]
    if first_evidence <= 0:
        raise ValueError("evidence must have a preceding causal prediction query")
    return build_original_image_key_block_mask(
        attention_mask=row.attention_mask,
        original_image_token_indices=torch.tensor(
            row.source_positions,
            dtype=torch.long,
            device=row.attention_mask.device,
        ),
        block_query_start=first_evidence - 1,
        block_query_end=final_evidence,
        dtype=source.main.dtype,
    )


def _forward_cell_losses(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    *,
    source: RepresentationVisualTensorBundle,
    row: RepresentationReadoutRow,
    candidate: RepresentationVisualTensorBundle,
    blocked_attention_mask: torch.Tensor,
) -> CausalEvidenceLosses:
    source_block = InjectedVisualBlock(
        kind="source_image",
        positions=row.source_positions,
        embeddings=source.main,
        deepstack=source.deepstack,
        deepstack_positions=tuple(row.source_positions for _ in source.deepstack),
    )
    candidate_block = InjectedVisualBlock(
        kind="focused_d",
        positions=row.d_positions,
        embeddings=candidate.main,
        deepstack=candidate.deepstack,
        deepstack_positions=tuple(row.d_positions for _ in candidate.deepstack),
    )
    result = family_adapter.forward_injected(
        model,
        InjectedForwardRequest(
            input_ids=row.input_ids,
            attention_mask=blocked_attention_mask,
            position_ids=row.position_ids,
            visual_blocks=(source_block, candidate_block),
            use_cache=False,
        ),
    )
    labels = torch.tensor(
        row.supervision.labels,
        dtype=torch.long,
        device=result.logits.device,
    ).unsqueeze(0)
    return causal_evidence_losses(result.logits, labels)


__all__ = [
    "StreamingBackwardMetrics",
    "StreamingGlobalNormalization",
    "StreamingGroupScores",
    "StreamingMultiGroupScores",
    "backward_streaming_same_image_group",
    "backward_streaming_same_image_groups",
    "score_streaming_same_image_group",
    "score_streaming_same_image_groups",
]
