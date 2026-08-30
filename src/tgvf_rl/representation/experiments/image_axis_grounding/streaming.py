"""Isolated image-axis grounding VJP over the accepted RP66 objective.

For every eligible anchor row, the frozen reader compares the anchor-image D
against an exact-grid wrong-image D while keeping the anchor transcript and
source-image block fixed.  Source-image keys are blocked at evidence queries,
so the two-way CE can only prefer the D that contains image-matched evidence.

The Qwen graph is reduced to detached gradients at live Adapter outputs.  The
correct-image gradients are merged into the accepted streaming payload.  The
donor gradients enter as hooks on loss-excluded temporary collective-padding
roots, allowing the core backward to traverse donor and anchor Adapter graphs
in one call.  This is required for the final-microstep FSDP2 sync/reshard
contract; a separate donor backward could be mistaken for the last backward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from tgvf_rl.qwen.base import QwenVLMFamilyAdapter
from tgvf_rl.representation.training.losses import (
    MatrixCEScoreMode,
    matrix_ce_cell_scores,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
)
from tgvf_rl.representation.training.readout import (
    SameImageReadoutGroup,
    assert_frozen_deterministic_readout_model,
)
from tgvf_rl.representation.training.streaming import (
    StreamingBackwardMetrics,
    StreamingGlobalNormalization,
    StreamingGroupScores,
    _StreamingCandidateGradients,
    _StreamingCell,
    _StreamingRow,
    _blocked_evidence_attention_mask,
    _candidate_output_tensors,
    _forward_cell_batch_losses,
    _partition_compatible_rows,
    backward_streaming_same_image_group,
    score_streaming_same_image_group,
)

from .native_pipeline import ImageAxisGroundingGroup
from .objective import ImageAxisGroundingObjectiveConfig


@dataclass(frozen=True, slots=True)
class ImageAxisGlobalNormalization:
    """Legacy global counts plus the independently masked image-axis count."""

    legacy: StreamingGlobalNormalization
    image_axis_valid_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.legacy, StreamingGlobalNormalization):
            raise TypeError("legacy normalization must be explicit")
        if (
            isinstance(self.image_axis_valid_rows, bool)
            or not isinstance(self.image_axis_valid_rows, int)
            or self.image_axis_valid_rows < 0
        ):
            raise ValueError("image_axis_valid_rows must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ImageAxisStreamingMetrics:
    """Detached local numerators; distributed aggregation remains trainer-owned."""

    legacy: StreamingBackwardMetrics
    image_axis_numerator: torch.Tensor
    local_image_axis_row_count: int
    correct_score_sum: torch.Tensor
    wrong_score_sum: torch.Tensor
    correct_top1_count: int
    image_axis_qwen_forward_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.legacy, StreamingBackwardMetrics):
            raise TypeError("image-axis metrics require legacy streaming metrics")
        for field_name in (
            "image_axis_numerator",
            "correct_score_sum",
            "wrong_score_sum",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 0
                or value.requires_grad
            ):
                raise ValueError(f"{field_name} must be a detached scalar tensor")
        for field_name in ("local_image_axis_row_count", "correct_top1_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.correct_top1_count > self.local_image_axis_row_count:
            raise ValueError("correct top-1 count cannot exceed image-axis rows")
        if self.local_image_axis_row_count == 0:
            if self.correct_top1_count or self.image_axis_qwen_forward_batch_sizes:
                raise ValueError("masked image-axis metrics cannot contain Qwen work")
        elif not self.image_axis_qwen_forward_batch_sizes:
            raise ValueError("eligible image-axis metrics require Qwen telemetry")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.image_axis_qwen_forward_batch_sizes
        ):
            raise ValueError("image-axis Qwen batch sizes must be positive integers")


@dataclass(frozen=True, slots=True)
class _ImageAxisScoreMaterialization:
    correct_gradients: tuple[tuple[torch.Tensor, ...], ...]
    donor_gradients: tuple[tuple[torch.Tensor, ...], ...]
    numerator: torch.Tensor
    correct_scores: torch.Tensor
    wrong_scores: torch.Tensor
    qwen_forward_batch_sizes: tuple[int, ...]


def backward_image_axis_grounding_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: ImageAxisGroundingGroup,
    *,
    image_axis_objective: ImageAxisGroundingObjectiveConfig,
    legacy_objective: RepresentationObjectiveConfigLike,
    normalization: ImageAxisGlobalNormalization,
) -> ImageAxisStreamingMetrics:
    """Backpropagate legacy plus image-axis terms through one Adapter boundary."""

    _validate_inputs(
        group,
        image_axis_objective=image_axis_objective,
        legacy_objective=legacy_objective,
        normalization=normalization,
    )
    assert_frozen_deterministic_readout_model(model)

    legacy_scores = score_streaming_same_image_group(
        family_adapter,
        model,
        group.base,
        objective=legacy_objective,
        normalization=normalization.legacy,
    )
    local_image_rows = sum(group.image_axis_row_mask)
    if local_image_rows:
        image_axis = _score_image_axis_rows(
            family_adapter,
            model,
            group,
            image_axis_objective=image_axis_objective,
            normalization=normalization,
        )
        merged_scores = _merge_correct_boundary_gradients(
            group.base,
            legacy_scores,
            image_axis.correct_gradients,
        )
    else:
        zero = legacy_scores.score_matrix.new_zeros(())
        image_axis = _ImageAxisScoreMaterialization(
            correct_gradients=_zero_group_gradients(group.base),
            donor_gradients=_zero_group_gradients(group.donor),
            numerator=zero,
            correct_scores=zero.new_empty((0,)),
            wrong_scores=zero.new_empty((0,)),
            qwen_forward_batch_sizes=(),
        )
        merged_scores = legacy_scores

    legacy = _single_boundary_backward(
        family_adapter,
        model,
        group,
        merged_scores,
        donor_gradients=image_axis.donor_gradients,
        donor_active=bool(local_image_rows),
        legacy_objective=legacy_objective,
        normalization=normalization.legacy,
    )
    return ImageAxisStreamingMetrics(
        legacy=legacy,
        image_axis_numerator=image_axis.numerator.sum().detach(),
        local_image_axis_row_count=local_image_rows,
        correct_score_sum=image_axis.correct_scores.sum().detach(),
        wrong_score_sum=image_axis.wrong_scores.sum().detach(),
        correct_top1_count=int(
            (
                torch.stack(
                    (image_axis.correct_scores, image_axis.wrong_scores),
                    dim=1,
                ).argmax(dim=1)
                == 0
            )
            .sum()
            .item()
        ),
        image_axis_qwen_forward_batch_sizes=image_axis.qwen_forward_batch_sizes,
    )


def _score_image_axis_rows(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: ImageAxisGroundingGroup,
    *,
    image_axis_objective: ImageAxisGroundingObjectiveConfig,
    normalization: ImageAxisGlobalNormalization,
) -> _ImageAxisScoreMaterialization:
    """Materialize balanced two-way CE and its correct/donor boundary VJPs."""

    local_rows = sum(group.image_axis_row_mask)
    if local_rows <= 0:
        raise ValueError("image-axis scoring requires an eligible group")
    if normalization.image_axis_valid_rows < local_rows:
        raise ValueError("global image-axis count is smaller than the local group")

    logical_rows = tuple(
        _StreamingRow(
            group_index=0,
            row_index=row_index,
            cells=(
                _StreamingCell(
                    group_index=0,
                    row_index=row_index,
                    column_index=0,
                    source=group.base.source_visual,
                    row=row,
                    candidate=correct.visual,
                    blocked_attention_mask=_blocked_evidence_attention_mask(
                        row,
                        group.base.source_visual,
                    ),
                ),
                _StreamingCell(
                    group_index=0,
                    row_index=row_index,
                    column_index=1,
                    source=group.base.source_visual,
                    row=row,
                    candidate=donor.visual,
                    blocked_attention_mask=_blocked_evidence_attention_mask(
                        row,
                        group.base.source_visual,
                    ),
                ),
            ),
        )
        for row_index, (row, correct, donor, active) in enumerate(
            zip(
                group.base.rows,
                group.base.candidates,
                group.donor.candidates,
                group.image_axis_row_mask,
                strict=True,
            )
        )
        if active
    )
    correct_accumulators = [
        [torch.zeros_like(tensor) for tensor in _candidate_output_tensors(candidate.visual)]
        for candidate in group.base.candidates
    ]
    donor_accumulators = [
        [torch.zeros_like(tensor) for tensor in _candidate_output_tensors(candidate.visual)]
        for candidate in group.donor.candidates
    ]
    score_pairs: list[torch.Tensor | None] = [None] * len(group.base.rows)
    forward_batch_sizes: list[int] = []
    for compatible_rows in _partition_compatible_rows(logical_rows):
        cells = tuple(cell for logical_row in compatible_rows for cell in logical_row.cells)
        losses = _forward_cell_batch_losses(family_adapter, model, cells)
        cell_scores = matrix_ce_cell_scores(
            losses,
            mode=MatrixCEScoreMode.BALANCED,
            temperature=image_axis_objective.image_axis_temperature,
        )
        forward_batch_sizes.append(len(cells))
        row_terms: list[torch.Tensor] = []
        cursor = 0
        for logical_row in compatible_rows:
            row_scores = cell_scores[cursor : cursor + 2]
            if row_scores.shape != (2,):
                raise RuntimeError("image-axis row did not contain correct/wrong cells")
            score_pairs[logical_row.row_index] = row_scores.detach()
            row_terms.append(
                F.cross_entropy(
                    row_scores.unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=row_scores.device),
                    reduction="sum",
                )
            )
            cursor += 2
        if cursor != len(cells):
            raise RuntimeError("image-axis row partition drifted")
        surrogate = torch.stack(row_terms).sum() * (
            image_axis_objective.image_axis_matrix_weight
            * normalization.legacy.data_parallel_world_size
            / normalization.image_axis_valid_rows
        )
        outputs: list[torch.Tensor] = []
        output_slots: list[tuple[bool, int, int]] = []
        for logical_row in compatible_rows:
            row_index = logical_row.row_index
            for is_donor, candidate in (
                (False, group.base.candidates[row_index]),
                (True, group.donor.candidates[row_index]),
            ):
                for path_index, tensor in enumerate(
                    _candidate_output_tensors(candidate.visual)
                ):
                    outputs.append(tensor)
                    output_slots.append((is_donor, row_index, path_index))
        gradients = torch.autograd.grad(
            surrogate,
            tuple(outputs),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        for (is_donor, row_index, path_index), gradient in zip(
            output_slots,
            gradients,
            strict=True,
        ):
            destination = (
                donor_accumulators if is_donor else correct_accumulators
            )[row_index][path_index]
            destination.add_(gradient.detach())

    materialized_pairs = tuple(value for value in score_pairs if value is not None)
    if len(materialized_pairs) != local_rows:
        raise RuntimeError("image-axis scoring did not cover every eligible row")
    pair_matrix = torch.stack(materialized_pairs)
    numerator = F.cross_entropy(
        pair_matrix,
        torch.zeros(local_rows, dtype=torch.long, device=pair_matrix.device),
        reduction="sum",
    )
    return _ImageAxisScoreMaterialization(
        correct_gradients=tuple(
            tuple(gradient.detach() for gradient in values)
            for values in correct_accumulators
        ),
        donor_gradients=tuple(
            tuple(gradient.detach() for gradient in values)
            for values in donor_accumulators
        ),
        numerator=numerator.detach(),
        correct_scores=pair_matrix[:, 0].detach(),
        wrong_scores=pair_matrix[:, 1].detach(),
        qwen_forward_batch_sizes=tuple(forward_batch_sizes),
    )


def _merge_correct_boundary_gradients(
    base: SameImageReadoutGroup,
    legacy_scores: StreamingGroupScores,
    correct_gradients: tuple[tuple[torch.Tensor, ...], ...],
) -> StreamingGroupScores:
    """Merge correct-image VJP before core's independent norm VJP."""

    payloads = legacy_scores.candidate_output_gradients
    if payloads is None or len(payloads) != len(base.candidates):
        raise ValueError("legacy score payload does not cover every base candidate")
    if len(correct_gradients) != len(base.candidates):
        raise ValueError("image-axis correct gradients do not cover every candidate")
    merged: list[_StreamingCandidateGradients] = []
    for candidate, payload, image_values in zip(
        base.candidates,
        payloads,
        correct_gradients,
        strict=True,
    ):
        tensors = _candidate_output_tensors(candidate.visual)
        if len(tensors) != len(payload.weighted_readout) or len(tensors) != len(
            image_values
        ):
            raise ValueError("legacy/image-axis boundary path count changed")
        values: list[torch.Tensor] = []
        for tensor, legacy_value, image_value in zip(
            tensors,
            payload.weighted_readout,
            image_values,
            strict=True,
        ):
            _assert_gradient_contract(tensor, legacy_value)
            _assert_gradient_contract(tensor, image_value)
            values.append(legacy_value + image_value)
        merged.append(_StreamingCandidateGradients(weighted_readout=tuple(values)))
    return replace(legacy_scores, candidate_output_gradients=tuple(merged))


def _single_boundary_backward(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: ImageAxisGroundingGroup,
    legacy_scores: StreamingGroupScores,
    *,
    donor_gradients: tuple[tuple[torch.Tensor, ...], ...],
    donor_active: bool,
    legacy_objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> StreamingBackwardMetrics:
    """Inject donor VJP at padding roots and delegate one backward to core."""

    if len(donor_gradients) != len(group.donor.candidates):
        raise ValueError("donor gradients do not cover every donor candidate")
    donor_visuals = tuple(candidate.visual for candidate in group.donor.candidates)
    augmented = replace(
        group.base,
        collective_padding=(
            *group.base.collective_padding,
            *donor_visuals,
            *group.donor.collective_padding,
        ),
    )
    handles: list[torch.utils.hooks.RemovableHandle] = []
    hook_counts: list[list[int]] = []
    try:
        if donor_active:
            for candidate, gradients in zip(
                group.donor.candidates,
                donor_gradients,
                strict=True,
            ):
                tensors = _candidate_output_tensors(candidate.visual)
                if len(tensors) != len(gradients):
                    raise ValueError("donor boundary path count changed")
                for tensor, gradient in zip(tensors, gradients, strict=True):
                    _assert_gradient_contract(tensor, gradient)
                    count = [0]
                    hook_counts.append(count)

                    def inject(
                        incoming: torch.Tensor,
                        *,
                        extra: torch.Tensor = gradient,
                        calls: list[int] = count,
                    ) -> torch.Tensor:
                        calls[0] += 1
                        if calls[0] != 1:
                            raise RuntimeError("donor boundary hook fired more than once")
                        return incoming + extra

                    handles.append(tensor.register_hook(inject))
        legacy = backward_streaming_same_image_group(
            family_adapter,
            model,
            augmented,
            legacy_scores,
            objective=legacy_objective,
            normalization=normalization,
        )
        if any(calls[0] != 1 for calls in hook_counts):
            raise RuntimeError("donor boundary hook did not fire exactly once")
        return legacy
    finally:
        for handle in handles:
            handle.remove()


def _zero_group_gradients(
    group: SameImageReadoutGroup,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(
        tuple(torch.zeros_like(tensor) for tensor in _candidate_output_tensors(candidate.visual))
        for candidate in group.candidates
    )


def _validate_inputs(
    group: object,
    *,
    image_axis_objective: object,
    legacy_objective: object,
    normalization: object,
) -> None:
    if not isinstance(group, ImageAxisGroundingGroup):
        raise TypeError("image-axis backward requires ImageAxisGroundingGroup")
    if not isinstance(image_axis_objective, ImageAxisGroundingObjectiveConfig):
        raise TypeError("image-axis objective must be explicit")
    if not isinstance(legacy_objective, RepresentationObjectiveConfigV2):
        raise TypeError("image-axis experiment requires the norm-aware legacy objective")
    if not isinstance(normalization, ImageAxisGlobalNormalization):
        raise TypeError("image-axis normalization must be explicit")
    mask = group.image_axis_row_mask
    if len(mask) != len(group.base.rows) or any(type(value) is not bool for value in mask):
        raise ValueError("image-axis row mask must contain one bool per base row")
    if not (all(mask) or not any(mask)):
        raise ValueError("image-axis row mask must be group-homogeneous")
    local_rows = sum(mask)
    if normalization.image_axis_valid_rows < local_rows:
        raise ValueError("global image-axis count is smaller than this local group")
    if tuple(row.sample_id for row in group.base.rows) != tuple(
        candidate.sample_id for candidate in group.donor.candidates
    ):
        raise ValueError("base rows and donor candidates changed identity/order")
    if group.base.collective_candidate_count != group.donor.collective_candidate_count:
        raise ValueError("base/donor collective candidate counts differ")

    base_outputs = tuple(
        tensor
        for candidate in group.base.candidates
        for tensor in _candidate_output_tensors(candidate.visual)
    )
    donor_outputs = tuple(
        tensor
        for candidate in group.donor.candidates
        for tensor in _candidate_output_tensors(candidate.visual)
    )
    if any(not tensor.requires_grad for tensor in (*base_outputs, *donor_outputs)):
        raise ValueError("base and donor candidates must retain live Adapter graphs")
    identities = tuple(id(tensor) for tensor in (*base_outputs, *donor_outputs))
    if len(set(identities)) != len(identities):
        raise ValueError("base/donor Adapter output tensors cannot be shared")
    for correct, donor in zip(
        group.base.candidates,
        group.donor.candidates,
        strict=True,
    ):
        correct_tensors = _candidate_output_tensors(correct.visual)
        donor_tensors = _candidate_output_tensors(donor.visual)
        if len(correct_tensors) != len(donor_tensors):
            raise ValueError("base/donor main+DeepStack path counts differ")
        for correct_tensor, donor_tensor in zip(
            correct_tensors,
            donor_tensors,
            strict=True,
        ):
            if (
                correct_tensor.shape != donor_tensor.shape
                or correct_tensor.dtype != donor_tensor.dtype
                or correct_tensor.device != donor_tensor.device
            ):
                raise ValueError("base/donor visual tensor contracts differ")


def _assert_gradient_contract(tensor: torch.Tensor, gradient: torch.Tensor) -> None:
    if gradient.requires_grad:
        raise ValueError("boundary gradients must be detached")
    if (
        tensor.shape != gradient.shape
        or tensor.dtype != gradient.dtype
        or tensor.device != gradient.device
    ):
        raise ValueError("boundary gradient tensor contract changed")


__all__ = [
    "ImageAxisGlobalNormalization",
    "ImageAxisStreamingMetrics",
    "backward_image_axis_grounding_group",
]
