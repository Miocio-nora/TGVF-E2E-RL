"""Memory-bounded Matrix-CE/readability execution over one same-image group."""

from __future__ import annotations

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
    "backward_streaming_same_image_group",
    "score_streaming_same_image_group",
]
