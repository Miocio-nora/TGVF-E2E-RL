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
    EVIDENCE_IGNORE_INDEX,
    MatrixCEScoreMode,
    causal_evidence_losses,
    historical_sample_norm_loss,
    matrix_ce_cell_scores,
)
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    resolve_matrix_ce_score_config,
)
from .readout import (
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    assert_frozen_deterministic_readout_model,
)


_MAX_PHYSICAL_QWEN_BATCH = 32


@dataclass(frozen=True, slots=True)
class _StreamingCandidateGradients:
    """Detached weighted-readout gradient at one Adapter-output boundary."""

    weighted_readout: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.weighted_readout:
            raise ValueError("streaming candidate gradient paths cannot be empty")
        for gradient in self.weighted_readout:
            if gradient.requires_grad:
                raise ValueError("streaming candidate gradients must be detached")


@dataclass(frozen=True, slots=True)
class _StreamingGradientContract:
    """Identity of the weights already applied by the single Qwen VJP."""

    objective_identity: str
    objective_schema_version: str
    matrix_ce_weight: float
    matrix_ce_mode: MatrixCEScoreMode
    matrix_ce_temperature: float
    l_gen_weight: float
    norm_weight: float | None
    matrix_valid_rows: int
    l_gen_samples: int
    data_parallel_world_size: int
    qwen_forward_batch_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StreamingGroupScores:
    """Detached values and row-local gradients from the single Qwen pass."""

    sample_ids: tuple[str, ...]
    score_matrix: torch.Tensor
    diagonal_l_gen: torch.Tensor
    evidence_token_counts: torch.Tensor
    historical_norm: torch.Tensor
    candidate_output_gradients: tuple[_StreamingCandidateGradients, ...] | None = None
    gradient_contract: _StreamingGradientContract | None = None

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
        if (self.candidate_output_gradients is None) != (
            self.gradient_contract is None
        ):
            raise ValueError(
                "streaming candidate gradients and their contract must coexist"
            )
        if self.candidate_output_gradients is not None:
            if len(self.candidate_output_gradients) != size:
                raise ValueError(
                    "streaming candidate gradients must cover every candidate"
                )
            if any(
                not isinstance(gradients, _StreamingCandidateGradients)
                for gradients in self.candidate_output_gradients
            ):
                raise TypeError("streaming candidate gradients use an invalid payload")


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
            or size > _MAX_PHYSICAL_QWEN_BATCH
            for size in self.qwen_forward_batch_sizes
        ):
            raise ValueError("multi-group Qwen batch sizes must be in [1, 32]")
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
    qwen_forward_batch_sizes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _StreamingCell:
    group_index: int
    row_index: int
    column_index: int
    source: RepresentationVisualTensorBundle
    row: RepresentationReadoutRow
    candidate: RepresentationVisualTensorBundle
    blocked_attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _StreamingRow:
    group_index: int
    row_index: int
    cells: tuple[_StreamingCell, ...]


def score_streaming_same_image_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
    *,
    objective: RepresentationObjectiveConfigLike | None = None,
    normalization: StreamingGlobalNormalization | None = None,
) -> StreamingGroupScores:
    """Score one block, optionally materializing its one-pass training VJP."""

    _validate_execution_inputs(family_adapter, model, group)
    group_scores, _ = _score_streaming_groups(
        family_adapter,
        model,
        (group,),
        objective=objective,
        normalization=normalization,
    )
    return group_scores[0]


def score_streaming_same_image_groups(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    groups: Sequence[SameImageReadoutGroup],
    *,
    objective: RepresentationObjectiveConfigLike | None = None,
    normalization: StreamingGlobalNormalization | None = None,
) -> StreamingMultiGroupScores:
    """Score separate CE blocks in packed, row-local physical Qwen batches.

    Two logical row waves are packed up to physical batch 32.  Every logical
    row still forms CE only against candidates from its own same-image group.
    Passing ``objective`` and ``normalization`` enables the training path: one
    weighted VJP is consumed at the candidate-D boundary per physical forward.
    """

    materialized = _validate_multi_group_execution_inputs(
        family_adapter,
        model,
        groups,
    )
    group_scores, forward_batch_sizes = _score_streaming_groups(
        family_adapter,
        model,
        materialized,
        objective=objective,
        normalization=normalization,
    )
    return StreamingMultiGroupScores(
        group_sample_ids=tuple(scores.sample_ids for scores in group_scores),
        group_scores=group_scores,
        qwen_forward_batch_sizes=tuple(forward_batch_sizes),
    )


def _score_streaming_groups(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    groups: tuple[SameImageReadoutGroup, ...],
    *,
    objective: RepresentationObjectiveConfigLike | None,
    normalization: StreamingGlobalNormalization | None,
) -> tuple[tuple[StreamingGroupScores, ...], tuple[int, ...]]:
    training = _validate_score_training_inputs(groups, objective, normalization)
    score_mode, score_temperature = (
        (MatrixCEScoreMode.LEGACY_SUMMED_NLL, 1.0)
        if objective is None
        else resolve_matrix_ce_score_config(objective)
    )
    score_cells: list[list[list[torch.Tensor | None]]] = [
        [[None for _ in group.candidates] for _ in group.rows] for group in groups
    ]
    diagonal_l_gen: list[list[torch.Tensor | None]] = [
        [None for _ in group.rows] for group in groups
    ]
    evidence_counts: list[list[torch.Tensor | None]] = [
        [None for _ in group.rows] for group in groups
    ]
    with torch.no_grad():
        norm_losses = [
            [
                historical_sample_norm_loss(
                    candidate.visual.main,
                    group.source_visual.main,
                    candidate.visual.deepstack,
                    group.source_visual.deepstack,
                )
                for candidate in group.candidates
            ]
            for group in groups
        ]

    accumulated_gradients = (
        [
            [
                [
                    torch.zeros_like(tensor)
                    for tensor in _candidate_output_tensors(candidate.visual)
                ]
                for candidate in group.candidates
            ]
            for group in groups
        ]
        if training
        else None
    )
    forward_batch_sizes: list[int] = []
    for paired_rows in _paired_multi_group_row_waves(groups):
        for compatible_rows in _partition_compatible_rows(paired_rows):
            cells = tuple(cell for row in compatible_rows for cell in row.cells)
            if training:
                losses = _forward_cell_batch_losses(family_adapter, model, cells)
            else:
                with torch.no_grad():
                    losses = _forward_cell_batch_losses(family_adapter, model, cells)
            cell_scores = matrix_ce_cell_scores(
                losses,
                mode=score_mode,
                temperature=score_temperature,
            )
            forward_batch_sizes.append(len(cells))

            row_matrix_terms: list[torch.Tensor] = []
            row_l_gen_terms: list[torch.Tensor] = []
            cursor = 0
            for logical_row in compatible_rows:
                row_size = len(logical_row.cells)
                row_slice = slice(cursor, cursor + row_size)
                row_scores = cell_scores[row_slice]
                for cell, value in zip(
                    logical_row.cells,
                    row_scores,
                    strict=True,
                ):
                    score_cells[cell.group_index][cell.row_index][cell.column_index] = (
                        value.detach()
                    )
                diagonal_index = cursor + logical_row.row_index
                diagonal_l_gen[logical_row.group_index][logical_row.row_index] = (
                    losses.per_sample_token_mean_nll[diagonal_index].detach()
                )
                evidence_counts[logical_row.group_index][logical_row.row_index] = (
                    losses.valid_token_counts[diagonal_index].detach()
                )
                if training:
                    row_matrix_terms.append(
                        F.cross_entropy(
                            row_scores.unsqueeze(0),
                            torch.tensor(
                                [logical_row.row_index],
                                dtype=torch.long,
                                device=row_scores.device,
                            ),
                            reduction="sum",
                        )
                    )
                    row_l_gen_terms.append(
                        losses.per_sample_token_mean_nll[diagonal_index]
                    )
                cursor += row_size
            if cursor != len(cells):
                raise RuntimeError("streaming row partition drifted")

            if training:
                assert objective is not None
                assert normalization is not None
                assert accumulated_gradients is not None
                surrogate = torch.stack(row_matrix_terms).sum() * (
                    objective.matrix_ce_weight
                    * normalization.data_parallel_world_size
                    / normalization.matrix_valid_rows
                ) + torch.stack(row_l_gen_terms).sum() * (
                    objective.l_gen_weight
                    * normalization.data_parallel_world_size
                    / normalization.l_gen_samples
                )
                candidate_slots = _unique_candidate_slots(compatible_rows)
                flat_outputs = tuple(
                    tensor
                    for group_index, column_index in candidate_slots
                    for tensor in _candidate_output_tensors(
                        groups[group_index].candidates[column_index].visual
                    )
                )
                gradients = torch.autograd.grad(
                    surrogate,
                    flat_outputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                gradient_cursor = 0
                for group_index, column_index in candidate_slots:
                    accumulators = accumulated_gradients[group_index][column_index]
                    for accumulator in accumulators:
                        accumulator.add_(gradients[gradient_cursor].detach())
                        gradient_cursor += 1
                if gradient_cursor != len(gradients):
                    raise RuntimeError("streaming candidate gradient partition drifted")

    contract = (
        None
        if not training
        else _gradient_contract(
            objective,
            normalization,
            tuple(forward_batch_sizes),
        )
    )
    group_scores = tuple(
        StreamingGroupScores(
            sample_ids=tuple(row.sample_id for row in group.rows),
            score_matrix=_stack_complete_matrix(
                score_cells[group_index],
                name="streaming score matrix",
            ).detach(),
            diagonal_l_gen=_stack_complete_vector(
                diagonal_l_gen[group_index],
                name="streaming diagonal L_gen",
            ).detach(),
            evidence_token_counts=_stack_complete_vector(
                evidence_counts[group_index],
                name="streaming evidence counts",
            ).detach(),
            historical_norm=torch.stack(norm_losses[group_index]).detach(),
            candidate_output_gradients=(
                None
                if accumulated_gradients is None
                else tuple(
                    _StreamingCandidateGradients(
                        weighted_readout=tuple(gradient.detach() for gradient in values)
                    )
                    for values in accumulated_gradients[group_index]
                )
            ),
            gradient_contract=contract,
        )
        for group_index, group in enumerate(groups)
    )
    return group_scores, tuple(forward_batch_sizes)


def backward_streaming_same_image_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: SameImageReadoutGroup,
    scores: StreamingGroupScores,
    *,
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> StreamingBackwardMetrics:
    """Traverse the Adapter once from a previously materialized Qwen VJP."""

    _validate_execution_inputs(family_adapter, model, group)
    if not isinstance(scores, StreamingGroupScores):
        raise TypeError("scores must be StreamingGroupScores")
    return _backward_streaming_groups(
        (group,),
        (scores,),
        objective=objective,
        normalization=normalization,
        expected_qwen_schedule=None,
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
    """Traverse the Adapter once for several independently scored CE blocks."""

    materialized = _validate_multi_group_execution_inputs(
        family_adapter,
        model,
        groups,
    )
    if not isinstance(scores, StreamingMultiGroupScores):
        raise TypeError("scores must be StreamingMultiGroupScores")
    sample_ids = tuple(
        tuple(row.sample_id for row in group.rows) for group in materialized
    )
    if scores.group_sample_ids != sample_ids:
        raise ValueError("multi-group streaming scores belong to another group/order")
    return _backward_streaming_groups(
        materialized,
        scores.group_scores,
        objective=objective,
        normalization=normalization,
        expected_qwen_schedule=scores.qwen_forward_batch_sizes,
    )


def _backward_streaming_groups(
    groups: tuple[SameImageReadoutGroup, ...],
    group_scores: tuple[StreamingGroupScores, ...],
    *,
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
    expected_qwen_schedule: tuple[int, ...] | None,
) -> StreamingBackwardMetrics:
    if not isinstance(
        objective,
        (
            RepresentationObjectiveConfig,
            RepresentationObjectiveConfigV2,
            RepresentationObjectiveConfigV3,
        ),
    ):
        raise TypeError("objective must be a representation objective config")
    if not isinstance(normalization, StreamingGlobalNormalization):
        raise TypeError("normalization must be StreamingGlobalNormalization")
    if len(groups) != len(group_scores):
        raise ValueError("streaming groups and score blocks must align")

    local_rows = sum(len(group.rows) for group in groups)
    if (
        normalization.matrix_valid_rows < local_rows
        or normalization.l_gen_samples < local_rows
    ):
        raise ValueError(
            "global normalization counts cannot be smaller than all local groups"
        )
    for group, scores in zip(groups, group_scores, strict=True):
        sample_ids = tuple(row.sample_id for row in group.rows)
        if scores.sample_ids != sample_ids:
            raise ValueError("streaming scores belong to a different group/order")
        if (
            scores.candidate_output_gradients is None
            or scores.gradient_contract is None
        ):
            raise ValueError(
                "training backward requires score materialization with objective "
                "and normalization"
            )
        _assert_gradient_contract(scores.gradient_contract, objective, normalization)
    contracts = tuple(scores.gradient_contract for scores in group_scores)
    first_contract = contracts[0]
    if first_contract is None:
        raise RuntimeError("streaming gradient contract unexpectedly disappeared")
    if any(contract != first_contract for contract in contracts[1:]):
        raise ValueError("streaming score blocks use different gradient contracts")
    if (
        expected_qwen_schedule is not None
        and first_contract.qwen_forward_batch_sizes != expected_qwen_schedule
    ):
        raise ValueError("streaming Qwen schedule differs from score telemetry")

    candidate_tensors = tuple(
        tuple(
            _candidate_output_tensors(candidate.visual)
            for candidate in group.candidates
        )
        for group in groups
    )
    accumulated_gradients: list[list[list[torch.Tensor]]] = []
    for group_index, (group_tensors, scores) in enumerate(
        zip(candidate_tensors, group_scores, strict=True)
    ):
        payloads = scores.candidate_output_gradients
        if payloads is None:
            raise RuntimeError("streaming candidate gradients unexpectedly disappeared")
        group_gradients: list[list[torch.Tensor]] = []
        for column_index, (tensors, payload) in enumerate(
            zip(group_tensors, payloads, strict=True)
        ):
            if any(not tensor.requires_grad for tensor in tensors):
                raise ValueError(
                    "every candidate main-D/DeepStack output must retain its "
                    "Adapter graph"
                )
            if len(tensors) != len(payload.weighted_readout):
                raise ValueError("streaming candidate gradient path count changed")
            values: list[torch.Tensor] = []
            for tensor, gradient in zip(
                tensors,
                payload.weighted_readout,
                strict=True,
            ):
                if (
                    tensor.shape != gradient.shape
                    or tensor.device != gradient.device
                    or tensor.dtype != gradient.dtype
                ):
                    raise ValueError(
                        "streaming candidate gradient tensor contract changed"
                    )
                values.append(gradient.clone())
            group_gradients.append(values)
        accumulated_gradients.append(group_gradients)

    if isinstance(objective, RepresentationObjectiveConfigV2):
        norm_scale = (
            objective.norm_weight
            * normalization.data_parallel_world_size
            / normalization.l_gen_samples
        )
        for group_index, (group, group_tensors, scores) in enumerate(
            zip(groups, candidate_tensors, group_scores, strict=True)
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
                    live_norm.detach(),
                    scores.historical_norm[column_index],
                ):
                    raise RuntimeError(
                        "deterministic streaming execution changed a norm value"
                    )
                norm_gradients = torch.autograd.grad(
                    live_norm * norm_scale,
                    tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for accumulator, gradient in zip(
                    accumulated_gradients[group_index][column_index],
                    norm_gradients,
                    strict=True,
                ):
                    accumulator.add_(gradient.detach())

    adapter_outputs: list[torch.Tensor] = []
    adapter_output_gradients: list[torch.Tensor] = []
    for group_tensors, group_gradients in zip(
        candidate_tensors,
        accumulated_gradients,
        strict=True,
    ):
        for tensors, gradients in zip(group_tensors, group_gradients, strict=True):
            adapter_outputs.extend(tensors)
            adapter_output_gradients.extend(gradients)
    for group in groups:
        for padding in group.collective_padding:
            tensors = _candidate_output_tensors(padding)
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

    matrix_numerator = torch.stack(
        tuple(
            F.cross_entropy(
                scores.score_matrix,
                torch.arange(
                    len(scores.sample_ids),
                    device=scores.score_matrix.device,
                ),
                reduction="sum",
            )
            for scores in group_scores
        )
    ).sum()
    l_gen_numerator = torch.stack(
        tuple(scores.diagonal_l_gen.sum() for scores in group_scores)
    ).sum()
    norm_numerator = (
        torch.stack(
            tuple(scores.historical_norm.sum() for scores in group_scores)
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
        qwen_forward_batch_sizes=first_contract.qwen_forward_batch_sizes,
    )


def _validate_score_training_inputs(
    groups: tuple[SameImageReadoutGroup, ...],
    objective: RepresentationObjectiveConfigLike | None,
    normalization: StreamingGlobalNormalization | None,
) -> bool:
    if objective is None:
        if normalization is not None:
            raise ValueError("normalization requires an explicit objective")
        return False
    if not isinstance(
        objective,
        (
            RepresentationObjectiveConfig,
            RepresentationObjectiveConfigV2,
            RepresentationObjectiveConfigV3,
        ),
    ):
        raise TypeError("objective must be a representation objective config")
    if normalization is None:
        return False
    if not isinstance(normalization, StreamingGlobalNormalization):
        raise TypeError("normalization must be StreamingGlobalNormalization")
    local_rows = sum(len(group.rows) for group in groups)
    if (
        normalization.matrix_valid_rows < local_rows
        or normalization.l_gen_samples < local_rows
    ):
        raise ValueError(
            "global normalization counts cannot be smaller than all local groups"
        )
    outputs = tuple(
        tensor
        for group in groups
        for candidate in group.candidates
        for tensor in _candidate_output_tensors(candidate.visual)
    )
    if any(not tensor.requires_grad for tensor in outputs):
        raise ValueError(
            "training score materialization requires every candidate Adapter graph"
        )
    if len({id(tensor) for tensor in outputs}) != len(outputs):
        raise ValueError("candidate Adapter outputs cannot be shared across columns")
    return True


def _gradient_contract(
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
    qwen_forward_batch_sizes: tuple[int, ...],
) -> _StreamingGradientContract:
    matrix_ce_mode, matrix_ce_temperature = resolve_matrix_ce_score_config(objective)
    return _StreamingGradientContract(
        objective_identity=objective.identity,
        objective_schema_version=objective.schema_version,
        matrix_ce_weight=objective.matrix_ce_weight,
        matrix_ce_mode=matrix_ce_mode,
        matrix_ce_temperature=matrix_ce_temperature,
        l_gen_weight=objective.l_gen_weight,
        norm_weight=(
            objective.norm_weight
            if isinstance(objective, RepresentationObjectiveConfigV2)
            else None
        ),
        matrix_valid_rows=normalization.matrix_valid_rows,
        l_gen_samples=normalization.l_gen_samples,
        data_parallel_world_size=normalization.data_parallel_world_size,
        qwen_forward_batch_sizes=qwen_forward_batch_sizes,
    )


def _assert_gradient_contract(
    contract: _StreamingGradientContract,
    objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> None:
    expected = _gradient_contract(
        objective,
        normalization,
        contract.qwen_forward_batch_sizes,
    )
    if contract != expected:
        raise ValueError(
            "backward objective/normalization differs from the materialized Qwen VJP"
        )


def _candidate_output_tensors(
    visual: RepresentationVisualTensorBundle,
) -> tuple[torch.Tensor, ...]:
    return (visual.main, *visual.deepstack)


def _unique_candidate_slots(
    rows: tuple[_StreamingRow, ...],
) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for row in rows:
        for cell in row.cells:
            slot = (cell.group_index, cell.column_index)
            if slot not in seen:
                seen.add(slot)
                ordered.append(slot)
    return tuple(ordered)


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


def _multi_group_row_waves(
    groups: tuple[SameImageReadoutGroup, ...],
) -> tuple[tuple[_StreamingRow, ...], ...]:
    maximum_size = max(len(group.rows) for group in groups)
    waves: list[tuple[_StreamingRow, ...]] = []
    for row_index in range(maximum_size):
        rows: list[_StreamingRow] = []
        for group_index, group in enumerate(groups):
            if row_index >= len(group.rows):
                continue
            row = group.rows[row_index]
            blocked_mask = _blocked_evidence_attention_mask(
                row,
                group.source_visual,
            )
            rows.append(
                _StreamingRow(
                    group_index=group_index,
                    row_index=row_index,
                    cells=tuple(
                        _StreamingCell(
                            group_index=group_index,
                            row_index=row_index,
                            column_index=column_index,
                            source=group.source_visual,
                            row=row,
                            candidate=candidate.visual,
                            blocked_attention_mask=blocked_mask,
                        )
                        for column_index, candidate in enumerate(group.candidates)
                    ),
                )
            )
        if rows:
            waves.append(tuple(rows))
    return tuple(waves)


def _paired_multi_group_row_waves(
    groups: tuple[SameImageReadoutGroup, ...],
) -> tuple[tuple[_StreamingRow, ...], ...]:
    waves = _multi_group_row_waves(groups)
    return tuple(
        tuple(row for wave in waves[index : index + 2] for row in wave)
        for index in range(0, len(waves), 2)
    )


def _partition_compatible_rows(
    rows: tuple[_StreamingRow, ...],
) -> tuple[tuple[_StreamingRow, ...], ...]:
    buckets: dict[tuple[object, ...], list[_StreamingRow]] = {}
    for row in rows:
        if len(row.cells) > _MAX_PHYSICAL_QWEN_BATCH:
            raise ValueError(
                "one logical Matrix-CE row exceeds the physical Qwen batch cap"
            )
        requests = tuple(
            _cell_request(
                cell.source,
                cell.row,
                cell.candidate,
                cell.blocked_attention_mask,
            )
            for cell in row.cells
        )
        key = _request_batch_key(requests[0])
        if any(_request_batch_key(request) != key for request in requests[1:]):
            raise ValueError("one logical Matrix-CE row has incompatible candidates")
        buckets.setdefault(key, []).append(row)

    partitions: list[tuple[_StreamingRow, ...]] = []
    for bucket in buckets.values():
        current: list[_StreamingRow] = []
        current_size = 0
        for row in bucket:
            row_size = len(row.cells)
            if current and current_size + row_size > _MAX_PHYSICAL_QWEN_BATCH:
                partitions.append(tuple(current))
                current = []
                current_size = 0
            current.append(row)
            current_size += row_size
        if current:
            partitions.append(tuple(current))
    return tuple(partitions)


def _request_batch_key(request: InjectedForwardRequest) -> tuple[object, ...]:
    """Return structural compatibility while permitting exact right padding."""

    position_prefix = (
        () if request.position_ids.ndim == 2 else (int(request.position_ids.shape[0]),)
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
        request.input_ids.ndim,
        request.input_ids.dtype,
        request.input_ids.device,
        request.attention_mask.ndim,
        request.attention_mask.dtype,
        request.attention_mask.device,
        request.position_ids.ndim,
        position_prefix,
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
    if len(cells) > _MAX_PHYSICAL_QWEN_BATCH:
        raise ValueError("a Qwen cell batch exceeds the physical batch cap")
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
        raise ValueError("Qwen cell batch contains structurally incompatible requests")
    maximum_sequence = max(int(request.input_ids.shape[1]) for request in requests)
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
        input_ids=torch.cat(
            tuple(
                _right_pad_input_ids(request.input_ids, maximum_sequence)
                for request in requests
            ),
            dim=0,
        ),
        attention_mask=torch.cat(
            tuple(
                _right_pad_attention_mask(request.attention_mask, maximum_sequence)
                for request in requests
            ),
            dim=0,
        ),
        position_ids=torch.cat(
            tuple(
                _right_pad_position_ids(request.position_ids, maximum_sequence)
                for request in requests
            ),
            dim=position_batch_dimension,
        ),
        visual_blocks=visual_blocks,
        use_cache=False,
    )
    result = family_adapter.forward_injected(model, batched_request)
    labels = torch.stack(
        tuple(
            _right_pad_labels(
                cell.row.supervision.labels,
                maximum_sequence,
                device=result.logits.device,
            )
            for cell in cells
        )
    )
    return causal_evidence_losses(result.logits, labels)


def _right_pad_input_ids(
    input_ids: torch.Tensor,
    sequence: int,
) -> torch.Tensor:
    padding = sequence - int(input_ids.shape[1])
    if padding < 0:
        raise ValueError("right-padding target is shorter than input IDs")
    if padding == 0:
        return input_ids
    safe_token = input_ids[:, -1:].expand(-1, padding)
    return torch.cat((input_ids, safe_token), dim=1)


def _right_pad_attention_mask(
    attention_mask: torch.Tensor,
    sequence: int,
) -> torch.Tensor:
    original = int(attention_mask.shape[-1])
    padding = sequence - original
    if padding < 0:
        raise ValueError("right-padding target is shorter than attention mask")
    if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (
        original,
        original,
    ):
        raise ValueError(
            "streaming right padding requires a square additive attention mask"
        )
    if padding == 0:
        return attention_mask
    if not attention_mask.dtype.is_floating_point:
        raise TypeError("streaming right padding requires floating attention bias")
    minimum = torch.finfo(attention_mask.dtype).min
    padded = torch.zeros(
        (*attention_mask.shape[:-2], sequence, sequence),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded[..., :original, :original] = attention_mask
    padded[..., :, original:] = minimum
    return padded


def _right_pad_position_ids(
    position_ids: torch.Tensor,
    sequence: int,
) -> torch.Tensor:
    original = int(position_ids.shape[-1])
    padding = sequence - original
    if padding < 0:
        raise ValueError("right-padding target is shorter than position IDs")
    if padding == 0:
        return position_ids
    increments = torch.arange(
        1,
        padding + 1,
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    increments = increments.reshape(
        *((1,) * (position_ids.ndim - 1)),
        padding,
    )
    return torch.cat((position_ids, position_ids[..., -1:] + increments), dim=-1)


def _right_pad_labels(
    labels: tuple[int, ...],
    sequence: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if len(labels) > sequence:
        raise ValueError("right-padding target is shorter than evidence labels")
    values = torch.full(
        (sequence,),
        EVIDENCE_IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    values[: len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return values


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
