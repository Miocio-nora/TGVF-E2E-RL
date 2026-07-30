"""Memory-bounded answer VJP composed with the accepted evidence VJP.

The frozen Qwen graph is reduced to gradients at each live Adapter output.
Those detached answer gradients are merged into the accepted streaming VJP
payload before its independent norm VJP and single Adapter backward.
Consequently Matrix/evidence/norm and answer/counterfactual signals cross the
Adapter/FSDP boundary exactly once; production streaming remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch.nn import functional as F

from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
)
from tgvf_rl.representation.training.losses import causal_evidence_losses
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
)
from tgvf_rl.representation.training.readout import (
    RepresentationVisualTensorBundle,
    assert_frozen_deterministic_readout_model,
)
from tgvf_rl.representation.training.streaming import (
    StreamingBackwardMetrics,
    StreamingGlobalNormalization,
    StreamingGroupScores,
    _StreamingCandidateGradients,
    backward_streaming_same_image_group,
    score_streaming_same_image_group,
)

from .config import AnswerSupervisionView
from .controls import AnswerUtilityArm, AnswerUtilityControlRow
from .native_pipeline import AnswerUtilityReadoutGroup
from .objective import AnswerUtilityObjectiveConfig


@dataclass(frozen=True, slots=True)
class AnswerUtilityStreamingMetrics:
    """Detached local numerators; distributed aggregation stays trainer-owned."""

    legacy: StreamingBackwardMetrics
    correct_answer_nll_numerator: torch.Tensor | None
    zero_answer_nll_numerator: torch.Tensor | None
    wrong_answer_nll_numerator: torch.Tensor | None
    correct_vs_zero_numerator: torch.Tensor | None
    correct_vs_wrong_numerator: torch.Tensor | None
    local_answer_sample_count: int
    answer_qwen_forward_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.legacy, StreamingBackwardMetrics):
            raise TypeError("answer metrics require legacy streaming metrics")
        if self.local_answer_sample_count < 0:
            raise ValueError("local answer sample count cannot be negative")
        values = (
            self.correct_answer_nll_numerator,
            self.zero_answer_nll_numerator,
            self.wrong_answer_nll_numerator,
            self.correct_vs_zero_numerator,
            self.correct_vs_wrong_numerator,
        )
        for value in values:
            if value is not None and (
                not isinstance(value, torch.Tensor)
                or value.ndim != 0
                or value.requires_grad
            ):
                raise ValueError("answer metric numerators must be detached scalars")
        if self.local_answer_sample_count == 0:
            if any(value is not None for value in values) or (
                self.answer_qwen_forward_batch_sizes
            ):
                raise ValueError("E0 metrics cannot contain answer forwards")
        elif self.correct_answer_nll_numerator is None:
            raise ValueError("answer metrics require correct-D NLL")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.answer_qwen_forward_batch_sizes
        ):
            raise ValueError("answer Qwen batch sizes must be positive integers")


@dataclass(frozen=True, slots=True)
class _AnswerScoreMaterialization:
    gradients: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    correct_nll: torch.Tensor
    zero_nll: torch.Tensor | None
    wrong_nll: torch.Tensor | None
    correct_vs_zero: torch.Tensor | None
    correct_vs_wrong: torch.Tensor | None
    qwen_forward_batch_sizes: tuple[int, ...]


def backward_answer_utility_group(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: AnswerUtilityReadoutGroup,
    *,
    objective: AnswerUtilityObjectiveConfig,
    legacy_objective: RepresentationObjectiveConfigLike,
    normalization: StreamingGlobalNormalization,
) -> AnswerUtilityStreamingMetrics:
    """Score both branches and traverse the live Adapter graph exactly once."""

    if not isinstance(group, AnswerUtilityReadoutGroup):
        raise TypeError("answer utility backward requires its typed group")
    if not isinstance(objective, AnswerUtilityObjectiveConfig):
        raise TypeError("answer utility objective must be explicit")
    if not isinstance(normalization, StreamingGlobalNormalization):
        raise TypeError("answer utility normalization must be explicit")
    _validate_legacy_objective(legacy_objective, objective)
    assert_frozen_deterministic_readout_model(model)

    legacy_scores = score_streaming_same_image_group(
        family_adapter,
        model,
        group.legacy,
        objective=legacy_objective,
        normalization=normalization,
    )
    if objective.answer_weight == 0.0:
        if group.supervision_view is not AnswerSupervisionView.NONE:
            raise ValueError("zero answer weight requires the no-answer E0 view")
        legacy = backward_streaming_same_image_group(
            family_adapter,
            model,
            group.legacy,
            legacy_scores,
            objective=legacy_objective,
            normalization=normalization,
        )
        return AnswerUtilityStreamingMetrics(
            legacy=legacy,
            correct_answer_nll_numerator=None,
            zero_answer_nll_numerator=None,
            wrong_answer_nll_numerator=None,
            correct_vs_zero_numerator=None,
            correct_vs_wrong_numerator=None,
            local_answer_sample_count=0,
            answer_qwen_forward_batch_sizes=(),
        )
    if group.supervision_view is AnswerSupervisionView.NONE:
        raise ValueError("active answer loss requires an answer supervision view")
    if (objective.correct_vs_zero_weight > 0.0) is not (group.requires_zero_control):
        raise ValueError("zero-D objective and group topology differ")
    if (objective.correct_vs_wrong_weight > 0.0) is not (group.requires_wrong_control):
        raise ValueError("wrong-D objective and group topology differ")
    if normalization.l_gen_samples < len(group.answer_supervisions):
        raise ValueError("global answer denominator is smaller than this local group")

    answer = _score_answer_rows(
        family_adapter,
        model,
        group,
        objective=objective,
        normalization=normalization,
    )
    combined_scores = _merge_answer_boundary_gradients(
        group.legacy,
        legacy_scores,
        answer.gradients,
    )
    legacy = backward_streaming_same_image_group(
        family_adapter,
        model,
        group.legacy,
        combined_scores,
        objective=legacy_objective,
        normalization=normalization,
    )
    return AnswerUtilityStreamingMetrics(
        legacy=legacy,
        correct_answer_nll_numerator=answer.correct_nll.sum().detach(),
        zero_answer_nll_numerator=(
            None if answer.zero_nll is None else answer.zero_nll.sum().detach()
        ),
        wrong_answer_nll_numerator=(
            None if answer.wrong_nll is None else answer.wrong_nll.sum().detach()
        ),
        correct_vs_zero_numerator=(
            None
            if answer.correct_vs_zero is None
            else answer.correct_vs_zero.sum().detach()
        ),
        correct_vs_wrong_numerator=(
            None
            if answer.correct_vs_wrong is None
            else answer.correct_vs_wrong.sum().detach()
        ),
        local_answer_sample_count=len(group.answer_supervisions),
        answer_qwen_forward_batch_sizes=answer.qwen_forward_batch_sizes,
    )


def _score_answer_rows(
    family_adapter: QwenVLMFamilyAdapter,
    model: object,
    group: AnswerUtilityReadoutGroup,
    *,
    objective: AnswerUtilityObjectiveConfig,
    normalization: StreamingGlobalNormalization,
) -> _AnswerScoreMaterialization:
    gradient_by_tensor: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    correct_values: list[torch.Tensor] = []
    zero_values: list[torch.Tensor] = []
    wrong_values: list[torch.Tensor] = []
    zero_comparisons: list[torch.Tensor] = []
    wrong_comparisons: list[torch.Tensor] = []
    forward_batch_sizes: list[int] = []
    for supervision, controls in zip(
        group.answer_supervisions,
        group.controls,
        strict=True,
    ):
        arms = [AnswerUtilityArm.CORRECT]
        if objective.correct_vs_zero_weight > 0.0:
            arms.append(AnswerUtilityArm.ZERO)
        if objective.correct_vs_wrong_weight > 0.0:
            arms.append(AnswerUtilityArm.WRONG_SAME_IMAGE_TARGET)
        requests = tuple(
            supervision.request(
                observation=controls.observation(arm),
                source=(
                    group.legacy.source_visual
                    if supervision.context_kind == "gold_evidence"
                    else None
                ),
            )
            for arm in arms
        )
        result = family_adapter.forward_injected(
            model,
            _batch_identical_answer_requests(requests),
        )
        labels = torch.tensor(
            tuple(supervision.labels for _ in arms),
            dtype=torch.long,
            device=result.logits.device,
        )
        losses = causal_evidence_losses(result.logits, labels)
        nll_by_arm = dict(zip(arms, losses.per_sample_token_mean_nll, strict=True))
        correct = nll_by_arm[AnswerUtilityArm.CORRECT]
        zero = nll_by_arm.get(AnswerUtilityArm.ZERO)
        wrong = nll_by_arm.get(AnswerUtilityArm.WRONG_SAME_IMAGE_TARGET)
        zero_comparison = (
            None if zero is None else _smooth_nll_margin(correct, zero, objective)
        )
        wrong_comparison = (
            None if wrong is None else _smooth_nll_margin(correct, wrong, objective)
        )
        row_loss = correct * objective.answer_weight
        if zero_comparison is not None:
            row_loss = row_loss + (zero_comparison * objective.correct_vs_zero_weight)
        if wrong_comparison is not None:
            row_loss = row_loss + (wrong_comparison * objective.correct_vs_wrong_weight)
        row_loss = row_loss * (
            normalization.data_parallel_world_size / normalization.l_gen_samples
        )
        live_tensors = _unique_live_control_tensors(
            controls,
            include_wrong=objective.correct_vs_wrong_weight > 0.0,
        )
        gradients = torch.autograd.grad(
            row_loss,
            live_tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        for tensor, gradient in zip(live_tensors, gradients, strict=True):
            key = id(tensor)
            detached = gradient.detach()
            if key in gradient_by_tensor:
                existing_tensor, existing_gradient = gradient_by_tensor[key]
                if existing_tensor is not tensor:
                    raise RuntimeError("answer gradient tensor identity collision")
                existing_gradient.add_(detached)
            else:
                gradient_by_tensor[key] = (tensor, detached.clone())
        correct_values.append(correct.detach())
        if zero is not None:
            zero_values.append(zero.detach())
            assert zero_comparison is not None
            zero_comparisons.append(zero_comparison.detach())
        if wrong is not None:
            wrong_values.append(wrong.detach())
            assert wrong_comparison is not None
            wrong_comparisons.append(wrong_comparison.detach())
        forward_batch_sizes.append(len(arms))
    return _AnswerScoreMaterialization(
        gradients=tuple(gradient_by_tensor.values()),
        correct_nll=torch.stack(correct_values),
        zero_nll=torch.stack(zero_values) if zero_values else None,
        wrong_nll=torch.stack(wrong_values) if wrong_values else None,
        correct_vs_zero=(torch.stack(zero_comparisons) if zero_comparisons else None),
        correct_vs_wrong=(
            torch.stack(wrong_comparisons) if wrong_comparisons else None
        ),
        qwen_forward_batch_sizes=tuple(forward_batch_sizes),
    )


def _batch_identical_answer_requests(
    requests: tuple[InjectedForwardRequest, ...],
) -> InjectedForwardRequest:
    if not requests:
        raise ValueError("answer request batch cannot be empty")
    first = requests[0]
    for request in requests[1:]:
        if (
            not torch.equal(request.input_ids, first.input_ids)
            or not torch.equal(request.attention_mask, first.attention_mask)
            or not torch.equal(request.position_ids, first.position_ids)
            or len(request.visual_blocks) != len(first.visual_blocks)
        ):
            raise ValueError("answer arms must share one exact native context")
        for actual, expected in zip(
            request.visual_blocks,
            first.visual_blocks,
            strict=True,
        ):
            if (
                actual.kind != expected.kind
                or actual.positions != expected.positions
                or actual.deepstack_positions != expected.deepstack_positions
            ):
                raise ValueError("answer arms may differ only in visual tensors")
    blocks = tuple(
        InjectedVisualBlock(
            kind=first.visual_blocks[index].kind,
            positions=first.visual_blocks[index].positions,
            embeddings=torch.cat(
                tuple(request.visual_blocks[index].embeddings for request in requests),
                dim=0,
            ),
            deepstack=tuple(
                torch.cat(
                    tuple(
                        request.visual_blocks[index].deepstack[branch]
                        for request in requests
                    ),
                    dim=0,
                )
                for branch in range(len(first.visual_blocks[index].deepstack))
            ),
            deepstack_positions=first.visual_blocks[index].deepstack_positions,
        )
        for index in range(len(first.visual_blocks))
    )
    position_batch_dimension = 0 if first.position_ids.ndim == 2 else 1
    return InjectedForwardRequest(
        input_ids=torch.cat(tuple(request.input_ids for request in requests), dim=0),
        attention_mask=torch.cat(
            tuple(request.attention_mask for request in requests), dim=0
        ),
        position_ids=torch.cat(
            tuple(request.position_ids for request in requests),
            dim=position_batch_dimension,
        ),
        visual_blocks=blocks,
        use_cache=False,
    )


def _unique_live_control_tensors(
    controls: AnswerUtilityControlRow,
    *,
    include_wrong: bool,
) -> tuple[torch.Tensor, ...]:
    if include_wrong and controls.wrong is None:
        raise ValueError("active wrong-D objective has no wrong observation")
    bundles = (
        (controls.correct, controls.wrong) if include_wrong else (controls.correct,)
    )
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for bundle in bundles:
        for tensor in _visual_tensors(bundle):
            if not tensor.requires_grad:
                raise ValueError("correct/wrong D must retain the live Adapter graph")
            if id(tensor) not in seen:
                seen.add(id(tensor))
                tensors.append(tensor)
    return tuple(tensors)


def _visual_tensors(
    visual: RepresentationVisualTensorBundle,
) -> tuple[torch.Tensor, ...]:
    return (
        (visual.main, *visual.deepstack)
        if visual.d_deepstack_active
        else (visual.main,)
    )


def _smooth_nll_margin(
    correct_nll: torch.Tensor,
    control_nll: torch.Tensor,
    config: AnswerUtilityObjectiveConfig,
) -> torch.Tensor:
    temperature = config.comparison_temperature
    return temperature * F.softplus(
        (correct_nll - control_nll + config.comparison_margin) / temperature
    )


def _merge_answer_boundary_gradients(
    group: object,
    legacy_scores: StreamingGroupScores,
    answer_gradients: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> StreamingGroupScores:
    """Merge answer VJP before the legacy norm VJP and one Adapter backward.

    A tensor hook is deliberately forbidden here: the legacy implementation
    first differentiates its norm term at the same candidate tensors and only
    then performs the Adapter-boundary backward.  A hook would fire in both
    operations and count the answer gradient twice.
    """

    candidates = getattr(group, "candidates", None)
    if not isinstance(candidates, tuple):
        raise TypeError("answer gradient merge requires typed legacy candidates")
    payloads = legacy_scores.candidate_output_gradients
    if payloads is None or len(payloads) != len(candidates):
        raise ValueError("legacy score payload does not cover every candidate")
    answer_by_tensor: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for tensor, gradient in answer_gradients:
        if not isinstance(tensor, torch.Tensor) or not isinstance(
            gradient, torch.Tensor
        ):
            raise TypeError("answer boundary gradients must be tensor pairs")
        if gradient.requires_grad:
            raise ValueError("answer boundary gradients must be detached")
        if id(tensor) in answer_by_tensor:
            raise ValueError("answer boundary gradient identities must be unique")
        answer_by_tensor[id(tensor)] = (tensor, gradient)

    merged_payloads: list[_StreamingCandidateGradients] = []
    consumed: set[int] = set()
    for candidate, payload in zip(candidates, payloads, strict=True):
        tensors = _visual_tensors(candidate.visual)
        if len(tensors) != len(payload.weighted_readout):
            raise ValueError("legacy/answer boundary path count changed")
        gradients: list[torch.Tensor] = []
        for tensor, legacy_gradient in zip(
            tensors,
            payload.weighted_readout,
            strict=True,
        ):
            pair = answer_by_tensor.get(id(tensor))
            if pair is None:
                extra = torch.zeros_like(legacy_gradient)
            else:
                original, extra = pair
                if original is not tensor:
                    raise RuntimeError("answer boundary tensor identity collision")
                consumed.add(id(tensor))
            if (
                legacy_gradient.shape != tensor.shape
                or legacy_gradient.dtype != tensor.dtype
                or legacy_gradient.device != tensor.device
                or extra.shape != tensor.shape
                or extra.dtype != tensor.dtype
                or extra.device != tensor.device
            ):
                raise ValueError("legacy/answer boundary gradient contract changed")
            gradients.append(legacy_gradient + extra)
        merged_payloads.append(
            _StreamingCandidateGradients(weighted_readout=tuple(gradients))
        )
    if consumed != set(answer_by_tensor):
        raise ValueError("answer gradient refers to a non-candidate Adapter output")
    return replace(
        legacy_scores,
        candidate_output_gradients=tuple(merged_payloads),
    )


def _validate_legacy_objective(
    legacy: RepresentationObjectiveConfigLike,
    answer: AnswerUtilityObjectiveConfig,
) -> None:
    if not isinstance(legacy, RepresentationObjectiveConfigV2):
        raise TypeError("answer utility requires the norm-aware legacy objective")
    if not math.isclose(
        legacy.matrix_ce_weight,
        answer.existing_matrix_weight,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("legacy Matrix-CE weight differs from answer sidecar")
    if not math.isclose(
        legacy.l_gen_weight,
        answer.existing_evidence_weight,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("legacy evidence weight differs from answer sidecar")
    if not math.isclose(
        legacy.norm_weight,
        answer.norm_weight,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("legacy norm weight differs from answer sidecar")


__all__ = [
    "AnswerUtilityStreamingMetrics",
    "backward_answer_utility_group",
]
