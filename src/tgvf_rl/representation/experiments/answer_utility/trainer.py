"""Optimizer-boundary trainer for the isolated answer-utility experiment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import ContextManager, Protocol, Sequence

import torch
from torch import nn

from tgvf_rl.qwen.base import QwenVLMFamilyAdapter
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAccumulationIdentity,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigLike,
)
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.streaming import StreamingGlobalNormalization
from tgvf_rl.representation.training.trainer import (
    RepresentationAccumulationController,
    RepresentationPrecision,
    RepresentationTrainerConfig,
    _adapter_owned_trainable_parameters,
    _assert_distributed_identity,
    _assert_gradients,
    _assert_parameter_ownership,
    _assert_qwen_has_no_gradients,
    _clip_adapter_grad_norm_,
    _global_integer_counts,
    _parameter_device,
    synchronize_collective_candidate_counts,
)

from .config import AnswerSupervisionView, AnswerUtilityExperimentProfile
from .native_pipeline import AnswerUtilityReadoutGroup
from .objective import AnswerUtilityObjectiveConfig
from .streaming import AnswerUtilityStreamingMetrics, backward_answer_utility_group


class AnswerUtilityGroupBuilder(Protocol):
    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> AnswerUtilityReadoutGroup: ...


@dataclass(frozen=True, slots=True)
class AnswerUtilityStepMetrics:
    global_step: int
    global_matrix_ce_loss: float
    global_evidence_loss: float
    global_norm_loss: float
    global_answer_nll: float | None
    global_zero_answer_nll: float | None
    global_wrong_answer_nll: float | None
    global_correct_vs_zero_loss: float | None
    global_correct_vs_wrong_loss: float | None
    global_total_loss: float
    global_sample_count: int
    gradient_norm_before_clip: float
    learning_rate: float
    local_sample_ids: tuple[str, ...]
    local_legacy_qwen_forward_batch_sizes: tuple[int, ...]
    local_answer_qwen_forward_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.global_step <= 0 or self.global_sample_count <= 0:
            raise ValueError("answer utility step/count must be positive")
        required = (
            self.global_matrix_ce_loss,
            self.global_evidence_loss,
            self.global_norm_loss,
            self.global_total_loss,
            self.gradient_norm_before_clip,
            self.learning_rate,
        )
        if any(not math.isfinite(value) for value in required):
            raise ValueError("answer utility metrics must be finite")
        if self.global_answer_nll is not None and not math.isfinite(
            self.global_answer_nll
        ):
            raise ValueError("answer NLL must be finite when present")


class AnswerUtilityTrainer:
    """Train only the TGVF Adapter with the isolated E1--E4 objective."""

    def __init__(
        self,
        *,
        adapter: TGVFAdapter,
        qwen_model: nn.Module,
        family_adapter: QwenVLMFamilyAdapter,
        samples: Sequence[RepresentationTrainingSample],
        sampler: SameImageBatchSampler,
        group_builder: AnswerUtilityGroupBuilder,
        profile: AnswerUtilityExperimentProfile,
        objective: AnswerUtilityObjectiveConfig,
        legacy_objective: RepresentationObjectiveConfigLike,
        supervision_view: AnswerSupervisionView,
        accumulation: RepresentationAccumulationIdentity,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        config: RepresentationTrainerConfig,
        accumulation_controller: RepresentationAccumulationController | None = None,
        initial_global_step: int = 0,
    ) -> None:
        if not isinstance(adapter, TGVFAdapter):
            raise TypeError("adapter must be TGVFAdapter")
        if not isinstance(qwen_model, nn.Module):
            raise TypeError("qwen_model must be nn.Module")
        if not isinstance(family_adapter, QwenVLMFamilyAdapter):
            raise TypeError("family_adapter must be QwenVLMFamilyAdapter")
        if not samples or any(
            not isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise TypeError("answer trainer samples must be non-empty/typed")
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("answer trainer requires SameImageBatchSampler")
        if not callable(group_builder):
            raise TypeError("answer group_builder must be callable")
        _validate_trainable_profile(profile, objective, supervision_view)
        if not isinstance(accumulation, RepresentationAccumulationIdentity):
            raise TypeError("answer trainer accumulation identity is invalid")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("answer trainer requires a torch optimizer")
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.LRScheduler
        ):
            raise TypeError("answer trainer scheduler must be a torch LRScheduler")
        if not isinstance(config, RepresentationTrainerConfig):
            raise TypeError("answer trainer execution config is invalid")
        if (
            isinstance(initial_global_step, bool)
            or not isinstance(initial_global_step, int)
            or initial_global_step < 0
        ):
            raise ValueError("initial_global_step must be non-negative")
        if accumulation_controller is not None and (
            not callable(getattr(accumulation_controller, "begin_microstep", None))
            or not callable(getattr(accumulation_controller, "finish_window", None))
        ):
            raise TypeError("accumulation controller interface is incomplete")
        _assert_parameter_ownership(adapter, qwen_model, optimizer)
        _assert_distributed_identity(accumulation.data_parallel_world_size)
        self.adapter = adapter
        self.qwen_model = qwen_model
        self.family_adapter = family_adapter
        self.samples = tuple(samples)
        self.sampler = sampler
        self.group_builder = group_builder
        self.profile = profile
        self.objective = objective
        self.legacy_objective = legacy_objective
        self.supervision_view = supervision_view
        self.accumulation = accumulation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.accumulation_controller = accumulation_controller
        self.global_step = initial_global_step
        self._failed = False

    def train_step(self) -> AnswerUtilityStepMetrics:
        if self._failed:
            raise RuntimeError("a failed AnswerUtilityTrainer cannot be reused")
        try:
            return self._train_step_impl()
        except BaseException:
            self._failed = True
            raise

    def _train_step_impl(self) -> AnswerUtilityStepMetrics:
        self.adapter.train(True)
        self.qwen_model.eval()
        _assert_parameter_ownership(self.adapter, self.qwen_model, self.optimizer)
        group_count, groups_per_microstep = _execution_group_counts(self.accumulation)
        batch_indices = tuple(self.sampler.next_batch() for _ in range(group_count))
        local_rows = sum(len(indices) for indices in batch_indices)
        global_rows, global_samples = _global_integer_counts(
            local_rows,
            local_rows,
            device=_parameter_device(self.adapter),
        )
        normalization = StreamingGlobalNormalization(
            matrix_valid_rows=global_rows,
            l_gen_samples=global_samples,
            data_parallel_world_size=self.accumulation.data_parallel_world_size,
        )
        collective_counts = synchronize_collective_candidate_counts(
            tuple(len(indices) for indices in batch_indices),
            device=_parameter_device(self.adapter),
        )
        self.optimizer.zero_grad(set_to_none=True)
        local_totals = torch.zeros(
            8,
            dtype=torch.float64,
            device=_parameter_device(self.adapter),
        )
        local_ids: list[str] = []
        legacy_schedule: list[int] = []
        answer_schedule: list[int] = []
        microstep_count = group_count // groups_per_microstep
        active_microstep = -1
        failure: BaseException | None = None
        try:
            for group_index, (indices, collective_count) in enumerate(
                zip(batch_indices, collective_counts, strict=True)
            ):
                microstep_index = group_index // groups_per_microstep
                if microstep_index != active_microstep:
                    active_microstep = microstep_index
                    if self.accumulation_controller is not None:
                        self.accumulation_controller.begin_microstep(
                            index=microstep_index,
                            count=microstep_count,
                        )
                logical_samples = tuple(self.samples[index] for index in indices)
                with self._autocast_context():
                    group = self.group_builder(
                        logical_samples,
                        self.adapter,
                        collective_candidate_count=collective_count,
                    )
                    _assert_group_identity(
                        group,
                        expected_ids=tuple(
                            sample.sample_id for sample in logical_samples
                        ),
                        collective_candidate_count=collective_count,
                        supervision_view=self.supervision_view,
                    )
                    metrics = backward_answer_utility_group(
                        self.family_adapter,
                        self.qwen_model,
                        group,
                        objective=self.objective,
                        legacy_objective=self.legacy_objective,
                        normalization=normalization,
                    )
                _accumulate_metrics(
                    local_totals,
                    metrics,
                    objective=self.objective,
                    expected_sample_count=len(logical_samples),
                )
                local_ids.extend(sample.sample_id for sample in logical_samples)
                legacy_schedule.extend(metrics.legacy.qwen_forward_batch_sizes)
                answer_schedule.extend(metrics.answer_qwen_forward_batch_sizes)
        except BaseException as error:
            failure = error
            raise
        finally:
            if self.accumulation_controller is not None:
                try:
                    self.accumulation_controller.finish_window()
                except BaseException as finish_error:
                    if failure is None:
                        raise
                    failure.add_note(
                        "answer accumulation-controller restoration also failed: "
                        f"{finish_error!r}"
                    )
        trainable = _adapter_owned_trainable_parameters(self.adapter)
        _assert_gradients(
            trainable, require_all=self.config.require_all_adapter_gradients
        )
        _assert_qwen_has_no_gradients(self.qwen_model)
        gradient_norm = _clip_adapter_grad_norm_(
            trainable,
            max_norm=self.config.max_grad_norm,
        )
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        _assert_qwen_has_no_gradients(self.qwen_model)
        global_totals = _global_sum(local_totals).tolist()
        matrix, evidence, norm, correct, zero, wrong, compare_zero, compare_wrong = (
            float(value) for value in global_totals
        )
        matrix_loss = matrix / global_rows
        evidence_loss = evidence / global_samples
        norm_loss = norm / global_samples
        answer_loss = (
            None if self.objective.answer_weight == 0.0 else correct / global_samples
        )
        zero_loss = (
            zero / global_samples
            if self.objective.correct_vs_zero_weight > 0.0
            else None
        )
        wrong_loss = (
            wrong / global_samples
            if self.objective.correct_vs_wrong_weight > 0.0
            else None
        )
        compare_zero_loss = (
            compare_zero / global_samples
            if self.objective.correct_vs_zero_weight > 0.0
            else None
        )
        compare_wrong_loss = (
            compare_wrong / global_samples
            if self.objective.correct_vs_wrong_weight > 0.0
            else None
        )
        total = (
            matrix_loss * self.objective.existing_matrix_weight
            + evidence_loss * self.objective.existing_evidence_weight
            + norm_loss * self.objective.norm_weight
        )
        if answer_loss is not None:
            total += answer_loss * self.objective.answer_weight
        if compare_zero_loss is not None:
            total += compare_zero_loss * self.objective.correct_vs_zero_weight
        if compare_wrong_loss is not None:
            total += compare_wrong_loss * self.objective.correct_vs_wrong_weight
        return AnswerUtilityStepMetrics(
            global_step=self.global_step,
            global_matrix_ce_loss=matrix_loss,
            global_evidence_loss=evidence_loss,
            global_norm_loss=norm_loss,
            global_answer_nll=answer_loss,
            global_zero_answer_nll=zero_loss,
            global_wrong_answer_nll=wrong_loss,
            global_correct_vs_zero_loss=compare_zero_loss,
            global_correct_vs_wrong_loss=compare_wrong_loss,
            global_total_loss=total,
            global_sample_count=global_samples,
            gradient_norm_before_clip=gradient_norm,
            learning_rate=learning_rate,
            local_sample_ids=tuple(local_ids),
            local_legacy_qwen_forward_batch_sizes=tuple(legacy_schedule),
            local_answer_qwen_forward_batch_sizes=tuple(answer_schedule),
        )

    def _autocast_context(self) -> ContextManager[object]:
        if self.config.precision is RepresentationPrecision.FP32:
            return nullcontext()
        device_type = _parameter_device(self.adapter).type
        if device_type not in {"cpu", "cuda"}:
            raise ValueError(f"BF16 is unsupported on device type {device_type!r}")
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def _execution_group_counts(
    accumulation: RepresentationAccumulationIdentity,
) -> tuple[int, int]:
    direct_groups = getattr(accumulation, "groups_per_rank_per_optimizer_step", 1)
    accumulation_steps = accumulation.gradient_accumulation_steps
    if direct_groups <= 1:
        return accumulation_steps, 1
    if direct_groups % accumulation_steps:
        raise ValueError("direct groups must divide over accumulation microsteps")
    return direct_groups, direct_groups // accumulation_steps


def _validate_trainable_profile(
    profile: AnswerUtilityExperimentProfile,
    objective: AnswerUtilityObjectiveConfig,
    supervision_view: AnswerSupervisionView,
) -> None:
    if not isinstance(profile, AnswerUtilityExperimentProfile):
        raise TypeError("answer trainer requires an experiment profile")
    if not profile.train_adapter:
        raise ValueError("evaluation-only E0 cannot enter the training path")
    if not isinstance(objective, AnswerUtilityObjectiveConfig):
        raise TypeError("answer trainer objective must be explicit")
    if objective.loss_weights != profile.expected_loss_weights:
        raise ValueError("answer trainer objective differs from its frozen profile")
    if supervision_view is not profile.answer_supervision_view:
        raise ValueError("answer trainer supervision differs from its frozen profile")
    no_answer = supervision_view is AnswerSupervisionView.NONE
    if no_answer is not (objective.answer_weight == 0.0):
        raise ValueError("answer weight and supervision view disagree")


def _assert_group_identity(
    group: AnswerUtilityReadoutGroup,
    *,
    expected_ids: tuple[str, ...],
    collective_candidate_count: int,
    supervision_view: AnswerSupervisionView,
) -> None:
    actual_ids = tuple(row.sample_id for row in group.legacy.rows)
    if actual_ids != expected_ids:
        raise ValueError("answer group builder changed sampler-owned identity/order")
    if group.legacy.collective_candidate_count != collective_candidate_count:
        raise ValueError("answer group builder changed collective candidate count")
    if group.supervision_view is not supervision_view:
        raise ValueError("answer group builder changed supervision view")


def _accumulate_metrics(
    totals: torch.Tensor,
    metrics: AnswerUtilityStreamingMetrics,
    *,
    objective: AnswerUtilityObjectiveConfig,
    expected_sample_count: int,
) -> None:
    if totals.shape != (8,):
        raise ValueError("answer metric accumulator must have eight components")
    optional = (
        metrics.correct_answer_nll_numerator,
        metrics.zero_answer_nll_numerator,
        metrics.wrong_answer_nll_numerator,
        metrics.correct_vs_zero_numerator,
        metrics.correct_vs_wrong_numerator,
    )
    answer_active = objective.answer_weight > 0.0
    expected_answer_samples = expected_sample_count if answer_active else 0
    if metrics.local_answer_sample_count != expected_answer_samples:
        raise RuntimeError("answer metric sample count differs from objective topology")
    expected_present = (
        answer_active,
        objective.correct_vs_zero_weight > 0.0,
        objective.correct_vs_wrong_weight > 0.0,
        objective.correct_vs_zero_weight > 0.0,
        objective.correct_vs_wrong_weight > 0.0,
    )
    if tuple(value is not None for value in optional) != expected_present:
        raise RuntimeError("answer metric numerators differ from objective topology")
    if answer_active and len(metrics.answer_qwen_forward_batch_sizes) != (
        expected_sample_count
    ):
        raise RuntimeError("answer forward schedule differs from sample count")
    values = (
        metrics.legacy.matrix_ce_numerator,
        metrics.legacy.l_gen_numerator,
        metrics.legacy.norm_numerator,
        *optional,
    )
    if values[2] is None:
        raise RuntimeError("answer utility requires historical norm metrics")
    materialized = torch.stack(
        tuple(
            (
                totals.new_zeros(())
                if value is None
                else value.detach().to(device=totals.device, dtype=totals.dtype)
            )
            for value in values
        )
    )
    totals.add_(materialized)


def _global_sum(values: torch.Tensor) -> torch.Tensor:
    result = values.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


__all__ = [
    "AnswerUtilityGroupBuilder",
    "AnswerUtilityStepMetrics",
    "AnswerUtilityTrainer",
]
