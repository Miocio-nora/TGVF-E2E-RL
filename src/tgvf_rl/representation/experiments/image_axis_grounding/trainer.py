"""Optimizer-boundary trainer for the isolated image-axis grounding arm."""

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
    RepresentationObjectiveConfigV2,
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

from .native_pipeline import ImageAxisGroundingGroup
from .streaming import (
    ImageAxisGlobalNormalization,
    ImageAxisStreamingMetrics,
    backward_image_axis_grounding_group,
)


IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION = (
    "image_axis_grounding_objective_v1"
)


@dataclass(frozen=True, slots=True)
class ImageAxisGroundingObjectiveConfig:
    """Frozen first-arm image-axis loss identity.

    The first experiment intentionally exposes no silent tuning surface.  A
    changed weight, temperature, or negative count is a new named experiment
    and must receive a new schema instead of mutating this baseline.
    """

    image_axis_matrix_weight: float = 1.0
    image_axis_temperature: float = 1.0
    negative_count: int = 1
    schema_version: str = IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("image-axis objective schema mismatch")
        if (
            isinstance(self.image_axis_matrix_weight, bool)
            or not isinstance(self.image_axis_matrix_weight, float)
            or not math.isfinite(self.image_axis_matrix_weight)
            or self.image_axis_matrix_weight != 1.0
        ):
            raise ValueError("v1 freezes image_axis_matrix_weight at 1.0")
        if (
            isinstance(self.image_axis_temperature, bool)
            or not isinstance(self.image_axis_temperature, float)
            or not math.isfinite(self.image_axis_temperature)
            or self.image_axis_temperature != 1.0
        ):
            raise ValueError("v1 freezes image_axis_temperature at 1.0")
        if (
            isinstance(self.negative_count, bool)
            or not isinstance(self.negative_count, int)
            or self.negative_count != 1
        ):
            raise ValueError("v1 requires exactly one wrong-image negative")

    @property
    def loss_weights(self) -> tuple[float]:
        return (self.image_axis_matrix_weight,)

    def validation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "image_axis_matrix_weight": self.image_axis_matrix_weight,
            "image_axis_temperature": self.image_axis_temperature,
            "negative_count": self.negative_count,
        }


class ImageAxisGroundingGroupBuilder(Protocol):
    """Build live paired groups and query eligibility without model forwards."""

    def image_axis_row_mask(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
    ) -> tuple[bool, ...]: ...

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> ImageAxisGroundingGroup: ...


@dataclass(frozen=True, slots=True)
class ImageAxisGroundingStepMetrics:
    global_step: int
    global_matrix_ce_loss: float
    global_l_gen_loss: float
    global_norm_loss: float
    global_weighted_norm_loss: float
    global_image_axis_loss: float
    global_weighted_image_axis_loss: float
    global_image_axis_score_gap: float | None
    global_image_axis_correct_top1: float | None
    global_total_loss: float
    global_row_count: int
    global_sample_count: int
    global_image_axis_row_count: int
    gradient_norm_before_clip: float
    learning_rate: float
    local_sample_ids: tuple[str, ...]
    local_qwen_forward_batch_sizes: tuple[int, ...]
    local_legacy_qwen_forward_batch_sizes: tuple[int, ...]
    local_image_axis_qwen_forward_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.global_step <= 0:
            raise ValueError("image-axis global step must be positive")
        if self.global_row_count <= 0 or self.global_sample_count <= 0:
            raise ValueError("legacy global counts must be positive")
        if self.global_image_axis_row_count < 0:
            raise ValueError("image-axis global count cannot be negative")
        required = (
            self.global_matrix_ce_loss,
            self.global_l_gen_loss,
            self.global_norm_loss,
            self.global_weighted_norm_loss,
            self.global_image_axis_loss,
            self.global_weighted_image_axis_loss,
            self.global_total_loss,
            self.gradient_norm_before_clip,
            self.learning_rate,
        )
        if any(not math.isfinite(value) for value in required):
            raise ValueError("image-axis step metrics must be finite")
        optional = (
            self.global_image_axis_score_gap,
            self.global_image_axis_correct_top1,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("image-axis diagnostic metrics must be finite")
        if self.global_image_axis_row_count == 0:
            if any(value is not None for value in optional):
                raise ValueError("zero eligible rows cannot have image diagnostics")
        elif any(value is None for value in optional):
            raise ValueError("eligible image rows require image diagnostics")
        if (
            self.global_image_axis_correct_top1 is not None
            and not 0.0 <= self.global_image_axis_correct_top1 <= 1.0
        ):
            raise ValueError("image-axis top-1 must lie in [0,1]")
        for field_name in (
            "local_qwen_forward_batch_sizes",
            "local_legacy_qwen_forward_batch_sizes",
            "local_image_axis_qwen_forward_batch_sizes",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in values
            ):
                raise ValueError(f"{field_name} must contain positive integers")
        if self.local_qwen_forward_batch_sizes != (
            *self.local_legacy_qwen_forward_batch_sizes,
            *self.local_image_axis_qwen_forward_batch_sizes,
        ):
            raise ValueError("combined Qwen telemetry differs from its two branches")


class ImageAxisGroundingTrainer:
    """Train only TGVF Adapter parameters with legacy plus image-axis CE."""

    def __init__(
        self,
        *,
        adapter: TGVFAdapter,
        qwen_model: nn.Module,
        family_adapter: QwenVLMFamilyAdapter,
        samples: Sequence[RepresentationTrainingSample],
        sampler: SameImageBatchSampler,
        group_builder: ImageAxisGroundingGroupBuilder,
        objective: RepresentationObjectiveConfigLike,
        image_axis_objective: ImageAxisGroundingObjectiveConfig,
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
            raise TypeError("image-axis trainer samples must be non-empty/typed")
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("image-axis trainer requires SameImageBatchSampler")
        if not callable(group_builder) or not callable(
            getattr(group_builder, "image_axis_row_mask", None)
        ):
            raise TypeError("image-axis group builder interface is incomplete")
        if not isinstance(objective, RepresentationObjectiveConfigV2):
            raise TypeError("image-axis trainer requires norm-aware legacy objective")
        if not isinstance(image_axis_objective, ImageAxisGroundingObjectiveConfig):
            raise TypeError("image_axis_objective must be explicit")
        if not isinstance(accumulation, RepresentationAccumulationIdentity):
            raise TypeError("image-axis accumulation identity is invalid")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("image-axis trainer requires a torch optimizer")
        if scheduler is not None and not isinstance(
            scheduler,
            torch.optim.lr_scheduler.LRScheduler,
        ):
            raise TypeError("image-axis scheduler must be a torch LRScheduler")
        if not isinstance(config, RepresentationTrainerConfig):
            raise TypeError("image-axis execution config is invalid")
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
        self.objective = objective
        self.image_axis_objective = image_axis_objective
        self.accumulation = accumulation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.accumulation_controller = accumulation_controller
        self.global_step = initial_global_step
        self._failed = False

    def train_step(self) -> ImageAxisGroundingStepMetrics:
        if self._failed:
            raise RuntimeError("a failed ImageAxisGroundingTrainer cannot be reused")
        try:
            return self._train_step_impl()
        except BaseException:
            self._failed = True
            raise

    def _train_step_impl(self) -> ImageAxisGroundingStepMetrics:
        self.adapter.train(True)
        self.qwen_model.eval()
        _assert_parameter_ownership(self.adapter, self.qwen_model, self.optimizer)
        group_count, groups_per_microstep = _execution_group_counts(self.accumulation)
        batch_indices = tuple(self.sampler.next_batch() for _ in range(group_count))
        if any(not indices for indices in batch_indices):
            raise RuntimeError("image-axis sampler emitted an empty batch")
        logical_groups = tuple(
            tuple(self.samples[index] for index in indices) for indices in batch_indices
        )
        predicted_masks = tuple(
            self.group_builder.image_axis_row_mask(samples)
            for samples in logical_groups
        )
        for samples, mask in zip(logical_groups, predicted_masks, strict=True):
            _validate_predicted_mask(mask, expected_size=len(samples))

        local_rows = sum(len(samples) for samples in logical_groups)
        local_image_rows = sum(sum(mask) for mask in predicted_masks)
        device = _parameter_device(self.adapter)
        global_rows, global_samples = _global_integer_counts(
            local_rows,
            local_rows,
            device=device,
        )
        global_image_rows = _global_image_axis_count(local_image_rows, device=device)
        normalization = ImageAxisGlobalNormalization(
            legacy=StreamingGlobalNormalization(
                matrix_valid_rows=global_rows,
                l_gen_samples=global_samples,
                data_parallel_world_size=self.accumulation.data_parallel_world_size,
            ),
            image_axis_valid_rows=global_image_rows,
        )
        collective_counts = synchronize_collective_candidate_counts(
            tuple(len(samples) for samples in logical_groups),
            device=device,
        )

        self.optimizer.zero_grad(set_to_none=True)
        local_totals = torch.zeros(7, dtype=torch.float64, device=device)
        local_ids: list[str] = []
        legacy_schedule: list[int] = []
        image_schedule: list[int] = []
        microstep_count = group_count // groups_per_microstep
        active_microstep = -1
        failure: BaseException | None = None
        try:
            for group_index, (samples, expected_mask, collective_count) in enumerate(
                zip(
                    logical_groups,
                    predicted_masks,
                    collective_counts,
                    strict=True,
                )
            ):
                microstep_index = group_index // groups_per_microstep
                if microstep_index != active_microstep:
                    active_microstep = microstep_index
                    if self.accumulation_controller is not None:
                        self.accumulation_controller.begin_microstep(
                            index=microstep_index,
                            count=microstep_count,
                        )
                with self._autocast_context():
                    group = self.group_builder(
                        samples,
                        self.adapter,
                        collective_candidate_count=collective_count,
                    )
                    _assert_group_identity(
                        group,
                        expected_ids=tuple(sample.sample_id for sample in samples),
                        expected_mask=expected_mask,
                        collective_candidate_count=collective_count,
                    )
                    metrics = backward_image_axis_grounding_group(
                        self.family_adapter,
                        self.qwen_model,
                        group,
                        image_axis_objective=self.image_axis_objective,
                        legacy_objective=self.objective,
                        normalization=normalization,
                    )
                _accumulate_metrics(
                    local_totals,
                    metrics,
                    expected_sample_count=len(samples),
                    expected_image_count=sum(expected_mask),
                )
                local_ids.extend(sample.sample_id for sample in samples)
                legacy_schedule.extend(metrics.legacy.qwen_forward_batch_sizes)
                image_schedule.extend(metrics.image_axis_qwen_forward_batch_sizes)
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
                        "image-axis accumulation-controller restoration also failed: "
                        f"{finish_error!r}"
                    )

        trainable = _adapter_owned_trainable_parameters(self.adapter)
        _assert_gradients(
            trainable,
            require_all=self.config.require_all_adapter_gradients,
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

        matrix, evidence, norm, image, correct, wrong, top1 = (
            float(value) for value in _global_sum(local_totals).tolist()
        )
        matrix_loss = matrix / global_rows
        evidence_loss = evidence / global_samples
        norm_loss = norm / global_samples
        weighted_norm = norm_loss * self.objective.norm_weight
        image_loss = image / global_image_rows if global_image_rows else 0.0
        weighted_image = (
            image_loss * self.image_axis_objective.image_axis_matrix_weight
        )
        image_gap = (
            (correct - wrong) / global_image_rows if global_image_rows else None
        )
        image_top1 = top1 / global_image_rows if global_image_rows else None
        total = (
            matrix_loss * self.objective.matrix_ce_weight
            + evidence_loss * self.objective.l_gen_weight
            + weighted_norm
            + weighted_image
        )
        return ImageAxisGroundingStepMetrics(
            global_step=self.global_step,
            global_matrix_ce_loss=matrix_loss,
            global_l_gen_loss=evidence_loss,
            global_norm_loss=norm_loss,
            global_weighted_norm_loss=weighted_norm,
            global_image_axis_loss=image_loss,
            global_weighted_image_axis_loss=weighted_image,
            global_image_axis_score_gap=image_gap,
            global_image_axis_correct_top1=image_top1,
            global_total_loss=total,
            global_row_count=global_rows,
            global_sample_count=global_samples,
            global_image_axis_row_count=global_image_rows,
            gradient_norm_before_clip=gradient_norm,
            learning_rate=learning_rate,
            local_sample_ids=tuple(local_ids),
            local_qwen_forward_batch_sizes=tuple((*legacy_schedule, *image_schedule)),
            local_legacy_qwen_forward_batch_sizes=tuple(legacy_schedule),
            local_image_axis_qwen_forward_batch_sizes=tuple(image_schedule),
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
    if direct_groups > 1:
        raise ValueError(
            "image-axis v1 rejects direct multi-group microsteps because each "
            "paired group owns one combined FSDP boundary backward"
        )
    return accumulation_steps, 1


def _validate_predicted_mask(mask: object, *, expected_size: int) -> None:
    if not isinstance(mask, tuple) or len(mask) != expected_size:
        raise ValueError("predicted image-axis mask must contain one value per row")
    if any(type(value) is not bool for value in mask):
        raise TypeError("predicted image-axis mask values must be bool")
    if not (all(mask) or not any(mask)):
        raise ValueError("predicted image-axis mask must be group-homogeneous")


def _assert_group_identity(
    group: ImageAxisGroundingGroup,
    *,
    expected_ids: tuple[str, ...],
    expected_mask: tuple[bool, ...],
    collective_candidate_count: int,
) -> None:
    if not isinstance(group, ImageAxisGroundingGroup):
        raise TypeError("image-axis builder returned an invalid group")
    base_ids = tuple(row.sample_id for row in group.base.rows)
    donor_ids = tuple(candidate.sample_id for candidate in group.donor.candidates)
    if base_ids != expected_ids or donor_ids != expected_ids:
        raise ValueError("image-axis builder changed sampler identity/order")
    if group.image_axis_row_mask != expected_mask:
        raise ValueError("built image-axis mask differs from manifest preflight")
    if (
        group.base.collective_candidate_count != collective_candidate_count
        or group.donor.collective_candidate_count != collective_candidate_count
    ):
        raise ValueError("image-axis builder changed collective candidate count")


def _global_image_axis_count(local_count: int, *, device: torch.device) -> int:
    if isinstance(local_count, bool) or not isinstance(local_count, int) or local_count < 0:
        raise ValueError("local image-axis count must be non-negative")
    value = torch.tensor(local_count, dtype=torch.int64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return int(value.item())


def _accumulate_metrics(
    totals: torch.Tensor,
    metrics: ImageAxisStreamingMetrics,
    *,
    expected_sample_count: int,
    expected_image_count: int,
) -> None:
    if totals.shape != (7,):
        raise ValueError("image-axis metric accumulator must have seven components")
    if (
        metrics.legacy.local_row_count != expected_sample_count
        or metrics.legacy.local_sample_count != expected_sample_count
        or metrics.local_image_axis_row_count != expected_image_count
    ):
        raise RuntimeError("image-axis streaming returned incorrect local counts")
    if metrics.legacy.norm_numerator is None:
        raise RuntimeError("image-axis experiment requires historical norm metrics")
    values = (
        metrics.legacy.matrix_ce_numerator,
        metrics.legacy.l_gen_numerator,
        metrics.legacy.norm_numerator,
        metrics.image_axis_numerator,
        metrics.correct_score_sum,
        metrics.wrong_score_sum,
        totals.new_tensor(float(metrics.correct_top1_count)),
    )
    materialized = torch.stack(
        tuple(
            value.detach().to(device=totals.device, dtype=totals.dtype)
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
    "IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION",
    "ImageAxisGroundingGroupBuilder",
    "ImageAxisGroundingObjectiveConfig",
    "ImageAxisGroundingStepMetrics",
    "ImageAxisGroundingTrainer",
]
