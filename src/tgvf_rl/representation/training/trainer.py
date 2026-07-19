"""Executable representation-phase optimizer loop over same-image groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
import math
from typing import ContextManager, Protocol

import torch
from torch import nn

from tgvf_rl.qwen.base import QwenVLMFamilyAdapter
from tgvf_rl.representation.adapter import TGVFAdapter

from .checkpoint import RepresentationAccumulationIdentity
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
)
from .readout import SameImageReadoutGroup
from .sampling import SameImageBatchSampler
from .schema import RepresentationTrainingSample
from .streaming import (
    StreamingGlobalNormalization,
    backward_streaming_same_image_group,
    backward_streaming_same_image_groups,
    score_streaming_same_image_group,
    score_streaming_same_image_groups,
)


_BORROWED_PROJECTION_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)


class RepresentationPrecision(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"


class RepresentationSchedulerKind(str, Enum):
    CONSTANT = "constant"
    LINEAR_WARMUP_DECAY = "linear_warmup_decay"
    HISTORICAL_COSINE = "historical_cosine"


@dataclass(frozen=True, slots=True)
class RepresentationOptimizerConfig:
    """Explicit AdamW contract; no scientific values are defaulted."""

    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    def __post_init__(self) -> None:
        _positive_finite(self.learning_rate, field_name="learning_rate")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(not isinstance(value, float) for value in self.betas)
            or not 0 <= self.betas[0] < 1
            or not 0 <= self.betas[1] < 1
        ):
            raise ValueError("betas must be an explicit pair in [0,1)")
        _positive_finite(self.eps, field_name="eps")
        _non_negative_finite(self.weight_decay, field_name="weight_decay")


@dataclass(frozen=True, slots=True)
class RepresentationSchedulerConfig:
    kind: RepresentationSchedulerKind
    total_steps: int
    warmup_steps: int
    min_lr_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepresentationSchedulerKind):
            raise TypeError("scheduler kind must be explicit")
        _positive_int(self.total_steps, field_name="total_steps")
        _non_negative_int(self.warmup_steps, field_name="warmup_steps")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if self.kind is RepresentationSchedulerKind.CONSTANT and self.warmup_steps:
            raise ValueError("constant scheduler cannot have warmup steps")
        if self.kind is RepresentationSchedulerKind.HISTORICAL_COSINE:
            if not isinstance(self.min_lr_ratio, float):
                raise TypeError(
                    "historical cosine min_lr_ratio must be an explicit float"
                )
            if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
                raise ValueError("historical cosine min_lr_ratio must be in [0,1]")
        elif self.min_lr_ratio is not None:
            raise ValueError(
                "min_lr_ratio is supported only by the historical cosine scheduler"
            )


@dataclass(frozen=True, slots=True)
class RepresentationTrainerConfig:
    precision: RepresentationPrecision
    max_grad_norm: float
    require_all_adapter_gradients: bool

    def __post_init__(self) -> None:
        if not isinstance(self.precision, RepresentationPrecision):
            raise TypeError("representation precision must be explicit")
        _positive_finite(self.max_grad_norm, field_name="max_grad_norm")
        if not isinstance(self.require_all_adapter_gradients, bool):
            raise TypeError("require_all_adapter_gradients must be bool")


class RepresentationGroupBuilder(Protocol):
    """Build one differentiable group from one sampler-owned image batch."""

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup: ...


@dataclass(frozen=True, slots=True)
class RepresentationStepMetrics:
    global_step: int
    global_matrix_ce_loss: float
    global_l_gen_loss: float
    global_norm_loss: float | None
    global_weighted_norm_loss: float | None
    global_total_loss: float
    global_row_count: int
    global_sample_count: int
    gradient_norm_before_clip: float
    learning_rate: float
    local_sample_ids: tuple[str, ...]


class RepresentationTrainer:
    """Synchronous optimizer-boundary trainer with exact global reductions."""

    def __init__(
        self,
        *,
        adapter: TGVFAdapter,
        qwen_model: nn.Module,
        family_adapter: QwenVLMFamilyAdapter,
        samples: Sequence[RepresentationTrainingSample],
        sampler: SameImageBatchSampler,
        group_builder: RepresentationGroupBuilder,
        objective: RepresentationObjectiveConfigLike,
        accumulation: RepresentationAccumulationIdentity,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        config: RepresentationTrainerConfig,
        initial_global_step: int = 0,
    ) -> None:
        if not isinstance(adapter, TGVFAdapter):
            raise TypeError("adapter must be TGVFAdapter")
        if not isinstance(qwen_model, nn.Module):
            raise TypeError("qwen_model must be nn.Module")
        if not isinstance(family_adapter, QwenVLMFamilyAdapter):
            raise TypeError("family_adapter must be QwenVLMFamilyAdapter")
        if not samples or not all(
            isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise TypeError(
                "samples must be non-empty RepresentationTrainingSample values"
            )
        if not isinstance(sampler, SameImageBatchSampler):
            raise TypeError("sampler must be SameImageBatchSampler")
        if not callable(group_builder):
            raise TypeError("group_builder must be callable")
        if not isinstance(
            objective,
            (RepresentationObjectiveConfig, RepresentationObjectiveConfigV2),
        ):
            raise TypeError("objective must be a representation objective config")
        if not isinstance(accumulation, RepresentationAccumulationIdentity):
            raise TypeError("accumulation must be RepresentationAccumulationIdentity")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.LRScheduler
        ):
            raise TypeError("scheduler must be a torch LRScheduler")
        if not isinstance(config, RepresentationTrainerConfig):
            raise TypeError("config must be RepresentationTrainerConfig")
        _non_negative_int(initial_global_step, field_name="initial_global_step")

        self.adapter = adapter
        self.qwen_model = qwen_model
        self.family_adapter = family_adapter
        self.samples = tuple(samples)
        self.sampler = sampler
        self.group_builder = group_builder
        self.objective = objective
        self.accumulation = accumulation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.global_step = initial_global_step
        _assert_parameter_ownership(adapter, qwen_model, optimizer)
        _assert_distributed_identity(accumulation.data_parallel_world_size)

    def train_step(self) -> RepresentationStepMetrics:
        """Execute one optimizer step over a complete accumulation window."""

        self.adapter.train(True)
        self.qwen_model.eval()
        _assert_parameter_ownership(self.adapter, self.qwen_model, self.optimizer)

        direct_groups = getattr(
            self.accumulation,
            "groups_per_rank_per_optimizer_step",
            1,
        )
        if direct_groups > 1 and self.accumulation.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "direct multi-group execution requires gradient accumulation one"
            )
        group_count = (
            direct_groups
            if direct_groups > 1
            else self.accumulation.gradient_accumulation_steps
        )
        batch_indices = tuple(self.sampler.next_batch() for _ in range(group_count))
        if any(not indices for indices in batch_indices):
            raise RuntimeError("representation sampler emitted an empty batch")
        local_rows = sum(len(indices) for indices in batch_indices)
        global_rows, global_samples = _global_integer_counts(
            local_rows,
            local_rows,
            device=_parameter_device(self.adapter),
        )
        collective_candidate_counts = synchronize_collective_candidate_counts(
            tuple(len(indices) for indices in batch_indices),
            device=_parameter_device(self.adapter),
        )
        normalization = StreamingGlobalNormalization(
            matrix_valid_rows=global_rows,
            l_gen_samples=global_samples,
            data_parallel_world_size=self.accumulation.data_parallel_world_size,
        )

        self.optimizer.zero_grad(set_to_none=True)
        local_matrix_numerator = 0.0
        local_l_gen_numerator = 0.0
        local_norm_numerator = (
            0.0 if isinstance(self.objective, RepresentationObjectiveConfigV2) else None
        )
        local_sample_ids: list[str] = []
        if direct_groups > 1:
            groups: list[SameImageReadoutGroup] = []
            expected_ids_by_group: list[tuple[str, ...]] = []
            with self._autocast_context():
                for indices, collective_candidate_count in zip(
                    batch_indices,
                    collective_candidate_counts,
                    strict=True,
                ):
                    logical_samples = tuple(self.samples[index] for index in indices)
                    group = self.group_builder(
                        logical_samples,
                        self.adapter,
                        collective_candidate_count=collective_candidate_count,
                    )
                    expected_ids = tuple(sample.sample_id for sample in logical_samples)
                    _assert_built_group_identity(
                        group,
                        expected_ids=expected_ids,
                        collective_candidate_count=collective_candidate_count,
                    )
                    groups.append(group)
                    expected_ids_by_group.append(expected_ids)
                scores = score_streaming_same_image_groups(
                    self.family_adapter,
                    self.qwen_model,
                    groups,
                )
                backward_metrics = backward_streaming_same_image_groups(
                    self.family_adapter,
                    self.qwen_model,
                    groups,
                    scores,
                    objective=self.objective,
                    normalization=normalization,
                )
            if (
                backward_metrics.local_row_count != local_rows
                or backward_metrics.local_sample_count != local_rows
            ):
                raise RuntimeError(
                    "direct multi-group execution returned incorrect local counts"
                )
            local_matrix_numerator = float(
                backward_metrics.matrix_ce_numerator.float().item()
            )
            local_l_gen_numerator = float(
                backward_metrics.l_gen_numerator.float().item()
            )
            if local_norm_numerator is not None:
                if backward_metrics.norm_numerator is None:
                    raise RuntimeError("objective v2 did not return a norm numerator")
                local_norm_numerator = float(
                    backward_metrics.norm_numerator.float().item()
                )
            local_sample_ids.extend(
                sample_id
                for expected_ids in expected_ids_by_group
                for sample_id in expected_ids
            )
        else:
            for indices, collective_candidate_count in zip(
                batch_indices,
                collective_candidate_counts,
                strict=True,
            ):
                logical_samples = tuple(self.samples[index] for index in indices)
                with self._autocast_context():
                    group = self.group_builder(
                        logical_samples,
                        self.adapter,
                        collective_candidate_count=collective_candidate_count,
                    )
                    expected_ids = tuple(sample.sample_id for sample in logical_samples)
                    _assert_built_group_identity(
                        group,
                        expected_ids=expected_ids,
                        collective_candidate_count=collective_candidate_count,
                    )
                    scores = score_streaming_same_image_group(
                        self.family_adapter, self.qwen_model, group
                    )
                    backward_metrics = backward_streaming_same_image_group(
                        self.family_adapter,
                        self.qwen_model,
                        group,
                        scores,
                        objective=self.objective,
                        normalization=normalization,
                    )
                local_matrix_numerator += float(
                    backward_metrics.matrix_ce_numerator.float().item()
                )
                local_l_gen_numerator += float(
                    backward_metrics.l_gen_numerator.float().item()
                )
                if local_norm_numerator is not None:
                    if backward_metrics.norm_numerator is None:
                        raise RuntimeError(
                            "objective v2 did not return a norm numerator"
                        )
                    local_norm_numerator += float(
                        backward_metrics.norm_numerator.float().item()
                    )
                local_sample_ids.extend(expected_ids)

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
        used_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        _assert_qwen_has_no_gradients(self.qwen_model)

        (
            global_matrix_numerator,
            global_l_gen_numerator,
            global_norm_numerator,
        ) = _global_float_sums(
            local_matrix_numerator,
            local_l_gen_numerator,
            local_norm_numerator,
            device=_parameter_device(self.adapter),
        )
        matrix_loss = global_matrix_numerator / global_rows
        l_gen_loss = global_l_gen_numerator / global_samples
        norm_loss = (
            None
            if global_norm_numerator is None
            else global_norm_numerator / global_samples
        )
        weighted_norm_loss = (
            None if norm_loss is None else norm_loss * self.objective.norm_weight  # type: ignore[union-attr]
        )
        total_loss = (
            matrix_loss * self.objective.matrix_ce_weight
            + l_gen_loss * self.objective.l_gen_weight
        )
        if weighted_norm_loss is not None:
            total_loss += weighted_norm_loss
        return RepresentationStepMetrics(
            global_step=self.global_step,
            global_matrix_ce_loss=matrix_loss,
            global_l_gen_loss=l_gen_loss,
            global_norm_loss=norm_loss,
            global_weighted_norm_loss=weighted_norm_loss,
            global_total_loss=total_loss,
            global_row_count=global_rows,
            global_sample_count=global_samples,
            gradient_norm_before_clip=float(gradient_norm.detach().float().item()),
            learning_rate=used_learning_rate,
            local_sample_ids=tuple(local_sample_ids),
        )

    def fit(
        self,
        *,
        target_global_step: int,
        on_step: Callable[[RepresentationStepMetrics], None] | None = None,
    ) -> tuple[RepresentationStepMetrics, ...]:
        _positive_int(target_global_step, field_name="target_global_step")
        if target_global_step <= self.global_step:
            raise ValueError("target_global_step must exceed the current global step")
        if on_step is not None and not callable(on_step):
            raise TypeError("on_step must be callable")
        results = []
        while self.global_step < target_global_step:
            result = self.train_step()
            results.append(result)
            if on_step is not None:
                on_step(result)
        return tuple(results)

    def _autocast_context(self) -> ContextManager[object]:
        if self.config.precision is RepresentationPrecision.FP32:
            return nullcontext()
        device_type = _parameter_device(self.adapter).type
        if device_type not in {"cpu", "cuda"}:
            raise ValueError(
                f"BF16 autocast is unsupported on device type {device_type!r}"
            )
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def build_representation_optimizer(
    adapter: TGVFAdapter,
    config: RepresentationOptimizerConfig,
) -> torch.optim.AdamW:
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be TGVFAdapter")
    if not isinstance(config, RepresentationOptimizerConfig):
        raise TypeError("config must be RepresentationOptimizerConfig")
    parameters = tuple(
        parameter for _, parameter in _adapter_owned_trainable_parameters(adapter)
    )
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_representation_scheduler(
    optimizer: torch.optim.Optimizer,
    config: RepresentationSchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(config, RepresentationSchedulerConfig):
        raise TypeError("config must be RepresentationSchedulerConfig")

    def multiplier(step: int) -> float:
        if config.kind is RepresentationSchedulerKind.CONSTANT:
            return 1.0
        if config.kind is RepresentationSchedulerKind.HISTORICAL_COSINE:
            if config.warmup_steps > 0 and step < config.warmup_steps:
                return max(
                    float(step + 1) / float(config.warmup_steps),
                    1e-8,
                )
            decay_steps = max(1, config.total_steps - config.warmup_steps)
            progress = min(
                1.0,
                max(
                    0.0,
                    float(step - config.warmup_steps + 1) / float(decay_steps),
                ),
            )
            # The pinned historical launcher evaluated cosine through a default
            # FP32 torch scalar rather than Python's double-precision math.cos.
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            if config.min_lr_ratio is None:  # guarded by config validation
                raise RuntimeError("historical cosine min_lr_ratio is absent")
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
        if step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        remaining = config.total_steps - step
        decay_steps = config.total_steps - config.warmup_steps
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _assert_parameter_ownership(
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    qwen_parameters = tuple(qwen_model.parameters())
    if any(parameter.requires_grad for parameter in qwen_parameters):
        raise ValueError("every Qwen parameter must be frozen")
    _assert_qwen_has_no_gradients(qwen_model)
    trainable = _adapter_owned_trainable_parameters(adapter)
    borrowed = tuple(
        (name, parameter)
        for name, parameter in adapter.named_parameters()
        if name.startswith(_BORROWED_PROJECTION_PREFIXES)
    )
    if not borrowed or any(parameter.requires_grad for _, parameter in borrowed):
        raise ValueError("borrowed Qwen merger parameters must exist and remain frozen")
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    expected_ids = {id(parameter) for _, parameter in trainable}
    actual_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_parameters) != len(actual_ids) or actual_ids != expected_ids:
        raise ValueError(
            "optimizer must own every and only trainable TGVF Adapter parameter"
        )


def _adapter_owned_trainable_parameters(
    adapter: TGVFAdapter,
) -> tuple[tuple[str, nn.Parameter], ...]:
    owned = tuple(
        (name, parameter)
        for name, parameter in adapter.named_parameters()
        if not name.startswith(_BORROWED_PROJECTION_PREFIXES)
    )
    if not owned:
        raise RuntimeError("TGVF Adapter has no owned parameters")
    frozen = tuple(name for name, parameter in owned if not parameter.requires_grad)
    if frozen:
        raise ValueError(f"Adapter-owned parameters unexpectedly frozen: {frozen}")
    return owned


def _assert_gradients(
    parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    require_all: bool,
) -> None:
    missing = tuple(name for name, parameter in parameters if parameter.grad is None)
    if require_all:
        missing_by_rank = _gather_missing_gradient_names(missing)
        if any(missing_by_rank):
            raise RuntimeError(
                "Adapter-owned parameters received no gradient by rank: "
                f"{missing_by_rank}"
            )
    local_finite = True
    has_dtensor = False
    for _name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        local_gradient, is_dtensor = _local_gradient_shard(gradient)
        has_dtensor = has_dtensor or is_dtensor
        local_finite = local_finite and bool(
            torch.isfinite(local_gradient).all().item()
        )
    if has_dtensor:
        local_finite = _distributed_boolean_and(
            local_finite,
            device=_parameter_device_from_pairs(parameters),
        )
    if not local_finite:
        raise FloatingPointError("non-finite Adapter gradient on at least one rank")


def _clip_adapter_grad_norm_(
    parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    max_norm: float,
) -> torch.Tensor:
    """Clip one global L2 norm for plain or one-dimensional FSDP2 shards.

    Composable FSDP2 exposes sharded gradients as DTensors.  Computing a norm
    independently on each rank would apply different clipping coefficients.
    This implementation sums each local shard's FP32 squared norm exactly
    once, performs one global SUM for FSDP2, and applies the same coefficient
    on every rank.  The ``1e-6`` denominator epsilon matches the fixed PyTorch
    L2 clipping convention exercised by the single-process parity tests.
    """

    _positive_finite(max_norm, field_name="max_norm")
    gradients: list[tuple[torch.Tensor, bool]] = []
    saw_dtensor = False
    saw_plain = False
    squared_norm: torch.Tensor | None = None
    for _name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        local_gradient, is_dtensor = _local_gradient_shard(gradient)
        saw_dtensor = saw_dtensor or is_dtensor
        saw_plain = saw_plain or not is_dtensor
        gradients.append((local_gradient, is_dtensor))
        contribution = local_gradient.detach().float().square().sum()
        squared_norm = (
            contribution if squared_norm is None else squared_norm + contribution
        )
    if not gradients or squared_norm is None:
        raise RuntimeError("cannot clip an empty Adapter gradient set")
    if saw_dtensor and saw_plain:
        raise RuntimeError("FSDP2 Adapter gradients cannot mix DTensor and plain state")
    if saw_dtensor:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "DTensor gradients require initialized distributed state"
            )
        torch.distributed.all_reduce(
            squared_norm,
            op=torch.distributed.ReduceOp.SUM,
        )
    total_norm = torch.sqrt(squared_norm)
    if not bool(torch.isfinite(total_norm).item()):
        raise FloatingPointError("non-finite global Adapter gradient norm")
    coefficient = torch.clamp(
        torch.tensor(max_norm, device=total_norm.device, dtype=total_norm.dtype)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for local_gradient, _is_dtensor in gradients:
        local_gradient.mul_(coefficient.to(dtype=local_gradient.dtype))
    return total_norm


def _local_gradient_shard(gradient: torch.Tensor) -> tuple[torch.Tensor, bool]:
    try:
        from torch.distributed.tensor import DTensor, Shard
    except (ImportError, AttributeError):  # pragma: no cover - old torch fails earlier
        DTensor = ()  # type: ignore[assignment,misc]
        Shard = ()  # type: ignore[assignment,misc]
    if not isinstance(gradient, DTensor):
        return gradient, False
    placements = gradient.placements
    if len(placements) != 1 or not isinstance(placements[0], Shard):
        raise RuntimeError(
            "representation FSDP2 gradients must use one one-dimensional Shard"
        )
    local = gradient.to_local()
    if not isinstance(local, torch.Tensor):
        raise TypeError("DTensor local gradient shard must be a Tensor")
    return local, True


def _distributed_boolean_and(value: bool, *, device: torch.device) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _gather_missing_gradient_names(
    missing: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed:
        return (missing,)
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, missing)
    if not all(
        isinstance(value, tuple) and all(isinstance(name, str) for name in value)
        for value in gathered
    ):
        raise TypeError("distributed missing-gradient gather returned malformed state")
    return tuple(gathered)  # type: ignore[arg-type]


def _parameter_device_from_pairs(
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> torch.device:
    for _name, parameter in parameters:
        local, _is_dtensor = _local_gradient_shard(
            parameter.grad if parameter.grad is not None else parameter
        )
        return local.device
    raise ValueError("Adapter parameter sequence is empty")


def _assert_qwen_has_no_gradients(model: nn.Module) -> None:
    polluted = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    if polluted:
        raise RuntimeError(f"frozen Qwen parameters received gradients: {polluted}")


def _assert_distributed_identity(expected_world_size: int) -> None:
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    actual_world_size = torch.distributed.get_world_size() if distributed else 1
    if actual_world_size != expected_world_size:
        raise ValueError(
            "data-parallel world size differs from the accumulation identity: "
            f"expected={expected_world_size} actual={actual_world_size}"
        )


def _assert_built_group_identity(
    group: SameImageReadoutGroup,
    *,
    expected_ids: tuple[str, ...],
    collective_candidate_count: int,
) -> None:
    actual_ids = tuple(row.sample_id for row in group.rows)
    if actual_ids != expected_ids:
        raise ValueError("group builder changed sampler-owned sample order/identity")
    if group.collective_candidate_count != collective_candidate_count:
        raise ValueError(
            "group builder did not materialize the synchronized collective "
            "candidate count"
        )


def _global_integer_counts(
    matrix_rows: int,
    l_gen_samples: int,
    *,
    device: torch.device,
) -> tuple[int, int]:
    counts = torch.tensor(
        [matrix_rows, l_gen_samples], dtype=torch.int64, device=device
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


def synchronize_collective_candidate_counts(
    local_counts: Sequence[int],
    *,
    device: torch.device,
) -> tuple[int, ...]:
    """Return the per-microstep global maximum K on every data-parallel rank.

    Composable FSDP issues collectives once per Adapter forward/backward.  The
    same-image parity sampler may emit four or five real candidates locally,
    so every rank first agrees on a maximum and materializes loss-excluded
    zero-gradient padding up to that count.
    """

    counts = tuple(local_counts)
    if not counts:
        raise ValueError("collective candidate counts cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in counts
    ):
        raise ValueError("every collective candidate count must be an integer K>=2")
    values = torch.tensor(counts, dtype=torch.int64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    result = tuple(int(value.item()) for value in values)
    if any(
        global_count < local_count for global_count, local_count in zip(result, counts)
    ):
        raise RuntimeError("collective candidate maximum is smaller than local K")
    return result


def _global_float_sums(
    matrix_numerator: float,
    l_gen_numerator: float,
    norm_numerator: float | None,
    *,
    device: torch.device,
) -> tuple[float, float, float | None]:
    local_values = [matrix_numerator, l_gen_numerator]
    if norm_numerator is not None:
        local_values.append(norm_numerator)
    values = torch.tensor(local_values, dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return (
        float(values[0].item()),
        float(values[1].item()),
        None if norm_numerator is None else float(values[2].item()),
    )


def _parameter_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("module must expose at least one parameter")
    return parameter.device


def _positive_finite(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be an explicit positive finite float")


def _non_negative_finite(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be an explicit non-negative finite float")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


__all__ = [
    "RepresentationGroupBuilder",
    "RepresentationOptimizerConfig",
    "RepresentationPrecision",
    "RepresentationSchedulerConfig",
    "RepresentationSchedulerKind",
    "RepresentationStepMetrics",
    "RepresentationTrainer",
    "RepresentationTrainerConfig",
    "build_representation_optimizer",
    "build_representation_scheduler",
    "synchronize_collective_candidate_counts",
]
