"""Deterministic, stateless validation for representation-phase training."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import math
import random
from typing import Iterator

import torch
from torch import nn

from tgvf_rl.qwen.base import QwenVLMFamilyAdapter
from tgvf_rl.representation.adapter import TGVFAdapter

from .losses import same_image_matrix_ce_loss_terms
from .objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigLike,
    RepresentationObjectiveConfigV2,
)
from .readout import SameImageReadoutGroup
from .sampling import SameImageBatchSampler
from .schema import RepresentationTrainingSample
from .streaming import score_streaming_same_image_group
from .trainer import (
    RepresentationGroupBuilder,
    synchronize_collective_candidate_counts,
)


@dataclass(frozen=True, slots=True)
class RepresentationValidationMetrics:
    """Globally reduced metrics for exactly one validation group per rank."""

    validation_event_index: int
    global_matrix_ce_loss: float
    global_l_gen_loss: float
    global_norm_loss: float | None
    global_weighted_norm_loss: float | None
    global_total_loss: float
    global_row_count: int
    global_sample_count: int
    global_evidence_token_count: int
    global_group_count: int
    local_rank: int
    data_parallel_world_size: int
    local_image_group_key: str
    local_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_negative_int(
            self.validation_event_index, field_name="validation_event_index"
        )
        for field_name in (
            "global_matrix_ce_loss",
            "global_l_gen_loss",
            "global_total_loss",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite float")
        for field_name in ("global_norm_loss", "global_weighted_norm_loss"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be None or a finite float")
        if (self.global_norm_loss is None) != (self.global_weighted_norm_loss is None):
            raise ValueError("raw and weighted validation norm metrics must align")
        for field_name in (
            "global_row_count",
            "global_sample_count",
            "global_evidence_token_count",
            "global_group_count",
            "data_parallel_world_size",
        ):
            _positive_int(getattr(self, field_name), field_name=field_name)
        _non_negative_int(self.local_rank, field_name="local_rank")
        if self.local_rank >= self.data_parallel_world_size:
            raise ValueError("local_rank must be smaller than data_parallel_world_size")
        if self.global_group_count != self.data_parallel_world_size:
            raise ValueError("validation must evaluate exactly one group per rank")
        if self.global_row_count != self.global_sample_count:
            raise ValueError("Matrix-CE row and L_gen sample counts must align")
        if self.global_evidence_token_count < self.global_sample_count:
            raise ValueError("every validation sample needs an evidence token")
        if not isinstance(self.local_image_group_key, str) or not (
            self.local_image_group_key.strip()
        ):
            raise ValueError("local_image_group_key must be non-empty")
        if len(self.local_sample_ids) < 2 or len(set(self.local_sample_ids)) != len(
            self.local_sample_ids
        ):
            raise ValueError(
                "local validation sample IDs must contain unique K>=2 rows"
            )


def evaluate_representation_validation_event(
    *,
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
    family_adapter: QwenVLMFamilyAdapter,
    samples: Sequence[RepresentationTrainingSample],
    group_builder: RepresentationGroupBuilder,
    objective: RepresentationObjectiveConfigLike,
    batch_size: int,
    sampler_seed: int,
    data_manifest_sha256: str,
    validation_event_index: int,
    data_parallel_world_size: int,
) -> RepresentationValidationMetrics:
    """Evaluate one deterministic same-image group on every data-parallel rank.

    A fresh :class:`SameImageBatchSampler` is constructed for every call and
    advanced from its initial state to ``validation_event_index``.  Therefore
    validation has no mutable sampler cursor that could enter a checkpoint or
    perturb the next training batch.  Model/RNG modes are similarly isolated;
    distributed side effects are one MAX reduction for FSDP collective padding
    and two SUM reductions for real-sample metrics.
    """

    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")
    if not isinstance(qwen_model, nn.Module):
        raise TypeError("qwen_model must be an nn.Module")
    if not isinstance(family_adapter, QwenVLMFamilyAdapter):
        raise TypeError("family_adapter must be a QwenVLMFamilyAdapter")
    if not samples or not all(
        isinstance(sample, RepresentationTrainingSample) for sample in samples
    ):
        raise TypeError("samples must be non-empty RepresentationTrainingSample values")
    if not callable(group_builder):
        raise TypeError("group_builder must be callable")
    if not isinstance(
        objective, (RepresentationObjectiveConfig, RepresentationObjectiveConfigV2)
    ):
        raise TypeError("objective must be a representation objective config")
    _positive_int(batch_size, field_name="batch_size")
    if batch_size < 2:
        raise ValueError("same-image validation requires batch_size >= 2")
    _integer(sampler_seed, field_name="sampler_seed")
    _sha256(data_manifest_sha256, field_name="data_manifest_sha256")
    _non_negative_int(validation_event_index, field_name="validation_event_index")
    _positive_int(data_parallel_world_size, field_name="data_parallel_world_size")

    rank, world_size = _distributed_identity(data_parallel_world_size)
    sampler = SameImageBatchSampler(
        samples,
        batch_size=batch_size,
        seed=sampler_seed,
        data_manifest_sha256=data_manifest_sha256,
        rank=rank,
        world_size=world_size,
    )
    batch_indices: tuple[int, ...] = ()
    for _ in range(validation_event_index + 1):
        batch_indices = sampler.next_batch()
    if not batch_indices:
        raise RuntimeError("fresh validation sampler emitted an empty batch")
    logical_samples = tuple(samples[index] for index in batch_indices)
    (collective_candidate_count,) = synchronize_collective_candidate_counts(
        (len(logical_samples),),
        device=_module_device(adapter),
    )

    # Qwen is not an optimization target in this phase. Freeze it explicitly,
    # rather than relying on a caller to have preserved the runtime mode.
    qwen_model.requires_grad_(False)
    qwen_model.eval()
    if any(parameter.grad is not None for parameter in qwen_model.parameters()):
        raise RuntimeError("frozen Qwen carries a gradient into validation")

    adapter_modes = tuple((module, module.training) for module in adapter.modules())
    adapter_parameter_state = tuple(
        (
            parameter,
            parameter.requires_grad,
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in adapter.parameters()
    )
    adapter.eval()
    qwen_model.eval()
    event_seed = _validation_event_seed(
        sampler_seed=sampler_seed,
        validation_event_index=validation_event_index,
        rank=rank,
        data_manifest_sha256=data_manifest_sha256,
    )
    try:
        with _isolated_validation_rng(event_seed, device=_module_device(adapter)):
            with torch.no_grad():
                group = group_builder(
                    logical_samples,
                    adapter,
                    collective_candidate_count=collective_candidate_count,
                )
                _validate_group(
                    group,
                    logical_samples,
                    collective_candidate_count=collective_candidate_count,
                )
                _assert_visuals_do_not_require_grad(group)
                scores = score_streaming_same_image_group(
                    family_adapter,
                    qwen_model,
                    group,
                    objective=objective,
                )
    finally:
        # Restore Adapter-owned training modes exactly. Shared borrowed Qwen
        # mergers are forced back to Qwen eval after this restoration.
        for module, was_training in adapter_modes:
            module.training = was_training
        qwen_model.eval()
    _assert_adapter_gradient_state_unchanged(adapter_parameter_state)
    if any(parameter.grad is not None for parameter in qwen_model.parameters()):
        raise RuntimeError("validation created a frozen-Qwen gradient")

    matrix_terms = same_image_matrix_ce_loss_terms((scores.score_matrix,))
    local_matrix_numerator = float(matrix_terms.numerator.float().item())
    local_l_gen_numerator = float(scores.diagonal_l_gen.float().sum().item())
    local_norm_numerator = (
        float(scores.historical_norm.float().sum().item())
        if isinstance(objective, RepresentationObjectiveConfigV2)
        else None
    )
    local_evidence_tokens = int(scores.evidence_token_counts.sum().item())
    local_rows = matrix_terms.valid_row_count
    local_samples = len(scores.sample_ids)
    if local_rows != len(logical_samples) or local_samples != len(logical_samples):
        raise RuntimeError("validation score counts differ from the selected batch")

    device = scores.score_matrix.device
    (
        global_matrix_numerator,
        global_l_gen_numerator,
        global_norm_numerator,
    ) = _global_float_sums(
        local_matrix_numerator,
        local_l_gen_numerator,
        local_norm_numerator,
        device=device,
    )
    (
        global_rows,
        global_samples,
        global_evidence_tokens,
        global_groups,
    ) = _global_integer_sums(
        local_rows,
        local_samples,
        local_evidence_tokens,
        1,
        device=device,
    )
    matrix_loss = global_matrix_numerator / global_rows
    l_gen_loss = global_l_gen_numerator / global_samples
    norm_loss = (
        None
        if global_norm_numerator is None
        else global_norm_numerator / global_samples
    )
    weighted_norm_loss = (
        None if norm_loss is None else norm_loss * objective.norm_weight  # type: ignore[union-attr]
    )
    total_loss = (
        matrix_loss * objective.matrix_ce_weight + l_gen_loss * objective.l_gen_weight
    )
    if weighted_norm_loss is not None:
        total_loss += weighted_norm_loss
    return RepresentationValidationMetrics(
        validation_event_index=validation_event_index,
        global_matrix_ce_loss=matrix_loss,
        global_l_gen_loss=l_gen_loss,
        global_norm_loss=norm_loss,
        global_weighted_norm_loss=weighted_norm_loss,
        global_total_loss=total_loss,
        global_row_count=global_rows,
        global_sample_count=global_samples,
        global_evidence_token_count=global_evidence_tokens,
        global_group_count=global_groups,
        local_rank=rank,
        data_parallel_world_size=world_size,
        local_image_group_key=group.image_group_key,
        local_sample_ids=scores.sample_ids,
    )


def _validate_group(
    group: object,
    samples: tuple[RepresentationTrainingSample, ...],
    *,
    collective_candidate_count: int,
) -> None:
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("group_builder must return a SameImageReadoutGroup")
    expected_ids = tuple(sample.sample_id for sample in samples)
    actual_ids = tuple(row.sample_id for row in group.rows)
    if actual_ids != expected_ids:
        raise ValueError("validation group builder changed sampler-owned sample order")
    expected_keys = {sample.image_group_key for sample in samples}
    if expected_keys != {group.image_group_key}:
        raise ValueError(
            "validation group builder changed the sampler-owned image group"
        )
    if group.collective_candidate_count != collective_candidate_count:
        raise ValueError(
            "validation group builder did not materialize the synchronized "
            "collective candidate count"
        )


def _assert_visuals_do_not_require_grad(group: SameImageReadoutGroup) -> None:
    bundles = (
        group.source_visual,
        *tuple(candidate.visual for candidate in group.candidates),
        *group.collective_padding,
    )
    for bundle in bundles:
        tensors = (bundle.main, *bundle.deepstack)
        if any(
            tensor.requires_grad or tensor.grad_fn is not None for tensor in tensors
        ):
            raise RuntimeError("validation visual tensors retained an autograd graph")


def _assert_adapter_gradient_state_unchanged(
    state: tuple[
        tuple[nn.Parameter, bool, torch.Tensor | None, torch.Tensor | None], ...
    ],
) -> None:
    for parameter, required_grad, original_grad, original_value in state:
        if parameter.requires_grad != required_grad:
            raise RuntimeError("validation changed an Adapter requires_grad flag")
        if parameter.grad is not original_grad:
            raise RuntimeError("validation replaced an existing Adapter gradient")
        if original_grad is not None and not torch.equal(original_grad, original_value):
            raise RuntimeError("validation mutated an existing Adapter gradient")


def _distributed_identity(expected_world_size: int) -> tuple[int, int]:
    initialized = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    world_size = torch.distributed.get_world_size() if initialized else 1
    rank = torch.distributed.get_rank() if initialized else 0
    if world_size != expected_world_size:
        raise ValueError(
            "validation data-parallel world size mismatch: "
            f"expected={expected_world_size} actual={world_size}"
        )
    return rank, world_size


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
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("validation loss numerator is non-finite")
    return (
        float(values[0].item()),
        float(values[1].item()),
        None if norm_numerator is None else float(values[2].item()),
    )


def _global_integer_sums(
    rows: int,
    samples: int,
    evidence_tokens: int,
    groups: int,
    *,
    device: torch.device,
) -> tuple[int, int, int, int]:
    values = torch.tensor(
        (rows, samples, evidence_tokens, groups),
        dtype=torch.int64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    result = tuple(int(value.item()) for value in values)
    if any(value <= 0 for value in result):
        raise RuntimeError("globally reduced validation counts must be positive")
    return result  # type: ignore[return-value]


def _validation_event_seed(
    *,
    sampler_seed: int,
    validation_event_index: int,
    rank: int,
    data_manifest_sha256: str,
) -> int:
    payload = (
        "representation-validation-event-v1\0"
        f"{sampler_seed}\0{validation_event_index}\0{rank}\0"
        f"{data_manifest_sha256}"
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def _isolated_validation_rng(seed: int, *, device: torch.device) -> Iterator[None]:
    python_state = random.getstate()
    cpu_state = torch.get_rng_state().clone()
    cuda_state: torch.Tensor | None = None
    cuda_device_index: int | None = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("validation Adapter is on CUDA but CUDA is unavailable")
        cuda_device_index = device.index
        if cuda_device_index is None:
            raise ValueError("validation CUDA device must have an explicit index")
        if torch.cuda.current_device() != cuda_device_index:
            raise RuntimeError(
                "validation Adapter device differs from the current CUDA rank device"
            )
        cuda_state = torch.cuda.get_rng_state(cuda_device_index).clone()
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if cuda_state is not None:
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            if cuda_device_index is None:  # defensive narrowing
                raise RuntimeError("validation lost its CUDA device identity")
            torch.cuda.set_rng_state(cuda_state, cuda_device_index)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("validation Adapter must expose parameters")
    return parameter.device


def _integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _positive_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _non_negative_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


__all__ = [
    "RepresentationValidationMetrics",
    "evaluate_representation_validation_event",
]
