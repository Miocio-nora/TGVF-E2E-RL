"""Distributed train-step timing and CUDA-memory evidence.

Measurements cover only one synchronous ``trainer.train_step`` call.  Model
loading, validation, checkpoint I/O, and artifact export are intentionally
outside this interval and must be reported separately by an experiment log.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from time import perf_counter_ns
from typing import TypeVar, cast

import torch


REPRESENTATION_TRAIN_STEP_PERFORMANCE_SCHEMA_VERSION = (
    "representation-train-step-performance-v1"
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RepresentationRankTrainStepResources:
    rank: int
    elapsed_ns: int
    starting_allocated_bytes: int
    starting_reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    ending_allocated_bytes: int
    ending_reserved_bytes: int

    def __post_init__(self) -> None:
        _non_negative_int(self.rank, field_name="rank")
        _positive_int(self.elapsed_ns, field_name="elapsed_ns")
        for field_name in (
            "starting_allocated_bytes",
            "starting_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "ending_allocated_bytes",
            "ending_reserved_bytes",
        ):
            _non_negative_int(getattr(self, field_name), field_name=field_name)
        if self.peak_allocated_bytes < max(
            self.starting_allocated_bytes,
            self.ending_allocated_bytes,
        ):
            raise ValueError("peak allocated bytes cannot be below resident values")
        if self.peak_reserved_bytes < max(
            self.starting_reserved_bytes,
            self.ending_reserved_bytes,
        ):
            raise ValueError("peak reserved bytes cannot be below resident values")


@dataclass(frozen=True, slots=True)
class RepresentationTrainStepPerformance:
    global_step: int
    global_row_count: int
    global_matrix_count: int
    ranks: tuple[RepresentationRankTrainStepResources, ...]
    schema_version: str = REPRESENTATION_TRAIN_STEP_PERFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _positive_int(self.global_step, field_name="global_step")
        _positive_int(self.global_row_count, field_name="global_row_count")
        _positive_int(self.global_matrix_count, field_name="global_matrix_count")
        if not isinstance(self.ranks, tuple) or not self.ranks:
            raise ValueError("train-step performance requires rank records")
        if any(
            not isinstance(record, RepresentationRankTrainStepResources)
            for record in self.ranks
        ):
            raise TypeError("train-step rank resources must be typed")
        if tuple(record.rank for record in self.ranks) != tuple(range(len(self.ranks))):
            raise ValueError("train-step rank resources must be sorted and complete")
        if self.schema_version != REPRESENTATION_TRAIN_STEP_PERFORMANCE_SCHEMA_VERSION:
            raise ValueError("representation train-step performance schema mismatch")

    @property
    def max_rank_elapsed_seconds(self) -> float:
        return max(record.elapsed_ns for record in self.ranks) / 1_000_000_000

    @property
    def global_rows_per_second(self) -> float:
        return self.global_row_count / self.max_rank_elapsed_seconds

    @property
    def global_matrices_per_second(self) -> float:
        return self.global_matrix_count / self.max_rank_elapsed_seconds


def measure_distributed_train_step(
    step: Callable[[], _T],
    *,
    device: torch.device,
    global_matrix_count: int,
    process_group: object = None,
) -> tuple[_T, RepresentationTrainStepPerformance]:
    """Synchronously measure and gather one real CUDA train step."""

    if not callable(step):
        raise TypeError("step must be callable")
    _positive_int(global_matrix_count, field_name="global_matrix_count")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("train-step resource measurement requires CUDA")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "distributed train-step measurement requires a process group"
        )
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)
    torch.distributed.barrier(group=process_group)
    torch.cuda.synchronize(device)
    starting_allocated = torch.cuda.memory_allocated(device)
    starting_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter_ns()
    result = step()
    torch.cuda.synchronize(device)
    elapsed = perf_counter_ns() - started
    local = RepresentationRankTrainStepResources(
        rank=rank,
        elapsed_ns=elapsed,
        starting_allocated_bytes=starting_allocated,
        starting_reserved_bytes=starting_reserved,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        ending_allocated_bytes=torch.cuda.memory_allocated(device),
        ending_reserved_bytes=torch.cuda.memory_reserved(device),
    )
    gathered: list[object] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local, group=process_group)
    if any(
        not isinstance(record, RepresentationRankTrainStepResources)
        for record in gathered
    ):
        raise TypeError("distributed train-step resource gather is malformed")
    global_step = getattr(result, "global_step", None)
    global_row_count = getattr(result, "global_row_count", None)
    _positive_int(global_step, field_name="measured result global_step")
    _positive_int(
        global_row_count,
        field_name="measured result global_row_count",
    )
    summary = RepresentationTrainStepPerformance(
        global_step=global_step,
        global_row_count=global_row_count,
        global_matrix_count=global_matrix_count,
        ranks=tuple(
            cast(RepresentationRankTrainStepResources, item) for item in gathered
        ),
    )
    for value in (
        summary.max_rank_elapsed_seconds,
        summary.global_rows_per_second,
        summary.global_matrices_per_second,
    ):
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("train-step throughput reduction is non-finite")
    return result, summary


def rank_resources_from_mapping(
    value: Mapping[str, object],
) -> RepresentationRankTrainStepResources:
    """Strict JSON-safe reconstruction used by report/comparison tools."""

    expected = {
        "rank",
        "elapsed_ns",
        "starting_allocated_bytes",
        "starting_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "ending_allocated_bytes",
        "ending_reserved_bytes",
    }
    if set(value) != expected:
        raise ValueError("rank resource mapping fields differ from the schema")
    return RepresentationRankTrainStepResources(
        **{name: value[name] for name in expected}  # type: ignore[arg-type]
    )


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "REPRESENTATION_TRAIN_STEP_PERFORMANCE_SCHEMA_VERSION",
    "RepresentationRankTrainStepResources",
    "RepresentationTrainStepPerformance",
    "measure_distributed_train_step",
    "rank_resources_from_mapping",
]
