"""Strict continuous-versus-teardown/resume comparison for representation runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError

from .distributed_checkpoint import (
    DistributedRepresentationRankState,
    load_distributed_representation_checkpoint_metadata,
    load_rank_zero_adapter_owned_state_export,
)


REPRESENTATION_RESUME_PARITY_SCHEMA_VERSION = "representation-resume-parity-v1"


@dataclass(frozen=True, slots=True)
class RepresentationResumeParityReport:
    run_identity_sha256: str
    global_step: int
    adapter_tensor_count: int
    world_size: int
    train_event_count: int
    validation_event_count: int
    model_local_shard_sha256: tuple[str, ...]
    optimizer_local_shard_sha256: tuple[str, ...]
    rank_state_sha256: tuple[str, ...]
    exact: bool = True
    schema_version: str = REPRESENTATION_RESUME_PARITY_SCHEMA_VERSION


def compare_representation_resume_lanes(
    *,
    continuous_artifact_path: str | Path,
    resumed_artifact_path: str | Path,
    continuous_checkpoint_path: str | Path,
    resumed_checkpoint_path: str | Path,
    continuous_metrics_path: str | Path,
    resumed_metrics_path: str | Path,
) -> RepresentationResumeParityReport:
    """Require exact scientific state after continuous and resumed execution.

    Wall time, CUDA memory, invocation config hashes, and metrics-history byte
    identities are lane-specific operational evidence and are intentionally
    excluded.  Adapter/optimizer/scheduler/sampler/RNG/shards and every train or
    validation scientific record must be identical.
    """

    continuous_export = load_rank_zero_adapter_owned_state_export(
        continuous_artifact_path
    )
    resumed_export = load_rank_zero_adapter_owned_state_export(resumed_artifact_path)
    if continuous_export.manifest != resumed_export.manifest:
        raise ReplayMismatchError(
            "continuous and resumed Adapter export manifests differ"
        )
    if continuous_export.state is None or resumed_export.state is None:
        raise ReplayMismatchError("resume parity requires materialized export tensors")
    if tuple(continuous_export.state) != tuple(resumed_export.state):
        raise ReplayMismatchError("continuous and resumed Adapter tensor names differ")
    for name in continuous_export.state:
        if not torch.equal(
            continuous_export.state[name],
            resumed_export.state[name],
        ):
            raise ReplayMismatchError(
                f"continuous and resumed Adapter tensor differs: {name}"
            )

    continuous_metadata = load_distributed_representation_checkpoint_metadata(
        continuous_checkpoint_path
    )
    resumed_metadata = load_distributed_representation_checkpoint_metadata(
        resumed_checkpoint_path
    )
    continuous_manifest = continuous_metadata.manifest
    resumed_manifest = resumed_metadata.manifest
    exact_manifest_fields = (
        "run_identity",
        "run_identity_sha256",
        "global_step",
        "world_size",
        "fsdp_reshard_after_forward",
        "owned_state_names",
        "optimizer_type",
        "optimizer_identity_sha256",
        "accumulation_identity_sha256",
        "trainer_execution_identity_sha256",
        "sampler_contract_identity_sha256",
        "scheduler_identity_sha256",
        "rank_state_sha256",
        "model_local_shard_sha256",
        "optimizer_local_shard_sha256",
        "torch_version",
    )
    for field_name in exact_manifest_fields:
        if getattr(continuous_manifest, field_name) != getattr(
            resumed_manifest, field_name
        ):
            raise ReplayMismatchError(
                f"continuous and resumed DCP manifest differ: {field_name}"
            )
    _assert_exact_rank_state_identities(
        continuous_metadata.rank_states,
        resumed_metadata.rank_states,
    )

    continuous_events = _scientific_metric_events(continuous_metrics_path)
    resumed_events = _scientific_metric_events(resumed_metrics_path)
    if continuous_events != resumed_events:
        raise ReplayMismatchError(
            "continuous and resumed train/validation metric records differ"
        )
    train_count = sum(record["event"] == "train" for record in continuous_events)
    validation_count = sum(
        record["event"] == "validation" for record in continuous_events
    )
    return RepresentationResumeParityReport(
        run_identity_sha256=continuous_manifest.run_identity_sha256,
        global_step=continuous_manifest.global_step,
        adapter_tensor_count=len(continuous_export.state),
        world_size=continuous_manifest.world_size,
        train_event_count=train_count,
        validation_event_count=validation_count,
        model_local_shard_sha256=continuous_manifest.model_local_shard_sha256,
        optimizer_local_shard_sha256=(continuous_manifest.optimizer_local_shard_sha256),
        rank_state_sha256=continuous_manifest.rank_state_sha256,
    )


def _scientific_metric_events(
    path: str | Path,
) -> tuple[dict[str, object], ...]:
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("metrics JSONL must be non-empty and newline terminated")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("metrics JSONL must be valid UTF-8") from error
    result: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"metrics JSONL line {line_number} is not strict JSON"
            ) from error
        if not isinstance(value, dict):
            raise TypeError(f"metrics JSONL line {line_number} is not an object")
        if value.get("event") not in {"train", "validation"}:
            continue
        record = dict(value)
        # Timing and allocator peaks are intentionally expected to differ after
        # process reconstruction.  Every scientific value remains compared.
        record.pop("performance", None)
        result.append(record)
    if not result or not any(record["event"] == "train" for record in result):
        raise ValueError("metrics JSONL contains no scientific train records")
    return tuple(result)


def _assert_exact_rank_state_identities(
    continuous: tuple[DistributedRepresentationRankState, ...],
    resumed: tuple[DistributedRepresentationRankState, ...],
) -> None:
    """Compare validated rank state without invoking Tensor truth conversion.

    Each metadata loader has already recomputed and checked the sampler,
    scheduler, and RNG content digests.  Comparing those digests therefore
    compares the full tensor-bearing state while avoiding Python container
    equality, which is undefined for multi-element tensors.
    """

    if len(continuous) != len(resumed):
        raise ReplayMismatchError("continuous and resumed rank-state counts differ")
    identity_fields = (
        "rank",
        "sampler_identity_sha256",
        "sampler_state_sha256",
        "rng_state_sha256",
        "scheduler_type",
        "scheduler_state_sha256",
        "schema_version",
    )
    for continuous_rank, resumed_rank in zip(continuous, resumed, strict=True):
        for field_name in identity_fields:
            if getattr(continuous_rank, field_name) != getattr(
                resumed_rank, field_name
            ):
                raise ReplayMismatchError(
                    "continuous and resumed scheduler/sampler/RNG rank state "
                    f"differs: rank={continuous_rank.rank} field={field_name}"
                )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


__all__ = [
    "REPRESENTATION_RESUME_PARITY_SCHEMA_VERSION",
    "RepresentationResumeParityReport",
    "compare_representation_resume_lanes",
]
