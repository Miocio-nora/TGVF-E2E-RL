"""Length-only response-budget replay for the accepted Policy-RL T1 run.

This module deliberately reuses the frozen prompt and generation boundary from
``policy_selection_vllm``.  It changes only which already-recorded logical
requests are scheduled: a later budget revision is eligible exactly when every
earlier revision ended with the backend ``length`` finish reason.

GPU libraries remain lazy imports so planning and audit commands are CPU-only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .policy_selection import SelectionCandidate, SelectionBranch, stable_selection_request_id
from .policy_selection_runtime import (
    T1_ATTEMPTS,
    T1RawGenerationEvidence,
    T1RunConfig,
    candidate_rank,
    load_t1_run_config,
    validate_chunk_manifest,
    write_content_addressed_chunk,
)
from .policy_selection_vllm import (
    _atomic_write_immutable,
    _canonical_json_bytes,
    _generate_one,
    _validate_prepared_output_root,
    _validate_runtime_versions,
    budget_chunk_index,
    load_t1_candidates,
    prepare_candidate_prompt,
    rank_candidate_chunks,
)


@dataclass(frozen=True, slots=True)
class T1LengthRetry:
    """One exact logical request eligible for the next response budget."""

    candidate: SelectionCandidate
    previous: T1RawGenerationEvidence
    shard_rank: int
    local_chunk_index: int

    @property
    def request_id(self) -> str:
        return self.previous.request_id

    @property
    def attempt_index(self) -> int:
        return self.previous.attempt_index

    def as_record(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "candidate_sha256": self.candidate.identity_sha256,
            "sample_id": self.candidate.sample_id,
            "source": self.candidate.source.value,
            "attempt_index": self.attempt_index,
            "attempt_seed": self.previous.attempt_seed,
            "previous_budget_revision": self.previous.budget_revision,
            "previous_evidence_sha256": self.previous.evidence_sha256,
            "previous_sampled_token_count": self.previous.sampled_token_count,
            "shard_rank": self.shard_rank,
            "local_chunk_index": self.local_chunk_index,
        }


@dataclass(frozen=True, slots=True)
class T1LengthRetrySelection:
    """Pending and already-complete requests for one requested revision."""

    pending_previous: tuple[T1RawGenerationEvidence, ...]
    completed_current: tuple[T1RawGenerationEvidence, ...]


def _retry_identity(evidence: T1RawGenerationEvidence) -> tuple[object, ...]:
    return (
        evidence.run_id,
        evidence.run_manifest_sha256,
        evidence.request_id,
        evidence.sample_id,
        evidence.candidate_sha256,
        evidence.source,
        evidence.attempt_index,
        evidence.attempt_seed,
        evidence.prompt_sha256,
        evidence.rendered_prompt_token_ids_sha256,
        evidence.prompt_token_count,
        evidence.image_sha256,
        evidence.source_width,
        evidence.source_height,
        evidence.source_mode,
        evidence.source_rgb_sha256,
        evidence.processed_width,
        evidence.processed_height,
        dict(evidence.backend),
    )


def validate_length_retry_identity(
    previous: T1RawGenerationEvidence, current: T1RawGenerationEvidence
) -> None:
    """Prove that two budget revisions are the same logical request/observation."""

    if previous.finish_reason != "length":
        raise ValueError("only a length finish may have a later budget revision")
    if current.budget_revision != previous.budget_revision + 1:
        raise ValueError("retry budget revisions must be consecutive")
    if _retry_identity(current) != _retry_identity(previous):
        raise ValueError("retry logical request or observation identity differs")
    if current.max_model_len <= previous.max_model_len:
        raise ValueError("retry max_model_len did not increase")
    if current.max_new_tokens <= previous.max_new_tokens:
        raise ValueError("retry max_new_tokens did not increase")
    if previous.sampled_token_ids is None or current.sampled_token_ids is None:
        raise ValueError("retry parity requires retained sampled token IDs")


def length_retry_prefix_audit(
    previous: T1RawGenerationEvidence, current: T1RawGenerationEvidence
) -> dict[str, object]:
    """Measure, but do not assume, token-prefix parity across engine budgets."""

    validate_length_retry_identity(previous, current)
    assert previous.sampled_token_ids is not None
    assert current.sampled_token_ids is not None
    common = 0
    for previous_token, current_token in zip(
        previous.sampled_token_ids, current.sampled_token_ids
    ):
        if previous_token != current_token:
            break
        common += 1
    previous_is_exact_prefix = (
        len(current.sampled_token_ids) >= len(previous.sampled_token_ids)
        and common == len(previous.sampled_token_ids)
    )
    first_divergence_index: int | None
    if previous_is_exact_prefix:
        first_divergence_index = None
    else:
        first_divergence_index = common
    return {
        "request_id": previous.request_id,
        "previous_budget_revision": previous.budget_revision,
        "current_budget_revision": current.budget_revision,
        "previous_evidence_sha256": previous.evidence_sha256,
        "current_evidence_sha256": current.evidence_sha256,
        "previous_sampled_token_count": previous.sampled_token_count,
        "current_sampled_token_count": current.sampled_token_count,
        "common_prefix_token_count": common,
        "previous_is_exact_prefix": previous_is_exact_prefix,
        "first_divergence_index": first_divergence_index,
        "current_finish_reason": current.finish_reason,
    }


def select_length_retry_evidence(
    evidences: Sequence[T1RawGenerationEvidence],
    *,
    expected_requests: Mapping[str, tuple[str, int]],
    budget_revision: int,
) -> T1LengthRetrySelection:
    """Select only consecutive length-finish histories for one later revision."""

    if type(budget_revision) is not int or budget_revision not in {1, 2}:
        raise ValueError("length retry budget_revision must be 1 or 2")
    histories: dict[str, dict[int, T1RawGenerationEvidence]] = {
        request_id: {} for request_id in expected_requests
    }
    for evidence in evidences:
        expected = expected_requests.get(evidence.request_id)
        if expected is None:
            raise ValueError("raw evidence contains an unknown logical request")
        if (evidence.candidate_sha256, evidence.attempt_index) != expected:
            raise ValueError("raw evidence logical request identity differs")
        history = histories[evidence.request_id]
        if evidence.budget_revision in history:
            raise ValueError("duplicate logical request budget revision")
        history[evidence.budget_revision] = evidence

    pending: list[T1RawGenerationEvidence] = []
    completed: list[T1RawGenerationEvidence] = []
    for request_id in sorted(histories):
        history = histories[request_id]
        if 0 not in history:
            raise ValueError("revision-0 evidence is incomplete")
        revisions = sorted(history)
        if revisions != list(range(revisions[-1] + 1)):
            raise ValueError("logical request budget history has a gap")
        for revision in revisions[1:]:
            validate_length_retry_identity(history[revision - 1], history[revision])
        if budget_revision in history:
            completed.append(history[budget_revision])
            continue
        latest_revision = revisions[-1]
        latest = history[latest_revision]
        if latest_revision >= budget_revision:
            raise ValueError("logical request has an invalid future revision")
        if latest.finish_reason != "length":
            continue
        if latest_revision != budget_revision - 1:
            raise ValueError("length retry cannot skip a budget revision")
        pending.append(latest)
    return T1LengthRetrySelection(tuple(pending), tuple(completed))


def _load_validated_evidence(
    run: T1RunConfig,
) -> tuple[tuple[object, T1RawGenerationEvidence], ...]:
    located: list[tuple[object, T1RawGenerationEvidence]] = []
    for manifest_path in sorted((run.output_root / "manifests").glob("*.json")):
        manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = validate_chunk_manifest(
            manifest_record, output_root=run.output_root, run=run
        )
        evidence_path = run.output_root / manifest.evidence_file
        payload = evidence_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.evidence_sha256:
            raise ValueError(
                "chunk evidence changed after manifest validation: "
                f"{evidence_path}"
            )
        for line_number, line in enumerate(payload.splitlines(), start=1):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    "invalid evidence JSON after manifest validation at "
                    f"{evidence_path}:{line_number}"
                ) from error
            evidence = T1RawGenerationEvidence.from_record(record)
            evidence.validate_against_run(run)
            located.append((manifest, evidence))
    return tuple(located)


def _expected_requests(
    candidates: Sequence[SelectionCandidate],
) -> dict[str, tuple[str, int]]:
    expected: dict[str, tuple[str, int]] = {}
    for candidate in candidates:
        for attempt_index in range(T1_ATTEMPTS):
            request_id = stable_selection_request_id(
                candidate_sha256=candidate.identity_sha256,
                branch=SelectionBranch.FULL_IMAGE,
                attempt_index=attempt_index,
            )
            logical = (candidate.identity_sha256, attempt_index)
            if request_id in expected:
                raise ValueError("duplicate expected logical request ID")
            expected[request_id] = logical
    return expected


def _candidate_locations(
    candidates: Sequence[SelectionCandidate], *, run: T1RunConfig
) -> dict[str, tuple[int, int, SelectionCandidate]]:
    locations: dict[str, tuple[int, int, SelectionCandidate]] = {}
    world_size = int(run.runtime["world_size"])
    for rank in range(world_size):
        chunks = rank_candidate_chunks(
            candidates,
            rank=rank,
            world_size=world_size,
            chunk_candidates=int(run.runtime["chunk_candidates"]),
        )
        for local_chunk_index, chunk in enumerate(chunks):
            for candidate in chunk:
                locations[candidate.identity_sha256] = (
                    rank,
                    local_chunk_index,
                    candidate,
                )
    if len(locations) != len(candidates):
        raise ValueError("candidate retry locations are incomplete")
    return locations


def plan_t1_length_retries(
    config_path: str | Path,
    *,
    budget_revision: int,
    rank: int | None = None,
) -> tuple[T1RunConfig, tuple[T1LengthRetry, ...], tuple[T1RawGenerationEvidence, ...]]:
    """Validate all evidence and return the exact pending retry plan."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    run.budget(budget_revision)
    if budget_revision not in {1, 2}:
        raise ValueError("length retry budget_revision must be 1 or 2")
    world_size = int(run.runtime["world_size"])
    if rank is not None and (type(rank) is not int or not 0 <= rank < world_size):
        raise ValueError("rank must be inside the configured world size")

    candidates = load_t1_candidates(run)
    locations = _candidate_locations(candidates, run=run)
    located = _load_validated_evidence(run)
    evidences = tuple(evidence for _, evidence in located)

    for manifest, evidence in located:
        evidence_rank, local_chunk_index, _ = locations[evidence.candidate_sha256]
        expected_chunk_index = budget_chunk_index(
            budget_revision=evidence.budget_revision,
            local_chunk_index=local_chunk_index,
        )
        if manifest.shard_rank != evidence_rank:
            raise ValueError("evidence manifest rank differs from candidate shard")
        if manifest.chunk_index != expected_chunk_index:
            raise ValueError("evidence manifest is outside its budget/chunk namespace")

    selection = select_length_retry_evidence(
        evidences,
        expected_requests=_expected_requests(candidates),
        budget_revision=budget_revision,
    )
    retries: list[T1LengthRetry] = []
    for previous in selection.pending_previous:
        shard_rank, local_chunk_index, candidate = locations[
            previous.candidate_sha256
        ]
        if rank is None or shard_rank == rank:
            retries.append(
                T1LengthRetry(
                    candidate=candidate,
                    previous=previous,
                    shard_rank=shard_rank,
                    local_chunk_index=local_chunk_index,
                )
            )
    retries.sort(
        key=lambda item: (
            item.shard_rank,
            item.local_chunk_index,
            item.candidate.identity_sha256,
            item.attempt_index,
        )
    )
    completed = tuple(
        evidence
        for evidence in selection.completed_current
        if rank is None
        or locations[evidence.candidate_sha256][0] == rank
    )
    return run, tuple(retries), completed


def t1_length_retry_status(
    config_path: str | Path, *, budget_revision: int, rank: int | None = None
) -> dict[str, object]:
    run, retries, completed = plan_t1_length_retries(
        config_path, budget_revision=budget_revision, rank=rank
    )
    return {
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "budget_revision": budget_revision,
        "rank": rank,
        "pending_count": len(retries),
        "completed_count": len(completed),
        "pending": [retry.as_record() for retry in retries],
        "completed_request_ids": sorted(
            evidence.request_id for evidence in completed
        ),
    }


async def run_t1_length_retry_worker(
    config_path: str | Path,
    *,
    rank: int,
    budget_revision: int,
    expected_request_ids: Sequence[str],
) -> dict[str, object]:
    """Generate only the explicitly acknowledged pending length retries."""

    path = Path(config_path).resolve()
    run, retries, completed = plan_t1_length_retries(
        path, budget_revision=budget_revision, rank=None
    )
    expected = tuple(sorted(expected_request_ids))
    planned = tuple(sorted(retry.request_id for retry in retries))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_request_ids must be a non-empty unique list")
    completed_by_request = {
        evidence.request_id: evidence for evidence in completed
    }
    acknowledged = set(planned) | set(completed_by_request)
    if acknowledged != set(expected):
        raise ValueError(
            "acknowledged length retry set differs: "
            f"pending={planned}, completed={tuple(sorted(completed_by_request))}, "
            f"expected={expected}"
        )
    world_size = int(run.runtime["world_size"])
    if any(retry.shard_rank != rank for retry in retries) or any(
        candidate_rank(evidence.candidate_sha256, world_size=world_size) != rank
        for evidence in completed
    ):
        raise ValueError("acknowledged retry request belongs to another rank")
    if not retries:
        return {
            "run_id": run.run_id,
            "rank": rank,
            "budget_revision": budget_revision,
            "records_written": 0,
            "records_resumed": len(completed),
            "manifests": [],
        }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(rank):
        raise ValueError(
            f"worker rank {rank} requires CUDA_VISIBLE_DEVICES={rank}, got {visible!r}"
        )
    _validate_runtime_versions(run)

    grouped: dict[int, list[T1LengthRetry]] = {}
    for retry in retries:
        grouped.setdefault(retry.local_chunk_index, []).append(retry)
    for local_chunk_index in grouped:
        chunk_index = budget_chunk_index(
            budget_revision=budget_revision,
            local_chunk_index=local_chunk_index,
        )
        manifest_path = (
            run.output_root
            / "manifests"
            / f"rank-{rank:02d}-chunk-{chunk_index:06d}.json"
        )
        if manifest_path.exists():
            raise ValueError("pending retry chunk already has an immutable manifest")

    from transformers import AutoProcessor
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    processor = AutoProcessor.from_pretrained(
        str(run.model["path"]),
        trust_remote_code=True,
        min_pixels=int(run.image["min_pixels"]),
        max_pixels=int(run.image["max_pixels"]),
        use_fast=True,
    )
    if len(processor.tokenizer) != run.model["tokenizer_length"]:
        raise ValueError("worker tokenizer length differs from the run identity")
    budget = run.budget(budget_revision)
    engine_args = AsyncEngineArgs(
        model=str(run.model["path"]),
        dtype=str(run.model["dtype"]),
        trust_remote_code=True,
        quantization=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        max_model_len=budget.max_model_len,
        max_num_seqs=int(run.runtime["max_num_seqs"]),
        max_num_batched_tokens=int(run.runtime["max_num_batched_tokens"]),
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=float(run.runtime["gpu_memory_utilization"]),
        seed=int(run.runtime["engine_seed"]),
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={
            "min_pixels": int(run.image["min_pixels"]),
            "max_pixels": int(run.image["max_pixels"]),
            "do_resize": True,
        },
        mm_processor_cache_gb=float(run.runtime["mm_processor_cache_gb"]),
        mm_encoder_attn_backend=str(run.runtime["mm_encoder_attn_backend"]),
        generation_config=str(run.runtime["generation_config_mode"]),
        enforce_eager=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    written = 0
    manifests: list[dict[str, object]] = []
    audit_sidecars: list[dict[str, object]] = []
    try:
        for local_chunk_index in sorted(grouped):
            chunk_retries = sorted(
                grouped[local_chunk_index],
                key=lambda item: (item.candidate.identity_sha256, item.attempt_index),
            )
            prepared_by_candidate: dict[str, Any] = {}
            try:
                for retry in chunk_retries:
                    candidate_sha = retry.candidate.identity_sha256
                    if candidate_sha not in prepared_by_candidate:
                        prepared_by_candidate[candidate_sha] = prepare_candidate_prompt(
                            retry.candidate, run=run, processor=processor
                        )
                tasks = [
                    asyncio.create_task(
                        _generate_one(
                            engine=engine,
                            sampling_params_type=SamplingParams,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            run=run,
                            prepared=prepared_by_candidate[
                                retry.candidate.identity_sha256
                            ],
                            attempt_index=retry.attempt_index,
                            budget_revision=budget_revision,
                        )
                    )
                    for retry in chunk_retries
                ]
                records = await asyncio.gather(*tasks)
                prefix_audits = [
                    length_retry_prefix_audit(retry.previous, record)
                    for retry, record in zip(chunk_retries, records, strict=True)
                ]
                chunk_index = budget_chunk_index(
                    budget_revision=budget_revision,
                    local_chunk_index=local_chunk_index,
                )
                manifest = write_content_addressed_chunk(
                    run.output_root,
                    records,
                    run=run,
                    shard_rank=rank,
                    chunk_index=chunk_index,
                )
                written += manifest.record_count
                manifest_record = {
                    "rank": rank,
                    "chunk_index": chunk_index,
                    "record_count": manifest.record_count,
                    "manifest_sha256": manifest.manifest_sha256,
                }
                manifests.append(manifest_record)
                audit_record = {
                    "schema_version": "tgvf.policy-selection.t1-length-retry-audit.v1",
                    "run_id": run.run_id,
                    "run_manifest_sha256": run.manifest_sha256,
                    "rank": rank,
                    "chunk_index": chunk_index,
                    "manifest_sha256": manifest.manifest_sha256,
                    "budget_revision": budget_revision,
                    "prefix_audits": prefix_audits,
                }
                audit_payload = _canonical_json_bytes(audit_record) + b"\n"
                audit_sha256 = hashlib.sha256(audit_payload).hexdigest()
                audit_relative = (
                    Path("runtime")
                    / "length-retry-audits"
                    / f"{audit_sha256}.json"
                )
                _atomic_write_immutable(
                    run.output_root / audit_relative, audit_payload
                )
                audit_sidecar = {
                    "path": audit_relative.as_posix(),
                    "sha256": audit_sha256,
                }
                audit_sidecars.append(audit_sidecar)
                print(
                    json.dumps(
                        {
                            "event": "length_retry_chunk_committed",
                            **manifest_record,
                            "audit": audit_sidecar,
                            "prefix_audits": prefix_audits,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                for prepared in prepared_by_candidate.values():
                    prepared.image.rgb.close()
    finally:
        engine.shutdown()
    return {
        "run_id": run.run_id,
        "rank": rank,
        "budget_revision": budget_revision,
        "records_written": written,
        "records_resumed": len(completed),
        "manifests": manifests,
        "audit_sidecars": audit_sidecars,
    }


__all__ = [
    "T1LengthRetry",
    "T1LengthRetrySelection",
    "plan_t1_length_retries",
    "run_t1_length_retry_worker",
    "select_length_retry_evidence",
    "t1_length_retry_status",
    "length_retry_prefix_audit",
    "validate_length_retry_identity",
]
