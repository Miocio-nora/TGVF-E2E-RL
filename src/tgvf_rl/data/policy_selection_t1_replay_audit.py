"""Fail-closed deterministic replay audit for immutable T1 evidence.

Planning and report validation are CPU-only.  Transformers, Torch, and vLLM
are imported only after the complete run/evidence audit and explicit GPU-rank
checks have passed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from tgvf_rl.artifact_contracts import canonical_json_sha256

from .policy_selection import (
    POLICY_SELECTION_PRIMARY_SOURCES,
    SelectionCandidate,
    SelectionSource,
)
from .policy_selection_runtime import (
    T1_ATTEMPTS,
    T1RawGenerationEvidence,
    T1RunConfig,
    candidate_rank,
    load_t1_run_config,
    native_prompt_identity_sha256,
    rendered_prompt_token_ids_sha256,
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
from .policy_selection_vllm_retry import (
    _load_validated_evidence,
    validate_length_retry_identity,
)


T1_REPLAY_AUDIT_SCHEMA = "tgvf.policy-selection.t1-deterministic-replay-audit.v1"
T1_REPLAY_AUDIT_PLAN_SCHEMA = (
    "tgvf.policy-selection.t1-deterministic-replay-audit-plan.v1"
)
T1_REPLAY_AUDIT_SELECTOR_SCHEMA = (
    "tgvf.policy-selection.t1-deterministic-replay-audit-selector.v1"
)
T1_REPLAY_AUDIT_SAMPLING_SCHEMA = (
    "tgvf.policy-selection.t1-deterministic-replay-sampling.v1"
)
T1_REPLAY_AUDIT_CANDIDATES_PER_SOURCE = 1
_AUDIT_DIRECTORY = Path("runtime") / "deterministic-replay-audits"
_IDENTITY_FIELDS = (
    "run_id",
    "run_manifest_sha256",
    "request_id",
    "sample_id",
    "candidate_sha256",
    "source",
    "attempt_index",
    "attempt_seed",
    "budget_revision",
    "max_model_len",
    "max_new_tokens",
    "prompt_sha256",
    "rendered_prompt_token_ids_sha256",
    "prompt_token_count",
    "image_sha256",
    "source_width",
    "source_height",
    "source_mode",
    "source_rgb_sha256",
    "processed_width",
    "processed_height",
    "backend",
)


_sha256_json = canonical_json_sha256


@dataclass(frozen=True, slots=True)
class LocatedT1Evidence:
    """One validated evidence record plus its immutable chunk provenance."""

    evidence: T1RawGenerationEvidence
    chunk_manifest_sha256: str
    chunk_evidence_sha256: str
    shard_rank: int
    chunk_index: int


@dataclass(frozen=True, slots=True)
class T1ReplayAuditSelection:
    """A fixed content-hash candidate and all of its recorded attempts."""

    candidate: SelectionCandidate
    selector_sha256: str
    local_chunk_index: int
    histories: tuple[tuple[LocatedT1Evidence, ...], ...]

    @property
    def effective(self) -> tuple[LocatedT1Evidence, ...]:
        return tuple(history[-1] for history in self.histories)


@dataclass(frozen=True, slots=True)
class T1ReplayAuditPlan:
    """Fully validated immutable replay plan for one original GPU shard."""

    config_path: Path
    run: T1RunConfig
    rank: int
    selections: tuple[T1ReplayAuditSelection, ...]
    raw_evidence_count: int
    logical_attempt_count: int

    @property
    def audited_attempt_count(self) -> int:
        return len(self.selections) * T1_ATTEMPTS

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": T1_REPLAY_AUDIT_PLAN_SCHEMA,
            "run_id": self.run.run_id,
            "run_manifest_sha256": self.run.manifest_sha256,
            "rank": self.rank,
            "physical_gpu": self.rank,
            "selector_schema": T1_REPLAY_AUDIT_SELECTOR_SCHEMA,
            "candidates_per_source": T1_REPLAY_AUDIT_CANDIDATES_PER_SOURCE,
            "raw_evidence_count": self.raw_evidence_count,
            "logical_attempt_count": self.logical_attempt_count,
            "audited_candidate_count": len(self.selections),
            "audited_attempt_count": self.audited_attempt_count,
            "selections": [
                {
                    "source": selection.candidate.source.value,
                    "sample_id": selection.candidate.sample_id,
                    "candidate_sha256": selection.candidate.identity_sha256,
                    "selector_sha256": selection.selector_sha256,
                    "local_chunk_index": selection.local_chunk_index,
                    "attempts": [
                        {
                            "attempt_index": history[-1].evidence.attempt_index,
                            "effective_budget_revision": (
                                history[-1].evidence.budget_revision
                            ),
                            "recorded_revisions": [
                                located.evidence.budget_revision for located in history
                            ],
                            "evidence_sha256": history[-1].evidence.evidence_sha256,
                            "sampled_token_ids_sha256": (
                                history[-1].evidence.sampled_token_ids_sha256
                            ),
                            "finish_reason": history[-1].evidence.finish_reason,
                        }
                        for history in selection.histories
                    ],
                }
                for selection in self.selections
            ],
        }


class T1ReplayAuditFailure(RuntimeError):
    """Raised after a content-addressed audit report records a failed replay."""

    def __init__(self, result: Mapping[str, object]) -> None:
        super().__init__("T1 deterministic replay audit failed")
        self.result = dict(result)


def _selector_sha256(
    *, run_manifest_sha256: str, rank: int, candidate: SelectionCandidate
) -> str:
    return _sha256_json(
        {
            "schema": T1_REPLAY_AUDIT_SELECTOR_SCHEMA,
            "run_manifest_sha256": run_manifest_sha256,
            "rank": rank,
            "source": candidate.source.value,
            "candidate_sha256": candidate.identity_sha256,
        }
    )


def select_replay_audit_candidates(
    candidates: Sequence[SelectionCandidate],
    *,
    run_manifest_sha256: str,
    rank: int,
    world_size: int,
) -> tuple[tuple[SelectionCandidate, str], ...]:
    """Choose one outcome-independent content-hash candidate per source."""

    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be inside world_size")
    selected: list[tuple[SelectionCandidate, str]] = []
    candidate_sources = {candidate.source for candidate in candidates}
    expected_sources = (
        (SelectionSource.TEACHER,)
        if candidate_sources == {SelectionSource.TEACHER}
        else POLICY_SELECTION_PRIMARY_SOURCES
    )
    for source in expected_sources:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.source is source
            and candidate_rank(candidate.identity_sha256, world_size=world_size) == rank
        ]
        if not eligible:
            raise ValueError(
                f"rank {rank} has no replay-audit candidate for source {source.value}"
            )
        ranked = sorted(
            (
                _selector_sha256(
                    run_manifest_sha256=run_manifest_sha256,
                    rank=rank,
                    candidate=candidate,
                ),
                candidate.identity_sha256,
                candidate,
            )
            for candidate in eligible
        )
        selector_sha, _, candidate = ranked[0]
        selected.append((candidate, selector_sha))
    return tuple(selected)


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
                if candidate.identity_sha256 in locations:
                    raise ValueError("candidate has more than one shard/chunk location")
                locations[candidate.identity_sha256] = (
                    rank,
                    local_chunk_index,
                    candidate,
                )
    if len(locations) != len(candidates):
        raise ValueError("candidate shard/chunk locations are incomplete")
    return locations


def _validate_evidence_candidate(
    evidence: T1RawGenerationEvidence, candidate: SelectionCandidate
) -> None:
    if evidence.candidate_sha256 != candidate.identity_sha256:
        raise ValueError("raw evidence candidate identity differs")
    if evidence.sample_id != candidate.sample_id:
        raise ValueError("raw evidence sample identity differs from candidate")
    if evidence.source is not candidate.source:
        raise ValueError("raw evidence source differs from candidate")
    if evidence.image_sha256 != candidate.image["sha256"]:
        raise ValueError("raw evidence image identity differs from candidate")


def _build_histories(
    *,
    run: T1RunConfig,
    candidates: Sequence[SelectionCandidate],
    located_records: Sequence[tuple[object, T1RawGenerationEvidence]],
) -> tuple[
    dict[tuple[str, int], tuple[LocatedT1Evidence, ...]],
    dict[str, tuple[int, int, SelectionCandidate]],
]:
    locations = _candidate_locations(candidates, run=run)
    histories: dict[tuple[str, int], dict[int, LocatedT1Evidence]] = {
        (candidate.identity_sha256, attempt_index): {}
        for candidate in candidates
        for attempt_index in range(T1_ATTEMPTS)
    }
    for manifest_value, evidence in located_records:
        manifest = manifest_value
        candidate_location = locations.get(evidence.candidate_sha256)
        if candidate_location is None:
            raise ValueError("raw evidence contains an unknown candidate")
        expected_rank, local_chunk_index, candidate = candidate_location
        _validate_evidence_candidate(evidence, candidate)
        expected_prompt_sha = native_prompt_identity_sha256(
            question=candidate.question,
            image_sha256=str(candidate.image["sha256"]),
            chat_template_sha256=str(run.model["chat_template_sha256"]),
        )
        if evidence.prompt_sha256 != expected_prompt_sha:
            raise ValueError("raw evidence prompt identity differs from candidate")
        expected_chunk_index = budget_chunk_index(
            budget_revision=evidence.budget_revision,
            local_chunk_index=local_chunk_index,
        )
        if manifest.shard_rank != expected_rank:
            raise ValueError("evidence manifest rank differs from candidate shard")
        if manifest.chunk_index != expected_chunk_index:
            raise ValueError("evidence manifest is outside its budget/chunk namespace")
        key = (evidence.candidate_sha256, evidence.attempt_index)
        history = histories[key]
        if evidence.budget_revision in history:
            raise ValueError("duplicate logical request budget revision")
        history[evidence.budget_revision] = LocatedT1Evidence(
            evidence=evidence,
            chunk_manifest_sha256=manifest.manifest_sha256,
            chunk_evidence_sha256=manifest.evidence_sha256,
            shard_rank=manifest.shard_rank,
            chunk_index=manifest.chunk_index,
        )

    normalized: dict[tuple[str, int], tuple[LocatedT1Evidence, ...]] = {}
    last_budget_revision = max(budget.revision for budget in run.response_budgets)
    for key, history_by_revision in histories.items():
        if not history_by_revision:
            raise ValueError("revision-0 evidence is incomplete")
        revisions = sorted(history_by_revision)
        if revisions[0] != 0 or revisions != list(range(revisions[-1] + 1)):
            raise ValueError("logical request budget history is incomplete")
        history = tuple(history_by_revision[revision] for revision in revisions)
        for previous, current in zip(history, history[1:]):
            validate_length_retry_identity(previous.evidence, current.evidence)
        latest = history[-1].evidence
        if latest.finish_reason == "length" and latest.budget_revision < (
            last_budget_revision
        ):
            raise ValueError("logical request still requires a response-budget retry")
        normalized[key] = history
    return normalized, locations


def plan_t1_replay_audit(config_path: str | Path, *, rank: int) -> T1ReplayAuditPlan:
    """Validate the complete immutable canary and build one fixed replay plan."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    world_size = int(run.runtime["world_size"])
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be inside the configured world size")
    if run.runtime["retain_token_ids"] is not True:
        raise ValueError("deterministic replay audit requires retained token IDs")

    candidates = load_t1_candidates(run)
    located_records = _load_validated_evidence(run)
    histories, locations = _build_histories(
        run=run,
        candidates=candidates,
        located_records=located_records,
    )
    selected = select_replay_audit_candidates(
        candidates,
        run_manifest_sha256=run.manifest_sha256,
        rank=rank,
        world_size=world_size,
    )
    selections: list[T1ReplayAuditSelection] = []
    for candidate, selector_sha in selected:
        shard_rank, local_chunk_index, _ = locations[candidate.identity_sha256]
        if shard_rank != rank:
            raise ValueError("selected candidate belongs to another rank")
        candidate_histories = tuple(
            histories[(candidate.identity_sha256, attempt_index)]
            for attempt_index in range(T1_ATTEMPTS)
        )
        for history in candidate_histories:
            for located in history:
                evidence = located.evidence
                if evidence.sampled_token_ids is None:
                    raise ValueError("audited evidence omitted sampled token IDs")
                if evidence.finish_reason == "error":
                    raise ValueError(
                        "generation-error evidence cannot pass exact replay"
                    )
        selections.append(
            T1ReplayAuditSelection(
                candidate=candidate,
                selector_sha256=selector_sha,
                local_chunk_index=local_chunk_index,
                histories=candidate_histories,
            )
        )
    selections.sort(key=lambda item: item.candidate.source.value)
    return T1ReplayAuditPlan(
        config_path=path,
        run=run,
        rank=rank,
        selections=tuple(selections),
        raw_evidence_count=len(located_records),
        logical_attempt_count=len(histories),
    )


def _evidence_identity(evidence: T1RawGenerationEvidence) -> dict[str, object]:
    return {
        field: (
            evidence.source.value
            if field == "source"
            else dict(evidence.backend)
            if field == "backend"
            else getattr(evidence, field)
        )
        for field in _IDENTITY_FIELDS
    }


def _sampling_identity(
    *, run: T1RunConfig, evidence: T1RawGenerationEvidence
) -> dict[str, object]:
    return {
        "schema": T1_REPLAY_AUDIT_SAMPLING_SCHEMA,
        "sampling": dict(run.sampling),
        "budget": {
            "revision": evidence.budget_revision,
            "max_model_len": evidence.max_model_len,
            "max_new_tokens": evidence.max_new_tokens,
        },
        "attempt_seed": evidence.attempt_seed,
    }


def compare_replayed_evidence(
    *,
    run: T1RunConfig,
    expected: LocatedT1Evidence,
    actual: T1RawGenerationEvidence,
) -> dict[str, object]:
    """Compare every request/observation identity and exact sampled result."""

    expected_identity = _evidence_identity(expected.evidence)
    actual_identity = _evidence_identity(actual)
    mismatched_identity_fields = [
        field
        for field in _IDENTITY_FIELDS
        if expected_identity[field] != actual_identity[field]
    ]
    expected_sampling = _sampling_identity(run=run, evidence=expected.evidence)
    actual_sampling = _sampling_identity(run=run, evidence=actual)
    expected_token_ids = expected.evidence.sampled_token_ids
    actual_token_ids = actual.sampled_token_ids
    token_ids_exact = (
        expected_token_ids is not None
        and actual_token_ids is not None
        and expected_token_ids == actual_token_ids
    )
    checks = {
        "request_observation_identity_exact": not mismatched_identity_fields,
        "sampling_identity_exact": expected_sampling == actual_sampling,
        "sampled_token_ids_exact": token_ids_exact,
        "sampled_token_ids_sha256_exact": (
            expected.evidence.sampled_token_ids_sha256
            == actual.sampled_token_ids_sha256
        ),
        "sampled_token_count_exact": (
            expected.evidence.sampled_token_count == actual.sampled_token_count
        ),
        "finish_reason_exact": (
            expected.evidence.finish_reason == actual.finish_reason
        ),
        "stop_reason_exact": expected.evidence.stop_reason == actual.stop_reason,
        "raw_text_exact": expected.evidence.raw_text == actual.raw_text,
    }
    return {
        "source": expected.evidence.source.value,
        "sample_id": expected.evidence.sample_id,
        "candidate_sha256": expected.evidence.candidate_sha256,
        "request_id": expected.evidence.request_id,
        "attempt_index": expected.evidence.attempt_index,
        "attempt_seed": expected.evidence.attempt_seed,
        "budget_revision": expected.evidence.budget_revision,
        "prompt_sha256": expected.evidence.prompt_sha256,
        "rendered_prompt_token_ids_sha256": (
            expected.evidence.rendered_prompt_token_ids_sha256
        ),
        "image_sha256": expected.evidence.image_sha256,
        "source_rgb_sha256": expected.evidence.source_rgb_sha256,
        "recorded_evidence_sha256": expected.evidence.evidence_sha256,
        "recorded_chunk_manifest_sha256": expected.chunk_manifest_sha256,
        "recorded_chunk_evidence_sha256": expected.chunk_evidence_sha256,
        "recorded_sampled_token_ids_sha256": (
            expected.evidence.sampled_token_ids_sha256
        ),
        "replayed_evidence_sha256": actual.evidence_sha256,
        "replayed_sampled_token_ids_sha256": actual.sampled_token_ids_sha256,
        "recorded_sampled_token_count": expected.evidence.sampled_token_count,
        "replayed_sampled_token_count": actual.sampled_token_count,
        "recorded_finish_reason": expected.evidence.finish_reason,
        "replayed_finish_reason": actual.finish_reason,
        "expected_identity_sha256": _sha256_json(expected_identity),
        "replayed_identity_sha256": _sha256_json(actual_identity),
        "sampling_identity_sha256": _sha256_json(expected_sampling),
        "replayed_sampling_identity_sha256": _sha256_json(actual_sampling),
        "mismatched_identity_fields": mismatched_identity_fields,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_replay_environment(plan: T1ReplayAuditPlan) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(plan.rank):
        raise ValueError(
            f"replay rank {plan.rank} requires CUDA_VISIBLE_DEVICES={plan.rank}, "
            f"got {visible!r}"
        )
    required = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": str(plan.run.runtime["engine_seed"]),
    }
    mismatches = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"replay runtime environment differs: {mismatches}")
    if os.environ.get("VLLM_ATTENTION_BACKEND") is not None:
        raise ValueError("VLLM_ATTENTION_BACKEND must be unset for this T1 replay")


def _engine_args(run: T1RunConfig, *, budget_revision: int, args_type: Any) -> Any:
    budget = run.budget(budget_revision)
    return args_type(
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


async def _replay_plan(plan: T1ReplayAuditPlan) -> list[dict[str, object]]:
    from transformers import AutoProcessor
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    run = plan.run
    processor = AutoProcessor.from_pretrained(
        str(run.model["path"]),
        trust_remote_code=True,
        min_pixels=int(run.image["min_pixels"]),
        max_pixels=int(run.image["max_pixels"]),
        use_fast=True,
    )
    if len(processor.tokenizer) != run.model["tokenizer_length"]:
        raise ValueError("replay tokenizer length differs from the run identity")

    prepared: dict[str, Any] = {}
    try:
        for selection in plan.selections:
            item = prepare_candidate_prompt(
                selection.candidate, run=run, processor=processor
            )
            prepared[selection.candidate.identity_sha256] = item
            for expected in selection.effective:
                preflight = {
                    "rendered_prompt_token_ids_sha256": (
                        expected.evidence.rendered_prompt_token_ids_sha256
                    ),
                    "prompt_token_count": expected.evidence.prompt_token_count,
                    "source_width": expected.evidence.source_width,
                    "source_height": expected.evidence.source_height,
                    "source_mode": expected.evidence.source_mode,
                    "source_rgb_sha256": expected.evidence.source_rgb_sha256,
                    "processed_width": expected.evidence.processed_width,
                    "processed_height": expected.evidence.processed_height,
                }
                replayed_preflight = {
                    "rendered_prompt_token_ids_sha256": (
                        rendered_prompt_token_ids_sha256(item.prompt_token_ids)
                    ),
                    "prompt_token_count": len(item.prompt_token_ids),
                    "source_width": item.image.source_width,
                    "source_height": item.image.source_height,
                    "source_mode": item.image.source_mode,
                    "source_rgb_sha256": item.image.source_rgb_sha256,
                    "processed_width": item.image.processed_width,
                    "processed_height": item.image.processed_height,
                }
                if preflight != replayed_preflight:
                    raise ValueError(
                        "replay prompt/image preflight differs from recorded evidence"
                    )

        comparisons: list[dict[str, object]] = []
        revisions = sorted(
            {
                located.evidence.budget_revision
                for selection in plan.selections
                for located in selection.effective
            }
        )
        for revision in revisions:
            work = [
                (selection, located)
                for selection in plan.selections
                for located in selection.effective
                if located.evidence.budget_revision == revision
            ]
            engine = AsyncLLM.from_engine_args(
                _engine_args(run, budget_revision=revision, args_type=AsyncEngineArgs)
            )
            try:
                tasks = [
                    asyncio.create_task(
                        _generate_one(
                            engine=engine,
                            sampling_params_type=SamplingParams,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            run=run,
                            prepared=prepared[selection.candidate.identity_sha256],
                            attempt_index=located.evidence.attempt_index,
                            budget_revision=revision,
                        )
                    )
                    for selection, located in work
                ]
                actual = await asyncio.gather(*tasks)
            finally:
                engine.shutdown()
            comparisons.extend(
                compare_replayed_evidence(
                    run=run,
                    expected=located,
                    actual=replayed,
                )
                for (_, located), replayed in zip(work, actual, strict=True)
            )
        comparisons.sort(
            key=lambda record: (
                str(record["source"]),
                str(record["candidate_sha256"]),
                int(record["attempt_index"]),
            )
        )
        return comparisons
    finally:
        for item in prepared.values():
            item.image.rgb.close()


def _write_audit_report(
    plan: T1ReplayAuditPlan, comparisons: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    passed = len(comparisons) == plan.audited_attempt_count and all(
        comparison.get("passed") is True for comparison in comparisons
    )
    report: dict[str, object] = {
        "schema_version": T1_REPLAY_AUDIT_SCHEMA,
        "run_id": plan.run.run_id,
        "run_manifest_sha256": plan.run.manifest_sha256,
        "rank": plan.rank,
        "physical_gpu": plan.rank,
        "model_identity_sha256": plan.run.model_identity_sha256,
        "processor_identity_sha256": plan.run.processor_identity_sha256,
        "runtime_identity_sha256": plan.run.runtime_identity_sha256,
        "sampling": dict(plan.run.sampling),
        "sampling_config_sha256": _sha256_json(dict(plan.run.sampling)),
        "selector_schema": T1_REPLAY_AUDIT_SELECTOR_SCHEMA,
        "candidates_per_source": T1_REPLAY_AUDIT_CANDIDATES_PER_SOURCE,
        "raw_evidence_count": plan.raw_evidence_count,
        "logical_attempt_count": plan.logical_attempt_count,
        "audited_candidate_count": len(plan.selections),
        "audited_attempt_count": plan.audited_attempt_count,
        "selection": [
            {
                "source": selection.candidate.source.value,
                "sample_id": selection.candidate.sample_id,
                "candidate_sha256": selection.candidate.identity_sha256,
                "selector_sha256": selection.selector_sha256,
                "local_chunk_index": selection.local_chunk_index,
            }
            for selection in plan.selections
        ],
        "comparisons": list(comparisons),
        "passed": passed,
    }
    payload = _canonical_json_bytes(report) + b"\n"
    report_sha256 = hashlib.sha256(payload).hexdigest()
    relative_path = _AUDIT_DIRECTORY / f"{report_sha256}.json"
    _atomic_write_immutable(plan.run.output_root / relative_path, payload)
    return {
        "run_id": plan.run.run_id,
        "rank": plan.rank,
        "audited_candidate_count": len(plan.selections),
        "audited_attempt_count": plan.audited_attempt_count,
        "passed": passed,
        "report_path": str(plan.run.output_root / relative_path),
        "report_sha256": report_sha256,
    }


async def run_t1_replay_audit(
    config_path: str | Path, *, rank: int
) -> dict[str, object]:
    """Replay a fixed three-source subset and publish an immutable report."""

    plan = plan_t1_replay_audit(config_path, rank=rank)
    _validate_replay_environment(plan)
    _validate_runtime_versions(plan.run)
    comparisons = await _replay_plan(plan)
    result = _write_audit_report(plan, comparisons)
    if result["passed"] is not True:
        raise T1ReplayAuditFailure(result)
    return result


__all__ = [
    "LocatedT1Evidence",
    "T1_REPLAY_AUDIT_CANDIDATES_PER_SOURCE",
    "T1_REPLAY_AUDIT_PLAN_SCHEMA",
    "T1_REPLAY_AUDIT_SCHEMA",
    "T1_REPLAY_AUDIT_SELECTOR_SCHEMA",
    "T1ReplayAuditFailure",
    "T1ReplayAuditPlan",
    "T1ReplayAuditSelection",
    "compare_replayed_evidence",
    "plan_t1_replay_audit",
    "run_t1_replay_audit",
    "select_replay_audit_candidates",
]
