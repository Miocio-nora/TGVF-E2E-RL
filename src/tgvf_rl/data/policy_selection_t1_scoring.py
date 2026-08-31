"""Deterministic scoring materialization for the accepted T1 canary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_TASK_KIND_POLICY,
    AttemptStatus,
    SelectionBranch,
    SelectionCandidate,
    SelectionSource,
    canonical_json_line,
    policy_selection_semantic_judge_task_kind,
    reduce_selection_attempts,
    stable_selection_request_id,
    summarize_selection_decisions,
)
from .policy_selection_runtime import (
    T1_ATTEMPTS,
    T1RawGenerationEvidence,
    T1RunConfig,
    _atomic_write_immutable,
    evidence_to_attempt_record,
    load_t1_run_config,
)
from .policy_selection_vllm import (
    _validate_prepared_output_root,
    load_t1_candidates,
)
from .policy_selection_vllm_retry import (
    _load_validated_evidence,
    validate_length_retry_identity,
)


T1_EFFECTIVE_GENERATION_SCHEMA = "tgvf.policy-selection.t1-effective-generation.v1"
T1_SEMANTIC_JUDGE_REQUEST_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-request.v1"
T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA = (
    "tgvf.policy-selection.t1-deterministic-scoring-manifest.v3"
)
T1_SCORING_DIRECTORY = Path("scoring") / "deterministic-v3"
T1_UNRETRIED_LENGTH_WAIVER_RUN_IDS = frozenset(
    {"T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123"}
)
T1_PATHOLOGICAL_GENERATION_LENGTH_REASON = "pathological_generation_length"
T1_PATHOLOGICAL_GENERATION_MIN_TOKENS = 98_000
_T1_PATHOLOGICAL_EXCLUSION_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_sha256",
        "candidate_sha256",
        "sample_id",
        "source",
        "reason",
        "question",
        "ground_truth",
        "generation_length_waivers",
    }
)
_T1_PATHOLOGICAL_WAIVER_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_sha256",
        "request_id",
        "sample_id",
        "candidate_sha256",
        "source",
        "attempt_index",
        "terminal_budget_revision",
        "terminal_finish_reason",
        "terminal_evidence_sha256",
        "terminal_sampled_token_count",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records_payload(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(record) for record in records)


def _file_record(relative: Path, payload: bytes, *, rows: int | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative.as_posix(),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _required_lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _parse_quality_exclusions(
    quality_config: object,
    *,
    run: T1RunConfig,
    candidates_by_sha: Mapping[str, SelectionCandidate],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    tuple[dict[str, Any], ...],
]:
    """Bind candidate exclusions and exact rev1-length waiver declarations."""

    if (
        not isinstance(quality_config, Mapping)
        or quality_config.get("schema_version")
        != "tgvf.policy-selection.t1-quality-exclusions.v1"
        or not isinstance(quality_config.get("exclusions"), list)
    ):
        raise ValueError("T1 quality-exclusion config schema differs")
    quality_exclusions: dict[str, Mapping[str, Any]] = {}
    length_waivers: dict[str, Mapping[str, Any]] = {}
    waiver_records: list[dict[str, Any]] = []
    for exclusion in quality_config["exclusions"]:
        if not isinstance(exclusion, Mapping):
            raise ValueError("T1 quality exclusion must be an object")
        candidate_sha256 = exclusion.get("candidate_sha256")
        candidate = candidates_by_sha.get(str(candidate_sha256))
        if candidate is None:
            raise ValueError("T1 quality exclusion refers to an unknown candidate")
        if (
            exclusion.get("sample_id") != candidate.sample_id
            or exclusion.get("source") != candidate.source.value
            or exclusion.get("question") != candidate.question
            or exclusion.get("ground_truth") != candidate.ground_truth
        ):
            raise ValueError("T1 quality exclusion evidence differs from candidate")
        reason = exclusion.get("reason")
        if reason == "source_ground_truth_truncated":
            if "generation_length_waivers" in exclusion:
                raise ValueError(
                    "source-ground-truth exclusion cannot declare length waivers"
                )
        elif reason == T1_PATHOLOGICAL_GENERATION_LENGTH_REASON:
            if set(exclusion) != _T1_PATHOLOGICAL_EXCLUSION_FIELDS:
                raise ValueError(
                    "pathological-generation exclusion fields differ"
                )
            if (
                run.selection.get("kind") != "teacher_full"
                or candidate.source is not SelectionSource.TEACHER
                or exclusion.get("run_id") != run.run_id
                or exclusion.get("run_manifest_sha256") != run.manifest_sha256
            ):
                raise ValueError(
                    "pathological-generation exclusion is outside its teacher run"
                )
            waivers = exclusion.get("generation_length_waivers")
            if not isinstance(waivers, list):
                raise ValueError(
                    "pathological-generation request waivers must be a list"
                )
            for waiver in waivers:
                if (
                    not isinstance(waiver, Mapping)
                    or set(waiver) != _T1_PATHOLOGICAL_WAIVER_FIELDS
                ):
                    raise ValueError(
                        "pathological-generation request waiver fields differ"
                    )
                attempt_index = waiver.get("attempt_index")
                if type(attempt_index) is not int or not 0 <= attempt_index < T1_ATTEMPTS:
                    raise ValueError(
                        "pathological-generation waiver attempt_index is invalid"
                    )
                expected_request_id = stable_selection_request_id(
                    candidate_sha256=candidate.identity_sha256,
                    branch=SelectionBranch.FULL_IMAGE,
                    attempt_index=attempt_index,
                )
                token_count = waiver.get("terminal_sampled_token_count")
                if (
                    waiver.get("run_id") != run.run_id
                    or waiver.get("run_manifest_sha256") != run.manifest_sha256
                    or waiver.get("request_id") != expected_request_id
                    or waiver.get("sample_id") != candidate.sample_id
                    or waiver.get("candidate_sha256") != candidate.identity_sha256
                    or waiver.get("source") != SelectionSource.TEACHER.value
                    or waiver.get("terminal_budget_revision") != 1
                    or waiver.get("terminal_finish_reason") != "length"
                    or type(token_count) is not int
                    or token_count <= T1_PATHOLOGICAL_GENERATION_MIN_TOKENS
                ):
                    raise ValueError(
                        "pathological-generation request waiver identity differs"
                    )
                _required_lower_sha256(
                    waiver.get("terminal_evidence_sha256"),
                    field="terminal_evidence_sha256",
                )
                if expected_request_id in length_waivers:
                    raise ValueError(
                        "duplicate pathological-generation request waiver"
                    )
                waiver_record = dict(waiver)
                length_waivers[expected_request_id] = waiver_record
                waiver_records.append(waiver_record)
        else:
            raise ValueError("T1 quality exclusion reason is unsupported")
        if str(candidate_sha256) in quality_exclusions:
            raise ValueError("duplicate T1 quality exclusion")
        quality_exclusions[str(candidate_sha256)] = exclusion
    waiver_records.sort(key=lambda item: str(item["request_id"]))
    return quality_exclusions, length_waivers, tuple(waiver_records)


def _validate_pathological_length_waiver(
    waiver: Mapping[str, Any],
    *,
    run: T1RunConfig,
    candidate: SelectionCandidate,
    evidence: T1RawGenerationEvidence,
    terminal_revision: int,
) -> None:
    """Prove that one declared waiver is the observed rev1 98k+ length finish."""

    if (
        run.selection.get("kind") != "teacher_full"
        or candidate.source is not SelectionSource.TEACHER
        or waiver.get("run_id") != run.run_id
        or waiver.get("run_manifest_sha256") != run.manifest_sha256
        or waiver.get("request_id") != evidence.request_id
        or waiver.get("sample_id") != candidate.sample_id
        or waiver.get("candidate_sha256") != candidate.identity_sha256
        or waiver.get("source") != candidate.source.value
        or waiver.get("attempt_index") != evidence.attempt_index
        or waiver.get("terminal_budget_revision") != terminal_revision
        or terminal_revision != 1
        or evidence.budget_revision != terminal_revision
        or waiver.get("terminal_finish_reason") != "length"
        or evidence.finish_reason != "length"
        or waiver.get("terminal_evidence_sha256") != evidence.evidence_sha256
        or waiver.get("terminal_sampled_token_count")
        != evidence.sampled_token_count
        or type(evidence.sampled_token_count) is not int
        or evidence.sampled_token_count <= T1_PATHOLOGICAL_GENERATION_MIN_TOKENS
    ):
        raise ValueError("pathological-generation waiver evidence identity differs")


def _apply_quality_exclusion(
    attempt: Mapping[str, Any], exclusion: Mapping[str, Any]
) -> dict[str, Any]:
    reason = str(exclusion["reason"])
    route = {
        "source_ground_truth_truncated": "source_ground_truth_invalid",
        T1_PATHOLOGICAL_GENERATION_LENGTH_REASON: (
            "source_generation_anomaly_invalid"
        ),
    }[reason]
    return {
        **attempt,
        "status": AttemptStatus.VERIFIER_ERROR.value,
        "correct": None,
        "verification_route": route,
        "verification_evidence": reason,
        "semantic_required": False,
        "semantic_judge_evidence_sha256": None,
    }


def _pathological_waiver_binding(
    waiver_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _records_payload(waiver_records)
    return {
        "pathological_generation_length_waiver_count": len(waiver_records),
        "pathological_generation_length_waivers_sha256": _sha256_bytes(payload),
    }


def _expected_requests(
    candidates: Sequence[SelectionCandidate],
) -> dict[str, tuple[SelectionCandidate, int]]:
    expected: dict[str, tuple[SelectionCandidate, int]] = {}
    for candidate in candidates:
        for attempt_index in range(T1_ATTEMPTS):
            request_id = stable_selection_request_id(
                candidate_sha256=candidate.identity_sha256,
                branch=SelectionBranch.FULL_IMAGE,
                attempt_index=attempt_index,
            )
            if request_id in expected:
                raise ValueError("duplicate expected T1 request ID")
            expected[request_id] = (candidate, attempt_index)
    return expected


def _effective_generations(
    run: T1RunConfig,
    candidates: Sequence[SelectionCandidate],
    *,
    pathological_length_waivers: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    tuple[tuple[SelectionCandidate, T1RawGenerationEvidence, bool], ...],
    tuple[dict[str, Any], ...],
]:
    length_waivers = (
        {} if pathological_length_waivers is None else pathological_length_waivers
    )
    consumed_length_waivers: set[str] = set()
    expected = _expected_requests(candidates)
    located = _load_validated_evidence(run)
    histories: dict[str, dict[int, tuple[Any, T1RawGenerationEvidence]]] = {
        request_id: {} for request_id in expected
    }
    for manifest, evidence in located:
        expected_item = expected.get(evidence.request_id)
        if expected_item is None:
            raise ValueError("generation evidence refers to an unknown T1 request")
        candidate, attempt_index = expected_item
        if (
            evidence.candidate_sha256 != candidate.identity_sha256
            or evidence.sample_id != candidate.sample_id
            or evidence.attempt_index != attempt_index
            or evidence.source is not candidate.source
        ):
            raise ValueError("generation evidence differs from its candidate request")
        history = histories[evidence.request_id]
        if evidence.budget_revision in history:
            raise ValueError("duplicate generation budget revision")
        history[evidence.budget_revision] = (manifest, evidence)

    maximum_revision = max(budget.revision for budget in run.response_budgets)
    selected: list[tuple[SelectionCandidate, T1RawGenerationEvidence, bool]] = []
    pointers: list[dict[str, Any]] = []
    for request_id, (candidate, attempt_index) in sorted(
        expected.items(), key=lambda item: (item[1][0].sample_id, item[1][1])
    ):
        history = histories[request_id]
        if 0 not in history:
            raise ValueError("revision-0 T1 generation is incomplete")
        revisions = sorted(history)
        if revisions != list(range(revisions[-1] + 1)):
            raise ValueError("T1 response-budget history has a gap")
        for revision in revisions[1:]:
            validate_length_retry_identity(
                history[revision - 1][1], history[revision][1]
            )
        terminal_revision = revisions[-1]
        manifest, evidence = history[terminal_revision]
        if evidence.finish_reason == "length":
            if terminal_revision < maximum_revision:
                waiver = length_waivers.get(request_id)
                if waiver is not None:
                    _validate_pathological_length_waiver(
                        waiver,
                        run=run,
                        candidate=candidate,
                        evidence=evidence,
                        terminal_revision=terminal_revision,
                    )
                    consumed_length_waivers.add(request_id)
                elif run.run_id not in T1_UNRETRIED_LENGTH_WAIVER_RUN_IDS:
                    raise ValueError(
                        "a length finish still requires a response-budget retry"
                    )
            # Accepted historical or exact pathological cases remain
            # truncated/unscoreable.  No later revision is synthesized.
            budget_exhausted = True
        else:
            budget_exhausted = False
        selected.append((candidate, evidence, budget_exhausted))
        pointers.append(
            {
                "schema_version": T1_EFFECTIVE_GENERATION_SCHEMA,
                "request_id": request_id,
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "source": candidate.source.value,
                "attempt_index": attempt_index,
                "selected_budget_revision": terminal_revision,
                "raw_generation_sha256": evidence.evidence_sha256,
                "finish_reason": evidence.finish_reason,
                "sampled_token_count": evidence.sampled_token_count,
                "budget_exhausted": budget_exhausted,
                "chunk_manifest_sha256": manifest.manifest_sha256,
            }
        )
    unconsumed = set(length_waivers) - consumed_length_waivers
    if unconsumed:
        raise ValueError(
            "pathological-generation request waiver was not consumed: "
            f"{min(unconsumed)}"
        )
    return tuple(selected), tuple(pointers)


def _option_count(candidate: SelectionCandidate) -> int | None:
    if candidate.source not in {
        SelectionSource.ARXIVQA,
        SelectionSource.TEACHER,
    }:
        return None
    metadata = candidate.canonical_record.get("selection_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("MCQ-capable candidate has no selection_metadata")
    if candidate.source is SelectionSource.TEACHER:
        task_kind = metadata.get("task_kind")
        if task_kind == "open":
            return None
        if task_kind != "mcq":
            raise ValueError("teacher candidate task_kind is invalid")
    option_count = metadata.get("option_count")
    if type(option_count) is not int or not 2 <= option_count <= 26:
        raise ValueError("candidate option_count is invalid")
    return option_count


def _judge_queue(
    attempts: Sequence[Mapping[str, Any]],
    selected: Sequence[tuple[SelectionCandidate, T1RawGenerationEvidence, bool]],
    *,
    run: T1RunConfig,
    judge_config_sha256: str,
) -> tuple[dict[str, Any], ...]:
    selected_by_request = {
        evidence.request_id: (candidate, evidence)
        for candidate, evidence, _ in selected
    }
    prompt_sha256 = str(run.verifier["semantic_judge"]["prompt_sha256"])
    grouped: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if attempt["status"] != AttemptStatus.VERIFIER_ERROR.value:
            continue
        if attempt.get("semantic_required") is not True:
            continue
        candidate, evidence = selected_by_request[attempt["request_id"]]
        if candidate.source is SelectionSource.ARXIVQA:
            raise ValueError("ArxivQA must never enter the semantic judge queue")
        answer = attempt.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("semantic judge candidate answer must be non-empty")
        if not isinstance(candidate.ground_truth, str):
            raise ValueError("semantic judge reference answer must be a string")
        task_kind = policy_selection_semantic_judge_task_kind(
            source=candidate.source,
            question=candidate.question,
            ground_truth=candidate.ground_truth,
        )
        payload = {
            "task_kind": task_kind,
            "question": candidate.question,
            "candidate_answer": answer,
            "reference_answer": candidate.ground_truth,
        }
        payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
        judge_request_id = f"t1-semantic-judge:{payload_sha256}"
        consumer = {
            "request_id": evidence.request_id,
            "sample_id": candidate.sample_id,
            "candidate_sha256": candidate.identity_sha256,
            "source": candidate.source.value,
            "attempt_index": evidence.attempt_index,
            "raw_generation_sha256": evidence.evidence_sha256,
        }
        existing = grouped.get(payload_sha256)
        if existing is None:
            grouped[payload_sha256] = {
                "schema_version": T1_SEMANTIC_JUDGE_REQUEST_SCHEMA,
                "judge_request_id": judge_request_id,
                "run_id": run.run_id,
                "run_manifest_sha256": run.manifest_sha256,
                "prompt_sha256": prompt_sha256,
                "judge_config_sha256": judge_config_sha256,
                "model_repository": run.verifier["semantic_judge"]["repository"],
                "model_served_name": run.verifier["semantic_judge"]["served_name"],
                "payload_sha256": payload_sha256,
                **payload,
                "consumers": [consumer],
            }
        else:
            if any(existing[field] != value for field, value in payload.items()):
                raise ValueError("semantic judge payload SHA collision")
            existing["consumers"].append(consumer)
    records: list[dict[str, Any]] = []
    for payload_sha256 in sorted(grouped):
        record = grouped[payload_sha256]
        record["consumers"] = sorted(
            record["consumers"],
            key=lambda item: (item["sample_id"], item["attempt_index"]),
        )
        record["consumer_count"] = len(record["consumers"])
        records.append(record)
    return tuple(records)


def materialize_t1_deterministic_scoring(
    config_path: str | Path,
    *,
    judge_config_path: str | Path,
    quality_exclusions_path: str | Path,
) -> dict[str, Any]:
    """Validate generation, score rule routes, and publish the minimal judge queue."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=True)
    _validate_prepared_output_root(run, path)
    judge_path = Path(judge_config_path).resolve()
    judge_payload = judge_path.read_bytes()
    judge_config_sha256 = _sha256_bytes(judge_payload)
    if judge_config_sha256 != run.verifier["semantic_judge"]["config_sha256"]:
        raise ValueError("semantic judge config SHA-256 differs from the run")
    judge_config = json.loads(judge_payload)
    if (
        not isinstance(judge_config, Mapping)
        or not isinstance(judge_config.get("prompt"), Mapping)
        or judge_config["prompt"].get("sha256")
        != run.verifier["semantic_judge"]["prompt_sha256"]
    ):
        raise ValueError("semantic judge prompt identity differs from the run")
    candidates = load_t1_candidates(run)
    candidates_by_sha = {
        candidate.identity_sha256: candidate for candidate in candidates
    }
    quality_path = Path(quality_exclusions_path).resolve()
    quality_payload = quality_path.read_bytes()
    quality_sha256 = _sha256_bytes(quality_payload)
    quality_config = json.loads(quality_payload)
    (
        quality_exclusions,
        pathological_length_waivers,
        pathological_waiver_records,
    ) = _parse_quality_exclusions(
        quality_config,
        run=run,
        candidates_by_sha=candidates_by_sha,
    )
    selected, effective_records = _effective_generations(
        run,
        candidates,
        pathological_length_waivers=pathological_length_waivers,
    )

    attempts: list[dict[str, Any]] = []
    for candidate, evidence, budget_exhausted in selected:
        attempt = evidence_to_attempt_record(
            evidence,
            expected_answer=candidate.ground_truth,
            option_count=_option_count(candidate),
            budget_exhausted=budget_exhausted,
            answer_parser=str(run.verifier["answer_parser"]),
        )
        if attempt is None:
            raise ValueError(
                "effective T1 evidence unexpectedly requests another retry"
            )
        exclusion = quality_exclusions.get(candidate.identity_sha256)
        if exclusion is not None:
            attempt = _apply_quality_exclusion(attempt, exclusion)
        attempts.append(attempt)
    attempts.sort(key=lambda item: (item["sample_id"], item["attempt_index"]))
    judge_requests = _judge_queue(
        attempts,
        selected,
        run=run,
        judge_config_sha256=judge_config_sha256,
    )
    decisions = reduce_selection_attempts(
        (candidate.canonical_record for candidate in candidates), attempts
    )
    summary = summarize_selection_decisions(decisions)

    status_counts = Counter(item["status"] for item in attempts)
    route_counts = Counter(item["verification_route"] for item in attempts)
    semantic_consumers = sum(item["consumer_count"] for item in judge_requests)
    source_attempts = Counter(item["source"] for item in attempts)
    source_semantic = Counter(
        consumer["source"]
        for request in judge_requests
        for consumer in request["consumers"]
    )
    pathological_candidate_count = sum(
        exclusion.get("reason") == T1_PATHOLOGICAL_GENERATION_LENGTH_REASON
        for exclusion in quality_exclusions.values()
    )
    pathological_waiver_binding = (
        _pathological_waiver_binding(pathological_waiver_records)
        if pathological_candidate_count
        else {}
    )
    report = {
        "schema_version": "tgvf.policy-selection.t1-deterministic-report.v3",
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "task_kind_policy": POLICY_SELECTION_TASK_KIND_POLICY,
        "candidate_count": len(candidates),
        "logical_attempt_count": len(attempts),
        "effective_generation_count": len(effective_records),
        "status_counts": dict(sorted(status_counts.items())),
        "verification_route_counts": dict(sorted(route_counts.items())),
        "source_attempt_counts": dict(sorted(source_attempts.items())),
        "semantic_judge_consumer_count": semantic_consumers,
        "semantic_judge_unique_request_count": len(judge_requests),
        "semantic_judge_consumers_by_source": dict(sorted(source_semantic.items())),
        "arxivqa_judge_calls": source_semantic[SelectionSource.ARXIVQA.value],
        "quality_exclusion_candidate_count": len(quality_exclusions),
        "quality_exclusion_attempt_count": sum(
            item["verification_route"]
            in {
                "source_ground_truth_invalid",
                "source_generation_anomaly_invalid",
            }
            for item in attempts
        ),
        "quality_exclusions_sha256": quality_sha256,
        "provisional_selection_summary": summary,
        **pathological_waiver_binding,
    }
    if pathological_candidate_count:
        report.update(
            {
                "pathological_generation_exclusion_candidate_count": (
                    pathological_candidate_count
                ),
                "pathological_generation_exclusion_attempt_count": sum(
                    item["verification_route"]
                    == "source_generation_anomaly_invalid"
                    for item in attempts
                ),
            }
        )

    output_root = run.output_root / T1_SCORING_DIRECTORY
    payloads: dict[str, tuple[Path, bytes, int | None]] = {
        "effective_generations": (
            Path("effective-generations.jsonl"),
            _records_payload(effective_records),
            len(effective_records),
        ),
        "attempts": (
            Path("attempts.jsonl"),
            _records_payload(attempts),
            len(attempts),
        ),
        "semantic_judge_requests": (
            Path("semantic-judge-requests.jsonl"),
            _records_payload(judge_requests),
            len(judge_requests),
        ),
        "provisional_decisions": (
            Path("provisional-decisions.jsonl"),
            _records_payload(decisions),
            len(decisions),
        ),
        "report": (
            Path("report.json"),
            _canonical_json_bytes(report) + b"\n",
            None,
        ),
    }
    files: dict[str, dict[str, Any]] = {}
    for name, (relative, payload, rows) in payloads.items():
        _atomic_write_immutable(output_root / relative, payload)
        files[name] = _file_record(relative, payload, rows=rows)
    manifest_identity = {
        "schema_version": T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        "run_id": run.run_id,
        "run_manifest_sha256": run.manifest_sha256,
        "task_kind_policy": POLICY_SELECTION_TASK_KIND_POLICY,
        "selection_candidates_sha256": run.selection["candidates_sha256"],
        "judge_config_sha256": judge_config_sha256,
        "quality_exclusions_sha256": quality_sha256,
        "files": files,
        **pathological_waiver_binding,
    }
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest_identity))
    manifest = {**manifest_identity, "manifest_sha256": manifest_sha256}
    manifest_payload = _canonical_json_bytes(manifest) + b"\n"
    _atomic_write_immutable(output_root / "manifest.json", manifest_payload)
    return {**report, "manifest_sha256": manifest_sha256, "files": files}


__all__ = [
    "T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA",
    "T1_EFFECTIVE_GENERATION_SCHEMA",
    "T1_SCORING_DIRECTORY",
    "T1_SEMANTIC_JUDGE_REQUEST_SCHEMA",
    "materialize_t1_deterministic_scoring",
]
