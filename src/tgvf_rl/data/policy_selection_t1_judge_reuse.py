"""Fail-closed reuse of unchanged T1 semantic-judge requests.

This module exists for the one intentional identity migration from the T1
``deterministic-v2/judge-v1`` artifacts to ``deterministic-v3/judge-v2``.
Only an *identical* semantic payload may reuse a response.  Source indices,
evidence, judge bindings, response semantics, and both queue files are fully
validated before a target record is published.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from tgvf_rl.judges.openai_compatible import (
    BoundOpenAICompatibleJudge,
    load_openai_compatible_judge,
)

from .policy_selection import POLICY_SELECTION_TASK_KIND_POLICY
from .policy_selection_runtime import _atomic_write_immutable, load_t1_run_config
from .policy_selection_t1_judge import (
    T1_JUDGE_DIRECTORY,
    T1_JUDGE_EVIDENCE_SCHEMA,
    T1_JUDGE_INDEX_SCHEMA,
    _canonical_json_bytes,
    _identity_record,
    _index_path,
    _load_completed_index,
    _request_payload,
    _sha256_bytes,
    _strict_response,
)
from .policy_selection_t1_scoring import (
    T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
    T1_SCORING_DIRECTORY,
    T1_SEMANTIC_JUDGE_REQUEST_SCHEMA,
)
from .policy_selection_vllm import _validate_prepared_output_root


T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA = (
    "tgvf.policy-selection.t1-deterministic-scoring-manifest.v2"
)
T1_LEGACY_JUDGE_EVIDENCE_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-evidence.v1"
T1_LEGACY_JUDGE_INDEX_SCHEMA = "tgvf.policy-selection.t1-semantic-judge-index.v1"
T1_LEGACY_SCORING_DIRECTORY = Path("scoring") / "deterministic-v2"
T1_LEGACY_JUDGE_DIRECTORY = Path("scoring") / "judge-v1"
T1_JUDGE_REUSE_PROVENANCE_SCHEMA = (
    "tgvf.policy-selection.t1-semantic-judge-reuse-provenance.v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "judge_request_id",
        "run_id",
        "run_manifest_sha256",
        "prompt_sha256",
        "judge_config_sha256",
        "model_repository",
        "model_served_name",
        "payload_sha256",
        "task_kind",
        "question",
        "candidate_answer",
        "reference_answer",
        "consumers",
        "consumer_count",
    }
)
_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "run_manifest_sha256",
        "scoring_manifest_sha256",
        "judge_request_id",
        "payload_sha256",
        "evidence_sha256",
        "evidence_file",
        "evidence_file_sha256",
        "verdict",
        "index_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class _RunIdentity:
    run_id: str
    run_manifest_sha256: str
    selection_candidates_sha256: str
    judge_config_sha256: str
    prompt_sha256: str
    model_repository: str
    model_served_name: str


@dataclass(frozen=True, slots=True)
class _QueueBinding:
    root: Path
    path: Path
    schema_version: str
    manifest_sha256: str
    quality_exclusions_sha256: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ReuseOutcome:
    status: str
    verdict: int | None = None


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_artifact_path(root: Path, relative: object, *, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} is not a canonical relative path")
    resolved_root = root.resolve()
    resolved = (root / Path(*pure.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} escapes its artifact root")
    return resolved


def _load_queue_binding(
    root: Path,
    *,
    expected_schema: str,
    identity: _RunIdentity,
    expected_task_kind_policy: str | None = None,
) -> _QueueBinding:
    manifest_path = root / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, Mapping):
        raise ValueError("deterministic scoring manifest must be an object")
    if raw_manifest != _canonical_json_bytes(manifest) + b"\n":
        raise ValueError("deterministic scoring manifest is not canonical JSON")
    manifest_identity = dict(manifest)
    manifest_sha256 = manifest_identity.pop("manifest_sha256", None)
    if manifest_sha256 != _sha256_bytes(_canonical_json_bytes(manifest_identity)):
        raise ValueError("deterministic scoring manifest SHA-256 differs")
    expected = {
        "schema_version": expected_schema,
        "run_id": identity.run_id,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "selection_candidates_sha256": identity.selection_candidates_sha256,
        "judge_config_sha256": identity.judge_config_sha256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"deterministic scoring manifest {field} differs")
    if (
        expected_task_kind_policy is not None
        and manifest.get("task_kind_policy") != expected_task_kind_policy
    ):
        raise ValueError("deterministic scoring manifest task_kind_policy differs")
    quality_sha256 = _require_sha256(
        manifest.get("quality_exclusions_sha256"),
        name="quality exclusions identity",
    )
    record = manifest.get("files", {}).get("semantic_judge_requests")
    if not isinstance(record, Mapping):
        raise ValueError("deterministic scoring judge queue is missing")
    rows = record.get("rows")
    byte_count = record.get("bytes")
    if type(rows) is not int or rows < 0:
        raise ValueError("deterministic scoring judge queue row count is invalid")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError("deterministic scoring judge queue byte count is invalid")
    queue_sha256 = _require_sha256(
        record.get("sha256"), name="deterministic scoring judge queue identity"
    )
    queue_path = _safe_artifact_path(
        root, record.get("path"), name="deterministic scoring judge queue path"
    )
    if not queue_path.is_file():
        raise ValueError("deterministic scoring judge queue is not a regular file")
    return _QueueBinding(
        root=root,
        path=queue_path,
        schema_version=expected_schema,
        manifest_sha256=str(manifest_sha256),
        quality_exclusions_sha256=quality_sha256,
        rows=rows,
        bytes=byte_count,
        sha256=queue_sha256,
    )


def _payload(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_kind": request.get("task_kind"),
        "question": request.get("question"),
        "candidate_answer": request.get("candidate_answer"),
        "reference_answer": request.get("reference_answer"),
    }


def _validate_request(request: object, *, identity: _RunIdentity) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
        raise ValueError("semantic judge request schema differs")
    expected = {
        "schema_version": T1_SEMANTIC_JUDGE_REQUEST_SCHEMA,
        "run_id": identity.run_id,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "prompt_sha256": identity.prompt_sha256,
        "judge_config_sha256": identity.judge_config_sha256,
        "model_repository": identity.model_repository,
        "model_served_name": identity.model_served_name,
    }
    for field, value in expected.items():
        if request.get(field) != value:
            raise ValueError(f"semantic judge request {field} differs")
    payload = _payload(request)
    if payload["task_kind"] not in {"math", "open_vqa"}:
        raise ValueError("semantic judge request task kind is invalid")
    if any(not isinstance(payload[field], str) for field in payload):
        raise ValueError("semantic judge request payload must contain strings")
    if not str(payload["candidate_answer"]).strip():
        raise ValueError("semantic judge candidate answer is empty")
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    if (
        request.get("payload_sha256") != payload_sha256
        or request.get("judge_request_id") != f"t1-semantic-judge:{payload_sha256}"
    ):
        raise ValueError("semantic judge request payload identity differs")
    consumers = request.get("consumers")
    if not isinstance(consumers, list) or not consumers:
        raise ValueError("semantic judge request consumers are invalid")
    if request.get("consumer_count") != len(consumers):
        raise ValueError("semantic judge request consumer count differs")
    ordering: list[tuple[str, int]] = []
    consumer_ids: set[str] = set()
    for consumer in consumers:
        if not isinstance(consumer, Mapping):
            raise ValueError("semantic judge consumer must be an object")
        sample_id = consumer.get("sample_id")
        attempt_index = consumer.get("attempt_index")
        request_id = consumer.get("request_id")
        if (
            not isinstance(sample_id, str)
            or type(attempt_index) is not int
            or attempt_index < 0
            or not isinstance(request_id, str)
            or not request_id
        ):
            raise ValueError("semantic judge consumer identity is invalid")
        if request_id in consumer_ids:
            raise ValueError("semantic judge request has a duplicate consumer")
        consumer_ids.add(request_id)
        ordering.append((sample_id, attempt_index))
    if ordering != sorted(ordering):
        raise ValueError("semantic judge request consumers are not canonical")
    return dict(request)


def _iter_validated_queue(
    binding: _QueueBinding, *, identity: _RunIdentity
) -> Iterator[dict[str, Any]]:
    hasher = hashlib.sha256()
    rows = 0
    byte_count = 0
    previous_payload_sha256: str | None = None
    with binding.path.open("rb") as handle:
        for raw_line in handle:
            hasher.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.endswith(b"\n") or not raw_line.strip():
                raise ValueError("semantic judge queue is not canonical JSONL")
            request = _validate_request(json.loads(raw_line), identity=identity)
            if raw_line != _canonical_json_bytes(request) + b"\n":
                raise ValueError("semantic judge queue row is not canonical JSON")
            payload_sha256 = str(request["payload_sha256"])
            if (
                previous_payload_sha256 is not None
                and payload_sha256 <= previous_payload_sha256
            ):
                raise ValueError("semantic judge queue order or uniqueness differs")
            previous_payload_sha256 = payload_sha256
            rows += 1
            yield request
    if rows != binding.rows:
        raise ValueError("semantic judge queue row count differs")
    if byte_count != binding.bytes:
        raise ValueError("semantic judge queue byte count differs")
    if hasher.hexdigest() != binding.sha256:
        raise ValueError("semantic judge queue SHA-256 differs")


def _fully_validate_queue(binding: _QueueBinding, *, identity: _RunIdentity) -> None:
    for _ in _iter_validated_queue(binding, identity=identity):
        pass


def _legacy_index_path(root: Path, payload_sha256: str) -> Path:
    sharded = root / "requests" / payload_sha256[:2] / f"{payload_sha256}.json"
    flat = root / "requests" / f"{payload_sha256}.json"
    sharded_exists = sharded.exists()
    flat_exists = flat.exists()
    if sharded_exists and flat_exists:
        if not sharded.is_file() or not flat.is_file():
            raise ValueError("legacy semantic judge index is not a regular file")
        if sharded.read_bytes() != flat.read_bytes():
            raise ValueError("legacy flat and sharded judge indices differ")
        return sharded
    if flat_exists:
        return flat
    return sharded


def _validated_legacy_evidence(
    source_judge_root: Path,
    request: Mapping[str, Any],
    *,
    identity: _RunIdentity,
    source_scoring_manifest_sha256: str,
    bound: BoundOpenAICompatibleJudge,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    index_path = _legacy_index_path(source_judge_root, str(request["payload_sha256"]))
    if not index_path.exists():
        return None
    if not index_path.is_file():
        raise ValueError("legacy semantic judge index is not a regular file")
    raw_index = index_path.read_bytes()
    index = json.loads(raw_index)
    if not isinstance(index, Mapping) or set(index) != _INDEX_FIELDS:
        raise ValueError("legacy semantic judge index schema differs")
    if raw_index != _canonical_json_bytes(index) + b"\n":
        raise ValueError("legacy semantic judge index is not canonical JSON")
    index_identity = dict(index)
    index_sha256 = index_identity.pop("index_sha256", None)
    if index_sha256 != _sha256_bytes(_canonical_json_bytes(index_identity)):
        raise ValueError("legacy semantic judge index SHA-256 differs")
    verdict = index.get("verdict")
    if type(verdict) is not int or verdict not in {0, 1}:
        raise ValueError("legacy semantic judge index verdict is invalid")
    expected_index = {
        "schema_version": T1_LEGACY_JUDGE_INDEX_SCHEMA,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "scoring_manifest_sha256": source_scoring_manifest_sha256,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
    }
    for field, value in expected_index.items():
        if index.get(field) != value:
            raise ValueError(f"legacy semantic judge index {field} differs")

    evidence_sha256 = _require_sha256(
        index.get("evidence_sha256"), name="legacy semantic judge evidence identity"
    )
    relative = index.get("evidence_file")
    canonical_paths = {
        f"evidence/{evidence_sha256}.json",
        f"evidence/{evidence_sha256[:2]}/{evidence_sha256}.json",
    }
    if relative not in canonical_paths:
        raise ValueError("legacy semantic judge evidence path differs")
    evidence_path = _safe_artifact_path(
        source_judge_root, relative, name="legacy semantic judge evidence path"
    )
    if not evidence_path.is_file():
        raise ValueError("legacy semantic judge evidence is not a regular file")
    raw_evidence = evidence_path.read_bytes()
    if _sha256_bytes(raw_evidence) != index.get("evidence_file_sha256"):
        raise ValueError("legacy semantic judge evidence file SHA-256 differs")
    evidence = json.loads(raw_evidence)
    if not isinstance(evidence, Mapping):
        raise ValueError("legacy semantic judge evidence must be an object")
    if raw_evidence != _canonical_json_bytes(evidence) + b"\n":
        raise ValueError("legacy semantic judge evidence is not canonical JSON")
    evidence_identity = dict(evidence)
    observed_evidence_sha256 = evidence_identity.pop("evidence_sha256", None)
    if observed_evidence_sha256 != evidence_sha256 or evidence_sha256 != _sha256_bytes(
        _canonical_json_bytes(evidence_identity)
    ):
        raise ValueError("legacy semantic judge evidence identity SHA-256 differs")

    expected_evidence = {
        "schema_version": T1_LEGACY_JUDGE_EVIDENCE_SCHEMA,
        "run_id": identity.run_id,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "scoring_manifest_sha256": source_scoring_manifest_sha256,
        "judge_request_id": request["judge_request_id"],
        "payload_sha256": request["payload_sha256"],
        "consumer_count": request["consumer_count"],
        "judge_config_sha256": identity.judge_config_sha256,
        "prompt_identity": _identity_record(bound.prompt_identity),
        "service_identity": _identity_record(bound.service_identity),
        "model_identity": _identity_record(bound.model_identity),
        "sampling_identity": _identity_record(bound.sampling_identity),
        "calibration_identity": _identity_record(bound.calibration_identity),
        "failure_policy_identity": _identity_record(bound.failure_policy_identity),
        "verdict": verdict,
    }
    for field, value in expected_evidence.items():
        if evidence.get(field) != value:
            raise ValueError(f"legacy semantic judge evidence {field} differs")

    compaction = evidence.get("judge_input_compaction", [])
    if not isinstance(compaction, list):
        raise ValueError("legacy semantic judge input compaction is invalid")
    # Reconstructing a compacted payload requires trusting a historical retry
    # algorithm.  Rejudging the tiny compacted tail is safer than transporting
    # it, so it deliberately does not enter the reusable set.
    if compaction:
        return dict(index), {**dict(evidence), "_reuse_skip": "input_compaction"}

    expected_request_payload = _request_payload(request, bound)
    if evidence.get("request_payload") != expected_request_payload:
        raise ValueError("legacy semantic judge HTTP request payload differs")
    request_payload_sha256 = _sha256_bytes(
        _canonical_json_bytes(expected_request_payload)
    )
    if evidence.get("request_payload_sha256") != request_payload_sha256:
        raise ValueError("legacy semantic judge HTTP request SHA-256 differs")

    response = evidence.get("response")
    response_json_sha256 = _sha256_bytes(_canonical_json_bytes(response))
    if evidence.get("response_json_sha256") != response_json_sha256:
        raise ValueError("legacy semantic judge response JSON SHA-256 differs")
    _require_sha256(
        evidence.get("raw_response_bytes_sha256"),
        name="legacy semantic judge raw response identity",
    )
    parsed_verdict, rationale, usage = _strict_response(
        response, expected_model=identity.model_served_name
    )
    choices = response["choices"]
    expected_response_fields = {
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "finish_reason": choices[0]["finish_reason"],
        "usage": usage,
        "verdict": parsed_verdict,
        "rationale": rationale,
    }
    for field, value in expected_response_fields.items():
        if evidence.get(field) != value:
            raise ValueError(f"legacy semantic judge response {field} differs")
    return dict(index), dict(evidence)


def _publish_rebound_evidence(
    target_judge_root: Path,
    target_request: Mapping[str, Any],
    *,
    identity: _RunIdentity,
    source_scoring_manifest_sha256: str,
    target_scoring_manifest_sha256: str,
    source_index: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    evidence_identity = {
        key: value for key, value in source_evidence.items() if key != "evidence_sha256"
    }
    evidence_identity.update(
        {
            "schema_version": T1_JUDGE_EVIDENCE_SCHEMA,
            "scoring_manifest_sha256": target_scoring_manifest_sha256,
            "consumer_count": target_request["consumer_count"],
            "reuse_provenance": {
                "schema_version": T1_JUDGE_REUSE_PROVENANCE_SCHEMA,
                "source_scoring_manifest_schema": (
                    T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA
                ),
                "source_scoring_manifest_sha256": (source_scoring_manifest_sha256),
                "source_judge_index_schema": T1_LEGACY_JUDGE_INDEX_SCHEMA,
                "source_judge_index_sha256": source_index["index_sha256"],
                "source_judge_evidence_schema": T1_LEGACY_JUDGE_EVIDENCE_SCHEMA,
                "source_judge_evidence_sha256": source_evidence["evidence_sha256"],
                "reuse_rule": "exact_semantic_payload_and_judge_binding_only",
            },
        }
    )
    evidence_sha256 = _sha256_bytes(_canonical_json_bytes(evidence_identity))
    evidence = {**evidence_identity, "evidence_sha256": evidence_sha256}
    evidence_payload = _canonical_json_bytes(evidence) + b"\n"
    evidence_relative = (
        Path("evidence") / evidence_sha256[:2] / f"{evidence_sha256}.json"
    )
    index_identity = {
        "schema_version": T1_JUDGE_INDEX_SCHEMA,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "scoring_manifest_sha256": target_scoring_manifest_sha256,
        "judge_request_id": target_request["judge_request_id"],
        "payload_sha256": target_request["payload_sha256"],
        "evidence_sha256": evidence_sha256,
        "evidence_file": evidence_relative.as_posix(),
        "evidence_file_sha256": _sha256_bytes(evidence_payload),
        "verdict": source_evidence["verdict"],
    }
    index_sha256 = _sha256_bytes(_canonical_json_bytes(index_identity))
    index = {**index_identity, "index_sha256": index_sha256}
    if not dry_run:
        _atomic_write_immutable(target_judge_root / evidence_relative, evidence_payload)
        _atomic_write_immutable(
            _index_path(target_judge_root, target_request),
            _canonical_json_bytes(index) + b"\n",
        )
    return index


def _reuse_exact_request(
    source_request: Mapping[str, Any],
    target_request: Mapping[str, Any],
    *,
    identity: _RunIdentity,
    bound: BoundOpenAICompatibleJudge,
    source_judge_root: Path,
    target_judge_root: Path,
    source_scoring_manifest_sha256: str,
    target_scoring_manifest_sha256: str,
    dry_run: bool,
) -> _ReuseOutcome:
    """Validate and optionally publish one exact-intersection request."""

    if source_request.get("payload_sha256") != target_request.get(
        "payload_sha256"
    ) or _payload(source_request) != _payload(target_request):
        raise ValueError("semantic judge reuse worker received a non-exact payload")
    existing = _load_completed_index(
        target_judge_root,
        target_request,
        run_manifest_sha256=identity.run_manifest_sha256,
        scoring_manifest_sha256=target_scoring_manifest_sha256,
    )
    if existing is not None:
        return _ReuseOutcome("target_already_complete")
    legacy = _validated_legacy_evidence(
        source_judge_root,
        source_request,
        identity=identity,
        source_scoring_manifest_sha256=source_scoring_manifest_sha256,
        bound=bound,
    )
    if legacy is None:
        return _ReuseOutcome("legacy_index_missing")
    source_index, source_evidence = legacy
    if source_evidence.get("_reuse_skip") == "input_compaction":
        return _ReuseOutcome("legacy_input_compaction_skipped")
    _publish_rebound_evidence(
        target_judge_root,
        target_request,
        identity=identity,
        source_scoring_manifest_sha256=source_scoring_manifest_sha256,
        target_scoring_manifest_sha256=target_scoring_manifest_sha256,
        source_index=source_index,
        source_evidence=source_evidence,
        dry_run=dry_run,
    )
    return _ReuseOutcome("reusable", int(source_evidence["verdict"]))


def _next_or_none(iterator: Iterator[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _reuse_legacy_judge_results(
    *,
    identity: _RunIdentity,
    bound: BoundOpenAICompatibleJudge,
    source_scoring_root: Path,
    source_judge_root: Path,
    target_scoring_root: Path,
    target_judge_root: Path,
    dry_run: bool = False,
    workers: int = 32,
    progress_every: int = 10_000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate both queues and republish only their exact payload intersection."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be bool")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("judge reuse workers must be in [1, 64]")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    source = _load_queue_binding(
        source_scoring_root,
        expected_schema=T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=identity,
    )
    target = _load_queue_binding(
        target_scoring_root,
        expected_schema=T1_DETERMINISTIC_SCORING_MANIFEST_SCHEMA,
        identity=identity,
        expected_task_kind_policy=POLICY_SELECTION_TASK_KIND_POLICY,
    )
    if source.quality_exclusions_sha256 != target.quality_exclusions_sha256:
        raise ValueError("source and target quality-exclusion identities differ")

    # Complete preflight happens before the first durable write.  The queues
    # are validated a second time while merging so a concurrent mutation also
    # fails closed.
    _fully_validate_queue(source, identity=identity)
    _fully_validate_queue(target, identity=identity)

    source_iterator = _iter_validated_queue(source, identity=identity)
    target_iterator = _iter_validated_queue(target, identity=identity)
    source_request = _next_or_none(source_iterator)
    target_request = _next_or_none(target_iterator)
    counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    pending: set[Future[_ReuseOutcome]] = set()
    maximum_in_flight = workers * 2
    next_progress = progress_every

    def collect(completed: set[Future[_ReuseOutcome]]) -> None:
        nonlocal next_progress
        for future in completed:
            outcome = future.result()
            counts["processed_unchanged"] += 1
            counts[outcome.status] += 1
            if outcome.status == "reusable":
                if outcome.verdict not in {0, 1}:
                    raise ValueError("judge reuse worker returned an invalid verdict")
                if not dry_run:
                    counts["written"] += 1
                verdict_counts[str(outcome.verdict)] += 1
        while counts["processed_unchanged"] >= next_progress:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "t1_judge_reuse_progress",
                        "unchanged_payload_submitted": counts["unchanged_payload"],
                        "unchanged_payload_completed": counts["processed_unchanged"],
                        "in_flight": len(pending),
                        "reusable": counts["reusable"],
                        "written": counts["written"],
                        "target_already_complete": counts["target_already_complete"],
                    }
                )
            next_progress += progress_every

    executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="t1-judge-reuse"
    )
    try:
        while source_request is not None or target_request is not None:
            source_sha = (
                str(source_request["payload_sha256"])
                if source_request is not None
                else None
            )
            target_sha = (
                str(target_request["payload_sha256"])
                if target_request is not None
                else None
            )
            if target_request is None or (
                source_request is not None and str(source_sha) < str(target_sha)
            ):
                counts["source_only"] += 1
                source_request = _next_or_none(source_iterator)
                continue
            if source_request is None or str(target_sha) < str(source_sha):
                counts["target_only"] += 1
                target_request = _next_or_none(target_iterator)
                continue

            assert source_request is not None and target_request is not None
            counts["unchanged_payload"] += 1
            if _payload(source_request) != _payload(target_request):
                raise ValueError("semantic judge payload SHA-256 collision")
            pending.add(
                executor.submit(
                    _reuse_exact_request,
                    source_request,
                    target_request,
                    identity=identity,
                    bound=bound,
                    source_judge_root=source_judge_root,
                    target_judge_root=target_judge_root,
                    source_scoring_manifest_sha256=source.manifest_sha256,
                    target_scoring_manifest_sha256=target.manifest_sha256,
                    dry_run=dry_run,
                )
            )
            source_request = _next_or_none(source_iterator)
            target_request = _next_or_none(target_iterator)
            if len(pending) >= maximum_in_flight:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(completed)

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(completed)
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if counts["processed_unchanged"] != counts["unchanged_payload"]:
        raise ValueError("judge reuse worker completion count differs")

    return {
        "schema_version": "tgvf.policy-selection.t1-judge-reuse-report.v1",
        "run_id": identity.run_id,
        "dry_run": dry_run,
        "workers": workers,
        "maximum_in_flight": maximum_in_flight,
        "source_scoring_manifest_sha256": source.manifest_sha256,
        "target_scoring_manifest_sha256": target.manifest_sha256,
        "source_request_count": source.rows,
        "target_request_count": target.rows,
        "unchanged_payload_count": counts["unchanged_payload"],
        "source_only_count": counts["source_only"],
        "target_only_count": counts["target_only"],
        "target_already_complete_count": counts["target_already_complete"],
        "legacy_index_missing_count": counts["legacy_index_missing"],
        "legacy_input_compaction_skipped_count": counts[
            "legacy_input_compaction_skipped"
        ],
        "reusable_count": counts["reusable"],
        "records_written": counts["written"],
        "reused_verdict_counts": dict(sorted(verdict_counts.items())),
    }


def reuse_t1_legacy_judge_results(
    config_path: str | Path,
    *,
    judge_config_path: str | Path,
    dry_run: bool = False,
    workers: int = 32,
    progress_every: int = 10_000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reuse validated judge-v1 results unchanged in the v3 scoring queue.

    No judge manifest is published here.  After this migration, the ordinary
    streaming judge handles target-only requests and the canonical judge
    runner validates every target index/evidence before publishing its v2
    manifest.
    """

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=False)
    _validate_prepared_output_root(run, path)
    expected_config_sha256 = str(run.verifier["semantic_judge"]["config_sha256"])
    bound = load_openai_compatible_judge(
        Path(judge_config_path).resolve(),
        expected_file_sha256=expected_config_sha256,
    )
    identity = _RunIdentity(
        run_id=run.run_id,
        run_manifest_sha256=run.manifest_sha256,
        selection_candidates_sha256=str(run.selection["candidates_sha256"]),
        judge_config_sha256=expected_config_sha256,
        prompt_sha256=str(run.verifier["semantic_judge"]["prompt_sha256"]),
        model_repository=str(run.verifier["semantic_judge"]["repository"]),
        model_served_name=str(run.verifier["semantic_judge"]["served_name"]),
    )
    if bound.provider.config.model_name != identity.model_served_name:
        raise ValueError("semantic judge served-model identity differs")
    target_root = run.output_root / T1_JUDGE_DIRECTORY
    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
    return _reuse_legacy_judge_results(
        identity=identity,
        bound=bound,
        source_scoring_root=run.output_root / T1_LEGACY_SCORING_DIRECTORY,
        source_judge_root=run.output_root / T1_LEGACY_JUDGE_DIRECTORY,
        target_scoring_root=run.output_root / T1_SCORING_DIRECTORY,
        target_judge_root=target_root,
        dry_run=dry_run,
        workers=workers,
        progress_every=progress_every,
        progress_callback=progress_callback,
    )


__all__ = [
    "T1_JUDGE_REUSE_PROVENANCE_SCHEMA",
    "T1_LEGACY_DETERMINISTIC_SCORING_MANIFEST_SCHEMA",
    "T1_LEGACY_JUDGE_DIRECTORY",
    "T1_LEGACY_JUDGE_EVIDENCE_SCHEMA",
    "T1_LEGACY_JUDGE_INDEX_SCHEMA",
    "T1_LEGACY_SCORING_DIRECTORY",
    "reuse_t1_legacy_judge_results",
]
