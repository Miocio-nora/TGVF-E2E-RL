"""Independent retained dataset materialization for the teacher-only T1 run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .policy_selection import (
    AttemptStatus,
    POLICY_SELECTION_DECISION_SCHEMA,
    SelectionCandidate,
    SelectionAttempt,
    SelectionBranch,
    SelectionSource,
    T1Decision,
    T2Decision,
    canonical_json_line,
    stable_selection_request_id,
)
from .policy_selection_runtime import T1_ATTEMPTS, load_t1_run_config
from .policy_selection_vllm import load_t1_candidates


POLICY_TEACHER_T1_RETAINED_SAMPLE_SCHEMA = "tgvf.policy-teacher-t1-retained.sample.v1"
POLICY_TEACHER_T1_RETAINED_MANIFEST_SCHEMA = (
    "tgvf.policy-teacher-t1-retained.manifest.v1"
)
T1_FINAL_SCORING_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-final-scoring-manifest.v2"

_FINAL_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "run_manifest_sha256",
    "scoring_manifest_sha256",
    "judge_manifest_sha256",
    "files",
    "manifest_sha256",
}
_FINAL_FILE_FIELDS = {
    "attempts": {"path", "rows", "sha256"},
    "decisions": {"path", "rows", "sha256"},
    "report": {"path", "sha256"},
}
_FINAL_FILE_PATHS = {
    "attempts": "attempts.jsonl",
    "decisions": "decisions.jsonl",
    "report": "report.json",
}
_DECISION_FIELDS = {
    "schema_version",
    "candidate_sha256",
    "sample_id",
    "source",
    "t1",
    "t2",
}
_T1_FIELDS = {"decision", "full_image", "reason"}
_T2_FIELDS = {"decision", "gt_region", "reason"}
_FULL_IMAGE_FIELDS = {
    "accuracy",
    "complete",
    "correct_count",
    "expected_attempts",
    "missing_indices",
    "observed_attempts",
    "scoreable_attempts",
    "status_counts",
}
_ATTEMPT_REQUIRED_FIELDS = {
    "schema_version",
    "request_id",
    "sample_id",
    "candidate_sha256",
    "source",
    "branch",
    "attempt_index",
    "status",
    "correct",
    "run_id",
    "run_manifest_sha256",
    "raw_generation_sha256",
    "budget_revision",
}


@dataclass(frozen=True, slots=True)
class TeacherT1RetainedResult:
    output_root: Path
    candidate_count: int
    retained_count: int
    unique_images: int
    samples_sha256: str
    manifest_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "candidate_count": self.candidate_count,
            "retained_count": self.retained_count,
            "unique_images": self.unique_images,
            "samples_sha256": self.samples_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json(payload: bytes, *, field: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{field} contains non-finite JSON number: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from error


def _jsonl_records(path: Path, *, field: str) -> Iterator[dict[str, Any]]:
    observed = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{field} line {line_number} is blank")
            value = _parse_json(raw_line, field=f"{field} line {line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{field} line {line_number} must be an object")
            if raw_line != canonical_json_line(value):
                raise ValueError(f"{field} line {line_number} is not canonical JSON")
            observed += 1
            yield value
    if observed == 0:
        raise ValueError(f"{field} must not be empty")


def _absolute_regular_file(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be an absolute regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{field} path must be normalized without symlink ancestors")
    return resolved


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise ValueError(f"{field} must be a safe normalized relative path")
    return path


def _load_final_manifest(
    path: Path,
    *,
    run_id: str,
    run_manifest_sha256: str,
    candidate_count: int,
) -> tuple[dict[str, Any], Path, Path, bytes]:
    path = _absolute_regular_file(path, field="teacher T1 final-scoring manifest")
    payload = path.read_bytes()
    value = _parse_json(payload, field="teacher T1 final-scoring manifest")
    if not isinstance(value, dict) or set(value) != _FINAL_MANIFEST_FIELDS:
        raise ValueError("teacher T1 final-scoring manifest fields differ")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise ValueError("teacher T1 final-scoring manifest is not canonical JSON")
    identity = dict(value)
    manifest_sha256 = _required_sha256(
        identity.pop("manifest_sha256"), field="final manifest identity"
    )
    if manifest_sha256 != _sha256_bytes(_canonical_json_bytes(identity)):
        raise ValueError("teacher T1 final-scoring manifest identity differs")
    for field in (
        "run_manifest_sha256",
        "scoring_manifest_sha256",
        "judge_manifest_sha256",
    ):
        _required_sha256(value[field], field=f"final manifest {field}")
    if (
        value["schema_version"] != T1_FINAL_SCORING_MANIFEST_SCHEMA
        or value["run_id"] != run_id
        or value["run_manifest_sha256"] != run_manifest_sha256
    ):
        raise ValueError("teacher T1 final-scoring run identity differs")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_FINAL_FILE_FIELDS):
        raise ValueError("teacher T1 final-scoring file set differs")
    for name, expected_fields in _FINAL_FILE_FIELDS.items():
        record = files.get(name)
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError(f"teacher T1 final-scoring {name} fields differ")
        if record["path"] != _FINAL_FILE_PATHS[name]:
            raise ValueError(f"teacher T1 final-scoring {name} path differs")
        relative = _safe_relative_path(record["path"], field=f"files.{name}.path")
        file_path = path.parent / relative
        file_path = _absolute_regular_file(
            file_path, field=f"teacher T1 final-scoring {name}"
        )
        expected_sha256 = _required_sha256(
            record["sha256"], field=f"final manifest files.{name}.sha256"
        )
        if _sha256_file(file_path) != expected_sha256:
            raise ValueError(f"teacher T1 final-scoring {name} SHA-256 differs")
        if name != "report":
            if type(record["rows"]) is not int or record["rows"] <= 0:
                raise ValueError(
                    f"teacher T1 final-scoring {name} rows must be positive"
                )
            with file_path.open("rb") as handle:
                rows = sum(bool(line.strip()) for line in handle)
            if rows != record["rows"]:
                raise ValueError(f"teacher T1 final-scoring {name} row count differs")
    if files["decisions"]["rows"] != candidate_count:
        raise ValueError("teacher T1 final decision coverage differs")
    if files["attempts"]["rows"] != candidate_count * T1_ATTEMPTS:
        raise ValueError("teacher T1 final attempt coverage differs")
    attempts_path = path.parent / _safe_relative_path(
        files["attempts"]["path"], field="files.attempts.path"
    )
    decisions_path = path.parent / _safe_relative_path(
        files["decisions"]["path"], field="files.decisions.path"
    )
    return value, attempts_path, decisions_path, payload


def _validate_t1_record(value: Any) -> T1Decision:
    if not isinstance(value, Mapping) or set(value) != _T1_FIELDS:
        raise ValueError("teacher T1 decision t1 fields differ")
    summary = value.get("full_image")
    if not isinstance(summary, Mapping) or set(summary) != _FULL_IMAGE_FIELDS:
        raise ValueError("teacher T1 decision full-image fields differ")
    if summary.get("expected_attempts") != T1_ATTEMPTS:
        raise ValueError("teacher T1 decision expected attempt count differs")
    observed = summary.get("observed_attempts")
    scoreable = summary.get("scoreable_attempts")
    correct = summary.get("correct_count")
    if (
        type(observed) is not int
        or not 0 <= observed <= T1_ATTEMPTS
        or type(scoreable) is not int
        or not 0 <= scoreable <= observed
        or type(correct) is not int
        or not 0 <= correct <= scoreable
    ):
        raise ValueError("teacher T1 decision summary counts are invalid")
    missing = summary.get("missing_indices")
    if (
        not isinstance(missing, list)
        or any(
            type(index) is not int or not 0 <= index < T1_ATTEMPTS for index in missing
        )
        or missing != sorted(set(missing))
    ):
        raise ValueError("teacher T1 decision missing attempt indices are invalid")
    status_counts = summary.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise ValueError("teacher T1 decision status counts are invalid")
    allowed_statuses = {status.value for status in AttemptStatus}
    normalized_status_counts: dict[str, int] = {}
    for status, count in status_counts.items():
        if status not in allowed_statuses or type(count) is not int or count <= 0:
            raise ValueError("teacher T1 decision status counts are invalid")
        normalized_status_counts[str(status)] = count
    if (
        sum(normalized_status_counts.values()) != observed
        or normalized_status_counts.get(AttemptStatus.SCORED.value, 0) != scoreable
    ):
        raise ValueError("teacher T1 decision status coverage differs")
    complete = not missing and scoreable == T1_ATTEMPTS
    if summary.get("complete") is not complete:
        raise ValueError("teacher T1 decision complete flag differs")
    expected_accuracy = correct / T1_ATTEMPTS if complete else None
    accuracy = summary.get("accuracy")
    if (expected_accuracy is None and accuracy is not None) or (
        expected_accuracy is not None
        and (type(accuracy) not in {int, float} or accuracy != expected_accuracy)
    ):
        raise ValueError("teacher T1 decision accuracy differs")

    if not complete:
        expected_decision = T1Decision.UNRESOLVED
        expected_reason = "requires_exactly_eight_scoreable_full_image_attempts"
    elif correct == 0:
        expected_decision = T1Decision.EXCLUDE_TOO_HARD
        expected_reason = "zero_of_eight_correct"
    elif correct == T1_ATTEMPTS:
        expected_decision = T1Decision.EXCLUDE_TOO_EASY
        expected_reason = "eight_of_eight_correct"
    else:
        expected_decision = T1Decision.RETAIN
        expected_reason = "between_one_and_seven_of_eight_correct"
    try:
        decision = T1Decision(value.get("decision"))
    except (TypeError, ValueError) as error:
        raise ValueError("teacher T1 decision value is invalid") from error
    if decision is T1Decision.RETAIN and not 1 <= correct <= 7:
        raise ValueError("retained teacher T1 row violates the 1--7/8 rule")
    if decision is not expected_decision or value.get("reason") != expected_reason:
        raise ValueError("teacher T1 decision and reason differ from its summary")
    return decision


def _validate_teacher_t2_record(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _T2_FIELDS:
        raise ValueError("teacher T1 decision t2 fields differ")
    if (
        value.get("decision") != T2Decision.NOT_APPLICABLE_PRESERVE_T1.value
        or value.get("reason") != "deepeyes_perception_utility_is_vstar_only"
        or value.get("gt_region") is not None
    ):
        raise ValueError("teacher T1 decision t2 contract differs")


def _load_decisions(
    path: Path,
    *,
    candidates: Mapping[str, SelectionCandidate],
) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    decisions: dict[str, Mapping[str, Any]] = {}
    counts: Counter[str] = Counter()
    for value in _jsonl_records(path, field="teacher T1 decisions"):
        if (
            set(value) != _DECISION_FIELDS
            or value.get("schema_version") != POLICY_SELECTION_DECISION_SCHEMA
        ):
            raise ValueError("teacher T1 decision schema differs")
        sample_id = value.get("sample_id")
        candidate = candidates.get(sample_id)
        if candidate is None:
            raise ValueError("teacher T1 decision refers to an unknown sample")
        if sample_id in decisions:
            raise ValueError("duplicate teacher T1 decision")
        candidate_sha256 = _required_sha256(
            value.get("candidate_sha256"), field="decision candidate_sha256"
        )
        if (
            candidate_sha256 != candidate.identity_sha256
            or value.get("source") != SelectionSource.TEACHER.value
            or candidate.source is not SelectionSource.TEACHER
        ):
            raise ValueError("teacher T1 decision candidate identity differs")
        decision = _validate_t1_record(value.get("t1"))
        _validate_teacher_t2_record(value.get("t2"))
        counts[decision.value] += 1
        decisions[str(sample_id)] = value
    if set(decisions) != set(candidates):
        raise ValueError(
            "teacher T1 decisions do not cover every candidate exactly once"
        )
    return decisions, counts


def _load_attempts(
    path: Path,
    *,
    candidates: Mapping[str, SelectionCandidate],
    run_id: str,
    run_manifest_sha256: str,
) -> dict[str, tuple[SelectionAttempt, ...]]:
    grouped: dict[str, dict[int, SelectionAttempt]] = {
        sample_id: {} for sample_id in candidates
    }
    request_ids: set[str] = set()
    for value in _jsonl_records(path, field="teacher T1 attempts"):
        if not _ATTEMPT_REQUIRED_FIELDS.issubset(value):
            raise ValueError("teacher T1 attempt required fields differ")
        attempt = SelectionAttempt.from_record(value)
        candidate = candidates.get(attempt.sample_id)
        if candidate is None:
            raise ValueError("teacher T1 attempt refers to an unknown sample")
        if (
            candidate.source is not SelectionSource.TEACHER
            or attempt.source is not SelectionSource.TEACHER
            or attempt.branch is not SelectionBranch.FULL_IMAGE
            or value.get("candidate_sha256") != candidate.identity_sha256
            or value.get("run_id") != run_id
            or value.get("run_manifest_sha256") != run_manifest_sha256
        ):
            raise ValueError("teacher T1 attempt identity differs")
        _required_sha256(
            value.get("raw_generation_sha256"), field="attempt raw_generation_sha256"
        )
        budget_revision = value.get("budget_revision")
        if type(budget_revision) is not int or not 0 <= budget_revision <= 2:
            raise ValueError("teacher T1 attempt budget revision is invalid")
        if not 0 <= attempt.attempt_index < T1_ATTEMPTS:
            raise ValueError("teacher T1 attempt index is outside 0--7")
        expected_request_id = stable_selection_request_id(
            candidate_sha256=candidate.identity_sha256,
            branch=SelectionBranch.FULL_IMAGE,
            attempt_index=attempt.attempt_index,
        )
        if attempt.request_id != expected_request_id:
            raise ValueError("teacher T1 attempt request identity differs")
        if attempt.request_id in request_ids:
            raise ValueError("duplicate teacher T1 attempt request ID")
        request_ids.add(attempt.request_id)
        by_index = grouped[attempt.sample_id]
        if attempt.attempt_index in by_index:
            raise ValueError("duplicate teacher T1 attempt index")
        by_index[attempt.attempt_index] = attempt

    expected_indices = set(range(T1_ATTEMPTS))
    complete: dict[str, tuple[SelectionAttempt, ...]] = {}
    for sample_id, by_index in grouped.items():
        if set(by_index) != expected_indices:
            raise ValueError(
                f"teacher T1 attempts do not cover indices 0--7 for {sample_id!r}"
            )
        complete[sample_id] = tuple(by_index[index] for index in range(T1_ATTEMPTS))
    return complete


def _attempt_summary(attempts: tuple[SelectionAttempt, ...]) -> dict[str, Any]:
    status_counts = Counter(attempt.status.value for attempt in attempts)
    scored = [attempt for attempt in attempts if attempt.status is AttemptStatus.SCORED]
    correct_count = sum(attempt.correct is True for attempt in scored)
    missing_indices = sorted(
        set(range(T1_ATTEMPTS)) - {attempt.attempt_index for attempt in attempts}
    )
    complete = not missing_indices and len(scored) == T1_ATTEMPTS
    return {
        "expected_attempts": T1_ATTEMPTS,
        "observed_attempts": len(attempts),
        "scoreable_attempts": len(scored),
        "correct_count": correct_count,
        "accuracy": correct_count / T1_ATTEMPTS if complete else None,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_indices": missing_indices,
        "complete": complete,
    }


def _validate_decisions_against_attempts(
    decisions: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[str, tuple[SelectionAttempt, ...]],
) -> None:
    if set(decisions) != set(attempts):
        raise ValueError("teacher T1 decision and attempt populations differ")
    for sample_id, decision in decisions.items():
        expected = _attempt_summary(attempts[sample_id])
        t1 = decision["t1"]
        if not isinstance(t1, Mapping) or t1.get("full_image") != expected:
            raise ValueError(
                f"teacher T1 decision summary differs from attempts for {sample_id!r}"
            )


def _shuffle_key(*, seed: int, candidate_sha256: str) -> str:
    return hashlib.sha256(
        b"tgvf-policy-teacher-t1-retained-shuffle-v1\0"
        + str(seed).encode("ascii")
        + b"\0"
        + candidate_sha256.encode("ascii")
    ).hexdigest()


def _teacher_candidate_metadata(
    candidate: SelectionCandidate,
) -> tuple[str, str, Mapping[str, Any]]:
    if candidate.source is not SelectionSource.TEACHER:
        raise ValueError("teacher retained candidate source differs")
    if (
        not isinstance(candidate.ground_truth, str)
        or not candidate.ground_truth.strip()
    ):
        raise ValueError("teacher retained ground_truth must be non-empty text")
    metadata = candidate.canonical_record.get("selection_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("teacher retained candidate has no selection_metadata")
    task_kind = metadata.get("task_kind")
    if task_kind == "open":
        if set(metadata) != {"task_kind", "answer_format"}:
            raise ValueError("teacher retained open metadata fields differ")
        if metadata.get("answer_format") != "open":
            raise ValueError("teacher retained open answer_format differs")
    elif task_kind == "mcq":
        if set(metadata) != {
            "task_kind",
            "answer_format",
            "option_count",
            "choices",
            "answer_text",
        }:
            raise ValueError("teacher retained MCQ metadata fields differ")
        if metadata.get("answer_format") != "multiple_choice":
            raise ValueError("teacher retained MCQ answer_format differs")
        option_count = metadata.get("option_count")
        choices = metadata.get("choices")
        if (
            type(option_count) is not int
            or not 2 <= option_count <= 26
            or not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or len(choices) != option_count
        ):
            raise ValueError("teacher retained MCQ option metadata differs")
        normalized_choices: list[str] = []
        for index, choice in enumerate(choices):
            expected_prefix = f"{chr(ord('A') + index)}. "
            if (
                not isinstance(choice, str)
                or not choice.startswith(expected_prefix)
                or not choice[len(expected_prefix) :].strip()
            ):
                raise ValueError("teacher retained MCQ choices differ")
            normalized_choices.append(choice)
        expected_suffix = "Choices:\n" + "\n".join(normalized_choices)
        if not candidate.question.endswith(expected_suffix):
            raise ValueError("teacher retained MCQ prompt omits canonical choices")
        expected_labels = {chr(ord("A") + index) for index in range(option_count)}
        if candidate.ground_truth not in expected_labels:
            raise ValueError("teacher retained MCQ ground_truth is out of range")
        answer_text = metadata.get("answer_text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError("teacher retained MCQ answer_text is invalid")
    else:
        raise ValueError("teacher retained candidate task_kind is invalid")
    source_dataset = candidate.provenance.get("source_dataset")
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("teacher retained candidate source_dataset is invalid")
    return str(task_kind), source_dataset, metadata


def _verify_retained_images(candidates: list[SelectionCandidate]) -> int:
    from PIL import Image

    expected_by_path: dict[Path, tuple[str, int, int]] = {}
    for candidate in candidates:
        image = candidate.image
        path = Path(str(image["path"]))
        if not path.is_absolute() or path.resolve(strict=False) != path:
            raise ValueError(
                "teacher retained image path must be absolute and normalized"
            )
        expected = (str(image["sha256"]), int(image["width"]), int(image["height"]))
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise ValueError("teacher retained image path has conflicting identities")
    for path, (sha256, width, height) in expected_by_path.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"teacher retained image is not a regular file: {path}")
        if _sha256_file(path) != sha256:
            raise ValueError(f"teacher retained image SHA-256 differs: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError(f"teacher retained image dimensions differ: {path}")
    return len(expected_by_path)


def materialize_teacher_t1_retained(
    config_path: str | Path,
    final_manifest_path: str | Path,
    output_root: str | Path,
    *,
    shuffle_seed: int = 42,
) -> TeacherT1RetainedResult:
    """Join final T1 decisions to teacher candidates and publish only retained rows."""

    config = Path(config_path).resolve()
    run = load_t1_run_config(config, verify_data_files=True)
    if run.selection["kind"] != "teacher_full" or {
        source.source for source in run.data_sources
    } != {SelectionSource.TEACHER}:
        raise ValueError("teacher retained materializer accepts only teacher_full runs")
    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a non-negative integer")
    candidates_sequence = load_t1_candidates(run)
    candidates = {candidate.sample_id: candidate for candidate in candidates_sequence}
    if len(candidates) != len(candidates_sequence):
        raise ValueError("teacher T1 candidates repeat sample IDs")
    candidate_metadata = {
        candidate.sample_id: _teacher_candidate_metadata(candidate)
        for candidate in candidates_sequence
    }
    final_path = Path(final_manifest_path).resolve()
    final_manifest, attempts_path, decisions_path, final_payload = _load_final_manifest(
        final_path,
        run_id=run.run_id,
        run_manifest_sha256=run.manifest_sha256,
        candidate_count=len(candidates),
    )
    decisions, decision_counts = _load_decisions(decisions_path, candidates=candidates)
    attempts = _load_attempts(
        attempts_path,
        candidates=candidates,
        run_id=run.run_id,
        run_manifest_sha256=run.manifest_sha256,
    )
    _validate_decisions_against_attempts(decisions, attempts)
    retained = [
        candidate
        for candidate in candidates_sequence
        if decisions[candidate.sample_id]["t1"]["decision"] == T1Decision.RETAIN.value
    ]
    if not retained:
        raise ValueError("teacher T1 retained no rows")
    retained.sort(
        key=lambda candidate: (
            _shuffle_key(seed=shuffle_seed, candidate_sha256=candidate.identity_sha256),
            candidate.sample_id,
        )
    )
    unique_images = _verify_retained_images(retained)
    task_counts: Counter[str] = Counter()
    source_dataset_counts: Counter[str] = Counter()
    sample_records: list[dict[str, Any]] = []
    for candidate in retained:
        task_kind, source_dataset, metadata = candidate_metadata[candidate.sample_id]
        task_counts[task_kind] += 1
        source_dataset_counts[source_dataset] += 1
        extra_info: dict[str, Any] = {
            "question": candidate.question,
            "source_dataset": source_dataset,
            "answer_format": metadata.get("answer_format"),
        }
        if task_kind == "mcq":
            extra_info["choices"] = metadata.get("choices")
            extra_info["option_count"] = metadata.get("option_count")
        sample_records.append(
            {
                "schema_version": POLICY_TEACHER_T1_RETAINED_SAMPLE_SCHEMA,
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "decision_sha256": _sha256_bytes(
                    _canonical_json_bytes(decisions[candidate.sample_id])
                ),
                "data_source": SelectionSource.TEACHER.value,
                "image": dict(candidate.image),
                "extra_info": extra_info,
                "reward_model": {"ground_truth": candidate.ground_truth},
                "provenance": dict(candidate.provenance),
                "selection": {
                    "decision_stage": "final",
                    "t1": dict(decisions[candidate.sample_id]["t1"]),
                },
            }
        )

    root = Path(output_root).resolve()
    if os.path.lexists(root):
        raise FileExistsError(f"teacher retained output root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        samples_payload = b"".join(canonical_json_line(row) for row in sample_records)
        samples_sha256 = _sha256_bytes(samples_payload)
        (temporary_root / "samples.jsonl").write_bytes(samples_payload)
        manifest = {
            "schema_version": POLICY_TEACHER_T1_RETAINED_MANIFEST_SCHEMA,
            "dataset_kind": "policy_teacher_t1_retained",
            "run_id": run.run_id,
            "run_manifest_sha256": run.manifest_sha256,
            "inputs": {
                "config": {
                    "path": str(config),
                    "sha256": _sha256_file(config),
                },
                "candidates": {
                    "path": str(run.selection["candidates_path"]),
                    "sha256": str(run.selection["candidates_sha256"]),
                    "rows": len(candidates),
                },
                "attempts": {
                    "path": str(attempts_path),
                    "sha256": str(final_manifest["files"]["attempts"]["sha256"]),
                    "rows": len(candidates) * T1_ATTEMPTS,
                    "coverage": "each-candidate-full-image-attempt-indices-0-through-7",
                },
                "decisions": {
                    "path": str(decisions_path),
                    "sha256": str(final_manifest["files"]["decisions"]["sha256"]),
                    "rows": len(decisions),
                },
                "final_scoring_manifest": {
                    "path": str(final_path),
                    "file_sha256": _sha256_bytes(final_payload),
                    "manifest_sha256": final_manifest["manifest_sha256"],
                    "scoring_manifest_sha256": final_manifest[
                        "scoring_manifest_sha256"
                    ],
                    "judge_manifest_sha256": final_manifest["judge_manifest_sha256"],
                },
            },
            "candidate_count": len(candidates),
            "decision_count": len(decisions),
            "retained_count": len(retained),
            "t1_decision_counts": {
                decision.value: decision_counts[decision.value]
                for decision in T1Decision
            },
            "task_kind_counts": {
                task_kind: task_counts[task_kind] for task_kind in ("mcq", "open")
            },
            "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
            "selection_policy": {
                "t1": "retain",
                "t2": "ignored",
                "post_t1_balancing": "none",
            },
            "shuffle": {
                "algorithm": "sha256-sort-v1",
                "seed": shuffle_seed,
            },
            "images": {
                "address": "absolute-path-plus-sha256",
                "bytes_verified": True,
                "unique_paths_verified": unique_images,
            },
            "samples": {
                "path": "samples.jsonl",
                "rows": len(sample_records),
                "sha256": samples_sha256,
            },
        }
        descriptor = dict(manifest)
        manifest["content_sha256"] = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest_payload = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / "manifest.json").write_bytes(manifest_payload)
        os.replace(temporary_root, root)
        return TeacherT1RetainedResult(
            output_root=root,
            candidate_count=len(candidates),
            retained_count=len(retained),
            unique_images=unique_images,
            samples_sha256=samples_sha256,
            manifest_sha256=_sha256_bytes(manifest_payload),
        )
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


__all__ = [
    "POLICY_TEACHER_T1_RETAINED_MANIFEST_SCHEMA",
    "POLICY_TEACHER_T1_RETAINED_SAMPLE_SCHEMA",
    "TeacherT1RetainedResult",
    "materialize_teacher_t1_retained",
]
