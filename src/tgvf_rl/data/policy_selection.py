"""CPU-only contracts for Qwen3-native Policy RL data selection.

DeepEyes is the hard methodological reference.  This module deliberately does
not import Torch, a model runtime, or an answer judge.  It prepares immutable
requests and reduces externally produced scores without guessing unresolved
perception-utility semantics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any


POLICY_SELECTION_CANDIDATE_SCHEMA = "tgvf.policy-selection.candidate.v1"
POLICY_SELECTION_REQUEST_SCHEMA = "tgvf.policy-selection.request.v1"
POLICY_SELECTION_ATTEMPT_SCHEMA = "tgvf.policy-selection.attempt.v1"
POLICY_SELECTION_DECISION_SCHEMA = "tgvf.policy-selection.decision.v1"
DEEPEYES_DIFFICULTY_ATTEMPTS = 8

DEEPEYES_REFERENCE_COUNTS = {
    "vstar": 22_362,
    "arxivqa": 13_659,
    "thinklite": 11_031,
}
DEEPEYES_REFERENCE_TOTAL = sum(DEEPEYES_REFERENCE_COUNTS.values())
DEEPEYES_REFERENCE_SHARES = {
    source: count / DEEPEYES_REFERENCE_TOTAL
    for source, count in DEEPEYES_REFERENCE_COUNTS.items()
}


class SelectionSource(str, Enum):
    VSTAR = "vstar"
    ARXIVQA = "arxivqa"
    THINKLITE = "thinklite"


class SelectionBranch(str, Enum):
    FULL_IMAGE = "full_image"
    GT_REGION = "gt_region"


class AttemptStatus(str, Enum):
    SCORED = "scored"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"
    VERIFIER_ERROR = "verifier_error"


class T1Decision(str, Enum):
    RETAIN = "retain"
    EXCLUDE_TOO_HARD = "exclude_too_hard"
    EXCLUDE_TOO_EASY = "exclude_too_easy"
    UNRESOLVED = "unresolved"


class T2Decision(str, Enum):
    NOT_APPLICABLE_PRESERVE_T1 = "not_applicable_preserve_t1"
    NOT_EVALUATED_T1_EXCLUDED = "not_evaluated_t1_excluded"
    UNRESOLVED = "unresolved"


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc
    return encoded.encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_positive_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _required_sha256(value: Any, *, field_name: str) -> str:
    value = _required_string(value, field_name=field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _normalized_json(value: Any, *, field_name: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be finite JSON data") from exc


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    sample_id: str
    source: SelectionSource
    question: str
    ground_truth: Any
    image: Mapping[str, Any]
    gt_regions: tuple[tuple[int, int, int, int], ...]
    provenance: Mapping[str, Any]
    canonical_record: Mapping[str, Any]
    identity_sha256: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SelectionCandidate":
        if not isinstance(record, Mapping):
            raise TypeError("candidate must be a mapping")
        schema = record.get("schema_version")
        if schema != POLICY_SELECTION_CANDIDATE_SCHEMA:
            raise ValueError(
                f"candidate schema_version must be {POLICY_SELECTION_CANDIDATE_SCHEMA!r}"
            )
        sample_id = _required_string(record.get("sample_id"), field_name="sample_id")
        try:
            source = SelectionSource(record.get("source"))
        except ValueError as exc:
            raise ValueError("source must be vstar, arxivqa, or thinklite") from exc
        question = _required_string(record.get("question"), field_name="question")
        if "ground_truth" not in record:
            raise ValueError("ground_truth is required")
        ground_truth = _normalized_json(record["ground_truth"], field_name="ground_truth")
        if ground_truth is None or (
            isinstance(ground_truth, str) and not ground_truth.strip()
        ):
            raise ValueError("ground_truth must be non-empty")

        image_value = record.get("image")
        if not isinstance(image_value, Mapping):
            raise ValueError("image must be a mapping")
        image = dict(_normalized_json(image_value, field_name="image"))
        _required_sha256(image.get("sha256"), field_name="image.sha256")
        width = _required_positive_int(image.get("width"), field_name="image.width")
        height = _required_positive_int(image.get("height"), field_name="image.height")
        if "path" in image:
            _required_string(image["path"], field_name="image.path")

        regions_value = record.get("gt_regions", [])
        if not isinstance(regions_value, Sequence) or isinstance(
            regions_value, (str, bytes, bytearray)
        ):
            raise ValueError("gt_regions must be a sequence")
        regions: list[tuple[int, int, int, int]] = []
        for index, region_value in enumerate(regions_value):
            if not isinstance(region_value, Sequence) or isinstance(
                region_value, (str, bytes, bytearray)
            ) or len(region_value) != 4:
                raise ValueError(f"gt_regions[{index}] must contain four integers")
            if any(type(coordinate) is not int for coordinate in region_value):
                raise ValueError(f"gt_regions[{index}] must contain four integers")
            left, top, right, bottom = region_value
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(
                    f"gt_regions[{index}] must be a non-empty source-pixel box"
                )
            regions.append((left, top, right, bottom))

        provenance_value = record.get("provenance")
        if not isinstance(provenance_value, Mapping) or not provenance_value:
            raise ValueError("provenance must be a non-empty mapping")
        provenance = dict(_normalized_json(provenance_value, field_name="provenance"))
        canonical = dict(_normalized_json(record, field_name="candidate"))
        return cls(
            sample_id=sample_id,
            source=source,
            question=question,
            ground_truth=ground_truth,
            image=image,
            gt_regions=tuple(regions),
            provenance=provenance,
            canonical_record=canonical,
            identity_sha256=_sha256(canonical),
        )


def stable_selection_request_id(
    *, candidate_sha256: str, branch: SelectionBranch, attempt_index: int
) -> str:
    _required_sha256(candidate_sha256, field_name="candidate_sha256")
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    identity = {
        "schema_version": POLICY_SELECTION_REQUEST_SCHEMA,
        "candidate_sha256": candidate_sha256,
        "branch": branch.value,
        "attempt_index": attempt_index,
    }
    return f"qwen3-selection:{_sha256(identity)}"


def build_selection_requests(
    candidates: Iterable[Mapping[str, Any]],
    *,
    oracle_attempts: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic GPU-work requests without importing a model runtime."""

    if type(oracle_attempts) is not int or oracle_attempts < 0:
        raise ValueError("oracle_attempts must be a non-negative integer")
    parsed = [SelectionCandidate.from_record(record) for record in candidates]
    sample_ids = [candidate.sample_id for candidate in parsed]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("candidate sample_id values must be unique")

    requests: list[dict[str, Any]] = []
    for candidate in sorted(parsed, key=lambda item: item.sample_id):
        branch_counts = [(SelectionBranch.FULL_IMAGE, DEEPEYES_DIFFICULTY_ATTEMPTS)]
        if candidate.source is SelectionSource.VSTAR and oracle_attempts:
            if not candidate.gt_regions:
                raise ValueError(
                    f"V* candidate {candidate.sample_id!r} has no gt_regions"
                )
            branch_counts.append((SelectionBranch.GT_REGION, oracle_attempts))
        for branch, count in branch_counts:
            for attempt_index in range(count):
                request_id = stable_selection_request_id(
                    candidate_sha256=candidate.identity_sha256,
                    branch=branch,
                    attempt_index=attempt_index,
                )
                model_input: dict[str, Any] = {
                    "image": dict(candidate.image),
                    "question": candidate.question,
                }
                if branch is SelectionBranch.GT_REGION:
                    model_input["gt_regions"] = [list(box) for box in candidate.gt_regions]
                requests.append(
                    {
                        "schema_version": POLICY_SELECTION_REQUEST_SCHEMA,
                        "request_id": request_id,
                        "sample_id": candidate.sample_id,
                        "candidate_sha256": candidate.identity_sha256,
                        "source": candidate.source.value,
                        "branch": branch.value,
                        "attempt_index": attempt_index,
                        "model_input": model_input,
                        "verifier_input": {"ground_truth": candidate.ground_truth},
                    }
                )
    return tuple(requests)


@dataclass(frozen=True, slots=True)
class SelectionAttempt:
    request_id: str
    sample_id: str
    source: SelectionSource
    branch: SelectionBranch
    attempt_index: int
    status: AttemptStatus
    correct: bool | None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SelectionAttempt":
        if not isinstance(record, Mapping):
            raise TypeError("attempt must be a mapping")
        if record.get("schema_version") != POLICY_SELECTION_ATTEMPT_SCHEMA:
            raise ValueError(
                f"attempt schema_version must be {POLICY_SELECTION_ATTEMPT_SCHEMA!r}"
            )
        request_id = _required_string(record.get("request_id"), field_name="request_id")
        sample_id = _required_string(record.get("sample_id"), field_name="sample_id")
        try:
            source = SelectionSource(record.get("source"))
            branch = SelectionBranch(record.get("branch"))
            status = AttemptStatus(record.get("status"))
        except ValueError as exc:
            raise ValueError("attempt contains an unsupported enum value") from exc
        attempt_index = record.get("attempt_index")
        if type(attempt_index) is not int or attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        correct = record.get("correct")
        if status is AttemptStatus.SCORED:
            if type(correct) is not bool:
                raise ValueError("a scored attempt requires boolean correct")
        elif correct is not None:
            raise ValueError("an unscored attempt must not carry correct")
        return cls(
            request_id=request_id,
            sample_id=sample_id,
            source=source,
            branch=branch,
            attempt_index=attempt_index,
            status=status,
            correct=correct,
        )


def _branch_summary(
    attempts: Sequence[SelectionAttempt], *, expected_attempts: int
) -> dict[str, Any]:
    by_index: dict[int, SelectionAttempt] = {}
    for attempt in attempts:
        if attempt.attempt_index in by_index:
            raise ValueError(
                f"duplicate attempt index {attempt.attempt_index} for "
                f"{attempt.sample_id}/{attempt.branch.value}"
            )
        if attempt.attempt_index >= expected_attempts:
            raise ValueError(
                f"attempt index {attempt.attempt_index} exceeds expected range"
            )
        by_index[attempt.attempt_index] = attempt
    status_counts = Counter(attempt.status.value for attempt in attempts)
    scored = [attempt for attempt in attempts if attempt.status is AttemptStatus.SCORED]
    correct_count = sum(attempt.correct is True for attempt in scored)
    missing_indices = sorted(set(range(expected_attempts)) - set(by_index))
    complete = not missing_indices and len(scored) == expected_attempts
    return {
        "expected_attempts": expected_attempts,
        "observed_attempts": len(attempts),
        "scoreable_attempts": len(scored),
        "correct_count": correct_count,
        "accuracy": correct_count / expected_attempts if complete else None,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_indices": missing_indices,
        "complete": complete,
    }


def reduce_selection_attempts(
    candidates: Iterable[Mapping[str, Any]],
    attempts: Iterable[Mapping[str, Any]],
    *,
    expected_oracle_attempts: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Apply T1 and preserve T2 evidence without guessing a utility threshold."""

    if type(expected_oracle_attempts) is not int or expected_oracle_attempts < 0:
        raise ValueError("expected_oracle_attempts must be a non-negative integer")
    parsed_candidates = [SelectionCandidate.from_record(record) for record in candidates]
    candidates_by_id = {candidate.sample_id: candidate for candidate in parsed_candidates}
    if len(candidates_by_id) != len(parsed_candidates):
        raise ValueError("candidate sample_id values must be unique")

    grouped: dict[tuple[str, SelectionBranch], list[SelectionAttempt]] = defaultdict(list)
    request_ids: set[str] = set()
    for record in attempts:
        attempt = SelectionAttempt.from_record(record)
        candidate = candidates_by_id.get(attempt.sample_id)
        if candidate is None:
            raise ValueError(f"attempt refers to unknown sample {attempt.sample_id!r}")
        if attempt.source is not candidate.source:
            raise ValueError(f"attempt source mismatch for {attempt.sample_id!r}")
        if attempt.branch is SelectionBranch.GT_REGION:
            if candidate.source is not SelectionSource.VSTAR:
                raise ValueError("gt_region attempts are valid only for V* candidates")
            if expected_oracle_attempts == 0:
                raise ValueError(
                    "expected_oracle_attempts must be explicit when gt_region attempts exist"
                )
        expected_request_id = stable_selection_request_id(
            candidate_sha256=candidate.identity_sha256,
            branch=attempt.branch,
            attempt_index=attempt.attempt_index,
        )
        if attempt.request_id != expected_request_id:
            raise ValueError(f"request_id identity mismatch for {attempt.sample_id!r}")
        if attempt.request_id in request_ids:
            raise ValueError(f"duplicate request_id {attempt.request_id!r}")
        request_ids.add(attempt.request_id)
        grouped[(attempt.sample_id, attempt.branch)].append(attempt)

    decisions: list[dict[str, Any]] = []
    for candidate in sorted(parsed_candidates, key=lambda item: item.sample_id):
        full_summary = _branch_summary(
            grouped[(candidate.sample_id, SelectionBranch.FULL_IMAGE)],
            expected_attempts=DEEPEYES_DIFFICULTY_ATTEMPTS,
        )
        if not full_summary["complete"]:
            t1_decision = T1Decision.UNRESOLVED
            t1_reason = "requires_exactly_eight_scoreable_full_image_attempts"
        elif full_summary["correct_count"] == 0:
            t1_decision = T1Decision.EXCLUDE_TOO_HARD
            t1_reason = "zero_of_eight_correct"
        elif full_summary["correct_count"] == DEEPEYES_DIFFICULTY_ATTEMPTS:
            t1_decision = T1Decision.EXCLUDE_TOO_EASY
            t1_reason = "eight_of_eight_correct"
        else:
            t1_decision = T1Decision.RETAIN
            t1_reason = "between_one_and_seven_of_eight_correct"

        oracle_summary: dict[str, Any] | None = None
        if candidate.source is not SelectionSource.VSTAR:
            t2_decision = T2Decision.NOT_APPLICABLE_PRESERVE_T1
            t2_reason = "deepeyes_perception_utility_is_vstar_only"
        elif t1_decision is not T1Decision.RETAIN:
            t2_decision = T2Decision.NOT_EVALUATED_T1_EXCLUDED
            t2_reason = "sample_did_not_pass_t1"
        else:
            if expected_oracle_attempts:
                oracle_summary = _branch_summary(
                    grouped[(candidate.sample_id, SelectionBranch.GT_REGION)],
                    expected_attempts=expected_oracle_attempts,
                )
            t2_decision = T2Decision.UNRESOLVED
            t2_reason = (
                "oracle_attempts_incomplete"
                if oracle_summary is not None and not oracle_summary["complete"]
                else "perception_utility_membership_rule_not_accepted"
            )

        decisions.append(
            {
                "schema_version": POLICY_SELECTION_DECISION_SCHEMA,
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "source": candidate.source.value,
                "t1": {
                    "decision": t1_decision.value,
                    "reason": t1_reason,
                    "full_image": full_summary,
                },
                "t2": {
                    "decision": t2_decision.value,
                    "reason": t2_reason,
                    "gt_region": oracle_summary,
                },
            }
        )
    return tuple(decisions)


def summarize_selection_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report T1 source mixture against DeepEyes without enforcing a tolerance."""

    total_by_source: Counter[str] = Counter()
    retained_by_source: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for record in decisions:
        source = SelectionSource(record.get("source")).value
        t1 = record.get("t1")
        if not isinstance(t1, Mapping):
            raise ValueError("decision t1 must be a mapping")
        decision = T1Decision(t1.get("decision"))
        total_by_source[source] += 1
        decision_counts[decision.value] += 1
        if decision is T1Decision.RETAIN:
            retained_by_source[source] += 1
    retained_total = sum(retained_by_source.values())
    source_report: dict[str, Any] = {}
    for source in SelectionSource:
        retained = retained_by_source[source.value]
        share = retained / retained_total if retained_total else None
        reference_share = DEEPEYES_REFERENCE_SHARES[source.value]
        source_report[source.value] = {
            "candidate_count": total_by_source[source.value],
            "retained_count": retained,
            "retained_share": share,
            "deepeyes_reference_count": DEEPEYES_REFERENCE_COUNTS[source.value],
            "deepeyes_reference_share": reference_share,
            "share_delta": share - reference_share if share is not None else None,
        }
    return {
        "reference": {
            "name": "ChenShawn/DeepEyes-Datasets-47k",
            "total": DEEPEYES_REFERENCE_TOTAL,
            "counts": dict(DEEPEYES_REFERENCE_COUNTS),
        },
        "candidate_total": sum(total_by_source.values()),
        "t1_retained_total": retained_total,
        "t1_decision_counts": dict(sorted(decision_counts.items())),
        "sources": source_report,
        "distribution_tolerance": None,
        "distribution_membership_enforced": False,
    }


def canonical_json_line(record: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(record) + b"\n"


def records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_line(record))
    return digest.hexdigest()
