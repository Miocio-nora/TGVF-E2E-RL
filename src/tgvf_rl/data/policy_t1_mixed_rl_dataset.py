"""Immutable all-source Policy-RL rows retained by final T1 scoring.

This module is deliberately independent from the historical ArxivQA-only
materializer.  It consumes the canonical ``final-v1`` scoring manifest, joins
all three T1 sources to its decisions, ignores T2 membership, and publishes
every T1-retained row without post-selection balancing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from .deepeyes47k import DeepEyesTaskKind
from .policy_selection import (
    POLICY_SELECTION_PRIMARY_SOURCES,
    POLICY_SELECTION_DECISION_SCHEMA,
    POLICY_SELECTION_TASK_KIND_POLICY,
    SelectionCandidate,
    SelectionSource,
    T1Decision,
    canonical_json_line,
    classify_policy_selection_task_kind,
)


POLICY_T1_MIXED_DATASET_KIND = "policy_t1_retained_mixed"
POLICY_T1_MIXED_SAMPLE_SCHEMA = "tgvf.policy-t1-mixed-rl.sample.v2"
POLICY_T1_MIXED_MANIFEST_SCHEMA = "tgvf.policy-t1-mixed-rl.manifest.v2"
POLICY_T1_MIXED_RUNTIME_SCHEMA = "tgvf.policy-t1-mixed-rl.runtime.v2"
POLICY_T1_MIXED_SAMPLES_FILE = "samples.jsonl"
POLICY_T1_MIXED_MANIFEST_FILE = "manifest.json"
POLICY_T1_MIXED_SHUFFLE_ALGORITHM = "sha256-sort-v1"
T1_FINAL_SCORING_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-final-scoring-manifest.v2"
T1_04_EXPECTED_SOURCE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        SelectionSource.VSTAR.value: 170_000,
        SelectionSource.ARXIVQA.value: 32_000,
        SelectionSource.THINKLITE.value: 69_842,
    }
)

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
_DECISION_FIELDS = {
    "schema_version",
    "candidate_sha256",
    "sample_id",
    "source",
    "t1",
    "t2",
}
_T1_FIELDS = {"decision", "full_image", "reason"}
_RETAINED_FULL_IMAGE_FIELDS = {
    "accuracy",
    "complete",
    "correct_count",
    "expected_attempts",
    "missing_indices",
    "observed_attempts",
    "scoreable_attempts",
    "status_counts",
}


class PolicyT1MixedMaterializationError(ValueError):
    """A mixed retained-pool input differs from its canonical identity."""


class PolicyT1MixedRuntimeValidationError(ValueError):
    """A mixed retained-pool artifact differs from its runtime binding."""


@dataclass(frozen=True, slots=True)
class _CandidateProjection:
    sample_id: str
    candidate_sha256: str
    source: SelectionSource
    question: str
    ground_truth: str
    task_kind: DeepEyesTaskKind
    image: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyT1MixedMaterializationResult:
    output_root: Path
    sample_count: int
    samples_sha256: str
    content_sha256: str
    manifest_file_sha256: str
    iteration_identity_sha256: str
    shuffle_seed: int
    source_counts: Mapping[str, Mapping[str, object]]

    def as_record(self) -> dict[str, object]:
        return {
            "dataset_kind": POLICY_T1_MIXED_DATASET_KIND,
            "root": str(self.output_root),
            "decision_stage": "final",
            "sample_count": self.sample_count,
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "shuffle_seed": self.shuffle_seed,
            "sources": json.loads(_canonical_json_bytes(dict(self.source_counts))),
        }


@dataclass(frozen=True, slots=True)
class PolicyT1MixedRuntimeBinding:
    """Content-addressed identity required by Policy RL at runtime."""

    manifest_file_sha256: str
    content_sha256: str
    shuffle_seed: int
    expected_sample_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _require_sha256(self.content_sha256, "content_sha256")
        if type(self.shuffle_seed) is not int or self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be a non-negative integer")
        if (
            type(self.expected_sample_count) is not int
            or self.expected_sample_count <= 0
        ):
            raise ValueError("expected_sample_count must be positive")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PolicyT1MixedMaterializationError(
            "mixed T1 artifact contains non-canonical JSON data"
        ) from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyT1MixedMaterializationError(
            f"{field_name} must be a lowercase SHA-256"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyT1MixedMaterializationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, field: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PolicyT1MixedMaterializationError(
                    f"{field} contains non-finite JSON number: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyT1MixedMaterializationError(
            f"{field} is not strict UTF-8 JSON"
        ) from error


def _safe_regular_file(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise PolicyT1MixedMaterializationError(
            f"{field} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    path = _safe_regular_file(path, field=str(path))
    observed = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise PolicyT1MixedMaterializationError(
                    f"{path}:{line_number}: blank lines are forbidden"
                )
            value = _parse_json(raw_line, field=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise PolicyT1MixedMaterializationError(
                    f"{path}:{line_number}: row must be an object"
                )
            if raw_line != canonical_json_line(value):
                raise PolicyT1MixedMaterializationError(
                    f"{path}:{line_number}: row is not canonical JSON"
                )
            observed += 1
            yield value
    if observed == 0:
        raise PolicyT1MixedMaterializationError(f"input is empty: {path}")


def _normalized_expected_source_counts(
    value: Mapping[str, int],
) -> dict[str, int]:
    expected_sources = {source.value for source in POLICY_SELECTION_PRIMARY_SOURCES}
    if set(value) != expected_sources:
        raise PolicyT1MixedMaterializationError(
            "expected_source_counts must bind vstar, arxivqa, and thinklite"
        )
    normalized: dict[str, int] = {}
    for source in POLICY_SELECTION_PRIMARY_SOURCES:
        count = value[source.value]
        if type(count) is not int or count <= 0:
            raise PolicyT1MixedMaterializationError(
                f"expected {source.value} count must be positive"
            )
        normalized[source.value] = count
    return normalized


def _load_final_manifest(
    path: str | Path,
) -> tuple[Path, Mapping[str, Any], str, str, Path, int, str]:
    manifest_path = _safe_regular_file(path, field="final_manifest")
    payload = manifest_path.read_bytes()
    value = _parse_json(payload, field="final_manifest")
    if not isinstance(value, dict) or set(value) != _FINAL_MANIFEST_FIELDS:
        raise PolicyT1MixedMaterializationError("final-v1 manifest schema differs")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise PolicyT1MixedMaterializationError(
            "final-v1 manifest is not canonical JSON"
        )
    if value.get("schema_version") != T1_FINAL_SCORING_MANIFEST_SCHEMA:
        raise PolicyT1MixedMaterializationError(
            "final-v1 manifest schema_version differs"
        )
    identity = dict(value)
    manifest_sha256 = _require_sha256(
        identity.pop("manifest_sha256", None), "final_manifest.manifest_sha256"
    )
    if _sha256_bytes(_canonical_json_bytes(identity)) != manifest_sha256:
        raise PolicyT1MixedMaterializationError(
            "final-v1 manifest identity SHA-256 differs"
        )
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise PolicyT1MixedMaterializationError("final_manifest.run_id is invalid")
    for field_name in (
        "run_manifest_sha256",
        "scoring_manifest_sha256",
        "judge_manifest_sha256",
    ):
        _require_sha256(value.get(field_name), f"final_manifest.{field_name}")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_FINAL_FILE_FIELDS):
        raise PolicyT1MixedMaterializationError("final-v1 files schema differs")
    for name, expected_fields in _FINAL_FILE_FIELDS.items():
        descriptor = files.get(name)
        if not isinstance(descriptor, Mapping) or set(descriptor) != expected_fields:
            raise PolicyT1MixedMaterializationError(
                f"final-v1 {name} descriptor schema differs"
            )
        if descriptor.get("path") != f"{name}.jsonl" and name != "report":
            raise PolicyT1MixedMaterializationError(f"final-v1 {name} path differs")
        if name == "report" and descriptor.get("path") != "report.json":
            raise PolicyT1MixedMaterializationError("final-v1 report path differs")
        _require_sha256(descriptor.get("sha256"), f"final_manifest.files.{name}.sha256")
        if name != "report" and (
            type(descriptor.get("rows")) is not int or descriptor["rows"] <= 0
        ):
            raise PolicyT1MixedMaterializationError(
                f"final-v1 {name} rows must be positive"
            )
    decisions = files["decisions"]
    decisions_path = manifest_path.parent / "decisions.jsonl"
    decisions_path = _safe_regular_file(decisions_path, field="final decisions")
    return (
        manifest_path,
        value,
        _sha256_bytes(payload),
        manifest_sha256,
        decisions_path,
        int(decisions["rows"]),
        str(decisions["sha256"]),
    )


def _decision_identity(
    record: Mapping[str, Any],
) -> tuple[str, str, SelectionSource, T1Decision, Mapping[str, Any]]:
    if (
        set(record) != _DECISION_FIELDS
        or record.get("schema_version") != POLICY_SELECTION_DECISION_SCHEMA
    ):
        raise PolicyT1MixedMaterializationError("T1 decision schema differs")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise PolicyT1MixedMaterializationError("decision.sample_id must be non-empty")
    candidate_sha256 = _require_sha256(
        record.get("candidate_sha256"), "decision.candidate_sha256"
    )
    try:
        source = SelectionSource(record.get("source"))
    except (TypeError, ValueError) as error:
        raise PolicyT1MixedMaterializationError(
            "decision.source is unsupported"
        ) from error
    t1 = record.get("t1")
    if not isinstance(t1, Mapping) or set(t1) != _T1_FIELDS:
        raise PolicyT1MixedMaterializationError("decision.t1 schema differs")
    try:
        decision = T1Decision(t1.get("decision"))
    except (TypeError, ValueError) as error:
        raise PolicyT1MixedMaterializationError("decision.t1 value differs") from error
    if not isinstance(t1.get("full_image"), Mapping):
        raise PolicyT1MixedMaterializationError(
            "decision.t1.full_image must be an object"
        )
    reason = t1.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PolicyT1MixedMaterializationError("decision.t1.reason must be non-empty")
    if decision is T1Decision.RETAIN:
        _validate_retained_t1(t1)
    # T2 is deliberately not inspected: it is provenance bound by the
    # decision SHA but has no membership effect for this retained pool.
    return sample_id, candidate_sha256, source, decision, t1


def _validate_retained_t1(t1: Mapping[str, Any]) -> None:
    full_image = t1["full_image"]
    if set(full_image) != _RETAINED_FULL_IMAGE_FIELDS:
        raise PolicyT1MixedMaterializationError(
            "retained decision full_image schema differs"
        )
    correct_count = full_image.get("correct_count")
    accuracy = full_image.get("accuracy")
    expected = {
        "complete": True,
        "expected_attempts": 8,
        "observed_attempts": 8,
        "scoreable_attempts": 8,
        "missing_indices": [],
        "status_counts": {"scored": 8},
    }
    if any(full_image.get(key) != value for key, value in expected.items()):
        raise PolicyT1MixedMaterializationError(
            "retained decision does not have eight complete scored attempts"
        )
    if type(correct_count) is not int or not 1 <= correct_count <= 7:
        raise PolicyT1MixedMaterializationError(
            "retained decision correct_count must be between one and seven"
        )
    if type(accuracy) not in {int, float} or accuracy != correct_count / 8:
        raise PolicyT1MixedMaterializationError(
            "retained decision accuracy differs from correct_count"
        )
    if t1.get("reason") != "between_one_and_seven_of_eight_correct":
        raise PolicyT1MixedMaterializationError("retained decision reason differs")


def _resolved_image(
    candidate: _CandidateProjection, *, hash_cache: dict[Path, str]
) -> tuple[Path, str, int, int]:
    raw_path = candidate.image.get("path")
    if not isinstance(raw_path, str):
        raise PolicyT1MixedMaterializationError("candidate image.path is required")
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PolicyT1MixedMaterializationError(
            "candidate image must be an absolute regular non-symlink file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise PolicyT1MixedMaterializationError(
            "candidate image path is not normalized"
        )
    image_sha256 = _require_sha256(candidate.image.get("sha256"), "image.sha256")
    observed_sha256 = hash_cache.get(resolved)
    if observed_sha256 is None:
        observed_sha256 = _sha256_file(resolved)
        hash_cache[resolved] = observed_sha256
    if observed_sha256 != image_sha256:
        raise PolicyT1MixedMaterializationError("candidate image SHA-256 differs")
    return (
        resolved,
        image_sha256,
        int(candidate.image["width"]),
        int(candidate.image["height"]),
    )


def _shuffle_key(sample_id: str, seed: int) -> tuple[str, str]:
    payload = f"{POLICY_T1_MIXED_SHUFFLE_ALGORITHM}\0{seed}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest(), sample_id


def policy_t1_mixed_iteration_identity_sha256(
    binding: PolicyT1MixedRuntimeBinding, *, samples_sha256: str
) -> str:
    """Return the immutable sample-iteration identity for one bound pool."""

    if not isinstance(binding, PolicyT1MixedRuntimeBinding):
        raise TypeError("binding must be PolicyT1MixedRuntimeBinding")
    _require_sha256(samples_sha256, "samples_sha256")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": POLICY_T1_MIXED_RUNTIME_SCHEMA,
                "dataset_kind": POLICY_T1_MIXED_DATASET_KIND,
                "decision_stage": "final",
                "sample_count": binding.expected_sample_count,
                "shuffle_algorithm": POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
                "shuffle_seed": binding.shuffle_seed,
                "manifest_file_sha256": binding.manifest_file_sha256,
                "content_sha256": binding.content_sha256,
                "samples_sha256": samples_sha256,
            }
        )
    )


def materialize_policy_t1_mixed_retained_pool(
    candidates_path: str | Path,
    final_manifest_path: str | Path,
    output_root: str | Path,
    *,
    shuffle_seed: int = 42,
    expected_source_counts: Mapping[str, int] = T1_04_EXPECTED_SOURCE_COUNTS,
) -> PolicyT1MixedMaterializationResult:
    """Publish every final T1 retain from V*, ArxivQA, and ThinkLite."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a non-negative integer")
    expected_counts = _normalized_expected_source_counts(expected_source_counts)
    expected_total = sum(expected_counts.values())
    candidates_path = _safe_regular_file(candidates_path, field="candidates")
    output_root = Path(output_root).resolve()
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to replace mixed Policy T1 artifact: {output_root}"
        )
    (
        final_manifest_path,
        final_manifest,
        final_manifest_file_sha256,
        final_manifest_identity_sha256,
        decisions_path,
        decision_rows,
        decision_file_sha256,
    ) = _load_final_manifest(final_manifest_path)
    if decision_rows != expected_total:
        raise PolicyT1MixedMaterializationError(
            "final-v1 decision row count differs from the expected T1 population"
        )
    if _sha256_file(decisions_path) != decision_file_sha256:
        raise PolicyT1MixedMaterializationError(
            "final-v1 decisions file SHA-256 differs"
        )

    candidates: dict[str, _CandidateProjection] = {}
    candidate_counts: Counter[str] = Counter()
    for record in _jsonl_records(candidates_path):
        candidate = SelectionCandidate.from_record(record)
        if candidate.sample_id in candidates:
            raise PolicyT1MixedMaterializationError(
                "duplicate mixed T1 candidate sample_id"
            )
        if (
            not isinstance(candidate.ground_truth, str)
            or not candidate.ground_truth.strip()
        ):
            raise PolicyT1MixedMaterializationError(
                "mixed T1 ground truth must be non-empty text"
            )
        candidates[candidate.sample_id] = _CandidateProjection(
            sample_id=candidate.sample_id,
            candidate_sha256=candidate.identity_sha256,
            source=candidate.source,
            question=candidate.question,
            ground_truth=candidate.ground_truth,
            task_kind=classify_policy_selection_task_kind(
                source=candidate.source,
                question=candidate.question,
                ground_truth=candidate.ground_truth,
            ),
            image=dict(candidate.image),
        )
        candidate_counts[candidate.source.value] += 1
    if len(candidates) != expected_total or dict(candidate_counts) != expected_counts:
        raise PolicyT1MixedMaterializationError(
            "candidate source counts differ from the expected T1 population"
        )

    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    decision_counts_by_source: dict[str, Counter[str]] = {
        source.value: Counter() for source in POLICY_SELECTION_PRIMARY_SOURCES
    }
    retained_task_kind_counts_by_source: dict[str, Counter[str]] = {
        source.value: Counter() for source in POLICY_SELECTION_PRIMARY_SOURCES
    }
    seen: set[str] = set()
    image_hash_cache: dict[Path, str] = {}
    observed_decision_rows = 0
    for decision_record in _jsonl_records(decisions_path):
        observed_decision_rows += 1
        sample_id, candidate_sha256, source, decision, t1 = _decision_identity(
            decision_record
        )
        candidate = candidates.get(sample_id)
        if candidate is None:
            raise PolicyT1MixedMaterializationError(
                "T1 decision refers to an unknown candidate"
            )
        if sample_id in seen:
            raise PolicyT1MixedMaterializationError("duplicate mixed T1 decision")
        seen.add(sample_id)
        if (
            candidate.candidate_sha256 != candidate_sha256
            or candidate.source is not source
        ):
            raise PolicyT1MixedMaterializationError(
                "T1 decision candidate identity or source differs"
            )
        decision_counts[decision.value] += 1
        decision_counts_by_source[source.value][decision.value] += 1
        if decision is not T1Decision.RETAIN:
            continue
        retained_task_kind_counts_by_source[source.value][
            candidate.task_kind.value
        ] += 1
        image_path, image_sha256, width, height = _resolved_image(
            candidate, hash_cache=image_hash_cache
        )
        records.append(
            {
                "schema_version": POLICY_T1_MIXED_SAMPLE_SCHEMA,
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.candidate_sha256,
                "decision_sha256": _sha256_bytes(
                    _canonical_json_bytes(decision_record)
                ),
                "image": {
                    "path": str(image_path),
                    "sha256": image_sha256,
                    "width": width,
                    "height": height,
                },
                "extra_info": {"question": candidate.question},
                "reward_model": {"ground_truth": candidate.ground_truth},
                "data_source": source.value,
                "task_kind": candidate.task_kind.value,
                "selection": {
                    "decision_stage": "final",
                    "t1": json.loads(_canonical_json_bytes(t1)),
                },
            }
        )
    if observed_decision_rows != decision_rows:
        raise PolicyT1MixedMaterializationError(
            "final-v1 decisions row count differs from its manifest"
        )
    if set(candidates) != seen:
        raise PolicyT1MixedMaterializationError(
            "T1 decisions do not cover the complete mixed candidate population"
        )
    observed_decision_source_counts = {
        source: sum(decision_counts_by_source[source].values())
        for source in expected_counts
    }
    if observed_decision_source_counts != expected_counts:
        raise PolicyT1MixedMaterializationError(
            "decision source counts differ from the expected T1 population"
        )
    if not records:
        raise PolicyT1MixedMaterializationError("T1 retained no mixed-source rows")

    records.sort(key=lambda row: _shuffle_key(str(row["sample_id"]), shuffle_seed))
    retained_total = len(records)
    source_report: dict[str, dict[str, object]] = {}
    for source in POLICY_SELECTION_PRIMARY_SOURCES:
        source_decisions = decision_counts_by_source[source.value]
        retained_count = source_decisions[T1Decision.RETAIN.value]
        source_report[source.value] = {
            "candidate_count": candidate_counts[source.value],
            "decision_count": sum(source_decisions.values()),
            "t1_decision_counts": {
                value.value: source_decisions[value.value] for value in T1Decision
            },
            "retained_count": retained_count,
            "retained_share": retained_count / retained_total,
            "task_kind_counts": {
                task_kind.value: retained_task_kind_counts_by_source[source.value][
                    task_kind.value
                ]
                for task_kind in DeepEyesTaskKind
            },
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".policy-t1-mixed-", dir=output_root.parent)
    )
    try:
        samples_path = temporary_root / POLICY_T1_MIXED_SAMPLES_FILE
        samples_digest = hashlib.sha256()
        with samples_path.open("xb") as handle:
            for record in records:
                line = canonical_json_line(record)
                handle.write(line)
                samples_digest.update(line)
        samples_sha256 = samples_digest.hexdigest()
        descriptor = {
            "schema_version": POLICY_T1_MIXED_MANIFEST_SCHEMA,
            "dataset_kind": POLICY_T1_MIXED_DATASET_KIND,
            "decision_stage": "final",
            "task_kind_policy": POLICY_SELECTION_TASK_KIND_POLICY,
            "selection_policy": {
                "t1": T1Decision.RETAIN.value,
                "t2": "ignored",
                "post_t1_balancing": "none",
            },
            "inputs": {
                "candidates": {
                    "path": str(candidates_path),
                    "sha256": _sha256_file(candidates_path),
                    "rows": len(candidates),
                    "source_counts": expected_counts,
                },
                "final_scoring_manifest": {
                    "path": str(final_manifest_path),
                    "file_sha256": final_manifest_file_sha256,
                    "manifest_sha256": final_manifest_identity_sha256,
                    "schema_version": final_manifest["schema_version"],
                    "run_id": final_manifest["run_id"],
                    "run_manifest_sha256": final_manifest["run_manifest_sha256"],
                    "scoring_manifest_sha256": final_manifest[
                        "scoring_manifest_sha256"
                    ],
                    "judge_manifest_sha256": final_manifest["judge_manifest_sha256"],
                },
                "decisions": {
                    "path": str(decisions_path),
                    "sha256": decision_file_sha256,
                    "rows": decision_rows,
                },
            },
            "candidate_count": len(candidates),
            "decision_count": observed_decision_rows,
            "t1_decision_counts": {
                value.value: decision_counts[value.value] for value in T1Decision
            },
            "retained_count": retained_total,
            "sources": source_report,
            "shuffle": {
                "algorithm": POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
                "seed": shuffle_seed,
            },
            "samples": {
                "path": POLICY_T1_MIXED_SAMPLES_FILE,
                "rows": retained_total,
                "sha256": samples_sha256,
            },
            "images": {
                "address": "absolute-path-plus-sha256",
                "bytes_verified": True,
                "unique_paths_verified": len(image_hash_cache),
            },
        }
        content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_payload = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / POLICY_T1_MIXED_MANIFEST_FILE).write_bytes(manifest_payload)
        manifest_file_sha256 = _sha256_bytes(manifest_payload)
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    runtime_binding = PolicyT1MixedRuntimeBinding(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        shuffle_seed=shuffle_seed,
        expected_sample_count=retained_total,
    )
    iteration_identity_sha256 = policy_t1_mixed_iteration_identity_sha256(
        runtime_binding, samples_sha256=samples_sha256
    )
    return PolicyT1MixedMaterializationResult(
        output_root=output_root,
        sample_count=retained_total,
        samples_sha256=samples_sha256,
        content_sha256=content_sha256,
        manifest_file_sha256=manifest_file_sha256,
        iteration_identity_sha256=iteration_identity_sha256,
        shuffle_seed=shuffle_seed,
        source_counts=MappingProxyType(source_report),
    )


@dataclass(frozen=True, slots=True)
class PolicyT1MixedRuntimeSample:
    sample_id: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str
    data_source: str
    task_kind: DeepEyesTaskKind
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PolicyT1MixedRuntimeDataset:
    root: Path
    binding: PolicyT1MixedRuntimeBinding
    samples_sha256: str
    iteration_identity_sha256: str
    samples: tuple[PolicyT1MixedRuntimeSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicyT1MixedRuntimeSample:
        return self.samples[index]


_MIXED_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_kind",
    "decision_stage",
    "task_kind_policy",
    "selection_policy",
    "inputs",
    "candidate_count",
    "decision_count",
    "t1_decision_counts",
    "retained_count",
    "sources",
    "shuffle",
    "samples",
    "images",
    "content_sha256",
}
_MIXED_ROW_FIELDS = {
    "schema_version",
    "sample_id",
    "candidate_sha256",
    "decision_sha256",
    "image",
    "extra_info",
    "reward_model",
    "data_source",
    "task_kind",
    "selection",
}
_SOURCE_REPORT_FIELDS = {
    "candidate_count",
    "decision_count",
    "t1_decision_counts",
    "retained_count",
    "retained_share",
    "task_kind_counts",
}


def _runtime_manifest(root: Path) -> tuple[Path, Path, bytes, Mapping[str, Any]]:
    manifest_path = root / POLICY_T1_MIXED_MANIFEST_FILE
    samples_path = root / POLICY_T1_MIXED_SAMPLES_FILE
    for path in (manifest_path, samples_path):
        if path.is_symlink() or not path.is_file():
            raise PolicyT1MixedRuntimeValidationError(
                "mixed Policy T1 artifact file is unsafe"
            )
    payload = manifest_path.read_bytes()
    try:
        manifest = _parse_json(payload, field="mixed Policy T1 manifest")
    except PolicyT1MixedMaterializationError as error:
        raise PolicyT1MixedRuntimeValidationError(str(error)) from error
    if not isinstance(manifest, Mapping):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 manifest must be an object"
        )
    return manifest_path, samples_path, payload, manifest


def _validate_runtime_manifest_semantics(
    manifest: Mapping[str, Any],
    *,
    binding: PolicyT1MixedRuntimeBinding,
    samples_sha256: str,
) -> None:
    if set(manifest) != _MIXED_MANIFEST_FIELDS:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 manifest schema differs"
        )
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if (
        manifest.get("schema_version") != POLICY_T1_MIXED_MANIFEST_SCHEMA
        or manifest.get("dataset_kind") != POLICY_T1_MIXED_DATASET_KIND
        or manifest.get("decision_stage") != "final"
        or manifest.get("task_kind_policy") != POLICY_SELECTION_TASK_KIND_POLICY
        or manifest.get("selection_policy")
        != {"t1": "retain", "t2": "ignored", "post_t1_balancing": "none"}
        or manifest.get("content_sha256") != binding.content_sha256
        or _sha256_bytes(_canonical_json_bytes(descriptor)) != binding.content_sha256
        or manifest.get("retained_count") != binding.expected_sample_count
        or manifest.get("samples")
        != {
            "path": POLICY_T1_MIXED_SAMPLES_FILE,
            "rows": binding.expected_sample_count,
            "sha256": samples_sha256,
        }
        or manifest.get("shuffle")
        != {
            "algorithm": POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
            "seed": binding.shuffle_seed,
        }
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 manifest identity differs"
        )
    candidate_count = manifest.get("candidate_count")
    decision_count = manifest.get("decision_count")
    if (
        type(candidate_count) is not int
        or candidate_count <= 0
        or decision_count != candidate_count
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 candidate/decision population differs"
        )
    decision_counts = manifest.get("t1_decision_counts")
    expected_decision_keys = {decision.value for decision in T1Decision}
    if (
        not isinstance(decision_counts, Mapping)
        or set(decision_counts) != expected_decision_keys
        or any(
            type(value) is not int or value < 0 for value in decision_counts.values()
        )
        or sum(decision_counts.values()) != candidate_count
        or decision_counts.get(T1Decision.RETAIN.value) != binding.expected_sample_count
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 decision counts differ"
        )
    images = manifest.get("images")
    if (
        not isinstance(images, Mapping)
        or set(images) != {"address", "bytes_verified", "unique_paths_verified"}
        or images.get("address") != "absolute-path-plus-sha256"
        or images.get("bytes_verified") is not True
        or type(images.get("unique_paths_verified")) is not int
        or not 0 < images["unique_paths_verified"] <= binding.expected_sample_count
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 image identity contract differs"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        source.value for source in POLICY_SELECTION_PRIMARY_SOURCES
    }:
        raise PolicyT1MixedRuntimeValidationError("mixed Policy T1 source set differs")
    retained_sum = 0
    candidate_sum = 0
    expected_task_kind_keys = {task_kind.value for task_kind in DeepEyesTaskKind}
    for source in POLICY_SELECTION_PRIMARY_SOURCES:
        report = sources[source.value]
        if not isinstance(report, Mapping) or set(report) != _SOURCE_REPORT_FIELDS:
            raise PolicyT1MixedRuntimeValidationError(
                "mixed Policy T1 source report schema differs"
            )
        source_candidates = report.get("candidate_count")
        source_decisions = report.get("decision_count")
        source_retained = report.get("retained_count")
        source_counts = report.get("t1_decision_counts")
        task_kind_counts = report.get("task_kind_counts")
        if (
            type(source_candidates) is not int
            or source_candidates <= 0
            or source_decisions != source_candidates
            or type(source_retained) is not int
            or source_retained < 0
            or not isinstance(source_counts, Mapping)
            or set(source_counts) != expected_decision_keys
            or any(
                type(value) is not int or value < 0 for value in source_counts.values()
            )
            or sum(source_counts.values()) != source_candidates
            or source_counts.get(T1Decision.RETAIN.value) != source_retained
            or not isinstance(task_kind_counts, Mapping)
            or set(task_kind_counts) != expected_task_kind_keys
            or any(
                type(value) is not int or value < 0
                for value in task_kind_counts.values()
            )
            or sum(task_kind_counts.values()) != source_retained
            or report.get("retained_share")
            != source_retained / binding.expected_sample_count
        ):
            raise PolicyT1MixedRuntimeValidationError(
                "mixed Policy T1 source report identity differs"
            )
        candidate_sum += source_candidates
        retained_sum += source_retained
    if (
        candidate_sum != candidate_count
        or retained_sum != binding.expected_sample_count
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 source totals differ"
        )


def verify_policy_t1_mixed_artifact_binding(
    root: str | Path,
    *,
    binding: PolicyT1MixedRuntimeBinding,
    samples_sha256: str,
) -> Mapping[str, Any]:
    """Verify the immutable mixed manifest and sample-file identities."""

    if not isinstance(binding, PolicyT1MixedRuntimeBinding):
        raise TypeError("binding must be PolicyT1MixedRuntimeBinding")
    _require_sha256(samples_sha256, "samples_sha256")
    root_path = Path(root)
    if not root_path.is_absolute() or root_path.is_symlink() or not root_path.is_dir():
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 root must be an absolute regular directory"
        )
    root_path = root_path.resolve(strict=True)
    manifest_path, samples_path, payload, manifest = _runtime_manifest(root_path)
    del manifest_path
    if _sha256_bytes(payload) != binding.manifest_file_sha256:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 manifest file hash differs"
        )
    if payload != _canonical_json_bytes(manifest) + b"\n":
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 manifest is not canonical"
        )
    _validate_runtime_manifest_semantics(
        manifest, binding=binding, samples_sha256=samples_sha256
    )
    if _sha256_file(samples_path) != samples_sha256:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 samples file hash differs"
        )
    return manifest


def load_policy_t1_mixed_runtime(
    root: str | Path, *, binding: PolicyT1MixedRuntimeBinding
) -> PolicyT1MixedRuntimeDataset:
    """Load every final T1 retain after validating artifact and image bytes."""

    root_path = Path(root)
    if not root_path.is_absolute():
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 root must be absolute"
        )
    root_path = root_path.resolve(strict=True)
    _, _, _, unverified_manifest = _runtime_manifest(root_path)
    samples_descriptor = unverified_manifest.get("samples")
    if not isinstance(samples_descriptor, Mapping):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 samples descriptor differs"
        )
    samples_sha256 = _require_sha256(samples_descriptor.get("sha256"), "samples.sha256")
    manifest = verify_policy_t1_mixed_artifact_binding(
        root_path, binding=binding, samples_sha256=samples_sha256
    )
    samples: list[PolicyT1MixedRuntimeSample] = []
    ordered_ids: list[str] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    task_kind_counts_by_source: dict[str, Counter[str]] = {
        source.value: Counter() for source in POLICY_SELECTION_PRIMARY_SOURCES
    }
    image_hash_cache: dict[Path, str] = {}
    try:
        records = _jsonl_records(root_path / POLICY_T1_MIXED_SAMPLES_FILE)
        for record in records:
            if (
                set(record) != _MIXED_ROW_FIELDS
                or record.get("schema_version") != POLICY_T1_MIXED_SAMPLE_SCHEMA
            ):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 sample schema differs"
                )
            sample_id = record.get("sample_id")
            image = record.get("image")
            extra_info = record.get("extra_info")
            reward_model = record.get("reward_model")
            selection = record.get("selection")
            if (
                not isinstance(sample_id, str)
                or not sample_id.strip()
                or sample_id in seen
                or not isinstance(image, Mapping)
                or set(image) != {"path", "sha256", "width", "height"}
                or not isinstance(extra_info, Mapping)
                or set(extra_info) != {"question"}
                or not isinstance(reward_model, Mapping)
                or set(reward_model) != {"ground_truth"}
                or not isinstance(selection, Mapping)
                or set(selection) != {"decision_stage", "t1"}
            ):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 sample fields differ"
                )
            source_value = record.get("data_source")
            try:
                source = SelectionSource(source_value)
            except (TypeError, ValueError) as error:
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 sample source differs"
                ) from error
            t1 = selection.get("t1")
            if (
                selection.get("decision_stage") != "final"
                or not isinstance(t1, Mapping)
                or set(t1) != _T1_FIELDS
                or t1.get("decision") != T1Decision.RETAIN.value
            ):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 sample route is not final/retain"
                )
            try:
                _validate_retained_t1(t1)
            except PolicyT1MixedMaterializationError as error:
                raise PolicyT1MixedRuntimeValidationError(str(error)) from error
            raw_image_path = image.get("path")
            if not isinstance(raw_image_path, str):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 image path differs"
                )
            image_path = Path(raw_image_path)
            image_sha256 = _require_sha256(image.get("sha256"), "image.sha256")
            width = image.get("width")
            height = image.get("height")
            if (
                not image_path.is_absolute()
                or image_path.is_symlink()
                or not image_path.is_file()
                or image_path.resolve(strict=True) != image_path
                or type(width) is not int
                or width <= 0
                or type(height) is not int
                or height <= 0
            ):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 source image differs"
                )
            observed_image_sha256 = image_hash_cache.get(image_path)
            if observed_image_sha256 is None:
                observed_image_sha256 = _sha256_file(image_path)
                image_hash_cache[image_path] = observed_image_sha256
            if observed_image_sha256 != image_sha256:
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 source image SHA-256 differs"
                )
            question = extra_info.get("question")
            ground_truth = reward_model.get("ground_truth")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(ground_truth, str)
                or not ground_truth.strip()
            ):
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 task text differs"
                )
            task_kind = classify_policy_selection_task_kind(
                source=source,
                question=question,
                ground_truth=ground_truth,
            )
            if record.get("task_kind") != task_kind.value:
                raise PolicyT1MixedRuntimeValidationError(
                    "mixed Policy T1 sample task-kind route differs"
                )
            candidate_sha256 = _require_sha256(
                record.get("candidate_sha256"), "candidate_sha256"
            )
            decision_sha256 = _require_sha256(
                record.get("decision_sha256"), "decision_sha256"
            )
            seen.add(sample_id)
            ordered_ids.append(sample_id)
            source_counts[source.value] += 1
            task_kind_counts_by_source[source.value][task_kind.value] += 1
            samples.append(
                PolicyT1MixedRuntimeSample(
                    sample_id=sample_id,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    question=question,
                    ground_truth=ground_truth,
                    data_source=source.value,
                    task_kind=task_kind,
                    metadata={
                        "candidate_sha256": candidate_sha256,
                        "decision_sha256": decision_sha256,
                        "decision_stage": "final",
                    },
                )
            )
    except PolicyT1MixedMaterializationError as error:
        raise PolicyT1MixedRuntimeValidationError(str(error)) from error
    if len(samples) != binding.expected_sample_count:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 sample count differs"
        )
    if ordered_ids != sorted(
        ordered_ids, key=lambda value: _shuffle_key(value, binding.shuffle_seed)
    ):
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 shuffle order differs"
        )
    expected_retained = {
        source: report["retained_count"]
        for source, report in manifest["sources"].items()
    }
    if dict(source_counts) != expected_retained:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 runtime source counts differ"
        )
    expected_task_kind_counts = {
        source: report["task_kind_counts"]
        for source, report in manifest["sources"].items()
    }
    observed_task_kind_counts = {
        source.value: {
            task_kind.value: task_kind_counts_by_source[source.value][task_kind.value]
            for task_kind in DeepEyesTaskKind
        }
        for source in POLICY_SELECTION_PRIMARY_SOURCES
    }
    if observed_task_kind_counts != expected_task_kind_counts:
        raise PolicyT1MixedRuntimeValidationError(
            "mixed Policy T1 runtime task-kind counts differ"
        )
    return PolicyT1MixedRuntimeDataset(
        root=root_path,
        binding=binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=policy_t1_mixed_iteration_identity_sha256(
            binding, samples_sha256=samples_sha256
        ),
        samples=tuple(samples),
    )


__all__ = [
    "POLICY_T1_MIXED_DATASET_KIND",
    "POLICY_T1_MIXED_MANIFEST_FILE",
    "POLICY_T1_MIXED_MANIFEST_SCHEMA",
    "POLICY_T1_MIXED_RUNTIME_SCHEMA",
    "POLICY_T1_MIXED_SAMPLE_SCHEMA",
    "POLICY_T1_MIXED_SAMPLES_FILE",
    "POLICY_T1_MIXED_SHUFFLE_ALGORITHM",
    "T1_04_EXPECTED_SOURCE_COUNTS",
    "PolicyT1MixedMaterializationError",
    "PolicyT1MixedMaterializationResult",
    "PolicyT1MixedRuntimeBinding",
    "PolicyT1MixedRuntimeDataset",
    "PolicyT1MixedRuntimeSample",
    "PolicyT1MixedRuntimeValidationError",
    "load_policy_t1_mixed_runtime",
    "materialize_policy_t1_mixed_retained_pool",
    "policy_t1_mixed_iteration_identity_sha256",
    "verify_policy_t1_mixed_artifact_binding",
]
