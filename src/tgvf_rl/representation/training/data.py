"""Audited retained-JSONL input for representation-phase training.

The source JSONL remains an external, immutable data artifact.  This module
reads one checksummed snapshot, applies the provenance-pinned focus-row filter,
and returns protocol-neutral samples together with a complete in-memory audit
manifest.  It never copies image or JSONL data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import warnings

from .schema import (
    RepresentationChoice,
    RepresentationSampleIdentity,
    RepresentationTrainingSample,
)


REPRESENTATION_DATA_TRANSFORM_VERSION = "retained_focus_rows_v1"
REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION = "representation_data_manifest_v1"
SPLIT_OVERLAP_REPORT_SCHEMA_VERSION = "representation_split_overlap_report_v1"

_FOCUS_TRAJECTORY_TYPE = "single_focus"
_FOCUS_EVIDENCE_STATE = "need_local_visual_evidence"
_OPTIONAL_SAMPLE_FIELDS = (
    "image_id",
    "stable_image_uid",
    "item_content_hash",
    "source_dataset",
    "source_profile",
    "evidence_type",
    "answer_type",
    "visual_difficulty",
    "target_leakage_risk",
)


class RepresentationDataError(ValueError):
    """The source cannot be interpreted under the fixed v1 transform."""


class RepresentationDataLeakageWarning(UserWarning):
    """A retained row has exact legacy short-answer/target term overlap."""


class RowExclusionReason(str, Enum):
    """Why a source row was not admitted to the representation population."""

    NOT_FOCUS_ROW = "not_focus_row"
    INVALID_FOCUS_METADATA = "invalid_focus_metadata"
    INVALID_REQUIRED_FIELD = "invalid_required_field"
    INVALID_OPTIONAL_FIELD = "invalid_optional_field"
    IMAGE_NOT_FOUND = "image_not_found"
    DUPLICATE_SAMPLE_ID = "duplicate_sample_id"
    DUPLICATE_GROUP_TARGET = "duplicate_group_target"


class DuplicateKind(str, Enum):
    SOURCE_ROW = "source_row"
    SAMPLE_ID = "sample_id"
    GROUP_TARGET = "group_target"


class SplitOverlapKind(str, Enum):
    IMAGE_GROUP_KEY = "image_group_key"
    IMAGE_PATH = "image_path"
    STABLE_IMAGE_UID = "stable_image_uid"
    ITEM_CONTENT_HASH = "item_content_hash"


class SplitOverlapPolicy(str, Enum):
    """Explicit policy applied to one content-bound train/validation audit."""

    REQUIRE_DISJOINT = "require_disjoint"
    ALLOW_RECORDED_IMAGE_PATH = "allow_recorded_image_path"


_SPLIT_OVERLAP_KIND_ORDER = {kind: index for index, kind in enumerate(SplitOverlapKind)}


@dataclass(frozen=True, slots=True)
class AcceptedRowManifestEntry:
    source_line: int
    source_row_sha256: str
    source_image_reference: str
    resolved_image_path: str
    sample: RepresentationSampleIdentity


@dataclass(frozen=True, slots=True)
class ExcludedRowManifestEntry:
    source_line: int
    source_row_sha256: str
    sample_id: str | None
    reasons: tuple[RowExclusionReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reasons, tuple):
            raise TypeError("excluded row reasons must be an immutable tuple")
        if not self.reasons:
            raise ValueError("excluded row must have at least one reason")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("excluded row reasons must be unique")


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    kind: DuplicateKind
    first_source_line: int
    duplicate_source_line: int
    key_sha256: str


@dataclass(frozen=True, slots=True)
class LeakageRecord:
    """Exact historical target/short-answer term-overlap signal."""

    source_line: int
    sample_id: str
    overlapping_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.overlapping_terms, tuple):
            raise TypeError("overlapping_terms must be an immutable tuple")
        if not self.overlapping_terms:
            raise ValueError("leakage record requires at least one overlapping term")
        if self.overlapping_terms != tuple(sorted(set(self.overlapping_terms))):
            raise ValueError("overlapping_terms must be unique and sorted")


@dataclass(frozen=True, slots=True)
class RepresentationDataManifest:
    schema_version: str
    transform_version: str
    source_path: str
    source_sha256: str
    accepted_rows: tuple[AcceptedRowManifestEntry, ...]
    excluded_rows: tuple[ExcludedRowManifestEntry, ...]
    duplicate_records: tuple[DuplicateRecord, ...]
    leakage_records: tuple[LeakageRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION:
            raise ValueError("representation data manifest schema mismatch")
        if self.transform_version != REPRESENTATION_DATA_TRANSFORM_VERSION:
            raise ValueError("representation data transform mismatch")
        _require_sha256(self.source_sha256, field_name="source_sha256")
        if not self.source_path:
            raise ValueError("source_path must be non-empty")
        for field_name in (
            "accepted_rows",
            "excluded_rows",
            "duplicate_records",
            "leakage_records",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be an immutable tuple")
        disposition_lines = tuple(
            row.source_line for row in (*self.accepted_rows, *self.excluded_rows)
        )
        if not disposition_lines:
            raise ValueError("representation data manifest has no source rows")
        if len(disposition_lines) != len(set(disposition_lines)):
            raise ValueError("a source line has more than one manifest disposition")
        if set(disposition_lines) != set(range(1, max(disposition_lines) + 1)):
            raise ValueError("manifest dispositions do not cover every source line")
        if tuple(row.source_line for row in self.accepted_rows) != tuple(
            sorted(row.source_line for row in self.accepted_rows)
        ):
            raise ValueError("accepted manifest rows must be source ordered")
        if tuple(row.source_line for row in self.excluded_rows) != tuple(
            sorted(row.source_line for row in self.excluded_rows)
        ):
            raise ValueError("excluded manifest rows must be source ordered")
        for row in (*self.accepted_rows, *self.excluded_rows):
            if row.source_line < 1:
                raise ValueError("source_line must be positive")
            _require_sha256(row.source_row_sha256, field_name="source_row_sha256")
        accepted_ids = tuple(row.sample.sample_id for row in self.accepted_rows)
        if len(accepted_ids) != len(set(accepted_ids)):
            raise ValueError("accepted manifest sample IDs must be unique")
        accepted_line_to_id = {
            row.source_line: row.sample.sample_id for row in self.accepted_rows
        }
        for record in self.duplicate_records:
            if (
                record.first_source_line < 1
                or record.duplicate_source_line <= record.first_source_line
            ):
                raise ValueError("duplicate record source lines are invalid")
            _require_sha256(record.key_sha256, field_name="duplicate key_sha256")
        for record in self.leakage_records:
            if accepted_line_to_id.get(record.source_line) != record.sample_id:
                raise ValueError("leakage record does not identify an accepted row")

    @property
    def manifest_sha256(self) -> str:
        """Canonical identity consumed by the same-image sampler/checkpoints."""

        payload = {
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "accepted_rows": [
                {
                    "source_line": row.source_line,
                    "source_row_sha256": row.source_row_sha256,
                    "source_image_reference": row.source_image_reference,
                    "resolved_image_path": row.resolved_image_path,
                    "sample": {
                        "schema_version": row.sample.schema_version,
                        "sample_id": row.sample.sample_id,
                        "image_group_key": row.sample.image_group_key,
                        "content_sha256": row.sample.content_sha256,
                    },
                }
                for row in self.accepted_rows
            ],
            "excluded_rows": [
                {
                    "source_line": row.source_line,
                    "source_row_sha256": row.source_row_sha256,
                    "sample_id": row.sample_id,
                    "reasons": [reason.value for reason in row.reasons],
                }
                for row in self.excluded_rows
            ],
            "duplicate_records": [
                {
                    "kind": record.kind.value,
                    "first_source_line": record.first_source_line,
                    "duplicate_source_line": record.duplicate_source_line,
                    "key_sha256": record.key_sha256,
                }
                for record in self.duplicate_records
            ],
            "leakage_records": [
                {
                    "source_line": record.source_line,
                    "sample_id": record.sample_id,
                    "overlapping_terms": list(record.overlapping_terms),
                }
                for record in self.leakage_records
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RepresentationDataset:
    samples: tuple[RepresentationTrainingSample, ...]
    manifest: RepresentationDataManifest

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple):
            raise TypeError("representation samples must be an immutable tuple")
        if not self.samples:
            raise ValueError("representation dataset has no accepted focus rows")
        if len(self.samples) != len(self.manifest.accepted_rows):
            raise ValueError("samples and accepted manifest rows disagree")
        identities = tuple(sample.identity for sample in self.samples)
        manifest_identities = tuple(row.sample for row in self.manifest.accepted_rows)
        if identities != manifest_identities:
            raise ValueError("sample order/identity differs from accepted manifest")


@dataclass(frozen=True, slots=True)
class SplitOverlapRecord:
    kind: SplitOverlapKind
    value_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SplitOverlapKind):
            raise TypeError("split overlap kind must be typed")
        _require_sha256(self.value_sha256, field_name="overlap value_sha256")
        for name, values in (
            ("train_sample_ids", self.train_sample_ids),
            ("validation_sample_ids", self.validation_sample_ids),
        ):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a non-empty immutable tuple")
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique and sorted")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "value_sha256": self.value_sha256,
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class SplitOverlapReport:
    records: tuple[SplitOverlapRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise TypeError("split overlap records must be an immutable tuple")
        if any(not isinstance(record, SplitOverlapRecord) for record in self.records):
            raise TypeError("split overlap report contains an untyped record")
        keys = tuple(
            (
                _SPLIT_OVERLAP_KIND_ORDER[record.kind],
                record.value_sha256,
                record.train_sample_ids,
                record.validation_sample_ids,
            )
            for record in self.records
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("split overlap records must be unique and sorted")

    @property
    def is_disjoint(self) -> bool:
        return not self.records

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": SPLIT_OVERLAP_REPORT_SCHEMA_VERSION,
            "records": [record.canonical_payload() for record in self.records],
        }

    def require_disjoint(self) -> None:
        if self.records:
            kinds = sorted({record.kind.value for record in self.records})
            raise RepresentationDataError(
                "train/validation representation groups overlap for: "
                + ", ".join(kinds)
            )

    def validate_policy(
        self,
        policy: SplitOverlapPolicy,
        *,
        expected_report_sha256: str | None,
    ) -> None:
        """Apply a fail-closed policy without deleting or rewriting any row."""

        if not isinstance(policy, SplitOverlapPolicy):
            raise TypeError("split overlap policy must be typed")
        if policy is SplitOverlapPolicy.REQUIRE_DISJOINT:
            if expected_report_sha256 is not None:
                raise ValueError(
                    "disjoint overlap policy cannot bind an accepted overlap report"
                )
            self.require_disjoint()
            return
        if expected_report_sha256 is None:
            raise ValueError(
                "recorded overlap policy requires an expected report SHA256"
            )
        _require_sha256(
            expected_report_sha256,
            field_name="expected split overlap report SHA256",
        )
        if self.identity_sha256 != expected_report_sha256:
            raise RepresentationDataError(
                "train/validation overlap report differs from the accepted identity"
            )
        disallowed = sorted(
            {record.kind.value for record in self.records}
            - {SplitOverlapKind.IMAGE_PATH.value}
        )
        if disallowed:
            raise RepresentationDataError(
                "recorded image-path overlap policy cannot accept: "
                + ", ".join(disallowed)
            )


def load_retained_representation_jsonl(
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    warn_on_leakage: bool = True,
) -> RepresentationDataset:
    """Load one exact retained JSONL snapshot under the fixed focus transform.

    Relative image references are resolved against the JSONL parent, matching
    the pinned historical data behavior.  A row is retained only when
    ``need_focus is True``, ``trajectory_type == 'single_focus'``, and
    ``evidence_state == 'need_local_visual_evidence'``.  Missing or loosely
    typed focus metadata is excluded instead of inheriting the historical
    truthy/default behavior.
    """

    _require_sha256(expected_source_sha256, field_name="expected_source_sha256")
    if type(warn_on_leakage) is not bool:
        raise TypeError("warn_on_leakage must be a bool")
    try:
        path = Path(source_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RepresentationDataError(
            f"source_path does not resolve to a file: {source_path}"
        ) from exc
    if not path.is_file():
        raise RepresentationDataError(f"source_path is not a file: {path}")
    source_bytes = path.read_bytes()
    actual_sha256 = sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_source_sha256:
        raise RepresentationDataError(
            "source JSONL SHA256 mismatch: "
            f"expected {expected_source_sha256}, got {actual_sha256}"
        )

    lines = source_bytes.splitlines()
    if not lines:
        raise RepresentationDataError("source JSONL is empty")

    samples: list[RepresentationTrainingSample] = []
    accepted: list[AcceptedRowManifestEntry] = []
    excluded: list[ExcludedRowManifestEntry] = []
    duplicates: list[DuplicateRecord] = []
    leakages: list[LeakageRecord] = []
    source_row_first_line: dict[str, int] = {}
    sample_id_first_line: dict[str, int] = {}
    group_target_first_line: dict[tuple[str, str], int] = {}

    for source_line, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise RepresentationDataError(
                f"source JSONL contains a blank row at line {source_line}"
            )
        source_row_sha256 = sha256(raw_line).hexdigest()
        prior_source_line = source_row_first_line.setdefault(
            source_row_sha256, source_line
        )
        if prior_source_line != source_line:
            duplicates.append(
                DuplicateRecord(
                    kind=DuplicateKind.SOURCE_ROW,
                    first_source_line=prior_source_line,
                    duplicate_source_line=source_line,
                    key_sha256=source_row_sha256,
                )
            )

        row = _decode_json_object(raw_line, source_line=source_line)
        sample_id = _optional_non_empty_string(row.get("uid"))

        filter_reasons = _focus_filter_reasons(row)
        if filter_reasons:
            excluded.append(
                ExcludedRowManifestEntry(
                    source_line=source_line,
                    source_row_sha256=source_row_sha256,
                    sample_id=sample_id,
                    reasons=filter_reasons,
                )
            )
            continue

        invalid_reason = _sample_field_validation_reason(row)
        if invalid_reason is not None:
            excluded.append(
                ExcludedRowManifestEntry(
                    source_line=source_line,
                    source_row_sha256=source_row_sha256,
                    sample_id=sample_id,
                    reasons=(invalid_reason,),
                )
            )
            continue

        assert sample_id is not None
        image_reference = row["image"]
        assert isinstance(image_reference, str)
        resolved_image = _resolve_source_image(
            image_reference, source_parent=path.parent
        )
        if resolved_image is None:
            excluded.append(
                ExcludedRowManifestEntry(
                    source_line=source_line,
                    source_row_sha256=source_row_sha256,
                    sample_id=sample_id,
                    reasons=(RowExclusionReason.IMAGE_NOT_FOUND,),
                )
            )
            continue

        optional_fields = {
            name: row[name] if name in row else None for name in _OPTIONAL_SAMPLE_FIELDS
        }
        sample = RepresentationTrainingSample(
            sample_id=sample_id,
            image=str(resolved_image),
            question=row["question"],
            target=row["target"],
            evidence_description=row["evidence_description"],
            short_answer=row["short_answer"],
            choices=_parse_choices(row.get("choices", [])),
            **optional_fields,
        )

        duplicate_reasons: list[RowExclusionReason] = []
        prior_sample_line = sample_id_first_line.get(sample.sample_id)
        if prior_sample_line is not None:
            duplicate_reasons.append(RowExclusionReason.DUPLICATE_SAMPLE_ID)
            duplicates.append(
                DuplicateRecord(
                    kind=DuplicateKind.SAMPLE_ID,
                    first_source_line=prior_sample_line,
                    duplicate_source_line=source_line,
                    key_sha256=_identity_sha256(sample.sample_id),
                )
            )

        group_target_key = (sample.image_group_key, sample.target)
        prior_target_line = group_target_first_line.get(group_target_key)
        if prior_target_line is not None:
            duplicate_reasons.append(RowExclusionReason.DUPLICATE_GROUP_TARGET)
            duplicates.append(
                DuplicateRecord(
                    kind=DuplicateKind.GROUP_TARGET,
                    first_source_line=prior_target_line,
                    duplicate_source_line=source_line,
                    key_sha256=_identity_sha256(*group_target_key),
                )
            )
        if duplicate_reasons:
            excluded.append(
                ExcludedRowManifestEntry(
                    source_line=source_line,
                    source_row_sha256=source_row_sha256,
                    sample_id=sample_id,
                    reasons=tuple(duplicate_reasons),
                )
            )
            continue
        sample_id_first_line[sample.sample_id] = source_line
        group_target_first_line[group_target_key] = source_line

        leakage = _legacy_leakage_record(source_line=source_line, sample=sample)
        if leakage is not None:
            leakages.append(leakage)
            if warn_on_leakage:
                warnings.warn(
                    "representation row "
                    f"{sample.sample_id!r} has target/short_answer term overlap: "
                    + ", ".join(leakage.overlapping_terms),
                    RepresentationDataLeakageWarning,
                    stacklevel=2,
                )

        samples.append(sample)
        accepted.append(
            AcceptedRowManifestEntry(
                source_line=source_line,
                source_row_sha256=source_row_sha256,
                source_image_reference=image_reference,
                resolved_image_path=str(resolved_image),
                sample=sample.identity,
            )
        )

    manifest = RepresentationDataManifest(
        schema_version=REPRESENTATION_DATA_MANIFEST_SCHEMA_VERSION,
        transform_version=REPRESENTATION_DATA_TRANSFORM_VERSION,
        source_path=str(path),
        source_sha256=actual_sha256,
        accepted_rows=tuple(accepted),
        excluded_rows=tuple(excluded),
        duplicate_records=tuple(duplicates),
        leakage_records=tuple(leakages),
    )
    return RepresentationDataset(samples=tuple(samples), manifest=manifest)


def train_validation_group_overlap(
    train: Sequence[RepresentationTrainingSample],
    validation: Sequence[RepresentationTrainingSample],
) -> SplitOverlapReport:
    """Audit the four exact manifest-level split keys recorded by the project."""

    if not train or not validation:
        raise ValueError("train and validation samples must both be non-empty")
    _require_sample_sequence(train, field_name="train")
    _require_sample_sequence(validation, field_name="validation")

    records: list[SplitOverlapRecord] = []
    accessors = (
        (SplitOverlapKind.IMAGE_GROUP_KEY, lambda sample: sample.image_group_key),
        (SplitOverlapKind.IMAGE_PATH, lambda sample: sample.image),
        (SplitOverlapKind.STABLE_IMAGE_UID, lambda sample: sample.stable_image_uid),
        (SplitOverlapKind.ITEM_CONTENT_HASH, lambda sample: sample.item_content_hash),
    )
    for kind, accessor in accessors:
        train_values = _samples_by_value(train, accessor)
        validation_values = _samples_by_value(validation, accessor)
        for value in sorted(set(train_values) & set(validation_values)):
            records.append(
                SplitOverlapRecord(
                    kind=kind,
                    value_sha256=_identity_sha256(value),
                    train_sample_ids=tuple(sorted(train_values[value])),
                    validation_sample_ids=tuple(sorted(validation_values[value])),
                )
            )
    records.sort(
        key=lambda record: (
            _SPLIT_OVERLAP_KIND_ORDER[record.kind],
            record.value_sha256,
            record.train_sample_ids,
            record.validation_sample_ids,
        )
    )
    return SplitOverlapReport(records=tuple(records))


def _focus_filter_reasons(
    row: Mapping[str, object],
) -> tuple[RowExclusionReason, ...]:
    need_focus = row.get("need_focus")
    trajectory_type = row.get("trajectory_type")
    evidence_state = row.get("evidence_state")
    if (
        type(need_focus) is not bool
        or not isinstance(trajectory_type, str)
        or not isinstance(evidence_state, str)
    ):
        return (RowExclusionReason.INVALID_FOCUS_METADATA,)
    if (
        not need_focus
        or trajectory_type != _FOCUS_TRAJECTORY_TYPE
        or evidence_state != _FOCUS_EVIDENCE_STATE
    ):
        return (RowExclusionReason.NOT_FOCUS_ROW,)
    return ()


def _sample_field_validation_reason(
    row: Mapping[str, object],
) -> RowExclusionReason | None:
    for name in (
        "uid",
        "image",
        "question",
        "target",
        "evidence_description",
        "short_answer",
    ):
        if _optional_non_empty_string(row.get(name)) is None:
            return RowExclusionReason.INVALID_REQUIRED_FIELD
    for name in _OPTIONAL_SAMPLE_FIELDS:
        if (
            name in row
            and row[name] is not None
            and _optional_non_empty_string(row[name]) is None
        ):
            return RowExclusionReason.INVALID_OPTIONAL_FIELD
    if "choices" in row and not _valid_choices(row["choices"]):
        return RowExclusionReason.INVALID_OPTIONAL_FIELD
    return None


def _valid_choices(value: object) -> bool:
    if not isinstance(value, list):
        return False
    labels: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "text"}:
            return False
        label = _optional_non_empty_string(item["label"])
        text = _optional_non_empty_string(item["text"])
        if label is None or text is None or label in labels:
            return False
        labels.add(label)
    return True


def _parse_choices(value: object) -> tuple[RepresentationChoice, ...]:
    """Parse choices after ``_sample_field_validation_reason`` accepted them."""

    assert isinstance(value, list)
    return tuple(
        RepresentationChoice(label=item["label"], text=item["text"])
        for item in value
        if isinstance(item, dict)
    )


def _resolve_source_image(image_reference: str, *, source_parent: Path) -> Path | None:
    if "\x00" in image_reference:
        return None
    candidate = Path(image_reference)
    if not candidate.is_absolute():
        candidate = source_parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return resolved if resolved.is_file() else None


def _decode_json_object(raw_line: bytes, *, source_line: int) -> dict[str, object]:
    try:
        text = raw_line.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RepresentationDataError) as exc:
        raise RepresentationDataError(
            f"line {source_line}: invalid JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RepresentationDataError(f"line {source_line}: JSON row must be an object")
    return value


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RepresentationDataError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise RepresentationDataError(f"non-finite JSON constant {value!r}")


def _legacy_terms(text: str) -> frozenset[str]:
    return frozenset(
        term.lower()
        for term in text.replace("/", " ").replace("-", " ").split()
        if len(term) > 2
    )


def _legacy_leakage_record(
    *,
    source_line: int,
    sample: RepresentationTrainingSample,
) -> LeakageRecord | None:
    overlap = tuple(
        sorted(_legacy_terms(sample.target) & _legacy_terms(sample.short_answer))
    )
    if not overlap:
        return None
    return LeakageRecord(
        source_line=source_line,
        sample_id=sample.sample_id,
        overlapping_terms=overlap,
    )


def _optional_non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and bool(value.strip()) else None


def _identity_sha256(*values: str) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _require_sample_sequence(
    samples: Sequence[RepresentationTrainingSample], *, field_name: str
) -> None:
    if not all(isinstance(sample, RepresentationTrainingSample) for sample in samples):
        raise TypeError(
            f"{field_name} must contain only RepresentationTrainingSample values"
        )


def _samples_by_value(
    samples: Sequence[RepresentationTrainingSample],
    accessor: Callable[[RepresentationTrainingSample], str | None],
) -> dict[str, set[str]]:
    by_value: dict[str, set[str]] = {}
    for sample in samples:
        value = accessor(sample)
        if value is not None:
            by_value.setdefault(value, set()).add(sample.sample_id)
    return by_value
