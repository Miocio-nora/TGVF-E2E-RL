"""Strict semantic-bound answer-bearing evidence-span sidecars for RP70.

The upstream teacher data does not contain authoritative evidence character
spans.  RP70 therefore consumes an explicit audited sidecar instead of deriving
spans at training time.  One header binds the sidecar to the complete set of
supervision-semantic rows, followed by exactly one record for every retained
UID.  Missing, extra, duplicated, unresolved, or semantically drifted records
are rejected.  Row order, source-line bookkeeping, image identity, and
non-model provenance are deliberately outside the annotation binding so that
RP67's authorized donor-image branch can reuse the same sparse supervision.

All offsets are Python Unicode code-point, half-open offsets into the exact
``evidence_description`` string.  There is intentionally no literal, casefold,
normalization, fuzzy, or semantic runtime matcher in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from tgvf_rl.representation.training.data import RepresentationDataset


ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION = "answer_bearing_span_sidecar_v2"
ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION = "answer_bearing_span_index_v3"
ANSWER_BEARING_SPAN_INDEX_SET_SCHEMA_VERSION = "answer_bearing_span_index_set_v3"
ANSWER_BEARING_SPAN_MATCH_POLICY = (
    "explicit_semantic_bound_unicode_codepoint_half_open_offsets_only_v2"
)
ANSWER_BEARING_SPAN_POPULATION_SCHEMA_VERSION = (
    "answer_bearing_span_retained_semantic_population_v2"
)
ANSWER_BEARING_SPAN_SEMANTIC_SCHEMA_VERSION = (
    "answer_bearing_span_supervision_semantics_v1"
)
VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON = (
    "verified_no_answer_bearing_evidence_in_bound_evidence_description_v1"
)

_HEADER_RECORD_TYPE = "header"
_SAMPLE_RECORD_TYPE = "sample"
_HEADER_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "policy",
        "retained_semantic_population_sha256",
        "retained_count",
        "status_statistics",
        "annotator_identity",
    }
)
_STATISTICS_FIELDS = frozenset(
    {
        "total_rows",
        "resolved_rows",
        "verified_no_answer_bearing_evidence_rows",
        "multiple_span_rows",
        "total_spans",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "record_type",
        "uid",
        "semantic_content_sha256",
        "question_sha256",
        "target_sha256",
        "evidence_description_sha256",
        "short_answer_sha256",
        "choices_sha256",
        "status",
        "reason",
        "spans",
    }
)
_SPAN_FIELDS = frozenset({"start", "end", "exact_text"})


class AnswerBearingSpanDataError(ValueError):
    """A sidecar cannot be proven identical to its semantic population."""


class AnswerBearingSpanStatus(str, Enum):
    """The only two fully audited states accepted by the production loader."""

    RESOLVED = "resolved"
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE = "verified_no_answer_bearing_evidence"


@dataclass(frozen=True, slots=True, order=True)
class EvidenceCharacterSpan:
    """One explicit Unicode code-point half-open evidence span."""

    start: int
    end: int
    exact_text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("evidence spans must satisfy 0 <= start < end")
        _require_non_empty_text(self.exact_text, field_name="span.exact_text")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "exact_text": self.exact_text,
        }


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanRecord:
    """One sidecar record bound to one supervision-semantic sample."""

    uid: str
    semantic_content_sha256: str
    question_sha256: str
    target_sha256: str
    evidence_description_sha256: str
    short_answer_sha256: str
    choices_sha256: str
    status: AnswerBearingSpanStatus
    reason: str | None
    value_character_spans: tuple[EvidenceCharacterSpan, ...]
    evidence_description: str = field(repr=False)
    short_answer: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.uid, field_name="uid")
        _require_non_empty_text(
            self.evidence_description,
            field_name="evidence_description",
        )
        _require_non_empty_text(self.short_answer, field_name="short_answer")
        for value, name in (
            (self.semantic_content_sha256, "semantic_content_sha256"),
            (self.question_sha256, "question_sha256"),
            (self.target_sha256, "target_sha256"),
            (
                self.evidence_description_sha256,
                "evidence_description_sha256",
            ),
            (self.short_answer_sha256, "short_answer_sha256"),
            (self.choices_sha256, "choices_sha256"),
        ):
            _require_sha256(value, field_name=name)
        if self.evidence_description_sha256 != _text_sha256(self.evidence_description):
            raise ValueError("evidence_description SHA256 differs from bound text")
        if self.short_answer_sha256 != _text_sha256(self.short_answer):
            raise ValueError("short_answer SHA256 differs from bound text")
        if not isinstance(self.status, AnswerBearingSpanStatus):
            raise TypeError("answer-bearing sidecar status must be typed")
        if not isinstance(self.value_character_spans, tuple) or any(
            not isinstance(span, EvidenceCharacterSpan)
            for span in self.value_character_spans
        ):
            raise TypeError("value_character_spans must be an immutable typed tuple")
        if self.value_character_spans != tuple(sorted(self.value_character_spans)):
            raise ValueError("evidence spans must be sorted and unique")
        if len(self.value_character_spans) != len(set(self.value_character_spans)):
            raise ValueError("evidence spans must be sorted and unique")
        prior_end = 0
        for index, span in enumerate(self.value_character_spans):
            if index and span.start < prior_end:
                raise ValueError("evidence spans must not overlap")
            if span.end > len(self.evidence_description):
                raise ValueError("evidence span lies outside evidence_description")
            if self.evidence_description[span.start : span.end] != span.exact_text:
                raise ValueError("evidence span exact_text differs from bound evidence")
            prior_end = span.end
        if self.status is AnswerBearingSpanStatus.RESOLVED:
            if not self.value_character_spans:
                raise ValueError("resolved sidecar records require non-empty spans")
            if self.reason is not None:
                raise ValueError("resolved sidecar records require reason=null")
        else:
            if self.value_character_spans:
                raise ValueError(
                    "verified-no-evidence sidecar records require empty spans"
                )
            if self.reason != VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON:
                raise ValueError(
                    "verified-no-evidence sidecar record has an invalid reason"
                )

    @property
    def matched(self) -> bool:
        """Compatibility alias for an explicitly resolved record."""

        return self.status is AnswerBearingSpanStatus.RESOLVED

    @property
    def multiple(self) -> bool:
        return len(self.value_character_spans) > 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_type": _SAMPLE_RECORD_TYPE,
            "uid": self.uid,
            "semantic_content_sha256": self.semantic_content_sha256,
            "question_sha256": self.question_sha256,
            "target_sha256": self.target_sha256,
            "evidence_description_sha256": self.evidence_description_sha256,
            "short_answer_sha256": self.short_answer_sha256,
            "choices_sha256": self.choices_sha256,
            "status": self.status.value,
            "reason": self.reason,
            "spans": [span.canonical_payload() for span in self.value_character_spans],
        }


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanStatistics:
    """Complete resolved/verified population accounting for one sidecar."""

    total_rows: int
    resolved_rows: int
    verified_no_answer_bearing_evidence_rows: int
    multiple_span_rows: int
    total_spans: int

    def __post_init__(self) -> None:
        for field_name in _STATISTICS_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.total_rows <= 0:
            raise ValueError("answer-bearing span statistics require at least one row")
        if (
            self.resolved_rows + self.verified_no_answer_bearing_evidence_rows
            != self.total_rows
        ):
            raise ValueError("sidecar statuses must partition the retained population")
        if self.multiple_span_rows > self.resolved_rows:
            raise ValueError("multiple-span rows must be resolved rows")
        if self.total_spans < self.resolved_rows + self.multiple_span_rows:
            raise ValueError("total span count is inconsistent with resolved rows")

    @property
    def matched_rows(self) -> int:
        """Compatibility alias for resolved rows."""

        return self.resolved_rows

    @property
    def unmatched_rows(self) -> int:
        """Compatibility alias for explicitly verified empty evidence rows."""

        return self.verified_no_answer_bearing_evidence_rows

    @property
    def multiple_rows(self) -> int:
        return self.multiple_span_rows

    @property
    def matched_occurrences(self) -> int:
        return self.total_spans

    @property
    def coverage(self) -> float:
        return self.resolved_rows / self.total_rows

    def canonical_payload(self) -> dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "resolved_rows": self.resolved_rows,
            "verified_no_answer_bearing_evidence_rows": (
                self.verified_no_answer_bearing_evidence_rows
            ),
            "multiple_span_rows": self.multiple_span_rows,
            "total_spans": self.total_spans,
        }


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanSidecarHeader:
    """Canonical first line of a semantic-bound RP70 sidecar."""

    retained_semantic_population_sha256: str
    retained_count: int
    statistics: AnswerBearingSpanStatistics
    annotator_identity: str
    schema_version: str = ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION
    policy: str = ANSWER_BEARING_SPAN_MATCH_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION:
            raise ValueError("answer-bearing sidecar schema mismatch")
        if self.policy != ANSWER_BEARING_SPAN_MATCH_POLICY:
            raise ValueError("answer-bearing sidecar policy mismatch")
        _require_sha256(
            self.retained_semantic_population_sha256,
            field_name="retained_semantic_population_sha256",
        )
        if (
            isinstance(self.retained_count, bool)
            or not isinstance(self.retained_count, int)
            or self.retained_count <= 0
        ):
            raise ValueError("retained_count must be a positive integer")
        if not isinstance(self.statistics, AnswerBearingSpanStatistics):
            raise TypeError("sidecar statistics must be typed")
        if self.statistics.total_rows != self.retained_count:
            raise ValueError("sidecar status statistics disagree with retained_count")
        _require_non_empty_text(
            self.annotator_identity,
            field_name="annotator_identity",
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_type": _HEADER_RECORD_TYPE,
            "schema_version": self.schema_version,
            "policy": self.policy,
            "retained_semantic_population_sha256": (
                self.retained_semantic_population_sha256
            ),
            "retained_count": self.retained_count,
            "status_statistics": self.statistics.canonical_payload(),
            "annotator_identity": self.annotator_identity,
        }


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanIndex:
    """One fully verified sidecar and its semantic retained population."""

    source_path: str
    source_sha256: str
    sidecar_path: str
    sidecar_sha256: str
    retained_semantic_population_sha256: str
    annotator_identity: str
    records: tuple[AnswerBearingSpanRecord, ...]
    statistics: AnswerBearingSpanStatistics
    header: AnswerBearingSpanSidecarHeader
    schema_version: str = ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION
    match_policy: str = ANSWER_BEARING_SPAN_MATCH_POLICY
    _by_uid: Mapping[str, AnswerBearingSpanRecord] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.source_path, field_name="source_path")
        _require_non_empty_text(self.sidecar_path, field_name="sidecar_path")
        for value, name in (
            (self.source_sha256, "source_sha256"),
            (self.sidecar_sha256, "sidecar_sha256"),
            (
                self.retained_semantic_population_sha256,
                "retained_semantic_population_sha256",
            ),
        ):
            _require_sha256(value, field_name=name)
        _require_non_empty_text(
            self.annotator_identity,
            field_name="annotator_identity",
        )
        if self.schema_version != ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION:
            raise ValueError("answer-bearing span index schema mismatch")
        if self.match_policy != ANSWER_BEARING_SPAN_MATCH_POLICY:
            raise ValueError("answer-bearing span index policy mismatch")
        if not isinstance(self.header, AnswerBearingSpanSidecarHeader):
            raise TypeError("answer-bearing span index header must be typed")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("answer-bearing span index requires records")
        if any(
            not isinstance(record, AnswerBearingSpanRecord) for record in self.records
        ):
            raise TypeError("answer-bearing span index contains an invalid record")
        uids = tuple(record.uid for record in self.records)
        if len(set(uids)) != len(uids):
            raise ValueError("answer-bearing span UIDs must be unique")
        if self.statistics != _statistics_for(self.records):
            raise ValueError("answer-bearing span statistics differ from records")
        expected_population = _population_sha256_from_records(self.records)
        if self.retained_semantic_population_sha256 != expected_population:
            raise ValueError("retained semantic population digest differs from records")
        if (
            self.header.retained_semantic_population_sha256
            != self.retained_semantic_population_sha256
        ):
            raise ValueError("sidecar header differs from index semantic binding")
        if self.header.retained_count != len(self.records):
            raise ValueError("sidecar header retained_count differs from records")
        if self.header.statistics != self.statistics:
            raise ValueError("sidecar header statistics differ from records")
        if self.header.annotator_identity != self.annotator_identity:
            raise ValueError("sidecar header annotator differs from index")
        object.__setattr__(
            self,
            "_by_uid",
            MappingProxyType({record.uid: record for record in self.records}),
        )
        object.__setattr__(self, "_identity_sha256", _index_identity_sha256(self))

    @property
    def by_uid(self) -> Mapping[str, AnswerBearingSpanRecord]:
        return self._by_uid

    def record_for(self, uid: str) -> AnswerBearingSpanRecord:
        _require_non_empty_text(uid, field_name="uid")
        try:
            return self.by_uid[uid]
        except KeyError as error:
            raise KeyError(f"answer-bearing span index has no UID {uid!r}") from error

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256


@dataclass(frozen=True, slots=True)
class AnswerBearingSpanIndexSet:
    """Union of complete semantic-bound split sidecars."""

    indices: tuple[AnswerBearingSpanIndex, ...]
    schema_version: str = ANSWER_BEARING_SPAN_INDEX_SET_SCHEMA_VERSION
    match_policy: str = ANSWER_BEARING_SPAN_MATCH_POLICY
    _records: tuple[AnswerBearingSpanRecord, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _statistics: AnswerBearingSpanStatistics = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_uid: Mapping[str, AnswerBearingSpanRecord] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_BEARING_SPAN_INDEX_SET_SCHEMA_VERSION:
            raise ValueError("answer-bearing span index-set schema mismatch")
        if self.match_policy != ANSWER_BEARING_SPAN_MATCH_POLICY:
            raise ValueError("answer-bearing span index-set policy mismatch")
        if not isinstance(self.indices, tuple) or not self.indices:
            raise ValueError("answer-bearing span index set cannot be empty")
        if any(not isinstance(index, AnswerBearingSpanIndex) for index in self.indices):
            raise TypeError("answer-bearing span index set contains an invalid index")
        child_identities = tuple(index.identity_sha256 for index in self.indices)
        if len(set(child_identities)) != len(child_identities):
            raise ValueError("answer-bearing span index set repeats a source index")
        records = tuple(record for index in self.indices for record in index.records)
        uids = tuple(record.uid for record in records)
        if len(set(uids)) != len(uids):
            raise ValueError(
                "answer-bearing span UID occurs in more than one source index"
            )
        statistics = _statistics_for(records)
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_statistics", statistics)
        object.__setattr__(
            self,
            "_by_uid",
            MappingProxyType({record.uid: record for record in records}),
        )
        object.__setattr__(self, "_identity_sha256", _index_set_identity_sha256(self))

    @property
    def records(self) -> tuple[AnswerBearingSpanRecord, ...]:
        return self._records

    @property
    def statistics(self) -> AnswerBearingSpanStatistics:
        return self._statistics

    @property
    def by_uid(self) -> Mapping[str, AnswerBearingSpanRecord]:
        return self._by_uid

    def record_for(self, uid: str) -> AnswerBearingSpanRecord:
        _require_non_empty_text(uid, field_name="uid")
        try:
            return self.by_uid[uid]
        except KeyError as error:
            raise KeyError(
                f"answer-bearing span index set has no UID {uid!r}"
            ) from error

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256


def answer_bearing_span_semantic_sha256(sample: object) -> str:
    """Bind fields that determine RP70 prompt, target, transcript, and labels.

    Visual identity and dataset provenance are intentionally excluded.  RP67
    changes only those excluded fields when it constructs a donor-image branch.
    """

    return _canonical_sha256(
        {
            "schema_version": ANSWER_BEARING_SPAN_SEMANTIC_SCHEMA_VERSION,
            "uid": getattr(sample, "sample_id"),
            "question": getattr(sample, "question"),
            "target": getattr(sample, "target"),
            "evidence_description": getattr(sample, "evidence_description"),
            "short_answer": getattr(sample, "short_answer"),
            "choices": _choices_payload(sample),
        }
    )


def answer_bearing_span_population_sha256(dataset: RepresentationDataset) -> str:
    """Digest the complete retained supervision semantics, independent of order."""

    if not isinstance(dataset, RepresentationDataset):
        raise TypeError("answer-bearing span population requires RepresentationDataset")
    payload = {
        "schema_version": ANSWER_BEARING_SPAN_POPULATION_SCHEMA_VERSION,
        "records": sorted(
            (
                _population_record_payload_for_sample(sample)
                for sample in dataset.samples
            ),
            key=lambda row: str(row["uid"]),
        ),
    }
    return _canonical_sha256(payload)


def _population_record_payload_for_sample(sample: object) -> dict[str, object]:
    return _population_record_payload(
        getattr(sample, "sample_id"),
        answer_bearing_span_semantic_sha256(sample),
        _text_sha256(getattr(sample, "question")),
        _text_sha256(getattr(sample, "target")),
        _text_sha256(getattr(sample, "evidence_description")),
        _text_sha256(getattr(sample, "short_answer")),
        _choices_sha256(sample),
    )


def render_answer_bearing_span_sidecar(
    dataset: RepresentationDataset,
    annotations: tuple[
        tuple[
            AnswerBearingSpanStatus | str,
            str | None,
            tuple[EvidenceCharacterSpan, ...],
        ],
        ...,
    ],
    *,
    annotator_identity: str,
) -> bytes:
    """Render canonical JSONL bytes from a complete ordered annotation tuple.

    Each annotation is ``(status, reason, spans)`` for the sample at the same
    tuple position.  This helper is pure: it neither reads nor writes a path.  The
    returned bytes still must be SHA-bound and reloaded through
    :func:`load_answer_bearing_span_index` before production use.
    """

    if not isinstance(dataset, RepresentationDataset):
        raise TypeError("sidecar rendering requires RepresentationDataset")
    if not isinstance(annotations, tuple):
        raise TypeError("sidecar annotations must be an immutable tuple")
    if len(annotations) != len(dataset.samples):
        raise ValueError(
            "sidecar annotations must contain exactly one item per retained sample"
        )
    _require_non_empty_text(annotator_identity, field_name="annotator_identity")
    records: list[AnswerBearingSpanRecord] = []
    for ordinal, (sample, annotation) in enumerate(
        zip(dataset.samples, annotations, strict=True)
    ):
        if not isinstance(annotation, tuple) or len(annotation) != 3:
            raise TypeError(
                "each sidecar annotation must be a (status, reason, spans) tuple"
            )
        status_raw, reason, spans = annotation
        try:
            status = AnswerBearingSpanStatus(status_raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"annotation {ordinal} has an unsupported or unresolved status"
            ) from error
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"annotation {ordinal} reason must be string or None")
        if not isinstance(spans, tuple) or any(
            not isinstance(span, EvidenceCharacterSpan) for span in spans
        ):
            raise TypeError(
                f"annotation {ordinal} spans must be an immutable typed tuple"
            )
        records.append(
            AnswerBearingSpanRecord(
                uid=sample.sample_id,
                semantic_content_sha256=(answer_bearing_span_semantic_sha256(sample)),
                question_sha256=_text_sha256(sample.question),
                target_sha256=_text_sha256(sample.target),
                evidence_description_sha256=_text_sha256(sample.evidence_description),
                short_answer_sha256=_text_sha256(sample.short_answer),
                choices_sha256=_choices_sha256(sample),
                status=status,
                reason=reason,
                value_character_spans=spans,
                evidence_description=sample.evidence_description,
                short_answer=sample.short_answer,
            )
        )
    materialized = tuple(records)
    statistics = _statistics_for(materialized)
    header = AnswerBearingSpanSidecarHeader(
        retained_semantic_population_sha256=(
            answer_bearing_span_population_sha256(dataset)
        ),
        retained_count=len(dataset.samples),
        statistics=statistics,
        annotator_identity=annotator_identity,
    )
    rows = (header.canonical_payload(),) + tuple(
        record.canonical_payload() for record in materialized
    )
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def load_answer_bearing_span_index(
    dataset: RepresentationDataset,
    sidecar_path: str | Path,
    *,
    expected_sidecar_sha256: str,
) -> AnswerBearingSpanIndex:
    """Load one complete sidecar against one already-retained dataset."""

    if not isinstance(dataset, RepresentationDataset):
        raise TypeError("answer-bearing span loader requires RepresentationDataset")
    _require_sha256(
        expected_sidecar_sha256,
        field_name="expected_sidecar_sha256",
    )
    try:
        path = Path(sidecar_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise AnswerBearingSpanDataError(
            f"sidecar_path does not resolve to a file: {sidecar_path}"
        ) from error
    if not path.is_file():
        raise AnswerBearingSpanDataError(f"sidecar_path is not a file: {path}")
    sidecar_bytes = path.read_bytes()
    actual_sidecar_sha256 = sha256(sidecar_bytes).hexdigest()
    if actual_sidecar_sha256 != expected_sidecar_sha256:
        raise AnswerBearingSpanDataError(
            "sidecar SHA256 mismatch: "
            f"expected {expected_sidecar_sha256}, got {actual_sidecar_sha256}"
        )
    lines = sidecar_bytes.splitlines()
    expected_line_count = len(dataset.samples) + 1
    if len(lines) != expected_line_count:
        raise AnswerBearingSpanDataError(
            "sidecar must contain one header plus exactly one record per retained "
            f"sample: expected {expected_line_count} lines, got {len(lines)}"
        )
    if any(not raw_line.strip() for raw_line in lines):
        raise AnswerBearingSpanDataError("sidecar JSONL contains a blank row")

    header = _parse_header(lines[0], sidecar_line=1)
    population_sha256 = answer_bearing_span_population_sha256(dataset)
    if header.retained_semantic_population_sha256 != population_sha256:
        raise AnswerBearingSpanDataError(
            "sidecar retained semantic population digest differs from dataset"
        )
    if header.retained_count != len(dataset.samples):
        raise AnswerBearingSpanDataError(
            "sidecar retained_count differs from retained dataset population"
        )

    sample_by_uid = {sample.sample_id: sample for sample in dataset.samples}
    if len(sample_by_uid) != len(dataset.samples):
        raise AnswerBearingSpanDataError("retained dataset contains duplicate UIDs")
    records_by_uid: dict[str, AnswerBearingSpanRecord] = {}
    for line_offset, raw_line in enumerate(lines[1:], start=2):
        row = _decode_json_object(raw_line, sidecar_line=line_offset)
        uid_value = row.get("uid")
        if not isinstance(uid_value, str) or not uid_value.strip():
            raise AnswerBearingSpanDataError(
                f"sidecar line {line_offset}: uid must be a non-empty string"
            )
        if uid_value in records_by_uid:
            raise AnswerBearingSpanDataError(
                f"sidecar contains duplicate UID {uid_value!r}"
            )
        try:
            sample = sample_by_uid[uid_value]
        except KeyError as error:
            raise AnswerBearingSpanDataError(
                f"sidecar contains unknown UID {uid_value!r}"
            ) from error
        record = _parse_record(
            raw_line,
            sidecar_line=line_offset,
            sample=sample,
        )
        _validate_record_binding(record, sample=sample)
        records_by_uid[record.uid] = record

    missing_uids = sorted(set(sample_by_uid) - set(records_by_uid))
    if missing_uids:
        raise AnswerBearingSpanDataError(
            f"sidecar is missing retained UIDs: {missing_uids}"
        )

    # Canonicalize independently of both source-row and sidecar-row order.
    materialized = tuple(records_by_uid[uid] for uid in sorted(records_by_uid))
    statistics = _statistics_for(materialized)
    if header.statistics != statistics:
        raise AnswerBearingSpanDataError(
            "sidecar header status statistics differ from sample records"
        )
    try:
        return AnswerBearingSpanIndex(
            source_path=dataset.manifest.source_path,
            source_sha256=dataset.manifest.source_sha256,
            sidecar_path=str(path),
            sidecar_sha256=actual_sidecar_sha256,
            retained_semantic_population_sha256=population_sha256,
            annotator_identity=header.annotator_identity,
            records=materialized,
            statistics=statistics,
            header=header,
        )
    except (TypeError, ValueError) as error:
        raise AnswerBearingSpanDataError(
            f"validated sidecar index is internally inconsistent: {error}"
        ) from error


def merge_answer_bearing_span_indices(
    *indices: AnswerBearingSpanIndex,
) -> AnswerBearingSpanIndexSet:
    """Merge complete split indices without changing either population."""

    return AnswerBearingSpanIndexSet(indices=tuple(indices))


def _parse_header(
    raw_line: bytes, *, sidecar_line: int
) -> AnswerBearingSpanSidecarHeader:
    row = _decode_json_object(raw_line, sidecar_line=sidecar_line)
    _require_exact_fields(row, _HEADER_FIELDS, context="sidecar header")
    if row["record_type"] != _HEADER_RECORD_TYPE:
        raise AnswerBearingSpanDataError("sidecar first row must be record_type=header")
    statistics_raw = row["status_statistics"]
    if not isinstance(statistics_raw, dict):
        raise AnswerBearingSpanDataError(
            "sidecar header status_statistics must be an object"
        )
    _require_exact_fields(
        statistics_raw,
        _STATISTICS_FIELDS,
        context="sidecar header status_statistics",
    )
    try:
        statistics = AnswerBearingSpanStatistics(
            **{
                name: _explicit_int(statistics_raw[name], field_name=name)
                for name in _STATISTICS_FIELDS
            }
        )
        return AnswerBearingSpanSidecarHeader(
            schema_version=_explicit_text(
                row["schema_version"],
                field_name="schema_version",
            ),
            policy=_explicit_text(row["policy"], field_name="policy"),
            retained_semantic_population_sha256=_explicit_text(
                row["retained_semantic_population_sha256"],
                field_name="retained_semantic_population_sha256",
            ),
            retained_count=_explicit_int(
                row["retained_count"],
                field_name="retained_count",
            ),
            statistics=statistics,
            annotator_identity=_explicit_text(
                row["annotator_identity"],
                field_name="annotator_identity",
            ),
        )
    except (TypeError, ValueError) as error:
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: invalid header: {error}"
        ) from error


def _parse_record(
    raw_line: bytes,
    *,
    sidecar_line: int,
    sample: object,
) -> AnswerBearingSpanRecord:
    row = _decode_json_object(raw_line, sidecar_line=sidecar_line)
    _require_exact_fields(row, _RECORD_FIELDS, context=f"sidecar line {sidecar_line}")
    if row["record_type"] != _SAMPLE_RECORD_TYPE:
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: record_type must be sample"
        )
    status_raw = row["status"]
    try:
        status = AnswerBearingSpanStatus(status_raw)
    except (TypeError, ValueError) as error:
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: unsupported or unresolved status "
            f"{status_raw!r}"
        ) from error
    spans_raw = row["spans"]
    if not isinstance(spans_raw, list):
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: spans must be an array"
        )
    spans: list[EvidenceCharacterSpan] = []
    for span_index, span_raw in enumerate(spans_raw):
        if not isinstance(span_raw, dict):
            raise AnswerBearingSpanDataError(
                f"sidecar line {sidecar_line}: span {span_index} must be an object"
            )
        _require_exact_fields(
            span_raw,
            _SPAN_FIELDS,
            context=f"sidecar line {sidecar_line} span {span_index}",
        )
        try:
            spans.append(
                EvidenceCharacterSpan(
                    start=_explicit_int(
                        span_raw["start"],
                        field_name="span.start",
                    ),
                    end=_explicit_int(
                        span_raw["end"],
                        field_name="span.end",
                    ),
                    exact_text=_explicit_text(
                        span_raw["exact_text"],
                        field_name="span.exact_text",
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise AnswerBearingSpanDataError(
                f"sidecar line {sidecar_line}: invalid span {span_index}: {error}"
            ) from error
    reason = row["reason"]
    if reason is not None and not isinstance(reason, str):
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: reason must be string or null"
        )
    # ``sample`` is kept generic only to keep this parser private; the public
    # loader already proves that it came from a RepresentationDataset.
    evidence_description = getattr(sample, "evidence_description")
    short_answer = getattr(sample, "short_answer")
    try:
        return AnswerBearingSpanRecord(
            uid=_explicit_text(row["uid"], field_name="uid"),
            semantic_content_sha256=_explicit_text(
                row["semantic_content_sha256"],
                field_name="semantic_content_sha256",
            ),
            question_sha256=_explicit_text(
                row["question_sha256"],
                field_name="question_sha256",
            ),
            target_sha256=_explicit_text(
                row["target_sha256"],
                field_name="target_sha256",
            ),
            evidence_description_sha256=_explicit_text(
                row["evidence_description_sha256"],
                field_name="evidence_description_sha256",
            ),
            short_answer_sha256=_explicit_text(
                row["short_answer_sha256"],
                field_name="short_answer_sha256",
            ),
            choices_sha256=_explicit_text(
                row["choices_sha256"],
                field_name="choices_sha256",
            ),
            status=status,
            reason=reason,
            value_character_spans=tuple(spans),
            evidence_description=evidence_description,
            short_answer=short_answer,
        )
    except (TypeError, ValueError) as error:
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: invalid sample record: {error}"
        ) from error


def _validate_record_binding(
    record: AnswerBearingSpanRecord,
    *,
    sample: object,
) -> None:
    expected = {
        "uid": getattr(sample, "sample_id"),
        "semantic_content_sha256": answer_bearing_span_semantic_sha256(sample),
        "question_sha256": _text_sha256(getattr(sample, "question")),
        "target_sha256": _text_sha256(getattr(sample, "target")),
        "evidence_description_sha256": _text_sha256(
            getattr(sample, "evidence_description")
        ),
        "short_answer_sha256": _text_sha256(getattr(sample, "short_answer")),
        "choices_sha256": _choices_sha256(sample),
    }
    for field_name, expected_value in expected.items():
        if getattr(record, field_name) != expected_value:
            raise AnswerBearingSpanDataError(
                f"sidecar record for UID {record.uid!r} has semantic drift: "
                f"{field_name}"
            )


def _statistics_for(
    records: tuple[AnswerBearingSpanRecord, ...],
) -> AnswerBearingSpanStatistics:
    return AnswerBearingSpanStatistics(
        total_rows=len(records),
        resolved_rows=sum(
            record.status is AnswerBearingSpanStatus.RESOLVED for record in records
        ),
        verified_no_answer_bearing_evidence_rows=sum(
            record.status is AnswerBearingSpanStatus.VERIFIED_NO_ANSWER_BEARING_EVIDENCE
            for record in records
        ),
        multiple_span_rows=sum(record.multiple for record in records),
        total_spans=sum(len(record.value_character_spans) for record in records),
    )


def _population_record_payload(
    uid: str,
    semantic_content_sha256: str,
    question_sha256: str,
    target_sha256: str,
    evidence_description_sha256: str,
    short_answer_sha256: str,
    choices_sha256: str,
) -> dict[str, object]:
    return {
        "uid": uid,
        "semantic_content_sha256": semantic_content_sha256,
        "question_sha256": question_sha256,
        "target_sha256": target_sha256,
        "evidence_description_sha256": evidence_description_sha256,
        "short_answer_sha256": short_answer_sha256,
        "choices_sha256": choices_sha256,
    }


def _population_sha256_from_records(
    records: tuple[AnswerBearingSpanRecord, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": ANSWER_BEARING_SPAN_POPULATION_SCHEMA_VERSION,
            "records": sorted(
                (
                    _population_record_payload(
                        record.uid,
                        record.semantic_content_sha256,
                        record.question_sha256,
                        record.target_sha256,
                        record.evidence_description_sha256,
                        record.short_answer_sha256,
                        record.choices_sha256,
                    )
                    for record in records
                ),
                key=lambda row: str(row["uid"]),
            ),
        }
    )


def _index_identity_sha256(index: AnswerBearingSpanIndex) -> str:
    return _canonical_sha256(
        {
            "schema_version": index.schema_version,
            "policy": index.match_policy,
            "sidecar_sha256": index.sidecar_sha256,
            "header": index.header.canonical_payload(),
            "records": [record.canonical_payload() for record in index.records],
        }
    )


def _index_set_identity_sha256(index_set: AnswerBearingSpanIndexSet) -> str:
    return _canonical_sha256(
        {
            "schema_version": index_set.schema_version,
            "policy": index_set.match_policy,
            "indices": [index.identity_sha256 for index in index_set.indices],
            "sidecar_sha256s": [index.sidecar_sha256 for index in index_set.indices],
            "statistics": index_set.statistics.canonical_payload(),
        }
    )


def _decode_json_object(raw_line: bytes, *, sidecar_line: int) -> dict[str, object]:
    try:
        text = raw_line.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        AnswerBearingSpanDataError,
    ) as error:
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: invalid JSON object: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AnswerBearingSpanDataError(
            f"sidecar line {sidecar_line}: JSON row must be an object"
        )
    return value


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AnswerBearingSpanDataError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise AnswerBearingSpanDataError(f"non-finite JSON constant {value!r}")


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = set(payload)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise AnswerBearingSpanDataError(
            f"{context} fields differ: missing={missing} unknown={unknown}"
        )


def _explicit_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _explicit_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _choices_payload(sample: object) -> list[dict[str, str]]:
    return [
        {"label": getattr(choice, "label"), "text": getattr(choice, "text")}
        for choice in getattr(sample, "choices")
    ]


def _choices_sha256(sample: object) -> str:
    return _canonical_sha256(_choices_payload(sample))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


__all__ = [
    "ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_INDEX_SET_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_MATCH_POLICY",
    "ANSWER_BEARING_SPAN_POPULATION_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_SEMANTIC_SCHEMA_VERSION",
    "ANSWER_BEARING_SPAN_SIDECAR_SCHEMA_VERSION",
    "VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON",
    "AnswerBearingSpanDataError",
    "AnswerBearingSpanIndex",
    "AnswerBearingSpanIndexSet",
    "AnswerBearingSpanRecord",
    "AnswerBearingSpanSidecarHeader",
    "AnswerBearingSpanStatistics",
    "AnswerBearingSpanStatus",
    "EvidenceCharacterSpan",
    "answer_bearing_span_population_sha256",
    "answer_bearing_span_semantic_sha256",
    "load_answer_bearing_span_index",
    "merge_answer_bearing_span_indices",
    "render_answer_bearing_span_sidecar",
]
