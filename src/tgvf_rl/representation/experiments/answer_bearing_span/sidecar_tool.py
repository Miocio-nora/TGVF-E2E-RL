"""CPU-only materialization of semantic-bound RP70 span sidecars.

This module deliberately consumes the ordinary representation-training config
and retained-data loader.  It does not infer spans, filter rows, or import the
representation runner.  Human/audited annotations must provide exactly one
UID-keyed record for every retained sample; JSONL row order is immaterial.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import (
    RepresentationDataset,
    load_retained_representation_jsonl,
)

from .data import (
    VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
    AnswerBearingSpanStatus,
    EvidenceCharacterSpan,
    load_answer_bearing_span_index,
    render_answer_bearing_span_sidecar,
)


ANSWER_BEARING_SPAN_MATERIALIZATION_SUMMARY_SCHEMA_VERSION = (
    "answer_bearing_span_sidecar_materialization_summary_v1"
)
_ANNOTATION_FIELDS = frozenset({"uid", "status", "reason", "spans"})
_SPAN_FIELDS = frozenset({"start", "end", "exact_text"})


class AnswerBearingSpanAnnotationError(ValueError):
    """An annotation JSONL is not a complete semantic-population audit."""


class _DuplicateJsonKeyError(ValueError):
    pass


def materialize_answer_bearing_span_sidecar(
    *,
    training_config_path: str | Path,
    split: Literal["train", "validation"] | str,
    annotations_path: str | Path,
    annotator_identity: str,
    output_path: str | Path,
) -> dict[str, object]:
    """Validate, canonically render, and atomically publish one RP70 sidecar."""

    if split not in {"train", "validation"}:
        raise ValueError("split must be exactly 'train' or 'validation'")
    if not isinstance(annotator_identity, str) or not annotator_identity.strip():
        raise ValueError("annotator_identity must be a non-empty string")

    output = _prospective_output_path(output_path)
    _require_output_absent(output)

    training = load_representation_training_config(training_config_path)
    split_config = training.data.train if split == "train" else training.data.validation
    dataset = load_retained_representation_jsonl(
        split_config.jsonl_path,
        expected_source_sha256=split_config.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    annotations, annotations_source, annotations_sha256 = _load_annotations(
        annotations_path,
        dataset=dataset,
    )
    payload = render_answer_bearing_span_sidecar(
        dataset,
        annotations,
        annotator_identity=annotator_identity,
    )
    sidecar_sha256 = sha256(payload).hexdigest()

    output = _publish_verified_sidecar_exclusive(
        output,
        payload=payload,
        dataset=dataset,
        sidecar_sha256=sidecar_sha256,
    )
    index = load_answer_bearing_span_index(
        dataset,
        output,
        expected_sidecar_sha256=sidecar_sha256,
    )
    return {
        "schema_version": (ANSWER_BEARING_SPAN_MATERIALIZATION_SUMMARY_SCHEMA_VERSION),
        "status": "materialized",
        "split": split,
        "training_config_path": str(training.source_path),
        "training_config_source_sha256": training.source_toml_sha256,
        "training_config_canonical_sha256": training.canonical_config_sha256,
        "source_path": dataset.manifest.source_path,
        "source_sha256": index.source_sha256,
        "retained_manifest_sha256": dataset.manifest.manifest_sha256,
        "annotations_path": str(annotations_source),
        "annotations_sha256": annotations_sha256,
        "output_path": str(output),
        "sidecar_sha256": index.sidecar_sha256,
        "index_sha256": index.identity_sha256,
        "population_sha256": index.retained_semantic_population_sha256,
        "annotator_identity": index.annotator_identity,
        "statistics": index.statistics.canonical_payload(),
    }


def _load_annotations(
    path_value: str | Path,
    *,
    dataset: RepresentationDataset,
) -> tuple[
    tuple[
        tuple[
            AnswerBearingSpanStatus,
            str | None,
            tuple[EvidenceCharacterSpan, ...],
        ],
        ...,
    ],
    Path,
    str,
]:
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise AnswerBearingSpanAnnotationError(
            f"annotations path does not resolve to a file: {path_value}"
        ) from error
    if not path.is_file():
        raise AnswerBearingSpanAnnotationError(
            f"annotations path is not a file: {path}"
        )
    raw = path.read_bytes()
    lines = raw.splitlines()
    expected_count = len(dataset.samples)
    if len(lines) != expected_count:
        raise AnswerBearingSpanAnnotationError(
            "annotations must contain exactly one row per retained sample: "
            f"expected {expected_count}, got {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise AnswerBearingSpanAnnotationError("annotations JSONL contains a blank row")

    decoded = tuple(
        _parse_annotation_row(line, annotation_line=line_number)
        for line_number, line in enumerate(lines, start=1)
    )
    uids = tuple(row[0] for row in decoded)
    duplicate_uids = sorted(uid for uid, count in Counter(uids).items() if count > 1)
    if duplicate_uids:
        raise AnswerBearingSpanAnnotationError(
            f"annotations contain duplicate UIDs: {duplicate_uids}"
        )
    expected_uids = tuple(sample.sample_id for sample in dataset.samples)
    expected_uid_set = frozenset(expected_uids)
    unknown_uids = sorted(set(uids) - expected_uid_set)
    if unknown_uids:
        raise AnswerBearingSpanAnnotationError(
            f"annotations contain unknown UIDs: {unknown_uids}"
        )
    missing_uids = sorted(expected_uid_set - set(uids))
    if missing_uids:
        raise AnswerBearingSpanAnnotationError(
            f"annotations are missing retained UIDs: {missing_uids}"
        )

    decoded_by_uid = {
        row[0]: (line_number, row) for line_number, row in enumerate(decoded, start=1)
    }
    annotations: list[
        tuple[
            AnswerBearingSpanStatus,
            str | None,
            tuple[EvidenceCharacterSpan, ...],
        ]
    ] = []
    for sample in dataset.samples:
        line_number, row = decoded_by_uid[sample.sample_id]
        _uid, status, reason, spans = row
        _validate_annotation_semantics(
            status=status,
            reason=reason,
            spans=spans,
            evidence_description=sample.evidence_description,
            annotation_line=line_number,
        )
        annotations.append((status, reason, spans))
    return tuple(annotations), path, sha256(raw).hexdigest()


def _parse_annotation_row(
    raw_line: bytes,
    *,
    annotation_line: int,
) -> tuple[
    str,
    AnswerBearingSpanStatus,
    str | None,
    tuple[EvidenceCharacterSpan, ...],
]:
    try:
        text = raw_line.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: invalid strict JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: row must be an object"
        )
    _require_exact_fields(
        value,
        _ANNOTATION_FIELDS,
        context=f"annotation line {annotation_line}",
    )
    uid = value["uid"]
    if not isinstance(uid, str) or not uid.strip():
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: uid must be a non-empty string"
        )
    try:
        status = AnswerBearingSpanStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: unsupported or unresolved status "
            f"{value['status']!r}"
        ) from error
    reason = value["reason"]
    if reason is not None and not isinstance(reason, str):
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: reason must be string or null"
        )
    raw_spans = value["spans"]
    if not isinstance(raw_spans, list):
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: spans must be an array"
        )
    spans: list[EvidenceCharacterSpan] = []
    for span_index, raw_span in enumerate(raw_spans):
        context = f"annotation line {annotation_line} span {span_index}"
        if not isinstance(raw_span, dict):
            raise AnswerBearingSpanAnnotationError(f"{context}: span must be an object")
        _require_exact_fields(raw_span, _SPAN_FIELDS, context=context)
        start = raw_span["start"]
        end = raw_span["end"]
        exact_text = raw_span["exact_text"]
        if type(start) is not int or type(end) is not int:
            raise AnswerBearingSpanAnnotationError(
                f"{context}: start and end must be integers"
            )
        if not isinstance(exact_text, str):
            raise AnswerBearingSpanAnnotationError(
                f"{context}: exact_text must be a string"
            )
        try:
            spans.append(
                EvidenceCharacterSpan(
                    start=start,
                    end=end,
                    exact_text=exact_text,
                )
            )
        except (TypeError, ValueError) as error:
            raise AnswerBearingSpanAnnotationError(f"{context}: {error}") from error
    return uid, status, reason, tuple(spans)


def _validate_annotation_semantics(
    *,
    status: AnswerBearingSpanStatus,
    reason: str | None,
    spans: tuple[EvidenceCharacterSpan, ...],
    evidence_description: str,
    annotation_line: int,
) -> None:
    if spans != tuple(sorted(spans)) or len(spans) != len(set(spans)):
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: spans must be sorted and unique"
        )
    previous_end = 0
    for span_index, span in enumerate(spans):
        if span_index and span.start < previous_end:
            raise AnswerBearingSpanAnnotationError(
                f"annotation line {annotation_line}: spans must not overlap"
            )
        if span.end > len(evidence_description) or (
            evidence_description[span.start : span.end] != span.exact_text
        ):
            raise AnswerBearingSpanAnnotationError(
                f"annotation line {annotation_line}: span {span_index} exact_text "
                "drifted from the retained evidence_description"
            )
        previous_end = span.end
    if status is AnswerBearingSpanStatus.RESOLVED:
        if reason is not None:
            raise AnswerBearingSpanAnnotationError(
                f"annotation line {annotation_line}: resolved reason must be null"
            )
        if not spans:
            raise AnswerBearingSpanAnnotationError(
                f"annotation line {annotation_line}: resolved status requires spans"
            )
        return
    if reason != VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON:
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: verified-no-evidence reason is invalid"
        )
    if spans:
        raise AnswerBearingSpanAnnotationError(
            f"annotation line {annotation_line}: verified-no-evidence requires no spans"
        )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate object field {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise _DuplicateJsonKeyError(f"non-standard JSON constant {value!r}")


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AnswerBearingSpanAnnotationError(
            f"{context}: fields differ; missing={missing}, extra={extra}"
        )


def _prospective_output_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.name:
        raise ValueError("output_path must name a file")
    return path


def _require_output_absent(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite sidecar output: {path}")


def _publish_verified_sidecar_exclusive(
    output: Path,
    *,
    payload: bytes,
    dataset: RepresentationDataset,
    sidecar_sha256: str,
) -> Path:
    parent = output.parent
    if not parent.exists():
        # Intentionally create only the explicit parent.  Missing ancestors are
        # an error rather than an invitation to mutate a broader path tree.
        parent.mkdir()
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {resolved_parent}")
    resolved_output = resolved_parent / output.name
    _require_output_absent(resolved_output)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved_output.name}.",
        suffix=".tmp",
        dir=resolved_parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Re-read the exact bytes through the production loader before making
        # the destination name visible.
        load_answer_bearing_span_index(
            dataset,
            temporary,
            expected_sidecar_sha256=sidecar_sha256,
        )
        try:
            os.link(temporary, resolved_output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite sidecar output: {resolved_output}"
            ) from error
        _fsync_directory(resolved_parent)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved_output


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotator-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = materialize_answer_bearing_span_sidecar(
        training_config_path=args.training_config,
        split=args.split,
        annotations_path=args.annotations,
        annotator_identity=args.annotator_identity,
        output_path=args.output,
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ANSWER_BEARING_SPAN_MATERIALIZATION_SUMMARY_SCHEMA_VERSION",
    "AnswerBearingSpanAnnotationError",
    "main",
    "materialize_answer_bearing_span_sidecar",
]
