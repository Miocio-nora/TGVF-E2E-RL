"""Apply a small reviewed override set to a complete RP70 annotation JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import (
    load_retained_representation_jsonl,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from .data import AnswerBearingSpanStatus, EvidenceCharacterSpan
from .data import VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON
from .deepseek_index_repair import evidence_tokens
from .sidecar_tool import (
    AnswerBearingSpanAnnotationError,
    _fsync_directory,
    _load_annotations,
    _object_without_duplicate_keys,
    _prospective_output_path,
    _reject_json_constant,
    _require_output_absent,
    _validate_annotation_semantics,
)


ANNOTATION_OVERRIDE_SUMMARY_SCHEMA_VERSION = "answer_bearing_span_override_merge_v1"
_Annotation = tuple[
    AnswerBearingSpanStatus,
    str | None,
    tuple[EvidenceCharacterSpan, ...],
]


def merge_answer_bearing_span_annotation_overrides(
    *,
    training_config_path: str | Path,
    split: Literal["train", "validation"] | str,
    base_annotations_path: str | Path,
    overrides_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Validate and exclusively publish one complete, reviewed annotation set."""

    if split not in {"train", "validation"}:
        raise ValueError("split must be exactly 'train' or 'validation'")
    output = _prospective_output_path(output_path)
    _require_output_absent(output)

    training = load_representation_training_config(training_config_path)
    split_config = training.data.train if split == "train" else training.data.validation
    dataset = load_retained_representation_jsonl(
        split_config.jsonl_path,
        expected_source_sha256=split_config.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    base, base_source, base_sha256 = _load_annotations(
        base_annotations_path,
        dataset=dataset,
    )
    overrides, override_source, override_sha256 = _load_overrides(
        overrides_path,
        samples_by_uid={sample.sample_id: sample for sample in dataset.samples},
    )

    merged: list[_Annotation] = []
    effective_changes = 0
    for sample, base_annotation in zip(dataset.samples, base, strict=True):
        annotation = overrides.get(sample.sample_id, base_annotation)
        merged.append(annotation)
        if sample.sample_id in overrides and annotation != base_annotation:
            effective_changes += 1
    if effective_changes == 0:
        raise AnswerBearingSpanAnnotationError(
            "override file makes no effective annotation change"
        )

    payload = _render_annotations(
        sample_uids=tuple(sample.sample_id for sample in dataset.samples),
        annotations=tuple(merged),
    )
    output = _publish_complete_annotations_exclusive(
        output,
        payload=payload,
        dataset=dataset,
    )
    return {
        "schema_version": ANNOTATION_OVERRIDE_SUMMARY_SCHEMA_VERSION,
        "status": "merged",
        "split": split,
        "source_sha256": dataset.manifest.source_sha256,
        "base_annotations_path": str(base_source),
        "base_annotations_sha256": base_sha256,
        "overrides_path": str(override_source),
        "overrides_sha256": override_sha256,
        "override_count": len(overrides),
        "effective_change_count": effective_changes,
        "override_uids": list(overrides),
        "output_path": str(output),
        "output_sha256": sha256(payload).hexdigest(),
    }


def _load_overrides(
    path_value: str | Path,
    *,
    samples_by_uid: Mapping[str, RepresentationTrainingSample],
) -> tuple[dict[str, _Annotation], Path, str]:
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise AnswerBearingSpanAnnotationError(
            f"overrides path does not resolve to a file: {path_value}"
        ) from error
    if not path.is_file():
        raise AnswerBearingSpanAnnotationError(f"overrides path is not a file: {path}")
    raw = path.read_bytes()
    lines = raw.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise AnswerBearingSpanAnnotationError(
            "overrides JSONL must contain non-blank rows"
        )
    decoded = tuple(
        _parse_token_override_row(line, annotation_line=line_number)
        for line_number, line in enumerate(lines, start=1)
    )
    counts = Counter(row[0] for row in decoded)
    duplicate_uids = sorted(uid for uid, count in counts.items() if count > 1)
    if duplicate_uids:
        raise AnswerBearingSpanAnnotationError(
            f"overrides contain duplicate UIDs: {duplicate_uids}"
        )
    unknown_uids = sorted(set(counts) - set(samples_by_uid))
    if unknown_uids:
        raise AnswerBearingSpanAnnotationError(
            f"overrides contain unknown UIDs: {unknown_uids}"
        )

    overrides: dict[str, _Annotation] = {}
    for line_number, (uid, raw_status, token_indices) in enumerate(decoded, start=1):
        sample = samples_by_uid[uid]
        status, reason, spans = _annotation_from_token_indices(
            sample,
            status=raw_status,
            token_indices=token_indices,
            annotation_line=line_number,
        )
        _validate_annotation_semantics(
            status=status,
            reason=reason,
            spans=spans,
            evidence_description=sample.evidence_description,
            annotation_line=line_number,
        )
        overrides[uid] = (status, reason, spans)
    return overrides, path, sha256(raw).hexdigest()


def _parse_token_override_row(
    raw_line: bytes,
    *,
    annotation_line: int,
) -> tuple[str, str, tuple[int, ...]]:
    try:
        value = json.loads(
            raw_line.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: invalid strict JSON: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "uid",
        "status",
        "token_indices",
    }:
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: fields must be exactly "
            "['status', 'token_indices', 'uid']"
        )
    uid = value["uid"]
    status = value["status"]
    raw_token_indices = value["token_indices"]
    if not isinstance(uid, str) or not uid.strip():
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: uid must be a non-empty string"
        )
    if status not in {"resolved", "no_span"}:
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: status must be resolved or no_span"
        )
    if not isinstance(raw_token_indices, list) or any(
        type(token_id) is not int for token_id in raw_token_indices
    ):
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: token_indices must be an integer array"
        )
    token_indices = tuple(raw_token_indices)
    if token_indices != tuple(sorted(set(token_indices))):
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: token_indices must be sorted and unique"
        )
    if status == "resolved" and not token_indices:
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: resolved requires token indices"
        )
    if status == "no_span" and token_indices:
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: no_span requires no token indices"
        )
    return uid, status, token_indices


def _annotation_from_token_indices(
    sample: RepresentationTrainingSample,
    *,
    status: str,
    token_indices: tuple[int, ...],
    annotation_line: int,
) -> _Annotation:
    if status == "no_span":
        return (
            AnswerBearingSpanStatus.VERIFIED_NO_ANSWER_BEARING_EVIDENCE,
            VERIFIED_NO_ANSWER_BEARING_EVIDENCE_REASON,
            (),
        )

    tokens = evidence_tokens(sample.evidence_description)
    if token_indices[0] < 0 or token_indices[-1] >= len(tokens):
        raise AnswerBearingSpanAnnotationError(
            f"override line {annotation_line}: token index is out of bounds for "
            f"{sample.sample_id!r} ({len(tokens)} tokens)"
        )
    runs: list[tuple[int, int]] = []
    run_start = token_indices[0]
    run_end = run_start
    for token_id in token_indices[1:]:
        if token_id == run_end + 1:
            run_end = token_id
        else:
            runs.append((run_start, run_end))
            run_start = token_id
            run_end = token_id
    runs.append((run_start, run_end))
    spans = tuple(
        EvidenceCharacterSpan(
            start=tokens[start_token].start,
            end=tokens[end_token].end,
            exact_text=sample.evidence_description[
                tokens[start_token].start : tokens[end_token].end
            ],
        )
        for start_token, end_token in runs
    )
    return AnswerBearingSpanStatus.RESOLVED, None, spans


def _render_annotations(
    *,
    sample_uids: Sequence[str],
    annotations: Sequence[_Annotation],
) -> bytes:
    if len(sample_uids) != len(annotations):
        raise ValueError("sample_uids and annotations must have equal length")
    rows: list[bytes] = []
    for uid, (status, reason, spans) in zip(sample_uids, annotations, strict=True):
        record = {
            "uid": uid,
            "status": status.value,
            "reason": reason,
            "spans": [
                {"start": span.start, "end": span.end, "exact_text": span.exact_text}
                for span in spans
            ],
        }
        rows.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    return b"".join(rows)


def _publish_complete_annotations_exclusive(
    output: Path,
    *,
    payload: bytes,
    dataset: object,
) -> Path:
    parent = output.parent
    if not parent.exists():
        parent.mkdir()
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {resolved_parent}")
    resolved_output = resolved_parent / output.name
    _require_output_absent(resolved_output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved_output.name}.", suffix=".tmp", dir=resolved_parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _load_annotations(temporary, dataset=dataset)
        try:
            os.link(temporary, resolved_output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite annotation output: {resolved_output}"
            ) from error
        _fsync_directory(resolved_parent)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved_output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = merge_answer_bearing_span_annotation_overrides(
        training_config_path=args.training_config,
        split=args.split,
        base_annotations_path=args.base_annotations,
        overrides_path=args.overrides,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "ANNOTATION_OVERRIDE_SUMMARY_SCHEMA_VERSION",
    "main",
    "merge_answer_bearing_span_annotation_overrides",
]
