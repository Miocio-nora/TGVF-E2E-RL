"""Materialize a deterministic final-answer-only VLMEvalKit scoring view.

The evaluated policy is allowed to emit a long native Thinking response, but
VLMEvalKit owns only the final-answer scoring step.  This module therefore
derives a new TSV whose ``prediction`` cell is the non-empty text after the
last ``</think>`` closer.  The inference TSV is never edited in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any


THINK_CLOSER = "</think>"
SCORING_VIEW_CONTRACT = "vlmevalkit-final-answer-view-v1"
INVALID_SENTINEL_PREFIX = "__TGVF_INVALID_FINAL_ANSWER__"
_OPTION_LABELS = tuple(reversed("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
_REQUIRED_COLUMNS = frozenset({"index", "prediction"})


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(1024 * 1024 * 1024)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames
            if fields is None:
                raise ValueError(f"TSV has no header: {path}")
            if len(fields) != len(set(fields)):
                raise ValueError(f"TSV has duplicate columns: {path}")
            missing = _REQUIRED_COLUMNS.difference(fields)
            if missing:
                raise ValueError(f"TSV lacks required columns {sorted(missing)}: {path}")
            rows: list[dict[str, str]] = []
            for row_number, raw_row in enumerate(reader, start=1):
                if None in raw_row:
                    raise ValueError(f"TSV row {row_number} has extra cells: {path}")
                if any(value is None for value in raw_row.values()):
                    raise ValueError(f"TSV row {row_number} has missing cells: {path}")
                rows.append({field: raw_row[field] for field in fields})
    finally:
        csv.field_size_limit(previous_limit)
    if not rows:
        raise ValueError(f"completed prediction TSV contains no rows: {path}")
    indices = [row["index"] for row in rows]
    if any(not index.strip() for index in indices):
        raise ValueError("prediction TSV contains an empty index")
    if len(indices) != len(set(indices)):
        raise ValueError("prediction TSV contains duplicate indices")
    return list(fields), rows


def _write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def _publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Publish bytes without allowing a pre-existing artifact to be replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _final_answer(prediction: str) -> tuple[str | None, str | None]:
    _, closer, suffix = prediction.rpartition(THINK_CLOSER)
    if not closer:
        return None, "missing_think_closer"
    if not suffix.strip():
        return None, "empty_final_answer"
    return suffix, None


def _is_mcq_row(row: Mapping[str, str], source_fields: Sequence[str]) -> bool:
    populated_options = sum(
        bool(row[field].strip())
        for field in source_fields
        if len(field) == 1 and field in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    return populated_options >= 2


def _unused_option_label(
    *,
    source_fields: set[str],
    answer: str,
) -> str:
    normalized_answer = answer.strip().upper()
    for label in _OPTION_LABELS:
        if label not in source_fields and label != normalized_answer:
            return label
    raise ValueError("no unused uppercase option label can force an invalid MCQ row wrong")


def _unique_sentinel(
    *,
    source_sha256: str,
    index: str,
    row_number: int,
    reason: str,
    forbidden: set[str],
) -> str:
    salt = 0
    while True:
        identity = f"{source_sha256}\0{index}\0{row_number}\0{reason}\0{salt}"
        suffix = sha256(identity.encode("utf-8")).hexdigest()
        sentinel = f"{INVALID_SENTINEL_PREFIX}{suffix}"
        if sentinel not in forbidden:
            forbidden.add(sentinel)
            return sentinel
        salt += 1


def _load_mathverse_problem_versions(path: Path) -> tuple[list[Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MathVerse source JSON must contain an array")
    return payload, _sha256_file(path)


def _enrich_mathverse_metadata(
    rows: list[dict[str, str]],
    *,
    source_json: Path,
) -> dict[str, Any]:
    if "source_row_index" not in rows[0] or "metadata" not in rows[0]:
        raise ValueError(
            "MathVerse enrichment requires source_row_index and metadata columns"
        )
    source_rows, source_hash = _load_mathverse_problem_versions(source_json)
    for row_number, row in enumerate(rows, start=1):
        try:
            source_row_index = int(row["source_row_index"])
        except ValueError as error:
            raise ValueError(
                f"row {row_number} has a non-integer source_row_index"
            ) from error
        if source_row_index < 0 or source_row_index >= len(source_rows):
            raise IndexError(
                f"row {row_number} source_row_index is outside MathVerse JSON"
            )
        source_row = source_rows[source_row_index]
        if not isinstance(source_row, Mapping):
            raise ValueError(f"MathVerse source row {source_row_index} is not an object")
        problem_version = source_row.get("problem_version")
        if not isinstance(problem_version, str) or not problem_version.strip():
            raise ValueError(
                f"MathVerse source row {source_row_index} lacks problem_version"
            )
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError as error:
            raise ValueError(f"row {row_number} metadata is not valid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"row {row_number} metadata is not a JSON object")
        existing = metadata.get("problem_version")
        if existing is not None and existing != problem_version:
            raise ValueError(
                f"row {row_number} metadata problem_version disagrees with source JSON"
            )
        metadata["problem_version"] = problem_version
        row["metadata"] = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return {
        "field": "metadata.problem_version",
        "joined_row_count": len(rows),
        "source_json": str(source_json),
        "source_json_sha256": source_hash,
    }


def _verify_derived_rows(
    *,
    source_fields: Sequence[str],
    source_rows: Sequence[Mapping[str, str]],
    derived_path: Path,
    expected_rows: Sequence[Mapping[str, str]],
    metadata_enriched: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    derived_fields, derived_rows = _read_tsv(derived_path)
    if len(derived_rows) != len(source_rows):
        raise RuntimeError("derived scoring view row count drifted")
    if [row["index"] for row in derived_rows] != [row["index"] for row in source_rows]:
        raise RuntimeError("derived scoring view index identity drifted")
    if derived_rows != list(expected_rows):
        raise RuntimeError("derived scoring view differs from the materialized rows")
    for source, derived in zip(source_rows, derived_rows, strict=True):
        for field in source_fields:
            if field == "prediction" or (metadata_enriched and field == "metadata"):
                continue
            if source[field] != derived[field]:
                raise RuntimeError(
                    f"derived scoring view changed non-prediction field {field!r}"
                )
    return derived_fields, derived_rows


def materialize_final_answer_view(
    *,
    source_tsv: str | Path,
    derived_tsv: str | Path,
    manifest_path: str | Path | None = None,
    mathverse_source_json: str | Path | None = None,
) -> dict[str, Any]:
    """Create an immutable final-answer scoring TSV and its identity manifest."""

    source = Path(source_tsv).resolve()
    derived = Path(derived_tsv).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else derived.with_suffix(derived.suffix + ".manifest.json")
    )
    if not source.is_file():
        raise FileNotFoundError(f"source prediction TSV does not exist: {source}")
    if source == derived:
        raise ValueError("source and derived TSV paths must differ")
    if derived.exists() or manifest.exists():
        raise FileExistsError("derived TSV and manifest are immutable and must not exist")

    source_fields, source_rows = _read_tsv(source)
    source_hash = _sha256_file(source)
    rows = [dict(row) for row in source_rows]
    metadata_enrichment = None
    if mathverse_source_json is not None:
        mathverse_path = Path(mathverse_source_json).resolve()
        if not mathverse_path.is_file():
            raise FileNotFoundError(f"MathVerse source JSON does not exist: {mathverse_path}")
        metadata_enrichment = _enrich_mathverse_metadata(
            rows,
            source_json=mathverse_path,
        )

    original_fields = set(source_fields)
    forbidden_sentinels = {value for row in source_rows for value in row.values()}
    injected_labels: set[str] = set()
    invalid_reasons = {"missing_think_closer": 0, "empty_final_answer": 0}
    mcq_row_count = 0
    invalid_mcq_count = 0
    invalid_non_mcq_count = 0
    closed_count = 0

    for row_number, row in enumerate(rows, start=1):
        is_mcq = _is_mcq_row(source_rows[row_number - 1], source_fields)
        mcq_row_count += int(is_mcq)
        answer, invalid_reason = _final_answer(row["prediction"])
        if invalid_reason is None:
            assert answer is not None
            row["prediction"] = answer
            closed_count += 1
            continue

        invalid_reasons[invalid_reason] += 1
        sentinel = _unique_sentinel(
            source_sha256=source_hash,
            index=row["index"],
            row_number=row_number,
            reason=invalid_reason,
            forbidden=forbidden_sentinels,
        )
        if is_mcq:
            label = _unused_option_label(
                source_fields=original_fields,
                answer=row.get("answer", ""),
            )
            injected_labels.add(label)
            row[label] = sentinel
            row["prediction"] = label
            invalid_mcq_count += 1
        else:
            row["prediction"] = sentinel
            invalid_non_mcq_count += 1

    derived_fields = [*source_fields, *sorted(injected_labels, reverse=True)]
    for row in rows:
        for label in injected_labels:
            row.setdefault(label, "")

    derived.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{derived.name}.",
        dir=derived.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _write_tsv(temporary, derived_fields, rows)
        verified_fields, verified_rows = _verify_derived_rows(
            source_fields=source_fields,
            source_rows=source_rows,
            derived_path=temporary,
            expected_rows=rows,
            metadata_enriched=metadata_enrichment is not None,
        )
        derived_bytes = temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)

    _publish_bytes_exclusive(derived, derived_bytes)
    derived_hash = _sha256_file(derived)
    invalid_count = invalid_mcq_count + invalid_non_mcq_count
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": SCORING_VIEW_CONTRACT,
        "source": {
            "path": str(source),
            "sha256": source_hash,
            "row_count": len(source_rows),
            "columns": source_fields,
        },
        "derived": {
            "path": str(derived),
            "sha256": derived_hash,
            "row_count": len(verified_rows),
            "columns": verified_fields,
        },
        "counts": {
            "row_count": len(rows),
            "closed_count": closed_count,
            "invalid_count": invalid_count,
            "missing_think_closer_count": invalid_reasons["missing_think_closer"],
            "empty_final_answer_count": invalid_reasons["empty_final_answer"],
            "mcq_row_count": mcq_row_count,
            "invalid_mcq_count": invalid_mcq_count,
            "invalid_non_mcq_count": invalid_non_mcq_count,
        },
        "invalid_policy": {
            "sentinel_prefix": INVALID_SENTINEL_PREFIX,
            "injected_option_columns": sorted(injected_labels, reverse=True),
            "llm_or_random_fallback_allowed": False,
        },
        "verification": {
            "index_order_and_values_identical": True,
            "non_prediction_source_fields_verified": True,
            "unchanged_non_prediction_source_fields_identical": True,
            "documented_modified_source_fields": (
                ["metadata"] if metadata_enrichment is not None else []
            ),
        },
        "mathverse_metadata_enrichment": metadata_enrichment,
    }
    manifest_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _publish_bytes_exclusive(manifest, manifest_bytes)
    except Exception:
        derived.unlink(missing_ok=True)
        raise
    return payload


__all__ = [
    "INVALID_SENTINEL_PREFIX",
    "SCORING_VIEW_CONTRACT",
    "THINK_CLOSER",
    "materialize_final_answer_view",
]
