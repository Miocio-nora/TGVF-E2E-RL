"""Deterministic full-population materialization for Qwen3 T1 scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .policy_selection import (
    SelectionCandidate,
    SelectionSource,
    canonical_json_line,
)


T1_FULL_SELECTION_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-full-manifest.v1"
T1_FULL_SELECTION_ALGORITHM_VERSION = "t1-full-source-concatenation-v1"
T1_FULL_SELECTION_ORDER = tuple(source.value for source in SelectionSource)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key: {key}")
        record[key] = value
    return record


def _atomic_publish(path: Path, temporary: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace full-selection artifact: {path}")
    os.replace(temporary, path)


def materialize_t1_full_selection(
    candidate_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Concatenate the three screened sources under an auditable fixed order."""

    paths = tuple(Path(path).resolve() for path in candidate_paths)
    if len(paths) != len(T1_FULL_SELECTION_ORDER):
        raise ValueError("full T1 selection requires exactly three source files")
    output = Path(output_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    if output == manifest_output:
        raise ValueError("full candidates and manifest paths must differ")
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"full T1 source must be a regular file: {path}")
    for destination in (output, manifest_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"refusing to replace full-selection artifact: {destination}"
            )

    candidate_ids: set[str] = set()
    sample_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    source_records: list[dict[str, Any]] = []
    output_digest = hashlib.sha256()
    output_rows = 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            for expected_source, source_path in zip(
                T1_FULL_SELECTION_ORDER, paths, strict=True
            ):
                source_rows = 0
                with source_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            raise ValueError(
                                f"{source_path}:{line_number}: blank lines are forbidden"
                            )
                        try:
                            value = json.loads(
                                line,
                                object_pairs_hook=_reject_duplicate_keys,
                                parse_constant=lambda constant: (_ for _ in ()).throw(
                                    ValueError(
                                        f"non-finite JSON number: {constant}"
                                    )
                                ),
                            )
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{source_path}:{line_number}: invalid JSON"
                            ) from exc
                        if not isinstance(value, Mapping):
                            raise ValueError(
                                f"{source_path}:{line_number}: row must be an object"
                            )
                        candidate = SelectionCandidate.from_record(value)
                        if candidate.source.value != expected_source:
                            raise ValueError(
                                f"{source_path}:{line_number}: expected source "
                                f"{expected_source}, got {candidate.source.value}"
                            )
                        if candidate.identity_sha256 in candidate_ids:
                            raise ValueError("duplicate full T1 candidate identity")
                        if candidate.sample_id in sample_ids:
                            raise ValueError("duplicate full T1 sample ID")
                        candidate_ids.add(candidate.identity_sha256)
                        sample_ids.add(candidate.sample_id)
                        payload = canonical_json_line(candidate.canonical_record)
                        target.write(payload)
                        output_digest.update(payload)
                        output_rows += 1
                        source_rows += 1
                        source_counts[expected_source] += 1
                if source_rows == 0:
                    raise ValueError(f"full T1 source is empty: {source_path}")
                source_records.append(
                    {
                        "source": expected_source,
                        "path": str(source_path),
                        "sha256": _sha256_file(source_path),
                        "rows": source_rows,
                    }
                )
            target.flush()
            os.fsync(target.fileno())
        candidates_sha256 = output_digest.hexdigest()
        manifest = {
            "schema_version": T1_FULL_SELECTION_MANIFEST_SCHEMA,
            "selection_algorithm_version": T1_FULL_SELECTION_ALGORITHM_VERSION,
            "selection_is_outcome_independent": True,
            "ordering": "vstar-arxivqa-thinklite-screened-row-order-v1",
            "sources": source_records,
            "source_counts": dict(source_counts),
            "rows": output_rows,
            "candidates_path": str(output),
            "candidates_sha256": candidates_sha256,
        }
        manifest_payload = canonical_json_line(manifest)
        manifest_descriptor, manifest_temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_output.name}.",
            suffix=".tmp",
            dir=manifest_output.parent,
        )
        manifest_temporary = Path(manifest_temporary_name)
        try:
            with os.fdopen(manifest_descriptor, "wb") as target:
                target.write(manifest_payload)
                target.flush()
                os.fsync(target.fileno())
            _atomic_publish(output, temporary)
            _atomic_publish(manifest_output, manifest_temporary)
        finally:
            if manifest_temporary.exists():
                manifest_temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "rows": output_rows,
        "source_counts": dict(source_counts),
        "candidates_sha256": candidates_sha256,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "output": str(output),
        "manifest_output": str(manifest_output),
    }


__all__ = [
    "T1_FULL_SELECTION_ALGORITHM_VERSION",
    "T1_FULL_SELECTION_MANIFEST_SCHEMA",
    "T1_FULL_SELECTION_ORDER",
    "materialize_t1_full_selection",
]
