"""Deterministic, outcome-independent source quotas for the full T1 run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_PRIMARY_SOURCES,
    SelectionCandidate,
    canonical_json_line,
)


T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA = (
    "tgvf.policy-selection.t1-source-quota-manifest.v1"
)
T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION = "t1-source-content-hash-bottom-k-v1"
T1_RECOMMENDED_SELECTION_NAMESPACE = "qwen3-instruct-t1-recommended-20260726-v1"
T1_RECOMMENDED_SELECTION_ORDER = tuple(
    source.value for source in POLICY_SELECTION_PRIMARY_SOURCES
)
T1_RECOMMENDED_SOURCE_QUOTAS = MappingProxyType(
    {"vstar": 170_000, "arxivqa": 32_000, "thinklite": 69_842}
)
T1_RECOMMENDED_SELECTION_ROWS = sum(T1_RECOMMENDED_SOURCE_QUOTAS.values())
_SELECTION_SCORE_DOMAIN = b"tgvf-policy-selection-t1-source-quota-v1\0"


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


def _candidate_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are forbidden")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda constant: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON number: {constant}")
                    ),
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield SelectionCandidate.from_record(value)


def _selection_score(*, namespace: str, source: str, candidate_sha256: str) -> str:
    state = (
        namespace.encode("utf-8")
        + b"\0"
        + source.encode("ascii")
        + b"\0"
        + candidate_sha256.encode("ascii")
    )
    return hashlib.sha256(_SELECTION_SCORE_DOMAIN + state).hexdigest()


def materialize_t1_recommended_selection(
    candidate_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    source_quotas: Mapping[str, int] = T1_RECOMMENDED_SOURCE_QUOTAS,
    namespace: str = T1_RECOMMENDED_SELECTION_NAMESPACE,
) -> dict[str, Any]:
    """Hash-sample fixed source quotas and atomically publish one selection root."""

    paths = tuple(Path(path).resolve() for path in candidate_paths)
    if len(paths) != len(T1_RECOMMENDED_SELECTION_ORDER):
        raise ValueError("recommended T1 selection requires exactly three sources")
    quotas = dict(source_quotas)
    if set(quotas) != set(T1_RECOMMENDED_SELECTION_ORDER) or any(
        type(value) is not int or value <= 0 for value in quotas.values()
    ):
        raise ValueError("source_quotas must contain three positive integer quotas")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"selection source must be a regular file: {path}")
    root = Path(output_root).resolve()
    if os.path.lexists(root):
        raise FileExistsError(f"selection output root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    candidate_ids: set[str] = set()
    sample_ids: set[str] = set()
    selected_by_source: dict[str, set[str]] = {}
    source_records: list[dict[str, Any]] = []
    try:
        for expected_source, source_path in zip(
            T1_RECOMMENDED_SELECTION_ORDER, paths, strict=True
        ):
            scored: list[tuple[str, str]] = []
            rows = 0
            for candidate in _candidate_lines(source_path):
                if candidate.source.value != expected_source:
                    raise ValueError(
                        f"{source_path}: expected {expected_source}, "
                        f"got {candidate.source.value}"
                    )
                if candidate.identity_sha256 in candidate_ids:
                    raise ValueError("duplicate recommended T1 candidate identity")
                if candidate.sample_id in sample_ids:
                    raise ValueError("duplicate recommended T1 sample ID")
                candidate_ids.add(candidate.identity_sha256)
                sample_ids.add(candidate.sample_id)
                scored.append(
                    (
                        _selection_score(
                            namespace=namespace,
                            source=expected_source,
                            candidate_sha256=candidate.identity_sha256,
                        ),
                        candidate.identity_sha256,
                    )
                )
                rows += 1
            quota = quotas[expected_source]
            if rows < quota:
                raise ValueError(
                    f"{expected_source} has {rows} candidates but quota is {quota}"
                )
            selected_pairs = heapq.nsmallest(quota, scored)
            selected_by_source[expected_source] = {
                identity for _, identity in selected_pairs
            }
            source_records.append(
                {
                    "source": expected_source,
                    "path": str(source_path),
                    "sha256": _sha256_file(source_path),
                    "rows": rows,
                    "quota": quota,
                    "selection_mode": (
                        "all" if rows == quota else "content_hash_bottom_k"
                    ),
                    "selection_cutoff_sha256": max(
                        score for score, _ in selected_pairs
                    ),
                }
            )

        candidates_path = temporary_root / "candidates.jsonl"
        candidates_digest = hashlib.sha256()
        selected_identity_digests: dict[str, hashlib._Hash] = {
            source: hashlib.sha256() for source in T1_RECOMMENDED_SELECTION_ORDER
        }
        selected_counts = {source: 0 for source in T1_RECOMMENDED_SELECTION_ORDER}
        with candidates_path.open("wb") as output:
            for expected_source, source_path in zip(
                T1_RECOMMENDED_SELECTION_ORDER, paths, strict=True
            ):
                selected = selected_by_source[expected_source]
                for candidate in _candidate_lines(source_path):
                    if candidate.identity_sha256 not in selected:
                        continue
                    payload = canonical_json_line(candidate.canonical_record)
                    output.write(payload)
                    candidates_digest.update(payload)
                    selected_identity_digests[expected_source].update(
                        candidate.identity_sha256.encode("ascii") + b"\n"
                    )
                    selected_counts[expected_source] += 1
                if selected_counts[expected_source] != quotas[expected_source]:
                    raise AssertionError("selected source count differs from quota")
            output.flush()
            os.fsync(output.fileno())

        for source_record in source_records:
            source = str(source_record["source"])
            source_record["selected_identities_sha256"] = selected_identity_digests[
                source
            ].hexdigest()
        candidates_sha256 = candidates_digest.hexdigest()
        rows = sum(selected_counts.values())
        manifest = {
            "schema_version": T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA,
            "selection_algorithm_version": (T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION),
            "selection_is_outcome_independent": True,
            "selection_namespace": namespace,
            "selection_score": (
                "sha256(domain-separator,nul,namespace,nul,source,nul,"
                "candidate-identity-sha256)-ascending-v1"
            ),
            "ordering": "vstar-arxivqa-thinklite-screened-row-order-v1",
            "sources": source_records,
            "source_quotas": quotas,
            "source_counts": selected_counts,
            "rows": rows,
            "logical_attempts": rows * 8,
            "candidates_path": str(root / "candidates.jsonl"),
            "candidates_sha256": candidates_sha256,
        }
        manifest_payload = canonical_json_line(manifest)
        manifest_path = temporary_root / "manifest.json"
        with manifest_path.open("wb") as output:
            output.write(manifest_payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_root, root)
        return {
            "rows": rows,
            "logical_attempts": rows * 8,
            "source_counts": selected_counts,
            "candidates_sha256": candidates_sha256,
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "output_root": str(root),
            "candidates_path": str(root / "candidates.jsonl"),
            "manifest_path": str(root / "manifest.json"),
        }
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


__all__ = [
    "T1_RECOMMENDED_SELECTION_ALGORITHM_VERSION",
    "T1_RECOMMENDED_SELECTION_MANIFEST_SCHEMA",
    "T1_RECOMMENDED_SELECTION_NAMESPACE",
    "T1_RECOMMENDED_SELECTION_ORDER",
    "T1_RECOMMENDED_SELECTION_ROWS",
    "T1_RECOMMENDED_SOURCE_QUOTAS",
    "materialize_t1_recommended_selection",
]
