"""Immutable file helpers for texture benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from .schema import canonical_json_sha256, file_sha256
from .task import TextureTask, load_texture_tasks


TEXTURE_BENCHMARK_IDENTITY_SCHEMA = "tgvf-texture-benchmark-identity-v1"


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def write_bytes_idempotent(path: str | Path, payload: bytes) -> Path:
    """Atomically create an artifact, accepting an identical prior file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError(f"artifact output is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise RuntimeError(f"immutable artifact differs: {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != payload
            ):
                raise RuntimeError(f"immutable artifact differs: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_jsonl_idempotent(
    path: str | Path, rows: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    payload = canonical_jsonl_bytes(rows)
    destination = write_bytes_idempotent(path, payload)
    return {
        "path": str(destination.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "line_count": payload.count(b"\n"),
    }


def write_json_idempotent(path: str | Path, value: object) -> dict[str, object]:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = write_bytes_idempotent(path, payload)
    return {
        "path": str(destination.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def reindex_task_rows(
    rows: Iterable[Mapping[str, object]], *, start: int = 0
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    for ordinal, source in enumerate(rows, start):
        row = dict(source)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("task row lacks an explicit sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate task sample_id: {sample_id}")
        sample_ids.add(sample_id)
        row["ordinal"] = ordinal
        row["row_number"] = ordinal
        row["index"] = sample_id
        result.append(row)
    return result


def validate_task_manifest(
    path: str | Path,
    *,
    expected_count: int,
    expected_sha256: str | None = None,
    verify_images: bool = True,
) -> tuple[TextureTask, ...]:
    return load_texture_tasks(
        path,
        expected_count=expected_count,
        expected_sha256=expected_sha256,
        verify_images=verify_images,
    )


def build_benchmark_identity(
    *,
    benchmark_id: str,
    tasks_path: str | Path,
    task_count: int,
    components: Mapping[str, object],
) -> dict[str, object]:
    path = Path(tasks_path).resolve()
    content: dict[str, object] = {
        "schema_version": TEXTURE_BENCHMARK_IDENTITY_SCHEMA,
        "benchmark_id": benchmark_id,
        "task_manifest": {
            "path": str(path),
            "sha256": file_sha256(path),
            "task_count": task_count,
            "single_image_count": task_count,
        },
        "components": dict(components),
    }
    return {**content, "identity_sha256": canonical_json_sha256(content)}


def validate_benchmark_identity(
    path: str | Path, *, verify_tasks: bool = True, verify_images: bool = False
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("texture benchmark identity is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("texture benchmark identity must be an object")
    if payload.get("schema_version") != TEXTURE_BENCHMARK_IDENTITY_SCHEMA:
        raise ValueError("texture benchmark identity schema differs")
    declared = payload.get("identity_sha256")
    content = dict(payload)
    content.pop("identity_sha256", None)
    if declared != canonical_json_sha256(content):
        raise ValueError("texture benchmark identity digest differs")
    binding = payload.get("task_manifest")
    if not isinstance(binding, dict):
        raise ValueError("texture benchmark task binding is malformed")
    tasks_path = Path(str(binding.get("path", "")))
    if not tasks_path.is_absolute() or not tasks_path.is_file():
        raise ValueError("texture benchmark task path is missing or relative")
    if file_sha256(tasks_path) != binding.get("sha256"):
        raise ValueError("texture benchmark task bytes changed")
    if verify_tasks:
        count = binding.get("task_count")
        if type(count) is not int or count <= 0:
            raise ValueError("texture benchmark task count is malformed")
        validate_task_manifest(
            tasks_path,
            expected_count=count,
            expected_sha256=str(binding["sha256"]),
            verify_images=verify_images,
        )
    return payload


__all__ = [
    "TEXTURE_BENCHMARK_IDENTITY_SCHEMA",
    "build_benchmark_identity",
    "canonical_jsonl_bytes",
    "reindex_task_rows",
    "validate_benchmark_identity",
    "validate_task_manifest",
    "write_bytes_idempotent",
    "write_json_idempotent",
    "write_jsonl_idempotent",
]
