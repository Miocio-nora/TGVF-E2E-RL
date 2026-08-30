"""Canonical artifact publication for representation internal evaluation.

This module deliberately owns only the immutable JSON publication boundary.
The evaluation runner re-exports these symbols so existing callers retain the
same public imports while artifact serialization stays independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .internal_evaluation import RepresentationInternalEvaluationReport


REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION = (
    "representation_internal_evaluation_artifact_v1"
)


@dataclass(frozen=True, slots=True)
class RepresentationInternalEvaluationArtifact:
    path: str
    payload_sha256: str
    byte_count: int
    schema_version: str = REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION


def save_representation_internal_evaluation_report_atomic(
    report: RepresentationInternalEvaluationReport,
    path: str | Path,
) -> RepresentationInternalEvaluationArtifact:
    """Create one immutable canonical JSON artifact without overwriting a file."""

    # Keep the import local so this lightweight publication module does not
    # import the model/evaluation runtime merely to expose its artifact type.
    from .internal_evaluation import RepresentationInternalEvaluationReport

    if not isinstance(report, RepresentationInternalEvaluationReport):
        raise TypeError("report must be RepresentationInternalEvaluationReport")
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("internal-evaluation artifact path must be absolute")
    if not destination.parent.is_dir():
        raise ValueError("internal-evaluation artifact parent must already exist")
    payload = (
        json.dumps(
            _json_value(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return RepresentationInternalEvaluationArtifact(
        path=str(destination),
        payload_sha256=sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"internal-evaluation report contains {type(value).__name__}")


# Preserve the established import and pickle identities while implementation
# ownership moves to this dependency-light module.
RepresentationInternalEvaluationArtifact.__module__ = (
    "tgvf_rl.representation.training.internal_evaluation"
)
save_representation_internal_evaluation_report_atomic.__module__ = (
    "tgvf_rl.representation.training.internal_evaluation"
)


__all__ = [
    "REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "RepresentationInternalEvaluationArtifact",
    "save_representation_internal_evaluation_report_atomic",
]
