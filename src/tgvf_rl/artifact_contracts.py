"""Small, dependency-free serialization contracts for immutable artifacts."""

from __future__ import annotations

import hashlib
import json


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository's compact, finite, UTF-8 canonical JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 hex digest of ``canonical_json_bytes(value)``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = ["canonical_json_bytes", "canonical_json_sha256"]
