"""Canonical JSON and scalar validation primitives for policy selection."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from tgvf_rl.public_api_compat import rebind_public_function

T1_ATTEMPT_SEED_SCHEMA = "tgvf.policy-selection.t1-attempt-seed.v1"
T1_ATTEMPTS = 8
T1_SHARD_COUNT = 4
T1_SEED_MODULUS = 2**31 - 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc
    return encoded.encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {constant}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field_name} fields differ; missing={missing}, unknown={unknown}"
        )


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, stripped string")
    return value


def _required_sha256(value: Any, *, field_name: str) -> str:
    value = _required_string(value, field_name=field_name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_int(
    value: Any, *, field_name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def _required_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _required_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def _absolute_normal_path(value: Any, *, field_name: str) -> Path:
    raw = _required_string(value, field_name=field_name)
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        raise ValueError(f"{field_name} must be an absolute normalized path")
    return path


def _safe_relative_path(value: Any, *, field_name: str) -> Path:
    raw = _required_string(value, field_name=field_name)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or os.path.normpath(raw) != raw:
        raise ValueError(f"{field_name} must be a safe normalized relative path")
    return path


def _json_clone(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def derive_t1_attempt_seed(
    *,
    run_manifest_sha256: str,
    candidate_sha256: str,
    attempt_index: int,
    seed_root: int = 0,
    seed_namespace: str = "qwen3-policy-selection-t1-v1",
) -> int:
    """Derive one batch/rank-invariant low-31-bit attempt seed."""

    _required_sha256(run_manifest_sha256, field_name="run_manifest_sha256")
    _required_sha256(candidate_sha256, field_name="candidate_sha256")
    _required_int(attempt_index, field_name="attempt_index", maximum=T1_ATTEMPTS - 1)
    _required_int(seed_root, field_name="seed_root", maximum=2**63 - 1)
    _required_string(seed_namespace, field_name="seed_namespace")
    state = {
        "schema": T1_ATTEMPT_SEED_SCHEMA,
        "run_manifest_sha256": run_manifest_sha256,
        "candidate_sha256": candidate_sha256,
        "attempt_index": attempt_index,
        "seed_root": seed_root,
        "seed_namespace": seed_namespace,
    }
    digest = hashlib.sha256(
        b"tgvf-policy-selection-t1-seed-v1\0" + _canonical_json_bytes(state)
    ).digest()
    return int.from_bytes(digest[:8], "big") % T1_SEED_MODULUS


def candidate_rank(candidate_sha256: str, *, world_size: int = T1_SHARD_COUNT) -> int:
    """Assign all attempts for one candidate to one stable rank."""

    value = _required_sha256(candidate_sha256, field_name="candidate_sha256")
    size = _required_int(world_size, field_name="world_size", minimum=1)
    return int(value, 16) % size


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
for _public_function in (derive_t1_attempt_seed, candidate_rank):
    rebind_public_function(
        _public_function,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
del _public_function

__all__ = [
    "T1_ATTEMPTS",
    "T1_ATTEMPT_SEED_SCHEMA",
    "T1_SEED_MODULUS",
    "T1_SHARD_COUNT",
    "candidate_rank",
    "derive_t1_attempt_seed",
]
