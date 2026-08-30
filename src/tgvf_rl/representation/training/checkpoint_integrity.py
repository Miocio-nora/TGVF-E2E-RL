"""Serialization and value-integrity primitives for representation checkpoints.

This leaf is deliberately schema-agnostic: it knows how to normalize and
digest checkpoint state and how to publish/load one torch payload, but it does
not import any representation checkpoint identity or manifest type.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.public_api_compat import rebind_public_function
from tgvf_rl.representation.adapter import TGVFAdapter


_HEX = frozenset("0123456789abcdef")


def _adapter_state_to_cpu(adapter: TGVFAdapter) -> dict[str, torch.Tensor]:
    _require_adapter(adapter)
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in adapter.artifact_state_dict().items()
    }


def _plain_cpu_state(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _plain_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_cpu_state(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint state value {type(value).__qualname__}")


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _update_state_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(_tensor_checksum(value).encode())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        ordered = sorted(value.items(), key=lambda item: _mapping_key(item[0]))
        for key, item in ordered:
            _update_state_digest(digest, key)
            _update_state_digest(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, Enum):
        digest.update(b"enum\0")
        _update_state_digest(digest, value.value)
    elif isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
    elif isinstance(value, bool):
        digest.update(b"bool\0")
        digest.update(b"1" if value else b"0")
    elif isinstance(value, int):
        digest.update(b"int\0")
        digest.update(str(value).encode())
    elif isinstance(value, float):
        digest.update(b"float\0")
        digest.update(json.dumps(value, allow_nan=False).encode())
    elif value is None:
        digest.update(b"none\0")
    else:
        raise TypeError(
            f"unsupported checkpoint digest value {type(value).__qualname__}"
        )


def _mapping_key(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", "1" if value else "0")
    if isinstance(value, int):
        return ("int", str(value))
    raise TypeError(f"unsupported checkpoint mapping key {type(value).__qualname__}")


def _tensor_checksum(value: torch.Tensor) -> str:
    # The shared checksum helper expects at least one dimension when re-viewing
    # bytes; optimizer state legitimately contains scalar step tensors.
    canonical = value if value.ndim else value.reshape(1)
    return tensor_checksum(canonical)


def _save_atomic(value: object, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _torch_load(path: str | Path) -> object:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise ReplayMismatchError(
            f"cannot load representation checkpoint: {error}"
        ) from error


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _require_adapter(adapter: object) -> None:
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _strictly_increasing_non_negative_ints(values: object, *, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{field_name} must contain integers")
    if any(value < 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError(f"{field_name} must be unique and strictly increasing")


def _positive_finite_float(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be an explicit positive finite float")


def _non_negative_finite_float(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be an explicit non-negative finite float")


def _finite_ratio(value: object, *, field_name: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be an explicit finite float in [0,1]")


def _runtime_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"runtime optimizer {field_name} must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"runtime optimizer {field_name} must be finite")
    return resolved


def _runtime_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"runtime optimizer {field_name} must be bool")
    return value


def _runtime_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"runtime optimizer {field_name} must be bool or None")
    return value


def _validate_run_identity_contract(
    identity: object,
    *,
    run_identity_type: type[object],
    run_identity_v3_type: type[object],
    schema_version_v2: str,
    schema_version_v3: str,
    validate_accumulation: Callable[[object], None],
) -> None:
    """Re-run nested schema invariants without importing schema declarations."""

    if not isinstance(identity, run_identity_type):
        raise TypeError("run identity must be a RepresentationRunIdentity")
    schema_version = getattr(identity, "schema_version", None)
    if type(identity) is run_identity_type:
        expected_schema_version = schema_version_v2
    elif type(identity) is run_identity_v3_type:
        expected_schema_version = schema_version_v3
    else:
        raise TypeError("unsupported representation run identity type")
    if schema_version != expected_schema_version:
        raise ValueError("representation run identity schema mismatch")
    identity.code.__post_init__()
    identity.model.__post_init__()
    identity.provider.__post_init__()
    identity.objective.__post_init__()
    identity.adapter_contract.__post_init__()
    validate_accumulation(identity.accumulation)
    identity.optimizer.__post_init__()
    if identity.scheduler is not None:
        identity.scheduler.__post_init__()
    identity.trainer_execution.__post_init__()
    identity.initialization.__post_init__()
    identity.sampler_contract.__post_init__()
    if isinstance(identity, run_identity_v3_type):
        identity.validation_identity.__post_init__()
    identity.__post_init__()


def _validate_tensor_manifest_contract(
    entries: object,
    *,
    entry_type: type[object],
) -> None:
    """Validate an ordered tensor manifest without importing its entry type."""

    if not isinstance(entries, tuple) or not entries:
        raise ValueError("Adapter tensor manifest must be a non-empty tuple")
    for entry in entries:
        if not isinstance(entry, entry_type):
            raise TypeError("Adapter tensor manifest has an invalid entry")
        entry.__post_init__()
    names = tuple(entry.name for entry in entries)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("Adapter tensor manifest names must be unique and sorted")


_PUBLIC_CHECKPOINT_MODULE = "tgvf_rl.representation.training.checkpoint"
_INTEGRITY_FUNCTIONS = (
    _adapter_state_to_cpu,
    _plain_cpu_state,
    _state_digest,
    _update_state_digest,
    _mapping_key,
    _tensor_checksum,
    _save_atomic,
    _torch_load,
    _qualified_type,
    _require_adapter,
    _non_empty_text,
    _sha256,
    _positive_int,
    _integer,
    _non_negative_int,
    _strictly_increasing_non_negative_ints,
    _positive_finite_float,
    _non_negative_finite_float,
    _finite_ratio,
    _runtime_float,
    _runtime_bool,
    _runtime_optional_bool,
)

for _function in _INTEGRITY_FUNCTIONS:
    rebind_public_function(
        _function,
        implementation_module=__name__,
        public_module=_PUBLIC_CHECKPOINT_MODULE,
    )
del _function


__all__ = [function.__name__ for function in _INTEGRITY_FUNCTIONS]
