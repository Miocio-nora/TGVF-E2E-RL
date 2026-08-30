"""Primitive TOML value and path validators for training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

from tgvf_rl.secure_file_read import (
    RegularFileProbe,
    SecureFileReadError,
    probe_regular_file_absolute_nofollow,
    read_regular_file_absolute_nofollow,
)


_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, table: str
) -> None:
    actual = set(value)
    missing = sorted(set(expected) - actual)
    unknown = sorted(actual - set(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"[{table}] fields do not match schema: {' '.join(details)}")


def _table(value: Mapping[str, Any], key: str, *, table: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(f"[{table}.{key}] must be a TOML table")
    if any(not isinstance(name, str) for name in item):
        raise TypeError(f"[{table}.{key}] keys must be strings")
    return item


def _string(value: Mapping[str, Any], key: str, *, table: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{table}.{key} must be a non-empty string")
    return item


def _boolean(value: Mapping[str, Any], key: str, *, table: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise TypeError(f"{table}.{key} must be a boolean")
    return item


def _int(value: Mapping[str, Any], key: str, *, table: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{table}.{key} must be an integer")
    return item


def _float(value: Mapping[str, Any], key: str, *, table: str) -> float:
    item = value.get(key)
    if not isinstance(item, float) or not math.isfinite(item):
        raise TypeError(f"{table}.{key} must be an explicit finite TOML float")
    return item


def _int_tuple(value: Mapping[str, Any], key: str, *, table: str) -> tuple[int, ...]:
    item = value.get(key)
    if not isinstance(item, list) or not item:
        raise TypeError(f"{table}.{key} must be a non-empty integer array")
    if any(isinstance(entry, bool) or not isinstance(entry, int) for entry in item):
        raise TypeError(f"{table}.{key} must contain only integers")
    return tuple(item)


def _float_tuple(
    value: Mapping[str, Any], key: str, *, table: str, length: int
) -> tuple[float, ...]:
    item = value.get(key)
    if not isinstance(item, list) or len(item) != length:
        raise TypeError(f"{table}.{key} must be a {length}-float array")
    if any(not isinstance(entry, float) or not math.isfinite(entry) for entry in item):
        raise TypeError(f"{table}.{key} must contain explicit finite TOML floats")
    return tuple(item)


def _path(value: Mapping[str, Any], key: str, *, table: str, allow_empty: bool) -> Path:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise TypeError(f"{table}.{key} must be a path string")
    return _absolute_path(Path(item), field_name=f"{table}.{key}")


def _absolute_path(value: Path, *, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    if not value.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if "\x00" in str(value):
        raise ValueError(f"{field_name} contains a null byte")
    return value


def _existing_file_path(value: str | Path, *, field_name: str) -> Path:
    return _require_existing_file_probe(value, field_name=field_name)[0]


def _require_existing_file_probe(
    value: str | Path,
    *,
    field_name: str,
) -> tuple[Path, RegularFileProbe]:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a path")
    path = Path(value)
    _absolute_path(path, field_name=field_name)
    try:
        probe = probe_regular_file_absolute_nofollow(path)
    except (OSError, SecureFileReadError, TypeError) as error:
        raise ValueError(f"{field_name} does not resolve to a file: {path}") from error
    return path, probe


def _optional_existing_file_probe(
    value: str | Path,
    *,
    field_name: str,
) -> tuple[Path, RegularFileProbe] | None:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a path")
    path = Path(value)
    _absolute_path(path, field_name=field_name)
    try:
        probe = probe_regular_file_absolute_nofollow(path)
    except FileNotFoundError:
        return None
    except (OSError, SecureFileReadError, TypeError) as error:
        raise ValueError(f"{field_name} does not resolve to a file: {path}") from error
    return path, probe


def _read_existing_file_bytes(
    value: str | Path,
    *,
    field_name: str,
) -> tuple[Path, bytes]:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a path")
    path = Path(value)
    _absolute_path(path, field_name=field_name)
    try:
        payload = read_regular_file_absolute_nofollow(path).payload
    except (OSError, SecureFileReadError, TypeError):
        raise ValueError(f"{field_name} does not resolve to a file: {path}")
    return path, payload


def _configuration_source_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("configuration path must be a path")
    try:
        # Lexically absolutize without resolving filesystem components.  The
        # secure reader below intentionally rejects every ancestor/leaf
        # symlink instead of preserving the legacy resolve-and-follow behavior.
        path = Path(os.path.abspath(Path(value).expanduser()))
    except OSError as error:
        raise ValueError(
            f"configuration path does not resolve to a file: {value}"
        ) from error
    return path


def _read_configuration_source(value: str | Path) -> tuple[Path, bytes]:
    path = _configuration_source_path(value)
    try:
        payload = read_regular_file_absolute_nofollow(path).payload
    except SecureFileReadError as error:
        if str(error) in {
            "opened object is not a regular file",
            "opened object ceased to be a regular file",
        }:
            raise ValueError(f"configuration path is not a file: {path}") from error
        raise ValueError(
            f"configuration path does not resolve to a file: {value}"
        ) from error
    except (OSError, TypeError) as error:
        raise ValueError(
            f"configuration path does not resolve to a file: {value}"
        ) from error
    return path, payload


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _safe_filename(value: str, *, field_name: str) -> None:
    _non_empty_text(value, field_name=field_name)
    if value in {".", ".."} or Path(value).name != value or "\x00" in value:
        raise ValueError(f"{field_name} must be a plain filename prefix")


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def _positive_int(value: object, *, field_name: str) -> None:
    _integer(value, field_name=field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")
