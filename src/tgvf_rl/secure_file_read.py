"""Explicit contracts for reading regular files without symlink races.

The public readers deliberately have different path semantics. Callers must
choose whether only the final leaf, every component of an absolute path, or
every descendant below an already-bound directory descriptor is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import stat
from typing import Final


_READ_CHUNK_BYTES: Final = 1024 * 1024


class SecureFileReadError(RuntimeError):
    """The platform or opened object cannot satisfy the declared contract."""


@dataclass(frozen=True, slots=True)
class RegularFileSnapshot:
    """Bytes plus descriptor metadata immediately before and after the read."""

    payload: bytes
    before: os.stat_result
    after: os.stat_result


def read_regular_file_leaf_nofollow(
    path: str | os.PathLike[str],
) -> RegularFileSnapshot:
    """Read a regular file while refusing a symlink at the final path leaf.

    Ancestor symlinks retain normal operating-system path semantics. Use
    :func:`read_regular_file_absolute_nofollow` when every ancestor is part of
    the trust boundary.
    """

    descriptor = _open_path(os.fspath(path), _file_flags())
    try:
        return _read_regular_descriptor(descriptor)
    finally:
        os.close(descriptor)


def read_regular_file_absolute_nofollow(
    path: str | os.PathLike[str],
) -> RegularFileSnapshot:
    """Read an absolute regular-file path without following any symlink."""

    components = _absolute_components(path, owner="regular-file path")
    if not components:
        raise SecureFileReadError("regular-file path does not name a file")
    parent_descriptor = _open_absolute_directory_components(components[:-1])
    try:
        descriptor = _open_path(
            components[-1],
            _file_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            return _read_regular_descriptor(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def read_regular_file_beneath_nofollow(
    root_descriptor: int,
    relative_path: str | os.PathLike[str],
) -> RegularFileSnapshot:
    """Read below an already-bound directory fd without following symlinks."""

    _require_dir_fd_support()
    components = _relative_components(relative_path)
    current_descriptor = os.dup(root_descriptor)
    try:
        if not stat.S_ISDIR(os.fstat(current_descriptor).st_mode):
            raise SecureFileReadError("bound root descriptor is not a directory")
        for component in components[:-1]:
            next_descriptor = _open_path(
                component,
                _directory_flags(),
                dir_fd=current_descriptor,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        descriptor = _open_path(
            components[-1],
            _file_flags(),
            dir_fd=current_descriptor,
        )
        try:
            return _read_regular_descriptor(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(current_descriptor)


def read_regular_file_beneath_absolute_directory_nofollow(
    root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
) -> RegularFileSnapshot:
    """Bind an absolute directory chain, then read one relative regular file."""

    root_descriptor = _open_absolute_directory_components(
        _absolute_components(root, owner="root directory")
    )
    try:
        return read_regular_file_beneath_nofollow(root_descriptor, relative_path)
    finally:
        os.close(root_descriptor)


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if not isinstance(value, int):
        raise SecureFileReadError(f"platform lacks required {name} support")
    return value


def _require_dir_fd_support() -> None:
    if os.open not in os.supports_dir_fd:
        raise SecureFileReadError("platform lacks required openat/dir_fd support")


def _open_path(path: str, flags: int, *, dir_fd: int | None = None) -> int:
    if dir_fd is None:
        return os.open(path, flags)
    return os.open(path, flags, dir_fd=dir_fd)


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | _required_flag("O_CLOEXEC")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_NONBLOCK")
    )


def _directory_flags() -> int:
    _require_dir_fd_support()
    return _file_flags() | _required_flag("O_DIRECTORY")


def _absolute_components(
    path: str | os.PathLike[str],
    *,
    owner: str,
) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"{owner} must be text")
    if "\x00" in raw:
        raise SecureFileReadError(f"{owner} contains NUL")
    pure = PurePosixPath(raw)
    if not pure.is_absolute():
        raise SecureFileReadError(f"{owner} must be absolute")
    if pure.root != "/":
        raise SecureFileReadError(f"{owner} must use the POSIX filesystem root")
    components = pure.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise SecureFileReadError(f"{owner} is not lexically normalized")
    return components


def _relative_components(path: str | os.PathLike[str]) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("relative file path must be text")
    if "\x00" in raw:
        raise SecureFileReadError("relative file path contains NUL")
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or not pure.parts
        or any(component in {"", ".", ".."} for component in pure.parts)
        or pure.as_posix() != raw
    ):
        raise SecureFileReadError(
            "relative file path must be normalized and beneath root"
        )
    return pure.parts


def _open_absolute_directory_components(components: tuple[str, ...]) -> int:
    current_descriptor = _open_path("/", _directory_flags())
    try:
        if not stat.S_ISDIR(os.fstat(current_descriptor).st_mode):
            raise SecureFileReadError("filesystem root is not a directory")
        for component in components:
            next_descriptor = _open_path(
                component,
                _directory_flags(),
                dir_fd=current_descriptor,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return current_descriptor
    except BaseException:
        os.close(current_descriptor)
        raise


def _read_regular_descriptor(descriptor: int) -> RegularFileSnapshot:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SecureFileReadError("opened object is not a regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if not stat.S_ISREG(after.st_mode):
        raise SecureFileReadError("opened object ceased to be a regular file")
    return RegularFileSnapshot(payload=b"".join(chunks), before=before, after=after)


__all__ = [
    "RegularFileSnapshot",
    "SecureFileReadError",
    "read_regular_file_absolute_nofollow",
    "read_regular_file_beneath_absolute_directory_nofollow",
    "read_regular_file_beneath_nofollow",
    "read_regular_file_leaf_nofollow",
]
