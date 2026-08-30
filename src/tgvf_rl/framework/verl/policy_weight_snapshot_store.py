"""Descriptor-bound storage primitives for exact Policy weight snapshots.

This leaf owns only filesystem publication and read mechanics.  Policy/run
identity, tensor integrity, and orchestration remain in ``policy_weight_sync``.
The generic secure reader supplies descriptor-rooted traversal; the immutable
LoRA publication path deliberately retains its stronger private-owner, mode,
and single-link inode contract.
"""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import os
from pathlib import Path
import stat
import tempfile
from uuid import uuid4

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    read_regular_file_beneath_nofollow,
    read_regular_file_leaf_nofollow,
)


def atomic_replace_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_immutable_bytes(path: Path, value: bytes, *, owner: str) -> None:
    """Publish one private immutable file without replacing a concurrent winner."""

    directory_descriptor = open_snapshot_root(path.parent, create_missing=True)
    temporary_name = f".{path.name}.{uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    temporary_exists = False
    lock_acquired = False
    active_error: BaseException | None = None
    try:
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
            lock_acquired = True
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            temporary_exists = True
            remaining = memoryview(value)
            while remaining:
                written = os.write(temporary_descriptor, remaining)
                if written <= 0:
                    raise OSError(f"short write while publishing {owner}")
                remaining = remaining[written:]
            os.fchmod(temporary_descriptor, 0o600)
            os.fsync(temporary_descriptor)
            os.close(temporary_descriptor)
            temporary_descriptor = None
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                assert_immutable_file_equals_at(
                    directory_descriptor,
                    path.name,
                    value,
                    owner=owner,
                )
            os.unlink(temporary_name, dir_fd=directory_descriptor)
            temporary_exists = False
            os.fsync(directory_descriptor)
            assert_snapshot_root_path_binding(path.parent, directory_descriptor)
        except OSError as error:
            raise ReplayMismatchError(f"could not publish immutable {owner}") from error
    except BaseException as error:
        active_error = error

    cleanup_errors: list[tuple[str, BaseException]] = []

    def attempt_cleanup(action: str, cleanup: Callable[[], None]) -> None:
        try:
            cleanup()
        except BaseException as error:
            cleanup_errors.append((action, error))

    if temporary_descriptor is not None:
        attempt_cleanup(
            "close temporary descriptor",
            lambda: os.close(temporary_descriptor),
        )
    if temporary_exists:
        attempt_cleanup(
            "unlink temporary file",
            lambda: os.unlink(temporary_name, dir_fd=directory_descriptor),
        )
    if lock_acquired:
        attempt_cleanup(
            "unlock immutable publication directory",
            lambda: fcntl.flock(directory_descriptor, fcntl.LOCK_UN),
        )
    attempt_cleanup(
        "close immutable publication directory",
        lambda: os.close(directory_descriptor),
    )

    if active_error is not None:
        for action, cleanup_error in cleanup_errors:
            active_error.add_note(
                f"immutable {owner} cleanup also failed while attempting to {action}: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if cleanup_errors:
            raise active_error from cleanup_errors[0][1]
        raise active_error
    if cleanup_errors:
        cleanup_failure = ReplayMismatchError(
            f"immutable {owner} cleanup failed after publication"
        )
        for action, cleanup_error in cleanup_errors[1:]:
            cleanup_failure.add_note(
                f"additional cleanup failure while attempting to {action}: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise cleanup_failure from cleanup_errors[0][1]


def assert_immutable_file_equals_at(
    directory_descriptor: int,
    name: str,
    value: bytes,
    *,
    owner: str,
) -> None:
    """Verify the LoRA-specific private, owned, single-link winner contract."""

    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise ReplayMismatchError(
                f"existing content-addressed {owner} has unsafe inode metadata"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        if b"".join(chunks) != value:
            raise ReplayMismatchError(f"existing content-addressed {owner} differs")
        os.fsync(descriptor)
    except OSError as error:
        raise ReplayMismatchError(
            f"existing content-addressed {owner} is unreadable or symlinked"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_bytes(path: Path, owner: str) -> bytes:
    """Retain the historical leaf-only facade for callers outside root closures."""

    try:
        return read_regular_file_leaf_nofollow(path).payload
    except SecureFileReadError as error:
        if "regular file" in str(error):
            raise ReplayMismatchError(f"{owner} must be a regular file") from error
        raise ReplayMismatchError(f"{owner} is missing or unreadable") from error
    except OSError as error:
        raise ReplayMismatchError(f"{owner} is missing or unreadable") from error


def open_snapshot_root(root: Path, *, create_missing: bool = False) -> int:
    """Open, and optionally create, a root without following component symlinks."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    normalized = Path(os.path.abspath(os.fspath(root)))
    if not normalized.is_absolute():
        raise ValueError("LoRA snapshot root must be absolute")
    descriptor = os.open(os.sep, flags)
    completed = False
    try:
        for part in normalized.parts[1:]:
            next_descriptor: int | None = None
            try:
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                    raise ReplayMismatchError(
                        "LoRA snapshot root path contains a non-directory"
                    )
            except BaseException:
                if next_descriptor is not None:
                    os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        completed = True
        return descriptor
    except OSError as error:
        raise ReplayMismatchError(
            "LoRA snapshot root path is missing, unreadable, or contains a symlink"
        ) from error
    finally:
        if not completed:
            os.close(descriptor)


def safe_snapshot_relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayMismatchError("LoRA snapshot contains an unsafe path")
    if "\x00" in value or value.startswith("/"):
        raise ReplayMismatchError("LoRA snapshot contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ReplayMismatchError("LoRA snapshot contains an unsafe path")
    return Path(*parts)


def assert_snapshot_root_path_binding(root: Path, root_descriptor: int) -> None:
    """Reject replacement of the lexical root while its closure was read."""

    expected = os.fstat(root_descriptor)
    observed_descriptor = open_snapshot_root(root)
    try:
        observed = os.fstat(observed_descriptor)
        if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
            raise ReplayMismatchError("LoRA snapshot root changed while loading")
    finally:
        os.close(observed_descriptor)


def read_relative_file_bytes_at(
    root_descriptor: int,
    relative_path: str,
    owner: str,
) -> bytes:
    """Read one regular file beneath an already-bound snapshot root."""

    relative = safe_snapshot_relative_path(relative_path)
    try:
        return read_regular_file_beneath_nofollow(
            root_descriptor,
            relative.as_posix(),
        ).payload
    except (OSError, SecureFileReadError) as error:
        raise ReplayMismatchError(
            f"{owner} is missing or unreadable (including symlink rejection)"
        ) from error


__all__ = [
    "assert_immutable_file_equals_at",
    "assert_snapshot_root_path_binding",
    "atomic_replace_bytes",
    "fsync_directory",
    "open_snapshot_root",
    "read_bytes",
    "read_relative_file_bytes_at",
    "safe_snapshot_relative_path",
    "write_immutable_bytes",
]
