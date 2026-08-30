"""Explicit contracts for reading regular files without symlink races.

The public readers deliberately have different path semantics. Callers must
choose whether only the final leaf, every component of an absolute path, or
every descendant below an already-bound directory descriptor is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import PurePosixPath
import stat
from typing import Final
from weakref import finalize


_READ_CHUNK_BYTES: Final = 1024 * 1024


class SecureFileReadError(RuntimeError):
    """The platform or opened object cannot satisfy the declared contract."""


def _close_descriptor_best_effort(descriptor: int) -> None:
    """Close a descriptor from a GC finalizer without leaking an exception."""

    try:
        os.close(descriptor)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class RegularFileSnapshot:
    """Bytes plus descriptor metadata immediately before and after the read."""

    payload: bytes
    before: os.stat_result
    after: os.stat_result


@dataclass(frozen=True, slots=True)
class RegularFileProbe:
    """Metadata for a regular file opened under the absolute nofollow contract."""

    metadata: os.stat_result


@dataclass(frozen=True, slots=True)
class ExclusiveRegularFileCreation:
    """Identity of bytes durably created beneath a retained directory.

    The final path is never removed by this module after ``O_EXCL`` succeeds.
    Consequently, any later error leaves the name reserved as a fail-closed
    tombstone rather than reopening a one-use slot.
    """

    relative_path: str
    payload_sha256: str
    byte_length: int
    metadata: os.stat_result


class RetainedRegularFileDescriptor:
    """Owned regular-file descriptor retained across a trust boundary.

    The descriptor is opened by the absolute nofollow contract and remains
    owned by this object until :meth:`close` is called.  Callers must close an
    abandoned binding explicitly; the context-manager methods make that
    ownership contract convenient for preflight code.
    """

    __slots__ = ("__weakref__", "_descriptor", "_finalizer")

    def __init__(self, descriptor: int) -> None:
        if isinstance(descriptor, bool) or not isinstance(descriptor, int):
            raise TypeError("descriptor must be an integer")
        if descriptor < 0:
            raise ValueError("descriptor must be non-negative")
        self._descriptor: int | None = descriptor
        self._finalizer = finalize(
            self,
            _close_descriptor_best_effort,
            descriptor,
        )

    @property
    def closed(self) -> bool:
        """Whether ownership of the descriptor has already been released."""

        return self._descriptor is None

    def fileno(self) -> int:
        """Return the still-owned descriptor or fail closed after release."""

        descriptor = self._descriptor
        if descriptor is None:
            raise SecureFileReadError("retained regular-file descriptor is closed")
        return descriptor

    def snapshot(self) -> RegularFileSnapshot:
        """Re-read the retained inode from offset zero without reopening a path."""

        descriptor = self.fileno()
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as error:
            raise SecureFileReadError(
                "retained regular-file descriptor cannot be rewound"
            ) from error
        return _read_regular_descriptor(descriptor)

    def close(self) -> None:
        """Idempotently release the owned descriptor."""

        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        self._finalizer.detach()
        os.close(descriptor)

    def __enter__(self) -> RetainedRegularFileDescriptor:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __reduce__(self) -> object:
        raise TypeError(
            "RetainedRegularFileDescriptor is process-local and not serializable"
        )


class RetainedDirectoryDescriptor:
    """Process-local directory descriptor bound to one absolute path inode."""

    __slots__ = (
        "__weakref__",
        "_descriptor",
        "_finalizer",
        "_metadata",
        "_owner_pid",
        "_path",
    )

    def __init__(
        self,
        descriptor: int,
        *,
        path: str,
        metadata: os.stat_result,
    ) -> None:
        if isinstance(descriptor, bool) or not isinstance(descriptor, int):
            raise TypeError("descriptor must be an integer")
        if descriptor < 0:
            raise ValueError("descriptor must be non-negative")
        if type(path) is not str:
            raise TypeError("retained directory path must be exactly str")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SecureFileReadError("retained descriptor is not a directory")
        self._descriptor: int | None = descriptor
        self._path = path
        self._metadata = metadata
        self._owner_pid = os.getpid()
        self._finalizer = finalize(
            self,
            _close_descriptor_best_effort,
            descriptor,
        )

    @property
    def path(self) -> str:
        return self._path

    @property
    def metadata(self) -> os.stat_result:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._descriptor is None

    def _assert_owner_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise SecureFileReadError(
                "retained directory descriptor cannot be used after fork"
            )

    def fileno(self) -> int:
        self._assert_owner_process()
        descriptor = self._descriptor
        if descriptor is None:
            raise SecureFileReadError("retained directory descriptor is closed")
        return descriptor

    def assert_path_binding(self) -> os.stat_result:
        """Require the descriptor and current absolute path to name one inode."""

        descriptor_metadata = os.fstat(self.fileno())
        _assert_same_directory_identity(
            self._metadata,
            descriptor_metadata,
            label="retained directory descriptor",
        )
        components = _absolute_components(
            self._path,
            owner="retained directory path",
        )
        reopened = _open_absolute_directory_components(components)
        try:
            path_metadata = os.fstat(reopened)
        finally:
            os.close(reopened)
        _assert_same_directory_identity(
            self._metadata,
            path_metadata,
            label="retained directory path",
        )
        return descriptor_metadata

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        self._finalizer.detach()
        os.close(descriptor)

    def __enter__(self) -> RetainedDirectoryDescriptor:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __reduce__(self) -> object:
        raise TypeError(
            "RetainedDirectoryDescriptor is process-local and not serializable"
        )


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


def retain_regular_file_absolute_nofollow(
    path: str | os.PathLike[str],
) -> RetainedRegularFileDescriptor:
    """Open and retain an absolute regular file without following symlinks.

    Every ancestor and the final leaf are opened through descriptors with
    symlink following disabled.  Ownership of the validated leaf descriptor
    transfers to the returned object; every failure path closes it.
    """

    components = _absolute_components(path, owner="regular-file path")
    if not components:
        raise SecureFileReadError("regular-file path does not name a file")
    try:
        parent_descriptor = _open_absolute_directory_components(components[:-1])
    except OSError as error:
        raise SecureFileReadError(
            "regular-file path is missing, unreadable, or contains a symlink"
        ) from error
    descriptor: int | None = None
    try:
        try:
            descriptor = _open_path(
                components[-1],
                _file_flags(),
                dir_fd=parent_descriptor,
            )
            _probe_regular_descriptor(descriptor)
            retained = RetainedRegularFileDescriptor(descriptor)
            descriptor = None
            return retained
        except OSError as error:
            raise SecureFileReadError(
                "regular-file path is missing, unreadable, or contains a symlink"
            ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def retain_directory_absolute_nofollow(
    path: str | os.PathLike[str],
) -> RetainedDirectoryDescriptor:
    """Open and retain an absolute directory without following any symlink."""

    raw = os.fspath(path)
    components = _absolute_components(raw, owner="directory path")
    try:
        descriptor = _open_absolute_directory_components(components)
    except OSError as error:
        raise SecureFileReadError(
            "directory path is missing, unreadable, or contains a symlink"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):  # pragma: no cover - O_DIRECTORY
            raise SecureFileReadError("opened object is not a directory")
        retained = RetainedDirectoryDescriptor(
            descriptor,
            path=raw,
            metadata=metadata,
        )
        descriptor = -1
        return retained
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def create_regular_file_exclusive_beneath_nofollow(
    root: RetainedDirectoryDescriptor,
    relative_path: str | os.PathLike[str],
    payload: bytes,
    *,
    mode: int,
) -> ExclusiveRegularFileCreation:
    """Durably create one regular file below a retained directory.

    Every descendant is opened relative to a retained directory descriptor.
    The leaf uses required ``O_NOFOLLOW`` and ``O_EXCL`` flags.  Once the leaf
    exists, no failure path unlinks it; callers therefore get fail-closed
    one-use semantics, including crashes or partial writes.
    """

    if type(root) is not RetainedDirectoryDescriptor:
        raise TypeError("root must be exactly RetainedDirectoryDescriptor")
    if type(payload) is not bytes:
        raise TypeError("exclusive file payload must be exactly bytes")
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("exclusive file mode must be an integer")
    if mode < 0 or mode > 0o777:
        raise ValueError("exclusive file mode must contain only permission bits")
    components = _relative_components(relative_path)
    root.assert_path_binding()
    parent_descriptor = os.dup(root.fileno())
    leaf_descriptor: int | None = None
    try:
        for component in components[:-1]:
            next_descriptor = _open_path(
                component,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        parent_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_metadata.st_mode):  # pragma: no cover
            raise SecureFileReadError("exclusive file parent is not a directory")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _required_flag("O_CLOEXEC")
            | _required_flag("O_NOFOLLOW")
            | _required_flag("O_NONBLOCK")
        )
        leaf_descriptor = _open_path(
            components[-1],
            flags,
            dir_fd=parent_descriptor,
        )
        os.fchmod(leaf_descriptor, mode)
        reserved = os.fstat(leaf_descriptor)
        if not stat.S_ISREG(reserved.st_mode):  # pragma: no cover - O_EXCL create
            raise SecureFileReadError("exclusive file leaf is not a regular file")

        # Persist the directory entry before writing content.  Any subsequent
        # exception deliberately leaves this name burned.
        os.fsync(parent_descriptor)
        _write_all(leaf_descriptor, payload)
        os.fsync(leaf_descriptor)
        written = os.fstat(leaf_descriptor)
        if (
            written.st_dev != reserved.st_dev
            or written.st_ino != reserved.st_ino
            or not stat.S_ISREG(written.st_mode)
            or stat.S_IMODE(written.st_mode) != mode
            or written.st_size != len(payload)
        ):
            raise SecureFileReadError(
                "exclusive file descriptor identity or content length changed"
            )

        verifier = _open_path(
            components[-1],
            _file_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            snapshot = _read_regular_descriptor(verifier)
        finally:
            os.close(verifier)
        if (
            snapshot.before.st_dev != written.st_dev
            or snapshot.before.st_ino != written.st_ino
            or snapshot.after.st_dev != written.st_dev
            or snapshot.after.st_ino != written.st_ino
            or snapshot.payload != payload
        ):
            raise SecureFileReadError(
                "exclusive file path no longer binds the created inode and bytes"
            )
        after_parent = os.fstat(parent_descriptor)
        _assert_same_directory_identity(
            parent_metadata,
            after_parent,
            label="exclusive file parent",
        )
        os.fsync(parent_descriptor)
        root.assert_path_binding()
        return ExclusiveRegularFileCreation(
            relative_path=PurePosixPath(*components).as_posix(),
            payload_sha256=sha256(payload).hexdigest(),
            byte_length=len(payload),
            metadata=written,
        )
    finally:
        if leaf_descriptor is not None:
            os.close(leaf_descriptor)
        os.close(parent_descriptor)


def probe_regular_file_absolute_nofollow(
    path: str | os.PathLike[str],
) -> RegularFileProbe:
    """Open and validate an absolute regular-file path without reading it.

    Every path component is opened relative to the descriptor for its parent,
    with symlink following disabled.  The leaf is opened nonblocking, checked
    with ``fstat``, and closed immediately.  The returned contract contains
    metadata only; this function never reads or allocates storage for payload
    bytes.
    """

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
            return _probe_regular_descriptor(descriptor)
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
    if (
        any(component in {"", ".", ".."} for component in components)
        or pure.as_posix() != raw
    ):
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


def _probe_regular_descriptor(descriptor: int) -> RegularFileProbe:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise SecureFileReadError("opened object is not a regular file")
    return RegularFileProbe(metadata=metadata)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0:  # pragma: no cover - regular-file write contract
            raise SecureFileReadError("exclusive regular-file write made no progress")
        offset += written


def _assert_same_directory_identity(
    expected: os.stat_result,
    observed: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
        or observed.st_uid != expected.st_uid
        or observed.st_gid != expected.st_gid
        or stat.S_IMODE(observed.st_mode) != stat.S_IMODE(expected.st_mode)
    ):
        raise SecureFileReadError(f"{label} identity changed")


__all__ = [
    "ExclusiveRegularFileCreation",
    "RegularFileProbe",
    "RegularFileSnapshot",
    "RetainedDirectoryDescriptor",
    "RetainedRegularFileDescriptor",
    "SecureFileReadError",
    "create_regular_file_exclusive_beneath_nofollow",
    "probe_regular_file_absolute_nofollow",
    "read_regular_file_absolute_nofollow",
    "read_regular_file_beneath_absolute_directory_nofollow",
    "read_regular_file_beneath_nofollow",
    "read_regular_file_leaf_nofollow",
    "retain_directory_absolute_nofollow",
    "retain_regular_file_absolute_nofollow",
]
