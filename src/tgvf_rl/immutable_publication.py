"""Race-safe, no-replace publication primitives for immutable file leaves."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import stat
from uuid import uuid4


class ImmutablePublicationError(RuntimeError):
    """Base class for a rejected immutable-file publication."""


class ImmutableDestinationTypeError(ImmutablePublicationError):
    """The destination leaf exists but is not a regular file."""


class ImmutableContentCollisionError(ImmutablePublicationError):
    """The destination leaf contains bytes other than the proposed payload."""


class ImmutablePublicationRaceError(ImmutablePublicationError):
    """The destination leaf did not stabilize while resolving a link race."""


_PUBLICATION_RACE_RETRIES = 16
_READ_CHUNK_BYTES = 1024 * 1024


def publish_bytes_create_only(path: str | os.PathLike[str], payload: bytes) -> None:
    """Durably create one immutable regular file and fail if its leaf exists.

    The caller owns parent-directory creation. Publication writes and fsyncs a
    private temporary file in that directory, then installs it with a hard link,
    which is an atomic no-replace operation. A regular existing destination
    raises :class:`FileExistsError`; a symlink or other non-regular leaf is
    rejected with :class:`ImmutableDestinationTypeError`.
    """

    _publish_bytes(path, payload, content_consistent=False)


def publish_bytes_content_consistent(
    path: str | os.PathLike[str], payload: bytes
) -> None:
    """Durably create an immutable file or accept a byte-identical winner.

    Concurrent writers of the same payload all succeed. A different regular
    winner raises :class:`ImmutableContentCollisionError`; a symlink or other
    non-regular destination raises :class:`ImmutableDestinationTypeError`.
    The caller owns parent-directory creation.
    """

    _publish_bytes(path, payload, content_consistent=True)


def _publish_bytes(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    content_consistent: bool,
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("immutable publication payload must be bytes")
    destination = Path(path)
    if destination.name in {"", ".", ".."}:
        raise ValueError("immutable publication destination must name a file")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_descriptor = os.open(destination.parent, directory_flags)
    temporary_name = f".{destination.name}.{uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    temporary_exists = False
    try:
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
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temporary_descriptor, remaining)
            if written <= 0:
                raise OSError("short write during immutable publication")
            remaining = remaining[written:]
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None

        for _attempt in range(_PUBLICATION_RACE_RETRIES):
            try:
                os.link(
                    temporary_name,
                    destination.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                outcome = _existing_regular_file_matches(
                    directory_descriptor,
                    destination.name,
                    payload,
                    destination=destination,
                )
                if outcome is None:
                    continue
                if not content_consistent:
                    raise FileExistsError(
                        errno.EEXIST,
                        "immutable destination already exists",
                        os.fspath(destination),
                    ) from error
                if outcome:
                    break
                raise ImmutableContentCollisionError(
                    f"immutable destination content differs: {destination}"
                ) from error
            else:
                break
        else:
            raise ImmutablePublicationRaceError(
                f"immutable destination did not stabilize: {destination}"
            )
    finally:
        try:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
        finally:
            try:
                if temporary_exists:
                    try:
                        os.unlink(temporary_name, dir_fd=directory_descriptor)
                    except FileNotFoundError:
                        pass
            finally:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)


def _existing_regular_file_matches(
    directory_descriptor: int,
    leaf_name: str,
    payload: bytes,
    *,
    destination: Path,
) -> bool | None:
    """Return equality, or ``None`` when a raced-away leaf should be retried."""

    try:
        path_stat = os.stat(
            leaf_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        raise _destination_type_error(destination)

    try:
        descriptor = os.open(
            leaf_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        try:
            raced_stat = os.stat(
                leaf_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if error.errno == errno.ELOOP or not stat.S_ISREG(raced_stat.st_mode):
            raise _destination_type_error(destination) from error
        raise
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise _destination_type_error(destination)
        if (opened_before.st_dev, opened_before.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            return None
        matches = opened_before.st_size == len(payload)
        offset = 0
        while matches and offset < len(payload):
            block = os.read(descriptor, min(_READ_CHUNK_BYTES, len(payload) - offset))
            if not block or block != payload[offset : offset + len(block)]:
                matches = False
                break
            offset += len(block)
        if matches and os.read(descriptor, 1):
            matches = False
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        rebound = os.stat(
            leaf_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(rebound.st_mode):
        raise _destination_type_error(destination)
    stable_identity = (opened_after.st_dev, opened_after.st_ino) == (
        rebound.st_dev,
        rebound.st_ino,
    )
    stable_content_metadata = (
        opened_before.st_size,
        opened_before.st_mtime_ns,
        opened_before.st_ctime_ns,
    ) == (
        opened_after.st_size,
        opened_after.st_mtime_ns,
        opened_after.st_ctime_ns,
    )
    if not stable_identity or not stable_content_metadata:
        return None
    return matches


def _destination_type_error(destination: Path) -> ImmutableDestinationTypeError:
    return ImmutableDestinationTypeError(
        f"immutable destination is not a regular file: {destination}"
    )


__all__ = [
    "ImmutableContentCollisionError",
    "ImmutableDestinationTypeError",
    "ImmutablePublicationError",
    "ImmutablePublicationRaceError",
    "publish_bytes_content_consistent",
    "publish_bytes_create_only",
]
