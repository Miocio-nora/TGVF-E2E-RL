from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
import threading

import pytest

from tgvf_rl import immutable_publication
from tgvf_rl.immutable_publication import (
    ImmutableContentCollisionError,
    ImmutableDestinationTypeError,
    publish_bytes_content_consistent,
    publish_bytes_create_only,
)


Publisher = Callable[[Path, bytes], None]


def test_create_only_publication_is_durable_and_never_accepts_existing_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    fsynced_types: list[int] = []
    link_calls: list[tuple[int | None, int | None, bool]] = []
    original_fsync = os.fsync
    original_link = os.link

    def record_fsync(descriptor: int) -> None:
        fsynced_types.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    def record_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        link_calls.append((src_dir_fd, dst_dir_fd, follow_symlinks))
        original_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(immutable_publication.os, "fsync", record_fsync)
    monkeypatch.setattr(immutable_publication.os, "link", record_link)

    publish_bytes_create_only(destination, b"first")

    assert destination.read_bytes() == b"first"
    assert any(stat.S_ISREG(mode) for mode in fsynced_types)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_types)
    assert len(link_calls) == 1
    assert link_calls[0][0] == link_calls[0][1]
    assert link_calls[0][2] is False
    with pytest.raises(FileExistsError, match="already exists"):
        publish_bytes_create_only(destination, b"first")
    assert destination.read_bytes() == b"first"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_content_consistent_publication_accepts_only_byte_identical_retry(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"

    publish_bytes_content_consistent(destination, b"stable")
    publish_bytes_content_consistent(destination, b"stable")

    with pytest.raises(ImmutableContentCollisionError, match="content differs"):
        publish_bytes_content_consistent(destination, b"different")
    assert destination.read_bytes() == b"stable"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


@pytest.mark.parametrize(
    "publisher",
    (publish_bytes_create_only, publish_bytes_content_consistent),
)
def test_immutable_publication_rejects_symlink_without_touching_target(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"protected")
    destination = tmp_path / "artifact.bin"
    destination.symlink_to(target)

    with pytest.raises(ImmutableDestinationTypeError, match="not a regular file"):
        publisher(destination, b"protected")

    assert destination.is_symlink()
    assert target.read_bytes() == b"protected"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


@pytest.mark.parametrize(
    "publisher",
    (publish_bytes_create_only, publish_bytes_content_consistent),
)
def test_immutable_publication_rejects_non_regular_destination(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.mkdir()

    with pytest.raises(ImmutableDestinationTypeError, match="not a regular file"):
        publisher(destination, b"payload")

    assert destination.is_dir()
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_content_consistent_concurrent_publishers_resolve_by_winner_content(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    barrier = threading.Barrier(2)

    def publish(payload: bytes) -> str:
        barrier.wait()
        try:
            publish_bytes_content_consistent(destination, payload)
        except ImmutableContentCollisionError:
            return "collision"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(publish, (b"alpha", b"beta")))

    assert sorted(outcomes) == ["collision", "published"]
    assert destination.read_bytes() in {b"alpha", b"beta"}
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_content_consistent_concurrent_identical_publishers_all_succeed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    worker_count = 8
    barrier = threading.Barrier(worker_count)

    def publish(_: int) -> None:
        barrier.wait()
        publish_bytes_content_consistent(destination, b"shared")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        tuple(executor.map(publish, range(worker_count)))

    assert destination.read_bytes() == b"shared"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_create_only_concurrent_publishers_have_exactly_one_winner(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    worker_count = 8
    barrier = threading.Barrier(worker_count)

    def publish(index: int) -> bool:
        barrier.wait()
        try:
            publish_bytes_create_only(destination, f"payload-{index}".encode())
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = tuple(executor.map(publish, range(worker_count)))

    assert sum(outcomes) == 1
    assert destination.read_bytes().startswith(b"payload-")
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_directory_descriptor_closes_when_durability_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    original_close = os.close
    original_fsync = os.fsync
    failed_directory_descriptor: int | None = None
    closed_descriptors: list[int] = []

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal failed_directory_descriptor
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            failed_directory_descriptor = descriptor
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(immutable_publication.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(immutable_publication.os, "close", record_close)

    with pytest.raises(OSError, match="directory fsync failed"):
        publish_bytes_create_only(destination, b"published-before-fsync")

    assert failed_directory_descriptor is not None
    assert failed_directory_descriptor in closed_descriptors
    assert destination.read_bytes() == b"published-before-fsync"
