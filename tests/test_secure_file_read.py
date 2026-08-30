from __future__ import annotations

import errno
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tgvf_rl.secure_file_read as secure_file_read
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    create_regular_file_exclusive_beneath_nofollow,
    probe_regular_file_absolute_nofollow,
    read_regular_file_absolute_nofollow,
    read_regular_file_beneath_nofollow,
    read_regular_file_leaf_nofollow,
    retain_directory_absolute_nofollow,
)


def test_leaf_reader_returns_bytes_and_descriptor_metadata(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"contract bytes")

    snapshot = read_regular_file_leaf_nofollow(source)

    assert snapshot.payload == b"contract bytes"
    assert snapshot.before.st_dev == snapshot.after.st_dev
    assert snapshot.before.st_ino == snapshot.after.st_ino
    assert snapshot.before.st_size == snapshot.after.st_size == len(snapshot.payload)


def test_leaf_and_absolute_readers_have_distinct_ancestor_semantics(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "payload.bin"
    source.write_bytes(b"payload")
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    assert read_regular_file_leaf_nofollow(alias / source.name).payload == b"payload"
    with pytest.raises(OSError):
        read_regular_file_absolute_nofollow(alias / source.name)


def test_absolute_probe_returns_only_metadata_without_reading_large_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-sparse.bin"
    expected_size = 16 * 1024**3
    with source.open("wb") as stream:
        stream.truncate(expected_size)

    def _unexpected_read(_descriptor: int, _size: int) -> bytes:
        raise AssertionError("presence probe must not read payload bytes")

    monkeypatch.setattr(secure_file_read.os, "read", _unexpected_read)

    probe = probe_regular_file_absolute_nofollow(source)

    assert probe.metadata.st_size == expected_size


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_absolute_probe_rejects_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "payload.bin"
    source.write_bytes(b"payload")
    if symlink_kind == "leaf":
        requested = tmp_path / "payload.bin"
        requested.symlink_to(source)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        requested = alias / source.name

    with pytest.raises(OSError):
        probe_regular_file_absolute_nofollow(requested)


def test_readers_reject_final_symlink(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)

    with pytest.raises(OSError):
        read_regular_file_leaf_nofollow(alias)
    with pytest.raises(OSError):
        read_regular_file_absolute_nofollow(alias)


@pytest.mark.parametrize(
    "relative", ["", "/absolute", "../escape", "a/../b", "a//b", "a/./b", "nul\x00byte"]
)
def test_beneath_reader_rejects_noncanonical_relative_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(SecureFileReadError):
            read_regular_file_beneath_nofollow(descriptor, relative)
    finally:
        os.close(descriptor)


def test_beneath_reader_is_anchored_to_directory_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"anchored")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        root.rename(tmp_path / "renamed-root")
        snapshot = read_regular_file_beneath_nofollow(
            descriptor,
            "nested/payload.bin",
        )
    finally:
        os.close(descriptor)

    assert snapshot.payload == b"anchored"


def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    program = (
        "from tgvf_rl.secure_file_read import "
        "SecureFileReadError, read_regular_file_leaf_nofollow; "
        "import sys; "
        "\ntry:\n read_regular_file_leaf_nofollow(sys.argv[1])"
        "\nexcept SecureFileReadError:\n raise SystemExit(0)"
        "\nraise SystemExit(1)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program, str(fifo)],
        check=False,
        timeout=2.0,
    )

    assert completed.returncode == 0


def test_absolute_probe_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    program = (
        "from tgvf_rl.secure_file_read import "
        "SecureFileReadError, probe_regular_file_absolute_nofollow; "
        "import sys; "
        "\ntry:\n probe_regular_file_absolute_nofollow(sys.argv[1])"
        "\nexcept SecureFileReadError:\n raise SystemExit(0)"
        "\nraise SystemExit(1)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program, str(fifo)],
        check=False,
        timeout=2.0,
    )

    assert completed.returncode == 0


def test_absolute_probe_closes_success_and_failure_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    nonregular = tmp_path / "directory"
    nonregular.mkdir()
    opened: list[int] = []
    original_open = secure_file_read._open_path

    def _record_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(secure_file_read, "_open_path", _record_open)

    probe_regular_file_absolute_nofollow(source)
    with pytest.raises(SecureFileReadError, match="not a regular file"):
        probe_regular_file_absolute_nofollow(nonregular)

    assert opened
    for descriptor in set(opened):
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_missing_nofollow_support_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    monkeypatch.delattr(secure_file_read.os, "O_NOFOLLOW")

    with pytest.raises(SecureFileReadError, match="O_NOFOLLOW"):
        read_regular_file_leaf_nofollow(source)


def test_retained_directory_exclusive_create_burns_name_without_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with retain_directory_absolute_nofollow(root) as binding:
        creation = create_regular_file_exclusive_beneath_nofollow(
            binding,
            "rank-0.json",
            b'{"status":"consumed"}\n',
            mode=0o600,
        )
        with pytest.raises(FileExistsError):
            create_regular_file_exclusive_beneath_nofollow(
                binding,
                "rank-0.json",
                b"replacement",
                mode=0o600,
            )

    assert (root / "rank-0.json").read_bytes() == b'{"status":"consumed"}\n'
    assert creation.metadata.st_ino == (root / "rank-0.json").stat().st_ino
    assert creation.payload_sha256


def test_exclusive_create_keeps_tombstone_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    def _fail_after_reservation(_descriptor: int, _payload: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(secure_file_read, "_write_all", _fail_after_reservation)
    with retain_directory_absolute_nofollow(root) as binding:
        with pytest.raises(OSError, match="injected write failure"):
            create_regular_file_exclusive_beneath_nofollow(
                binding,
                "rank-1.json",
                b"receipt",
                mode=0o600,
            )

    tombstone = root / "rank-1.json"
    assert tombstone.exists()
    assert tombstone.read_bytes() == b""


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_retained_directory_rejects_symlink_components(
    tmp_path: Path,
    kind: str,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    if kind == "leaf":
        requested = tmp_path / "alias"
        requested.symlink_to(actual, target_is_directory=True)
    else:
        ancestor = tmp_path / "ancestor"
        ancestor.symlink_to(actual, target_is_directory=True)
        requested = ancestor / "nested"
        (actual / "nested").mkdir()

    with pytest.raises(SecureFileReadError, match="symlink"):
        retain_directory_absolute_nofollow(requested)


def test_retained_directory_detects_absolute_path_inode_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    with retain_directory_absolute_nofollow(root) as binding:
        root.rename(tmp_path / "old-private")
        root.mkdir()
        with pytest.raises(SecureFileReadError, match="identity changed"):
            binding.assert_path_binding()
