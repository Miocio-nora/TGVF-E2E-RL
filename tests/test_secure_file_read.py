from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import tgvf_rl.secure_file_read as secure_file_read
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    read_regular_file_absolute_nofollow,
    read_regular_file_beneath_nofollow,
    read_regular_file_leaf_nofollow,
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


def test_readers_reject_final_symlink(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)

    with pytest.raises(OSError):
        read_regular_file_leaf_nofollow(alias)
    with pytest.raises(OSError):
        read_regular_file_absolute_nofollow(alias)


@pytest.mark.parametrize("relative", ["", "/absolute", "../escape", "a/../b", "a//b", "a/./b", "nul\x00byte"])
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


def test_missing_nofollow_support_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    monkeypatch.delattr(secure_file_read.os, "O_NOFOLLOW")

    with pytest.raises(SecureFileReadError, match="O_NOFOLLOW"):
        read_regular_file_leaf_nofollow(source)
