"""Dependency-light argparse validators for runtime-locator authority."""

from __future__ import annotations

import argparse
from pathlib import Path


def absolute_runtime_locator_manifest_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "runtime-locator manifest path must be lexical absolute"
        )
    return path


def lowercase_runtime_locator_manifest_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "runtime-locator manifest SHA256 must be 64 lowercase hex characters"
        )
    return value


def positive_runtime_locator_manifest_byte_length(value: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise argparse.ArgumentTypeError(
            "runtime-locator manifest byte length must be a positive integer"
        )
    return int(value)


__all__ = [
    "absolute_runtime_locator_manifest_path",
    "lowercase_runtime_locator_manifest_sha256",
    "positive_runtime_locator_manifest_byte_length",
]
