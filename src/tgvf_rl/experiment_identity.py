"""Small fail-closed identities shared by bounded experiment drivers."""

from __future__ import annotations

import re


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(value: object) -> str:
    """Return a safe explicit run identifier; never infer one from a path."""

    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must be 1-128 characters, start with an alphanumeric, "
            "and contain only ASCII letters, digits, '.', '_' or '-'"
        )
    return value


__all__ = ["validate_run_id"]
