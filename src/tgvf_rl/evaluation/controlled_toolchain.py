"""Deterministic compiler environments for local vLLM processes.

The host shell can carry compiler flags from unrelated projects.  vLLM and
Triton may compile extensions lazily, so merely overriding ``CC`` and ``CXX``
does not isolate a launch: inherited ``CFLAGS``, ``LD``, or Conda variables can
still redirect the toolchain.  This module owns the shared fail-closed purge
and controlled-overlay contract used by benchmark supervisors.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path


CONTROLLED_TOOLCHAIN_SCHEMA = "tgvf.controlled-toolchain-environment.v1"
PURGED_TOOLCHAIN_ENVIRONMENT = (
    "ADDR2LINE",
    "AR",
    "AS",
    "BUILD",
    "CC",
    "CC_FOR_BUILD",
    "CFLAGS",
    "CMAKE_ARGS",
    "CMAKE_PREFIX_PATH",
    "COMPILER_PATH",
    "CPP",
    "CPPFLAGS",
    "CPATH",
    "CXX",
    "CXXFILT",
    "CXXFLAGS",
    "CXX_FOR_BUILD",
    "DEBUG_CFLAGS",
    "DEBUG_CPPFLAGS",
    "DEBUG_CXXFLAGS",
    "ELFEDIT",
    "GCC",
    "GCC_AR",
    "GCC_EXEC_PREFIX",
    "GCC_NM",
    "GCC_RANLIB",
    "GPROF",
    "GXX",
    "HOST",
    "LD",
    "LDFLAGS",
    "LD_GOLD",
    "LIBRARY_PATH",
    "NM",
    "NVCC_PREPEND_FLAGS",
    "NVCC_PREPEND_FLAGS_BACKUP",
    "OBJCOPY",
    "OBJDUMP",
    "PATH",
    "RANLIB",
    "READELF",
    "SIZE",
    "STRINGS",
    "STRIP",
    "build_alias",
    "host_alias",
)
PURGED_ENVIRONMENT_PREFIXES = ("CONDA_", "_CONDA_")
SYSTEM_PATH_SUFFIX = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)


def python312_toolchain_environment(
    *, python_environment_root: str | Path, python_header_root: str | Path
) -> dict[str, str]:
    """Build the one controlled Python 3.12 compiler overlay."""

    environment_root = Path(python_environment_root)
    header_root = Path(python_header_root)
    return {
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CPATH": os.pathsep.join((str(header_root), str(header_root / "python3.12"))),
        "LIBRARY_PATH": str(environment_root / "lib"),
        "PATH": os.pathsep.join((str(environment_root / "bin"), *SYSTEM_PATH_SUFFIX)),
    }


def controlled_toolchain_contract(
    controlled: Mapping[str, str],
) -> dict[str, object]:
    """Return the deterministic purge/overlay identity recorded by callers."""

    normalized = _normalized_controlled(controlled)
    return {
        "schema_version": CONTROLLED_TOOLCHAIN_SCHEMA,
        "inheritance": "parent-minus-purged-toolchain-then-controlled-overlay",
        "purged_exact": list(PURGED_TOOLCHAIN_ENVIRONMENT),
        "purged_prefixes": list(PURGED_ENVIRONMENT_PREFIXES),
        "controlled": normalized,
    }


def controlled_toolchain_verification(
    environment: Mapping[str, str], *, controlled: Mapping[str, str]
) -> dict[str, object]:
    """Fail unless all controlled values match and all other purge names vanish."""

    normalized = _normalized_controlled(controlled)
    mismatched = tuple(
        name for name, value in normalized.items() if environment.get(name) != value
    )
    controlled_names = frozenset(normalized)
    forbidden_exact = tuple(
        name
        for name in PURGED_TOOLCHAIN_ENVIRONMENT
        if name not in controlled_names and name in environment
    )
    forbidden_prefixed = tuple(
        sorted(
            name
            for name in environment
            if any(name.startswith(prefix) for prefix in PURGED_ENVIRONMENT_PREFIXES)
        )
    )
    if mismatched or forbidden_exact or forbidden_prefixed:
        raise RuntimeError("controlled toolchain environment verification failed")
    return {
        "controlled_names": list(normalized),
        "purged_exact_absent": [
            name
            for name in PURGED_TOOLCHAIN_ENVIRONMENT
            if name not in controlled_names
        ],
        "purged_prefixes_absent": list(PURGED_ENVIRONMENT_PREFIXES),
        "verified": True,
    }


def build_controlled_toolchain_environment(
    *,
    controlled: Mapping[str, str],
    inherited: Mapping[str, str] | None = None,
    overlay: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Purge a parent environment and apply exact controlled/runtime values."""

    environment = dict(os.environ if inherited is None else inherited)
    for name in tuple(environment):
        if name in PURGED_TOOLCHAIN_ENVIRONMENT or any(
            name.startswith(prefix) for prefix in PURGED_ENVIRONMENT_PREFIXES
        ):
            environment.pop(name)
    environment.update(_normalized_controlled(controlled))
    if overlay is not None:
        environment.update(
            _normalized_strings(overlay, name="runtime overlay", allow_empty=True)
        )
    controlled_toolchain_verification(environment, controlled=controlled)
    return environment


def _normalized_controlled(controlled: Mapping[str, str]) -> dict[str, str]:
    normalized = _normalized_strings(controlled, name="controlled toolchain")
    required = {"CC", "CXX", "CPATH", "LIBRARY_PATH", "PATH"}
    if set(normalized) != required:
        raise ValueError("controlled toolchain fields differ")
    return normalized


def _normalized_strings(
    value: Mapping[str, str], *, name: str, allow_empty: bool = False
) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or (not allow_empty and not item)
        for key, item in value.items()
    ):
        raise ValueError(f"{name} must contain non-empty string pairs")
    return dict(value)


__all__ = [
    "CONTROLLED_TOOLCHAIN_SCHEMA",
    "PURGED_ENVIRONMENT_PREFIXES",
    "PURGED_TOOLCHAIN_ENVIRONMENT",
    "build_controlled_toolchain_environment",
    "controlled_toolchain_contract",
    "controlled_toolchain_verification",
    "python312_toolchain_environment",
]
