"""Exact, monotone debt accounting for oversized production modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


PRODUCTION_MODULE_LINE_LIMIT = 1000


class ModuleSizePolicyError(ValueError):
    """Raised when module-size policy metadata is not fail-closed."""


@dataclass(frozen=True, slots=True)
class ProductionModuleSizeException:
    path: str
    owner: str
    reason: str
    next_split_seam: str
    current_ceiling: int

    def as_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "owner": self.owner,
            "reason": self.reason,
            "next_split_seam": self.next_split_seam,
            "current_ceiling": self.current_ceiling,
        }


@dataclass(frozen=True, slots=True)
class ModuleSizeFinding:
    kind: str
    path: str
    message: str
    evidence: Mapping[str, object]


def load_production_module_size_exceptions(
    payload: object,
    *,
    source_root: str,
    line_limit: int = PRODUCTION_MODULE_LINE_LIMIT,
) -> tuple[ProductionModuleSizeException, ...]:
    """Parse an exact, path-sorted oversized-module debt inventory."""

    if not isinstance(payload, list):
        raise ModuleSizePolicyError(
            "production_module_size_exceptions must be a list"
        )
    exceptions: list[ProductionModuleSizeException] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict) or set(value) != {
            "path",
            "owner",
            "reason",
            "next_split_seam",
            "current_ceiling",
        }:
            raise ModuleSizePolicyError(
                f"production module size exception {index} has an unexpected field set"
            )
        path = _production_module_path(
            value["path"],
            source_root=source_root,
            field=f"production module size exception {index} path",
        )
        owner = _metadata_text(
            value["owner"],
            field=f"production module size exception {index} owner",
        )
        reason = _metadata_text(
            value["reason"],
            field=f"production module size exception {index} reason",
        )
        next_split_seam = _metadata_text(
            value["next_split_seam"],
            field=f"production module size exception {index} next_split_seam",
        )
        current_ceiling = value["current_ceiling"]
        if type(current_ceiling) is not int or current_ceiling <= line_limit:
            raise ModuleSizePolicyError(
                "production module size exception current_ceiling must be an "
                f"integer greater than {line_limit}"
            )
        exceptions.append(
            ProductionModuleSizeException(
                path=path,
                owner=owner,
                reason=reason,
                next_split_seam=next_split_seam,
                current_ceiling=current_ceiling,
            )
        )

    paths = tuple(item.path for item in exceptions)
    if len(set(paths)) != len(paths):
        raise ModuleSizePolicyError(
            "production module size exceptions contain duplicate paths"
        )
    if paths != tuple(sorted(paths)):
        raise ModuleSizePolicyError(
            "production module size exceptions must be sorted by path"
        )
    return tuple(exceptions)


def audit_production_module_sizes(
    observed_line_counts: Mapping[str, int],
    exceptions: Sequence[ProductionModuleSizeException],
    *,
    line_limit: int = PRODUCTION_MODULE_LINE_LIMIT,
) -> tuple[tuple[ModuleSizeFinding, ...], tuple[ModuleSizeFinding, ...]]:
    """Compare exact physical line counts with the registered debt snapshot."""

    registered = {item.path: item for item in exceptions}
    debts: list[ModuleSizeFinding] = []
    violations: list[ModuleSizeFinding] = []

    for path in sorted(observed_line_counts.keys() - registered.keys()):
        observed = observed_line_counts[path]
        if observed > line_limit:
            violations.append(
                ModuleSizeFinding(
                    kind="new_oversized_production_module",
                    path=path,
                    message="new production modules may not exceed the line limit",
                    evidence={
                        "line_limit": line_limit,
                        "observed_line_count": observed,
                    },
                )
            )

    for path, exception in sorted(registered.items()):
        observed = observed_line_counts.get(path)
        if observed is None:
            violations.append(
                ModuleSizeFinding(
                    kind="stale_module_size_exception",
                    path=path,
                    message="registered oversized-module debt no longer names a readable production module",
                    evidence={"registered_ceiling": exception.current_ceiling},
                )
            )
            continue
        if observed <= line_limit:
            violations.append(
                ModuleSizeFinding(
                    kind="stale_module_size_exception",
                    path=path,
                    message="module is within the line limit and its exception must be removed",
                    evidence={
                        "line_limit": line_limit,
                        "registered_ceiling": exception.current_ceiling,
                        "observed_line_count": observed,
                    },
                )
            )
            continue

        debts.append(
            ModuleSizeFinding(
                kind="oversized_production_module",
                path=path,
                message="oversized production module remains registered debt",
                evidence={
                    "owner": exception.owner,
                    "reason": exception.reason,
                    "next_split_seam": exception.next_split_seam,
                    "registered_ceiling": exception.current_ceiling,
                    "observed_line_count": observed,
                },
            )
        )
        if observed > exception.current_ceiling:
            violations.append(
                ModuleSizeFinding(
                    kind="module_size_ceiling_exceeded",
                    path=path,
                    message="production module grew beyond its registered ceiling",
                    evidence={
                        "registered_ceiling": exception.current_ceiling,
                        "observed_line_count": observed,
                    },
                )
            )
        elif observed < exception.current_ceiling:
            violations.append(
                ModuleSizeFinding(
                    kind="stale_module_size_ceiling",
                    path=path,
                    message="registered ceiling must be lowered to the exact current line count",
                    evidence={
                        "registered_ceiling": exception.current_ceiling,
                        "observed_line_count": observed,
                    },
                )
            )

    return tuple(debts), tuple(violations)


def compare_module_size_exception_ratchet(
    baseline: Sequence[ProductionModuleSizeException],
    candidate: Sequence[ProductionModuleSizeException],
) -> tuple[ModuleSizeFinding, ...]:
    """Reject newly allowlisted debt or a ceiling relaxed from a base policy."""

    baseline_by_path = {item.path: item for item in baseline}
    candidate_by_path = {item.path: item for item in candidate}
    violations: list[ModuleSizeFinding] = []
    for path in sorted(candidate_by_path.keys() - baseline_by_path.keys()):
        violations.append(
            ModuleSizeFinding(
                kind="new_module_size_exception",
                path=path,
                message="candidate policy may not add oversized-module exceptions",
                evidence={
                    "candidate_ceiling": candidate_by_path[path].current_ceiling
                },
            )
        )
    for path in sorted(candidate_by_path.keys() & baseline_by_path.keys()):
        baseline_ceiling = baseline_by_path[path].current_ceiling
        candidate_ceiling = candidate_by_path[path].current_ceiling
        if candidate_ceiling > baseline_ceiling:
            violations.append(
                ModuleSizeFinding(
                    kind="module_size_ceiling_relaxed",
                    path=path,
                    message="candidate policy may not raise an existing module-size ceiling",
                    evidence={
                        "baseline_ceiling": baseline_ceiling,
                        "candidate_ceiling": candidate_ceiling,
                    },
                )
            )
    return tuple(violations)


def _production_module_path(value: object, *, source_root: str, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModuleSizePolicyError(f"{field} must be a portable relative path")
    path = PurePosixPath(value)
    root = PurePosixPath(source_root)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or not path.is_relative_to(root)
        or path.suffix != ".py"
    ):
        raise ModuleSizePolicyError(
            f"{field} must be a canonical Python path under {source_root}"
        )
    return value


def _metadata_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ModuleSizePolicyError(
            f"{field} must be non-empty, trimmed text without control characters"
        )
    return value


__all__ = [
    "PRODUCTION_MODULE_LINE_LIMIT",
    "ModuleSizeFinding",
    "ModuleSizePolicyError",
    "ProductionModuleSizeException",
    "audit_production_module_sizes",
    "compare_module_size_exception_ratchet",
    "load_production_module_size_exceptions",
]
