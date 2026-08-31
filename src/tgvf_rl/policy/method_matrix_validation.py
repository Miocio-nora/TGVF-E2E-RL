"""Cross-arm consistency checks for config-driven policy method matrices.

The run-config loader validates each arm in isolation.  This module validates
the scientific comparison represented by a *set* of loaded configs: all
controls remain equal while the named method treatment is allowed to differ.
No resolution, horizon, batch size, rollout count, or seed is fixed here;
those values are selected by config and must simply agree across the matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

from .config import PolicyMethodProfile
from .run_config_schema import PolicyE2ESmokeRunConfig


DEFAULT_REQUIRED_METHOD_PROFILES = (
    PolicyMethodProfile.NO_TOOL,
    PolicyMethodProfile.CROP,
    PolicyMethodProfile.TGVF_SHORT,
    PolicyMethodProfile.TGVF_TARGET_GUIDE_V2,
    PolicyMethodProfile.ATOMIC,
)

# These are the only cross-arm differences that define, identify, or locate a
# treatment.  Every other loaded field participates in the shared fingerprint.
_ALLOWED_TREATMENT_SUBTREES = frozenset(
    {
        ("output",),
    }
)
_ALLOWED_TREATMENT_LEAVES = frozenset(
    {
        ("run_id",),
        ("source_path",),
        ("source_sha256",),
        ("canonical_json",),
        ("method", "profile"),
        ("method", "legacy_schema_alias"),
        ("policy", "method"),
        ("policy", "tool_profile"),
        ("policy", "enabled_tool_names"),
        ("policy", "max_tgvf_call_attempts"),
        ("policy", "sampling", "stop_strings"),
        ("protocol", "prompt_sha256"),
        ("protocol", "cap_error_sha256"),
        ("protocol", "tool_profile"),
        ("protocol", "tool_schema_sha256"),
        ("protocol", "enabled_tool_names"),
        ("protocol", "maximum_tool_calls"),
        ("protocol", "success_observation_protocol_id"),
        ("representation", "adapter_update_mode"),
        ("distributed", "weight_sync_mode"),
        ("reward", "judge_reason"),
        # A resume location is an output-derived path.  Resume mode, horizon,
        # checkpoints, and every other training control remain shared.
        ("training", "resume_from_path"),
    }
)
ALLOWED_TREATMENT_DIFFERENCE_PATHS = tuple(
    sorted(
        {
            *(".".join(path) + ".*" for path in _ALLOWED_TREATMENT_SUBTREES),
            *(".".join(path) for path in _ALLOWED_TREATMENT_LEAVES),
        }
    )
)


@dataclass(frozen=True, slots=True)
class MethodMatrixMismatch:
    """One explicit cross-arm mismatch at a canonical configuration path."""

    path: str
    reference_profile: PolicyMethodProfile | None
    reference_value: object
    actual_profile: PolicyMethodProfile | None
    actual_value: object


@dataclass(frozen=True, slots=True)
class MethodMatrixValidation:
    """Successful validation receipt for one complete method matrix."""

    matrix_id: str
    required_profiles: tuple[PolicyMethodProfile, ...]
    reference_profile: PolicyMethodProfile
    shared_fingerprint_sha256: str
    shared_canonical_json: str
    shared_leaf_paths: tuple[str, ...]
    allowed_treatment_difference_paths: tuple[str, ...]


class MethodMatrixValidationError(ValueError):
    """Raised when matrix membership or a shared config control differs."""

    def __init__(self, mismatches: Iterable[MethodMatrixMismatch]) -> None:
        normalized = tuple(mismatches)
        if not normalized:
            raise ValueError("matrix validation error requires a mismatch")
        self.mismatches = normalized
        self.mismatch_paths = tuple(dict.fromkeys(item.path for item in normalized))
        super().__init__(
            "policy method matrix differs at: " + ", ".join(self.mismatch_paths)
        )


def validate_policy_method_matrix(
    configs: Iterable[PolicyE2ESmokeRunConfig],
    *,
    required_profiles: Iterable[PolicyMethodProfile] = (
        DEFAULT_REQUIRED_METHOD_PROFILES
    ),
) -> MethodMatrixValidation:
    """Validate one config-selected comparison and fingerprint shared controls.

    ``required_profiles`` makes partial diagnostic matrices possible without
    weakening the default five-arm comparison.  It controls membership only;
    no scientific value is supplied or defaulted by this validator.
    """

    arms = tuple(configs)
    required = _normalize_required_profiles(required_profiles)
    if not arms:
        raise MethodMatrixValidationError(
            (
                MethodMatrixMismatch(
                    path="$profiles",
                    reference_profile=None,
                    reference_value=tuple(profile.value for profile in required),
                    actual_profile=None,
                    actual_value=(),
                ),
            )
        )
    for index, config in enumerate(arms):
        if not isinstance(config, PolicyE2ESmokeRunConfig):
            raise TypeError(
                f"configs[{index}] must be PolicyE2ESmokeRunConfig, "
                f"got {type(config).__name__}"
            )

    membership_mismatches, by_profile = _validate_membership(arms, required)
    if membership_mismatches:
        raise MethodMatrixValidationError(membership_mismatches)

    ordered_arms = tuple(by_profile[profile] for profile in required)
    reference = ordered_arms[0]
    reference_profile = required[0]
    reference_projection = _shared_projection(reference)
    mismatches: list[MethodMatrixMismatch] = []
    for profile, arm in zip(required[1:], ordered_arms[1:], strict=True):
        actual_projection = _shared_projection(arm)
        for path, expected, actual in _value_mismatches(
            reference_projection,
            actual_projection,
        ):
            mismatches.append(
                MethodMatrixMismatch(
                    path=path,
                    reference_profile=reference_profile,
                    reference_value=expected,
                    actual_profile=profile,
                    actual_value=actual,
                )
            )
    if mismatches:
        raise MethodMatrixValidationError(mismatches)

    fingerprint_record = {
        "schema_version": "tgvf-policy-method-matrix-shared-controls-v1",
        "matrix_id": reference.method.matrix_id,
        "required_profiles": [profile.value for profile in required],
        "shared_controls": reference_projection,
    }
    shared_canonical_json = _canonical_json(fingerprint_record)
    return MethodMatrixValidation(
        matrix_id=reference.method.matrix_id,
        required_profiles=required,
        reference_profile=reference_profile,
        shared_fingerprint_sha256=hashlib.sha256(
            shared_canonical_json.encode("utf-8")
        ).hexdigest(),
        shared_canonical_json=shared_canonical_json,
        shared_leaf_paths=tuple(_leaf_paths(reference_projection)),
        allowed_treatment_difference_paths=(ALLOWED_TREATMENT_DIFFERENCE_PATHS),
    )


def _normalize_required_profiles(
    profiles: Iterable[PolicyMethodProfile],
) -> tuple[PolicyMethodProfile, ...]:
    normalized = tuple(profiles)
    if not normalized:
        raise ValueError("required_profiles must not be empty")
    if any(not isinstance(profile, PolicyMethodProfile) for profile in normalized):
        raise TypeError("required_profiles must contain PolicyMethodProfile values")
    if len(set(normalized)) != len(normalized):
        raise ValueError("required_profiles must be unique")
    return normalized


def _validate_membership(
    arms: tuple[PolicyE2ESmokeRunConfig, ...],
    required: tuple[PolicyMethodProfile, ...],
) -> tuple[
    tuple[MethodMatrixMismatch, ...],
    dict[PolicyMethodProfile, PolicyE2ESmokeRunConfig],
]:
    mismatches: list[MethodMatrixMismatch] = []
    by_profile: dict[PolicyMethodProfile, PolicyE2ESmokeRunConfig] = {}
    matrix_id: str | None = None
    for index, arm in enumerate(arms):
        binding = arm.method
        if binding is None:
            mismatches.append(
                MethodMatrixMismatch(
                    path=f"arms[{index}].method",
                    reference_profile=None,
                    reference_value="method binding",
                    actual_profile=None,
                    actual_value=None,
                )
            )
            continue
        profile = binding.profile
        if matrix_id is None:
            matrix_id = binding.matrix_id
        elif binding.matrix_id != matrix_id:
            mismatches.append(
                MethodMatrixMismatch(
                    path=f"arms[{profile.value}].method.matrix_id",
                    reference_profile=None,
                    reference_value=matrix_id,
                    actual_profile=profile,
                    actual_value=binding.matrix_id,
                )
            )
        if profile in by_profile:
            mismatches.append(
                MethodMatrixMismatch(
                    path=f"$profiles.{profile.value}",
                    reference_profile=profile,
                    reference_value="exactly one arm",
                    actual_profile=profile,
                    actual_value="duplicate arm",
                )
            )
        else:
            by_profile[profile] = arm

    required_set = set(required)
    for profile in required:
        if profile not in by_profile:
            mismatches.append(
                MethodMatrixMismatch(
                    path=f"$profiles.{profile.value}",
                    reference_profile=profile,
                    reference_value="one required arm",
                    actual_profile=None,
                    actual_value="missing",
                )
            )
    for profile in sorted(set(by_profile) - required_set, key=lambda item: item.value):
        mismatches.append(
            MethodMatrixMismatch(
                path=f"$profiles.{profile.value}",
                reference_profile=None,
                reference_value="not selected",
                actual_profile=profile,
                actual_value="unexpected arm",
            )
        )
    return tuple(mismatches), by_profile


def _shared_projection(config: PolicyE2ESmokeRunConfig) -> dict[str, object]:
    normalized = _normalize_value(config)
    if not isinstance(normalized, dict):  # pragma: no cover - dataclass invariant
        raise TypeError("normalized policy config must be an object")
    projected = _project_shared(normalized)
    if not isinstance(projected, dict):  # pragma: no cover - root is never excluded
        raise TypeError("shared policy config projection must be an object")
    return projected


_EXCLUDED = object()


def _project_shared(value: object, path: tuple[str, ...] = ()) -> object:
    if path in _ALLOWED_TREATMENT_SUBTREES or path in _ALLOWED_TREATMENT_LEAVES:
        return _EXCLUDED
    if isinstance(value, dict):
        projected: dict[str, object] = {}
        for key, child in value.items():
            child_projection = _project_shared(child, (*path, key))
            if child_projection is not _EXCLUDED:
                projected[key] = child_projection
        return projected
    if isinstance(value, list):
        return [
            child_projection
            for index, child in enumerate(value)
            if (child_projection := _project_shared(child, (*path, str(index))))
            is not _EXCLUDED
        ]
    return value


def _normalize_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("method matrix values must be finite")
        return value
    if isinstance(value, Enum):
        return _normalize_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("method matrix mapping keys must be strings")
            normalized[key] = _normalize_value(child)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_normalize_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        children = [_normalize_value(child) for child in value]
        return sorted(children, key=_canonical_json)
    raise TypeError(
        "unsupported method matrix config value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


_MISSING = object()


def _value_mismatches(
    expected: object,
    actual: object,
    path: tuple[str, ...] = (),
) -> list[tuple[str, object, object]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[tuple[str, object, object]] = []
        for key in sorted(set(expected) | set(actual)):
            expected_child = expected.get(key, _MISSING)
            actual_child = actual.get(key, _MISSING)
            child_path = (*path, key)
            if expected_child is _MISSING or actual_child is _MISSING:
                mismatches.append(
                    (
                        ".".join(child_path),
                        "<missing>" if expected_child is _MISSING else expected_child,
                        "<missing>" if actual_child is _MISSING else actual_child,
                    )
                )
            else:
                mismatches.extend(
                    _value_mismatches(expected_child, actual_child, child_path)
                )
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        mismatches = []
        for index in range(max(len(expected), len(actual))):
            expected_child = expected[index] if index < len(expected) else _MISSING
            actual_child = actual[index] if index < len(actual) else _MISSING
            child_path = (*path, str(index))
            if expected_child is _MISSING or actual_child is _MISSING:
                mismatches.append(
                    (
                        ".".join(child_path),
                        "<missing>" if expected_child is _MISSING else expected_child,
                        "<missing>" if actual_child is _MISSING else actual_child,
                    )
                )
            else:
                mismatches.extend(
                    _value_mismatches(expected_child, actual_child, child_path)
                )
        return mismatches
    if expected != actual:
        return [(".".join(path), expected, actual)]
    return []


def _leaf_paths(value: object, path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key in sorted(value):
            paths.extend(_leaf_paths(value[key], (*path, key)))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_leaf_paths(child, (*path, str(index))))
        return paths
    return [".".join(path)]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "ALLOWED_TREATMENT_DIFFERENCE_PATHS",
    "DEFAULT_REQUIRED_METHOD_PROFILES",
    "MethodMatrixMismatch",
    "MethodMatrixValidation",
    "MethodMatrixValidationError",
    "validate_policy_method_matrix",
]
