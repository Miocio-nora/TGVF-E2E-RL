"""Shared constants and primitive validation for the result registry."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping

from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations as _freeze_public_class_annotations,
    rebind_public_class as _rebind_public_class,
    rebind_public_function as _rebind_public_function,
)
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    read_regular_file_beneath_absolute_directory_nofollow,
)

RESULT_REGISTRY_SCHEMA = "tgvf.policy-result-registry.v2"
COREDEV_COMPONENTS = (
    "vstar",
    "hr_average_all",
    "blink_single_180",
    "ocr_mean",
    "mmmu_single_269",
    "mathvista",
    "mathverse_five_version_macro",
)
INTERVENTION_AXES = (
    "method",
    "optimizer_step",
    "weights",
    "training_contract_identity",
    "training_image_max_pixels",
    "declared_evaluation_image_max_pixels",
    "effective_evaluation_image_max_pixels",
    "runtime_identity",
    "parser_identity",
    "action_boundary_identity",
    "observation_identity",
    "prompt_identity",
    "generation_identity",
)
INVARIANT_FIELDS = (
    "task_manifest_sha256",
    "rng_identity",
    "scorer_identity",
    "inference_sample_count",
    "scored_sample_count",
    "slice_count",
)
_SHA256_LENGTH = 64
_SCORE_MATCH_ABS_TOLERANCE = 5e-5
_PREREGISTRATION_SCHEMA = "tgvf.comparison-preregistration.v1"
_GOLDEN_PROMOTION_BLOCKED_REASON = (
    "result registry v2 cannot promote golden results: score artifacts are not "
    "mechanically bound to the evaluation identity, trajectory-set identity, "
    "weights, and comparison contract; use a future provenance-receipt schema"
)


class RegistryValidationError(ValueError):
    """Raised when registry evidence is incomplete or internally inconsistent."""


class IncomparableResultsError(RegistryValidationError):
    """Raised when a requested delta crosses a comparison contract."""


def _object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing:
        raise RegistryValidationError(
            f"{context} is missing required keys: {sorted(missing)}"
        )
    if unknown:
        raise RegistryValidationError(
            f"{context} contains unknown keys: {sorted(unknown)}"
        )


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{context} must be non-empty text")
    return value


def _sha256(value: object, *, context: str) -> str:
    text = _text(value, context=context)
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RegistryValidationError(f"{context} must be lowercase SHA-256")
    return text


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryValidationError(f"{context} must be a positive integer")
    return value


def _optional_positive_int(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, context=context)


def _finite_score(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryValidationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise RegistryValidationError(f"{context} must be finite and in [0, 100]")
    return number


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_repository_regular_file(
    repository_root: Path,
    relative_path: str,
    *,
    context: str,
) -> bytes:
    """Read one bound file without following a symlink in its path chain."""

    root = repository_root.absolute()
    candidate = root / relative_path
    try:
        return read_regular_file_beneath_absolute_directory_nofollow(
            root,
            relative_path,
        ).payload
    except (OSError, SecureFileReadError, TypeError) as error:
        raise RegistryValidationError(
            f"{context} escapes or cannot be opened without following symlinks "
            f"under repository root: {candidate}"
        ) from error


_PUBLIC_MODULE = "tgvf_rl.evaluation.result_registry"


def _publish_result_registry_schema(
    *,
    annotation_frozen_types: tuple[type[object], ...],
    public_types: tuple[type[object], ...],
    public_functions: tuple[FunctionType, ...],
    implementation_globals: dict[str, object],
) -> None:
    """Publish schema-owned objects without importing the historical facade."""

    for contract_type in annotation_frozen_types:
        _freeze_public_class_annotations(
            contract_type,
            implementation_globals=implementation_globals,
        )
    for contract_type in public_types:
        _rebind_public_class(
            contract_type,
            implementation_module="tgvf_rl.evaluation.result_registry_schema",
            public_module=_PUBLIC_MODULE,
        )
    for function in public_functions:
        _rebind_public_function(
            function,
            implementation_module="tgvf_rl.evaluation.result_registry_schema",
            public_module=_PUBLIC_MODULE,
        )


_RESULT_REGISTRY_SUPPORT_TYPES = (
    RegistryValidationError,
    IncomparableResultsError,
)
_RESULT_REGISTRY_SUPPORT_FUNCTIONS = (
    _object,
    _exact_keys,
    _text,
    _sha256,
    _positive_int,
    _optional_positive_int,
    _finite_score,
    _canonical_sha256,
    _read_repository_regular_file,
)
for _contract_type in _RESULT_REGISTRY_SUPPORT_TYPES:
    _rebind_public_class(
        _contract_type,
        implementation_module=__name__,
        public_module=_PUBLIC_MODULE,
    )
for _function in _RESULT_REGISTRY_SUPPORT_FUNCTIONS:
    _rebind_public_function(
        _function,
        implementation_module=__name__,
        public_module=_PUBLIC_MODULE,
    )
del _contract_type, _function
