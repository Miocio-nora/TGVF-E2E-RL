"""Fail-closed child-process environment profiles and secret capabilities.

The launch environment is assembled from an empty mapping.  The caller may
contribute only names owned by one fixed profile, and a later process boundary
may add only that profile's explicit overlay names.  Host environment values
are never copied: their names are retained solely as an authorization audit.

Secrets deliberately use a separate process-local capability.  In particular,
``OPENROUTER_API_KEY`` can never become part of an outer launch binding that
Ray or torchrun could fan out to unrelated descendants.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path, PurePath
import re


CHILD_ENVIRONMENT_SCHEMA = "tgvf-child-environment-v1"
REPRESENTATION_TORCHRUN_PROFILE = "representation-torchrun-v1"
POLICY_VERL_DRIVER_PROFILE = "policy-verl-driver-v1"
OPENROUTER_SECRET_ENVIRONMENT_NAME = "OPENROUTER_API_KEY"
OPENROUTER_SECRET_REQUIREMENT_SCHEMA = "tgvf-openrouter-secret-requirement-v1"
RUNTIME_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ENVIRONMENT_VALUE_BYTES = 1024 * 1024

_COMMON_BASELINE = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.defpath,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(RUNTIME_PACKAGE_ROOT),
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "TZ": "UTC",
}

_RUNTIME_PATH_NAMES = frozenset(
    {
        "HF_HOME",
        "HOME",
        "TMPDIR",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    }
)

CLI_WORKER_LATE_ENVIRONMENT_NAMES = (
    "TGVF_CLI_CONSUMPTION_RECEIPT_PATH",
    "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256",
    "TGVF_CLI_EXECUTION_IDENTITY_JSON",
    "TGVF_CLI_GATE_DIRECTORY",
    "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH",
    "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA",
)
POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES = (
    "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH",
    "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256",
)
TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES = (
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "WORLD_SIZE",
)

_CLI_WORKER_OVERLAY_NAMES = frozenset(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
_POLICY_COMPILE_RECEIPT_OVERLAY_NAMES = frozenset(
    POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES
)

_REPRESENTATION_OWNED_NAMES = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "TOKENIZERS_PARALLELISM",
        *_RUNTIME_PATH_NAMES,
    }
)
_REPRESENTATION_LATE_OVERLAY_NAMES = frozenset(
    {
        *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
        *_CLI_WORKER_OVERLAY_NAMES,
    }
)

_POLICY_OWNED_NAMES = frozenset(
    {
        "CC",
        "CPATH",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "CXX",
        "PYTHONHASHSEED",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256",
        "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_PATH",
        "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256",
        "TGVF_POLICY_HORIZON_EXTENSION_PATH",
        "TGVF_POLICY_HORIZON_EXTENSION_SHA256",
        "TGVF_POLICY_RUN_CONFIG_PATH",
        "TGVF_POLICY_RUN_ID",
        "TGVF_POLICY_RUN_IDENTITY",
        "TGVF_POLICY_RUN_IDENTITY_SHA256",
        "TGVF_POLICY_SERVER_TIMEOUT_SECONDS",
        "TGVF_POLICY_STATE_DIR",
        "TOKENIZERS_PARALLELISM",
        "VERL_FULL_DETERMINISM",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_BATCH_INVARIANT",
        "VLLM_PLUGINS",
        *_RUNTIME_PATH_NAMES,
    }
)
_POLICY_LATE_OVERLAY_NAMES = frozenset(
    {
        *_POLICY_COMPILE_RECEIPT_OVERLAY_NAMES,
        *_CLI_WORKER_OVERLAY_NAMES,
    }
)


@dataclass(frozen=True, slots=True)
class _ProfileContract:
    name: str
    fixed_entries: tuple[tuple[str, str], ...]
    owned_names: frozenset[str]
    late_overlay_names: frozenset[str]


_PROFILE_CONTRACTS = {
    REPRESENTATION_TORCHRUN_PROFILE: _ProfileContract(
        name=REPRESENTATION_TORCHRUN_PROFILE,
        fixed_entries=(
            ("OMP_NUM_THREADS", "1"),
            ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        ),
        owned_names=_REPRESENTATION_OWNED_NAMES,
        late_overlay_names=_REPRESENTATION_LATE_OVERLAY_NAMES,
    ),
    POLICY_VERL_DRIVER_PROFILE: _ProfileContract(
        name=POLICY_VERL_DRIVER_PROFILE,
        fixed_entries=(
            ("RAY_USAGE_STATS_ENABLED", "0"),
            ("VLLM_NO_USAGE_STATS", "1"),
        ),
        owned_names=_POLICY_OWNED_NAMES,
        late_overlay_names=_POLICY_LATE_OVERLAY_NAMES,
    ),
}
SUPPORTED_CHILD_ENVIRONMENT_PROFILES = tuple(sorted(_PROFILE_CONTRACTS))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_environment_name(name: object, *, owner: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{owner} environment names must be strings")
    if not _ENVIRONMENT_NAME_RE.fullmatch(name):
        raise ValueError(f"{owner} environment name is invalid")
    return name


def _validate_environment_value(value: object, *, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner} environment values must be strings")
    if "\x00" in value:
        raise ValueError(f"{owner} environment value contains NUL")
    if len(value.encode("utf-8")) > _MAX_ENVIRONMENT_VALUE_BYTES:
        raise ValueError(f"{owner} environment value is too large")
    return value


def _validate_environment_mapping(
    values: Mapping[str, str], *, owner: str
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{owner} environment must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = _validate_environment_name(raw_name, owner=owner)
        value = _validate_environment_value(raw_value, owner=owner)
        result[name] = value
    return result


def _validate_host_environment_names(values: Mapping[str, str]) -> tuple[str, ...]:
    """Validate only host names without retrieving any host value."""

    if not isinstance(values, Mapping):
        raise TypeError("host environment must be a mapping")
    names = {_validate_environment_name(raw_name, owner="host") for raw_name in values}
    return tuple(sorted(names))


def _contract(profile: object) -> _ProfileContract:
    if not isinstance(profile, str):
        raise TypeError("child environment profile must be a string")
    try:
        return _PROFILE_CONTRACTS[profile]
    except KeyError as error:
        raise ValueError(
            f"unsupported child environment profile: {profile!r}"
        ) from error


def _profile_record(contract: _ProfileContract) -> dict[str, object]:
    return {
        "schema_version": CHILD_ENVIRONMENT_SCHEMA,
        "profile": contract.name,
        "common_baseline": dict(sorted(_COMMON_BASELINE.items())),
        "profile_fixed_entries": dict(contract.fixed_entries),
        "owned_names": sorted(contract.owned_names),
        "late_overlay_names": sorted(contract.late_overlay_names),
    }


def _names_sha256(names: tuple[str, ...]) -> str:
    return _canonical_sha256(list(names))


def _is_rejected_host_name(name: str, contract: _ProfileContract) -> bool:
    upper = name.upper()
    if (
        name in _COMMON_BASELINE
        or name in dict(contract.fixed_entries)
        or name in contract.owned_names
        or name in contract.late_overlay_names
    ):
        return True
    if upper in {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GLOBIGNORE",
        "LD_PRELOAD",
        "LIBRARY_PATH",
        "RAY_ADDRESS",
        "SHELLOPTS",
    }:
        return True
    if upper in {
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "WORLD_SIZE",
    }:
        return True
    if upper.endswith(("_PROXY", "_API_KEY", "_AUTH_TOKEN", "_PASSWORD")):
        return True
    return upper.startswith(
        (
            "AWS_",
            "AZURE_",
            "CUDA_",
            "DYLD_",
            "GIT_",
            "GOOGLE_",
            "HF_",
            "LD_",
            "NCCL_",
            "OPENAI_",
            "OPENROUTER_",
            "PET_",
            "PYTHON",
            "RAY_",
            "TGVF_",
            "TORCHELASTIC_",
            "TORCH_NCCL_",
            "VERL_",
            "VLLM_",
        )
    )


def _assert_path_value(name: str, value: str) -> None:
    if name not in _RUNTIME_PATH_NAMES and not name.endswith("_PATH"):
        return
    path = PurePath(value)
    if not value or not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"owned environment path {name} must be lexical absolute")


def _assert_owned_value(name: str, value: str) -> None:
    _assert_path_value(name, value)
    if name == "CUBLAS_WORKSPACE_CONFIG" and value != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the fixed contract")
    if name == "TOKENIZERS_PARALLELISM" and value != "false":
        raise ValueError("TOKENIZERS_PARALLELISM must be false")
    if name == "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" and value != "1":
        raise ValueError("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES must be one")
    if name in {"VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT"} and value != "0":
        raise ValueError(f"{name} must be zero")
    if name == "PYTHONHASHSEED" and (
        not value.isdecimal() or not 0 <= int(value) <= 4_294_967_295
    ):
        raise ValueError("PYTHONHASHSEED must be an unsigned 32-bit decimal integer")
    if name.endswith("_SHA256") and not _SHA256_RE.fullmatch(value):
        raise ValueError(f"owned environment digest {name} is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class ChildEnvironmentBinding:
    """Immutable exact environment plus a host-name-only audit trail."""

    profile: str
    entries: tuple[tuple[str, str], ...]
    owned_names: tuple[str, ...]
    late_overlay_names: tuple[str, ...]
    ignored_host_names: tuple[str, ...]
    rejected_host_names: tuple[str, ...]

    def __post_init__(self) -> None:
        contract = _contract(self.profile)
        for field_name in (
            "owned_names",
            "late_overlay_names",
            "ignored_host_names",
            "rejected_host_names",
        ):
            names = getattr(self, field_name)
            if not isinstance(names, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if names != tuple(sorted(set(names))):
                raise ValueError(f"{field_name} must be sorted and unique")
            for name in names:
                _validate_environment_name(name, owner=field_name)
        if set(self.ignored_host_names) & set(self.rejected_host_names):
            raise ValueError("ignored and rejected host-name audits overlap")
        if not isinstance(self.entries, tuple):
            raise TypeError("child environment entries must be a tuple")
        normalized_entries = _validate_environment_mapping(
            dict(self.entries), owner="bound child"
        )
        if len(normalized_entries) != len(self.entries):
            raise ValueError("child environment entries contain duplicate names")
        if self.entries != tuple(sorted(normalized_entries.items())):
            raise ValueError("child environment entries must be sorted")
        if not set(self.owned_names).issubset(contract.owned_names):
            raise ValueError("bound child environment has unknown owned names")
        if not set(self.late_overlay_names).issubset(contract.late_overlay_names):
            raise ValueError("bound child environment has unknown late-overlay names")
        if set(self.owned_names) & set(self.late_overlay_names):
            raise ValueError("owned and late-overlay names overlap")
        expected_entries = {
            **_COMMON_BASELINE,
            **dict(contract.fixed_entries),
        }
        variable_entry_names = set(normalized_entries).difference(expected_entries)
        if variable_entry_names != set(self.owned_names).union(self.late_overlay_names):
            raise ValueError("bound child environment field set differs from profile")
        for name in self.owned_names:
            expected_entries[name] = normalized_entries[name]
            _assert_owned_value(name, normalized_entries[name])
        for name in self.late_overlay_names:
            expected_entries[name] = normalized_entries[name]
        if normalized_entries != expected_entries:
            raise ValueError("bound child environment field set differs from profile")
        if OPENROUTER_SECRET_ENVIRONMENT_NAME in normalized_entries:
            raise ValueError("OpenRouter secret cannot enter an outer environment")

    def __repr__(self) -> str:
        return (
            f"ChildEnvironmentBinding(profile={self.profile!r}, "
            f"entry_count={len(self.entries)}, "
            f"environment_sha256={self.environment_sha256!r})"
        )

    @property
    def profile_sha256(self) -> str:
        return _canonical_sha256(_profile_record(_contract(self.profile)))

    @property
    def environment_sha256(self) -> str:
        return _canonical_sha256(dict(self.entries))

    def as_environment(self) -> dict[str, str]:
        """Return a mutable copy suitable for an exact subprocess API."""

        return dict(self.entries)

    def with_late_overlay(self, overlay: Mapping[str, str]) -> ChildEnvironmentBinding:
        """Add only unused names delegated to this profile's next boundary."""

        values = _validate_environment_mapping(overlay, owner="late overlay")
        contract = _contract(self.profile)
        existing = set(dict(self.entries))
        conflicts = existing.intersection(values)
        if conflicts:
            raise ValueError(
                "late overlay cannot overwrite existing names: "
                + ", ".join(sorted(conflicts))
            )
        unknown = set(values).difference(contract.late_overlay_names)
        if unknown:
            raise ValueError(
                "late overlay contains names outside the profile: "
                + ", ".join(sorted(unknown))
            )
        entries = dict(self.entries)
        entries.update(values)
        return ChildEnvironmentBinding(
            profile=self.profile,
            entries=tuple(sorted(entries.items())),
            owned_names=self.owned_names,
            late_overlay_names=tuple(
                sorted(set(self.late_overlay_names).union(values))
            ),
            ignored_host_names=self.ignored_host_names,
            rejected_host_names=self.rejected_host_names,
        )

    def authorization_parameters(self) -> dict[str, str]:
        """Return canonical identities without any host or secret values."""

        return {
            "child_environment_schema": CHILD_ENVIRONMENT_SCHEMA,
            "child_environment_profile": self.profile,
            "child_environment_profile_sha256": self.profile_sha256,
            "child_environment_sha256": self.environment_sha256,
            "child_environment_entry_count": str(len(self.entries)),
            "child_environment_owned_name_count": str(len(self.owned_names)),
            "child_environment_owned_names_sha256": _names_sha256(self.owned_names),
            "child_environment_late_overlay_name_count": str(
                len(self.late_overlay_names)
            ),
            "child_environment_late_overlay_names_sha256": _names_sha256(
                self.late_overlay_names
            ),
            "child_environment_ignored_host_name_count": str(
                len(self.ignored_host_names)
            ),
            "child_environment_ignored_host_names_sha256": _names_sha256(
                self.ignored_host_names
            ),
            "child_environment_rejected_host_name_count": str(
                len(self.rejected_host_names)
            ),
            "child_environment_rejected_host_names_sha256": _names_sha256(
                self.rejected_host_names
            ),
        }


def build_child_environment(
    profile: str,
    *,
    owned_environment: Mapping[str, str] | None = None,
    host_environment: Mapping[str, str] | None = None,
) -> ChildEnvironmentBinding:
    """Build one profile from empty state while auditing host names only."""

    contract = _contract(profile)
    owned = _validate_environment_mapping(
        {} if owned_environment is None else owned_environment,
        owner="owned",
    )
    unknown = set(owned).difference(contract.owned_names)
    if unknown:
        raise ValueError(
            "owned environment contains names outside the profile: "
            + ", ".join(sorted(unknown))
        )
    reserved = set(_COMMON_BASELINE).union(dict(contract.fixed_entries))
    conflicts = reserved.intersection(owned)
    if conflicts:
        raise ValueError(
            "owned environment cannot overwrite reserved names: "
            + ", ".join(sorted(conflicts))
        )
    for name, value in owned.items():
        _assert_owned_value(name, value)

    source = os.environ if host_environment is None else host_environment
    host_names = _validate_host_environment_names(source)
    rejected = tuple(
        name for name in host_names if _is_rejected_host_name(name, contract)
    )
    ignored = tuple(name for name in host_names if name not in rejected)
    entries = {
        **_COMMON_BASELINE,
        **dict(contract.fixed_entries),
        **owned,
    }
    return ChildEnvironmentBinding(
        profile=contract.name,
        entries=tuple(sorted(entries.items())),
        owned_names=tuple(sorted(owned)),
        late_overlay_names=(),
        ignored_host_names=ignored,
        rejected_host_names=rejected,
    )


def profile_owned_environment_names(profile: str) -> tuple[str, ...]:
    """Return the exact caller-owned name inventory for one profile."""

    return tuple(sorted(_contract(profile).owned_names))


def profile_late_overlay_environment_names(profile: str) -> tuple[str, ...]:
    """Return the exact delegated late-overlay inventory for one profile."""

    return tuple(sorted(_contract(profile).late_overlay_names))


def _verify_materialized_environment(
    profile: str,
    environment: Mapping[str, str],
    authorization_parameters: Mapping[str, str],
    *,
    exact_late_names: frozenset[str],
) -> None:
    """Verify one child view against its pre-consumption base identity."""

    values = _validate_environment_mapping(environment, owner="materialized child")
    parameters = _validate_environment_mapping(
        authorization_parameters,
        owner="child authorization",
    )
    contract = _contract(profile)
    expected_profile_sha256 = _canonical_sha256(_profile_record(contract))
    expected_parameters = {
        "child_environment_schema": CHILD_ENVIRONMENT_SCHEMA,
        "child_environment_profile": profile,
        "child_environment_profile_sha256": expected_profile_sha256,
    }
    for name, expected in expected_parameters.items():
        if parameters.get(name) != expected:
            raise ValueError(f"child authorization parameter differs: {name}")
    missing_late = exact_late_names.difference(values)
    if missing_late:
        raise ValueError(
            "materialized child environment lacks required late names: "
            + ", ".join(sorted(missing_late))
        )
    base = {
        name: value for name, value in values.items() if name not in exact_late_names
    }
    if parameters.get("child_environment_entry_count") != str(len(base)):
        raise ValueError("materialized child environment entry count differs")
    if parameters.get("child_environment_sha256") != _canonical_sha256(base):
        raise ValueError("materialized child environment identity differs")


def verify_policy_driver_child_environment(
    environment: Mapping[str, str],
    authorization_parameters: Mapping[str, str],
) -> None:
    """Verify the exact Policy driver environment before importing Ray/veRL."""

    _verify_materialized_environment(
        POLICY_VERL_DRIVER_PROFILE,
        environment,
        authorization_parameters,
        exact_late_names=frozenset(
            {
                *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
                *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
            }
        ),
    )


def verify_representation_torchrun_child_environment(
    environment: Mapping[str, str],
    authorization_parameters: Mapping[str, str],
) -> None:
    """Verify the exact base plus pinned torchrun worker field inventory."""

    _verify_materialized_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        environment,
        authorization_parameters,
        exact_late_names=frozenset(
            {
                *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
                *TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
            }
        ),
    )


def _scrub_required_environment_names(
    environment: MutableMapping[str, str],
    names: tuple[str, ...],
    *,
    owner: str,
) -> None:
    if not isinstance(environment, MutableMapping):
        raise TypeError(f"{owner} environment must be mutable")
    missing = set(names).difference(environment)
    if missing:
        raise RuntimeError(
            f"{owner} environment lacks fields to scrub: " + ", ".join(sorted(missing))
        )
    for name in names:
        del environment[name]
    if set(names).intersection(environment):  # pragma: no cover - mapping invariant
        raise RuntimeError(f"{owner} environment retained scrubbed fields")


def scrub_policy_driver_authorization_environment(
    environment: MutableMapping[str, str],
) -> None:
    """Remove validated one-use CLI and compile receipts before Ray starts."""

    _scrub_required_environment_names(
        environment,
        (
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
        ),
        owner="Policy driver authorization",
    )


def scrub_representation_worker_authorization_environment(
    environment: MutableMapping[str, str],
) -> None:
    """Remove validated one-use CLI proof before rank-local descendants start."""

    _scrub_required_environment_names(
        environment,
        CLI_WORKER_LATE_ENVIRONMENT_NAMES,
        owner="representation worker authorization",
    )


@dataclass(frozen=True, slots=True)
class OpenRouterSecretRequirement:
    """Serializable role/name requirement containing no secret material."""

    role: str
    environment_names: tuple[str, ...] = field(
        default=(OPENROUTER_SECRET_ENVIRONMENT_NAME,), init=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.role, str)
            or not self.role.strip()
            or "\x00" in self.role
            or "\r" in self.role
            or "\n" in self.role
        ):
            raise ValueError("OpenRouter secret role is invalid")

    def authorization_parameters(self) -> dict[str, str]:
        return {
            "secret_requirement_schema": OPENROUTER_SECRET_REQUIREMENT_SCHEMA,
            "secret_requirement_role": self.role,
            "secret_requirement_name_count": str(len(self.environment_names)),
            "secret_requirement_names": json.dumps(
                list(self.environment_names), separators=(",", ":")
            ),
            "secret_requirement_names_sha256": _names_sha256(self.environment_names),
        }


class OpenRouterSecretBinding:
    """Single-use process-local input capability for a future secret broker."""

    __slots__ = ("_requirement", "_secret_bytes", "_spent")

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("OpenRouterSecretBinding is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        requirement: OpenRouterSecretRequirement,
        secret_value: str,
    ) -> None:
        if not isinstance(requirement, OpenRouterSecretRequirement):
            raise TypeError("requirement must be OpenRouterSecretRequirement")
        value = _validate_environment_value(secret_value, owner="OpenRouter secret")
        if not value or "\r" in value or "\n" in value:
            raise ValueError("OpenRouter secret value is invalid")
        self._requirement = requirement
        self._secret_bytes = bytearray(value, "utf-8")
        self._spent = False

    @property
    def requirement(self) -> OpenRouterSecretRequirement:
        return self._requirement

    @property
    def spent(self) -> bool:
        """Whether the secret has been consumed or explicitly discarded."""

        return self._spent

    def __repr__(self) -> str:
        return (
            f"OpenRouterSecretBinding(role={self._requirement.role!r}, "
            f"environment_names={self._requirement.environment_names!r}, "
            f"spent={self._spent!r}, secret=<redacted>)"
        )

    def __reduce__(self) -> object:
        raise TypeError("OpenRouterSecretBinding is process-local and not serializable")

    def __copy__(self) -> object:
        raise TypeError("OpenRouterSecretBinding is process-local and not copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("OpenRouterSecretBinding is process-local and not copyable")

    def authorization_parameters(self) -> dict[str, str]:
        """Record only the secret role and exact required name inventory."""

        return self._requirement.authorization_parameters()

    def consume_into_broker_fd(
        self,
        requirement: OpenRouterSecretRequirement,
        descriptor: int,
    ) -> None:
        """Write one length-framed secret to a broker pipe, then wipe it.

        This API deliberately cannot return a string or environment mapping.
        The receiving broker protocol remains a separate runtime-closure task;
        callers must never pass this descriptor through Ray metadata.
        """

        if not isinstance(requirement, OpenRouterSecretRequirement):
            raise TypeError("requirement must be OpenRouterSecretRequirement")
        if requirement != self._requirement:
            raise PermissionError("OpenRouter secret requirement differs")
        if isinstance(descriptor, bool) or not isinstance(descriptor, int):
            raise TypeError("broker descriptor must be an integer")
        if descriptor < 0:
            raise ValueError("broker descriptor must be non-negative")
        if self._spent:
            raise RuntimeError("OpenRouter secret capability is already spent")
        object.__setattr__(self, "_spent", True)
        payload = memoryview(self._secret_bytes)
        header = len(payload).to_bytes(4, "big")
        try:
            for segment in (memoryview(header), payload):
                offset = 0
                while offset < len(segment):
                    written = os.write(descriptor, segment[offset:])
                    if written <= 0:
                        raise OSError("broker descriptor made no write progress")
                    offset += written
        finally:
            payload.release()
            self._wipe()

    def close(self) -> None:
        """Idempotently discard an unused secret without materializing it."""

        object.__setattr__(self, "_spent", True)
        self._wipe()

    def _wipe(self) -> None:
        for index in range(len(self._secret_bytes)):
            self._secret_bytes[index] = 0

    def __del__(self) -> None:
        secret = getattr(self, "_secret_bytes", None)
        if isinstance(secret, bytearray):
            for index in range(len(secret)):
                secret[index] = 0


def bind_openrouter_api_key(
    requirement: OpenRouterSecretRequirement,
    secret_value: str,
) -> OpenRouterSecretBinding:
    """Bind an exact value without consulting or copying the host environment."""

    return OpenRouterSecretBinding(requirement, secret_value)


__all__ = [
    "CHILD_ENVIRONMENT_SCHEMA",
    "CLI_WORKER_LATE_ENVIRONMENT_NAMES",
    "OPENROUTER_SECRET_ENVIRONMENT_NAME",
    "OPENROUTER_SECRET_REQUIREMENT_SCHEMA",
    "POLICY_VERL_DRIVER_PROFILE",
    "POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES",
    "REPRESENTATION_TORCHRUN_PROFILE",
    "RUNTIME_PACKAGE_ROOT",
    "SUPPORTED_CHILD_ENVIRONMENT_PROFILES",
    "TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES",
    "ChildEnvironmentBinding",
    "OpenRouterSecretBinding",
    "OpenRouterSecretRequirement",
    "bind_openrouter_api_key",
    "build_child_environment",
    "profile_late_overlay_environment_names",
    "profile_owned_environment_names",
    "scrub_policy_driver_authorization_environment",
    "scrub_representation_worker_authorization_environment",
    "verify_policy_driver_child_environment",
    "verify_representation_torchrun_child_environment",
]
