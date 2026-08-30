"""Leaf contracts for a verified canonical worker startup.

This module deliberately performs no import dispatch and grants no launch
authority.  It only defines the immutable identity that a future bootstrap
must verify and the process-local evidence produced after that verification.
Keeping the contract free of project imports lets a bootstrap inspect it
before importing a training runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re


WORKER_STARTUP_SCHEMA = "tgvf-worker-startup-v1"
WORKER_STARTUP_ENVELOPE_SCHEMA = "tgvf-worker-startup-envelope-v1"
POLICY_DRIVER_ROLE = "policy-driver"
REPRESENTATION_LAUNCHER_ROLE = "representation-launcher"
REPRESENTATION_MEMBER_ROLE = "representation-member"
SUPPORTED_WORKER_STARTUP_ROLES = (
    POLICY_DRIVER_ROLE,
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
)

_SUPPORTED_ROLE_SET = frozenset(SUPPORTED_WORKER_STARTUP_ROLES)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COORDINATE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:/-]*$")
_VERIFIED_WORKER_STARTUP_SENTINEL = object()

_IDENTITY_RECORD_FIELDS = {
    "schema",
    "role",
    "command",
    "target",
    "runtime_package_sha256",
    "dependency_roots_sha256",
    "identity_sha256",
}
_ENVELOPE_RECORD_FIELDS = {
    "schema",
    "entry_role",
    "identities",
}
_ENVELOPE_AUTHORIZATION_PARAMETER_NAMES = frozenset(
    {
        "worker_startup_envelope_schema",
        "worker_startup_envelope_json",
        "worker_startup_envelope_sha256",
    }
)
_POLICY_ENVELOPE_ROLES = frozenset({POLICY_DRIVER_ROLE})
_REPRESENTATION_ENVELOPE_ROLES = frozenset(
    {REPRESENTATION_LAUNCHER_ROLE, REPRESENTATION_MEMBER_ROLE}
)
_ROLE_ORDER = {role: index for index, role in enumerate(SUPPORTED_WORKER_STARTUP_ROLES)}


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"worker startup JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"worker startup JSON contains non-finite value {value!r}")


def _load_strict_json(value: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("worker startup JSON is malformed") from error


def _require_supported_role(role: object) -> str:
    if type(role) is not str or role not in _SUPPORTED_ROLE_SET:
        raise ValueError(
            "worker startup role must be exactly one of: "
            + ", ".join(SUPPORTED_WORKER_STARTUP_ROLES)
        )
    return role


def _require_target(value: object) -> str:
    if type(value) is not str or not _COORDINATE_RE.fullmatch(value):
        raise ValueError("worker startup target must be a canonical coordinate")
    if ".." in value or value.endswith((".", ":", "/", "-")):
        raise ValueError("worker startup target contains an ambiguous segment")
    return value


def _require_command(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("worker startup command must be a non-empty exact tuple")
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"worker startup command[{index}] must be exactly str")
        if not item:
            raise ValueError(f"worker startup command[{index}] must be non-empty")
        if "\x00" in item or "\r" in item or "\n" in item:
            raise ValueError(
                f"worker startup command[{index}] contains a forbidden character"
            )
        try:
            item.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                f"worker startup command[{index}] must be valid UTF-8 text"
            ) from error
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"worker startup {name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class WorkerStartupIdentity:
    """Serializable content identity that a worker bootstrap must verify."""

    role: str
    command: tuple[str, ...]
    target: str
    runtime_package_sha256: str
    dependency_roots_sha256: str

    def __post_init__(self) -> None:
        _require_supported_role(self.role)
        _require_command(self.command)
        _require_target(self.target)
        _require_sha256(
            self.runtime_package_sha256,
            name="runtime_package_sha256",
        )
        _require_sha256(
            self.dependency_roots_sha256,
            name="dependency_roots_sha256",
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": WORKER_STARTUP_SCHEMA,
                "role": self.role,
                "command": list(self.command),
                "target": self.target,
                "runtime_package_sha256": self.runtime_package_sha256,
                "dependency_roots_sha256": self.dependency_roots_sha256,
            }
        )

    def require_role(self, required_role: str) -> None:
        """Require one exact supported role, never a prefix or role family."""

        expected = _require_supported_role(required_role)
        if self.role != expected:
            raise PermissionError(
                f"worker startup role differs: expected {expected!r}, "
                f"received {self.role!r}"
            )

    def as_record(self) -> dict[str, object]:
        """Return the exact JSON-safe identity record including its digest."""

        return {
            "schema": WORKER_STARTUP_SCHEMA,
            "role": self.role,
            "command": list(self.command),
            "target": self.target,
            "runtime_package_sha256": self.runtime_package_sha256,
            "dependency_roots_sha256": self.dependency_roots_sha256,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> WorkerStartupIdentity:
        """Reconstruct one identity from an exact, digest-bound record."""

        if type(value) is not dict or set(value) != _IDENTITY_RECORD_FIELDS:
            raise ValueError("worker startup identity record field set differs")
        if value["schema"] != WORKER_STARTUP_SCHEMA:
            raise ValueError("worker startup identity record schema differs")
        command = value["command"]
        if type(command) is not list:
            raise ValueError("worker startup identity record command must be a list")
        identity = cls(
            role=value["role"],  # type: ignore[arg-type]
            command=tuple(command),  # type: ignore[arg-type]
            target=value["target"],  # type: ignore[arg-type]
            runtime_package_sha256=value[  # type: ignore[arg-type]
                "runtime_package_sha256"
            ],
            dependency_roots_sha256=value[  # type: ignore[arg-type]
                "dependency_roots_sha256"
            ],
        )
        record_sha256 = _require_sha256(
            value["identity_sha256"],
            name="identity_record_sha256",
        )
        if record_sha256 != identity.identity_sha256:
            raise ValueError("worker startup identity record digest differs")
        return identity

    def authorization_parameters(self) -> dict[str, str]:
        """Return the complete deterministic startup identity for a gate."""

        return {
            "worker_startup_schema": WORKER_STARTUP_SCHEMA,
            "worker_startup_role": self.role,
            "worker_startup_command_json": _canonical_json(list(self.command)),
            "worker_startup_command_sha256": _canonical_sha256(list(self.command)),
            "worker_startup_target": self.target,
            "worker_startup_runtime_package_sha256": self.runtime_package_sha256,
            "worker_startup_dependency_roots_sha256": (self.dependency_roots_sha256),
            "worker_startup_identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkerStartupEnvelope:
    """Atomic role-keyed startup identities bound into one CLI authorization."""

    entry_role: str
    identities: tuple[WorkerStartupIdentity, ...]

    def __post_init__(self) -> None:
        entry_role = _require_supported_role(self.entry_role)
        if entry_role not in {POLICY_DRIVER_ROLE, REPRESENTATION_LAUNCHER_ROLE}:
            raise ValueError(
                "worker startup envelope entry role must be policy-driver or "
                "representation-launcher"
            )
        if type(self.identities) is not tuple:
            raise TypeError("worker startup envelope identities must be an exact tuple")
        roles: set[str] = set()
        for identity in self.identities:
            if type(identity) is not WorkerStartupIdentity:
                raise TypeError(
                    "worker startup envelope identities must be exactly "
                    "WorkerStartupIdentity"
                )
            if identity.role in roles:
                raise ValueError(
                    f"worker startup envelope repeats role {identity.role!r}"
                )
            roles.add(identity.role)
        expected_roles = (
            _POLICY_ENVELOPE_ROLES
            if entry_role == POLICY_DRIVER_ROLE
            else _REPRESENTATION_ENVELOPE_ROLES
        )
        if frozenset(roles) != expected_roles:
            missing = sorted(expected_roles.difference(roles))
            extra = sorted(roles.difference(expected_roles))
            raise ValueError(
                "worker startup envelope role set differs: "
                f"missing={missing!r}, extra={extra!r}"
            )
        ordered = tuple(
            sorted(self.identities, key=lambda identity: _ROLE_ORDER[identity.role])
        )
        object.__setattr__(self, "identities", ordered)

    def identity_for_role(self, role: str) -> WorkerStartupIdentity:
        """Return one exact identity already admitted by this envelope shape."""

        expected = _require_supported_role(role)
        for identity in self.identities:
            if identity.role == expected:
                return identity
        raise PermissionError(
            f"worker startup envelope does not contain role {expected!r}"
        )

    def as_record(self) -> dict[str, object]:
        """Return the canonical role-keyed JSON-safe envelope record."""

        return {
            "schema": WORKER_STARTUP_ENVELOPE_SCHEMA,
            "entry_role": self.entry_role,
            "identities": {
                identity.role: identity.as_record() for identity in self.identities
            },
        }

    @property
    def envelope_sha256(self) -> str:
        return _canonical_sha256(self.as_record())

    def to_json(self) -> str:
        """Serialize the envelope with one deterministic canonical encoding."""

        return _canonical_json(self.as_record())

    @classmethod
    def from_record(cls, value: object) -> WorkerStartupEnvelope:
        """Reconstruct an envelope from an exact nested record."""

        if type(value) is not dict or set(value) != _ENVELOPE_RECORD_FIELDS:
            raise ValueError("worker startup envelope record field set differs")
        if value["schema"] != WORKER_STARTUP_ENVELOPE_SCHEMA:
            raise ValueError("worker startup envelope record schema differs")
        identity_records = value["identities"]
        if type(identity_records) is not dict:
            raise ValueError("worker startup envelope identities must be an object")
        identities: list[WorkerStartupIdentity] = []
        for role, identity_record in identity_records.items():
            if type(role) is not str:
                raise ValueError(
                    "worker startup envelope identity role must be a string"
                )
            identity = WorkerStartupIdentity.from_record(identity_record)
            if role != identity.role:
                raise ValueError(
                    "worker startup envelope identity key differs from record role"
                )
            identities.append(identity)
        return cls(
            entry_role=value["entry_role"],  # type: ignore[arg-type]
            identities=tuple(identities),
        )

    @classmethod
    def from_json(cls, value: object) -> WorkerStartupEnvelope:
        """Parse only strict JSON and require its one canonical byte spelling."""

        if type(value) is not str:
            raise TypeError("worker startup envelope JSON must be exactly str")
        envelope = cls.from_record(_load_strict_json(value))
        if envelope.to_json() != value:
            raise ValueError("worker startup envelope JSON is not canonical")
        return envelope

    @classmethod
    def from_authorization_parameters(
        cls,
        parameters: object,
        *,
        expected_entry_role: str,
    ) -> WorkerStartupEnvelope:
        """Verify one complete envelope group inside broader CLI parameters."""

        if type(parameters) is not dict:
            raise TypeError(
                "worker startup envelope authorization parameters must be an exact dict"
            )
        if any(type(key) is not str for key in parameters):
            raise TypeError(
                "worker startup envelope authorization parameter keys must be "
                "exactly str"
            )
        startup_names = {key for key in parameters if key.startswith("worker_startup_")}
        missing = sorted(_ENVELOPE_AUTHORIZATION_PARAMETER_NAMES.difference(parameters))
        extra = sorted(
            startup_names.difference(_ENVELOPE_AUTHORIZATION_PARAMETER_NAMES)
        )
        if missing or extra:
            raise ValueError(
                "worker startup envelope authorization parameter group differs: "
                f"missing={missing!r}, extra={extra!r}"
            )
        values = {
            name: parameters[name] for name in _ENVELOPE_AUTHORIZATION_PARAMETER_NAMES
        }
        if any(type(value) is not str for value in values.values()):
            raise TypeError(
                "worker startup envelope authorization parameter values must be "
                "exactly str"
            )
        if values["worker_startup_envelope_schema"] != WORKER_STARTUP_ENVELOPE_SCHEMA:
            raise ValueError("worker startup envelope authorization schema differs")
        expected = _require_supported_role(expected_entry_role)
        envelope = cls.from_json(values["worker_startup_envelope_json"])
        supplied_sha256 = _require_sha256(
            values["worker_startup_envelope_sha256"],
            name="envelope_authorization_sha256",
        )
        if supplied_sha256 != envelope.envelope_sha256:
            raise ValueError("worker startup envelope authorization SHA256 differs")
        if envelope.entry_role != expected:
            raise PermissionError(
                "worker startup envelope entry role differs: "
                f"expected {expected!r}, received {envelope.entry_role!r}"
            )
        return envelope

    def authorization_parameters(self) -> dict[str, str]:
        """Return one atomic, collision-free authorization parameter group."""

        return {
            "worker_startup_envelope_schema": WORKER_STARTUP_ENVELOPE_SCHEMA,
            "worker_startup_envelope_json": self.to_json(),
            "worker_startup_envelope_sha256": self.envelope_sha256,
        }


class VerifiedWorkerStartup:
    """Non-transferable evidence that one identity passed startup checks.

    The verifier is intentionally not implemented in this leaf.  A later
    bootstrap boundary may construct this object only after authenticating the
    startup envelope and checking the immutable runtime package.  PID binding
    prevents an inherited object from silently acting as evidence after fork.
    """

    __slots__ = ("_identity", "_process_id")

    def __init__(
        self,
        _identity: WorkerStartupIdentity,
        *,
        required_role: str,
    ) -> None:
        del required_role
        raise TypeError(
            "VerifiedWorkerStartup can only be minted by the worker bootstrap"
        )

    @classmethod
    def _mint_for_bootstrap(
        cls,
        identity: WorkerStartupIdentity,
        *,
        required_role: str,
        _sentinel: object,
    ) -> VerifiedWorkerStartup:
        if _sentinel is not _VERIFIED_WORKER_STARTUP_SENTINEL:
            raise TypeError("worker bootstrap mint sentinel differs")
        if type(identity) is not WorkerStartupIdentity:
            raise TypeError("identity must be exactly WorkerStartupIdentity")
        identity.require_role(required_role)
        verified = object.__new__(cls)
        object.__setattr__(verified, "_identity", identity)
        object.__setattr__(verified, "_process_id", os.getpid())
        return verified

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("VerifiedWorkerStartup is immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("VerifiedWorkerStartup cannot be subclassed")

    def __repr__(self) -> str:
        return (
            "VerifiedWorkerStartup("
            f"role={self._identity.role!r}, "
            f"command_sha256={_canonical_sha256(list(self._identity.command))!r}, "
            f"target={self._identity.target!r})"
        )

    def __reduce__(self) -> object:
        raise TypeError("VerifiedWorkerStartup is process-local and not serializable")

    def __copy__(self) -> object:
        raise TypeError("VerifiedWorkerStartup is process-local and not copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("VerifiedWorkerStartup is process-local and not copyable")

    def _require_current_process(self) -> None:
        if os.getpid() != self._process_id:
            raise RuntimeError("VerifiedWorkerStartup belongs to a different process")

    @property
    def identity(self) -> WorkerStartupIdentity:
        self._require_current_process()
        return self._identity

    def require_role(self, required_role: str) -> WorkerStartupIdentity:
        """Return the identity only for the exact role in the creating process."""

        self._require_current_process()
        self._identity.require_role(required_role)
        return self._identity

    def authorization_parameters(self) -> dict[str, str]:
        self._require_current_process()
        return self._identity.authorization_parameters()


def _mint_verified_worker_startup_for_bootstrap(
    identity: WorkerStartupIdentity,
    *,
    required_role: str,
) -> VerifiedWorkerStartup:
    """Mint process-local evidence after bootstrap verification only.

    This private hook is reserved for the future minimal bootstrap module.  It
    does not perform package or envelope verification itself and must never be
    re-exported by a public facade.
    """

    return VerifiedWorkerStartup._mint_for_bootstrap(
        identity,
        required_role=required_role,
        _sentinel=_VERIFIED_WORKER_STARTUP_SENTINEL,
    )


__all__ = [
    "POLICY_DRIVER_ROLE",
    "REPRESENTATION_LAUNCHER_ROLE",
    "REPRESENTATION_MEMBER_ROLE",
    "SUPPORTED_WORKER_STARTUP_ROLES",
    "WORKER_STARTUP_ENVELOPE_SCHEMA",
    "WORKER_STARTUP_SCHEMA",
    "VerifiedWorkerStartup",
    "WorkerStartupEnvelope",
    "WorkerStartupIdentity",
]
