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
    "WORKER_STARTUP_SCHEMA",
    "VerifiedWorkerStartup",
    "WorkerStartupIdentity",
]
