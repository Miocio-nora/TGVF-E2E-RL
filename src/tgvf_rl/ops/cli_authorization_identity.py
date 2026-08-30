"""Immutable config, interpreter, and child-environment launch identities.

This leaf module owns path and byte binding.  Gate consumption and launcher
liveness deliberately remain in :mod:`tgvf_rl.ops.cli_authorization` so an
identity object cannot consume its own authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Mapping

from tgvf_rl.secure_file_read import (
    RetainedRegularFileDescriptor,
    SecureFileReadError,
    retain_regular_file_absolute_nofollow,
)

from .launch_gate import LaunchAuthorizationError, make_run_identity


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_POLICY_CONFIG_ROOT = REPOSITORY_ROOT / "configs/canonical/policy"
CANONICAL_REPRESENTATION_CONFIG_ROOT = (
    REPOSITORY_ROOT / "configs/canonical/representation"
)
CANONICAL_EVALUATION_CONFIG_ROOT = REPOSITORY_ROOT / "configs/canonical/evaluation"
_ENVIRONMENT_SANITIZATION_POLICY = "strip-credentials-and-python-injection-v1"
_COMPATIBILITY_MODULE = "tgvf_rl.ops.cli_authorization"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CLIExecutionAuthorizationIdentity:
    """Immutable identity required to authorize one public CLI dispatch."""

    run_id: str
    phase: str
    command_id: str
    run_identity_sha256: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("run_id", "phase", "command_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{field_name} must be a non-empty string without NUL")
        if not isinstance(self.run_identity_sha256, str) or not _SHA256_RE.fullmatch(
            self.run_identity_sha256
        ):
            raise ValueError("run_identity_sha256 must be a lowercase SHA-256 digest")
        normalized = tuple(sorted(self.parameters))
        if normalized != self.parameters:
            raise ValueError("CLI authorization parameters must be sorted")
        keys: set[str] = set()
        for key, value in self.parameters:
            if (
                not isinstance(key, str)
                or not key
                or "\x00" in key
                or key == "run_identity_sha256"
            ):
                raise ValueError("CLI authorization parameter name is invalid")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("CLI authorization parameter value must be a string")
            if key in keys:
                raise ValueError("CLI authorization parameter names must be unique")
            keys.add(key)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        phase: str,
        command_id: str,
        run_identity_sha256: str,
        parameters: Mapping[str, str] | None = None,
    ) -> CLIExecutionAuthorizationIdentity:
        return cls(
            run_id=run_id,
            phase=phase,
            command_id=command_id,
            run_identity_sha256=run_identity_sha256,
            parameters=tuple(sorted(dict(parameters or {}).items())),
        )

    @property
    def gate_run_identity(self) -> dict[str, object]:
        parameters = {
            "run_identity_sha256": self.run_identity_sha256,
            **dict(self.parameters),
        }
        return make_run_identity(
            run_id=self.run_id,
            phase=self.phase,
            command_id=self.command_id,
            parameters=parameters,
        )

    def as_record(self) -> dict[str, object]:
        """Return the strict JSON-safe identity inherited by a worker."""

        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "command_id": self.command_id,
            "run_identity_sha256": self.run_identity_sha256,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_record(cls, value: object) -> CLIExecutionAuthorizationIdentity:
        if not isinstance(value, dict) or set(value) != {
            "run_id",
            "phase",
            "command_id",
            "run_identity_sha256",
            "parameters",
        }:
            raise LaunchAuthorizationError(
                "CLI worker execution identity has an unexpected field set"
            )
        parameters = value["parameters"]
        if not isinstance(parameters, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in parameters.items()
        ):
            raise LaunchAuthorizationError(
                "CLI worker execution identity parameters are invalid"
            )
        try:
            return cls.create(
                run_id=value["run_id"],
                phase=value["phase"],
                command_id=value["command_id"],
                run_identity_sha256=value["run_identity_sha256"],
                parameters=parameters,
            )
        except (TypeError, ValueError) as error:
            raise LaunchAuthorizationError(
                "CLI worker execution identity is malformed"
            ) from error


@dataclass(frozen=True, slots=True)
class CLIWorkerAuthorization:
    """Exact consumed authorization and live launcher inherited by workers."""

    consumption_receipt_path: Path
    consumption_receipt_sha256: str
    launcher_liveness_receipt_path: Path


@dataclass(frozen=True, slots=True)
class CanonicalConfigBinding:
    """Securely opened identity of one public mutating command configuration."""

    canonical_root: Path
    source_path: Path
    resolved_path: Path
    source_sha256: str
    byte_length: int
    device: int
    inode: int
    mode: int

    def authorization_parameters(self) -> dict[str, str]:
        return {
            "canonical_config_root": str(self.canonical_root),
            "canonical_config_path": str(self.source_path),
            "canonical_config_realpath": str(self.resolved_path),
            "canonical_config_sha256": self.source_sha256,
            "canonical_config_size": str(self.byte_length),
            "canonical_config_device": str(self.device),
            "canonical_config_inode": str(self.inode),
        }


@dataclass(frozen=True, slots=True)
class PythonExecutableIdentity:
    """Serializable identity of exact securely opened interpreter bytes."""

    declared_path: Path
    resolved_path: Path
    sha256: str
    byte_length: int
    device: int
    inode: int
    mode: int

    def authorization_parameters(self) -> dict[str, str]:
        return {
            "python_executable": str(self.declared_path),
            "python_executable_realpath": str(self.resolved_path),
            "python_executable_sha256": self.sha256,
            "python_executable_size": str(self.byte_length),
            "python_executable_device": str(self.device),
            "python_executable_inode": str(self.inode),
            "python_executable_mode": oct(self.mode & 0o7777),
        }


class PythonExecutableBinding:
    """Process-local ownership of the descriptor paired with an identity.

    The descriptor is intentionally absent from :class:`PythonExecutableIdentity`
    and every authorization record.  This object is an in-process capability,
    not a serializable authority artifact.
    """

    __slots__ = ("_identity", "_retained")

    def __init__(
        self,
        identity: PythonExecutableIdentity,
        retained: RetainedRegularFileDescriptor,
    ) -> None:
        if not isinstance(identity, PythonExecutableIdentity):
            raise TypeError("identity must be PythonExecutableIdentity")
        if not isinstance(retained, RetainedRegularFileDescriptor):
            raise TypeError("retained must be RetainedRegularFileDescriptor")
        self._identity = identity
        self._retained = retained

    @property
    def identity(self) -> PythonExecutableIdentity:
        """Return the immutable serializable half of this capability."""

        return self._identity

    @property
    def closed(self) -> bool:
        return self._retained.closed

    def fileno(self) -> int:
        return self._retained.fileno()

    def close(self) -> None:
        """Idempotently release an abandoned or failed launch binding."""

        self._retained.close()

    def __enter__(self) -> PythonExecutableBinding:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __reduce__(self) -> object:
        raise TypeError("PythonExecutableBinding is process-local and not serializable")


def bind_canonical_config_path(
    path: str | Path,
    *,
    canonical_root: str | Path,
) -> CanonicalConfigBinding:
    """Bind a config below one strict root without following any symlink."""

    root = _lexical_absolute_path(canonical_root, label="canonical config root")
    source = _lexical_absolute_path(path, label="public mutating config")
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise LaunchAuthorizationError(
            f"public mutating config is outside canonical root {root}: {source}"
        ) from error
    if not relative.parts:
        raise LaunchAuthorizationError("public mutating config cannot be a directory")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_source = source.resolve(strict=True)
        resolved_source.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise LaunchAuthorizationError(
            "public mutating config does not resolve inside its canonical root"
        ) from error
    raw, observed = _read_regular_file_without_symlinks(
        source,
        label="public mutating config",
    )
    if resolved_source != source:
        raise LaunchAuthorizationError(
            "public mutating config lexical path differs from its real path"
        )
    return CanonicalConfigBinding(
        canonical_root=root,
        source_path=source,
        resolved_path=resolved_source,
        source_sha256=sha256(raw).hexdigest(),
        byte_length=observed.st_size,
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
    )


def verify_canonical_config_binding(binding: CanonicalConfigBinding) -> None:
    """Re-open and compare the same config identity immediately before use."""

    if not isinstance(binding, CanonicalConfigBinding):
        raise TypeError("binding must be CanonicalConfigBinding")
    rebound = bind_canonical_config_path(
        binding.source_path,
        canonical_root=binding.canonical_root,
    )
    if rebound != binding:
        raise LaunchAuthorizationError(
            "canonical config identity changed after launch preflight"
        )


def assert_loaded_config_matches_binding(
    config: object,
    binding: CanonicalConfigBinding,
    *,
    source_sha256_attribute: str,
) -> None:
    """Prove the format-specific loader consumed the bytes bound above."""

    verify_canonical_config_binding(binding)
    source_path = getattr(config, "source_path", None)
    source_sha256 = getattr(config, source_sha256_attribute, None)
    if (
        not isinstance(source_path, Path)
        or source_path.resolve() != binding.resolved_path
    ):
        raise LaunchAuthorizationError(
            "loaded config source path differs from canonical config binding"
        )
    if source_sha256 != binding.source_sha256:
        raise LaunchAuthorizationError(
            "loaded config source bytes differ from canonical config binding"
        )


def bind_current_python_executable(
    requested: str | Path,
    *,
    current_executable: str | Path | None = None,
) -> PythonExecutableIdentity:
    """Return a serializable identity and close its temporary descriptor."""

    binding = bind_current_python_executable_for_exec(
        requested,
        current_executable=current_executable,
        require_fd_exec_support=False,
    )
    try:
        return binding.identity
    finally:
        binding.close()


def bind_current_python_executable_for_exec(
    requested: str | Path,
    *,
    current_executable: str | Path | None = None,
    require_fd_exec_support: bool = True,
) -> PythonExecutableBinding:
    """Bind and retain the interpreter inode that a later exec must consume."""

    if require_fd_exec_support:
        assert_fd_exec_supported()
    current = _lexical_absolute_path(
        sys.executable if current_executable is None else current_executable,
        label="current Python executable",
    )
    candidate = _lexical_absolute_path(requested, label="requested Python executable")
    if candidate != current:
        raise LaunchAuthorizationError(
            "--python must be the exact current audited sys.executable"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise LaunchAuthorizationError(
            "current Python executable cannot be resolved"
        ) from error
    process_metadata = _running_process_executable_metadata(
        resolved,
        required=require_fd_exec_support,
    )
    retained: RetainedRegularFileDescriptor | None = None
    try:
        retained = retain_regular_file_absolute_nofollow(resolved)
        snapshot = retained.snapshot()
        _assert_stable_snapshot(snapshot.before, snapshot.after, label="Python")
        observed = snapshot.after
        if process_metadata is not None and (
            observed.st_dev,
            observed.st_ino,
        ) != (
            process_metadata.st_dev,
            process_metadata.st_ino,
        ):
            raise LaunchAuthorizationError(
                "bound Python executable is not the running process executable"
            )
        if stat.S_ISLNK(observed.st_mode) or observed.st_mode & 0o111 == 0:
            raise LaunchAuthorizationError(
                "resolved Python executable must be a non-symlink executable regular file"
            )
        identity = PythonExecutableIdentity(
            declared_path=candidate,
            resolved_path=resolved,
            sha256=sha256(snapshot.payload).hexdigest(),
            byte_length=observed.st_size,
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=observed.st_mode,
        )
        binding = PythonExecutableBinding(identity, retained)
        retained = None
        return binding
    except (OSError, SecureFileReadError) as error:
        raise LaunchAuthorizationError(
            "resolved Python executable is missing, unreadable, or contains a symlink: "
            f"{resolved}"
        ) from error
    finally:
        if retained is not None:
            retained.close()


def _running_process_executable_metadata(
    resolved: Path,
    *,
    required: bool,
) -> os.stat_result | None:
    """Bind the candidate path to the immutable image of this process."""

    process_executable = Path("/proc/self/exe")
    try:
        process_resolved = process_executable.resolve(strict=True)
        process_metadata = os.stat(process_executable, follow_symlinks=True)
    except OSError as error:
        if required:
            raise LaunchAuthorizationError(
                "running process executable identity is unavailable"
            ) from error
        return None
    if process_resolved != resolved:
        raise LaunchAuthorizationError(
            "sys.executable does not resolve to the running process executable"
        )
    if not stat.S_ISREG(process_metadata.st_mode):
        raise LaunchAuthorizationError(
            "running process executable is not a regular file"
        )
    return process_metadata


def verify_python_executable_identity(identity: PythonExecutableIdentity) -> None:
    """Reverify an identity through its declared path for non-exec consumers."""

    if not isinstance(identity, PythonExecutableIdentity):
        raise TypeError("identity must be PythonExecutableIdentity")
    rebound = bind_current_python_executable(identity.declared_path)
    if rebound != identity:
        raise LaunchAuthorizationError(
            "Python executable identity changed after launch preflight"
        )


def verify_python_executable_binding(binding: PythonExecutableBinding) -> int:
    """Revalidate metadata and bytes on the retained descriptor, then return it."""

    if not isinstance(binding, PythonExecutableBinding):
        raise TypeError("binding must be PythonExecutableBinding")
    try:
        snapshot = binding._retained.snapshot()  # noqa: SLF001
    except (OSError, SecureFileReadError) as error:
        raise LaunchAuthorizationError(
            "retained Python executable descriptor cannot be verified"
        ) from error
    _assert_stable_snapshot(snapshot.before, snapshot.after, label="Python")
    observed = snapshot.after
    identity = binding.identity
    if (
        sha256(snapshot.payload).hexdigest() != identity.sha256
        or observed.st_size != identity.byte_length
        or observed.st_dev != identity.device
        or observed.st_ino != identity.inode
        or observed.st_mode != identity.mode
        or observed.st_mode & 0o111 == 0
    ):
        raise LaunchAuthorizationError(
            "retained Python executable identity changed after launch preflight"
        )
    return binding.fileno()


def assert_fd_exec_supported() -> None:
    """Fail closed unless this platform accepts an fd as ``os.execve`` path."""

    if os.execve not in os.supports_fd:
        raise LaunchAuthorizationError(
            "platform lacks required file-descriptor os.execve support"
        )


def sanitized_child_environment(
    base: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove credential and interpreter-injection state from a child env."""

    source = os.environ if base is None else base
    result: dict[str, str] = {}
    stripped: list[str] = []
    for name, value in source.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("child environment keys and values must be strings")
        if _is_sensitive_or_injectable_environment_name(name):
            stripped.append(name)
        else:
            result[name] = value
    return result, tuple(sorted(stripped))


def environment_sanitization_parameters(
    stripped_names: tuple[str, ...],
) -> dict[str, str]:
    """Record the non-secret residual identity without exposing secret values."""

    if tuple(sorted(stripped_names)) != stripped_names:
        raise ValueError("stripped environment names must be sorted")
    raw = json.dumps(list(stripped_names), separators=(",", ":")).encode("utf-8")
    return {
        "child_environment_policy": _ENVIRONMENT_SANITIZATION_POLICY,
        "stripped_environment_name_count": str(len(stripped_names)),
        "stripped_environment_names_sha256": sha256(raw).hexdigest(),
    }


def _lexical_absolute_path(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if "\x00" in os.fspath(raw) or ".." in raw.parts:
        raise LaunchAuthorizationError(f"{label} path is unsafe")
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise LaunchAuthorizationError(f"{label} path must be absolute")
    return absolute


def _read_regular_file_without_symlinks(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    source = _lexical_absolute_path(path, label=label)
    try:
        with retain_regular_file_absolute_nofollow(source) as retained:
            snapshot = retained.snapshot()
    except (OSError, SecureFileReadError) as error:
        raise LaunchAuthorizationError(
            f"{label} is missing, unreadable, or contains a symlink: {source}"
        ) from error
    _assert_stable_snapshot(snapshot.before, snapshot.after, label=label)
    return snapshot.payload, snapshot.after


def _assert_stable_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    fields_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
    )
    fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
    )
    if fields_before != fields_after:
        raise LaunchAuthorizationError(f"{label} changed while it was read")


def _is_sensitive_or_injectable_environment_name(name: str) -> bool:
    upper = name.upper()
    if upper in {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "LD_PRELOAD",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }:
        return True
    if upper.startswith(("AWS_", "AZURE_", "OPENAI_", "OPENROUTER_")):
        return True
    return upper.endswith(
        (
            "_API_KEY",
            "_ACCESS_KEY",
            "_SECRET_KEY",
            "_AUTH_TOKEN",
            "_PASSWORD",
            "_CREDENTIALS",
        )
    ) or upper in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


# These established public objects remain picklable and introspectable at the
# original facade coordinate after the physical split.
for _compatibility_object in (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    CanonicalConfigBinding,
    PythonExecutableIdentity,
    assert_loaded_config_matches_binding,
    bind_canonical_config_path,
    bind_current_python_executable,
    environment_sanitization_parameters,
    sanitized_child_environment,
    verify_canonical_config_binding,
    verify_python_executable_identity,
):
    _compatibility_object.__module__ = _COMPATIBILITY_MODULE


__all__ = [
    "CANONICAL_EVALUATION_CONFIG_ROOT",
    "CANONICAL_POLICY_CONFIG_ROOT",
    "CANONICAL_REPRESENTATION_CONFIG_ROOT",
    "CLIExecutionAuthorizationIdentity",
    "CLIWorkerAuthorization",
    "CanonicalConfigBinding",
    "PythonExecutableBinding",
    "PythonExecutableIdentity",
    "assert_fd_exec_supported",
    "assert_loaded_config_matches_binding",
    "bind_canonical_config_path",
    "bind_current_python_executable",
    "bind_current_python_executable_for_exec",
    "environment_sanitization_parameters",
    "sanitized_child_environment",
    "verify_canonical_config_binding",
    "verify_python_executable_binding",
    "verify_python_executable_identity",
]
