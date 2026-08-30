"""Run-bound authorization consumption for public mutating CLI commands.

The public CLI may derive an execution identity from a validated run config,
but it must not mint readiness or authorization on the caller's behalf.  This
adapter therefore accepts only explicit operator-created gate artifacts,
requires their exact canonical identity, and consumes them against the fixed
repository execution policy immediately before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Mapping, NoReturn

from .launch_gate import (
    CONSUMPTION_SCHEMA,
    LaunchAuthorizationError,
    _load_execution_policy,
    assert_process_liveness,
    consume_launch_authorization,
    gate_status,
    make_run_identity,
    write_process_liveness_receipt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_EXECUTION_POLICY_PATH = (
    REPOSITORY_ROOT / "configs/ops/experiment_execution_policy.json"
)
CANONICAL_POLICY_CONFIG_ROOT = REPOSITORY_ROOT / "configs/canonical/policy"
CANONICAL_REPRESENTATION_CONFIG_ROOT = (
    REPOSITORY_ROOT / "configs/canonical/representation"
)
CANONICAL_EVALUATION_CONFIG_ROOT = REPOSITORY_ROOT / "configs/canonical/evaluation"
CLI_WORKER_AUTHORIZATION_SCHEMA = "tgvf-cli-worker-authorization-environment-v1"
_CLI_WORKER_ENVIRONMENT_NAMES = (
    "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA",
    "TGVF_CLI_EXECUTION_IDENTITY_JSON",
    "TGVF_CLI_GATE_DIRECTORY",
    "TGVF_CLI_CONSUMPTION_RECEIPT_PATH",
    "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256",
    "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH",
)
_ENVIRONMENT_SANITIZATION_POLICY = "strip-credentials-and-python-injection-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONSUMPTION_FIELDS = {
    "schema_version",
    "status",
    "token_id",
    "consumed_at",
    "consumed_by",
    "consumer_pid",
    "consumer_host",
    "authorization_sha256",
    "ready_receipt_sha256",
    "run_identity_sha256",
    "run_id",
    "phase",
    "execution_policy_id",
    "execution_policy_revision",
    "execution_policy_sha256",
    "execution_mode",
    "freeze_override_id",
    "freeze_override_sha256",
}


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
    ) -> "CLIExecutionAuthorizationIdentity":
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
        """Return the strict JSON-safe identity inherited by an authorized worker."""

        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "command_id": self.command_id,
            "run_identity_sha256": self.run_identity_sha256,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_record(cls, value: object) -> "CLIExecutionAuthorizationIdentity":
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
    """The exact current interpreter and securely opened final executable bytes."""

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


def bind_canonical_config_path(
    path: str | Path,
    *,
    canonical_root: str | Path,
) -> CanonicalConfigBinding:
    """Bind a config below one strict root without following any symlink.

    This check intentionally runs before a format-specific loader.  Legacy
    configuration trees remain readable by validation and planning commands,
    but they cannot authorize a mutating command.
    """

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
        # openat/O_NOFOLLOW above already rejects every symlink.  Keep this
        # explicit equality so mount/path aliasing cannot silently change the
        # identity recorded in the launch gate.
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
    """Re-open and compare the same config identity immediately before exec/run."""

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
    """Accept only the interpreter running this audited CLI process."""

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
    process_executable = Path("/proc/self/exe")
    if process_executable.exists():
        try:
            if process_executable.resolve(strict=True) != resolved:
                raise LaunchAuthorizationError(
                    "sys.executable does not resolve to the running process executable"
                )
        except OSError as error:
            raise LaunchAuthorizationError(
                "running process executable identity cannot be resolved"
            ) from error
    raw, observed = _read_regular_file_without_symlinks(
        resolved,
        label="resolved Python executable",
    )
    if stat.S_ISLNK(observed.st_mode) or observed.st_mode & 0o111 == 0:
        raise LaunchAuthorizationError(
            "resolved Python executable must be a non-symlink executable regular file"
        )
    if not os.access(candidate, os.X_OK):
        raise LaunchAuthorizationError("current Python executable is not executable")
    return PythonExecutableIdentity(
        declared_path=candidate,
        resolved_path=resolved,
        sha256=sha256(raw).hexdigest(),
        byte_length=observed.st_size,
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
    )


def verify_python_executable_identity(identity: PythonExecutableIdentity) -> None:
    """Reverify the current interpreter path, target, metadata, and bytes."""

    if not isinstance(identity, PythonExecutableIdentity):
        raise TypeError("identity must be PythonExecutableIdentity")
    rebound = bind_current_python_executable(identity.declared_path)
    if rebound != identity:
        raise LaunchAuthorizationError(
            "Python executable identity changed after launch preflight"
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


def cli_worker_authorization_environment(
    identity: CLIExecutionAuthorizationIdentity,
    worker_authorization: CLIWorkerAuthorization,
    *,
    gate_directory: str | Path,
) -> dict[str, str]:
    """Serialize only a consumed receipt and live-launcher proof for a worker."""

    if not isinstance(identity, CLIExecutionAuthorizationIdentity):
        raise TypeError("identity must be CLIExecutionAuthorizationIdentity")
    if not isinstance(worker_authorization, CLIWorkerAuthorization):
        raise TypeError("worker_authorization must be CLIWorkerAuthorization")
    gate = _existing_gate_directory(gate_directory)
    identity_json = json.dumps(
        identity.as_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": CLI_WORKER_AUTHORIZATION_SCHEMA,
        "TGVF_CLI_EXECUTION_IDENTITY_JSON": identity_json,
        "TGVF_CLI_GATE_DIRECTORY": str(gate),
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": str(
            worker_authorization.consumption_receipt_path
        ),
        "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": (
            worker_authorization.consumption_receipt_sha256
        ),
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": str(
            worker_authorization.launcher_liveness_receipt_path
        ),
    }


def verify_cli_worker_authorization_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    expected_phase: str,
    expected_command_id: str,
) -> CLIExecutionAuthorizationIdentity:
    """First worker action: reconstruct and verify consumed outer authorization."""

    values = os.environ if environment is None else environment
    present = tuple(name for name in _CLI_WORKER_ENVIRONMENT_NAMES if values.get(name))
    if len(present) != len(_CLI_WORKER_ENVIRONMENT_NAMES):
        qualifier = "missing" if not present else "partial"
        raise LaunchAuthorizationError(
            f"CLI worker authorization environment is {qualifier}"
        )
    if values[_CLI_WORKER_ENVIRONMENT_NAMES[0]] != CLI_WORKER_AUTHORIZATION_SCHEMA:
        raise LaunchAuthorizationError("CLI worker authorization schema differs")
    try:
        record = json.loads(values["TGVF_CLI_EXECUTION_IDENTITY_JSON"])
    except (TypeError, json.JSONDecodeError) as error:
        raise LaunchAuthorizationError(
            "CLI worker execution identity is not valid JSON"
        ) from error
    identity = CLIExecutionAuthorizationIdentity.from_record(record)
    if identity.phase != expected_phase or identity.command_id != expected_command_id:
        raise LaunchAuthorizationError(
            "CLI worker execution identity is for a different phase or command"
        )
    verify_cli_worker_authorization(
        identity,
        gate_directory=values["TGVF_CLI_GATE_DIRECTORY"],
        consumption_receipt_path=values["TGVF_CLI_CONSUMPTION_RECEIPT_PATH"],
        expected_consumption_receipt_sha256=values[
            "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256"
        ],
        launcher_liveness_receipt_path=values[
            "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH"
        ],
    )
    return identity


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
    """Use openat/O_NOFOLLOW for every ancestor and the final file."""

    source = _lexical_absolute_path(path, label=label)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise LaunchAuthorizationError(
            f"{label} cannot be verified without openat/O_NOFOLLOW support"
        )
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        for component in source.parts[1:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LaunchAuthorizationError(f"{label} is not a regular file: {source}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        ):
            raise LaunchAuthorizationError(f"{label} changed while it was read")
        return b"".join(chunks), after
    except LaunchAuthorizationError:
        raise
    except OSError as error:
        raise LaunchAuthorizationError(
            f"{label} is missing, unreadable, or contains a symlink: {source}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


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


def assert_legacy_standalone_execution_quarantined(tool_id: str) -> NoReturn:
    """Reject an unmigrated standalone launcher under the fixed repo policy.

    Legacy controllers intentionally have no token or freeze-override arguments.
    They cannot be revived by copying the public CLI's authorization artifacts;
    migration to a run-bound canonical entry point is required instead.  Reading
    and validating the fixed repository policy here makes a missing, malformed,
    or relocated policy fail closed as well.
    """

    if not isinstance(tool_id, str) or not tool_id.strip() or "\x00" in tool_id:
        raise LaunchAuthorizationError("legacy standalone tool ID is invalid")
    policy, policy_sha256 = _load_execution_policy(REPOSITORY_EXECUTION_POLICY_PATH)
    if policy["execution_mode"] == "frozen":
        raise LaunchAuthorizationError(
            f"legacy standalone tool {tool_id} is quarantined by frozen "
            f"repository policy {policy['policy_id']} revision "
            f"{policy['revision']} ({policy_sha256}); this entry point accepts "
            "no authorization token or freeze override and cannot execute"
        )
    raise LaunchAuthorizationError(
        f"legacy standalone tool {tool_id} remains quarantined because it has "
        "no run-bound canonical launch authorization; migrate it before execution"
    )


def assert_legacy_standalone_mode_quarantined(
    tool_id: str,
    *,
    selected_mode: str,
    read_only_modes: tuple[str, ...],
    blocked_modes: tuple[str, ...],
) -> None:
    """Allow only enumerated read-only modes; permanently reject all others."""

    if not isinstance(selected_mode, str) or not selected_mode.strip():
        raise LaunchAuthorizationError("legacy standalone selected mode is invalid")
    for label, modes in (
        ("read-only", read_only_modes),
        ("blocked", blocked_modes),
    ):
        if (
            not isinstance(modes, tuple)
            or not modes
            or any(
                not isinstance(mode, str) or not mode.strip() or "\x00" in mode
                for mode in modes
            )
            or len(set(modes)) != len(modes)
        ):
            raise LaunchAuthorizationError(
                f"legacy standalone {label} mode inventory is invalid"
            )
    overlap = set(read_only_modes) & set(blocked_modes)
    if overlap:
        raise LaunchAuthorizationError(
            f"legacy standalone mode inventories overlap: {sorted(overlap)}"
        )
    if selected_mode in blocked_modes:
        assert_legacy_standalone_execution_quarantined(
            f"{tool_id} mode={selected_mode}"
        )
    if selected_mode not in read_only_modes:
        raise LaunchAuthorizationError(
            f"legacy standalone tool {tool_id} has unclassified mode "
            f"{selected_mode!r}; execution is denied"
        )


def _existing_gate_directory(path: str | Path) -> Path:
    gate = Path(path).expanduser().absolute()
    if gate.is_symlink() or not gate.is_dir():
        raise LaunchAuthorizationError(
            f"CLI authorization gate must be an existing non-symlink directory: {gate}"
        )
    return gate


def _require_non_symlink_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise LaunchAuthorizationError(
            f"{label} must be an existing non-symlink directory: {path}"
        )


def _assert_ready_identity(
    gate: Path,
    identity: CLIExecutionAuthorizationIdentity,
) -> dict[str, object]:
    status = gate_status(gate)
    if status["ready_error"] is not None:
        raise LaunchAuthorizationError(
            f"CLI authorization ready receipt is invalid: {status['ready_error']}"
        )
    ready = status["ready"]
    if not isinstance(ready, dict):
        raise LaunchAuthorizationError("CLI authorization ready receipt is missing")
    expected = identity.gate_run_identity
    if (
        ready.get("run_identity") != expected
        or ready.get("run_identity_sha256") != expected["identity_sha256"]
    ):
        raise LaunchAuthorizationError(
            "CLI authorization ready receipt differs from command, run, phase, "
            "run identity, or execution parameters"
        )
    return ready


def _sha256_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise LaunchAuthorizationError(
            f"{label} must be an existing non-symlink regular file: {path}"
        )
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LaunchAuthorizationError(
            f"{label} must be an existing non-symlink regular file: {path}"
        )
    try:
        raw = path.read_bytes()
        if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
            raise LaunchAuthorizationError(f"{label} SHA256 differs")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchAuthorizationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise LaunchAuthorizationError(f"{label} must contain one JSON object")
    return payload


def _assert_process_descends_from(ancestor_pid: int) -> None:
    """Require the current Linux process to descend from the live launcher."""

    if isinstance(ancestor_pid, bool) or not isinstance(ancestor_pid, int):
        raise LaunchAuthorizationError("CLI launcher PID is invalid")
    observed: set[int] = set()
    current = os.getpid()
    for _ in range(256):
        if current == ancestor_pid:
            return
        if current <= 1 or current in observed:
            break
        observed.add(current)
        stat_path = Path("/proc") / str(current) / "stat"
        try:
            stat_text = stat_path.read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as error:
            raise LaunchAuthorizationError(
                "CLI worker cannot verify its Linux /proc launcher ancestry"
            ) from error
        closing = stat_text.rfind(")")
        if closing < 0:
            raise LaunchAuthorizationError("CLI worker /proc ancestry is malformed")
        fields = stat_text[closing + 2 :].split()
        try:
            current = int(fields[1])
        except (IndexError, ValueError) as error:
            raise LaunchAuthorizationError(
                "CLI worker /proc ancestry has no parent PID"
            ) from error
    raise LaunchAuthorizationError(
        "CLI worker process is not a descendant of the authorized live launcher"
    )


def consume_cli_execution_authorization(
    identity: CLIExecutionAuthorizationIdentity,
    *,
    gate_directory: str | Path,
    authorization_token_path: str | Path,
    freeze_override_path: str | Path | None,
) -> dict[str, object]:
    """Consume explicit gate artifacts only after exact identity comparison."""

    if not isinstance(identity, CLIExecutionAuthorizationIdentity):
        raise TypeError("identity must be CLIExecutionAuthorizationIdentity")
    gate = _existing_gate_directory(gate_directory)
    _assert_ready_identity(gate, identity)
    expected = identity.gate_run_identity
    policy, policy_sha256 = _load_execution_policy(REPOSITORY_EXECUTION_POLICY_PATH)
    override = (
        None
        if freeze_override_path is None
        else _lexical_absolute_path(freeze_override_path, label="freeze override")
    )
    if policy["execution_mode"] == "frozen" and override is None:
        raise LaunchAuthorizationError(
            "repository execution is frozen; --freeze-override is required"
        )
    if policy["execution_mode"] == "open" and override is not None:
        raise LaunchAuthorizationError(
            "--freeze-override must be omitted while repository execution is open"
        )
    consumption = consume_launch_authorization(
        gate,
        Path(authorization_token_path),
        REPOSITORY_EXECUTION_POLICY_PATH,
        expected_run_id=identity.run_id,
        expected_phase=identity.phase,
        freeze_override_path=override,
        consumed_by="tgvf-rl-public-cli",
    )
    if consumption.get("run_identity_sha256") != expected["identity_sha256"]:
        raise RuntimeError("launch-gate consumption changed the expected CLI identity")
    if (
        consumption.get("execution_policy_id") != policy["policy_id"]
        or consumption.get("execution_policy_revision") != policy["revision"]
        or consumption.get("execution_policy_sha256") != policy_sha256
        or consumption.get("execution_mode") != policy["execution_mode"]
    ):
        raise LaunchAuthorizationError(
            "repository execution policy changed during CLI authorization consumption"
        )
    return consumption


def assert_canonical_runtime_launch_enabled() -> None:
    """Reject canonical mutation while any non-overridable runtime blocker remains.

    A freeze override controls experiment scheduling only. It cannot waive an
    incomplete executable, worker-envelope, artifact, compiler, or environment
    security contract.
    """

    policy, _ = _load_execution_policy(REPOSITORY_EXECUTION_POLICY_PATH)
    closure = policy["runtime_closure"]
    if closure["launch_enabled"] is not True:
        blockers = ", ".join(closure["blocker_ids"])
        raise LaunchAuthorizationError(
            "canonical runtime closure is incomplete and cannot be bypassed by "
            f"a freeze override: {blockers}"
        )


def materialize_cli_worker_authorization(
    identity: CLIExecutionAuthorizationIdentity,
    consumption: Mapping[str, object],
    *,
    gate_directory: str | Path,
) -> CLIWorkerAuthorization:
    """Publish launcher liveness after consumption and before outer exec."""

    if not isinstance(identity, CLIExecutionAuthorizationIdentity):
        raise TypeError("identity must be CLIExecutionAuthorizationIdentity")
    gate = _existing_gate_directory(gate_directory)
    _assert_ready_identity(gate, identity)
    token_id = consumption.get("token_id")
    consumption_path_value = consumption.get("consumption_path")
    if not isinstance(token_id, str) or not _TOKEN_ID_RE.fullmatch(token_id):
        raise LaunchAuthorizationError("CLI consumption token ID is malformed")
    if not isinstance(consumption_path_value, str):
        raise LaunchAuthorizationError("CLI consumption receipt path is missing")
    consumption_path = Path(consumption_path_value).expanduser().absolute()
    _require_non_symlink_directory(
        gate / "consumptions",
        label="CLI consumptions directory",
    )
    expected_consumption_path = gate / "consumptions" / f"{token_id}.json"
    if consumption_path != expected_consumption_path:
        raise LaunchAuthorizationError(
            "CLI consumption receipt is outside its exact gate directory"
        )
    consumption_sha256 = _sha256_file(
        consumption_path,
        label="CLI consumption receipt",
    )
    launches_root = gate / "cli-launches"
    if launches_root.is_symlink():
        raise LaunchAuthorizationError("CLI launches directory must not be a symlink")
    token_launch_directory = launches_root / token_id
    if token_launch_directory.exists() or token_launch_directory.is_symlink():
        raise LaunchAuthorizationError(
            "CLI launcher directory already exists for this one-time token"
        )
    liveness_path = token_launch_directory / "launcher-liveness.json"
    liveness = write_process_liveness_receipt(
        liveness_path,
        run_identity=identity.gate_run_identity,
    )
    if liveness["pid"] != consumption.get("consumer_pid"):
        raise LaunchAuthorizationError(
            "CLI authorization consumer differs from the outer launcher process"
        )
    return CLIWorkerAuthorization(
        consumption_receipt_path=consumption_path,
        consumption_receipt_sha256=consumption_sha256,
        launcher_liveness_receipt_path=liveness_path,
    )


def verify_cli_worker_authorization(
    identity: CLIExecutionAuthorizationIdentity,
    *,
    gate_directory: str | Path,
    consumption_receipt_path: str | Path,
    expected_consumption_receipt_sha256: str,
    launcher_liveness_receipt_path: str | Path,
) -> dict[str, object]:
    """Verify a live, non-replayable outer launch before worker dispatch."""

    if not isinstance(identity, CLIExecutionAuthorizationIdentity):
        raise TypeError("identity must be CLIExecutionAuthorizationIdentity")
    if not isinstance(
        expected_consumption_receipt_sha256, str
    ) or not _SHA256_RE.fullmatch(expected_consumption_receipt_sha256):
        raise LaunchAuthorizationError(
            "expected consumption receipt SHA256 is malformed"
        )
    gate = _existing_gate_directory(gate_directory)
    ready = _assert_ready_identity(gate, identity)
    _require_non_symlink_directory(
        gate / "consumptions",
        label="CLI consumptions directory",
    )
    consumption_path = Path(consumption_receipt_path).expanduser().absolute()
    consumption = _load_json(
        consumption_path,
        label="CLI consumption receipt",
        expected_sha256=expected_consumption_receipt_sha256,
    )
    if set(consumption) != _CONSUMPTION_FIELDS:
        raise LaunchAuthorizationError(
            "CLI consumption receipt has an unexpected field set"
        )
    if (
        consumption.get("schema_version") != CONSUMPTION_SCHEMA
        or consumption.get("status") != "consumed"
        or consumption.get("consumed_by") != "tgvf-rl-public-cli"
    ):
        raise LaunchAuthorizationError("CLI consumption receipt identity is invalid")
    token_id = consumption.get("token_id")
    if not isinstance(token_id, str) or not _TOKEN_ID_RE.fullmatch(token_id):
        raise LaunchAuthorizationError("CLI consumption token ID is malformed")
    if consumption_path != gate / "consumptions" / f"{token_id}.json":
        raise LaunchAuthorizationError(
            "CLI consumption receipt is outside its exact gate directory"
        )
    expected = identity.gate_run_identity
    if (
        consumption.get("run_id") != identity.run_id
        or consumption.get("phase") != identity.phase
        or consumption.get("run_identity_sha256") != expected["identity_sha256"]
        or consumption.get("ready_receipt_sha256")
        != _sha256_file(gate / "ready.json", label="CLI ready receipt")
        or ready.get("run_identity_sha256") != expected["identity_sha256"]
    ):
        raise LaunchAuthorizationError(
            "CLI consumption receipt is not bound to the worker run identity"
        )
    policy, policy_sha256 = _load_execution_policy(REPOSITORY_EXECUTION_POLICY_PATH)
    if (
        consumption.get("execution_policy_id") != policy["policy_id"]
        or consumption.get("execution_policy_revision") != policy["revision"]
        or consumption.get("execution_policy_sha256") != policy_sha256
        or consumption.get("execution_mode") != policy["execution_mode"]
    ):
        raise LaunchAuthorizationError(
            "CLI consumption does not bind the current repository execution policy"
        )
    override_id = consumption.get("freeze_override_id")
    override_sha256 = consumption.get("freeze_override_sha256")
    if policy["execution_mode"] == "frozen":
        if (
            not isinstance(override_id, str)
            or not override_id
            or not isinstance(override_sha256, str)
            or not _SHA256_RE.fullmatch(override_sha256)
        ):
            raise LaunchAuthorizationError(
                "CLI frozen-mode consumption lacks its exact freeze override"
            )
    elif override_id is not None or override_sha256 is not None:
        raise LaunchAuthorizationError(
            "CLI open-mode consumption unexpectedly binds a freeze override"
        )
    if consumption.get("consumer_host") != socket.gethostname():
        raise LaunchAuthorizationError(
            "CLI consumption belongs to a different launcher host"
        )
    liveness_path = Path(launcher_liveness_receipt_path).expanduser().absolute()
    _require_non_symlink_directory(
        gate / "cli-launches",
        label="CLI launches directory",
    )
    _require_non_symlink_directory(
        gate / "cli-launches" / token_id,
        label="CLI token launch directory",
    )
    expected_liveness_path = gate / "cli-launches" / token_id / "launcher-liveness.json"
    if liveness_path != expected_liveness_path:
        raise LaunchAuthorizationError(
            "CLI launcher liveness receipt is outside its exact token directory"
        )
    liveness = assert_process_liveness(
        liveness_path,
        expected_run_id=identity.run_id,
        expected_phase=identity.phase,
    )
    consumer_pid = consumption.get("consumer_pid")
    if (
        isinstance(consumer_pid, bool)
        or not isinstance(consumer_pid, int)
        or consumer_pid <= 0
        or liveness.get("pid") != consumer_pid
        or liveness.get("run_identity_sha256") != expected["identity_sha256"]
    ):
        raise LaunchAuthorizationError(
            "CLI worker authorization is not owned by its live outer launcher"
        )
    _assert_process_descends_from(consumer_pid)
    return consumption


__all__ = [
    "CANONICAL_EVALUATION_CONFIG_ROOT",
    "CANONICAL_POLICY_CONFIG_ROOT",
    "CANONICAL_REPRESENTATION_CONFIG_ROOT",
    "CLI_WORKER_AUTHORIZATION_SCHEMA",
    "CLIExecutionAuthorizationIdentity",
    "CLIWorkerAuthorization",
    "CanonicalConfigBinding",
    "PythonExecutableIdentity",
    "REPOSITORY_EXECUTION_POLICY_PATH",
    "assert_loaded_config_matches_binding",
    "assert_canonical_runtime_launch_enabled",
    "assert_legacy_standalone_execution_quarantined",
    "assert_legacy_standalone_mode_quarantined",
    "bind_canonical_config_path",
    "bind_current_python_executable",
    "cli_worker_authorization_environment",
    "consume_cli_execution_authorization",
    "environment_sanitization_parameters",
    "materialize_cli_worker_authorization",
    "sanitized_child_environment",
    "verify_canonical_config_binding",
    "verify_cli_worker_authorization",
    "verify_cli_worker_authorization_from_environment",
    "verify_python_executable_identity",
]
