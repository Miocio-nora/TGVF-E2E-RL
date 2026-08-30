"""Process-local launch plans and final CLI execution boundaries.

This leaf owns the executable command/environment construction used by the
public CLI.  The outer CLI remains responsible for deterministic preflight and
one-time authorization, while this module owns the retained executable
descriptor from the prepared plan through ``os.execve``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    CanonicalConfigBinding,
    PythonExecutableBinding,
    PythonExecutableIdentity,
    assert_fd_exec_supported,
    assert_loaded_config_matches_binding,
    cli_worker_authorization_environment,
    environment_sanitization_parameters,
    sanitized_child_environment,
    verify_canonical_config_binding,
    verify_python_executable_binding,
)


_REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedRepresentationLaunch:
    """Immutable authorization plan plus its process-local Python capability."""

    config: Any
    config_binding: CanonicalConfigBinding
    python_identity: PythonExecutableIdentity
    stop_after_global_step: int | None
    command_prefix: tuple[str, ...]
    child_environment: tuple[tuple[str, str], ...]
    stripped_environment_names: tuple[str, ...]
    python_binding: PythonExecutableBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.python_binding is not None
            and self.python_binding.identity != self.python_identity
        ):
            raise ValueError("prepared Python binding differs from its identity")

    def close_python_binding(self) -> None:
        """Release the process-local executable capability if still owned."""

        if self.python_binding is not None:
            self.python_binding.close()

    @property
    def prepared_identity_sha256(self) -> str:
        record = {
            "schema_version": "tgvf-prepared-representation-launch-v1",
            "canonical_config": self.config_binding.authorization_parameters(),
            "python": self.python_identity.authorization_parameters(),
            "stop_after_global_step": self.stop_after_global_step,
            "command_prefix": list(self.command_prefix),
            "child_environment_sha256": _canonical_json_sha256(
                dict(self.child_environment)
            ),
            "stripped_environment_names": list(self.stripped_environment_names),
        }
        return _canonical_json_sha256(record)

    def authorization_parameters(self) -> dict[str, str]:
        return {
            **self.config_binding.authorization_parameters(),
            **self.python_identity.authorization_parameters(),
            **environment_sanitization_parameters(self.stripped_environment_names),
            "prepared_representation_launch_sha256": self.prepared_identity_sha256,
            "child_environment_sha256": _canonical_json_sha256(
                dict(self.child_environment)
            ),
        }


def _representation_command_prefix(
    config: Any,
    *,
    python_executable: Path,
    stop_after_global_step: int | None,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={config.fsdp2.world_size}",
        "-m",
        "tgvf_rl.cli",
        "run-representation",
        str(config.source_path),
    ]
    if stop_after_global_step is not None:
        command.extend(("--stop-after-global-step", str(stop_after_global_step)))
    return tuple(command)


def _representation_child_environment(
    config: Any,
    *,
    base: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    environment, stripped = sanitized_child_environment(base)
    torchrun_owned = {
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
    }
    inherited_conflicts = tuple(
        sorted(name for name in torchrun_owned if name in environment)
    )
    for name in inherited_conflicts:
        environment.pop(name)
    stripped_names = tuple(sorted(set(stripped).union(inherited_conflicts)))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(gpu_id) for gpu_id in config.fsdp2.physical_gpu_ids
            ),
            "CUBLAS_WORKSPACE_CONFIG": _REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "WORLD_SIZE": str(config.fsdp2.world_size),
        }
    )
    return environment, stripped_names


def _representation_torchrun_command(
    config: Any,
    *,
    python_executable: Path,
    stop_after_global_step: int | None,
    gate_directory: Path,
    worker_authorization: CLIWorkerAuthorization,
) -> tuple[str, ...]:
    executable = python_executable.expanduser().absolute()
    command = list(
        _representation_command_prefix(
            config,
            python_executable=executable,
            stop_after_global_step=stop_after_global_step,
        )
    )
    command.extend(
        (
            "--launcher-python-executable",
            str(executable),
            "--gate-directory",
            str(gate_directory.expanduser().absolute()),
            "--launch-consumption-receipt",
            str(worker_authorization.consumption_receipt_path),
            "--launch-consumption-sha256",
            worker_authorization.consumption_receipt_sha256,
            "--launcher-liveness-receipt",
            str(worker_authorization.launcher_liveness_receipt_path),
        )
    )
    return tuple(command)


def _execute_representation_torchrun(
    prepared: PreparedRepresentationLaunch,
    *,
    launch_identity: CLIExecutionAuthorizationIdentity,
    gate_directory: Path,
    worker_authorization: CLIWorkerAuthorization,
) -> None:
    """Revalidate the prepared plan and replace the process via its bound fd."""

    python_binding = prepared.python_binding
    if python_binding is None:
        raise RuntimeError("prepared representation launch has no bound Python fd")
    try:
        if python_binding.identity != prepared.python_identity:
            raise RuntimeError(
                "prepared representation Python identity differs from its bound fd"
            )
        verify_canonical_config_binding(prepared.config_binding)
        assert_loaded_config_matches_binding(
            prepared.config,
            prepared.config_binding,
            source_sha256_attribute="source_toml_sha256",
        )
        command = _representation_torchrun_command(
            prepared.config,
            python_executable=prepared.python_identity.declared_path,
            stop_after_global_step=prepared.stop_after_global_step,
            gate_directory=gate_directory,
            worker_authorization=worker_authorization,
        )
        if command[: len(prepared.command_prefix)] != prepared.command_prefix:
            raise RuntimeError("representation command changed after authorization")
        environment = dict(prepared.child_environment)
        environment.update(
            cli_worker_authorization_environment(
                launch_identity,
                worker_authorization,
                gate_directory=gate_directory,
            )
        )
        if not command or command[0] != str(prepared.python_identity.declared_path):
            raise RuntimeError("representation argv[0] lost its declared Python path")
        descriptor = verify_python_executable_binding(python_binding)
        assert_fd_exec_supported()
        os.execve(descriptor, command, environment)
    finally:
        python_binding.close()


def _execute_policy_run(
    prepared: Any,
    *,
    launch_identity: CLIExecutionAuthorizationIdentity,
    worker_authorization: CLIWorkerAuthorization,
    gate_directory: Path,
) -> None:
    """Delegate the already-authorized policy plan to its fd-bound executor."""

    from tgvf_rl.policy.launch import execute_policy_e2e_smoke

    execute_policy_e2e_smoke(
        prepared,
        launch_identity=launch_identity,
        worker_authorization=worker_authorization,
        gate_directory=gate_directory,
    )


# Preserve the historical serialization coordinate while ``tgvf_rl.cli``
# exposes this class by one exact direct import from this leaf.
PreparedRepresentationLaunch.__module__ = "tgvf_rl.cli"
