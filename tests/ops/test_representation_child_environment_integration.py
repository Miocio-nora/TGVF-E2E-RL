from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl import cli
from tgvf_rl.ops import cli_launch
from tgvf_rl.ops.child_environment import (
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    POLICY_VERL_DRIVER_PROFILE,
    REPRESENTATION_TORCHRUN_PROFILE,
    RUNTIME_PACKAGE_ROOT,
    TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
    build_child_environment,
)
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    CanonicalConfigBinding,
    PythonExecutableIdentity,
    bind_current_python_executable_for_exec,
)


_CLI_WORKER_ENVIRONMENT = {
    "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": "/gate/consumptions/token.json",
    "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": "c" * 64,
    "TGVF_CLI_EXECUTION_IDENTITY_JSON": "{}",
    "TGVF_CLI_GATE_DIRECTORY": "/gate",
    "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": ("/gate/cli-launches/token/live.json"),
    "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": (
        "tgvf-cli-worker-authorization-environment-v1"
    ),
}


class _ExecIntercept(RuntimeError):
    pass


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_id="REPRESENTATION-CHILD-ENV",
        canonical_config_sha256="a" * 64,
        source_toml_sha256="b" * 64,
        source_path=tmp_path / "representation.toml",
        fsdp2=SimpleNamespace(
            world_size=2,
            physical_gpu_ids=(4, 7),
        ),
    )


def _config_binding(config: SimpleNamespace) -> CanonicalConfigBinding:
    return CanonicalConfigBinding(
        canonical_root=config.source_path.parent,
        source_path=config.source_path,
        resolved_path=config.source_path,
        source_sha256=config.source_toml_sha256,
        byte_length=1,
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o600,
    )


def _synthetic_python_identity() -> PythonExecutableIdentity:
    path = Path("/audited/python")
    return PythonExecutableIdentity(
        declared_path=path,
        resolved_path=path,
        sha256="d" * 64,
        byte_length=100,
        device=3,
        inode=4,
        mode=stat.S_IFREG | 0o755,
    )


def _prepared(
    config: SimpleNamespace,
    environment_binding: object,
    *,
    python_identity: PythonExecutableIdentity | None = None,
    python_binding: object | None = None,
) -> cli.PreparedRepresentationLaunch:
    identity = python_identity or _synthetic_python_identity()
    return cli.PreparedRepresentationLaunch(
        config=config,
        config_binding=_config_binding(config),
        python_identity=identity,
        stop_after_global_step=32,
        command_prefix=cli_launch._representation_command_prefix(
            config,
            python_executable=identity.declared_path,
            stop_after_global_step=32,
        ),
        child_environment_binding=environment_binding,  # type: ignore[arg-type]
        python_binding=python_binding,  # type: ignore[arg-type]
    )


def test_representation_outer_environment_is_allowlisted_and_host_is_audit_only(
    tmp_path: Path,
) -> None:
    hostile_host = {
        "NCCL_DEBUG": "TRACE",
        "PET_LOG_DIR": "/attacker/pet",
        "PYTHON_EXECUTABLE": "/attacker/python",
        "SAFE_VALUE": "must-not-pass",
        "WORLD_SIZE": "99",
    }
    binding = cli_launch._representation_child_environment(
        _config(tmp_path),
        host_environment=hostile_host,
    )
    environment = binding.as_environment()

    assert binding.profile == REPRESENTATION_TORCHRUN_PROFILE
    assert environment == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "4,7",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(RUNTIME_PACKAGE_ROOT),
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TZ": "UTC",
    }
    assert binding.ignored_host_names == ("SAFE_VALUE",)
    assert binding.rejected_host_names == (
        "NCCL_DEBUG",
        "PET_LOG_DIR",
        "PYTHON_EXECUTABLE",
        "WORLD_SIZE",
    )
    assert set(environment).isdisjoint(hostile_host)
    assert "WORLD_SIZE" not in environment


def test_pinned_torchrun_adds_only_the_exact_worker_environment_contract(
    tmp_path: Path,
) -> None:
    worker_script = tmp_path / "capture_worker_environment.py"
    worker_script.write_text(
        """\
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import tgvf_rl
import tgvf_rl.cli as cli_module
import tgvf_rl.ops.child_environment as child_environment_module


output_directory = Path(sys.argv[1])
rank = os.environ["RANK"]
(output_directory / f"worker-{rank}.json").write_text(
    json.dumps(dict(os.environ), sort_keys=True),
    encoding="utf-8",
)
(output_directory / f"origins-{rank}.json").write_text(
    json.dumps(
        {
            "package": tgvf_rl.__file__,
            "cli": cli_module.__file__,
            "child_environment": child_environment_module.__file__,
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    output_directory = tmp_path / "worker-environments"
    output_directory.mkdir()
    base_binding = build_child_environment(
        REPRESENTATION_TORCHRUN_PROFILE,
        owned_environment={
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
        host_environment={},
    )
    assert set(_CLI_WORKER_ENVIRONMENT) == set(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
    parent_environment = base_binding.with_late_overlay(
        _CLI_WORKER_ENVIRONMENT
    ).as_environment()

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker_script),
            str(output_directory),
        ),
        cwd=tmp_path,
        env=parent_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"torchrun stdout:\n{completed.stdout}\ntorchrun stderr:\n{completed.stderr}"
    )

    workers = {
        rank: json.loads(
            (output_directory / f"worker-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    }
    worker_origins = {
        rank: json.loads(
            (output_directory / f"origins-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    }
    runtime_root = RUNTIME_PACKAGE_ROOT.resolve()
    for origins in worker_origins.values():
        assert set(origins) == {"package", "cli", "child_environment"}
        assert all(
            Path(origin).resolve().is_relative_to(runtime_root)
            for origin in origins.values()
        )
    parent_names = set(parent_environment)
    torchrun_names = set(TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES)
    for environment in workers.values():
        assert set(environment).difference(parent_names) == torchrun_names
        assert set(environment).issuperset(parent_names)
        assert {
            name: environment[name] for name in parent_environment
        } == parent_environment
        assert environment["CUDA_VISIBLE_DEVICES"] == ""

    assert {environment["RANK"] for environment in workers.values()} == {"0", "1"}
    assert {environment["LOCAL_RANK"] for environment in workers.values()} == {
        "0",
        "1",
    }
    assert {environment["ROLE_RANK"] for environment in workers.values()} == {
        "0",
        "1",
    }
    for environment in workers.values():
        assert environment["WORLD_SIZE"] == "2"
        assert environment["LOCAL_WORLD_SIZE"] == "2"
        assert environment["ROLE_WORLD_SIZE"] == "2"
        assert environment["GROUP_RANK"] == "0"
        assert environment["GROUP_WORLD_SIZE"] == "1"
        assert environment["ROLE_NAME"] == "default"
        assert environment["TORCHELASTIC_RESTART_COUNT"] == "0"
        assert environment["TORCHELASTIC_MAX_RESTARTS"] == "0"

    shared_dynamic_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    )
    for name in shared_dynamic_names:
        assert workers[0][name]
        assert workers[0][name] == workers[1][name]
    assert workers[0]["MASTER_PORT"].isdecimal()
    assert 0 < int(workers[0]["MASTER_PORT"]) <= 65_535
    error_files = {
        environment["TORCHELASTIC_ERROR_FILE"] for environment in workers.values()
    }
    assert len(error_files) == 2
    assert all(Path(path).is_absolute() for path in error_files)


def test_prepared_launch_requires_exact_unoverlaid_representation_binding(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(TypeError, match="ChildEnvironmentBinding"):
        _prepared(config, ())

    policy_binding = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        host_environment={},
    )
    with pytest.raises(ValueError, match="torchrun environment profile"):
        _prepared(config, policy_binding)

    representation_binding = cli_launch._representation_child_environment(
        config,
        host_environment={},
    )
    already_overlaid = representation_binding.with_late_overlay(
        {"TGVF_CLI_GATE_DIRECTORY": "/gate"}
    )
    with pytest.raises(ValueError, match="already contains a late overlay"):
        _prepared(config, already_overlaid)


@pytest.mark.parametrize(
    "invalid_overlay",
    [
        {
            name: value
            for name, value in _CLI_WORKER_ENVIRONMENT.items()
            if name != "TGVF_CLI_GATE_DIRECTORY"
        },
        {**_CLI_WORKER_ENVIRONMENT, "SAFE_VALUE": "not-delegated"},
        {
            **{
                name: value
                for name, value in _CLI_WORKER_ENVIRONMENT.items()
                if name != "TGVF_CLI_GATE_DIRECTORY"
            },
            "PATH": "/attacker",
        },
    ],
    ids=("missing", "unknown", "baseline-conflict"),
)
def test_final_environment_materializer_rejects_nonexact_cli_overlay(
    tmp_path: Path,
    invalid_overlay: dict[str, str],
) -> None:
    binding = cli_launch._representation_child_environment(
        _config(tmp_path),
        host_environment={},
    )
    with pytest.raises(RuntimeError, match="unexpected field set"):
        cli_launch._materialize_representation_child_environment(
            binding,
            invalid_overlay,
        )


def test_final_exec_materializes_only_outer_binding_and_exact_cli_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    environment_binding = cli_launch._representation_child_environment(
        config,
        host_environment={
            "NCCL_DEBUG": "TRACE",
            "PYTHON_EXECUTABLE": "/attacker/python",
            "SAFE_VALUE": "must-not-pass",
        },
    )
    python_binding = bind_current_python_executable_for_exec(sys.executable)
    prepared = _prepared(
        config,
        environment_binding,
        python_identity=python_binding.identity,
        python_binding=python_binding,
    )
    launch_identity = CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=cli.REPRESENTATION_TRAINING_PHASE,
        command_id=cli._REPRESENTATION_COMMAND_ID,
        run_identity_sha256=config.canonical_config_sha256,
        parameters=prepared.authorization_parameters(),
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=tmp_path / "consumption.json",
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=tmp_path / "liveness.json",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_launch,
        "verify_canonical_config_binding",
        lambda _binding: None,
    )
    monkeypatch.setattr(
        cli_launch,
        "assert_loaded_config_matches_binding",
        lambda *_args, **_kwargs: None,
    )

    def fake_execve(
        descriptor: int,
        argv: tuple[str, ...],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            descriptor=descriptor,
            argv=argv,
            environment=environment,
        )
        raise _ExecIntercept

    monkeypatch.setattr(cli_launch.os, "execve", fake_execve)
    monkeypatch.setattr(cli_launch.os, "supports_fd", frozenset({fake_execve}))

    with pytest.raises(_ExecIntercept):
        cli_launch._execute_representation_torchrun(
            prepared,
            launch_identity=launch_identity,
            gate_directory=tmp_path,
            worker_authorization=worker,
        )

    worker_environment = cli_launch.cli_worker_authorization_environment(
        launch_identity,
        worker,
        gate_directory=tmp_path,
    )
    expected_environment = {
        **environment_binding.as_environment(),
        **worker_environment,
    }
    assert captured["environment"] == expected_environment
    assert set(expected_environment) == set(environment_binding.as_environment()).union(
        worker_environment
    )
    assert "WORLD_SIZE" not in expected_environment
    assert "SAFE_VALUE" not in expected_environment
    assert "NCCL_DEBUG" not in expected_environment
    assert "PYTHON_EXECUTABLE" not in expected_environment
    assert python_binding.closed


def test_consumed_identity_rejects_environment_binding_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first_environment = cli_launch._representation_child_environment(
        config,
        host_environment={"SAFE_FIRST": "ignored"},
    )
    second_environment = cli_launch._representation_child_environment(
        config,
        host_environment={"NCCL_DEBUG": "rejected"},
    )
    assert first_environment.environment_sha256 == second_environment.environment_sha256

    python_binding = bind_current_python_executable_for_exec(sys.executable)
    prepared = _prepared(
        config,
        first_environment,
        python_identity=python_binding.identity,
        python_binding=python_binding,
    )
    original_prepared_sha256 = prepared.prepared_identity_sha256
    launch_identity = CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=cli.REPRESENTATION_TRAINING_PHASE,
        command_id=cli._REPRESENTATION_COMMAND_ID,
        run_identity_sha256=config.canonical_config_sha256,
        parameters=prepared.authorization_parameters(),
    )
    object.__setattr__(prepared, "child_environment_binding", second_environment)
    assert prepared.prepared_identity_sha256 != original_prepared_sha256

    with pytest.raises(
        RuntimeError,
        match="consumed representation authorization differs",
    ):
        cli_launch._execute_representation_torchrun(
            prepared,
            launch_identity=launch_identity,
            gate_directory=tmp_path,
            worker_authorization=CLIWorkerAuthorization(
                consumption_receipt_path=tmp_path / "consumption.json",
                consumption_receipt_sha256="c" * 64,
                launcher_liveness_receipt_path=tmp_path / "liveness.json",
            ),
        )
    assert python_binding.closed


def test_prepared_launch_keeps_legacy_facade_pickle_coordinate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prepared = _prepared(
        config,
        cli_launch._representation_child_environment(
            config,
            host_environment={},
        ),
    )

    assert cli.PreparedRepresentationLaunch is cli_launch.PreparedRepresentationLaunch
    assert prepared.__class__.__module__ == "tgvf_rl.cli"
    assert pickle.loads(pickle.dumps(prepared)) == prepared


def test_authorization_parameters_bind_environment_audit_not_host_values(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = _prepared(
        config,
        cli_launch._representation_child_environment(
            config,
            host_environment={"SAFE_NAME": "first-secret-value"},
        ),
    )
    same_names_new_values = _prepared(
        config,
        cli_launch._representation_child_environment(
            config,
            host_environment={"SAFE_NAME": "second-secret-value"},
        ),
    )
    changed_names = replace(
        first,
        child_environment_binding=cli_launch._representation_child_environment(
            config,
            host_environment={"NCCL_DEBUG": "TRACE"},
        ),
    )

    assert (
        first.prepared_identity_sha256 == same_names_new_values.prepared_identity_sha256
    )
    assert first.prepared_identity_sha256 != changed_names.prepared_identity_sha256
    assert "first-secret-value" not in repr(first.authorization_parameters())
    assert "second-secret-value" not in repr(
        same_names_new_values.authorization_parameters()
    )


def test_final_representation_boundary_has_no_host_copy_or_mapping_update() -> None:
    source = Path(cli_launch.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    protected_names = {
        "_execute_representation_torchrun",
        "_materialize_representation_child_environment",
    }
    protected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in protected_names
    ]
    assert {node.name for node in protected} == protected_names

    for function in protected:
        for node in ast.walk(function):
            assert not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr == "environ"
            )
            assert not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
            )
