from __future__ import annotations

import ast
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from tgvf_rl import cli
from tgvf_rl.ops import cli_authorization as facade
from tgvf_rl.ops import cli_authorization_identity as identity_module
from tgvf_rl.ops import cli_launch
from tgvf_rl.ops.child_environment import (
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
    POLICY_VERL_DRIVER_PROFILE,
    REPRESENTATION_TORCHRUN_PROFILE,
    build_child_environment,
)
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    CanonicalConfigBinding,
    PythonExecutableBinding,
    PythonExecutableIdentity,
    bind_current_python_executable_for_exec,
    verify_python_executable_binding,
)
from tgvf_rl.ops.launch_gate import LaunchAuthorizationError
from tgvf_rl.policy import launch as policy_launch
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    retain_regular_file_absolute_nofollow,
)
from tests.runtime_locator_support import verified_runtime_locator_evidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _ExecIntercept(RuntimeError):
    pass


def _child_environment_binding(profile: str):
    return build_child_environment(profile, host_environment={})


def _cli_worker_environment() -> dict[str, str]:
    values = {
        "TGVF_CLI_CONSUMPTION_RECEIPT_PATH": "/gate/consumption.json",
        "TGVF_CLI_CONSUMPTION_RECEIPT_SHA256": "c" * 64,
        "TGVF_CLI_EXECUTION_IDENTITY_JSON": "{}",
        "TGVF_CLI_GATE_DIRECTORY": "/gate",
        "TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH": "/gate/liveness.json",
        "TGVF_CLI_WORKER_AUTHORIZATION_SCHEMA": (
            "tgvf-cli-worker-authorization-environment-v1"
        ),
    }
    assert set(values) == set(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
    return values


def _representation_launch_identity(
    prepared: cli.PreparedRepresentationLaunch,
) -> CLIExecutionAuthorizationIdentity:
    return CLIExecutionAuthorizationIdentity.create(
        run_id=prepared.config.run_id,
        phase="representation_training",
        command_id="fd-exec-test",
        run_identity_sha256=prepared.config.canonical_config_sha256,
        parameters=prepared.authorization_parameters(),
    )


def _copied_executable(tmp_path: Path, name: str = "python-copy") -> Path:
    path = tmp_path / name
    shutil.copy2(Path(sys.executable).resolve(), path)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _binding_for_path(path: Path) -> PythonExecutableBinding:
    retained = retain_regular_file_absolute_nofollow(path)
    try:
        snapshot = retained.snapshot()
        observed = snapshot.after
        identity = PythonExecutableIdentity(
            declared_path=path,
            resolved_path=path,
            sha256=sha256(snapshot.payload).hexdigest(),
            byte_length=observed.st_size,
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=observed.st_mode,
        )
        binding = PythonExecutableBinding(identity, retained)
        retained = None  # type: ignore[assignment]
        return binding
    finally:
        if retained is not None:
            retained.close()


def _config_binding(path: Path) -> CanonicalConfigBinding:
    return CanonicalConfigBinding(
        canonical_root=path.parent,
        source_path=path,
        resolved_path=path,
        source_sha256="a" * 64,
        byte_length=1,
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o600,
    )


def _install_fake_fd_exec(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    captured: dict[str, object],
) -> None:
    def fake_execve(
        descriptor: int,
        argv: tuple[str, ...],
        environment: dict[str, str],
    ) -> None:
        assert isinstance(descriptor, int) and not isinstance(descriptor, bool)
        captured.update(
            descriptor=descriptor,
            inode=os.fstat(descriptor).st_ino,
            argv=argv,
            environment=environment,
        )
        raise _ExecIntercept("stop before process replacement")

    module_os = getattr(module, "os")
    monkeypatch.setattr(module_os, "execve", fake_execve)
    monkeypatch.setattr(module_os, "supports_fd", frozenset({fake_execve}))


def test_retained_absolute_descriptor_rejects_nonregular_and_ancestor_symlink(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(SecureFileReadError, match="regular file"):
        retain_regular_file_absolute_nofollow(directory)

    real = tmp_path / "real"
    real.mkdir()
    executable = _copied_executable(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(SecureFileReadError, match="symlink"):
        retain_regular_file_absolute_nofollow(alias / executable.name)


def test_representation_exec_uses_retained_inode_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = _copied_executable(tmp_path, "bound-python")
    binding = _binding_for_path(declared)
    original_inode = binding.identity.inode
    descriptor = binding.fileno()
    replacement = _copied_executable(tmp_path, "replacement-python")
    os.replace(replacement, declared)
    assert declared.stat().st_ino != original_inode

    config_path = tmp_path / "run.toml"
    config = SimpleNamespace(
        run_id="REPRESENTATION-FD-EXEC",
        canonical_config_sha256="a" * 64,
        source_path=config_path,
        source_toml_sha256="b" * 64,
    )
    prepared = cli.PreparedRepresentationLaunch(
        config=config,
        config_binding=_config_binding(config_path),
        python_identity=binding.identity,
        stop_after_global_step=32,
        command_prefix=(str(declared),),
        child_environment_binding=_child_environment_binding(
            REPRESENTATION_TORCHRUN_PROFILE
        ),
        python_binding=binding,
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=tmp_path / "consumption.json",
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=tmp_path / "liveness.json",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_launch, "verify_canonical_config_binding", lambda _value: None
    )
    monkeypatch.setattr(
        cli_launch, "assert_loaded_config_matches_binding", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        cli_launch,
        "_representation_torchrun_command",
        lambda *_a, **_k: (str(declared), "-V"),
    )
    monkeypatch.setattr(
        cli_launch,
        "cli_worker_authorization_environment",
        lambda *_a, **_k: _cli_worker_environment(),
    )
    _install_fake_fd_exec(monkeypatch, cli_launch, captured)

    with pytest.raises(_ExecIntercept):
        cli._execute_representation_torchrun(
            prepared,
            launch_identity=_representation_launch_identity(prepared),
            gate_directory=tmp_path,
            worker_authorization=worker,
        )

    assert captured["descriptor"] == descriptor
    assert captured["inode"] == original_inode
    assert captured["argv"][0] == str(declared)  # type: ignore[index]
    assert binding.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_representation_exec_rejects_identity_drift_before_final_verification(
    tmp_path: Path,
) -> None:
    declared = _copied_executable(tmp_path)
    binding = _binding_for_path(declared)
    descriptor = binding.fileno()
    config_path = tmp_path / "run.toml"
    prepared = cli.PreparedRepresentationLaunch(
        config=SimpleNamespace(
            source_path=config_path,
            source_toml_sha256="b" * 64,
        ),
        config_binding=_config_binding(config_path),
        python_identity=binding.identity,
        stop_after_global_step=32,
        command_prefix=(str(declared),),
        child_environment_binding=_child_environment_binding(
            REPRESENTATION_TORCHRUN_PROFILE
        ),
        python_binding=binding,
    )
    object.__setattr__(
        prepared,
        "python_identity",
        replace(binding.identity, sha256="f" * 64),
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=tmp_path / "consumption.json",
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=tmp_path / "liveness.json",
    )

    with pytest.raises(RuntimeError, match="differs from its bound fd"):
        cli._execute_representation_torchrun(
            prepared,
            launch_identity=object(),  # type: ignore[arg-type]
            gate_directory=tmp_path,
            worker_authorization=worker,
        )

    assert binding.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_policy_exec_uses_integer_bound_fd_and_closes_on_exec_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = _copied_executable(tmp_path)
    binding = _binding_for_path(declared)
    descriptor = binding.fileno()
    command = (
        str(declared),
        "-m",
        "tgvf_rl.framework.verl.policy_main",
    )
    receipt = SimpleNamespace(receipt_sha256="d" * 64)
    plan = SimpleNamespace(
        assert_launch_ready=lambda: None,
        preflight_live_prerequisites=lambda: receipt,
        command=lambda _python: command,
        as_record=lambda: {"launch_ready": True},
    )
    config = SimpleNamespace(
        run_id="POLICY-FD-EXEC",
        identity_sha256="a" * 64,
        source_sha256="b" * 64,
        output=SimpleNamespace(root=tmp_path),
    )
    compile_prerequisites = SimpleNamespace(
        identity_sha256="e" * 64,
        manifest_source_sha256="f" * 64,
    )
    compile_authorization = SimpleNamespace(authorization_parameters=lambda: {})
    runtime_locator_evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    prepared = policy_launch.PreparedPolicyLaunch(
        config=config,
        plan=plan,
        compile_prerequisites=compile_prerequisites,
        compile_receipt=receipt,
        compile_authorization=compile_authorization,
        python_identity=binding.identity,
        command=command,
        child_environment_binding=_child_environment_binding(
            POLICY_VERL_DRIVER_PROFILE
        ),
        repository_root=tmp_path,
        runtime_locator_evidence=runtime_locator_evidence,
        python_binding=binding,
    )
    identity = CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase="policy_training",
        command_id="fd-exec-test",
        run_identity_sha256=config.identity_sha256,
        parameters=prepared.authorization_parameters(),
    )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=tmp_path / "consumption.json",
        consumption_receipt_sha256="c" * 64,
        launcher_liveness_receipt_path=tmp_path / "liveness.json",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        policy_launch,
        "assert_policy_execution_identity",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        policy_launch,
        "materialize_policy_compile_prerequisite_receipt",
        lambda *_a, **_k: tmp_path / "receipt.json",
    )
    monkeypatch.setattr(
        policy_launch,
        "cli_worker_authorization_environment",
        lambda *_a, **_k: _cli_worker_environment(),
    )
    _install_fake_fd_exec(monkeypatch, policy_launch, captured)

    with pytest.raises(_ExecIntercept):
        policy_launch.execute_policy_e2e_smoke(
            prepared,
            launch_identity=identity,
            worker_authorization=worker,
            gate_directory=tmp_path,
        )

    assert captured["descriptor"] == descriptor
    assert captured["argv"][0] == str(declared)  # type: ignore[index]
    captured_environment = captured["environment"]
    assert isinstance(captured_environment, dict)
    assert set(captured_environment) == {
        *_child_environment_binding(POLICY_VERL_DRIVER_PROFILE).as_environment(),
        *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
        *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
    }
    assert "OPENROUTER_API_KEY" not in captured_environment
    assert binding.closed
    runtime_locator_evidence.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_fd_exec_support_is_required_before_python_descriptor_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def unexpected_open(_path: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("descriptor must not open on an unsupported platform")

    monkeypatch.setattr(identity_module.os, "supports_fd", frozenset())
    monkeypatch.setattr(
        identity_module,
        "retain_regular_file_absolute_nofollow",
        unexpected_open,
    )
    with pytest.raises(LaunchAuthorizationError, match="file-descriptor os.execve"):
        bind_current_python_executable_for_exec(sys.executable)
    assert not opened


def test_preflight_failure_closes_the_bound_python_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tgvf_rl.representation.training import runner

    binding = _binding_for_path(_copied_executable(tmp_path))
    descriptor = binding.fileno()
    config_path = tmp_path / "run.toml"
    config = SimpleNamespace(source_path=config_path, source_toml_sha256="b" * 64)
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    monkeypatch.setattr(
        cli, "assert_loaded_config_matches_binding", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        cli,
        "bind_current_python_executable_for_exec",
        lambda _path: binding,
    )
    monkeypatch.setattr(
        runner,
        "_validate_invocation_stop",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        cli._preflight_representation_launch(
            config,
            config_binding=_config_binding(config_path),
            python_executable=binding.identity.declared_path,
            stop_after_global_step=32,
        )

    assert binding.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_authorization_failure_closes_prepared_python_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding_for_path(_copied_executable(tmp_path))
    descriptor = binding.fileno()
    config_path = tmp_path / "run.toml"
    config_binding = _config_binding(config_path)
    config = SimpleNamespace(
        run_id="REPRESENTATION-AUTH-FAIL",
        canonical_config_sha256="a" * 64,
        source_toml_sha256="b" * 64,
        source_path=config_path,
        fsdp2=SimpleNamespace(world_size=1),
    )
    prepared = cli.PreparedRepresentationLaunch(
        config=config,
        config_binding=config_binding,
        python_identity=binding.identity,
        stop_after_global_step=32,
        command_prefix=(str(binding.identity.declared_path),),
        child_environment_binding=_child_environment_binding(
            REPRESENTATION_TORCHRUN_PROFILE
        ),
        python_binding=binding,
    )
    monkeypatch.setattr(cli, "assert_canonical_runtime_launch_enabled", lambda: None)
    monkeypatch.setattr(
        cli,
        "bind_canonical_config_path",
        lambda *_a, **_k: config_binding,
    )
    monkeypatch.setattr(cli, "load_representation_training_config", lambda _p: config)
    monkeypatch.setattr(
        cli,
        "_preflight_representation_launch",
        lambda *_a, **_k: prepared,
    )
    monkeypatch.setattr(
        cli,
        "_consume_command_authorization",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("authorization failed")),
    )

    assert (
        cli.main(
            [
                "launch-representation",
                str(config_path),
                "--stop-after-global-step",
                "32",
                "--python",
                str(binding.identity.declared_path),
                "--gate-directory",
                str(tmp_path / "gate"),
                "--authorization-token",
                str(tmp_path / "token.json"),
            ]
        )
        == 2
    )
    assert binding.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_identity_and_authorization_are_fd_free_and_compatibly_picklable(
    tmp_path: Path,
) -> None:
    binding = _binding_for_path(_copied_executable(tmp_path))
    try:
        identity = binding.identity
        record = asdict(identity)
        assert set(record) == {
            "declared_path",
            "resolved_path",
            "sha256",
            "byte_length",
            "device",
            "inode",
            "mode",
        }
        json.dumps(identity.authorization_parameters())
        assert identity.__class__.__module__ == "tgvf_rl.ops.cli_authorization"
        assert identity.__class__.__qualname__ == "PythonExecutableIdentity"
        assert pickle.loads(pickle.dumps(identity)) == identity
        assert (
            facade.PythonExecutableIdentity is identity_module.PythonExecutableIdentity
        )
        assert get_type_hints(facade.bind_current_python_executable) == get_type_hints(
            identity_module.bind_current_python_executable
        )
        execution_identity = CLIExecutionAuthorizationIdentity.create(
            run_id="COMPATIBILITY",
            phase="test",
            command_id="test:compatibility:v1",
            run_identity_sha256="a" * 64,
            parameters=identity.authorization_parameters(),
        )
        assert execution_identity.__class__.__module__ == (
            "tgvf_rl.ops.cli_authorization"
        )
        assert pickle.loads(pickle.dumps(execution_identity)) == execution_identity
        assert "fd" not in execution_identity.as_record()
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(binding)
    finally:
        binding.close()


def test_representation_plan_keeps_legacy_cli_coordinate_after_leaf_split(
    tmp_path: Path,
) -> None:
    executable = _copied_executable(tmp_path)
    binding = _binding_for_path(executable)
    try:
        config_path = tmp_path / "run.toml"
        prepared = cli.PreparedRepresentationLaunch(
            config=SimpleNamespace(
                source_path=config_path,
                source_toml_sha256="b" * 64,
            ),
            config_binding=_config_binding(config_path),
            python_identity=binding.identity,
            stop_after_global_step=32,
            command_prefix=(str(executable),),
            child_environment_binding=_child_environment_binding(
                REPRESENTATION_TORCHRUN_PROFILE
            ),
        )
        assert (
            cli.PreparedRepresentationLaunch is cli_launch.PreparedRepresentationLaunch
        )
        assert prepared.__class__.__module__ == "tgvf_rl.cli"
        assert pickle.loads(pickle.dumps(prepared)) == prepared
    finally:
        binding.close()


@pytest.mark.parametrize("order", ["leaf-first", "facade-first"])
def test_identity_split_is_import_order_independent(order: str) -> None:
    imports = (
        "import tgvf_rl.ops.cli_authorization_identity as leaf\n"
        "import tgvf_rl.ops.cli_authorization as facade\n"
        if order == "leaf-first"
        else "import tgvf_rl.ops.cli_authorization as facade\n"
        "import tgvf_rl.ops.cli_authorization_identity as leaf\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            imports
            + "assert facade.PythonExecutableIdentity is leaf.PythonExecutableIdentity\n"
            + "assert facade.CLIExecutionAuthorizationIdentity is "
            "leaf.CLIExecutionAuthorizationIdentity\n",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr


def test_production_exec_boundaries_have_no_path_or_proc_fd_fallback() -> None:
    for relative_path in (
        "src/tgvf_rl/ops/cli_launch.py",
        "src/tgvf_rl/policy/launch.py",
    ):
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "/proc/self/fd" not in source
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "execve"
        ]
        assert len(calls) == 1
        assert isinstance(calls[0].args[0], ast.Name)
        assert calls[0].args[0].id == "descriptor"


def test_verify_binding_never_reopens_the_declared_path(tmp_path: Path) -> None:
    declared = _copied_executable(tmp_path, "original")
    binding = _binding_for_path(declared)
    replacement = _copied_executable(tmp_path, "replacement")
    os.replace(replacement, declared)
    try:
        assert verify_python_executable_binding(binding) == binding.fileno()
        assert os.fstat(binding.fileno()).st_ino == binding.identity.inode
        assert declared.stat().st_ino != binding.identity.inode
    finally:
        binding.close()
