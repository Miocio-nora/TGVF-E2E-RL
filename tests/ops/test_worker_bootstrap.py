from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from tgvf_rl.ops.cli_authorization import (
    REPOSITORY_EXECUTION_POLICY_PATH,
    bind_current_python_executable,
    cli_worker_authorization_environment,
    consume_cli_execution_authorization,
    materialize_cli_worker_authorization,
)
from tgvf_rl.ops.child_environment import (
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
    POLICY_VERL_DRIVER_PROFILE,
    REPRESENTATION_TORCHRUN_PROFILE,
    TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES,
    build_child_environment,
)
from tgvf_rl.ops.cli_authorization_identity import (
    CLIExecutionAuthorizationIdentity,
)
from tgvf_rl.ops.launch_gate import (
    issue_freeze_override,
    issue_launch_authorization,
    materialize_ready_receipt,
)
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    REPRESENTATION_LAUNCHER_ROLE,
    REPRESENTATION_MEMBER_ROLE,
    WorkerStartupEnvelope,
    WorkerStartupIdentity,
)
from tgvf_rl.worker_bootstrap import (
    WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE,
    WORKER_BOOTSTRAP_INSPECTION_SCHEMA,
    WORKER_BOOTSTRAP_MODES,
    WORKER_BOOTSTRAP_MODULE,
    WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE,
    WorkerBootstrapInspection,
)
from tgvf_rl import worker_bootstrap as bootstrap_module


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_POLICY_TARGET = "tgvf_rl.framework.verl.policy_main:main"
_REPRESENTATION_LAUNCHER_TARGET = "tgvf_rl.ops.representation_launcher:main"
_REPRESENTATION_MEMBER_TARGET = (
    "tgvf_rl.representation.training.runner:run_representation_training"
)
_FORBIDDEN_ROOTS = (
    "hydra",
    "numpy",
    "omegaconf",
    "ray",
    "site",
    "sysconfig",
    "torch",
    "transformers",
    "verl",
    "vllm",
)


def _bootstrap_command(mode: str, *arguments: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-B",
        "-P",
        "-S",
        "-m",
        WORKER_BOOTSTRAP_MODULE,
        mode,
        *arguments,
    )


def _identity(
    *,
    role: str,
    command: tuple[str, ...],
    target: str,
    runtime_digest: str,
    dependency_digest: str,
) -> WorkerStartupIdentity:
    return WorkerStartupIdentity(
        role=role,
        command=command,
        target=target,
        runtime_package_sha256=runtime_digest,
        dependency_roots_sha256=dependency_digest,
    )


def _envelope(mode: str) -> WorkerStartupEnvelope:
    if mode == "run-policy":
        return WorkerStartupEnvelope(
            entry_role=POLICY_DRIVER_ROLE,
            identities=(
                _identity(
                    role=POLICY_DRIVER_ROLE,
                    command=_bootstrap_command(mode, "++trainer.total_epochs=1"),
                    target=_POLICY_TARGET,
                    runtime_digest="a" * 64,
                    dependency_digest="b" * 64,
                ),
            ),
        )
    if mode == "run-representation-member":
        return WorkerStartupEnvelope(
            entry_role=REPRESENTATION_LAUNCHER_ROLE,
            identities=(
                _identity(
                    role=REPRESENTATION_LAUNCHER_ROLE,
                    command=_bootstrap_command(
                        "run-representation-launcher",
                        "/runtime/config.toml",
                    ),
                    target=_REPRESENTATION_LAUNCHER_TARGET,
                    runtime_digest="c" * 64,
                    dependency_digest="d" * 64,
                ),
                _identity(
                    role=REPRESENTATION_MEMBER_ROLE,
                    command=_bootstrap_command(mode, "/runtime/config.toml"),
                    target=_REPRESENTATION_MEMBER_TARGET,
                    runtime_digest="e" * 64,
                    dependency_digest="f" * 64,
                ),
            ),
        )
    raise AssertionError(f"unexpected test mode {mode}")


def _execution_identity(
    mode: str,
    envelope: WorkerStartupEnvelope,
    *,
    parameter_updates: dict[str, str] | None = None,
    parameter_removals: tuple[str, ...] = (),
) -> CLIExecutionAuthorizationIdentity:
    phase, command_id = (
        ("policy_training", "tgvf-rl:run-policy:v4")
        if mode == "run-policy"
        else (
            "representation_training",
            "tgvf-rl:launch-representation:v2",
        )
    )
    child_environment = build_child_environment(
        POLICY_VERL_DRIVER_PROFILE
        if mode == "run-policy"
        else REPRESENTATION_TORCHRUN_PROFILE,
        host_environment={},
    )
    parameters = {
        **envelope.authorization_parameters(),
        **child_environment.authorization_parameters(),
        **bind_current_python_executable(sys.executable).authorization_parameters(),
    }
    parameters.update(parameter_updates or {})
    for name in parameter_removals:
        parameters.pop(name, None)
    return CLIExecutionAuthorizationIdentity.create(
        run_id="WORKER-BOOTSTRAP-INSPECTION",
        phase=phase,
        command_id=command_id,
        run_identity_sha256="9" * 64,
        parameters=parameters,
    )


def _authorized_environment(
    tmp_path: Path,
    identity: CLIExecutionAuthorizationIdentity,
) -> dict[str, str]:
    evidence = tmp_path / "validated-input.json"
    evidence.write_text('{"status":"validated"}\n', encoding="utf-8")
    gate = tmp_path / "gate"
    materialize_ready_receipt(
        gate,
        run_identity=identity.gate_run_identity,
        evidence_paths={"validated_input": evidence},
    )
    token_path, _ = issue_launch_authorization(
        gate,
        ttl_seconds=300,
        authorized_by="worker-bootstrap-test",
    )
    override_path, _ = issue_freeze_override(
        gate,
        REPOSITORY_EXECUTION_POLICY_PATH,
        reason="dependency-light worker bootstrap inspection test",
        ttl_seconds=300,
        authorized_by="worker-bootstrap-test",
    )
    consumption = consume_cli_execution_authorization(
        identity,
        gate_directory=gate,
        authorization_token_path=token_path,
        freeze_override_path=override_path,
    )
    worker = materialize_cli_worker_authorization(
        identity,
        consumption,
        gate_directory=gate,
    )
    return cli_worker_authorization_environment(
        identity,
        worker,
        gate_directory=gate,
    )


def _isolated_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "PYTHONPATH": str(_SOURCE_ROOT),
        "PYTHONUTF8": "1",
        **(extra or {}),
    }


def _materialized_role_environment(
    mode: str,
    worker_environment: dict[str, str],
) -> dict[str, str]:
    profile = (
        POLICY_VERL_DRIVER_PROFILE
        if mode == "run-policy"
        else REPRESENTATION_TORCHRUN_PROFILE
    )
    binding = build_child_environment(profile, host_environment={})
    if mode == "run-policy":
        role_overlay = {
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH": (
                "/runtime/compile-receipt.json"
            ),
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256": "7" * 64,
        }
        assert set(role_overlay) == set(
            POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES
        )
    else:
        role_overlay = {
            "GROUP_RANK": "0",
            "GROUP_WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29400",
            "RANK": "0",
            "ROLE_NAME": "default",
            "ROLE_RANK": "0",
            "ROLE_WORLD_SIZE": "2",
            "TORCHELASTIC_ERROR_FILE": "/tmp/torchelastic/rank-0/error.json",
            "TORCHELASTIC_MAX_RESTARTS": "0",
            "TORCHELASTIC_RESTART_COUNT": "0",
            "TORCHELASTIC_RUN_ID": "worker-bootstrap-inspection",
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "WORLD_SIZE": "2",
        }
        assert set(role_overlay) == set(TORCHRUN_WORKER_LATE_ENVIRONMENT_NAMES)
    assert set(worker_environment) == set(CLI_WORKER_LATE_ENVIRONMENT_NAMES)
    return binding.with_late_overlay(
        {**worker_environment, **role_overlay}
    ).as_environment()


def _run_valid_main(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    envelope = _envelope(mode)
    identity = _execution_identity(mode, envelope)
    worker_environment = _materialized_role_environment(
        mode,
        _authorized_environment(tmp_path, identity),
    )
    command = envelope.identity_for_role(
        POLICY_DRIVER_ROLE
        if mode == "run-policy"
        else REPRESENTATION_MEMBER_ROLE
    ).command
    return subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=worker_environment,
    )


def _run_changed_contract_main(
    tmp_path: Path,
    mode: str,
    *,
    parameter_updates: dict[str, str] | None = None,
    parameter_removals: tuple[str, ...] = (),
    environment_updates: dict[str, str] | None = None,
    environment_removals: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    envelope = _envelope(mode)
    identity = _execution_identity(
        mode,
        envelope,
        parameter_updates=parameter_updates,
        parameter_removals=parameter_removals,
    )
    environment = _materialized_role_environment(
        mode,
        _authorized_environment(tmp_path, identity),
    )
    environment.update(environment_updates or {})
    for name in environment_removals:
        environment.pop(name, None)
    selected_role = (
        POLICY_DRIVER_ROLE
        if mode == "run-policy"
        else REPRESENTATION_MEMBER_ROLE
    )
    return subprocess.run(
        envelope.identity_for_role(selected_role).command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_inspection_record_is_explicitly_non_authoritative() -> None:
    inspection = WorkerBootstrapInspection(
        mode="run-policy",
        entry_role=POLICY_DRIVER_ROLE,
        selected_role=POLICY_DRIVER_ROLE,
        command_sha256="1" * 64,
        target_arguments_sha256="2" * 64,
        target=_POLICY_TARGET,
        startup_envelope_sha256="3" * 64,
        startup_identity_sha256="4" * 64,
        declared_runtime_package_sha256="5" * 64,
        declared_dependency_roots_sha256="6" * 64,
        blockers=("immutable-runtime-code-package-missing",),
    )

    assert WORKER_BOOTSTRAP_MODES == (
        "run-policy",
        "run-representation-member",
    )
    assert WORKER_BOOTSTRAP_INSPECTION_SCHEMA == (
        "tgvf-worker-bootstrap-inspection-v2"
    )
    assert inspection.as_record() == {
        "schema_version": WORKER_BOOTSTRAP_INSPECTION_SCHEMA,
        "authorization_scope": WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE,
        "record_trust": "ordinary-caller-constructible-diagnostic",
        "mode": "run-policy",
        "entry_role": POLICY_DRIVER_ROLE,
        "selected_role": POLICY_DRIVER_ROLE,
        "command_sha256": "1" * 64,
        "target_arguments_sha256": "2" * 64,
        "target": _POLICY_TARGET,
        "startup_envelope_sha256": "3" * 64,
        "startup_identity_sha256": "4" * 64,
        "declared_runtime_package_sha256": "5" * 64,
        "declared_dependency_roots_sha256": "6" * 64,
        "outer_cli_receipt_checked_by_existing_verifier": True,
        "outer_process_relation": "existing-descendant-check-only",
        "cli_environment_namespace_exact": True,
        "current_python_executable_identity_checked": True,
        "current_python_descriptor_retained": False,
        "role_child_environment_base_identity_checked": True,
        "role_child_environment_late_field_inventory_checked": True,
        "role_child_environment_late_values_checked": False,
        "heavy_import_roots_absent": True,
        "interpreter_flags_accepted": True,
        "default_import_machinery_shape_checked": True,
        "single_threaded": True,
        "trace_profile_absent": True,
        "runtime_origin_verified": False,
        "immutable_runtime_verified": False,
        "target_imported": False,
        "verified_worker_startup_minted": False,
        "dispatch_authorized": False,
        "blockers": ["immutable-runtime-code-package-missing"],
    }
    assert not hasattr(inspection, "identity")


def test_inspection_record_is_frozen_but_not_execution_evidence() -> None:
    inspection = WorkerBootstrapInspection(
        mode="run-policy",
        entry_role=POLICY_DRIVER_ROLE,
        selected_role=POLICY_DRIVER_ROLE,
        command_sha256="1" * 64,
        target_arguments_sha256="2" * 64,
        target=_POLICY_TARGET,
        startup_envelope_sha256="3" * 64,
        startup_identity_sha256="4" * 64,
        declared_runtime_package_sha256="5" * 64,
        declared_dependency_roots_sha256="6" * 64,
        blockers=("blocked",),
    )

    with pytest.raises(AttributeError):
        inspection.mode = "run-representation-member"  # type: ignore[misc]
    assert copy.copy(inspection) == inspection
    assert pickle.loads(pickle.dumps(inspection)) == inspection


@pytest.mark.parametrize(
    "command",
    [
        list(_bootstrap_command("run-policy")),
        (sys.executable, "-P", "-B", "-S", "-m", WORKER_BOOTSTRAP_MODULE, "run-policy"),
        (sys.executable, "-B", "-P", "-m", WORKER_BOOTSTRAP_MODULE, "run-policy"),
        (sys.executable, "-B", "-P", "-S", "-m", "tgvf_rl.cli", "run-policy"),
        (sys.executable, "-B", "-P", "-S", "-m", WORKER_BOOTSTRAP_MODULE, "policy"),
        ("/different/python", "-B", "-P", "-S", "-m", WORKER_BOOTSTRAP_MODULE, "run-policy"),
        (sys.executable, "-B", "-P", "-S", "-m", WORKER_BOOTSTRAP_MODULE, "run-policy\n"),
    ],
)
def test_process_command_requires_one_exact_firebreak_prefix(command: object) -> None:
    with pytest.raises(RuntimeError):
        bootstrap_module._require_process_command(  # noqa: SLF001
            command,
            mode="run-policy",
        )


def test_bootstrap_import_is_dependency_light_in_isolated_python() -> None:
    script = f"""
import sys
import tgvf_rl.worker_bootstrap as module
for root in {repr(_FORBIDDEN_ROOTS)}:
    assert not any(name == root or name.startswith(root + '.') for name in sys.modules), root
assert not hasattr(module, '_mint_verified_worker_startup_for_bootstrap')
assert module.WORKER_BOOTSTRAP_AUTHORIZATION_SCOPE == 'inspection-only'
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "flags",
    [
        ("-P", "-S"),
        ("-B", "-S"),
        ("-B", "-P"),
    ],
)
def test_main_refuses_when_any_interpreter_firebreak_flag_is_missing(
    flags: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            *flags,
            "-m",
            WORKER_BOOTSTRAP_MODULE,
            "run-policy",
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_preloaded_heavy_module_refuses_before_project_authorization_import() -> None:
    script = """
import sys
import types
sys.modules['torch'] = types.ModuleType('torch')
import tgvf_rl.worker_bootstrap as module
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-S",
            "-c",
            script,
            "run-policy",
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_preloaded_project_verifier_refuses_before_receipt_check() -> None:
    script = """
import sys
import types
sys.modules['tgvf_rl.ops'] = types.ModuleType('tgvf_rl.ops')
import tgvf_rl.worker_bootstrap as module
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-S",
            "-c",
            script,
            "run-policy",
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_custom_meta_path_finder_refuses_before_project_import() -> None:
    script = """
import sys
import tgvf_rl.worker_bootstrap as module
class Finder:
    def find_spec(self, fullname, path=None, target=None):
        return None
sys.meta_path.insert(0, Finder())
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_custom_path_hook_or_importer_cache_refuses_before_project_import() -> None:
    script = """
import sys
import tgvf_rl.worker_bootstrap as module
sys.path_hooks.append(lambda path: (_ for _ in ()).throw(ImportError(path)))
sys.path_importer_cache['/attacker'] = object()
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_changed_cached_file_finder_refuses_before_project_import() -> None:
    script = """
import sys
import tgvf_rl.worker_bootstrap as module
finder = next(
    value
    for value in sys.path_importer_cache.values()
    if value is not None and type(value).__name__ == 'FileFinder'
)
finder._loaders = tuple(finder._loaders) + (('.injected', object),)
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_delayed_project_import_closure_rejects_target_module() -> None:
    script = """
import sys
import types
import tgvf_rl.worker_bootstrap as module
import tgvf_rl.ops.cli_authorization
import tgvf_rl.ops.child_environment
import tgvf_rl.ops.worker_startup
sys.modules['tgvf_rl.framework'] = types.ModuleType('tgvf_rl.framework')
try:
    module._require_interpreter_firebreak(allow_project_verifiers=True)
except module.WorkerBootstrapInspectionError:
    pass
else:
    raise AssertionError('changed delayed project import closure was accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("hook", ["trace", "profile"])
def test_active_trace_or_profile_hook_refuses_before_receipt_check(hook: str) -> None:
    setter = "sys.settrace" if hook == "trace" else "sys.setprofile"
    script = f"""
import sys
import tgvf_rl.worker_bootstrap as module
{setter}(lambda *args: None)
code = module.main()
{setter}(None)
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_second_thread_refuses_before_receipt_check() -> None:
    script = """
import sys
import threading
import tgvf_rl.worker_bootstrap as module
release = threading.Event()
thread = threading.Thread(target=release.wait)
thread.start()
try:
    code = module.main()
finally:
    release.set()
    thread.join()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_unregistered_native_thread_refuses_before_receipt_check() -> None:
    script = """
import _thread
import sys
import tgvf_rl.worker_bootstrap as module
release = _thread.allocate_lock()
started = _thread.allocate_lock()
done = _thread.allocate_lock()
for lock in (release, started, done):
    lock.acquire()
def worker():
    started.release()
    release.acquire()
    done.release()
_thread.start_new_thread(worker, ())
started.acquire()
try:
    code = module.main()
finally:
    release.release()
    done.acquire()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script, "run-policy"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_main_rejects_sys_argv_drift_from_original_process_command() -> None:
    script = """
import sys
import tgvf_rl.worker_bootstrap as module
sys.orig_argv = [
    sys.executable, '-B', '-P', '-S', '-m', module.WORKER_BOOTSTRAP_MODULE,
    'run-policy', 'authorized-argument',
]
sys.argv = [module.__file__, 'run-policy', 'changed-argument']
code = module.main()
assert 'tgvf_rl.ops.cli_authorization' not in sys.modules
assert code == module.WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_main_refuses_missing_inherited_receipt_before_heavy_import() -> None:
    completed = subprocess.run(
        _bootstrap_command("run-policy"),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


@pytest.mark.parametrize(
    ("mode", "selected_role", "expected_target", "expected_blocker"),
    [
        (
            "run-policy",
            POLICY_DRIVER_ROLE,
            _POLICY_TARGET,
            "runtime-locator-worker-reverification-missing",
        ),
        (
            "run-representation-member",
            REPRESENTATION_MEMBER_ROLE,
            _REPRESENTATION_MEMBER_TARGET,
            "representation-member-consumption-not-performed-by-bootstrap",
        ),
    ],
)
def test_authorized_inspection_still_exits_nonzero_without_target_dispatch(
    tmp_path: Path,
    mode: str,
    selected_role: str,
    expected_target: str,
    expected_blocker: str,
) -> None:
    completed = _run_valid_main(tmp_path, mode)

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    record = json.loads(completed.stdout)
    assert record["mode"] == mode
    assert record["selected_role"] == selected_role
    assert record["target"] == expected_target
    assert record["outer_cli_receipt_checked_by_existing_verifier"] is True
    assert record["current_python_executable_identity_checked"] is True
    assert record["current_python_descriptor_retained"] is False
    assert record["role_child_environment_base_identity_checked"] is True
    assert record["role_child_environment_late_field_inventory_checked"] is True
    assert record["role_child_environment_late_values_checked"] is False
    assert record["heavy_import_roots_absent"] is True
    assert record["runtime_origin_verified"] is False
    assert record["target_imported"] is False
    assert record["verified_worker_startup_minted"] is False
    assert record["dispatch_authorized"] is False
    assert "hostile-same-process-import-machinery-unclosed" in record["blockers"]
    assert "canonical-worker-bootstrap-routing-missing" in record["blockers"]
    assert (
        "role-specific-child-environment-late-value-validation-missing"
        in record["blockers"]
    )
    assert expected_blocker in record["blockers"]
    assert "inspection-only scaffold cannot dispatch" in completed.stderr


@pytest.mark.parametrize(
    "parameter_name",
    [
        "python_executable",
        "python_executable_realpath",
        "python_executable_sha256",
        "python_executable_size",
        "python_executable_device",
        "python_executable_inode",
        "python_executable_mode",
    ],
)
def test_current_python_identity_parameter_drift_is_rejected(
    tmp_path: Path,
    parameter_name: str,
) -> None:
    completed = _run_changed_contract_main(
        tmp_path,
        "run-policy",
        parameter_updates={parameter_name: "changed"},
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_current_python_identity_parameter_namespace_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    completed = _run_changed_contract_main(
        tmp_path,
        "run-policy",
        parameter_updates=(
            {"python_executable_legacy_identity": "changed"}
            if mutation == "extra"
            else None
        ),
        parameter_removals=(
            ("python_executable_mode",) if mutation == "missing" else ()
        ),
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


@pytest.mark.parametrize(
    ("mode", "mutation", "name"),
    [
        ("run-policy", "missing-base", "PATH"),
        ("run-policy", "changed-base", "PATH"),
        ("run-policy", "extra-base", "UNEXPECTED_CHILD_FIELD"),
        (
            "run-policy",
            "missing-late",
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH",
        ),
        ("run-representation-member", "missing-base", "PATH"),
        ("run-representation-member", "changed-base", "PATH"),
        (
            "run-representation-member",
            "extra-base",
            "UNEXPECTED_CHILD_FIELD",
        ),
        ("run-representation-member", "missing-late", "RANK"),
    ],
)
def test_role_child_environment_structure_drift_is_rejected(
    tmp_path: Path,
    mode: str,
    mutation: str,
    name: str,
) -> None:
    completed = _run_changed_contract_main(
        tmp_path,
        mode,
        environment_updates=(
            {name: "/changed"}
            if mutation in {"changed-base", "extra-base"}
            else None
        ),
        environment_removals=(
            (name,) if mutation in {"missing-base", "missing-late"} else ()
        ),
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


@pytest.mark.parametrize(
    ("mode", "name", "value"),
    [
        (
            "run-policy",
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256",
            "not-a-digest",
        ),
        ("run-representation-member", "RANK", "not-a-rank"),
        ("run-representation-member", "MASTER_PORT", "not-a-port"),
    ],
)
def test_late_environment_values_remain_an_explicit_inspection_blocker(
    tmp_path: Path,
    mode: str,
    name: str,
    value: str,
) -> None:
    completed = _run_changed_contract_main(
        tmp_path,
        mode,
        environment_updates={name: value},
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    record = json.loads(completed.stdout)
    assert record["role_child_environment_base_identity_checked"] is True
    assert record["role_child_environment_late_field_inventory_checked"] is True
    assert record["role_child_environment_late_values_checked"] is False
    assert (
        "role-specific-child-environment-late-value-validation-missing"
        in record["blockers"]
    )
    assert "inspection-only scaffold cannot dispatch" in completed.stderr


@pytest.mark.parametrize("mode", WORKER_BOOTSTRAP_MODES)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_child_environment_parameter_namespace_is_exact(
    tmp_path: Path,
    mode: str,
    mutation: str,
) -> None:
    completed = _run_changed_contract_main(
        tmp_path,
        mode,
        parameter_updates=(
            {"child_environment_legacy_identity": "changed"}
            if mutation == "extra"
            else None
        ),
        parameter_removals=(
            ("child_environment_owned_names_sha256",)
            if mutation == "missing"
            else ()
        ),
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


@pytest.mark.parametrize("mutation", ["extra-prefix", "noncanonical-json"])
def test_cli_environment_namespace_and_identity_spelling_are_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    envelope = _envelope("run-policy")
    identity = _execution_identity("run-policy", envelope)
    environment = _materialized_role_environment(
        "run-policy",
        _authorized_environment(tmp_path, identity),
    )
    if mutation == "extra-prefix":
        environment["TGVF_CLI_LEGACY_REPLAY_AUTHORITY"] = "injected"
    else:
        environment["TGVF_CLI_EXECUTION_IDENTITY_JSON"] += " "

    completed = subprocess.run(
        envelope.identity_for_role(POLICY_DRIVER_ROLE).command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_changed_fixed_target_is_rejected_without_inspection_record(
    tmp_path: Path,
) -> None:
    envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(
            _identity(
                role=POLICY_DRIVER_ROLE,
                command=_bootstrap_command("run-policy"),
                target="tgvf_rl.alternate:main",
                runtime_digest="a" * 64,
                dependency_digest="b" * 64,
            ),
        ),
    )
    identity = _execution_identity("run-policy", envelope)
    environment = _materialized_role_environment(
        "run-policy",
        _authorized_environment(tmp_path, identity),
    )

    completed = subprocess.run(
        _bootstrap_command("run-policy"),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_changed_process_command_is_rejected_without_inspection_record(
    tmp_path: Path,
) -> None:
    envelope = _envelope("run-policy")
    identity = _execution_identity("run-policy", envelope)
    environment = _materialized_role_environment(
        "run-policy",
        _authorized_environment(tmp_path, identity),
    )

    completed = subprocess.run(
        _bootstrap_command("run-policy", "++trainer.total_epochs=2"),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == WORKER_BOOTSTRAP_REFUSAL_EXIT_CODE
    assert completed.stdout == ""
    assert "did not establish dispatch authority" in completed.stderr


def test_bootstrap_source_has_no_target_dispatch_or_evidence_mint() -> None:
    source_path = _SOURCE_ROOT / "tgvf_rl" / "worker_bootstrap.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    top_level_import_roots = {
        statement.module.partition(".")[0]
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom) and statement.module is not None
    }
    top_level_import_roots.update(
        alias.name.partition(".")[0]
        for statement in tree.body
        if isinstance(statement, ast.Import)
        for alias in statement.names
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }

    assert top_level_import_roots <= {
        "__future__",
        "builtins",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "sys",
        "threading",
    }
    assert not {"eval", "exec", "__import__"}.intersection(called_names)
    assert "importlib" not in top_level_import_roots
    assert "runpy" not in top_level_import_roots
    assert "subprocess" not in top_level_import_roots
    assert not {
        (owner, attribute)
        for owner, attribute in called_attributes
        if owner == "os" and attribute.startswith("exec")
    }
    assert "_mint_verified_worker_startup_for_bootstrap" not in source
    assert "representation_member_selection" not in source
    assert "representation_member_consumption" not in source


def test_canonical_launch_paths_remain_unwired_from_inspection_bootstrap() -> None:
    production_paths = (
        _SOURCE_ROOT / "tgvf_rl" / "cli.py",
        _SOURCE_ROOT / "tgvf_rl" / "ops" / "cli_launch.py",
        _SOURCE_ROOT / "tgvf_rl" / "policy" / "launch.py",
        _SOURCE_ROOT / "tgvf_rl" / "framework" / "verl" / "policy_main.py",
    )

    for path in production_paths:
        assert WORKER_BOOTSTRAP_MODULE not in path.read_text(encoding="utf-8")


def test_python_m_executes_mutable_parent_package_before_bootstrap(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "shadow"
    package = shadow_root / "tgvf_rl"
    package.mkdir(parents=True)
    marker = tmp_path / "parent-imported.txt"
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['TGVF_TEST_PARENT_MARKER']).write_text('parent')\n",
        encoding="utf-8",
    )
    (package / "worker_bootstrap.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "assert Path(os.environ['TGVF_TEST_PARENT_MARKER']).read_text() == 'parent'\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-m", WORKER_BOOTSTRAP_MODULE],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **_isolated_environment(),
            "PYTHONPATH": str(shadow_root),
            "TGVF_TEST_PARENT_MARKER": str(marker),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "parent"


def test_safe_path_and_no_site_still_admit_explicit_pythonpath(
    tmp_path: Path,
) -> None:
    module_root = tmp_path / "explicit-root"
    module_root.mkdir()
    (module_root / "explicit_probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-P",
            "-S",
            "-c",
            "import explicit_probe; print(explicit_probe.VALUE)",
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**_isolated_environment(), "PYTHONPATH": str(module_root)},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "7\n"
