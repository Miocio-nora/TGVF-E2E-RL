from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tgvf_rl.ops.child_environment import (
    POLICY_VERL_DRIVER_PROFILE,
    build_child_environment,
)
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    PythonExecutableIdentity,
    bind_current_python_executable_for_exec,
)
from tgvf_rl.ops.runtime_locator import RuntimeLocatorVerificationError
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    WORKER_STARTUP_ENVELOPE_SCHEMA,
    WorkerStartupEnvelope,
    WorkerStartupIdentity,
)
from tgvf_rl.policy.launch import (
    POLICY_DRIVER_STARTUP_TARGET,
    PreparedPolicyLaunch,
    execute_policy_e2e_smoke,
)
from tests.runtime_locator_support import (
    ALTERNATE_TARGET,
    verified_runtime_locator_evidence,
)


_STARTUP_PARAMETER_KEYS = {
    "worker_startup_envelope_schema",
    "worker_startup_envelope_json",
    "worker_startup_envelope_sha256",
}


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _python_binding():
    executable = Path(sys.executable).absolute()
    return bind_current_python_executable_for_exec(
        executable,
        current_executable=executable,
        require_fd_exec_support=False,
    )


def _prepared_arguments(
    tmp_path: Path,
    *,
    python_identity: PythonExecutableIdentity,
    runtime_locator_evidence: object,
    command: tuple[str, ...] | None = None,
    compile_parameters: dict[str, str] | None = None,
) -> dict[str, object]:
    config = SimpleNamespace(
        identity_sha256="a" * 64,
        source_sha256="b" * 64,
    )
    plan_record = {"launch_ready": True, "backend": "verl"}
    plan = SimpleNamespace(as_record=lambda: plan_record)
    compile_authorization = SimpleNamespace(
        authorization_parameters=lambda: (
            {"compile_binding_sha256": "c" * 64}
            if compile_parameters is None
            else compile_parameters
        )
    )
    return {
        "config": config,
        "plan": plan,
        "compile_prerequisites": SimpleNamespace(),
        "compile_receipt": SimpleNamespace(),
        "compile_authorization": compile_authorization,
        "python_identity": python_identity,
        "command": (
            (
                str(python_identity.declared_path),
                "-m",
                "tgvf_rl.framework.verl.policy_main",
                "--config",
                "/canonical/policy.toml",
            )
            if command is None
            else command
        ),
        "child_environment_binding": build_child_environment(
            POLICY_VERL_DRIVER_PROFILE,
            host_environment={},
        ),
        "repository_root": tmp_path.resolve(),
        "runtime_locator_evidence": runtime_locator_evidence,
    }


def test_prepared_policy_launch_binds_exact_single_role_envelope_and_identity(
    tmp_path: Path,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    try:
        prepared = PreparedPolicyLaunch(
            **_prepared_arguments(
                tmp_path,
                python_identity=binding.identity,
                runtime_locator_evidence=evidence,
            )
        )
        envelope = prepared.worker_startup_envelope
        assert envelope.entry_role == POLICY_DRIVER_ROLE
        assert len(envelope.identities) == 1
        identity = envelope.identity_for_role(POLICY_DRIVER_ROLE)
        assert identity.command == prepared.command
        assert identity.target == POLICY_DRIVER_STARTUP_TARGET
        assert identity.runtime_package_sha256 == evidence.runtime_package_sha256
        assert identity.dependency_roots_sha256 == evidence.dependency_roots_sha256

        expected_record = {
            "schema_version": "tgvf-prepared-policy-launch-v3",
            "run_identity_sha256": prepared.config.identity_sha256,
            "config_source_sha256": prepared.config.source_sha256,
            "horizon_extension_sha256": None,
            "plan": prepared.plan.as_record(),
            "compile": prepared.compile_authorization.authorization_parameters(),
            "python": prepared.python_identity.authorization_parameters(),
            "command": list(prepared.command),
            "worker_startup_envelope": envelope.as_record(),
            "child_environment": (
                prepared.child_environment_binding.authorization_parameters()
            ),
            "repository_root": str(prepared.repository_root),
        }
        assert prepared.prepared_identity_sha256 == _canonical_sha256(expected_record)

        parameters = prepared.authorization_parameters()
        startup_keys = {key for key in parameters if key.startswith("worker_startup_")}
        assert startup_keys == _STARTUP_PARAMETER_KEYS
        assert (
            parameters["worker_startup_envelope_schema"]
            == WORKER_STARTUP_ENVELOPE_SCHEMA
        )
        assert (
            WorkerStartupEnvelope.from_authorization_parameters(
                parameters,
                expected_entry_role=POLICY_DRIVER_ROLE,
            )
            == envelope
        )
        assert (
            parameters["prepared_policy_launch_sha256"]
            == prepared.prepared_identity_sha256
        )
    finally:
        evidence.close()
        binding.close()


def test_runtime_locator_evidence_is_borrowed_not_retained(
    tmp_path: Path,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    prepared = PreparedPolicyLaunch(
        **_prepared_arguments(
            tmp_path,
            python_identity=binding.identity,
            runtime_locator_evidence=evidence,
        )
    )
    evidence.close()
    try:
        assert evidence.closed is True
        assert (
            prepared.authorization_parameters()["worker_startup_envelope_sha256"]
            == prepared.worker_startup_envelope.envelope_sha256
        )
    finally:
        binding.close()


def test_direct_construction_requires_exact_live_locator_evidence(
    tmp_path: Path,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    arguments = _prepared_arguments(
        tmp_path,
        python_identity=binding.identity,
        runtime_locator_evidence=evidence,
    )
    try:
        missing = dict(arguments)
        missing.pop("runtime_locator_evidence")
        with pytest.raises(TypeError, match="runtime_locator_evidence"):
            PreparedPolicyLaunch(**missing)
        with pytest.raises(TypeError, match="exactly VerifiedRuntimeLocator"):
            PreparedPolicyLaunch(
                **{
                    **arguments,
                    "runtime_locator_evidence": object(),
                }
            )

        evidence.close()
        with pytest.raises(RuntimeLocatorVerificationError, match="closed"):
            PreparedPolicyLaunch(**arguments)
    finally:
        if not evidence.closed:
            evidence.close()
        binding.close()


def test_prepared_policy_launch_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _PreparedPolicyLaunchSubclass(PreparedPolicyLaunch):
            pass


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"resolved_path": Path("/different/python")}, "executable path differs"),
        ({"sha256": "f" * 64}, "executable SHA256 differs"),
        ({"byte_length": 1}, "executable byte length differs"),
    ],
)
def test_locator_executable_must_match_python_identity(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    altered_identity = replace(binding.identity, **replacement)
    try:
        with pytest.raises(ValueError, match=message):
            PreparedPolicyLaunch(
                **_prepared_arguments(
                    tmp_path,
                    python_identity=altered_identity,
                    runtime_locator_evidence=evidence,
                )
            )
    finally:
        evidence.close()
        binding.close()


def test_fixed_target_and_exact_argv_are_required(tmp_path: Path) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
        target_coordinates=(ALTERNATE_TARGET,),
    )
    try:
        with pytest.raises(ValueError, match="fixed driver target"):
            PreparedPolicyLaunch(
                **_prepared_arguments(
                    tmp_path,
                    python_identity=binding.identity,
                    runtime_locator_evidence=evidence,
                )
            )
    finally:
        evidence.close()

    other_tmp = tmp_path / "argv-mismatch"
    other_tmp.mkdir()
    evidence = verified_runtime_locator_evidence(
        other_tmp,
        executable=binding.identity.resolved_path,
    )
    try:
        with pytest.raises(ValueError, match=r"argv\[0\]"):
            PreparedPolicyLaunch(
                **_prepared_arguments(
                    other_tmp,
                    python_identity=binding.identity,
                    runtime_locator_evidence=evidence,
                    command=("/different/python", "-m", "worker"),
                )
            )
        with pytest.raises(ValueError, match="fixed driver module"):
            PreparedPolicyLaunch(
                **_prepared_arguments(
                    other_tmp,
                    python_identity=binding.identity,
                    runtime_locator_evidence=evidence,
                    command=(
                        str(binding.identity.declared_path),
                        "-m",
                        "tgvf_rl.framework.verl.other_main",
                    ),
                )
            )
    finally:
        evidence.close()
        binding.close()


def test_callers_cannot_inject_or_tamper_with_startup_envelope(
    tmp_path: Path,
) -> None:
    assert (
        "worker_startup_envelope"
        not in inspect.signature(PreparedPolicyLaunch).parameters
    )
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    arguments = _prepared_arguments(
        tmp_path,
        python_identity=binding.identity,
        runtime_locator_evidence=evidence,
    )
    fake_envelope = WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(
            WorkerStartupIdentity(
                role=POLICY_DRIVER_ROLE,
                command=arguments["command"],  # type: ignore[arg-type]
                target=POLICY_DRIVER_STARTUP_TARGET,
                runtime_package_sha256="1" * 64,
                dependency_roots_sha256="2" * 64,
            ),
        ),
    )
    try:
        with pytest.raises(TypeError, match="worker_startup_envelope"):
            PreparedPolicyLaunch(
                **arguments,
                worker_startup_envelope=fake_envelope,  # type: ignore[call-arg]
            )
        prepared = PreparedPolicyLaunch(**arguments)
        object.__setattr__(prepared, "worker_startup_envelope", fake_envelope)
        with pytest.raises(RuntimeError, match="envelope changed"):
            _ = prepared.prepared_identity_sha256
        with pytest.raises(RuntimeError, match="envelope changed"):
            prepared.authorization_parameters()
    finally:
        evidence.close()
        binding.close()


def test_startup_authorization_group_rejects_collision(
    tmp_path: Path,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    try:
        prepared = PreparedPolicyLaunch(
            **_prepared_arguments(
                tmp_path,
                python_identity=binding.identity,
                runtime_locator_evidence=evidence,
                compile_parameters={
                    "worker_startup_envelope_schema": "downgrade-collision"
                },
            )
        )
        with pytest.raises(RuntimeError, match="parameters collide"):
            prepared.authorization_parameters()
    finally:
        evidence.close()
        binding.close()


def test_aggregate_authorization_rejects_cross_group_startup_namespace_extra(
    tmp_path: Path,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    try:
        prepared = PreparedPolicyLaunch(
            **_prepared_arguments(
                tmp_path,
                python_identity=binding.identity,
                runtime_locator_evidence=evidence,
                compile_parameters={
                    "compile_binding_sha256": "c" * 64,
                    "worker_startup_legacy_sha256": "d" * 64,
                },
            )
        )
        with pytest.raises(RuntimeError, match="startup authorization namespace"):
            prepared.authorization_parameters()
    finally:
        evidence.close()
        binding.close()


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        (
            "extra-startup-key",
            "consumed Policy worker startup authorization differs",
        ),
        ("identity-subclass", "exactly CLIExecutionAuthorizationIdentity"),
    ],
)
def test_execution_rejects_untrusted_consumed_startup_identity(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    arguments = _prepared_arguments(
        tmp_path,
        python_identity=binding.identity,
        runtime_locator_evidence=evidence,
    )
    arguments["python_binding"] = binding
    prepared = PreparedPolicyLaunch(**arguments)
    observed_parameters = prepared.authorization_parameters()
    if variant == "extra-startup-key":
        observed_parameters["worker_startup_legacy_sha256"] = "d" * 64
    launch_identity: CLIExecutionAuthorizationIdentity = (
        CLIExecutionAuthorizationIdentity.create(
            run_id="POLICY-STARTUP-NAMESPACE-TEST",
            phase="policy_training",
            command_id="policy-startup-namespace-test",
            run_identity_sha256=prepared.config.identity_sha256,
            parameters=observed_parameters,
        )
    )
    if variant == "identity-subclass":

        class _IdentitySubclass(CLIExecutionAuthorizationIdentity):
            pass

        launch_identity = _IdentitySubclass(
            run_id=launch_identity.run_id,
            phase=launch_identity.phase,
            command_id=launch_identity.command_id,
            run_identity_sha256=launch_identity.run_identity_sha256,
            parameters=launch_identity.parameters,
        )
    worker = CLIWorkerAuthorization(
        consumption_receipt_path=tmp_path / "consumption.json",
        consumption_receipt_sha256="e" * 64,
        launcher_liveness_receipt_path=tmp_path / "liveness.json",
    )
    try:
        with pytest.raises((TypeError, RuntimeError), match=message):
            execute_policy_e2e_smoke(
                prepared,
                launch_identity=launch_identity,
                worker_authorization=worker,
                gate_directory=tmp_path,
            )
    finally:
        evidence.close()
        binding.close()


@pytest.mark.parametrize(
    "startup_parameters",
    [
        {
            "worker_startup_envelope_schema": WORKER_STARTUP_ENVELOPE_SCHEMA,
            "worker_startup_envelope_json": "{}",
        },
        {
            "worker_startup_envelope_schema": WORKER_STARTUP_ENVELOPE_SCHEMA,
            "worker_startup_envelope_json": "{}",
            "worker_startup_envelope_sha256": "0" * 64,
            "worker_startup_legacy_sha256": "1" * 64,
        },
    ],
)
def test_startup_authorization_group_rejects_downgrade_or_extra_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    startup_parameters: dict[str, str],
) -> None:
    binding = _python_binding()
    evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    try:
        prepared = PreparedPolicyLaunch(
            **_prepared_arguments(
                tmp_path,
                python_identity=binding.identity,
                runtime_locator_evidence=evidence,
            )
        )
        monkeypatch.setattr(
            WorkerStartupEnvelope,
            "authorization_parameters",
            lambda _self: startup_parameters,
        )
        with pytest.raises(RuntimeError, match="parameter group differs"):
            prepared.authorization_parameters()
    finally:
        evidence.close()
        binding.close()
