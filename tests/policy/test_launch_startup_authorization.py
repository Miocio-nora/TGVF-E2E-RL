from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tgvf_rl.policy import launch as policy_launch_module
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
    PolicyRuntimeLocatorAuthorizationProof,
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
_RUNTIME_LOCATOR_PARAMETER_KEYS = {
    "runtime_locator_manifest_path",
    "runtime_locator_manifest_sha256",
    "runtime_locator_manifest_byte_length",
    "runtime_locator_manifest_identity_sha256",
    "runtime_locator_cache_tag",
    "runtime_locator_target_coordinates_json",
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
            "schema_version": "tgvf-prepared-policy-launch-v4",
            "run_identity_sha256": prepared.config.identity_sha256,
            "config_source_sha256": prepared.config.source_sha256,
            "horizon_extension_sha256": None,
            "plan": prepared.plan.as_record(),
            "compile": prepared.compile_authorization.authorization_parameters(),
            "runtime_locator": (
                prepared.runtime_locator_authorization.authorization_parameters()
            ),
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
        assert {
            key for key in parameters if key.startswith("runtime_locator_")
        } == _RUNTIME_LOCATOR_PARAMETER_KEYS
        assert parameters["runtime_locator_manifest_path"] == str(
            evidence.manifest.manifest_source_path
        )
        assert parameters["runtime_locator_manifest_sha256"] == (
            evidence.manifest.manifest_source_sha256
        )
        assert parameters["runtime_locator_manifest_byte_length"] == str(
            evidence.manifest.manifest_source_byte_length
        )
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


def test_preflight_loads_exact_locator_authority_after_static_checks_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _python_binding()
    fixture_evidence = verified_runtime_locator_evidence(
        tmp_path,
        executable=binding.identity.resolved_path,
    )
    manifest = fixture_evidence.manifest
    fixture_evidence.close()
    binding.close()

    events: list[str] = []
    compile_prerequisites = object()
    prerequisite_receipt = object()
    prepared = object()

    class _Plan:
        def preflight_live_prerequisites(self) -> object:
            events.append("compile-live")
            return prerequisite_receipt

        def assert_launch_ready(self) -> None:
            events.append("plan-ready")

    monkeypatch.setattr(
        policy_launch_module,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "_load_compile_prerequisites",
        lambda path: events.append("compile-load") or compile_prerequisites,
    )
    monkeypatch.setattr(
        policy_launch_module,
        "build_policy_e2e_smoke_verl_plan",
        lambda *_args, **_kwargs: events.append("plan-build") or _Plan(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "assert_policy_execution_identity",
        lambda *_args, **_kwargs: events.append("execution-identity"),
    )
    real_load = policy_launch_module.load_runtime_locator_manifest
    real_verify = policy_launch_module.verify_runtime_locator_manifest_scaffold
    minted_evidence = []

    def load_locator(path: object, **kwargs: object) -> object:
        events.append("locator-load")
        assert path == manifest.manifest_source_path
        assert kwargs == {
            "expected_source_sha256": manifest.manifest_source_sha256,
            "expected_source_byte_length": manifest.manifest_source_byte_length,
        }
        return real_load(path, **kwargs)  # type: ignore[arg-type]

    def verify_locator(observed: object, **kwargs: object) -> object:
        events.append("locator-verify")
        assert kwargs == {
            "expected_cache_tag": sys.implementation.cache_tag,
            "expected_target_coordinates": (POLICY_DRIVER_STARTUP_TARGET,),
        }
        evidence = real_verify(observed, **kwargs)  # type: ignore[arg-type]
        minted_evidence.append(evidence)
        return evidence

    monkeypatch.setattr(
        policy_launch_module,
        "load_runtime_locator_manifest",
        load_locator,
    )
    monkeypatch.setattr(
        policy_launch_module,
        "verify_runtime_locator_manifest_scaffold",
        verify_locator,
    )
    monkeypatch.setattr(
        policy_launch_module,
        "_prepare_policy_launch_with_runtime_locator_evidence",
        lambda **_kwargs: events.append("prepare") or prepared,
    )

    assert (
        policy_launch_module.preflight_policy_launch_for_authorization(
            object(),  # type: ignore[arg-type]
            compile_prerequisite_manifest_path=Path("/compile.json"),
            runtime_locator_manifest_path=manifest.manifest_source_path,
            runtime_locator_manifest_source_sha256=(
                manifest.manifest_source_sha256
            ),
            runtime_locator_manifest_source_byte_length=(
                manifest.manifest_source_byte_length
            ),
        )
        is prepared
    )
    assert events == [
        "runtime-closure",
        "compile-load",
        "plan-build",
        "compile-live",
        "plan-ready",
        "execution-identity",
        "locator-load",
        "locator-verify",
        "prepare",
    ]
    assert len(minted_evidence) == 1
    assert minted_evidence[0].closed is True


def test_compile_refusal_precedes_runtime_locator_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Plan:
        def preflight_live_prerequisites(self) -> object:
            events.append("compile-live")
            raise RuntimeError("synthetic compile refusal")

    monkeypatch.setattr(
        policy_launch_module,
        "assert_canonical_runtime_launch_enabled",
        lambda: events.append("runtime-closure"),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "_load_compile_prerequisites",
        lambda _path: events.append("compile-load") or object(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "build_policy_e2e_smoke_verl_plan",
        lambda *_args, **_kwargs: events.append("plan-build") or _Plan(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "load_runtime_locator_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime locator loaded before compile refusal")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic compile refusal"):
        policy_launch_module.preflight_policy_launch_for_authorization(
            object(),  # type: ignore[arg-type]
            compile_prerequisite_manifest_path=Path("/compile.json"),
            runtime_locator_manifest_path=Path("/must-not-load.json"),
            runtime_locator_manifest_source_sha256="invalid",
            runtime_locator_manifest_source_byte_length=0,
        )
    assert events == [
        "runtime-closure",
        "compile-load",
        "plan-build",
        "compile-live",
    ]


def test_locator_close_failure_closes_prepared_python_binding_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Plan:
        def preflight_live_prerequisites(self) -> object:
            return object()

        def assert_launch_ready(self) -> None:
            return None

    class _Evidence:
        def close(self) -> None:
            events.append("evidence-close")
            raise OSError("synthetic locator close failure")

    class _Prepared:
        def close_python_binding(self) -> None:
            events.append("python-close")

    monkeypatch.setattr(
        policy_launch_module,
        "assert_canonical_runtime_launch_enabled",
        lambda: None,
    )
    monkeypatch.setattr(
        policy_launch_module,
        "_load_compile_prerequisites",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "build_policy_e2e_smoke_verl_plan",
        lambda *_args, **_kwargs: _Plan(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "assert_policy_execution_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_launch_module,
        "load_runtime_locator_manifest",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "verify_runtime_locator_manifest_scaffold",
        lambda *_args, **_kwargs: _Evidence(),
    )
    monkeypatch.setattr(
        policy_launch_module,
        "_prepare_policy_launch_with_runtime_locator_evidence",
        lambda **_kwargs: _Prepared(),
    )

    with pytest.raises(OSError, match="synthetic locator close failure"):
        policy_launch_module.preflight_policy_launch_for_authorization(
            object(),  # type: ignore[arg-type]
            compile_prerequisite_manifest_path=Path("/compile.json"),
            runtime_locator_manifest_path=Path("/runtime.json"),
            runtime_locator_manifest_source_sha256="d" * 64,
            runtime_locator_manifest_source_byte_length=1,
        )
    assert events == ["evidence-close", "python-close"]


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


def test_runtime_locator_authorization_is_internal_and_tamper_evident(
    tmp_path: Path,
) -> None:
    assert (
        "runtime_locator_authorization"
        not in inspect.signature(PreparedPolicyLaunch).parameters
    )
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
        changed = replace(
            prepared.runtime_locator_authorization,
            manifest_source_sha256="e" * 64,
        )
        assert type(changed) is PolicyRuntimeLocatorAuthorizationProof
        object.__setattr__(prepared, "runtime_locator_authorization", changed)
        with pytest.raises(RuntimeError, match="authorization changed"):
            _ = prepared.prepared_identity_sha256
        with pytest.raises(RuntimeError, match="authorization changed"):
            prepared.authorization_parameters()
    finally:
        evidence.close()
        binding.close()


@pytest.mark.parametrize(
    ("injected_name", "message"),
    [
        ("runtime_locator_manifest_sha256", "parameters collide"),
        ("runtime_locator_legacy_sha256", "locator authorization namespace"),
    ],
)
def test_runtime_locator_authorization_namespace_is_fully_protected(
    tmp_path: Path,
    injected_name: str,
    message: str,
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
                    injected_name: "d" * 64,
                },
            )
        )
        with pytest.raises(RuntimeError, match=message):
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
        (
            "extra-runtime-locator-key",
            "consumed Policy runtime-locator authorization namespace differs",
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
    elif variant == "extra-runtime-locator-key":
        observed_parameters["runtime_locator_legacy_sha256"] = "d" * 64
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
