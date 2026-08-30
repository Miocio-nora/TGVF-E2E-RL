"""Fail-closed process boundary for an accepted Policy E2E smoke config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import NoReturn

from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY,
    PolicyCompilePrerequisiteBinding,
    PolicyCompilePrerequisiteReceipt,
    load_policy_compile_prerequisite_manifest,
    materialize_policy_compile_prerequisite_receipt,
)
from tgvf_rl.framework.verl.launcher import (
    UpstreamVerlLaunchPlan,
    build_policy_e2e_smoke_verl_plan,
)
from tgvf_rl.ops.child_environment import (
    CLI_WORKER_LATE_ENVIRONMENT_NAMES,
    POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
    POLICY_VERL_DRIVER_PROFILE,
    ChildEnvironmentBinding,
    build_child_environment,
)
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    PythonExecutableBinding,
    PythonExecutableIdentity,
    assert_fd_exec_supported,
    assert_canonical_runtime_launch_enabled,
    bind_current_python_executable_for_exec,
    cli_worker_authorization_environment,
    verify_python_executable_binding,
)
from tgvf_rl.ops.runtime_locator import VerifiedRuntimeLocatorScaffoldEvidence
from tgvf_rl.ops.worker_startup import (
    POLICY_DRIVER_ROLE,
    WorkerStartupEnvelope,
    WorkerStartupIdentity,
)

from .horizon_extension import (
    PolicyHorizonExtension,
    validate_policy_horizon_extension_resume,
)
from .run_config import PolicyE2ESmokeRunConfig


POLICY_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POLICY_DRIVER_STARTUP_TARGET = "tgvf_rl.framework.verl.policy_main:main"
_POLICY_DRIVER_MAIN_MODULE = POLICY_DRIVER_STARTUP_TARGET.partition(":")[0]
_EXPERIMENT_LEDGER_PATH = "docs/EXPERIMENT_LEDGER.md"
_WORKER_STARTUP_ENVELOPE_AUTHORIZATION_KEYS = frozenset(
    {
        "worker_startup_envelope_schema",
        "worker_startup_envelope_json",
        "worker_startup_envelope_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyCompileAuthorizationProof:
    """Content identities verified before a one-time launch token is consumed."""

    manifest_source_path: Path
    manifest_source_sha256: str
    binding_sha256: str
    receipt_sha256: str
    closure_policy: str = POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY

    def __post_init__(self) -> None:
        path = Path(self.manifest_source_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "compile-prerequisite proof manifest path must be absolute"
            )
        object.__setattr__(self, "manifest_source_path", path)
        for field_name in (
            "manifest_source_sha256",
            "binding_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA256")
        if self.closure_policy != POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY:
            raise ValueError("compile-prerequisite proof closure policy differs")

    def authorization_parameters(self) -> dict[str, str]:
        return {
            "compile_prerequisite_manifest_path": str(self.manifest_source_path),
            "compile_prerequisite_manifest_sha256": self.manifest_source_sha256,
            "compile_prerequisite_binding_sha256": self.binding_sha256,
            "compile_prerequisite_receipt_sha256": self.receipt_sha256,
            "compile_prerequisite_closure_policy": self.closure_policy,
        }


@dataclass(frozen=True, slots=True)
class PreparedPolicyLaunch:
    """Complete immutable result of every deterministic pre-consumption check."""

    config: PolicyE2ESmokeRunConfig
    plan: UpstreamVerlLaunchPlan
    compile_prerequisites: PolicyCompilePrerequisiteBinding
    compile_receipt: PolicyCompilePrerequisiteReceipt
    compile_authorization: PolicyCompileAuthorizationProof
    python_identity: PythonExecutableIdentity
    command: tuple[str, ...]
    child_environment_binding: ChildEnvironmentBinding
    repository_root: Path
    runtime_locator_evidence: InitVar[VerifiedRuntimeLocatorScaffoldEvidence]
    horizon_extension: PolicyHorizonExtension | None = None
    python_binding: PythonExecutableBinding | None = None
    worker_startup_envelope: WorkerStartupEnvelope = field(init=False)
    _worker_startup_envelope_sha256: str = field(init=False, repr=False)

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("PreparedPolicyLaunch cannot be subclassed")

    def __post_init__(
        self,
        runtime_locator_evidence: VerifiedRuntimeLocatorScaffoldEvidence,
    ) -> None:
        if not isinstance(self.child_environment_binding, ChildEnvironmentBinding):
            raise TypeError("child_environment_binding must be ChildEnvironmentBinding")
        if self.child_environment_binding.profile != POLICY_VERL_DRIVER_PROFILE:
            raise ValueError("prepared Policy child environment profile differs")
        if self.child_environment_binding.late_overlay_names:
            raise ValueError(
                "prepared Policy outer environment already contains a late overlay"
            )
        if (
            self.python_binding is not None
            and self.python_binding.identity != self.python_identity
        ):
            raise ValueError("prepared Python binding differs from its identity")
        envelope = _build_policy_worker_startup_envelope(
            command=self.command,
            python_identity=self.python_identity,
            runtime_locator_evidence=runtime_locator_evidence,
        )
        object.__setattr__(self, "worker_startup_envelope", envelope)
        object.__setattr__(
            self,
            "_worker_startup_envelope_sha256",
            envelope.envelope_sha256,
        )

    def close_python_binding(self) -> None:
        """Release the process-local executable capability if still owned."""

        if self.python_binding is not None:
            self.python_binding.close()

    @property
    def prepared_identity_sha256(self) -> str:
        startup_envelope = self._validated_worker_startup_envelope()
        record = {
            "schema_version": "tgvf-prepared-policy-launch-v3",
            "run_identity_sha256": self.config.identity_sha256,
            "config_source_sha256": self.config.source_sha256,
            "horizon_extension_sha256": (
                None
                if self.horizon_extension is None
                else self.horizon_extension.source_sha256
            ),
            "plan": self.plan.as_record(),
            "compile": self.compile_authorization.authorization_parameters(),
            "python": self.python_identity.authorization_parameters(),
            "command": list(self.command),
            "worker_startup_envelope": startup_envelope.as_record(),
            "child_environment": (
                self.child_environment_binding.authorization_parameters()
            ),
            "repository_root": str(self.repository_root),
        }
        return _canonical_json_sha256(record)

    def authorization_parameters(self) -> dict[str, str]:
        startup_envelope = self._validated_worker_startup_envelope()
        startup_parameters = startup_envelope.authorization_parameters()
        if (
            type(startup_parameters) is not dict
            or set(startup_parameters) != _WORKER_STARTUP_ENVELOPE_AUTHORIZATION_KEYS
        ):
            raise RuntimeError(
                "Policy worker startup authorization parameter group differs"
            )
        merged = _merge_disjoint_authorization_parameter_groups(
            (
                "compile prerequisites",
                self.compile_authorization.authorization_parameters(),
            ),
            ("Python executable", self.python_identity.authorization_parameters()),
            (
                "child environment",
                self.child_environment_binding.authorization_parameters(),
            ),
            ("worker startup envelope", startup_parameters),
            (
                "prepared Policy launch",
                {"prepared_policy_launch_sha256": self.prepared_identity_sha256},
            ),
        )
        try:
            reconstructed = WorkerStartupEnvelope.from_authorization_parameters(
                merged,
                expected_entry_role=POLICY_DRIVER_ROLE,
            )
        except (TypeError, ValueError, PermissionError) as error:
            raise RuntimeError(
                "Policy worker startup authorization namespace differs"
            ) from error
        if reconstructed != startup_envelope:
            raise RuntimeError("Policy worker startup authorization envelope differs")
        return merged

    def _validated_worker_startup_envelope(self) -> WorkerStartupEnvelope:
        envelope = self.worker_startup_envelope
        if type(envelope) is not WorkerStartupEnvelope:
            raise RuntimeError("prepared Policy worker startup envelope type differs")
        if envelope.envelope_sha256 != self._worker_startup_envelope_sha256:
            raise RuntimeError("prepared Policy worker startup envelope changed")
        if envelope.entry_role != POLICY_DRIVER_ROLE:
            raise RuntimeError("prepared Policy worker startup entry role differs")
        identity = envelope.identity_for_role(POLICY_DRIVER_ROLE)
        if identity.command != self.command:
            raise RuntimeError("prepared Policy worker startup command differs")
        if identity.target != POLICY_DRIVER_STARTUP_TARGET:
            raise RuntimeError("prepared Policy worker startup target differs")
        return envelope


def build_policy_launch_record(
    config: PolicyE2ESmokeRunConfig,
    *,
    python_executable: str | Path | None = None,
    horizon_extension: PolicyHorizonExtension | None = None,
    compile_prerequisite_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Build a JSON-safe plan; blocked plans deliberately omit executable argv."""

    compile_prerequisites = _load_compile_prerequisites(
        compile_prerequisite_manifest_path
    )
    plan = build_policy_e2e_smoke_verl_plan(
        config,
        horizon_extension=horizon_extension,
        compile_prerequisites=compile_prerequisites,
    )
    executable = Path(python_executable or sys.executable).absolute()
    record = plan.as_record()
    record["python_executable"] = str(executable)
    if plan.launch_ready:
        record["command"] = list(plan.command(executable))
    else:
        record["command"] = None
    return record


def preflight_policy_launch_for_authorization(
    config: PolicyE2ESmokeRunConfig,
    *,
    compile_prerequisite_manifest_path: str | Path | None,
    runtime_locator_evidence: VerifiedRuntimeLocatorScaffoldEvidence | None = None,
    python_executable: str | Path | None = None,
    repository_root: str | Path = POLICY_REPOSITORY_ROOT,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> PreparedPolicyLaunch:
    """Reject every static/live blocker before consuming launch authorization.

    Plan construction remains pure.  When a manifest is present, its minimum
    declarations are content-verified before plan readiness is asserted.  The
    current v1 manifest then still fails on its explicit recursive-header and
    system-toolchain residual, so no authorization token can be burned for a
    launch that is already known to be incomplete.
    """

    assert_canonical_runtime_launch_enabled()
    compile_prerequisites = _load_compile_prerequisites(
        compile_prerequisite_manifest_path
    )
    plan = build_policy_e2e_smoke_verl_plan(
        config,
        horizon_extension=horizon_extension,
        compile_prerequisites=compile_prerequisites,
    )
    if compile_prerequisites is None:
        plan.assert_launch_ready()
        raise AssertionError("missing compile manifest unexpectedly became ready")
    prerequisite_receipt = plan.preflight_live_prerequisites()
    plan.assert_launch_ready()
    assert_policy_execution_identity(
        config,
        repository_root=repository_root,
        horizon_extension=horizon_extension,
    )
    if runtime_locator_evidence is None:
        raise RuntimeError(
            "verified Policy runtime locator scaffold evidence is required"
        )
    python_binding = bind_current_python_executable_for_exec(
        python_executable or sys.executable
    )
    try:
        python_identity = python_binding.identity
        command = plan.command(python_identity.declared_path)
        child_environment_binding = _policy_child_environment_binding(plan)
        compile_authorization = PolicyCompileAuthorizationProof(
            manifest_source_path=compile_prerequisites.manifest_source_path,
            manifest_source_sha256=compile_prerequisites.manifest_source_sha256,
            binding_sha256=compile_prerequisites.identity_sha256,
            receipt_sha256=prerequisite_receipt.receipt_sha256,
            closure_policy=compile_prerequisites.closure_policy,
        )
        return PreparedPolicyLaunch(
            config=config,
            plan=plan,
            compile_prerequisites=compile_prerequisites,
            compile_receipt=prerequisite_receipt,
            compile_authorization=compile_authorization,
            python_identity=python_identity,
            command=command,
            child_environment_binding=child_environment_binding,
            repository_root=Path(repository_root).resolve(),
            runtime_locator_evidence=runtime_locator_evidence,
            horizon_extension=horizon_extension,
            python_binding=python_binding,
        )
    except BaseException:
        python_binding.close()
        raise


def _build_policy_worker_startup_envelope(
    *,
    command: tuple[str, ...],
    python_identity: PythonExecutableIdentity,
    runtime_locator_evidence: VerifiedRuntimeLocatorScaffoldEvidence,
) -> WorkerStartupEnvelope:
    """Build authorization data from borrowed, already-verified evidence.

    The evidence remains owned by the caller.  This function neither retains
    nor closes its descriptor capabilities.
    """

    if type(runtime_locator_evidence) is not VerifiedRuntimeLocatorScaffoldEvidence:
        raise TypeError(
            "runtime_locator_evidence must be exactly "
            "VerifiedRuntimeLocatorScaffoldEvidence"
        )
    if type(python_identity) is not PythonExecutableIdentity:
        raise TypeError("python_identity must be exactly PythonExecutableIdentity")
    if type(command) is not tuple or not command:
        raise TypeError("prepared Policy command must be a non-empty exact tuple")
    if command[0] != str(python_identity.declared_path):
        raise ValueError("prepared Policy argv[0] differs from declared Python path")
    if len(command) < 3 or command[1:3] != ("-m", _POLICY_DRIVER_MAIN_MODULE):
        raise ValueError("prepared Policy argv does not invoke the fixed driver module")

    manifest = runtime_locator_evidence.manifest
    executable = manifest.executable
    if executable.path != python_identity.resolved_path:
        raise ValueError(
            "Policy runtime locator executable path differs from resolved Python path"
        )
    if executable.sha256 != python_identity.sha256:
        raise ValueError(
            "Policy runtime locator executable SHA256 differs from Python identity"
        )
    if executable.byte_length != python_identity.byte_length:
        raise ValueError(
            "Policy runtime locator executable byte length differs from Python identity"
        )
    if POLICY_DRIVER_STARTUP_TARGET not in manifest.target_coordinates:
        raise ValueError(
            "Policy runtime locator does not declare the fixed driver target"
        )

    identity = WorkerStartupIdentity(
        role=POLICY_DRIVER_ROLE,
        command=command,
        target=POLICY_DRIVER_STARTUP_TARGET,
        runtime_package_sha256=runtime_locator_evidence.runtime_package_sha256,
        dependency_roots_sha256=runtime_locator_evidence.dependency_roots_sha256,
    )
    return WorkerStartupEnvelope(
        entry_role=POLICY_DRIVER_ROLE,
        identities=(identity,),
    )


def _merge_disjoint_authorization_parameter_groups(
    *groups: tuple[str, dict[str, str]],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for label, parameters in groups:
        if type(label) is not str or not label:
            raise TypeError("Policy authorization parameter group label differs")
        if type(parameters) is not dict:
            raise TypeError(
                f"Policy {label} authorization parameters must be an exact dict"
            )
        if any(type(key) is not str for key in parameters):
            raise TypeError(
                f"Policy {label} authorization parameter keys must be exactly str"
            )
        if any(type(value) is not str for value in parameters.values()):
            raise TypeError(
                f"Policy {label} authorization parameter values must be exactly str"
            )
        collisions = sorted(set(merged).intersection(parameters))
        if collisions:
            raise RuntimeError(
                f"Policy {label} authorization parameters collide: {collisions!r}"
            )
        merged.update(parameters)
    return merged


def policy_child_environment(
    plan: UpstreamVerlLaunchPlan,
    *,
    base: Mapping[str, str] | None = None,
    include_sanitization_record: bool = False,
) -> dict[str, str] | tuple[dict[str, str], tuple[str, ...]]:
    """Return the strict driver profile without inheriting host values."""

    binding = _policy_child_environment_binding(plan, base=base)
    result = binding.as_environment()
    if include_sanitization_record:
        rejected = tuple(
            sorted((*binding.ignored_host_names, *binding.rejected_host_names))
        )
        return result, rejected
    return result


def _policy_child_environment_binding(
    plan: UpstreamVerlLaunchPlan,
    *,
    base: Mapping[str, str] | None = None,
) -> ChildEnvironmentBinding:
    """Bind only profile-owned plan entries atop a fixed safe baseline."""

    if not isinstance(plan, UpstreamVerlLaunchPlan):
        raise TypeError("plan must be UpstreamVerlLaunchPlan")
    return build_child_environment(
        POLICY_VERL_DRIVER_PROFILE,
        owned_environment=plan.environment,
        host_environment=base,
    )


def assert_policy_execution_identity(
    config: PolicyE2ESmokeRunConfig,
    *,
    repository_root: str | Path = POLICY_REPOSITORY_ROOT,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> None:
    """Verify code/config/output identities immediately before process replacement."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("config must be PolicyE2ESmokeRunConfig")
    root = Path(repository_root).resolve()
    if not (root / ".git").is_dir():
        raise RuntimeError("Policy launch repository root is not a Git worktree")
    observed_commit = _git_output(root, "rev-parse", "HEAD")
    configured_commit = (
        horizon_extension.code_commit
        if horizon_extension is not None
        else config.code.commit
    )
    additional_allowed_paths: tuple[Path, ...] = ()
    if horizon_extension is not None:
        horizon_extension.validate_for_config(config)
        validate_policy_horizon_extension_resume(horizon_extension, config)
        additional_allowed_paths = (horizon_extension.source_path,)
    _assert_code_commit_or_ledger_only_descendant(
        root,
        configured_commit=configured_commit,
        observed_commit=observed_commit,
        config_source_path=config.source_path,
        additional_allowed_paths=additional_allowed_paths,
    )
    for args in (
        ("diff", "--quiet", "--ignore-submodules", "--"),
        ("diff", "--cached", "--quiet", "--ignore-submodules", "--"),
    ):
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                "could not inspect policy launch worktree: " + completed.stderr.strip()
            )
        if completed.returncode == 1:
            raise RuntimeError("policy launch requires no tracked Git changes")
    if _sha256_file(config.source_path) != config.source_sha256:
        raise RuntimeError("policy run config changed after strict validation")
    if horizon_extension is not None and (
        _sha256_file(horizon_extension.source_path) != horizon_extension.source_sha256
    ):
        raise RuntimeError("policy horizon extension changed after strict validation")
    if config.training.resume_mode == "disable":
        if config.output.root.exists():
            raise RuntimeError(
                "fresh policy launch output.root already exists; refusing overwrite"
            )
    elif config.training.resume_mode == "auto":
        # One immutable config is deliberately valid both before the first
        # launch and after a clean-process restart.  Upstream veRL selects the
        # latest checkpoint under ``default_local_dir`` when it exists; the
        # paired project checkpoint then enforces the exact run identity.
        if config.output.root.exists() and not config.output.root.is_dir():
            raise RuntimeError("policy auto-resume output.root is not a directory")
        if config.training.resume_from_path is not None:
            raise RuntimeError("policy auto-resume must not bind an explicit path")
    elif config.training.resume_mode == "resume_path":
        if not config.output.root.is_dir():
            raise RuntimeError("policy resume output.root is missing")
        if config.training.resume_from_path is None:
            raise RuntimeError("policy resume path disappeared after validation")
    else:  # pragma: no cover - run-config construction owns this invariant
        raise RuntimeError("unsupported policy resume mode")


def execute_policy_e2e_smoke(
    prepared: PreparedPolicyLaunch,
    *,
    launch_identity: CLIExecutionAuthorizationIdentity,
    worker_authorization: CLIWorkerAuthorization,
    gate_directory: str | Path,
) -> NoReturn:
    """Execute only the immutable plan proven before token consumption.

    In particular this boundary accepts no manifest or config path from which a
    second, potentially different launch identity could be constructed.
    """

    if type(prepared) is not PreparedPolicyLaunch:
        raise TypeError("prepared must be PreparedPolicyLaunch")
    python_binding = prepared.python_binding
    if python_binding is None:
        raise RuntimeError("prepared Policy launch has no bound Python fd")
    try:
        if python_binding.identity != prepared.python_identity:
            raise RuntimeError(
                "prepared Policy Python capability differs from its authorization identity"
            )
        if type(launch_identity) is not CLIExecutionAuthorizationIdentity:
            raise TypeError(
                "launch_identity must be exactly CLIExecutionAuthorizationIdentity"
            )
        expected_parameters = prepared.authorization_parameters()
        observed_parameters = dict(launch_identity.parameters)
        try:
            observed_startup_envelope = (
                WorkerStartupEnvelope.from_authorization_parameters(
                    observed_parameters,
                    expected_entry_role=POLICY_DRIVER_ROLE,
                )
            )
        except (TypeError, ValueError, PermissionError) as error:
            raise RuntimeError(
                "consumed Policy worker startup authorization differs"
            ) from error
        if observed_startup_envelope != prepared._validated_worker_startup_envelope():
            raise RuntimeError(
                "consumed Policy worker startup envelope differs from prepared launch"
            )
        for name, expected in expected_parameters.items():
            if observed_parameters.get(name) != expected:
                raise RuntimeError(
                    "consumed Policy authorization differs from prepared launch: "
                    f"{name}"
                )
        if (
            launch_identity.run_id != prepared.config.run_id
            or launch_identity.run_identity_sha256 != prepared.config.identity_sha256
        ):
            raise RuntimeError(
                "consumed Policy authorization has a different run identity"
            )
        prepared.plan.assert_launch_ready()
        assert_policy_execution_identity(
            prepared.config,
            repository_root=prepared.repository_root,
            horizon_extension=prepared.horizon_extension,
        )
        prerequisite_receipt = prepared.plan.preflight_live_prerequisites()
        if prerequisite_receipt != prepared.compile_receipt:
            raise RuntimeError(
                "Policy compile prerequisites changed after authorization preflight"
            )
        if (
            prepared.plan.command(prepared.python_identity.declared_path)
            != prepared.command
        ):
            raise RuntimeError("prepared Policy command changed after authorization")
        if not prepared.command or prepared.command[0] != str(
            prepared.python_identity.declared_path
        ):
            raise RuntimeError("Policy argv[0] lost its declared Python path")
        prerequisite_receipt_path = materialize_policy_compile_prerequisite_receipt(
            prerequisite_receipt,
            state_directory=prepared.config.output.root
            / "runtime-policy-state"
            / "compile-prerequisite-attestations",
        )
        late_environment = {
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256": (
                prerequisite_receipt.receipt_sha256
            ),
            "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH": str(
                prerequisite_receipt_path
            ),
            **cli_worker_authorization_environment(
                launch_identity,
                worker_authorization,
                gate_directory=gate_directory,
            ),
        }
        expected_late_names = {
            *CLI_WORKER_LATE_ENVIRONMENT_NAMES,
            *POLICY_COMPILE_RECEIPT_LATE_ENVIRONMENT_NAMES,
        }
        if set(late_environment) != expected_late_names:
            raise RuntimeError("Policy late child environment field set differs")
        environment = prepared.child_environment_binding.with_late_overlay(
            late_environment
        ).as_environment()
        descriptor = verify_python_executable_binding(python_binding)
        assert_fd_exec_supported()
        os.execve(descriptor, prepared.command, environment)
    finally:
        python_binding.close()


def _load_compile_prerequisites(
    manifest_path: str | Path | None,
) -> PolicyCompilePrerequisiteBinding | None:
    if manifest_path is None:
        return None
    return load_policy_compile_prerequisite_manifest(manifest_path)


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not verify policy launch Git identity") from error
    return completed.stdout.strip()


def _assert_code_commit_or_ledger_only_descendant(
    root: Path,
    *,
    configured_commit: str,
    observed_commit: str,
    config_source_path: Path | None = None,
    additional_allowed_paths: tuple[Path, ...] = (),
) -> None:
    """Require descendant implementation recovery to be committed with its ledger.

    A tracked run config cannot name the hash of the commit that contains its
    own bytes.  The executable code identity is therefore committed first; a
    descendant commit may add that exact run-config path and the experiment
    ledger. A later committed bug fix is allowed only when the observed commit
    also updates the experiment ledger, keeping recovery provenance explicit.
    Uncommitted work remains forbidden by the surrounding launch validation.
    """

    if configured_commit == observed_commit:
        return
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", configured_commit, observed_commit),
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestor.returncode not in (0, 1):
        raise RuntimeError(
            "could not verify configured Policy code ancestry: "
            + ancestor.stderr.strip()
        )
    if ancestor.returncode == 1:
        raise RuntimeError(
            "policy config code commit is not an ancestor of the launch worktree: "
            f"configured={configured_commit} observed={observed_commit}"
        )
    changed = frozenset(
        line
        for line in _git_output(
            root,
            "diff",
            "--name-only",
            f"{configured_commit}..{observed_commit}",
            "--",
        ).splitlines()
        if line
    )
    allowed = {_EXPERIMENT_LEDGER_PATH}
    if config_source_path is not None:
        try:
            config_relative = config_source_path.resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "policy run config must be inside the launch repository"
            ) from error
        allowed.add(config_relative.as_posix())
    for allowed_path in additional_allowed_paths:
        try:
            relative = allowed_path.resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "policy launch extension must be inside the launch repository"
            ) from error
        allowed.add(relative.as_posix())
    if _EXPERIMENT_LEDGER_PATH not in changed:
        raise RuntimeError(
            "policy launch descendant lacks its planned experiment ledger"
        )
    unexpected = changed.difference(allowed)
    if unexpected:
        latest_changed = frozenset(
            line
            for line in _git_output(
                root,
                "diff",
                "--name-only",
                f"{observed_commit}^..{observed_commit}",
                "--",
            ).splitlines()
            if line
        )
        if _EXPERIMENT_LEDGER_PATH not in latest_changed:
            raise RuntimeError(
                "policy launch recovery code must update the experiment ledger "
                f"in the observed commit: {tuple(sorted(unexpected))!r}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "POLICY_DRIVER_STARTUP_TARGET",
    "POLICY_REPOSITORY_ROOT",
    "PreparedPolicyLaunch",
    "PolicyCompileAuthorizationProof",
    "assert_policy_execution_identity",
    "build_policy_launch_record",
    "execute_policy_e2e_smoke",
    "policy_child_environment",
    "preflight_policy_launch_for_authorization",
]
