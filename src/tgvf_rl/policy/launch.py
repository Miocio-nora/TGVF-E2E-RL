"""Fail-closed process boundary for an accepted Policy E2E smoke config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import NoReturn

from tgvf_rl.framework.verl.compile_prerequisites import (
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
from tgvf_rl.ops.cli_authorization import (
    CLIExecutionAuthorizationIdentity,
    CLIWorkerAuthorization,
    PythonExecutableIdentity,
    assert_canonical_runtime_launch_enabled,
    bind_current_python_executable,
    cli_worker_authorization_environment,
    environment_sanitization_parameters,
    sanitized_child_environment,
    verify_python_executable_identity,
)

from .horizon_extension import (
    PolicyHorizonExtension,
    validate_policy_horizon_extension_resume,
)
from .run_config import PolicyE2ESmokeRunConfig


POLICY_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_LEDGER_PATH = "docs/EXPERIMENT_LEDGER.md"


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
    child_environment: tuple[tuple[str, str], ...]
    stripped_environment_names: tuple[str, ...]
    repository_root: Path
    horizon_extension: PolicyHorizonExtension | None = None

    @property
    def prepared_identity_sha256(self) -> str:
        record = {
            "schema_version": "tgvf-prepared-policy-launch-v1",
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
            "child_environment_sha256": _canonical_json_sha256(
                dict(self.child_environment)
            ),
            "stripped_environment_names": list(self.stripped_environment_names),
            "repository_root": str(self.repository_root),
        }
        return _canonical_json_sha256(record)

    def authorization_parameters(self) -> dict[str, str]:
        return {
            **self.compile_authorization.authorization_parameters(),
            **self.python_identity.authorization_parameters(),
            **environment_sanitization_parameters(self.stripped_environment_names),
            "prepared_policy_launch_sha256": self.prepared_identity_sha256,
        }


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
    python_identity = bind_current_python_executable(
        python_executable or sys.executable
    )
    command = plan.command(python_identity.declared_path)
    child_environment, stripped_environment_names = policy_child_environment(
        plan,
        include_sanitization_record=True,
    )
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
        child_environment=tuple(sorted(child_environment.items())),
        stripped_environment_names=stripped_environment_names,
        repository_root=Path(repository_root).resolve(),
        horizon_extension=horizon_extension,
    )


def policy_child_environment(
    plan: UpstreamVerlLaunchPlan,
    *,
    base: Mapping[str, str] | None = None,
    include_sanitization_record: bool = False,
) -> dict[str, str] | tuple[dict[str, str], tuple[str, ...]]:
    """Apply the exact launch environment, overriding inherited launch values."""

    if not isinstance(plan, UpstreamVerlLaunchPlan):
        raise TypeError("plan must be UpstreamVerlLaunchPlan")
    result, stripped = sanitized_child_environment(base)
    result.update(plan.environment)
    if include_sanitization_record:
        return result, stripped
    return result


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

    if not isinstance(prepared, PreparedPolicyLaunch):
        raise TypeError("prepared must be PreparedPolicyLaunch")
    if not isinstance(launch_identity, CLIExecutionAuthorizationIdentity):
        raise TypeError("launch_identity must be CLIExecutionAuthorizationIdentity")
    expected_parameters = prepared.authorization_parameters()
    observed_parameters = dict(launch_identity.parameters)
    for name, expected in expected_parameters.items():
        if observed_parameters.get(name) != expected:
            raise RuntimeError(
                f"consumed Policy authorization differs from prepared launch: {name}"
            )
    if (
        launch_identity.run_id != prepared.config.run_id
        or launch_identity.run_identity_sha256 != prepared.config.identity_sha256
    ):
        raise RuntimeError("consumed Policy authorization has a different run identity")
    prepared.plan.assert_launch_ready()
    assert_policy_execution_identity(
        prepared.config,
        repository_root=prepared.repository_root,
        horizon_extension=prepared.horizon_extension,
    )
    verify_python_executable_identity(prepared.python_identity)
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
    prerequisite_receipt_path = materialize_policy_compile_prerequisite_receipt(
        prerequisite_receipt,
        state_directory=prepared.config.output.root
        / "runtime-policy-state"
        / "compile-prerequisite-attestations",
    )
    environment = dict(prepared.child_environment)
    environment["TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256"] = (
        prerequisite_receipt.receipt_sha256
    )
    environment["TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH"] = str(
        prerequisite_receipt_path
    )
    environment["TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256"] = (
        prepared.compile_prerequisites.identity_sha256
    )
    environment["TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256"] = (
        prepared.compile_prerequisites.manifest_source_sha256
    )
    environment.update(
        cli_worker_authorization_environment(
            launch_identity,
            worker_authorization,
            gate_directory=gate_directory,
        )
    )
    os.execve(
        str(prepared.python_identity.declared_path),
        prepared.command,
        environment,
    )


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
    "POLICY_REPOSITORY_ROOT",
    "PreparedPolicyLaunch",
    "PolicyCompileAuthorizationProof",
    "assert_policy_execution_identity",
    "build_policy_launch_record",
    "execute_policy_e2e_smoke",
    "policy_child_environment",
    "preflight_policy_launch_for_authorization",
]
