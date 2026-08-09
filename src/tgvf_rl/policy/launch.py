"""Fail-closed process boundary for an accepted Policy E2E smoke config."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import NoReturn

from tgvf_rl.framework.verl.launcher import (
    UpstreamVerlLaunchPlan,
    build_policy_e2e_smoke_verl_plan,
)

from .horizon_extension import (
    PolicyHorizonExtension,
    validate_policy_horizon_extension_resume,
)
from .run_config import PolicyE2ESmokeRunConfig


POLICY_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_LEDGER_PATH = "docs/EXPERIMENT_LEDGER.md"


def build_policy_launch_record(
    config: PolicyE2ESmokeRunConfig,
    *,
    python_executable: str | Path | None = None,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> dict[str, object]:
    """Build a JSON-safe plan; blocked plans deliberately omit executable argv."""

    plan = build_policy_e2e_smoke_verl_plan(
        config, horizon_extension=horizon_extension
    )
    executable = Path(python_executable or sys.executable).absolute()
    record = plan.as_record()
    record["python_executable"] = str(executable)
    if plan.launch_ready:
        record["command"] = list(plan.command(executable))
    else:
        record["command"] = None
    return record


def policy_child_environment(
    plan: UpstreamVerlLaunchPlan,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply the exact launch environment, overriding inherited launch values."""

    if not isinstance(plan, UpstreamVerlLaunchPlan):
        raise TypeError("plan must be UpstreamVerlLaunchPlan")
    result = dict(os.environ if base is None else base)
    result.update(plan.environment)
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
    if not root.is_dir() or _git_output(
        root, "rev-parse", "--is-inside-work-tree"
    ) != "true":
        raise RuntimeError("Policy launch repository root is not a Git worktree")
    observed_root = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if observed_root != root:
        raise RuntimeError("Policy launch repository root differs from Git toplevel")
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
                "could not inspect policy launch worktree: "
                + completed.stderr.strip()
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
    config: PolicyE2ESmokeRunConfig,
    *,
    python_executable: str | Path | None = None,
    base_environment: Mapping[str, str] | None = None,
    repository_root: str | Path = POLICY_REPOSITORY_ROOT,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> NoReturn:
    """Replace the CLI process with upstream veRL after every local preflight."""

    plan = build_policy_e2e_smoke_verl_plan(
        config, horizon_extension=horizon_extension
    )
    plan.assert_launch_ready()
    assert_policy_execution_identity(
        config,
        repository_root=repository_root,
        horizon_extension=horizon_extension,
    )
    executable = Path(python_executable or sys.executable).absolute()
    command = plan.command(executable)
    environment = policy_child_environment(plan, base=base_environment)
    os.execve(str(executable), command, environment)


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


__all__ = [
    "POLICY_REPOSITORY_ROOT",
    "assert_policy_execution_identity",
    "build_policy_launch_record",
    "execute_policy_e2e_smoke",
    "policy_child_environment",
]
