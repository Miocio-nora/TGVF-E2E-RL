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

from .run_config import PolicyE2ESmokeRunConfig


POLICY_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_LEDGER_PATH = "docs/EXPERIMENT_LEDGER.md"


def build_policy_launch_record(
    config: PolicyE2ESmokeRunConfig,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, object]:
    """Build a JSON-safe plan; blocked plans deliberately omit executable argv."""

    plan = build_policy_e2e_smoke_verl_plan(config)
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
) -> None:
    """Verify code/config/output identities immediately before process replacement."""

    if not isinstance(config, PolicyE2ESmokeRunConfig):
        raise TypeError("config must be PolicyE2ESmokeRunConfig")
    root = Path(repository_root).resolve()
    if not (root / ".git").is_dir():
        raise RuntimeError("Policy launch repository root is not a Git worktree")
    observed_commit = _git_output(root, "rev-parse", "HEAD")
    _assert_code_commit_or_ledger_only_descendant(
        root,
        configured_commit=config.code.commit,
        observed_commit=observed_commit,
        config_source_path=config.source_path,
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
) -> NoReturn:
    """Replace the CLI process with upstream veRL after every local preflight."""

    plan = build_policy_e2e_smoke_verl_plan(config)
    plan.assert_launch_ready()
    assert_policy_execution_identity(config, repository_root=repository_root)
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
) -> None:
    """Allow only the unavoidable post-code config/ledger commit.

    A tracked run config cannot name the hash of the commit that contains its
    own bytes.  The executable code identity is therefore committed first; a
    descendant commit may add that exact run-config path and the experiment
    ledger.  Any implementation change after the configured code commit still
    fails.  ``config_source_path=None`` retains the ledger-only helper surface
    used by focused tests.
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
    if _EXPERIMENT_LEDGER_PATH not in changed or not changed.issubset(allowed):
        raise RuntimeError(
            "policy launch descendant contains changes beyond its exact run "
            f"config and planned experiment ledger: {tuple(sorted(changed))!r}"
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
