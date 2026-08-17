from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tgvf_rl.policy.launch import _assert_code_commit_or_ledger_only_descendant


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=TGVF Test",
        "-c",
        "user.email=tgvf@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def test_launch_commit_allows_only_a_ledger_only_descendant(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("VERSION = 1\n", encoding="utf-8")
    code_commit = _commit(tmp_path, "code identity")

    _assert_code_commit_or_ledger_only_descendant(
        tmp_path,
        configured_commit=code_commit,
        observed_commit=code_commit,
    )

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "EXPERIMENT_LEDGER.md").write_text(
        "PLANNED\n", encoding="utf-8"
    )
    ledger_commit = _commit(tmp_path, "plan experiment")
    _assert_code_commit_or_ledger_only_descendant(
        tmp_path,
        configured_commit=code_commit,
        observed_commit=ledger_commit,
    )

    (tmp_path / "src" / "runtime.py").write_text("VERSION = 2\n", encoding="utf-8")
    changed_commit = _commit(tmp_path, "change runtime")
    with pytest.raises(RuntimeError, match="must update the experiment ledger"):
        _assert_code_commit_or_ledger_only_descendant(
            tmp_path,
            configured_commit=code_commit,
            observed_commit=changed_commit,
        )

    (tmp_path / "docs" / "EXPERIMENT_LEDGER.md").write_text(
        "PLANNED\nRECOVERY: runtime fix\n", encoding="utf-8"
    )
    recovery_commit = _commit(tmp_path, "record runtime recovery")
    _assert_code_commit_or_ledger_only_descendant(
        tmp_path,
        configured_commit=code_commit,
        observed_commit=recovery_commit,
    )


def test_launch_commit_allows_the_tracked_config_and_ledger_descendant(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("VERSION = 1\n", encoding="utf-8")
    code_commit = _commit(tmp_path, "code identity")

    config_path = tmp_path / "configs" / "policy" / "run.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(f'commit = "{code_commit}"\n', encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "EXPERIMENT_LEDGER.md").write_text(
        "PLANNED\n", encoding="utf-8"
    )
    manifest_commit = _commit(tmp_path, "plan experiment")

    _assert_code_commit_or_ledger_only_descendant(
        tmp_path,
        configured_commit=code_commit,
        observed_commit=manifest_commit,
        config_source_path=config_path,
    )


def test_launch_commit_allows_the_tracked_config_only_descendant(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("VERSION = 1\n", encoding="utf-8")
    code_commit = _commit(tmp_path, "code identity")

    config_path = tmp_path / "configs" / "policy" / "run.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(f'commit = "{code_commit}"\n', encoding="utf-8")
    config_commit = _commit(tmp_path, "bind experiment config")

    _assert_code_commit_or_ledger_only_descendant(
        tmp_path,
        configured_commit=code_commit,
        observed_commit=config_commit,
        config_source_path=config_path,
    )
