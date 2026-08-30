from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.ops.repository_boundaries import (
    CANONICAL_CONFIG_ROOTS,
    EVIDENCE_ONLY_CONFIG_ROOTS,
    REPOSITORY_BOUNDARY_AUDIT_SCHEMA,
    REPOSITORY_BOUNDARY_POLICY_SCHEMA,
    MachinePathDebt,
    RepositoryBoundaryError,
    audit_repository_boundaries,
    content_tree_inventory_sha256,
    load_repository_boundary_policy,
    relative_path_inventory_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_TOOL = REPOSITORY_ROOT / "tools/audit_repository_boundaries.py"
EMPTY_INVENTORY_SHA256 = relative_path_inventory_sha256(())
EMPTY_CONTENT_TREE_SHA256 = content_tree_inventory_sha256(())


def _make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in (
        "src/tgvf_rl",
        "tools",
        *CANONICAL_CONFIG_ROOTS,
        *EVIDENCE_ONLY_CONFIG_ROOTS,
    ):
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (repository / "src/tgvf_rl/neutral.py").write_text(
        '"""Neutral test module."""\n', encoding="utf-8"
    )
    (repository / "tools/helper.py").write_text(
        '"""Neutral test tool."""\n', encoding="utf-8"
    )
    return repository


def _write_policy(
    repository: Path,
    *,
    run_allowlist: tuple[str, ...] = (),
    machine_allowlist: tuple[MachinePathDebt, ...] = (),
    evidence_overrides: dict[str, tuple[int, str, str]] | None = None,
) -> Path:
    overrides = evidence_overrides or {}
    inventories = []
    for root in EVIDENCE_ONLY_CONFIG_ROOTS:
        file_count, path_digest, content_digest = overrides.get(
            root,
            (0, EMPTY_INVENTORY_SHA256, EMPTY_CONTENT_TREE_SHA256),
        )
        inventories.append(
            {
                "path": root,
                "file_count": file_count,
                "relative_paths_sha256": path_digest,
                "content_tree_sha256": content_digest,
            }
        )
    payload = {
        "schema_version": REPOSITORY_BOUNDARY_POLICY_SCHEMA,
        "policy_id": "TEST-BOUNDARY-POLICY",
        "revision": 1,
        "source_root": "src/tgvf_rl",
        "tools_root": "tools",
        "canonical_config_roots": list(CANONICAL_CONFIG_ROOTS),
        "evidence_only_config_inventories": inventories,
        "run_specific_code_allowlist": list(run_allowlist),
        "machine_path_debt_allowlist": [item.as_record() for item in machine_allowlist],
    }
    path = repository / "configs/ops/repository_boundary_policy.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _kinds(report: object, attribute: str) -> set[str]:
    return {item.kind for item in getattr(report, attribute)}


def test_clean_repository_passes_with_evidence_roots_reported_as_debt(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)

    report = audit_repository_boundaries(repository, policy)

    assert report.status == "pass"
    assert not report.violations
    assert len(report.debts) == len(EVIDENCE_ONLY_CONFIG_ROOTS)
    assert _kinds(report, "debts") == {"evidence_only_config_root"}


def test_exact_historical_code_and_machine_path_debt_is_visible_but_allowed(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    run_path = "tools/run_rp67_legacy.py"
    (repository / run_path).write_text('"""Historical tool."""\n', encoding="utf-8")
    literal = "/nvmesv/example-user/models/example"
    source_path = "src/tgvf_rl/neutral.py"
    (repository / source_path).write_text(
        f"MODEL_PATH = {literal!r}\n", encoding="utf-8"
    )
    policy = _write_policy(
        repository,
        run_allowlist=(run_path,),
        machine_allowlist=(
            MachinePathDebt(
                source_path,
                sha256(literal.encode("utf-8")).hexdigest(),
                1,
            ),
        ),
    )

    report = audit_repository_boundaries(repository, policy)

    assert report.status == "pass"
    assert {"run_specific_code", "machine_absolute_path"} <= _kinds(report, "debts")


def test_new_run_specific_code_and_machine_path_are_violations(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    (repository / "tools/launch_rp99.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src/tgvf_rl/neutral.py").write_text(
        "MODEL_PATH = '/home/alice/model'\n", encoding="utf-8"
    )

    report = audit_repository_boundaries(repository, policy)

    assert report.status == "blocked"
    assert {
        "new_run_specific_code",
        "new_machine_absolute_path",
    } <= _kinds(report, "violations")


def test_neutral_source_cannot_import_quarantined_run_module(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    run_path = "src/tgvf_rl/policy/rp67_legacy.py"
    (repository / run_path).parent.mkdir(parents=True)
    (repository / run_path).write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src/tgvf_rl/neutral.py").write_text(
        "from .policy import rp67_legacy\n", encoding="utf-8"
    )
    policy = _write_policy(repository, run_allowlist=(run_path,))

    report = audit_repository_boundaries(repository, policy)

    findings = [
        item
        for item in report.violations
        if item.kind == "neutral_imports_run_specific"
    ]
    assert len(findings) == 1
    assert findings[0].path == "src/tgvf_rl/neutral.py"
    assert findings[0].evidence == {"imports": ["tgvf_rl.policy.rp67_legacy"]}


def test_canonical_configs_reject_run_names_and_machine_paths(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    config = repository / "configs/canonical/policy/rp99_trial.json"
    config.write_text('{"model": "/home/alice/model"}\n', encoding="utf-8")

    report = audit_repository_boundaries(repository, policy)

    assert {
        "run_specific_canonical_config",
        "machine_path_in_canonical_config",
    } <= _kinds(report, "violations")


def test_missing_or_symlinked_canonical_config_roots_fail_closed(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    missing = repository / "configs/canonical/representation"
    missing.rmdir()
    linked = repository / "configs/canonical/policy"
    linked.rmdir()
    linked.symlink_to(repository / "configs/ops", target_is_directory=True)

    report = audit_repository_boundaries(repository, policy)

    assert any(
        item.kind == "missing_boundary_root"
        and item.path == "configs/canonical/representation"
        for item in report.violations
    )
    assert any(
        item.kind == "symlink_boundary" and item.path == "configs/canonical/policy"
        for item in report.violations
    )


def test_evidence_only_inventory_addition_is_a_violation(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    (repository / "configs/evaluation/new-result.json").write_text(
        "{}\n", encoding="utf-8"
    )

    report = audit_repository_boundaries(repository, policy)

    finding = next(
        item
        for item in report.violations
        if item.kind == "evidence_config_inventory_drift"
    )
    assert finding.path == "configs/evaluation"
    assert finding.evidence["registered_file_count"] == 0
    assert finding.evidence["observed_file_count"] == 1


def test_evidence_only_content_change_is_a_violation(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    relative = "frozen.json"
    original = b'{"x":1}\n'
    changed = b'{"x":2}\n'
    assert (
        sum(left != right for left, right in zip(original, changed, strict=True)) == 1
    )
    config = repository / "configs/evaluation" / relative
    config.write_bytes(original)
    policy = _write_policy(
        repository,
        evidence_overrides={
            "configs/evaluation": (
                1,
                relative_path_inventory_sha256((relative,)),
                content_tree_inventory_sha256(((relative, original),)),
            )
        },
    )
    assert audit_repository_boundaries(repository, policy).status == "pass"

    config.write_bytes(changed)
    report = audit_repository_boundaries(repository, policy)

    finding = next(
        item
        for item in report.violations
        if item.kind == "evidence_config_inventory_drift"
    )
    assert finding.evidence["registered_file_count"] == 1
    assert finding.evidence["observed_file_count"] == 1
    assert (
        finding.evidence["registered_relative_paths_sha256"]
        == finding.evidence["observed_relative_paths_sha256"]
    )
    assert (
        finding.evidence["registered_content_tree_sha256"]
        != finding.evidence["observed_content_tree_sha256"]
    )


def test_evidence_only_path_change_is_still_a_violation(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    original_relative = "frozen.json"
    changed_relative = "renamed.json"
    content = b"{}\n"
    original = repository / "configs/evaluation" / original_relative
    original.write_bytes(content)
    policy = _write_policy(
        repository,
        evidence_overrides={
            "configs/evaluation": (
                1,
                relative_path_inventory_sha256((original_relative,)),
                content_tree_inventory_sha256(((original_relative, content),)),
            )
        },
    )
    assert audit_repository_boundaries(repository, policy).status == "pass"

    original.rename(repository / "configs/evaluation" / changed_relative)
    report = audit_repository_boundaries(repository, policy)

    finding = next(
        item
        for item in report.violations
        if item.kind == "evidence_config_inventory_drift"
    )
    assert finding.evidence["registered_file_count"] == 1
    assert finding.evidence["observed_file_count"] == 1
    assert (
        finding.evidence["registered_relative_paths_sha256"]
        != finding.evidence["observed_relative_paths_sha256"]
    )


def test_policy_requires_content_tree_digest(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    del payload["evidence_only_config_inventories"][0]["content_tree_sha256"]
    policy.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RepositoryBoundaryError, match="unexpected field set"):
        load_repository_boundary_policy(policy)


def test_content_tree_hash_uses_unambiguous_framing() -> None:
    assert content_tree_inventory_sha256((("a", b"bc"),)) != (
        content_tree_inventory_sha256((("ab", b"c"),))
    )


def test_stale_allowlists_and_occurrence_drift_fail_the_ratchet(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    literal = "/nvmesv/example-user/models/example"
    source_path = "src/tgvf_rl/neutral.py"
    (repository / source_path).write_text(
        f"FIRST = {literal!r}\nSECOND = {literal!r}\n", encoding="utf-8"
    )
    policy = _write_policy(
        repository,
        run_allowlist=("tools/run_rp67_removed.py",),
        machine_allowlist=(
            MachinePathDebt(
                source_path,
                sha256(literal.encode("utf-8")).hexdigest(),
                1,
            ),
        ),
    )

    report = audit_repository_boundaries(repository, policy)

    assert {
        "stale_run_specific_allowlist",
        "machine_path_count_drift",
    } <= _kinds(report, "violations")


def test_non_utf8_and_symlink_inputs_fail_closed(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    (repository / "tools/non_utf8.bin").write_bytes(b"\xff\xfe")
    target = tmp_path / "outside.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src/tgvf_rl/link.py").symlink_to(target)

    report = audit_repository_boundaries(repository, policy)

    assert {
        "non_utf8_canonical_file",
        "symlink_boundary",
    } <= _kinds(report, "violations")


def test_duplicate_policy_key_is_rejected(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = repository / "configs/ops/repository_boundary_policy.json"
    policy.write_text('{"schema_version": "x", "schema_version": "y"}\n')

    with pytest.raises(RepositoryBoundaryError, match="duplicate JSON key"):
        load_repository_boundary_policy(policy)


def test_cli_exit_status_and_json_are_fail_closed(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    policy = _write_policy(repository)
    command = [
        sys.executable,
        str(AUDIT_TOOL),
        "--repository-root",
        str(repository),
        "--policy",
        str(policy),
        "--compact",
    ]

    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "pass"

    (repository / "tools/run_rp99_new.py").write_text("VALUE = 1\n")
    blocked = subprocess.run(command, check=False, capture_output=True, text=True)
    assert blocked.returncode == 1
    payload = json.loads(blocked.stdout)
    assert payload["status"] == "blocked"
    assert payload["summary"]["violation_count"] == 1


def test_cli_error_uses_current_audit_schema(tmp_path: Path) -> None:
    missing_repository = tmp_path / "missing"

    failed = subprocess.run(
        [
            sys.executable,
            str(AUDIT_TOOL),
            "--repository-root",
            str(missing_repository),
            "--compact",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    payload = json.loads(failed.stderr)
    assert payload["schema_version"] == REPOSITORY_BOUNDARY_AUDIT_SCHEMA
    assert payload["status"] == "error"
