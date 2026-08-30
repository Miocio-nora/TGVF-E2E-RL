from __future__ import annotations

from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/ci.yml"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
COMPATIBILITY_LOCKS = (
    REPOSITORY_ROOT / "requirements/compatibility.lock",
    REPOSITORY_ROOT / "requirements/compatibility-torch211-cu129.lock",
)


def test_ci_linter_version_is_exactly_bound_to_compatibility_locks() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    test_dependencies = project["project"]["optional-dependencies"]["test"]

    assert "ruff==0.15.22" in test_dependencies
    assert not [item for item in test_dependencies if item.startswith("ruff>=")]
    for lock in COMPATIBILITY_LOCKS:
        assert "ruff==0.15.22" in lock.read_text(encoding="utf-8").splitlines()


def test_ci_compares_module_size_exceptions_with_a_v3_git_baseline() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert (
        "BOUNDARY_BASELINE_SHA: "
        "${{ github.event.pull_request.base.sha || github.event.before }}"
    ) in workflow
    assert (
        'git show "$BOUNDARY_BASELINE_SHA:configs/ops/repository_boundary_policy.json"'
    ) in workflow
    assert '"tgvf-repository-boundary-policy-v3"' in workflow
    assert 'audit_args+=(--baseline-policy "$BOUNDARY_BASELINE_POLICY")' in workflow


def test_ci_keeps_candidate_audit_mandatory_when_no_v3_baseline_exists() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python tools/audit_repository_boundaries.py" in workflow
    assert "--policy configs/ops/repository_boundary_policy.json" in workflow
    assert 'echo "BOUNDARY_BASELINE_POLICY=" >> "$GITHUB_ENV"' in workflow
