from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/ci.yml"


def test_ci_compares_module_size_exceptions_with_a_v3_git_baseline() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert (
        "BOUNDARY_BASELINE_SHA: "
        "${{ github.event.pull_request.base.sha || github.event.before }}"
    ) in workflow
    assert (
        'git show "$BOUNDARY_BASELINE_SHA:'
        'configs/ops/repository_boundary_policy.json"'
    ) in workflow
    assert '"tgvf-repository-boundary-policy-v3"' in workflow
    assert 'audit_args+=(--baseline-policy "$BOUNDARY_BASELINE_POLICY")' in workflow


def test_ci_keeps_candidate_audit_mandatory_when_no_v3_baseline_exists() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python tools/audit_repository_boundaries.py" in workflow
    assert "--policy configs/ops/repository_boundary_policy.json" in workflow
    assert 'echo "BOUNDARY_BASELINE_POLICY=" >> "$GITHUB_ENV"' in workflow
