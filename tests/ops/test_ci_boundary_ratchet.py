from __future__ import annotations

from pathlib import Path
import tomllib

from tgvf_rl.compatibility_stack import (
    CONTROL_COMPATIBILITY_STACK,
    audited_compatibility_stack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/ci.yml"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
CI_CONSTRAINTS = REPOSITORY_ROOT / "requirements/ci.constraints"
COMPATIBILITY_LOCKS = (
    REPOSITORY_ROOT / "requirements/compatibility.lock",
    REPOSITORY_ROOT / "requirements/compatibility-torch211-cu129.lock",
)


def _exact_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator == "==", f"CI constraint must be exact: {line}"
        normalized = name.lower()
        assert normalized not in requirements
        requirements[normalized] = version
    return requirements


def _locked_versions(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")) or "==" not in line:
            continue
        name, version = line.split("==", 1)
        requirements[name.lower()] = version
    return requirements


def test_ci_linter_version_is_exactly_bound_to_compatibility_locks() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    test_dependencies = project["project"]["optional-dependencies"]["test"]

    assert "ruff==0.15.22" in test_dependencies
    assert not [item for item in test_dependencies if item.startswith("ruff>=")]
    for lock in COMPATIBILITY_LOCKS:
        assert "ruff==0.15.22" in lock.read_text(encoding="utf-8").splitlines()


def test_ci_direct_dependencies_are_bound_to_the_primary_compatibility_lock() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    expected_names = {
        requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
        for group in (
            project["project"]["dependencies"],
            project["project"]["optional-dependencies"]["qwen"],
            project["project"]["optional-dependencies"]["test"],
        )
        for requirement in group
    }
    constraints = _exact_requirements(CI_CONSTRAINTS)

    assert set(constraints) == expected_names
    primary_lock = _locked_versions(COMPATIBILITY_LOCKS[0])
    assert constraints == {name: primary_lock[name] for name in expected_names}
    control_stack = audited_compatibility_stack(CONTROL_COMPATIBILITY_STACK)
    assert constraints["torch"] == control_stack.torch_distribution_version
    assert (
        constraints["transformers"] == control_stack.transformers_distribution_version
    )


def test_ci_install_uses_the_exact_direct_dependency_constraints() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "--constraint requirements/ci.constraints" in workflow
    assert "-e '.[test,qwen]'" in workflow


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
