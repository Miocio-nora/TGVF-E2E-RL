from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_QUARANTINED_CONTROLLERS = (
    "src/tgvf_rl/framework/verl/prl13_main.py",
    "spikes/verl_compat/fsdp2_smoke.py",
    "spikes/verl_compat/qwen3_patch_embed_probe.py",
    "spikes/verl_compat/qwen3_vllm_latent_smoke.py",
    "tools/analyze_prl13_group_signal.py",
    "tools/check_policy_rl_judge_routes.py",
    "tools/handoff_representation_image_axis_evaluation.py",
    "tools/judge_policy_data_selection_t1.py",
    "tools/launch_prl13_native_deepeyes.py",
    "tools/launch_prl14_cleanfinal16.py",
    "tools/launch_representation_answer_utility_evaluation.py",
    "tools/materialize_policy_selection_sources.py",
    "tools/materialize_policy_result_table.py",
    "tools/materialize_policy_t1_mixed_retained_pool.py",
    "tools/materialize_qwen3_grounding_manifest.py",
    "tools/materialize_representation_internal_evaluation_manifests.py",
    "tools/run_coredev_2511_vlmevalkit.py",
    "tools/run_overnight_pipeline.py",
    "tools/run_representation_oracle_d_utility.py",
    "tools/run_rp67_step2000_acc_pipeline.py",
    "tools/run_rp68_post_training_evaluations.py",
    "tools/run_rp69_post_training_evaluations.py",
    "tools/smoke_policy_live_tool_chain.py",
    "tools/smoke_prl13_qwen3_full_model_grad.py",
    "tools/smoke_qwen3_cached_continuation.py",
    "tools/smoke_qwen3_contextual_vision_reuse.py",
    "tools/smoke_qwen3_crop_tools.py",
    "tools/supervise_rp67_t1_schedule.py",
    "tools/watchdog_rp67_step2000_acc_pipeline.py",
    "tools/profile_representation_step.py",
    "tools/prove_policy_auto_resume.py",
    "tools/run_representation_answer_utility_semantic_rescore.py",
    "tools/smoke_qwen25_72b_judge.py",
    "tools/summarize_coredev_2511.py",
    "tools/validate_vlmevalkit_deployment.py",
    "tools/watch_representation_wandb.py",
)
LEGACY_QUARANTINED_SHELL_CONTROLLERS = (
    "tools/launch_policy_data_selection_t1_subshard.sh",
)
EXECUTABLE_LEGACY_PYTHON_CONTROLLERS = (
    "tools/launch_prl13_native_deepeyes.py",
    "tools/launch_prl14_cleanfinal16.py",
)
MIXED_MODE_STANDALONE_CONTROLLERS = (
    "spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py",
    "tools/audit_policy_data_selection_t1_replay.py",
    "tools/canary_qwen3_instruct_crop_prompt.py",
    "tools/control_local_qwen3_vl_32b_tgvf_visual_judge.py",
    "tools/compact_policy_checkpoint_storage.py",
    "tools/run_policy_benchmark.py",
    "tools/run_policy_coredev_2511.py",
    "tools/run_policy_data_selection_t1.py",
    "tools/run_policy_data_selection_t1_retry.py",
    "tools/run_representation_answer_utility.py",
    "tools/run_representation_answer_utility_evaluation.py",
    "tools/run_representation_image_axis_grounding.py",
    "tools/smoke_policy_data_selection_t1_resume.py",
)


def test_control_plane_audit_passes_and_reports_legacy_inventory() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert all(
        row["status"] == "migrated" for row in report["canonical_active_controllers"]
    )
    assert report["legacy_quarantined_controllers"]
    assert all(
        row["status"] == "legacy_quarantined_fail_closed"
        for row in report["legacy_quarantined_controllers"]
    )
    assert all(
        row["status"] == "legacy_quarantined_fail_closed"
        for row in report["legacy_quarantined_shell_controllers"]
    )
    assert all(
        row["status"] == "read_only_modes_only"
        for row in report["mixed_mode_standalone_controllers"]
    )
    assert report["canonical_active_controllers"] == [
        {
            "control_kind": "neutral_public_cli",
            "missing_markers": [],
            "path": "src/tgvf_rl/cli.py",
            "status": "migrated",
        }
    ]
    assert {row["command"]: row["status"] for row in report["public_cli_controls"]} == {
        "launch-representation": "migrated",
        "run-policy": "migrated",
        "run-representation": "migrated",
        "run-representation-internal-evaluation": "migrated",
    }


def test_stabilization_policy_v2_is_frozen_and_runtime_closed() -> None:
    policy = json.loads(
        (REPOSITORY_ROOT / "configs/ops/experiment_execution_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["schema_version"] == "tgvf-experiment-execution-policy-v2"
    assert policy["revision"] == 3
    assert policy["execution_mode"] == "frozen"
    assert policy["freeze_override"] == {
        "max_ttl_seconds": 3600,
        "reason_required": True,
        "required_when_frozen": True,
    }
    assert policy["runtime_closure"] == {
        "blocker_ids": [
            "atomic_authority_transaction_missing",
            "child_environment_allowlist_missing",
            "immutable_runtime_code_package_missing",
            "policy_recursive_compile_closure_missing",
            "representation_eval_safe_artifact_missing",
            "worker_member_claims_missing",
            "worker_startup_envelope_missing",
        ],
        "launch_enabled": False,
    }


def test_remaining_quarantined_shell_has_valid_syntax() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(REPOSITORY_ROOT / "tools/launch_policy_data_selection_t1_subshard.sh"),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_named_experiment_controllers_are_not_canonical() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    canonical_paths = {row["path"] for row in report["canonical_active_controllers"]}
    assert canonical_paths == {"src/tgvf_rl/cli.py"}


def test_policy_benchmark_status_and_validate_are_structurally_read_only() -> None:
    path = REPOSITORY_ROOT / "tools/run_policy_benchmark.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("_status", "_validate"):
        calls = {
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "policy_evaluation_identity" in calls
        assert "write_policy_evaluation_identity" not in calls
        assert "materialize_vllm_lora_adapter" not in calls


def _minimal_audit_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "configs/ops").mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "pyproject.toml", repository / "pyproject.toml")
    shutil.copy2(
        REPOSITORY_ROOT / "configs/ops/experiment_execution_policy.json",
        repository / "configs/ops/experiment_execution_policy.json",
    )
    surface_policy_path = REPOSITORY_ROOT / "configs/ops/execution_surface_policy.json"
    shutil.copy2(
        surface_policy_path,
        repository / "configs/ops/execution_surface_policy.json",
    )
    surface_policy = json.loads(surface_policy_path.read_text(encoding="utf-8"))
    for relative in (row["path"] for row in surface_policy["surfaces"]):
        (repository / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, repository / relative)
    return repository


def _refresh_surface_hash(repository: Path, relative: str) -> None:
    policy_path = repository / "configs/ops/execution_surface_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    rows = [row for row in policy["surfaces"] if row["path"] == relative]
    assert len(rows) == 1
    rows[0]["content_sha256"] = sha256((repository / relative).read_bytes()).hexdigest()
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_control_audit(repository: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "blocked"
    return report


def _direct_call_statement(
    branch: ast.If | ast.FunctionDef,
    call_name: str,
) -> ast.stmt:
    matches: list[ast.stmt] = []
    controlled_statements = list(branch.body)
    for statement in branch.body:
        if isinstance(statement, ast.Try):
            controlled_statements.extend(statement.body)
    for statement in controlled_statements:
        value: ast.expr | None = None
        if isinstance(statement, ast.Expr):
            value = statement.value
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == call_name
        ):
            matches.append(statement)
    assert len(matches) == 1
    return matches[0]


def _replace_statement_with_caught_call(source: str, statement: ast.stmt) -> str:
    lines = source.splitlines(keepends=True)
    original = "".join(lines[statement.lineno - 1 : statement.end_lineno])
    indentation = " " * statement.col_offset
    replacement = (
        f"{indentation}try:\n"
        + textwrap.indent(original, "    ")
        + f"{indentation}except Exception:\n"
        + f"{indentation}    pass\n"
    )
    return (
        "".join(lines[: statement.lineno - 1])
        + replacement
        + "".join(lines[statement.end_lineno :])
    )


@pytest.mark.parametrize(
    "call_name",
    [
        "_execute_representation_torchrun",
        "_execute_policy_run",
        "_run_representation_internal_evaluation",
        "verify_cli_worker_authorization_from_environment",
    ],
)
def test_control_plane_audit_blocks_each_unguarded_cli_dispatch(
    tmp_path: Path,
    call_name: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / "src/tgvf_rl/cli.py"
    source = cli_path.read_text(encoding="utf-8")
    main_offset = source.index("def main(")
    prefix, main_source = source[:main_offset], source[main_offset:]
    needle = f"{call_name}("
    assert main_source.count(needle) == 1
    cli_path.write_text(
        prefix + main_source.replace(needle, f"unsafe_{needle}", 1),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "blocked"
    assert any("public CLI control" in item for item in report["violations"])


@pytest.mark.parametrize(
    ("command", "guard"),
    (
        ("launch-representation", "_consume_command_authorization"),
        ("run-policy", "_consume_command_authorization"),
        (
            "run-representation-internal-evaluation",
            "_consume_command_authorization",
        ),
        (
            "run-representation",
            "verify_cli_worker_authorization_from_environment",
        ),
    ),
)
def test_control_plane_audit_blocks_caught_public_cli_guard_after_hash_refresh(
    tmp_path: Path,
    command: str,
    guard: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative)
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == command
    ]
    assert len(branches) == 1
    statement = _direct_call_statement(branches[0], guard)
    cli_path.write_text(
        _replace_statement_with_caught_call(source, statement),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        f"public CLI control {command}" in item and "uncaught direct" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize(
    ("command", "dispatch"),
    (
        ("launch-representation", "_execute_representation_torchrun"),
        ("run-policy", "_execute_policy_run"),
    ),
)
@pytest.mark.parametrize(
    "mutation",
    ("caught-dispatch", "extra-try-effect", "extra-finally", "wrong-finally"),
)
def test_control_plane_audit_enforces_exact_prepared_lifetime_try_finally(
    tmp_path: Path,
    command: str,
    dispatch: str,
    mutation: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    branch = _cli_command_branch(source, command)
    lifetime_tries = [
        statement for statement in branch.body if isinstance(statement, ast.Try)
    ]
    assert len(lifetime_tries) == 1
    lifetime_try = lifetime_tries[0]
    dispatch_statement = _direct_call_statement(branch, dispatch)
    if mutation == "caught-dispatch":
        mutated = _replace_statement_with_caught_call(source, dispatch_statement)
    elif mutation == "extra-try-effect":
        mutated = _insert_source_before(
            source,
            dispatch_statement,
            "os.system('unsafe-between-authorization-and-dispatch')",
        )
    elif mutation == "extra-finally":
        assert len(lifetime_try.finalbody) == 1
        mutated = _insert_source_after(
            source,
            lifetime_try.finalbody[0],
            "os.system('unsafe-finalizer-effect')",
        )
    else:
        assert len(lifetime_try.finalbody) == 1
        finalizer = lifetime_try.finalbody[0]
        lines = source.splitlines(keepends=True)
        original = "".join(lines[finalizer.lineno - 1 : finalizer.end_lineno])
        assert original.count("prepared.close_python_binding()") == 1
        mutated = (
            "".join(lines[: finalizer.lineno - 1])
            + original.replace(
                "prepared.close_python_binding()",
                "prepared.abandon_python_binding()",
                1,
            )
            + "".join(lines[finalizer.end_lineno :])
        )
    cli_path.write_text(mutated, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        f"public CLI control {command}" in item
        and "exact fail-closed branch statement/call shape" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize(
    "preflight",
    (
        "_preflight_representation_launch",
        "_preflight_policy_run",
        "_preflight_representation_internal_evaluation",
    ),
)
@pytest.mark.parametrize("mutation", ("delete", "caught", "nested-side-effect"))
def test_control_plane_audit_requires_uncaught_first_preflight_runtime_closure(
    tmp_path: Path,
    preflight: str,
    mutation: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative)
    functions = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == preflight
    ]
    assert len(functions) == 1
    closure = _direct_call_statement(
        functions[0],
        "assert_canonical_runtime_launch_enabled",
    )
    if mutation == "caught":
        mutated = _replace_statement_with_caught_call(source, closure)
    elif mutation == "nested-side-effect":
        lines = source.splitlines(keepends=True)
        original = "".join(lines[closure.lineno - 1 : closure.end_lineno])
        assert original.count("assert_canonical_runtime_launch_enabled()") == 1
        mutated = (
            "".join(lines[: closure.lineno - 1])
            + original.replace(
                "assert_canonical_runtime_launch_enabled()",
                "assert_canonical_runtime_launch_enabled(os.system('unsafe'))",
                1,
            )
            + "".join(lines[closure.end_lineno :])
        )
    else:
        lines = source.splitlines(keepends=True)
        mutated = (
            "".join(lines[: closure.lineno - 1])
            + " " * closure.col_offset
            + "pass\n"
            + "".join(lines[closure.end_lineno :])
        )
    cli_path.write_text(mutated, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        f"{preflight} starts with direct runtime-closure assertion" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_CONTROLLERS)
def test_control_plane_audit_blocks_legacy_tool_without_first_statement_guard(
    tmp_path: Path,
    relative: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    needle = "assert_legacy_standalone_execution_quarantined("
    assert source.count(needle) == 1
    main_start = source.index("def main")
    prefix, main_source = source[:main_start], source[main_start:]
    tool_path.write_text(
        prefix + main_source.replace(needle, "bypass_legacy_quarantine(", 1),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "blocked"
    assert any(relative in item for item in report["violations"])
    row = next(
        item
        for item in report["legacy_quarantined_controllers"]
        if item["path"] == relative
    )
    assert row["status"] == "legacy_quarantine_bypassable"


@pytest.mark.parametrize("relative", MIXED_MODE_STANDALONE_CONTROLLERS)
def test_control_plane_audit_blocks_mixed_tool_without_mode_guard(
    tmp_path: Path,
    relative: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    needle = "assert_legacy_standalone_mode_quarantined("
    assert source.count(needle) == 1
    main_start = source.index("def main")
    prefix, main_source = source[:main_start], source[main_start:]
    tool_path.write_text(
        prefix + main_source.replace(needle, "bypass_legacy_mode_guard(", 1),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(relative in item for item in report["violations"])


def test_control_plane_audit_blocks_caught_mixed_guard_after_hash_refresh(
    tmp_path: Path,
) -> None:
    relative = "tools/run_policy_data_selection_t1_retry.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative)
    main = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "main"
    )
    guard = next(
        statement
        for statement in main.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "assert_legacy_standalone_mode_quarantined"
    )
    lines = source.splitlines(keepends=True)
    original = "".join(lines[guard.lineno - 1 : guard.end_lineno])
    replacement = (
        "    try:\n"
        + textwrap.indent(original, "    ")
        + "    except Exception:\n"
        + "        pass\n"
    )
    tool_path.write_text(
        "".join(lines[: guard.lineno - 1])
        + replacement
        + "".join(lines[guard.end_lineno :]),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "uncaught direct" in item for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_fake_parse_args_receiver_after_hash_refresh(
    tmp_path: Path,
) -> None:
    relative = "spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    trusted = "    args = _parser().parse_args(argv)\n"
    assert source.count(trusted) == 1
    tool_path.write_text(
        source.replace(trusted, "    args = attacker.parse_args(argv)\n", 1),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "trusted argparse parse" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_SHELL_CONTROLLERS)
def test_control_plane_audit_blocks_shell_without_exec_quarantine(
    tmp_path: Path,
    relative: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    assert source.count("quarantine-legacy") == 1
    tool_path.write_text(
        source.replace("quarantine-legacy", "bypass-quarantine", 1),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(relative in item for item in report["violations"])


@pytest.mark.parametrize("marker_kind", ("comment", "heredoc"))
def test_control_plane_audit_rejects_non_executable_shell_markers_after_hash_refresh(
    tmp_path: Path,
    marker_kind: str,
) -> None:
    relative = "tools/launch_policy_data_selection_t1_subshard.sh"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    marker = (
        'exec /usr/bin/python3 -I -- "$script_directory/check_launch_gate.py" '
        "quarantine-legacy \\\n"
        f'  --tool-id "{relative}"\n'
    )
    if marker_kind == "comment":
        payload = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "# "
            + marker.replace("\n", "\n# ")
            + "/usr/bin/printf 'unsafe dispatch\\n'\n"
        )
    else:
        payload = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            ": <<'QUARANTINE_MARKER'\n" + marker + "QUARANTINE_MARKER\n"
            "/usr/bin/printf 'unsafe dispatch\\n'\n"
        )
    tool_path.write_text(payload, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "exact fixed quarantine-wrapper bytes" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_SHELL_CONTROLLERS)
def test_shell_quarantine_runs_before_hostile_environment_hooks(
    tmp_path: Path,
    relative: str,
) -> None:
    attacker_directory = tmp_path / "attacker-bin"
    attacker_directory.mkdir()
    path_marker = tmp_path / "path-hook-ran"
    bash_env_marker = tmp_path / "bash-env-ran"
    pythonpath_marker = tmp_path / "pythonpath-ran"
    for name in ("bash", "dirname"):
        attacker = attacker_directory / name
        attacker.write_text(
            "#!/bin/sh\n" + f"/usr/bin/touch {path_marker!s}\n" + "exit 91\n",
            encoding="utf-8",
        )
        attacker.chmod(0o755)
    bash_environment = tmp_path / "bash-environment.sh"
    bash_environment.write_text(
        f"/usr/bin/touch {bash_env_marker!s}\n",
        encoding="utf-8",
    )
    python_module = attacker_directory / "argparse.py"
    python_module.write_text(
        f"open({str(pythonpath_marker)!r}, 'w').close()\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "BASH_ENV": str(bash_environment),
            "PATH": str(attacker_directory),
            "PYTHONPATH": str(attacker_directory),
        }
    )

    completed = subprocess.run(
        [str(REPOSITORY_ROOT / relative)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert "remains quarantined" in completed.stderr or "is quarantined" in (
        completed.stderr
    )
    assert not path_marker.exists()
    assert not bash_env_marker.exists()
    assert not pythonpath_marker.exists()


def _hostile_entrypoint_environment(
    tmp_path: Path,
) -> tuple[dict[str, str], dict[str, Path]]:
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    markers = {
        "path": tmp_path / "path-python-ran",
        "sitecustomize": tmp_path / "sitecustomize-ran",
        "bash_env": tmp_path / "bash-env-ran",
        "adjacent_controller": tmp_path / "adjacent-controller-ran",
    }
    fake_python = attacker_directory / "python3"
    fake_python.write_text(
        "#!/bin/sh\n" + f"/usr/bin/touch {markers['path']!s}\n" + "exit 91\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    (attacker_directory / "sitecustomize.py").write_text(
        f"open({str(markers['sitecustomize'])!r}, 'w').close()\n",
        encoding="utf-8",
    )
    bash_environment = tmp_path / "hostile-bash-environment.sh"
    bash_environment.write_text(
        f"/usr/bin/touch {markers['bash_env']!s}\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "BASH_ENV": str(bash_environment),
            "PATH": str(attacker_directory),
            "PYTHONPATH": str(attacker_directory),
        }
    )
    return environment, markers


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_CONTROLLERS)
def test_isolated_python_quarantine_ignores_hostile_startup_environment(
    tmp_path: Path,
    relative: str,
) -> None:
    environment, markers = _hostile_entrypoint_environment(tmp_path)

    completed = subprocess.run(
        ["/usr/bin/python3", "-I", str(REPOSITORY_ROOT / relative)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert "remains quarantined" in completed.stderr or "is quarantined" in (
        completed.stderr
    )
    assert not any(marker.exists() for marker in markers.values())


@pytest.mark.parametrize("relative", EXECUTABLE_LEGACY_PYTHON_CONTROLLERS)
def test_fixed_isolated_shebang_ignores_path_and_pythonpath_hooks(
    tmp_path: Path,
    relative: str,
) -> None:
    environment, markers = _hostile_entrypoint_environment(tmp_path)
    controller = REPOSITORY_ROOT / relative
    assert controller.stat().st_mode & 0o111

    completed = subprocess.run(
        [str(controller)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert not any(marker.exists() for marker in markers.values())


def test_plain_python_startup_is_an_explicit_external_trust_boundary(
    tmp_path: Path,
) -> None:
    environment, markers = _hostile_entrypoint_environment(tmp_path)

    completed = subprocess.run(
        [
            "/usr/bin/python3",
            str(REPOSITORY_ROOT / "tools/launch_prl14_cleanfinal16.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert markers["sitecustomize"].is_file()
    assert not markers["path"].exists()
    assert not markers["adjacent_controller"].exists()


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_SHELL_CONTROLLERS)
def test_shell_quarantine_rejects_final_symlink_before_adjacent_attack(
    tmp_path: Path,
    relative: str,
) -> None:
    environment, markers = _hostile_entrypoint_environment(tmp_path)
    linked_controller = tmp_path / Path(relative).name
    linked_controller.symlink_to(REPOSITORY_ROOT / relative)
    (tmp_path / "check_launch_gate.py").write_text(
        f"open({str(markers['adjacent_controller'])!r}, 'w').close()\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [str(linked_controller)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert "refusing symlinked legacy shell controller" in completed.stderr
    assert not any(marker.exists() for marker in markers.values())


@pytest.mark.parametrize("relative", LEGACY_QUARANTINED_SHELL_CONTROLLERS)
def test_shell_quarantine_resolves_symlinked_ancestor_to_repository(
    tmp_path: Path,
    relative: str,
) -> None:
    environment, markers = _hostile_entrypoint_environment(tmp_path)
    tools_alias = tmp_path / "tools-alias"
    tools_alias.symlink_to(REPOSITORY_ROOT / "tools", target_is_directory=True)
    controller = tools_alias / Path(relative).name

    completed = subprocess.run(
        [str(controller)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr or completed.stdout
    assert "remains quarantined" in completed.stderr or "is quarantined" in (
        completed.stderr
    )
    assert not any(marker.exists() for marker in markers.values())


def test_control_plane_audit_rejects_nonisolated_python_second_hop(
    tmp_path: Path,
) -> None:
    relative = "tools/launch_prl14_cleanfinal16.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    isolated_argument = '            "-I",\n'
    assert source.count(isolated_argument) == 1
    tool_path.write_text(
        source.replace(isolated_argument, '            "-E",\n', 1),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "before all runtime imports" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


def test_control_plane_audit_rejects_path_resolved_python_shebang(
    tmp_path: Path,
) -> None:
    relative = "tools/launch_prl14_cleanfinal16.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/python3 -I\n")
    tool_path.write_text(
        source.replace("#!/usr/bin/python3 -I\n", "#!/usr/bin/env python3\n", 1),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "fixed isolated Python shebang" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize(
    "preguard_source",
    ("import torch\n", "dispatch_before_quarantine()\n"),
)
def test_control_plane_audit_blocks_permanent_preguard_runtime_or_side_effect(
    tmp_path: Path,
    preguard_source: str,
) -> None:
    relative = "tools/launch_prl14_cleanfinal16.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    assert source.count(future) == 1
    tool_path.write_text(
        source.replace(future, future + preguard_source, 1),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "before all runtime imports" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_caught_top_level_main_dispatch(
    tmp_path: Path,
) -> None:
    relative = "tools/analyze_prl13_group_signal.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    direct = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
    assert source.count(direct) == 1
    caught = (
        'if __name__ == "__main__":\n'
        "    try:\n"
        "        raise SystemExit(main())\n"
        "    except Exception:\n"
        "        pass\n"
        "    dispatch_after_caught_guard()\n"
    )
    tool_path.write_text(source.replace(direct, caught, 1), encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "uncaught direct top-level main dispatch" in item
        for item in report["violations"]
    )
    assert not any(
        relative in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_unknown_gpu_surface(tmp_path: Path) -> None:
    repository = _minimal_audit_repository(tmp_path)
    unknown = repository / "tools/new_unclassified_gpu_entry.py"
    unknown.write_text(
        "def main():\n    return torch.cuda.device_count()\n\n"
        "if __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(
        "unmanifested execution surface" in item
        and "new_unclassified_gpu_entry.py" in item
        for item in report["violations"]
    )


def test_execution_surface_manifest_is_exact_and_content_bound() -> None:
    policy = json.loads(
        (REPOSITORY_ROOT / "configs/ops/execution_surface_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["schema_version"] == "tgvf-execution-surface-policy-v2"
    assert policy["revision"] == 4
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    inventory = report["execution_surface_inventory"]
    assert len(inventory) == 79
    assert all(row["status"] == "content_bound" for row in inventory)
    assert [row["path"] for row in inventory] == sorted(
        row["path"] for row in inventory
    )
    assert all(
        row["authorization_kind"] == "bounded_artifact_write"
        for row in inventory
        if row["classification"] == "artifact_materializer"
    )
    support = next(
        row for row in inventory if row["classification"] == "import_only_support"
    )
    assert support["path"] == "spikes/verl_compat/verl_sync_fixed_reward.py"
    check_gate = next(
        row for row in inventory if row["path"] == "tools/check_launch_gate.py"
    )
    assert check_gate["allowed_modes"] == [
        "audit-control-plane",
        "authorize",
        "consume",
        "override-freeze",
        "quarantine-legacy",
        "ready",
        "status",
        "wait",
    ]


def test_control_plane_audit_blocks_execution_surface_content_drift(
    tmp_path: Path,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    target = repository / "tools/benchmark_prl13_schedule_index.py"
    target.write_bytes(target.read_bytes() + b"\n# one-byte-class drift\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(
        "benchmark_prl13_schedule_index.py" in item and "content SHA256 differs" in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_missing_manifest_row(tmp_path: Path) -> None:
    repository = _minimal_audit_repository(tmp_path)
    policy_path = repository / "configs/ops/execution_surface_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    removed = policy["surfaces"].pop()
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(
        item == f"unmanifested execution surface discovered: {removed['path']}"
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_extra_manifest_row(tmp_path: Path) -> None:
    repository = _minimal_audit_repository(tmp_path)
    policy_path = repository / "configs/ops/execution_surface_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    extra = dict(policy["surfaces"][-1])
    extra["path"] = "tools/removed_historical_entry.py"
    policy["surfaces"].append(extra)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(
        item
        == "manifested execution surface is missing: tools/removed_historical_entry.py"
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_recursive_shell_surface(tmp_path: Path) -> None:
    repository = _minimal_audit_repository(tmp_path)
    unknown = repository / "tools/nested/new_entry.sh"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/check_launch_gate.py"),
            "audit-control-plane",
            "--repository-root",
            str(repository),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert any(
        item == "unmanifested execution surface discovered: tools/nested/new_entry.sh"
        for item in report["violations"]
    )


def _cli_command_branch(source: str, command: str) -> ast.If:
    tree = ast.parse(source, filename="src/tgvf_rl/cli.py")
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == command
    ]
    assert len(matches) == 1
    return matches[0]


def _insert_source_before(source: str, statement: ast.stmt, payload: str) -> str:
    lines = source.splitlines(keepends=True)
    return (
        "".join(lines[: statement.lineno - 1])
        + " " * statement.col_offset
        + payload
        + "\n"
        + "".join(lines[statement.lineno - 1 :])
    )


def _insert_source_after(source: str, statement: ast.stmt, payload: str) -> str:
    lines = source.splitlines(keepends=True)
    return (
        "".join(lines[: statement.end_lineno])
        + " " * statement.col_offset
        + payload
        + "\n"
        + "".join(lines[statement.end_lineno :])
    )


@pytest.mark.parametrize(
    "placement",
    ("main-prefix", "branch-before-closure", "branch-after-closure"),
)
def test_control_plane_audit_blocks_public_cli_preguard_side_effects(
    tmp_path: Path,
    placement: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    if placement == "main-prefix":
        marker = "    result: object | None = None\n"
        assert source.count(marker) == 1
        mutated = source.replace(
            marker,
            marker + "    os.system('unsafe-before-command-guard')\n",
            1,
        )
    else:
        branch = _cli_command_branch(source, "launch-representation")
        closure = branch.body[0]
        if placement == "branch-before-closure":
            mutated = _insert_source_before(
                source, closure, "os.system('unsafe-before-closure')"
            )
        else:
            mutated = _insert_source_after(
                source, closure, "os.system('unsafe-before-preflight')"
            )
    cli_path.write_text(mutated, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "public CLI structure" in item
        or "exact fail-closed branch statement/call shape" in item
        or "runtime-closure assertion as the first" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize(
    ("command", "call_name"),
    (
        ("launch-representation", "assert_canonical_runtime_launch_enabled"),
        ("run-policy", "_preflight_policy_run"),
    ),
)
def test_control_plane_audit_blocks_nested_side_effect_in_guard_or_preflight(
    tmp_path: Path,
    command: str,
    call_name: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    branch = _cli_command_branch(source, command)
    statement = _direct_call_statement(branch, call_name)
    segment = ast.get_source_segment(source, statement)
    assert segment is not None and segment.count(f"{call_name}(") == 1
    replacement = segment.replace(
        f"{call_name}(",
        f"{call_name}(os.system('unsafe-nested-effect'), ",
        1,
    )
    lines = source.splitlines(keepends=True)
    cli_path.write_text(
        "".join(lines[: statement.lineno - 1])
        + " " * statement.col_offset
        + replacement
        + ("\n" if not replacement.endswith("\n") else "")
        + "".join(lines[statement.end_lineno :]),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "exact fail-closed branch statement/call shape" in item
        or "runtime-closure assertion as the first" in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_public_control_lambda_rebind(
    tmp_path: Path,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    marker = "\ndef main(argv: Sequence[str] | None = None) -> int:\n"
    assert source.count(marker) == 1
    cli_path.write_text(
        source.replace(
            marker,
            "\n_consume_command_authorization = lambda *_args, **_kwargs: None\n"
            + marker,
            1,
        ),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "_consume_command_authorization provenance" in item
        and "Store/Delete/shadow/rebind" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("mutation", ("rebind", "wrong-import"))
def test_control_plane_audit_requires_exact_cli_launch_dispatch_imports(
    tmp_path: Path,
    mutation: str,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    if mutation == "rebind":
        marker = "\ndef main(argv: Sequence[str] | None = None) -> int:\n"
        assert source.count(marker) == 1
        mutated = source.replace(
            marker,
            "\n_execute_policy_run = lambda *_args, **_kwargs: None\n" + marker,
            1,
        )
    else:
        marker = "from tgvf_rl.ops.cli_launch import ("
        assert source.count(marker) == 1
        mutated = source.replace(
            marker,
            "from attacker_control import (",
            1,
        )
    cli_path.write_text(mutated, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "_execute_policy_run provenance" in item
        and (
            "Store/Delete/shadow/rebind" in item
            or "direct import from tgvf_rl.ops.cli_launch" in item
        )
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_internal_worker_guard_argument_effect(
    tmp_path: Path,
) -> None:
    relative = "src/tgvf_rl/cli.py"
    repository = _minimal_audit_repository(tmp_path)
    cli_path = repository / relative
    source = cli_path.read_text(encoding="utf-8")
    marker = "expected_phase=REPRESENTATION_TRAINING_PHASE,"
    assert source.count(marker) == 1
    cli_path.write_text(
        source.replace(
            marker,
            "expected_phase=(os.system('unsafe-worker-effect') or "
            "REPRESENTATION_TRAINING_PHASE),",
            1,
        ),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "run-representation" in item
        and "exact inherited-worker guard/closure call shape" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("mutation", ("rebind", "wrong-import", "parser-rebind"))
def test_control_plane_audit_blocks_mixed_guard_provenance_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    relative = "tools/run_policy_data_selection_t1_retry.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    if mutation in {"rebind", "parser-rebind"}:
        marker = "\ndef main() -> None:\n"
        assert source.count(marker) == 1
        rebound = (
            "assert_legacy_standalone_mode_quarantined"
            if mutation == "rebind"
            else "_parser"
        )
        mutated = source.replace(
            marker,
            f"\n{rebound} = lambda *a, **k: None\n" + marker,
            1,
        )
    else:
        marker = "from tgvf_rl.ops.cli_authorization import ("
        assert source.count(marker) == 1
        mutated = source.replace(
            marker,
            "from attacker_control import (",
            1,
        )
    tool_path.write_text(mutated, encoding="utf-8")
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "provenance" in item for item in report["violations"]
    )


@pytest.mark.parametrize(
    "candidate_kind",
    (
        "tool-side-effect-no-main",
        "src-executable-no-main",
        "src-main-module",
        "src-python-shebang",
        "console-entrypoint",
    ),
)
def test_control_plane_audit_discovers_non_main_execution_candidates(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    if candidate_kind == "tool-side-effect-no-main":
        relative = "tools/unlisted_import_side_effect.py"
        payload = "import os\nos.system('unsafe-import-effect')\n"
    elif candidate_kind == "src-executable-no-main":
        relative = "src/tgvf_rl/unlisted_executable.py"
        payload = "VALUE = 1\n"
    elif candidate_kind == "src-main-module":
        relative = "src/tgvf_rl/unlisted_package/__main__.py"
        payload = "VALUE = 1\n"
    elif candidate_kind == "src-python-shebang":
        relative = "src/tgvf_rl/unlisted_shebang.py"
        payload = "#!/usr/bin/python3 -I\nVALUE = 1\n"
    else:
        relative = "src/tgvf_rl/unlisted_console.py"
        payload = "def main():\n    return 0\n"
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    if candidate_kind == "src-executable-no-main":
        path.chmod(0o755)
    if candidate_kind == "console-entrypoint":
        pyproject_path = repository / "pyproject.toml"
        pyproject = pyproject_path.read_text(encoding="utf-8")
        marker = 'tgvf-rl = "tgvf_rl.cli:main"\n'
        assert pyproject.count(marker) == 1
        pyproject_path.write_text(
            pyproject.replace(
                marker,
                marker + 'unlisted-console = "tgvf_rl.unlisted_console:main"\n',
                1,
            ),
            encoding="utf-8",
        )

    report = _run_control_audit(repository)

    assert any(
        item == f"unmanifested execution surface discovered: {relative}"
        for item in report["violations"]
    )


@pytest.mark.parametrize(
    "payload",
    ("import os\n", "import os\nos.system('unsafe-import-effect')\n"),
)
def test_control_plane_audit_blocks_import_only_support_side_effect(
    tmp_path: Path,
    payload: str,
) -> None:
    relative = "spikes/verl_compat/verl_sync_fixed_reward.py"
    repository = _minimal_audit_repository(tmp_path)
    support_path = repository / relative
    source = support_path.read_text(encoding="utf-8")
    marker = "from __future__ import annotations\n"
    assert source.count(marker) == 1
    support_path.write_text(
        source.replace(
            marker,
            marker + "\n" + payload,
            1,
        ),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        relative in item and "import-time executable syntax" in item
        for item in report["violations"]
    )


@pytest.mark.parametrize("drift", ("allowed-mode", "capability"))
def test_control_plane_audit_binds_check_gate_policy_to_parser_and_capabilities(
    tmp_path: Path,
    drift: str,
) -> None:
    repository = _minimal_audit_repository(tmp_path)
    policy_path = repository / "configs/ops/execution_surface_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in policy["surfaces"]
        if item["path"] == "tools/check_launch_gate.py"
    )
    if drift == "allowed-mode":
        row["allowed_modes"].remove("wait")
    else:
        row["capabilities"]["network"] = True
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    report = _run_control_audit(repository)

    expected = (
        "allowed_modes differ" if drift == "allowed-mode" else "capabilities differ"
    )
    assert any(
        "tools/check_launch_gate.py" in item and expected in item
        for item in report["violations"]
    )


def test_control_plane_audit_blocks_check_gate_parser_mode_drift(
    tmp_path: Path,
) -> None:
    relative = "tools/check_launch_gate.py"
    repository = _minimal_audit_repository(tmp_path)
    tool_path = repository / relative
    source = tool_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative)
    parser = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "_parser"
    )
    returned = parser.body[-1]
    assert isinstance(returned, ast.Return)
    tool_path.write_text(
        _insert_source_before(
            source,
            returned,
            'commands.add_parser("unreviewed-control-mode")',
        ),
        encoding="utf-8",
    )
    _refresh_surface_hash(repository, relative)

    report = _run_control_audit(repository)

    assert any(
        "parser modes differ from the audited control contract" in item
        for item in report["violations"]
    )
