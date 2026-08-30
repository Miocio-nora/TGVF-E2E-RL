#!/usr/bin/env python3
"""Create, authorize, consume, inspect, and audit fail-closed launch gates."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import sys
import tomllib
from typing import NoReturn, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.launch_gate import (  # noqa: E402
    LaunchGateError,
    consume_launch_authorization,
    gate_status,
    issue_freeze_override,
    issue_launch_authorization,
    make_run_identity,
    materialize_ready_receipt,
    wait_for_artifact,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


DEFAULT_EXECUTION_POLICY = (
    REPOSITORY_ROOT / "configs/ops/experiment_execution_policy.json"
)
DEFAULT_EXECUTION_SURFACE_POLICY = (
    REPOSITORY_ROOT / "configs/ops/execution_surface_policy.json"
)
EXECUTION_SURFACE_POLICY_SCHEMA = "tgvf-execution-surface-policy-v2"
EXECUTION_SURFACE_POLICY_REVISION = 5
EXECUTION_SURFACE_CONTENT_BINDING = "sha256-file-bytes-v1"
EXECUTION_SURFACE_ROOTS = ("src", "tools", "spikes")
EXECUTION_SHELL_ROOTS = ("tools", "spikes")
EXECUTION_SURFACE_CAPABILITIES = frozenset(
    {"gpu", "network", "subprocess", "arbitrary_exec", "destructive_fs"}
)
EXECUTION_SURFACE_CLASSIFICATIONS = frozenset(
    {
        "artifact_materializer",
        "canonical_internal_worker",
        "canonical_public_cli",
        "control_utility",
        "import_only_support",
        "mixed_read_only_and_quarantined",
        "permanent_quarantine",
        "read_only_utility",
    }
)
EXECUTION_SURFACE_AUTHORIZATION_KINDS = frozenset(
    {
        "bounded_artifact_write",
        "canonical_run_bound",
        "inherited_worker_receipt",
        "none_offline",
        "permanent_quarantine",
        "read_only_mode_guard",
        "repository_control",
    }
)
HIGH_RISK_WORKLOAD_KINDS = frozenset(
    {"evaluation", "external_judge", "gpu_probe", "training"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Only the neutral package CLI is canonical during stabilization.  Named
# PRL/RP controllers remain evidence, never alternate execution authorities.
CANONICAL_ACTIVE_CONTROLLERS = {
    "src/tgvf_rl/cli.py": "neutral_public_cli",
}
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
MIXED_MODE_STANDALONE_CONTROLLERS = {
    "spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py": (
        ("plan",),
        ("launch-gpu",),
        "launch_gpu",
    ),
    "tools/audit_policy_data_selection_t1_replay.py": (
        ("plan",),
        ("run",),
        "command",
    ),
    "tools/canary_qwen3_instruct_crop_prompt.py": (
        ("validate",),
        ("execute",),
        "validate_only",
    ),
    "tools/control_local_qwen3_vl_32b_tgvf_visual_judge.py": (
        ("preflight", "status", "wait"),
        ("canary", "launch", "stop"),
        "action",
    ),
    "tools/compact_policy_checkpoint_storage.py": (
        ("inventory",),
        ("compact", "delete-non-scientific"),
        "command",
    ),
    "tools/run_policy_benchmark.py": (
        ("status", "validate"),
        ("prepare", "worker"),
        "mode",
    ),
    "tools/run_policy_coredev_2511.py": (
        ("status",),
        ("prepare", "worker"),
        "mode",
    ),
    "tools/run_policy_data_selection_t1.py": (
        ("status",),
        ("prepare", "worker"),
        "command",
    ),
    "tools/run_policy_data_selection_t1_retry.py": (
        ("plan",),
        ("worker",),
        "command",
    ),
    "tools/run_representation_answer_utility.py": (
        ("validate",),
        ("execute",),
        "validate_only",
    ),
    "tools/run_representation_answer_utility_evaluation.py": (
        ("validate",),
        ("execute",),
        "validate_only",
    ),
    "tools/run_representation_image_axis_grounding.py": (
        ("validate",),
        ("execute",),
        "validate_only",
    ),
    "tools/smoke_policy_data_selection_t1_resume.py": (
        ("plan",),
        ("baseline", "interrupt", "resume"),
        "command",
    ),
}
PUBLIC_MUTATING_CLI_CONTROLS = {
    "launch-representation": (
        "_consume_command_authorization",
        "_execute_representation_torchrun",
    ),
    "run-policy": (
        "_consume_command_authorization",
        "_execute_policy_run",
    ),
    "run-representation-internal-evaluation": (
        "_consume_command_authorization",
        "_run_representation_internal_evaluation",
    ),
}
PUBLIC_MUTATING_CLI_PREFLIGHTS = {
    "launch-representation": "_preflight_representation_launch",
    "run-policy": "_preflight_policy_run",
    "run-representation-internal-evaluation": (
        "_preflight_representation_internal_evaluation"
    ),
}
PUBLIC_MUTATING_CLI_FIRST_GUARD = "assert_canonical_runtime_launch_enabled"
PUBLIC_MUTATING_CLI_BRANCH_CONTRACTS = {
    "run-representation-internal-evaluation": (
        (None, ("assert_canonical_runtime_launch_enabled",)),
        ("config_binding", ("bind_canonical_config_path",)),
        ("config", ("_load_representation_internal_evaluation_config",)),
        (None, ("assert_loaded_config_matches_binding",)),
        ("python_identity", ("bind_current_python_executable",)),
        (None, ("_preflight_representation_internal_evaluation",)),
        ("identity", ("_representation_internal_evaluation_authorization_identity",)),
        (None, ("_consume_command_authorization",)),
        (None, ("verify_python_executable_identity",)),
        ("result", ("_run_representation_internal_evaluation",)),
    ),
}
PUBLIC_MUTATING_CLI_PREPARED_LIFETIME_CONTRACTS = {
    "launch-representation": (
        (
            (None, ("assert_canonical_runtime_launch_enabled",)),
            ("config_binding", ("bind_canonical_config_path",)),
            ("config", ("load_representation_training_config",)),
            ("prepared", ("_preflight_representation_launch",)),
        ),
        (
            ("identity", ("_representation_training_authorization_identity",)),
            ("consumption", ("_consume_command_authorization",)),
            ("worker_authorization", ("materialize_cli_worker_authorization",)),
            (None, ("_execute_representation_torchrun",)),
        ),
    ),
    "run-policy": (
        (
            (None, ("assert_canonical_runtime_launch_enabled",)),
            ("config_binding", ("bind_canonical_config_path",)),
            ("config", ("_load_policy_run_config",)),
            (None, ("assert_loaded_config_matches_binding",)),
            ("@extension-none", ()),
            ("@horizon-extension-if", ("_load_policy_horizon_extension",)),
            ("prepared", ("_preflight_policy_run",)),
        ),
        (
            (
                None,
                ("_assert_installed_stack_identity", "audited_compatibility_stack"),
            ),
            (
                None,
                ("verify_verl_distribution_identity", "audited_compatibility_stack"),
            ),
            ("identity", ("_policy_training_authorization_identity",)),
            ("consumption", ("_consume_command_authorization",)),
            ("worker_authorization", ("materialize_cli_worker_authorization",)),
            (None, ("verify_canonical_config_binding",)),
            (None, ("_execute_policy_run",)),
        ),
    ),
}
INTERNAL_MUTATING_CLI_CONTROLS = {
    "run-representation": (
        "verify_cli_worker_authorization_from_environment",
        "_run_representation_training",
    )
}
CLI_IMPORTED_CONTROL_SYMBOLS = {
    PUBLIC_MUTATING_CLI_FIRST_GUARD: "tgvf_rl.ops.cli_authorization",
    "verify_cli_worker_authorization_from_environment": (
        "tgvf_rl.ops.cli_authorization"
    ),
    "_execute_representation_torchrun": "tgvf_rl.ops.cli_launch",
    "_execute_policy_run": "tgvf_rl.ops.cli_launch",
}
CLI_LOCAL_CONTROL_SYMBOLS = frozenset(
    {
        guard
        for guard, _dispatch in (
            *PUBLIC_MUTATING_CLI_CONTROLS.values(),
            *INTERNAL_MUTATING_CLI_CONTROLS.values(),
        )
        if guard != "verify_cli_worker_authorization_from_environment"
    }
    | {
        dispatch
        for _guard, dispatch in (
            *PUBLIC_MUTATING_CLI_CONTROLS.values(),
            *INTERNAL_MUTATING_CLI_CONTROLS.values(),
        )
    }
    | set(PUBLIC_MUTATING_CLI_PREFLIGHTS.values())
).difference(CLI_IMPORTED_CONTROL_SYMBOLS)
CHECK_LAUNCH_GATE_ALLOWED_MODES = (
    "audit-control-plane",
    "authorize",
    "consume",
    "override-freeze",
    "quarantine-legacy",
    "ready",
    "status",
    "wait",
)
CHECK_LAUNCH_GATE_CAPABILITIES = {
    "gpu": False,
    "network": False,
    "subprocess": False,
    "arbitrary_exec": False,
    "destructive_fs": False,
}
IMPORT_ONLY_SUPPORT_MODULES = frozenset(
    {"spikes/verl_compat/verl_sync_fixed_reward.py"}
)
IMPORT_ONLY_SUPPORT_ALLOWED_IMPORTS = {
    "spikes/verl_compat/verl_sync_fixed_reward.py": frozenset({"__future__", "typing"})
}
LEGACY_QUARANTINE_GUARD = "assert_legacy_standalone_execution_quarantined"
LEGACY_MODE_QUARANTINE_GUARD = "assert_legacy_standalone_mode_quarantined"
LEGACY_QUARANTINE_MODULE = "tgvf_rl.ops.cli_authorization"


def _key_value(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or "\x00" in key or "\x00" in item:
        raise argparse.ArgumentTypeError("expected a non-empty KEY=VALUE")
    return key, item


def _named_path(value: str) -> tuple[str, Path]:
    key, item = _key_value(value)
    if not item:
        raise argparse.ArgumentTypeError("evidence path must be non-empty")
    return key, Path(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ready = commands.add_parser("ready", help="publish an immutable ready receipt")
    ready.add_argument("--gate-directory", type=Path, required=True)
    ready.add_argument("--run-id", required=True)
    ready.add_argument("--phase", required=True)
    ready.add_argument("--command-id", required=True)
    ready.add_argument("--identity-field", action="append", type=_key_value, default=[])
    ready.add_argument("--evidence", action="append", type=_named_path, required=True)

    authorize = commands.add_parser(
        "authorize", help="explicitly issue one short-lived launch token"
    )
    authorize.add_argument("--gate-directory", type=Path, required=True)
    authorize.add_argument("--ttl-seconds", type=float, default=3600)
    authorize.add_argument("--authorized-by")
    authorize.add_argument("--yes-authorize", action="store_true", required=True)

    override = commands.add_parser(
        "override-freeze", help="explicitly record one audited freeze exception"
    )
    override.add_argument("--gate-directory", type=Path, required=True)
    override.add_argument(
        "--execution-policy", type=Path, default=DEFAULT_EXECUTION_POLICY
    )
    override.add_argument("--reason", required=True)
    override.add_argument("--ttl-seconds", type=float, default=900)
    override.add_argument("--authorized-by")
    override.add_argument("--yes-override-freeze", action="store_true", required=True)

    consume = commands.add_parser(
        "consume", help="atomically consume authorization immediately before launch"
    )
    consume.add_argument("--gate-directory", type=Path, required=True)
    consume.add_argument("--token", type=Path, required=True)
    consume.add_argument(
        "--execution-policy", type=Path, default=DEFAULT_EXECUTION_POLICY
    )
    consume.add_argument("--freeze-override", type=Path)
    consume.add_argument("--expected-run-id", required=True)
    consume.add_argument("--expected-phase", required=True)
    consume.add_argument("--consumed-by")

    wait = commands.add_parser(
        "wait", help="wait for one non-empty artifact with timeout and liveness"
    )
    wait.add_argument("--path", type=Path, required=True)
    wait.add_argument("--timeout-seconds", type=float, required=True)
    wait.add_argument("--poll-seconds", type=float, default=5.0)
    wait.add_argument("--liveness-receipt", type=Path)
    wait.add_argument("--expected-run-id")
    wait.add_argument("--expected-phase")

    status = commands.add_parser("status", help="show a redacted gate summary")
    status.add_argument("--gate-directory", type=Path, required=True)

    quarantine = commands.add_parser(
        "quarantine-legacy",
        help="permanently reject one historical standalone execution entry",
    )
    quarantine.add_argument("--tool-id", required=True)

    audit = commands.add_parser(
        "audit-control-plane", help="audit canonical controls and list legacy gaps"
    )
    audit.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def _command_name(test: ast.expr) -> str | None:
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
        or len(test.comparators) != 1
        or not isinstance(test.left, ast.Attribute)
        or test.left.attr != "command"
        or not isinstance(test.left.value, ast.Name)
        or test.left.value.id != "args"
        or not isinstance(test.comparators[0], ast.Constant)
        or not isinstance(test.comparators[0].value, str)
    ):
        return None
    return test.comparators[0].value


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _direct_statement_call(statement: ast.stmt) -> ast.Call | None:
    value: ast.expr | None = None
    if isinstance(statement, ast.Expr):
        value = statement.value
    elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
        value = statement.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value
    return None


def _direct_branch_call_positions(branch: ast.If, name: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, statement in enumerate(branch.body)
        if (call := _direct_statement_call(statement)) is not None
        and call.func.id == name
    )


def _controlled_branch_call_positions(
    command: str,
    branch: ast.If,
    name: str,
) -> tuple[int, ...]:
    """Locate direct calls in the audited outer/owned-lifetime sequence only."""

    statements = list(branch.body)
    if command in PUBLIC_MUTATING_CLI_PREPARED_LIFETIME_CONTRACTS:
        final_statement = statements.pop() if statements else None
        if isinstance(final_statement, ast.Try):
            statements.extend(final_statement.body)
    return tuple(
        index
        for index, statement in enumerate(statements)
        if (call := _direct_statement_call(statement)) is not None
        and call.func.id == name
    )


def _statement_matches_public_branch_contract(
    statement: ast.stmt,
    target: str | None,
    expected_calls: tuple[str, ...],
) -> bool:
    expected_extension_if = ast.parse(
        "if args.horizon_extension is not None:\n"
        "    extension = _load_policy_horizon_extension("
        "args.horizon_extension, config)\n"
    ).body[0]
    expected_extension_none = ast.parse("extension = None\n").body[0]
    if target == "@extension-none":
        return ast.dump(statement, include_attributes=False) == ast.dump(
            expected_extension_none, include_attributes=False
        )
    if target == "@horizon-extension-if":
        return ast.dump(statement, include_attributes=False) == ast.dump(
            expected_extension_if, include_attributes=False
        )
    forbidden_nested = (
        ast.Await,
        ast.Delete,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.ListComp,
        ast.NamedExpr,
        ast.SetComp,
        ast.Yield,
        ast.YieldFrom,
    )
    call = _direct_statement_call(statement)
    if call is None or not expected_calls or call.func.id != expected_calls[0]:
        return False
    if target is None:
        if not isinstance(statement, ast.Expr):
            return False
    elif not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == target
    ):
        return False
    calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)]
    if any(not isinstance(node.func, ast.Name) for node in calls):
        return False
    if tuple(node.func.id for node in calls) != expected_calls:
        return False
    if any(
        isinstance(node, forbidden_nested)
        for argument in (*call.args, *(item.value for item in call.keywords))
        for node in ast.walk(argument)
    ):
        return False
    return not (
        call.func.id == PUBLIC_MUTATING_CLI_FIRST_GUARD and (call.args or call.keywords)
    )


def _statements_match_public_branch_contract(
    statements: list[ast.stmt],
    contract: tuple[tuple[str | None, tuple[str, ...]], ...],
) -> bool:
    return len(statements) == len(contract) and all(
        _statement_matches_public_branch_contract(statement, target, expected_calls)
        for statement, (target, expected_calls) in zip(
            statements, contract, strict=True
        )
    )


def _prepared_lifetime_try_matches_contract(
    statement: ast.stmt,
    contract: tuple[tuple[str | None, tuple[str, ...]], ...],
) -> bool:
    expected_finally = ast.parse(
        "try:\n    pass\nfinally:\n    prepared.close_python_binding()\n"
    ).body[0]
    assert isinstance(expected_finally, ast.Try)
    return (
        isinstance(statement, ast.Try)
        and not statement.handlers
        and not statement.orelse
        and len(statement.finalbody) == 1
        and ast.dump(statement.finalbody[0], include_attributes=False)
        == ast.dump(expected_finally.finalbody[0], include_attributes=False)
        and _statements_match_public_branch_contract(statement.body, contract)
    )


def _public_branch_matches_contract(command: str, branch: ast.If) -> bool:
    lifetime_contract = PUBLIC_MUTATING_CLI_PREPARED_LIFETIME_CONTRACTS.get(command)
    if lifetime_contract is not None:
        prefix_contract, try_contract = lifetime_contract
        return (
            len(branch.body) == len(prefix_contract) + 1
            and _statements_match_public_branch_contract(
                branch.body[:-1], prefix_contract
            )
            and _prepared_lifetime_try_matches_contract(branch.body[-1], try_contract)
        )
    contract = PUBLIC_MUTATING_CLI_BRANCH_CONTRACTS[command]
    return _statements_match_public_branch_contract(branch.body, contract)


def _internal_worker_guard_prefix_matches(branch: ast.If) -> bool:
    expected = ast.parse(
        "launch_identity = verify_cli_worker_authorization_from_environment(\n"
        "    expected_phase=REPRESENTATION_TRAINING_PHASE,\n"
        "    expected_command_id=_REPRESENTATION_COMMAND_ID,\n"
        ")\n"
        "assert_canonical_runtime_launch_enabled()\n"
    ).body
    return len(branch.body) >= 2 and all(
        ast.dump(observed, include_attributes=False)
        == ast.dump(wanted, include_attributes=False)
        for observed, wanted in zip(branch.body[:2], expected, strict=True)
    )


def _main_command_branches(
    tree: ast.Module,
) -> tuple[dict[str, ast.If], list[str]]:
    """Return only the real ``main`` command chain, never lookalike dead code."""

    violations: list[str] = []
    main_functions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "main"
    ]
    if len(main_functions) != 1 or main_functions[0].decorator_list:
        return {}, ["exactly one undecorated module-level main function"]
    main_function = main_functions[0]
    expected_prefix = ast.parse(
        "args = _parser().parse_args(argv)\nresult: object | None = None\n"
    ).body
    expected_tail = ast.parse(
        "if result is not None:\n"
        "    print(json.dumps(result, indent=2, sort_keys=True, default=list))\n"
        "return 0\n"
    ).body
    expected_handler = ast.parse(
        "try:\n"
        "    pass\n"
        "except (OSError, TypeError, ValueError, RuntimeError) as error:\n"
        "    print(f'contract error: {error}', file=sys.stderr)\n"
        "    return 2\n"
    ).body[0]
    assert isinstance(expected_handler, ast.Try)
    envelope_matches = (
        len(main_function.body) == 5
        and all(
            ast.dump(observed, include_attributes=False)
            == ast.dump(expected, include_attributes=False)
            for observed, expected in zip(
                main_function.body[:2], expected_prefix, strict=True
            )
        )
        and isinstance(main_function.body[2], ast.Try)
        and all(
            ast.dump(observed, include_attributes=False)
            == ast.dump(expected, include_attributes=False)
            for observed, expected in zip(
                main_function.body[3:], expected_tail, strict=True
            )
        )
    )
    if not envelope_matches:
        violations.append("main pre-dispatch/result envelope differs")
    main_try = (
        main_function.body[2]
        if len(main_function.body) > 2 and isinstance(main_function.body[2], ast.Try)
        else None
    )
    if main_try is not None and (
        len(main_try.body) != 1
        or not isinstance(main_try.body[0], ast.If)
        or main_try.orelse
        or main_try.finalbody
        or len(main_try.handlers) != 1
        or ast.dump(main_try.handlers[0], include_attributes=False)
        != ast.dump(expected_handler.handlers[0], include_attributes=False)
    ):
        violations.append("main exception envelope differs")
    roots = [
        statement
        for owner in (() if main_try is None else (main_try,))
        for statement in owner.body
        if isinstance(statement, ast.If) and _command_name(statement.test) is not None
    ]
    if len(roots) != 1:
        return {}, [*violations, "exactly one direct main command dispatch chain"]

    ordered: list[ast.If] = []
    branch: ast.If | None = roots[0]
    while branch is not None:
        ordered.append(branch)
        branch = (
            branch.orelse[0]
            if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If)
            else None
        )
    names = [_command_name(item.test) for item in ordered]
    if any(name is None for name in names) or len(set(names)) != len(names):
        violations.append(
            "main command dispatch chain has invalid or duplicate choices"
        )
    all_command_branches = [
        item
        for item in ast.walk(main_function)
        if isinstance(item, ast.If) and _command_name(item.test) is not None
    ]
    if {id(item) for item in all_command_branches} != {id(item) for item in ordered}:
        violations.append(
            "args.command tests exist outside the direct main dispatch chain"
        )
    return {
        name: item
        for name, item in zip(names, ordered, strict=True)
        if name is not None
    }, violations


def _import_bound_name(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    if alias.asname is not None:
        return alias.asname
    if isinstance(node, ast.Import):
        return alias.name.partition(".")[0]
    return alias.name


def _protected_symbol_provenance(
    tree: ast.Module,
    *,
    local_definitions: frozenset[str],
    exact_imports: dict[str, str],
) -> dict[str, list[str]]:
    """Reject every alternate binding for security-sensitive call targets."""

    protected = local_definitions | exact_imports.keys()
    issues = {name: [] for name in protected}
    module_body_ids = {id(statement) for statement in tree.body}
    for name in sorted(local_definitions):
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if (
            len(definitions) != 1
            or id(definitions[0]) not in module_body_ids
            or definitions[0].decorator_list
        ):
            issues[name].append("exactly one undecorated module-level definition")
        if any(
            _import_bound_name(node, alias) == name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        ):
            issues[name].append("no import may replace the local definition")
    for name, module in exact_imports.items():
        exact = []
        alternate = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for alias in node.names:
                if _import_bound_name(node, alias) != name:
                    continue
                if (
                    isinstance(node, ast.ImportFrom)
                    and id(node) in module_body_ids
                    and node.level == 0
                    and node.module == module
                    and alias.name == name
                    and alias.asname is None
                ):
                    exact.append(node)
                else:
                    alternate.append(node)
        if len(exact) != 1:
            issues[name].append(f"exactly one direct import from {module}")
        if alternate:
            issues[name].append("no alternate import binding")

    for node in ast.walk(tree):
        rebound: str | None = None
        if (
            isinstance(node, ast.Name)
            and node.id in protected
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            rebound = node.id
        elif isinstance(node, ast.arg) and node.arg in protected:
            rebound = node.arg
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in protected
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            rebound = node.attr
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in protected
        ):
            rebound = node.slice.value
        elif isinstance(node, ast.ClassDef) and node.name in protected:
            rebound = node.name
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in exact_imports
        ):
            rebound = node.name
        elif isinstance(node, ast.ExceptHandler) and node.name in protected:
            rebound = node.name
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in protected.intersection(node.names):
                issues[name].append("no global/nonlocal rebinding declaration")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in protected:
            rebound = node.name
        if rebound is not None:
            issues[rebound].append("no Store/Delete/shadow/rebind occurrence")

    if any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    ):
        for name in protected:
            issues[name].append("no wildcard import may alter control symbols")

    dynamic_namespace_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {"delattr", "eval", "exec", "globals", "locals", "setattr", "vars"}
    ]
    if dynamic_namespace_calls:
        for name in protected:
            issues[name].append("no dynamic namespace access")
    return {name: sorted(set(values)) for name, values in issues.items() if values}


def _named_call_count(node: ast.AST, name: str) -> int:
    return sum(
        1
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    )


def _branch_named_call_count(branch: ast.If, name: str) -> int:
    return sum(_named_call_count(statement, name) for statement in branch.body)


def _preflight_starts_with_runtime_closure(
    tree: ast.Module,
    name: str,
) -> bool:
    matches = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == name
    ]
    if len(matches) != 1 or matches[0].decorator_list:
        return False
    function = matches[0]
    first = _first_executable_statement(function)
    call = _direct_statement_call(first) if first is not None else None
    return (
        call is not None
        and call.func.id == "assert_canonical_runtime_launch_enabled"
        and not call.args
        and not call.keywords
        and _named_call_count(function, "assert_canonical_runtime_launch_enabled") == 1
    )


def _frozenset_assignment(
    tree: ast.Module,
    *,
    name: str,
) -> set[str] | None:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
        ):
            try:
                value = ast.literal_eval(node.value.args[0])
            except (TypeError, ValueError):
                return None
            if isinstance(value, set) and all(isinstance(item, str) for item in value):
                return value
    return None


def _audit_public_cli_controls(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    cli_path = repository_root / "src/tgvf_rl/cli.py"
    rows: list[dict[str, object]] = []
    violations: list[str] = []
    if not cli_path.is_file():
        return rows, ["public CLI source is missing: src/tgvf_rl/cli.py"]
    source = cli_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(cli_path))
    except SyntaxError as error:
        return rows, [f"public CLI source is not valid Python: {error}"]
    branches, branch_violations = _main_command_branches(tree)
    violations.extend(f"public CLI structure: {item}" for item in branch_violations)
    provenance = _protected_symbol_provenance(
        tree,
        local_definitions=CLI_LOCAL_CONTROL_SYMBOLS,
        exact_imports=CLI_IMPORTED_CONTROL_SYMBOLS,
    )
    expected_public = set(PUBLIC_MUTATING_CLI_CONTROLS)
    expected_internal = set(INTERNAL_MUTATING_CLI_CONTROLS)
    declared_public = _frozenset_assignment(
        tree,
        name="PUBLIC_MUTATING_COMMANDS",
    )
    declared_internal = _frozenset_assignment(
        tree,
        name="INTERNAL_MUTATING_COMMANDS",
    )
    if declared_public != expected_public:
        violations.append(
            "public CLI mutating command inventory differs from launch-gate audit"
        )
    if declared_internal != expected_internal:
        violations.append(
            "internal CLI mutating command inventory differs from launch-gate audit"
        )
    required_parser_markers = {
        "launch-representation": (
            "_add_execution_authorization_arguments(launch_representation)",
        ),
        "run-policy": ("_add_execution_authorization_arguments(run_policy)",),
        "run-representation-internal-evaluation": (
            "_add_execution_authorization_arguments(run_representation_evaluation)",
        ),
        "run-representation": (
            '"--launch-consumption-receipt"',
            '"--launch-consumption-sha256"',
            '"--launcher-liveness-receipt"',
        ),
    }
    controls = {
        **PUBLIC_MUTATING_CLI_CONTROLS,
        **INTERNAL_MUTATING_CLI_CONTROLS,
    }
    for command, (guard, dispatch) in controls.items():
        missing: list[str] = []
        preflight = PUBLIC_MUTATING_CLI_PREFLIGHTS.get(command)
        protected_for_command = {
            guard,
            dispatch,
            *(() if preflight is None else (preflight,)),
            PUBLIC_MUTATING_CLI_FIRST_GUARD,
        }
        for name in sorted(protected_for_command):
            missing.extend(
                f"{name} provenance: {item}" for item in provenance.get(name, [])
            )
        branch = branches.get(command)
        if branch is None:
            missing.append("exactly one command branch")
            guard_positions: tuple[int, ...] = ()
            dispatch_positions: tuple[int, ...] = ()
        else:
            guard_positions = _controlled_branch_call_positions(command, branch, guard)
            dispatch_positions = _controlled_branch_call_positions(
                command, branch, dispatch
            )
            if (
                len(guard_positions) != 1
                or _branch_named_call_count(branch, guard) != 1
            ):
                missing.append(f"exactly one uncaught direct {guard}")
            if (
                len(dispatch_positions) != 1
                or _branch_named_call_count(branch, dispatch) != 1
            ):
                missing.append(f"exactly one uncaught direct {dispatch}")
            if (
                len(guard_positions) == 1
                and len(dispatch_positions) == 1
                and guard_positions[0] >= dispatch_positions[0]
            ):
                missing.append(f"direct {guard} before {dispatch}")
        if (
            command
            in {
                *PUBLIC_MUTATING_CLI_BRANCH_CONTRACTS,
                *PUBLIC_MUTATING_CLI_PREPARED_LIFETIME_CONTRACTS,
            }
            and branch is not None
            and not _public_branch_matches_contract(command, branch)
        ):
            missing.append(
                "exact fail-closed branch statement/call shape without extra effects"
            )
        if preflight is not None:
            first_guard_positions = (
                ()
                if branch is None
                else _direct_branch_call_positions(
                    branch, PUBLIC_MUTATING_CLI_FIRST_GUARD
                )
            )
            if (
                branch is None
                or first_guard_positions != (0,)
                or _branch_named_call_count(branch, PUBLIC_MUTATING_CLI_FIRST_GUARD)
                != 1
            ):
                missing.append(
                    "runtime-closure assertion as the first direct branch statement"
                )
            preflight_positions = (
                ()
                if branch is None
                else _direct_branch_call_positions(branch, preflight)
            )
            if (
                branch is None
                or len(preflight_positions) != 1
                or _branch_named_call_count(branch, preflight) != 1
            ):
                missing.append(f"exactly one uncaught direct {preflight}")
            elif (
                len(guard_positions) == 1
                and preflight_positions[0] >= guard_positions[0]
            ):
                missing.append(f"direct {preflight} before {guard}")
            elif (
                first_guard_positions == (0,)
                and preflight_positions[0] <= first_guard_positions[0]
            ):
                missing.append(f"direct runtime-closure assertion before {preflight}")
            if not _preflight_starts_with_runtime_closure(tree, preflight):
                missing.append(
                    f"{preflight} starts with direct runtime-closure assertion"
                )
        elif branch is not None:
            closure_positions = _direct_branch_call_positions(
                branch, PUBLIC_MUTATING_CLI_FIRST_GUARD
            )
            if (
                guard_positions != (0,)
                or closure_positions != (1,)
                or _branch_named_call_count(branch, PUBLIC_MUTATING_CLI_FIRST_GUARD)
                != 1
            ):
                missing.append(
                    f"direct {guard} then runtime-closure assertion before worker work"
                )
            if not _internal_worker_guard_prefix_matches(branch):
                missing.append("exact inherited-worker guard/closure call shape")
        missing.extend(
            marker
            for marker in required_parser_markers[command]
            if marker not in source
        )
        if missing:
            violations.append(
                f"public CLI control {command} lacks: {', '.join(missing)}"
            )
        rows.append(
            {
                "command": command,
                "surface": (
                    "public"
                    if command in PUBLIC_MUTATING_CLI_CONTROLS
                    else "internal_worker"
                ),
                "guard": guard,
                "preflight": preflight,
                "dispatch": dispatch,
                "status": "migrated" if not missing else "unmigrated",
                "missing_markers": missing,
            }
        )
    return rows, violations


def _first_executable_statement(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.stmt | None:
    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements.pop(0)
    return None if not statements else statements[0]


def _is_module_main_test(test: ast.expr) -> bool:
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
        or len(test.comparators) != 1
    ):
        return False
    left, right = test.left, test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _early_quarantine_parent_depth(relative: str) -> int | None:
    if relative.startswith("tools/"):
        return 1
    if relative.startswith("spikes/"):
        return 2
    if relative == "src/tgvf_rl/framework/verl/prl13_main.py":
        return 4
    return None


def _expected_early_quarantine(relative: str) -> ast.If | None:
    parent_depth = _early_quarantine_parent_depth(relative)
    if parent_depth is None:
        return None
    source = f"""if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range({parent_depth + 1}):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            {relative!r},
        ),
    )
"""
    statement = ast.parse(source).body[0]
    assert isinstance(statement, ast.If)
    return statement


def _has_exact_early_quarantine(tree: ast.Module, relative: str) -> bool:
    expected = _expected_early_quarantine(relative)
    if expected is None:
        return False
    index = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        index = 1
    while (
        index < len(tree.body)
        and isinstance(tree.body[index], ast.ImportFrom)
        and tree.body[index].module == "__future__"
    ):
        index += 1
    if index >= len(tree.body):
        return False
    return ast.dump(tree.body[index], include_attributes=False) == ast.dump(
        expected,
        include_attributes=False,
    )


def _is_direct_main_dispatch(statement: ast.stmt) -> bool:
    call: ast.Call | None = None
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
    elif (
        isinstance(statement, ast.Raise)
        and statement.cause is None
        and isinstance(statement.exc, ast.Call)
        and isinstance(statement.exc.func, ast.Name)
        and statement.exc.func.id == "SystemExit"
        and len(statement.exc.args) == 1
        and not statement.exc.keywords
        and isinstance(statement.exc.args[0], ast.Call)
    ):
        call = statement.exc.args[0]
    return bool(
        call is not None
        and isinstance(call.func, ast.Name)
        and call.func.id == "main"
        and not call.args
        and not call.keywords
    )


def _has_direct_module_main_dispatch(
    tree: ast.Module,
    *,
    expected_blocks: int,
) -> bool:
    blocks = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.If) and _is_module_main_test(statement.test)
    ]
    if len(blocks) != expected_blocks:
        return False
    dispatch = blocks[-1]
    return (
        tree.body[-1] is dispatch
        and not dispatch.orelse
        and len(dispatch.body) == 1
        and _is_direct_main_dispatch(dispatch.body[0])
    )


def _audit_legacy_quarantine(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    violations: list[str] = []
    for relative in LEGACY_QUARANTINED_CONTROLLERS:
        path = repository_root / relative
        missing: list[str] = []
        if not path.is_file():
            missing.append("<file missing>")
        else:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                missing.append(f"valid Python source ({error})")
            else:
                first_line = source.partition("\n")[0]
                isolated_shebang = "#!/usr/bin/python3 -I"
                if first_line != isolated_shebang:
                    missing.append("fixed isolated Python shebang")
                if not _has_exact_early_quarantine(tree, relative):
                    missing.append(
                        "exact top-level exec quarantine before all runtime imports"
                    )
                imported = any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == LEGACY_QUARANTINE_MODULE
                    and any(
                        alias.name == LEGACY_QUARANTINE_GUARD
                        and alias.asname in {None, LEGACY_QUARANTINE_GUARD}
                        for alias in node.names
                    )
                    for node in tree.body
                )
                if not imported:
                    missing.append(f"direct import of {LEGACY_QUARANTINE_GUARD}")
                main_functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main"
                ]
                if len(main_functions) != 1:
                    missing.append("exactly one module-level main function")
                else:
                    main_function = main_functions[0]
                    if main_function.decorator_list:
                        missing.append("undecorated main function")
                    first = _first_executable_statement(main_function)
                    guarded = (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Call)
                        and isinstance(first.value.func, ast.Name)
                        and first.value.func.id == LEGACY_QUARANTINE_GUARD
                        and len(first.value.args) == 1
                        and not first.value.keywords
                        and isinstance(first.value.args[0], ast.Constant)
                        and first.value.args[0].value == relative
                    )
                    if not guarded:
                        missing.append(
                            f"{LEGACY_QUARANTINE_GUARD}({relative!r}) as the "
                            "first executable main statement"
                        )
                    guard_calls = [
                        node
                        for node in ast.walk(main_function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == LEGACY_QUARANTINE_GUARD
                    ]
                    if len(guard_calls) != 1:
                        missing.append(
                            f"exactly one direct {LEGACY_QUARANTINE_GUARD} call"
                        )
                if not _has_direct_module_main_dispatch(tree, expected_blocks=2):
                    missing.append("uncaught direct top-level main dispatch")
        if missing:
            violations.append(
                f"legacy standalone controller {relative} is not fail-closed: "
                f"{', '.join(missing)}"
            )
        rows.append(
            {
                "path": relative,
                "status": (
                    "legacy_quarantined_fail_closed"
                    if not missing
                    else "legacy_quarantine_bypassable"
                ),
                "exists": path.is_file(),
                "guard": LEGACY_QUARANTINE_GUARD,
                "missing_markers": missing,
            }
        )
    return rows, violations


def _literal_string_tuple(call: ast.Call, *, keyword: str) -> tuple[str, ...] | None:
    value = next(
        (item.value for item in call.keywords if item.arg == keyword),
        None,
    )
    if value is None:
        return None
    try:
        literal = ast.literal_eval(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(literal, tuple) or not all(
        isinstance(item, str) for item in literal
    ):
        return None
    return literal


def _mode_selector_matches(value: ast.expr | None, selector: str) -> bool:
    if value is None:
        return False
    if selector not in {"launch_gpu", "validate_only"}:
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "args"
            and value.attr == selector
        )
    flag_name = "launch_gpu" if selector == "launch_gpu" else "validate_only"
    true_mode = "launch-gpu" if selector == "launch_gpu" else "validate"
    false_mode = "plan" if selector == "launch_gpu" else "execute"
    return (
        isinstance(value, ast.IfExp)
        and isinstance(value.test, ast.Attribute)
        and isinstance(value.test.value, ast.Name)
        and value.test.value.id == "args"
        and value.test.attr == flag_name
        and isinstance(value.body, ast.Constant)
        and value.body.value == true_mode
        and isinstance(value.orelse, ast.Constant)
        and value.orelse.value == false_mode
    )


_PARSER_DERIVATION_METHODS = frozenset(
    {
        "add_argument_group",
        "add_mutually_exclusive_group",
        "add_parser",
        "add_subparsers",
    }
)
_PARSER_REGISTRATION_METHODS = frozenset({"add_argument", "add_parser"})


def _single_name_assignment(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
    ):
        return None
    return statement.targets[0].id, statement.value


def _is_argparse_constructor(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "argparse"
        and call.func.attr == "ArgumentParser"
    )


def _trusted_parser_method(
    call: ast.Call,
    *,
    trusted_receivers: set[str],
    methods: frozenset[str],
) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in trusted_receivers
        and call.func.attr in methods
    )


def _has_only_bounded_nested_parser_calls(call: ast.Call) -> bool:
    for nested in ast.walk(call):
        if nested is call or not isinstance(nested, ast.Call):
            continue
        if not (
            isinstance(nested.func, ast.Name)
            and nested.func.id == "Path"
            and len(nested.args) == 1
            and not nested.keywords
        ):
            return False
    return True


def _bounded_parser_builder_statements(
    statements: list[ast.stmt],
    *,
    trusted_receivers: set[str],
) -> bool:
    for statement in statements:
        assignment = _single_name_assignment(statement)
        if assignment is not None:
            target, value = assignment
            if not isinstance(value, ast.Call):
                return False
            if _is_argparse_constructor(value):
                if target != "parser" or not _has_only_bounded_nested_parser_calls(
                    value
                ):
                    return False
                trusted_receivers.add(target)
                continue
            if not (
                _trusted_parser_method(
                    value,
                    trusted_receivers=trusted_receivers,
                    methods=_PARSER_DERIVATION_METHODS,
                )
                and _has_only_bounded_nested_parser_calls(value)
            ):
                return False
            trusted_receivers.add(target)
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            if not (
                _trusted_parser_method(
                    statement.value,
                    trusted_receivers=trusted_receivers,
                    methods=_PARSER_REGISTRATION_METHODS,
                )
                and _has_only_bounded_nested_parser_calls(statement.value)
            ):
                return False
            continue
        if isinstance(statement, ast.For):
            if (
                not isinstance(statement.target, ast.Name)
                or statement.orelse
                or not isinstance(statement.iter, (ast.Tuple, ast.List))
                or any(
                    not isinstance(item, ast.Constant)
                    or not isinstance(item.value, str)
                    for item in statement.iter.elts
                )
            ):
                return False
            if not _bounded_parser_builder_statements(
                statement.body,
                trusted_receivers=set(trusted_receivers),
            ):
                return False
            continue
        return False
    return True


def _bounded_parser_factory(tree: ast.Module) -> bool:
    factories = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "_parser"
    ]
    if len(factories) != 1:
        return False
    factory = factories[0]
    if factory.decorator_list or len(factory.body) < 2:
        return False
    returned = factory.body[-1]
    if not (
        isinstance(returned, ast.Return)
        and isinstance(returned.value, ast.Name)
        and returned.value.id == "parser"
    ):
        return False
    trusted: set[str] = set()
    return (
        _bounded_parser_builder_statements(
            factory.body[:-1],
            trusted_receivers=trusted,
        )
        and "parser" in trusted
    )


def _bounded_parse_assignment(
    statement: ast.stmt,
    *,
    trusted_receivers: set[str],
    tree: ast.Module,
    main_function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    assignment = _single_name_assignment(statement)
    if assignment is None or assignment[0] != "args":
        return False
    value = assignment[1]
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Attribute)
        or value.func.attr != "parse_args"
        or value.keywords
    ):
        return False
    allowed_argv = {argument.arg for argument in main_function.args.args}
    if len(value.args) > 1 or any(
        not isinstance(argument, ast.Name) or argument.id not in allowed_argv
        for argument in value.args
    ):
        return False
    receiver = value.func.value
    if isinstance(receiver, ast.Name):
        return receiver.id in trusted_receivers
    return bool(
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Name)
        and receiver.func.id == "_parser"
        and not receiver.args
        and not receiver.keywords
        and _bounded_parser_factory(tree)
    )


def _bounded_parser_prefix(
    tree: ast.Module,
    main_function: ast.FunctionDef | ast.AsyncFunctionDef,
    guard_index: int,
) -> bool:
    prefix = list(main_function.body[:guard_index])
    if (
        prefix
        and isinstance(prefix[0], ast.Expr)
        and isinstance(prefix[0].value, ast.Constant)
        and isinstance(prefix[0].value.value, str)
    ):
        prefix.pop(0)
    if not prefix:
        return False
    trusted: set[str] = set()
    for index, statement in enumerate(prefix):
        if _bounded_parse_assignment(
            statement,
            trusted_receivers=trusted,
            tree=tree,
            main_function=main_function,
        ):
            return index == len(prefix) - 1
        if not _bounded_parser_builder_statements(
            [statement],
            trusted_receivers=trusted,
        ):
            return False
    return False


def _audit_mixed_mode_quarantine(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    violations: list[str] = []
    for relative, (
        read_only_modes,
        blocked_modes,
        selector,
    ) in MIXED_MODE_STANDALONE_CONTROLLERS.items():
        path = repository_root / relative
        missing: list[str] = []
        if not path.is_file():
            missing.append("<file missing>")
        else:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                missing.append(f"valid Python source ({error})")
            else:
                local_parser = any(
                    isinstance(statement, ast.FunctionDef)
                    and statement.name == "_parser"
                    for statement in tree.body
                )
                provenance = _protected_symbol_provenance(
                    tree,
                    local_definitions=frozenset(
                        {"main"} | ({"_parser"} if local_parser else set())
                    ),
                    exact_imports={
                        LEGACY_MODE_QUARANTINE_GUARD: LEGACY_QUARANTINE_MODULE
                    },
                )
                missing.extend(
                    f"{name} provenance: {item}"
                    for name, items in sorted(provenance.items())
                    for item in items
                )
                main_functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main"
                ]
                if len(main_functions) != 1:
                    missing.append("exactly one module-level main function")
                else:
                    main_function = main_functions[0]
                    if main_function.decorator_list:
                        missing.append("undecorated main function")
                    guard_calls = [
                        node
                        for node in ast.walk(main_function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == LEGACY_MODE_QUARANTINE_GUARD
                    ]
                    direct_guards = [
                        (index, statement.value)
                        for index, statement in enumerate(main_function.body)
                        if isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and statement.value.func.id == LEGACY_MODE_QUARANTINE_GUARD
                    ]
                    if len(guard_calls) != 1 or len(direct_guards) != 1:
                        missing.append(
                            "exactly one uncaught direct "
                            f"{LEGACY_MODE_QUARANTINE_GUARD} call"
                        )
                    else:
                        guard_index, guard = direct_guards[0]
                        if not _bounded_parser_prefix(
                            tree,
                            main_function,
                            guard_index,
                        ):
                            missing.append(
                                "guard immediately after trusted argparse parse"
                            )
                        tool_id = (
                            guard.args[0].value
                            if len(guard.args) == 1
                            and isinstance(guard.args[0], ast.Constant)
                            else None
                        )
                        selected = next(
                            (
                                item.value
                                for item in guard.keywords
                                if item.arg == "selected_mode"
                            ),
                            None,
                        )
                        if tool_id != relative:
                            missing.append("exact guarded tool ID")
                        if not _mode_selector_matches(selected, selector):
                            missing.append(f"exact {selector} mode selector")
                        if (
                            _literal_string_tuple(guard, keyword="read_only_modes")
                            != read_only_modes
                        ):
                            missing.append("exact read-only mode inventory")
                        if (
                            _literal_string_tuple(guard, keyword="blocked_modes")
                            != blocked_modes
                        ):
                            missing.append("exact blocked mode inventory")
                if not _has_direct_module_main_dispatch(tree, expected_blocks=1):
                    missing.append("uncaught direct top-level main dispatch")
        if missing:
            violations.append(
                f"mixed-mode standalone controller {relative} is not "
                f"fail-closed: {', '.join(missing)}"
            )
        rows.append(
            {
                "path": relative,
                "status": (
                    "read_only_modes_only"
                    if not missing
                    else "mutating_mode_quarantine_bypassable"
                ),
                "read_only_modes": list(read_only_modes),
                "blocked_modes": list(blocked_modes),
                "guard": LEGACY_MODE_QUARANTINE_GUARD,
                "missing_markers": missing,
            }
        )
    return rows, violations


def _audit_shell_quarantine(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    violations: list[str] = []
    for relative in LEGACY_QUARANTINED_SHELL_CONTROLLERS:
        path = repository_root / relative
        missing: list[str] = []
        if path.is_symlink() or not path.is_file():
            missing.append("<file missing>")
        else:
            expected = (
                "#!/bin/bash -p\n"
                "set -euo pipefail\n"
                "\n"
                "script_source=${BASH_SOURCE[0]}\n"
                "readonly script_source\n"
                'if [[ -L "$script_source" ]]; then\n'
                "  builtin printf '%s\\n' "
                '"refusing symlinked legacy shell controller: '
                '$script_source" >&2\n'
                "  exit 3\n"
                "fi\n"
                'if [[ "$script_source" == /* ]]; then\n'
                "  script_parent=${script_source%/*}\n"
                'elif [[ "$script_source" == */* ]]; then\n'
                "  script_parent=./${script_source%/*}\n"
                "else\n"
                "  script_parent=.\n"
                "fi\n"
                "readonly script_parent\n"
                'builtin cd -P -- "$script_parent"\n'
                "script_directory=$PWD\n"
                "readonly script_directory\n"
                "exec /usr/bin/python3 -I -- "
                '"$script_directory/check_launch_gate.py" quarantine-legacy \\\n'
                f'  --tool-id "{relative}"\n'
            ).encode("utf-8")
            if path.read_bytes() != expected:
                missing.append("exact fixed quarantine-wrapper bytes")
            if path.stat().st_mode & 0o111 == 0:
                missing.append("executable mode")
        if missing:
            violations.append(
                f"legacy shell controller {relative} is not fail-closed: "
                f"{', '.join(missing)}"
            )
        rows.append(
            {
                "path": relative,
                "status": (
                    "legacy_quarantined_fail_closed"
                    if not missing
                    else "legacy_quarantine_bypassable"
                ),
                "guard": "check_launch_gate.py quarantine-legacy",
                "missing_markers": missing,
            }
        )
    return rows, violations


def _discover_execution_surfaces(
    repository_root: Path,
) -> tuple[dict[str, str], list[str]]:
    discovered: dict[str, str] = {}
    violations: list[str] = []
    pyproject_path = repository_root / "pyproject.toml"
    console_modules: set[str] = set()
    if pyproject_path.is_symlink() or not pyproject_path.is_file():
        violations.append("pyproject.toml is missing or is a symlink")
    else:
        try:
            with pyproject_path.open("rb") as stream:
                pyproject = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            violations.append(f"pyproject.toml is not valid TOML: {error}")
        else:
            project = pyproject.get("project")
            scripts = project.get("scripts", {}) if isinstance(project, dict) else {}
            if not isinstance(scripts, dict):
                violations.append("pyproject project.scripts must be a table")
            else:
                for script_name, target in sorted(scripts.items()):
                    if not isinstance(script_name, str) or not isinstance(target, str):
                        violations.append("pyproject console scripts must be strings")
                        continue
                    module, separator, attribute = target.partition(":")
                    if (
                        not separator
                        or not attribute
                        or not all(part.isidentifier() for part in module.split("."))
                    ):
                        violations.append(
                            f"console script {script_name!r} has an invalid target"
                        )
                        continue
                    module_file = (
                        repository_root / "src" / Path(*module.split("."))
                    ).with_suffix(".py")
                    package_file = (
                        repository_root
                        / "src"
                        / Path(*module.split("."))
                        / "__init__.py"
                    )
                    candidates = [
                        path
                        for path in (module_file, package_file)
                        if path.is_file() and not path.is_symlink()
                    ]
                    if len(candidates) != 1:
                        violations.append(
                            f"console script {script_name!r} target module is not "
                            "one exact regular source file"
                        )
                        continue
                    console_modules.add(
                        candidates[0].relative_to(repository_root).as_posix()
                    )
    for root_name in EXECUTION_SURFACE_ROOTS:
        root = repository_root / root_name
        if not root.is_dir():
            violations.append(f"execution-surface root is missing: {root_name}")
            continue
        for path in sorted(root.rglob("*.py")):
            if not path.is_file():
                continue
            relative = path.relative_to(repository_root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                violations.append(
                    f"cannot parse Python execution-surface candidate {relative}: "
                    f"{error}"
                )
                continue
            has_main_guard = any(
                isinstance(statement, ast.If) and _is_module_main_test(statement.test)
                for statement in tree.body
            )
            first_line = path.read_bytes().partition(b"\n")[0]
            has_python_shebang = bool(
                re.match(rb"^#!.*\bpython(?:3(?:\.\d+)?)?\b", first_line)
            )
            is_executable = bool(
                path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            )
            if root_name in {"tools", "spikes"}:
                discovered[relative] = (
                    "python_main"
                    if has_main_guard or has_python_shebang or is_executable
                    else "python_module"
                )
            elif (
                has_main_guard
                or path.name == "__main__.py"
                or has_python_shebang
                or is_executable
                or relative in console_modules
            ):
                discovered[relative] = "python_main"
    for root_name in EXECUTION_SHELL_ROOTS:
        root = repository_root / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.sh")):
            if path.is_file():
                discovered[path.relative_to(repository_root).as_posix()] = "shell"
    return discovered, violations


def _load_execution_surface_policy(
    repository_root: Path,
) -> tuple[dict[str, object] | None, list[str]]:
    path = repository_root / "configs/ops/execution_surface_policy.json"
    if path.is_symlink() or not path.is_file():
        return None, ["execution-surface policy is missing or is a symlink"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, [f"execution-surface policy is not valid UTF-8 JSON: {error}"]
    if not isinstance(payload, dict):
        return None, ["execution-surface policy must be one JSON object"]
    required = {
        "schema_version",
        "revision",
        "content_binding",
        "python_roots",
        "shell_roots",
        "surfaces",
    }
    violations: list[str] = []
    if set(payload) != required:
        violations.append("execution-surface policy fields differ from schema")
    if payload.get("schema_version") != EXECUTION_SURFACE_POLICY_SCHEMA:
        violations.append("execution-surface policy schema_version differs")
    if payload.get("revision") != EXECUTION_SURFACE_POLICY_REVISION:
        violations.append("execution-surface policy revision differs")
    if payload.get("content_binding") != EXECUTION_SURFACE_CONTENT_BINDING:
        violations.append("execution-surface content binding differs")
    if payload.get("python_roots") != list(EXECUTION_SURFACE_ROOTS):
        violations.append("execution-surface Python roots differ")
    if payload.get("shell_roots") != list(EXECUTION_SHELL_ROOTS):
        violations.append("execution-surface shell roots differ")
    if not isinstance(payload.get("surfaces"), list):
        violations.append("execution-surface policy surfaces must be a list")
    return payload, violations


def _literal_assignment(statement: ast.stmt) -> bool:
    value: ast.expr | None = None
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        value = statement.value
        targets = list(statement.targets)
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
        targets = [statement.target]
    if (
        value is None
        or not targets
        or any(not isinstance(target, ast.Name) for target in targets)
    ):
        return False
    try:
        ast.literal_eval(value)
    except (TypeError, ValueError):
        return False
    return True


def _literal_function_defaults(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    defaults: list[ast.expr | None] = [
        *function.args.defaults,
        *function.args.kw_defaults,
    ]
    for value in defaults:
        if value is None:
            continue
        try:
            ast.literal_eval(value)
        except (TypeError, ValueError):
            return False
    return not function.decorator_list


def _import_only_support_module(
    tree: ast.Module,
    *,
    allowed_imports: frozenset[str],
) -> bool:
    """Recognize modules whose import-time syntax cannot dispatch repository work."""

    future_annotations = False
    for index, statement in enumerate(tree.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            if statement.level != 0 or any(
                alias.asname is not None or alias.name != "annotations"
                for alias in statement.names
            ):
                return False
            future_annotations = True
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            imported = (
                {alias.name.partition(".")[0] for alias in statement.names}
                if isinstance(statement, ast.Import)
                else {statement.module}
            )
            level = statement.level if isinstance(statement, ast.ImportFrom) else 0
            if level != 0 or not imported.issubset(allowed_imports):
                return False
            continue
        if _literal_assignment(statement):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _literal_function_defaults(statement):
                return False
            continue
        # Class bodies and decorators execute at import time.  They can be
        # admitted in a later schema only with an equally strict body audit.
        return False
    return future_annotations


def _check_launch_gate_parser_modes(
    repository_root: Path,
) -> tuple[tuple[str, ...] | None, list[str]]:
    relative = "tools/check_launch_gate.py"
    path = repository_root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        return None, [f"cannot inspect check-launch-gate parser: {error}"]
    factories = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "_parser"
    ]
    if len(factories) != 1 or factories[0].decorator_list:
        return None, ["check-launch-gate requires one undecorated _parser"]
    if not _bounded_parser_factory(tree):
        return None, ["check-launch-gate parser contains unbounded executable syntax"]
    factory = factories[0]
    parser_stores = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Name)
        and node.id == "parser"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    if len(parser_stores) != 1:
        return None, ["check-launch-gate parser receiver is rebound"]
    commands_assignments = [
        statement
        for statement in factory.body
        if (
            (assignment := _single_name_assignment(statement)) is not None
            and assignment[0] == "commands"
        )
    ]
    if len(commands_assignments) != 1:
        return None, ["check-launch-gate parser commands receiver differs"]
    assignment = _single_name_assignment(commands_assignments[0])
    assert assignment is not None
    subparsers = assignment[1]
    if not (
        isinstance(subparsers, ast.Call)
        and isinstance(subparsers.func, ast.Attribute)
        and isinstance(subparsers.func.value, ast.Name)
        and subparsers.func.value.id == "parser"
        and subparsers.func.attr == "add_subparsers"
        and not subparsers.args
        and {
            keyword.arg: keyword.value
            for keyword in subparsers.keywords
            if keyword.arg is not None
        }.keys()
        == {"dest", "required"}
        and any(
            keyword.arg == "dest"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "command"
            for keyword in subparsers.keywords
        )
        and any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in subparsers.keywords
        )
    ):
        return None, ["check-launch-gate parser subcommand binding differs"]
    stores = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Name)
        and node.id == "commands"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    parser_calls = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "commands"
        and node.func.attr == "add_parser"
    ]
    receiver_loads = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Name)
        and node.id == "commands"
        and isinstance(node.ctx, ast.Load)
    ]
    if len(stores) != 1 or len(receiver_loads) != len(parser_calls):
        return None, [
            "check-launch-gate parser commands receiver is rebound or escapes"
        ]
    choices: list[str] = []
    for call in parser_calls:
        if (
            len(call.args) != 1
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
            or not call.args[0].value
        ):
            return None, ["check-launch-gate parser has a nonliteral command choice"]
        choices.append(call.args[0].value)
    if len(set(choices)) != len(choices):
        return None, ["check-launch-gate parser has duplicate command choices"]
    if not (
        factory.body
        and isinstance(factory.body[-1], ast.Return)
        and isinstance(factory.body[-1].value, ast.Name)
        and factory.body[-1].value.id == "parser"
    ):
        return None, ["check-launch-gate parser does not directly return parser"]
    return tuple(sorted(choices)), []


def _audit_execution_surface_inventory(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    discovered, violations = _discover_execution_surfaces(repository_root)
    check_gate_modes, parser_violations = _check_launch_gate_parser_modes(
        repository_root
    )
    violations.extend(parser_violations)
    if (
        check_gate_modes is not None
        and check_gate_modes != CHECK_LAUNCH_GATE_ALLOWED_MODES
    ):
        violations.append(
            "check-launch-gate parser modes differ from the audited control contract"
        )
    policy, policy_violations = _load_execution_surface_policy(repository_root)
    violations.extend(policy_violations)
    if policy is None or not isinstance(policy.get("surfaces"), list):
        return [], violations

    required_fields = {
        "path",
        "kind",
        "classification",
        "allowed_modes",
        "blocked_modes",
        "authorization_kind",
        "capabilities",
        "workload_kinds",
        "content_sha256",
    }
    manifest_paths: list[str] = []
    rows: list[dict[str, object]] = []
    permanent_python: set[str] = set()
    permanent_shell: set[str] = set()
    mixed_python: set[str] = set()
    import_only_python: set[str] = set()
    authorization_by_classification = {
        "artifact_materializer": "bounded_artifact_write",
        "canonical_internal_worker": "inherited_worker_receipt",
        "canonical_public_cli": "canonical_run_bound",
        "control_utility": "repository_control",
        "import_only_support": "none_offline",
        "mixed_read_only_and_quarantined": "read_only_mode_guard",
        "permanent_quarantine": "permanent_quarantine",
        "read_only_utility": "none_offline",
    }
    for index, raw in enumerate(policy["surfaces"]):
        label = f"execution-surface policy row {index}"
        row_violations: list[str] = []
        if not isinstance(raw, dict):
            violations.append(f"{label} must be an object")
            continue
        if set(raw) != required_fields:
            row_violations.append("fields differ from schema")
        relative = raw.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or ".." in Path(relative).parts
        ):
            row_violations.append("path is not a normalized repository-relative path")
            relative = f"<invalid-row-{index}>"
        manifest_paths.append(relative)
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in {
            "python_main",
            "python_module",
            "shell",
        }:
            row_violations.append("kind is invalid")
        classification = raw.get("classification")
        if (
            not isinstance(classification, str)
            or classification not in EXECUTION_SURFACE_CLASSIFICATIONS
        ):
            row_violations.append("classification is invalid")
        authorization_kind = raw.get("authorization_kind")
        if (
            not isinstance(authorization_kind, str)
            or authorization_kind not in EXECUTION_SURFACE_AUTHORIZATION_KINDS
        ):
            row_violations.append("authorization_kind is invalid")
        elif (
            not isinstance(classification, str)
            or authorization_by_classification.get(classification) != authorization_kind
        ):
            row_violations.append("authorization_kind differs from classification")
        allowed_modes = raw.get("allowed_modes")
        blocked_modes = raw.get("blocked_modes")
        for name, modes in (
            ("allowed_modes", allowed_modes),
            ("blocked_modes", blocked_modes),
        ):
            if (
                not isinstance(modes, list)
                or any(not isinstance(mode, str) or not mode for mode in modes)
                or len(set(modes)) != len(modes)
            ):
                row_violations.append(f"{name} is invalid")
        if isinstance(allowed_modes, list) and isinstance(blocked_modes, list):
            if set(allowed_modes) & set(blocked_modes):
                row_violations.append("allowed and blocked modes overlap")
            if classification == "permanent_quarantine" and (
                allowed_modes != [] or blocked_modes != ["*"]
            ):
                row_violations.append("permanent quarantine modes differ")
            if classification == "mixed_read_only_and_quarantined" and (
                not allowed_modes or not blocked_modes
            ):
                row_violations.append("mixed-mode inventories must be non-empty")
            if classification == "import_only_support" and (
                allowed_modes != ["import-only"]
                or blocked_modes != ["direct-execution"]
            ):
                row_violations.append("import-only support modes differ")
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, dict) or set(capabilities) != set(
            EXECUTION_SURFACE_CAPABILITIES
        ):
            row_violations.append("capabilities fields differ")
            capabilities = {}
        elif any(type(value) is not bool for value in capabilities.values()):
            row_violations.append("capabilities must be booleans")
        workload_kinds = raw.get("workload_kinds")
        if (
            not isinstance(workload_kinds, list)
            or not workload_kinds
            or any(not isinstance(item, str) or not item for item in workload_kinds)
            or len(set(workload_kinds)) != len(workload_kinds)
        ):
            row_violations.append("workload_kinds is invalid")
            workload_kinds = []
        high_risk = bool(
            capabilities.get("gpu")
            or capabilities.get("network")
            or capabilities.get("arbitrary_exec")
            or capabilities.get("destructive_fs")
            or HIGH_RISK_WORKLOAD_KINDS.intersection(workload_kinds)
        )
        if high_risk and (
            not isinstance(classification, str)
            or classification
            not in {
                "canonical_internal_worker",
                "canonical_public_cli",
                "mixed_read_only_and_quarantined",
                "permanent_quarantine",
            }
        ):
            row_violations.append("high-risk noncanonical surface is not quarantined")
        if (
            classification == "canonical_public_cli"
            and relative != "src/tgvf_rl/cli.py"
        ):
            row_violations.append("canonical public CLI path differs")
        if classification == "canonical_internal_worker" and relative != (
            "src/tgvf_rl/framework/verl/policy_main.py"
        ):
            row_violations.append("canonical internal worker path differs")
        if relative == "tools/check_launch_gate.py":
            if classification != "control_utility":
                row_violations.append("check-launch-gate classification differs")
            if allowed_modes != list(CHECK_LAUNCH_GATE_ALLOWED_MODES):
                row_violations.append(
                    "allowed_modes differ from check-launch-gate parser"
                )
            if blocked_modes != []:
                row_violations.append("check-launch-gate blocked_modes differ")
            if capabilities != CHECK_LAUNCH_GATE_CAPABILITIES:
                row_violations.append(
                    "capabilities differ from check-launch-gate control contract"
                )

        path = repository_root / relative
        expected_digest = raw.get("content_sha256")
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest
        ):
            row_violations.append("content_sha256 is invalid")
        if relative not in discovered:
            row_violations.append("manifest path is not a discovered execution surface")
        elif discovered[relative] != kind:
            row_violations.append("manifest kind differs from discovered kind")
        if path.is_symlink() or not path.is_file():
            row_violations.append("surface is missing or is a symlink")
        elif isinstance(expected_digest, str) and _SHA256_RE.fullmatch(expected_digest):
            observed_digest = sha256(path.read_bytes()).hexdigest()
            if observed_digest != expected_digest:
                row_violations.append("content SHA256 differs")

        if classification == "permanent_quarantine":
            if kind == "python_main":
                permanent_python.add(relative)
            elif kind == "shell":
                permanent_shell.add(relative)
        elif classification == "mixed_read_only_and_quarantined":
            if kind != "python_main":
                row_violations.append("mixed-mode surface must be Python")
            mixed_python.add(relative)
        elif classification == "import_only_support":
            import_only_python.add(relative)
            if kind != "python_module":
                row_violations.append("import-only support must be a Python module")
            elif path.is_file() and not path.is_symlink():
                try:
                    support_tree = ast.parse(
                        path.read_text(encoding="utf-8"), filename=relative
                    )
                except (OSError, UnicodeDecodeError, SyntaxError):
                    row_violations.append("import-only support is not valid Python")
                else:
                    if not _import_only_support_module(
                        support_tree,
                        allowed_imports=IMPORT_ONLY_SUPPORT_ALLOWED_IMPORTS.get(
                            relative, frozenset()
                        ),
                    ):
                        row_violations.append(
                            "import-only support has import-time executable syntax"
                        )
        if kind == "python_module" and classification != "import_only_support":
            row_violations.append(
                "Python module without an entrypoint must be strict import-only support"
            )
        violations.extend(f"{label} {relative}: {item}" for item in row_violations)
        rows.append(
            {**raw, "status": "content_bound" if not row_violations else "invalid"}
        )

    if manifest_paths != sorted(manifest_paths):
        violations.append("execution-surface manifest rows are not path-sorted")
    if len(set(manifest_paths)) != len(manifest_paths):
        violations.append("execution-surface manifest contains duplicate paths")
    manifest_set = set(manifest_paths)
    for relative in sorted(set(discovered) - manifest_set):
        violations.append(f"unmanifested execution surface discovered: {relative}")
    for relative in sorted(manifest_set - set(discovered)):
        violations.append(f"manifested execution surface is missing: {relative}")
    if permanent_python != set(LEGACY_QUARANTINED_CONTROLLERS):
        violations.append("permanent Python quarantine inventory differs from manifest")
    if permanent_shell != set(LEGACY_QUARANTINED_SHELL_CONTROLLERS):
        violations.append("permanent shell quarantine inventory differs from manifest")
    if mixed_python != set(MIXED_MODE_STANDALONE_CONTROLLERS):
        violations.append("mixed-mode quarantine inventory differs from manifest")
    if import_only_python != set(IMPORT_ONLY_SUPPORT_MODULES):
        violations.append("import-only support inventory differs from manifest")
    return rows, violations


def audit_control_plane(repository_root: Path) -> dict[str, object]:
    repository_root = repository_root.expanduser().resolve(strict=True)
    violations: list[str] = []
    tools_root = repository_root / "tools"
    forbidden_tmux_mutation = "tmux " + "set-environment -g"
    for path in sorted(tools_root.iterdir()):
        if path.is_file() and path.suffix in {".py", ".sh"}:
            text = path.read_text(encoding="utf-8")
            if forbidden_tmux_mutation in text:
                violations.append(f"global tmux environment mutation: {path.name}")
    requirements = {
        "neutral_public_cli": (
            "PUBLIC_MUTATING_COMMANDS",
            "_consume_command_authorization",
            "verify_cli_worker_authorization",
        ),
    }
    canonical: list[dict[str, object]] = []
    for relative, control_kind in CANONICAL_ACTIVE_CONTROLLERS.items():
        path = repository_root / relative
        missing_markers: list[str] = []
        if not path.is_file():
            missing_markers.append("<file missing>")
        else:
            content = path.read_text(encoding="utf-8")
            missing_markers.extend(
                marker for marker in requirements[control_kind] if marker not in content
            )
        if missing_markers:
            violations.append(
                f"canonical controller {relative} lacks: {', '.join(missing_markers)}"
            )
        canonical.append(
            {
                "path": relative,
                "control_kind": control_kind,
                "status": "migrated" if not missing_markers else "unmigrated",
                "missing_markers": missing_markers,
            }
        )
    legacy, legacy_violations = _audit_legacy_quarantine(repository_root)
    violations.extend(legacy_violations)
    legacy_shell, legacy_shell_violations = _audit_shell_quarantine(repository_root)
    violations.extend(legacy_shell_violations)
    mixed_mode, mixed_mode_violations = _audit_mixed_mode_quarantine(repository_root)
    violations.extend(mixed_mode_violations)
    execution_surfaces, execution_surface_violations = (
        _audit_execution_surface_inventory(repository_root)
    )
    violations.extend(execution_surface_violations)
    public_cli, cli_violations = _audit_public_cli_controls(repository_root)
    violations.extend(cli_violations)
    policy_path = repository_root / "configs/ops/experiment_execution_policy.json"
    policy: object = None
    if not policy_path.is_file():
        violations.append("repository execution policy is missing")
    else:
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            violations.append("repository execution policy is not valid JSON")
        else:
            if not isinstance(policy, dict) or policy.get("execution_mode") != "frozen":
                violations.append("stabilization execution policy is not frozen")
    return {
        "status": "pass" if not violations else "blocked",
        "repository_root": str(repository_root),
        "execution_policy": policy,
        "canonical_active_controllers": canonical,
        "delegated_launchers": [],
        "legacy_quarantined_controllers": legacy,
        "legacy_quarantined_shell_controllers": legacy_shell,
        "mixed_mode_standalone_controllers": mixed_mode,
        "execution_surface_inventory": execution_surfaces,
        "public_cli_controls": public_cli,
        "violations": violations,
    }


def _print_blocked(error: LaunchGateError) -> NoReturn:
    print(
        json.dumps(
            {
                "status": "blocked",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit(3)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ready":
            parameters = dict(args.identity_field)
            if len(parameters) != len(args.identity_field):
                raise LaunchGateError("identity field keys must be unique")
            evidence = dict(args.evidence)
            if len(evidence) != len(args.evidence):
                raise LaunchGateError("evidence names must be unique")
            identity = make_run_identity(
                run_id=args.run_id,
                phase=args.phase,
                command_id=args.command_id,
                parameters=parameters,
            )
            result: object = materialize_ready_receipt(
                args.gate_directory,
                run_identity=identity,
                evidence_paths=evidence,
            )
        elif args.command == "authorize":
            path, value = issue_launch_authorization(
                args.gate_directory,
                ttl_seconds=args.ttl_seconds,
                authorized_by=args.authorized_by,
            )
            result = {
                "status": value["status"],
                "token_path": str(path),
                "token_id": value["token_id"],
                "run_id": value["run_id"],
                "phase": value["phase"],
                "expires_at": value["expires_at"],
            }
        elif args.command == "override-freeze":
            path, value = issue_freeze_override(
                args.gate_directory,
                args.execution_policy,
                reason=args.reason,
                ttl_seconds=args.ttl_seconds,
                authorized_by=args.authorized_by,
            )
            result = {
                "status": value["status"],
                "freeze_override_path": str(path),
                "override_id": value["override_id"],
                "run_id": value["run_id"],
                "phase": value["phase"],
                "reason": value["reason"],
                "expires_at": value["expires_at"],
            }
        elif args.command == "consume":
            result = consume_launch_authorization(
                args.gate_directory,
                args.token,
                args.execution_policy,
                expected_run_id=args.expected_run_id,
                expected_phase=args.expected_phase,
                freeze_override_path=args.freeze_override,
                consumed_by=args.consumed_by,
            )
        elif args.command == "wait":
            observed = wait_for_artifact(
                args.path,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
                liveness_receipt_path=args.liveness_receipt,
                expected_run_id=args.expected_run_id,
                expected_phase=args.expected_phase,
            )
            result = {"status": "ready", "path": str(observed)}
        elif args.command == "status":
            result = gate_status(args.gate_directory)
        elif args.command == "quarantine-legacy":
            assert_legacy_standalone_execution_quarantined(args.tool_id)
        else:
            result = audit_control_plane(args.repository_root)
            if result["status"] != "pass":
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
    except LaunchGateError as error:
        _print_blocked(error)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
