"""Small, fail-closed command line surface for validation and bounded runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping, Sequence

from tgvf_rl import SCHEMA_VERSION, __version__
from tgvf_rl.compatibility_stack import (
    AUDITED_COMPATIBILITY_STACKS,
    CONTROL_COMPATIBILITY_STACK,
    AuditedCompatibilityStack,
    audited_compatibility_stack,
)
from tgvf_rl.experiment_identity import validate_run_id
from tgvf_rl.framework.verl import (
    VerlAdapterConfig,
    VerlRuntimeRequirements,
    load_verl_public_api,
    verify_verl_distribution_identity,
)
from tgvf_rl.ops.cli_authorization import (
    CANONICAL_EVALUATION_CONFIG_ROOT,
    CANONICAL_POLICY_CONFIG_ROOT,
    CANONICAL_REPRESENTATION_CONFIG_ROOT,
    CLIExecutionAuthorizationIdentity,
    CanonicalConfigBinding,
    PythonExecutableIdentity,
    assert_canonical_runtime_launch_enabled,
    assert_loaded_config_matches_binding,
    bind_canonical_config_path,
    bind_current_python_executable,
    bind_current_python_executable_for_exec,
    consume_cli_execution_authorization,
    materialize_cli_worker_authorization,
    verify_canonical_config_binding,
    verify_cli_worker_authorization_from_environment,
    verify_python_executable_identity,
)
from tgvf_rl.ops.cli_launch import (
    PreparedRepresentationLaunch,
    _canonical_json_sha256 as _canonical_json_sha256,
    _execute_policy_run,
    _execute_representation_torchrun,
    _representation_child_environment,
    _representation_command_prefix,
    _representation_torchrun_command as _representation_torchrun_command,
)
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)


SMOKE_CONFIG_SCHEMA = "tgvf-fsdp2-smoke-v1"
POLICY_TRAINING_PHASE = "policy_training"
REPRESENTATION_TRAINING_PHASE = "representation_training"
REPRESENTATION_INTERNAL_EVALUATION_PHASE = "representation_internal_evaluation"
PUBLIC_MUTATING_COMMANDS = frozenset(
    {
        "launch-representation",
        "run-policy",
        "run-representation-internal-evaluation",
    }
)
INTERNAL_MUTATING_COMMANDS = frozenset({"run-representation"})
_REPRESENTATION_COMMAND_ID = "tgvf-rl:launch-representation:v2"
_POLICY_COMMAND_ID = "tgvf-rl:run-policy:v3"


def _require(mapping: Mapping[str, Any], key: str, expected: object) -> None:
    value = mapping.get(key)
    if value != expected:
        raise ValueError(f"{key} must be {expected!r}, got {value!r}")


def validate_smoke_config(
    path: Path, *, stack_selector: str = CONTROL_COMPATIBILITY_STACK
) -> Mapping[str, Any]:
    """Validate the bounded infrastructure smoke without adding hidden defaults."""

    selected_stack = audited_compatibility_stack(stack_selector)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _require(config, "schema_version", SMOKE_CONFIG_SCHEMA)
    _require(config, "scope", "synthetic_infrastructure_only")
    validate_run_id(config.get("run_id"))

    stack = config.get("stack")
    if not isinstance(stack, Mapping):
        raise ValueError("[stack] is required")
    _require(stack, "verl_commit", selected_stack.verl_commit)
    _require(stack, "rollout_backend", "vllm")
    _require(stack, "behavior_logprobs", "processed_logprobs")
    _require(stack, "vllm_enable_mm_embeds", True)
    _require(stack, "sharding_strategy", "fsdp2")
    _require(stack, "world_size", 2)
    _require(stack, "physical_gpu_ids", [2, 3])
    _require(stack, "logical_gpu_ids", [0, 1])
    _require(stack, "full_determinism", True)
    _require(stack, "adapter_dropout", 0.0)
    _require(stack, "trainer_mode", "sync")
    _require(stack, "asynchronous_staleness_steps", 0)

    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("[checkpoint] is required")
    _require(checkpoint, "format", "torch.distributed.checkpoint")
    _require(checkpoint, "strict", True)
    _require(checkpoint, "async_save", False)
    _require(checkpoint, "contents", ["model", "optimizer", "extra"])
    _require(checkpoint, "resume_parity_atol", 0.0)
    _require(checkpoint, "resume_parity_rtol", 0.0)

    objective = config.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("[objective] is required")
    _require(objective, "identity", "synthetic-fsdp2-mse-v1")
    _require(objective, "equation", "mean((model(x) - target) ** 2)")
    _require(objective, "normalization", "global element mean")
    _require(objective, "production_rl", False)

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("[model] is required")
    _require(model, "identity", "synthetic-tiny-fsdp2-model-v1")
    _require(model, "hidden_size", 16)
    _require(model, "layers", 2)
    _require(model, "dtype", "float32")
    _require(model, "seed", 20260719)
    return config


def _assert_installed_stack_identity(stack: AuditedCompatibilityStack) -> str:
    if sys.version_info[:2] != stack.python_major_minor:
        raise RuntimeError(
            f"selected {stack.selector!r} stack requires Python "
            f"{stack.python_major_minor[0]}.{stack.python_major_minor[1]}; "
            f"observed {sys.version_info.major}.{sys.version_info.minor}"
        )
    try:
        torch_distribution_version = metadata.version("torch")
        transformers_distribution_version = metadata.version("transformers")
        vllm_distribution_version = metadata.version("vllm")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"selected {stack.selector!r} stack distribution is not installed: "
            f"{error.name}"
        ) from error
    import torch

    torch_runtime_version = str(torch.__version__)
    observed = (
        torch_distribution_version,
        torch_runtime_version,
        transformers_distribution_version,
        vllm_distribution_version,
    )
    expected = (
        stack.torch_distribution_version,
        stack.torch_runtime_version,
        stack.transformers_distribution_version,
        stack.vllm_distribution_version,
    )
    if observed != expected:
        raise RuntimeError(
            f"installed stack differs from selected {stack.selector!r} identity: "
            f"expected torch_distribution={expected[0]!r}, "
            f"torch_runtime={expected[1]!r}, transformers={expected[2]!r}, "
            f"vllm={expected[3]!r}; "
            f"observed torch_distribution={observed[0]!r}, "
            f"torch_runtime={observed[1]!r}, transformers={observed[2]!r}, "
            f"vllm={observed[3]!r}"
        )
    return torch_runtime_version


def _environment_payload(
    *, live: bool, stack_selector: str = CONTROL_COMPATIBILITY_STACK
) -> dict[str, Any]:
    stack = audited_compatibility_stack(stack_selector)
    adapter_config = VerlAdapterConfig(
        runtime=VerlRuntimeRequirements(verl_commit=stack.verl_commit),
        max_tool_calls=2,
    )
    payload: dict[str, Any] = {
        "project_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "compatibility_stack": asdict(stack),
        "verl_candidate_commit": stack.verl_commit,
        "required_overrides": dict(adapter_config.public_config_overrides()),
        "required_environment": dict(adapter_config.required_environment()),
        "override_scope": "synthetic two-call fixture; production cap remains unset",
    }
    if live:
        torch_runtime_version = _assert_installed_stack_identity(stack)
        identity = verify_verl_distribution_identity(expected_commit=stack.verl_commit)
        api = load_verl_public_api(expected_commit=stack.verl_commit)
        payload.update(
            {
                "verl_distribution": {
                    "version": identity.package_version,
                    "source_url": identity.source_url,
                    "commit": identity.commit,
                    "source_kind": identity.source_kind,
                    "source_clean": identity.source_clean,
                },
                "versions": {
                    name: metadata.version(name)
                    for name in ("torch", "transformers", "vllm", "verl")
                },
                "torch_runtime_version": torch_runtime_version,
                "public_api": {
                    "agent_loop_output": api.agent_loop_output.__name__,
                    "agent_loop_manager": api.agent_loop_manager.__name__,
                    "data_proto": api.data_proto.__name__,
                    "fsdp_engine_config": api.fsdp_engine_config.__name__,
                    "checkpoint_handler": api.checkpoint_handler.__name__,
                },
            }
        )
    return payload


def _add_execution_authorization_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--gate-directory",
        type=Path,
        required=True,
        help="existing launch-gate directory containing the exact ready receipt",
    )
    parser.add_argument(
        "--authorization-token",
        type=Path,
        required=True,
        help="explicit operator-issued one-time launch authorization token",
    )
    parser.add_argument(
        "--freeze-override",
        type=Path,
        default=None,
        help=(
            "one-time override required only when the bound repository policy "
            "is frozen; it must be omitted in open mode"
        ),
    )


def _representation_training_authorization_identity(
    prepared: PreparedRepresentationLaunch,
) -> CLIExecutionAuthorizationIdentity:
    config = prepared.config
    return CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=REPRESENTATION_TRAINING_PHASE,
        command_id=_REPRESENTATION_COMMAND_ID,
        run_identity_sha256=config.canonical_config_sha256,
        parameters={
            **prepared.authorization_parameters(),
            "config_source_sha256": config.source_toml_sha256,
            "stop_after_global_step": (
                "none"
                if prepared.stop_after_global_step is None
                else str(prepared.stop_after_global_step)
            ),
            "nproc_per_node": str(config.fsdp2.world_size),
            "torchrun_executable": str(prepared.python_identity.declared_path),
            "torchrun_module": "torch.distributed.run",
            "world_size": str(config.fsdp2.world_size),
        },
    )


def _representation_internal_evaluation_authorization_identity(
    config: Any,
    *,
    config_binding: CanonicalConfigBinding,
    python_identity: PythonExecutableIdentity,
) -> CLIExecutionAuthorizationIdentity:
    return CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=REPRESENTATION_INTERNAL_EVALUATION_PHASE,
        command_id="tgvf-rl:run-representation-internal-evaluation:v1",
        run_identity_sha256=config.source_sha256,
        parameters={
            **config_binding.authorization_parameters(),
            **python_identity.authorization_parameters(),
            "artifact_manifest_sha256": config.artifact_manifest_sha256,
            "expected_global_step": str(config.expected_global_step),
            "expected_training_run_identity_sha256": (
                config.expected_run_identity_sha256
            ),
            "physical_gpu_id": str(config.physical_gpu_id),
        },
    )


def _policy_training_authorization_identity(
    prepared: Any,
    *,
    config_binding: CanonicalConfigBinding,
) -> CLIExecutionAuthorizationIdentity:
    config = prepared.config
    return CLIExecutionAuthorizationIdentity.create(
        run_id=config.run_id,
        phase=POLICY_TRAINING_PHASE,
        command_id=_POLICY_COMMAND_ID,
        run_identity_sha256=config.identity_sha256,
        parameters={
            **prepared.authorization_parameters(),
            **config_binding.authorization_parameters(),
            "config_source_sha256": config.source_sha256,
            "horizon_extension_sha256": (
                "none"
                if prepared.horizon_extension is None
                else prepared.horizon_extension.source_sha256
            ),
        },
    )


def _consume_command_authorization(
    args: argparse.Namespace,
    identity: CLIExecutionAuthorizationIdentity,
) -> dict[str, object]:
    return consume_cli_execution_authorization(
        identity,
        gate_directory=args.gate_directory,
        authorization_token_path=args.authorization_token,
        freeze_override_path=args.freeze_override,
    )


def _preflight_representation_launch(
    config: Any,
    *,
    config_binding: CanonicalConfigBinding,
    python_executable: Path,
    stop_after_global_step: int | None,
) -> PreparedRepresentationLaunch:
    """Complete all deterministic outer checks before consuming a token."""

    assert_canonical_runtime_launch_enabled()

    assert_loaded_config_matches_binding(
        config,
        config_binding,
        source_sha256_attribute="source_toml_sha256",
    )
    python_binding = bind_current_python_executable_for_exec(python_executable)
    try:
        python_identity = python_binding.identity
        from tgvf_rl.representation.training import runner as representation_runner

        representation_runner._validate_invocation_stop(  # noqa: SLF001
            config,
            stop_after_global_step,
        )
        representation_runner._verify_live_code_identity(config)  # noqa: SLF001
        environment, stripped_names = _representation_child_environment(config)
        command_prefix = _representation_command_prefix(
            config,
            python_executable=python_identity.declared_path,
            stop_after_global_step=stop_after_global_step,
        )
        return PreparedRepresentationLaunch(
            config=config,
            config_binding=config_binding,
            python_identity=python_identity,
            stop_after_global_step=stop_after_global_step,
            command_prefix=command_prefix,
            child_environment=tuple(sorted(environment.items())),
            stripped_environment_names=stripped_names,
            python_binding=python_binding,
        )
    except BaseException:
        python_binding.close()
        raise


def _run_representation_training(
    path: Path,
    *,
    stop_after_global_step: int | None,
) -> dict[str, object] | None:
    from tgvf_rl.representation.training.runner import run_representation_training

    return run_representation_training(
        path,
        stop_after_global_step=stop_after_global_step,
    )


def _load_representation_internal_evaluation_config(path: Path) -> Any:
    from tgvf_rl.representation.training.evaluation_runner import (
        load_representation_internal_evaluation_run_config,
    )

    return load_representation_internal_evaluation_run_config(path)


def _run_representation_internal_evaluation(
    config: Any,
    *,
    config_binding: CanonicalConfigBinding,
) -> dict[str, object]:
    from tgvf_rl.representation.training.evaluation_runner import (
        run_representation_internal_evaluation,
    )

    # Keep the binding recheck at the actual runner boundary.  The runner
    # receives the already-loaded immutable dataclass and never reopens the
    # authorized config path after the one-time token is consumed.
    verify_canonical_config_binding(config_binding)
    assert_loaded_config_matches_binding(
        config,
        config_binding,
        source_sha256_attribute="source_sha256",
    )
    return run_representation_internal_evaluation(config)


def _preflight_representation_internal_evaluation(config: Any) -> None:
    """Run every deterministic evaluation check before authorization consume."""

    # The current representation artifact is a legacy pickle loaded with
    # ``weights_only=False``.  Runtime closure therefore rejects this command
    # before the artifact path can be opened or a launch token consumed.
    assert_canonical_runtime_launch_enabled()

    from tgvf_rl.representation.training import evaluation_runner

    evaluation_runner._require_launch_environment(config)  # noqa: SLF001
    evaluation_runner._verify_live_code_identity(config)  # noqa: SLF001
    training = evaluation_runner.load_representation_training_config(
        config.training_config_path
    )
    evaluation_runner._require_file_sha256(  # noqa: SLF001
        config.training_config_path,
        config.training_config_sha256,
        name="training config",
    )
    evaluation_runner._require_file_sha256(  # noqa: SLF001
        config.artifact_path,
        config.artifact_file_sha256,
        name="Adapter artifact",
    )
    manifests = [
        (
            "ordered-group manifest",
            config.evaluation.ordered_group_manifest_path,
            config.evaluation.ordered_group_manifest_sha256,
        ),
        (
            "counterfactual manifest",
            config.evaluation.counterfactual_manifest_path,
            config.evaluation.counterfactual_manifest_sha256,
        ),
    ]
    if config.evaluation.grounding_manifest_path is not None:
        manifests.append(
            (
                "grounding manifest",
                config.evaluation.grounding_manifest_path,
                config.evaluation.grounding_manifest_sha256,
            )
        )
    for name, path, digest in manifests:
        if path is None or digest is None:
            raise RuntimeError(f"{name} identity is incomplete")
        evaluation_runner._require_file_sha256(path, digest, name=name)  # noqa: SLF001
    report_path = config.evaluation.report_path
    if report_path is None:
        raise RuntimeError("internal-evaluation report path is missing")
    if report_path.exists():
        raise FileExistsError(
            f"internal-evaluation report already exists: {report_path}"
        )
    if not report_path.parent.is_dir():
        raise FileNotFoundError(
            f"internal-evaluation report parent does not exist: {report_path.parent}"
        )
    export = evaluation_runner.load_rank_zero_adapter_owned_state_export(
        config.artifact_path
    )
    manifest = export.manifest
    run_identity = manifest.run_identity
    if evaluation_runner.state_digest(manifest) != config.artifact_manifest_sha256:
        raise ValueError("Adapter artifact manifest SHA256 mismatch")
    if (
        manifest.run_identity_sha256 != config.expected_run_identity_sha256
        or run_identity.identity_sha256 != config.expected_run_identity_sha256
        or manifest.global_step != config.expected_global_step
    ):
        raise ValueError("Adapter artifact run identity or global step mismatch")
    evaluation_runner._validate_training_artifact_binding(  # noqa: SLF001
        training,
        run_identity,
    )


def _load_policy_run_config(path: Path) -> Any:
    from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

    return load_policy_e2e_smoke_run_config(path)


def _load_policy_horizon_extension(path: Path, config: Any) -> Any:
    from tgvf_rl.policy.horizon_extension import load_policy_horizon_extension

    return load_policy_horizon_extension(path, config, validate_artifacts=True)


def _preflight_policy_run(
    config: Any,
    *,
    python_executable: Path,
    horizon_extension: Any | None,
    compile_prerequisite_manifest_path: Path,
) -> Any:
    assert_canonical_runtime_launch_enabled()

    from tgvf_rl.policy.launch import preflight_policy_launch_for_authorization

    return preflight_policy_launch_for_authorization(
        config,
        python_executable=python_executable,
        horizon_extension=horizon_extension,
        compile_prerequisite_manifest_path=compile_prerequisite_manifest_path,
    )


def _assert_worker_identity_parameters(
    identity: CLIExecutionAuthorizationIdentity,
    expected: Mapping[str, str],
) -> None:
    observed = dict(identity.parameters)
    for name, value in expected.items():
        if observed.get(name) != value:
            raise RuntimeError(
                f"representation worker identity differs from launcher parameter {name}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tgvf-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    info = subparsers.add_parser("compat-info", help="print accepted stack identity")
    info.add_argument(
        "--live",
        action="store_true",
        help="import the installed veRL candidate and verify exact provenance",
    )
    info.add_argument(
        "--stack",
        choices=tuple(AUDITED_COMPATIBILITY_STACKS),
        default=CONTROL_COMPATIBILITY_STACK,
        help="named audited compatibility stack (default: control)",
    )
    validate = subparsers.add_parser(
        "validate-smoke-config", help="validate a bounded TOML smoke identity"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--stack",
        choices=tuple(AUDITED_COMPATIBILITY_STACKS),
        default=CONTROL_COMPATIBILITY_STACK,
        help="named audited compatibility stack (default: control)",
    )
    validate_representation = subparsers.add_parser(
        "validate-representation-config",
        help="validate a complete Qwen3 representation-training TOML identity",
    )
    validate_representation.add_argument("path", type=Path)
    launch_representation = subparsers.add_parser(
        "launch-representation",
        help="authorize once, then replace this process with strict torchrun workers",
    )
    launch_representation.add_argument("path", type=Path)
    launch_representation.add_argument(
        "--stop-after-global-step",
        type=int,
        default=None,
        help=(
            "stop after this optimizer-boundary checkpoint without publishing "
            "the final Adapter; the configured scheduler horizon is unchanged"
        ),
    )
    launch_representation.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable).absolute(),
        help="absolute Python executable used for torch.distributed.run",
    )
    _add_execution_authorization_arguments(launch_representation)
    run_representation = subparsers.add_parser(
        "run-representation",
        help="internal torchrun worker; direct historical invocation is rejected",
    )
    run_representation.add_argument("path", type=Path)
    run_representation.add_argument(
        "--stop-after-global-step",
        type=int,
        default=None,
    )
    run_representation.add_argument(
        "--launcher-python-executable",
        type=Path,
        required=True,
    )
    run_representation.add_argument("--gate-directory", type=Path, required=True)
    run_representation.add_argument(
        "--launch-consumption-receipt",
        type=Path,
        required=True,
    )
    run_representation.add_argument(
        "--launch-consumption-sha256",
        required=True,
    )
    run_representation.add_argument(
        "--launcher-liveness-receipt",
        type=Path,
        required=True,
    )
    run_representation_evaluation = subparsers.add_parser(
        "run-representation-internal-evaluation",
        help="evaluate one completed representation Adapter on a single GPU",
    )
    run_representation_evaluation.add_argument("path", type=Path)
    _add_execution_authorization_arguments(run_representation_evaluation)
    compare_resume = subparsers.add_parser(
        "compare-representation-resume",
        help="compare continuous and teardown/resumed representation outputs",
    )
    compare_resume.add_argument("--continuous-artifact", type=Path, required=True)
    compare_resume.add_argument("--resumed-artifact", type=Path, required=True)
    compare_resume.add_argument("--continuous-checkpoint", type=Path, required=True)
    compare_resume.add_argument("--resumed-checkpoint", type=Path, required=True)
    compare_resume.add_argument("--continuous-metrics", type=Path, required=True)
    compare_resume.add_argument("--resumed-metrics", type=Path, required=True)
    validate_policy = subparsers.add_parser(
        "validate-policy-config",
        help="validate one strict non-formal Policy E2E smoke identity",
    )
    validate_policy.add_argument("path", type=Path)
    plan_policy = subparsers.add_parser(
        "plan-policy",
        help="print the exact upstream veRL Policy E2E launch plan without execution",
    )
    plan_policy.add_argument("path", type=Path)
    plan_policy.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable).absolute(),
        help="absolute Python executable for the rendered upstream command",
    )
    plan_policy.add_argument(
        "--horizon-extension",
        type=Path,
        default=None,
        help="explicit audited Policy horizon-extension JSON manifest",
    )
    plan_policy.add_argument(
        "--compile-prerequisite-manifest",
        type=Path,
        default=None,
        help=(
            "optional strict content-bound compile-prerequisite JSON manifest; "
            "omission is represented as an explicit launch blocker"
        ),
    )
    run_policy = subparsers.add_parser(
        "run-policy",
        help="replace this process with a launch-ready upstream veRL Policy E2E run",
    )
    run_policy.add_argument("path", type=Path)
    run_policy.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable).absolute(),
        help="absolute audited-stack Python executable (default: current Python)",
    )
    run_policy.add_argument(
        "--horizon-extension",
        type=Path,
        default=None,
        help="explicit audited Policy horizon-extension JSON manifest",
    )
    run_policy.add_argument(
        "--compile-prerequisite-manifest",
        type=Path,
        required=True,
        help="strict content-bound compile-prerequisite JSON manifest",
    )
    _add_execution_authorization_arguments(run_policy)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result: object | None = None
    try:
        if args.command == "compat-info":
            result = _environment_payload(live=args.live, stack_selector=args.stack)
        elif args.command == "validate-smoke-config":
            result = dict(validate_smoke_config(args.path, stack_selector=args.stack))
        elif args.command == "validate-representation-config":
            result = load_representation_training_config(args.path).validation_payload()
        elif args.command == "launch-representation":
            assert_canonical_runtime_launch_enabled()
            config_binding = bind_canonical_config_path(
                args.path,
                canonical_root=CANONICAL_REPRESENTATION_CONFIG_ROOT,
            )
            config = load_representation_training_config(config_binding.source_path)
            prepared = _preflight_representation_launch(
                config,
                config_binding=config_binding,
                python_executable=args.python,
                stop_after_global_step=args.stop_after_global_step,
            )
            try:
                identity = _representation_training_authorization_identity(prepared)
                consumption = _consume_command_authorization(
                    args,
                    identity,
                )
                worker_authorization = materialize_cli_worker_authorization(
                    identity,
                    consumption,
                    gate_directory=args.gate_directory,
                )
                _execute_representation_torchrun(
                    prepared,
                    launch_identity=identity,
                    gate_directory=args.gate_directory,
                    worker_authorization=worker_authorization,
                )
            finally:
                prepared.close_python_binding()
        elif args.command == "run-representation":
            launch_identity = verify_cli_worker_authorization_from_environment(
                expected_phase=REPRESENTATION_TRAINING_PHASE,
                expected_command_id=_REPRESENTATION_COMMAND_ID,
            )
            assert_canonical_runtime_launch_enabled()
            if (
                os.environ.get("TGVF_CLI_GATE_DIRECTORY")
                != str(args.gate_directory.expanduser().absolute())
                or os.environ.get("TGVF_CLI_CONSUMPTION_RECEIPT_PATH")
                != str(args.launch_consumption_receipt.expanduser().absolute())
                or os.environ.get("TGVF_CLI_CONSUMPTION_RECEIPT_SHA256")
                != args.launch_consumption_sha256
                or os.environ.get("TGVF_CLI_LAUNCHER_LIVENESS_RECEIPT_PATH")
                != str(args.launcher_liveness_receipt.expanduser().absolute())
            ):
                raise RuntimeError(
                    "representation worker argv differs from authorization environment"
                )
            config_binding = bind_canonical_config_path(
                args.path,
                canonical_root=CANONICAL_REPRESENTATION_CONFIG_ROOT,
            )
            config = load_representation_training_config(config_binding.source_path)
            assert_loaded_config_matches_binding(
                config,
                config_binding,
                source_sha256_attribute="source_toml_sha256",
            )
            python_identity = bind_current_python_executable(
                args.launcher_python_executable
            )
            _assert_worker_identity_parameters(
                launch_identity,
                {
                    **config_binding.authorization_parameters(),
                    **python_identity.authorization_parameters(),
                    "config_source_sha256": config.source_toml_sha256,
                    "stop_after_global_step": (
                        "none"
                        if args.stop_after_global_step is None
                        else str(args.stop_after_global_step)
                    ),
                    "world_size": str(config.fsdp2.world_size),
                },
            )
            result = _run_representation_training(
                config_binding.source_path,
                stop_after_global_step=args.stop_after_global_step,
            )
        elif args.command == "run-representation-internal-evaluation":
            assert_canonical_runtime_launch_enabled()
            config_binding = bind_canonical_config_path(
                args.path,
                canonical_root=CANONICAL_EVALUATION_CONFIG_ROOT,
            )
            config = _load_representation_internal_evaluation_config(
                config_binding.source_path
            )
            assert_loaded_config_matches_binding(
                config,
                config_binding,
                source_sha256_attribute="source_sha256",
            )
            python_identity = bind_current_python_executable(sys.executable)
            _preflight_representation_internal_evaluation(config)
            identity = _representation_internal_evaluation_authorization_identity(
                config,
                config_binding=config_binding,
                python_identity=python_identity,
            )
            _consume_command_authorization(
                args,
                identity,
            )
            verify_python_executable_identity(python_identity)
            result = _run_representation_internal_evaluation(
                config,
                config_binding=config_binding,
            )
        elif args.command == "compare-representation-resume":
            from tgvf_rl.representation.training.resume_parity import (
                compare_representation_resume_lanes,
            )

            result = asdict(
                compare_representation_resume_lanes(
                    continuous_artifact_path=args.continuous_artifact,
                    resumed_artifact_path=args.resumed_artifact,
                    continuous_checkpoint_path=args.continuous_checkpoint,
                    resumed_checkpoint_path=args.resumed_checkpoint,
                    continuous_metrics_path=args.continuous_metrics,
                    resumed_metrics_path=args.resumed_metrics,
                )
            )
        elif args.command == "validate-policy-config":
            from tgvf_rl.policy.run_config import (
                load_policy_e2e_smoke_run_config,
            )

            config = load_policy_e2e_smoke_run_config(args.path)
            result = {
                "schema_version": config.schema_version,
                "run_id": config.run_id,
                "run_identity_sha256": config.identity_sha256,
                "source_sha256": config.source_sha256,
                "formal_pilot": config.formal_pilot,
                "physical_gpu_ids": list(config.distributed.physical_gpu_ids),
                "world_size": config.distributed.world_size,
                "placement": config.distributed.placement,
                "vllm_tensor_parallel_size": (
                    config.distributed.vllm_tensor_parallel_size
                ),
                "vllm_capacity": {
                    "gpu_memory_utilization": (
                        config.capacity.vllm_gpu_memory_utilization
                    ),
                    "max_num_batched_tokens": (
                        config.capacity.vllm_max_num_batched_tokens
                    ),
                    "max_model_len": config.capacity.vllm_max_model_len,
                    "max_num_seqs": config.capacity.vllm_max_num_seqs,
                },
                "gpu_work_launched": False,
            }
        elif args.command == "plan-policy":
            from tgvf_rl.policy.launch import build_policy_launch_record
            from tgvf_rl.policy.run_config import (
                load_policy_e2e_smoke_run_config,
            )

            config = load_policy_e2e_smoke_run_config(args.path)
            extension = None
            if args.horizon_extension is not None:
                from tgvf_rl.policy.horizon_extension import (
                    load_policy_horizon_extension,
                )

                extension = load_policy_horizon_extension(
                    args.horizon_extension, config, validate_artifacts=True
                )
            result = build_policy_launch_record(
                config,
                python_executable=args.python,
                horizon_extension=extension,
                compile_prerequisite_manifest_path=(args.compile_prerequisite_manifest),
            )
            result["gpu_work_launched"] = False
        elif args.command == "run-policy":
            assert_canonical_runtime_launch_enabled()
            config_binding = bind_canonical_config_path(
                args.path,
                canonical_root=CANONICAL_POLICY_CONFIG_ROOT,
            )
            config = _load_policy_run_config(config_binding.source_path)
            assert_loaded_config_matches_binding(
                config,
                config_binding,
                source_sha256_attribute="source_sha256",
            )
            extension = None
            if args.horizon_extension is not None:
                extension = _load_policy_horizon_extension(
                    args.horizon_extension, config
                )
            prepared = _preflight_policy_run(
                config,
                python_executable=args.python,
                horizon_extension=extension,
                compile_prerequisite_manifest_path=(args.compile_prerequisite_manifest),
            )
            try:
                _assert_installed_stack_identity(
                    audited_compatibility_stack(CONTROL_COMPATIBILITY_STACK)
                )
                verify_verl_distribution_identity(
                    expected_commit=audited_compatibility_stack(
                        CONTROL_COMPATIBILITY_STACK
                    ).verl_commit
                )
                identity = _policy_training_authorization_identity(
                    prepared,
                    config_binding=config_binding,
                )
                consumption = _consume_command_authorization(
                    args,
                    identity,
                )
                worker_authorization = materialize_cli_worker_authorization(
                    identity,
                    consumption,
                    gate_directory=args.gate_directory,
                )
                verify_canonical_config_binding(config_binding)
                _execute_policy_run(
                    prepared,
                    launch_identity=identity,
                    worker_authorization=worker_authorization,
                    gate_directory=args.gate_directory,
                )
            finally:
                prepared.close_python_binding()
        else:  # pragma: no cover - argparse owns the command choices
            raise AssertionError(f"unhandled command {args.command}")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"contract error: {error}", file=sys.stderr)
        return 2
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, default=list))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
