"""Small, fail-closed command line surface for validation and bounded runs."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping, Sequence

from tgvf_rl import SCHEMA_VERSION, __version__
from tgvf_rl.framework.verl import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    VerlAdapterConfig,
    load_verl_public_api,
    verify_verl_distribution_identity,
)
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)


SMOKE_CONFIG_SCHEMA = "tgvf-fsdp2-smoke-v1"


def _require(mapping: Mapping[str, Any], key: str, expected: object) -> None:
    value = mapping.get(key)
    if value != expected:
        raise ValueError(f"{key} must be {expected!r}, got {value!r}")


def validate_smoke_config(path: Path) -> Mapping[str, Any]:
    """Validate the bounded infrastructure smoke without adding hidden defaults."""

    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _require(config, "schema_version", SMOKE_CONFIG_SCHEMA)
    _require(config, "scope", "synthetic_infrastructure_only")

    stack = config.get("stack")
    if not isinstance(stack, Mapping):
        raise ValueError("[stack] is required")
    _require(stack, "verl_commit", SPIKE_CANDIDATE_VERL_COMMIT)
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


def _environment_payload(*, live: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "verl_candidate_commit": SPIKE_CANDIDATE_VERL_COMMIT,
        "required_overrides": dict(
            VerlAdapterConfig(max_tool_calls=2).public_config_overrides()
        ),
        "required_environment": dict(VerlAdapterConfig.required_environment()),
        "override_scope": "synthetic two-call fixture; production cap remains unset",
    }
    if live:
        identity = verify_verl_distribution_identity()
        api = load_verl_public_api()
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tgvf-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    info = subparsers.add_parser("compat-info", help="print accepted stack identity")
    info.add_argument(
        "--live",
        action="store_true",
        help="import the installed veRL candidate and verify exact provenance",
    )
    validate = subparsers.add_parser(
        "validate-smoke-config", help="validate a bounded TOML smoke identity"
    )
    validate.add_argument("path", type=Path)
    validate_representation = subparsers.add_parser(
        "validate-representation-config",
        help="validate a complete Qwen3 representation-training TOML identity",
    )
    validate_representation.add_argument("path", type=Path)
    run_representation = subparsers.add_parser(
        "run-representation",
        help="run a strict Qwen3 representation configuration under torchrun",
    )
    run_representation.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compat-info":
            result = _environment_payload(live=args.live)
        elif args.command == "validate-smoke-config":
            result = dict(validate_smoke_config(args.path))
        elif args.command == "validate-representation-config":
            result = load_representation_training_config(
                args.path
            ).validation_payload()
        elif args.command == "run-representation":
            from tgvf_rl.representation.training.runner import (
                run_representation_training,
            )

            result = run_representation_training(args.path)
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
