#!/usr/bin/env python3
"""Validate or run a private or production-source answer-utility candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_mode_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one formal500 private Adapter or the exact RP66 production "
            "source on the same first200 held-out manifest."
        )
    )
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--run-config", type=Path)
    candidate.add_argument("--production-source", action="store_true")
    parser.add_argument("--source-evaluation-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--eos-token-id", type=int, action="append")
    parser.add_argument(
        "--decode-mode", choices=("cached", "no_cache"), default="cached"
    )
    parser.add_argument(
        "--arm-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum compatible same-sample arms per cached decode batch; "
            "the default 1 preserves scalar generation"
        ),
    )
    parser.add_argument("--arm", action="append")
    parser.add_argument("--include-direct-replacement", action="store_true")
    parser.add_argument("--physical-gpu-id", type=int)
    return parser


def _configured_gpu(path: Path) -> int:
    payload = tomllib.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    value = payload.get("execution", {}).get("physical_gpu_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("run config execution.physical_gpu_id is invalid")
    return value


def _enter_launch_environment(gpu_id: int) -> None:
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("physical GPU ID must be non-negative")
    required = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    conflicts = {
        name: (os.environ.get(name), expected)
        for name, expected in required.items()
        if os.environ.get(name) is not None and os.environ.get(name) != expected
    }
    if conflicts:
        raise ValueError(f"launch environment conflicts with evaluator: {conflicts}")
    marker = "TGVF_ANSWER_UTILITY_EVALUATION_LAUNCH_READY"
    if os.environ.get(marker) == "1":
        if any(os.environ.get(name) != value for name, value in required.items()):
            raise ValueError("evaluation re-exec environment is incomplete")
        return
    environment = dict(os.environ)
    environment.update(required)
    environment[marker] = "1"
    os.execve(sys.executable, (sys.executable, *sys.argv), environment)
    raise RuntimeError("evaluation launcher re-exec unexpectedly returned")


def main() -> None:
    args = _parser().parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/run_representation_answer_utility_evaluation.py",
        selected_mode="validate" if args.validate_only else "execute",
        read_only_modes=("validate",),
        blocked_modes=("execute",),
    )
    run_config = (
        None if args.run_config is None else args.run_config.expanduser().resolve()
    )
    source_config = args.source_evaluation_config.expanduser().resolve()
    if not args.validate_only:
        if args.output_root is None:
            raise ValueError("--output-root is required unless --validate-only is used")
        gpu_id = (
            _configured_gpu(source_config if args.production_source else run_config)
            if args.physical_gpu_id is None
            else args.physical_gpu_id
        )
        _enter_launch_environment(gpu_id)
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from tgvf_rl.representation.experiments.answer_utility.evaluation import (  # noqa: E402
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS,
        AnswerUtilityEvaluationArm,
        run_answer_utility_evaluation,
        run_production_source_answer_utility_evaluation,
        validate_answer_utility_evaluation,
        validate_production_source_answer_utility_evaluation,
    )

    arms = (
        tuple(AnswerUtilityEvaluationArm(value) for value in args.arm)
        if args.arm
        else DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    )
    if args.include_direct_replacement:
        arms = (
            *arms,
            AnswerUtilityEvaluationArm.DIRECT_ZERO_REPLACEMENT,
            AnswerUtilityEvaluationArm.DIRECT_CORRECT_REPLACEMENT,
            AnswerUtilityEvaluationArm.DIRECT_WRONG_REPLACEMENT,
        )
    keywords = {
        "arms": arms,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_ids": args.eos_token_id,
        "decode_mode": args.decode_mode,
        "arm_batch_size": args.arm_batch_size,
        "group_start": args.group_start,
        "group_limit": args.group_limit,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    if args.validate_only:
        if args.production_source:
            result = validate_production_source_answer_utility_evaluation(
                source_config, **keywords
            )
        else:
            assert run_config is not None
            result = validate_answer_utility_evaluation(
                run_config, source_config, **keywords
            )
    else:
        assert args.output_root is not None
        if args.production_source:
            result = run_production_source_answer_utility_evaluation(
                source_config,
                output_root=args.output_root.expanduser().resolve(),
                **keywords,
            )
        else:
            assert run_config is not None
            result = run_answer_utility_evaluation(
                run_config,
                source_config,
                output_root=args.output_root.expanduser().resolve(),
                **keywords,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
