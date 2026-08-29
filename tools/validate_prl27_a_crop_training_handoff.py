#!/usr/bin/env python3
"""Fail-closed gates for the PRL-26-C to corrected PRL-27-A handoff."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from validate_prl26_train512_training_handoff import (  # noqa: E402
    _canonical_tracker_step,
    _read_json,
    _read_json_lines,
    _validate_generation,
    _validate_metrics_rows,
    resource_release_audit,
    validate_source_completion,
)
from tgvf_rl.environment.native_appender import (  # noqa: E402
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256,
    render_qwen_native_matched_crop_success_environment_text,
)
from tgvf_rl.environment import ResponseBudgetScope  # noqa: E402
from tgvf_rl.framework.verl.policy_live_runtime import (  # noqa: E402
    _rp66_matched_source_route,
    _rp66_response_budget_controls,
    _success_environment_text_renderer,
)
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (  # noqa: E402
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile  # noqa: E402


PIXEL512 = 262_144
TARGET_STEP = 32
EXPECTED_GPU_IDS = tuple(range(8))
EXPECTED_SOURCE_RUN_ID = (
    "PRL-26-C-TRAIN512-S32-PARITY-TGVF-SHORT-"
    "QWEN3-INSTRUCT-BS16-N16-TEACHER25-WS8"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "17efa65a8622f36c337a0691bed2dd9acf799562a2109595d8a43acf3b7ebe17"
)
EXPECTED_SOURCE_IDENTITY_SHA256 = (
    "5a11f22ce58b3497c6f854bab5a68841fe02e46639735c0a3c23f1fe6b205b28"
)
EXPECTED_TARGET_RUN_ID = (
    "PRL-27-A-TRAIN512-S32-CROP-EXACT-CONTINUATION-"
    "QWEN3-INSTRUCT-BS16-N16-TEACHER25-WS8"
)
EXPECTED_TARGET_CONFIG_SHA256 = (
    "20f898a9121d0d656dd4ac9b735178416a63f0d79a55256072f53e476483a9ff"
)
EXPECTED_TARGET_IDENTITY_SHA256 = (
    "68f9a2c7c528be660b9bf3a47a31c4c403fe5faef6066b0355d0b9230dad8d1d"
)
EXPECTED_CORE_FIX_COMMIT = "ecddc379d392d154c91783d7651528b20d40afba"
EXPECTED_CONTINUATION_SHA256 = (
    "f745fa6cfcc3ba9eb27125a49581fd823fb5930b7b0a51b28e51982999fa2d0a"
)
EXPECTED_TARGET_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-27-A-train512-s32-crop-exact-continuation-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
_AUTHORIZATION_SCHEMA = "tgvf.prl27-a-crop-fresh-authorization.v1"


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _load_config(path: Path) -> Any:
    return load_policy_e2e_smoke_run_config(
        path.resolve(), allow_external_agent_loop_config=True
    )


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"git identity check failed: {error}") from error
    return result.stdout.strip()


def _require_ancestor(repository: Path, ancestor: str, descendant: str) -> None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"git ancestry check failed: {error}") from error
    if result.returncode != 0:
        _fail("admitted training HEAD does not descend from the Crop core fix")


def _validate_source_contract(config: Any) -> None:
    if (
        config.run_id != EXPECTED_SOURCE_RUN_ID
        or config.source_sha256 != EXPECTED_SOURCE_CONFIG_SHA256
        or config.identity_sha256 != EXPECTED_SOURCE_IDENTITY_SHA256
        or config.policy.image_max_pixels != PIXEL512
        or config.training.maximum_optimizer_steps != TARGET_STEP
        or config.distributed.world_size != 8
        or config.distributed.physical_gpu_ids != EXPECTED_GPU_IDS
    ):
        _fail("PRL-26-C source contract differs")


def _validate_target_contract(config: Any, repository: Path) -> dict[str, object]:
    if (
        config.schema_version
        != POLICY_E2E_CROP_TFREE_EXACT_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        or config.run_id != EXPECTED_TARGET_RUN_ID
        or config.source_sha256 != EXPECTED_TARGET_CONFIG_SHA256
        or config.identity_sha256 != EXPECTED_TARGET_IDENTITY_SHA256
        or config.code.commit != EXPECTED_CORE_FIX_COMMIT
        or config.policy.image_max_pixels != PIXEL512
        or config.protocol.tool_profile is not NativeToolCapabilityProfile.CROP_ONLY
        or config.protocol.enabled_tool_names != ("image_zoom_in_tool",)
        or config.protocol.maximum_tool_calls != 6
        or config.policy.sampling.stop_strings != ("</tool_call>",)
        or config.policy.sampling.include_stop_str_in_output is not True
        or config.policy.sampling.ignore_eos is not False
        or config.policy.sampling.temperature != 1.0
        or config.policy.sampling.do_sample is not True
        or config.training.maximum_optimizer_steps != TARGET_STEP
        or config.training.checkpoint_steps != (0, 8, 16, 24, 32)
        or config.training.permanent_checkpoint_steps != (8, 16, 24, 32)
        or config.training.resume_mode != "auto"
        or config.training.resume_from_path is not None
        or config.distributed.world_size != 8
        or config.distributed.physical_gpu_ids != EXPECTED_GPU_IDS
        or config.output.root != EXPECTED_TARGET_ROOT
        or config.output.checkpoint_directory != EXPECTED_TARGET_ROOT / "checkpoints"
        or config.output.metrics_path != EXPECTED_TARGET_ROOT / "metrics.jsonl"
    ):
        _fail("PRL-27-A target contract differs")
    try:
        config.output.root.relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        _fail("PRL-27-A output root must remain outside the clean worktree")

    renderer = _success_environment_text_renderer(
        tool_profile=config.protocol.tool_profile,
        matched_visual_observation=True,
    )
    if (
        renderer is not render_qwen_native_matched_crop_success_environment_text
        or QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256
        != EXPECTED_CONTINUATION_SHA256
        or _rp66_matched_source_route(config, {"data_source": "vstar"})
        != (False, True)
    ):
        _fail("PRL-27-A exact matched-Crop continuation route differs")
    budget_scope, single_response_tokens = _rp66_response_budget_controls(
        launch_mode="formal",
        direct_only=False,
        matched_visual_observation=True,
    )
    if (
        budget_scope is not ResponseBudgetScope.TOTAL_RESPONSE
        or single_response_tokens != 10_240
    ):
        _fail("PRL-27-A formal response-budget route differs")

    plan = build_trainable_tgvf_verl_launch_plan(
        config, mode="formal", target_step=TARGET_STEP
    )
    custom = plan.overrides.get("actor_rollout_ref.rollout.custom")
    protocol = custom.get("protocol") if isinstance(custom, Mapping) else None
    if (
        plan.environment.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"
        or plan.overrides.get("data.mm_processor_kwargs.max_pixels") != PIXEL512
        or plan.overrides.get("actor_rollout_ref.rollout.n") != 16
        or plan.overrides.get("data.train_batch_size") != 16
        or plan.overrides.get("actor_rollout_ref.actor.optim.total_training_steps")
        != TARGET_STEP
        or plan.overrides.get("actor_rollout_ref.model.external_lib")
        != "tgvf_rl.framework.verl.trainable_crop_external"
        or not isinstance(protocol, Mapping)
        or protocol.get("maximum_tool_calls") != 6
    ):
        _fail("PRL-27-A composed formal launch plan differs")
    return {
        "continuation_sha256": EXPECTED_CONTINUATION_SHA256,
        "response_budget_scope": budget_scope.value,
        "single_response_max_tokens": single_response_tokens,
    }


def validate_contracts(
    *,
    repository: Path,
    source_config_path: Path,
    target_config_path: Path,
    admitted_head: str,
    require_clean: bool,
) -> dict[str, object]:
    repository = repository.resolve()
    if len(admitted_head) != 40 or any(c not in "0123456789abcdef" for c in admitted_head):
        _fail("admitted training HEAD is malformed")
    if _git_output(repository, "rev-parse", "--show-toplevel") != str(repository):
        _fail("training repository root differs")
    if _git_output(repository, "rev-parse", "HEAD") != admitted_head:
        _fail("training repository HEAD differs")
    if require_clean and _git_output(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        _fail("training repository is dirty")
    _require_ancestor(repository, EXPECTED_CORE_FIX_COMMIT, admitted_head)

    source = _load_config(source_config_path)
    target = _load_config(target_config_path)
    _validate_source_contract(source)
    target_runtime = _validate_target_contract(target, repository)
    return {
        "schema_version": "tgvf.prl27-a-training-contract-audit.v1",
        "status": "accepted",
        "admitted_head": admitted_head,
        "source_run_id": source.run_id,
        "source_identity_sha256": source.identity_sha256,
        "target_run_id": target.run_id,
        "target_identity_sha256": target.identity_sha256,
        "target_output_root": str(target.output.root),
        **target_runtime,
    }


def _authorization_payload(config: Any, *, admitted_head: str) -> dict[str, object]:
    return {
        "schema_version": _AUTHORIZATION_SCHEMA,
        "status": "fresh-s0-authorized",
        "run_id": config.run_id,
        "run_config_identity_sha256": config.identity_sha256,
        "run_config_file_sha256": config.source_sha256,
        "core_fix_commit": EXPECTED_CORE_FIX_COMMIT,
        "continuation_sha256": EXPECTED_CONTINUATION_SHA256,
        "output_root": str(config.output.root),
        "admitted_training_head": admitted_head,
        "root_absent_at_authorization": True,
    }


def _validate_authorization(value: object, config: Any, *, admitted_head: str) -> None:
    if not isinstance(value, dict):
        _fail("PRL-27-A fresh-root authorization is malformed")
    authorized_at = value.get("authorized_at_utc")
    content = {key: item for key, item in value.items() if key != "authorized_at_utc"}
    if not isinstance(authorized_at, str) or content != _authorization_payload(
        config, admitted_head=admitted_head
    ):
        _fail("PRL-27-A fresh-root authorization identity differs")


def validate_target_readiness(
    *, config_path: Path, authorization_path: Path, admitted_head: str
) -> dict[str, object]:
    config = _load_config(config_path)
    _validate_target_contract(config, REPOSITORY_ROOT)
    root = config.output.root
    root_exists = os.path.lexists(root)
    authorization_exists = os.path.lexists(authorization_path)
    if authorization_path.is_symlink():
        _fail("PRL-27-A fresh-root authorization cannot be a symlink")
    if root_exists:
        if root.is_symlink() or not root.is_dir():
            _fail("PRL-27-A output root is not a real directory")
        if not authorization_exists:
            _fail("PRL-27-A root exists without its fresh-S0 authorization")
        authorization = _read_json(
            authorization_path, "PRL-27-A fresh-root authorization"
        )
        _validate_authorization(authorization, config, admitted_head=admitted_head)
        tracker = config.output.checkpoint_directory / "latest_checkpointed_iteration.txt"
        step = _canonical_tracker_step(tracker, "PRL-27-A")
        if step > TARGET_STEP:
            _fail("PRL-27-A tracker exceeds S32")
        _validate_generation(
            config,
            config.output.checkpoint_directory / f"global_step_{step}",
            optimizer_step=step,
        )
        metrics = _read_json_lines(config.output.metrics_path, "PRL-27-A metrics")
        _validate_metrics_rows(
            metrics, expected_step=step, owner="PRL-27-A metrics"
        )
        launch_mode = "resume"
    else:
        if authorization_exists:
            authorization = _read_json(
                authorization_path, "PRL-27-A fresh-root authorization"
            )
            _validate_authorization(authorization, config, admitted_head=admitted_head)
        else:
            authorization_path.parent.mkdir(parents=True, exist_ok=True)
            authorization = _authorization_payload(config, admitted_head=admitted_head)
            authorization["authorized_at_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                with authorization_path.open("x", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            authorization,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as error:
                raise RuntimeError(
                    "PRL-27-A fresh-root authorization raced another handoff"
                ) from error
        if os.path.lexists(root):
            _fail("PRL-27-A output root appeared during fresh-S0 authorization")
        step = 0
        launch_mode = "fresh-s0"
    return {
        "schema_version": "tgvf.prl27-a-crop-launch-readiness.v1",
        "status": "accepted",
        "launch_mode": launch_mode,
        "run_id": config.run_id,
        "run_identity_sha256": config.identity_sha256,
        "output_root": str(root),
        "checkpointed_step": step,
        "authorization_path": str(authorization_path.resolve()),
    }


def _write_result(value: Mapping[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contracts = commands.add_parser("contracts")
    contracts.add_argument("--repository", required=True, type=Path)
    contracts.add_argument("--source-config", required=True, type=Path)
    contracts.add_argument("--target-config", required=True, type=Path)
    contracts.add_argument("--admitted-head", required=True)
    contracts.add_argument("--require-clean", action="store_true")

    source = commands.add_parser("source-complete")
    source.add_argument("--config", required=True, type=Path)
    source.add_argument("--events", required=True, type=Path)

    target = commands.add_parser("target-ready")
    target.add_argument("--config", required=True, type=Path)
    target.add_argument("--authorization", required=True, type=Path)
    target.add_argument("--admitted-head", required=True)

    complete = commands.add_parser("target-complete")
    complete.add_argument("--config", required=True, type=Path)
    complete.add_argument("--events", required=True, type=Path)

    resources = commands.add_parser("resources-free")
    resources.add_argument("--memory-threshold-mib", type=int, default=32)

    args = parser.parse_args(argv)
    try:
        if args.command == "contracts":
            result = validate_contracts(
                repository=args.repository,
                source_config_path=args.source_config,
                target_config_path=args.target_config,
                admitted_head=args.admitted_head,
                require_clean=args.require_clean,
            )
        elif args.command == "source-complete":
            source_config = _load_config(args.config)
            _validate_source_contract(source_config)
            result = validate_source_completion(
                config_path=args.config,
                events_path=args.events,
                target_step=TARGET_STEP,
            )
        elif args.command == "target-ready":
            result = validate_target_readiness(
                config_path=args.config,
                authorization_path=args.authorization,
                admitted_head=args.admitted_head,
            )
        elif args.command == "target-complete":
            target_config = _load_config(args.config)
            _validate_target_contract(target_config, REPOSITORY_ROOT)
            result = validate_source_completion(
                config_path=args.config,
                events_path=args.events,
                target_step=TARGET_STEP,
            )
        else:
            result = resource_release_audit(
                memory_threshold_mib=args.memory_threshold_mib
            )
    except RuntimeError as error:
        print(f"PRL-27-A handoff rejected: {error}", file=sys.stderr)
        return 1
    _write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "validate_contracts",
    "validate_target_readiness",
]
