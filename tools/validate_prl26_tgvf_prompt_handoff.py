#!/usr/bin/env python3
"""Fail-closed gates for the PRL-26 C/D TGVF prompt handoff."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tools.validate_prl26_train512_training_handoff import (  # noqa: E402
    _canonical_tracker_step,
    _read_json,
    _read_json_lines,
    _validate_generation,
    _validate_metrics_rows,
)
from tgvf_rl.policy.run_config import (  # noqa: E402
    POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (  # noqa: E402
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (  # noqa: E402
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
)


_IMPLEMENTATION_COMMIT = "396a25819871f753f40242b5137e4c6f9fd49348"
_EXPECTED_FORMAL_GPU_IDS = tuple(range(8))
_EXPECTED_CHECKPOINTS = (0, 8, 16, 24, 32)
_EXPECTED_PERMANENT = (8, 16, 24, 32)
_RESULT_SCHEMA = "tgvf.prl26-train512-s32-pixel512-results.v1"
_PROVENANCE_SCHEMA = "tgvf.prl15-launch-provenance.v1"


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("TGVF handoff Git identity is unavailable") from error


def _leaf_differences(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        result: set[tuple[str, ...]] = set()
        for key in keys:
            result.update(
                _leaf_differences(left.get(key), right.get(key), path + (str(key),))
            )
        return result
    return set() if left == right else {path}


def _toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"run config is unreadable: {path}") from error


def _validate_arm_config(
    config: Any,
    *,
    schema: str,
    prompt_sha256: str,
    canary: bool,
    physical_gpu_ids: tuple[int, ...],
) -> None:
    expected_steps = 1 if canary else 32
    expected_checkpoints = (0, 1) if canary else _EXPECTED_CHECKPOINTS
    expected_permanent = (1,) if canary else _EXPECTED_PERMANENT
    expected_resume = "disable" if canary else "auto"
    if (
        config.schema_version != schema
        or config.code.commit != _IMPLEMENTATION_COMMIT
        or config.policy.image_max_pixels != 512 * 512
        or config.protocol.prompt_sha256 != prompt_sha256
        or config.protocol.tool_profile.value != "tgvf_only"
        or config.protocol.enabled_tool_names != ("tgvf_focus_tool",)
        or config.protocol.maximum_tool_calls != 6
        or config.policy.sampling.stop_strings != ("</tool_call>",)
        or config.policy.sampling.include_stop_str_in_output is not True
        or config.policy.sampling.stop_token_ids != (151_645,)
        or config.distributed.physical_gpu_ids != physical_gpu_ids
        or config.distributed.logical_gpu_ids != tuple(range(len(physical_gpu_ids)))
        or config.distributed.world_size != len(physical_gpu_ids)
        or config.training.maximum_optimizer_steps != expected_steps
        or config.scheduler.total_steps != expected_steps
        or config.training.checkpoint_steps != expected_checkpoints
        or config.training.permanent_checkpoint_steps != expected_permanent
        or config.training.resume_mode != expected_resume
        or config.training.resume_from_path is not None
        or config.representation.adapter_update_mode.value != "frozen_adapter"
        or config.optimizer.learning_rate != 1.0e-6
        or config.dataset.runtime_binding.schedule_seed != 42
        or config.rollout_rng.master_seed != 42
        or config.reward.tool_utility_reward_enabled is not False
        or config.reward.focus_reward_enabled is not False
        or config.reward.grounding_reward_enabled is not False
        or config.reward.visual_quality_judge_config_path is not None
    ):
        _fail(f"TGVF prompt arm contract differs: {config.run_id}")


def validate_contracts(
    *,
    repository: Path,
    short_canary_path: Path,
    full_canary_path: Path,
    short_formal_path: Path,
    full_formal_path: Path,
) -> dict[str, object]:
    root = repository.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        _fail("TGVF handoff repository root differs")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("TGVF handoff repository must be clean")
    head = _git(root, "rev-parse", "HEAD")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", _IMPLEMENTATION_COMMIT, head),
        cwd=root,
        check=False,
    ).returncode:
        _fail("TGVF prompt implementation is not an ancestor of handoff HEAD")

    paths = {
        "short_canary": short_canary_path.resolve(),
        "full_canary": full_canary_path.resolve(),
        "short_formal": short_formal_path.resolve(),
        "full_formal": full_formal_path.resolve(),
    }
    for path in paths.values():
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("TGVF run config escaped the clean repository") from error

    configs = {name: load_policy_e2e_smoke_run_config(path) for name, path in paths.items()}
    _validate_arm_config(
        configs["short_canary"],
        schema=POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        prompt_sha256=TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        canary=True,
        physical_gpu_ids=(0, 1, 2, 3),
    )
    _validate_arm_config(
        configs["full_canary"],
        schema=(
            POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        ),
        prompt_sha256=TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        canary=True,
        physical_gpu_ids=(4, 5, 6, 7),
    )
    _validate_arm_config(
        configs["short_formal"],
        schema=POLICY_E2E_TGVF_SHORT_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA,
        prompt_sha256=TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        canary=False,
        physical_gpu_ids=_EXPECTED_FORMAL_GPU_IDS,
    )
    _validate_arm_config(
        configs["full_formal"],
        schema=(
            POLICY_E2E_TGVF_TARGET_GUIDE_V2_TFREE_PIXEL512_PARITY_RUN_CONFIG_SCHEMA
        ),
        prompt_sha256=TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        canary=False,
        physical_gpu_ids=_EXPECTED_FORMAL_GPU_IDS,
    )

    allowed = {
        ("schema_version",),
        ("run_id",),
        ("protocol", "prompt_sha256"),
        ("output", "root"),
        ("output", "checkpoint_directory"),
        ("output", "metrics_path"),
    }
    if _leaf_differences(
        _toml(paths["short_formal"]), _toml(paths["full_formal"])
    ) != allowed:
        _fail("formal Short and Full configs differ beyond Target treatment identity")
    canary_allowed = allowed | {("distributed", "physical_gpu_ids")}
    if _leaf_differences(
        _toml(paths["short_canary"]), _toml(paths["full_canary"])
    ) != canary_allowed:
        _fail("canary Short and Full configs differ beyond Target treatment and GPUs")

    for config in configs.values():
        try:
            config.output.root.resolve().relative_to(root)
        except ValueError:
            pass
        else:
            _fail("TGVF output root is inside the clean training repository")
    return {
        "schema_version": "tgvf.prl26-tgvf-prompt-handoff-contracts.v1",
        "status": "accepted",
        "repository": str(root),
        "repository_head": head,
        "implementation_commit": _IMPLEMENTATION_COMMIT,
        "configs": {
            name: {
                "path": str(paths[name]),
                "file_sha256": configs[name].source_sha256,
                "identity_sha256": configs[name].identity_sha256,
                "output_root": str(configs[name].output.root),
                "output_exists": os.path.lexists(configs[name].output.root),
            }
            for name in configs
        },
    }


def validate_prerequisite(
    *, result_path: Path, complete_marker: Path, failed_marker: Path
) -> dict[str, object]:
    if (
        complete_marker.is_symlink()
        or not complete_marker.is_file()
        or failed_marker.exists()
        or failed_marker.is_symlink()
    ):
        _fail("PRL26 A/B evaluation completion boundary differs")
    payload = _read_json(result_path, "PRL26 A/B result table")
    if not isinstance(payload, dict):
        _fail("PRL26 A/B result table is not an object")
    arms = payload.get("arms")
    if (
        payload.get("schema_version") != _RESULT_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("contract") != "fresh-S0 Train@512 S32; matched Eval@512"
        or not isinstance(arms, dict)
        or set(arms) != {"no_tool", "crop"}
    ):
        _fail("PRL26 A/B result table contract differs")
    accepted: dict[str, object] = {}
    for name in ("no_tool", "crop"):
        arm = arms[name]
        if not isinstance(arm, dict):
            _fail(f"PRL26 A/B {name} result is malformed")
        macro = arm.get("macro_star_percent")
        subsets = arm.get("seven_subset_statistics")
        summary_path = Path(str(arm.get("summary_path", "")))
        if (
            arm.get("train_image_max_pixels") != 512 * 512
            or arm.get("evaluation_image_max_pixels") != 512 * 512
            or arm.get("optimizer_step") != 32
            or not isinstance(macro, (int, float))
            or not math.isfinite(float(macro))
            or not isinstance(subsets, list)
            or len(subsets) != 7
            or summary_path.is_symlink()
            or not summary_path.is_file()
        ):
            _fail(f"PRL26 A/B {name} result is incomplete")
        accepted[name] = {
            "macro_star_percent": float(macro),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
        }
    return {
        "schema_version": "tgvf.prl26-tgvf-prompt-prerequisite.v1",
        "status": "accepted",
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "arms": accepted,
    }


def validate_canary_completion(
    *, config_path: Path, repository: Path, expected_head: str
) -> dict[str, object]:
    if len(expected_head) != 40:
        _fail("expected canary repository HEAD is malformed")
    config = load_policy_e2e_smoke_run_config(config_path.resolve())
    root = config.output.root / "canary"
    tracker = root / "checkpoints/latest_checkpointed_iteration.txt"
    if _canonical_tracker_step(tracker, "TGVF canary") != 1:
        _fail("TGVF canary tracker is not exactly S1")
    rows = _read_json_lines(root / "metrics.jsonl", "TGVF canary metrics")
    _validate_metrics_rows(rows, expected_step=1, owner="TGVF canary metrics")
    state, pair, files = _validate_generation(
        config, root / "checkpoints/global_step_1", optimizer_step=1
    )
    if state.progress.optimizer_step != 1 or pair.optimizer_step != 1:
        _fail("TGVF canary checkpoint owners differ")
    if len(files) != 14:
        _fail("TGVF canary checkpoint component count differs")
    provenance_rows = _read_json_lines(
        root / "launch-provenance.jsonl", "TGVF canary launch provenance"
    )
    if len(provenance_rows) != 1:
        _fail("TGVF canary must have exactly one launch provenance record")
    provenance = provenance_rows[0]
    project = provenance.get("project")
    if (
        provenance.get("schema_version") != _PROVENANCE_SCHEMA
        or provenance.get("mode") != "canary"
        or provenance.get("target_step") != 1
        or provenance.get("run_id") != config.run_id
        or provenance.get("run_identity_sha256") != config.identity_sha256
        or provenance.get("run_config_file_sha256") != config.source_sha256
        or not isinstance(project, dict)
        or Path(str(project.get("root", ""))).resolve() != repository.resolve()
        or project.get("commit") != expected_head
        or project.get("clean") is not True
        or project.get("changes") != []
    ):
        _fail("TGVF canary launch provenance differs")
    return {
        "schema_version": "tgvf.prl26-tgvf-prompt-canary-completion.v1",
        "status": "accepted",
        "run_id": config.run_id,
        "run_identity_sha256": config.identity_sha256,
        "optimizer_step": 1,
        "metrics_sha256": _sha256(root / "metrics.jsonl"),
        "project_state_sha256": state.integrity_sha256,
        "pair_integrity_sha256": pair.integrity_sha256,
        "repository_head": expected_head,
    }


def _write(value: Mapping[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contracts = subparsers.add_parser("contracts")
    contracts.add_argument("--repository", required=True, type=Path)
    contracts.add_argument("--short-canary", required=True, type=Path)
    contracts.add_argument("--full-canary", required=True, type=Path)
    contracts.add_argument("--short-formal", required=True, type=Path)
    contracts.add_argument("--full-formal", required=True, type=Path)
    prerequisite = subparsers.add_parser("prerequisite")
    prerequisite.add_argument("--result", required=True, type=Path)
    prerequisite.add_argument("--complete-marker", required=True, type=Path)
    prerequisite.add_argument("--failed-marker", required=True, type=Path)
    canary = subparsers.add_parser("canary-complete")
    canary.add_argument("--config", required=True, type=Path)
    canary.add_argument("--repository", required=True, type=Path)
    canary.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "contracts":
            value = validate_contracts(
                repository=args.repository,
                short_canary_path=args.short_canary,
                full_canary_path=args.full_canary,
                short_formal_path=args.short_formal,
                full_formal_path=args.full_formal,
            )
        elif args.command == "prerequisite":
            value = validate_prerequisite(
                result_path=args.result,
                complete_marker=args.complete_marker,
                failed_marker=args.failed_marker,
            )
        else:
            value = validate_canary_completion(
                config_path=args.config,
                repository=args.repository,
                expected_head=args.expected_head,
            )
    except RuntimeError as error:
        print(f"PRL-26 C/D handoff rejected: {error}", file=sys.stderr)
        return 1
    _write(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "validate_canary_completion",
    "validate_contracts",
    "validate_prerequisite",
]
