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
from tgvf_rl.evaluation.coredev_results import (  # noqa: E402
    extract_coredev_macro_star,
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
_A_B_CONTROL_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/"
    "PRL26-TRAIN512-S32-PIXEL512-COREDEV2511-V1"
)
_EXPECTED_RESULT_PATH = _A_B_CONTROL_ROOT / "train512-s32-pixel512-results.json"
_EXPECTED_COMPLETE_MARKER = _A_B_CONTROL_ROOT / "runtime/evaluation-complete"
_EXPECTED_FAILED_MARKER = _A_B_CONTROL_ROOT / "runtime/failed"
_EXPECTED_HANDOFF_PATH = _A_B_CONTROL_ROOT / "runtime/bound-handoff.json"
_EXPECTED_COVERAGE = {
    "official_manifest_rows": 2511,
    "evaluated_single_image_rows": 2240,
    "held_multi_image_rows": 271,
    "subset_count": 7,
}
_EXPECTED_HANDOFF_COVERAGE = {
    "official_manifest_rows": 2511,
    "evaluated_single_image_rows": 2240,
    "held_multi_image_rows": 271,
    "datasets": 7,
}
_EXPECTED_EVALUATION_IDS = {
    "no_tool": "PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1",
    "crop": "PRL26-B-TRAIN512-S32-CROP-MATCHED-COREDEV2511-PIXEL512-BOUNDARYFIX-V1",
}
_EXPECTED_METHODS = {
    "no_tool": "NoTool Train@512 S32",
    "crop": "Crop Train@512 S32",
}
_EXPECTED_SUMMARY_PATHS = {
    "no_tool": Path(
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-26-A-train512-s32-parity-notool-qwen3-instruct-bs16-n16-"
        "teacher25-ws8/evaluation/"
        "PRL26-A-TRAIN512-S32-NOTOOL-MATCHED-COREDEV2511-S32-PIXEL512-V1/"
        "matched/step32/scoring/coredev-official-v1/"
        "coredev-2511-eval-summary.json"
    ),
    "crop": Path(
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
        "PRL-26-B-train512-s32-parity-crop-qwen3-instruct-bs16-n16-"
        "teacher25-ws8/evaluation/"
        "PRL26-B-TRAIN512-S32-CROP-MATCHED-COREDEV2511-PIXEL512-BOUNDARYFIX-V1/"
        "step32/scoring/coredev-official-v1/coredev-2511-eval-summary.json"
    ),
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_regular_file(path: Path, owner: str, *, nonempty: bool = True) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{owner} is not a regular file")
    if nonempty and path.stat().st_size <= 0:
        _fail(f"{owner} is empty")


def _require_sha256(value: object, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{owner} is not a lowercase SHA256")
    return value


def _validate_bound_handoff(path: Path) -> dict[str, object]:
    _require_regular_file(path, "PRL26 A/B bound handoff")
    payload = _read_json(path, "PRL26 A/B bound handoff")
    if not isinstance(payload, dict):
        _fail("PRL26 A/B bound handoff is not an object")
    identity = _require_sha256(
        payload.get("identity_sha256"), "PRL26 A/B handoff identity"
    )
    content = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if identity != _canonical_sha256(content):
        _fail("PRL26 A/B handoff canonical identity differs")
    if (
        payload.get("schema_version")
        != "tgvf.prl26-train512-s32-evaluation-handoff.v1"
        or payload.get("status") != "ready"
        or payload.get("train_image_max_pixels") != 512 * 512
        or payload.get("evaluation_image_max_pixels") != 512 * 512
        or payload.get("optimizer_step") != 32
        or payload.get("coverage") != _EXPECTED_HANDOFF_COVERAGE
    ):
        _fail("PRL26 A/B handoff contract differs")
    for name in ("no_tool", "crop"):
        arm = payload.get(name)
        if not isinstance(arm, dict):
            _fail(f"PRL26 A/B handoff {name} arm is malformed")
        completion_path = Path(str(arm.get("completion_path", "")))
        config_path = Path(str(arm.get("config_path", "")))
        _require_regular_file(completion_path, f"PRL26 A/B {name} completion")
        _require_regular_file(config_path, f"PRL26 A/B {name} config")
        if (
            arm.get("evaluation_id") != _EXPECTED_EVALUATION_IDS[name]
            or arm.get("completion_file_sha256") != _sha256(completion_path)
            or arm.get("config_file_sha256") != _sha256(config_path)
        ):
            _fail(f"PRL26 A/B handoff {name} provenance differs")
        _require_sha256(
            arm.get("run_identity_sha256"), f"PRL26 A/B {name} run identity"
        )
        _require_sha256(
            arm.get("checkpoint_pair_integrity_sha256"),
            f"PRL26 A/B {name} checkpoint pair",
        )
    crop = payload["crop"]
    assert isinstance(crop, dict)
    plan_path = Path(str(crop.get("bound_plan_path", "")))
    _require_regular_file(plan_path, "PRL26 A/B bound Crop plan")
    if (
        crop.get("bound_plan_file_sha256") != _sha256(plan_path)
        or crop.get("tool_action_boundary")
        != {"stop_strings": ["</tool_call>"], "include_stop_str_in_output": True}
    ):
        _fail("PRL26 A/B bound Crop plan or action boundary differs")
    return payload


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
        result_path.resolve() != _EXPECTED_RESULT_PATH.resolve()
        or complete_marker.resolve() != _EXPECTED_COMPLETE_MARKER.resolve()
        or failed_marker.resolve() != _EXPECTED_FAILED_MARKER.resolve()
    ):
        _fail("PRL26 A/B prerequisite canonical paths differ")
    _require_regular_file(result_path, "PRL26 A/B result table")
    _require_regular_file(
        complete_marker, "PRL26 A/B completion marker", nonempty=False
    )
    if os.path.lexists(failed_marker):
        _fail("PRL26 A/B evaluation completion boundary differs")
    handoff = _validate_bound_handoff(_EXPECTED_HANDOFF_PATH)
    payload = _read_json(result_path, "PRL26 A/B result table")
    if not isinstance(payload, dict):
        _fail("PRL26 A/B result table is not an object")
    arms = payload.get("arms")
    if (
        payload.get("schema_version") != _RESULT_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("contract") != "fresh-S0 Train@512 S32; matched Eval@512"
        or payload.get("coverage") != _EXPECTED_COVERAGE
        or payload.get("handoff_identity_sha256") != handoff["identity_sha256"]
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
            arm.get("method") != _EXPECTED_METHODS[name]
            or arm.get("train_image_max_pixels") != 512 * 512
            or arm.get("evaluation_image_max_pixels") != 512 * 512
            or arm.get("optimizer_step") != 32
            or not isinstance(macro, (int, float))
            or not math.isfinite(float(macro))
            or not isinstance(subsets, list)
            or len(subsets) != 7
            or summary_path.resolve() != _EXPECTED_SUMMARY_PATHS[name].resolve()
        ):
            _fail(f"PRL26 A/B {name} result is incomplete")
        _require_regular_file(summary_path, f"PRL26 A/B {name} summary")
        summary = _read_json(summary_path, f"PRL26 A/B {name} summary")
        if not isinstance(summary, dict):
            _fail(f"PRL26 A/B {name} summary is not an object")
        if (
            summary.get("schema_version") != 1
            or summary.get("status") != "pass"
            or summary.get("phase") != "eval"
            or summary.get("model") != "Qwen3-VL-8B-Instruct"
            or summary.get("sample_count") != 2511
            or summary.get("slice_count") != 7
            or not isinstance(summary.get("slices"), list)
            or len(summary["slices"]) != 7
            or subsets != summary["slices"]
        ):
            _fail(f"PRL26 A/B {name} summary contract differs")
        try:
            headline = extract_coredev_macro_star(summary)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"PRL26 A/B {name} frozen CoreDev extraction failed: {error}"
            ) from error
        if arm.get("headline") != headline or not math.isclose(
            float(macro),
            float(headline["macro_star_percent"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            _fail(f"PRL26 A/B {name} published headline differs")
        accepted[name] = {
            "macro_star_percent": float(macro),
            "headline": headline,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
        }
    return {
        "schema_version": "tgvf.prl26-tgvf-prompt-prerequisite.v1",
        "status": "accepted",
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "handoff_path": str(_EXPECTED_HANDOFF_PATH.resolve()),
        "handoff_sha256": _sha256(_EXPECTED_HANDOFF_PATH),
        "handoff_identity_sha256": handoff["identity_sha256"],
        "coverage": _EXPECTED_COVERAGE,
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
