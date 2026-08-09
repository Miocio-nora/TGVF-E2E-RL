"""World-4/micro-1 Crop control matched to the PRL15 RP66 pilot.

The released DeepEyes actor objective is an equal average of token means over
fixed local micro-batches. Its scalar therefore depends on the actor micro
shape whenever trajectory lengths differ. This launcher keeps Crop's native
tool/prompt path while matching PRL15's optimization and serving shape so the
tool comparison does not silently include a micro-batch objective change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from tgvf_rl.policy.deepeyes_native_contract import DeepEyesNativeRunContract
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig

from .deepeyes_native_launcher import (
    DeepEyesNativeVerlLaunchPlan,
    apply_launch_environment,
    build_deepeyes_native_verl_launch_plan,
)
from .prl13_main import (
    compose_pinned_deepeyes_config,
    preflight_pinned_deepeyes_config,
    run_pinned_deepeyes_config,
)
from .trainable_tgvf_launcher import build_trainable_tgvf_verl_launch_plan


CROP_MATCHED_CONTROL_SCHEMA = "tgvf.prl15-crop-ws4-micro1-control.v1"
CROP_MATCHED_CONTROL_RUN_ID = (
    "PRL-15-C0-QWEN3-INSTRUCT-FULL-CROP-BS16-N16-WS4-MICRO1-8STEP"
)
CROP_MATCHED_CONTROL_OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-15-C0-qwen3-instruct-full-crop-bs16-n16-ws4-micro1-8step"
)
CROP_MATCHED_CONTROL_TARGET_STEP = 8
CROP_MATCHED_CONTROL_JUDGE_CONFIG = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/policy/judges/"
    "prl13_qwen25_72b_binary_text_resilient.json"
)
CROP_MATCHED_CONTROL_COMPARISON_SPEC = (
    Path(__file__).resolve().parents[4]
    / "configs/policy/controls/prl15_crop_rp66_matched.json"
)


@dataclass(frozen=True, slots=True)
class ControlComparisonSpec:
    """Editable scientific comparison declaration, separate from runtime code."""

    required_equal: tuple[str, ...]
    arm_specific: tuple[str, ...]
    note: str

    def __post_init__(self) -> None:
        for owner, rows in (
            ("required_equal", self.required_equal),
            ("arm_specific", self.arm_specific),
        ):
            if not rows or any(not isinstance(row, str) or not row for row in rows):
                raise ValueError(f"control comparison {owner} must be non-empty paths")
            if len(set(rows)) != len(rows):
                raise ValueError(f"control comparison {owner} contains duplicates")
        overlap = set(self.required_equal) & set(self.arm_specific)
        if overlap:
            raise ValueError(f"control comparison paths overlap: {sorted(overlap)!r}")
        if not self.note:
            raise ValueError("control comparison note must be non-empty")


def load_control_comparison_spec(path: str | Path) -> ControlComparisonSpec:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "required_equal",
        "arm_specific",
        "note",
    }:
        raise ValueError("control comparison spec fields differ")
    if payload["schema_version"] != "tgvf.control-comparison.v1":
        raise ValueError("control comparison spec schema differs")
    return ControlComparisonSpec(
        required_equal=tuple(payload["required_equal"]),
        arm_specific=tuple(payload["arm_specific"]),
        note=str(payload["note"]),
    )


@dataclass(frozen=True, slots=True)
class CropMatchedControlPlan:
    """Validated Crop control plus its explicit cross-arm equality proof."""

    launch: DeepEyesNativeVerlLaunchPlan
    matched_values: Mapping[str, object]
    arm_differences: Mapping[str, tuple[object, object]]
    unclassified_differences: Mapping[str, tuple[object, object]]
    comparison_spec_path: Path
    comparison_note: str
    schema_version: str = CROP_MATCHED_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CROP_MATCHED_CONTROL_SCHEMA:
            raise ValueError("Crop matched-control schema differs")
        if self.launch.mode != "formal" or self.launch.target_step != 8:
            raise ValueError("Crop matched control must be a formal step-8 plan")
        object.__setattr__(
            self, "matched_values", MappingProxyType(dict(self.matched_values))
        )
        object.__setattr__(
            self, "arm_differences", MappingProxyType(dict(self.arm_differences))
        )
        object.__setattr__(
            self,
            "unclassified_differences",
            MappingProxyType(dict(self.unclassified_differences)),
        )
        object.__setattr__(
            self, "comparison_spec_path", Path(self.comparison_spec_path).resolve()
        )

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": CROP_MATCHED_CONTROL_RUN_ID,
            "output_root": str(CROP_MATCHED_CONTROL_OUTPUT_ROOT),
            "target_step": CROP_MATCHED_CONTROL_TARGET_STEP,
            "scientific_control": {
                "crop_prompt_and_tool_are_arm_specific": True,
                "rp66_adapter_is_absent": True,
                "world_size": 4,
                "actor_micro_batch_size_per_gpu": 1,
                "loss_objective": "deepeyes_official_micro_token_mean",
                "reason": (
                    "remove actor micro-batch reduction as a Crop-vs-RP66 "
                    "confound"
                ),
            },
            "matched_values": dict(self.matched_values),
            "comparison": {
                "spec_path": str(self.comparison_spec_path),
                "note": self.comparison_note,
                "arm_differences": dict(self.arm_differences),
                "unclassified_differences": dict(self.unclassified_differences),
                "unclassified_differences_are_fatal": False,
            },
            "launch": self.launch.as_record(),
        }


def build_crop_matched_control_plan(
    crop_contract: DeepEyesNativeRunContract,
    rp66_config: PolicyE2ESmokeRunConfig,
    *,
    comparison_spec_path: str | Path = CROP_MATCHED_CONTROL_COMPARISON_SPEC,
) -> CropMatchedControlPlan:
    """Build Crop from an editable match declaration and record every difference."""

    comparison_path = Path(comparison_spec_path).resolve(strict=True)
    comparison = load_control_comparison_spec(comparison_path)
    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")
    base = build_deepeyes_native_verl_launch_plan(
        crop_contract, mode="formal", target_step=CROP_MATCHED_CONTROL_TARGET_STEP
    )
    values = dict(base.overrides)
    values.update(
        {
            "trainer.experiment_name": CROP_MATCHED_CONTROL_RUN_ID,
            "trainer.default_local_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "checkpoints"
            ),
            "trainer.rollout_data_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "trajectories"
            ),
            "trainer.validation_data_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "validation"
            ),
            "trainer.total_training_steps": CROP_MATCHED_CONTROL_TARGET_STEP,
            "trainer.save_freq": 1,
            "trainer.test_freq": 0,
            "trainer.val_before_train": False,
            "trainer.resume_mode": "disable",
            "trainer.max_actor_ckpt_to_keep": 4,
            "reward.deepeyes_official.judge_service_config_path": (
                CROP_MATCHED_CONTROL_JUDGE_CONFIG
            ),
            "reward.deepeyes_official.judge_service_config_sha256": (
                rp66_config.reward.judge_config_sha256
            ),
        }
    )
    missing = tuple(
        path for path in comparison.required_equal if path not in rp66.overrides
    )
    if missing:
        raise ValueError(
            "comparison spec selects values absent from RP66: " + ", ".join(missing)
        )
    # The active RP66 plan is the source of truth. Changing a common variable
    # in a later run config automatically propagates into this Crop control;
    # no Python constant or checker rewrite is required.
    for path in comparison.required_equal:
        values[path] = rp66.overrides[path]
    environment = dict(base.environment)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    launch = replace(base, overrides=values, environment=environment)

    mismatches = {
        path: (launch.overrides.get(path), rp66.overrides.get(path))
        for path in comparison.required_equal
        if launch.overrides.get(path) != rp66.overrides.get(path)
    }
    if mismatches:
        raise ValueError(f"Crop/RP66 matched-control values differ: {mismatches!r}")
    matched = {path: launch.overrides[path] for path in comparison.required_equal}
    all_differences = {
        path: (launch.overrides.get(path), rp66.overrides.get(path))
        for path in sorted(set(launch.overrides) | set(rp66.overrides))
        if launch.overrides.get(path) != rp66.overrides.get(path)
    }
    arm_differences = {
        path: all_differences[path]
        for path in comparison.arm_specific
        if path in all_differences
    }
    classified = set(comparison.required_equal) | set(comparison.arm_specific)
    unclassified = {
        path: difference
        for path, difference in all_differences.items()
        if path not in classified
    }
    return CropMatchedControlPlan(
        launch=launch,
        matched_values=matched,
        arm_differences=arm_differences,
        unclassified_differences=unclassified,
        comparison_spec_path=comparison_path,
        comparison_note=comparison.note,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-contract", required=True, type=Path)
    parser.add_argument("--rp66-config", required=True, type=Path)
    parser.add_argument(
        "--comparison-spec",
        type=Path,
        default=CROP_MATCHED_CONTROL_COMPARISON_SPEC,
        help="Editable declaration of matched and arm-specific fields.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start GPU/API work; omit for compose-only preflight.",
    )
    args = parser.parse_args(argv)

    from tgvf_rl.policy.deepeyes_native_contract import (
        load_deepeyes_native_run_contract,
    )
    from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

    crop_contract = load_deepeyes_native_run_contract(args.crop_contract.resolve())
    rp66_config = load_policy_e2e_smoke_run_config(args.rp66_config.resolve())
    control = build_crop_matched_control_plan(
        crop_contract,
        rp66_config,
        comparison_spec_path=args.comparison_spec,
    )
    composed = compose_pinned_deepeyes_config(control.launch.hydra_override_args())
    preflight = preflight_pinned_deepeyes_config(composed)
    record = {**control.as_record(), "compose_preflight": preflight}
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    if not args.launch:
        return 0

    crop_contract.assert_launchable(Path(__file__).resolve().parents[4])
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    if CROP_MATCHED_CONTROL_OUTPUT_ROOT.exists():
        raise RuntimeError("Crop matched-control output root already exists")
    apply_launch_environment(control.launch)
    run_pinned_deepeyes_config(composed)
    control.launch.assert_target_checkpoint_complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CROP_MATCHED_CONTROL_OUTPUT_ROOT",
    "CROP_MATCHED_CONTROL_JUDGE_CONFIG",
    "CROP_MATCHED_CONTROL_RUN_ID",
    "CROP_MATCHED_CONTROL_SCHEMA",
    "CROP_MATCHED_CONTROL_TARGET_STEP",
    "CROP_MATCHED_CONTROL_COMPARISON_SPEC",
    "ControlComparisonSpec",
    "CropMatchedControlPlan",
    "build_crop_matched_control_plan",
    "load_control_comparison_spec",
    "main",
]
