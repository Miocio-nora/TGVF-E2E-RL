"""Audit PRL15 against the completed PRL14 Crop-16 control.

The existing Crop run is the source of truth. This module never derives or
launches a replacement Crop run from TGVF settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from tgvf_rl.policy.deepeyes_native_contract import DeepEyesNativeRunContract
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig

from .deepeyes_native_launcher import (
    DeepEyesNativeVerlLaunchPlan,
    build_deepeyes_native_verl_launch_plan,
)
from .prl13_main import (
    compose_pinned_deepeyes_config,
    preflight_pinned_deepeyes_config,
)
from .prl14_crop16_reference import (
    PRL14_CROP16_COMPARISON_STEP,
    apply_prl14_crop16_common_controls,
    load_prl14_crop16_completion,
)
from .trainable_tgvf_launcher import build_trainable_tgvf_verl_launch_plan


CROP_MATCHED_CONTROL_SCHEMA = "tgvf.prl15-crop16-reference-audit.v1"
CROP_MATCHED_CONTROL_RUN_ID = (
    "PRL-14-A-QWEN3-INSTRUCT-GRPO-BS16-N16-NATIVE-CROP-T1-"
    "CLEANFINAL-16STEP-WS8"
)
CROP_MATCHED_CONTROL_OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-"
    "cleanfinal-16step-ws8"
)
CROP_MATCHED_CONTROL_TARGET_STEP = PRL14_CROP16_COMPARISON_STEP
CROP_MATCHED_CONTROL_COMPARISON_SPEC = (
    Path(__file__).resolve().parents[4]
    / "configs/policy/controls/prl15_crop_rp66_matched.json"
)


@dataclass(frozen=True, slots=True)
class SemanticEqualField:
    """Two arm-specific config paths that represent one scientific value."""

    name: str
    control_path: str
    treatment_path: str

    def __post_init__(self) -> None:
        for field_name in ("name", "control_path", "treatment_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"semantic equality {field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ControlComparisonSpec:
    """Editable scientific comparison declaration, separate from runtime code."""

    required_equal: tuple[str, ...]
    semantic_equal: tuple[SemanticEqualField, ...]
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
        names = tuple(field.name for field in self.semantic_equal)
        if len(set(names)) != len(names):
            raise ValueError("semantic equality names contain duplicates")


def load_control_comparison_spec(path: str | Path) -> ControlComparisonSpec:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "required_equal",
        "semantic_equal",
        "arm_specific",
        "note",
    }:
        raise ValueError("control comparison spec fields differ")
    if payload["schema_version"] != "tgvf.control-comparison.v1":
        raise ValueError("control comparison spec schema differs")
    return ControlComparisonSpec(
        required_equal=tuple(payload["required_equal"]),
        semantic_equal=tuple(
            SemanticEqualField(
                name=row["name"],
                control_path=row["control_path"],
                treatment_path=row["treatment_path"],
            )
            for row in payload["semantic_equal"]
        ),
        arm_specific=tuple(payload["arm_specific"]),
        note=str(payload["note"]),
    )


@dataclass(frozen=True, slots=True)
class CropMatchedControlPlan:
    """Validated Crop control plus its explicit cross-arm equality proof."""

    launch: DeepEyesNativeVerlLaunchPlan
    matched_values: Mapping[str, object]
    semantic_matched_values: Mapping[str, object]
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
            self,
            "semantic_matched_values",
            MappingProxyType(dict(self.semantic_matched_values)),
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
                "world_size": 8,
                "actor_micro_batch_size_per_gpu": 32,
                "loss_objective": "deepeyes_official_micro_token_mean",
                "reason": (
                    "use the completed Crop-16 experiment as the external "
                    "control; do not synthesize Crop from TGVF"
                ),
            },
            "matched_values": dict(self.matched_values),
            "semantic_matched_values": dict(self.semantic_matched_values),
            "comparison": {
                "spec_path": str(self.comparison_spec_path),
                "note": self.comparison_note,
                "arm_differences": dict(self.arm_differences),
                "unclassified_differences": dict(self.unclassified_differences),
                "raw_unclassified_overrides_are_informational": True,
            },
            "launch": self.launch.as_record(),
        }


def build_crop_matched_control_plan(
    crop_contract: DeepEyesNativeRunContract,
    rp66_config: PolicyE2ESmokeRunConfig,
    *,
    comparison_spec_path: str | Path = CROP_MATCHED_CONTROL_COMPARISON_SPEC,
) -> CropMatchedControlPlan:
    """Reconstruct Crop-16 controls and compare TGVF against them."""

    comparison_path = Path(comparison_spec_path).resolve(strict=True)
    comparison = load_control_comparison_spec(comparison_path)
    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")
    base = build_deepeyes_native_verl_launch_plan(
        crop_contract, mode="formal", target_step=CROP_MATCHED_CONTROL_TARGET_STEP
    )
    completion = load_prl14_crop16_completion()
    values = dict(completion.overrides)
    apply_prl14_crop16_common_controls(values)
    # Compare the already-retained step-8 prefix. The completed run continued
    # to step 16, but the constant scheduler and optimizer horizon (-1) make
    # updates 1--8 independent of that later stopping condition.
    values["trainer.total_training_steps"] = CROP_MATCHED_CONTROL_TARGET_STEP
    missing = tuple(
        path
        for path in comparison.required_equal
        if path not in rp66.overrides or path not in values
    )
    if missing:
        raise ValueError(
            "comparison spec selects values absent from Crop-16 or RP66: "
            + ", ".join(missing)
        )
    semantic_matched: dict[str, object] = {}
    for field in comparison.semantic_equal:
        control_value = _select_override_path(values, field.control_path)
        treatment_value = _select_override_path(rp66.overrides, field.treatment_path)
        if control_value != treatment_value:
            raise ValueError(
                f"Crop/RP66 semantic control differs for {field.name}: "
                f"{control_value!r} != {treatment_value!r}"
            )
        semantic_matched[field.name] = control_value
    environment = dict(completion.environment)
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
        semantic_matched_values=semantic_matched,
        arm_differences=arm_differences,
        unclassified_differences=unclassified,
        comparison_spec_path=comparison_path,
        comparison_note=comparison.note,
    )


def _select_override_path(values: Mapping[str, object], path: str) -> object:
    """Resolve a flat Hydra override or a child inside one mapping override."""

    if path in values:
        return values[path]
    segments = path.split(".")
    for split_at in range(len(segments) - 1, 0, -1):
        prefix = ".".join(segments[:split_at])
        if prefix not in values:
            continue
        selected = values[prefix]
        for segment in segments[split_at:]:
            if not isinstance(selected, Mapping) or segment not in selected:
                raise ValueError(f"comparison path is absent: {path}")
            selected = selected[segment]
        return selected
    raise ValueError(f"comparison path is absent: {path}")


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
        help="Forbidden: PRL14 already exists and must not be duplicated.",
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
    raise RuntimeError(
        "PRL14 Crop-16 is the completed control; launching a synthesized Crop "
        "replacement is forbidden"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CROP_MATCHED_CONTROL_OUTPUT_ROOT",
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
