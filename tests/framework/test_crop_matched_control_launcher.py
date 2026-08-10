from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tomllib

import pytest
from omegaconf import OmegaConf

from tgvf_rl.framework.verl.crop_matched_control_launcher import (
    CROP_MATCHED_CONTROL_COMPARISON_SCHEMA,
    CROP_MATCHED_CONTROL_COMPARISON_SPEC,
    CROP_MATCHED_CONTROL_OUTPUT_ROOT,
    MATHEMATICALLY_EQUIVALENT_EXECUTION_EFFECT,
    RESIDENCY_SAFETY_ONLY_SCOPE,
    CropMatchedControlPlan,
    build_crop_matched_control_plan,
    load_control_comparison_spec,
)
from tgvf_rl.framework.verl.deepeyes_native_launcher import _hydra_literal
from tgvf_rl.framework.verl.prl13_main import compose_pinned_deepeyes_config
from tgvf_rl.framework.verl.prl14_crop16_reference import (
    PRL14_CROP16_COMMON_OVERRIDES,
    PRL14_CROP16_REMOVED_OVERRIDES,
    load_prl14_crop16_completion,
)
from tgvf_rl.framework.verl.trainable_tgvf_launcher import (
    build_trainable_tgvf_verl_launch_plan,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).parents[2]
_CROP = (
    _ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.toml"
)
_RP66 = (
    _ROOT / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_matched_8step_ws8.toml"
)
_RP66_EXACT = (
    _ROOT / "configs/policy/runs/"
    "prl_16_f1_qwen3_instruct_full_frozen_rp66_bs16_n16_t1_"
    "crop16_exact_matched_8step_ws8.toml"
)
_EXACT_COMPARISON_SPEC = (
    _ROOT / "configs/policy/controls/"
    "prl16_frozen_rp66_crop_exact_optimizer_residency.json"
)
_ACTOR_OPTIMIZER_OFFLOAD_PATH = "actor_rollout_ref.actor.fsdp_config.optimizer_offload"


def _plans():
    rp66_config = load_policy_e2e_smoke_run_config(_RP66)
    control = build_crop_matched_control_plan(
        load_deepeyes_native_run_contract(_CROP), rp66_config
    )
    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")
    return control, rp66


def test_control_matches_every_declared_common_runtime_value() -> None:
    control, rp66 = _plans()
    comparison = load_control_comparison_spec(CROP_MATCHED_CONTROL_COMPARISON_SPEC)

    assert set(control.matched_values) == set(comparison.required_equal)
    for path in comparison.required_equal:
        assert control.launch.overrides[path] == rp66.overrides[path]
    assert (
        control.launch.overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"]
        == 32
    )
    assert control.launch.overrides["trainer.n_gpus_per_node"] == 8
    assert control.launch.overrides["data.train_batch_size"] == 16
    assert control.launch.overrides["actor_rollout_ref.rollout.n"] == 16
    assert control.comparison_note == comparison.note
    assert control.semantic_matched_values == {
        "judge_config_sha256": rp66.config.reward.judge_config_sha256
    }
    optimizer_offload_path = "actor_rollout_ref.actor.fsdp_config.optimizer_offload"
    assert optimizer_offload_path not in comparison.required_equal
    assert len(control.allowed_execution_deviations) == 1
    deviation = control.allowed_execution_deviations[0]
    assert deviation.name == "actor_optimizer_state_cpu_residency"
    assert deviation.path == optimizer_offload_path
    assert deviation.control_value is False
    assert deviation.treatment_value is True
    assert deviation.effect == MATHEMATICALLY_EQUIVALENT_EXECUTION_EFFECT
    assert deviation.scope == RESIDENCY_SAFETY_ONLY_SCOPE
    assert control.launch.overrides[optimizer_offload_path] is False
    assert rp66.overrides[optimizer_offload_path] is True
    assert optimizer_offload_path not in control.unclassified_differences
    recorded = control.as_record()["comparison"]["allowed_execution_deviations"][0]
    assert recorded["path"] == optimizer_offload_path
    assert recorded["verified"] is True
    assert control.unclassified_differences


def test_exact_control_matches_crop_actor_optimizer_residency(tmp_path: Path) -> None:
    legacy = load_control_comparison_spec(CROP_MATCHED_CONTROL_COMPARISON_SPEC)
    exact = load_control_comparison_spec(_EXACT_COMPARISON_SPEC)

    assert set(exact.required_equal) == {
        *legacy.required_equal,
        _ACTOR_OPTIMIZER_OFFLOAD_PATH,
    }
    assert len(exact.required_equal) == len(legacy.required_equal) + 1
    assert exact.semantic_equal == legacy.semantic_equal
    assert exact.arm_specific == legacy.arm_specific
    assert exact.allowed_execution_deviations == ()

    text = _RP66_EXACT.read_text(encoding="utf-8")
    payload = tomllib.loads(text)
    dependency_paths = (
        Path(payload["reward"]["judge_config_path"]),
        Path(payload["framework"]["agent_loop_config_path"]),
    )
    dependency_roots = {path.parents[3] for path in dependency_paths}
    assert len(dependency_roots) == 1
    portable = text.replace(str(dependency_roots.pop()), str(_ROOT))
    portable_path = tmp_path / _RP66_EXACT.name
    portable_path.write_text(portable, encoding="utf-8")
    rp66_config = load_policy_e2e_smoke_run_config(portable_path)
    control = build_crop_matched_control_plan(
        load_deepeyes_native_run_contract(_CROP),
        rp66_config,
        comparison_spec_path=_EXACT_COMPARISON_SPEC,
    )
    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")

    assert control.launch.overrides[_ACTOR_OPTIMIZER_OFFLOAD_PATH] is False
    assert rp66.overrides[_ACTOR_OPTIMIZER_OFFLOAD_PATH] is False
    assert control.matched_values[_ACTOR_OPTIMIZER_OFFLOAD_PATH] is False
    assert control.allowed_execution_deviations == ()
    assert _ACTOR_OPTIMIZER_OFFLOAD_PATH not in control.unclassified_differences


def test_crop16_common_controls_match_the_hash_verified_completed_run() -> None:
    completion = load_prl14_crop16_completion()
    composed = compose_pinned_deepeyes_config(
        tuple(
            f"++{path}={_hydra_literal(value)}"
            for path, value in completion.overrides.items()
        )
    )

    for path, expected in PRL14_CROP16_COMMON_OVERRIDES.items():
        actual = OmegaConf.select(composed, path, default=object())
        if OmegaConf.is_config(actual):
            actual = OmegaConf.to_container(actual, resolve=True)
        assert actual == expected, path
    for path in PRL14_CROP16_REMOVED_OVERRIDES:
        assert OmegaConf.select(composed, path, default=None) is None, path


def test_control_keeps_only_the_expected_crop_arm_differences() -> None:
    control, rp66 = _plans()
    values = control.launch.overrides

    assert values["trainer.default_local_dir"] == str(
        CROP_MATCHED_CONTROL_OUTPUT_ROOT / "checkpoints"
    )
    assert (
        values["actor_rollout_ref.rollout.agent.default_agent_loop"]
        != (rp66.overrides["actor_rollout_ref.rollout.agent.default_agent_loop"])
    )
    assert "actor_rollout_ref.rollout.checkpoint_manager_class" not in values
    assert values["reward.deepeyes_official.judge_service_config_sha256"] == (
        rp66.config.reward.judge_config_sha256
    )


def test_control_rejects_a_non_step8_launch() -> None:
    control, _ = _plans()
    with pytest.raises(ValueError, match="formal step-8"):
        CropMatchedControlPlan(
            launch=replace(
                control.launch,
                target_step=20,
                overrides={
                    **control.launch.overrides,
                    "trainer.total_training_steps": 20,
                },
            ),
            matched_values=control.matched_values,
            semantic_matched_values=control.semantic_matched_values,
            allowed_execution_deviations=control.allowed_execution_deviations,
            arm_differences=control.arm_differences,
            unclassified_differences=control.unclassified_differences,
            comparison_spec_path=control.comparison_spec_path,
            comparison_note=control.comparison_note,
        )


def test_common_values_are_checked_against_crop16_not_copied_from_rp66(
    tmp_path: Path,
) -> None:
    original = load_control_comparison_spec(CROP_MATCHED_CONTROL_COMPARISON_SPEC)
    payload = {
        "schema_version": CROP_MATCHED_CONTROL_COMPARISON_SCHEMA,
        "note": "test Crop-16 reference declaration",
        "required_equal": ["actor_rollout_ref.rollout.temperature"],
        "semantic_equal": [],
        "allowed_execution_deviations": [],
        "arm_specific": ["trainer.experiment_name"],
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rp66_config = load_policy_e2e_smoke_run_config(_RP66)
    control = build_crop_matched_control_plan(
        load_deepeyes_native_run_contract(_CROP),
        rp66_config,
        comparison_spec_path=path,
    )
    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")

    assert original.required_equal
    assert control.matched_values == {
        "actor_rollout_ref.rollout.temperature": rp66.overrides[
            "actor_rollout_ref.rollout.temperature"
        ]
    }
    assert control.comparison_note == "test Crop-16 reference declaration"


def test_allowed_execution_deviation_declared_values_are_enforced(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        CROP_MATCHED_CONTROL_COMPARISON_SPEC.read_text(encoding="utf-8")
    )
    payload["allowed_execution_deviations"][0]["treatment_value"] = "enabled"
    path = tmp_path / "wrong-execution-deviation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="allowed execution deviation differs.*actor_optimizer_state_cpu_residency",
    ):
        build_crop_matched_control_plan(
            load_deepeyes_native_run_contract(_CROP),
            load_policy_e2e_smoke_run_config(_RP66),
            comparison_spec_path=path,
        )


def test_execution_deviation_cannot_overlap_a_strict_equal_field(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        CROP_MATCHED_CONTROL_COMPARISON_SPEC.read_text(encoding="utf-8")
    )
    payload["required_equal"].append(
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload"
    )
    path = tmp_path / "overlapping-execution-deviation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="control comparison paths overlap"):
        load_control_comparison_spec(path)
