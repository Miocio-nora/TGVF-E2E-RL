from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tgvf_rl.framework.verl.crop_matched_control_launcher import (
    CROP_MATCHED_CONTROL_COMPARISON_SPEC,
    CROP_MATCHED_CONTROL_OUTPUT_ROOT,
    CropMatchedControlPlan,
    build_crop_matched_control_plan,
    load_control_comparison_spec,
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
    _ROOT
    / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.toml"
)
_RP66 = (
    _ROOT
    / "configs/policy/runs/"
    "prl_15_r0_qwen3_instruct_full_rp66_bs16_n16_t1_matched_8step_gpu0123.toml"
)


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
    assert control.launch.overrides[
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
    ] == 1
    assert control.launch.overrides["trainer.n_gpus_per_node"] == 4
    assert control.launch.overrides["data.train_batch_size"] == 16
    assert control.launch.overrides["actor_rollout_ref.rollout.n"] == 16
    assert control.comparison_note == comparison.note
    assert control.unclassified_differences


def test_control_keeps_only_the_expected_crop_arm_differences() -> None:
    control, rp66 = _plans()
    values = control.launch.overrides

    assert values["trainer.default_local_dir"] == str(
        CROP_MATCHED_CONTROL_OUTPUT_ROOT / "checkpoints"
    )
    assert values["actor_rollout_ref.rollout.agent.default_agent_loop"] != (
        rp66.overrides["actor_rollout_ref.rollout.agent.default_agent_loop"]
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
            arm_differences=control.arm_differences,
            unclassified_differences=control.unclassified_differences,
            comparison_spec_path=control.comparison_spec_path,
            comparison_note=control.comparison_note,
        )


def test_common_values_follow_the_active_rp66_plan_without_python_edits(
    tmp_path: Path,
) -> None:
    original = load_control_comparison_spec(CROP_MATCHED_CONTROL_COMPARISON_SPEC)
    payload = {
        "schema_version": "tgvf.control-comparison.v1",
        "note": "test live declaration",
        "required_equal": ["actor_rollout_ref.rollout.temperature"],
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
    assert control.comparison_note == "test live declaration"
