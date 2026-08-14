from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_CONTROL = (
    _RUNS
    / "prl_17_r2_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_novisual_8step_ws8.toml"
)
_TREATMENT = (
    _RUNS
    / "prl_19_r0_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_visual_api_8step_ws8.toml"
)


def _different_leaf_paths(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> set[str]:
    paths: set[str] = set()
    for key in left.keys() | right.keys():
        left_value = left.get(key)
        right_value = right.get(key)
        path = (*prefix, key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            paths.update(_different_leaf_paths(left_value, right_value, path))
        elif left_value != right_value:
            paths.add(".".join(path))
    return paths


def test_prl19_changes_only_visual_reward_and_operational_identity() -> None:
    control = load_policy_e2e_smoke_run_config(
        _CONTROL.resolve(), allow_external_agent_loop_config=True
    )
    treatment = load_policy_e2e_smoke_run_config(_TREATMENT.resolve())

    assert control.representation.adapter_trainable is False
    assert treatment.representation.adapter_trainable is False
    assert control.reward.tool_utility_reward_enabled is False
    assert treatment.reward.tool_utility_reward_enabled is False
    assert control.reward.focus_reward_enabled is False
    assert control.reward.grounding_reward_enabled is False
    assert treatment.reward.focus_reward_enabled is True
    assert treatment.reward.grounding_reward_enabled is True
    assert treatment.reward.visual_quality_judge_identity is not None

    assert _different_leaf_paths(control.as_record(), treatment.as_record()) == {
        "code.commit",
        "framework.agent_loop_config_path",
        "output.checkpoint_directory",
        "output.metrics_path",
        "output.root",
        "reward.focus_reward_enabled",
        "reward.grounding_reward_enabled",
        "reward.judge_config_path",
        "reward.judge_reason",
        "reward.visual_quality_judge_config_path",
        "reward.visual_quality_judge_config_sha256",
        "reward.visual_quality_judge_mode",
        "run_id",
        "training.permanent_checkpoint_steps",
    }
