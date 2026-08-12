from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tgvf_rl.policy.run_config import (
    RP66AdapterUpdateMode,
    load_policy_e2e_smoke_run_config,
)


_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "configs/policy/runs"
_FROZEN = (
    _RUNS
    / "prl_17_r2_qwen3_instruct_full_frozen_rp67_bs16_n16_"
    "tfree_novisual_8step_ws8.toml"
)
_JOINT = (
    _RUNS
    / "prl_18_r0_qwen3_instruct_full_joint_rp67_bs16_n16_"
    "tfree_novisual_8step_ws8.toml"
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


def test_prl18_changes_only_adapter_optimization_and_management_identity() -> None:
    frozen = load_policy_e2e_smoke_run_config(
        _FROZEN.resolve(), allow_external_agent_loop_config=True
    )
    joint = load_policy_e2e_smoke_run_config(
        _JOINT.resolve(), allow_external_agent_loop_config=True
    )

    assert frozen.representation.adapter_update_mode is (
        RP66AdapterUpdateMode.FROZEN_ADAPTER
    )
    assert frozen.representation.adapter_trainable is False
    assert joint.representation.adapter_update_mode is RP66AdapterUpdateMode.JOINT
    assert joint.representation.adapter_trainable is True

    # Paths, provenance, output identity and retained-copy policy are
    # operational differences.  Every model/data/prompt/reward/optimizer/
    # sampling/distributed leaf is byte-equivalent to the frozen control.
    assert _different_leaf_paths(frozen.as_record(), joint.as_record()) == {
        "code.commit",
        "framework.agent_loop_config_path",
        "output.checkpoint_directory",
        "output.metrics_path",
        "output.root",
        "representation.adapter_update_mode",
        "reward.judge_config_path",
        "reward.judge_reason",
        "run_id",
        "training.permanent_checkpoint_steps",
    }

