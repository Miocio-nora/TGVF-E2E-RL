from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "configs/policy/runs"
JOINT_SUPERVISOR = ROOT / "tools/supervise_prl24_b_fmt2_joint_bs64_8step.sh"
JOINT_EVAL_PLAN = (
    ROOT
    / "configs/evaluation/prl24_b_fmt2_joint_rp67_tfree_teacher25_bs64_step8_paired_seed_coredev2511_plan.json"
)
FROZEN_EVAL_PLAN = (
    ROOT
    / "configs/evaluation/prl24_a_fmt2_frozen_rp67_tfree_teacher25_bs64_step4_step8_paired_seed_coredev2511_plan.json"
)


def _load(name: str) -> dict[str, object]:
    return tomllib.loads((RUNS / name).read_text(encoding="utf-8"))


def _remove_expected_joint_identity_changes(value: dict[str, object]) -> dict[str, object]:
    result = deepcopy(value)
    result.pop("run_id")
    representation = result["representation"]
    reward = result["reward"]
    assert isinstance(representation, dict)
    assert isinstance(reward, dict)
    representation.pop("adapter_update_mode")
    reward.pop("judge_reason")
    result.pop("output")
    return result


def test_prl24_b_fmt2_joint_changes_only_adapter_and_identity_from_frozen() -> None:
    frozen = _load(
        "prl_24_a_fmt2_qwen3_instruct_full_frozen_rp67_bs64_n16_tfree_teacher25_8step_ws8.toml"
    )
    joint = _load(
        "prl_24_b_fmt2_joint_qwen3_instruct_full_rp67_bs64_n16_tfree_teacher25_8step_ws8.toml"
    )

    assert frozen["representation"]["adapter_update_mode"] == "frozen_adapter"
    assert joint["representation"]["adapter_update_mode"] == "joint"
    assert frozen["reward"]["protocol_error_penalty"] == 2.0
    assert joint["reward"]["protocol_error_penalty"] == 2.0
    assert _remove_expected_joint_identity_changes(frozen) == (
        _remove_expected_joint_identity_changes(joint)
    )


def test_prl24_b_joint_canary_is_small_and_uses_fmt2() -> None:
    canary = _load(
        "prl_24_b_fmt2_joint_c0_qwen3_instruct_full_rp67_bs4_n2_tfree_teacher25_1step_ws4.toml"
    )

    assert canary["representation"]["adapter_update_mode"] == "joint"
    assert canary["reward"]["protocol_error_penalty"] == 2.0
    assert canary["accumulation"]["global_prompt_batch_size"] == 4
    assert canary["sampling"]["trajectories_per_prompt"] == 2
    assert canary["distributed"]["world_size"] == 4
    assert canary["training"]["logger"] == ["console"]


def test_prl24_b_joint_supervisor_checks_canary_mode_artifacts() -> None:
    script = JOINT_SUPERVISOR.read_text(encoding="utf-8")

    assert 'canary_mode_root="$canary_root/canary"' in script
    assert (
        'local pointer="$canary_mode_root/runtime-policy-state/'
        'latest-lora-snapshot.json"' in script
    )
    assert (
        'local tracker="$canary_mode_root/checkpoints/'
        'latest_checkpointed_iteration.txt"' in script
    )


def test_prl24_b_joint_step8_eval_is_paired_with_frozen_step8() -> None:
    joint = json.loads(JOINT_EVAL_PLAN.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_EVAL_PLAN.read_text(encoding="utf-8"))
    policy_config = ROOT / joint["policy_config"]

    assert hashlib.sha256(policy_config.read_bytes()).hexdigest() == joint[
        "policy_config_sha256"
    ]
    assert joint["arms"] == [
        {
            "name": "step8",
            "optimizer_step": 8,
            "qwen_source": "output.root/permanent-checkpoints/global_step_8",
            "rp66_source": (
                "output.root/runtime-policy-state/lora-manifests/"
                "step-00000008-*.json"
            ),
        }
    ]
    assert joint["required_pairing"]["adapter_update_mode"] == "joint"
    assert joint["required_pairing"]["rp66_state_must_remain_constant"] is False
    assert joint["protocol"] == frozen["protocol"]
    assert joint["paired_rng"] == frozen["paired_rng"]
    assert joint["scoring"]["datasets"] == frozen["scoring"]["datasets"]
    assert joint["scoring"]["judge_config_sha256"] == frozen["scoring"][
        "judge_config_sha256"
    ]
