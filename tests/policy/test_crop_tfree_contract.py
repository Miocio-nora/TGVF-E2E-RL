from __future__ import annotations

from pathlib import Path
import re

import pytest

from tgvf_rl.policy.crop_tfree_contract import (
    CROP_TFREE_CODE_PLACEHOLDER,
    CROP_TFREE_RUN_SCHEMA,
    load_crop_tfree_run_contract,
)


_ROOT = Path(__file__).parents[2]
_CONFIG = (
    _ROOT / "configs/policy/runs/"
    "prl_21_r0_qwen3_instruct_full_crop_bs16_n16_tfree_16step_ws8.toml"
)


def test_prl21_overlay_records_only_the_matched_tfree_treatment() -> None:
    contract = load_crop_tfree_run_contract(
        _CONFIG,
        repository_root=_ROOT,
    )

    assert contract.payload["schema_version"] == CROP_TFREE_RUN_SCHEMA
    assert re.fullmatch(r"[0-9a-f]{40}", contract.code_commit)
    assert contract.reward_manager_class.endswith("DeepEyesCropTFreeRewardManager")
    reward = contract.payload["reward"]
    assert reward == {
        "profile": "stage3-shaped-v1-tfree",
        "manager_class": (
            "tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward."
            "DeepEyesCropTFreeRewardManager"
        ),
        "answer_weight": 2.0,
        "repeated_call_penalty": 0.05,
        "protocol_error_penalty": 1.0,
        "tool_utility_reward_enabled": False,
        "focus_reward_enabled": False,
        "grounding_reward_enabled": False,
        "positive_crop_bonus_enabled": False,
        "answer_verifier": "deepeyes_extraction_qwen25_72b_text",
    }
    assert contract.reward_manager_module_path == (
        "pkg://tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward"
    )


def test_prl21_overlay_matches_the_intended_formal_shape() -> None:
    contract = load_crop_tfree_run_contract(_CONFIG, repository_root=_ROOT)
    matched = contract.payload["matched_training"]

    assert matched["training_mode"] == "full"
    assert matched["vision_trainable"] is True
    assert matched["projector_trainable"] is True
    assert matched["language_trainable"] is True
    assert matched["global_prompt_batch_size"] == 16
    assert matched["trajectories_per_prompt"] == 16
    assert matched["world_size"] == 8
    assert matched["actor_micro_batch_size_per_gpu"] == 32
    assert matched["gradient_accumulation_steps"] == 1
    assert matched["learning_rate"] == 1.0e-6
    assert matched["maximum_optimizer_steps"] == 16


def test_placeholder_cannot_be_used_for_a_launchable_contract(tmp_path: Path) -> None:
    placeholder = tmp_path / _CONFIG.name
    placeholder.write_text(
        _CONFIG.read_text(encoding="utf-8").replace(
            'commit = "46586bf7685e4a58240458e94905e0bf69dbc843"',
            f'commit = "{CROP_TFREE_CODE_PLACEHOLDER}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bound code.commit"):
        load_crop_tfree_run_contract(
            placeholder,
            repository_root=_ROOT,
            allow_placeholder=False,
        )
