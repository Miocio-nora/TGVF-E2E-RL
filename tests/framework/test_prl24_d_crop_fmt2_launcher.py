from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).parents[2]
_LAUNCHER = _ROOT / "tools/launch_prl21_crop_tfree16.py"
_CONFIG = (
    _ROOT / "configs/policy/runs/"
    "prl_24_d_fmt2_qwen3_instruct_full_crop_bs64_n16_tfree_teacher25_16step_ws8.toml"
)


def _module():
    spec = importlib.util.spec_from_file_location("launch_prl24_d_crop_fmt2", _LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(module):
    return module.load_crop_tfree_run_contract(
        _CONFIG,
        repository_root=_ROOT,
        allow_placeholder=False,
    )


def test_prl24_d_formal_plan_is_bs64_n16_fmt2() -> None:
    module = _module()
    contract = _contract(module)
    plan = module._build_plan(contract, mode="formal")
    values = plan.overrides

    assert contract.reward_profile == "stage3-shaped-v1-tfree-fmt2"
    assert contract.protocol_error_penalty == 2.0
    assert contract.gradient_accumulation_steps == 4
    assert contract.single_pass_execution is True
    assert contract.crop_aware_batching is True
    assert contract.permanent_checkpoint_steps == (2, 4, 8, 12, 16)
    assert values["reward.reward_manager.name"] == (
        "DeepEyesCropTFreeFMT2RewardManager"
    )
    assert values["data.train_batch_size"] == 64
    assert values["data.gen_batch_size"] == 64
    assert values["data.mm_processor_kwargs.max_pixels"] == 1_003_520
    assert values["actor_rollout_ref.rollout.n"] == 16
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == 64
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 32
    assert values["trainer.n_gpus_per_node"] == 8
    assert values["trainer.total_training_steps"] == 16
    assert values["actor_rollout_ref.actor.policy_loss.loss_mode"] == (
        "deepeyes_single_pass_micro_token_mean"
    )
    assert values["actor_rollout_ref.actor.calculate_entropy"] is False
    assert values["algorithm.rollout_correction.bypass_mode"] is True
    assert values["algorithm.rollout_correction.rollout_is"] is None
    assert values["algorithm.rollout_correction.rollout_rs"] is None
    assert plan.environment["TGVF_PRL24_SINGLE_PASS_EXECUTION"] == "1"
    assert plan.environment["TGVF_PRL24_CROP_AWARE_BATCHING"] == "1"

    # 64 prompts * 16 trajectories / (8 ranks * 32 trajectories/rank)
    # gives the four forward/backward microbatches recorded by the contract.
    assert (
        64 * 16 // (8 * values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"])
        == contract.gradient_accumulation_steps
    )


def test_prl24_d_smoke_keeps_fmt2_without_wandb() -> None:
    module = _module()
    plan = module._build_plan(_contract(module), mode="smoke")

    assert plan.overrides["reward.reward_manager.name"] == (
        "DeepEyesCropTFreeFMT2RewardManager"
    )
    assert plan.overrides["trainer.logger"] == ["console"]
    assert plan.overrides["trainer.total_training_steps"] == 1
    assert plan.overrides["data.mm_processor_kwargs.max_pixels"] == 1_003_520
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert plan.environment["TGVF_PRL24_SINGLE_PASS_EXECUTION"] == "1"
    assert plan.environment["TGVF_PRL24_CROP_AWARE_BATCHING"] == "1"


def test_prl24_d_single_pass_compose_preflight_is_fail_closed() -> None:
    module = _module()
    plan = module._build_plan(_contract(module), mode="formal")
    config = module.compose_pinned_deepeyes_config(plan.hydra_override_args())
    result = module.preflight_pinned_deepeyes_config(config)

    assert result["policy_loss_mode"] == ("deepeyes_single_pass_micro_token_mean")
    assert result["single_pass_self_anchor"] is True
    assert result["need_reference_policy"] is False
    assert result["need_critic"] is False
