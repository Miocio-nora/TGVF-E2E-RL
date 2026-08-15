from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


_ROOT = Path(__file__).parents[2]
_LAUNCHER = _ROOT / "tools/launch_prl21_crop_tfree16.py"
_EVALUATOR = _ROOT / "tools/supervise_prl21_crop_tfree16_eval.sh"
_CONFIG = (
    _ROOT / "configs/policy/runs/"
    "prl_21_r0_qwen3_instruct_full_crop_bs16_n16_tfree_16step_ws8.toml"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "launch_prl21_crop_tfree16", _LAUNCHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(module):
    return module.load_crop_tfree_run_contract(
        _CONFIG,
        repository_root=_ROOT,
    )


def test_formal_plan_is_prl14_matched_except_for_reward_manager_and_horizon() -> None:
    module = _module()
    plan = module._build_plan(_contract(module), mode="formal")
    values = plan.overrides

    assert values["reward.reward_manager.name"] == "DeepEyesCropTFreeRewardManager"
    assert (
        values["reward.reward_manager.module.name"] == "DeepEyesCropTFreeRewardManager"
    )
    assert values["reward.reward_manager.module.path"] == (
        "pkg://tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward"
    )
    assert values["data.train_batch_size"] == 16
    assert values["data.gen_batch_size"] == 16
    assert values["actor_rollout_ref.rollout.n"] == 16
    assert values["trainer.n_gpus_per_node"] == 8
    assert values["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 8
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 32
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 32
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == 32
    assert values["actor_rollout_ref.actor.optim.lr"] == 1.0e-6
    assert values["trainer.total_training_steps"] == 16
    assert values["trainer.save_freq"] == 1
    assert values["trainer.test_freq"] == 0
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_smoke_uses_the_same_reward_but_never_logs_to_wandb() -> None:
    module = _module()
    plan = module._build_plan(_contract(module), mode="smoke")
    values = plan.overrides

    assert values["reward.reward_manager.name"] == "DeepEyesCropTFreeRewardManager"
    assert values["trainer.logger"] == ["console"]
    assert values["data.train_batch_size"] == 4
    assert values["actor_rollout_ref.rollout.n"] == 2
    assert values["trainer.total_training_steps"] == 1
    assert values["trainer.n_gpus_per_node"] == 4
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_evaluator_has_one_generic_execution_path_and_rejects_extra_arguments() -> None:
    script = _EVALUATOR.read_text(encoding="utf-8")

    assert 'evaluator="$repo_root/tools/run_paired_policy_evaluation.py"' in script
    assert "--mode run" in script
    assert "--resume-scoring" not in script
    assert "run_prl15_paired_evaluation.py" not in script
    assert "materialize_scoring_views" not in script

    completed = subprocess.run(
        ["bash", str(_EVALUATOR), "unexpected-argument"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("usage: ")
