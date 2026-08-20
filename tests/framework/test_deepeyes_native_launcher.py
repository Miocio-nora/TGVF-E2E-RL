from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.framework.verl.deepeyes_native_launcher import (
    DEEPEYES_NATIVE_SMOKE_N,
    DEEPEYES_NATIVE_SMOKE_PROMPTS,
    DEEPEYES_NATIVE_STRESS_N,
    DEEPEYES_NATIVE_STRESS_PROMPTS,
    DEEPEYES_NATIVE_STRESS_SCOPE,
    DEEPEYES_NATIVE_VLLM_ATTENTION_BACKEND,
    DEEPEYES_NATIVE_VLLM_MM_ENCODER_ATTN_BACKEND,
    DeepEyesNativeVerlLaunchPlan,
    apply_launch_environment,
    build_deepeyes_native_verl_launch_plan,
)
from tgvf_rl.framework.verl.native_deepeyes_manager import (
    PRL13_AGENT_LOOP_MANAGER_FQN,
)
from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODULE,
)
from tgvf_rl.framework.verl.prl13_main import (
    compose_pinned_deepeyes_config,
    create_prl13_task_runner_class,
    preflight_pinned_deepeyes_config,
)
from tgvf_rl.framework.verl.torch_bert_padding import (
    PRL13_TORCH_BERT_PADDING_SCHEMA,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    load_deepeyes_native_run_contract,
)


_ROOT = Path(__file__).parents[2]
_TEMPLATE = (
    _ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.template.toml"
)


def _contract():
    return load_deepeyes_native_run_contract(_TEMPLATE)


def test_task_runner_wraps_exact_upstream_class_without_global_setup_hook() -> None:
    from verl.trainer.main_ppo_v0 import TaskRunner

    wrapped = create_prl13_task_runner_class()
    wrapped_class = wrapped.__ray_actor_class__
    assert wrapped_class.__name__ == "PRL13TaskRunner"
    assert wrapped_class.__mro__[1] is TaskRunner.__ray_actor_class__


def test_formal_plan_is_upstream_full_model_fsdp2_and_segmented() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="formal", target_step=8
    )
    values = plan.overrides
    assert values["trainer.use_v1"] is False
    assert values["trainer.total_training_steps"] == 8
    assert values["trainer.save_freq"] == 8
    assert values["trainer.test_freq"] == 8
    assert values["trainer.val_before_train"] is False
    assert values["actor_rollout_ref.actor.strategy"] == "fsdp2"
    assert values["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 4
    assert values["actor_rollout_ref.model.lora_rank"] == 0
    assert (
        values["actor_rollout_ref.model.override_config.attn_implementation"] == "sdpa"
    )
    assert "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook" not in values
    assert values["actor_rollout_ref.rollout.free_cache_engine"] is True
    assert values["actor_rollout_ref.rollout.enable_sleep_mode"] is True
    assert values["actor_rollout_ref.rollout.max_model_len"] == 32_768
    assert (
        values[
            "actor_rollout_ref.rollout.engine_kwargs.vllm."
            "mm_encoder_attn_backend"
        ]
        == DEEPEYES_NATIVE_VLLM_MM_ENCODER_ATTN_BACKEND
        == "TORCH_SDPA"
    )
    assert values[
        "actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt"
    ] == {"image": 7, "video": 0}
    assert (
        values["actor_rollout_ref.model.external_lib"]
        == NATIVE_DEEPEYES_POLICY_LOSS_MODULE
    )
    assert (
        values["actor_rollout_ref.actor.policy_loss.loss_mode"]
        == NATIVE_DEEPEYES_POLICY_LOSS_MODE
    )
    assert values["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 4
    assert values["actor_rollout_ref.rollout.multi_turn.max_user_turns"] == 6
    assert values["actor_rollout_ref.rollout.multi_turn.max_assistant_turns"] == 7
    assert values["actor_rollout_ref.rollout.enable_prefix_caching"] is False
    assert (
        plan.environment["VLLM_ATTENTION_BACKEND"]
        == DEEPEYES_NATIVE_VLLM_ATTENTION_BACKEND
        == "TRITON_ATTN"
    )
    assert plan.environment["VERL_FULL_DETERMINISM"] == "0"
    assert plan.environment["VLLM_BATCH_INVARIANT"] == "0"
    assert values["actor_rollout_ref.actor.checkpoint.save_contents"] == [
        "model",
        "hf_model",
        "optimizer",
        "extra",
    ]
    assert values["reward.reward_manager.source"] == "importlib"
    assert values["reward.num_workers"] == 1
    assert (
        values["actor_rollout_ref.rollout.agent.agent_loop_manager_class"]
        == PRL13_AGENT_LOOP_MANAGER_FQN
    )
    assert "actor_rollout_ref.rollout.checkpoint_manager_class" not in values


def test_smoke_plan_uses_independent_route_and_same_native_components() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="smoke", target_step=1
    )
    values = plan.overrides
    assert values["data.train_batch_size"] == DEEPEYES_NATIVE_SMOKE_PROMPTS
    assert values["trainer.save_freq"] == 1
    assert values["trainer.test_freq"] == 1
    assert values["actor_rollout_ref.rollout.n"] == DEEPEYES_NATIVE_SMOKE_N
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 1
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 2
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == 2
    assert values["actor_rollout_ref.rollout.free_cache_engine"] is True
    assert values["data.train_files"][0].endswith("prl13-smoke.schedule")
    assert values["data.train_max_samples"] == -1
    assert values["actor_rollout_ref.actor.strategy"] == "fsdp2"
    assert values["reward.reward_manager.source"] == "importlib"


@pytest.mark.parametrize(
    ("mode", "local_trajectories", "log_prob_micro", "actor_micro"),
    (
        ("smoke", 2, 2, 1),
        ("stress", 16, 8, 4),
        ("formal", 1024, 8, 4),
    ),
)
def test_each_launch_shape_is_statically_divisible_for_fsdp(
    mode: str,
    local_trajectories: int,
    log_prob_micro: int,
    actor_micro: int,
) -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode=mode, target_step=1
    )
    values = plan.overrides
    world_size = values["trainer.n_gpus_per_node"] * values["trainer.nnodes"]
    expanded = values["data.train_batch_size"] * values["actor_rollout_ref.rollout.n"]

    assert expanded // world_size == local_trajectories
    assert local_trajectories % values[
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
    ] == 0
    assert values[
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
    ] == log_prob_micro
    local_ppo_mini = (
        values["actor_rollout_ref.actor.ppo_mini_batch_size"]
        * values["actor_rollout_ref.rollout.n"]
        // world_size
    )
    assert local_ppo_mini % values[
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
    ] == 0
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == actor_micro


def test_launch_plan_rejects_invalid_static_log_prob_partition() -> None:
    valid = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="smoke", target_step=1
    )
    with pytest.raises(ValueError, match="log-prob micro-batch"):
        DeepEyesNativeVerlLaunchPlan(
            contract=valid.contract,
            mode=valid.mode,
            target_step=valid.target_step,
            overrides={
                **valid.overrides,
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 8,
            },
            environment=valid.environment,
        )


def test_smoke_step_two_is_an_absolute_one_batch_resume_horizon(
    tmp_path: Path,
) -> None:
    first = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="smoke", target_step=1
    )
    resumed = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="smoke", target_step=2
    )

    assert first.checkpoint_root == resumed.checkpoint_root
    assert first.overrides["trainer.total_training_steps"] == 1
    assert first.overrides["trainer.total_epochs"] == 1
    assert resumed.overrides["trainer.resume_mode"] == "auto"
    assert resumed.overrides["trainer.total_training_steps"] == 2
    assert resumed.overrides["trainer.total_epochs"] == 2
    assert resumed.overrides["trainer.save_freq"] == 2
    assert resumed.overrides["data.train_files"] == first.overrides["data.train_files"]

    resumed = DeepEyesNativeVerlLaunchPlan(
        contract=resumed.contract,
        mode=resumed.mode,
        target_step=resumed.target_step,
        overrides={
            **resumed.overrides,
            "trainer.default_local_dir": str(tmp_path),
        },
        environment=resumed.environment,
    )
    (tmp_path / "global_step_1").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    assert resumed.latest_checkpoint_step() == 1
    assert resumed.horizon_already_satisfied() is False

    (tmp_path / "global_step_2").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2", encoding="utf-8")
    assert resumed.horizon_already_satisfied() is True


def test_stress_plan_keeps_formal_shape_on_fixed_four_rows() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="stress", target_step=1
    )
    values = plan.overrides

    assert values["data.train_batch_size"] == DEEPEYES_NATIVE_STRESS_PROMPTS == 4
    assert values["data.gen_batch_size"] == 4
    assert values["data.train_files"] == values["data.val_files"]
    assert values["data.train_files"][0].endswith("prl13-smoke.schedule")
    assert values["actor_rollout_ref.rollout.n"] == DEEPEYES_NATIVE_STRESS_N == 16
    assert values["data.max_response_length"] == 20_480
    assert (
        values["actor_rollout_ref.rollout.custom"]["protocol"][
            "single_response_max_tokens"
        ]
        == 10_240
    )
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == 4
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 4
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 8
    assert values["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == 8
    assert values["actor_rollout_ref.actor.strategy"] == "fsdp2"
    assert values["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == 4
    assert values["actor_rollout_ref.actor.fsdp_config.param_offload"] is True
    assert values["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] is True
    assert values["actor_rollout_ref.actor.fsdp_config.offload_policy"] is True
    assert values["actor_rollout_ref.rollout.gpu_memory_utilization"] == 0.8
    assert values["actor_rollout_ref.rollout.free_cache_engine"] is True
    assert values["actor_rollout_ref.model.lora_rank"] == 0
    assert values["actor_rollout_ref.model.lora.freeze_vision_model"] is False
    assert values["actor_rollout_ref.model.lora.freeze_vision_projection"] is False
    assert values["actor_rollout_ref.model.lora.freeze_language_model"] is False
    assert plan.as_record()["canary_scope"] == DEEPEYES_NATIVE_STRESS_SCOPE


def test_stress_step_two_reuses_root_and_contracts_updated_policy_rollout() -> None:
    first = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="stress", target_step=1
    )
    resumed = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="stress", target_step=2
    )

    assert first.checkpoint_root == resumed.checkpoint_root
    assert first.checkpoint_root.name == "checkpoints"
    assert first.checkpoint_root.parent.name == "stress"
    assert first.overrides["trainer.total_epochs"] == 1
    assert resumed.overrides["trainer.total_epochs"] == 2
    assert resumed.overrides["trainer.total_training_steps"] == 2
    assert resumed.overrides["trainer.resume_mode"] == "auto"
    assert resumed.overrides["trainer.save_freq"] == 2
    assert resumed.overrides["trainer.test_freq"] == 2
    assert resumed.overrides["trainer.max_actor_ckpt_to_keep"] == 2
    assert resumed.overrides["data.train_files"] == first.overrides["data.train_files"]
    assert resumed.as_record()["step_2_rollout_contract"] == (
        "rollout follows the synchronized step-1 actor update"
    )


def test_resident_stress_changes_only_systems_shape_and_output_root() -> None:
    baseline = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="stress", target_step=1
    )
    resident = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="resident-stress", target_step=1
    )
    values = resident.overrides

    assert resident.checkpoint_root.parent.name == "stress-resident"
    assert values["data.train_files"] == baseline.overrides["data.train_files"]
    assert values["data.train_batch_size"] == 4
    assert values["actor_rollout_ref.rollout.n"] == 16
    assert values["actor_rollout_ref.actor.fsdp_config.param_offload"] is False
    assert values["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] is False
    assert values["actor_rollout_ref.actor.fsdp_config.offload_policy"] is False
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 8
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 16


def test_resident_fast_stress_spends_b200_memory_to_remove_recompute() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="resident-fast-stress", target_step=1
    )
    values = plan.overrides

    assert plan.checkpoint_root.parent.name == "stress-resident-fast"
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is False
    assert (
        values["actor_rollout_ref.actor.fsdp_config.reshard_after_forward"] is False
    )
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 16
    assert values["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == 16


def test_resident_flex_stress_uses_packed_block_mask_attention() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="resident-flex-stress", target_step=1
    )
    values = plan.overrides

    assert plan.checkpoint_root.parent.name == "stress-resident-flex"
    assert values[
        "actor_rollout_ref.model.override_config.attn_implementation"
    ] == "sdpa"
    assert values["actor_rollout_ref.model.override_config.text_config"] == {
        "_attn_implementation_internal": "flex_attention"
    }
    assert values["actor_rollout_ref.model.override_config.vision_config"] == {
        "_attn_implementation_internal": "sdpa"
    }
    assert values["actor_rollout_ref.model.use_remove_padding"] is True
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is True
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 8


def test_resident_wide_flex_stress_uses_one_local_actor_batch() -> None:
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="resident-wide-flex-stress", target_step=2
    )
    values = plan.overrides

    assert plan.checkpoint_root.parent.name == "stress-resident-wide-flex"
    assert values["actor_rollout_ref.model.enable_gradient_checkpointing"] is True
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 16
    assert values["trainer.save_freq"] == 2
    assert values["trainer.test_freq"] == 2


def test_resident_flex_launch_prepends_local_python_headers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tgvf_rl.framework.verl import deepeyes_native_launcher as launcher

    include_root = tmp_path / "usr/include"
    python_include = include_root / "python3.12"
    python_include.mkdir(parents=True)
    (python_include / "Python.h").write_text("/* test */\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_PYTHON312_DEV_ROOT", include_root)
    monkeypatch.setenv("CPATH", "/existing/include")
    plan = build_deepeyes_native_verl_launch_plan(
        _contract(), mode="resident-flex-stress", target_step=1
    )

    apply_launch_environment(plan)

    assert launcher.os.environ["CPATH"].split(launcher.os.pathsep) == [
        str(python_include),
        str(include_root),
        "/existing/include",
    ]


def test_native_launcher_resolves_shared_worktree_dependencies(tmp_path: Path) -> None:
    from tgvf_rl.framework.verl import deepeyes_native_launcher as launcher

    shared_root = tmp_path / "main"
    worktree_root = tmp_path / "worktree"
    git_dir = shared_root / ".git/worktrees/prl24-d"
    header = (
        shared_root
        / ".deps/python312-dev/root/usr/include/python3.12/Python.h"
    )
    git_dir.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    header.write_text("/* test */\n", encoding="utf-8")
    worktree_root.mkdir()
    (worktree_root / ".git").write_text(
        f"gitdir: {git_dir}\n", encoding="utf-8"
    )

    assert launcher._shared_dependency_repository_root(worktree_root) == shared_root


@pytest.mark.parametrize(
    "mode",
    [
        "smoke",
        "stress",
        "resident-stress",
        "resident-fast-stress",
        "resident-flex-stress",
        "resident-wide-flex-stress",
    ],
)
@pytest.mark.parametrize("target_step", [0, 3])
def test_canaries_reject_non_resume_horizons(mode: str, target_step: int) -> None:
    with pytest.raises(ValueError, match="step 1 or 2"):
        build_deepeyes_native_verl_launch_plan(
            _contract(), mode=mode, target_step=target_step
        )


def test_real_pinned_hydra_compose_and_custom_class_resolution() -> None:
    for mode, target in (
        ("formal", 8),
        ("smoke", 1),
        ("smoke", 2),
        ("stress", 1),
        ("stress", 2),
        ("resident-stress", 1),
        ("resident-stress", 2),
        ("resident-fast-stress", 1),
        ("resident-fast-stress", 2),
        ("resident-flex-stress", 1),
        ("resident-flex-stress", 2),
        ("resident-wide-flex-stress", 1),
        ("resident-wide-flex-stress", 2),
    ):
        plan = build_deepeyes_native_verl_launch_plan(
            _contract(), mode=mode, target_step=target
        )
        config = compose_pinned_deepeyes_config(plan.hydra_override_args())
        result = preflight_pinned_deepeyes_config(config)
        assert result == {
            "need_reference_policy": False,
            "need_critic": False,
            "dataset_class": (
                "tgvf_rl.framework.verl.deepeyes_official_dataset."
                "TGVFDeepEyesOfficialDataset"
            ),
            "reward_manager_class": (
                "tgvf_rl.rewards.deepeyes_verl_reward.DeepEyesOfficialRewardManager"
            ),
            "agent_loop_manager_class": PRL13_AGENT_LOOP_MANAGER_FQN,
            "padding_backend": PRL13_TORCH_BERT_PADDING_SCHEMA,
            "policy_loss_mode": NATIVE_DEEPEYES_POLICY_LOSS_MODE,
        }
        assert (
            config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
            == PRL13_AGENT_LOOP_MANAGER_FQN
        )
        assert config.actor_rollout_ref.rollout.get("checkpoint_manager_class") is None
        assert config.actor_rollout_ref.rollout.get("deepeyes_official") is None
        assert (
            config.actor_rollout_ref.rollout.custom.protocol.coordinate_mapper
            == "qwen_0_1000_to_source_v1"
        )
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm[
                "mm_encoder_attn_backend"
            ]
            == "TORCH_SDPA"
        )
        assert config.actor_rollout_ref.rollout.engine_kwargs.vllm[
            "limit_mm_per_prompt"
        ] == {"image": 7, "video": 0}
