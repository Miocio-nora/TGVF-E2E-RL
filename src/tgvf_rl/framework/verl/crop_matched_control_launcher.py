"""World-4/micro-1 Crop control matched to the PRL15 RP66 pilot.

The released DeepEyes actor objective is an equal average of token means over
fixed local micro-batches. Its scalar therefore depends on the actor micro
shape whenever trajectory lengths differ. This launcher keeps Crop's native
tool/prompt path while matching PRL15's optimization and serving shape so the
tool comparison does not silently include a micro-batch objective change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from tgvf_rl.policy.deepeyes_native_contract import DeepEyesNativeRunContract
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig

from .deepeyes_native_launcher import (
    DeepEyesNativeVerlLaunchPlan,
    apply_launch_environment,
    build_deepeyes_native_verl_launch_plan,
)
from .prl13_main import (
    compose_pinned_deepeyes_config,
    preflight_pinned_deepeyes_config,
    run_pinned_deepeyes_config,
)
from .trainable_tgvf_launcher import build_trainable_tgvf_verl_launch_plan


CROP_MATCHED_CONTROL_SCHEMA = "tgvf.prl15-crop-ws4-micro1-control.v1"
CROP_MATCHED_CONTROL_RUN_ID = (
    "PRL-15-C0-QWEN3-INSTRUCT-FULL-CROP-BS16-N16-WS4-MICRO1-8STEP"
)
CROP_MATCHED_CONTROL_OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-15-C0-qwen3-instruct-full-crop-bs16-n16-ws4-micro1-8step"
)
CROP_MATCHED_CONTROL_TARGET_STEP = 8
CROP_MATCHED_CONTROL_JUDGE_CONFIG = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/policy/judges/"
    "prl13_qwen25_72b_binary_text_resilient.json"
)

# Prompt, tool runtime, trainable RP66 state/synchronization, and output paths
# define the experimental arms and are deliberately not in this equality set.
MATCHED_OVERRIDE_PATHS = (
    "data.train_files",
    "data.val_files",
    "data.train_batch_size",
    "data.gen_batch_size",
    "data.seed",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.model.lora.rank",
    "actor_rollout_ref.model.lora.freeze_vision_model",
    "actor_rollout_ref.model.lora.freeze_vision_projection",
    "actor_rollout_ref.model.lora.freeze_language_model",
    "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor_rollout_ref.actor.freeze_vision_tower",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_epochs",
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.policy_loss.loss_mode",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.optim.lr_scheduler_type",
    "actor_rollout_ref.actor.optim.lr_warmup_steps",
    "actor_rollout_ref.actor.optim.clip_grad",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.use_kl_in_reward",
    "algorithm.kl_ctrl.kl_coef",
    "trainer.nnodes",
    "trainer.n_gpus_per_node",
    "trainer.total_training_steps",
    "trainer.val_before_train",
)


@dataclass(frozen=True, slots=True)
class CropMatchedControlPlan:
    """Validated Crop control plus its explicit cross-arm equality proof."""

    launch: DeepEyesNativeVerlLaunchPlan
    matched_values: Mapping[str, object]
    schema_version: str = CROP_MATCHED_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CROP_MATCHED_CONTROL_SCHEMA:
            raise ValueError("Crop matched-control schema differs")
        if self.launch.mode != "formal" or self.launch.target_step != 8:
            raise ValueError("Crop matched control must be a formal step-8 plan")
        object.__setattr__(
            self, "matched_values", MappingProxyType(dict(self.matched_values))
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
                "world_size": 4,
                "actor_micro_batch_size_per_gpu": 1,
                "loss_objective": "deepeyes_official_micro_token_mean",
                "reason": (
                    "remove actor micro-batch reduction as a Crop-vs-RP66 "
                    "confound"
                ),
            },
            "matched_values": dict(self.matched_values),
            "launch": self.launch.as_record(),
        }


def build_crop_matched_control_plan(
    crop_contract: DeepEyesNativeRunContract,
    rp66_config: PolicyE2ESmokeRunConfig,
) -> CropMatchedControlPlan:
    """Build Crop and prove every declared common runtime value equals RP66."""

    rp66 = build_trainable_tgvf_verl_launch_plan(rp66_config, mode="formal")
    base = build_deepeyes_native_verl_launch_plan(
        crop_contract, mode="formal", target_step=CROP_MATCHED_CONTROL_TARGET_STEP
    )
    values = dict(base.overrides)
    values.update(
        {
            "trainer.experiment_name": CROP_MATCHED_CONTROL_RUN_ID,
            "trainer.default_local_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "checkpoints"
            ),
            "trainer.rollout_data_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "trajectories"
            ),
            "trainer.validation_data_dir": str(
                CROP_MATCHED_CONTROL_OUTPUT_ROOT / "validation"
            ),
            "trainer.total_training_steps": CROP_MATCHED_CONTROL_TARGET_STEP,
            "trainer.save_freq": 1,
            "trainer.test_freq": 0,
            "trainer.val_before_train": False,
            "trainer.resume_mode": "disable",
            "trainer.max_actor_ckpt_to_keep": 4,
            "data.train_batch_size": 16,
            "data.gen_batch_size": 16,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 16,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 16,
            "trainer.n_gpus_per_node": 4,
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": 4,
            "actor_rollout_ref.ref.fsdp_config.fsdp_size": 4,
            "actor_rollout_ref.rollout.agent.num_workers": 4,
            "actor_rollout_ref.actor.fsdp_config.param_offload": False,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
            "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
            "actor_rollout_ref.ref.fsdp_config.param_offload": False,
            "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
            "actor_rollout_ref.model.enable_gradient_checkpointing": False,
            "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": False,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.45,
            "actor_rollout_ref.model.use_fused_kernels": False,
            "reward.deepeyes_official.judge_service_config_path": (
                CROP_MATCHED_CONTROL_JUDGE_CONFIG
            ),
            "reward.deepeyes_official.judge_service_config_sha256": (
                rp66_config.reward.judge_config_sha256
            ),
        }
    )
    environment = dict(base.environment)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    launch = replace(base, overrides=values, environment=environment)

    mismatches = {
        path: (launch.overrides.get(path), rp66.overrides.get(path))
        for path in MATCHED_OVERRIDE_PATHS
        if launch.overrides.get(path) != rp66.overrides.get(path)
    }
    if mismatches:
        raise ValueError(f"Crop/RP66 matched-control values differ: {mismatches!r}")
    matched = {path: launch.overrides[path] for path in MATCHED_OVERRIDE_PATHS}
    return CropMatchedControlPlan(launch=launch, matched_values=matched)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-contract", required=True, type=Path)
    parser.add_argument("--rp66-config", required=True, type=Path)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start GPU/API work; omit for compose-only preflight.",
    )
    args = parser.parse_args(argv)

    from tgvf_rl.policy.deepeyes_native_contract import (
        load_deepeyes_native_run_contract,
    )
    from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

    crop_contract = load_deepeyes_native_run_contract(args.crop_contract.resolve())
    rp66_config = load_policy_e2e_smoke_run_config(args.rp66_config.resolve())
    control = build_crop_matched_control_plan(crop_contract, rp66_config)
    composed = compose_pinned_deepeyes_config(control.launch.hydra_override_args())
    preflight = preflight_pinned_deepeyes_config(composed)
    record = {**control.as_record(), "compose_preflight": preflight}
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    if not args.launch:
        return 0

    crop_contract.assert_launchable(Path(__file__).resolve().parents[4])
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    if CROP_MATCHED_CONTROL_OUTPUT_ROOT.exists():
        raise RuntimeError("Crop matched-control output root already exists")
    apply_launch_environment(control.launch)
    run_pinned_deepeyes_config(composed)
    control.launch.assert_target_checkpoint_complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CROP_MATCHED_CONTROL_OUTPUT_ROOT",
    "CROP_MATCHED_CONTROL_JUDGE_CONFIG",
    "CROP_MATCHED_CONTROL_RUN_ID",
    "CROP_MATCHED_CONTROL_SCHEMA",
    "CROP_MATCHED_CONTROL_TARGET_STEP",
    "MATCHED_OVERRIDE_PATHS",
    "CropMatchedControlPlan",
    "build_crop_matched_control_plan",
    "main",
]
