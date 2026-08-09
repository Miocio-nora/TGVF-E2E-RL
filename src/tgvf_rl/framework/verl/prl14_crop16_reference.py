"""Immutable common controls from the completed PRL14 Crop-16 run.

PRL15 is a treatment of PRL14: native Crop is replaced by trainable RP66/TGVF.
This module intentionally points in that direction. It must never derive the
Crop control from a TGVF configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType


PRL14_CROP16_RUN_NAME = (
    "PRL-14-A-QWEN3-INSTRUCT-GRPO-BS16-N16-NATIVE-CROP-T1-"
    "CLEANFINAL-16STEP-WS8"
)
PRL14_CROP16_COMPLETION_SHA256 = (
    "3907b310642aa542cf7ffcb6dec12c2a23d87634cdd9f696b5f984eacb1f70f1"
)
PRL14_CROP16_WANDB_RUN_IDS = ("bzkj3ptg", "z5k7y418")
PRL14_CROP16_ACTUAL_TARGET_STEP = 16
PRL14_CROP16_COMPARISON_STEP = 8
PRL14_CROP16_COMPLETION_PATH = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-"
    "cleanfinal-16step-ws8/completion.json"
)


@dataclass(frozen=True, slots=True)
class PRL14Crop16Completion:
    """Hash-verified immutable record produced by the completed control."""

    overrides: Mapping[str, object]
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )


def load_prl14_crop16_completion() -> PRL14Crop16Completion:
    raw = PRL14_CROP16_COMPLETION_PATH.read_bytes()
    if sha256(raw).hexdigest() != PRL14_CROP16_COMPLETION_SHA256:
        raise ValueError("completed PRL14 Crop-16 record SHA256 differs")
    payload = json.loads(raw)
    if (
        payload.get("run_name") != PRL14_CROP16_RUN_NAME
        or payload.get("actual_target_step") != PRL14_CROP16_ACTUAL_TARGET_STEP
        or not isinstance(payload.get("overrides"), Mapping)
        or not isinstance(payload.get("environment"), Mapping)
    ):
        raise ValueError("completed PRL14 Crop-16 identity differs")
    return PRL14Crop16Completion(
        overrides=payload["overrides"],
        environment=payload["environment"],
    )

# Only common scientific/execution controls belong here. Tool/prompt, RP66
# state, the TGVF Dataset/AgentLoop and paired checkpoint plumbing are the
# deliberately different treatment fields.
PRL14_CROP16_COMMON_OVERRIDES = MappingProxyType(
    {
        "data.train_batch_size": 16,
        "data.gen_batch_size": 16,
        "data.max_prompt_length": 8192,
        "data.max_response_length": 20480,
        "data.filter_overlong_prompts": True,
        "data.shuffle": False,
        "data.seed": 42,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.model.lora.freeze_vision_model": False,
        "actor_rollout_ref.model.lora.freeze_vision_projection": False,
        "actor_rollout_ref.model.lora.freeze_language_model": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.use_liger": False,
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.model.fused_kernel_options": {"impl_backend": "torch"},
        "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.model.override_config.text_config": {
            "_attn_implementation_internal": "flex_attention"
        },
        "actor_rollout_ref.model.override_config.vision_config": {
            "_attn_implementation_internal": "sdpa"
        },
        "actor_rollout_ref.actor.freeze_vision_tower": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.optim.total_training_steps": -1,
        "actor_rollout_ref.actor.optim.override_optimizer_config": None,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 8,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": True,
        "actor_rollout_ref.actor.fsdp_config.full_determinism": False,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": True,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 8,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.ref.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
        "actor_rollout_ref.ref.fsdp_config.reshard_after_forward": True,
        "actor_rollout_ref.ref.fsdp_config.full_determinism": False,
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile": True,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.rollout.n": 16,
        "actor_rollout_ref.rollout.response_length": 20480,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.65,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
        "actor_rollout_ref.rollout.max_model_len": 32768,
        "actor_rollout_ref.rollout.max_num_seqs": 1024,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enable_prefix_caching": False,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.agent.num_workers": 8,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 8,
        "trainer.save_freq": 1,
        "trainer.test_freq": 0,
        "trainer.val_before_train": False,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.resume_mode": "auto",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "algorithm.kl_ctrl.kl_coef": 0.0,
    }
)

PRL14_CROP16_REMOVED_OVERRIDES = (
    "actor_rollout_ref.actor.fsdp_config.mixed_precision",
    "actor_rollout_ref.ref.fsdp_config.mixed_precision",
    "actor_rollout_ref.rollout.repetition_penalty",
)


def apply_prl14_crop16_common_controls(
    values: MutableMapping[str, object],
) -> None:
    """Mutate a treatment plan so every common field equals Crop-16."""

    for path in PRL14_CROP16_REMOVED_OVERRIDES:
        values.pop(path, None)
    for path, value in PRL14_CROP16_COMMON_OVERRIDES.items():
        values[path] = deepcopy(value)


def assert_prl14_crop16_common_controls(values: Mapping[str, object]) -> None:
    mismatches = {
        path: (values.get(path), expected)
        for path, expected in PRL14_CROP16_COMMON_OVERRIDES.items()
        if values.get(path) != expected
    }
    unexpected = {
        path: values[path]
        for path in PRL14_CROP16_REMOVED_OVERRIDES
        if path in values
    }
    if mismatches or unexpected:
        raise ValueError(
            "PRL15 differs from the completed PRL14 Crop-16 control: "
            f"mismatches={mismatches!r}, unexpected={unexpected!r}"
        )


__all__ = [
    "PRL14_CROP16_COMMON_OVERRIDES",
    "PRL14_CROP16_ACTUAL_TARGET_STEP",
    "PRL14_CROP16_COMPARISON_STEP",
    "PRL14_CROP16_COMPLETION_PATH",
    "PRL14_CROP16_COMPLETION_SHA256",
    "PRL14_CROP16_REMOVED_OVERRIDES",
    "PRL14_CROP16_RUN_NAME",
    "PRL14_CROP16_WANDB_RUN_IDS",
    "PRL14Crop16Completion",
    "apply_prl14_crop16_common_controls",
    "assert_prl14_crop16_common_controls",
    "load_prl14_crop16_completion",
]
