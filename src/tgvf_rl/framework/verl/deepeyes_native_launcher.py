"""Executable pinned-veRL launch plan for the PRL13 DeepEyes control.

The plan deliberately uses upstream e003 full-model FSDP2, an upstream-derived
manager that only rectangularizes heterogeneous worker sidecars, and the
standard v0 ``TaskRunner``.  It does not import
the older project Policy launcher, LoRA runner, exact replay engine, or custom
checkpoint engine.  Formal training is launched in absolute horizon segments
``1 -> 8 -> 20 -> 45 -> 80`` so upstream's mandatory last-step save gives the
non-uniform checkpoint gates without validating before a save.  The independent
smoke can run ``1 -> 2`` so the second invocation exercises checkpoint restore
and a rollout from the updated policy.  The separate ``stress`` canary uses the
same deterministic four rows but restores the formal rollout/training shape
(``n=16`` and the formal token/micro-batch limits).  It is a shape canary, not a
worst-case visual-token benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tgvf_rl.framework.verl.deepeyes_official_dataset import (
    DEEPEYES_OFFICIAL_DATASET_SCHEMA,
    DEEPEYES_SMOKE_SENTINEL,
)
from tgvf_rl.framework.verl.native_deepeyes_manager import (
    PRL13_AGENT_LOOP_MANAGER_FQN,
)
from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODULE,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    DEEPEYES_NATIVE_CHECKPOINT_GATES,
    DeepEyesNativeRunContract,
)


DEEPEYES_NATIVE_LAUNCH_SCHEMA = "tgvf.deepeyes-native-verl-launch.v1"
DEEPEYES_NATIVE_VERL_COMMIT = "e003163181731412595257a72ec173071efb125f"
DEEPEYES_NATIVE_FORMAL_HORIZONS = DEEPEYES_NATIVE_CHECKPOINT_GATES
DEEPEYES_NATIVE_SMOKE_HORIZONS = (1, 2)
DEEPEYES_NATIVE_SMOKE_PROMPTS = 4
DEEPEYES_NATIVE_SMOKE_N = 2
DEEPEYES_NATIVE_VLLM_ATTENTION_BACKEND = "TRITON_ATTN"
DEEPEYES_NATIVE_VLLM_MM_ENCODER_ATTN_BACKEND = "TORCH_SDPA"
DEEPEYES_NATIVE_STRESS_HORIZONS = (1, 2)
DEEPEYES_NATIVE_STRESS_PROMPTS = DEEPEYES_NATIVE_SMOKE_PROMPTS
DEEPEYES_NATIVE_STRESS_N = 16
DEEPEYES_NATIVE_STRESS_SCOPE = (
    "formal_shape_only_fixed_four_source_covering_rows_"
    "not_maximum_visual_token_selection"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PYTHON312_DEV_ROOT = (
    _REPOSITORY_ROOT / ".deps/python312-dev/root/usr/include"
)

LaunchMode = Literal[
    "formal",
    "smoke",
    "stress",
    "resident-stress",
    "resident-fast-stress",
    "resident-flex-stress",
    "resident-wide-flex-stress",
]


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"PRL13 contract is missing {path}")
        value = value[part]
    return value


def _hydra_literal(value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, Mapping):
        fields: list[str] = []
        for key in sorted(value):
            if not key or any(
                not (character.isalnum() or character in "_-") for character in key
            ):
                raise ValueError(f"Hydra mapping key is unsafe: {key!r}")
            fields.append(f"{key}:{_hydra_literal(value[key])}")
        return "{" + ",".join(fields) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_literal(item) for item in value) + "]"
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _official_dataset_binding(contract: DeepEyesNativeRunContract) -> dict[str, object]:
    dataset = contract.payload["dataset"]
    protocol = contract.payload["protocol"]
    return {
        "schema_version": DEEPEYES_OFFICIAL_DATASET_SCHEMA,
        "root": dataset["root"],
        "candidate_sidecar_path": dataset["candidate_sidecar_path"],
        "manifest_file_sha256": dataset["manifest_file_sha256"],
        "content_sha256": dataset["content_sha256"],
        "samples_sha256": dataset["samples_sha256"],
        "candidate_sidecar_sha256": dataset["candidate_sidecar_sha256"],
        "expected_sample_count": dataset["sample_count"],
        "schedule_mode": dataset["schedule_mode"],
        "schedule_seed": dataset["schedule_seed"],
        "probe_seed": dataset["probe_seed"],
        "visual_prompt_bundle_sha256": protocol["visual_prompt_bundle_sha256"],
        "thinklite_prompt_bundle_sha256": protocol["thinklite_prompt_bundle_sha256"],
    }


def _formal_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    payload = contract.payload
    model = payload["model"]
    data = payload["dataset"]
    reward = payload["reward"]
    rollout = payload["rollout"]
    training = payload["training"]
    output = payload["output"]
    actor_checkpoint_contents = list(training["checkpoint_contents"])
    overrides: dict[str, object] = {
        # Upstream e003 v0 execution path.  No project exact-replay TaskRunner.
        "trainer.use_v1": False,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 4,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": target_step,
        # Each invocation stops exactly on one absolute gate.  Pinned e003
        # saves the last step, synchronizes the updated actor into rollout, and
        # only then runs last-step validation when both frequencies are > 0.
        "trainer.save_freq": target_step,
        "trainer.test_freq": target_step,
        "trainer.val_before_train": False,
        "trainer.resume_mode": "auto",
        "trainer.default_local_dir": output["checkpoint_directory"],
        "trainer.max_actor_ckpt_to_keep": len(DEEPEYES_NATIVE_FORMAL_HORIZONS),
        "trainer.balance_batch": True,
        "trainer.logger": list(training["logger"]),
        "trainer.project_name": training["project_name"],
        "trainer.experiment_name": contract.run_id,
        "trainer.rollout_data_dir": str(Path(output["root"]) / "trajectories"),
        "trainer.validation_data_dir": str(Path(output["root"]) / "validation"),
        # Exact non-repeating schedule selected by the custom Dataset.  Dataset
        # tools are disabled because AgentLoop owns the tool protocol.
        "data.train_files": list(data["train_files"]),
        "data.val_files": list(data["probe_files"]),
        "data.train_max_samples": -1,
        "data.val_max_samples": -1,
        "data.train_batch_size": data["batch_size"],
        "data.gen_batch_size": data["batch_size"],
        "data.max_prompt_length": rollout["max_prompt_length"],
        "data.max_response_length": rollout["max_response_length"],
        "data.shuffle": False,
        "data.seed": data["schedule_seed"],
        "data.validation_shuffle": False,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "data.return_multi_modal_inputs": True,
        "data.tool_config_path": None,
        "data.function_tool_path": None,
        "data.custom_cls.path": data["verl_dataset_module_path"],
        "data.custom_cls.name": data["verl_dataset_class_name"],
        "data.deepeyes_official": _official_dataset_binding(contract),
        # Qwen3-VL-8B full model; vision, projector and language stay trainable.
        "actor_rollout_ref.model.path": model["path"],
        # The external module registers the exact DeepEyes actor reduction;
        # it also installs the project-owned Torch padding backend in every
        # FSDP worker before pinned veRL can request flash_attn utilities.
        "actor_rollout_ref.model.external_lib": (
            NATIVE_DEEPEYES_POLICY_LOSS_MODULE
        ),
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.model.lora.freeze_vision_model": False,
        "actor_rollout_ref.model.lora.freeze_vision_projection": False,
        "actor_rollout_ref.model.lora.freeze_language_model": False,
        # The accepted environment intentionally has no flash-attn wheel.
        # Transformers SDPA is already proven for this exact Qwen3 snapshot.
        "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": True,
        "actor_rollout_ref.actor.freeze_vision_tower": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": data["batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": _nested(
            payload, "optimization.actor_micro_batch_size_per_gpu"
        ),
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.loss_agg_mode": (
            NATIVE_DEEPEYES_LOSS_AGG_MODE
        ),
        "actor_rollout_ref.actor.policy_loss.loss_mode": _nested(
            payload, "optimization.actor_loss_reduction"
        ),
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.checkpoint.save_contents": actor_checkpoint_contents,
        "actor_rollout_ref.actor.checkpoint.load_contents": actor_checkpoint_contents,
        "actor_rollout_ref.actor.optim.optimizer": "AdamW",
        "actor_rollout_ref.actor.optim.lr": _nested(
            payload, "optimization.learning_rate"
        ),
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.clip_grad": 1.0,
        # Ref is disabled by KL=0, but its dormant config is still pinned.
        "actor_rollout_ref.ref.strategy": "fsdp2",
        "actor_rollout_ref.ref.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.offload_policy": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": rollout[
            "reference_logprob_micro_batch_size_per_gpu"
        ],
        # Standard async AgentLoopManager + YAML-registered DeepEyes loop.
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": rollout["trajectories_per_prompt"],
        "actor_rollout_ref.rollout.temperature": rollout["temperature"],
        "actor_rollout_ref.rollout.top_p": rollout["top_p"],
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": rollout[
            "tensor_parallel_size"
        ],
        "actor_rollout_ref.rollout.gpu_memory_utilization": rollout[
            "gpu_memory_utilization"
        ],
        # Qwen3-VL advertises a 262k architectural context.  Without an
        # explicit experiment bound vLLM rejects max_num_batched_tokens=32768
        # when chunked prefill is disabled.  The Policy contract itself is
        # prompt 8192 + response 20480 and retains 4096 tokens of runtime
        # headroom while remaining equal to max_num_batched_tokens.
        "actor_rollout_ref.rollout.max_model_len": rollout["max_num_batched_tokens"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": rollout[
            "max_num_batched_tokens"
        ],
        # vLLM 0.12's bundled FA2 PTX is not driver-compatible on the accepted
        # B200 hosts.  Keep language attention on its driver-portable Triton
        # path (environment below) and select the stock per-ViT SDPA override
        # independently; this is the same accepted split used elsewhere in the
        # repository for native Qwen3-VL pixels.
        "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend": (
            DEEPEYES_NATIVE_VLLM_MM_ENCODER_ATTN_BACKEND
        ),
        # One source image plus at most six successful Crop observations.  The
        # protocol rejects video, so do not make vLLM profile or admit it.
        "actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt": {
            "image": 1 + _nested(payload, "protocol.max_active_perception"),
            "video": 0,
        },
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enable_prefix_caching": False,
        # e003's colocated full-model lifecycle sleeps vLLM after generation
        # before the actor returns to GPU, then wakes weights/KV around the
        # complete naive sync. Disabling this leaves vLLM's 0.8 allocation
        # resident during full-model backward.
        "actor_rollout_ref.rollout.free_cache_engine": rollout["free_cache_engine"],
        "actor_rollout_ref.rollout.enable_sleep_mode": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": rollout[
            "rollout_logprob_micro_batch_size_per_gpu"
        ],
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.multi_turn.tool_config_path": _nested(
            payload, "framework.tool_config_path"
        ),
        "actor_rollout_ref.rollout.multi_turn.function_tool_path": None,
        "actor_rollout_ref.rollout.multi_turn.format": "hermes",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns": rollout[
            "max_user_turns"
        ],
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": rollout[
            "max_assistant_turns"
        ],
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls": 1,
        "actor_rollout_ref.rollout.agent.num_workers": 4,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "single_turn_agent",
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class": (
            PRL13_AGENT_LOOP_MANAGER_FQN
        ),
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": _nested(
            payload, "framework.agent_loop_config_path"
        ),
        # ``RolloutConfig`` is a strict dataclass in pinned e003.  Project-owned
        # protocol/reward bindings therefore live under its explicit extension
        # field instead of being added as unknown rollout fields.
        "actor_rollout_ref.rollout.custom": {
            "schema_version": DEEPEYES_NATIVE_LAUNCH_SCHEMA,
            "protocol": {
                "single_response_max_tokens": rollout[
                    "single_response_max_tokens"
                ],
                "coordinate_mapper": _nested(
                    payload, "protocol.coordinate_mapper"
                ),
            },
        },
        # Experimental RewardLoopManager; one worker creates all per-row tasks,
        # while the manager-owned semaphore is the global request bound.
        "reward.num_workers": reward["reward_num_workers"],
        "reward.custom_reward_function.path": None,
        "reward.reward_manager.source": "importlib",
        "reward.reward_manager.name": reward["reward_manager_class_name"],
        "reward.reward_manager.module.path": reward["reward_manager_module_path"],
        "reward.reward_manager.module.name": reward["reward_manager_class_name"],
        "reward.deepeyes_official.judge_service_config_path": reward[
            "judge_service_config_path"
        ],
        "reward.deepeyes_official.judge_service_config_sha256": reward[
            "judge_service_config_sha256"
        ],
        "algorithm.adv_estimator": "grpo",
        "algorithm.gamma": 1.0,
        "algorithm.lam": 1.0,
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "algorithm.kl_ctrl.type": "fixed",
        "algorithm.kl_ctrl.kl_coef": 0.0,
    }
    return overrides


def _smoke_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    if target_step not in DEEPEYES_NATIVE_SMOKE_HORIZONS:
        raise ValueError("PRL13 smoke target must be step 1 or 2")
    overrides = _formal_overrides(contract, target_step=target_step)
    output_root = Path(_nested(contract.payload, "output.root")) / "smoke"
    overrides.update(
        {
            # The four-row smoke dataset is exactly one batch.  At step 1 its
            # dataloader is therefore at an epoch boundary; target_step epochs
            # let upstream resume at epoch 1 and execute only absolute step 2.
            "trainer.total_epochs": target_step,
            "trainer.experiment_name": contract.run_id + "-SMOKE",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            "trainer.max_actor_ckpt_to_keep": 1,
            "data.train_files": [str(DEEPEYES_SMOKE_SENTINEL)],
            "data.val_files": [str(DEEPEYES_SMOKE_SENTINEL)],
            "data.train_batch_size": DEEPEYES_NATIVE_SMOKE_PROMPTS,
            "data.gen_batch_size": DEEPEYES_NATIVE_SMOKE_PROMPTS,
            "data.max_response_length": 4096,
            "actor_rollout_ref.actor.ppo_mini_batch_size": (
                DEEPEYES_NATIVE_SMOKE_PROMPTS
            ),
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
            # The smoke produces 4 prompts * n=2 trajectories globally, hence
            # only two trajectories per FSDP rank.  The formal per-rank
            # log-prob micro-batch of eight is valid for BS256*n16 but cannot
            # divide this deliberately tiny canary.  Keep the scientific
            # formal shape unchanged and use the largest valid smoke value.
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.rollout.n": DEEPEYES_NATIVE_SMOKE_N,
        }
    )
    return overrides


def _stress_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    """Use four deterministic rows while retaining the formal memory shape."""

    if target_step not in DEEPEYES_NATIVE_STRESS_HORIZONS:
        raise ValueError("PRL13 stress target must be step 1 or 2")
    overrides = _formal_overrides(contract, target_step=target_step)
    output_root = Path(_nested(contract.payload, "output.root")) / "stress"
    overrides.update(
        {
            # One four-prompt batch per epoch makes 1 -> 2 an absolute resume
            # sequence.  Step 2 therefore generates after the step-1 actor
            # update has been synchronized, whether run continuously or via
            # auto-resume from the saved step-1 checkpoint.
            "trainer.total_epochs": target_step,
            "trainer.experiment_name": contract.run_id + "-STRESS",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            "trainer.max_actor_ckpt_to_keep": len(DEEPEYES_NATIVE_STRESS_HORIZONS),
            # This deliberately reuses the fixed source-covering smoke rows.
            # It does not claim to select the pool's largest visual-token rows.
            "data.train_files": [str(DEEPEYES_SMOKE_SENTINEL)],
            "data.val_files": [str(DEEPEYES_SMOKE_SENTINEL)],
            "data.train_batch_size": DEEPEYES_NATIVE_STRESS_PROMPTS,
            "data.gen_batch_size": DEEPEYES_NATIVE_STRESS_PROMPTS,
            "actor_rollout_ref.actor.ppo_mini_batch_size": (
                DEEPEYES_NATIVE_STRESS_PROMPTS
            ),
            "actor_rollout_ref.rollout.n": DEEPEYES_NATIVE_STRESS_N,
        }
    )
    return overrides


def _resident_stress_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    """A/B the same n16 stress with model and optimizer resident on B200."""

    overrides = _stress_overrides(contract, target_step=target_step)
    output_root = Path(_nested(contract.payload, "output.root")) / "stress-resident"
    overrides.update(
        {
            "trainer.experiment_name": contract.run_id + "-STRESS-RESIDENT",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            "actor_rollout_ref.actor.fsdp_config.param_offload": False,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
            "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
            "actor_rollout_ref.ref.fsdp_config.param_offload": False,
            "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
            # 64 trajectories are 16 per rank.  Start with conservative 8-way
            # actor batches and one-pass 16-way log-prob batches; both are far
            # below the available 183 GiB but materially larger than PRL13.
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 16,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 16,
        }
    )
    return overrides


def _resident_fast_stress_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    """Use B200 memory to remove recompute and FSDP reshard overhead."""

    overrides = _resident_stress_overrides(contract, target_step=target_step)
    output_root = (
        Path(_nested(contract.payload, "output.root")) / "stress-resident-fast"
    )
    overrides.update(
        {
            "trainer.experiment_name": contract.run_id + "-STRESS-RESIDENT-FAST",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            "actor_rollout_ref.model.enable_gradient_checkpointing": False,
            "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": False,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 16,
        }
    )
    return overrides


def _resident_flex_stress_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    """A/B packed FlexAttention against the proven-safe resident stress."""

    overrides = _resident_stress_overrides(contract, target_step=target_step)
    output_root = (
        Path(_nested(contract.payload, "output.root")) / "stress-resident-flex"
    )
    overrides.update(
        {
            "trainer.experiment_name": contract.run_id
            + "-STRESS-RESIDENT-FLEX",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            # SDPA materializes the packed document mask and computes over the
            # concatenated sequence.  Torch FlexAttention consumes the same
            # position-reset contract as a block-sparse BlockMask, preserving
            # sample isolation without paying cross-document quadratic cost.
            "actor_rollout_ref.model.override_config.text_config": {
                "_attn_implementation_internal": "flex_attention",
            },
            "actor_rollout_ref.model.override_config.vision_config": {
                "_attn_implementation_internal": "sdpa",
            },
        }
    )
    return overrides


def _resident_wide_flex_stress_overrides(
    contract: DeepEyesNativeRunContract, *, target_step: int
) -> dict[str, object]:
    """Measure stable-step throughput with one actor micro-batch per rank."""

    overrides = _resident_flex_stress_overrides(contract, target_step=target_step)
    output_root = (
        Path(_nested(contract.payload, "output.root"))
        / "stress-resident-wide-flex"
    )
    overrides.update(
        {
            "trainer.experiment_name": contract.run_id
            + "-STRESS-RESIDENT-WIDE-FLEX",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            # The four-prompt n=16 canary has exactly 16 trajectories per
            # rank.  Gradient checkpointing remains enabled, unlike the OOM'd
            # fast profile, so process all local trajectories in one batch.
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 16,
        }
    )
    return overrides


@dataclass(frozen=True, slots=True)
class DeepEyesNativeVerlLaunchPlan:
    contract: DeepEyesNativeRunContract
    mode: LaunchMode
    target_step: int
    overrides: Mapping[str, object]
    environment: Mapping[str, str]
    schema_version: str = DEEPEYES_NATIVE_LAUNCH_SCHEMA
    verl_commit: str = DEEPEYES_NATIVE_VERL_COMMIT

    def __post_init__(self) -> None:
        if self.schema_version != DEEPEYES_NATIVE_LAUNCH_SCHEMA:
            raise ValueError("PRL13 launch schema differs")
        if self.verl_commit != DEEPEYES_NATIVE_VERL_COMMIT:
            raise ValueError("PRL13 launch requires pinned veRL e003")
        if self.mode not in {
            "formal",
            "smoke",
            "stress",
            "resident-stress",
            "resident-fast-stress",
            "resident-flex-stress",
            "resident-wide-flex-stress",
        }:
            raise ValueError("PRL13 launch mode differs")
        if self.mode == "formal" and self.target_step not in (
            DEEPEYES_NATIVE_FORMAL_HORIZONS
        ):
            raise ValueError("formal target is not a PRL13 horizon gate")
        if (
            self.mode == "smoke"
            and self.target_step not in DEEPEYES_NATIVE_SMOKE_HORIZONS
        ):
            raise ValueError("smoke target must be step 1 or 2")
        if (
            self.mode
            in {
                "stress",
                "resident-stress",
                "resident-fast-stress",
                "resident-flex-stress",
            }
            and self.target_step not in DEEPEYES_NATIVE_STRESS_HORIZONS
        ):
            raise ValueError("stress target must be step 1 or 2")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        self._assert_native_path()
        self._assert_static_micro_batch_shapes()

    def _assert_native_path(self) -> None:
        values = self.overrides
        if values.get("trainer.use_v1") is not False:
            raise ValueError("PRL13 must use the pinned v0 TaskRunner")
        if values.get("actor_rollout_ref.actor.strategy") != "fsdp2":
            raise ValueError("PRL13 actor is not FSDP2")
        if values.get("actor_rollout_ref.model.lora_rank") != 0:
            raise ValueError("PRL13 unexpectedly enabled LoRA")
        if (
            values.get("actor_rollout_ref.model.external_lib")
            != NATIVE_DEEPEYES_POLICY_LOSS_MODULE
        ):
            raise ValueError("PRL13 actor-loss external module identity differs")
        if (
            values.get(
                "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
            )
            != PRL13_AGENT_LOOP_MANAGER_FQN
        ):
            raise ValueError("PRL13 heterogeneous manager identity differs")
        if "actor_rollout_ref.rollout.checkpoint_manager_class" in values:
            raise ValueError("PRL13 must retain upstream checkpoint manager")
        serialized = json.dumps(values, default=str).casefold()
        for forbidden in ("exact_replay", "losslessreplay", "rp66"):
            if forbidden in serialized:
                raise ValueError(f"PRL13 launcher contains legacy path {forbidden}")

    def _assert_static_micro_batch_shapes(self) -> None:
        """Reject fixed-batch shapes that pinned veRL cannot partition.

        Pinned e003 validates prompt-level PPO sizes, but its no-padding FSDP
        engine later partitions the expanded ``prompts * rollout.n`` batch on
        every rank.  Catch those deterministic divisibility failures before
        Ray, model loading, rollout, or paid judge calls.
        """

        values = self.overrides
        world_size = int(values["trainer.n_gpus_per_node"]) * int(
            values["trainer.nnodes"]
        )
        prompt_batch = int(values["data.train_batch_size"])
        rollout_n = int(values["actor_rollout_ref.rollout.n"])
        trajectory_batch = prompt_batch * rollout_n
        if trajectory_batch % world_size:
            raise ValueError(
                "PRL13 expanded trajectory batch is not divisible by world size"
            )
        local_trajectories = trajectory_batch // world_size

        log_prob_micro = int(
            values[
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
            ]
        )
        if local_trajectories % log_prob_micro:
            raise ValueError(
                "PRL13 local trajectory batch is not divisible by the "
                "log-prob micro-batch"
            )

        ppo_mini_prompts = int(
            values["actor_rollout_ref.actor.ppo_mini_batch_size"]
        )
        ppo_mini_trajectories = ppo_mini_prompts * rollout_n
        if ppo_mini_trajectories % world_size:
            raise ValueError(
                "PRL13 expanded PPO mini-batch is not divisible by world size"
            )
        local_ppo_mini = ppo_mini_trajectories // world_size
        actor_micro = int(
            values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"]
        )
        if local_ppo_mini % actor_micro:
            raise ValueError(
                "PRL13 local PPO mini-batch is not divisible by the actor "
                "micro-batch"
            )

    @property
    def checkpoint_root(self) -> Path:
        return Path(str(self.overrides["trainer.default_local_dir"]))

    def hydra_override_args(self) -> tuple[str, ...]:
        return tuple(
            f"++{path}={_hydra_literal(value)}"
            for path, value in self.overrides.items()
        )

    def latest_checkpoint_step(self) -> int | None:
        tracker = self.checkpoint_root / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return None
        if tracker.is_symlink() or not tracker.is_file():
            raise RuntimeError("PRL13 checkpoint tracker is not a regular file")
        text = tracker.read_text(encoding="utf-8").strip()
        try:
            step = int(text)
        except ValueError as error:
            raise RuntimeError("PRL13 checkpoint tracker is malformed") from error
        if step < 0 or not (self.checkpoint_root / f"global_step_{step}").is_dir():
            raise RuntimeError("PRL13 latest checkpoint directory is incomplete")
        return step

    def horizon_already_satisfied(self) -> bool:
        latest = self.latest_checkpoint_step()
        return latest is not None and latest >= self.target_step

    def assert_target_checkpoint_complete(self) -> Path:
        root = self.checkpoint_root / f"global_step_{self.target_step}"
        actor = root / "actor"
        tracker = self.checkpoint_root / "latest_checkpointed_iteration.txt"
        if (
            not root.is_dir()
            or not actor.is_dir()
            or not (root / "data.pt").is_file()
            or not tracker.is_file()
            or tracker.read_text(encoding="utf-8").strip() != str(self.target_step)
        ):
            raise RuntimeError("PRL13 target checkpoint is incomplete")
        if not any(actor.rglob("*")):
            raise RuntimeError("PRL13 target actor checkpoint is empty")
        return root

    def as_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "contract_sha256": self.contract.identity_sha256,
            "run_id": self.contract.run_id,
            "mode": self.mode,
            "target_step": self.target_step,
            "verl_commit": self.verl_commit,
            "overrides": dict(self.overrides),
            "environment": dict(self.environment),
            "required_secret_environment": ["OPENROUTER_API_KEY"],
        }
        if self.mode in {
            "stress",
            "resident-stress",
            "resident-fast-stress",
            "resident-flex-stress",
            "resident-wide-flex-stress",
        }:
            record["canary_scope"] = DEEPEYES_NATIVE_STRESS_SCOPE
        if self.mode in {
            "smoke",
            "stress",
            "resident-stress",
            "resident-fast-stress",
            "resident-flex-stress",
            "resident-wide-flex-stress",
        } and self.target_step == 2:
            record["step_2_rollout_contract"] = (
                "rollout follows the synchronized step-1 actor update"
            )
        return record


def build_deepeyes_native_verl_launch_plan(
    contract: DeepEyesNativeRunContract,
    *,
    mode: LaunchMode = "formal",
    target_step: int = 1,
) -> DeepEyesNativeVerlLaunchPlan:
    if mode == "formal":
        overrides = _formal_overrides(contract, target_step=target_step)
    elif mode == "smoke":
        overrides = _smoke_overrides(contract, target_step=target_step)
    elif mode == "stress":
        overrides = _stress_overrides(contract, target_step=target_step)
    elif mode == "resident-stress":
        overrides = _resident_stress_overrides(contract, target_step=target_step)
    elif mode == "resident-fast-stress":
        overrides = _resident_fast_stress_overrides(
            contract, target_step=target_step
        )
    elif mode == "resident-flex-stress":
        overrides = _resident_flex_stress_overrides(
            contract, target_step=target_step
        )
    elif mode == "resident-wide-flex-stress":
        overrides = _resident_wide_flex_stress_overrides(
            contract, target_step=target_step
        )
    else:
        raise ValueError(f"unsupported PRL13 launch mode: {mode!r}")
    return DeepEyesNativeVerlLaunchPlan(
        contract=contract,
        mode=mode,
        target_step=target_step,
        overrides=overrides,
        environment={
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(value)
                for value in _nested(contract.payload, "distributed.physical_gpu_ids")
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
            "VLLM_ATTENTION_BACKEND": DEEPEYES_NATIVE_VLLM_ATTENTION_BACKEND,
            "VERL_FULL_DETERMINISM": "0",
            "VLLM_BATCH_INVARIANT": "0",
        },
    )


def apply_launch_environment(plan: DeepEyesNativeVerlLaunchPlan) -> None:
    for name, value in plan.environment.items():
        os.environ[name] = value
    text_override = plan.overrides.get(
        "actor_rollout_ref.model.override_config.text_config"
    )
    if isinstance(text_override, Mapping) and text_override.get(
        "_attn_implementation_internal"
    ) == "flex_attention":
        python_include = _PYTHON312_DEV_ROOT / "python3.12"
        if not (python_include / "Python.h").is_file():
            raise RuntimeError("PRL13 FlexAttention requires local Python 3.12 headers")
        entries = [str(python_include), str(_PYTHON312_DEV_ROOT)]
        inherited = os.environ.get("CPATH")
        if inherited:
            entries.append(inherited)
        os.environ["CPATH"] = os.pathsep.join(entries)


__all__ = [
    "DEEPEYES_NATIVE_FORMAL_HORIZONS",
    "DEEPEYES_NATIVE_LAUNCH_SCHEMA",
    "DEEPEYES_NATIVE_SMOKE_HORIZONS",
    "DEEPEYES_NATIVE_SMOKE_N",
    "DEEPEYES_NATIVE_SMOKE_PROMPTS",
    "DEEPEYES_NATIVE_STRESS_HORIZONS",
    "DEEPEYES_NATIVE_STRESS_N",
    "DEEPEYES_NATIVE_STRESS_PROMPTS",
    "DEEPEYES_NATIVE_STRESS_SCOPE",
    "DEEPEYES_NATIVE_VERL_COMMIT",
    "DEEPEYES_NATIVE_VLLM_ATTENTION_BACKEND",
    "DEEPEYES_NATIVE_VLLM_MM_ENCODER_ATTN_BACKEND",
    "DeepEyesNativeVerlLaunchPlan",
    "apply_launch_environment",
    "build_deepeyes_native_verl_launch_plan",
]
