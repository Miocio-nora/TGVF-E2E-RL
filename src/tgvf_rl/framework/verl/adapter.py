"""Narrow composition root for the pinned public veRL integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import os
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from tgvf_rl.checkpoint import CheckpointCoordinator
from tgvf_rl.framework.vllm import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_ATTENTION_BACKEND,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.policy.config import PolicyPilotV1Config

from .checkpoint_bridge import (
    SDPOTeacherCheckpointContributor,
    StatefulTeacher,
    register_sdpo_teacher_checkpoint,
    validate_fsdp2_checkpoint_config,
)
from .compatibility import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    TORCH211_CANDIDATE_VERL_COMMIT,
    VerlPublicAPI,
    VerlRuntimeRequirements,
    load_verl_public_api,
)
from .data_bridge import (
    DataProtoIntegrityView,
    build_verl_data_proto,
    validate_data_proto_integrity,
)
from .objective_bridge import (
    ProjectPolicyLoss,
    register_project_policy_loss,
    validate_policy_pilot_v1_verl_grpo_parity,
)
from .rollout_bridge import (
    LosslessAgentLoopManager,
    RolloutBridgeRecord,
    build_agent_loop_output,
)


LOSSLESS_AGENT_LOOP_MANAGER_FQN = (
    "tgvf_rl.framework.verl.rollout_bridge.LosslessAgentLoopManager"
)
LOSSLESS_TRANSFER_QUEUE_AGENT_LOOP_MANAGER_FQN = (
    "tgvf_rl.framework.verl.rollout_bridge.LosslessTransferQueueAgentLoopManager"
)
TGVF_VLLM_PLUGIN_NAME = "tgvf_qwen3_precomputed"


@dataclass(frozen=True, slots=True)
class VerlAdapterConfig:
    """Accepted configuration choices exposed as plain veRL override paths."""

    runtime: VerlRuntimeRequirements = field(default_factory=VerlRuntimeRequirements)
    agent_loop_manager_fqn: str | None = None
    max_tool_calls: int | None = None
    policy_pilot: PolicyPilotV1Config | None = None

    def __post_init__(self) -> None:
        expected_manager = {
            SPIKE_CANDIDATE_VERL_COMMIT: LOSSLESS_AGENT_LOOP_MANAGER_FQN,
            TORCH211_CANDIDATE_VERL_COMMIT: (
                LOSSLESS_TRANSFER_QUEUE_AGENT_LOOP_MANAGER_FQN
            ),
        }[self.runtime.verl_commit]
        if self.agent_loop_manager_fqn is None:
            object.__setattr__(self, "agent_loop_manager_fqn", expected_manager)
        elif self.agent_loop_manager_fqn != expected_manager:
            raise ValueError(
                "the selected veRL identity requires its exact lossless agent-loop "
                "transport manager"
            )
        if self.max_tool_calls is not None and (
            type(self.max_tool_calls) is not int or self.max_tool_calls <= 1
        ):
            raise ValueError("an explicit multi-call cap greater than one is required")
        if self.policy_pilot is not None:
            if not isinstance(self.policy_pilot, PolicyPilotV1Config):
                raise TypeError("policy_pilot must be PolicyPilotV1Config")
            if self.runtime.verl_commit != SPIKE_CANDIDATE_VERL_COMMIT:
                raise ValueError(
                    "Policy Pilot v1 is bound to the accepted e003 veRL control stack"
                )
            pilot_calls = self.policy_pilot.max_tgvf_call_attempts
            if self.max_tool_calls is None:
                object.__setattr__(self, "max_tool_calls", pilot_calls)
            elif self.max_tool_calls != pilot_calls:
                raise ValueError(
                    "Policy Pilot v1 requires exactly four TGVF call attempts"
                )
        validate_fsdp2_checkpoint_config(self.runtime.fsdp2)

    def public_config_overrides(self) -> Mapping[str, object]:
        if self.max_tool_calls is None:
            raise ValueError(
                "max_tool_calls is an unset research choice; live rollout must set it explicitly"
            )
        fsdp = self.runtime.fsdp2
        values = {
            "actor_rollout_ref.rollout.name": self.runtime.rollout_backend,
            "actor_rollout_ref.rollout.calculate_log_probs": True,
            "actor_rollout_ref.rollout.logprobs_mode": self.runtime.logprobs_mode,
            # vLLM 0.12 batch-invariant mode rejects the accepted TRITON_ATTN
            # backend.  Multi-turn tool trajectories are not bitwise
            # batch-invariant in upstream veRL in any case.  The project owns
            # a content-addressed seed per turn and records actual behavior
            # logprobs; actor/reference replay remains fully deterministic.
            "actor_rollout_ref.rollout.full_determinism": False,
            "actor_rollout_ref.rollout.enable_prefix_caching": False,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_mm_embeds": True,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb": 0,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend": (
                TGVF_VLLM_MM_ENCODER_ATTN_BACKEND
            ),
            "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides": {
                "architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]
            },
            "actor_rollout_ref.rollout.limit_images": 1 + self.max_tool_calls,
            "actor_rollout_ref.rollout.agent.agent_loop_manager_class": self.agent_loop_manager_fqn,
            "actor_rollout_ref.actor.strategy": fsdp.actor_strategy,
            "actor_rollout_ref.ref.strategy": fsdp.reference_strategy,
            "actor_rollout_ref.ref.fsdp_config.fsdp_size": fsdp.fsdp_size,
            "actor_rollout_ref.ref.fsdp_config.full_determinism": fsdp.full_determinism,
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": fsdp.fsdp_size,
            "actor_rollout_ref.actor.fsdp_config.full_determinism": fsdp.full_determinism,
            "actor_rollout_ref.actor.checkpoint.async_save": fsdp.checkpoint_async_save,
            "actor_rollout_ref.actor.checkpoint.strict": fsdp.checkpoint_strict,
            "actor_rollout_ref.actor.checkpoint.save_contents": fsdp.checkpoint_save_contents,
            "actor_rollout_ref.actor.checkpoint.load_contents": fsdp.checkpoint_load_contents,
            "trainer.use_v1": (
                self.runtime.verl_commit == TORCH211_CANDIDATE_VERL_COMMIT
            ),
            "trainer.v1.trainer_mode": self.runtime.trainer_mode,
        }
        if self.policy_pilot is not None:
            pilot = self.policy_pilot
            sampling = pilot.sampling
            lora = pilot.lora
            grpo = pilot.grpo
            values.update(
                {
                    "actor_rollout_ref.model.path": pilot.model_path,
                    "actor_rollout_ref.model.lora_rank": lora.rank,
                    "actor_rollout_ref.model.lora_alpha": lora.alpha,
                    "actor_rollout_ref.model.target_modules": lora.target_modules,
                    "actor_rollout_ref.model.exclude_modules": lora.exclude_modules,
                    "actor_rollout_ref.model.external_lib": (
                        grpo.verl_external_loss_module
                    ),
                    "actor_rollout_ref.actor.freeze_vision_tower": True,
                    "actor_rollout_ref.actor.optim.lr": lora.initial_learning_rate,
                    "actor_rollout_ref.actor.optim.clip_grad": (
                        grpo.maximum_gradient_norm
                    ),
                    "actor_rollout_ref.actor.ppo_epochs": grpo.update_epochs,
                    "actor_rollout_ref.actor.clip_ratio": grpo.clip_epsilon_low,
                    "actor_rollout_ref.actor.clip_ratio_low": grpo.clip_epsilon_low,
                    "actor_rollout_ref.actor.clip_ratio_high": grpo.clip_epsilon_high,
                    "actor_rollout_ref.actor.clip_ratio_c": grpo.dual_clip,
                    "actor_rollout_ref.actor.loss_agg_mode": grpo.loss_aggregation,
                    "actor_rollout_ref.actor.policy_loss.loss_mode": (
                        grpo.verl_execution_loss_mode
                    ),
                    "actor_rollout_ref.actor.entropy_coeff": (grpo.entropy_coefficient),
                    "actor_rollout_ref.actor.calculate_entropy": False,
                    "actor_rollout_ref.actor.use_kl_loss": False,
                    "actor_rollout_ref.actor.kl_loss_coef": (grpo.kl_loss_coefficient),
                    "actor_rollout_ref.rollout.n": (sampling.trajectories_per_prompt),
                    "actor_rollout_ref.rollout.temperature": sampling.temperature,
                    "actor_rollout_ref.rollout.top_p": sampling.top_p,
                    "actor_rollout_ref.rollout.top_k": sampling.top_k,
                    "actor_rollout_ref.rollout.repetition_penalty": (
                        sampling.repetition_penalty
                    ),
                    "actor_rollout_ref.rollout.do_sample": sampling.do_sample,
                    "actor_rollout_ref.rollout.response_length": (
                        sampling.max_response_length
                    ),
                    "actor_rollout_ref.rollout.over_sample_rate": (
                        grpo.rollout_over_sample_rate
                    ),
                    "actor_rollout_ref.rollout.multi_turn.enable": True,
                    "data.max_response_length": sampling.max_response_length,
                    "data.mm_processor_kwargs.max_pixels": pilot.image_max_pixels,
                    "data.filter_overlong_prompts": False,
                    "data.truncation": "error",
                    "algorithm.adv_estimator": grpo.advantage_estimator,
                    "algorithm.norm_adv_by_std_in_grpo": (
                        grpo.sample_standard_deviation
                    ),
                    "algorithm.use_kl_in_reward": False,
                    "algorithm.kl_ctrl.kl_coef": grpo.kl_reward_coefficient,
                    "algorithm.filter_groups": {
                        "enable": False,
                        "metric": None,
                        "max_num_gen_batches": 0,
                    },
                    "algorithm.rollout_correction.rollout_is": (
                        grpo.rollout_importance_sampling
                    ),
                    "algorithm.rollout_correction.rollout_rs": (
                        grpo.rollout_rejection_sampling
                    ),
                    "algorithm.rollout_correction.bypass_mode": (
                        grpo.rollout_correction_bypass_mode
                    ),
                    "algorithm.rollout_correction.loss_type": (
                        grpo.rollout_correction_loss_type
                    ),
                    "algorithm.rollout_correction.rollout_is_batch_normalize": (
                        grpo.rollout_is_batch_normalize
                    ),
                }
            )
        if self.runtime.verl_commit == TORCH211_CANDIDATE_VERL_COMMIT:
            values.update(
                {
                    "actor_rollout_ref.rollout.free_cache_engine": False,
                    "actor_rollout_ref.rollout.enable_sleep_mode": False,
                    "actor_rollout_ref.rollout.checkpoint_engine.backend": "naive",
                }
            )
        return MappingProxyType(values)

    @staticmethod
    def required_environment() -> Mapping[str, str]:
        return MappingProxyType(
            {
                "VLLM_PLUGINS": TGVF_VLLM_PLUGIN_NAME,
                "VLLM_ATTENTION_BACKEND": TGVF_VLLM_ATTENTION_BACKEND,
                "VERL_FULL_DETERMINISM": "0",
                "VLLM_BATCH_INVARIANT": "0",
            }
        )

    @staticmethod
    def validate_runtime_environment(
        environ: Mapping[str, str] | None = None,
    ) -> None:
        values = os.environ if environ is None else environ
        if values.get("VLLM_PLUGINS") != TGVF_VLLM_PLUGIN_NAME:
            raise ValueError(
                "VLLM_PLUGINS must select only the repo-owned precomputed-Qwen3 plugin"
            )
        if values.get("VLLM_ATTENTION_BACKEND") != TGVF_VLLM_ATTENTION_BACKEND:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND must be TRITON_ATTN for the accepted "
                "driver-portable path"
            )
        if values.get("VERL_FULL_DETERMINISM") != "0":
            raise ValueError(
                "rollout-level VERL_FULL_DETERMINISM must be disabled for the "
                "accepted multi-turn TRITON_ATTN path"
            )
        if values.get("VLLM_BATCH_INVARIANT") != "0":
            raise ValueError(
                "VLLM_BATCH_INVARIANT must be disabled with TRITON_ATTN"
            )


class VerlAdapter:
    """Calls public veRL objects while keeping all algorithm state project-owned."""

    def __init__(
        self,
        config: VerlAdapterConfig | None = None,
        *,
        public_api: VerlPublicAPI | None = None,
    ) -> None:
        self.config = config or VerlAdapterConfig()
        self._public_api = public_api

    @property
    def public_api(self) -> VerlPublicAPI:
        if self._public_api is None:
            # Validate before veRL spawns Ray/vLLM child processes: registering
            # only in this process cannot cover the vLLM core and workers.
            self.config.public_config_overrides()
            self.config.validate_runtime_environment()
            public_api = load_verl_public_api(
                expected_commit=self.config.runtime.verl_commit
            )
            if self.config.policy_pilot is not None:
                import_module(self.config.policy_pilot.grpo.verl_external_loss_module)
                validate_policy_pilot_v1_verl_grpo_parity(public_api)
            self._public_api = public_api
        return self._public_api

    def build_agent_loop_output(
        self, record: RolloutBridgeRecord, *, metrics: object
    ) -> Any:
        return build_agent_loop_output(
            record,
            metrics=metrics,
            agent_loop_output_cls=self.public_api.agent_loop_output,
        )

    def build_data_proto(self, records: Iterable[RolloutBridgeRecord]) -> Any:
        if self.config.policy_pilot is not None:
            raise RuntimeError(
                "Policy Pilot DataProto construction requires a retained "
                "DataProtoPayload attached to PolicyBatchLifecycle; use "
                "build_data_proto_payload/to_verl_data_proto explicitly"
            )
        return build_verl_data_proto(records, data_proto_cls=self.public_api.data_proto)

    @staticmethod
    def validate_data_proto(data: object) -> DataProtoIntegrityView:
        return validate_data_proto_integrity(data)

    def register_policy_loss(
        self,
        name: str,
        project_loss: ProjectPolicyLoss,
    ) -> Any:
        return register_project_policy_loss(
            name,
            project_loss,
            registrar=self.public_api.register_policy_loss,
        )

    @staticmethod
    def register_sdpo_teacher(
        coordinator: CheckpointCoordinator,
        teacher: StatefulTeacher,
    ) -> SDPOTeacherCheckpointContributor:
        return register_sdpo_teacher_checkpoint(coordinator, teacher)


__all__ = [
    "LOSSLESS_AGENT_LOOP_MANAGER_FQN",
    "LOSSLESS_TRANSFER_QUEUE_AGENT_LOOP_MANAGER_FQN",
    "LosslessAgentLoopManager",
    "TGVF_VLLM_PLUGIN_NAME",
    "VerlAdapter",
    "VerlAdapterConfig",
]
