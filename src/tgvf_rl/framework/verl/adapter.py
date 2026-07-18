"""Narrow composition root for the pinned public veRL integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from tgvf_rl.checkpoint import CheckpointCoordinator
from tgvf_rl.framework.vllm import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_ATTENTION_BACKEND,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)

from .checkpoint_bridge import (
    SDPOTeacherCheckpointContributor,
    StatefulTeacher,
    register_sdpo_teacher_checkpoint,
    validate_fsdp2_checkpoint_config,
)
from .compatibility import VerlPublicAPI, VerlRuntimeRequirements, load_verl_public_api
from .data_bridge import (
    DataProtoIntegrityView,
    build_verl_data_proto,
    validate_data_proto_integrity,
)
from .objective_bridge import ProjectPolicyLoss, register_project_policy_loss
from .rollout_bridge import (
    LosslessAgentLoopManager,
    RolloutBridgeRecord,
    build_agent_loop_output,
)


LOSSLESS_AGENT_LOOP_MANAGER_FQN = (
    "tgvf_rl.framework.verl.rollout_bridge.LosslessAgentLoopManager"
)
TGVF_VLLM_PLUGIN_NAME = "tgvf_qwen3_precomputed"


@dataclass(frozen=True, slots=True)
class VerlAdapterConfig:
    """Accepted configuration choices exposed as plain veRL override paths."""

    runtime: VerlRuntimeRequirements = field(default_factory=VerlRuntimeRequirements)
    agent_loop_manager_fqn: str = LOSSLESS_AGENT_LOOP_MANAGER_FQN
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        if self.agent_loop_manager_fqn != LOSSLESS_AGENT_LOOP_MANAGER_FQN:
            raise ValueError("the lossless public custom AgentLoopManager is required")
        if self.max_tool_calls is not None and (
            type(self.max_tool_calls) is not int or self.max_tool_calls <= 1
        ):
            raise ValueError("an explicit multi-call cap greater than one is required")
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
            "actor_rollout_ref.rollout.full_determinism": fsdp.full_determinism,
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
            "actor_rollout_ref.model.lora.dropout": fsdp.adapter_dropout,
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
            "trainer.v1.trainer_mode": self.runtime.trainer_mode,
        }
        return MappingProxyType(values)

    @staticmethod
    def required_environment() -> Mapping[str, str]:
        return MappingProxyType(
            {
                "VLLM_PLUGINS": TGVF_VLLM_PLUGIN_NAME,
                "VLLM_ATTENTION_BACKEND": TGVF_VLLM_ATTENTION_BACKEND,
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
            self._public_api = load_verl_public_api()
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
    "LosslessAgentLoopManager",
    "TGVF_VLLM_PLUGIN_NAME",
    "VerlAdapter",
    "VerlAdapterConfig",
]
