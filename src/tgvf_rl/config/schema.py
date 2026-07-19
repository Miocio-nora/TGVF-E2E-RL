"""Explicit configuration identities; research decisions have no hidden defaults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tgvf_rl.conditioning.base import TargetConditioningConfig
from tgvf_rl.contracts.data import DataManifestIdentity
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity


class RunGate(str, Enum):
    SKELETON = "skeleton"
    SYNTHETIC_ROLLOUT = "synthetic_rollout"
    GRPO_SMOKE = "grpo_smoke"
    SDPO_SMOKE = "sdpo_smoke"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class StackConfig:
    verl_commit: str
    rollout_backend: str
    sharding_strategy: str
    full_determinism: bool
    physical_gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.rollout_backend.lower() != "vllm":
            raise ValueError("vLLM is the only rollout backend")
        if self.sharding_strategy.lower() != "fsdp2":
            raise ValueError("FSDP2 is required")
        if not self.full_determinism:
            raise ValueError("full determinism is required for smoke")
        if self.physical_gpu_ids and any(
            device not in {2, 3} for device in self.physical_gpu_ids
        ):
            raise ValueError("only physical GPUs 2 and 3 are authorized")


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    gate: RunGate
    stack: StackConfig
    primary_model: ModelIdentity
    secondary_model: ModelIdentity | None
    target_conditioning: TargetConditioningConfig
    representation_artifact: ArtifactIdentity | None
    data_manifest: DataManifestIdentity | None
    prompt_identity: ArtifactIdentity | None
    reward_identity: ArtifactIdentity | None
    objective_identity: ArtifactIdentity | None
    max_tool_calls: int | None
