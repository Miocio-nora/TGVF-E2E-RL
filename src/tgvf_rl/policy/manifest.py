"""Content-addressed launch identity for Policy Pilot v1.

The accepted constants live in :mod:`tgvf_rl.policy.config`.  This module
binds the run-specific inputs that ``PROJECT_TASK.md`` section 0.8 deliberately
leaves open.  Open inputs have no defaults: constructing a manifest is the
launch-time proof that they were selected and content-addressed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from tgvf_rl.conditioning import TargetConditioningConfig
from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ArtifactIdentity, CodeIdentity, ModelIdentity
from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_TOTAL_ROWS,
)
from tgvf_rl.protocol import TGVF_FOCUS_TOOL_SCHEMA_SHA256

from .checkpoint import PilotRunIdentityHashes
from .config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_MODEL_FAMILY,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_PATH,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
    PolicyPilotV1Config,
)


POLICY_PILOT_V1_RUN_MANIFEST_SCHEMA = "policy-pilot-v1-run-manifest-v1"
POLICY_PILOT_V1_JUDGE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
POLICY_PILOT_V1_REWARD_WEIGHTS = (0.8, 0.2, 1.2)


@dataclass(frozen=True, slots=True)
class PilotJudgeBindings:
    """All still-open identities of the required formal-Pilot 72B judge."""

    model: ArtifactIdentity
    prompt: ArtifactIdentity
    service: ArtifactIdentity
    sampling: ArtifactIdentity
    calibration: ArtifactIdentity
    failure_policy: ArtifactIdentity

    def __post_init__(self) -> None:
        _require_artifacts(
            (
                ("model", self.model),
                ("prompt", self.prompt),
                ("service", self.service),
                ("sampling", self.sampling),
                ("calibration", self.calibration),
                ("failure_policy", self.failure_policy),
            )
        )
        if self.model.name != POLICY_PILOT_V1_JUDGE_MODEL_NAME:
            raise ValueError(
                "Policy Pilot v1 judge model must be "
                f"{POLICY_PILOT_V1_JUDGE_MODEL_NAME!r}"
            )


@dataclass(frozen=True, slots=True)
class PilotObjectiveBindings:
    """Content identities for the accepted math and its concrete verifiers."""

    grpo: ArtifactIdentity
    reward_pipeline: ArtifactIdentity
    answer_verifier: ArtifactIdentity
    format_verifier: ArtifactIdentity
    conditional_tool_verifier: ArtifactIdentity
    diagnostic_kl_estimator: ArtifactIdentity

    def __post_init__(self) -> None:
        _require_artifacts(
            (
                ("grpo", self.grpo),
                ("reward_pipeline", self.reward_pipeline),
                ("answer_verifier", self.answer_verifier),
                ("format_verifier", self.format_verifier),
                ("conditional_tool_verifier", self.conditional_tool_verifier),
                ("diagnostic_kl_estimator", self.diagnostic_kl_estimator),
            )
        )


@dataclass(frozen=True, slots=True)
class PilotExecutionBindings:
    """Content-addressed bindings for deployment choices left open in §0.8."""

    code: CodeIdentity
    dependencies: ArtifactIdentity
    hardware_topology: ArtifactIdentity
    optimizer: ArtifactIdentity
    scheduler: ArtifactIdentity
    precision_batching: ArtifactIdentity
    weight_sync: ArtifactIdentity
    sampler_rng: ArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.code, CodeIdentity):
            raise TypeError("code must be a CodeIdentity")
        _require_artifacts(
            (
                ("dependencies", self.dependencies),
                ("hardware_topology", self.hardware_topology),
                ("optimizer", self.optimizer),
                ("scheduler", self.scheduler),
                ("precision_batching", self.precision_batching),
                ("weight_sync", self.weight_sync),
                ("sampler_rng", self.sampler_rng),
            )
        )


@dataclass(frozen=True, slots=True)
class PolicyPilotV1RunManifest:
    """Complete, immutable identity required before a formal Pilot launch."""

    run_id: str
    policy: PolicyPilotV1Config
    base_model: ModelIdentity
    processor: ArtifactIdentity
    tokenizer_fixture: ArtifactIdentity
    chat_template_fixture: ArtifactIdentity
    native_transcript_fixture: ArtifactIdentity
    prompt: ArtifactIdentity
    cap_error_and_recovery_fixture: ArtifactIdentity
    tgvf_adapter: ArtifactIdentity
    target_conditioning: TargetConditioningConfig
    dataset_manifest: ArtifactIdentity
    dataset_shuffle_seed: int
    objectives: PilotObjectiveBindings
    judge: PilotJudgeBindings
    execution: PilotExecutionBindings
    maximum_optimizer_steps: int
    checkpoint_and_evaluation_steps: tuple[int, ...]
    schema_version: str = POLICY_PILOT_V1_RUN_MANIFEST_SCHEMA
    dataset_id: str = DEEPEYES47K_DATASET_ID
    dataset_snapshot: str = DEEPEYES47K_SNAPSHOT
    dataset_rows: int = DEEPEYES47K_TOTAL_ROWS
    tool_schema_sha256: str = TGVF_FOCUS_TOOL_SCHEMA_SHA256
    reward_weights: tuple[float, float, float] = POLICY_PILOT_V1_REWARD_WEIGHTS

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_weights", tuple(self.reward_weights))
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.policy, PolicyPilotV1Config):
            raise TypeError("policy must be a PolicyPilotV1Config")
        if not self.policy.sampling.is_run_bound:
            raise ContractUnsetError(
                "formal Pilot manifest requires bound min-p and stop/EOS sampling "
                "identities"
            )
        if not isinstance(self.base_model, ModelIdentity):
            raise TypeError("base_model must be a ModelIdentity")
        for name, value, expected in (
            ("family", self.base_model.family, POLICY_PILOT_V1_MODEL_FAMILY),
            ("name", self.base_model.model_name, POLICY_PILOT_V1_MODEL_NAME),
            ("path", self.base_model.revision_or_path, POLICY_PILOT_V1_MODEL_PATH),
            (
                "tokenizer length",
                self.base_model.tokenizer_length,
                POLICY_PILOT_V1_TOKENIZER_LENGTH,
            ),
            (
                "chat-template SHA256",
                self.base_model.chat_template_sha256,
                POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
            ),
        ):
            if value != expected:
                raise ValueError(
                    f"Policy Pilot v1 requires base-model {name}={expected!r}, "
                    f"got {value!r}"
                )
        if not isinstance(self.target_conditioning, TargetConditioningConfig):
            raise TypeError(
                "target_conditioning must be a TargetConditioningConfig"
            )
        _require_artifacts(
            (
                ("processor", self.processor),
                ("tokenizer_fixture", self.tokenizer_fixture),
                ("chat_template_fixture", self.chat_template_fixture),
                ("native_transcript_fixture", self.native_transcript_fixture),
                ("prompt", self.prompt),
                (
                    "cap_error_and_recovery_fixture",
                    self.cap_error_and_recovery_fixture,
                ),
                ("tgvf_adapter", self.tgvf_adapter),
                ("dataset_manifest", self.dataset_manifest),
            )
        )
        for name, value, expected in (
            ("schema_version", self.schema_version, POLICY_PILOT_V1_RUN_MANIFEST_SCHEMA),
            ("dataset_id", self.dataset_id, DEEPEYES47K_DATASET_ID),
            ("dataset_snapshot", self.dataset_snapshot, DEEPEYES47K_SNAPSHOT),
            ("dataset_rows", self.dataset_rows, DEEPEYES47K_TOTAL_ROWS),
            (
                "tool_schema_sha256",
                self.tool_schema_sha256,
                TGVF_FOCUS_TOOL_SCHEMA_SHA256,
            ),
            ("reward_weights", self.reward_weights, POLICY_PILOT_V1_REWARD_WEIGHTS),
        ):
            if value != expected:
                raise ValueError(
                    f"Policy Pilot v1 requires {name}={expected!r}, got {value!r}"
                )
        if type(self.dataset_shuffle_seed) is not int or self.dataset_shuffle_seed < 0:
            raise ValueError("dataset_shuffle_seed must be a non-negative integer")
        if not isinstance(self.objectives, PilotObjectiveBindings):
            raise TypeError("objectives must be PilotObjectiveBindings")
        if not isinstance(self.judge, PilotJudgeBindings):
            raise TypeError("judge must be PilotJudgeBindings")
        if not isinstance(self.execution, PilotExecutionBindings):
            raise TypeError("execution must be PilotExecutionBindings")
        if type(self.maximum_optimizer_steps) is not int or self.maximum_optimizer_steps <= 0:
            raise ValueError("maximum_optimizer_steps must be a positive integer")

        steps = tuple(self.checkpoint_and_evaluation_steps)
        object.__setattr__(self, "checkpoint_and_evaluation_steps", steps)
        if (
            not steps
            or steps[0] != 0
            or any(type(step) is not int or step < 0 for step in steps)
            or any(left >= right for left, right in zip(steps, steps[1:]))
        ):
            raise ValueError(
                "checkpoint_and_evaluation_steps must be strictly increasing and "
                "start at step 0"
            )
        if steps[-1] > self.maximum_optimizer_steps:
            raise ValueError(
                "checkpoint/evaluation step exceeds maximum_optimizer_steps"
            )

    def as_record(self) -> dict[str, Any]:
        """Return the complete canonical-JSON-compatible manifest record."""

        normalized = _normalize_identity_value(self)
        assert isinstance(normalized, dict)
        return normalized

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.as_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def checkpoint_run_identity(self) -> PilotRunIdentityHashes:
        """Build the aggregate plus audit-friendly component checkpoint hashes."""

        return PilotRunIdentityHashes.from_hashes(
            self.run_id,
            {
                "chat_template": self.base_model.chat_template_sha256,
                "chat_template_fixture": self.chat_template_fixture.sha256,
                "data_manifest": self.dataset_manifest.sha256,
                "native_transcript_fixture": self.native_transcript_fixture.sha256,
                "pilot_manifest": self.identity_sha256,
                "policy_config": self.policy.identity_sha256,
                "prompt": self.prompt.sha256,
                "tgvf_adapter": self.tgvf_adapter.sha256,
                "tokenizer_fixture": self.tokenizer_fixture.sha256,
            },
        )


def _require_artifacts(values: tuple[tuple[str, object], ...]) -> None:
    for name, value in values:
        if not isinstance(value, ArtifactIdentity):
            raise TypeError(f"{name} must be an ArtifactIdentity")


def _normalize_identity_value(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_identity_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_identity_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_identity_value(item) for item in value]
    return value


__all__ = [
    "POLICY_PILOT_V1_JUDGE_MODEL_NAME",
    "POLICY_PILOT_V1_REWARD_WEIGHTS",
    "POLICY_PILOT_V1_RUN_MANIFEST_SCHEMA",
    "PilotExecutionBindings",
    "PilotJudgeBindings",
    "PilotObjectiveBindings",
    "PolicyPilotV1RunManifest",
]
