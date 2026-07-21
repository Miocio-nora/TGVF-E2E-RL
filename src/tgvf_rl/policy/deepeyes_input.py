"""DeepEyes inputs rendered with the accepted native Policy prompt bundle.

This module stops before processor/tokenizer work.  It binds the formal
materialized-dataset identity to one Pilot manifest, constructs the native
Qwen user-message/tool envelope, derives one execution-unique ``n=8`` group,
and supplies the immutable answer metadata needed by reward construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyes47KRuntimeBinding,
    DeepEyes47KRuntimeSample,
    DeepEyesTaskKind,
)
from tgvf_rl.protocol import (
    NativeToolCapabilityProfile,
    TGVF_FOCUS_TOOL_NAME,
    build_native_tool_schemas,
    TGVF_VISUAL_TOOL_PROMPTS_VERSION,
    VisualToolPromptIdentity,
    build_visual_tool_prompt_messages,
    native_policy_messages_sha256,
    native_tool_set_sha256,
    visual_tool_prompt_identity,
)
from tgvf_rl.rewards import (
    AnswerTaskKind,
    RewardContext,
    reward_context_from_trajectory,
)
from tgvf_rl.trajectories.schema import (
    TrajectoryIdentity,
    TrajectoryRecord,
)

from .batch import POLICY_PILOT_V1_GROUP_SIZE
from .manifest import PolicyPilotV1RunManifest


POLICY_DEEPEYES_DATASET_BINDING_SCHEMA = "tgvf.policy-pilot.deepeyes47k-binding.v1"
POLICY_NATIVE_PROMPT_INPUT_SCHEMA = "tgvf.policy-pilot.native-prompt-input.v1"
POLICY_EXECUTION_GROUP_UID_SCHEMA = "tgvf.policy-pilot.execution-group.v1"
POLICY_EXECUTION_GROUP_UID_PREFIX = "tgvf-pilot-execution-group:"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_non_empty_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class BoundDeepEyesPolicyDataset:
    """Proof that a formal runtime binding matches one Pilot run manifest."""

    run_id: str
    pilot_manifest_sha256: str
    manifest_file_sha256: str
    content_sha256: str
    shuffle_seed: int
    dataset_id: str = DEEPEYES47K_DATASET_ID
    snapshot: str = DEEPEYES47K_SNAPSHOT
    sample_count: int = DEEPEYES47K_TOTAL_ROWS
    schema_version: str = POLICY_DEEPEYES_DATASET_BINDING_SCHEMA

    def __post_init__(self) -> None:
        _require_non_empty_text(self.run_id, field_name="run_id")
        _require_sha256(self.pilot_manifest_sha256, field_name="pilot_manifest_sha256")
        _require_sha256(self.manifest_file_sha256, field_name="manifest_file_sha256")
        _require_sha256(self.content_sha256, field_name="content_sha256")
        _require_non_negative_integer(self.shuffle_seed, field_name="shuffle_seed")
        expected = (
            ("dataset_id", self.dataset_id, DEEPEYES47K_DATASET_ID),
            ("snapshot", self.snapshot, DEEPEYES47K_SNAPSHOT),
            ("sample_count", self.sample_count, DEEPEYES47K_TOTAL_ROWS),
            (
                "schema_version",
                self.schema_version,
                POLICY_DEEPEYES_DATASET_BINDING_SCHEMA,
            ),
        )
        for field_name, value, required in expected:
            if value != required:
                raise ValueError(
                    f"formal policy dataset requires {field_name}={required!r}"
                )

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": self.schema_version,
                    "run_id": self.run_id,
                    "pilot_manifest_sha256": self.pilot_manifest_sha256,
                    "dataset_id": self.dataset_id,
                    "snapshot": self.snapshot,
                    "sample_count": self.sample_count,
                    "manifest_file_sha256": self.manifest_file_sha256,
                    "content_sha256": self.content_sha256,
                    "shuffle_seed": self.shuffle_seed,
                }
            )
        )


def bind_formal_deepeyes47k_to_pilot(
    manifest: PolicyPilotV1RunManifest,
    runtime_binding: DeepEyes47KRuntimeBinding,
) -> BoundDeepEyesPolicyDataset:
    """Fail closed unless the formal loader identity is the Pilot data identity."""

    if not isinstance(manifest, PolicyPilotV1RunManifest):
        raise TypeError("manifest must be a PolicyPilotV1RunManifest")
    if not isinstance(runtime_binding, DeepEyes47KRuntimeBinding):
        raise TypeError("runtime_binding must be a DeepEyes47KRuntimeBinding")
    if runtime_binding.fixture:
        raise IdentityMismatchError(
            "formal Policy Pilot cannot bind a DeepEyes fixture runtime"
        )
    if runtime_binding.expected_sample_count != manifest.dataset_rows:
        raise IdentityMismatchError(
            "DeepEyes runtime row count differs from the Pilot manifest"
        )
    if runtime_binding.manifest_file_sha256 != manifest.dataset_manifest.sha256:
        raise IdentityMismatchError(
            "DeepEyes runtime manifest-file SHA differs from the Pilot manifest"
        )
    if runtime_binding.shuffle_seed != manifest.dataset_shuffle_seed:
        raise IdentityMismatchError(
            "DeepEyes runtime shuffle seed differs from the Pilot manifest"
        )
    return BoundDeepEyesPolicyDataset(
        run_id=manifest.run_id,
        pilot_manifest_sha256=manifest.identity_sha256,
        manifest_file_sha256=runtime_binding.manifest_file_sha256,
        content_sha256=runtime_binding.content_sha256,
        shuffle_seed=runtime_binding.shuffle_seed,
    )


@dataclass(frozen=True, slots=True)
class NativePolicyPromptInput:
    """One prompt-free sample rendered only to native Qwen message objects."""

    sample_id: str
    prompt_group_uid: str
    image_path: Path
    image_sha256: str
    question: str
    messages: tuple[Mapping[str, Any], ...]
    tool_profile: NativeToolCapabilityProfile = NativeToolCapabilityProfile.TGVF_ONLY
    tool_names: tuple[str, ...] = (TGVF_FOCUS_TOOL_NAME,)
    schema_version: str = POLICY_NATIVE_PROMPT_INPUT_SCHEMA

    def __post_init__(self) -> None:
        _require_non_empty_text(self.sample_id, field_name="sample_id")
        _require_non_empty_text(self.prompt_group_uid, field_name="prompt_group_uid")
        object.__setattr__(self, "image_path", Path(self.image_path))
        if not self.image_path.is_absolute() or not self.image_path.is_file():
            raise ValueError("policy prompt image_path must be an absolute file")
        _require_sha256(self.image_sha256, field_name="image_sha256")
        _require_non_empty_text(self.question, field_name="question")
        object.__setattr__(self, "messages", tuple(self.messages))
        expected_messages = build_visual_tool_prompt_messages(
            self.question,
            tool_profile=self.tool_profile,
        )
        if self.messages != expected_messages:
            raise ValueError(
                "policy native prompt differs from TGVF Visual Tool Prompts v1"
            )
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if self.tool_names != self.tool_profile.tool_names:
            raise ValueError("tool_names differ from the selected tool_profile")
        if self.schema_version != POLICY_NATIVE_PROMPT_INPUT_SCHEMA:
            raise ValueError("policy native prompt schema mismatch")

    @property
    def tool_schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(build_native_tool_schemas(self.tool_names))

    @property
    def tool_schema_sha256(self) -> str:
        return native_tool_set_sha256(self.tool_names)

    @property
    def prompt_version(self) -> str:
        return TGVF_VISUAL_TOOL_PROMPTS_VERSION

    @property
    def prompt_identity(self) -> VisualToolPromptIdentity:
        return visual_tool_prompt_identity(self.tool_profile)

    @property
    def system_prompt_sha256(self) -> str:
        return self.prompt_identity.system_prompt_sha256

    @property
    def response_version(self) -> str:
        return self.prompt_identity.response_version

    @property
    def shared_user_prompt_template_sha256(self) -> str:
        return self.prompt_identity.shared_user_prompt_template_sha256

    @property
    def prompt_bundle_sha256(self) -> str:
        return self.prompt_identity.bundle_sha256

    @property
    def success_response_template_sha256(self) -> str:
        return self.prompt_identity.success_response_template_sha256

    @property
    def messages_sha256(self) -> str:
        return native_policy_messages_sha256(self.messages)


def build_qwen_policy_user_prompt(
    sample: DeepEyes47KRuntimeSample,
    *,
    tool_profile: NativeToolCapabilityProfile = NativeToolCapabilityProfile.TGVF_ONLY,
) -> NativePolicyPromptInput:
    """Build image-plus-question input without reading any historical prompt."""

    if not isinstance(sample, DeepEyes47KRuntimeSample):
        raise TypeError("sample must be a DeepEyes47KRuntimeSample")
    if not isinstance(tool_profile, NativeToolCapabilityProfile):
        raise TypeError("tool_profile must be NativeToolCapabilityProfile")
    messages = build_visual_tool_prompt_messages(
        sample.question,
        tool_profile=tool_profile,
    )
    return NativePolicyPromptInput(
        sample_id=sample.sample_id,
        prompt_group_uid=sample.prompt_group_uid,
        image_path=sample.image_path,
        image_sha256=sample.image_sha256,
        question=sample.question,
        messages=messages,
        tool_profile=tool_profile,
        tool_names=tool_profile.tool_names,
    )


@dataclass(frozen=True, slots=True)
class PolicyExecutionGroupIdentity:
    """The execution-unique group ID plus its fixed eight trajectory IDs."""

    run_id: str
    sample_id: str
    prompt_group_uid: str
    optimizer_step: int
    data_cursor: int
    execution_nonce: str
    group_uid: str
    trajectory_identities: tuple[TrajectoryIdentity, ...]
    schema_version: str = POLICY_EXECUTION_GROUP_UID_SCHEMA

    def __post_init__(self) -> None:
        _require_non_empty_text(self.run_id, field_name="run_id")
        _require_non_empty_text(self.sample_id, field_name="sample_id")
        _require_non_empty_text(self.prompt_group_uid, field_name="prompt_group_uid")
        _require_non_negative_integer(self.optimizer_step, field_name="optimizer_step")
        _require_non_negative_integer(self.data_cursor, field_name="data_cursor")
        _require_non_empty_text(self.execution_nonce, field_name="execution_nonce")
        if not self.group_uid.startswith(POLICY_EXECUTION_GROUP_UID_PREFIX):
            raise ValueError("execution group UID has the wrong namespace")
        object.__setattr__(
            self, "trajectory_identities", tuple(self.trajectory_identities)
        )
        if len(self.trajectory_identities) != POLICY_PILOT_V1_GROUP_SIZE:
            raise ValueError("Policy Pilot execution groups require exactly n=8")
        if tuple(
            identity.rollout_index for identity in self.trajectory_identities
        ) != tuple(range(POLICY_PILOT_V1_GROUP_SIZE)):
            raise ValueError("Policy Pilot rollout indices must be exactly 0..7")
        if any(
            identity.run_id != self.run_id
            or identity.sample_id != self.sample_id
            or identity.group_id != self.group_uid
            for identity in self.trajectory_identities
        ):
            raise ValueError("trajectory identity differs from its execution group")
        canonical_ids = tuple(
            identity.canonical_id for identity in self.trajectory_identities
        )
        if len(set(canonical_ids)) != POLICY_PILOT_V1_GROUP_SIZE:
            raise ValueError("Policy Pilot trajectory identities must be unique")
        if self.schema_version != POLICY_EXECUTION_GROUP_UID_SCHEMA:
            raise ValueError("policy execution-group schema mismatch")


def derive_policy_execution_group(
    sample: DeepEyes47KRuntimeSample,
    *,
    run_id: str,
    optimizer_step: int,
    data_cursor: int,
    execution_nonce: str,
) -> PolicyExecutionGroupIdentity:
    """Derive a stable ID for one unique execution of a reusable prompt."""

    if not isinstance(sample, DeepEyes47KRuntimeSample):
        raise TypeError("sample must be a DeepEyes47KRuntimeSample")
    _require_non_empty_text(run_id, field_name="run_id")
    _require_non_negative_integer(optimizer_step, field_name="optimizer_step")
    _require_non_negative_integer(data_cursor, field_name="data_cursor")
    _require_non_empty_text(execution_nonce, field_name="execution_nonce")
    payload = {
        "schema_version": POLICY_EXECUTION_GROUP_UID_SCHEMA,
        "run_id": run_id,
        "sample_id": sample.sample_id,
        "prompt_group_uid": sample.prompt_group_uid,
        "optimizer_step": optimizer_step,
        "data_cursor": data_cursor,
        "execution_nonce": execution_nonce,
    }
    group_uid = POLICY_EXECUTION_GROUP_UID_PREFIX + _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    identities = tuple(
        TrajectoryIdentity(
            run_id=run_id,
            sample_id=sample.sample_id,
            rollout_index=rollout_index,
            group_id=group_uid,
        )
        for rollout_index in range(POLICY_PILOT_V1_GROUP_SIZE)
    )
    return PolicyExecutionGroupIdentity(
        run_id=run_id,
        sample_id=sample.sample_id,
        prompt_group_uid=sample.prompt_group_uid,
        optimizer_step=optimizer_step,
        data_cursor=data_cursor,
        execution_nonce=execution_nonce,
        group_uid=group_uid,
        trajectory_identities=identities,
    )


_ANSWER_TASK_KIND_BY_DEEPEYES_KIND = MappingProxyType(
    {
        DeepEyesTaskKind.MCQ: AnswerTaskKind.MULTIPLE_CHOICE,
        DeepEyesTaskKind.MATH: AnswerTaskKind.MATH,
        DeepEyesTaskKind.OPEN: AnswerTaskKind.OPEN_VQA,
    }
)


@dataclass(frozen=True, slots=True)
class DeepEyesRewardSource:
    """Prompt-free answer metadata retained for reward routing and logging."""

    sample_id: str
    question: str
    expected_answer: str
    task_kind: AnswerTaskKind
    data_source: str


class DeepEyesRewardContextProvider:
    """Build reward contexts from immutable, prompt-free DeepEyes metadata."""

    def __init__(self, samples: Sequence[DeepEyes47KRuntimeSample]) -> None:
        sources: dict[str, DeepEyesRewardSource] = {}
        for sample in samples:
            if not isinstance(sample, DeepEyes47KRuntimeSample):
                raise TypeError("reward sources must be DeepEyes runtime samples")
            if sample.sample_id in sources:
                raise ValueError("reward sources contain a duplicate sample_id")
            if not isinstance(sample.ground_truth, str) or not sample.ground_truth:
                raise ValueError(
                    "Policy Pilot DeepEyes ground truth must be a non-empty string"
                )
            sources[sample.sample_id] = DeepEyesRewardSource(
                sample_id=sample.sample_id,
                question=sample.question,
                expected_answer=sample.ground_truth,
                task_kind=_ANSWER_TASK_KIND_BY_DEEPEYES_KIND[sample.task_kind],
                data_source=sample.data_source,
            )
        if not sources:
            raise ValueError("reward context provider requires at least one sample")
        self._sources: Mapping[str, DeepEyesRewardSource] = MappingProxyType(sources)

    def source_for_sample(self, sample_id: str) -> DeepEyesRewardSource:
        _require_non_empty_text(sample_id, field_name="sample_id")
        try:
            return self._sources[sample_id]
        except KeyError as error:
            raise IdentityMismatchError(
                "trajectory sample is absent from the DeepEyes reward binding"
            ) from error

    def build(
        self,
        *,
        request: object,
        trajectory: TrajectoryRecord,
    ) -> RewardContext:
        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be a TrajectoryRecord")
        request_identity = getattr(request, "identity", None)
        if not isinstance(request_identity, TrajectoryIdentity):
            raise TypeError("request must expose a TrajectoryIdentity")
        if request_identity != trajectory.identity:
            raise IdentityMismatchError(
                "reward request identity differs from the completed trajectory"
            )
        source = self.source_for_sample(trajectory.identity.sample_id)
        return reward_context_from_trajectory(
            trajectory,
            question=source.question,
            expected_answer=source.expected_answer,
            task_kind=source.task_kind,
            data_source=source.data_source,
        )


__all__ = [
    "POLICY_DEEPEYES_DATASET_BINDING_SCHEMA",
    "POLICY_EXECUTION_GROUP_UID_PREFIX",
    "POLICY_EXECUTION_GROUP_UID_SCHEMA",
    "POLICY_NATIVE_PROMPT_INPUT_SCHEMA",
    "BoundDeepEyesPolicyDataset",
    "DeepEyesRewardContextProvider",
    "DeepEyesRewardSource",
    "NativePolicyPromptInput",
    "PolicyExecutionGroupIdentity",
    "bind_formal_deepeyes47k_to_pilot",
    "build_qwen_policy_user_prompt",
    "derive_policy_execution_group",
]
