"""Shared veRL Dataset for the immutable T1 plus teacher-quarter schedule.

The materialized artifact owns sample selection and order.  This module only
adapts those rows to one of the three already-established visual protocols:
native Crop, matched TGVF, or atomic Crop+TGVF.  Keeping that choice in one
binding prevents the teacher mixture from acquiring tool-specific copies or
silently different schedules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from tgvf_rl.data.policy_teacher_quarter_mix import (
    POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE,
    PolicyTeacherQuarterMixRuntimeBinding,
    load_policy_teacher_quarter_mix_runtime,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    build_crop_tgvf_visual_messages,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_PROMPT_IDENTITY,
    VISUAL_PROMPT_IDENTITY,
    agent_name_for_source,
    build_thinklite_messages,
    build_visual_messages,
    validate_source_task_kind,
    validate_tools_kwargs_for_source,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    build_tgvf_visual_messages,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile

from .deepeyes_official_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    TGVFDeepEyesOfficialDataset,
    _copy_message,
    _observed_image_sha256,
    _verified_schedule_index,
)
from .smoke_dataset import (
    _config_value,
    _configured_image_max_pixels,
    _materialize_source_image_prompt_token_ids,
    _plain_mapping,
    _require_sha256,
    _token_ids_sha256,
)
from .tgvf_deepeyes_matched_dataset import (
    CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    _render_native_instruct_prompt,
)


POLICY_TEACHER_QUARTER_MIX_VERL_DATASET_SCHEMA = (
    "tgvf.verl-policy-teacher-quarter-mix.v1"
)
POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS = (
    "tgvf_rl.framework.verl.policy_teacher_quarter_mix_dataset."
    "PolicyTeacherQuarterMixDataset"
)
POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.policy_teacher_quarter_mix_dataset"
)
POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME = "policy_teacher_quarter_mix"

_CONFIG_FIELDS = {
    "schema_version",
    "root",
    "manifest_file_sha256",
    "content_sha256",
    "samples_sha256",
    "iteration_identity_sha256",
    "schedule_seed",
    "expected_sample_count",
    "tool_profile",
    "visual_prompt_bundle_sha256",
    "thinklite_prompt_bundle_sha256",
    "tokenizer_length",
    "model_name",
    "chat_template_sha256",
}


def _visual_prompt_contract(
    profile: NativeToolCapabilityProfile,
) -> tuple[object, str, object, str, str]:
    """Return identity, version, builder, agent, and source route."""

    if profile is NativeToolCapabilityProfile.CROP_ONLY:
        return (
            VISUAL_PROMPT_IDENTITY,
            VISUAL_PROMPT_IDENTITY.version,
            build_visual_messages,
            "",
            "native_crop_visual_tool",
        )
    if profile is NativeToolCapabilityProfile.TGVF_ONLY:
        return (
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
            TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
            build_tgvf_visual_messages,
            TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
            "matched_tgvf_visual_tool",
        )
    if profile is NativeToolCapabilityProfile.CROP_TGVF:
        return (
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
            build_crop_tgvf_visual_messages,
            CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
            "matched_crop_tgvf_visual_tool",
        )
    raise ValueError(f"unsupported teacher-quarter tool profile: {profile!r}")


def _smoke_expectation(
    profile: NativeToolCapabilityProfile, data_source: str
) -> str:
    if data_source == "vstar":
        return {
            NativeToolCapabilityProfile.CROP_ONLY: "crop_possible",
            NativeToolCapabilityProfile.TGVF_ONLY: "tgvf_possible",
            NativeToolCapabilityProfile.CROP_TGVF: "crop_tgvf_possible",
        }[profile]
    if data_source == "arxivqa":
        return "direct_no_call"
    return "no_tool"


@dataclass(frozen=True, slots=True)
class PolicyTeacherQuarterMixDatasetBinding:
    """Bind one immutable schedule to a renderer without changing its rows."""

    root: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    iteration_identity_sha256: str
    schedule_seed: int
    expected_sample_count: int
    tool_profile: NativeToolCapabilityProfile
    visual_prompt_bundle_sha256: str
    thinklite_prompt_bundle_sha256: str
    tokenizer_length: int = POLICY_PILOT_V1_TOKENIZER_LENGTH
    model_name: str = POLICY_PILOT_V1_MODEL_NAME
    chat_template_sha256: str = POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256
    schema_version: str = POLICY_TEACHER_QUARTER_MIX_VERL_DATASET_SCHEMA

    def __post_init__(self) -> None:
        root = Path(self.root)
        object.__setattr__(self, "root", root)
        if not root.is_absolute():
            raise ValueError("teacher-quarter dataset root must be absolute")
        for field_name in (
            "manifest_file_sha256",
            "content_sha256",
            "samples_sha256",
            "iteration_identity_sha256",
            "visual_prompt_bundle_sha256",
            "thinklite_prompt_bundle_sha256",
            "chat_template_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if type(self.schedule_seed) is not int or self.schedule_seed < 0:
            raise ValueError("teacher-quarter schedule_seed must be non-negative")
        if (
            type(self.expected_sample_count) is not int
            or self.expected_sample_count <= 0
        ):
            raise ValueError("teacher-quarter expected_sample_count must be positive")
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if type(self.tokenizer_length) is not int or self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        if self.model_name != POLICY_PILOT_V1_MODEL_NAME:
            raise ValueError("teacher-quarter model identity differs")
        if self.tokenizer_length != POLICY_PILOT_V1_TOKENIZER_LENGTH:
            raise ValueError("teacher-quarter tokenizer length differs")
        if self.chat_template_sha256 != POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256:
            raise ValueError("teacher-quarter chat-template identity differs")
        prompt_identity, _, _, _, _ = _visual_prompt_contract(self.tool_profile)
        if self.visual_prompt_bundle_sha256 != prompt_identity.bundle_sha256:
            raise ValueError("teacher-quarter visual prompt identity differs")
        if self.thinklite_prompt_bundle_sha256 != (
            THINKLITE_PROMPT_IDENTITY.bundle_sha256
        ):
            raise ValueError("teacher-quarter ThinkLite prompt identity differs")
        if self.schema_version != POLICY_TEACHER_QUARTER_MIX_VERL_DATASET_SCHEMA:
            raise ValueError("teacher-quarter veRL dataset schema differs")

    @property
    def runtime_binding(self) -> PolicyTeacherQuarterMixRuntimeBinding:
        return PolicyTeacherQuarterMixRuntimeBinding(
            manifest_file_sha256=self.manifest_file_sha256,
            content_sha256=self.content_sha256,
            schedule_seed=self.schedule_seed,
            expected_sample_count=self.expected_sample_count,
        )

    @property
    def samples_path(self) -> Path:
        return self.root / POLICY_TEACHER_QUARTER_MIX_SAMPLES_FILE

    def as_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "schedule_seed": self.schedule_seed,
            "expected_sample_count": self.expected_sample_count,
            "tool_profile": self.tool_profile.value,
            "visual_prompt_bundle_sha256": self.visual_prompt_bundle_sha256,
            "thinklite_prompt_bundle_sha256": self.thinklite_prompt_bundle_sha256,
            "tokenizer_length": self.tokenizer_length,
            "model_name": self.model_name,
            "chat_template_sha256": self.chat_template_sha256,
        }

    @classmethod
    def from_config(cls, value: object) -> "PolicyTeacherQuarterMixDatasetBinding":
        mapping = dict(
            _plain_mapping(value, f"data.{POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME}")
        )
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.policy_teacher_quarter_mix fields differ")
        mapping["tool_profile"] = NativeToolCapabilityProfile(
            mapping["tool_profile"]
        )
        return cls(**mapping)


class PolicyTeacherQuarterMixDataset(TGVFDeepEyesOfficialDataset):
    """Render the same teacher-quarter schedule for all three policy arms."""

    def __init__(
        self,
        data_files: str | Sequence[str],
        tokenizer: object,
        config: object,
        processor: object | None = None,
        max_samples: int = -1,
    ) -> None:
        del tokenizer
        if processor is None:
            raise TypeError("teacher-quarter dataset requires a Qwen processor")
        if max_samples != -1:
            raise ValueError("teacher-quarter schedule forbids max_samples filtering")
        binding = PolicyTeacherQuarterMixDatasetBinding.from_config(
            _config_value(config, POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME)
        )
        paths = (
            (data_files,)
            if isinstance(data_files, (str, Path))
            else tuple(data_files)
        )
        normalized = tuple(str(path) for path in paths)
        if normalized == (str(binding.samples_path),):
            runtime = load_policy_teacher_quarter_mix_runtime(
                binding.root, binding=binding.runtime_binding
            )
            if runtime.samples_sha256 != binding.samples_sha256:
                raise ValueError("teacher-quarter samples identity differs")
            if runtime.iteration_identity_sha256 != binding.iteration_identity_sha256:
                raise ValueError("teacher-quarter iteration identity differs")
            samples: Sequence[object] = runtime.samples
            split = "train"
        elif normalized == (str(DEEPEYES_PROBE_SENTINEL),):
            schedule = _verified_schedule_index()
            samples = schedule.probe
            split = "probe"
        elif normalized == (str(DEEPEYES_SMOKE_SENTINEL),):
            schedule = _verified_schedule_index()
            samples = schedule.smoke
            split = "smoke"
        elif normalized == (str(DEEPEYES_TRAIN_SENTINEL),):
            schedule = _verified_schedule_index()
            samples = schedule.train
            split = "legacy_train"
        else:
            raise ValueError(
                "teacher-quarter data_files must select its train schedule or the "
                "unchanged PRL13 probe/smoke split"
            )
        self.binding = binding
        self.samples = samples
        self.split = split
        self.processor = processor
        self.image_max_pixels = _configured_image_max_pixels(config)
        if split != "train":
            self.schedule_index_file_sha256 = schedule.file_sha256
            self.schedule_index_identity_sha256 = schedule.identity_sha256
            self.schedule_identity_sha256 = schedule.schedule_identity_sha256
            self.probe_manifest = schedule.probe_manifest

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.samples[index]
        image_path = Path(sample.image_path)
        observed_image_sha256 = _observed_image_sha256(str(image_path))
        if observed_image_sha256 != sample.image_sha256:
            raise ValueError(
                f"teacher-quarter source image SHA-256 differs for {sample.sample_id}"
            )
        task_kind = (
            sample.task_kind.value
            if hasattr(sample.task_kind, "value")
            else str(sample.task_kind)
        )
        family = validate_source_task_kind(sample.data_source, task_kind)
        metadata = dict(getattr(sample, "metadata", {}))
        tools_kwargs = (
            dict(sample.tools_kwargs)
            if hasattr(sample, "tools_kwargs")
            else dict(metadata.get("tools_kwargs", {}))
        )
        validate_tools_kwargs_for_source(sample.data_source, tools_kwargs)
        candidate_sha256 = getattr(sample, "candidate_sha256", None) or metadata.get(
            "candidate_sha256"
        )
        if not isinstance(candidate_sha256, str):
            raise ValueError("teacher-quarter candidate identity is missing")
        _require_sha256(candidate_sha256, "candidate_sha256")

        prompt_identity, prompt_version, builder, matched_agent, source_route = (
            _visual_prompt_contract(self.binding.tool_profile)
        )
        if family == "thinklite":
            raw_prompt: list[dict[str, Any]] = [
                _copy_message(message)
                for message in build_thinklite_messages(
                    sample.question,
                    image=str(image_path),
                    task_kind=task_kind,
                )
            ]
            prompt_bundle_sha256 = THINKLITE_PROMPT_IDENTITY.bundle_sha256
            prompt_version = THINKLITE_PROMPT_IDENTITY.version
            need_tools_kwargs = False
            source_route = "single_turn_no_tool"
        else:
            raw_prompt = [
                _copy_message(message)
                for message in builder(sample.question, image=str(image_path))
            ]
            prompt_bundle_sha256 = prompt_identity.bundle_sha256
            need_tools_kwargs = True

        if self.binding.tool_profile is NativeToolCapabilityProfile.CROP_ONLY:
            agent_name = agent_name_for_source(sample.data_source)
        else:
            agent_name = str(matched_agent)
        extra_info = {
            "index": index,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": task_kind,
            "data_source": sample.data_source,
            "source_dataset": getattr(
                sample, "source_dataset", sample.data_source
            ),
            "source_route": source_route,
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": need_tools_kwargs,
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "prompt_version": prompt_version,
            "dataset_split": self.split,
            "dataset_iteration_identity_sha256": (
                self.binding.iteration_identity_sha256
            ),
            "mixture_role": metadata.get(
                "mixture_role", self.split if self.split != "train" else None
            ),
            "parent": metadata.get("parent"),
        }
        if self.split != "train":
            extra_info.update(
                {
                    "schedule_index_file_sha256": (
                        self.schedule_index_file_sha256
                    ),
                    "schedule_index_identity_sha256": (
                        self.schedule_index_identity_sha256
                    ),
                    "schedule_identity_sha256": self.schedule_identity_sha256,
                }
            )
        if self.split == "smoke":
            extra_info["smoke_expectation"] = _smoke_expectation(
                self.binding.tool_profile, sample.data_source
            )
        result: dict[str, object] = {
            "raw_prompt": raw_prompt,
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "data_source": sample.data_source,
            "reward_model": {"ground_truth": sample.ground_truth},
            "extra_info": extra_info,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": task_kind,
            "source_image_path": str(image_path),
            "source_image_sha256": sample.image_sha256,
            "candidate_sha256": candidate_sha256,
            "agent_name": agent_name,
            "tools_kwargs": tools_kwargs,
            "index": index,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
        if self.binding.tool_profile is not NativeToolCapabilityProfile.CROP_ONLY:
            rendered_text, canonical_token_ids = _render_native_instruct_prompt(
                processor=self.processor,
                raw_prompt=raw_prompt,
                binding=self.binding,
            )
            prompt_token_ids = _materialize_source_image_prompt_token_ids(
                processor=self.processor,
                canonical_token_ids=canonical_token_ids,
                prompt_text=rendered_text,
                image_path=image_path,
                image_max_pixels=self.image_max_pixels,
            )
            result.update(
                {
                    "initial_prompt_token_ids": prompt_token_ids,
                    "initial_prompt_text_sha256": hashlib.sha256(
                        rendered_text.encode("utf-8")
                    ).hexdigest(),
                    "initial_prompt_token_ids_sha256": _token_ids_sha256(
                        prompt_token_ids
                    ),
                    "initial_prompt_chat_template_sha256": (
                        self.binding.chat_template_sha256
                    ),
                }
            )
        return result


__all__ = [
    "POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME",
    "POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS",
    "POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH",
    "POLICY_TEACHER_QUARTER_MIX_VERL_DATASET_SCHEMA",
    "PolicyTeacherQuarterMixDataset",
    "PolicyTeacherQuarterMixDatasetBinding",
]
