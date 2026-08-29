"""veRL Dataset binding for generic Teacher50/100 policy schedules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tgvf_rl.data.policy_teacher_ratio_mix import (
    POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE,
    PolicyTeacherRatioMixRuntimeBinding,
    load_policy_teacher_ratio_mix_runtime,
    policy_teacher_ratio_mix_profile,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
)
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.protocol import NativeToolCapabilityProfile

from .deepeyes_official_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    _verified_schedule_index,
)
from .policy_teacher_quarter_mix_dataset import (
    PolicyTeacherQuarterMixDataset,
    _visual_prompt_contract,
)
from .smoke_dataset import (
    _config_value,
    _configured_image_max_pixels,
    _plain_mapping,
    _require_sha256,
)


POLICY_TEACHER_RATIO_MIX_VERL_DATASET_SCHEMA = (
    "tgvf.verl-policy-teacher-ratio-mix.v2"
)
POLICY_TEACHER_RATIO_MIX_DATASET_CLASS = (
    "tgvf_rl.framework.verl.policy_teacher_ratio_mix_dataset."
    "PolicyTeacherRatioMixDataset"
)
POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.policy_teacher_ratio_mix_dataset"
)
POLICY_TEACHER_RATIO_MIX_CONFIG_NAME = "policy_teacher_ratio_mix"

_CONFIG_FIELDS = {
    "schema_version",
    "root",
    "manifest_file_sha256",
    "content_sha256",
    "samples_sha256",
    "iteration_identity_sha256",
    "schedule_seed",
    "expected_sample_count",
    "teacher_percentage",
    "tool_profile",
    "visual_prompt_bundle_sha256",
    "thinklite_prompt_bundle_sha256",
    "tokenizer_length",
    "model_name",
    "chat_template_sha256",
}


@dataclass(frozen=True, slots=True)
class PolicyTeacherRatioMixDatasetBinding:
    root: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    iteration_identity_sha256: str
    schedule_seed: int
    expected_sample_count: int
    teacher_percentage: int
    tool_profile: NativeToolCapabilityProfile
    visual_prompt_bundle_sha256: str
    thinklite_prompt_bundle_sha256: str
    tokenizer_length: int = POLICY_PILOT_V1_TOKENIZER_LENGTH
    model_name: str = POLICY_PILOT_V1_MODEL_NAME
    chat_template_sha256: str = POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256
    schema_version: str = POLICY_TEACHER_RATIO_MIX_VERL_DATASET_SCHEMA

    def __post_init__(self) -> None:
        root = Path(self.root)
        object.__setattr__(self, "root", root)
        if not root.is_absolute():
            raise ValueError("teacher-ratio dataset root must be absolute")
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
            raise ValueError("teacher-ratio schedule_seed must be non-negative")
        policy_teacher_ratio_mix_profile(self.teacher_percentage)
        if self.expected_sample_count != 20_480:
            raise ValueError("teacher-ratio expected_sample_count must be 20,480")
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if self.model_name != POLICY_PILOT_V1_MODEL_NAME:
            raise ValueError("teacher-ratio model identity differs")
        if self.tokenizer_length != POLICY_PILOT_V1_TOKENIZER_LENGTH:
            raise ValueError("teacher-ratio tokenizer length differs")
        if self.chat_template_sha256 != POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256:
            raise ValueError("teacher-ratio chat-template identity differs")
        _visual_prompt_contract(
            self.tool_profile,
            self.visual_prompt_bundle_sha256,
        )
        if self.thinklite_prompt_bundle_sha256 != (
            THINKLITE_PROMPT_IDENTITY.bundle_sha256
        ):
            raise ValueError("teacher-ratio ThinkLite prompt identity differs")
        if self.schema_version != POLICY_TEACHER_RATIO_MIX_VERL_DATASET_SCHEMA:
            raise ValueError("teacher-ratio veRL dataset schema differs")

    @property
    def runtime_binding(self) -> PolicyTeacherRatioMixRuntimeBinding:
        return PolicyTeacherRatioMixRuntimeBinding(
            manifest_file_sha256=self.manifest_file_sha256,
            content_sha256=self.content_sha256,
            schedule_seed=self.schedule_seed,
            expected_sample_count=self.expected_sample_count,
            teacher_percentage=self.teacher_percentage,
        )

    @property
    def samples_path(self) -> Path:
        return self.root / POLICY_TEACHER_RATIO_MIX_SAMPLES_FILE

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
            "teacher_percentage": self.teacher_percentage,
            "tool_profile": self.tool_profile.value,
            "visual_prompt_bundle_sha256": self.visual_prompt_bundle_sha256,
            "thinklite_prompt_bundle_sha256": self.thinklite_prompt_bundle_sha256,
            "tokenizer_length": self.tokenizer_length,
            "model_name": self.model_name,
            "chat_template_sha256": self.chat_template_sha256,
        }

    @classmethod
    def from_config(cls, value: object) -> "PolicyTeacherRatioMixDatasetBinding":
        mapping = dict(
            _plain_mapping(value, f"data.{POLICY_TEACHER_RATIO_MIX_CONFIG_NAME}")
        )
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.policy_teacher_ratio_mix fields differ")
        mapping["tool_profile"] = NativeToolCapabilityProfile(
            mapping["tool_profile"]
        )
        return cls(**mapping)


class PolicyTeacherRatioMixDataset(PolicyTeacherQuarterMixDataset):
    """Reuse the accepted renderer while loading a ratio-bound schedule."""

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
            raise TypeError("teacher-ratio dataset requires a Qwen processor")
        if max_samples != -1:
            raise ValueError("teacher-ratio schedule forbids max_samples filtering")
        binding = PolicyTeacherRatioMixDatasetBinding.from_config(
            _config_value(config, POLICY_TEACHER_RATIO_MIX_CONFIG_NAME)
        )
        paths = (
            (data_files,) if isinstance(data_files, (str, Path)) else tuple(data_files)
        )
        normalized = tuple(str(path) for path in paths)
        if normalized == (str(binding.samples_path),):
            runtime = load_policy_teacher_ratio_mix_runtime(
                binding.root, binding=binding.runtime_binding
            )
            if runtime.samples_sha256 != binding.samples_sha256:
                raise ValueError("teacher-ratio samples identity differs")
            if runtime.iteration_identity_sha256 != binding.iteration_identity_sha256:
                raise ValueError("teacher-ratio iteration identity differs")
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
                "teacher-ratio data_files must select its train schedule or the "
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


__all__ = [
    "POLICY_TEACHER_RATIO_MIX_CONFIG_NAME",
    "POLICY_TEACHER_RATIO_MIX_DATASET_CLASS",
    "POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH",
    "POLICY_TEACHER_RATIO_MIX_VERL_DATASET_SCHEMA",
    "PolicyTeacherRatioMixDataset",
    "PolicyTeacherRatioMixDatasetBinding",
]
