"""veRL custom dataset for the immutable all-source final T1 retained pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tgvf_rl.data import (
    PolicyT1MixedRuntimeBinding,
    load_policy_t1_mixed_runtime,
)
from tgvf_rl.protocol import (
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
    visual_tool_prompt_identity,
)
from tgvf_rl.protocol.native import native_assistant_dialect_for_model

from .smoke_dataset import (
    VERL_SELECTED_SAMPLE_AGENT_NAME,
    _config_value,
    _configured_image_max_pixels,
    _materialize_source_image_prompt_token_ids,
    _plain_mapping,
    _require_sha256,
    _token_ids_sha256,
    build_visual_tool_smoke_messages,
)


VERL_POLICY_T1_MIXED_DATASET_SCHEMA = "tgvf-verl-policy-t1-mixed-v1"
_CONFIG_FIELDS = {
    "schema_version",
    "root",
    "manifest_file_sha256",
    "content_sha256",
    "samples_sha256",
    "iteration_identity_sha256",
    "shuffle_seed",
    "decision_stage",
    "expected_sample_count",
    "prompt_bundle_sha256",
    "tool_profile",
    "tokenizer_length",
    "model_name",
    "agent_name",
}


@dataclass(frozen=True, slots=True)
class VerlPolicyT1MixedDatasetBinding:
    root: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    iteration_identity_sha256: str
    shuffle_seed: int
    expected_sample_count: int
    prompt_bundle_sha256: str
    tool_profile: NativeToolCapabilityProfile
    tokenizer_length: int
    model_name: str
    decision_stage: str = "final"
    agent_name: str = VERL_SELECTED_SAMPLE_AGENT_NAME
    schema_version: str = VERL_POLICY_T1_MIXED_DATASET_SCHEMA

    def __post_init__(self) -> None:
        root = Path(self.root)
        object.__setattr__(self, "root", root)
        if not root.is_absolute():
            raise ValueError("Policy T1 mixed dataset root must be absolute")
        for field_name in (
            "manifest_file_sha256",
            "content_sha256",
            "samples_sha256",
            "iteration_identity_sha256",
            "prompt_bundle_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if type(self.shuffle_seed) is not int or self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be non-negative")
        if (
            type(self.expected_sample_count) is not int
            or self.expected_sample_count <= 0
        ):
            raise ValueError("expected_sample_count must be positive")
        if self.decision_stage != "final":
            raise ValueError("Policy T1 mixed dataset requires decision_stage='final'")
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if type(self.tokenizer_length) is not int or self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        dialect = native_assistant_dialect_for_model(self.model_name)
        if self.agent_name != VERL_SELECTED_SAMPLE_AGENT_NAME:
            raise ValueError("Policy T1 mixed dataset agent_name differs")
        if self.schema_version != VERL_POLICY_T1_MIXED_DATASET_SCHEMA:
            raise ValueError("Policy T1 mixed veRL dataset schema differs")
        if (
            self.prompt_bundle_sha256
            != visual_tool_prompt_identity(
                self.tool_profile, assistant_dialect=dialect
            ).bundle_sha256
        ):
            raise ValueError("Policy T1 mixed prompt bundle identity differs")

    @property
    def runtime_binding(self) -> PolicyT1MixedRuntimeBinding:
        return PolicyT1MixedRuntimeBinding(
            manifest_file_sha256=self.manifest_file_sha256,
            content_sha256=self.content_sha256,
            shuffle_seed=self.shuffle_seed,
            expected_sample_count=self.expected_sample_count,
        )

    def as_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "shuffle_seed": self.shuffle_seed,
            "decision_stage": self.decision_stage,
            "expected_sample_count": self.expected_sample_count,
            "prompt_bundle_sha256": self.prompt_bundle_sha256,
            "tool_profile": self.tool_profile.value,
            "tokenizer_length": self.tokenizer_length,
            "model_name": self.model_name,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_config(cls, value: object) -> "VerlPolicyT1MixedDatasetBinding":
        mapping = dict(_plain_mapping(value, "data.tgvf_policy_t1_mixed"))
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.tgvf_policy_t1_mixed fields differ")
        mapping["tool_profile"] = NativeToolCapabilityProfile(mapping["tool_profile"])
        return cls(**mapping)


class TGVFPolicyT1MixedDataset(Dataset):
    """Render verified V*, ArxivQA, and ThinkLite final T1 retains."""

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
            raise TypeError("native Qwen Policy T1 mixed dataset requires a processor")
        if max_samples != -1:
            raise ValueError(
                "Policy T1 mixed dataset forbids secondary max-sample selection"
            )
        binding = VerlPolicyT1MixedDatasetBinding.from_config(
            _config_value(config, "tgvf_policy_t1_mixed")
        )
        samples_path = binding.root / "samples.jsonl"
        paths = (
            (data_files,) if isinstance(data_files, (str, Path)) else tuple(data_files)
        )
        if tuple(str(path) for path in paths) != (str(samples_path),):
            raise ValueError(
                "upstream data_files differ from the bound Policy T1 mixed file"
            )
        runtime = load_policy_t1_mixed_runtime(
            binding.root, binding=binding.runtime_binding
        )
        if runtime.samples_sha256 != binding.samples_sha256:
            raise ValueError("Policy T1 mixed samples identity differs")
        if runtime.iteration_identity_sha256 != binding.iteration_identity_sha256:
            raise ValueError("Policy T1 mixed iteration identity differs")

        tool_names = binding.tool_profile.tool_names
        dialect = native_assistant_dialect_for_model(binding.model_name)
        renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=binding.tokenizer_length,
            tool_names=tool_names,
            tool_schemas=tuple(build_native_tool_schemas(tool_names)),
            assistant_dialect=dialect,
        )
        renderer.assert_tokenizer_length()
        renderer.assert_chat_template_identity()
        renderer.assert_tool_schema_identity()
        self.binding = binding
        self.runtime = runtime
        self.processor = processor
        self.renderer = renderer
        self.assistant_dialect = dialect
        self.image_max_pixels = _configured_image_max_pixels(config)

    def __len__(self) -> int:
        return len(self.runtime)

    def __getitem__(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.runtime[index]
        prompt_messages = build_visual_tool_prompt_messages(
            sample.question,
            tool_profile=self.binding.tool_profile,
            assistant_dialect=self.assistant_dialect,
        )
        raw_prompt = build_visual_tool_smoke_messages(
            sample.question,
            tool_profile=self.binding.tool_profile,
            image_path=sample.image_path,
            assistant_dialect=self.assistant_dialect,
        )
        rendered = self.renderer.render(prompt_messages, add_generation_prompt=True)
        self.renderer.assert_generation_prefill(rendered, self.renderer.tokenizer)
        prompt_ids = _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=rendered.token_ids,
            prompt_text=rendered.text,
            image_path=sample.image_path,
            image_max_pixels=self.image_max_pixels,
        )
        return {
            "raw_prompt": [_copy_message(message) for message in raw_prompt],
            "initial_prompt_token_ids": prompt_ids,
            "initial_prompt_text_sha256": rendered.text_sha256,
            "initial_prompt_token_ids_sha256": _token_ids_sha256(prompt_ids),
            "prompt_bundle_sha256": self.binding.prompt_bundle_sha256,
            "sample_id": sample.sample_id,
            "dataset_iteration_identity_sha256": self.binding.iteration_identity_sha256,
            "source_image_path": str(sample.image_path),
            "source_image_sha256": sample.image_sha256,
            "question": sample.question,
            "data_source": sample.data_source,
            "task_kind": sample.task_kind.value,
            "reward_model": {"ground_truth": sample.ground_truth},
            "extra_info": {
                "index": index,
                "sample_id": sample.sample_id,
                "task_kind": sample.task_kind.value,
            },
            "index": index,
            "agent_name": self.binding.agent_name,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message["content"]
    return {
        **message,
        "content": [dict(item) for item in content]
        if not isinstance(content, str)
        else content,
    }


__all__ = [
    "TGVFPolicyT1MixedDataset",
    "VERL_POLICY_T1_MIXED_DATASET_SCHEMA",
    "VerlPolicyT1MixedDatasetBinding",
]
