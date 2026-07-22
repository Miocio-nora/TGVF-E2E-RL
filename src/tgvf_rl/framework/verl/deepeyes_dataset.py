"""Upstream-compatible full DeepEyes-47K Policy RL dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tgvf_rl.data import (
    DeepEyes47KRuntimeBinding,
    load_deepeyes47k_runtime,
)
from tgvf_rl.protocol import (
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
    visual_tool_prompt_identity,
)

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


VERL_DEEPEYES47K_DATASET_SCHEMA = "tgvf-verl-deepeyes47k-v1"
_CONFIG_FIELDS = {
    "schema_version",
    "root",
    "manifest_file_sha256",
    "content_sha256",
    "samples_sha256",
    "iteration_identity_sha256",
    "shuffle_seed",
    "fixture",
    "expected_sample_count",
    "prompt_bundle_sha256",
    "tool_profile",
    "tokenizer_length",
    "agent_name",
}


@dataclass(frozen=True, slots=True)
class VerlDeepEyes47KDatasetBinding:
    root: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    iteration_identity_sha256: str
    shuffle_seed: int
    fixture: bool
    expected_sample_count: int
    prompt_bundle_sha256: str
    tool_profile: NativeToolCapabilityProfile
    tokenizer_length: int
    agent_name: str = VERL_SELECTED_SAMPLE_AGENT_NAME
    schema_version: str = VERL_DEEPEYES47K_DATASET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
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
        if type(self.fixture) is not bool:
            raise TypeError("fixture must be bool")
        if type(self.expected_sample_count) is not int or self.expected_sample_count <= 0:
            raise ValueError("expected_sample_count must be positive")
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if type(self.tokenizer_length) is not int or self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        if self.agent_name != VERL_SELECTED_SAMPLE_AGENT_NAME:
            raise ValueError("DeepEyes dataset agent_name differs")
        if self.schema_version != VERL_DEEPEYES47K_DATASET_SCHEMA:
            raise ValueError("DeepEyes dataset schema differs")
        expected_prompt = visual_tool_prompt_identity(self.tool_profile).bundle_sha256
        if self.prompt_bundle_sha256 != expected_prompt:
            raise ValueError("DeepEyes prompt bundle identity differs")

    @property
    def runtime_binding(self) -> DeepEyes47KRuntimeBinding:
        if self.fixture:
            return DeepEyes47KRuntimeBinding.fixture_binding(
                manifest_file_sha256=self.manifest_file_sha256,
                content_sha256=self.content_sha256,
                shuffle_seed=self.shuffle_seed,
                expected_sample_count=self.expected_sample_count,
            )
        return DeepEyes47KRuntimeBinding.formal(
            manifest_file_sha256=self.manifest_file_sha256,
            content_sha256=self.content_sha256,
            shuffle_seed=self.shuffle_seed,
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
            "fixture": self.fixture,
            "expected_sample_count": self.expected_sample_count,
            "prompt_bundle_sha256": self.prompt_bundle_sha256,
            "tool_profile": self.tool_profile.value,
            "tokenizer_length": self.tokenizer_length,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_config(cls, value: object) -> "VerlDeepEyes47KDatasetBinding":
        mapping = dict(_plain_mapping(value, "data.tgvf_deepeyes47k"))
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.tgvf_deepeyes47k fields differ")
        mapping["tool_profile"] = NativeToolCapabilityProfile(
            mapping["tool_profile"]
        )
        return cls(**mapping)


class TGVFDeepEyes47KDataset(Dataset):
    """Render every verified mixed-task sample with the accepted native prompt."""

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
            raise TypeError("native Qwen DeepEyes dataset requires a processor")
        if max_samples != -1:
            raise ValueError("DeepEyes dataset forbids secondary max-sample selection")
        binding = VerlDeepEyes47KDatasetBinding.from_config(
            _config_value(config, "tgvf_deepeyes47k")
        )
        samples_path = binding.root / "samples.jsonl"
        paths = (
            (data_files,) if isinstance(data_files, (str, Path)) else tuple(data_files)
        )
        if tuple(str(path) for path in paths) != (str(samples_path),):
            raise ValueError("upstream data_files differ from the bound DeepEyes file")
        runtime = load_deepeyes47k_runtime(
            binding.root,
            binding=binding.runtime_binding,
        )
        if runtime.samples_sha256 != binding.samples_sha256:
            raise ValueError("DeepEyes samples identity differs")
        if runtime.iteration_identity_sha256 != binding.iteration_identity_sha256:
            raise ValueError("DeepEyes iteration identity differs")

        tool_names = binding.tool_profile.tool_names
        tool_schemas = tuple(build_native_tool_schemas(tool_names))
        renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=binding.tokenizer_length,
            tool_names=tool_names,
            tool_schemas=tool_schemas,
        )
        renderer.assert_tokenizer_length()
        renderer.assert_chat_template_identity()
        renderer.assert_tool_schema_identity()
        self.binding = binding
        self.runtime = runtime
        self.processor = processor
        self.renderer = renderer
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
        )
        raw_prompt = build_visual_tool_smoke_messages(
            sample.question,
            tool_profile=self.binding.tool_profile,
            image_path=sample.image_path,
        )
        rendered = self.renderer.render(prompt_messages, add_generation_prompt=True)
        self.renderer.assert_generation_prefill(rendered, self.renderer.tokenizer)
        model_prompt_token_ids = _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=rendered.token_ids,
            prompt_text=rendered.text,
            image_path=sample.image_path,
            image_max_pixels=self.image_max_pixels,
        )
        ground_truth = sample.ground_truth
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError("Policy Pilot DeepEyes ground truth must be text")
        return {
            "raw_prompt": [_copy_message(message) for message in raw_prompt],
            "initial_prompt_token_ids": model_prompt_token_ids,
            "initial_prompt_text_sha256": rendered.text_sha256,
            "initial_prompt_token_ids_sha256": _token_ids_sha256(
                model_prompt_token_ids
            ),
            "prompt_bundle_sha256": self.binding.prompt_bundle_sha256,
            "sample_id": sample.sample_id,
            "dataset_iteration_identity_sha256": (
                self.binding.iteration_identity_sha256
            ),
            "source_image_path": str(sample.image_path),
            "source_image_sha256": sample.image_sha256,
            "question": sample.question,
            "data_source": sample.data_source,
            "task_kind": sample.task_kind.value,
            "reward_model": {"ground_truth": ground_truth},
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
        "content": (
            [dict(item) for item in content]
            if not isinstance(content, str)
            else content
        ),
    }


__all__ = [
    "TGVFDeepEyes47KDataset",
    "VERL_DEEPEYES47K_DATASET_SCHEMA",
    "VerlDeepEyes47KDatasetBinding",
]
