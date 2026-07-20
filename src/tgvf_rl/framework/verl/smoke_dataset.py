"""Read-only upstream dataset entry for one selected Policy E2E smoke sample."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.protocol import NativeProtocolRenderer


VERL_SELECTED_SAMPLE_DATASET_SCHEMA = "tgvf-verl-selected-sample-v1"
VERL_SELECTED_SAMPLE_AGENT_NAME = "tgvf_native_policy"
_CONFIG_FIELDS = {
    "schema_version",
    "samples_path",
    "samples_sha256",
    "sample_id",
    "cursor",
    "iteration_identity_sha256",
    "image_path",
    "image_sha256",
    "question",
    "ground_truth",
    "data_source",
    "prompt_sha256",
    "tokenizer_length",
    "repeat_count",
    "agent_name",
}


@dataclass(frozen=True, slots=True)
class VerlSelectedSampleDatasetBinding:
    """Serializable data-only input consumed by the upstream custom dataset."""

    samples_path: Path
    samples_sha256: str
    sample_id: str
    cursor: int
    iteration_identity_sha256: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str
    data_source: str
    prompt_sha256: str
    tokenizer_length: int
    repeat_count: int
    agent_name: str = VERL_SELECTED_SAMPLE_AGENT_NAME
    schema_version: str = VERL_SELECTED_SAMPLE_DATASET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples_path", Path(self.samples_path))
        object.__setattr__(self, "image_path", Path(self.image_path))
        for field_name in (
            "samples_sha256",
            "iteration_identity_sha256",
            "image_sha256",
            "prompt_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in ("sample_id", "question", "ground_truth", "data_source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if type(self.tokenizer_length) is not int or self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        if type(self.repeat_count) is not int or self.repeat_count <= 0:
            raise ValueError("repeat_count must be positive")
        if self.agent_name != VERL_SELECTED_SAMPLE_AGENT_NAME:
            raise ValueError("selected-sample dataset agent_name differs")
        if self.schema_version != VERL_SELECTED_SAMPLE_DATASET_SCHEMA:
            raise ValueError("selected-sample dataset schema differs")

    @classmethod
    def from_run_config(
        cls, config: PolicyE2ESmokeRunConfig
    ) -> "VerlSelectedSampleDatasetBinding":
        if not isinstance(config, PolicyE2ESmokeRunConfig):
            raise TypeError("config must be PolicyE2ESmokeRunConfig")
        sample = config.dataset.selected_sample
        return cls(
            samples_path=config.dataset.root / "samples.jsonl",
            samples_sha256=config.dataset.samples_sha256,
            sample_id=sample.sample_id,
            cursor=config.dataset.cursor,
            iteration_identity_sha256=(
                config.dataset.iteration_identity_sha256
            ),
            image_path=sample.image_path,
            image_sha256=sample.image_sha256,
            question=sample.question,
            ground_truth=sample.ground_truth,
            data_source=sample.data_source,
            prompt_sha256=config.protocol.prompt_sha256,
            tokenizer_length=config.model.tokenizer_length,
            # The selected row is replayed deterministically; this gives the
            # upstream drop-last dataloader exactly the configured prompt count.
            repeat_count=(
                config.accumulation.global_prompt_batch_size
                * config.training.maximum_optimizer_steps
            ),
        )

    def as_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "samples_path": str(self.samples_path),
            "samples_sha256": self.samples_sha256,
            "sample_id": self.sample_id,
            "cursor": self.cursor,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "image_path": str(self.image_path),
            "image_sha256": self.image_sha256,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "prompt_sha256": self.prompt_sha256,
            "tokenizer_length": self.tokenizer_length,
            "repeat_count": self.repeat_count,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_config(cls, value: object) -> "VerlSelectedSampleDatasetBinding":
        mapping = _plain_mapping(value, "data.tgvf_selected_sample")
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.tgvf_selected_sample fields differ")
        return cls(**dict(mapping))


class TGVFSelectedSampleDataset(Dataset):
    """Upstream-compatible deterministic dataset with no materialization writes."""

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
            raise TypeError("native Qwen selected-sample dataset requires a processor")
        if max_samples != -1:
            raise ValueError("selected-sample dataset forbids secondary max-sample selection")
        binding = VerlSelectedSampleDatasetBinding.from_config(
            _config_value(config, "tgvf_selected_sample")
        )
        paths = (
            (data_files,)
            if isinstance(data_files, (str, Path))
            else tuple(data_files)
        )
        if tuple(str(path) for path in paths) != (str(binding.samples_path),):
            raise ValueError("upstream data_files differ from the bound samples file")
        _verify_bound_files(binding)

        renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=binding.tokenizer_length,
        )
        prompt_messages = (
            {
                "role": "user",
                "content": (
                    {"type": "image"},
                    {"type": "text", "text": binding.question},
                ),
            },
        )
        rendered = renderer.render(prompt_messages, add_generation_prompt=True)
        renderer.assert_generation_prefill(rendered, renderer.tokenizer)
        if rendered.text_sha256 != binding.prompt_sha256:
            raise ValueError("selected native prompt text SHA256 differs from run config")

        self.binding = binding
        self._row: Mapping[str, object] = {
            "raw_prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(binding.image_path)},
                        {"type": "text", "text": binding.question},
                    ],
                }
            ],
            "initial_prompt_token_ids": rendered.token_ids,
            "initial_prompt_text_sha256": rendered.text_sha256,
            "initial_prompt_token_ids_sha256": rendered.token_ids_sha256,
            "sample_id": binding.sample_id,
            "dataset_iteration_identity_sha256": (
                binding.iteration_identity_sha256
            ),
            "source_image_path": str(binding.image_path),
            "source_image_sha256": binding.image_sha256,
            "data_source": binding.data_source,
            "reward_model": {"ground_truth": binding.ground_truth},
            "extra_info": {
                "index": binding.cursor,
                "sample_id": binding.sample_id,
                "task_kind": "mcq",
            },
            "index": binding.cursor,
            "agent_name": binding.agent_name,
        }

    def __len__(self) -> int:
        return self.binding.repeat_count

    def __getitem__(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        row = dict(self._row)
        row["raw_prompt"] = [
            {**message, "content": [dict(item) for item in message["content"]]}
            for message in self._row["raw_prompt"]
        ]
        row["reward_model"] = dict(self._row["reward_model"])
        row["extra_info"] = dict(self._row["extra_info"])
        # Upstream's collator requires at least one tensor field.
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return row


def _verify_bound_files(binding: VerlSelectedSampleDatasetBinding) -> None:
    if binding.samples_path.is_symlink() or not binding.samples_path.is_file():
        raise ValueError("bound samples file must be a regular file")
    if _sha256_file(binding.samples_path) != binding.samples_sha256:
        raise ValueError("bound samples file SHA256 mismatch")
    selected: object | None = None
    with binding.samples_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == binding.cursor:
                selected = json.loads(line)
                break
    if not isinstance(selected, Mapping):
        raise ValueError("bound selected sample is missing")
    expected = {
        "sample_id": binding.sample_id,
        "task_kind": "mcq",
        "data_source": binding.data_source,
        "extra_info": {"question": binding.question},
        "reward_model": {"ground_truth": binding.ground_truth},
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError("bound selected sample content differs")
    image = selected.get("image")
    if not isinstance(image, Mapping) or image.get("sha256") != binding.image_sha256:
        raise ValueError("bound selected sample image identity differs")
    if binding.image_path.is_symlink() or not binding.image_path.is_file():
        raise ValueError("bound selected image must be a regular file")
    if _sha256_file(binding.image_path) != binding.image_sha256:
        raise ValueError("bound selected image SHA256 mismatch")


def _plain_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError(f"{field_name} must be a mapping")


def _config_value(config: object, key: str) -> object:
    if isinstance(config, Mapping):
        if key not in config:
            raise ValueError(f"data.{key} is missing")
        return config[key]
    try:
        return getattr(config, key)
    except AttributeError as error:
        raise ValueError(f"data.{key} is missing") from error


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TGVFSelectedSampleDataset",
    "VERL_SELECTED_SAMPLE_AGENT_NAME",
    "VERL_SELECTED_SAMPLE_DATASET_SCHEMA",
    "VerlSelectedSampleDatasetBinding",
]
