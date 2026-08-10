"""Read-only upstream dataset entry for one selected Policy E2E smoke sample."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch
from torch.utils.data import Dataset

from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
)
from tgvf_rl.protocol.native import native_assistant_dialect_for_model


VERL_SELECTED_SAMPLE_DATASET_SCHEMA = "tgvf-verl-selected-sample-v3"
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
    "question_utf8_base64",
    "ground_truth_utf8_base64",
    "data_source",
    "prompt_sha256",
    "tool_profile",
    "tokenizer_length",
    "model_name",
    "repeat_count",
    "agent_name",
}


def build_tgvf_only_smoke_messages(
    question: str,
    *,
    image_path: Path | None = None,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_THINKING
    ),
) -> tuple[Mapping[str, Any], ...]:
    """Use the accepted TGVF-only v1 prompt in render and raw-row forms."""

    return build_visual_tool_smoke_messages(
        question,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        image_path=image_path,
        assistant_dialect=assistant_dialect,
    )


def build_visual_tool_smoke_messages(
    question: str,
    *,
    tool_profile: NativeToolCapabilityProfile,
    image_path: Path | None = None,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_THINKING
    ),
) -> tuple[Mapping[str, Any], ...]:
    """Build the accepted native prompt for one explicit visual-tool arm."""

    if not isinstance(tool_profile, NativeToolCapabilityProfile):
        raise TypeError("tool_profile must be NativeToolCapabilityProfile")
    messages = build_visual_tool_prompt_messages(
        question,
        tool_profile=tool_profile,
        assistant_dialect=assistant_dialect,
    )
    if image_path is None:
        return messages
    path = Path(image_path)
    system, user = messages
    user_content = tuple(user["content"])
    image_item = {**user_content[0], "image": str(path)}
    return (
        dict(system),
        {
            "role": "user",
            "content": (image_item, dict(user_content[1])),
        },
    )


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
    tool_profile: NativeToolCapabilityProfile
    tokenizer_length: int
    model_name: str
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
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if type(self.tokenizer_length) is not int or self.tokenizer_length <= 0:
            raise ValueError("tokenizer_length must be positive")
        native_assistant_dialect_for_model(self.model_name)
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
            iteration_identity_sha256=(config.dataset.iteration_identity_sha256),
            image_path=sample.image_path,
            image_sha256=sample.image_sha256,
            question=sample.question,
            ground_truth=sample.ground_truth,
            data_source=sample.data_source,
            prompt_sha256=config.protocol.prompt_sha256,
            tool_profile=config.protocol.tool_profile,
            tokenizer_length=config.model.tokenizer_length,
            model_name=config.model.model_name,
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
            # Hydra's override grammar preserves ``\\n`` and backslashes
            # literally instead of decoding JSON string escapes.  Carry the
            # two free-form dataset strings as ASCII so the subprocess/Ray
            # boundary is byte exact for multiline and LaTeX-rich examples.
            "question_utf8_base64": _encode_utf8_base64(self.question),
            "ground_truth_utf8_base64": _encode_utf8_base64(self.ground_truth),
            "data_source": self.data_source,
            "prompt_sha256": self.prompt_sha256,
            "tool_profile": self.tool_profile.value,
            "tokenizer_length": self.tokenizer_length,
            "model_name": self.model_name,
            "repeat_count": self.repeat_count,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_config(cls, value: object) -> "VerlSelectedSampleDatasetBinding":
        mapping = _plain_mapping(value, "data.tgvf_selected_sample")
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.tgvf_selected_sample fields differ")
        decoded = dict(mapping)
        decoded["question"] = _decode_utf8_base64(
            decoded.pop("question_utf8_base64"), "question_utf8_base64"
        )
        decoded["ground_truth"] = _decode_utf8_base64(
            decoded.pop("ground_truth_utf8_base64"), "ground_truth_utf8_base64"
        )
        decoded["tool_profile"] = NativeToolCapabilityProfile(
            decoded["tool_profile"]
        )
        return cls(**decoded)


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
            raise ValueError(
                "selected-sample dataset forbids secondary max-sample selection"
            )
        binding = VerlSelectedSampleDatasetBinding.from_config(
            _config_value(config, "tgvf_selected_sample")
        )
        image_max_pixels = _configured_image_max_pixels(config)
        paths = (
            (data_files,) if isinstance(data_files, (str, Path)) else tuple(data_files)
        )
        if tuple(str(path) for path in paths) != (str(binding.samples_path),):
            raise ValueError("upstream data_files differ from the bound samples file")
        _verify_bound_files(binding)

        tool_names = binding.tool_profile.tool_names
        tool_schemas = tuple(build_native_tool_schemas(tool_names))
        assistant_dialect = native_assistant_dialect_for_model(binding.model_name)
        renderer = NativeProtocolRenderer(
            processor,
            expected_tokenizer_length=binding.tokenizer_length,
            tool_names=tool_names,
            tool_schemas=tool_schemas,
            assistant_dialect=assistant_dialect,
        )
        if renderer.tool_schemas != tool_schemas:
            raise RuntimeError("native renderer lost the selected policy tool schemas")
        prompt_messages = build_visual_tool_smoke_messages(
            binding.question,
            tool_profile=binding.tool_profile,
            assistant_dialect=assistant_dialect,
        )
        raw_prompt = build_visual_tool_smoke_messages(
            binding.question,
            tool_profile=binding.tool_profile,
            image_path=binding.image_path,
            assistant_dialect=assistant_dialect,
        )
        rendered = renderer.render(prompt_messages, add_generation_prompt=True)
        renderer.assert_generation_prefill(rendered, renderer.tokenizer)
        if rendered.text_sha256 != binding.prompt_sha256:
            raise ValueError(
                "selected native prompt text SHA256 differs from run config"
            )
        model_prompt_token_ids = _materialize_source_image_prompt_token_ids(
            processor=processor,
            canonical_token_ids=rendered.token_ids,
            prompt_text=rendered.text,
            image_path=binding.image_path,
            image_max_pixels=image_max_pixels,
        )
        renderer.assert_tokenizer_length()
        renderer.assert_chat_template_identity()
        renderer.assert_tool_schema_identity()

        self.binding = binding
        self._row: Mapping[str, object] = {
            "raw_prompt": [
                {
                    **message,
                    "content": (
                        [dict(item) for item in message["content"]]
                        if not isinstance(message["content"], str)
                        else message["content"]
                    ),
                }
                for message in raw_prompt
            ],
            "initial_prompt_token_ids": model_prompt_token_ids,
            "initial_prompt_text_sha256": rendered.text_sha256,
            "initial_prompt_token_ids_sha256": _token_ids_sha256(
                model_prompt_token_ids
            ),
            "sample_id": binding.sample_id,
            "dataset_iteration_identity_sha256": (binding.iteration_identity_sha256),
            "source_image_path": str(binding.image_path),
            "source_image_sha256": binding.image_sha256,
            "question": binding.question,
            "data_source": binding.data_source,
            "task_kind": "mcq",
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
            {
                **message,
                "content": (
                    [dict(item) for item in message["content"]]
                    if not isinstance(message["content"], str)
                    else message["content"]
                ),
            }
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


def _encode_utf8_base64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _decode_utf8_base64(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
        return raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ValueError(f"{field_name} is not canonical UTF-8 base64") from error


def _config_value(config: object, key: str) -> object:
    if isinstance(config, Mapping):
        if key not in config:
            raise ValueError(f"data.{key} is missing")
        return config[key]
    try:
        return getattr(config, key)
    except AttributeError as error:
        raise ValueError(f"data.{key} is missing") from error


def _configured_image_max_pixels(config: object) -> int:
    processor_kwargs = _plain_mapping(
        _config_value(config, "mm_processor_kwargs"),
        "data.mm_processor_kwargs",
    )
    value = processor_kwargs.get("max_pixels")
    if type(value) is not int or value <= 0:
        raise ValueError("data.mm_processor_kwargs.max_pixels must be positive")
    return value


def _materialize_source_image_prompt_token_ids(
    *,
    processor: object,
    canonical_token_ids: Sequence[int],
    prompt_text: str,
    image_path: Path | None = None,
    source_rgb: torch.Tensor | None = None,
    image_max_pixels: int,
) -> tuple[int, ...]:
    """Expand one verified source image with processor-owned merged tokens.

    Dataset callers bind an immutable path.  Benchmark callers have already
    decoded and hash-verified the exact file bytes, so accepting that RGB
    tensor avoids reopening a mutable path between verification and prompt
    materialization.
    """

    if (image_path is None) == (source_rgb is None):
        raise ValueError("exactly one of image_path or source_rgb is required")

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if not isinstance(size, Mapping):
        raise TypeError("Qwen image processor must expose a size mapping")
    shortest_edge = size.get("shortest_edge")
    merge_size = getattr(image_processor, "merge_size", None)
    if type(shortest_edge) is not int or shortest_edge <= 0:
        raise ValueError("Qwen image processor shortest_edge must be positive")
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Qwen image processor merge_size must be positive")
    if image_max_pixels < shortest_edge:
        raise ValueError("image max pixels is below the processor minimum")
    if not callable(processor):
        raise TypeError("Qwen processor must be callable for image materialization")

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency guard
        raise RuntimeError("Pillow is required to materialize source images") from error
    if source_rgb is not None:
        if (
            source_rgb.device.type != "cpu"
            or source_rgb.dtype != torch.uint8
            or source_rgb.ndim != 3
            or source_rgb.shape[-1] != 3
        ):
            raise ValueError("source_rgb must be one CPU uint8 [H,W,3] tensor")
        image = Image.fromarray(source_rgb.numpy(), mode="RGB")
        batch = processor(
            text=[prompt_text],
            images=[image],
            padding=False,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": shortest_edge,
                    "longest_edge": image_max_pixels,
                }
            },
        )
    else:
        assert image_path is not None
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            batch = processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": shortest_edge,
                        "longest_edge": image_max_pixels,
                    }
                },
            )
    if not isinstance(batch, Mapping):
        raise TypeError("Qwen processor output must be a mapping")
    model_token_ids = _one_integer_row(batch.get("input_ids"), "input_ids")
    image_grid = _one_integer_row(batch.get("image_grid_thw"), "image_grid_thw")
    if len(image_grid) != 3 or any(value <= 0 for value in image_grid):
        raise ValueError("Qwen processor image_grid_thw must be one positive [T,H,W] row")
    premerge_count = image_grid[0] * image_grid[1] * image_grid[2]
    merge_area = merge_size**2
    if premerge_count % merge_area:
        raise ValueError("Qwen processor image grid is not divisible by merge area")
    merged_count = premerge_count // merge_area

    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise TypeError("Qwen tokenizer must resolve the image placeholder token")
    visual_token_id = convert("<|image_pad|>")
    if type(visual_token_id) is not int or visual_token_id < 0:
        raise TypeError("Qwen image placeholder must resolve to a token ID")
    _assert_single_visual_expansion(
        canonical_token_ids=canonical_token_ids,
        model_token_ids=model_token_ids,
        visual_token_id=visual_token_id,
        expected_visual_tokens=merged_count,
    )
    return model_token_ids


def _one_integer_row(value: object, field_name: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"Qwen processor {field_name} must have one row")
        value = value.detach().to(device="cpu").tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Qwen processor {field_name} must be a sequence")
    rows = tuple(value)
    if len(rows) != 1 or not isinstance(rows[0], Sequence) or isinstance(
        rows[0], (str, bytes)
    ):
        raise ValueError(f"Qwen processor {field_name} must have one row")
    row = tuple(rows[0])
    if not row or any(type(item) is not int or item < 0 for item in row):
        raise ValueError(
            f"Qwen processor {field_name} must contain non-negative integers"
        )
    return row


def _assert_single_visual_expansion(
    *,
    canonical_token_ids: Sequence[int],
    model_token_ids: Sequence[int],
    visual_token_id: int,
    expected_visual_tokens: int,
) -> None:
    canonical = tuple(canonical_token_ids)
    model = tuple(model_token_ids)
    if canonical.count(visual_token_id) != 1:
        raise ValueError("selected native prompt must contain one source image placeholder")
    if expected_visual_tokens <= 0:
        raise ValueError("source image must produce at least one merged visual token")
    visual_position = canonical.index(visual_token_id)
    prefix = canonical[:visual_position]
    suffix = canonical[visual_position + 1 :]
    expected = prefix + (visual_token_id,) * expected_visual_tokens + suffix
    if model != expected:
        raise ValueError(
            "processor tokenization differs from the canonical prompt outside the "
            "source-image expansion"
        )


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    raw = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    return hashlib.sha256(raw).hexdigest()


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
    "build_tgvf_only_smoke_messages",
]
