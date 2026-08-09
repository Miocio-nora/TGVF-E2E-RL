"""PRL15 TGVF rows over the immutable PRL13 DeepEyes schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal

import torch

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_CANDIDATE_SIDECAR,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_ROOT,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_TRAIN_SEED,
)
from tgvf_rl.framework.verl.deepeyes_official_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    TGVFDeepEyesOfficialDataset,
    _config_value,
    _copy_message,
    _observed_image_sha256,
    _plain_mapping,
    _split_from_files,
    _verified_schedule_index,
)
from tgvf_rl.framework.verl.smoke_dataset import (
    _configured_image_max_pixels,
    _materialize_source_image_prompt_token_ids,
    _token_ids_sha256,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_PROMPT_IDENTITY,
    build_thinklite_messages,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_PROMPT_VERSION,
    build_tgvf_visual_messages,
)
from tgvf_rl.tokenizer_invariants import effective_tokenizer_length


TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA = "tgvf.verl-deepeyes-matched-dataset.v1"
TGVF_DEEPEYES_MATCHED_DATASET_CLASS = (
    "tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset.TGVFDeepEyesMatchedDataset"
)
TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME = "prl15_tgvf_deepeyes_matched_visual"

DatasetSplit = Literal["train", "probe", "smoke"]

_CONFIG_FIELDS = {
    "root",
    "candidate_sidecar_path",
    "manifest_file_sha256",
    "content_sha256",
    "samples_sha256",
    "candidate_sidecar_sha256",
    "expected_sample_count",
    "schedule_mode",
    "schedule_seed",
    "probe_seed",
    "visual_prompt_bundle_sha256",
    "thinklite_prompt_bundle_sha256",
    "model_name",
    "tokenizer_length",
    "chat_template_sha256",
    "schema_version",
}


@dataclass(frozen=True, slots=True)
class DeepEyesTGVFMatchedDatasetBinding:
    """Bind PRL15 prompts to the unchanged PRL13 data/schedule identity."""

    root: Path
    candidate_sidecar_path: Path
    manifest_file_sha256: str
    content_sha256: str
    samples_sha256: str
    candidate_sidecar_sha256: str
    expected_sample_count: int
    schedule_mode: Literal["stratified", "natural"]
    schedule_seed: int
    probe_seed: int
    visual_prompt_bundle_sha256: str
    thinklite_prompt_bundle_sha256: str
    model_name: str
    tokenizer_length: int
    chat_template_sha256: str
    schema_version: str = TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self, "candidate_sidecar_path", Path(self.candidate_sidecar_path)
        )
        expected = {
            "root": DEEPEYES_T1_ROOT,
            "candidate_sidecar_path": DEEPEYES_CANDIDATE_SIDECAR,
            "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
            "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
            "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
            "candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
            "expected_sample_count": DEEPEYES_T1_SAMPLE_COUNT,
            "schedule_seed": DEEPEYES_TRAIN_SEED,
            "probe_seed": DEEPEYES_PROBE_SEED,
            "visual_prompt_bundle_sha256": (
                TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
            ),
            "thinklite_prompt_bundle_sha256": (THINKLITE_PROMPT_IDENTITY.bundle_sha256),
            "model_name": POLICY_PILOT_V1_MODEL_NAME,
            "tokenizer_length": POLICY_PILOT_V1_TOKENIZER_LENGTH,
            "chat_template_sha256": POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
            "schema_version": TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"matched TGVF dataset {name} differs")
        if self.schedule_mode not in {"stratified", "natural"}:
            raise ValueError("matched TGVF dataset schedule_mode differs")

    @classmethod
    def from_config(cls, value: object) -> "DeepEyesTGVFMatchedDatasetBinding":
        mapping = dict(_plain_mapping(value, "data.deepeyes_tgvf_matched"))
        if set(mapping) != _CONFIG_FIELDS:
            raise ValueError("data.deepeyes_tgvf_matched fields differ")
        return cls(**mapping)


class TGVFDeepEyesMatchedDataset(TGVFDeepEyesOfficialDataset):
    """Change only the visual policy protocol over the exact PRL13 schedule."""

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
            raise TypeError("matched TGVF dataset requires a Qwen processor")
        if max_samples != -1:
            raise ValueError("matched TGVF schedule forbids max_samples filtering")
        binding = DeepEyesTGVFMatchedDatasetBinding.from_config(
            _config_value(config, "deepeyes_tgvf_matched")
        )
        if binding.schedule_mode != "stratified":
            raise ValueError(
                "matched TGVF schedule index currently binds only the stratified arm"
            )
        _assert_bound_qwen3_instruct_processor(processor, binding)
        split: DatasetSplit = _split_from_files(data_files)
        schedule = _verified_schedule_index()
        if split == "train":
            selected = schedule.train
        elif split == "probe":
            selected = schedule.probe
        else:
            selected = schedule.smoke
        self.binding = binding
        self.split = split
        self.schedule_index_file_sha256 = schedule.file_sha256
        self.schedule_index_identity_sha256 = schedule.identity_sha256
        self.schedule_identity_sha256 = schedule.schedule_identity_sha256
        self.probe_manifest = schedule.probe_manifest
        self.samples = selected
        self.processor = processor
        self.image_max_pixels = _configured_image_max_pixels(config)

    def __getitem__(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.samples[index]
        observed_image_sha256 = _observed_image_sha256(str(sample.image_path))
        if observed_image_sha256 != sample.image_sha256:
            raise ValueError(
                f"schedule-index source image SHA-256 differs for {sample.sample_id}"
            )

        if sample.data_source == "thinklite":
            raw_prompt: list[dict[str, Any]] = [
                _copy_message(message)
                for message in build_thinklite_messages(
                    sample.question,
                    image=str(sample.image_path),
                    task_kind=sample.task_kind,
                )
            ]
            prompt_bundle_sha256 = THINKLITE_PROMPT_IDENTITY.bundle_sha256
            prompt_version = THINKLITE_PROMPT_IDENTITY.version
            need_tools_kwargs = False
            source_route = "single_turn_no_tool"
        else:
            raw_prompt = [
                _copy_message(message)
                for message in build_tgvf_visual_messages(
                    sample.question, image=str(sample.image_path)
                )
            ]
            prompt_bundle_sha256 = TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
            prompt_version = TGVF_DEEPEYES_MATCHED_PROMPT_VERSION
            need_tools_kwargs = True
            source_route = "matched_tgvf_visual_tool"

        rendered_text, canonical_token_ids = _render_native_instruct_prompt(
            processor=self.processor,
            raw_prompt=raw_prompt,
            binding=self.binding,
        )
        prompt_token_ids = _materialize_source_image_prompt_token_ids(
            processor=self.processor,
            canonical_token_ids=canonical_token_ids,
            prompt_text=rendered_text,
            image_path=sample.image_path,
            image_max_pixels=self.image_max_pixels,
        )
        tools_kwargs = dict(sample.tools_kwargs)
        extra_info = {
            "index": index,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": sample.task_kind,
            "data_source": sample.data_source,
            "source_route": source_route,
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": need_tools_kwargs,
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "prompt_version": prompt_version,
            "schedule_index_file_sha256": self.schedule_index_file_sha256,
            "schedule_index_identity_sha256": self.schedule_index_identity_sha256,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "dataset_split": self.split,
        }
        if self.split == "smoke":
            extra_info["smoke_expectation"] = (
                "tgvf_possible"
                if sample.data_source == "vstar"
                else "direct_no_call"
                if sample.data_source == "arxivqa"
                else "no_tool"
            )
        return {
            "raw_prompt": raw_prompt,
            "initial_prompt_token_ids": prompt_token_ids,
            "initial_prompt_text_sha256": hashlib.sha256(
                rendered_text.encode("utf-8")
            ).hexdigest(),
            "initial_prompt_token_ids_sha256": _token_ids_sha256(prompt_token_ids),
            "initial_prompt_chat_template_sha256": (self.binding.chat_template_sha256),
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "data_source": sample.data_source,
            "reward_model": {"ground_truth": sample.ground_truth},
            "extra_info": extra_info,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "task_kind": sample.task_kind,
            "source_image_path": str(sample.image_path),
            "source_image_sha256": sample.image_sha256,
            "candidate_sha256": sample.candidate_sha256,
            # Every row must produce an exact replay bundle.  The matched loop
            # dispatches ThinkLite to its own strict single-turn/no-tool path.
            "agent_name": TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
            "tools_kwargs": tools_kwargs,
            "index": index,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }


def _assert_bound_qwen3_instruct_processor(
    processor: object, binding: DeepEyesTGVFMatchedDatasetBinding
) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("matched TGVF processor must expose its tokenizer")
    if effective_tokenizer_length(tokenizer) != binding.tokenizer_length:
        raise ValueError("matched TGVF tokenizer length differs")
    template = getattr(processor, "chat_template", None) or getattr(
        tokenizer, "chat_template", None
    )
    if not isinstance(template, str) or not template:
        raise ValueError("matched TGVF Qwen chat template must be explicit")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != (
        binding.chat_template_sha256
    ):
        raise ValueError("matched TGVF Qwen chat template identity differs")
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise TypeError("matched TGVF processor must apply its native chat template")


def _render_native_instruct_prompt(
    *,
    processor: object,
    raw_prompt: Sequence[Mapping[str, Any]],
    binding: DeepEyesTGVFMatchedDatasetBinding,
) -> tuple[str, tuple[int, ...]]:
    _assert_bound_qwen3_instruct_processor(processor, binding)
    rendered = processor.apply_chat_template(
        [_copy_message(message) for message in raw_prompt],
        tools=[],
        tokenize=False,
        add_generation_prompt=True,
    )
    _assert_bound_qwen3_instruct_processor(processor, binding)
    if not isinstance(rendered, str):
        raise TypeError("matched TGVF chat template did not return text")
    if "<answer>" in rendered or "</answer>" in rendered:
        raise ValueError("matched TGVF prompt contains a forbidden answer wrapper")
    expected_prefill = "<|im_start|>assistant\n"
    if not rendered.endswith(expected_prefill):
        raise ValueError("matched TGVF prompt is not Qwen3-VL Instruct native form")
    if rendered.rsplit(expected_prefill, 1)[-1]:
        raise ValueError("matched TGVF prompt has content after assistant prefill")
    tokenizer = processor.tokenizer
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if (
        not isinstance(token_ids, (list, tuple))
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise TypeError("matched TGVF tokenizer returned invalid prompt IDs")
    _assert_bound_qwen3_instruct_processor(processor, binding)
    return rendered, tuple(token_ids)


__all__ = [
    "DEEPEYES_PROBE_SENTINEL",
    "DEEPEYES_SMOKE_SENTINEL",
    "DEEPEYES_TRAIN_SENTINEL",
    "DeepEyesTGVFMatchedDatasetBinding",
    "TGVF_DEEPEYES_MATCHED_DATASET_CLASS",
    "TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA",
    "TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME",
    "TGVFDeepEyesMatchedDataset",
]
