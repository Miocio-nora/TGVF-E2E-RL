from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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
    DeepEyesOfficialSample,
)
from tgvf_rl.data.deepeyes_official_schedule_index import DeepEyesScheduleIndex
from tgvf_rl.framework.verl import tgvf_deepeyes_matched_dataset as dataset_module
from tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset import (
    CROP_TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA,
    CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    DEEPEYES_TRAIN_SENTINEL,
    CropTGVFDeepEyesMatchedDataset,
    TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    TGVFDeepEyesMatchedDataset,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_BOXED_INSTRUCTION,
    THINKLITE_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
    TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    TGVF_DEEPEYES_MATCHED_USER_PROMPT,
)


class _FakeTokenizer:
    image_token = "<|image_pad|>"
    image_token_id = 90_000

    def __len__(self) -> int:
        return dataset_module.POLICY_PILOT_V1_TOKENIZER_LENGTH

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.image_token, cursor):
                result.append(self.image_token_id)
                cursor += len(self.image_token)
            else:
                result.extend(byte + 1 for byte in text[cursor].encode("utf-8"))
                cursor += 1
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == self.image_token
        return self.image_token_id


class _FakeInstructProcessor:
    chat_template = "fake-qwen3-vl-instruct-native-template"

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 3136, "longest_edge": 16_777_216},
            merge_size=2,
        )
        self.rendered_texts: list[str] = []
        self.image_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tools == []
        assert tokenize is False
        assert add_generation_prompt is True
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<|im_start|>{message['role']}\n")
            content = message["content"]
            if isinstance(content, str):
                chunks.append(content)
            else:
                for item in content:
                    if item["type"] == "image":
                        chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
                    else:
                        chunks.append(item["text"])
            chunks.append("<|im_end|>\n")
        chunks.append("<|im_start|>assistant\n")
        rendered = "".join(chunks)
        self.rendered_texts.append(rendered)
        return rendered

    def __call__(
        self,
        *,
        text,
        images,
        padding,
        return_tensors,
        images_kwargs,
    ):
        assert len(text) == len(images) == 1
        assert images[0].mode == "RGB"
        assert padding is False
        assert return_tensors == "pt"
        self.image_calls.append(dict(images_kwargs))
        canonical = self.tokenizer.encode(text[0], add_special_tokens=False)
        position = canonical.index(self.tokenizer.image_token_id)
        expanded = (
            canonical[:position]
            + [self.tokenizer.image_token_id] * 4
            + canonical[position + 1 :]
        )
        return {
            "input_ids": torch.tensor((expanded,), dtype=torch.long),
            "image_grid_thw": torch.tensor(((1, 4, 4),), dtype=torch.long),
        }


def _config(template_sha256: str) -> dict[str, object]:
    return {
        "deepeyes_tgvf_matched": {
            "schema_version": TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA,
            "root": str(DEEPEYES_T1_ROOT),
            "candidate_sidecar_path": str(DEEPEYES_CANDIDATE_SIDECAR),
            "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
            "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
            "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
            "candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
            "expected_sample_count": DEEPEYES_T1_SAMPLE_COUNT,
            "schedule_mode": "stratified",
            "schedule_seed": DEEPEYES_TRAIN_SEED,
            "probe_seed": DEEPEYES_PROBE_SEED,
            "visual_prompt_bundle_sha256": (
                TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
            ),
            "thinklite_prompt_bundle_sha256": (THINKLITE_PROMPT_IDENTITY.bundle_sha256),
            "model_name": dataset_module.POLICY_PILOT_V1_MODEL_NAME,
            "tokenizer_length": dataset_module.POLICY_PILOT_V1_TOKENIZER_LENGTH,
            "chat_template_sha256": template_sha256,
        },
        "mm_processor_kwargs": {"max_pixels": 512 * 512},
    }


def _crop_tgvf_config(template_sha256: str) -> dict[str, object]:
    return {
        "deepeyes_crop_tgvf_matched": {
            "schema_version": CROP_TGVF_DEEPEYES_MATCHED_DATASET_SCHEMA,
            "root": str(DEEPEYES_T1_ROOT),
            "candidate_sidecar_path": str(DEEPEYES_CANDIDATE_SIDECAR),
            "manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
            "content_sha256": DEEPEYES_T1_CONTENT_SHA256,
            "samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
            "candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
            "expected_sample_count": DEEPEYES_T1_SAMPLE_COUNT,
            "schedule_mode": "stratified",
            "schedule_seed": DEEPEYES_TRAIN_SEED,
            "probe_seed": DEEPEYES_PROBE_SEED,
            "visual_prompt_bundle_sha256": (
                CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
            ),
            "thinklite_prompt_bundle_sha256": (THINKLITE_PROMPT_IDENTITY.bundle_sha256),
            "model_name": dataset_module.POLICY_PILOT_V1_MODEL_NAME,
            "tokenizer_length": dataset_module.POLICY_PILOT_V1_TOKENIZER_LENGTH,
            "chat_template_sha256": template_sha256,
        },
        "mm_processor_kwargs": {"max_pixels": 512 * 512},
    }


def _sample(
    *,
    index: int,
    source: str,
    image_path: Path,
    image_sha256: str,
) -> DeepEyesOfficialSample:
    return DeepEyesOfficialSample(
        index=index,
        sample_id=f"{source}:{index}",
        candidate_sha256=f"{index + 1:x}" * 64,
        data_source=source,
        task_kind="math" if source == "thinklite" else "mcq",
        question=f"Which value is shown in {source}?",
        ground_truth=f"ground-truth-{source}",
        image_path=image_path,
        image_sha256=image_sha256,
        image_width=16,
        image_height=8,
        gt_regions=((1, 1, 8, 7),) if source == "vstar" else None,
    )


def _contains_answer_wrapper(value: object) -> bool:
    if isinstance(value, str):
        return "<answer>" in value or "</answer>" in value
    if isinstance(value, dict):
        return any(_contains_answer_wrapper(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_answer_wrapper(item) for item in value)
    return False


def test_matched_dataset_reuses_source_rows_and_materializes_native_images(
    tmp_path: Path, monkeypatch
) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "source.png"
    image_module.new("RGB", (16, 8), color=(12, 34, 56)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = tuple(
        _sample(
            index=index,
            source=source,
            image_path=image_path,
            image_sha256=image_sha256,
        )
        for index, source in enumerate(("vstar", "arxivqa", "thinklite"))
    )
    schedule = DeepEyesScheduleIndex(
        path=tmp_path / "schedule.json",
        file_sha256="a" * 64,
        identity_sha256="b" * 64,
        schedule_identity_sha256="c" * 64,
        probe_manifest={"name": "fixture"},
        train=samples,
        probe=samples,
        smoke=samples,
    )
    processor = _FakeInstructProcessor()
    template_sha256 = hashlib.sha256(
        processor.chat_template.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        dataset_module,
        "POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256",
        template_sha256,
    )
    monkeypatch.setattr(dataset_module, "_verified_schedule_index", lambda: schedule)

    dataset = TGVFDeepEyesMatchedDataset(
        str(DEEPEYES_TRAIN_SENTINEL),
        tokenizer=processor.tokenizer,
        processor=processor,
        config=_config(template_sha256),
    )
    rows = [dataset[index] for index in range(3)]

    for sample, row in zip(samples, rows, strict=True):
        assert row["sample_id"] == sample.sample_id
        assert row["source_image_path"] == str(sample.image_path)
        assert row["source_image_sha256"] == sample.image_sha256
        assert row["reward_model"] == {"ground_truth": sample.ground_truth}
        assert row["data_source"] == sample.data_source
        assert row["agent_name"] == TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
        assert row["initial_prompt_chat_template_sha256"] == template_sha256
        assert (
            tuple(row["initial_prompt_token_ids"]).count(
                processor.tokenizer.image_token_id
            )
            == 4
        )
        assert not _contains_answer_wrapper(row["raw_prompt"])

    vstar, arxivqa, thinklite = rows
    assert vstar["raw_prompt"][0] == {
        "role": "system",
        "content": TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    }
    assert vstar["raw_prompt"][1]["content"][1]["text"].endswith(
        TGVF_DEEPEYES_MATCHED_USER_PROMPT
    )
    assert vstar["prompt_bundle_sha256"] == (
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert arxivqa["extra_info"]["source_route"] == "matched_tgvf_visual_tool"
    assert arxivqa["extra_info"]["need_tools_kwargs"] is True
    assert thinklite["extra_info"]["source_route"] == "single_turn_no_tool"
    assert thinklite["extra_info"]["need_tools_kwargs"] is False
    assert thinklite["tools_kwargs"] == {}
    assert thinklite["prompt_bundle_sha256"] == (
        THINKLITE_PROMPT_IDENTITY.bundle_sha256
    )
    thinklite_text = thinklite["raw_prompt"][0]["content"][1]["text"]
    assert thinklite_text.endswith(THINKLITE_BOXED_INSTRUCTION)
    assert all("<answer>" not in text for text in processor.rendered_texts)
    assert len(processor.image_calls) == 3
    assert all(
        call == {"size": {"shortest_edge": 3136, "longest_edge": 512 * 512}}
        for call in processor.image_calls
    )


def test_atomic_crop_tgvf_dataset_changes_only_the_visual_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "source.png"
    image_module.new("RGB", (16, 8), color=(12, 34, 56)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = tuple(
        _sample(
            index=index,
            source=source,
            image_path=image_path,
            image_sha256=image_sha256,
        )
        for index, source in enumerate(("vstar", "arxivqa", "thinklite"))
    )
    schedule = DeepEyesScheduleIndex(
        path=tmp_path / "schedule.json",
        file_sha256="a" * 64,
        identity_sha256="b" * 64,
        schedule_identity_sha256="c" * 64,
        probe_manifest={"name": "fixture"},
        train=samples,
        probe=samples,
        smoke=samples,
    )
    processor = _FakeInstructProcessor()
    template_sha256 = hashlib.sha256(
        processor.chat_template.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        dataset_module,
        "POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256",
        template_sha256,
    )
    monkeypatch.setattr(dataset_module, "_verified_schedule_index", lambda: schedule)

    dataset = CropTGVFDeepEyesMatchedDataset(
        str(DEEPEYES_TRAIN_SENTINEL),
        tokenizer=processor.tokenizer,
        processor=processor,
        config=_crop_tgvf_config(template_sha256),
    )
    rows = [dataset[index] for index in range(3)]

    for sample, row in zip(samples, rows, strict=True):
        assert row["sample_id"] == sample.sample_id
        assert row["source_image_sha256"] == sample.image_sha256
        assert row["reward_model"] == {"ground_truth": sample.ground_truth}
        assert row["agent_name"] == CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
        assert not _contains_answer_wrapper(row["raw_prompt"])

    vstar, arxivqa, thinklite = rows
    assert vstar["raw_prompt"][0] == {
        "role": "system",
        "content": CROP_TGVF_DEEPEYES_MATCHED_SYSTEM_PROMPT,
    }
    assert vstar["raw_prompt"][1]["content"][1]["text"].endswith(
        CROP_TGVF_DEEPEYES_MATCHED_USER_PROMPT
    )
    assert vstar["prompt_bundle_sha256"] == (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
    )
    assert vstar["extra_info"]["source_route"] == ("matched_crop_tgvf_visual_tool")
    assert arxivqa["extra_info"]["source_route"] == ("matched_crop_tgvf_visual_tool")
    assert thinklite["extra_info"]["source_route"] == "single_turn_no_tool"
    assert thinklite["prompt_bundle_sha256"] == (
        THINKLITE_PROMPT_IDENTITY.bundle_sha256
    )
    assert all("<answer>" not in text for text in processor.rendered_texts)
