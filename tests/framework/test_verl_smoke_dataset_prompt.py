from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import torch

from tgvf_rl.framework.verl.smoke_dataset import (
    TGVFSelectedSampleDataset,
    VerlSelectedSampleDatasetBinding,
    build_tgvf_only_smoke_messages,
)
from tgvf_rl.protocol import (
    NativeProtocolRenderer,
    TGVF_FOCUS_TOOL_NAME,
    TGVF_ONLY_SYSTEM_PROMPT,
    NativeToolCapabilityProfile,
    build_native_tool_schemas,
)


class _VisualTokenizer:
    chat_template = "fake-qwen3-thinking-template-v1"
    image_token = "<|image_pad|>"
    image_token_id = 9000

    def __len__(self) -> int:
        return 10_000

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


class _SourceImageProcessor:
    chat_template = _VisualTokenizer.chat_template

    def __init__(self) -> None:
        self.tokenizer = _VisualTokenizer()
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 3136, "longest_edge": 16_777_216},
            merge_size=2,
        )
        self.image_call: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert tools[0]["function"]["name"] == TGVF_FOCUS_TOOL_NAME
        system, user = messages
        user_text = user["content"][1]["text"]
        return (
            f"<|im_start|>system\n{system['content']}<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|>"
            f"<|vision_end|>{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

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
        self.image_call = dict(images_kwargs)
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


def test_verl_smoke_render_and_raw_rows_share_exact_tgvf_only_prompt() -> None:
    question = "What value is shown?"
    render_messages = build_tgvf_only_smoke_messages(question)
    raw_messages = build_tgvf_only_smoke_messages(
        question,
        image_path=Path("/dataset/image.png"),
    )

    assert render_messages[0] == {
        "role": "system",
        "content": TGVF_ONLY_SYSTEM_PROMPT,
    }
    assert raw_messages[0] == render_messages[0]
    assert render_messages[1]["content"][0] == {"type": "image"}
    assert raw_messages[1]["content"][0] == {
        "type": "image",
        "image": "/dataset/image.png",
    }
    assert raw_messages[1]["content"][1] == render_messages[1]["content"][1]
    assert render_messages[1]["content"][1]["text"].startswith(
        "\nWhat value is shown?\n\n"
    )

    tool_names = NativeToolCapabilityProfile.TGVF_ONLY.tool_names
    assert tool_names == (TGVF_FOCUS_TOOL_NAME,)
    assert (
        tuple(
            schema["function"]["name"]
            for schema in build_native_tool_schemas(tool_names)
        )
        == tool_names
    )


def test_selected_sample_materializes_real_source_visual_token_count(
    tmp_path: Path,
) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "source.png"
    image_module.new("RGB", (16, 8), color=(12, 34, 56)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    question = "What value is shown?"
    sample_id = "fixture:0"
    sample = {
        "sample_id": sample_id,
        "task_kind": "mcq",
        "data_source": "multiple_choice",
        "extra_info": {"question": question},
        "reward_model": {"ground_truth": "B"},
        "image": {"path": image_path.name, "sha256": image_sha256},
    }
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(sample, separators=(",", ":")) + "\n")
    samples_sha256 = hashlib.sha256(samples_path.read_bytes()).hexdigest()

    processor = _SourceImageProcessor()
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=len(processor.tokenizer),
        tool_names=NativeToolCapabilityProfile.TGVF_ONLY.tool_names,
        tool_schemas=build_native_tool_schemas(
            NativeToolCapabilityProfile.TGVF_ONLY.tool_names
        ),
    )
    canonical = renderer.render(
        build_tgvf_only_smoke_messages(question),
        add_generation_prompt=True,
    )
    binding = VerlSelectedSampleDatasetBinding(
        samples_path=samples_path,
        samples_sha256=samples_sha256,
        sample_id=sample_id,
        cursor=0,
        iteration_identity_sha256="1" * 64,
        image_path=image_path,
        image_sha256=image_sha256,
        question=question,
        ground_truth="B",
        data_source="multiple_choice",
        prompt_sha256=canonical.text_sha256,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        tokenizer_length=len(processor.tokenizer),
        repeat_count=1,
    )
    dataset = TGVFSelectedSampleDataset(
        str(samples_path),
        processor.tokenizer,
        {
            "tgvf_selected_sample": binding.as_config(),
            "mm_processor_kwargs": {"max_pixels": 512 * 512},
        },
        processor=processor,
    )

    row = dataset[0]
    model_ids = tuple(row["initial_prompt_token_ids"])
    visual_id = processor.tokenizer.image_token_id
    assert canonical.token_ids.count(visual_id) == 1
    assert model_ids.count(visual_id) == 4
    assert len(model_ids) == len(canonical.token_ids) + 3
    assert row["initial_prompt_text_sha256"] == canonical.text_sha256
    assert row["initial_prompt_text_sha256"] == binding.prompt_sha256
    expected_ids_sha = hashlib.sha256(
        b"".join(struct.pack("<I", token_id) for token_id in model_ids)
    ).hexdigest()
    assert row["initial_prompt_token_ids_sha256"] == expected_ids_sha
    assert processor.image_call == {
        "size": {"shortest_edge": 3136, "longest_edge": 512 * 512}
    }
