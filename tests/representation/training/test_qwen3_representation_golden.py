from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Sequence

import pytest
import torch
from torch import nn

from tgvf_rl.protocol.native import NativeProtocolRenderer, RenderedTranscript
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import (
    TGVF_FOCUS_TOOL_NAME,
    TGVF_FOCUS_TOOL_SCHEMA_SHA256,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.native_pipeline import (
    NATIVE_ACTION_TARGET_SCHEMA_VERSION,
    REPRESENTATION_PROMPT_IDENTITY_V2,
    REPRESENTATION_PROMPT_SCHEMA_VERSION_V2,
    Qwen3NativeRepresentationGroupBuilder,
    RepresentationPromptConfig,
    _processor_batch,
    _qwen3_expansion,
    build_native_representation_messages,
    render_native_action_target,
)
from tgvf_rl.representation.training.runtime import Qwen3RepresentationRuntime
from tgvf_rl.representation.training.qwen3_counterfactual import (
    QWEN3_D_ONLY_TOOL_REASONING,
    _qwen3_geometry_carrier,
    materialize_qwen3_d_only_processor_prefix,
)
from tgvf_rl.representation.training.schema import (
    REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION_V2,
    RepresentationTrainingSample,
)
from tgvf_rl.representation.training.transcript import (
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION_V2,
    MODEL_EVIDENCE_SCHEMA_VERSION,
    NATIVE_REPRESENTATION_PRE_REASONING,
    TOKEN_EXPANSION_SCHEMA_VERSION,
    render_native_evidence_labels,
)


FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "qwen3_native_representation_smoke_v1.json"
)
V2_FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "qwen3_native_representation_smoke_v2.json"
)
D_ONLY_FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "qwen3_native_d_only_smoke_v1.json"
)
_FIXTURE_SCHEMA = "qwen3_native_representation_processor_golden_v1"
_FIXTURE_SCHEMA_V2 = "qwen3_native_representation_processor_golden_v2"
_MODEL_PATH = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
_EXPECTED_TOKENIZER_LENGTH = 151669
_PROMPT_IDENTITY = "qwen3-representation-smoke-only-v1"
_PROMPT_TEMPLATE = (
    "[SMOKE-ONLY REPRESENTATION FIXTURE; NOT A PRODUCTION PROMPT]\n"
    "Question: {question}\n"
    "Requested target: {target}\n"
    "Use the native focus tool once for that target."
)
_PROMPT_TEMPLATE_V2 = "{question}"
_QUESTION = "Which detail appears in the requested region?"
_TARGET = "左侧铭牌/section-A"
_EVIDENCE = "The focused crop shows 蓝色/C-7 on the metal tag."
_SHORT_ANSWER = "C-7."
_IMAGE_SIZE = (56, 56)
_IMAGE_ALGORITHM = "rgb_xy_mod_256_v1"


class _LanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 1)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _VocabularyOnlyQwen(nn.Module):
    """CPU-only vocabulary boundary; it never executes model weights."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _LanguageModel(vocabulary_size)
        self.lm_head = nn.Linear(1, vocabulary_size, bias=False)


def _load_accepted_processor(model_path: Path) -> Any:
    if not model_path.is_dir():
        pytest.skip("accepted local Qwen3 processor is unavailable")
    transformers = pytest.importorskip(
        "transformers", reason="accepted local Qwen3 processor is unavailable"
    )
    return transformers.AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )


def _make_image() -> Any:
    pil_image = pytest.importorskip(
        "PIL.Image", reason="accepted local Qwen3 processor is unavailable"
    )
    image = pil_image.new("RGB", _IMAGE_SIZE)
    image.putdata(
        [
            (
                (17 * x + 3 * y) % 256,
                (5 * x + 29 * y + 11) % 256,
                (31 * x + 7 * y + 19) % 256,
            )
            for y in range(_IMAGE_SIZE[1])
            for x in range(_IMAGE_SIZE[0])
        ]
    )
    return image


def test_golden_image_max_pixels_cap_changes_real_processor_grid() -> None:
    processor = _load_accepted_processor(Path(_MODEL_PATH))
    pil_image = pytest.importorskip(
        "PIL.Image", reason="accepted local Qwen3 processor is unavailable"
    )
    image = pil_image.new("RGB", (1024, 1024))
    text = "<|vision_start|><|image_pad|><|vision_end|>"

    uncapped = _processor_batch(processor, text=text, images=(image,))
    capped = _processor_batch(
        processor,
        text=text,
        images=(image,),
        image_max_pixels=512 * 512,
    )

    assert uncapped["image_grid_thw"].tolist() == [[1, 64, 64]]
    assert capped["image_grid_thw"].tolist() == [[1, 32, 32]]
    assert uncapped["pixel_values"].shape[0] == 4096
    assert capped["pixel_values"].shape[0] == 1024
    assert image.size == (1024, 1024)


def _prompt() -> RepresentationPromptConfig:
    prompt_sha256 = hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    return RepresentationPromptConfig(
        identity=_PROMPT_IDENTITY,
        template=_PROMPT_TEMPLATE,
        expected_sha256=prompt_sha256,
    )


def _prompt_v2() -> RepresentationPromptConfig:
    prompt_sha256 = hashlib.sha256(_PROMPT_TEMPLATE_V2.encode("utf-8")).hexdigest()
    return RepresentationPromptConfig(
        identity=REPRESENTATION_PROMPT_IDENTITY_V2,
        template=_PROMPT_TEMPLATE_V2,
        expected_sha256=prompt_sha256,
        schema_version=REPRESENTATION_PROMPT_SCHEMA_VERSION_V2,
    )


def _sample() -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id="qwen3-representation-smoke-row",
        image="in-memory://qwen3-representation-smoke-56x56-rgb",
        question=_QUESTION,
        target=_TARGET,
        evidence_description=_EVIDENCE,
    )


def _sample_v2() -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id="qwen3-representation-smoke-row-v2",
        image="in-memory://qwen3-representation-smoke-56x56-rgb",
        question=_QUESTION,
        target=_TARGET,
        evidence_description=_EVIDENCE,
        short_answer=_SHORT_ANSWER,
        identity_schema_version=REPRESENTATION_SAMPLE_IDENTITY_SCHEMA_VERSION_V2,
    )


def _runtime(processor: Any, renderer: NativeProtocolRenderer) -> Any:
    runtime = object.__new__(Qwen3RepresentationRuntime)
    runtime.processor = processor
    runtime.renderer = renderer
    runtime.model = _VocabularyOnlyQwen(len(processor.tokenizer))
    runtime.vision_tower = nn.Linear(1, 1, bias=False)
    return runtime


def _token_sha256(values: Sequence[int]) -> str:
    raw = b"".join(struct.pack("<I", int(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()


def _signed_int_sha256(values: Sequence[int]) -> str:
    raw = b"".join(struct.pack("<q", int(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()


def _transcript_record(transcript: RenderedTranscript) -> dict[str, Any]:
    return {
        "text_characters": len(transcript.text),
        "text_utf8_bytes": len(transcript.text.encode("utf-8")),
        "text_sha256": transcript.text_sha256,
        "token_length": len(transcript.token_ids),
        "token_ids_sha256": transcript.token_ids_sha256,
    }


def _expanded_input_record(input_ids: torch.Tensor) -> dict[str, Any]:
    values = tuple(int(value) for value in input_ids[0].tolist())
    return {
        "length": len(values),
        "token_ids_sha256": _token_sha256(values),
    }


def _block_record(block: Sequence[int]) -> dict[str, int]:
    values = tuple(int(value) for value in block)
    return {
        "start": values[0],
        "end": values[-1] + 1,
        "length": len(values),
    }


def _compute_golden(
    *,
    prompt: RepresentationPromptConfig | None = None,
    sample: RepresentationTrainingSample | None = None,
    fixture_schema: str = _FIXTURE_SCHEMA,
    production_prompt: bool = False,
) -> dict[str, Any]:
    model_path = Path(_MODEL_PATH)
    processor = _load_accepted_processor(model_path)
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=_EXPECTED_TOKENIZER_LENGTH,
    )
    runtime = _runtime(processor, renderer)
    prompt = _prompt() if prompt is None else prompt
    sample = _sample() if sample is None else sample
    image = _make_image()
    messages = build_native_representation_messages(sample, prompt)
    is_v2 = prompt.schema_version == REPRESENTATION_PROMPT_SCHEMA_VERSION_V2

    if is_v2:
        assert messages[0] == {
            "role": "user",
            "content": (
                {"type": "image"},
                {"type": "text", "text": sample.question},
            ),
        }
        assert sample.target not in messages[0]["content"][1]["text"]
        assert messages[1]["reasoning_content"] == NATIVE_REPRESENTATION_PRE_REASONING
        assert messages[3]["reasoning_content"] == sample.evidence_description
        assert messages[3]["content"] == sample.short_answer

    action = render_native_action_target(runtime, messages)
    parsed = StrictToolCallParser().parse(action.sampled_turn)
    builder = object.__new__(Qwen3NativeRepresentationGroupBuilder)
    builder.runtime = runtime
    builder.image_max_pixels = None
    model_action = builder._materialize_action(action, image)
    action_expansion = _qwen3_expansion(
        runtime,
        action.transcript.token_ids,
        model_action.input_ids,
    )

    canonical_evidence = render_native_evidence_labels(
        renderer,
        messages,
        evidence_description=sample.evidence_description,
    )
    assert canonical_evidence.schema_version == (
        CANONICAL_EVIDENCE_SCHEMA_VERSION_V2
        if is_v2
        else CANONICAL_EVIDENCE_SCHEMA_VERSION
    )
    evidence_batch = _processor_batch(
        processor,
        text=canonical_evidence.transcript.text,
        images=(image, image),
    )
    model_evidence = Qwen3VLAdapter().materialize_representation_supervision(
        runtime.model,
        renderer.tokenizer,
        canonical_evidence,
        evidence_batch["input_ids"].to(dtype=torch.long),
    )

    canonical_evidence_ids = tuple(
        canonical_evidence.transcript.token_ids[position]
        for position in canonical_evidence.evidence_token_positions
    )
    model_evidence_ids = tuple(
        model_evidence.model_token_ids[position]
        for position in model_evidence.evidence_token_positions
    )
    assert canonical_evidence_ids == model_evidence_ids
    assert parsed.target == _TARGET
    assert "/" in parsed.target and not parsed.target.isascii()
    assert image.mode == "RGB" and image.size == _IMAGE_SIZE
    assert tuple(int(value) for value in model_action.image_grid_thw[0].tolist()) == (
        1,
        16,
        16,
    )
    assert tuple(
        tuple(int(value) for value in row.tolist())
        for row in evidence_batch["image_grid_thw"]
    ) == ((1, 16, 16), (1, 16, 16))

    action_blocks = tuple(
        mapped
        for mapped in action_expansion.canonical_to_model_positions
        if len(mapped) > 1
    )
    evidence_blocks = model_evidence.visual_expansion_blocks
    assert len(action_blocks) == 1
    assert len(evidence_blocks) == 2

    answer_ignored_positions: tuple[int, ...] = ()
    expanded_answer_ignored_positions: tuple[int, ...] = ()
    if is_v2:
        assert sample.short_answer is not None
        assert canonical_evidence.answer_text == sample.short_answer
        answer_start = canonical_evidence.transcript.text.rfind(sample.short_answer)
        answer_end = answer_start + len(sample.short_answer)
        assert answer_start > canonical_evidence.evidence_char_end
        answer_ignored_positions = tuple(
            position
            for position, (start, end) in enumerate(canonical_evidence.token_offsets)
            if start < answer_end and end > answer_start
        )
        assert answer_ignored_positions
        assert all(
            canonical_evidence.canonical_labels[position] == EVIDENCE_IGNORE_INDEX
            for position in answer_ignored_positions
        )
        expanded_answer_ignored_positions = tuple(
            model_position
            for canonical_position in answer_ignored_positions
            for model_position in model_evidence.canonical_to_model_positions[
                canonical_position
            ]
        )
        assert expanded_answer_ignored_positions
        assert all(
            model_evidence.labels[position] == EVIDENCE_IGNORE_INDEX
            for position in expanded_answer_ignored_positions
        )

    sample_record = {
        "question": sample.question,
        "target": sample.target,
        "evidence_description": sample.evidence_description,
    }
    if is_v2:
        sample_record["short_answer"] = sample.short_answer

    evidence_supervision = {
        "ignore_index": EVIDENCE_IGNORE_INDEX,
        "canonical_positions": list(canonical_evidence.evidence_token_positions),
        "canonical_positions_sha256": _token_sha256(
            canonical_evidence.evidence_token_positions
        ),
        "canonical_token_ids_sha256": _token_sha256(canonical_evidence_ids),
        "processor_expanded_positions": list(model_evidence.evidence_token_positions),
        "processor_expanded_positions_sha256": _token_sha256(
            model_evidence.evidence_token_positions
        ),
        "processor_expanded_token_ids_sha256": _token_sha256(model_evidence_ids),
        "processor_expanded_labels_sha256": _signed_int_sha256(model_evidence.labels),
    }
    if is_v2:
        evidence_supervision.update(
            {
                "answer_text": sample.short_answer,
                "answer_ignored_canonical_positions": list(answer_ignored_positions),
                "answer_ignored_canonical_positions_sha256": _token_sha256(
                    answer_ignored_positions
                ),
                "answer_ignored_processor_positions": list(
                    expanded_answer_ignored_positions
                ),
                "answer_ignored_processor_positions_sha256": _token_sha256(
                    expanded_answer_ignored_positions
                ),
            }
        )

    image_bytes = image.tobytes()
    golden = {
        "fixture_schema": fixture_schema,
        "smoke_only": True,
        "production_prompt": production_prompt,
        "model": {
            "family": "qwen3_vl",
            "path": _MODEL_PATH,
            "processor_class": (
                f"{type(processor).__module__}.{type(processor).__qualname__}"
            ),
            "tokenizer_class": (
                f"{type(renderer.tokenizer).__module__}."
                f"{type(renderer.tokenizer).__qualname__}"
            ),
            "tokenizer_length": len(renderer.tokenizer),
            "chat_template_sha256": renderer.chat_template_sha256,
            "image_placeholder_token_id": int(
                renderer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            ),
        },
        "protocol": {
            "tool_name": TGVF_FOCUS_TOOL_NAME,
            "tool_schema_sha256": TGVF_FOCUS_TOOL_SCHEMA_SHA256,
            "representation_prompt_schema": prompt.schema_version,
            "native_action_target_schema": NATIVE_ACTION_TARGET_SCHEMA_VERSION,
            "canonical_evidence_schema": canonical_evidence.schema_version,
            "model_evidence_schema": MODEL_EVIDENCE_SCHEMA_VERSION,
            "token_expansion_schema": TOKEN_EXPANSION_SCHEMA_VERSION,
        },
        "prompt": {
            "identity": prompt.identity,
            "template": prompt.template,
            "sha256": prompt.sha256,
            "rendered_sha256": hashlib.sha256(
                prompt.render(sample).encode("utf-8")
            ).hexdigest(),
        },
        "sample": sample_record,
        "image": {
            "construction": _IMAGE_ALGORITHM,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "raw_rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "raw_rgb_bytes": len(image_bytes),
        },
        "canonical_transcripts": {
            "action_generation_prefill": _transcript_record(action.generation_prefill),
            "action": _transcript_record(action.transcript),
            "evidence_generation_prefill": _transcript_record(
                canonical_evidence.generation_prefill
            ),
            "evidence": _transcript_record(canonical_evidence.transcript),
        },
        "parsed_target": {
            "text": parsed.target,
            "sampled_token_span": {
                "start": parsed.target_span.token_start,
                "end": parsed.target_span.token_end,
            },
            "sampled_character_span": {
                "start": parsed.target_span.offsets.char_start,
                "end": parsed.target_span.offsets.char_end,
            },
            "sampled_utf8_byte_span": {
                "start": parsed.target_span.offsets.byte_start,
                "end": parsed.target_span.offsets.byte_end,
            },
            "canonical_transcript_token_span": {
                "start": action.canonical_target_span.start,
                "end": action.canonical_target_span.end,
            },
            "canonical_token_ids": list(action.canonical_target_token_ids),
            "canonical_token_ids_sha256": _token_sha256(
                action.canonical_target_token_ids
            ),
            "processor_expanded_token_span": {
                "start": model_action.target_span.start,
                "end": model_action.target_span.end,
            },
            "processor_expanded_token_ids": list(model_action.target_token_ids),
        },
        "processor_inputs": {
            "action": {
                **_expanded_input_record(model_action.input_ids),
                "image_grid_thw": [
                    [int(value) for value in model_action.image_grid_thw[0].tolist()]
                ],
                "visual_blocks": [_block_record(block) for block in action_blocks],
            },
            "evidence": {
                **_expanded_input_record(evidence_batch["input_ids"]),
                "image_grid_thw": [
                    [int(value) for value in row.tolist()]
                    for row in evidence_batch["image_grid_thw"]
                ],
                "visual_blocks": [_block_record(block) for block in evidence_blocks],
            },
        },
        "evidence_supervision": evidence_supervision,
    }
    if is_v2:
        golden["trajectory_contract"] = {
            "user_text": messages[0]["content"][1]["text"],
            "pre_reasoning": messages[1]["reasoning_content"],
            "tool_target": parsed.target,
            "post_reasoning": messages[3]["reasoning_content"],
            "final_answer": messages[3]["content"],
        }
    return golden


def test_qwen3_native_representation_processor_golden() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = _compute_golden()
    assert actual == expected


def test_qwen3_native_representation_v2_processor_golden() -> None:
    expected = json.loads(V2_FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = _compute_golden(
        prompt=_prompt_v2(),
        sample=_sample_v2(),
        fixture_schema=_FIXTURE_SCHEMA_V2,
        production_prompt=True,
    )
    assert actual == expected


def test_qwen3_native_d_only_processor_golden() -> None:
    processor = _load_accepted_processor(Path(_MODEL_PATH))
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=_EXPECTED_TOKENIZER_LENGTH,
    )
    geometry_carrier = _qwen3_geometry_carrier(processor, (1, 16, 16))
    prefix = materialize_qwen3_d_only_processor_prefix(
        processor=processor,
        renderer=renderer,
        sample=_sample(),
        prompt=_prompt(),
        geometry_image=geometry_carrier,
    )
    expanded_ids = tuple(int(value) for value in prefix.input_ids[0].tolist())
    image_token_id = int(renderer.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    actual = {
        "fixture_schema": "qwen3_native_d_only_processor_golden_v1",
        "smoke_only": True,
        "production_prompt": False,
        "model_path": _MODEL_PATH,
        "tokenizer_length": len(renderer.tokenizer),
        "chat_template_sha256": renderer.chat_template_sha256,
        "prompt_identity": prefix.prompt_identity,
        "tool_reasoning": QWEN3_D_ONLY_TOOL_REASONING,
        "source_image_placeholder_count": 0,
        "geometry_carrier": {
            "mode": geometry_carrier.mode,
            "size": list(geometry_carrier.size),
            "all_black": not any(geometry_carrier.tobytes()),
        },
        "canonical_d_placeholder_count": prefix.transcript.token_ids.count(
            image_token_id
        ),
        "transcript": _transcript_record(prefix.transcript),
        "processor_input": {
            **_expanded_input_record(prefix.input_ids),
            "attention_mask_all_one": bool(prefix.attention_mask.bool().all().item()),
            "attention_mask_dtype": str(prefix.attention_mask.dtype),
            "image_grid_thw": prefix.image_grid_thw.tolist(),
            "expanded_image_token_count": expanded_ids.count(image_token_id),
            "d_block": _block_record(prefix.d_positions),
            "geometry_pixel_values_shape": list(prefix.geometry_pixel_values_shape),
        },
    }
    expected = json.loads(D_ONLY_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert actual == expected
