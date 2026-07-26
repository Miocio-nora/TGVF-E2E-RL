#!/usr/bin/env python3
"""Real Qwen3-Instruct initial/post-crop prompt-trigger canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Mapping

from PIL import Image

from tgvf_rl.environment.native_appender import (
    render_qwen_native_success_environment_text,
)
from tgvf_rl.protocol import (
    NativeAssistantDialect,
    NativeProtocolRenderer,
    NativeToolCapabilityProfile,
    StrictToolCallParser,
    build_native_tool_schemas,
    build_visual_tool_prompt_messages,
    visual_tool_prompt_identity,
)
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan
from tgvf_rl.protocol.tool_prompts import (
    IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT,
    QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER,
)
from tgvf_rl.qwen.crop_coordinates import map_qwen3_crop_bbox_to_source


SCHEMA = "qwen3-instruct-crop-prompt-canary-v1"
DIALECT = NativeAssistantDialect.QWEN3_VL_INSTRUCT
PROFILE = NativeToolCapabilityProfile.CROP_ONLY


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("canary config schema mismatch")
    return value, _sha_bytes(raw)


def _sampled_turn(tokenizer: Any, text: str) -> SampledAssistantTurn:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple(tuple(int(v) for v in pair) for pair in encoded["offset_mapping"])
    boundaries = [0]
    for character in text:
        boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
    spans = tuple(
        TokenByteSpan(index, token_id, boundaries[start], boundaries[end])
        for index, (token_id, (start, end)) in enumerate(
            zip(token_ids, offsets, strict=True)
        )
    )
    return SampledAssistantTurn(text, token_ids, spans)


def _controlled_call(config: Mapping[str, Any]) -> str:
    turn = config["controlled_tool_turn"]
    arguments = json.dumps(
        {
            "name": "image_zoom_in_tool",
            "arguments": {
                "bbox_2d": turn["bbox_2d"],
                "label": turn["label"],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<think>{turn['reasoning']}</think>\n<tool_call>\n{arguments}\n</tool_call>"


def _tool_messages(
    initial: tuple[Mapping[str, Any], ...], controlled_call: str
) -> tuple[Mapping[str, Any], ...]:
    return initial + (
        {"role": "assistant", "content": controlled_call},
        {
            "role": "tool",
            "content": (
                {"type": "text", "text": IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT + "\n"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": "\n\n" + QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER,
                },
            ),
        },
    )


def _validate(config: Mapping[str, Any], processor: Any) -> dict[str, Any]:
    model = config["model"]
    if len(processor.tokenizer) != model["tokenizer_length"]:
        raise ValueError("tokenizer length mismatch")
    template_sha = _sha_text(processor.chat_template)
    if template_sha != model["chat_template_sha256"]:
        raise ValueError("chat template SHA256 mismatch")
    identity = visual_tool_prompt_identity(PROFILE, assistant_dialect=DIALECT)
    prompt_config = config["prompt"]
    if (
        prompt_config["assistant_dialect"] != DIALECT.value
        or prompt_config["version"] != identity.version
        or prompt_config["response_version"] != identity.response_version
        or prompt_config["bundle_sha256"] != identity.bundle_sha256
    ):
        raise ValueError("V4 Instruct prompt identity mismatch")
    image_path = Path(config["sample"]["image_path"])
    if _sha_bytes(image_path.read_bytes()) != config["sample"]["image_sha256"]:
        raise ValueError("source image SHA256 mismatch")

    schemas = tuple(build_native_tool_schemas(PROFILE.tool_names))
    renderer = NativeProtocolRenderer(
        processor,
        expected_tokenizer_length=model["tokenizer_length"],
        tool_names=PROFILE.tool_names,
        tool_schemas=schemas,
        assistant_dialect=DIALECT,
    )
    initial_messages = build_visual_tool_prompt_messages(
        config["sample"]["question"],
        tool_profile=PROFILE,
        assistant_dialect=DIALECT,
    )
    initial = renderer.render(initial_messages, add_generation_prompt=True)
    renderer.assert_generation_prefill(initial, processor.tokenizer)
    controlled = _controlled_call(config)
    parsed = StrictToolCallParser(enabled_tool_names=PROFILE.tool_names).parse(
        _sampled_turn(processor.tokenizer, controlled)
    )
    followup_messages = _tool_messages(initial_messages, controlled)
    followup = renderer.render(followup_messages, add_generation_prompt=True)
    renderer.assert_generation_prefill(followup, processor.tokenizer)
    exact_expected = (
        initial.text
        + controlled
        + render_qwen_native_success_environment_text(
            parsed,
            assistant_dialect=DIALECT,
        )
    )
    if followup.text != exact_expected:
        raise ValueError("rerendered post-crop prompt differs from native append bytes")
    if not followup.text.endswith("<|im_start|>assistant\n"):
        raise ValueError("Instruct follow-up has an unexpected assistant prefill")

    with Image.open(image_path) as opened:
        width, height = opened.size
    mapping = map_qwen3_crop_bbox_to_source(
        tuple(config["controlled_tool_turn"]["bbox_2d"]),
        source_width=width,
        source_height=height,
    )
    return {
        "prompt_bundle_sha256": identity.bundle_sha256,
        "initial_prompt_text_sha256": initial.text_sha256,
        "initial_prompt_token_ids_sha256": initial.token_ids_sha256,
        "post_crop_prompt_text_sha256": followup.text_sha256,
        "post_crop_prompt_token_ids_sha256": followup.token_ids_sha256,
        "post_crop_prompt_exact_native_append": True,
        "assistant_prefill": "<|im_start|>assistant\\n",
        "source_size": [width, height],
        "source_bbox_2d": list(mapping.source_bbox_2d),
        "controlled_call": controlled,
        "initial_messages": initial_messages,
        "followup_messages": followup_messages,
        "initial_prompt_text": initial.text,
        "post_crop_prompt_text": followup.text,
    }


def _analyze(text: str, token_ids: list[int], max_tokens: int) -> dict[str, Any]:
    stripped = text.lstrip()
    token_count = len(token_ids)
    opener_count = text.count("<think>")
    closer_count = text.count("</think>")
    tool_open_count = text.count("<tool_call>")
    tool_close_count = text.count("</tool_call>")
    valid_think = (
        stripped.startswith("<think>")
        and opener_count == 1
        and closer_count == 1
        and text.index("<think>") < text.index("</think>")
    )
    max_tokens_reached = token_count >= max_tokens
    if text.rstrip().endswith("</tool_call>"):
        termination = "tool_stop_string"
    elif max_tokens_reached:
        termination = "max_tokens"
    else:
        termination = "eos_or_generation_stop"
    return {
        "text": text,
        "text_sha256": _sha_text(text),
        "token_ids": token_ids,
        "token_ids_sha256": _sha_bytes(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ),
        "token_count": token_count,
        "think_opener_count": opener_count,
        "think_closer_count": closer_count,
        "starts_with_think": stripped.startswith("<think>"),
        "valid_single_think": valid_think,
        "tool_open_count": tool_open_count,
        "tool_close_count": tool_close_count,
        "has_tool_action": tool_open_count > 0 or tool_close_count > 0,
        "valid_tool_envelope_count": tool_open_count == tool_close_count == 1,
        "termination": termination,
        "max_tokens_reached": max_tokens_reached,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    token_counts = [int(record["token_count"]) for record in records]
    terminations: dict[str, int] = {}
    for record in records:
        termination = str(record["termination"])
        terminations[termination] = terminations.get(termination, 0) + 1
    return {
        "samples": count,
        "starts_with_think": sum(bool(r["starts_with_think"]) for r in records),
        "valid_single_think": sum(bool(r["valid_single_think"]) for r in records),
        "think_openers": sum(int(r["think_opener_count"]) for r in records),
        "think_closers": sum(int(r["think_closer_count"]) for r in records),
        "has_tool_action": sum(bool(r["has_tool_action"]) for r in records),
        "valid_tool_envelope": sum(
            bool(r["valid_tool_envelope_count"]) for r in records
        ),
        "max_tokens_reached": sum(bool(r["max_tokens_reached"]) for r in records),
        "termination_counts": terminations,
        "sampled_tokens_total": sum(token_counts),
        "sampled_tokens_mean": mean(token_counts),
        "sampled_tokens_min": min(token_counts),
        "sampled_tokens_max": max(token_counts),
    }


def _generate(
    *,
    model: Any,
    processor: Any,
    prompt_text: str,
    images: list[Image.Image],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    sampling = config["sampling"]
    size = processor.image_processor.size
    batch = processor(
        text=[prompt_text],
        images=images,
        padding=False,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": size["shortest_edge"],
                "longest_edge": config["model"]["image_max_pixels"],
            }
        },
    ).to("cuda:0")
    input_length = int(batch["input_ids"].shape[1])
    records = []
    for seed in sampling["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        started = perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **batch,
                do_sample=sampling["do_sample"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                max_new_tokens=sampling["max_new_tokens_per_turn"],
                stop_strings=sampling["stop_strings"],
                tokenizer=processor.tokenizer,
            )
        generated = output[0, input_length:].detach().cpu().tolist()
        text = processor.tokenizer.decode(generated, skip_special_tokens=False)
        record = _analyze(text, generated, sampling["max_new_tokens_per_turn"])
        record.update({"seed": seed, "elapsed_seconds": perf_counter() - started})
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, config_sha = _load_config(args.config.resolve())
    if args.physical_gpu != config["execution"]["physical_gpu"]:
        raise ValueError("physical GPU differs from config")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config["model"]["path"], local_files_only=True, trust_remote_code=False
    )
    golden = _validate(config, processor)
    public_golden = {
        k: v
        for k, v in golden.items()
        if not k.endswith("messages") and not k.endswith("text")
    }
    public_golden.update({"config_sha256": config_sha, "validated": True})
    if args.validate_only:
        print(json.dumps(public_golden, ensure_ascii=False, sort_keys=True))
        return 0

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("canary requires exactly one CUDA-visible GPU")
    output_root = Path(config["execution"]["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite canary output: {output_root}")
    output_root.mkdir(parents=True)
    source_path = Path(config["sample"]["image_path"])
    source = Image.open(source_path).convert("RGB")
    source_bbox = tuple(public_golden["source_bbox_2d"])
    crop = source.crop(source_bbox)
    crop_path = output_root / "controlled_crop.png"
    crop.save(crop_path)

    from transformers import AutoModelForImageTextToText

    model = (
        AutoModelForImageTextToText.from_pretrained(
            config["model"]["path"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation=config["model"]["attention"],
            low_cpu_mem_usage=True,
        )
        .to("cuda:0")
        .eval()
    )
    initial_records = _generate(
        model=model,
        processor=processor,
        prompt_text=golden["initial_prompt_text"],
        images=[source],
        config=config,
    )
    followup_records = _generate(
        model=model,
        processor=processor,
        prompt_text=golden["post_crop_prompt_text"],
        images=[source, crop],
        config=config,
    )
    result = {
        **public_golden,
        "schema_version": SCHEMA,
        "physical_gpu": args.physical_gpu,
        "crop_path": str(crop_path),
        "crop_sha256": _sha_bytes(crop_path.read_bytes()),
        "initial_prompt": initial_records,
        "post_successful_crop": followup_records,
        "metrics": {
            "initial_prompt": _summarize(initial_records),
            "post_successful_crop": _summarize(followup_records),
        },
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(result_path),
                "result_sha256": _sha_bytes(result_path.read_bytes()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
