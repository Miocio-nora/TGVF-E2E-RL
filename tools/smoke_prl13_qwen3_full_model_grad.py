#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Real single-GPU gradient gate for PRL13's native full-model path.

This is deliberately not an RL update.  It builds one official visible
DeepEyes original+Crop transcript with the real Qwen3-VL processor, supervises
only the final assistant answer, and proves that native pixels backpropagate
through the vision tower, visual merger, and language model.  The later veRL
smoke remains responsible for FSDP, vLLM weight sync, checkpointing, and
resume.
"""

from __future__ import annotations

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/smoke_prl13_qwen3_full_model_grad.py",
        ),
    )

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from PIL import Image

from tgvf_rl.policy.deepeyes_official_protocol import (
    SYSTEM_PROMPT_V2,
    USER_PROMPT_V2,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


SCHEMA = "tgvf.prl13-qwen3-full-model-gradient-smoke.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    fields = value.split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("bbox must be x1,y1,x2,y2")
    try:
        left, top, right, bottom = (int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox coordinates must be integers") from error
    if not (left < right and top < bottom):
        raise argparse.ArgumentTypeError("bbox must be non-empty")
    return left, top, right, bottom


def _messages(
    *,
    question: str,
    answer: str | None,
    bbox: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    bbox_json = json.dumps(list(bbox), separators=(",", ":"))
    tool_call = (
        "<think>I need a closer view of the relevant region.</think>"
        '<tool_call>{"name":"image_zoom_in_tool","arguments":'
        f'{{"bbox_2d":{bbox_json},"label":"relevant region"}}'
        "}</tool_call>"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question + USER_PROMPT_V2},
            ],
        },
        {"role": "assistant", "content": tool_call},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_response>"},
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT_V2 + "</tool_response>"},
            ],
        },
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "<think>The crop provides the needed visual evidence.</think>"
                    f"{answer}"
                ),
            }
        )
    return messages


def _group_for_parameter(name: str) -> str | None:
    normalized = name.casefold()
    if "model.visual" in normalized or normalized.startswith("visual"):
        if any(
            marker in normalized for marker in ("merger", "projector", "projection")
        ):
            return "projection"
        return "vision"
    if any(
        marker in normalized
        for marker in ("language_model", "model.layers", "lm_head", "embed_tokens")
    ):
        return "language"
    return None


def _gradient_records(model: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {
        name: {
            "parameter_tensors": 0,
            "gradient_tensors": 0,
            "nonzero_gradient_tensors": 0,
            "squared_norm": 0.0,
            "representative": None,
        }
        for name in ("vision", "projection", "language")
    }
    representatives: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        group = _group_for_parameter(name)
        if group is None or not parameter.requires_grad:
            continue
        record = groups[group]
        record["parameter_tensors"] += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        record["gradient_tensors"] += 1
        detached = gradient.detach().float()
        norm = float(detached.norm().item())
        if not math.isfinite(norm):
            raise RuntimeError(f"non-finite {group} gradient: {name}")
        record["squared_norm"] += norm * norm
        if norm > 0.0:
            record["nonzero_gradient_tensors"] += 1
            current = representatives.get(group)
            if current is None or norm > current["tensor_norm"]:
                flat_index = int(detached.abs().argmax().item())
                representatives[group] = {
                    "name": name,
                    "parameter": parameter,
                    "flat_index": flat_index,
                    "tensor_norm": norm,
                    "gradient_value": float(detached.flatten()[flat_index].item()),
                    "before": float(
                        parameter.detach().flatten()[flat_index].float().item()
                    ),
                }
    public: dict[str, dict[str, Any]] = {}
    for group, record in groups.items():
        total = math.sqrt(float(record.pop("squared_norm")))
        representative = representatives.get(group)
        public[group] = {
            **record,
            "gradient_norm": total,
            "representative": None
            if representative is None
            else {
                key: value
                for key, value in representative.items()
                if key != "parameter"
            },
        }
        if total <= 0.0 or representative is None:
            raise RuntimeError(f"missing nonzero {group} gradient")
    return public, representatives


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/smoke_prl13_qwen3_full_model_grad.py"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--bbox", type=_parse_bbox, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-learning-rate", type=float, default=1.0e-2)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("gradient smoke requires exactly one CUDA-visible GPU")
    model_path = args.model.resolve(strict=True)
    image_path = args.image.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _sha256_file(image_path) != args.image_sha256:
        raise ValueError("source image SHA-256 differs")
    if not math.isfinite(args.smoke_learning_rate) or args.smoke_learning_rate <= 0:
        raise ValueError("smoke learning rate must be positive and finite")

    started = perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    with Image.open(image_path) as opened:
        original = opened.convert("RGB").copy()
    crop = original.crop(args.bbox)
    if crop.width <= 30 or crop.height <= 30:
        raise ValueError("smoke crop must satisfy DeepEyes' >30 pixel rule")
    images = [original, crop]

    prefix_messages = _messages(
        question=args.question,
        answer=None,
        bbox=args.bbox,
    )
    full_messages = _messages(
        question=args.question,
        answer=args.answer,
        bbox=args.bbox,
    )
    prefix_text = processor.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix_inputs = processor(
        text=[prefix_text], images=images, padding=False, return_tensors="pt"
    )
    inputs = processor(
        text=[full_text], images=images, padding=False, return_tensors="pt"
    )
    prefix_length = int(prefix_inputs["input_ids"].shape[1])
    input_length = int(inputs["input_ids"].shape[1])
    if not 0 < prefix_length < input_length:
        raise RuntimeError("final answer supervision span is empty")
    if tuple(inputs["image_grid_thw"].shape) != (2, 3):
        raise RuntimeError("actor processor did not preserve original+Crop ordering")
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_length] = -100
    inputs["labels"] = labels
    inputs = inputs.to("cuda:0")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.train()
    model.config.use_cache = False
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("full-model smoke found frozen parameters")
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.smoke_learning_rate)
    optimizer.zero_grad(set_to_none=True)
    result = model(**inputs)
    loss = float(result.loss.detach().float().item())
    if not math.isfinite(loss):
        raise RuntimeError("full-model smoke loss is non-finite")
    result.loss.backward()
    gradients, representatives = _gradient_records(model)
    optimizer.step()

    for group, representative in representatives.items():
        after = float(
            representative["parameter"]
            .detach()
            .flatten()[representative["flat_index"]]
            .float()
            .item()
        )
        gradients[group]["representative"]["after"] = after
        gradients[group]["representative"]["changed"] = (
            after != representative["before"]
        )
        if after == representative["before"]:
            raise RuntimeError(f"representative {group} weight did not change")

    record = {
        "schema_version": SCHEMA,
        "scientific_rl_update": False,
        "scope": "real_qwen3_native_pixel_gradient_path_only",
        "model_path": str(model_path),
        "model_config_sha256": _sha256_file(model_path / "config.json"),
        "image_path": str(image_path),
        "image_sha256": args.image_sha256,
        "original_size": list(original.size),
        "crop_bbox": list(args.bbox),
        "crop_size": list(crop.size),
        "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
        "prompt_tokens": prefix_length,
        "total_tokens": input_length,
        "supervised_tokens": input_length - prefix_length,
        "loss": loss,
        "gradients": gradients,
        "smoke_optimizer": "sgd",
        "smoke_learning_rate": args.smoke_learning_rate,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": perf_counter() - started,
        "real_fsdp_vllm_sync_proven": False,
        "checkpoint_resume_proven": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
