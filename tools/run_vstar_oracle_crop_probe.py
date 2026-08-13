#!/usr/bin/env python3
"""Run the isolated VStar oracle-crop visibility probe.

This diagnostic deliberately does not enter the policy tool runtime.  It asks
one frozen Qwen checkpoint to answer the same VStar question from either the
original image, the original plus an oracle crop, or the original plus a
same-size gray placebo.  The oracle box comes only from VStar's held-out
sidecar annotation and is never exposed as text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.vstar_oracle_crop_probe import (  # noqa: E402
    DEFAULT_MEDIUM_CONTROL_IDS,
    DEFAULT_TINY_PRIMARY_IDS,
    build_vstar_oracle_probe_cases,
    extract_exact_option_label,
    make_oracle_crop_pair,
)


DEFAULT_VSTAR_ROOT = Path("/nvmesv/dredvpn009/datasets/benchmarks/vstar_bench/snapshot")
DEFAULT_MODEL = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-"
    "8step-ws8/evaluation/PRL19-R0-FROZEN-RP67-TFREE-VISUAL-API-"
    "COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/step8/runtime/"
    "qwen-only-bundle/model"
)
DEFAULT_TOKENIZER = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
DEFAULT_OUTPUT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/"
    "DIAG-VSTAR-ORACLE-CROP-STEP8-FG-V1"
)
ARMS = ("original", "oracle_crop", "gray_placebo")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vstar-root", type=Path, default=DEFAULT_VSTAR_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--image-max-pixels", type=int, default=1_003_520)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-cases",
        type=int,
        help=(
            "execute only the first N fixed cases for a bounded smoke; the "
            "manifest always retains all 32 cases"
        ),
    )
    parser.add_argument("--mode", choices=("prepare", "run"), default="run")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _question(case: Any) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in case.options)
    return (
        f"Question: {case.question}\nOptions:\n{options}\n"
        "Please inspect the supplied image evidence and select the correct answer. "
        "End your response with the option letter."
    )


def _messages(question: str, *, image_count: int) -> list[dict[str, object]]:
    if image_count not in {1, 2}:
        raise ValueError("oracle probe supports one or two images")
    content: list[dict[str, object]] = [
        {"type": "image", "image": "<image>"} for _ in range(image_count)
    ]
    if image_count == 2:
        content.append(
            {
                "type": "text",
                "text": (
                    "The first image is the full scene. The second image is a "
                    "magnified view of the region relevant to the question.\n"
                ),
            }
        )
    content.append({"type": "text", "text": question})
    return [
        {
            "role": "system",
            "content": "You are a helpful visual question answering assistant.",
        },
        {"role": "user", "content": content},
    ]


def _prepared_prompt(
    processor: Any,
    messages: list[dict[str, object]],
    images: list[Image.Image],
    image_max_pixels: int,
) -> dict[str, object]:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=[]
    )
    return {
        "prompt": text,
        "multi_modal_data": {"image": images},
        "mm_processor_kwargs": {
            "min_pixels": 65_536,
            "max_pixels": image_max_pixels,
            "do_resize": True,
        },
    }


def _prepare(args: argparse.Namespace) -> tuple[Any, ...]:
    cases = build_vstar_oracle_probe_cases(
        args.vstar_root,
        tiny_ids=DEFAULT_TINY_PRIMARY_IDS,
        medium_ids=DEFAULT_MEDIUM_CONTROL_IDS,
    )
    manifest = {
        "schema_version": "tgvf.vstar-oracle-crop-probe-manifest.v1",
        "purpose": "internal_visibility_diagnostic_only_not_formal_benchmark",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "vstar_root": str(args.vstar_root.resolve()),
        "test_questions_sha256": _sha256_file(args.vstar_root / "test_questions.jsonl"),
        "tiny_primary_ids": list(DEFAULT_TINY_PRIMARY_IDS),
        "medium_control_ids": list(DEFAULT_MEDIUM_CONTROL_IDS),
        "arms": list(ARMS),
        "crop_rule": "square_side=max(32,4*max(gt_width,gt_height)); shift_inside_source",
        "image_max_pixels": args.image_max_pixels,
        "sampling": {
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "cases": [case.as_manifest_record() for case in cases],
    }
    manifest["identity_sha256"] = _canonical_sha256(manifest)
    target = args.output_root / "manifest.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing oracle-probe manifest differs")
    else:
        _atomic_write_json(target, manifest)
    return cases


def _completed(path: Path) -> dict[tuple[int, str], dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[tuple[int, str], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (int(row["row_id"]), str(row["arm"]))
        if key in rows and rows[key] != row:
            raise RuntimeError("duplicate oracle-probe result differs")
        rows[key] = row
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, list[dict[str, object]]] = {
        arm: [row for row in rows if row["arm"] == arm] for arm in ARMS
    }
    arm_summary = {
        arm: {
            "count": len(values),
            "correct": sum(bool(value["correct"]) for value in values),
            "accuracy": (
                sum(bool(value["correct"]) for value in values) / len(values)
                if values
                else None
            ),
        }
        for arm, values in by_arm.items()
    }
    keyed = {(int(row["row_id"]), str(row["arm"])): row for row in rows}
    paired = []
    for row_id in sorted({int(row["row_id"]) for row in rows}):
        if all((row_id, arm) in keyed for arm in ARMS):
            original = bool(keyed[row_id, "original"]["correct"])
            oracle = bool(keyed[row_id, "oracle_crop"]["correct"])
            placebo = bool(keyed[row_id, "gray_placebo"]["correct"])
            paired.append(
                {
                    "row_id": row_id,
                    "original_correct": original,
                    "oracle_correct": oracle,
                    "placebo_correct": placebo,
                    "oracle_rescue": (not original) and oracle,
                    "placebo_rescue": (not original) and placebo,
                    "oracle_specific_rescue": (not original) and oracle and not placebo,
                }
            )
    return {
        "schema_version": "tgvf.vstar-oracle-crop-probe-summary.v1",
        "arms": arm_summary,
        "paired_complete": len(paired),
        "original_failures": sum(not row["original_correct"] for row in paired),
        "oracle_rescues": sum(row["oracle_rescue"] for row in paired),
        "placebo_rescues": sum(row["placebo_rescue"] for row in paired),
        "oracle_specific_rescues": sum(row["oracle_specific_rescue"] for row in paired),
        "paired": paired,
    }


def main() -> int:
    args = _parser().parse_args()
    cases = _prepare(args)
    if args.mode == "prepare":
        print(
            json.dumps(
                {"prepared": len(cases), "output_root": str(args.output_root)}, indent=2
            )
        )
        return 0

    if args.max_cases is not None and not 1 <= args.max_cases <= len(cases):
        raise ValueError(f"max-cases must be in [1, {len(cases)}]")
    execution_cases = cases[: args.max_cases] if args.max_cases is not None else cases

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=False,
        seed=args.seed,
        limit_mm_per_prompt={"image": 2, "video": 0},
        mm_processor_kwargs={
            "min_pixels": 65_536,
            "max_pixels": args.image_max_pixels,
            "do_resize": True,
        },
        generation_config="vllm",
        # Match the driver-portable vision path used by the project's
        # successful full-model CoreDev evaluations on B200.
        mm_encoder_attn_backend="TORCH_SDPA",
    )
    result_path = args.output_root / "results.jsonl"
    existing = _completed(result_path)
    pending: list[tuple[Any, str, list[Image.Image], dict[str, object]]] = []
    for case in execution_cases:
        original = Image.open(case.image_path).convert("RGB")
        crop_pair = make_oracle_crop_pair(original, case.gt_xywh)
        oracle = crop_pair.oracle
        placebo = crop_pair.placebo
        source_box = crop_pair.source_bbox_xyxy
        for arm, images in (
            ("original", [original.copy()]),
            ("oracle_crop", [original.copy(), oracle.copy()]),
            ("gray_placebo", [original.copy(), placebo.copy()]),
        ):
            if (case.row_id, arm) in existing:
                for image in images:
                    image.close()
                continue
            pending.append(
                (
                    case,
                    arm,
                    images,
                    {
                        "source_crop_xyxy": list(source_box),
                        "crop_size": list(oracle.size),
                    },
                )
            )
        original.close()
        oracle.close()
        placebo.close()

    if pending:
        prompts = [
            _prepared_prompt(
                processor,
                _messages(_question(case), image_count=len(images)),
                images,
                args.image_max_pixels,
            )
            for case, _arm, images, _extra in pending
        ]
        sampling = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            seed=args.seed,
            max_tokens=args.max_tokens,
            stop_token_ids=[151645],
            ignore_eos=False,
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        started = time.time()
        outputs = engine.generate(prompts, sampling, use_tqdm=True)
        if len(outputs) != len(pending):
            raise RuntimeError("oracle probe output count differs")
        for (case, arm, images, extra), output in zip(pending, outputs, strict=True):
            try:
                text = output.outputs[0].text
                options = dict(case.options)
                extracted = extract_exact_option_label(text, options)
                method = (
                    "vstar_oracle_exact_final_option_v1"
                    if extracted is not None
                    else "unparsed"
                )
                row = {
                    "schema_version": "tgvf.vstar-oracle-crop-probe-result.v1",
                    "row_id": case.row_id,
                    "stratum": case.stratum,
                    "arm": arm,
                    "question": case.question,
                    "options": options,
                    "gold": case.answer,
                    "prediction": text,
                    "extracted_option": extracted,
                    "extraction_method": method,
                    "correct": extracted == case.answer,
                    "image_path": str(case.image_path),
                    "gt_xywh": list(case.gt_xywh),
                    "target_object": case.target_object,
                    "bbox_area_fraction": case.bbox_area_fraction,
                    **extra,
                }
                row["result_identity_sha256"] = _canonical_sha256(row)
                _append_jsonl(result_path, row)
            finally:
                for image in images:
                    image.close()
        print(
            json.dumps(
                {"generated": len(pending), "wall_seconds": time.time() - started}
            )
        )

    rows = list(_completed(result_path).values())
    summary = _summary(rows)
    summary["manifest_identity_sha256"] = json.loads(
        (args.output_root / "manifest.json").read_text(encoding="utf-8")
    )["identity_sha256"]
    _atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
