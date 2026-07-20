#!/usr/bin/env python3
"""Materialize the fixed Qwen3 representation internal-evaluation population."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from tgvf_rl.representation.training.data import load_retained_representation_jsonl


VALIDATION_PATH = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl"
)
VALIDATION_SOURCE_SHA256 = (
    "a228d28db76625d166dab874806c9034a244a683d41c7cecdc7f10f1aa754308"
)
RETAINED_MANIFEST_SHA256 = (
    "f47bbff7c63ffa381ce2e2e263130c783057c6d408575ef4c4e3dd5b019c5a33"
)
MODEL_PATH = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
IMAGE_MAX_PIXELS = 262144
EXPECTED_GRID = (1, 26, 38)
SEED = 42
PAIR_A = "tgvf_v3_teacher_val_2k:chartqa:train_004209:2"
PAIR_B = "tgvf_v3_teacher_val_2k:chartqa:train_010577:0"
OUTPUT_DIRECTORY = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/representation/"
    "internal_evaluation"
)


def main() -> None:
    dataset = load_retained_representation_jsonl(
        VALIDATION_PATH,
        expected_source_sha256=VALIDATION_SOURCE_SHA256,
        warn_on_leakage=False,
    )
    if dataset.manifest.manifest_sha256 != RETAINED_MANIFEST_SHA256:
        raise RuntimeError("retained validation manifest changed")
    groups: dict[str, list[object]] = defaultdict(list)
    for sample in dataset.samples:
        groups[sample.image_group_key].append(sample)
    exact_k4 = {key: values for key, values in groups.items() if len(values) == 4}

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    image_processor_size = processor.image_processor.size
    image_size = {
        "shortest_edge": image_processor_size["shortest_edge"],
        "longest_edge": IMAGE_MAX_PIXELS,
    }
    selected: list[tuple[str, list[object]]] = []
    for key, samples in exact_k4.items():
        with Image.open(samples[0].image) as image:
            batch = processor(
                text=["<|vision_start|><|image_pad|><|vision_end|>"],
                images=[image.convert("RGB")],
                return_tensors="pt",
                images_kwargs={"size": image_size},
            )
        grid = tuple(int(value) for value in batch["image_grid_thw"][0].tolist())
        if grid == EXPECTED_GRID:
            selected.append((key, samples))
    selected.sort(
        key=lambda item: sha256(
            (
                "representation-internal-eval-groups-v1\0"
                + RETAINED_MANIFEST_SHA256
                + "\0"
                + str(SEED)
                + "\0"
                + item[0]
            ).encode("utf-8")
        ).hexdigest()
    )
    selected_ids = {sample.sample_id for _, samples in selected for sample in samples}
    if len(selected) != 31 or len(selected_ids) != 124:
        raise RuntimeError(
            f"expected 31 exact-K4 groups/124 rows, got {len(selected)}/{len(selected_ids)}"
        )
    if not {PAIR_A, PAIR_B}.issubset(selected_ids):
        raise RuntimeError("fixed counterfactual pair is absent from selected groups")

    ordered_payload = {
        "schema_version": "representation_internal_evaluation_group_manifest_v1",
        "identity": "qwen3-v3-val2k-grid1x26x38-exact-k4-seed42-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "groups": [
            {
                "image_group_key": key,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "content_sha256": sample.content_sha256,
                    }
                    for sample in samples
                ],
            }
            for key, samples in selected
        ],
    }
    counterfactual_payload = {
        "schema_version": "qwen3_counterfactual_manifest_v1",
        "identity": "qwen3-v3-val2k-chartqa-2016-value-pair-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "pairs": [
            {
                "schema_version": "qwen3_counterfactual_pair_v1",
                "pair_id": "chartqa-2016-value-262-vs-38",
                "sample_a_id": PAIR_A,
                "sample_b_id": PAIR_B,
                "expected_value_a": "262",
                "expected_value_b": "38",
                "pair_audit_identity": (
                    "v3-val2k-exact-question-target-distinct-image-"
                    "matched-qwen3-grid-v1"
                ),
            }
        ],
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputs = {
        OUTPUT_DIRECTORY / "qwen3_v3_val2k_grid1x26x38_exact_k4_seed42_v1.json": (
            ordered_payload
        ),
        OUTPUT_DIRECTORY / "qwen3_v3_val2k_chartqa_2016_counterfactual_v1.json": (
            counterfactual_payload
        ),
    }
    for path, payload in outputs.items():
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite changed manifest: {path}")
        path.write_text(encoded, encoding="utf-8")
        print(f"{sha256(encoded.encode('utf-8')).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
