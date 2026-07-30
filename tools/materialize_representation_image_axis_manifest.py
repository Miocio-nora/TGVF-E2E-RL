#!/usr/bin/env python3
"""Materialize the immutable RP66 image-axis wrong-image donor manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Sequence

from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorSourceBinding,
    build_image_axis_donor_manifest,
    load_qwen_image_grid_contract,
    materialize_image_axis_donor_manifest,
)
from tgvf_rl.representation.experiments.image_axis_grounding.runner import (
    RP66_USABLE_IMAGE_GROUP_COUNT,
)
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.validation_identity import (
    build_retained_image_raw_byte_manifest,
)


RP66_MATCHED_IMAGE_GROUP_COUNT = 8_173
RP66_MASKED_IMAGE_GROUP_COUNT = 36


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-training-config",
        type=Path,
        required=True,
        help="Strict RP66 core training TOML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON; an unequal existing file is never overwritten.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--expected-anchor-count", type=int, default=RP66_USABLE_IMAGE_GROUP_COUNT
    )
    parser.add_argument(
        "--expected-matched-count", type=int, default=RP66_MATCHED_IMAGE_GROUP_COUNT
    )
    parser.add_argument(
        "--expected-masked-count", type=int, default=RP66_MASKED_IMAGE_GROUP_COUNT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    training = load_representation_training_config(args.base_training_config)
    if training.fsdp2.world_size != 2 or training.data.train.batch_size != 4:
        raise ValueError("RP66 image-axis materialization requires world_size=2 and K=4")

    train_data = load_retained_representation_jsonl(
        training.data.train.jsonl_path,
        expected_source_sha256=training.data.train.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    samplers = tuple(
        SameImageBatchSampler(
            train_data.samples,
            batch_size=training.data.train.batch_size,
            seed=training.data.train.sampler_seed,
            data_manifest_sha256=train_data.manifest.manifest_sha256,
            rank=rank,
            world_size=training.fsdp2.world_size,
        )
        for rank in range(training.fsdp2.world_size)
    )
    usable_keys = frozenset(
        key for sampler in samplers for key in sampler.owned_group_keys
    )
    if len(usable_keys) != args.expected_anchor_count:
        raise ValueError(
            "usable anchor count mismatch: "
            f"expected {args.expected_anchor_count}, got {len(usable_keys)}"
        )
    anchor_samples = tuple(
        sample for sample in train_data.samples if sample.image_group_key in usable_keys
    )

    raw_images = build_retained_image_raw_byte_manifest(train_data.manifest)
    preprocessor_path = training.model.local_path / "preprocessor_config.json"
    preprocessor_bytes = preprocessor_path.read_bytes()
    grid_contract = load_qwen_image_grid_contract(
        preprocessor_path,
        image_max_pixels=training.model.image_max_pixels,
    )
    manifest = build_image_axis_donor_manifest(
        anchor_samples,
        train_data.samples,
        grid_contract=grid_contract,
        source_binding=ImageAxisDonorSourceBinding(
            train_source_sha256=training.data.train.source_sha256,
            retained_manifest_sha256=train_data.manifest.manifest_sha256,
            raw_image_manifest_sha256=raw_images.manifest_sha256,
            preprocessor_config_sha256=sha256(preprocessor_bytes).hexdigest(),
        ),
        random_seed=args.random_seed,
    )
    if manifest.matched_count != args.expected_matched_count:
        raise ValueError(
            "matched donor count mismatch: "
            f"expected {args.expected_matched_count}, got {manifest.matched_count}"
        )
    if manifest.unmatched_count != args.expected_masked_count:
        raise ValueError(
            "masked donor count mismatch: "
            f"expected {args.expected_masked_count}, got {manifest.unmatched_count}"
        )

    output = materialize_image_axis_donor_manifest(manifest, args.output)
    payload = {
        "status": "materialized",
        "output_path": str(output),
        "output_file_sha256": sha256(output.read_bytes()).hexdigest(),
        "manifest_identity_sha256": manifest.identity_sha256,
        "anchor_population_sha256": manifest.anchor_population_sha256,
        "donor_population_sha256": manifest.donor_population_sha256,
        "assignment_count": len(manifest.assignments),
        "matched_count": manifest.matched_count,
        "masked_count": manifest.unmatched_count,
        "match_tier_counts": dict(manifest.match_tier_counts),
        "source_binding": {
            "train_source_sha256": manifest.source_binding.train_source_sha256,
            "retained_manifest_sha256": (
                manifest.source_binding.retained_manifest_sha256
            ),
            "raw_image_manifest_sha256": (
                manifest.source_binding.raw_image_manifest_sha256
            ),
            "preprocessor_config_sha256": (
                manifest.source_binding.preprocessor_config_sha256
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
