"""Two-rank launcher that injects the isolated image-axis builder and trainer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator

from tgvf_rl.representation.training import runner as core_runner
from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.trainer import RepresentationTrainer
from tgvf_rl.representation.training.validation_identity import (
    build_retained_image_raw_byte_manifest,
)

from .config import (
    ImageAxisGroundingExperimentConfig,
    load_image_axis_grounding_experiment_config,
)
from .matching import (
    ImageAxisDonorManifest,
    load_image_axis_donor_manifest,
    load_qwen_image_grid_contract,
)
from .native_pipeline import ImageAxisGroundedNativeGroupBuilder
from .trainer import ImageAxisGroundingTrainer


IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION = "image-axis-grounding-runner-v1"
RP66_USABLE_IMAGE_GROUP_COUNT = 8_209


def validate_image_axis_grounding_experiment(
    config_path: str | Path,
) -> dict[str, object]:
    """Perform every CPU-side identity check without touching CUDA."""

    config = load_image_axis_grounding_experiment_config(config_path)
    manifest = _load_and_validate_manifest(config)
    return {
        **config.validation_payload(),
        "runner_schema_version": IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION,
        "donor_manifest_identity_sha256": manifest.identity_sha256,
        "donor_assignment_count": len(manifest.assignments),
        "donor_matched_count": sum(
            assignment.matched for assignment in manifest.assignments
        ),
        "donor_masked_count": sum(
            not assignment.matched for assignment in manifest.assignments
        ),
    }


def run_image_axis_grounding_experiment(
    config_path: str | Path,
    *,
    stop_after_global_step: int | None = None,
) -> dict[str, object] | None:
    """Run the strict inner FSDP2 job with removable process-local injection."""

    config = load_image_axis_grounding_experiment_config(config_path)
    manifest = _load_and_validate_manifest(config)
    with _inject_image_axis_components(config, manifest):
        result = core_runner.run_representation_training(
            config.treatment_training_config_path,
            stop_after_global_step=stop_after_global_step,
        )
    if result is None:
        return None
    return {
        "schema_version": IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION,
        "status": result.get("status"),
        "experiment_run_id": config.run_id,
        "experiment_config_sha256": config.source_toml_sha256,
        "experiment_canonical_sha256": config.canonical_config_sha256,
        "base_training_config_sha256": config.base_training_config_sha256,
        "treatment_training_config_sha256": (
            config.treatment_training_config_sha256
        ),
        "donor_manifest_sha256": config.donor_manifest_sha256,
        "donor_manifest_identity_sha256": manifest.identity_sha256,
        "image_axis_objective": asdict(config.objective),
        "core_result": result,
    }


def _load_and_validate_manifest(
    config: ImageAxisGroundingExperimentConfig,
) -> ImageAxisDonorManifest:
    """Rebuild every cheap source identity before any CUDA initialization.

    The outer config's file SHA proves which donor manifest was requested, but
    it does not by itself prove that the manifest still describes the bytes
    consumed by the treatment.  Re-reading the retained JSONL, hashing every
    retained image, binding the local processor config, and reconstructing the
    exact two-rank K=4 sampler closure makes that relationship fail closed.
    """

    manifest = load_image_axis_donor_manifest(config.donor_manifest_path)
    source = manifest.source_binding
    training = config.treatment_training
    if source.train_source_sha256 != training.data.train.source_sha256:
        raise ValueError("donor manifest binds a different train JSONL")

    train_data = load_retained_representation_jsonl(
        training.data.train.jsonl_path,
        expected_source_sha256=training.data.train.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    if source.retained_manifest_sha256 != train_data.manifest.manifest_sha256:
        raise ValueError("donor manifest retained-data SHA256 mismatch")

    raw_images = build_retained_image_raw_byte_manifest(train_data.manifest)
    if source.raw_image_manifest_sha256 != raw_images.manifest_sha256:
        raise ValueError("donor manifest raw-image SHA256 mismatch")

    preprocessor_path = training.model.local_path / "preprocessor_config.json"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Qwen preprocessor config is missing: {preprocessor_path}"
        )
    preprocessor_sha256 = sha256(preprocessor_path.read_bytes()).hexdigest()
    if source.preprocessor_config_sha256 != preprocessor_sha256:
        raise ValueError("donor manifest preprocessor-config SHA256 mismatch")
    observed_grid_contract = load_qwen_image_grid_contract(
        preprocessor_path,
        image_max_pixels=training.model.image_max_pixels,
    )
    if observed_grid_contract != manifest.grid_contract:
        raise ValueError("donor manifest Qwen image-grid contract mismatch")

    if training.fsdp2.world_size != 2 or training.data.train.batch_size != 4:
        raise ValueError("RP66 image-axis preflight requires world_size=2 and K=4")
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
    usable_group_keys = tuple(
        sorted(key for sampler in samplers for key in sampler.owned_group_keys)
    )
    if len(usable_group_keys) != len(set(usable_group_keys)):
        raise RuntimeError("distributed sampler assigned one image group twice")
    if len(usable_group_keys) != RP66_USABLE_IMAGE_GROUP_COUNT:
        raise ValueError(
            "RP66 usable image-group population changed: expected "
            f"{RP66_USABLE_IMAGE_GROUP_COUNT}, got {len(usable_group_keys)}"
        )
    assignment_keys = tuple(
        assignment.anchor_image_group_key for assignment in manifest.assignments
    )
    if len(assignment_keys) != len(set(assignment_keys)):
        raise ValueError("donor manifest contains duplicate anchor assignments")
    if assignment_keys != usable_group_keys:
        missing = sorted(set(usable_group_keys) - set(assignment_keys))
        extra = sorted(set(assignment_keys) - set(usable_group_keys))
        raise ValueError(
            "donor assignments differ from the exact world2/K4 sampler closure: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return manifest


@contextmanager
def _inject_image_axis_components(
    config: ImageAxisGroundingExperimentConfig,
    manifest: ImageAxisDonorManifest,
) -> Iterator[None]:
    """Patch only the two construction seams used by the core process runner."""

    original_builder = core_runner.Qwen3NativeRepresentationGroupBuilder
    original_trainer = core_runner.RepresentationTrainer
    if original_builder is not Qwen3NativeRepresentationGroupBuilder:
        raise RuntimeError("core group-builder seam was already patched")
    if original_trainer is not RepresentationTrainer:
        raise RuntimeError("core trainer seam was already patched")

    def group_builder_factory(**kwargs: Any) -> ImageAxisGroundedNativeGroupBuilder:
        base = original_builder(**kwargs)
        return ImageAxisGroundedNativeGroupBuilder(
            base_builder=base,
            donor_manifest=manifest,
        )

    def trainer_factory(**kwargs: Any) -> ImageAxisGroundingTrainer:
        return ImageAxisGroundingTrainer(
            **kwargs,
            image_axis_objective=config.objective,
        )

    core_runner.Qwen3NativeRepresentationGroupBuilder = group_builder_factory  # type: ignore[assignment]
    core_runner.RepresentationTrainer = trainer_factory  # type: ignore[assignment]
    try:
        yield
    finally:
        core_runner.RepresentationTrainer = original_trainer
        core_runner.Qwen3NativeRepresentationGroupBuilder = original_builder


__all__ = [
    "IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION",
    "RP66_USABLE_IMAGE_GROUP_COUNT",
    "run_image_axis_grounding_experiment",
    "validate_image_axis_grounding_experiment",
]
