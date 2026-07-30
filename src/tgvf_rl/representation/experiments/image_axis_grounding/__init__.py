"""Removable image-axis grounding ablation for RP66 representation training."""

from .config import (
    IMAGE_AXIS_GROUNDING_CONFIG_SCHEMA_VERSION,
    IMAGE_AXIS_GROUNDING_SCOPE,
    ImageAxisGroundingExperimentConfig,
    image_axis_treatment_objective_identity,
    load_image_axis_grounding_experiment_config,
)
from .matching import (
    IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION,
    IMAGE_AXIS_DONOR_MATCHING_RULE,
    ImageAxisDonorAssignment,
    ImageAxisDonorManifest,
    ImageAxisDonorSourceBinding,
    QwenImageGridContract,
    build_image_axis_donor_manifest,
    load_image_axis_donor_manifest,
    load_qwen_image_grid_contract,
    materialize_image_axis_donor_manifest,
    qwen_image_grid_thw,
)
from .native_pipeline import (
    ImageAxisGroundedNativeGroupBuilder,
    ImageAxisGroundingGroup,
)
from .runner import (
    IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION,
    RP66_USABLE_IMAGE_GROUP_COUNT,
    run_image_axis_grounding_experiment,
    validate_image_axis_grounding_experiment,
)
from .trainer import ImageAxisGroundingObjectiveConfig, ImageAxisGroundingTrainer

__all__ = [
    "IMAGE_AXIS_GROUNDING_CONFIG_SCHEMA_VERSION",
    "IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION",
    "IMAGE_AXIS_GROUNDING_SCOPE",
    "IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION",
    "IMAGE_AXIS_DONOR_MATCHING_RULE",
    "ImageAxisDonorAssignment",
    "ImageAxisDonorManifest",
    "ImageAxisDonorSourceBinding",
    "ImageAxisGroundedNativeGroupBuilder",
    "ImageAxisGroundingExperimentConfig",
    "ImageAxisGroundingGroup",
    "ImageAxisGroundingObjectiveConfig",
    "ImageAxisGroundingTrainer",
    "QwenImageGridContract",
    "RP66_USABLE_IMAGE_GROUP_COUNT",
    "build_image_axis_donor_manifest",
    "load_image_axis_donor_manifest",
    "load_image_axis_grounding_experiment_config",
    "load_qwen_image_grid_contract",
    "materialize_image_axis_donor_manifest",
    "qwen_image_grid_thw",
    "image_axis_treatment_objective_identity",
    "run_image_axis_grounding_experiment",
    "validate_image_axis_grounding_experiment",
]
