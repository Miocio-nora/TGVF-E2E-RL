from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.representation.experiments.image_axis_grounding.config import (
    ImageAxisGroundingObjectiveConfig,
    image_axis_treatment_objective_identity,
    load_image_axis_grounding_experiment_config,
)
from tgvf_rl.representation.training.objective import RepresentationObjectiveKind


def test_treatment_objective_identity_binds_every_experimental_input() -> None:
    objective = ImageAxisGroundingObjectiveConfig(
        image_axis_matrix_weight=1.0,
        image_axis_temperature=1.0,
        negative_count=1,
    )

    assert image_axis_treatment_objective_identity(
        base_training_config_sha256="a" * 64,
        donor_manifest_sha256="b" * 64,
        objective=objective,
    ) == (
        "balanced-matrix-ce-l-gen-norm-plus-image-axis-v1:"
        f"base={'a' * 64}:donor={'b' * 64}:"
        "weight=0x1.0000000000000p+0:"
        "temperature=0x1.0000000000000p+0:negatives=1"
    )


def test_no_matrix_ce_treatment_identity_names_the_ablation() -> None:
    identity = image_axis_treatment_objective_identity(
        base_training_config_sha256="a" * 64,
        donor_manifest_sha256="b" * 64,
        objective=ImageAxisGroundingObjectiveConfig(),
        base_objective_kind=(
            RepresentationObjectiveKind.L_GEN_AND_NORM_NO_MATRIX_CE_ABLATION
        ),
    )

    assert identity.startswith(
        "l-gen-norm-no-matrix-ce-plus-image-axis-ablation-v1:"
    )


def test_outer_config_rejects_unknown_root_field_before_bound_file_access(
    tmp_path: Path,
) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(
        'schema_version = "image-axis-grounding-config-v1"\n'
        'scope = "isolated_representation_image_axis_grounding"\n'
        'run_id = "run"\n'
        'unexpected = "not-allowed"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[root\] fields differ"):
        load_image_axis_grounding_experiment_config(config)


@pytest.mark.parametrize(
    ("weight", "temperature", "negative_count"),
    ((0.5, 1.0, 1), (1.0, 0.5, 1), (1.0, 1.0, 2)),
)
def test_v1_objective_has_no_unplanned_tuning_surface(
    weight: float,
    temperature: float,
    negative_count: int,
) -> None:
    with pytest.raises(ValueError, match="v1"):
        ImageAxisGroundingObjectiveConfig(
            image_axis_matrix_weight=weight,
            image_axis_temperature=temperature,
            negative_count=negative_count,
        )
