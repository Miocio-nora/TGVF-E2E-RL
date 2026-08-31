"""CPU-preflighted launcher for the isolated RP70 span-supervision arm."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from tgvf_rl.representation.experiments.image_axis_grounding.config import (
    ImageAxisGroundingExperimentConfig,
)
from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorManifest,
)
from tgvf_rl.representation.experiments.image_axis_grounding.native_pipeline import (
    ImageAxisGroundedNativeGroupBuilder,
)
from tgvf_rl.representation.experiments.image_axis_grounding.runner import (
    _load_and_validate_manifest,
)
from tgvf_rl.representation.experiments.image_axis_grounding.trainer import (
    ImageAxisGroundingTrainer,
)
from tgvf_rl.representation.training import runner as core_runner
from tgvf_rl.representation.training.data import (
    load_retained_representation_jsonl,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.trainer import RepresentationTrainer

from .config import (
    AnswerBearingSpanExperimentConfig,
    load_answer_bearing_span_experiment_config,
)
from .data import (
    AnswerBearingSpanIndex,
    AnswerBearingSpanIndexSet,
    load_answer_bearing_span_index,
    merge_answer_bearing_span_indices,
)


ANSWER_BEARING_SPAN_RUNNER_SCHEMA_VERSION = "answer-bearing-span-runner-v1"


def validate_answer_bearing_span_experiment(
    config_path: str | Path,
) -> dict[str, object]:
    """Perform all config, donor, and span-index checks without touching CUDA."""

    config = load_answer_bearing_span_experiment_config(config_path)
    manifest = _load_and_validate_manifest(_as_image_axis_config(config))
    train_index, test_index, index_set = _load_span_index_set(config)
    return {
        **config.validation_payload(),
        "runner_schema_version": ANSWER_BEARING_SPAN_RUNNER_SCHEMA_VERSION,
        "donor_manifest_identity_sha256": manifest.identity_sha256,
        "donor_assignment_count": len(manifest.assignments),
        "donor_matched_count": sum(
            assignment.matched for assignment in manifest.assignments
        ),
        "donor_masked_count": sum(
            not assignment.matched for assignment in manifest.assignments
        ),
        **_span_index_payload(train_index, test_index, index_set),
    }


def run_answer_bearing_span_experiment(
    config_path: str | Path,
    *,
    stop_after_global_step: int | None = None,
) -> dict[str, object] | None:
    """Run RP70 through process-local builder/trainer seams and restore both."""

    config = load_answer_bearing_span_experiment_config(config_path)
    manifest = _load_and_validate_manifest(_as_image_axis_config(config))
    train_index, test_index, index_set = _load_span_index_set(config)
    loss_supervision_factory = _make_loss_supervision_factory(index_set)
    with _inject_answer_bearing_span_components(
        config,
        manifest,
        loss_supervision_factory=loss_supervision_factory,
    ):
        result = core_runner.run_representation_training(
            config.treatment_training_config_path,
            stop_after_global_step=stop_after_global_step,
        )
    if result is None:
        return None
    return {
        "schema_version": ANSWER_BEARING_SPAN_RUNNER_SCHEMA_VERSION,
        "status": result.get("status"),
        "experiment_run_id": config.run_id,
        "experiment_config_sha256": config.source_toml_sha256,
        "experiment_canonical_sha256": config.canonical_config_sha256,
        "base_training_config_sha256": config.base_training_config_sha256,
        "treatment_training_config_sha256": (config.treatment_training_config_sha256),
        "donor_manifest_sha256": config.donor_manifest_sha256,
        "donor_manifest_identity_sha256": manifest.identity_sha256,
        "span_policy": config.span.validation_payload(),
        "image_axis_objective": asdict(config.objective),
        **_span_index_payload(train_index, test_index, index_set),
        "core_result": result,
    }


def _as_image_axis_config(
    config: AnswerBearingSpanExperimentConfig,
) -> ImageAxisGroundingExperimentConfig:
    """Expose the exact structural fields required by RP67 donor preflight.

    RP67's preflight is intentionally reused rather than copied.  Its runtime
    implementation is structural, but this explicit adapter keeps static type
    drift visible and prevents RP70 from inheriting RP67's objective identity.
    """

    return ImageAxisGroundingExperimentConfig(
        schema_version="image-axis-grounding-config-v1",
        scope="isolated_representation_image_axis_grounding",
        run_id=config.run_id,
        base_training_config_path=config.base_training_config_path,
        base_training_config_sha256=config.base_training_config_sha256,
        treatment_training_config_path=config.treatment_training_config_path,
        treatment_training_config_sha256=config.treatment_training_config_sha256,
        donor_manifest_path=config.donor_manifest_path,
        donor_manifest_sha256=config.donor_manifest_sha256,
        objective=config.objective,
        source_path=config.source_path,
        source_toml_sha256=config.source_toml_sha256,
        canonical_config_sha256=config.canonical_config_sha256,
        base_training=config.base_training,
        treatment_training=config.treatment_training,
    )


def _load_span_index_set(
    config: AnswerBearingSpanExperimentConfig,
) -> tuple[AnswerBearingSpanIndex, AnswerBearingSpanIndex, AnswerBearingSpanIndexSet]:
    """Index both complete UID populations under semantic sidecar binding."""

    training = config.treatment_training
    train_dataset = load_retained_representation_jsonl(
        training.data.train.jsonl_path,
        expected_source_sha256=training.data.train.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    test_dataset = load_retained_representation_jsonl(
        training.data.validation.jsonl_path,
        expected_source_sha256=training.data.validation.source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    train = load_answer_bearing_span_index(
        train_dataset,
        config.train_span_sidecar_path,
        expected_sidecar_sha256=config.train_span_sidecar_sha256,
    )
    test = load_answer_bearing_span_index(
        test_dataset,
        config.test_span_sidecar_path,
        expected_sidecar_sha256=config.test_span_sidecar_sha256,
    )
    merged = merge_answer_bearing_span_indices(train, test)
    return train, test, merged


def _make_loss_supervision_factory(index_set: AnswerBearingSpanIndexSet) -> Any:
    """Late import keeps config/preflight free of model and CUDA initialization."""

    from .supervision import AnswerBearingSpanSupervisionFactory

    return AnswerBearingSpanSupervisionFactory(index_set)


def _span_index_payload(
    train: AnswerBearingSpanIndex,
    test: AnswerBearingSpanIndex,
    merged: AnswerBearingSpanIndexSet,
) -> dict[str, object]:
    return {
        "span_index_set_identity_sha256": merged.identity_sha256,
        "train_span_index_identity_sha256": train.identity_sha256,
        "test_span_index_identity_sha256": test.identity_sha256,
        "train_span_sidecar_sha256": train.sidecar_sha256,
        "test_span_sidecar_sha256": test.sidecar_sha256,
        "train_span_population_sha256": train.retained_semantic_population_sha256,
        "test_span_population_sha256": test.retained_semantic_population_sha256,
        "train_span_annotator_identity": train.annotator_identity,
        "test_span_annotator_identity": test.annotator_identity,
        "train_span_statistics": train.statistics.canonical_payload(),
        "test_span_statistics": test.statistics.canonical_payload(),
        "combined_span_statistics": merged.statistics.canonical_payload(),
    }


@contextmanager
def _inject_answer_bearing_span_components(
    config: AnswerBearingSpanExperimentConfig,
    manifest: ImageAxisDonorManifest,
    *,
    loss_supervision_factory: Any,
) -> Iterator[None]:
    """Patch the native-builder and trainer seams for one process invocation."""

    original_builder = core_runner.Qwen3NativeRepresentationGroupBuilder
    original_trainer = core_runner.RepresentationTrainer
    original_driver_seal = core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL
    if original_builder is not Qwen3NativeRepresentationGroupBuilder:
        raise RuntimeError("core group-builder seam was already patched")
    if original_trainer is not RepresentationTrainer:
        raise RuntimeError("core trainer seam was already patched")
    if original_driver_seal is not None:
        raise RuntimeError("core experiment-driver seam was already active")
    if not callable(loss_supervision_factory):
        raise TypeError("RP70 loss-supervision factory must be callable")

    def builder_factory(**kwargs: Any) -> Qwen3NativeRepresentationGroupBuilder:
        if "readout_loss_supervision_factory" in kwargs:
            raise ValueError("core attempted to override RP70 span supervision")
        return original_builder(
            **kwargs,
            readout_loss_supervision_factory=loss_supervision_factory,
        )

    def trainer_factory(**kwargs: Any) -> ImageAxisGroundingTrainer:
        base_builder = kwargs.pop("group_builder")
        if not isinstance(base_builder, Qwen3NativeRepresentationGroupBuilder):
            raise TypeError("RP70 trainer requires the native Qwen3 group builder")
        return ImageAxisGroundingTrainer(
            **kwargs,
            group_builder=ImageAxisGroundedNativeGroupBuilder(
                base_builder=base_builder,
                donor_manifest=manifest,
            ),
            image_axis_objective=config.objective,
        )

    core_runner.Qwen3NativeRepresentationGroupBuilder = builder_factory  # type: ignore[assignment]
    core_runner.RepresentationTrainer = trainer_factory  # type: ignore[assignment]
    core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL = (
        core_runner._ANSWER_BEARING_SPAN_DRIVER_SEAL
    )
    try:
        yield
    finally:
        core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL = original_driver_seal
        core_runner.RepresentationTrainer = original_trainer
        core_runner.Qwen3NativeRepresentationGroupBuilder = original_builder


__all__ = [
    "ANSWER_BEARING_SPAN_RUNNER_SCHEMA_VERSION",
    "run_answer_bearing_span_experiment",
    "validate_answer_bearing_span_experiment",
]
