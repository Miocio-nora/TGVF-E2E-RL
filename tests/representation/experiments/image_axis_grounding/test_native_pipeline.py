from __future__ import annotations

from hashlib import sha256
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorAssignment,
    ImageAxisDonorManifest,
    ImageAxisDonorSourceBinding,
    QwenImageGridContract,
)
from tgvf_rl.representation.experiments.image_axis_grounding.native_pipeline import (
    ImageAxisGroundedNativeGroupBuilder,
    ImageAxisGroundingGroup,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sample(
    index: int,
    *,
    image: str = "/fixture/anchor.png",
    image_id: str = "anchor",
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"sample-{index}",
        image=image,
        image_id=image_id,
        question=f"question {index}",
        target=f"target {index}",
        evidence_description=f"evidence {index}",
        short_answer=f"answer {index}",
        stable_image_uid="anchor-stable-uid",
        item_content_hash=f"anchor-content-{index}",
    )


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token if index in evidence_positions else EVIDENCE_IGNORE_INDEX
            for index, token in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


class _RecordingBaseBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[RepresentationTrainingSample, ...]] = []

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        _adapter: object,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        self.calls.append(samples)
        source_identity = f"source::{samples[0].image_group_key}"
        source = _bundle(0.2, requires_grad=False)
        rows = []
        candidates = []
        for index, sample in enumerate(samples):
            token_ids = (1, 2, 2, 2, 2, 3, 5 + index * 2, 6 + index * 2)
            rows.append(
                RepresentationReadoutRow(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=source_identity,
                    supervision=_supervision(token_ids),
                    input_ids=torch.tensor((token_ids,), dtype=torch.long),
                    attention_mask=torch.ones(1, 8, dtype=torch.long),
                    position_ids=torch.arange(8).view(1, 8),
                    source_positions=(1, 2),
                    d_positions=(3, 4),
                )
            )
            candidates.append(
                RepresentationCandidateObservation(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=source_identity,
                    target_conditioning_provider=(
                        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                    ),
                    projection_identities=("main", "branch-8", "branch-16", "branch-24"),
                    visual=_bundle(0.3 + index * 0.1, requires_grad=True),
                    image_grid_thw=(1, 2, 2),
                )
            )
        padding = tuple(
            _bundle(0.9, requires_grad=True)
            for _ in range(collective_candidate_count - len(samples))
        )
        return SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=source_identity,
            source_visual=source,
            rows=tuple(rows),
            candidates=tuple(candidates),
            collective_padding=padding,
        )


def _bundle(value: float, *, requires_grad: bool) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 6), value, requires_grad=requires_grad),
        deepstack=tuple(
            torch.full((1, 2, 6), value + index * 0.01, requires_grad=requires_grad)
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


def _manifest(assignment: ImageAxisDonorAssignment) -> ImageAxisDonorManifest:
    return ImageAxisDonorManifest(
        random_seed=42,
        grid_contract=QwenImageGridContract(16, 2, 1024, 4096),
        source_binding=ImageAxisDonorSourceBinding(
            train_source_sha256="1" * 64,
            retained_manifest_sha256="2" * 64,
            raw_image_manifest_sha256="3" * 64,
            preprocessor_config_sha256="4" * 64,
        ),
        anchor_population_sha256="5" * 64,
        donor_population_sha256="6" * 64,
        assignments=(assignment,),
    )


def test_wrapper_recomputes_donor_branch_with_anchor_question_and_target() -> None:
    samples = (_sample(0), _sample(1))
    assignment = ImageAxisDonorAssignment(
        anchor_image_group_key="anchor",
        anchor_image="/fixture/anchor.png",
        anchor_image_sha256=_sha("anchor"),
        image_grid_thw=(1, 2, 2),
        donor_sample_id="donor-representative",
        donor_sample_content_sha256=_sha("donor sample"),
        donor_image_group_key="donor",
        donor_image="/fixture/donor.png",
        donor_image_sha256=_sha("donor"),
        match_tier="exact_grid_same_source_dataset",
    )
    base_builder = _RecordingBaseBuilder()
    builder = ImageAxisGroundedNativeGroupBuilder(
        base_builder=base_builder,  # type: ignore[arg-type]
        donor_manifest=_manifest(assignment),
    )

    assert builder.image_axis_row_mask(samples) == (True, True)
    assert base_builder.calls == []
    group = builder(samples, object(), collective_candidate_count=3)  # type: ignore[arg-type]

    assert isinstance(group, ImageAxisGroundingGroup)
    assert group.eligible
    assert group.image_axis_row_mask == (True, True)
    assert group.base.collective_candidate_count == 3
    assert group.donor.collective_candidate_count == 3
    assert len(base_builder.calls) == 2
    donor_samples = base_builder.calls[1]
    assert tuple(sample.sample_id for sample in donor_samples) == (
        "sample-0",
        "sample-1",
    )
    assert tuple(sample.question for sample in donor_samples) == tuple(
        sample.question for sample in samples
    )
    assert tuple(sample.target for sample in donor_samples) == tuple(
        sample.target for sample in samples
    )
    assert {sample.image for sample in donor_samples} == {"/fixture/donor.png"}
    assert {sample.image_group_key for sample in donor_samples} == {"donor"}
    assert all(sample.stable_image_uid is None for sample in donor_samples)
    assert all(sample.item_content_hash is None for sample in donor_samples)
    assert tuple(candidate.sample_id for candidate in group.donor.candidates) == (
        "sample-0",
        "sample-1",
    )


def test_unmatched_wrapper_keeps_baseline_and_materializes_masked_duplicate() -> None:
    samples = (_sample(0), _sample(1))
    assignment = ImageAxisDonorAssignment(
        anchor_image_group_key="anchor",
        anchor_image="/fixture/anchor.png",
        anchor_image_sha256=_sha("anchor"),
        image_grid_thw=(1, 2, 2),
        unmatched_reason="no_exact_grid_answer_disjoint_distinct_image",
    )
    base_builder = _RecordingBaseBuilder()
    builder = ImageAxisGroundedNativeGroupBuilder(
        base_builder=base_builder,  # type: ignore[arg-type]
        donor_manifest=_manifest(assignment),
    )

    assert builder.image_axis_row_mask(samples) == (False, False)
    group = builder(samples, object(), collective_candidate_count=2)  # type: ignore[arg-type]

    assert not group.eligible
    assert group.image_axis_row_mask == (False, False)
    assert len(base_builder.calls) == 2
    assert base_builder.calls[0] is samples
    assert base_builder.calls[1] is samples
    assert group.base.image_group_key == group.donor.image_group_key == "anchor"
    assert (
        group.base.source_visual_identity == group.donor.source_visual_identity
        == "source::anchor"
    )


def test_validation_group_outside_train_manifest_is_explicitly_masked() -> None:
    train_assignment = ImageAxisDonorAssignment(
        anchor_image_group_key="anchor",
        anchor_image="/fixture/anchor.png",
        anchor_image_sha256=_sha("anchor"),
        image_grid_thw=(1, 2, 2),
        unmatched_reason="no_exact_grid_answer_disjoint_distinct_image",
    )
    validation_samples = (
        _sample(0, image="/fixture/validation.png", image_id="validation"),
        _sample(1, image="/fixture/validation.png", image_id="validation"),
    )
    base_builder = _RecordingBaseBuilder()
    builder = ImageAxisGroundedNativeGroupBuilder(
        base_builder=base_builder,  # type: ignore[arg-type]
        donor_manifest=_manifest(train_assignment),
    )

    assert builder.image_axis_row_mask(validation_samples) == (False, False)
    group = builder(
        validation_samples,
        object(),  # type: ignore[arg-type]
        collective_candidate_count=2,
    )

    assert not group.eligible
    assert group.base.image_group_key == group.donor.image_group_key == "validation"
    assert len(base_builder.calls) == 2
    assert base_builder.calls == [validation_samples, validation_samples]
