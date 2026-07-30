"""Isolated native-group wrapper for the image-axis grounding ablation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.readout import SameImageReadoutGroup
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from .matching import ImageAxisDonorAssignment, ImageAxisDonorManifest


@dataclass(frozen=True, slots=True)
class ImageAxisGroundingGroup:
    """One legacy RP66 group plus an aligned wrong-image Adapter branch.

    ``donor`` is always materialized.  For an unmatched manifest assignment it
    is a second forward of the anchor group and ``image_axis_row_mask`` is all
    false; the loss layer must then supply zero VJPs.  This keeps the Adapter
    forward/backward collective count identical across ranks without dropping
    the legacy RP66 objective for those groups.
    """

    base: SameImageReadoutGroup
    donor: SameImageReadoutGroup
    image_axis_row_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, SameImageReadoutGroup):
            raise TypeError("image-axis base must be SameImageReadoutGroup")
        if not isinstance(self.donor, SameImageReadoutGroup):
            raise TypeError("image-axis donor must be SameImageReadoutGroup")
        if not isinstance(self.image_axis_row_mask, tuple) or any(
            type(value) is not bool for value in self.image_axis_row_mask
        ):
            raise TypeError("image-axis row mask must be an immutable bool tuple")
        if len(self.image_axis_row_mask) != len(self.base.rows):
            raise ValueError("image-axis row mask must align with base rows")
        if len(set(self.image_axis_row_mask)) != 1:
            raise ValueError("image-axis eligibility must be group-homogeneous")
        if len(self.donor.rows) != len(self.base.rows):
            raise ValueError("image-axis donor rows must align with base rows")
        if self.donor.collective_candidate_count != self.base.collective_candidate_count:
            raise ValueError("base and donor collective candidate counts differ")
        base_ids = tuple(row.sample_id for row in self.base.rows)
        donor_row_ids = tuple(row.sample_id for row in self.donor.rows)
        donor_candidate_ids = tuple(
            candidate.sample_id for candidate in self.donor.candidates
        )
        if donor_row_ids != base_ids or donor_candidate_ids != base_ids:
            raise ValueError("image-axis donor order must retain anchor sample IDs")
        base_grids = tuple(candidate.image_grid_thw for candidate in self.base.candidates)
        donor_grids = tuple(
            candidate.image_grid_thw for candidate in self.donor.candidates
        )
        if base_grids != donor_grids or any(grid is None for grid in base_grids):
            raise ValueError("base and image-axis donor Qwen grids differ")
        if self.base.source_visual.main.shape != self.donor.source_visual.main.shape:
            raise ValueError("base and image-axis donor source tensor shapes differ")
        if tuple(
            tensor.shape for tensor in self.base.source_visual.deepstack
        ) != tuple(tensor.shape for tensor in self.donor.source_visual.deepstack):
            raise ValueError("base and image-axis donor DeepStack shapes differ")
        eligible = self.image_axis_row_mask[0]
        source_is_distinct = (
            self.base.image_group_key != self.donor.image_group_key
            and self.base.source_visual_identity
            != self.donor.source_visual_identity
        )
        if eligible and not source_is_distinct:
            raise ValueError("eligible image-axis donor source must be distinct")
        if not eligible and (
            self.base.image_group_key != self.donor.image_group_key
            or self.base.source_visual_identity != self.donor.source_visual_identity
        ):
            raise ValueError("masked image-axis branch must duplicate the anchor source")

    @property
    def collective_candidate_count(self) -> int:
        """Collective Adapter forwards in each of the two aligned branches."""

        return self.base.collective_candidate_count

    @property
    def eligible(self) -> bool:
        return self.image_axis_row_mask[0]


class ImageAxisGroundedNativeGroupBuilder:
    """Call the production native builder twice under a frozen donor manifest."""

    def __init__(
        self,
        *,
        base_builder: Qwen3NativeRepresentationGroupBuilder,
        donor_manifest: ImageAxisDonorManifest,
    ) -> None:
        if not callable(base_builder):
            raise TypeError("image-axis base builder must be callable")
        if not isinstance(donor_manifest, ImageAxisDonorManifest):
            raise TypeError("donor_manifest must be ImageAxisDonorManifest")
        self.base_builder = base_builder
        self.donor_manifest = donor_manifest

    @property
    def donor_manifest_identity_sha256(self) -> str:
        return self.donor_manifest.identity_sha256

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> ImageAxisGroundingGroup:
        if not samples or any(
            not isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise ValueError("image-axis group requires typed anchor samples")
        image_group_keys = {sample.image_group_key for sample in samples}
        image_paths = {sample.image for sample in samples}
        if len(image_group_keys) != 1 or len(image_paths) != 1:
            raise ValueError("image-axis anchors must share one exact image")
        assignment = self.donor_manifest.assignment_for(samples[0].image_group_key)
        _validate_assignment_for_samples(assignment, samples)
        row_mask = self.image_axis_row_mask(samples)

        base = self.base_builder(
            samples,
            adapter,
            collective_candidate_count=collective_candidate_count,
        )
        donor_samples = (
            _materialize_donor_samples(samples, assignment)
            if assignment.matched
            else samples
        )
        donor = self.base_builder(
            donor_samples,
            adapter,
            collective_candidate_count=collective_candidate_count,
        )
        if assignment.matched and any(
            candidate.image_grid_thw != assignment.image_grid_thw
            for candidate in donor.candidates
        ):
            raise ValueError("materialized donor grid differs from bound manifest")
        if any(
            candidate.image_grid_thw != assignment.image_grid_thw
            for candidate in base.candidates
        ):
            raise ValueError("materialized anchor grid differs from bound manifest")
        return ImageAxisGroundingGroup(
            base=base,
            donor=donor,
            image_axis_row_mask=row_mask,
        )

    def image_axis_row_mask(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
    ) -> tuple[bool, ...]:
        """Return group eligibility from the manifest without running a model."""

        if not samples or any(
            not isinstance(sample, RepresentationTrainingSample) for sample in samples
        ):
            raise ValueError("image-axis mask requires typed anchor samples")
        if len({sample.image_group_key for sample in samples}) != 1:
            raise ValueError("image-axis mask samples must share one image group")
        assignment = self.donor_manifest.assignment_for(samples[0].image_group_key)
        _validate_assignment_for_samples(assignment, samples)
        return (assignment.matched,) * len(samples)


def _validate_assignment_for_samples(
    assignment: ImageAxisDonorAssignment,
    samples: tuple[RepresentationTrainingSample, ...],
) -> None:
    if assignment.anchor_image_group_key != samples[0].image_group_key:
        raise ValueError("image-axis assignment binds a different anchor group")
    if assignment.anchor_image != samples[0].image:
        raise ValueError("image-axis assignment binds a different anchor path")


def _materialize_donor_samples(
    samples: tuple[RepresentationTrainingSample, ...],
    assignment: ImageAxisDonorAssignment,
) -> tuple[RepresentationTrainingSample, ...]:
    """Change only the image source/group used to compute D, retaining Q/T."""

    if not assignment.matched:
        raise ValueError("cannot materialize an unmatched image-axis donor")
    if assignment.donor_image is None or assignment.donor_image_group_key is None:
        raise RuntimeError("matched image-axis assignment lost donor identity")
    return tuple(
        replace(
            sample,
            image=assignment.donor_image,
            image_id=assignment.donor_image_group_key,
            # These fields describe the anchor image and are not model-visible.
            # Clearing them avoids attaching false provenance to the donor image.
            stable_image_uid=None,
            item_content_hash=None,
        )
        for sample in samples
    )


__all__ = [
    "ImageAxisGroundedNativeGroupBuilder",
    "ImageAxisGroundingGroup",
]
