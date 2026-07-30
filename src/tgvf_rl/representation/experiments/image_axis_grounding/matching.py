"""Identity-bound wrong-image donors for the image-axis grounding ablation.

The matcher is deliberately experiment-private.  It selects a real image that
has the exact Qwen visual grid of an anchor while rejecting the anchor image,
byte-identical aliases, stable-image aliases, and every donor group sharing a
normalized short answer with the anchor group.  Every anchor receives an
explicit assignment, including anchors for which no admissible donor exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal
import unicodedata

from tgvf_rl.representation.training.schema import RepresentationTrainingSample


IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION = "image_axis_donor_manifest_v1"
IMAGE_AXIS_DONOR_MATCHING_RULE = (
    "exact_qwen_grid_answer_disjoint_distinct_image_bytes_"
    "tiered_source_profile_domain_sha256_rank_v1"
)

ImageAxisDonorMatchTier = Literal[
    "exact_grid_same_source_dataset",
    "exact_grid_same_source_profile",
    "exact_grid_cross_domain",
]
ImageAxisDonorUnmatchedReason = Literal[
    "no_exact_grid_answer_disjoint_distinct_image"
]

_MATCH_TIERS = (
    "exact_grid_same_source_dataset",
    "exact_grid_same_source_profile",
    "exact_grid_cross_domain",
)
_UNMATCHED_REASON = "no_exact_grid_answer_disjoint_distinct_image"


@dataclass(frozen=True, slots=True)
class QwenImageGridContract:
    """Geometry inputs used by Qwen's image ``smart_resize`` contract."""

    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.patch_size, "patch_size"),
            (self.merge_size, "merge_size"),
            (self.min_pixels, "min_pixels"),
            (self.max_pixels, "max_pixels"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen image-grid {name} must be a positive integer")
        if self.max_pixels < self.min_pixels:
            raise ValueError("Qwen image-grid max_pixels is below min_pixels")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class ImageAxisDonorSourceBinding:
    """External artifacts from which the realized donor population was built."""

    train_source_sha256: str
    retained_manifest_sha256: str
    raw_image_manifest_sha256: str
    preprocessor_config_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.train_source_sha256, "train source SHA256"),
            (self.retained_manifest_sha256, "retained manifest SHA256"),
            (self.raw_image_manifest_sha256, "raw-image manifest SHA256"),
            (self.preprocessor_config_sha256, "preprocessor config SHA256"),
        ):
            _require_sha256(value, name=name)


@dataclass(frozen=True, slots=True)
class ImageAxisDonorAssignment:
    """One matched donor or an explicit unmatched result for an anchor image."""

    anchor_image_group_key: str
    anchor_image: str
    anchor_image_sha256: str
    image_grid_thw: tuple[int, int, int]
    donor_sample_id: str | None = None
    donor_sample_content_sha256: str | None = None
    donor_image_group_key: str | None = None
    donor_image: str | None = None
    donor_image_sha256: str | None = None
    match_tier: ImageAxisDonorMatchTier | None = None
    unmatched_reason: ImageAxisDonorUnmatchedReason | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.anchor_image_group_key, name="anchor image group key")
        _non_empty_text(self.anchor_image, name="anchor image path")
        _require_sha256(self.anchor_image_sha256, name="anchor image SHA256")
        _validate_grid(self.image_grid_thw)
        donor_values = (
            self.donor_sample_id,
            self.donor_sample_content_sha256,
            self.donor_image_group_key,
            self.donor_image,
            self.donor_image_sha256,
            self.match_tier,
        )
        if self.unmatched_reason is None:
            if any(value is None for value in donor_values):
                raise ValueError("matched image-axis assignment requires every donor field")
            _non_empty_text(self.donor_sample_id, name="donor sample ID")
            _require_sha256(
                self.donor_sample_content_sha256,
                name="donor sample content SHA256",
            )
            _non_empty_text(self.donor_image_group_key, name="donor image group key")
            _non_empty_text(self.donor_image, name="donor image path")
            _require_sha256(self.donor_image_sha256, name="donor image SHA256")
            if self.match_tier not in _MATCH_TIERS:
                raise ValueError("unknown image-axis donor match tier")
            if self.anchor_image_group_key == self.donor_image_group_key:
                raise ValueError("image-axis donor must use a distinct image group")
            if self.anchor_image == self.donor_image:
                raise ValueError("image-axis donor must use a distinct image path")
            if self.anchor_image_sha256 == self.donor_image_sha256:
                raise ValueError("image-axis donor must use distinct image bytes")
        else:
            if self.unmatched_reason != _UNMATCHED_REASON:
                raise ValueError("unknown image-axis unmatched reason")
            if any(value is not None for value in donor_values):
                raise ValueError("unmatched image-axis assignment cannot carry a donor")

    @property
    def matched(self) -> bool:
        return self.unmatched_reason is None


@dataclass(frozen=True, slots=True)
class ImageAxisDonorManifest:
    """Complete immutable assignment population and its canonical identity."""

    random_seed: int
    grid_contract: QwenImageGridContract
    source_binding: ImageAxisDonorSourceBinding
    anchor_population_sha256: str
    donor_population_sha256: str
    assignments: tuple[ImageAxisDonorAssignment, ...]
    matching_rule: str = IMAGE_AXIS_DONOR_MATCHING_RULE
    schema_version: str = IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("image-axis donor seed must be an integer")
        if not isinstance(self.grid_contract, QwenImageGridContract):
            raise TypeError("grid_contract must be QwenImageGridContract")
        if not isinstance(self.source_binding, ImageAxisDonorSourceBinding):
            raise TypeError("source_binding must be ImageAxisDonorSourceBinding")
        _require_sha256(
            self.anchor_population_sha256,
            name="anchor population SHA256",
        )
        _require_sha256(
            self.donor_population_sha256,
            name="donor population SHA256",
        )
        if not isinstance(self.assignments, tuple) or not self.assignments:
            raise ValueError("image-axis manifest requires assignments")
        if any(
            not isinstance(assignment, ImageAxisDonorAssignment)
            for assignment in self.assignments
        ):
            raise TypeError("image-axis assignments must be typed")
        keys = tuple(
            assignment.anchor_image_group_key for assignment in self.assignments
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("image-axis assignments must have unique sorted anchors")
        if self.matching_rule != IMAGE_AXIS_DONOR_MATCHING_RULE:
            raise ValueError("image-axis matching rule mismatch")
        if self.schema_version != IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION:
            raise ValueError("image-axis donor manifest schema mismatch")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_payload())

    @property
    def matched_count(self) -> int:
        return sum(assignment.matched for assignment in self.assignments)

    @property
    def unmatched_count(self) -> int:
        return len(self.assignments) - self.matched_count

    @property
    def match_tier_counts(self) -> Mapping[str, int]:
        return {
            tier: sum(assignment.match_tier == tier for assignment in self.assignments)
            for tier in _MATCH_TIERS
        }

    def assignment_for(self, image_group_key: str) -> ImageAxisDonorAssignment:
        _non_empty_text(image_group_key, name="anchor image group key")
        for assignment in self.assignments:
            if assignment.anchor_image_group_key == image_group_key:
                return assignment
        raise KeyError(f"image-axis manifest has no anchor group {image_group_key!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matching_rule": self.matching_rule,
            "random_seed": self.random_seed,
            "grid_contract": asdict(self.grid_contract),
            "grid_contract_identity_sha256": self.grid_contract.identity_sha256,
            "source_binding": asdict(self.source_binding),
            "anchor_population_sha256": self.anchor_population_sha256,
            "donor_population_sha256": self.donor_population_sha256,
            "matched_count": self.matched_count,
            "unmatched_count": self.unmatched_count,
            "match_tier_counts": dict(self.match_tier_counts),
            "assignments": [
                {
                    **asdict(assignment),
                    "image_grid_thw": list(assignment.image_grid_thw),
                    "matched": assignment.matched,
                }
                for assignment in self.assignments
            ],
        }

    def to_bound_payload(self) -> dict[str, object]:
        """JSON payload carrying and proving its own canonical identity."""

        return {"identity_sha256": self.identity_sha256, **self.to_payload()}


@dataclass(frozen=True, slots=True)
class _ImageGroupDescriptor:
    representative: RepresentationTrainingSample
    samples: tuple[RepresentationTrainingSample, ...]
    normalized_answers: frozenset[str]
    stable_image_uids: frozenset[str]
    image_grid_thw: tuple[int, int, int]
    image_sha256: str


def load_qwen_image_grid_contract(
    preprocessor_config_path: str | Path,
    *,
    image_max_pixels: int | None = None,
) -> QwenImageGridContract:
    """Load the exact geometry constants without loading a Qwen model."""

    path = Path(preprocessor_config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Qwen preprocessor config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Qwen preprocessor config must be an object")
    size = payload.get("size")
    if not isinstance(size, Mapping):
        raise ValueError("Qwen preprocessor size contract is missing")
    max_pixels = size.get("longest_edge") if image_max_pixels is None else image_max_pixels
    return QwenImageGridContract(
        patch_size=payload.get("patch_size"),
        merge_size=payload.get("merge_size"),
        min_pixels=size.get("shortest_edge"),
        max_pixels=max_pixels,
    )


def qwen_image_grid_thw(
    image_path: str | Path,
    contract: QwenImageGridContract,
) -> tuple[int, int, int]:
    """Predict Qwen's single-image grid using its native smart-resize formula."""

    if not isinstance(contract, QwenImageGridContract):
        raise TypeError("contract must be QwenImageGridContract")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("Pillow is required for image-axis donor matching") from error
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        width, height = image.size
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"unsupported Qwen image geometry: {path}")
    factor = contract.patch_size * contract.merge_size
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > contract.max_pixels:
        scale = math.sqrt((height * width) / contract.max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < contract.min_pixels:
        scale = math.sqrt(contract.min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return (
        1,
        resized_height // contract.patch_size,
        resized_width // contract.patch_size,
    )


def build_image_axis_donor_manifest(
    anchor_samples: Sequence[RepresentationTrainingSample],
    donor_samples: Sequence[RepresentationTrainingSample],
    *,
    grid_contract: QwenImageGridContract,
    source_binding: ImageAxisDonorSourceBinding,
    random_seed: int,
) -> ImageAxisDonorManifest:
    """Build deterministic donor assignments without changing anchor order."""

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("image-axis donor seed must be an integer")
    if not isinstance(source_binding, ImageAxisDonorSourceBinding):
        raise TypeError("source_binding must be ImageAxisDonorSourceBinding")
    anchors = _typed_samples(anchor_samples, name="anchor")
    donors = _typed_samples(donor_samples, name="donor")
    grid_cache: dict[str, tuple[int, int, int]] = {}
    sha_cache: dict[str, str] = {}

    def image_grid(path: str) -> tuple[int, int, int]:
        if path not in grid_cache:
            grid_cache[path] = qwen_image_grid_thw(path, grid_contract)
        return grid_cache[path]

    def image_sha(path: str) -> str:
        if path not in sha_cache:
            sha_cache[path] = _file_sha256(path)
        return sha_cache[path]

    anchor_descriptors = _group_descriptors(
        anchors,
        image_grid=image_grid,
        image_sha=image_sha,
    )
    donor_descriptors = _group_descriptors(
        donors,
        image_grid=image_grid,
        image_sha=image_sha,
    )

    by_grid_source: dict[
        tuple[tuple[int, int, int], str | None], list[_ImageGroupDescriptor]
    ] = {}
    by_grid_profile: dict[
        tuple[tuple[int, int, int], str | None], list[_ImageGroupDescriptor]
    ] = {}
    by_grid: dict[tuple[int, int, int], list[_ImageGroupDescriptor]] = {}
    for descriptor in donor_descriptors:
        representative = descriptor.representative
        by_grid_source.setdefault(
            (descriptor.image_grid_thw, representative.source_dataset), []
        ).append(descriptor)
        by_grid_profile.setdefault(
            (descriptor.image_grid_thw, representative.source_profile), []
        ).append(descriptor)
        by_grid.setdefault(descriptor.image_grid_thw, []).append(descriptor)

    assignments: list[ImageAxisDonorAssignment] = []
    for anchor in anchor_descriptors:
        representative = anchor.representative
        tiers: tuple[
            tuple[ImageAxisDonorMatchTier, Sequence[_ImageGroupDescriptor]], ...
        ] = (
            (
                "exact_grid_same_source_dataset",
                by_grid_source.get(
                    (anchor.image_grid_thw, representative.source_dataset), ()
                ),
            ),
            (
                "exact_grid_same_source_profile",
                by_grid_profile.get(
                    (anchor.image_grid_thw, representative.source_profile), ()
                ),
            ),
            ("exact_grid_cross_domain", by_grid.get(anchor.image_grid_thw, ())),
        )
        chosen: tuple[ImageAxisDonorMatchTier, _ImageGroupDescriptor] | None = None
        for tier, raw_candidates in tiers:
            candidates = tuple(
                descriptor
                for descriptor in raw_candidates
                if _admissible_donor(anchor, descriptor)
            )
            ordered = sorted(
                candidates,
                key=lambda descriptor: (
                    sha256(
                        (
                            f"{random_seed}\0{representative.image_group_key}\0"
                            f"{descriptor.representative.image_group_key}"
                        ).encode("utf-8")
                    ).digest(),
                    descriptor.representative.image_group_key,
                ),
            )
            if ordered:
                chosen = (tier, ordered[0])
                break
        if chosen is None:
            assignments.append(
                ImageAxisDonorAssignment(
                    anchor_image_group_key=representative.image_group_key,
                    anchor_image=representative.image,
                    anchor_image_sha256=anchor.image_sha256,
                    image_grid_thw=anchor.image_grid_thw,
                    unmatched_reason=_UNMATCHED_REASON,
                )
            )
            continue
        tier, donor = chosen
        donor_sample = donor.representative
        assignments.append(
            ImageAxisDonorAssignment(
                anchor_image_group_key=representative.image_group_key,
                anchor_image=representative.image,
                anchor_image_sha256=anchor.image_sha256,
                image_grid_thw=anchor.image_grid_thw,
                donor_sample_id=donor_sample.sample_id,
                donor_sample_content_sha256=donor_sample.content_sha256,
                donor_image_group_key=donor_sample.image_group_key,
                donor_image=donor_sample.image,
                donor_image_sha256=donor.image_sha256,
                match_tier=tier,
            )
        )

    return ImageAxisDonorManifest(
        random_seed=random_seed,
        grid_contract=grid_contract,
        source_binding=source_binding,
        anchor_population_sha256=_population_sha256(anchors, anchor_descriptors),
        donor_population_sha256=_population_sha256(donors, donor_descriptors),
        assignments=tuple(
            sorted(assignments, key=lambda assignment: assignment.anchor_image_group_key)
        ),
    )


def materialize_image_axis_donor_manifest(
    manifest: ImageAxisDonorManifest,
    output_path: str | Path,
) -> Path:
    """Atomically persist one immutable manifest, idempotently if identical."""

    if not isinstance(manifest, ImageAxisDonorManifest):
        raise TypeError("manifest must be ImageAxisDonorManifest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        observed = load_image_axis_donor_manifest(output)
        if observed != manifest:
            raise FileExistsError(
                f"refusing to replace a different image-axis manifest: {output}"
            )
        return output
    encoded = (
        json.dumps(
            manifest.to_bound_payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Do not silently overwrite a path created after the existence check.
        if output.exists():
            observed = load_image_axis_donor_manifest(output)
            if observed != manifest:
                raise FileExistsError(
                    f"refusing to replace a different image-axis manifest: {output}"
                )
            return output
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_image_axis_donor_manifest(
    path: str | Path,
) -> ImageAxisDonorManifest:
    """Load only the exact persisted schema and rederive every identity field."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"image-axis donor manifest is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("image-axis donor manifest must be a JSON object")
    expected_keys = {
        "identity_sha256",
        "schema_version",
        "matching_rule",
        "random_seed",
        "grid_contract",
        "grid_contract_identity_sha256",
        "source_binding",
        "anchor_population_sha256",
        "donor_population_sha256",
        "matched_count",
        "unmatched_count",
        "match_tier_counts",
        "assignments",
    }
    if set(payload) != expected_keys:
        raise ValueError("image-axis donor manifest fields differ from the exact schema")
    grid_payload = payload["grid_contract"]
    source_payload = payload["source_binding"]
    assignment_payloads = payload["assignments"]
    if not isinstance(grid_payload, Mapping) or set(grid_payload) != {
        "patch_size",
        "merge_size",
        "min_pixels",
        "max_pixels",
    }:
        raise ValueError("image-axis grid contract fields differ from the exact schema")
    if not isinstance(source_payload, Mapping) or set(source_payload) != {
        "train_source_sha256",
        "retained_manifest_sha256",
        "raw_image_manifest_sha256",
        "preprocessor_config_sha256",
    }:
        raise ValueError("image-axis source binding fields differ from the exact schema")
    if not isinstance(assignment_payloads, list) or not assignment_payloads:
        raise ValueError("image-axis assignments must be a non-empty JSON list")
    assignment_keys = {
        "anchor_image_group_key",
        "anchor_image",
        "anchor_image_sha256",
        "image_grid_thw",
        "donor_sample_id",
        "donor_sample_content_sha256",
        "donor_image_group_key",
        "donor_image",
        "donor_image_sha256",
        "match_tier",
        "unmatched_reason",
        "matched",
    }
    assignments: list[ImageAxisDonorAssignment] = []
    for raw_assignment in assignment_payloads:
        if not isinstance(raw_assignment, Mapping) or set(raw_assignment) != assignment_keys:
            raise ValueError(
                "image-axis assignment fields differ from the exact schema"
            )
        raw_grid = raw_assignment["image_grid_thw"]
        if not isinstance(raw_grid, list):
            raise TypeError("persisted image-axis grid must be a JSON list")
        assignment = ImageAxisDonorAssignment(
            **{
                key: value
                for key, value in raw_assignment.items()
                if key not in {"image_grid_thw", "matched"}
            },
            image_grid_thw=tuple(raw_grid),
        )
        if raw_assignment["matched"] is not assignment.matched:
            raise ValueError("persisted image-axis matched flag is inconsistent")
        assignments.append(assignment)
    manifest = ImageAxisDonorManifest(
        random_seed=payload["random_seed"],
        grid_contract=QwenImageGridContract(**grid_payload),
        source_binding=ImageAxisDonorSourceBinding(**source_payload),
        anchor_population_sha256=payload["anchor_population_sha256"],
        donor_population_sha256=payload["donor_population_sha256"],
        assignments=tuple(assignments),
        matching_rule=payload["matching_rule"],
        schema_version=payload["schema_version"],
    )
    if payload != manifest.to_bound_payload():
        raise ValueError("image-axis manifest identity or derived fields differ")
    return manifest


def _typed_samples(
    raw_samples: Sequence[RepresentationTrainingSample],
    *,
    name: str,
) -> tuple[RepresentationTrainingSample, ...]:
    if isinstance(raw_samples, (str, bytes)) or not isinstance(raw_samples, Sequence):
        raise TypeError(f"image-axis {name} samples must be a sequence")
    samples = tuple(raw_samples)
    if not samples:
        raise ValueError(f"image-axis {name} sample population is empty")
    if any(not isinstance(sample, RepresentationTrainingSample) for sample in samples):
        raise TypeError(f"image-axis {name} population must contain typed samples")
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"image-axis {name} population has duplicate sample IDs")
    return samples


def _group_descriptors(
    samples: Sequence[RepresentationTrainingSample],
    *,
    image_grid,
    image_sha,
) -> tuple[_ImageGroupDescriptor, ...]:
    grouped: dict[str, list[RepresentationTrainingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.image_group_key, []).append(sample)
    descriptors: list[_ImageGroupDescriptor] = []
    for group_key in sorted(grouped):
        group = tuple(sorted(grouped[group_key], key=lambda sample: sample.sample_id))
        paths = {sample.image for sample in group}
        if len(paths) != 1:
            raise ValueError(f"image group {group_key!r} contains multiple image paths")
        representative = group[0]
        descriptors.append(
            _ImageGroupDescriptor(
                representative=representative,
                samples=group,
                normalized_answers=frozenset(
                    _normalized_answer(sample.short_answer) for sample in group
                ),
                stable_image_uids=frozenset(
                    sample.stable_image_uid
                    for sample in group
                    if sample.stable_image_uid is not None
                ),
                image_grid_thw=image_grid(representative.image),
                image_sha256=image_sha(representative.image),
            )
        )
    return tuple(descriptors)


def _admissible_donor(
    anchor: _ImageGroupDescriptor,
    donor: _ImageGroupDescriptor,
) -> bool:
    return (
        donor.representative.image_group_key
        != anchor.representative.image_group_key
        and donor.representative.image != anchor.representative.image
        and donor.image_grid_thw == anchor.image_grid_thw
        and donor.image_sha256 != anchor.image_sha256
        and anchor.normalized_answers.isdisjoint(donor.normalized_answers)
        and anchor.stable_image_uids.isdisjoint(donor.stable_image_uids)
    )


def _population_sha256(
    samples: Sequence[RepresentationTrainingSample],
    descriptors: Sequence[_ImageGroupDescriptor],
) -> str:
    return _canonical_sha256(
        {
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "sample_content_sha256": sample.content_sha256,
                    "image_group_key": sample.image_group_key,
                }
                for sample in sorted(samples, key=lambda item: item.sample_id)
            ],
            "images": [
                {
                    "image_group_key": descriptor.representative.image_group_key,
                    "image": descriptor.representative.image,
                    "image_sha256": descriptor.image_sha256,
                    "image_grid_thw": list(descriptor.image_grid_thw),
                }
                for descriptor in descriptors
            ],
        }
    )


def _normalized_answer(value: str) -> str:
    _non_empty_text(value, name="short answer")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_grid(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError("image grid must contain three positive integers")


def _non_empty_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


__all__ = [
    "IMAGE_AXIS_DONOR_MANIFEST_SCHEMA_VERSION",
    "IMAGE_AXIS_DONOR_MATCHING_RULE",
    "ImageAxisDonorAssignment",
    "ImageAxisDonorManifest",
    "ImageAxisDonorSourceBinding",
    "QwenImageGridContract",
    "build_image_axis_donor_manifest",
    "load_image_axis_donor_manifest",
    "load_qwen_image_grid_contract",
    "materialize_image_axis_donor_manifest",
    "qwen_image_grid_thw",
]
