"""Deterministic counterfactual-input matching for answer-utility evaluation.

This leaf owns only donor contracts and CPU-side input matching/materialization.
The historical :mod:`.runner` facade re-exports the same objects so existing
imports and pickle coordinates remain stable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal
import unicodedata

from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function
from tgvf_rl.representation.training.config import RepresentationTrainingConfig
from tgvf_rl.representation.training.post_training_evaluation import file_sha256
from tgvf_rl.representation.training.schema import RepresentationTrainingSample

from ..controls import _normalized_answer_identity


_PUBLIC_RUNNER_MODULE = (
    "tgvf_rl.representation.experiments.answer_utility.evaluation.runner"
)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class AnswerUtilityWrongImageDonor:
    """One identity-bound image donor for a same-target wrong-image D arm."""

    anchor_image_group_key: str
    anchor_image_sha256: str
    donor_sample_id: str
    donor_sample_content_sha256: str
    donor_image_group_key: str
    donor_image: str
    donor_image_sha256: str
    image_grid_thw: tuple[int, int, int]
    match_tier: Literal[
        "exact_grid_same_source_dataset",
        "exact_grid_same_source_profile",
        "exact_grid_cross_domain",
    ]

    def __post_init__(self) -> None:
        for value, name in (
            (self.anchor_image_group_key, "anchor image group key"),
            (self.donor_sample_id, "donor sample ID"),
            (self.donor_image_group_key, "donor image group key"),
            (self.donor_image, "donor image path"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        for value, name in (
            (self.anchor_image_sha256, "anchor image SHA256"),
            (self.donor_sample_content_sha256, "donor sample content SHA256"),
            (self.donor_image_sha256, "donor image SHA256"),
        ):
            _require_sha256(value, name=name)
        if self.anchor_image_group_key == self.donor_image_group_key:
            raise ValueError("wrong-image donor must use a distinct image group")
        if self.anchor_image_sha256 == self.donor_image_sha256:
            raise ValueError("wrong-image donor must use distinct image bytes")
        if (
            not isinstance(self.image_grid_thw, tuple)
            or len(self.image_grid_thw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.image_grid_thw
            )
        ):
            raise ValueError("wrong-image donor grid must be positive integer THW")
        if self.match_tier not in {
            "exact_grid_same_source_dataset",
            "exact_grid_same_source_profile",
            "exact_grid_cross_domain",
        }:
            raise ValueError("unknown wrong-image donor match tier")


@dataclass(frozen=True, slots=True)
class _QwenImageGridContract:
    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.patch_size, "patch size"),
            (self.merge_size, "merge size"),
            (self.min_pixels, "minimum pixels"),
            (self.max_pixels, "maximum pixels"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen image-grid {name} must be positive")
        if self.max_pixels < self.min_pixels:
            raise ValueError("Qwen image-grid maximum is below its minimum")


def build_answer_safe_wrong_mapping(
    groups: Sequence[tuple[int, Sequence[RepresentationTrainingSample]]],
) -> dict[str, str]:
    """Choose a deterministic same-image, different-target/different-answer D."""

    mapping: dict[str, str] = {}
    for _ordinal, raw_group in groups:
        group = tuple(raw_group)
        if len(group) < 2:
            raise ValueError("answer-safe wrong D requires K>=2")
        if len({sample.image_group_key for sample in group}) != 1:
            raise ValueError("wrong-D candidates must share one image group")
        for index, sample in enumerate(group):
            answer = _normalized_answer_identity(sample.short_answer)
            target = _normalized_target_identity(sample.target)
            candidate = next(
                (
                    group[(index + offset) % len(group)]
                    for offset in range(1, len(group))
                    if _normalized_answer_identity(
                        group[(index + offset) % len(group)].short_answer
                    )
                    != answer
                    and _normalized_target_identity(
                        group[(index + offset) % len(group)].target
                    )
                    != target
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    "image group has no same-image different-target/different-answer "
                    f"wrong D for sample {sample.sample_id}"
                )
            mapping[sample.sample_id] = candidate.sample_id
    if len(mapping) != sum(len(tuple(group)) for _ordinal, group in groups):
        raise ValueError("wrong-D mapping contains duplicate sample IDs")
    return mapping


def build_same_target_wrong_image_mapping(
    groups: Sequence[tuple[int, Sequence[RepresentationTrainingSample]]],
    donor_samples: Sequence[RepresentationTrainingSample],
    *,
    grid_contract: _QwenImageGridContract,
    random_seed: int,
) -> dict[str, AnswerUtilityWrongImageDonor]:
    """Bind each anchor group to a deterministic, exact-grid, distinct image.

    Donor labels are used only to reject accidental answer-equivalent images;
    the returned object contains no donor answer, target, or evidence and only
    its image is permitted to reach D materialization.
    """

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("wrong-image mapping seed must be an integer")
    if isinstance(donor_samples, (str, bytes)) or not isinstance(
        donor_samples, Sequence
    ):
        raise TypeError("wrong-image donor samples must be a sequence")
    donors_by_group: dict[str, list[RepresentationTrainingSample]] = {}
    for sample in donor_samples:
        if not isinstance(sample, RepresentationTrainingSample):
            raise TypeError("wrong-image donor pool must contain typed samples")
        donors_by_group.setdefault(sample.image_group_key, []).append(sample)
    if not donors_by_group:
        raise ValueError("wrong-image donor pool is empty")

    grid_by_path: dict[str, tuple[int, int, int]] = {}

    def image_grid(path: str) -> tuple[int, int, int]:
        observed = grid_by_path.get(path)
        if observed is None:
            observed = _qwen_image_grid_thw(path, grid_contract)
            grid_by_path[path] = observed
        return observed

    donor_descriptors: list[
        tuple[
            RepresentationTrainingSample,
            frozenset[str],
            frozenset[str],
            tuple[int, int, int],
        ]
    ] = []
    for raw_group in donors_by_group.values():
        donor_group = tuple(raw_group)
        paths = frozenset(sample.image for sample in donor_group)
        if len(paths) != 1:
            raise ValueError("one donor image group contains multiple image paths")
        representative = min(donor_group, key=lambda sample: sample.sample_id)
        donor_descriptors.append(
            (
                representative,
                frozenset(
                    _normalized_answer_identity(sample.short_answer)
                    for sample in donor_group
                ),
                frozenset(
                    sample.stable_image_uid
                    for sample in donor_group
                    if sample.stable_image_uid is not None
                ),
                image_grid(representative.image),
            )
        )

    by_grid_source: dict[
        tuple[tuple[int, int, int], str | None],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    by_grid_profile: dict[
        tuple[tuple[int, int, int], str | None],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    by_grid: dict[
        tuple[int, int, int],
        list[
            tuple[
                RepresentationTrainingSample,
                frozenset[str],
                frozenset[str],
                tuple[int, int, int],
            ]
        ],
    ] = {}
    for descriptor in donor_descriptors:
        representative, _answers, _uids, grid = descriptor
        by_grid_source.setdefault((grid, representative.source_dataset), []).append(
            descriptor
        )
        by_grid_profile.setdefault((grid, representative.source_profile), []).append(
            descriptor
        )
        by_grid.setdefault(grid, []).append(descriptor)

    mapping: dict[str, AnswerUtilityWrongImageDonor] = {}
    for _ordinal, raw_group in groups:
        anchor_group = tuple(raw_group)
        if not anchor_group or any(
            not isinstance(sample, RepresentationTrainingSample)
            for sample in anchor_group
        ):
            raise ValueError("wrong-image anchors must be non-empty typed groups")
        if (
            len({sample.image_group_key for sample in anchor_group}) != 1
            or len({sample.image for sample in anchor_group}) != 1
        ):
            raise ValueError("one wrong-image anchor group must share one image")
        anchor = min(anchor_group, key=lambda sample: sample.sample_id)
        anchor_answers = frozenset(
            _normalized_answer_identity(sample.short_answer) for sample in anchor_group
        )
        anchor_uids = frozenset(
            sample.stable_image_uid
            for sample in anchor_group
            if sample.stable_image_uid is not None
        )
        anchor_grid = image_grid(anchor.image)
        anchor_image_sha256 = file_sha256(anchor.image)
        tiers: tuple[
            tuple[
                Literal[
                    "exact_grid_same_source_dataset",
                    "exact_grid_same_source_profile",
                    "exact_grid_cross_domain",
                ],
                Sequence[
                    tuple[
                        RepresentationTrainingSample,
                        frozenset[str],
                        frozenset[str],
                        tuple[int, int, int],
                    ]
                ],
            ],
            ...,
        ] = (
            (
                "exact_grid_same_source_dataset",
                by_grid_source.get((anchor_grid, anchor.source_dataset), ()),
            ),
            (
                "exact_grid_same_source_profile",
                by_grid_profile.get((anchor_grid, anchor.source_profile), ()),
            ),
            ("exact_grid_cross_domain", by_grid.get(anchor_grid, ())),
        )
        chosen: AnswerUtilityWrongImageDonor | None = None
        for match_tier, raw_candidates in tiers:
            candidates = tuple(
                descriptor
                for descriptor in raw_candidates
                if descriptor[0].image_group_key != anchor.image_group_key
                and descriptor[0].image != anchor.image
                and anchor_answers.isdisjoint(descriptor[1])
                and anchor_uids.isdisjoint(descriptor[2])
            )
            ordered = sorted(
                candidates,
                key=lambda descriptor: sha256(
                    (
                        f"{random_seed}\0{anchor.image_group_key}\0"
                        f"{descriptor[0].image_group_key}"
                    ).encode("utf-8")
                ).digest(),
            )
            for representative, _answers, _uids, donor_grid in ordered:
                donor_image_sha256 = file_sha256(representative.image)
                if donor_image_sha256 == anchor_image_sha256:
                    continue
                if donor_grid != anchor_grid:
                    raise RuntimeError("wrong-image mapping admitted a grid mismatch")
                chosen = AnswerUtilityWrongImageDonor(
                    anchor_image_group_key=anchor.image_group_key,
                    anchor_image_sha256=anchor_image_sha256,
                    donor_sample_id=representative.sample_id,
                    donor_sample_content_sha256=representative.content_sha256,
                    donor_image_group_key=representative.image_group_key,
                    donor_image=representative.image,
                    donor_image_sha256=donor_image_sha256,
                    image_grid_thw=donor_grid,
                    match_tier=match_tier,
                )
                break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                "no exact-grid, answer-disjoint, byte-distinct wrong image for "
                f"anchor group {anchor.image_group_key}"
            )
        mapping[anchor.image_group_key] = chosen
    if len(mapping) != len(tuple(groups)):
        raise ValueError("wrong-image mapping contains duplicate anchor groups")
    return mapping


def _load_qwen_image_grid_contract(
    training: RepresentationTrainingConfig,
) -> _QwenImageGridContract:
    path = training.model.local_path / "preprocessor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Qwen preprocessor config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    size = payload.get("size")
    if not isinstance(size, Mapping):
        raise ValueError("Qwen preprocessor size contract is missing")
    patch_size = payload.get("patch_size")
    merge_size = payload.get("merge_size")
    min_pixels = size.get("shortest_edge")
    configured_max = training.model.image_max_pixels
    max_pixels = size.get("longest_edge") if configured_max is None else configured_max
    return _QwenImageGridContract(
        patch_size=patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _qwen_image_grid_thw(
    image_path: str, contract: _QwenImageGridContract
) -> tuple[int, int, int]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("Pillow is required for wrong-image matching") from error
    with Image.open(image_path) as image:
        width, height = image.size
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"unsupported Qwen image geometry: {image_path}")
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


def _same_target_wrong_image_model_inputs(
    rows: Sequence[Any], donor: AnswerUtilityWrongImageDonor
) -> tuple[Any, ...]:
    """Change only the D-source image while retaining anchor Q and target."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("wrong-image D materialization requires anchor rows")
    if {row.image_group_key for row in materialized} != {donor.anchor_image_group_key}:
        raise ValueError("wrong-image donor does not bind the anchor group")
    return tuple(
        replace(
            row,
            image=donor.donor_image,
            image_group_key=donor.donor_image_group_key,
        )
        for row in materialized
    )


def _normalized_target_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target identity requires non-empty text")
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _require_sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")


for _contract_type in (AnswerUtilityWrongImageDonor, _QwenImageGridContract):
    rebind_public_class(
        _contract_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNNER_MODULE,
    )
for _helper in (
    build_answer_safe_wrong_mapping,
    build_same_target_wrong_image_mapping,
    _load_qwen_image_grid_contract,
    _qwen_image_grid_thw,
    _same_target_wrong_image_model_inputs,
    _normalized_target_identity,
):
    rebind_public_function(
        _helper,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNNER_MODULE,
    )
del _contract_type, _helper


__all__ = [
    "AnswerUtilityWrongImageDonor",
    "_QwenImageGridContract",
    "_load_qwen_image_grid_contract",
    "_qwen_image_grid_thw",
    "_same_target_wrong_image_model_inputs",
    "build_answer_safe_wrong_mapping",
    "build_same_target_wrong_image_mapping",
]
