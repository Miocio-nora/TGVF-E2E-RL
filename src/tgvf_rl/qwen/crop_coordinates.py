"""Family-owned conversion from model crop coordinates to source pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


QWEN3_CROP_COORDINATE_SPACE = "qwen3_relative_0_1000"
QWEN3_CROP_CONVERSION_VERSION = "qwen3-relative-1000-floor-v1"
QWEN25_CROP_COORDINATE_SPACE = "qwen2_5_processor_resized_absolute"
QWEN25_CROP_CONVERSION_VERSION = "qwen2_5-resized-absolute-floor-v1"
CANONICAL_SOURCE_PIXEL_COORDINATE_SPACE = "canonical_source_pixels"
CANONICAL_SOURCE_PIXEL_CONVERSION_VERSION = "canonical-source-pixels-v1"


@dataclass(frozen=True, slots=True)
class CropCoordinateMapping:
    """Exact provenance for one model-box to immutable-source conversion."""

    coordinate_space: str
    conversion_version: str
    coordinate_reference_width: int
    coordinate_reference_height: int
    model_bbox_2d: tuple[int, int, int, int]
    source_bbox_2d: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.coordinate_space or not self.conversion_version:
            raise ValueError("crop coordinate identities must be non-empty")
        if self.coordinate_reference_width <= 0 or self.coordinate_reference_height <= 0:
            raise ValueError("crop coordinate reference dimensions must be positive")
        _validate_non_empty_bbox(self.model_bbox_2d, name="model")
        _validate_non_empty_bbox(self.source_bbox_2d, name="source")


class CropCoordinateMapper(Protocol):
    """Family adapter surface consumed by the crop executor."""

    crop_coordinate_space: str
    crop_coordinate_conversion_version: str

    def map_crop_bbox_to_source(
        self,
        bbox_2d: tuple[int, int, int, int],
        *,
        source_width: int,
        source_height: int,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> CropCoordinateMapping: ...


class CanonicalSourcePixelCropCoordinateMapper:
    """Explicit canonical mapper for executor fixtures and trusted callers.

    This is deliberately never selected implicitly by a model runtime.
    """

    crop_coordinate_space = CANONICAL_SOURCE_PIXEL_COORDINATE_SPACE
    crop_coordinate_conversion_version = CANONICAL_SOURCE_PIXEL_CONVERSION_VERSION

    def map_crop_bbox_to_source(
        self,
        bbox_2d: tuple[int, int, int, int],
        *,
        source_width: int,
        source_height: int,
        processor_resized_size: tuple[int, int] | None = None,
    ) -> CropCoordinateMapping:
        _validate_source_size(source_width, source_height)
        if processor_resized_size is not None:
            raise ValueError("canonical source-pixel mapping has no processor resize")
        bbox = _validate_non_empty_bbox(bbox_2d, name="model")
        return CropCoordinateMapping(
            coordinate_space=self.crop_coordinate_space,
            conversion_version=self.crop_coordinate_conversion_version,
            coordinate_reference_width=source_width,
            coordinate_reference_height=source_height,
            model_bbox_2d=bbox,
            source_bbox_2d=bbox,
        )


def map_qwen3_crop_bbox_to_source(
    bbox_2d: tuple[int, int, int, int],
    *,
    source_width: int,
    source_height: int,
    processor_resized_size: tuple[int, int] | None = None,
) -> CropCoordinateMapping:
    """Decode Qwen3's official relative 0..1000 box onto source RGB."""

    _validate_source_size(source_width, source_height)
    if processor_resized_size is not None:
        _validate_size(processor_resized_size, name="processor resized")
    bbox = _validate_non_empty_bbox(bbox_2d, name="model")
    if any(value < 0 or value > 1000 for value in bbox):
        raise ValueError("Qwen3 crop coordinates must lie within 0..1000")
    source_bbox = _scale_bbox_floor(
        bbox,
        source_width=source_width,
        source_height=source_height,
        reference_width=1000,
        reference_height=1000,
    )
    _validate_non_empty_bbox(source_bbox, name="converted source")
    return CropCoordinateMapping(
        coordinate_space=QWEN3_CROP_COORDINATE_SPACE,
        conversion_version=QWEN3_CROP_CONVERSION_VERSION,
        coordinate_reference_width=1000,
        coordinate_reference_height=1000,
        model_bbox_2d=bbox,
        source_bbox_2d=source_bbox,
    )


def map_qwen25_crop_bbox_to_source(
    bbox_2d: tuple[int, int, int, int],
    *,
    source_width: int,
    source_height: int,
    processor_resized_size: tuple[int, int] | None = None,
) -> CropCoordinateMapping:
    """Invert Qwen2.5-VL absolute resized-image coordinates onto source RGB."""

    _validate_source_size(source_width, source_height)
    if processor_resized_size is None:
        raise ValueError(
            "Qwen2.5-VL crop conversion requires exact processor-resized dimensions"
        )
    resized_width, resized_height = _validate_size(
        processor_resized_size, name="processor resized"
    )
    bbox = _validate_non_empty_bbox(bbox_2d, name="model")
    left, top, right, bottom = bbox
    if left < 0 or top < 0 or right > resized_width or bottom > resized_height:
        raise ValueError(
            "Qwen2.5-VL crop coordinates lie outside the processor-resized image"
        )
    source_bbox = _scale_bbox_floor(
        bbox,
        source_width=source_width,
        source_height=source_height,
        reference_width=resized_width,
        reference_height=resized_height,
    )
    _validate_non_empty_bbox(source_bbox, name="converted source")
    return CropCoordinateMapping(
        coordinate_space=QWEN25_CROP_COORDINATE_SPACE,
        conversion_version=QWEN25_CROP_CONVERSION_VERSION,
        coordinate_reference_width=resized_width,
        coordinate_reference_height=resized_height,
        model_bbox_2d=bbox,
        source_bbox_2d=source_bbox,
    )


def _scale_bbox_floor(
    bbox: tuple[int, int, int, int],
    *,
    source_width: int,
    source_height: int,
    reference_width: int,
    reference_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        left * source_width // reference_width,
        top * source_height // reference_height,
        right * source_width // reference_width,
        bottom * source_height // reference_height,
    )


def _validate_non_empty_bbox(
    bbox_2d: tuple[int, int, int, int], *, name: str
) -> tuple[int, int, int, int]:
    if len(bbox_2d) != 4 or any(type(value) is not int for value in bbox_2d):
        raise ValueError(f"{name} bbox must contain exactly four integers")
    bbox = tuple(bbox_2d)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{name} bbox must be non-empty")
    return bbox


def _validate_source_size(source_width: int, source_height: int) -> None:
    if type(source_width) is not int or type(source_height) is not int:
        raise TypeError("source image dimensions must be integers")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")


def _validate_size(size: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        len(size) != 2
        or any(type(value) is not int for value in size)
        or any(value <= 0 for value in size)
    ):
        raise ValueError(f"{name} dimensions must contain two positive integers")
    return size


__all__ = [
    "CANONICAL_SOURCE_PIXEL_CONVERSION_VERSION",
    "CANONICAL_SOURCE_PIXEL_COORDINATE_SPACE",
    "CanonicalSourcePixelCropCoordinateMapper",
    "CropCoordinateMapper",
    "CropCoordinateMapping",
    "QWEN25_CROP_CONVERSION_VERSION",
    "QWEN25_CROP_COORDINATE_SPACE",
    "QWEN3_CROP_CONVERSION_VERSION",
    "QWEN3_CROP_COORDINATE_SPACE",
    "map_qwen25_crop_bbox_to_source",
    "map_qwen3_crop_bbox_to_source",
]
