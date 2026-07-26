from __future__ import annotations

import pytest

from tgvf_rl.qwen.crop_coordinates import (
    QWEN25_CROP_CONVERSION_VERSION,
    QWEN25_CROP_COORDINATE_SPACE,
    QWEN3_CROP_CONVERSION_VERSION,
    QWEN3_CROP_COORDINATE_SPACE,
)
from tgvf_rl.qwen.qwen25_vl import Qwen25VLAdapter
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter


def test_qwen3_relative_box_maps_to_non_square_source_with_official_floor() -> None:
    mapping = Qwen3VLAdapter().map_crop_bbox_to_source(
        (75, 306, 435, 710),
        source_width=500,
        source_height=333,
    )

    assert mapping.coordinate_space == QWEN3_CROP_COORDINATE_SPACE
    assert mapping.conversion_version == QWEN3_CROP_CONVERSION_VERSION
    assert (mapping.coordinate_reference_width, mapping.coordinate_reference_height) == (
        1000,
        1000,
    )
    assert mapping.model_bbox_2d == (75, 306, 435, 710)
    assert mapping.source_bbox_2d == (37, 101, 217, 236)


def test_qwen3_full_grid_maps_exactly_to_full_source() -> None:
    mapping = Qwen3VLAdapter().map_crop_bbox_to_source(
        (0, 0, 1000, 1000),
        source_width=317,
        source_height=113,
        processor_resized_size=(672, 224),
    )

    assert mapping.source_bbox_2d == (0, 0, 317, 113)
    assert mapping.coordinate_reference_width == 1000
    assert mapping.coordinate_reference_height == 1000


@pytest.mark.parametrize(
    "bbox",
    ((-1, 0, 100, 100), (0, 0, 1001, 100), (500, 0, 500, 100)),
)
def test_qwen3_invalid_model_boxes_fail_closed(
    bbox: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        Qwen3VLAdapter().map_crop_bbox_to_source(
            bbox,
            source_width=500,
            source_height=333,
        )


def test_qwen3_box_that_collapses_on_small_source_fails_closed() -> None:
    with pytest.raises(ValueError, match="converted source bbox must be non-empty"):
        Qwen3VLAdapter().map_crop_bbox_to_source(
            (1, 1, 2, 2),
            source_width=5,
            source_height=4,
        )


def test_qwen25_requires_exact_processor_resized_geometry() -> None:
    with pytest.raises(ValueError, match="requires exact processor-resized"):
        Qwen25VLAdapter().map_crop_bbox_to_source(
            (64, 32, 192, 96),
            source_width=1000,
            source_height=500,
        )


def test_qwen25_resized_absolute_box_inverts_to_source() -> None:
    mapping = Qwen25VLAdapter().map_crop_bbox_to_source(
        (64, 32, 192, 96),
        source_width=1000,
        source_height=500,
        processor_resized_size=(256, 128),
    )

    assert mapping.coordinate_space == QWEN25_CROP_COORDINATE_SPACE
    assert mapping.conversion_version == QWEN25_CROP_CONVERSION_VERSION
    assert (mapping.coordinate_reference_width, mapping.coordinate_reference_height) == (
        256,
        128,
    )
    assert mapping.source_bbox_2d == (250, 125, 750, 375)


def test_qwen25_box_outside_resized_image_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the processor-resized image"):
        Qwen25VLAdapter().map_crop_bbox_to_source(
            (0, 0, 257, 128),
            source_width=1000,
            source_height=500,
            processor_resized_size=(256, 128),
        )
