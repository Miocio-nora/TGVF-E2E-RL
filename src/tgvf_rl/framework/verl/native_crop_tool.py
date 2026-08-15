"""Native-pixel Crop tool for the PRL13 DeepEyes runtime.

Unlike the historical project Crop adapter, this tool returns a cropped PIL
image.  veRL's upstream ``ToolAgentLoop`` then appends that image to the
trajectory and re-runs the Qwen3-VL processor, producing actor-side native
``pixel_values``.  Crops always reference the original image, matching
DeepEyes' ``VisualToolBoxV2`` behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from PIL import Image

from tgvf_rl.qwen.crop_coordinates import (
    CropCoordinateMapping,
    QWEN3_CROP_CONVERSION_VERSION,
    QWEN3_CROP_COORDINATE_SPACE,
    map_qwen3_crop_bbox_to_source,
)
from tgvf_rl.policy.deepeyes_official_protocol import USER_PROMPT_V2

from .native_deepeyes_runtime import NATIVE_DEEPEYES_MAX_CROPS


NATIVE_CROP_TOOL_NAME = "image_zoom_in_tool"
DEFAULT_POST_TOOL_PROMPT = USER_PROMPT_V2


@dataclass(frozen=True, slots=True)
class NativeCropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True, slots=True)
class NativeCropResult:
    image: Image.Image
    model_box: NativeCropBox
    box: NativeCropBox
    coordinate_space: str
    conversion_version: str
    coordinate_reference_size: tuple[int, int]
    source_size: tuple[int, int]
    crop_area_fraction: float
    first_call_iou: float | None
    best_call_iou: float | None
    best_gt_coverage: float | None


@dataclass(slots=True)
class _ToolInstance:
    gt_regions: tuple[NativeCropBox, ...]


@dataclass(frozen=True, slots=True)
class _FallbackToolResponse:
    """Used only by CPU tests when veRL is not importable."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None


def _tool_response(**kwargs: Any) -> object:
    try:
        from verl.tools.schemas import ToolResponse
    except ModuleNotFoundError:
        return _FallbackToolResponse(**kwargs)
    return ToolResponse(**kwargs)


def normalize_native_crop_box(
    bbox_2d: object, *, image_width: int, image_height: int
) -> NativeCropBox:
    """Map one Qwen3 0..1000 box exactly once into source pixels."""

    mapping = _native_crop_mapping(
        bbox_2d, image_width=image_width, image_height=image_height
    )
    return NativeCropBox(*mapping.source_bbox_2d)


def _native_crop_mapping(
    bbox_2d: object, *, image_width: int, image_height: int
) -> CropCoordinateMapping:
    """Validate model coordinates and invoke the family-owned mapper."""

    if type(image_width) is not int or image_width <= 0:
        raise ValueError("image_width must be positive")
    if type(image_height) is not int or image_height <= 0:
        raise ValueError("image_height must be positive")
    if (
        not isinstance(bbox_2d, Sequence)
        or isinstance(bbox_2d, (str, bytes, bytearray))
        or len(bbox_2d) != 4
    ):
        raise ValueError("bbox_2d must contain four coordinates")
    if any(type(coordinate) is not int for coordinate in bbox_2d):
        raise ValueError("Qwen3 bbox_2d coordinates must be integers")
    mapping = map_qwen3_crop_bbox_to_source(
        tuple(bbox_2d),
        source_width=image_width,
        source_height=image_height,
    )
    result = NativeCropBox(*mapping.source_bbox_2d)
    if result.width <= 30 or result.height <= 30:
        raise ValueError("crop dimensions must both be greater than 30 pixels")
    if max(result.width, result.height) / min(result.width, result.height) > 100:
        raise ValueError("crop aspect ratio must not exceed 100")
    return mapping


def _intersection_area(left: NativeCropBox, right: NativeCropBox) -> int:
    width = max(0, min(left.right, right.right) - max(left.left, right.left))
    height = max(0, min(left.bottom, right.bottom) - max(left.top, right.top))
    return width * height


def crop_iou(crop: NativeCropBox, target: NativeCropBox) -> float:
    intersection = _intersection_area(crop, target)
    union = crop.area + target.area - intersection
    return 0.0 if union <= 0 else intersection / union


def crop_gt_coverage(crop: NativeCropBox, target: NativeCropBox) -> float:
    intersection = _intersection_area(crop, target)
    return 0.0 if target.area <= 0 else intersection / target.area


def crop_original_image(
    original_image: Image.Image,
    bbox_2d: object,
    *,
    gt_regions: Sequence[NativeCropBox] = (),
    prior_best_iou: float | None = None,
) -> NativeCropResult:
    if not isinstance(original_image, Image.Image):
        raise TypeError("native Crop requires an original PIL image")
    mapping = _native_crop_mapping(
        bbox_2d,
        image_width=original_image.width,
        image_height=original_image.height,
    )
    model_box = NativeCropBox(*mapping.model_bbox_2d)
    box = NativeCropBox(*mapping.source_bbox_2d)
    cropped = original_image.crop(tuple(box.as_list()))
    ious = [crop_iou(box, target) for target in gt_regions]
    coverages = [crop_gt_coverage(box, target) for target in gt_regions]
    current_best = max(ious) if ious else None
    if current_best is not None and prior_best_iou is not None:
        cumulative_best = max(current_best, prior_best_iou)
    else:
        cumulative_best = current_best if current_best is not None else prior_best_iou
    return NativeCropResult(
        image=cropped,
        model_box=model_box,
        box=box,
        coordinate_space=mapping.coordinate_space,
        conversion_version=mapping.conversion_version,
        coordinate_reference_size=(
            mapping.coordinate_reference_width,
            mapping.coordinate_reference_height,
        ),
        source_size=(original_image.width, original_image.height),
        crop_area_fraction=box.area / (original_image.width * original_image.height),
        first_call_iou=current_best,
        best_call_iou=cumulative_best,
        best_gt_coverage=max(coverages) if coverages else None,
    )


def _parse_gt_regions(value: object) -> tuple[NativeCropBox, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("gt_regions must be a sequence")
    regions: list[NativeCropBox] = []
    for index, region in enumerate(value):
        if (
            not isinstance(region, Sequence)
            or isinstance(region, (str, bytes, bytearray))
            or len(region) != 4
        ):
            raise ValueError(f"gt_regions[{index}] must contain four integers")
        if any(type(coordinate) is not int for coordinate in region):
            raise ValueError(f"gt_regions[{index}] must contain four integers")
        left, top, right, bottom = region
        box = NativeCropBox(left, top, right, bottom)
        if box.width <= 0 or box.height <= 0:
            raise ValueError(f"gt_regions[{index}] is empty")
        regions.append(box)
    return tuple(regions)


def ensure_native_crop_audit_fields(
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    """Install the complete per-trajectory Crop audit surface.

    ``crop_call_count`` counts calls that reached this tool's ``execute``
    method, including rejected bounding boxes. ``crop_action_count`` counts
    only calls that returned a crop image. Observation-token spans are owned by
    the agent loop and count rendered tool responses, so they intentionally do
    not have to equal either counter (for example, a parser error can render an
    observation without reaching ``execute``).
    """

    defaults: dict[str, Any] = {
        "crop_call_count": 0,
        "crop_action_count": 0,
        "crop_model_boxes": [],
        "crop_source_boxes": [],
        "crop_boxes": [],
        "crop_area_fractions": [],
        "crop_first_call_iou": None,
        "crop_best_call_iou": None,
        "crop_best_gt_coverage": None,
        "crop_error_count": 0,
        "crop_observation_token_spans": [],
        "decoder_context_overflow": 0,
        "decoder_prompt_length_at_overflow": 0,
        "decoder_max_model_length_at_overflow": 0,
        "crop_coordinate_space": QWEN3_CROP_COORDINATE_SPACE,
        "crop_coordinate_conversion_version": QWEN3_CROP_CONVERSION_VERSION,
        "crop_coordinate_reference_size": [1000, 1000],
    }
    for key, value in defaults.items():
        if key not in extra_fields:
            extra_fields[key] = list(value) if isinstance(value, list) else value
    for key in (
        "crop_coordinate_space",
        "crop_coordinate_conversion_version",
        "crop_coordinate_reference_size",
    ):
        if extra_fields[key] != defaults[key]:
            raise ValueError(f"native Crop {key} identity differs")
    return extra_fields


class NativeDeepEyesCropTool:
    """veRL-native stateful tool with no dependency on project replay code."""

    def __init__(self, config: Mapping[str, Any], tool_schema: object) -> None:
        if tool_schema is None:
            raise ValueError("NativeDeepEyesCropTool requires an explicit schema")
        function = getattr(tool_schema, "function", None)
        name = getattr(function, "name", None)
        if name != NATIVE_CROP_TOOL_NAME:
            raise ValueError("native Crop tool schema name differs")
        self.config = dict(config)
        self.tool_schema = tool_schema
        self.name = name
        max_crops = self.config.get("max_crops")
        if max_crops != NATIVE_DEEPEYES_MAX_CROPS:
            raise ValueError("PRL13 requires exactly six maximum Crop calls")
        post_prompt = self.config.get("post_tool_prompt", DEFAULT_POST_TOOL_PROMPT)
        if not isinstance(post_prompt, str) or not post_prompt.strip():
            raise ValueError("post_tool_prompt must be non-empty")
        self.post_tool_prompt = post_prompt
        self._instances: dict[str, _ToolInstance] = {}

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, object]:
        instance_id = instance_id or uuid4().hex
        create_kwargs = kwargs.get("create_kwargs", {})
        if not isinstance(create_kwargs, Mapping):
            raise ValueError("native Crop create_kwargs must be a mapping")
        self._instances[instance_id] = _ToolInstance(
            gt_regions=_parse_gt_regions(create_kwargs.get("gt_regions", ()))
        )
        return instance_id, _tool_response()

    async def execute(
        self,
        instance_id: str,
        parameters: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[object, float, dict[str, object]]:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise ValueError("unknown native Crop instance")
        if not isinstance(parameters, Mapping):
            raise ValueError("native Crop parameters must be a mapping")
        agent_data = kwargs.get("agent_data")
        image_data = getattr(agent_data, "image_data", None)
        if not isinstance(image_data, list) or not image_data:
            raise ValueError("native Crop trajectory has no original image")
        original_image = image_data[0]
        if not isinstance(original_image, Image.Image):
            raise TypeError("native Crop original image is not PIL")
        extra_fields = getattr(agent_data, "extra_fields", None)
        if not isinstance(extra_fields, dict):
            raise ValueError("native Crop agent_data lacks extra_fields")
        metrics = ensure_native_crop_audit_fields(extra_fields)
        metrics["crop_call_count"] += 1
        if metrics["crop_call_count"] > NATIVE_DEEPEYES_MAX_CROPS:
            metrics["crop_error_count"] += 1
            return (
                _tool_response(text="Error: maximum of six Crop calls exceeded."),
                0.0,
                {"status": "max_crops_exceeded"},
            )
        try:
            result = crop_original_image(
                original_image,
                parameters.get("bbox_2d"),
                gt_regions=instance.gt_regions,
                prior_best_iou=metrics["crop_best_call_iou"],
            )
        except (TypeError, ValueError) as error:
            metrics["crop_error_count"] += 1
            return (
                _tool_response(text=f"Error: {error}"),
                0.0,
                {"status": "invalid_crop", "error": str(error)},
            )

        metrics["crop_action_count"] += 1
        metrics["crop_model_boxes"].append(result.model_box.as_list())
        metrics["crop_source_boxes"].append(result.box.as_list())
        metrics["crop_boxes"].append(result.box.as_list())
        metrics["crop_area_fractions"].append(result.crop_area_fraction)
        if metrics["crop_action_count"] == 1:
            metrics["crop_first_call_iou"] = result.first_call_iou
        metrics["crop_best_call_iou"] = result.best_call_iou
        if result.best_gt_coverage is not None:
            old_coverage = metrics["crop_best_gt_coverage"]
            metrics["crop_best_gt_coverage"] = (
                result.best_gt_coverage
                if old_coverage is None
                else max(old_coverage, result.best_gt_coverage)
            )
        return (
            _tool_response(text=self.post_tool_prompt, image=[result.image]),
            0.0,
            {
                "status": "success",
                "model_bbox_2d": result.model_box.as_list(),
                "source_bbox_2d": result.box.as_list(),
                "effective_bbox_2d": result.box.as_list(),
                "crop_box": result.box.as_list(),
                "coordinate_space": result.coordinate_space,
                "conversion_version": result.conversion_version,
                "coordinate_reference_size": list(result.coordinate_reference_size),
                "source_size": list(result.source_size),
                "crop_area_fraction": result.crop_area_fraction,
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        del instance_id, kwargs
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._instances.pop(instance_id, None)


__all__ = [
    "DEFAULT_POST_TOOL_PROMPT",
    "NATIVE_CROP_TOOL_NAME",
    "NativeCropBox",
    "NativeCropResult",
    "NativeDeepEyesCropTool",
    "crop_gt_coverage",
    "crop_iou",
    "crop_original_image",
    "ensure_native_crop_audit_fields",
    "normalize_native_crop_box",
]
