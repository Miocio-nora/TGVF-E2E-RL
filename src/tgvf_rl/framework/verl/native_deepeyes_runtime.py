"""Framework-neutral runtime gates for native-pixel DeepEyes training.

PRL13 is intentionally a separate policy-training path.  These checks make it
impossible for that path to degrade silently into the historical RP66/TGVF
adapter path or a frozen-vision LoRA run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from typing import Any

from PIL import Image


NATIVE_DEEPEYES_RUNTIME_SCHEMA = "tgvf.prl13-native-runtime.v1"
NATIVE_DEEPEYES_VISUAL_AGENT = "prl13_native_deepeyes_visual"
NATIVE_DEEPEYES_THINKLITE_AGENT = "single_turn_agent"
NATIVE_DEEPEYES_MAX_CROPS = 6
NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS = 10_240
NATIVE_DEEPEYES_POLICY_LOSS_MODE = "deepeyes_official_micro_token_mean"
NATIVE_DEEPEYES_POLICY_LOSS_MODULE = (
    "tgvf_rl.framework.verl.deepeyes_actor_loss"
)
NATIVE_DEEPEYES_LOSS_AGG_MODE = "token-mean"

_VISUAL_SOURCES = frozenset({"vstar", "arxivqa"})
_MATH_SOURCE = "thinklite"
_FORBIDDEN_ROW_KEYS = frozenset(
    {
        "image_embeds",
        "precomputed_image_embeds",
        "representation",
        "representation_handle",
        "representation_sidecar",
        "trajectory_replay_bundle",
        "exact_replay_bundle",
        "tgvf_context",
        "rp66_adapter",
        "lora_adapter_path",
        "adapter_path",
    }
)
_FORBIDDEN_MODEL_NAME_PARTS = ("rp66", "tgvf", "precomputed")


@dataclass(frozen=True, slots=True)
class NativeFullModelTrainability:
    """Explicit trainability surface required by the PRL13 main arm."""

    lora_rank: int
    vision_trainable: bool
    vision_projection_trainable: bool
    language_trainable: bool
    native_pixels: bool
    precomputed_image_embeds: bool
    legacy_adapter_path: str | None = None
    model_implementation: str = "native_qwen3_vl"

    def validate(self) -> None:
        if self.lora_rank != 0:
            raise ValueError("PRL13 requires full-model training with lora_rank=0")
        if not self.vision_trainable:
            raise ValueError("PRL13 requires a trainable vision tower")
        if not self.vision_projection_trainable:
            raise ValueError("PRL13 requires a trainable vision projection")
        if not self.language_trainable:
            raise ValueError("PRL13 requires trainable language parameters")
        if not self.native_pixels or self.precomputed_image_embeds:
            raise ValueError("PRL13 requires native pixels and forbids image_embeds")
        if self.legacy_adapter_path not in (None, ""):
            raise ValueError("PRL13 forbids RP66/TGVF adapter loading")
        normalized = self.model_implementation.casefold()
        if any(part in normalized for part in _FORBIDDEN_MODEL_NAME_PARTS):
            raise ValueError("PRL13 model implementation names a legacy path")

    def as_record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": NATIVE_DEEPEYES_RUNTIME_SCHEMA,
            "lora_rank": self.lora_rank,
            "vision_trainable": self.vision_trainable,
            "vision_projection_trainable": self.vision_projection_trainable,
            "language_trainable": self.language_trainable,
            "native_pixels": self.native_pixels,
            "precomputed_image_embeds": self.precomputed_image_embeds,
            "legacy_adapter_path": self.legacy_adapter_path,
            "model_implementation": self.model_implementation,
        }


STRICT_NATIVE_FULL_MODEL_TRAINABILITY = NativeFullModelTrainability(
    lora_rank=0,
    vision_trainable=True,
    vision_projection_trainable=True,
    language_trainable=True,
    native_pixels=True,
    precomputed_image_embeds=False,
)


def native_deepeyes_agent_name(data_source: str) -> str:
    """Route visual rows to Crop and ThinkLite rows to a no-tool loop."""

    if data_source in _VISUAL_SOURCES:
        return NATIVE_DEEPEYES_VISUAL_AGENT
    if data_source == _MATH_SOURCE:
        return NATIVE_DEEPEYES_THINKLITE_AGENT
    raise ValueError(f"unsupported PRL13 data_source: {data_source!r}")


def _walk_mapping(value: object, *, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for raw_key, item in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        yield path, item
        if isinstance(item, Mapping):
            yield from _walk_mapping(item, prefix=path)


def assert_native_pixel_row(row: Mapping[str, object]) -> None:
    """Reject precomputed representations and require a native image message."""

    observed_keys = {path.rsplit(".", 1)[-1] for path, _ in _walk_mapping(row)}
    forbidden = sorted(observed_keys & _FORBIDDEN_ROW_KEYS)
    if forbidden:
        raise ValueError(f"PRL13 row contains forbidden fields: {forbidden}")
    raw_prompt = row.get("raw_prompt")
    if not isinstance(raw_prompt, Sequence) or isinstance(
        raw_prompt, (str, bytes, bytearray)
    ):
        raise ValueError("PRL13 row requires raw_prompt chat messages")
    image_items = 0
    for message in raw_prompt:
        if not isinstance(message, Mapping):
            raise ValueError("raw_prompt messages must be mappings")
        content = message.get("content")
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    image = item.get("image")
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, (str, os.PathLike)):
                        if not os.fspath(image):
                            raise ValueError("native image content path is empty")
                    else:
                        raise ValueError(
                            "native image content requires a local path or PIL image"
                        )
                    image_items += 1
    source = row.get("data_source")
    expected_agent = native_deepeyes_agent_name(str(source))
    if row.get("agent_name") != expected_agent:
        raise ValueError("PRL13 row agent routing differs")
    if source in _VISUAL_SOURCES and image_items != 1:
        raise ValueError("visual PRL13 rows require exactly one original image")
    if source == _MATH_SOURCE and image_items != 0:
        raise ValueError("ThinkLite PRL13 rows must be text-only and no-tool")


def assert_native_multimodal_inputs(
    multi_modal_inputs: Mapping[str, object],
    *,
    original_image_count: int,
    successful_crop_count: int,
) -> None:
    """Prove that actor inputs are processor-produced pixels, not embeds."""

    if type(original_image_count) is not int or original_image_count < 0:
        raise ValueError("original_image_count must be non-negative")
    if type(successful_crop_count) is not int or successful_crop_count < 0:
        raise ValueError("successful_crop_count must be non-negative")
    if any(key in multi_modal_inputs for key in _FORBIDDEN_ROW_KEYS):
        raise ValueError("actor multi_modal_inputs contains a precomputed payload")
    expected = original_image_count + successful_crop_count
    if expected == 0:
        if multi_modal_inputs:
            raise ValueError("text-only route unexpectedly has multimodal inputs")
        return
    pixel_values = multi_modal_inputs.get("pixel_values")
    grid = multi_modal_inputs.get("image_grid_thw")
    if pixel_values is None or grid is None:
        raise ValueError("native visual actor input lacks pixels or image_grid_thw")
    grid_shape = getattr(grid, "shape", None)
    if not isinstance(grid_shape, Sequence) or not grid_shape:
        raise ValueError("image_grid_thw has no batch dimension")
    if int(grid_shape[0]) != expected:
        raise ValueError(
            f"native image count differs: expected {expected}, got {grid_shape[0]}"
        )


def assert_observation_mask(
    response_mask: Sequence[int], observation_spans: Sequence[Sequence[int]]
) -> None:
    """Require every tool-observation token to be excluded from policy loss."""

    for index, span in enumerate(observation_spans):
        if len(span) != 2 or any(type(value) is not int for value in span):
            raise ValueError(f"observation span {index} must be [start, stop]")
        start, stop = span
        if not 0 <= start < stop <= len(response_mask):
            raise ValueError(f"observation span {index} is out of range")
        if any(response_mask[position] != 0 for position in range(start, stop)):
            raise ValueError("tool observation participates in policy loss")


def assert_trainable_parameter_groups(
    named_parameters: Iterable[tuple[str, object]],
) -> dict[str, int]:
    """Check that full-model training exposes both visual and language params."""

    counts = {"vision": 0, "language": 0, "projection": 0, "lora": 0}
    trainable_total = 0
    for name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        trainable_total += 1
        normalized = name.casefold()
        if "lora" in normalized or "adapter" in normalized:
            counts["lora"] += 1
        if any(marker in normalized for marker in ("visual", "vision")):
            counts["vision"] += 1
        elif any(
            marker in normalized
            for marker in ("language", "model.layers", "lm_head", "embed_tokens")
        ):
            counts["language"] += 1
        if any(
            marker in normalized for marker in ("merger", "projector", "projection")
        ):
            counts["projection"] += 1
    if trainable_total == 0:
        raise ValueError("PRL13 model has no trainable parameters")
    if counts["lora"]:
        raise ValueError("PRL13 full-model path exposes trainable LoRA/adapter params")
    if not counts["vision"]:
        raise ValueError("PRL13 model has no trainable vision parameters")
    if not counts["projection"]:
        raise ValueError("PRL13 model has no trainable vision projection")
    if not counts["language"]:
        raise ValueError("PRL13 model has no trainable language parameters")
    return counts


def finite_nonzero_gradient_norm(parameters: Iterable[object]) -> float:
    """CPU-smoke helper shared by the real worker instrumentation."""

    squared = 0.0
    observed = False
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        if hasattr(gradient, "detach"):
            gradient = gradient.detach()
        if hasattr(gradient, "float"):
            gradient = gradient.float()
        norm = float(gradient.norm().item())
        if not math.isfinite(norm):
            raise ValueError("non-finite gradient in PRL13 full-model smoke")
        squared += norm * norm
        observed = True
    total = math.sqrt(squared)
    if not observed or total <= 0.0:
        raise ValueError("missing or zero gradient in PRL13 full-model smoke")
    return total


__all__ = [
    "NATIVE_DEEPEYES_MAX_CROPS",
    "NATIVE_DEEPEYES_LOSS_AGG_MODE",
    "NATIVE_DEEPEYES_POLICY_LOSS_MODE",
    "NATIVE_DEEPEYES_POLICY_LOSS_MODULE",
    "NATIVE_DEEPEYES_RUNTIME_SCHEMA",
    "NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS",
    "NATIVE_DEEPEYES_THINKLITE_AGENT",
    "NATIVE_DEEPEYES_VISUAL_AGENT",
    "NativeFullModelTrainability",
    "STRICT_NATIVE_FULL_MODEL_TRAINABILITY",
    "assert_native_multimodal_inputs",
    "assert_native_pixel_row",
    "assert_observation_mask",
    "assert_trainable_parameter_groups",
    "finite_nonzero_gradient_norm",
    "native_deepeyes_agent_name",
]
