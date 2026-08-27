"""Native-pixel policy evaluation under the DeepEyes-derived visible prompt.

This path is intentionally separate from the historical PRL11 evaluator.  It
passes the original image and every accepted crop back to stock Qwen3-VL as
PIL pixels, keeps the DeepEyes V2 system/tool dialect visible, applies the
project's clean direct-final user suffix, and always crops the immutable
original image in source-pixel coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np
from PIL import Image
import torch

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.framework.vllm import ContentAddressedVLLMTurnRNG
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_MAX_ACTIVE_PERCEPTION,
    DEEPEYES_TOOL_NAME,
    SYSTEM_PROMPT_V2,
    USER_PROMPT_V2,
    VISUAL_PROMPT_IDENTITY,
    build_visual_messages,
    direct_answer_after_last_tool_call,
    parse_hermes_crop_call,
)
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.qwen.crop_coordinates import (
    QWEN3_CROP_CONVERSION_VERSION,
    QWEN3_CROP_COORDINATE_SPACE,
    map_qwen3_crop_bbox_to_source,
)
from tgvf_rl.rewards.deepeyes_official import extract_visual_answer
from tgvf_rl.trajectories.schema import TrajectoryIdentity

from .policy_coredev import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
    CoreDevTask,
    PolicyCoreDevConfig,
    PolicyEvaluationSnapshot,
    StandaloneTGVFVLLMManager,
    evaluation_image_max_pixels,
    full_model_protocol_audit_fields,
    load_verified_task_image,
    paired_evaluation_rng_for_task,
    policy_evaluation_identity,
)


_TOOL_CALL_CONTENT = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_VLLM_CONTEXT_SAFETY_TOKENS = 16


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rgb_tensor_to_pil(rgb: torch.Tensor) -> Image.Image:
    if (
        not isinstance(rgb, torch.Tensor)
        or rgb.dtype is not torch.uint8
        or rgb.ndim != 3
        or rgb.shape[-1] != 3
        or rgb.device.type != "cpu"
    ):
        raise ValueError("official-visible source must be CPU uint8 RGB")
    return Image.fromarray(np.asarray(rgb.contiguous().numpy()), mode="RGB")


def _official_visible_processor_geometry(
    processor: object,
) -> tuple[Mapping[str, object], int, int]:
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not isinstance(size, Mapping):
        raise ValueError("official-visible processor size is invalid")
    shortest_edge = size.get("shortest_edge")
    if type(shortest_edge) is not int or shortest_edge <= 0:
        raise ValueError("official-visible processor shortest_edge is invalid")
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError("official-visible processor patch_size is invalid")
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("official-visible processor merge_size is invalid")
    return size, patch_size, merge_size


def _official_visible_mm_processor_kwargs(
    processor: object,
    image_max_pixels: int,
) -> dict[str, object]:
    """Build the one Qwen3 image-size contract used before and during decode."""

    if type(image_max_pixels) is not int or image_max_pixels <= 0:
        raise ValueError("official-visible image_max_pixels must be positive")
    size, _patch_size, _merge_size = _official_visible_processor_geometry(processor)
    shortest_edge = size["shortest_edge"]
    if image_max_pixels < shortest_edge:
        raise ValueError(
            "official-visible image_max_pixels is below processor shortest_edge"
        )
    return {
        "images_kwargs": {
            "size": {
                "shortest_edge": shortest_edge,
                "longest_edge": image_max_pixels,
            }
        }
    }


def _official_visible_visual_counts(
    processor: object,
    image_grid_thw: torch.Tensor,
    *,
    image_count: int,
    image_max_pixels: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape) != (image_count, 3):
        raise ValueError("official-visible image_grid_thw is malformed")
    _size, patch_size, merge_size = _official_visible_processor_geometry(processor)
    merge_area = merge_size**2
    visual_counts: list[int] = []
    represented_pixel_areas: list[int] = []
    for grid in image_grid_thw:
        values = tuple(value.item() for value in grid)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("official-visible image grid values are invalid")
        temporal, grid_height, grid_width = values
        if temporal != 1:
            raise ValueError("official-visible PIL image grid must be single-frame")
        premerge_count = temporal * grid_height * grid_width
        if premerge_count % merge_area:
            raise ValueError("official-visible image grid is not merge-aligned")
        represented_pixel_area = grid_height * grid_width * patch_size**2
        if represented_pixel_area > image_max_pixels:
            raise ValueError("official-visible processor grid exceeds image_max_pixels")
        visual_counts.append(premerge_count // merge_area)
        represented_pixel_areas.append(represented_pixel_area)
    return tuple(visual_counts), tuple(represented_pixel_areas)


@dataclass(frozen=True, slots=True)
class OfficialVisibleCropBox:
    requested_bbox_2d: tuple[int, int, int, int]
    source_bbox_2d: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.source_bbox_2d[2] - self.source_bbox_2d[0]

    @property
    def height(self) -> int:
        return self.source_bbox_2d[3] - self.source_bbox_2d[1]


def normalize_official_visible_crop_box(
    bbox_2d: object,
    *,
    image_width: int,
    image_height: int,
) -> OfficialVisibleCropBox:
    """Map a Qwen3 0..1000 box exactly once into original pixels."""

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
    effective = mapping.source_bbox_2d
    width = effective[2] - effective[0]
    height = effective[3] - effective[1]
    if width <= 30 or height <= 30:
        raise ValueError("crop dimensions must both be greater than 30 pixels")
    if max(width, height) / min(width, height) > 100:
        raise ValueError("crop aspect ratio must not exceed 100")
    return OfficialVisibleCropBox(mapping.model_bbox_2d, effective)


def normalize_qwen3_official_visible_crop_box(
    bbox_2d: object,
    *,
    image_width: int,
    image_height: int,
) -> OfficialVisibleCropBox:
    """Compatibility name for the one canonical Qwen3 mapping path."""

    return normalize_official_visible_crop_box(
        bbox_2d,
        image_width=image_width,
        image_height=image_height,
    )


def official_visible_observation_message(*, image: object) -> dict[str, object]:
    """Render PRL13's user-framed visible tool observation message."""

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "<tool_response>"},
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT_V2 + "</tool_response>"},
        ],
    }


def official_visible_error_message(error: str) -> dict[str, str]:
    if not isinstance(error, str) or not error:
        raise ValueError("official-visible tool error must be non-empty")
    return {
        "role": "user",
        "content": f"<tool_response>Error: {error}</tool_response>",
    }


def _render_prompt(
    processor: object,
    messages: Sequence[Mapping[str, object]],
) -> tuple[str, tuple[int, ...]]:
    apply = getattr(processor, "apply_chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(apply) or not callable(encode):
        raise TypeError(
            "official-visible evaluator requires a Qwen processor/tokenizer"
        )
    text = apply(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        tools=[],
    )
    if not isinstance(text, str) or not text:
        raise TypeError("Qwen chat template did not return prompt text")
    token_ids = encode(text, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("Qwen tokenizer did not return token IDs")
    normalized = tuple(token_ids)
    if not normalized or any(
        type(value) is not int or value < 0 for value in normalized
    ):
        raise ValueError("Qwen prompt token IDs are invalid")
    return text, normalized


def _render_native_prompt(
    processor: object,
    messages: Sequence[Mapping[str, object]],
    *,
    images: Sequence[Image.Image],
    image_max_pixels: int,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """Render the same expanded native prompt IDs used by veRL's agent loop."""

    text, canonical_ids = _render_prompt(processor, messages)
    if not images or any(not isinstance(image, Image.Image) for image in images):
        raise TypeError("official-visible native prompt requires PIL images")
    process = getattr(processor, "__call__", None)
    if not callable(process):
        raise TypeError("official-visible processor is not callable")
    mm_processor_kwargs = _official_visible_mm_processor_kwargs(
        processor, image_max_pixels
    )
    batch = process(
        text=[text],
        images=list(images),
        return_tensors="pt",
        **mm_processor_kwargs,
    )
    if not isinstance(batch, Mapping):
        raise TypeError("official-visible processor output must be a mapping")
    input_ids = batch.get("input_ids")
    image_grid_thw = batch.get("image_grid_thw")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not isinstance(image_grid_thw, torch.Tensor)
    ):
        raise ValueError("official-visible native processor tensors are malformed")
    visual_counts, _represented_pixel_areas = _official_visible_visual_counts(
        processor,
        image_grid_thw,
        image_count=len(images),
        image_max_pixels=image_max_pixels,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise TypeError("official-visible tokenizer cannot resolve image_pad")
    image_pad_id = convert("<|image_pad|>")
    expanded_ids = tuple(int(value) for value in input_ids[0].tolist())
    if canonical_ids.count(image_pad_id) != len(images):
        raise ValueError("official-visible canonical image placeholders differ")
    if expanded_ids.count(image_pad_id) != sum(visual_counts):
        raise ValueError("official-visible expanded visual token count differs")
    return text, expanded_ids, visual_counts


def validate_official_visible_processor(
    processor: object,
    *,
    tokenizer_length: int,
    image_max_pixels: int,
) -> dict[str, object]:
    """CPU-only proof that the processor sees the pinned clean-final protocol."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or len(tokenizer) != tokenizer_length:
        raise ValueError("official-visible tokenizer length differs")
    messages = list(build_visual_messages("STATIC_PROTOCOL_PROBE", image="<image>"))
    initial_text, initial_ids = _render_prompt(processor, messages)
    if SYSTEM_PROMPT_V2 not in initial_text or USER_PROMPT_V2 not in initial_text:
        raise ValueError("pinned DeepEyes-derived prompt literals are not visible")
    if initial_text.count("<|image_pad|>") != 1:
        raise ValueError("official initial prompt must expose one native image")
    messages.extend(
        [
            {
                "role": "assistant",
                "content": (
                    '<think>probe</think><tool_call>{"name":"image_zoom_in_tool",'
                    '"arguments":{"bbox_2d":[0,0,64,64]}}</tool_call>'
                ),
            },
            official_visible_observation_message(image="<image>"),
        ]
    )
    continuation_text, continuation_ids = _render_prompt(processor, messages)
    visible_envelope = "<tool_response>"
    if (
        continuation_text.count("<|image_pad|>") != 2
        or visible_envelope not in continuation_text
        or USER_PROMPT_V2 + "</tool_response>" not in continuation_text
    ):
        raise ValueError("official visible crop observation rendering differs")
    probe_images = (
        Image.new("RGB", (2048, 1536), (10, 20, 30)),
        Image.new("RGB", (1536, 2048), (40, 50, 60)),
    )
    source_pixel_areas = tuple(image.width * image.height for image in probe_images)
    if not any(area > image_max_pixels for area in source_pixel_areas):
        raise ValueError("official-visible static probe must exceed image_max_pixels")
    try:
        _, expanded_ids, visual_counts = _render_native_prompt(
            processor,
            messages,
            images=probe_images,
            image_max_pixels=image_max_pixels,
        )
    finally:
        for image in probe_images:
            image.close()
    processor_size, patch_size, merge_size = _official_visible_processor_geometry(
        processor
    )
    represented_pixel_areas = tuple(
        count * merge_size**2 * patch_size**2 for count in visual_counts
    )
    return {
        "schema_version": "tgvf-official-visible-processor-static-proof-v2",
        "prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
        "initial_prompt_token_ids_sha256": _canonical_sha256(initial_ids),
        "continuation_prompt_token_ids_sha256": _canonical_sha256(continuation_ids),
        "initial_prompt_token_count": len(initial_ids),
        "continuation_prompt_token_count": len(continuation_ids),
        "continuation_expanded_prompt_token_count": len(expanded_ids),
        "configured_image_max_pixels": image_max_pixels,
        "processor_image_size": dict(processor_size),
        "effective_processor_image_size": {
            "shortest_edge": processor_size["shortest_edge"],
            "longest_edge": image_max_pixels,
        },
        "processor_patch_size": patch_size,
        "processor_merge_size": merge_size,
        "synthetic_native_source_pixel_areas": list(source_pixel_areas),
        "synthetic_native_represented_pixel_areas": list(represented_pixel_areas),
        "synthetic_native_visual_token_counts": list(visual_counts),
        "native_original_image_count": 1,
        "native_crop_image_count": 1,
        "tools_argument_empty": True,
        "visible_system_schema": True,
        "observation_role": "user",
    }


@dataclass(frozen=True, slots=True)
class OfficialVisibleTrajectory:
    identity: TrajectoryIdentity
    model: ModelIdentity
    behavior_policy: PolicyVersion
    stop: str
    final_answer: str | None
    assistant_turns: tuple[dict[str, object], ...]
    tool_calls: tuple[dict[str, object], ...]
    tool_errors: tuple[dict[str, object], ...]
    native_image_sha256s: tuple[str, ...]

    @property
    def trajectory_sha256(self) -> str:
        return _canonical_sha256(
            {
                "identity": asdict(self.identity),
                "model": asdict(self.model),
                "behavior_policy": asdict(self.behavior_policy),
                "stop": self.stop,
                "final_answer": self.final_answer,
                "assistant_turns": self.assistant_turns,
                "tool_calls": self.tool_calls,
                "tool_errors": self.tool_errors,
                "native_image_sha256s": self.native_image_sha256s,
            }
        )


class OfficialVisiblePolicyEvaluator:
    """One rollout-zero DeepEyes-derived clean-final native-pixel evaluator."""

    def __init__(
        self,
        *,
        config: PolicyCoreDevConfig,
        run: PolicyE2ESmokeRunConfig | object,
        manager: StandaloneTGVFVLLMManager,
        processor: object,
        snapshot: PolicyEvaluationSnapshot | object,
        evaluation_identity: Mapping[str, object],
    ) -> None:
        if config.evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
            raise ValueError("official-visible evaluator received another protocol")
        if not manager.native_pixels or manager.capture_hidden:
            raise ValueError("official-visible evaluator requires stock native pixels")
        from .policy_full_model_snapshot import (
            FullModelEvaluationSnapshot,
            full_model_policy_evaluation_identity,
        )

        full_model = isinstance(snapshot, FullModelEvaluationSnapshot)
        if snapshot.run != run:
            raise ValueError("official-visible run differs from frozen snapshot")
        expected_identity = (
            full_model_policy_evaluation_identity(config, snapshot)
            if full_model
            else policy_evaluation_identity(config, snapshot)
        )
        if dict(evaluation_identity) != expected_identity:
            raise ValueError("official-visible evaluation identity differs")
        protocol = expected_identity.get("protocol")
        if not isinstance(protocol, Mapping) or (
            protocol.get("profile") != DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
            or protocol.get("native_pixels") is not True
            or protocol.get("precomputed_image_embeds") is not False
            or protocol.get("crop_coordinate_space") != QWEN3_CROP_COORDINATE_SPACE
            or protocol.get("crop_coordinate_conversion_version")
            != QWEN3_CROP_CONVERSION_VERSION
        ):
            raise ValueError("official-visible protocol identity is malformed")
        if full_model:
            if manager.lora_request is not None:
                raise ValueError(
                    "full-model official-visible evaluation forbids LoRARequest"
                )
            if protocol.get("coordinate_mapper") != "qwen_0_1000_to_source_v1":
                raise ValueError(
                    "full-model official-visible coordinate mapper differs"
                )
        self.config = config
        self.run = run
        self.manager = manager
        self.processor = processor
        self.snapshot = snapshot
        self.evaluation_identity = expected_identity
        self.policy_version = snapshot.policy_version
        self.full_model = full_model
        self.image_max_pixels = evaluation_image_max_pixels(config, snapshot)
        self.tokenizer = getattr(processor, "tokenizer", None)
        if not callable(getattr(self.tokenizer, "decode", None)):
            raise TypeError("official-visible evaluator requires tokenizer.decode")

    async def _sample_turn(
        self,
        *,
        trajectory_id: str,
        prompt_ids: tuple[int, ...],
        images: Sequence[Image.Image],
        turn_index: int,
        consumed_tokens: int,
        sample_id: str,
        rollout_index: int,
    ) -> tuple[str, tuple[int, ...], tuple[float, ...], str]:
        remaining = self.run.policy.sampling.remaining_response_tokens(consumed_tokens)
        # vLLM 0.12 can sample one token beyond max_model_len when a request
        # lands exactly on the boundary, which terminates the whole engine core.
        # Keep a tiny runtime-only guard band; ordinary response budgets are
        # unchanged, while near-limit trajectories stop as context_limit.
        available = (
            self.config.max_model_len - len(prompt_ids) - _VLLM_CONTEXT_SAFETY_TOKENS
        )
        maximum = min(remaining, available)
        if maximum <= 0:
            raise ValueError("official-visible prompt exhausted max_model_len")
        parameters = dict(
            self.run.policy.sampling.as_vllm_parameters(max_tokens=maximum)
        )
        rng_port = (
            paired_evaluation_rng_for_task(
                self.evaluation_identity,
                sample_id=sample_id,
                rollout_index=rollout_index,
            )
            if getattr(self.config, "paired_seed_namespace", None) is not None
            else ContentAddressedVLLMTurnRNG(
                master_seed=self.run.rollout_rng.master_seed,
                stream_identity=trajectory_id,
            )
        )
        rng = rng_port.for_turn(
            prompt_ids,
            turn_index=turn_index,
            behavior_policy=self.policy_version,
        )
        parameters["seed"] = rng.seed
        output = await self.manager.generate(
            request_id=trajectory_id,
            prompt_ids=list(prompt_ids),
            sampling_params=parameters,
            image_data=list(images),
            mm_processor_kwargs=_official_visible_mm_processor_kwargs(
                self.processor, self.image_max_pixels
            ),
            tgvf_expected_step=self.policy_version.optimizer_step,
        )
        token_ids = tuple(getattr(output, "token_ids", ()))
        logprobs = tuple(getattr(output, "log_probs", ()))
        if not token_ids or len(token_ids) != len(logprobs):
            raise RuntimeError("official-visible sampled tokens/logprobs differ")
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
        if not isinstance(text, str):
            raise TypeError("official-visible tokenizer decode did not return text")
        extra = getattr(output, "extra_fields", {})
        stop_reason = str(extra.get("tgvf_vllm_finish_reason", "unknown"))
        return text, token_ids, logprobs, stop_reason

    async def evaluate(self, task: CoreDevTask) -> OfficialVisibleTrajectory:
        if not task.single_image:
            raise ValueError("official DeepEyes visual protocol requires one image")
        source_rgb = load_verified_task_image(task)
        original = _rgb_tensor_to_pil(source_rgb)
        messages: list[dict[str, object]] = list(
            build_visual_messages(task.question, image="<image>")
        )
        images: list[Image.Image] = [original]
        image_sha256s = [hashlib.sha256(original.tobytes()).hexdigest()]
        identity = TrajectoryIdentity(
            self.config.evaluation_id,
            task.bound_sample_id,
            0,
            (
                f"coredev:{task.ordinal}"
                if self.config.uses_legacy_coredev_manifest
                else f"benchmark:{task.ordinal}"
            ),
        )
        trajectory_id = identity.canonical_id
        assistant_turns: list[dict[str, object]] = []
        tool_calls: list[dict[str, object]] = []
        tool_errors: list[dict[str, object]] = []
        consumed_tokens = 0
        stop = "malformed_action"
        final_answer: str | None = None
        attempts = 0
        try:
            for turn_index in range(DEEPEYES_MAX_ACTIVE_PERCEPTION + 1):
                _prompt_text, prompt_ids, visual_token_counts = _render_native_prompt(
                    self.processor,
                    messages,
                    images=images,
                    image_max_pixels=self.image_max_pixels,
                )
                try:
                    text, token_ids, _logprobs, stop_reason = await self._sample_turn(
                        trajectory_id=trajectory_id,
                        prompt_ids=prompt_ids,
                        images=images,
                        turn_index=turn_index,
                        consumed_tokens=consumed_tokens,
                        sample_id=identity.sample_id,
                        rollout_index=identity.rollout_index,
                    )
                except ValueError as error:
                    # Native image tokens are accounted for inside vLLM, so a
                    # crop-heavy trajectory can cross the context limit after
                    # our text-only preflight.  This is a sample outcome, not a
                    # reason to abort the worker and discard the rest of its rank.
                    if not any(
                        marker in str(error)
                        for marker in (
                            "longer than the maximum model length",
                            "official-visible prompt exhausted max_model_len",
                        )
                    ):
                        raise
                    tool_errors.append(
                        {
                            "attempt_index": attempts,
                            "assistant_turn_index": turn_index,
                            "function_name": DEEPEYES_TOOL_NAME,
                            "code": "context_limit",
                            "payload_json": json.dumps(
                                {"error": str(error)},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "recoverable": False,
                        }
                    )
                    stop = "context_limit"
                    break
                consumed_tokens += len(token_ids)
                tool_call_contents = _TOOL_CALL_CONTENT.findall(text)
                is_tool_call = bool(tool_call_contents)
                assistant_turns.append(
                    {
                        "turn_index": turn_index,
                        "raw_text": text,
                        "sampled_token_count": len(token_ids),
                        "sampled_token_ids_sha256": _canonical_sha256(token_ids),
                        "expanded_prompt_token_count": len(prompt_ids),
                        "native_visual_token_counts": list(visual_token_counts),
                        "is_tool_call": is_tool_call,
                        "stop_reason": stop_reason,
                    }
                )
                messages.append({"role": "assistant", "content": text})
                trailing_final = direct_answer_after_last_tool_call(text)
                if is_tool_call and trailing_final is not None:
                    # A well-formed request hard-stops at </tool_call>.  Never
                    # degrade a boundary violation into a direct final answer:
                    # doing so silently skips the requested Crop and inflates
                    # answer-over-action results.
                    tool_errors.append(
                        {
                            "attempt_index": attempts,
                            "assistant_turn_index": turn_index,
                            "function_name": DEEPEYES_TOOL_NAME,
                            "code": "tool_call_terminal_suffix",
                            "payload_json": json.dumps(
                                {
                                    "response_text_sha256": hashlib.sha256(
                                        text.encode("utf-8")
                                    ).hexdigest(),
                                    "suffix_sha256": hashlib.sha256(
                                        trailing_final.encode("utf-8")
                                    ).hexdigest(),
                                    "suffix_char_count": len(trailing_final),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "recoverable": False,
                        }
                    )
                    stop = "malformed_action"
                    break
                if not is_tool_call:
                    extraction = extract_visual_answer(text)
                    final_answer = extraction.answer or None
                    stop = (
                        "final_answer"
                        if final_answer is not None
                        else "malformed_action"
                    )
                    break
                if attempts >= DEEPEYES_MAX_ACTIVE_PERCEPTION:
                    stop = "tool_call_cap"
                    break
                attempt_index = attempts
                attempts += 1
                try:
                    # VisualToolBoxV2 and the PRL13 agent loop execute the last
                    # call if a completion contains more than one.
                    last_call = "<tool_call>" + tool_call_contents[-1] + "</tool_call>"
                    parsed = parse_hermes_crop_call(last_call)
                    arguments = parsed["arguments"]
                    assert isinstance(arguments, dict)
                    box = normalize_official_visible_crop_box(
                        arguments["bbox_2d"],
                        image_width=original.width,
                        image_height=original.height,
                    )
                    crop = original.crop(box.source_bbox_2d)
                except (KeyError, TypeError, ValueError) as error:
                    tool_errors.append(
                        {
                            "attempt_index": attempt_index,
                            "assistant_turn_index": turn_index,
                            "function_name": DEEPEYES_TOOL_NAME,
                            "code": "invalid_crop",
                            "payload_json": json.dumps(
                                {"error": str(error)},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "recoverable": True,
                        }
                    )
                    messages.append(official_visible_error_message(str(error)))
                    continue
                crop_sha256 = hashlib.sha256(crop.tobytes()).hexdigest()
                label = arguments.get("label")
                call_index = len(tool_calls)
                tool_calls.append(
                    {
                        "call_index": call_index,
                        "attempt_index": attempt_index,
                        "assistant_turn_index": turn_index,
                        "function_name": DEEPEYES_TOOL_NAME,
                        "raw_call_text": text,
                        "bbox_2d": list(box.requested_bbox_2d),
                        "source_bbox_2d": list(box.source_bbox_2d),
                        "effective_bbox_2d": list(box.source_bbox_2d),
                        "coordinate_space": QWEN3_CROP_COORDINATE_SPACE,
                        "conversion_version": QWEN3_CROP_CONVERSION_VERSION,
                        "coordinate_reference_size": [1000, 1000],
                        "source_size": [original.width, original.height],
                        "label": label,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "crop_rgb_sha256": crop_sha256,
                        "crop_source": "immutable_original_image",
                    }
                )
                images.append(crop)
                image_sha256s.append(crop_sha256)
                messages.append(official_visible_observation_message(image="<image>"))
            else:  # pragma: no cover - loop always terminates at the call cap
                stop = "tool_call_cap"
            return OfficialVisibleTrajectory(
                identity=identity,
                model=self.run.model,
                behavior_policy=self.policy_version,
                stop=stop,
                final_answer=final_answer,
                assistant_turns=tuple(assistant_turns),
                tool_calls=tuple(tool_calls),
                tool_errors=tuple(tool_errors),
                native_image_sha256s=tuple(image_sha256s),
            )
        finally:
            for image in images:
                image.close()
            await self.manager.release_trajectory(trajectory_id)


def official_visible_trajectory_audit_payload(
    task: CoreDevTask,
    trajectory: OfficialVisibleTrajectory,
    *,
    evaluation_identity: Mapping[str, object],
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """Materialize the same resume identity plus native-pixel protocol proof."""

    execution = evaluation_identity.get("execution")
    policy_snapshot = evaluation_identity.get("policy_snapshot")
    task_manifest = evaluation_identity.get("task_manifest")
    model_identity = evaluation_identity.get("model_identity")
    protocol = evaluation_identity.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (
            execution,
            policy_snapshot,
            task_manifest,
            model_identity,
            protocol,
        )
    ):
        raise ValueError("official-visible evaluation identity is malformed")
    assert isinstance(execution, Mapping)
    assert isinstance(policy_snapshot, Mapping)
    assert isinstance(task_manifest, Mapping)
    assert isinstance(model_identity, Mapping)
    assert isinstance(protocol, Mapping)
    if (
        execution.get("world_size") != world_size
        or not 0 <= rank < world_size
        or task.ordinal % world_size != rank
    ):
        raise ValueError("official-visible rank assignment differs")
    if asdict(trajectory.model) != dict(model_identity):
        raise ValueError("official-visible trajectory model differs")
    expected_policy = PolicyVersion(
        run_id=str(policy_snapshot.get("run_id")),
        optimizer_step=policy_snapshot.get("optimizer_step"),
        weights_sha256=str(policy_snapshot.get("weights_sha256")),
    )
    if trajectory.behavior_policy != expected_policy:
        raise ValueError("official-visible trajectory policy differs")
    snapshot_backend = policy_snapshot.get("snapshot_backend", "lora")
    if snapshot_backend not in {"lora", "full_model"}:
        raise ValueError("official-visible policy snapshot backend differs")
    snapshot_audit: dict[str, object]
    if snapshot_backend == "full_model":
        required_full = {
            "snapshot_identity_sha256",
            "checkpoint_sha256",
            "source_tree_sha256",
            "materialization_identity_sha256",
            "materialized_model_tree_sha256",
        }
        if (
            not required_full <= set(policy_snapshot)
            or policy_snapshot.get("lora_request") is not None
        ):
            raise ValueError("official-visible full-model identity is incomplete")
        snapshot_audit = {
            "policy_snapshot_backend": "full_model",
            "policy_full_snapshot_identity_sha256": policy_snapshot[
                "snapshot_identity_sha256"
            ],
            "policy_checkpoint_sha256": policy_snapshot["checkpoint_sha256"],
            "policy_source_tree_sha256": policy_snapshot["source_tree_sha256"],
            "policy_materialization_identity_sha256": policy_snapshot[
                "materialization_identity_sha256"
            ],
            "policy_materialized_model_tree_sha256": policy_snapshot[
                "materialized_model_tree_sha256"
            ],
        }
    else:
        required_lora = {
            "pointer_file_sha256",
            "manifest_file_sha256",
            "tensor_file_sha256",
        }
        if not required_lora <= set(policy_snapshot):
            raise ValueError("official-visible LoRA identity is incomplete")
        snapshot_audit = {
            "policy_snapshot_backend": "lora",
            "policy_pointer_file_sha256": policy_snapshot["pointer_file_sha256"],
            "policy_manifest_file_sha256": policy_snapshot["manifest_file_sha256"],
            "policy_tensor_file_sha256": policy_snapshot["tensor_file_sha256"],
        }
    payload: dict[str, object] = {
        "schema_version": POLICY_BENCHMARK_TRAJECTORY_AUDIT_SCHEMA,
        "selection_reasons": ["representative_rollout_zero"],
        "evaluation_identity_sha256": evaluation_identity["identity_sha256"],
        "policy_run_identity_sha256": policy_snapshot["run_identity_sha256"],
        **snapshot_audit,
        "policy_config_identity_sha256": evaluation_identity[
            "policy_run_config_identity_sha256"
        ],
        "task_manifest_sha256": task_manifest["sha256"],
        "model_identity": dict(model_identity),
        "rank": rank,
        "world_size": world_size,
        "evaluation_id": trajectory.identity.run_id,
        "sample_id": trajectory.identity.sample_id,
        "group_uid": trajectory.identity.group_id,
        "rollout_index": trajectory.identity.rollout_index,
        "ordinal": task.ordinal,
        "dataset": task.dataset,
        "row_number": task.row_number,
        "index": task.index,
        "question": task.question,
        "image_paths": list(task.image_paths),
        "image_sha256s": list(task.image_sha256s),
        "image_dimensions": [list(item) for item in task.image_dimensions],
        "trajectory_id": trajectory.identity.canonical_id,
        "trajectory_sha256": trajectory.trajectory_sha256,
        "policy_run_id": trajectory.behavior_policy.run_id,
        "optimizer_step": trajectory.behavior_policy.optimizer_step,
        "policy_weights_sha256": trajectory.behavior_policy.weights_sha256,
        "stop": trajectory.stop,
        "final_answer": trajectory.final_answer,
        "assistant_turns": list(trajectory.assistant_turns),
        "tool_calls": list(trajectory.tool_calls),
        "tool_errors": list(trajectory.tool_errors),
        "successful_observation_count": len(trajectory.tool_calls),
        "evaluation_protocol": protocol["profile"],
        "native_pixels": True,
        "precomputed_image_embeds": False,
        "legacy_adapter_loaded": False,
        "native_original_image_count": 1,
        "native_original_rgb_sha256": trajectory.native_image_sha256s[0],
        "native_crop_image_count": len(trajectory.tool_calls),
        "native_total_image_count": 1 + len(trajectory.tool_calls),
        "native_image_sha256s": list(trajectory.native_image_sha256s),
        "crop_coordinate_space": protocol["crop_coordinate_space"],
        "crop_coordinate_conversion_version": protocol[
            "crop_coordinate_conversion_version"
        ],
        "crop_coordinate_reference_size": protocol["crop_coordinate_reference_size"],
        "crop_source": protocol["crop_source"],
        "observation_role": protocol["observation_role"],
        "observation_envelope": protocol["observation_envelope"],
        "prompt_bundle_sha256": protocol["prompt_bundle_sha256"],
    }
    if snapshot_backend == "full_model":
        payload.update(
            full_model_protocol_audit_fields(evaluation_identity, policy_snapshot)
        )
    if "sampling_rng" in evaluation_identity:
        rng = paired_evaluation_rng_for_task(
            evaluation_identity,
            sample_id=trajectory.identity.sample_id,
            rollout_index=trajectory.identity.rollout_index,
        )
        payload["sampling_rng"] = dict(evaluation_identity["sampling_rng"])
        payload["paired_rng_stream_identity_sha256"] = rng.stream_identity_sha256
    payload["result_identity_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "OfficialVisibleCropBox",
    "OfficialVisiblePolicyEvaluator",
    "OfficialVisibleTrajectory",
    "normalize_official_visible_crop_box",
    "normalize_qwen3_official_visible_crop_box",
    "official_visible_error_message",
    "official_visible_observation_message",
    "official_visible_trajectory_audit_payload",
    "validate_official_visible_processor",
]
