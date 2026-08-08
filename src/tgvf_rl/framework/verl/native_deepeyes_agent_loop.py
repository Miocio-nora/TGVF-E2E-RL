"""Thin, upstream-compatible native DeepEyes ToolAgentLoop specialization."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

from PIL import Image

from tgvf_rl.policy.deepeyes_official_protocol import (
    direct_answer_after_last_tool_call,
)

from .native_crop_tool import ensure_native_crop_audit_fields
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_MAX_CROPS,
    NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS,
    NATIVE_DEEPEYES_VISUAL_AGENT,
    assert_native_pixel_row,
    assert_observation_mask,
)


def deepeyes_user_observation_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map upstream role=tool observations to DeepEyes V2 role=user.

    VisualToolBoxV2 renders ``<tool_response><image>...prompt`` inside a user
    turn.  Upstream veRL uses a structured tool turn.  This adapter changes
    only the incremental observation rendering and retains the native image
    object supplied separately to Qwen's processor.
    """

    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            raise ValueError("DeepEyes observation adapter accepts only tool turns")
        content = message.get("content", "")
        if isinstance(content, list):
            mapped_content: list[dict[str, Any]] = [
                {"type": "text", "text": "<tool_response>"}
            ]
            saw_image = False
            trailing_text: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("structured tool content must contain mappings")
                if item.get("type") == "image":
                    mapped_content.append({"type": "image"})
                    saw_image = True
                elif item.get("type") == "text":
                    trailing_text.append(str(item.get("text", "")))
            mapped_content.append(
                {
                    "type": "text",
                    "text": "".join(trailing_text) + "</tool_response>",
                }
            )
            if not saw_image:
                raise ValueError("successful visual observation lacks an image")
            converted.append({"role": "user", "content": mapped_content})
        else:
            converted.append(
                {
                    "role": "user",
                    "content": f"<tool_response>{content}</tool_response>",
                }
            )
    return converted


def _load_original_image(value: object) -> Image.Image:
    """Decode one source image without the smart-resize used by qwen-vl-utils.

    DeepEyes crops in the coordinate system of the unprocessed source image.
    The returned object owns its pixels, so it remains usable after the source
    file is closed and may safely be shared with the rollout processor.
    """

    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("native DeepEyes image must be a local path or PIL image")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("native DeepEyes image path must be absolute")
    if not path.is_file():
        raise ValueError(f"native DeepEyes image path is not a file: {path}")
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _original_images_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[Image.Image]:
    image_values: list[object] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(
            content, (str, bytes, bytearray)
        ):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                raise ValueError("structured native prompt content must be mappings")
            item_type = item.get("type")
            if item_type == "image":
                if "image" not in item:
                    raise ValueError("native image content is missing its payload")
                image_values.append(item["image"])
            elif item_type in {"video", "audio"}:
                raise ValueError("PRL13 native Crop does not support video or audio")
    if len(image_values) != 1:
        raise ValueError("visual PRL13 rows require exactly one original image")
    return [_load_original_image(image_values[0])]


def _build_native_deepeyes_agent_loop_class() -> type[Any]:
    try:
        from verl.experimental.agent_loop.tool_agent_loop import (
            AgentData,
            AgentState,
            ToolAgentLoop,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - runtime-only branch
        missing = error

        class _UnavailableNativeDeepEyesAgentLoop:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(
                    "NativeDeepEyesAgentLoop requires the pinned veRL runtime"
                ) from missing

        return _UnavailableNativeDeepEyesAgentLoop

    class _NativeDeepEyesAgentLoop(ToolAgentLoop):
        """Use upstream multimodal assembly with the pinned clean-final prompt.

        The visible DeepEyes-derived system message already contains its exact
        tool schema.  Passing schemas to Qwen's native template would duplicate
        and mutate that prompt, so only this one pending-state method differs
        from upstream.  Tool parsing and execution still use the globally
        loaded native tool registry.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.tool_parser_name != "hermes":
                raise ValueError("PRL13 native Crop requires the Hermes parser")
            if set(self.tools) != {"image_zoom_in_tool"}:
                raise ValueError(
                    "PRL13 visual loop must expose only image_zoom_in_tool"
                )
            if self.max_parallel_calls != 1:
                raise ValueError("PRL13 requires one Crop call per assistant turn")
            if self.max_user_turns != NATIVE_DEEPEYES_MAX_CROPS:
                raise ValueError("PRL13 requires six possible Crop observation turns")
            if self.max_assistant_turns != NATIVE_DEEPEYES_MAX_CROPS + 1:
                raise ValueError(
                    "PRL13 requires six Crop turns plus one final-answer turn"
                )

        async def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            images: list[Any] | None = None,
            videos: list[Any] | None = None,
            audios: list[Any] | None = None,
            mm_processor_kwargs: dict[str, Any] | None = None,
            remove_system_prompt: bool = False,
        ) -> list[int]:
            if (
                remove_system_prompt
                and messages
                and all(message.get("role") == "tool" for message in messages)
            ):
                messages = deepeyes_user_observation_messages(messages)
            return await super().apply_chat_template(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                remove_system_prompt=remove_system_prompt,
            )

        async def process_multi_modal_info(
            self, messages: list[dict[str, Any]]
        ) -> dict[str, list[Image.Image]]:
            """Load source-native pixels and avoid qwen-vl-utils smart resize.

            The pinned veRL ``RLHFDataset`` route both requires an optional
            dependency absent from the accepted runtime and may resize before
            the tool sees the image.  Either behavior breaks the official
            DeepEyes original-image Crop coordinate contract.
            """

            images = await asyncio.get_running_loop().run_in_executor(
                None, _original_images_from_messages, messages
            )
            return {"images": images}

        async def _handle_pending_state(
            self, agent_data: AgentData, sampling_params: dict[str, Any]
        ) -> AgentState:
            # Leave ``extra_fields`` empty until the first generation.  Pinned
            # ToolAgentLoop treats an empty mapping as the signal to copy vLLM's
            # policy-version metadata (global/min/max_global_steps).  Installing
            # Crop audit defaults here would make that metadata disappear only
            # for visual trajectories, so mixed visual/non-visual worker chunks
            # could no longer be concatenated.  ``run`` installs the complete
            # zero-valued audit surface after direct-answer trajectories finish;
            # the Crop tool installs it on demand for tool-using trajectories.
            del sampling_params
            if self.enable_continuous_token:
                prompt_ids = await self.ct_build_initial_tokens(
                    agent_data.messages, tools=[]
                )
            else:
                prompt_ids = await self.apply_chat_template(
                    agent_data.messages,
                    tools=[],
                    images=agent_data.image_data,
                    videos=agent_data.video_data,
                    audios=agent_data.audio_data,
                    mm_processor_kwargs=agent_data.mm_processor_kwargs,
                )
            agent_data.prompt_ids = prompt_ids
            return AgentState.GENERATING

        async def _handle_generating_state(
            self,
            agent_data: AgentData,
            sampling_params: dict[str, Any],
            ignore_termination: bool = False,
        ) -> AgentState:
            remaining_response_tokens = self.response_length - len(
                agent_data.response_mask
            )
            if remaining_response_tokens <= 0:
                return AgentState.TERMINATED
            turn_sampling_params = {
                **sampling_params,
                "max_tokens": min(
                    NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS,
                    remaining_response_tokens,
                ),
            }
            state = await super()._handle_generating_state(
                agent_data,
                turn_sampling_params,
                ignore_termination=ignore_termination,
            )
            if state is not AgentState.PROCESSING_TOOLS:
                return state

            # Preserve answer-over-action precedence without an answer wrapper:
            # a tool call followed by direct plain text is a completed turn;
            # a turn ending at the tool call executes only its last action.
            text = self.tokenizer.decode(agent_data.response_ids)
            if direct_answer_after_last_tool_call(text) is not None:
                agent_data.tool_calls = []
                return AgentState.TERMINATED
            if len(agent_data.tool_calls) > 1:
                agent_data.tool_calls = [agent_data.tool_calls[-1]]
            return state

        async def _handle_processing_tools_state(
            self, agent_data: AgentData
        ) -> AgentState:
            before = len(agent_data.response_mask)
            state = await super()._handle_processing_tools_state(agent_data)
            after = len(agent_data.response_mask)
            if after > before:
                fields = ensure_native_crop_audit_fields(agent_data.extra_fields)
                spans = fields["crop_observation_token_spans"]
                spans.append([before, after])
                assert_observation_mask(agent_data.response_mask, spans)
            return state

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            source = kwargs.get("data_source")
            if source not in {"vstar", "arxivqa"}:
                raise ValueError("native Crop agent may only execute visual PRL13 rows")
            # This catches image_embeds, exact-replay fields, bad agent routing,
            # and a missing native source-image message before vLLM is invoked.
            assert_native_pixel_row(
                {**kwargs, "agent_name": NATIVE_DEEPEYES_VISUAL_AGENT}
            )
            output = await super().run(sampling_params, **kwargs)
            audit = ensure_native_crop_audit_fields(output.extra_fields)
            images = (output.multi_modal_data or {}).get("images")
            if not isinstance(images, list) or not images:
                raise ValueError("native DeepEyes output lost its original pixels")
            crop_count = len(images) - 1
            if crop_count < 0 or crop_count > NATIVE_DEEPEYES_MAX_CROPS:
                raise ValueError("native DeepEyes output has an invalid crop count")
            if audit["crop_action_count"] != crop_count:
                raise ValueError("native Crop successful-action/image counts disagree")
            spans = audit["crop_observation_token_spans"]
            assert_observation_mask(output.response_mask, spans)
            output.extra_fields.update(
                {
                    "native_original_image_count": 1,
                    "native_crop_image_count": crop_count,
                    "native_total_image_count": len(images),
                    "response_token_count": len(output.response_ids),
                    "native_pixels_proven": True,
                    "legacy_adapter_loaded": False,
                    "observation_role": "user",
                    "observation_envelope": "<tool_response><image>...</tool_response>",
                }
            )
            return output

    _NativeDeepEyesAgentLoop.__name__ = "NativeDeepEyesAgentLoop"
    _NativeDeepEyesAgentLoop.__qualname__ = "NativeDeepEyesAgentLoop"
    _NativeDeepEyesAgentLoop.__module__ = __name__
    return _NativeDeepEyesAgentLoop


NativeDeepEyesAgentLoop = _build_native_deepeyes_agent_loop_class()


__all__ = [
    "NATIVE_DEEPEYES_VISUAL_AGENT",
    "NativeDeepEyesAgentLoop",
    "deepeyes_user_observation_messages",
]
