"""Exact Qwen-native tool-response token appender for policy rollouts."""

from __future__ import annotations

from typing import Protocol

from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.schema import (
    NativeToolCall,
    ParsedCropTGVFCall,
    ParsedImageZoomInCall,
    ParsedToolCall,
    StandardToolError,
)
from tgvf_rl.protocol.tool_prompts import (
    QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER,
    render_successful_visual_tool_response,
)

from .agent_loop import SampledPolicyTurn


QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX = "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
QWEN_NATIVE_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_NATIVE_RESPONSE_SUFFIX = (
    "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
)
QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX = (
    "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
)


class NativeToolTurnRegistrar(Protocol):
    """Bind an appended prompt to its exact visual/error rollout state."""

    def register_tool_turn(
        self,
        *,
        previous_prompt_token_ids: tuple[int, ...],
        sampled_turn: SampledPolicyTurn,
        updated_prompt_token_ids: tuple[int, ...],
        observation: ObservationHandle | StandardToolError,
        call_index: int,
    ) -> None: ...


class NativeObservationVisualTokenCountResolver(Protocol):
    """Resolve the merged visual-token count from one verified record handle."""

    def resolve_visual_token_count(self, observation: ObservationHandle) -> int: ...


class QwenNativeToolObservationAppender:
    """Append environment-owned native bytes without rerendering prior turns.

    The sampled assistant continuation remains byte-for-byte policy owned.  A
    successful visual response contributes one native image placeholder; the
    registrar separately binds it to the rollout-recorded crop or main ``D``
    plus D-DeepStack tensors. Errors contain canonical deterministic JSON and
    no visual placeholder.
    """

    def __init__(
        self,
        *,
        tokenizer: object,
        registrar: NativeToolTurnRegistrar,
        visual_token_count_resolver: NativeObservationVisualTokenCountResolver
        | None = None,
        assistant_dialect: NativeAssistantDialect = (
            NativeAssistantDialect.QWEN3_VL_THINKING
        ),
    ) -> None:
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("Qwen native appender requires tokenizer.encode()")
        if not callable(getattr(registrar, "register_tool_turn", None)):
            raise TypeError("registrar must implement register_tool_turn()")
        if visual_token_count_resolver is not None and not callable(
            getattr(visual_token_count_resolver, "resolve_visual_token_count", None)
        ):
            raise TypeError(
                "visual_token_count_resolver must implement "
                "resolve_visual_token_count()"
            )
        if not isinstance(assistant_dialect, NativeAssistantDialect):
            raise TypeError("assistant_dialect must be NativeAssistantDialect")
        self.tokenizer = tokenizer
        self.registrar = registrar
        self.visual_token_count_resolver = visual_token_count_resolver
        self.assistant_dialect = assistant_dialect

    def append(
        self,
        prompt_token_ids: tuple[int, ...],
        sampled_turn: SampledPolicyTurn,
        observation: ObservationHandle | StandardToolError,
        *,
        call_index: int,
        parsed_call: NativeToolCall | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        prompt = tuple(prompt_token_ids)
        if not prompt:
            raise ValueError("native tool appender requires a non-empty prompt")
        if not isinstance(sampled_turn, SampledPolicyTurn):
            raise TypeError("sampled_turn must be SampledPolicyTurn")
        if not isinstance(observation, (ObservationHandle, StandardToolError)):
            raise TypeError("observation must be an exact handle or standard error")
        if type(call_index) is not int or call_index < 0:
            raise ValueError("call_index must be a non-negative integer")

        # The sampled token IDs are authoritative policy output.  Do not
        # re-tokenize policy-owned text here: several byte-level token
        # sequences decode to the same text even though a fresh encode chooses
        # a different, canonical segmentation.  The sampler's exact byte-span
        # decoder and SampledAssistantTurn validate the text/token alignment
        # before parsing; this boundary must preserve those IDs verbatim.
        if isinstance(observation, ObservationHandle):
            if parsed_call is None:
                raise ValueError("successful tool response requires its parsed call")
            _validate_parsed_call_matches_turn(parsed_call, sampled_turn)
            environment_text = render_qwen_native_success_environment_text(
                parsed_call,
                assistant_dialect=self.assistant_dialect,
            )
        else:
            if parsed_call is not None:
                raise ValueError("error tool response must not receive a parsed call")
            environment_text = (
                QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
                + observation.canonical_json
                + qwen_native_response_suffix(self.assistant_dialect)
            )
        environment_ids = self._encode(environment_text)
        if (
            isinstance(observation, ObservationHandle)
            and self.visual_token_count_resolver is not None
        ):
            environment_ids = self._expand_visual_placeholder(
                environment_ids,
                observation,
            )
        if not environment_ids:
            raise ValueError("native tool response encoded to no tokens")
        updated = prompt + sampled_turn.token_ids + environment_ids
        self.registrar.register_tool_turn(
            previous_prompt_token_ids=prompt,
            sampled_turn=sampled_turn,
            updated_prompt_token_ids=updated,
            observation=observation,
            call_index=call_index,
        )
        return updated, environment_ids

    def _encode(self, text: str) -> tuple[int, ...]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, (list, tuple)) or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise TypeError("tokenizer returned invalid native token IDs")
        return tuple(token_ids)

    def _expand_visual_placeholder(
        self,
        environment_ids: tuple[int, ...],
        observation: ObservationHandle,
    ) -> tuple[int, ...]:
        resolver = self.visual_token_count_resolver
        if resolver is None:  # pragma: no cover - guarded by caller
            return environment_ids
        count = resolver.resolve_visual_token_count(observation)
        if type(count) is not int or count <= 0:
            raise ValueError(
                "observation visual token count must be a positive integer"
            )
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise TypeError(
                "visual expansion requires tokenizer.convert_tokens_to_ids()"
            )
        visual_token_id = convert("<|image_pad|>")
        if type(visual_token_id) is not int or visual_token_id < 0:
            raise TypeError("Qwen image placeholder must resolve to a token ID")
        positions = tuple(
            index
            for index, token_id in enumerate(environment_ids)
            if token_id == visual_token_id
        )
        if len(positions) != 1:
            raise ValueError(
                "successful native tool response must encode one image placeholder"
            )
        position = positions[0]
        return (
            environment_ids[:position]
            + (visual_token_id,) * count
            + environment_ids[position + 1 :]
        )


def render_qwen_native_success_payload(
    parsed_call: NativeToolCall,
    *,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_THINKING
    ),
) -> str:
    """Render exact accepted response text, newline, and visual placeholder."""

    arguments: dict[str, object]
    if isinstance(parsed_call, ParsedToolCall):
        arguments = {"target": parsed_call.target}
    elif isinstance(parsed_call, ParsedImageZoomInCall):
        arguments = {"bbox_2d": list(parsed_call.bbox_2d)}
        if parsed_call.label is not None:
            arguments["label"] = parsed_call.label
    elif isinstance(parsed_call, ParsedCropTGVFCall):
        arguments = {
            "bbox_2d": list(parsed_call.bbox_2d),
            "target": parsed_call.target,
        }
    else:  # pragma: no cover - closed-union expansion guard
        raise TypeError("successful response requires a parsed native tool call")
    response_text = render_successful_visual_tool_response(
        parsed_call.name,
        arguments,
        assistant_dialect=assistant_dialect,
    )
    payload = response_text + "\n" + QWEN_NATIVE_IMAGE_PLACEHOLDER
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        # Match DeepEyes' trigger placement: the fresh visual evidence comes
        # first, then the instruction that starts the next policy-owned turn.
        payload += "\n\n" + QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER
    return payload


def render_qwen_native_success_environment_text(
    parsed_call: NativeToolCall,
    *,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_THINKING
    ),
) -> str:
    """Wrap one exact successful visual response in Qwen native turn bytes."""

    return (
        QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
        + render_qwen_native_success_payload(
            parsed_call,
            assistant_dialect=assistant_dialect,
        )
        + qwen_native_response_suffix(assistant_dialect)
    )


def qwen_native_response_suffix(
    assistant_dialect: NativeAssistantDialect,
) -> str:
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
        return QWEN_NATIVE_RESPONSE_SUFFIX
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        return QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX
    raise TypeError("assistant_dialect must be NativeAssistantDialect")


def _validate_parsed_call_matches_turn(
    parsed_call: NativeToolCall,
    sampled_turn: SampledPolicyTurn,
) -> None:
    if (
        parsed_call.sampled_text != sampled_turn.text
        or parsed_call.sampled_token_ids != sampled_turn.token_ids
        or parsed_call.sampled_token_byte_spans != sampled_turn.token_byte_spans
    ):
        raise ValueError("parsed tool call differs from sampled assistant turn")


__all__ = [
    "NativeObservationVisualTokenCountResolver",
    "NativeToolTurnRegistrar",
    "QWEN_NATIVE_IMAGE_PLACEHOLDER",
    "QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX",
    "QWEN_NATIVE_RESPONSE_SUFFIX",
    "QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX",
    "QwenNativeToolObservationAppender",
    "qwen_native_response_suffix",
    "render_qwen_native_success_environment_text",
    "render_qwen_native_success_payload",
]
