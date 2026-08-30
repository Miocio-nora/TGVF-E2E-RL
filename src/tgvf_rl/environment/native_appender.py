"""Exact Qwen-native tool-response token appender for policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_TOOL_NAME,
    USER_PROMPT_V2,
)
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
    validate_success_observation_protocol,
)
from tgvf_rl.protocol.schema import (
    NativeToolCall,
    NativeToolCapabilityProfile,
    ParsedCropTGVFCall,
    ParsedImageZoomInCall,
    ParsedToolCall,
    StandardToolError,
)
from tgvf_rl.protocol.tool_prompts import (
    IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT,
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
# Qwen3-VL Instruct renders the public DeepEyes Crop observation as a user
# turn.  Its chat-template bytes deliberately have no newline immediately
# inside the ``tool_response`` envelope.
QWEN_NATIVE_MATCHED_CROP_SUCCESS_PREFIX = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>"
)
QWEN_NATIVE_MATCHED_CROP_SUCCESS_SUFFIX = (
    "</tool_response><|im_end|>\n<|im_start|>assistant\n"
)
QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT = (
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_PREFIX
    + QWEN_NATIVE_IMAGE_PLACEHOLDER
    + USER_PROMPT_V2
    + QWEN_NATIVE_MATCHED_CROP_SUCCESS_SUFFIX
)
QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256 = hashlib.sha256(
    QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT.encode("utf-8")
).hexdigest()

# Historical PRL-25-B/PRL-26-B Instruct Crop training used this generic native
# continuation.  It remains readable only through the explicit legacy protocol
# ID below; canonical Crop must never reach it as an implicit fallback.
QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT = (
    QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
    + IMAGE_ZOOM_IN_SUCCESS_RESPONSE_TEXT
    + "\n"
    + QWEN_NATIVE_IMAGE_PLACEHOLDER
    + "\n\n"
    + QWEN3_INSTRUCT_TOOL_RESPONSE_REASONING_REMINDER
    + QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX
)
QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256 = hashlib.sha256(
    QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT.encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeSuccessObservationContract:
    """Bind one protocol identity to its tool surface and model dialect.

    The renderer is intentionally selected by the protocol ID rather than
    supplied as an arbitrary callable.  This keeps rollout append and replay
    layout on one immutable, auditable byte contract.
    """

    protocol_id: NativeSuccessObservationProtocolId
    tool_profile: NativeToolCapabilityProfile
    assistant_dialect: NativeAssistantDialect

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, NativeSuccessObservationProtocolId):
            raise TypeError("protocol_id must be NativeSuccessObservationProtocolId")
        validate_success_observation_protocol(
            self.protocol_id,
            tool_profile=self.tool_profile,
            assistant_dialect=self.assistant_dialect,
        )

    def render(self, parsed_call: NativeToolCall) -> str:
        """Render and validate the exact success bytes for one accepted call."""

        _validate_call_matches_tool_profile(parsed_call, self.tool_profile)
        if (
            self.protocol_id
            is NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
        ):
            rendered = render_qwen_native_matched_crop_success_environment_text(
                parsed_call,
                assistant_dialect=self.assistant_dialect,
            )
        elif (
            self.protocol_id
            is NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC86_V1
        ):
            rendered = (
                render_qwen_native_legacy_crop_generic86_success_environment_text(
                    parsed_call,
                    assistant_dialect=self.assistant_dialect,
                )
            )
        else:
            rendered = render_qwen_native_success_environment_text(
                parsed_call,
                assistant_dialect=self.assistant_dialect,
            )
        _validate_success_environment_text(rendered)
        return rendered


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
        observation_contract: NativeSuccessObservationContract,
        visual_token_count_resolver: NativeObservationVisualTokenCountResolver
        | None = None,
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
        if not isinstance(observation_contract, NativeSuccessObservationContract):
            raise TypeError(
                "observation_contract must be NativeSuccessObservationContract"
            )
        self.tokenizer = tokenizer
        self.registrar = registrar
        self.observation_contract = observation_contract
        self.visual_token_count_resolver = visual_token_count_resolver

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
            environment_text = self.observation_contract.render(parsed_call)
        else:
            if parsed_call is not None:
                raise ValueError("error tool response must not receive a parsed call")
            environment_text = (
                QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX
                + observation.canonical_json
                + qwen_native_response_suffix(
                    self.observation_contract.assistant_dialect
                )
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


def render_qwen_native_matched_crop_success_environment_text(
    parsed_call: NativeToolCall,
    *,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_INSTRUCT
    ),
) -> str:
    """Render the canonical DeepEyes Crop continuation used by evaluation."""

    _validate_crop_call_and_instruct_dialect(
        parsed_call,
        assistant_dialect=assistant_dialect,
        contract_label="matched Crop",
    )
    if (
        "<answer>" in QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT
        or "</answer>" in QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT
    ):
        raise RuntimeError("matched Crop observation introduced an answer wrapper")
    return QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT


def render_qwen_native_legacy_crop_generic86_success_environment_text(
    parsed_call: NativeToolCall,
    *,
    assistant_dialect: NativeAssistantDialect = (
        NativeAssistantDialect.QWEN3_VL_INSTRUCT
    ),
) -> str:
    """Render the immutable historical Instruct Crop generic continuation."""

    _validate_crop_call_and_instruct_dialect(
        parsed_call,
        assistant_dialect=assistant_dialect,
        contract_label="legacy generic86 Crop",
    )
    return QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT


def qwen_native_response_suffix(
    assistant_dialect: NativeAssistantDialect,
) -> str:
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING:
        return QWEN_NATIVE_RESPONSE_SUFFIX
    if assistant_dialect is NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        return QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX
    raise TypeError("assistant_dialect must be NativeAssistantDialect")


def _validate_crop_call_and_instruct_dialect(
    parsed_call: NativeToolCall,
    *,
    assistant_dialect: NativeAssistantDialect,
    contract_label: str,
) -> None:
    if not isinstance(parsed_call, ParsedImageZoomInCall):
        raise TypeError(f"{contract_label} response requires a parsed Crop call")
    if parsed_call.name != DEEPEYES_TOOL_NAME:
        raise ValueError(f"{contract_label} response received another tool")
    if assistant_dialect is not NativeAssistantDialect.QWEN3_VL_INSTRUCT:
        raise ValueError(f"{contract_label} response requires Qwen3-VL Instruct")


def _validate_call_matches_tool_profile(
    parsed_call: NativeToolCall,
    tool_profile: NativeToolCapabilityProfile,
) -> None:
    expected_types = {
        NativeToolCapabilityProfile.CROP_ONLY: ParsedImageZoomInCall,
        NativeToolCapabilityProfile.TGVF_ONLY: ParsedToolCall,
        NativeToolCapabilityProfile.CROP_TGVF: ParsedCropTGVFCall,
    }
    expected_type = expected_types[tool_profile]
    if not isinstance(parsed_call, expected_type):
        raise TypeError(
            f"{tool_profile.value} observation contract received another call type"
        )
    if parsed_call.name not in tool_profile.tool_names:
        raise ValueError(
            f"{tool_profile.value} observation contract received another tool"
        )


def _validate_success_environment_text(rendered: object) -> None:
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("native success observation must be non-empty text")
    if rendered.count(QWEN_NATIVE_IMAGE_PLACEHOLDER) != 1:
        raise ValueError(
            "native success observation must contain exactly one image placeholder"
        )


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
    "NativeSuccessObservationContract",
    "NativeToolTurnRegistrar",
    "QWEN_NATIVE_IMAGE_PLACEHOLDER",
    "QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX",
    "QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT",
    "QWEN_NATIVE_LEGACY_CROP_GENERIC86_SUCCESS_TEXT_SHA256",
    "QWEN_NATIVE_MATCHED_CROP_SUCCESS_PREFIX",
    "QWEN_NATIVE_MATCHED_CROP_SUCCESS_SUFFIX",
    "QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT",
    "QWEN_NATIVE_MATCHED_CROP_SUCCESS_TEXT_SHA256",
    "QWEN_NATIVE_RESPONSE_SUFFIX",
    "QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX",
    "QwenNativeToolObservationAppender",
    "qwen_native_response_suffix",
    "render_qwen_native_legacy_crop_generic86_success_environment_text",
    "render_qwen_native_matched_crop_success_environment_text",
    "render_qwen_native_success_environment_text",
    "render_qwen_native_success_payload",
]
