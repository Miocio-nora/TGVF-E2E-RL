"""Exact Qwen-native tool-response token appender for policy rollouts."""

from __future__ import annotations

from typing import Protocol

from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.schema import StandardToolError

from .agent_loop import SampledPolicyTurn


QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX = "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
QWEN_NATIVE_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_NATIVE_RESPONSE_SUFFIX = (
    "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
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


class QwenNativeToolObservationAppender:
    """Append environment-owned native bytes without rerendering prior turns.

    The sampled assistant continuation remains byte-for-byte policy owned.  A
    successful TGVF response contributes one native image placeholder; the
    registrar separately binds that placeholder to the rollout-recorded main
    ``D`` and D-DeepStack tensors.  Errors contain the canonical deterministic
    JSON payload and no visual placeholder.
    """

    def __init__(
        self, *, tokenizer: object, registrar: NativeToolTurnRegistrar
    ) -> None:
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("Qwen native appender requires tokenizer.encode()")
        if not callable(getattr(registrar, "register_tool_turn", None)):
            raise TypeError("registrar must implement register_tool_turn()")
        self.tokenizer = tokenizer
        self.registrar = registrar

    def append(
        self,
        prompt_token_ids: tuple[int, ...],
        sampled_turn: SampledPolicyTurn,
        observation: ObservationHandle | StandardToolError,
        *,
        call_index: int,
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

        sampled_ids = self._encode(sampled_turn.text)
        if sampled_ids != sampled_turn.token_ids:
            raise ValueError(
                "sampled assistant text does not round-trip to its exact token IDs"
            )
        payload = (
            QWEN_NATIVE_IMAGE_PLACEHOLDER
            if isinstance(observation, ObservationHandle)
            else observation.canonical_json
        )
        environment_text = (
            QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX + payload + QWEN_NATIVE_RESPONSE_SUFFIX
        )
        environment_ids = self._encode(environment_text)
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


__all__ = [
    "NativeToolTurnRegistrar",
    "QWEN_NATIVE_IMAGE_PLACEHOLDER",
    "QWEN_NATIVE_RESPONSE_SUFFIX",
    "QWEN_NATIVE_SUCCESS_RESPONSE_PREFIX",
    "QwenNativeToolObservationAppender",
]
