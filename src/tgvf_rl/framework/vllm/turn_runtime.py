"""Minimal deterministic runtime support for live vLLM policy turns.

This module owns no model, tokenizer construction, tool algorithm, or data
loading.  It supplies three narrow runtime pieces needed by the existing
sampler/client/appender boundaries:

* deterministic content-addressed per-turn RNG identities;
* exact fast-tokenizer token/UTF-8 byte-span verification; and
* a thread-safe, single-use registry for prompt and visual turn state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.schema import (
    SampledAssistantTurn,
    StandardToolError,
    TokenByteSpan,
)

from .live_client import VLLMLivePromptInputs
from .preexpanded_prompt import (
    rebind_preexpanded_prompt_contract,
    require_preexpanded_prompt_contract,
)
from .sampler import (
    VLLMOutputDecodingContract,
    VLLMPolicyTurnRequest,
    VLLMTurnRNGIdentity,
)


TURN_RNG_SCHEMA = "tgvf-vllm-turn-rng-v1"
TURN_CONTEXT_SCHEMA = "tgvf-vllm-live-turn-context-v1"
_VLLM_SEED_MODULUS = 2**31 - 1


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return value


def _prompt_token_ids(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a token-ID sequence")
    normalized = tuple(value)
    if not normalized or any(
        type(token_id) is not int or token_id < 0 for token_id in normalized
    ):
        raise ValueError(
            f"{field_name} must contain non-negative token IDs and be non-empty"
        )
    return normalized


def prompt_token_ids_sha256(prompt_token_ids: Sequence[int]) -> str:
    """Content-address one exact tokenized prompt without text reconstruction."""

    prompt = _prompt_token_ids(prompt_token_ids, "prompt_token_ids")
    return _canonical_sha256(
        {"schema": "tgvf-tokenized-prompt-v1", "prompt_token_ids": prompt}
    )


class ContentAddressedVLLMTurnRNG:
    """Stateless deterministic vLLM RNG derivation for one named run stream."""

    def __init__(self, *, master_seed: int, stream_identity: str) -> None:
        if type(master_seed) is not int or master_seed < 0 or master_seed >= 2**63:
            raise ValueError("master_seed must be an integer in [0, 2**63)")
        if not isinstance(stream_identity, str) or not stream_identity:
            raise ValueError("stream_identity must be a non-empty string")
        self.master_seed = master_seed
        self.stream_identity = stream_identity
        self.stream_identity_sha256 = hashlib.sha256(
            stream_identity.encode("utf-8")
        ).hexdigest()

    def for_turn(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        turn_index: int,
        behavior_policy: PolicyVersion,
    ) -> VLLMTurnRNGIdentity:
        prompt = _prompt_token_ids(prompt_token_ids, "RNG prompt_token_ids")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        if not isinstance(behavior_policy, PolicyVersion):
            raise TypeError("behavior_policy must be PolicyVersion")

        state_payload = {
            "schema": TURN_RNG_SCHEMA,
            "master_seed": self.master_seed,
            "stream_identity_sha256": self.stream_identity_sha256,
            "behavior_policy": {
                "run_id": behavior_policy.run_id,
                "optimizer_step": behavior_policy.optimizer_step,
                "weights_sha256": behavior_policy.weights_sha256,
            },
            "prompt_token_ids_sha256": prompt_token_ids_sha256(prompt),
            "turn_index": turn_index,
        }
        rng_state_sha256 = _canonical_sha256(state_payload)
        seed_digest = hashlib.sha256(
            b"tgvf-vllm-seed-v1\0" + bytes.fromhex(rng_state_sha256)
        ).digest()
        seed = int.from_bytes(seed_digest[:8], "big") % _VLLM_SEED_MODULUS
        return VLLMTurnRNGIdentity(
            seed=seed,
            rng_state_sha256=rng_state_sha256,
        )


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse byte alphabet used by Qwen's ByteLevel tokenizer."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    extra_index = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_index)
            extra_index += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(
            byte_values, unicode_values, strict=True
        )
    }


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()


def _replacement_decoded_byte_boundaries(raw: bytes) -> tuple[int, ...]:
    """Map raw ByteLevel boundaries through UTF-8 ``errors='replace'``.

    A sampled sequence may legally end part-way through a UTF-8 scalar.  The
    Qwen tokenizer then emits U+FFFD, so raw token bytes and the UTF-8 bytes of
    decoded text have different lengths.  This map keeps every sampled token
    while assigning the decoder-owned replacement bytes contiguously.
    """

    boundaries = [0] * (len(raw) + 1)
    raw_cursor = 0
    decoded_cursor = 0
    replacement_width = len("\ufffd".encode("utf-8"))
    while raw_cursor < len(raw):
        suffix = raw[raw_cursor:]
        try:
            suffix.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            valid_end = raw_cursor + error.start
            for boundary in range(raw_cursor, valid_end + 1):
                boundaries[boundary] = decoded_cursor + boundary - raw_cursor
            decoded_cursor += valid_end - raw_cursor

            invalid_end = raw_cursor + error.end
            invalid_width = invalid_end - valid_end
            if invalid_width <= 0:
                raise ReplayMismatchError("UTF-8 decoder reported an empty error span")
            for offset in range(1, invalid_width + 1):
                boundaries[valid_end + offset] = decoded_cursor + (
                    replacement_width * offset // invalid_width
                )
            decoded_cursor += replacement_width
            raw_cursor = invalid_end
        else:
            for boundary in range(raw_cursor, len(raw) + 1):
                boundaries[boundary] = decoded_cursor + boundary - raw_cursor
            decoded_cursor += len(raw) - raw_cursor
            raw_cursor = len(raw)
    if boundaries[-1] != decoded_cursor:
        raise ReplayMismatchError("UTF-8 replacement boundary map is incomplete")
    return tuple(boundaries)


class FastTokenizerTokenByteSpanDecoder:
    """Recover exact UTF-8 byte coverage for final Qwen ByteLevel tokens."""

    def __init__(self, tokenizer: object) -> None:
        if getattr(tokenizer, "is_fast", None) is not True:
            raise TypeError("exact byte spans require a fast tokenizer")
        if not callable(getattr(tokenizer, "convert_ids_to_tokens", None)):
            raise TypeError(
                "exact byte spans require token-ID to ByteLevel-piece conversion"
            )
        self.tokenizer = tokenizer

    def spans_for_output(
        self,
        *,
        text: str,
        token_ids: tuple[int, ...],
        decoding: VLLMOutputDecodingContract,
    ) -> tuple[TokenByteSpan, ...]:
        if not isinstance(text, str):
            raise TypeError("sampled output text must be str")
        expected_ids = _prompt_token_ids(token_ids, "sampled token_ids")
        if not isinstance(decoding, VLLMOutputDecodingContract):
            raise TypeError("decoding must be VLLMOutputDecodingContract")
        if (
            not decoding.detokenize
            or decoding.skip_special_tokens
            or decoding.spaces_between_special_tokens
        ):
            raise IdentityMismatchError(
                "exact byte spans require detokenize=true, skip_special_tokens=false, "
                "and spaces_between_special_tokens=false"
            )

        # Sampled token IDs are authoritative.  Re-encoding ``text`` is not a
        # valid identity check: a model may sample a non-canonical segmentation
        # such as ["a", "b"] even when the tokenizer would encode "ab" as one
        # merged token.  Qwen's ByteLevel pieces retain the original bytes for
        # every sampled token, including tokens that split one Unicode scalar.
        result = self._byte_level_spans(text=text, token_ids=expected_ids)
        SampledAssistantTurn(text, expected_ids, result)
        return result

    def _byte_level_spans(
        self,
        *,
        text: str,
        token_ids: tuple[int, ...],
    ) -> tuple[TokenByteSpan, ...]:
        converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if not callable(converter):
            raise ReplayMismatchError(
                "overlapping tokenizer offsets require ByteLevel token pieces"
            )
        raw_tokens = converter(list(token_ids), skip_special_tokens=False)
        if not isinstance(raw_tokens, Sequence) or isinstance(
            raw_tokens, (str, bytes)
        ):
            raise TypeError("convert_ids_to_tokens must return a token sequence")
        if len(raw_tokens) != len(token_ids):
            raise ReplayMismatchError(
                "ByteLevel token-piece count differs from sampled token IDs"
            )

        added_ids: set[int] = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
        get_added_vocab = getattr(self.tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            added_vocab = get_added_vocab()
            if not isinstance(added_vocab, Mapping):
                raise TypeError("get_added_vocab must return a mapping")
            added_ids.update(
                token_id
                for token_id in added_vocab.values()
                if type(token_id) is int
            )

        token_bytes: list[bytes] = []
        for token_id, raw_token in zip(token_ids, raw_tokens, strict=True):
            if raw_token is None:
                # Qwen's LM head is padded beyond the tokenizer vocabulary.
                # Those sampled rows are real policy tokens (and therefore
                # retain their logprob), but the tokenizer deterministically
                # decodes them to zero bytes.  Preserve them as zero-width
                # spans instead of losing the sampled-token identity.
                decoded = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                    spaces_between_special_tokens=False,
                )
                if decoded != "":
                    raise ReplayMismatchError(
                        "unmapped tokenizer row did not decode to an empty string"
                    )
                token_bytes.append(b"")
                continue
            if not isinstance(raw_token, str):
                raise TypeError("converted ByteLevel token must be str")
            if token_id in added_ids:
                piece = raw_token.encode("utf-8")
            else:
                try:
                    piece = bytes(_GPT2_BYTE_DECODER[char] for char in raw_token)
                except KeyError as error:
                    raise ReplayMismatchError(
                        "token pieces do not use the audited ByteLevel alphabet"
                    ) from error
            if not piece:
                raise ReplayMismatchError(
                    "sampled ByteLevel tokens must have non-empty byte pieces"
                )
            token_bytes.append(piece)

        raw_bytes = b"".join(token_bytes)
        text_bytes = text.encode("utf-8")
        if raw_bytes == text_bytes:
            decoded_boundaries = tuple(range(len(raw_bytes) + 1))
        else:
            decoded_text = raw_bytes.decode("utf-8", errors="replace")
            if decoded_text != text:
                raise ReplayMismatchError(
                    "ByteLevel token pieces do not reconstruct sampled text exactly"
                )
            decoded_boundaries = _replacement_decoded_byte_boundaries(raw_bytes)
            if decoded_boundaries[-1] != len(text_bytes):
                raise ReplayMismatchError(
                    "ByteLevel replacement spans do not cover sampled text exactly"
                )

        raw_cursor = 0
        result: list[TokenByteSpan] = []
        for token_index, (token_id, piece) in enumerate(
            zip(token_ids, token_bytes, strict=True)
        ):
            raw_end = raw_cursor + len(piece)
            result.append(
                TokenByteSpan(
                    token_index=token_index,
                    token_id=token_id,
                    byte_start=decoded_boundaries[raw_cursor],
                    byte_end=decoded_boundaries[raw_end],
                )
            )
            raw_cursor = raw_end
        return tuple(result)


@dataclass(frozen=True, slots=True)
class VLLMResolvedObservationPayload:
    """One resolver-owned precomputed main-D/DeepStack multimodal item."""

    observation: ObservationHandle
    call_index: int
    modality: str
    multi_modal_data_item: object
    payload_sha256: str
    multi_modal_uuid: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationHandle):
            raise TypeError("resolved payload requires ObservationHandle")
        if not self.observation.observation_id:
            raise ValueError("observation ID must be non-empty")
        _require_sha256(self.observation.record_sha256, "observation record identity")
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("resolved observation call_index must be non-negative")
        if not isinstance(self.modality, str) or not self.modality:
            raise ValueError("resolved observation modality must be non-empty")
        if self.multi_modal_data_item is None:
            raise ValueError("resolved observation multimodal item cannot be None")
        _require_sha256(self.payload_sha256, "resolved observation payload identity")
        if self.multi_modal_uuid is not None and (
            not isinstance(self.multi_modal_uuid, str) or not self.multi_modal_uuid
        ):
            raise ValueError("multi_modal_uuid must be non-empty when provided")


class VLLMObservationPayloadResolver(Protocol):
    """Resolve one recorded observation without rerunning the TGVF Adapter."""

    def resolve(
        self,
        observation: ObservationHandle,
        *,
        call_index: int,
    ) -> VLLMResolvedObservationPayload: ...


@dataclass(slots=True)
class _RegisteredTurn:
    turn_index: int
    prompt_token_ids: tuple[int, ...]
    prompt_sha256: str
    inputs: VLLMLivePromptInputs
    context_claimed: bool = False
    request: VLLMPolicyTurnRequest | None = None
    tool_turn_registered: bool = False


class LiveVLLMTurnContextRegistry:
    """Single-use prompt/payload state shared by sampler, client, and appender."""

    def __init__(self, *, observation_resolver: VLLMObservationPayloadResolver) -> None:
        if not callable(getattr(observation_resolver, "resolve", None)):
            raise TypeError("observation_resolver must implement resolve()")
        self.observation_resolver = observation_resolver
        self._lock = RLock()
        self._turns: dict[int, _RegisteredTurn] = {}
        self._prompt_hashes: set[str] = set()
        self._context_hashes: set[str] = set()
        self._request_ids: set[str] = set()
        self._request_hashes: set[str] = set()
        self._successful_call_indices: set[int] = set()
        self._error_attempt_indices: set[int] = set()
        self._observation_identities: set[tuple[str, str]] = set()

    def register_initial_prompt(
        self,
        prompt_token_ids: tuple[int, ...],
        inputs: VLLMLivePromptInputs,
        *,
        turn_index: int = 0,
    ) -> None:
        prompt = _prompt_token_ids(prompt_token_ids, "initial prompt_token_ids")
        if turn_index != 0:
            raise ValueError("initial prompt must use turn_index 0")
        if not isinstance(inputs, VLLMLivePromptInputs):
            raise TypeError("initial prompt requires exact VLLMLivePromptInputs")
        require_preexpanded_prompt_contract(
            inputs.mm_processor_kwargs,
            prompt_token_ids=prompt,
            expected_image_items=_image_item_count(inputs.multi_modal_data),
        )
        prompt_sha = prompt_token_ids_sha256(prompt)
        with self._lock:
            if self._turns:
                raise ReplayMismatchError(
                    "initial live turn context is already registered"
                )
            self._insert_turn(
                turn_index=turn_index,
                prompt=prompt,
                prompt_sha=prompt_sha,
                inputs=inputs,
            )

    def sha256_for_turn(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        turn_index: int,
    ) -> str:
        prompt = _prompt_token_ids(prompt_token_ids, "context prompt_token_ids")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        with self._lock:
            turn = self._require_turn(turn_index)
            if prompt != turn.prompt_token_ids:
                raise ReplayMismatchError("prompt differs from registered live turn")
            if prompt_token_ids_sha256(prompt) != turn.prompt_sha256:
                raise IdentityMismatchError("registered prompt hash changed")
            if turn.context_claimed:
                raise ReplayMismatchError("live turn context hash was already consumed")
            turn.context_claimed = True
            return turn.inputs.backend_prompt_payload_sha256

    def for_request(self, request: VLLMPolicyTurnRequest) -> VLLMLivePromptInputs:
        if not isinstance(request, VLLMPolicyTurnRequest):
            raise TypeError("live turn registry requires VLLMPolicyTurnRequest")
        with self._lock:
            turn = self._require_turn(request.turn_index)
            if not turn.context_claimed:
                raise ReplayMismatchError(
                    "sampler must claim the live context before client resolution"
                )
            if turn.request is not None:
                raise ReplayMismatchError("live turn request was already resolved")
            if request.prompt_token_ids != turn.prompt_token_ids:
                raise ReplayMismatchError("request prompt differs from registered turn")
            if (
                request.backend_prompt_payload_sha256
                != turn.inputs.backend_prompt_payload_sha256
            ):
                raise IdentityMismatchError(
                    "request context hash differs from registered visual payload"
                )
            request_hash = request.backend_request_sha256
            if (
                request.request_id in self._request_ids
                or request_hash in self._request_hashes
            ):
                raise ReplayMismatchError("live vLLM request identity was reused")
            self._request_ids.add(request.request_id)
            self._request_hashes.add(request_hash)
            turn.request = request
            return turn.inputs

    def register_tool_turn(
        self,
        *,
        previous_prompt_token_ids: tuple[int, ...],
        sampled_turn: object,
        updated_prompt_token_ids: tuple[int, ...],
        observation: ObservationHandle | StandardToolError,
        call_index: int,
    ) -> None:
        previous_prompt = _prompt_token_ids(
            previous_prompt_token_ids, "previous prompt_token_ids"
        )
        updated_prompt = _prompt_token_ids(
            updated_prompt_token_ids, "updated prompt_token_ids"
        )
        if type(call_index) is not int or call_index < 0:
            raise ValueError("call_index must be a non-negative integer")

        # Import lazily to preserve the framework package's safe import order.
        from tgvf_rl.environment.agent_loop import SampledPolicyTurn

        if not isinstance(sampled_turn, SampledPolicyTurn):
            raise TypeError("sampled_turn must be SampledPolicyTurn")
        if not isinstance(observation, (ObservationHandle, StandardToolError)):
            raise TypeError("observation must be an exact handle or standard error")

        with self._lock:
            turn = self._turn_for_exact_prompt(previous_prompt)
            if turn.request is None:
                raise ReplayMismatchError(
                    "tool turn cannot register before live request resolution"
                )
            if turn.tool_turn_registered:
                raise ReplayMismatchError("tool turn was already registered")
            if (
                sampled_turn.backend_request_sha256
                != turn.request.backend_request_sha256
            ):
                raise IdentityMismatchError(
                    "sampled turn came from a different backend request"
                )
            if sampled_turn.sampling.policy_version != turn.request.behavior_policy:
                raise IdentityMismatchError(
                    "sampled turn behavior policy differs from live request"
                )
            expected_prefix = previous_prompt + sampled_turn.token_ids
            if (
                len(updated_prompt) <= len(expected_prefix)
                or updated_prompt[: len(expected_prefix)] != expected_prefix
            ):
                raise ReplayMismatchError(
                    "updated prompt did not preserve prompt plus sampled turn"
                )

            next_turn_index = turn.turn_index + 1
            if isinstance(observation, ObservationHandle):
                next_inputs, event_identity = self._append_observation(
                    turn.inputs, observation=observation, call_index=call_index
                )
            else:
                next_inputs, event_identity = self._preserve_visual_error(
                    turn.inputs, error=observation, call_index=call_index
                )

            rebound_mm_processor_kwargs = rebind_preexpanded_prompt_contract(
                next_inputs.mm_processor_kwargs,
                prompt_token_ids=updated_prompt,
                expected_image_items=_image_item_count(next_inputs.multi_modal_data),
            )

            updated_prompt_sha = prompt_token_ids_sha256(updated_prompt)
            next_context_sha = _canonical_sha256(
                {
                    "schema": TURN_CONTEXT_SCHEMA,
                    "previous_context_sha256": (
                        turn.inputs.backend_prompt_payload_sha256
                    ),
                    "updated_prompt_token_ids_sha256": updated_prompt_sha,
                    "next_turn_index": next_turn_index,
                    "backend_request_sha256": sampled_turn.backend_request_sha256,
                    "backend_response_sha256": sampled_turn.backend_response_sha256,
                    "event": event_identity,
                }
            )
            next_inputs = VLLMLivePromptInputs(
                backend_prompt_payload_sha256=next_context_sha,
                multi_modal_data=next_inputs.multi_modal_data,
                mm_processor_kwargs=rebound_mm_processor_kwargs,
                multi_modal_uuids=next_inputs.multi_modal_uuids,
            )
            self._insert_turn(
                turn_index=next_turn_index,
                prompt=updated_prompt,
                prompt_sha=updated_prompt_sha,
                inputs=next_inputs,
            )
            if isinstance(observation, ObservationHandle):
                self._successful_call_indices.add(call_index)
                self._observation_identities.add(
                    (observation.observation_id, observation.record_sha256)
                )
            else:
                self._error_attempt_indices.add(observation.attempt_index)
            turn.tool_turn_registered = True

    def _append_observation(
        self,
        previous: VLLMLivePromptInputs,
        *,
        observation: ObservationHandle,
        call_index: int,
    ) -> tuple[VLLMLivePromptInputs, Mapping[str, object]]:
        expected_call_index = len(self._successful_call_indices)
        if (
            call_index != expected_call_index
            or call_index in self._successful_call_indices
        ):
            raise ReplayMismatchError(
                "successful observation call indices must be unique and contiguous"
            )
        observation_identity = (observation.observation_id, observation.record_sha256)
        if observation_identity in self._observation_identities:
            raise ReplayMismatchError("observation handle was reused")
        resolved = self.observation_resolver.resolve(observation, call_index=call_index)
        if not isinstance(resolved, VLLMResolvedObservationPayload):
            raise TypeError(
                "observation_resolver must return VLLMResolvedObservationPayload"
            )
        if resolved.observation != observation or resolved.call_index != call_index:
            raise IdentityMismatchError(
                "resolved observation identity/call differs from registrar input"
            )

        previous_items = previous.multi_modal_data.get(resolved.modality)
        if not isinstance(previous_items, Sequence) or isinstance(
            previous_items, (str, bytes)
        ):
            raise ReplayMismatchError(
                "recorded multimodal modality must be an item sequence"
            )
        next_data = dict(previous.multi_modal_data)
        next_data[resolved.modality] = [
            *previous_items,
            resolved.multi_modal_data_item,
        ]

        next_uuids: Mapping[str, object] | None
        if previous.multi_modal_uuids is None:
            if resolved.multi_modal_uuid is not None:
                raise ReplayMismatchError(
                    "cannot append one UUID when prior multimodal items have none"
                )
            next_uuids = None
        else:
            previous_uuids = previous.multi_modal_uuids.get(resolved.modality)
            if not isinstance(previous_uuids, Sequence) or isinstance(
                previous_uuids, (str, bytes)
            ):
                raise ReplayMismatchError(
                    "recorded multimodal UUIDs must be an item sequence"
                )
            if len(previous_uuids) != len(previous_items):
                raise ReplayMismatchError(
                    "recorded multimodal data/UUID cardinalities differ"
                )
            if resolved.multi_modal_uuid is None:
                raise ReplayMismatchError(
                    "UUID-enabled multimodal state requires an observation UUID"
                )
            next_uuid_mapping = dict(previous.multi_modal_uuids)
            next_uuid_mapping[resolved.modality] = [
                *previous_uuids,
                resolved.multi_modal_uuid,
            ]
            next_uuids = next_uuid_mapping

        return (
            VLLMLivePromptInputs(
                backend_prompt_payload_sha256=previous.backend_prompt_payload_sha256,
                multi_modal_data=next_data,
                mm_processor_kwargs=previous.mm_processor_kwargs,
                multi_modal_uuids=next_uuids,
            ),
            {
                "kind": "observation",
                "call_index": call_index,
                "observation_id": observation.observation_id,
                "observation_record_sha256": observation.record_sha256,
                "payload_sha256": resolved.payload_sha256,
                "modality": resolved.modality,
                "multi_modal_uuid": resolved.multi_modal_uuid,
            },
        )

    def _preserve_visual_error(
        self,
        previous: VLLMLivePromptInputs,
        *,
        error: StandardToolError,
        call_index: int,
    ) -> tuple[VLLMLivePromptInputs, Mapping[str, object]]:
        if call_index != error.attempt_index:
            raise IdentityMismatchError(
                "error registrar call_index must equal error attempt_index"
            )
        if error.attempt_index in self._error_attempt_indices:
            raise ReplayMismatchError("tool error attempt index was reused")
        return (
            previous,
            {
                "kind": "error",
                "call_index": call_index,
                "error_payload_sha256": error.payload_sha256,
            },
        )

    def _insert_turn(
        self,
        *,
        turn_index: int,
        prompt: tuple[int, ...],
        prompt_sha: str,
        inputs: VLLMLivePromptInputs,
    ) -> None:
        context_sha = inputs.backend_prompt_payload_sha256
        if turn_index in self._turns:
            raise ReplayMismatchError("live turn index was reused")
        if prompt_sha in self._prompt_hashes:
            raise ReplayMismatchError("live prompt content/hash was reused")
        if context_sha in self._context_hashes:
            raise ReplayMismatchError("live visual context hash was reused")
        self._turns[turn_index] = _RegisteredTurn(
            turn_index=turn_index,
            prompt_token_ids=prompt,
            prompt_sha256=prompt_sha,
            inputs=inputs,
        )
        self._prompt_hashes.add(prompt_sha)
        self._context_hashes.add(context_sha)

    def _require_turn(self, turn_index: int) -> _RegisteredTurn:
        turn = self._turns.get(turn_index)
        if turn is None:
            raise ReplayMismatchError("live turn index is not registered")
        return turn

    def _turn_for_exact_prompt(self, prompt: tuple[int, ...]) -> _RegisteredTurn:
        prompt_sha = prompt_token_ids_sha256(prompt)
        matches = tuple(
            turn
            for turn in self._turns.values()
            if turn.prompt_sha256 == prompt_sha and turn.prompt_token_ids == prompt
        )
        if len(matches) != 1:
            raise ReplayMismatchError("previous prompt has no unique live turn context")
        return matches[0]


def _image_item_count(multi_modal_data: Mapping[str, object]) -> int:
    if set(multi_modal_data) != {"image"}:
        raise ReplayMismatchError(
            "pre-expanded live context requires exactly the image modality"
        )
    items = multi_modal_data["image"]
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("live image multimodal data must be an item sequence")
    if not items:
        raise ReplayMismatchError("live image multimodal data cannot be empty")
    return len(items)


__all__ = [
    "ContentAddressedVLLMTurnRNG",
    "FastTokenizerTokenByteSpanDecoder",
    "LiveVLLMTurnContextRegistry",
    "TURN_CONTEXT_SCHEMA",
    "TURN_RNG_SCHEMA",
    "VLLMObservationPayloadResolver",
    "VLLMResolvedObservationPayload",
    "prompt_token_ids_sha256",
]
