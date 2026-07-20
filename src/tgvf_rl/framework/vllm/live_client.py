"""Live, dependency-injected ``vLLM.LLM.generate`` policy-turn client.

The module deliberately imports no vLLM package at import time.  A caller
owns the live engine, the already-materialized multimodal payload, and the
audited token-to-byte decoder.  This client only constructs one token prompt,
constructs one ``SamplingParams`` value, executes one final-only completion,
and normalizes the actual processed sampled-token log probabilities.

It never regenerates a TGVF observation and never infers multimodal tensors
from the content hash carried by :class:`VLLMPolicyTurnRequest`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.protocol.schema import TokenByteSpan

from .registration import SUPPORTED_VLLM_VERSION
from .sampler import (
    VLLM_PROCESSED_LOGPROBS_MODE,
    VLLMOutputDecodingContract,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMTokenLogprob,
)


@dataclass(frozen=True, slots=True)
class VLLMLivePromptInputs:
    """One content-identified, already-materialized vLLM token-prompt payload."""

    backend_prompt_payload_sha256: str
    multi_modal_data: Mapping[str, object]
    mm_processor_kwargs: Mapping[str, object] | None = None
    multi_modal_uuids: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if len(self.backend_prompt_payload_sha256) != 64 or any(
            char not in "0123456789abcdef"
            for char in self.backend_prompt_payload_sha256
        ):
            raise ValueError("backend prompt payload identity must be lowercase SHA256")
        if not isinstance(self.multi_modal_data, Mapping) or not self.multi_modal_data:
            raise TypeError("live vLLM multi_modal_data must be a non-empty mapping")
        for field_name in ("mm_processor_kwargs", "multi_modal_uuids"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping or None")


class VLLMLivePromptInputsPort(Protocol):
    """Resolve the exact rollout-materialized visual payload for one request."""

    def for_request(self, request: VLLMPolicyTurnRequest) -> VLLMLivePromptInputs: ...


class VLLMTokenByteSpanDecoderPort(Protocol):
    """Return audited byte coverage for the exact final vLLM completion."""

    def spans_for_output(
        self,
        *,
        text: str,
        token_ids: tuple[int, ...],
        decoding: VLLMOutputDecodingContract,
    ) -> Sequence[TokenByteSpan]: ...


SamplingParamsFactory = Callable[[Mapping[str, object]], object]


class VLLMLivePolicyTurnClient:
    """Execute a single audited policy turn against an injected live vLLM engine.

    The engine must expose ``model_config.logprobs_mode`` so this boundary can
    prove that returned values are processed log probabilities.  Merely
    labelling raw log probabilities as processed is rejected.
    """

    logprobs_mode = VLLM_PROCESSED_LOGPROBS_MODE

    def __init__(
        self,
        *,
        engine: object,
        prompt_inputs: VLLMLivePromptInputsPort,
        token_byte_span_decoder: VLLMTokenByteSpanDecoderPort,
        backend_version: str = SUPPORTED_VLLM_VERSION,
        sampling_params_factory: SamplingParamsFactory | None = None,
    ) -> None:
        if not callable(getattr(engine, "generate", None)):
            raise TypeError("live vLLM engine must implement generate()")
        if not callable(getattr(prompt_inputs, "for_request", None)):
            raise TypeError("prompt_inputs must implement for_request()")
        if not callable(getattr(token_byte_span_decoder, "spans_for_output", None)):
            raise TypeError("token_byte_span_decoder must implement spans_for_output()")
        if backend_version != SUPPORTED_VLLM_VERSION:
            raise IdentityMismatchError(
                "live policy client requires the exact audited vLLM 0.12.0 backend"
            )
        if sampling_params_factory is not None and not callable(
            sampling_params_factory
        ):
            raise TypeError("sampling_params_factory must be callable or None")

        self.engine = engine
        self.prompt_inputs = prompt_inputs
        self.token_byte_span_decoder = token_byte_span_decoder
        self.backend_version = backend_version
        self._sampling_params_factory = (
            sampling_params_factory or _default_sampling_params_factory
        )
        self._validate_engine_logprobs_mode()

    def generate(self, request: VLLMPolicyTurnRequest) -> VLLMPolicyTurnResponse:
        if not isinstance(request, VLLMPolicyTurnRequest):
            raise TypeError("live vLLM client requires VLLMPolicyTurnRequest")
        if request.backend_version != self.backend_version:
            raise IdentityMismatchError(
                "request vLLM version differs from the injected live client"
            )
        if request.logprobs_mode != self.logprobs_mode:
            raise IdentityMismatchError(
                "live vLLM request must require processed_logprobs"
            )
        self._validate_engine_logprobs_mode()

        inputs = self.prompt_inputs.for_request(request)
        if not isinstance(inputs, VLLMLivePromptInputs):
            raise TypeError("prompt_inputs must return VLLMLivePromptInputs")
        if (
            inputs.backend_prompt_payload_sha256
            != request.backend_prompt_payload_sha256
        ):
            raise IdentityMismatchError(
                "materialized vLLM prompt payload differs from the request identity"
            )

        prompt: dict[str, object] = {
            "prompt_token_ids": list(request.prompt_token_ids),
            # Do not copy, rebuild, or normalize recorded visual tensors.
            "multi_modal_data": inputs.multi_modal_data,
        }
        if inputs.mm_processor_kwargs is not None:
            prompt["mm_processor_kwargs"] = inputs.mm_processor_kwargs
        if inputs.multi_modal_uuids is not None:
            prompt["multi_modal_uuids"] = inputs.multi_modal_uuids

        sampling_params = self._sampling_params_factory(request.sampling_parameters)
        if sampling_params is None:
            raise TypeError("sampling_params_factory returned None")
        raw_outputs = self.engine.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        raw_request, raw_completion = _single_final_completion(
            raw_outputs, request=request
        )

        text = getattr(raw_completion, "text", None)
        if not isinstance(text, str):
            raise TypeError("vLLM completion text must be str")
        token_ids = _token_ids(
            getattr(raw_completion, "token_ids", None),
            field_name="vLLM completion token_ids",
        )
        token_logprobs = _normalize_logprobs(
            getattr(raw_completion, "logprobs", None), token_ids=token_ids
        )

        spans = self.token_byte_span_decoder.spans_for_output(
            text=text,
            token_ids=token_ids,
            decoding=request.decoding,
        )
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
            raise TypeError("token byte-span decoder must return a sequence")

        finish_reason = getattr(raw_completion, "finish_reason", None)
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ReplayMismatchError("vLLM completion is not final")
        stop_reason = getattr(raw_completion, "stop_reason", None)
        if isinstance(stop_reason, bool) or (
            stop_reason is not None and not isinstance(stop_reason, (int, str))
        ):
            raise TypeError("vLLM stop_reason must be int, str, or None")

        # _single_final_completion already validates these fields.  Keep the
        # local names explicit so no backend-owned identity is silently used
        # as the project request identity.
        del raw_request
        return VLLMPolicyTurnResponse(
            request_id=request.request_id,
            backend_request_sha256=request.backend_request_sha256,
            prompt_token_ids=request.prompt_token_ids,
            text=text,
            token_ids=token_ids,
            token_byte_spans=tuple(spans),
            token_logprobs=token_logprobs,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            output_index=0,
            finished=True,
        )

    def _validate_engine_logprobs_mode(self) -> None:
        model_config = getattr(self.engine, "model_config", None)
        actual_mode = getattr(model_config, "logprobs_mode", None)
        if actual_mode != VLLM_PROCESSED_LOGPROBS_MODE:
            raise IdentityMismatchError(
                "injected vLLM engine must expose "
                "model_config.logprobs_mode='processed_logprobs'"
            )


def _default_sampling_params_factory(
    parameters: Mapping[str, object],
) -> object:
    """Build vLLM 0.12 SamplingParams without an import-time dependency."""

    values = dict(parameters)
    output_kind = values.get("output_kind")
    if output_kind != "final_only":
        raise IdentityMismatchError(
            "live policy client requires vLLM final_only output_kind"
        )
    # VLLMPolicyTurnRequest freezes JSON lists as tuples.  vLLM 0.12's
    # SamplingParams validates these public sequence fields as concrete lists.
    for field_name in ("stop", "stop_token_ids", "bad_words", "allowed_token_ids"):
        if isinstance(values.get(field_name), tuple):
            values[field_name] = list(values[field_name])

    from vllm import SamplingParams  # type: ignore[import-not-found]
    from vllm.sampling_params import (  # type: ignore[import-not-found]
        RequestOutputKind,
    )

    values["output_kind"] = RequestOutputKind.FINAL_ONLY
    return SamplingParams(**values)


def _single_final_completion(
    raw_outputs: object,
    *,
    request: VLLMPolicyTurnRequest,
) -> tuple[object, object]:
    if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
        raise TypeError("vLLM generate() must return a response sequence")
    if len(raw_outputs) != 1:
        raise ReplayMismatchError("vLLM returned an unexpected request count")
    raw_request = raw_outputs[0]
    if getattr(raw_request, "finished", None) is not True:
        raise ReplayMismatchError("vLLM request output is not final")
    echoed_prompt = _token_ids(
        getattr(raw_request, "prompt_token_ids", None),
        field_name="vLLM echoed prompt_token_ids",
    )
    if echoed_prompt != request.prompt_token_ids:
        raise ReplayMismatchError("vLLM changed the submitted prompt token IDs")

    completions = getattr(raw_request, "outputs", None)
    if not isinstance(completions, Sequence) or isinstance(completions, (str, bytes)):
        raise TypeError("vLLM request outputs must be a sequence")
    if len(completions) != 1:
        raise ReplayMismatchError("vLLM returned an unexpected completion count")
    completion = completions[0]
    if getattr(completion, "index", None) != 0:
        raise ReplayMismatchError("vLLM completion index must be zero")
    return raw_request, completion


def _token_ids(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    normalized = tuple(value)
    if any(type(token_id) is not int or token_id < 0 for token_id in normalized):
        raise ValueError(f"{field_name} must contain non-negative integers")
    return normalized


def _normalize_logprobs(
    value: object,
    *,
    token_ids: tuple[int, ...],
) -> tuple[tuple[VLLMTokenLogprob, ...], ...]:
    if value is None:
        raise ReplayMismatchError("vLLM omitted sampled-token logprobs")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("vLLM completion logprobs must be a position sequence")
    if len(value) != len(token_ids):
        raise ReplayMismatchError(
            "vLLM sampled token and logprob position counts differ"
        )

    normalized_positions: list[tuple[VLLMTokenLogprob, ...]] = []
    for position_index, (sampled_token_id, position) in enumerate(
        zip(token_ids, value, strict=True)
    ):
        if not isinstance(position, Mapping) or not position:
            raise ReplayMismatchError(
                f"vLLM logprob position {position_index} is missing"
            )
        entries: list[VLLMTokenLogprob] = []
        selected_count = 0
        for raw_token_id, raw_entry in position.items():
            if type(raw_token_id) is not int or raw_token_id < 0:
                raise ValueError("vLLM logprob mapping keys must be token IDs")
            raw_logprob = getattr(raw_entry, "logprob", None)
            if isinstance(raw_logprob, bool) or not isinstance(
                raw_logprob, (int, float)
            ):
                raise TypeError("vLLM logprob entry must expose numeric logprob")
            logprob = float(raw_logprob)
            if not math.isfinite(logprob):
                raise ReplayMismatchError("vLLM returned a non-finite logprob")
            rank = getattr(raw_entry, "rank", None)
            decoded_token = getattr(raw_entry, "decoded_token", None)
            entry = VLLMTokenLogprob(
                token_id=raw_token_id,
                logprob=logprob,
                rank=rank,
                decoded_token=decoded_token,
            )
            entries.append(entry)
            if raw_token_id == sampled_token_id:
                selected_count += 1
        if selected_count != 1:
            raise ReplayMismatchError(
                "sampled token ID is absent from its processed-logprob mapping"
            )
        normalized_positions.append(
            tuple(sorted(entries, key=lambda entry: entry.token_id))
        )
    return tuple(normalized_positions)


__all__ = [
    "SamplingParamsFactory",
    "VLLMLivePolicyTurnClient",
    "VLLMLivePromptInputs",
    "VLLMLivePromptInputsPort",
    "VLLMTokenByteSpanDecoderPort",
]
