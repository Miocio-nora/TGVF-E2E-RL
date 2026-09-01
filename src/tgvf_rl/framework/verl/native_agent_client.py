"""Exact sync/async client for veRL's async policy server manager.

The accepted upstream veRL ``LLMServerClient.generate`` API returns only
``TokenOutput(token_ids, log_probs, stop_reason, extra_fields)``.  It does not
echo prompt IDs, tag the log-probability convention, or retain vLLM's distinct
``finish_reason``/``stop_reason`` pair.  This leaf therefore:

* passes exact prompt IDs and recorded precomputed image items itself;
* requires an explicit accepted ``processed_logprobs`` runtime identity;
* preserves the returned selected-token log probabilities without replay; and
* restores termination only when emitted terminal tokens or the exact requested
  length make the result unambiguous, otherwise failing closed.

The native async path awaits the server manager directly on its owning event
loop.  The compatibility sync path can still run in a worker thread and submit
calls back with ``asyncio.run_coroutine_threadsafe``.  The implementation has no
dependency on the public native-agent-loop facade.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import inspect
import math
import threading
from types import MappingProxyType

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)
from tgvf_rl.framework.async_worker import run_side_effecting_in_thread
from tgvf_rl.framework.vllm import (
    FastTokenizerTokenByteSpanDecoder,
    VLLM_PROCESSED_LOGPROBS_MODE,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMTokenLogprob,
)
from tgvf_rl.framework.vllm.live_client import VLLMLivePromptInputsPort
from tgvf_rl.framework.vllm.preexpanded_prompt import (
    require_preexpanded_prompt_contract,
)
from tgvf_rl.framework.vllm.registration import SUPPORTED_VLLM_VERSION
from tgvf_rl.public_api_compat import (
    rebind_public_class,
    rebind_public_function,
)

from .compatibility import SUPPORTED_LOGPROBS_MODE


_LEGACY_PUBLIC_MODULE = "tgvf_rl.framework.verl.native_agent_loop"


@dataclass(frozen=True, slots=True)
class _PreparedVerlGeneration:
    """CPU-validated immutable inputs for one owner-loop server invocation."""

    request_id: str
    prompt_ids: tuple[int, ...]
    sampling_params: Mapping[str, object]
    image_data: tuple[object, ...]
    mm_processor_kwargs: Mapping[str, object]
    expected_step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "image_data", tuple(self.image_data))
        object.__setattr__(
            self,
            "sampling_params",
            MappingProxyType(
                {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in self.sampling_params.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "mm_processor_kwargs",
            MappingProxyType(dict(self.mm_processor_kwargs)),
        )


class VerlAsyncServerPolicyTurnClient:
    """Exact turn client backed by one upstream async server manager."""

    def __init__(
        self,
        *,
        server_manager: object,
        event_loop: asyncio.AbstractEventLoop,
        tokenizer: object,
        prompt_inputs: VLLMLivePromptInputsPort,
        token_byte_span_decoder: FastTokenizerTokenByteSpanDecoder,
        sticky_request_id: str,
        max_model_len: int,
        server_timeout_seconds: float,
        backend_version: str = SUPPORTED_VLLM_VERSION,
        logprobs_mode: str,
    ) -> None:
        if not callable(getattr(server_manager, "generate", None)):
            raise TypeError("veRL server_manager must implement async generate()")
        if (
            not isinstance(event_loop, asyncio.AbstractEventLoop)
            or not event_loop.is_running()
        ):
            raise RuntimeError(
                "veRL server bridge requires the running owner event loop"
            )
        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("veRL server bridge tokenizer must implement decode()")
        if not callable(getattr(prompt_inputs, "for_request", None)):
            raise TypeError("prompt_inputs must implement for_request()")
        if not isinstance(token_byte_span_decoder, FastTokenizerTokenByteSpanDecoder):
            raise TypeError(
                "token_byte_span_decoder must be FastTokenizerTokenByteSpanDecoder"
            )
        if not isinstance(sticky_request_id, str) or not sticky_request_id:
            raise ValueError("sticky_request_id must be non-empty")
        if type(max_model_len) is not int or max_model_len <= 1:
            raise ValueError("max_model_len must be an integer greater than one")
        if (
            isinstance(server_timeout_seconds, bool)
            or not isinstance(server_timeout_seconds, (int, float))
            or not math.isfinite(float(server_timeout_seconds))
            or float(server_timeout_seconds) <= 0
        ):
            raise ValueError("server_timeout_seconds must be finite and positive")
        if backend_version != SUPPORTED_VLLM_VERSION:
            raise IdentityMismatchError(
                "veRL native sampler requires the accepted vLLM 0.12.0 backend"
            )
        if (
            logprobs_mode != SUPPORTED_LOGPROBS_MODE
            or logprobs_mode != VLLM_PROCESSED_LOGPROBS_MODE
        ):
            raise IdentityMismatchError(
                "veRL native sampler requires processed_logprobs runtime identity"
            )

        self.server_manager = server_manager
        self.event_loop = event_loop
        self.tokenizer = tokenizer
        self.prompt_inputs = prompt_inputs
        self.token_byte_span_decoder = token_byte_span_decoder
        self.sticky_request_id = sticky_request_id
        self.max_model_len = max_model_len
        self.server_timeout_seconds = float(server_timeout_seconds)
        self.backend_version = backend_version
        self.logprobs_mode = logprobs_mode
        self._event_loop_thread_id = threading.get_ident()

    def generate(self, request: VLLMPolicyTurnRequest) -> VLLMPolicyTurnResponse:
        if threading.get_ident() == self._event_loop_thread_id:
            raise RuntimeError(
                "sync veRL policy client must run in the native-loop worker thread"
            )
        prepared = self._prepare_generation(request)
        generation = self._generate_prepared(prepared)
        try:
            future = asyncio.run_coroutine_threadsafe(generation, self.event_loop)
        except BaseException:
            generation.close()
            raise
        try:
            output = future.result(timeout=self.server_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError("veRL server_manager generation timed out") from error
        return self._response_from_output(request, output)

    async def generate_async(
        self,
        request: VLLMPolicyTurnRequest,
    ) -> VLLMPolicyTurnResponse:
        """Await one server-manager turn on its owner loop without a parked thread."""

        if asyncio.get_running_loop() is not self.event_loop:
            raise RuntimeError(
                "async veRL policy client must run on the server-manager owner loop"
            )
        prepared = await run_side_effecting_in_thread(
            self._prepare_generation,
            request,
        )
        try:
            output = await asyncio.wait_for(
                self._generate_prepared(prepared),
                timeout=self.server_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError("veRL server_manager generation timed out") from error
        return await asyncio.to_thread(self._response_from_output, request, output)

    def _prepare_generation(
        self,
        request: VLLMPolicyTurnRequest,
    ) -> _PreparedVerlGeneration:
        if not isinstance(request, VLLMPolicyTurnRequest):
            raise TypeError("veRL policy client requires VLLMPolicyTurnRequest")
        if request.backend_version != self.backend_version:
            raise IdentityMismatchError("request/backend vLLM versions differ")
        if request.logprobs_mode != self.logprobs_mode:
            raise IdentityMismatchError("request logprob convention is not processed")

        inputs = self.prompt_inputs.for_request(request)
        if inputs.multi_modal_uuids is not None:
            raise ContractUnsetError(
                "upstream veRL LLMServerClient.generate has no multi_modal_uuids field"
            )
        modalities = set(inputs.multi_modal_data)
        if modalities != {"image"}:
            raise ContractUnsetError(
                "this Qwen-VL veRL boundary requires exactly multi_modal_data['image']"
            )
        image_data = inputs.multi_modal_data["image"]
        if not isinstance(image_data, Sequence) or isinstance(image_data, (str, bytes)):
            raise TypeError("recorded image_data must be a multimodal item sequence")
        if not image_data:
            raise ReplayMismatchError("recorded image_data cannot be empty")
        require_preexpanded_prompt_contract(
            inputs.mm_processor_kwargs,
            prompt_token_ids=request.prompt_token_ids,
            expected_image_items=len(image_data),
        )

        sampling_params = _verl_server_sampling_parameters(
            request, max_model_len=self.max_model_len
        )
        return _PreparedVerlGeneration(
            request_id=self.sticky_request_id,
            prompt_ids=request.prompt_token_ids,
            sampling_params=sampling_params,
            image_data=tuple(image_data),
            mm_processor_kwargs=inputs.mm_processor_kwargs,
            expected_step=request.behavior_policy.optimizer_step,
        )

    async def _generate_prepared(
        self,
        prepared: _PreparedVerlGeneration,
    ) -> object:
        """Invoke and await the server manager only on its owning event loop."""

        if asyncio.get_running_loop() is not self.event_loop:
            raise RuntimeError("veRL server generation must run on its owner loop")
        if not isinstance(prepared, _PreparedVerlGeneration):
            raise TypeError("veRL generation requires prepared request inputs")
        sampling_params = dict(prepared.sampling_params)
        for field_name in (
            "stop",
            "stop_token_ids",
            "bad_words",
            "allowed_token_ids",
        ):
            if isinstance(sampling_params.get(field_name), tuple):
                sampling_params[field_name] = list(sampling_params[field_name])
        awaitable = self.server_manager.generate(
            request_id=prepared.request_id,
            prompt_ids=list(prepared.prompt_ids),
            sampling_params=sampling_params,
            image_data=list(prepared.image_data),
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=dict(prepared.mm_processor_kwargs),
            tgvf_expected_step=prepared.expected_step,
        )
        if not inspect.isawaitable(awaitable):
            raise TypeError("veRL server_manager.generate() must return an awaitable")
        return await awaitable

    def _response_from_output(
        self,
        request: VLLMPolicyTurnRequest,
        output: object,
    ) -> VLLMPolicyTurnResponse:
        token_ids = _token_ids(getattr(output, "token_ids", None))
        logprobs = _selected_processed_logprobs(
            getattr(output, "log_probs", None), token_ids=token_ids
        )
        extra_fields = getattr(output, "extra_fields", None)
        _validate_policy_step_evidence(extra_fields, request=request)
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=request.decoding.skip_special_tokens,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=(
                request.decoding.spaces_between_special_tokens
            ),
        )
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode() must return str")
        spans = self.token_byte_span_decoder.spans_for_output(
            text=text,
            token_ids=token_ids,
            decoding=request.decoding,
        )
        finish_reason, stop_reason = _recover_termination(
            request=request,
            token_ids=token_ids,
            text=text,
            upstream_stop_reason=getattr(output, "stop_reason", None),
            exact_finish_reason=(
                extra_fields.get("tgvf_vllm_finish_reason")
                if isinstance(extra_fields, Mapping)
                else None
            ),
            exact_stop_reason=(
                extra_fields.get("tgvf_vllm_stop_reason")
                if isinstance(extra_fields, Mapping)
                else None
            ),
        )
        return VLLMPolicyTurnResponse(
            request_id=request.request_id,
            backend_request_sha256=request.backend_request_sha256,
            prompt_token_ids=request.prompt_token_ids,
            text=text,
            token_ids=token_ids,
            token_byte_spans=spans,
            token_logprobs=tuple(
                (
                    VLLMTokenLogprob(
                        token_id=token_id,
                        logprob=logprob,
                    ),
                )
                for token_id, logprob in zip(token_ids, logprobs, strict=True)
            ),
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )


def _verl_server_sampling_parameters(
    request: VLLMPolicyTurnRequest,
    *,
    max_model_len: int,
) -> dict[str, object]:
    values = dict(request.sampling_parameters)
    if values.get("logprobs") != 0:
        raise IdentityMismatchError(
            "normalized vLLM request must select sampled-token logprobs only"
        )
    if values.get("n") != 1 or values.get("prompt_logprobs") is not None:
        raise IdentityMismatchError(
            "veRL policy turn requires n=1 and no prompt logprobs"
        )
    if values.get("output_kind") != "final_only":
        raise IdentityMismatchError("veRL policy turn requires final_only output")
    if values.get("logits_processors") is not None:
        raise IdentityMismatchError("Policy Pilot forbids rollout logit processors")
    max_tokens = values.get("max_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("veRL policy turn max_tokens must be positive")
    if len(request.prompt_token_ids) + max_tokens > max_model_len:
        raise ContractUnsetError(
            "requested turn would trigger upstream's silent max_tokens clamp"
        )

    # The upstream manager accepts a bool and its vLLM server converts True to
    # SamplingParams(logprobs=0).  Passing the already-normalized integer zero
    # would be interpreted as false and silently discard behavior logprobs.
    values["logprobs"] = True
    values.pop("output_kind")
    for field_name in ("stop", "stop_token_ids", "bad_words", "allowed_token_ids"):
        if isinstance(values.get(field_name), tuple):
            values[field_name] = list(values[field_name])
    return values


def _token_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("veRL TokenOutput.token_ids must be a sequence")
    token_ids = tuple(value)
    if not token_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise ReplayMismatchError(
            "veRL TokenOutput.token_ids must be non-empty non-negative integers"
        )
    return token_ids


def _selected_processed_logprobs(
    value: object,
    *,
    token_ids: tuple[int, ...],
) -> tuple[float, ...]:
    if value is None:
        raise ReplayMismatchError("veRL TokenOutput omitted behavior log_probs")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("veRL TokenOutput.log_probs must be a sequence")
    if len(value) != len(token_ids):
        raise ReplayMismatchError("veRL token_ids/log_probs counts differ")
    normalized: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("veRL selected behavior logprob must be numeric")
        item = float(raw)
        if not math.isfinite(item) or item > 1e-6:
            raise ReplayMismatchError(
                "veRL selected processed behavior logprob is invalid"
            )
        normalized.append(item)
    return tuple(normalized)


def _validate_policy_step_evidence(
    value: object,
    *,
    request: VLLMPolicyTurnRequest,
) -> None:
    if not isinstance(value, Mapping):
        raise IdentityMismatchError(
            "veRL TokenOutput must carry min/max policy-step evidence"
        )
    minimum = value.get("min_global_steps")
    maximum = value.get("max_global_steps")
    expected = request.behavior_policy.optimizer_step
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or minimum != maximum
        or minimum != expected
    ):
        raise IdentityMismatchError(
            "veRL rollout policy-step evidence differs from behavior policy"
        )
    mode = value.get("logprobs_mode")
    if mode is not None and mode != VLLM_PROCESSED_LOGPROBS_MODE:
        raise IdentityMismatchError("veRL output reports a non-processed logprob mode")


def _recover_termination(
    *,
    request: VLLMPolicyTurnRequest,
    token_ids: tuple[int, ...],
    text: str,
    upstream_stop_reason: object,
    exact_finish_reason: object = None,
    exact_stop_reason: object = None,
) -> tuple[str, int | str | None]:
    if upstream_stop_reason != "completed":
        raise ReplayMismatchError(
            "veRL generation did not return one completed final TokenOutput"
        )
    if exact_finish_reason is not None:
        if exact_finish_reason not in ("stop", "length"):
            raise ReplayMismatchError("exact vLLM finish reason is unsupported")
        if isinstance(exact_stop_reason, bool) or (
            exact_stop_reason is not None
            and not isinstance(exact_stop_reason, (int, str))
        ):
            raise TypeError("exact vLLM stop reason must be int, str, or None")
        if exact_finish_reason == "length":
            if exact_stop_reason is not None:
                raise ReplayMismatchError(
                    "length termination cannot carry an exact stop reason"
                )
            return "length", None
        return "stop", exact_stop_reason
    parameters = request.sampling_parameters
    max_tokens = int(parameters["max_tokens"])
    reached_length = len(token_ids) == max_tokens
    stop_strings = tuple(parameters["stop"])
    include_stop = parameters["include_stop_str_in_output"]
    if stop_strings and include_stop is not True:
        raise ContractUnsetError(
            "veRL TokenOutput cannot recover excluded stop-string identity"
        )
    matching_strings = tuple(stop for stop in stop_strings if text.endswith(stop))
    matching_token = (
        token_ids[-1] if token_ids[-1] in tuple(parameters["stop_token_ids"]) else None
    )
    terminal_evidence_count = len(matching_strings) + int(matching_token is not None)
    if terminal_evidence_count > 1:
        raise ReplayMismatchError(
            "veRL completed output has multiple stop termination signals"
        )
    if len(matching_strings) == 1:
        return "stop", matching_strings[0]
    if matching_token is not None:
        return "stop", matching_token
    if reached_length:
        return "length", None
    raise ReplayMismatchError(
        "veRL completed output has no recoverable stop/length evidence"
    )


# Preserve the historical facade/pickle coordinates while leaving a one-way
# static dependency from the facade to this implementation leaf.
rebind_public_class(
    VerlAsyncServerPolicyTurnClient,
    implementation_module=__name__,
    public_module=_LEGACY_PUBLIC_MODULE,
)
for _legacy_function in (
    _verl_server_sampling_parameters,
    _token_ids,
    _selected_processed_logprobs,
    _validate_policy_step_evidence,
    _recover_termination,
):
    rebind_public_function(
        _legacy_function,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _legacy_function


__all__ = ["VerlAsyncServerPolicyTurnClient"]
