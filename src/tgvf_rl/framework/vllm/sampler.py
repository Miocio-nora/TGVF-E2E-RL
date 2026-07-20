"""Fail-closed vLLM behavior-sampling boundary for native tool turns.

The public vLLM response contains sampled token IDs and selected-token
log-probabilities, but it does not contain the exact token-to-UTF-8 byte
coverage required by the native tool parser.  This adapter therefore consumes
an audited client envelope which must already carry that coverage.  A live
client is not allowed to reconstruct it silently: it must prove its decoder
contract and return spans which exactly cover the emitted text.

This is a normalization boundary, not live ``LLM.generate`` wiring.  The
adapter owns the parts which *are* fixed: one vLLM output, processed
log-probabilities, actual selected-token extraction, request/response content
hashes, the behavior ``SamplingIdentity``, and the one-complete-tool-call turn
boundary.  Exact action/final-turn suffixes, stop/EOS outcomes, and RNG
derivation remain content-identified injected run inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)
from tgvf_rl.contracts.identity import PolicyVersion, _validate_sha256
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    SamplingIdentity,
    TokenSpan,
)
from tgvf_rl.protocol.schema import (
    SampledAssistantTurn,
    TOOL_CALL_CLOSE,
    TokenByteSpan,
)

from .registration import SUPPORTED_VLLM_VERSION

if TYPE_CHECKING:
    from tgvf_rl.environment.agent_loop import SampledPolicyTurn


VLLM_POLICY_TURN_REQUEST_SCHEMA = "tgvf-vllm-policy-turn-request-v1"
VLLM_POLICY_TURN_RESPONSE_SCHEMA = "tgvf-vllm-policy-turn-response-v1"
VLLM_PROCESSED_LOGPROBS_MODE = "processed_logprobs"

_SAMPLING_PARAMETER_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop_token_ids",
        "stop",
        "include_stop_str_in_output",
        "ignore_eos",
        "max_tokens",
        "logprobs",
    }
)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_real(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _integer(value: object, field_name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, list):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.items()
    }


@dataclass(frozen=True, slots=True)
class VLLMTurnRNGIdentity:
    """Explicit per-turn seed plus the exact caller-owned RNG-state identity."""

    seed: int
    rng_state_sha256: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("vLLM turn seed must be an integer")
        _validate_sha256(self.rng_state_sha256)


class VLLMTurnRNGPort(Protocol):
    def for_turn(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        turn_index: int,
        behavior_policy: PolicyVersion,
    ) -> VLLMTurnRNGIdentity: ...


class VLLMRequestContextIdentityPort(Protocol):
    def sha256_for_turn(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        turn_index: int,
    ) -> str:
        """Hash the complete backend prompt payload, including visual state."""


@dataclass(frozen=True, slots=True)
class VLLMOutputDecodingContract:
    """Explicit vLLM detokenization fields needed by the native parser."""

    detokenize: bool
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: str

    def __post_init__(self) -> None:
        for field_name in (
            "detokenize",
            "skip_special_tokens",
            "spaces_between_special_tokens",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if not self.detokenize:
            raise ContractUnsetError(
                "PolicySamplerPort requires exact sampled assistant text"
            )
        if self.skip_special_tokens:
            raise ContractUnsetError(
                "native think/tool markers cannot be skipped during detokenization"
            )
        if self.output_kind not in {"cumulative", "final_only"}:
            raise ValueError("output_kind must be 'cumulative' or 'final_only'")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "detokenize": self.detokenize,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
            "output_kind": self.output_kind,
        }


@dataclass(frozen=True, slots=True)
class VLLMTerminationOutcome:
    """One exact vLLM finish/stop pair accepted by a run-bound golden."""

    finish_reason: str
    stop_reason: int | str | None

    def __post_init__(self) -> None:
        if not self.finish_reason:
            raise ValueError("termination finish_reason must be non-empty")
        if isinstance(self.stop_reason, bool) or (
            self.stop_reason is not None
            and not isinstance(self.stop_reason, (int, str))
        ):
            raise TypeError("termination stop_reason must be int, str, or None")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class VLLMTurnTerminationContract:
    """Run-bound stop/suffix identity; this module supplies no default."""

    required_request_stop_strings: tuple[str, ...]
    required_request_stop_token_ids: tuple[int, ...]
    include_stop_str_in_output: bool
    tool_call_terminal_suffixes: tuple[str, ...]
    tool_call_outcomes: tuple[VLLMTerminationOutcome, ...]
    final_turn_outcomes: tuple[VLLMTerminationOutcome, ...]

    def __post_init__(self) -> None:
        for name in (
            "required_request_stop_strings",
            "required_request_stop_token_ids",
            "tool_call_terminal_suffixes",
            "tool_call_outcomes",
            "final_turn_outcomes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.required_request_stop_strings and not self.required_request_stop_token_ids:
            raise ContractUnsetError("a termination contract requires explicit stops")
        if any(not item for item in self.required_request_stop_strings):
            raise ValueError("required stop strings must be non-empty")
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in self.required_request_stop_token_ids
        ):
            raise ValueError("required stop token IDs must be non-negative")
        if len(set(self.required_request_stop_strings)) != len(
            self.required_request_stop_strings
        ) or len(set(self.required_request_stop_token_ids)) != len(
            self.required_request_stop_token_ids
        ):
            raise ValueError("required termination stops must be unique")
        if not isinstance(self.include_stop_str_in_output, bool):
            raise TypeError("include_stop_str_in_output must be bool")
        if not self.tool_call_terminal_suffixes or len(
            set(self.tool_call_terminal_suffixes)
        ) != len(self.tool_call_terminal_suffixes):
            raise ValueError("tool-call terminal suffixes must be non-empty and unique")
        if any(not isinstance(suffix, str) for suffix in self.tool_call_terminal_suffixes):
            raise TypeError("tool-call terminal suffixes must be strings")
        for name in ("tool_call_outcomes", "final_turn_outcomes"):
            outcomes = getattr(self, name)
            if not outcomes or any(
                not isinstance(outcome, VLLMTerminationOutcome)
                for outcome in outcomes
            ):
                raise TypeError(f"{name} must contain accepted termination outcomes")
            if len(set(outcomes)) != len(outcomes):
                raise ValueError(f"{name} must be unique")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": "tgvf-vllm-turn-termination-v1",
            "required_request_stop_strings": self.required_request_stop_strings,
            "required_request_stop_token_ids": self.required_request_stop_token_ids,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "tool_call_terminal_suffixes": self.tool_call_terminal_suffixes,
            "tool_call_outcomes": tuple(
                outcome.canonical_payload for outcome in self.tool_call_outcomes
            ),
            "final_turn_outcomes": tuple(
                outcome.canonical_payload for outcome in self.final_turn_outcomes
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload)


@dataclass(frozen=True, slots=True)
class VLLMTokenLogprob:
    """One normalized entry from one vLLM output-position logprob mapping."""

    token_id: int
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None

    def __post_init__(self) -> None:
        if type(self.token_id) is not int or self.token_id < 0:
            raise ValueError("vLLM logprob token_id must be non-negative")
        value = _finite_real(self.logprob, "vLLM token logprob")
        if value > 1e-6:
            raise ValueError("vLLM token logprob must be non-positive")
        object.__setattr__(self, "logprob", value)
        if self.rank is not None and (
            type(self.rank) is not int or self.rank < 1
        ):
            raise ValueError("vLLM token rank must be positive when present")
        if self.decoded_token is not None and not isinstance(
            self.decoded_token, str
        ):
            raise TypeError("decoded_token must be str or None")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "logprob": self.logprob,
            "rank": self.rank,
            "decoded_token": self.decoded_token,
        }


@dataclass(frozen=True, slots=True)
class VLLMPolicyTurnRequest:
    """Exact normalized request which an audited vLLM client must execute."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_parameters: Mapping[str, object]
    turn_index: int
    behavior_policy: PolicyVersion
    rng: VLLMTurnRNGIdentity
    backend_prompt_payload_sha256: str
    backend_version: str
    logprobs_mode: str
    decoding: VLLMOutputDecodingContract
    termination_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("vLLM request_id must be non-empty")
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        if not self.prompt_token_ids or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.prompt_token_ids
        ):
            raise ValueError("vLLM prompt token IDs must be non-empty and non-negative")
        object.__setattr__(
            self,
            "sampling_parameters",
            _freeze_json_mapping(self.sampling_parameters),
        )
        if self.turn_index < 0:
            raise ValueError("vLLM turn index must be non-negative")
        if not isinstance(self.behavior_policy, PolicyVersion):
            raise TypeError("behavior_policy must be PolicyVersion")
        if not isinstance(self.rng, VLLMTurnRNGIdentity):
            raise TypeError("rng must be VLLMTurnRNGIdentity")
        _validate_sha256(self.backend_prompt_payload_sha256)
        if not self.backend_version or not self.logprobs_mode:
            raise ValueError("vLLM backend identity must be explicit")
        if not isinstance(self.decoding, VLLMOutputDecodingContract):
            raise TypeError("decoding must be VLLMOutputDecodingContract")
        _validate_sha256(self.termination_contract_sha256)

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": VLLM_POLICY_TURN_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids,
            "sampling_parameters": _json_mapping(self.sampling_parameters),
            "turn_index": self.turn_index,
            "behavior_policy": {
                "run_id": self.behavior_policy.run_id,
                "optimizer_step": self.behavior_policy.optimizer_step,
                "weights_sha256": self.behavior_policy.weights_sha256,
            },
            "rng": {
                "seed": self.rng.seed,
                "rng_state_sha256": self.rng.rng_state_sha256,
            },
            "backend_prompt_payload_sha256": self.backend_prompt_payload_sha256,
            "backend_version": self.backend_version,
            "logprobs_mode": self.logprobs_mode,
            "decoding": self.decoding.canonical_payload,
            "termination_contract_sha256": self.termination_contract_sha256,
        }

    @property
    def backend_request_sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload)


@dataclass(frozen=True, slots=True)
class VLLMPolicyTurnResponse:
    """Final single-completion response emitted by the audited client."""

    request_id: str
    backend_request_sha256: str
    prompt_token_ids: tuple[int, ...]
    text: str
    token_ids: tuple[int, ...]
    token_byte_spans: tuple[TokenByteSpan, ...]
    token_logprobs: tuple[tuple[VLLMTokenLogprob, ...], ...]
    finish_reason: str
    stop_reason: int | str | None
    output_index: int = 0
    finished: bool = True

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("vLLM response request_id must be non-empty")
        _validate_sha256(self.backend_request_sha256)
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_byte_spans", tuple(self.token_byte_spans))
        object.__setattr__(
            self,
            "token_logprobs",
            tuple(tuple(position) for position in self.token_logprobs),
        )
        if len(self.token_logprobs) != len(self.token_ids):
            raise ValueError("vLLM token IDs and logprob positions must align")
        if any(not position for position in self.token_logprobs):
            raise ValueError("every vLLM output position requires logprob entries")
        if any(
            not isinstance(entry, VLLMTokenLogprob)
            for position in self.token_logprobs
            for entry in position
        ):
            raise TypeError("token_logprobs must contain VLLMTokenLogprob entries")
        SampledAssistantTurn(self.text, self.token_ids, self.token_byte_spans)
        if not self.finish_reason:
            raise ValueError("vLLM finish_reason must be non-empty")
        if self.stop_reason is not None and not isinstance(
            self.stop_reason, (int, str)
        ):
            raise TypeError("vLLM stop_reason must be int, str, or None")
        if self.output_index != 0:
            raise ValueError("one PolicySamplerPort request accepts only output index 0")
        if self.finished is not True:
            raise ValueError("PolicySamplerPort accepts only a final vLLM response")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": VLLM_POLICY_TURN_RESPONSE_SCHEMA,
            "request_id": self.request_id,
            "backend_request_sha256": self.backend_request_sha256,
            "prompt_token_ids": self.prompt_token_ids,
            "text": self.text,
            "token_ids": self.token_ids,
            "token_byte_spans": tuple(
                {
                    "token_index": span.token_index,
                    "token_id": span.token_id,
                    "byte_start": span.byte_start,
                    "byte_end": span.byte_end,
                }
                for span in self.token_byte_spans
            ),
            "token_logprobs": tuple(
                tuple(entry.canonical_payload for entry in position)
                for position in self.token_logprobs
            ),
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "output_index": self.output_index,
            "finished": self.finished,
        }

    @property
    def backend_response_sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload)


class VLLMPolicyTurnClient(Protocol):
    backend_version: str
    logprobs_mode: str

    def generate(self, request: VLLMPolicyTurnRequest) -> VLLMPolicyTurnResponse: ...


class VLLMPolicySampler:
    """Normalize an audited vLLM envelope into ``PolicySamplerPort`` output.

    A separate live client must prove exact text/byte spans and execute the
    request without changing it; this class does not wire ``LLM.generate``.
    """

    def __init__(
        self,
        *,
        client: VLLMPolicyTurnClient,
        behavior_policy: PolicyVersion,
        rng: VLLMTurnRNGPort,
        request_context: VLLMRequestContextIdentityPort,
        decoding: VLLMOutputDecodingContract,
        termination: VLLMTurnTerminationContract,
    ) -> None:
        if not callable(getattr(client, "generate", None)):
            raise TypeError("client must implement generate()")
        if not isinstance(behavior_policy, PolicyVersion):
            raise TypeError("behavior_policy must be PolicyVersion")
        if not callable(getattr(rng, "for_turn", None)):
            raise TypeError("rng must implement for_turn()")
        if not callable(getattr(request_context, "sha256_for_turn", None)):
            raise TypeError("request_context must implement sha256_for_turn()")
        if not isinstance(decoding, VLLMOutputDecodingContract):
            raise TypeError("decoding must be VLLMOutputDecodingContract")
        if not isinstance(termination, VLLMTurnTerminationContract):
            raise TypeError("termination must be VLLMTurnTerminationContract")
        self.client = client
        self.behavior_policy = behavior_policy
        self.rng = rng
        self.request_context = request_context
        self.decoding = decoding
        self.termination = termination
        self._validate_backend_identity()

    def sample(
        self,
        prompt_token_ids: tuple[int, ...],
        sampling_parameters: Mapping[str, object],
        *,
        turn_index: int,
    ) -> SampledPolicyTurn:
        # Keep the optional vLLM package importable while ``agent_loop`` itself
        # is being initialized through ``tgvf_rl.policy``.
        from tgvf_rl.environment.agent_loop import SampledPolicyTurn

        prompt = tuple(prompt_token_ids)
        if not prompt or any(
            type(token_id) is not int or token_id < 0 for token_id in prompt
        ):
            raise ValueError("sampler prompt token IDs must be non-empty and non-negative")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        self._validate_backend_identity()
        normalized = _normalize_sampling_parameters(sampling_parameters)
        self._validate_request_stops(normalized)

        rng = self.rng.for_turn(
            prompt,
            turn_index=turn_index,
            behavior_policy=self.behavior_policy,
        )
        if not isinstance(rng, VLLMTurnRNGIdentity):
            raise TypeError("rng port must return VLLMTurnRNGIdentity")
        context_sha256 = self.request_context.sha256_for_turn(
            prompt, turn_index=turn_index
        )
        _validate_sha256(context_sha256)

        backend_parameters = _backend_sampling_parameters(
            normalized, rng=rng, decoding=self.decoding
        )
        request_id = _request_id(
            prompt,
            turn_index=turn_index,
            policy=self.behavior_policy,
            rng=rng,
            context_sha256=context_sha256,
            sampling_parameters=backend_parameters,
            termination_sha256=self.termination.sha256,
        )
        request = VLLMPolicyTurnRequest(
            request_id=request_id,
            prompt_token_ids=prompt,
            sampling_parameters=backend_parameters,
            turn_index=turn_index,
            behavior_policy=self.behavior_policy,
            rng=rng,
            backend_prompt_payload_sha256=context_sha256,
            backend_version=self.client.backend_version,
            logprobs_mode=self.client.logprobs_mode,
            decoding=self.decoding,
            termination_contract_sha256=self.termination.sha256,
        )
        response = self.client.generate(request)
        if not isinstance(response, VLLMPolicyTurnResponse):
            raise TypeError("vLLM client must return VLLMPolicyTurnResponse")
        self._validate_response(request, response, normalized)

        selected_logprobs = tuple(
            _selected_logprob(token_id, entries)
            for token_id, entries in zip(
                response.token_ids, response.token_logprobs, strict=True
            )
        )
        sampling = _sampling_identity(
            policy=self.behavior_policy,
            backend_version=self.client.backend_version,
            rng=rng,
            parameters=normalized,
        )
        return SampledPolicyTurn(
            text=response.text,
            token_ids=response.token_ids,
            token_byte_spans=response.token_byte_spans,
            behavior_logprobs=selected_logprobs,
            sampling=sampling,
            think_token_span=_think_token_span(response),
            stop_reason=_stop_reason_text(response),
            backend_request_sha256=request.backend_request_sha256,
            backend_response_sha256=response.backend_response_sha256,
        )

    def _validate_backend_identity(self) -> None:
        if getattr(self.client, "backend_version", None) != SUPPORTED_VLLM_VERSION:
            raise IdentityMismatchError(
                "Policy Pilot sampler requires the exact audited vLLM 0.12.0 backend"
            )
        if getattr(self.client, "logprobs_mode", None) != VLLM_PROCESSED_LOGPROBS_MODE:
            raise IdentityMismatchError(
                "vLLM engine must use logprobs_mode='processed_logprobs'"
            )

    def _validate_request_stops(self, parameters: Mapping[str, object]) -> None:
        stop_strings = set(parameters["stop"])
        stop_token_ids = set(parameters["stop_token_ids"])
        if not set(self.termination.required_request_stop_strings).issubset(
            stop_strings
        ) or not set(self.termination.required_request_stop_token_ids).issubset(
            stop_token_ids
        ):
            raise ContractUnsetError(
                "sampling request does not satisfy the run-bound termination stops"
            )
        if (
            parameters["include_stop_str_in_output"]
            is not self.termination.include_stop_str_in_output
        ):
            raise ContractUnsetError(
                "sampling request differs from termination stop-output ownership"
            )

    def _validate_response(
        self,
        request: VLLMPolicyTurnRequest,
        response: VLLMPolicyTurnResponse,
        parameters: Mapping[str, object],
    ) -> None:
        if response.request_id != request.request_id:
            raise IdentityMismatchError("vLLM response request_id differs from request")
        if response.backend_request_sha256 != request.backend_request_sha256:
            raise IdentityMismatchError("vLLM response echoes a different request hash")
        if response.prompt_token_ids != request.prompt_token_ids:
            raise ReplayMismatchError("vLLM response prompt token IDs changed")
        if len(response.token_ids) > int(parameters["max_tokens"]):
            raise ReplayMismatchError("vLLM response exceeded requested max_tokens")

        close_count = response.text.count(TOOL_CALL_CLOSE)
        if close_count > 1:
            raise ReplayMismatchError(
                "vLLM emitted more than one complete tool-call closing marker"
            )
        if close_count == 1:
            close_end = response.text.index(TOOL_CALL_CLOSE) + len(TOOL_CALL_CLOSE)
            suffix = response.text[close_end:]
            if suffix not in self.termination.tool_call_terminal_suffixes:
                raise ReplayMismatchError(
                    "vLLM emitted a tool-call suffix outside the run-bound contract"
                )
            outcome = VLLMTerminationOutcome(
                response.finish_reason, response.stop_reason
            )
            if outcome not in self.termination.tool_call_outcomes:
                raise ReplayMismatchError(
                    "tool-call termination differs from the run-bound contract"
                )
        else:
            outcome = VLLMTerminationOutcome(
                response.finish_reason, response.stop_reason
            )
            if outcome not in self.termination.final_turn_outcomes:
                raise ReplayMismatchError(
                    "final/non-complete-call termination differs from the run-bound contract"
                )


def _normalize_sampling_parameters(
    parameters: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(parameters, Mapping):
        raise TypeError("sampling_parameters must be a mapping")
    keys = set(parameters)
    if keys != _SAMPLING_PARAMETER_KEYS:
        raise ContractUnsetError(
            "vLLM sampling parameters must bind exactly the Policy Pilot fields; "
            f"missing={sorted(_SAMPLING_PARAMETER_KEYS - keys)!r}, "
            f"extra={sorted(keys - _SAMPLING_PARAMETER_KEYS)!r}"
        )

    temperature = _finite_real(parameters["temperature"], "temperature")
    top_p = _finite_real(parameters["top_p"], "top_p")
    min_p = _finite_real(parameters["min_p"], "min_p")
    repetition_penalty = _finite_real(
        parameters["repetition_penalty"], "repetition_penalty"
    )
    presence_penalty = _finite_real(
        parameters["presence_penalty"], "presence_penalty"
    )
    frequency_penalty = _finite_real(
        parameters["frequency_penalty"], "frequency_penalty"
    )
    top_k = _integer(parameters["top_k"], "top_k", minimum=-1)
    max_tokens = _integer(parameters["max_tokens"], "max_tokens", minimum=1)
    if temperature < 1e-2:
        raise ValueError("audited processed-logprob sampling requires temperature >= 0.01")
    if not 0.0 < top_p <= 1.0 or not 0.0 <= min_p <= 1.0:
        raise ValueError("invalid top_p/min_p")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if not -2.0 <= presence_penalty <= 2.0 or not -2.0 <= frequency_penalty <= 2.0:
        raise ValueError("presence/frequency penalties must lie in [-2, 2]")

    raw_stop_ids = parameters["stop_token_ids"]
    raw_stops = parameters["stop"]
    if not isinstance(raw_stop_ids, (list, tuple)) or any(
        type(token_id) is not int or token_id < 0 for token_id in raw_stop_ids
    ):
        raise TypeError("stop_token_ids must be a sequence of non-negative integers")
    if len(set(raw_stop_ids)) != len(raw_stop_ids):
        raise ValueError("stop_token_ids must be unique")
    if not isinstance(raw_stops, (list, tuple)) or any(
        not isinstance(stop, str) or not stop for stop in raw_stops
    ):
        raise TypeError("stop must be a sequence of non-empty strings")
    if len(set(raw_stops)) != len(raw_stops):
        raise ValueError("stop strings must be unique")
    for name in ("include_stop_str_in_output", "ignore_eos"):
        if not isinstance(parameters[name], bool):
            raise TypeError(f"{name} must be bool")
    if parameters["logprobs"] is not True:
        raise ContractUnsetError(
            "actual sampled-token vLLM logprobs must be requested explicitly"
        )

    return MappingProxyType(
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop_token_ids": tuple(raw_stop_ids),
            "stop": tuple(raw_stops),
            "include_stop_str_in_output": parameters[
                "include_stop_str_in_output"
            ],
            "ignore_eos": parameters["ignore_eos"],
            "max_tokens": max_tokens,
            "logprobs": True,
        }
    )


def _backend_sampling_parameters(
    parameters: Mapping[str, object],
    *,
    rng: VLLMTurnRNGIdentity,
    decoding: VLLMOutputDecodingContract,
) -> Mapping[str, object]:
    values = dict(parameters)
    values.update(
        {
            "stop_token_ids": list(parameters["stop_token_ids"]),
            "stop": list(parameters["stop"]),
            "seed": rng.seed,
            # vLLM 0.12 uses zero to request only the sampled-token entry.
            "logprobs": 0,
            "n": 1,
            "min_tokens": 0,
            "prompt_logprobs": None,
            "flat_logprobs": False,
            "detokenize": decoding.detokenize,
            "skip_special_tokens": decoding.skip_special_tokens,
            "spaces_between_special_tokens": decoding.spaces_between_special_tokens,
            "logits_processors": None,
            "truncate_prompt_tokens": None,
            "output_kind": decoding.output_kind,
            "bad_words": [],
            "structured_outputs": None,
            "logit_bias": None,
            "allowed_token_ids": None,
            "extra_args": None,
        }
    )
    return MappingProxyType(values)


def _request_id(
    prompt_token_ids: tuple[int, ...],
    *,
    turn_index: int,
    policy: PolicyVersion,
    rng: VLLMTurnRNGIdentity,
    context_sha256: str,
    sampling_parameters: Mapping[str, object],
    termination_sha256: str,
) -> str:
    digest = _canonical_json_sha256(
        {
            "schema": "tgvf-vllm-request-id-v1",
            "prompt_token_ids": prompt_token_ids,
            "turn_index": turn_index,
            "policy": (
                policy.run_id,
                policy.optimizer_step,
                policy.weights_sha256,
            ),
            "seed": rng.seed,
            "rng_state_sha256": rng.rng_state_sha256,
            "context_sha256": context_sha256,
            "sampling_parameters": _json_mapping(sampling_parameters),
            "termination_sha256": termination_sha256,
        }
    )
    return f"tgvf-policy-{digest}"


def _sampling_identity(
    *,
    policy: PolicyVersion,
    backend_version: str,
    rng: VLLMTurnRNGIdentity,
    parameters: Mapping[str, object],
) -> SamplingIdentity:
    return SamplingIdentity(
        policy_version=policy,
        backend="vllm",
        backend_version=backend_version,
        seed=rng.seed,
        rng_state_sha256=rng.rng_state_sha256,
        temperature=float(parameters["temperature"]),
        top_p=float(parameters["top_p"]),
        top_k=int(parameters["top_k"]),
        min_p=float(parameters["min_p"]),
        repetition_penalty=float(parameters["repetition_penalty"]),
        presence_penalty=float(parameters["presence_penalty"]),
        frequency_penalty=float(parameters["frequency_penalty"]),
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
        max_tokens=int(parameters["max_tokens"]),
        do_sample=True,
        stop_token_ids=tuple(parameters["stop_token_ids"]),
        stop_strings=tuple(parameters["stop"]),
        include_stop_str_in_output=bool(parameters["include_stop_str_in_output"]),
        ignore_eos=bool(parameters["ignore_eos"]),
    )


def _selected_logprob(
    token_id: int, entries: tuple[VLLMTokenLogprob, ...]
) -> float:
    selected = tuple(entry for entry in entries if entry.token_id == token_id)
    if len(selected) != 1:
        raise ReplayMismatchError(
            "sampled token must occur exactly once in its vLLM logprob mapping"
        )
    return selected[0].logprob


def _think_token_span(response: VLLMPolicyTurnResponse) -> TokenSpan:
    marker = "</think>"
    if response.text.count(marker) != 1:
        raise ReplayMismatchError(
            "sampled assistant turn must contain exactly one </think> marker"
        )
    close_char_end = response.text.index(marker) + len(marker)
    close_byte_end = len(response.text[:close_char_end].encode("utf-8"))
    covering = tuple(
        span
        for span in response.token_byte_spans
        if span.byte_end > 0 and span.byte_start < close_byte_end
    )
    if not covering or covering[-1].byte_end < close_byte_end:
        raise ReplayMismatchError("sampled token spans do not cover </think>")
    return TokenSpan(0, covering[-1].token_index + 1)


def _stop_reason_text(response: VLLMPolicyTurnResponse) -> str:
    return json.dumps(
        {
            "backend": "vllm",
            "finish_reason": response.finish_reason,
            "stop_reason": response.stop_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "VLLM_POLICY_TURN_REQUEST_SCHEMA",
    "VLLM_POLICY_TURN_RESPONSE_SCHEMA",
    "VLLM_PROCESSED_LOGPROBS_MODE",
    "VLLMOutputDecodingContract",
    "VLLMPolicySampler",
    "VLLMPolicyTurnClient",
    "VLLMPolicyTurnRequest",
    "VLLMPolicyTurnResponse",
    "VLLMRequestContextIdentityPort",
    "VLLMTokenLogprob",
    "VLLMTerminationOutcome",
    "VLLMTurnTerminationContract",
    "VLLMTurnRNGIdentity",
    "VLLMTurnRNGPort",
]
