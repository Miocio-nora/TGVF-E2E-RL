"""Lossless bridge from veRL's async server manager to the native sync loop.

The accepted upstream veRL ``LLMServerClient.generate`` API returns only
``TokenOutput(token_ids, log_probs, stop_reason, extra_fields)``.  It does not
echo prompt IDs, tag the log-probability convention, or retain vLLM's distinct
``finish_reason``/``stop_reason`` pair.  This bridge therefore:

* passes exact prompt IDs and recorded precomputed image items itself;
* requires an explicit accepted ``processed_logprobs`` runtime identity;
* preserves the returned selected-token log probabilities without replay; and
* restores termination only when emitted terminal tokens or the exact requested
  length make the result unambiguous, otherwise failing closed.

The synchronous framework-neutral loop runs in a worker thread.  Its sampler
submits async server-manager calls back to the owning event loop with
``asyncio.run_coroutine_threadsafe``.  No upstream or site-package patch is
required.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import partial
import hashlib
import inspect
import json
import math
import threading
from typing import TYPE_CHECKING, Any, Protocol

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)
from tgvf_rl.framework.vllm import (
    FastTokenizerTokenByteSpanDecoder,
    VLLM_PROCESSED_LOGPROBS_MODE,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMTokenLogprob,
    VLLMTurnRNGPort,
    VLLMTurnTerminationContract,
)
from tgvf_rl.framework.vllm.live_client import VLLMLivePromptInputsPort
from tgvf_rl.framework.vllm.preexpanded_prompt import (
    require_preexpanded_prompt_contract,
)
from tgvf_rl.framework.vllm.registration import SUPPORTED_VLLM_VERSION
from tgvf_rl.framework.vllm.sampler import VLLMRequestContextIdentityPort
from .compatibility import SUPPORTED_LOGPROBS_MODE

if TYPE_CHECKING:
    from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
    from tgvf_rl.environment.agent_loop import (
        FrameworkNeutralAgentLoop,
        PolicySamplerPort,
        RolloutRequest,
    )
    from tgvf_rl.trajectories.schema import TrajectoryRecord
    from tgvf_rl.observations.schema import TrajectorySourceVisual
    from tgvf_rl.policy.config import PilotSamplingConfig


VERL_NATIVE_AGENT_LOOP_BRIDGE_SCHEMA = "tgvf-verl-native-agent-loop-bridge-v1"


class VerlAsyncServerPolicyTurnClient:
    """Synchronous turn client backed by one upstream async server manager."""

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
        awaitable = self.server_manager.generate(
            request_id=self.sticky_request_id,
            prompt_ids=list(request.prompt_token_ids),
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=inputs.mm_processor_kwargs,
        )
        if not inspect.isawaitable(awaitable):
            raise TypeError("veRL server_manager.generate() must return an awaitable")
        future = asyncio.run_coroutine_threadsafe(awaitable, self.event_loop)
        try:
            output = future.result(timeout=self.server_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError("veRL server_manager generation timed out") from error

        token_ids = _token_ids(getattr(output, "token_ids", None))
        logprobs = _selected_processed_logprobs(
            getattr(output, "log_probs", None), token_ids=token_ids
        )
        _validate_policy_step_evidence(
            getattr(output, "extra_fields", None), request=request
        )
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
) -> tuple[str, int | str | None]:
    if upstream_stop_reason != "completed":
        raise ReplayMismatchError(
            "veRL generation did not return one completed final TokenOutput"
        )
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


@dataclass(frozen=True, slots=True)
class VerlNativeAgentLoopInvocation:
    """Caller-owned native-loop dependencies for one upstream sample."""

    request: RolloutRequest
    native_loop_factory: Callable[[PolicySamplerPort], FrameworkNeutralAgentLoop]
    prompt_context: VLLMLivePromptInputsPort | VLLMRequestContextIdentityPort
    rng: VLLMTurnRNGPort
    decoding: VLLMOutputDecodingContract
    termination: VLLMTurnTerminationContract
    sticky_request_id: str
    max_model_len: int
    output_builder: Callable[[TrajectoryRecord], object]

    def __post_init__(self) -> None:
        # Keep the package-level veRL export safe while ``environment.agent_loop``
        # itself is importing policy configuration through ``policy.__init__``.
        from tgvf_rl.environment.agent_loop import RolloutRequest

        if not isinstance(self.request, RolloutRequest):
            raise TypeError("invocation request must be RolloutRequest")
        for value, method, name in (
            (self.native_loop_factory, "__call__", "native_loop_factory"),
            (self.prompt_context, "for_request", "prompt_context"),
            (self.prompt_context, "sha256_for_turn", "prompt_context"),
            (self.rng, "for_turn", "rng"),
            (self.output_builder, "__call__", "output_builder"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{name} must implement {method}()")
        if not isinstance(self.decoding, VLLMOutputDecodingContract):
            raise TypeError("invocation decoding contract is invalid")
        if not isinstance(self.termination, VLLMTurnTerminationContract):
            raise TypeError("invocation termination contract is invalid")
        if not self.sticky_request_id:
            raise ValueError("invocation sticky_request_id must be non-empty")
        if type(self.max_model_len) is not int or self.max_model_len <= 1:
            raise ValueError("invocation max_model_len must be greater than one")


class VerlNativeAgentLoopInvocationFactory(Protocol):
    def build(
        self,
        *,
        sampling_params: Mapping[str, object],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeAgentLoopInvocation: ...


class CurrentPolicyVersionPort(Protocol):
    def current_policy_version(self) -> "PolicyVersion": ...


@dataclass(frozen=True, slots=True)
class VerlNativeTrajectoryComponents:
    """Live trajectory dependencies materialized for one exact identity."""

    source_visual: "TrajectorySourceVisual"
    native_loop_factory: Callable[["PolicySamplerPort"], "FrameworkNeutralAgentLoop"]
    prompt_context: VLLMLivePromptInputsPort | VLLMRequestContextIdentityPort
    output_builder: Callable[["TrajectoryRecord"], object]

    def __post_init__(self) -> None:
        from tgvf_rl.observations.schema import TrajectorySourceVisual

        if not isinstance(self.source_visual, TrajectorySourceVisual):
            raise TypeError("trajectory components require TrajectorySourceVisual")
        for value, method, name in (
            (self.native_loop_factory, "__call__", "native_loop_factory"),
            (self.prompt_context, "for_request", "prompt_context"),
            (self.prompt_context, "sha256_for_turn", "prompt_context"),
            (self.output_builder, "__call__", "output_builder"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{name} must implement {method}()")


class VerlNativeTrajectoryComponentsPort(Protocol):
    def build_trajectory_components(
        self,
        *,
        identity: object,
        model: "ModelIdentity",
        behavior_policy: "PolicyVersion",
        initial_prompt_token_ids: tuple[int, ...],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeTrajectoryComponents: ...


class BoundVerlNativeAgentLoopInvocationFactory:
    """Concrete thread-safe mapping of upstream repeated rows to native runs."""

    def __init__(
        self,
        *,
        run_id: str,
        model: "ModelIdentity",
        sampling_contract: "PilotSamplingConfig",
        policy_version: CurrentPolicyVersionPort,
        trajectory_components: VerlNativeTrajectoryComponentsPort,
        decoding: VLLMOutputDecodingContract,
        termination: VLLMTurnTerminationContract,
        rollout_master_seed: int,
        max_model_len: int,
        rollouts_per_prompt: int = 8,
    ) -> None:
        from tgvf_rl.contracts.identity import ModelIdentity
        from tgvf_rl.policy.config import PilotSamplingConfig

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("native invocation run_id must be non-empty")
        if not isinstance(model, ModelIdentity):
            raise TypeError("native invocation model must be ModelIdentity")
        if not isinstance(sampling_contract, PilotSamplingConfig):
            raise TypeError("sampling_contract must be PilotSamplingConfig")
        if not sampling_contract.is_run_bound:
            raise ValueError("native invocation sampling contract is not run-bound")
        if not callable(getattr(policy_version, "current_policy_version", None)):
            raise TypeError("policy_version must implement current_policy_version()")
        if not callable(
            getattr(trajectory_components, "build_trajectory_components", None)
        ):
            raise TypeError(
                "trajectory_components must implement build_trajectory_components()"
            )
        if not isinstance(decoding, VLLMOutputDecodingContract):
            raise TypeError("native invocation decoding contract is invalid")
        if not isinstance(termination, VLLMTurnTerminationContract):
            raise TypeError("native invocation termination contract is invalid")
        if type(rollout_master_seed) is not int or rollout_master_seed < 0:
            raise ValueError("rollout_master_seed must be non-negative")
        if type(max_model_len) is not int or max_model_len <= 1:
            raise ValueError("max_model_len must be greater than one")
        if type(rollouts_per_prompt) is not int or rollouts_per_prompt != 8:
            raise ValueError("Policy Pilot native invocation requires exactly n=8")
        if sampling_contract.trajectories_per_prompt != rollouts_per_prompt:
            raise ValueError("sampling contract and invocation group size differ")
        if sampling_contract.include_stop_str_in_output is not True:
            raise ValueError("native tool-call closer must remain policy-sampled")

        self.run_id = run_id
        self.model = model
        self.sampling_contract = sampling_contract
        self.policy_version = policy_version
        self.trajectory_components = trajectory_components
        self.decoding = decoding
        self.termination = termination
        self.rollout_master_seed = rollout_master_seed
        self.max_model_len = max_model_len
        self.rollouts_per_prompt = rollouts_per_prompt
        self._lock = threading.Lock()
        self._next_rollout_index: dict[str, int] = {}
        self._completed_group_uids: set[str] = set()

    def build(
        self,
        *,
        sampling_params: Mapping[str, object],
        sample_fields: Mapping[str, object],
    ) -> VerlNativeAgentLoopInvocation:
        from tgvf_rl.contracts.identity import PolicyVersion
        from tgvf_rl.environment.agent_loop import RolloutRequest
        from tgvf_rl.framework.vllm import ContentAddressedVLLMTurnRNG
        from tgvf_rl.trajectories.schema import TrajectoryIdentity

        if not isinstance(sampling_params, Mapping):
            raise TypeError("upstream sampling_params must be a mapping")
        if not isinstance(sample_fields, Mapping):
            raise TypeError("upstream sample_fields must be a mapping")
        self._validate_upstream_sampling(sampling_params)
        sample_id = _required_sample_text(sample_fields, "sample_id")
        group_nonce = _required_sample_text(sample_fields, "uid")
        data_cursor = _required_sample_integer(sample_fields, "index")
        prompt_ids = _required_prompt_ids(sample_fields.get("initial_prompt_token_ids"))
        behavior_policy = self.policy_version.current_policy_version()
        if not isinstance(behavior_policy, PolicyVersion):
            raise TypeError("policy version port must return PolicyVersion")
        if behavior_policy.run_id != self.run_id:
            raise IdentityMismatchError("served behavior policy belongs to another run")
        group_uid = _native_group_uid(
            run_id=self.run_id,
            sample_id=sample_id,
            group_nonce=group_nonce,
            data_cursor=data_cursor,
            behavior_policy=behavior_policy,
        )
        identity = TrajectoryIdentity(
            self.run_id,
            sample_id,
            self._claim_rollout_index(group_uid),
            group_uid,
        )
        components = self.trajectory_components.build_trajectory_components(
            identity=identity,
            model=self.model,
            behavior_policy=behavior_policy,
            initial_prompt_token_ids=prompt_ids,
            sample_fields=dict(sample_fields),
        )
        if not isinstance(components, VerlNativeTrajectoryComponents):
            raise TypeError(
                "trajectory components port must return VerlNativeTrajectoryComponents"
            )
        request = RolloutRequest(
            "trajectory-v1",
            identity,
            self.model,
            behavior_policy,
            components.source_visual,
            prompt_ids,
            {},
            self.sampling_contract,
        )
        return VerlNativeAgentLoopInvocation(
            request=request,
            native_loop_factory=components.native_loop_factory,
            prompt_context=components.prompt_context,
            rng=ContentAddressedVLLMTurnRNG(
                master_seed=self.rollout_master_seed,
                stream_identity=identity.canonical_id,
            ),
            decoding=self.decoding,
            termination=self.termination,
            sticky_request_id=identity.canonical_id,
            max_model_len=self.max_model_len,
            output_builder=components.output_builder,
        )

    def _claim_rollout_index(self, group_uid: str) -> int:
        with self._lock:
            if group_uid in self._completed_group_uids:
                raise ReplayMismatchError("upstream reused a completed n=8 group uid")
            index = self._next_rollout_index.get(group_uid, 0)
            if index >= self.rollouts_per_prompt:
                raise ReplayMismatchError(
                    "upstream generated more than n=8 trajectories"
                )
            if index + 1 == self.rollouts_per_prompt:
                self._next_rollout_index.pop(group_uid, None)
                self._completed_group_uids.add(group_uid)
            else:
                self._next_rollout_index[group_uid] = index + 1
            return index

    def _validate_upstream_sampling(
        self, sampling_params: Mapping[str, object]
    ) -> None:
        expected = {
            "temperature": self.sampling_contract.temperature,
            "top_p": self.sampling_contract.top_p,
            "top_k": self.sampling_contract.top_k,
            "repetition_penalty": self.sampling_contract.repetition_penalty,
            "logprobs": True,
        }
        mismatches = {
            key: (sampling_params.get(key), value)
            for key, value in expected.items()
            if sampling_params.get(key) != value
        }
        if mismatches:
            raise IdentityMismatchError(
                f"upstream sampling parameters differ from run contract: {mismatches!r}"
            )


def _native_group_uid(
    *,
    run_id: str,
    sample_id: str,
    group_nonce: str,
    data_cursor: int,
    behavior_policy: "PolicyVersion",
) -> str:
    payload = json.dumps(
        {
            "schema": "tgvf-verl-native-group-v1",
            "run_id": run_id,
            "sample_id": sample_id,
            "group_nonce": group_nonce,
            "data_cursor": data_cursor,
            "behavior_policy": {
                "optimizer_step": behavior_policy.optimizer_step,
                "weights_sha256": behavior_policy.weights_sha256,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "tgvf-verl-group:" + hashlib.sha256(payload).hexdigest()


def _required_sample_text(fields: Mapping[str, object], key: str) -> str:
    value = fields.get(key)
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if not isinstance(value, str) or not value:
        raise ValueError(f"upstream sample field {key!r} must be non-empty text")
    return value


def _required_sample_integer(fields: Mapping[str, object], key: str) -> int:
    value = fields.get(key)
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if type(value) is not int or value < 0:
        raise ValueError(f"upstream sample field {key!r} must be non-negative int")
    return value


def _required_prompt_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("initial_prompt_token_ids must be a sequence")
    result = tuple(value)
    if not result or any(
        type(token_id) is not int or token_id < 0 for token_id in result
    ):
        raise ValueError("initial_prompt_token_ids must contain non-negative integers")
    return result


class VerlFrameworkNeutralAgentLoop:
    """Hydra-instantiable structural implementation of upstream AgentLoopBase.

    Upstream's worker only instantiates the configured target and awaits
    ``run``; it performs no ``isinstance(AgentLoopBase)`` check.  Avoiding an
    import-time veRL subclass keeps the repo's optional bridge importable while
    retaining the exact constructor/run surface used by the pinned worker.
    """

    def __init__(
        self,
        trainer_config: object,
        server_manager: object,
        tokenizer: object,
        processor: object,
        dataset_cls: object,
        data_config: object,
        *,
        invocation_factory: VerlNativeAgentLoopInvocationFactory | partial[object],
        logprobs_mode: str,
        server_timeout_seconds: float = 600.0,
        **kwargs: object,
    ) -> None:
        del kwargs
        # veRL deliberately wraps DictConfig before Hydra instantiation so
        # Hydra does not recursively resolve it.  Match AgentLoopBase's
        # constructor contract and hand the underlying configs to our runtime.
        if type(trainer_config).__name__ == "DictConfigWrap":
            trainer_config = trainer_config.config
        if type(data_config).__name__ == "DictConfigWrap":
            data_config = data_config.config
        if isinstance(invocation_factory, partial):
            invocation_factory = invocation_factory(
                trainer_config=trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=dataset_cls,
                data_config=data_config,
            )
        if not callable(getattr(invocation_factory, "build", None)):
            raise TypeError(
                "invocation_factory must implement build() or be the Hydra "
                "partial for PolicyE2ERuntimeInvocationFactory"
            )
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.invocation_factory = invocation_factory
        self.logprobs_mode = logprobs_mode
        self.server_timeout_seconds = server_timeout_seconds

    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs: object,
    ) -> object:
        event_loop = asyncio.get_running_loop()
        invocation = self.invocation_factory.build(
            sampling_params=dict(sampling_params),
            sample_fields=dict(kwargs),
        )
        if not isinstance(invocation, VerlNativeAgentLoopInvocation):
            raise TypeError(
                "invocation_factory must return VerlNativeAgentLoopInvocation"
            )
        decoder = FastTokenizerTokenByteSpanDecoder(self.tokenizer)
        client = VerlAsyncServerPolicyTurnClient(
            server_manager=self.server_manager,
            event_loop=event_loop,
            tokenizer=self.tokenizer,
            prompt_inputs=invocation.prompt_context,
            token_byte_span_decoder=decoder,
            sticky_request_id=invocation.sticky_request_id,
            max_model_len=invocation.max_model_len,
            server_timeout_seconds=self.server_timeout_seconds,
            logprobs_mode=self.logprobs_mode,
        )
        sampler = VLLMPolicySampler(
            client=client,
            behavior_policy=invocation.request.behavior_policy,
            rng=invocation.rng,
            request_context=invocation.prompt_context,
            decoding=invocation.decoding,
            termination=invocation.termination,
        )

        def execute_sync() -> tuple[TrajectoryRecord, object]:
            from tgvf_rl.environment.agent_loop import FrameworkNeutralAgentLoop

            native_loop = invocation.native_loop_factory(sampler)
            if not isinstance(native_loop, FrameworkNeutralAgentLoop):
                raise TypeError(
                    "native_loop_factory must return FrameworkNeutralAgentLoop"
                )
            trajectory = native_loop.run(invocation.request)
            return trajectory, invocation.output_builder(trajectory)

        trajectory, output = await asyncio.to_thread(execute_sync)
        _validate_structural_agent_loop_output(
            output,
            expected_prompt_ids=invocation.request.initial_prompt_token_ids,
            expected_num_turns=len(trajectory.assistant_turns),
        )
        return output


def _validate_structural_agent_loop_output(
    output: object,
    *,
    expected_prompt_ids: tuple[int, ...],
    expected_num_turns: int,
) -> None:
    prompt_ids = tuple(getattr(output, "prompt_ids", ()))
    response_ids = tuple(getattr(output, "response_ids", ()))
    response_mask = tuple(getattr(output, "response_mask", ()))
    response_logprobs = getattr(output, "response_logprobs", None)
    if prompt_ids != expected_prompt_ids:
        raise ReplayMismatchError("upstream AgentLoopOutput changed exact prompt IDs")
    if not response_ids or len(response_ids) != len(response_mask):
        raise ReplayMismatchError("upstream response IDs/mask do not align")
    if response_logprobs is None or len(response_logprobs) != len(response_ids):
        raise ReplayMismatchError(
            "upstream AgentLoopOutput omitted aligned behavior logprobs"
        )
    if any(mask not in (0, 1) for mask in response_mask):
        raise ReplayMismatchError("upstream response_mask must contain only 0/1")
    for mask, raw_logprob in zip(response_mask, response_logprobs, strict=True):
        if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
            raise TypeError("upstream response logprobs must be numeric")
        logprob = float(raw_logprob)
        if not math.isfinite(logprob) or (mask == 1 and logprob > 1e-6):
            raise ReplayMismatchError("upstream behavior logprob is invalid")
        if mask == 0 and logprob != 0.0:
            raise ReplayMismatchError(
                "environment-owned response tokens must carry zero placeholder logprob"
            )
    if getattr(output, "num_turns", None) != expected_num_turns:
        raise ReplayMismatchError("upstream AgentLoopOutput num_turns changed")


__all__ = [
    "BoundVerlNativeAgentLoopInvocationFactory",
    "CurrentPolicyVersionPort",
    "VERL_NATIVE_AGENT_LOOP_BRIDGE_SCHEMA",
    "VerlAsyncServerPolicyTurnClient",
    "VerlFrameworkNeutralAgentLoop",
    "VerlNativeAgentLoopInvocation",
    "VerlNativeAgentLoopInvocationFactory",
    "VerlNativeTrajectoryComponents",
    "VerlNativeTrajectoryComponentsPort",
]
