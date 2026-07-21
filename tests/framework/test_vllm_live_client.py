from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import subprocess
import sys

import pytest

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.vllm import (
    VLLM_PROCESSED_LOGPROBS_MODE,
    VLLMLivePolicyTurnClient,
    VLLMLivePromptInputs,
    VLLMOutputDecodingContract,
    VLLMPolicyTurnRequest,
    VLLMTurnRNGIdentity,
    bind_preexpanded_prompt_contract,
)
from tgvf_rl.protocol import TokenByteSpan


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def _request(
    *,
    backend_prompt_payload_sha256: str = SHA1,
    logprobs_mode: str = VLLM_PROCESSED_LOGPROBS_MODE,
) -> VLLMPolicyTurnRequest:
    return VLLMPolicyTurnRequest(
        request_id="live-request-0",
        prompt_token_ids=(10, 20),
        sampling_parameters={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_token_ids": [151645],
            "stop": ["</tool_call>"],
            "include_stop_str_in_output": True,
            "ignore_eos": False,
            "max_tokens": 64,
            "logprobs": 0,
            "seed": 42,
            "n": 1,
            "min_tokens": 0,
            "prompt_logprobs": None,
            "flat_logprobs": False,
            "detokenize": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "logits_processors": None,
            "truncate_prompt_tokens": None,
            "output_kind": "final_only",
            "bad_words": [],
            "structured_outputs": None,
            "logit_bias": None,
            "allowed_token_ids": None,
            "extra_args": None,
        },
        turn_index=0,
        behavior_policy=PolicyVersion("pilot", 3, SHA0),
        rng=VLLMTurnRNGIdentity(seed=42, rng_state_sha256=SHA2),
        backend_prompt_payload_sha256=backend_prompt_payload_sha256,
        backend_version="0.12.0",
        logprobs_mode=logprobs_mode,
        decoding=VLLMOutputDecodingContract(
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind="final_only",
        ),
        termination_contract_sha256=SHA2,
    )


@dataclass
class _RawLogprob:
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


def _raw_request_output(
    prompt_token_ids: list[int],
    *,
    logprobs: object = ...,
    completion_token_ids: tuple[int, ...] = (101, 102),
    completion_index: int = 0,
    finished: bool = True,
) -> object:
    if logprobs is ...:
        logprobs = [
            {
                999: _RawLogprob(-3.0, 4, "x"),
                101: _RawLogprob(-0.25, 1, "a"),
            },
            {102: _RawLogprob(-0.5, 1, "b")},
        ]
    completion = SimpleNamespace(
        index=completion_index,
        text="ab",
        token_ids=completion_token_ids,
        logprobs=logprobs,
        finish_reason="stop",
        stop_reason=151645,
    )
    return SimpleNamespace(
        request_id="backend-owned-id",
        prompt_token_ids=prompt_token_ids,
        outputs=[completion],
        finished=finished,
    )


class _Engine:
    def __init__(self, *, logprobs_mode: str = VLLM_PROCESSED_LOGPROBS_MODE):
        self.model_config = SimpleNamespace(logprobs_mode=logprobs_mode)
        self.calls: list[tuple[object, object, object]] = []
        self.output_factory = _raw_request_output

    def generate(self, prompts, *, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [self.output_factory(list(prompts[0]["prompt_token_ids"]))]


class _Inputs:
    def __init__(self, inputs: VLLMLivePromptInputs):
        self.inputs = inputs
        self.requests: list[VLLMPolicyTurnRequest] = []

    def for_request(self, request):
        self.requests.append(request)
        return self.inputs


class _Spans:
    def __init__(self):
        self.calls: list[tuple[str, tuple[int, ...], object]] = []

    def spans_for_output(self, *, text, token_ids, decoding):
        self.calls.append((text, token_ids, decoding))
        return (
            TokenByteSpan(0, token_ids[0], 0, 1),
            TokenByteSpan(1, token_ids[1], 1, 2),
        )


class _SamplingParams:
    def __init__(self, values):
        self.values = dict(values)


class _SamplingParamsFactory:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.outputs: list[_SamplingParams] = []

    def __call__(self, values):
        self.calls.append(dict(values))
        output = _SamplingParams(values)
        self.outputs.append(output)
        return output


def _client(
    engine: _Engine,
    *,
    payload_sha256: str = SHA1,
) -> tuple[
    VLLMLivePolicyTurnClient,
    VLLMLivePromptInputs,
    _Inputs,
    _Spans,
    _SamplingParamsFactory,
]:
    latent = object()
    mm_data = {"image": [latent]}
    mm_kwargs = bind_preexpanded_prompt_contract(
        {"do_resize": False},
        prompt_token_ids=(10, 20),
        image_token_id=20,
        expected_image_items=1,
    )
    mm_uuids = {"image": ["source-and-recorded-d"]}
    inputs_value = VLLMLivePromptInputs(
        backend_prompt_payload_sha256=payload_sha256,
        multi_modal_data=mm_data,
        mm_processor_kwargs=mm_kwargs,
        multi_modal_uuids=mm_uuids,
    )
    inputs = _Inputs(inputs_value)
    spans = _Spans()
    params = _SamplingParamsFactory()
    client = VLLMLivePolicyTurnClient(
        engine=engine,
        prompt_inputs=inputs,
        token_byte_span_decoder=spans,
        sampling_params_factory=params,
    )
    return client, inputs_value, inputs, spans, params


def test_live_client_passes_exact_prompt_latents_and_sampling_params() -> None:
    engine = _Engine()
    client, inputs_value, inputs, spans, params = _client(engine)
    request = _request()

    response = client.generate(request)

    assert inputs.requests == [request]
    assert len(engine.calls) == 1
    prompts, sampling_params, use_tqdm = engine.calls[0]
    assert use_tqdm is False
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [10, 20]
    assert prompts[0]["multi_modal_data"] is inputs_value.multi_modal_data
    assert prompts[0]["mm_processor_kwargs"] is inputs_value.mm_processor_kwargs
    assert prompts[0]["multi_modal_uuids"] is inputs_value.multi_modal_uuids
    assert sampling_params is params.outputs[0]
    assert params.calls == [dict(request.sampling_parameters)]
    assert sampling_params.values["seed"] == 42
    assert sampling_params.values["logprobs"] == 0
    assert sampling_params.values["output_kind"] == "final_only"

    assert spans.calls == [("ab", (101, 102), request.decoding)]
    assert response.request_id == request.request_id
    assert response.backend_request_sha256 == request.backend_request_sha256
    assert response.prompt_token_ids == request.prompt_token_ids
    assert response.token_ids == (101, 102)
    assert tuple(
        next(entry.logprob for entry in position if entry.token_id == token_id)
        for token_id, position in zip(
            response.token_ids, response.token_logprobs, strict=True
        )
    ) == (-0.25, -0.5)
    assert tuple(entry.token_id for entry in response.token_logprobs[0]) == (
        101,
        999,
    )


def test_live_client_import_does_not_import_optional_vllm_package() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tgvf_rl.framework.vllm.live_client; "
                "assert 'vllm' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_live_client_rejects_unproven_or_changed_logprob_convention() -> None:
    with pytest.raises(IdentityMismatchError, match="processed_logprobs"):
        _client(_Engine(logprobs_mode="raw_logprobs"))

    engine = _Engine()
    client, *_ = _client(engine)
    engine.model_config.logprobs_mode = "raw_logprobs"
    with pytest.raises(IdentityMismatchError, match="processed_logprobs"):
        client.generate(_request())


def test_live_client_rejects_request_with_non_processed_convention() -> None:
    client, *_ = _client(_Engine())
    with pytest.raises(IdentityMismatchError, match="processed_logprobs"):
        client.generate(_request(logprobs_mode="raw_logprobs"))


def test_live_client_rejects_materialized_payload_identity_mismatch() -> None:
    client, *_ = _client(_Engine(), payload_sha256=SHA2)
    with pytest.raises(IdentityMismatchError, match="prompt payload"):
        client.generate(_request(backend_prompt_payload_sha256=SHA1))


@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        (None, "omitted"),
        ([{101: _RawLogprob(-0.25)}], "counts differ"),
        (
            [
                {777: _RawLogprob(-0.25)},
                {102: _RawLogprob(-0.5)},
            ],
            "sampled token ID is absent",
        ),
    ],
)
def test_live_client_rejects_missing_or_misaligned_sampled_logprobs(
    logprobs: object,
    message: str,
) -> None:
    engine = _Engine()
    engine.output_factory = lambda prompt: _raw_request_output(
        prompt, logprobs=logprobs
    )
    client, *_ = _client(engine)

    with pytest.raises(ReplayMismatchError, match=message):
        client.generate(_request())


@pytest.mark.parametrize(
    ("output_factory", "message"),
    [
        (
            lambda prompt: _raw_request_output([999]),
            "changed the submitted prompt token IDs",
        ),
        (
            lambda prompt: _raw_request_output(prompt, completion_index=1),
            "completion index",
        ),
        (
            lambda prompt: _raw_request_output(prompt, finished=False),
            "not final",
        ),
    ],
)
def test_live_client_rejects_backend_count_or_identity_mismatch(
    output_factory,
    message: str,
) -> None:
    engine = _Engine()
    engine.output_factory = output_factory
    client, *_ = _client(engine)

    with pytest.raises(ReplayMismatchError, match=message):
        client.generate(_request())
