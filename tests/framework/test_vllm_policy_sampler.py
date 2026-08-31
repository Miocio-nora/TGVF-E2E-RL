from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import subprocess
import sys

import pytest

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement
from tgvf_rl.framework.vllm import (
    VLLM_PROCESSED_LOGPROBS_MODE,
    VLLMOutputDecodingContract,
    VLLMPolicySampler,
    VLLMPolicyTurnRequest,
    VLLMPolicyTurnResponse,
    VLLMTerminationOutcome,
    VLLMTokenLogprob,
    VLLMTurnRNGIdentity,
    VLLMTurnTerminationContract,
)
from tgvf_rl.policy import PilotSamplingConfig
from tgvf_rl.protocol import TokenByteSpan


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
POLICY = PolicyVersion("pilot", 7, SHA0)


def test_sampler_module_is_safe_under_agent_loop_first_import_order() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tgvf_rl.environment.agent_loop; "
                "import tgvf_rl.framework.vllm.sampler"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _char_tokens(text: str) -> tuple[tuple[int, ...], tuple[TokenByteSpan, ...]]:
    token_ids: list[int] = []
    spans: list[TokenByteSpan] = []
    byte_cursor = 0
    for index, char in enumerate(text):
        token_id = 1000 + index
        width = len(char.encode("utf-8"))
        token_ids.append(token_id)
        spans.append(TokenByteSpan(index, token_id, byte_cursor, byte_cursor + width))
        byte_cursor += width
    return tuple(token_ids), tuple(spans)


class _RNG:
    def for_turn(self, prompt_token_ids, *, turn_index, behavior_policy):
        assert behavior_policy == POLICY
        return VLLMTurnRNGIdentity(
            seed=42 + turn_index,
            rng_state_sha256=sha256(
                f"{prompt_token_ids}:{turn_index}".encode()
            ).hexdigest(),
        )


class _Context:
    def sha256_for_turn(self, prompt_token_ids, *, turn_index):
        return sha256(f"context:{prompt_token_ids}:{turn_index}".encode()).hexdigest()


class _Client:
    backend_version = "0.12.0"
    logprobs_mode = VLLM_PROCESSED_LOGPROBS_MODE

    def __init__(
        self,
        text: str,
        *,
        finish_reason: str = "stop",
        stop_reason: int | str | None = None,
        missing_selected: bool = False,
        request_hash_override: str | None = None,
        request_id_override: str | None = None,
        prompt_override: tuple[int, ...] | None = None,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.missing_selected = missing_selected
        self.request_hash_override = request_hash_override
        self.request_id_override = request_id_override
        self.prompt_override = prompt_override
        self.requests: list[VLLMPolicyTurnRequest] = []
        self.responses: list[VLLMPolicyTurnResponse] = []

    def generate(self, request: VLLMPolicyTurnRequest) -> VLLMPolicyTurnResponse:
        self.requests.append(request)
        token_ids, spans = _char_tokens(self.text)
        token_logprobs = tuple(
            (
                VLLMTokenLogprob(
                    token_id=(token_id + 1 if self.missing_selected else token_id),
                    logprob=-0.01 * (index + 1),
                    rank=index + 1,
                    decoded_token=char,
                ),
            )
            for index, (token_id, char) in enumerate(
                zip(token_ids, self.text, strict=True)
            )
        )
        response = VLLMPolicyTurnResponse(
            request_id=self.request_id_override or request.request_id,
            backend_request_sha256=(
                self.request_hash_override or request.backend_request_sha256
            ),
            prompt_token_ids=self.prompt_override or request.prompt_token_ids,
            text=self.text,
            token_ids=token_ids,
            token_byte_spans=spans,
            token_logprobs=token_logprobs,
            finish_reason=self.finish_reason,
            stop_reason=self.stop_reason,
        )
        self.responses.append(response)
        return response


def _parameters(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
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
        "max_tokens": 8192,
        "logprobs": True,
    }
    values.update(updates)
    return values


def _sampler(client: _Client) -> VLLMPolicySampler:
    return VLLMPolicySampler(
        client=client,
        behavior_policy=POLICY,
        rng=_RNG(),
        request_context=_Context(),
        decoding=VLLMOutputDecodingContract(
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind="final_only",
        ),
        termination=_termination(),
    )


def _termination(
    *,
    required_stop_strings: tuple[str, ...] = ("</tool_call>",),
    required_stop_token_ids: tuple[int, ...] = (),
    include_stop_str_in_output: bool = True,
    tool_suffixes: tuple[str, ...] = ("",),
    tool_outcomes: tuple[VLLMTerminationOutcome, ...] = (
        VLLMTerminationOutcome("stop", "</tool_call>"),
    ),
    final_outcomes: tuple[VLLMTerminationOutcome, ...] = (
        VLLMTerminationOutcome("stop", None),
        VLLMTerminationOutcome("stop", 151645),
        VLLMTerminationOutcome("length", None),
    ),
    preserve_invalid_tool_call_output: bool = False,
) -> VLLMTurnTerminationContract:
    return VLLMTurnTerminationContract(
        required_request_stop_strings=required_stop_strings,
        required_request_stop_token_ids=required_stop_token_ids,
        include_stop_str_in_output=include_stop_str_in_output,
        tool_call_terminal_suffixes=tool_suffixes,
        tool_call_outcomes=tool_outcomes,
        final_turn_outcomes=final_outcomes,
        preserve_invalid_tool_call_output=preserve_invalid_tool_call_output,
    )


def test_sampler_captures_actual_selected_logprobs_and_exact_identities() -> None:
    text = (
        "inspect\n</think>\n<tool_call>"
        '{"name":"tgvf_focus_tool","arguments":{"target":"赤い看板"}}'
        "</tool_call>"
    )
    client = _Client(text, stop_reason="</tool_call>")
    sampler = _sampler(client)

    sampled = sampler.sample((10, 20, 30), _parameters(), turn_index=0)

    request = client.requests[0]
    response = client.responses[0]
    assert sampled.text == text
    assert sampled.token_ids == response.token_ids
    assert sampled.behavior_logprobs == tuple(
        -0.01 * (index + 1) for index in range(len(response.token_ids))
    )
    assert sampled.sampling.policy_version == POLICY
    assert sampled.sampling.seed == 42
    assert sampled.sampling.backend_version == "0.12.0"
    assert sampled.sampling.measurement is LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS
    assert sampled.sampling.max_tokens == 8192
    assert sampled.sampling.stop_strings == ("</tool_call>",)
    assert sampled.backend_request_sha256 == request.backend_request_sha256
    assert sampled.backend_response_sha256 == response.backend_response_sha256
    assert sampled.think_token_span.start == 0
    assert sampled.think_token_span.end < len(sampled.token_ids)

    sent = request.sampling_parameters
    assert sent["seed"] == 42
    assert sent["logprobs"] == 0
    assert sent["n"] == 1
    assert sent["prompt_logprobs"] is None
    assert sent["detokenize"] is True
    assert sent["skip_special_tokens"] is False
    assert request.backend_prompt_payload_sha256 == _Context().sha256_for_turn(
        (10, 20, 30), turn_index=0
    )


def test_sampler_identity_satisfies_bound_policy_pilot_contract() -> None:
    config = PilotSamplingConfig().bind_run_inputs(
        min_p=0.0,
        stop_token_ids=(151645,),
        stop_strings=("</tool_call>",),
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    client = _Client("reason</think>answer", stop_reason=None)
    sampled = _sampler(client).sample(
        (1, 2), config.as_vllm_parameters(max_tokens=8000), turn_index=1
    )

    config.validate_sampling_identity(sampled.sampling, expected_max_tokens=8000)
    assert sampled.sampling.seed == 43
    assert sampled.sampling.asynchronous_staleness_steps == 0


def test_final_answer_may_terminate_on_explicit_eos_without_becoming_tool_call() -> (
    None
):
    text = "reason</think>final answer"
    client = _Client(text, finish_reason="stop", stop_reason=151645)

    sampled = _sampler(client).sample((1,), _parameters(), turn_index=0)

    assert sampled.text == text
    assert "</tool_call>" not in sampled.text
    assert '"stop_reason":151645' in sampled.stop_reason


@pytest.mark.parametrize(
    "text",
    (
        "unfinished reasoning without a closer",
        "reason</think>extra</think>answer",
        "<think>duplicate opener</think>answer",
    ),
)
def test_model_think_format_errors_preserve_sampled_behavior(text: str) -> None:
    client = _Client(text, finish_reason="length", stop_reason=None)

    sampled = _sampler(client).sample((1,), _parameters(), turn_index=0)
    response = client.responses[0]

    assert sampled.think_token_span is None
    assert sampled.text == text
    assert sampled.token_ids == response.token_ids
    assert sampled.behavior_logprobs == tuple(
        -0.01 * (index + 1) for index in range(len(response.token_ids))
    )


@pytest.mark.parametrize("mode", ("raw_logprobs", "raw_logits", "processed_logits"))
def test_sampler_rejects_non_behavior_logprob_modes(mode: str) -> None:
    client = _Client("reason</think>answer")
    client.logprobs_mode = mode

    with pytest.raises(IdentityMismatchError, match="processed_logprobs"):
        _sampler(client)


def test_sampler_rechecks_backend_identity_on_every_turn() -> None:
    client = _Client("reason</think>answer")
    sampler = _sampler(client)
    client.backend_version = "0.12.1"

    with pytest.raises(IdentityMismatchError, match="0.12.0"):
        sampler.sample((1,), _parameters(), turn_index=0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"stop": []}, "termination stops"),
        ({"include_stop_str_in_output": False}, "stop-output ownership"),
        ({"logprobs": False}, "logprobs"),
        ({"min_p": None}, "real number"),
    ],
)
def test_unresolved_or_implicit_sampling_fields_fail_closed(
    updates: dict[str, object], message: str
) -> None:
    sampler = _sampler(_Client("reason</think>answer"))

    with pytest.raises((ContractUnsetError, TypeError), match=message):
        sampler.sample((1,), _parameters(**updates), turn_index=0)


def test_approved_close_token_id_can_supply_explicit_turn_stop() -> None:
    close_id = 151658
    client = _Client("reason</think><tool_call>{}</tool_call>", stop_reason=close_id)
    sampler = VLLMPolicySampler(
        client=client,
        behavior_policy=POLICY,
        rng=_RNG(),
        request_context=_Context(),
        decoding=VLLMOutputDecodingContract(True, False, False, "final_only"),
        termination=_termination(
            required_stop_strings=(),
            required_stop_token_ids=(close_id,),
            tool_outcomes=(VLLMTerminationOutcome("stop", close_id),),
        ),
    )

    sampled = sampler.sample(
        (1,),
        _parameters(stop=[], stop_token_ids=[151645, close_id]),
        turn_index=0,
    )
    assert sampled.sampling.stop_token_ids == (151645, close_id)


def test_missing_sampled_token_entry_is_never_replaced_by_a_replayed_logprob() -> None:
    client = _Client("reason</think>answer", missing_selected=True)

    with pytest.raises(ReplayMismatchError, match="exactly once"):
        _sampler(client).sample((1,), _parameters(), turn_index=0)


@pytest.mark.parametrize(
    "client",
    [
        _Client("reason</think>answer", request_id_override="wrong"),
        _Client("reason</think>answer", request_hash_override=SHA1),
        _Client("reason</think>answer", prompt_override=(9,)),
    ],
)
def test_response_must_echo_the_exact_request_identity(client: _Client) -> None:
    with pytest.raises((IdentityMismatchError, ReplayMismatchError)):
        _sampler(client).sample((1,), _parameters(), turn_index=0)


@pytest.mark.parametrize(
    ("text", "finish_reason", "stop_reason", "message"),
    [
        (
            "r</think><tool_call>{}</tool_call></tool_call>",
            "stop",
            "</tool_call>",
            "more than one",
        ),
        (
            "r</think><tool_call>{}</tool_call>trailing",
            "stop",
            "</tool_call>",
            "suffix outside",
        ),
        (
            "r</think><tool_call>{}</tool_call>",
            "length",
            None,
            "termination differs",
        ),
    ],
)
def test_backend_must_enforce_one_complete_tool_call_turn_boundary(
    text: str,
    finish_reason: str,
    stop_reason: str | None,
    message: str,
) -> None:
    client = _Client(
        text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )

    with pytest.raises(ReplayMismatchError, match=message):
        _sampler(client).sample((1,), _parameters(), turn_index=0)


def test_run_bound_suffix_may_include_native_eos_when_golden_selects_it() -> None:
    suffix = "<|im_end|>"
    text = f"r</think><tool_call>{{}}</tool_call>{suffix}"
    client = _Client(text, finish_reason="stop", stop_reason=151645)
    sampler = VLLMPolicySampler(
        client=client,
        behavior_policy=POLICY,
        rng=_RNG(),
        request_context=_Context(),
        decoding=VLLMOutputDecodingContract(True, False, False, "final_only"),
        termination=_termination(
            required_stop_strings=(),
            required_stop_token_ids=(151645,),
            tool_suffixes=(suffix,),
            tool_outcomes=(VLLMTerminationOutcome("stop", 151645),),
        ),
    )

    sampled = sampler.sample(
        (1,),
        _parameters(stop=[], stop_token_ids=[151645]),
        turn_index=0,
    )
    assert sampled.text.endswith(f"</tool_call>{suffix}")


@pytest.mark.parametrize(
    "text",
    (
        "r</think><tool_call>{}</tool_call>trailing",
        "r</think><tool_call>{}</tool_call><tool_call>{}</tool_call>",
    ),
)
def test_direct_only_contract_preserves_invalid_tool_output_for_penalty(
    text: str,
) -> None:
    client = _Client(text, finish_reason="stop", stop_reason=151645)
    sampler = VLLMPolicySampler(
        client=client,
        behavior_policy=POLICY,
        rng=_RNG(),
        request_context=_Context(),
        decoding=VLLMOutputDecodingContract(True, False, False, "final_only"),
        termination=_termination(
            required_stop_strings=(),
            required_stop_token_ids=(151645,),
            tool_outcomes=(VLLMTerminationOutcome("stop", 151645),),
            preserve_invalid_tool_call_output=True,
        ),
    )

    sampled = sampler.sample(
        (1,),
        _parameters(stop=[], stop_token_ids=[151645]),
        turn_index=0,
    )
    assert sampled.text == text


def test_response_hash_changes_when_actual_behavior_logprob_changes() -> None:
    text = "reason</think>answer"
    client = _Client(text)
    sampled = _sampler(client).sample((1,), _parameters(), turn_index=0)
    response = client.responses[0]
    changed = replace(
        response,
        token_logprobs=(
            (replace(response.token_logprobs[0][0], logprob=-0.5),),
            *response.token_logprobs[1:],
        ),
    )

    assert changed.backend_response_sha256 != sampled.backend_response_sha256


def test_output_decoder_contract_has_no_implicit_native_marker_behavior() -> None:
    with pytest.raises(ContractUnsetError, match="markers"):
        VLLMOutputDecodingContract(True, True, False, "final_only")
    with pytest.raises(ContractUnsetError, match="assistant text"):
        VLLMOutputDecodingContract(False, False, False, "final_only")
