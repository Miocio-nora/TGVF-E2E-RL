from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier

import pytest

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.framework.vllm import (
    ContentAddressedVLLMTurnRNG,
    FastTokenizerTokenByteSpanDecoder,
    LiveVLLMTurnContextRegistry,
    VLLMLivePromptInputs,
    VLLMOutputDecodingContract,
    VLLMPolicyTurnRequest,
    VLLMResolvedObservationPayload,
    VLLMTurnRNGIdentity,
    bind_preexpanded_prompt_contract,
    prompt_token_ids_sha256,
    split_preexpanded_prompt_contract,
)
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol import TokenByteSpan
from tgvf_rl.protocol.schema import StandardToolError


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
POLICY = PolicyVersion("pilot-runtime", 7, SHA0)
DECODING = VLLMOutputDecodingContract(True, False, False, "final_only")
IMAGE_TOKEN_ID = 9000


class _FastTokenizer:
    is_fast = True

    def __init__(self, *, ids=None, tokens=None):
        self.ids = ids
        self.tokens = tokens
        self.all_special_ids = []

    @staticmethod
    def get_added_vocab():
        return {}

    def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        assert token_ids == self.ids
        return self.tokens
def test_content_addressed_rng_is_stable_and_binds_every_identity() -> None:
    rng = ContentAddressedVLLMTurnRNG(
        master_seed=42, stream_identity="pilot/train/trajectory-0003"
    )

    first = rng.for_turn((10, 20, 30), turn_index=1, behavior_policy=POLICY)
    repeated = rng.for_turn((10, 20, 30), turn_index=1, behavior_policy=POLICY)

    assert first == repeated
    assert 0 <= first.seed < 2**31 - 1
    assert len(first.rng_state_sha256) == 64
    assert (
        rng.stream_identity_sha256 == sha256(b"pilot/train/trajectory-0003").hexdigest()
    )
    assert prompt_token_ids_sha256((10, 20, 30)) != prompt_token_ids_sha256(
        (10, 20, 31)
    )

    changed = {
        ContentAddressedVLLMTurnRNG(
            master_seed=43, stream_identity=rng.stream_identity
        ).for_turn((10, 20, 30), turn_index=1, behavior_policy=POLICY),
        ContentAddressedVLLMTurnRNG(
            master_seed=42, stream_identity="different-stream"
        ).for_turn((10, 20, 30), turn_index=1, behavior_policy=POLICY),
        rng.for_turn((10, 20, 31), turn_index=1, behavior_policy=POLICY),
        rng.for_turn((10, 20, 30), turn_index=2, behavior_policy=POLICY),
        rng.for_turn(
            (10, 20, 30),
            turn_index=1,
            behavior_policy=PolicyVersion("pilot-runtime", 8, SHA1),
        ),
    }
    assert len(changed) == 5
    assert first not in changed


def test_fast_tokenizer_decoder_produces_exact_unicode_utf8_spans() -> None:
    text = "a赤🙂"
    tokenizer = _FastTokenizer(
        ids=[11, 12, 13],
        tokens=["a", "èµ¤", "ðŁĻĤ"],
    )
    decoder = FastTokenizerTokenByteSpanDecoder(tokenizer)

    spans = decoder.spans_for_output(
        text=text,
        token_ids=(11, 12, 13),
        decoding=DECODING,
    )

    assert spans == (
        TokenByteSpan(0, 11, 0, 1),
        TokenByteSpan(1, 12, 1, 4),
        TokenByteSpan(2, 13, 4, 8),
    )


def test_fast_tokenizer_decoder_handles_one_unicode_character_split_across_tokens(
) -> None:
    tokenizer = _FastTokenizer(
        ids=[21, 22],
        # The UTF-8 bytes E2 80 8D for ZERO WIDTH JOINER are split 2 + 1.
        tokens=["âĢ", "į"],
    )

    spans = FastTokenizerTokenByteSpanDecoder(tokenizer).spans_for_output(
        text="\u200d",
        token_ids=(21, 22),
        decoding=DECODING,
    )

    assert spans == (
        TokenByteSpan(0, 21, 0, 2),
        TokenByteSpan(1, 22, 2, 3),
    )


def test_fast_tokenizer_decoder_handles_truncated_utf8_with_replacement() -> None:
    tokenizer = _FastTokenizer(
        ids=[21, 22],
        # E2 80 is an incomplete three-byte scalar. Qwen's ByteLevel decoder
        # emits one U+FFFD instead of preserving the two raw bytes.
        tokens=["â", "Ģ"],
    )

    spans = FastTokenizerTokenByteSpanDecoder(tokenizer).spans_for_output(
        text="\ufffd",
        token_ids=(21, 22),
        decoding=DECODING,
    )

    assert spans == (
        TokenByteSpan(0, 21, 0, 1),
        TokenByteSpan(1, 22, 1, 3),
    )


def test_fast_tokenizer_decoder_keeps_replacement_and_following_ascii_exact() -> None:
    tokenizer = _FastTokenizer(ids=[21, 22], tokens=["â", "a"])

    spans = FastTokenizerTokenByteSpanDecoder(tokenizer).spans_for_output(
        text="\ufffda",
        token_ids=(21, 22),
        decoding=DECODING,
    )

    assert spans == (
        TokenByteSpan(0, 21, 0, 3),
        TokenByteSpan(1, 22, 3, 4),
    )


def test_fast_tokenizer_decoder_accepts_noncanonical_sampled_segmentation() -> None:
    tokenizer = _FastTokenizer(ids=[1, 2], tokens=["a", "b"])

    spans = FastTokenizerTokenByteSpanDecoder(tokenizer).spans_for_output(
        text="ab",
        token_ids=(1, 2),
        decoding=DECODING,
    )

    assert spans == (
        TokenByteSpan(0, 1, 0, 1),
        TokenByteSpan(1, 2, 1, 2),
    )


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["a"], "count differs"),
        (["a", "c"], "do not reconstruct"),
        (["a", "中"], "audited ByteLevel alphabet"),
        (["a", ""], "non-empty byte pieces"),
    ],
)
def test_fast_tokenizer_decoder_rejects_invalid_byte_level_pieces(
    tokens,
    message,
) -> None:
    decoder = FastTokenizerTokenByteSpanDecoder(
        _FastTokenizer(ids=[1, 2], tokens=tokens)
    )
    with pytest.raises(ReplayMismatchError, match=message):
        decoder.spans_for_output(text="ab", token_ids=(1, 2), decoding=DECODING)


def test_span_decoder_requires_an_explicit_fast_tokenizer() -> None:
    tokenizer = _FastTokenizer()
    tokenizer.is_fast = False
    with pytest.raises(TypeError, match="fast tokenizer"):
        FastTokenizerTokenByteSpanDecoder(tokenizer)


def test_span_decoder_rejects_spaces_inserted_between_special_tokens() -> None:
    decoder = FastTokenizerTokenByteSpanDecoder(
        _FastTokenizer(ids=[1], tokens=["a"])
    )
    decoding = VLLMOutputDecodingContract(True, False, True, "final_only")

    with pytest.raises(IdentityMismatchError, match="spaces_between_special_tokens"):
        decoder.spans_for_output(text="a", token_ids=(1,), decoding=decoding)


class _Resolver:
    def __init__(self):
        self.calls = []
        self.items = []

    def resolve(self, observation, *, call_index):
        self.calls.append((observation, call_index))
        item = {"image_embeds": object(), "image_grid_thw": object()}
        self.items.append(item)
        return VLLMResolvedObservationPayload(
            observation=observation,
            call_index=call_index,
            modality="image",
            multi_modal_data_item=item,
            payload_sha256=sha256(
                f"{observation.observation_id}:{call_index}".encode()
            ).hexdigest(),
            multi_modal_uuid=f"observation-{call_index}",
        )


def _initial_inputs(prompt) -> VLLMLivePromptInputs:
    return VLLMLivePromptInputs(
        backend_prompt_payload_sha256=SHA1,
        multi_modal_data={"image": [{"source": object()}]},
        mm_processor_kwargs=bind_preexpanded_prompt_contract(
            {"do_resize": False},
            prompt_token_ids=prompt,
            image_token_id=IMAGE_TOKEN_ID,
            expected_image_items=1,
        ),
        multi_modal_uuids={"image": ["source-image"]},
    )


def _request(
    prompt,
    *,
    turn_index,
    context_sha256,
    request_id=None,
) -> VLLMPolicyTurnRequest:
    return VLLMPolicyTurnRequest(
        request_id=request_id or f"request-{turn_index}",
        prompt_token_ids=tuple(prompt),
        sampling_parameters={"seed": 100 + turn_index},
        turn_index=turn_index,
        behavior_policy=POLICY,
        rng=VLLMTurnRNGIdentity(
            seed=100 + turn_index,
            rng_state_sha256=sha256(f"rng-{turn_index}".encode()).hexdigest(),
        ),
        backend_prompt_payload_sha256=context_sha256,
        backend_version="0.12.0",
        logprobs_mode="processed_logprobs",
        decoding=DECODING,
        termination_contract_sha256=SHA2,
    )


def _sampled(request: VLLMPolicyTurnRequest, *, suffix: str) -> SampledPolicyTurn:
    text = f"inspect-{suffix}</think><tool_call>{{}}</tool_call>"
    token_ids = tuple(ord(character) for character in text)
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    sampling = SamplingIdentity(
        policy_version=request.behavior_policy,
        backend="vllm",
        backend_version="0.12.0",
        seed=request.rng.seed,
        rng_state_sha256=request.rng.rng_state_sha256,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    think_end = text.index("</think>") + len("</think>")
    return SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.1 for _ in token_ids),
        sampling=sampling,
        think_token_span=TokenSpan(0, think_end),
        stop_reason="tool-call",
        backend_request_sha256=request.backend_request_sha256,
        backend_response_sha256=sha256(
            f"response-{request.turn_index}".encode()
        ).hexdigest(),
    )


def _resolve_turn(registry, prompt, *, turn_index):
    context_sha = registry.sha256_for_turn(prompt, turn_index=turn_index)
    request = _request(prompt, turn_index=turn_index, context_sha256=context_sha)
    inputs = registry.for_request(request)
    return request, inputs


def test_registry_appends_success_payload_and_preserves_it_across_error() -> None:
    resolver = _Resolver()
    registry = LiveVLLMTurnContextRegistry(observation_resolver=resolver)
    prompt0 = (10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 20)
    initial = _initial_inputs(prompt0)
    registry.register_initial_prompt(prompt0, initial)

    request0, returned0 = _resolve_turn(registry, prompt0, turn_index=0)
    assert returned0 is initial
    sampled0 = _sampled(request0, suffix="zero")
    prompt1 = (
        prompt0
        + sampled0.token_ids
        + (
            9001,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            9002,
        )
    )
    handle = ObservationHandle("observation-0", sha256(b"record-0").hexdigest())
    registry.register_tool_turn(
        previous_prompt_token_ids=prompt0,
        sampled_turn=sampled0,
        updated_prompt_token_ids=prompt1,
        observation=handle,
        call_index=0,
    )

    request1, returned1 = _resolve_turn(registry, prompt1, turn_index=1)
    assert returned1.backend_prompt_payload_sha256 != SHA1
    assert (
        returned1.multi_modal_data["image"][0] is initial.multi_modal_data["image"][0]
    )
    assert returned1.multi_modal_data["image"][1] is resolver.items[0]
    assert returned1.multi_modal_uuids == {"image": ["source-image", "observation-0"]}

    sampled1 = _sampled(request1, suffix="one")
    prompt2 = prompt1 + sampled1.token_ids + (9002,)
    error = StandardToolError(
        code="tool_execution_failed",
        message="failed",
        attempt_index=1,
        recoverable=True,
        maximum_tool_calls=4,
    )
    registry.register_tool_turn(
        previous_prompt_token_ids=prompt1,
        sampled_turn=sampled1,
        updated_prompt_token_ids=prompt2,
        observation=error,
        call_index=1,
    )

    _, returned2 = _resolve_turn(registry, prompt2, turn_index=2)
    assert returned2.backend_prompt_payload_sha256 not in {
        initial.backend_prompt_payload_sha256,
        returned1.backend_prompt_payload_sha256,
    }
    assert returned2.multi_modal_data is returned1.multi_modal_data
    assert returned2.multi_modal_uuids is returned1.multi_modal_uuids
    contract1, clean1 = split_preexpanded_prompt_contract(returned1.mm_processor_kwargs)
    contract2, clean2 = split_preexpanded_prompt_contract(returned2.mm_processor_kwargs)
    assert clean1 == clean2 == {"do_resize": False}
    assert contract1.prompt_token_ids_sha256 != contract2.prompt_token_ids_sha256
    assert resolver.calls == [(handle, 0)]


def test_registry_rejects_prompt_turn_hash_request_and_call_reuse() -> None:
    resolver = _Resolver()
    registry = LiveVLLMTurnContextRegistry(observation_resolver=resolver)
    prompt = (1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2)
    initial = _initial_inputs(prompt)
    registry.register_initial_prompt(prompt, initial)
    with pytest.raises(ReplayMismatchError, match="already registered"):
        registry.register_initial_prompt(prompt, initial)

    context_sha = registry.sha256_for_turn(prompt, turn_index=0)
    with pytest.raises(ReplayMismatchError, match="already consumed"):
        registry.sha256_for_turn(prompt, turn_index=0)

    wrong_context = _request(
        prompt, turn_index=0, context_sha256=SHA2, request_id="wrong-context"
    )
    with pytest.raises(IdentityMismatchError, match="context hash"):
        registry.for_request(wrong_context)

    request = _request(prompt, turn_index=0, context_sha256=context_sha)
    registry.for_request(request)
    with pytest.raises(ReplayMismatchError, match="already resolved"):
        registry.for_request(request)

    sampled = _sampled(request, suffix="reuse")
    updated = (
        prompt
        + sampled.token_ids
        + (
            99,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            100,
        )
    )
    handle = ObservationHandle("observation-0", sha256(b"record-0").hexdigest())
    with pytest.raises(ReplayMismatchError, match="unique and contiguous"):
        registry.register_tool_turn(
            previous_prompt_token_ids=prompt,
            sampled_turn=sampled,
            updated_prompt_token_ids=updated,
            observation=handle,
            call_index=1,
        )
    registry.register_tool_turn(
        previous_prompt_token_ids=prompt,
        sampled_turn=sampled,
        updated_prompt_token_ids=updated,
        observation=handle,
        call_index=0,
    )
    with pytest.raises(ReplayMismatchError, match="already registered"):
        registry.register_tool_turn(
            previous_prompt_token_ids=prompt,
            sampled_turn=sampled,
            updated_prompt_token_ids=updated,
            observation=handle,
            call_index=0,
        )


def test_registry_rejects_error_call_mismatch_without_consuming_turn() -> None:
    registry = LiveVLLMTurnContextRegistry(observation_resolver=_Resolver())
    prompt = (3, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 4)
    registry.register_initial_prompt(prompt, _initial_inputs(prompt))
    request, _ = _resolve_turn(registry, prompt, turn_index=0)
    sampled = _sampled(request, suffix="error")
    updated = prompt + sampled.token_ids + (77,)
    error = StandardToolError(
        code="tool_execution_failed",
        message="failed",
        attempt_index=2,
        recoverable=True,
        maximum_tool_calls=4,
    )
    with pytest.raises(IdentityMismatchError, match="attempt_index"):
        registry.register_tool_turn(
            previous_prompt_token_ids=prompt,
            sampled_turn=sampled,
            updated_prompt_token_ids=updated,
            observation=error,
            call_index=1,
        )
    registry.register_tool_turn(
        previous_prompt_token_ids=prompt,
        sampled_turn=sampled,
        updated_prompt_token_ids=updated,
        observation=error,
        call_index=2,
    )


def test_registry_context_claim_is_thread_safe_and_single_use() -> None:
    registry = LiveVLLMTurnContextRegistry(observation_resolver=_Resolver())
    prompt = (5, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 6)
    registry.register_initial_prompt(prompt, _initial_inputs(prompt))
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        try:
            return registry.sha256_for_turn(prompt, turn_index=0)
        except ReplayMismatchError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: claim(), range(2)))
    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, ReplayMismatchError) for result in results) == 1
