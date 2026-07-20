from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.framework.verl import (
    build_padded_data_proto_payload,
    make_objective_sentinels,
    trajectory_to_rollout_bridge,
)
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.policy.exact_replay import (
    ExactPolicyReplayMaterializer,
    RecordedPolicyForwardBinding,
    RecordedPolicyForwardOutput,
    RecordedPolicyForwardStateProof,
    ReplayParameterization,
    require_single_sequence_replay_bundle,
)
from tgvf_rl.qwen.base import (
    RecordedReplayRequest,
    gather_behavior_measure_logprobs,
)
from tgvf_rl.trajectories import BehaviorTraceStore, VLLMBehaviorRecorder
from tgvf_rl.trajectories.schema import (
    AssistantTurnRecord,
    ToolCallRecord,
    ToolObservationRecord,
    TrajectoryIdentity,
    TrajectoryRecord,
    TrajectoryStop,
)
from tgvf_rl.trajectories.validation import TrajectoryValidator
from tests.support import (
    SHA0,
    populated_observation_store,
    policy_version,
    trajectory_source_visual,
)


BASE_WEIGHTS_SHA256 = "a" * 64
LORA_STATE_SHA256 = "b" * 64
REFERENCE_VERSION = PolicyVersion("qwen3-frozen-base", 0, BASE_WEIGHTS_SHA256)
FORWARD_IMPLEMENTATION_SHA256 = "1" * 64


class _FakeRecordedForward:
    def __init__(
        self,
        binding: RecordedPolicyForwardBinding,
        *,
        logit_scale: float,
        mutate_recorded_d: bool = False,
        drift_after_forward: bool = False,
    ) -> None:
        self.binding = binding
        self.actual_state_proof = binding.expected_state_proof()
        self.weight = torch.tensor(
            logit_scale,
            dtype=torch.float32,
            requires_grad=binding.role is ComponentRole.CURRENT,
        )
        self.mutate_recorded_d = mutate_recorded_d
        self.drift_after_forward = drift_after_forward
        self.grad_enabled: list[bool] = []
        self.seen_visual: list[tuple[tuple[torch.Tensor, ...], ...]] = []
        self.seen_input_ids: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def capture_state_proof(self) -> RecordedPolicyForwardStateProof:
        return self.actual_state_proof

    def forward_recorded(
        self, request: RecordedReplayRequest
    ) -> RecordedPolicyForwardOutput:
        self.grad_enabled.append(torch.is_grad_enabled())
        self.seen_input_ids.append(request.input_ids.detach().clone())
        self.seen_visual.append(
            tuple(
                (block.embeddings.detach().clone(),)
                + tuple(branch.detach().clone() for branch in block.deepstack)
                for block in request.visual_blocks
            )
        )
        if self.mutate_recorded_d:
            request.visual_blocks[-1].embeddings.zero_()
        batch, sequence = request.input_ids.shape
        vocabulary = 128
        vocabulary_axis = torch.linspace(
            -1.0, 1.0, vocabulary, dtype=torch.float32
        ).view(1, 1, vocabulary)
        position_axis = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
        logits = (
            self.weight * vocabulary_axis
            + 0.001 * position_axis * vocabulary_axis.square()
        ).expand(batch, -1, -1)
        self.outputs.append(logits)
        if self.drift_after_forward:
            self.actual_state_proof = replace(
                self.actual_state_proof,
                attention_backend="drifted_backend",
            )
        return RecordedPolicyForwardOutput(logits)


def _bindings(
    model,
    *,
    current_version: PolicyVersion | None = None,
):
    current = RecordedPolicyForwardBinding(
        role=ComponentRole.CURRENT,
        model=model,
        policy_version=current_version or policy_version(),
        parameterization=ReplayParameterization.BASE_PLUS_LORA,
        base_weights_sha256=BASE_WEIGHTS_SHA256,
        lora_state_sha256=LORA_STATE_SHA256,
        parameters_frozen=False,
        deterministic_forward=True,
        lora_dropout=0.0,
        model_training=True,
        compute_dtype="float32",
        autocast_enabled=False,
        autocast_dtype=None,
        attention_backend="cpu_fixture_sdpa",
        forward_implementation_sha256=FORWARD_IMPLEMENTATION_SHA256,
    )
    proximal_old = RecordedPolicyForwardBinding(
        role=ComponentRole.PROXIMAL_OLD,
        model=model,
        policy_version=current_version or policy_version(),
        parameterization=ReplayParameterization.BASE_PLUS_LORA,
        base_weights_sha256=BASE_WEIGHTS_SHA256,
        lora_state_sha256=LORA_STATE_SHA256,
        parameters_frozen=True,
        deterministic_forward=True,
        lora_dropout=0.0,
        model_training=False,
        compute_dtype="float32",
        autocast_enabled=False,
        autocast_dtype=None,
        attention_backend="cpu_fixture_sdpa",
        forward_implementation_sha256=FORWARD_IMPLEMENTATION_SHA256,
    )
    reference = RecordedPolicyForwardBinding(
        role=ComponentRole.REFERENCE,
        model=model,
        policy_version=REFERENCE_VERSION,
        parameterization=ReplayParameterization.FROZEN_BASE,
        base_weights_sha256=BASE_WEIGHTS_SHA256,
        lora_state_sha256=None,
        parameters_frozen=True,
        deterministic_forward=True,
        lora_dropout=0.0,
        model_training=False,
        compute_dtype="float32",
        autocast_enabled=False,
        autocast_dtype=None,
        attention_backend="cpu_fixture_sdpa",
        forward_implementation_sha256=FORWARD_IMPLEMENTATION_SHA256,
    )
    return current, proximal_old, reference


def _payload(
    *,
    different_reference_mask: bool = False,
    non_identity_sampling: bool = False,
):
    store, observation_handle = populated_observation_store()
    observation = store.resolve_record(observation_handle)
    version = policy_version()
    identity = TrajectoryIdentity("smoke", "sample", 0, "group")
    sampling = SamplingIdentity(
        policy_version=version,
        backend="vllm",
        backend_version="exact-replay-fixture",
        seed=42,
        rng_state_sha256=SHA0,
        temperature=1.0,
        top_p=0.9 if non_identity_sampling else 1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    behavior_store = BehaviorTraceStore()
    recorder = VLLMBehaviorRecorder(behavior_store)

    action_tokens = OwnedTokenSequence(
        (20, 21, 22),
        (TokenOwnership.POLICY_SAMPLED,) * 3,
    )
    action_trace = recorder.record(
        trajectory_id=identity.canonical_id,
        assistant_turn_index=0,
        tokens=action_tokens,
        actual_sampled_logprobs=(-0.2, -0.3, -0.4),
        sampling=sampling,
        behavior_policy=version,
        backend_request_sha256="c" * 64,
        backend_response_sha256="d" * 64,
    )
    answer_tokens = OwnedTokenSequence(
        (30, 31, 32, 33),
        (TokenOwnership.POLICY_SAMPLED,) * 4,
    )
    answer_trace = recorder.record(
        trajectory_id=identity.canonical_id,
        assistant_turn_index=1,
        tokens=answer_tokens,
        actual_sampled_logprobs=(-0.5, -0.6, -0.7, -0.8),
        sampling=replace(sampling, seed=43),
        behavior_policy=version,
        backend_request_sha256="e" * 64,
        backend_response_sha256="f" * 64,
    )
    native_observation_tokens = (90, 91)
    trajectory = TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=identity,
        model=observation.model,
        behavior_policy=version,
        assistant_turns=(
            AssistantTurnRecord(
                turn_index=0,
                raw_text="inspect</think><tool_call>fixture</tool_call>",
                tokens=action_tokens,
                behavior_trace=action_trace,
                think_span=TokenSpan(0, 1),
                is_tool_call=True,
            ),
            AssistantTurnRecord(
                turn_index=1,
                raw_text="read evidence</think>answer",
                tokens=answer_tokens,
                behavior_trace=answer_trace,
                think_span=TokenSpan(0, 2),
                is_tool_call=False,
            ),
        ),
        tool_calls=(
            ToolCallRecord(
                call_index=0,
                assistant_turn_index=0,
                function_name="tgvf_focus_tool",
                target="red label",
                target_token_span=TokenSpan(1, 3),
                target_char_span=(0, 9),
                raw_call_text="fixture",
            ),
        ),
        observations=(
            ToolObservationRecord(
                call_index=0,
                handle=observation_handle,
                template_token_ids=native_observation_tokens,
            ),
        ),
        final_answer="answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )

    prompt_ids = (101, 102, 103)
    response_ids = (
        action_tokens.token_ids + native_observation_tokens + answer_tokens.token_ids
    )
    final_ids = list(prompt_ids + response_ids)
    sequence = len(final_ids)
    input_ids = store.put_tensor(
        "exact-replay.input_ids",
        torch.tensor([final_ids], dtype=torch.int64),
    )
    position_ids = store.put_tensor(
        "exact-replay.position_ids",
        torch.arange(sequence, dtype=torch.int64).view(1, sequence),
    )
    attention_mask = store.put_tensor(
        "exact-replay.attention-mask",
        torch.ones((1, sequence), dtype=torch.bool),
    )
    if different_reference_mask:
        reference_values = torch.ones((1, sequence), dtype=torch.bool)
        reference_values[0, -1] = False
        reference_attention_mask = store.put_tensor(
            "exact-replay.reference-attention-mask", reference_values
        )
    else:
        reference_attention_mask = attention_mask
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="exact-replay-0",
        trajectory_id=identity.canonical_id,
        model=observation.model,
        behavior_policy=version,
        source_visual=trajectory_source_visual(observation),
        observation_handles=(observation_handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            policy_attention_mask=attention_mask,
            reference_attention_mask=reference_attention_mask,
            teacher_attention_mask=attention_mask,
        ),
        cache_mode="no_cache",
        cache_prefix_length=0,
        deterministic_forward=True,
        adapter_dropout=0.0,
    )
    replay_handle = store.put_replay(replay)
    bridge = trajectory_to_rollout_bridge(
        trajectory,
        validator=TrajectoryValidator(store, behavior_store),
        initial_prompt_token_ids=prompt_ids,
        native_tool_appended_token_ids=(native_observation_tokens,),
        replay_handle=replay_handle,
        sentinel_fields=make_objective_sentinels("exact-replay-row"),
        reward_score=0.8,
    )
    return (
        build_padded_data_proto_payload((bridge,), pad_token_id=0),
        observation,
    )


def _materializer(
    payload,
    observation,
    *,
    mutate_current_d: bool = False,
    proximal_logit_scale: float = 0.4,
    drift_current_state: bool = False,
):
    current_binding, proximal_old_binding, reference_binding = _bindings(
        observation.model
    )
    current = _FakeRecordedForward(
        current_binding,
        logit_scale=0.4,
        mutate_recorded_d=mutate_current_d,
        drift_after_forward=drift_current_state,
    )
    proximal_old = _FakeRecordedForward(
        proximal_old_binding, logit_scale=proximal_logit_scale
    )
    reference = _FakeRecordedForward(reference_binding, logit_scale=-0.2)
    materializer = ExactPolicyReplayMaterializer(
        current_forward=current,
        proximal_old_forward=proximal_old,
        reference_forward=reference,
    )
    return materializer, current, proximal_old, reference


def test_exact_replay_uses_same_materialized_d_and_selects_only_policy_tokens() -> None:
    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(payload, observation)

    result = materializer.materialize(payload)
    mask = payload.tensor_batch["response_mask"].bool()
    assert result.logprobs.policy_sampled_mask.equal(mask)
    assert torch.equal(
        result.logprobs.behavior.values,
        payload.tensor_batch["rollout_log_probs"],
    )
    assert torch.count_nonzero(result.logprobs.current.values[~mask]).item() == 0
    assert torch.count_nonzero(result.logprobs.reference.values[~mask]).item() == 0
    assert not torch.equal(
        result.logprobs.current.values[mask],
        result.logprobs.reference.values[mask],
    )
    assert not torch.equal(
        result.logprobs.current.values[mask],
        result.logprobs.behavior.values[mask],
    )
    torch.testing.assert_close(
        result.logprobs.current.values,
        result.logprobs.proximal_old.values,
        rtol=0,
        atol=0,
    )
    assert (
        result.logprobs.current.values.untyped_storage().data_ptr()
        != result.logprobs.proximal_old.values.untyped_storage().data_ptr()
    )
    assert result.policy_replay_bundle_sha256s == (
        payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0].bundle_sha256,
    )
    assert result.reference_replay_bundle_sha256s == (
        payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0].bundle_sha256,
    )
    assert result.proximal_old_replay_bundle_sha256s == (
        payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0].bundle_sha256,
    )
    assert len(result.exact_replay_evidence_sha256s) == 1

    response_indices = tuple(
        int(index) for index in torch.nonzero(mask[0], as_tuple=False).flatten()
    )
    prompt_length = len(payload.non_tensor_batch["tgvf_exact_prompt_ids"][0])
    sampled_positions = torch.tensor(
        [[prompt_length + index for index in response_indices]],
        dtype=torch.int64,
    )
    sampling = payload.non_tensor_batch["tgvf_behavior_trace_records"][0][
        0
    ].behavior.sampling
    expected_selected = gather_behavior_measure_logprobs(
        current.outputs[0],
        current.seen_input_ids[0],
        sampled_positions,
        sampling,
    ).squeeze(0)
    expected_current = torch.zeros_like(result.logprobs.current.values[0]).scatter(
        0,
        torch.tensor(response_indices, dtype=torch.int64),
        expected_selected,
    )
    torch.testing.assert_close(
        result.logprobs.current.values[0], expected_current, rtol=0, atol=0
    )
    sampled_ids = current.seen_input_ids[0][0, sampled_positions[0]]
    wrong_response_local_logits = current.outputs[0][
        0, torch.tensor(response_indices, dtype=torch.int64)
    ]
    wrong_selected = (
        torch.log_softmax(wrong_response_local_logits.float(), dim=-1)
        .gather(-1, sampled_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    assert not torch.equal(expected_selected, wrong_selected)

    assert current.grad_enabled == [True]
    assert proximal_old.grad_enabled == [False]
    assert reference.grad_enabled == [False]
    assert (
        len(current.seen_visual[0])
        == len(proximal_old.seen_visual[0])
        == len(reference.seen_visual[0])
        == 2
    )
    for current_block, reference_block in zip(
        current.seen_visual[0], reference.seen_visual[0], strict=True
    ):
        assert len(current_block) == len(reference_block)
        for current_tensor, reference_tensor in zip(
            current_block, reference_block, strict=True
        ):
            torch.testing.assert_close(current_tensor, reference_tensor, rtol=0, atol=0)
    for current_block, proximal_block in zip(
        current.seen_visual[0], proximal_old.seen_visual[0], strict=True
    ):
        for current_tensor, proximal_tensor in zip(
            current_block, proximal_block, strict=True
        ):
            torch.testing.assert_close(current_tensor, proximal_tensor, rtol=0, atol=0)
    torch.testing.assert_close(
        current.seen_visual[0][1][0],
        payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0]
        .tensor_payloads[
            tuple(
                item.sha256
                for item in payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][
                    0
                ].tensor_payloads
            ).index(observation.payload.main_d.address.digest)
        ]
        .tensor.unsqueeze(0),
        rtol=0,
        atol=0,
    )

    result.logprobs.current.values[mask].sum().backward()
    assert current.weight.grad is not None
    assert current.weight.grad.abs().item() > 0
    assert proximal_old.weight.grad is None
    assert reference.weight.grad is None

    proximal_before = result.logprobs.proximal_old.values.clone()
    with torch.no_grad():
        current.weight.add_(10.0)
    torch.testing.assert_close(
        result.logprobs.proximal_old.values, proximal_before, rtol=0, atol=0
    )


def test_exact_replay_rejects_transport_tamper_without_forwarding() -> None:
    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(payload, observation)
    bundle = payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0]
    d_digest = observation.payload.main_d.address.digest
    d_payload = next(item for item in bundle.tensor_payloads if item.sha256 == d_digest)
    d_payload.tensor.add_(1)

    with pytest.raises(ReplayMismatchError, match="replay tensor payload"):
        materializer.materialize(payload)
    assert (
        current.grad_enabled
        == proximal_old.grad_enabled
        == reference.grad_enabled
        == []
    )


def test_zero_staleness_rejects_distinct_proximal_replay_values() -> None:
    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(
        payload,
        observation,
        proximal_logit_scale=0.41,
    )

    with pytest.raises(
        ReplayMismatchError,
        match="zero-staleness current and proximal-old replay logprobs differ",
    ):
        materializer.materialize(payload)
    assert current.grad_enabled == [True]
    assert proximal_old.grad_enabled == [False]
    assert reference.grad_enabled == [False]
    assert current.outputs[0] is not proximal_old.outputs[0]


def test_forward_state_proof_is_checked_before_and_after_each_call() -> None:
    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(payload, observation)
    current.actual_state_proof = replace(
        current.actual_state_proof,
        base_weights_sha256="8" * 64,
    )
    with pytest.raises(IdentityMismatchError, match="actual forward state differs"):
        materializer.materialize(payload)
    assert (
        current.grad_enabled
        == proximal_old.grad_enabled
        == reference.grad_enabled
        == []
    )

    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(
        payload,
        observation,
        drift_current_state=True,
    )
    with pytest.raises(IdentityMismatchError, match="state drifted during replay"):
        materializer.materialize(payload)
    assert current.grad_enabled == [True]
    assert proximal_old.grad_enabled == reference.grad_enabled == []


def test_exact_replay_rejects_behavior_version_and_model_state_mismatches() -> None:
    payload, observation = _payload()
    stale_version = PolicyVersion("smoke", 1, "9" * 64)
    current_binding, proximal_old_binding, reference_binding = _bindings(
        observation.model, current_version=stale_version
    )
    materializer = ExactPolicyReplayMaterializer(
        current_forward=_FakeRecordedForward(current_binding, logit_scale=0.1),
        proximal_old_forward=_FakeRecordedForward(
            proximal_old_binding, logit_scale=0.1
        ),
        reference_forward=_FakeRecordedForward(reference_binding, logit_scale=0.0),
    )
    with pytest.raises(IdentityMismatchError, match="zero-staleness"):
        materializer.materialize(payload)

    with pytest.raises(ValueError, match="must not load LoRA"):
        RecordedPolicyForwardBinding(
            role=ComponentRole.REFERENCE,
            model=observation.model,
            policy_version=REFERENCE_VERSION,
            parameterization=ReplayParameterization.FROZEN_BASE,
            base_weights_sha256=BASE_WEIGHTS_SHA256,
            lora_state_sha256=LORA_STATE_SHA256,
            parameters_frozen=True,
            deterministic_forward=True,
            lora_dropout=0.0,
            model_training=False,
            compute_dtype="float32",
            autocast_enabled=False,
            autocast_dtype=None,
            attention_backend="cpu_fixture_sdpa",
            forward_implementation_sha256=FORWARD_IMPLEMENTATION_SHA256,
        )


def test_one_replay_bundle_cannot_hide_a_batched_sequence() -> None:
    payload, _ = _payload()
    bundle = payload.non_tensor_batch["tgvf_trajectory_replay_bundle"][0]
    store, handle = ObservationStore.from_replay_bundle(bundle)
    replay = store.resolve_replay(handle)

    def repeat_ref(name, ref):
        value = store.resolve_verified(ref)
        repeats = (2,) + (1,) * (value.ndim - 1)
        return store.put_tensor(name, value.repeat(repeats))

    tensors = replay.tensors
    batched_replay = replace(
        replay,
        replay_id="exact-replay-batched-invalid",
        tensors=TrajectoryReplayTensorRefs(
            input_ids=repeat_ref("batched.input_ids", tensors.input_ids),
            position_ids=repeat_ref("batched.position_ids", tensors.position_ids),
            attention_mask=repeat_ref("batched.attention_mask", tensors.attention_mask),
            policy_attention_mask=repeat_ref(
                "batched.policy_attention_mask", tensors.policy_attention_mask
            ),
            reference_attention_mask=repeat_ref(
                "batched.reference_attention_mask",
                tensors.reference_attention_mask,
            ),
            teacher_attention_mask=repeat_ref(
                "batched.teacher_attention_mask", tensors.teacher_attention_mask
            ),
        ),
    )
    batched_bundle = store.export_replay_bundle(store.put_replay(batched_replay))
    with pytest.raises(ReplayMismatchError, match="exactly one sequence"):
        require_single_sequence_replay_bundle(batched_bundle)


def test_exact_replay_rejects_role_mask_drift() -> None:
    mask_payload, observation = _payload(different_reference_mask=True)
    materializer, _, _, _ = _materializer(mask_payload, observation)
    with pytest.raises(
        ReplayMismatchError, match="current/proximal/reference replay state"
    ):
        materializer.materialize(mask_payload)


def test_exact_replay_detects_forward_mutation_and_never_recomputes_observation() -> (
    None
):
    payload, observation = _payload()
    materializer, current, proximal_old, reference = _materializer(
        payload, observation, mutate_current_d=True
    )
    with pytest.raises(ReplayMismatchError, match="current forward modified"):
        materializer.materialize(payload)
    assert current.grad_enabled == [True]
    assert proximal_old.grad_enabled == reference.grad_enabled == []


def test_exact_replay_fails_closed_for_unapproved_sampling_transforms() -> None:
    payload, observation = _payload(non_identity_sampling=True)
    materializer, current, proximal_old, reference = _materializer(payload, observation)
    with pytest.raises(ValueError, match="non-identity vLLM transform parity"):
        materializer.materialize(payload)
    assert (
        current.grad_enabled
        == proximal_old.grad_enabled
        == reference.grad_enabled
        == []
    )
