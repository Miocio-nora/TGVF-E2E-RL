from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import hashlib
import importlib
from types import SimpleNamespace

import pytest
import torch

import tgvf_rl.framework.verl.compatibility as verl_compatibility
from tgvf_rl.checkpoint import CheckpointCoordinator
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.framework.verl import (
    ACTUAL_RESPONSE_LOGPROBS_FIELD,
    EXACT_OBSERVATION_HANDLES_FIELD,
    LOSSLESS_AGENT_LOOP_MANAGER_FQN,
    LOSSLESS_TRANSFER_QUEUE_AGENT_LOOP_MANAGER_FQN,
    OBJECTIVE_SENTINELS_FIELD,
    TRAJECTORY_PAYLOAD_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
    TRAJECTORY_REPLAY_HANDLE_FIELD,
    FSDP2BridgeConfig,
    LosslessAgentLoopManager,
    LosslessTransferQueueAgentLoopManager,
    RolloutBridgeRecord,
    SDPOTeacherCheckpointContributor,
    TGVF_VLLM_PLUGIN_NAME,
    TORCH211_CANDIDATE_VERL_COMMIT,
    VERL_AGENT_LOOP_RETURN_TRANSPORT,
    VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT,
    VerlAdapter,
    VerlAdapterConfig,
    VerlConfigurationError,
    VerlCompatibilityError,
    VerlDistributionIdentity,
    VerlRuntimeRequirements,
    VerlUnavailableError,
    build_agent_loop_output,
    build_data_proto_payload,
    build_padded_data_proto_payload,
    load_verl_public_api,
    make_objective_sentinels,
    parse_agent_loop_output,
    release_verl_data_proto_sidecars,
    register_project_policy_loss,
    register_sdpo_teacher_checkpoint,
    require_vllm_backend,
    trajectory_to_rollout_bridge,
    to_verl_data_proto,
    validate_data_proto_integrity,
    validate_policy_pilot_v1_verl_grpo_parity,
    validate_verl_config_mapping,
    verl_is_available,
    verify_verl_distribution_identity,
)
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
    TokenSpan,
)
from tgvf_rl.observations.store import (
    TrajectoryReplayHandle,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.policy import (
    POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE,
    PilotSamplingConfig,
    PolicyPilotV1Config,
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


SHA_A = "a" * 64
SHA_B = "b" * 64


def _materialize_dotted_overrides(overrides: dict[str, object]) -> dict[str, object]:
    root: dict[str, object] = {}
    for dotted_path, value in overrides.items():
        parts = dotted_path.split(".")
        current = root
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            assert isinstance(child, dict)
            current = child
        current[parts[-1]] = value
    return root


def _record(
    *,
    suffix: int = 0,
    tool_call_count: int = 1,
    prompt_ids: tuple[int, ...] = (1, 2),
    reward_score: float | None = None,
) -> RolloutBridgeRecord:
    if tool_call_count not in {0, 1, 2}:
        raise ValueError("fixture supports zero, one or two tool calls")
    observation_store, observation_handle = populated_observation_store()
    version = policy_version()
    trajectory_id = TrajectoryIdentity("smoke", f"sample-{suffix}", suffix, "group")
    sampling = SamplingIdentity(
        policy_version=version,
        backend="vllm",
        backend_version="fixture-processed-logprobs",
        seed=7 + suffix,
        rng_state_sha256=SHA0,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.05,
        presence_penalty=0.1,
        frequency_penalty=0.0,
        logit_processors=("fixture_processor",),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )
    behavior_store = BehaviorTraceStore()
    behavior_recorder = VLLMBehaviorRecorder(behavior_store)
    base_observation = observation_store.resolve_record(observation_handle)
    base_observation = replace(
        base_observation,
        observation_id=f"observation-0-{suffix}",
        condition=replace(
            base_observation.condition,
            trajectory_ids=(trajectory_id.canonical_id,),
        ),
    )
    observation_handle = observation_store.put(base_observation)
    observation_handles = [] if tool_call_count == 0 else [observation_handle]
    targets = [] if tool_call_count == 0 else ["red label"]
    if tool_call_count == 2:
        targets.append("lower label")
        second_positions = (8, 9)
        second_observation = replace(
            base_observation,
            observation_id=f"observation-1-{suffix}",
            call_index=1,
            condition=replace(
                base_observation.condition,
                sampled_target_text_sha256=hashlib.sha256(b"lower label").hexdigest(),
                call_indices=(1,),
            ),
            branches=tuple(
                replace(branch, injection_positions=second_positions)
                for branch in base_observation.branches
            ),
            layout=replace(
                base_observation.layout,
                d_positions=second_positions,
                deepstack_injection_positions=tuple(
                    second_positions for _ in base_observation.branches
                ),
            ),
        )
        observation_handles.append(observation_store.put(second_observation))

    if tool_call_count == 0:
        native_token_rows = ()
    elif tool_call_count == 1:
        native_token_rows = (tuple(100 + index for index in range(7)),)
    else:
        native_token_rows = ((100, 101), (102, 103, 104))
    assistant_turns = []
    tool_calls = []
    observations = []
    response_ids: tuple[int, ...] = ()
    if tool_call_count == 0:
        tokens = OwnedTokenSequence(
            (10 + suffix, 11 + suffix, 12 + suffix, 13 + suffix),
            (TokenOwnership.POLICY_SAMPLED,) * 4,
        )
        behavior_handle = behavior_recorder.record(
            trajectory_id=trajectory_id.canonical_id,
            assistant_turn_index=0,
            tokens=tokens,
            actual_sampled_logprobs=(-0.125, -0.5, -1.75, -0.25),
            sampling=sampling,
            behavior_policy=version,
            backend_request_sha256=SHA_A,
            backend_response_sha256=SHA_B,
        )
        assistant_turns.append(
            AssistantTurnRecord(
                turn_index=0,
                raw_text="reason</think>answer",
                tokens=tokens,
                behavior_trace=behavior_handle,
                think_span=TokenSpan(0, 2),
                is_tool_call=False,
            )
        )
        response_ids = tokens.token_ids
    for call_index, (target, handle, native_tokens) in enumerate(
        zip(targets, observation_handles, native_token_rows, strict=True)
    ):
        token_base = 10 + suffix + 10 * call_index
        tokens = OwnedTokenSequence(
            (token_base, token_base + 1, token_base + 2),
            (TokenOwnership.POLICY_SAMPLED,) * 3,
        )
        behavior_handle = behavior_recorder.record(
            trajectory_id=trajectory_id.canonical_id,
            assistant_turn_index=call_index,
            tokens=tokens,
            actual_sampled_logprobs=(
                -0.125 - suffix - call_index,
                -0.5 - call_index,
                -1.75 - suffix - call_index,
            ),
            sampling=replace(sampling, seed=sampling.seed + call_index),
            behavior_policy=version,
            backend_request_sha256=SHA_A,
            backend_response_sha256=SHA_B,
        )
        assistant_turns.append(
            AssistantTurnRecord(
                turn_index=call_index,
                raw_text="reason</think><tool_call>fixture</tool_call>",
                tokens=tokens,
                behavior_trace=behavior_handle,
                think_span=TokenSpan(0, 3),
                is_tool_call=True,
            )
        )
        tool_calls.append(
            ToolCallRecord(
                call_index,
                call_index,
                "tgvf_focus_tool",
                target,
                TokenSpan(1, 3),
                (10, 10 + len(target)),
                f"fixture call {call_index}",
            )
        )
        observations.append(ToolObservationRecord(call_index, handle, native_tokens))
        response_ids += tokens.token_ids + native_tokens

    model = base_observation.model
    trajectory = TrajectoryRecord(
        schema_version="trajectory-v1",
        identity=trajectory_id,
        model=model,
        behavior_policy=version,
        assistant_turns=tuple(assistant_turns),
        tool_calls=tuple(tool_calls),
        observations=tuple(observations),
        final_answer="fixture answer" if tool_call_count == 0 else None,
        stop=(
            TrajectoryStop.DIRECT_ANSWER
            if tool_call_count == 0
            else TrajectoryStop.MAX_TOKENS
        ),
    )
    final_ids = prompt_ids + response_ids
    input_ids = observation_store.put_tensor(
        f"replay.{suffix}.input_ids", torch.tensor([final_ids], dtype=torch.int64)
    )
    sequence_length = len(final_ids)
    replay_position_ids = observation_store.put_tensor(
        f"replay.{suffix}.position_ids",
        torch.arange(sequence_length, dtype=torch.int64).view(1, sequence_length),
    )
    replay_attention = observation_store.put_tensor(
        f"replay.{suffix}.attention_mask",
        torch.ones(1, sequence_length, dtype=torch.bool),
    )
    replay_visibility = observation_store.put_tensor(
        f"replay.{suffix}.visibility",
        torch.ones(1, sequence_length, dtype=torch.bool),
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id=f"replay-{suffix}-{tool_call_count}",
        trajectory_id=trajectory_id.canonical_id,
        model=model,
        behavior_policy=version,
        source_visual=trajectory_source_visual(base_observation),
        observation_handles=tuple(observation_handles),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=replay_position_ids,
            attention_mask=replay_attention,
            policy_attention_mask=replay_visibility,
            reference_attention_mask=replay_visibility,
            teacher_attention_mask=replay_visibility,
        ),
    )
    replay_handle = observation_store.put_replay(replay)
    return trajectory_to_rollout_bridge(
        trajectory,
        validator=TrajectoryValidator(observation_store, behavior_store),
        initial_prompt_token_ids=prompt_ids,
        native_tool_appended_token_ids=native_token_rows,
        replay_handle=replay_handle,
        sentinel_fields=make_objective_sentinels(f"row-{suffix}"),
        extra_fields={"custom_exact_field": ("raw", suffix)},
        reward_score=reward_score,
    )


class _FakeAgentLoopOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeDataProto:
    def __init__(self, batch, non_tensor_batch, meta_info):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return cls(tensors, non_tensors, meta_info)


def test_optional_import_and_public_symbol_resolution_without_installed_verl() -> None:
    assert isinstance(verl_is_available(), bool)

    def unavailable(_: str):
        raise ModuleNotFoundError("deliberately absent")

    with pytest.raises(VerlUnavailableError, match="compatibility candidate"):
        load_verl_public_api(importer=unavailable)

    modules = {
        "verl.experimental.agent_loop": SimpleNamespace(
            AgentLoopOutput=type("AgentLoopOutput", (), {}),
            AgentLoopManager=type("AgentLoopManager", (), {}),
        ),
        "verl.protocol": SimpleNamespace(DataProto=type("DataProto", (), {})),
        "verl.trainer.ppo.core_algos": SimpleNamespace(
            register_policy_loss=lambda name: name,
            get_policy_loss_fn=lambda name: name,
            compute_grpo_outcome_advantage=lambda *args, **kwargs: None,
            compute_policy_loss_bypass_mode=lambda *args, **kwargs: None,
        ),
        "verl.workers.config": SimpleNamespace(
            FSDPEngineConfig=type("FSDPEngineConfig", (), {})
        ),
        "verl.utils.checkpoint": SimpleNamespace(
            CheckpointHandler=type("CheckpointHandler", (), {})
        ),
    }
    api = load_verl_public_api(importer=modules.__getitem__)
    assert api.agent_loop_output.__name__ == "AgentLoopOutput"
    assert api.data_proto.__name__ == "DataProto"
    assert api.agent_loop_transport == VERL_AGENT_LOOP_RETURN_TRANSPORT


def test_candidate_public_api_selects_transfer_queue_manager() -> None:
    modules = {
        "verl.experimental.agent_loop": SimpleNamespace(
            AgentLoopOutput=type("AgentLoopOutput", (), {}),
        ),
        "verl.trainer.ppo.v1": SimpleNamespace(
            AgentLoopManagerTQ=type("AgentLoopManagerTQ", (), {}),
        ),
        "verl.protocol": SimpleNamespace(DataProto=type("DataProto", (), {})),
        "verl.trainer.ppo.core_algos": SimpleNamespace(
            register_policy_loss=lambda name: name,
            get_policy_loss_fn=lambda name: name,
            compute_grpo_outcome_advantage=lambda *args, **kwargs: None,
            compute_policy_loss_bypass_mode=lambda *args, **kwargs: None,
        ),
        "verl.workers.config": SimpleNamespace(
            FSDPEngineConfig=type("FSDPEngineConfig", (), {})
        ),
        "verl.utils.checkpoint": SimpleNamespace(
            CheckpointHandler=type("CheckpointHandler", (), {})
        ),
    }

    api = load_verl_public_api(
        importer=modules.__getitem__,
        expected_commit=TORCH211_CANDIDATE_VERL_COMMIT,
    )

    assert api.agent_loop_manager.__name__ == "AgentLoopManagerTQ"
    assert api.agent_loop_transport == VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT


def test_verl_distribution_identity_requires_an_explicit_audited_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = "638b8ff84f279e054982f1f4633a546f3c6ced68"
    identity = VerlDistributionIdentity(
        package_version="0.9.0.dev0",
        source_url="https://github.com/verl-project/verl.git",
        commit=candidate,
        source_kind="vcs",
        source_clean=None,
    )
    monkeypatch.setattr(
        verl_compatibility,
        "installed_verl_distribution_identity",
        lambda: identity,
    )
    assert verify_verl_distribution_identity(expected_commit=candidate) is identity
    with pytest.raises(VerlCompatibilityError, match="differs"):
        verify_verl_distribution_identity()
    with pytest.raises(VerlCompatibilityError, match="not an accepted"):
        verify_verl_distribution_identity(expected_commit="f" * 40)


def test_runtime_requirements_are_vllm_only_and_fsdp2_strict() -> None:
    requirements = VerlRuntimeRequirements()
    assert requirements.rollout_backend == "vllm"
    assert requirements.calculate_log_probs is True
    assert requirements.fsdp2.actor_strategy == "fsdp2"
    assert (
        VerlRuntimeRequirements(verl_commit=TORCH211_CANDIDATE_VERL_COMMIT).verl_commit
        == TORCH211_CANDIDATE_VERL_COMMIT
    )

    for backend in ("sglang", "hf", "VLLM-extra"):
        with pytest.raises(VerlConfigurationError, match="vLLM"):
            require_vllm_backend(backend)
    with pytest.raises(VerlConfigurationError, match="response_logprobs"):
        VerlRuntimeRequirements(calculate_log_probs=False)
    with pytest.raises(VerlConfigurationError, match="synchronous"):
        FSDP2BridgeConfig(checkpoint_async_save=True)
    with pytest.raises(VerlConfigurationError, match="two ranks"):
        FSDP2BridgeConfig(world_size=1, fsdp_size=1)


def test_concrete_verl_config_mapping_checks_public_paths() -> None:
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "calculate_log_probs": True,
                "logprobs_mode": "processed_logprobs",
                "enable_prefix_caching": False,
                "engine_kwargs": {
                    "vllm": {
                        "enable_mm_embeds": True,
                        "mm_processor_cache_gb": 0,
                        "mm_encoder_attn_backend": "TORCH_SDPA",
                        "hf_overrides": {
                            "architectures": ["TGVFQwen3VLForConditionalGeneration"]
                        },
                    }
                },
                "limit_images": 3,
            },
            "model": {"lora": {"dropout": 0.0}},
            "actor": {
                "strategy": "fsdp2",
                "fsdp_config": {"fsdp_size": 2, "full_determinism": True},
                "checkpoint": {
                    "async_save": False,
                    "strict": True,
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                },
            },
            "ref": {
                "strategy": "fsdp2",
                "fsdp_config": {"fsdp_size": 2, "full_determinism": True},
            },
        },
        "trainer": {"use_v1": False, "v1": {"trainer_mode": "sync"}},
    }
    validate_verl_config_mapping(config)
    config["actor_rollout_ref"]["rollout"]["calculate_log_probs"] = False
    with pytest.raises(VerlConfigurationError, match="calculate_log_probs"):
        validate_verl_config_mapping(config)
    config["actor_rollout_ref"]["rollout"]["calculate_log_probs"] = True
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"][
        "mm_encoder_attn_backend"
    ] = "FLASH_ATTN"
    with pytest.raises(VerlConfigurationError, match="TORCH_SDPA"):
        validate_verl_config_mapping(config)


def test_agent_loop_output_preserves_actual_values_handles_and_extra_fields() -> None:
    record = _record()
    metrics = object()
    output = build_agent_loop_output(
        record,
        metrics=metrics,
        agent_loop_output_cls=_FakeAgentLoopOutput,
    )

    assert tuple(output.response_logprobs) == record.response_logprobs
    assert (
        output.extra_fields[ACTUAL_RESPONSE_LOGPROBS_FIELD] == record.response_logprobs
    )
    assert (
        output.extra_fields[EXACT_OBSERVATION_HANDLES_FIELD]
        == record.exact_observation_handles
    )
    assert output.extra_fields[OBJECTIVE_SENTINELS_FIELD] == dict(
        record.sentinel_fields
    )
    assert output.extra_fields["custom_exact_field"] == ("raw", 0)
    parsed = parse_agent_loop_output(output)
    assert parsed == record
    assert record.response_mask == (1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    assert record.response_logprobs[3:] == (0.0,) * 7
    assert (
        record.behavior_trace_records[
            0
        ].behavior.sampling.has_identity_sampling_transforms
        is False
    )

    output.response_logprobs[0] = -999.0
    with pytest.raises(ValueError, match="changed"):
        parse_agent_loop_output(output)


def test_agent_loop_transport_rejects_unbound_replay_or_absent_trajectory() -> None:
    record = _record()
    output = build_agent_loop_output(
        record,
        metrics=object(),
        agent_loop_output_cls=_FakeAgentLoopOutput,
    )
    output.extra_fields[TRAJECTORY_REPLAY_HANDLE_FIELD] = TrajectoryReplayHandle(
        "forged-replay", SHA_A
    )
    with pytest.raises(ValueError, match="provenance"):
        parse_agent_loop_output(output)

    output = build_agent_loop_output(
        record,
        metrics=object(),
        agent_loop_output_cls=_FakeAgentLoopOutput,
    )
    output.extra_fields[TRAJECTORY_PAYLOAD_FIELD] = None
    with pytest.raises(TypeError, match="complete TrajectoryRecord"):
        parse_agent_loop_output(output)


def test_unique_converter_interleaves_two_calls_in_global_token_order() -> None:
    record = _record(tool_call_count=2)
    first_turn, second_turn = record.trajectory_payload.assistant_turns
    first_observation, second_observation = record.trajectory_payload.observations
    assert record.response_ids == (
        first_turn.tokens.token_ids
        + first_observation.template_token_ids
        + second_turn.tokens.token_ids
        + second_observation.template_token_ids
    )
    assert record.response_mask == (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0)
    assert record.response_logprobs[3:5] == (0.0, 0.0)
    assert record.response_logprobs[8:] == (0.0, 0.0, 0.0)


def test_dataproto_roundtrip_keeps_exact_sidecars_and_detects_tensor_overwrite() -> (
    None
):
    records = (_record(), _record(suffix=1))
    payload = build_data_proto_payload(records)
    data = to_verl_data_proto(payload, data_proto_cls=_FakeDataProto)
    view = validate_data_proto_integrity(data)

    assert view.observation_handles == tuple(
        row.exact_observation_handles for row in records
    )
    assert view.actual_response_logprobs == tuple(
        row.response_logprobs for row in records
    )
    assert view.trajectory_payloads == tuple(row.trajectory_payload for row in records)
    assert [dict(item) for item in view.objective_sentinels] == [
        dict(row.sentinel_fields) for row in records
    ]
    assert tuple(data.non_tensor_batch["custom_exact_field"]) == (
        ("raw", 0),
        ("raw", 1),
    )

    data.batch["rollout_log_probs"][0, 0] += 0.5
    with pytest.raises(ValueError, match="differs"):
        validate_data_proto_integrity(data)


def test_worker_local_dataproto_sidecar_release_is_explicit_and_idempotent() -> None:
    payload = build_data_proto_payload((_record(),))
    driver_data = to_verl_data_proto(payload, data_proto_cls=_FakeDataProto)
    worker_data = _FakeDataProto(
        driver_data.batch,
        dict(driver_data.non_tensor_batch),
        dict(driver_data.meta_info),
    )
    worker_data.non_tensor_batch["worker_metric"] = "preserve"
    expected_count = len(payload.non_tensor_batch)

    assert release_verl_data_proto_sidecars(worker_data) == expected_count
    assert release_verl_data_proto_sidecars(worker_data) == 0
    assert worker_data.non_tensor_batch == {"worker_metric": "preserve"}
    assert driver_data.non_tensor_batch

    assert payload.release_sidecars()
    assert not driver_data.non_tensor_batch


def test_policy_pilot_adapter_rejects_ownerless_dataproto_convenience() -> None:
    pilot = PolicyPilotV1Config(sampling=PilotSamplingConfig(min_p=0.0))
    adapter = VerlAdapter(
        VerlAdapterConfig(policy_pilot=pilot),
        public_api=SimpleNamespace(data_proto=_FakeDataProto),
    )

    with pytest.raises(RuntimeError, match="retained DataProtoPayload"):
        adapter.build_data_proto((_record(),))


def test_explicit_variable_padding_preserves_direct_one_and_multi_call_rows() -> None:
    pad_token_id = 99
    records = (
        _record(suffix=0, tool_call_count=0, prompt_ids=(1,)),
        _record(suffix=1, tool_call_count=1, prompt_ids=(1, 2)),
        _record(suffix=2, tool_call_count=2, prompt_ids=(1, 2, 3, 4)),
    )
    payload = build_padded_data_proto_payload(records, pad_token_id=pad_token_id)
    data = to_verl_data_proto(payload, data_proto_cls=_FakeDataProto)
    view = validate_data_proto_integrity(data)

    assert data.batch["prompts"].shape == (3, 4)
    assert data.batch["responses"].shape == (3, 11)
    assert tuple(data.batch["prompts"][0].tolist()) == (99, 99, 99, 1)
    assert tuple(data.batch["prompts"][1].tolist()) == (99, 99, 1, 2)
    assert tuple(data.batch["prompts"][2].tolist()) == (1, 2, 3, 4)
    for index, record in enumerate(records):
        response_length = len(record.response_ids)
        assert tuple(data.batch["responses"][index, :response_length].tolist()) == (
            record.response_ids
        )
        assert bool(
            (data.batch["responses"][index, response_length:] == pad_token_id).all()
        )
        assert (
            tuple(data.batch["response_mask"][index, :response_length].tolist())
            == record.response_mask
        )
        assert bool((data.batch["response_mask"][index, response_length:] == 0).all())
        assert bool(
            (data.batch["rollout_log_probs"][index, response_length:] == 0).all()
        )
        assert view.actual_response_logprobs[index] == record.response_logprobs
        assert view.replay_bundles[index] is record.replay_bundle

    assert view.pad_token_id == pad_token_id
    assert view.prompt_token_ownership[0] == (
        TokenOwnership.PADDING,
        TokenOwnership.PADDING,
        TokenOwnership.PADDING,
        TokenOwnership.TEMPLATE,
    )
    assert view.response_token_ownership[0][4:] == (TokenOwnership.PADDING,) * 7
    assert all(
        owner.policy_loss_mask == 0 and not owner.requires_behavior_logprob
        for ownership in (
            view.prompt_token_ownership,
            view.response_token_ownership,
        )
        for row in ownership
        for owner in row
        if owner is TokenOwnership.PADDING
    )


def test_variable_padding_is_fail_closed_and_rejects_integrity_tampering() -> None:
    records = (
        _record(tool_call_count=0, prompt_ids=(1,)),
        _record(suffix=1, tool_call_count=2, prompt_ids=(1, 2, 3)),
    )
    with pytest.raises(TypeError, match="pad_token_id"):
        build_padded_data_proto_payload(records)  # type: ignore[call-arg]
    for invalid in (None, True, -1):
        with pytest.raises(ValueError, match="pad_token_id"):
            build_padded_data_proto_payload(  # type: ignore[arg-type]
                records, pad_token_id=invalid
            )

    data = to_verl_data_proto(
        build_padded_data_proto_payload(records, pad_token_id=99),
        data_proto_cls=_FakeDataProto,
    )
    data.batch["prompts"][0, 0] = 98
    with pytest.raises(ValueError, match="left prompt padding"):
        validate_data_proto_integrity(data)

    data = to_verl_data_proto(
        build_padded_data_proto_payload(records, pad_token_id=99),
        data_proto_cls=_FakeDataProto,
    )
    ownership = list(data.non_tensor_batch["tgvf_batched_response_token_ownership"][0])
    ownership[-1] = TokenOwnership.TOOL_OBSERVATION.value
    data.non_tensor_batch["tgvf_batched_response_token_ownership"][0] = tuple(ownership)
    with pytest.raises(ValueError, match="response token ownership"):
        validate_data_proto_integrity(data)

    data = to_verl_data_proto(
        build_padded_data_proto_payload(records, pad_token_id=99),
        data_proto_cls=_FakeDataProto,
    )
    exact_length = len(records[0].response_ids)
    data.batch["rollout_log_probs"][0, exact_length] = -0.5
    with pytest.raises(ValueError, match="padded rollout_log_probs"):
        validate_data_proto_integrity(data)


def test_dataproto_rejects_replay_bundle_tensor_mutation() -> None:
    data = to_verl_data_proto(
        build_data_proto_payload((_record(),)), data_proto_cls=_FakeDataProto
    )
    bundle = data.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD][0]
    bundle.tensor_payloads[0].tensor.zero_()

    with pytest.raises(ReplayMismatchError, match="replay tensor payload"):
        validate_data_proto_integrity(data)


def test_public_policy_loss_hook_matches_live_rollout_is_weights_keyword() -> None:
    captured = {}

    def project_loss(call):
        captured["call"] = call
        selected = call.response_mask.bool()
        return call.log_prob[selected].sum(), {"selected": int(selected.sum())}

    registered = {}

    def registrar(name):
        def decorate(function):
            registered[name] = function
            return function

        return decorate

    wrapped = register_project_policy_loss(
        "tgvf_test_exact_transport",
        project_loss,
        registrar=registrar,
    )
    assert registered["tgvf_test_exact_transport"] is wrapped

    current = torch.tensor([[-0.2, -0.3]], requires_grad=True)
    is_weights = torch.tensor([[0.8, 1.2]])
    loss, metrics = wrapped(
        old_log_prob=torch.tensor([[-0.4, -0.5]]),
        log_prob=current,
        advantages=torch.tensor([[1.0, 2.0]]),
        response_mask=torch.tensor([[1, 1]]),
        loss_agg_mode="token-mean",
        config={"identity": "unchanged"},
        rollout_is_weights=is_weights,
    )
    assert loss.requires_grad
    assert metrics == {"selected": 2}
    assert captured["call"].rollout_is_weights is is_weights

    loss_without_correction, _ = wrapped(
        old_log_prob=torch.tensor([[-0.4, -0.5]]),
        log_prob=current,
        advantages=torch.tensor([[1.0, 2.0]]),
        response_mask=torch.tensor([[1, 1]]),
        loss_agg_mode="token-mean",
        config={},
        rollout_is_weights=None,
    )
    assert loss_without_correction.requires_grad

    with pytest.raises(TypeError, match="rollout_log_probs"):
        wrapped(
            old_log_prob=torch.tensor([[-0.4, -0.5]]),
            log_prob=current,
            advantages=torch.tensor([[1.0, 2.0]]),
            response_mask=torch.tensor([[1, 1]]),
            loss_agg_mode="token-mean",
            config={},
            rollout_log_probs=torch.tensor([[-1.2, -1.3]]),
        )


@dataclass
class _TeacherState:
    value: torch.Tensor
    update_count: int

    def state_dict(self):
        return {"value": self.value.clone(), "update_count": self.update_count}

    def load_state_dict(self, state):
        self.value = state["value"].clone()
        self.update_count = state["update_count"]


def test_sdpo_teacher_is_a_separate_checkpoint_contributor() -> None:
    teacher = _TeacherState(torch.tensor([1.0, 2.0]), 7)
    contributor = SDPOTeacherCheckpointContributor(teacher)
    saved = contributor.checkpoint_state()
    teacher.value.add_(10)
    teacher.update_count = 8
    contributor.restore_checkpoint_state(saved)
    assert torch.equal(teacher.value, torch.tensor([1.0, 2.0]))
    assert teacher.update_count == 7

    coordinator = CheckpointCoordinator()
    registered = register_sdpo_teacher_checkpoint(coordinator, teacher)
    assert registered.checkpoint_name == "sdpo_teacher_state"
    with pytest.raises(ValueError, match="duplicate"):
        coordinator.register(registered)


def test_custom_manager_uses_composition_and_validates_delegate_dataproto() -> None:
    data = to_verl_data_proto(
        build_data_proto_payload((_record(),)), data_proto_cls=_FakeDataProto
    )

    class Delegate:
        def generate_sequences(self, prompts):
            assert prompts == "prompts"
            return data

    class PublicManager:
        @classmethod
        def create(cls, *args, **kwargs):
            return Delegate()

    api = SimpleNamespace(
        agent_loop_manager=PublicManager,
        agent_loop_transport=VERL_AGENT_LOOP_RETURN_TRANSPORT,
    )
    manager = LosslessAgentLoopManager.create(_public_api=api)
    assert type(manager) is LosslessAgentLoopManager
    assert manager.generate_sequences("prompts") is data


def test_candidate_manager_preserves_transfer_queue_dispatch_semantics() -> None:
    class Delegate:
        def generate_sequences(self, prompts):
            assert prompts == "tensordict-prompts"
            return None

    class PublicManager:
        @classmethod
        def create(cls, *args, **kwargs):
            return Delegate()

    api = SimpleNamespace(
        agent_loop_manager=PublicManager,
        agent_loop_transport=VERL_AGENT_LOOP_TRANSFER_QUEUE_TRANSPORT,
    )
    manager = LosslessTransferQueueAgentLoopManager.create(_public_api=api)

    assert type(manager) is LosslessTransferQueueAgentLoopManager
    assert manager.generate_sequences("tensordict-prompts") is None

    with pytest.raises(VerlCompatibilityError, match="transport differs"):
        LosslessAgentLoopManager.create(_public_api=api)


def test_adapter_config_exposes_only_accepted_public_overrides() -> None:
    config = VerlAdapterConfig(max_tool_calls=2)
    overrides = config.public_config_overrides()
    assert overrides["actor_rollout_ref.rollout.name"] == "vllm"
    assert overrides["actor_rollout_ref.rollout.calculate_log_probs"] is True
    assert overrides["actor_rollout_ref.rollout.full_determinism"] is True
    assert overrides["actor_rollout_ref.actor.strategy"] == "fsdp2"
    assert "actor_rollout_ref.model.lora.dropout" not in overrides
    assert overrides["actor_rollout_ref.actor.checkpoint.async_save"] is False
    assert overrides["actor_rollout_ref.rollout.limit_images"] == 3
    assert (
        overrides[
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend"
        ]
        == "TORCH_SDPA"
    )
    assert overrides["actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"] == {
        "architectures": ["TGVFQwen3VLForConditionalGeneration"]
    }
    assert config.required_environment() == {
        "VLLM_PLUGINS": TGVF_VLLM_PLUGIN_NAME,
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }
    config.validate_runtime_environment(
        {
            "VLLM_PLUGINS": TGVF_VLLM_PLUGIN_NAME,
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        }
    )
    with pytest.raises(ValueError, match="TRITON_ATTN"):
        config.validate_runtime_environment(
            {
                "VLLM_PLUGINS": TGVF_VLLM_PLUGIN_NAME,
                "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            }
        )
    assert (
        overrides["actor_rollout_ref.rollout.agent.agent_loop_manager_class"]
        == LOSSLESS_AGENT_LOOP_MANAGER_FQN
    )


def test_policy_pilot_uses_real_e003_lora_and_optimizer_fields() -> None:
    pilot = PolicyPilotV1Config(sampling=PilotSamplingConfig(min_p=0.0))
    config = VerlAdapterConfig(policy_pilot=pilot)
    overrides = dict(config.public_config_overrides())

    assert config.max_tool_calls == 4
    assert overrides["actor_rollout_ref.rollout.limit_images"] == 5
    assert overrides["actor_rollout_ref.rollout.n"] == 8
    assert overrides["actor_rollout_ref.rollout.do_sample"] is True
    assert overrides["actor_rollout_ref.rollout.response_length"] == 8192
    assert overrides["data.max_response_length"] == 8192
    assert overrides["data.mm_processor_kwargs.max_pixels"] == 262144
    assert overrides["actor_rollout_ref.model.lora_rank"] == 64
    assert overrides["actor_rollout_ref.model.lora_alpha"] == 64
    assert overrides["actor_rollout_ref.model.target_modules"].startswith(
        "^model\\.language_model"
    )
    assert overrides["actor_rollout_ref.model.exclude_modules"] is None
    assert (
        overrides["actor_rollout_ref.model.external_lib"]
        == POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE
    )
    assert "actor_rollout_ref.model.lora.dropout" not in overrides
    assert overrides["actor_rollout_ref.actor.optim.lr"] == 1.0e-5
    assert overrides["actor_rollout_ref.actor.optim.clip_grad"] == 1.0
    assert overrides["actor_rollout_ref.actor.ppo_epochs"] == 1
    assert overrides["actor_rollout_ref.actor.clip_ratio_c"] == 3.0
    assert overrides["actor_rollout_ref.actor.policy_loss.loss_mode"] == "bypass_mode"
    assert overrides["algorithm.rollout_correction.bypass_mode"] is True
    assert overrides["algorithm.rollout_correction.loss_type"] == "ppo_clip"
    assert overrides["algorithm.rollout_correction.rollout_is"] is None
    assert overrides["algorithm.rollout_correction.rollout_rs"] is None
    assert overrides["algorithm.rollout_correction.rollout_is_batch_normalize"] is False
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] is False
    assert overrides["actor_rollout_ref.actor.kl_loss_coef"] == 0.0
    assert overrides["actor_rollout_ref.actor.entropy_coeff"] == 0.0
    assert overrides["algorithm.use_kl_in_reward"] is False
    assert overrides["algorithm.filter_groups"]["enable"] is False
    assert overrides["actor_rollout_ref.rollout.over_sample_rate"] == 0.0
    assert "actor_rollout_ref.rollout.min_p" not in overrides
    assert "actor_rollout_ref.rollout.presence_penalty" not in overrides
    assert "actor_rollout_ref.rollout.frequency_penalty" not in overrides

    concrete = _materialize_dotted_overrides(overrides)
    validate_verl_config_mapping(concrete, expected_policy_pilot=pilot)
    concrete["actor_rollout_ref"]["model"]["lora_rank"] = 32
    with pytest.raises(VerlConfigurationError, match="Policy Pilot v1"):
        validate_verl_config_mapping(concrete, expected_policy_pilot=pilot)


@pytest.mark.skipif(not verl_is_available(), reason="pinned veRL is not installed")
def test_pinned_verl_bypass_grpo_fails_exact_unclamped_pilot_oracle() -> None:
    api = load_verl_public_api()
    with pytest.raises(
        VerlCompatibilityError,
        match=r"loss_error=.*gradient_error=",
    ):
        validate_policy_pilot_v1_verl_grpo_parity(
            api,
            policy_loss=api.compute_policy_loss_bypass_mode,
        )


@pytest.mark.skipif(not verl_is_available(), reason="pinned veRL is not installed")
def test_external_module_registers_exact_bypass_loss_and_matches_oracle() -> None:
    api = load_verl_public_api()
    module = importlib.import_module(POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE)
    importlib.reload(module)

    registered = api.get_policy_loss_fn("bypass_mode")
    assert registered.__module__ == POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE
    report = validate_policy_pilot_v1_verl_grpo_parity(api)
    assert report.selected_policy_token_count == 40
    assert report.maximum_advantage_absolute_error <= 1.0e-12
    assert report.loss_absolute_error <= 1.0e-12
    assert report.maximum_gradient_absolute_error <= 1.0e-12


def test_candidate_adapter_selects_v1_transfer_queue_and_no_sleep() -> None:
    runtime = VerlRuntimeRequirements(verl_commit=TORCH211_CANDIDATE_VERL_COMMIT)
    config = VerlAdapterConfig(runtime=runtime, max_tool_calls=2)
    overrides = config.public_config_overrides()

    assert config.agent_loop_manager_fqn == (
        LOSSLESS_TRANSFER_QUEUE_AGENT_LOOP_MANAGER_FQN
    )
    assert overrides["trainer.use_v1"] is True
    assert overrides["actor_rollout_ref.rollout.free_cache_engine"] is False
    assert overrides["actor_rollout_ref.rollout.enable_sleep_mode"] is False
    assert overrides["actor_rollout_ref.rollout.checkpoint_engine.backend"] == "naive"

    concrete = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "calculate_log_probs": True,
                "logprobs_mode": "processed_logprobs",
                "enable_prefix_caching": False,
                "free_cache_engine": False,
                "enable_sleep_mode": False,
                "checkpoint_engine": {"backend": "naive"},
                "engine_kwargs": {
                    "vllm": {
                        "enable_mm_embeds": True,
                        "mm_processor_cache_gb": 0,
                        "mm_encoder_attn_backend": "TORCH_SDPA",
                        "hf_overrides": {
                            "architectures": ["TGVFQwen3VLForConditionalGeneration"]
                        },
                    }
                },
                "limit_images": 3,
            },
            "model": {"lora": {"dropout": 0.0}},
            "actor": {
                "strategy": "fsdp2",
                "fsdp_config": {"fsdp_size": 2, "full_determinism": True},
                "checkpoint": {
                    "async_save": False,
                    "strict": True,
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                },
            },
            "ref": {
                "strategy": "fsdp2",
                "fsdp_config": {"fsdp_size": 2, "full_determinism": True},
            },
        },
        "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
    }
    validate_verl_config_mapping(
        concrete,
        expected_verl_commit=TORCH211_CANDIDATE_VERL_COMMIT,
    )
    broken = deepcopy(concrete)
    broken["actor_rollout_ref"]["rollout"]["enable_sleep_mode"] = True
    with pytest.raises(VerlConfigurationError, match="enable_sleep_mode=false"):
        validate_verl_config_mapping(
            broken,
            expected_verl_commit=TORCH211_CANDIDATE_VERL_COMMIT,
        )
