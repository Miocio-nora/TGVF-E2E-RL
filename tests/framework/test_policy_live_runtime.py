from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import torch
import pytest

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.data import PolicyTeacherQuarterMixRuntimeBinding
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.environment.focus_tool import (
    PrecomputedTGVFObservationPayload,
    SourceVisualTensorBundle,
)
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX,
    QWEN_NATIVE_RESPONSE_SUFFIX,
    QwenNativeToolObservationAppender,
)
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)
from tgvf_rl.protocol.action_boundary import NativeActionBoundaryProtocolId
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.verl.policy_live_runtime import (
    Qwen3PolicyE2ELiveRuntimeBuilder,
    _DisabledNoToolRuntime,
    _RemoteAtomicCropTGVFToolRuntime,
    _RemoteTGVFFocusToolRuntime,
    _BoundTGVFVisualQualityRuntimeJudge,
    _build_reward_pipeline,
    _build_stage3_reward_runtime,
    _default_metrics_factory,
    _required_success_observation_protocol_id,
    _required_action_boundary_protocol_id,
    _required_server_methods_for_profile,
    _sample_uses_no_tool_runtime,
    _success_observation_contract,
    _validate_sample_fields,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFCropMaterializationResult,
    preprocessed_visual_identity_sha256,
)
from tgvf_rl.judges import (
    TGVFVisualQualityJudgeConfig,
    TGVFVisualQualityJudgeProvider,
    tgvf_visual_quality_prompt_identity,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
)
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.rewards.context import reward_context_from_trajectory
from tgvf_rl.rewards.schema import AnswerTaskKind
from tgvf_rl.rewards.stage3_shaped import QualityJudgeScore
from tgvf_rl.representation.adapter import TGVFAdapterMetadata
from tgvf_rl.representation.deepstack import DDeepStackPayload
from tgvf_rl.trajectories.schema import TrajectoryStop
from tests.framework.test_verl_bridges import _record
from tgvf_rl.observations.store import ObservationHandle, ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.schema import (
    NativeToolCapabilityProfile,
    TokenByteSpan,
)


SHA = "7" * 64
BRANCH_LAYERS = (8, 16, 24)


def test_default_live_metrics_factory_uses_the_pinned_verl_public_model() -> None:
    pytest.importorskip(
        "verl.experimental.agent_loop.agent_loop",
        reason="live metrics identity requires the optional pinned veRL",
    )
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    assert isinstance(_default_metrics_factory(object(), object()), AgentLoopMetrics)


def test_live_builder_has_no_agent_loop_model_loader_surface() -> None:
    signature = inspect.signature(Qwen3PolicyE2ELiveRuntimeBuilder.__init__)

    assert "model_loader" not in signature.parameters
    assert "success_observation_protocol_id" not in signature.parameters
    assert "action_boundary_protocol_id" not in signature.parameters


def test_no_tool_live_profile_requires_no_tool_rpc_and_runtime_fails_closed() -> None:
    assert _required_server_methods_for_profile(
        NativeToolCapabilityProfile.NO_TOOL
    ) == ("materialize_source", "generate")
    assert _required_server_methods_for_profile(
        NativeToolCapabilityProfile.CROP_ONLY
    ) == ("materialize_source", "generate", "materialize_crop")
    assert _required_server_methods_for_profile(
        NativeToolCapabilityProfile.TGVF_ONLY
    ) == ("materialize_source", "generate", "materialize_focus")
    assert _required_server_methods_for_profile(
        NativeToolCapabilityProfile.CROP_TGVF
    ) == ("materialize_source", "generate", "materialize_crop_tgvf")

    with pytest.raises(RuntimeError, match="cannot execute"):
        _DisabledNoToolRuntime().execute(object(), object())


def test_tfree_stage3_runtime_builds_without_utility_or_visual_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_identity = ArtifactIdentity("test", "answer", "v1", "1" * 64)
    answer_verifier = SimpleNamespace(verify=lambda _context: None)
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime._build_rule_first_answer_verifier",
        lambda _config: (answer_identity, answer_verifier),
    )

    def reject_visual_load(*_args, **_kwargs):
        raise AssertionError("disabled visual judge must not be loaded")

    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime.load_tgvf_visual_quality_judge",
        reject_visual_load,
    )
    config = SimpleNamespace(
        identity_sha256="2" * 64,
        reward=SimpleNamespace(
            profile="stage3-shaped-v1",
            answer_weight=None,
            format_weight=None,
            conditional_tool_weight=None,
            answer_verifier="fixture",
            answer_verifier_sha256="3" * 64,
            judge_config_sha256="4" * 64,
            tool_utility=None,
            tool_utility_reward_enabled=False,
            focus_reward_enabled=False,
            grounding_reward_enabled=False,
            visual_quality_judge_mode="disabled",
            visual_quality_judge_config_path=None,
            visual_quality_judge_config_sha256=None,
            visual_quality_judge_identity=None,
            answer_reward_scale=2.0,
            repeated_call_penalty=0.05,
            protocol_error_penalty=2.0,
        ),
    )

    spec, verifier, visual_provider = _build_stage3_reward_runtime(config)

    assert verifier is answer_verifier
    assert visual_provider is None
    assert spec.tool_utility_reward_enabled is False
    assert spec.visual_quality_enabled is False
    assert spec.answer_reward_scale == 2.0
    assert spec.repeated_call_penalty == 0.05
    assert spec.protocol_error_penalty == 2.0
    assert spec.tool_utility_sidecar_sha256 is None
    assert spec.visual_judge_identity is None


def _teacher25_runtime_binding() -> PolicyTeacherQuarterMixRuntimeBinding:
    return PolicyTeacherQuarterMixRuntimeBinding(
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        schedule_seed=42,
        expected_sample_count=20_480,
    )


def test_teacher25_thinklite_alone_uses_explicit_no_tool_runtime() -> None:
    binding = _teacher25_runtime_binding()

    assert _sample_uses_no_tool_runtime(
        NativeToolCapabilityProfile.TGVF_ONLY,
        "thinklite",
        runtime_binding=binding,
    )
    assert not _sample_uses_no_tool_runtime(
        NativeToolCapabilityProfile.TGVF_ONLY,
        "teacher",
        runtime_binding=binding,
    )
    assert not _sample_uses_no_tool_runtime(
        NativeToolCapabilityProfile.TGVF_ONLY,
        "thinklite",
        runtime_binding=object(),
    )
    assert _sample_uses_no_tool_runtime(
        NativeToolCapabilityProfile.NO_TOOL,
        "teacher",
        runtime_binding=object(),
    )


@pytest.mark.parametrize(
    ("data_source", "task_kind", "prompt_sha256"),
    (
        ("teacher", "open", SHA),
        ("thinklite", "math", THINKLITE_PROMPT_IDENTITY.bundle_sha256),
    ),
)
def test_teacher25_normalized_row_matches_bound_runtime_identity(
    tmp_path: Path,
    data_source: str,
    task_kind: str,
    prompt_sha256: str,
) -> None:
    image_path = (tmp_path / f"{data_source}.png").resolve()
    iteration_sha256 = "3" * 64
    sample_id = f"teacher25:{data_source}"
    record = {
        "sample_id": sample_id,
        "question": "What is shown?",
        "ground_truth": "answer",
        "data_source": data_source,
        "task_kind": task_kind,
        "image": {"path": str(image_path), "sha256": "4" * 64},
    }
    config = SimpleNamespace(
        dataset=SimpleNamespace(
            selected_sample=None,
            runtime_binding=_teacher25_runtime_binding(),
            iteration_identity_sha256=iteration_sha256,
        ),
        protocol=SimpleNamespace(prompt_sha256=SHA),
    )
    fields = {
        "sample_id": sample_id,
        "dataset_iteration_identity_sha256": iteration_sha256,
        "prompt_bundle_sha256": prompt_sha256,
        "source_image_path": str(image_path),
        "source_image_sha256": "4" * 64,
        "question": "What is shown?",
        "data_source": data_source,
        "task_kind": task_kind,
        "reward_model": {"ground_truth": "answer"},
    }

    _validate_sample_fields(
        config,
        sample_id,
        fields,
        sample_index={sample_id: record},
    )

    fields["dataset_iteration_identity_sha256"] = "5" * 64
    with pytest.raises(
        IdentityMismatchError,
        match="dataset_iteration_identity_sha256",
    ):
        _validate_sample_fields(
            config,
            sample_id,
            fields,
            sample_index={sample_id: record},
        )


def test_remote_atomic_runtime_records_bound_crop_conditioned_d(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.environment.test_crop_tgvf_runtime import _fixture

    local, materializer, store, pixels, _embedding, context, parsed = _fixture(
        tmp_path,
        bbox=(0, 250, 800, 1000),
    )
    source_ref = context.trajectory_source_visual.source_pixels
    assert source_ref is not None
    source_state = context.trajectory_source_visual.state
    source_visual = SourceVisualTensorBundle(
        image_sha256=source_ref.address.digest,
        premerge_main=store.resolve_verified(source_state.premerge_main),
        premerge_deepstack=tuple(
            store.resolve_verified(ref) for ref in source_state.premerge_deepstack
        ),
        merged_main=store.resolve_verified(source_state.merged_main),
        merged_deepstack=tuple(
            store.resolve_verified(ref) for ref in source_state.merged_deepstack
        ),
        image_grid_thw=source_state.image_grid_thw,
        spatial_merge_size=source_state.spatial_merge_size,
        decoded_rgb_sha256=source_ref.address.digest,
    )
    crop = pixels[1:4, 0:4, :].contiguous()
    crop_visual = materializer.materialize_crop_tgvf_visual(
        crop,
        parsed_call=parsed,
        call_index=0,
    ).source_visual
    target_count = len(parsed.target_span.token_ids)
    contract = local.loaded_adapter.binding.adapter_contract
    d = torch.full((1, 8), 9.0)
    observation = PrecomputedTGVFObservationPayload(
        main_d=d,
        d_deepstack=DDeepStackPayload(
            branch_layers=contract.deepstack_branch_layers,
            branches=tuple(d.add(index + 1) for index in range(3)),
            projection_identities=contract.deepstack_projection_identities,
        ),
        metadata=TGVFAdapterMetadata(
            branch_layers=contract.deepstack_branch_layers,
            main_projection_identity=contract.main_projection_identity,
            deepstack_projection_identities=(contract.deepstack_projection_identities),
            batched=False,
            batch_size=1,
            target_token_count=target_count,
            pre_merge_visual_token_count=4,
            d_token_count=1,
            condition_provenance=None,
        ),
    )
    hq = torch.arange(target_count * 8, dtype=torch.float32).reshape(target_count, 8)
    calls: list[dict[str, object]] = []

    class Server:
        async def materialize_crop_tgvf(self, **kwargs: object):
            calls.append(dict(kwargs))
            return TGVFCropMaterializationResult(
                source_image_sha256=str(kwargs["source_image_sha256"]),
                crop_sha256=str(kwargs["crop_sha256"]),
                preprocessed_visual_sha256=str(kwargs["preprocessed_visual_sha256"]),
                image_grid_thw=tuple(
                    int(value) for value in kwargs["image_grid_thw"][0].tolist()
                ),
                call_index=int(kwargs["call_index"]),
                model_bbox_2d=tuple(kwargs["model_bbox_2d"]),
                target_start=int(kwargs["target_start"]),
                target_end=int(kwargs["target_end"]),
                target_token_ids=tuple(kwargs["expected_target_token_ids"]),
                provider=str(kwargs["provider"]),
                hq=hq,
                crop_visual=crop_visual,
                observation=observation,
            )

    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime.preprocess_qwen3_rgb",
        lambda **kwargs: (torch.ones((4, 6)), torch.tensor([[1, 2, 2]])),
    )
    config = SimpleNamespace(
        model=context.model,
        representation=SimpleNamespace(
            conditioning=local.loaded_adapter.binding.conditioning,
            artifact=local.loaded_adapter.binding.artifact,
        ),
    )

    async def exercise():
        remote = _RemoteAtomicCropTGVFToolRuntime(
            event_loop=asyncio.get_running_loop(),
            server_client=Server(),
            config=config,
            source_visual=source_visual,
            layout_builder=local.layout_builder,
            observation_store=store,
            execution_ledger=local.execution_ledger,
            contextual_forward_identity=None,
            branch_merger_identities=local.branch_merger_identities,
            crop_processor_identity=local.crop_processor_identity,
            crop_layout_identity=local.crop_layout_identity,
            processor=object(),
            image_max_pixels=1_003_520,
            observation_contract=local.observation_contract,
        )
        first = await asyncio.to_thread(remote.execute, parsed, context)
        second = await asyncio.to_thread(remote.execute, parsed, context)
        return first, second

    first, second = asyncio.run(exercise())

    assert first == second
    assert len(calls) == 1
    assert calls[0]["preprocessed_visual_sha256"] == (
        preprocessed_visual_identity_sha256(
            calls[0]["pixel_values"],
            calls[0]["image_grid_thw"],
        )
    )
    record = store.resolve_record(first)
    torch.testing.assert_close(store.resolve_verified(record.payload.main_d), d)
    assert not torch.equal(
        store.resolve_verified(record.payload.main_d),
        store.resolve_verified(record.crop_visual.source.merged_main),
    )


def test_live_observation_contract_never_infers_crop_renderer() -> None:
    matched = _success_observation_contract(
        protocol_id=(NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    assert matched.protocol_id is (
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    )

    exact_profiles = {
        NativeSuccessObservationProtocolId.NO_TOOL_NO_EXECUTION_V1: (
            NativeToolCapabilityProfile.NO_TOOL
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_TGVF_MATCHED_V1: (
            NativeToolCapabilityProfile.TGVF_ONLY
        ),
        NativeSuccessObservationProtocolId.DEEPEYES_ATOMIC_MATCHED_V1: (
            NativeToolCapabilityProfile.CROP_TGVF
        ),
    }
    for protocol_id, profile in exact_profiles.items():
        contract = _success_observation_contract(
            protocol_id=protocol_id,
            tool_profile=profile,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )
        assert contract.protocol_id is protocol_id
        assert contract.tool_profile is profile

    with pytest.raises(ValueError, match="explicit matched or legacy Crop protocol"):
        _success_observation_contract(
            protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
        )


def test_live_builder_protocol_id_must_come_from_run_identity() -> None:
    with pytest.raises(ValueError, match="requires protocol"):
        _required_success_observation_protocol_id(SimpleNamespace())
    with pytest.raises(ValueError, match="is invalid"):
        _required_success_observation_protocol_id(
            SimpleNamespace(success_observation_protocol_id="matched")
        )
    assert (
        _required_success_observation_protocol_id(
            SimpleNamespace(
                success_observation_protocol_id=(
                    NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1.value
                )
            )
        )
        is NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    )


def test_live_action_boundary_must_be_explicit_and_strict() -> None:
    with pytest.raises(ValueError, match="requires protocol"):
        _required_action_boundary_protocol_id(SimpleNamespace())
    with pytest.raises(ValueError, match="is invalid"):
        _required_action_boundary_protocol_id(
            SimpleNamespace(action_boundary_protocol_id="strict")
        )
    with pytest.raises(ValueError, match="requires strict terminal"):
        _required_action_boundary_protocol_id(
            SimpleNamespace(
                action_boundary_protocol_id=(
                    NativeActionBoundaryProtocolId.LEGACY_ANSWER_OVER_ACTION_V1
                )
            )
        )
    assert (
        _required_action_boundary_protocol_id(
            SimpleNamespace(
                action_boundary_protocol_id=(
                    NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
                )
            )
        )
        is NativeActionBoundaryProtocolId.STRICT_SINGLE_TERMINAL_TOOL_CALL_V2
    )


def test_live_reward_pipeline_binds_configured_named_weight_profile() -> None:
    def config(tool_weight: float, *, deepeyes: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            schema_version=(
                POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA
                if deepeyes
                else "legacy-test-schema"
            ),
            identity_sha256=SHA,
            reward=SimpleNamespace(
                answer_verifier="exact_match",
                answer_verifier_sha256="8" * 64,
                answer_weight=0.8,
                format_weight=0.2,
                conditional_tool_weight=tool_weight,
                judge_config_path=None,
                judge_config_sha256=None,
            ),
            protocol=SimpleNamespace(enabled_tool_names=("tgvf_focus_tool",)),
        )

    legacy = _build_reward_pipeline(config(1.2))
    answer_primary = _build_reward_pipeline(config(0.2))
    deepeyes = _build_reward_pipeline(config(1.2, deepeyes=True))

    assert legacy.spec.weights == (0.8, 0.2, 1.2)
    assert legacy.spec.pipeline_identity.version == "0.8-0.2-1.2"
    assert answer_primary.spec.weights == (0.8, 0.2, 0.2)
    assert answer_primary.spec.pipeline_identity.version == "0.8-0.2-0.2"
    assert (
        answer_primary.spec.pipeline_identity.sha256
        != legacy.spec.pipeline_identity.sha256
    )
    assert deepeyes.spec.deepeyes_source_aware is True
    assert (
        deepeyes.spec.pipeline_identity.sha256 != legacy.spec.pipeline_identity.sha256
    )


def test_live_visual_quality_adapter_consumes_typed_provider_result(
    tmp_path: Path,
) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nlive-runtime-test"
    image_path = (tmp_path / "source.png").resolve()
    image_path.write_bytes(image_bytes)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"focus_score":2,"grounding_score":1}'
                            }
                        }
                    ]
                }
            ).encode()

    def identity(name: str, digit: str) -> ArtifactIdentity:
        return ArtifactIdentity("live-visual-test", name, "v1", digit * 64)

    provider = TGVFVisualQualityJudgeProvider(
        TGVFVisualQualityJudgeConfig(
            base_url="https://judge.invalid/v1",
            model_name="fixture-vision-judge",
            prompt_identity=tgvf_visual_quality_prompt_identity(),
            service_identity=identity("service", "1"),
            model_identity=identity("model", "2"),
            sampling_identity=identity("sampling", "3"),
        ),
        opener=lambda *_args, **_kwargs: Response(),
    )
    trajectory = replace(
        _record(tool_call_count=1).trajectory_payload,
        final_answer="fixture answer",
        stop=TrajectoryStop.FINAL_ANSWER,
    )
    context = reward_context_from_trajectory(
        trajectory,
        question="Which option is correct?",
        expected_answer="fixture answer",
        task_kind=AnswerTaskKind.MULTIPLE_CHOICE,
    )
    adapter = _BoundTGVFVisualQualityRuntimeJudge(
        provider=provider,
        image_path=image_path,
        image_sha256=__import__("hashlib").sha256(image_bytes).hexdigest(),
    )

    result = adapter.judge(
        request=SimpleNamespace(identity=trajectory.identity),
        trajectory=trajectory,
        context=context,
    )

    assert result.focus_score is QualityJudgeScore.PASS
    assert result.grounding_score is QualityJudgeScore.PARTIAL


class _NativeTokenizer:
    _native = {
        "<|vision_start|>": 1,
        "<|image_pad|>": 2,
        "<|vision_end|>": 3,
    }

    def __init__(self, name_or_path: str = "/fixture") -> None:
        self.name_or_path = name_or_path
        self.encoded_texts: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._native[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            for token, token_id in self._native.items():
                if text.startswith(token, cursor):
                    result.append(token_id)
                    cursor += len(token)
                    break
            else:
                result.append(10 + (ord(text[cursor]) % 200))
                cursor += 1
        return result


class _Registrar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def register_tool_turn(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))


def _sampled_focus_call(assistant_dialect: NativeAssistantDialect):
    reasoning = (
        "inspect</think>"
        if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING
        else "<think>inspect</think>"
    )
    text = (
        reasoning + '<tool_call>{"name":"tgvf_focus_tool","arguments":'
        '{"target":"the gauge needle position"}}</tool_call>'
    )
    token_ids = tuple(1000 + index for index in range(len(text)))
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    policy = PolicyVersion("fixture", 0, "1" * 64)
    sampled = SampledPolicyTurn(
        text=text,
        token_ids=token_ids,
        token_byte_spans=spans,
        behavior_logprobs=tuple(-0.1 for _ in token_ids),
        sampling=SamplingIdentity(
            policy_version=policy,
            backend="vllm",
            backend_version="fixture",
            seed=42,
            rng_state_sha256="2" * 64,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        ),
        think_token_span=TokenSpan(0, text.index("</think>") + len("</think>")),
        stop_reason="tool_call_stop",
        backend_request_sha256="3" * 64,
        backend_response_sha256="4" * 64,
        assistant_dialect=assistant_dialect,
    )
    parsed = StrictToolCallParser(enabled_tool_names=("tgvf_focus_tool",)).parse(
        sampled.parser_turn()
    )
    return sampled, parsed


@pytest.mark.parametrize(
    "assistant_dialect",
    (
        NativeAssistantDialect.QWEN3_VL_THINKING,
        NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    ),
)
def test_policy_layout_focus_and_final_expansion_share_one_idempotent_coordinate(
    assistant_dialect: NativeAssistantDialect,
) -> None:
    model_name = (
        "Qwen3-VL-8B-Thinking"
        if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING
        else "Qwen3-VL-8B-Instruct"
    )
    model_path = f"/fixture/{model_name}"
    tokenizer = _NativeTokenizer(model_path)
    model = ModelIdentity("qwen3_vl", model_name, model_path, 256, SHA)
    store = ObservationStore()
    positions = (1, 2, 3, 4)
    main = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    branches = tuple(torch.full((1, 4, 8), float(index)) for index in range(3))
    source_bundle = SourceVisualTensorBundle(
        image_sha256=SHA,
        premerge_main=main,
        premerge_deepstack=branches,
        merged_main=main,
        merged_deepstack=branches,
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=1,
    )
    source = record_trajectory_source_visual(
        trajectory_id="run/sample/0/group",
        source_visual=source_bundle,
        source_positions=positions,
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=(positions,) * 3,
        observation_store=store,
    )

    rope_inputs: list[tuple[int, ...]] = []

    def get_rope_index(*, input_ids, image_grid_thw, **_kwargs):
        rope_inputs.append(tuple(input_ids[0].tolist()))
        sequence = input_ids.shape[-1]
        position_ids = torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1)
        return position_ids, torch.zeros((1, 1), dtype=torch.long)

    builder = Qwen3NativeToolLayoutBuilder(
        tokenizer=tokenizer,
        model_identity=model,
        observation_store=store,
        get_rope_index=get_rope_index,
    )
    initial_ids = (1, 2, 2, 2, 2, 3, 99)

    # The dataset already expanded the source placeholder.  Final replay must
    # prove it, not turn the four-token run into seven tokens.
    final_layout = builder.expand_recorded_visual_sequence(
        initial_ids,
        trajectory_source_visual=source,
        observation_handles=(),
    )
    assert tuple(final_layout.input_ids[0].tolist()) == initial_ids

    sampled, parsed = _sampled_focus_call(assistant_dialect)
    observation_contract = NativeSuccessObservationContract(
        protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
        tool_profile=NativeToolCapabilityProfile.TGVF_ONLY,
        assistant_dialect=assistant_dialect,
    )
    environment_success_text = observation_contract.render(parsed)
    conditioning_ids = initial_ids + parsed.sampled_token_ids
    encoded_before_layout = len(tokenizer.encoded_texts)
    focus_layout = builder.build_focus_from_recorded_prefix(
        conditioning_input_ids=conditioning_ids,
        parsed_call=parsed,
        trajectory_source_visual=source,
        prior_observation_handles=(),
        source_visual=source_bundle,
        environment_success_text=environment_success_text,
    )
    layout_encoded_texts = tokenizer.encoded_texts[encoded_before_layout:]

    assert focus_layout.visual_layout.original_image_positions == positions
    assert len(focus_layout.visual_layout.d_positions) == 4
    assert all(
        branch_positions == focus_layout.visual_layout.d_positions
        for branch_positions in focus_layout.visual_layout.deepstack_injection_positions
    )
    assert rope_inputs[0] == initial_ids
    assert rope_inputs[1][: len(conditioning_ids)] == conditioning_ids
    assert layout_encoded_texts == [environment_success_text]
    expected_suffix = (
        QWEN_NATIVE_RESPONSE_SUFFIX
        if assistant_dialect is NativeAssistantDialect.QWEN3_VL_THINKING
        else QWEN_NATIVE_INSTRUCT_RESPONSE_SUFFIX
    )
    assert environment_success_text.endswith(expected_suffix)

    registrar = _Registrar()
    appender = QwenNativeToolObservationAppender(
        tokenizer=tokenizer,
        registrar=registrar,
        observation_contract=observation_contract,
    )
    _updated, _environment_ids = appender.append(
        initial_ids,
        sampled,
        ObservationHandle("focus-observation", "5" * 64),
        call_index=0,
        parsed_call=parsed,
    )
    assert layout_encoded_texts[0].encode("utf-8") == tokenizer.encoded_texts[
        -1
    ].encode("utf-8")
    assert len(registrar.calls) == 1


def test_remote_focus_runtime_and_layout_require_explicit_observation_contract() -> (
    None
):
    runtime_parameter = inspect.signature(
        _RemoteTGVFFocusToolRuntime.__init__
    ).parameters["observation_contract"]
    layout_parameter = inspect.signature(
        Qwen3NativeToolLayoutBuilder.build_focus_from_recorded_prefix
    ).parameters["environment_success_text"]

    assert runtime_parameter.default is inspect.Parameter.empty
    assert layout_parameter.default is inspect.Parameter.empty
