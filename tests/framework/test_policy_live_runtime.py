from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import torch
import pytest

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity, TokenSpan
from tgvf_rl.environment.agent_loop import SampledPolicyTurn
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
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
    _RemoteTGVFFocusToolRuntime,
    _BoundTGVFVisualQualityRuntimeJudge,
    _build_reward_pipeline,
    _default_metrics_factory,
    _required_success_observation_protocol_id,
    _required_action_boundary_protocol_id,
    _required_server_methods_for_profile,
    _success_observation_contract,
)
from tgvf_rl.judges import (
    TGVFVisualQualityJudgeConfig,
    TGVFVisualQualityJudgeProvider,
    tgvf_visual_quality_prompt_identity,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
)
from tgvf_rl.rewards.context import reward_context_from_trajectory
from tgvf_rl.rewards.schema import AnswerTaskKind
from tgvf_rl.rewards.stage3_shaped import QualityJudgeScore
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

    with pytest.raises(RuntimeError, match="cannot execute"):
        _DisabledNoToolRuntime().execute(object(), object())


def test_live_observation_contract_never_infers_crop_renderer() -> None:
    matched = _success_observation_contract(
        protocol_id=(NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_INSTRUCT,
    )
    assert matched.protocol_id is (
        NativeSuccessObservationProtocolId.DEEPEYES_CROP_MATCHED_V1
    )

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
