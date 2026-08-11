from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from dataclasses import replace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.data.tgvf_tool_utility import (
    TGVFToolUtilityLabelBinding,
    TGVFToolUtilityRuntimeBinding,
)
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.agent_loop import ResponseBudgetScope
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.verl.policy_live_runtime import (
    Qwen3PolicyE2ELiveRuntimeBuilder,
    _BoundTGVFVisualQualityRuntimeJudge,
    _Qwen3PolicyTrajectoryComponents,
    _build_reward_pipeline,
    _default_metrics_factory,
    _final_token_materialization,
    _rp66_matched_source_route,
    _rp66_response_budget_controls,
    _trainable_rp66_launch_mode,
)
from tgvf_rl.framework.verl.policy_runtime import PolicyAgentLoopWorkerPlacement
from tgvf_rl.framework.verl.native_deepeyes_runtime import (
    NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS,
)
from tgvf_rl.judges import (
    TGVFVisualQualityJudgeConfig,
    TGVFVisualQualityJudgeProvider,
    tgvf_visual_quality_prompt_identity,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
    POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
    POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
)
from tgvf_rl.rewards.context import reward_context_from_trajectory
from tgvf_rl.rewards.schema import AnswerTaskKind
from tgvf_rl.rewards.stage3_shaped import QualityJudgeScore
from tgvf_rl.trajectories.schema import TrajectoryStop
from tests.framework.test_verl_bridges import _record
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan


SHA = "7" * 64
BRANCH_LAYERS = (8, 16, 24)


def test_default_live_metrics_factory_uses_the_pinned_verl_public_model() -> None:
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    assert isinstance(_default_metrics_factory(object(), object()), AgentLoopMetrics)


def test_live_builder_has_no_agent_loop_model_loader_surface() -> None:
    signature = inspect.signature(Qwen3PolicyE2ELiveRuntimeBuilder.__init__)

    assert "model_loader" not in signature.parameters


def test_live_reward_pipeline_binds_configured_named_weight_profile() -> None:
    def config(
        tool_weight: float,
        *,
        schema_version: str = "legacy-test-schema",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            schema_version=schema_version,
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
    deepeyes = _build_reward_pipeline(
        config(
            1.2,
            schema_version=POLICY_E2E_DEEPEYES_SCALED_CROP_RUN_CONFIG_SCHEMA,
        )
    )
    trainable_rp66 = _build_reward_pipeline(
        config(
            1.2,
            schema_version=POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        )
    )
    frozen_rp66 = _build_reward_pipeline(
        config(
            1.2,
            schema_version=POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        )
    )

    assert legacy.spec.weights == (0.8, 0.2, 1.2)
    assert legacy.spec.pipeline_identity.version == "0.8-0.2-1.2"
    assert answer_primary.spec.weights == (0.8, 0.2, 0.2)
    assert answer_primary.spec.pipeline_identity.version == "0.8-0.2-0.2"
    assert (
        answer_primary.spec.pipeline_identity.sha256
        != legacy.spec.pipeline_identity.sha256
    )
    assert deepeyes.spec.deepeyes_source_aware is True
    assert trainable_rp66.spec.deepeyes_source_aware is True
    assert frozen_rp66.spec.deepeyes_source_aware is True
    assert (
        deepeyes.spec.pipeline_identity.sha256 != legacy.spec.pipeline_identity.sha256
    )


@pytest.mark.parametrize(
    "schema_version",
    (
        POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
    ),
)
@pytest.mark.parametrize("data_source", ("vstar", "arxivqa"))
def test_trainable_rp66_visual_rows_select_matched_observation_renderer(
    schema_version: str,
    data_source: str,
) -> None:
    config = SimpleNamespace(schema_version=schema_version)

    assert _rp66_matched_source_route(config, {"data_source": data_source}) == (
        False,
        True,
    )


@pytest.mark.parametrize(
    "schema_version",
    (
        POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
    ),
)
def test_trainable_rp66_thinklite_rows_are_direct_only(schema_version: str) -> None:
    config = SimpleNamespace(schema_version=schema_version)

    assert _rp66_matched_source_route(config, {"data_source": "thinklite"}) == (
        True,
        False,
    )


def test_trainable_rp66_runtime_rejects_unknown_source() -> None:
    config = SimpleNamespace(
        schema_version=POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA
    )

    with pytest.raises(ValueError, match="unsupported data_source"):
        _rp66_matched_source_route(config, {"data_source": "unknown"})


@pytest.mark.parametrize("launch_mode", ("formal", "smoke"))
def test_matched_visual_rows_use_crop_total_horizon(
    launch_mode: str,
) -> None:
    assert _rp66_response_budget_controls(
        launch_mode=launch_mode,
        direct_only=False,
        matched_visual_observation=True,
    ) == (
        ResponseBudgetScope.TOTAL_RESPONSE,
        NATIVE_DEEPEYES_SINGLE_RESPONSE_MAX_TOKENS,
    )


def test_functional_canary_retains_separate_policy_token_transport() -> None:
    assert _rp66_response_budget_controls(
        launch_mode="canary",
        direct_only=False,
        matched_visual_observation=True,
    ) == (ResponseBudgetScope.POLICY_SAMPLED, None)


def test_matched_thinklite_direct_route_keeps_single_turn_policy_budget() -> None:
    assert _rp66_response_budget_controls(
        launch_mode="formal",
        direct_only=True,
        matched_visual_observation=False,
    ) == (ResponseBudgetScope.POLICY_SAMPLED, None)


@pytest.mark.parametrize(
    "schema_version",
    (
        POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
    ),
)
def test_trainable_rp66_launch_mode_comes_from_live_trainer_config(
    schema_version: str,
) -> None:
    config = SimpleNamespace(schema_version=schema_version)
    trainer = {
        "actor_rollout_ref": {
            "rollout": {"custom": {"launch_mode": "smoke"}}
        }
    }

    assert _trainable_rp66_launch_mode(config, trainer) == "smoke"


def test_rp66_control_v2_uses_official_deepeyes_judge_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge_path = Path("/fixture/deepeyes-judge.json")
    judge_sha256 = "9" * 64
    loaded_service = SimpleNamespace(maximum_concurrency=16)
    calls: list[tuple[Path, str]] = []

    def load_service(path: Path, *, expected_file_sha256: str) -> object:
        calls.append((path, expected_file_sha256))
        return loaded_service

    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime."
        "load_deepeyes_judge_service_config",
        load_service,
    )
    judge_calls: list[tuple[object, int]] = []

    def build_judge(
        service: object, *, local_maximum_concurrency: int
    ) -> tuple[str, object, int]:
        judge_calls.append((service, local_maximum_concurrency))
        return ("official-deepeyes", service, local_maximum_concurrency)

    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime.AsyncDeepEyesOpenRouterJudge",
        build_judge,
    )
    config = SimpleNamespace(
        schema_version=POLICY_E2E_RP66_CONTROL_RUN_CONFIG_SCHEMA,
        reward=SimpleNamespace(
            judge_config_path=judge_path,
            judge_config_sha256=judge_sha256,
        ),
    )

    components = _Qwen3PolicyTrajectoryComponents(
        context=SimpleNamespace(
            config=config,
            placement=PolicyAgentLoopWorkerPlacement(3, 3, 3, 8),
        ),
        layout_builder=object(),
        server_client=object(),
        contextual_forward_identity=None,
        branch_merger_identities=(),
        observation_store=object(),
        behavior_store=object(),
        focus_execution_ledger=object(),
        crop_execution_ledger=object(),
        metrics_factory=lambda *_args: object(),
        agent_loop_output_cls=None,
        sample_index={},
        launch_mode="canary",
    )

    assert calls == [(judge_path, judge_sha256)]
    assert components.official_deepeyes_judge == (
        "official-deepeyes",
        loaded_service,
        2,
    )
    assert judge_calls == [(loaded_service, 2)]
    assert components.reward_pipeline is None
    assert components.stage3_reward_runtime is None


def test_rp66_shaped_control_reuses_answer_judge_and_disables_visual_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_service = SimpleNamespace(maximum_concurrency=16)
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime."
        "load_deepeyes_judge_service_config",
        lambda *_args, **_kwargs: loaded_service,
    )
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime.AsyncDeepEyesOpenRouterJudge",
        lambda service, *, local_maximum_concurrency: (
            "official-deepeyes",
            service,
            local_maximum_concurrency,
        ),
    )
    label = TGVFToolUtilityLabelBinding(
        sample_id="sample",
        training_index=0,
        utility_label="optional",
        confidence=0.5,
        row_sha256="4" * 64,
    )
    utility = TGVFToolUtilityRuntimeBinding(
        sidecar_path=Path("/fixture/tool-utility.jsonl"),
        sidecar_sha256="5" * 64,
        manifest_path=Path("/fixture/manifest.json"),
        manifest_sha256="6" * 64,
        dataset_iteration_identity_sha256="7" * 64,
        labels=MappingProxyType({label.sample_id: label}),
    )
    config = SimpleNamespace(
        schema_version=POLICY_E2E_RP66_SHAPED_CONTROL_RUN_CONFIG_SCHEMA,
        identity_sha256="8" * 64,
        reward=SimpleNamespace(
            profile="stage3-shaped-v1",
            judge_config_path=Path("/fixture/deepeyes-judge.json"),
            judge_config_sha256="9" * 64,
            tool_utility=utility,
            focus_reward_enabled=False,
            grounding_reward_enabled=False,
            visual_quality_judge_identity=None,
        ),
    )

    components = _Qwen3PolicyTrajectoryComponents(
        context=SimpleNamespace(
            config=config,
            placement=PolicyAgentLoopWorkerPlacement(7, 7, 7, 8),
        ),
        layout_builder=object(),
        server_client=object(),
        contextual_forward_identity=None,
        branch_merger_identities=(),
        observation_store=object(),
        behavior_store=object(),
        focus_execution_ledger=object(),
        crop_execution_ledger=object(),
        metrics_factory=lambda *_args: object(),
        agent_loop_output_cls=None,
        sample_index={},
        launch_mode="smoke",
    )

    assert components.official_deepeyes_judge == (
        "official-deepeyes",
        loaded_service,
        2,
    )
    assert components.async_stage3_spec is not None
    assert components.async_stage3_spec.visual_quality_enabled is False
    assert components.async_stage3_spec.visual_judge_identity is None


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
    name_or_path = "/fixture"
    _native = {
        "<|vision_start|>": 1,
        "<|image_pad|>": 2,
        "<|vision_end|>": 3,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._native[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
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


def _parsed_focus_call():
    text = (
        "inspect</think>"
        '<tool_call>{"name":"tgvf_focus_tool","arguments":'
        '{"target":"the gauge needle position"}}</tool_call>'
    )
    token_ids = tuple(1000 + index for index in range(len(text)))
    spans = tuple(
        TokenByteSpan(index, token_id, index, index + 1)
        for index, token_id in enumerate(token_ids)
    )
    return StrictToolCallParser(enabled_tool_names=("tgvf_focus_tool",)).parse(
        SampledAssistantTurn(text, token_ids, spans)
    )


def test_policy_layout_focus_and_final_expansion_share_one_idempotent_coordinate() -> (
    None
):
    tokenizer = _NativeTokenizer()
    model = ModelIdentity("qwen3_vl", "fixture", "/fixture", 256, SHA)
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
        preprocessed_pixel_values=torch.ones((4, 3), dtype=torch.float32),
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

    parsed = _parsed_focus_call()
    conditioning_ids = initial_ids + parsed.sampled_token_ids
    focus_layout = builder.build_focus_from_recorded_prefix(
        conditioning_input_ids=conditioning_ids,
        parsed_call=parsed,
        trajectory_source_visual=source,
        prior_observation_handles=(),
        source_visual=source_bundle,
    )

    assert focus_layout.visual_layout.original_image_positions == positions
    assert len(focus_layout.visual_layout.d_positions) == 4
    assert all(
        branch_positions == focus_layout.visual_layout.d_positions
        for branch_positions in focus_layout.visual_layout.deepstack_injection_positions
    )
    assert rope_inputs[0] == initial_ids
    assert rope_inputs[1][: len(conditioning_ids)] == conditioning_ids

    # A sampled assistant can emit Qwen's reserved visual tokens.  They remain
    # exact policy-owned input IDs, but are not extra multimodal items and must
    # not be offered to Qwen's M-RoPE image-discovery path as such.
    policy_suffix = (1, 2, 3, 1)
    final_ids = initial_ids + policy_suffix
    policy_positions = tuple(range(len(initial_ids), len(final_ids)))
    malformed_policy_layout = builder.expand_recorded_visual_sequence(
        final_ids,
        trajectory_source_visual=source,
        observation_handles=(),
        policy_token_positions=policy_positions,
    )
    assert tuple(malformed_policy_layout.input_ids[0].tolist()) == final_ids
    assert rope_inputs[-1] == initial_ids + (3, 2, 3, 3)

    # The same malformed opener outside policy ownership still proves real
    # prompt/environment corruption and remains fatal.
    with pytest.raises(ReplayMismatchError, match="malformed native"):
        builder.expand_recorded_visual_sequence(
            initial_ids + (1,),
            trajectory_source_visual=source,
            observation_handles=(),
        )

    with pytest.raises(ReplayMismatchError, match="ownership overlaps"):
        builder.expand_recorded_visual_sequence(
            initial_ids,
            trajectory_source_visual=source,
            observation_handles=(),
            policy_token_positions=(positions[0],),
        )


def test_final_materialization_tracks_only_policy_owned_positions() -> None:
    record = _record(tool_call_count=2, prompt_ids=(1, 2))

    final_ids, native_rows, policy_positions = _final_token_materialization(
        record.prompt_ids,
        record.trajectory_payload,
    )

    assert final_ids == record.prompt_ids + record.response_ids
    assert native_rows == ((100, 101), (102, 103, 104))
    assert policy_positions == (2, 3, 4, 7, 8, 9)
