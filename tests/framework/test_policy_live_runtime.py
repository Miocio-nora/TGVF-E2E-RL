from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import torch

from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.verl.policy_live_runtime import (
    Qwen3PolicyE2ELiveRuntimeBuilder,
    _BoundTGVFVisualQualityRuntimeJudge,
    _build_reward_pipeline,
    _default_metrics_factory,
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
