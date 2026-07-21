from __future__ import annotations

import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.environment.focus_tool import SourceVisualTensorBundle
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.verl.policy_live_runtime import (
    _default_metrics_factory,
    _injected_visual_blocks,
    _single_sequence_visual_features,
)
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    DeepStackBranchRecord,
    FocusedObservationRecord,
    ObservationMasks,
    VisualLayout,
)
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan


SHA = "7" * 64
BRANCH_LAYERS = (8, 16, 24)


def test_default_live_metrics_factory_uses_the_pinned_verl_public_model() -> None:
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

    assert isinstance(_default_metrics_factory(object(), object()), AgentLoopMetrics)


def test_live_visual_features_restore_the_single_sequence_batch_axis() -> None:
    unbatched = torch.zeros((234, 4096), dtype=torch.bfloat16)

    normalized = _single_sequence_visual_features(
        unbatched, count=234, name="source visual embeddings"
    )

    assert normalized.shape == (1, 234, 4096)
    assert normalized[0].data_ptr() == unbatched.data_ptr()


def test_live_injected_blocks_batch_source_and_focused_d_only_at_forward_boundary() -> None:
    store = ObservationStore()
    trajectory_id = "run/sample/0/group"
    source_positions = (1, 2)
    d_positions = (6, 7)
    source_main = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    source_branches = tuple(
        torch.full((2, 8), float(10 + index), dtype=torch.bfloat16)
        for index in range(3)
    )
    source_bundle = SourceVisualTensorBundle(
        image_sha256=SHA,
        premerge_main=torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        premerge_deepstack=tuple(
            torch.full((8, 4), float(index), dtype=torch.bfloat16)
            for index in range(3)
        ),
        merged_main=source_main,
        merged_deepstack=source_branches,
        image_grid_thw=(1, 2, 4),
        spatial_merge_size=2,
    )
    source = record_trajectory_source_visual(
        trajectory_id=trajectory_id,
        source_visual=source_bundle,
        source_positions=source_positions,
        deepstack_branch_layers=BRANCH_LAYERS,
        deepstack_injection_positions=(source_positions,) * 3,
        observation_store=store,
    )

    main_d_tensor = torch.full((2, 8), 20.0, dtype=torch.bfloat16)
    d_branch_tensors = tuple(
        torch.full((2, 8), float(30 + index), dtype=torch.bfloat16)
        for index in range(3)
    )

    def put(name: str, tensor: torch.Tensor):
        return store.put_tensor(name, tensor, trajectory_id=trajectory_id)

    main_d = put("call.0.main_d", main_d_tensor)
    d_branches = tuple(
        put(f"call.0.d_deepstack.{layer}", tensor)
        for layer, tensor in zip(BRANCH_LAYERS, d_branch_tensors, strict=True)
    )
    sequence = 10
    position_ids = put(
        "call.0.position_ids", torch.arange(sequence).view(1, sequence)
    )
    mask = put(
        "call.0.attention_mask", torch.ones((1, sequence), dtype=torch.bool)
    )
    merger_ids = tuple(
        ArtifactIdentity(
            "qwen3-vl", f"deepstack-merger-{layer}", "fixture", str(index + 1) * 64
        )
        for index, layer in enumerate(BRANCH_LAYERS)
    )
    policy = PolicyVersion("run", 0, "5" * 64)
    record = FocusedObservationRecord(
        schema_version="focused-observation-v1",
        observation_id="observation-0",
        call_index=0,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 256, SHA),
        representation=ArtifactIdentity("tgvf", "adapter", "fixture", "6" * 64),
        condition=ConditionProvenance(
            provider="contextual_hidden_state",
            sampled_target_text_sha256="7" * 64,
            sampled_target_token_start=1,
            sampled_target_token_end=2,
            conditioning_target_token_start=4,
            conditioning_target_token_end=5,
            source_sequence_length=6,
            source_input_ids_sha256="8" * 64,
            trajectory_ids=(trajectory_id,),
            call_indices=(0,),
            hidden_layer=-1,
            contextual_forward_identity=ArtifactIdentity(
                "policy", "contextual-forward", "fixture", "9" * 64
            ),
            policy_version=policy,
        ),
        source_visual=source.state,
        payload=TensorPayloadSet(
            main_d=main_d,
            deepstack=d_branches,
            position_ids=position_ids,
            attention_mask=mask,
        ),
        branches=tuple(
            DeepStackBranchRecord(layer, ref, d_positions, merger)
            for layer, ref, merger in zip(
                BRANCH_LAYERS, d_branches, merger_ids, strict=True
            )
        ),
        layout=VisualLayout(
            sequence_length=sequence,
            original_image_positions=source_positions,
            d_positions=d_positions,
            deepstack_branch_layers=BRANCH_LAYERS,
            deepstack_injection_positions=(d_positions,) * 3,
        ),
        masks=ObservationMasks(mask, mask, mask, None),
        cache=CacheContract("no_cache", 0, None, None, True, 0.0),
    )
    handle = store.put(record)

    blocks = _injected_visual_blocks(
        store=store,
        trajectory_id=trajectory_id,
        source=source,
        observation_handles=(handle,),
        device=torch.device("cpu"),
    )

    assert tuple(block.kind for block in blocks) == ("source_image", "focused_d")
    assert blocks[0].positions == source_positions
    assert blocks[1].positions == d_positions
    assert blocks[0].deepstack_positions == (source_positions,) * 3
    assert blocks[1].deepstack_positions == (d_positions,) * 3
    assert blocks[0].embeddings.shape == (1, 2, 8)
    assert blocks[1].embeddings.shape == (1, 2, 8)
    assert tuple(tensor.shape for tensor in blocks[0].deepstack) == ((1, 2, 8),) * 3
    assert tuple(tensor.shape for tensor in blocks[1].deepstack) == ((1, 2, 8),) * 3
    torch.testing.assert_close(blocks[0].embeddings[0], source_main)
    torch.testing.assert_close(blocks[1].embeddings[0], main_d_tensor)
    for actual, expected in zip(blocks[0].deepstack, source_branches, strict=True):
        torch.testing.assert_close(actual[0], expected)
    for actual, expected in zip(blocks[1].deepstack, d_branch_tensors, strict=True):
        torch.testing.assert_close(actual[0], expected)

    # Content-addressed rollout artifacts remain unbatched; only the local
    # single-sequence injected-forward consumer restores B=1.
    assert store.resolve_verified(source.state.merged_main).shape == (2, 8)
    assert store.resolve_verified(main_d).shape == (2, 8)
    assert tuple(store.resolve_verified(ref).shape for ref in d_branches) == (
        (2, 8),
    ) * 3


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


def test_policy_layout_focus_and_final_expansion_share_one_idempotent_coordinate() -> None:
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
