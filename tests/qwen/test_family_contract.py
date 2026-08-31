from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.contracts.identity import (
    ArtifactIdentity,
    ModelIdentity,
    PolicyVersion,
    SupportLevel,
)
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    DeepStackBranchRecord,
    FocusedObservationRecord,
    ObservationMasks,
    SourceVisualState,
    TrajectorySourceVisual,
    VisualLayout,
)
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.qwen.base import (
    ReplayConsumer,
    _prove_native_streaming_injected_request,
    gather_next_token_logprobs,
    injected_request_from_recorded,
    materialize_deepstack,
    materialize_inputs_embeds,
    resolve_replay_request,
)
from tgvf_rl.qwen.qwen25_vl import Qwen25VLAdapter
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 8, bias=False)
        self.last_deepstack_visual_embeds = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        self.last_deepstack_visual_embeds = deepstack_visual_embeds
        hidden = self.proj(inputs_embeds)
        if deepstack_visual_embeds:
            for branch in deepstack_visual_embeds:
                hidden = hidden.clone()
                hidden[visual_pos_masks] += branch
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class TinyQwen(nn.Module):
    def __init__(self, *, native_deepstack_enabled: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            tgvf_native_deepstack_enabled=native_deepstack_enabled
        )
        self.model = SimpleNamespace(language_model=TinyLanguageModel())
        self.lm_head = nn.Linear(8, 32, bias=False)


def _replay(*, branches: int, calls: int = 2):
    store = ObservationStore()
    model = ModelIdentity("qwen3_vl", "tiny", "/tiny", 32, "1" * 64)
    policy = PolicyVersion("run", 0, "2" * 64)
    representation = ArtifactIdentity("tgvf", "adapter", "tiny", "3" * 64)
    merger = ArtifactIdentity("qwen", "merger", "tiny", "4" * 64)
    source = store.put_tensor("source.main", torch.randn(1, 2, 8))
    source_branches = tuple(
        store.put_tensor(f"source.branch.{index}", torch.randn(1, 2, 8))
        for index in range(branches)
    )
    handles = []
    sequence = 8
    common_mask = store.put_tensor(
        "final.common_mask", torch.ones(1, sequence, dtype=torch.bool)
    )
    for call_index in range(calls):
        d_positions = (3 + 2 * call_index, 4 + 2 * call_index)
        main_d = store.put_tensor(f"call.{call_index}.main_d", torch.randn(1, 2, 8))
        branch_refs = tuple(
            store.put_tensor(f"call.{call_index}.branch.{index}", torch.randn(1, 2, 8))
            for index in range(branches)
        )
        position_ref = store.put_tensor(
            f"call.{call_index}.position_ids", torch.arange(sequence).view(1, sequence)
        )
        record = FocusedObservationRecord(
            schema_version="focused-observation-v1",
            observation_id=f"observation-{call_index}",
            call_index=call_index,
            model=model,
            representation=representation,
            condition=ConditionProvenance(
                provider="contextual_hidden_state",
                sampled_target_text_sha256="5" * 64,
                sampled_target_token_start=1,
                sampled_target_token_end=2,
                conditioning_target_token_start=1,
                conditioning_target_token_end=2,
                source_sequence_length=sequence,
                source_input_ids_sha256="6" * 64,
                trajectory_ids=("trajectory",),
                call_indices=(call_index,),
                hidden_layer=1,
                contextual_forward_identity=ArtifactIdentity(
                    "policy", "contextual-forward", "fixture", "4" * 64
                ),
                policy_version=policy,
            ),
            source_visual=SourceVisualState(
                image_sha256="7" * 64,
                premerge_main=source,
                premerge_deepstack=source_branches,
                merged_main=source,
                merged_deepstack=source_branches,
                image_grid_thw=(1, 2, 2),
                spatial_merge_size=1,
            ),
            payload=TensorPayloadSet(
                main_d=main_d,
                deepstack=branch_refs,
                position_ids=position_ref,
                attention_mask=common_mask,
            ),
            branches=tuple(
                DeepStackBranchRecord(index, ref, d_positions, merger)
                for index, ref in enumerate(branch_refs)
            ),
            layout=VisualLayout(
                sequence_length=sequence,
                original_image_positions=(1, 2),
                d_positions=d_positions,
                deepstack_branch_layers=tuple(range(branches)),
                deepstack_injection_positions=tuple(
                    d_positions for _ in range(branches)
                ),
            ),
            masks=ObservationMasks(common_mask, common_mask, common_mask, None),
            cache=CacheContract("no_cache", 0, None, None, True, 0.0),
        )
        handles.append(store.put(record))
    input_ids = store.put_tensor(
        "final.input_ids", torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    )
    positions = store.put_tensor(
        "final.position_ids", torch.arange(sequence).view(1, sequence)
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id=f"replay-{branches}-{calls}",
        trajectory_id="trajectory",
        model=model,
        behavior_policy=policy,
        source_visual=TrajectorySourceVisual(
            state=SourceVisualState(
                image_sha256="7" * 64,
                premerge_main=source,
                premerge_deepstack=source_branches,
                merged_main=source,
                merged_deepstack=source_branches,
                image_grid_thw=(1, 2, 2),
                spatial_merge_size=1,
            ),
            positions=(1, 2),
            deepstack_branch_layers=tuple(range(branches)),
            deepstack_injection_positions=tuple((1, 2) for _ in range(branches)),
        ),
        observation_handles=tuple(handles),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=positions,
            attention_mask=common_mask,
            policy_attention_mask=common_mask,
            reference_attention_mask=common_mask,
            teacher_attention_mask=common_mask,
        ),
    )
    return store, store.put_replay(replay)


def test_qwen3_recorded_forward_is_deterministic_and_uses_all_branches() -> None:
    torch.manual_seed(3)
    model = TinyQwen()
    store, replay = _replay(branches=3)
    adapter = Qwen3VLAdapter()
    first = adapter.forward_recorded(model, store, replay, ReplayConsumer.POLICY)
    second = adapter.forward_recorded(model, store, replay, ReplayConsumer.POLICY)
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    assert first.visual_position_mask.sum().item() == 6


@pytest.mark.parametrize("native_deepstack_enabled", (True, False))
def test_qwen3_recorded_forward_applies_native_deepstack_control(
    native_deepstack_enabled: bool,
) -> None:
    model = TinyQwen(native_deepstack_enabled=native_deepstack_enabled)
    store, replay = _replay(branches=3, calls=1)

    Qwen3VLAdapter().forward_recorded(
        model,
        store,
        replay,
        ReplayConsumer.POLICY,
    )

    observed = model.model.language_model.last_deepstack_visual_embeds
    if native_deepstack_enabled:
        assert isinstance(observed, (list, tuple))
        assert len(observed) == 3
    else:
        assert observed is None


def test_zero_tool_policy_and_reference_consume_the_same_source_bundle() -> None:
    torch.manual_seed(13)
    model = TinyQwen()
    store, replay = _replay(branches=3, calls=0)
    bundle = store.export_replay_bundle(replay)
    adapter = Qwen3VLAdapter()

    policy = adapter.forward_replay_bundle(model, bundle, ReplayConsumer.POLICY)
    reference = adapter.forward_replay_bundle(model, bundle, ReplayConsumer.REFERENCE)

    torch.testing.assert_close(policy.logits, reference.logits, rtol=0, atol=0)
    assert torch.equal(policy.visual_position_mask, reference.visual_position_mask)
    assert policy.visual_position_mask.sum().item() == 2


def test_qwen25_support_is_not_misrepresented_as_qwen3_deepstack() -> None:
    adapter = Qwen25VLAdapter()
    assert adapter.capabilities.support_level is SupportLevel.SYNTHETIC
    store, replay = _replay(branches=1, calls=1)
    with pytest.raises(ValueError, match="no accepted DeepStack"):
        adapter.forward_recorded(TinyQwen(), store, replay, ReplayConsumer.REFERENCE)


def test_replay_logprobs_use_preceding_logits() -> None:
    logits = torch.zeros(1, 4, 5)
    logits[0, 1, 3] = 4.0
    tokens = torch.tensor([[0, 1, 3, 2]])
    gathered = gather_next_token_logprobs(logits, tokens, torch.tensor([[2]]))
    expected = torch.log_softmax(logits[0, 1], dim=-1)[3]
    torch.testing.assert_close(gathered[0, 0], expected)


def test_recorded_deepstack_materializes_on_embedding_device_and_target_dtype() -> None:
    model = TinyQwen()
    store, replay = _replay(branches=3, calls=1)
    recorded = resolve_replay_request(store, replay, ReplayConsumer.POLICY)
    request = injected_request_from_recorded(recorded)
    inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)

    branches = materialize_deepstack(
        request,
        visual_mask,
        target_dtype=torch.float64,
    )

    assert len(branches) == 3
    assert all(branch.device == inputs_embeds.device for branch in branches)
    assert all(branch.dtype == torch.float64 for branch in branches)


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    ((float("nan"), "must be finite"), (0.25, "positive bias")),
)
def test_generic_injected_forward_rejects_invalid_additive_mask_contents(
    invalid_value: float,
    message: str,
) -> None:
    model = TinyQwen()
    store, replay = _replay(branches=3, calls=1)
    request = injected_request_from_recorded(
        resolve_replay_request(store, replay, ReplayConsumer.POLICY)
    )
    batch, sequence = request.input_ids.shape
    attention_mask = torch.zeros(batch, 1, sequence, sequence)
    attention_mask[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match=message):
        materialize_inputs_embeds(
            model,
            replace(request, attention_mask=attention_mask),
        )


def test_generic_deepstack_materialization_rejects_layout_mismatch() -> None:
    model = TinyQwen()
    store, replay = _replay(branches=3, calls=1)
    request = injected_request_from_recorded(
        resolve_replay_request(store, replay, ReplayConsumer.POLICY)
    )
    first = request.visual_blocks[0]
    mismatched_positions = (0, first.positions[-1])
    request = replace(
        request,
        visual_blocks=(
            replace(
                first,
                deepstack_positions=(
                    mismatched_positions,
                    *first.deepstack_positions[1:],
                ),
            ),
            *request.visual_blocks[1:],
        ),
    )
    inputs_embeds, visual_mask = materialize_inputs_embeds(model, request)

    with pytest.raises(ValueError, match="must equal the main visual positions"):
        materialize_deepstack(
            request,
            visual_mask,
            target_dtype=inputs_embeds.dtype,
        )


def test_native_injected_request_proof_rejects_post_construction_mutation() -> None:
    model = TinyQwen()
    store, replay = _replay(branches=3, calls=1)
    request = injected_request_from_recorded(
        resolve_replay_request(store, replay, ReplayConsumer.POLICY)
    )
    batch, sequence = request.input_ids.shape
    proven = _prove_native_streaming_injected_request(
        replace(
            request,
            attention_mask=torch.zeros(batch, 1, sequence, sequence),
        )
    )
    proven.attention_mask[0, 0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="changed after construction proof"):
        materialize_inputs_embeds(model, proven)
