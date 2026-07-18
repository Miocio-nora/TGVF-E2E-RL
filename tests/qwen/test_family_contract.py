from __future__ import annotations

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
    VisualLayout,
)
from tgvf_rl.observations.store import (
    ObservationStore,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.qwen.base import ReplayConsumer, gather_next_token_logprobs
from tgvf_rl.qwen.qwen25_vl import Qwen25VLAdapter
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 8, bias=False)

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
        hidden = self.proj(inputs_embeds)
        if deepstack_visual_embeds:
            for branch in deepstack_visual_embeds:
                hidden = hidden.clone()
                hidden[visual_pos_masks] += branch
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
                source_input_ids_sha256="6" * 64,
                trajectory_ids=("trajectory",),
                call_indices=(call_index,),
                hidden_layer=1,
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
