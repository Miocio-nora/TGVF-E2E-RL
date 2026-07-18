from __future__ import annotations

import torch
from torch import nn

from tgvf_rl.conditioning.providers import ContextualHiddenStateConditionProvider
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import TokenSpan
from tgvf_rl.environment.focus_tool import (
    ReplayLayoutTensors,
    SourceVisualTensorBundle,
    TGVFFocusTool,
    ToolExecutionRequest,
)
from tgvf_rl.observations.schema import VisualLayout
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.schema import SampledAssistantTurn, TokenByteSpan
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.deepstack import FrozenProjectionPort


class Merger(nn.Module):
    def __init__(self, d_v: int, d_lm: int) -> None:
        super().__init__()
        self.proj = nn.Linear(4 * d_v, d_lm, bias=False)

    def forward(self, tokens):
        return self.proj(tokens.reshape(-1, tokens.shape[-1] * 4))


def _parsed_call():
    text = (
        '<tool_call>{"name":"tgvf_focus_tool","arguments":{"target":"x"}}</tool_call>'
    )
    ids = tuple(ord(char) for char in text)
    spans = tuple(
        TokenByteSpan(index, token, index, index + 1) for index, token in enumerate(ids)
    )
    return StrictToolCallParser().parse(SampledAssistantTurn(text, ids, spans))


def test_focus_tool_materializes_main_and_deepstack_once() -> None:
    torch.manual_seed(2)
    model_id = ModelIdentity("qwen3_vl", "tiny", "/tiny", 256, "0" * 64)
    parsed = _parsed_call()
    input_ids = torch.tensor(parsed.sampled_token_ids)
    span = TokenSpan(parsed.target_span.token_start, parsed.target_span.token_end)
    condition = ContextualHiddenStateConditionProvider(
        model_identity=model_id, hidden_layer=2
    ).build(
        hidden_states=torch.randn(len(input_ids), 8),
        input_ids=input_ids,
        target_span=span,
        expected_target_token_ids=parsed.target_span.token_ids,
        trajectory_id="trajectory",
        call_index=0,
    )
    main_projection = FrozenProjectionPort(
        Merger(4, 8), identity="main", input_dim=4, output_dim=8, spatial_merge_size=2
    )
    branch_projection = FrozenProjectionPort(
        Merger(4, 8),
        identity="branch8",
        input_dim=4,
        output_dim=8,
        spatial_merge_size=2,
    )
    adapter = TGVFAdapter(
        d_lm=8,
        d_v=4,
        main_projection=main_projection,
        deepstack_projections=(branch_projection,),
        branch_layers=(8,),
    )
    store = ObservationStore()
    layout = VisualLayout(
        sequence_length=8,
        original_image_positions=(1,),
        d_positions=(4,),
        deepstack_branch_layers=(8,),
        deepstack_injection_positions=((4,),),
    )
    mask = torch.ones(1, 8, dtype=torch.bool)
    request = ToolExecutionRequest(
        trajectory_id="trajectory",
        call_index=0,
        parsed_call=parsed,
        condition=condition,
        source_visual=SourceVisualTensorBundle(
            image_sha256="1" * 64,
            premerge_main=torch.randn(4, 4),
            premerge_deepstack=(torch.randn(4, 4),),
            merged_main=torch.randn(1, 8),
            merged_deepstack=(torch.randn(1, 8),),
            image_grid_thw=(1, 2, 2),
            spatial_merge_size=2,
        ),
        layout=ReplayLayoutTensors(
            position_ids=torch.arange(8).view(1, 8),
            attention_mask=mask,
            policy_visible_mask=mask,
            reference_visible_mask=mask,
            teacher_visible_mask=mask,
            token_type_ids=None,
            original_image_key_block_mask=None,
            cache_position=None,
            rope_delta=None,
            visual_layout=layout,
        ),
        model=model_id,
        policy_version=PolicyVersion("smoke", 0, "2" * 64),
        representation=ArtifactIdentity("tgvf", "adapter", "synthetic", "3" * 64),
        branch_merger_identities=(
            ArtifactIdentity("qwen", "merger-8", "synthetic", "4" * 64),
        ),
    )
    result = TGVFFocusTool(adapter, store).execute(request)
    replay = store.resolve_record(result.handle)
    torch.testing.assert_close(
        store.resolve_verified(replay.payload.main_d),
        result.adapter_output.main_d,
        rtol=0,
        atol=0,
    )
    assert (
        replay.condition.source_input_ids_sha256
        == condition.provenance.source_input_ids_sha256
    )
    assert len(replay.payload.deepstack) == 1
