from __future__ import annotations

import pickle
from types import SimpleNamespace

import torch

from tgvf_rl.environment.focus_tool import (
    PrecomputedTGVFObservationPayload,
    SourceVisualTensorBundle,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFFocusMaterializationResult,
    TGVFVLLMWorkerExtension,
    _BehaviorTraceBuffer,
    _focus_from_utility_wire,
    _focus_to_utility_wire,
    _source_from_utility_wire,
    _source_to_utility_wire,
    _tensor_from_utility_wire,
    _tensor_to_utility_wire,
)
from tgvf_rl.representation.adapter import TGVFAdapterMetadata
from tgvf_rl.representation.deepstack import DDeepStackPayload


def test_source_tensor_wire_survives_untyped_vllm_utility_transport() -> None:
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    arguments = {
        "pixel_values_wire": _tensor_to_utility_wire(source),
        "image_grid_thw": (1, 2, 4),
    }

    transported = MsgpackDecoder().decode(MsgpackEncoder().encode(arguments))

    pickle.dumps(transported)
    restored = _tensor_from_utility_wire(transported["pixel_values_wire"])
    torch.testing.assert_close(restored, source)

    visual = SourceVisualTensorBundle(
        image_sha256="a" * 64,
        premerge_main=torch.ones((4, 2), dtype=torch.bfloat16),
        premerge_deepstack=tuple(
            torch.full((4, 2), float(i), dtype=torch.bfloat16) for i in range(3)
        ),
        merged_main=torch.ones((1, 8), dtype=torch.bfloat16),
        merged_deepstack=tuple(
            torch.full((1, 8), float(i), dtype=torch.bfloat16) for i in range(3)
        ),
        image_grid_thw=(1, 2, 2),
        spatial_merge_size=2,
        decoded_rgb_sha256="a" * 64,
    )
    visual_wire = MsgpackDecoder(UtilityResult).decode(
        MsgpackEncoder().encode(UtilityResult(_source_to_utility_wire(visual)))
    ).result
    visual_restored = _source_from_utility_wire(visual_wire)
    assert isinstance(visual_restored, SourceVisualTensorBundle)
    torch.testing.assert_close(visual_restored.premerge_main, visual.premerge_main)


def test_focus_result_restores_all_nested_types_across_vllm_utility_transport() -> None:
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    layers = (8, 16, 24)
    identities = ("deep-8", "deep-16", "deep-24")
    result = TGVFFocusMaterializationResult(
        hq=torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        observation=PrecomputedTGVFObservationPayload(
            main_d=torch.ones((2, 8), dtype=torch.bfloat16),
            d_deepstack=DDeepStackPayload(
                branch_layers=layers,
                branches=tuple(
                    torch.full((2, 8), float(i), dtype=torch.bfloat16)
                    for i in range(3)
                ),
                projection_identities=identities,
            ),
            metadata=TGVFAdapterMetadata(
                branch_layers=layers,
                main_projection_identity="main",
                deepstack_projection_identities=identities,
                batched=False,
                batch_size=1,
                target_token_count=3,
                pre_merge_visual_token_count=8,
                d_token_count=2,
                condition_provenance=None,
            ),
        ),
    )

    transported_wire = MsgpackDecoder(UtilityResult).decode(
        MsgpackEncoder().encode(UtilityResult(_focus_to_utility_wire(result)))
    ).result
    transported = _focus_from_utility_wire(transported_wire)

    assert isinstance(transported, TGVFFocusMaterializationResult)
    assert isinstance(transported.observation, PrecomputedTGVFObservationPayload)
    assert isinstance(transported.observation.d_deepstack, DDeepStackPayload)
    assert isinstance(transported.observation.metadata, TGVFAdapterMetadata)
    torch.testing.assert_close(transported.hq, result.hq)
    torch.testing.assert_close(transported.observation.main_d, result.observation.main_d)


def test_vllm_behavior_trace_captures_generated_token_hidden_states_and_releases() -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    trace = _BehaviorTraceBuffer(
        prompt_length=3,
        capacity=4,
        hidden=torch.zeros((4, 2)),
        covered=bytearray(4),
    )
    extension._tgvf_behavior_traces = {"turn-0": trace}
    extension._tgvf_source_cache = {"trajectory-0": object()}
    extension.model_runner = SimpleNamespace(
        execute_model_state=(
            SimpleNamespace(num_scheduled_tokens={"turn-0": 2}),
            None,
            None,
            None,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            None,
            None,
            None,
        ),
        input_batch=SimpleNamespace(
            req_ids=["turn-0"],
            num_computed_tokens_cpu=[3],
        ),
    )

    extension._tgvf_capture_execute_state()

    torch.testing.assert_close(
        trace.hidden[:2], torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert trace.covered == bytearray((1, 1, 0, 0))
    assert extension.tgvf_release_trajectory(
        "trajectory-0", ("turn-0",)
    )
    assert extension._tgvf_sources() == {}
    assert extension._tgvf_traces() == {}
