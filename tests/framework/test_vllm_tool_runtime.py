from __future__ import annotations

import pickle
from types import SimpleNamespace

import torch

from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFVLLMWorkerExtension,
    _BehaviorTraceBuffer,
    _tensor_from_utility_wire,
    _tensor_to_utility_wire,
)


def test_source_tensor_wire_survives_untyped_vllm_utility_transport() -> None:
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    arguments = {
        "pixel_values_wire": _tensor_to_utility_wire(source),
        "image_grid_thw": (1, 2, 4),
    }

    transported = MsgpackDecoder().decode(MsgpackEncoder().encode(arguments))

    pickle.dumps(transported)
    restored = _tensor_from_utility_wire(transported["pixel_values_wire"])
    torch.testing.assert_close(restored, source)


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
