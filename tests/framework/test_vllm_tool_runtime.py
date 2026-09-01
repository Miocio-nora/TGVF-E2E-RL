from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.environment.focus_tool import (
    PrecomputedTGVFObservationPayload,
    SourceVisualTensorBundle,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFCropMaterializationResult,
    TGVFFocusMaterializationResult,
    TGVFVLLMWorkerExtension,
    _BehaviorTraceBuffer,
    _focus_from_utility_wire,
    _focus_to_utility_wire,
    _crop_tgvf_from_utility_wire,
    _crop_tgvf_to_utility_wire,
    _source_from_utility_wire,
    _source_to_utility_wire,
    _tensor_from_utility_wire,
    _tensor_to_utility_wire,
    _runtime_classes,
    preprocessed_visual_identity_sha256,
)
from tgvf_rl.representation.adapter import TGVFAdapterMetadata
from tgvf_rl.representation.deepstack import DDeepStackPayload


def test_source_tensor_wire_survives_untyped_vllm_utility_transport() -> None:
    pytest.importorskip(
        "vllm.v1.serial_utils",
        reason="utility wire compatibility requires optional vLLM",
    )
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
    visual_wire = (
        MsgpackDecoder(UtilityResult)
        .decode(MsgpackEncoder().encode(UtilityResult(_source_to_utility_wire(visual))))
        .result
    )
    visual_restored = _source_from_utility_wire(visual_wire)
    assert isinstance(visual_restored, SourceVisualTensorBundle)
    torch.testing.assert_close(visual_restored.premerge_main, visual.premerge_main)


def test_focus_result_restores_all_nested_types_across_vllm_utility_transport() -> None:
    pytest.importorskip(
        "vllm.v1.serial_utils",
        reason="utility wire compatibility requires optional vLLM",
    )
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
                    torch.full((2, 8), float(i), dtype=torch.bfloat16) for i in range(3)
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

    transported_wire = (
        MsgpackDecoder(UtilityResult)
        .decode(MsgpackEncoder().encode(UtilityResult(_focus_to_utility_wire(result))))
        .result
    )
    transported = _focus_from_utility_wire(transported_wire)

    assert isinstance(transported, TGVFFocusMaterializationResult)
    assert isinstance(transported.observation, PrecomputedTGVFObservationPayload)
    assert isinstance(transported.observation.d_deepstack, DDeepStackPayload)
    assert isinstance(transported.observation.metadata, TGVFAdapterMetadata)
    torch.testing.assert_close(transported.hq, result.hq)
    torch.testing.assert_close(
        transported.observation.main_d, result.observation.main_d
    )


def test_crop_tgvf_wire_binds_crop_target_visual_and_d() -> None:
    layers = (8, 16, 24)
    identities = ("deep-8", "deep-16", "deep-24")
    crop_sha256 = "c" * 64
    observation = PrecomputedTGVFObservationPayload(
        main_d=torch.ones((2, 8), dtype=torch.bfloat16),
        d_deepstack=DDeepStackPayload(
            branch_layers=layers,
            branches=tuple(
                torch.full((2, 8), float(index), dtype=torch.bfloat16)
                for index in range(3)
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
    )
    crop_visual = SourceVisualTensorBundle(
        image_sha256=crop_sha256,
        premerge_main=torch.ones((8, 4), dtype=torch.bfloat16),
        premerge_deepstack=tuple(
            torch.full((8, 4), float(index), dtype=torch.bfloat16) for index in range(3)
        ),
        merged_main=torch.ones((2, 8), dtype=torch.bfloat16),
        merged_deepstack=tuple(
            torch.full((2, 8), float(index), dtype=torch.bfloat16) for index in range(3)
        ),
        image_grid_thw=(1, 2, 4),
        spatial_merge_size=2,
        decoded_rgb_sha256=crop_sha256,
    )
    expected = TGVFCropMaterializationResult(
        source_image_sha256="a" * 64,
        crop_sha256=crop_sha256,
        preprocessed_visual_sha256="d" * 64,
        image_grid_thw=(1, 2, 4),
        call_index=2,
        model_bbox_2d=(10, 20, 800, 900),
        target_start=4,
        target_end=7,
        target_token_ids=(31, 32, 33),
        provider="contextual_hidden_state",
        hq=torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        crop_visual=crop_visual,
        observation=observation,
    )

    restored = _crop_tgvf_from_utility_wire(_crop_tgvf_to_utility_wire(expected))

    assert restored.model_bbox_2d == expected.model_bbox_2d
    assert restored.target_token_ids == expected.target_token_ids
    torch.testing.assert_close(
        restored.crop_visual.premerge_main,
        crop_visual.premerge_main,
    )
    torch.testing.assert_close(restored.observation.main_d, observation.main_d)


def test_vllm_behavior_trace_captures_generated_token_hidden_states_and_releases() -> (
    None
):
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

    torch.testing.assert_close(trace.hidden[:2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert trace.covered == bytearray((1, 1, 0, 0))
    assert extension.tgvf_release_trajectory("trajectory-0", ("turn-0",))
    assert extension._tgvf_sources() == {}
    assert extension._tgvf_traces() == {}


def test_crop_rpc_reuses_the_rollout_worker_visual_path() -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_source_cache = {"trajectory-0": object()}
    expected = SourceVisualTensorBundle(
        image_sha256="c" * 64,
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
        decoded_rgb_sha256="c" * 64,
    )
    calls = []

    def materialize_visual(**kwargs):
        calls.append(kwargs)
        return expected

    extension._tgvf_materialize_visual = materialize_visual
    wire = extension.tgvf_materialize_crop(
        "trajectory-0",
        0,
        _tensor_to_utility_wire(torch.ones((4, 6))),
        (1, 2, 2),
        "c" * 64,
    )

    restored = _source_from_utility_wire(wire)
    assert restored.image_sha256 == expected.image_sha256
    assert calls[0]["image_sha256"] == "c" * 64
    assert tuple(calls[0]["image_grid_thw"].shape) == (1, 3)


def test_worker_atomic_crop_tgvf_runs_crop_vision_and_adapter_once() -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    source_sha256 = "a" * 64
    crop_sha256 = "c" * 64
    extension._tgvf_source_cache = {
        "trajectory-0": SimpleNamespace(image_sha256=source_sha256)
    }
    crop_visual = SourceVisualTensorBundle(
        image_sha256=crop_sha256,
        premerge_main=torch.ones((8, 4), dtype=torch.bfloat16),
        premerge_deepstack=tuple(
            torch.full((8, 4), float(index), dtype=torch.bfloat16) for index in range(3)
        ),
        merged_main=torch.ones((2, 8), dtype=torch.bfloat16),
        merged_deepstack=tuple(
            torch.full((2, 8), float(index), dtype=torch.bfloat16) for index in range(3)
        ),
        image_grid_thw=(1, 2, 4),
        spatial_merge_size=2,
        decoded_rgb_sha256=crop_sha256,
    )
    layers = (8, 16, 24)
    identities = ("deep-8", "deep-16", "deep-24")
    adapter_output = SimpleNamespace(
        main_d=torch.ones((2, 8), dtype=torch.bfloat16),
        d_deepstack=DDeepStackPayload(
            branch_layers=layers,
            branches=tuple(
                torch.full((2, 8), float(index), dtype=torch.bfloat16)
                for index in range(3)
            ),
            projection_identities=identities,
        ),
        metadata=TGVFAdapterMetadata(
            branch_layers=layers,
            main_projection_identity="main",
            deepstack_projection_identities=identities,
            batched=False,
            batch_size=1,
            target_token_count=2,
            pre_merge_visual_token_count=8,
            d_token_count=2,
            condition_provenance=None,
        ),
    )
    calls: list[tuple[str, object]] = []

    def materialize_visual(**kwargs: object) -> SourceVisualTensorBundle:
        calls.append(("vision", kwargs))
        return crop_visual

    class Adapter:
        def __call__(self, adapter_input: object) -> object:
            calls.append(("adapter", adapter_input))
            return adapter_output

    extension._tgvf_materialize_visual = materialize_visual
    extension._tgvf_target_hidden_states = lambda **kwargs: torch.ones(
        (2, 8), dtype=torch.bfloat16
    )
    extension._tgvf_adapter = lambda: Adapter()
    extension._tgvf_behavior_traces = {"turn-0": object()}
    pixels = torch.ones((8, 6))

    wire = extension.tgvf_materialize_crop_tgvf(
        "trajectory-0",
        "turn-0",
        1,
        _tensor_to_utility_wire(pixels),
        (1, 2, 4),
        source_sha256,
        crop_sha256,
        preprocessed_visual_identity_sha256(pixels, (1, 2, 4)),
        (10, 20, 800, 900),
        3,
        5,
        (41, 42),
        "contextual_hidden_state",
    )
    result = _crop_tgvf_from_utility_wire(wire)

    assert [call[0] for call in calls] == ["vision", "adapter"]
    assert result.source_image_sha256 == source_sha256
    assert result.crop_sha256 == crop_sha256
    assert result.target_token_ids == (41, 42)
    assert "turn-0" not in extension._tgvf_behavior_traces


def test_worker_atomic_crop_tgvf_rejects_swapped_preprocessed_tensor() -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    source_sha256 = "a" * 64
    extension._tgvf_source_cache = {
        "trajectory-0": SimpleNamespace(image_sha256=source_sha256)
    }
    declared_pixels = torch.zeros((8, 6))
    swapped_pixels = torch.ones((8, 6))

    with pytest.raises(
        IdentityMismatchError,
        match="preprocessed visual content changed across RPC",
    ):
        extension.tgvf_materialize_crop_tgvf(
            "trajectory-0",
            "turn-0",
            0,
            _tensor_to_utility_wire(swapped_pixels),
            (1, 2, 4),
            source_sha256,
            "b" * 64,
            preprocessed_visual_identity_sha256(
                declared_pixels,
                (1, 2, 4),
            ),
            (10, 20, 800, 900),
            3,
            5,
            (41, 42),
            "contextual_hidden_state",
        )


def test_vllm_server_shutdown_stops_http_engine_and_bound_sockets() -> None:
    pytest.importorskip(
        "verl",
        reason="vLLM server lifecycle adapter requires optional pinned veRL",
    )
    events: list[str] = []

    class Engine:
        output_handler = None
        engine_core = object()

        def shutdown(self):
            events.append("engine")
            self.output_handler.cancel()

    class Socket:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self):
            events.append(self.name)

    async def exercise() -> object:
        server_cls = _runtime_classes()[3]
        server = object.__new__(server_cls)
        engine = Engine()

        async def output_handler() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("handler")

        engine.output_handler = asyncio.create_task(output_handler())
        server.engine = engine
        server._server_task = asyncio.create_task(asyncio.Event().wait())
        server._master_sock = Socket("master")
        server._dp_rpc_sock = Socket("rpc")
        server._dp_master_sock = Socket("dp-master")
        await server.tgvf_shutdown()
        return server, engine

    server, engine = asyncio.run(exercise())
    assert events == ["engine", "handler", "master", "rpc", "dp-master"]
    assert server.engine is None
    assert engine.output_handler is None
    assert engine.engine_core is None
    assert server._master_sock is None


def test_vllm_checkpoint_sleep_uses_level_one_and_fans_out() -> None:
    pytest.importorskip(
        "verl",
        reason="vLLM server lifecycle adapter requires optional pinned veRL",
    )
    events: list[object] = []

    class Engine:
        async def wait_for_requests_to_drain(self) -> None:
            events.append("drain")

        async def sleep(self, *, level: int) -> None:
            events.append(("sleep", level))

    class RemoteMethod:
        def __init__(self, name: str) -> None:
            self.name = name

        async def remote(self) -> None:
            events.append(self.name)

    async def exercise() -> None:
        _manager_cls, _client_cls, replica_cls, server_cls = _runtime_classes()
        server = object.__new__(server_cls)
        server.node_rank = 0
        server.config = SimpleNamespace(free_cache_engine=True)
        server.engine = Engine()
        await server.tgvf_sleep_for_checkpoint()

        replica = object.__new__(replica_cls)
        replica.servers = (
            SimpleNamespace(tgvf_sleep_for_checkpoint=RemoteMethod("server-0")),
            SimpleNamespace(tgvf_sleep_for_checkpoint=RemoteMethod("server-1")),
        )
        await replica.sleep_for_checkpoint()

    asyncio.run(exercise())
    assert events == ["drain", ("sleep", 1), "server-0", "server-1"]
