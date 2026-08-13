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
    TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
    TGVFCropMaterializationResult,
    TGVFFocusMaterializationResult,
    TGVF_PHASE_BOUNDARY_QUIESCE_SCHEMA,
    TGVF_VLLM_FINISH_REASON_FIELD,
    TGVF_VLLM_STOP_REASON_FIELD,
    TGVFVLLMWorkerExtension,
    _BehaviorTraceBuffer,
    _adapter_owned_state_from_utility_wire,
    _adapter_owned_state_to_utility_wire,
    _focus_from_utility_wire,
    _focus_to_utility_wire,
    _crop_tgvf_from_utility_wire,
    _crop_tgvf_to_utility_wire,
    _runtime_classes,
    _source_from_utility_wire,
    _source_to_utility_wire,
    _tensor_from_utility_wire,
    _tensor_to_utility_wire,
    adapter_owned_state_sha256,
    bind_tgvf_adapter_state_update_manager,
    preprocessed_visual_identity_sha256,
)
from tgvf_rl.representation.adapter import TGVFAdapterMetadata
from tgvf_rl.representation.deepstack import DDeepStackPayload


class _FakeAdapter:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = {name: tensor.clone() for name, tensor in state.items()}
        self.load_count = 0
        self.requires_grad_value = True
        self.training = True

    def artifact_state_dict(self, *, keep_vars: bool = False):
        del keep_vars
        return self.state

    def load_artifact_state_dict(self, state):
        self.load_count += 1
        self.state = {name: tensor.clone() for name, tensor in state.items()}

    def requires_grad_(self, value: bool):
        self.requires_grad_value = value
        return self

    def eval(self):
        self.training = False
        return self


def _adapter_ack(
    optimizer_step: int,
    state_sha256: str,
    tensor_count: int,
    *,
    applied: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
        "optimizer_step": optimizer_step,
        "state_sha256": state_sha256,
        "tensor_count": tensor_count,
        "applied": applied,
        "cleared_source_count": 0,
        "cleared_trace_count": 0,
    }


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
    visual_wire = (
        MsgpackDecoder(UtilityResult)
        .decode(MsgpackEncoder().encode(UtilityResult(_source_to_utility_wire(visual))))
        .result
    )
    visual_restored = _source_from_utility_wire(visual_wire)
    assert isinstance(visual_restored, SourceVisualTensorBundle)
    torch.testing.assert_close(visual_restored.premerge_main, visual.premerge_main)


def test_preprocessed_visual_identity_binds_tensor_and_geometry() -> None:
    pixels = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    identity = preprocessed_visual_identity_sha256(pixels, (1, 2, 2))

    assert identity == preprocessed_visual_identity_sha256(
        pixels.clone(), torch.tensor([[1, 2, 2]])
    )
    assert identity != preprocessed_visual_identity_sha256(
        pixels.add(1), (1, 2, 2)
    )
    assert identity != preprocessed_visual_identity_sha256(pixels, (1, 1, 4))


def test_adapter_owned_state_wire_and_digest_are_order_independent() -> None:
    state = {
        "branch.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "query.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    reordered = dict(reversed(tuple(state.items())))

    assert adapter_owned_state_sha256(state) == adapter_owned_state_sha256(reordered)
    restored = _adapter_owned_state_from_utility_wire(
        _adapter_owned_state_to_utility_wire(state)
    )
    assert set(restored) == set(state)
    for name in state:
        torch.testing.assert_close(restored[name], state[name])


def test_worker_adapter_update_is_strict_versioned_and_clears_caches() -> None:
    initial = {
        "query.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
        "value.bias": torch.zeros((2,), dtype=torch.bfloat16),
    }
    updated = {
        "query.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "value.bias": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
    }
    adapter = _FakeAdapter(initial)
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_adapter_module = adapter
    extension._tgvf_source_cache = {"source": object()}
    extension._tgvf_behavior_traces = {"trace": object()}
    digest = adapter_owned_state_sha256(updated)

    ack = extension.tgvf_update_adapter_owned_state(
        7,
        digest,
        _adapter_owned_state_to_utility_wire(updated),
    )

    assert ack == {
        "schema_version": TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
        "optimizer_step": 7,
        "state_sha256": digest,
        "tensor_count": 2,
        "applied": True,
        "cleared_source_count": 1,
        "cleared_trace_count": 1,
    }
    assert adapter.load_count == 1
    assert not adapter.requires_grad_value
    assert not adapter.training
    for name in updated:
        torch.testing.assert_close(adapter.state[name], updated[name])
    assert extension._tgvf_sources() == {}
    assert extension._tgvf_traces() == {}

    extension._tgvf_source_cache["retry"] = object()
    retry = extension.tgvf_update_adapter_owned_state(
        7,
        digest,
        _adapter_owned_state_to_utility_wire(updated),
    )
    assert retry["applied"] is False
    assert retry["cleared_source_count"] == 1
    assert adapter.load_count == 1
    assert extension._tgvf_sources() == {}

    conflicting = {**updated, "value.bias": updated["value.bias"] + 1}
    with pytest.raises(IdentityMismatchError, match="same Adapter optimizer step"):
        extension.tgvf_update_adapter_owned_state(
            7,
            adapter_owned_state_sha256(conflicting),
            _adapter_owned_state_to_utility_wire(conflicting),
        )
    with pytest.raises(RuntimeError, match="stale Adapter-owned state"):
        extension.tgvf_update_adapter_owned_state(
            6,
            digest,
            _adapter_owned_state_to_utility_wire(updated),
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"query.weight": torch.zeros((3, 2), dtype=torch.bfloat16)},
        {"query.weight": torch.zeros((2, 3), dtype=torch.float32)},
        {"unexpected.weight": torch.zeros((2, 3), dtype=torch.bfloat16)},
    ],
)
def test_worker_adapter_update_rejects_state_before_mutating(
    invalid: dict[str, torch.Tensor],
) -> None:
    initial = {"query.weight": torch.ones((2, 3), dtype=torch.bfloat16)}
    adapter = _FakeAdapter(initial)
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_adapter_module = adapter
    extension._tgvf_source_cache = {"source": object()}
    extension._tgvf_behavior_traces = {"trace": object()}

    with pytest.raises(ValueError, match="keys mismatch|shape/dtype"):
        extension.tgvf_update_adapter_owned_state(
            1,
            adapter_owned_state_sha256(invalid),
            _adapter_owned_state_to_utility_wire(invalid),
        )

    assert adapter.load_count == 0
    torch.testing.assert_close(adapter.state["query.weight"], initial["query.weight"])
    assert set(extension._tgvf_sources()) == {"source"}
    assert set(extension._tgvf_traces()) == {"trace"}


def test_worker_adapter_update_rejects_payload_digest_mismatch() -> None:
    state = {"query.weight": torch.ones((2, 3), dtype=torch.bfloat16)}
    adapter = _FakeAdapter(state)
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_adapter_module = adapter
    extension._tgvf_source_cache = {"source": object()}
    extension._tgvf_behavior_traces = {"trace": object()}

    with pytest.raises(IdentityMismatchError, match="update digest differs"):
        extension.tgvf_update_adapter_owned_state(
            1,
            "f" * 64,
            _adapter_owned_state_to_utility_wire(state),
        )

    assert adapter.load_count == 0
    assert set(extension._tgvf_sources()) == {"source"}
    assert set(extension._tgvf_traces()) == {"trace"}


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


def test_crop_tgvf_wire_binds_source_bbox_target_crop_visual_and_d() -> None:
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
            torch.full((8, 4), float(index), dtype=torch.bfloat16)
            for index in range(3)
        ),
        merged_main=torch.ones((2, 8), dtype=torch.bfloat16),
        merged_deepstack=tuple(
            torch.full((2, 8), float(index), dtype=torch.bfloat16)
            for index in range(3)
        ),
        image_grid_thw=(1, 2, 4),
        spatial_merge_size=2,
        decoded_rgb_sha256=crop_sha256,
    )
    expected = TGVFCropMaterializationResult(
        source_image_sha256="a" * 64,
        crop_sha256=crop_sha256,
        preprocessed_visual_sha256="c" * 64,
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

    assert restored.source_image_sha256 == expected.source_image_sha256
    assert restored.crop_sha256 == expected.crop_sha256
    assert restored.model_bbox_2d == expected.model_bbox_2d
    assert restored.target_token_ids == expected.target_token_ids
    torch.testing.assert_close(restored.crop_visual.premerge_main, crop_visual.premerge_main)
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


def test_worker_phase_boundary_quiesce_clears_tensors_and_reports_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    shared_source = torch.ones((4,), dtype=torch.float32)
    extension._tgvf_source_cache = {
        "trajectory-0": {
            "main": shared_source,
            "shared-alias": shared_source,
        }
    }
    extension._tgvf_behavior_traces = {
        "turn-0": _BehaviorTraceBuffer(
            prompt_length=2,
            capacity=3,
            hidden=torch.ones((3, 2), dtype=torch.float16),
            covered=bytearray((1, 0, 0)),
        )
    }
    released = False
    events: list[object] = []

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 20 if released else 100,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: 40 if released else 160,
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (700, 1000) if released else (400, 1000),
    )
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.vllm_tool_runtime.gc.collect",
        lambda: events.append("gc") or 7,
    )

    def empty_cache() -> None:
        nonlocal released
        events.append("empty_cache")
        released = True

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("sync", device))
    )

    report = extension.tgvf_quiesce_phase_boundary()

    assert extension._tgvf_source_cache == {}
    assert extension._tgvf_behavior_traces == {}
    assert events == ["gc", ("sync", 3), "empty_cache", ("sync", 3)]
    assert report == {
        "schema_version": TGVF_PHASE_BOUNDARY_QUIESCE_SCHEMA,
        "device": 3,
        "source_count_before": 1,
        "source_count_after": 0,
        "trace_count_before": 1,
        "trace_count_after": 0,
        "source_tensor_bytes_before": 16,
        "trace_tensor_bytes_before": 12,
        "tensor_bytes_before": 28,
        "tensor_bytes_after": 0,
        "gc_collected_objects": 7,
        "memory_before": {
            "allocated_bytes": 100,
            "reserved_bytes": 160,
            "free_bytes": 400,
            "total_bytes": 1000,
        },
        "memory_after": {
            "allocated_bytes": 20,
            "reserved_bytes": 40,
            "free_bytes": 700,
            "total_bytes": 1000,
        },
    }


def test_worker_phase_boundary_quiesce_clears_even_if_diagnostics_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_source_cache = {"trajectory-0": torch.ones((1,))}
    extension._tgvf_behavior_traces = {"turn-0": torch.ones((1,))}
    events: list[str] = []

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.vllm_tool_runtime._logical_tensor_bytes",
        lambda value: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )
    monkeypatch.setattr(
        "tgvf_rl.framework.verl.vllm_tool_runtime.gc.collect", lambda: 0
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))

    with pytest.raises(RuntimeError, match="telemetry failed"):
        extension.tgvf_quiesce_phase_boundary()

    assert extension._tgvf_source_cache == {}
    assert extension._tgvf_behavior_traces == {}
    assert events == ["empty"]


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


def test_worker_atomic_crop_tgvf_materializes_crop_vision_and_adapter_once() -> None:
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
            torch.full((8, 4), float(index), dtype=torch.bfloat16)
            for index in range(3)
        ),
        merged_main=torch.ones((2, 8), dtype=torch.bfloat16),
        merged_deepstack=tuple(
            torch.full((2, 8), float(index), dtype=torch.bfloat16)
            for index in range(3)
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
    calls: list[object] = []

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

    wire = extension.tgvf_materialize_crop_tgvf(
        "trajectory-0",
        "turn-0",
        1,
        _tensor_to_utility_wire(torch.ones((8, 6))),
        (1, 2, 4),
        source_sha256,
        crop_sha256,
        preprocessed_visual_identity_sha256(
            torch.ones((8, 6)),
            (1, 2, 4),
        ),
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
    assert result.model_bbox_2d == (10, 20, 800, 900)
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


def test_http_server_adapter_update_uses_collective_rpc_and_validates_ack() -> None:
    state = {"query.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    digest = adapter_owned_state_sha256(state)
    calls: list[tuple[str, dict[str, object]]] = []

    class Engine:
        async def collective_rpc(self, *, method: str, kwargs: dict[str, object]):
            calls.append((method, kwargs))
            restored = _adapter_owned_state_from_utility_wire(kwargs["state_wire"])
            torch.testing.assert_close(restored["query.weight"], state["query.weight"])
            return [_adapter_ack(4, digest, 1)]

    async def exercise() -> dict[str, object]:
        server_cls = _runtime_classes()[3]
        server = object.__new__(server_cls)
        server.global_steps = 4
        server.engine = Engine()
        return await server.tgvf_update_adapter_owned_state(
            optimizer_step=4,
            state_sha256=digest,
            state=state,
        )

    ack = asyncio.run(exercise())
    assert ack == _adapter_ack(4, digest, 1)
    assert calls[0][0] == "tgvf_update_adapter_owned_state"
    assert calls[0][1]["optimizer_step"] == 4
    assert calls[0][1]["state_sha256"] == digest


def test_http_server_phase_boundary_quiesce_uses_collective_rpc_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = [
        {
            "schema_version": TGVF_PHASE_BOUNDARY_QUIESCE_SCHEMA,
            "source_count_before": 2,
            "trace_count_before": 3,
        }
    ]
    calls: list[tuple[str, dict[str, object]]] = []

    class Engine:
        async def collective_rpc(self, *, method: str, kwargs: dict[str, object]):
            calls.append((method, kwargs))
            return expected

    async def exercise() -> object:
        server_cls = _runtime_classes()[3]
        server = object.__new__(server_cls)
        server.engine = Engine()
        return await server.tgvf_quiesce_phase_boundary()

    with caplog.at_level("INFO"):
        result = asyncio.run(exercise())

    assert result is expected
    assert calls == [("tgvf_quiesce_phase_boundary", {})]
    assert "source_count_before" in caplog.text


def test_replica_sleep_drains_then_quiesces_without_fail_closed_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []

    class RemoteMethod:
        def __init__(
            self, name: str, *, result: object = None, error: Exception | None = None
        ) -> None:
            self.name = name
            self.result = result
            self.error = error

        async def remote(self) -> object:
            events.append(self.name)
            if self.error is not None:
                raise self.error
            return self.result

    server0 = SimpleNamespace(
        wait_for_requests_to_drain=RemoteMethod("drain"),
        clear_kv_cache=RemoteMethod("clear-0"),
        tgvf_quiesce_phase_boundary=RemoteMethod(
            "quiesce-0", result=[{"source_count_before": 1}]
        ),
        sleep=RemoteMethod("sleep-0"),
    )
    server1 = SimpleNamespace(
        clear_kv_cache=RemoteMethod("clear-1"),
        tgvf_quiesce_phase_boundary=RemoteMethod(
            "quiesce-1", error=RuntimeError("diagnostic RPC failed")
        ),
        sleep=RemoteMethod("sleep-1"),
    )

    async def exercise() -> tuple[object, ...]:
        replica_cls = _runtime_classes()[2]
        replica = object.__new__(replica_cls)
        replica.servers = [server0, server1]
        return await replica.sleep()

    with caplog.at_level("WARNING"):
        reports = asyncio.run(exercise())

    assert events[0] == "drain"
    assert set(events[1:3]) == {"clear-0", "clear-1"}
    assert set(events[3:5]) == {"quiesce-0", "quiesce-1"}
    assert set(events[5:]) == {"sleep-0", "sleep-1"}
    assert reports[0] == [{"source_count_before": 1}]
    assert isinstance(reports[1], RuntimeError)
    assert "diagnostic RPC failed" in caplog.text


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    (
        ("stop", None),
        ("length", None),
        ("stop", "</tool_call>"),
    ),
    ids=("hidden-eos", "length", "tool-stop"),
)
def test_http_server_preserves_exact_vllm_termination_before_verl_collapse(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
    stop_reason: int | str | None,
) -> None:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class Engine:
        def generate(self, *args: object, **kwargs: object) -> object:
            del args, kwargs

            async def outputs():
                yield SimpleNamespace(
                    outputs=(
                        SimpleNamespace(
                            finish_reason=finish_reason,
                            stop_reason=stop_reason,
                        ),
                    )
                )

            return outputs()

    async def collapsed_upstream_generate(
        self: object,
        *,
        request_id: str,
        **kwargs: object,
    ) -> object:
        del kwargs
        final = None
        async for item in self.engine.generate(request_id=request_id):
            final = item
        assert final is not None
        return SimpleNamespace(
            stop_reason="completed",
            extra_fields={"global_steps": 3},
        )

    monkeypatch.setattr(vLLMHttpServer, "generate", collapsed_upstream_generate)

    async def exercise() -> object:
        server_cls = _runtime_classes()[3]
        server = object.__new__(server_cls)
        server.engine = Engine()
        return await server.generate(
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 16},
            request_id="backend-request-0",
        )

    output = asyncio.run(exercise())
    assert output.extra_fields[TGVF_VLLM_FINISH_REASON_FIELD] == finish_reason
    assert output.extra_fields[TGVF_VLLM_STOP_REASON_FIELD] == stop_reason


def test_client_and_manager_adapter_update_entries_validate_every_server_ack() -> None:
    state = {"query.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    digest = adapter_owned_state_sha256(state)
    calls: list[tuple[str, int]] = []

    class RemoteMethod:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def remote(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["state"] is state
            assert kwargs["state_sha256"] == digest
            calls.append((self.server_name, int(kwargs["optimizer_step"])))
            return _adapter_ack(9, digest, 1)

    async def exercise():
        manager_cls, client_cls, _replica_cls, _server_cls = _runtime_classes()
        client_server = SimpleNamespace(
            tgvf_update_adapter_owned_state=RemoteMethod("client")
        )
        client = object.__new__(client_cls)
        client._tgvf_routes = {"request": ("client-server", client_server)}
        client_ack = await client.update_adapter_owned_state(
            request_id="request",
            optimizer_step=9,
            state_sha256=digest,
            state=state,
        )

        manager_servers = tuple(
            SimpleNamespace(tgvf_update_adapter_owned_state=RemoteMethod(name))
            for name in ("server-0", "server-1")
        )
        manager = object.__new__(manager_cls)
        manager.rollout_replicas = [SimpleNamespace(servers=manager_servers)]
        manager_acks = await manager.update_adapter_owned_state(
            optimizer_step=9,
            state_sha256=digest,
            state=state,
        )
        return client_ack, manager_acks

    client_ack, manager_acks = asyncio.run(exercise())
    assert client_ack == _adapter_ack(9, digest, 1)
    assert manager_acks == (
        _adapter_ack(9, digest, 1),
        _adapter_ack(9, digest, 1),
    )
    assert calls == [("client", 9), ("server-0", 9), ("server-1", 9)]


def test_adapter_update_manager_binds_existing_rollout_replicas() -> None:
    replicas = [SimpleNamespace(servers=(object(),))]

    manager = bind_tgvf_adapter_state_update_manager(replicas)

    assert manager.rollout_replicas == replicas
    assert manager.rollout_replicas is not replicas


def test_manager_adapter_update_rejects_one_divergent_ack() -> None:
    state = {"query.weight": torch.ones((1,), dtype=torch.float32)}
    digest = adapter_owned_state_sha256(state)

    class RemoteMethod:
        async def remote(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return _adapter_ack(5, "f" * 64, 1)

    async def exercise() -> None:
        manager_cls = _runtime_classes()[0]
        manager = object.__new__(manager_cls)
        manager.rollout_replicas = [
            SimpleNamespace(
                servers=(
                    SimpleNamespace(tgvf_update_adapter_owned_state=RemoteMethod()),
                )
            )
        ]
        await manager.update_adapter_owned_state(
            optimizer_step=5,
            state_sha256=digest,
            state=state,
        )

    with pytest.raises(RuntimeError, match="ACK state digest differs"):
        asyncio.run(exercise())


def test_vllm_server_shutdown_stops_http_engine_and_bound_sockets() -> None:
    events: list[str] = []

    class Engine:
        output_handler = None
        engine_core = None

        def shutdown(self):
            events.append("engine")

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

        async def core_reader(name: str) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append(name)

        engine.output_handler = asyncio.create_task(output_handler())
        resources = SimpleNamespace(
            output_queue_task=asyncio.create_task(core_reader("core-output")),
            stats_update_task=asyncio.create_task(core_reader("core-stats")),
        )
        engine.engine_core = SimpleNamespace(resources=resources)
        server.engine = engine
        server._server_task = asyncio.create_task(asyncio.Event().wait())
        server._master_sock = Socket("master")
        server._dp_rpc_sock = Socket("rpc")
        server._dp_master_sock = Socket("dp-master")
        await server.tgvf_shutdown()
        return server, engine, resources

    server, engine, resources = asyncio.run(exercise())
    assert set(events[:3]) == {"handler", "core-output", "core-stats"}
    assert events[3:] == ["engine", "master", "rpc", "dp-master"]
    assert server.engine is None
    assert engine.output_handler is None
    assert engine.engine_core is None
    assert resources.output_queue_task is None
    assert resources.stats_update_task is None
    assert server._master_sock is None
