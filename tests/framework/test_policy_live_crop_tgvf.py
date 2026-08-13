from __future__ import annotations

import asyncio
from threading import Thread
from types import SimpleNamespace

import torch

from tgvf_rl.environment.crop_tgvf_tool import PreparedCropTGVFInput
from tgvf_rl.framework.verl.policy_live_runtime import _RemoteCropTGVFToolRuntime
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVFCropTGVFMaterializationResult,
    _adapter_payload_to_cpu,
)
from tgvf_rl.observations.schema import CropTGVFObservationRecord
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.qwen.crop_coordinates import CanonicalSourcePixelCropCoordinateMapper
from tgvf_rl.representation import TGVFAdapterInput

from tests.environment.test_crop_tgvf_runtime import _fixture


def test_remote_crop_tgvf_uses_immutable_crop_and_single_atomic_rpc(
    tmp_path, monkeypatch
) -> None:
    local, materializer, store, pixels, _embedding, context, parsed = _fixture(tmp_path)
    processor_calls: list[tuple[torch.Tensor, int]] = []

    def preprocess(*, processor, rgb, image_max_pixels):
        del processor
        processor_calls.append((rgb.clone(), image_max_pixels))
        return torch.ones((4, 6)), torch.tensor(((1, 2, 2),))

    monkeypatch.setattr(
        "tgvf_rl.framework.verl.policy_live_runtime.preprocess_qwen3_rgb",
        preprocess,
    )
    rpc_calls: list[dict[str, object]] = []

    class Server:
        async def materialize_crop_tgvf(self, **kwargs):
            rpc_calls.append(kwargs)
            crop = pixels[1:4, 0:4, :].contiguous()
            visual = materializer.materialize_source_visual(
                crop,
                parsed_call=parsed,
                call_index=0,
            )
            target_count = len(parsed.target_span.token_ids)
            hq = torch.arange(target_count * 8, dtype=torch.float32).reshape(
                target_count, 8
            )
            with torch.inference_mode():
                output = local.loaded_adapter.adapter(
                    TGVFAdapterInput(
                        target_hidden_states=hq,
                        pre_merge_visual_tokens=visual.premerge_main,
                        deepstack_pre_merge_visual_tokens=visual.premerge_deepstack,
                    )
                )
            return TGVFCropTGVFMaterializationResult(
                crop_visual=visual,
                hq=hq,
                observation=_adapter_payload_to_cpu(output),
                trajectory_id=context.trajectory_identity.canonical_id,
                call_index=0,
                source_image_sha256=tensor_checksum(pixels),
                crop_rgb_sha256=tensor_checksum(crop),
                source_width=5,
                source_height=4,
                crop_bbox=(0, 1, 4, 4),
                crop_width=4,
                crop_height=3,
            )

    binding = local.loaded_adapter.binding
    config = SimpleNamespace(
        representation=SimpleNamespace(
            artifact=binding.artifact,
            conditioning=binding.conditioning,
        )
    )
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        remote = _RemoteCropTGVFToolRuntime(
            event_loop=loop,
            server_client=Server(),
            processor=object(),
            config=config,
            image_max_pixels=512 * 512,
            layout_builder=local.layout_builder,
            observation_store=store,
            execution_ledger=local.execution_ledger,
            contextual_forward_identity=None,
            branch_merger_identities=local.branch_merger_identities,
            crop_processor_identity=local.crop_processor_identity,
            crop_layout_identity=local.crop_layout_identity,
            coordinate_mapper=CanonicalSourcePixelCropCoordinateMapper(),
        )
        handle = remote.execute(parsed, context)
        repeated = remote.execute(parsed, context)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert repeated == handle
    assert len(rpc_calls) == 1
    assert len(processor_calls) == 1
    prepared_rgb, cap = processor_calls[0]
    assert cap == 512 * 512
    torch.testing.assert_close(prepared_rgb, pixels[1:4, 0:4, :], rtol=0, atol=0)
    call = rpc_calls[0]
    assert call["source_image_sha256"] == tensor_checksum(pixels)
    assert call["crop_rgb_sha256"] == tensor_checksum(prepared_rgb)
    assert call["crop_bbox"] == (0, 1, 4, 4)
    assert call["crop_width"] == 4
    assert call["crop_height"] == 3
    assert call["sampled_output_ids"] == context.sampled_turn.token_ids
    assert call["expected_target_token_ids"] == parsed.target_span.token_ids
    record = store.resolve_record(handle)
    assert isinstance(record, CropTGVFObservationRecord)
    assert record.source_pixels_sha256 == tensor_checksum(pixels)
    assert record.effective_bbox_2d == (0, 1, 4, 4)


def test_remote_crop_tgvf_rejects_rpc_audit_drift(tmp_path) -> None:
    local, _materializer, _store, pixels, _embedding, context, _parsed = _fixture(
        tmp_path
    )
    crop = pixels[1:4, 0:4, :].contiguous()
    prepared = local.atomic_tool.prepare_precomputed_crop(
        trajectory_id=context.trajectory_identity.canonical_id,
        call_index=0,
        parsed_call=_parsed,
        trajectory_source_visual=context.trajectory_source_visual,
    )
    assert isinstance(prepared, PreparedCropTGVFInput)
    visual = local.atomic_tool.materializer.materialize_source_visual(
        crop, parsed_call=_parsed, call_index=0
    )
    target_count = len(_parsed.target_span.token_ids)
    hq = torch.ones((target_count, 8))
    with torch.inference_mode():
        output = local.loaded_adapter.adapter(
            TGVFAdapterInput(
                target_hidden_states=hq,
                pre_merge_visual_tokens=visual.premerge_main,
                deepstack_pre_merge_visual_tokens=visual.premerge_deepstack,
            )
        )
    drifted = TGVFCropTGVFMaterializationResult(
        crop_visual=visual,
        hq=hq,
        observation=_adapter_payload_to_cpu(output),
        trajectory_id=prepared.trajectory_id,
        call_index=prepared.call_index,
        source_image_sha256="f" * 64,
        crop_rgb_sha256=tensor_checksum(crop),
        source_width=prepared.source_width,
        source_height=prepared.source_height,
        crop_bbox=prepared.effective_bbox_2d,
        crop_width=4,
        crop_height=3,
    )

    try:
        _RemoteCropTGVFToolRuntime._validate_materialized(
            drifted,
            prepared=prepared,
            target_token_count=target_count,
        )
    except Exception as error:
        assert "RPC audit differs" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("drifted worker audit was accepted")
