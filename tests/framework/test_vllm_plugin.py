from __future__ import annotations

import inspect

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.framework.vllm import (
    SUPPORTED_VLLM_VERSION,
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    VLLMPublicPluginAPI,
    pack_qwen3_vllm_replay,
    register_tgvf_qwen3_vllm_plugin,
)
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


def _recorded_replay(*, branch_layers: tuple[int, ...] = (8, 16, 24)):
    store = ObservationStore()
    model = ModelIdentity("qwen3_vl", "tiny", "/tiny", 32, "1" * 64)
    policy = PolicyVersion("run", 0, "2" * 64)
    representation = ArtifactIdentity("tgvf", "adapter", "tiny", "3" * 64)
    merger_ids = tuple(
        ArtifactIdentity("qwen", f"merger-{layer}", "tiny", f"{4 + i}" * 64)
        for i, layer in enumerate(branch_layers)
    )
    source_premerge = store.put_tensor(
        "source.premerge.main", torch.arange(32, dtype=torch.float32).view(16, 2)
    )
    source_premerge_branches = tuple(
        store.put_tensor(
            f"source.premerge.branch.{layer}",
            torch.full((16, 2), float(i + 1)),
        )
        for i, layer in enumerate(branch_layers)
    )
    source_main = store.put_tensor(
        "source.merged.main", torch.arange(8, dtype=torch.float32).view(4, 2)
    )
    source_branches = tuple(
        store.put_tensor(
            f"source.merged.branch.{layer}",
            torch.full((4, 2), float(10 + i)),
        )
        for i, layer in enumerate(branch_layers)
    )
    sequence = 16
    mask = store.put_tensor("replay.mask", torch.ones(1, sequence, dtype=torch.bool))
    position_ids = store.put_tensor(
        "replay.position_ids", torch.arange(sequence).view(1, sequence)
    )
    handles = []
    for call_index, d_positions in enumerate(((6, 7, 8, 9), (11, 12, 13, 14))):
        main_d = store.put_tensor(
            f"call.{call_index}.main_d",
            torch.full((4, 2), float(20 + call_index)),
        )
        d_branches = tuple(
            store.put_tensor(
                f"call.{call_index}.branch.{layer}",
                torch.full((4, 2), float(30 + 3 * call_index + index)),
            )
            for index, layer in enumerate(branch_layers)
        )
        observation_positions = store.put_tensor(
            f"call.{call_index}.positions",
            torch.arange(sequence).view(1, sequence),
        )
        record = FocusedObservationRecord(
            schema_version="focused-observation-v1",
            observation_id=f"observation-{call_index}",
            call_index=call_index,
            model=model,
            representation=representation,
            condition=ConditionProvenance(
                provider="contextual_hidden_state",
                sampled_target_text_sha256="8" * 64,
                sampled_target_token_start=1,
                sampled_target_token_end=2,
                source_input_ids_sha256="9" * 64,
                trajectory_ids=("trajectory",),
                call_indices=(call_index,),
                hidden_layer=1,
                policy_version=policy,
            ),
            source_visual=SourceVisualState(
                image_sha256="a" * 64,
                premerge_main=source_premerge,
                premerge_deepstack=source_premerge_branches,
                merged_main=source_main,
                merged_deepstack=source_branches,
                image_grid_thw=(1, 4, 4),
                spatial_merge_size=2,
            ),
            payload=TensorPayloadSet(
                main_d=main_d,
                deepstack=d_branches,
                position_ids=observation_positions,
                attention_mask=mask,
            ),
            branches=tuple(
                DeepStackBranchRecord(layer, ref, d_positions, merger)
                for layer, ref, merger in zip(
                    branch_layers, d_branches, merger_ids, strict=True
                )
            ),
            layout=VisualLayout(
                sequence_length=sequence,
                original_image_positions=(1, 2, 3, 4),
                d_positions=d_positions,
                deepstack_branch_layers=branch_layers,
                deepstack_injection_positions=tuple(d_positions for _ in branch_layers),
            ),
            masks=ObservationMasks(mask, mask, mask, None),
            cache=CacheContract("no_cache", 0, None, None, True, 0.0),
        )
        handles.append(store.put(record))

    input_ids = store.put_tensor(
        "replay.input_ids", torch.arange(sequence).view(1, sequence)
    )
    replay = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="replay",
        trajectory_id="trajectory",
        model=model,
        behavior_policy=policy,
        observation_handles=tuple(handles),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=mask,
            policy_attention_mask=mask,
            reference_attention_mask=mask,
            teacher_attention_mask=mask,
        ),
    )
    return store, store.put_replay(replay)


def test_packer_emits_source_then_each_call_with_main_plus_three_branches() -> None:
    store, replay = _recorded_replay()
    packed = pack_qwen3_vllm_replay(store, replay)

    assert tuple((item.kind, item.call_index) for item in packed.items) == (
        ("source_image", None),
        ("focused_d", 0),
        ("focused_d", 1),
    )
    assert all(item.image_embeds.shape == (4, 8) for item in packed.items)
    assert all(item.image_grid_thw == (1, 4, 4) for item in packed.items)
    mm_data = packed.as_vllm_multi_modal_data()
    assert len(mm_data["image"]) == 3
    assert tuple(item["image_embeds"].shape for item in mm_data["image"]) == (
        (4, 8),
        (4, 8),
        (4, 8),
    )
    assert torch.equal(
        torch.cat([item["image_grid_thw"] for item in mm_data["image"]]),
        torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]]),
    )
    assert len(packed.image_uuids) == 3
    assert len(set(packed.image_uuids)) == 3

    expected_source = torch.cat(
        (
            store.resolve_verified(
                store.resolve_record(
                    store.resolve_replay(replay).observation_handles[0]
                ).source_visual.merged_main
            ),
            *(
                store.resolve_verified(ref)
                for ref in store.resolve_record(
                    store.resolve_replay(replay).observation_handles[0]
                ).source_visual.merged_deepstack
            ),
        ),
        dim=-1,
    )
    torch.testing.assert_close(packed.items[0].image_embeds, expected_source)


def test_packer_fails_closed_on_wrong_layers_and_post_pack_mutation() -> None:
    bad_store, bad_replay = _recorded_replay(branch_layers=(8, 16, 25))
    with pytest.raises(ReplayMismatchError, match="branch order/layers"):
        pack_qwen3_vllm_replay(bad_store, bad_replay)

    store, replay = _recorded_replay()
    packed = pack_qwen3_vllm_replay(store, replay)
    packed.items[0].image_embeds.add_(1)
    with pytest.raises(ReplayMismatchError, match="checksum changed"):
        packed.as_vllm_multi_modal_data()


def test_public_registration_calls_both_general_vllm_registries() -> None:
    calls: list[tuple[object, ...]] = []

    class Model:
        pass

    class Processor:
        pass

    class Info:
        pass

    class Dummy:
        pass

    class ModelRegistry:
        @staticmethod
        def register_model(name, cls):
            calls.append(("model", name, cls))

    class MultiModalRegistry:
        @staticmethod
        def register_processor(processor, *, info, dummy_inputs):
            calls.append(("processor", processor, info, dummy_inputs))
            return lambda cls: cls

    registration = register_tgvf_qwen3_vllm_plugin(
        api=VLLMPublicPluginAPI(
            model_registry=ModelRegistry,
            multimodal_registry=MultiModalRegistry,
            model_cls=Model,
            processor_cls=Processor,
            processing_info_cls=Info,
            dummy_inputs_cls=Dummy,
            version=SUPPORTED_VLLM_VERSION,
        )
    )
    assert registration.architecture == TGVF_QWEN3_VLLM_ARCHITECTURE
    assert calls == [
        ("processor", Processor, Info, Dummy),
        ("model", TGVF_QWEN3_VLLM_ARCHITECTURE, Model),
    ]


def test_vllm_012_parser_cpu_probe_and_live_registry(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    from tgvf_rl.framework.vllm.qwen3_plugin import TGVFQwen3VLDataParser

    store, replay = _recorded_replay()
    packed = pack_qwen3_vllm_replay(store, replay)
    parser = TGVFQwen3VLDataParser(2, video_needs_metadata=True)
    parsed = parser.parse_mm_data(packed.as_vllm_multi_modal_data())
    image_items = parsed["image"]
    assert len(image_items) == 3
    assert tuple(image_items[index]["image_embeds"].shape for index in range(3)) == (
        (4, 8),
        (4, 8),
        (4, 8),
    )

    registration = register_tgvf_qwen3_vllm_plugin()
    assert registration.version == "0.12.0"
    assert TGVF_QWEN3_VLLM_ARCHITECTURE in ModelRegistry.get_supported_archs()


def test_precomputed_parser_rejects_premerge_row_semantics(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("vllm")
    from tgvf_rl.framework.vllm.qwen3_plugin import TGVFQwen3VLDataParser

    parser = TGVFQwen3VLDataParser(2)
    with pytest.raises(ValueError, match="rows do not match"):
        parser.parse_mm_data(
            {
                "image": {
                    "image_embeds": torch.zeros(16, 8),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }
            }
        )


def test_live_plugin_preserves_vllm_new_style_model_signature() -> None:
    pytest.importorskip("vllm")
    from tgvf_rl.framework.vllm.qwen3_plugin import (
        TGVFQwen3VLForConditionalGeneration,
    )

    signature = inspect.signature(TGVFQwen3VLForConditionalGeneration)
    assert tuple(signature.parameters) == ("vllm_config", "prefix")
    assert signature.parameters["vllm_config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["prefix"].default == "model"
