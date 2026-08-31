from __future__ import annotations

from dataclasses import replace
import inspect
from importlib import metadata
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.framework.vllm import (
    SUPPORTED_VLLM_VERSIONS,
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    VLLM_012_LORA_PDL_MODE,
    VLLM_UPSTREAM_LORA_PDL_MODE,
    Qwen3VLLMObservationPayloadResolver,
    VLLMCompatibilityError,
    VLLMPublicPluginAPI,
    load_vllm_public_plugin_api,
    install_vllm_lora_pdl_compatibility,
    install_verl_preexpanded_prompt_compatibility,
    pack_qwen3_vllm_replay,
    pack_qwen3_vllm_replay_bundle,
    register_tgvf_qwen3_vllm_plugin,
)
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


def _recorded_replay(*, branch_layers: tuple[int, ...] = (8, 16, 24), calls: int = 2):
    if calls not in {0, 1, 2}:
        raise ValueError("fixture supports zero, one, or two calls")
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
    source_state = SourceVisualState(
        image_sha256="a" * 64,
        premerge_main=source_premerge,
        premerge_deepstack=source_premerge_branches,
        merged_main=source_main,
        merged_deepstack=source_branches,
        image_grid_thw=(1, 4, 4),
        spatial_merge_size=2,
    )
    sequence = 16
    mask = store.put_tensor("replay.mask", torch.ones(1, sequence, dtype=torch.bool))
    position_ids = store.put_tensor(
        "replay.position_ids", torch.arange(sequence).view(1, sequence)
    )
    handles = []
    for call_index, d_positions in enumerate(((6, 7, 8, 9), (11, 12, 13, 14))[:calls]):
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
                conditioning_target_token_start=1,
                conditioning_target_token_end=2,
                source_sequence_length=sequence,
                source_input_ids_sha256="9" * 64,
                trajectory_ids=("trajectory",),
                call_indices=(call_index,),
                hidden_layer=1,
                contextual_forward_identity=ArtifactIdentity(
                    "policy", "contextual-forward", "fixture", "7" * 64
                ),
                policy_version=policy,
            ),
            source_visual=source_state,
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
        replay_id=f"replay-{calls}",
        trajectory_id="trajectory",
        model=model,
        behavior_policy=policy,
        source_visual=TrajectorySourceVisual(
            state=source_state,
            positions=(1, 2, 3, 4),
            deepstack_branch_layers=branch_layers,
            deepstack_injection_positions=tuple((1, 2, 3, 4) for _ in branch_layers),
        ),
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
                store.resolve_replay(replay).source_visual.state.merged_main
            ),
            *(
                store.resolve_verified(ref)
                for ref in store.resolve_replay(
                    replay
                ).source_visual.state.merged_deepstack
            ),
        ),
        dim=-1,
    )
    torch.testing.assert_close(packed.items[0].image_embeds, expected_source)


def test_live_resolver_appends_one_exact_recorded_observation() -> None:
    store, replay = _recorded_replay(calls=1)
    observation = store.resolve_replay(replay).observation_handles[0]
    resolver = Qwen3VLLMObservationPayloadResolver(
        store=store,
        include_multi_modal_uuid=True,
    )

    resolved = resolver.resolve(observation, call_index=0)

    assert resolved.observation == observation
    assert resolved.call_index == 0
    assert resolved.modality == "image"
    assert resolved.multi_modal_uuid == resolved.payload_sha256
    item = resolved.multi_modal_data_item
    assert item["image_embeds"].shape == (4, 8)
    assert torch.equal(item["image_grid_thw"], torch.tensor(((1, 4, 4),)))


def test_vllm_worker_packs_the_transported_replay_bundle_without_recompute() -> None:
    store, replay = _recorded_replay()
    direct = pack_qwen3_vllm_replay(store, replay)
    transported = pack_qwen3_vllm_replay_bundle(store.export_replay_bundle(replay))

    assert transported.replay_handle == direct.replay_handle
    assert transported.image_uuids == direct.image_uuids
    torch.testing.assert_close(
        transported.image_embeds, direct.image_embeds, rtol=0, atol=0
    )


def test_vllm_zero_call_replay_packs_only_the_mandatory_source() -> None:
    store, replay = _recorded_replay(calls=0)
    assert store.resolve_replay(replay).observation_handles == ()

    packed = pack_qwen3_vllm_replay_bundle(store.export_replay_bundle(replay))

    assert tuple((item.kind, item.call_index) for item in packed.items) == (
        ("source_image", None),
    )
    assert len(packed.as_vllm_multi_modal_data()["image"]) == 1


def test_multi_call_replay_rejects_observation_source_identity_mismatch() -> None:
    store, replay_handle = _recorded_replay()
    replay = store.resolve_replay(replay_handle)
    second = store.resolve_record(replay.observation_handles[1])
    mismatched = replace(
        second,
        observation_id="observation-1-mismatched-source",
        source_visual=replace(second.source_visual, image_sha256="b" * 64),
    )
    mismatched_handle = store.put(mismatched)

    with pytest.raises(ReplayMismatchError, match="trajectory source"):
        store.put_replay(
            replace(
                replay,
                replay_id="replay-mismatched-source",
                observation_handles=(
                    replay.observation_handles[0],
                    mismatched_handle,
                ),
            )
        )


def test_packer_fails_closed_on_wrong_layers_and_post_pack_mutation() -> None:
    bad_store, bad_replay = _recorded_replay(branch_layers=(8, 16, 25))
    with pytest.raises(ReplayMismatchError, match="branch order/layers"):
        pack_qwen3_vllm_replay(bad_store, bad_replay)

    store, replay = _recorded_replay()
    packed = pack_qwen3_vllm_replay(store, replay)
    packed.items[0].image_embeds.add_(1)
    with pytest.raises(ReplayMismatchError, match="checksum changed"):
        packed.as_vllm_multi_modal_data()


@pytest.mark.parametrize("version", sorted(SUPPORTED_VLLM_VERSIONS))
def test_public_registration_calls_both_general_vllm_registries(
    version: str,
) -> None:
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
            version=version,
        )
    )
    assert registration.architecture == TGVF_QWEN3_VLLM_ARCHITECTURE
    assert registration.version == version
    assert calls == [
        ("processor", Processor, Info, Dummy),
        ("model", TGVF_QWEN3_VLLM_ARCHITECTURE, Model),
    ]


def test_verl_preexpanded_prompt_patch_preserves_the_complete_token_run() -> None:
    def dedup(prompt_ids, processor):
        del processor
        return prompt_ids[:2]

    utils = SimpleNamespace(qwen2_5_vl_dedup_image_tokens=dedup)
    server = SimpleNamespace(qwen2_5_vl_dedup_image_tokens=dedup)
    modules = {
        "verl.workers.rollout.utils": utils,
        "verl.workers.rollout.vllm_rollout.vllm_async_server": server,
    }

    install_verl_preexpanded_prompt_compatibility(importer=modules.__getitem__)

    prompt = [1, 2, 2, 2, 3]
    assert server.qwen2_5_vl_dedup_image_tokens(prompt, object()) is prompt
    assert utils.qwen2_5_vl_dedup_image_tokens is server.qwen2_5_vl_dedup_image_tokens


def test_public_registration_rejects_an_unaudited_neighbor_build() -> None:
    with pytest.raises(VLLMCompatibilityError, match="unsupported version"):
        register_tgvf_qwen3_vllm_plugin(
            api=VLLMPublicPluginAPI(
                model_registry=object(),
                multimodal_registry=object(),
                model_cls=object,
                processor_cls=object,
                processing_info_cls=object,
                dummy_inputs_cls=object,
                version="0.23.0",
            )
        )


def test_vllm_012_lora_pdl_compatibility_disables_every_imported_alias() -> None:
    modules = {
        name: SimpleNamespace(supports_pdl=lambda _device=None: True)
        for name in (
            "vllm.lora.ops.triton_ops.utils",
            "vllm.lora.ops.triton_ops.lora_expand_op",
            "vllm.lora.ops.triton_ops.lora_shrink_op",
            "vllm.lora.ops.triton_ops.fused_moe_lora_op",
        )
    }

    mode = install_vllm_lora_pdl_compatibility(
        vllm_version="0.12.0",
        triton_version="3.5.0",
        importer=modules.__getitem__,
    )

    assert mode == VLLM_012_LORA_PDL_MODE
    assert all(module.supports_pdl(None) is False for module in modules.values())


def test_vllm_lora_pdl_compatibility_is_version_locked() -> None:
    with pytest.raises(VLLMCompatibilityError, match="exact Triton 3.5.0"):
        install_vllm_lora_pdl_compatibility(
            vllm_version="0.12.0",
            triton_version="3.5.1",
            importer=lambda _name: SimpleNamespace(supports_pdl=lambda: True),
        )
    assert (
        install_vllm_lora_pdl_compatibility(
            vllm_version="0.23.0+cu129",
            triton_version="3.5.1",
            importer=lambda _name: pytest.fail("newer vLLM must remain upstream"),
        )
        == VLLM_UPSTREAM_LORA_PDL_MODE
    )


def test_processing_info_selects_tgvf_parser_for_three_latent_items(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("vllm")
    from tgvf_rl.framework.vllm.qwen3_plugin import (
        TGVFQwen3VLDataParser,
        TGVFQwen3VLProcessingInfo,
    )

    class ModelConfig:
        @staticmethod
        def get_multimodal_config():
            return SimpleNamespace(enable_mm_embeds=True)

        @staticmethod
        def get_inputs_embeds_size():
            return 8

    class Context:
        model_config = ModelConfig()

        @staticmethod
        def get_hf_config(_expected_type):
            return SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))

    info = TGVFQwen3VLProcessingInfo(Context())
    parser = info.get_data_parser()
    assert isinstance(parser, TGVFQwen3VLDataParser)

    store, replay = _recorded_replay()
    packed = pack_qwen3_vllm_replay(store, replay)
    parsed = parser.parse_mm_data(packed.as_vllm_multi_modal_data())
    image_items = parsed["image"]
    assert len(image_items) == 3
    assert tuple(image_items[index]["image_embeds"].shape for index in range(3)) == (
        (4, 8),
        (4, 8),
        (4, 8),
    )


def test_supported_vllm_parser_cpu_probe_and_live_registry(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    from tgvf_rl.framework.vllm.qwen3_plugin import (
        TGVFQwen3VLDataParser,
        TGVFQwen3VLProcessingInfo,
    )

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

    public_api = load_vllm_public_plugin_api()
    assert public_api.processing_info_cls is TGVFQwen3VLProcessingInfo
    registration = register_tgvf_qwen3_vllm_plugin(api=public_api)
    assert registration.version == metadata.version("vllm")
    assert registration.version in SUPPORTED_VLLM_VERSIONS
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


@pytest.mark.parametrize("enabled", (True, False))
def test_vllm_native_deepstack_control_preserves_shape_and_controls_values(
    enabled: bool,
) -> None:
    from tgvf_rl.framework.vllm.qwen3_plugin import (
        apply_native_deepstack_tensor_control,
    )

    source = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    observed = apply_native_deepstack_tensor_control(source, enabled=enabled)

    assert observed.shape == source.shape
    assert observed.dtype == source.dtype
    if enabled:
        assert observed is source
    else:
        assert torch.count_nonzero(observed).item() == 0
