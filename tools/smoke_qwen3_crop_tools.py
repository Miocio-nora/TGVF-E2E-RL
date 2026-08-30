#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Run one real-Qwen plain-crop and atomic crop+TGVF implementation smoke."""

from __future__ import annotations

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/smoke_qwen3_crop_tools.py",
        ),
    )

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from tgvf_rl.conditioning import (
    TargetConditioningProviderKind,
)
from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    SamplingIdentity,
    TokenSpan,
)
from tgvf_rl.environment.agent_loop import SampledPolicyTurn, ToolExecutionContext
from tgvf_rl.environment.adapter_runtime import (
    BehaviorHiddenStateMaterialization,
    BranchMergerRuntimeBinding,
    FrozenBehaviorHiddenStateCapturePort,
    RepresentationArtifactRuntimeBinding,
    load_frozen_tgvf_adapter,
)
from tgvf_rl.environment.crop_runtime import CropExecutionLedger, ImageZoomInToolRuntime
from tgvf_rl.environment.crop_tgvf_runtime import AtomicCropTGVFToolRuntime
from tgvf_rl.environment.crop_tgvf_tool import AtomicCropTGVFTool
from tgvf_rl.environment.focus_runtime import FocusExecutionLedger
from tgvf_rl.environment.native_appender import (
    NativeSuccessObservationContract,
    QWEN_NATIVE_IMAGE_PLACEHOLDER,
)
from tgvf_rl.environment.qwen3_crop_materializer import Qwen3CropVisualMaterializer
from tgvf_rl.environment.qwen3_tool_layout import Qwen3NativeToolLayoutBuilder
from tgvf_rl.environment.source_visual import record_trajectory_source_visual
from tgvf_rl.framework.vllm import Qwen3VLLMObservationPayloadResolver
from tgvf_rl.observations.schema import CropObservationRecord, CropTGVFObservationRecord
from tgvf_rl.observations.store import ObservationStore
from tgvf_rl.protocol.parser import StrictToolCallParser
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.protocol.observation_contract import (
    NativeSuccessObservationProtocolId,
)
from tgvf_rl.protocol.schema import (
    TGVF_CROP_TOOL_NAME,
    IMAGE_ZOOM_IN_TOOL_NAME,
    SampledAssistantTurn,
    TokenByteSpan,
    NativeToolCapabilityProfile,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.runtime import (
    ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256,
    ACCEPTED_QWEN3_MODEL_PATH,
    ACCEPTED_QWEN3_TOKENIZER_LENGTH,
    create_qwen3_representation_runtime,
)
from tgvf_rl.qwen import InjectedForwardRequest, InjectedVisualBlock, Qwen3VLAdapter
from tgvf_rl.trajectories.schema import TrajectoryIdentity
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


MODEL_NAME = "Qwen3-VL-8B-Thinking"
BRANCH_LAYERS = (8, 16, 24)
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-max-pixels", type=int, default=512 * 512)
    return parser.parse_args()


def _artifact(name: str, payload: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        namespace="tool-smoke",
        name=name,
        version="v1",
        sha256=sha256(payload.encode("utf-8")).hexdigest(),
    )


def _tokenize_sampled_turn(tokenizer: Any, text: str) -> SampledAssistantTurn:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple(
        tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]
    )
    if len(token_ids) != len(offsets):
        raise ValueError("sampled tool text token/offset counts differ")
    byte_boundaries = [0]
    for character in text:
        byte_boundaries.append(byte_boundaries[-1] + len(character.encode("utf-8")))
    spans: list[TokenByteSpan] = []
    cursor = 0
    for index, (token_id, (start, end)) in enumerate(
        zip(token_ids, offsets, strict=True)
    ):
        if start != cursor or end <= start:
            raise ValueError("sampled tool offsets are not exact and contiguous")
        spans.append(
            TokenByteSpan(
                token_index=index,
                token_id=token_id,
                byte_start=byte_boundaries[start],
                byte_end=byte_boundaries[end],
            )
        )
        cursor = end
    if cursor != len(text):
        raise ValueError("sampled tool offsets do not cover the complete text")
    return SampledAssistantTurn(text, token_ids, tuple(spans))


def _sampled_policy_turn(
    tokenizer: Any,
    text: str,
    policy: PolicyVersion,
) -> tuple[SampledPolicyTurn, object]:
    parser_turn = _tokenize_sampled_turn(tokenizer, text)
    sampled = SampledPolicyTurn(
        text=text,
        token_ids=parser_turn.token_ids,
        token_byte_spans=parser_turn.token_byte_spans,
        behavior_logprobs=tuple(-0.1 for _ in parser_turn.token_ids),
        sampling=SamplingIdentity(
            policy_version=policy,
            backend="vllm",
            backend_version="0.12.0",
            seed=42,
            rng_state_sha256=ONE_SHA,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            logit_processors=(),
            measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            asynchronous_staleness_steps=0,
        ),
        think_token_span=TokenSpan(0, 1),
        stop_reason="tool_call",
        backend_request_sha256=ZERO_SHA,
        backend_response_sha256=ONE_SHA,
    )
    parsed = StrictToolCallParser().parse(sampled.parser_turn())
    return sampled, parsed


def _synthetic_rgb() -> torch.Tensor:
    height = width = 256
    y = torch.arange(height, dtype=torch.int32).view(height, 1)
    x = torch.arange(width, dtype=torch.int32).view(1, width)
    red = ((x + y) % 256).to(torch.uint8).expand(height, width)
    green = ((2 * x + y) % 256).to(torch.uint8).expand(height, width)
    blue = ((x + 3 * y) % 256).to(torch.uint8).expand(height, width)
    return torch.stack((red, green, blue), dim=-1).contiguous()


def _expand_source_prompt(
    prompt_ids: tuple[int, ...],
    *,
    image_pad_id: int,
    merged_token_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pad_indices = tuple(
        index for index, token_id in enumerate(prompt_ids) if token_id == image_pad_id
    )
    if len(pad_indices) != 1:
        raise ValueError("smoke prompt must contain one canonical source placeholder")
    start = pad_indices[0]
    expanded = (
        prompt_ids[:start]
        + (image_pad_id,) * merged_token_count
        + prompt_ids[start + 1 :]
    )
    return expanded, tuple(range(start, start + merged_token_count))


def _context(
    *,
    identity: TrajectoryIdentity,
    model: ModelIdentity,
    policy: PolicyVersion,
    source: object,
    prompt_ids: tuple[int, ...],
    sampled: SampledPolicyTurn,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        trajectory_identity=identity,
        model=model,
        behavior_policy=policy,
        trajectory_source_visual=source,
        prior_observation_handles=(),
        prompt_token_ids_before_turn=prompt_ids,
        sampled_turn=sampled,
        assistant_turn_index=0,
        attempt_index=0,
        call_index=0,
    )


class _Qwen3ContextualHiddenStateDependency:
    """Run the exact frozen source-visual prefix for a contextual smoke."""

    def __init__(
        self,
        *,
        model: Any,
        source_visual: Any,
        source_positions: tuple[int, ...],
        image_pad_id: int,
        forward_identity: ArtifactIdentity,
    ) -> None:
        self.model = model
        self.source_visual = source_visual
        self.source_positions = source_positions
        self.image_pad_id = image_pad_id
        self.forward_identity = forward_identity
        self.family = Qwen3VLAdapter()

    def capture_hidden_states(self, request: Any) -> BehaviorHiddenStateMaterialization:
        if request.hidden_layer != -1:
            raise ValueError("smoke contextual forward exposes only the final layer")
        ids = request.input_ids.reshape(1, -1)
        if (
            tuple(
                index
                for index, token_id in enumerate(request.input_ids.tolist())
                if token_id == self.image_pad_id
            )
            != self.source_positions
        ):
            raise ValueError("contextual smoke source positions changed")
        cpu_ids = ids.detach().to(device="cpu")
        cpu_mask = torch.ones_like(cpu_ids, dtype=torch.bool)
        core = getattr(self.model, "model", None)
        get_rope_index = getattr(core, "get_rope_index", None)
        if not callable(get_rope_index):
            raise TypeError("Qwen3 smoke model has no native M-RoPE helper")
        positions, _delta = get_rope_index(
            input_ids=cpu_ids,
            image_grid_thw=torch.tensor(
                (self.source_visual.image_grid_thw,), dtype=torch.long
            ),
            video_grid_thw=None,
            attention_mask=cpu_mask,
        )
        result = self.family.forward_injected(
            self.model,
            InjectedForwardRequest(
                input_ids=ids,
                attention_mask=cpu_mask.to(device=ids.device),
                position_ids=positions,
                visual_blocks=(
                    InjectedVisualBlock(
                        kind="source_image",
                        positions=self.source_positions,
                        embeddings=self.source_visual.merged_main.unsqueeze(0),
                        deepstack=tuple(
                            value.unsqueeze(0)
                            for value in self.source_visual.merged_deepstack
                        ),
                        deepstack_positions=tuple(
                            self.source_positions
                            for _ in self.source_visual.merged_deepstack
                        ),
                    ),
                ),
            ),
        )
        return BehaviorHiddenStateMaterialization(
            policy_version=request.call.identity.behavior_policy,
            forward_identity=self.forward_identity,
            hidden_layer=request.hidden_layer,
            hidden_states=result.hidden_states[0].detach(),
            deterministic_forward=True,
            policy_adapter_dropout=0.0,
        )


def main() -> int:
    assert_legacy_standalone_execution_quarantined("tools/smoke_qwen3_crop_tools.py")
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output}")
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA GPU")

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model_identity = ModelIdentity(
        family="qwen3_vl",
        model_name=MODEL_NAME,
        revision_or_path=ACCEPTED_QWEN3_MODEL_PATH,
        tokenizer_length=ACCEPTED_QWEN3_TOKENIZER_LENGTH,
        chat_template_sha256=ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256,
    )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        ACCEPTED_QWEN3_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_length_before = len(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        ACCEPTED_QWEN3_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device=device)
    export = load_rank_zero_adapter_owned_state_export(args.adapter)
    if export.state is None:
        raise RuntimeError("representation export has no Adapter state")
    run_identity = export.manifest.run_identity
    if run_identity.model != model_identity:
        raise RuntimeError("representation artifact model differs from smoke model")
    representation_runtime = create_qwen3_representation_runtime(
        model=model,
        processor=processor,
        model_identity=model_identity,
        conditioning_config=run_identity.provider,
        adapter_dtype=torch.bfloat16,
        fixture_mode=False,
    )
    representation_identity = ArtifactIdentity(
        "tgvf",
        run_identity.run_id,
        f"step-{export.manifest.global_step}",
        state_digest(export.manifest),
    )
    artifact_binding = RepresentationArtifactRuntimeBinding(
        artifact_path=args.adapter,
        artifact=representation_identity,
        expected_run_id=run_identity.run_id,
        expected_run_identity_sha256=run_identity.identity_sha256,
        model=run_identity.model,
        conditioning=run_identity.provider,
        adapter_contract=run_identity.adapter_contract,
    )
    loaded_adapter = load_frozen_tgvf_adapter(
        binding=artifact_binding,
        adapter=representation_runtime.adapter,
    )
    model.requires_grad_(False).eval()

    store = ObservationStore()
    materializer = Qwen3CropVisualMaterializer.from_model(
        model=model,
        processor=processor,
        model_identity=model_identity,
        image_max_pixels=args.image_max_pixels,
    )
    layout_builder = Qwen3NativeToolLayoutBuilder.from_model(
        model=model,
        processor=processor,
        model_identity=model_identity,
        observation_store=store,
    )
    policy = PolicyVersion("tool-smoke", 0, ZERO_SHA)
    source_rgb = _synthetic_rgb()
    fused_text = (
        "inspect the fine print</think>\n<tool_call>"
        '{"name":"tgvf_crop_tool","arguments":'
        '{"bbox_2d":[32,48,224,208],"target":"small red digits"}}'
        "</tool_call>"
    )
    fused_sampled, fused_parsed = _sampled_policy_turn(
        processor.tokenizer, fused_text, policy
    )
    if getattr(fused_parsed, "name", None) != TGVF_CROP_TOOL_NAME:
        raise TypeError("fused parser returned another call type")
    source_visual = materializer.materialize_source_visual(
        source_rgb,
        parsed_call=fused_parsed,
        call_index=0,
    )
    canonical_prompt_ids = tuple(
        processor.tokenizer.encode(
            "Inspect this image carefully.\n" + QWEN_NATIVE_IMAGE_PLACEHOLDER,
            add_special_tokens=False,
        )
    )
    image_pad_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    prompt_ids, source_positions = _expand_source_prompt(
        canonical_prompt_ids,
        image_pad_id=image_pad_id,
        merged_token_count=int(source_visual.merged_main.shape[-2]),
    )

    def bind_source(identity: TrajectoryIdentity):
        return record_trajectory_source_visual(
            trajectory_id=identity.canonical_id,
            source_visual=source_visual,
            source_positions=source_positions,
            deepstack_branch_layers=BRANCH_LAYERS,
            deepstack_injection_positions=tuple(
                source_positions for _ in BRANCH_LAYERS
            ),
            observation_store=store,
            source_rgb=source_rgb,
        )

    processor_identity = _artifact(
        "qwen3-crop-processor",
        f"{ACCEPTED_QWEN3_MODEL_PATH}:{args.image_max_pixels}",
    )
    layout_identity = _artifact(
        "qwen3-native-crop-layout",
        f"{ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256}:native-rope-v1",
    )
    merger_identities = tuple(
        _artifact(f"qwen3-merger-{layer}", identity)
        for layer, identity in zip(
            BRANCH_LAYERS,
            representation_runtime.projection_identities[1:],
            strict=True,
        )
    )
    branch_mergers = tuple(
        BranchMergerRuntimeBinding(projection_identity, artifact)
        for projection_identity, artifact in zip(
            run_identity.adapter_contract.deepstack_projection_identities,
            merger_identities,
            strict=True,
        )
    )

    plain_identity = TrajectoryIdentity("tool-smoke", "plain", 0, "plain")
    plain_source = bind_source(plain_identity)
    plain_text = (
        "zoom in</think>\n<tool_call>"
        '{"name":"image_zoom_in_tool","arguments":'
        '{"bbox_2d":[16,24,192,216]}}'
        "</tool_call>"
    )
    plain_sampled, plain_parsed = _sampled_policy_turn(
        processor.tokenizer, plain_text, policy
    )
    if getattr(plain_parsed, "name", None) != IMAGE_ZOOM_IN_TOOL_NAME:
        raise TypeError("plain parser returned another call type")
    plain_observation_contract = NativeSuccessObservationContract(
        protocol_id=(
            NativeSuccessObservationProtocolId.LEGACY_CROP_GENERIC_THINKING_V1
        ),
        tool_profile=NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
    )
    plain_runtime = ImageZoomInToolRuntime(
        model=model_identity,
        materializer=materializer,
        layout_builder=layout_builder,
        observation_store=store,
        crop_processor_identity=processor_identity,
        crop_layout_identity=layout_identity,
        execution_ledger=CropExecutionLedger(),
        coordinate_mapper=Qwen3VLAdapter(),
        observation_contract=plain_observation_contract,
    )
    plain_handle = plain_runtime.execute(
        plain_parsed,
        _context(
            identity=plain_identity,
            model=model_identity,
            policy=policy,
            source=plain_source,
            prompt_ids=prompt_ids,
            sampled=plain_sampled,
        ),
    )

    fused_identity = TrajectoryIdentity("tool-smoke", "fused", 0, "fused")
    fused_source = bind_source(fused_identity)
    if (
        run_identity.provider.provider
        is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    ):
        contextual_forward_identity = _artifact(
            "qwen3-contextual-injected-forward",
            "Qwen3VLAdapter.forward_injected:no-cache:source-image",
        )
        hidden_state_capture = FrozenBehaviorHiddenStateCapturePort(
            model=model_identity,
            forward_identity=contextual_forward_identity,
            dependency=_Qwen3ContextualHiddenStateDependency(
                model=model,
                source_visual=source_visual,
                source_positions=source_positions,
                image_pad_id=image_pad_id,
                forward_identity=contextual_forward_identity,
            ),
        )
    else:
        contextual_forward_identity = None
        hidden_state_capture = None
    fused_runtime = AtomicCropTGVFToolRuntime(
        conditioning_provider=representation_runtime.conditioning_provider,
        hidden_state_capture=hidden_state_capture,
        atomic_tool=AtomicCropTGVFTool(
            materializer=materializer,
            adapter=loaded_adapter.adapter,
            store=store,
            coordinate_mapper=Qwen3VLAdapter(),
        ),
        layout_builder=layout_builder,
        loaded_adapter=loaded_adapter,
        branch_mergers=branch_mergers,
        crop_processor_identity=processor_identity,
        crop_layout_identity=layout_identity,
        conditioning_input_device=device,
        contextual_forward_identity=contextual_forward_identity,
        execution_ledger=FocusExecutionLedger(),
        observation_contract=NativeSuccessObservationContract(
            protocol_id=NativeSuccessObservationProtocolId.GENERIC_NATIVE_V1,
            tool_profile=NativeToolCapabilityProfile.CROP_TGVF,
            assistant_dialect=NativeAssistantDialect.QWEN3_VL_THINKING,
        ),
    )
    fused_handle = fused_runtime.execute(
        fused_parsed,
        _context(
            identity=fused_identity,
            model=model_identity,
            policy=policy,
            source=fused_source,
            prompt_ids=prompt_ids,
            sampled=fused_sampled,
        ),
    )

    plain_record = store.resolve_record(plain_handle)
    fused_record = store.resolve_record(fused_handle)
    if not isinstance(plain_record, CropObservationRecord):
        raise TypeError("plain runtime did not store CropObservationRecord")
    if not isinstance(fused_record, CropTGVFObservationRecord):
        raise TypeError("fused runtime did not store CropTGVFObservationRecord")
    if (
        fused_record.layout.deepstack_branch_layers != BRANCH_LAYERS
        or len(fused_record.branches) != 3
    ):
        raise RuntimeError("fused runtime did not preserve all Qwen3 branches")
    resolver = Qwen3VLLMObservationPayloadResolver(
        store=store,
        include_multi_modal_uuid=True,
    )
    plain_payload = resolver.resolve(plain_handle, call_index=0)
    fused_payload = resolver.resolve(fused_handle, call_index=0)
    if len(processor.tokenizer) != tokenizer_length_before:
        raise RuntimeError("crop tools changed tokenizer length")
    if any(
        parameter.requires_grad
        for module in (model, representation_runtime.adapter)
        for parameter in module.parameters()
    ):
        raise RuntimeError("smoke left model or Adapter trainable")

    report = {
        "schema_version": "qwen3-crop-tools-real-model-materialization-smoke-v2",
        "status": "pass",
        "model": asdict(model_identity),
        "adapter_sha256": representation_identity.sha256,
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": len(processor.tokenizer),
        "source_grid_thw": source_visual.image_grid_thw,
        "source_merged_tokens": int(source_visual.merged_main.shape[-2]),
        "plain": {
            "success_observation_protocol_id": (
                plain_observation_contract.protocol_id.value
            ),
            "record_sha256": plain_handle.record_sha256,
            "requested_bbox_2d": plain_record.requested_bbox_2d,
            "effective_bbox_2d": plain_record.effective_bbox_2d,
            "grid_thw": plain_record.crop_visual.image_grid_thw,
            "visual_tokens": len(plain_record.crop_visual.positions),
            "deepstack_branch_count": len(plain_record.crop_visual.merged_deepstack),
            "vllm_payload_sha256": plain_payload.payload_sha256,
        },
        "atomic_crop_tgvf": {
            "record_sha256": fused_handle.record_sha256,
            "requested_bbox_2d": fused_record.requested_bbox_2d,
            "effective_bbox_2d": fused_record.effective_bbox_2d,
            "target": fused_parsed.target,
            "conditioning_provider": fused_record.condition.provider,
            "grid_thw": fused_record.crop_visual.source.image_grid_thw,
            "main_d_tokens": len(fused_record.layout.d_positions),
            "deepstack_branch_layers": fused_record.layout.deepstack_branch_layers,
            "vllm_payload_sha256": fused_payload.payload_sha256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
