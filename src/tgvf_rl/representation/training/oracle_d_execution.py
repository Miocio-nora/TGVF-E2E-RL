"""Visual materialization, decoding, and parity execution for oracle-D utility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import Any, Literal

import torch

from tgvf_rl.conditioning import (
    TargetConditioningProviderKind,
    TargetConditioningRequest,
)
from tgvf_rl.conditioning.base import _bind_canonical_input_ids
from tgvf_rl.public_api_compat import rebind_public_function
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
    batch_identical_injected_requests,
)
from tgvf_rl.representation.adapter import TGVFAdapterOutput

from .native_pipeline import (
    ModelActionTarget,
    NativeActionTarget,
    Qwen3NativeRepresentationGroupBuilder,
    _adapter_output_bundle,
    _move_processor_batch,
    _native_action_target_from_rendered,
    _processor_batch,
    _single_visual_expansion_count,
    _source_bundle,
)
from .oracle_d_schema import (
    OracleArmContext,
    OracleBatchCompatibilityError,
    OracleDUtilityArm,
    OracleDUtilityModelInput,
    OracleGeneratedAnswer,
    OracleGroupVisuals,
    OracleImageOnlyParity,
    build_image_only_messages,
    build_oracle_target_messages,
)
from .readout import RepresentationVisualTensorBundle
from .runtime import (
    Qwen3ContextualHiddenStateStack,
    Qwen3RepresentationRuntime,
    Qwen3VisionFeatures,
    Qwen3VisionPreMergeRequest,
)


def materialize_oracle_group_visuals(
    *,
    model_inputs: Sequence[OracleDUtilityModelInput],
    runtime: Qwen3RepresentationRuntime,
    group_builder: Qwen3NativeRepresentationGroupBuilder,
) -> OracleGroupVisuals:
    """Generate every correct D for one same-image group using the train path."""

    rows = tuple(model_inputs)
    if not rows or any(not isinstance(row, OracleDUtilityModelInput) for row in rows):
        raise ValueError("oracle visual materialization requires typed model inputs")
    if (
        len({row.image_group_key for row in rows}) != 1
        or len({row.image for row in rows}) != 1
    ):
        raise ValueError("oracle visual materialization requires one exact image group")
    if group_builder.runtime is not runtime:
        raise ValueError("group builder and oracle runtime differ")
    messages = tuple(
        build_oracle_target_messages(
            row,
            include_source_image=True,
            assistant_dialect=runtime.renderer.assistant_dialect,
        )
        for row in rows
    )
    with runtime.validated_group_execution():
        prefills = runtime.renderer.render_many(
            tuple(turns[:1] for turns in messages), add_generation_prompt=True
        )
        transcripts = runtime.renderer.render_many(
            tuple(turns[:2] for turns in messages), add_generation_prompt=False
        )
        actions: tuple[NativeActionTarget, ...] = tuple(
            _native_action_target_from_rendered(
                runtime,
                messages=turns,
                prefill=prefill,
                transcript=transcript,
            )
            for turns, prefill, transcript in zip(
                messages, prefills, transcripts, strict=True
            )
        )
        image = group_builder.image_loader(rows[0].image)
        if image is None:
            raise ValueError("image_loader returned None")
        first_action, first_expansion = (
            group_builder._materialize_action_with_expansion(actions[0], image)
        )
        visual_token_count = _single_visual_expansion_count(first_expansion)
        model_actions: tuple[ModelActionTarget, ...] = (
            first_action,
            *tuple(
                group_builder._materialize_action_from_shared_visual(
                    action,
                    reference=first_action,
                    visual_token_count=visual_token_count,
                )
                for action in actions[1:]
            ),
        )
        vision = runtime.extract_vision_features(
            Qwen3VisionPreMergeRequest(
                pixel_values=first_action.pixel_values,
                image_grid_thw=first_action.image_grid_thw,
            )
        )
        if int(vision.merged_main.shape[-2]) != visual_token_count:
            raise ValueError("source vision tokens differ from action expansion")
        correct: dict[str, RepresentationVisualTensorBundle] = {}
        for row, action in zip(rows, model_actions, strict=True):
            condition = _oracle_target_condition(
                runtime=runtime,
                model_input=row,
                action=action,
                vision=vision,
            )
            with torch.no_grad():
                output = runtime.adapter(runtime.make_adapter_input(condition, vision))
            if not isinstance(output, TGVFAdapterOutput):
                raise TypeError("Stage1 Adapter returned an invalid output")
            correct[row.sample_id] = _detached_bundle(_adapter_output_bundle(output))
        return OracleGroupVisuals(
            source=_detached_bundle(_source_bundle(vision)),
            correct_d_by_sample_id=correct,
            image_grid_thw=vision.image_grid_thw,
        )


def greedy_oracle_answer(
    *,
    context: OracleArmContext,
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
    decode_mode: Literal["cached", "no_cache"],
) -> OracleGeneratedAnswer:
    """Run deterministic native Thinking greedy generation for one arm."""

    if not eos_token_ids or len(set(eos_token_ids)) != len(eos_token_ids):
        raise ValueError("eos_token_ids must be non-empty and unique")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in eos_token_ids
    ):
        raise ValueError("eos_token_ids must be non-negative integers")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if decode_mode not in {"cached", "no_cache"}:
        raise ValueError("decode_mode must be cached or no_cache")
    generated: list[int] = []
    stop_reason: Literal["natural_stop", "length_cap"] = "length_cap"
    if decode_mode == "cached":
        if not family_adapter.capabilities.native_injected_kv_cache:
            raise ValueError("family adapter has no injected KV-cache path")
        materialized = context.materialize((), runtime)
        with torch.no_grad():
            result = family_adapter.prefill_injected_cache(runtime.model, materialized)
        past_key_values = result.past_key_values
        next_logits = result.logits[0, -1].float()
        for token_index in range(max_new_tokens):
            token_id = _greedy_token(next_logits)
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "natural_stop"
                break
            if token_index + 1 == max_new_tokens:
                break
            full_request = context.materialize(tuple(generated), runtime)
            cache_position = torch.tensor(
                (full_request.input_ids.shape[1] - 1,),
                dtype=torch.long,
                device=full_request.input_ids.device,
            )
            with torch.no_grad():
                result = family_adapter.forward_cached_token(
                    runtime.model,
                    CachedTokenForwardRequest(
                        input_ids=full_request.input_ids[:, -1:],
                        attention_mask=full_request.attention_mask,
                        position_ids=full_request.position_ids[..., -1:],
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                    ),
                )
            past_key_values = result.past_key_values
            next_logits = result.logits[0, -1].float()
    else:
        for _ in range(max_new_tokens):
            materialized = context.materialize(tuple(generated), runtime)
            with torch.no_grad():
                result = family_adapter.forward_injected(runtime.model, materialized)
            token_id = _greedy_token(result.logits[0, -1].float())
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "natural_stop"
                break
    token_ids = tuple(generated)
    if not token_ids:
        raise RuntimeError("oracle greedy generation produced no token")
    if any(
        token_id in context.forbidden_multimodal_token_ids for token_id in token_ids
    ):
        raise RuntimeError("oracle greedy generation emitted a multimodal token")
    text = runtime.tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    runtime.renderer.assert_tokenizer_length()
    if not isinstance(text, str) or not text:
        raise RuntimeError("oracle greedy token IDs decoded to empty text")
    return OracleGeneratedAnswer(
        token_ids=token_ids,
        text=text,
        stop_reason=stop_reason,
    )


def greedy_oracle_answers_batched(
    *,
    contexts: Sequence[OracleArmContext],
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[OracleGeneratedAnswer, ...]:
    """Greedily decode compatible oracle arms through one shared KV-cache batch."""

    lanes = tuple(contexts)
    if len(lanes) < 2:
        raise OracleBatchCompatibilityError(
            "batched oracle generation requires at least two arms"
        )
    if not eos_token_ids or len(set(eos_token_ids)) != len(eos_token_ids):
        raise ValueError("eos_token_ids must be non-empty and unique")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in eos_token_ids
    ):
        raise ValueError("eos_token_ids must be non-negative integers")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not family_adapter.capabilities.native_injected_kv_cache:
        raise ValueError("family adapter has no injected KV-cache path")

    prefixes = tuple(context.materialize((), runtime) for context in lanes)
    try:
        batched_prefix = batch_identical_injected_requests(prefixes)
    except ValueError as error:
        raise OracleBatchCompatibilityError(
            "oracle arms do not share one exact native generation prefix"
        ) from error
    with torch.no_grad():
        result = family_adapter.prefill_injected_cache(runtime.model, batched_prefix)
    past_key_values = result.past_key_values
    next_logits = _batched_next_logits(result.logits, lane_count=len(lanes))
    generated: list[list[int]] = [[] for _context in lanes]
    cache_suffixes: list[list[int]] = [[] for _context in lanes]
    finished = [False for _context in lanes]
    stop_reasons: list[Literal["natural_stop", "length_cap"]] = [
        "length_cap" for _context in lanes
    ]
    finished_fill_token_id = eos_token_ids[0]

    for token_index in range(max_new_tokens):
        predicted = tuple(
            int(token_id)
            for token_id in torch.argmax(next_logits, dim=-1).detach().cpu().tolist()
        )
        if len(predicted) != len(lanes):
            raise RuntimeError("batched oracle logits lost a decode lane")
        for lane_index, token_id in enumerate(predicted):
            if finished[lane_index]:
                cache_suffixes[lane_index].append(finished_fill_token_id)
                continue
            generated[lane_index].append(token_id)
            cache_suffixes[lane_index].append(token_id)
            if token_id in eos_token_ids:
                finished[lane_index] = True
                stop_reasons[lane_index] = "natural_stop"
        if all(finished) or token_index + 1 == max_new_tokens:
            break

        full_requests = tuple(
            context.materialize(tuple(cache_suffixes[lane_index]), runtime)
            for lane_index, context in enumerate(lanes)
        )
        sequence_lengths = {
            int(request.input_ids.shape[1]) for request in full_requests
        }
        if len(sequence_lengths) != 1:
            raise OracleBatchCompatibilityError(
                "batched oracle arms produced different cached sequence lengths"
            )
        first_request = full_requests[0]
        position_batch_dimension = 0 if first_request.position_ids.ndim == 2 else 1
        cache_position = torch.tensor(
            (first_request.input_ids.shape[1] - 1,),
            dtype=torch.long,
            device=first_request.input_ids.device,
        )
        with torch.no_grad():
            result = family_adapter.forward_cached_token(
                runtime.model,
                CachedTokenForwardRequest(
                    input_ids=torch.cat(
                        tuple(request.input_ids[:, -1:] for request in full_requests),
                        dim=0,
                    ),
                    attention_mask=torch.cat(
                        tuple(request.attention_mask for request in full_requests),
                        dim=0,
                    ),
                    position_ids=torch.cat(
                        tuple(
                            request.position_ids[..., -1:] for request in full_requests
                        ),
                        dim=position_batch_dimension,
                    ),
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
            )
        past_key_values = result.past_key_values
        next_logits = _batched_next_logits(result.logits, lane_count=len(lanes))

    answers: list[OracleGeneratedAnswer] = []
    for context, token_values, stop_reason in zip(
        lanes,
        generated,
        stop_reasons,
        strict=True,
    ):
        token_ids = tuple(token_values)
        if not token_ids:
            raise RuntimeError("batched oracle greedy generation produced no token")
        if any(
            token_id in context.forbidden_multimodal_token_ids for token_id in token_ids
        ):
            raise RuntimeError("oracle greedy generation emitted a multimodal token")
        text = runtime.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        runtime.renderer.assert_tokenizer_length()
        if not isinstance(text, str) or not text:
            raise RuntimeError("oracle greedy token IDs decoded to empty text")
        answers.append(
            OracleGeneratedAnswer(
                token_ids=token_ids,
                text=text,
                stop_reason=stop_reason,
            )
        )
    return tuple(answers)


def _batched_next_logits(logits: torch.Tensor, *, lane_count: int) -> torch.Tensor:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != lane_count
        or logits.shape[1] == 0
        or logits.shape[2] == 0
    ):
        raise RuntimeError("batched oracle generation returned invalid logits")
    next_logits = logits[:, -1].float()
    if not bool(torch.isfinite(next_logits).all()):
        raise RuntimeError("batched oracle generation returned invalid logits")
    return next_logits


def verify_image_only_injected_native_parity(
    *,
    model_input: OracleDUtilityModelInput,
    context: OracleArmContext,
    runtime: Qwen3RepresentationRuntime,
    family_adapter: QwenVLMFamilyAdapter,
    image_loader: Any,
    image_max_pixels: int | None,
) -> OracleImageOnlyParity:
    """Compare the first image-only next token to Qwen's native top-level path."""

    if context.arm is not OracleDUtilityArm.IMAGE_ONLY:
        raise ValueError("native parity requires the image_only arm")
    if not callable(image_loader):
        raise TypeError("image_loader must be callable")
    messages = build_image_only_messages(model_input)
    rendered_text, canonical_ids = _render_direct_without_tools(runtime, messages)
    if _integer_sequence_sha256(canonical_ids) != context.canonical_token_ids_sha256:
        raise RuntimeError("image-only parity rerender changed canonical token IDs")
    image = image_loader(model_input.image)
    if image is None:
        raise ValueError("image_loader returned None during image-only parity")
    processor_batch = _processor_batch(
        runtime.processor,
        text=rendered_text,
        images=(image,),
        image_max_pixels=image_max_pixels,
    )
    input_ids, attention_mask, pixel_values, grid = _move_processor_batch(
        runtime, processor_batch
    )
    if tuple(int(value) for value in input_ids[0].detach().cpu().tolist()) != tuple(
        int(value) for value in context.prefix_input_ids[0].tolist()
    ):
        raise RuntimeError(
            "manual injected source prefix differs from Qwen processor IDs"
        )
    observed_grid = tuple(int(value) for value in grid[0].detach().cpu().tolist())
    expected_grid = tuple(int(value) for value in context.image_grid_thw[0].tolist())
    if observed_grid != expected_grid:
        raise RuntimeError(
            "manual injected source grid differs from Qwen processor grid"
        )
    with torch.no_grad():
        native = runtime.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            use_cache=False,
            return_dict=True,
        )
        injected = family_adapter.forward_injected(
            runtime.model, context.materialize((), runtime)
        )
    native_all_logits = getattr(native, "logits", None)
    if not isinstance(native_all_logits, torch.Tensor):
        raise RuntimeError("native Qwen image-only forward returned no logits")
    native_logits = native_all_logits[0, -1].float()
    injected_logits = injected.logits[0, -1].float()
    if native_logits.shape != injected_logits.shape or not bool(
        torch.isfinite(native_logits).all() and torch.isfinite(injected_logits).all()
    ):
        raise RuntimeError("native/injected image-only logits are invalid")
    absolute = (native_logits - injected_logits).abs()
    native_top1 = int(torch.argmax(native_logits).item())
    injected_top1 = int(torch.argmax(injected_logits).item())
    result = OracleImageOnlyParity(
        sample_id=model_input.sample_id,
        native_top1_token_id=native_top1,
        injected_top1_token_id=injected_top1,
        top1_match=native_top1 == injected_top1,
        max_abs_logit_difference=float(absolute.max().item()),
        mean_abs_logit_difference=float(absolute.mean().item()),
        native_prefix_token_count=int(input_ids.shape[1]),
        image_grid_thw=observed_grid,
    )
    if not result.top1_match:
        raise RuntimeError(
            "image-only injected/native first-token top-1 parity failed: "
            f"native={native_top1}, injected={injected_top1}"
        )
    return result


def _oracle_target_condition(
    *,
    runtime: Qwen3RepresentationRuntime,
    model_input: OracleDUtilityModelInput,
    action: ModelActionTarget,
    vision: Qwen3VisionFeatures,
) -> Any:
    action.assert_bound_invariants()
    conditioning_ids = action.input_ids[0]
    request = TargetConditioningRequest(
        input_ids=conditioning_ids,
        target_span=action.target_span,
        expected_target_token_ids=action.target_token_ids,
        trajectory_id=f"oracle-d-utility:{model_input.sample_id}",
        call_index=0,
        model_identity=runtime.model_identity,
        canonical_input_ids_proof=_bind_canonical_input_ids(
            conditioning_ids, action.model_token_ids
        ),
    )
    contextual = None
    if runtime.conditioning_config.provider is (
        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    ):
        with torch.no_grad():
            output = runtime.model(
                input_ids=action.input_ids,
                attention_mask=action.attention_mask,
                pixel_values=action.pixel_values,
                image_grid_thw=action.image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        raw_layers = getattr(output, "hidden_states", None)
        if not isinstance(raw_layers, (tuple, list)) or not raw_layers:
            raise RuntimeError("frozen Qwen did not return contextual hidden states")
        layers = tuple(layer.detach().clone() for layer in raw_layers)
        if any(layer.shape[:2] != action.input_ids.shape for layer in layers):
            raise ValueError("Qwen contextual states do not align with action IDs")
        contextual = Qwen3ContextualHiddenStateStack(
            tuple(layer[0] for layer in layers)
        )
    return runtime.build_target_condition(request, contextual_hidden_states=contextual)


def _render_direct_without_tools(
    runtime: Qwen3RepresentationRuntime,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[int, ...]]:
    """Render the true image-only baseline without exposing a tool schema."""

    runtime.renderer.assert_tokenizer_length()
    runtime.renderer.assert_chat_template_identity()
    try:
        text = runtime.processor.apply_chat_template(
            list(messages),
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as error:
        raise TypeError(
            "Qwen processor rejected the direct image-only prompt"
        ) from error
    if not isinstance(text, str) or not text.endswith(
        runtime.renderer.assistant_dialect.generation_prefill_text
    ):
        raise ValueError("direct image-only transcript has an invalid native prefill")
    token_ids = tuple(
        int(token_id)
        for token_id in runtime.tokenizer.encode(text, add_special_tokens=False)
    )
    runtime.renderer.assert_tokenizer_length()
    runtime.renderer.assert_chat_template_identity()
    return text, token_ids


def _injected_block(
    kind: str,
    positions: tuple[int, ...],
    visual: RepresentationVisualTensorBundle,
) -> InjectedVisualBlock:
    return InjectedVisualBlock(
        kind=kind,
        positions=positions,
        embeddings=visual.main,
        deepstack=visual.deepstack,
        deepstack_positions=tuple(positions for _ in visual.deepstack),
    )


def _zero_bundle(
    reference: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.zeros_like(reference.main),
        deepstack=tuple(torch.zeros_like(branch) for branch in reference.deepstack),
        branch_layers=reference.branch_layers,
        d_deepstack_active=reference.d_deepstack_active,
    )


def _detached_bundle(
    value: RepresentationVisualTensorBundle,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=value.main.detach().clone(),
        deepstack=tuple(branch.detach().clone() for branch in value.deepstack),
        branch_layers=value.branch_layers,
        d_deepstack_active=value.d_deepstack_active,
    )


def _assert_visual_bundle_match(
    reference: RepresentationVisualTensorBundle,
    candidate: RepresentationVisualTensorBundle,
) -> None:
    if (
        candidate.main.shape != reference.main.shape
        or candidate.main.dtype != reference.main.dtype
        or candidate.main.device != reference.main.device
        or candidate.branch_layers != reference.branch_layers
        or len(candidate.deepstack) != len(reference.deepstack)
        or any(
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            for left, right in zip(
                candidate.deepstack, reference.deepstack, strict=True
            )
        )
    ):
        raise ValueError("oracle source/D visual bundle contracts differ")


def _greedy_token(logits: torch.Tensor) -> int:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("oracle generation produced invalid logits")
    return int(torch.argmax(logits).item())


def _qwen3_multimodal_token_ids(
    runtime: Qwen3RepresentationRuntime,
) -> frozenset[int]:
    ids: list[int] = []
    for token in (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ):
        token_id = runtime.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"Qwen3 control token {token!r} has no integer ID")
        if runtime.tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Qwen3 control token {token!r} does not round trip")
        ids.append(token_id)
    return frozenset(ids)


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    return sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_IMPLEMENTATION_MODULE = __name__
_PUBLIC_MODULE = "tgvf_rl.representation.training.oracle_d_utility"
for _public_function in (
    materialize_oracle_group_visuals,
    greedy_oracle_answer,
    greedy_oracle_answers_batched,
    verify_image_only_injected_native_parity,
    _batched_next_logits,
    _oracle_target_condition,
    _render_direct_without_tools,
    _injected_block,
    _zero_bundle,
    _detached_bundle,
    _assert_visual_bundle_match,
    _greedy_token,
    _qwen3_multimodal_token_ids,
    _integer_sequence_sha256,
):
    rebind_public_function(
        _public_function,
        implementation_module=_IMPLEMENTATION_MODULE,
        public_module=_PUBLIC_MODULE,
    )


__all__ = [
    "greedy_oracle_answer",
    "greedy_oracle_answers_batched",
    "materialize_oracle_group_visuals",
    "verify_image_only_injected_native_parity",
]
