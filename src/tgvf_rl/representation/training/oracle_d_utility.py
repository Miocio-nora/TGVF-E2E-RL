"""Resumable oracle-target evaluation of Stage1 focused-D answer utility.

The public facade retains the historical API and the arm-context monkeypatch
boundary.  Schemas/scoring, execution mechanics, and durable publication live
in one-way implementation leaves.
"""

from __future__ import annotations

from collections import OrderedDict as OrderedDict
from collections.abc import Mapping as Mapping, Sequence as Sequence
from contextlib import contextmanager as contextmanager
from dataclasses import asdict as asdict, dataclass as dataclass
from datetime import datetime as datetime, timezone as timezone
from enum import Enum as Enum
from fractions import Fraction as Fraction
from hashlib import sha256 as sha256
import json as json
import os as os
from pathlib import Path as Path
import re as re
import subprocess as subprocess
import tempfile as tempfile
import time as time
from typing import Any as Any, Iterator as Iterator, Literal as Literal

import torch as torch

from tgvf_rl.checkpoint.coordinator import state_digest as state_digest
from tgvf_rl.conditioning import (
    TargetConditioningProviderKind as TargetConditioningProviderKind,
    TargetConditioningRequest as TargetConditioningRequest,
)
from tgvf_rl.protocol.native import NativeAssistantDialect as NativeAssistantDialect
from tgvf_rl.protocol.schema import TGVF_FOCUS_TOOL_NAME as TGVF_FOCUS_TOOL_NAME
from tgvf_rl.qwen.base import (
    CachedTokenForwardRequest as CachedTokenForwardRequest,
    InjectedForwardRequest as InjectedForwardRequest,
    InjectedVisualBlock as InjectedVisualBlock,
    QwenVLMFamilyAdapter as QwenVLMFamilyAdapter,
    batch_identical_injected_requests as batch_identical_injected_requests,
)
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter as Qwen3VLAdapter
from tgvf_rl.representation.adapter import TGVFAdapterOutput as TGVFAdapterOutput

from .config import (
    RepresentationTrainingConfig as RepresentationTrainingConfig,
    load_representation_training_config as load_representation_training_config,
)
from .data import (
    load_retained_representation_jsonl as load_retained_representation_jsonl,
)
from .distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export as load_rank_zero_adapter_owned_state_export,
)
from .evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _require_file_sha256,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
    load_representation_internal_evaluation_run_config as load_representation_internal_evaluation_run_config,
)
from .native_pipeline import (
    ModelActionTarget as ModelActionTarget,
    NativeActionTarget as NativeActionTarget,
    Qwen3NativeRepresentationGroupBuilder as Qwen3NativeRepresentationGroupBuilder,
    _expand_native_visual_placeholders,
    _qwen3_position_ids,
)
from .oracle_d_execution import (
    _assert_visual_bundle_match,
    _batched_next_logits as _batched_next_logits,
    _detached_bundle as _detached_bundle,
    _greedy_token as _greedy_token,
    _injected_block,
    _integer_sequence_sha256,
    _oracle_target_condition as _oracle_target_condition,
    _qwen3_multimodal_token_ids,
    _render_direct_without_tools,
    _zero_bundle,
    greedy_oracle_answer as greedy_oracle_answer,
    greedy_oracle_answers_batched as greedy_oracle_answers_batched,
    materialize_oracle_group_visuals as materialize_oracle_group_visuals,
    verify_image_only_injected_native_parity as verify_image_only_injected_native_parity,
)
from .oracle_d_ledger import (
    _OracleRunLedger,
    _arm_contract,
    _atomic_write_bytes as _atomic_write_bytes,
    _atomic_write_json,
    _canonical_json_bytes as _canonical_json_bytes,
    _canonical_sha256 as _canonical_sha256,
    _git_head as _git_head,
    _paired_summary as _paired_summary,
    _run_identity_payload,
)
from .oracle_d_schema import (
    DEFAULT_ORACLE_D_UTILITY_ARMS as DEFAULT_ORACLE_D_UTILITY_ARMS,
    DEFAULT_THINKING_EOS_TOKEN_IDS as DEFAULT_THINKING_EOS_TOKEN_IDS,
    NATIVE_REPRESENTATION_PRE_REASONING as NATIVE_REPRESENTATION_PRE_REASONING,
    ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION as ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
    ORACLE_D_UTILITY_SCHEMA_VERSION as ORACLE_D_UTILITY_SCHEMA_VERSION,
    ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION as ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION,
    OracleAnswerScore as OracleAnswerScore,
    OracleArmContext as OracleArmContext,
    OracleBatchCompatibilityError as OracleBatchCompatibilityError,
    OracleDUtilityArm as OracleDUtilityArm,
    OracleDUtilityGroundTruth as OracleDUtilityGroundTruth,
    OracleDUtilityModelInput as OracleDUtilityModelInput,
    OracleGeneratedAnswer as OracleGeneratedAnswer,
    OracleGroupVisuals as OracleGroupVisuals,
    OracleImageOnlyParity as OracleImageOnlyParity,
    _choice_label_for_expected as _choice_label_for_expected,
    _multiple_choice_label as _multiple_choice_label,
    _normalize_answer as _normalize_answer,
    _parse_number as _parse_number,
    _require_model_input,
    _strip_terminal_markers as _strip_terminal_markers,
    _thinking_final_answer as _thinking_final_answer,
    build_image_only_messages as build_image_only_messages,
    build_oracle_target_messages as build_oracle_target_messages,
    score_oracle_generated_answer as score_oracle_generated_answer,
    split_oracle_d_utility_sample as split_oracle_d_utility_sample,
)
from .readout import (
    RepresentationVisualTensorBundle as RepresentationVisualTensorBundle,
)
from .runtime import (
    Qwen3ContextualHiddenStateStack as Qwen3ContextualHiddenStateStack,
    Qwen3RepresentationRuntime as Qwen3RepresentationRuntime,
    Qwen3VisionFeatures as Qwen3VisionFeatures,
    Qwen3VisionPreMergeRequest as Qwen3VisionPreMergeRequest,
    create_qwen3_representation_runtime as create_qwen3_representation_runtime,
)
from .schema import (
    RepresentationChoice as RepresentationChoice,
    RepresentationTrainingSample as RepresentationTrainingSample,
)


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"


def prepare_oracle_arm_context(
    *,
    model_input: OracleDUtilityModelInput,
    arm: OracleDUtilityArm,
    runtime: Qwen3RepresentationRuntime,
    source: RepresentationVisualTensorBundle,
    correct_d: RepresentationVisualTensorBundle,
    image_grid_thw: tuple[int, int, int],
    matched_wrong_d: RepresentationVisualTensorBundle | None = None,
) -> OracleArmContext:
    """Render one arm and bind exactly its source/D injection blocks."""

    _require_model_input(model_input)
    if not isinstance(arm, OracleDUtilityArm):
        raise TypeError("arm must be OracleDUtilityArm")
    direct_replacement = arm in {
        OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT,
        OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT,
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
    }
    if direct_replacement and (
        not correct_d.d_deepstack_active or len(correct_d.deepstack) != 3
    ):
        raise ValueError(
            "direct D replacement requires active main D plus all three DeepStack branches"
        )
    if arm is OracleDUtilityArm.IMAGE_ONLY or direct_replacement:
        messages = build_image_only_messages(model_input)
        rendered_text, canonical_ids = _render_direct_without_tools(runtime, messages)
        include_source = arm is OracleDUtilityArm.IMAGE_ONLY
        if arm is OracleDUtilityArm.IMAGE_ONLY:
            selected_d = None
        elif arm is OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT:
            selected_d = _zero_bundle(correct_d)
        elif arm is OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT:
            selected_d = correct_d
        else:
            if matched_wrong_d is None:
                raise ValueError(
                    "direct_matched_wrong_D_replacement requires a same-image wrong D"
                )
            selected_d = matched_wrong_d
    else:
        include_source = arm in {
            OracleDUtilityArm.IMAGE_TARGET_ZERO_D,
            OracleDUtilityArm.IMAGE_CORRECT_D,
        }
        messages = build_oracle_target_messages(
            model_input,
            include_source_image=include_source,
            assistant_dialect=runtime.renderer.assistant_dialect,
        )
        rendered = runtime.renderer.render(messages, add_generation_prompt=True)
        runtime.renderer.assert_generation_prefill(rendered, runtime.tokenizer)
        rendered_text, canonical_ids = rendered.text, rendered.token_ids
        if arm in {
            OracleDUtilityArm.TARGET_ZERO_D_ONLY,
            OracleDUtilityArm.IMAGE_TARGET_ZERO_D,
        }:
            selected_d = _zero_bundle(correct_d)
        elif arm in {
            OracleDUtilityArm.CORRECT_D_ONLY,
            OracleDUtilityArm.IMAGE_CORRECT_D,
        }:
            selected_d = correct_d
        elif arm is OracleDUtilityArm.MATCHED_WRONG_D:
            if matched_wrong_d is None:
                raise ValueError("matched_wrong_D requires a same-image wrong D")
            selected_d = matched_wrong_d
        else:  # pragma: no cover - enum is closed
            raise AssertionError("unhandled oracle D utility arm")
    _assert_visual_bundle_match(source, correct_d)
    if selected_d is not None:
        _assert_visual_bundle_match(correct_d, selected_d)
    token_count = int(source.main.shape[1])
    placeholder_count = (1 if include_source else 0) + (
        1 if selected_d is not None else 0
    )
    model_ids, expansion = _expand_native_visual_placeholders(
        runtime,
        canonical_ids,
        visual_token_counts=tuple(token_count for _ in range(placeholder_count)),
    )
    visual_id = runtime.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    blocks = tuple(
        mapped
        for token_id, mapped in zip(
            expansion.canonical_token_ids,
            expansion.canonical_to_model_positions,
            strict=True,
        )
        if token_id == visual_id
    )
    if len(blocks) != placeholder_count:
        raise RuntimeError("oracle arm visual expansion changed block count")
    block_index = 0
    source_positions: tuple[int, ...] = ()
    d_positions: tuple[int, ...] = ()
    injected: list[InjectedVisualBlock] = []
    if include_source:
        source_positions = blocks[block_index]
        block_index += 1
        injected.append(_injected_block("source_image", source_positions, source))
    if selected_d is not None:
        d_positions = blocks[block_index]
        injected.append(_injected_block("focused_d", d_positions, selected_d))
    prefix_ids = torch.tensor((model_ids,), dtype=torch.long)
    attention_mask = torch.ones_like(prefix_ids)
    grid = torch.tensor(
        tuple(image_grid_thw for _ in range(placeholder_count)),
        dtype=torch.long,
    )
    positions = _qwen3_position_ids(
        runtime.model,
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        image_grid_thw=grid,
    )
    return OracleArmContext(
        arm=arm,
        rendered_text_sha256=sha256(rendered_text.encode("utf-8")).hexdigest(),
        canonical_token_ids_sha256=_integer_sequence_sha256(canonical_ids),
        prefix_input_ids=prefix_ids,
        prefix_position_ids=positions,
        image_grid_thw=grid,
        visual_blocks=tuple(injected),
        source_positions=source_positions,
        d_positions=d_positions,
        forbidden_multimodal_token_ids=_qwen3_multimodal_token_ids(runtime),
    )


def run_oracle_d_utility_evaluation(
    source_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[OracleDUtilityArm | str] = DEFAULT_ORACLE_D_UTILITY_ARMS,
    max_new_tokens: int,
    eos_token_ids: Sequence[int] = DEFAULT_THINKING_EOS_TOKEN_IDS,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run or exactly resume one single-GPU, image-group-sharded evaluation."""

    selected_arms = _normalize_arms(arms)
    selected_eos_token_ids = _normalize_eos_token_ids(eos_token_ids)
    _validate_selection(
        max_new_tokens=max_new_tokens,
        eos_token_ids=selected_eos_token_ids,
        decode_mode=decode_mode,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    source_config = load_representation_internal_evaluation_run_config(
        source_config_path
    )
    if source_config.evaluation_data_path is None or (
        source_config.evaluation_data_source_sha256 is None
    ):
        raise ValueError("oracle D utility requires an explicit evaluation_data split")
    training = load_representation_training_config(source_config.training_config_path)
    if Path(training.model.local_path).name != "Qwen3-VL-8B-Thinking":
        raise ValueError("oracle D utility entry is pinned to Qwen3-VL-8B-Thinking")
    _require_file_sha256(
        source_config.training_config_path,
        source_config.training_config_sha256,
        name="training config",
    )
    _require_file_sha256(
        source_config.artifact_path,
        source_config.artifact_file_sha256,
        name="Adapter artifact",
    )
    export = load_rank_zero_adapter_owned_state_export(source_config.artifact_path)
    manifest = export.manifest
    run_identity = manifest.run_identity
    if state_digest(manifest) != source_config.artifact_manifest_sha256:
        raise ValueError("Adapter artifact manifest SHA256 mismatch")
    if (
        manifest.run_identity_sha256 != source_config.expected_run_identity_sha256
        or run_identity.identity_sha256 != source_config.expected_run_identity_sha256
    ):
        raise ValueError("Adapter artifact run identity mismatch")
    if manifest.global_step != source_config.expected_global_step:
        raise ValueError("Adapter artifact global step mismatch")
    _validate_training_artifact_binding(training, run_identity)
    data = load_retained_representation_jsonl(
        source_config.evaluation_data_path,
        expected_source_sha256=source_config.evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )
    all_groups = _ordered_sample_groups(data.samples)
    selected_groups = _select_groups(
        all_groups,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    wrong_d_arms = {
        OracleDUtilityArm.MATCHED_WRONG_D,
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
    }
    if wrong_d_arms.intersection(selected_arms) and any(
        len(group) < 2 for _, group in selected_groups
    ):
        raise ValueError(
            "matched-wrong D arms require every selected image group to have K>=2"
        )
    model_truth_rows = tuple(
        split_oracle_d_utility_sample(sample)
        for _group_ordinal, group in selected_groups
        for sample in group
    )
    model_inputs = tuple(row[0] for row in model_truth_rows)
    identity_payload = _run_identity_payload(
        source_config=source_config,
        training=training,
        data_manifest_sha256=data.manifest.manifest_sha256,
        model_inputs=model_inputs,
        arms=selected_arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=selected_eos_token_ids,
        decode_mode=decode_mode,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    output = Path(output_root).resolve()
    ledger = _OracleRunLedger(
        output,
        identity_payload=identity_payload,
        expected_keys=tuple(
            (row.sample_id, arm.value) for row in model_inputs for arm in selected_arms
        ),
    )
    with ledger.locked():
        ledger.prepare()
        if ledger.complete:
            return ledger.summary()
        _require_single_gpu_environment()
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        _enable_determinism()
        _seed_current_process(source_config.evaluation.random_seed)
        processor, model = _load_qwen(training, device=device)
        tokenizer_length_before = len(processor.tokenizer)
        runtime = create_qwen3_representation_runtime(
            model=model,
            processor=processor,
            model_identity=training.model_identity,
            conditioning_config=training.provider,
            adapter_dtype=_torch_dtype(training.model.dtype),
            adapter_variant=training.adapter_variant,
            fixture_mode=False,
        )
        if runtime.renderer.assistant_dialect is not (
            NativeAssistantDialect.QWEN3_VL_THINKING
        ):
            raise ValueError("oracle D utility requires the native Thinking dialect")
        model_eos_token_ids = _model_eos_token_ids(model)
        if set(model_eos_token_ids) != set(selected_eos_token_ids):
            raise ValueError(
                "configured EOS IDs differ from the local Thinking generation config: "
                f"configured={selected_eos_token_ids}, model={model_eos_token_ids}"
            )
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("oracle D utility changed tokenizer length")
        run_identity.adapter_contract.assert_matches(runtime.adapter)
        if export.state is None:
            raise RuntimeError("Adapter export has no tensor state")
        runtime.adapter.load_artifact_state_dict(export.state)
        runtime.adapter.requires_grad_(False)
        runtime.adapter.eval()
        model.requires_grad_(False)
        model.eval()
        family_adapter = Qwen3VLAdapter()
        group_builder = Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=family_adapter,
            prompt=training.prompt,
            image_loader=_load_rgb_image,
            image_max_pixels=training.model.image_max_pixels,
        )
        truth_by_id = {truth.sample_id: truth for _model, truth in model_truth_rows}
        started = time.monotonic()
        parity_path = output / "image_only_native_parity.json"
        parity_checked = OracleDUtilityArm.IMAGE_ONLY not in selected_arms
        if parity_path.exists():
            parity_payload = json.loads(parity_path.read_text(encoding="utf-8"))
            if (
                parity_payload.get("run_identity_sha256") != ledger.identity_sha256
                or parity_payload.get("top1_match") is not True
            ):
                raise ValueError(
                    "existing image-only native parity artifact is invalid"
                )
            parity_checked = True
        for group_ordinal, samples in selected_groups:
            group_models = tuple(
                split_oracle_d_utility_sample(sample)[0] for sample in samples
            )
            pending = tuple(
                row
                for row in group_models
                if any(
                    not ledger.has(row.sample_id, arm.value) for arm in selected_arms
                )
            )
            if not pending:
                continue
            visuals = materialize_oracle_group_visuals(
                model_inputs=group_models,
                runtime=runtime,
                group_builder=group_builder,
            )
            if not parity_checked:
                parity_context = prepare_oracle_arm_context(
                    model_input=group_models[0],
                    arm=OracleDUtilityArm.IMAGE_ONLY,
                    runtime=runtime,
                    source=visuals.source,
                    correct_d=visuals.correct_d_by_sample_id[group_models[0].sample_id],
                    image_grid_thw=visuals.image_grid_thw,
                )
                parity = verify_image_only_injected_native_parity(
                    model_input=group_models[0],
                    context=parity_context,
                    runtime=runtime,
                    family_adapter=family_adapter,
                    image_loader=group_builder.image_loader,
                    image_max_pixels=group_builder.image_max_pixels,
                )
                _atomic_write_json(
                    parity_path,
                    {
                        "schema_version": "oracle_image_only_native_parity_v1",
                        "run_identity_sha256": ledger.identity_sha256,
                        **asdict(parity),
                    },
                )
                parity_checked = True
            for row_index, row in enumerate(group_models):
                correct_d = visuals.correct_d_by_sample_id[row.sample_id]
                wrong_d = (
                    visuals.correct_d_by_sample_id[
                        group_models[(row_index + 1) % len(group_models)].sample_id
                    ]
                    if len(group_models) > 1
                    else None
                )
                for arm in selected_arms:
                    if ledger.has(row.sample_id, arm.value):
                        continue
                    context = prepare_oracle_arm_context(
                        model_input=row,
                        arm=arm,
                        runtime=runtime,
                        source=visuals.source,
                        correct_d=correct_d,
                        image_grid_thw=visuals.image_grid_thw,
                        matched_wrong_d=wrong_d,
                    )
                    generated = greedy_oracle_answer(
                        context=context,
                        runtime=runtime,
                        family_adapter=family_adapter,
                        eos_token_ids=selected_eos_token_ids,
                        max_new_tokens=max_new_tokens,
                        decode_mode=decode_mode,
                    )
                    score = score_oracle_generated_answer(
                        generated.text,
                        truth_by_id[row.sample_id],
                        assistant_dialect=runtime.renderer.assistant_dialect,
                        generation_stop_reason=generated.stop_reason,
                    )
                    ledger.commit(
                        {
                            "schema_version": ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
                            "run_identity_sha256": ledger.identity_sha256,
                            "sample_id": row.sample_id,
                            "sample_content_sha256": row.sample_content_sha256,
                            "image_group_key": row.image_group_key,
                            "group_ordinal": group_ordinal,
                            "arm": arm.value,
                            "arm_contract": _arm_contract(arm),
                            "wrong_d_source_sample_id": (
                                group_models[
                                    (row_index + 1) % len(group_models)
                                ].sample_id
                                if arm
                                in {
                                    OracleDUtilityArm.MATCHED_WRONG_D,
                                    OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT,
                                }
                                else None
                            ),
                            "rendered_prefix_text_sha256": context.rendered_text_sha256,
                            "canonical_prefix_token_ids_sha256": context.canonical_token_ids_sha256,
                            "prefix_token_count": int(
                                context.prefix_input_ids.shape[1]
                            ),
                            "source_visual_token_count": len(context.source_positions),
                            "d_visual_token_count": len(context.d_positions),
                            "generated_token_ids": list(generated.token_ids),
                            "generated_text": generated.text,
                            "generation_stop_reason": generated.stop_reason,
                            "score": asdict(score),
                            "expected_short_answer": truth_by_id[
                                row.sample_id
                            ].short_answer,
                            "elapsed_seconds_since_process_start": time.monotonic()
                            - started,
                        }
                    )
            del visuals
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("oracle D utility changed tokenizer length")
        return ledger.summary()


def _ordered_sample_groups(
    samples: Sequence[RepresentationTrainingSample],
) -> tuple[tuple[int, tuple[RepresentationTrainingSample, ...]], ...]:
    grouped: OrderedDict[str, list[RepresentationTrainingSample]] = OrderedDict()
    for sample in samples:
        grouped.setdefault(sample.image_group_key, []).append(sample)
    return tuple(
        (ordinal, tuple(group)) for ordinal, group in enumerate(grouped.values())
    )


def _select_groups(
    groups: Sequence[tuple[int, tuple[RepresentationTrainingSample, ...]]],
    *,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[int, tuple[RepresentationTrainingSample, ...]], ...]:
    after_start = tuple(groups[group_start:])
    sharded = tuple(
        group
        for index, group in enumerate(after_start)
        if index % shard_count == shard_index
    )
    selected = sharded if group_limit is None else sharded[:group_limit]
    if not selected:
        raise ValueError("oracle D utility selection contains no image group")
    return selected


def _normalize_arms(
    arms: Sequence[OracleDUtilityArm | str],
) -> tuple[OracleDUtilityArm, ...]:
    if isinstance(arms, (str, bytes)):
        raise TypeError("arms must be a sequence")
    try:
        result = tuple(
            arm if isinstance(arm, OracleDUtilityArm) else OracleDUtilityArm(arm)
            for arm in arms
        )
    except ValueError as error:
        raise ValueError("unknown oracle D utility arm") from error
    if not result or len(set(result)) != len(result):
        raise ValueError("oracle D utility arms must be non-empty and unique")
    return result


def _validate_selection(
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    decode_mode: str,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> None:
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not eos_token_ids:
        raise ValueError("eos_token_ids must be non-empty")
    if decode_mode not in {"cached", "no_cache"}:
        raise ValueError("decode_mode must be cached or no_cache")
    for name, value in (("group_start", group_start), ("shard_index", shard_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
    ):
        raise ValueError("shard_count must be positive")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    if group_limit is not None and (
        isinstance(group_limit, bool)
        or not isinstance(group_limit, int)
        or group_limit <= 0
    ):
        raise ValueError("group_limit must be positive when set")


def _require_single_gpu_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    required = {
        "CUBLAS_WORKSPACE_CONFIG": _REQUIRED_CUBLAS_WORKSPACE,
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    mismatches = {
        name: (expected, os.environ.get(name))
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if not visible or "," in visible:
        mismatches["CUDA_VISIBLE_DEVICES"] = ("one physical GPU ID", visible)
    if mismatches:
        raise ValueError(f"oracle D utility launch environment mismatch: {mismatches}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("oracle D utility requires exactly one visible CUDA GPU")


def _normalize_eos_token_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("eos_token_ids must be a sequence of integers")
    result = tuple(values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in result
        )
    ):
        raise ValueError("eos_token_ids must be unique non-negative integers")
    return result


def _model_eos_token_ids(model: torch.nn.Module) -> tuple[int, ...]:
    generation_config = getattr(model, "generation_config", None)
    value = getattr(generation_config, "eos_token_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
    else:
        raise ValueError("local Thinking model has no explicit generation EOS IDs")
    return _normalize_eos_token_ids(result)


__all__ = [
    "DEFAULT_ORACLE_D_UTILITY_ARMS",
    "DEFAULT_THINKING_EOS_TOKEN_IDS",
    "ORACLE_D_UTILITY_SCHEMA_VERSION",
    "OracleAnswerScore",
    "OracleBatchCompatibilityError",
    "OracleDUtilityArm",
    "OracleDUtilityGroundTruth",
    "OracleDUtilityModelInput",
    "OracleGeneratedAnswer",
    "OracleImageOnlyParity",
    "build_image_only_messages",
    "build_oracle_target_messages",
    "greedy_oracle_answer",
    "greedy_oracle_answers_batched",
    "materialize_oracle_group_visuals",
    "prepare_oracle_arm_context",
    "run_oracle_d_utility_evaluation",
    "score_oracle_generated_answer",
    "split_oracle_d_utility_sample",
    "verify_image_only_injected_native_parity",
]
