"""Strict, removable Instruct evaluation of answer-utility Adapter artifacts.

The production oracle-D evaluator owns the native prompt rendering, visual
injection, greedy generation, and durable ledger.  This module supplies the
experiment-private artifact loader, reader prompt, scorer, and RP66 Instruct
bindings that the production evaluator deliberately does not accept.
"""

from __future__ import annotations

from collections.abc import Mapping as Mapping, Sequence
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Literal

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.config import (
    RepresentationTrainingConfig,
    load_representation_training_config,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION,
    OracleArmContext,
    OracleBatchCompatibilityError,
    OracleDUtilityArm,
    OracleGeneratedAnswer,
    _OracleRunLedger,
    _atomic_write_json,
    _injected_block,
    _model_eos_token_ids,
    _require_single_gpu_environment,
    greedy_oracle_answer,
    greedy_oracle_answers_batched,
    materialize_oracle_group_visuals,
    prepare_oracle_arm_context,
    split_oracle_d_utility_sample,
    verify_image_only_injected_native_parity,
)
from tgvf_rl.representation.training.post_training_evaluation import file_sha256
from tgvf_rl.representation.training.runtime import (
    create_qwen3_representation_runtime,
)
from tgvf_rl.representation.training.schema import (
    RepresentationTrainingSample as RepresentationTrainingSample,
)

from ..config import load_answer_utility_experiment_config
from ..run_config import AnswerUtilityRunConfig, load_answer_utility_run_config
from .input_identity import (
    _build_evaluation_identity_payload,
    _evaluation_arm_contract,
    _extended_summary,
    _implementation_file_manifest_impl,
    _oracle_arm_for,
    _uses_wrong_d,
    _validated_payload,
)
from .input_matching import (
    AnswerUtilityWrongImageDonor,
    _QwenImageGridContract as _QwenImageGridContract,
    _load_qwen_image_grid_contract as _load_qwen_image_grid_contract,
    _normalized_target_identity as _normalized_target_identity,
    _qwen_image_grid_thw as _qwen_image_grid_thw,
    _same_target_wrong_image_model_inputs,
    build_answer_safe_wrong_mapping,
    build_same_target_wrong_image_mapping,
)
from .inputs import (
    ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION,
    ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
    DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS,
    DEFAULT_INSTRUCT_EOS_TOKEN_IDS,
    AnswerUtilityAdapterArtifact,
    AnswerUtilityEvaluationArm,
    AnswerUtilityEvaluationCandidate,
    AnswerUtilityEvaluationInputs,
    _InputLoaderBindings,
    _assert_evaluation_output_isolated,
    _audit_completed_training_metrics_impl,
    _has_same_target_wrong_image_arm as _has_same_target_wrong_image_arm,
    _load_private_inputs_impl,
    _load_production_source_inputs_impl,
    _load_validated_production_export_impl,
    _materialize_common_inputs_impl,
    _normalize_arm_batch_size,
    _normalize_evaluation_arms,
    _require_instruct_training_impl,
    _validate_private_source_bindings_impl,
    load_answer_utility_adapter_artifact,
)
from .scoring import (
    INSTRUCT_READER_CONTRACT_VERSION,
    INSTRUCT_SCORING_CONTRACT_VERSION,
    reader_question,
    score_instruct_generated_answer,
)


def validate_answer_utility_evaluation(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Perform the complete binding/materialization preflight without CUDA."""

    inputs = _load_private_inputs(
        run_config_path,
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _validated_payload(inputs)


def validate_production_source_answer_utility_evaluation(
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Preflight the exact production Adapter as the E0 comparison candidate."""

    inputs = _load_production_source_inputs(
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _validated_payload(inputs)


def run_answer_utility_evaluation(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run or exactly resume one manifest-ordered single-GPU evaluation shard."""

    inputs = _load_private_inputs(
        run_config_path,
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _run_loaded_inputs(inputs, output_root=output_root)


def run_production_source_answer_utility_evaluation(
    source_evaluation_config_path: str | Path,
    *,
    output_root: str | Path,
    arms: Sequence[AnswerUtilityEvaluationArm | str] = (
        DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    ),
    max_new_tokens: int | None = None,
    eos_token_ids: Sequence[int] | None = None,
    decode_mode: Literal["cached", "no_cache"] = "cached",
    arm_batch_size: int = 1,
    group_start: int = 0,
    group_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    """Run the exact production Adapter through the common E0 evaluation path."""

    inputs = _load_production_source_inputs(
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _run_loaded_inputs(inputs, output_root=output_root)


def _evaluation_identity_payload(
    inputs: AnswerUtilityEvaluationInputs,
    model_inputs: Sequence[Any],
) -> dict[str, Any]:
    return _build_evaluation_identity_payload(
        inputs,
        model_inputs,
        implementation_file_manifest=_implementation_file_manifest,
        evaluation_arm_contract=_evaluation_arm_contract,
    )


def _implementation_file_manifest() -> dict[str, str]:
    return _implementation_file_manifest_impl()


def _run_loaded_inputs(
    inputs: AnswerUtilityEvaluationInputs, *, output_root: str | Path
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    _assert_evaluation_output_isolated(output, inputs)
    model_truth_rows = tuple(
        split_oracle_d_utility_sample(sample)
        for _ordinal, group in inputs.selected_groups
        for sample in group
    )
    model_inputs = tuple(model for model, _truth in model_truth_rows)
    identity_payload = _evaluation_identity_payload(inputs, model_inputs)
    ledger = _OracleRunLedger(
        output,
        identity_payload=identity_payload,
        expected_keys=tuple(
            (row.sample_id, arm.value) for row in model_inputs for arm in inputs.arms
        ),
    )
    with ledger.locked():
        ledger.prepare()
        if ledger.complete:
            return _extended_summary(ledger)
        _require_single_gpu_environment()
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        _enable_determinism()
        _seed_current_process(inputs.source_evaluation.evaluation.random_seed)
        processor, model = _load_qwen(inputs.training, device=device)
        tokenizer_length_before = len(processor.tokenizer)
        runtime = create_qwen3_representation_runtime(
            model=model,
            processor=processor,
            model_identity=inputs.training.model_identity,
            conditioning_config=inputs.training.provider,
            adapter_dtype=_torch_dtype(inputs.training.model.dtype),
            adapter_variant=inputs.training.adapter_variant,
            fixture_mode=False,
        )
        if runtime.renderer.assistant_dialect is not (
            NativeAssistantDialect.QWEN3_VL_INSTRUCT
        ):
            raise ValueError("answer-utility evaluation requires Qwen3-VL Instruct")
        observed_eos = _model_eos_token_ids(model)
        if set(observed_eos) != set(inputs.eos_token_ids):
            raise ValueError(
                "configured EOS IDs differ from local Instruct generation config: "
                f"configured={inputs.eos_token_ids}, model={observed_eos}"
            )
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("answer-utility evaluation changed tokenizer length")
        runtime.adapter.load_artifact_state_dict(inputs.candidate.adapter_state)
        runtime.adapter.requires_grad_(False)
        runtime.adapter.eval()
        model.requires_grad_(False)
        model.eval()
        family_adapter = Qwen3VLAdapter()
        group_builder = Qwen3NativeRepresentationGroupBuilder(
            runtime=runtime,
            family_adapter=family_adapter,
            prompt=inputs.training.prompt,
            image_loader=_load_rgb_image,
            image_max_pixels=inputs.training.model.image_max_pixels,
        )
        truth_by_id = {truth.sample_id: truth for _model, truth in model_truth_rows}
        started = time.monotonic()
        parity_path = output / "image_only_native_parity.json"
        parity_checked = AnswerUtilityEvaluationArm.IMAGE_ONLY not in inputs.arms
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
        for group_ordinal, samples in inputs.selected_groups:
            group_models = tuple(
                split_oracle_d_utility_sample(sample)[0] for sample in samples
            )
            if all(
                ledger.has(row.sample_id, arm.value)
                for row in group_models
                for arm in inputs.arms
            ):
                continue
            visuals = materialize_oracle_group_visuals(
                model_inputs=group_models,
                runtime=runtime,
                group_builder=group_builder,
            )
            wrong_image_visuals = None
            wrong_image_donor = None
            if (
                AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
                in inputs.arms
                and any(
                    not ledger.has(
                        row.sample_id,
                        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value,
                    )
                    for row in group_models
                )
            ):
                wrong_image_donor = inputs.same_target_wrong_image_by_group_key.get(
                    group_models[0].image_group_key
                )
                if wrong_image_donor is None:
                    raise RuntimeError("same-target wrong-image donor is missing")
                if (
                    file_sha256(group_models[0].image)
                    != wrong_image_donor.anchor_image_sha256
                    or file_sha256(wrong_image_donor.donor_image)
                    != wrong_image_donor.donor_image_sha256
                ):
                    raise RuntimeError(
                        "anchor or wrong-image donor bytes changed after identity binding"
                    )
                wrong_image_models = _same_target_wrong_image_model_inputs(
                    group_models, wrong_image_donor
                )
                wrong_image_visuals = materialize_oracle_group_visuals(
                    model_inputs=wrong_image_models,
                    runtime=runtime,
                    group_builder=group_builder,
                )
                if wrong_image_visuals.image_grid_thw != visuals.image_grid_thw or (
                    wrong_image_visuals.image_grid_thw
                    != wrong_image_donor.image_grid_thw
                ):
                    raise RuntimeError(
                        "runtime Qwen grid differs from the bound wrong-image mapping"
                    )
            if not parity_checked:
                first = group_models[0]
                first_reader = _reader_model_input(first)
                parity_context = prepare_oracle_arm_context(
                    model_input=first_reader,
                    arm=OracleDUtilityArm.IMAGE_ONLY,
                    runtime=runtime,
                    source=visuals.source,
                    correct_d=visuals.correct_d_by_sample_id[first.sample_id],
                    image_grid_thw=visuals.image_grid_thw,
                )
                parity = verify_image_only_injected_native_parity(
                    model_input=first_reader,
                    context=parity_context,
                    runtime=runtime,
                    family_adapter=family_adapter,
                    image_loader=group_builder.image_loader,
                    image_max_pixels=group_builder.image_max_pixels,
                )
                _atomic_write_json(
                    parity_path,
                    {
                        "schema_version": "answer_utility_image_only_parity_v1",
                        "run_identity_sha256": ledger.identity_sha256,
                        **asdict(parity),
                    },
                )
                parity_checked = True
            by_id = {row.sample_id: row for row in group_models}
            for row in group_models:
                reader_row = _reader_model_input(row)
                correct_d = visuals.correct_d_by_sample_id[row.sample_id]
                wrong_source_id = inputs.wrong_source_by_sample_id[row.sample_id]
                wrong_d = visuals.correct_d_by_sample_id[wrong_source_id]
                same_target_wrong_image_d = (
                    None
                    if wrong_image_visuals is None
                    else wrong_image_visuals.correct_d_by_sample_id[row.sample_id]
                )
                pending: list[tuple[AnswerUtilityEvaluationArm, OracleArmContext]] = []
                for arm in inputs.arms:
                    if ledger.has(row.sample_id, arm.value):
                        continue
                    context = _prepare_context(
                        arm=arm,
                        model_input=reader_row,
                        runtime=runtime,
                        source=visuals.source,
                        correct_d=correct_d,
                        wrong_d=wrong_d,
                        same_target_wrong_image_d=same_target_wrong_image_d,
                        image_grid_thw=visuals.image_grid_thw,
                    )
                    pending.append((arm, context))
                generated_answers = _generate_pending_answers(
                    contexts=tuple(context for _arm, context in pending),
                    runtime=runtime,
                    family_adapter=family_adapter,
                    eos_token_ids=inputs.eos_token_ids,
                    max_new_tokens=inputs.max_new_tokens,
                    decode_mode=inputs.decode_mode,
                    arm_batch_size=inputs.arm_batch_size,
                )
                for (arm, context), generated in zip(
                    pending,
                    generated_answers,
                    strict=True,
                ):
                    score = score_instruct_generated_answer(
                        generated.text,
                        truth_by_id[row.sample_id],
                        generation_stop_reason=generated.stop_reason,
                    )
                    record: dict[str, Any] = {
                        "schema_version": ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION,
                        "oracle_record_schema_compatibility": (
                            ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION
                        ),
                        "run_identity_sha256": ledger.identity_sha256,
                        "candidate_kind": inputs.candidate.kind,
                        "candidate_id": inputs.candidate.candidate_id,
                        "candidate_training_run_identity_sha256": (
                            inputs.candidate.training_run_identity_sha256
                        ),
                        "sample_id": row.sample_id,
                        "sample_content_sha256": row.sample_content_sha256,
                        "reader_question_sha256": sha256(
                            reader_row.question.encode("utf-8")
                        ).hexdigest(),
                        "reader_contract_version": (INSTRUCT_READER_CONTRACT_VERSION),
                        "scoring_contract_version": (INSTRUCT_SCORING_CONTRACT_VERSION),
                        "image_group_key": row.image_group_key,
                        "group_ordinal": group_ordinal,
                        "arm": arm.value,
                        "arm_contract": _evaluation_arm_contract(arm),
                        "wrong_d_source_sample_id": (
                            wrong_source_id if _uses_wrong_d(arm) else None
                        ),
                        "wrong_d_source_target_sha256": (
                            sha256(
                                by_id[wrong_source_id].target.encode("utf-8")
                            ).hexdigest()
                            if _uses_wrong_d(arm)
                            else None
                        ),
                        "rendered_prefix_text_sha256": context.rendered_text_sha256,
                        "canonical_prefix_token_ids_sha256": (
                            context.canonical_token_ids_sha256
                        ),
                        "prefix_token_count": int(context.prefix_input_ids.shape[1]),
                        "source_visual_token_count": len(context.source_positions),
                        "d_visual_token_count": len(context.d_positions),
                        "generated_token_ids": list(generated.token_ids),
                        "generated_text": generated.text,
                        "generation_stop_reason": generated.stop_reason,
                        "score": asdict(score),
                        "expected_short_answer": truth_by_id[
                            row.sample_id
                        ].short_answer,
                        "elapsed_seconds_since_process_start": (
                            time.monotonic() - started
                        ),
                    }
                    if _uses_same_target_wrong_image_d(arm):
                        if wrong_image_donor is None:
                            raise RuntimeError(
                                "wrong-image record lost its bound donor"
                            )
                        record.update(
                            {
                                "wrong_image_source_sample_id": (
                                    wrong_image_donor.donor_sample_id
                                ),
                                "wrong_image_source_sample_content_sha256": (
                                    wrong_image_donor.donor_sample_content_sha256
                                ),
                                "wrong_image_source_group_key": (
                                    wrong_image_donor.donor_image_group_key
                                ),
                                "wrong_image_source_file_sha256": (
                                    wrong_image_donor.donor_image_sha256
                                ),
                                "anchor_image_file_sha256": (
                                    wrong_image_donor.anchor_image_sha256
                                ),
                                "wrong_image_grid_thw": list(
                                    wrong_image_donor.image_grid_thw
                                ),
                                "wrong_image_match_tier": (
                                    wrong_image_donor.match_tier
                                ),
                                "d_materialization_question_sha256": sha256(
                                    row.question.encode("utf-8")
                                ).hexdigest(),
                                "d_materialization_target_sha256": sha256(
                                    row.target.encode("utf-8")
                                ).hexdigest(),
                                "d_materialization_contextual_hidden_state": (
                                    "recomputed_on_wrong_image_with_anchor_question_and_target"
                                ),
                            }
                        )
                    ledger.commit(record)
            del visuals
            if wrong_image_visuals is not None:
                del wrong_image_visuals
        if len(processor.tokenizer) != tokenizer_length_before:
            raise RuntimeError("answer-utility evaluation changed tokenizer length")
        return _extended_summary(ledger)


def _input_loader_bindings() -> _InputLoaderBindings:
    """Capture the historical facade hooks at the call boundary."""

    return _InputLoaderBindings(
        normalize_evaluation_arms=_normalize_evaluation_arms,
        load_run_config=load_answer_utility_run_config,
        load_experiment_config=load_answer_utility_experiment_config,
        load_training_config=load_representation_training_config,
        load_source_evaluation_config=(
            load_representation_internal_evaluation_run_config
        ),
        require_instruct_training=_require_instruct_training,
        validate_private_source_bindings=_validate_private_source_bindings,
        load_validated_production_export=_load_validated_production_export,
        load_adapter_artifact=load_answer_utility_adapter_artifact,
        audit_completed_training_metrics=_audit_completed_training_metrics,
        materialize_common_inputs=_materialize_common_inputs,
    )


def _load_private_inputs(
    run_config_path: str | Path,
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int = 1,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> AnswerUtilityEvaluationInputs:
    return _load_private_inputs_impl(
        run_config_path,
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
        bindings=_input_loader_bindings(),
    )


def _load_production_source_inputs(
    source_evaluation_config_path: str | Path,
    *,
    arms: Sequence[AnswerUtilityEvaluationArm | str],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int = 1,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> AnswerUtilityEvaluationInputs:
    return _load_production_source_inputs_impl(
        source_evaluation_config_path,
        arms=arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
        bindings=_input_loader_bindings(),
    )


def _materialize_common_inputs(
    *,
    training: RepresentationTrainingConfig,
    source_evaluation: Any,
    candidate: AnswerUtilityEvaluationCandidate,
    selected_arms: tuple[AnswerUtilityEvaluationArm, ...],
    max_new_tokens: int | None,
    eos_token_ids: Sequence[int] | None,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> AnswerUtilityEvaluationInputs:
    return _materialize_common_inputs_impl(
        training=training,
        source_evaluation=source_evaluation,
        candidate=candidate,
        selected_arms=selected_arms,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        decode_mode=decode_mode,
        arm_batch_size=arm_batch_size,
        group_start=group_start,
        group_limit=group_limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _validate_private_source_bindings(
    run: AnswerUtilityRunConfig,
    experiment: Any,
    source_evaluation: Any,
) -> None:
    _validate_private_source_bindings_impl(run, experiment, source_evaluation)


def _load_validated_production_export(
    training: RepresentationTrainingConfig,
    source_evaluation: Any,
) -> Any:
    return _load_validated_production_export_impl(
        training,
        source_evaluation,
        file_sha256_fn=file_sha256,
        load_export_fn=load_rank_zero_adapter_owned_state_export,
        state_digest_fn=state_digest,
        validate_training_artifact_binding_fn=_validate_training_artifact_binding,
    )


def _require_instruct_training(training: RepresentationTrainingConfig) -> None:
    _require_instruct_training_impl(training)


def _audit_completed_training_metrics(
    run: AnswerUtilityRunConfig,
    expected_variant: str,
    artifact: AnswerUtilityAdapterArtifact,
) -> None:
    _audit_completed_training_metrics_impl(run, expected_variant, artifact)


def _generate_pending_answers(
    *,
    contexts: Sequence[OracleArmContext],
    runtime: Any,
    family_adapter: Any,
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
    decode_mode: Literal["cached", "no_cache"],
    arm_batch_size: int,
) -> tuple[OracleGeneratedAnswer, ...]:
    """Decode same-row arms in bounded compatible batches with exact fallback."""

    pending = tuple(contexts)
    selected_batch_size = _normalize_arm_batch_size(arm_batch_size)
    generated: list[OracleGeneratedAnswer] = []
    for start in range(0, len(pending), selected_batch_size):
        batch = pending[start : start + selected_batch_size]
        if decode_mode == "cached" and len(batch) > 1:
            try:
                generated.extend(
                    greedy_oracle_answers_batched(
                        contexts=batch,
                        runtime=runtime,
                        family_adapter=family_adapter,
                        eos_token_ids=eos_token_ids,
                        max_new_tokens=max_new_tokens,
                    )
                )
                continue
            except OracleBatchCompatibilityError:
                pass
        generated.extend(
            greedy_oracle_answer(
                context=context,
                runtime=runtime,
                family_adapter=family_adapter,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                decode_mode=decode_mode,
            )
            for context in batch
        )
    return tuple(generated)


def _prepare_context(
    *,
    arm: AnswerUtilityEvaluationArm,
    model_input: Any,
    runtime: Any,
    source: Any,
    correct_d: Any,
    wrong_d: Any,
    same_target_wrong_image_d: Any | None,
    image_grid_thw: tuple[int, int, int],
) -> OracleArmContext:
    oracle_arm = _oracle_arm_for(arm)
    context = prepare_oracle_arm_context(
        model_input=model_input,
        arm=oracle_arm,
        runtime=runtime,
        source=source,
        correct_d=correct_d,
        matched_wrong_d=wrong_d,
        image_grid_thw=image_grid_thw,
    )
    if arm not in {
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE,
    }:
        return context
    if not context.d_positions or len(context.visual_blocks) != 2:
        raise RuntimeError("image-plus-correct template did not produce source plus D")
    selected_d = (
        same_target_wrong_image_d
        if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
        else wrong_d
    )
    if selected_d is None:
        raise ValueError("same-target wrong-image arm requires its bound D")
    blocks = (
        *context.visual_blocks[:-1],
        _injected_block("focused_d", context.d_positions, selected_d),
    )
    return replace(context, visual_blocks=blocks)


def _reader_model_input(model_input: Any) -> Any:
    """Change only the answer reader prompt, never D materialization inputs."""

    return replace(model_input, question=reader_question(model_input.question))


def _uses_same_target_wrong_image_d(arm: AnswerUtilityEvaluationArm) -> bool:
    return arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE


__all__ = [
    "ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION",
    "DEFAULT_INSTRUCT_EOS_TOKEN_IDS",
    "DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS",
    "AnswerUtilityAdapterArtifact",
    "AnswerUtilityEvaluationArm",
    "AnswerUtilityEvaluationCandidate",
    "AnswerUtilityWrongImageDonor",
    "build_answer_safe_wrong_mapping",
    "build_same_target_wrong_image_mapping",
    "load_answer_utility_adapter_artifact",
    "run_answer_utility_evaluation",
    "run_production_source_answer_utility_evaluation",
    "validate_answer_utility_evaluation",
    "validate_production_source_answer_utility_evaluation",
]
