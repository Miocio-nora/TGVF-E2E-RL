"""Identity payloads for materialized answer-utility evaluation inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from tgvf_rl.protocol.native import NativeAssistantDialect
from tgvf_rl.public_api_compat import rebind_public_function
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleDUtilityArm,
    _OracleRunLedger,
    _arm_contract,
    _atomic_write_json,
    _paired_summary,
)

from .input_matching import AnswerUtilityWrongImageDonor
from .inputs import (
    ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
    AnswerUtilityEvaluationArm,
    AnswerUtilityEvaluationInputs,
    _has_same_target_wrong_image_arm,
    _normalize_arm_batch_size,
)
from .scoring import (
    INSTRUCT_READER_CONTRACT_VERSION,
    INSTRUCT_READER_INSTRUCTION,
    INSTRUCT_SCORING_CONTRACT_VERSION,
    reader_question,
)


def _validated_payload(inputs: AnswerUtilityEvaluationInputs) -> dict[str, object]:
    samples = tuple(
        sample for _ordinal, group in inputs.selected_groups for sample in group
    )
    candidate = inputs.candidate
    payload: dict[str, object] = {
        "schema_version": ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
        "status": "validated",
        "candidate_kind": candidate.kind,
        "candidate_id": candidate.candidate_id,
        "training_model_name": inputs.training.model.model_name,
        "training_model_path": str(inputs.training.model.local_path),
        "candidate_training_run_identity_sha256": (
            candidate.training_run_identity_sha256
        ),
        "candidate_artifact_file_sha256": candidate.adapter_file_sha256,
        "candidate_adapter_state_sha256": candidate.adapter_state_sha256,
        "candidate_global_step": candidate.global_step,
        "private_run_id": candidate.private_run_id,
        "production_source_artifact_sha256": (
            candidate.production_source_artifact_sha256
        ),
        "production_source_manifest_sha256": (
            candidate.production_source_manifest_sha256
        ),
        "production_source_run_identity_sha256": (
            candidate.production_source_run_identity_sha256
        ),
        "production_source_global_step": candidate.production_source_global_step,
        "evaluation_data_manifest_sha256": inputs.data_manifest_sha256,
        "ordered_group_manifest_identity": inputs.ordered_group_manifest_identity,
        "selected_group_ordinals": [
            ordinal for ordinal, _group in inputs.selected_groups
        ],
        "selected_group_count": len(inputs.selected_groups),
        "selected_sample_count": len(samples),
        "arms": [arm.value for arm in inputs.arms],
        "answer_safe_wrong_mapping_count": len(inputs.wrong_source_by_sample_id),
        "max_new_tokens": inputs.max_new_tokens,
        "eos_token_ids": list(inputs.eos_token_ids),
        "decode_mode": inputs.decode_mode,
        "arm_batch_size": inputs.arm_batch_size,
        "decoder_implementation": _decoder_implementation(inputs),
        "shard_index": inputs.shard_index,
        "shard_count": inputs.shard_count,
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        payload.update(
            {
                "same_target_wrong_image_mapping_count": len(
                    inputs.same_target_wrong_image_by_group_key
                ),
                "wrong_image_pool_manifest_sha256": (
                    inputs.wrong_image_pool_manifest_sha256
                ),
                "wrong_image_pool_source_sha256": (
                    inputs.training.data.train.source_sha256
                ),
                "wrong_image_match_tiers": dict(
                    _wrong_image_match_tier_counts(
                        inputs.same_target_wrong_image_by_group_key
                    )
                ),
            }
        )
    return payload


def _build_evaluation_identity_payload(
    inputs: AnswerUtilityEvaluationInputs,
    model_inputs: Sequence[Any],
    *,
    implementation_file_manifest: Callable[[], Mapping[str, str]],
    evaluation_arm_contract: Callable[[Any], Mapping[str, Any]],
) -> dict[str, Any]:
    implementation_files = implementation_file_manifest()
    candidate = inputs.candidate
    arm_batch_size = _normalize_arm_batch_size(getattr(inputs, "arm_batch_size", 1))
    payload: dict[str, Any] = {
        "schema_version": ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
        "claim_scope": "oracle_target_conditioned_D_utility_not_end_to_end_tool_selection",
        "candidate_kind": candidate.kind,
        "candidate_id": candidate.candidate_id,
        "candidate_artifact_path": str(candidate.adapter_path),
        "candidate_artifact_file_sha256": candidate.adapter_file_sha256,
        "candidate_adapter_state_sha256": candidate.adapter_state_sha256,
        "candidate_training_run_identity_sha256": (
            candidate.training_run_identity_sha256
        ),
        "candidate_global_step": candidate.global_step,
        "private_run_id": candidate.private_run_id,
        "private_run_config_path": (
            None
            if candidate.private_run_config_path is None
            else str(candidate.private_run_config_path)
        ),
        "private_run_config_sha256": candidate.private_run_config_sha256,
        "private_experiment_config_sha256": (
            candidate.private_experiment_config_sha256
        ),
        "production_source_artifact_path": str(
            candidate.production_source_artifact_path
        ),
        "production_source_artifact_sha256": (
            candidate.production_source_artifact_sha256
        ),
        "production_source_manifest_sha256": (
            candidate.production_source_manifest_sha256
        ),
        "production_source_run_identity_sha256": (
            candidate.production_source_run_identity_sha256
        ),
        "production_source_global_step": candidate.production_source_global_step,
        "source_evaluation_config_path": str(inputs.source_evaluation.source_path),
        "source_evaluation_config_sha256": inputs.source_evaluation.source_sha256,
        "model_name": inputs.training.model.model_name,
        "model_path": str(inputs.training.model.local_path),
        "assistant_dialect": NativeAssistantDialect.QWEN3_VL_INSTRUCT.value,
        "data_manifest_sha256": inputs.data_manifest_sha256,
        "ordered_group_manifest_identity": inputs.ordered_group_manifest_identity,
        "ordered_group_manifest_sha256": (
            inputs.source_evaluation.evaluation.ordered_group_manifest_sha256
        ),
        "ordered_selected_samples": [
            _selected_sample_identity(inputs, row) for row in model_inputs
        ],
        "arms": [arm.value for arm in inputs.arms],
        "arm_contracts": {
            arm.value: evaluation_arm_contract(arm) for arm in inputs.arms
        },
        "max_new_tokens": inputs.max_new_tokens,
        "eos_token_ids": list(inputs.eos_token_ids),
        "decode_mode": inputs.decode_mode,
        "arm_batch_size": arm_batch_size,
        "decoder_implementation": _decoder_implementation(inputs),
        "greedy": True,
        "random_seed": inputs.source_evaluation.evaluation.random_seed,
        "group_start": inputs.group_start,
        "group_limit": inputs.group_limit,
        "shard_index": inputs.shard_index,
        "shard_count": inputs.shard_count,
        "wrong_d_rule": "same_image_different_normalized_target_and_answer_cyclic_v1",
        "d_materialization_question": "original_unmodified_question",
        "reader_contract_version": INSTRUCT_READER_CONTRACT_VERSION,
        "reader_instruction": INSTRUCT_READER_INSTRUCTION,
        "reader_instruction_sha256": sha256(
            INSTRUCT_READER_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "reader_instruction_applies_to_all_arms": True,
        "scoring_contract_version": INSTRUCT_SCORING_CONTRACT_VERSION,
        "scoring_choices_model_visible": False,
        "scoring_label_route_enabled": False,
        "choice_text_reference_mapping_enabled": True,
        "ground_truth_model_input": False,
        "post_focus_transcript_model_input": False,
        "implementation_file_manifest": implementation_files,
        "implementation_sha256": sha256(
            json.dumps(
                implementation_files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        if inputs.wrong_image_pool_manifest_sha256 is None:
            raise RuntimeError("wrong-image arm lost its donor-pool identity")
        ordered_donors = tuple(
            inputs.same_target_wrong_image_by_group_key[group[0].image_group_key]
            for _ordinal, group in inputs.selected_groups
        )
        payload.update(
            {
                "wrong_image_negative_pool_path": str(
                    inputs.training.data.train.jsonl_path
                ),
                "wrong_image_negative_pool_manifest_sha256": (
                    inputs.wrong_image_pool_manifest_sha256
                ),
                "wrong_image_negative_pool_source_sha256": (
                    inputs.training.data.train.source_sha256
                ),
                "wrong_image_mapping_rule": (
                    "rp66_train_exact_qwen_grid_answer_disjoint_distinct_bytes_"
                    "tiered_source_profile_domain_sha256_rank_v1"
                ),
                "wrong_image_mapping_random_seed": (
                    inputs.source_evaluation.evaluation.random_seed
                ),
                "wrong_image_match_tiers": dict(
                    _wrong_image_match_tier_counts(
                        inputs.same_target_wrong_image_by_group_key
                    )
                ),
                "same_target_wrong_image_mapping": [
                    {
                        **asdict(donor),
                        "image_grid_thw": list(donor.image_grid_thw),
                    }
                    for donor in ordered_donors
                ],
                "wrong_image_d_materialization_question": (
                    "anchor_original_unmodified_question"
                ),
                "wrong_image_d_materialization_target": "anchor_oracle_target",
                "wrong_image_contextual_hidden_state": (
                    "recomputed_by_frozen_qwen_on_wrong_image_anchor_question_target"
                ),
                "wrong_image_reader_source": (
                    "anchor_image_anchor_question_anchor_target"
                ),
            }
        )
    return payload


def _selected_sample_identity(
    inputs: AnswerUtilityEvaluationInputs, row: Any
) -> dict[str, Any]:
    identity = {
        "sample_id": row.sample_id,
        "sample_content_sha256": row.sample_content_sha256,
        "image_group_key": row.image_group_key,
        "reader_question_sha256": sha256(
            reader_question(row.question).encode("utf-8")
        ).hexdigest(),
        "wrong_d_source_sample_id": inputs.wrong_source_by_sample_id[row.sample_id],
    }
    if _has_same_target_wrong_image_arm(inputs.arms):
        donor = inputs.same_target_wrong_image_by_group_key[row.image_group_key]
        identity.update(
            {
                "wrong_image_donor_sample_id": donor.donor_sample_id,
                "wrong_image_donor_sample_content_sha256": (
                    donor.donor_sample_content_sha256
                ),
                "wrong_image_donor_group_key": donor.donor_image_group_key,
                "wrong_image_donor_file_sha256": donor.donor_image_sha256,
                "wrong_image_match_tier": donor.match_tier,
                "wrong_image_grid_thw": list(donor.image_grid_thw),
                "d_materialization_question_sha256": sha256(
                    row.question.encode("utf-8")
                ).hexdigest(),
                "d_materialization_target_sha256": sha256(
                    row.target.encode("utf-8")
                ).hexdigest(),
            }
        )
    return identity


def _implementation_file_manifest_impl() -> dict[str, str]:
    """Bind every local implementation family used by generation/scoring."""

    source_root = Path(__file__).resolve().parents[5]
    patterns = (
        "tgvf_rl/checkpoint/*.py",
        "tgvf_rl/conditioning/*.py",
        "tgvf_rl/protocol/*.py",
        "tgvf_rl/qwen/*.py",
        "tgvf_rl/representation/adapter.py",
        "tgvf_rl/representation/training/*.py",
        "tgvf_rl/representation/experiments/answer_utility/*.py",
        "tgvf_rl/representation/experiments/answer_utility/evaluation/*.py",
    )
    files = tuple(
        sorted(
            {path for pattern in patterns for path in source_root.glob(pattern)},
            key=lambda path: str(path.relative_to(source_root)),
        )
    )
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("answer-utility evaluation implementation files are missing")
    return {
        str(path.relative_to(source_root)): sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _decoder_implementation(inputs: AnswerUtilityEvaluationInputs) -> str:
    arm_batch_size = _normalize_arm_batch_size(getattr(inputs, "arm_batch_size", 1))
    if inputs.decode_mode == "cached" and arm_batch_size > 1:
        return "same_sample_compatible_arm_batched_cached_greedy_v1"
    return "scalar_greedy_v1"


def _wrong_image_match_tier_counts(
    mapping: Mapping[str, AnswerUtilityWrongImageDonor],
) -> dict[str, int]:
    counts = {
        "exact_grid_same_source_dataset": 0,
        "exact_grid_same_source_profile": 0,
        "exact_grid_cross_domain": 0,
    }
    for donor in mapping.values():
        counts[donor.match_tier] += 1
    return counts


def _oracle_arm_for(arm: AnswerUtilityEvaluationArm) -> OracleDUtilityArm:
    if arm in {
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE,
    }:
        return OracleDUtilityArm.IMAGE_CORRECT_D
    return OracleDUtilityArm(arm.value)


def _uses_wrong_d(arm: AnswerUtilityEvaluationArm) -> bool:
    return arm in {
        AnswerUtilityEvaluationArm.D_ONLY_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
        AnswerUtilityEvaluationArm.DIRECT_WRONG_REPLACEMENT,
    }


def _evaluation_arm_contract(
    arm: AnswerUtilityEvaluationArm,
) -> dict[str, Any]:
    if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE:
        return {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": (
                "same_anchor_question_target_recomputed_on_exact_grid_"
                "answer_disjoint_distinct_wrong_image_stage1_main_and_all_deepstack"
            ),
            "contextual_target_hidden_state": "recomputed_on_wrong_image",
        }
    if arm is AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG:
        return {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "answer_safe_same_image_different_target_and_answer_stage1_main_and_all_deepstack",
        }
    contract = dict(_arm_contract(_oracle_arm_for(arm)))
    if _uses_wrong_d(arm):
        contract["d"] = (
            "answer_safe_same_image_different_target_and_answer_stage1_main_and_all_deepstack"
        )
    return contract


def _extended_summary(ledger: _OracleRunLedger) -> dict[str, object]:
    summary = ledger.summary()
    records = tuple(ledger._completed[key] for key in ledger.expected_keys)  # noqa: SLF001
    paired = dict(summary["paired_effects"])
    comparisons = (
        (
            "D_only_specificity",
            AnswerUtilityEvaluationArm.D_ONLY_CORRECT.value,
            AnswerUtilityEvaluationArm.D_ONLY_WRONG.value,
        ),
        (
            "image_plus_D_specificity",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG.value,
        ),
        (
            "image_plus_wrong_vs_zero",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_ZERO.value,
        ),
        (
            "image_plus_D_image_grounding",
            AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT.value,
            AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value,
        ),
    )
    available = set(summary["arms"])
    for name, treatment, control in comparisons:
        if treatment in available and control in available:
            paired[name] = _paired_summary(records, treatment, control)
    summary["paired_effects"] = paired
    summary["evaluation_schema_version"] = ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION
    summary["arm_batch_size"] = ledger.identity_payload.get("arm_batch_size", 1)
    summary["decoder_implementation"] = ledger.identity_payload.get(
        "decoder_implementation",
        "scalar_greedy_v1",
    )
    interpretation = (
        "Correct-vs-zero measures D content utility; correct-vs-answer-safe-wrong "
        "measures target specificity. All D arms condition on an oracle target and do "
        "not measure autonomous tool selection."
    )
    if AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE.value in available:
        interpretation += (
            " Correct-vs-same-target-wrong-image measures image dependence while "
            "holding the reader image, question, target, and D geometry fixed."
        )
    summary["interpretation"] = interpretation
    _atomic_write_json(ledger.summary_path, summary)
    return summary


_PUBLIC_RUNNER_MODULE = (
    "tgvf_rl.representation.experiments.answer_utility.evaluation.runner"
)
for _helper in (
    _evaluation_arm_contract,
    _extended_summary,
    _oracle_arm_for,
    _uses_wrong_d,
):
    rebind_public_function(
        _helper,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNNER_MODULE,
    )
del _helper


__all__ = [
    "_build_evaluation_identity_payload",
    "_decoder_implementation",
    "_evaluation_arm_contract",
    "_extended_summary",
    "_implementation_file_manifest_impl",
    "_oracle_arm_for",
    "_uses_wrong_d",
    "_validated_payload",
    "_wrong_image_match_tier_counts",
]
