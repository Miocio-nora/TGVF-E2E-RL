"""Executable internal evaluation for a representation-phase artifact.

The runner consumes the same family adapter, frozen Qwen model, TGVF Adapter,
and group-builder interface as representation training.  It retains full
lower-is-better NLL matrices and evaluates every latent observation as one
atomic main-``D`` plus ordered D-DeepStack bundle.

Native counterfactual generation remains family/runtime owned.  It enters this
module through mandatory typed callbacks, which makes the seam executable in
CPU fixtures without pretending that a tiny fixture is a real generation
backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import math
from pathlib import Path

import torch
from torch import nn

from tgvf_rl.conditioning.base import (
    TargetConditioningProviderKind as TargetConditioningProviderKind,
)
from tgvf_rl.qwen.base import (
    InjectedForwardRequest,
    InjectedVisualBlock,
    QwenVLMFamilyAdapter,
)
from tgvf_rl.representation.adapter import TGVFAdapter
from tgvf_rl.representation.deepstack import build_original_image_key_block_mask

from .internal_evaluation_contract import (
    ATTENTION_TOPK,
    CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT,
    DETERMINISTIC_RANDOM_D_ALGORITHM,
    READOUT_FORWARD_PATH,
    READOUT_MASK_MODE,
    READOUT_POSITION_SOURCE,
    READOUT_VISUAL_SWAP_UNIT,
    REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION,
    TARGET_PRESENCE_LOG_ODDS_CONTRACT,
    ContinuationStopReason as ContinuationStopReason,
    ContinuationVariant as ContinuationVariant,
    NativeBinaryLogOddsEvaluator,
    NativeBinaryLogOddsOutput,
    NativeBinaryLogOddsRequest,
    NativeCausalValueFlipEvaluator,
    NativeCausalValueFlipOutput,
    NativeCausalValueFlipRequest,
    NativeContinuationCacheParity,
    NativeCounterfactualCase,
    NativeCounterfactualEvaluationRecord,
    NativeCounterfactualSummary,
    NativeDOnlyContext,
    NativeFreeContinuationEvaluator,
    NativeFreeContinuationOutput,
    NativeFreeContinuationRecord,
    NativeFreeContinuationRequest,
    NativeGenerationForward,
    NativeInjectedRequestMaterializer,
    NativeTargetPresenceCase,
    NativeTargetPresenceContinuationRecord,
    NativeTargetPresenceEvaluationRecord,
    NativeTargetPresenceSummary,
    NativeTeacherForcedForward,
    RepresentationBranchHealthRecord,
    RepresentationBranchHealthSummary,
    RepresentationGroupedInternalMetrics,
    RepresentationInternalEvaluationGroupRecord,
    RepresentationInternalEvaluationIdentity,
    RepresentationInternalEvaluationReport,
    RepresentationInternalEvaluationSampleRecord,
    RepresentationInternalHealthSummary,
    RepresentationReadoutControlIdentity,
    RepresentationReadoutExecutionContract,
    RepresentationSampleHealthRecord,
    TargetPresenceVariant as TargetPresenceVariant,
    _assert_visual_bundle_contract as _assert_visual_bundle_contract,
    _finite_float as _finite_float,
    _lowercase_sha256 as _lowercase_sha256,
    _non_empty_text as _non_empty_text,
    _non_negative_int as _non_negative_int,
)
from .internal_evaluation_native_runtime import (
    InjectedNativeCounterfactualEvaluator,
    _evaluate_native_case,
    _evaluate_target_presence_case,
    _greedy_token_id as _greedy_token_id,
    _summarize_counterfactuals,
    _summarize_target_presence,
    _validate_materialized_request as _validate_materialized_request,
    _validated_generated_ids as _validated_generated_ids,
    _zero_visual_bundle as _zero_visual_bundle,
    create_injected_native_counterfactual_evaluator,
)
from .losses import causal_evidence_losses
from .internal_evaluation_artifact import (
    REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION,
    RepresentationInternalEvaluationArtifact,
    _publish_representation_internal_evaluation_report_atomic,
)
from .metrics import (
    AttentionDiagnostics as AttentionDiagnostics,
    AttentionDiagnosticsSummary as AttentionDiagnosticsSummary,
    NormComparisonDiagnostics as NormComparisonDiagnostics,
    QueryMetrics as QueryMetrics,
    QueryScoreMatrixMetrics as QueryScoreMatrixMetrics,
    ReadoutMetrics as ReadoutMetrics,
    ReadoutNLLs,
    ReadoutSampleMetrics as ReadoutSampleMetrics,
    RepresentationHealthSummary as RepresentationHealthSummary,
    attention_diagnostics,
    grouped_query_row_metrics,
    grouped_readout_metrics,
    norm_comparison_diagnostics,
    query_score_matrix_metrics,
    readout_sample_metrics,
    summarize_attention_diagnostics,
    summarize_query_score_matrices,
    summarize_readout,
    summarize_representation_health,
)
from .readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    assert_frozen_deterministic_readout_model,
)
from .schema import RepresentationTrainingSample
from .streaming import StreamingGroupScores, score_streaming_same_image_group
from .trainer import RepresentationGroupBuilder


_UNSPECIFIED_GROUP_LABEL = "__unspecified__"
_GROUPING_DIMENSIONS = (
    "evidence_type",
    "source_profile",
    "answer_type",
    "visual_difficulty",
)


def run_representation_internal_evaluation(
    *,
    identity: RepresentationInternalEvaluationIdentity,
    adapter: TGVFAdapter,
    qwen_model: nn.Module,
    family_adapter: QwenVLMFamilyAdapter,
    sample_groups: Sequence[Sequence[RepresentationTrainingSample]],
    group_builder: RepresentationGroupBuilder,
    native_counterfactual_cases: Sequence[NativeCounterfactualCase],
    causal_value_flip_evaluator: NativeCausalValueFlipEvaluator,
    free_continuation_evaluator: NativeFreeContinuationEvaluator,
    native_target_presence_cases: Sequence[NativeTargetPresenceCase] = (),
    target_presence_log_odds_evaluator: NativeBinaryLogOddsEvaluator | None = None,
    target_presence_free_continuation_evaluator: (
        NativeFreeContinuationEvaluator | None
    ) = None,
) -> RepresentationInternalEvaluationReport:
    """Run all accepted historical controls and native causal evaluations."""

    if not isinstance(identity, RepresentationInternalEvaluationIdentity):
        raise TypeError("identity must be RepresentationInternalEvaluationIdentity")
    if not isinstance(adapter, TGVFAdapter):
        raise TypeError("adapter must be a TGVFAdapter")
    if not isinstance(qwen_model, nn.Module):
        raise TypeError("qwen_model must be an nn.Module")
    if not isinstance(family_adapter, QwenVLMFamilyAdapter):
        raise TypeError("family_adapter must be a QwenVLMFamilyAdapter")
    if not callable(group_builder):
        raise TypeError("group_builder must be callable")
    if not callable(causal_value_flip_evaluator) or not callable(
        free_continuation_evaluator
    ):
        raise TypeError("native internal evaluators must be callable")
    groups_of_samples = _validated_sample_groups(sample_groups)
    native_cases = _validated_native_cases(native_counterfactual_cases)
    target_presence_cases = _validated_target_presence_cases(
        native_target_presence_cases
    )
    if target_presence_cases and (
        target_presence_log_odds_evaluator is None
        or target_presence_free_continuation_evaluator is None
    ):
        raise ValueError("target-presence cases require both native evaluators")
    if not target_presence_cases and (
        target_presence_log_odds_evaluator is not None
        or target_presence_free_continuation_evaluator is not None
    ):
        raise ValueError("target-presence evaluators require target-presence cases")
    if family_adapter.capabilities.family != native_cases[0].context.family:
        raise ValueError("native counterfactual family differs from Qwen adapter")
    if any(
        case.context.family != family_adapter.capabilities.family
        for case in native_cases
    ):
        raise ValueError("native counterfactual cases cannot mix Qwen families")
    if any(
        len(observation.deepstack) != family_adapter.capabilities.deepstack_branch_count
        for case in native_cases
        for observation in (case.observation_a, case.observation_b)
    ):
        raise ValueError("native counterfactual branch count differs from Qwen family")
    if any(
        context.family != family_adapter.capabilities.family
        for case in target_presence_cases
        for context in (case.positive_context, case.negative_context)
    ):
        raise ValueError("target-presence cases differ from Qwen family")
    if any(
        len(observation.deepstack) != family_adapter.capabilities.deepstack_branch_count
        for case in target_presence_cases
        for observation in (
            case.positive_observation,
            case.negative_observation,
        )
    ):
        raise ValueError("target-presence branch count differs from Qwen family")

    assert_frozen_deterministic_readout_model(qwen_model)
    adapter_modes = tuple((module, module.training) for module in adapter.modules())
    adapter_parameter_state = tuple(
        (parameter, parameter.requires_grad, parameter.grad)
        for parameter in adapter.parameters()
    )
    built_groups: list[SameImageReadoutGroup] = []
    try:
        adapter.eval()
        with torch.no_grad():
            for samples in groups_of_samples:
                group = group_builder(
                    samples,
                    adapter,
                    collective_candidate_count=len(samples),
                )
                _validate_built_group(group, samples, identity=identity)
                built_groups.append(group)
    finally:
        for module, was_training in adapter_modes:
            module.training = was_training
    _assert_adapter_state_unchanged(adapter_parameter_state)

    groups = tuple(built_groups)
    _validate_cross_group_contract(groups)
    samples_by_id = {
        sample.sample_id: sample
        for group_samples in groups_of_samples
        for sample in group_samples
    }

    score_records = tuple(
        score_streaming_same_image_group(family_adapter, qwen_model, group)
        for group in groups
    )
    nll_matrices = tuple(_mean_nll_matrix(record) for record in score_records)
    query_records = tuple(query_score_matrix_metrics(matrix) for matrix in nll_matrices)

    group_records: list[RepresentationInternalEvaluationGroupRecord] = []
    sample_records: list[RepresentationInternalEvaluationSampleRecord] = []
    for group_index, (group, scores, query) in enumerate(
        zip(groups, score_records, query_records, strict=True)
    ):
        wrong_different_pairs = tuple(
            _find_wrong_different_candidate(
                groups,
                group_index=group_index,
                row_index=row_index,
                expected=candidate.visual,
            )
            for row_index, candidate in enumerate(group.candidates)
        )
        paired_group_keys = {
            pair[0].image_group_key
            for pair in wrong_different_pairs
            if pair is not None
        }
        group_records.append(
            RepresentationInternalEvaluationGroupRecord(
                image_group_key=group.image_group_key,
                source_visual_identity=group.source_visual_identity,
                sample_ids=scores.sample_ids,
                wrong_different_image_group_key=(
                    next(iter(paired_group_keys))
                    if len(paired_group_keys) == 1
                    else None
                ),
                query=query,
            )
        )
        for row_index, (row, candidate) in enumerate(
            zip(group.rows, group.candidates, strict=True)
        ):
            same_index = (row_index + 1) % len(group.candidates)
            wrong_same = group.candidates[same_index]
            wrong_different_pair = wrong_different_pairs[row_index]
            different_group, wrong_different = (
                wrong_different_pair
                if wrong_different_pair is not None
                else (None, None)
            )
            random_seed = _random_control_seed(identity, row.sample_id)
            random_visual = _deterministic_random_visual_bundle(
                candidate.visual, seed=random_seed
            )
            token_count = int(scores.evidence_token_counts[row_index].item())
            if token_count <= 0:
                raise RuntimeError("internal readout has no evidence token")
            correct_nll = _token_mean_nll_from_cell_score(
                scores.score_matrix[row_index, row_index], token_count
            )
            diagonal_l_gen = float(scores.diagonal_l_gen[row_index].float().item())
            if not math.isclose(
                correct_nll, diagonal_l_gen, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError("query diagonal and readability NLL diverged")
            wrong_same_nll = _token_mean_nll_from_cell_score(
                scores.score_matrix[row_index, same_index], token_count
            )
            target_only_nll, target_only_count = _score_readout_control(
                family_adapter,
                qwen_model,
                source=group.source_visual,
                row=row,
                candidate=None,
            )
            random_nll, random_count = _score_readout_control(
                family_adapter,
                qwen_model,
                source=group.source_visual,
                row=row,
                candidate=random_visual,
            )
            different_nll: float | None = None
            different_count: int | None = None
            if wrong_different is not None:
                different_nll, different_count = _score_readout_control(
                    family_adapter,
                    qwen_model,
                    source=group.source_visual,
                    row=row,
                    candidate=wrong_different.visual,
                )
            available_counts = {target_only_count, random_count}
            if different_count is not None:
                available_counts.add(different_count)
            if available_counts != {token_count}:
                raise RuntimeError(
                    "control evidence-token counts differ from correct D"
                )
            nlls = ReadoutNLLs(
                correct_d=correct_nll,
                target_only=target_only_nll,
                random_d=random_nll,
                wrong_same_image_d=wrong_same_nll,
                wrong_different_image_d=different_nll,
            )
            sample = samples_by_id[row.sample_id]
            sample_records.append(
                RepresentationInternalEvaluationSampleRecord(
                    sample_id=sample.sample_id,
                    sample_content_sha256=sample.content_sha256,
                    image_group_key=group.image_group_key,
                    source_visual_identity=group.source_visual_identity,
                    target_conditioning_provider=(
                        candidate.target_conditioning_provider.value
                    ),
                    projection_identities=candidate.projection_identities,
                    evidence_type=sample.evidence_type,
                    source_profile=sample.source_profile,
                    answer_type=sample.answer_type,
                    visual_difficulty=sample.visual_difficulty,
                    controls=RepresentationReadoutControlIdentity(
                        correct_candidate_sample_id=candidate.sample_id,
                        wrong_same_image_candidate_sample_id=wrong_same.sample_id,
                        wrong_different_image_candidate_sample_id=(
                            wrong_different.sample_id
                            if wrong_different is not None
                            else None
                        ),
                        wrong_different_image_group_key=(
                            different_group.image_group_key
                            if different_group is not None
                            else None
                        ),
                        random_d_seed=random_seed,
                    ),
                    nlls=nlls,
                    readout=readout_sample_metrics(nlls),
                    health=_sample_health(group, candidate),
                )
            )
    sample_record_tuple = tuple(sample_records)
    readout_rows = tuple(record.nlls for record in sample_record_tuple)
    readout_summary = summarize_readout(readout_rows)
    query_summary = summarize_query_score_matrices(nll_matrices)
    grouped_metrics = _grouped_metrics(
        groups_of_samples,
        sample_record_tuple,
        nll_matrices,
    )
    health = _summarize_health(sample_record_tuple)
    native_records = tuple(
        _evaluate_native_case(
            case,
            causal_value_flip_evaluator=causal_value_flip_evaluator,
            free_continuation_evaluator=free_continuation_evaluator,
        )
        for case in native_cases
    )
    target_presence_records: tuple[NativeTargetPresenceEvaluationRecord, ...] = ()
    target_presence_summary: NativeTargetPresenceSummary | None = None
    if target_presence_cases:
        assert target_presence_log_odds_evaluator is not None
        assert target_presence_free_continuation_evaluator is not None
        target_presence_records = tuple(
            _evaluate_target_presence_case(
                case,
                log_odds_evaluator=target_presence_log_odds_evaluator,
                free_continuation_evaluator=(
                    target_presence_free_continuation_evaluator
                ),
            )
            for case in target_presence_cases
        )
        target_presence_summary = _summarize_target_presence(target_presence_records)
    return RepresentationInternalEvaluationReport(
        identity=identity,
        execution=RepresentationReadoutExecutionContract(
            family=family_adapter.capabilities.family
        ),
        groups=tuple(group_records),
        samples=sample_record_tuple,
        readout=readout_summary,
        query=query_summary,
        grouped_metrics=grouped_metrics,
        health=health,
        native_counterfactuals=native_records,
        native_counterfactual_summary=_summarize_counterfactuals(native_records),
        native_target_presence=target_presence_records,
        native_target_presence_summary=target_presence_summary,
    )


def _token_mean_nll_from_cell_score(
    summed_log_likelihood: torch.Tensor,
    valid_token_count: int,
) -> float:
    """Preserve the score tensor's reduction dtype before exporting FP32."""

    if (
        summed_log_likelihood.ndim != 0
        or not summed_log_likelihood.dtype.is_floating_point
    ):
        raise TypeError("Matrix-CE cell score must be one floating scalar")
    if isinstance(valid_token_count, bool) or not isinstance(valid_token_count, int):
        raise TypeError("valid evidence-token count must be an integer")
    if valid_token_count <= 0:
        raise ValueError("valid evidence-token count must be positive")
    return float((-summed_log_likelihood / valid_token_count).float().item())


def _mean_nll_matrix(scores: StreamingGroupScores) -> torch.Tensor:
    """Convert row-local summed log likelihoods to Golden-compatible mean NLL."""

    if scores.score_matrix.ndim != 2 or scores.score_matrix.shape[0] != (
        scores.evidence_token_counts.numel()
    ):
        raise ValueError("score matrix and evidence-token counts do not align")
    if bool((scores.evidence_token_counts <= 0).any().item()):
        raise ValueError("query matrix contains an empty evidence row")
    rows = tuple(
        (-scores.score_matrix[row_index] / int(token_count.item())).float().cpu()
        for row_index, token_count in enumerate(scores.evidence_token_counts)
    )
    return torch.stack(rows)


def save_representation_internal_evaluation_report_atomic(
    report: RepresentationInternalEvaluationReport,
    path: str | Path,
) -> RepresentationInternalEvaluationArtifact:
    """Create one immutable canonical JSON artifact without overwriting a file."""

    if not isinstance(report, RepresentationInternalEvaluationReport):
        raise TypeError("report must be RepresentationInternalEvaluationReport")
    return _publish_representation_internal_evaluation_report_atomic(report, path)


def _validated_sample_groups(
    sample_groups: Sequence[Sequence[RepresentationTrainingSample]],
) -> tuple[tuple[RepresentationTrainingSample, ...], ...]:
    if isinstance(sample_groups, (str, bytes)) or not isinstance(
        sample_groups, Sequence
    ):
        raise TypeError("sample_groups must be a nested sequence")
    materialized = tuple(tuple(group) for group in sample_groups)
    if len(materialized) < 2:
        raise ValueError(
            "wrong-different-image evaluation requires at least two groups"
        )
    all_ids: list[str] = []
    group_keys: list[str] = []
    for group in materialized:
        if len(group) < 2 or any(
            not isinstance(sample, RepresentationTrainingSample) for sample in group
        ):
            raise TypeError("every evaluation group must contain K>=2 typed samples")
        keys = {sample.image_group_key for sample in group}
        if len(keys) != 1:
            raise ValueError("one internal-evaluation group must share one image key")
        group_keys.append(next(iter(keys)))
        all_ids.extend(sample.sample_id for sample in group)
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("internal-evaluation image groups must be distinct")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("internal-evaluation sample IDs must be globally unique")
    return materialized


def _validated_native_cases(
    cases: Sequence[NativeCounterfactualCase],
) -> tuple[NativeCounterfactualCase, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("native_counterfactual_cases must be a sequence")
    materialized = tuple(cases)
    if not materialized or any(
        not isinstance(case, NativeCounterfactualCase) for case in materialized
    ):
        raise TypeError("at least one typed native counterfactual case is required")
    if len({case.case_id for case in materialized}) != len(materialized):
        raise ValueError("native counterfactual case IDs must be unique")
    return materialized


def _validated_target_presence_cases(
    cases: Sequence[NativeTargetPresenceCase],
) -> tuple[NativeTargetPresenceCase, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("native_target_presence_cases must be a sequence")
    materialized = tuple(cases)
    if any(not isinstance(case, NativeTargetPresenceCase) for case in materialized):
        raise TypeError("target-presence cases must be typed")
    if len({case.case_id for case in materialized}) != len(materialized):
        raise ValueError("target-presence case IDs must be unique")
    return materialized


def _validate_built_group(
    group: object,
    samples: tuple[RepresentationTrainingSample, ...],
    *,
    identity: RepresentationInternalEvaluationIdentity,
) -> None:
    if not isinstance(group, SameImageReadoutGroup):
        raise TypeError("group_builder must return SameImageReadoutGroup")
    expected_ids = tuple(sample.sample_id for sample in samples)
    if tuple(row.sample_id for row in group.rows) != expected_ids:
        raise ValueError("group_builder changed internal-evaluation sample order")
    if group.image_group_key != samples[0].image_group_key:
        raise ValueError("group_builder changed internal-evaluation image identity")
    if group.collective_padding:
        raise ValueError("internal evaluation must not materialize collective padding")
    for candidate in group.candidates:
        if candidate.target_conditioning_provider is not (
            identity.target_conditioning_provider
        ):
            raise ValueError(
                "group candidate provider differs from evaluation identity"
            )
        if candidate.attention is None:
            raise ValueError(
                "internal evaluation requires retained Adapter attention diagnostics"
            )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in (candidate.visual.main, *candidate.visual.deepstack)
        ):
            raise RuntimeError("internal-evaluation D retained an autograd graph")


def _validate_cross_group_contract(groups: tuple[SameImageReadoutGroup, ...]) -> None:
    first = groups[0]
    expected_projection = first.candidates[0].projection_identities
    expected_provider = first.candidates[0].target_conditioning_provider
    source_identities = tuple(group.source_visual_identity for group in groups)
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("internal-evaluation source visual IDs must be distinct")
    for group in groups:
        for candidate in group.candidates:
            if candidate.projection_identities != expected_projection:
                raise ValueError("internal-evaluation groups mix projection identities")
            if candidate.target_conditioning_provider is not expected_provider:
                raise ValueError("internal-evaluation groups mix providers")


def _find_wrong_different_candidate(
    groups: tuple[SameImageReadoutGroup, ...],
    *,
    group_index: int,
    row_index: int,
    expected: RepresentationVisualTensorBundle,
) -> tuple[SameImageReadoutGroup, RepresentationCandidateObservation] | None:
    """Select a deterministic compatible control without constraining the suite."""

    for offset in range(1, len(groups)):
        group = groups[(group_index + offset) % len(groups)]
        preferred = row_index % len(group.candidates)
        candidate_indices = (
            preferred,
            *(index for index in range(len(group.candidates)) if index != preferred),
        )
        for candidate_index in candidate_indices:
            candidate = group.candidates[candidate_index]
            if _visual_bundle_contract_matches(candidate.visual, expected):
                return group, candidate
    return None


def _score_readout_control(
    family_adapter: QwenVLMFamilyAdapter,
    model: nn.Module,
    *,
    source: RepresentationVisualTensorBundle,
    row: RepresentationReadoutRow,
    candidate: RepresentationVisualTensorBundle | None,
) -> tuple[float, int]:
    blocked_mask = build_original_image_key_block_mask(
        attention_mask=row.attention_mask,
        original_image_token_indices=torch.tensor(
            row.source_positions,
            dtype=torch.long,
            device=row.attention_mask.device,
        ),
        block_query_start=row.source_image_block_query_start,
        block_query_end=row.source_image_block_query_end,
        dtype=source.main.dtype,
    )
    blocks = [_injected_block("source_image", row.source_positions, source)]
    if candidate is not None:
        blocks.append(_injected_block("focused_d", row.d_positions, candidate))
    with torch.no_grad():
        result = family_adapter.forward_injected(
            model,
            InjectedForwardRequest(
                input_ids=row.input_ids,
                attention_mask=blocked_mask,
                position_ids=row.position_ids,
                visual_blocks=tuple(blocks),
                use_cache=False,
            ),
        )
        labels = torch.tensor(
            row.loss_labels,
            dtype=torch.long,
            device=result.logits.device,
        ).unsqueeze(0)
        losses = causal_evidence_losses(result.logits, labels)
    return (
        float(losses.per_sample_token_mean_nll[0].float().item()),
        int(losses.valid_token_counts[0].item()),
    )


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


def _deterministic_random_visual_bundle(
    visual: RepresentationVisualTensorBundle,
    *,
    seed: int,
) -> RepresentationVisualTensorBundle:
    hidden = int(visual.main.shape[-1])
    if hidden < 2:
        raise ValueError("deterministic random-D requires hidden size at least two")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(hidden, generator=generator, device="cpu")
    if torch.equal(permutation, torch.arange(hidden)):
        permutation = permutation.roll(1)
    permutation = permutation.to(device=visual.main.device)
    tensors = tuple(
        tensor.index_select(-1, permutation).detach()
        for tensor in (visual.main, *visual.deepstack)
    )
    return RepresentationVisualTensorBundle(
        main=tensors[0],
        deepstack=tensors[1:],
        branch_layers=visual.branch_layers,
        d_deepstack_active=visual.d_deepstack_active,
    )


def _random_control_seed(
    identity: RepresentationInternalEvaluationIdentity,
    sample_id: str,
) -> int:
    payload = (
        f"{DETERMINISTIC_RANDOM_D_ALGORITHM}\0{identity.evaluation_id}\0"
        f"{identity.random_seed}\0{sample_id}"
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_health(
    group: SameImageReadoutGroup,
    candidate: RepresentationCandidateObservation,
) -> RepresentationSampleHealthRecord:
    attention = candidate.attention
    if attention is None:  # guarded while validating the built group
        raise RuntimeError("candidate attention diagnostics disappeared")
    main_attention = attention_diagnostics(
        sub_slot_attention_weights=attention.main,
        topk=ATTENTION_TOPK,
    )
    if main_attention is None:
        raise RuntimeError("main Adapter attention is not evaluable")
    branches: list[RepresentationBranchHealthRecord] = []
    if candidate.visual.d_deepstack_active:
        for layer, d, source, weights in zip(
            candidate.visual.branch_layers,
            candidate.visual.deepstack,
            group.source_visual.deepstack,
            attention.deepstack,
            strict=True,
        ):
            branch_attention = attention_diagnostics(
                sub_slot_attention_weights=weights,
                topk=ATTENTION_TOPK,
            )
            if branch_attention is None:
                raise RuntimeError("D-DeepStack Adapter attention is not evaluable")
            branches.append(
                RepresentationBranchHealthRecord(
                    branch_layer=layer,
                    norm=norm_comparison_diagnostics(d, source),
                    attention=branch_attention,
                )
            )
    return RepresentationSampleHealthRecord(
        main=norm_comparison_diagnostics(
            candidate.visual.main, group.source_visual.main
        ),
        main_attention=main_attention,
        branches=tuple(branches),
    )


def _summarize_health(
    records: tuple[RepresentationInternalEvaluationSampleRecord, ...],
) -> RepresentationInternalHealthSummary:
    branch_layers = records[0].health.branches
    expected_layers = tuple(branch.branch_layer for branch in branch_layers)
    if any(
        tuple(branch.branch_layer for branch in record.health.branches)
        != expected_layers
        for record in records
    ):
        raise ValueError("sample health records mix DeepStack branch order")
    branches = tuple(
        RepresentationBranchHealthSummary(
            branch_layer=layer,
            representation=summarize_representation_health(
                tuple(record.health.branches[index].norm for record in records)
            ),
            attention=summarize_attention_diagnostics(
                tuple(record.health.branches[index].attention for record in records)
            ),
        )
        for index, layer in enumerate(expected_layers)
    )
    return RepresentationInternalHealthSummary(
        main=summarize_representation_health(
            tuple(record.health.main for record in records)
        ),
        main_attention=summarize_attention_diagnostics(
            tuple(record.health.main_attention for record in records)
        ),
        branches=branches,
    )


def _grouped_metrics(
    sample_groups: tuple[tuple[RepresentationTrainingSample, ...], ...],
    records: tuple[RepresentationInternalEvaluationSampleRecord, ...],
    nll_matrices: tuple[torch.Tensor, ...],
) -> tuple[RepresentationGroupedInternalMetrics, ...]:
    records_by_id = {record.sample_id: record for record in records}
    result: list[RepresentationGroupedInternalMetrics] = []
    for dimension in _GROUPING_DIMENSIONS:
        flat_labels = tuple(
            _metadata_label(getattr(records_by_id[sample.sample_id], dimension))
            for group in sample_groups
            for sample in group
        )
        nested_labels = tuple(
            tuple(
                _metadata_label(getattr(records_by_id[sample.sample_id], dimension))
                for sample in group
            )
            for group in sample_groups
        )
        readout_by_label = grouped_readout_metrics(
            tuple(record.nlls for record in records),
            flat_labels,
        )
        query_by_label = grouped_query_row_metrics(nll_matrices, nested_labels)
        if set(readout_by_label) != set(query_by_label):
            raise RuntimeError("grouped readout/query labels diverged")
        result.extend(
            RepresentationGroupedInternalMetrics(
                dimension=dimension,
                label=label,
                readout=readout_by_label[label],
                query=query_by_label[label],
            )
            for label in sorted(readout_by_label)
        )
    return tuple(result)


def _visual_bundle_contract_matches(
    actual: RepresentationVisualTensorBundle,
    expected: RepresentationVisualTensorBundle,
) -> bool:
    if actual.d_deepstack_active != expected.d_deepstack_active:
        return False
    if actual.branch_layers != expected.branch_layers:
        return False
    return all(
        actual_tensor.shape == expected_tensor.shape
        and actual_tensor.dtype == expected_tensor.dtype
        and actual_tensor.device == expected_tensor.device
        for actual_tensor, expected_tensor in zip(
            (actual.main, *actual.deepstack),
            (expected.main, *expected.deepstack),
            strict=True,
        )
    )


def _assert_adapter_state_unchanged(
    state: tuple[tuple[nn.Parameter, bool, torch.Tensor | None], ...],
) -> None:
    for parameter, requires_grad, gradient in state:
        if parameter.requires_grad != requires_grad:
            raise RuntimeError("internal evaluation changed Adapter trainability")
        if parameter.grad is not gradient:
            raise RuntimeError("internal evaluation replaced an Adapter gradient")


def _metadata_label(value: str | None) -> str:
    return _UNSPECIFIED_GROUP_LABEL if value is None else value


__all__ = [
    "ATTENTION_TOPK",
    "CAUSAL_VALUE_FLIP_LOG_ODDS_CONTRACT",
    "DETERMINISTIC_RANDOM_D_ALGORITHM",
    "READOUT_FORWARD_PATH",
    "READOUT_MASK_MODE",
    "READOUT_POSITION_SOURCE",
    "READOUT_VISUAL_SWAP_UNIT",
    "TARGET_PRESENCE_LOG_ODDS_CONTRACT",
    "REPRESENTATION_INTERNAL_EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "REPRESENTATION_INTERNAL_EVALUATION_SCHEMA_VERSION",
    "NativeCausalValueFlipEvaluator",
    "NativeCausalValueFlipOutput",
    "NativeCausalValueFlipRequest",
    "NativeBinaryLogOddsEvaluator",
    "NativeBinaryLogOddsOutput",
    "NativeBinaryLogOddsRequest",
    "NativeCounterfactualCase",
    "NativeCounterfactualEvaluationRecord",
    "NativeCounterfactualSummary",
    "NativeContinuationCacheParity",
    "NativeDOnlyContext",
    "NativeGenerationForward",
    "NativeInjectedRequestMaterializer",
    "NativeFreeContinuationEvaluator",
    "NativeFreeContinuationOutput",
    "NativeFreeContinuationRecord",
    "NativeFreeContinuationRequest",
    "NativeTeacherForcedForward",
    "NativeTargetPresenceCase",
    "NativeTargetPresenceContinuationRecord",
    "NativeTargetPresenceEvaluationRecord",
    "NativeTargetPresenceSummary",
    "InjectedNativeCounterfactualEvaluator",
    "RepresentationBranchHealthRecord",
    "RepresentationBranchHealthSummary",
    "RepresentationGroupedInternalMetrics",
    "RepresentationInternalEvaluationArtifact",
    "RepresentationInternalEvaluationGroupRecord",
    "RepresentationInternalEvaluationIdentity",
    "RepresentationInternalEvaluationReport",
    "RepresentationInternalEvaluationSampleRecord",
    "RepresentationInternalHealthSummary",
    "RepresentationReadoutControlIdentity",
    "RepresentationReadoutExecutionContract",
    "RepresentationSampleHealthRecord",
    "run_representation_internal_evaluation",
    "create_injected_native_counterfactual_evaluator",
    "save_representation_internal_evaluation_report_atomic",
]
