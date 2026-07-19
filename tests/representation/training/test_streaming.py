from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tgvf_rl.representation.training.streaming as streaming_module
from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.losses import (
    EVIDENCE_IGNORE_INDEX,
    EvidenceReadabilityLossTerms,
    HistoricalNormLossTerms,
    MatrixCEScoreMode,
    SameImageMatrixCELossTerms,
    historical_norm_loss_terms,
    historical_sample_norm_loss,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
    compose_reference_representation_objective,
)
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    synthetic_same_image_layout_readout_terms,
)
from tgvf_rl.representation.training.streaming import (
    StreamingGlobalNormalization,
    backward_streaming_same_image_groups,
    backward_streaming_same_image_group,
    score_streaming_same_image_groups,
    score_streaming_same_image_group,
)
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


class _RecordingLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.attention_masks: list[torch.Tensor] = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        visual_pos_masks,
        deepstack_visual_embeds,
        **kwargs,
    ):
        self.attention_masks.append(attention_mask.detach().cpu())
        hidden = inputs_embeds.clone()
        for branch in deepstack_visual_embeds:
            hidden = hidden.clone()
            hidden[visual_pos_masks] += branch
        if attention_mask.ndim == 4:
            minimum = torch.finfo(attention_mask.dtype).min
            valid_keys = (attention_mask > minimum).any(dim=-2).squeeze(1)
        else:
            valid_keys = attention_mask.bool()
        hidden = (
            hidden + (hidden * valid_keys.unsqueeze(-1)).sum(dim=1, keepdim=True) * 0.1
        )
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyQwenContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _RecordingLanguageModel()


class _TinyFrozenQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyQwenContainer()
        self.lm_head = nn.Linear(4, 16, bias=False)


def _frozen_model() -> _TinyFrozenQwen:
    model = _TinyFrozenQwen().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _supervision(
    token_ids: tuple[int, ...],
    *,
    evidence_positions: tuple[int, ...] = (6, 7),
) -> ModelEvidenceSupervision:
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token if index in evidence_positions else EVIDENCE_IGNORE_INDEX
            for index, token in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=(
            (0,),
            (1, 2),
            (3, 4),
            *((position,) for position in range(5, len(token_ids))),
        ),
    )


def _bundle(value: float, *, requires_grad: bool) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 4), value, requires_grad=requires_grad),
        deepstack=tuple(
            torch.full(
                (1, 2, 4),
                value * (index + 1) / 10,
                requires_grad=requires_grad,
            )
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


def _group(
    *,
    padding_count: int = 0,
    group_id: str = "image-1",
    size: int = 2,
    candidate_offset: float = 0.0,
) -> SameImageReadoutGroup:
    if size < 2 or size > 4:
        raise ValueError("tiny streaming fixture supports 2 <= K <= 4")
    rows = []
    candidates = []
    for index in range(size):
        evidence = (5 + index * 2, 6 + index * 2)
        sample_id = f"{group_id}-sample-{index}"
        token_ids = (1, 2, 2, 2, 2, 3, *evidence)
        rows.append(
            RepresentationReadoutRow(
                sample_id=sample_id,
                image_group_key=group_id,
                source_visual_identity=f"source-{group_id}",
                supervision=_supervision(token_ids),
                input_ids=torch.tensor([token_ids], dtype=torch.long),
                attention_mask=torch.ones(1, 8, dtype=torch.bool),
                position_ids=torch.arange(8).view(1, 8),
                source_positions=(1, 2),
                d_positions=(3, 4),
            )
        )
        candidates.append(
            RepresentationCandidateObservation(
                sample_id=sample_id,
                image_group_key=group_id,
                source_visual_identity=f"source-{group_id}",
                target_conditioning_provider=(
                    TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                ),
                projection_identities=("main", "branch-8", "branch-16", "branch-24"),
                visual=_bundle(
                    candidate_offset + float(index + 1),
                    requires_grad=True,
                ),
            )
        )
    return SameImageReadoutGroup(
        image_group_key=group_id,
        source_visual_identity=f"source-{group_id}",
        source_visual=_bundle(
            0.25 + candidate_offset / 10,
            requires_grad=False,
        ),
        rows=tuple(rows),
        candidates=tuple(candidates),
        collective_padding=tuple(
            _bundle(10.0 + index, requires_grad=True) for index in range(padding_count)
        ),
    )


def _candidate_tensors(group: SameImageReadoutGroup) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor
        for candidate in group.candidates
        for tensor in (candidate.visual.main, *candidate.visual.deepstack)
    )


def _mixed_length_group(*, group_id: str) -> SameImageReadoutGroup:
    group = _group(group_id=group_id)
    long_row = group.rows[1]
    token_ids = (*long_row.supervision.model_token_ids, 4)
    long_row = replace(
        long_row,
        supervision=_supervision(token_ids),
        input_ids=torch.tensor([token_ids], dtype=torch.long),
        attention_mask=torch.ones(1, len(token_ids), dtype=torch.bool),
        position_ids=torch.arange(len(token_ids)).view(1, len(token_ids)),
    )
    return replace(group, rows=(group.rows[0], long_row))


def _different_evidence_length_group(*, group_id: str) -> SameImageReadoutGroup:
    group = _group(group_id=group_id)
    short_row, long_row = group.rows
    short_row = replace(
        short_row,
        supervision=_supervision(
            short_row.supervision.model_token_ids,
            evidence_positions=(7,),
        ),
    )
    long_token_ids = (*long_row.supervision.model_token_ids, 4)
    long_row = replace(
        long_row,
        supervision=_supervision(
            long_token_ids,
            evidence_positions=(6, 7, 8),
        ),
        input_ids=torch.tensor([long_token_ids], dtype=torch.long),
        attention_mask=torch.ones(1, len(long_token_ids), dtype=torch.bool),
        position_ids=torch.arange(len(long_token_ids)).view(1, len(long_token_ids)),
    )
    return replace(group, rows=(short_row, long_row))


def _objective() -> RepresentationObjectiveConfig:
    return RepresentationObjectiveConfig(
        identity="test-matrix-and-readable",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=0.7,
        l_gen_weight=1.3,
    )


def _objective_v2() -> RepresentationObjectiveConfigV2:
    return RepresentationObjectiveConfigV2(
        identity="test-direct-multi-group-historical-norm",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=0.7,
        l_gen_weight=1.3,
        norm_weight=0.1,
    )


def _objective_v3(
    *,
    mode: MatrixCEScoreMode = MatrixCEScoreMode.BALANCED,
    temperature: float = 0.5,
) -> RepresentationObjectiveConfigV3:
    return RepresentationObjectiveConfigV3(
        identity=f"test-{mode.value}-matrix-ce",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=0.7,
        l_gen_weight=1.3,
        norm_weight=0.1,
        matrix_ce_mode=mode,
        matrix_ce_temperature=temperature,
    )


def test_direct_four_k4_groups_keep_blockwise_ce_and_batch_qwen_cells() -> None:
    torch.manual_seed(41)
    model = _frozen_model()
    groups = tuple(
        _group(
            group_id=f"direct-image-{index}",
            size=4,
            candidate_offset=float(index) * 0.4,
        )
        for index in range(4)
    )

    objective = _objective_v2()
    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=16,
        l_gen_samples=16,
    )
    scores = score_streaming_same_image_groups(
        Qwen3VLAdapter(),
        model,
        groups,
        objective=objective,
        normalization=normalization,
    )

    assert len(scores.group_scores) == 4
    assert all(score.score_matrix.shape == (4, 4) for score in scores.group_scores)
    assert scores.qwen_forward_batch_sizes == (32, 32)
    assert len(model.model.language_model.attention_masks) == 2
    assert all(
        mask.shape == (32, 1, 8, 8)
        for mask in model.model.language_model.attention_masks
    )

    expected_matrix_numerator = torch.stack(
        tuple(
            F.cross_entropy(
                score.score_matrix,
                torch.arange(4),
                reduction="sum",
            )
            for score in scores.group_scores
        )
    ).sum()
    expected_l_gen_numerator = torch.stack(
        tuple(score.diagonal_l_gen.sum() for score in scores.group_scores)
    ).sum()
    expected_norm_numerator = torch.stack(
        tuple(score.historical_norm.sum() for score in scores.group_scores)
    ).sum()
    expected_total = (
        expected_matrix_numerator / 16 * objective.matrix_ce_weight
        + expected_l_gen_numerator / 16 * objective.l_gen_weight
        + expected_norm_numerator / 16 * objective.norm_weight
    )
    incorrect_cross_group_matrix = torch.block_diag(
        *(score.score_matrix for score in scores.group_scores)
    )
    incorrect_cross_group_ce = F.cross_entropy(
        incorrect_cross_group_matrix,
        torch.arange(16),
    )
    assert incorrect_cross_group_matrix.shape == (16, 16)
    assert not torch.allclose(
        expected_matrix_numerator / 16,
        incorrect_cross_group_ce,
    )

    metrics = backward_streaming_same_image_groups(
        Qwen3VLAdapter(),
        model,
        groups,
        scores,
        objective=objective,
        normalization=normalization,
    )

    assert metrics.local_row_count == metrics.local_sample_count == 16
    assert torch.equal(metrics.matrix_ce_numerator, expected_matrix_numerator)
    assert torch.equal(metrics.l_gen_numerator, expected_l_gen_numerator)
    assert metrics.norm_numerator is not None
    assert torch.equal(metrics.norm_numerator, expected_norm_numerator)
    assert torch.allclose(metrics.weighted_local_mean, expected_total)
    assert metrics.qwen_forward_batch_sizes == (32, 32)
    assert len(model.model.language_model.attention_masks) == 2
    assert all(
        tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().item())
        for group in groups
        for tensor in _candidate_tensors(group)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_direct_two_group_backward_matches_combined_blockwise_reference() -> None:
    torch.manual_seed(47)
    full_model = _frozen_model()
    streaming_model = _frozen_model()
    streaming_model.load_state_dict(full_model.state_dict())
    full_groups = (
        _group(group_id="reference-a", candidate_offset=0.0),
        _group(group_id="reference-b", candidate_offset=0.7),
    )
    streaming_groups = (
        _group(group_id="reference-a", candidate_offset=0.0),
        _group(group_id="reference-b", candidate_offset=0.7),
    )
    objective = _objective_v2()
    full_terms = tuple(
        synthetic_same_image_layout_readout_terms(
            Qwen3VLAdapter(),
            full_model,
            group,
        )
        for group in full_groups
    )
    full_norm_terms = tuple(
        historical_norm_loss_terms(
            tuple(
                historical_sample_norm_loss(
                    candidate.visual.main,
                    group.source_visual.main,
                    candidate.visual.deepstack,
                    group.source_visual.deepstack,
                )
                for candidate in group.candidates
            )
        )
        for group in full_groups
    )
    reference_value = compose_reference_representation_objective(
        SameImageMatrixCELossTerms(
            numerator=torch.stack(
                tuple(terms.matrix_ce.numerator for terms in full_terms)
            ).sum(),
            valid_row_count=4,
        ),
        EvidenceReadabilityLossTerms(
            numerator=torch.stack(
                tuple(terms.l_gen.numerator for terms in full_terms)
            ).sum(),
            sample_count=4,
        ),
        objective,
        HistoricalNormLossTerms(
            numerator=torch.stack(
                tuple(terms.numerator for terms in full_norm_terms)
            ).sum(),
            sample_count=4,
        ),
    )
    expected_gradients = torch.autograd.grad(
        reference_value.total_loss,
        tuple(tensor for group in full_groups for tensor in _candidate_tensors(group)),
    )

    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=4,
        l_gen_samples=4,
    )
    scores = score_streaming_same_image_groups(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_groups,
        objective=objective,
        normalization=normalization,
    )
    score_forward_count = len(streaming_model.model.language_model.attention_masks)
    metrics = backward_streaming_same_image_groups(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_groups,
        scores,
        objective=objective,
        normalization=normalization,
    )

    assert scores.qwen_forward_batch_sizes == (8,)
    assert metrics.qwen_forward_batch_sizes == (8,)
    assert score_forward_count == 1
    assert len(streaming_model.model.language_model.attention_masks) == 1
    assert all(score.score_matrix.shape == (2, 2) for score in scores.group_scores)
    assert torch.allclose(
        metrics.weighted_local_mean,
        reference_value.total_loss.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    for actual, expected in zip(
        (
            tensor.grad
            for group in streaming_groups
            for tensor in _candidate_tensors(group)
        ),
        expected_gradients,
        strict=True,
    ):
        assert actual is not None
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_balanced_streaming_gradients_match_full_autograd_for_unequal_lengths() -> None:
    torch.manual_seed(51)
    legacy_model = _frozen_model()
    full_model = _frozen_model()
    streaming_model = _frozen_model()
    full_model.load_state_dict(legacy_model.state_dict())
    streaming_model.load_state_dict(legacy_model.state_dict())
    legacy_group = _different_evidence_length_group(group_id="balanced-reference")
    full_group = _different_evidence_length_group(group_id="balanced-reference")
    streaming_group = _different_evidence_length_group(group_id="balanced-reference")
    objective = _objective_v3(temperature=0.5)

    legacy_terms = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(),
        legacy_model,
        legacy_group,
    )
    full_terms = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(),
        full_model,
        full_group,
        matrix_ce_mode=objective.matrix_ce_mode,
        matrix_ce_temperature=objective.matrix_ce_temperature,
    )
    expected_balanced_scores = (
        legacy_terms.score_matrix
        / legacy_terms.evidence_token_counts.unsqueeze(1)
        / objective.matrix_ce_temperature
    )
    assert torch.equal(full_terms.evidence_token_counts, torch.tensor([1, 3]))
    assert torch.allclose(
        full_terms.score_matrix,
        expected_balanced_scores,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        full_terms.l_gen.numerator,
        legacy_terms.l_gen.numerator,
        atol=1e-6,
        rtol=1e-6,
    )

    norm_terms = historical_norm_loss_terms(
        tuple(
            historical_sample_norm_loss(
                candidate.visual.main,
                full_group.source_visual.main,
                candidate.visual.deepstack,
                full_group.source_visual.deepstack,
            )
            for candidate in full_group.candidates
        )
    )
    reference_value = compose_reference_representation_objective(
        full_terms.matrix_ce,
        full_terms.l_gen,
        objective,
        norm_terms,
    )
    expected_gradients = torch.autograd.grad(
        reference_value.total_loss,
        _candidate_tensors(full_group),
    )
    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=2,
        l_gen_samples=2,
    )
    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        objective=objective,
        normalization=normalization,
    )
    with pytest.raises(ValueError, match="differs from the materialized Qwen VJP"):
        backward_streaming_same_image_group(
            Qwen3VLAdapter(),
            streaming_model,
            streaming_group,
            scores,
            objective=_objective_v3(temperature=1.0),
            normalization=normalization,
        )
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=normalization,
    )

    assert torch.allclose(
        scores.score_matrix,
        full_terms.score_matrix.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        metrics.matrix_ce_numerator,
        full_terms.matrix_ce.numerator.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        metrics.weighted_local_mean,
        reference_value.total_loss.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    for actual, expected in zip(
        _candidate_tensors(streaming_group),
        expected_gradients,
        strict=True,
    ):
        assert actual.grad is not None
        assert torch.allclose(actual.grad, expected, atol=1e-6, rtol=1e-6)


def test_explicit_legacy_streaming_mode_is_bitwise_historical() -> None:
    torch.manual_seed(52)
    historical_model = _frozen_model()
    explicit_model = _frozen_model()
    explicit_model.load_state_dict(historical_model.state_dict())
    historical = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        historical_model,
        _different_evidence_length_group(group_id="legacy-reference"),
    )
    explicit = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        explicit_model,
        _different_evidence_length_group(group_id="legacy-reference"),
        objective=_objective_v3(
            mode=MatrixCEScoreMode.LEGACY_SUMMED_NLL,
            temperature=1.0,
        ),
    )

    assert torch.equal(explicit.score_matrix, historical.score_matrix)
    assert torch.equal(explicit.diagonal_l_gen, historical.diagonal_l_gen)
    assert torch.equal(
        explicit.evidence_token_counts,
        historical.evidence_token_counts,
    )


def test_right_padded_mixed_length_rows_match_unpadded_reference() -> None:
    torch.manual_seed(53)
    full_model = _frozen_model()
    streaming_model = _frozen_model()
    streaming_model.load_state_dict(full_model.state_dict())
    full_group = _mixed_length_group(group_id="mixed-reference")
    streaming_group = _mixed_length_group(group_id="mixed-reference")
    objective = _objective_v2()

    full_terms = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(),
        full_model,
        full_group,
    )
    norm_terms = historical_norm_loss_terms(
        tuple(
            historical_sample_norm_loss(
                candidate.visual.main,
                full_group.source_visual.main,
                candidate.visual.deepstack,
                full_group.source_visual.deepstack,
            )
            for candidate in full_group.candidates
        )
    )
    reference_value = compose_reference_representation_objective(
        full_terms.matrix_ce,
        full_terms.l_gen,
        objective,
        norm_terms,
    )
    expected_gradients = torch.autograd.grad(
        reference_value.total_loss,
        _candidate_tensors(full_group),
    )
    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=2,
        l_gen_samples=2,
    )

    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        objective=objective,
        normalization=normalization,
    )
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=normalization,
    )

    assert metrics.qwen_forward_batch_sizes == (4,)
    assert len(streaming_model.model.language_model.attention_masks) == 1
    padded_mask = streaming_model.model.language_model.attention_masks[0]
    assert padded_mask.shape == (4, 1, 9, 9)
    short_mask_model = _frozen_model()
    score_streaming_same_image_group(
        Qwen3VLAdapter(),
        short_mask_model,
        _group(group_id="short-mask-reference"),
    )
    assert torch.equal(
        padded_mask[0, :, :8, :8],
        short_mask_model.model.language_model.attention_masks[0][0],
    )
    minimum = torch.finfo(padded_mask.dtype).min
    assert bool((padded_mask[:2, :, :, 8] == minimum).all().item())
    assert torch.allclose(
        scores.score_matrix,
        full_terms.score_matrix.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        metrics.weighted_local_mean,
        reference_value.total_loss.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    for actual, expected in zip(
        (tensor.grad for tensor in _candidate_tensors(streaming_group)),
        expected_gradients,
        strict=True,
    ):
        assert actual is not None
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_streaming_backward_matches_full_graph_reference() -> None:
    torch.manual_seed(19)
    full_model = _frozen_model()
    streaming_model = _frozen_model()
    streaming_model.load_state_dict(full_model.state_dict())
    full_group = _group()
    streaming_group = _group()
    objective = _objective()

    full_terms = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(), full_model, full_group
    )
    full_value = compose_reference_representation_objective(
        full_terms.matrix_ce, full_terms.l_gen, objective
    )
    expected_gradients = torch.autograd.grad(
        full_value.total_loss, _candidate_tensors(full_group)
    )

    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=2,
        l_gen_samples=2,
    )
    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        objective=objective,
        normalization=normalization,
    )
    score_forward_count = len(streaming_model.model.language_model.attention_masks)
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=normalization,
    )
    actual_gradients = tuple(
        tensor.grad for tensor in _candidate_tensors(streaming_group)
    )

    assert metrics.local_row_count == 2
    assert metrics.local_sample_count == 2
    assert metrics.qwen_forward_batch_sizes == (4,)
    assert score_forward_count == 1
    assert len(streaming_model.model.language_model.attention_masks) == 1
    assert torch.allclose(metrics.weighted_local_mean, full_value.total_loss.detach())
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert actual is not None
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert all(parameter.grad is None for parameter in streaming_model.parameters())


def test_streaming_v2_norm_matches_full_graph_and_reports_raw_weighted_values() -> None:
    torch.manual_seed(29)
    full_model = _frozen_model()
    streaming_model = _frozen_model()
    streaming_model.load_state_dict(full_model.state_dict())
    full_group = _group()
    streaming_group = _group()
    objective = RepresentationObjectiveConfigV2(
        identity="test-matrix-readable-historical-norm",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=0.7,
        l_gen_weight=1.3,
        norm_weight=0.1,
    )

    full_terms = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(), full_model, full_group
    )
    norm_terms = historical_norm_loss_terms(
        tuple(
            historical_sample_norm_loss(
                candidate.visual.main,
                full_group.source_visual.main,
                candidate.visual.deepstack,
                full_group.source_visual.deepstack,
            )
            for candidate in full_group.candidates
        )
    )
    full_value = compose_reference_representation_objective(
        full_terms.matrix_ce,
        full_terms.l_gen,
        objective,
        norm_terms,
    )
    # This local K=2 group is one half of a K=4 global accumulation window.
    # Matrix, readability, and norm all use their global sample denominator.
    expected_gradients = torch.autograd.grad(
        full_value.total_loss / 2, _candidate_tensors(full_group)
    )

    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=4,
        l_gen_samples=4,
    )
    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        objective=objective,
        normalization=normalization,
    )
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=normalization,
    )

    assert metrics.norm_numerator is not None
    assert metrics.weighted_norm_local_mean is not None
    assert torch.equal(metrics.norm_numerator, norm_terms.numerator.detach())
    assert torch.equal(
        metrics.weighted_norm_local_mean,
        norm_terms.mean.detach() * objective.norm_weight,
    )
    assert torch.allclose(metrics.weighted_local_mean, full_value.total_loss.detach())
    for actual, expected in zip(
        (tensor.grad for tensor in _candidate_tensors(streaming_group)),
        expected_gradients,
        strict=True,
    ):
        assert actual is not None
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_streaming_norm_finite_check_has_one_normal_path_host_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _group()
    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(tensor: torch.Tensor, *args: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    streaming_module._validate_streaming_norm_tensors_finite((group,))

    assert item_calls == 1


def test_streaming_norm_determinism_comparison_is_fused_before_vjps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _group()
    model = _frozen_model()
    objective = _objective_v2()
    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=2,
        l_gen_samples=2,
    )
    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        model,
        group,
        objective=objective,
        normalization=normalization,
    )
    changed_norm = scores.historical_norm.clone()
    changed_norm[1] = torch.nextafter(
        changed_norm[1],
        torch.full_like(changed_norm[1], float("inf")),
    )
    changed_scores = replace(scores, historical_norm=changed_norm)
    original_equal = torch.equal
    equal_calls: list[tuple[torch.Size, torch.Size]] = []

    def counted_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
        equal_calls.append((first.shape, second.shape))
        return original_equal(first, second)

    monkeypatch.setattr(torch, "equal", counted_equal)
    with pytest.raises(RuntimeError, match="changed a norm value"):
        backward_streaming_same_image_group(
            Qwen3VLAdapter(),
            model,
            group,
            changed_scores,
            objective=objective,
            normalization=normalization,
        )

    assert equal_calls == [(torch.Size([2]), torch.Size([2]))]
    assert all(tensor.grad is None for tensor in _candidate_tensors(group))


def test_collective_padding_has_zero_gradient_and_does_not_change_real_objective() -> (
    None
):
    torch.manual_seed(23)
    unpadded_model = _frozen_model()
    padded_model = _frozen_model()
    padded_model.load_state_dict(unpadded_model.state_dict())
    unpadded_group = _group()
    padded_group = _group(padding_count=1)
    objective = _objective()
    normalization = StreamingGlobalNormalization(
        matrix_valid_rows=2,
        l_gen_samples=2,
    )

    unpadded_scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        unpadded_model,
        unpadded_group,
        objective=objective,
        normalization=normalization,
    )
    unpadded_metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        unpadded_model,
        unpadded_group,
        unpadded_scores,
        objective=objective,
        normalization=normalization,
    )
    padded_scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        padded_model,
        padded_group,
        objective=objective,
        normalization=normalization,
    )
    padded_metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        padded_model,
        padded_group,
        padded_scores,
        objective=objective,
        normalization=normalization,
    )

    assert padded_group.collective_candidate_count == 3
    assert torch.equal(padded_scores.score_matrix, unpadded_scores.score_matrix)
    assert torch.equal(padded_scores.diagonal_l_gen, unpadded_scores.diagonal_l_gen)
    assert torch.equal(
        padded_metrics.matrix_ce_numerator,
        unpadded_metrics.matrix_ce_numerator,
    )
    assert torch.equal(
        padded_metrics.l_gen_numerator,
        unpadded_metrics.l_gen_numerator,
    )
    assert padded_metrics.local_row_count == padded_metrics.local_sample_count == 2
    for padded_tensor, unpadded_tensor in zip(
        _candidate_tensors(padded_group),
        _candidate_tensors(unpadded_group),
        strict=True,
    ):
        assert padded_tensor.grad is not None
        assert unpadded_tensor.grad is not None
        assert torch.allclose(padded_tensor.grad, unpadded_tensor.grad)
    for padding in padded_group.collective_padding:
        for tensor in (padding.main, *padding.deepstack):
            assert tensor.grad is not None
            assert torch.count_nonzero(tensor.grad).item() == 0


def test_streaming_readout_blocks_source_keys_for_causal_evidence_queries() -> None:
    torch.manual_seed(5)
    model = _frozen_model()

    score_streaming_same_image_group(Qwen3VLAdapter(), model, _group())

    mask = model.model.language_model.attention_masks[0]
    assert mask.shape == (4, 1, 8, 8)
    minimum = torch.finfo(mask.dtype).min
    assert mask[0, 0, 5, 1].item() == minimum
    assert mask[0, 0, 6, 2].item() == minimum
    assert mask[0, 0, 4, 1].item() == 0.0


def test_streaming_normalization_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        StreamingGlobalNormalization(matrix_valid_rows=0, l_gen_samples=2)

    group = _group()
    model = _frozen_model()
    scores = score_streaming_same_image_group(Qwen3VLAdapter(), model, group)
    with pytest.raises(ValueError, match="cannot be smaller"):
        backward_streaming_same_image_group(
            Qwen3VLAdapter(),
            model,
            group,
            scores,
            objective=_objective(),
            normalization=StreamingGlobalNormalization(
                matrix_valid_rows=1,
                l_gen_samples=2,
            ),
        )
