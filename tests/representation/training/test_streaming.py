from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.losses import (
    EVIDENCE_IGNORE_INDEX,
    historical_norm_loss_terms,
    historical_sample_norm_loss,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
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
    backward_streaming_same_image_group,
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
        hidden = hidden + hidden.sum(dim=1, keepdim=True) * 0.1
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


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token if index in evidence_positions else EVIDENCE_IGNORE_INDEX
            for index, token in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
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


def _group(*, padding_count: int = 0) -> SameImageReadoutGroup:
    rows = []
    candidates = []
    for index, evidence in enumerate(((5, 6), (7, 8))):
        sample_id = f"sample-{index}"
        token_ids = (1, 2, 2, 2, 2, 3, *evidence)
        rows.append(
            RepresentationReadoutRow(
                sample_id=sample_id,
                image_group_key="image-1",
                source_visual_identity="source-sha",
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
                image_group_key="image-1",
                source_visual_identity="source-sha",
                target_conditioning_provider=(
                    TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                ),
                projection_identities=("main", "branch-8", "branch-16", "branch-24"),
                visual=_bundle(float(index + 1), requires_grad=True),
            )
        )
    return SameImageReadoutGroup(
        image_group_key="image-1",
        source_visual_identity="source-sha",
        source_visual=_bundle(0.25, requires_grad=False),
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


def _objective() -> RepresentationObjectiveConfig:
    return RepresentationObjectiveConfig(
        identity="test-matrix-and-readable",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=0.7,
        l_gen_weight=1.3,
    )


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

    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(), streaming_model, streaming_group
    )
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=StreamingGlobalNormalization(
            matrix_valid_rows=2,
            l_gen_samples=2,
        ),
    )
    actual_gradients = tuple(
        tensor.grad for tensor in _candidate_tensors(streaming_group)
    )

    assert metrics.local_row_count == 2
    assert metrics.local_sample_count == 2
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

    scores = score_streaming_same_image_group(
        Qwen3VLAdapter(), streaming_model, streaming_group
    )
    metrics = backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        streaming_model,
        streaming_group,
        scores,
        objective=objective,
        normalization=StreamingGlobalNormalization(
            matrix_valid_rows=4,
            l_gen_samples=4,
        ),
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
        Qwen3VLAdapter(), unpadded_model, unpadded_group
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
        Qwen3VLAdapter(), padded_model, padded_group
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
    assert mask.shape == (1, 1, 8, 8)
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
