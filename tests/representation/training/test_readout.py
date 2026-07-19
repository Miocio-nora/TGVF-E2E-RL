from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
    synthetic_same_image_layout_readout_terms,
)
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


class _ContextualTinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds,
        visual_pos_masks,
        deepstack_visual_embeds,
        **kwargs,
    ):
        hidden = inputs_embeds.clone()
        for branch in deepstack_visual_embeds:
            hidden = hidden.clone()
            hidden[visual_pos_masks] += branch
        hidden = hidden + hidden.sum(dim=1, keepdim=True) * 0.1
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyQwenContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _ContextualTinyLanguageModel()


class _TinyFrozenQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyQwenContainer()
        self.lm_head = nn.Linear(4, 16, bias=False)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    labels = tuple(
        token_id if position in evidence_positions else EVIDENCE_IGNORE_INDEX
        for position, token_id in enumerate(token_ids)
    )
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=labels,
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


def _bundle(
    value: float, *, requires_grad: bool = False
) -> RepresentationVisualTensorBundle:
    main = torch.full((1, 2, 4), value, requires_grad=requires_grad)
    branches = tuple(
        torch.full(
            (1, 2, 4),
            value * (index + 1) / 10,
            requires_grad=requires_grad,
        )
        for index in range(3)
    )
    return RepresentationVisualTensorBundle(main, branches, (8, 16, 24))


def _group(
    provider: TargetConditioningProviderKind = (
        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    ),
) -> SameImageReadoutGroup:
    rows = []
    candidates = []
    for index, evidence_ids in enumerate(((5, 6), (7, 8))):
        sample_id = f"sample-{index}"
        token_ids = (1, 2, 2, 2, 2, 3, *evidence_ids)
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
                target_conditioning_provider=provider,
                projection_identities=("main", "branch-8", "branch-16", "branch-24"),
                visual=_bundle(float(index + 1), requires_grad=True),
            )
        )
    return SameImageReadoutGroup(
        image_group_key="image-1",
        source_visual_identity="source-sha",
        source_visual=_bundle(0.25),
        rows=tuple(rows),
        candidates=tuple(candidates),
    )


@pytest.mark.parametrize("provider", tuple(TargetConditioningProviderKind))
def test_synthetic_readout_builds_full_matrix_and_both_loss_terms(
    provider: TargetConditioningProviderKind,
) -> None:
    torch.manual_seed(4)
    model = _freeze(_TinyFrozenQwen())

    result = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(), model, _group(provider)
    )

    assert result.sample_ids == ("sample-0", "sample-1")
    assert result.score_matrix.shape == (2, 2)
    assert result.matrix_ce.valid_row_count == 2
    assert result.l_gen.sample_count == 2
    assert torch.equal(result.evidence_token_counts, torch.tensor([2, 2]))
    assert not torch.equal(result.score_matrix[:, 0], result.score_matrix[:, 1])
    assert result.score_matrix.requires_grad
    assert result.matrix_ce.numerator.requires_grad
    assert result.l_gen.numerator.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result.matrix_ce.numerator.ndim == 0
    assert result.l_gen.numerator.ndim == 0


def test_synthetic_readout_requires_a_frozen_eval_qwen() -> None:
    model = _TinyFrozenQwen()
    with pytest.raises(ValueError, match="eval mode"):
        synthetic_same_image_layout_readout_terms(Qwen3VLAdapter(), model, _group())

    model.eval()
    with pytest.raises(ValueError, match="disable gradients"):
        synthetic_same_image_layout_readout_terms(Qwen3VLAdapter(), model, _group())


def test_synthetic_layout_gradients_reach_candidate_main_and_every_branch() -> None:
    torch.manual_seed(12)
    model = _freeze(_TinyFrozenQwen())
    group = _group()
    result = synthetic_same_image_layout_readout_terms(Qwen3VLAdapter(), model, group)
    candidate_tensors = tuple(
        tensor
        for candidate in group.candidates
        for tensor in (candidate.visual.main, *candidate.visual.deepstack)
    )

    gradients = torch.autograd.grad(
        result.matrix_ce.numerator + result.l_gen.numerator,
        candidate_tensors,
        allow_unused=False,
    )

    assert len(gradients) == 2 * (1 + 3)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("changed_component", ("main", 0, 1, 2))
def test_each_atomic_candidate_main_and_branch_reaches_matrix_scores(
    changed_component: str | int,
) -> None:
    torch.manual_seed(9)
    baseline = _group()
    zero_main = torch.zeros(1, 2, 4)
    zero_branches = tuple(torch.zeros(1, 2, 4) for _ in range(3))
    changed_main = zero_main.clone()
    changed_branches = list(branch.clone() for branch in zero_branches)
    if changed_component == "main":
        changed_main.fill_(1.0)
    else:
        changed_branches[changed_component].fill_(1.0)  # type: ignore[index]

    candidate_zero = RepresentationCandidateObservation(
        sample_id="sample-0",
        image_group_key="image-1",
        source_visual_identity="source-sha",
        target_conditioning_provider=(
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ),
        projection_identities=("main", "branch-8", "branch-16", "branch-24"),
        visual=RepresentationVisualTensorBundle(zero_main, zero_branches, (8, 16, 24)),
    )
    candidate_changed = RepresentationCandidateObservation(
        sample_id="sample-1",
        image_group_key="image-1",
        source_visual_identity="source-sha",
        target_conditioning_provider=(
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ),
        projection_identities=("main", "branch-8", "branch-16", "branch-24"),
        visual=RepresentationVisualTensorBundle(
            changed_main, tuple(changed_branches), (8, 16, 24)
        ),
    )
    group = SameImageReadoutGroup(
        image_group_key=baseline.image_group_key,
        source_visual_identity=baseline.source_visual_identity,
        source_visual=baseline.source_visual,
        rows=baseline.rows,
        candidates=(candidate_zero, candidate_changed),
    )

    result = synthetic_same_image_layout_readout_terms(
        Qwen3VLAdapter(), _freeze(_TinyFrozenQwen()), group
    )

    assert not torch.equal(result.score_matrix[:, 0], result.score_matrix[:, 1])


def test_group_rejects_row_candidate_diagonal_or_branch_contract_drift() -> None:
    group = _group()
    with pytest.raises(ValueError, match="diagonal identity"):
        SameImageReadoutGroup(
            image_group_key=group.image_group_key,
            source_visual_identity=group.source_visual_identity,
            source_visual=group.source_visual,
            rows=group.rows,
            candidates=tuple(reversed(group.candidates)),
        )

    drifted = RepresentationCandidateObservation(
        sample_id="sample-1",
        image_group_key="image-1",
        source_visual_identity="source-sha",
        target_conditioning_provider=(
            TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ),
        projection_identities=("main", "branch-8", "branch-16", "branch-24"),
        visual=RepresentationVisualTensorBundle(
            torch.ones(1, 2, 4),
            tuple(torch.ones(1, 2, 4) for _ in range(3)),
            (8, 16, 25),
        ),
    )
    with pytest.raises(ValueError, match="layer order"):
        SameImageReadoutGroup(
            image_group_key=group.image_group_key,
            source_visual_identity=group.source_visual_identity,
            source_visual=group.source_visual,
            rows=group.rows,
            candidates=(group.candidates[0], drifted),
        )


def test_row_rejects_swapping_source_and_d_placeholder_blocks() -> None:
    row = _group().rows[0]
    with pytest.raises(ValueError, match="native placeholder order"):
        RepresentationReadoutRow(
            sample_id=row.sample_id,
            image_group_key=row.image_group_key,
            source_visual_identity=row.source_visual_identity,
            supervision=row.supervision,
            input_ids=row.input_ids,
            attention_mask=row.attention_mask,
            position_ids=row.position_ids,
            source_positions=row.d_positions,
            d_positions=row.source_positions,
        )
