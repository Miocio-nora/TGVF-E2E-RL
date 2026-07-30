from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tgvf_rl.representation.experiments.image_axis_grounding.streaming as image_streaming
from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.experiments.image_axis_grounding.native_pipeline import (
    ImageAxisGroundingGroup,
)
from tgvf_rl.representation.experiments.image_axis_grounding.streaming import (
    ImageAxisGlobalNormalization,
    backward_image_axis_grounding_group,
)
from tgvf_rl.representation.experiments.image_axis_grounding.trainer import (
    ImageAxisGroundingObjectiveConfig,
)
from tgvf_rl.representation.training.losses import (
    EVIDENCE_IGNORE_INDEX,
    MatrixCEScoreMode,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
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
            hidden
            + (hidden * valid_keys.unsqueeze(-1)).sum(dim=1, keepdim=True) * 0.1
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


def _bundle(value: float, *, requires_grad: bool) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 4), value, requires_grad=requires_grad),
        deepstack=tuple(
            torch.full(
                (1, 2, 4),
                value * float(index + 1) / 10.0,
                requires_grad=requires_grad,
            )
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


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
        canonical_to_model_positions=(
            (0,),
            (1, 2),
            (3, 4),
            (5,),
            (6,),
            (7,),
        ),
    )


def _same_image_group(
    *,
    group_id: str,
    source_identity: str,
    candidate_offset: float,
    sample_ids: tuple[str, str] = ("sample-0", "sample-1"),
    padding_count: int = 1,
) -> SameImageReadoutGroup:
    rows: list[RepresentationReadoutRow] = []
    candidates: list[RepresentationCandidateObservation] = []
    for index, sample_id in enumerate(sample_ids):
        evidence = (5 + index * 2, 6 + index * 2)
        token_ids = (1, 2, 2, 2, 2, 3, *evidence)
        rows.append(
            RepresentationReadoutRow(
                sample_id=sample_id,
                image_group_key=group_id,
                source_visual_identity=source_identity,
                supervision=_supervision(token_ids),
                input_ids=torch.tensor((token_ids,), dtype=torch.long),
                attention_mask=torch.ones((1, 8), dtype=torch.bool),
                position_ids=torch.arange(8).view(1, 8),
                source_positions=(1, 2),
                d_positions=(3, 4),
            )
        )
        candidates.append(
            RepresentationCandidateObservation(
                sample_id=sample_id,
                image_group_key=group_id,
                source_visual_identity=source_identity,
                target_conditioning_provider=(
                    TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                ),
                projection_identities=("main", "ds8", "ds16", "ds24"),
                visual=_bundle(candidate_offset + index + 1.0, requires_grad=True),
                image_grid_thw=(1, 1, 2),
            )
        )
    return SameImageReadoutGroup(
        image_group_key=group_id,
        source_visual_identity=source_identity,
        source_visual=_bundle(0.25, requires_grad=False),
        rows=tuple(rows),
        candidates=tuple(candidates),
        collective_padding=tuple(
            _bundle(20.0 + index, requires_grad=True)
            for index in range(padding_count)
        ),
    )


def _paired_group(*, eligible: bool) -> ImageAxisGroundingGroup:
    base = _same_image_group(
        group_id="anchor-image",
        source_identity="anchor-source-sha",
        candidate_offset=0.0,
    )
    donor = _same_image_group(
        group_id="donor-image" if eligible else "anchor-image",
        source_identity="donor-source-sha" if eligible else "anchor-source-sha",
        candidate_offset=-0.7 if eligible else 0.0,
    )
    return ImageAxisGroundingGroup(
        base=base,
        donor=donor,
        image_axis_row_mask=(eligible, eligible),
    )


def _legacy_objective() -> RepresentationObjectiveConfigV3:
    return RepresentationObjectiveConfigV3(
        identity="test-image-axis-plus-rp66",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
        matrix_ce_mode=MatrixCEScoreMode.BALANCED,
        matrix_ce_temperature=1.0,
    )


def _normalization(*, image_rows: int) -> ImageAxisGlobalNormalization:
    return ImageAxisGlobalNormalization(
        legacy=StreamingGlobalNormalization(
            matrix_valid_rows=2,
            l_gen_samples=2,
        ),
        image_axis_valid_rows=image_rows,
    )


def _candidate_paths(
    group: SameImageReadoutGroup,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(
        (candidate.visual.main, *candidate.visual.deepstack)
        for candidate in group.candidates
    )


def test_image_axis_vjp_matches_separate_references_and_uses_one_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(73)
    seed_model = _frozen_model()
    legacy_model = _frozen_model()
    image_model = _frozen_model()
    treatment_model = _frozen_model()
    for model in (legacy_model, image_model, treatment_model):
        model.load_state_dict(seed_model.state_dict())
    legacy_group = _paired_group(eligible=True)
    image_group = _paired_group(eligible=True)
    treatment_group = _paired_group(eligible=True)
    objective = _legacy_objective()
    image_objective = ImageAxisGroundingObjectiveConfig()
    normalization = _normalization(image_rows=2)

    legacy_scores = score_streaming_same_image_group(
        Qwen3VLAdapter(),
        legacy_model,
        legacy_group.base,
        objective=objective,
        normalization=normalization.legacy,
    )
    backward_streaming_same_image_group(
        Qwen3VLAdapter(),
        legacy_model,
        legacy_group.base,
        legacy_scores,
        objective=objective,
        normalization=normalization.legacy,
    )
    legacy_gradients = tuple(
        tuple(tensor.grad.detach().clone() for tensor in paths)
        for paths in _candidate_paths(legacy_group.base)
    )
    image_materialization = image_streaming._score_image_axis_rows(
        Qwen3VLAdapter(),
        image_model,
        image_group,
        image_axis_objective=image_objective,
        normalization=normalization,
    )

    original_backward = torch.autograd.backward
    backward_calls: list[int] = []

    def recording_backward(*args: object, **kwargs: object) -> None:
        backward_calls.append(1)
        original_backward(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", recording_backward)
    metrics = backward_image_axis_grounding_group(
        Qwen3VLAdapter(),
        treatment_model,
        treatment_group,
        image_axis_objective=image_objective,
        legacy_objective=objective,
        normalization=normalization,
    )

    assert backward_calls == [1]
    assert metrics.local_image_axis_row_count == 2
    assert metrics.legacy.qwen_forward_batch_sizes == (4,)
    assert metrics.image_axis_qwen_forward_batch_sizes == (4,)
    assert len(treatment_model.model.language_model.attention_masks) == 2
    image_mask = treatment_model.model.language_model.attention_masks[1]
    minimum = torch.finfo(image_mask.dtype).min
    assert image_mask.shape == (4, 1, 8, 8)
    evidence_positions = treatment_group.base.rows[0].supervision.evidence_token_positions
    query_start = evidence_positions[0] - 1
    query_end = evidence_positions[-1]
    assert bool(
        (
            image_mask[
                :,
                :,
                query_start:query_end,
                treatment_group.base.rows[0].source_positions,
            ]
            == minimum
        )
        .all()
        .item()
    )

    for actual_paths, legacy_paths, image_paths in zip(
        _candidate_paths(treatment_group.base),
        legacy_gradients,
        image_materialization.correct_gradients,
        strict=True,
    ):
        for actual, legacy_gradient, image_gradient in zip(
            actual_paths,
            legacy_paths,
            image_paths,
            strict=True,
        ):
            assert actual.grad is not None
            assert torch.allclose(
                actual.grad,
                legacy_gradient + image_gradient,
                atol=1e-6,
                rtol=1e-6,
            )
    for actual_paths, image_paths in zip(
        _candidate_paths(treatment_group.donor),
        image_materialization.donor_gradients,
        strict=True,
    ):
        for actual, image_gradient in zip(actual_paths, image_paths, strict=True):
            assert actual.grad is not None
            assert torch.allclose(
                actual.grad,
                image_gradient,
                atol=1e-6,
                rtol=1e-6,
            )
            assert bool((actual.grad != 0).any().item())
    for padding in (
        *treatment_group.base.collective_padding,
        *treatment_group.donor.collective_padding,
    ):
        for tensor in (padding.main, *padding.deepstack):
            assert tensor.grad is not None
            assert torch.equal(tensor.grad, torch.zeros_like(tensor))
    assert all(parameter.grad is None for parameter in treatment_model.parameters())


def test_zero_eligible_group_keeps_legacy_and_zero_backprops_every_donor_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(79)
    model = _frozen_model()
    group = _paired_group(eligible=False)
    original_backward = torch.autograd.backward
    backward_calls: list[int] = []

    def recording_backward(*args: object, **kwargs: object) -> None:
        backward_calls.append(1)
        original_backward(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", recording_backward)
    metrics = backward_image_axis_grounding_group(
        Qwen3VLAdapter(),
        model,
        group,
        image_axis_objective=ImageAxisGroundingObjectiveConfig(),
        legacy_objective=_legacy_objective(),
        normalization=_normalization(image_rows=0),
    )

    assert backward_calls == [1]
    assert metrics.local_image_axis_row_count == 0
    assert metrics.correct_top1_count == 0
    assert metrics.image_axis_qwen_forward_batch_sizes == ()
    assert metrics.image_axis_numerator.item() == 0.0
    assert metrics.correct_score_sum.item() == 0.0
    assert metrics.wrong_score_sum.item() == 0.0
    assert len(model.model.language_model.attention_masks) == 1
    assert all(
        tensor.grad is not None and torch.equal(tensor.grad, torch.zeros_like(tensor))
        for paths in _candidate_paths(group.donor)
        for tensor in paths
    )
    assert all(
        tensor.grad is not None and torch.equal(tensor.grad, torch.zeros_like(tensor))
        for padding in group.donor.collective_padding
        for tensor in (padding.main, *padding.deepstack)
    )
    assert any(
        tensor.grad is not None and bool((tensor.grad != 0).any().item())
        for paths in _candidate_paths(group.base)
        for tensor in paths
    )


def test_image_axis_rejects_mixed_mask_and_shared_adapter_outputs() -> None:
    eligible = _paired_group(eligible=True)
    with pytest.raises(ValueError, match="group-homogeneous"):
        replace(eligible, image_axis_row_mask=(True, False))

    masked = _paired_group(eligible=False)
    shared_candidates = tuple(
        replace(donor, visual=base.visual)
        for base, donor in zip(
            masked.base.candidates,
            masked.donor.candidates,
            strict=True,
        )
    )
    shared = replace(masked, donor=replace(masked.donor, candidates=shared_candidates))
    with pytest.raises(ValueError, match="cannot be shared"):
        image_streaming._validate_inputs(
            shared,
            image_axis_objective=ImageAxisGroundingObjectiveConfig(),
            legacy_objective=_legacy_objective(),
            normalization=_normalization(image_rows=0),
        )
