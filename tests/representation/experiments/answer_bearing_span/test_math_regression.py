from __future__ import annotations

from dataclasses import dataclass
import math
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import pytest
import torch

from tgvf_rl.representation.training.losses import (
    EVIDENCE_IGNORE_INDEX,
    MatrixCEScoreMode,
    causal_evidence_losses,
    historical_sample_norm_loss,
    matrix_ce_cell_scores,
    same_image_matrix_ce_loss_terms,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.readout import (
    RepresentationReadoutLossSupervision,
    RepresentationVisualTensorBundle,
)
from tgvf_rl.representation.training.streaming import _candidate_norm_loss


_VOCABULARY_SIZE = 4
_TARGET_CELL_PROBABILITIES = (
    ((0.80, 0.50), (0.20, 0.40)),
    ((0.30, 0.20), (0.70, 0.60)),
)
_IMAGE_CELL_PROBABILITIES = (
    ((0.80, 0.50), (0.25, 0.20)),
    ((0.70, 0.60), (0.40, 0.30)),
)


@dataclass(frozen=True, slots=True)
class _RP70Components:
    target_scores: torch.Tensor
    target_matrix: torch.Tensor
    l_gen: torch.Tensor
    image_scores: torch.Tensor
    image_matrix: torch.Tensor
    target_token_counts: torch.Tensor
    image_token_counts: torch.Tensor


def _span_supervision(
    *, value_token: int, answer_token: int
) -> RepresentationReadoutLossSupervision:
    labels = (
        EVIDENCE_IGNORE_INDEX,
        value_token,
        EVIDENCE_IGNORE_INDEX,
        answer_token,
        EVIDENCE_IGNORE_INDEX,
    )
    return RepresentationReadoutLossSupervision(
        identity=f"rp70-test-{value_token}-{answer_token}",
        labels=labels,
        supervised_token_positions=(1, 3),
        evidence_value_token_positions=(1,),
        answer_token_positions=(3,),
        source_image_block_query_start=0,
        source_image_block_query_end=3,
    )


def _target_distribution(token: int, probability: float) -> torch.Tensor:
    remainder = (1.0 - probability) / (_VOCABULARY_SIZE - 1)
    probabilities = torch.full(
        (_VOCABULARY_SIZE,),
        remainder,
        dtype=torch.float64,
    )
    probabilities[token] = probability
    return probabilities.log()


def _cell_logits(
    supervision: RepresentationReadoutLossSupervision,
    probabilities: tuple[float, float],
) -> torch.Tensor:
    value_token = supervision.labels[supervision.evidence_value_token_positions[0]]
    answer_token = supervision.labels[supervision.answer_token_positions[0]]
    logits = torch.zeros(5, _VOCABULARY_SIZE, dtype=torch.float64)
    # Causal query 0 predicts the value label at position 1.  Query 2 predicts
    # the answer label at position 3.  Query 1 predicts a masked prose token.
    logits[0] = _target_distribution(value_token, probabilities[0])
    logits[2] = _target_distribution(answer_token, probabilities[1])
    logits[1] = torch.tensor((3.0, -2.0, 1.0, -4.0), dtype=torch.float64)
    return logits


def _materialize_cells(
    probabilities: tuple[tuple[tuple[float, float], ...], ...],
    supervisions: tuple[
        RepresentationReadoutLossSupervision,
        RepresentationReadoutLossSupervision,
    ],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    labels: list[tuple[int, ...]] = []
    for row_index, row_probabilities in enumerate(probabilities):
        for cell_probabilities in row_probabilities:
            supervision = supervisions[row_index]
            logits.append(_cell_logits(supervision, cell_probabilities))
            labels.append(supervision.labels)
    return torch.stack(logits), torch.tensor(labels, dtype=torch.long)


def _components(
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    image_logits: torch.Tensor,
    image_labels: torch.Tensor,
) -> _RP70Components:
    target_losses = causal_evidence_losses(target_logits, target_labels)
    target_scores = matrix_ce_cell_scores(
        target_losses,
        mode=MatrixCEScoreMode.BALANCED,
        temperature=1.0,
    ).reshape(2, 2)
    target_matrix = same_image_matrix_ce_loss_terms((target_scores,)).mean
    diagonal_indices = torch.tensor((0, 3), dtype=torch.long)
    l_gen = target_losses.per_sample_token_mean_nll[diagonal_indices].mean()

    image_losses = causal_evidence_losses(image_logits, image_labels)
    image_scores = matrix_ce_cell_scores(
        image_losses,
        mode=MatrixCEScoreMode.BALANCED,
        temperature=1.0,
    ).reshape(2, 2)
    image_matrix = _diagonal_row_ce(image_scores, labels=(0, 0))
    return _RP70Components(
        target_scores=target_scores,
        target_matrix=target_matrix,
        l_gen=l_gen,
        image_scores=image_scores,
        image_matrix=image_matrix,
        target_token_counts=target_losses.valid_token_counts,
        image_token_counts=image_losses.valid_token_counts,
    )


def _diagonal_row_ce(scores: torch.Tensor, *, labels: tuple[int, ...]) -> torch.Tensor:
    row_losses = tuple(
        torch.logsumexp(row, dim=0) - row[label]
        for row, label in zip(scores, labels, strict=True)
    )
    return torch.stack(row_losses).mean()


def _hand_score(probabilities: tuple[float, float]) -> float:
    return (math.log(probabilities[0]) + math.log(probabilities[1])) / 2.0


def _hand_score_matrix(
    probabilities: tuple[tuple[tuple[float, float], ...], ...],
) -> torch.Tensor:
    return torch.tensor(
        tuple(tuple(_hand_score(cell) for cell in row) for row in probabilities),
        dtype=torch.float64,
    )


def _rp70_inputs() -> tuple[
    tuple[RepresentationReadoutLossSupervision, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    supervisions = (
        _span_supervision(value_token=0, answer_token=1),
        _span_supervision(value_token=2, answer_token=0),
    )
    target_logits, target_labels = _materialize_cells(
        _TARGET_CELL_PROBABILITIES,
        supervisions,
    )
    image_logits, image_labels = _materialize_cells(
        _IMAGE_CELL_PROBABILITIES,
        supervisions,
    )
    return (
        supervisions,
        target_logits,
        target_labels,
        image_logits,
        image_labels,
    )


def test_sparse_value_and_answer_labels_drive_both_rp70_matrices() -> None:
    (
        supervisions,
        target_logits,
        target_labels,
        image_logits,
        image_labels,
    ) = _rp70_inputs()

    actual = _components(
        target_logits,
        target_labels,
        image_logits,
        image_labels,
    )
    expected_target_scores = _hand_score_matrix(_TARGET_CELL_PROBABILITIES)
    expected_image_scores = _hand_score_matrix(_IMAGE_CELL_PROBABILITIES)

    assert all(
        supervision.supervised_token_positions == (1, 3)
        and supervision.evidence_value_token_positions == (1,)
        and supervision.answer_token_positions == (3,)
        for supervision in supervisions
    )
    assert torch.equal(actual.target_token_counts, torch.full((4,), 2))
    assert torch.equal(actual.image_token_counts, torch.full((4,), 2))
    assert torch.allclose(actual.target_scores, expected_target_scores, atol=1e-12)
    assert torch.allclose(actual.image_scores, expected_image_scores, atol=1e-12)
    assert torch.allclose(
        actual.l_gen,
        -torch.stack(
            (expected_target_scores[0, 0], expected_target_scores[1, 1])
        ).mean(),
        atol=1e-12,
    )
    assert torch.allclose(
        actual.target_matrix,
        _diagonal_row_ce(expected_target_scores, labels=(0, 1)),
        atol=1e-12,
    )
    assert torch.allclose(
        actual.image_matrix,
        _diagonal_row_ce(expected_image_scores, labels=(0, 0)),
        atol=1e-12,
    )


def test_masked_evidence_logits_cannot_change_any_rp70_readout_loss() -> None:
    _, target_logits, target_labels, image_logits, image_labels = _rp70_inputs()
    baseline = _components(
        target_logits,
        target_labels,
        image_logits,
        image_labels,
    )

    changed_target = target_logits.clone()
    changed_image = image_logits.clone()
    # Logits at query 1 predict label position 2: descriptive evidence prose,
    # which RP70 masks.  Make the perturbation deliberately extreme.
    changed_target[:, 1] = torch.tensor(
        (1000.0, -1000.0, 777.0, -555.0), dtype=torch.float64
    )
    changed_image[:, 1] = torch.tensor(
        (-999.0, 999.0, -444.0, 333.0), dtype=torch.float64
    )
    perturbed = _components(
        changed_target,
        target_labels,
        changed_image,
        image_labels,
    )

    assert torch.equal(perturbed.l_gen, baseline.l_gen)
    assert torch.equal(perturbed.target_matrix, baseline.target_matrix)
    assert torch.equal(perturbed.image_matrix, baseline.image_matrix)
    assert torch.equal(perturbed.target_scores, baseline.target_scores)
    assert torch.equal(perturbed.image_scores, baseline.image_scores)


@pytest.mark.parametrize(
    ("causal_query", "target_token"),
    ((0, 0), (2, 1)),
    ids=("evidence-value", "final-answer"),
)
def test_value_or_answer_logit_changes_all_expected_rp70_readout_losses(
    causal_query: int,
    target_token: int,
) -> None:
    _, target_logits, target_labels, image_logits, image_labels = _rp70_inputs()
    baseline = _components(
        target_logits,
        target_labels,
        image_logits,
        image_labels,
    )

    changed_target = target_logits.clone()
    changed_image = image_logits.clone()
    degraded = _target_distribution(target_token, 0.01)
    # Cell 0 is row 0's diagonal target cell and row 0's correct-image cell.
    changed_target[0, causal_query] = degraded
    changed_image[0, causal_query] = degraded
    perturbed = _components(
        changed_target,
        target_labels,
        changed_image,
        image_labels,
    )

    assert perturbed.l_gen > baseline.l_gen
    assert perturbed.target_matrix > baseline.target_matrix
    assert perturbed.image_matrix > baseline.image_matrix


def _visual_bundle(
    main: torch.Tensor,
    deepstack: tuple[torch.Tensor, ...],
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=main,
        deepstack=deepstack,
        branch_layers=(8, 16, 24),
    )


def _norm_inputs() -> tuple[
    torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
    main_d = torch.tensor(
        [[[3.0, 4.0], [5.0, 12.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    main_source = torch.tensor(
        [[[0.0, 2.0], [0.0, 6.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    deepstack_d = tuple(
        torch.tensor(
            [[[value, 0.0], [0.0, value + 1.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        for value in (2.0, 4.0, 8.0)
    )
    deepstack_source = tuple(
        torch.tensor(
            [[[value, 0.0], [0.0, value * 2.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        for value in (1.0, 2.0, 4.0)
    )
    return main_d, main_source, deepstack_d, deepstack_source


def test_rp70_norm_is_exact_rp66_value_and_gradient_with_weight_point_one() -> None:
    rp66_inputs = _norm_inputs()
    rp70_inputs = _norm_inputs()
    rp66_loss = historical_sample_norm_loss(*rp66_inputs)
    rp70_loss = _candidate_norm_loss(
        _visual_bundle(rp70_inputs[0], rp70_inputs[2]),
        _visual_bundle(rp70_inputs[1], rp70_inputs[3]),
    )

    rp66_trainable = (rp66_inputs[0], *rp66_inputs[2])
    rp70_trainable = (rp70_inputs[0], *rp70_inputs[2])
    rp66_gradients = torch.autograd.grad(rp66_loss, rp66_trainable)
    rp70_gradients = torch.autograd.grad(rp70_loss, rp70_trainable)
    objective = RepresentationObjectiveConfigV3(
        identity="rp70-answer-bearing-span-test",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
        matrix_ce_mode=MatrixCEScoreMode.BALANCED,
        matrix_ce_temperature=1.0,
    )

    assert torch.equal(rp70_loss, rp66_loss)
    for rp70_gradient, rp66_gradient in zip(
        rp70_gradients,
        rp66_gradients,
        strict=True,
    ):
        assert torch.equal(rp70_gradient, rp66_gradient)
    assert rp66_inputs[1].grad is None
    assert all(source.grad is None for source in rp66_inputs[3])
    assert rp70_inputs[1].grad is None
    assert all(source.grad is None for source in rp70_inputs[3])
    assert objective.norm_weight == 0.1
    assert torch.equal(rp70_loss * objective.norm_weight, rp66_loss * 0.1)
