from __future__ import annotations

import pytest
import torch

from tgvf_rl.representation.training.losses import (
    EvidenceReadabilityLossTerms,
    HistoricalNormLossTerms,
    MatrixCEScoreMode,
    SameImageMatrixCELossTerms,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
    compose_reference_representation_objective,
    resolve_matrix_ce_score_config,
)


def test_v3_binds_balanced_matrix_ce_mode_and_temperature() -> None:
    balanced = RepresentationObjectiveConfigV3(
        identity="balanced-matrix-ce",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
        matrix_ce_mode=MatrixCEScoreMode.BALANCED,
        matrix_ce_temperature=0.75,
    )
    legacy_v2 = RepresentationObjectiveConfigV2(
        identity="historical-matrix-ce",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
    )

    assert resolve_matrix_ce_score_config(balanced) == (
        MatrixCEScoreMode.BALANCED,
        0.75,
    )
    assert resolve_matrix_ce_score_config(legacy_v2) == (
        MatrixCEScoreMode.LEGACY_SUMMED_NLL,
        1.0,
    )

    with pytest.raises(ValueError, match="legacy_summed_nll requires"):
        RepresentationObjectiveConfigV3(
            identity="invalid-tempered-legacy",
            kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
            matrix_ce_weight=1.0,
            l_gen_weight=1.0,
            norm_weight=0.1,
            matrix_ce_mode=MatrixCEScoreMode.LEGACY_SUMMED_NLL,
            matrix_ce_temperature=0.75,
        )


def test_v2_baseline_composes_and_logs_raw_and_weighted_norm() -> None:
    config = RepresentationObjectiveConfigV2(
        identity="native-qwen3-historical-norm-baseline",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
    )
    result = compose_reference_representation_objective(
        SameImageMatrixCELossTerms(torch.tensor(4.0), 2),
        EvidenceReadabilityLossTerms(torch.tensor(8.0), 2),
        config,
        HistoricalNormLossTerms(torch.tensor(2.0), 2),
    )

    assert result.norm_loss is not None
    assert result.weighted_norm is not None
    assert torch.equal(result.norm_loss, torch.tensor(1.0))
    assert torch.equal(result.weighted_norm, torch.tensor(0.1))
    assert torch.equal(result.total_loss, torch.tensor(6.1))
    assert result.norm_sample_count == 2


def test_v2_rejects_no_norm_while_v1_retains_named_matrix_only_ablation() -> None:
    with pytest.raises(ValueError, match="historical norm weight 0.1"):
        RepresentationObjectiveConfigV2(
            identity="bad-baseline",
            kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
            matrix_ce_weight=1.0,
            l_gen_weight=1.0,
            norm_weight=0.0,
        )
    with pytest.raises(ValueError, match="requires the.*baseline"):
        RepresentationObjectiveConfigV2(
            identity="forbidden-matrix-only-v2",
            kind=RepresentationObjectiveKind.MATRIX_CE_ONLY_ABLATION,
            matrix_ce_weight=1.0,
            l_gen_weight=0.0,
            norm_weight=0.0,
        )
    v1_ablation = RepresentationObjectiveConfig(
        identity="historical-matrix-only-v1",
        kind=RepresentationObjectiveKind.MATRIX_CE_ONLY_ABLATION,
        matrix_ce_weight=1.0,
        l_gen_weight=0.0,
    )
    result = compose_reference_representation_objective(
        SameImageMatrixCELossTerms(torch.tensor(2.0), 2),
        EvidenceReadabilityLossTerms(torch.tensor(4.0), 2),
        v1_ablation,
    )

    assert result.norm_loss is None
    assert result.weighted_norm is None
    assert result.total_loss.item() == 1.0


def test_baseline_composes_and_returns_matrix_ce_and_l_gen_separately() -> None:
    matrix_numerator = torch.tensor(9.0)
    l_gen_numerator = torch.tensor(8.0)
    config = RepresentationObjectiveConfig(
        identity="native-qwen3-contextual-baseline",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=2.0,
        l_gen_weight=0.5,
    )

    result = compose_reference_representation_objective(
        SameImageMatrixCELossTerms(matrix_numerator, valid_row_count=4),
        EvidenceReadabilityLossTerms(l_gen_numerator, sample_count=4),
        config,
    )

    assert torch.equal(result.matrix_ce_loss, torch.tensor(2.25))
    assert torch.equal(result.l_gen_loss, torch.tensor(2.0))
    assert torch.equal(result.weighted_matrix_ce, torch.tensor(4.5))
    assert torch.equal(result.weighted_l_gen, torch.tensor(1.0))
    assert torch.equal(result.total_loss, torch.tensor(5.5))


def test_baseline_rejects_zero_l_gen_but_named_ablation_requires_it() -> None:
    with pytest.raises(ValueError, match="baseline requires a nonzero L_gen"):
        RepresentationObjectiveConfig(
            identity="invalid-baseline",
            kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
            matrix_ce_weight=1.0,
            l_gen_weight=0.0,
        )
    with pytest.raises(ValueError, match="ablation requires L_gen weight zero"):
        RepresentationObjectiveConfig(
            identity="invalid-ablation",
            kind=RepresentationObjectiveKind.MATRIX_CE_ONLY_ABLATION,
            matrix_ce_weight=1.0,
            l_gen_weight=0.2,
        )

    config = RepresentationObjectiveConfig(
        identity="matrix-only-readability-ablation",
        kind=RepresentationObjectiveKind.MATRIX_CE_ONLY_ABLATION,
        matrix_ce_weight=1.0,
        l_gen_weight=0.0,
    )
    result = compose_reference_representation_objective(
        SameImageMatrixCELossTerms(torch.tensor(2.0), 2),
        EvidenceReadabilityLossTerms(torch.tensor(4.0), 2),
        config,
    )
    assert result.l_gen_loss.item() == 2.0
    assert result.weighted_l_gen.item() == 0.0
    assert result.total_loss.item() == 1.0

    with pytest.raises(ValueError, match="v1 does not support"):
        RepresentationObjectiveConfig(
            identity="v2-kind-under-v1-schema",
            kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
            matrix_ce_weight=1.0,
            l_gen_weight=0.0,
        )


def test_objective_weights_and_reference_terms_fail_closed() -> None:
    with pytest.raises(TypeError, match="explicit float"):
        RepresentationObjectiveConfig(
            identity="implicit-integer-weight",
            kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
            matrix_ce_weight=1,  # type: ignore[arg-type]
            l_gen_weight=1.0,
        )

    config = RepresentationObjectiveConfig(
        identity="valid",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
    )
    with pytest.raises(ValueError, match="valid Matrix-CE rows"):
        compose_reference_representation_objective(
            SameImageMatrixCELossTerms(torch.tensor(0.0), 0),
            EvidenceReadabilityLossTerms(torch.tensor(1.0), 1),
            config,
        )
    with pytest.raises(ValueError, match="same logical batch"):
        compose_reference_representation_objective(
            SameImageMatrixCELossTerms(torch.tensor(2.0), 2),
            EvidenceReadabilityLossTerms(torch.tensor(3.0), 3),
            config,
        )
    with pytest.raises(ValueError, match="share device and dtype"):
        compose_reference_representation_objective(
            SameImageMatrixCELossTerms(torch.tensor(1.0, dtype=torch.float32), 1),
            EvidenceReadabilityLossTerms(torch.tensor(1.0, dtype=torch.float64), 1),
            config,
        )
