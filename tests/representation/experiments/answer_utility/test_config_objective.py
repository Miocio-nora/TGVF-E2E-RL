from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tgvf_rl.representation.experiments.answer_utility.config import (
    ANSWER_UTILITY_EXPERIMENT_CONFIG_SCHEMA_VERSION,
    ANSWER_UTILITY_EXPERIMENT_SCOPE,
    AnswerSupervisionView,
    AnswerUtilityExperimentVariant,
    answer_utility_experiment_profile,
    load_answer_utility_experiment_config,
)
from tgvf_rl.representation.experiments.answer_utility.objective import (
    ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION,
    AnswerUtilityObjectiveConfig,
    AnswerUtilityObjectiveTerms,
    compose_answer_utility_objective,
)
from tgvf_rl.representation.experiments.answer_utility.trainer import (
    _validate_trainable_profile,
)


def _objective(
    variant: AnswerUtilityExperimentVariant,
    *,
    margin: float | None = None,
) -> AnswerUtilityObjectiveConfig:
    profile = answer_utility_experiment_profile(variant)
    answer, zero, wrong, evidence, matrix, norm = profile.expected_loss_weights
    comparisons_active = zero > 0.0 or wrong > 0.0
    return AnswerUtilityObjectiveConfig(
        answer_weight=answer,
        correct_vs_zero_weight=zero,
        correct_vs_wrong_weight=wrong,
        existing_evidence_weight=evidence,
        existing_matrix_weight=matrix,
        norm_weight=norm,
        comparison_margin=(0.5 if comparisons_active else 0.0)
        if margin is None
        else margin,
        comparison_temperature=1.0,
    )


def _write_sidecar(
    tmp_path: Path,
    *,
    variant: AnswerUtilityExperimentVariant = AnswerUtilityExperimentVariant.E4,
    base_sha256: str | None = None,
    objective_suffix: str = "",
    root_suffix: str = "",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base-training.toml"
    base.write_text('schema_version = "fixture"\n', encoding="utf-8")
    digest = (
        sha256(base.read_bytes()).hexdigest() if base_sha256 is None else base_sha256
    )
    profile = answer_utility_experiment_profile(variant)
    answer, zero, wrong, evidence, matrix, norm = profile.expected_loss_weights
    margin = 0.5 if zero > 0.0 or wrong > 0.0 else 0.0
    sidecar = tmp_path / "answer-utility.toml"
    sidecar.write_text(
        f'''schema_version = "{ANSWER_UTILITY_EXPERIMENT_CONFIG_SCHEMA_VERSION}"
scope = "{ANSWER_UTILITY_EXPERIMENT_SCOPE}"
run_id = "answer-utility-fixture"
variant = "{variant.value}"
base_training_config_path = "{base}"
base_training_config_sha256 = "{digest}"
{root_suffix}
[objective]
schema_version = "{ANSWER_UTILITY_OBJECTIVE_SCHEMA_VERSION}"
answer_weight = {answer}
correct_vs_zero_weight = {zero}
correct_vs_wrong_weight = {wrong}
existing_evidence_weight = {evidence}
existing_matrix_weight = {matrix}
norm_weight = {norm}
comparison_margin = {margin}
comparison_temperature = 1.0
{objective_suffix}
''',
        encoding="utf-8",
    )
    return sidecar, base


def test_e0_to_e4_profiles_freeze_views_controls_and_weights() -> None:
    expected = {
        AnswerUtilityExperimentVariant.E0: (
            AnswerSupervisionView.NONE,
            False,
            False,
            False,
            (0.0, 0.0, 0.0, 1.0, 1.0, 0.1),
        ),
        AnswerUtilityExperimentVariant.E0_CONTINUATION: (
            AnswerSupervisionView.NONE,
            True,
            False,
            False,
            (0.0, 0.0, 0.0, 1.0, 1.0, 0.1),
        ),
        AnswerUtilityExperimentVariant.E1: (
            AnswerSupervisionView.GOLD_EVIDENCE,
            True,
            False,
            False,
            (1.0, 0.0, 0.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E2: (
            AnswerSupervisionView.CLEAN_D_ONLY,
            True,
            False,
            False,
            (1.0, 0.0, 0.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E3: (
            AnswerSupervisionView.GOLD_EVIDENCE,
            True,
            True,
            True,
            (1.0, 1.0, 1.0, 0.25, 0.25, 0.1),
        ),
        AnswerUtilityExperimentVariant.E4: (
            AnswerSupervisionView.CLEAN_D_ONLY,
            True,
            True,
            True,
            (1.0, 1.0, 1.0, 0.25, 0.25, 0.1),
        ),
    }

    for variant, values in expected.items():
        profile = answer_utility_experiment_profile(variant)
        assert (
            profile.answer_supervision_view,
            profile.train_adapter,
            profile.requires_zero_control,
            profile.requires_wrong_control,
            profile.expected_loss_weights,
        ) == values

    with pytest.raises(TypeError, match="must be AnswerUtilityExperimentVariant"):
        answer_utility_experiment_profile("e4")  # type: ignore[arg-type]


def test_strict_sidecar_loads_and_byte_binds_the_base_training_config(
    tmp_path: Path,
) -> None:
    sidecar, base = _write_sidecar(tmp_path)

    config = load_answer_utility_experiment_config(sidecar)

    assert config.variant is AnswerUtilityExperimentVariant.E4
    assert config.profile.answer_supervision_view is AnswerSupervisionView.CLEAN_D_ONLY
    assert config.base_training_config_path == base
    assert config.base_training_config_sha256 == sha256(base.read_bytes()).hexdigest()
    assert len(config.source_toml_sha256) == 64
    assert len(config.canonical_config_sha256) == 64
    payload = config.validation_payload()
    assert payload["objective_loss_weights"] == [1.0, 1.0, 1.0, 0.25, 0.25, 0.1]
    assert payload["gpu_work_launched"] is False

    base.write_text('schema_version = "changed"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_answer_utility_experiment_config(sidecar)


def test_sidecar_rejects_schema_drift_and_profile_weight_override(
    tmp_path: Path,
) -> None:
    unknown, _ = _write_sidecar(tmp_path / "unknown", root_suffix="unknown = true")
    with pytest.raises(ValueError, match="unknown"):
        load_answer_utility_experiment_config(unknown)

    drift, _ = _write_sidecar(tmp_path / "drift")
    drift.write_text(
        drift.read_text(encoding="utf-8").replace(
            "existing_evidence_weight = 0.25",
            "existing_evidence_weight = 0.5",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen e4 profile"):
        load_answer_utility_experiment_config(drift)

    non_float, _ = _write_sidecar(tmp_path / "non-float")
    non_float.write_text(
        non_float.read_text(encoding="utf-8").replace(
            "answer_weight = 1.0", "answer_weight = 1"
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="explicit finite TOML float"):
        load_answer_utility_experiment_config(non_float)


def test_e4_objective_composes_absolute_counterfactual_and_auxiliary_terms() -> None:
    config = _objective(AnswerUtilityExperimentVariant.E4)
    correct = torch.tensor(1.0, requires_grad=True)
    zero = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor(3.0, requires_grad=True)
    evidence = torch.tensor(4.0, requires_grad=True)
    matrix = torch.tensor(8.0, requires_grad=True)
    norm = torch.tensor(10.0, requires_grad=True)

    result = compose_answer_utility_objective(
        AnswerUtilityObjectiveTerms(
            correct_answer_nll=correct,
            zero_answer_nll=zero,
            wrong_answer_nll=wrong,
            existing_evidence_loss=evidence,
            existing_matrix_loss=matrix,
            norm_loss=norm,
        ),
        config,
    )

    expected_zero = F.softplus(torch.tensor(-0.5))
    expected_wrong = F.softplus(torch.tensor(-1.5))
    expected_total = (
        correct
        + expected_zero
        + expected_wrong
        + 0.25 * evidence
        + 0.25 * matrix
        + 0.1 * norm
    )
    assert torch.allclose(result.correct_vs_zero_loss, expected_zero)
    assert torch.allclose(result.correct_vs_wrong_loss, expected_wrong)
    assert torch.allclose(result.total_loss, expected_total)

    result.total_loss.backward()
    assert correct.grad is not None and correct.grad.item() > 0.0
    assert zero.grad is not None and zero.grad.item() < 0.0
    assert wrong.grad is not None and wrong.grad.item() < 0.0
    assert evidence.grad is not None and evidence.grad.item() == pytest.approx(0.25)
    assert matrix.grad is not None and matrix.grad.item() == pytest.approx(0.25)
    assert norm.grad is not None and norm.grad.item() == pytest.approx(0.1)


def test_e0_composes_only_the_unchanged_legacy_objective() -> None:
    result = compose_answer_utility_objective(
        AnswerUtilityObjectiveTerms(
            existing_evidence_loss=torch.tensor(2.0),
            existing_matrix_loss=torch.tensor(3.0),
            norm_loss=torch.tensor(4.0),
        ),
        _objective(AnswerUtilityExperimentVariant.E0),
    )

    assert result.answer_nll is None
    assert result.correct_vs_zero_loss is None
    assert result.correct_vs_wrong_loss is None
    assert torch.equal(result.total_loss, torch.tensor(5.4))


def test_only_named_e0_continuation_can_train_the_no_answer_profile() -> None:
    e0 = answer_utility_experiment_profile(AnswerUtilityExperimentVariant.E0)
    with pytest.raises(ValueError, match="evaluation-only E0"):
        _validate_trainable_profile(
            e0,
            _objective(AnswerUtilityExperimentVariant.E0),
            AnswerSupervisionView.NONE,
        )

    continuation = answer_utility_experiment_profile(
        AnswerUtilityExperimentVariant.E0_CONTINUATION
    )
    _validate_trainable_profile(
        continuation,
        _objective(AnswerUtilityExperimentVariant.E0_CONTINUATION),
        AnswerSupervisionView.NONE,
    )


def test_objective_fails_closed_on_missing_inactive_or_incompatible_terms() -> None:
    e4 = _objective(AnswerUtilityExperimentVariant.E4)
    complete = {
        "correct_answer_nll": torch.tensor(1.0),
        "zero_answer_nll": torch.tensor(2.0),
        "wrong_answer_nll": torch.tensor(3.0),
        "existing_evidence_loss": torch.tensor(4.0),
        "existing_matrix_loss": torch.tensor(5.0),
        "norm_loss": torch.tensor(6.0),
    }
    missing = dict(complete)
    missing["wrong_answer_nll"] = None
    with pytest.raises(ValueError, match="requires wrong_answer_nll"):
        compose_answer_utility_objective(AnswerUtilityObjectiveTerms(**missing), e4)

    mixed = dict(complete)
    mixed["norm_loss"] = torch.tensor(6.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="share device and dtype"):
        compose_answer_utility_objective(AnswerUtilityObjectiveTerms(**mixed), e4)

    e2 = _objective(AnswerUtilityExperimentVariant.E2)
    with pytest.raises(ValueError, match="inactive objective term zero_answer_nll"):
        compose_answer_utility_objective(
            AnswerUtilityObjectiveTerms(
                correct_answer_nll=torch.tensor(1.0),
                zero_answer_nll=torch.tensor(2.0),
                existing_evidence_loss=torch.tensor(3.0),
                existing_matrix_loss=torch.tensor(4.0),
                norm_loss=torch.tensor(5.0),
            ),
            e2,
        )

    with pytest.raises(TypeError, match="explicit float"):
        AnswerUtilityObjectiveConfig(
            answer_weight=1,  # type: ignore[arg-type]
            correct_vs_zero_weight=0.0,
            correct_vs_wrong_weight=0.0,
            existing_evidence_weight=0.25,
            existing_matrix_weight=0.25,
            norm_weight=0.1,
            comparison_margin=0.0,
            comparison_temperature=1.0,
        )
    with pytest.raises(ValueError, match="positive margin"):
        AnswerUtilityObjectiveConfig(
            answer_weight=1.0,
            correct_vs_zero_weight=1.0,
            correct_vs_wrong_weight=1.0,
            existing_evidence_weight=0.25,
            existing_matrix_weight=0.25,
            norm_weight=0.1,
            comparison_margin=0.0,
            comparison_temperature=1.0,
        )
