from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import pytest

from tgvf_rl.representation.adapter import TGVFAdapterVariant
from tgvf_rl.representation.experiments.answer_bearing_span import config
from tgvf_rl.representation.experiments.image_axis_grounding.trainer import (
    ImageAxisGroundingObjectiveConfig,
)
from tgvf_rl.representation.training.losses import MatrixCEScoreMode
from tgvf_rl.representation.training.objective import RepresentationObjectiveKind


def test_objective_identity_binds_base_donor_both_splits_and_span_policy() -> None:
    span = config.AnswerBearingSpanPolicyConfig()
    objective = ImageAxisGroundingObjectiveConfig()

    identity = config.answer_bearing_span_treatment_objective_identity(
        base_training_config_sha256="a" * 64,
        donor_manifest_sha256="b" * 64,
        train_source_sha256="c" * 64,
        test_source_sha256="d" * 64,
        train_span_sidecar_sha256="e" * 64,
        test_span_sidecar_sha256="f" * 64,
        span=span,
        objective=objective,
    )

    assert identity == (
        "answer-bearing-span-balanced-matrix-ce-l-gen-norm-plus-image-axis-v1:"
        f"base={'a' * 64}:donor={'b' * 64}:train={'c' * 64}:test={'d' * 64}:"
        f"train_spans={'e' * 64}:test_spans={'f' * 64}:"
        f"span={span.identity}:weight=0x1.0000000000000p+0:"
        "temperature=0x1.0000000000000p+0:negatives=1:"
        "driver=answer-bearing-span-runner-v1"
    )


def test_span_policy_requires_explicit_source_bound_adjudication() -> None:
    assert config.ANSWER_BEARING_SPAN_MATCH_POLICY in (
        config.ANSWER_BEARING_SPAN_POLICY
    )
    assert config.ANSWER_BEARING_SPAN_SUPERVISION_POLICY in (
        config.ANSWER_BEARING_SPAN_POLICY
    )
    assert config.ANSWER_BEARING_SPAN_INDEX_SCHEMA_VERSION in (
        config.ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION
    )
    assert config.ANSWER_BEARING_SPAN_SUPERVISION_SCHEMA_VERSION in (
        config.ANSWER_BEARING_SPAN_POLICY_SCHEMA_VERSION
    )
    assert config.ANSWER_BEARING_SPAN_POLICY.endswith(
        "semantic-bound-explicit-no-unadjudicated"
    )
    with pytest.raises(ValueError, match="fixed"):
        config.AnswerBearingSpanPolicyConfig(policy="case-insensitive-fallback")
    with pytest.raises(ValueError, match="schema"):
        config.AnswerBearingSpanPolicyConfig(schema_version="v2")


def test_outer_config_rejects_unknown_root_before_bound_file_access(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rp70.toml"
    path.write_text(
        'schema_version = "answer-bearing-span-config-v1"\n'
        'scope = "isolated_representation_answer_bearing_span"\n'
        'run_id = "rp70"\n'
        'unexpected = "not-allowed"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[root\] fields differ"):
        config.load_answer_bearing_span_experiment_config(path)


def _terms(*, identity: str, matrix_weight: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        identity=identity,
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=matrix_weight,
        l_gen_weight=1.0,
        norm_weight=0.1,
        matrix_ce_mode=MatrixCEScoreMode.BALANCED,
        matrix_ce_temperature=1.0,
    )


def _execution(terms: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        objective=terms,
        manifold_enabled=False,
        manifold_weight=0.0,
    )


def _parity_fixture(*, matrix_weight: float = 1.0):
    shared = {
        field: object()
        for field in (
            "model",
            "provider",
            "data",
            "prompt",
            "optimizer",
            "scheduler",
            "execution",
            "initialization",
            "fsdp2",
        )
    }
    shared["scheduler"] = SimpleNamespace(total_steps=2000)
    base_terms = _terms(identity="rp66")
    treatment_terms = _terms(identity="expected-rp70", matrix_weight=matrix_weight)
    base = SimpleNamespace(
        run_id="rp66",
        adapter_variant=TGVFAdapterVariant.FULL_D_DEEPSTACK,
        objective=_execution(base_terms),
        training=SimpleNamespace(
            gradient_accumulation_steps=4,
            groups_per_rank_per_optimizer_step=1,
            log_every_optimizer_steps=10,
            target_optimizer_steps=2000,
            validation_every_optimizer_steps=500,
        ),
        checkpoint=SimpleNamespace(
            save_every_optimizer_steps=500,
            directory=Path("/tmp/rp66/checkpoints"),
        ),
        resume=SimpleNamespace(enabled=False, checkpoint_path=None),
        post_training_internal_evaluation=SimpleNamespace(enabled=True),
        output=object(),
        **shared,
    )
    treatment = SimpleNamespace(
        run_id="rp70",
        adapter_variant=TGVFAdapterVariant.FULL_D_DEEPSTACK,
        objective=_execution(treatment_terms),
        training=SimpleNamespace(
            gradient_accumulation_steps=4,
            groups_per_rank_per_optimizer_step=1,
            log_every_optimizer_steps=10,
            target_optimizer_steps=500,
            validation_every_optimizer_steps=2000,
        ),
        checkpoint=SimpleNamespace(
            save_every_optimizer_steps=500,
            directory=Path("/tmp/rp70/checkpoints"),
        ),
        resume=SimpleNamespace(enabled=False, checkpoint_path=None),
        post_training_internal_evaluation=SimpleNamespace(enabled=False),
        output=object(),
        **shared,
    )
    outer = SimpleNamespace(
        run_id="rp70",
        base_training=base,
        treatment_training=treatment,
        expected_treatment_objective_identity="expected-rp70",
    )
    return outer


def test_500_step_probe_accepts_only_fixed_legacy_loss_geometry() -> None:
    config._validate_treatment_parity(_parity_fixture())

    with pytest.raises(ValueError, match="matrix_ce_weight"):
        config._validate_treatment_parity(_parity_fixture(matrix_weight=0.9))


def test_real_500_step_probe_config_parses_and_binds_exact_objective(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    base_path = (
        root
        / "configs/representation/qwen3_instruct_balanced_t1_contextual_2000step_gpu01.toml"
    )
    template_path = (
        root / "configs/representation/experiments/image_axis_grounding/"
        "rp67_qwen3_instruct_image_axis_grounded_500_gpu01.toml"
    )
    base_sha = sha256(base_path.read_bytes()).hexdigest()
    donor_path = tmp_path / "donor.json"
    donor_path.write_bytes(b"rp67-donor-fixture")
    donor_sha = sha256(donor_path.read_bytes()).hexdigest()
    train_sidecar_path = tmp_path / "train-spans.jsonl"
    train_sidecar_path.write_bytes(b"train-span-fixture")
    train_sidecar_sha = sha256(train_sidecar_path.read_bytes()).hexdigest()
    test_sidecar_path = tmp_path / "test-spans.jsonl"
    test_sidecar_path.write_bytes(b"test-span-fixture")
    test_sidecar_sha = sha256(test_sidecar_path.read_bytes()).hexdigest()
    base = config.load_representation_training_config(base_path)
    span = config.AnswerBearingSpanPolicyConfig()
    image_objective = ImageAxisGroundingObjectiveConfig()
    expected_identity = config.answer_bearing_span_treatment_objective_identity(
        base_training_config_sha256=base_sha,
        donor_manifest_sha256=donor_sha,
        train_source_sha256=base.data.train.source_sha256,
        test_source_sha256=base.data.validation.source_sha256,
        train_span_sidecar_sha256=train_sidecar_sha,
        test_span_sidecar_sha256=test_sidecar_sha,
        span=span,
        objective=image_objective,
    )

    treatment_text = template_path.read_text(encoding="utf-8")
    old_run_id = "RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-500-GPU01"
    new_run_id = "RP-70-QWEN3-INSTRUCT-ANSWER-BEARING-SPAN-500-GPU01"
    treatment_text = treatment_text.replace(old_run_id, new_run_id)
    treatment_text = treatment_text.replace(
        next(
            line
            for line in treatment_text.splitlines()
            if line.startswith(
                'identity = "balanced-matrix-ce-l-gen-norm-plus-image-axis'
            )
        ),
        f'identity = "{expected_identity}"',
    )
    treatment_path = tmp_path / "rp70-treatment.toml"
    treatment_path.write_text(treatment_text, encoding="utf-8")
    treatment_sha = sha256(treatment_path.read_bytes()).hexdigest()

    outer = tmp_path / "rp70.toml"
    outer.write_text(
        f'schema_version = "{config.ANSWER_BEARING_SPAN_CONFIG_SCHEMA_VERSION}"\n'
        f'scope = "{config.ANSWER_BEARING_SPAN_SCOPE}"\n'
        f'run_id = "{new_run_id}"\n'
        f'base_training_config_path = "{base_path}"\n'
        f'base_training_config_sha256 = "{base_sha}"\n'
        f'treatment_training_config_path = "{treatment_path}"\n'
        f'treatment_training_config_sha256 = "{treatment_sha}"\n'
        f'donor_manifest_path = "{donor_path}"\n'
        f'donor_manifest_sha256 = "{donor_sha}"\n\n'
        f'train_span_sidecar_path = "{train_sidecar_path}"\n'
        f'train_span_sidecar_sha256 = "{train_sidecar_sha}"\n'
        f'test_span_sidecar_path = "{test_sidecar_path}"\n'
        f'test_span_sidecar_sha256 = "{test_sidecar_sha}"\n\n'
        "[span]\n"
        f'schema_version = "{span.schema_version}"\n'
        f'policy = "{span.policy}"\n\n'
        "[objective]\n"
        "image_axis_matrix_weight = 1.0\n"
        "image_axis_temperature = 1.0\n"
        "negative_count = 1\n",
        encoding="utf-8",
    )

    loaded = config.load_answer_bearing_span_experiment_config(outer)

    assert loaded.treatment_training.training.target_optimizer_steps == 500
    assert loaded.expected_treatment_objective_identity == expected_identity
    assert loaded.treatment_training.objective.objective.identity == expected_identity
