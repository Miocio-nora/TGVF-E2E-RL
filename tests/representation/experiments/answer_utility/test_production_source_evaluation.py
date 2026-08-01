from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.representation.experiments.answer_utility.evaluation import runner
from tgvf_rl.representation.experiments.answer_utility.evaluation.runner import (
    ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION,
    ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION,
    AnswerUtilityEvaluationCandidate,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate(tmp_path: Path, *, private: bool) -> AnswerUtilityEvaluationCandidate:
    adapter_path = (tmp_path / ("private.pt" if private else "source.pt")).resolve()
    source_path = (tmp_path / "source.pt").resolve()
    return AnswerUtilityEvaluationCandidate(
        kind="private_formal500" if private else "production_source",
        candidate_id="private" if private else "source",
        adapter_path=adapter_path,
        adapter_file_sha256=_SHA_A if private else _SHA_B,
        adapter_state_sha256=_SHA_C,
        adapter_state={"weight": torch.ones(1)},
        global_step=500 if private else 2000,
        training_run_identity_sha256=_SHA_A if private else _SHA_B,
        production_source_artifact_path=source_path,
        production_source_artifact_sha256=_SHA_B,
        production_source_manifest_sha256=_SHA_C,
        production_source_run_identity_sha256=_SHA_B,
        production_source_global_step=2000,
        protected_paths=(tmp_path.resolve(),),
        private_run_id="private" if private else None,
        private_run_config_path=(tmp_path / "private.toml").resolve()
        if private
        else None,
        private_run_config_sha256=_SHA_A if private else None,
        private_experiment_config_sha256=_SHA_B if private else None,
    )


def _source_binding_fixture(tmp_path: Path) -> tuple[object, object, object]:
    training_path = tmp_path / "training.toml"
    training_path.write_text("training", encoding="utf-8")
    artifact_path = tmp_path / "adapter.pt"
    artifact_path.write_bytes(b"adapter")
    data_path = (tmp_path / "test.jsonl").resolve()
    run_identity = SimpleNamespace(identity_sha256=_SHA_A)
    export = SimpleNamespace(
        manifest=SimpleNamespace(
            run_identity_sha256=_SHA_A,
            run_identity=run_identity,
            global_step=2000,
        ),
        state={"weight": torch.ones(1)},
    )
    training = SimpleNamespace(
        source_path=training_path.resolve(),
        source_toml_sha256=_file_sha256(training_path),
        data=SimpleNamespace(
            validation=SimpleNamespace(jsonl_path=data_path, source_sha256=_SHA_C)
        ),
    )
    source = SimpleNamespace(
        training_config_path=training_path.resolve(),
        training_config_sha256=_file_sha256(training_path),
        artifact_path=artifact_path.resolve(),
        artifact_file_sha256=_file_sha256(artifact_path),
        artifact_manifest_sha256=_SHA_B,
        expected_run_identity_sha256=_SHA_A,
        expected_global_step=2000,
        evaluation_data_path=data_path,
        evaluation_data_source_sha256=_SHA_C,
    )
    return training, source, export


def test_evaluation_and_record_schemas_are_v2() -> None:
    assert ANSWER_UTILITY_EVALUATION_SCHEMA_VERSION.endswith("-v2")
    assert ANSWER_UTILITY_EVALUATION_RECORD_SCHEMA_VERSION.endswith("-v2")


def test_validated_source_export_checks_file_manifest_identity_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, source, export = _source_binding_fixture(tmp_path)
    binding_calls: list[object] = []
    monkeypatch.setattr(runner, "state_digest", lambda _manifest: _SHA_B)
    monkeypatch.setattr(
        runner, "load_rank_zero_adapter_owned_state_export", lambda _path: export
    )
    monkeypatch.setattr(
        runner,
        "_validate_training_artifact_binding",
        lambda _training, identity: binding_calls.append(identity),
    )

    observed = runner._load_validated_production_export(training, source)

    assert observed is export
    assert binding_calls == [export.manifest.run_identity]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_file_sha256", _SHA_C, "Adapter SHA256 mismatch"),
        ("artifact_manifest_sha256", _SHA_C, "manifest SHA256 mismatch"),
        ("expected_run_identity_sha256", _SHA_C, "identity/step mismatch"),
        ("expected_global_step", 1999, "identity/step mismatch"),
    ),
)
def test_validated_source_export_rejects_tampered_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    training, source, export = _source_binding_fixture(tmp_path)
    setattr(source, field, value)
    monkeypatch.setattr(runner, "state_digest", lambda _manifest: _SHA_B)
    monkeypatch.setattr(
        runner, "load_rank_zero_adapter_owned_state_export", lambda _path: export
    )
    monkeypatch.setattr(
        runner, "_validate_training_artifact_binding", lambda *_args: None
    )

    with pytest.raises(ValueError, match=message):
        runner._load_validated_production_export(training, source)


def test_production_source_loader_never_uses_private_metrics_or_run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, source, export = _source_binding_fixture(tmp_path)
    training.run_id = "RP-66"
    source_config_path = (tmp_path / "source-evaluation.toml").resolve()
    monkeypatch.setattr(
        runner,
        "load_representation_internal_evaluation_run_config",
        lambda _path: source,
    )
    training_loads: list[tuple[Path, bool]] = []

    def load_training(
        path: Path, *, allow_existing_post_training_report: bool = False
    ) -> object:
        training_loads.append((path, allow_existing_post_training_report))
        return training

    monkeypatch.setattr(runner, "load_representation_training_config", load_training)
    monkeypatch.setattr(runner, "_require_instruct_training", lambda _config: None)
    monkeypatch.setattr(
        runner, "_load_validated_production_export", lambda *_args: export
    )
    monkeypatch.setattr(
        runner,
        "load_answer_utility_run_config",
        lambda _path: pytest.fail("production source loaded a private run config"),
    )
    monkeypatch.setattr(
        runner,
        "_audit_completed_training_metrics",
        lambda *_args: pytest.fail("production source audited private metrics"),
    )
    monkeypatch.setattr(
        runner,
        "_materialize_common_inputs",
        lambda **keywords: keywords["candidate"],
    )

    candidate = runner._load_production_source_inputs(
        source_config_path,
        arms=runner.DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS,
        max_new_tokens=None,
        eos_token_ids=None,
        decode_mode="cached",
        group_start=0,
        group_limit=None,
        shard_index=0,
        shard_count=1,
    )

    assert candidate.kind == "production_source"
    assert candidate.private_run_id is None
    assert candidate.adapter_path == source.artifact_path
    assert candidate.training_run_identity_sha256 == _SHA_A
    assert training_loads == [(source.training_config_path, True)]


def test_candidate_identity_distinguishes_source_and_private_and_binds_scorer_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_implementation_file_manifest", lambda: {})
    source_evaluation = SimpleNamespace(
        source_path=(tmp_path / "source-eval.toml").resolve(),
        source_sha256=_SHA_A,
        evaluation=SimpleNamespace(
            ordered_group_manifest_sha256=_SHA_B,
            random_seed=42,
        ),
    )
    training = SimpleNamespace(
        model=SimpleNamespace(
            model_name="Qwen3-VL-8B-Instruct",
            local_path=(tmp_path / "Qwen3-VL-8B-Instruct").resolve(),
        )
    )

    def payload(candidate: AnswerUtilityEvaluationCandidate) -> dict[str, object]:
        inputs = SimpleNamespace(
            candidate=candidate,
            source_evaluation=source_evaluation,
            training=training,
            data_manifest_sha256=_SHA_C,
            ordered_group_manifest_identity=_SHA_A,
            wrong_source_by_sample_id={},
            arms=(),
            max_new_tokens=64,
            eos_token_ids=(151645, 151643),
            decode_mode="cached",
            group_start=0,
            group_limit=None,
            shard_index=0,
            shard_count=1,
        )
        return runner._evaluation_identity_payload(inputs, ())

    private = payload(_candidate(tmp_path, private=True))
    production = payload(_candidate(tmp_path, private=False))

    assert private["candidate_kind"] == "private_formal500"
    assert production["candidate_kind"] == "production_source"
    assert (
        private["candidate_training_run_identity_sha256"]
        != production["candidate_training_run_identity_sha256"]
    )
    assert private["private_run_config_path"] is not None
    assert production["private_run_config_path"] is None
    assert production["choice_text_reference_mapping_enabled"] is True


def test_cli_requires_exactly_one_candidate_and_reads_source_gpu(
    tmp_path: Path,
) -> None:
    tool_path = (
        Path(__file__).resolve().parents[4]
        / "tools"
        / "run_representation_answer_utility_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("answer_utility_eval_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.toml"
    source.write_text("[execution]\nphysical_gpu_id = 3\n", encoding="utf-8")

    parsed = module._parser().parse_args(
        [
            "--production-source",
            "--source-evaluation-config",
            str(source),
            "--validate-only",
        ]
    )

    assert parsed.production_source is True
    assert parsed.run_config is None
    assert module._configured_gpu(source) == 3
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            ["--source-evaluation-config", str(source), "--validate-only"]
        )
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--production-source",
                "--run-config",
                str(source),
                "--source-evaluation-config",
                str(source),
                "--validate-only",
            ]
        )
