from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tomllib

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_rp69_step1000_first200_evaluations.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "rp69_step1000_first200_evaluations", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluation
_SPEC.loader.exec_module(evaluation)


def _layout(training_root: Path) -> dict[str, object]:
    return {
        "run_id": evaluation.EXPECTED_RUN_ID,
        "adapter_path": training_root / "adapter.pt",
        "metrics_path": training_root / "metrics.jsonl",
        "checkpoint_directory": training_root / "checkpoints",
        "resume_checkpoint_path": training_root / "checkpoints" / "step500",
        "target_optimizer_steps": evaluation.EXPECTED_STEP,
    }


def _receipt(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": evaluation.RECEIPT_SCHEMA,
        "status": "complete",
        "training_config_path": str(evaluation.DEFAULT_TRAINING_CONFIG),
        "training_config_sha256": evaluation.EXPECTED_TRAINING_CONFIG_SHA256,
        "adapter_path": str(tmp_path / "adapter.pt"),
        "adapter_file_sha256": "a" * 64,
        "adapter_manifest_sha256": "b" * 64,
        "run_identity_sha256": "c" * 64,
        "evaluation_data_path": str(tmp_path / "validation.jsonl"),
        "evaluation_data_source_sha256": evaluation.EXPECTED_VALIDATION_SHA256,
        "conditioning_provider": "contextual_hidden_state",
        "prompt_identity": "qwen3-representation-image-question-v1",
        "prompt_sha256": "d" * 64,
        "resume_lineage": {
            "status": "complete",
            "source_global_step": 500,
            "target_global_step": 1000,
        },
    }


def test_exact_target_config_is_bound_to_step500_resume_lineage() -> None:
    layout = evaluation._static_training_layout(evaluation.DEFAULT_TRAINING_CONFIG)

    assert evaluation._file_sha256(evaluation.DEFAULT_TRAINING_CONFIG) == (
        evaluation.EXPECTED_TRAINING_CONFIG_SHA256
    )
    assert layout["run_id"] == evaluation.EXPECTED_RUN_ID
    assert layout["target_optimizer_steps"] == 1000
    assert layout["resume_checkpoint_path"] == evaluation.EXPECTED_RESUME_CHECKPOINT


def test_shared_controller_dependency_is_hash_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert evaluation._file_sha256(evaluation._SHARED_CONTROLLER_PATH) == (
        evaluation.SHARED_CONTROLLER_SHA256
    )
    drifted = tmp_path / "run_rp69_post_training_evaluations.py"
    drifted.write_text("raise RuntimeError('untrusted dependency')\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_SHARED_CONTROLLER_PATH", drifted)

    with pytest.raises(RuntimeError, match="pinned RP69 evaluation controller"):
        evaluation._load_shared_controller()


@pytest.mark.parametrize(
    ("old", "new", "mismatch"),
    (
        (
            "[resume]\nenabled = true",
            "[resume]\nenabled = false",
            "resume.enabled",
        ),
        (
            "[resume]\nenabled = true\ncheckpoint_path",
            "[resume]\nenabled = true\nstrict_identity = false\ncheckpoint_path",
            "resume.strict_identity",
        ),
    ),
)
def test_target_config_rejects_resume_contract_drift(
    tmp_path: Path, old: str, new: str, mismatch: str
) -> None:
    text = evaluation.DEFAULT_TRAINING_CONFIG.read_text(encoding="utf-8")
    assert old in text
    if mismatch == "resume.strict_identity":
        text = text.replace(old, new, 1).replace(
            "\nstrict_identity = true\n\n[checkpoint]",
            "\n\n[checkpoint]",
            1,
        )
    else:
        text = text.replace(old, new, 1)
    path = tmp_path / "drifted.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(evaluation.ControllerError, match=mismatch):
        evaluation._static_training_layout(path)


def test_default_output_is_isolated_and_has_no_full867_paths() -> None:
    output = evaluation._default_output_root()
    paths = evaluation._paths(output)

    evaluation._assert_output_root_isolated(
        output,
        _layout(evaluation.EXPECTED_ARTIFACT_ROOT),
    )
    assert output != evaluation.EXPECTED_ARTIFACT_ROOT
    assert evaluation.EXPECTED_ARTIFACT_ROOT not in output.parents
    assert not any("full" in key.lower() for key in paths)
    assert not any("full867" in str(path).lower() for path in paths.values())


def test_output_isolation_also_protects_source_rp69_artifacts() -> None:
    with pytest.raises(evaluation.ControllerError, match="RP69/RP69R1"):
        evaluation._assert_output_root_isolated(
            evaluation.SOURCE_ARTIFACT_ROOT / "evaluation",
            _layout(evaluation.EXPECTED_ARTIFACT_ROOT),
        )


def test_dry_run_is_write_free_and_only_plans_minimal_first200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_config = tmp_path / "training.toml"
    training_config.write_text("run_id = 'rp69r1'\n", encoding="utf-8")
    training_root = tmp_path / "training"
    output = tmp_path / "evaluation" / "rp69r1"
    monkeypatch.setattr(
        evaluation,
        "_assert_static_inputs",
        lambda _path: _layout(training_root),
    )

    result = evaluation._dry_run(
        training_config_path=training_config,
        output_root=output,
        int_gpu=0,
        first_gpus=(1,),
        judge_gpus=(0, 1),
    )

    assert result["status"] == "dry_run_complete"
    assert result["files_written"] is False
    assert result["gpu_processes_started"] is False
    assert result["evaluation_scope"] == {
        "scope": evaluation.MINIMAL_SCOPE,
        "full867": "not_run",
    }
    assert set(result["commands"]) == {"int_diag", "acc_first200"}
    assert set(result["config_preview_sha256"]) == {
        "int_diag",
        "acc_first200",
        "note",
    }
    assert result["shared_controller"] == {
        "path": str(evaluation._SHARED_CONTROLLER_PATH),
        "sha256": evaluation.SHARED_CONTROLLER_SHA256,
    }
    assert not output.exists()


def test_materialized_configs_are_step1000_first200_only(tmp_path: Path) -> None:
    paths = evaluation._paths(tmp_path / "evaluation")
    evaluation._materialize_configs(
        _receipt(tmp_path),
        paths,
        int_gpu=0,
        first_gpus=(1,),
    )

    generated = sorted((paths["root"] / "generated-configs").glob("*.toml"))
    assert generated == [paths["first_config"], paths["int_config"]]
    for path in generated:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        assert config["artifact"]["expected_global_step"] == 1000
        assert config["evaluation"]["ordered_group_manifest_path"] == str(
            evaluation.FIRST_MANIFEST
        )
        assert "full867" not in path.read_text(encoding="utf-8").lower()


def test_lineage_validator_rejects_wrong_source_identity_before_artifact_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(Path("/tmp/rp69r1-test"))
    run_identity = str(receipt["run_identity_sha256"])
    lineage = {
        "schema_version": evaluation.LINEAGE_SCHEMA,
        "status": "complete",
        "migration_kind": "terminal_probe_horizon_extension",
        "source": {
            "training_config_path": str(evaluation.SOURCE_TRAINING_CONFIG),
            "training_config_sha256": evaluation.SOURCE_TRAINING_CONFIG_SHA256,
            "checkpoint_path": str(evaluation.SOURCE_CHECKPOINT),
            "checkpoint_metadata_sha256": (
                evaluation.SOURCE_CHECKPOINT_METADATA_SHA256
            ),
            "metrics_path": str(evaluation.SOURCE_METRICS),
            "metrics_full_sha256": evaluation.SOURCE_METRICS_SHA256,
            "run_id": "wrong-source-run",
            "run_identity_sha256": evaluation.SOURCE_RUN_IDENTITY_SHA256,
            "planned_target_optimizer_steps": 500,
            "checkpoint_global_step": 500,
        },
        "target": {
            "training_config_path": str(evaluation.DEFAULT_TRAINING_CONFIG),
            "training_config_sha256": evaluation.EXPECTED_TRAINING_CONFIG_SHA256,
            "output_root": str(evaluation.EXPECTED_ARTIFACT_ROOT),
            "checkpoint_path": str(evaluation.EXPECTED_RESUME_CHECKPOINT),
            "metrics_path": str(evaluation.EXPECTED_ARTIFACT_ROOT / "metrics.jsonl"),
            "run_id": evaluation.EXPECTED_RUN_ID,
            "run_identity_sha256": run_identity,
            "planned_target_optimizer_steps": 1000,
            "lineage_manifest_path": str(evaluation.EXPECTED_LINEAGE_MANIFEST),
        },
        "preserved_state": {},
        "rewritten_state": {},
    }
    monkeypatch.setattr(evaluation, "_load_json_object", lambda *_args, **_kw: lineage)

    with pytest.raises(evaluation.ControllerError, match="lineage identity differs"):
        evaluation._validate_resume_lineage(
            training_config_path=evaluation.DEFAULT_TRAINING_CONFIG,
            receipt=receipt,
        )


def test_resume_dcp_lineage_rehashes_both_source_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_checkpoint = tmp_path / "source-step500"
    target_checkpoint = tmp_path / "target-step500"
    for checkpoint in (source_checkpoint, target_checkpoint):
        dcp = checkpoint / "dcp"
        dcp.mkdir(parents=True)
        (dcp / "__0_0.distcp").write_bytes(b"identical-dcp-payload")
    monkeypatch.setattr(evaluation, "SOURCE_CHECKPOINT", source_checkpoint)
    monkeypatch.setattr(evaluation, "EXPECTED_RESUME_CHECKPOINT", target_checkpoint)
    declared = evaluation._dcp_tree(source_checkpoint / "dcp")
    source = {"dcp_payload": declared}
    target = {"dcp_payload": declared}

    assert (
        evaluation._validate_resume_dcp_lineage(source=source, target=target)
        == declared
    )

    (target_checkpoint / "dcp" / "__0_0.distcp").write_bytes(b"target drift")
    with pytest.raises(evaluation.ControllerError, match="differs from lineage"):
        evaluation._validate_resume_dcp_lineage(source=source, target=target)

    (target_checkpoint / "dcp" / "__0_0.distcp").write_bytes(b"identical-dcp-payload")
    (source_checkpoint / "dcp" / "__0_0.distcp").write_bytes(b"source drift")
    with pytest.raises(evaluation.ControllerError, match="differs from lineage"):
        evaluation._validate_resume_dcp_lineage(source=source, target=target)


def test_completion_marker_declares_minimal_scope_without_full_artifact(
    tmp_path: Path,
) -> None:
    paths = evaluation._paths(tmp_path / "evaluation")
    required = (
        paths["receipt"],
        paths["int_config"],
        paths["first_config"],
        paths["int_report"],
        paths["first_generation"] / "launch-summary.json",
        paths["first_semantic"] / "summary.json",
        paths["first_semantic"] / "manifest.json",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    evaluation._write_complete_marker(receipt=_receipt(tmp_path), paths=paths)
    marker = json.loads(paths["complete"].read_text(encoding="utf-8"))

    assert marker["schema_version"] == evaluation.COMPLETE_SCHEMA
    assert marker["evaluation_scope"] == {
        "kind": "minimal_recovery",
        "scope": evaluation.MINIMAL_SCOPE,
        "full867": "not_run",
        "int_diag_samples": 200,
        "acc_val_samples": 200,
    }
    assert set(marker["generated_configs"]) == {"int_diag", "first200"}
    assert marker["first200"]["samples"] == 200
    assert "full867" not in marker
    assert marker["shared_controller_file_sha256"] == (
        evaluation.SHARED_CONTROLLER_SHA256
    )
    assert marker["shared_controller"]["sha256"] == (
        evaluation.SHARED_CONTROLLER_SHA256
    )


def test_argument_contract_keeps_parallel_generation_disjoint() -> None:
    args = SimpleNamespace(
        poll_seconds=1.0,
        wait_timeout_seconds=1.0,
        gpu_wait_timeout_seconds=1.0,
        evaluation_timeout_seconds=1.0,
        int_gpu=0,
        first_gpus=(0,),
        judge_gpus=(0, 1),
    )

    with pytest.raises(evaluation.ControllerError, match="must be disjoint"):
        evaluation._validate_arguments(args)


@pytest.mark.parametrize(
    ("int_gpu", "first_gpus", "judge_gpus"),
    (
        (2, (1,), (0, 1)),
        (0, (2,), (0, 1)),
        (0, (1, 2), (0, 1)),
        (0, (1,), (2, 3)),
    ),
)
def test_cli_rejects_every_noncanonical_gpu_plan(
    int_gpu: int,
    first_gpus: tuple[int, ...],
    judge_gpus: tuple[int, ...],
) -> None:
    args = SimpleNamespace(
        poll_seconds=1.0,
        wait_timeout_seconds=1.0,
        gpu_wait_timeout_seconds=1.0,
        evaluation_timeout_seconds=1.0,
        int_gpu=int_gpu,
        first_gpus=first_gpus,
        judge_gpus=judge_gpus,
    )

    with pytest.raises(evaluation.ControllerError, match="exact GPU plan"):
        evaluation._validate_arguments(args)
