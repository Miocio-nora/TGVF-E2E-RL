from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tgvf_rl.representation.experiments.image_axis_grounding.handoff import (
    HandoffRejectedError,
    ImageAxisHandoffExpectation,
    ImageAxisHandoffReceipt,
    InspectedAdapterArtifact,
    main,
    verify_image_axis_training_completion,
)


RUN_ID = "RP-TEST-IMAGE-AXIS"
RUN_IDENTITY = "1" * 64
EXPERIMENT_CONFIG_SHA256 = "2" * 64
TRAINING_CONFIG_SHA256 = "3" * 64
ARTIFACT_MANIFEST_SHA256 = "4" * 64
ARTIFACT_FILE_SHA256 = "5" * 64
GLOBAL_STEP = 500


def _completion_fixture(tmp_path: Path):
    artifact = tmp_path / "adapter.pt"
    artifact.write_bytes(b"fixture Adapter bytes")
    metrics = tmp_path / "metrics.jsonl"
    outer_log = tmp_path / "invocation.log"
    core = {
        "schema_version": "representation-runner-v1",
        "status": "complete",
        "run_id": RUN_ID,
        "run_identity_sha256": RUN_IDENTITY,
        "source_toml_sha256": TRAINING_CONFIG_SHA256,
        "global_step": GLOBAL_STEP,
        "final_artifact_path": str(artifact),
        "final_artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
        "final_artifact_write_mode": "written",
        "metrics_jsonl_path": str(metrics),
    }
    outer = {
        "schema_version": "image-axis-grounding-runner-v1",
        "status": "complete",
        "experiment_run_id": RUN_ID,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "treatment_training_config_sha256": TRAINING_CONFIG_SHA256,
        "core_result": core,
    }
    completion = {"event": "complete", **core}
    metrics.write_text(
        json.dumps({"event": "train", "global_step": GLOBAL_STEP})
        + "\n"
        + json.dumps(completion)
        + "\n",
        encoding="utf-8",
    )
    outer_log.write_text(
        "torchrun warning before the result\n" + json.dumps(outer) + "\n",
        encoding="utf-8",
    )
    expectation = ImageAxisHandoffExpectation(
        run_id=RUN_ID,
        run_identity_sha256=RUN_IDENTITY,
        global_step=GLOBAL_STEP,
        experiment_config_sha256=EXPERIMENT_CONFIG_SHA256,
        training_config_sha256=TRAINING_CONFIG_SHA256,
        artifact_path=artifact,
        metrics_path=metrics,
        artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        artifact_file_sha256=ARTIFACT_FILE_SHA256,
    )
    inspected = InspectedAdapterArtifact(
        path=artifact,
        file_sha256=ARTIFACT_FILE_SHA256,
        manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        run_id=RUN_ID,
        run_identity_sha256=RUN_IDENTITY,
        global_step=GLOBAL_STEP,
        tensor_count=17,
    )
    return outer_log, metrics, artifact, outer, expectation, inspected


def test_complete_training_authorizes_handoff(tmp_path: Path) -> None:
    outer_log, _, artifact, _, expectation, inspected = _completion_fixture(tmp_path)

    receipt = verify_image_axis_training_completion(
        training_exit_code=0,
        outer_result_path=outer_log,
        expectation=expectation,
        artifact_inspector=lambda _: inspected,
    )

    assert receipt.status == "authorized"
    assert receipt.run_identity_sha256 == RUN_IDENTITY
    assert receipt.global_step == GLOBAL_STEP
    assert receipt.artifact_path == str(artifact)
    assert receipt.artifact_manifest_sha256 == ARTIFACT_MANIFEST_SHA256
    assert receipt.artifact_tensor_count == 17


def test_paused_exit_zero_does_not_authorize_handoff(tmp_path: Path) -> None:
    outer_log, metrics, _, outer, expectation, inspected = _completion_fixture(tmp_path)
    outer["status"] = "paused_at_optimizer_boundary"
    outer["core_result"]["status"] = "paused_at_optimizer_boundary"
    outer_log.write_text(json.dumps(outer) + "\n", encoding="utf-8")
    metrics.write_text(
        json.dumps({"event": "train", "global_step": 10}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HandoffRejectedError, match="no final complete record"):
        verify_image_axis_training_completion(
            training_exit_code=0,
            outer_result_path=outer_log,
            expectation=expectation,
            artifact_inspector=lambda _: inspected,
        )


def test_identity_mismatch_does_not_authorize_handoff(tmp_path: Path) -> None:
    outer_log, _, _, outer, expectation, inspected = _completion_fixture(tmp_path)
    outer["core_result"]["run_identity_sha256"] = "9" * 64
    outer_log.write_text(json.dumps(outer) + "\n", encoding="utf-8")

    with pytest.raises(HandoffRejectedError, match="core run identity mismatch"):
        verify_image_axis_training_completion(
            training_exit_code=0,
            outer_result_path=outer_log,
            expectation=expectation,
            artifact_inspector=lambda _: inspected,
        )


def test_artifact_manifest_mismatch_does_not_authorize_handoff(
    tmp_path: Path,
) -> None:
    outer_log, _, artifact, _, expectation, inspected = _completion_fixture(tmp_path)
    mismatched = InspectedAdapterArtifact(
        path=artifact,
        file_sha256=inspected.file_sha256,
        manifest_sha256="8" * 64,
        run_id=inspected.run_id,
        run_identity_sha256=inspected.run_identity_sha256,
        global_step=inspected.global_step,
        tensor_count=inspected.tensor_count,
    )

    with pytest.raises(HandoffRejectedError, match="manifest identities differ"):
        verify_image_axis_training_completion(
            training_exit_code=0,
            outer_result_path=outer_log,
            expectation=expectation,
            artifact_inspector=lambda _: mismatched,
        )


def test_nonzero_training_exit_does_not_read_or_authorize(tmp_path: Path) -> None:
    expectation = ImageAxisHandoffExpectation(
        run_id=RUN_ID,
        run_identity_sha256=RUN_IDENTITY,
        global_step=GLOBAL_STEP,
        experiment_config_sha256=EXPERIMENT_CONFIG_SHA256,
        training_config_sha256=TRAINING_CONFIG_SHA256,
        artifact_path=tmp_path / "missing-adapter.pt",
        metrics_path=tmp_path / "missing-metrics.jsonl",
    )

    with pytest.raises(HandoffRejectedError, match="exited nonzero"):
        verify_image_axis_training_completion(
            training_exit_code=7,
            outer_result_path=tmp_path / "missing.log",
            expectation=expectation,
        )


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "--training-exit-code",
        "0",
        "--outer-result-log",
        str(tmp_path / "invocation.log"),
        "--expected-run-id",
        RUN_ID,
        "--expected-run-identity-sha256",
        RUN_IDENTITY,
        "--expected-global-step",
        str(GLOBAL_STEP),
        "--expected-experiment-config-sha256",
        EXPERIMENT_CONFIG_SHA256,
        "--expected-training-config-sha256",
        TRAINING_CONFIG_SHA256,
        "--expected-artifact-path",
        str(tmp_path / "adapter.pt"),
        "--expected-metrics-path",
        str(tmp_path / "metrics.jsonl"),
    ]


def test_cli_executes_argv_only_after_verified_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = ImageAxisHandoffReceipt(
        run_id=RUN_ID,
        run_identity_sha256=RUN_IDENTITY,
        global_step=GLOBAL_STEP,
        experiment_config_sha256=EXPERIMENT_CONFIG_SHA256,
        training_config_sha256=TRAINING_CONFIG_SHA256,
        artifact_path=str(tmp_path / "adapter.pt"),
        artifact_file_sha256=ARTIFACT_FILE_SHA256,
        artifact_manifest_sha256=ARTIFACT_MANIFEST_SHA256,
        artifact_tensor_count=17,
        metrics_path=str(tmp_path / "metrics.jsonl"),
        metrics_file_sha256="6" * 64,
        outer_result_path=str(tmp_path / "invocation.log"),
        outer_result_file_sha256="7" * 64,
    )
    monkeypatch.setattr(
        "tgvf_rl.representation.experiments.image_axis_grounding.handoff."
        "verify_image_axis_training_completion",
        lambda **_: receipt,
    )

    class ExecCalled(Exception):
        pass

    def fake_execvp(file: str, argv: list[str]) -> None:
        assert file == "/bin/echo"
        assert argv == ["/bin/echo", "validated"]
        raise ExecCalled

    monkeypatch.setattr(os, "execvp", fake_execvp)
    with pytest.raises(ExecCalled):
        main([*_cli_args(tmp_path), "--execute", "--", "/bin/echo", "validated"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "authorized"
    assert payload["downstream_argv"] == ["/bin/echo", "validated"]


def test_cli_rejection_never_executes_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(**_: object) -> None:
        raise HandoffRejectedError("incomplete")

    monkeypatch.setattr(
        "tgvf_rl.representation.experiments.image_axis_grounding.handoff."
        "verify_image_axis_training_completion",
        reject,
    )
    monkeypatch.setattr(
        os,
        "execvp",
        lambda *_: pytest.fail("rejected handoff attempted downstream exec"),
    )

    assert (
        main([*_cli_args(tmp_path), "--execute", "--", "/bin/echo", "invalid"])
        == 2
    )
