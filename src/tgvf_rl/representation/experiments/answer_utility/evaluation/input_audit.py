"""Completed-training ledger audit for answer-utility input candidates."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from ..run_config import AnswerUtilityRunConfig
from ..runner import ANSWER_UTILITY_METRICS_SCHEMA_VERSION
from .input_matching import _require_sha256


def _audit_completed_training_metrics_impl(
    run: AnswerUtilityRunConfig,
    expected_variant: str,
    artifact: Any,
) -> None:
    if not run.metrics_path.is_file():
        raise FileNotFoundError("formal answer-utility metrics ledger is missing")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        run.metrics_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid formal metrics JSON at line {line_number}"
            ) from error
        if not isinstance(row, Mapping) or (
            row.get("schema_version") != ANSWER_UTILITY_METRICS_SCHEMA_VERSION
        ):
            raise ValueError(f"formal metrics schema mismatch at line {line_number}")
        rows.append(row)
    if not rows or rows[0].get("event") != "start":
        raise ValueError("formal metrics ledger has no start record")
    identity = rows[0].get("run_identity_sha256")
    _require_sha256(identity, name="metrics run identity")
    if rows[0].get("run_id") != run.run_id or (
        rows[0].get("run_config") != run.identity_payload()
    ):
        raise ValueError("formal metrics start record differs from run sidecar")
    active_step = 0
    completed = False
    for line_number, row in enumerate(rows[1:], 2):
        event = row.get("event")
        if completed:
            raise ValueError("formal metrics contain records after completion")
        if event == "step":
            if row.get("run_identity_sha256") != identity:
                raise ValueError(
                    f"formal metrics identity mismatch at line {line_number}"
                )
            step = row.get("global_step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step <= active_step
            ):
                raise ValueError(
                    f"formal metrics step order mismatch at line {line_number}"
                )
            active_step = step
        elif event == "resume":
            if row.get("run_identity_sha256") != identity or (
                row.get("from_global_step") != active_step
            ):
                raise ValueError(
                    f"formal metrics resume mismatch at line {line_number}"
                )
        elif event == "stop":
            result = row.get("result")
            if not isinstance(result, Mapping) or (
                result.get("run_identity_sha256") != identity
                or result.get("global_step") != active_step
            ):
                raise ValueError(f"formal metrics stop mismatch at line {line_number}")
        elif event == "complete":
            result = row.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("formal metrics completion result is missing")
            expected = {
                "status": "complete",
                "run_id": run.run_id,
                "variant": expected_variant,
                "run_identity_sha256": identity,
                "global_step": run.target_optimizer_steps,
                "planned_target_optimizer_steps": run.target_optimizer_steps,
                "artifact_path": str(run.final_artifact_path),
                "metrics_path": str(run.metrics_path),
            }
            if any(result.get(name) != value for name, value in expected.items()):
                raise ValueError("formal metrics completion differs from run/artifact")
            if active_step != run.target_optimizer_steps:
                raise ValueError("formal metrics did not reach the target step")
            completed = True
        else:
            raise ValueError(
                f"unknown formal metrics event at line {line_number}: {event!r}"
            )
    if not completed or rows[-1].get("event") != "complete":
        raise ValueError("formal answer-utility training is not complete")
    if identity != artifact.run_identity_sha256:
        raise ValueError("private artifact and formal metrics run identities differ")


def _validate_private_source_bindings_impl(
    run: AnswerUtilityRunConfig,
    experiment: Any,
    source_evaluation: Any,
) -> None:
    if source_evaluation.training_config_path != experiment.base_training_config_path:
        raise ValueError("evaluation source identifies another base training config")
    if (
        source_evaluation.training_config_sha256
        != experiment.base_training_config_sha256
    ):
        raise ValueError("evaluation/base training config SHA256 mismatch")
    if source_evaluation.artifact_path != run.source_artifact.path:
        raise ValueError("evaluation source identifies another production Adapter")
    if source_evaluation.artifact_file_sha256 != run.source_artifact.file_sha256:
        raise ValueError("evaluation/source production artifact file SHA256 mismatch")
    if (
        source_evaluation.artifact_manifest_sha256
        != run.source_artifact.manifest_sha256
    ):
        raise ValueError("evaluation/source production artifact manifest mismatch")
    if (
        source_evaluation.expected_run_identity_sha256
        != run.source_artifact.expected_run_identity_sha256
    ):
        raise ValueError("evaluation/source production run identity mismatch")
    if (
        source_evaluation.expected_global_step
        != run.source_artifact.expected_global_step
    ):
        raise ValueError("evaluation/source production global step mismatch")


__all__ = [
    "_audit_completed_training_metrics_impl",
    "_validate_private_source_bindings_impl",
]
