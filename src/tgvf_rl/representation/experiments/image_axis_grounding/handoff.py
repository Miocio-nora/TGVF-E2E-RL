"""Fail-closed handoff from image-axis training to held-out evaluation.

The image-axis training CLI deliberately returns zero for a bounded smoke run.
Consequently, a shell ``train && evaluate`` chain is not a completion gate.  This
module authorizes a downstream command only after the current invocation result,
the durable metrics ledger, and the exported Adapter all prove the same complete
training identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any


IMAGE_AXIS_HANDOFF_RECEIPT_SCHEMA_VERSION = (
    "image-axis-grounding-evaluation-handoff-v1"
)
IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION = "image-axis-grounding-runner-v1"
REPRESENTATION_RUNNER_SCHEMA_VERSION = "representation-runner-v1"
_HEX = frozenset("0123456789abcdef")


class HandoffRejectedError(RuntimeError):
    """The training evidence did not authorize a downstream evaluation."""


@dataclass(frozen=True, slots=True)
class ImageAxisHandoffExpectation:
    """Immutable identity and output paths expected by one handoff."""

    run_id: str
    run_identity_sha256: str
    global_step: int
    experiment_config_sha256: str
    training_config_sha256: str
    artifact_path: Path
    metrics_path: Path
    artifact_manifest_sha256: str | None = None
    artifact_file_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.run_id, name="expected run ID")
        _require_sha256(self.run_identity_sha256, name="expected run identity")
        _require_positive_int(self.global_step, name="expected global step")
        _require_sha256(
            self.experiment_config_sha256,
            name="expected experiment-config SHA256",
        )
        _require_sha256(
            self.training_config_sha256,
            name="expected training-config SHA256",
        )
        _require_absolute_path(self.artifact_path, name="expected artifact path")
        _require_absolute_path(self.metrics_path, name="expected metrics path")
        if self.artifact_manifest_sha256 is not None:
            _require_sha256(
                self.artifact_manifest_sha256,
                name="expected artifact-manifest SHA256",
            )
        if self.artifact_file_sha256 is not None:
            _require_sha256(
                self.artifact_file_sha256,
                name="expected artifact-file SHA256",
            )


@dataclass(frozen=True, slots=True)
class InspectedAdapterArtifact:
    """Identity read back from a validated rank-zero Adapter export."""

    path: Path
    file_sha256: str
    manifest_sha256: str
    run_id: str
    run_identity_sha256: str
    global_step: int
    tensor_count: int


@dataclass(frozen=True, slots=True)
class ImageAxisHandoffReceipt:
    """Durable values that authorized a downstream evaluation command."""

    run_id: str
    run_identity_sha256: str
    global_step: int
    experiment_config_sha256: str
    training_config_sha256: str
    artifact_path: str
    artifact_file_sha256: str
    artifact_manifest_sha256: str
    artifact_tensor_count: int
    metrics_path: str
    metrics_file_sha256: str
    outer_result_path: str
    outer_result_file_sha256: str
    training_exit_code: int = 0
    status: str = "authorized"
    schema_version: str = IMAGE_AXIS_HANDOFF_RECEIPT_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return asdict(self)


ArtifactInspector = Callable[[Path], InspectedAdapterArtifact]


def verify_image_axis_training_completion(
    *,
    training_exit_code: int,
    outer_result_path: str | Path,
    expectation: ImageAxisHandoffExpectation,
    artifact_inspector: ArtifactInspector | None = None,
) -> ImageAxisHandoffReceipt:
    """Authorize evaluation only when all independent completion records agree."""

    if isinstance(training_exit_code, bool) or not isinstance(training_exit_code, int):
        raise TypeError("training exit code must be an integer")
    if training_exit_code != 0:
        raise HandoffRejectedError(
            f"training process exited nonzero: {training_exit_code}"
        )

    outer_path = _existing_regular_file(
        Path(outer_result_path), name="outer-result log"
    )
    metrics_path = _existing_regular_file(
        expectation.metrics_path, name="metrics ledger"
    )
    artifact_path = _existing_regular_file(
        expectation.artifact_path, name="Adapter artifact"
    )
    outer = _load_unique_outer_result(outer_path)
    core = _mapping(outer.get("core_result"), name="outer core_result")
    completion = _load_final_metrics_completion(metrics_path)

    _require_equal(
        outer.get("schema_version"),
        IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION,
        name="outer schema version",
    )
    _require_equal(outer.get("status"), "complete", name="outer status")
    _require_equal(
        outer.get("experiment_run_id"), expectation.run_id, name="outer run ID"
    )
    _require_equal(
        outer.get("experiment_config_sha256"),
        expectation.experiment_config_sha256,
        name="outer experiment-config SHA256",
    )
    _require_equal(
        outer.get("treatment_training_config_sha256"),
        expectation.training_config_sha256,
        name="outer training-config SHA256",
    )

    _validate_core_completion(core, expectation=expectation)
    _validate_metrics_completion(completion, expectation=expectation)

    core_artifact_path = _resolved_record_path(
        core.get("final_artifact_path"), name="core final-artifact path"
    )
    metrics_artifact_path = _resolved_record_path(
        completion.get("final_artifact_path"),
        name="metrics final-artifact path",
    )
    if core_artifact_path != artifact_path or metrics_artifact_path != artifact_path:
        raise HandoffRejectedError("completion records identify another artifact path")
    core_metrics_path = _resolved_record_path(
        core.get("metrics_jsonl_path"), name="core metrics path"
    )
    metrics_record_path = _resolved_record_path(
        completion.get("metrics_jsonl_path"), name="metrics self path"
    )
    if core_metrics_path != metrics_path or metrics_record_path != metrics_path:
        raise HandoffRejectedError("completion records identify another metrics path")

    inspector = _inspect_rank_zero_adapter if artifact_inspector is None else artifact_inspector
    try:
        inspected = inspector(artifact_path)
    except HandoffRejectedError:
        raise
    except Exception as error:
        raise HandoffRejectedError(
            f"Adapter artifact integrity validation failed: {type(error).__name__}: {error}"
        ) from error
    _validate_inspected_artifact(inspected, expectation=expectation)

    core_manifest = _require_sha256(
        core.get("final_artifact_manifest_sha256"),
        name="core artifact-manifest SHA256",
    )
    metrics_manifest = _require_sha256(
        completion.get("final_artifact_manifest_sha256"),
        name="metrics artifact-manifest SHA256",
    )
    if core_manifest != metrics_manifest or core_manifest != inspected.manifest_sha256:
        raise HandoffRejectedError(
            "outer/core/metrics and Adapter manifest identities differ"
        )
    if (
        expectation.artifact_manifest_sha256 is not None
        and core_manifest != expectation.artifact_manifest_sha256
    ):
        raise HandoffRejectedError("Adapter manifest differs from the expected identity")
    if (
        expectation.artifact_file_sha256 is not None
        and inspected.file_sha256 != expectation.artifact_file_sha256
    ):
        raise HandoffRejectedError("Adapter file differs from the expected SHA256")

    return ImageAxisHandoffReceipt(
        run_id=expectation.run_id,
        run_identity_sha256=expectation.run_identity_sha256,
        global_step=expectation.global_step,
        experiment_config_sha256=expectation.experiment_config_sha256,
        training_config_sha256=expectation.training_config_sha256,
        artifact_path=str(artifact_path),
        artifact_file_sha256=inspected.file_sha256,
        artifact_manifest_sha256=inspected.manifest_sha256,
        artifact_tensor_count=inspected.tensor_count,
        metrics_path=str(metrics_path),
        metrics_file_sha256=_file_sha256(metrics_path),
        outer_result_path=str(outer_path),
        outer_result_file_sha256=_file_sha256(outer_path),
    )


def _validate_core_completion(
    core: Mapping[str, Any], *, expectation: ImageAxisHandoffExpectation
) -> None:
    _require_equal(
        core.get("schema_version"),
        REPRESENTATION_RUNNER_SCHEMA_VERSION,
        name="core schema version",
    )
    _require_equal(core.get("status"), "complete", name="core status")
    _require_equal(core.get("run_id"), expectation.run_id, name="core run ID")
    _require_equal(
        core.get("run_identity_sha256"),
        expectation.run_identity_sha256,
        name="core run identity",
    )
    _require_equal(
        core.get("source_toml_sha256"),
        expectation.training_config_sha256,
        name="core training-config SHA256",
    )
    _require_equal(
        core.get("global_step"), expectation.global_step, name="core global step"
    )
    if core.get("final_artifact_write_mode") not in {"written", "reused"}:
        raise HandoffRejectedError("core artifact write mode is not committed")


def _validate_metrics_completion(
    completion: Mapping[str, Any], *, expectation: ImageAxisHandoffExpectation
) -> None:
    _require_equal(completion.get("event"), "complete", name="metrics event")
    _require_equal(completion.get("status"), "complete", name="metrics status")
    _require_equal(
        completion.get("schema_version"),
        REPRESENTATION_RUNNER_SCHEMA_VERSION,
        name="metrics schema version",
    )
    _require_equal(
        completion.get("run_id"), expectation.run_id, name="metrics run ID"
    )
    _require_equal(
        completion.get("run_identity_sha256"),
        expectation.run_identity_sha256,
        name="metrics run identity",
    )
    _require_equal(
        completion.get("source_toml_sha256"),
        expectation.training_config_sha256,
        name="metrics training-config SHA256",
    )
    _require_equal(
        completion.get("global_step"),
        expectation.global_step,
        name="metrics global step",
    )
    if completion.get("final_artifact_write_mode") not in {"written", "reused"}:
        raise HandoffRejectedError("metrics artifact write mode is not committed")


def _validate_inspected_artifact(
    artifact: InspectedAdapterArtifact, *, expectation: ImageAxisHandoffExpectation
) -> None:
    if artifact.path != expectation.artifact_path.resolve(strict=True):
        raise HandoffRejectedError("inspector returned another artifact path")
    _require_sha256(artifact.file_sha256, name="inspected artifact-file SHA256")
    _require_sha256(
        artifact.manifest_sha256, name="inspected artifact-manifest SHA256"
    )
    _require_equal(artifact.run_id, expectation.run_id, name="artifact run ID")
    _require_equal(
        artifact.run_identity_sha256,
        expectation.run_identity_sha256,
        name="artifact run identity",
    )
    _require_equal(
        artifact.global_step, expectation.global_step, name="artifact global step"
    )
    _require_positive_int(artifact.tensor_count, name="artifact tensor count")


def _inspect_rank_zero_adapter(path: Path) -> InspectedAdapterArtifact:
    from tgvf_rl.checkpoint.coordinator import state_digest
    from tgvf_rl.representation.training.distributed_checkpoint import (
        load_rank_zero_adapter_owned_state_export,
    )

    export = load_rank_zero_adapter_owned_state_export(path)
    manifest = export.manifest
    if export.state is None:
        raise HandoffRejectedError("rank-zero Adapter export has no writer state")
    return InspectedAdapterArtifact(
        path=path,
        file_sha256=_file_sha256(path),
        manifest_sha256=state_digest(manifest),
        run_id=manifest.run_identity.run_id,
        run_identity_sha256=manifest.run_identity_sha256,
        global_step=manifest.global_step,
        tensor_count=len(manifest.tensor_names),
    )


def _load_unique_outer_result(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    candidates: list[Mapping[str, Any]] = []
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, Mapping):
        candidates.append(value)
    else:
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise HandoffRejectedError("outer-result log is not UTF-8") from error
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(item, Mapping)
                and item.get("schema_version")
                == IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION
            ):
                candidates.append(item)
    candidates = [
        item
        for item in candidates
        if item.get("schema_version") == IMAGE_AXIS_GROUNDING_RUNNER_SCHEMA_VERSION
    ]
    if len(candidates) != 1:
        raise HandoffRejectedError(
            "outer-result log must contain exactly one image-axis invocation result"
        )
    return candidates[0]


def _load_final_metrics_completion(path: Path) -> Mapping[str, Any]:
    final: Mapping[str, Any] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HandoffRejectedError(
                    f"metrics line {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise HandoffRejectedError(
                    f"metrics line {line_number} is not an object"
                )
            final = value
    if final is None:
        raise HandoffRejectedError("metrics ledger is empty")
    if final.get("event") != "complete" or final.get("status") != "complete":
        raise HandoffRejectedError("metrics ledger has no final complete record")
    return final


def _existing_regular_file(path: Path, *, name: str) -> Path:
    source = path.expanduser()
    if not source.is_absolute():
        source = source.resolve()
    try:
        resolved = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise HandoffRejectedError(f"{name} is missing: {source}") from error
    if not resolved.is_file():
        raise HandoffRejectedError(f"{name} is not a regular file: {resolved}")
    return resolved


def _resolved_record_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HandoffRejectedError(f"{name} is not an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise HandoffRejectedError(f"{name} is not an absolute path")
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise HandoffRejectedError(f"{name} is missing: {path}") from error


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffRejectedError(f"{name} is not an object")
    return value


def _require_equal(observed: object, expected: object, *, name: str) -> None:
    if observed != expected:
        raise HandoffRejectedError(
            f"{name} mismatch: expected {expected!r}, got {observed!r}"
        )


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise HandoffRejectedError(f"{name} must be a lowercase SHA256")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute Path")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-exit-code", type=int, required=True)
    parser.add_argument("--outer-result-log", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-run-identity-sha256", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--expected-experiment-config-sha256", required=True)
    parser.add_argument("--expected-training-config-sha256", required=True)
    parser.add_argument("--expected-artifact-path", type=Path, required=True)
    parser.add_argument("--expected-metrics-path", type=Path, required=True)
    parser.add_argument("--expected-artifact-manifest-sha256")
    parser.add_argument("--expected-artifact-file-sha256")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Replace this process with the verified downstream argv.",
    )
    parser.add_argument("downstream_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    downstream = list(args.downstream_argv)
    if downstream[:1] == ["--"]:
        downstream = downstream[1:]
    if args.execute and not downstream:
        parser.error("--execute requires a downstream argv after --")
    if downstream and not args.execute:
        parser.error("downstream argv requires --execute")
    try:
        expectation = ImageAxisHandoffExpectation(
            run_id=args.expected_run_id,
            run_identity_sha256=args.expected_run_identity_sha256,
            global_step=args.expected_global_step,
            experiment_config_sha256=args.expected_experiment_config_sha256,
            training_config_sha256=args.expected_training_config_sha256,
            artifact_path=args.expected_artifact_path.expanduser().resolve(),
            metrics_path=args.expected_metrics_path.expanduser().resolve(),
            artifact_manifest_sha256=args.expected_artifact_manifest_sha256,
            artifact_file_sha256=args.expected_artifact_file_sha256,
        )
        receipt = verify_image_axis_training_completion(
            training_exit_code=args.training_exit_code,
            outer_result_path=args.outer_result_log,
            expectation=expectation,
        )
    except (HandoffRejectedError, TypeError, ValueError) as error:
        print(f"handoff rejected: {error}", file=sys.stderr)
        return 2
    payload = receipt.to_record()
    if downstream:
        payload["downstream_argv"] = downstream
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    if args.execute:
        os.execvp(downstream[0], downstream)
        raise RuntimeError("downstream exec unexpectedly returned")
    return 0


__all__ = [
    "HandoffRejectedError",
    "IMAGE_AXIS_HANDOFF_RECEIPT_SCHEMA_VERSION",
    "ImageAxisHandoffExpectation",
    "ImageAxisHandoffReceipt",
    "InspectedAdapterArtifact",
    "main",
    "verify_image_axis_training_completion",
]
