"""CPU contracts for the single-GPU T1 interruption/resume smoke.

The smoke deliberately reuses one immutable T1 run configuration for both the
continuous control and the interrupted execution.  Reusing the exact config is
important: T1 attempt seeds bind the run-manifest hash, which includes the
configured output root.  After the control completes, its output directory is
archived and the same configured root is prepared again for stop/resume.

This module imports neither Torch nor a model runtime.  GPU process management
belongs to ``tools/smoke_policy_data_selection_t1_resume.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from tgvf_rl.artifact_contracts import canonical_json_bytes, canonical_json_sha256

from .policy_selection_runtime import (
    T1_ATTEMPTS,
    T1RunConfig,
    load_t1_run_config,
    validate_chunk_manifest,
)
from .policy_selection_vllm import load_t1_candidates, rank_candidate_chunks


T1_RESUME_SMOKE_SCHEMA = "tgvf.policy-selection.t1-resume-smoke-plan.v1"
T1_RESUME_SMOKE_REPORT_SCHEMA = "tgvf.policy-selection.t1-resume-smoke-report.v1"
T1_RESUME_SMOKE_RANK = 3
T1_RESUME_SMOKE_MAX_CHUNKS = 2
T1_RESUME_SMOKE_STOP_AFTER_CHUNKS = 1
_INSTRUCT_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOT = _REPO_ROOT / "artifacts" / "data" / "policy_selection" / "t1"


_canonical_json_bytes = canonical_json_bytes


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing immutable smoke artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_artifact_path(path: Path, *, field_name: str) -> Path:
    resolved = path.absolute()
    if resolved != path or path.is_symlink():
        raise ValueError(f"{field_name} must be an absolute non-symlink path")
    try:
        resolved.relative_to(_ARTIFACT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field_name} must remain under {_ARTIFACT_ROOT}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class T1ResumeSmokePlan:
    config_path: Path
    config_file_sha256: str
    run: T1RunConfig
    rank: int
    max_chunks: int
    stop_after_committed_chunks: int
    candidate_ids: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
    output_root: Path
    continuous_baseline_root: Path
    audit_root: Path
    plan_sha256: str

    @property
    def expected_records(self) -> int:
        return len(self.candidate_sha256s) * T1_ATTEMPTS

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": T1_RESUME_SMOKE_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "config_path": str(self.config_path),
            "config_file_sha256": self.config_file_sha256,
            "run_id": self.run.run_id,
            "run_manifest_sha256": self.run.manifest_sha256,
            "model_repository": self.run.model["repository"],
            "prompt": dict(self.run.prompt),
            "physical_gpu": self.rank,
            "worker_rank": self.rank,
            "max_chunks": self.max_chunks,
            "stop_after_committed_chunks": self.stop_after_committed_chunks,
            "candidate_ids": list(self.candidate_ids),
            "candidate_sha256s": list(self.candidate_sha256s),
            "expected_records": self.expected_records,
            "output_root": str(self.output_root),
            "continuous_baseline_root": str(self.continuous_baseline_root),
            "audit_root": str(self.audit_root),
        }


def build_t1_resume_smoke_plan(
    config_path: str | Path, *, verify_data_files: bool = True
) -> T1ResumeSmokePlan:
    """Validate and return the one accepted GPU-3 resume-smoke plan."""

    path = Path(config_path).resolve()
    run = load_t1_run_config(path, verify_data_files=verify_data_files)
    if run.model["repository"] != _INSTRUCT_REPOSITORY:
        raise ValueError("resume smoke requires Qwen3-VL-8B-Instruct")
    expected_prompt = {
        "schema": "qwen-native-user-image-question-v1",
        "user_content_order": ("image", "question"),
        "no_system": True,
        "no_tools": True,
        "add_generation_prompt": True,
    }
    if dict(run.prompt) != expected_prompt:
        raise ValueError(
            "resume smoke must retain the tool-free T1 image/question prompt"
        )
    output_root = _require_artifact_path(run.output_root, field_name="output_root")
    baseline_root = _require_artifact_path(
        output_root.with_name(f"{output_root.name}-continuous-baseline"),
        field_name="continuous_baseline_root",
    )
    audit_root = _require_artifact_path(
        output_root.with_name(f"{output_root.name}-resume-audit"),
        field_name="audit_root",
    )
    candidates = load_t1_candidates(run)
    chunks = rank_candidate_chunks(
        candidates,
        rank=T1_RESUME_SMOKE_RANK,
        world_size=int(run.runtime["world_size"]),
        chunk_candidates=int(run.runtime["chunk_candidates"]),
    )[:T1_RESUME_SMOKE_MAX_CHUNKS]
    if len(chunks) != T1_RESUME_SMOKE_MAX_CHUNKS or any(
        len(chunk) != int(run.runtime["chunk_candidates"]) for chunk in chunks
    ):
        raise ValueError("resume smoke requires two complete rank-3 chunks")
    selected = tuple(candidate for chunk in chunks for candidate in chunk)
    identity = {
        "schema_version": T1_RESUME_SMOKE_SCHEMA,
        "config_file_sha256": _sha256_file(path),
        "run_manifest_sha256": run.manifest_sha256,
        "physical_gpu": T1_RESUME_SMOKE_RANK,
        "worker_rank": T1_RESUME_SMOKE_RANK,
        "max_chunks": T1_RESUME_SMOKE_MAX_CHUNKS,
        "stop_after_committed_chunks": T1_RESUME_SMOKE_STOP_AFTER_CHUNKS,
        "candidate_sha256s": [candidate.identity_sha256 for candidate in selected],
        "output_root": str(output_root),
        "continuous_baseline_root": str(baseline_root),
        "audit_root": str(audit_root),
    }
    return T1ResumeSmokePlan(
        config_path=path,
        config_file_sha256=str(identity["config_file_sha256"]),
        run=run,
        rank=T1_RESUME_SMOKE_RANK,
        max_chunks=T1_RESUME_SMOKE_MAX_CHUNKS,
        stop_after_committed_chunks=T1_RESUME_SMOKE_STOP_AFTER_CHUNKS,
        candidate_ids=tuple(candidate.sample_id for candidate in selected),
        candidate_sha256s=tuple(candidate.identity_sha256 for candidate in selected),
        output_root=output_root,
        continuous_baseline_root=baseline_root,
        audit_root=audit_root,
        plan_sha256=canonical_json_sha256(identity),
    )


def _expected_manifest_paths(plan: T1ResumeSmokePlan) -> tuple[Path, ...]:
    return tuple(
        plan.output_root / "manifests" / f"rank-{plan.rank:02d}-chunk-{index:06d}.json"
        for index in range(plan.max_chunks)
    )


def validate_t1_resume_smoke_prefix(
    plan: T1ResumeSmokePlan, *, committed_chunks: int
) -> dict[str, Any]:
    """Validate an exact committed prefix in the configured active root."""

    if not 0 <= committed_chunks <= plan.max_chunks:
        raise ValueError("committed_chunks is outside the smoke plan")
    expected_paths = _expected_manifest_paths(plan)
    present = tuple(sorted((plan.output_root / "manifests").glob("*.json")))
    if present != expected_paths[:committed_chunks]:
        raise ValueError("committed manifests are not the exact planned prefix")
    rows: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(present):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = validate_chunk_manifest(
            record,
            output_root=plan.output_root,
            run=plan.run,
            expected_rank=plan.rank,
            expected_chunk_index=index,
        )
        rows.append(
            {
                "chunk_index": index,
                "manifest_file": manifest_path.name,
                "manifest_file_sha256": _sha256_file(manifest_path),
                "manifest_sha256": manifest.manifest_sha256,
                "evidence_file": manifest.evidence_file.as_posix(),
                "evidence_sha256": manifest.evidence_sha256,
                "logical_keys_sha256": manifest.logical_keys_sha256,
                "record_count": manifest.record_count,
            }
        )
    snapshot_identity = {
        "schema": "tgvf.policy-selection.t1-resume-smoke-snapshot.v1",
        "plan_sha256": plan.plan_sha256,
        "committed_chunks": committed_chunks,
        "records": sum(int(row["record_count"]) for row in rows),
        "chunks": rows,
    }
    return {
        **snapshot_identity,
        "snapshot_sha256": canonical_json_sha256(snapshot_identity),
    }


def write_t1_resume_smoke_artifact(path: str | Path, value: dict[str, Any]) -> None:
    _atomic_write_immutable(Path(path), _canonical_json_bytes(value) + b"\n")


def archive_t1_continuous_baseline(plan: T1ResumeSmokePlan) -> dict[str, Any]:
    """Validate and archive the continuous control without changing its bytes."""

    snapshot = validate_t1_resume_smoke_prefix(plan, committed_chunks=plan.max_chunks)
    if plan.continuous_baseline_root.exists():
        raise FileExistsError(plan.continuous_baseline_root)
    plan.audit_root.mkdir(parents=True, exist_ok=True)
    write_t1_resume_smoke_artifact(
        plan.audit_root / "continuous-baseline-snapshot.json", snapshot
    )
    os.replace(plan.output_root, plan.continuous_baseline_root)
    return snapshot


def t1_resume_smoke_core_digest(root: Path, plan: T1ResumeSmokePlan) -> str:
    """Digest only immutable run/chunk state; logs and audit files are excluded."""

    paths = [root / "run-identity.json", root / "run-config.canonical.json"]
    paths.extend(
        root / "manifests" / f"rank-{plan.rank:02d}-chunk-{index:06d}.json"
        for index in range(plan.max_chunks)
    )
    manifest_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths[2:]
    ]
    paths.extend(root / str(record["evidence_file"]) for record in manifest_records)
    identity = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return canonical_json_sha256(identity)


def compare_t1_resume_with_continuous(plan: T1ResumeSmokePlan) -> dict[str, Any]:
    """Require byte-identical control and stop/resume immutable outputs."""

    resumed = validate_t1_resume_smoke_prefix(plan, committed_chunks=plan.max_chunks)
    baseline_snapshot_path = plan.audit_root / "continuous-baseline-snapshot.json"
    if not baseline_snapshot_path.is_file():
        raise FileNotFoundError(baseline_snapshot_path)
    baseline_snapshot = json.loads(baseline_snapshot_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    compared: list[dict[str, Any]] = []
    relative_paths = [Path("run-identity.json"), Path("run-config.canonical.json")]
    for index in range(plan.max_chunks):
        manifest_relative = Path(
            "manifests", f"rank-{plan.rank:02d}-chunk-{index:06d}.json"
        )
        relative_paths.append(manifest_relative)
        baseline_manifest = json.loads(
            (plan.continuous_baseline_root / manifest_relative).read_text(
                encoding="utf-8"
            )
        )
        relative_paths.append(Path(str(baseline_manifest["evidence_file"])))
    for relative in relative_paths:
        baseline_path = plan.continuous_baseline_root / relative
        resumed_path = plan.output_root / relative
        baseline_sha = _sha256_file(baseline_path)
        resumed_sha = _sha256_file(resumed_path)
        equal = baseline_sha == resumed_sha
        if not equal:
            mismatches.append(relative.as_posix())
        compared.append(
            {
                "relative_path": relative.as_posix(),
                "continuous_sha256": baseline_sha,
                "resumed_sha256": resumed_sha,
                "byte_identical": equal,
            }
        )
    if mismatches:
        raise ValueError(
            "stop/resume differs from continuous output: " + ", ".join(mismatches)
        )
    identity = {
        "schema_version": T1_RESUME_SMOKE_REPORT_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "run_manifest_sha256": plan.run.manifest_sha256,
        "continuous_snapshot_sha256": baseline_snapshot["snapshot_sha256"],
        "resumed_snapshot_sha256": resumed["snapshot_sha256"],
        "continuous_core_sha256": t1_resume_smoke_core_digest(
            plan.continuous_baseline_root, plan
        ),
        "resumed_core_sha256": t1_resume_smoke_core_digest(plan.output_root, plan),
        "expected_records": plan.expected_records,
        "compared": compared,
        "result": "PASS",
    }
    return {
        **identity,
        "report_sha256": canonical_json_sha256(identity),
    }


__all__ = [
    "T1_RESUME_SMOKE_MAX_CHUNKS",
    "T1_RESUME_SMOKE_RANK",
    "T1_RESUME_SMOKE_REPORT_SCHEMA",
    "T1_RESUME_SMOKE_SCHEMA",
    "T1_RESUME_SMOKE_STOP_AFTER_CHUNKS",
    "T1ResumeSmokePlan",
    "archive_t1_continuous_baseline",
    "build_t1_resume_smoke_plan",
    "compare_t1_resume_with_continuous",
    "t1_resume_smoke_core_digest",
    "validate_t1_resume_smoke_prefix",
    "write_t1_resume_smoke_artifact",
]
