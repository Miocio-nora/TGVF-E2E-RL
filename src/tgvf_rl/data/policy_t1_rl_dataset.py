"""Immutable ArxivQA Policy-RL rows selected by T1 difficulty scoring.

The artifact intentionally contains no historical prompt text.  It joins the
canonical selection candidates to either provisional or final T1 decisions,
keeps only ``arxivqa`` rows whose T1 decision is ``retain``, and emits the
prompt-free fields consumed by the project-owned veRL dataset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from .deepeyes47k import DeepEyesTaskKind
from .policy_selection import (
    POLICY_SELECTION_DECISION_SCHEMA,
    SelectionCandidate,
    SelectionSource,
    T1Decision,
    canonical_json_line,
)


POLICY_T1_ARXIVQA_DATASET_KIND = "policy_t1_retained_arxivqa"
POLICY_T1_ARXIVQA_SAMPLE_SCHEMA = "tgvf.policy-t1-arxivqa-rl.sample.v1"
POLICY_T1_ARXIVQA_MANIFEST_SCHEMA = "tgvf.policy-t1-arxivqa-rl.manifest.v1"
POLICY_T1_ARXIVQA_RUNTIME_SCHEMA = "tgvf.policy-t1-arxivqa-rl.runtime.v1"
POLICY_T1_ARXIVQA_SAMPLES_FILE = "samples.jsonl"
POLICY_T1_ARXIVQA_MANIFEST_FILE = "manifest.json"
POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM = "sha256-sort-v1"


class PolicyT1DecisionStage(str, Enum):
    PROVISIONAL = "provisional"
    FINAL = "final"


class PolicyT1RLDatasetValidationError(ValueError):
    """A selected Policy-RL artifact differs from its bound identity."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PolicyT1RLDatasetValidationError(
            "Policy T1 artifact contains non-canonical JSON data"
        ) from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyT1RLDatasetValidationError(
            f"{field_name} must be a lowercase SHA-256"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyT1RLDatasetValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PolicyT1RLDatasetValidationError(
            f"input must be a regular non-symlink file: {path}"
        )
    with path.open("rb") as handle:
        observed = 0
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise PolicyT1RLDatasetValidationError(
                    f"{path}:{line_number}: blank lines are forbidden"
                )
            try:
                text = raw_line.decode("utf-8", errors="strict")
                value = json.loads(
                    text,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        PolicyT1RLDatasetValidationError(
                            f"non-finite JSON number: {value}"
                        )
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PolicyT1RLDatasetValidationError(
                    f"{path}:{line_number}: invalid strict UTF-8 JSON"
                ) from error
            if not isinstance(value, dict):
                raise PolicyT1RLDatasetValidationError(
                    f"{path}:{line_number}: row must be an object"
                )
            if raw_line != canonical_json_line(value):
                raise PolicyT1RLDatasetValidationError(
                    f"{path}:{line_number}: row is not canonical JSON"
                )
            observed += 1
            yield value
    if observed == 0:
        raise PolicyT1RLDatasetValidationError(f"input is empty: {path}")


def _decision_identity(record: Mapping[str, Any]) -> tuple[str, str, T1Decision]:
    if (
        set(record)
        != {
            "schema_version",
            "candidate_sha256",
            "sample_id",
            "source",
            "t1",
            "t2",
        }
        or record.get("schema_version") != POLICY_SELECTION_DECISION_SCHEMA
    ):
        raise PolicyT1RLDatasetValidationError("T1 decision schema differs")
    candidate_sha256 = _require_sha256(
        record.get("candidate_sha256"), "decision.candidate_sha256"
    )
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise PolicyT1RLDatasetValidationError("decision.sample_id must be non-empty")
    if record.get("source") != SelectionSource.ARXIVQA.value:
        raise PolicyT1RLDatasetValidationError(
            "ArxivQA decision route received another source"
        )
    t1 = record.get("t1")
    if not isinstance(t1, Mapping) or set(t1) != {
        "decision",
        "full_image",
        "reason",
    }:
        raise PolicyT1RLDatasetValidationError("decision.t1 schema differs")
    try:
        decision = T1Decision(t1.get("decision"))
    except ValueError as error:
        raise PolicyT1RLDatasetValidationError("decision.t1 value differs") from error
    if not isinstance(t1.get("full_image"), Mapping):
        raise PolicyT1RLDatasetValidationError(
            "decision.t1.full_image must be an object"
        )
    if not isinstance(t1.get("reason"), str) or not t1["reason"].strip():
        raise PolicyT1RLDatasetValidationError("decision.t1.reason must be non-empty")
    return sample_id, candidate_sha256, decision


def _shuffle_key(sample_id: str, seed: int) -> tuple[str, str]:
    payload = f"{POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM}\0{seed}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest(), sample_id


def _resolved_image(candidate: SelectionCandidate) -> tuple[Path, str, int, int]:
    raw_path = candidate.image.get("path")
    if not isinstance(raw_path, str):
        raise PolicyT1RLDatasetValidationError("candidate image.path is required")
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PolicyT1RLDatasetValidationError(
            "candidate image must be an absolute regular non-symlink file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise PolicyT1RLDatasetValidationError("candidate image path is not normalized")
    image_sha256 = _require_sha256(candidate.image.get("sha256"), "image.sha256")
    if _sha256_file(resolved) != image_sha256:
        raise PolicyT1RLDatasetValidationError("candidate image SHA-256 differs")
    return (
        resolved,
        image_sha256,
        int(candidate.image["width"]),
        int(candidate.image["height"]),
    )


@dataclass(frozen=True, slots=True)
class PolicyT1RLMaterializationResult:
    output_root: Path
    sample_count: int
    samples_sha256: str
    content_sha256: str
    manifest_file_sha256: str
    iteration_identity_sha256: str
    shuffle_seed: int
    decision_stage: PolicyT1DecisionStage

    def as_record(self) -> dict[str, object]:
        return {
            "dataset_kind": POLICY_T1_ARXIVQA_DATASET_KIND,
            "root": str(self.output_root),
            "decision_stage": self.decision_stage.value,
            "sample_count": self.sample_count,
            "manifest_file_sha256": self.manifest_file_sha256,
            "content_sha256": self.content_sha256,
            "samples_sha256": self.samples_sha256,
            "iteration_identity_sha256": self.iteration_identity_sha256,
            "shuffle_seed": self.shuffle_seed,
        }


@dataclass(frozen=True, slots=True)
class PolicyT1RLRuntimeBinding:
    manifest_file_sha256: str
    content_sha256: str
    shuffle_seed: int
    decision_stage: PolicyT1DecisionStage
    expected_sample_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _require_sha256(self.content_sha256, "content_sha256")
        if type(self.shuffle_seed) is not int or self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be a non-negative integer")
        if not isinstance(self.decision_stage, PolicyT1DecisionStage):
            raise TypeError("decision_stage must be PolicyT1DecisionStage")
        if (
            type(self.expected_sample_count) is not int
            or self.expected_sample_count <= 0
        ):
            raise ValueError("expected_sample_count must be positive")


def policy_t1_rl_iteration_identity_sha256(
    binding: PolicyT1RLRuntimeBinding, *, samples_sha256: str
) -> str:
    _require_sha256(samples_sha256, "samples_sha256")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": POLICY_T1_ARXIVQA_RUNTIME_SCHEMA,
                "dataset_kind": POLICY_T1_ARXIVQA_DATASET_KIND,
                "decision_stage": binding.decision_stage.value,
                "sample_count": binding.expected_sample_count,
                "shuffle_algorithm": POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM,
                "shuffle_seed": binding.shuffle_seed,
                "manifest_file_sha256": binding.manifest_file_sha256,
                "content_sha256": binding.content_sha256,
                "samples_sha256": samples_sha256,
            }
        )
    )


def materialize_policy_t1_arxivqa_rl_dataset(
    candidates_path: str | Path,
    decisions_path: str | Path,
    output_root: str | Path,
    *,
    decision_stage: PolicyT1DecisionStage,
    shuffle_seed: int = 42,
) -> PolicyT1RLMaterializationResult:
    """Join T1 decisions to candidates and publish retained ArxivQA rows."""

    if not isinstance(decision_stage, PolicyT1DecisionStage):
        raise TypeError("decision_stage must be PolicyT1DecisionStage")
    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a non-negative integer")
    candidates_path = Path(candidates_path).resolve()
    decisions_path = Path(decisions_path).resolve()
    output_root = Path(output_root).resolve()
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to replace Policy T1 artifact: {output_root}")

    candidate_file_sha256 = _sha256_file(candidates_path)
    decision_file_sha256 = _sha256_file(decisions_path)
    candidates: dict[str, SelectionCandidate] = {}
    candidate_rows = 0
    for record in _jsonl_records(candidates_path):
        candidate_rows += 1
        candidate = SelectionCandidate.from_record(record)
        if candidate.source is not SelectionSource.ARXIVQA:
            continue
        if candidate.sample_id in candidates:
            raise PolicyT1RLDatasetValidationError(
                "duplicate ArxivQA candidate sample_id"
            )
        candidates[candidate.sample_id] = candidate

    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    seen_arxivqa: set[str] = set()
    decision_rows = 0
    for decision_record in _jsonl_records(decisions_path):
        decision_rows += 1
        if decision_record.get("source") != SelectionSource.ARXIVQA.value:
            continue
        sample_id, candidate_sha256, decision = _decision_identity(decision_record)
        candidate = candidates.get(sample_id)
        if candidate is None:
            raise PolicyT1RLDatasetValidationError(
                "ArxivQA T1 decision refers to an unknown candidate"
            )
        if sample_id in seen_arxivqa:
            raise PolicyT1RLDatasetValidationError("duplicate ArxivQA T1 decision")
        seen_arxivqa.add(sample_id)
        if candidate.identity_sha256 != candidate_sha256:
            raise PolicyT1RLDatasetValidationError(
                "T1 decision candidate identity differs"
            )
        decision_counts[decision.value] += 1
        if decision is not T1Decision.RETAIN:
            continue
        if (
            not isinstance(candidate.ground_truth, str)
            or not candidate.ground_truth.strip()
        ):
            raise PolicyT1RLDatasetValidationError(
                "retained ArxivQA ground truth must be non-empty text"
            )
        image_path, image_sha256, width, height = _resolved_image(candidate)
        records.append(
            {
                "schema_version": POLICY_T1_ARXIVQA_SAMPLE_SCHEMA,
                "sample_id": candidate.sample_id,
                "candidate_sha256": candidate.identity_sha256,
                "decision_sha256": _sha256_bytes(
                    _canonical_json_bytes(decision_record)
                ),
                "image": {
                    "path": str(image_path),
                    "sha256": image_sha256,
                    "width": width,
                    "height": height,
                },
                "extra_info": {"question": candidate.question},
                "reward_model": {"ground_truth": candidate.ground_truth},
                "data_source": SelectionSource.ARXIVQA.value,
                "task_kind": DeepEyesTaskKind.MCQ.value,
                "selection": {
                    "decision_stage": decision_stage.value,
                    "t1": json.loads(_canonical_json_bytes(decision_record["t1"])),
                },
            }
        )
    if set(candidates) != seen_arxivqa:
        raise PolicyT1RLDatasetValidationError(
            "T1 decisions do not cover the complete ArxivQA candidate population"
        )
    if not records:
        raise PolicyT1RLDatasetValidationError("T1 retained no ArxivQA rows")
    records.sort(key=lambda row: _shuffle_key(str(row["sample_id"]), shuffle_seed))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".policy-t1-arxivqa-", dir=output_root.parent)
    )
    try:
        samples_payload = b"".join(canonical_json_line(row) for row in records)
        samples_sha256 = _sha256_bytes(samples_payload)
        (temporary_root / POLICY_T1_ARXIVQA_SAMPLES_FILE).write_bytes(samples_payload)
        descriptor = {
            "schema_version": POLICY_T1_ARXIVQA_MANIFEST_SCHEMA,
            "dataset_kind": POLICY_T1_ARXIVQA_DATASET_KIND,
            "decision_stage": decision_stage.value,
            "source": SelectionSource.ARXIVQA.value,
            "inputs": {
                "candidates": {
                    "path": str(candidates_path),
                    "sha256": candidate_file_sha256,
                    "rows": candidate_rows,
                    "arxivqa_rows": len(candidates),
                },
                "decisions": {
                    "path": str(decisions_path),
                    "sha256": decision_file_sha256,
                    "rows": decision_rows,
                    "arxivqa_rows": len(seen_arxivqa),
                },
            },
            "t1_decision_counts": {
                value.value: decision_counts[value.value] for value in T1Decision
            },
            "sample_count": len(records),
            "shuffle": {
                "algorithm": POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM,
                "seed": shuffle_seed,
            },
            "samples": {
                "path": POLICY_T1_ARXIVQA_SAMPLES_FILE,
                "rows": len(records),
                "sha256": samples_sha256,
            },
            "images": {"address": "absolute-path-plus-sha256", "bytes_verified": True},
        }
        content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_payload = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / POLICY_T1_ARXIVQA_MANIFEST_FILE).write_bytes(manifest_payload)
        manifest_file_sha256 = _sha256_bytes(manifest_payload)
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    binding = PolicyT1RLRuntimeBinding(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        shuffle_seed=shuffle_seed,
        decision_stage=decision_stage,
        expected_sample_count=len(records),
    )
    return PolicyT1RLMaterializationResult(
        output_root=output_root,
        sample_count=len(records),
        samples_sha256=samples_sha256,
        content_sha256=content_sha256,
        manifest_file_sha256=manifest_file_sha256,
        iteration_identity_sha256=policy_t1_rl_iteration_identity_sha256(
            binding, samples_sha256=samples_sha256
        ),
        shuffle_seed=shuffle_seed,
        decision_stage=decision_stage,
    )


@dataclass(frozen=True, slots=True)
class PolicyT1RLRuntimeSample:
    sample_id: str
    image_path: Path
    image_sha256: str
    question: str
    ground_truth: str
    data_source: str
    task_kind: DeepEyesTaskKind
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PolicyT1RLRuntimeDataset:
    root: Path
    binding: PolicyT1RLRuntimeBinding
    samples_sha256: str
    iteration_identity_sha256: str
    samples: tuple[PolicyT1RLRuntimeSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicyT1RLRuntimeSample:
        return self.samples[index]


_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_kind",
    "decision_stage",
    "source",
    "inputs",
    "t1_decision_counts",
    "sample_count",
    "shuffle",
    "samples",
    "images",
    "content_sha256",
}
_ROW_FIELDS = {
    "schema_version",
    "sample_id",
    "candidate_sha256",
    "decision_sha256",
    "image",
    "extra_info",
    "reward_model",
    "data_source",
    "task_kind",
    "selection",
}


def verify_policy_t1_rl_artifact_binding(
    root: str | Path,
    *,
    binding: PolicyT1RLRuntimeBinding,
    samples_sha256: str,
) -> Mapping[str, Any]:
    """Verify immutable manifest/sample-file identity without opening images."""

    root = Path(root).resolve(strict=True)
    manifest_path = root / POLICY_T1_ARXIVQA_MANIFEST_FILE
    samples_path = root / POLICY_T1_ARXIVQA_SAMPLES_FILE
    for path in (manifest_path, samples_path):
        if path.is_symlink() or not path.is_file():
            raise PolicyT1RLDatasetValidationError("Policy T1 artifact file is unsafe")
    manifest_payload = manifest_path.read_bytes()
    if _sha256_bytes(manifest_payload) != binding.manifest_file_sha256:
        raise PolicyT1RLDatasetValidationError("Policy T1 manifest file hash differs")
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as error:
        raise PolicyT1RLDatasetValidationError(
            "Policy T1 manifest JSON is invalid"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise PolicyT1RLDatasetValidationError("Policy T1 manifest schema differs")
    if manifest_payload != _canonical_json_bytes(manifest) + b"\n":
        raise PolicyT1RLDatasetValidationError("Policy T1 manifest is not canonical")
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if (
        manifest.get("content_sha256") != binding.content_sha256
        or _sha256_bytes(_canonical_json_bytes(descriptor)) != binding.content_sha256
        or manifest.get("schema_version") != POLICY_T1_ARXIVQA_MANIFEST_SCHEMA
        or manifest.get("dataset_kind") != POLICY_T1_ARXIVQA_DATASET_KIND
        or manifest.get("decision_stage") != binding.decision_stage.value
        or manifest.get("source") != SelectionSource.ARXIVQA.value
        or manifest.get("sample_count") != binding.expected_sample_count
        or manifest.get("shuffle")
        != {
            "algorithm": POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM,
            "seed": binding.shuffle_seed,
        }
        or manifest.get("samples")
        != {
            "path": POLICY_T1_ARXIVQA_SAMPLES_FILE,
            "rows": binding.expected_sample_count,
            "sha256": samples_sha256,
        }
    ):
        raise PolicyT1RLDatasetValidationError("Policy T1 manifest identity differs")
    if _sha256_file(samples_path) != samples_sha256:
        raise PolicyT1RLDatasetValidationError("Policy T1 samples file hash differs")
    return manifest


def load_policy_t1_rl_runtime(
    root: str | Path, *, binding: PolicyT1RLRuntimeBinding
) -> PolicyT1RLRuntimeDataset:
    root = Path(root).resolve(strict=True)
    manifest_path = root / POLICY_T1_ARXIVQA_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_bytes())
    samples_sha256 = manifest["samples"]["sha256"]
    verify_policy_t1_rl_artifact_binding(
        root, binding=binding, samples_sha256=samples_sha256
    )
    samples: list[PolicyT1RLRuntimeSample] = []
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for record in _jsonl_records(root / POLICY_T1_ARXIVQA_SAMPLES_FILE):
        if (
            set(record) != _ROW_FIELDS
            or record.get("schema_version") != POLICY_T1_ARXIVQA_SAMPLE_SCHEMA
        ):
            raise PolicyT1RLDatasetValidationError("Policy T1 sample schema differs")
        sample_id = record.get("sample_id")
        image = record.get("image")
        extra_info = record.get("extra_info")
        reward_model = record.get("reward_model")
        selection = record.get("selection")
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or sample_id in seen
            or not isinstance(image, Mapping)
            or set(image) != {"path", "sha256", "width", "height"}
            or not isinstance(extra_info, Mapping)
            or set(extra_info) != {"question"}
            or not isinstance(reward_model, Mapping)
            or set(reward_model) != {"ground_truth"}
            or not isinstance(selection, Mapping)
            or set(selection) != {"decision_stage", "t1"}
        ):
            raise PolicyT1RLDatasetValidationError("Policy T1 sample fields differ")
        if (
            record.get("data_source") != SelectionSource.ARXIVQA.value
            or record.get("task_kind") != DeepEyesTaskKind.MCQ.value
            or selection.get("decision_stage") != binding.decision_stage.value
            or not isinstance(selection.get("t1"), Mapping)
            or selection["t1"].get("decision") != T1Decision.RETAIN.value
        ):
            raise PolicyT1RLDatasetValidationError("Policy T1 sample route differs")
        image_path = Path(str(image.get("path")))
        image_sha256 = _require_sha256(image.get("sha256"), "image.sha256")
        if (
            not image_path.is_absolute()
            or image_path.is_symlink()
            or not image_path.is_file()
            or image_path.resolve(strict=True) != image_path
            or _sha256_file(image_path) != image_sha256
        ):
            raise PolicyT1RLDatasetValidationError("Policy T1 source image differs")
        question = extra_info.get("question")
        ground_truth = reward_model.get("ground_truth")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(ground_truth, str)
            or not ground_truth.strip()
        ):
            raise PolicyT1RLDatasetValidationError("Policy T1 task text differs")
        _require_sha256(record.get("candidate_sha256"), "candidate_sha256")
        _require_sha256(record.get("decision_sha256"), "decision_sha256")
        seen.add(sample_id)
        ordered_ids.append(sample_id)
        samples.append(
            PolicyT1RLRuntimeSample(
                sample_id=sample_id,
                image_path=image_path,
                image_sha256=image_sha256,
                question=question,
                ground_truth=ground_truth,
                data_source=SelectionSource.ARXIVQA.value,
                task_kind=DeepEyesTaskKind.MCQ,
                metadata={
                    "candidate_sha256": record["candidate_sha256"],
                    "decision_sha256": record["decision_sha256"],
                    "decision_stage": binding.decision_stage.value,
                },
            )
        )
    if len(samples) != binding.expected_sample_count:
        raise PolicyT1RLDatasetValidationError("Policy T1 sample count differs")
    if ordered_ids != sorted(
        ordered_ids, key=lambda value: _shuffle_key(value, binding.shuffle_seed)
    ):
        raise PolicyT1RLDatasetValidationError("Policy T1 shuffle order differs")
    return PolicyT1RLRuntimeDataset(
        root=root,
        binding=binding,
        samples_sha256=samples_sha256,
        iteration_identity_sha256=policy_t1_rl_iteration_identity_sha256(
            binding, samples_sha256=samples_sha256
        ),
        samples=tuple(samples),
    )


__all__ = [
    "POLICY_T1_ARXIVQA_DATASET_KIND",
    "POLICY_T1_ARXIVQA_MANIFEST_FILE",
    "POLICY_T1_ARXIVQA_MANIFEST_SCHEMA",
    "POLICY_T1_ARXIVQA_RUNTIME_SCHEMA",
    "POLICY_T1_ARXIVQA_SAMPLE_SCHEMA",
    "POLICY_T1_ARXIVQA_SAMPLES_FILE",
    "POLICY_T1_ARXIVQA_SHUFFLE_ALGORITHM",
    "PolicyT1DecisionStage",
    "PolicyT1RLDatasetValidationError",
    "PolicyT1RLMaterializationResult",
    "PolicyT1RLRuntimeBinding",
    "PolicyT1RLRuntimeDataset",
    "PolicyT1RLRuntimeSample",
    "load_policy_t1_rl_runtime",
    "materialize_policy_t1_arxivqa_rl_dataset",
    "policy_t1_rl_iteration_identity_sha256",
    "verify_policy_t1_rl_artifact_binding",
]
