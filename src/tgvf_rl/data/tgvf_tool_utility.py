"""Counterfactual tool-utility schedules and sidecars for Policy RL.

The mixed-v2 Policy dataset already owns its deterministic training order.  A
fresh run uses ``data.shuffle=false`` and a sequential sampler, so an
``N``-step run with prompt batch size ``B`` consumes exactly the first
``N * B`` rows.  This module exports that prefix for forced-TGVF evaluation and
only assigns utility labels after all real counterfactual attempts are
present.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .policy_t1_mixed_rl_dataset import (
    POLICY_T1_MIXED_DATASET_KIND,
    POLICY_T1_MIXED_MANIFEST_FILE,
    POLICY_T1_MIXED_SAMPLE_SCHEMA,
    POLICY_T1_MIXED_SAMPLES_FILE,
    POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
    POLICY_T1_MIXED_RUNTIME_SCHEMA,
)


TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA = "tgvf.tool-utility.schedule.v1"
TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA = "tgvf.tool-utility.schedule-row.v1"
TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA = "tgvf.tool-utility.attempt.v1"
TGVF_TOOL_UTILITY_SIDECAR_SCHEMA = "tgvf.tool-utility.sidecar.v1"
TGVF_TOOL_UTILITY_SIDECAR_ROW_SCHEMA = "tgvf.tool-utility.sidecar-row.v1"

SCHEDULE_FILE = "schedule.jsonl"
SCHEDULE_MANIFEST_FILE = "schedule-manifest.json"
SIDECAR_FILE = "tool-utility.jsonl"
SIDECAR_MANIFEST_FILE = "manifest.json"


class TGVFToolUtilityError(ValueError):
    """A utility input is incomplete or differs from its declared identity."""


@dataclass(frozen=True, slots=True)
class TGVFToolUtilityScheduleResult:
    output_root: Path
    schedule_path: Path
    manifest_path: Path
    sample_count: int
    canary_sample_count: int
    schedule_sha256: str
    manifest_sha256: str

    def as_record(self) -> dict[str, object]:
        return {
            "output_root": str(self.output_root),
            "schedule_path": str(self.schedule_path),
            "manifest_path": str(self.manifest_path),
            "sample_count": self.sample_count,
            "canary_sample_count": self.canary_sample_count,
            "schedule_sha256": self.schedule_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class TGVFToolUtilitySidecarResult:
    output_root: Path
    sidecar_path: Path
    manifest_path: Path
    sample_count: int
    sidecar_sha256: str
    manifest_sha256: str
    label_counts: Mapping[str, int]

    def as_record(self) -> dict[str, object]:
        return {
            "output_root": str(self.output_root),
            "sidecar_path": str(self.sidecar_path),
            "manifest_path": str(self.manifest_path),
            "sample_count": self.sample_count,
            "sidecar_sha256": self.sidecar_sha256,
            "manifest_sha256": self.manifest_sha256,
            "label_counts": dict(self.label_counts),
        }


@dataclass(frozen=True, slots=True)
class TGVFToolUtilityLabelBinding:
    """One immutable runtime label, bound to its exact sidecar row."""

    sample_id: str
    training_index: int
    utility_label: str
    confidence: float
    row_sha256: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise TGVFToolUtilityError("utility label sample_id must be non-empty")
        if type(self.training_index) is not int or self.training_index < 0:
            raise TGVFToolUtilityError(
                "utility label training_index must be non-negative"
            )
        if self.utility_label not in {"needed", "optional", "unnecessary"}:
            raise TGVFToolUtilityError("utility label is not accepted")
        if type(self.confidence) is not float or not 0.0 <= self.confidence <= 1.0:
            raise TGVFToolUtilityError("utility label confidence must be within [0,1]")
        _require_sha256(self.row_sha256, "utility label row_sha256")


@dataclass(frozen=True, slots=True)
class TGVFToolUtilityRuntimeBinding:
    """Verified runtime view of one content-addressed utility sidecar."""

    sidecar_path: Path
    sidecar_sha256: str
    manifest_path: Path
    manifest_sha256: str
    dataset_iteration_identity_sha256: str
    labels: Mapping[str, TGVFToolUtilityLabelBinding]

    def __post_init__(self) -> None:
        _require_sha256(self.sidecar_sha256, "sidecar_sha256")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(
            self.dataset_iteration_identity_sha256,
            "dataset_iteration_identity_sha256",
        )
        if not self.labels:
            raise TGVFToolUtilityError("utility runtime binding requires labels")

    def label_for_sample(self, sample_id: str) -> TGVFToolUtilityLabelBinding:
        """Resolve exactly one sample or fail closed instead of guessing."""

        try:
            return self.labels[sample_id]
        except KeyError as error:
            raise TGVFToolUtilityError(
                f"utility sidecar has no label for sample {sample_id!r}"
            ) from error


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
        raise TGVFToolUtilityError("utility artifact is not canonical JSON") from error


def _canonical_json_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TGVFToolUtilityError(f"{field} must be a lowercase SHA-256")
    return value


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TGVFToolUtilityError(f"{field} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TGVFToolUtilityError(f"{field} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TGVFToolUtilityError(f"{field} must contain an object")
    return value


def _jsonl(path: Path, *, field: str) -> Iterator[tuple[dict[str, Any], bytes]]:
    if path.is_symlink() or not path.is_file():
        raise TGVFToolUtilityError(f"{field} must be a regular non-symlink file")
    with path.open("rb") as handle:
        for line_number, payload in enumerate(handle, start=1):
            if not payload.strip():
                raise TGVFToolUtilityError(f"{field}:{line_number} is blank")
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TGVFToolUtilityError(
                    f"{field}:{line_number} is not UTF-8 JSON"
                ) from error
            if not isinstance(value, dict):
                raise TGVFToolUtilityError(f"{field}:{line_number} must be an object")
            yield value, payload


def _shuffle_key(sample_id: str, seed: int) -> tuple[str, str]:
    payload = f"{POLICY_T1_MIXED_SHUFFLE_ALGORITHM}\0{seed}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest(), sample_id


def _iteration_identity(
    *,
    manifest_file_sha256: str,
    content_sha256: str,
    samples_sha256: str,
    sample_count: int,
    shuffle_seed: int,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": POLICY_T1_MIXED_RUNTIME_SCHEMA,
                "dataset_kind": POLICY_T1_MIXED_DATASET_KIND,
                "decision_stage": "final",
                "sample_count": sample_count,
                "shuffle_algorithm": POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
                "shuffle_seed": shuffle_seed,
                "manifest_file_sha256": manifest_file_sha256,
                "content_sha256": content_sha256,
                "samples_sha256": samples_sha256,
            }
        )
    )


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise TGVFToolUtilityError(f"{field} must be a positive integer")
    return value


def _full_image_counts(row: Mapping[str, Any]) -> dict[str, int]:
    selection = row.get("selection")
    if not isinstance(selection, Mapping):
        raise TGVFToolUtilityError("mixed-v2 row lacks selection")
    t1 = selection.get("t1")
    if not isinstance(t1, Mapping) or t1.get("decision") != "retain":
        raise TGVFToolUtilityError("schedule row is not a retained T1 sample")
    full = t1.get("full_image")
    if not isinstance(full, Mapping):
        raise TGVFToolUtilityError("schedule row lacks full-image statistics")
    counts = {
        "correct": _positive_integer_or_zero(
            full.get("correct_count"), "correct_count"
        ),
        "expected": _positive_integer(
            full.get("expected_attempts"), "expected_attempts"
        ),
        "observed": _positive_integer(
            full.get("observed_attempts"), "observed_attempts"
        ),
        "scoreable": _positive_integer(
            full.get("scoreable_attempts"), "scoreable_attempts"
        ),
    }
    if (
        full.get("complete") is not True
        or counts["expected"] != 8
        or counts["observed"] != counts["expected"]
        or counts["scoreable"] != counts["expected"]
        or not 0 <= counts["correct"] <= counts["expected"]
    ):
        raise TGVFToolUtilityError(
            "full-image statistics are not complete 8-shot scores"
        )
    expected_accuracy = counts["correct"] / counts["expected"]
    accuracy = full.get("accuracy")
    if type(accuracy) not in (int, float) or float(accuracy) != expected_accuracy:
        raise TGVFToolUtilityError("full-image accuracy differs from attempt counts")
    return counts


def _positive_integer_or_zero(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TGVFToolUtilityError(f"{field} must be a non-negative integer")
    return value


def _schedule_projection(
    row: Mapping[str, Any],
    *,
    training_index: int,
    global_prompt_batch_size: int,
    canary_sample_count: int,
) -> dict[str, object]:
    if row.get("schema_version") != POLICY_T1_MIXED_SAMPLE_SCHEMA:
        raise TGVFToolUtilityError("mixed-v2 row schema differs")
    sample_id = row.get("sample_id")
    image = row.get("image")
    extra_info = row.get("extra_info")
    reward_model = row.get("reward_model")
    if (
        not isinstance(sample_id, str)
        or not sample_id
        or not isinstance(image, Mapping)
        or not isinstance(extra_info, Mapping)
        or not isinstance(reward_model, Mapping)
    ):
        raise TGVFToolUtilityError("mixed-v2 schedule fields differ")
    image_path = image.get("path")
    question = extra_info.get("question")
    ground_truth = reward_model.get("ground_truth")
    if (
        not isinstance(image_path, str)
        or not image_path
        or not isinstance(question, str)
        or not question
        or not isinstance(ground_truth, str)
        or not ground_truth
    ):
        raise TGVFToolUtilityError("mixed-v2 task payload differs")
    image_sha256 = _require_sha256(image.get("sha256"), "image.sha256")
    full_counts = _full_image_counts(row)
    return {
        "schema_version": TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA,
        "training_index": training_index,
        "optimizer_step": training_index // global_prompt_batch_size + 1,
        "prompt_index_in_step": training_index % global_prompt_batch_size,
        "is_canary": training_index < canary_sample_count,
        "sample_id": sample_id,
        "candidate_sha256": _require_sha256(
            row.get("candidate_sha256"), "candidate_sha256"
        ),
        "data_source": row.get("data_source"),
        "task_kind": row.get("task_kind"),
        "image": {"path": image_path, "sha256": image_sha256},
        "question": question,
        "ground_truth": ground_truth,
        "p_full": full_counts["correct"] / full_counts["expected"],
        "full_attempt_counts": full_counts,
    }


def _write_manifest(path: Path, identity: dict[str, object]) -> str:
    manifest_sha256 = _sha256_bytes(_canonical_json_bytes(identity))
    path.write_bytes(
        _canonical_json_bytes({**identity, "manifest_sha256": manifest_sha256}) + b"\n"
    )
    return manifest_sha256


def materialize_tgvf_tool_utility_schedule(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    global_prompt_batch_size: int = 16,
    optimizer_steps: int = 80,
    canary_sample_count: int = 128,
) -> TGVFToolUtilityScheduleResult:
    """Export the exact sequential prefix consumed by a fresh Policy run."""

    batch_size = _positive_integer(global_prompt_batch_size, "global_prompt_batch_size")
    steps = _positive_integer(optimizer_steps, "optimizer_steps")
    sample_count = batch_size * steps
    if (
        type(canary_sample_count) is not int
        or canary_sample_count <= 0
        or canary_sample_count > sample_count
        or canary_sample_count % batch_size
    ):
        raise TGVFToolUtilityError(
            "canary_sample_count must be a positive whole number of prompt batches"
        )

    dataset = Path(dataset_root).resolve(strict=True)
    manifest_path = dataset / POLICY_T1_MIXED_MANIFEST_FILE
    samples_path = dataset / POLICY_T1_MIXED_SAMPLES_FILE
    manifest = _load_object(manifest_path, field="mixed-v2 manifest")
    if manifest.get("dataset_kind") != POLICY_T1_MIXED_DATASET_KIND:
        raise TGVFToolUtilityError("dataset is not a mixed retained T1 pool")
    shuffle = manifest.get("shuffle")
    samples_descriptor = manifest.get("samples")
    if (
        not isinstance(shuffle, Mapping)
        or shuffle.get("algorithm") != POLICY_T1_MIXED_SHUFFLE_ALGORITHM
        or type(shuffle.get("seed")) is not int
        or not isinstance(samples_descriptor, Mapping)
        or samples_descriptor.get("path") != POLICY_T1_MIXED_SAMPLES_FILE
    ):
        raise TGVFToolUtilityError("mixed-v2 iteration metadata differs")
    shuffle_seed = int(shuffle["seed"])
    dataset_sample_count = _positive_integer(
        samples_descriptor.get("rows"), "samples.rows"
    )
    if sample_count > dataset_sample_count:
        raise TGVFToolUtilityError("training prefix exceeds the mixed-v2 dataset")
    expected_samples_sha256 = _require_sha256(
        samples_descriptor.get("sha256"), "samples.sha256"
    )
    content_sha256 = _require_sha256(manifest.get("content_sha256"), "content_sha256")

    digest = hashlib.sha256()
    schedule_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    previous_key: tuple[str, str] | None = None
    observed_rows = 0
    for row, payload in _jsonl(samples_path, field="mixed-v2 samples"):
        digest.update(payload)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise TGVFToolUtilityError("mixed-v2 sample_id differs")
        key = _shuffle_key(sample_id, shuffle_seed)
        if sample_id in seen or (previous_key is not None and key <= previous_key):
            raise TGVFToolUtilityError(
                "mixed-v2 rows differ from sequential hash order"
            )
        seen.add(sample_id)
        previous_key = key
        if observed_rows < sample_count:
            schedule_rows.append(
                _schedule_projection(
                    row,
                    training_index=observed_rows,
                    global_prompt_batch_size=batch_size,
                    canary_sample_count=canary_sample_count,
                )
            )
        observed_rows += 1
    if observed_rows != dataset_sample_count:
        raise TGVFToolUtilityError("mixed-v2 sample row count differs")
    observed_samples_sha256 = digest.hexdigest()
    if observed_samples_sha256 != expected_samples_sha256:
        raise TGVFToolUtilityError("mixed-v2 samples SHA-256 differs")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    schedule_path = output / SCHEDULE_FILE
    with schedule_path.open("wb") as handle:
        for row in schedule_rows:
            handle.write(_canonical_json_line(row))
    schedule_sha256 = _sha256_bytes(schedule_path.read_bytes())
    manifest_file_sha256 = _sha256_bytes(manifest_path.read_bytes())
    iteration_identity = _iteration_identity(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        samples_sha256=observed_samples_sha256,
        sample_count=dataset_sample_count,
        shuffle_seed=shuffle_seed,
    )
    schedule_manifest_path = output / SCHEDULE_MANIFEST_FILE
    manifest_identity: dict[str, object] = {
        "schema_version": TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA,
        "dataset": {
            "kind": POLICY_T1_MIXED_DATASET_KIND,
            "manifest_file_sha256": manifest_file_sha256,
            "content_sha256": content_sha256,
            "samples_sha256": observed_samples_sha256,
            "sample_count": dataset_sample_count,
            "iteration_identity_sha256": iteration_identity,
            "shuffle_algorithm": POLICY_T1_MIXED_SHUFFLE_ALGORITHM,
            "shuffle_seed": shuffle_seed,
        },
        "schedule": {
            "selection": "sequential-prefix-v1",
            "global_prompt_batch_size": batch_size,
            "optimizer_steps": steps,
            "sample_count": sample_count,
            "canary_sample_count": canary_sample_count,
            "canary_optimizer_steps": canary_sample_count // batch_size,
        },
        "files": {
            "schedule": {
                "path": SCHEDULE_FILE,
                "rows": sample_count,
                "sha256": schedule_sha256,
            }
        },
    }
    manifest_sha256 = _write_manifest(schedule_manifest_path, manifest_identity)
    return TGVFToolUtilityScheduleResult(
        output_root=output,
        schedule_path=schedule_path,
        manifest_path=schedule_manifest_path,
        sample_count=sample_count,
        canary_sample_count=canary_sample_count,
        schedule_sha256=schedule_sha256,
        manifest_sha256=manifest_sha256,
    )


def _validated_schedule(
    schedule_root: Path, *, sample_count: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    manifest_path = schedule_root / SCHEDULE_MANIFEST_FILE
    manifest = _load_object(manifest_path, field="utility schedule manifest")
    manifest_sha256 = _require_sha256(
        manifest.pop("manifest_sha256", None), "schedule manifest_sha256"
    )
    if _sha256_bytes(_canonical_json_bytes(manifest)) != manifest_sha256:
        raise TGVFToolUtilityError("utility schedule manifest identity differs")
    if manifest.get("schema_version") != TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA:
        raise TGVFToolUtilityError("utility schedule schema differs")
    files = manifest.get("files")
    descriptor = files.get("schedule") if isinstance(files, Mapping) else None
    if not isinstance(descriptor, Mapping) or descriptor.get("path") != SCHEDULE_FILE:
        raise TGVFToolUtilityError("utility schedule file descriptor differs")
    declared_rows = _positive_integer(descriptor.get("rows"), "schedule rows")
    selected_rows = declared_rows if sample_count is None else sample_count
    if type(selected_rows) is not int or not 0 < selected_rows <= declared_rows:
        raise TGVFToolUtilityError("sidecar sample_count exceeds the schedule")
    expected_sha256 = _require_sha256(descriptor.get("sha256"), "schedule sha256")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    observed = 0
    for row, payload in _jsonl(schedule_root / SCHEDULE_FILE, field="utility schedule"):
        digest.update(payload)
        if observed < selected_rows:
            if row.get("schema_version") != TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA:
                raise TGVFToolUtilityError("utility schedule row schema differs")
            if row.get("training_index") != observed:
                raise TGVFToolUtilityError("utility schedule position differs")
            rows.append(row)
        observed += 1
    if observed != declared_rows or digest.hexdigest() != expected_sha256:
        raise TGVFToolUtilityError("utility schedule bytes differ")
    return rows, manifest, manifest_sha256


def _load_attempts(
    path: Path,
    *,
    selected_sample_ids: set[str],
    run_id: str,
    run_identity_sha256: str,
    attempts_per_sample: int,
) -> tuple[dict[str, list[dict[str, Any]]], int, str]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    observed = 0
    seen: set[tuple[str, int]] = set()
    for row, payload in _jsonl(path, field="forced-TGVF attempts"):
        digest.update(payload)
        if row.get("schema_version") != TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA:
            raise TGVFToolUtilityError("forced-TGVF attempt schema differs")
        sample_id = row.get("sample_id")
        attempt_index = row.get("attempt_index")
        status = row.get("status")
        correct = row.get("correct")
        if sample_id not in selected_sample_ids:
            raise TGVFToolUtilityError(
                "forced-TGVF attempt is outside selected schedule"
            )
        if (
            type(attempt_index) is not int
            or not 0 <= attempt_index < attempts_per_sample
        ):
            raise TGVFToolUtilityError("forced-TGVF attempt index differs")
        if (sample_id, attempt_index) in seen:
            raise TGVFToolUtilityError("duplicate forced-TGVF attempt")
        if (
            row.get("run_id") != run_id
            or row.get("run_identity_sha256") != run_identity_sha256
        ):
            raise TGVFToolUtilityError("forced-TGVF run identity differs")
        if not isinstance(status, str) or not status:
            raise TGVFToolUtilityError("forced-TGVF attempt status differs")
        if status == "scored" and type(correct) is not bool:
            raise TGVFToolUtilityError("scored forced-TGVF attempt lacks correctness")
        if status != "scored" and correct is not None:
            raise TGVFToolUtilityError("unscored forced-TGVF attempt has correctness")
        seen.add((sample_id, attempt_index))
        grouped[sample_id].append(row)
        observed += 1
    return dict(grouped), observed, digest.hexdigest()


def _utility_label(
    delta: float, *, needed_threshold: float, unnecessary_threshold: float
) -> str:
    if delta >= needed_threshold:
        return "needed"
    if delta <= unnecessary_threshold:
        return "unnecessary"
    return "optional"


def materialize_tgvf_tool_utility_sidecar(
    schedule_root: str | Path,
    attempts_path: str | Path,
    output_root: str | Path,
    *,
    run_id: str,
    run_identity_sha256: str,
    sample_count: int | None = None,
    attempts_per_sample: int = 8,
    needed_threshold: float = 0.25,
    unnecessary_threshold: float = -0.25,
    confidence: float = 0.5,
) -> TGVFToolUtilitySidecarResult:
    """Aggregate complete real attempts into Stage3-style utility labels."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise TGVFToolUtilityError("run_id must be non-empty")
    run_identity = _require_sha256(run_identity_sha256, "run_identity_sha256")
    attempts_per_sample = _positive_integer(attempts_per_sample, "attempts_per_sample")
    if (
        type(needed_threshold) not in (int, float)
        or type(unnecessary_threshold) not in (int, float)
        or not -1 <= unnecessary_threshold < needed_threshold <= 1
    ):
        raise TGVFToolUtilityError(
            "utility thresholds must satisfy -1 <= low < high <= 1"
        )
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise TGVFToolUtilityError("confidence must be between zero and one")

    schedule = Path(schedule_root).resolve(strict=True)
    schedule_rows, schedule_manifest, schedule_manifest_sha256 = _validated_schedule(
        schedule, sample_count=sample_count
    )
    sample_ids = {str(row["sample_id"]) for row in schedule_rows}
    if len(sample_ids) != len(schedule_rows):
        raise TGVFToolUtilityError("utility schedule contains duplicate sample IDs")
    attempts_file = Path(attempts_path).resolve(strict=True)
    grouped, attempt_rows, attempts_sha256 = _load_attempts(
        attempts_file,
        selected_sample_ids=sample_ids,
        run_id=run_id,
        run_identity_sha256=run_identity,
        attempts_per_sample=attempts_per_sample,
    )

    sidecar_rows: list[dict[str, object]] = []
    label_counts = {"needed": 0, "optional": 0, "unnecessary": 0}
    for schedule_row in schedule_rows:
        sample_id = str(schedule_row["sample_id"])
        attempts = sorted(
            grouped.get(sample_id, ()), key=lambda row: row["attempt_index"]
        )
        scored = [row for row in attempts if row["status"] == "scored"]
        if len(attempts) != attempts_per_sample or len(scored) != attempts_per_sample:
            raise TGVFToolUtilityError(
                f"{sample_id} lacks {attempts_per_sample} scored forced-TGVF attempts"
            )
        if [row["attempt_index"] for row in attempts] != list(
            range(attempts_per_sample)
        ):
            raise TGVFToolUtilityError(f"{sample_id} attempt indices are incomplete")
        correct = sum(bool(row["correct"]) for row in scored)
        p_tgvf = correct / attempts_per_sample
        p_full = float(schedule_row["p_full"])
        delta = p_tgvf - p_full
        label = _utility_label(
            delta,
            needed_threshold=float(needed_threshold),
            unnecessary_threshold=float(unnecessary_threshold),
        )
        label_counts[label] += 1
        sidecar_rows.append(
            {
                "schema_version": TGVF_TOOL_UTILITY_SIDECAR_ROW_SCHEMA,
                "training_index": schedule_row["training_index"],
                "optimizer_step": schedule_row["optimizer_step"],
                "prompt_index_in_step": schedule_row["prompt_index_in_step"],
                "sample_id": sample_id,
                "p_full": p_full,
                "p_tgvf": p_tgvf,
                "delta": delta,
                "utility_label": label,
                "confidence": float(confidence),
                "attempt_counts": {
                    "full": schedule_row["full_attempt_counts"],
                    "tgvf": {
                        "correct": correct,
                        "expected": attempts_per_sample,
                        "observed": len(attempts),
                        "scoreable": len(scored),
                    },
                },
                "run_identity": {"run_id": run_id, "sha256": run_identity},
            }
        )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    sidecar_path = output / SIDECAR_FILE
    with sidecar_path.open("wb") as handle:
        for row in sidecar_rows:
            handle.write(_canonical_json_line(row))
    sidecar_sha256 = _sha256_bytes(sidecar_path.read_bytes())
    manifest_path = output / SIDECAR_MANIFEST_FILE
    manifest_identity: dict[str, object] = {
        "schema_version": TGVF_TOOL_UTILITY_SIDECAR_SCHEMA,
        "run_identity": {"run_id": run_id, "sha256": run_identity},
        "schedule": {
            "manifest_sha256": schedule_manifest_sha256,
            "dataset": schedule_manifest["dataset"],
            "sample_count": len(sidecar_rows),
            "selection": "sequential-prefix-v1",
        },
        "labeling": {
            "p_full_attempts": 8,
            "p_tgvf_attempts": attempts_per_sample,
            "needed_threshold": float(needed_threshold),
            "unnecessary_threshold": float(unnecessary_threshold),
            "confidence": float(confidence),
            "label_counts": label_counts,
        },
        "files": {
            "attempts": {
                "path": str(attempts_file),
                "rows": attempt_rows,
                "sha256": attempts_sha256,
            },
            "sidecar": {
                "path": SIDECAR_FILE,
                "rows": len(sidecar_rows),
                "sha256": sidecar_sha256,
            },
        },
    }
    manifest_sha256 = _write_manifest(manifest_path, manifest_identity)
    return TGVFToolUtilitySidecarResult(
        output_root=output,
        sidecar_path=sidecar_path,
        manifest_path=manifest_path,
        sample_count=len(sidecar_rows),
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=manifest_sha256,
        label_counts=label_counts,
    )


def load_tgvf_tool_utility_runtime_binding(
    sidecar_path: str | Path,
    *,
    expected_sidecar_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    expected_dataset_iteration_identity_sha256: str,
) -> TGVFToolUtilityRuntimeBinding:
    """Load a sidecar only after proving bytes, manifest, dataset, and rows.

    Runtime reward code must never accept a label based on a path alone.  This
    loader binds both artifact hashes and the source dataset iteration before
    exposing sample-local label/confidence values.
    """

    raw_sidecar = Path(sidecar_path)
    raw_manifest = Path(manifest_path)
    for path, field in (
        (raw_sidecar, "utility sidecar"),
        (raw_manifest, "utility sidecar manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise TGVFToolUtilityError(f"{field} must be a regular non-symlink file")
    sidecar = raw_sidecar.resolve(strict=True)
    manifest_file = raw_manifest.resolve(strict=True)
    sidecar_sha256 = _require_sha256(expected_sidecar_sha256, "expected_sidecar_sha256")
    manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    dataset_iteration = _require_sha256(
        expected_dataset_iteration_identity_sha256,
        "expected_dataset_iteration_identity_sha256",
    )
    if _sha256_bytes(sidecar.read_bytes()) != sidecar_sha256:
        raise TGVFToolUtilityError("utility sidecar SHA-256 differs")

    manifest = _load_object(manifest_file, field="utility sidecar manifest")
    declared_manifest_sha256 = _require_sha256(
        manifest.pop("manifest_sha256", None), "manifest manifest_sha256"
    )
    computed_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(manifest))
    if (
        declared_manifest_sha256 != computed_manifest_sha256
        or declared_manifest_sha256 != manifest_sha256
    ):
        raise TGVFToolUtilityError("utility sidecar manifest identity differs")
    if manifest.get("schema_version") != TGVF_TOOL_UTILITY_SIDECAR_SCHEMA:
        raise TGVFToolUtilityError("utility sidecar manifest schema differs")
    schedule = manifest.get("schedule")
    files = manifest.get("files")
    descriptor = files.get("sidecar") if isinstance(files, Mapping) else None
    if (
        not isinstance(schedule, Mapping)
        or not isinstance(schedule.get("dataset"), Mapping)
        or schedule["dataset"].get("iteration_identity_sha256") != dataset_iteration
    ):
        raise TGVFToolUtilityError("utility sidecar dataset identity differs")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("path") != sidecar.name
        or descriptor.get("sha256") != sidecar_sha256
    ):
        raise TGVFToolUtilityError("utility sidecar file descriptor differs")
    declared_rows = _positive_integer(descriptor.get("rows"), "sidecar rows")
    if schedule.get("sample_count") != declared_rows:
        raise TGVFToolUtilityError("utility sidecar schedule row count differs")

    labels: dict[str, TGVFToolUtilityLabelBinding] = {}
    label_counts = {"needed": 0, "optional": 0, "unnecessary": 0}
    digest = hashlib.sha256()
    for expected_index, (row, payload) in enumerate(
        _jsonl(sidecar, field="utility sidecar")
    ):
        digest.update(payload)
        if row.get("schema_version") != TGVF_TOOL_UTILITY_SIDECAR_ROW_SCHEMA:
            raise TGVFToolUtilityError("utility sidecar row schema differs")
        sample_id = row.get("sample_id")
        label = row.get("utility_label")
        confidence = row.get("confidence")
        training_index = row.get("training_index")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in labels
            or training_index != expected_index
            or label not in label_counts
            or type(confidence) not in (int, float)
        ):
            raise TGVFToolUtilityError("utility sidecar label row differs")
        bound = TGVFToolUtilityLabelBinding(
            sample_id=sample_id,
            training_index=expected_index,
            utility_label=str(label),
            confidence=float(confidence),
            row_sha256=_sha256_bytes(_canonical_json_bytes(row)),
        )
        labels[sample_id] = bound
        label_counts[bound.utility_label] += 1
    if len(labels) != declared_rows or digest.hexdigest() != sidecar_sha256:
        raise TGVFToolUtilityError("utility sidecar rows or bytes differ")
    labeling = manifest.get("labeling")
    if (
        not isinstance(labeling, Mapping)
        or labeling.get("label_counts") != label_counts
    ):
        raise TGVFToolUtilityError("utility sidecar label counts differ")
    declared_confidence = labeling.get("confidence")
    if type(declared_confidence) not in (int, float) or any(
        label.confidence != float(declared_confidence) for label in labels.values()
    ):
        raise TGVFToolUtilityError("utility sidecar confidence binding differs")
    return TGVFToolUtilityRuntimeBinding(
        sidecar_path=sidecar,
        sidecar_sha256=sidecar_sha256,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha256,
        dataset_iteration_identity_sha256=dataset_iteration,
        labels=MappingProxyType(labels),
    )


__all__ = [
    "SIDECAR_FILE",
    "SIDECAR_MANIFEST_FILE",
    "SCHEDULE_FILE",
    "SCHEDULE_MANIFEST_FILE",
    "TGVF_TOOL_UTILITY_ATTEMPT_SCHEMA",
    "TGVF_TOOL_UTILITY_SCHEDULE_ROW_SCHEMA",
    "TGVF_TOOL_UTILITY_SCHEDULE_SCHEMA",
    "TGVF_TOOL_UTILITY_SIDECAR_ROW_SCHEMA",
    "TGVF_TOOL_UTILITY_SIDECAR_SCHEMA",
    "TGVFToolUtilityError",
    "TGVFToolUtilityScheduleResult",
    "TGVFToolUtilitySidecarResult",
    "TGVFToolUtilityLabelBinding",
    "TGVFToolUtilityRuntimeBinding",
    "load_tgvf_tool_utility_runtime_binding",
    "materialize_tgvf_tool_utility_schedule",
    "materialize_tgvf_tool_utility_sidecar",
]
