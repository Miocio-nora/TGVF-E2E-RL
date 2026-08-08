"""Materialize an image-disjoint held-out Crop grounding probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from PIL import Image


CROP_GROUNDING_PROBE_SCHEMA = "crop-grounding-probe-sample-id-manifest-v1"
CROP_GROUNDING_TASK_SCHEMA = "crop-grounding-policy-task-manifest-v1"
CANDIDATE_MANIFEST_SCHEMA = "tgvf.policy-selection.source-manifest.v1"
CANDIDATE_SCHEMA = "tgvf.policy-selection.candidate.v1"
POLICY_SAMPLE_SCHEMA = "tgvf.policy-t1-mixed-rl.sample.v2"
DEFAULT_STRATA = (
    "GQA_data.json",
    "llava_focus_data.json",
    "spatial_relation_data.json",
    "vaw_attribute_data.json",
)


@dataclass(frozen=True, slots=True)
class ProbeCandidate:
    """The fields needed to materialize one immutable evaluation task."""

    sample_id: str
    question: str
    image_path: str
    image_sha256: str
    image_dimensions: tuple[int, int]
    source_file: str
    source_row_index: int
    rank_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _verify_file(path: Path, expected_sha256: str, *, name: str) -> str:
    expected = _require_sha256(expected_sha256, name=f"{name} expected SHA256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{name} SHA256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _image_identity(path: Path) -> tuple[str, tuple[int, int]]:
    """Hash and decode one regular, non-symlink image from the same FD bytes."""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"probe image is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    except OSError as error:
        raise ValueError(f"probe image is missing or unreadable: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            dimensions = (int(opened.width), int(opened.height))
            opened.verify()
    except OSError as error:
        raise ValueError(f"probe image cannot be decoded: {path}") from error
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"probe image dimensions are invalid: {path}")
    return hashlib.sha256(payload).hexdigest(), dimensions


def _candidate_binding(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Path, str, int, str]:
    manifest_file_sha256 = _verify_file(
        manifest_path,
        expected_manifest_sha256,
        name="candidate manifest",
    )
    manifest = _load_json_object(manifest_path, name="candidate manifest")
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError("candidate manifest schema differs")
    if manifest.get("source") != "vstar":
        raise ValueError("grounding probe requires the VStar/SEAL candidate source")
    candidate_rows = _nonnegative_int(
        manifest.get("candidate_rows"), name="candidate_rows"
    )
    binding = manifest.get("candidates")
    if not isinstance(binding, Mapping):
        raise ValueError("candidate manifest candidates binding is missing")
    relative = binding.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("candidate JSONL path must be safe and relative")
    candidate_path = (manifest_path.parent / relative).resolve()
    try:
        candidate_path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("candidate JSONL escapes its manifest directory") from error
    candidate_sha256 = _require_sha256(
        binding.get("sha256"), name="candidate JSONL SHA256"
    )
    _verify_file(candidate_path, candidate_sha256, name="candidate JSONL")
    return candidate_path, candidate_sha256, candidate_rows, manifest_file_sha256


def _training_exclusions(path: Path) -> tuple[set[str], set[str], int]:
    sample_ids: set[str] = set()
    image_sha256s: set[str] = set()
    row_count = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot open training samples: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            row_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid training sample JSON at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"training sample line {line_number} is not an object")
            if payload.get("schema_version") != POLICY_SAMPLE_SCHEMA:
                raise ValueError(
                    f"training sample line {line_number} has an unsupported schema"
                )
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"training sample line {line_number} has no sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate training sample_id: {sample_id}")
            sample_ids.add(sample_id)
            if payload.get("data_source") != "vstar":
                continue
            image = payload.get("image")
            if not isinstance(image, Mapping):
                raise ValueError(f"VStar training sample {sample_id} has no image")
            image_sha256s.add(
                _require_sha256(image.get("sha256"), name=f"{sample_id} image SHA256")
            )
    if row_count == 0:
        raise ValueError("training samples JSONL is empty")
    return sample_ids, image_sha256s, row_count


def _candidate_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def _load_eligible_candidates(
    path: Path,
    *,
    expected_rows: int,
    excluded_sample_ids: set[str],
    excluded_image_sha256s: set[str],
    seed: int,
    strata: Sequence[str],
) -> tuple[dict[str, list[ProbeCandidate]], int]:
    allowed_strata = set(strata)
    eligible = {stratum: [] for stratum in strata}
    seen_ids: set[str] = set()
    row_count = 0
    image_disjoint_count = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot open candidate JSONL: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            row_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid candidate JSON at line {line_number}"
                ) from error
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != CANDIDATE_SCHEMA
            ):
                raise ValueError(f"candidate line {line_number} schema differs")
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"candidate line {line_number} has no sample_id")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate candidate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            image = payload.get("image")
            provenance = payload.get("provenance")
            question = payload.get("question")
            if not isinstance(image, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError(f"candidate {sample_id} lacks image/provenance")
            image_sha256 = _require_sha256(
                image.get("sha256"), name=f"{sample_id} image SHA256"
            )
            if (
                sample_id in excluded_sample_ids
                or image_sha256 in excluded_image_sha256s
            ):
                continue
            image_disjoint_count += 1
            source_file = provenance.get("source_file")
            if source_file not in allowed_strata:
                continue
            image_path = image.get("path")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(image_path, str)
                or not Path(image_path).is_absolute()
                or not Path(image_path).is_file()
            ):
                raise ValueError(
                    f"eligible candidate {sample_id} task fields are invalid"
                )
            expected_dimensions = (
                _positive_int(image.get("width"), name=f"{sample_id} image width"),
                _positive_int(image.get("height"), name=f"{sample_id} image height"),
            )
            actual_sha256, actual_dimensions = _image_identity(Path(image_path))
            if actual_sha256 != image_sha256:
                raise ValueError(f"eligible candidate {sample_id} image SHA256 differs")
            if actual_dimensions != expected_dimensions:
                raise ValueError(
                    f"eligible candidate {sample_id} image dimensions differ"
                )
            raw_regions = payload.get("gt_regions")
            if not isinstance(raw_regions, list) or not raw_regions:
                raise ValueError(f"eligible candidate {sample_id} has no GT regions")
            source_row_index = _nonnegative_int(
                provenance.get("source_row_index"),
                name=f"{sample_id} source_row_index",
            )
            eligible[source_file].append(
                ProbeCandidate(
                    sample_id=sample_id,
                    question=question,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    image_dimensions=expected_dimensions,
                    source_file=source_file,
                    source_row_index=source_row_index,
                    rank_sha256=_candidate_rank(seed, sample_id),
                )
            )
    if row_count != expected_rows:
        raise ValueError(
            f"candidate row count mismatch: expected {expected_rows}, observed {row_count}"
        )
    for stratum in strata:
        eligible[stratum].sort(key=lambda item: (item.rank_sha256, item.sample_id))
    return eligible, image_disjoint_count


def _select_probe(
    eligible: Mapping[str, Sequence[ProbeCandidate]],
    *,
    strata: Sequence[str],
    per_stratum: int,
) -> tuple[ProbeCandidate, ...]:
    selected: list[ProbeCandidate] = []
    selected_images: set[str] = set()
    for stratum in strata:
        stratum_selected = 0
        for candidate in eligible[stratum]:
            if candidate.image_sha256 in selected_images:
                continue
            selected.append(candidate)
            selected_images.add(candidate.image_sha256)
            stratum_selected += 1
            if stratum_selected == per_stratum:
                break
        if stratum_selected != per_stratum:
            raise ValueError(
                f"stratum {stratum} has only {stratum_selected} unique held-out images"
            )
    return tuple(selected)


def _write_idempotent(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable output is not a regular file: {path}")
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(f"immutable output differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def materialize_crop_grounding_probe(
    *,
    candidate_manifest_path: str | Path,
    candidate_manifest_sha256: str,
    training_samples_path: str | Path,
    training_samples_sha256: str,
    output_root: str | Path,
    seed: int,
    per_stratum: int,
    strata: Sequence[str] = DEFAULT_STRATA,
) -> dict[str, Any]:
    """Write a frozen task JSONL and its recursively bound probe manifest."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    per_stratum = _positive_int(per_stratum, name="per_stratum")
    strata = tuple(strata)
    if (
        not strata
        or len(set(strata)) != len(strata)
        or any(not isinstance(stratum, str) or not stratum for stratum in strata)
    ):
        raise ValueError("strata must be unique non-empty strings")
    candidate_manifest = Path(candidate_manifest_path).resolve()
    training_samples = Path(training_samples_path).resolve()
    output = Path(output_root).resolve()
    candidate_path, candidate_sha256, candidate_rows, candidate_manifest_file_sha256 = (
        _candidate_binding(
            candidate_manifest,
            expected_manifest_sha256=candidate_manifest_sha256,
        )
    )
    training_file_sha256 = _verify_file(
        training_samples, training_samples_sha256, name="training samples"
    )
    excluded_ids, excluded_images, training_rows = _training_exclusions(
        training_samples
    )
    eligible, image_disjoint_count = _load_eligible_candidates(
        candidate_path,
        expected_rows=candidate_rows,
        excluded_sample_ids=excluded_ids,
        excluded_image_sha256s=excluded_images,
        seed=seed,
        strata=strata,
    )
    selected = _select_probe(
        eligible,
        strata=strata,
        per_stratum=per_stratum,
    )
    task_rows = [
        {
            "ordinal": ordinal,
            "row_number": candidate.source_row_index,
            "dataset": "VStarSEALHeldoutGrounding",
            "index": candidate.sample_id,
            "sample_id": candidate.sample_id,
            "question": candidate.question,
            "image_paths": [candidate.image_path],
            "image_sha256s": [candidate.image_sha256],
            "image_dimensions": [list(candidate.image_dimensions)],
            "metadata": [["source_file", candidate.source_file]],
        }
        for ordinal, candidate in enumerate(selected)
    ]
    task_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in task_rows
    ).encode("utf-8")
    task_path = output / "tasks.jsonl"
    _write_idempotent(task_path, task_bytes)
    task_sha256 = hashlib.sha256(task_bytes).hexdigest()
    ordered_ids = [candidate.sample_id for candidate in selected]
    selection_identity = {
        "seed": seed,
        "rank": "sha256(seed\\0sample_id)-ascending-v1",
        "strata": list(strata),
        "per_stratum": per_stratum,
        "exclude_training_sample_ids": True,
        "exclude_training_image_sha256s": True,
        "unique_probe_images": True,
        "ordered_sample_ids": ordered_ids,
    }
    probe_manifest = {
        "schema_version": CROP_GROUNDING_PROBE_SCHEMA,
        "task_schema_version": CROP_GROUNDING_TASK_SCHEMA,
        "sample_count": len(selected),
        "ordered_sample_ids": ordered_ids,
        "candidate_manifest_file_sha256": candidate_manifest_file_sha256,
        "candidates_jsonl_sha256": candidate_sha256,
        "training_exclusion": {
            "samples_file_sha256": training_file_sha256,
            "sample_rows": training_rows,
            "excluded_sample_id_count": len(excluded_ids),
            "excluded_vstar_image_sha256_count": len(excluded_images),
        },
        "task_manifest": {
            "path": task_path.name,
            "sha256": task_sha256,
            "row_count": len(task_rows),
        },
        "selection": {
            **selection_identity,
            "eligible_image_disjoint_candidate_count": image_disjoint_count,
            "identity_sha256": _canonical_sha256(selection_identity),
        },
    }
    probe_bytes = (
        json.dumps(probe_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    probe_path = output / "probe-manifest.json"
    _write_idempotent(probe_path, probe_bytes)
    return {
        "probe_manifest_path": str(probe_path),
        "probe_manifest_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "task_manifest_path": str(task_path),
        "task_manifest_sha256": task_sha256,
        "sample_count": len(selected),
        "stratum_counts": {stratum: per_stratum for stratum in strata},
    }


__all__ = [
    "CROP_GROUNDING_PROBE_SCHEMA",
    "CROP_GROUNDING_TASK_SCHEMA",
    "DEFAULT_STRATA",
    "ProbeCandidate",
    "file_sha256",
    "materialize_crop_grounding_probe",
]
