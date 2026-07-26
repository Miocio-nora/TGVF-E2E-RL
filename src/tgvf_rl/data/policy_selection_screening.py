"""CPU-only held-out image screening for Policy RL source candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .policy_selection import SelectionCandidate, canonical_json_line


POLICY_SELECTION_HELDOUT_IMAGE_SCHEMA = "tgvf.policy-selection.heldout-image.v1"
POLICY_SELECTION_LEAKAGE_SCHEMA = "tgvf.policy-selection.heldout-leakage.v1"
POLICY_SELECTION_SCREENING_MANIFEST_SCHEMA = (
    "tgvf.policy-selection.screening-manifest.v1"
)


@dataclass(frozen=True, slots=True)
class PolicySelectionScreeningResult:
    output_root: Path
    input_rows: int
    eligible_rows: int
    leakage_rows: int
    heldout_unique_images: int
    manifest_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "input_rows": self.input_rows,
            "eligible_rows": self.eligible_rows,
            "leakage_rows": self.leakage_rows,
            "heldout_unique_images": self.heldout_unique_images,
            "manifest_sha256": self.manifest_sha256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _load_heldout_images(
    tasks_path: Path,
) -> tuple[dict[str, dict[str, set[str]]], int, int]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    task_rows = 0
    image_references = 0
    with tasks_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            task_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"held-out task row {row_index} must be a mapping")
            dataset = _required_string(row.get("dataset"), field_name="dataset")
            image_paths = row.get("image_paths")
            if not isinstance(image_paths, Sequence) or isinstance(
                image_paths, (str, bytes)
            ):
                raise ValueError(f"held-out task row {row_index} has invalid images")
            for value in image_paths:
                image_path = Path(
                    _required_string(value, field_name="held-out image path")
                )
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                by_path[image_path].add(dataset)
                image_references += 1

    by_hash: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"datasets": set(), "paths": set()}
    )
    for image_path, datasets in by_path.items():
        image_sha256 = _sha256_file(image_path)
        by_hash[image_sha256]["datasets"].update(datasets)
        by_hash[image_sha256]["paths"].add(str(image_path))
    return by_hash, task_rows, image_references


def screen_policy_selection_candidates(
    candidates_path: Path, heldout_tasks_path: Path, output_root: Path
) -> PolicySelectionScreeningResult:
    candidates_path = Path(candidates_path)
    heldout_tasks_path = Path(heldout_tasks_path)
    output_root = Path(output_root)
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    if not heldout_tasks_path.is_file():
        raise FileNotFoundError(heldout_tasks_path)
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")

    heldout, task_rows, image_references = _load_heldout_images(heldout_tasks_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    candidates_digest = hashlib.sha256()
    leakage_digest = hashlib.sha256()
    heldout_digest = hashlib.sha256()
    input_rows = 0
    eligible_rows = 0
    leakage_rows = 0
    sample_ids: set[str] = set()
    try:
        with (
            candidates_path.open("r", encoding="utf-8") as source,
            (temporary_root / "candidates.jsonl").open("wb") as eligible,
            (temporary_root / "heldout_leakage.jsonl").open("wb") as leakage,
            (temporary_root / "heldout_images.jsonl").open("wb") as heldout_output,
        ):
            for image_sha256, binding in sorted(heldout.items()):
                line = canonical_json_line(
                    {
                        "schema_version": POLICY_SELECTION_HELDOUT_IMAGE_SCHEMA,
                        "sha256": image_sha256,
                        "datasets": sorted(binding["datasets"]),
                        "paths": sorted(binding["paths"]),
                    }
                )
                heldout_output.write(line)
                heldout_digest.update(line)
            for row_index, line in enumerate(source):
                input_rows += 1
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"candidate row {row_index} must be a mapping")
                candidate = SelectionCandidate.from_record(record)
                if candidate.sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id: {candidate.sample_id}")
                sample_ids.add(candidate.sample_id)
                image_sha256 = str(candidate.image["sha256"])
                binding = heldout.get(image_sha256)
                if binding is None:
                    canonical = canonical_json_line(record)
                    eligible.write(canonical)
                    candidates_digest.update(canonical)
                    eligible_rows += 1
                    continue
                excluded = canonical_json_line(
                    {
                        "schema_version": POLICY_SELECTION_LEAKAGE_SCHEMA,
                        "sample_id": candidate.sample_id,
                        "source": candidate.source.value,
                        "image_sha256": image_sha256,
                        "heldout_datasets": sorted(binding["datasets"]),
                        "heldout_paths": sorted(binding["paths"]),
                    }
                )
                leakage.write(excluded)
                leakage_digest.update(excluded)
                leakage_rows += 1

        descriptor = {
            "schema_version": POLICY_SELECTION_SCREENING_MANIFEST_SCHEMA,
            "input": {
                "candidates_path": str(candidates_path.resolve()),
                "candidates_sha256": _sha256_file(candidates_path),
                "rows": input_rows,
            },
            "heldout": {
                "tasks_path": str(heldout_tasks_path.resolve()),
                "tasks_sha256": _sha256_file(heldout_tasks_path),
                "task_rows": task_rows,
                "image_references": image_references,
                "unique_paths": sum(len(value["paths"]) for value in heldout.values()),
                "unique_image_hashes": len(heldout),
                "images_sha256": heldout_digest.hexdigest(),
                "match_rule": "exact-original-image-sha256-v1",
            },
            "eligible": {
                "path": "candidates.jsonl",
                "rows": eligible_rows,
                "sha256": candidates_digest.hexdigest(),
            },
            "leakage": {
                "path": "heldout_leakage.jsonl",
                "rows": leakage_rows,
                "sha256": leakage_digest.hexdigest(),
            },
        }
        content_sha256 = hashlib.sha256(_canonical_json_bytes(descriptor)).hexdigest()
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        os.replace(temporary_root, output_root)
        return PolicySelectionScreeningResult(
            output_root=output_root,
            input_rows=input_rows,
            eligible_rows=eligible_rows,
            leakage_rows=leakage_rows,
            heldout_unique_images=len(heldout),
            manifest_sha256=manifest_sha256,
        )
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
