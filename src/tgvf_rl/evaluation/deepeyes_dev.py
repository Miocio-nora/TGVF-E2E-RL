"""Materialize the immutable DeepEyes-style external development suite.

The suite intentionally contains only the three high-resolution benchmarks
reported by DeepEyes: V*Bench, HR-Bench-4K, and HR-Bench-8K.  It reuses the
already pinned 191/200 CoreDev slices for the first two datasets and creates a
matching, deterministic 200-row HR-Bench-8K slice.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


DEEPEYES_DEV_SCHEMA = "tgvf-deepeyes-dev591-v1"
DEEPEYES_DEV_ID = "DeepEyesDev591-seed20260625-v1"
DEEPEYES_DEV_SEED = 20260625


@dataclass(frozen=True, slots=True)
class FileIdentity:
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class HRBenchAuthority:
    repository: str
    revision: str
    filename: str
    file: FileIdentity
    git_lfs_oid: str
    xet_hash: str


OFFICIAL_HRBENCH8K = HRBenchAuthority(
    repository="DreamMr/HR-Bench",
    revision="83b9013d6293b85dc507e87199ca52517536939c",
    filename="hr_bench_8k.parquet",
    file=FileIdentity(
        size_bytes=2_826_567_123,
        sha256="d4f41878a0d93afcf2673f547cecb0f23bb07230a0d07ef286bfa2277d9b63e2",
    ),
    git_lfs_oid="d4f41878a0d93afcf2673f547cecb0f23bb07230a0d07ef286bfa2277d9b63e2",
    xet_hash="8e6c4da0610c67441002b6504bb7c649e8716a0f4d753cbdd50c26e5d46b6cb6",
)

PINNED_EXISTING_SLICES: dict[str, tuple[int, str]] = {
    "VStarBench": (
        191,
        "dc3118ddfef156f5ab2a5b586f21f495ba6b2a494cdb26c79b379682842f63d1",
    ),
    "HRBench4K": (
        200,
        "226b51c2ecbcb973f45fa911641475770061ee5b07eb99f76ce25e815efc607c",
    ),
}

_HR_COLUMNS = (
    "index",
    "answer",
    "question",
    "A",
    "B",
    "C",
    "D",
    "category",
    "cycle_category",
    "image",
)
_STRATUM_KEYS = ("category", "cycle_category", "answer")
_EXPECTED_STRATA = {
    (category, str(cycle), answer): 100
    for category in ("cross", "single")
    for cycle, answer in enumerate("ABCD")
}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _slug(value: object) -> str:
    characters = (character if character.isalnum() else "_" for character in str(value))
    return "_".join("".join(characters).split("_")).strip("_")


def hrbench_sample_id(
    *, population_id: str, source_file: str, raw_id: object, row_index: int
) -> str:
    return f"{population_id}/{_slug(source_file)}/{_slug(raw_id)}_{row_index:06d}"


def select_hrbench_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEEPEYES_DEV_SEED,
    per_stratum: int = 25,
    population_id: str = "hr_bench_8k_800",
    source_file: str = "hr_bench_8k/snapshot/hr_bench_8k.parquet",
) -> tuple[int, ...]:
    """Select rows using the exact stable-hash/round-robin CoreDev rule."""

    groups: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
    for row_index, row in enumerate(rows):
        key = tuple(str(row.get(field, "")) for field in _STRATUM_KEYS)
        sample_id = hrbench_sample_id(
            population_id=population_id,
            source_file=source_file,
            raw_id=row.get("index", row_index),
            row_index=row_index,
        )
        stable_key = sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append((stable_key, row_index))
    if set(groups) != set(_EXPECTED_STRATA):
        raise ValueError("HR-Bench category/cycle_category/answer strata differ")
    if any(len(group) != 100 for group in groups.values()):
        raise ValueError("HR-Bench stratum population count differs from 100")
    for group in groups.values():
        group.sort()
    selected: list[int] = []
    ordered_keys = sorted(groups)
    for offset in range(per_stratum):
        for key in ordered_keys:
            selected.append(groups[key][offset][1])
    return tuple(selected)


def official_mcq_prompt(row: Mapping[str, Any]) -> str:
    options = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    return (
        f"Question: {row['question']}\nOptions:\n{options}\n"
        "Please select the correct answer from the options above. \n"
    )


def _image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("benchmark image payload has an unsupported signature")


def _write_bytes_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"artifact identity collision: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_text_exact(path: Path, text: str) -> None:
    _write_bytes_exact(path, text.encode("utf-8"))


def _link_or_copy_exact(source: Path, target: Path) -> None:
    """Create a private exact copy; evaluation images must not share inodes."""

    target.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256_file(source)
    if target.exists():
        if sha256_file(target) != source_digest:
            raise RuntimeError(f"artifact identity collision: {target}")
        return
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    if sha256_file(target) != source_digest:
        raise RuntimeError(f"copied benchmark image differs: {target}")


def image_file_identity(path: str | Path) -> tuple[str, tuple[int, int]]:
    """Return exact file bytes and decoded [width,height] identity."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("benchmark image must be a regular non-symlink file")
    payload = source.read_bytes()
    digest = sha256(payload).hexdigest()
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            dimensions = (int(opened.width), int(opened.height))
    except OSError as error:
        raise ValueError(f"benchmark image cannot be decoded: {source}") from error
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"benchmark image dimensions are invalid: {source}")
    return digest, dimensions


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def _tsv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _materialize_existing_slice(
    *, dataset: str, source_tsv: Path, output_root: Path
) -> tuple[Path, list[dict[str, str]]]:
    expected_count, expected_sha256 = PINNED_EXISTING_SLICES[dataset]
    if sha256_file(source_tsv) != expected_sha256:
        raise ValueError(f"pinned {dataset} TSV SHA256 differs")
    fieldnames, rows = _read_tsv(source_tsv)
    if len(rows) != expected_count:
        raise ValueError(f"pinned {dataset} TSV row count differs")
    if "image_path" not in fieldnames:
        raise ValueError(f"pinned {dataset} TSV has no image_path")
    rewritten: list[dict[str, str]] = []
    for row_number, source in enumerate(rows):
        if source.get("sample_id") != source.get("index"):
            raise ValueError(f"pinned {dataset} sample_id differs from index")
        image_source = Path(source["image_path"])
        if not image_source.is_absolute() or not image_source.is_file():
            raise ValueError(f"pinned {dataset} image is relative or missing")
        target = (
            output_root
            / "images"
            / dataset
            / f"{row_number:04d}{image_source.suffix.lower()}"
        )
        _link_or_copy_exact(image_source, target)
        row = dict(source)
        row["image_path"] = str(target.resolve())
        rewritten.append(row)
    target_tsv = output_root / "datasets" / f"{dataset}.tsv"
    _write_text_exact(target_tsv, _tsv_text(fieldnames, rewritten))
    return target_tsv, rewritten


def _validate_hrbench_source(
    path: Path, *, authority: HRBenchAuthority, expected_rows: int
) -> None:
    if path.stat().st_size != authority.file.size_bytes:
        raise ValueError("HR-Bench-8K source byte count differs")
    if sha256_file(path) != authority.file.sha256:
        raise ValueError("HR-Bench-8K source SHA256 differs")
    if authority.git_lfs_oid != authority.file.sha256:
        raise ValueError("HR-Bench-8K Git LFS object identity is inconsistent")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("HR-Bench-8K source row count differs")
    if tuple(parquet.schema.names) != _HR_COLUMNS:
        raise ValueError("HR-Bench-8K source schema differs")


def _materialize_hrbench8k(
    *,
    source: Path,
    output_root: Path,
    seed: int,
    authority: HRBenchAuthority,
) -> tuple[Path, list[dict[str, Any]], tuple[int, ...]]:
    _validate_hrbench_source(source, authority=authority, expected_rows=800)
    import pyarrow.parquet as pq

    metadata_rows = pq.read_table(
        source, columns=["index", "answer", "category", "cycle_category"]
    ).to_pylist()
    selected_indices = select_hrbench_rows(metadata_rows, seed=seed)
    selected_set = set(selected_indices)
    # The official parquet has one row group, so reading once is both faster
    # and more reliable than reopening the 2.8 GB image column 200 times.
    source_rows = pq.read_table(source).to_pylist()
    if len(source_rows) != 800:
        raise ValueError("HR-Bench-8K decoded row count differs")
    selected_rows: list[dict[str, Any]] = []
    output_fields = (
        "index",
        "sample_id",
        "population_id",
        "source_file",
        "source_row_index",
        "answer",
        "question",
        "A",
        "B",
        "C",
        "D",
        "category",
        "cycle_category",
        "image_path",
    )
    if len(selected_set) != 200:
        raise ValueError("HR-Bench-8K selected row identities are not unique")
    for output_number, row_index in enumerate(selected_indices):
        source_row = source_rows[row_index]
        if source_row["index"] != metadata_rows[row_index]["index"]:
            raise ValueError("HR-Bench-8K metadata/image row alignment differs")
        image_text = source_row["image"]
        if not isinstance(image_text, str):
            raise TypeError("HR-Bench-8K image is not base64 text")
        try:
            payload = base64.b64decode(image_text, validate=True)
        except ValueError as error:
            raise ValueError("HR-Bench-8K image is invalid base64") from error
        suffix = _image_suffix(payload)
        image_path = (
            output_root
            / "images"
            / "HRBench8K"
            / f"{output_number:04d}_source{row_index:06d}{suffix}"
        )
        _write_bytes_exact(image_path, payload)
        sample_id = hrbench_sample_id(
            population_id="hr_bench_8k_800",
            source_file="hr_bench_8k/snapshot/hr_bench_8k.parquet",
            raw_id=source_row["index"],
            row_index=row_index,
        )
        selected_rows.append(
            {
                "index": sample_id,
                "sample_id": sample_id,
                "population_id": "hr_bench_8k_800",
                "source_file": "hr_bench_8k/snapshot/hr_bench_8k.parquet",
                "source_row_index": row_index,
                "answer": source_row["answer"],
                "question": source_row["question"],
                "A": source_row["A"],
                "B": source_row["B"],
                "C": source_row["C"],
                "D": source_row["D"],
                "category": source_row["category"],
                "cycle_category": source_row["cycle_category"],
                "image_path": str(image_path.resolve()),
            }
        )
    target_tsv = output_root / "datasets" / "HRBench8K.tsv"
    _write_text_exact(target_tsv, _tsv_text(output_fields, selected_rows))
    return target_tsv, selected_rows, selected_indices


def _task_rows(
    dataset_rows: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dataset, rows in dataset_rows:
        for row_number, row in enumerate(rows):
            index = str(row["index"])
            image_path = str(Path(str(row["image_path"])).resolve())
            answer = str(row.get("answer", "")).strip().upper()
            options = tuple(
                (letter, str(row.get(letter, "")))
                for letter in "ABCD"
                if str(row.get(letter, ""))
            )
            # VStar includes binary questions while HR-Bench uses four-way
            # MCQ. Preserve the official prompt's blank C/D lines, but retain
            # only real choices in the machine-readable scoring identity.
            if answer not in dict(options) or not 2 <= len(options) <= 4:
                raise ValueError(f"{dataset} row {index} has invalid MCQ gold/options")
            image_sha256, image_dimensions = image_file_identity(image_path)
            tasks.append(
                {
                    "answer": answer,
                    "dataset": dataset,
                    "image_dimensions": [list(image_dimensions)],
                    "image_paths": [image_path],
                    "image_sha256s": [image_sha256],
                    "index": index,
                    "metadata": [
                        [field, str(row[field])]
                        for field in ("category", "cycle_category")
                        if field in row and row[field] is not None
                    ],
                    "options": [list(item) for item in options],
                    "ordinal": len(tasks),
                    "question": official_mcq_prompt(row),
                    "row_number": row_number,
                    "sample_id": index,
                }
            )
    return tasks


def _file_inventory(root: Path, *, exclude: set[Path]) -> list[dict[str, Any]]:
    inventory = []
    excluded = {path.resolve() for path in exclude}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded or ".cache" in path.parts:
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def materialize_deepeyes_dev591(
    *,
    vstar_tsv: str | Path,
    hrbench4k_tsv: str | Path,
    hrbench8k_parquet: str | Path,
    output_root: str | Path,
    authority: HRBenchAuthority = OFFICIAL_HRBENCH8K,
    seed: int = DEEPEYES_DEV_SEED,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(hrbench8k_parquet).resolve()
    vstar_path, vstar_rows = _materialize_existing_slice(
        dataset="VStarBench", source_tsv=Path(vstar_tsv).resolve(), output_root=output
    )
    hr4k_path, hr4k_rows = _materialize_existing_slice(
        dataset="HRBench4K",
        source_tsv=Path(hrbench4k_tsv).resolve(),
        output_root=output,
    )
    hr8k_path, hr8k_rows, selected_indices = _materialize_hrbench8k(
        source=source, output_root=output, seed=seed, authority=authority
    )
    tasks = _task_rows(
        (
            ("VStarBench", vstar_rows),
            ("HRBench4K", hr4k_rows),
            ("HRBench8K", hr8k_rows),
        )
    )
    if len(tasks) != 591 or any(task["sample_id"] != task["index"] for task in tasks):
        raise RuntimeError("DeepEyesDev591 task identity differs")
    task_path = output / "tasks" / "deepeyes-dev-591.jsonl"
    _write_text_exact(
        task_path,
        "".join(
            json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n"
            for task in tasks
        ),
    )
    manifest_path = output / "manifest.json"
    inventory = _file_inventory(output, exclude={manifest_path})
    stratum_counts: dict[str, int] = {}
    for row in hr8k_rows:
        key = "|".join(str(row[field]) for field in _STRATUM_KEYS)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    manifest: dict[str, Any] = {
        "schema_version": DEEPEYES_DEV_SCHEMA,
        "identity": DEEPEYES_DEV_ID,
        "seed": seed,
        "sample_count": len(tasks),
        "single_image_count": sum(len(task["image_paths"]) == 1 for task in tasks),
        "dataset_counts": {
            "VStarBench": len(vstar_rows),
            "HRBench4K": len(hr4k_rows),
            "HRBench8K": len(hr8k_rows),
        },
        "official_prompt_contract": "VLMEvalKit ImageMCQDataset.build_prompt",
        "task_manifest": {
            "path": task_path.relative_to(output).as_posix(),
            "size_bytes": task_path.stat().st_size,
            "sha256": sha256_file(task_path),
        },
        "datasets": {
            "VStarBench": {
                "path": vstar_path.relative_to(output).as_posix(),
                "sha256": sha256_file(vstar_path),
                "source_tsv_sha256": PINNED_EXISTING_SLICES["VStarBench"][1],
            },
            "HRBench4K": {
                "path": hr4k_path.relative_to(output).as_posix(),
                "sha256": sha256_file(hr4k_path),
                "source_tsv_sha256": PINNED_EXISTING_SLICES["HRBench4K"][1],
            },
            "HRBench8K": {
                "path": hr8k_path.relative_to(output).as_posix(),
                "sha256": sha256_file(hr8k_path),
                "selection_rule": "stable_sha256_seeded_per_stratum_round_robin_v1",
                "stratification_keys": list(_STRATUM_KEYS),
                "selected_source_row_indices": list(selected_indices),
                "selected_source_row_indices_sha256": _canonical_sha256(
                    selected_indices
                ),
                "counts_by_stratum": dict(sorted(stratum_counts.items())),
            },
        },
        "hrbench8k_authority": {
            "repository": authority.repository,
            "revision": authority.revision,
            "filename": authority.filename,
            "resolve_url": (
                f"https://huggingface.co/datasets/{authority.repository}/resolve/"
                f"{authority.revision}/{authority.filename}"
            ),
            "source_path": str(source),
            "size_bytes": authority.file.size_bytes,
            "sha256": authority.file.sha256,
            "git_lfs_oid_sha256": authority.git_lfs_oid,
            "xet_hash": authority.xet_hash,
        },
        "files": inventory,
        "recursive_files_sha256": _canonical_sha256(inventory),
    }
    _write_text_exact(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


__all__ = [
    "DEEPEYES_DEV_ID",
    "DEEPEYES_DEV_SCHEMA",
    "DEEPEYES_DEV_SEED",
    "FileIdentity",
    "HRBenchAuthority",
    "OFFICIAL_HRBENCH8K",
    "hrbench_sample_id",
    "materialize_deepeyes_dev591",
    "official_mcq_prompt",
    "select_hrbench_rows",
    "sha256_file",
]
