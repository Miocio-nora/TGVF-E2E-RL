"""Materialize the pinned CoreDev-2511 membership as VLMEvalKit TSV slices."""

from __future__ import annotations

import ast
import base64
import csv
from hashlib import md5, sha256
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .vlmevalkit import (
    COREDEV_2511,
    COREDEV_2511_MANIFEST_ID,
    VLMEVALKIT_REVIEW_COMMIT,
    CoreDevManifestEntry,
    load_coredev_2511_manifest,
)


COREDEV_ARTIFACT_MANIFEST = "coredev_2511_vlmevalkit_artifacts.json"
COREDEV_LLM_JUDGE_REPOSITORY = "Qwen/Qwen2.5-72B-Instruct"
COREDEV_LLM_JUDGE_MODEL = "Qwen2.5-72B-Instruct"
COREDEV_DATASET_CLASSES = {
    "VStarBench": "ImageMCQDataset",
    "HRBench4K": "HRBenchDataset",
    "BLINK": "ImageMCQDataset",
    "OCRBench_v2": "OCRBench_v2",
    "MMMU_Pro_10c": "MMMUProDataset",
    "MathVista_MINI": "MathVista",
    "MathVerse_MINI": "MathVerse",
}
COREDEV_JUDGE_CONTRACTS = {
    "VStarBench": "qwen2_5_72b_fallback_or_exact_matching",
    "HRBench4K": "qwen2_5_72b_fallback_or_exact_matching",
    "BLINK": "qwen2_5_72b_fallback_or_exact_matching",
    "OCRBench_v2": "none_rule_based",
    "MMMU_Pro_10c": "qwen2_5_72b_fallback_or_exact_matching",
    "MathVista_MINI": "required_qwen2_5_72b_judge",
    "MathVerse_MINI": "required_qwen2_5_72b_judge",
}


def coredev_runtime_class_name(dataset_name: str) -> str:
    if dataset_name not in COREDEV_DATASET_CLASSES:
        raise ValueError(f"unknown CoreDev dataset: {dataset_name}")
    return f"CoreDev2511{dataset_name.replace('_', '')}Slice"


def register_coredev_vlmevalkit_slices(
    dataset_module: Any,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Register data-locator-only subclasses that inherit official scorers."""

    registered = {}
    for artifact in artifacts["slices"]:
        dataset_name = artifact["dataset"]
        class_name = COREDEV_DATASET_CLASSES[dataset_name]
        dataset_class = getattr(dataset_module, class_name)
        dataset_class.DATASET_URL[dataset_name] = artifact["tsv"]
        dataset_class.DATASET_MD5[dataset_name] = artifact["tsv_md5"]
        wrapper_name = coredev_runtime_class_name(dataset_name)

        def init_slice(
            self,
            dataset=dataset_name,
            skip_noimg=True,
            *,
            _base=dataset_class,
        ):
            _base.__init__(self, dataset=dataset, skip_noimg=skip_noimg)

        wrapper_class = type(
            wrapper_name,
            (dataset_class,),
            {"__init__": init_slice, "__module__": dataset_module.__name__},
        )
        setattr(dataset_module, wrapper_name, wrapper_class)
        registered[dataset_name] = wrapper_name
    return registered


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = sha256() if algorithm == "sha256" else md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_list(value: Any) -> list[Any]:
    if _missing(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        for decoder in (json.loads, ast.literal_eval):
            try:
                decoded = decoder(value)
            except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if hasattr(decoded, "tolist"):
                decoded = decoded.tolist()
            if isinstance(decoded, (list, tuple)):
                return list(decoded)
    return [value]


def _selected_rows(path: Path, row_indices: set[int]) -> dict[int, dict[str, Any]]:
    if path.suffix == ".jsonl":
        found = {}
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if row_index in row_indices:
                    found[row_index] = json.loads(line)
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected an array in {path}")
        found = {
            row_index: dict(payload[row_index])
            for row_index in row_indices
            if row_index < len(payload)
        }
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        found = {}
        cursor = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64):
            stop = cursor + batch.num_rows
            offsets = sorted(index - cursor for index in row_indices if cursor <= index < stop)
            for offset in offsets:
                found[cursor + offset] = batch.slice(offset, 1).to_pylist()[0]
            cursor = stop
            if len(found) == len(row_indices):
                break
    else:
        raise ValueError(f"unsupported CoreDev source format: {path}")
    missing = row_indices.difference(found)
    if missing:
        raise IndexError(f"{path} lacks selected rows: {sorted(missing)[:5]}")
    return found


def _image_suffix(path_hint: str | None, payload: bytes) -> str:
    if path_hint:
        suffix = Path(path_hint).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return suffix
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG"):
        return ".png"
    if payload.startswith(b"RIFF"):
        return ".webp"
    return ".png"


def _media_bytes(value: Any, *, source_path: Path) -> tuple[bytes, str | None]:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if isinstance(payload, bytes):
            return payload, value.get("path")
        value = value.get("path")
    if isinstance(value, memoryview):
        return value.tobytes(), None
    if isinstance(value, bytearray):
        return bytes(value), None
    if isinstance(value, bytes):
        return value, None
    if not isinstance(value, str) or not value:
        raise ValueError("selected benchmark row has no image payload")
    if value.startswith(("/9j/", "iVBOR", "UklGR", "R0lGOD")):
        return base64.b64decode(value), None
    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = source_path.parent / image_path
        if not image_path.is_file():
            image_path = source_path.parent / "images" / value
    if not image_path.is_file():
        raise FileNotFoundError(f"selected image does not exist: {image_path}")
    return image_path.read_bytes(), image_path.name


def _write_images(
    values: list[Any],
    *,
    source_path: Path,
    image_dir: Path,
    row_number: int,
) -> str:
    paths = []
    for image_number, value in enumerate(values):
        payload, path_hint = _media_bytes(value, source_path=source_path)
        suffix = _image_suffix(path_hint, payload)
        target = image_dir / f"{row_number:04d}_{image_number}{suffix}"
        target.write_bytes(payload)
        paths.append(target.name)
    if not paths:
        raise ValueError("selected benchmark row contains no usable images")
    return paths[0] if len(paths) == 1 else repr(paths)


def _published_image_paths(value: str, published_dir: Path) -> str:
    names = [str(item) for item in _as_list(value)]
    paths = [str(published_dir / name) for name in names]
    return paths[0] if len(paths) == 1 else repr(paths)


_VSTAR_OPTION = re.compile(r"^\(([A-Z])\)\s*(.+)$")


def _vstar_fields(record: dict[str, Any]) -> dict[str, Any]:
    question_lines: list[str] = []
    options: dict[str, str] = {}
    options_started = False
    for line in str(record["text"]).splitlines():
        match = _VSTAR_OPTION.match(line.strip())
        if match:
            options_started = True
            options[match.group(1)] = match.group(2)
        elif not options_started:
            question_lines.append(line)
    expected = {chr(65 + index) for index in range(len(options))}
    if len(options) < 2 or set(options) != expected:
        raise ValueError("VStar row does not contain contiguous choices from A")
    return {
        "question": "\n".join(question_lines).strip(),
        **options,
        "answer": str(record["label"]).strip(),
        "category": record["category"],
    }


def _common(entry: CoreDevManifestEntry) -> dict[str, Any]:
    return {
        "index": entry.sample_id,
        "sample_id": entry.sample_id,
        "population_id": entry.population_id,
        "source_file": entry.source_file,
        "source_row_index": entry.row_index,
    }


def _transform_row(
    entry: CoreDevManifestEntry,
    record: dict[str, Any],
    *,
    dataset_name: str,
    source_path: Path,
    image_dir: Path,
    row_number: int,
    ocr_reference: list[dict[str, Any]],
) -> dict[str, Any]:
    common = _common(entry)
    if dataset_name == "VStarBench":
        fields = _vstar_fields(record)
        images = [record["image"]]
    elif dataset_name == "HRBench4K":
        fields = {
            key: record[key]
            for key in ("answer", "question", "A", "B", "C", "D", "category", "cycle_category")
        }
        images = [record["image"]]
    elif dataset_name == "BLINK":
        choices = [str(value) for value in _as_list(record["choices"])]
        answer_match = re.search(r"[A-Z]", str(record["answer"]).upper())
        if answer_match is None:
            raise ValueError("BLINK row has no option-letter answer")
        fields = {
            "question": record["question"],
            **{chr(65 + index): choice for index, choice in enumerate(choices)},
            "answer": answer_match.group(0),
            "category": record["sub_task"],
        }
        images = [
            record[key]
            for key in ("image_1", "image_2", "image_3", "image_4")
            if key in record and not _missing(record[key])
        ]
    elif dataset_name == "OCRBench_v2":
        raw_id = int(record["id"])
        reference = ocr_reference[raw_id]
        if str(reference["question"]) != str(record["question"]):
            raise ValueError(f"OCRBench-v2 row {raw_id} differs from its official record")
        fields = {
            "question": reference["question"],
            "answer": repr(list(reference["answers"])),
            "category": reference["type"],
            "eval": reference.get("eval", "without eval"),
            "bbox": repr(reference["bbox"]) if "bbox" in reference else "without bbox",
            "content": repr(reference["content"]) if "content" in reference else "without content",
        }
        images = [record["image"]]
    elif dataset_name == "MMMU_Pro_10c":
        choices = [str(value) for value in _as_list(record["options"])]
        fields = {
            "question": record["question"],
            **{chr(65 + index): choice for index, choice in enumerate(choices)},
            "answer": str(record["answer"]),
            "category": record["subject"],
            "subject": record["subject"],
            "split": "test",
        }
        images = [
            record[key]
            for key in tuple(f"image_{index}" for index in range(1, 8))
            if key in record and not _missing(record[key])
        ]
    elif dataset_name == "MathVista_MINI":
        choices = [str(value) for value in _as_list(record.get("choices"))]
        metadata = dict(record["metadata"])
        answer_option = ""
        if record["question_type"] == "multi_choice":
            answer_option = chr(65 + choices.index(str(record["answer"])))
        fields = {
            "question": record["query"],
            "answer": str(record["answer"]),
            "question_type": record["question_type"],
            "answer_type": record["answer_type"],
            "answer_option": answer_option,
            "choices": repr(choices),
            "task": metadata["task"],
            "skills": repr(_as_list(metadata["skills"])),
            "unit": record.get("unit", ""),
            "precision": record.get("precision", ""),
        }
        images = [record["decoded_image"]]
    elif dataset_name == "MathVerse_MINI":
        fields = {
            "question": record["query_wo"],
            "query_cot": record["query_cot"],
            "question_for_eval": record["question_for_eval"],
            "answer": str(record["answer"]),
            "metadata": json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True),
        }
        images = [record["image"]]
    else:  # pragma: no cover - fixed suite exhaustiveness
        raise ValueError(f"unknown CoreDev dataset: {dataset_name}")

    image_path = _write_images(
        images,
        source_path=source_path,
        image_dir=image_dir,
        row_number=row_number,
    )
    return {**common, **fields, "image_path": image_path}


def _image_tree_hash(image_dir: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in image_dir.iterdir() if item.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


def materialize_coredev_2511(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create an immutable seven-TSV suite and return its artifact manifest."""

    manifest = load_coredev_2511_manifest(manifest_path)
    if output_root.exists():
        return verify_coredev_2511_artifacts(output_root / COREDEV_ARTIFACT_MANIFEST)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        grouped: dict[str, dict[int, dict[str, Any]]] = {}
        for entry in manifest.entries:
            grouped.setdefault(entry.source_file, {})
        for source_file in grouped:
            indices = {
                entry.row_index
                for entry in manifest.entries
                if entry.source_file == source_file
            }
            grouped[source_file] = _selected_rows(benchmark_root / source_file, indices)

        ocr_reference_path = (
            benchmark_root
            / "ocrbench_v2/official_code/OCRBench_v2/pred_folder/internvl2_5_26b.json"
        )
        ocr_reference = json.loads(ocr_reference_path.read_text(encoding="utf-8"))
        if not isinstance(ocr_reference, list) or len(ocr_reference) != 10_000:
            raise ValueError("OCRBench-v2 official record reference drifted")

        import pandas as pd

        slice_artifacts = []
        for spec in COREDEV_2511.slices:
            entries = manifest.by_source[spec.source_id]
            image_dir = temporary / "images" / spec.vlmeval_dataset
            image_dir.mkdir(parents=True)
            rows = []
            for row_number, entry in enumerate(entries):
                rows.append(
                    _transform_row(
                        entry,
                        grouped[entry.source_file][entry.row_index],
                        dataset_name=spec.vlmeval_dataset,
                        source_path=benchmark_root / entry.source_file,
                        image_dir=image_dir,
                        row_number=row_number,
                        ocr_reference=ocr_reference,
                    )
                )
            published_image_dir = output_root / "images" / spec.vlmeval_dataset
            for row in rows:
                row["image_path"] = _published_image_paths(
                    row["image_path"], published_image_dir
                )
            tsv_path = temporary / f"{spec.vlmeval_dataset}.tsv"
            frame = pd.DataFrame(rows)
            frame.to_csv(
                tsv_path,
                sep="\t",
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            sample_digest = sha256(
                ("\n".join(entry.sample_id for entry in entries) + "\n").encode("utf-8")
            ).hexdigest()
            slice_artifacts.append(
                {
                    "source_id": spec.source_id,
                    "population_id": spec.population_id,
                    "dataset": spec.vlmeval_dataset,
                    "dataset_class": COREDEV_DATASET_CLASSES[spec.vlmeval_dataset],
                    "judge_contract": COREDEV_JUDGE_CONTRACTS[spec.vlmeval_dataset],
                    "sample_count": len(rows),
                    "sample_ids_sha256": sample_digest,
                    "tsv": tsv_path.name,
                    "tsv_md5": _hash_file(tsv_path, "md5"),
                    "tsv_sha256": _hash_file(tsv_path),
                    "image_count": len(tuple(image_dir.iterdir())),
                    "image_tree_sha256": _image_tree_hash(image_dir),
                    "columns": list(frame.columns),
                }
            )

        artifact_manifest = {
            "schema_version": 1,
            "identity": "coredev-2511-vlmevalkit-7055d301-v1",
            "membership_manifest_id": COREDEV_2511_MANIFEST_ID,
            "membership_manifest_sha256": manifest.manifest_sha256,
            "membership_source_file_sha256": manifest.source_file_sha256,
            "vlmevalkit_commit": VLMEVALKIT_REVIEW_COMMIT,
            "llm_judge_model": COREDEV_LLM_JUDGE_MODEL,
            "sample_count": len(manifest.entries),
            "slices": slice_artifacts,
        }
        (temporary / COREDEV_ARTIFACT_MANIFEST).write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
        return artifact_manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_coredev_2511_artifacts(identity_path: Path) -> dict[str, Any]:
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("sample_count") != 2511:
        raise ValueError("CoreDev artifact identity header drifted")
    if payload.get("vlmevalkit_commit") != VLMEVALKIT_REVIEW_COMMIT:
        raise ValueError("CoreDev artifact VLMEvalKit commit drifted")
    root = identity_path.parent
    if len(payload.get("slices", ())) != 7:
        raise ValueError("CoreDev artifact suite must contain seven slices")
    for artifact in payload["slices"]:
        tsv_path = root / artifact["tsv"]
        if _hash_file(tsv_path) != artifact["tsv_sha256"]:
            raise ValueError(f"CoreDev TSV SHA256 mismatch: {tsv_path.name}")
        if _hash_file(tsv_path, "md5") != artifact["tsv_md5"]:
            raise ValueError(f"CoreDev TSV MD5 mismatch: {tsv_path.name}")
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != artifact["sample_count"]:
            raise ValueError(f"CoreDev TSV row count mismatch: {tsv_path.name}")
        if list(rows[0]) != artifact["columns"]:
            raise ValueError(f"CoreDev TSV columns mismatch: {tsv_path.name}")
        sample_digest = sha256(
            ("\n".join(row["sample_id"] for row in rows) + "\n").encode("utf-8")
        ).hexdigest()
        if sample_digest != artifact["sample_ids_sha256"]:
            raise ValueError(f"CoreDev TSV sample order mismatch: {tsv_path.name}")
        image_dir = root / "images" / artifact["dataset"]
        if len(tuple(image_dir.iterdir())) != artifact["image_count"]:
            raise ValueError(f"CoreDev image count mismatch: {artifact['dataset']}")
        if _image_tree_hash(image_dir) != artifact["image_tree_sha256"]:
            raise ValueError(f"CoreDev image hash mismatch: {artifact['dataset']}")
    return payload
