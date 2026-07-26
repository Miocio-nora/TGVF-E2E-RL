"""Pinned source adapters for CPU-only Policy RL candidate materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from .policy_selection import (
    POLICY_SELECTION_CANDIDATE_SCHEMA,
    SelectionCandidate,
    canonical_json_line,
)


POLICY_SELECTION_SOURCE_MANIFEST_SCHEMA = "tgvf.policy-selection.source-manifest.v1"
POLICY_SELECTION_REJECTION_SCHEMA = "tgvf.policy-selection.rejection.v1"

VSTAR_DATASET_ID = "craigwu/seal_vqa_data"
VSTAR_REVISION = "72f07263e9dd1dc5812a9ed4d8595f42cce7cf44"
VSTAR_ANNOTATION_FILES = {
    "GQA_data.json": "008303acb00c3b88efe26c10c64d1e3a90f0c670f8e65bf1913440772a0fe516",
    "llava_focus_data.json": "566005caa57887ed8ab54b242018f53dd9248dbbcd5a214d9be0bcebbee1aeeb",
    "spatial_relation_data.json": "6914fa0cd26e1700c0df1bff85d3c5bf9cc2135399e5180f38f80fb0a69c3c37",
    "vaw_attribute_data.json": "5e32861aafbb3597f6f632b303b9bcd23bb40053602e4dbe230ad647c975de88",
}
VSTAR_IMAGE_MANIFEST_SCHEMA = "tgvf.policy-selection.vstar-images.v1"
VSTAR_IMAGE_ARCHIVE_SPECS: dict[str, dict[str, Any]] = {
    "coco2014": {
        "filename": "train2014.zip",
        "expected_bytes": 13_510_573_713,
        "member_prefix": "train2014",
        "source_url": "http://images.cocodataset.org/zips/train2014.zip",
    },
    "coco2017": {
        "filename": "train2017.zip",
        "expected_bytes": 19_336_861_798,
        "member_prefix": "train2017",
        "source_url": "http://images.cocodataset.org/zips/train2017.zip",
    },
    "gqa": {
        "filename": "gqa-images.zip",
        "expected_bytes": 21_817_965_542,
        "member_prefix": "images",
        "source_url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
    },
}
VSTAR_GQA_MIRROR = {
    "dataset_id": "zihuwang/ReGuLaR",
    "revision": "b8215201a5fd854c30135f2f0d432f032e364bfa",
    "filename": "images_gqa.tar.zst",
    "expected_bytes": 8_679_513_198,
    "sha256": "f0b15a85bb66c98cea0c7543e86706e7c92c6129660ad3f5a06879d80543caec",
    "source_url": (
        "https://huggingface.co/datasets/zihuwang/ReGuLaR/resolve/"
        "b8215201a5fd854c30135f2f0d432f032e364bfa/"
        "image_archives/images_gqa.tar.zst"
    ),
}
VSTAR_GQA_OFFICIAL_IMAGE_ROOTS = (
    "https://cs.stanford.edu/people/rak248/VG_100K",
    "https://cs.stanford.edu/people/rak248/VG_100K_2",
)

ARXIVQA_DATASET_ID = "MMInstruction/ArxivQA"
ARXIVQA_REVISION = "85a6dca0e2bdc6f0268ae519be8913f83a83cafd"
ARXIVQA_ANNOTATIONS_SHA256 = (
    "d66cee08f152747ea58ce7851310c97229dcc6103901b3041ea988ad11eb1b96"
)
ARXIVQA_IMAGES_SHA256 = (
    "3c746e5828a768c94c392af4ed7d659fe4b85199da73155a31cf239575f3061b"
)
ARXIVQA_OPTION_TRANSFORM_VERSION = "arxivqa-canonical-options-v2"

_ARXIVQA_MARKDOWN_FIGURE_HEADING = re.compile(
    r"^#{1,6}[ \t]+[^\r\n]*\bfigures?\b[^\r\n]*$", re.IGNORECASE
)
_ARXIVQA_OPTION_LABEL_PREFIX = re.compile(
    r"^(?:\[([A-Z])\]|([A-Z])[.):])[ \t]+(.+)$", re.IGNORECASE
)

THINKLITE_DATASET_ID = "russwang/ThinkLite-VL-70k"
THINKLITE_REVISION = "5c86ea41d624e27e53002af47b8cf4538aa2c88f"
THINKLITE_PARQUET_SHA256 = (
    "237f128bef57faccaa9b3ef36dfb4c01c4e5591240f08aca1734bf81c26dbdeb"
)


class PolicySelectionSourceDependencyError(RuntimeError):
    """A dependency needed only for source materialization is unavailable."""


@dataclass(frozen=True, slots=True)
class VstarSourceQualityQuarantine:
    """One source-pinned V* row that must never become a candidate."""

    quarantine_id: str
    dataset_id: str
    revision: str
    source_file: str
    source_file_sha256: str
    source_row_index: int
    source_row_sha256: str
    sample_id: str
    observed_candidate_sha256: str
    reason: str
    question: str
    ground_truth: str

    def as_record(self) -> dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "source_file": self.source_file,
            "source_file_sha256": self.source_file_sha256,
            "source_row_index": self.source_row_index,
            "source_row_sha256": self.source_row_sha256,
            "sample_id": self.sample_id,
            "observed_candidate_sha256": self.observed_candidate_sha256,
            "reason": self.reason,
            "question": self.question,
            "ground_truth": self.ground_truth,
        }


VSTAR_SOURCE_QUALITY_QUARANTINE_VERSION = "vstar-source-quality-quarantine-v1"
VSTAR_SOURCE_QUALITY_QUARANTINES = (
    VstarSourceQualityQuarantine(
        quarantine_id="vstar-llava-focus-row-18042-truncated-ground-truth-v1",
        dataset_id=VSTAR_DATASET_ID,
        revision=VSTAR_REVISION,
        source_file="llava_focus_data.json",
        source_file_sha256=(
            "566005caa57887ed8ab54b242018f53dd9248dbbcd5a214d9be0bcebbee1aeeb"
        ),
        source_row_index=18_042,
        source_row_sha256=(
            "e063fba7441dfe766e0cb8a581d110fdba8051d8a7f22e9c8bfadee98261ef67"
        ),
        sample_id=(
            "policy-candidate:"
            "067f32d28171933f7c638d955b7e8f8d6532add52602630d8fe7f0c9c950bb9e"
        ),
        observed_candidate_sha256=(
            "bd8222dad80f0a6e576b1f6f48dc37bab2717cc2f65e9c2bd6722bc6d334e1b4"
        ),
        reason="source_ground_truth_truncated",
        question="How many bears are in the image?",
        ground_truth="There are",
    ),
)


@dataclass(frozen=True, slots=True)
class SourceMaterializationResult:
    output_root: Path
    source: str
    source_rows: int
    candidate_rows: int
    rejected_rows: int
    unique_images: int
    candidates_sha256: str
    rejections_sha256: str
    manifest_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "source": self.source,
            "source_rows": self.source_rows,
            "candidate_rows": self.candidate_rows,
            "rejected_rows": self.rejected_rows,
            "unique_images": self.unique_images,
            "candidates_sha256": self.candidates_sha256,
            "rejections_sha256": self.rejections_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class VstarImageMaterializationResult:
    output_root: Path
    image_count: int
    manifest_sha256: str
    content_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "image_count": self.image_count,
            "manifest_sha256": self.manifest_sha256,
            "content_sha256": self.content_sha256,
        }


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc
    return encoded.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path.name} SHA-256 mismatch: {actual} != {expected_sha256}")


def _safe_relative_path(value: Any, *, field_name: str) -> Path:
    path = Path(_required_string(value, field_name=field_name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe relative path")
    return path


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _load_pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise PolicySelectionSourceDependencyError(
            "Pillow is required for source image validation"
        ) from exc
    return Image


def _load_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise PolicySelectionSourceDependencyError(
            "pyarrow is required for ThinkLite parquet materialization"
        ) from exc
    return parquet


def _image_extension(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    raise ValueError("unsupported image byte signature")


def _decoded_size(path: Path) -> tuple[int, int]:
    Image = _load_pillow_image()
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("decoded image has non-positive dimensions")
    return int(width), int(height)


def _decoded_size_bytes(payload: bytes) -> tuple[int, int]:
    from io import BytesIO

    Image = _load_pillow_image()
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("decoded image has non-positive dimensions")
    return int(width), int(height)


def _stable_sample_id(
    *, dataset_id: str, revision: str, source_file: str, source_row_index: int
) -> str:
    identity = {
        "dataset_id": dataset_id,
        "revision": revision,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }
    return f"policy-candidate:{_sha256_bytes(_canonical_json_bytes(identity))}"


def _conversation_value(row: Mapping[str, Any], *, role: str) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, Sequence):
        raise ValueError("conversations must be a sequence")
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    for turn in conversations:
        if isinstance(turn, Mapping) and turn.get("from") in aliases:
            return _required_string(turn.get("value"), field_name=f"{role} turn")
    raise ValueError(f"missing {role} conversation turn")


def _vstar_question(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    raw = _conversation_value(row, role="human")
    lines = [
        line
        for line in raw.splitlines()
        if line.strip() != "<image>"
        and not line.strip().startswith("Additional visual information to focus on:")
    ]
    return _required_string("\n".join(lines), field_name="derived question")


def _vstar_answer(row: Mapping[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return _conversation_value(row, role="gpt")


def _canonicalize_xywh_boxes(
    targets: Any, *, width: int, height: int
) -> tuple[list[list[int]], list[dict[str, Any]], int]:
    if not isinstance(targets, Sequence) or not targets:
        raise ValueError("target_instances must be a non-empty sequence")
    boxes: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    clipped = 0
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"target_instances[{index}] must be a mapping")
        raw_box = target.get("bbox")
        if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
            raise ValueError(f"target_instances[{index}].bbox must contain four values")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in raw_box
        ):
            raise ValueError(f"target_instances[{index}].bbox must be finite numeric")
        x, y, box_width, box_height = (float(value) for value in raw_box)
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"target_instances[{index}].bbox must have positive size")
        unbounded = [
            math.floor(x),
            math.floor(y),
            math.ceil(x + box_width),
            math.ceil(y + box_height),
        ]
        box = [
            max(0, min(width, unbounded[0])),
            max(0, min(height, unbounded[1])),
            max(0, min(width, unbounded[2])),
            max(0, min(height, unbounded[3])),
        ]
        if box != unbounded:
            clipped += 1
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"target_instances[{index}].bbox is empty after clipping")
        boxes.append(box)
        metadata.append(
            {
                "name": target.get("name"),
                "instance_id": target.get("instance_id"),
                "raw_xywh": list(raw_box),
                "canonical_xyxy": box,
            }
        )
    return boxes, metadata, clipped


class _ArtifactWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        source: str,
        source_identity: Mapping[str, Any],
    ) -> None:
        self.output_root = Path(output_root)
        if os.path.lexists(self.output_root):
            raise FileExistsError(f"output root already exists: {self.output_root}")
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_root.name}-", dir=self.output_root.parent
            )
        )
        self.source = source
        self.source_identity = dict(source_identity)
        self.candidates_handle = (self.temporary_root / "candidates.jsonl").open("wb")
        self.rejections_handle = (self.temporary_root / "rejected.jsonl").open("wb")
        self.candidates_digest = hashlib.sha256()
        self.rejections_digest = hashlib.sha256()
        self.source_rows = 0
        self.candidate_rows = 0
        self.rejected_rows = 0
        self.image_hashes: set[str] = set()
        self.statistics: dict[str, int] = {}

    def add_candidate(self, record: Mapping[str, Any]) -> None:
        SelectionCandidate.from_record(record)
        line = canonical_json_line(record)
        self.candidates_handle.write(line)
        self.candidates_digest.update(line)
        self.candidate_rows += 1
        self.image_hashes.add(str(record["image"]["sha256"]))

    def add_rejection(
        self,
        *,
        source_file: str,
        source_row_index: int,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "schema_version": POLICY_SELECTION_REJECTION_SCHEMA,
            "source": self.source,
            "source_file": source_file,
            "source_row_index": source_row_index,
            "reason": reason,
        }
        if metadata is not None:
            record["metadata"] = dict(metadata)
        line = canonical_json_line(record)
        self.rejections_handle.write(line)
        self.rejections_digest.update(line)
        self.rejected_rows += 1

    def finish(self) -> SourceMaterializationResult:
        self.candidates_handle.close()
        self.rejections_handle.close()
        candidates_sha256 = self.candidates_digest.hexdigest()
        rejections_sha256 = self.rejections_digest.hexdigest()
        descriptor = {
            "schema_version": POLICY_SELECTION_SOURCE_MANIFEST_SCHEMA,
            "source": self.source,
            "source_identity": self.source_identity,
            "source_rows": self.source_rows,
            "candidate_rows": self.candidate_rows,
            "rejected_rows": self.rejected_rows,
            "unique_images": len(self.image_hashes),
            "candidates": {
                "path": "candidates.jsonl",
                "sha256": candidates_sha256,
            },
            "rejections": {
                "path": "rejected.jsonl",
                "sha256": rejections_sha256,
            },
            "statistics": dict(sorted(self.statistics.items())),
        }
        content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        (self.temporary_root / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        os.replace(self.temporary_root, self.output_root)
        return SourceMaterializationResult(
            output_root=self.output_root,
            source=self.source,
            source_rows=self.source_rows,
            candidate_rows=self.candidate_rows,
            rejected_rows=self.rejected_rows,
            unique_images=len(self.image_hashes),
            candidates_sha256=candidates_sha256,
            rejections_sha256=rejections_sha256,
            manifest_sha256=manifest_sha256,
        )

    def abort(self) -> None:
        self.candidates_handle.close()
        self.rejections_handle.close()
        shutil.rmtree(self.temporary_root, ignore_errors=True)


def _download_visual_genome_image(filename: str, target: Path) -> tuple[str, int, str]:
    if Path(filename).name != filename:
        raise ValueError(f"unsafe Visual Genome filename: {filename}")
    last_error: BaseException | None = None
    for root in VSTAR_GQA_OFFICIAL_IMAGE_ROOTS:
        url = f"{root}/{filename}"
        for attempt in range(5):
            part = target.with_name(f".{target.name}.download")
            try:
                request = Request(
                    url, headers={"User-Agent": "tgvf-data-materializer/1"}
                )
                digest = hashlib.sha256()
                byte_count = 0
                with (
                    urlopen(request, timeout=120) as response,
                    part.open("xb") as output,
                ):
                    if response.status != 200:
                        raise OSError(
                            f"unexpected HTTP status {response.status}: {url}"
                        )
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                if byte_count <= 0:
                    raise ValueError(f"empty Visual Genome image: {url}")
                with part.open("rb") as handle:
                    if not handle.read(3).startswith(b"\xff\xd8\xff"):
                        raise ValueError(f"non-JPEG Visual Genome payload: {url}")
                os.replace(part, target)
                return digest.hexdigest(), byte_count, root
            except HTTPError as exc:
                part.unlink(missing_ok=True)
                last_error = exc
                if exc.code == 404:
                    break
                if exc.code not in {408, 425, 429} and exc.code < 500:
                    raise
            except (OSError, URLError, ValueError) as exc:
                part.unlink(missing_ok=True)
                last_error = exc
            if attempt < 4:
                time.sleep(min(2**attempt, 8))
    raise FileNotFoundError(
        f"could not resolve {filename} from official Visual Genome roots"
    ) from last_error


def _materialize_gqa_from_mirror(
    *,
    mirror_path: Path,
    required: Mapping[str, Path],
    temporary_root: Path,
) -> tuple[dict[str, Any], int]:
    _verify_file(mirror_path, str(VSTAR_GQA_MIRROR["sha256"]))
    actual_bytes = mirror_path.stat().st_size
    expected_bytes = int(VSTAR_GQA_MIRROR["expected_bytes"])
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{mirror_path.name} byte-size mismatch: {actual_bytes} != {expected_bytes}"
        )

    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(mirror_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for member in listing:
        path = Path(member)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe GQA mirror member: {member}")
        if path.parts and path.parts[:2] != ("images", "gqa"):
            raise ValueError(f"unexpected GQA mirror member: {member}")

    extraction_root = temporary_root / ".gqa-mirror-extraction"
    extraction_root.mkdir()
    subprocess.run(
        ["tar", "--zstd", "-xf", str(mirror_path), "-C", str(extraction_root)],
        check=True,
    )
    mirror_hits = 0
    missing: dict[str, Path] = {}
    for member, relative_image in required.items():
        filename = Path(member).name
        source = extraction_root / "images" / "gqa" / filename
        target = temporary_root / relative_image
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            raise ValueError(f"GQA mirror image is a symlink: {source}")
        if source.is_file():
            os.replace(source, target)
            mirror_hits += 1
        else:
            missing[filename] = target
    shutil.rmtree(extraction_root)

    fallback_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(_download_visual_genome_image, filename, target): filename
            for filename, target in missing.items()
        }
        for future in as_completed(futures):
            filename = futures[future]
            sha256, byte_count, root = future.result()
            fallback_records.append(
                {
                    "filename": filename,
                    "sha256": sha256,
                    "bytes": byte_count,
                    "source_root": root,
                }
            )
    fallback_records.sort(key=lambda value: str(value["filename"]))
    return (
        {
            "filename": mirror_path.name,
            "dataset_id": VSTAR_GQA_MIRROR["dataset_id"],
            "revision": VSTAR_GQA_MIRROR["revision"],
            "source_url": VSTAR_GQA_MIRROR["source_url"],
            "sha256": VSTAR_GQA_MIRROR["sha256"],
            "archive_bytes": actual_bytes,
            "required_images": len(required),
            "mirror_images": mirror_hits,
            "official_fallback_images": len(fallback_records),
            "official_fallback_roots": list(VSTAR_GQA_OFFICIAL_IMAGE_ROOTS),
            "official_fallback_bindings_sha256": _sha256_bytes(
                _canonical_json_bytes(fallback_records)
            ),
            "official_fallback_bytes": sum(
                int(value["bytes"]) for value in fallback_records
            ),
        },
        len(required),
    )


def materialize_vstar_images(
    annotations_root: Path,
    archives_root: Path,
    output_root: Path,
    *,
    gqa_mirror_archive: Path | None = None,
) -> VstarImageMaterializationResult:
    annotations_root = Path(annotations_root)
    archives_root = Path(archives_root)
    output_root = Path(output_root)
    for filename, sha256 in VSTAR_ANNOTATION_FILES.items():
        _verify_file(annotations_root / filename, sha256)
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")

    required: dict[str, dict[str, Path]] = {
        name: {} for name in VSTAR_IMAGE_ARCHIVE_SPECS
    }
    for filename in VSTAR_ANNOTATION_FILES:
        with (annotations_root / filename).open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"{filename} must contain a JSON list")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{filename}[{row_index}] must be a mapping")
            relative_image = _safe_relative_path(
                row.get("image"), field_name=f"{filename}[{row_index}].image"
            )
            if len(relative_image.parts) < 3:
                raise ValueError(f"unexpected V* image path: {relative_image}")
            source = relative_image.parts[0]
            if source not in VSTAR_IMAGE_ARCHIVE_SPECS:
                raise ValueError(f"unsupported V* image source: {source}")
            expected_prefix = VSTAR_IMAGE_ARCHIVE_SPECS[source]["member_prefix"]
            member = Path(*relative_image.parts[1:]).as_posix()
            if relative_image.parts[1] != expected_prefix:
                raise ValueError(f"unexpected {source} archive member: {member}")
            required[source][member] = relative_image

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    archive_records: dict[str, dict[str, Any]] = {}
    image_count = 0
    try:
        for source, spec in sorted(VSTAR_IMAGE_ARCHIVE_SPECS.items()):
            if source == "gqa" and gqa_mirror_archive is not None:
                record, extracted_count = _materialize_gqa_from_mirror(
                    mirror_path=Path(gqa_mirror_archive),
                    required=required[source],
                    temporary_root=temporary_root,
                )
                archive_records[source] = record
                image_count += extracted_count
                continue
            archive_path = archives_root / str(spec["filename"])
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            actual_bytes = archive_path.stat().st_size
            expected_bytes = int(spec["expected_bytes"])
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"{archive_path.name} byte-size mismatch: "
                    f"{actual_bytes} != {expected_bytes}"
                )
            archive_sha256 = _sha256_file(archive_path)
            extracted_bytes = 0
            missing = set(required[source])
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    member = info.filename.rstrip("/")
                    relative_image = required[source].get(member)
                    if relative_image is None:
                        continue
                    mode = info.external_attr >> 16
                    if info.is_dir() or stat.S_ISLNK(mode):
                        raise ValueError(f"unsafe archive member: {info.filename}")
                    target = temporary_root / relative_image
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.open(info) as source_handle,
                        target.open("xb") as output,
                    ):
                        shutil.copyfileobj(
                            source_handle, output, length=8 * 1024 * 1024
                        )
                    if target.stat().st_size != info.file_size:
                        raise ValueError(
                            f"extracted byte-size mismatch: {info.filename}"
                        )
                    extracted_bytes += info.file_size
                    image_count += 1
                    missing.remove(member)
            if missing:
                examples = ", ".join(sorted(missing)[:3])
                raise FileNotFoundError(
                    f"{archive_path.name} is missing {len(missing)} required images: "
                    f"{examples}"
                )
            archive_records[source] = {
                "filename": archive_path.name,
                "source_url": spec["source_url"],
                "sha256": archive_sha256,
                "archive_bytes": actual_bytes,
                "required_images": len(required[source]),
                "extracted_bytes": extracted_bytes,
            }

        descriptor = {
            "schema_version": VSTAR_IMAGE_MANIFEST_SCHEMA,
            "annotation_files": dict(VSTAR_ANNOTATION_FILES),
            "archives": archive_records,
            "image_count": image_count,
            "extraction": (
                "required-members-only-coco-zip-gqa-mirror-official-fallback-v1"
                if gqa_mirror_archive is not None
                else "required-members-only-python-zipfile-v1"
            ),
        }
        content_sha256 = _sha256_bytes(_canonical_json_bytes(descriptor))
        manifest = {**descriptor, "content_sha256": content_sha256}
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        os.replace(temporary_root, output_root)
        return VstarImageMaterializationResult(
            output_root=output_root,
            image_count=image_count,
            manifest_sha256=manifest_sha256,
            content_sha256=content_sha256,
        )
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _vstar_source_quality_quarantine_index(
    quarantines: Sequence[VstarSourceQualityQuarantine],
) -> dict[tuple[str, int], VstarSourceQualityQuarantine]:
    indexed: dict[tuple[str, int], VstarSourceQualityQuarantine] = {}
    quarantine_ids: set[str] = set()
    for index, quarantine in enumerate(quarantines):
        if not isinstance(quarantine, VstarSourceQualityQuarantine):
            raise TypeError(
                f"source_quality_quarantines[{index}] must be a "
                "VstarSourceQualityQuarantine"
            )
        if quarantine.dataset_id != VSTAR_DATASET_ID:
            raise RuntimeError("V* source-quality quarantine dataset identity mismatch")
        if quarantine.revision != VSTAR_REVISION:
            raise RuntimeError("V* source-quality quarantine revision mismatch")
        if quarantine.source_file not in VSTAR_ANNOTATION_FILES:
            raise RuntimeError("V* source-quality quarantine file is not pinned")
        if (
            VSTAR_ANNOTATION_FILES[quarantine.source_file]
            != quarantine.source_file_sha256
        ):
            raise RuntimeError(
                "V* source-quality quarantine annotation SHA-256 mismatch"
            )
        if quarantine.source_row_index < 0:
            raise ValueError(
                "V* source-quality quarantine row index must be non-negative"
            )
        expected_sample_id = _stable_sample_id(
            dataset_id=quarantine.dataset_id,
            revision=quarantine.revision,
            source_file=quarantine.source_file,
            source_row_index=quarantine.source_row_index,
        )
        if quarantine.sample_id != expected_sample_id:
            raise RuntimeError("V* source-quality quarantine sample identity mismatch")
        for field_name, value in (
            ("source_file_sha256", quarantine.source_file_sha256),
            ("source_row_sha256", quarantine.source_row_sha256),
            ("observed_candidate_sha256", quarantine.observed_candidate_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"V* source-quality quarantine {field_name} must be a SHA-256"
                )
        for field_name, value in (
            ("quarantine_id", quarantine.quarantine_id),
            ("reason", quarantine.reason),
            ("question", quarantine.question),
            ("ground_truth", quarantine.ground_truth),
        ):
            _required_string(
                value, field_name=f"V* source-quality quarantine {field_name}"
            )
        location = (quarantine.source_file, quarantine.source_row_index)
        if location in indexed:
            raise ValueError("duplicate V* source-quality quarantine location")
        if quarantine.quarantine_id in quarantine_ids:
            raise ValueError("duplicate V* source-quality quarantine ID")
        indexed[location] = quarantine
        quarantine_ids.add(quarantine.quarantine_id)
    return indexed


def _validate_vstar_source_quality_quarantine(
    quarantine: VstarSourceQualityQuarantine,
    *,
    row: Mapping[str, Any],
    candidate: SelectionCandidate,
) -> None:
    row_sha256 = _sha256_bytes(_canonical_json_bytes(row))
    if row_sha256 != quarantine.source_row_sha256:
        raise RuntimeError("V* source-quality quarantine row SHA-256 mismatch")
    if candidate.sample_id != quarantine.sample_id:
        raise RuntimeError("V* source-quality quarantine candidate sample ID mismatch")
    if candidate.identity_sha256 != quarantine.observed_candidate_sha256:
        raise RuntimeError("V* source-quality quarantine candidate SHA-256 mismatch")
    if candidate.question != quarantine.question:
        raise RuntimeError("V* source-quality quarantine question mismatch")
    if candidate.ground_truth != quarantine.ground_truth:
        raise RuntimeError("V* source-quality quarantine ground truth mismatch")


def materialize_vstar_candidates(
    annotations_root: Path,
    image_root: Path,
    output_root: Path,
) -> SourceMaterializationResult:
    annotations_root = Path(annotations_root)
    image_root = Path(image_root)
    source_quality_quarantines = VSTAR_SOURCE_QUALITY_QUARANTINES
    for filename, sha256 in VSTAR_ANNOTATION_FILES.items():
        _verify_file(annotations_root / filename, sha256)
    image_manifest_path = image_root / "manifest.json"
    if not image_manifest_path.is_file():
        raise FileNotFoundError(image_manifest_path)
    image_manifest_bytes = image_manifest_path.read_bytes()
    image_manifest = json.loads(image_manifest_bytes)
    if not isinstance(image_manifest, Mapping):
        raise ValueError("V* image manifest must be a mapping")
    if image_manifest.get("schema_version") != VSTAR_IMAGE_MANIFEST_SCHEMA:
        raise ValueError("V* image manifest schema is not accepted")
    if image_manifest.get("annotation_files") != VSTAR_ANNOTATION_FILES:
        raise ValueError("V* image manifest annotation identity mismatch")
    declared_content_sha256 = image_manifest.get("content_sha256")
    image_descriptor = {
        key: value for key, value in image_manifest.items() if key != "content_sha256"
    }
    if declared_content_sha256 != _sha256_bytes(
        _canonical_json_bytes(image_descriptor)
    ):
        raise ValueError("V* image manifest content SHA-256 mismatch")
    for directory in ("gqa/images", "coco2014/train2014", "coco2017/train2017"):
        if not (image_root / directory).is_dir():
            raise FileNotFoundError(image_root / directory)

    quarantine_by_location = _vstar_source_quality_quarantine_index(
        source_quality_quarantines
    )
    writer = _ArtifactWriter(
        output_root=output_root,
        source="vstar",
        source_identity={
            "dataset_id": VSTAR_DATASET_ID,
            "revision": VSTAR_REVISION,
            "annotation_files": dict(VSTAR_ANNOTATION_FILES),
            "image_materialization": {
                "schema_version": VSTAR_IMAGE_MANIFEST_SCHEMA,
                "manifest_sha256": _sha256_bytes(image_manifest_bytes),
                "content_sha256": declared_content_sha256,
                "archives": image_manifest.get("archives"),
            },
            "bbox_conversion": "xywh-float-to-clipped-half-open-xyxy-v1",
            "source_quality_quarantine": {
                "version": VSTAR_SOURCE_QUALITY_QUARANTINE_VERSION,
                "entries": [
                    quarantine.as_record() for quarantine in source_quality_quarantines
                ],
            },
        },
    )
    image_cache: dict[Path, tuple[str, int, int]] = {}
    applied_quarantine_ids: set[str] = set()
    try:
        for filename in VSTAR_ANNOTATION_FILES:
            with (annotations_root / filename).open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f"{filename} must contain a JSON list")
            for row_index, row in enumerate(rows):
                writer.source_rows += 1
                try:
                    if not isinstance(row, Mapping):
                        raise ValueError("source row must be a mapping")
                    relative_image = Path(
                        _required_string(row.get("image"), field_name="image")
                    )
                    if relative_image.is_absolute() or ".." in relative_image.parts:
                        raise ValueError("image path must be a safe relative path")
                    image_path = image_root / relative_image
                    metadata = image_cache.get(image_path)
                    if metadata is None:
                        if not image_path.is_file():
                            raise FileNotFoundError(image_path)
                        image_sha256 = _sha256_file(image_path)
                        width, height = _decoded_size(image_path)
                        metadata = (image_sha256, width, height)
                        image_cache[image_path] = metadata
                    image_sha256, width, height = metadata
                    boxes, target_metadata, clipped = _canonicalize_xywh_boxes(
                        row.get("target_instances"), width=width, height=height
                    )
                    writer.statistics["clipped_gt_boxes"] = (
                        writer.statistics.get("clipped_gt_boxes", 0) + clipped
                    )
                    record = {
                        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
                        "sample_id": _stable_sample_id(
                            dataset_id=VSTAR_DATASET_ID,
                            revision=VSTAR_REVISION,
                            source_file=filename,
                            source_row_index=row_index,
                        ),
                        "source": "vstar",
                        "question": _vstar_question(row),
                        "ground_truth": _vstar_answer(row),
                        "image": {
                            "path": str(image_path.resolve()),
                            "sha256": image_sha256,
                            "width": width,
                            "height": height,
                        },
                        "gt_regions": boxes,
                        "provenance": {
                            "dataset_id": VSTAR_DATASET_ID,
                            "revision": VSTAR_REVISION,
                            "source_file": filename,
                            "source_file_sha256": VSTAR_ANNOTATION_FILES[filename],
                            "source_row_index": row_index,
                            "source_image_path": relative_image.as_posix(),
                            "bbox_input_format": "xywh",
                            "bbox_conversion": "floor-origin-ceil-end-clip-v1",
                        },
                        "selection_metadata": {
                            "targets": target_metadata,
                            "source_search": row.get("search"),
                        },
                    }
                    candidate = SelectionCandidate.from_record(record)
                    quarantine = quarantine_by_location.get((filename, row_index))
                    if quarantine is not None:
                        _validate_vstar_source_quality_quarantine(
                            quarantine, row=row, candidate=candidate
                        )
                        writer.add_rejection(
                            source_file=filename,
                            source_row_index=row_index,
                            reason=(f"SourceQualityQuarantine: {quarantine.reason}"),
                            metadata={
                                "quarantine_id": quarantine.quarantine_id,
                                "sample_id": candidate.sample_id,
                                "candidate_sha256": candidate.identity_sha256,
                                "source_row_sha256": quarantine.source_row_sha256,
                            },
                        )
                        applied_quarantine_ids.add(quarantine.quarantine_id)
                        continue
                    writer.add_candidate(record)
                except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                    writer.add_rejection(
                        source_file=filename,
                        source_row_index=row_index,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
            del rows
        expected_quarantine_ids = {
            quarantine.quarantine_id for quarantine in source_quality_quarantines
        }
        if applied_quarantine_ids != expected_quarantine_ids:
            missing = sorted(expected_quarantine_ids - applied_quarantine_ids)
            unexpected = sorted(applied_quarantine_ids - expected_quarantine_ids)
            raise RuntimeError(
                "V* source-quality quarantine application mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        writer.statistics["source_quality_quarantines"] = len(applied_quarantine_ids)
        writer.statistics["image_cache_entries"] = len(image_cache)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


def _arxivqa_question(question: str, options: Any) -> str:
    question = _required_string(question, field_name="question")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("options must be a non-empty sequence")
    normalized = [
        _required_string(option, field_name=f"options[{index}]")
        for index, option in enumerate(options)
    ]
    if len(normalized) < 2:
        raise ValueError("options must contain at least two choices")
    return question + "\nChoices:\n" + "\n".join(normalized)


def _arxivqa_label_source_index(label: Any, *, raw_option_count: int) -> int:
    raw_label = _required_string(label, field_name="label")
    normalized = raw_label.strip().upper()
    match = re.fullmatch(r"\[([A-Z])\](?:[ \t]+.*)?", normalized)
    if match is None:
        match = re.fullmatch(r"([A-Z])", normalized)
    if match is None:
        match = re.fullmatch(r"([A-Z])[.):](?:[ \t]*.*)?", normalized)
    if match is None:
        raise ValueError("label does not begin with an unambiguous choice letter")
    source_index = ord(match.group(1)) - ord("A")
    if source_index >= raw_option_count:
        raise ValueError("label choice is outside the source option range")
    return source_index


def _arxivqa_removed_option_reason(option: str) -> str | None:
    if option == "-":
        return "separator"
    if _ARXIVQA_MARKDOWN_FIGURE_HEADING.fullmatch(option) is not None:
        return "markdown_figure_heading"
    return None


def _canonicalize_arxivqa_options(options: Any, label: Any) -> dict[str, Any]:
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("options must be a non-empty sequence")
    raw_options: list[str] = []
    normalized_options: list[str] = []
    for index, option in enumerate(options):
        if not isinstance(option, str) or not option.strip():
            raise ValueError(f"options[{index}] must be a non-empty string")
        raw_options.append(option)
        normalized_options.append(option.strip())
    retained: list[tuple[int, str]] = []
    removed_options: list[dict[str, Any]] = []
    for source_index, (raw_option, normalized_option) in enumerate(
        zip(raw_options, normalized_options, strict=True)
    ):
        reason = _arxivqa_removed_option_reason(normalized_option)
        if reason is None:
            retained.append((source_index, normalized_option))
        else:
            removed_options.append(
                {
                    "source_index": source_index,
                    "raw_option": raw_option,
                    "reason": reason,
                }
            )
    if len(retained) < 2:
        raise ValueError("cleaned options must contain at least two choices")
    if len(retained) > 26:
        raise ValueError("cleaned options exceed canonical A-Z label capacity")

    prefix_matches: dict[int, re.Match[str]] = {}
    has_mismatched_prefix = False
    for source_index, raw_option in retained:
        prefix_match = _ARXIVQA_OPTION_LABEL_PREFIX.fullmatch(raw_option)
        if prefix_match is None:
            continue
        prefix = (prefix_match.group(1) or prefix_match.group(2)).upper()
        expected = chr(ord("A") + source_index) if source_index < 26 else None
        if prefix == expected:
            prefix_matches[source_index] = prefix_match
        else:
            has_mismatched_prefix = True

    # A single leading scientific abbreviation such as ``B. cereus`` is not
    # enough evidence that the source encoded option labels.  Require at least
    # two position-consistent prefixes and no contradictory prefix in the row.
    strip_prefixes = len(prefix_matches) >= 2 and not has_mismatched_prefix
    option_texts: list[str] = []
    source_option_indices: list[int] = []
    for source_index, raw_option in retained:
        prefix_match = prefix_matches.get(source_index) if strip_prefixes else None
        option_text = prefix_match.group(3).strip() if prefix_match else raw_option
        option_texts.append(_required_string(option_text, field_name="cleaned option"))
        source_option_indices.append(source_index)

    canonical_options = [
        f"{chr(ord('A') + clean_index)}. {option_text}"
        for clean_index, option_text in enumerate(option_texts)
    ]
    label_source_index = _arxivqa_label_source_index(
        label, raw_option_count=len(raw_options)
    )
    source_to_clean = {
        source_index: clean_index
        for clean_index, source_index in enumerate(source_option_indices)
    }
    if label_source_index not in source_to_clean:
        raise ValueError("label points to an option removed by canonical cleanup")
    label_clean_index = source_to_clean[label_source_index]
    return {
        "options": canonical_options,
        "raw_options": raw_options,
        "source_option_indices": source_option_indices,
        "removed_options": removed_options,
        "option_count": len(canonical_options),
        "option_transform_version": ARXIVQA_OPTION_TRANSFORM_VERSION,
        "label": chr(ord("A") + label_clean_index),
        "label_source_index": label_source_index,
        "label_clean_index": label_clean_index,
    }


def _arxivqa_label(label: Any, options: Any) -> str:
    return str(_canonicalize_arxivqa_options(options, label)["label"])


def materialize_arxivqa_candidates(
    annotations_path: Path,
    images_archive_path: Path,
    image_root: Path,
    output_root: Path,
) -> SourceMaterializationResult:
    annotations_path = Path(annotations_path)
    images_archive_path = Path(images_archive_path)
    image_root = Path(image_root)
    _verify_file(annotations_path, ARXIVQA_ANNOTATIONS_SHA256)
    _verify_file(images_archive_path, ARXIVQA_IMAGES_SHA256)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    writer = _ArtifactWriter(
        output_root=output_root,
        source="arxivqa",
        source_identity={
            "dataset_id": ARXIVQA_DATASET_ID,
            "revision": ARXIVQA_REVISION,
            "annotations_sha256": ARXIVQA_ANNOTATIONS_SHA256,
            "images_archive_sha256": ARXIVQA_IMAGES_SHA256,
            "question_render": "question-plus-canonical-choices-v2",
            "option_transform_version": ARXIVQA_OPTION_TRANSFORM_VERSION,
        },
    )
    image_cache: dict[Path, tuple[str, int, int]] = {}
    try:
        with annotations_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                writer.source_rows += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError("source row must be a mapping")
                    relative_image = Path(
                        _required_string(row.get("image"), field_name="image")
                    )
                    if relative_image.is_absolute() or ".." in relative_image.parts:
                        raise ValueError("image path must be a safe relative path")
                    image_path = image_root / relative_image
                    metadata = image_cache.get(image_path)
                    if metadata is None:
                        if not image_path.is_file():
                            raise FileNotFoundError(image_path)
                        image_sha256 = _sha256_file(image_path)
                        width, height = _decoded_size(image_path)
                        metadata = (image_sha256, width, height)
                        image_cache[image_path] = metadata
                    image_sha256, width, height = metadata
                    options = row.get("options")
                    raw_label = row.get("label")
                    option_transform = _canonicalize_arxivqa_options(options, raw_label)
                    removed_options = option_transform["removed_options"]
                    writer.statistics["removed_options"] = writer.statistics.get(
                        "removed_options", 0
                    ) + len(removed_options)
                    record = {
                        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
                        "sample_id": _stable_sample_id(
                            dataset_id=ARXIVQA_DATASET_ID,
                            revision=ARXIVQA_REVISION,
                            source_file=annotations_path.name,
                            source_row_index=row_index,
                        ),
                        "source": "arxivqa",
                        "question": _arxivqa_question(
                            row.get("question"), option_transform["options"]
                        ),
                        "ground_truth": option_transform["label"],
                        "image": {
                            "path": str(image_path.resolve()),
                            "sha256": image_sha256,
                            "width": width,
                            "height": height,
                        },
                        "provenance": {
                            "dataset_id": ARXIVQA_DATASET_ID,
                            "revision": ARXIVQA_REVISION,
                            "source_file": annotations_path.name,
                            "source_file_sha256": ARXIVQA_ANNOTATIONS_SHA256,
                            "source_row_index": row_index,
                            "source_id": row.get("id"),
                            "source_image_path": relative_image.as_posix(),
                        },
                        "selection_metadata": {
                            "options": option_transform["options"],
                            "raw_options": option_transform["raw_options"],
                            "source_option_indices": option_transform[
                                "source_option_indices"
                            ],
                            "removed_options": removed_options,
                            "option_count": option_transform["option_count"],
                            "option_transform_version": option_transform[
                                "option_transform_version"
                            ],
                            "raw_label": raw_label,
                            "label_source_index": option_transform[
                                "label_source_index"
                            ],
                            "label_clean_index": option_transform["label_clean_index"],
                            "rationale": row.get("rationale"),
                        },
                    }
                    writer.add_candidate(record)
                except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                    writer.add_rejection(
                        source_file=annotations_path.name,
                        source_row_index=row_index,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
        writer.statistics["image_cache_entries"] = len(image_cache)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


def materialize_thinklite_candidates(
    parquet_path: Path, output_root: Path
) -> SourceMaterializationResult:
    parquet_path = Path(parquet_path)
    _verify_file(parquet_path, THINKLITE_PARQUET_SHA256)
    parquet = _load_pyarrow_parquet()
    source = parquet.ParquetFile(parquet_path)
    writer = _ArtifactWriter(
        output_root=output_root,
        source="thinklite",
        source_identity={
            "dataset_id": THINKLITE_DATASET_ID,
            "revision": THINKLITE_REVISION,
            "parquet_sha256": THINKLITE_PARQUET_SHA256,
        },
    )
    image_root = writer.temporary_root / "images"
    image_root.mkdir()
    final_image_root = Path(output_root).resolve() / "images"
    try:
        row_index = 0
        for batch in source.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                writer.source_rows += 1
                try:
                    if not isinstance(row, Mapping):
                        raise ValueError("source row must be a mapping")
                    question = _required_string(
                        row.get("problem"), field_name="problem"
                    )
                    if question.startswith("<image>"):
                        question = _required_string(
                            question[len("<image>") :], field_name="problem"
                        )
                    ground_truth = row.get("ground_truth")
                    if not isinstance(ground_truth, str) or not ground_truth.strip():
                        ground_truth = row.get("answer")
                    ground_truth = _required_string(
                        ground_truth, field_name="ground_truth"
                    )
                    payload = row.get("image")
                    if isinstance(payload, memoryview):
                        payload = payload.tobytes()
                    if not isinstance(payload, bytes) or not payload:
                        raise ValueError("image must contain non-empty bytes")
                    image_sha256 = _sha256_bytes(payload)
                    extension = _image_extension(payload)
                    width, height = _decoded_size_bytes(payload)
                    image_name = image_sha256 + extension
                    image_path = image_root / image_name
                    if not image_path.exists():
                        image_path.write_bytes(payload)
                    record = {
                        "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
                        "sample_id": _stable_sample_id(
                            dataset_id=THINKLITE_DATASET_ID,
                            revision=THINKLITE_REVISION,
                            source_file=parquet_path.name,
                            source_row_index=row_index,
                        ),
                        "source": "thinklite",
                        "question": question,
                        "ground_truth": ground_truth,
                        "image": {
                            "path": str(final_image_root / image_name),
                            "sha256": image_sha256,
                            "width": width,
                            "height": height,
                        },
                        "provenance": {
                            "dataset_id": THINKLITE_DATASET_ID,
                            "revision": THINKLITE_REVISION,
                            "source_file": parquet_path.name,
                            "source_file_sha256": THINKLITE_PARQUET_SHA256,
                            "source_row_index": row_index,
                            "source_id": row.get("id"),
                        },
                        "selection_metadata": {
                            "choices": row.get("choices"),
                        },
                    }
                    writer.add_candidate(record)
                except (OSError, TypeError, ValueError) as exc:
                    writer.add_rejection(
                        source_file=parquet_path.name,
                        source_row_index=row_index,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                row_index += 1
        writer.statistics["parquet_declared_rows"] = int(source.metadata.num_rows)
        if writer.source_rows != source.metadata.num_rows:
            raise ValueError("ThinkLite parquet row count changed during iteration")
        return writer.finish()
    except BaseException:
        writer.abort()
        raise
