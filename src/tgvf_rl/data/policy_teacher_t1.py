"""Immutable candidate preparation for the independent Stage-1 teacher T1 run."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_CANDIDATE_SCHEMA,
    SelectionCandidate,
    canonical_json_line,
)


TEACHER_T1_CANDIDATE_MANIFEST_SCHEMA = (
    "tgvf.policy-selection.teacher-t1-candidates-manifest.v1"
)
TEACHER_T1_EXCLUSION_SCHEMA = "tgvf.policy-selection.teacher-t1-exclusion.v1"
TEACHER_T1_SELECTION_ALGORITHM = "teacher-train-source-uid-full-v1"
TEACHER_T1_EXPECTED_TRAIN_SHA256 = (
    "c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c"
)
TEACHER_T1_EXPECTED_TEST_SHA256 = (
    "de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d"
)
TEACHER_T1_EXPECTED_CANDIDATES = 39_584
TEACHER_T1_EXPECTED_TRAIN_ROWS = 39_998
TEACHER_T1_EXPECTED_TEST_ROWS = 867

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MCQ_LABEL = re.compile(r"^\s*([A-Z])[.)](?:\s+|$)")
_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}")


@dataclass(frozen=True, slots=True)
class TeacherT1CandidateResult:
    output_root: Path
    input_rows: int
    unique_source_uids: int
    candidate_rows: int
    exclusion_rows: int
    unique_images: int
    candidates_sha256: str
    manifest_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "input_rows": self.input_rows,
            "unique_source_uids": self.unique_source_uids,
            "candidate_rows": self.candidate_rows,
            "exclusion_rows": self.exclusion_rows,
            "unique_images": self.unique_images,
            "candidates_sha256": self.candidates_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teacher {field} must be a non-empty string")
    return value.strip()


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line is forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            value["__source_row_index"] = line_number - 1
            records.append(value)
    return tuple(records)


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _normalized_choice_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _encoded_image_identity(
    raw_path: Any,
    *,
    cache: dict[Path, tuple[str, int, int]],
) -> tuple[Path, str, int, int]:
    path = Path(_required_string(raw_path, field="image"))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"teacher image must be an absolute regular file: {path}")
    cached = cache.get(path)
    if cached is None:
        from PIL import Image

        image_sha256 = _sha256_file(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"teacher image has invalid dimensions: {path}")
        cached = (image_sha256, width, height)
        cache[path] = cached
    return path, cached[0], cached[1], cached[2]


def _teacher_test_hashes(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_cache: dict[Path, tuple[str, int, int]],
) -> set[str]:
    return {
        _encoded_image_identity(row.get("image"), cache=image_cache)[1] for row in rows
    }


def _coredev_hashes(tasks_path: Path) -> tuple[set[str], int, int]:
    hashes: set[str] = set()
    paths: set[Path] = set()
    task_rows = 0
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"CoreDev row {line_number} must be an object")
            images = value.get("image_paths")
            if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
                raise ValueError(f"CoreDev row {line_number} image_paths is invalid")
            task_rows += 1
            for raw_path in images:
                path = Path(_required_string(raw_path, field="CoreDev image path"))
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.add(path)
    for path in paths:
        hashes.add(_sha256_file(path))
    return hashes, task_rows, len(paths)


def _group_train_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_required_string(row.get("source_uid"), field="source_uid")].append(row)
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    stable_fields = (
        "image",
        "question",
        "choices",
        "answer",
        "short_answer",
        "answer_format",
        "source_dataset",
        "stable_image_uid",
    )
    for source_uid, members in grouped.items():
        first = members[0]
        for member in members[1:]:
            if any(member.get(field) != first.get(field) for field in stable_fields):
                raise ValueError(
                    f"teacher focus rows disagree inside source_uid {source_uid!r}"
                )
        result[source_uid] = tuple(
            sorted(
                members,
                key=lambda row: (
                    int(row.get("focus_step_index", 0)),
                    str(row.get("uid", "")),
                ),
            )
        )
    return result


def _canonical_task(
    row: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    question = _required_string(row.get("question"), field="question")
    answer_format = _required_string(row.get("answer_format"), field="answer_format")
    answer = _required_string(row.get("answer"), field="answer")
    short_answer = _required_string(row.get("short_answer"), field="short_answer")
    if answer_format == "multiple_choice":
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, Sequence) or isinstance(
            raw_choices, (str, bytes)
        ):
            raise ValueError("multiple-choice teacher row has invalid choices")
        choices: list[str] = []
        normalized_choice_text: list[str] = []
        for index, raw_choice in enumerate(raw_choices):
            if not isinstance(raw_choice, Mapping):
                raise ValueError("teacher choice must be an object")
            expected_label = chr(ord("A") + index)
            label = _required_string(raw_choice.get("label"), field="choice.label")
            text = _required_string(raw_choice.get("text"), field="choice.text")
            if label != expected_label:
                raise ValueError("teacher choice labels must be contiguous from A")
            choices.append(f"{label}. {text}")
            normalized_choice_text.append(_normalized_choice_text(text))
        if not 2 <= len(choices) <= 26:
            raise ValueError("teacher MCQ must have between 2 and 26 choices")
        if len(set(normalized_choice_text)) != len(normalized_choice_text):
            raise ValueError("teacher MCQ has duplicate normalized choice text")
        match = _MCQ_LABEL.match(answer)
        if match is None:
            raise ValueError("teacher MCQ answer is not one canonical label")
        label = match.group(1)
        if label >= chr(ord("A") + len(choices)):
            raise ValueError("teacher MCQ answer label is outside its choices")
        rendered = f"{question}\nChoices:\n" + "\n".join(choices)
        return (
            rendered,
            label,
            {
                "task_kind": "mcq",
                "answer_format": answer_format,
                "option_count": len(choices),
                "choices": choices,
                "answer_text": short_answer,
            },
        )
    if answer_format != "open":
        raise ValueError(f"unsupported teacher answer_format: {answer_format!r}")
    if _normalized_text(answer) != _normalized_text(short_answer):
        raise ValueError("teacher open answer and short_answer conflict")
    return (
        question,
        short_answer,
        {
            "task_kind": "open",
            "answer_format": answer_format,
        },
    )


def _stable_sample_id(*, train_sha256: str, source_uid: str) -> str:
    identity = {
        "schema_version": "tgvf.policy-selection.teacher-t1-sample-id.v1",
        "train_sha256": train_sha256,
        "source_uid": source_uid,
    }
    return (
        "policy-teacher-candidate:"
        + hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    )


def _image_suffix(path: Path) -> str:
    suffix = path.suffix.casefold()
    return suffix if _SAFE_SUFFIX.fullmatch(suffix) is not None else ".img"


def _link_image(
    *,
    source_path: Path,
    image_sha256: str,
    temporary_root: Path,
    final_root: Path,
    materialized: dict[str, Path],
) -> Path:
    existing = materialized.get(image_sha256)
    if existing is not None:
        return existing
    relative = (
        Path("images") / image_sha256[:2] / (image_sha256 + _image_suffix(source_path))
    )
    temporary = temporary_root / relative
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, temporary)
    except OSError:
        shutil.copyfile(source_path, temporary)
    if _sha256_file(temporary) != image_sha256:
        raise ValueError("materialized teacher image SHA-256 differs")
    final_path = final_root / relative
    materialized[image_sha256] = final_path
    return final_path


def materialize_teacher_t1_candidates(
    train_path: str | Path,
    test_path: str | Path,
    coredev_tasks_path: str | Path,
    output_root: str | Path,
    *,
    expected_train_sha256: str = TEACHER_T1_EXPECTED_TRAIN_SHA256,
    expected_test_sha256: str = TEACHER_T1_EXPECTED_TEST_SHA256,
    expected_train_rows: int = TEACHER_T1_EXPECTED_TRAIN_ROWS,
    expected_test_rows: int = TEACHER_T1_EXPECTED_TEST_ROWS,
    expected_candidates: int | None = TEACHER_T1_EXPECTED_CANDIDATES,
) -> TeacherT1CandidateResult:
    """Publish one full, outcome-independent teacher candidate population."""

    train = Path(train_path)
    test = Path(test_path)
    coredev = Path(coredev_tasks_path)
    root = Path(output_root).resolve()
    if any(not path.is_file() for path in (train, test, coredev)):
        missing = [str(path) for path in (train, test, coredev) if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    try:
        root.relative_to(_REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            "teacher T1 output_root must remain inside this repository"
        ) from error
    if os.path.lexists(root):
        raise FileExistsError(f"teacher T1 candidate root already exists: {root}")
    train_sha256 = _sha256_file(train)
    test_sha256 = _sha256_file(test)
    if train_sha256 != expected_train_sha256:
        raise ValueError("teacher train SHA-256 differs from RP66/RP67")
    if test_sha256 != expected_test_sha256:
        raise ValueError("teacher test SHA-256 differs from RP66/RP67")

    train_rows = _read_jsonl(train)
    test_rows = _read_jsonl(test)
    if len(train_rows) != expected_train_rows or len(test_rows) != expected_test_rows:
        raise ValueError("teacher split row count differs from its frozen identity")
    grouped = _group_train_rows(train_rows)
    image_cache: dict[Path, tuple[str, int, int]] = {}
    test_hashes = _teacher_test_hashes(test_rows, image_cache=image_cache)
    coredev_hash_set, coredev_rows, coredev_paths = _coredev_hashes(coredev)

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    prompt_owners: dict[tuple[str, str], str] = {}
    materialized_images: dict[str, Path] = {}
    candidate_source_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    try:
        for source_uid in sorted(grouped):
            members = grouped[source_uid]
            row = members[0]
            source_path, image_sha256, width, height = _encoded_image_identity(
                row.get("image"), cache=image_cache
            )
            reasons: list[str] = []
            if image_sha256 in coredev_hash_set:
                reasons.append("coredev_exact_image_sha256")
            if image_sha256 in test_hashes:
                reasons.append("teacher_test_exact_image_sha256")
            rendered_question: str | None = None
            ground_truth: str | None = None
            selection_metadata: dict[str, Any] | None = None
            if not reasons:
                try:
                    rendered_question, ground_truth, selection_metadata = (
                        _canonical_task(row)
                    )
                except ValueError as error:
                    reasons.append(str(error))
            if not reasons:
                assert rendered_question is not None
                prompt_key = (image_sha256, rendered_question)
                owner = prompt_owners.get(prompt_key)
                if owner is not None:
                    reasons.append("duplicate_image_and_rendered_question")
                else:
                    prompt_owners[prompt_key] = source_uid
            if reasons:
                for reason in reasons:
                    exclusion_counts[reason] += 1
                exclusions.append(
                    {
                        "schema_version": TEACHER_T1_EXCLUSION_SCHEMA,
                        "source_uid": source_uid,
                        "source_dataset": row.get("source_dataset"),
                        "image_sha256": image_sha256,
                        "reasons": reasons,
                    }
                )
                continue

            assert rendered_question is not None
            assert ground_truth is not None
            assert selection_metadata is not None
            image_path = _link_image(
                source_path=source_path,
                image_sha256=image_sha256,
                temporary_root=temporary_root,
                final_root=root,
                materialized=materialized_images,
            )
            source_dataset = _required_string(
                row.get("source_dataset"), field="source_dataset"
            )
            candidate = {
                "schema_version": POLICY_SELECTION_CANDIDATE_SCHEMA,
                "sample_id": _stable_sample_id(
                    train_sha256=train_sha256, source_uid=source_uid
                ),
                "source": "teacher",
                "question": rendered_question,
                "ground_truth": ground_truth,
                "image": {
                    "path": str(image_path),
                    "sha256": image_sha256,
                    "width": width,
                    "height": height,
                },
                "gt_regions": [],
                "provenance": {
                    "dataset_id": "tgvf_v4_teacher_50k_clean_imend",
                    "source_file": str(train.resolve()),
                    "source_file_sha256": train_sha256,
                    "source_uid": source_uid,
                    "source_uids": [str(member.get("uid")) for member in members],
                    "source_row_indices": [
                        int(member["__source_row_index"]) for member in members
                    ],
                    "source_dataset": source_dataset,
                    "stable_image_uid": row.get("stable_image_uid"),
                    "original_image_path": str(source_path),
                    "selection_program": TEACHER_T1_SELECTION_ALGORITHM,
                },
                "selection_metadata": selection_metadata,
            }
            SelectionCandidate.from_record(candidate)
            candidates.append(candidate)
            candidate_source_counts[source_dataset] += 1

        if expected_candidates is not None and len(candidates) != expected_candidates:
            raise ValueError(
                f"teacher T1 candidate count {len(candidates)} != {expected_candidates}"
            )
        candidate_payload = b"".join(canonical_json_line(row) for row in candidates)
        exclusion_payload = b"".join(canonical_json_line(row) for row in exclusions)
        candidates_sha256 = hashlib.sha256(candidate_payload).hexdigest()
        exclusions_sha256 = hashlib.sha256(exclusion_payload).hexdigest()
        (temporary_root / "candidates.jsonl").write_bytes(candidate_payload)
        (temporary_root / "exclusions.jsonl").write_bytes(exclusion_payload)
        manifest = {
            "schema_version": TEACHER_T1_CANDIDATE_MANIFEST_SCHEMA,
            "selection_algorithm_version": TEACHER_T1_SELECTION_ALGORITHM,
            "selection_is_outcome_independent": True,
            "source": {
                "path": str(train.resolve()),
                "sha256": train_sha256,
                "rows": len(train_rows),
                "unique_source_uids": len(grouped),
            },
            "teacher_test_screen": {
                "path": str(test.resolve()),
                "sha256": test_sha256,
                "rows": len(test_rows),
                "unique_image_hashes": len(test_hashes),
            },
            "coredev_screen": {
                "path": str(coredev.resolve()),
                "sha256": _sha256_file(coredev),
                "rows": coredev_rows,
                "unique_image_paths": coredev_paths,
                "unique_image_hashes": len(coredev_hash_set),
            },
            "deduplication": {
                "focus_rows": "source_uid-collapse-v1",
                "prompt_rows": "exact-image-sha256-plus-rendered-question-first-v1",
            },
            "source_counts": {"teacher": len(candidates)},
            "source_dataset_counts": dict(sorted(candidate_source_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
            "exclusions": {
                "path": str(root / "exclusions.jsonl"),
                "rows": len(exclusions),
                "sha256": exclusions_sha256,
            },
            "images": {
                "addressing": "encoded-file-sha256-v1",
                "unique_files": len(materialized_images),
                "repo_local_regular_files": True,
            },
            "candidates": {
                "path": str(root / "candidates.jsonl"),
                "rows": len(candidates),
                "sha256": candidates_sha256,
            },
            "logical_attempts": len(candidates) * 8,
        }
        descriptor = dict(manifest)
        manifest["content_sha256"] = hashlib.sha256(
            _canonical_json_bytes(descriptor)
        ).hexdigest()
        manifest_payload = _canonical_json_bytes(manifest) + b"\n"
        (temporary_root / "manifest.json").write_bytes(manifest_payload)
        os.replace(temporary_root, root)
        return TeacherT1CandidateResult(
            output_root=root,
            input_rows=len(train_rows),
            unique_source_uids=len(grouped),
            candidate_rows=len(candidates),
            exclusion_rows=len(exclusions),
            unique_images=len(materialized_images),
            candidates_sha256=candidates_sha256,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


__all__ = [
    "TEACHER_T1_CANDIDATE_MANIFEST_SCHEMA",
    "TEACHER_T1_EXPECTED_CANDIDATES",
    "TEACHER_T1_EXPECTED_TEST_SHA256",
    "TEACHER_T1_EXPECTED_TRAIN_SHA256",
    "TEACHER_T1_SELECTION_ALGORITHM",
    "TeacherT1CandidateResult",
    "materialize_teacher_t1_candidates",
]
