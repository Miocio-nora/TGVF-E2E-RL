"""Lightweight single-image task schema shared by texture benchmark tools.

This intentionally mirrors the JSON fields accepted by ``CoreDevTask`` while
remaining importable without the CUDA/vLLM policy-evaluation dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

from PIL import Image

from .schema import file_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class TextureTask:
    ordinal: int
    dataset: str
    row_number: int
    index: str
    question: str
    image_paths: tuple[str, ...]
    sample_id: str
    answer: str
    options: tuple[tuple[str, str], ...]
    metadata: tuple[tuple[str, str], ...] = ()
    image_sha256s: tuple[str, ...] = ()
    image_dimensions: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_paths", tuple(self.image_paths))
        object.__setattr__(
            self,
            "options",
            tuple(self.options.items())
            if isinstance(self.options, Mapping)
            else tuple(tuple(item) for item in self.options),
        )
        object.__setattr__(
            self,
            "metadata",
            tuple(self.metadata.items())
            if isinstance(self.metadata, Mapping)
            else tuple(tuple(item) for item in self.metadata),
        )
        object.__setattr__(self, "image_sha256s", tuple(self.image_sha256s))
        object.__setattr__(
            self,
            "image_dimensions",
            tuple(tuple(item) for item in self.image_dimensions),
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("texture task ordinal must be non-negative")
        if type(self.row_number) is not int or self.row_number < 0:
            raise ValueError("texture task row_number must be non-negative")
        for name in ("dataset", "index", "sample_id", "question", "answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"texture task {name} must be non-empty")
        if self.index != self.sample_id:
            raise ValueError("texture task index must equal sample_id")
        if len(self.image_paths) != 1:
            raise ValueError("texture benchmark tool protocol requires one image")
        if len(self.image_sha256s) != 1 or len(self.image_dimensions) != 1:
            raise ValueError("texture task requires one bound image identity")
        require_sha256(self.image_sha256s[0], name="texture task image SHA256")
        dimensions = self.image_dimensions[0]
        if len(dimensions) != 2 or any(
            type(value) is not int or value <= 0 for value in dimensions
        ):
            raise ValueError("texture task image dimensions are invalid")
        option_names = tuple(item[0] for item in self.options)
        if (
            len(option_names) < 2
            or len(option_names) != len(set(option_names))
            or any(
                len(item) != 2
                or not all(isinstance(value, str) and value for value in item)
                for item in self.options
            )
            or self.answer not in option_names
        ):
            raise ValueError("texture task options/gold are malformed")
        metadata_names = tuple(item[0] for item in self.metadata)
        if len(metadata_names) != len(set(metadata_names)) or any(
            len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            for item in self.metadata
        ):
            raise ValueError("texture task metadata is malformed")

    @property
    def single_image(self) -> bool:
        return True

    @property
    def bound_sample_id(self) -> str:
        return self.sample_id


def image_file_identity(path: str | Path) -> tuple[str, tuple[int, int]]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise ValueError(f"benchmark image must be an absolute regular file: {source}")
    payload = source.read_bytes()
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            dimensions = (int(opened.width), int(opened.height))
    except (OSError, ValueError) as error:
        raise ValueError(f"benchmark image cannot be decoded: {source}") from error
    return hashlib.sha256(payload).hexdigest(), dimensions


def load_texture_tasks(
    path: str | Path,
    *,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
    verify_images: bool = True,
) -> tuple[TextureTask, ...]:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise ValueError("texture task manifest SHA256 differs")
    try:
        tasks = tuple(
            TextureTask(**json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("texture task manifest is unreadable") from error
    if expected_count is not None and len(tasks) != expected_count:
        raise ValueError("texture task manifest count differs")
    if tuple(task.ordinal for task in tasks) != tuple(range(len(tasks))):
        raise ValueError("texture task ordinals must be contiguous")
    sample_ids = tuple(task.sample_id for task in tasks)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("texture task sample IDs must be unique")
    image_cache: dict[Path, tuple[str, tuple[int, int]]] = {}
    for task in tasks:
        image_path = Path(task.image_paths[0])
        if not image_path.is_absolute() or not image_path.is_file():
            raise ValueError("texture task contains a relative or missing image")
        if verify_images:
            observed = image_cache.get(image_path)
            if observed is None:
                observed = image_file_identity(image_path)
                image_cache[image_path] = observed
            if observed != (task.image_sha256s[0], task.image_dimensions[0]):
                raise ValueError(
                    f"texture task image identity changed: {task.sample_id}"
                )
    return tasks


__all__ = ["TextureTask", "image_file_identity", "load_texture_tasks"]
