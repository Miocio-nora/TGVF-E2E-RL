"""Typed identities shared by all LAS&T and MMAD evaluation arms.

The benchmark assets and the model pipeline are deliberately separate.  A
single task manifest is consumed by every arm; only the exact model/snapshot
and tool protocol are allowed to vary between arms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping


TEXTURE_BENCHMARK_MATRIX_SCHEMA = "tgvf-texture-benchmark-matrix-v1"
TEXTURE_BENCHMARK_RUN_SCHEMA = "tgvf-texture-benchmark-run-v1"
DEFAULT_MIN_PIXELS = 256 * 256
DEFAULT_MAX_PIXELS = 512 * 512
DEEPEYES_OFFICIAL_VISIBLE_CROP_PROTOCOL = "deepeyes_official_visible_native_crop_v1"
_SHA256_CHARS = frozenset("0123456789abcdef")


def canonical_json_sha256(value: object) -> str:
    """Hash JSON using the repository's stable, whitespace-free encoding."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


class PipelineKind(StrEnum):
    ORIGINAL = "original"
    CROP = "crop"
    TGVF = "tgvf"
    TGVF_CROP = "tgvf_crop"

    @property
    def policy_tool_profile(self) -> str | None:
        return {
            PipelineKind.ORIGINAL: None,
            PipelineKind.CROP: "crop_only",
            PipelineKind.TGVF: "tgvf_only",
            PipelineKind.TGVF_CROP: "crop_tgvf",
        }[self]


class PipelineBackend(StrEnum):
    STOCK_QWEN_VLLM = "stock_qwen_vllm"
    POLICY_BENCHMARK = "policy_benchmark"


@dataclass(frozen=True, slots=True)
class VisionPreprocessConfig:
    """A shared Qwen visual pre-processing contract.

    ``max_pixels`` is an area cap.  It is intentionally not a request to
    stretch or pre-render source assets to a square image.
    """

    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    preserve_aspect_ratio: bool = True
    pre_resize_assets: bool = False
    qwen_patch_size: int = 16
    qwen_merge_size: int = 2

    def __post_init__(self) -> None:
        for name in ("min_pixels", "max_pixels", "qwen_patch_size", "qwen_merge_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"vision {name} must be a positive integer")
        if self.min_pixels > self.max_pixels:
            raise ValueError("vision min_pixels cannot exceed max_pixels")
        if self.preserve_aspect_ratio is not True:
            raise ValueError("texture benchmark vision must preserve aspect ratio")
        if self.pre_resize_assets is not False:
            raise ValueError("texture benchmark assets must retain native resolution")

    @property
    def qwen_resize_factor(self) -> int:
        """Spatial smart-resize factor used by the pinned Qwen3-VL processor."""

        return self.qwen_patch_size * self.qwen_merge_size

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class PipelineArm:
    """One exact model/tool arm in the paired four-way comparison."""

    arm_id: str
    kind: PipelineKind
    backend: PipelineBackend
    model_path: Path | None = None
    policy_config_path: Path | None = None
    lora_pointer_path: Path | None = None
    full_model_snapshot_manifest_path: Path | None = None
    full_model_materialization_receipt_path: Path | None = None
    paired_qwen_model_path: Path | None = None
    paired_rp66_pointer_path: Path | None = None
    paired_snapshot_receipt_path: Path | None = None
    expected_optimizer_step: int | None = None
    evaluation_protocol: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", PipelineKind(self.kind))
            object.__setattr__(self, "backend", PipelineBackend(self.backend))
        except ValueError as error:
            raise ValueError(
                "texture benchmark pipeline kind/backend is invalid"
            ) from error
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ValueError("pipeline arm_id must be non-empty")
        for name in (
            "model_path",
            "policy_config_path",
            "lora_pointer_path",
            "full_model_snapshot_manifest_path",
            "full_model_materialization_receipt_path",
            "paired_qwen_model_path",
            "paired_rp66_pointer_path",
            "paired_snapshot_receipt_path",
        ):
            value = getattr(self, name)
            if value is not None:
                path = Path(value)
                if not path.is_absolute():
                    raise ValueError(f"pipeline {name} must be absolute")
                object.__setattr__(self, name, path)
        if self.backend is PipelineBackend.STOCK_QWEN_VLLM:
            if self.kind is not PipelineKind.ORIGINAL:
                raise ValueError("stock Qwen backend is reserved for original arm")
            if self.model_path is None:
                raise ValueError("original arm requires model_path")
            forbidden = (
                self.policy_config_path,
                self.lora_pointer_path,
                self.full_model_snapshot_manifest_path,
                self.full_model_materialization_receipt_path,
                self.paired_qwen_model_path,
                self.paired_rp66_pointer_path,
                self.paired_snapshot_receipt_path,
                self.expected_optimizer_step,
                self.evaluation_protocol,
            )
            if any(value is not None for value in forbidden):
                raise ValueError("original arm cannot carry policy snapshot fields")
            return
        if self.kind is PipelineKind.ORIGINAL:
            raise ValueError("original arm must use stock Qwen backend")
        if self.policy_config_path is None:
            raise ValueError("tool arm requires policy_config_path")
        if (
            type(self.expected_optimizer_step) is not int
            or self.expected_optimizer_step < 0
        ):
            raise ValueError("tool arm requires a non-negative optimizer step")
        lora = self.lora_pointer_path is not None
        full = (
            self.full_model_snapshot_manifest_path is not None
            or self.full_model_materialization_receipt_path is not None
        )
        paired = (
            self.paired_qwen_model_path is not None
            or self.paired_rp66_pointer_path is not None
            or self.paired_snapshot_receipt_path is not None
        )
        if sum((lora, full, paired)) != 1:
            raise ValueError(
                "tool arm must bind exactly one LoRA, full-model, or paired snapshot"
            )
        if full and (
            self.full_model_snapshot_manifest_path is None
            or self.full_model_materialization_receipt_path is None
        ):
            raise ValueError("full-model tool arm requires manifest and receipt")
        if full and (
            self.kind is not PipelineKind.CROP
            or self.evaluation_protocol != DEEPEYES_OFFICIAL_VISIBLE_CROP_PROTOCOL
        ):
            raise ValueError(
                "full-model tool arm requires the crop official-visible protocol"
            )
        if self.kind is PipelineKind.TGVF_CROP and self.evaluation_protocol != (
            "training_run"
        ):
            raise ValueError("atomic crop+TGVF arm requires training_run protocol")
        if paired:
            if (
                self.paired_qwen_model_path is None
                or self.paired_snapshot_receipt_path is None
            ):
                raise ValueError("paired tool arm requires Qwen model and receipt")
            if self.expected_optimizer_step == 0:
                if self.paired_rp66_pointer_path is not None:
                    raise ValueError("paired step zero forbids an RP66 pointer")
            elif self.paired_rp66_pointer_path is None:
                raise ValueError("paired nonzero step requires an RP66 pointer")
            if self.evaluation_protocol != "training_run":
                raise ValueError("paired tool arm requires training_run protocol")
        if (
            not isinstance(self.evaluation_protocol, str)
            or not self.evaluation_protocol
        ):
            raise ValueError("tool arm requires an evaluation_protocol")

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["backend"] = self.backend.value
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True, slots=True)
class TextureBenchmarkMatrix:
    """Exact paired suite input and all four model pipeline identities."""

    matrix_id: str
    task_manifest_path: Path
    task_manifest_sha256: str
    task_count: int
    output_root: Path
    arms: tuple[PipelineArm, ...]
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    vision: VisionPreprocessConfig = VisionPreprocessConfig()
    schema_version: str = TEXTURE_BENCHMARK_MATRIX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TEXTURE_BENCHMARK_MATRIX_SCHEMA:
            raise ValueError("texture benchmark matrix schema differs")
        if not isinstance(self.matrix_id, str) or not self.matrix_id.strip():
            raise ValueError("matrix_id must be non-empty")
        for name in ("task_manifest_path", "output_root"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"matrix {name} must be absolute")
            object.__setattr__(self, name, path)
        require_sha256(self.task_manifest_sha256, name="task manifest SHA256")
        if type(self.task_count) is not int or self.task_count <= 0:
            raise ValueError("matrix task_count must be positive")
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))
        if (
            len(self.gpu_ids) != 4
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in self.gpu_ids)
            or len(set(self.gpu_ids)) != 4
        ):
            raise ValueError(
                "matrix gpu_ids must contain four distinct non-negative IDs"
            )
        if not isinstance(self.vision, VisionPreprocessConfig):
            object.__setattr__(self, "vision", VisionPreprocessConfig(**self.vision))
        normalized_arms = tuple(
            arm if isinstance(arm, PipelineArm) else PipelineArm(**arm)
            for arm in self.arms
        )
        object.__setattr__(self, "arms", normalized_arms)
        if not normalized_arms:
            raise ValueError("matrix requires at least one pipeline arm")
        kinds = tuple(arm.kind for arm in normalized_arms)
        if len(kinds) != len(set(kinds)):
            raise ValueError("matrix permits at most one arm of each pipeline kind")
        ids = tuple(arm.arm_id for arm in normalized_arms)
        if len(ids) != len(set(ids)):
            raise ValueError("matrix pipeline arm IDs must be unique")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "task_manifest_path": str(self.task_manifest_path),
            "task_manifest_sha256": self.task_manifest_sha256,
            "task_count": self.task_count,
            "output_root": str(self.output_root),
            "gpu_ids": list(self.gpu_ids),
            "vision": asdict(self.vision),
            "arms": [arm.identity_payload() for arm in self.arms],
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.identity_payload())

    @property
    def missing_pipeline_kinds(self) -> tuple[PipelineKind, ...]:
        present = {arm.kind for arm in self.arms}
        return tuple(kind for kind in PipelineKind if kind not in present)

    @property
    def complete_four_arm_matrix(self) -> bool:
        return not self.missing_pipeline_kinds

    def require_complete_arms(self) -> None:
        missing = self.missing_pipeline_kinds
        if missing:
            raise ValueError(
                "complete texture comparison requires these pipeline arms: "
                + ", ".join(kind.value for kind in missing)
            )

    def validate_files(self, *, verify_manifest_hash: bool = True) -> None:
        if not self.task_manifest_path.is_file():
            raise FileNotFoundError(self.task_manifest_path)
        if verify_manifest_hash and file_sha256(self.task_manifest_path) != (
            self.task_manifest_sha256
        ):
            raise ValueError("texture benchmark task manifest SHA256 differs")
        for arm in self.arms:
            paths = (
                arm.model_path,
                arm.policy_config_path,
                arm.lora_pointer_path,
                arm.full_model_snapshot_manifest_path,
                arm.full_model_materialization_receipt_path,
                arm.paired_qwen_model_path,
                arm.paired_rp66_pointer_path,
            )
            for path in (item for item in paths if item is not None):
                if not path.exists():
                    raise FileNotFoundError(path)
            if arm.backend is PipelineBackend.POLICY_BENCHMARK:
                assert arm.policy_config_path is not None
                observed = _policy_tool_profile(arm.policy_config_path)
                if observed != arm.kind.policy_tool_profile:
                    raise ValueError(
                        f"pipeline {arm.kind.value} policy tool profile differs: "
                        f"expected {arm.kind.policy_tool_profile}, observed {observed}"
                    )

    def resolved_arm_identities(self) -> tuple[dict[str, object], ...]:
        """Resolve small-file hashes while marking model-tree hashes deferred.

        The stock runner and paired snapshot materializer own the authoritative
        full model-tree hashes.  This diagnostic is therefore not itself a
        complete model identity; it makes that deferral explicit while binding
        every small policy/pointer/receipt input before GPU allocation.
        """

        resolved: list[dict[str, object]] = []
        for arm in self.arms:
            files: dict[str, str] = {}
            directories: dict[str, str] = {}
            for name in (
                "model_path",
                "policy_config_path",
                "lora_pointer_path",
                "full_model_snapshot_manifest_path",
                "full_model_materialization_receipt_path",
                "paired_qwen_model_path",
                "paired_rp66_pointer_path",
            ):
                path = getattr(arm, name)
                if path is None:
                    continue
                if path.is_file():
                    files[name] = file_sha256(path)
                elif path.is_dir():
                    directories[name] = "deferred_to_backend_tree_identity"
                else:
                    raise FileNotFoundError(path)
            content = {
                "arm": arm.identity_payload(),
                "file_sha256s": files,
                "directory_identity": directories,
            }
            resolved.append(
                {
                    **content,
                    "resolved_identity_sha256": canonical_json_sha256(content),
                }
            )
        return tuple(resolved)


def _policy_tool_profile(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"pipeline policy config is unreadable: {path}") from error
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("pipeline policy config lacks a protocol table")
    profile = protocol.get("tool_profile")
    if isinstance(profile, str) and profile:
        return profile
    tool_name = protocol.get("tool_name")
    legacy = {
        "image_zoom_in_tool": "crop_only",
        "tgvf_focus_tool": "tgvf_only",
        "tgvf_crop_tool": "crop_tgvf",
    }
    if isinstance(tool_name, str) and tool_name in legacy:
        return legacy[tool_name]
    raise ValueError("pipeline policy config has no recognized visual-tool profile")


def _mapping(value: object, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be an object")
    return value


def load_texture_benchmark_matrix(path: str | Path) -> TextureBenchmarkMatrix:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("texture benchmark matrix is unreadable") from error
    payload = dict(_mapping(raw, owner="texture benchmark matrix"))
    arms = payload.get("arms")
    if not isinstance(arms, list):
        raise ValueError("texture benchmark matrix arms must be a list")
    payload["arms"] = tuple(
        PipelineArm(**dict(_mapping(item, owner="pipeline arm"))) for item in arms
    )
    vision = payload.get("vision", {})
    payload["vision"] = VisionPreprocessConfig(
        **dict(_mapping(vision, owner="vision configuration"))
    )
    return TextureBenchmarkMatrix(**payload)


__all__ = [
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_PIXELS",
    "PipelineArm",
    "PipelineBackend",
    "PipelineKind",
    "TEXTURE_BENCHMARK_MATRIX_SCHEMA",
    "TEXTURE_BENCHMARK_RUN_SCHEMA",
    "TextureBenchmarkMatrix",
    "VisionPreprocessConfig",
    "canonical_json_sha256",
    "file_sha256",
    "load_texture_benchmark_matrix",
    "require_sha256",
]
