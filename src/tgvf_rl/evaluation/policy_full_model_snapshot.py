"""Strict full-model snapshots for PRL13 official-visible evaluation.

This module is deliberately separate from the historical policy-evaluation
snapshot, whose public ABI is a LoRA pointer.  A full-model snapshot is either
the immutable step-zero Hugging Face tree or one complete upstream veRL FSDP
checkpoint.  It is never represented as a zero/merged LoRA adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import torch

from tgvf_rl.contracts.identity import ModelIdentity, PolicyVersion
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_VLLM_WORKER_EXTENSION_FQN,
)
from tgvf_rl.framework.vllm.registration import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
)
from tgvf_rl.policy.deepeyes_native_contract import (
    DeepEyesNativeRunContract,
    load_deepeyes_native_run_contract,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMAS,
)
from tgvf_rl.protocol import NativeToolCapabilityProfile


FULL_MODEL_SNAPSHOT_SCHEMA = "tgvf-prl13-full-model-snapshot-v1"
FULL_MODEL_SNAPSHOT_SCHEMA_V2 = "tgvf-full-model-snapshot-v2"
FULL_MODEL_MATERIALIZATION_SCHEMA = "tgvf-prl13-full-model-materialization-v1"
FULL_MODEL_EVALUATION_BACKEND = "full_model"
FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS = 1_003_520

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FSDP_STEP = re.compile(r"^global_step_([0-9]+)$")
_MODEL_SHARD = re.compile(r"^model_world_size_([0-9]+)_rank_([0-9]+)[.]pt$")
_OPTIMIZER_SHARD = re.compile(r"^optim_world_size_([0-9]+)_rank_([0-9]+)[.]pt$")
_EXTRA_SHARD = re.compile(r"^extra_state_world_size_([0-9]+)_rank_([0-9]+)[.]pt$")
_HF_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_IGNORED_TREE_TOP_LEVEL = frozenset({".cache", ".git"})
_EXPECTED_HF_MODEL_TYPE = "qwen3_vl"
_EXPECTED_HF_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
_VLLM_ENGINE_CONTEXT_HEADROOM = 1024
_FORBIDDEN_FULL_MODEL_NAMES = {
    "adapter_config.json",
    "latest-lora-snapshot.json",
    "lora_train_meta.json",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or str(candidate) != value
    ):
        raise ValueError("snapshot file path must be canonical and relative")
    return value


@dataclass(frozen=True, order=True, slots=True)
class FullModelFileDigest:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("snapshot file size must be a non-negative integer")
        _require_sha256(self.sha256, name="snapshot file sha256")

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _scan_regular_tree(root: Path) -> tuple[FullModelFileDigest, ...]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("full-model source must be a directory")
    rows: list[FullModelFileDigest] = []
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if relative_path.parts[0] in _IGNORED_TREE_TOP_LEVEL:
            continue
        if path.is_symlink():
            raise ValueError(f"full-model tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"full-model tree contains a special file: {path}")
        relative = relative_path.as_posix()
        stat = path.stat()
        rows.append(
            FullModelFileDigest(
                relative_path=relative,
                size_bytes=stat.st_size,
                sha256=_sha256_file(path),
            )
        )
    if not rows:
        raise ValueError("full-model tree contains no regular files")
    return tuple(rows)


def _file_digest(path: Path, *, relative_path: str) -> FullModelFileDigest:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"full-model closure file is not regular: {path}")
    return FullModelFileDigest(
        relative_path=relative_path,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _prefixed_file_digests(
    files: Sequence[FullModelFileDigest], *, prefix: str
) -> tuple[FullModelFileDigest, ...]:
    return tuple(
        FullModelFileDigest(
            relative_path=f"{prefix}/{item.relative_path}",
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
        for item in files
    )


def _strip_file_digest_prefix(
    files: Sequence[FullModelFileDigest], *, prefix: str
) -> tuple[FullModelFileDigest, ...]:
    marker = f"{prefix}/"
    return tuple(
        FullModelFileDigest(
            relative_path=item.relative_path.removeprefix(marker),
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
        for item in files
        if item.relative_path.startswith(marker)
    )


def _assert_tree_names_safe(root: Path) -> None:
    """Reject unsafe/adapter entries without reading checkpoint payload bytes."""

    root = root.resolve(strict=True)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in _IGNORED_TREE_TOP_LEVEL:
            continue
        if path.is_symlink():
            raise ValueError(f"full-model tree contains a symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"full-model tree contains a special file: {path}")
        name = path.name.casefold()
        if (
            name in _FORBIDDEN_FULL_MODEL_NAMES
            or name.startswith("adapter_model")
            or name.startswith("lora_")
        ):
            raise ValueError(
                f"full-model snapshot contains a LoRA/adapter artifact: {path}"
            )


def _tree_sha256(files: Sequence[FullModelFileDigest]) -> str:
    return _canonical_sha256([item.as_record() for item in files])


def _assert_no_lora_artifacts(files: Sequence[FullModelFileDigest]) -> None:
    for item in files:
        name = Path(item.relative_path).name.casefold()
        if (
            name in _FORBIDDEN_FULL_MODEL_NAMES
            or name.startswith("adapter_model")
            or name.startswith("lora_")
        ):
            raise ValueError(
                f"full-model snapshot contains a LoRA/adapter artifact: {item.relative_path}"
            )


def _state_dict_parameter_keys(path: Path) -> tuple[str, ...]:
    """Read only checkpoint metadata/storages through mmap and return its keys."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older torch only
        state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("FSDP model shard is not a non-empty state dict")
    keys = tuple(sorted(str(key) for key in state))
    del state
    if any("lora_" in key.casefold() or ".adapter_" in key.casefold() for key in keys):
        raise ValueError("full-model FSDP shard contains adapter-only parameter names")
    return keys


def _component_key_proof(keys: Sequence[str]) -> tuple[str, int, int]:
    vision = sum(
        1
        for key in keys
        if any(fragment in key.casefold() for fragment in ("visual", "vision"))
    )
    language = sum(
        1
        for key in keys
        if any(
            fragment in key.casefold()
            for fragment in ("language_model", "lm_head", "embed_tokens")
        )
    )
    if vision == 0 or language == 0:
        raise ValueError(
            "full-model checkpoint must contain both vision and language parameters"
        )
    return _canonical_sha256(list(keys)), vision, language


class FullModelSourceKind(str, Enum):
    BASE_HF = "base_hf"
    VERL_FSDP = "verl_fsdp"


@dataclass(frozen=True, slots=True)
class FullModelCheckpointOwner:
    """Identity of the run that owns checkpoint bytes, not its eval protocol."""

    run_id: str
    run_identity_sha256: str
    config_path: str
    config_file_sha256: str
    completion_path: str
    completion_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("full-model checkpoint owner run_id must be non-empty")
        if (
            not Path(self.config_path).is_absolute()
            or not Path(self.completion_path).is_absolute()
        ):
            raise ValueError("full-model checkpoint owner paths must be absolute")
        _require_sha256(self.run_identity_sha256, name="checkpoint owner run identity")
        _require_sha256(self.config_file_sha256, name="checkpoint owner config file")
        _require_sha256(
            self.completion_file_sha256, name="checkpoint owner completion file"
        )

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullModelSnapshotManifest:
    run_id: str
    run_contract_path: str
    run_contract_file_sha256: str
    run_identity_sha256: str
    model_identity: ModelIdentity
    optimizer_step: int
    source_kind: FullModelSourceKind
    source_path: str
    source_files: tuple[FullModelFileDigest, ...]
    source_tree_sha256: str
    checkpoint_sha256: str
    weights_sha256: str
    fsdp_world_size: int | None
    parameter_keys_sha256: str
    parameter_key_count: int
    vision_parameter_key_count: int
    language_parameter_key_count: int
    embedded_hf_model_path: str | None = None
    checkpoint_owner: FullModelCheckpointOwner | None = None
    schema_version: str = FULL_MODEL_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version not in {
            FULL_MODEL_SNAPSHOT_SCHEMA,
            FULL_MODEL_SNAPSHOT_SCHEMA_V2,
        }:
            raise ValueError("full-model snapshot schema differs")
        if (
            self.schema_version == FULL_MODEL_SNAPSHOT_SCHEMA
            and self.checkpoint_owner is not None
        ):
            raise ValueError("v1 full-model snapshot forbids checkpoint owner fields")
        if self.schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2 and not isinstance(
            self.checkpoint_owner, FullModelCheckpointOwner
        ):
            raise ValueError(
                "v2 full-model snapshot requires checkpoint owner identity"
            )
        if not self.run_id:
            raise ValueError("full-model snapshot run_id must be non-empty")
        if (
            not Path(self.run_contract_path).is_absolute()
            or not Path(self.source_path).is_absolute()
        ):
            raise ValueError("full-model snapshot paths must be absolute")
        for name, value in (
            ("run contract", self.run_contract_file_sha256),
            ("run identity", self.run_identity_sha256),
            ("source tree", self.source_tree_sha256),
            ("checkpoint", self.checkpoint_sha256),
            ("weights", self.weights_sha256),
            ("parameter keys", self.parameter_keys_sha256),
        ):
            _require_sha256(value, name=f"{name} sha256")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if (
            not self.source_files
            or tuple(sorted(self.source_files)) != self.source_files
        ):
            raise ValueError("full-model source files must be non-empty and sorted")
        if len({item.relative_path for item in self.source_files}) != len(
            self.source_files
        ):
            raise ValueError("full-model source file paths must be unique")
        if _tree_sha256(self.source_files) != self.source_tree_sha256:
            raise ValueError("full-model source tree digest differs")
        _assert_no_lora_artifacts(self.source_files)
        if (
            type(self.parameter_key_count) is not int
            or self.parameter_key_count <= 0
            or type(self.vision_parameter_key_count) is not int
            or self.vision_parameter_key_count <= 0
            or type(self.language_parameter_key_count) is not int
            or self.language_parameter_key_count <= 0
        ):
            raise ValueError("full-model component parameter proof is incomplete")
        if self.source_kind is FullModelSourceKind.BASE_HF:
            if self.optimizer_step != 0 or self.fsdp_world_size is not None:
                raise ValueError("base HF snapshot must be optimizer step zero")
        elif self.source_kind is FullModelSourceKind.VERL_FSDP:
            if self.optimizer_step <= 0 or self.fsdp_world_size is None:
                raise ValueError(
                    "veRL FSDP snapshot requires a positive step/world size"
                )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError("full-model source kind differs")
        if (
            self.embedded_hf_model_path is not None
            and not Path(self.embedded_hf_model_path).is_absolute()
        ):
            raise ValueError("embedded HF model path must be absolute")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.as_record(include_identity=False))

    @property
    def policy_version(self) -> PolicyVersion:
        run_id = (
            self.run_id
            if self.checkpoint_owner is None
            else self.checkpoint_owner.run_id
        )
        return PolicyVersion(run_id, self.optimizer_step, self.weights_sha256)

    @property
    def owner_run_identity_sha256(self) -> str:
        return (
            self.run_identity_sha256
            if self.checkpoint_owner is None
            else self.checkpoint_owner.run_identity_sha256
        )

    def as_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_contract_path": self.run_contract_path,
            "run_contract_file_sha256": self.run_contract_file_sha256,
            "run_identity_sha256": self.run_identity_sha256,
            "model_identity": asdict(self.model_identity),
            "optimizer_step": self.optimizer_step,
            "source_kind": self.source_kind.value,
            "source_path": self.source_path,
            "source_files": [item.as_record() for item in self.source_files],
            "source_tree_sha256": self.source_tree_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weights_sha256": self.weights_sha256,
            "fsdp_world_size": self.fsdp_world_size,
            "parameter_keys_sha256": self.parameter_keys_sha256,
            "parameter_key_count": self.parameter_key_count,
            "vision_parameter_key_count": self.vision_parameter_key_count,
            "language_parameter_key_count": self.language_parameter_key_count,
            "embedded_hf_model_path": self.embedded_hf_model_path,
        }
        if self.schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2:
            assert self.checkpoint_owner is not None
            record["checkpoint_owner"] = self.checkpoint_owner.as_record()
        if include_identity:
            record["identity_sha256"] = _canonical_sha256(record)
        return record


def _model_identity(contract: DeepEyesNativeRunContract) -> ModelIdentity:
    model = contract.payload["model"]
    assert isinstance(model, Mapping)
    return ModelIdentity(
        family=str(model["family"]),
        model_name=str(model["name"]),
        revision_or_path=str(Path(str(model["path"])).resolve()),
        tokenizer_length=POLICY_PILOT_V1_TOKENIZER_LENGTH,
        chat_template_sha256=POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    )


def _hf_weight_files(
    files: Sequence[FullModelFileDigest], *, prefix: str = ""
) -> tuple[FullModelFileDigest, ...]:
    result = tuple(
        item
        for item in files
        if item.relative_path.startswith(prefix)
        and Path(item.relative_path).suffix.casefold() in _HF_WEIGHT_SUFFIXES
        and not Path(item.relative_path).name.casefold().startswith("adapter_model")
    )
    if not result:
        raise ValueError("Hugging Face full-model tree contains no weight files")
    return result


def _hf_parameter_keys(
    root: Path, files: Sequence[FullModelFileDigest]
) -> tuple[str, ...]:
    keys: set[str] = set()
    safetensors_files = [
        root / item.relative_path
        for item in files
        if Path(item.relative_path).suffix.casefold() == ".safetensors"
    ]
    if safetensors_files:
        from safetensors import safe_open

        for path in safetensors_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys.update(str(key) for key in handle.keys())
    else:
        bin_files = [
            root / item.relative_path
            for item in files
            if Path(item.relative_path).suffix.casefold() == ".bin"
        ]
        for path in bin_files:
            keys.update(_state_dict_parameter_keys(path))
    if any("lora_" in key.casefold() or ".adapter_" in key.casefold() for key in keys):
        raise ValueError("Hugging Face tree contains adapter parameter names")
    return tuple(sorted(keys))


def _require_hf_tree(
    root: Path, files: Sequence[FullModelFileDigest]
) -> tuple[str, int, int, int, str]:
    names = {item.relative_path for item in files}
    if "config.json" not in names:
        raise ValueError("Hugging Face full-model tree lacks config.json")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or any(
        key in config for key in ("peft_type", "base_model_name_or_path")
    ):
        raise ValueError("Hugging Face config is not a standalone full model")
    architectures = config.get("architectures")
    if (
        config.get("model_type") != _EXPECTED_HF_MODEL_TYPE
        or not isinstance(architectures, list)
        or _EXPECTED_HF_ARCHITECTURE not in architectures
    ):
        raise ValueError(
            "Hugging Face config is not the PRL13 Qwen3-VL full-model architecture"
        )
    weights = _hf_weight_files(files)
    keys = _hf_parameter_keys(root, weights)
    key_sha, vision, language = _component_key_proof(keys)
    return key_sha, len(keys), vision, language, _tree_sha256(weights)


def _rank_files(
    actor_files: Sequence[FullModelFileDigest],
    pattern: re.Pattern[str],
    world_size: int,
) -> tuple[FullModelFileDigest, ...]:
    rows: dict[int, FullModelFileDigest] = {}
    for item in actor_files:
        match = pattern.fullmatch(Path(item.relative_path).name)
        if match is None:
            continue
        if int(match.group(1)) != world_size:
            raise ValueError("FSDP shard filename world size differs")
        rank = int(match.group(2))
        if rank in rows:
            raise ValueError("duplicate FSDP rank shard")
        rows[rank] = item
    if set(rows) != set(range(world_size)):
        raise ValueError("FSDP checkpoint rank shard set is incomplete")
    return tuple(rows[rank] for rank in range(world_size))


def _rank_paths(
    actor: Path, pattern: re.Pattern[str], world_size: int
) -> tuple[Path, ...]:
    """Validate one FSDP rank-file set using directory metadata only."""

    rows: dict[int, Path] = {}
    for path in actor.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("FSDP rank shard is not a regular file")
        if int(match.group(1)) != world_size:
            raise ValueError("FSDP shard filename world size differs")
        rank = int(match.group(2))
        if rank in rows:
            raise ValueError("duplicate FSDP rank shard")
        rows[rank] = path
    if set(rows) != set(range(world_size)):
        raise ValueError("FSDP checkpoint rank shard set is incomplete")
    return tuple(rows[rank] for rank in range(world_size))


def _load_fsdp_world_size(actor: Path) -> tuple[Path, int]:
    fsdp_config_path = actor / "fsdp_config.json"
    if fsdp_config_path.is_symlink() or not fsdp_config_path.is_file():
        raise ValueError("veRL actor checkpoint lacks fsdp_config.json")
    fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
    if not isinstance(fsdp_config, Mapping) or set(fsdp_config) != {
        "FSDP_version",
        "world_size",
    }:
        raise ValueError("veRL fsdp_config.json fields differ")
    world_size = fsdp_config["world_size"]
    if (
        fsdp_config["FSDP_version"] != 2
        or type(world_size) is not int
        or world_size <= 0
    ):
        raise ValueError("PRL13 evaluation requires a positive FSDP2 world size")
    return fsdp_config_path, world_size


def _assert_runtime_world_size(
    contract: DeepEyesNativeRunContract,
    *,
    checkpoint_world_size: int,
    runtime_fsdp_world_size: int | None,
) -> None:
    formal_world_size = contract.payload["distributed"]["world_size"]
    if runtime_fsdp_world_size is None:
        if checkpoint_world_size != formal_world_size:
            raise ValueError(
                "FSDP checkpoint world size differs from the PRL13 run; "
                "bind runtime_fsdp_world_size explicitly"
            )
        return
    if type(runtime_fsdp_world_size) is not int or runtime_fsdp_world_size <= 0:
        raise ValueError("runtime_fsdp_world_size must be a positive integer")
    if checkpoint_world_size != runtime_fsdp_world_size:
        raise ValueError("FSDP checkpoint world size differs from the bound runtime")


def build_full_model_snapshot_manifest(
    contract: DeepEyesNativeRunContract,
    *,
    source_path: str | Path,
    optimizer_step: int,
    runtime_fsdp_world_size: int | None = None,
    checkpoint_owner: FullModelCheckpointOwner | None = None,
) -> FullModelSnapshotManifest:
    """Hash and validate one base-HF or complete upstream veRL FSDP closure."""

    if not isinstance(contract, DeepEyesNativeRunContract):
        raise TypeError("contract must be a DeepEyesNativeRunContract")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer_step must be a non-negative integer")
    source = Path(source_path).resolve(strict=True)
    run_file_sha = _sha256_file(contract.source_path)
    model_identity = _model_identity(contract)
    if checkpoint_owner is not None:
        for name, path, expected_sha256 in (
            (
                "config",
                Path(checkpoint_owner.config_path),
                checkpoint_owner.config_file_sha256,
            ),
            (
                "completion",
                Path(checkpoint_owner.completion_path),
                checkpoint_owner.completion_file_sha256,
            ),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"checkpoint owner {name} must be a regular file")
            if _sha256_file(path) != expected_sha256:
                raise ValueError(f"checkpoint owner {name} bytes changed")
    schema_version = (
        FULL_MODEL_SNAPSHOT_SCHEMA
        if checkpoint_owner is None
        else FULL_MODEL_SNAPSHOT_SCHEMA_V2
    )

    if optimizer_step == 0:
        if runtime_fsdp_world_size is not None:
            raise ValueError("step-zero snapshot forbids runtime_fsdp_world_size")
        source_files = _scan_regular_tree(source)
        _assert_no_lora_artifacts(source_files)
        expected = Path(model_identity.revision_or_path).resolve(strict=True)
        if source != expected:
            raise ValueError(
                "step-zero snapshot must bind the run's exact base model path"
            )
        key_sha, key_count, vision_count, language_count, weights_sha = (
            _require_hf_tree(source, source_files)
        )
        return FullModelSnapshotManifest(
            run_id=contract.run_id,
            run_contract_path=str(contract.source_path),
            run_contract_file_sha256=run_file_sha,
            run_identity_sha256=contract.identity_sha256,
            model_identity=model_identity,
            optimizer_step=0,
            source_kind=FullModelSourceKind.BASE_HF,
            source_path=str(source),
            source_files=source_files,
            source_tree_sha256=_tree_sha256(source_files),
            checkpoint_sha256=_tree_sha256(source_files),
            weights_sha256=weights_sha,
            fsdp_world_size=None,
            parameter_keys_sha256=key_sha,
            parameter_key_count=key_count,
            vision_parameter_key_count=vision_count,
            language_parameter_key_count=language_count,
            embedded_hf_model_path=str(source),
            checkpoint_owner=checkpoint_owner,
            schema_version=schema_version,
        )

    match = _FSDP_STEP.fullmatch(source.name)
    if match is None or int(match.group(1)) != optimizer_step:
        raise ValueError(
            "FSDP checkpoint directory must be global_step_<optimizer_step>"
        )
    actor = source / "actor"
    if actor.is_symlink() or not actor.is_dir():
        raise ValueError("veRL checkpoint lacks a regular actor directory")
    data_path = source / "data.pt"
    if not data_path.is_file() or data_path.is_symlink():
        raise ValueError("veRL checkpoint lacks its dataloader cursor data.pt")
    fsdp_config_path, world_size = _load_fsdp_world_size(actor)
    _assert_runtime_world_size(
        contract,
        checkpoint_world_size=world_size,
        runtime_fsdp_world_size=runtime_fsdp_world_size,
    )
    _assert_tree_names_safe(source)
    model_paths = _rank_paths(actor, _MODEL_SHARD, world_size)
    _rank_paths(actor, _OPTIMIZER_SHARD, world_size)
    _rank_paths(actor, _EXTRA_SHARD, world_size)

    # Upstream veRL may save a complete, directly evaluable HF closure beside
    # its FSDP resume shards.  In that case the 96+ GiB model/optimizer payload
    # is not part of evaluation: validate its rank-file set above, but hash only
    # the HF weights and the two small checkpoint identity/cursor files.
    embedded = actor / "huggingface"
    embedded_path: str | None = None
    embedded_files: tuple[FullModelFileDigest, ...] = ()
    if embedded.is_dir() and not embedded.is_symlink():
        embedded_files = _scan_regular_tree(embedded)
        try:
            embedded_proof = _require_hf_tree(embedded, embedded_files)
        except ValueError as error:
            if any(
                Path(item.relative_path).suffix.casefold() in _HF_WEIGHT_SUFFIXES
                for item in embedded_files
            ):
                raise ValueError("embedded HF weights are incomplete") from error
        else:
            embedded_path = str(embedded.resolve())

    if embedded_path is not None:
        key_sha, key_count, vision_count, language_count, weights_sha = embedded_proof
        source_files = tuple(
            sorted(
                (
                    _file_digest(
                        fsdp_config_path, relative_path="actor/fsdp_config.json"
                    ),
                    _file_digest(data_path, relative_path="data.pt"),
                    *_prefixed_file_digests(embedded_files, prefix="actor/huggingface"),
                )
            )
        )
        source_tree_sha = _tree_sha256(source_files)
        return FullModelSnapshotManifest(
            run_id=contract.run_id,
            run_contract_path=str(contract.source_path),
            run_contract_file_sha256=run_file_sha,
            run_identity_sha256=contract.identity_sha256,
            model_identity=model_identity,
            optimizer_step=optimizer_step,
            source_kind=FullModelSourceKind.VERL_FSDP,
            source_path=str(source),
            source_files=source_files,
            source_tree_sha256=source_tree_sha,
            checkpoint_sha256=source_tree_sha,
            weights_sha256=weights_sha,
            fsdp_world_size=world_size,
            parameter_keys_sha256=key_sha,
            parameter_key_count=key_count,
            vision_parameter_key_count=vision_count,
            language_parameter_key_count=language_count,
            embedded_hf_model_path=embedded_path,
            checkpoint_owner=checkpoint_owner,
            schema_version=schema_version,
        )

    # Config/tokenizer-only embedded trees still require an upstream FSDP merge.
    # Preserve the historical, fully hashed closure for that uncommon path.
    source_files = _scan_regular_tree(source)
    _assert_no_lora_artifacts(source_files)
    actor_files = tuple(
        FullModelFileDigest(
            relative_path=Path(item.relative_path).relative_to("actor").as_posix(),
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
        for item in source_files
        if item.relative_path.startswith("actor/")
    )
    model_shards = _rank_files(actor_files, _MODEL_SHARD, world_size)
    _rank_files(actor_files, _OPTIMIZER_SHARD, world_size)
    _rank_files(actor_files, _EXTRA_SHARD, world_size)
    keys = _state_dict_parameter_keys(model_paths[0])
    key_sha, vision_count, language_count = _component_key_proof(keys)
    weights_sha = _tree_sha256(model_shards)

    return FullModelSnapshotManifest(
        run_id=contract.run_id,
        run_contract_path=str(contract.source_path),
        run_contract_file_sha256=run_file_sha,
        run_identity_sha256=contract.identity_sha256,
        model_identity=model_identity,
        optimizer_step=optimizer_step,
        source_kind=FullModelSourceKind.VERL_FSDP,
        source_path=str(source),
        source_files=source_files,
        source_tree_sha256=_tree_sha256(source_files),
        checkpoint_sha256=_tree_sha256(source_files),
        weights_sha256=weights_sha,
        fsdp_world_size=world_size,
        parameter_keys_sha256=key_sha,
        parameter_key_count=len(keys),
        vision_parameter_key_count=vision_count,
        language_parameter_key_count=language_count,
        embedded_hf_model_path=None,
        checkpoint_owner=checkpoint_owner,
        schema_version=schema_version,
    )


class FullModelMaterializationMode(str, Enum):
    BASE_HF = "base_hf"
    EMBEDDED_HF = "embedded_hf"
    VERL_FSDP_MERGE = "verl_fsdp_merge"


@dataclass(frozen=True, slots=True)
class FullModelMaterializationReceipt:
    snapshot_identity_sha256: str
    mode: FullModelMaterializationMode
    model_path: str
    model_files: tuple[FullModelFileDigest, ...]
    model_tree_sha256: str
    command: tuple[str, ...]
    schema_version: str = FULL_MODEL_MATERIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_MODEL_MATERIALIZATION_SCHEMA:
            raise ValueError("full-model materialization schema differs")
        _require_sha256(self.snapshot_identity_sha256, name="snapshot identity")
        _require_sha256(self.model_tree_sha256, name="materialized model tree")
        if not Path(self.model_path).is_absolute():
            raise ValueError("materialized model path must be absolute")
        if not self.model_files or tuple(sorted(self.model_files)) != self.model_files:
            raise ValueError("materialized model files must be non-empty and sorted")
        if _tree_sha256(self.model_files) != self.model_tree_sha256:
            raise ValueError("materialized model tree digest differs")
        _assert_no_lora_artifacts(self.model_files)
        if self.mode is FullModelMaterializationMode.VERL_FSDP_MERGE:
            if not self.command:
                raise ValueError(
                    "FSDP materialization must bind its executable command"
                )
        elif self.command:
            raise ValueError("non-merge materialization must not claim a command")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.as_record(include_identity=False))

    def as_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "snapshot_identity_sha256": self.snapshot_identity_sha256,
            "mode": self.mode.value,
            "model_path": self.model_path,
            "model_files": [item.as_record() for item in self.model_files],
            "model_tree_sha256": self.model_tree_sha256,
            "command": list(self.command),
        }
        if include_identity:
            record["identity_sha256"] = _canonical_sha256(record)
        return record


def full_model_merger_command(
    manifest: FullModelSnapshotManifest, target_dir: str | Path
) -> tuple[str, ...]:
    if manifest.source_kind is not FullModelSourceKind.VERL_FSDP:
        raise ValueError("only veRL FSDP snapshots require the model merger")
    target = Path(target_dir).resolve()
    actor = Path(manifest.source_path) / "actor"
    return (
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor),
        "--target_dir",
        str(target),
        "--use_cpu_initialization",
    )


def full_model_materialization_preflight(
    manifest: FullModelSnapshotManifest, *, target_dir: str | Path | None = None
) -> dict[str, object]:
    source_weight_bytes = sum(
        item.size_bytes
        for item in manifest.source_files
        if _MODEL_SHARD.fullmatch(Path(item.relative_path).name)
        or Path(item.relative_path).suffix.casefold() in _HF_WEIGHT_SUFFIXES
    )
    materialization_required = (
        manifest.source_kind is FullModelSourceKind.VERL_FSDP
        and manifest.embedded_hf_model_path is None
    )
    command: tuple[str, ...] = ()
    if materialization_required:
        if target_dir is None:
            raise ValueError("FSDP merge preflight requires target_dir")
        command = full_model_merger_command(manifest, target_dir)
    return {
        "schema_version": "tgvf-prl13-full-model-materialization-preflight-v1",
        "snapshot_identity_sha256": manifest.identity_sha256,
        "source_kind": manifest.source_kind.value,
        "optimizer_step": manifest.optimizer_step,
        "fsdp_world_size": manifest.fsdp_world_size,
        "materialization_required": materialization_required,
        "embedded_hf_model_path": manifest.embedded_hf_model_path,
        "source_weight_bytes": source_weight_bytes,
        # The upstream CPU merger may simultaneously retain the input shards,
        # merged state dict, and initialized target model.  Keep this estimate
        # deliberately conservative so the preflight does not bless a host
        # that only has enough RAM for the serialized weights themselves.
        "estimated_peak_cpu_bytes": int(source_weight_bytes * 4.0),
        "estimated_output_bytes": int(source_weight_bytes * 1.1),
        "command": list(command),
        "gpu_required": False,
        "distributed_launch_required": False,
    }


def _default_command_runner(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


def materialize_full_model_snapshot(
    manifest: FullModelSnapshotManifest,
    *,
    target_dir: str | Path | None = None,
    command_runner: Callable[[tuple[str, ...]], None] = _default_command_runner,
) -> FullModelMaterializationReceipt:
    """Resolve a full snapshot to a directly loadable, non-LoRA HF model."""

    if not isinstance(manifest, FullModelSnapshotManifest):
        raise TypeError("manifest must be a FullModelSnapshotManifest")
    if manifest.source_kind is FullModelSourceKind.BASE_HF:
        mode = FullModelMaterializationMode.BASE_HF
        model_path = Path(manifest.source_path)
        model_files: tuple[FullModelFileDigest, ...] | None = manifest.source_files
        command: tuple[str, ...] = ()
    elif manifest.embedded_hf_model_path is not None:
        mode = FullModelMaterializationMode.EMBEDDED_HF
        model_path = Path(manifest.embedded_hf_model_path)
        model_files = _strip_file_digest_prefix(
            manifest.source_files, prefix="actor/huggingface"
        )
        if not model_files:
            raise ValueError("embedded HF snapshot lacks its hashed evaluation closure")
        command = ()
    else:
        model_files = None
        if target_dir is None:
            raise ValueError(
                "FSDP snapshot without embedded HF weights requires target_dir"
            )
        model_path = Path(target_dir).resolve()
        source = Path(manifest.source_path).resolve(strict=True)
        if (
            model_path == source
            or source in model_path.parents
            or model_path in source.parents
        ):
            raise ValueError(
                "materialized model target must be outside the checkpoint tree"
            )
        if model_path.exists():
            if (
                model_path.is_symlink()
                or not model_path.is_dir()
                or any(model_path.iterdir())
            ):
                raise ValueError(
                    "materialized model target must be absent or an empty directory"
                )
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)
        command = full_model_merger_command(manifest, model_path)
        command_runner(command)
        mode = FullModelMaterializationMode.VERL_FSDP_MERGE
    model_path = model_path.resolve(strict=True)
    if model_files is None:
        model_files = _scan_regular_tree(model_path)
    _assert_no_lora_artifacts(model_files)
    _require_hf_tree(model_path, model_files)
    return FullModelMaterializationReceipt(
        snapshot_identity_sha256=manifest.identity_sha256,
        mode=mode,
        model_path=str(model_path),
        model_files=model_files,
        model_tree_sha256=_tree_sha256(model_files),
        command=command,
    )


def _write_immutable_json(path: Path, record: Mapping[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable full-model record collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write for full-model record")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def write_full_model_snapshot_manifest(
    path: str | Path, manifest: FullModelSnapshotManifest
) -> None:
    _write_immutable_json(Path(path), manifest.as_record())


def write_full_model_materialization_receipt(
    path: str | Path, receipt: FullModelMaterializationReceipt
) -> None:
    _write_immutable_json(Path(path), receipt.as_record())


def _parse_file_records(value: object, *, name: str) -> tuple[FullModelFileDigest, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    rows: list[FullModelFileDigest] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"{name} entry fields differ")
        rows.append(FullModelFileDigest(**item))
    return tuple(rows)


def load_full_model_snapshot_manifest(path: str | Path) -> FullModelSnapshotManifest:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("full-model manifest must be a regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    common = {
        "schema_version",
        "run_id",
        "run_contract_path",
        "run_contract_file_sha256",
        "run_identity_sha256",
        "model_identity",
        "optimizer_step",
        "source_kind",
        "source_path",
        "source_files",
        "source_tree_sha256",
        "checkpoint_sha256",
        "weights_sha256",
        "fsdp_world_size",
        "parameter_keys_sha256",
        "parameter_key_count",
        "vision_parameter_key_count",
        "language_parameter_key_count",
        "embedded_hf_model_path",
        "identity_sha256",
    }
    if not isinstance(value, Mapping):
        raise ValueError("full-model manifest must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version == FULL_MODEL_SNAPSHOT_SCHEMA:
        expected = common
    elif schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2:
        expected = {*common, "checkpoint_owner"}
    else:
        raise ValueError("full-model manifest schema differs")
    if set(value) != expected:
        raise ValueError("full-model manifest fields differ")
    identity = value["identity_sha256"]
    payload = {key: nested for key, nested in value.items() if key != "identity_sha256"}
    if _canonical_sha256(payload) != identity:
        raise ValueError("full-model manifest identity differs")
    model = payload["model_identity"]
    if not isinstance(model, Mapping) or set(model) != {
        "family",
        "model_name",
        "revision_or_path",
        "tokenizer_length",
        "chat_template_sha256",
    }:
        raise ValueError("full-model manifest model identity fields differ")
    owner = payload.get("checkpoint_owner")
    if schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2:
        if not isinstance(owner, Mapping) or set(owner) != {
            "run_id",
            "run_identity_sha256",
            "config_path",
            "config_file_sha256",
            "completion_path",
            "completion_file_sha256",
        }:
            raise ValueError("full-model checkpoint owner fields differ")
        checkpoint_owner = FullModelCheckpointOwner(**owner)
    else:
        checkpoint_owner = None
    return FullModelSnapshotManifest(
        **{
            **payload,
            **(
                {"checkpoint_owner": checkpoint_owner}
                if schema_version == FULL_MODEL_SNAPSHOT_SCHEMA_V2
                else {}
            ),
            "model_identity": ModelIdentity(**model),
            "source_kind": FullModelSourceKind(payload["source_kind"]),
            "source_files": _parse_file_records(
                payload["source_files"], name="source_files"
            ),
        }
    )


def load_full_model_materialization_receipt(
    path: str | Path,
) -> FullModelMaterializationReceipt:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("full-model receipt must be a regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "snapshot_identity_sha256",
        "mode",
        "model_path",
        "model_files",
        "model_tree_sha256",
        "command",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("full-model receipt fields differ")
    identity = value["identity_sha256"]
    payload = {key: nested for key, nested in value.items() if key != "identity_sha256"}
    if _canonical_sha256(payload) != identity:
        raise ValueError("full-model receipt identity differs")
    return FullModelMaterializationReceipt(
        **{
            **payload,
            "mode": FullModelMaterializationMode(payload["mode"]),
            "model_files": _parse_file_records(
                payload["model_files"], name="model_files"
            ),
            "command": tuple(payload["command"]),
        }
    )


@dataclass(frozen=True, slots=True)
class _NativeEvaluationSampling:
    max_response_length: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    stop_token_ids: tuple[int, ...]
    stop_strings: tuple[str, ...]
    include_stop_str_in_output: bool
    ignore_eos: bool

    def identity_record(self) -> Mapping[str, object]:
        """Return the turn-invariant sampling/action-boundary identity."""

        return MappingProxyType(
            {
                "max_response_length": self.max_response_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "stop_token_ids": list(self.stop_token_ids),
                "stop_strings": list(self.stop_strings),
                "include_stop_str_in_output": self.include_stop_str_in_output,
                "ignore_eos": self.ignore_eos,
            }
        )

    def remaining_response_tokens(self, consumed_policy_tokens: int) -> int:
        if type(consumed_policy_tokens) is not int or consumed_policy_tokens < 0:
            raise ValueError("consumed_policy_tokens must be non-negative")
        remaining = self.max_response_length - consumed_policy_tokens
        if remaining <= 0:
            raise ValueError("the trajectory policy-token budget is exhausted")
        return remaining

    def as_vllm_parameters(self, *, max_tokens: int) -> Mapping[str, object]:
        if (
            type(max_tokens) is not int
            or not 0 < max_tokens <= self.max_response_length
        ):
            raise ValueError("max_tokens lies outside the trajectory budget")
        return MappingProxyType(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "stop_token_ids": list(self.stop_token_ids),
                "stop": list(self.stop_strings),
                "include_stop_str_in_output": self.include_stop_str_in_output,
                "ignore_eos": self.ignore_eos,
                "max_tokens": max_tokens,
                "logprobs": True,
            }
        )


@dataclass(frozen=True, slots=True)
class FullModelEvaluationSnapshot:
    """Verified runtime view consumed by the official-visible evaluator."""

    contract: DeepEyesNativeRunContract
    manifest: FullModelSnapshotManifest
    receipt: FullModelMaterializationReceipt
    run: object

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DeepEyesNativeRunContract):
            raise TypeError("full-model snapshot contract type differs")
        if not isinstance(self.manifest, FullModelSnapshotManifest) or not isinstance(
            self.receipt, FullModelMaterializationReceipt
        ):
            raise TypeError("full-model snapshot manifest/receipt type differs")
        if (
            self.contract.run_id != self.manifest.run_id
            or self.contract.identity_sha256 != self.manifest.run_identity_sha256
            or self.receipt.snapshot_identity_sha256 != self.manifest.identity_sha256
        ):
            raise ValueError("full-model snapshot identities disagree")

    @property
    def policy_version(self) -> PolicyVersion:
        return self.manifest.policy_version

    @property
    def run_identity_sha256(self) -> str:
        return self.manifest.owner_run_identity_sha256

    @property
    def model_path(self) -> Path:
        return Path(self.receipt.model_path)


def _native_evaluation_run_view(
    contract: DeepEyesNativeRunContract, manifest: FullModelSnapshotManifest
) -> object:
    rollout = contract.payload["rollout"]
    dataset = contract.payload["dataset"]
    assert isinstance(rollout, Mapping) and isinstance(dataset, Mapping)
    return SimpleNamespace(
        run_id=manifest.policy_version.run_id,
        identity_sha256=manifest.owner_run_identity_sha256,
        protocol_run_id=contract.run_id,
        protocol_run_identity_sha256=contract.identity_sha256,
        protocol_contract_path=str(contract.source_path),
        protocol_contract_file_sha256=manifest.run_contract_file_sha256,
        model=manifest.model_identity,
        policy=SimpleNamespace(
            # This is the already accepted Qwen3-VL evaluation processor cap.
            # It is included in the full evaluation identity by the caller.
            image_max_pixels=FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS,
            sampling=_NativeEvaluationSampling(
                max_response_length=int(rollout["max_response_length"]),
                temperature=float(rollout["temperature"]),
                top_p=float(rollout["top_p"]),
                top_k=-1,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                # The owner runs bind this same action boundary explicitly.
                # The native DeepEyes protocol contract predates those fields,
                # so reconstruct them here instead of silently inheriting
                # vLLM defaults during full-model evaluation.
                stop_token_ids=(151_645,),
                stop_strings=("</tool_call>",),
                include_stop_str_in_output=True,
                ignore_eos=False,
            ),
        ),
        rollout_rng=SimpleNamespace(master_seed=int(dataset["schedule_seed"])),
    )


def _is_large_checkpoint_payload(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.suffix.casefold() in _HF_WEIGHT_SUFFIXES
        or _MODEL_SHARD.fullmatch(path.name) is not None
        or _OPTIMIZER_SHARD.fullmatch(path.name) is not None
        or _EXTRA_SHARD.fullmatch(path.name) is not None
    )


def _verify_file_records_lightweight(
    root: Path,
    files: Sequence[FullModelFileDigest],
    *,
    exact_tree: bool,
) -> None:
    """Verify names/sizes and small bytes, but never stream model payloads."""

    root = root.resolve(strict=True)
    expected = {item.relative_path: item for item in files}
    if exact_tree:
        actual: set[str] = set()
        for path in root.rglob("*"):
            relative_path = path.relative_to(root)
            if relative_path.parts[0] in _IGNORED_TREE_TOP_LEVEL:
                continue
            if path.is_symlink():
                raise ValueError(f"full-model tree contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"full-model tree contains a special file: {path}")
            actual.add(relative_path.as_posix())
        if actual != set(expected):
            raise ValueError("materialized full-model file set changed")
    for relative_path, item in expected.items():
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("full-model closure file is absent or not regular")
        if path.stat().st_size != item.size_bytes:
            raise ValueError("full-model closure file size changed")
        if not _is_large_checkpoint_payload(relative_path):
            if _sha256_file(path) != item.sha256:
                raise ValueError("full-model closure small-file bytes changed")


def _verify_fsdp_rank_sets(manifest: FullModelSnapshotManifest) -> None:
    if manifest.source_kind is not FullModelSourceKind.VERL_FSDP:
        return
    assert manifest.fsdp_world_size is not None
    actor = Path(manifest.source_path) / "actor"
    _, config_world_size = _load_fsdp_world_size(actor)
    if config_world_size != manifest.fsdp_world_size:
        raise ValueError("FSDP runtime world size changed")
    _rank_paths(actor, _MODEL_SHARD, config_world_size)
    _rank_paths(actor, _OPTIMIZER_SHARD, config_world_size)
    _rank_paths(actor, _EXTRA_SHARD, config_world_size)


def _receipt_files_from_manifest(
    manifest: FullModelSnapshotManifest,
    receipt: FullModelMaterializationReceipt,
) -> tuple[FullModelFileDigest, ...] | None:
    if receipt.mode is FullModelMaterializationMode.BASE_HF:
        if manifest.source_kind is not FullModelSourceKind.BASE_HF:
            raise ValueError("base-HF receipt/source kinds disagree")
        if Path(receipt.model_path) != Path(manifest.source_path):
            raise ValueError("base-HF receipt model path differs")
        return manifest.source_files
    if receipt.mode is FullModelMaterializationMode.EMBEDDED_HF:
        if manifest.embedded_hf_model_path is None or Path(receipt.model_path) != Path(
            manifest.embedded_hf_model_path
        ):
            raise ValueError("embedded-HF receipt model path differs")
        files = _strip_file_digest_prefix(
            manifest.source_files, prefix="actor/huggingface"
        )
        if not files:
            raise ValueError("embedded-HF manifest lacks its evaluation closure")
        return files
    if receipt.mode is not FullModelMaterializationMode.VERL_FSDP_MERGE:
        raise ValueError("full-model receipt mode differs")
    if manifest.source_kind is not FullModelSourceKind.VERL_FSDP:
        raise ValueError("FSDP merge receipt/source kinds disagree")
    return None


def _assert_model_header_matches_manifest(
    model_path: Path,
    model_files: Sequence[FullModelFileDigest],
    manifest: FullModelSnapshotManifest,
) -> None:
    key_sha, key_count, vision_count, language_count, weights_sha = _require_hf_tree(
        model_path, model_files
    )
    if (
        manifest.embedded_hf_model_path is not None
        or manifest.source_kind is FullModelSourceKind.BASE_HF
    ):
        if (
            key_sha != manifest.parameter_keys_sha256
            or key_count != manifest.parameter_key_count
            or vision_count != manifest.vision_parameter_key_count
            or language_count != manifest.language_parameter_key_count
            or weights_sha != manifest.weights_sha256
        ):
            raise ValueError("materialized full-model header differs from snapshot")


def load_full_model_evaluation_snapshot(
    manifest_path: str | Path,
    receipt_path: str | Path,
    *,
    require_launchable_run: bool = True,
    runtime_lightweight: bool = False,
) -> FullModelEvaluationSnapshot:
    """Load a snapshot with strong or runtime-safe lightweight verification.

    Snapshot/config materialization keeps the default strong byte verification.
    Evaluation workers may opt into ``runtime_lightweight`` after those immutable
    records have been bound; that path checks file sets, sizes, small-file bytes,
    FSDP ranks, and HF weight headers without re-hashing multi-GiB payloads.
    """

    if type(runtime_lightweight) is not bool:
        raise TypeError("runtime_lightweight must be a bool")
    manifest = load_full_model_snapshot_manifest(manifest_path)
    receipt = load_full_model_materialization_receipt(receipt_path)
    contract = load_deepeyes_native_run_contract(
        manifest.run_contract_path, allow_template=not require_launchable_run
    )
    if _sha256_file(contract.source_path) != manifest.run_contract_file_sha256:
        raise ValueError("full-model run contract bytes changed")
    if (
        contract.run_id != manifest.run_id
        or contract.identity_sha256 != manifest.run_identity_sha256
        or _model_identity(contract) != manifest.model_identity
    ):
        raise ValueError("full-model run/model identity changed")
    if manifest.checkpoint_owner is not None:
        for name, path, expected_sha256 in (
            (
                "config",
                Path(manifest.checkpoint_owner.config_path),
                manifest.checkpoint_owner.config_file_sha256,
            ),
            (
                "completion",
                Path(manifest.checkpoint_owner.completion_path),
                manifest.checkpoint_owner.completion_file_sha256,
            ),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"full-model checkpoint owner {name} changed")
            if _sha256_file(path) != expected_sha256:
                raise ValueError(f"full-model checkpoint owner {name} bytes changed")
    if receipt.snapshot_identity_sha256 != manifest.identity_sha256:
        raise ValueError("full-model materialization belongs to another snapshot")
    model_path = Path(receipt.model_path).resolve(strict=True)
    manifest_receipt_files = _receipt_files_from_manifest(manifest, receipt)
    if runtime_lightweight:
        _assert_tree_names_safe(Path(manifest.source_path))
        _verify_fsdp_rank_sets(manifest)
        _verify_file_records_lightweight(
            Path(manifest.source_path), manifest.source_files, exact_tree=False
        )
        _verify_file_records_lightweight(
            model_path, receipt.model_files, exact_tree=True
        )
        _assert_model_header_matches_manifest(model_path, receipt.model_files, manifest)
    else:
        rebuilt = build_full_model_snapshot_manifest(
            contract,
            source_path=manifest.source_path,
            optimizer_step=manifest.optimizer_step,
            runtime_fsdp_world_size=(
                manifest.fsdp_world_size
                if manifest.source_kind is FullModelSourceKind.VERL_FSDP
                else None
            ),
            checkpoint_owner=manifest.checkpoint_owner,
        )
        if rebuilt != manifest:
            raise ValueError("full-model checkpoint closure changed")
        if manifest_receipt_files is not None:
            model_files = manifest_receipt_files
        else:
            model_files = _scan_regular_tree(model_path)
        if model_files != receipt.model_files:
            raise ValueError("materialized full-model bytes changed")
        _assert_model_header_matches_manifest(model_path, model_files, manifest)
    return FullModelEvaluationSnapshot(
        contract=contract,
        manifest=manifest,
        receipt=receipt,
        run=_native_evaluation_run_view(contract, manifest),
    )


def full_model_snapshot_identity_record(
    snapshot: FullModelEvaluationSnapshot,
) -> dict[str, object]:
    if not isinstance(snapshot, FullModelEvaluationSnapshot):
        raise TypeError("snapshot must be a FullModelEvaluationSnapshot")
    record: dict[str, object] = {
        "snapshot_backend": FULL_MODEL_EVALUATION_BACKEND,
        "run_id": snapshot.policy_version.run_id,
        "run_identity_sha256": snapshot.run_identity_sha256,
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "snapshot_identity_sha256": snapshot.manifest.identity_sha256,
        "checkpoint_sha256": snapshot.manifest.checkpoint_sha256,
        "source_tree_sha256": snapshot.manifest.source_tree_sha256,
        "source_kind": snapshot.manifest.source_kind.value,
        "fsdp_world_size": snapshot.manifest.fsdp_world_size,
        "materialization_identity_sha256": snapshot.receipt.identity_sha256,
        "materialized_model_tree_sha256": snapshot.receipt.model_tree_sha256,
        "materialization_mode": snapshot.receipt.mode.value,
        "evaluation_sampling": dict(snapshot.run.policy.sampling.identity_record()),
        "lora_request": None,
    }
    if snapshot.manifest.checkpoint_owner is not None:
        record["protocol_run_id"] = snapshot.manifest.run_id
        record["protocol_run_identity_sha256"] = snapshot.manifest.run_identity_sha256
        record["protocol_contract_path"] = snapshot.manifest.run_contract_path
        record["protocol_contract_file_sha256"] = (
            snapshot.manifest.run_contract_file_sha256
        )
        record["checkpoint_owner_config_path"] = (
            snapshot.manifest.checkpoint_owner.config_path
        )
        record["checkpoint_owner_config_file_sha256"] = (
            snapshot.manifest.checkpoint_owner.config_file_sha256
        )
        record["checkpoint_owner_completion_path"] = (
            snapshot.manifest.checkpoint_owner.completion_path
        )
        record["checkpoint_owner_completion_file_sha256"] = (
            snapshot.manifest.checkpoint_owner.completion_file_sha256
        )
    return record


def full_model_policy_evaluation_identity(
    config: object,
    snapshot: FullModelEvaluationSnapshot,
) -> dict[str, object]:
    """Bind one official-visible suite to exact full-model/checkpoint bytes."""

    if not isinstance(snapshot, FullModelEvaluationSnapshot):
        raise TypeError("snapshot must be a FullModelEvaluationSnapshot")
    # Keep one protocol/coordinate identity implementation for both backends.
    # The late import avoids a module cycle: policy_coredev owns the generic
    # task/execution envelope while this module owns the full-model closure.
    from .policy_coredev import policy_evaluation_identity

    return policy_evaluation_identity(config, snapshot)


def full_model_vllm_engine_kwargs(
    snapshot: FullModelEvaluationSnapshot,
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    inference_concurrency_per_gpu: int,
    gpu_memory_utilization: float,
    enable_chunked_prefill: bool,
    precomputed_image_embeds: bool = False,
) -> dict[str, object]:
    """Stock Qwen3-VL vLLM args; adapter loading is explicitly disabled."""

    if not isinstance(snapshot, FullModelEvaluationSnapshot):
        raise TypeError("snapshot must be a FullModelEvaluationSnapshot")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            max_model_len,
            max_num_batched_tokens,
            inference_concurrency_per_gpu,
        )
    ):
        raise ValueError("full-model vLLM capacities must be positive integers")
    if (
        not isinstance(gpu_memory_utilization, (int, float))
        or not 0 < float(gpu_memory_utilization) < 1
    ):
        raise ValueError("gpu_memory_utilization must lie in (0,1)")
    protocol = getattr(snapshot.run, "protocol", None)
    no_tool = (
        getattr(protocol, "tool_profile", None)
        is NativeToolCapabilityProfile.NO_TOOL
    )
    if type(precomputed_image_embeds) is not bool:
        raise TypeError("precomputed_image_embeds must be boolean")
    if precomputed_image_embeds and not (
        getattr(snapshot.run, "schema_version", None)
        in POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMAS
        and getattr(protocol, "tool_profile", None)
        is NativeToolCapabilityProfile.CROP_ONLY
    ):
        raise ValueError(
            "precomputed full-model runtime is restricted to exact matched Crop"
        )
    maximum_tool_calls = getattr(protocol, "maximum_tool_calls", 6)
    common = {
        "model": str(snapshot.model_path),
        # A veRL checkpoint can carry an exported tokenizer whose metadata differs
        # from the tokenizer used by the run.  Keep the policy weights checkpointed,
        # but bind tokenization to the run's exact base model for paired evaluation.
        "tokenizer": snapshot.run.model.revision_or_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "distributed_executor_backend": "mp",
        # Request budgets remain capped by the evaluation config.  vLLM 0.12
        # can account native multimodal placeholders slightly differently and
        # otherwise kills its engine at 32769 tokens, so the engine alone gets
        # implementation headroom without granting extra generation tokens.
        "max_model_len": max_model_len + _VLLM_ENGINE_CONTEXT_HEADROOM,
        "max_num_seqs": inference_concurrency_per_gpu,
        "max_num_batched_tokens": max(
            max_num_batched_tokens,
            max_model_len + _VLLM_ENGINE_CONTEXT_HEADROOM,
        ),
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "logprobs_mode": "processed_logprobs",
        "enforce_eager": False,
        "seed": snapshot.run.rollout_rng.master_seed,
        "enable_lora": False,
        "mm_processor_cache_gb": 0,
        # Use the driver-portable vision path already accepted by the project.
        "mm_encoder_attn_backend": "TORCH_SDPA",
        "limit_mm_per_prompt": {
            "image": (
                1
                if no_tool
                else 1 + int(maximum_tool_calls)
            ),
            "video": 0,
        },
    }
    if not precomputed_image_embeds:
        return common
    return {
        **common,
        "worker_extension_cls": TGVF_VLLM_WORKER_EXTENSION_FQN,
        "enable_mm_embeds": True,
        "mm_encoder_attn_backend": TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
        "hf_overrides": {"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
    }


def _full_model_uses_precomputed_crop_runtime(
    config: object,
    snapshot: FullModelEvaluationSnapshot,
) -> bool:
    """Select the training runtime only for the explicitly matched Crop owner."""

    return (
        getattr(config, "evaluation_protocol", None) == "training_run"
        and getattr(snapshot.run, "schema_version", None)
        in POLICY_E2E_CROP_TFREE_EXACT_MATCHED_RUN_CONFIG_SCHEMAS
        and getattr(getattr(snapshot.run, "protocol", None), "tool_profile", None)
        is NativeToolCapabilityProfile.CROP_ONLY
    )


async def build_full_model_standalone_manager(
    config: object,
    snapshot: FullModelEvaluationSnapshot,
) -> tuple[object, object, object]:
    """Construct stock vLLM over full weights, never a ``LoRARequest``."""

    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from .policy_coredev import StandaloneTGVFVLLMManager

    precomputed_image_embeds = _full_model_uses_precomputed_crop_runtime(
        config, snapshot
    )
    if precomputed_image_embeds:
        policy_config_path = Path(getattr(config, "policy_config_path")).resolve()
        if not policy_config_path.is_file() or policy_config_path.is_symlink():
            raise ValueError(
                "training-run full-model Crop requires a regular policy config"
            )
        os.environ["TGVF_POLICY_RUN_CONFIG_PATH"] = str(policy_config_path)
    kwargs = full_model_vllm_engine_kwargs(
        snapshot,
        max_model_len=int(getattr(config, "max_model_len")),
        max_num_batched_tokens=int(getattr(config, "max_num_batched_tokens")),
        inference_concurrency_per_gpu=int(
            getattr(config, "inference_concurrency_per_gpu")
        ),
        gpu_memory_utilization=float(getattr(config, "gpu_memory_utilization")),
        enable_chunked_prefill=bool(getattr(config, "enable_chunked_prefill")),
        precomputed_image_embeds=precomputed_image_embeds,
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    manager = StandaloneTGVFVLLMManager(
        engine,
        None,
        capture_hidden=False,
        native_pixels=not precomputed_image_embeds,
    )
    if manager.lora_request is not None:
        raise RuntimeError(
            "full-model evaluator unexpectedly constructed a LoRARequest"
        )
    return manager, engine, snapshot.run


__all__ = [
    "FULL_MODEL_EVALUATION_BACKEND",
    "FULL_MODEL_EVALUATION_IMAGE_MAX_PIXELS",
    "FULL_MODEL_MATERIALIZATION_SCHEMA",
    "FULL_MODEL_SNAPSHOT_SCHEMA",
    "FULL_MODEL_SNAPSHOT_SCHEMA_V2",
    "FullModelCheckpointOwner",
    "FullModelEvaluationSnapshot",
    "FullModelFileDigest",
    "FullModelMaterializationMode",
    "FullModelMaterializationReceipt",
    "FullModelSnapshotManifest",
    "FullModelSourceKind",
    "build_full_model_snapshot_manifest",
    "build_full_model_standalone_manager",
    "full_model_materialization_preflight",
    "full_model_merger_command",
    "full_model_policy_evaluation_identity",
    "full_model_snapshot_identity_record",
    "full_model_vllm_engine_kwargs",
    "load_full_model_evaluation_snapshot",
    "load_full_model_materialization_receipt",
    "load_full_model_snapshot_manifest",
    "materialize_full_model_snapshot",
    "write_full_model_materialization_receipt",
    "write_full_model_snapshot_manifest",
]
