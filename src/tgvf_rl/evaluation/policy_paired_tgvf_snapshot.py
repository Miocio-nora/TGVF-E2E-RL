"""Immutable full-Qwen plus RP66 snapshots for paired TGVF evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyLoRASnapshot,
    PolicyWeightSyncState,
    load_lora_snapshot_pointer,
)
from tgvf_rl.framework.verl.vllm_tool_runtime import adapter_owned_state_sha256
from tgvf_rl.policy.run_config import (
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)


PAIRED_TGVF_EVALUATION_BACKEND = "full_model_trainable_rp66"
PAIRED_TGVF_SNAPSHOT_SCHEMA = "tgvf.paired-qwen-rp66-snapshot.v1"
RP66SnapshotKind = Literal["stage1_artifact", "runtime_snapshot"]
_LARGE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return _sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _require_sha256(value: object, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{owner} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class PairedModelFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative = Path(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.relative_path
        ):
            raise ValueError("paired Qwen file path is unsafe")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("paired Qwen file size is invalid")
        _require_sha256(self.sha256, owner="paired Qwen file digest")

    def as_record(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PairedTGVFSnapshotReceipt:
    run_id: str
    run_identity_sha256: str
    policy_config_path: str
    policy_config_sha256: str
    optimizer_step: int
    qwen_model_path: str
    qwen_files: tuple[PairedModelFile, ...]
    qwen_tree_sha256: str
    rp66_kind: RP66SnapshotKind
    rp66_state_sha256: str
    rp66_storage_sha256: str
    rp66_pointer_path: str | None
    rp66_pointer_sha256: str | None
    combined_weights_sha256: str
    identity_sha256: str
    schema_version: str = PAIRED_TGVF_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_TGVF_SNAPSHOT_SCHEMA:
            raise ValueError("paired TGVF snapshot schema differs")
        if not self.run_id:
            raise ValueError("paired TGVF run_id is empty")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("paired TGVF optimizer step is invalid")
        if not self.qwen_files:
            raise ValueError("paired TGVF Qwen closure is empty")
        if len({item.relative_path for item in self.qwen_files}) != len(
            self.qwen_files
        ):
            raise ValueError("paired TGVF Qwen closure has duplicate files")
        for owner, value in (
            ("run identity", self.run_identity_sha256),
            ("policy config", self.policy_config_sha256),
            ("Qwen tree", self.qwen_tree_sha256),
            ("RP66 state", self.rp66_state_sha256),
            ("RP66 storage", self.rp66_storage_sha256),
            ("combined weights", self.combined_weights_sha256),
            ("receipt identity", self.identity_sha256),
        ):
            _require_sha256(value, owner=owner)
        if self.rp66_kind == "stage1_artifact":
            if self.optimizer_step != 0:
                raise ValueError("stage1 RP66 is valid only at optimizer step zero")
            if self.rp66_pointer_path is not None or self.rp66_pointer_sha256 is not None:
                raise ValueError("stage1 RP66 snapshot must not bind a runtime pointer")
        elif self.rp66_kind == "runtime_snapshot":
            if self.optimizer_step == 0:
                raise ValueError("runtime RP66 snapshot requires a nonzero step")
            if self.rp66_pointer_path is None or self.rp66_pointer_sha256 is None:
                raise ValueError("runtime RP66 snapshot requires a fixed pointer")
            _require_sha256(self.rp66_pointer_sha256, owner="RP66 pointer")
        else:
            raise ValueError("paired TGVF RP66 snapshot kind differs")
        if self.qwen_tree_sha256 != _qwen_tree_sha256(self.qwen_files):
            raise ValueError("paired TGVF Qwen tree identity differs")
        expected_combined = _canonical_sha256(
            {
                "schema": "tgvf.paired-qwen-rp66-weights.v1",
                "optimizer_step": self.optimizer_step,
                "qwen_tree_sha256": self.qwen_tree_sha256,
                "rp66_state_sha256": self.rp66_state_sha256,
            }
        )
        if not hmac.compare_digest(expected_combined, self.combined_weights_sha256):
            raise ValueError("paired TGVF combined weight identity differs")
        if self.identity_sha256 != _canonical_sha256(
            self.as_record(include_identity=False)
        ):
            raise ValueError("paired TGVF receipt identity differs")

    def as_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "policy_config_path": self.policy_config_path,
            "policy_config_sha256": self.policy_config_sha256,
            "optimizer_step": self.optimizer_step,
            "qwen_model_path": self.qwen_model_path,
            "qwen_files": [item.as_record() for item in self.qwen_files],
            "qwen_tree_sha256": self.qwen_tree_sha256,
            "rp66_kind": self.rp66_kind,
            "rp66_state_sha256": self.rp66_state_sha256,
            "rp66_storage_sha256": self.rp66_storage_sha256,
            "rp66_pointer_path": self.rp66_pointer_path,
            "rp66_pointer_sha256": self.rp66_pointer_sha256,
            "combined_weights_sha256": self.combined_weights_sha256,
        }
        if include_identity:
            record["identity_sha256"] = self.identity_sha256
        return record


@dataclass(frozen=True, slots=True)
class PairedTGVFEvaluationSnapshot:
    run: PolicyE2ESmokeRunConfig
    receipt: PairedTGVFSnapshotReceipt
    rp66_tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.run.run_id != self.receipt.run_id:
            raise ValueError("paired TGVF run_id differs")
        if self.run.identity_sha256 != self.receipt.run_identity_sha256:
            raise ValueError("paired TGVF run identity differs")
        if adapter_owned_state_sha256(self.rp66_tensors) != (
            self.receipt.rp66_state_sha256
        ):
            raise ValueError("paired TGVF RP66 tensor identity differs")

    @property
    def policy_version(self) -> PolicyVersion:
        return PolicyVersion(
            self.run.run_id,
            self.receipt.optimizer_step,
            self.receipt.combined_weights_sha256,
        )

    @property
    def run_identity_sha256(self) -> str:
        return self.receipt.run_identity_sha256

    @property
    def model_path(self) -> Path:
        return Path(self.receipt.qwen_model_path)


def _qwen_tree_sha256(files: Sequence[PairedModelFile]) -> str:
    return _canonical_sha256(
        {
            "schema": "tgvf.paired-qwen-tree.v1",
            "files": [item.as_record() for item in sorted(files, key=lambda x: x.relative_path)],
        }
    )


def _scan_qwen_model(model_path: Path) -> tuple[PairedModelFile, ...]:
    root = model_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("paired Qwen model path is not a directory")
    files: list[PairedModelFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"paired Qwen tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"paired Qwen tree contains a special file: {path}")
        files.append(
            PairedModelFile(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    _validate_qwen_weight_index(root)
    return tuple(files)


def _validate_qwen_weight_index(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError("paired Qwen model lacks a regular safetensors index")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("paired Qwen safetensors index has no weight map")
    names = tuple(str(name) for name in weight_map)
    if any("tgvf_adapter" in name for name in names):
        raise ValueError("paired Qwen model illegally embeds RP66-owned tensors")
    if not any("visual" in name for name in names) or not any(
        "language_model" in name for name in names
    ):
        raise ValueError("paired Qwen model lacks vision/language parameter families")


def _copy_runtime_snapshot_closure(
    snapshot: PolicyLoRASnapshot, *, destination: Path
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    state_root = snapshot.pointer_file.parent
    manifest_relative = snapshot.manifest_file.relative_to(state_root)
    tensor_relative = snapshot.tensor_file.relative_to(state_root)
    for relative, payload in (
        (tensor_relative, snapshot.tensor_bytes),
        (manifest_relative, snapshot.manifest_bytes),
    ):
        target = destination / relative
        _write_immutable(target, payload)
    pointer = destination / f"step-{snapshot.policy_version.optimizer_step:08d}-pointer.json"
    _write_immutable(pointer, snapshot.pointer_bytes)
    return pointer


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable paired snapshot collision: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_paired_tgvf_snapshot(
    *,
    policy_config_path: str | Path,
    optimizer_step: int,
    qwen_model_path: str | Path,
    receipt_path: str | Path,
    rp66_pointer_path: str | Path | None = None,
) -> PairedTGVFSnapshotReceipt:
    """Hash and freeze one same-step Qwen/RP66 evaluation closure."""

    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    config_path = Path(policy_config_path).resolve(strict=True)
    run = load_policy_e2e_smoke_run_config(
        config_path, allow_external_agent_loop_config=True
    )
    model_path = Path(qwen_model_path).resolve(strict=True)
    qwen_files = _scan_qwen_model(model_path)
    qwen_tree_sha256 = _qwen_tree_sha256(qwen_files)
    destination = Path(receipt_path).resolve()

    if optimizer_step == 0:
        if model_path != Path(run.model.revision_or_path).resolve(strict=True):
            raise ValueError("step-zero paired Qwen must be the run's base model")
        if rp66_pointer_path is not None:
            raise ValueError("step-zero paired snapshot forbids an RP66 pointer")
        export = load_rank_zero_adapter_owned_state_export(
            run.representation.artifact_path
        )
        if export.state is None:
            raise RuntimeError("stage1 RP66 artifact omitted Adapter-owned state")
        if state_digest(export.manifest) != run.representation.artifact.sha256:
            raise ValueError("stage1 RP66 manifest identity differs")
        rp66_tensors = export.state
        rp66_kind: RP66SnapshotKind = "stage1_artifact"
        rp66_storage_sha256 = run.representation.artifact_file_sha256
        frozen_pointer = None
        pointer_sha256 = None
    else:
        if rp66_pointer_path is None:
            raise ValueError("nonzero paired snapshot requires an RP66 pointer")
        pointer = Path(rp66_pointer_path).resolve(strict=True)
        state = PolicyWeightSyncState(
            directory=pointer.parent,
            run_id=run.run_id,
            run_identity_sha256=run.identity_sha256,
        )
        runtime_snapshot = load_lora_snapshot_pointer(
            state,
            pointer_path=pointer,
            expected_optimizer_step=optimizer_step,
        )
        frozen_pointer = _copy_runtime_snapshot_closure(
            runtime_snapshot, destination=destination.parent / "rp66-state"
        )
        rp66_tensors = runtime_snapshot.tensors
        rp66_kind = "runtime_snapshot"
        rp66_storage_sha256 = runtime_snapshot.policy_version.weights_sha256
        pointer_sha256 = runtime_snapshot.pointer_file_sha256

    rp66_state_sha256 = adapter_owned_state_sha256(rp66_tensors)
    combined_weights_sha256 = _canonical_sha256(
        {
            "schema": "tgvf.paired-qwen-rp66-weights.v1",
            "optimizer_step": optimizer_step,
            "qwen_tree_sha256": qwen_tree_sha256,
            "rp66_state_sha256": rp66_state_sha256,
        }
    )
    values: dict[str, Any] = {
        "run_id": run.run_id,
        "run_identity_sha256": run.identity_sha256,
        "policy_config_path": str(config_path),
        "policy_config_sha256": _sha256_file(config_path),
        "optimizer_step": optimizer_step,
        "qwen_model_path": str(model_path),
        "qwen_files": qwen_files,
        "qwen_tree_sha256": qwen_tree_sha256,
        "rp66_kind": rp66_kind,
        "rp66_state_sha256": rp66_state_sha256,
        "rp66_storage_sha256": rp66_storage_sha256,
        "rp66_pointer_path": None if frozen_pointer is None else str(frozen_pointer),
        "rp66_pointer_sha256": pointer_sha256,
        "combined_weights_sha256": combined_weights_sha256,
        "identity_sha256": "0" * 64,
    }
    provisional = PairedTGVFSnapshotReceipt.__new__(PairedTGVFSnapshotReceipt)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "schema_version", PAIRED_TGVF_SNAPSHOT_SCHEMA)
    identity = _canonical_sha256(provisional.as_record(include_identity=False))
    values["identity_sha256"] = identity
    receipt = PairedTGVFSnapshotReceipt(**values)
    encoded = (
        json.dumps(receipt.as_record(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _write_immutable(destination, encoded)
    return receipt


def load_paired_tgvf_snapshot(
    receipt_path: str | Path, *, runtime_lightweight: bool = True
) -> PairedTGVFEvaluationSnapshot:
    """Load a paired closure; large Qwen bytes are not rehashed at runtime."""

    path = Path(receipt_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("paired TGVF receipt must be a JSON object")
    expected_fields = {
        "schema_version", "run_id", "run_identity_sha256", "policy_config_path",
        "policy_config_sha256", "optimizer_step", "qwen_model_path", "qwen_files",
        "qwen_tree_sha256", "rp66_kind", "rp66_state_sha256",
        "rp66_storage_sha256", "rp66_pointer_path", "rp66_pointer_sha256",
        "combined_weights_sha256", "identity_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("paired TGVF receipt fields differ")
    raw_files = payload["qwen_files"]
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise ValueError("paired TGVF Qwen files must be a sequence")
    files = tuple(PairedModelFile(**dict(item)) for item in raw_files)
    receipt = PairedTGVFSnapshotReceipt(
        **{**dict(payload), "qwen_files": files}
    )
    config_path = Path(receipt.policy_config_path).resolve(strict=True)
    if _sha256_file(config_path) != receipt.policy_config_sha256:
        raise ValueError("paired TGVF policy config bytes changed")
    run = load_policy_e2e_smoke_run_config(
        config_path, allow_external_agent_loop_config=True
    )
    model_root = Path(receipt.qwen_model_path).resolve(strict=True)
    for item in receipt.qwen_files:
        candidate = model_root / item.relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("paired Qwen closure file is absent or not regular")
        if candidate.stat().st_size != item.size_bytes:
            raise ValueError("paired Qwen closure file size changed")
        if not runtime_lightweight or candidate.suffix.casefold() not in _LARGE_WEIGHT_SUFFIXES:
            if _sha256_file(candidate) != item.sha256:
                raise ValueError("paired Qwen closure file bytes changed")
    _validate_qwen_weight_index(model_root)

    if receipt.rp66_kind == "stage1_artifact":
        if receipt.rp66_storage_sha256 != run.representation.artifact_file_sha256:
            raise ValueError("paired stage1 RP66 storage identity changed")
        export = load_rank_zero_adapter_owned_state_export(
            run.representation.artifact_path
        )
        if export.state is None:
            raise RuntimeError("stage1 RP66 artifact omitted Adapter-owned state")
        if state_digest(export.manifest) != run.representation.artifact.sha256:
            raise ValueError("paired stage1 RP66 manifest identity changed")
        rp66_tensors = export.state
    else:
        assert receipt.rp66_pointer_path is not None
        pointer = Path(receipt.rp66_pointer_path).resolve(strict=True)
        state = PolicyWeightSyncState(
            directory=pointer.parent,
            run_id=run.run_id,
            run_identity_sha256=run.identity_sha256,
        )
        snapshot = load_lora_snapshot_pointer(
            state,
            pointer_path=pointer,
            expected_pointer_file_sha256=receipt.rp66_pointer_sha256,
            expected_optimizer_step=receipt.optimizer_step,
        )
        if snapshot.policy_version.weights_sha256 != receipt.rp66_storage_sha256:
            raise ValueError("paired RP66 storage identity changed")
        rp66_tensors = snapshot.tensors
    return PairedTGVFEvaluationSnapshot(
        run=run, receipt=receipt, rp66_tensors=rp66_tensors
    )


def paired_tgvf_snapshot_identity_record(
    snapshot: PairedTGVFEvaluationSnapshot,
) -> dict[str, object]:
    return {
        "snapshot_backend": PAIRED_TGVF_EVALUATION_BACKEND,
        "run_id": snapshot.policy_version.run_id,
        "run_identity_sha256": snapshot.run_identity_sha256,
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "snapshot_identity_sha256": snapshot.receipt.identity_sha256,
        "qwen_tree_sha256": snapshot.receipt.qwen_tree_sha256,
        "rp66_state_sha256": snapshot.receipt.rp66_state_sha256,
        "rp66_storage_sha256": snapshot.receipt.rp66_storage_sha256,
        "rp66_kind": snapshot.receipt.rp66_kind,
        "lora_request": None,
    }


__all__ = [
    "PAIRED_TGVF_EVALUATION_BACKEND",
    "PAIRED_TGVF_SNAPSHOT_SCHEMA",
    "PairedTGVFEvaluationSnapshot",
    "PairedTGVFSnapshotReceipt",
    "load_paired_tgvf_snapshot",
    "materialize_paired_tgvf_snapshot",
    "paired_tgvf_snapshot_identity_record",
]
