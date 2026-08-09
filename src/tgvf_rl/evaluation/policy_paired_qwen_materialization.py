"""Materialize the Qwen-only member of a trainable-RP66 FSDP checkpoint.

The trainable TGVF actor checkpoint intentionally contains both the full Qwen
state and the RP66 Adapter state.  Evaluation loads RP66 through its separate
content-addressed runtime snapshot, so feeding a stock veRL merge directly to
vLLM would embed RP66 twice.  This module wraps the upstream FSDP merger and
removes exactly the Adapter-owned namespace before Hugging Face serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyWeightSyncState,
    load_lora_snapshot_pointer,
)
from tgvf_rl.framework.verl.checkpoint_bridge import (
    PolicyPilotVerlCheckpointPair,
)
from tgvf_rl.policy.checkpoint import PilotProjectCheckpointState
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config


QWEN_ONLY_MATERIALIZATION_SCHEMA = "tgvf.prl15-qwen-only-materialization.v1"
_ADAPTER_PREFIX = "tgvf_adapter."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _model_index_keys(model_path: Path) -> frozenset[str]:
    index = model_path / "model.safetensors.index.json"
    if not index.is_file() or index.is_symlink():
        raise ValueError("Qwen model lacks a regular safetensors index")
    payload = json.loads(index.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Qwen safetensors index has no weight map")
    if any(type(key) is not str for key in weight_map):
        raise TypeError("Qwen safetensors index keys must be text")
    return frozenset(weight_map)


def _partition_checkpoint_keys(
    keys: object,
    *,
    expected_qwen_keys: frozenset[str],
    expected_adapter_keys: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    if not isinstance(keys, (set, frozenset, list, tuple)) or any(
        type(key) is not str for key in keys
    ):
        raise TypeError("checkpoint state keys must be a string collection")
    actual = frozenset(keys)
    adapter = frozenset(
        key.removeprefix(_ADAPTER_PREFIX)
        for key in actual
        if key.startswith(_ADAPTER_PREFIX)
    )
    qwen = frozenset(key for key in actual if not key.startswith(_ADAPTER_PREFIX))
    if qwen != expected_qwen_keys:
        raise ValueError("FSDP checkpoint Qwen keys differ from the bound base model")
    if adapter != expected_adapter_keys:
        raise ValueError("FSDP checkpoint RP66 keys differ from the paired snapshot")
    return qwen, adapter


def _take_qwen_only_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    expected_qwen_keys: frozenset[str],
    expected_adapter_keys: frozenset[str],
) -> dict[str, torch.Tensor]:
    _partition_checkpoint_keys(
        set(state_dict),
        expected_qwen_keys=expected_qwen_keys,
        expected_adapter_keys=expected_adapter_keys,
    )
    for key in tuple(state_dict):
        if key.startswith(_ADAPTER_PREFIX):
            del state_dict[key]
    if frozenset(state_dict) != expected_qwen_keys:
        raise RuntimeError("Qwen-only state extraction changed the parameter closure")
    return state_dict


def _scan_model_tree(model_path: Path) -> tuple[dict[str, object], ...]:
    root = model_path.resolve(strict=True)
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"materialized Qwen tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"materialized Qwen tree contains a special file: {path}")
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError("materialized Qwen tree is empty")
    return tuple(records)


def _load_json(path: Path, *, owner: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{owner} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{owner} must be a JSON object")
    return payload


def _checkpoint_pair_record(checkpoint: Path, *, optimizer_step: int) -> dict[str, str]:
    actor = checkpoint / "actor"
    pair_path = actor / "tgvf_policy_checkpoint_pair.json"
    state_path = actor / "tgvf_policy_project_state.json"
    pair = PolicyPilotVerlCheckpointPair.from_checkpoint_mapping(
        _load_json(pair_path, owner="paired checkpoint marker")
    )
    state = PilotProjectCheckpointState.from_checkpoint_mapping(
        _load_json(state_path, owner="paired project state")
    )
    if pair.optimizer_step != optimizer_step:
        raise ValueError("paired checkpoint marker step differs")
    if state.progress.optimizer_step != optimizer_step:
        raise ValueError("paired project state step differs")
    if pair.run_id != state.run_identity.run_id:
        raise ValueError("paired checkpoint run_id differs")
    if pair.project_state_sha256 != state.integrity_sha256:
        raise ValueError("paired checkpoint project-state identity differs")
    return {
        "run_id": pair.run_id,
        "pair_integrity_sha256": pair.integrity_sha256,
        "project_state_sha256": pair.project_state_sha256,
        "policy_weights_sha256": state.policy_version.weights_sha256,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_materialized_model(
    model_path: Path, *, expected_qwen_keys: frozenset[str]
) -> tuple[dict[str, object], ...]:
    actual = _model_index_keys(model_path)
    if actual != expected_qwen_keys:
        raise ValueError("materialized Qwen index differs from the base parameter closure")
    if any(key.startswith(_ADAPTER_PREFIX) for key in actual):
        raise ValueError("materialized Qwen model still embeds RP66 parameters")
    return _scan_model_tree(model_path)


def _merge_qwen_only(
    *,
    actor_path: Path,
    target_path: Path,
    expected_qwen_keys: frozenset[str],
    expected_adapter_keys: frozenset[str],
) -> None:
    # Keep the upstream dependency lazy: status/wait modes must not import the
    # merger or allocate tensors.
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    class QwenOnlyFSDPModelMerger(FSDPModelMerger):
        def save_hf_model_and_tokenizer(
            self, state_dict: dict[str, torch.Tensor]
        ) -> None:
            qwen_only = _take_qwen_only_state_dict(
                state_dict,
                expected_qwen_keys=expected_qwen_keys,
                expected_adapter_keys=expected_adapter_keys,
            )
            super().save_hf_model_and_tokenizer(qwen_only)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(target_path),
        local_dir=str(actor_path),
        hf_model_config_path=str(actor_path / "huggingface"),
        use_cpu_initialization=True,
    )
    merger = QwenOnlyFSDPModelMerger(config)
    try:
        merger.merge_and_save()
    finally:
        merger.cleanup()


def materialize_qwen_only_policy_checkpoint(
    *,
    policy_config_path: str | Path,
    optimizer_step: int,
    checkpoint_path: str | Path,
    rp66_pointer_path: str | Path,
    bundle_path: str | Path,
) -> Path:
    """Return an atomic, resumable Qwen-only HF model directory."""

    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise ValueError("Qwen-only materialization requires a positive step")
    config_path = Path(policy_config_path).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if checkpoint.name != f"global_step_{optimizer_step}":
        raise ValueError("Qwen-only checkpoint path and optimizer step differ")
    actor = checkpoint / "actor"
    if actor.is_symlink() or not actor.is_dir():
        raise ValueError("Qwen-only checkpoint lacks a regular actor directory")
    hf_config = actor / "huggingface/config.json"
    if hf_config.is_symlink() or not hf_config.is_file():
        raise ValueError("Qwen-only checkpoint lacks its Hugging Face config")
    bundle = Path(bundle_path).resolve()
    if checkpoint == bundle or checkpoint in bundle.parents or bundle in checkpoint.parents:
        raise ValueError("Qwen-only evaluation bundle must be outside checkpoint storage")

    run = load_policy_e2e_smoke_run_config(
        config_path, allow_external_agent_loop_config=True
    )
    pair = _checkpoint_pair_record(checkpoint, optimizer_step=optimizer_step)
    if pair["run_id"] != run.run_id:
        raise ValueError("Qwen-only checkpoint belongs to another run")

    pointer = Path(rp66_pointer_path).resolve(strict=True)
    snapshot = load_lora_snapshot_pointer(
        PolicyWeightSyncState(
            directory=pointer.parent,
            run_id=run.run_id,
            run_identity_sha256=run.identity_sha256,
        ),
        pointer_path=pointer,
        expected_optimizer_step=optimizer_step,
    )
    if snapshot.policy_version.weights_sha256 != pair["policy_weights_sha256"]:
        raise ValueError("checkpoint and RP66 pointer weight identities differ")

    base_model = Path(run.model.revision_or_path).resolve(strict=True)
    expected_qwen_keys = _model_index_keys(base_model)
    expected_adapter_keys = frozenset(snapshot.tensors)
    rank_zero = actor / f"model_world_size_{run.distributed.world_size}_rank_0.pt"
    state = torch.load(rank_zero, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("rank-zero FSDP model state must be a mapping")
    _partition_checkpoint_keys(
        set(state),
        expected_qwen_keys=expected_qwen_keys,
        expected_adapter_keys=expected_adapter_keys,
    )
    del state

    identity = {
        "schema_version": QWEN_ONLY_MATERIALIZATION_SCHEMA,
        "run_id": run.run_id,
        "run_identity_sha256": run.identity_sha256,
        "optimizer_step": optimizer_step,
        "checkpoint_path": str(checkpoint),
        "pair_integrity_sha256": pair["pair_integrity_sha256"],
        "project_state_sha256": pair["project_state_sha256"],
        "policy_weights_sha256": pair["policy_weights_sha256"],
        "rp66_pointer_sha256": _sha256_file(pointer),
        "base_model_path": str(base_model),
        "base_model_index_sha256": _sha256_file(
            base_model / "model.safetensors.index.json"
        ),
        "qwen_parameter_keys_sha256": _canonical_sha256(sorted(expected_qwen_keys)),
        "qwen_parameter_key_count": len(expected_qwen_keys),
        "rp66_parameter_keys_sha256": _canonical_sha256(
            sorted(expected_adapter_keys)
        ),
        "rp66_parameter_key_count": len(expected_adapter_keys),
    }
    receipt_name = "materialization-receipt.json"
    if bundle.is_dir() and not bundle.is_symlink():
        receipt = _load_json(bundle / receipt_name, owner="Qwen-only receipt")
        model = bundle / "model"
        model_files = _validate_materialized_model(
            model, expected_qwen_keys=expected_qwen_keys
        )
        expected_receipt = {
            **identity,
            "model_files": list(model_files),
            "model_tree_sha256": _canonical_sha256(model_files),
        }
        if dict(receipt) != expected_receipt:
            raise RuntimeError("existing Qwen-only evaluation bundle identity differs")
        return model
    if bundle.exists() or bundle.is_symlink():
        raise RuntimeError("Qwen-only evaluation bundle path is not a directory")

    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle.name}.", suffix=".partial", dir=bundle.parent
        )
    )
    try:
        model = temporary / "model"
        _merge_qwen_only(
            actor_path=actor,
            target_path=model,
            expected_qwen_keys=expected_qwen_keys,
            expected_adapter_keys=expected_adapter_keys,
        )
        model_files = _validate_materialized_model(
            model, expected_qwen_keys=expected_qwen_keys
        )
        receipt = {
            **identity,
            "model_files": list(model_files),
            "model_tree_sha256": _canonical_sha256(model_files),
        }
        _write_json(temporary / receipt_name, receipt)
        os.rename(temporary, bundle)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return bundle / "model"


__all__ = [
    "QWEN_ONLY_MATERIALIZATION_SCHEMA",
    "materialize_qwen_only_policy_checkpoint",
]
