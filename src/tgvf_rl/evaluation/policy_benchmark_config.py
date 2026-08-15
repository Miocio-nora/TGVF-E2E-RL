"""Create an exact-snapshot configuration for a generic policy benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyWeightSyncState,
    load_lora_snapshot_pointer,
)
from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

from .policy_coredev import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    FULL_MODEL_EVALUATION_BACKEND,
    TRAINING_RUN_EVALUATION_PROTOCOL,
    POLICY_BENCHMARK_SCHEMA,
    PolicyCoreDevConfig,
    load_benchmark_tasks,
)
from .policy_full_model_snapshot import load_full_model_evaluation_snapshot
from .policy_paired_tgvf_snapshot import (
    PAIRED_TGVF_EVALUATION_BACKEND,
    materialize_paired_tgvf_snapshot,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("policy benchmark config output is not a regular file")
        if path.read_bytes() != payload:
            raise RuntimeError("immutable policy benchmark config differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError("immutable policy benchmark config differs")
    finally:
        temporary.unlink(missing_ok=True)


def materialize_policy_benchmark_config(
    *,
    evaluation_id: str,
    policy_config_path: str | Path,
    lora_pointer_path: str | Path,
    expected_optimizer_step: int,
    task_manifest_path: str | Path,
    expected_task_count: int,
    expected_single_image_count: int,
    output_root: str | Path,
    config_path: str | Path,
    inference_concurrency_per_gpu: int = 8,
    max_model_len: int = 32768,
    max_num_batched_tokens: int = 32768,
    enable_chunked_prefill: bool = False,
    gpu_memory_utilization: float = 0.9,
    image_max_pixels: int | None = None,
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3),
    evaluation_protocol: str = TRAINING_RUN_EVALUATION_PROTOCOL,
) -> dict[str, Any]:
    """Strictly inspect one pointer closure and write a resumable config.

    The resulting file binds the pointer bytes, run identity, optimizer step,
    content-level LoRA weights, and the full task manifest. If ``latest`` is
    advanced after this command, evaluator preparation fails rather than
    silently evaluating another checkpoint.
    """

    policy_config = Path(policy_config_path).resolve()
    pointer = Path(lora_pointer_path)
    if not pointer.is_absolute():
        raise ValueError("LoRA pointer path must be absolute")
    task_manifest = Path(task_manifest_path).resolve()
    evaluation_output = Path(output_root).resolve()
    destination = Path(config_path).resolve()
    run = load_policy_e2e_smoke_run_config(
        policy_config, allow_external_agent_loop_config=True
    )
    frozen_policy_config = evaluation_output / "runtime" / "frozen-policy-config.toml"
    _write_immutable(frozen_policy_config, policy_config.read_bytes())
    state = PolicyWeightSyncState(
        directory=pointer.parent,
        run_id=run.run_id,
        run_identity_sha256=run.identity_sha256,
    )
    snapshot = load_lora_snapshot_pointer(
        state,
        pointer_path=pointer,
        expected_optimizer_step=expected_optimizer_step,
    )
    task_sha256 = _sha256_file(task_manifest)
    load_benchmark_tasks(
        task_manifest,
        expected_task_count=expected_task_count,
        expected_single_image_count=expected_single_image_count,
        expected_sha256=task_sha256,
        verify_image_contents=True,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    payload: dict[str, Any] = {
        "schema_version": POLICY_BENCHMARK_SCHEMA,
        "evaluation_id": evaluation_id,
        "policy_config_path": str(frozen_policy_config),
        "lora_pointer_path": str(pointer),
        "lora_pointer_sha256": snapshot.pointer_file_sha256,
        "expected_policy_run_id": snapshot.policy_version.run_id,
        "expected_policy_run_identity_sha256": snapshot.run_identity_sha256,
        "expected_optimizer_step": snapshot.policy_version.optimizer_step,
        "expected_policy_weights_sha256": snapshot.policy_version.weights_sha256,
        "output_root": str(evaluation_output),
        "gpu_ids": list(gpu_ids),
        "inference_concurrency_per_gpu": inference_concurrency_per_gpu,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": enable_chunked_prefill,
        "gpu_memory_utilization": gpu_memory_utilization,
        "evaluation_protocol": evaluation_protocol,
        "task_manifest_path": str(task_manifest),
        "task_manifest_sha256": task_sha256,
        "expected_task_count": expected_task_count,
        "expected_single_image_count": expected_single_image_count,
    }
    if image_max_pixels is not None:
        payload["image_max_pixels"] = image_max_pixels
    PolicyCoreDevConfig(**payload)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(destination, encoded)
    return payload


def materialize_full_model_policy_benchmark_config(
    *,
    evaluation_id: str,
    policy_config_path: str | Path,
    snapshot_manifest_path: str | Path,
    materialization_receipt_path: str | Path,
    expected_optimizer_step: int,
    task_manifest_path: str | Path,
    expected_task_count: int,
    expected_single_image_count: int,
    output_root: str | Path,
    config_path: str | Path,
    inference_concurrency_per_gpu: int = 8,
    max_model_len: int = 32768,
    max_num_batched_tokens: int = 32768,
    enable_chunked_prefill: bool = False,
    gpu_memory_utilization: float = 0.9,
    image_max_pixels: int | None = None,
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3),
    paired_seed_namespace: str | None = None,
) -> dict[str, Any]:
    """Bind an official-visible suite to one exact standalone full model.

    The manifest binds the source checkpoint and run contract; the receipt
    binds the directly loadable Hugging Face tree.  Both record files and the
    manifest identity are copied into the evaluator config so a changed or
    accidentally selected checkpoint fails before vLLM construction.
    """

    policy_config = Path(policy_config_path).resolve()
    snapshot_manifest = Path(snapshot_manifest_path).resolve()
    materialization_receipt = Path(materialization_receipt_path).resolve()
    task_manifest = Path(task_manifest_path).resolve()
    evaluation_output = Path(output_root).resolve()
    destination = Path(config_path).resolve()

    snapshot_manifest_sha256 = _sha256_file(snapshot_manifest)
    materialization_receipt_sha256 = _sha256_file(materialization_receipt)
    snapshot = load_full_model_evaluation_snapshot(
        snapshot_manifest,
        materialization_receipt,
    )
    if snapshot.policy_version.optimizer_step != expected_optimizer_step:
        raise ValueError("full-model optimizer step differs from requested step")
    expected_policy_config_sha256 = (
        snapshot.manifest.run_contract_file_sha256
        if getattr(snapshot.manifest, "checkpoint_owner", None) is None
        else snapshot.manifest.checkpoint_owner.config_file_sha256
    )
    if _sha256_file(policy_config) != expected_policy_config_sha256:
        raise ValueError("policy config bytes differ from full-model checkpoint owner")

    task_sha256 = _sha256_file(task_manifest)
    load_benchmark_tasks(
        task_manifest,
        expected_task_count=expected_task_count,
        expected_single_image_count=expected_single_image_count,
        expected_sha256=task_sha256,
        verify_image_contents=True,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    frozen_policy_config = evaluation_output / "runtime" / "frozen-policy-config.toml"
    payload: dict[str, Any] = {
        "schema_version": POLICY_BENCHMARK_SCHEMA,
        "evaluation_id": evaluation_id,
        "policy_config_path": str(frozen_policy_config),
        "lora_pointer_path": None,
        "snapshot_backend": FULL_MODEL_EVALUATION_BACKEND,
        "full_model_snapshot_manifest_path": str(snapshot_manifest),
        "full_model_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "full_model_materialization_receipt_path": str(materialization_receipt),
        "full_model_materialization_receipt_sha256": (materialization_receipt_sha256),
        "required_snapshot_identity_sha256": snapshot.manifest.identity_sha256,
        "expected_policy_run_id": snapshot.policy_version.run_id,
        "expected_policy_run_identity_sha256": snapshot.run_identity_sha256,
        "expected_optimizer_step": snapshot.policy_version.optimizer_step,
        "expected_policy_weights_sha256": snapshot.policy_version.weights_sha256,
        "output_root": str(evaluation_output),
        "gpu_ids": list(gpu_ids),
        "inference_concurrency_per_gpu": inference_concurrency_per_gpu,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": enable_chunked_prefill,
        "gpu_memory_utilization": gpu_memory_utilization,
        "evaluation_protocol": DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        "task_manifest_path": str(task_manifest),
        "task_manifest_sha256": task_sha256,
        "expected_task_count": expected_task_count,
        "expected_single_image_count": expected_single_image_count,
    }
    if image_max_pixels is not None:
        payload["image_max_pixels"] = image_max_pixels
    if paired_seed_namespace is not None:
        payload["paired_seed_namespace"] = paired_seed_namespace
    PolicyCoreDevConfig(**payload)
    _write_immutable(frozen_policy_config, policy_config.read_bytes())
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(destination, encoded)
    return payload


def materialize_paired_tgvf_policy_benchmark_config(
    *,
    evaluation_id: str,
    policy_config_path: str | Path,
    optimizer_step: int,
    qwen_model_path: str | Path,
    rp66_pointer_path: str | Path | None,
    paired_snapshot_receipt_path: str | Path,
    task_manifest_path: str | Path,
    expected_task_count: int,
    expected_single_image_count: int,
    output_root: str | Path,
    config_path: str | Path,
    inference_concurrency_per_gpu: int = 8,
    max_model_len: int = 32768,
    max_num_batched_tokens: int = 32768,
    enable_chunked_prefill: bool = False,
    gpu_memory_utilization: float = 0.9,
    image_max_pixels: int | None = None,
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3),
    paired_seed_namespace: str | None = None,
) -> dict[str, Any]:
    """Bind one same-step full-Qwen/RP66 pair to a benchmark arm."""

    policy_config = Path(policy_config_path).resolve()
    task_manifest = Path(task_manifest_path).resolve()
    evaluation_output = Path(output_root).resolve()
    destination = Path(config_path).resolve()
    receipt_path = Path(paired_snapshot_receipt_path).resolve()
    receipt = materialize_paired_tgvf_snapshot(
        policy_config_path=policy_config,
        optimizer_step=optimizer_step,
        qwen_model_path=qwen_model_path,
        receipt_path=receipt_path,
        rp66_pointer_path=rp66_pointer_path,
    )
    receipt_sha256 = _sha256_file(receipt_path)
    task_sha256 = _sha256_file(task_manifest)
    load_benchmark_tasks(
        task_manifest,
        expected_task_count=expected_task_count,
        expected_single_image_count=expected_single_image_count,
        expected_sha256=task_sha256,
        verify_image_contents=True,
        require_explicit_sample_ids=True,
        require_image_identities=True,
    )
    frozen_policy_config = evaluation_output / "runtime" / "frozen-policy-config.toml"
    _write_immutable(frozen_policy_config, policy_config.read_bytes())
    payload: dict[str, Any] = {
        "schema_version": POLICY_BENCHMARK_SCHEMA,
        "evaluation_id": evaluation_id,
        "policy_config_path": str(frozen_policy_config),
        "lora_pointer_path": None,
        "snapshot_backend": PAIRED_TGVF_EVALUATION_BACKEND,
        "paired_snapshot_receipt_path": str(receipt_path),
        "paired_snapshot_receipt_sha256": receipt_sha256,
        "expected_policy_run_id": receipt.run_id,
        "expected_policy_run_identity_sha256": receipt.run_identity_sha256,
        "expected_optimizer_step": receipt.optimizer_step,
        "expected_policy_weights_sha256": receipt.combined_weights_sha256,
        "output_root": str(evaluation_output),
        "gpu_ids": list(gpu_ids),
        "inference_concurrency_per_gpu": inference_concurrency_per_gpu,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": enable_chunked_prefill,
        "gpu_memory_utilization": gpu_memory_utilization,
        "evaluation_protocol": TRAINING_RUN_EVALUATION_PROTOCOL,
        "task_manifest_path": str(task_manifest),
        "task_manifest_sha256": task_sha256,
        "expected_task_count": expected_task_count,
        "expected_single_image_count": expected_single_image_count,
    }
    if image_max_pixels is not None:
        payload["image_max_pixels"] = image_max_pixels
    if paired_seed_namespace is not None:
        payload["paired_seed_namespace"] = paired_seed_namespace
    PolicyCoreDevConfig(**payload)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(destination, encoded)
    return payload


__all__ = [
    "materialize_full_model_policy_benchmark_config",
    "materialize_paired_tgvf_policy_benchmark_config",
    "materialize_policy_benchmark_config",
]
