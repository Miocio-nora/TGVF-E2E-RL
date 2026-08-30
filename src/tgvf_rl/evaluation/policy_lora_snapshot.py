"""LoRA snapshot and vLLM adapter boundary for policy evaluation.

This module owns the LoRA-specific half of standalone policy evaluation: the
strict in-memory snapshot value, the content-addressed PEFT directory exposed
to vLLM, its pre-consumption integrity verifier, and the LoRA engine contract.
It intentionally has no dependency on either the evaluator facade or the
full-model snapshot backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.framework.verl.policy_weight_sync import PolicyLoRASnapshot
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_VLLM_WORKER_EXTENSION_FQN,
)
from tgvf_rl.framework.vllm.registration import (
    TGVF_QWEN3_VLLM_ARCHITECTURE,
    TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
)
from tgvf_rl.observations.store import tensor_checksum
from tgvf_rl.policy.run_config import PolicyE2ESmokeRunConfig
from tgvf_rl.public_api_compat import (
    rebind_public_class,
    rebind_public_function,
)

from .policy_evaluation_config import (
    DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
    PolicyCoreDevConfig,
    _require_sha256,
)
from .policy_evaluation_identity import (
    canonical_json_sha256 as _canonical_json_sha256,
)
from .policy_vllm_manager import AdapterIntegrityVerifier


VLLM_LORA_ADAPTER_SCHEMA = "tgvf-policy-vllm-lora-adapter-v2"
VLLM_LORA_ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"
VLLM_LORA_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
VLLM_LORA_ADAPTER_IDENTITY_FILENAME = "identity.json"
VLLM_LORA_ENGINE_ATTESTATION = "unavailable-in-vllm-0.12-public-api"
VLLM_LORA_RESIDUAL_RACE = (
    "same-UID mutation between the final pre-generate verification and "
    "vLLM's lazy adapter file read remains outside the public API boundary"
)


@dataclass(frozen=True, slots=True)
class PolicyEvaluationSnapshot:
    """One strictly loaded policy snapshot reused for the whole process."""

    run: PolicyE2ESmokeRunConfig
    lora: PolicyLoRASnapshot

    def __post_init__(self) -> None:
        if self.lora.policy_version.run_id != self.run.run_id:
            raise ValueError("policy snapshot run_id differs from policy config")
        if self.lora.run_identity_sha256 != self.run.identity_sha256:
            raise ValueError("policy snapshot run identity differs from policy config")

    @property
    def policy_version(self) -> PolicyVersion:
        return self.lora.policy_version


@dataclass(frozen=True, slots=True)
class VLLMLoRAAdapterIntegrityVerifier(AdapterIntegrityVerifier):
    """Re-read the exact private PEFT closure before vLLM can consume it.

    vLLM 0.12 does not expose a digest receipt for the adapter bytes loaded by
    the engine.  The verifier therefore binds the request path and checks all
    three files immediately before every ``generate``.  A same-UID writer can
    still race the small interval between that check and vLLM's lazy file read;
    ``residual_race`` records that limitation instead of claiming a sealed
    engine-side identity.
    """

    adapter_root: Path
    materialization_identity_sha256: str
    root_device: int
    root_inode: int
    adapter_model_bytes: bytes
    adapter_model_sha256: str
    adapter_config_bytes: bytes
    adapter_config_sha256: str
    identity_bytes: bytes
    identity_sha256: str
    engine_loaded_identity_attestation: str = VLLM_LORA_ENGINE_ATTESTATION
    residual_race: str = VLLM_LORA_RESIDUAL_RACE

    def __post_init__(self) -> None:
        root = Path(self.adapter_root)
        if not root.is_absolute():
            raise ValueError("vLLM LoRA adapter root must be absolute")
        object.__setattr__(
            self,
            "adapter_root",
            Path(os.path.abspath(os.fspath(root))),
        )
        _require_sha256(
            self.materialization_identity_sha256,
            name="vLLM LoRA materialization identity",
        )
        for name, payload, digest in self._expected_files():
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"vLLM LoRA {name} bytes must be non-empty")
            _require_sha256(digest, name=f"vLLM LoRA {name} digest")
            if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest):
                raise ValueError(f"vLLM LoRA {name} digest differs from bytes")
        if self.root_device < 0 or self.root_inode <= 0:
            raise ValueError("vLLM LoRA root filesystem identity is invalid")
        if self.engine_loaded_identity_attestation != VLLM_LORA_ENGINE_ATTESTATION:
            raise ValueError("vLLM LoRA engine attestation statement changed")
        if self.residual_race != VLLM_LORA_RESIDUAL_RACE:
            raise ValueError("vLLM LoRA residual-race statement changed")

    def _expected_files(self) -> tuple[tuple[str, bytes, str], ...]:
        return (
            (
                VLLM_LORA_ADAPTER_MODEL_FILENAME,
                self.adapter_model_bytes,
                self.adapter_model_sha256,
            ),
            (
                VLLM_LORA_ADAPTER_CONFIG_FILENAME,
                self.adapter_config_bytes,
                self.adapter_config_sha256,
            ),
            (
                VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
                self.identity_bytes,
                self.identity_sha256,
            ),
        )

    def verify(self, *, phase: str) -> None:
        if not isinstance(phase, str) or not phase:
            raise ValueError("vLLM LoRA verification phase must be non-empty")
        descriptor = _open_vllm_lora_adapter_root(
            self.adapter_root,
            owner=f"vLLM LoRA adapter during {phase}",
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                metadata.st_dev != self.root_device
                or metadata.st_ino != self.root_inode
            ):
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter root changed during {phase}"
                )
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter root is not private during {phase}"
                )
            names = set(os.listdir(descriptor))
            expected_names = {
                name for name, _payload, _digest in self._expected_files()
            }
            if names != expected_names:
                raise ReplayMismatchError(
                    f"vLLM LoRA adapter files changed during {phase}"
                )
            for name, expected_bytes, expected_digest in self._expected_files():
                observed = _read_private_vllm_lora_file_at(
                    descriptor,
                    name,
                    phase=phase,
                )
                observed_digest = hashlib.sha256(observed).hexdigest()
                if not hmac.compare_digest(observed_digest, expected_digest):
                    raise ReplayMismatchError(
                        f"vLLM LoRA {name} digest changed during {phase}"
                    )
                if observed != expected_bytes:
                    raise ReplayMismatchError(
                        f"vLLM LoRA {name} bytes changed during {phase}"
                    )
        finally:
            os.close(descriptor)

    def assert_lora_request_binding(self, lora_request: object) -> None:
        request_path = getattr(lora_request, "lora_path", None)
        if not isinstance(request_path, str) or not request_path:
            raise TypeError("vLLM LoRARequest must expose its lora_path")
        normalized = Path(os.path.abspath(request_path))
        if normalized != self.adapter_root:
            raise IdentityMismatchError(
                "vLLM LoRARequest path differs from verified adapter root"
            )


def _vllm_lora_adapter_payloads(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
) -> tuple[str, bytes, bytes]:
    adapter_config = {
        "base_model_name_or_path": str(snapshot.run.model.revision_or_path),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 64,
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_bytes = (json.dumps(adapter_config, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    materialization_content = {
        "schema_version": VLLM_LORA_ADAPTER_SCHEMA,
        "evaluation_id": config.evaluation_id,
        "optimizer_step": snapshot.policy_version.optimizer_step,
        "policy_run_id": snapshot.policy_version.run_id,
        "policy_run_identity_sha256": snapshot.lora.run_identity_sha256,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "pointer_file_sha256": snapshot.lora.pointer_file_sha256,
        "manifest_file_sha256": snapshot.lora.manifest_file_sha256,
        "adapter_model_file_sha256": snapshot.lora.tensor_file_sha256,
        "adapter_config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "base_model_name_or_path": str(snapshot.run.model.revision_or_path),
    }
    materialization_identity_sha256 = _canonical_json_sha256(materialization_content)
    identity = {
        **materialization_content,
        "materialization_identity_sha256": materialization_identity_sha256,
        "adapter_model_file": VLLM_LORA_ADAPTER_MODEL_FILENAME,
        "adapter_config_file": VLLM_LORA_ADAPTER_CONFIG_FILENAME,
        "engine_loaded_identity_attestation": VLLM_LORA_ENGINE_ATTESTATION,
        "residual_race": VLLM_LORA_RESIDUAL_RACE,
    }
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return materialization_identity_sha256, config_bytes, identity_bytes


def _open_absolute_directory_nofollow(
    path: Path,
    *,
    owner: str,
    create_missing: bool = False,
) -> int:
    """Traverse an absolute directory from ``/`` without following symlinks."""

    normalized = Path(os.path.abspath(os.fspath(path)))
    if not normalized.is_absolute():
        raise ValueError(f"{owner} path must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(os.sep, flags)
    completed = False
    try:
        for part in normalized.parts[1:]:
            next_descriptor: int | None = None
            try:
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                    raise ReplayMismatchError(f"{owner} path contains a non-directory")
            except BaseException:
                if next_descriptor is not None:
                    os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        completed = True
        return descriptor
    except OSError as error:
        raise ReplayMismatchError(
            f"{owner} path is missing, unreadable, or contains a symlink"
        ) from error
    finally:
        if not completed:
            os.close(descriptor)


def _open_vllm_lora_adapter_root(path: Path, *, owner: str) -> int:
    return _open_absolute_directory_nofollow(path, owner=owner)


def _open_or_create_private_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    owner: str,
) -> int:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{owner} name must be one safe path component")
    descriptor: int | None = None
    completed = False
    try:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(f"{owner} is not a current-user directory")
        os.fchmod(descriptor, 0o700)
        completed = True
        return descriptor
    except OSError as error:
        raise ReplayMismatchError(
            f"{owner} is unreadable, non-directory, or symlinked"
        ) from error
    finally:
        if descriptor is not None and not completed:
            os.close(descriptor)


def _read_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    *,
    phase: str,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayMismatchError(f"vLLM LoRA {name} is not regular during {phase}")
        if (
            metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise ReplayMismatchError(f"vLLM LoRA {name} is not private during {phase}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    except OSError as error:
        raise ReplayMismatchError(
            f"vLLM LoRA {name} is missing, unreadable, or symlinked during {phase}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("vLLM LoRA output name must be one safe path component")
    lock_acquired = False
    try:
        fcntl.flock(root_descriptor, fcntl.LOCK_EX)
        lock_acquired = True
        try:
            _assert_private_vllm_lora_file_equals_at(
                root_descriptor,
                name,
                payload,
            )
        except FileNotFoundError:
            pass
        else:
            return

        _publish_private_vllm_lora_file_at(root_descriptor, name, payload)
    finally:
        if lock_acquired:
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)


def _assert_private_vllm_lora_file_equals_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayMismatchError(f"vLLM LoRA output {name} is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(
                f"vLLM LoRA output {name} is not owned by the current user"
            )
        if metadata.st_nlink != 1:
            raise ReplayMismatchError(
                f"vLLM LoRA output {name} has an unexpected hardlink"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        if b"".join(chunks) != payload:
            raise ReplayMismatchError(
                f"content-addressed vLLM LoRA output {name} differs"
            )
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ReplayMismatchError(
            f"vLLM LoRA output {name} is unreadable or symlinked"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_private_vllm_lora_file_at(
    root_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    """Publish exact bytes without replacing a concurrent immutable winner."""

    temporary_name = f".{name}.{uuid4().hex}.tmp"
    temporary_descriptor: int | None = None
    temporary_exists = False
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        temporary_exists = True
        view = memoryview(payload)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise OSError("short write while materializing vLLM LoRA")
            view = view[written:]
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _assert_private_vllm_lora_file_equals_at(
                root_descriptor,
                name,
                payload,
            )
        os.unlink(temporary_name, dir_fd=root_descriptor)
        temporary_exists = False
        os.fsync(root_descriptor)
    except OSError as error:
        raise ReplayMismatchError(
            f"could not publish private vLLM LoRA output {name}"
        ) from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass


def build_vllm_lora_adapter_integrity_verifier(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
    adapter_root: Path,
) -> VLLMLoRAAdapterIntegrityVerifier:
    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    identity_sha256, config_bytes, identity_bytes = _vllm_lora_adapter_payloads(
        config,
        snapshot,
    )
    expected_root = config.output_root / "runtime" / "lora-adapters" / identity_sha256
    expected_root = Path(os.path.abspath(os.fspath(expected_root)))
    normalized_root = Path(os.path.abspath(os.fspath(adapter_root)))
    if normalized_root != expected_root:
        raise IdentityMismatchError(
            "vLLM LoRA adapter root differs from content identity"
        )
    descriptor = _open_vllm_lora_adapter_root(
        normalized_root,
        owner="vLLM LoRA adapter",
    )
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    verifier = VLLMLoRAAdapterIntegrityVerifier(
        adapter_root=normalized_root,
        materialization_identity_sha256=identity_sha256,
        root_device=metadata.st_dev,
        root_inode=metadata.st_ino,
        adapter_model_bytes=snapshot.lora.tensor_bytes,
        adapter_model_sha256=snapshot.lora.tensor_file_sha256,
        adapter_config_bytes=config_bytes,
        adapter_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        identity_bytes=identity_bytes,
        identity_sha256=hashlib.sha256(identity_bytes).hexdigest(),
    )
    verifier.verify(phase="verifier construction")
    return verifier


def materialize_vllm_lora_adapter(
    config: PolicyCoreDevConfig,
    snapshot: PolicyEvaluationSnapshot,
) -> Path:
    """Expose an exact runtime snapshot through vLLM's PEFT directory ABI.

    Creation is relative to progressively opened, no-follow directory fds and
    the lexical path is re-opened and verified before use.  A same-UID writer
    can still rename ancestors after the final verification; that remaining
    pre-consumption race is covered by ``VLLM_LORA_RESIDUAL_RACE``.
    """

    if not isinstance(snapshot, PolicyEvaluationSnapshot):
        raise TypeError("snapshot must be a PolicyEvaluationSnapshot")
    identity_sha256, config_bytes, identity_bytes = _vllm_lora_adapter_payloads(
        config,
        snapshot,
    )
    adapter_parent = Path(
        os.path.abspath(os.fspath(config.output_root / "runtime" / "lora-adapters"))
    )
    adapter_root = adapter_parent / identity_sha256
    parent_descriptor = _open_absolute_directory_nofollow(
        adapter_parent,
        owner="vLLM LoRA adapter parent",
        create_missing=True,
    )
    descriptor: int | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if parent_metadata.st_uid != os.geteuid():
            raise ReplayMismatchError(
                "vLLM LoRA adapter parent is not owned by the current user"
            )
        os.fchmod(parent_descriptor, 0o700)
        descriptor = _open_or_create_private_directory_at(
            parent_descriptor,
            identity_sha256,
            owner="content-addressed vLLM LoRA adapter",
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_MODEL_FILENAME,
            snapshot.lora.tensor_bytes,
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_CONFIG_FILENAME,
            config_bytes,
        )
        _write_private_vllm_lora_file_at(
            descriptor,
            VLLM_LORA_ADAPTER_IDENTITY_FILENAME,
            identity_bytes,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    build_vllm_lora_adapter_integrity_verifier(
        config,
        snapshot,
        adapter_root,
    ).verify(phase="post-materialization")
    return adapter_root


def policy_lora_request_name(snapshot: PolicyEvaluationSnapshot) -> str:
    """Name the vLLM adapter after the already frozen optimizer step."""

    step = snapshot.policy_version.optimizer_step
    if type(step) is not int or step < 0:
        raise ValueError("LoRA pointer optimizer_step must be a non-negative integer")
    return f"policy-step{step}"


def _base_equivalent_step_zero_lora(
    snapshot: PolicyEvaluationSnapshot,
) -> dict[str, object]:
    """Prove that the step-zero adapter has an exactly zero LoRA delta."""

    if snapshot.policy_version.optimizer_step != 0:
        raise ValueError("base-equivalent LoRA proof requires optimizer step zero")
    tensors = snapshot.lora.tensors
    if not tensors:
        raise ValueError("step-zero LoRA snapshot contains no tensors")
    a_by_stem: dict[str, torch.Tensor] = {}
    b_by_stem: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.endswith(".lora_A.weight"):
            a_by_stem[name.removesuffix(".lora_A.weight")] = tensor
        elif name.endswith(".lora_B.weight"):
            b_by_stem[name.removesuffix(".lora_B.weight")] = tensor
        else:
            raise ValueError("step-zero snapshot contains a non-LoRA tensor")
    if not a_by_stem or set(a_by_stem) != set(b_by_stem):
        raise ValueError("step-zero LoRA A/B tensor names differ")
    b_evidence: list[dict[str, object]] = []
    for stem in sorted(b_by_stem):
        tensor = b_by_stem[stem]
        if torch.count_nonzero(tensor).item() != 0:
            raise ValueError("step-zero LoRA B tensor is not exactly zero")
        b_evidence.append(
            {
                "name": f"{stem}.lora_B.weight",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "tensor_sha256": tensor_checksum(tensor),
            }
        )
    proof_content = {
        "schema_version": "tgvf-base-equivalent-step-zero-lora-v1",
        "optimizer_step": 0,
        "weights_sha256": snapshot.policy_version.weights_sha256,
        "tensor_file_sha256": snapshot.lora.tensor_file_sha256,
        "lora_pair_count": len(a_by_stem),
        "only_lora_a_and_b": True,
        "all_lora_b_exactly_zero": True,
        "lora_b_tensors": b_evidence,
    }
    return {
        **proof_content,
        "proof_sha256": _canonical_json_sha256(proof_content),
    }


def _standalone_engine_kwargs(
    config: PolicyCoreDevConfig, run: PolicyE2ESmokeRunConfig
) -> dict[str, object]:
    """Build explicit vLLM arguments, including suite-specific context limits."""

    common: dict[str, object] = dict(
        model=run.model.revision_or_path,
        dtype="bfloat16",
        trust_remote_code=True,
        distributed_executor_backend="mp",
        max_model_len=config.max_model_len,
        max_num_seqs=config.inference_concurrency_per_gpu,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=False,
        gpu_memory_utilization=config.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        enforce_eager=False,
        seed=run.rollout_rng.master_seed,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={
            "image": 1
            + (
                6
                if config.evaluation_protocol
                == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
                else run.protocol.maximum_tool_calls
            ),
            "video": 0,
        },
    )
    if config.evaluation_protocol == DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL:
        # This is the PRL13 comparison path: source/crop PIL images are handled
        # by stock Qwen3-VL/vLLM.  Loading the recorded-feature architecture or
        # worker extension here would silently turn the control into PRL11.
        return common
    return {
        **common,
        "worker_extension_cls": TGVF_VLLM_WORKER_EXTENSION_FQN,
        "enable_mm_embeds": True,
        "mm_encoder_attn_backend": TGVF_VLLM_MM_ENCODER_ATTN_BACKEND,
        "hf_overrides": {"architectures": [TGVF_QWEN3_VLLM_ARCHITECTURE]},
    }


_LEGACY_MODULE = "tgvf_rl.evaluation.policy_coredev"


# These contracts were historically defined by policy_coredev. The facade
# re-exports these exact objects so old imports and pickle payloads stay valid.
for _legacy_class in (PolicyEvaluationSnapshot, VLLMLoRAAdapterIntegrityVerifier):
    rebind_public_class(
        _legacy_class,
        implementation_module=__name__,
        public_module=_LEGACY_MODULE,
    )
for _function, _legacy_name in (
    (_vllm_lora_adapter_payloads, "_vllm_lora_adapter_payloads"),
    (_open_absolute_directory_nofollow, "_open_absolute_directory_nofollow"),
    (_open_vllm_lora_adapter_root, "_open_vllm_lora_adapter_root"),
    (
        _open_or_create_private_directory_at,
        "_open_or_create_private_directory_at",
    ),
    (_read_private_vllm_lora_file_at, "_read_private_vllm_lora_file_at"),
    (_write_private_vllm_lora_file_at, "_write_private_vllm_lora_file_at"),
    (
        _assert_private_vllm_lora_file_equals_at,
        "_assert_private_vllm_lora_file_equals_at",
    ),
    (_publish_private_vllm_lora_file_at, "_publish_private_vllm_lora_file_at"),
    (
        build_vllm_lora_adapter_integrity_verifier,
        "build_vllm_lora_adapter_integrity_verifier",
    ),
    (materialize_vllm_lora_adapter, "materialize_vllm_lora_adapter"),
    (policy_lora_request_name, "policy_lora_request_name"),
    (_base_equivalent_step_zero_lora, "_base_equivalent_step_zero_lora"),
    (_standalone_engine_kwargs, "_standalone_engine_kwargs"),
):
    rebind_public_function(
        _function,
        implementation_module=__name__,
        public_module=_LEGACY_MODULE,
        public_name=_legacy_name,
        public_qualname=_legacy_name,
    )
del _function, _legacy_class, _legacy_name


__all__ = [
    "PolicyEvaluationSnapshot",
    "VLLMLoRAAdapterIntegrityVerifier",
    "VLLM_LORA_ADAPTER_CONFIG_FILENAME",
    "VLLM_LORA_ADAPTER_IDENTITY_FILENAME",
    "VLLM_LORA_ADAPTER_MODEL_FILENAME",
    "VLLM_LORA_ADAPTER_SCHEMA",
    "VLLM_LORA_ENGINE_ATTESTATION",
    "VLLM_LORA_RESIDUAL_RACE",
    "build_vllm_lora_adapter_integrity_verifier",
    "materialize_vllm_lora_adapter",
    "policy_lora_request_name",
]
