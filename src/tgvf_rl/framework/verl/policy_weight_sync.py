"""Exact LoRA publication around pinned veRL actor-to-vLLM weight sync.

The pinned checkpoint manager transports tensors, but it does not expose the
content identity of the LoRA state that a rollout server has just accepted.
This module adds a small file-backed rendezvous without changing that transport:

* the manager atomically publishes a unique request before every sync;
* rank zero tees the ``base_sync_done=True`` LoRA iterator into an immutable,
  content-addressed safetensors artifact while yielding the original objects;
* a latest-pointer commit is written only after the complete iterator is
  consumed; and
* the manager validates that exact request and optimizer step after upstream
  veRL returns.

Importing this module does not import veRL, Ray, or CUDA.  The pinned manager is
resolved only when the production wrapper is constructed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping
import concurrent.futures
from dataclasses import dataclass
from functools import wraps
import hashlib
import hmac
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
import tgvf_rl.framework.verl.policy_weight_snapshot_store as _snapshot_store


# Keep the historical private facade intact while the filesystem mechanics live
# in an import-acyclic, framework-neutral leaf.
_assert_immutable_file_equals_at = _snapshot_store.assert_immutable_file_equals_at
_assert_snapshot_root_path_binding = (
    _snapshot_store.assert_snapshot_root_path_binding
)
_atomic_replace_bytes = _snapshot_store.atomic_replace_bytes
_fsync_directory = _snapshot_store.fsync_directory
_open_snapshot_root = _snapshot_store.open_snapshot_root
_read_bytes = _snapshot_store.read_bytes
_read_relative_file_bytes_at = _snapshot_store.read_relative_file_bytes_at
_safe_snapshot_relative_path = _snapshot_store.safe_snapshot_relative_path
_write_immutable_bytes = _snapshot_store.write_immutable_bytes


POLICY_WEIGHT_SYNC_REQUEST_SCHEMA = "tgvf-policy-weight-sync-request-v1"
POLICY_LORA_SNAPSHOT_SCHEMA = "tgvf-policy-lora-snapshot-v1"
POLICY_LORA_LATEST_SCHEMA = "tgvf-policy-lora-latest-v1"
POLICY_WEIGHT_SYNC_REQUEST_FILENAME = "weight-sync-request.json"
POLICY_LORA_LATEST_FILENAME = "latest-lora-snapshot.json"
POLICY_LORA_SNAPSHOT_DIRECTORY = "lora-snapshots"
POLICY_LORA_MANIFEST_DIRECTORY = "lora-manifests"
POLICY_REQUIRED_WORLD_SIZE = 4

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class PolicyWeightSyncState:
    """Filesystem and run identity shared by trainer and actor ranks."""

    directory: Path
    run_id: str
    run_identity_sha256: str

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        if not directory.is_absolute():
            raise ValueError("TGVF_POLICY_STATE_DIR must be absolute")
        if not self.run_id:
            raise ValueError("TGVF_POLICY_RUN_ID must be non-empty")
        _require_sha256(self.run_identity_sha256, "TGVF_POLICY_RUN_IDENTITY_SHA256")
        # Preserve the lexical root.  Resolving here would silently accept a
        # symlinked state directory before the secure loader gets a chance to
        # open the root itself with O_NOFOLLOW.
        object.__setattr__(
            self,
            "directory",
            Path(os.path.abspath(os.fspath(directory))),
        )

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "PolicyWeightSyncState":
        values = os.environ if environment is None else environment
        required = (
            "TGVF_POLICY_STATE_DIR",
            "TGVF_POLICY_RUN_ID",
            "TGVF_POLICY_RUN_IDENTITY_SHA256",
        )
        missing = tuple(name for name in required if not values.get(name))
        if missing:
            raise RuntimeError(
                "Policy weight-sync environment is incomplete: " + ", ".join(missing)
            )
        return cls(
            directory=Path(values[required[0]]),
            run_id=values[required[1]],
            run_identity_sha256=values[required[2]],
        )

    @property
    def request_path(self) -> Path:
        return self.directory / POLICY_WEIGHT_SYNC_REQUEST_FILENAME

    @property
    def latest_path(self) -> Path:
        return self.directory / POLICY_LORA_LATEST_FILENAME


@dataclass(frozen=True, slots=True)
class PolicyWeightSyncRequest:
    run_id: str
    run_identity_sha256: str
    optimizer_step: int
    nonce: str
    request_sha256: str
    schema_version: str = POLICY_WEIGHT_SYNC_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_WEIGHT_SYNC_REQUEST_SCHEMA:
            raise ValueError("unsupported Policy weight-sync request schema")
        if not self.run_id or not self.nonce:
            raise ValueError("Policy weight-sync request identity is incomplete")
        _nonnegative_step(self.optimizer_step)
        _require_sha256(self.run_identity_sha256, "run identity")
        _require_sha256(self.request_sha256, "request identity")
        if not hmac.compare_digest(self.request_sha256, self.computed_sha256):
            raise ValueError("Policy weight-sync request digest differs")

    @property
    def computed_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.content_mapping())).hexdigest()

    def content_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "optimizer_step": self.optimizer_step,
            "nonce": self.nonce,
        }

    def as_mapping(self) -> dict[str, object]:
        return {**self.content_mapping(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class PolicyLoRASnapshot:
    """Strictly validated latest LoRA artifact and its policy identity."""

    policy_version: PolicyVersion
    run_identity_sha256: str
    request_sha256: str
    pointer_file: Path
    pointer_file_sha256: str
    pointer_bytes: bytes
    tensor_file: Path
    tensor_file_sha256: str
    tensor_bytes: bytes
    manifest_file: Path
    manifest_file_sha256: str
    manifest_bytes: bytes
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        _require_sha256(self.run_identity_sha256, "run identity")
        _require_sha256(self.request_sha256, "request identity")
        _require_sha256(self.pointer_file_sha256, "pointer file digest")
        _require_sha256(self.tensor_file_sha256, "safetensors file digest")
        _require_sha256(self.manifest_file_sha256, "manifest file digest")
        for owner, payload in (
            ("pointer", self.pointer_bytes),
            ("manifest", self.manifest_bytes),
            ("tensor", self.tensor_bytes),
        ):
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"LoRA snapshot {owner} bytes must be non-empty")
        if not all(
            path.is_absolute()
            for path in (self.pointer_file, self.tensor_file, self.manifest_file)
        ):
            raise ValueError("LoRA snapshot paths must be absolute")


def publish_policy_weight_sync_request(
    state: PolicyWeightSyncState,
    optimizer_step: int,
    *,
    nonce: str | None = None,
) -> PolicyWeightSyncRequest:
    """Atomically replace the active actor-to-vLLM sync request."""

    if not isinstance(state, PolicyWeightSyncState):
        raise TypeError("state must be PolicyWeightSyncState")
    _nonnegative_step(optimizer_step)
    request_nonce = nonce or uuid4().hex
    if not isinstance(request_nonce, str) or not request_nonce:
        raise ValueError("Policy weight-sync request nonce must be non-empty")
    content = {
        "schema_version": POLICY_WEIGHT_SYNC_REQUEST_SCHEMA,
        "run_id": state.run_id,
        "run_identity_sha256": state.run_identity_sha256,
        "optimizer_step": optimizer_step,
        "nonce": request_nonce,
    }
    request = PolicyWeightSyncRequest(
        **content,
        request_sha256=hashlib.sha256(_canonical_json_bytes(content)).hexdigest(),
    )
    _atomic_replace_bytes(
        state.request_path,
        _canonical_json_bytes(request.as_mapping()) + b"\n",
    )
    return request


def load_policy_weight_sync_request(
    state: PolicyWeightSyncState,
) -> PolicyWeightSyncRequest:
    """Load the active request and prove it belongs to this run identity."""

    try:
        root_descriptor = _open_snapshot_root(state.directory)
    except ReplayMismatchError as error:
        raise ReplayMismatchError(
            "Policy weight-sync request is missing or unreadable"
        ) from error
    try:
        try:
            request_bytes = _read_relative_file_bytes_at(
                root_descriptor,
                POLICY_WEIGHT_SYNC_REQUEST_FILENAME,
                "Policy weight-sync request",
            )
        except ReplayMismatchError as error:
            raise ReplayMismatchError(
                "Policy weight-sync request is missing or unreadable"
            ) from error
        _assert_snapshot_root_path_binding(state.directory, root_descriptor)
    finally:
        os.close(root_descriptor)
    mapping = _strict_json_bytes_mapping(
        request_bytes,
        {
            "schema_version",
            "run_id",
            "run_identity_sha256",
            "optimizer_step",
            "nonce",
            "request_sha256",
        },
        "Policy weight-sync request",
    )
    try:
        request = PolicyWeightSyncRequest(**mapping)
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError("Policy weight-sync request is malformed") from error
    _require_run_identity(state, request.run_id, request.run_identity_sha256)
    return request


def wrap_lora_parameter_stream_for_snapshot(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    base_sync_done: bool,
    rank: int | None = None,
    world_size: int | None = None,
    global_steps: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Tee a completed LoRA-only stream to rank-zero immutable storage.

    The yielded tuple objects, names, tensors, order, dtype, device, and storage
    are unchanged.  CPU clones are private to the snapshot.  Base-model streams
    (``base_sync_done=False``) pass through without reading or writing state.
    """

    if type(base_sync_done) is not bool:
        raise TypeError("base_sync_done must be bool")
    iterator = iter(weights)
    if not base_sync_done:
        return iterator
    resolved_rank, resolved_world_size = _distributed_identity(
        rank=rank,
        world_size=world_size,
        environment=environment,
    )
    if resolved_world_size != POLICY_REQUIRED_WORLD_SIZE:
        raise ValueError("Policy Pilot LoRA publication requires exactly four ranks")
    state = PolicyWeightSyncState.from_environment(environment)
    request = load_policy_weight_sync_request(state)
    if global_steps is not None:
        _nonnegative_step(global_steps)
        if request.optimizer_step != global_steps:
            raise IdentityMismatchError(
                "LoRA stream optimizer step differs from the active sync request"
            )
    if resolved_rank != 0:
        return iterator
    return _rank_zero_snapshot_iterator(iterator, state=state, request=request)


def load_latest_lora_snapshot(
    state: PolicyWeightSyncState,
    *,
    expected_optimizer_step: int | None = None,
    expected_request_sha256: str | None = None,
) -> PolicyLoRASnapshot:
    """Load and verify the latest pointer, manifest, safetensors, and tensors."""

    return load_lora_snapshot_pointer(
        state,
        pointer_path=state.latest_path,
        expected_optimizer_step=expected_optimizer_step,
        expected_request_sha256=expected_request_sha256,
    )


def load_lora_snapshot_pointer(
    state: PolicyWeightSyncState,
    *,
    pointer_path: str | Path,
    expected_pointer_file_sha256: str | None = None,
    expected_optimizer_step: int | None = None,
    expected_request_sha256: str | None = None,
) -> PolicyLoRASnapshot:
    """Strictly load one fixed pointer and its complete immutable closure."""

    if not isinstance(state, PolicyWeightSyncState):
        raise TypeError("state must be PolicyWeightSyncState")
    pointer = Path(os.path.abspath(os.fspath(pointer_path)))
    pointer_owner = (
        "latest LoRA pointer" if pointer == state.latest_path else "LoRA pointer"
    )
    if not pointer.is_absolute():
        raise ValueError("LoRA pointer path must be absolute")
    if pointer.parent != state.directory:
        raise ReplayMismatchError("LoRA pointer is outside its state directory")
    if expected_pointer_file_sha256 is not None:
        _require_sha256(
            expected_pointer_file_sha256, "expected LoRA pointer file digest"
        )
    if expected_optimizer_step is not None:
        _nonnegative_step(expected_optimizer_step)
    if expected_request_sha256 is not None:
        _require_sha256(expected_request_sha256, "expected request identity")
    root_descriptor = _open_snapshot_root(state.directory)
    try:
        pointer_bytes = _read_relative_file_bytes_at(
            root_descriptor,
            pointer.name,
            pointer_owner,
        )
        pointer_file_sha256 = _sha256_bytes(pointer_bytes)
        if expected_pointer_file_sha256 is not None and not hmac.compare_digest(
            pointer_file_sha256, expected_pointer_file_sha256
        ):
            raise ReplayMismatchError("LoRA pointer file digest mismatch")
        latest = _strict_json_bytes_mapping(
            pointer_bytes,
            {
                "schema_version",
                "run_id",
                "run_identity_sha256",
                "optimizer_step",
                "request_sha256",
                "weights_sha256",
                "manifest_file",
                "manifest_file_sha256",
                "integrity_sha256",
            },
            "LoRA pointer",
        )
        _verify_integrity_field(latest, "integrity_sha256", "LoRA pointer")
        if latest["schema_version"] != POLICY_LORA_LATEST_SCHEMA:
            raise ReplayMismatchError("LoRA pointer schema differs")
        _require_run_identity(
            state,
            _required_text(latest, "run_id"),
            _required_sha(latest, "run_identity_sha256"),
        )
        step = _required_step(latest, "optimizer_step")
        request_sha256 = _required_sha(latest, "request_sha256")
        weights_sha256 = _required_sha(latest, "weights_sha256")
        if expected_optimizer_step is not None and step != expected_optimizer_step:
            raise IdentityMismatchError("latest LoRA snapshot optimizer step differs")
        if expected_request_sha256 is not None and not hmac.compare_digest(
            request_sha256, expected_request_sha256
        ):
            raise IdentityMismatchError("latest LoRA snapshot request identity differs")

        manifest_relative = _safe_snapshot_relative_path(
            _required_text(latest, "manifest_file")
        )
        manifest_path = state.directory / manifest_relative
        manifest_file_sha256 = _required_sha(latest, "manifest_file_sha256")
        manifest_bytes = _read_relative_file_bytes_at(
            root_descriptor,
            manifest_relative.as_posix(),
            "LoRA manifest",
        )
        if not hmac.compare_digest(_sha256_bytes(manifest_bytes), manifest_file_sha256):
            raise ReplayMismatchError("LoRA manifest file digest mismatch")
        manifest = _strict_json_bytes_mapping(
            manifest_bytes,
            {
                "schema_version",
                "run_id",
                "run_identity_sha256",
                "optimizer_step",
                "request_sha256",
                "weights_sha256",
                "tensor_file",
                "tensor_file_sha256",
                "tensor_names",
                "tensor_metadata",
                "integrity_sha256",
            },
            "LoRA manifest",
        )
        _verify_integrity_field(manifest, "integrity_sha256", "LoRA manifest")
        if manifest["schema_version"] != POLICY_LORA_SNAPSHOT_SCHEMA:
            raise ReplayMismatchError("LoRA manifest schema differs")
        _require_run_identity(
            state,
            _required_text(manifest, "run_id"),
            _required_sha(manifest, "run_identity_sha256"),
        )
        for field, expected in (
            ("optimizer_step", step),
            ("request_sha256", request_sha256),
            ("weights_sha256", weights_sha256),
        ):
            if manifest[field] != expected:
                raise ReplayMismatchError(f"LoRA manifest {field} differs from latest")

        tensor_relative = _safe_snapshot_relative_path(
            _required_text(manifest, "tensor_file")
        )
        tensor_path = state.directory / tensor_relative
        tensor_file_sha256 = _required_sha(manifest, "tensor_file_sha256")
        tensor_bytes = _read_relative_file_bytes_at(
            root_descriptor,
            tensor_relative.as_posix(),
            "LoRA safetensors",
        )
        if not hmac.compare_digest(_sha256_bytes(tensor_bytes), tensor_file_sha256):
            raise ReplayMismatchError("LoRA safetensors file digest mismatch")
        _assert_snapshot_root_path_binding(state.directory, root_descriptor)
    finally:
        os.close(root_descriptor)
    tensors = _load_safetensors_bytes(tensor_bytes)
    actual_weights_sha256 = lora_parameter_mapping_sha256(tensors)
    if not hmac.compare_digest(actual_weights_sha256, weights_sha256):
        raise ReplayMismatchError("LoRA tensor content identity mismatch")
    _verify_tensor_manifest(tensors, manifest)
    version = PolicyVersion(state.run_id, step, weights_sha256)
    return PolicyLoRASnapshot(
        policy_version=version,
        run_identity_sha256=state.run_identity_sha256,
        request_sha256=request_sha256,
        pointer_file=pointer,
        pointer_file_sha256=pointer_file_sha256,
        pointer_bytes=pointer_bytes,
        tensor_file=tensor_path,
        tensor_file_sha256=tensor_file_sha256,
        tensor_bytes=tensor_bytes,
        manifest_file=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        manifest_bytes=manifest_bytes,
        tensors=tensors,
    )


def load_latest_policy_version(
    state: PolicyWeightSyncState,
    *,
    expected_optimizer_step: int | None = None,
    expected_request_sha256: str | None = None,
) -> PolicyVersion:
    """Return the strictly verified policy identity of the latest snapshot."""

    return load_latest_lora_snapshot(
        state,
        expected_optimizer_step=expected_optimizer_step,
        expected_request_sha256=expected_request_sha256,
    ).policy_version


def lora_parameter_mapping_sha256(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """Hash sorted tensor names, dtype, shape, and exact contiguous CPU bytes."""

    normalized = _normalized_tensor_mapping(tensors)
    digest = hashlib.sha256()
    digest.update(b"tgvf-policy-lora-parameter-mapping-v1\0")
    for name in sorted(normalized):
        tensor = normalized[name]
        raw = tensor.view(torch.uint8).numpy().tobytes()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "byte_length": len(raw),
        }
        encoded = _canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(raw)
    return digest.hexdigest()


def _auto_await(function: Callable[..., Any]) -> Callable[..., Any]:
    """Match pinned veRL's sync-call/async-await manager surface."""

    @wraps(function)
    def wrapper(*args: object, **kwargs: object) -> object:
        coroutine = function(*args, **kwargs)
        if not inspect.iscoroutine(coroutine):
            return coroutine
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(coroutine)
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is not None and caller.f_code.co_flags & inspect.CO_COROUTINE:
            return coroutine
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coroutine).result()

    return wrapper


class TGVFPolicyCheckpointEngineManager:
    """Pinned manager facade that proves one exact LoRA version per sync."""

    def __init__(
        self,
        config: object,
        actor_wg: object,
        replicas: list[object],
        *,
        upstream_manager_factory: Callable[..., object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._state = PolicyWeightSyncState.from_environment(environment)
        factory = upstream_manager_factory or _load_pinned_manager_class()
        self._upstream = factory(
            config=config,
            actor_wg=actor_wg,
            replicas=replicas,
        )
        if not callable(getattr(self._upstream, "update_weights", None)):
            raise TypeError("pinned checkpoint manager must implement update_weights()")
        self._last_policy_version: PolicyVersion | None = None

    @property
    def last_policy_version(self) -> PolicyVersion | None:
        return self._last_policy_version

    @_auto_await
    async def update_weights(self, global_steps: int | None = None) -> object:
        if global_steps is None:
            raise ValueError("Policy weight sync requires explicit global_steps")
        _nonnegative_step(global_steps)
        request = publish_policy_weight_sync_request(self._state, global_steps)
        result = self._upstream.update_weights(global_steps=global_steps)
        if inspect.isawaitable(result):
            result = await result
        self._last_policy_version = load_latest_policy_version(
            self._state,
            expected_optimizer_step=global_steps,
            expected_request_sha256=request.request_sha256,
        )
        return result

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._upstream, name)


def _rank_zero_snapshot_iterator(
    iterator: Iterator[tuple[str, torch.Tensor]],
    *,
    state: PolicyWeightSyncState,
    request: PolicyWeightSyncRequest,
) -> Iterator[tuple[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    for item in iterator:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("LoRA parameter stream items must be (name, tensor) tuples")
        name, tensor = item
        if not isinstance(name, str) or not name:
            raise ValueError("LoRA parameter names must be non-empty strings")
        if name in captured:
            raise ValueError(f"duplicate LoRA parameter name {name!r}")
        captured[name] = _snapshot_tensor(tensor)
        yield item
    _publish_lora_snapshot(state, request=request, tensors=captured)


def _publish_lora_snapshot(
    state: PolicyWeightSyncState,
    *,
    request: PolicyWeightSyncRequest,
    tensors: Mapping[str, torch.Tensor],
) -> PolicyVersion:
    _require_run_identity(state, request.run_id, request.run_identity_sha256)
    normalized = _normalized_tensor_mapping(tensors)
    weights_sha256 = lora_parameter_mapping_sha256(normalized)
    tensor_bytes = _save_safetensors_bytes(normalized)
    tensor_file_sha256 = _sha256_bytes(tensor_bytes)
    tensor_relative = Path(POLICY_LORA_SNAPSHOT_DIRECTORY) / (
        f"{weights_sha256}.safetensors"
    )
    tensor_path = state.directory / tensor_relative
    _write_immutable_bytes(tensor_path, tensor_bytes, owner="LoRA safetensors")
    metadata = {
        name: {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": _sha256_bytes(tensor.view(torch.uint8).numpy().tobytes()),
        }
        for name, tensor in sorted(normalized.items())
    }
    manifest_relative = Path(POLICY_LORA_MANIFEST_DIRECTORY) / (
        f"step-{request.optimizer_step:08d}-{request.request_sha256}.json"
    )
    manifest_content = {
        "schema_version": POLICY_LORA_SNAPSHOT_SCHEMA,
        "run_id": state.run_id,
        "run_identity_sha256": state.run_identity_sha256,
        "optimizer_step": request.optimizer_step,
        "request_sha256": request.request_sha256,
        "weights_sha256": weights_sha256,
        "tensor_file": tensor_relative.as_posix(),
        "tensor_file_sha256": tensor_file_sha256,
        "tensor_names": sorted(normalized),
        "tensor_metadata": metadata,
    }
    manifest = _with_integrity(manifest_content)
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
    manifest_path = state.directory / manifest_relative
    _write_immutable_bytes(manifest_path, manifest_bytes, owner="LoRA manifest")
    latest_content = {
        "schema_version": POLICY_LORA_LATEST_SCHEMA,
        "run_id": state.run_id,
        "run_identity_sha256": state.run_identity_sha256,
        "optimizer_step": request.optimizer_step,
        "request_sha256": request.request_sha256,
        "weights_sha256": weights_sha256,
        "manifest_file": manifest_relative.as_posix(),
        "manifest_file_sha256": _sha256_bytes(manifest_bytes),
    }
    _atomic_replace_bytes(
        state.latest_path,
        _canonical_json_bytes(_with_integrity(latest_content)) + b"\n",
    )
    return PolicyVersion(state.run_id, request.optimizer_step, weights_sha256)


def _verify_tensor_manifest(
    tensors: Mapping[str, torch.Tensor], manifest: Mapping[str, object]
) -> None:
    names = manifest["tensor_names"]
    metadata = manifest["tensor_metadata"]
    if not isinstance(names, list) or any(type(name) is not str for name in names):
        raise ReplayMismatchError("LoRA manifest tensor_names is malformed")
    if names != sorted(tensors):
        raise ReplayMismatchError("LoRA manifest tensor names differ")
    if not isinstance(metadata, Mapping) or set(metadata) != set(tensors):
        raise ReplayMismatchError("LoRA manifest tensor metadata differs")
    for name, tensor in tensors.items():
        row = metadata[name]
        if not isinstance(row, Mapping) or set(row) != {"dtype", "shape", "sha256"}:
            raise ReplayMismatchError("LoRA tensor metadata row is malformed")
        expected = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": _sha256_bytes(tensor.view(torch.uint8).numpy().tobytes()),
        }
        if dict(row) != expected:
            raise ReplayMismatchError(f"LoRA tensor metadata differs for {name!r}")


def _normalized_tensor_mapping(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("LoRA snapshot requires at least one tensor")
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("LoRA tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"LoRA value {name!r} must be a torch.Tensor")
        normalized[name] = _snapshot_tensor(tensor)
    return normalized


def _snapshot_tensor(tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("LoRA parameter stream values must be torch.Tensor")
    if tensor.layout is not torch.strided:
        raise TypeError("LoRA snapshot supports only strided tensors")
    if not tensor.is_floating_point():
        raise TypeError("LoRA snapshot tensors must be floating point")
    return tensor.detach().to(device="cpu").contiguous().clone()


def _save_safetensors_bytes(tensors: Mapping[str, torch.Tensor]) -> bytes:
    try:
        from safetensors.torch import save

        return save({name: tensors[name] for name in sorted(tensors)})
    except Exception as error:
        raise RuntimeError("could not serialize exact LoRA safetensors") from error


def _load_safetensors_bytes(value: bytes) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load

        tensors = load(value)
    except Exception as error:
        raise ReplayMismatchError("LoRA safetensors is unreadable") from error
    try:
        return _normalized_tensor_mapping(tensors)
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(
            "LoRA safetensors tensor mapping is invalid"
        ) from error


def _distributed_identity(
    *,
    rank: int | None,
    world_size: int | None,
    environment: Mapping[str, str] | None,
) -> tuple[int, int]:
    values = os.environ if environment is None else environment
    if (
        rank is None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        rank = torch.distributed.get_rank()
    if (
        world_size is None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        world_size = torch.distributed.get_world_size()
    try:
        if rank is None:
            rank = int(values["RANK"])
        if world_size is None:
            world_size = int(values["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "LoRA publication requires explicit distributed identity"
        ) from error
    if type(rank) is not int or type(world_size) is not int:
        raise TypeError("rank and world_size must be integers")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid LoRA publication distributed identity")
    return rank, world_size


def _load_pinned_manager_class() -> type[object]:
    from .compatibility import (
        SPIKE_CANDIDATE_VERL_COMMIT,
        verify_verl_distribution_identity,
    )

    verify_verl_distribution_identity(expected_commit=SPIKE_CANDIDATE_VERL_COMMIT)
    try:
        manager = getattr(
            import_module("verl.checkpoint_engine"), "CheckpointEngineManager"
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "pinned veRL CheckpointEngineManager is unavailable"
        ) from error
    if not isinstance(manager, type):
        raise TypeError("pinned veRL CheckpointEngineManager is not a class")
    return manager


def _with_integrity(content: Mapping[str, object]) -> dict[str, object]:
    result = dict(content)
    result["integrity_sha256"] = hashlib.sha256(
        _canonical_json_bytes(content)
    ).hexdigest()
    return result


def _verify_integrity_field(
    mapping: Mapping[str, object], field: str, owner: str
) -> None:
    expected = _required_sha(mapping, field)
    content = {name: value for name, value in mapping.items() if name != field}
    actual = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ReplayMismatchError(f"{owner} integrity mismatch")


def _strict_json_mapping(
    path: Path, expected: set[str], owner: str
) -> Mapping[str, object]:
    return _strict_json_bytes_mapping(_read_bytes(path, owner), expected, owner)


def _strict_json_bytes_mapping(
    value: bytes, expected: set[str], owner: str
) -> Mapping[str, object]:
    try:
        decoded = json.loads(value)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReplayMismatchError(f"{owner} is unreadable") from error
    if not isinstance(decoded, Mapping):
        raise ReplayMismatchError(f"{owner} must be a mapping")
    if any(type(name) is not str for name in decoded):
        raise ReplayMismatchError(f"{owner} field names must be strings")
    actual = set(decoded)
    if actual != expected:
        raise ReplayMismatchError(
            f"{owner} fields differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    return decoded


def _require_run_identity(
    state: PolicyWeightSyncState, run_id: str, run_identity_sha256: str
) -> None:
    if run_id != state.run_id:
        raise IdentityMismatchError("Policy weight-sync run_id differs")
    if not hmac.compare_digest(run_identity_sha256, state.run_identity_sha256):
        raise IdentityMismatchError("Policy weight-sync run identity differs")


def _required_text(mapping: Mapping[str, object], name: str) -> str:
    value = mapping[name]
    if not isinstance(value, str) or not value:
        raise ReplayMismatchError(f"{name} must be non-empty text")
    return value


def _required_sha(mapping: Mapping[str, object], name: str) -> str:
    value = _required_text(mapping, name)
    try:
        _require_sha256(value, name)
    except ValueError as error:
        raise ReplayMismatchError(f"{name} is malformed") from error
    return value


def _required_step(mapping: Mapping[str, object], name: str) -> int:
    value = mapping[name]
    try:
        _nonnegative_step(value)
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(f"{name} is malformed") from error
    return value


def _nonnegative_step(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("optimizer step must be a non-negative integer")


def _require_sha256(value: object, owner: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{owner} must be a lowercase SHA256")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("Policy weight-sync state is not canonical JSON") from error


__all__ = [
    "POLICY_LORA_LATEST_FILENAME",
    "POLICY_LORA_LATEST_SCHEMA",
    "POLICY_LORA_MANIFEST_DIRECTORY",
    "POLICY_LORA_SNAPSHOT_DIRECTORY",
    "POLICY_LORA_SNAPSHOT_SCHEMA",
    "POLICY_REQUIRED_WORLD_SIZE",
    "POLICY_WEIGHT_SYNC_REQUEST_FILENAME",
    "POLICY_WEIGHT_SYNC_REQUEST_SCHEMA",
    "PolicyLoRASnapshot",
    "PolicyWeightSyncRequest",
    "PolicyWeightSyncState",
    "TGVFPolicyCheckpointEngineManager",
    "load_latest_lora_snapshot",
    "load_latest_policy_version",
    "load_lora_snapshot_pointer",
    "load_policy_weight_sync_request",
    "lora_parameter_mapping_sha256",
    "publish_policy_weight_sync_request",
    "wrap_lora_parameter_stream_for_snapshot",
]
