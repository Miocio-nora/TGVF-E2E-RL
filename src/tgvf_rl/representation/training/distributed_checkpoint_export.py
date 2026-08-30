"""Rank-zero Adapter export contracts and atomic artifact IO."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)

from .checkpoint import RepresentationRunIdentity
from .distributed_checkpoint_integrity import (
    _fsync_directory,
    _plain_cpu_tensor_state,
    _tensor_checksum,
)
from .distributed_checkpoint_schema import (
    _BORROWED_QWEN_PREFIXES,
    _non_empty_text,
    _non_negative_int,
    _sha256,
    _sorted_unique_names,
    _validate_run_identity,
)


RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION = "rank-zero-adapter-export-v1"


@dataclass(frozen=True, slots=True)
class RankZeroAdapterOwnedStateManifest:
    run_identity: RepresentationRunIdentity
    run_identity_sha256: str
    global_step: int
    tensor_names: tuple[str, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    tensor_dtypes: tuple[str, ...]
    tensor_sha256: tuple[str, ...]
    schema_version: str = RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_identity(self.run_identity)
        _sha256(self.run_identity_sha256, field_name="run_identity_sha256")
        if self.run_identity_sha256 != self.run_identity.identity_sha256:
            raise ValueError("rank-zero export run identity digest mismatch")
        _non_negative_int(self.global_step, field_name="global_step")
        _sorted_unique_names(self.tensor_names, field_name="tensor_names")
        count = len(self.tensor_names)
        if not (
            len(self.tensor_shapes)
            == len(self.tensor_dtypes)
            == len(self.tensor_sha256)
            == count
        ):
            raise ValueError("rank-zero export tensor manifest fields must align")
        for shape in self.tensor_shapes:
            if not shape or any(size < 0 for size in shape):
                raise ValueError("rank-zero export tensor shape is invalid")
        for dtype in self.tensor_dtypes:
            _non_empty_text(dtype, field_name="tensor dtype")
        for digest in self.tensor_sha256:
            _sha256(digest, field_name="tensor_sha256")
        if self.schema_version != RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION:
            raise ValueError("rank-zero Adapter export schema mismatch")


@dataclass(frozen=True, slots=True)
class RankZeroAdapterOwnedStateExport:
    manifest: RankZeroAdapterOwnedStateManifest
    state: dict[str, torch.Tensor] | None
    writer_rank: int = 0

    @property
    def is_writer(self) -> bool:
        return self.state is not None


def save_rank_zero_adapter_owned_state_export_atomic(
    path: str | Path,
    export: RankZeroAdapterOwnedStateExport,
) -> bool:
    """Atomically publish one gathered deployable export without overwriting.

    Every rank may call this function. Non-writer exports return ``False``
    before touching the filesystem; only rank zero's export creates the file.
    """

    _validate_rank_zero_export(export, require_state=False)
    if not export.is_writer:
        return False
    _validate_rank_zero_export(export, require_state=True)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("rank-zero Adapter exports never overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(export, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard-link publish is atomic and fails if another
        # writer already created the no-overwrite destination.
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def load_rank_zero_adapter_owned_state_export(
    path: str | Path,
    *,
    expected_run_identity: RepresentationRunIdentity | None = None,
) -> RankZeroAdapterOwnedStateExport:
    """Load and integrity-check a deployable rank-zero Adapter export."""

    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise ReplayMismatchError(
            f"cannot load rank-zero Adapter export: {error}"
        ) from error
    if not isinstance(value, RankZeroAdapterOwnedStateExport):
        raise ReplayMismatchError("file is not a rank-zero Adapter export")
    _validate_rank_zero_export(value, require_state=True)
    if expected_run_identity is not None and (
        value.manifest.run_identity != expected_run_identity
        or value.manifest.run_identity_sha256 != expected_run_identity.identity_sha256
    ):
        raise IdentityMismatchError("rank-zero Adapter export run identity mismatch")
    return value


def _validate_rank_zero_export(
    export: RankZeroAdapterOwnedStateExport,
    *,
    require_state: bool,
) -> None:
    if not isinstance(export, RankZeroAdapterOwnedStateExport):
        raise TypeError("export must be a RankZeroAdapterOwnedStateExport")
    if export.writer_rank != 0:
        raise ValueError("rank-zero Adapter export writer rank must be zero")
    if not isinstance(export.manifest, RankZeroAdapterOwnedStateManifest):
        raise TypeError("rank-zero Adapter export manifest has the wrong type")
    export.manifest.__post_init__()
    if export.state is None:
        if require_state:
            raise ReplayMismatchError(
                "rank-zero Adapter export tensor state is missing"
            )
        return
    if not isinstance(export.state, dict):
        raise TypeError("rank-zero Adapter export state must be a dict")
    state = _plain_cpu_tensor_state(export.state)
    manifest = export.manifest
    names = tuple(sorted(state))
    if names != manifest.tensor_names:
        raise ReplayMismatchError("rank-zero Adapter export tensor names mismatch")
    shapes = tuple(tuple(state[name].shape) for name in names)
    dtypes = tuple(str(state[name].dtype) for name in names)
    digests = tuple(_tensor_checksum(state[name]) for name in names)
    if shapes != manifest.tensor_shapes:
        raise ReplayMismatchError("rank-zero Adapter export tensor shapes mismatch")
    if dtypes != manifest.tensor_dtypes:
        raise ReplayMismatchError("rank-zero Adapter export tensor dtypes mismatch")
    if digests != manifest.tensor_sha256:
        raise ReplayMismatchError("rank-zero Adapter export tensor checksum mismatch")
    if any(name.startswith(_BORROWED_QWEN_PREFIXES) for name in names):
        raise ReplayMismatchError(
            "rank-zero Adapter export contains borrowed Qwen state"
        )


_DISTRIBUTED_CHECKPOINT_EXPORT_TYPES = (
    RankZeroAdapterOwnedStateManifest,
    RankZeroAdapterOwnedStateExport,
)
_DISTRIBUTED_CHECKPOINT_EXPORT_FUNCTIONS = (
    save_rank_zero_adapter_owned_state_export_atomic,
    load_rank_zero_adapter_owned_state_export,
    _validate_rank_zero_export,
)
_LEGACY_PUBLIC_MODULE = "tgvf_rl.representation.training.distributed_checkpoint"

for _export_type in _DISTRIBUTED_CHECKPOINT_EXPORT_TYPES:
    freeze_public_class_annotations(
        _export_type,
        implementation_globals=globals(),
    )
    rebind_public_class(
        _export_type,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _export_type

for _export_function in _DISTRIBUTED_CHECKPOINT_EXPORT_FUNCTIONS:
    rebind_public_function(
        _export_function,
        implementation_module=__name__,
        public_module=_LEGACY_PUBLIC_MODULE,
    )
del _export_function

__all__ = [
    "RANK_ZERO_ADAPTER_EXPORT_SCHEMA_VERSION",
    "RankZeroAdapterOwnedStateExport",
    "RankZeroAdapterOwnedStateManifest",
    "load_rank_zero_adapter_owned_state_export",
    "save_rank_zero_adapter_owned_state_export_atomic",
]
