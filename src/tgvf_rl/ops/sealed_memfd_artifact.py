"""Linux sealed-memfd artifacts for a future immutable startup boundary.

This leaf provides only a content-immutable byte capability.  It does not
install an import root, verify a runtime locator, consume launch authority, or
dispatch a worker.  A future pre-runtime trampoline may use the serializable
identity to reopen an artifact through the live owner's ``/proc/<pid>/fd``
entry and then retain the verified local descriptor.  Passing that proc path
directly as a Python script is *not* an authenticity boundary: Python reads
script bytes before those bytes can verify themselves.

The primitive is intentionally Linux- and procfs-specific.  Callers must fail
closed when memfd sealing or cross-process proc-fd reopening is unavailable;
there is no mutable-path fallback.  Descriptor numbers remain a same-process
capability: code that deliberately closes, ``dup2``-rebinds, or concurrently
mutates an owned fd is outside this primitive's closure and must not run before
the trusted startup boundary finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from weakref import finalize


SEALED_MEMFD_ARTIFACT_SCHEMA = "tgvf-sealed-memfd-artifact-v1"
MAX_SEALED_MEMFD_ARTIFACT_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PURPOSE_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_MEMFD_NAME_MAX_BYTES = 200
_READ_CHUNK_BYTES = 1024 * 1024
_BINDING_MINT_SENTINEL = object()
_REFERENCE_MINT_SENTINEL = object()


class SealedMemfdArtifactError(RuntimeError):
    """Raised when an immutable artifact cannot be created or verified."""


def _required_seals() -> int:
    names = ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    missing = [name for name in names if not hasattr(fcntl, name)]
    if missing or not hasattr(fcntl, "F_ADD_SEALS") or not hasattr(
        fcntl, "F_GET_SEALS"
    ):
        raise SealedMemfdArtifactError(
            "kernel/Python memfd sealing support is unavailable"
        )
    return (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_SEAL
    )


def _process_start_ticks(pid: int) -> int:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError) as error:
        raise SealedMemfdArtifactError(
            f"sealed artifact owner process {pid} is not live"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise SealedMemfdArtifactError(
            f"cannot inspect sealed artifact owner process {pid}"
        ) from error
    closing = raw.rfind(")")
    if closing < 0:
        raise SealedMemfdArtifactError(
            f"sealed artifact owner process {pid} stat is malformed"
        )
    fields = raw[closing + 2 :].split()
    try:
        ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise SealedMemfdArtifactError(
            f"sealed artifact owner process {pid} has no start time"
        ) from error
    if ticks <= 0:
        raise SealedMemfdArtifactError(
            f"sealed artifact owner process {pid} start time is invalid"
        )
    return ticks


def _require_exact_int(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"sealed artifact {name} must be an integer >= {minimum}")
    return value


def _require_purpose(value: object) -> str:
    if type(value) is not str or not _PURPOSE_RE.fullmatch(value):
        raise ValueError("sealed artifact purpose is not canonical")
    return value


def _require_sha256(value: object) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError("sealed artifact SHA256 must be lowercase hexadecimal")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"sealed artifact JSON repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"sealed artifact JSON contains non-finite value {value!r}")


def _load_strict_json(value: str) -> object:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("sealed artifact JSON must be valid UTF-8 text") from error
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("sealed artifact JSON is malformed") from error


@dataclass(frozen=True, slots=True)
class SealedMemfdArtifactIdentity:
    """Serializable locator and exact identity for one sealed memfd inode."""

    purpose: str
    owner_pid: int
    owner_process_start_ticks: int
    owner_descriptor: int
    sha256: str
    byte_length: int
    device: int
    inode: int
    mode: int
    seals: int
    schema_version: str = SEALED_MEMFD_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_MEMFD_ARTIFACT_SCHEMA:
            raise ValueError("sealed artifact schema differs")
        _require_purpose(self.purpose)
        _require_exact_int(self.owner_pid, name="owner_pid", minimum=1)
        _require_exact_int(
            self.owner_process_start_ticks,
            name="owner_process_start_ticks",
            minimum=1,
        )
        _require_exact_int(
            self.owner_descriptor,
            name="owner_descriptor",
            minimum=0,
        )
        _require_sha256(self.sha256)
        byte_length = _require_exact_int(
            self.byte_length,
            name="byte_length",
            minimum=1,
        )
        if byte_length > MAX_SEALED_MEMFD_ARTIFACT_BYTES:
            raise ValueError("sealed artifact byte_length exceeds the fixed ceiling")
        _require_exact_int(self.device, name="device", minimum=0)
        _require_exact_int(self.inode, name="inode", minimum=1)
        mode = _require_exact_int(self.mode, name="mode", minimum=1)
        if not stat.S_ISREG(mode):
            raise ValueError("sealed artifact mode is not a regular file")
        seals = _require_exact_int(self.seals, name="seals", minimum=1)
        if seals & _required_seals() != _required_seals():
            raise ValueError("sealed artifact identity lacks mandatory seals")

    @property
    def proc_fd_path(self) -> Path:
        """Return the live-owner procfs path; opening it still requires verification."""

        return Path("/proc") / str(self.owner_pid) / "fd" / str(
            self.owner_descriptor
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "owner_pid": self.owner_pid,
            "owner_process_start_ticks": self.owner_process_start_ticks,
            "owner_descriptor": self.owner_descriptor,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "seals": self.seals,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self._identity_record())

    def as_record(self) -> dict[str, object]:
        return {**self._identity_record(), "identity_sha256": self.identity_sha256}

    def to_json(self) -> str:
        return _canonical_json(self.as_record())

    @classmethod
    def from_record(cls, value: object) -> SealedMemfdArtifactIdentity:
        fields = {
            "schema_version",
            "purpose",
            "owner_pid",
            "owner_process_start_ticks",
            "owner_descriptor",
            "sha256",
            "byte_length",
            "device",
            "inode",
            "mode",
            "seals",
            "identity_sha256",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("sealed artifact identity field set differs")
        identity = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            purpose=value["purpose"],  # type: ignore[arg-type]
            owner_pid=value["owner_pid"],  # type: ignore[arg-type]
            owner_process_start_ticks=value[  # type: ignore[arg-type]
                "owner_process_start_ticks"
            ],
            owner_descriptor=value["owner_descriptor"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
            byte_length=value["byte_length"],  # type: ignore[arg-type]
            device=value["device"],  # type: ignore[arg-type]
            inode=value["inode"],  # type: ignore[arg-type]
            mode=value["mode"],  # type: ignore[arg-type]
            seals=value["seals"],  # type: ignore[arg-type]
        )
        supplied = _require_sha256(value["identity_sha256"])
        if supplied != identity.identity_sha256:
            raise ValueError("sealed artifact identity digest differs")
        return identity

    @classmethod
    def from_json(cls, value: object) -> SealedMemfdArtifactIdentity:
        if type(value) is not str:
            raise TypeError("sealed artifact JSON must be exactly str")
        identity = cls.from_record(_load_strict_json(value))
        if identity.to_json() != value:
            raise ValueError("sealed artifact JSON is not canonical")
        return identity


def _metadata_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_exact_descriptor(descriptor: int, expected_length: int) -> str:
    digest = sha256()
    observed = 0
    while True:
        limit = min(_READ_CHUNK_BYTES, expected_length - observed + 1)
        block = os.pread(descriptor, limit, observed)
        if not block:
            break
        digest.update(block)
        observed += len(block)
        if observed > expected_length:
            raise SealedMemfdArtifactError("sealed artifact grew while read")
    if observed != expected_length:
        raise SealedMemfdArtifactError("sealed artifact read was incomplete")
    return digest.hexdigest()


def _verify_descriptor(
    descriptor: int,
    identity: SealedMemfdArtifactIdentity,
) -> None:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SealedMemfdArtifactError("sealed artifact is not a regular file")
    expected_metadata = (
        identity.device,
        identity.inode,
        identity.mode,
        identity.byte_length,
    )
    observed_metadata = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
    )
    if observed_metadata != expected_metadata:
        raise SealedMemfdArtifactError("sealed artifact metadata differs")
    observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    if observed_seals != identity.seals:
        raise SealedMemfdArtifactError("sealed artifact seal set differs")
    if observed_seals & _required_seals() != _required_seals():
        raise SealedMemfdArtifactError("sealed artifact mandatory seals are absent")
    observed_sha256 = _hash_exact_descriptor(descriptor, identity.byte_length)
    after = os.fstat(descriptor)
    if _metadata_signature(after) != _metadata_signature(before):
        raise SealedMemfdArtifactError("sealed artifact changed while verified")
    if observed_sha256 != identity.sha256:
        raise SealedMemfdArtifactError("sealed artifact SHA256 differs")


def _close_descriptors_if_still_owned(
    descriptors: tuple[int, ...],
    device: int,
    inode: int,
    seals: int,
) -> None:
    """Close only the exact fd identity originally owned by this capability.

    The inode and seal checks prevent a stale integer fd from closing an
    unrelated file after a caller bypasses the capability and closes the raw
    descriptor.  A post-fork copy closes only its own descriptor-table entries;
    the parent capability and its references remain live.
    """

    for descriptor in descriptors:
        try:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != (device, inode):
                continue
            if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
                continue
            os.close(descriptor)
        except OSError:
            pass


def _close_unpublished_descriptor_best_effort(descriptor: int) -> None:
    """Close a factory-local descriptor that has never escaped to a caller."""

    try:
        os.close(descriptor)
    except OSError:
        pass


class SealedMemfdArtifactBinding:
    """PID-bound ownership of a newly created sealed memfd descriptor."""

    __slots__ = (
        "__weakref__",
        "_closed",
        "_descriptor",
        "_finalizer",
        "_guard_descriptor",
        "_identity",
        "_process_id",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "SealedMemfdArtifactBinding can only be minted by "
            "create_sealed_memfd_artifact"
        )

    @classmethod
    def _mint(
        cls,
        descriptor: int,
        guard_descriptor: int,
        identity: SealedMemfdArtifactIdentity,
        *,
        _sentinel: object,
    ) -> SealedMemfdArtifactBinding:
        if _sentinel is not _BINDING_MINT_SENTINEL:
            raise TypeError("sealed artifact binding mint sentinel differs")
        binding = object.__new__(cls)
        object.__setattr__(binding, "_descriptor", descriptor)
        object.__setattr__(binding, "_guard_descriptor", guard_descriptor)
        object.__setattr__(binding, "_identity", identity)
        object.__setattr__(binding, "_process_id", os.getpid())
        object.__setattr__(binding, "_closed", False)
        object.__setattr__(
            binding,
            "_finalizer",
            finalize(
                binding,
                _close_descriptors_if_still_owned,
                (descriptor, guard_descriptor),
                identity.device,
                identity.inode,
                identity.seals,
            ),
        )
        return binding

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("sealed artifact binding is immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("sealed artifact binding cannot be subclassed")

    def __reduce__(self) -> object:
        raise TypeError("sealed artifact binding is process-local")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("sealed artifact binding is process-local")

    def __copy__(self) -> object:
        raise TypeError("sealed artifact binding is not copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("sealed artifact binding is not copyable")

    def _require_owner(self) -> None:
        if self._closed or not self._finalizer.alive:
            raise SealedMemfdArtifactError("sealed artifact binding is closed")
        if os.getpid() != self._process_id:
            raise SealedMemfdArtifactError(
                "sealed artifact binding belongs to a different process"
            )

    @property
    def closed(self) -> bool:
        return self._closed or not self._finalizer.alive

    @property
    def identity(self) -> SealedMemfdArtifactIdentity:
        self._require_owner()
        return self._identity

    @property
    def proc_fd_path(self) -> Path:
        self._require_owner()
        return self._identity.proc_fd_path

    def fileno(self) -> int:
        """Return a borrowed fd number; callers must not close or rebind it."""

        self._require_owner()
        return self._descriptor

    def verify(self) -> None:
        self._require_owner()
        if _process_start_ticks(self._process_id) != (
            self._identity.owner_process_start_ticks
        ):
            raise SealedMemfdArtifactError("sealed artifact owner identity differs")
        _verify_descriptor(self._descriptor, self._identity)

    def close(self) -> None:
        if not self._closed:
            if self._finalizer.alive:
                self._finalizer()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> SealedMemfdArtifactBinding:
        self._require_owner()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class VerifiedSealedMemfdArtifactReference:
    """A local descriptor reopened and checked from a live owner's identity."""

    __slots__ = (
        "__weakref__",
        "_closed",
        "_descriptor",
        "_finalizer",
        "_guard_descriptor",
        "_identity",
        "_process_id",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedSealedMemfdArtifactReference can only be minted by "
            "open_verified_sealed_memfd_artifact"
        )

    @classmethod
    def _mint(
        cls,
        descriptor: int,
        guard_descriptor: int,
        identity: SealedMemfdArtifactIdentity,
        *,
        _sentinel: object,
    ) -> VerifiedSealedMemfdArtifactReference:
        if _sentinel is not _REFERENCE_MINT_SENTINEL:
            raise TypeError("sealed artifact reference mint sentinel differs")
        reference = object.__new__(cls)
        object.__setattr__(reference, "_descriptor", descriptor)
        object.__setattr__(reference, "_guard_descriptor", guard_descriptor)
        object.__setattr__(reference, "_identity", identity)
        object.__setattr__(reference, "_process_id", os.getpid())
        object.__setattr__(reference, "_closed", False)
        object.__setattr__(
            reference,
            "_finalizer",
            finalize(
                reference,
                _close_descriptors_if_still_owned,
                (descriptor, guard_descriptor),
                identity.device,
                identity.inode,
                identity.seals,
            ),
        )
        return reference

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("sealed artifact reference is immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("sealed artifact reference cannot be subclassed")

    def __reduce__(self) -> object:
        raise TypeError("sealed artifact reference is process-local")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("sealed artifact reference is process-local")

    def __copy__(self) -> object:
        raise TypeError("sealed artifact reference is not copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("sealed artifact reference is not copyable")

    def _require_current_process(self) -> None:
        if self._closed or not self._finalizer.alive:
            raise SealedMemfdArtifactError("sealed artifact reference is closed")
        if os.getpid() != self._process_id:
            raise SealedMemfdArtifactError(
                "sealed artifact reference belongs to a different process"
            )

    @property
    def closed(self) -> bool:
        return self._closed or not self._finalizer.alive

    @property
    def identity(self) -> SealedMemfdArtifactIdentity:
        self._require_current_process()
        return self._identity

    @property
    def local_proc_fd_path(self) -> Path:
        self._require_current_process()
        return Path("/proc/self/fd") / str(self._descriptor)

    def fileno(self) -> int:
        """Return a borrowed fd number; callers must not close or rebind it."""

        self._require_current_process()
        return self._descriptor

    def verify(self) -> None:
        self._require_current_process()
        _verify_descriptor(self._descriptor, self._identity)

    def close(self) -> None:
        if not self._closed:
            if self._finalizer.alive:
                self._finalizer()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> VerifiedSealedMemfdArtifactReference:
        self._require_current_process()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _require_memfd_platform() -> None:
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        raise SealedMemfdArtifactError("os.memfd_create sealing is unavailable")
    if not Path("/proc/self/fd").is_dir():
        raise SealedMemfdArtifactError("procfs fd access is unavailable")
    _required_seals()


def _require_memfd_name(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("sealed artifact memfd name must be a non-empty str")
    if "\x00" in value or "/" in value or "\r" in value or "\n" in value:
        raise ValueError("sealed artifact memfd name contains a forbidden character")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("sealed artifact memfd name must be valid UTF-8") from error
    if len(encoded) > _MEMFD_NAME_MAX_BYTES:
        raise ValueError("sealed artifact memfd name is too long")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise SealedMemfdArtifactError("sealed artifact write made no progress")
        offset += written


def create_sealed_memfd_artifact(
    payload: bytes,
    *,
    purpose: str,
    name: str,
) -> SealedMemfdArtifactBinding:
    """Copy exact bytes into a new sealed inode and retain its owner descriptor."""

    if type(payload) is not bytes or not payload:
        raise ValueError("sealed artifact payload must be non-empty exact bytes")
    if len(payload) > MAX_SEALED_MEMFD_ARTIFACT_BYTES:
        raise ValueError("sealed artifact payload exceeds the fixed byte ceiling")
    canonical_purpose = _require_purpose(purpose)
    canonical_name = _require_memfd_name(name)
    _require_memfd_platform()
    flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    descriptor: int | None = None
    guard_descriptor: int | None = None
    try:
        descriptor = os.memfd_create(canonical_name, flags)
        os.fchmod(descriptor, 0o400)
        _write_all(descriptor, payload)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _required_seals())
        metadata = os.fstat(descriptor)
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        identity = SealedMemfdArtifactIdentity(
            purpose=canonical_purpose,
            owner_pid=os.getpid(),
            owner_process_start_ticks=_process_start_ticks(os.getpid()),
            owner_descriptor=descriptor,
            sha256=sha256(payload).hexdigest(),
            byte_length=len(payload),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            seals=seals,
        )
        _verify_descriptor(descriptor, identity)
        if os.get_inheritable(descriptor):
            raise SealedMemfdArtifactError(
                "new sealed artifact descriptor unexpectedly inherits across exec"
            )
        guard_descriptor = os.dup(descriptor)
        if os.get_inheritable(guard_descriptor):
            raise SealedMemfdArtifactError(
                "sealed artifact guard descriptor unexpectedly inherits"
            )
        binding = SealedMemfdArtifactBinding._mint(
            descriptor,
            guard_descriptor,
            identity,
            _sentinel=_BINDING_MINT_SENTINEL,
        )
        descriptor = None
        guard_descriptor = None
        return binding
    except SealedMemfdArtifactError:
        raise
    except OSError as error:
        raise SealedMemfdArtifactError(
            "failed to create and seal immutable memfd artifact"
        ) from error
    finally:
        if descriptor is not None:
            _close_unpublished_descriptor_best_effort(descriptor)
        if guard_descriptor is not None:
            _close_unpublished_descriptor_best_effort(guard_descriptor)


def open_verified_sealed_memfd_artifact(
    identity: SealedMemfdArtifactIdentity,
) -> VerifiedSealedMemfdArtifactReference:
    """Reopen one live owner's sealed inode and retain a verified local fd."""

    if type(identity) is not SealedMemfdArtifactIdentity:
        raise TypeError("sealed artifact identity type differs")
    _require_memfd_platform()
    if _process_start_ticks(identity.owner_pid) != (
        identity.owner_process_start_ticks
    ):
        raise SealedMemfdArtifactError("sealed artifact owner process was replaced")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    guard_descriptor: int | None = None
    try:
        descriptor = os.open(identity.proc_fd_path, flags)
        _verify_descriptor(descriptor, identity)
        if _process_start_ticks(identity.owner_pid) != (
            identity.owner_process_start_ticks
        ):
            raise SealedMemfdArtifactError(
                "sealed artifact owner process changed during reopen"
            )
        if os.get_inheritable(descriptor):
            raise SealedMemfdArtifactError(
                "reopened sealed artifact descriptor unexpectedly inherits"
            )
        guard_descriptor = os.dup(descriptor)
        if os.get_inheritable(guard_descriptor):
            raise SealedMemfdArtifactError(
                "reopened sealed artifact guard unexpectedly inherits"
            )
        reference = VerifiedSealedMemfdArtifactReference._mint(
            descriptor,
            guard_descriptor,
            identity,
            _sentinel=_REFERENCE_MINT_SENTINEL,
        )
        descriptor = None
        guard_descriptor = None
        return reference
    except SealedMemfdArtifactError:
        raise
    except OSError as error:
        raise SealedMemfdArtifactError(
            "sealed artifact owner fd is unavailable through procfs"
        ) from error
    finally:
        if descriptor is not None:
            _close_unpublished_descriptor_best_effort(descriptor)
        if guard_descriptor is not None:
            _close_unpublished_descriptor_best_effort(guard_descriptor)


__all__ = [
    "MAX_SEALED_MEMFD_ARTIFACT_BYTES",
    "SEALED_MEMFD_ARTIFACT_SCHEMA",
    "SealedMemfdArtifactBinding",
    "SealedMemfdArtifactError",
    "SealedMemfdArtifactIdentity",
    "VerifiedSealedMemfdArtifactReference",
    "create_sealed_memfd_artifact",
    "open_verified_sealed_memfd_artifact",
]
