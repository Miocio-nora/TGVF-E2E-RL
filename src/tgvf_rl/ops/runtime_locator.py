"""Descriptor verifier and process-local evidence for a runtime scaffold.

This leaf verifies only explicit declarations from
``runtime_locator_manifest``.  It never discovers paths from the interpreter,
environment, ``sysconfig``, or ``site`` and treats declared ``.pth`` files as
inert bytes.  Root descriptors remain open in the evidence object.  A future
consumer may append only a retained, descriptor-backed import root directly
to its import search path; it must not call ``site.addsitedir`` or execute
``.pth`` files.  The runtime package is a declared pure-Python closure, but a
formal bootstrap must still verify the exact loader and module origin.

This is not an immutable runtime.  Sequential traversal has no atomic
observation point: a same-UID writer can change an earlier file while a later
file is checked, or change executable/tree bytes after verification.  The
verified executable descriptor is also not retained for execution.  These
residuals always block a claim of complete runtime closure.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from weakref import finalize

from tgvf_rl.ops.runtime_locator_manifest import (
    RUNTIME_LOCATOR_MANIFEST_SCHEMA,
    RuntimeExecutableDeclaration,
    RuntimeFileDeclaration,
    RuntimeLocatorManifest,
    RuntimeLocatorManifestError,
    RuntimeLocatorVerificationError,
    RuntimeTreeDeclaration,
    load_runtime_locator_manifest,
)
from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    retain_regular_file_absolute_nofollow,
)


RUNTIME_LOCATOR_SCAFFOLD_EVIDENCE_SCHEMA = "tgvf-runtime-locator-scaffold-evidence-v1"
RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL = (
    "same-uid-mutable-executable-and-runtime-trees-have-no-atomic-observation-"
    "during-or-after-verification-v1"
)
RUNTIME_LOCATOR_SCAFFOLD_BLOCKER = (
    "runtime locator v1 verifies declared mutable executable/tree bytes but "
    "does not execute a retained executable inode, provide an immutable "
    "package, or provide one coherent observation during/after verification"
)
_READ_CHUNK_BYTES = 1024 * 1024
_SCAFFOLD_MINT_SENTINEL = object()


class VerifiedRuntimeLocatorScaffoldEvidence:
    """PID-bound, non-transferable evidence retaining verified root FDs."""

    __slots__ = (
        "__weakref__",
        "_closed",
        "_descriptors",
        "_descriptor_identities",
        "_finalizer",
        "_manifest",
        "_process_id",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedRuntimeLocatorScaffoldEvidence can only be minted by "
            "verify_runtime_locator_manifest_scaffold"
        )

    @classmethod
    def _mint(
        cls,
        manifest: RuntimeLocatorManifest,
        descriptors: tuple[int, ...],
        *,
        _sentinel: object,
    ) -> VerifiedRuntimeLocatorScaffoldEvidence:
        if _sentinel is not _SCAFFOLD_MINT_SENTINEL:
            raise TypeError("runtime locator scaffold mint sentinel differs")
        if type(manifest) is not RuntimeLocatorManifest:
            raise TypeError("runtime locator manifest type differs")
        if type(descriptors) is not tuple or not descriptors:
            raise TypeError("retained runtime root descriptors must be a tuple")
        identities: list[tuple[int, int]] = []
        for descriptor in descriptors:
            if type(descriptor) is not int or descriptor < 0:
                raise TypeError("retained runtime root descriptor differs")
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeLocatorVerificationError(
                    "retained runtime root is not a directory"
                )
            identities.append((metadata.st_dev, metadata.st_ino))
        evidence = object.__new__(cls)
        object.__setattr__(evidence, "_manifest", manifest)
        object.__setattr__(evidence, "_process_id", os.getpid())
        object.__setattr__(evidence, "_descriptors", descriptors)
        object.__setattr__(evidence, "_descriptor_identities", tuple(identities))
        object.__setattr__(evidence, "_closed", False)
        object.__setattr__(
            evidence,
            "_finalizer",
            finalize(evidence, _close_descriptors_best_effort, descriptors),
        )
        return evidence

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedRuntimeLocatorScaffoldEvidence is immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("VerifiedRuntimeLocatorScaffoldEvidence cannot be subclassed")

    def __copy__(self) -> object:
        raise TypeError("runtime locator scaffold evidence is process-local")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("runtime locator scaffold evidence is process-local")

    def __reduce__(self) -> object:
        raise TypeError("runtime locator scaffold evidence is not serializable")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("runtime locator scaffold evidence is not serializable")

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"pid={self._process_id}"
        return (
            "VerifiedRuntimeLocatorScaffoldEvidence("
            f"manifest_identity_sha256={self._manifest.identity_sha256!r}, {state})"
        )

    @property
    def closed(self) -> bool:
        return self._closed or not self._finalizer.alive

    @property
    def manifest(self) -> RuntimeLocatorManifest:
        """Return the declaration, not an atomic observed filesystem snapshot."""

        self._require_usable()
        return self._manifest

    @property
    def runtime_package_sha256(self) -> str:
        """Return the declared import-root identity, not a snapshot digest."""

        self._require_usable()
        return self._manifest.runtime_package_sha256

    @property
    def dependency_roots_sha256(self) -> str:
        """Return declared ordered-root identity, not a snapshot digest."""

        self._require_usable()
        return self._manifest.dependency_roots_sha256

    @property
    def launch_blockers(self) -> tuple[str, ...]:
        self._require_usable()
        return (RUNTIME_LOCATOR_SCAFFOLD_BLOCKER,)

    @property
    def evidence_sha256(self) -> str:
        return _canonical_json_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return declared identities and explicitly non-atomic scan evidence."""

        self._require_usable()
        roots = (self._manifest.runtime_package,) + self._manifest.dependency_roots
        return {
            "schema_version": RUNTIME_LOCATOR_SCAFFOLD_EVIDENCE_SCHEMA,
            "manifest_source_path": str(self._manifest.manifest_source_path),
            "manifest_source_sha256": self._manifest.manifest_source_sha256,
            "manifest_source_byte_length": (self._manifest.manifest_source_byte_length),
            "manifest_identity_sha256": self._manifest.identity_sha256,
            "verified_process_id": self._process_id,
            "cache_tag": self._manifest.cache_tag,
            "executable_path": str(self._manifest.executable.path),
            "executable_sha256": self._manifest.executable.sha256,
            "target_coordinates": list(self._manifest.target_coordinates),
            "runtime_package_sha256": self._manifest.runtime_package_sha256,
            "dependency_roots_sha256": self._manifest.dependency_roots_sha256,
            "retained_roots": [
                {
                    "path": str(tree.root),
                    "device": identity[0],
                    "inode": identity[1],
                }
                for tree, identity in zip(
                    roots,
                    self._descriptor_identities,
                    strict=True,
                )
            ],
            "closure_complete": False,
            "unbound_residuals": [RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL],
        }

    def duplicate_runtime_import_root_directory_fd(self) -> int:
        """Duplicate the retained import root directly above ``tgvf_rl/``."""

        self._require_usable()
        return _duplicate_directory_fd(
            self._descriptors[0],
            self._descriptor_identities[0],
        )

    def duplicate_dependency_root_directory_fds(self) -> tuple[int, ...]:
        self._require_usable()
        duplicates: list[int] = []
        try:
            for descriptor, identity in zip(
                self._descriptors[1:],
                self._descriptor_identities[1:],
                strict=True,
            ):
                duplicates.append(_duplicate_directory_fd(descriptor, identity))
            return tuple(duplicates)
        except BaseException:
            _close_descriptors_best_effort(tuple(duplicates))
            raise

    def close(self) -> None:
        """Idempotently release retained descriptors in any inherited process."""

        # CPython 3.12 is pinned for this scaffold. finalize.detach() removes
        # one live registry entry and returns its callback tuple to exactly one
        # caller; that one-shot transfer is the descriptor-ownership CAS.
        detached = self._finalizer.detach()
        object.__setattr__(self, "_closed", True)
        if detached is None:
            return
        _object, _callback, _args, _kwargs = detached
        _close_descriptors_strict(self._descriptors)

    def __enter__(self) -> VerifiedRuntimeLocatorScaffoldEvidence:
        self._require_usable()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _require_usable(self) -> None:
        if os.getpid() != self._process_id:
            raise RuntimeLocatorVerificationError(
                "runtime locator scaffold evidence belongs to a different process"
            )
        if self._closed or not self._finalizer.alive:
            raise RuntimeLocatorVerificationError(
                "runtime locator scaffold evidence is closed"
            )
        for descriptor, identity in zip(
            self._descriptors,
            self._descriptor_identities,
            strict=True,
        ):
            try:
                metadata = os.fstat(descriptor)
            except OSError as error:
                raise RuntimeLocatorVerificationError(
                    "retained runtime root descriptor is unavailable"
                ) from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != identity
            ):
                raise RuntimeLocatorVerificationError(
                    "retained runtime root descriptor identity differs"
                )
        if self._closed or not self._finalizer.alive:
            raise RuntimeLocatorVerificationError(
                "runtime locator scaffold evidence closed during validation"
            )


def verify_runtime_locator_manifest_scaffold(
    manifest: RuntimeLocatorManifest,
    *,
    expected_cache_tag: str,
    expected_target_coordinates: tuple[str, ...],
) -> VerifiedRuntimeLocatorScaffoldEvidence:
    """Revalidate explicit source, executable, and exact declared trees."""

    if type(manifest) is not RuntimeLocatorManifest:
        raise TypeError("manifest must be exactly RuntimeLocatorManifest")
    if type(expected_cache_tag) is not str:
        raise TypeError("expected_cache_tag must be exactly str")
    if type(expected_target_coordinates) is not tuple or any(
        type(item) is not str for item in expected_target_coordinates
    ):
        raise TypeError("expected_target_coordinates must be a tuple of str")
    if expected_cache_tag != manifest.cache_tag:
        raise RuntimeLocatorVerificationError("runtime cache tag differs")
    if expected_target_coordinates != manifest.target_coordinates:
        raise RuntimeLocatorVerificationError("runtime target coordinates differ")
    rebound = load_runtime_locator_manifest(
        manifest.manifest_source_path,
        expected_source_sha256=manifest.manifest_source_sha256,
        expected_source_byte_length=manifest.manifest_source_byte_length,
    )
    if rebound != manifest:
        raise RuntimeLocatorVerificationError(
            "runtime locator manifest binding changed after load"
        )
    _verify_executable(manifest.executable)

    trees = (manifest.runtime_package,) + manifest.dependency_roots
    descriptors: list[int] = []
    transferred = False
    try:
        for tree in trees:
            descriptors.append(_open_absolute_directory(tree.root))
        for tree, descriptor in zip(trees, descriptors, strict=True):
            _verify_tree(tree, descriptor)
            _verify_root_path_still_bound(tree.root, descriptor)
        evidence = VerifiedRuntimeLocatorScaffoldEvidence._mint(
            manifest,
            tuple(descriptors),
            _sentinel=_SCAFFOLD_MINT_SENTINEL,
        )
        transferred = True
        return evidence
    except RuntimeLocatorVerificationError:
        raise
    except OSError as error:
        raise RuntimeLocatorVerificationError(
            "runtime locator operating-system verification failed"
        ) from error
    finally:
        if not transferred:
            _close_descriptors_best_effort(tuple(descriptors))


def _verify_executable(declaration: RuntimeExecutableDeclaration) -> None:
    """Verify one nofollow inode, then close it with the TOCTOU residual open."""

    try:
        with retain_regular_file_absolute_nofollow(declaration.path) as retained:
            descriptor = retained.fileno()
            before = os.fstat(descriptor)
            if before.st_size != declaration.byte_length:
                raise RuntimeLocatorVerificationError("runtime executable size differs")
            if before.st_mode & 0o111 == 0:
                raise RuntimeLocatorVerificationError(
                    "runtime executable lacks an executable mode bit"
                )
            digest = sha256()
            observed_length = 0
            while True:
                limit = min(
                    _READ_CHUNK_BYTES,
                    declaration.byte_length - observed_length + 1,
                )
                block = os.read(descriptor, limit)
                if not block:
                    break
                digest.update(block)
                observed_length += len(block)
                if observed_length > declaration.byte_length:
                    raise RuntimeLocatorVerificationError(
                        "runtime executable grew while read"
                    )
            after = os.fstat(descriptor)
            if _metadata_signature(after) != _metadata_signature(before):
                raise RuntimeLocatorVerificationError(
                    "runtime executable changed while read"
                )
            if observed_length != declaration.byte_length:
                raise RuntimeLocatorVerificationError(
                    "runtime executable read was incomplete"
                )
            if digest.hexdigest() != declaration.sha256:
                raise RuntimeLocatorVerificationError(
                    "runtime executable SHA256 differs"
                )
    except RuntimeLocatorVerificationError:
        raise
    except (OSError, SecureFileReadError) as error:
        raise RuntimeLocatorVerificationError(
            "runtime executable is unavailable, non-regular, or a symlink"
        ) from error


def _verify_tree(tree: RuntimeTreeDeclaration, root_descriptor: int) -> None:
    declared_directories = set(tree.directories)
    declared_files = {item.path: item for item in tree.files}
    observed_directories: set[str] = set()
    observed_files: set[str] = set()

    def walk(directory_descriptor: int, prefix: str) -> None:
        before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeLocatorVerificationError("runtime directory kind changed")
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise RuntimeLocatorVerificationError(
                f"runtime directory is unreadable: {prefix or '.'}"
            ) from error
        for name in names:
            _require_directory_entry_name(name)
            relative_path = f"{prefix}/{name}" if prefix else name
            if _is_forbidden_bytecode_path(relative_path):
                raise RuntimeLocatorVerificationError(
                    f"runtime tree contains forbidden bytecode: {relative_path}"
                )
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise RuntimeLocatorVerificationError(
                    f"runtime entry became unavailable: {relative_path}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeLocatorVerificationError(
                    f"runtime tree contains a symlink: {relative_path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                if relative_path not in declared_directories:
                    raise RuntimeLocatorVerificationError(
                        f"runtime tree contains extra directory: {relative_path}"
                    )
                child_descriptor = _open_directory_at(
                    directory_descriptor,
                    name,
                    expected_metadata=metadata,
                    relative_path=relative_path,
                )
                observed_directories.add(relative_path)
                try:
                    walk(child_descriptor, relative_path)
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(metadata.st_mode):
                declaration = declared_files.get(relative_path)
                if declaration is None:
                    raise RuntimeLocatorVerificationError(
                        f"runtime tree contains extra file: {relative_path}"
                    )
                _verify_file_at(
                    directory_descriptor,
                    name,
                    relative_path=relative_path,
                    expected_metadata=metadata,
                    declaration=declaration,
                )
                observed_files.add(relative_path)
            else:
                raise RuntimeLocatorVerificationError(
                    "runtime tree contains a non-regular entry: " + relative_path
                )
        after = os.fstat(directory_descriptor)
        if _metadata_signature(after) != _metadata_signature(before):
            raise RuntimeLocatorVerificationError(
                f"runtime directory changed while scanned: {prefix or '.'}"
            )

    walk(root_descriptor, "")
    missing = (declared_directories - observed_directories) | (
        set(declared_files) - observed_files
    )
    if missing:
        raise RuntimeLocatorVerificationError(
            f"runtime tree is missing declared entries: {sorted(missing)}"
        )


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_metadata: os.stat_result,
    relative_path: str,
) -> int:
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise RuntimeLocatorVerificationError(
            f"runtime directory is unavailable or a symlink: {relative_path}"
        ) from error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != (
            expected_metadata.st_dev,
            expected_metadata.st_ino,
        ):
            raise RuntimeLocatorVerificationError(
                f"runtime directory identity changed: {relative_path}"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_file_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
    expected_metadata: os.stat_result,
    declaration: RuntimeFileDeclaration,
) -> None:
    try:
        descriptor = os.open(
            name,
            _file_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise RuntimeLocatorVerificationError(
            f"runtime file is unavailable or a symlink: {relative_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            expected_metadata.st_dev,
            expected_metadata.st_ino,
        ):
            raise RuntimeLocatorVerificationError(
                f"runtime file identity changed: {relative_path}"
            )
        if before.st_size != declaration.byte_length:
            raise RuntimeLocatorVerificationError(
                f"runtime file size differs: {relative_path}"
            )
        digest = sha256()
        observed_length = 0
        while True:
            block = os.read(descriptor, _READ_CHUNK_BYTES)
            if not block:
                break
            observed_length += len(block)
            if observed_length > declaration.byte_length:
                raise RuntimeLocatorVerificationError(
                    f"runtime file grew while read: {relative_path}"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        if _metadata_signature(after) != _metadata_signature(before):
            raise RuntimeLocatorVerificationError(
                f"runtime file changed while read: {relative_path}"
            )
        if observed_length != declaration.byte_length:
            raise RuntimeLocatorVerificationError(
                f"runtime file read was incomplete: {relative_path}"
            )
        if digest.hexdigest() != declaration.sha256:
            raise RuntimeLocatorVerificationError(
                f"runtime file SHA256 differs: {relative_path}"
            )
    finally:
        os.close(descriptor)


def _open_absolute_directory(path: Path) -> int:
    descriptor = os.open("/", _directory_open_flags())
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeLocatorVerificationError(
                f"runtime tree root is not a directory: {path}"
            )
        return descriptor
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, RuntimeLocatorVerificationError):
            raise
        raise RuntimeLocatorVerificationError(
            f"runtime tree root is unavailable or contains a symlink: {path}"
        ) from error


def _verify_root_path_still_bound(path: Path, retained_descriptor: int) -> None:
    rebound_descriptor = _open_absolute_directory(path)
    try:
        retained = os.fstat(retained_descriptor)
        rebound = os.fstat(rebound_descriptor)
        if (retained.st_dev, retained.st_ino) != (rebound.st_dev, rebound.st_ino):
            raise RuntimeLocatorVerificationError(
                f"runtime tree root path changed while verified: {path}"
            )
    finally:
        os.close(rebound_descriptor)


def _require_directory_entry_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeLocatorVerificationError(
            "runtime tree contains a non-canonical entry name"
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RuntimeLocatorVerificationError(
            "runtime tree contains a non-UTF-8 entry name"
        ) from error
    return value


def _is_forbidden_bytecode_path(value: str) -> bool:
    path = PurePosixPath(value)
    return any(part.casefold() == "__pycache__" for part in path.parts) or (
        path.suffix.casefold() == ".pyc"
    )


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int:
        raise RuntimeLocatorVerificationError(f"platform lacks required {name}")
    return value


def _directory_open_flags() -> int:
    if os.open not in os.supports_dir_fd:
        raise RuntimeLocatorVerificationError("platform lacks openat support")
    return (
        os.O_RDONLY
        | _required_flag("O_CLOEXEC")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_NONBLOCK")
        | _required_flag("O_DIRECTORY")
    )


def _file_open_flags() -> int:
    return (
        os.O_RDONLY
        | _required_flag("O_CLOEXEC")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_NONBLOCK")
    )


def _metadata_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _close_descriptors_best_effort(descriptors: tuple[int, ...]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _close_descriptors_strict(descriptors: tuple[int, ...]) -> None:
    """Close every owned descriptor, then expose the first close failure."""

    first_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _duplicate_directory_fd(descriptor: int, identity: tuple[int, int]) -> int:
    """Duplicate then validate the duplicate itself against its bound inode."""

    try:
        duplicate = os.dup(descriptor)
    except OSError as error:
        raise RuntimeLocatorVerificationError(
            "retained runtime root closed before descriptor duplication"
        ) from error
    try:
        metadata = os.fstat(duplicate)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise RuntimeLocatorVerificationError(
                "duplicated runtime root descriptor identity differs"
            )
        return duplicate
    except RuntimeLocatorVerificationError:
        _close_descriptors_best_effort((duplicate,))
        raise
    except OSError as error:
        _close_descriptors_best_effort((duplicate,))
        raise RuntimeLocatorVerificationError(
            "duplicated runtime root descriptor could not be verified"
        ) from error
    except BaseException:
        _close_descriptors_best_effort((duplicate,))
        raise


__all__ = [
    "RUNTIME_LOCATOR_MANIFEST_SCHEMA",
    "RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL",
    "RUNTIME_LOCATOR_SCAFFOLD_BLOCKER",
    "RUNTIME_LOCATOR_SCAFFOLD_EVIDENCE_SCHEMA",
    "RuntimeExecutableDeclaration",
    "RuntimeFileDeclaration",
    "RuntimeLocatorManifest",
    "RuntimeLocatorManifestError",
    "RuntimeLocatorVerificationError",
    "RuntimeTreeDeclaration",
    "VerifiedRuntimeLocatorScaffoldEvidence",
    "load_runtime_locator_manifest",
    "verify_runtime_locator_manifest_scaffold",
]
