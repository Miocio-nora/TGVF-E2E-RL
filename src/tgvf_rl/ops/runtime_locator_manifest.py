"""Strict canonical manifest contract for the runtime-locator scaffold.

This dependency-light leaf parses only an externally SHA-256 and byte-length
bound file.  The JSON field ``runtime_package.root`` is the exact Python
import root above ``tgvf_rl/``, not the package directory itself.  It never
searches ``sys.prefix`` or ``sysconfig``, reads ambient environment variables,
imports ``site``, or executes ``.pth`` files.

The runtime package is deliberately a pure-Python target closure: native
import-library suffixes are forbidden there, while dependency roots may still
declare them.  This scaffold does not emulate every ``FileFinder`` rule; a
formal bootstrap must additionally verify each exact loader and module origin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re

from tgvf_rl.secure_file_read import (
    SecureFileReadError,
    retain_regular_file_absolute_nofollow,
)


RUNTIME_LOCATOR_MANIFEST_SCHEMA = "tgvf-runtime-locator-manifest-v1"

_MANIFEST_FIELDS = {
    "schema_version",
    "cache_tag",
    "executable",
    "target_coordinates",
    "runtime_package",
    "dependency_roots",
}
_EXECUTABLE_FIELDS = {"path", "sha256", "byte_length"}
_TREE_FIELDS = {"root", "directories", "files"}
_FILE_FIELDS = {"path", "sha256", "byte_length"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CACHE_TAG_RE = re.compile(r"^[a-z][a-z0-9_]*-[a-z0-9_]+$")
_TARGET_COORDINATE_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_READ_CHUNK_BYTES = 1024 * 1024


class RuntimeLocatorVerificationError(RuntimeError):
    """I/O, external-binding, or runtime verification refusal.

    Semantic schema and caller-type errors deliberately remain ``ValueError``
    and ``TypeError`` respectively.
    """


class RuntimeLocatorManifestError(RuntimeLocatorVerificationError):
    """The externally bound manifest source cannot be opened or matched."""


@dataclass(frozen=True, slots=True)
class RuntimeFileDeclaration:
    """One exact regular file beneath a declared runtime tree."""

    path: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _require_relative_entry_path(self.path, label="runtime file")
        _require_sha256(self.sha256, label=f"runtime file {self.path} sha256")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError(
                f"runtime file {self.path} byte_length must be non-negative"
            )

    def as_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_record(cls, value: object) -> RuntimeFileDeclaration:
        record = _require_exact_object(value, _FILE_FIELDS, "runtime file")
        return cls(
            path=record["path"],  # type: ignore[arg-type]
            sha256=record["sha256"],  # type: ignore[arg-type]
            byte_length=record["byte_length"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeTreeDeclaration:
    """Exact inventory at one absolute future import-search root."""

    root: Path
    directories: tuple[str, ...]
    files: tuple[RuntimeFileDeclaration, ...]

    def __post_init__(self) -> None:
        root = _lexical_absolute_path(self.root, label="runtime tree root")
        if root == Path("/"):
            raise ValueError("runtime tree root may not be the filesystem root")
        if _is_forbidden_bytecode_path(root.parts[-1]):
            raise ValueError("runtime tree root may not be bytecode storage")
        object.__setattr__(self, "root", root)
        if type(self.directories) is not tuple:
            raise TypeError("runtime tree directories must be exactly tuple")
        if any(type(item) is not str for item in self.directories):
            raise TypeError("runtime tree directory entries must be exactly str")
        for directory in self.directories:
            _require_relative_entry_path(directory, label="runtime directory")
        if self.directories != tuple(sorted(set(self.directories))):
            raise ValueError("runtime tree directories must be unique and path-sorted")
        if type(self.files) is not tuple:
            raise TypeError("runtime tree files must be exactly tuple")
        if any(type(item) is not RuntimeFileDeclaration for item in self.files):
            raise TypeError("runtime tree files have an unexpected type")
        if not self.files:
            raise ValueError("runtime tree must declare at least one regular file")
        file_paths = tuple(item.path for item in self.files)
        if file_paths != tuple(sorted(set(file_paths))):
            raise ValueError("runtime tree files must be unique and path-sorted")
        directory_set = set(self.directories)
        if directory_set.intersection(file_paths):
            raise ValueError("runtime tree path has conflicting entry kinds")
        for entry in (*self.directories, *file_paths):
            if _relative_parent_paths(entry) - directory_set:
                raise ValueError(
                    f"runtime tree entry {entry!r} has an undeclared parent"
                )

    @property
    def tree_sha256(self) -> str:
        return _canonical_json_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "directories": list(self.directories),
            "files": [item.as_record() for item in self.files],
        }

    @classmethod
    def from_record(cls, value: object) -> RuntimeTreeDeclaration:
        record = _require_exact_object(value, _TREE_FIELDS, "runtime tree")
        directories = record["directories"]
        files = record["files"]
        if type(directories) is not list:
            raise ValueError("runtime tree directories must be a JSON array")
        if type(files) is not list:
            raise ValueError("runtime tree files must be a JSON array")
        return cls(
            root=_declared_absolute_path(record["root"], label="runtime tree root"),
            directories=tuple(directories),  # type: ignore[arg-type]
            files=tuple(RuntimeFileDeclaration.from_record(item) for item in files),
        )


@dataclass(frozen=True, slots=True)
class RuntimeExecutableDeclaration:
    """Exact regular executable authorized by path, length, and bytes."""

    path: Path
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _lexical_absolute_path(self.path, label="runtime executable path"),
        )
        _require_sha256(self.sha256, label="runtime executable sha256")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("runtime executable byte_length must be non-negative")

    def as_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_record(cls, value: object) -> RuntimeExecutableDeclaration:
        record = _require_exact_object(
            value,
            _EXECUTABLE_FIELDS,
            "runtime executable",
        )
        return cls(
            path=_declared_absolute_path(
                record["path"],
                label="runtime executable path",
            ),
            sha256=record["sha256"],  # type: ignore[arg-type]
            byte_length=record["byte_length"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeLocatorManifest:
    """Canonical semantic manifest and externally bound source identity."""

    manifest_source_path: Path
    manifest_source_sha256: str
    manifest_source_byte_length: int
    cache_tag: str
    executable: RuntimeExecutableDeclaration
    target_coordinates: tuple[str, ...]
    runtime_package: RuntimeTreeDeclaration
    dependency_roots: tuple[RuntimeTreeDeclaration, ...]
    schema_version: str = RUNTIME_LOCATOR_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_LOCATOR_MANIFEST_SCHEMA:
            raise ValueError("runtime locator manifest schema differs")
        source = _lexical_absolute_path(
            self.manifest_source_path,
            label="runtime locator manifest source",
        )
        object.__setattr__(self, "manifest_source_path", source)
        _require_sha256(
            self.manifest_source_sha256,
            label="runtime locator manifest source sha256",
        )
        if (
            type(self.manifest_source_byte_length) is not int
            or self.manifest_source_byte_length <= 0
        ):
            raise ValueError("runtime locator manifest source length must be positive")
        _require_cache_tag(self.cache_tag)
        if type(self.executable) is not RuntimeExecutableDeclaration:
            raise TypeError("runtime executable declaration type differs")
        if type(self.target_coordinates) is not tuple:
            raise TypeError("runtime target_coordinates must be exactly tuple")
        if not self.target_coordinates:
            raise ValueError("runtime target_coordinates may not be empty")
        for coordinate in self.target_coordinates:
            _require_target_coordinate(coordinate)
        if len(set(self.target_coordinates)) != len(self.target_coordinates):
            raise ValueError("runtime target_coordinates may not repeat")
        if type(self.runtime_package) is not RuntimeTreeDeclaration:
            raise TypeError("runtime package declaration type differs")
        _require_runtime_package_import_contract(
            self.runtime_package,
            self.target_coordinates,
        )
        if type(self.dependency_roots) is not tuple:
            raise TypeError("runtime dependency_roots must be exactly tuple")
        if not self.dependency_roots:
            raise ValueError("runtime dependency_roots may not be empty")
        if any(
            type(item) is not RuntimeTreeDeclaration for item in self.dependency_roots
        ):
            raise TypeError("runtime dependency_roots have an unexpected type")
        roots = (self.runtime_package.root,) + tuple(
            item.root for item in self.dependency_roots
        )
        _require_disjoint_roots(roots)
        if any(_is_beneath(source, root) for root in roots):
            raise ValueError("runtime locator manifest source may not be inside a root")

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.as_record())

    @property
    def runtime_package_sha256(self) -> str:
        return self.runtime_package.tree_sha256

    @property
    def dependency_roots_sha256(self) -> str:
        """Return the order-sensitive dependency-root identity."""

        return _canonical_json_sha256(
            [item.as_record() for item in self.dependency_roots]
        )

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cache_tag": self.cache_tag,
            "executable": self.executable.as_record(),
            "target_coordinates": list(self.target_coordinates),
            "runtime_package": self.runtime_package.as_record(),
            "dependency_roots": [item.as_record() for item in self.dependency_roots],
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.as_record()).decode("utf-8")


def load_runtime_locator_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_source_sha256: str,
    expected_source_byte_length: int,
) -> RuntimeLocatorManifest:
    """Load one digest/length-bound manifest in its only canonical spelling."""

    _require_sha256(
        expected_source_sha256,
        label="expected runtime locator manifest source sha256",
    )
    if type(expected_source_byte_length) is not int or expected_source_byte_length <= 0:
        raise ValueError("expected manifest source length must be positive")
    source = _lexical_absolute_path(
        manifest_path,
        label="runtime locator manifest source",
    )
    raw = _read_bound_manifest_source(source, expected_source_byte_length)
    observed_source_sha256 = sha256(raw).hexdigest()
    if observed_source_sha256 != expected_source_sha256:
        raise RuntimeLocatorManifestError(
            "runtime locator manifest source SHA256 differs"
        )
    payload = _strict_json_object(raw, label="runtime locator manifest")
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise ValueError("runtime locator manifest JSON is not canonical")
    _require_exact_fields(payload, _MANIFEST_FIELDS, "runtime locator manifest")
    if payload["schema_version"] != RUNTIME_LOCATOR_MANIFEST_SCHEMA:
        raise ValueError("runtime locator manifest schema differs")
    targets = payload["target_coordinates"]
    dependencies = payload["dependency_roots"]
    if type(targets) is not list:
        raise ValueError("runtime target_coordinates must be a JSON array")
    if type(dependencies) is not list:
        raise ValueError("runtime dependency_roots must be a JSON array")
    manifest = RuntimeLocatorManifest(
        manifest_source_path=source,
        manifest_source_sha256=observed_source_sha256,
        manifest_source_byte_length=expected_source_byte_length,
        cache_tag=payload["cache_tag"],  # type: ignore[arg-type]
        executable=RuntimeExecutableDeclaration.from_record(payload["executable"]),
        target_coordinates=tuple(targets),  # type: ignore[arg-type]
        runtime_package=RuntimeTreeDeclaration.from_record(payload["runtime_package"]),
        dependency_roots=tuple(
            RuntimeTreeDeclaration.from_record(item) for item in dependencies
        ),
    )
    if manifest.to_json().encode("utf-8") + b"\n" != raw:
        raise ValueError("runtime locator semantic record is not canonical")
    return manifest


def _read_bound_manifest_source(path: Path, expected_length: int) -> bytes:
    """Check descriptor size before a bounded read of the nofollow inode."""

    try:
        with retain_regular_file_absolute_nofollow(path) as retained:
            descriptor = retained.fileno()
            before = os.fstat(descriptor)
            if before.st_size != expected_length:
                raise RuntimeLocatorManifestError(
                    "runtime locator manifest source size differs"
                )
            chunks: list[bytes] = []
            observed_length = 0
            while True:
                limit = min(
                    _READ_CHUNK_BYTES,
                    expected_length - observed_length + 1,
                )
                block = os.read(descriptor, limit)
                if not block:
                    break
                chunks.append(block)
                observed_length += len(block)
                if observed_length > expected_length:
                    raise RuntimeLocatorManifestError(
                        "runtime locator manifest grew while read"
                    )
            after = os.fstat(descriptor)
            if _metadata_signature(after) != _metadata_signature(before):
                raise RuntimeLocatorManifestError(
                    "runtime locator manifest changed while read"
                )
            if observed_length != expected_length:
                raise RuntimeLocatorManifestError(
                    "runtime locator manifest read was incomplete"
                )
            return b"".join(chunks)
    except RuntimeLocatorManifestError:
        raise
    except (OSError, SecureFileReadError) as error:
        raise RuntimeLocatorManifestError(
            "runtime locator manifest is unavailable, non-regular, or a symlink"
        ) from error


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _require_exact_object(
    value: object,
    expected_fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    _require_exact_fields(value, expected_fields, label)
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected_fields: set[str],
    label: str,
) -> None:
    if set(value) != expected_fields:
        raise ValueError(f"{label} fields differ")


def _declared_absolute_path(value: object, *, label: str) -> Path:
    if type(value) is not str:
        raise ValueError(f"{label} must be an absolute canonical string")
    return _lexical_absolute_path(value, label=label)


def _lexical_absolute_path(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty text path")
    try:
        raw.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be valid UTF-8 text") from error
    pure = PurePosixPath(raw)
    if (
        not pure.is_absolute()
        or pure.root != "/"
        or pure.as_posix() != raw
        or any(component in {"", ".", ".."} for component in pure.parts[1:])
        or any(_has_control_character(component) for component in pure.parts[1:])
    ):
        raise ValueError(f"{label} must be a canonical POSIX absolute path")
    return Path(raw)


def _require_relative_entry_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} path must be canonical relative text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} path must be valid UTF-8 text") from error
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(component in {"", ".", ".."} for component in pure.parts)
        or any(_has_control_character(component) for component in pure.parts)
    ):
        raise ValueError(f"{label} path must be canonical and beneath its root")
    if _is_forbidden_bytecode_path(value):
        raise ValueError(f"{label} path may not contain pyc or __pycache__")
    return value


def _relative_parent_paths(value: str) -> set[str]:
    parents: set[str] = set()
    parent = PurePosixPath(value).parent
    while parent != PurePosixPath("."):
        parents.add(parent.as_posix())
        parent = parent.parent
    return parents


def _is_forbidden_bytecode_path(value: str) -> bool:
    path = PurePosixPath(value)
    return any(part.casefold() == "__pycache__" for part in path.parts) or (
        path.suffix.casefold() == ".pyc"
    )


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_cache_tag(value: object) -> str:
    if type(value) is not str or not _CACHE_TAG_RE.fullmatch(value):
        raise ValueError("runtime cache_tag must be one canonical explicit tag")
    return value


def _require_target_coordinate(value: object) -> str:
    if type(value) is not str or not _TARGET_COORDINATE_RE.fullmatch(value):
        raise ValueError("runtime target must be an exact module:callable coordinate")
    return value


def _require_disjoint_roots(roots: tuple[Path, ...]) -> None:
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if _is_beneath(left, right) or _is_beneath(right, left):
                raise ValueError("runtime and dependency roots must be disjoint")


def _require_runtime_package_import_contract(
    runtime_package: RuntimeTreeDeclaration,
    target_coordinates: tuple[str, ...],
) -> None:
    """Bind one pure-Python import root and unambiguous target modules."""

    declared_files = {item.path for item in runtime_package.files}
    native_files = sorted(
        path
        for path in declared_files
        if PurePosixPath(path).suffix.casefold() in {".so", ".pyd", ".dll", ".dylib"}
    )
    if native_files:
        raise ValueError(
            "runtime package pure-Python closure contains a native import library: "
            f"{native_files}"
        )
    if "tgvf_rl/__init__.py" not in declared_files:
        raise ValueError("runtime package import root must declare tgvf_rl/__init__.py")
    for coordinate in target_coordinates:
        module_name = coordinate.partition(":")[0]
        if module_name != "tgvf_rl" and not module_name.startswith("tgvf_rl."):
            raise ValueError("runtime target must be in the tgvf_rl namespace")
        module_parts = module_name.split(".")
        for length in range(1, len(module_parts)):
            prefix_path = "/".join(module_parts[:length])
            if f"{prefix_path}/__init__.py" not in declared_files:
                raise ValueError(
                    "runtime target intermediate package has no __init__.py: "
                    f"{prefix_path}"
                )
            if f"{prefix_path}.py" in declared_files:
                raise ValueError(
                    "runtime target intermediate package is shadowed by a module: "
                    f"{prefix_path}.py"
                )
        module_path = module_name.replace(".", "/")
        candidates = {
            f"{module_path}.py",
            f"{module_path}/__init__.py",
        }
        observed = candidates.intersection(declared_files)
        if not observed:
            raise ValueError(
                f"runtime target module {module_name!r} has no import candidate"
            )
        if len(observed) != 1:
            raise ValueError(
                f"runtime target module {module_name!r} has ambiguous import candidates"
            )


def _is_beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _metadata_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "RUNTIME_LOCATOR_MANIFEST_SCHEMA",
    "RuntimeExecutableDeclaration",
    "RuntimeFileDeclaration",
    "RuntimeLocatorManifest",
    "RuntimeLocatorManifestError",
    "RuntimeLocatorVerificationError",
    "RuntimeTreeDeclaration",
    "load_runtime_locator_manifest",
]
