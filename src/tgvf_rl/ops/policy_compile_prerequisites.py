"""Content-bound minimum prerequisites for lazy Policy runtime compilation.

The v1 manifest intentionally binds only the compiler front doors and the two
Python headers whose absence exposed the original launch failure.  It does not
claim to close the recursive Python header tree or the compiler's assembler,
linker, runtime, and built-in tool dependencies.  Consequently a v1 binding is
auditable and preflightable, but remains a launch blocker until a later schema
defines and verifies those residual dependencies.

This dependency-light module is the canonical implementation.  The historical
``tgvf_rl.framework.verl.compile_prerequisites`` module re-exports these exact
objects, and their established import and pickle coordinates remain bound to
that facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)


POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA = (
    "tgvf-policy-compile-prerequisite-manifest-v1"
)
POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA = (
    "tgvf-policy-compile-prerequisite-binding-v2"
)
POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA = (
    "tgvf-policy-compile-prerequisite-receipt-v2"
)
POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY = "minimum-declared-prerequisites-v1"
POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL = (
    "recursive-python-headers-and-system-toolchain-unbound-v1"
)
POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER = (
    "an explicit Policy compile-prerequisite manifest is required"
)
POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER = (
    "compile-prerequisite manifest v1 binds only the minimum declared files; "
    "recursive Python headers and the compiler system-toolchain remain unbound"
)

_FILE_NAMES = ("c_compiler", "cxx_compiler", "python_h", "pyconfig_h")
_EXECUTABLE_REQUIREMENTS = {
    "c_compiler": True,
    "cxx_compiler": True,
    "python_h": False,
    "pyconfig_h": False,
}
_MANIFEST_FIELDS = {"schema_version", "closure_policy", "files"}
_FILE_FIELDS = {"path", "sha256", "byte_length", "executable_required"}
_RECEIPT_FIELDS = {
    "schema_version",
    "manifest_source_path",
    "manifest_source_sha256",
    "binding_sha256",
    "closure_policy",
    "closure_complete",
    "unbound_residuals",
    "files",
    "receipt_sha256",
}
_RECEIPT_FILE_FIELDS = {
    "name",
    "declared_path",
    "sha256",
    "byte_length",
    "executable_required",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PolicyCompilePrerequisiteFile:
    """One exact file declaration from the strict manifest."""

    name: str
    path: Path
    sha256: str
    byte_length: int
    executable_required: bool

    def __post_init__(self) -> None:
        if self.name not in _FILE_NAMES:
            raise ValueError(f"unknown compile-prerequisite file name: {self.name!r}")
        path = Path(self.path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"compile-prerequisite {self.name} path must be absolute")
        if "\x00" in os.fspath(path):
            raise ValueError(f"compile-prerequisite {self.name} path contains NUL")
        object.__setattr__(self, "path", path)
        _require_sha256(self.sha256, f"compile-prerequisite {self.name} sha256")
        if type(self.byte_length) is not int or self.byte_length <= 0:
            raise ValueError(
                f"compile-prerequisite {self.name} byte_length must be positive"
            )
        if type(self.executable_required) is not bool:
            raise TypeError(
                f"compile-prerequisite {self.name} executable_required must be bool"
            )
        if self.executable_required is not _EXECUTABLE_REQUIREMENTS[self.name]:
            raise ValueError(
                f"compile-prerequisite {self.name} executable requirement differs"
            )

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "executable_required": self.executable_required,
        }


@dataclass(frozen=True, slots=True)
class PolicyCompilePrerequisiteBinding:
    """Manifest provenance plus the minimum declared prerequisite identities."""

    manifest_source_path: Path
    manifest_source_sha256: str
    files: tuple[PolicyCompilePrerequisiteFile, ...]
    closure_policy: str = POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY
    schema_version: str = POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA:
            raise ValueError("Policy compile-prerequisite binding schema differs")
        if self.closure_policy != POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY:
            raise ValueError("Policy compile-prerequisite closure policy differs")
        source = Path(self.manifest_source_path)
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("compile-prerequisite manifest source must be absolute")
        object.__setattr__(self, "manifest_source_path", source)
        _require_sha256(
            self.manifest_source_sha256,
            "compile-prerequisite manifest source sha256",
        )
        files = tuple(self.files)
        if (
            any(not isinstance(item, PolicyCompilePrerequisiteFile) for item in files)
            or tuple(item.name for item in files) != _FILE_NAMES
        ):
            raise ValueError("compile-prerequisite binding file inventory differs")
        object.__setattr__(self, "files", files)
        if self.python_h.path.name != "Python.h":
            raise ValueError("python_h must declare Python.h")
        if self.pyconfig_h.path.name != "pyconfig.h":
            raise ValueError("pyconfig_h must declare pyconfig.h")
        if self.python_h.path.parent != self.pyconfig_h.path.parent:
            raise ValueError("Python.h and pyconfig.h must share one include directory")
        if self.python_include.name != "python3.12":
            raise ValueError("Policy rollout requires an explicit Python 3.12 include")

    @property
    def _by_name(self) -> Mapping[str, PolicyCompilePrerequisiteFile]:
        return MappingProxyType({item.name: item for item in self.files})

    @property
    def c_compiler(self) -> Path:
        return self._by_name["c_compiler"].path

    @property
    def cxx_compiler(self) -> Path:
        return self._by_name["cxx_compiler"].path

    @property
    def python_h(self) -> PolicyCompilePrerequisiteFile:
        return self._by_name["python_h"]

    @property
    def pyconfig_h(self) -> PolicyCompilePrerequisiteFile:
        return self._by_name["pyconfig_h"]

    @property
    def python_include(self) -> Path:
        return self.python_h.path.parent

    @property
    def python_include_root(self) -> Path:
        return self.python_include.parent

    @property
    def cpath(self) -> str:
        return os.pathsep.join(
            (str(self.python_include_root), str(self.python_include))
        )

    @property
    def launch_blockers(self) -> tuple[str, ...]:
        return (POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER,)

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_source_path": str(self.manifest_source_path),
            "manifest_source_sha256": self.manifest_source_sha256,
            "closure_policy": self.closure_policy,
            "closure_complete": False,
            "unbound_residuals": [POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL],
            "files": [item.as_record() for item in self.files],
            "cpath": self.cpath,
        }


@dataclass(frozen=True, slots=True)
class PolicyCompilePrerequisiteFileReceipt:
    """Observed content identity for one securely opened declaration."""

    name: str
    declared_path: Path
    sha256: str
    byte_length: int
    executable_required: bool

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "declared_path": str(self.declared_path),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "executable_required": self.executable_required,
        }


@dataclass(frozen=True, slots=True)
class PolicyCompilePrerequisiteReceipt:
    """Proof that every minimum declaration matched immediately at preflight."""

    binding: PolicyCompilePrerequisiteBinding
    files: tuple[PolicyCompilePrerequisiteFileReceipt, ...]
    schema_version: str = POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA:
            raise ValueError("Policy compile-prerequisite receipt schema differs")
        expected = tuple(
            (
                item.name,
                item.path,
                item.sha256,
                item.byte_length,
                item.executable_required,
            )
            for item in self.binding.files
        )
        observed = tuple(
            (
                item.name,
                item.declared_path,
                item.sha256,
                item.byte_length,
                item.executable_required,
            )
            for item in self.files
        )
        if observed != expected:
            raise ValueError("compile-prerequisite receipt differs from manifest")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_json_sha256(self.content_record())

    def content_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_source_path": str(self.binding.manifest_source_path),
            "manifest_source_sha256": self.binding.manifest_source_sha256,
            "binding_sha256": self.binding.identity_sha256,
            "closure_policy": self.binding.closure_policy,
            "closure_complete": False,
            "unbound_residuals": [POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL],
            "files": [item.as_record() for item in self.files],
        }

    def as_record(self) -> dict[str, object]:
        return {**self.content_record(), "receipt_sha256": self.receipt_sha256}


def load_policy_compile_prerequisite_manifest(
    manifest_path: str | Path,
) -> PolicyCompilePrerequisiteBinding:
    """Load one exact UTF-8 JSON manifest without following symlinks."""

    source = _lexical_absolute_path(manifest_path, label="manifest source")
    raw, _ = _read_regular_file_without_symlinks(source)
    payload = _strict_json_object(raw, label="compile-prerequisite manifest")
    _require_exact_fields(payload, _MANIFEST_FIELDS, "compile-prerequisite manifest")
    if payload["schema_version"] != POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA:
        raise ValueError("Policy compile-prerequisite manifest schema differs")
    if payload["closure_policy"] != POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY:
        raise ValueError("Policy compile-prerequisite manifest closure policy differs")
    file_payloads = payload["files"]
    if not isinstance(file_payloads, dict) or set(file_payloads) != set(_FILE_NAMES):
        raise ValueError("compile-prerequisite manifest file inventory differs")
    declarations: list[PolicyCompilePrerequisiteFile] = []
    for name in _FILE_NAMES:
        item = file_payloads[name]
        if not isinstance(item, dict):
            raise ValueError(f"compile-prerequisite {name} must be an object")
        _require_exact_fields(item, _FILE_FIELDS, f"compile-prerequisite {name}")
        declarations.append(
            PolicyCompilePrerequisiteFile(
                name=name,
                path=_declared_absolute_path(item["path"], label=name),
                sha256=item["sha256"],
                byte_length=item["byte_length"],
                executable_required=item["executable_required"],
            )
        )
    return PolicyCompilePrerequisiteBinding(
        manifest_source_path=source,
        manifest_source_sha256=sha256(raw).hexdigest(),
        files=tuple(declarations),
    )


def preflight_policy_compile_prerequisites(
    binding: PolicyCompilePrerequisiteBinding,
) -> PolicyCompilePrerequisiteReceipt:
    """Verify the manifest source and every minimum declared file exactly."""

    if not isinstance(binding, PolicyCompilePrerequisiteBinding):
        raise TypeError("binding must be PolicyCompilePrerequisiteBinding")
    manifest_raw, _ = _read_regular_file_without_symlinks(binding.manifest_source_path)
    if sha256(manifest_raw).hexdigest() != binding.manifest_source_sha256:
        raise RuntimeError("Policy compile-prerequisite manifest changed after load")
    rebound = load_policy_compile_prerequisite_manifest(binding.manifest_source_path)
    if rebound != binding:
        raise RuntimeError("Policy compile-prerequisite manifest binding changed")
    receipts = tuple(_verify_declared_file(item) for item in binding.files)
    return PolicyCompilePrerequisiteReceipt(binding=binding, files=receipts)


def materialize_policy_compile_prerequisite_receipt(
    receipt: PolicyCompilePrerequisiteReceipt,
    *,
    state_directory: str | Path,
) -> Path:
    """Write one private content-addressed attestation for child revalidation."""

    if not isinstance(receipt, PolicyCompilePrerequisiteReceipt):
        raise TypeError("receipt must be PolicyCompilePrerequisiteReceipt")
    state = _lexical_absolute_path(state_directory, label="runtime state directory")
    _make_private_directory(state)
    raw = _canonical_json_bytes(receipt.as_record()) + b"\n"
    path = state / f"compile-prerequisite-{receipt.receipt_sha256}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        existing, _ = _read_regular_file_without_symlinks(path)
        if existing != raw:
            raise RuntimeError("content-addressed prerequisite receipt differs")
        return path
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def verify_policy_compile_prerequisite_receipt(
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_binding_sha256: str,
    expected_manifest_sha256: str,
) -> PolicyCompilePrerequisiteReceipt:
    """Consume an attestation and re-open every declaration in this process."""

    for value, label in (
        (expected_receipt_sha256, "expected receipt sha256"),
        (expected_binding_sha256, "expected binding sha256"),
        (expected_manifest_sha256, "expected manifest sha256"),
    ):
        _require_sha256(value, label)
    path = _lexical_absolute_path(receipt_path, label="receipt path")
    raw, _ = _read_regular_file_without_symlinks(path)
    payload = _strict_json_object(raw, label="compile-prerequisite receipt")
    _require_exact_fields(payload, _RECEIPT_FIELDS, "compile-prerequisite receipt")
    content = dict(payload)
    recorded_receipt_sha256 = content.pop("receipt_sha256")
    if recorded_receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError("Policy compile-prerequisite receipt identity differs")
    if _canonical_json_sha256(content) != expected_receipt_sha256:
        raise RuntimeError("Policy compile-prerequisite receipt content differs")
    if content["schema_version"] != POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA:
        raise RuntimeError("Policy compile-prerequisite receipt schema differs")
    if content["binding_sha256"] != expected_binding_sha256:
        raise RuntimeError("Policy compile-prerequisite binding identity differs")
    if content["manifest_source_sha256"] != expected_manifest_sha256:
        raise RuntimeError("Policy compile-prerequisite manifest identity differs")
    if (
        content["closure_policy"] != POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY
        or content["closure_complete"] is not False
        or content["unbound_residuals"] != [POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL]
    ):
        raise RuntimeError("Policy compile-prerequisite closure declaration differs")
    source = _declared_absolute_path(
        content["manifest_source_path"], label="receipt manifest source"
    )
    binding = load_policy_compile_prerequisite_manifest(source)
    if binding.manifest_source_sha256 != expected_manifest_sha256:
        raise RuntimeError("Policy compile-prerequisite manifest changed")
    if binding.identity_sha256 != expected_binding_sha256:
        raise RuntimeError("Policy compile-prerequisite binding changed")
    observed = preflight_policy_compile_prerequisites(binding)
    if observed.receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError(
            "Policy compile prerequisites changed after launcher preflight"
        )
    if content != observed.content_record():
        raise RuntimeError("Policy compile-prerequisite receipt record differs")
    _validate_receipt_file_shape(content["files"])
    return observed


def verify_policy_compile_prerequisites_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    required: bool,
    require_closure_complete: bool = False,
) -> PolicyCompilePrerequisiteReceipt | None:
    """Revalidate a launcher's receipt before importing the lazy runtime."""

    values = os.environ if environment is None else environment
    names = (
        "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_PATH",
        "TGVF_POLICY_COMPILE_PREREQUISITE_RECEIPT_SHA256",
        "TGVF_POLICY_COMPILE_PREREQUISITE_BINDING_SHA256",
        "TGVF_POLICY_COMPILE_PREREQUISITE_MANIFEST_SHA256",
    )
    present = tuple(name for name in names if values.get(name))
    if not present:
        if required:
            raise RuntimeError(
                "Policy compile-prerequisite receipt environment is missing"
            )
        return None
    if len(present) != len(names):
        raise RuntimeError("Policy compile-prerequisite receipt environment is partial")
    receipt = verify_policy_compile_prerequisite_receipt(
        values[names[0]],
        expected_receipt_sha256=values[names[1]],
        expected_binding_sha256=values[names[2]],
        expected_manifest_sha256=values[names[3]],
    )
    if require_closure_complete and receipt.binding.launch_blockers:
        raise RuntimeError(
            "Policy compile-prerequisite closure is incomplete: "
            + "; ".join(receipt.binding.launch_blockers)
        )
    return receipt


def _verify_declared_file(
    declaration: PolicyCompilePrerequisiteFile,
) -> PolicyCompilePrerequisiteFileReceipt:
    raw, mode = _read_regular_file_without_symlinks(declaration.path)
    observed_sha256 = sha256(raw).hexdigest()
    if len(raw) != declaration.byte_length:
        raise RuntimeError(
            f"Policy compile prerequisite size differs: {declaration.path}"
        )
    if observed_sha256 != declaration.sha256:
        raise RuntimeError(
            f"Policy compile prerequisite SHA256 differs: {declaration.path}"
        )
    if declaration.executable_required and mode & 0o111 == 0:
        raise RuntimeError(
            f"Policy compile prerequisite is not executable: {declaration.path}"
        )
    return PolicyCompilePrerequisiteFileReceipt(
        name=declaration.name,
        declared_path=declaration.path,
        sha256=observed_sha256,
        byte_length=len(raw),
        executable_required=declaration.executable_required,
    )


def _read_regular_file_without_symlinks(path: Path) -> tuple[bytes, int]:
    """Read an absolute file with openat/O_NOFOLLOW for every path component."""

    path = _lexical_absolute_path(path, label="prerequisite file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open("/", directory_flags)
    descriptor: int | None = None
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(
                component,
                directory_flags | no_follow,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            path.name,
            os.O_RDONLY | no_follow,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Policy compile prerequisite is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(
                f"Policy compile prerequisite changed while read: {path}"
            )
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise RuntimeError(
                f"Policy compile prerequisite read was incomplete: {path}"
            )
        return raw, after.st_mode
    except OSError as error:
        raise RuntimeError(
            f"Policy compile prerequisite is missing, unreadable, or a symlink: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _make_private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("Policy compile-prerequisite state path is not a directory")
    mode = path.stat().st_mode
    if mode & 0o077:
        path.chmod(mode & ~0o077)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ: {sorted(value)}")


def _validate_receipt_file_shape(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(_FILE_NAMES):
        raise RuntimeError("Policy compile-prerequisite receipt file inventory differs")
    for expected_name, item in zip(_FILE_NAMES, value, strict=True):
        if not isinstance(item, dict) or set(item) != _RECEIPT_FILE_FIELDS:
            raise RuntimeError("Policy compile-prerequisite receipt file fields differ")
        if item["name"] != expected_name:
            raise RuntimeError("Policy compile-prerequisite receipt file order differs")


def _declared_absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"compile-prerequisite {label} path must be a string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"compile-prerequisite {label} path must be lexical absolute")
    return path


def _lexical_absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts or "\x00" in os.fspath(path):
        raise ValueError(f"{label} must be a lexical absolute path")
    return path


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


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:  # pragma: no cover - defensive OS boundary
            raise RuntimeError("could not write prerequisite receipt")
        remaining = remaining[written:]


_HISTORICAL_COMPILE_PREREQUISITES_MODULE = (
    "tgvf_rl.framework.verl.compile_prerequisites"
)
_PUBLIC_CONTRACT_TYPES = (
    PolicyCompilePrerequisiteFile,
    PolicyCompilePrerequisiteBinding,
    PolicyCompilePrerequisiteFileReceipt,
    PolicyCompilePrerequisiteReceipt,
)
for _contract_type in _PUBLIC_CONTRACT_TYPES:
    freeze_public_class_annotations(
        _contract_type,
        implementation_globals=globals(),
    )
    rebind_public_class(
        _contract_type,
        implementation_module=__name__,
        public_module=_HISTORICAL_COMPILE_PREREQUISITES_MODULE,
    )
for _public_function in (
    load_policy_compile_prerequisite_manifest,
    materialize_policy_compile_prerequisite_receipt,
    preflight_policy_compile_prerequisites,
    verify_policy_compile_prerequisite_receipt,
    verify_policy_compile_prerequisites_from_environment,
):
    rebind_public_function(
        _public_function,
        implementation_module=__name__,
        public_module=_HISTORICAL_COMPILE_PREREQUISITES_MODULE,
    )
del _contract_type, _public_function


__all__ = [
    "POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY",
    "POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER",
    "POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER",
    "POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL",
    "PolicyCompilePrerequisiteBinding",
    "PolicyCompilePrerequisiteFile",
    "PolicyCompilePrerequisiteFileReceipt",
    "PolicyCompilePrerequisiteReceipt",
    "load_policy_compile_prerequisite_manifest",
    "materialize_policy_compile_prerequisite_receipt",
    "preflight_policy_compile_prerequisites",
    "verify_policy_compile_prerequisite_receipt",
    "verify_policy_compile_prerequisites_from_environment",
]
