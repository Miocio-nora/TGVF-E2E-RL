from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import socket
import stat
import subprocess
import sys
import threading
from typing import Any

import pytest

from tgvf_rl.ops import runtime_locator as runtime_locator_module
from tgvf_rl.ops import runtime_locator_manifest as runtime_locator_manifest_module
from tgvf_rl.ops.runtime_locator import (
    RUNTIME_LOCATOR_MANIFEST_SCHEMA,
    RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL,
    RUNTIME_LOCATOR_SCAFFOLD_BLOCKER,
    RuntimeFileDeclaration,
    RuntimeLocatorManifest,
    RuntimeLocatorManifestError,
    RuntimeLocatorVerificationError,
    RuntimeTreeDeclaration,
    VerifiedRuntimeLocatorScaffoldEvidence,
    load_runtime_locator_manifest,
    verify_runtime_locator_manifest_scaffold,
)


_REPOSITORY_ROOT = Path(__file__).parents[2]
_CACHE_TAG = "cpython-312"
_TARGETS = (
    "tgvf_rl.module:VALUE",
    "tgvf_rl.nested.worker:main",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tree_record(root: Path) -> dict[str, object]:
    entries = tuple(root.rglob("*"))
    directories = sorted(
        path.relative_to(root).as_posix() for path in entries if path.is_dir()
    )
    files = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
                "byte_length": path.stat().st_size,
            }
            for path in entries
            if path.is_file()
        ),
        key=lambda item: item["path"],
    )
    return {"root": str(root), "directories": directories, "files": files}


@dataclass(frozen=True)
class _Fixture:
    manifest_path: Path
    payload: dict[str, Any]
    source_sha256: str
    source_byte_length: int
    executable: Path
    import_root: Path
    package_root: Path
    dependency_roots: tuple[Path, ...]
    pth_marker: Path


def _write_payload(path: Path, payload: object) -> str:
    raw = _canonical(payload) + b"\n"
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def _make_fixture(tmp_path: Path) -> _Fixture:
    executable = tmp_path / "bin" / "python3.12"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    import_root = tmp_path / "runtime"
    package_root = import_root / "tgvf_rl"
    (package_root / "nested").mkdir(parents=True)
    (package_root / "second").mkdir()
    (package_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package_root / "nested" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nested" / "worker.py").write_text(
        "def main():\n    return 1\n",
        encoding="utf-8",
    )
    (package_root / "second" / "helper.py").write_text(
        "HELPER = True\n",
        encoding="utf-8",
    )

    dependency_one = tmp_path / "dependencies" / "first"
    dependency_two = tmp_path / "dependencies" / "second"
    dependency_one.mkdir(parents=True)
    (dependency_two / "data").mkdir(parents=True)
    (dependency_one / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8")
    (dependency_one / "libfixture.so").write_bytes(b"inert native dependency\n")
    pth_marker = tmp_path / "pth-was-executed"
    (dependency_one / "inert.pth").write_text(
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).touch()\n",
        encoding="utf-8",
    )
    (dependency_two / "beta.py").write_text("BETA = 2\n", encoding="utf-8")
    (dependency_two / "data" / "table.txt").write_text(
        "bound data\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "authority" / "runtime-locator.json"
    manifest_path.parent.mkdir()
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_LOCATOR_MANIFEST_SCHEMA,
        "cache_tag": _CACHE_TAG,
        "executable": {
            "path": str(executable),
            "sha256": _sha(executable),
            "byte_length": executable.stat().st_size,
        },
        "target_coordinates": list(_TARGETS),
        "runtime_package": _tree_record(import_root),
        "dependency_roots": [
            _tree_record(dependency_one),
            _tree_record(dependency_two),
        ],
    }
    source_sha256 = _write_payload(manifest_path, payload)
    return _Fixture(
        manifest_path=manifest_path,
        payload=payload,
        source_sha256=source_sha256,
        source_byte_length=manifest_path.stat().st_size,
        executable=executable,
        import_root=import_root,
        package_root=package_root,
        dependency_roots=(dependency_one, dependency_two),
        pth_marker=pth_marker,
    )


def _load(fixture: _Fixture) -> RuntimeLocatorManifest:
    return load_runtime_locator_manifest(
        fixture.manifest_path,
        expected_source_sha256=fixture.source_sha256,
        expected_source_byte_length=fixture.source_byte_length,
    )


def _verify(
    manifest: RuntimeLocatorManifest,
) -> VerifiedRuntimeLocatorScaffoldEvidence:
    return verify_runtime_locator_manifest_scaffold(
        manifest,
        expected_cache_tag=_CACHE_TAG,
        expected_target_coordinates=_TARGETS,
    )


def test_exact_manifest_verifies_and_retains_ordered_root_descriptors(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)

    with _verify(manifest) as evidence:
        assert manifest.to_json().encode() + b"\n" == fixture.manifest_path.read_bytes()
        assert manifest.manifest_source_sha256 == fixture.source_sha256
        assert manifest.executable.sha256 == _sha(fixture.executable)
        assert manifest.target_coordinates == _TARGETS
        assert manifest.runtime_package.root == fixture.import_root
        assert "tgvf_rl/__init__.py" in {
            item.path for item in manifest.runtime_package.files
        }
        assert manifest.runtime_package_sha256 == evidence.runtime_package_sha256
        assert manifest.dependency_roots_sha256 == evidence.dependency_roots_sha256
        assert evidence.launch_blockers == (RUNTIME_LOCATOR_SCAFFOLD_BLOCKER,)
        record = evidence.as_record()
        assert record["closure_complete"] is False
        assert record["unbound_residuals"] == [RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL]
        assert record["verified_process_id"] == os.getpid()
        assert record["manifest_source_byte_length"] == fixture.source_byte_length
        assert len(record["retained_roots"]) == 3
        assert len(evidence.evidence_sha256) == 64

        package_fd = evidence.duplicate_runtime_import_root_directory_fd()
        dependency_fds = evidence.duplicate_dependency_root_directory_fds()
        try:
            assert stat.S_ISDIR(os.fstat(package_fd).st_mode)
            assert len(dependency_fds) == 2
            assert all(stat.S_ISDIR(os.fstat(fd).st_mode) for fd in dependency_fds)
        finally:
            os.close(package_fd)
            for descriptor in dependency_fds:
                os.close(descriptor)

    assert evidence.closed is True
    with pytest.raises(RuntimeLocatorVerificationError, match="closed"):
        _ = evidence.manifest
    assert not fixture.pth_marker.exists(), ".pth bytes must never be executed"


def test_import_root_loads_authorized_targets_in_fresh_isolated_python(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    evidence = _verify(_load(fixture))
    import_root_fd = evidence.duplicate_runtime_import_root_directory_fd()
    dependency_fds = evidence.duplicate_dependency_root_directory_fds()
    script = """
import importlib
from pathlib import Path
import sys

import_fd, first_dep_fd, second_dep_fd = map(int, sys.argv[1:4])
root = Path(sys.argv[4]).resolve(strict=True)
first_dep = Path(sys.argv[5]).resolve(strict=True)
marker = Path(sys.argv[6])
sys.path.insert(0, f'/proc/self/fd/{import_fd}')
sys.path.append(f'/proc/self/fd/{first_dep_fd}')
sys.path.append(f'/proc/self/fd/{second_dep_fd}')
module = importlib.import_module('tgvf_rl.module')
worker = importlib.import_module('tgvf_rl.nested.worker')
alpha = importlib.import_module('alpha')
assert module.VALUE == 1
assert worker.main() == 1
assert alpha.ALPHA == 1
for imported in (module, worker):
    origin = Path(imported.__spec__.origin).resolve(strict=True)
    assert origin.is_relative_to(root)
    assert Path(imported.__file__).resolve(strict=True) == origin
assert Path(alpha.__spec__.origin).resolve(strict=True).is_relative_to(first_dep)
assert not marker.exists()
"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-P",
                "-S",
                "-c",
                script,
                str(import_root_fd),
                *(str(descriptor) for descriptor in dependency_fds),
                str(fixture.import_root),
                str(fixture.dependency_roots[0]),
                str(fixture.pth_marker),
            ],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            env={},
            pass_fds=(import_root_fd, *dependency_fds),
        )
        assert completed.returncode == 0, completed.stderr
    finally:
        os.close(import_root_fd)
        for descriptor in dependency_fds:
            os.close(descriptor)
        evidence.close()


def test_facade_exposes_typed_io_errors_but_schema_errors_remain_value_errors() -> None:
    assert "RuntimeLocatorManifestError" in runtime_locator_module.__all__
    assert issubclass(RuntimeLocatorManifestError, RuntimeLocatorVerificationError)
    assert issubclass(RuntimeLocatorVerificationError, RuntimeError)
    assert not issubclass(RuntimeLocatorVerificationError, ValueError)


def test_dependency_root_order_and_all_authorized_coordinates_are_digest_bound(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    reordered = RuntimeLocatorManifest(
        manifest_source_path=manifest.manifest_source_path,
        manifest_source_sha256=manifest.manifest_source_sha256,
        manifest_source_byte_length=manifest.manifest_source_byte_length,
        cache_tag=manifest.cache_tag,
        executable=manifest.executable,
        target_coordinates=tuple(reversed(manifest.target_coordinates)),
        runtime_package=manifest.runtime_package,
        dependency_roots=tuple(reversed(manifest.dependency_roots)),
    )

    assert reordered.runtime_package_sha256 == manifest.runtime_package_sha256
    assert reordered.dependency_roots_sha256 != manifest.dependency_roots_sha256
    assert reordered.identity_sha256 != manifest.identity_sha256
    assert [item.root for item in manifest.dependency_roots] == list(
        fixture.dependency_roots
    )


def test_manifest_source_digest_and_one_canonical_spelling_are_required(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    with pytest.raises(RuntimeLocatorVerificationError, match="source SHA256"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256="0" * 64,
            expected_source_byte_length=fixture.source_byte_length,
        )

    raw = fixture.manifest_path.read_bytes().replace(b"{", b"{ ", 1)
    fixture.manifest_path.write_bytes(raw)
    with pytest.raises(ValueError, match="not canonical"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=sha256(raw).hexdigest(),
            expected_source_byte_length=len(raw),
        )


def test_externally_bound_source_length_refuses_before_any_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    read_calls: list[object] = []

    def unexpected_read(*args: object, **_kwargs: object) -> bytes:
        read_calls.append(args)
        raise AssertionError("manifest bytes must not be read after a size mismatch")

    monkeypatch.setattr(runtime_locator_manifest_module.os, "read", unexpected_read)
    with pytest.raises(RuntimeLocatorVerificationError, match="source size differs"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=fixture.source_sha256,
            expected_source_byte_length=fixture.source_byte_length + 1,
        )
    assert read_calls == []


def test_manifest_rejects_duplicate_unknown_nested_and_nonfinite_fields(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    canonical = fixture.manifest_path.read_bytes()
    duplicate = canonical.replace(
        b'"cache_tag":"cpython-312"',
        b'"cache_tag":"cpython-312","cache_tag":"cpython-312"',
        1,
    )
    fixture.manifest_path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate field"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=sha256(duplicate).hexdigest(),
            expected_source_byte_length=len(duplicate),
        )

    nested_duplicate = canonical.replace(
        b'"path":"tgvf_rl/__init__.py"',
        b'"path":"tgvf_rl/__init__.py","path":"tgvf_rl/__init__.py"',
        1,
    )
    assert nested_duplicate != canonical
    fixture.manifest_path.write_bytes(nested_duplicate)
    with pytest.raises(ValueError, match="duplicate field 'path'"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=sha256(nested_duplicate).hexdigest(),
            expected_source_byte_length=len(nested_duplicate),
        )

    unknown = copy.deepcopy(fixture.payload)
    unknown["ambient_default"] = True
    digest = _write_payload(fixture.manifest_path, unknown)
    with pytest.raises(ValueError, match="fields differ"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )

    nested_unknown = copy.deepcopy(fixture.payload)
    nested_unknown["runtime_package"]["files"][0]["mode"] = 0o644
    digest = _write_payload(fixture.manifest_path, nested_unknown)
    with pytest.raises(ValueError, match="fields differ"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )

    nonfinite = canonical.replace(
        b'"cache_tag":"cpython-312"',
        b'"cache_tag":NaN',
        1,
    )
    fixture.manifest_path.write_bytes(nonfinite)
    with pytest.raises(ValueError, match="non-finite"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=sha256(nonfinite).hexdigest(),
            expected_source_byte_length=len(nonfinite),
        )


def test_direct_records_require_exact_builtin_containers_and_safe_field_errors(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    tree_record = copy.deepcopy(fixture.payload["runtime_package"])

    class DictSubclass(dict[str, object]):
        pass

    class ListSubclass(list[object]):
        pass

    with pytest.raises(ValueError, match="must be an object"):
        RuntimeFileDeclaration.from_record(DictSubclass(tree_record["files"][0]))
    tree_record["directories"] = ListSubclass(tree_record["directories"])
    with pytest.raises(ValueError, match="JSON array"):
        RuntimeTreeDeclaration.from_record(tree_record)
    with pytest.raises(ValueError, match="fields differ"):
        RuntimeFileDeclaration.from_record({1: "heterogeneous", "path": "x.py"})


@pytest.mark.parametrize("inventory", ["directories", "files"])
def test_manifest_rejects_reordered_tree_inventory(
    tmp_path: Path,
    inventory: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"][inventory].reverse()
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="unique and path-sorted"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


@pytest.mark.parametrize("empty_field", ["dependency_roots", "runtime_files"])
def test_manifest_refuses_empty_dependency_or_runtime_closure(
    tmp_path: Path,
    empty_field: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    if empty_field == "dependency_roots":
        payload["dependency_roots"] = []
        expected = "dependency_roots may not be empty"
    else:
        payload["runtime_package"]["files"] = []
        expected = "at least one regular file"
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match=expected):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


@pytest.mark.parametrize(
    ("missing_path", "expected"),
    [
        ("tgvf_rl/__init__.py", "must declare tgvf_rl/__init__.py"),
        (
            "tgvf_rl/nested/__init__.py",
            "intermediate package has no __init__.py",
        ),
        ("tgvf_rl/module.py", "has no import candidate"),
    ],
)
def test_import_root_requires_package_init_and_each_target_module(
    tmp_path: Path,
    missing_path: str,
    expected: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"]["files"] = [
        item
        for item in payload["runtime_package"]["files"]
        if item["path"] != missing_path
    ]
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match=expected):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


def test_target_module_rejects_file_and_package_candidate_ambiguity(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"]["directories"].append("tgvf_rl/module")
    payload["runtime_package"]["directories"].sort()
    payload["runtime_package"]["files"].append(
        {
            "path": "tgvf_rl/module/__init__.py",
            "sha256": sha256(b"").hexdigest(),
            "byte_length": 0,
        }
    )
    payload["runtime_package"]["files"].sort(key=lambda item: item["path"])
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="ambiguous import candidates"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


def test_target_module_must_remain_in_tgvf_rl_namespace(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["target_coordinates"] = ["other.module:main"]
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="tgvf_rl namespace"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


def test_target_intermediate_package_rejects_same_name_module_shadow(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"]["files"].append(
        {
            "path": "tgvf_rl/nested.py",
            "sha256": sha256(b"").hexdigest(),
            "byte_length": 0,
        }
    )
    payload["runtime_package"]["files"].sort(key=lambda item: item["path"])
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="intermediate package is shadowed"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


@pytest.mark.parametrize(
    "native_path",
    [
        "tgvf_rl/module.so",
        "tgvf_rl/module.cpython-312-x86_64-linux-gnu.so",
        "tgvf_rl/module.pyd",
        "tgvf_rl/module.dll",
        "tgvf_rl/module.dylib",
        "tgvf_rl/nested/__init__.cpython-312.so",
    ],
)
def test_runtime_package_rejects_native_import_library_shadows(
    tmp_path: Path,
    native_path: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"]["files"].append(
        {
            "path": native_path,
            "sha256": sha256(b"").hexdigest(),
            "byte_length": 0,
        }
    )
    payload["runtime_package"]["files"].sort(key=lambda item: item["path"])
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="pure-Python closure.*native"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra-file", "extra-directory", "tampered"],
)
def test_tree_verification_rejects_missing_extra_and_tampered_entries(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    declared = fixture.package_root / "module.py"
    if mutation == "missing":
        declared.unlink()
        expected = "missing declared entries"
    elif mutation == "extra-file":
        (fixture.package_root / "extra.py").write_text("extra\n", encoding="utf-8")
        expected = "extra file"
    elif mutation == "extra-directory":
        (fixture.package_root / "extra").mkdir()
        expected = "extra directory"
    else:
        declared.write_text("VALUE = 2\n", encoding="utf-8")
        expected = "SHA256 differs"

    with pytest.raises(RuntimeLocatorVerificationError, match=expected):
        _verify(manifest)


@pytest.mark.parametrize("mutation", ["same-size-content", "execute-bit"])
def test_executable_verification_rejects_content_or_mode_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    if mutation == "same-size-content":
        original = fixture.executable.read_bytes()
        changed = original.replace(b"exit 0", b"exit 1")
        assert len(changed) == len(original)
        fixture.executable.write_bytes(changed)
        fixture.executable.chmod(0o755)
        expected = "executable SHA256 differs"
    else:
        fixture.executable.chmod(0o644)
        expected = "executable mode bit"
    with pytest.raises(RuntimeLocatorVerificationError, match=expected):
        _verify(manifest)


def test_root_path_rebinding_after_fd_verification_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    original_verify_tree = runtime_locator_module._verify_tree
    rebound = False

    def verify_then_rebind(tree: object, descriptor: int) -> None:
        nonlocal rebound
        original_verify_tree(tree, descriptor)
        if not rebound and tree.root == fixture.import_root:
            displaced = fixture.import_root.with_name("displaced-import-root")
            fixture.import_root.rename(displaced)
            fixture.import_root.mkdir()
            rebound = True

    monkeypatch.setattr(runtime_locator_module, "_verify_tree", verify_then_rebind)
    with pytest.raises(RuntimeLocatorVerificationError, match="root path changed"):
        _verify(manifest)
    assert rebound is True


def test_unexpected_tree_os_error_is_wrapped_as_verification_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)

    def refuse_tree(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic descriptor refusal")

    monkeypatch.setattr(runtime_locator_module, "_verify_tree", refuse_tree)
    with pytest.raises(
        RuntimeLocatorVerificationError,
        match="operating-system verification failed",
    ):
        _verify(manifest)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "socket"])
def test_tree_verification_rejects_symlink_fifo_and_socket_entries(
    tmp_path: Path,
    kind: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    special = fixture.package_root / "special"
    socket_handle: socket.socket | None = None
    if kind == "symlink":
        special.symlink_to(fixture.executable)
        expected = "symlink"
    elif kind == "fifo":
        os.mkfifo(special)
        expected = "non-regular"
    else:
        socket_handle = socket.socket(socket.AF_UNIX)
        socket_handle.bind(str(special))
        expected = "non-regular"
    try:
        with pytest.raises(RuntimeLocatorVerificationError, match=expected):
            _verify(manifest)
    finally:
        if socket_handle is not None:
            socket_handle.close()


def test_declared_file_and_root_symlinks_fail_without_following(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    declared = fixture.package_root / "module.py"
    target = fixture.package_root / "real-module.py"
    declared.rename(target)
    declared.symlink_to(target.name)
    with pytest.raises(RuntimeLocatorVerificationError, match="symlink"):
        _verify(manifest)

    fixture = _make_fixture(tmp_path / "root-case")
    manifest = _load(fixture)
    real_root = fixture.import_root.with_name("real-import-root")
    fixture.import_root.rename(real_root)
    fixture.import_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(RuntimeLocatorVerificationError, match="symlink"):
        _verify(manifest)


def test_manifest_ancestor_symlink_and_device_executable_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    linked_authority = tmp_path / "linked-authority"
    linked_authority.symlink_to(fixture.manifest_path.parent, target_is_directory=True)
    linked_manifest = linked_authority / fixture.manifest_path.name
    with pytest.raises(RuntimeLocatorVerificationError, match="symlink"):
        load_runtime_locator_manifest(
            linked_manifest,
            expected_source_sha256=fixture.source_sha256,
            expected_source_byte_length=fixture.source_byte_length,
        )

    device_payload = copy.deepcopy(fixture.payload)
    device_payload["executable"] = {
        "path": "/dev/null",
        "sha256": sha256(b"").hexdigest(),
        "byte_length": 0,
    }
    digest = _write_payload(fixture.manifest_path, device_payload)
    device_manifest = load_runtime_locator_manifest(
        fixture.manifest_path,
        expected_source_sha256=digest,
        expected_source_byte_length=fixture.manifest_path.stat().st_size,
    )
    with pytest.raises(RuntimeLocatorVerificationError, match="non-regular"):
        _verify(device_manifest)


@pytest.mark.parametrize("invalid_path", ["bad\x00.py", "nested/../escape.py"])
def test_manifest_rejects_nul_and_parent_traversal_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["runtime_package"]["files"][0]["path"] = invalid_path
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="canonical|beneath"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


def test_manifest_rejects_overlapping_runtime_and_dependency_roots(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.payload)
    payload["dependency_roots"][0] = copy.deepcopy(payload["runtime_package"])
    digest = _write_payload(fixture.manifest_path, payload)
    with pytest.raises(ValueError, match="roots must be disjoint"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )


@pytest.mark.parametrize("entry", ["cached.pyc", "__pycache__"])
def test_declared_or_observed_bytecode_storage_is_forbidden(
    tmp_path: Path,
    entry: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    invalid = copy.deepcopy(fixture.payload)
    if entry.endswith(".pyc"):
        invalid["runtime_package"]["files"][0]["path"] = entry
    else:
        invalid["runtime_package"]["directories"].append(entry)
        invalid["runtime_package"]["directories"].sort()
    digest = _write_payload(fixture.manifest_path, invalid)
    with pytest.raises(ValueError, match="pyc|bytecode"):
        load_runtime_locator_manifest(
            fixture.manifest_path,
            expected_source_sha256=digest,
            expected_source_byte_length=fixture.manifest_path.stat().st_size,
        )

    fixture = _make_fixture(tmp_path / "observed")
    manifest = _load(fixture)
    observed = fixture.package_root / entry
    if entry.endswith(".pyc"):
        observed.write_bytes(b"bytecode")
    else:
        observed.mkdir()
    with pytest.raises(RuntimeLocatorVerificationError, match="forbidden bytecode"):
        _verify(manifest)


def test_source_tree_and_explicit_expectations_are_revalidated(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    fixture.manifest_path.write_bytes(fixture.manifest_path.read_bytes() + b"\n")
    with pytest.raises(
        RuntimeLocatorVerificationError, match="size differs|source SHA"
    ):
        _verify(manifest)

    fixture = _make_fixture(tmp_path / "expectations")
    manifest = _load(fixture)
    with pytest.raises(RuntimeLocatorVerificationError, match="cache tag"):
        verify_runtime_locator_manifest_scaffold(
            manifest,
            expected_cache_tag="cpython-311",
            expected_target_coordinates=_TARGETS,
        )
    with pytest.raises(RuntimeLocatorVerificationError, match="target coordinates"):
        verify_runtime_locator_manifest_scaffold(
            manifest,
            expected_cache_tag=_CACHE_TAG,
            expected_target_coordinates=(_TARGETS[0],),
        )


def test_scaffold_evidence_is_pid_bound_noncopyable_and_nonpickleable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    evidence = _verify(_load(fixture))
    with pytest.raises(TypeError, match="only be minted"):
        VerifiedRuntimeLocatorScaffoldEvidence()
    with pytest.raises(TypeError, match="process-local"):
        copy.copy(evidence)
    with pytest.raises(TypeError, match="process-local"):
        copy.deepcopy(evidence)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(evidence)

    real_process_id = os.getpid()
    monkeypatch.setattr(
        runtime_locator_module.os, "getpid", lambda: real_process_id + 1
    )
    with pytest.raises(RuntimeLocatorVerificationError, match="different process"):
        _ = evidence.runtime_package_sha256
    evidence.close()


def test_evidence_keeps_executable_and_tree_postverify_mutation_as_blocker(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    evidence = _verify(_load(fixture))
    fixture.executable.write_bytes(b"#!/bin/sh\nexit 1\n")
    (fixture.package_root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    record = evidence.as_record()
    assert record["closure_complete"] is False
    assert record["unbound_residuals"] == [RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL]
    assert "executable" in RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL
    assert "during-or-after" in RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL
    assert evidence.launch_blockers == (RUNTIME_LOCATOR_SCAFFOLD_BLOCKER,)
    evidence.close()


def test_sequential_tree_scan_is_explicitly_not_one_atomic_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = _load(fixture)
    original_verify_file = runtime_locator_module._verify_file_at
    first_file = fixture.package_root / "__init__.py"
    mutated = False

    def verify_then_mutate(*args: object, **kwargs: object) -> None:
        nonlocal mutated
        original_verify_file(*args, **kwargs)
        if kwargs["relative_path"] == "tgvf_rl/__init__.py":
            first_file.write_text("VALUE = 2\n", encoding="utf-8")
            mutated = True

    monkeypatch.setattr(runtime_locator_module, "_verify_file_at", verify_then_mutate)
    evidence = _verify(manifest)
    assert mutated is True
    assert evidence.as_record()["closure_complete"] is False
    assert "no-atomic-observation" in RUNTIME_LOCATOR_SAME_UID_TOCTOU_RESIDUAL
    evidence.close()


def test_duplicate_refuses_descriptor_reused_by_concurrent_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    evidence = _verify(_load(fixture))
    descriptors = object.__getattribute__(evidence, "_descriptors")
    import_root_fd = descriptors[0]
    wrong_root = tmp_path / "wrong-root"
    wrong_root.mkdir()
    duplicate_entered = threading.Event()
    allow_duplicate = threading.Event()
    errors: list[BaseException] = []
    real_dup = runtime_locator_module.os.dup

    def delayed_dup(descriptor: int) -> int:
        duplicate_entered.set()
        assert allow_duplicate.wait(timeout=30)
        return real_dup(descriptor)

    def duplicate_worker() -> None:
        try:
            evidence.duplicate_runtime_import_root_directory_fd()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(runtime_locator_module.os, "dup", delayed_dup)
    worker = threading.Thread(target=duplicate_worker)
    worker.start()
    assert duplicate_entered.wait(timeout=30)
    evidence.close()

    wrong_fd = os.open(wrong_root, os.O_RDONLY | os.O_DIRECTORY)
    extra_fd: int | None = None
    if wrong_fd != import_root_fd:
        os.dup2(wrong_fd, import_root_fd)
        extra_fd = wrong_fd
        wrong_fd = import_root_fd
    allow_duplicate.set()
    worker.join(timeout=30)
    try:
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeLocatorVerificationError)
        assert "duplicated runtime root descriptor identity differs" in str(errors[0])
    finally:
        os.close(wrong_fd)
        if extra_fd is not None:
            os.close(extra_fd)


def test_concurrent_close_transfers_descriptor_ownership_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _verify(_load(_make_fixture(tmp_path)))
    descriptors = object.__getattribute__(evidence, "_descriptors")
    close_calls: list[int] = []
    real_close = runtime_locator_module.os.close
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def tracked_close(descriptor: int) -> None:
        if descriptor in descriptors:
            close_calls.append(descriptor)
        real_close(descriptor)

    def close_worker() -> None:
        try:
            barrier.wait(timeout=30)
            evidence.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(runtime_locator_module.os, "close", tracked_close)
    workers = tuple(threading.Thread(target=close_worker) for _ in range(2))
    for worker in workers:
        worker.start()
    barrier.wait(timeout=30)
    for worker in workers:
        worker.join(timeout=30)

    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert evidence.closed is True
    assert sorted(close_calls) == sorted(descriptors)


def test_explicit_close_attempts_all_descriptors_then_raises_first_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _verify(_load(_make_fixture(tmp_path)))
    descriptors = object.__getattribute__(evidence, "_descriptors")
    close_calls: list[int] = []
    real_close = runtime_locator_module.os.close

    def injected_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        if descriptor == descriptors[0]:
            raise OSError("synthetic first close failure")
        real_close(descriptor)

    monkeypatch.setattr(runtime_locator_module.os, "close", injected_close)
    try:
        with pytest.raises(OSError, match="synthetic first close failure"):
            evidence.close()
        assert evidence.closed is True
        assert close_calls == list(descriptors)
    finally:
        real_close(descriptors[0])


def test_runtime_locator_leaf_import_has_isolated_python_firebreak() -> None:
    source_root = _REPOSITORY_ROOT / "src"
    script = """
import importlib
import sys

manifest = importlib.import_module('tgvf_rl.ops.runtime_locator_manifest')
assert manifest.RUNTIME_LOCATOR_MANIFEST_SCHEMA
assert 'tgvf_rl.ops.runtime_locator' not in sys.modules
module = importlib.import_module('tgvf_rl.ops.runtime_locator')
assert module.RUNTIME_LOCATOR_MANIFEST_SCHEMA
for forbidden in ('torch', 'numpy', 'hydra', 'ray', 'verl', 'site',
                  'sysconfig', 'tgvf_rl.framework'):
    assert not any(
        name == forbidden or name.startswith(forbidden + '.')
        for name in sys.modules
    ), forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-P", "-S", "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(source_root)},
    )

    assert completed.returncode == 0, completed.stderr
