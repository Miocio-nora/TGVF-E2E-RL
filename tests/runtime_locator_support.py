"""Small public-verifier fixtures shared by startup-authorization tests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

from tgvf_rl.ops.runtime_locator import (
    RUNTIME_LOCATOR_MANIFEST_SCHEMA,
    VerifiedRuntimeLocatorScaffoldEvidence,
    load_runtime_locator_manifest,
    verify_runtime_locator_manifest_scaffold,
)


POLICY_DRIVER_TARGET = "tgvf_rl.framework.verl.policy_main:main"
ALTERNATE_TARGET = "tgvf_rl.alternate:main"


def _sha256_file(path: Path) -> str:
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
                "sha256": _sha256_file(path),
                "byte_length": path.stat().st_size,
            }
            for path in entries
            if path.is_file()
        ),
        key=lambda item: item["path"],
    )
    return {"root": str(root), "directories": directories, "files": files}


def verified_runtime_locator_evidence(
    tmp_path: Path,
    *,
    executable: Path,
    target_coordinates: tuple[str, ...] = (POLICY_DRIVER_TARGET,),
) -> VerifiedRuntimeLocatorScaffoldEvidence:
    """Create exact usable evidence through the public loader and verifier."""

    runtime_root = tmp_path / "runtime-locator-package"
    package = runtime_root / "tgvf_rl"
    policy_package = package / "framework" / "verl"
    policy_package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "framework" / "__init__.py").write_text("", encoding="utf-8")
    (policy_package / "__init__.py").write_text("", encoding="utf-8")
    (policy_package / "policy_main.py").write_text(
        "def main():\n    return None\n",
        encoding="utf-8",
    )
    (package / "alternate.py").write_text(
        "def main():\n    return None\n",
        encoding="utf-8",
    )

    dependency_root = tmp_path / "runtime-locator-dependency"
    dependency_root.mkdir()
    (dependency_root / "dependency.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    cache_tag = sys.implementation.cache_tag
    assert isinstance(cache_tag, str)
    payload = {
        "schema_version": RUNTIME_LOCATOR_MANIFEST_SCHEMA,
        "cache_tag": cache_tag,
        "executable": {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "byte_length": executable.stat().st_size,
        },
        "target_coordinates": list(target_coordinates),
        "runtime_package": _tree_record(runtime_root),
        "dependency_roots": [_tree_record(dependency_root)],
    }
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    manifest_path = tmp_path / "runtime-locator-authority" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(raw)
    manifest = load_runtime_locator_manifest(
        manifest_path,
        expected_source_sha256=sha256(raw).hexdigest(),
        expected_source_byte_length=len(raw),
    )
    return verify_runtime_locator_manifest_scaffold(
        manifest,
        expected_cache_tag=cache_tag,
        expected_target_coordinates=target_coordinates,
    )


__all__ = [
    "ALTERNATE_TARGET",
    "POLICY_DRIVER_TARGET",
    "verified_runtime_locator_evidence",
]
