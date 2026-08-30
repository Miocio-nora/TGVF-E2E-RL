from __future__ import annotations

from dataclasses import replace
import ast
import copy
import errno
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
import textwrap

import pytest

from tgvf_rl.ops.sealed_memfd_artifact import (
    MAX_SEALED_MEMFD_ARTIFACT_BYTES,
    SEALED_MEMFD_ARTIFACT_SCHEMA,
    SealedMemfdArtifactBinding,
    SealedMemfdArtifactError,
    SealedMemfdArtifactIdentity,
    VerifiedSealedMemfdArtifactReference,
    create_sealed_memfd_artifact,
    open_verified_sealed_memfd_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPOSITORY_ROOT / "src" / "tgvf_rl" / "ops" / "sealed_memfd_artifact.py"
)
PAYLOAD = b"immutable-stage-zero-bytes\n"


@pytest.fixture
def artifact() -> SealedMemfdArtifactBinding:
    binding = create_sealed_memfd_artifact(
        PAYLOAD,
        purpose="stage0-script",
        name="tgvf-test-stage0",
    )
    try:
        yield binding
    finally:
        binding.close()


def test_create_binds_exact_bytes_inode_owner_and_mandatory_seals(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    identity = artifact.identity
    metadata = os.fstat(artifact.fileno())

    assert identity.schema_version == SEALED_MEMFD_ARTIFACT_SCHEMA
    assert identity.purpose == "stage0-script"
    assert identity.owner_pid == os.getpid()
    assert identity.owner_descriptor == artifact.fileno()
    assert identity.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert identity.byte_length == len(PAYLOAD)
    assert (identity.device, identity.inode) == (
        metadata.st_dev,
        metadata.st_ino,
    )
    assert identity.mode == metadata.st_mode
    assert stat.S_ISREG(identity.mode)
    required = (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_SEAL
    )
    assert identity.seals & required == required
    assert fcntl.fcntl(artifact.fileno(), fcntl.F_GET_SEALS) == identity.seals
    assert os.pread(artifact.fileno(), len(PAYLOAD), 0) == PAYLOAD
    assert not os.get_inheritable(artifact.fileno())
    assert artifact.proc_fd_path == Path(
        f"/proc/{os.getpid()}/fd/{artifact.fileno()}"
    )
    artifact.verify()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda descriptor: os.pwrite(descriptor, b"X", 0),
        lambda descriptor: os.ftruncate(descriptor, len(PAYLOAD) - 1),
        lambda descriptor: os.ftruncate(descriptor, len(PAYLOAD) + 1),
    ],
)
def test_kernel_refuses_content_size_mutation_after_seal(
    artifact: SealedMemfdArtifactBinding,
    mutation: object,
) -> None:
    with pytest.raises(OSError) as captured:
        mutation(artifact.fileno())  # type: ignore[operator]
    assert captured.value.errno in {errno.EPERM, errno.EBUSY}
    artifact.verify()


def test_seal_set_cannot_be_extended_after_final_seal(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    with pytest.raises(OSError) as captured:
        fcntl.fcntl(
            artifact.fileno(),
            fcntl.F_ADD_SEALS,
            artifact.identity.seals,
        )
    assert captured.value.errno == errno.EPERM
    artifact.verify()


def test_reopen_retains_verified_local_descriptor_after_owner_closes(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    reference = open_verified_sealed_memfd_artifact(artifact.identity)
    try:
        owner_proc_fd_path = artifact.proc_fd_path
        assert reference.identity == artifact.identity
        assert reference.fileno() != artifact.fileno()
        assert reference.local_proc_fd_path == Path(
            f"/proc/self/fd/{reference.fileno()}"
        )
        assert not os.get_inheritable(reference.fileno())
        assert os.pread(reference.fileno(), len(PAYLOAD), 0) == PAYLOAD
        artifact.close()
        assert not owner_proc_fd_path.exists()
        reference.verify()
        assert os.pread(reference.fileno(), len(PAYLOAD), 0) == PAYLOAD
    finally:
        reference.close()


def test_reopen_refuses_closed_owner_descriptor(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    identity = artifact.identity
    artifact.close()

    with pytest.raises(
        SealedMemfdArtifactError,
        match="unavailable through procfs",
    ):
        open_verified_sealed_memfd_artifact(identity)


def test_child_reopens_parent_fd_without_descriptor_inheritance(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    script = textwrap.dedent(
        """
        import fcntl
        import hashlib
        import json
        import os
        import sys

        identity = json.loads(sys.argv[1])
        descriptor = os.open(
            f"/proc/{identity['owner_pid']}/fd/{identity['owner_descriptor']}",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        try:
            payload = os.pread(descriptor, identity['byte_length'] + 1, 0)
            metadata = os.fstat(descriptor)
            print(json.dumps({
                'sha256': hashlib.sha256(payload).hexdigest(),
                'byte_length': len(payload),
                'device': metadata.st_dev,
                'inode': metadata.st_ino,
                'seals': fcntl.fcntl(descriptor, fcntl.F_GET_SEALS),
                'inheritable': os.get_inheritable(descriptor),
            }, sort_keys=True))
        finally:
            os.close(descriptor)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script, artifact.identity.to_json()],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "byte_length": artifact.identity.byte_length,
        "device": artifact.identity.device,
        "inheritable": False,
        "inode": artifact.identity.inode,
        "seals": artifact.identity.seals,
        "sha256": artifact.identity.sha256,
    }


def test_fresh_child_uses_production_identity_and_reopen_api(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    script = textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        import sys
        sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r})
        from tgvf_rl.ops.sealed_memfd_artifact import (
            SealedMemfdArtifactIdentity,
            open_verified_sealed_memfd_artifact,
        )

        identity = SealedMemfdArtifactIdentity.from_json(sys.argv[1])
        reference = open_verified_sealed_memfd_artifact(identity)
        try:
            payload = os.pread(reference.fileno(), identity.byte_length + 1, 0)
            print(json.dumps({{
                'identity_sha256': reference.identity.identity_sha256,
                'local_path': str(reference.local_proc_fd_path),
                'payload_sha256': hashlib.sha256(payload).hexdigest(),
                'torch_loaded': 'torch' in sys.modules,
            }}, sort_keys=True))
        finally:
            reference.close()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script, artifact.identity.to_json()],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["identity_sha256"] == artifact.identity.identity_sha256
    assert observed["local_path"].startswith("/proc/self/fd/")
    assert observed["payload_sha256"] == artifact.identity.sha256
    assert observed["torch_loaded"] is False


def test_procfd_script_mechanically_starts_fresh_isolated_python() -> None:
    # This is a compatibility probe, not an authenticity claim.  Canonical
    # routing still requires a pre-runtime trampoline to verify the procfd
    # identity before any sealed artifact byte executes.
    script = (
        b"import json,sys\n"
        b"print(json.dumps({"
        b"'isolated':sys.flags.isolated,"
        b"'ignore_environment':sys.flags.ignore_environment,"
        b"'no_site':sys.flags.no_site,"
        b"'safe_path':sys.flags.safe_path,"
        b"'torch_loaded':'torch' in sys.modules"
        b"},sort_keys=True))\n"
    )
    binding = create_sealed_memfd_artifact(
        script,
        purpose="stage0-script",
        name="tgvf-test-python-entry",
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-I", "-S", str(binding.proc_fd_path)],
            check=False,
            close_fds=True,
            capture_output=True,
            text=True,
        )
    finally:
        binding.close()

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "safe_path": True,
        "torch_loaded": False,
    }


def test_identity_json_is_exact_canonical_and_digest_bound(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    identity = artifact.identity
    encoded = identity.to_json()

    assert SealedMemfdArtifactIdentity.from_json(encoded) == identity
    assert json.dumps(
        identity.as_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == encoded

    tampered = identity.as_record()
    tampered["byte_length"] = identity.byte_length + 1
    with pytest.raises(ValueError, match="identity digest differs"):
        SealedMemfdArtifactIdentity.from_record(tampered)


@pytest.mark.parametrize(
    ("payload", "purpose", "name", "expected"),
    [
        (bytearray(PAYLOAD), "stage0-script", "valid", "exact bytes"),
        (b"", "stage0-script", "valid", "non-empty"),
        (PAYLOAD, "Stage0", "valid", "purpose"),
        (PAYLOAD, "stage0-script", "", "non-empty str"),
        (PAYLOAD, "stage0-script", "bad/name", "forbidden character"),
        (PAYLOAD, "stage0-script", "bad\x00name", "forbidden character"),
        (PAYLOAD, "stage0-script", "x" * 201, "too long"),
    ],
)
def test_factory_rejects_noncanonical_inputs(
    payload: object,
    purpose: str,
    name: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        create_sealed_memfd_artifact(  # type: ignore[arg-type]
            payload,
            purpose=purpose,
            name=name,
        )


def test_identity_rejects_boolean_numeric_fields(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    record = artifact.identity.as_record()
    record["owner_pid"] = True

    with pytest.raises(ValueError, match="owner_pid"):
        SealedMemfdArtifactIdentity.from_record(record)


def test_identity_rejects_artifact_above_fixed_size_ceiling(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    with pytest.raises(ValueError, match="fixed ceiling"):
        replace(
            artifact.identity,
            byte_length=MAX_SEALED_MEMFD_ARTIFACT_BYTES + 1,
        )


def test_identity_refuses_duplicate_noncanonical_and_extra_json(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    encoded = artifact.identity.to_json()
    duplicate = encoded[:-1] + ',"purpose":"stage0-script"}'
    with pytest.raises(ValueError, match="repeats key"):
        SealedMemfdArtifactIdentity.from_json(duplicate)

    noncanonical = json.dumps(artifact.identity.as_record(), indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        SealedMemfdArtifactIdentity.from_json(noncanonical)

    extra = artifact.identity.as_record()
    extra["unexpected"] = "field"
    with pytest.raises(ValueError, match="field set differs"):
        SealedMemfdArtifactIdentity.from_record(extra)


def test_reopen_refuses_owner_start_descriptor_and_seal_tampering(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    identity = artifact.identity
    with pytest.raises(SealedMemfdArtifactError, match="owner process was replaced"):
        open_verified_sealed_memfd_artifact(
            replace(
                identity,
                owner_process_start_ticks=identity.owner_process_start_ticks + 1,
            )
        )

    with pytest.raises(SealedMemfdArtifactError):
        open_verified_sealed_memfd_artifact(
            replace(identity, owner_descriptor=identity.owner_descriptor + 100_000)
        )

    with pytest.raises(SealedMemfdArtifactError, match="seal set differs"):
        open_verified_sealed_memfd_artifact(
            replace(identity, seals=identity.seals | (1 << 30))
        )


def test_reopen_refuses_recomputed_identity_with_false_payload_sha(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    forged = replace(artifact.identity, sha256="0" * 64)
    assert forged.identity_sha256 != artifact.identity.identity_sha256

    with pytest.raises(SealedMemfdArtifactError, match="SHA256 differs"):
        open_verified_sealed_memfd_artifact(forged)


def test_binding_detects_post_seal_metadata_change(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    os.fchmod(artifact.fileno(), 0o600)

    with pytest.raises(SealedMemfdArtifactError, match="metadata differs"):
        artifact.verify()


def test_capability_types_are_constructor_sealed_uncopyable_and_unserializable(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    with pytest.raises(TypeError, match="only be minted"):
        SealedMemfdArtifactBinding()
    with pytest.raises(TypeError, match="only be minted"):
        VerifiedSealedMemfdArtifactReference()
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(artifact)
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(artifact)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(artifact)

    reference = open_verified_sealed_memfd_artifact(artifact.identity)
    try:
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(reference)
        with pytest.raises(TypeError, match="not copyable"):
            copy.copy(reference)
        with pytest.raises(TypeError, match="not copyable"):
            copy.deepcopy(reference)
    finally:
        reference.close()


def test_close_is_idempotent_and_all_access_fails_after_close(
    artifact: SealedMemfdArtifactBinding,
) -> None:
    reference = open_verified_sealed_memfd_artifact(artifact.identity)
    owner_fd = artifact.fileno()
    local_fd = reference.fileno()

    artifact.close()
    artifact.close()
    reference.close()
    reference.close()

    assert artifact.closed
    assert reference.closed
    with pytest.raises(OSError):
        os.fstat(owner_fd)
    with pytest.raises(OSError):
        os.fstat(local_fd)
    with pytest.raises(SealedMemfdArtifactError, match="closed"):
        _ = artifact.identity
    with pytest.raises(SealedMemfdArtifactError, match="closed"):
        _ = reference.identity


@pytest.mark.parametrize("capability_kind", ["binding", "reference"])
def test_close_does_not_close_reused_unrelated_raw_fd(
    artifact: SealedMemfdArtifactBinding,
    capability_kind: str,
) -> None:
    capability: SealedMemfdArtifactBinding | VerifiedSealedMemfdArtifactReference
    if capability_kind == "binding":
        capability = artifact
    else:
        capability = open_verified_sealed_memfd_artifact(artifact.identity)
    raw_descriptor = capability.fileno()
    os.close(raw_descriptor)
    replacement = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    assert replacement == raw_descriptor
    try:
        capability.close()
        assert stat.S_ISCHR(os.fstat(replacement).st_mode)
    finally:
        os.close(replacement)


def test_gc_finalizer_does_not_close_reused_unrelated_raw_fd() -> None:
    binding = create_sealed_memfd_artifact(
        PAYLOAD,
        purpose="runtime-zip",
        name="tgvf-test-finalizer-reuse",
    )
    raw_descriptor = binding.fileno()
    os.close(raw_descriptor)
    replacement = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    assert replacement == raw_descriptor

    del binding
    gc.collect()
    try:
        assert stat.S_ISCHR(os.fstat(replacement).st_mode)
    finally:
        os.close(replacement)


def test_normal_close_releases_primary_and_private_guard_descriptors() -> None:
    before = set(os.listdir("/proc/self/fd"))
    binding = create_sealed_memfd_artifact(
        PAYLOAD,
        purpose="runtime-zip",
        name="tgvf-test-fd-lifetime",
    )
    after_binding = set(os.listdir("/proc/self/fd"))
    reference = open_verified_sealed_memfd_artifact(binding.identity)
    after_reference = set(os.listdir("/proc/self/fd"))

    assert len(after_binding) == len(before) + 2
    assert len(after_reference) == len(after_binding) + 2
    reference.close()
    binding.close()
    assert set(os.listdir("/proc/self/fd")) == before


def test_forked_copy_closes_only_child_descriptors_and_parent_stays_valid() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import os
        import sys
        sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r})
        from tgvf_rl.ops.sealed_memfd_artifact import (
            create_sealed_memfd_artifact,
        )

        artifact = create_sealed_memfd_artifact(
            b'fork-lifetime-probe',
            purpose='stage0-script',
            name='tgvf-test-fork-lifetime',
        )
        descriptor = artifact.fileno()
        child = os.fork()
        if child == 0:
            artifact.close()
            try:
                os.fstat(descriptor)
            except OSError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(child, 0)
        artifact.verify()
        print(json.dumps({{
            'child_exit': os.waitstatus_to_exitcode(status),
            'parent_fd_open': os.fstat(descriptor).st_ino == artifact.identity.inode,
        }}, sort_keys=True))
        artifact.close()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "child_exit": 0,
        "parent_fd_open": True,
    }


def test_open_requires_exact_identity_type() -> None:
    with pytest.raises(TypeError, match="identity type differs"):
        open_verified_sealed_memfd_artifact({})  # type: ignore[arg-type]


def test_leaf_has_no_project_torch_dispatch_or_process_creation_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(
        name == "tgvf_rl" or name.startswith("tgvf_rl.") for name in imported
    )
    assert not imported.intersection(
        {"importlib", "runpy", "subprocess", "multiprocessing"}
    )
    assert not called_names.intersection(
        {"exec", "execv", "execve", "execl", "execlp", "Popen", "run"}
    )


def test_leaf_import_does_not_load_torch_or_training_runtime() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys
        sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r})
        import tgvf_rl.ops.sealed_memfd_artifact as module
        print(json.dumps({{
            'module': module.__name__,
            'torch_loaded': 'torch' in sys.modules,
            'runner_loaded': (
                'tgvf_rl.representation.training.runner' in sys.modules
            ),
        }}, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "module": "tgvf_rl.ops.sealed_memfd_artifact",
        "runner_loaded": False,
        "torch_loaded": False,
    }
