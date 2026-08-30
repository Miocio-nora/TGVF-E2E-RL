from __future__ import annotations

import gc
from hashlib import sha256
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tgvf_rl.ops.cli_authorization import (
    PythonExecutableBinding,
    PythonExecutableIdentity,
    verify_python_executable_binding,
)
from tgvf_rl.ops.launch_gate import LaunchAuthorizationError
from tgvf_rl.secure_file_read import retain_regular_file_absolute_nofollow


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _copy_executable(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copy2(Path(sys.executable).resolve(), destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    return destination


def _binding_for_path(path: Path) -> PythonExecutableBinding:
    retained = retain_regular_file_absolute_nofollow(path)
    try:
        snapshot = retained.snapshot()
        observed = snapshot.after
        identity = PythonExecutableIdentity(
            declared_path=path,
            resolved_path=path,
            sha256=sha256(snapshot.payload).hexdigest(),
            byte_length=observed.st_size,
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=observed.st_mode,
        )
        binding = PythonExecutableBinding(identity, retained)
        retained = None  # type: ignore[assignment]
        return binding
    finally:
        if retained is not None:
            retained.close()


def test_bind_rejects_candidate_replaced_after_running_process_identity_check(
    tmp_path: Path,
) -> None:
    copied_python = _copy_executable(tmp_path, "race-python")
    replacement = tmp_path / "replacement"
    shutil.copy2("/bin/false", replacement)
    code = """
from pathlib import Path
import os
import sys

import tgvf_rl.ops.cli_authorization_identity as identity_module
from tgvf_rl.ops.launch_gate import LaunchAuthorizationError

real_retain = identity_module.retain_regular_file_absolute_nofollow
candidate = Path(sys.executable).resolve()
replacement = Path(sys.argv[1])
captured = []

def replace_then_retain(path):
    os.replace(replacement, candidate)
    retained = real_retain(path)
    captured.append(retained)
    return retained

identity_module.retain_regular_file_absolute_nofollow = replace_then_retain
try:
    identity_module.bind_current_python_executable_for_exec(sys.executable)
except LaunchAuthorizationError as error:
    assert "not the running process executable" in str(error), error
    assert len(captured) == 1
    assert captured[0].closed
else:
    raise AssertionError("replacement interpreter was accepted")
"""
    completed = subprocess.run(
        [str(copied_python), "-c", code, str(replacement)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_abandoned_binding_and_prepared_owner_release_descriptor(
    tmp_path: Path,
) -> None:
    executable = _copy_executable(tmp_path, "abandoned-python")
    binding = _binding_for_path(executable)
    descriptor = binding.fileno()
    prepared = SimpleNamespace(python_binding=binding)
    del binding
    del prepared
    gc.collect()

    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_explicit_close_detaches_finalizer_before_descriptor_number_reuse(
    tmp_path: Path,
) -> None:
    executable = _copy_executable(tmp_path, "reused-python")
    retained = retain_regular_file_absolute_nofollow(executable)
    descriptor = retained.fileno()
    retained.close()
    retained.close()

    replacement_descriptor = os.open(executable, os.O_RDONLY)
    try:
        if replacement_descriptor != descriptor:
            os.dup2(replacement_descriptor, descriptor)
            os.close(replacement_descriptor)
            replacement_descriptor = descriptor
        del retained
        gc.collect()
        assert stat.S_ISREG(os.fstat(replacement_descriptor).st_mode)
    finally:
        os.close(replacement_descriptor)


def test_retained_descriptor_is_process_local_and_not_serializable(
    tmp_path: Path,
) -> None:
    retained = retain_regular_file_absolute_nofollow(
        _copy_executable(tmp_path, "unpicklable-python")
    )
    try:
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(retained)
    finally:
        retained.close()


@pytest.mark.parametrize("tamper", ["bytes", "size", "mode"])
def test_same_fd_verification_rejects_inode_content_or_mode_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    executable = _copy_executable(tmp_path, f"tamper-{tamper}-python")
    binding = _binding_for_path(executable)
    try:
        if tamper in {"bytes", "size"}:
            with executable.open("r+b") as stream:
                if tamper == "bytes":
                    first = stream.read(1)
                    assert first
                    stream.seek(0)
                    stream.write(bytes([first[0] ^ 0xFF]))
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\x00")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            executable.chmod(executable.stat().st_mode & ~stat.S_IXUSR)

        with pytest.raises(
            LaunchAuthorizationError,
            match="identity changed",
        ):
            verify_python_executable_binding(binding)
    finally:
        binding.close()


def test_real_fd_exec_uses_retained_inode_after_declared_path_replacement(
    tmp_path: Path,
) -> None:
    declared = _copy_executable(tmp_path, "bound-python")
    retained = retain_regular_file_absolute_nofollow(declared)
    try:
        descriptor = retained.fileno()
        replacement = tmp_path / "replacement-false"
        shutil.copy2("/bin/false", replacement)
        os.replace(replacement, declared)
        code = (
            "import os,sys; "
            "fd=int(sys.argv[1]); declared=sys.argv[2]; "
            "os.execve(fd, [declared, '-c', "
            "\"print('retained-fd-exec-ok')\"], os.environ.copy())"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(descriptor), str(declared)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(descriptor,),
            timeout=30,
        )
    finally:
        retained.close()

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "retained-fd-exec-ok"


def test_binding_identity_is_read_only(tmp_path: Path) -> None:
    binding = _binding_for_path(_copy_executable(tmp_path, "immutable-identity-python"))
    try:
        with pytest.raises(AttributeError):
            binding.identity = binding.identity  # type: ignore[misc]
    finally:
        binding.close()


def test_fd_exec_platform_contract_fails_closed_without_proc_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tgvf_rl.ops import cli_authorization_identity as identity_module

    opened = False
    real_resolve = identity_module.Path.resolve

    def fail_proc_resolve(path: Path, *, strict: bool = False) -> Path:
        if path == Path("/proc/self/exe"):
            raise OSError("proc identity unavailable")
        return real_resolve(path, strict=strict)

    def unexpected_open(_path: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("candidate fd opened without a process identity")

    monkeypatch.setattr(identity_module.Path, "resolve", fail_proc_resolve)
    monkeypatch.setattr(
        identity_module,
        "retain_regular_file_absolute_nofollow",
        unexpected_open,
    )
    with pytest.raises(
        LaunchAuthorizationError,
        match="identity is unavailable",
    ):
        identity_module.bind_current_python_executable_for_exec(sys.executable)
    assert not opened
