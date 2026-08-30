from __future__ import annotations

import errno
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import tgvf_rl.secure_file_read as secure_file_read
from tgvf_rl.representation.training import config_binding, config_values
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.config_binding import _verify_external_files
from tgvf_rl.representation.training.config_values import (
    _existing_file_path,
    _optional_existing_file_probe,
    _read_configuration_source,
    _read_existing_file_bytes,
)


_REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "chat_template.json",
    "model.safetensors.index.json",
)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _write_model_fixture(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    for name in _REQUIRED_MODEL_FILES:
        (model / name).write_bytes(b"{}")
    return model


def _binding_config(
    root: Path,
    *,
    train_path: Path | None = None,
    train_payload: bytes = b'{"split":"train"}\n',
    train_sha256: str | None = None,
    evaluation: object | None = None,
) -> SimpleNamespace:
    model = _write_model_fixture(root)
    if train_path is None:
        train_path = root / "train.jsonl"
        train_path.write_bytes(train_payload)
    validation_payload = b'{"split":"validation"}\n'
    validation_path = root / "validation.jsonl"
    validation_path.write_bytes(validation_payload)
    return SimpleNamespace(
        model=SimpleNamespace(local_path=model),
        data=SimpleNamespace(
            train=SimpleNamespace(
                jsonl_path=train_path,
                source_sha256=(
                    _digest(train_payload) if train_sha256 is None else train_sha256
                ),
            ),
            validation=SimpleNamespace(
                jsonl_path=validation_path,
                source_sha256=_digest(validation_payload),
            ),
        ),
        output=SimpleNamespace(
            final_artifact_path=root / "outputs" / "adapter.pt",
            metrics_jsonl_path=root / "outputs" / "metrics.jsonl",
        ),
        checkpoint=SimpleNamespace(directory=root / "checkpoints"),
        resume=SimpleNamespace(enabled=False, checkpoint_path=None),
        post_training_internal_evaluation=evaluation,
    )


def _evaluation_config(root: Path, *, report_path: Path) -> SimpleNamespace:
    ordered_payload = b'{"kind":"ordered"}'
    ordered = root / "ordered.json"
    ordered.write_bytes(ordered_payload)
    counterfactual_payload = b'{"kind":"counterfactual"}'
    counterfactual = root / "counterfactual.json"
    counterfactual.write_bytes(counterfactual_payload)
    return SimpleNamespace(
        enabled=True,
        ordered_group_manifest_path=ordered,
        ordered_group_manifest_sha256=_digest(ordered_payload),
        counterfactual_manifest_path=counterfactual,
        counterfactual_manifest_sha256=_digest(counterfactual_payload),
        grounding_manifest_path=None,
        grounding_manifest_sha256=None,
        report_path=report_path,
    )


def _assert_descriptors_closed(descriptors: list[int]) -> None:
    assert descriptors
    for descriptor in set(descriptors):
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_public_loader_intentionally_rejects_source_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "config.toml"
    source.write_text('schema_version = "invalid"\n', encoding="utf-8")
    if symlink_kind == "leaf":
        requested = tmp_path / "config.toml"
        requested.symlink_to(source)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        requested = alias / source.name

    # Intentional hardening from the legacy resolve-and-follow behavior.
    with pytest.raises(
        ValueError,
        match="configuration path does not resolve to a file",
    ):
        load_representation_training_config(
            requested,
            verify_external_files=False,
        )


def test_public_loader_rejects_source_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "config.fifo"
    ready = tmp_path / "fifo-probe-ready"
    os.mkfifo(fifo)
    program = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        from tgvf_rl.representation.training.config import (
            load_representation_training_config,
        )

        Path(sys.argv[2]).write_text("ready", encoding="utf-8")
        try:
            load_representation_training_config(
                sys.argv[1],
                verify_external_files=False,
            )
        except ValueError:
            raise SystemExit(0)
        raise SystemExit(1)
        """
    )

    process = subprocess.Popen(
        [sys.executable, "-c", program, str(fifo), str(ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import_deadline = time.monotonic() + 30.0
        while not ready.is_file() and process.poll() is None:
            if time.monotonic() >= import_deadline:
                process.kill()
                _stdout, stderr = process.communicate(timeout=5.0)
                pytest.fail(f"FIFO probe import did not become ready: {stderr}")
            time.sleep(0.01)
        if not ready.is_file():
            _stdout, stderr = process.communicate(timeout=5.0)
            pytest.fail(f"FIFO probe exited before loader invocation: {stderr}")
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            _stdout, stderr = process.communicate(timeout=5.0)
            pytest.fail(f"configuration loader blocked on FIFO: {stderr}")
        _stdout, stderr = process.communicate(timeout=5.0)
    finally:
        if process.poll() is None:  # pragma: no cover - defensive cleanup
            process.kill()
            process.wait(timeout=5.0)

    assert returncode == 0, stderr


def test_source_missing_and_nonregular_errors_preserve_valueerror_text(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(
        ValueError,
        match="configuration path does not resolve to a file",
    ):
        load_representation_training_config(
            missing,
            verify_external_files=False,
        )

    directory = tmp_path / "directory.toml"
    directory.mkdir()
    with pytest.raises(
        ValueError,
        match=rf"configuration path is not a file: {directory}",
    ):
        load_representation_training_config(
            directory,
            verify_external_files=False,
        )


def test_source_read_stays_anchored_after_bound_ancestor_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source-root"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    source = nested / "config.toml"
    source.write_bytes(b"legitimate snapshot")
    outside = tmp_path / "outside"
    outside_nested = outside / "nested"
    outside_nested.mkdir(parents=True)
    (outside_nested / source.name).write_bytes(b"attacker snapshot")
    archived = tmp_path / "source-root-before-race"
    original_open = secure_file_read._open_path
    swapped = False

    def _swap_after_root_is_bound(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == nested.name and dir_fd is not None and not swapped:
            swapped = True
            source_root.rename(archived)
            source_root.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(secure_file_read, "_open_path", _swap_after_root_is_bound)

    observed_path, payload = _read_configuration_source(source)

    assert swapped
    assert observed_path == source
    assert payload == b"legitimate snapshot"


def test_source_read_rejects_root_rebind_before_component_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    source = source_root / "config.toml"
    source.write_bytes(b"legitimate snapshot")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / source.name).write_bytes(b"attacker snapshot")
    archived = tmp_path / "source-root-before-race"
    original_open = secure_file_read._open_path
    swapped = False

    def _swap_before_root_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == source_root.name and dir_fd is not None and not swapped:
            swapped = True
            source_root.rename(archived)
            source_root.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(secure_file_read, "_open_path", _swap_before_root_open)

    with pytest.raises(
        ValueError,
        match="configuration path does not resolve to a file",
    ):
        _read_configuration_source(source)
    assert swapped


def test_public_loader_hashes_and_parses_one_opened_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    canonical = (
        repository_root
        / "configs"
        / "representation"
        / "qwen3_v4_contextual_hidden_state_v4.toml"
    ).read_bytes()
    source = tmp_path / "config.toml"
    source.write_bytes(canonical)
    attacker = tmp_path / "attacker.toml"
    attacker.write_bytes(b'not = "the parsed snapshot"\n')
    archived = tmp_path / "config-before-race.toml"
    original_open = secure_file_read._open_path
    swapped = False

    def _replace_source_after_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, dir_fd=dir_fd)
        if path == source.name and dir_fd is not None and not swapped:
            swapped = True
            source.rename(archived)
            attacker.rename(source)
        return descriptor

    monkeypatch.setattr(secure_file_read, "_open_path", _replace_source_after_open)

    loaded = load_representation_training_config(
        source,
        verify_external_files=False,
    )

    assert swapped
    assert loaded.run_id == "REP-QWEN3-V4-CONTEXTUAL-V4"
    assert loaded.source_toml_sha256 == _digest(canonical)
    assert source.read_bytes() == b'not = "the parsed snapshot"\n'


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_external_binding_intentionally_rejects_data_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    payload = b'{"split":"train"}\n'
    source = actual / "train.jsonl"
    source.write_bytes(payload)
    if symlink_kind == "leaf":
        requested = tmp_path / "train.jsonl"
        requested.symlink_to(source)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        requested = alias / source.name
    config = _binding_config(
        tmp_path,
        train_path=requested,
        train_payload=payload,
    )

    with pytest.raises(
        ValueError,
        match=r"data\.train\.jsonl_path does not resolve to a file",
    ):
        _verify_external_files(config)


def test_external_binding_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "train.fifo"
    ready = tmp_path / "external-binding-fifo-probe-ready"
    os.mkfifo(fifo)
    config = _binding_config(
        tmp_path,
        train_path=fifo,
        train_sha256="0" * 64,
    )
    program = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from tgvf_rl.representation.training.config_binding import (
            _verify_external_files,
        )

        root = Path(sys.argv[1])
        Path(sys.argv[4]).write_text("ready", encoding="utf-8")
        config = SimpleNamespace(
            model=SimpleNamespace(local_path=root / "model"),
            data=SimpleNamespace(
                train=SimpleNamespace(
                    jsonl_path=Path(sys.argv[2]),
                    source_sha256="0" * 64,
                ),
                validation=SimpleNamespace(
                    jsonl_path=root / "validation.jsonl",
                    source_sha256=sys.argv[3],
                ),
            ),
            output=SimpleNamespace(
                final_artifact_path=root / "outputs" / "adapter.pt",
                metrics_jsonl_path=root / "outputs" / "metrics.jsonl",
            ),
            checkpoint=SimpleNamespace(directory=root / "checkpoints"),
            resume=SimpleNamespace(enabled=False, checkpoint_path=None),
            post_training_internal_evaluation=None,
        )
        try:
            _verify_external_files(config)
        except ValueError:
            raise SystemExit(0)
        raise SystemExit(1)
        """
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path),
            str(fifo),
            config.data.validation.source_sha256,
            str(ready),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import_deadline = time.monotonic() + 30.0
        while not ready.is_file() and process.poll() is None:
            if time.monotonic() >= import_deadline:
                process.kill()
                _stdout, stderr = process.communicate(timeout=5.0)
                pytest.fail(f"external binding probe import did not become ready: {stderr}")
            time.sleep(0.01)
        if not ready.is_file():
            _stdout, stderr = process.communicate(timeout=5.0)
            pytest.fail(f"external binding probe exited before verification: {stderr}")
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            _stdout, stderr = process.communicate(timeout=5.0)
            pytest.fail(f"external binding verification blocked on FIFO: {stderr}")
        _stdout, stderr = process.communicate(timeout=5.0)
    finally:
        if process.poll() is None:  # pragma: no cover - defensive cleanup
            process.kill()
            process.wait(timeout=5.0)

    assert returncode == 0, stderr


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_external_binding_rejects_model_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    config = _binding_config(tmp_path)
    model = config.model.local_path
    if symlink_kind == "leaf":
        target = tmp_path / "outside-config.json"
        target.write_bytes(b"{}")
        candidate = model / "config.json"
        candidate.unlink()
        candidate.symlink_to(target)
        expected_missing = "config.json"
    else:
        alias = tmp_path / "model-alias"
        alias.symlink_to(model, target_is_directory=True)
        config.model.local_path = alias
        expected_missing = "config.json"

    with pytest.raises(
        ValueError,
        match=rf"accepted Qwen3 directory is incomplete: .*{expected_missing}",
    ):
        _verify_external_files(config)


def test_external_hash_uses_payload_from_opened_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legitimate = b'{"split":"legitimate"}\n'
    train = tmp_path / "train.jsonl"
    train.write_bytes(legitimate)
    attacker = tmp_path / "attacker.jsonl"
    attacker.write_bytes(b'{"split":"attacker"}\n')
    archived = tmp_path / "train-before-race.jsonl"
    config = _binding_config(
        tmp_path,
        train_path=train,
        train_sha256=_digest(legitimate),
    )
    original_open = secure_file_read._open_path
    swapped = False

    def _replace_leaf_after_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, dir_fd=dir_fd)
        if path == train.name and dir_fd is not None and not swapped:
            swapped = True
            train.rename(archived)
            attacker.rename(train)
        return descriptor

    monkeypatch.setattr(secure_file_read, "_open_path", _replace_leaf_after_open)

    _verify_external_files(config)

    assert swapped
    assert train.read_bytes() == b'{"split":"attacker"}\n'


def test_external_hash_stays_anchored_after_bound_ancestor_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input-root"
    nested = input_root / "nested"
    nested.mkdir(parents=True)
    legitimate = b'{"split":"legitimate"}\n'
    train = nested / "train.jsonl"
    train.write_bytes(legitimate)
    outside = tmp_path / "outside"
    outside_nested = outside / "nested"
    outside_nested.mkdir(parents=True)
    (outside_nested / train.name).write_bytes(b'{"split":"attacker"}\n')
    archived = tmp_path / "input-root-before-race"
    config = _binding_config(
        tmp_path,
        train_path=train,
        train_sha256=_digest(legitimate),
    )
    original_open = secure_file_read._open_path
    swapped = False

    def _swap_after_input_root_is_bound(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == nested.name and dir_fd is not None and not swapped:
            swapped = True
            input_root.rename(archived)
            input_root.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(
        secure_file_read,
        "_open_path",
        _swap_after_input_root_is_bound,
    )

    _verify_external_files(config)

    assert swapped


def test_report_existence_uses_secure_regular_file_contract(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_bytes(b"existing report")
    evaluation = _evaluation_config(tmp_path, report_path=report)
    config = _binding_config(tmp_path, evaluation=evaluation)

    with pytest.raises(
        ValueError,
        match="post_training_internal_evaluation.report_path already exists",
    ):
        _verify_external_files(config)

    _verify_external_files(
        config,
        allow_existing_post_training_report=True,
    )


def test_model_and_report_presence_probes_never_read_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.json"
    evaluation = _evaluation_config(tmp_path, report_path=report)
    config = _binding_config(tmp_path, evaluation=evaluation)
    expected_size = 16 * 1024**3
    with (config.model.local_path / "config.json").open("wb") as stream:
        stream.truncate(expected_size)
    with report.open("wb") as stream:
        stream.truncate(expected_size)
    payloads = {
        config.data.train.jsonl_path: b'{"split":"train"}\n',
        config.data.validation.jsonl_path: b'{"split":"validation"}\n',
        evaluation.ordered_group_manifest_path: b'{"kind":"ordered"}',
        evaluation.counterfactual_manifest_path: b'{"kind":"counterfactual"}',
    }
    payload_reads: list[Path] = []

    def _known_payload_read(
        value: str | Path,
        *,
        field_name: str,
    ) -> tuple[Path, bytes]:
        del field_name
        path = Path(value)
        payload_reads.append(path)
        if path not in payloads:
            raise AssertionError(f"presence-only path was read: {path}")
        return path, payloads[path]

    def _unexpected_os_read(_descriptor: int, _size: int) -> bytes:
        raise AssertionError("presence probe must not call os.read")

    monkeypatch.setattr(
        config_binding,
        "_read_existing_file_bytes",
        _known_payload_read,
    )
    monkeypatch.setattr(secure_file_read.os, "read", _unexpected_os_read)

    _verify_external_files(
        config,
        allow_existing_post_training_report=True,
    )

    assert payload_reads == [
        config.data.train.jsonl_path,
        config.data.validation.jsonl_path,
        evaluation.ordered_group_manifest_path,
        evaluation.counterfactual_manifest_path,
    ]


def test_existing_file_path_uses_metadata_probe_without_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-sparse.bin"
    with source.open("wb") as stream:
        stream.truncate(16 * 1024**3)

    def _unexpected_read(_descriptor: int, _size: int) -> bytes:
        raise AssertionError("path validation must not read payload bytes")

    monkeypatch.setattr(secure_file_read.os, "read", _unexpected_read)

    assert _existing_file_path(source, field_name="external.path") == source


def test_optional_probe_treats_only_missing_file_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.json"
    assert (
        _optional_existing_file_probe(missing, field_name="evaluation.report") is None
    )

    def _permission_denied(_path: Path) -> object:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(
        config_values,
        "probe_regular_file_absolute_nofollow",
        _permission_denied,
    )
    with pytest.raises(
        ValueError,
        match="evaluation.report does not resolve to a file",
    ):
        _optional_existing_file_probe(missing, field_name="evaluation.report")


def test_optional_probe_rejects_enotdir_and_nonregular_paths(tmp_path: Path) -> None:
    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="does not resolve to a file"):
        _optional_existing_file_probe(
            parent_file / "report.json",
            field_name="evaluation.report",
        )

    directory = tmp_path / "report-directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="does not resolve to a file"):
        _optional_existing_file_probe(
            directory,
            field_name="evaluation.report",
        )


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_report_existence_intentionally_rejects_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    actual = tmp_path / "actual-report"
    actual.mkdir()
    report = actual / "report.json"
    report.write_bytes(b"existing report")
    if symlink_kind == "leaf":
        requested = tmp_path / "report.json"
        requested.symlink_to(report)
    else:
        alias = tmp_path / "report-alias"
        alias.symlink_to(actual, target_is_directory=True)
        requested = alias / report.name
    evaluation = _evaluation_config(tmp_path, report_path=requested)
    config = _binding_config(tmp_path, evaluation=evaluation)

    with pytest.raises(
        ValueError,
        match=(
            "post_training_internal_evaluation.report_path does not resolve to a file"
        ),
    ):
        _verify_external_files(
            config,
            allow_existing_post_training_report=True,
        )


def test_config_secure_reads_close_success_and_failure_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "config.toml"
    source.write_bytes(b"snapshot")
    target = tmp_path / "target.json"
    target.write_bytes(b"payload")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    opened: list[int] = []
    original_open = secure_file_read._open_path

    def _record_open(
        path: str,
        flags: int,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(secure_file_read, "_open_path", _record_open)

    assert _read_configuration_source(source)[1] == b"snapshot"
    _verify_external_files(_binding_config(tmp_path))
    with pytest.raises(ValueError):
        _read_existing_file_bytes(alias, field_name="external.alias")

    _assert_descriptors_closed(opened)
