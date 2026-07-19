from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl import cli
from tgvf_rl.representation.training import runner as runner_module


@pytest.fixture(autouse=True)
def _forbid_real_accelerator_or_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CPU runner contract test attempted accelerator startup")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)


def _launch_config() -> SimpleNamespace:
    return SimpleNamespace(fsdp2=SimpleNamespace(world_size=2))


def _set_valid_launch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "WORLD_SIZE": "2",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_launch_environment_is_exact_and_rank_values_are_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config()
    _set_valid_launch_environment(monkeypatch)

    runner_module._require_launch_environment(config)
    assert runner_module._environment_integer("RANK") == 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2")
    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module._require_launch_environment(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    monkeypatch.setenv("RANK", "١")
    with pytest.raises(ValueError, match="ASCII|torchrun integer|non-negative"):
        runner_module._require_launch_environment(config)
    monkeypatch.setenv("RANK", "-1")
    with pytest.raises(ValueError, match="non-negative torchrun integer"):
        runner_module._require_launch_environment(config)


def test_public_runner_checks_launch_contract_before_distributed_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _launch_config()
    _set_valid_launch_environment(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    verified: list[object] = []
    monkeypatch.setattr(
        runner_module,
        "load_representation_training_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_live_code_identity",
        lambda value: verified.append(value),
    )

    with pytest.raises(ValueError, match="launch environment mismatch"):
        runner_module.run_representation_training("/unused/config.toml")

    assert verified == []


def test_public_runner_lifecycle_can_be_wired_without_starting_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    calls: list[object] = []
    state = {"initialized": False}
    monkeypatch.setattr(
        runner_module,
        "load_representation_training_config",
        lambda path: calls.append(("load", Path(path))) or config,
    )
    monkeypatch.setattr(
        runner_module,
        "_require_launch_environment",
        lambda value: calls.append(("environment", value)),
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_live_code_identity",
        lambda value: calls.append(("code", value)),
    )

    def init_process_group(*, backend: str, timeout: object) -> None:
        calls.append(("init", backend, timeout))
        state["initialized"] = True

    def destroy_process_group() -> None:
        calls.append("destroy")
        state["initialized"] = False

    monkeypatch.setattr(
        torch.distributed, "init_process_group", init_process_group
    )
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", destroy_process_group
    )
    monkeypatch.setattr(
        runner_module,
        "_run_initialized",
        lambda value, *, rank: calls.append(("run", value, rank))
        or {"status": "synthetic", "rank": rank},
    )

    result = runner_module.run_representation_training("/unused/config.toml")

    assert result == {"status": "synthetic", "rank": 0}
    assert [call[0] for call in calls if isinstance(call, tuple)] == [
        "load",
        "environment",
        "code",
        "init",
        "run",
    ]
    assert calls[-1] == "destroy"
    assert state["initialized"] is False


def test_checkpoint_path_and_step_have_one_strict_ascii_round_trip(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            directory=tmp_path,
            filename_prefix="representation",
        )
    )

    path = runner_module._checkpoint_path(config, 7)
    assert path == tmp_path / "representation-step-00000007"
    assert runner_module._checkpoint_step(path, "representation") == 7
    maximum = runner_module._checkpoint_path(config, 99_999_999)
    assert runner_module._checkpoint_step(maximum, "representation") == 99_999_999

    for invalid_step in (True, 0, -1, 100_000_000, 1.5):
        with pytest.raises(ValueError, match=r"\[1, 99999999\]"):
            runner_module._checkpoint_path(config, invalid_step)  # type: ignore[arg-type]
    invalid_names = (
        "other-step-00000001",
        "representation-step-00000000",
        "representation-step-0000001",
        "representation-step-000000001",
        "representation-step-٠٠٠٠٠٠٠١",
        "representation-step-0000000x",
    )
    for name in invalid_names:
        with pytest.raises(ValueError):
            runner_module._checkpoint_step(tmp_path / name, "representation")


def test_metric_jsonl_append_is_strict_incremental_and_rejects_nan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    runner_module._append_metric(
        path,
        {
            "z": (1, 2),
            "path": Path("/tmp/artifact"),
            "text": "目标证据",
            "loss": 1.25,
        },
    )
    runner_module._append_metric(path, {"event": "second", "step": 2})

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert json.loads(raw_lines[0]) == {
        "loss": 1.25,
        "path": "/tmp/artifact",
        "text": "目标证据",
        "z": [1, 2],
    }
    assert json.loads(raw_lines[1]) == {"event": "second", "step": 2}
    before = path.read_bytes()

    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="JSON compliant"):
            runner_module._append_metric(path, {"loss": invalid})
        assert path.read_bytes() == before
    with pytest.raises(TypeError, match="not JSON serializable"):
        runner_module._append_metric(path, {"invalid": object()})
    assert path.read_bytes() == before


def test_collective_metric_append_propagates_rank_zero_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, *, src: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_append_metric",
        lambda _path, _payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="disk full"):
        runner_module._append_metric_rank_zero_collective(
            tmp_path / "metrics.jsonl",
            {"event": "train"},
        )


def _write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _resume_metrics_records() -> list[dict[str, object]]:
    identity = "a" * 64
    return [
        {
            "event": "start",
            "schema_version": runner_module.REPRESENTATION_RUNNER_SCHEMA_VERSION,
            "run_id": "representation-smoke",
            "run_identity_sha256": identity,
            "initial_global_step": 0,
        },
        {
            "event": "train",
            "run_identity_sha256": identity,
            "global_step": 1,
        },
        {
            "event": "validation",
            "run_identity_sha256": identity,
            "global_step": 1,
        },
        {
            "event": "train",
            "run_identity_sha256": identity,
            "global_step": 2,
        },
    ]


def test_resume_metrics_history_requires_exact_unfinished_checkpoint_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    _write_metrics(path, records)

    loaded = runner_module._validate_resume_metrics_history(
        path,
        run_id="representation-smoke",
        run_identity_sha256="a" * 64,
        checkpoint_global_step=2,
    )

    assert list(loaded) == records


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("identity", "changes run identity"),
        ("complete", "cannot be resumed"),
        ("advanced", "advanced beyond"),
        ("missing_checkpoint_train", "exactly one train event"),
        ("decreasing", "non-decreasing"),
    ),
)
def test_resume_metrics_history_rejects_drift_and_branching(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    if mutation == "identity":
        records[2]["run_identity_sha256"] = "b" * 64
    elif mutation == "complete":
        records.append(
            {
                "event": "complete",
                "run_identity_sha256": "a" * 64,
                "global_step": 2,
            }
        )
    elif mutation == "advanced":
        records[-1]["global_step"] = 3
    elif mutation == "missing_checkpoint_train":
        records.pop()
    elif mutation == "decreasing":
        records[1]["global_step"] = 2
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    _write_metrics(path, records)

    with pytest.raises((TypeError, ValueError), match=message):
        runner_module._validate_resume_metrics_history(
            path,
            run_id="representation-smoke",
            run_identity_sha256="a" * 64,
            checkpoint_global_step=2,
        )


def test_resume_metrics_history_rejects_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = _resume_metrics_records()
    _write_metrics(path, records)
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    with pytest.raises(ValueError, match="end with a newline"):
        runner_module._validate_resume_metrics_history(
            path,
            run_id="representation-smoke",
            run_identity_sha256="a" * 64,
            checkpoint_global_step=2,
        )


def test_checkpoint_retention_removes_only_current_invocation_paths(
    tmp_path: Path,
) -> None:
    preexisting = tmp_path / "representation-step-00000001"
    retired = tmp_path / "representation-step-00000002"
    current = tmp_path / "representation-step-00000003"
    for path in (preexisting, retired, current):
        path.mkdir()
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            directory=tmp_path,
            filename_prefix="representation",
        )
    )

    runner_module._remove_created_checkpoints_rank_zero(
        config,
        paths=(retired,),
        current=current,
    )

    assert preexisting.is_dir()
    assert not retired.exists()
    assert current.is_dir()


def test_final_export_can_strictly_reuse_resume_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "adapter.pt"
    path.touch()
    run_identity = object()
    manifest = SimpleNamespace(run_identity=run_identity, digest="same")
    export = SimpleNamespace(is_writer=True, manifest=manifest)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, *, src: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_rank_zero_adapter_owned_state_export",
        lambda candidate, *, expected_run_identity: (
            SimpleNamespace(manifest=manifest)
            if candidate == path and expected_run_identity is run_identity
            else (_ for _ in ()).throw(AssertionError("wrong export identity"))
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "save_rank_zero_adapter_owned_state_export_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identical resume artifact must not be overwritten")
        ),
    )

    assert (
        runner_module._save_rank_zero_export_collective(
            path,
            export,
            allow_existing_identical=True,
        )
        == "reused"
    )


def test_current_process_seed_never_calls_cuda_manual_seed_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = torch.random.get_rng_state()
    previous_python = random.getstate()
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(seed))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda _seed: (_ for _ in ()).throw(
            AssertionError("manual_seed_all touches peer visible devices")
        ),
    )
    try:
        runner_module._seed_current_process(731)
        assert torch.initial_seed() == 731
        assert random.random() == random.Random(731).random()
        assert calls == [731]
    finally:
        torch.random.set_rng_state(previous)
        random.setstate(previous_python)


def test_qwen_serial_loader_broadcasts_active_rank_failure_without_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transformers = ModuleType("transformers")

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            return object()

    class ModelLoader:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> nn.Module:
            raise RuntimeError("synthetic loader failure")

    fake_transformers.AutoProcessor = ProcessorLoader  # type: ignore[attr-defined]
    fake_transformers.AutoModelForImageTextToText = ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(values: list[object], value: object) -> None:
        values[:] = [value, value]

    sources: list[int] = []
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _values, *, src: sources.append(src),
    )
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda: (_ for _ in ()).throw(AssertionError("barrier can deadlock")),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            local_path="/synthetic/qwen",
            local_files_only=True,
            trust_remote_code=False,
            dtype="float32",
            attention_backend="sdpa",
        ),
        fsdp2=SimpleNamespace(world_size=2),
    )

    with pytest.raises(RuntimeError, match="rank 0.*synthetic loader failure"):
        runner_module._load_qwen3(config, device=torch.device("cpu"), rank=0)
    assert sources == [0]


def test_representation_dtype_mapping_has_no_implicit_fallback() -> None:
    assert runner_module._torch_dtype("bfloat16") is torch.bfloat16
    assert runner_module._torch_dtype("float32") is torch.float32
    for name in ("float16", "bf16", "auto", ""):
        with pytest.raises(ValueError, match="unsupported representation dtype"):
            runner_module._torch_dtype(name)


def test_importing_cli_does_not_import_representation_runner() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    script = (
        "import sys\n"
        "import tgvf_rl.cli\n"
        "name = 'tgvf_rl.representation.training.runner'\n"
        "raise SystemExit(1 if name in sys.modules else 0)\n"
    )

    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=repository_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_cli_run_command_lazily_dispatches_and_prints_rank_zero_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[Path] = []
    fake_runner = ModuleType("tgvf_rl.representation.training.runner")

    def fake_run(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"status": "synthetic", "gpu_work_launched": False}

    fake_runner.run_representation_training = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "tgvf_rl.representation.training.runner",
        fake_runner,
    )

    assert cli.main(["run-representation", "/unused/representation.toml"]) == 0
    assert calls == [Path("/unused/representation.toml")]
    assert json.loads(capsys.readouterr().out) == {
        "gpu_work_launched": False,
        "status": "synthetic",
    }
