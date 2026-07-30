from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from tgvf_rl.representation.experiments.answer_utility.run_config import (
    ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION,
    ANSWER_UTILITY_RUN_SCOPE,
    AnswerUtilityRunConfig,
    load_answer_utility_run_config,
)
from tgvf_rl.representation.experiments.answer_utility.runner import (
    ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
    AnswerUtilityCheckpoint,
    _answer_utility_state_digest,
    _atomic_torch_save,
    _load_checkpoint,
    _run_identity_sha256,
    _validate_output_preflight,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_run_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    experiment = tmp_path / "answer-utility-experiment.toml"
    experiment.write_text('schema_version = "fixture"\n', encoding="utf-8")
    artifact = tmp_path / "source-adapter.pt"
    artifact.write_bytes(b"fixture-adapter")
    output = tmp_path / "answer-utility-output"
    run_config = tmp_path / "answer-utility-run.toml"
    run_config.write_text(
        f'''schema_version = "{ANSWER_UTILITY_RUN_CONFIG_SCHEMA_VERSION}"
scope = "{ANSWER_UTILITY_RUN_SCOPE}"
run_id = "answer-utility-run-fixture"

[experiment]
config_path = "{experiment}"
config_sha256 = "{_sha(experiment)}"

[source_artifact]
path = "{artifact}"
file_sha256 = "{_sha(artifact)}"
manifest_sha256 = "{_SHA_A}"
expected_run_identity_sha256 = "{_SHA_B}"
expected_global_step = 2000

[execution]
physical_gpu_id = 0
seed = 17

[optimizer]
learning_rate = 0.000001

[training]
target_optimizer_steps = 80
checkpoint_every_optimizer_steps = 10
log_every_optimizer_steps = 5

[output]
directory = "{output}"

[resume]
enabled = false
checkpoint_path = "none"
''',
        encoding="utf-8",
    )
    return run_config, experiment, artifact


def _checkpoint(*, run_identity_sha256: str = _SHA_A) -> AnswerUtilityCheckpoint:
    adapter_state = {"projector.weight": torch.arange(6).reshape(2, 3)}
    optimizer_state: dict[str, object] = {
        "state": {0: {"step": 2, "exp_avg": torch.ones(2)}},
        "param_groups": [{"lr": 1e-6, "params": [0]}],
    }
    sampler_state: dict[str, object] = {"epoch": 3, "cursor": 7}
    rng_state: dict[str, object] = {
        "python": (3, (1, 2, 3), None),
        "torch_cpu": torch.tensor((4, 5, 6), dtype=torch.uint8),
    }
    return AnswerUtilityCheckpoint(
        run_identity_sha256=run_identity_sha256,
        global_step=12,
        adapter_state=adapter_state,
        optimizer_state=optimizer_state,
        sampler_state=sampler_state,
        rng_state=rng_state,
        adapter_state_sha256=_answer_utility_state_digest(adapter_state),
        optimizer_state_sha256=_answer_utility_state_digest(optimizer_state),
        sampler_state_sha256=_answer_utility_state_digest(sampler_state),
        rng_state_sha256=_answer_utility_state_digest(rng_state),
    )


def test_load_run_config_is_strict_and_binds_both_input_files(
    tmp_path: Path,
) -> None:
    path, experiment, artifact = _write_run_config(tmp_path)

    loaded = load_answer_utility_run_config(path)

    assert loaded.experiment_config_path == experiment
    assert loaded.experiment_config_sha256 == _sha(experiment)
    assert loaded.source_artifact.path == artifact
    assert loaded.source_artifact.file_sha256 == _sha(artifact)
    assert loaded.source_toml_sha256 == _sha(path)
    assert len(loaded.canonical_config_sha256) == 64

    experiment.write_text('schema_version = "changed"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="experiment config SHA256 mismatch"):
        load_answer_utility_run_config(path)

    experiment.write_text('schema_version = "fixture"\n', encoding="utf-8")
    artifact.write_bytes(b"changed-adapter")
    with pytest.raises(ValueError, match="source Adapter artifact SHA256 mismatch"):
        load_answer_utility_run_config(path)


def test_load_run_config_rejects_unknown_toml_field(tmp_path: Path) -> None:
    path, _, _ = _write_run_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "learning_rate = 0.000001",
            "learning_rate = 0.000001\nmomentum = 0.9",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[optimizer\] fields differ"):
        load_answer_utility_run_config(path)


def test_run_identity_ignores_resume_cursor_and_sidecar_byte_hashes(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_run_config(tmp_path)
    fresh = load_answer_utility_run_config(path)
    resumed: AnswerUtilityRunConfig = replace(
        fresh,
        resume_checkpoint_path=(tmp_path / "checkpoints" / "step-12.pt"),
        source_toml_sha256=_SHA_A,
        canonical_config_sha256=_SHA_B,
    )
    inputs = {
        "experiment_canonical_sha256": _SHA_A,
        "base_training_config_sha256": _SHA_B,
        "train_manifest_sha256": _SHA_C,
    }

    assert _run_identity_sha256(fresh, **inputs) == _run_identity_sha256(
        resumed, **inputs
    )
    assert _run_identity_sha256(
        replace(fresh, learning_rate=2e-6), **inputs
    ) != _run_identity_sha256(fresh, **inputs)


def test_checkpoint_validates_state_digests_and_run_identity(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    checkpoint = _checkpoint()
    _atomic_torch_save(path, checkpoint)

    loaded = _load_checkpoint(
        path,
        expected_run_identity_sha256=_SHA_A,
        map_location="cpu",
    )
    assert loaded.global_step == 12
    assert torch.equal(
        loaded.adapter_state["projector.weight"],
        checkpoint.adapter_state["projector.weight"],
    )

    with pytest.raises(ValueError, match="belongs to another"):
        _load_checkpoint(
            path,
            expected_run_identity_sha256=_SHA_B,
            map_location="cpu",
        )

    tampered = _checkpoint()
    object.__setattr__(
        tampered,
        "adapter_state",
        {"projector.weight": torch.full((2, 3), 99)},
    )
    tampered_path = tmp_path / "tampered-checkpoint.pt"
    _atomic_torch_save(tampered_path, tampered)
    with pytest.raises(ValueError, match="Adapter state digest mismatch"):
        _load_checkpoint(
            tampered_path,
            expected_run_identity_sha256=_SHA_A,
            map_location="cpu",
        )


def test_checkpoint_digest_supports_real_adamw_scalar_step_tensor() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW((parameter,), lr=1e-6)
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    state = optimizer.state_dict()

    step = state["state"][0]["step"]
    assert isinstance(step, torch.Tensor) and step.ndim == 0
    digest = _answer_utility_state_digest(state)

    assert len(digest) == 64
    assert digest == _answer_utility_state_digest(state)


def test_resume_preflight_validates_latest_checkpoint_and_metrics_before_gpu(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_run_config(tmp_path)
    fresh = load_answer_utility_run_config(path)
    fresh.checkpoint_directory.mkdir(parents=True)
    checkpoint_path = fresh.checkpoint_directory / "answer-utility-step-00000012.pt"
    _atomic_torch_save(checkpoint_path, _checkpoint())
    records = (
        {
            "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
            "event": "start",
            "run_identity_sha256": _SHA_A,
        },
        {
            "schema_version": ANSWER_UTILITY_METRICS_SCHEMA_VERSION,
            "event": "step",
            "run_identity_sha256": _SHA_A,
            "global_step": 12,
        },
    )
    fresh.metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    resumed = replace(fresh, resume_checkpoint_path=checkpoint_path)

    checkpoint, high_water = _validate_output_preflight(
        resumed,
        run_identity_sha256=_SHA_A,
    )

    assert checkpoint is not None and checkpoint.global_step == 12
    assert high_water == 12

    newer = fresh.checkpoint_directory / "answer-utility-step-00000013.pt"
    _atomic_torch_save(
        newer,
        replace(_checkpoint(), global_step=13),
    )
    with pytest.raises(ValueError, match="latest durable"):
        _validate_output_preflight(resumed, run_identity_sha256=_SHA_A)


def test_checkpoint_global_step_rejects_boolean() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(_checkpoint(), global_step=True)


def test_atomic_torch_save_never_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "immutable.pt"
    _atomic_torch_save(path, {"version": 1})

    with pytest.raises(FileExistsError):
        _atomic_torch_save(path, {"version": 2})

    assert torch.load(path, map_location="cpu", weights_only=False) == {"version": 1}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
