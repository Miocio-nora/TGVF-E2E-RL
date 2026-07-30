from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.representation.experiments.image_axis_grounding import runner


def _preflight_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    preprocessor = tmp_path / "model" / "preprocessor_config.json"
    preprocessor.parent.mkdir()
    preprocessor.write_bytes(b"processor-v1")
    keys = tuple(f"group-{index:05d}" for index in range(8_209))
    source = SimpleNamespace(
        train_source_sha256="1" * 64,
        retained_manifest_sha256="2" * 64,
        raw_image_manifest_sha256="3" * 64,
        preprocessor_config_sha256=sha256(preprocessor.read_bytes()).hexdigest(),
    )
    grid = object()
    manifest = SimpleNamespace(
        source_binding=source,
        grid_contract=grid,
        assignments=tuple(
            SimpleNamespace(anchor_image_group_key=key) for key in keys
        ),
    )
    train_manifest = SimpleNamespace(manifest_sha256="2" * 64)
    train_data = SimpleNamespace(samples=(object(),), manifest=train_manifest)
    training = SimpleNamespace(
        data=SimpleNamespace(
            train=SimpleNamespace(
                source_sha256="1" * 64,
                jsonl_path=tmp_path / "train.jsonl",
                batch_size=4,
                sampler_seed=42,
            ),
            warn_on_target_leakage=False,
        ),
        model=SimpleNamespace(
            local_path=preprocessor.parent,
            image_max_pixels=262_144,
        ),
        fsdp2=SimpleNamespace(world_size=2),
    )
    config = SimpleNamespace(
        donor_manifest_path=tmp_path / "donors.json",
        treatment_training=training,
    )
    monkeypatch.setattr(runner, "load_image_axis_donor_manifest", lambda _: manifest)
    monkeypatch.setattr(
        runner,
        "load_retained_representation_jsonl",
        lambda *args, **kwargs: train_data,
    )
    monkeypatch.setattr(
        runner,
        "build_retained_image_raw_byte_manifest",
        lambda _: SimpleNamespace(manifest_sha256="3" * 64),
    )
    monkeypatch.setattr(
        runner,
        "load_qwen_image_grid_contract",
        lambda *args, **kwargs: grid,
    )

    class FakeSampler:
        def __init__(self, *args, rank: int, **kwargs) -> None:
            midpoint = len(keys) // 2
            self.owned_group_keys = keys[:midpoint] if rank == 0 else keys[midpoint:]

    monkeypatch.setattr(runner, "SameImageBatchSampler", FakeSampler)
    return config, manifest, source, keys


def test_manifest_preflight_rebuilds_all_source_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, _, _ = _preflight_fixture(tmp_path, monkeypatch)

    assert runner._load_and_validate_manifest(config) is manifest


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("retained_manifest_sha256", "retained-data SHA256 mismatch"),
        ("raw_image_manifest_sha256", "raw-image SHA256 mismatch"),
        ("preprocessor_config_sha256", "preprocessor-config SHA256 mismatch"),
    ),
)
def test_manifest_preflight_rejects_source_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    config, _, source, _ = _preflight_fixture(tmp_path, monkeypatch)
    setattr(source, field, "f" * 64)

    with pytest.raises(ValueError, match=message):
        runner._load_and_validate_manifest(config)


def test_manifest_preflight_rejects_non_sampler_assignment_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, _, keys = _preflight_fixture(tmp_path, monkeypatch)
    manifest.assignments = tuple(
        SimpleNamespace(anchor_image_group_key=key)
        for key in (*keys[:-1], "wrong-group")
    )

    with pytest.raises(ValueError, match="exact world2/K4 sampler closure"):
        runner._load_and_validate_manifest(config)


def test_manifest_preflight_rejects_changed_usable_group_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _, _ = _preflight_fixture(tmp_path, monkeypatch)

    class ShortSampler:
        def __init__(self, *args, rank: int, **kwargs) -> None:
            self.owned_group_keys = (f"rank-{rank}",)

    monkeypatch.setattr(runner, "SameImageBatchSampler", ShortSampler)
    with pytest.raises(ValueError, match="expected 8209, got 2"):
        runner._load_and_validate_manifest(config)
