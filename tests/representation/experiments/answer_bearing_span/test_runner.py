from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import pytest

from tgvf_rl.representation.experiments.answer_bearing_span import runner
from tgvf_rl.representation.experiments.image_axis_grounding.matching import (
    ImageAxisDonorManifest,
)
from tgvf_rl.representation.experiments.image_axis_grounding.native_pipeline import (
    ImageAxisGroundedNativeGroupBuilder,
)


def test_span_index_loader_covers_train_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = object()
    test = object()
    merged = object()
    train_dataset = object()
    test_dataset = object()
    data_calls: list[tuple[Path, str, bool]] = []
    span_calls: list[tuple[object, Path, str]] = []

    def load_data(
        path: Path,
        *,
        expected_source_sha256: str,
        warn_on_leakage: bool,
    ):
        data_calls.append((path, expected_source_sha256, warn_on_leakage))
        return train_dataset if path.name == "train.jsonl" else test_dataset

    def load_spans(dataset: object, path: Path, *, expected_sidecar_sha256: str):
        span_calls.append((dataset, path, expected_sidecar_sha256))
        return train if path.name == "train-spans.jsonl" else test

    def merge(first: object, second: object):
        assert (first, second) == (train, test)
        return merged

    monkeypatch.setattr(runner, "load_retained_representation_jsonl", load_data)
    monkeypatch.setattr(runner, "load_answer_bearing_span_index", load_spans)
    monkeypatch.setattr(runner, "merge_answer_bearing_span_indices", merge)
    config = SimpleNamespace(
        treatment_training=SimpleNamespace(
            data=SimpleNamespace(
                warn_on_target_leakage=False,
                train=SimpleNamespace(
                    jsonl_path=Path("/data/train.jsonl"), source_sha256="a" * 64
                ),
                validation=SimpleNamespace(
                    jsonl_path=Path("/data/test.jsonl"), source_sha256="b" * 64
                ),
            )
        ),
        train_span_sidecar_path=Path("/data/train-spans.jsonl"),
        train_span_sidecar_sha256="c" * 64,
        test_span_sidecar_path=Path("/data/test-spans.jsonl"),
        test_span_sidecar_sha256="d" * 64,
    )

    assert runner._load_span_index_set(config) == (train, test, merged)
    assert data_calls == [
        (Path("/data/train.jsonl"), "a" * 64, False),
        (Path("/data/test.jsonl"), "b" * 64, False),
    ]
    assert span_calls == [
        (train_dataset, Path("/data/train-spans.jsonl"), "c" * 64),
        (test_dataset, Path("/data/test-spans.jsonl"), "d" * 64),
    ]


def test_span_index_payload_exposes_sidecar_population_and_annotator_identity() -> None:
    def index(prefix: str) -> SimpleNamespace:
        return SimpleNamespace(
            identity_sha256=prefix * 64,
            sidecar_sha256=prefix.upper() * 64,
            retained_semantic_population_sha256=(prefix * 2) * 32,
            annotator_identity=f"annotator-{prefix}:v1",
            statistics=SimpleNamespace(canonical_payload=lambda: {"total_rows": 1}),
        )

    train = index("a")
    test = index("b")
    merged = SimpleNamespace(
        identity_sha256="c" * 64,
        statistics=SimpleNamespace(canonical_payload=lambda: {"total_rows": 2}),
    )

    payload = runner._span_index_payload(train, test, merged)

    assert payload["train_span_sidecar_sha256"] == "A" * 64
    assert payload["test_span_population_sha256"] == "bb" * 32
    assert payload["train_span_annotator_identity"] == "annotator-a:v1"
    assert payload["combined_span_statistics"] == {"total_rows": 2}


def test_injection_adds_span_factory_before_image_axis_wrapper_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_builder: dict[str, object] = {}
    captured_trainer: dict[str, object] = {}

    class FakeNativeBuilder:
        def __init__(self, **kwargs: object) -> None:
            captured_builder.update(kwargs)

        def __call__(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

    class FakeCoreTrainer:
        pass

    class FakeImageAxisTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured_trainer.update(kwargs)

    monkeypatch.setattr(
        runner, "Qwen3NativeRepresentationGroupBuilder", FakeNativeBuilder
    )
    monkeypatch.setattr(runner, "RepresentationTrainer", FakeCoreTrainer)
    monkeypatch.setattr(
        runner.core_runner, "Qwen3NativeRepresentationGroupBuilder", FakeNativeBuilder
    )
    monkeypatch.setattr(runner.core_runner, "RepresentationTrainer", FakeCoreTrainer)
    monkeypatch.setattr(runner, "ImageAxisGroundingTrainer", FakeImageAxisTrainer)

    config = SimpleNamespace(objective=object())
    manifest = object.__new__(ImageAxisDonorManifest)

    class SpanFactory:
        def __call__(self, *args: object) -> object:
            return object()

    span_factory = SpanFactory()
    with runner._inject_answer_bearing_span_components(
        config,
        manifest,
        loss_supervision_factory=span_factory,
    ):
        assert runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL is (
            runner.core_runner._ANSWER_BEARING_SPAN_DRIVER_SEAL
        )
        builder = runner.core_runner.Qwen3NativeRepresentationGroupBuilder(
            runtime="runtime"
        )
        assert isinstance(builder, FakeNativeBuilder)
        assert captured_builder == {
            "runtime": "runtime",
            "readout_loss_supervision_factory": span_factory,
        }

        trainer = runner.core_runner.RepresentationTrainer(
            group_builder=builder,
            sentinel="kept",
        )
        assert isinstance(trainer, FakeImageAxisTrainer)
        wrapped = captured_trainer["group_builder"]
        assert isinstance(wrapped, ImageAxisGroundedNativeGroupBuilder)
        assert wrapped.base_builder is builder
        assert captured_trainer["sentinel"] == "kept"

    assert runner.core_runner.Qwen3NativeRepresentationGroupBuilder is FakeNativeBuilder
    assert runner.core_runner.RepresentationTrainer is FakeCoreTrainer
    assert runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL is None


def test_rp70_objective_requires_experiment_driver_attestation() -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            objective=SimpleNamespace(
                identity=(
                    "answer-bearing-span-balanced-matrix-ce-l-gen-norm-"
                    "plus-image-axis-v1:fixture"
                )
            )
        )
    )

    with pytest.raises(RuntimeError, match="direct core.*forbidden"):
        runner.core_runner._require_experiment_driver_attestation(config)

    original = runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL
    runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL = (
        runner.core_runner._ANSWER_BEARING_SPAN_DRIVER_SEAL
    )
    try:
        runner.core_runner._require_experiment_driver_attestation(config)
    finally:
        runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL = original


def test_direct_public_core_runner_rejects_rp70_before_runtime_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        objective=SimpleNamespace(
            objective=SimpleNamespace(
                identity=(
                    "answer-bearing-span-balanced-matrix-ce-l-gen-norm-"
                    "plus-image-axis-v1:fixture"
                )
            )
        )
    )
    calls: list[str] = []

    def load(_path: object) -> object:
        calls.append("load")
        return config

    def forbidden(name: str):
        def call(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"RP70 direct core launch reached {name}")

        return call

    monkeypatch.setattr(runner.core_runner, "_ACTIVE_EXPERIMENT_DRIVER_SEAL", None)
    monkeypatch.setattr(
        runner.core_runner,
        "load_representation_training_config",
        load,
    )
    monkeypatch.setattr(
        runner.core_runner,
        "_validate_invocation_stop",
        forbidden("stop validation"),
    )
    monkeypatch.setattr(
        runner.core_runner,
        "_require_launch_environment",
        forbidden("launch environment"),
    )
    monkeypatch.setattr(
        runner.core_runner,
        "_verify_live_code_identity",
        forbidden("live-code verification"),
    )
    monkeypatch.setattr(
        runner.core_runner.torch.distributed,
        "init_process_group",
        forbidden("distributed initialization"),
    )
    monkeypatch.setattr(
        runner.core_runner.torch.cuda,
        "set_device",
        forbidden("CUDA device initialization"),
    )

    with pytest.raises(RuntimeError, match="direct core.*forbidden"):
        runner.core_runner.run_representation_training("/unused/rp70.toml")

    assert calls == ["load"]


def test_injection_restores_all_core_state_when_body_raises() -> None:
    original_builder = runner.core_runner.Qwen3NativeRepresentationGroupBuilder
    original_trainer = runner.core_runner.RepresentationTrainer
    original_driver_seal = runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL

    with pytest.raises(LookupError, match="synthetic core failure"):
        with runner._inject_answer_bearing_span_components(
            SimpleNamespace(objective=object()),
            object.__new__(ImageAxisDonorManifest),
            loss_supervision_factory=lambda *args: object(),
        ):
            assert (
                runner.core_runner.Qwen3NativeRepresentationGroupBuilder
                is not original_builder
            )
            assert runner.core_runner.RepresentationTrainer is not original_trainer
            assert runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL is (
                runner.core_runner._ANSWER_BEARING_SPAN_DRIVER_SEAL
            )
            raise LookupError("synthetic core failure")

    assert runner.core_runner.Qwen3NativeRepresentationGroupBuilder is original_builder
    assert runner.core_runner.RepresentationTrainer is original_trainer
    assert runner.core_runner._ACTIVE_EXPERIMENT_DRIVER_SEAL is original_driver_seal


def test_injection_rejects_prepatched_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.core_runner,
        "Qwen3NativeRepresentationGroupBuilder",
        lambda **_: object(),
    )

    with pytest.raises(RuntimeError, match="already patched"):
        with runner._inject_answer_bearing_span_components(
            SimpleNamespace(objective=object()),
            object.__new__(ImageAxisDonorManifest),
            loss_supervision_factory=lambda *args: object(),
        ):
            pass
