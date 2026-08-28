from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.evaluation import policy_benchmark_config as benchmark_config
from tgvf_rl.evaluation import policy_coredev as coredev


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_PLAN = _ROOT / (
    "configs/evaluation/"
    "prl25_b_crop_exact_step32_true1m_resolution_rng_extension_"
    "v5_coredev2511_plan.json"
)
_S80_PLAN = _ROOT / (
    "configs/evaluation/"
    "prl25_b_crop_exact_step80_true1m_resolution_rng_projection_"
    "v5_coredev2511_plan.json"
)
_REFERENCE = _ROOT / (
    "configs/evaluation/"
    "prl25_b_crop_exact_step80_true1m_true512_resolution_pair_"
    "v4_coredev2511_plan.json"
)
_SPEC = importlib.util.spec_from_file_location("step32_true1m_runner", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def _payload() -> dict[str, object]:
    return json.loads(_PLAN.read_text(encoding="utf-8"))


def _write_plan(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _snapshot(*, step: int) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            model=SimpleNamespace(model_name="Qwen3-VL-8B-Instruct"),
            policy=SimpleNamespace(
                image_max_pixels=1003520,
                sampling=SimpleNamespace(temperature=1.0, do_sample=True),
            ),
            rollout_rng=SimpleNamespace(master_seed=42),
        ),
        policy_version=SimpleNamespace(optimizer_step=step),
    )


def _rng_config(payload: dict[str, object]) -> SimpleNamespace:
    paired = payload["paired_rng"]
    assert isinstance(paired, dict)
    return SimpleNamespace(
        evaluation_protocol=coredev.DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        paired_seed_namespace=paired["seed_namespace"],
        paired_rng_protocol_projection=(
            coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
        ),
        evaluation_image_max_pixels=1003520,
    )


def test_v5_plan_loads_one_step32_true1m_arm_and_exact_owner() -> None:
    plan = _RUNNER._load_plan(_PLAN)
    runtime = _RUNNER._load_evaluation_runtime(plan)

    assert plan["schema_version"] == _RUNNER.PLAN_SCHEMA_V5
    assert [
        (
            arm["name"],
            arm["optimizer_step"],
            arm["evaluation_image_max_pixels"],
        )
        for arm in plan["arms"]
    ] == [("step32", 32, 1003520)]
    assert _RUNNER._arm_image_max_pixels(plan, "step32") == 1003520
    assert _RUNNER._arm_rng_protocol_projection(plan, "step32") == (
        coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
    )
    assert runtime.backend == _RUNNER.FULL_MODEL_BACKEND
    assert runtime.checkpoint_owner.run_id == plan["checkpoint_owner"]["run_id"]


def test_v5_step32_stream_exactly_matches_referenced_s80_true1m_stream() -> None:
    plan = _payload()
    reference = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    config = _rng_config(plan)
    step32 = coredev.paired_evaluation_rng_contract(
        config,
        _snapshot(step=32),
        task_manifest_sha256=plan["task_manifest_sha256"],
    )
    step80 = coredev.paired_evaluation_rng_contract(
        config,
        _snapshot(step=80),
        task_manifest_sha256=reference["task_manifest_sha256"],
    )
    assert step32 == step80
    assert step32 is not None
    assert (
        step32["arm_protocol_sha256"]
        == plan["paired_rng"]["arm_protocol_sha256"]["step32"]
    )
    assert (
        step32["arm_protocol_sha256"]
        == reference["paired_rng"]["arm_protocol_sha256"]["pixel1003520"]
    )
    assert (
        step32["seed_protocol_sha256"]
        == reference["paired_rng"]["seed_protocol_sha256"]
    )

    rng32 = coredev.paired_evaluation_rng_for_task(
        {"sampling_rng": step32}, sample_id="sample-7", rollout_index=0
    )
    rng80 = coredev.paired_evaluation_rng_for_task(
        {"sampling_rng": step80}, sample_id="sample-7", rollout_index=0
    )
    turn32 = rng32.for_turn(
        (1, 2, 3),
        turn_index=2,
        behavior_policy=PolicyVersion("crop-s32", 32, "a" * 64),
    )
    turn80 = rng80.for_turn(
        (9, 8, 7),
        turn_index=2,
        behavior_policy=PolicyVersion("crop-s80", 80, "b" * 64),
    )
    assert turn32 == turn80


def test_v5_step80_projection_reuses_exact_reference_arm_and_rng() -> None:
    plan = _RUNNER._load_plan(_S80_PLAN)
    reference = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    arm = plan["arms"][0]

    assert (
        arm["name"],
        arm["optimizer_step"],
        arm["evaluation_image_max_pixels"],
    ) == ("pixel1003520", 80, 1003520)
    assert arm == reference["arms"][0]
    assert (
        plan["paired_rng"]["arm_protocol_sha256"][arm["name"]]
        == (reference["paired_rng"]["arm_protocol_sha256"][arm["name"]])
    )
    assert (
        plan["paired_rng"]["seed_protocol_sha256"]
        == reference["paired_rng"]["seed_protocol_sha256"]
    )


def test_v5_step80_projection_reuses_v4_shared_snapshot_paths(
    tmp_path: Path,
) -> None:
    v4 = _RUNNER._load_plan(_REFERENCE)
    v5 = _RUNNER._load_plan(_S80_PLAN)

    v4_paths = _RUNNER._arm_paths_for_plan(v4, tmp_path, "pixel1003520")
    v5_paths = _RUNNER._arm_paths_for_plan(v5, tmp_path, "pixel1003520")
    for field in (
        "full_model_snapshot",
        "full_model_receipt",
        "full_model_merge",
    ):
        assert v5_paths[field] == v4_paths[field]
        assert "runtime/resolution-pair-shared-full-model" in str(v5_paths[field])


def test_v5_step32_reaches_full_model_config_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = tmp_path / "owner.toml"
    manifest = tmp_path / "snapshot.json"
    receipt = tmp_path / "receipt.json"
    tasks = tmp_path / "tasks.jsonl"
    for path, content in (
        (policy, b"owner"),
        (manifest, b"manifest"),
        (receipt, b"receipt"),
        (tasks, b"task\n"),
    ):
        path.write_bytes(content)
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    snapshot = SimpleNamespace(
        manifest=SimpleNamespace(
            checkpoint_owner=SimpleNamespace(config_file_sha256=policy_sha256),
            run_contract_file_sha256="0" * 64,
            identity_sha256="1" * 64,
        ),
        policy_version=SimpleNamespace(
            run_id="crop-s32", optimizer_step=32, weights_sha256="2" * 64
        ),
        run_identity_sha256="3" * 64,
    )
    monkeypatch.setattr(
        benchmark_config,
        "load_full_model_evaluation_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(benchmark_config, "load_benchmark_tasks", lambda *_a, **_k: ())

    output = tmp_path / "step32"
    payload = benchmark_config.materialize_full_model_policy_benchmark_config(
        evaluation_id="step32-true1m",
        policy_config_path=policy,
        snapshot_manifest_path=manifest,
        materialization_receipt_path=receipt,
        expected_optimizer_step=32,
        task_manifest_path=tasks,
        expected_task_count=2511,
        expected_single_image_count=2240,
        output_root=output,
        config_path=output / "benchmark-config.json",
        gpu_ids=(0, 1, 2, 3),
        paired_seed_namespace="resolution-projected/step32/true1m/v1",
        paired_rng_protocol_projection=(
            coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
        ),
        evaluation_image_max_pixels=1003520,
    )

    materialized = coredev.PolicyCoreDevConfig(**payload)
    assert materialized.expected_optimizer_step == 32
    assert materialized.evaluation_image_max_pixels == 1003520
    assert (output / "benchmark-config.json").is_file()
    with pytest.raises(ValueError, match="step32 true1M"):
        coredev.PolicyCoreDevConfig(
            **{**payload, "evaluation_image_max_pixels": 262144}
        )
    with pytest.raises(ValueError, match="step32 true1M"):
        coredev.PolicyCoreDevConfig(**{**payload, "expected_optimizer_step": 16})


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["arms"][0].__setitem__(
                "evaluation_image_max_pixels", 262144
            ),
            "arm differs",
        ),
        (
            lambda payload: payload["paired_rng"].__setitem__(
                "seed_namespace", "different/namespace"
            ),
            "differs from referenced S80 true1M arm",
        ),
        (
            lambda payload: payload["rng_reference"].__setitem__(
                "plan_sha256", "0" * 64
            ),
            "reference identity differs",
        ),
    ],
)
def test_v5_rejects_cap_rng_and_reference_drift(
    tmp_path: Path, mutation, error: str
) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises((ValueError, RuntimeError), match=error):
        _RUNNER._load_plan(_write_plan(tmp_path, payload))
