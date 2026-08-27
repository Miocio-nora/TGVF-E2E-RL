from __future__ import annotations

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
_PLAN = (
    _ROOT
    / "configs/evaluation/"
    "prl25_b_crop_exact_step80_true1m_true512_"
    "resolution_pair_v4_coredev2511_plan.json"
)
_SPEC = importlib.util.spec_from_file_location("resolution_pair_runner", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def _payload() -> dict[str, object]:
    return json.loads(_PLAN.read_text(encoding="utf-8"))


def _write_plan(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            model=SimpleNamespace(model_name="Qwen3-VL-8B-Instruct"),
            policy=SimpleNamespace(
                image_max_pixels=1003520,
                sampling=SimpleNamespace(temperature=1.0, do_sample=True),
            ),
            rollout_rng=SimpleNamespace(master_seed=42),
        ),
        policy_version=SimpleNamespace(optimizer_step=80),
    )


def _rng_config(*, cap: int) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation_protocol=coredev.DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL,
        paired_seed_namespace=(
            "coredev2511-official-v1/prl25-b-crop-exact/step80/"
            "pixel1003520-vs-pixel262144/temp1/seed42/v1"
        ),
        paired_rng_protocol_projection=(
            coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
        ),
        evaluation_image_max_pixels=cap,
    )


def test_v4_plan_loads_one_same_step_checkpoint_with_two_arm_caps() -> None:
    plan = _RUNNER._load_plan(_PLAN)
    runtime = _RUNNER._load_evaluation_runtime(plan)

    assert plan["schema_version"] == _RUNNER.PLAN_SCHEMA_V4
    assert "evaluation_image_max_pixels" not in plan
    assert [
        (arm["name"], arm["optimizer_step"], arm["evaluation_image_max_pixels"])
        for arm in plan["arms"]
    ] == [
        ("pixel1003520", 80, 1003520),
        ("pixel262144", 80, 262144),
    ]
    assert plan["arms"][0]["source"] == plan["arms"][1]["source"]
    assert runtime.backend == _RUNNER.FULL_MODEL_BACKEND
    assert runtime.checkpoint_owner.run_id == plan["checkpoint_owner"]["run_id"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["arms"][0].__setitem__(
                "evaluation_image_max_pixels", 999999
            ),
            "step/cap",
        ),
        (
            lambda payload: payload["arms"][1].__setitem__(
                "optimizer_step", 79
            ),
            "step/cap",
        ),
        (
            lambda payload: payload["arms"][1]["source"].__setitem__(
                "relative_path", "permanent-checkpoints/global_step_79"
            ),
            "share one checkpoint source",
        ),
        (
            lambda payload: payload["resolution_pair"].__setitem__(
                "excluded_protocol_fields", ["image_max_pixels", "prompt_sha256"]
            ),
            "projection differs",
        ),
        (
            lambda payload: payload["paired_rng"].__setitem__(
                "excluded_protocol_fields", ["image_max_pixels"]
            ),
            "RNG fields differ",
        ),
        (
            lambda payload: payload.__setitem__(
                "evaluation_image_max_pixels", 262144
            ),
            "plan fields/status differ",
        ),
    ],
)
def test_v4_plan_rejects_cap_step_source_and_projection_drift(
    tmp_path: Path, mutation, error: str
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=error):
        _RUNNER._load_plan(_write_plan(tmp_path, payload))


def test_resolution_projection_retains_arm_protocol_but_shares_seed_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _payload()
    snapshot = _snapshot()
    one_m = coredev.paired_evaluation_rng_contract(
        _rng_config(cap=1003520),
        snapshot,
        task_manifest_sha256=plan["task_manifest_sha256"],
    )
    pixel512 = coredev.paired_evaluation_rng_contract(
        _rng_config(cap=262144),
        snapshot,
        task_manifest_sha256=plan["task_manifest_sha256"],
    )
    assert one_m is not None and pixel512 is not None

    assert one_m["arm_protocol_sha256"] != pixel512["arm_protocol_sha256"]
    assert one_m["arm_protocol_sha256"] == plan["paired_rng"][
        "arm_protocol_sha256"
    ]["pixel1003520"]
    assert pixel512["arm_protocol_sha256"] == plan["paired_rng"][
        "arm_protocol_sha256"
    ]["pixel262144"]
    assert one_m["seed_protocol_sha256"] == pixel512["seed_protocol_sha256"]
    assert one_m["seed_protocol_sha256"] == plan["paired_rng"][
        "seed_protocol_sha256"
    ]
    assert one_m["protocol_projection"] == {
        "kind": "image_max_pixels_resolution_pair_v1",
        "excluded_protocol_field": "image_max_pixels",
        "axis_values": [262144, 1003520],
    }

    one_m_rng = coredev.paired_evaluation_rng_for_task(
        {"sampling_rng": one_m}, sample_id="sample-7", rollout_index=0
    )
    pixel512_rng = coredev.paired_evaluation_rng_for_task(
        {"sampling_rng": pixel512}, sample_id="sample-7", rollout_index=0
    )
    assert one_m_rng.stream_identity == {
        "schema_version": coredev.RESOLUTION_PAIRED_POLICY_EVALUATION_RNG_SCHEMA,
        "seed_namespace": plan["paired_rng"]["seed_namespace"],
        "master_seed": 42,
        "task_manifest_sha256": plan["task_manifest_sha256"],
        "sample_id": "sample-7",
        "rollout_index": 0,
        "seed_protocol_sha256": plan["paired_rng"]["seed_protocol_sha256"],
    }
    assert "protocol_sha256" not in one_m_rng.stream_identity
    policy = PolicyVersion("crop-s80", 80, "a" * 64)
    resolution_turn = one_m_rng.for_turn(
        (1, 2, 3), turn_index=2, behavior_policy=policy
    )
    assert resolution_turn == pixel512_rng.for_turn(
        (9, 8, 7), turn_index=2, behavior_policy=policy
    )

    legacy_rng = coredev.PairedEvaluationVLLMTurnRNG(
        master_seed=42,
        seed_namespace=plan["paired_rng"]["seed_namespace"],
        task_manifest_sha256=plan["task_manifest_sha256"],
        protocol_sha256=plan["paired_rng"]["seed_protocol_sha256"],
        sample_id="sample-7",
        rollout_index=0,
    )
    assert legacy_rng.stream_identity["schema_version"] == (
        coredev.PAIRED_POLICY_EVALUATION_RNG_SCHEMA
    )
    assert "protocol_sha256" in legacy_rng.stream_identity
    assert "seed_protocol_sha256" not in legacy_rng.stream_identity
    assert legacy_rng.for_turn(
        (1, 2, 3), turn_index=2, behavior_policy=policy
    ) != resolution_turn

    task_changed = coredev.paired_evaluation_rng_contract(
        _rng_config(cap=1003520),
        snapshot,
        task_manifest_sha256="f" * 64,
    )
    assert task_changed is not None
    task_rng = coredev.paired_evaluation_rng_for_task(
        {"sampling_rng": task_changed}, sample_id="sample-7", rollout_index=0
    )
    assert task_rng.for_turn(
        (1,), turn_index=2, behavior_policy=policy
    ) != one_m_rng.for_turn((1,), turn_index=2, behavior_policy=policy)

    from tgvf_rl.policy import deepeyes_official_protocol

    monkeypatch.setattr(
        deepeyes_official_protocol, "USER_PROMPT_V2_SHA256", "e" * 64
    )
    prompt_changed = coredev.paired_evaluation_rng_contract(
        _rng_config(cap=1003520),
        snapshot,
        task_manifest_sha256=plan["task_manifest_sha256"],
    )
    assert prompt_changed is not None
    assert prompt_changed["seed_protocol_sha256"] != one_m[
        "seed_protocol_sha256"
    ]


def test_full_model_config_materializer_serializes_only_fixed_projection(
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
            checkpoint_owner=SimpleNamespace(
                config_file_sha256=policy_sha256
            ),
            run_contract_file_sha256="0" * 64,
            identity_sha256="1" * 64,
        ),
        policy_version=SimpleNamespace(
            run_id="crop-s80", optimizer_step=80, weights_sha256="2" * 64
        ),
        run_identity_sha256="3" * 64,
    )
    monkeypatch.setattr(
        benchmark_config,
        "load_full_model_evaluation_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(benchmark_config, "load_benchmark_tasks", lambda *_a, **_k: ())

    payloads = []
    for name, cap, gpus in (
        ("pixel1003520", 1003520, (0, 1, 2, 3)),
        ("pixel262144", 262144, (4, 5, 6, 7)),
    ):
        output = tmp_path / name
        payloads.append(
            benchmark_config.materialize_full_model_policy_benchmark_config(
                evaluation_id=name,
                policy_config_path=policy,
                snapshot_manifest_path=manifest,
                materialization_receipt_path=receipt,
                expected_optimizer_step=80,
                task_manifest_path=tasks,
                expected_task_count=2511,
                expected_single_image_count=2240,
                output_root=output,
                config_path=output / "benchmark-config.json",
                gpu_ids=gpus,
                paired_seed_namespace="resolution-pair/v1",
                paired_rng_protocol_projection=(
                    coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
                ),
                evaluation_image_max_pixels=cap,
            )
        )

    assert [item["evaluation_image_max_pixels"] for item in payloads] == [
        1003520,
        262144,
    ]
    assert {item["paired_rng_protocol_projection"] for item in payloads} == {
        coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
    }
    assert {item["full_model_snapshot_manifest_path"] for item in payloads} == {
        str(manifest.resolve())
    }
    for payload in payloads:
        coredev.PolicyCoreDevConfig(**payload)
        with pytest.raises(ValueError, match="projection differs"):
            coredev.PolicyCoreDevConfig(
                **{
                    **payload,
                    "paired_rng_protocol_projection": "arbitrary_field_drop_v1",
                }
            )


def test_v4_runner_assigns_per_arm_caps_and_one_shared_model_tree(
    tmp_path: Path,
) -> None:
    plan = _RUNNER._load_plan(_PLAN)

    assert _RUNNER._arm_image_max_pixels(plan, "pixel1003520") == 1003520
    assert _RUNNER._arm_image_max_pixels(plan, "pixel262144") == 262144
    first = _RUNNER._arm_paths_for_plan(plan, tmp_path, "pixel1003520")
    second = _RUNNER._arm_paths_for_plan(plan, tmp_path, "pixel262144")
    assert first["root"] != second["root"]
    assert first["config"] != second["config"]
    assert first["full_model_snapshot"] == second["full_model_snapshot"]
    assert first["full_model_receipt"] == second["full_model_receipt"]
    assert first["full_model_merge"] == second["full_model_merge"]


def test_materialized_resolution_pair_rejects_weight_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _payload()

    class FakeFullModelSnapshot:
        def __init__(self, *, weights: str) -> None:
            self.policy_version = SimpleNamespace(
                optimizer_step=80, weights_sha256=weights
            )
            self.manifest = SimpleNamespace(
                checkpoint_sha256="b" * 64,
                source_tree_sha256="c" * 64,
                source_path="/same/global_step_80",
                checkpoint_owner="owner",
            )

    configs = {
        name: tmp_path / f"{name}.json"
        for name in ("pixel1003520", "pixel262144")
    }
    config_values = {
        arm["name"]: SimpleNamespace(
            snapshot_backend=coredev.FULL_MODEL_EVALUATION_BACKEND,
            evaluation_protocol=(
                coredev.DEEPEYES_OFFICIAL_VISIBLE_EVALUATION_PROTOCOL
            ),
            expected_optimizer_step=80,
            evaluation_image_max_pixels=arm["evaluation_image_max_pixels"],
            paired_rng_protocol_projection=(
                coredev.IMAGE_MAX_PIXELS_RESOLUTION_PAIR_PROJECTION
            ),
        )
        for arm in plan["arms"]
    }
    snapshots = {
        "pixel1003520": FakeFullModelSnapshot(weights="a" * 64),
        "pixel262144": FakeFullModelSnapshot(weights="a" * 64),
    }
    by_path = {path: name for name, path in configs.items()}
    monkeypatch.setattr(_RUNNER, "FullModelEvaluationSnapshot", FakeFullModelSnapshot)
    monkeypatch.setattr(
        _RUNNER,
        "load_policy_coredev_config",
        lambda path: config_values[by_path[path]],
    )
    monkeypatch.setattr(
        _RUNNER,
        "load_policy_evaluation_snapshot",
        lambda config: snapshots[
            next(name for name, value in config_values.items() if value is config)
        ],
    )
    monkeypatch.setattr(
        _RUNNER,
        "paired_evaluation_rng_contract",
        lambda config, _snapshot, **_kwargs: {
            "arm_protocol_sha256": plan["paired_rng"]["arm_protocol_sha256"][
                next(name for name, value in config_values.items() if value is config)
            ],
            "seed_protocol_sha256": plan["paired_rng"]["seed_protocol_sha256"],
        },
    )

    _RUNNER._validate_materialized_resolution_pairing(plan, configs)
    snapshots["pixel262144"] = FakeFullModelSnapshot(weights="d" * 64)
    with pytest.raises(RuntimeError, match="share one checkpoint"):
        _RUNNER._validate_materialized_resolution_pairing(plan, configs)
