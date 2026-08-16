from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_PLAN = _ROOT / "configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json"
_PRL17_PLAN = (
    _ROOT / "configs/evaluation/prl17_r0_frozen_rp66_step4_step8_coredev2511_plan.json"
)
_PRL21_PLAN = (
    _ROOT
    / "configs/evaluation/prl21_r0_crop_tfree_step8_step16_full_model_coredev2511_plan.json"
)
_SPEC = importlib.util.spec_from_file_location("prl15_paired_evaluation", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_r1_plan_and_judge_commands_are_fully_bound() -> None:
    plan = _MODULE._load_plan(_PLAN)
    judge = _MODULE._load_judge_config(plan)

    judge_command = _MODULE._judge_command(judge)
    assert judge_command[1:3] == ["serve", judge["model"]["local_path"]]
    assert judge_command[judge_command.index("--tensor-parallel-size") + 1] == "2"
    assert judge_command[judge_command.index("--seed") + 1] == "42"
    assert "--enable-prefix-caching" in judge_command

    score_command = _MODULE._official_score_command(
        dataset="MathVerse_MINI",
        scoring_root=Path("/tmp/prl15-score"),
        source_run_id="T20260810_Gdeadbeef",
        judge=judge,
        plan=plan,
    )
    assert score_command[score_command.index("--model") + 1] == ("Qwen3-VL-8B-Instruct")
    assert score_command[score_command.index("--mode") + 1] == "eval"
    assert score_command[score_command.index("--judge") + 1] == ("Qwen2.5-72B-Instruct")
    assert "--reuse" in score_command
    assert score_command[score_command.index("--reuse-aux") + 1] == "infer"
    assert (
        score_command[score_command.index("--tgvf-reuse-source-run-id") + 1]
        == "T20260810_Gdeadbeef"
    )
    assert Path(
        score_command[score_command.index("--tgvf-reuse-manifest") + 1]
    ) == Path(
        "/tmp/prl15-score/MathVerse_MINI/Qwen3-VL-8B-Instruct/"
        "T20260810_Gdeadbeef/final-answer-view-manifest.json"
    )


def test_vlmevalkit_scoring_run_id_is_legal_stable_and_arm_specific() -> None:
    semantic_id = "PRL15-R1-RP66-COREDEV2511-V1-STEP0"
    expected = "T20260810_G" + hashlib.sha256(semantic_id.encode("utf-8")).hexdigest()

    first = _MODULE._vlmevalkit_scoring_run_id(
        run_id_prefix="T20260810-PRL15-R1-RP66",
        arm_evaluation_id=semantic_id,
    )
    repeated = _MODULE._vlmevalkit_scoring_run_id(
        run_id_prefix="T20260810-PRL15-R1-RP66",
        arm_evaluation_id=semantic_id,
    )
    step8 = _MODULE._vlmevalkit_scoring_run_id(
        run_id_prefix="T20260810-PRL15-R1-RP66",
        arm_evaluation_id="PRL15-R1-RP66-COREDEV2511-V1-STEP8",
    )

    assert first == repeated == expected
    assert step8 != first
    assert re.fullmatch(r"T\d{8}_G[0-9a-f]{64}", first)


@pytest.mark.parametrize(
    "run_id_prefix",
    [
        "PRL15-R1-RP66",
        "T2026081-PRL15-R1-RP66",
        "T20260810",
        "T20261340-PRL15-R1-RP66",
    ],
)
def test_vlmevalkit_scoring_run_id_rejects_malformed_plan_prefix(
    run_id_prefix: str,
) -> None:
    with pytest.raises(ValueError, match="prefix"):
        _MODULE._vlmevalkit_scoring_run_id(
            run_id_prefix=run_id_prefix,
            arm_evaluation_id="PRL15-R1-RP66-COREDEV2511-V1-STEP0",
        )


def test_plan_loader_rejects_non_generatable_scoring_prefix(tmp_path: Path) -> None:
    payload = json.loads(_PLAN.read_text(encoding="utf-8"))
    payload["scoring"]["run_id_prefix"] = "T20260810"
    path = tmp_path / "invalid-plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run prefix"):
        _MODULE._load_plan(path)


def test_plan_loader_accepts_ordered_nonzero_checkpoint_pair() -> None:
    plan = _MODULE._load_plan(_PRL17_PLAN)

    assert [(arm["name"], arm["optimizer_step"]) for arm in plan["arms"]] == [
        ("step4", 4),
        ("step8", 8),
    ]


def test_v3_full_model_plan_separates_owner_protocol_and_preserves_arm_ids() -> None:
    plan = _MODULE._load_plan(_PRL21_PLAN)
    runtime = _MODULE._load_evaluation_runtime(plan)

    assert runtime.backend == _MODULE.FULL_MODEL_BACKEND
    assert runtime.checkpoint_owner.run_id == plan["checkpoint_owner"]["run_id"]
    assert runtime.protocol_contract.run_id == plan["protocol_contract"]["run_id"]
    assert runtime.checkpoint_owner.run_id != runtime.protocol_contract.run_id
    assert [_MODULE._arm_evaluation_id(plan, arm["name"]) for arm in plan["arms"]] == [
        "PRL21-R0-CROP-TFREE-COREDEV2511-STEP8-TEMP1-SEED42-V1",
        "PRL21-R0-CROP-TFREE-COREDEV2511-STEP16-TEMP1-SEED42-V1",
    ]
    assert plan["scoring"]["execution"] == {
        "mode": "eval",
        "reuse": True,
        "reuse_aux": "infer",
    }
    assert plan["paired_rng"]["seed_namespace"] == (
        "coredev2511-official-v1/crop-tfree/step8-step16/temp1/seed42/v1"
    )
    assert plan["paired_rng"]["protocol_sha256"] == (
        "03b5cbf38841ab5bb97200eef41234b2a33d064b283df696b1c6ffcce3c9e79d"
    )
    assert _MODULE._sampling_report(plan, runtime)["paired_rng"] == plan["paired_rng"]


def test_v3_runtime_rejects_paired_rng_task_seed_and_protocol_drift(
    tmp_path: Path,
) -> None:
    for field, wrong_value, error in (
        ("task_manifest_sha256", "0" * 64, "task manifest"),
        ("master_seed", 43, "task/seed"),
        ("protocol_sha256", "f" * 64, "protocol_sha256"),
    ):
        payload = json.loads(_PRL21_PLAN.read_text(encoding="utf-8"))
        payload["paired_rng"][field] = wrong_value
        path = tmp_path / f"invalid-{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        if field == "task_manifest_sha256":
            with pytest.raises(ValueError, match=error):
                _MODULE._load_plan(path)
        else:
            plan = _MODULE._load_plan(path)
            with pytest.raises(RuntimeError, match=error):
                _MODULE._load_evaluation_runtime(plan)


def test_v3_protocol_base_step_zero_is_not_an_owner_retained_checkpoint() -> None:
    plan = _MODULE._load_plan(_PRL21_PLAN)
    runtime = _MODULE._load_evaluation_runtime(plan)
    step_zero_plan = json.loads(json.dumps(plan))
    step_zero_plan["arms"] = [
        {
            "name": "step0",
            "evaluation_id": "FIXTURE-STEP0",
            "optimizer_step": 0,
            "source": {"kind": "protocol_base_model"},
        }
    ]

    _MODULE._validate_v3_runtime(
        step_zero_plan,
        runtime.checkpoint_owner,
        runtime.protocol_contract,
    )


def test_existing_paired_tgvf_arm_accepts_its_serialized_snapshot_backend(
    tmp_path: Path,
) -> None:
    plan = _MODULE._load_plan(_PLAN)
    runtime = SimpleNamespace(backend=_MODULE.PAIRED_TGVF_BACKEND)
    paths = _MODULE._arm_paths(tmp_path, "step8")
    values = {
        **_MODULE._expected_arm_runtime_settings(plan, runtime),
        "evaluation_id": _MODULE._arm_evaluation_id(plan, "step8"),
        "evaluation_protocol": plan["protocol"]["evaluation_protocol"],
        "output_root": paths["root"].resolve(),
        "gpu_ids": (0, 1, 2, 3),
        "task_manifest_path": _MODULE._resolve_repo_path(plan["task_manifest_path"]),
        "task_manifest_sha256": plan["task_manifest_sha256"],
        "expected_task_count": plan["expected_task_count"],
        "expected_single_image_count": plan["expected_single_image_count"],
        "paired_seed_namespace": _MODULE._paired_seed_namespace(plan),
    }

    assert _MODULE.PAIRED_TGVF_BACKEND == "paired_tgvf"
    assert values["snapshot_backend"] == "full_model_trainable_rp66"
    _MODULE._validate_existing_arm_config(
        SimpleNamespace(**values),
        plan=plan,
        runtime=runtime,
        paths=paths,
        arm="step8",
        gpu_ids=(0, 1, 2, 3),
    )


def test_existing_paired_tgvf_arm_rejects_runtime_selector_as_snapshot_backend(
    tmp_path: Path,
) -> None:
    plan = _MODULE._load_plan(_PLAN)
    runtime = SimpleNamespace(backend=_MODULE.PAIRED_TGVF_BACKEND)
    paths = _MODULE._arm_paths(tmp_path, "step8")
    values = {
        **_MODULE._expected_arm_runtime_settings(plan, runtime),
        "evaluation_id": _MODULE._arm_evaluation_id(plan, "step8"),
        "evaluation_protocol": plan["protocol"]["evaluation_protocol"],
        "output_root": paths["root"].resolve(),
        "gpu_ids": (0, 1, 2, 3),
        "task_manifest_path": _MODULE._resolve_repo_path(plan["task_manifest_path"]),
        "task_manifest_sha256": plan["task_manifest_sha256"],
        "expected_task_count": plan["expected_task_count"],
        "expected_single_image_count": plan["expected_single_image_count"],
        "paired_seed_namespace": _MODULE._paired_seed_namespace(plan),
        "snapshot_backend": _MODULE.PAIRED_TGVF_BACKEND,
    }

    with pytest.raises(RuntimeError, match="snapshot_backend"):
        _MODULE._validate_existing_arm_config(
            SimpleNamespace(**values),
            plan=plan,
            runtime=runtime,
            paths=paths,
            arm="step8",
            gpu_ids=(0, 1, 2, 3),
        )


def test_v3_owner_completion_tamper_fails_before_runtime_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(_PRL21_PLAN.read_text(encoding="utf-8"))
    payload["checkpoint_owner"]["completion_sha256"] = "0" * 64
    path = tmp_path / "invalid-v3-plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    launched: list[object] = []
    monkeypatch.setattr(
        _MODULE.subprocess,
        "Popen",
        lambda *_args, **_kwargs: launched.append(object()),
    )

    with pytest.raises(RuntimeError, match="completion identity"):
        _MODULE._load_plan(path)
    assert launched == []


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("gpu_ids", (0, 1, 2, 3)),
        ("gpu_memory_utilization", 0.9),
        ("evaluation_protocol", "training_run"),
        ("snapshot_backend", "paired_tgvf"),
        ("task_manifest_path", Path("/wrong/tasks.jsonl")),
        ("task_manifest_sha256", "0" * 64),
        ("expected_task_count", 1),
        ("expected_single_image_count", 1),
        ("max_model_len", 1),
        ("max_num_batched_tokens", 1),
        ("enable_chunked_prefill", True),
        ("inference_concurrency_per_gpu", 1),
        ("paired_seed_namespace", None),
    ],
)
def test_existing_full_model_arm_rejects_runtime_config_drift(
    tmp_path: Path, field: str, wrong_value: object
) -> None:
    plan = _MODULE._load_plan(_PRL21_PLAN)
    runtime = _MODULE._load_evaluation_runtime(plan)
    paths = _MODULE._arm_paths(tmp_path, "step8")
    values = {
        **_MODULE._expected_arm_runtime_settings(plan, runtime),
        "evaluation_id": _MODULE._arm_evaluation_id(plan, "step8"),
        "evaluation_protocol": plan["protocol"]["evaluation_protocol"],
        "output_root": paths["root"].resolve(),
        "gpu_ids": (4, 5, 6, 7),
        "task_manifest_path": _MODULE._resolve_repo_path(plan["task_manifest_path"]),
        "task_manifest_sha256": plan["task_manifest_sha256"],
        "expected_task_count": plan["expected_task_count"],
        "expected_single_image_count": plan["expected_single_image_count"],
        "paired_seed_namespace": _MODULE._paired_seed_namespace(plan),
    }
    values[field] = wrong_value

    with pytest.raises(RuntimeError, match=field):
        _MODULE._validate_existing_arm_config(
            SimpleNamespace(**values),
            plan=plan,
            runtime=runtime,
            paths=paths,
            arm="step8",
            gpu_ids=(4, 5, 6, 7),
        )


def test_score_mode_rejects_wait_flags_before_plan_or_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    monkeypatch.setattr(_MODULE, "_load_plan", lambda _path: touched.append("plan"))
    monkeypatch.setattr(
        _MODULE.subprocess,
        "Popen",
        lambda *_args, **_kwargs: touched.append("popen"),
    )
    monkeypatch.setattr(
        _MODULE.sys,
        "argv",
        [str(_TOOL), "--mode", "score", "--wait-for-gpus"],
    )

    with pytest.raises(ValueError, match="cannot wait"):
        _MODULE.main()
    assert touched == []


def test_scoring_materialization_keeps_semantic_and_legacy_ids_separate(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _MODULE._load_plan(_PLAN)
    semantic_id = f"{plan['evaluation_id']}-STEP0"
    config = SimpleNamespace(
        evaluation_id=semantic_id,
        output_root=tmp_path / "step0",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(_MODULE, "load_policy_coredev_config", lambda _: config)

    def materialize(**kwargs):
        captured.update(kwargs)
        return {"evaluation_id": kwargs["evaluation_id"], "run_id": kwargs["run_id"]}

    monkeypatch.setattr(
        _MODULE, "materialize_policy_coredev_scoring_views", materialize
    )
    monkeypatch.setattr(
        _MODULE,
        "_load_existing_official_scoring_view",
        lambda *_args, **_kwargs: {
            "evaluation_id": captured["evaluation_id"],
            "run_id": captured["run_id"],
        },
    )

    result = _MODULE._materialize_official_scoring_view(
        tmp_path / "benchmark-config.json", plan, arm="step0"
    )

    expected_run_id = _MODULE._vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=semantic_id,
    )
    assert captured["evaluation_id"] == semantic_id
    assert captured["run_id"] == expected_run_id
    assert Path(captured["logical_output_root"]) == (
        config.output_root / "scoring" / plan["scoring"]["view_name"]
    )
    assert result == {"evaluation_id": semantic_id, "run_id": expected_run_id}


def test_scoring_materialization_failure_never_publishes_partial_view(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _MODULE._load_plan(_PLAN)
    config = SimpleNamespace(
        evaluation_id=f"{plan['evaluation_id']}-STEP0",
        output_root=tmp_path / "step0",
    )
    monkeypatch.setattr(_MODULE, "load_policy_coredev_config", lambda _: config)

    def fail_after_partial_write(**kwargs):
        staging = Path(kwargs["output_root"])
        (staging / "raw").mkdir(parents=True)
        (staging / "raw/partial.tsv").write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted materialization")

    monkeypatch.setattr(
        _MODULE,
        "materialize_policy_coredev_scoring_views",
        fail_after_partial_write,
    )
    root = config.output_root / "scoring" / plan["scoring"]["view_name"]

    with pytest.raises(RuntimeError, match="interrupted"):
        _MODULE._materialize_official_scoring_view(
            tmp_path / "benchmark-config.json", plan, arm="step0"
        )

    assert not root.exists()
    assert not tuple(root.parent.glob(f".{root.name}.staging-*"))


def test_score_mode_reuses_existing_arms_without_policy_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    plan = {
        "evaluation_id": "PAIR",
        "policy_config": "configs/policy.toml",
        "scoring": {"view_name": "coredev"},
    }
    run = SimpleNamespace(output=SimpleNamespace(root=tmp_path / "training"))
    runtime = SimpleNamespace(
        backend=_MODULE.PAIRED_TGVF_BACKEND,
        checkpoint_owner=run,
        protocol_contract=run,
        output_root=run.output.root,
    )
    judge = {
        "devices": {"physical": [0, 1]},
        "server": {"base_url": "http://127.0.0.1:8012/v1"},
    }
    configs = {
        "step0": tmp_path / "evaluation/step0/benchmark-config.json",
        "step8": tmp_path / "evaluation/step8/benchmark-config.json",
    }
    loaded_gpu_ids: list[object] = []
    forbidden: list[str] = []

    monkeypatch.setattr(_MODULE, "_load_plan", lambda _path: plan)
    monkeypatch.setattr(_MODULE, "_load_evaluation_runtime", lambda _plan: runtime)
    monkeypatch.setattr(
        _MODULE,
        "_load_judge_config",
        lambda _plan, **_kwargs: judge,
    )

    def load_existing(**kwargs):
        loaded_gpu_ids.append(kwargs["gpu_ids"])
        return configs[kwargs["arm"]]

    monkeypatch.setattr(_MODULE, "_load_existing_arm", load_existing)
    monkeypatch.setattr(
        _MODULE,
        "_load_existing_official_scoring_view",
        lambda _config, _plan, *, arm: {"materialized": arm},
    )
    monkeypatch.setattr(
        _MODULE,
        "_accepted_scored_arm",
        lambda config, _plan, _judge: {
            "status": "pass",
            "root": str(tmp_path / "scoring" / config.parent.name),
        },
    )
    monkeypatch.setattr(
        _MODULE,
        "_scoring_root",
        lambda config, _plan: tmp_path / "scoring" / config.parent.name,
    )
    monkeypatch.setattr(_MODULE, "write_json_atomic", lambda *_args: None)
    monkeypatch.setattr(
        _MODULE, "_write_evaluation_complete", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        _MODULE,
        "_sampling_report",
        lambda *_args: {"source": "bound_policy_run_config"},
    )
    monkeypatch.setattr(
        _MODULE,
        "_identity_contract_report",
        lambda *_args: {"backend": "paired_tgvf"},
    )
    monkeypatch.setattr(
        _MODULE, "_arm_evaluation_identity_sha256", lambda _path: "a" * 64
    )
    for name in (
        "_prepare",
        "_validate",
        "_materialize_arm",
        "_materialize_official_scoring_view",
        "_launch_workers",
        "_wait_for_gpus",
    ):
        monkeypatch.setattr(
            _MODULE,
            name,
            lambda *_args, _name=name, **_kwargs: forbidden.append(_name),
        )
    monkeypatch.setattr(
        _MODULE.sys,
        "argv",
        [
            str(_TOOL),
            "--mode",
            "score",
            "--output-root",
            str(tmp_path / "evaluation"),
        ],
    )

    assert _MODULE.main() == 0
    assert forbidden == []
    assert loaded_gpu_ids == [None, None]


def test_v2_report_retains_exact_legacy_schema_shape(monkeypatch) -> None:
    plan = {
        "schema_version": _MODULE.PLAN_SCHEMA_V2,
        "evaluation_id": "LEGACY-PAIR",
        "arms": [
            {"name": "step0", "optimizer_step": 0},
            {"name": "step8", "optimizer_step": 8},
        ],
    }
    configs = {"step0": Path("step0.json"), "step8": Path("step8.json")}
    summaries = {"step0": {"score": 1}, "step8": {"score": 2}}
    monkeypatch.setattr(_MODULE, "_sampling_report", lambda *_args: {"temp": 1})
    monkeypatch.setattr(
        _MODULE, "_arm_evaluation_identity_sha256", lambda _path: "a" * 64
    )

    report = _MODULE._build_paired_report(
        plan=plan,
        runtime=SimpleNamespace(),
        configs=configs,
        materialization={"step0": {}, "step8": {}},
        official_summaries=summaries,
        arms=(("step0", 0), ("step8", 8)),
    )

    assert set(report) == {
        "schema_version",
        "evaluation_id",
        "coverage",
        "materialization",
        "sampling",
        "arms",
        "step0",
        "step8",
    }
    assert report["schema_version"] == _MODULE.PAIR_SUMMARY_SCHEMA
    assert set(report["arms"]["step0"]) == {
        "optimizer_step",
        "evaluation_identity_sha256",
        "official_summary",
    }


def test_legacy_summary_shape_does_not_gain_v3_headline(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "scoring"
    root.mkdir()
    (root / "coredev-2511-eval-summary.json").write_text("{}\n", encoding="utf-8")
    accepted = {"status": "pass", "sample_count": 2511}
    judge = {"server": {"base_url": "http://judge.invalid/v1"}}
    calls: list[str] = []
    monkeypatch.setattr(
        _MODULE, "summarize_coredev_results", lambda **_kwargs: dict(accepted)
    )
    monkeypatch.setattr(
        _MODULE,
        "extract_coredev_macro_star",
        lambda _summary: calls.append("headline") or {"macro_star_percent": 1.0},
    )

    assert _MODULE._accepted_official_summary(root, judge) == accepted
    assert calls == []
    with_headline = _MODULE._accepted_official_summary(
        root, judge, include_headline=True
    )
    assert with_headline == {
        **accepted,
        "headline": {"macro_star_percent": 1.0},
    }


def test_v3_report_exposes_owner_protocol_and_explicit_arm_ids(monkeypatch) -> None:
    plan = {
        "schema_version": _MODULE.PLAN_SCHEMA_V3,
        "evaluation_id": "GENERIC-PAIR",
        "arms": [
            {"name": "step8", "optimizer_step": 8, "evaluation_id": "ARM-8"},
            {"name": "step16", "optimizer_step": 16, "evaluation_id": "ARM-16"},
        ],
    }
    configs = {"step8": Path("step8.json"), "step16": Path("step16.json")}
    summaries = {"step8": {"score": 8}, "step16": {"score": 16}}
    monkeypatch.setattr(_MODULE, "_sampling_report", lambda *_args: {"temp": 1})
    monkeypatch.setattr(
        _MODULE,
        "_identity_contract_report",
        lambda *_args: {"checkpoint_owner": "owner", "protocol": "protocol"},
    )
    monkeypatch.setattr(
        _MODULE, "_arm_evaluation_identity_sha256", lambda _path: "b" * 64
    )

    report = _MODULE._build_paired_report(
        plan=plan,
        runtime=SimpleNamespace(),
        configs=configs,
        materialization={"step8": {}, "step16": {}},
        official_summaries=summaries,
        arms=(("step8", 8), ("step16", 16)),
    )

    assert report["schema_version"] == _MODULE.GENERIC_PAIR_SUMMARY_SCHEMA
    assert report["identity_contracts"] == {
        "checkpoint_owner": "owner",
        "protocol": "protocol",
    }
    assert report["arms"]["step8"]["evaluation_id"] == "ARM-8"
    assert report["arms"]["step16"]["evaluation_id"] == "ARM-16"


def test_evaluation_process_lock_rejects_duplicate_output(tmp_path: Path) -> None:
    output = tmp_path / "same-evaluation"
    assert not _MODULE._EVALUATION_LOCK_HANDLES
    _MODULE._acquire_evaluation_process_lock(output)
    try:
        with pytest.raises(RuntimeError, match="another paired evaluator"):
            _MODULE._acquire_evaluation_process_lock(output)
    finally:
        while _MODULE._EVALUATION_LOCK_HANDLES:
            handle = _MODULE._EVALUATION_LOCK_HANDLES.pop()
            _MODULE.fcntl.flock(handle.fileno(), _MODULE.fcntl.LOCK_UN)
            handle.close()


def test_missing_arm_scorers_launch_together_and_drain_before_failure(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    class Process:
        def __init__(self, name: str, code: int) -> None:
            self.name = name
            self.code = code
            self.pid = 1000 + len(events)

        def wait(self) -> int:
            events.append(f"wait:{self.name}")
            return self.code

    def launch(_config, _plan, _judge, *, arm, log_root):
        assert log_root == tmp_path
        events.append(f"launch:{arm}")
        code = 1 if arm == "step0" else 0
        return [(f"dataset-{arm}", Process(arm, code))]

    summarized: list[str] = []
    monkeypatch.setattr(_MODULE, "_launch_score_arm", launch)
    monkeypatch.setattr(_MODULE, "_worker_group_exists", lambda _process: False)
    monkeypatch.setattr(
        _MODULE,
        "_summarize_scored_arm",
        lambda config, _plan, _judge: summarized.append(config.name) or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="after all workers drained"):
        _MODULE._score_missing_arms(
            {"step0": Path("step0"), "step8": Path("step8")},
            {"step0": None, "step8": None},
            {},
            {},
            log_root=tmp_path,
        )

    assert events == [
        "launch:step0",
        "launch:step8",
        "wait:step0",
        "wait:step8",
    ]
    assert summarized == ["step8"]


def test_gpu_release_wait_requires_two_stable_free_polls(monkeypatch) -> None:
    readings = iter(
        (
            {index: (3000 if index == 7 else 0) for index in range(8)},
            {index: 0 for index in range(8)},
            {index: 0 for index in range(8)},
        )
    )
    sleeps: list[int] = []
    monkeypatch.setattr(_MODULE, "_gpu_memory_mib", lambda: next(readings))
    monkeypatch.setattr(_MODULE.time, "sleep", sleeps.append)

    _MODULE._wait_for_gpus(
        tuple(range(8)),
        timeout_seconds=60,
        poll_seconds=3,
        free_threshold_mib=1024,
    )

    assert sleeps == [3, 3]


def test_policy_workers_launch_in_isolated_process_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(gpu_ids=(0, 1, 2, 3), output_root=tmp_path)
    launches: list[dict[str, object]] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def popen(command, **kwargs):
        launches.append({"command": command, **kwargs})
        return Process(1000 + len(launches))

    monkeypatch.setattr(_MODULE, "load_policy_coredev_config", lambda _: config)
    monkeypatch.setattr(_MODULE.subprocess, "Popen", popen)

    processes = _MODULE._launch_workers(tmp_path / "benchmark-config.json")

    assert len(processes) == len(launches) == 4
    assert all(launch["start_new_session"] is True for launch in launches)
    assert [
        launch["command"][launch["command"].index("--rank") + 1] for launch in launches
    ] == ["0", "1", "2", "3"]
    for process in processes:
        _MODULE._ACTIVE_PROCESS_GROUPS.pop(process.pid, None)


def test_spawn_defers_interrupt_until_child_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        pid = 987654

    process = Process()

    def popen(command, **kwargs):
        events.append(f"spawn:{command[0]}:{kwargs['start_new_session']}")
        _MODULE._controlled_interrupt(_MODULE.signal.SIGTERM, None)
        assert _MODULE._DEFERRED_INTERRUPT_SIGNAL == _MODULE.signal.SIGTERM
        return process

    monkeypatch.setattr(_MODULE.subprocess, "Popen", popen)

    with pytest.raises(_MODULE._EvaluationInterrupted):
        _MODULE._spawn_registered_process(["child"], env={})

    assert _MODULE._ACTIVE_PROCESS_GROUPS[process.pid] is process
    assert events == [
        "spawn:child:True",
    ]
    assert _MODULE._SPAWN_CRITICAL_SECTION_ACTIVE is False
    assert _MODULE._DEFERRED_INTERRUPT_SIGNAL is None
    _MODULE._ACTIVE_PROCESS_GROUPS.pop(process.pid, None)


def test_spawned_child_does_not_inherit_blocked_termination_signals() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import signal; "
            "blocked=signal.pthread_sigmask(signal.SIG_BLOCK, []); "
            "print(','.join(str(int(item)) for item in sorted(blocked)))"
        ),
    ]

    process = _MODULE._spawn_registered_process(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=10)
    _MODULE._ACTIVE_PROCESS_GROUPS.pop(process.pid, None)

    assert process.returncode == 0, stderr
    blocked = {int(item) for item in stdout.strip().split(",") if item}
    assert blocked.isdisjoint(_MODULE._CHILD_SPAWN_SIGNALS)


def test_controlled_interrupt_is_not_swallowed_by_judge_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(poll=lambda: None)
    judge = {
        "server": {"base_url": "http://judge.invalid/v1"},
        "model": {"served_name": "judge"},
    }
    sleeps: list[int] = []
    monkeypatch.setattr(
        _MODULE,
        "check_qwen25_72b_judge",
        lambda **_kwargs: (_ for _ in ()).throw(_MODULE._EvaluationInterrupted()),
    )
    monkeypatch.setattr(_MODULE.time, "sleep", sleeps.append)

    with pytest.raises(_MODULE._EvaluationInterrupted):
        _MODULE._wait_for_judge(
            process,
            judge,
            timeout_seconds=60,
            poll_seconds=5,
        )

    assert sleeps == []


def test_cleanup_blocks_repeat_signals_until_children_and_handlers_are_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    process = SimpleNamespace(pid=24680)
    _MODULE._ACTIVE_PROCESS_GROUPS[process.pid] = process

    def pthread_sigmask(operation, mask):
        if operation == _MODULE.signal.SIG_BLOCK:
            events.append("signals-blocked")
            return {_MODULE.signal.SIGUSR1}
        assert operation == _MODULE.signal.SIG_SETMASK
        assert mask == {_MODULE.signal.SIGUSR1}
        events.append("signals-restored")
        return set()

    def terminate(processes):
        assert events == ["signals-blocked"]
        assert processes == [process]
        events.append("children-drained")
        _MODULE._ACTIVE_PROCESS_GROUPS.clear()

    monkeypatch.setattr(_MODULE.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(_MODULE, "_terminate_worker_groups", terminate)
    monkeypatch.setattr(
        _MODULE.signal,
        "signal",
        lambda candidate, previous: events.append(
            f"handler-restored:{candidate.name}:{previous}"
        ),
    )

    _MODULE._cleanup_evaluation_runtime({_MODULE.signal.SIGTERM: "old"})

    assert events == [
        "signals-blocked",
        "children-drained",
        "handler-restored:SIGTERM:old",
        "signals-restored",
    ]


def test_failed_policy_worker_drains_all_isolated_groups_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, code: int | None) -> None:
            self.code = code

        def poll(self) -> int | None:
            return self.code

        def wait(self) -> int:
            assert self.code is not None
            return self.code

    processes = [Process(1), Process(None)]
    drained: list[object] = []
    monkeypatch.setattr(
        _MODULE,
        "_terminate_worker_groups",
        lambda observed: drained.extend(observed),
    )

    with pytest.raises(RuntimeError, match="worker 0 exited with 1"):
        _MODULE._wait_workers(processes, owner="step8 + step16")

    assert drained == processes


def test_already_exited_failed_workers_still_drain_orphan_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, code: int) -> None:
            self.code = code

        def poll(self) -> int:
            return self.code

        def wait(self) -> int:
            return self.code

    processes = [Process(0), Process(1)]
    drained: list[object] = []
    monkeypatch.setattr(
        _MODULE,
        "_terminate_worker_groups",
        lambda observed: drained.extend(observed),
    )

    with pytest.raises(RuntimeError, match=r"workers failed: \[0, 1\]"):
        _MODULE._wait_workers(processes)

    assert drained == processes


def test_step8_wait_binds_complete_fsdp_shards_without_assuming_embedded_hf(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "global_step_8"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "huggingface").mkdir()
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "8", encoding="utf-8"
    )
    for path in (
        checkpoint / "data.pt",
        actor / "fsdp_config.json",
        actor / "huggingface/config.json",
        actor / "tgvf_policy_checkpoint_pair.json",
        actor / "tgvf_policy_project_state.json",
        *(actor / f"model_world_size_4_rank_{rank}.pt" for rank in range(4)),
    ):
        path.write_bytes(b"complete")
    output_root = tmp_path / "output"
    pointer = output_root / "runtime-policy-state/latest-lora-snapshot.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"optimizer_step": 8}\n', encoding="utf-8")
    run = SimpleNamespace(
        output=SimpleNamespace(
            checkpoint_directory=checkpoint_root,
            root=output_root,
        ),
        distributed=SimpleNamespace(world_size=4),
    )

    _MODULE._wait_for_step8(run, timeout_seconds=1, poll_seconds=1)

    assert not (actor / "huggingface/model.safetensors.index.json").exists()


def test_step8_arm_materializes_qwen_only_before_pairing(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _MODULE._load_plan(_PLAN)
    output_root = tmp_path / "training"
    checkpoint_root = output_root / "checkpoints"
    run = SimpleNamespace(
        output=SimpleNamespace(
            checkpoint_directory=checkpoint_root,
            root=output_root,
        )
    )
    calls = {}
    qwen_model = tmp_path / "qwen-only/model"

    def materialize_qwen(**kwargs):
        calls["qwen"] = kwargs
        return qwen_model

    def materialize_pair(**kwargs):
        calls["pair"] = kwargs
        return {}

    monkeypatch.setattr(
        _MODULE, "materialize_qwen_only_policy_checkpoint", materialize_qwen
    )
    monkeypatch.setattr(
        _MODULE, "materialize_paired_tgvf_policy_benchmark_config", materialize_pair
    )
    arm_root = tmp_path / "evaluation"

    config = _MODULE._materialize_arm(
        plan=plan,
        run=run,
        arm="step8",
        step=8,
        output_base=arm_root,
        gpu_ids=(0, 1, 2, 3),
    )

    assert config == arm_root / "step8/benchmark-config.json"
    assert calls["qwen"]["checkpoint_path"] == checkpoint_root / "global_step_8"
    assert calls["qwen"]["rp66_pointer_path"] == (
        output_root / "runtime-policy-state/latest-lora-snapshot.json"
    )
    assert calls["pair"]["qwen_model_path"] == qwen_model
    assert calls["pair"]["rp66_pointer_path"] == calls["qwen"]["rp66_pointer_path"]


def test_nonzero_step_source_falls_back_to_permanent_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "training"
    checkpoint_root = output_root / "checkpoints"
    permanent = output_root / "permanent-checkpoints/global_step_4"
    permanent.mkdir(parents=True)
    pointer = tmp_path / "step4-pointer.json"
    run = SimpleNamespace(
        output=SimpleNamespace(
            checkpoint_directory=checkpoint_root,
            root=output_root,
        )
    )
    monkeypatch.setattr(
        _MODULE,
        "_materialize_step_pointer",
        lambda *_args, **_kwargs: pointer,
    )

    checkpoint, observed_pointer = _MODULE._step_sources(
        run,
        step=4,
        output_base=tmp_path / "evaluation",
        arm="step4",
    )

    assert checkpoint == permanent
    assert observed_pointer == pointer
