from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[2]
_TOOL = _ROOT / "tools/run_prl15_paired_evaluation.py"
_PLAN = _ROOT / "configs/evaluation/prl15_r1_rp66_step0_step8_coredev2511_plan.json"
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
        judge=judge,
        plan=plan,
    )
    assert score_command[score_command.index("--model") + 1] == (
        "Qwen3-VL-8B-Instruct"
    )
    assert score_command[score_command.index("--mode") + 1] == "eval"
    assert score_command[score_command.index("--judge") + 1] == (
        "Qwen2.5-72B-Instruct"
    )
    assert "--reuse" in score_command
    assert score_command[score_command.index("--reuse-aux") + 1] == "infer"


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

    result = _MODULE._materialize_official_scoring_view(
        tmp_path / "benchmark-config.json", plan, arm="step0"
    )

    expected_run_id = _MODULE._vlmevalkit_scoring_run_id(
        run_id_prefix=plan["scoring"]["run_id_prefix"],
        arm_evaluation_id=semantic_id,
    )
    assert captured["evaluation_id"] == semantic_id
    assert captured["run_id"] == expected_run_id
    assert result == {"evaluation_id": semantic_id, "run_id": expected_run_id}


def test_score_mode_reuses_existing_arms_without_policy_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    plan = {
        "evaluation_id": "PAIR",
        "policy_config": "configs/policy.toml",
        "scoring": {"view_name": "coredev"},
    }
    run = SimpleNamespace(output=SimpleNamespace(root=tmp_path / "training"))
    judge = {
        "devices": {"physical": [0, 1]},
        "server": {"base_url": "http://127.0.0.1:8012/v1"},
    }
    configs = {
        "step0": tmp_path / "evaluation/step0/benchmark-config.json",
        "step8": tmp_path / "evaluation/step8/benchmark-config.json",
    }
    forbidden: list[str] = []
    scoring_materialized: list[str] = []

    monkeypatch.setattr(_MODULE, "_load_plan", lambda _path: plan)
    monkeypatch.setattr(
        _MODULE, "load_policy_e2e_smoke_run_config", lambda *_args, **_kwargs: run
    )
    monkeypatch.setattr(_MODULE, "_validate_plan_run", lambda *_args: None)
    monkeypatch.setattr(_MODULE, "_load_judge_config", lambda _plan: judge)
    monkeypatch.setattr(
        _MODULE,
        "_load_existing_arm",
        lambda **kwargs: configs[kwargs["arm"]],
    )
    monkeypatch.setattr(
        _MODULE,
        "_load_existing_official_scoring_view",
        lambda _config, _plan, *, arm: {"materialized": arm},
    )
    monkeypatch.setattr(
        _MODULE,
        "_materialize_official_scoring_view",
        lambda _config, _plan, *, arm: scoring_materialized.append(arm),
    )
    monkeypatch.setattr(
        _MODULE,
        "_accepted_official_summary",
        lambda root, _judge: {"status": "pass", "root": str(root)},
    )
    monkeypatch.setattr(
        _MODULE,
        "_scoring_root",
        lambda config, _plan: tmp_path / "scoring" / config.parent.name,
    )
    monkeypatch.setattr(_MODULE, "write_json_atomic", lambda *_args: None)
    for name in ("_prepare", "_validate", "_materialize_arm", "_launch_workers"):
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
    assert scoring_materialized == ["step0", "step8"]


def test_missing_arm_scorers_launch_together_and_drain_before_failure(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    class Process:
        def __init__(self, name: str, code: int) -> None:
            self.name = name
            self.code = code

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
    pointer.write_bytes(b"pointer")
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
    assert calls["pair"]["rp66_pointer_path"] == calls["qwen"][
        "rp66_pointer_path"
    ]
