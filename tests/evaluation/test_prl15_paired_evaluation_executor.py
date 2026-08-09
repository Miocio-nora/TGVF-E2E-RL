from __future__ import annotations

import importlib.util
from pathlib import Path


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
