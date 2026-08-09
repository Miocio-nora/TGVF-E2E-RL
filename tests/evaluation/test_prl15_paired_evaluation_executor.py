from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
