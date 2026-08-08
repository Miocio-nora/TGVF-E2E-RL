from __future__ import annotations

from pathlib import Path

from tgvf_rl.policy.deepeyes_native_controller import (
    DEEPEYES_NATIVE_EVAL_DATASETS,
    CheckpointReceipt,
    ControllerAction,
    ControllerActionKind,
    DeepEyesNativeControllerState,
    EvaluationReceipt,
    load_controller_state,
    run_deepeyes_native_controller,
    save_controller_state,
)


def _evaluation(step: int, weights: str) -> EvaluationReceipt:
    return EvaluationReceipt(
        step=step,
        checkpoint_weights_sha256=weights,
        results={
            "T1-PROBE256": {
                source: {"accuracy": 0.5, "tool_rate": 0.2, "crop_rate": 0.1}
                for source in ("vstar", "arxivqa", "thinklite")
            },
            "DeepEyesDev591": {"accuracy": 0.5},
            "Grounding-200": {"best_iou": 0.5},
            "CoreDev2511": {"accuracy": 0.5},
        },
    )


def test_controller_evaluates_base_then_saves_before_every_checkpoint_eval() -> None:
    assert DEEPEYES_NATIVE_EVAL_DATASETS == (
        "T1-PROBE256",
        "DeepEyesDev591",
        "Grounding-200",
        "CoreDev2511",
    )
    state = DeepEyesNativeControllerState(
        run_id="run",
        base_model_path="/models/base",
        base_model_weights_sha256="0" * 64,
    )
    calls: list[tuple[object, ...]] = []

    def save(step: int) -> CheckpointReceipt:
        calls.append(("save", step))
        return CheckpointReceipt(step, f"/checkpoints/{step}", str(step)[-1] * 64)

    def evaluate(step: int, path: str, weights: str) -> EvaluationReceipt:
        calls.append(("evaluate", step, path, weights))
        return _evaluation(step, weights)

    def train(start: int, end: int) -> None:
        calls.append(("train", start, end))

    completed = run_deepeyes_native_controller(
        state,
        save_checkpoint=save,
        evaluate_saved_checkpoint=evaluate,
        train_to_step=train,
    )
    assert completed.current_step == 80
    assert set(completed.checkpoints) == {1, 8, 20, 45, 80}
    assert set(completed.evaluations) == {0, 8, 20, 45, 80}
    assert calls[0] == ("evaluate", 0, "/models/base", "0" * 64)
    for gate in (8, 20, 45, 80):
        save_index = calls.index(("save", gate))
        eval_index = next(
            index
            for index, call in enumerate(calls)
            if call[:2] == ("evaluate", gate)
        )
        assert save_index < eval_index
    assert not any(call[:2] == ("evaluate", 1) for call in calls)


def test_saved_gate_resume_evaluates_without_retraining_or_resaving(tmp_path: Path) -> None:
    state = DeepEyesNativeControllerState(
        run_id="run",
        base_model_path="/models/base",
        base_model_weights_sha256="0" * 64,
        current_step=8,
        checkpoints={8: CheckpointReceipt(8, "/checkpoints/8", "8" * 64)},
        events=[
            {"kind": "training_completed", "from_step": 1, "step": 8},
            {"kind": "checkpoint_saved", "step": 8},
        ],
    )
    assert state.next_action() == ControllerAction(ControllerActionKind.EVALUATE, 8)
    path = tmp_path / "state.json"
    save_controller_state(path, state)
    loaded = load_controller_state(
        path,
        run_id="run",
        base_model_path="/models/base",
        base_model_weights_sha256="0" * 64,
    )
    assert loaded.next_action() == ControllerAction(ControllerActionKind.EVALUATE, 8)
