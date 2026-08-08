"""Resume-safe save-before-evaluate controller for the PRL13 gate sequence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .deepeyes_native_contract import (
    DEEPEYES_NATIVE_CHECKPOINT_GATES,
    DEEPEYES_NATIVE_EVALUATION_GATES,
)


DEEPEYES_NATIVE_CONTROLLER_SCHEMA = "tgvf.deepeyes-native-controller.v1"
DEEPEYES_NATIVE_EVAL_DATASETS = (
    "T1-PROBE256",
    "DeepEyesDev591",
    "Grounding-200",
    "CoreDev2511",
)
_T1_PROBE_SOURCES = ("vstar", "arxivqa", "thinklite")
_T1_PROBE_METRICS = ("accuracy", "tool_rate", "crop_rate")


class ControllerActionKind(str, Enum):
    SAVE = "save"
    EVALUATE = "evaluate"
    TRAIN_TO = "train_to"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class ControllerAction:
    kind: ControllerActionKind
    step: int


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    step: int
    path: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if self.step not in DEEPEYES_NATIVE_CHECKPOINT_GATES:
            raise ValueError("checkpoint receipt step is not a PRL13 gate")
        if not self.path or len(self.weights_sha256) != 64:
            raise ValueError("checkpoint receipt identity differs")

    def as_record(self) -> dict[str, object]:
        return {
            "step": self.step,
            "path": self.path,
            "weights_sha256": self.weights_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    step: int
    checkpoint_weights_sha256: str
    results: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if self.step not in DEEPEYES_NATIVE_EVALUATION_GATES:
            raise ValueError("evaluation receipt step is not a PRL13 gate")
        if len(self.checkpoint_weights_sha256) != 64:
            raise ValueError("evaluation checkpoint identity differs")
        if set(self.results) != set(DEEPEYES_NATIVE_EVAL_DATASETS):
            raise ValueError("PRL13 evaluation receipt dataset set differs")
        probe = self.results["T1-PROBE256"]
        if not isinstance(probe, Mapping) or set(probe) != set(_T1_PROBE_SOURCES):
            raise ValueError("T1-PROBE256 must report all three sources")
        for source in _T1_PROBE_SOURCES:
            metrics = probe[source]
            if not isinstance(metrics, Mapping) or not set(_T1_PROBE_METRICS) <= set(
                metrics
            ):
                raise ValueError(
                    "T1-PROBE256 source metrics require accuracy/tool_rate/crop_rate"
                )
        for dataset in DEEPEYES_NATIVE_EVAL_DATASETS[1:]:
            metrics = self.results[dataset]
            if not isinstance(metrics, Mapping) or not metrics:
                raise ValueError(f"{dataset} evaluation metrics are empty")

    def as_record(self) -> dict[str, object]:
        return {
            "step": self.step,
            "checkpoint_weights_sha256": self.checkpoint_weights_sha256,
            "results": json.loads(_canonical_json(self.results)),
        }


@dataclass(slots=True)
class DeepEyesNativeControllerState:
    run_id: str
    base_model_path: str
    base_model_weights_sha256: str
    current_step: int = 0
    checkpoints: dict[int, CheckpointReceipt] = field(default_factory=dict)
    evaluations: dict[int, EvaluationReceipt] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    schema_version: str = DEEPEYES_NATIVE_CONTROLLER_SCHEMA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("controller run_id must be non-empty")
        if not self.base_model_path or len(self.base_model_weights_sha256) != 64:
            raise ValueError("controller base-model identity differs")
        if self.schema_version != DEEPEYES_NATIVE_CONTROLLER_SCHEMA:
            raise ValueError("controller state schema differs")
        if type(self.current_step) is not int or not 0 <= self.current_step <= 80:
            raise ValueError("controller current_step is outside [0,80]")
        for step, checkpoint in self.checkpoints.items():
            if step != checkpoint.step or step > self.current_step:
                raise ValueError("controller checkpoint map differs")
        for step, evaluation in self.evaluations.items():
            checkpoint = self.checkpoints.get(step)
            expected_weights = (
                self.base_model_weights_sha256
                if step == 0
                else checkpoint.weights_sha256 if checkpoint is not None else None
            )
            if step != evaluation.step or (
                expected_weights != evaluation.checkpoint_weights_sha256
            ):
                raise ValueError("evaluation is not bound to a saved checkpoint")
        self._assert_event_order()

    def next_action(self) -> ControllerAction:
        step = self.current_step
        if (
            step in DEEPEYES_NATIVE_CHECKPOINT_GATES
            and step not in self.checkpoints
        ):
            return ControllerAction(ControllerActionKind.SAVE, step)
        if step in DEEPEYES_NATIVE_EVALUATION_GATES and step not in self.evaluations:
            return ControllerAction(ControllerActionKind.EVALUATE, step)
        if step == DEEPEYES_NATIVE_EVALUATION_GATES[-1]:
            return ControllerAction(ControllerActionKind.COMPLETE, step)
        next_gate = next(
            gate
            for gate in sorted(
                set(DEEPEYES_NATIVE_CHECKPOINT_GATES)
                | set(DEEPEYES_NATIVE_EVALUATION_GATES)
            )
            if gate > step
        )
        return ControllerAction(ControllerActionKind.TRAIN_TO, next_gate)

    def record_checkpoint(self, receipt: CheckpointReceipt) -> None:
        if self.next_action() != ControllerAction(
            ControllerActionKind.SAVE, receipt.step
        ):
            raise RuntimeError("checkpoint was not the controller's next action")
        self.checkpoints[receipt.step] = receipt
        self.events.append({"kind": "checkpoint_saved", "step": receipt.step})

    def record_evaluation(self, receipt: EvaluationReceipt) -> None:
        if self.next_action() != ControllerAction(
            ControllerActionKind.EVALUATE, receipt.step
        ):
            raise RuntimeError("evaluation was not the controller's next action")
        expected_weights = (
            self.base_model_weights_sha256
            if receipt.step == 0
            else self.checkpoints[receipt.step].weights_sha256
        )
        if expected_weights != receipt.checkpoint_weights_sha256:
            raise ValueError("evaluation weights differ from saved checkpoint")
        self.evaluations[receipt.step] = receipt
        self.events.append({"kind": "evaluation_completed", "step": receipt.step})

    def record_training(self, target_step: int) -> None:
        if self.next_action() != ControllerAction(
            ControllerActionKind.TRAIN_TO, target_step
        ):
            raise RuntimeError("training target was not the controller's next action")
        prior = self.current_step
        self.current_step = target_step
        self.events.append(
            {"kind": "training_completed", "from_step": prior, "step": target_step}
        )

    def _assert_event_order(self) -> None:
        saved: set[int] = set()
        for event in self.events:
            kind = event.get("kind")
            step = event.get("step")
            if kind == "checkpoint_saved":
                if type(step) is not int:
                    raise ValueError("checkpoint event step differs")
                saved.add(step)
            elif kind == "evaluation_completed":
                if step != 0 and step not in saved:
                    raise ValueError("evaluation event appears before checkpoint save")
            elif kind != "training_completed":
                raise ValueError("controller event kind differs")

    def as_record(self) -> dict[str, object]:
        record = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "base_model_path": self.base_model_path,
            "base_model_weights_sha256": self.base_model_weights_sha256,
            "current_step": self.current_step,
            "checkpoints": {
                str(step): receipt.as_record()
                for step, receipt in sorted(self.checkpoints.items())
            },
            "evaluations": {
                str(step): receipt.as_record()
                for step, receipt in sorted(self.evaluations.items())
            },
            "events": list(self.events),
        }
        return {**record, "state_sha256": _sha256_json(record)}


def run_deepeyes_native_controller(
    state: DeepEyesNativeControllerState,
    *,
    save_checkpoint: Callable[[int], CheckpointReceipt],
    evaluate_saved_checkpoint: Callable[[int, str, str], EvaluationReceipt],
    train_to_step: Callable[[int, int], None],
    persist_state: Callable[[DeepEyesNativeControllerState], None] | None = None,
) -> DeepEyesNativeControllerState:
    """Drive all gates; persist after every completed side effect.

    A resumed state whose checkpoint was saved immediately proceeds to that
    checkpoint's evaluation.  A resumed state without the checkpoint saves it
    first.  Step 0 is the sole exception: it evaluates the immutable bound base
    model directly and is not mislabelled as a checkpoint.
    """

    while True:
        action = state.next_action()
        if action.kind is ControllerActionKind.COMPLETE:
            return state
        if action.kind is ControllerActionKind.SAVE:
            state.record_checkpoint(save_checkpoint(action.step))
        elif action.kind is ControllerActionKind.EVALUATE:
            if action.step == 0:
                subject_path = state.base_model_path
                subject_sha256 = state.base_model_weights_sha256
            else:
                checkpoint = state.checkpoints[action.step]
                subject_path = checkpoint.path
                subject_sha256 = checkpoint.weights_sha256
            state.record_evaluation(
                evaluate_saved_checkpoint(action.step, subject_path, subject_sha256)
            )
        elif action.kind is ControllerActionKind.TRAIN_TO:
            train_to_step(state.current_step, action.step)
            state.record_training(action.step)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise RuntimeError(f"unsupported controller action {action.kind}")
        if persist_state is not None:
            persist_state(state)


def save_controller_state(path: str | Path, state: DeepEyesNativeControllerState) -> None:
    """Atomically persist one state record for crash-safe resume."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(state.as_record()) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_controller_state(
    path: str | Path,
    *,
    run_id: str,
    base_model_path: str,
    base_model_weights_sha256: str,
) -> DeepEyesNativeControllerState:
    source = Path(path)
    if not source.exists():
        return DeepEyesNativeControllerState(
            run_id=run_id,
            base_model_path=base_model_path,
            base_model_weights_sha256=base_model_weights_sha256,
        )
    if source.is_symlink() or not source.is_file():
        raise ValueError("controller state must be a regular non-symlink file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("controller state must be an object")
    identity = dict(value)
    state_sha256 = identity.pop("state_sha256", None)
    if state_sha256 != _sha256_json(identity):
        raise ValueError("controller state SHA-256 differs")
    if identity.get("run_id") != run_id:
        raise ValueError("controller resume run_id differs")
    if (
        identity.get("base_model_path") != base_model_path
        or identity.get("base_model_weights_sha256")
        != base_model_weights_sha256
    ):
        raise ValueError("controller resume base-model identity differs")
    checkpoints = {
        int(step): CheckpointReceipt(**receipt)
        for step, receipt in identity["checkpoints"].items()
    }
    evaluations = {
        int(step): EvaluationReceipt(**receipt)
        for step, receipt in identity["evaluations"].items()
    }
    return DeepEyesNativeControllerState(
        run_id=run_id,
        base_model_path=base_model_path,
        base_model_weights_sha256=base_model_weights_sha256,
        current_step=identity["current_step"],
        checkpoints=checkpoints,
        evaluations=evaluations,
        events=identity["events"],
        schema_version=identity["schema_version"],
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "DEEPEYES_NATIVE_CONTROLLER_SCHEMA",
    "DEEPEYES_NATIVE_EVAL_DATASETS",
    "CheckpointReceipt",
    "ControllerAction",
    "ControllerActionKind",
    "DeepEyesNativeControllerState",
    "EvaluationReceipt",
    "load_controller_state",
    "run_deepeyes_native_controller",
    "save_controller_state",
]
