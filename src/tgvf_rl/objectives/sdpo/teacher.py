"""Feedback-conditioned teacher context and checkpointable teacher state."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayHandle,
)
from tgvf_rl.qwen.base import (
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
)

from .schema import (
    TeacherContextOverflow,
    TeacherContextRecord,
    TeacherContextSpec,
    TeacherControllerSpec,
    TeacherReplayArtifact,
    TeacherRegularization,
    TeacherTurnAlignment,
    _tensor_sha256,
)


def parameter_mapping_sha256(parameters: Mapping[str, torch.Tensor]) -> str:
    """Content hash including each parameter's name, dtype, shape, and bytes."""

    normalized = _validated_parameter_mapping(parameters, expected_dtype=None)
    digest = hashlib.sha256()
    digest.update(b"tgvf-sdpo-parameter-mapping-v1\0")
    for name in sorted(normalized):
        tensor = normalized[name]
        cpu = tensor.detach().to(device="cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        metadata = (
            f"{name}\0{str(cpu.dtype).removeprefix('torch.')}\0"
            f"{','.join(str(dimension) for dimension in cpu.shape)}\0{len(raw)}\0"
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(raw)
    return digest.hexdigest()


class TeacherController:
    """Stateful EMA or trust-region SDPO teacher with strict checkpoint replay."""

    _SCHEMA_VERSION = "sdpo-teacher-controller-v1"

    def __init__(
        self,
        spec: TeacherControllerSpec,
        initial_parameters: Mapping[str, torch.Tensor],
        initial_policy_version: PolicyVersion,
    ) -> None:
        normalized = _validated_parameter_mapping(
            initial_parameters,
            expected_dtype=spec.parameter_dtype,
        )
        actual_identity = parameter_mapping_sha256(normalized)
        if actual_identity != spec.initial_teacher_sha256:
            raise IdentityMismatchError(
                "initial teacher parameters do not match the controller spec"
            )
        self.spec = spec
        self._initial_parameters = _clone_mapping(normalized)
        self._teacher_parameters = _clone_mapping(normalized)
        self._initial_policy_version = initial_policy_version
        self._last_policy_version = initial_policy_version
        self._update_calls = 0
        self._update_count = 0

    @property
    def update_calls(self) -> int:
        return self._update_calls

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def last_policy_version(self) -> PolicyVersion:
        return self._last_policy_version

    @property
    def state_identity_sha256(self) -> str:
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "spec_identity_sha256": self.spec.identity_sha256,
            "initial_policy_version": _policy_version_state(
                self._initial_policy_version
            ),
            "last_policy_version": _policy_version_state(self._last_policy_version),
            "update_calls": self._update_calls,
            "update_count": self._update_count,
            "initial_parameters_sha256": parameter_mapping_sha256(
                self._initial_parameters
            ),
            "teacher_parameters_sha256": parameter_mapping_sha256(
                self._teacher_parameters
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def teacher_parameters(self) -> dict[str, torch.Tensor]:
        """Return clones so external code cannot mutate checkpointed teacher state."""

        return _clone_mapping(self._teacher_parameters)

    def update(
        self,
        student_parameters: Mapping[str, torch.Tensor],
        student_policy_version: PolicyVersion,
    ) -> bool:
        """Observe a newer student and apply a scheduled controller update."""

        if student_policy_version.run_id != self._last_policy_version.run_id:
            raise IdentityMismatchError(
                "teacher updates cannot switch policy run identity"
            )
        if (
            student_policy_version.optimizer_step
            <= self._last_policy_version.optimizer_step
        ):
            raise IdentityMismatchError(
                "teacher update requires a strictly newer policy version"
            )
        normalized = _validated_parameter_mapping(
            student_parameters,
            expected_dtype=self.spec.parameter_dtype,
        )
        _validate_parameter_layout(self._teacher_parameters, normalized)
        actual_student_sha256 = parameter_mapping_sha256(normalized)
        if actual_student_sha256 != student_policy_version.weights_sha256:
            raise IdentityMismatchError(
                "student policy version does not identify the supplied parameters"
            )

        self._update_calls += 1
        self._last_policy_version = student_policy_version
        if self._update_calls % self.spec.update_interval != 0:
            return False

        if self.spec.regularization is TeacherRegularization.EMA_PARAMETERS:
            weight = self.spec.student_weight
            with torch.no_grad():
                for name, teacher_parameter in self._teacher_parameters.items():
                    student_parameter = (
                        normalized[name].detach().to(device=teacher_parameter.device)
                    )
                    teacher_parameter.mul_(1.0 - weight).add_(
                        student_parameter, alpha=weight
                    )
        elif (
            self.spec.regularization
            is not TeacherRegularization.TRUST_REGION_LOGIT_INTERPOLATION
        ):
            raise ValueError(
                f"unknown teacher regularization: {self.spec.regularization!r}"
            )
        # Trust-region mode retains the frozen initial parameter state; its
        # feedback-conditioned teacher is materialized by interpolate_logits.
        self._update_count += 1
        return True

    def interpolate_logits(
        self,
        frozen_reference_logits: torch.Tensor,
        feedback_conditioned_student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize the no-grad trust-region teacher distribution logits."""

        if (
            self.spec.regularization
            is not TeacherRegularization.TRUST_REGION_LOGIT_INTERPOLATION
        ):
            raise ValueError(
                "logit interpolation is available only in trust-region mode"
            )
        if frozen_reference_logits.shape != feedback_conditioned_student_logits.shape:
            raise ReplayMismatchError("trust-region logit shape mismatch")
        if frozen_reference_logits.device != feedback_conditioned_student_logits.device:
            raise ReplayMismatchError("trust-region logits must share a device")
        if frozen_reference_logits.dtype != feedback_conditioned_student_logits.dtype:
            raise ReplayMismatchError("trust-region logits must share a dtype")
        if not frozen_reference_logits.dtype.is_floating_point:
            raise TypeError("trust-region logits must use a floating dtype")
        if not bool(
            torch.isfinite(frozen_reference_logits.detach()).all().item()
        ) or not bool(
            torch.isfinite(feedback_conditioned_student_logits.detach()).all().item()
        ):
            raise ValueError("trust-region logits must be finite")
        weight = self.spec.student_weight
        return (
            (1.0 - weight) * frozen_reference_logits.detach()
            + weight * feedback_conditioned_student_logits.detach()
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "spec_identity_sha256": self.spec.identity_sha256,
            "regularization": self.spec.regularization.value,
            "initial_policy_version": _policy_version_state(
                self._initial_policy_version
            ),
            "last_policy_version": _policy_version_state(self._last_policy_version),
            "update_calls": self._update_calls,
            "update_count": self._update_count,
            "initial_parameters": _clone_mapping(
                self._initial_parameters, device="cpu"
            ),
            "teacher_parameters": _clone_mapping(
                self._teacher_parameters, device="cpu"
            ),
            "initial_parameters_sha256": parameter_mapping_sha256(
                self._initial_parameters
            ),
            "teacher_parameters_sha256": parameter_mapping_sha256(
                self._teacher_parameters
            ),
            "controller_state_sha256": self.state_identity_sha256,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required_keys = {
            "schema_version",
            "spec_identity_sha256",
            "regularization",
            "initial_policy_version",
            "last_policy_version",
            "update_calls",
            "update_count",
            "initial_parameters",
            "teacher_parameters",
            "initial_parameters_sha256",
            "teacher_parameters_sha256",
            "controller_state_sha256",
        }
        if set(state) != required_keys:
            raise ReplayMismatchError("malformed teacher-controller checkpoint keys")
        if state["schema_version"] != self._SCHEMA_VERSION:
            raise ReplayMismatchError(
                "unsupported teacher-controller checkpoint schema"
            )
        if state["spec_identity_sha256"] != self.spec.identity_sha256:
            raise IdentityMismatchError("teacher-controller spec identity mismatch")
        if state["regularization"] != self.spec.regularization.value:
            raise IdentityMismatchError("teacher regularization mode mismatch")

        initial = _mapping_from_state(
            state["initial_parameters"], self.spec.parameter_dtype
        )
        teacher = _mapping_from_state(
            state["teacher_parameters"], self.spec.parameter_dtype
        )
        _validate_parameter_layout(initial, teacher)
        if parameter_mapping_sha256(initial) != state["initial_parameters_sha256"]:
            raise ReplayMismatchError("initial teacher parameter checksum mismatch")
        if parameter_mapping_sha256(teacher) != state["teacher_parameters_sha256"]:
            raise ReplayMismatchError("evolved teacher parameter checksum mismatch")
        if parameter_mapping_sha256(initial) != self.spec.initial_teacher_sha256:
            raise IdentityMismatchError("checkpoint initial teacher identity mismatch")

        initial_policy = _policy_version_from_state(state["initial_policy_version"])
        last_policy = _policy_version_from_state(state["last_policy_version"])
        update_calls = _strict_nonnegative_int(state["update_calls"], "update_calls")
        update_count = _strict_nonnegative_int(state["update_count"], "update_count")
        if update_count > update_calls:
            raise ReplayMismatchError(
                "teacher update count exceeds observed update calls"
            )
        if initial_policy.run_id != last_policy.run_id:
            raise IdentityMismatchError(
                "teacher checkpoint changes policy run identity"
            )
        if last_policy.optimizer_step < initial_policy.optimizer_step:
            raise IdentityMismatchError(
                "teacher checkpoint policy version moved backwards"
            )

        device = next(iter(self._teacher_parameters.values())).device
        self._initial_parameters = _clone_mapping(initial, device=device)
        self._teacher_parameters = _clone_mapping(teacher, device=device)
        self._initial_policy_version = initial_policy
        self._last_policy_version = last_policy
        self._update_calls = update_calls
        self._update_count = update_count
        if state["controller_state_sha256"] != self.state_identity_sha256:
            raise ReplayMismatchError(
                "teacher-controller aggregate state checksum mismatch"
            )

    @classmethod
    def from_state_dict(
        cls,
        spec: TeacherControllerSpec,
        state: Mapping[str, object],
    ) -> "TeacherController":
        initial = _mapping_from_state(
            state.get("initial_parameters"), spec.parameter_dtype
        )
        initial_policy = _policy_version_from_state(state.get("initial_policy_version"))
        controller = cls(spec, initial, initial_policy)
        controller.load_state_dict(state)
        return controller


def build_teacher_context(
    *,
    spec: TeacherContextSpec,
    context_id: str,
    trajectory_identity_sha256: str,
    feedback_identity_sha256: str,
    feedback_token_indices: tuple[int, ...],
    student_token_ids: tuple[int, ...],
    teacher_token_ids: tuple[int, ...],
    replay_handle: TrajectoryReplayHandle,
    observation_handles: tuple[ObservationHandle, ...],
    turns: tuple[TeacherTurnAlignment, ...],
) -> TeacherContextRecord:
    """Record already-tokenized context; never decode and re-tokenize a response."""

    if spec.overflow is not TeacherContextOverflow.ERROR:
        raise ValueError(
            f"unsupported teacher context overflow behavior: {spec.overflow!r}"
        )
    if len(student_token_ids) > spec.maximum_context_tokens:
        raise ReplayMismatchError(
            "student transcript exceeds the explicit teacher context limit"
        )
    if len(teacher_token_ids) > spec.maximum_context_tokens:
        raise ReplayMismatchError(
            "feedback-conditioned transcript exceeds the teacher context limit"
        )
    return TeacherContextRecord(
        context_id=context_id,
        trajectory_identity_sha256=trajectory_identity_sha256,
        context_spec_identity_sha256=spec.identity_sha256,
        feedback_identity_sha256=feedback_identity_sha256,
        feedback_token_indices=feedback_token_indices,
        student_token_ids=student_token_ids,
        teacher_token_ids=teacher_token_ids,
        replay_handle=replay_handle,
        observation_handles=observation_handles,
        turns=turns,
    )


def replay_teacher_on_recorded_observations(
    *,
    adapter: QwenVLMFamilyAdapter,
    model: Any,
    store: ObservationStore,
    replay_handle: TrajectoryReplayHandle,
    controller: TeacherController,
    teacher_policy_version: PolicyVersion,
) -> TeacherReplayArtifact:
    """Run the loaded teacher on one exact, store-verified replay.

    The function refuses a model whose actual parameters do not equal both the
    controller's current teacher state and the supplied teacher policy version.
    """

    if not isinstance(adapter, QwenVLMFamilyAdapter):
        raise TypeError("teacher replay requires a QwenVLMFamilyAdapter")
    if not isinstance(store, ObservationStore):
        raise TypeError("teacher replay requires an ObservationStore")
    if not isinstance(controller, TeacherController):
        raise TypeError("teacher replay requires a TeacherController")
    if not isinstance(teacher_policy_version, PolicyVersion):
        raise TypeError("teacher_policy_version must be PolicyVersion")
    replay = store.resolve_replay(replay_handle)
    if replay.model.family != adapter.capabilities.family:
        raise IdentityMismatchError(
            "teacher replay model family differs from the Qwen family adapter"
        )
    if teacher_policy_version.run_id != controller.last_policy_version.run_id:
        raise IdentityMismatchError(
            "teacher policy version changes the controller policy run identity"
        )
    if (
        teacher_policy_version.optimizer_step
        != controller.last_policy_version.optimizer_step
    ):
        raise IdentityMismatchError(
            "teacher policy version step differs from the controller state"
        )
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("teacher model must expose named_parameters()")
    loaded_parameters = dict(named_parameters())
    loaded_sha256 = parameter_mapping_sha256(loaded_parameters)
    controller_parameters_sha256 = parameter_mapping_sha256(
        controller.teacher_parameters()
    )
    if loaded_sha256 != controller_parameters_sha256:
        raise IdentityMismatchError(
            "loaded teacher parameters differ from the controller's current teacher"
        )
    if teacher_policy_version.weights_sha256 != loaded_sha256:
        raise IdentityMismatchError(
            "teacher policy version does not identify the loaded model parameters"
        )
    if bool(getattr(model, "training", False)):
        raise ReplayMismatchError(
            "teacher replay model must be in evaluation mode for deterministic forward"
        )

    controller_state_sha256 = controller.state_identity_sha256
    with torch.no_grad():
        result = adapter.forward_recorded(
            model,
            store,
            replay_handle,
            ReplayConsumer.TEACHER,
        )
    if not isinstance(result, RecordedReplayResult):
        raise TypeError("Qwen family adapter returned an invalid replay result")
    logits = result.logits.detach().contiguous().clone()
    expected_sequence = replay.tensors.input_ids.descriptor.shape[-1]
    if logits.ndim != 3 or logits.shape[:2] != (1, expected_sequence):
        raise ReplayMismatchError(
            "teacher logits do not match the exact recorded replay sequence"
        )
    if controller.state_identity_sha256 != controller_state_sha256:
        raise ReplayMismatchError("teacher controller changed during replay")
    if parameter_mapping_sha256(dict(model.named_parameters())) != loaded_sha256:
        raise ReplayMismatchError("loaded teacher parameters changed during replay")

    return TeacherReplayArtifact(
        schema_version="sdpo-teacher-replay-v1",
        replay_handle=replay_handle,
        replay_record_sha256=replay_handle.record_sha256,
        teacher_controller_spec_identity_sha256=controller.spec.identity_sha256,
        teacher_controller_state_identity_sha256=controller_state_sha256,
        teacher_policy_version=teacher_policy_version,
        loaded_model_parameters_sha256=loaded_sha256,
        adapter_family=adapter.capabilities.family,
        logits=logits,
        logits_sha256=_tensor_sha256(logits),
    )


def validate_teacher_replay(
    *,
    context_spec: TeacherContextSpec,
    contexts: tuple[TeacherContextRecord, ...],
    artifacts: tuple[TeacherReplayArtifact, ...],
    store: ObservationStore,
    distillation_target_mask: torch.Tensor,
) -> None:
    """Verify exact observations and exact multi-turn response alignment."""

    if (
        distillation_target_mask.ndim != 2
        or distillation_target_mask.dtype is not torch.bool
    ):
        raise TypeError("distillation target mask must be bool [batch, sequence]")
    if len(contexts) != distillation_target_mask.shape[0]:
        raise ReplayMismatchError(
            "one teacher context is required per student sequence"
        )
    if len(artifacts) != len(contexts):
        raise ReplayMismatchError(
            "one teacher replay artifact is required per teacher context"
        )
    if not isinstance(store, ObservationStore):
        raise TypeError("teacher replay validation requires an ObservationStore")
    for batch_index, (context, artifact) in enumerate(
        zip(contexts, artifacts, strict=True)
    ):
        if context.context_spec_identity_sha256 != context_spec.identity_sha256:
            raise IdentityMismatchError(
                "teacher context was built with a different context spec"
            )
        artifact.validate_integrity()
        if artifact.replay_handle != context.replay_handle:
            raise ReplayMismatchError(
                "teacher replay artifact does not match the context replay handle"
            )
        replay = store.resolve_replay(context.replay_handle)
        if artifact.replay_record_sha256 != context.replay_handle.record_sha256:
            raise ReplayMismatchError(
                "teacher replay artifact digest differs from the exact replay"
            )
        if artifact.adapter_family != replay.model.family:
            raise IdentityMismatchError(
                "teacher replay adapter family differs from the recorded model"
            )
        context.validate_exact_observations(replay.observation_handles)
        resolved_ids = store.resolve_verified(replay.tensors.input_ids)
        if resolved_ids.shape != (1, len(context.teacher_token_ids)):
            raise ReplayMismatchError(
                "teacher context length differs from the exact replay input IDs"
            )
        if tuple(int(value) for value in resolved_ids[0].tolist()) != (
            context.teacher_token_ids
        ):
            raise ReplayMismatchError(
                "teacher context token IDs differ from the exact replay input IDs"
            )
        trajectory_sha256 = hashlib.sha256(
            replay.trajectory_id.encode("utf-8")
        ).hexdigest()
        if trajectory_sha256 != context.trajectory_identity_sha256:
            raise IdentityMismatchError(
                "teacher context trajectory identity differs from the exact replay"
            )
        if artifact.logits.shape[1] != len(context.teacher_token_ids):
            raise ReplayMismatchError(
                "teacher replay logits differ from the context transcript length"
            )
        aligned = context.aligned_student_mask(
            distillation_target_mask.shape[1],
            device=distillation_target_mask.device,
        )
        if not torch.equal(aligned, distillation_target_mask[batch_index]):
            raise ReplayMismatchError(
                "teacher token alignment differs from distillation target mask"
            )


def _validated_parameter_mapping(
    parameters: Mapping[str, torch.Tensor],
    expected_dtype: str | None,
) -> dict[str, torch.Tensor]:
    if not isinstance(parameters, Mapping) or not parameters:
        raise TypeError("teacher parameter mapping must be non-empty")
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in parameters.items():
        if not isinstance(name, str) or not name:
            raise TypeError("teacher parameter names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"teacher parameter {name!r} is not a tensor")
        if tensor.layout is not torch.strided:
            raise TypeError("teacher controller supports strided parameter tensors")
        if not tensor.dtype.is_floating_point:
            raise TypeError("teacher parameters must use a floating dtype")
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if expected_dtype is not None and dtype_name != expected_dtype:
            raise ReplayMismatchError(
                f"teacher parameter {name!r} has dtype {dtype_name}, expected {expected_dtype}"
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(f"teacher parameter {name!r} contains non-finite values")
        normalized[name] = tensor
    return normalized


def _validate_parameter_layout(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    if set(expected) != set(actual):
        raise ReplayMismatchError("teacher/student parameter names differ")
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if expected_tensor.shape != actual_tensor.shape:
            raise ReplayMismatchError(
                f"teacher/student parameter shape differs for {name!r}"
            )
        if expected_tensor.dtype != actual_tensor.dtype:
            raise ReplayMismatchError(
                f"teacher/student parameter dtype differs for {name!r}"
            )


def _clone_mapping(
    parameters: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        .to(device=device if device is not None else tensor.device)
        .contiguous()
        .clone()
        for name, tensor in parameters.items()
    }


def _mapping_from_state(value: object, expected_dtype: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ReplayMismatchError(
            "teacher checkpoint parameter payload is not a mapping"
        )
    try:
        return _validated_parameter_mapping(value, expected_dtype)
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(
            "malformed teacher checkpoint parameter payload"
        ) from error


def _policy_version_state(version: PolicyVersion) -> dict[str, object]:
    return {
        "run_id": version.run_id,
        "optimizer_step": version.optimizer_step,
        "weights_sha256": version.weights_sha256,
    }


def _policy_version_from_state(value: object) -> PolicyVersion:
    if not isinstance(value, Mapping) or set(value) != {
        "run_id",
        "optimizer_step",
        "weights_sha256",
    }:
        raise ReplayMismatchError("malformed policy version in teacher checkpoint")
    run_id = value["run_id"]
    optimizer_step = value["optimizer_step"]
    weights_sha256 = value["weights_sha256"]
    if (
        not isinstance(run_id, str)
        or not isinstance(optimizer_step, int)
        or isinstance(optimizer_step, bool)
        or not isinstance(weights_sha256, str)
    ):
        raise ReplayMismatchError("malformed policy version field types")
    try:
        return PolicyVersion(run_id, optimizer_step, weights_sha256)
    except ValueError as error:
        raise ReplayMismatchError(
            "invalid policy version in teacher checkpoint"
        ) from error


def _strict_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReplayMismatchError(
            f"teacher checkpoint {name} must be a non-negative integer"
        )
    return value
