"""Immutable schemas for feedback-conditioned pure SDPO."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import PolicyVersion
from tgvf_rl.observations.store import (
    ObservationHandle,
    TrajectoryReplayHandle,
)

from ..base import (
    RatioDenominator,
    ReductionSpec,
    ReferenceKLSpec,
    spec_identity_sha256,
)


class SDPOLossMode(str, Enum):
    FULL_VOCAB_DIVERGENCE = "full_vocab_divergence"
    SAMPLED_TOKEN_REVERSE_KL_SCORE_FUNCTION = "sampled_token_reverse_kl_score_function"


class FullVocabularyDivergence(str, Enum):
    KL_TEACHER_TO_STUDENT = "kl_teacher_to_student"
    KL_STUDENT_TO_TEACHER = "kl_student_to_teacher"
    GENERALIZED_JENSEN_SHANNON = "generalized_jensen_shannon"


class ImportanceSamplingMode(str, Enum):
    NONE = "none"
    CLIPPED = "clipped"


class TeacherRegularization(str, Enum):
    EMA_PARAMETERS = "ema_parameters"
    TRUST_REGION_LOGIT_INTERPOLATION = "trust_region_logit_interpolation"


class TeacherContextOverflow(str, Enum):
    """The initial contract refuses hidden truncation of teacher context."""

    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ImportanceSamplingSpec:
    mode: ImportanceSamplingMode
    denominator: RatioDenominator | None
    minimum_ratio: float | None
    maximum_ratio: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ImportanceSamplingMode):
            raise TypeError("importance-sampling mode must be ImportanceSamplingMode")
        if self.mode is ImportanceSamplingMode.NONE:
            if any(
                value is not None
                for value in (self.denominator, self.minimum_ratio, self.maximum_ratio)
            ):
                raise ValueError(
                    "disabled importance sampling requires explicit None parameters"
                )
            return
        if self.mode is ImportanceSamplingMode.CLIPPED:
            if not isinstance(self.denominator, RatioDenominator):
                raise TypeError(
                    "importance-sampling denominator must be RatioDenominator"
                )
            if (
                self.denominator is None
                or self.minimum_ratio is None
                or self.maximum_ratio is None
            ):
                raise ValueError(
                    "clipped importance sampling requires denominator and both bounds"
                )
            _require_real(self.minimum_ratio, "minimum importance-sampling ratio")
            _require_real(self.maximum_ratio, "maximum importance-sampling ratio")
            if not math.isfinite(self.minimum_ratio) or not math.isfinite(
                self.maximum_ratio
            ):
                raise ValueError("importance-sampling bounds must be finite")
            if not 0 < self.minimum_ratio <= self.maximum_ratio:
                raise ValueError(
                    "importance-sampling bounds must satisfy 0 < min <= max"
                )
            if self.denominator is not RatioDenominator.BEHAVIOR:
                raise ValueError(
                    "sampled-token SDPO importance sampling must use the actual behavior policy denominator"
                )
            return
        raise ValueError(f"unknown importance-sampling mode: {self.mode!r}")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class TeacherControllerSpec:
    """Teacher evolution contract.

    ``student_weight`` means the coefficient of the current student in both
    modes: ``(1-w)*teacher + w*student`` for EMA parameters and
    ``(1-w)*reference_logits + w*feedback_student_logits`` for trust-region
    interpolation.
    """

    regularization: TeacherRegularization
    student_weight: float
    update_interval: int
    initial_teacher_sha256: str
    parameter_dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.regularization, TeacherRegularization):
            raise TypeError("teacher regularization must be TeacherRegularization")
        _require_real(self.student_weight, "teacher student_weight")
        if (
            not math.isfinite(self.student_weight)
            or not 0.0 <= self.student_weight <= 1.0
        ):
            raise ValueError("teacher student_weight must be finite and in [0, 1]")
        if not isinstance(self.update_interval, int) or isinstance(
            self.update_interval, bool
        ):
            raise TypeError("teacher update_interval must be an integer")
        if self.update_interval <= 0:
            raise ValueError("teacher update_interval must be positive")
        _validate_sha256(self.initial_teacher_sha256)
        if not self.parameter_dtype:
            raise ValueError("teacher parameter_dtype must be explicit")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class TeacherContextSpec:
    template_sha256: str
    maximum_context_tokens: int
    overflow: TeacherContextOverflow
    require_feedback: bool
    require_exact_token_alignment: bool

    def __post_init__(self) -> None:
        if not isinstance(self.overflow, TeacherContextOverflow):
            raise TypeError("teacher context overflow must be TeacherContextOverflow")
        if not isinstance(self.require_feedback, bool) or not isinstance(
            self.require_exact_token_alignment, bool
        ):
            raise TypeError("teacher context requirements must be bool")
        _validate_sha256(self.template_sha256)
        if not isinstance(self.maximum_context_tokens, int) or isinstance(
            self.maximum_context_tokens, bool
        ):
            raise TypeError("maximum teacher context length must be an integer")
        if self.maximum_context_tokens <= 0:
            raise ValueError("maximum teacher context length must be positive")
        if not self.require_feedback:
            raise ValueError("SDPO teacher context must be feedback-conditioned")
        if not self.require_exact_token_alignment:
            raise ValueError("SDPO requires exact response-token alignment")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


@dataclass(frozen=True, slots=True)
class TeacherTurnAlignment:
    """Exact mapping for policy tokens in one assistant turn."""

    assistant_turn_index: int
    student_token_indices: tuple[int, ...]
    teacher_token_indices: tuple[int, ...]
    visible_observation_handles: tuple[ObservationHandle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assistant_turn_index, int) or isinstance(
            self.assistant_turn_index, bool
        ):
            raise TypeError("assistant turn index must be an integer")
        if self.assistant_turn_index < 0:
            raise ValueError("assistant turn index must be non-negative")
        for name, value in (
            ("student_token_indices", self.student_token_indices),
            ("teacher_token_indices", self.teacher_token_indices),
            ("visible_observation_handles", self.visible_observation_handles),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        if not self.student_token_indices:
            raise ValueError("a turn alignment must contain at least one policy token")
        if len(self.student_token_indices) != len(self.teacher_token_indices):
            raise ValueError(
                "student and teacher token alignments must have equal length"
            )
        _validate_strictly_increasing(
            self.student_token_indices, "student token indices"
        )
        _validate_strictly_increasing(
            self.teacher_token_indices, "teacher token indices"
        )
        for handle in self.visible_observation_handles:
            _validate_observation_handle(handle)


@dataclass(frozen=True, slots=True)
class TeacherContextRecord:
    """Tokenized teacher replay with no decode/re-tokenize ambiguity."""

    context_id: str
    trajectory_identity_sha256: str
    context_spec_identity_sha256: str
    feedback_identity_sha256: str
    feedback_token_indices: tuple[int, ...]
    student_token_ids: tuple[int, ...]
    teacher_token_ids: tuple[int, ...]
    replay_handle: TrajectoryReplayHandle
    observation_handles: tuple[ObservationHandle, ...]
    turns: tuple[TeacherTurnAlignment, ...]

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("teacher context ID must be non-empty")
        _validate_sha256(self.trajectory_identity_sha256)
        _validate_sha256(self.context_spec_identity_sha256)
        _validate_sha256(self.feedback_identity_sha256)
        _validate_replay_handle(self.replay_handle)
        for name, value in (
            ("feedback_token_indices", self.feedback_token_indices),
            ("student_token_ids", self.student_token_ids),
            ("teacher_token_ids", self.teacher_token_ids),
            ("observation_handles", self.observation_handles),
            ("turns", self.turns),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        if not self.student_token_ids or not self.teacher_token_ids:
            raise ValueError("teacher context token sequences must be non-empty")
        if not self.feedback_token_indices:
            raise ValueError(
                "feedback-conditioned teacher context requires feedback tokens"
            )
        _validate_strictly_increasing(
            self.feedback_token_indices, "feedback token indices"
        )
        if self.feedback_token_indices[-1] >= len(self.teacher_token_ids):
            raise ValueError("feedback token index is outside the teacher transcript")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in self.student_token_ids + self.teacher_token_ids
        ):
            raise TypeError("token IDs must be integers")
        if any(
            token_id < 0 for token_id in self.student_token_ids + self.teacher_token_ids
        ):
            raise ValueError("token IDs must be non-negative")
        for handle in self.observation_handles:
            _validate_observation_handle(handle)

        previous_turn = -1
        seen_student: set[int] = set()
        seen_teacher: set[int] = set()
        previous_visible_count = 0
        for turn in self.turns:
            if turn.assistant_turn_index <= previous_turn:
                raise ValueError("teacher turn alignments must be strictly ordered")
            previous_turn = turn.assistant_turn_index
            visible = turn.visible_observation_handles
            if visible != self.observation_handles[: len(visible)]:
                raise ReplayMismatchError(
                    "teacher turn observation visibility is not an exact trajectory prefix"
                )
            if len(visible) < previous_visible_count:
                raise ReplayMismatchError(
                    "teacher observation visibility cannot go backwards"
                )
            previous_visible_count = len(visible)

            for student_index, teacher_index in zip(
                turn.student_token_indices, turn.teacher_token_indices
            ):
                if student_index >= len(self.student_token_ids):
                    raise ValueError(
                        "student alignment index is outside the transcript"
                    )
                if teacher_index >= len(self.teacher_token_ids):
                    raise ValueError(
                        "teacher alignment index is outside the transcript"
                    )
                if student_index in seen_student or teacher_index in seen_teacher:
                    raise ValueError(
                        "a token index may appear in only one aligned turn"
                    )
                seen_student.add(student_index)
                seen_teacher.add(teacher_index)
                if (
                    self.student_token_ids[student_index]
                    != self.teacher_token_ids[teacher_index]
                ):
                    raise ReplayMismatchError(
                        "aligned student and teacher response token IDs differ"
                    )
        if seen_teacher.intersection(self.feedback_token_indices):
            raise ReplayMismatchError("feedback tokens overlap aligned response tokens")

    def aligned_student_mask(
        self, sequence_length: int, *, device: torch.device
    ) -> torch.Tensor:
        if sequence_length != len(self.student_token_ids):
            raise ReplayMismatchError(
                "student transcript length differs from objective sequence length"
            )
        mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        for turn in self.turns:
            mask[list(turn.student_token_indices)] = True
        return mask

    def validate_exact_observations(
        self,
        expected_handles: tuple[ObservationHandle, ...],
    ) -> None:
        if self.observation_handles != expected_handles:
            raise ReplayMismatchError(
                "teacher replay did not use the exact rollout observation handles"
            )


@dataclass(frozen=True, slots=True)
class TeacherReplayArtifact:
    """No-grad teacher logits bound to one verified recorded replay.

    The tensor content hash is stored separately so mutating the otherwise
    frozen artifact's tensor is detected before an objective consumes it.
    """

    schema_version: str
    replay_handle: TrajectoryReplayHandle
    replay_record_sha256: str
    teacher_controller_spec_identity_sha256: str
    teacher_controller_state_identity_sha256: str
    teacher_policy_version: PolicyVersion
    loaded_model_parameters_sha256: str
    adapter_family: str
    logits: torch.Tensor
    logits_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "sdpo-teacher-replay-v1":
            raise ValueError("unsupported teacher replay artifact schema")
        _validate_replay_handle(self.replay_handle)
        for value in (
            self.replay_record_sha256,
            self.teacher_controller_spec_identity_sha256,
            self.teacher_controller_state_identity_sha256,
            self.loaded_model_parameters_sha256,
            self.logits_sha256,
        ):
            _validate_sha256(value)
        if self.replay_record_sha256 != self.replay_handle.record_sha256:
            raise ReplayMismatchError(
                "teacher replay artifact record digest differs from its replay handle"
            )
        if not isinstance(self.teacher_policy_version, PolicyVersion):
            raise TypeError("teacher policy version must be PolicyVersion")
        if (
            self.teacher_policy_version.weights_sha256
            != self.loaded_model_parameters_sha256
        ):
            raise ReplayMismatchError(
                "teacher policy version does not identify the loaded model parameters"
            )
        if not self.adapter_family:
            raise ValueError("teacher replay adapter family must be non-empty")
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("teacher replay logits must be a torch.Tensor")
        if self.logits.ndim != 3 or self.logits.shape[0] != 1:
            raise ValueError(
                "teacher replay logits must have shape [1, sequence, vocab]"
            )
        if self.logits.shape[1] == 0 or self.logits.shape[2] == 0:
            raise ValueError("teacher replay logits must have non-empty axes")
        if not self.logits.dtype.is_floating_point:
            raise TypeError("teacher replay logits must use a floating dtype")
        if self.logits.requires_grad:
            raise ValueError("teacher replay logits must be gradient-free")
        if not bool(torch.isfinite(self.logits).all().item()):
            raise ValueError("teacher replay logits must be finite")
        frozen_logits = self.logits.detach().contiguous().clone()
        object.__setattr__(self, "logits", frozen_logits)
        self.validate_integrity()

    def validate_integrity(self) -> None:
        if _tensor_sha256(self.logits) != self.logits_sha256:
            raise ReplayMismatchError("teacher replay logits checksum mismatch")

    @property
    def identity_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "replay_id": self.replay_handle.replay_id,
            "replay_record_sha256": self.replay_record_sha256,
            "teacher_controller_spec_identity_sha256": (
                self.teacher_controller_spec_identity_sha256
            ),
            "teacher_controller_state_identity_sha256": (
                self.teacher_controller_state_identity_sha256
            ),
            "teacher_policy_version": {
                "run_id": self.teacher_policy_version.run_id,
                "optimizer_step": self.teacher_policy_version.optimizer_step,
                "weights_sha256": self.teacher_policy_version.weights_sha256,
            },
            "loaded_model_parameters_sha256": self.loaded_model_parameters_sha256,
            "adapter_family": self.adapter_family,
            "logits_sha256": self.logits_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SDPOSpec:
    """Complete mathematical identity for a pure SDPO objective."""

    loss_mode: SDPOLossMode
    full_vocab_divergence: FullVocabularyDivergence | None
    jsd_alpha: float | None
    importance_sampling: ImportanceSamplingSpec
    distillation_coefficient: float
    reference_kl: ReferenceKLSpec
    reduction: ReductionSpec
    teacher_controller: TeacherControllerSpec
    teacher_context: TeacherContextSpec
    distribution_tolerance: float
    stop_teacher_gradient: bool

    def __post_init__(self) -> None:
        if not isinstance(self.loss_mode, SDPOLossMode):
            raise TypeError("loss_mode must be SDPOLossMode")
        if self.full_vocab_divergence is not None and not isinstance(
            self.full_vocab_divergence, FullVocabularyDivergence
        ):
            raise TypeError(
                "full_vocab_divergence must be FullVocabularyDivergence or None"
            )
        if not isinstance(self.importance_sampling, ImportanceSamplingSpec):
            raise TypeError("importance_sampling must be ImportanceSamplingSpec")
        if not isinstance(self.reference_kl, ReferenceKLSpec):
            raise TypeError("reference_kl must be ReferenceKLSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise TypeError("reduction must be ReductionSpec")
        if not isinstance(self.teacher_controller, TeacherControllerSpec):
            raise TypeError("teacher_controller must be TeacherControllerSpec")
        if not isinstance(self.teacher_context, TeacherContextSpec):
            raise TypeError("teacher_context must be TeacherContextSpec")
        if not isinstance(self.stop_teacher_gradient, bool):
            raise TypeError("stop_teacher_gradient must be bool")
        _require_real(self.distillation_coefficient, "distillation coefficient")
        if (
            not math.isfinite(self.distillation_coefficient)
            or self.distillation_coefficient <= 0
        ):
            raise ValueError("distillation coefficient must be finite and positive")
        _require_real(self.distribution_tolerance, "distribution tolerance")
        if (
            not math.isfinite(self.distribution_tolerance)
            or self.distribution_tolerance <= 0
        ):
            raise ValueError("distribution tolerance must be finite and positive")
        if not self.stop_teacher_gradient:
            raise ValueError("teacher distributions must be stop-gradient")

        if self.loss_mode is SDPOLossMode.FULL_VOCAB_DIVERGENCE:
            if self.full_vocab_divergence is None:
                raise ValueError("full-vocabulary SDPO requires an explicit divergence")
            if self.importance_sampling.mode is not ImportanceSamplingMode.NONE:
                raise ValueError(
                    "importance sampling applies only to sampled-token SDPO"
                )
            if (
                self.full_vocab_divergence
                is FullVocabularyDivergence.GENERALIZED_JENSEN_SHANNON
            ):
                if self.jsd_alpha is not None:
                    _require_real(self.jsd_alpha, "generalized JSD alpha")
                if self.jsd_alpha is None or not math.isfinite(self.jsd_alpha):
                    raise ValueError("generalized JSD requires a finite alpha")
                if not 0.0 < self.jsd_alpha < 1.0:
                    raise ValueError(
                        "generalized JSD alpha must be strictly between zero and one"
                    )
            elif self.jsd_alpha is not None:
                raise ValueError("jsd_alpha must be None for a KL divergence")
            return

        if self.loss_mode is SDPOLossMode.SAMPLED_TOKEN_REVERSE_KL_SCORE_FUNCTION:
            if self.full_vocab_divergence is not None or self.jsd_alpha is not None:
                raise ValueError(
                    "sampled-token SDPO requires full-vocabulary fields to be None"
                )
            return
        raise ValueError(f"unknown SDPO loss mode: {self.loss_mode!r}")

    @property
    def identity_sha256(self) -> str:
        return spec_identity_sha256(self)


def _validate_strictly_increasing(values: tuple[int, ...], name: str) -> None:
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError(f"{name} must contain integers")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must be non-negative")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def _validate_observation_handle(handle: ObservationHandle) -> None:
    if not isinstance(handle, ObservationHandle):
        raise TypeError("observation handles must be ObservationHandle instances")
    if not handle.observation_id:
        raise ValueError("observation ID must be non-empty")
    _validate_sha256(handle.record_sha256)


def _validate_replay_handle(handle: TrajectoryReplayHandle) -> None:
    if not isinstance(handle, TrajectoryReplayHandle):
        raise TypeError("teacher replay handle must be TrajectoryReplayHandle")
    if not handle.replay_id:
        raise ValueError("teacher replay ID must be non-empty")
    _validate_sha256(handle.record_sha256)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"tgvf-sdpo-teacher-logits-v1\0")
    digest.update(str(cpu.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in cpu.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"expected lowercase SHA256, got {value!r}")


def _require_real(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
