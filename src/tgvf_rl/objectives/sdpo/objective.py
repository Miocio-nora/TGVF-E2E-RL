"""Pure-tensor SDPO losses with exact teacher replay validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.observations.store import ObservationStore

from ..base import (
    ObjectiveResult,
    PolicyLogProbSet,
    RatioDenominator,
    reduce_token_loss,
    reference_kl_per_token,
    tensors_share_storage,
)
from .schema import (
    FullVocabularyDivergence,
    ImportanceSamplingMode,
    SDPOLossMode,
    SDPOSpec,
    TeacherContextRecord,
    TeacherReplayArtifact,
)
from .teacher import (
    TeacherController,
    parameter_mapping_sha256,
    validate_teacher_replay,
)


@dataclass(frozen=True, slots=True)
class SDPOInputs:
    """Materialized inputs for one pure SDPO objective evaluation.

    The caller supplies only verified teacher replay artifacts.  Full-vocabulary
    or sampled teacher log probabilities are derived internally from their raw
    logits and ``TeacherContextRecord`` token alignment.
    """

    policy: PolicyLogProbSet
    teacher_contexts: tuple[TeacherContextRecord, ...]
    teacher_replay_artifacts: tuple[TeacherReplayArtifact, ...]
    distillation_target_mask: torch.Tensor
    student_full_log_probs: torch.Tensor | None

    def __post_init__(self) -> None:
        if not isinstance(self.distillation_target_mask, torch.Tensor):
            raise TypeError("distillation_target_mask must be a torch.Tensor")
        if self.distillation_target_mask.dtype is not torch.bool:
            raise TypeError("distillation_target_mask must have dtype bool")
        if self.distillation_target_mask.shape != self.policy.policy_sampled_mask.shape:
            raise ReplayMismatchError(
                "distillation and policy-sampled masks differ in shape"
            )
        if (
            self.distillation_target_mask.device
            != self.policy.policy_sampled_mask.device
        ):
            raise ReplayMismatchError(
                "distillation and policy-sampled masks differ in device"
            )
        if not bool(self.distillation_target_mask.any().item()):
            raise ValueError(
                "pure SDPO requires at least one aligned distillation token"
            )
        if bool(
            (self.distillation_target_mask & ~self.policy.policy_sampled_mask)
            .any()
            .item()
        ):
            raise ReplayMismatchError("SDPO target mask includes template-owned tokens")
        if len(self.teacher_contexts) != self.policy.current.values.shape[0]:
            raise ReplayMismatchError(
                "one teacher context is required per policy sequence"
            )
        if len(self.teacher_replay_artifacts) != len(self.teacher_contexts):
            raise ReplayMismatchError(
                "one teacher replay artifact is required per teacher context"
            )
        for artifact in self.teacher_replay_artifacts:
            if not isinstance(artifact, TeacherReplayArtifact):
                raise TypeError(
                    "teacher_replay_artifacts must contain TeacherReplayArtifact instances"
                )
            artifact.validate_integrity()

        _validate_optional_tensor(
            self.student_full_log_probs,
            name="student_full_log_probs",
            rank=3,
            requires_grad=True,
        )
        if self.student_full_log_probs is not None:
            if self.student_full_log_probs.device != self.policy.current.values.device:
                raise ReplayMismatchError(
                    "student_full_log_probs is on a different device from policy replay"
                )
            if (
                self.student_full_log_probs.shape[:2]
                != self.policy.current.values.shape
            ):
                raise ReplayMismatchError(
                    "student full distribution shape does not match policy replay"
                )


def compute_sdpo_loss(
    spec: SDPOSpec,
    inputs: SDPOInputs,
    *,
    teacher_controller: TeacherController,
    observation_store: ObservationStore,
) -> ObjectiveResult:
    """Compute a pure SDPO loss; no reward or GRPO term is accepted here."""

    if (
        teacher_controller.spec.identity_sha256
        != spec.teacher_controller.identity_sha256
    ):
        raise IdentityMismatchError(
            "SDPO compute received a different teacher controller spec"
        )
    current_teacher_state = teacher_controller.state_identity_sha256
    current_teacher_parameters = parameter_mapping_sha256(
        teacher_controller.teacher_parameters()
    )
    for artifact in inputs.teacher_replay_artifacts:
        if (
            artifact.teacher_controller_spec_identity_sha256
            != spec.teacher_controller.identity_sha256
        ):
            raise IdentityMismatchError(
                "teacher replay artifact was produced by a different controller spec"
            )
        if artifact.teacher_controller_state_identity_sha256 != current_teacher_state:
            raise IdentityMismatchError(
                "teacher replay artifact does not match the controller's current state"
            )
        if artifact.loaded_model_parameters_sha256 != current_teacher_parameters:
            raise IdentityMismatchError(
                "teacher replay artifact does not match the current teacher parameters"
            )
    validate_teacher_replay(
        context_spec=spec.teacher_context,
        contexts=inputs.teacher_contexts,
        artifacts=inputs.teacher_replay_artifacts,
        store=observation_store,
        distillation_target_mask=inputs.distillation_target_mask,
    )
    target_mask = inputs.distillation_target_mask

    if spec.loss_mode is SDPOLossMode.FULL_VOCAB_DIVERGENCE:
        if inputs.student_full_log_probs is None:
            raise ValueError(
                "full-vocabulary SDPO requires a differentiable student distribution"
            )
        aligned_teacher_full_log_probs = _align_teacher_full_log_probs(inputs)
        _validate_log_distributions(
            inputs.student_full_log_probs,
            target_mask,
            tolerance=spec.distribution_tolerance,
            name="student",
        )
        _validate_log_distributions(
            aligned_teacher_full_log_probs,
            target_mask,
            tolerance=spec.distribution_tolerance,
            name="teacher",
        )
        _validate_current_samples_match_full_distribution(spec, inputs)
        per_token_distillation = _full_vocabulary_divergence(
            student_log_probs=inputs.student_full_log_probs,
            teacher_log_probs=aligned_teacher_full_log_probs,
            divergence=spec.full_vocab_divergence,
            jsd_alpha=spec.jsd_alpha,
        )
        importance_weights = torch.ones_like(per_token_distillation)
        reported_reverse_kl = per_token_distillation
    elif spec.loss_mode is SDPOLossMode.SAMPLED_TOKEN_REVERSE_KL_SCORE_FUNCTION:
        if inputs.student_full_log_probs is not None:
            raise ValueError(
                "student full distribution must be None in sampled-token SDPO mode"
            )
        teacher_sampled = _align_teacher_sampled_log_probs(inputs)
        log_ratio = inputs.policy.current.values - teacher_sampled.detach()
        # This is the explicitly named score-function surrogate from the SDPO
        # reference: the sampled reverse-KL score is stop-gradient while the
        # current log probability supplies the policy gradient.
        per_token_distillation = log_ratio.detach() * inputs.policy.current.values
        importance_weights = _importance_weights(spec, inputs.policy)
        per_token_distillation = importance_weights * per_token_distillation
        reported_reverse_kl = log_ratio
    else:
        raise ValueError(f"unknown SDPO loss mode: {spec.loss_mode!r}")

    reference_kl = reference_kl_per_token(
        inputs.policy.current.values,
        inputs.policy.reference.values,
        spec.reference_kl,
    )
    raw_per_token_loss = (
        spec.distillation_coefficient * per_token_distillation
        + spec.reference_kl.coefficient * reference_kl
    )
    per_token_loss = torch.where(
        target_mask,
        raw_per_token_loss,
        torch.zeros_like(raw_per_token_loss),
    )
    loss = reduce_token_loss(per_token_loss, target_mask, spec.reduction)

    distillation_loss = reduce_token_loss(
        torch.where(
            target_mask,
            per_token_distillation,
            torch.zeros_like(per_token_distillation),
        ),
        target_mask,
        spec.reduction,
    )
    reference_kl_loss = reduce_token_loss(
        torch.where(target_mask, reference_kl, torch.zeros_like(reference_kl)),
        target_mask,
        spec.reduction,
    )
    reverse_kl_metric = reported_reverse_kl[target_mask].mean()
    metrics = {
        "loss": loss.detach(),
        "distillation_loss": distillation_loss.detach(),
        "distillation_contribution": (
            spec.distillation_coefficient * distillation_loss
        ).detach(),
        "reference_kl": reference_kl_loss.detach(),
        "reference_kl_contribution": (
            spec.reference_kl.coefficient * reference_kl_loss
        ).detach(),
        "sampled_reverse_kl_or_full_divergence": reverse_kl_metric.detach(),
        "mean_importance_weight": importance_weights[target_mask].mean().detach(),
        "distillation_token_count": float(target_mask.sum().item()),
    }
    return ObjectiveResult(
        loss=loss,
        per_token_loss=per_token_loss,
        metrics=metrics,
        spec_identity_sha256=spec.identity_sha256,
    )


def _full_vocabulary_divergence(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    divergence: FullVocabularyDivergence | None,
    jsd_alpha: float | None,
) -> torch.Tensor:
    student_probabilities = student_log_probs.exp()
    teacher_probabilities = teacher_log_probs.exp()
    if divergence is FullVocabularyDivergence.KL_TEACHER_TO_STUDENT:
        return (teacher_probabilities * (teacher_log_probs - student_log_probs)).sum(
            dim=-1
        )
    if divergence is FullVocabularyDivergence.KL_STUDENT_TO_TEACHER:
        return (student_probabilities * (student_log_probs - teacher_log_probs)).sum(
            dim=-1
        )
    if divergence is FullVocabularyDivergence.GENERALIZED_JENSEN_SHANNON:
        assert jsd_alpha is not None
        log_mixture = torch.logaddexp(
            student_log_probs + math.log(1.0 - jsd_alpha),
            teacher_log_probs + math.log(jsd_alpha),
        )
        student_to_mixture = (
            student_probabilities * (student_log_probs - log_mixture)
        ).sum(dim=-1)
        teacher_to_mixture = (
            teacher_probabilities * (teacher_log_probs - log_mixture)
        ).sum(dim=-1)
        return (1.0 - jsd_alpha) * student_to_mixture + jsd_alpha * teacher_to_mixture
    raise ValueError(f"unknown full-vocabulary divergence: {divergence!r}")


def _importance_weights(spec: SDPOSpec, policy: PolicyLogProbSet) -> torch.Tensor:
    importance = spec.importance_sampling
    if importance.mode is ImportanceSamplingMode.NONE:
        return torch.ones_like(policy.current.values)
    if importance.mode is ImportanceSamplingMode.CLIPPED:
        if importance.denominator is not RatioDenominator.BEHAVIOR:
            raise ValueError(
                "sampled-token SDPO importance sampling requires behavior denominator"
            )
        assert importance.minimum_ratio is not None
        assert importance.maximum_ratio is not None
        denominator = policy.ratio_denominator(importance.denominator)
        ratios = torch.exp(policy.current.values.detach() - denominator)
        return ratios.clamp(min=importance.minimum_ratio, max=importance.maximum_ratio)
    raise ValueError(f"unknown importance-sampling mode: {importance.mode!r}")


def _align_teacher_full_log_probs(inputs: SDPOInputs) -> torch.Tensor:
    assert inputs.student_full_log_probs is not None
    student = inputs.student_full_log_probs
    vocabulary_size = student.shape[-1]
    aligned = student.detach().new_zeros(student.shape)
    for batch_index, (context, artifact) in enumerate(
        zip(inputs.teacher_contexts, inputs.teacher_replay_artifacts, strict=True)
    ):
        if artifact.logits.device != student.device:
            raise ReplayMismatchError(
                "teacher replay logits are on a different device from the student"
            )
        if artifact.logits.dtype != student.dtype:
            raise ReplayMismatchError(
                "teacher replay logits use a different dtype from the student"
            )
        if artifact.logits.shape[-1] != vocabulary_size:
            raise ReplayMismatchError(
                "teacher and student full distributions differ in vocabulary size"
            )
        teacher_log_probs = artifact.logits[0].log_softmax(dim=-1)
        for turn in context.turns:
            for student_index, teacher_index in zip(
                turn.student_token_indices,
                turn.teacher_token_indices,
                strict=True,
            ):
                if teacher_index == 0:
                    raise ReplayMismatchError(
                        "an aligned teacher token has no preceding replay logit"
                    )
                aligned[batch_index, student_index] = teacher_log_probs[
                    teacher_index - 1
                ]
    if tensors_share_storage(student, aligned):
        raise ReplayMismatchError("student and teacher distributions share storage")
    return aligned.detach()


def _align_teacher_sampled_log_probs(inputs: SDPOInputs) -> torch.Tensor:
    aligned = inputs.policy.current.values.detach().new_zeros(
        inputs.policy.current.values.shape
    )
    for batch_index, (context, artifact) in enumerate(
        zip(inputs.teacher_contexts, inputs.teacher_replay_artifacts, strict=True)
    ):
        if artifact.logits.device != aligned.device:
            raise ReplayMismatchError(
                "teacher replay logits are on a different device from policy replay"
            )
        teacher_log_probs = artifact.logits[0].log_softmax(dim=-1)
        vocabulary_size = teacher_log_probs.shape[-1]
        for turn in context.turns:
            for student_index, teacher_index in zip(
                turn.student_token_indices,
                turn.teacher_token_indices,
                strict=True,
            ):
                if teacher_index == 0:
                    raise ReplayMismatchError(
                        "an aligned teacher token has no preceding replay logit"
                    )
                token_id = context.teacher_token_ids[teacher_index]
                if token_id >= vocabulary_size:
                    raise ReplayMismatchError(
                        "aligned teacher token lies outside the replay vocabulary"
                    )
                aligned[batch_index, student_index] = teacher_log_probs[
                    teacher_index - 1, token_id
                ]
    if tensors_share_storage(inputs.policy.current.values, aligned):
        raise ReplayMismatchError(
            "student and teacher sampled log probabilities share storage"
        )
    return aligned.detach()


def _validate_current_samples_match_full_distribution(
    spec: SDPOSpec,
    inputs: SDPOInputs,
) -> None:
    assert inputs.student_full_log_probs is not None
    vocabulary_size = inputs.student_full_log_probs.shape[-1]
    token_ids = torch.tensor(
        [context.student_token_ids for context in inputs.teacher_contexts],
        dtype=torch.long,
        device=inputs.student_full_log_probs.device,
    )
    if bool((token_ids >= vocabulary_size).any().item()):
        raise ReplayMismatchError(
            "student transcript contains token outside the full distribution"
        )
    sampled = inputs.student_full_log_probs.gather(
        dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    mask = inputs.distillation_target_mask
    if not torch.allclose(
        sampled[mask].detach(),
        inputs.policy.current.values[mask].detach(),
        rtol=0.0,
        atol=spec.distribution_tolerance,
    ):
        raise ReplayMismatchError(
            "current sampled log probabilities differ from the supplied full distribution"
        )


def _validate_log_distributions(
    log_probabilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    tolerance: float,
    name: str,
) -> None:
    normalizers = torch.logsumexp(log_probabilities.detach(), dim=-1)
    if bool((normalizers[mask].abs() > tolerance).any().item()):
        raise ReplayMismatchError(
            f"{name} full-vocabulary log probabilities are not normalized"
        )


def _validate_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    name: str,
    rank: int,
    requires_grad: bool | None,
) -> None:
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if requires_grad is not None and tensor.requires_grad is not requires_grad:
        qualifier = "require" if requires_grad else "forbid"
        raise ValueError(f"{name} must {qualifier} gradients")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"expected lowercase SHA256, got {value!r}")
