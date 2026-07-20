from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch
from torch import nn

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion, SupportLevel
from tgvf_rl.objectives import (
    OBJECTIVE_REGISTRY,
    FullVocabularyDivergence,
    ImportanceSamplingMode,
    ImportanceSamplingSpec,
    LogProbSource,
    LossReduction,
    PolicyLogProbSet,
    RatioDenominator,
    ReductionSpec,
    ReferenceKLEstimator,
    ReferenceKLSpec,
    RoleLogProbs,
    SDPOInputs,
    SDPOLossMode,
    SDPOSpec,
    TeacherContextOverflow,
    TeacherContextSpec,
    TeacherController,
    TeacherControllerSpec,
    TeacherRegularization,
    TeacherTurnAlignment,
    build_teacher_context,
    compute_sdpo_loss,
    parameter_mapping_sha256,
    replay_teacher_on_recorded_observations,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayHandle,
    TrajectoryReplayRecord,
    TrajectoryReplayTensorRefs,
)
from tgvf_rl.qwen.base import (
    FamilyCapabilities,
    QwenVLMFamilyAdapter,
    RecordedReplayResult,
    ReplayConsumer,
    resolve_replay_request,
)

from tests.support import populated_observation_store, trajectory_source_visual


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
SHA4 = "4" * 64
SHA5 = "5" * 64
SHA6 = "6" * 64
SHA7 = "7" * 64
SHA8 = "8" * 64

STUDENT_TOKEN_IDS = (4, 0, 1, 4, 2, 4, 3, 1)
TEACHER_TOKEN_IDS = STUDENT_TOKEN_IDS
TARGET_MASK = torch.tensor([[False, True, True, False, True, False, True, True]])


class _TinyTeacherModel(nn.Module):
    def __init__(self, parameters: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.projection = nn.Module()
        self.projection.register_parameter(
            "weight", nn.Parameter(parameters["projection.weight"].clone())
        )


class _TeacherReplayAdapter(QwenVLMFamilyAdapter):
    capabilities = FamilyCapabilities(
        family="qwen3_vl",
        support_level=SupportLevel.SYNTHETIC,
        native_thinking_prefill=True,
        deepstack_branch_count=1,
        recorded_d_forward=True,
        native_tool_template=True,
    )

    def forward_recorded(
        self,
        model: _TinyTeacherModel,
        store: ObservationStore,
        replay_handle: TrajectoryReplayHandle,
        consumer: ReplayConsumer,
    ) -> RecordedReplayResult:
        assert consumer is ReplayConsumer.TEACHER
        request = resolve_replay_request(store, replay_handle, consumer)
        weight = model.projection.weight
        base = torch.arange(
            request.input_ids.shape[1] * 5,
            dtype=weight.dtype,
            device=weight.device,
        ).view(1, request.input_ids.shape[1], 5)
        logits = base.remainder(11).div(7.0) + weight.sum().mul(0.01)
        return RecordedReplayResult(
            logits=logits,
            hidden_states=logits,
            past_key_values=None,
            visual_position_mask=torch.zeros_like(
                request.attention_mask, dtype=torch.bool
            ),
        )


def _version(step: int, digit: str) -> PolicyVersion:
    return PolicyVersion("sdpo-test", step, digit * 64)


def _context_spec() -> TeacherContextSpec:
    return TeacherContextSpec(
        template_sha256=SHA3,
        maximum_context_tokens=32,
        overflow=TeacherContextOverflow.ERROR,
        require_feedback=True,
        require_exact_token_alignment=True,
    )


def _controller_spec(
    regularization: TeacherRegularization = TeacherRegularization.EMA_PARAMETERS,
) -> tuple[TeacherControllerSpec, dict[str, torch.Tensor]]:
    parameters = {"projection.weight": torch.tensor([0.0, 2.0], dtype=torch.float64)}
    spec = TeacherControllerSpec(
        regularization=regularization,
        student_weight=0.25,
        update_interval=1,
        initial_teacher_sha256=parameter_mapping_sha256(parameters),
        parameter_dtype="float64",
    )
    return spec, parameters


def _teacher_bundle():
    controller_spec, parameters = _controller_spec()
    initial_parameters_sha256 = parameter_mapping_sha256(parameters)
    initial_version = PolicyVersion("sdpo-test", 0, initial_parameters_sha256)
    controller = TeacherController(
        controller_spec,
        parameters,
        initial_version,
    )
    model = _TinyTeacherModel(parameters).eval()
    store, observation_handle = populated_observation_store()
    observation = store.resolve_record(observation_handle)
    sequence = len(TEACHER_TOKEN_IDS)
    input_ids = store.put_tensor(
        "sdpo.teacher.input_ids",
        torch.tensor([TEACHER_TOKEN_IDS], dtype=torch.long),
    )
    position_ids = store.put_tensor(
        "sdpo.teacher.position_ids", torch.arange(sequence).view(1, sequence)
    )
    attention_mask = store.put_tensor(
        "sdpo.teacher.attention_mask",
        torch.ones(1, sequence, dtype=torch.bool),
    )
    replay_record = TrajectoryReplayRecord(
        schema_version="trajectory-replay-v1",
        replay_id="sdpo-teacher-replay",
        trajectory_id="sdpo-trajectory",
        model=observation.model,
        behavior_policy=observation.condition.policy_version,
        source_visual=trajectory_source_visual(observation),
        observation_handles=(observation_handle,),
        tensors=TrajectoryReplayTensorRefs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            policy_attention_mask=attention_mask,
            reference_attention_mask=attention_mask,
            teacher_attention_mask=attention_mask,
        ),
    )
    replay_handle = store.put_replay(replay_record)
    context = build_teacher_context(
        spec=_context_spec(),
        context_id="feedback-context-0",
        trajectory_identity_sha256=hashlib.sha256(
            replay_record.trajectory_id.encode("utf-8")
        ).hexdigest(),
        feedback_identity_sha256=SHA5,
        feedback_token_indices=(0,),
        student_token_ids=STUDENT_TOKEN_IDS,
        teacher_token_ids=TEACHER_TOKEN_IDS,
        replay_handle=replay_handle,
        observation_handles=(observation_handle,),
        turns=(
            TeacherTurnAlignment(
                assistant_turn_index=0,
                student_token_indices=(1, 2),
                teacher_token_indices=(1, 2),
                visible_observation_handles=(),
            ),
            TeacherTurnAlignment(
                assistant_turn_index=1,
                student_token_indices=(4,),
                teacher_token_indices=(4,),
                visible_observation_handles=(observation_handle,),
            ),
            TeacherTurnAlignment(
                assistant_turn_index=2,
                student_token_indices=(6, 7),
                teacher_token_indices=(6, 7),
                visible_observation_handles=(observation_handle,),
            ),
        ),
    )
    artifact = replay_teacher_on_recorded_observations(
        adapter=_TeacherReplayAdapter(),
        model=model,
        store=store,
        replay_handle=replay_handle,
        controller=controller,
        teacher_policy_version=initial_version,
    )
    return context, artifact, controller, store


def _policy(
    current: torch.Tensor,
    *,
    mask: torch.Tensor = TARGET_MASK,
    behavior: torch.Tensor | None = None,
    proximal: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
) -> PolicyLogProbSet:
    behavior = current.detach() - 0.12 if behavior is None else behavior
    proximal = current.detach() + 0.07 if proximal is None else proximal
    reference = current.detach() - 0.19 if reference is None else reference
    return PolicyLogProbSet(
        behavior=RoleLogProbs(
            ComponentRole.BEHAVIOR,
            behavior.clone(),
            _version(0, "0"),
            LogProbSource.ROLLOUT_RECORDED,
            "9" * 64,
        ),
        proximal_old=RoleLogProbs(
            ComponentRole.PROXIMAL_OLD,
            proximal.clone(),
            _version(1, "1"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        current=RoleLogProbs(
            ComponentRole.CURRENT,
            current,
            _version(2, "2"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        reference=RoleLogProbs(
            ComponentRole.REFERENCE,
            reference.clone(),
            _version(0, "3"),
            LogProbSource.DETERMINISTIC_REPLAY,
            "9" * 64,
        ),
        policy_sampled_mask=mask.clone(),
    )


def _full_spec(
    divergence: FullVocabularyDivergence,
    *,
    alpha: float | None,
) -> SDPOSpec:
    controller_spec, _ = _controller_spec()
    return SDPOSpec(
        loss_mode=SDPOLossMode.FULL_VOCAB_DIVERGENCE,
        full_vocab_divergence=divergence,
        jsd_alpha=alpha,
        importance_sampling=ImportanceSamplingSpec(
            mode=ImportanceSamplingMode.NONE,
            denominator=None,
            minimum_ratio=None,
            maximum_ratio=None,
        ),
        distillation_coefficient=1.3,
        reference_kl=ReferenceKLSpec(
            estimator=ReferenceKLEstimator.K2_SQUARED_LOG_RATIO,
            coefficient=0.2,
        ),
        reduction=ReductionSpec(
            mode=LossReduction.TOKEN_MEAN,
            fixed_token_normalizer=None,
        ),
        teacher_controller=controller_spec,
        teacher_context=_context_spec(),
        distribution_tolerance=1.0e-10,
        stop_teacher_gradient=True,
    )


def _full_inputs():
    base = torch.arange(40, dtype=torch.float64).view(1, 8, 5)
    student_logits = ((base.remainder(7) - 3.0) / 5.0).clone().requires_grad_(True)
    student_log_probs = student_logits.log_softmax(dim=-1)
    token_ids = torch.tensor([STUDENT_TOKEN_IDS], dtype=torch.long)
    sampled_current = student_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    policy = _policy(sampled_current)
    context, artifact, controller, store = _teacher_bundle()
    inputs = SDPOInputs(
        policy=policy,
        teacher_contexts=(context,),
        teacher_replay_artifacts=(artifact,),
        distillation_target_mask=TARGET_MASK.clone(),
        student_full_log_probs=student_log_probs,
    )
    teacher_log_probs = torch.zeros_like(student_log_probs).detach()
    raw_teacher = artifact.logits[0].log_softmax(dim=-1)
    for turn in context.turns:
        for student_index, teacher_index in zip(
            turn.student_token_indices, turn.teacher_token_indices, strict=True
        ):
            teacher_log_probs[0, student_index] = raw_teacher[teacher_index - 1]
    return inputs, student_logits, teacher_log_probs, controller, store


@pytest.mark.parametrize(
    ("divergence", "alpha"),
    [
        (FullVocabularyDivergence.KL_TEACHER_TO_STUDENT, None),
        (FullVocabularyDivergence.KL_STUDENT_TO_TEACHER, None),
        (FullVocabularyDivergence.GENERALIZED_JENSEN_SHANNON, 0.35),
    ],
)
def test_full_vocabulary_sdpo_value_and_gradient_match_cpu_oracle(
    divergence: FullVocabularyDivergence,
    alpha: float | None,
) -> None:
    inputs, student_logits, teacher_log_probs, controller, store = _full_inputs()
    spec = _full_spec(divergence, alpha=alpha)
    result = compute_sdpo_loss(
        spec,
        inputs,
        teacher_controller=controller,
        observation_store=store,
    )

    assert inputs.student_full_log_probs is not None
    student_log_probs = inputs.student_full_log_probs
    student_probabilities = student_log_probs.exp()
    teacher_probabilities = teacher_log_probs.exp()
    if divergence is FullVocabularyDivergence.KL_TEACHER_TO_STUDENT:
        divergence_tokens = (
            teacher_probabilities * (teacher_log_probs - student_log_probs)
        ).sum(-1)
    elif divergence is FullVocabularyDivergence.KL_STUDENT_TO_TEACHER:
        divergence_tokens = (
            student_probabilities * (student_log_probs - teacher_log_probs)
        ).sum(-1)
    else:
        assert alpha is not None
        mixture = torch.logaddexp(
            student_log_probs
            + torch.log(torch.tensor(1.0 - alpha, dtype=torch.float64)),
            teacher_log_probs + torch.log(torch.tensor(alpha, dtype=torch.float64)),
        )
        divergence_tokens = (1.0 - alpha) * (
            student_probabilities * (student_log_probs - mixture)
        ).sum(-1) + alpha * (teacher_probabilities * (teacher_log_probs - mixture)).sum(
            -1
        )

    sampled_log_ratio = inputs.policy.current.values - inputs.policy.reference.values
    reference_k2 = 0.5 * sampled_log_ratio.square()
    expected = (1.3 * divergence_tokens + 0.2 * reference_k2)[TARGET_MASK].mean()
    torch.testing.assert_close(result.loss, expected)

    expected_gradient = torch.autograd.grad(
        expected, student_logits, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad(result.loss, student_logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert not teacher_log_probs.requires_grad
    assert torch.count_nonzero(result.per_token_loss[~TARGET_MASK]).item() == 0


def test_sampled_token_reverse_kl_with_clipped_importance_sampling_matches_oracle() -> (
    None
):
    current = torch.tensor(
        [[-1.2, -0.8, -1.5, -0.7, -1.0, -1.1, -0.4, -1.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    behavior = torch.tensor(
        [[-1.0, -1.4, -1.2, -0.7, -1.3, -1.1, -0.9, -1.0]],
        dtype=torch.float64,
    )
    policy = _policy(current, behavior=behavior)
    context, artifact, controller, store = _teacher_bundle()
    controller_spec, _ = _controller_spec()
    spec = SDPOSpec(
        loss_mode=SDPOLossMode.SAMPLED_TOKEN_REVERSE_KL_SCORE_FUNCTION,
        full_vocab_divergence=None,
        jsd_alpha=None,
        importance_sampling=ImportanceSamplingSpec(
            mode=ImportanceSamplingMode.CLIPPED,
            denominator=RatioDenominator.BEHAVIOR,
            minimum_ratio=0.7,
            maximum_ratio=1.15,
        ),
        distillation_coefficient=0.9,
        reference_kl=ReferenceKLSpec(
            estimator=ReferenceKLEstimator.K2_SQUARED_LOG_RATIO,
            coefficient=0.13,
        ),
        reduction=ReductionSpec(
            mode=LossReduction.TOKEN_MEAN,
            fixed_token_normalizer=None,
        ),
        teacher_controller=controller_spec,
        teacher_context=_context_spec(),
        distribution_tolerance=1.0e-8,
        stop_teacher_gradient=True,
    )
    inputs = SDPOInputs(
        policy=policy,
        teacher_contexts=(context,),
        teacher_replay_artifacts=(artifact,),
        distillation_target_mask=TARGET_MASK.clone(),
        student_full_log_probs=None,
    )

    result = compute_sdpo_loss(
        spec,
        inputs,
        teacher_controller=controller,
        observation_store=store,
    )
    teacher = torch.zeros_like(current)
    raw_teacher = artifact.logits[0].log_softmax(dim=-1)
    for turn in context.turns:
        for student_index, teacher_index in zip(
            turn.student_token_indices, turn.teacher_token_indices, strict=True
        ):
            teacher[0, student_index] = raw_teacher[
                teacher_index - 1, context.teacher_token_ids[teacher_index]
            ]
    importance = torch.exp(current.detach() - behavior).clamp(0.7, 1.15)
    score_function_surrogate = importance * (current.detach() - teacher) * current
    reference_k2 = 0.5 * (current - policy.reference.values).square()
    expected = (0.9 * score_function_surrogate + 0.13 * reference_k2)[
        TARGET_MASK
    ].mean()
    torch.testing.assert_close(result.loss, expected)
    expected_gradient = torch.autograd.grad(expected, current, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(result.loss, current)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual_gradient[~TARGET_MASK]).item() == 0


def test_teacher_replay_rejects_any_observation_handle_change() -> None:
    inputs, _, _, controller, store = _full_inputs()
    spec = _full_spec(FullVocabularyDivergence.KL_TEACHER_TO_STUDENT, alpha=None)
    wrong_context = replace(
        inputs.teacher_contexts[0],
        replay_handle=TrajectoryReplayHandle("forged-replay", SHA8),
    )
    mismatched = replace(inputs, teacher_contexts=(wrong_context,))
    with pytest.raises(ReplayMismatchError, match="context replay handle"):
        compute_sdpo_loss(
            spec,
            mismatched,
            teacher_controller=controller,
            observation_store=store,
        )


def test_ema_teacher_state_updates_and_checkpoint_round_trips_exactly() -> None:
    spec, initial = _controller_spec(TeacherRegularization.EMA_PARAMETERS)
    controller = TeacherController(
        spec,
        initial,
        PolicyVersion("sdpo-test", 0, parameter_mapping_sha256(initial)),
    )
    student = {"projection.weight": torch.tensor([4.0, 6.0], dtype=torch.float64)}
    applied = controller.update(
        student,
        PolicyVersion("sdpo-test", 1, parameter_mapping_sha256(student)),
    )
    assert applied
    torch.testing.assert_close(
        controller.teacher_parameters()["projection.weight"],
        torch.tensor([1.0, 3.0], dtype=torch.float64),
    )
    identity_before = controller.state_identity_sha256
    state = controller.state_dict()
    restored = TeacherController.from_state_dict(spec, state)
    assert restored.state_identity_sha256 == identity_before
    assert restored.update_count == 1
    torch.testing.assert_close(
        restored.teacher_parameters()["projection.weight"],
        controller.teacher_parameters()["projection.weight"],
    )

    external_copy = restored.teacher_parameters()
    external_copy["projection.weight"].add_(100.0)
    torch.testing.assert_close(
        restored.teacher_parameters()["projection.weight"],
        torch.tensor([1.0, 3.0], dtype=torch.float64),
    )


def test_trust_region_teacher_interpolates_logits_and_checkpoints_controller_state() -> (
    None
):
    spec, initial = _controller_spec(
        TeacherRegularization.TRUST_REGION_LOGIT_INTERPOLATION
    )
    controller = TeacherController(
        spec,
        initial,
        PolicyVersion("sdpo-test", 0, parameter_mapping_sha256(initial)),
    )
    reference_logits = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    feedback_logits = torch.tensor(
        [[4.0, 6.0]], dtype=torch.float64, requires_grad=True
    )
    teacher_logits = controller.interpolate_logits(reference_logits, feedback_logits)
    torch.testing.assert_close(
        teacher_logits, torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    )
    assert not teacher_logits.requires_grad

    student = {"projection.weight": torch.tensor([9.0, 11.0], dtype=torch.float64)}
    assert controller.update(
        student,
        PolicyVersion("sdpo-test", 1, parameter_mapping_sha256(student)),
    )
    restored = TeacherController.from_state_dict(spec, controller.state_dict())
    assert restored.update_calls == 1
    assert restored.update_count == 1
    torch.testing.assert_close(
        restored.interpolate_logits(reference_logits, feedback_logits),
        teacher_logits,
    )


def test_teacher_replay_rejects_forged_logits_after_artifact_creation() -> None:
    inputs, _, _, controller, store = _full_inputs()
    inputs.teacher_replay_artifacts[0].logits[0, 0, 0].add_(1.0)
    spec = _full_spec(FullVocabularyDivergence.KL_TEACHER_TO_STUDENT, alpha=None)
    with pytest.raises(ReplayMismatchError, match="logits checksum"):
        compute_sdpo_loss(
            spec,
            inputs,
            teacher_controller=controller,
            observation_store=store,
        )


def test_teacher_replay_rejects_stale_controller_state() -> None:
    inputs, _, _, controller, store = _full_inputs()
    student = {"projection.weight": torch.tensor([4.0, 6.0], dtype=torch.float64)}
    controller.update(
        student,
        PolicyVersion("sdpo-test", 1, parameter_mapping_sha256(student)),
    )
    spec = _full_spec(FullVocabularyDivergence.KL_TEACHER_TO_STUDENT, alpha=None)
    with pytest.raises(IdentityMismatchError, match="current state"):
        compute_sdpo_loss(
            spec,
            inputs,
            teacher_controller=controller,
            observation_store=store,
        )


def test_teacher_context_cannot_self_attest_forged_observation_handles() -> None:
    inputs, _, _, controller, store = _full_inputs()
    context = inputs.teacher_contexts[0]
    forged_handle = ObservationHandle("forged-observation", SHA8)
    forged_turns = tuple(
        replace(
            turn,
            visible_observation_handles=(forged_handle,)
            if turn.visible_observation_handles
            else (),
        )
        for turn in context.turns
    )
    forged_context = replace(
        context,
        observation_handles=(forged_handle,),
        turns=forged_turns,
    )
    forged_inputs = replace(inputs, teacher_contexts=(forged_context,))
    spec = _full_spec(FullVocabularyDivergence.KL_TEACHER_TO_STUDENT, alpha=None)
    with pytest.raises(ReplayMismatchError, match="exact rollout observation"):
        compute_sdpo_loss(
            spec,
            forged_inputs,
            teacher_controller=controller,
            observation_store=store,
        )


def test_full_vocabulary_student_distribution_must_require_gradients() -> None:
    inputs, _, _, _, _ = _full_inputs()
    assert inputs.student_full_log_probs is not None
    with pytest.raises(ValueError, match="must require gradients"):
        replace(inputs, student_full_log_probs=inputs.student_full_log_probs.detach())


def test_sdpo_importance_sampling_rejects_proximal_denominator() -> None:
    with pytest.raises(ValueError, match="actual behavior policy"):
        ImportanceSamplingSpec(
            mode=ImportanceSamplingMode.CLIPPED,
            denominator=RatioDenominator.PROXIMAL_OLD,
            minimum_ratio=0.8,
            maximum_ratio=1.2,
        )


def test_teacher_update_rejects_policy_version_parameter_hash_mismatch() -> None:
    spec, initial = _controller_spec()
    controller = TeacherController(
        spec,
        initial,
        PolicyVersion("sdpo-test", 0, parameter_mapping_sha256(initial)),
    )
    student = {"projection.weight": torch.tensor([4.0, 6.0], dtype=torch.float64)}
    with pytest.raises(IdentityMismatchError, match="supplied parameters"):
        controller.update(student, PolicyVersion("sdpo-test", 1, SHA8))


def test_pure_objective_registry_has_no_implicit_hybrid_and_specs_are_hashed() -> None:
    assert OBJECTIVE_REGISTRY.names == ("grpo", "sdpo")
    with pytest.raises(KeyError, match="unknown objective"):
        OBJECTIVE_REGISTRY.get("grpo_sdpo")

    first = _full_spec(FullVocabularyDivergence.GENERALIZED_JENSEN_SHANNON, alpha=0.35)
    second = replace(first, jsd_alpha=0.40)
    assert first.identity_sha256 != second.identity_sha256
    assert (
        first.identity_sha256
        == _full_spec(
            FullVocabularyDivergence.GENERALIZED_JENSEN_SHANNON,
            alpha=0.35,
        ).identity_sha256
    )
