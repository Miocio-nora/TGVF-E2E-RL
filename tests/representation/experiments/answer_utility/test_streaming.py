from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tgvf_rl.representation.experiments.answer_utility.streaming as streaming_module
import tgvf_rl.representation.experiments.answer_utility.trainer as trainer_module
from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.protocol.native import RenderedTranscript
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock
from tgvf_rl.representation.experiments.answer_utility.config import (
    AnswerSupervisionView,
)
from tgvf_rl.representation.experiments.answer_utility.controls import (
    AnswerUtilityControlRow,
    zero_visual_bundle,
)
from tgvf_rl.representation.experiments.answer_utility.native_pipeline import (
    AnswerUtilityReadoutGroup,
)
from tgvf_rl.representation.experiments.answer_utility.objective import (
    AnswerUtilityObjectiveConfig,
)
from tgvf_rl.representation.experiments.answer_utility.supervision import (
    NativeAnswerSupervision,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.streaming import (
    StreamingBackwardMetrics,
    StreamingGlobalNormalization,
    _StreamingCandidateGradients,
)
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


def _bundle(
    value: float,
    *,
    requires_grad: bool,
) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.full((1, 2, 4), value, requires_grad=requires_grad),
        deepstack=tuple(
            torch.full(
                (1, 2, 4),
                value * float(index + 2),
                requires_grad=requires_grad,
            )
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


def _request(value: float) -> InjectedForwardRequest:
    visual = _bundle(value, requires_grad=True)
    sequence = 5
    return InjectedForwardRequest(
        input_ids=torch.tensor(((1, 1, 2, 3, 4),), dtype=torch.long),
        attention_mask=torch.ones((1, sequence), dtype=torch.bool),
        position_ids=torch.arange(sequence).view(1, 1, sequence).expand(3, -1, -1),
        visual_blocks=(
            InjectedVisualBlock(
                kind="focused_d",
                positions=(0, 1),
                embeddings=visual.main,
                deepstack=visual.deepstack,
                deepstack_positions=((0, 1), (0, 1), (0, 1)),
            ),
        ),
        use_cache=False,
    )


def _answer_supervision(sample_id: str) -> NativeAnswerSupervision:
    token_ids = (1, 1, 2, 3)
    return NativeAnswerSupervision(
        sample_id=sample_id,
        context_kind="clean_d_only",
        transcript=RenderedTranscript(
            text="fixture answer",
            token_ids=token_ids,
            token_ids_sha256="fixture-token-sha",
            text_sha256="fixture-text-sha",
            chat_template_sha256="fixture-template-sha",
            tool_schema_sha256="fixture-tool-sha",
            tokenizer_length=8,
        ),
        input_ids=torch.tensor((token_ids,), dtype=torch.long),
        attention_mask=torch.ones((1, len(token_ids)), dtype=torch.bool),
        position_ids=torch.arange(len(token_ids))
        .view(1, 1, len(token_ids))
        .expand(3, -1, -1),
        labels=(EVIDENCE_IGNORE_INDEX, EVIDENCE_IGNORE_INDEX, 2, 3),
        answer_positions=(2,),
        eos_positions=(3,),
        source_positions=(),
        d_positions=(0, 1),
        answer_text="answer",
        evidence_field_injected=False,
    )


class _AnswerLogitAdapter:
    """Tiny family adapter whose answer logits depend on every D path."""

    def __init__(self) -> None:
        self.last_logits: torch.Tensor | None = None

    def forward_injected(
        self,
        model: object,
        request: InjectedForwardRequest,
    ) -> SimpleNamespace:
        del model
        block = request.visual_blocks[-1]
        score = block.embeddings.mean(dim=(1, 2))
        for branch in block.deepstack:
            score = score + branch.mean(dim=(1, 2))
        pattern = torch.zeros(
            (request.input_ids.shape[1], 8),
            dtype=score.dtype,
            device=score.device,
        )
        pattern[1, 2] = 1.0  # predicts the short-answer token at label position 2
        pattern[2, 3] = 2.0  # predicts native EOS at label position 3
        self.last_logits = score[:, None, None] * pattern[None, :, :]
        return SimpleNamespace(logits=self.last_logits)


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _TinyQwenContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyFrozenQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyQwenContainer()
        self.lm_head = nn.Linear(4, 8, bias=False)


def _frozen_qwen() -> _TinyFrozenQwen:
    model = _TinyFrozenQwen().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _answer_objective() -> AnswerUtilityObjectiveConfig:
    return AnswerUtilityObjectiveConfig(
        answer_weight=1.0,
        correct_vs_zero_weight=1.0,
        correct_vs_wrong_weight=1.0,
        existing_evidence_weight=0.0,
        existing_matrix_weight=0.0,
        norm_weight=0.0,
        comparison_margin=0.5,
        comparison_temperature=1.0,
    )


def _legacy_supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token_id if position in evidence_positions else EVIDENCE_IGNORE_INDEX
            for position, token_id in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


def _legacy_group() -> SameImageReadoutGroup:
    rows: list[RepresentationReadoutRow] = []
    candidates: list[RepresentationCandidateObservation] = []
    for index in range(2):
        sample_id = f"sample-{index}"
        token_ids = (1, 2, 2, 2, 2, 3, 5 + index, 7 + index)
        rows.append(
            RepresentationReadoutRow(
                sample_id=sample_id,
                image_group_key="image-1",
                source_visual_identity="source-sha",
                supervision=_legacy_supervision(token_ids),
                input_ids=torch.tensor((token_ids,), dtype=torch.long),
                attention_mask=torch.ones((1, len(token_ids)), dtype=torch.bool),
                position_ids=torch.arange(len(token_ids)).view(1, len(token_ids)),
                source_positions=(1, 2),
                d_positions=(3, 4),
            )
        )
        candidates.append(
            RepresentationCandidateObservation(
                sample_id=sample_id,
                image_group_key="image-1",
                source_visual_identity="source-sha",
                target_conditioning_provider=(
                    TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
                ),
                projection_identities=("main", "ds8", "ds16", "ds24"),
                visual=_bundle(float(index + 1), requires_grad=True),
            )
        )
    return SameImageReadoutGroup(
        image_group_key="image-1",
        source_visual_identity="source-sha",
        source_visual=_bundle(0.25, requires_grad=False),
        rows=tuple(rows),
        candidates=tuple(candidates),
    )


def test_answer_arm_batching_preserves_one_exact_context_and_all_visual_paths() -> None:
    requests = tuple(_request(value) for value in (1.0, 2.0, 3.0))

    batched = streaming_module._batch_identical_answer_requests(requests)

    assert batched.input_ids.shape == (3, 5)
    assert torch.equal(batched.input_ids, requests[0].input_ids.expand(3, -1))
    assert torch.equal(
        batched.attention_mask,
        requests[0].attention_mask.expand(3, -1),
    )
    assert batched.position_ids.shape == (3, 3, 5)
    block = batched.visual_blocks[0]
    assert block.embeddings.shape == (3, 2, 4)
    assert len(block.deepstack) == 3
    for batch_index, request in enumerate(requests):
        expected = request.visual_blocks[0]
        assert torch.equal(block.embeddings[batch_index], expected.embeddings[0])
        for actual_branch, expected_branch in zip(
            block.deepstack,
            expected.deepstack,
            strict=True,
        ):
            assert torch.equal(actual_branch[batch_index], expected_branch[0])

    changed_context = replace(
        requests[1],
        input_ids=torch.tensor(((1, 1, 2, 7, 4),), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="share one exact native context"):
        streaming_module._batch_identical_answer_requests(
            (requests[0], changed_context)
        )


def test_answer_and_eos_nll_with_margins_push_correct_and_wrong_in_opposite_directions() -> (
    None
):
    correct = _bundle(0.2, requires_grad=True)
    wrong = _bundle(-0.1, requires_grad=True)
    controls = AnswerUtilityControlRow(
        sample_id="sample-0",
        correct_source_sample_id="sample-0",
        wrong_source_sample_id="sample-1",
        correct=correct,
        zero=zero_visual_bundle(correct),
        wrong=wrong,
    )
    supervision = _answer_supervision("sample-0")
    group = SimpleNamespace(
        answer_supervisions=(supervision,),
        controls=(controls,),
        legacy=SimpleNamespace(source_visual=None),
    )
    adapter = _AnswerLogitAdapter()

    result = streaming_module._score_answer_rows(
        adapter,  # type: ignore[arg-type]
        object(),
        group,  # type: ignore[arg-type]
        objective=_answer_objective(),
        normalization=StreamingGlobalNormalization(
            matrix_valid_rows=1,
            l_gen_samples=1,
        ),
    )

    assert supervision.supervised_positions == (2, 3)
    assert result.qwen_forward_batch_sizes == (3,)
    assert adapter.last_logits is not None
    expected_correct = F.cross_entropy(
        adapter.last_logits[0, (1, 2)],
        torch.tensor((2, 3), dtype=torch.long),
    )
    assert torch.allclose(result.correct_nll[0], expected_correct)
    assert result.zero_nll is not None
    assert result.wrong_nll is not None
    assert result.correct_vs_zero is not None
    assert result.correct_vs_wrong is not None

    gradients = {id(tensor): gradient for tensor, gradient in result.gradients}
    for tensor in (correct.main, *correct.deepstack):
        assert torch.all(gradients[id(tensor)] < 0)
    for tensor in (wrong.main, *wrong.deepstack):
        assert torch.all(gradients[id(tensor)] > 0)
    assert all(
        not tensor.requires_grad
        for tensor in (controls.zero.main, *controls.zero.deepstack)
    )


@dataclass(frozen=True)
class _FakeLegacyScores:
    candidate_output_gradients: tuple[_StreamingCandidateGradients, ...]


def test_answer_boundary_vjp_is_merged_before_exactly_one_legacy_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_group = _legacy_group()
    controls = tuple(
        AnswerUtilityControlRow(
            sample_id=candidate.sample_id,
            correct_source_sample_id=candidate.sample_id,
            wrong_source_sample_id=legacy_group.candidates[(index + 1) % 2].sample_id,
            correct=candidate.visual,
            zero=zero_visual_bundle(candidate.visual),
            wrong=legacy_group.candidates[(index + 1) % 2].visual,
        )
        for index, candidate in enumerate(legacy_group.candidates)
    )
    group = AnswerUtilityReadoutGroup(
        legacy=legacy_group,
        answer_supervisions=tuple(
            SimpleNamespace(sample_id=row.sample_id, context_kind="clean_d_only")
            for row in legacy_group.rows
        ),
        controls=controls,
        supervision_view=AnswerSupervisionView.CLEAN_D_ONLY,
        requires_zero_control=True,
        requires_wrong_control=True,
    )
    boundary = legacy_group.candidates[0].visual.main
    calls = {"score": 0, "backward": 0}

    def fake_score(*args: object, **kwargs: object) -> _FakeLegacyScores:
        del args, kwargs
        calls["score"] += 1
        return _FakeLegacyScores(
            candidate_output_gradients=tuple(
                _StreamingCandidateGradients(
                    weighted_readout=tuple(
                        torch.full_like(tensor, 2.0)
                        for tensor in (
                            candidate.visual.main,
                            *candidate.visual.deepstack,
                        )
                    )
                )
                for candidate in legacy_group.candidates
            )
        )

    def fake_answer_score(*args: object, **kwargs: object):
        del args, kwargs
        return streaming_module._AnswerScoreMaterialization(
            gradients=((boundary, torch.full_like(boundary, 3.0)),),
            correct_nll=torch.tensor((1.0, 2.0)),
            zero_nll=torch.tensor((2.0, 3.0)),
            wrong_nll=torch.tensor((3.0, 4.0)),
            correct_vs_zero=torch.tensor((0.2, 0.3)),
            correct_vs_wrong=torch.tensor((0.1, 0.2)),
            qwen_forward_batch_sizes=(3, 3),
        )

    def fake_backward(
        _family: object,
        _model: object,
        passed_group: SameImageReadoutGroup,
        scores: _FakeLegacyScores,
        **kwargs: object,
    ) -> StreamingBackwardMetrics:
        del kwargs
        calls["backward"] += 1
        outputs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for candidate, payload in zip(
            passed_group.candidates,
            scores.candidate_output_gradients,
            strict=True,
        ):
            tensors = (candidate.visual.main, *candidate.visual.deepstack)
            outputs.extend(tensors)
            gradients.extend(payload.weighted_readout)
        norm = torch.stack(tuple(tensor.square().mean() for tensor in outputs)).sum()
        norm_gradients = torch.autograd.grad(norm, tuple(outputs))
        torch.autograd.backward(
            tuple(outputs),
            grad_tensors=tuple(
                gradient + 0.1 * norm_gradient
                for gradient, norm_gradient in zip(
                    gradients,
                    norm_gradients,
                    strict=True,
                )
            ),
        )
        return StreamingBackwardMetrics(
            matrix_ce_numerator=torch.tensor(1.0),
            l_gen_numerator=torch.tensor(2.0),
            norm_numerator=torch.tensor(3.0),
            local_row_count=2,
            local_sample_count=2,
            weighted_local_mean=torch.tensor(4.0),
            weighted_norm_local_mean=torch.tensor(0.3),
        )

    monkeypatch.setattr(
        streaming_module, "score_streaming_same_image_group", fake_score
    )
    monkeypatch.setattr(streaming_module, "_score_answer_rows", fake_answer_score)
    monkeypatch.setattr(
        streaming_module,
        "backward_streaming_same_image_group",
        fake_backward,
    )
    model = _frozen_qwen()
    objective = AnswerUtilityObjectiveConfig(
        answer_weight=1.0,
        correct_vs_zero_weight=1.0,
        correct_vs_wrong_weight=1.0,
        existing_evidence_weight=0.25,
        existing_matrix_weight=0.25,
        norm_weight=0.1,
        comparison_margin=0.5,
        comparison_temperature=1.0,
    )
    legacy_objective = RepresentationObjectiveConfigV2(
        identity="answer-utility-hook-test",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=0.25,
        l_gen_weight=0.25,
        norm_weight=0.1,
    )

    metrics = streaming_module.backward_answer_utility_group(
        SimpleNamespace(),  # type: ignore[arg-type]
        model,
        group,
        objective=objective,
        legacy_objective=legacy_objective,
        normalization=StreamingGlobalNormalization(
            matrix_valid_rows=2,
            l_gen_samples=2,
        ),
    )

    assert calls == {"score": 1, "backward": 1}
    assert boundary.grad is not None
    # 2 legacy + 3 answer exactly once, plus the independent norm VJP.
    expected = torch.full_like(boundary, 5.0) + 0.1 * (
        2.0 * boundary.detach() / boundary.numel()
    )
    assert boundary.grad is not None
    assert torch.allclose(boundary.grad, expected)
    assert metrics.answer_qwen_forward_batch_sizes == (3, 3)
    assert metrics.correct_answer_nll_numerator is not None
    assert metrics.correct_answer_nll_numerator.item() == pytest.approx(3.0)


def test_no_answer_continuation_executes_only_the_legacy_streaming_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_group = _legacy_group()
    group = AnswerUtilityReadoutGroup(
        legacy=legacy_group,
        answer_supervisions=(),
        controls=(),
        supervision_view=AnswerSupervisionView.NONE,
        requires_zero_control=False,
        requires_wrong_control=False,
    )
    calls = {"score": 0, "backward": 0}

    def fake_score(*args: object, **kwargs: object) -> _FakeLegacyScores:
        del args, kwargs
        calls["score"] += 1
        return _FakeLegacyScores(candidate_output_gradients=())

    def fake_backward(*args: object, **kwargs: object) -> StreamingBackwardMetrics:
        del args, kwargs
        calls["backward"] += 1
        return StreamingBackwardMetrics(
            matrix_ce_numerator=torch.tensor(2.0),
            l_gen_numerator=torch.tensor(4.0),
            norm_numerator=torch.tensor(6.0),
            local_row_count=2,
            local_sample_count=2,
            weighted_local_mean=torch.tensor(3.0),
            weighted_norm_local_mean=torch.tensor(0.3),
        )

    monkeypatch.setattr(
        streaming_module, "score_streaming_same_image_group", fake_score
    )
    monkeypatch.setattr(
        streaming_module,
        "backward_streaming_same_image_group",
        fake_backward,
    )
    monkeypatch.setattr(
        streaming_module,
        "_score_answer_rows",
        lambda *args, **kwargs: pytest.fail("no-answer path scored answer rows"),
    )
    objective = AnswerUtilityObjectiveConfig(
        answer_weight=0.0,
        correct_vs_zero_weight=0.0,
        correct_vs_wrong_weight=0.0,
        existing_evidence_weight=1.0,
        existing_matrix_weight=1.0,
        norm_weight=0.1,
        comparison_margin=0.0,
        comparison_temperature=1.0,
    )
    legacy_objective = RepresentationObjectiveConfigV2(
        identity="answer-utility-e0-continuation-test",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
    )

    metrics = streaming_module.backward_answer_utility_group(
        SimpleNamespace(),  # type: ignore[arg-type]
        _frozen_qwen(),
        group,
        objective=objective,
        legacy_objective=legacy_objective,
        normalization=StreamingGlobalNormalization(
            matrix_valid_rows=2,
            l_gen_samples=2,
        ),
    )

    assert calls == {"score": 1, "backward": 1}
    assert metrics.local_answer_sample_count == 0
    assert metrics.answer_qwen_forward_batch_sizes == ()
    assert metrics.correct_answer_nll_numerator is None


def test_active_answer_objective_rejects_missing_answer_metrics() -> None:
    metrics = streaming_module.AnswerUtilityStreamingMetrics(
        legacy=StreamingBackwardMetrics(
            matrix_ce_numerator=torch.tensor(2.0),
            l_gen_numerator=torch.tensor(4.0),
            norm_numerator=torch.tensor(6.0),
            local_row_count=2,
            local_sample_count=2,
            weighted_local_mean=torch.tensor(3.0),
            weighted_norm_local_mean=torch.tensor(0.3),
        ),
        correct_answer_nll_numerator=None,
        zero_answer_nll_numerator=None,
        wrong_answer_nll_numerator=None,
        correct_vs_zero_numerator=None,
        correct_vs_wrong_numerator=None,
        local_answer_sample_count=0,
        answer_qwen_forward_batch_sizes=(),
    )

    with pytest.raises(RuntimeError, match="sample count differs"):
        trainer_module._accumulate_metrics(
            torch.zeros(8, dtype=torch.float64),
            metrics,
            objective=_answer_objective(),
            expected_sample_count=2,
        )
