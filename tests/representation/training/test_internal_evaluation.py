from __future__ import annotations

from hashlib import sha256
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.base import InjectedForwardRequest, InjectedVisualBlock
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training.internal_evaluation import (
    DETERMINISTIC_RANDOM_D_ALGORITHM,
    NativeCausalValueFlipRequest,
    NativeCausalValueFlipOutput,
    NativeCounterfactualCase,
    NativeDOnlyContext,
    NativeGenerationForward,
    NativeFreeContinuationRequest,
    NativeFreeContinuationOutput,
    NativeTeacherForcedForward,
    RepresentationInternalEvaluationIdentity,
    _deterministic_random_visual_bundle,
    create_injected_native_counterfactual_evaluator,
    run_representation_internal_evaluation,
    save_representation_internal_evaluation_report_atomic,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.readout import (
    RepresentationAttentionTensorBundle,
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


class _ToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens.reshape(-1, 16))


def _projection(identity: str) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        _ToyMerger(),
        identity=identity,
        input_dim=4,
        output_dim=6,
        spatial_merge_size=2,
    )


def _adapter() -> TGVFAdapter:
    torch.manual_seed(500)
    return TGVFAdapter(
        d_lm=6,
        d_v=4,
        attn_dim=5,
        main_projection=_projection("main"),
        deepstack_projections=tuple(
            _projection(f"branch-{layer}") for layer in (8, 16, 24)
        ),
        branch_layers=(8, 16, 24),
    )


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 6)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds,
        visual_pos_masks,
        deepstack_visual_embeds,
        **kwargs,
    ):
        hidden = inputs_embeds.clone()
        for branch in deepstack_visual_embeds:
            hidden = hidden.clone()
            hidden[visual_pos_masks] += branch
        hidden = hidden + hidden.cumsum(dim=1) * 0.03
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyContainer()
        self.lm_head = nn.Linear(6, 64, bias=False)


def _qwen() -> _TinyQwen:
    torch.manual_seed(501)
    model = _TinyQwen()
    model.requires_grad_(False)
    model.eval()
    return model


class _RecordingFamilyAdapter(Qwen3VLAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.visual_block_contracts: list[tuple[tuple[str, int], ...]] = []

    def forward_injected(self, model, request):
        self.visual_block_contracts.append(
            tuple((block.kind, len(block.deepstack)) for block in request.visual_blocks)
        )
        return super().forward_injected(model, request)


def _sample(group: str, member: int) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"{group}-{member}",
        image=f"/{group}.png",
        image_id=group,
        question="What local value is visible?",
        target=f"region-{member}",
        evidence_description=f"The local value is {group.upper()}{member}.",
        evidence_type="text" if member == 0 else "symbol",
        source_profile=f"profile-{group}",
        answer_type="short",
        visual_difficulty="easy" if member == 0 else "hard",
    )


def _sample_groups() -> tuple[tuple[RepresentationTrainingSample, ...], ...]:
    return tuple(tuple(_sample(group, index) for index in range(2)) for group in "ab")


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token_id if index in evidence_positions else EVIDENCE_IGNORE_INDEX
            for index, token_id in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


class _GroupBuilder:
    def __init__(self, *, retain_attention: bool = True) -> None:
        self.retain_attention = retain_attention
        self.calls: list[tuple[tuple[str, ...], bool, bool]] = []

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        assert collective_candidate_count == len(samples)
        self.calls.append(
            (
                tuple(sample.sample_id for sample in samples),
                torch.is_grad_enabled(),
                adapter.training,
            )
        )
        group_offset = 0.0 if samples[0].image_group_key == "a" else 0.4
        source = RepresentationVisualTensorBundle(
            main=torch.arange(12, dtype=torch.float32).reshape(1, 2, 6) / 20
            + group_offset,
            deepstack=tuple(
                torch.arange(12, dtype=torch.float32).reshape(1, 2, 6) / 25
                + group_offset
                + branch * 0.1
                for branch in range(3)
            ),
            branch_layers=(8, 16, 24),
        )
        rows = []
        candidates = []
        for member, sample in enumerate(samples):
            token_ids = (
                1,
                2,
                2,
                2,
                2,
                3,
                10 + member * 2,
                11 + member * 2,
            )
            rows.append(
                RepresentationReadoutRow(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=f"source-{sample.image_group_key}",
                    supervision=_supervision(token_ids),
                    input_ids=torch.tensor([token_ids], dtype=torch.long),
                    attention_mask=torch.ones(1, 8, dtype=torch.bool),
                    position_ids=torch.arange(8).view(1, 8),
                    source_positions=(1, 2),
                    d_positions=(3, 4),
                )
            )
            target = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 30
            target = target + group_offset + member * 0.2
            visual = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
            visual = visual + group_offset
            output = adapter(
                target_hidden_states=target,
                pre_merge_visual_tokens=visual,
                deepstack_pre_merge_visual_tokens=tuple(
                    visual + branch * 0.15 for branch in range(3)
                ),
            )
            candidates.append(
                RepresentationCandidateObservation(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=f"source-{sample.image_group_key}",
                    target_conditioning_provider=(
                        TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
                    ),
                    projection_identities=(
                        output.metadata.main_projection_identity,
                        *output.metadata.deepstack_projection_identities,
                    ),
                    visual=RepresentationVisualTensorBundle(
                        main=output.main_d.unsqueeze(0),
                        deepstack=tuple(
                            branch.unsqueeze(0)
                            for branch in output.deepstack_visual_embeds
                        ),
                        branch_layers=output.metadata.branch_layers,
                    ),
                    attention=(
                        RepresentationAttentionTensorBundle(
                            main=(
                                output.main_attention.target_to_visual_attention.detach().clone()
                            ),
                            deepstack=tuple(
                                branch.target_to_visual_attention.detach().clone()
                                for branch in output.deepstack_attention
                            ),
                            branch_layers=output.metadata.branch_layers,
                        )
                        if self.retain_attention
                        else None
                    ),
                )
            )
        return SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=f"source-{samples[0].image_group_key}",
            source_visual=source,
            rows=tuple(rows),
            candidates=tuple(candidates),
        )


def _bundle(value: float) -> RepresentationVisualTensorBundle:
    return RepresentationVisualTensorBundle(
        main=torch.arange(12, dtype=torch.float32).reshape(1, 2, 6) + value,
        deepstack=tuple(
            torch.arange(12, dtype=torch.float32).reshape(1, 2, 6) + value + index
            for index in range(3)
        ),
        branch_layers=(8, 16, 24),
    )


def _native_case() -> NativeCounterfactualCase:
    context = NativeDOnlyContext(
        context_id="fresh-d-only-context",
        transcript_identity="native-counterfactual-transcript-v1",
        family="qwen3_vl",
        input_ids=torch.tensor([[1, 2, 2, 3, 4]], dtype=torch.long),
        attention_mask=torch.ones(1, 5, dtype=torch.bool),
        position_ids=torch.arange(5).view(1, 1, 5).expand(3, 1, 5),
        d_positions=(1, 2),
        image_grid_thw=((1, 2, 4),),
    )
    return NativeCounterfactualCase(
        case_id="flip-0",
        pair_identity="counterfactual-pair-sha",
        sample_a_id="counterfactual-a",
        sample_b_id="counterfactual-b",
        expected_value_a="OPEN",
        expected_value_b="CLOSED",
        observation_a_identity="observation-a",
        observation_b_identity="observation-b",
        context=context,
        observation_a=_bundle(0.5),
        observation_b=_bundle(1.5),
    )


class _NativeEvaluators:
    def __init__(self) -> None:
        self.causal_calls = []
        self.continuation_calls = []

    def causal(self, request):
        self.causal_calls.append(request)
        assert request.case.context.source_image_positions == ()
        assert not request.case.context.pre_d_text_kv_reused
        assert len(request.case.observation_a.deepstack) == 3
        return NativeCausalValueFlipOutput(
            case_id=request.case.case_id,
            observation_a_log_odds_a_over_b=2.5,
            observation_b_log_odds_a_over_b=-1.75,
        )

    def continuation(self, request):
        self.continuation_calls.append(request)
        assert request.context.cache_mode == "no_cache"
        assert len(request.observation.deepstack) == 3
        return NativeFreeContinuationOutput(
            case_id=request.case_id,
            variant=request.variant,
            generated_token_ids=(20, 21),
            generated_text=f"The value is {request.expected_value}.",
            extracted_value=request.expected_value,
            stop_reason="natural_stop",
        )


class _TinyNativeMaterializer:
    _value_ids = {"OPEN": (30,), "CLOSED": (31,)}

    def value_token_ids(
        self, context: NativeDOnlyContext, value: str
    ) -> tuple[int, ...]:
        assert context.source_image_positions == ()
        return self._value_ids[value]

    @staticmethod
    def _request(
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        suffix: tuple[int, ...],
    ) -> InjectedForwardRequest:
        suffix_tensor = torch.tensor([suffix], dtype=torch.long)
        input_ids = torch.cat((context.input_ids, suffix_tensor), dim=1)
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        block = InjectedVisualBlock(
            kind="focused_d",
            positions=context.d_positions,
            embeddings=observation.main,
            deepstack=observation.deepstack,
            deepstack_positions=tuple(
                context.d_positions for _ in observation.deepstack
            ),
        )
        return InjectedForwardRequest(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            position_ids=positions,
            visual_blocks=(block,),
            use_cache=False,
        )

    def teacher_forced(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        continuation_token_ids: tuple[int, ...],
    ) -> NativeTeacherForcedForward:
        request = self._request(
            context=context,
            observation=observation,
            suffix=continuation_token_ids,
        )
        start = context.input_ids.shape[1]
        return NativeTeacherForcedForward(
            request=request,
            continuation_positions=tuple(
                range(start, start + len(continuation_token_ids))
            ),
        )

    def generation_step(
        self,
        *,
        context: NativeDOnlyContext,
        observation: RepresentationVisualTensorBundle,
        generated_token_ids: tuple[int, ...],
    ) -> NativeGenerationForward:
        request = self._request(
            context=context,
            observation=observation,
            suffix=generated_token_ids,
        )
        return NativeGenerationForward(
            request=request,
            next_token_logit_position=request.input_ids.shape[1] - 1,
        )

    def decode_generated(self, token_ids: tuple[int, ...]) -> str:
        return "tokens:" + ",".join(str(token_id) for token_id in token_ids)

    def extract_expected_value(
        self, generated_text: str, expected_value: str
    ) -> str | None:
        expected_id = self._value_ids[expected_value][0]
        return expected_value if str(expected_id) in generated_text else None


def _identity() -> RepresentationInternalEvaluationIdentity:
    return RepresentationInternalEvaluationIdentity(
        evaluation_id="tiny-internal-evaluation-v1",
        model_identity="tiny-qwen3-vl",
        checkpoint_identity="adapter-checkpoint-17",
        data_manifest_sha256=sha256(b"tiny-eval-data").hexdigest(),
        prompt_identity="native-prompt-tbd-smoke-only",
        target_conditioning_provider=(
            TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
        ),
        random_seed=1234,
    )


def _run(*, builder: _GroupBuilder | None = None):
    adapter = _adapter()
    adapter.train()
    qwen = _qwen()
    family = _RecordingFamilyAdapter()
    group_builder = builder or _GroupBuilder()
    native = _NativeEvaluators()
    report = run_representation_internal_evaluation(
        identity=_identity(),
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=family,
        sample_groups=_sample_groups(),
        group_builder=group_builder,
        native_counterfactual_cases=(_native_case(),),
        causal_value_flip_evaluator=native.causal,
        free_continuation_evaluator=native.continuation,
    )
    return report, adapter, qwen, family, group_builder, native


def test_internal_runner_executes_all_controls_health_and_native_callbacks(
    tmp_path,
) -> None:
    report, adapter, qwen, family, builder, native = _run()

    assert len(report.groups) == 2
    assert len(report.samples) == 4
    assert report.readout.sample_count == 4
    assert report.readout.complete_control_sample_count == 4
    assert report.query.group_count == 2
    assert report.query.sample_count == 4
    assert report.execution.qwen_frozen
    assert not report.execution.second_full_qwen_forward_used
    assert report.execution.visual_swap_unit == "atomic_main_and_all_deepstack_v1"
    assert all(len(group.query.nll_matrix) == 2 for group in report.groups)
    assert all(
        len(row) == 2 for group in report.groups for row in group.query.nll_matrix
    )
    assert all(record.nlls.wrong_same_image_d is not None for record in report.samples)
    assert all(
        record.nlls.wrong_different_image_d is not None for record in report.samples
    )
    assert len({record.controls.random_d_seed for record in report.samples}) == 4
    assert all(
        record.controls.random_d_algorithm == DETERMINISTIC_RANDOM_D_ALGORITHM
        for record in report.samples
    )
    assert report.health.main.sample_count == 4
    assert report.health.main_attention.observation_count == 4
    assert tuple(branch.branch_layer for branch in report.health.branches) == (
        8,
        16,
        24,
    )
    assert all(
        branch.representation.sample_count == 4 for branch in report.health.branches
    )
    assert all(
        branch.attention.observation_count == 4 for branch in report.health.branches
    )
    assert {metric.dimension for metric in report.grouped_metrics} == {
        "evidence_type",
        "source_profile",
        "answer_type",
        "visual_difficulty",
    }

    assert report.native_counterfactuals[0].expected_direction_flip
    assert report.native_counterfactuals[0].log_odds_separation == pytest.approx(4.25)
    assert all(
        continuation.value_consistent and continuation.healthy_termination
        for continuation in report.native_counterfactuals[0].continuations
    )
    assert len(native.causal_calls) == 1
    assert [call.variant for call in native.continuation_calls] == [
        "value_a",
        "value_b",
    ]

    assert builder.calls == [
        (("a-0", "a-1"), False, False),
        (("b-0", "b-1"), False, False),
    ]
    assert adapter.training
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert not qwen.training
    assert all(not parameter.requires_grad for parameter in qwen.parameters())
    assert any(len(contract) == 1 for contract in family.visual_block_contracts)
    assert all(
        all(branch_count == 3 for _kind, branch_count in contract)
        for contract in family.visual_block_contracts
    )

    artifact_path = (tmp_path / "internal-evaluation.json").resolve()
    artifact = save_representation_internal_evaluation_report_atomic(
        report, artifact_path
    )
    raw = artifact_path.read_bytes()
    payload = json.loads(raw)
    assert artifact.payload_sha256 == sha256(raw).hexdigest()
    assert artifact.byte_count == len(raw)
    assert payload["identity"]["checkpoint_identity"] == "adapter-checkpoint-17"
    assert (
        payload["identity"]["data_manifest_sha256"]
        == sha256(b"tiny-eval-data").hexdigest()
    )
    assert len(payload["groups"][0]["query"]["nll_matrix"]) == 2
    with pytest.raises(FileExistsError):
        save_representation_internal_evaluation_report_atomic(report, artifact_path)


def test_random_d_is_deterministic_and_preserves_atomic_tensor_health() -> None:
    visual = _bundle(0.25)
    first = _deterministic_random_visual_bundle(visual, seed=17)
    second = _deterministic_random_visual_bundle(visual, seed=17)
    different = _deterministic_random_visual_bundle(visual, seed=18)

    for original, randomized, repeated in zip(
        (visual.main, *visual.deepstack),
        (first.main, *first.deepstack),
        (second.main, *second.deepstack),
        strict=True,
    ):
        assert torch.equal(randomized, repeated)
        assert torch.equal(original.sort(dim=-1).values, randomized.sort(dim=-1).values)
        assert torch.equal(
            torch.linalg.vector_norm(original, dim=-1),
            torch.linalg.vector_norm(randomized, dim=-1),
        )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            (first.main, *first.deepstack),
            (different.main, *different.deepstack),
            strict=True,
        )
    )


def test_concrete_native_evaluator_executes_real_injected_qwen_forwards() -> None:
    qwen = _qwen()
    family = _RecordingFamilyAdapter()
    evaluator = create_injected_native_counterfactual_evaluator(
        model=qwen,
        family_adapter=family,
        materializer=_TinyNativeMaterializer(),
        eos_token_ids=(63,),
        max_new_tokens=2,
    )
    case = _native_case()

    causal = evaluator.causal_value_flip(NativeCausalValueFlipRequest(case))
    continuation = evaluator.free_continuation(
        NativeFreeContinuationRequest(
            case_id=case.case_id,
            variant="value_a",
            expected_value=case.expected_value_a,
            context=case.context,
            observation_identity=case.observation_a_identity,
            observation=case.observation_a,
        )
    )

    assert math.isfinite(causal.observation_a_log_odds_a_over_b)
    assert math.isfinite(causal.observation_b_log_odds_a_over_b)
    assert 1 <= len(continuation.generated_token_ids) <= 2
    assert continuation.generated_text.startswith("tokens:")
    assert continuation.stop_reason in {"natural_stop", "length_cap"}
    assert len(family.visual_block_contracts) >= 5
    assert all(
        contract == (("focused_d", 3),) for contract in family.visual_block_contracts
    )

    report = run_representation_internal_evaluation(
        identity=_identity(),
        adapter=_adapter(),
        qwen_model=qwen,
        family_adapter=family,
        sample_groups=_sample_groups(),
        group_builder=_GroupBuilder(),
        native_counterfactual_cases=(case,),
        causal_value_flip_evaluator=evaluator.causal_value_flip,
        free_continuation_evaluator=evaluator.free_continuation,
    )
    assert len(report.native_counterfactuals) == 1
    assert len(report.native_counterfactuals[0].continuations) == 2


def test_internal_runner_fails_closed_without_attention_or_fresh_native_context() -> (
    None
):
    with pytest.raises(ValueError, match="remove source images"):
        NativeDOnlyContext(
            context_id="bad",
            transcript_identity="bad-transcript",
            family="qwen3_vl",
            input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
            position_ids=torch.arange(4).view(1, 1, 4).expand(3, 1, 4),
            d_positions=(1, 2),
            image_grid_thw=((1, 2, 4),),
            source_image_positions=(0,),
        )

    with pytest.raises(ValueError, match="retained Adapter attention"):
        _run(builder=_GroupBuilder(retain_attention=False))
