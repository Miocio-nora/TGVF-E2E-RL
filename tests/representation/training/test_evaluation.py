from __future__ import annotations

from hashlib import sha256
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tgvf_rl.representation.training.evaluation as evaluation_module
from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training.evaluation import (
    _isolated_validation_rng,
    evaluate_representation_validation_event,
)
from tgvf_rl.representation.training.losses import (
    EVIDENCE_IGNORE_INDEX,
    MatrixCEScoreMode,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveConfigV3,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.sampling import (
    SameImageBatchSampler,
    same_image_group_owner,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.transcript import ModelEvidenceSupervision


DATA_SHA256 = sha256(b"deterministic-validation-data").hexdigest()


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
    torch.manual_seed(801)
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
        self.embed_tokens = nn.Embedding(24, 6)

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
        hidden = hidden + hidden.sum(dim=1, keepdim=True) * 0.04
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyContainer()
        self.lm_head = nn.Linear(6, 24, bias=False)


def _qwen(*, initially_trainable: bool = True) -> _TinyQwen:
    torch.manual_seed(802)
    model = _TinyQwen()
    model.requires_grad_(initially_trainable)
    model.train(initially_trainable)
    return model


def _sample(group: str, member: int) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"{group}-{member}",
        image=f"/{group}.png",
        image_id=group,
        question="What is shown?",
        target=f"target-{member}",
        evidence_description=f"evidence-{member}",
        short_answer=f"answer-{member}",
    )


def _samples(groups: tuple[str, ...] = ("a", "b", "c")):
    return tuple(_sample(group, member) for group in groups for member in range(2))


def _supervision(token_ids: tuple[int, ...]) -> ModelEvidenceSupervision:
    evidence_positions = (6, 7)
    return ModelEvidenceSupervision(
        family="qwen3_vl",
        model_token_ids=token_ids,
        labels=tuple(
            token if index in evidence_positions else EVIDENCE_IGNORE_INDEX
            for index, token in enumerate(token_ids)
        ),
        evidence_token_positions=evidence_positions,
        visual_model_positions=(1, 2, 3, 4),
        canonical_to_model_positions=((0,), (1, 2), (3, 4), (5,), (6,), (7,)),
    )


class _RecordingGroupBuilder:
    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.calls: list[tuple[tuple[str, ...], bool, bool, float, float]] = []
        self.collective_counts: list[int] = []
        self.padding_counts: list[int] = []

    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        if collective_candidate_count < len(samples):
            raise ValueError("collective candidate count cannot be smaller than real K")
        if self.reverse:
            samples = tuple(reversed(samples))
        self.collective_counts.append(collective_candidate_count)
        python_draw = random.random()
        torch_draw = float(torch.rand(()).item())
        self.calls.append(
            (
                tuple(sample.sample_id for sample in samples),
                torch.is_grad_enabled(),
                adapter.training,
                python_draw,
                torch_draw,
            )
        )
        rows = []
        candidates = []
        source = RepresentationVisualTensorBundle(
            main=torch.full((1, 2, 6), 0.2),
            deepstack=tuple(torch.full((1, 2, 6), 0.05) for _ in range(3)),
            branch_layers=(8, 16, 24),
        )
        for index, sample in enumerate(samples):
            token_ids = (1, 2, 2, 2, 2, 3, 7 + index * 2, 8 + index * 2)
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
                    source_key_block_query_start=3,
                )
            )
            target = torch.full((3, 6), 0.1 + index * 0.1 + python_draw * 0.01)
            visual = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
            visual = visual + torch_draw * 0.01
            output = adapter(
                target_hidden_states=target,
                pre_merge_visual_tokens=visual,
                deepstack_pre_merge_visual_tokens=tuple(
                    visual + branch_index * 0.1 for branch_index in range(3)
                ),
            )
            candidates.append(
                RepresentationCandidateObservation(
                    sample_id=sample.sample_id,
                    image_group_key=sample.image_group_key,
                    source_visual_identity=f"source-{sample.image_group_key}",
                    target_conditioning_provider=(
                        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
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
                )
            )
        padding = []
        for _ in range(collective_candidate_count - len(samples)):
            target = torch.full((3, 6), 0.1 + python_draw * 0.01)
            visual = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
            visual = visual + torch_draw * 0.01
            output = adapter(
                target_hidden_states=target,
                pre_merge_visual_tokens=visual,
                deepstack_pre_merge_visual_tokens=tuple(
                    visual + branch_index * 0.1 for branch_index in range(3)
                ),
            )
            padding.append(
                RepresentationVisualTensorBundle(
                    main=output.main_d.unsqueeze(0),
                    deepstack=tuple(
                        branch.unsqueeze(0) for branch in output.deepstack_visual_embeds
                    ),
                    branch_layers=output.metadata.branch_layers,
                )
            )
        self.padding_counts.append(len(padding))
        return SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=f"source-{samples[0].image_group_key}",
            source_visual=source,
            rows=tuple(rows),
            candidates=tuple(candidates),
            collective_padding=tuple(padding),
        )


def _objective() -> RepresentationObjectiveConfig:
    return RepresentationObjectiveConfig(
        identity="validation-objective",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=0.6,
        l_gen_weight=1.1,
    )


def _objective_v2() -> RepresentationObjectiveConfigV2:
    return RepresentationObjectiveConfigV2(
        identity="validation-objective-with-norm",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
    )


def _objective_v3() -> RepresentationObjectiveConfigV3:
    return RepresentationObjectiveConfigV3(
        identity="validation-balanced-matrix-ce",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
        matrix_ce_mode=MatrixCEScoreMode.BALANCED,
    )


def _evaluate(
    *,
    adapter: TGVFAdapter,
    qwen: nn.Module,
    samples: tuple[RepresentationTrainingSample, ...],
    builder: _RecordingGroupBuilder,
    event: int,
    world_size: int = 1,
    objective: RepresentationObjectiveConfig
    | RepresentationObjectiveConfigV2
    | RepresentationObjectiveConfigV3
    | None = None,
):
    return evaluate_representation_validation_event(
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=Qwen3VLAdapter(),
        samples=samples,
        group_builder=builder,
        objective=_objective() if objective is None else objective,
        batch_size=2,
        sampler_seed=41,
        data_manifest_sha256=DATA_SHA256,
        validation_event_index=event,
        data_parallel_world_size=world_size,
    )


def test_validation_is_fresh_deterministic_no_grad_and_rng_isolated() -> None:
    samples = _samples()
    adapter = _adapter().train(True)
    qwen = _qwen(initially_trainable=True)
    builder = _RecordingGroupBuilder()
    owned_parameter = next(
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    )
    owned_parameter.grad = torch.full_like(owned_parameter, 0.125)
    original_gradient = owned_parameter.grad
    original_gradient_value = original_gradient.clone()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    expected_sampler = SameImageBatchSampler(
        samples,
        batch_size=2,
        seed=41,
        data_manifest_sha256=DATA_SHA256,
    )
    expected_indices = ()
    for _ in range(4):
        expected_indices = expected_sampler.next_batch()
    expected_ids = tuple(samples[index].sample_id for index in expected_indices)

    first = _evaluate(
        adapter=adapter, qwen=qwen, samples=samples, builder=builder, event=3
    )
    second = _evaluate(
        adapter=adapter, qwen=qwen, samples=samples, builder=builder, event=3
    )

    assert first == second
    assert first.local_sample_ids == expected_ids
    assert first.global_row_count == first.global_sample_count == 2
    assert first.global_evidence_token_count == 4
    assert first.global_group_count == 1
    assert first.global_total_loss == pytest.approx(
        first.global_matrix_ce_loss * 0.6 + first.global_l_gen_loss * 1.1
    )
    assert builder.calls[0] == builder.calls[1]
    assert builder.calls[0][1:3] == (False, False)
    assert builder.collective_counts == [2, 2]
    assert builder.padding_counts == [0, 0]
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert adapter.training
    assert owned_parameter.grad is original_gradient
    assert torch.equal(owned_parameter.grad, original_gradient_value)
    assert not qwen.training
    assert all(not parameter.requires_grad for parameter in qwen.parameters())
    assert all(parameter.grad is None for parameter in qwen.parameters())


def test_validation_v2_reports_raw_and_weighted_historical_norm() -> None:
    metrics = _evaluate(
        adapter=_adapter(),
        qwen=_qwen(initially_trainable=False).eval(),
        samples=_samples(),
        builder=_RecordingGroupBuilder(),
        event=0,
        objective=_objective_v2(),
    )

    assert metrics.global_norm_loss is not None
    assert metrics.global_norm_loss > 0
    assert metrics.global_weighted_norm_loss == pytest.approx(
        metrics.global_norm_loss * 0.1
    )
    assert metrics.global_total_loss == pytest.approx(
        metrics.global_matrix_ce_loss
        + metrics.global_l_gen_loss
        + metrics.global_weighted_norm_loss
    )


def test_validation_forwards_balanced_objective_without_changing_l_gen_or_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _evaluate(
        adapter=_adapter(),
        qwen=_qwen(initially_trainable=False).eval(),
        samples=_samples(),
        builder=_RecordingGroupBuilder(),
        event=0,
        objective=_objective_v2(),
    )
    balanced_objective = _objective_v3()
    recorded_objectives: list[object] = []
    original_score = evaluation_module.score_streaming_same_image_group

    def recording_score(*args: object, **kwargs: object):
        recorded_objectives.append(kwargs.get("objective"))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(
        evaluation_module,
        "score_streaming_same_image_group",
        recording_score,
    )
    balanced = _evaluate(
        adapter=_adapter(),
        qwen=_qwen(initially_trainable=False).eval(),
        samples=_samples(),
        builder=_RecordingGroupBuilder(),
        event=0,
        objective=balanced_objective,
    )

    assert recorded_objectives == [balanced_objective]
    assert balanced.global_l_gen_loss == pytest.approx(legacy.global_l_gen_loss)
    assert balanced.global_norm_loss == pytest.approx(legacy.global_norm_loss)
    assert balanced.global_weighted_norm_loss == pytest.approx(
        legacy.global_weighted_norm_loss
    )


def _groups_for_both_ranks() -> tuple[str, str]:
    groups: dict[int, str] = {}
    candidate = 0
    while len(groups) < 2:
        key = f"distributed-image-{candidate}"
        groups.setdefault(same_image_group_owner(key, world_size=2), key)
        candidate += 1
    return groups[0], groups[1]


def test_validation_uses_global_numerators_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples(_groups_for_both_ranks())
    adapter = _adapter()
    qwen = _qwen(initially_trainable=False).eval()
    builder = _RecordingGroupBuilder()
    reductions: dict[str, tuple[float, ...] | tuple[int, ...]] = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def all_reduce(tensor: torch.Tensor, *, op: object) -> None:
        if op == torch.distributed.ReduceOp.MAX:
            reductions["maximum"] = tuple(int(value) for value in tensor.tolist())
            tensor.copy_(torch.tensor((3,), dtype=tensor.dtype))
            return
        assert op == torch.distributed.ReduceOp.SUM
        if tensor.dtype == torch.float64:
            reductions["float"] = tuple(float(value) for value in tensor.tolist())
            tensor.add_(torch.tensor((3.0, 5.0), dtype=tensor.dtype))
        else:
            reductions["integer"] = tuple(int(value) for value in tensor.tolist())
            tensor.add_(torch.tensor((2, 2, 4, 1), dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    metrics = _evaluate(
        adapter=adapter,
        qwen=qwen,
        samples=samples,
        builder=builder,
        event=0,
        world_size=2,
    )

    local_matrix, local_l_gen = reductions["float"]
    assert reductions["maximum"] == (2,)
    assert reductions["integer"] == (2, 2, 4, 1)
    assert builder.collective_counts == [3]
    assert builder.padding_counts == [1]
    assert metrics.global_matrix_ce_loss == pytest.approx((local_matrix + 3.0) / 4)
    assert metrics.global_l_gen_loss == pytest.approx((local_l_gen + 5.0) / 4)
    assert metrics.global_row_count == metrics.global_sample_count == 4
    assert metrics.global_evidence_token_count == 8
    assert metrics.global_group_count == 2
    assert metrics.local_rank == 0
    assert metrics.data_parallel_world_size == 2


def test_validation_fails_on_world_drift_and_group_builder_reordering() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="world size mismatch"):
        _evaluate(
            adapter=_adapter(),
            qwen=_qwen(),
            samples=samples,
            builder=_RecordingGroupBuilder(),
            event=0,
            world_size=2,
        )

    with pytest.raises(ValueError, match="sample order"):
        _evaluate(
            adapter=_adapter(),
            qwen=_qwen(),
            samples=samples,
            builder=_RecordingGroupBuilder(reverse=True),
            event=0,
        )


def test_validation_event_index_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _evaluate(
            adapter=_adapter(),
            qwen=_qwen(),
            samples=_samples(),
            builder=_RecordingGroupBuilder(),
            event=-1,
        )


def test_validation_cuda_rng_touches_only_the_current_adapter_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("global CUDA RNG APIs and torch.manual_seed are forbidden")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def get_rng_state(device: int) -> torch.Tensor:
        calls.append(("get", device))
        return torch.tensor((7, 8), dtype=torch.uint8)

    def manual_seed(seed: int) -> None:
        calls.append(("seed", seed))

    def set_rng_state(state: torch.Tensor, device: int) -> None:
        calls.append(("set", tuple(int(value) for value in state.tolist()), device))

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rng_state)
    monkeypatch.setattr(torch.cuda, "manual_seed", manual_seed)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)

    with _isolated_validation_rng(123, device=torch.device("cuda", 1)):
        calls.append(("body",))

    assert calls == [
        ("get", 1),
        ("seed", 123),
        ("body",),
        ("set", (7, 8), 1),
    ]


def test_validation_cuda_rng_rejects_non_current_adapter_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CUDA RNG state must not be read before device validation")

    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    with pytest.raises(RuntimeError, match="current CUDA rank device"):
        with _isolated_validation_rng(123, device=torch.device("cuda", 1)):
            pass
