from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from threading import Event, Lock, get_ident
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tgvf_rl.representation.training.trainer as trainer_module
from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
)
from tgvf_rl.representation.training.losses import EVIDENCE_IGNORE_INDEX
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.readout import (
    RepresentationCandidateObservation,
    RepresentationReadoutRow,
    RepresentationVisualTensorBundle,
    SameImageReadoutGroup,
)
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.trainer import (
    RepresentationOptimizerConfig,
    RepresentationPrecision,
    RepresentationSchedulerConfig,
    RepresentationSchedulerKind,
    RepresentationTrainer,
    RepresentationTrainerConfig,
    build_representation_optimizer,
    build_representation_scheduler,
    synchronize_collective_candidate_counts,
    _accumulate_local_metric_numerators_,
    _assert_gradients,
    _global_metric_sums,
)
from tgvf_rl.representation.training.streaming import StreamingBackwardMetrics
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
    torch.manual_seed(101)
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
        self.embed_tokens = nn.Embedding(20, 6)

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
        hidden = hidden + hidden.sum(dim=1, keepdim=True) * 0.05
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyContainer()
        self.lm_head = nn.Linear(6, 20, bias=False)


def _qwen() -> _TinyQwen:
    torch.manual_seed(202)
    model = _TinyQwen().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sample(image: str, member: int) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"{image}-{member}",
        image=f"/{image}.png",
        image_id=image,
        question="What is shown?",
        target=f"target-{member}",
        evidence_description=f"evidence-{member}",
    )


def _samples() -> tuple[RepresentationTrainingSample, ...]:
    return tuple(_sample(image, member) for image in ("a", "b") for member in range(2))


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


class _GroupBuilder:
    def __call__(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        adapter: TGVFAdapter,
        *,
        collective_candidate_count: int,
    ) -> SameImageReadoutGroup:
        if collective_candidate_count < len(samples):
            raise ValueError("collective candidate count cannot be smaller than real K")
        rows = []
        candidates = []
        source = RepresentationVisualTensorBundle(
            main=torch.full((1, 2, 6), 0.2),
            deepstack=tuple(torch.full((1, 2, 6), 0.05) for _ in range(3)),
            branch_layers=(8, 16, 24),
        )
        for index, sample in enumerate(samples):
            token_ids = (1, 2, 2, 2, 2, 3, 5 + index * 2, 6 + index * 2)
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
            target = torch.full((3, 6), 0.1 + index * 0.1)
            visual = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
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
            target = torch.full((3, 6), 0.1)
            visual = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
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
        return SameImageReadoutGroup(
            image_group_key=samples[0].image_group_key,
            source_visual_identity=f"source-{samples[0].image_group_key}",
            source_visual=source,
            rows=tuple(rows),
            candidates=tuple(candidates),
            collective_padding=tuple(padding),
        )


class _RecordingAccumulationController:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def begin_microstep(self, *, index: int, count: int) -> None:
        self.events.append(("begin", index, count))

    def finish_window(self) -> None:
        self.events.append(("finish",))


class _FailOnSecondGroupBuilder(_GroupBuilder):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs) -> SameImageReadoutGroup:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected second-microstep failure")
        return super().__call__(*args, **kwargs)


class _PrefetchGroupBuilder(_GroupBuilder):
    def __init__(self, *, fail_prepare_call: int | None = None) -> None:
        self.fail_prepare_call = fail_prepare_call
        self.main_thread_id = get_ident()
        self.prepare_sample_ids: list[tuple[str, ...]] = []
        self.materialize_sample_ids: list[tuple[str, ...]] = []
        self.prepare_thread_ids: list[int] = []
        self.materialize_thread_ids: list[int] = []
        self.sync_calls = 0
        self.active_prepares = 0
        self.max_active_prepares = 0
        self.second_prepare_started = Event()
        self.release_second_prepare = Event()
        self.second_prepare_finished = Event()
        self.overlapped_second_prepare_with_first_materialize = False
        self._lock = Lock()

    def __call__(self, *args, **kwargs) -> SameImageReadoutGroup:
        self.sync_calls += 1
        raise AssertionError("prefetch-capable builder used synchronous __call__")

    def _prepare_cpu_group(
        self,
        samples: tuple[RepresentationTrainingSample, ...],
        *,
        collective_candidate_count: int,
    ) -> object:
        sample_ids = tuple(sample.sample_id for sample in samples)
        with self._lock:
            self.prepare_sample_ids.append(sample_ids)
            self.prepare_thread_ids.append(get_ident())
            call_index = len(self.prepare_sample_ids)
            self.active_prepares += 1
            self.max_active_prepares = max(
                self.max_active_prepares,
                self.active_prepares,
            )
        try:
            if call_index == 2:
                self.second_prepare_started.set()
                if self.fail_prepare_call != call_index:
                    if not self.release_second_prepare.wait(timeout=5):
                        raise RuntimeError("second CPU preparation was not released")
            if self.fail_prepare_call == call_index:
                raise RuntimeError(f"injected CPU preparation failure {call_index}")
            return samples, collective_candidate_count
        finally:
            with self._lock:
                self.active_prepares -= 1
            if call_index == 2:
                self.second_prepare_finished.set()

    def _materialize_prepared_group(
        self,
        prepared: object,
        adapter: TGVFAdapter,
    ) -> SameImageReadoutGroup:
        if (
            not isinstance(prepared, tuple)
            or len(prepared) != 2
            or not isinstance(prepared[0], tuple)
            or not isinstance(prepared[1], int)
        ):
            raise TypeError("malformed prepared group fixture")
        samples = prepared[0]
        collective_candidate_count = prepared[1]
        sample_ids = tuple(sample.sample_id for sample in samples)
        with self._lock:
            self.materialize_sample_ids.append(sample_ids)
            self.materialize_thread_ids.append(get_ident())
            materialize_index = len(self.materialize_sample_ids)
        if materialize_index == 1 and self.fail_prepare_call is None:
            if not self.second_prepare_started.wait(timeout=5):
                raise RuntimeError("next CPU preparation did not start one group ahead")
            with self._lock:
                self.overlapped_second_prepare_with_first_materialize = (
                    self.active_prepares == 1
                )
            self.release_second_prepare.set()
        return super().__call__(
            samples,
            adapter,
            collective_candidate_count=collective_candidate_count,
        )


class _PrepareOnlyGroupBuilder(_GroupBuilder):
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.sync_thread_ids: list[int] = []

    def _prepare_cpu_group(self, *args, **kwargs) -> object:
        self.prepare_calls += 1
        raise AssertionError("a partial private hook must not enable prefetch")

    def __call__(self, *args, **kwargs) -> SameImageReadoutGroup:
        self.sync_thread_ids.append(get_ident())
        return super().__call__(*args, **kwargs)


def _objective() -> RepresentationObjectiveConfig:
    return RepresentationObjectiveConfig(
        identity="trainer-test-objective",
        kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
        matrix_ce_weight=0.6,
        l_gen_weight=1.1,
    )


def _objective_v2() -> RepresentationObjectiveConfigV2:
    return RepresentationObjectiveConfigV2(
        identity="trainer-test-historical-norm-objective",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=0.6,
        l_gen_weight=1.1,
        norm_weight=0.1,
    )


def _trainer_for_group_builder(
    group_builder: _GroupBuilder,
    *,
    accumulation_controller: _RecordingAccumulationController | None = None,
) -> RepresentationTrainer:
    samples = _samples()
    adapter = _adapter()
    qwen = _qwen()
    optimizer = build_representation_optimizer(
        adapter,
        RepresentationOptimizerConfig(
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    )
    return RepresentationTrainer(
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=Qwen3VLAdapter(),
        samples=samples,
        sampler=SameImageBatchSampler(
            samples,
            batch_size=2,
            seed=17,
            data_manifest_sha256=sha256(b"trainer-prefetch-data").hexdigest(),
        ),
        group_builder=group_builder,
        objective=_objective(),
        accumulation=RepresentationAccumulationIdentity(
            gradient_accumulation_steps=2,
            data_parallel_world_size=1,
        ),
        optimizer=optimizer,
        scheduler=None,
        config=RepresentationTrainerConfig(
            precision=RepresentationPrecision.FP32,
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
        accumulation_controller=accumulation_controller,
    )


@pytest.mark.parametrize(
    "objective",
    (_objective(), _objective_v2()),
    ids=("historical-v1-no-norm", "v2-required-norm"),
)
def test_trainer_executes_accumulated_optimizer_step_and_keeps_qwen_frozen(
    objective: RepresentationObjectiveConfig | RepresentationObjectiveConfigV2,
) -> None:
    samples = _samples()
    adapter = _adapter()
    qwen = _qwen()
    optimizer = build_representation_optimizer(
        adapter,
        RepresentationOptimizerConfig(
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    )
    scheduler = build_representation_scheduler(
        optimizer,
        RepresentationSchedulerConfig(
            kind=RepresentationSchedulerKind.LINEAR_WARMUP_DECAY,
            total_steps=2,
            warmup_steps=0,
        ),
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    sampler = SameImageBatchSampler(
        samples,
        batch_size=2,
        seed=17,
        data_manifest_sha256=sha256(b"trainer-test-data").hexdigest(),
    )
    accumulation_controller = _RecordingAccumulationController()
    trainer = RepresentationTrainer(
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=Qwen3VLAdapter(),
        samples=samples,
        sampler=sampler,
        group_builder=_GroupBuilder(),
        objective=objective,
        accumulation=RepresentationAccumulationIdentity(
            gradient_accumulation_steps=2,
            data_parallel_world_size=1,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        config=RepresentationTrainerConfig(
            precision=RepresentationPrecision.FP32,
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
        accumulation_controller=accumulation_controller,
    )

    metrics = trainer.train_step()

    assert metrics.global_step == 1
    assert metrics.global_row_count == 4
    assert metrics.global_sample_count == 4
    assert metrics.global_matrix_ce_loss > 0
    assert metrics.global_l_gen_loss > 0
    expected_total = (
        metrics.global_matrix_ce_loss * 0.6 + metrics.global_l_gen_loss * 1.1
    )
    if isinstance(objective, RepresentationObjectiveConfigV2):
        assert metrics.global_norm_loss is not None
        assert metrics.global_weighted_norm_loss == pytest.approx(
            metrics.global_norm_loss * 0.1
        )
        expected_total += metrics.global_weighted_norm_loss
    else:
        assert metrics.global_norm_loss is None
        assert metrics.global_weighted_norm_loss is None
    assert metrics.global_total_loss == pytest.approx(expected_total)
    assert metrics.gradient_norm_before_clip > 0
    assert metrics.learning_rate == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert len(metrics.local_sample_ids) == 4
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in adapter.named_parameters()
        if name in before
    )
    assert all(not parameter.requires_grad for parameter in qwen.parameters())
    assert all(parameter.grad is None for parameter in qwen.parameters())
    assert accumulation_controller.events == [
        ("begin", 0, 2),
        ("begin", 1, 2),
        ("finish",),
    ]


def test_trainer_prefetches_only_one_cpu_group_ahead_of_materialization() -> None:
    builder = _PrefetchGroupBuilder()
    trainer = _trainer_for_group_builder(builder)

    metrics = trainer.train_step()

    assert metrics.global_step == 1
    assert builder.sync_calls == 0
    assert builder.prepare_sample_ids == builder.materialize_sample_ids
    assert (
        tuple(
            sample_id
            for group_sample_ids in builder.materialize_sample_ids
            for sample_id in group_sample_ids
        )
        == metrics.local_sample_ids
    )
    assert len(builder.prepare_sample_ids) == 2
    assert len(set(builder.prepare_thread_ids)) == 1
    assert builder.prepare_thread_ids[0] != builder.main_thread_id
    assert builder.materialize_thread_ids == [builder.main_thread_id] * 2
    assert builder.max_active_prepares == 1
    assert builder.overlapped_second_prepare_with_first_materialize
    assert builder.second_prepare_finished.is_set()
    assert builder.active_prepares == 0


def test_cpu_prefetch_failure_is_drained_and_trainer_is_fail_stop() -> None:
    builder = _PrefetchGroupBuilder(fail_prepare_call=2)
    controller = _RecordingAccumulationController()
    trainer = _trainer_for_group_builder(
        builder,
        accumulation_controller=controller,
    )

    with pytest.raises(RuntimeError, match="injected CPU preparation failure 2"):
        trainer.train_step()

    assert trainer.global_step == 0
    assert builder.sync_calls == 0
    assert len(builder.prepare_sample_ids) == 2
    assert len(builder.materialize_sample_ids) == 1
    assert builder.second_prepare_finished.is_set()
    assert builder.active_prepares == 0
    assert controller.events == [
        ("begin", 0, 2),
        ("begin", 1, 2),
        ("finish",),
    ]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        trainer.train_step()


def test_partial_prefetch_hooks_keep_the_synchronous_builder_path() -> None:
    builder = _PrepareOnlyGroupBuilder()
    trainer = _trainer_for_group_builder(builder)
    main_thread_id = get_ident()

    metrics = trainer.train_step()

    assert metrics.global_step == 1
    assert builder.prepare_calls == 0
    assert builder.sync_thread_ids == [main_thread_id, main_thread_id]


def test_partial_accumulation_failure_is_fail_stop() -> None:
    samples = _samples()
    adapter = _adapter()
    qwen = _qwen()
    optimizer = build_representation_optimizer(
        adapter,
        RepresentationOptimizerConfig(
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    )
    controller = _RecordingAccumulationController()
    trainer = RepresentationTrainer(
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=Qwen3VLAdapter(),
        samples=samples,
        sampler=SameImageBatchSampler(
            samples,
            batch_size=2,
            seed=17,
            data_manifest_sha256=sha256(b"trainer-fail-stop-data").hexdigest(),
        ),
        group_builder=_FailOnSecondGroupBuilder(),
        objective=_objective(),
        accumulation=RepresentationAccumulationIdentity(
            gradient_accumulation_steps=2,
            data_parallel_world_size=1,
        ),
        optimizer=optimizer,
        scheduler=None,
        config=RepresentationTrainerConfig(
            precision=RepresentationPrecision.FP32,
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
        accumulation_controller=controller,
    )

    with pytest.raises(RuntimeError, match="injected second-microstep failure"):
        trainer.train_step()
    assert trainer.global_step == 0
    assert controller.events == [
        ("begin", 0, 2),
        ("begin", 1, 2),
        ("finish",),
    ]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        trainer.train_step()


@pytest.mark.parametrize(
    ("gradient_accumulation_steps", "expected_window_sizes", "expected_qwen_batches"),
    (
        (1, (4,), (32, 32)),
        (2, (2, 2), (16, 16, 16, 16)),
    ),
)
def test_trainer_partitions_four_direct_groups_without_changing_global_normalization(
    monkeypatch: pytest.MonkeyPatch,
    gradient_accumulation_steps: int,
    expected_window_sizes: tuple[int, ...],
    expected_qwen_batches: tuple[int, ...],
) -> None:
    samples = tuple(
        _sample(image, member) for image in ("a", "b", "c", "d") for member in range(4)
    )
    adapter = _adapter()
    qwen = _qwen()
    optimizer = build_representation_optimizer(
        adapter,
        RepresentationOptimizerConfig(
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    )
    trainer = RepresentationTrainer(
        adapter=adapter,
        qwen_model=qwen,
        family_adapter=Qwen3VLAdapter(),
        samples=samples,
        sampler=SameImageBatchSampler(
            samples,
            batch_size=4,
            seed=19,
            data_manifest_sha256=sha256(b"trainer-direct-groups").hexdigest(),
        ),
        group_builder=_GroupBuilder(),
        objective=_objective_v2(),
        accumulation=RepresentationAccumulationIdentityV2(
            gradient_accumulation_steps=gradient_accumulation_steps,
            data_parallel_world_size=1,
            groups_per_rank_per_optimizer_step=4,
        ),
        optimizer=optimizer,
        scheduler=None,
        config=RepresentationTrainerConfig(
            precision=RepresentationPrecision.FP32,
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
    )
    original_score = trainer_module.score_streaming_same_image_groups
    original_backward = trainer_module.backward_streaming_same_image_groups
    score_windows: list[int] = []
    backward_windows: list[int] = []
    normalization_rows: list[int] = []

    def record_score(family_adapter, model, groups, **kwargs):
        score_windows.append(len(groups))
        normalization_rows.append(kwargs["normalization"].matrix_valid_rows)
        return original_score(family_adapter, model, groups, **kwargs)

    def record_backward(family_adapter, model, groups, scores, **kwargs):
        backward_windows.append(len(groups))
        normalization_rows.append(kwargs["normalization"].matrix_valid_rows)
        return original_backward(family_adapter, model, groups, scores, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "score_streaming_same_image_groups",
        record_score,
    )
    monkeypatch.setattr(
        trainer_module,
        "backward_streaming_same_image_groups",
        record_backward,
    )

    metrics = trainer.train_step()

    assert metrics.global_step == 1
    assert metrics.global_row_count == 16
    assert metrics.global_sample_count == 16
    assert len(metrics.local_sample_ids) == 16
    assert len(set(metrics.local_sample_ids)) == 16
    assert score_windows == list(expected_window_sizes)
    assert backward_windows == list(expected_window_sizes)
    assert normalization_rows == [16] * (2 * len(expected_window_sizes))
    assert metrics.local_qwen_forward_batch_sizes == expected_qwen_batches
    assert metrics.global_norm_loss is not None
    assert metrics.gradient_norm_before_clip > 0


@pytest.mark.parametrize(
    ("total_steps", "warmup_steps", "expected_used", "expected_final"),
    (
        (4, 0, (1.0, 0.75, 0.5, 0.25), 0.0),
        (5, 2, (0.5, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0), 0.0),
    ),
)
def test_linear_scheduler_controls_the_learning_rate_used_by_each_update(
    total_steps: int,
    warmup_steps: int,
    expected_used: tuple[float, ...],
    expected_final: float,
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    base_learning_rate = 2.0
    optimizer = torch.optim.SGD([parameter], lr=base_learning_rate)
    scheduler = build_representation_scheduler(
        optimizer,
        RepresentationSchedulerConfig(
            kind=RepresentationSchedulerKind.LINEAR_WARMUP_DECAY,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    used = []
    for _ in range(total_steps):
        used.append(float(optimizer.param_groups[0]["lr"]) / base_learning_rate)
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    assert used == pytest.approx(expected_used)
    assert optimizer.param_groups[0]["lr"] / base_learning_rate == pytest.approx(
        expected_final
    )


def test_exact_historical_cosine_schedule_and_state_resume() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    config = RepresentationSchedulerConfig(
        kind=RepresentationSchedulerKind.HISTORICAL_COSINE,
        total_steps=6,
        warmup_steps=2,
        min_lr_ratio=0.1,
    )
    scheduler = build_representation_scheduler(optimizer, config)
    used = []
    checkpoint = None
    for update in range(6):
        used.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
        if update == 2:
            checkpoint = (
                deepcopy(optimizer.state_dict()),
                deepcopy(scheduler.state_dict()),
            )

    # Values are the pinned launcher's (step+1) warmup/decay indexing and FP32
    # torch.cos evaluation, not a generic library cosine default.
    assert used == pytest.approx(
        (
            0.5,
            1.0,
            0.8681980460882187,
            0.5499999803298753,
            0.23180195391178132,
            0.1,
        ),
        rel=0,
        abs=1e-15,
    )
    assert optimizer.param_groups[0]["lr"] == 0.1

    assert checkpoint is not None
    resumed_parameter = nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=1.0)
    resumed_scheduler = build_representation_scheduler(resumed_optimizer, config)
    resumed_optimizer.load_state_dict(checkpoint[0])
    resumed_scheduler.load_state_dict(checkpoint[1])
    resumed_used = []
    for _ in range(3):
        resumed_used.append(float(resumed_optimizer.param_groups[0]["lr"]))
        resumed_optimizer.step()
        resumed_scheduler.step()
    assert resumed_used == used[3:]
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_historical_cosine_requires_explicit_bounded_minimum_ratio() -> None:
    with pytest.raises(TypeError, match="explicit float"):
        RepresentationSchedulerConfig(
            kind=RepresentationSchedulerKind.HISTORICAL_COSINE,
            total_steps=2000,
            warmup_steps=100,
        )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        RepresentationSchedulerConfig(
            kind=RepresentationSchedulerKind.HISTORICAL_COSINE,
            total_steps=2000,
            warmup_steps=100,
            min_lr_ratio=1.1,
        )


def test_collective_candidate_counts_use_each_microsteps_global_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(values: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.MAX
        assert tuple(int(value) for value in values.tolist()) == (4, 5)
        remote_counts = torch.tensor((5, 4), dtype=values.dtype)
        values.copy_(torch.maximum(values, remote_counts))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert synchronize_collective_candidate_counts(
        (4, 5), device=torch.device("cpu")
    ) == (5, 5)


def test_gradient_validation_uses_one_numeric_collective_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = nn.Parameter(torch.ones(3))
    second = nn.Parameter(torch.ones(2))
    first.grad = torch.tensor((1.0, 2.0, 3.0))
    second.grad = torch.tensor((4.0, 5.0))
    reductions: list[tuple[tuple[int, ...], object]] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(values: torch.Tensor, *, op: object) -> None:
        reductions.append((tuple(values.tolist()), op))

    def reject_object_gather(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("success path must not gather Python objects")

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        reject_object_gather,
    )

    _assert_gradients(
        (("first", first), ("second", second)),
        require_all=True,
    )

    assert reductions == [((1, 1), torch.distributed.ReduceOp.MIN)]


def test_gradient_validation_keeps_ranked_names_on_missing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = nn.Parameter(torch.ones(3))
    second = nn.Parameter(torch.ones(2))
    first.grad = torch.ones_like(first)
    second.grad = None
    gathered_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def all_reduce(values: torch.Tensor, *, op: object) -> None:
        assert tuple(values.tolist()) == (0, 1)
        assert op == torch.distributed.ReduceOp.MIN

    def all_gather_object(
        gathered: list[object],
        missing: tuple[str, ...],
    ) -> None:
        gathered_calls.append(missing)
        gathered[:] = [("second",), ("remote_second",)]

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match="remote_second"):
        _assert_gradients(
            (("first", first), ("second", second)),
            require_all=True,
        )

    assert gathered_calls == [("second",)]


def test_gradient_validation_keeps_nonfinite_failure_contract() -> None:
    first = nn.Parameter(torch.ones(3))
    second = nn.Parameter(torch.ones(2))
    first.grad = torch.tensor((1.0, float("nan"), 3.0))
    second.grad = torch.ones_like(second)

    with pytest.raises(
        FloatingPointError,
        match="non-finite Adapter gradient on at least one rank",
    ):
        _assert_gradients(
            (("first", first), ("second", second)),
            require_all=True,
        )


def test_metric_numerators_stay_tensor_resident_until_one_global_pack() -> None:
    totals = torch.zeros(3, dtype=torch.float64)
    for matrix, l_gen, norm in ((1.25, 2.5, 3.75), (4.0, 5.0, 6.0)):
        metrics = StreamingBackwardMetrics(
            matrix_ce_numerator=torch.tensor(matrix),
            l_gen_numerator=torch.tensor(l_gen),
            norm_numerator=torch.tensor(norm),
            local_row_count=1,
            local_sample_count=1,
            weighted_local_mean=torch.tensor(0.0),
            weighted_norm_local_mean=torch.tensor(0.0),
        )
        _accumulate_local_metric_numerators_(totals, metrics)

    assert torch.equal(totals, torch.tensor((5.25, 7.5, 9.75)))
    assert _global_metric_sums(totals, has_norm=True) == (5.25, 7.5, 9.75)


def test_trainer_rejects_optimizer_that_owns_frozen_qwen_or_omits_adapter() -> None:
    samples = _samples()
    adapter = _adapter()
    qwen = _qwen()
    wrong = torch.optim.AdamW([adapter.target_proj.weight], lr=1e-3)
    sampler = SameImageBatchSampler(
        samples,
        batch_size=2,
        seed=1,
        data_manifest_sha256=sha256(b"wrong-optimizer").hexdigest(),
    )

    with pytest.raises(ValueError, match="every and only"):
        RepresentationTrainer(
            adapter=adapter,
            qwen_model=qwen,
            family_adapter=Qwen3VLAdapter(),
            samples=samples,
            sampler=sampler,
            group_builder=_GroupBuilder(),
            objective=_objective(),
            accumulation=RepresentationAccumulationIdentity(1, 1),
            optimizer=wrong,
            scheduler=None,
            config=RepresentationTrainerConfig(
                precision=RepresentationPrecision.FP32,
                max_grad_norm=1.0,
                require_all_adapter_gradients=True,
            ),
        )
