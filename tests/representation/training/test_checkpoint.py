from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter, TGVFAdapterVariant
from tgvf_rl.representation.training.checkpoint import (
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION,
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2,
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
    RepresentationAdapterContractIdentity,
    RepresentationAdapterContractIdentityV2,
    RepresentationInitializationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationRunIdentityV3,
    RepresentationSamplerContractIdentity,
    RepresentationSchedulerIdentity,
    RepresentationSchedulerIdentityV2,
    RepresentationTrainerExecutionIdentity,
    capture_representation_rng_state,
    representation_adapter_contract_identity,
    load_representation_adapter_artifact,
    load_representation_training_checkpoint,
    restore_representation_adapter_artifact,
    restore_representation_rng_state,
    restore_representation_training_checkpoint,
    save_representation_adapter_artifact_atomic,
    save_representation_training_checkpoint_atomic,
)
from tgvf_rl.representation.training.data import SplitOverlapPolicy
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.sampling import SameImageBatchSampler
from tgvf_rl.representation.training.schema import RepresentationTrainingSample
from tgvf_rl.representation.training.trainer import (
    RepresentationPrecision,
    RepresentationSchedulerConfig,
    RepresentationSchedulerKind,
    RepresentationTrainerConfig,
    build_representation_scheduler,
)
from tgvf_rl.representation.training.validation_identity import (
    REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION,
    RepresentationValidationDataIdentity,
)


DATA_MANIFEST_SHA256 = "2" * 64
SCHEDULER_TOTAL_STEPS = 10


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


def _adapter(
    seed: int,
    variant: TGVFAdapterVariant = TGVFAdapterVariant.FULL_D_DEEPSTACK,
) -> TGVFAdapter:
    torch.manual_seed(seed)
    return TGVFAdapter(
        d_lm=6,
        d_v=4,
        attn_dim=5,
        main_projection=_projection("qwen-main-merger@model-revision"),
        deepstack_projections=tuple(
            _projection(f"qwen-merger-{layer}@model-revision") for layer in (8, 16, 24)
        ),
        branch_layers=(8, 16, 24),
        variant=variant,
    )


def test_main_d_only_contract_content_binds_structural_variant() -> None:
    adapter = _adapter(3, TGVFAdapterVariant.MAIN_D_ONLY)

    contract = representation_adapter_contract_identity(adapter)

    assert isinstance(contract, RepresentationAdapterContractIdentityV2)
    assert contract.variant == "main_d_only"
    contract.assert_matches(adapter)
    with pytest.raises(ValueError, match="only the full D-DeepStack"):
        RepresentationAdapterContractIdentity.from_adapter(adapter)
    with pytest.raises(IdentityMismatchError, match="variant"):
        contract.assert_matches(_adapter(3))


def test_vision_routing_contract_is_isolated_from_historical_rp66() -> None:
    routing = _adapter(3, TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING)

    contract = representation_adapter_contract_identity(routing)

    assert isinstance(contract, RepresentationAdapterContractIdentityV2)
    assert contract.variant == "full_d_deepstack_vision_routing"
    contract.assert_matches(routing)
    with pytest.raises(IdentityMismatchError, match="variant"):
        contract.assert_matches(_adapter(3))


def test_unidirectional_contract_is_isolated_from_bidirectional_adapter() -> None:
    one_way = _adapter(
        3, TGVFAdapterVariant.FULL_D_DEEPSTACK_UNIDIRECTIONAL_TARGET_TO_VISUAL
    )
    contract = representation_adapter_contract_identity(one_way)

    assert isinstance(contract, RepresentationAdapterContractIdentityV2)
    assert contract.variant == "full_d_deepstack_unidirectional_target_to_visual"
    contract.assert_matches(one_way)
    with pytest.raises(IdentityMismatchError, match="variant"):
        contract.assert_matches(_adapter(3))


def test_post_merger_contract_binds_width_merge_size_and_identity_writeback() -> None:
    torch.manual_seed(3)
    post = TGVFAdapter(
        d_lm=6,
        d_v=6,
        attn_dim=4,
        main_projection=_projection("qwen-main-merger@model-revision"),
        deepstack_projections=tuple(
            _projection(f"qwen-merger-{layer}@model-revision")
            for layer in (8, 16, 24)
        ),
        branch_layers=(8, 16, 24),
        variant=TGVFAdapterVariant.FULL_D_DEEPSTACK_POST_MERGER,
    )
    contract = representation_adapter_contract_identity(post)

    assert isinstance(contract, RepresentationAdapterContractIdentityV2)
    assert contract.variant == "full_d_deepstack_post_merger"
    assert contract.d_v == 6
    assert contract.spatial_merge_size == 1
    assert contract.main_projection_identity.endswith(
        "::post-merger-identity-writeback"
    )
    contract.assert_matches(post)


def _run_identity(
    adapter: TGVFAdapter,
    *,
    sampler: SameImageBatchSampler,
    optimizer: torch.optim.Optimizer,
    initialization_seed: int,
    with_scheduler: bool,
) -> RepresentationRunIdentity:
    return RepresentationRunIdentity(
        run_id="cpu-resume-fixture",
        code=CodeIdentity(
            repository="Miocio-nora/TGVF-E2E-RL", commit="fixture-commit"
        ),
        model=ModelIdentity(
            family="qwen3_vl",
            model_name="Qwen3-VL-8B-Thinking",
            revision_or_path="/stable/qwen3",
            tokenizer_length=151669,
            chat_template_sha256="1" * 64,
        ),
        provider=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE,
            hidden_layer=-1,
        ),
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        prompt_sha256="3" * 64,
        objective=RepresentationObjectiveConfig(
            identity="matrix-ce-plus-l-gen-fixture",
            kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
            matrix_ce_weight=1.0,
            l_gen_weight=0.25,
        ),
        adapter_contract=RepresentationAdapterContractIdentity.from_adapter(adapter),
        accumulation=RepresentationAccumulationIdentity(
            gradient_accumulation_steps=1,
            data_parallel_world_size=1,
        ),
        optimizer=RepresentationOptimizerIdentity.from_optimizer(optimizer),
        scheduler=(
            RepresentationSchedulerIdentity.from_config(_scheduler_config())
            if with_scheduler
            else None
        ),
        trainer_execution=RepresentationTrainerExecutionIdentity.from_config(
            RepresentationTrainerConfig(
                precision=RepresentationPrecision.FP32,
                max_grad_norm=1.0,
                require_all_adapter_gradients=True,
            )
        ),
        initialization=RepresentationInitializationIdentity.from_adapter(
            adapter,
            kind="fresh_random",
            seed=initialization_seed,
            source_artifact_sha256=None,
        ),
        sampler_contract=RepresentationSamplerContractIdentity.from_sampler(sampler),
    )


def _validation_identity() -> RepresentationValidationDataIdentity:
    return RepresentationValidationDataIdentity(
        train_retained_manifest_sha256=DATA_MANIFEST_SHA256,
        validation_retained_manifest_sha256="4" * 64,
        validation_batch_k=4,
        validation_sampler_seed=73,
        validation_every_optimizer_steps=5,
        evaluator_schema_version=REPRESENTATION_VALIDATION_EVALUATOR_SCHEMA_VERSION,
        overlap_policy=SplitOverlapPolicy.REQUIRE_DISJOINT,
        overlap_report_sha256="5" * 64,
        overlap_record_count=0,
        overlap_kinds=(),
        train_image_manifest_sha256="6" * 64,
        train_image_file_count=2,
        train_image_total_size_bytes=20,
        validation_image_manifest_sha256="7" * 64,
        validation_image_file_count=1,
        validation_image_total_size_bytes=10,
    )


def _run_identity_v3(
    adapter: TGVFAdapter,
    *,
    sampler: SameImageBatchSampler,
    optimizer: torch.optim.Optimizer,
    initialization_seed: int,
    with_scheduler: bool,
    planned_target_optimizer_steps: int = 2,
) -> RepresentationRunIdentityV3:
    legacy = _run_identity(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=initialization_seed,
        with_scheduler=with_scheduler,
    )
    common = {
        name: getattr(legacy, name)
        for name in RepresentationRunIdentity.__dataclass_fields__
        if name != "schema_version"
    }
    return RepresentationRunIdentityV3(
        **common,
        validation_identity=_validation_identity(),
        planned_target_optimizer_steps=planned_target_optimizer_steps,
    )


def _samples() -> tuple[RepresentationTrainingSample, ...]:
    return tuple(
        RepresentationTrainingSample(
            sample_id=f"{image_id}-{target_index}",
            image=f"images/{image_id}.jpg",
            image_id=image_id,
            question="What is visible?",
            target=f"target {target_index}",
            evidence_description=f"evidence {target_index}",
            short_answer=f"answer {target_index}",
        )
        for image_id in ("image-a", "image-b")
        for target_index in range(4)
    )


def _sampler() -> SameImageBatchSampler:
    return SameImageBatchSampler(
        _samples(),
        batch_size=4,
        seed=73,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )


def _optimizer(adapter: TGVFAdapter) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=3e-4,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    return build_representation_scheduler(optimizer, _scheduler_config())


def _scheduler_config() -> RepresentationSchedulerConfig:
    return RepresentationSchedulerConfig(
        kind=RepresentationSchedulerKind.CONSTANT,
        total_steps=SCHEDULER_TOTAL_STEPS,
        warmup_steps=0,
    )


def test_scheduler_v2_identity_binds_minimum_ratio_without_changing_v1_type() -> None:
    v1 = RepresentationSchedulerIdentity.from_config(_scheduler_config())
    first_config = RepresentationSchedulerConfig(
        kind=RepresentationSchedulerKind.HISTORICAL_COSINE,
        total_steps=2000,
        warmup_steps=100,
        min_lr_ratio=0.1,
    )
    second_config = RepresentationSchedulerConfig(
        kind=RepresentationSchedulerKind.HISTORICAL_COSINE,
        total_steps=2000,
        warmup_steps=100,
        min_lr_ratio=0.2,
    )
    first = RepresentationSchedulerIdentity.from_config(first_config)
    second = RepresentationSchedulerIdentity.from_config(second_config)

    assert type(v1) is RepresentationSchedulerIdentity
    assert isinstance(first, RepresentationSchedulerIdentityV2)
    assert isinstance(second, RepresentationSchedulerIdentityV2)
    assert first.min_lr_ratio == 0.1
    assert second.min_lr_ratio == 0.2
    assert first.identity_sha256 != second.identity_sha256


def test_accumulation_v2_binds_direct_groups_without_changing_v1_type(
    tmp_path,
) -> None:
    legacy = RepresentationAccumulationIdentity(
        gradient_accumulation_steps=1,
        data_parallel_world_size=1,
    )
    direct = RepresentationAccumulationIdentityV2(
        gradient_accumulation_steps=1,
        data_parallel_world_size=1,
        groups_per_rank_per_optimizer_step=3,
    )
    partitioned = RepresentationAccumulationIdentityV2(
        gradient_accumulation_steps=2,
        data_parallel_world_size=1,
        groups_per_rank_per_optimizer_step=4,
    )

    assert type(legacy) is RepresentationAccumulationIdentity
    assert legacy.schema_version == REPRESENTATION_ACCUMULATION_SCHEMA_VERSION
    assert type(direct) is RepresentationAccumulationIdentityV2
    assert direct.schema_version == REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2
    assert partitioned.groups_per_accumulation_microstep == 2
    assert partitioned.identity_sha256 != direct.identity_sha256
    assert direct.identity_sha256 != legacy.identity_sha256
    assert (
        direct.identity_sha256
        != replace(
            direct,
            groups_per_rank_per_optimizer_step=4,
        ).identity_sha256
    )

    with pytest.raises(ValueError, match="more than one direct group"):
        RepresentationAccumulationIdentityV2(
            gradient_accumulation_steps=1,
            data_parallel_world_size=1,
            groups_per_rank_per_optimizer_step=1,
        )
    with pytest.raises(ValueError, match="evenly divisible"):
        RepresentationAccumulationIdentityV2(
            gradient_accumulation_steps=2,
            data_parallel_world_size=1,
            groups_per_rank_per_optimizer_step=3,
        )
    with pytest.raises(ValueError, match="more than one direct group per accumulation"):
        RepresentationAccumulationIdentityV2(
            gradient_accumulation_steps=2,
            data_parallel_world_size=1,
            groups_per_rank_per_optimizer_step=2,
        )

    adapter = _adapter(23)
    sampler = _sampler()
    optimizer = _optimizer(adapter)
    identity = replace(
        _run_identity(
            adapter,
            sampler=sampler,
            optimizer=optimizer,
            initialization_seed=23,
            with_scheduler=False,
        ),
        accumulation=partitioned,
    )
    path = tmp_path / "direct-groups-training.pt"
    save_representation_training_checkpoint_atomic(
        path,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=partitioned,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )

    loaded = load_representation_training_checkpoint(path)
    assert type(loaded.manifest.run_identity.accumulation) is (
        RepresentationAccumulationIdentityV2
    )
    assert loaded.manifest.run_identity.accumulation == partitioned


def test_run_identity_v3_binds_validation_and_planned_horizon_without_breaking_v2(
    tmp_path,
) -> None:
    adapter = _adapter(8)
    sampler = _sampler()
    optimizer = _optimizer(adapter)
    legacy = _run_identity(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=8,
        with_scheduler=True,
    )
    identity = _run_identity_v3(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=8,
        with_scheduler=True,
        planned_target_optimizer_steps=2,
    )

    assert type(legacy) is RepresentationRunIdentity
    assert legacy.schema_version == "representation-run-identity-v2"
    assert identity.schema_version == "representation-run-identity-v3"
    assert identity.validation_identity == _validation_identity()
    assert identity.planned_target_optimizer_steps == 2
    assert replace(identity, planned_target_optimizer_steps=3).identity_sha256 != (
        identity.identity_sha256
    )
    changed_validation = replace(
        identity.validation_identity,
        validation_sampler_seed=74,
    )
    assert replace(
        identity, validation_identity=changed_validation
    ).identity_sha256 != (identity.identity_sha256)
    with pytest.raises(ValueError, match="exceed the scheduler horizon"):
        replace(
            identity,
            planned_target_optimizer_steps=SCHEDULER_TOTAL_STEPS + 1,
        )

    path = tmp_path / "v3-adapter.pt"
    save_representation_adapter_artifact_atomic(
        path,
        adapter=adapter,
        run_identity=identity,
        global_step=0,
    )
    loaded = load_representation_adapter_artifact(path)
    assert loaded.manifest.run_identity == identity
    assert isinstance(loaded.manifest.run_identity, RepresentationRunIdentityV3)


def _training_step(
    adapter: TGVFAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = adapter(
        target_hidden_states=torch.randn(3, 6),
        pre_merge_visual_tokens=torch.randn(8, 4),
        deepstack_pre_merge_visual_tokens=tuple(torch.randn(8, 4) for _ in range(3)),
    )
    loss = output.main_d.square().mean() + sum(
        branch.square().mean() for branch in output.deepstack_visual_embeds
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def test_deployable_artifact_contains_only_adapter_owned_tensors(tmp_path) -> None:
    source = _adapter(11)
    sampler = _sampler()
    optimizer = _optimizer(source)
    identity = _run_identity(
        source,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=11,
        with_scheduler=False,
    )
    path = tmp_path / "adapter.pt"

    manifest = save_representation_adapter_artifact_atomic(
        path, adapter=source, run_identity=identity, global_step=9
    )
    artifact = load_representation_adapter_artifact(path)

    assert manifest.global_step == 9
    assert (
        artifact.manifest.artifact_identity_sha256 == manifest.artifact_identity_sha256
    )
    assert set(artifact.adapter_state) == set(source.artifact_state_dict())
    assert not any(
        name.startswith(("main_projection.", "d_deepstack_projections."))
        for name in artifact.adapter_state
    )

    target = _adapter(19)
    borrowed_names = set(target.state_dict()) - set(target.artifact_state_dict())
    borrowed_before = {
        name: target.state_dict()[name].clone() for name in borrowed_names
    }
    restore_representation_adapter_artifact(
        path, adapter=target, expected_run_identity=identity
    )

    for name, value in source.artifact_state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
    for name, value in borrowed_before.items():
        torch.testing.assert_close(target.state_dict()[name], value)

    wrong_identity = replace(identity, prompt_sha256="4" * 64)
    with pytest.raises(IdentityMismatchError, match="run identity mismatch"):
        restore_representation_adapter_artifact(
            path, adapter=target, expected_run_identity=wrong_identity
        )

    embedding_identity = replace(
        identity,
        provider=TargetConditioningConfig(
            provider=TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING,
            embedding_identity="qwen3-input-embedding@model-revision",
        ),
    )
    assert (
        embedding_identity.provider_identity_sha256 != identity.provider_identity_sha256
    )
    with pytest.raises(IdentityMismatchError, match="run identity mismatch"):
        restore_representation_adapter_artifact(
            path, adapter=target, expected_run_identity=embedding_identity
        )


def test_training_checkpoint_has_exact_cpu_next_step_resume_parity(tmp_path) -> None:
    source = _adapter(31)
    optimizer = _optimizer(source)
    scheduler = _scheduler(optimizer)
    sampler = _sampler()
    identity = _run_identity(
        source,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=31,
        with_scheduler=True,
    )

    torch.manual_seed(101)
    random.seed(202)
    _training_step(source, optimizer, scheduler)
    sampler.next_batch()
    path = tmp_path / "training.pt"
    saved_manifest = save_representation_training_checkpoint_atomic(
        path,
        adapter=source,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=1,
    )
    with pytest.raises(ReplayMismatchError, match="AdamW step counters"):
        save_representation_training_checkpoint_atomic(
            tmp_path / "wrong-global-step.pt",
            adapter=source,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
            global_step=2,
        )

    expected_python_draw = random.random()
    expected_batch = sampler.next_batch()
    expected_loss = _training_step(source, optimizer, scheduler)
    expected_state = {
        name: tensor.detach().clone()
        for name, tensor in source.artifact_state_dict().items()
    }

    resumed = _adapter(31)
    resumed_optimizer = _optimizer(resumed)
    resumed_scheduler = _scheduler(resumed_optimizer)
    resumed_sampler = _sampler()
    resumed_identity = _run_identity(
        resumed,
        sampler=resumed_sampler,
        optimizer=resumed_optimizer,
        initialization_seed=31,
        with_scheduler=True,
    )
    result = restore_representation_training_checkpoint(
        path,
        adapter=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        sampler=resumed_sampler,
        expected_run_identity=resumed_identity,
        accumulation=resumed_identity.accumulation,
        trainer_execution=resumed_identity.trainer_execution,
    )

    assert result.exact
    assert result.global_step == 1
    assert result.next_global_step == 2
    assert (
        result.checkpoint_identity_sha256 == saved_manifest.checkpoint_identity_sha256
    )
    assert random.random() == expected_python_draw
    assert resumed_sampler.next_batch() == expected_batch
    actual_loss = _training_step(resumed, resumed_optimizer, resumed_scheduler)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for name, expected in expected_state.items():
        torch.testing.assert_close(
            resumed.artifact_state_dict()[name], expected, rtol=0, atol=0
        )


def test_checkpoint_rejects_tensor_tampering_and_external_optimizer_parameter(
    tmp_path,
) -> None:
    adapter = _adapter(41)
    optimizer = _optimizer(adapter)
    sampler = _sampler()
    identity = _run_identity(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=41,
        with_scheduler=False,
    )
    path = tmp_path / "training.pt"
    save_representation_training_checkpoint_atomic(
        path,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )
    checkpoint = load_representation_training_checkpoint(path)
    first = next(iter(checkpoint.adapter_state.values()))
    first.add_(1)
    torch.save(checkpoint, path)
    with pytest.raises(ReplayMismatchError, match="Adapter state digest mismatch"):
        load_representation_training_checkpoint(path)

    external = nn.Parameter(torch.ones(()))
    invalid_optimizer = torch.optim.AdamW(
        [
            *(
                parameter
                for parameter in adapter.parameters()
                if parameter.requires_grad
            ),
            external,
        ],
        lr=3e-4,
    )
    with pytest.raises(ValueError, match="external parameter"):
        save_representation_training_checkpoint_atomic(
            tmp_path / "invalid.pt",
            adapter=adapter,
            optimizer=invalid_optimizer,
            scheduler=None,
            sampler=sampler,
            run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
            global_step=0,
        )


def test_resume_rejects_optimizer_scheduler_sampler_and_run_contract_drift(
    tmp_path,
) -> None:
    adapter = _adapter(51)
    optimizer = _optimizer(adapter)
    scheduler = _scheduler(optimizer)
    sampler = _sampler()
    identity = _run_identity(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=51,
        with_scheduler=True,
    )
    path = tmp_path / "strict-training.pt"
    save_representation_training_checkpoint_atomic(
        path,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )

    wrong_lr_optimizer = torch.optim.AdamW(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=9e-4,
    )
    wrong_lr_scheduler = _scheduler(wrong_lr_optimizer)
    with pytest.raises(IdentityMismatchError, match="optimizer hyperparameters"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=wrong_lr_optimizer,
            scheduler=wrong_lr_scheduler,
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )

    wrong_schedule_optimizer = _optimizer(adapter)
    wrong_scheduler = build_representation_scheduler(
        wrong_schedule_optimizer,
        RepresentationSchedulerConfig(
            kind=RepresentationSchedulerKind.LINEAR_WARMUP_DECAY,
            total_steps=SCHEDULER_TOTAL_STEPS,
            warmup_steps=2,
        ),
    )
    with pytest.raises(IdentityMismatchError, match="scheduler construction"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=wrong_schedule_optimizer,
            scheduler=wrong_scheduler,
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )

    fresh_optimizer = _optimizer(adapter)
    with pytest.raises(IdentityMismatchError, match="runtime accumulation"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=fresh_optimizer,
            scheduler=_scheduler(fresh_optimizer),
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=RepresentationAccumulationIdentity(2, 1),
            trainer_execution=identity.trainer_execution,
        )

    changed_execution = replace(
        identity,
        trainer_execution=replace(
            identity.trainer_execution,
            precision="bf16",
            max_grad_norm=0.5,
        ),
    )
    fresh_optimizer = _optimizer(adapter)
    with pytest.raises(IdentityMismatchError, match="precision/gradient clipping"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=fresh_optimizer,
            scheduler=_scheduler(fresh_optimizer),
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=changed_execution.trainer_execution,
        )

    changed_initialization = replace(
        identity,
        initialization=replace(identity.initialization, seed=52),
    )
    fresh_optimizer = _optimizer(adapter)
    with pytest.raises(IdentityMismatchError, match="run identity mismatch"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=fresh_optimizer,
            scheduler=_scheduler(fresh_optimizer),
            sampler=_sampler(),
            expected_run_identity=changed_initialization,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )

    wrong_sampler = SameImageBatchSampler(
        _samples(),
        batch_size=4,
        seed=74,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    fresh_optimizer = _optimizer(adapter)
    with pytest.raises(IdentityMismatchError, match="sampler contract"):
        restore_representation_training_checkpoint(
            path,
            adapter=adapter,
            optimizer=fresh_optimizer,
            scheduler=_scheduler(fresh_optimizer),
            sampler=wrong_sampler,
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )


def test_checkpoint_v1_run_identity_fails_closed(tmp_path) -> None:
    adapter = _adapter(61)
    optimizer = _optimizer(adapter)
    sampler = _sampler()
    identity = _run_identity(
        adapter,
        sampler=sampler,
        optimizer=optimizer,
        initialization_seed=61,
        with_scheduler=False,
    )
    path = tmp_path / "old-schema.pt"
    save_representation_training_checkpoint_atomic(
        path,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )
    checkpoint = load_representation_training_checkpoint(path)
    object.__setattr__(
        checkpoint.manifest.run_identity,
        "schema_version",
        "representation-run-identity-v1",
    )
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="run identity schema mismatch"):
        load_representation_training_checkpoint(path)


def test_rng_state_captures_and_restores_only_current_cuda_device(monkeypatch) -> None:
    cuda_state = torch.tensor([3, 1, 4, 1, 5], dtype=torch.uint8)
    get_calls: list[int] = []
    set_calls: list[tuple[torch.Tensor, int]] = []

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    def get_rng_state(device: int) -> torch.Tensor:
        get_calls.append(device)
        return cuda_state.clone()

    def set_rng_state(state: torch.Tensor, device: int) -> None:
        set_calls.append((state.clone(), device))

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("capture must not read every CUDA generator"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _states: pytest.fail("restore must not write every CUDA generator"),
    )

    state = capture_representation_rng_state()

    assert get_calls == [2]
    assert state["schema_version"] == "representation-rng-state-v2"
    assert set(state) == {"schema_version", "python", "torch_cpu", "torch_cuda"}
    assert isinstance(state["torch_cuda"], dict)
    assert state["torch_cuda"]["device_index"] == 2
    assert state["torch_cuda"]["visible_device_count"] == 4
    torch.testing.assert_close(state["torch_cuda"]["state"], cuda_state)

    restore_representation_rng_state(state)

    assert len(set_calls) == 1
    restored_state, restored_device = set_calls[0]
    assert restored_device == 2
    torch.testing.assert_close(restored_state, cuda_state)


@pytest.mark.parametrize(
    ("current_device", "device_count", "message"),
    (
        (1, 4, "current CUDA device index differs from checkpoint"),
        (2, 3, "visible CUDA device count differs from checkpoint"),
    ),
)
def test_rng_restore_rejects_local_cuda_topology_mismatch_before_writes(
    monkeypatch,
    current_device: int,
    device_count: int,
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: torch.tensor([device], dtype=torch.uint8),
    )
    state = capture_representation_rng_state()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda *_args, **_kwargs: pytest.fail(
            "topology mismatch must fail before restoring CUDA RNG"
        ),
    )

    with pytest.raises(ReplayMismatchError, match=message):
        restore_representation_rng_state(state)


def test_rng_state_remains_cpu_only_without_initialized_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("CPU capture must not inspect a CUDA device"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda _device: pytest.fail("CPU capture must not read CUDA RNG"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda *_args, **_kwargs: pytest.fail("CPU restore must not write CUDA RNG"),
    )

    state = capture_representation_rng_state()
    assert set(state) == {"schema_version", "python", "torch_cpu"}

    restore_representation_rng_state(state)
