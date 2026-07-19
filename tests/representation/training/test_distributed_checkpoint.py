from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import CodeIdentity, ModelIdentity
from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training.checkpoint import (
    RepresentationAccumulationIdentity,
    RepresentationAdapterContractIdentity,
    RepresentationInitializationIdentity,
    RepresentationOptimizerIdentity,
    RepresentationRunIdentity,
    RepresentationSamplerContractIdentity,
    RepresentationTrainerExecutionIdentity,
)
from tgvf_rl.representation.training import distributed_checkpoint as dcp_module
from tgvf_rl.representation.training.distributed_checkpoint import (
    RankZeroAdapterOwnedStateExport,
    gather_rank_zero_full_adapter_owned_state,
    load_rank_zero_adapter_owned_state_export,
    restore_distributed_representation_checkpoint,
    save_distributed_representation_checkpoint_atomic,
    save_rank_zero_adapter_owned_state_export_atomic,
)
from tgvf_rl.representation.training.fsdp2 import (
    RepresentationFSDP2Binding,
    RepresentationFSDP2Config,
    build_representation_fsdp2_plan,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfig,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.sampling import (
    SameImageBatchSampler,
    same_image_group_owner,
)
from tgvf_rl.representation.training.schema import RepresentationTrainingSample


DATA_MANIFEST_SHA256 = "2" * 64


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(16, 6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.reshape(-1, 16))


class _QwenOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = _Merger()
        self.branches = nn.ModuleList(_Merger() for _ in range(3))


def _projection(module: nn.Module, identity: str) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        module,
        identity=identity,
        input_dim=4,
        output_dim=6,
        spatial_merge_size=2,
    )


def _adapter_pair(seed: int) -> tuple[_QwenOwner, TGVFAdapter]:
    torch.manual_seed(seed)
    qwen = _QwenOwner()
    adapter = TGVFAdapter(
        d_lm=6,
        d_v=4,
        attn_dim=5,
        main_projection=_projection(qwen.main, "qwen.main@revision"),
        deepstack_projections=tuple(
            _projection(module, f"qwen.branch.{index}@revision")
            for index, module in enumerate(qwen.branches)
        ),
        branch_layers=(8, 16, 24),
    )
    qwen.requires_grad_(False)
    qwen.eval()
    return qwen, adapter


class _FakeFSDPModule:
    pass


def _binding(seed: int) -> RepresentationFSDP2Binding:
    qwen, adapter = _adapter_pair(seed)
    plan = build_representation_fsdp2_plan(adapter, qwen)
    named_modules = dict(adapter.named_modules())
    owned_modules = tuple(named_modules[name] for name in plan.owned_group_module_names)
    for module in (*owned_modules, adapter):
        module.__class__ = type(
            f"FakeFSDP{module.__class__.__name__}",
            (_FakeFSDPModule, module.__class__),
            {},
        )
    borrowed = tuple(
        parameter
        for name, parameter in adapter.named_parameters()
        if name.startswith(("main_projection.", "d_deepstack_projections."))
    )
    return RepresentationFSDP2Binding(
        adapter=adapter,
        config=RepresentationFSDP2Config(world_size=2, reshard_after_forward=True),
        mesh=object(),
        plan=plan,
        _borrowed_qwen_merger_parameters=borrowed,
        _borrowed_qwen_merger_buffers=(),
        _owned_group_modules=owned_modules,
    )


def _samples() -> tuple[RepresentationTrainingSample, ...]:
    keys_by_owner: dict[int, list[str]] = {0: [], 1: []}
    candidate = 0
    while any(len(keys) < 2 for keys in keys_by_owner.values()):
        key = f"image-{candidate}"
        owner = same_image_group_owner(key, world_size=2)
        if len(keys_by_owner[owner]) < 2:
            keys_by_owner[owner].append(key)
        candidate += 1
    return tuple(
        RepresentationTrainingSample(
            sample_id=f"{image}-{member}",
            image=f"images/{image}.jpg",
            image_id=image,
            question="What is visible?",
            target=f"target {member}",
            evidence_description=f"evidence {member}",
        )
        for owner in (0, 1)
        for image in keys_by_owner[owner]
        for member in range(4)
    )


def _sampler(rank: int = 0) -> SameImageBatchSampler:
    return SameImageBatchSampler(
        _samples(),
        batch_size=4,
        seed=73,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        rank=rank,
        world_size=2,
    )


def _optimizer(binding: RepresentationFSDP2Binding) -> torch.optim.AdamW:
    return torch.optim.AdamW(binding.optimizer_parameters(), lr=3e-4)


def _identity(
    binding: RepresentationFSDP2Binding,
    optimizer: torch.optim.Optimizer,
    sampler: SameImageBatchSampler,
    *,
    initialization_seed: int,
) -> RepresentationRunIdentity:
    return RepresentationRunIdentity(
        run_id="distributed-cpu-contract",
        code=CodeIdentity("Miocio-nora/TGVF-E2E-RL", "fixture-commit"),
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
            identity="distributed-objective",
            kind=RepresentationObjectiveKind.MATRIX_CE_AND_L_GEN,
            matrix_ce_weight=1.0,
            l_gen_weight=0.25,
        ),
        adapter_contract=RepresentationAdapterContractIdentity.from_adapter(
            binding.adapter
        ),
        accumulation=RepresentationAccumulationIdentity(1, 2),
        optimizer=RepresentationOptimizerIdentity.from_optimizer(optimizer),
        scheduler=None,
        trainer_execution=RepresentationTrainerExecutionIdentity(
            precision="fp32",
            max_grad_norm=1.0,
            require_all_adapter_gradients=True,
        ),
        initialization=RepresentationInitializationIdentity.from_adapter(
            binding.adapter,
            kind="fresh_random",
            seed=initialization_seed,
            source_artifact_sha256=None,
        ),
        sampler_contract=RepresentationSamplerContractIdentity.from_sampler(sampler),
    )


@dataclass
class _FakeOptions:
    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False
    flatten_optimizer_state_dict: bool = False


class _FakeDCP:
    def __init__(self) -> None:
        self.saved: dict[str, object] | None = None
        self.options: list[_FakeOptions] = []
        self.leak_borrowed = False
        self.calls: list[str] = []

    def get_model_state_dict(self, model, *, options):
        self.calls.append("get_model_state_dict")
        self.options.append(options)
        state = {
            name: value.detach().clone()
            for name, value in model.artifact_state_dict().items()
        }
        if self.leak_borrowed:
            borrowed_name, borrowed = next(
                (name, value)
                for name, value in model.named_parameters()
                if name.startswith("main_projection.")
            )
            state[borrowed_name] = borrowed.detach().clone()
        return state

    def get_optimizer_state_dict(self, model, optimizer, *, options):
        del model
        self.calls.append("get_optimizer_state_dict")
        self.options.append(options)
        return deepcopy(optimizer.state_dict())

    def save(self, state, *, checkpoint_id, process_group):
        del process_group
        self.calls.append("dcp_save")
        checkpoint_id.mkdir(parents=True)
        self.saved = deepcopy(state)
        torch.save(self.saved, checkpoint_id / "fake_payload.pt")

    def load(self, state, *, checkpoint_id, process_group):
        del process_group
        self.calls.append("dcp_load")
        loaded = torch.load(
            checkpoint_id / "fake_payload.pt",
            map_location="cpu",
            weights_only=False,
        )
        state.clear()
        state.update(loaded)

    def set_model_state_dict(self, model, state, *, options):
        self.calls.append("set_model_state_dict")
        self.options.append(options)
        model.load_artifact_state_dict(state)
        missing = tuple(
            name
            for name, _ in model.named_parameters()
            if name.startswith(("main_projection.", "d_deepstack_projections."))
        )
        return SimpleNamespace(missing_keys=list(missing), unexpected_keys=[])

    def set_optimizer_state_dict(self, model, optimizer, state, *, options):
        del model
        self.calls.append("set_optimizer_state_dict")
        self.options.append(options)
        optimizer.load_state_dict(state)

    def api(self) -> dcp_module._DistributedCheckpointAPI:
        return dcp_module._DistributedCheckpointAPI(
            get_model_state_dict=self.get_model_state_dict,
            get_optimizer_state_dict=self.get_optimizer_state_dict,
            set_model_state_dict=self.set_model_state_dict,
            set_optimizer_state_dict=self.set_optimizer_state_dict,
            dcp_save=self.save,
            dcp_load=self.load,
            state_dict_options_type=_FakeOptions,
            fsdp_module_type=_FakeFSDPModule,
        )


def _mock_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialized: bool = True,
    peer_failure_phase: str | None = None,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _values, **_kwargs: None,
    )

    def all_gather_object(output, value, group=None):
        del group
        output[0] = deepcopy(value)
        other = deepcopy(value)
        other["rank"] = 1
        if value.get("kind") == dcp_module._COLLECTIVE_OUTCOME_KIND:
            if value.get("phase") == peer_failure_phase:
                other["error_type"] = "builtins.RuntimeError"
                other["error_message"] = "injected peer failure"
        else:
            other["sampler_identity_sha256"] = "f" * 64
            other["sampler_state"]["identity_sha256"] = "f" * 64
        output[1] = other

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)


def test_distributed_checkpoint_fails_closed_without_initialized_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    binding = _binding(11)
    optimizer = _optimizer(binding)
    sampler = _sampler()
    identity = _identity(binding, optimizer, sampler, initialization_seed=11)
    fake = _FakeDCP()
    monkeypatch.setattr(dcp_module, "_load_distributed_checkpoint_api", fake.api)
    _mock_distributed(monkeypatch, initialized=False)

    with pytest.raises(RuntimeError, match="requires initialized"):
        save_distributed_representation_checkpoint_atomic(
            tmp_path / "checkpoint",
            binding=binding,
            optimizer=optimizer,
            scheduler=None,
            sampler=sampler,
            run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
            global_step=0,
        )
    assert not fake.calls


def test_sharded_public_dcp_round_trip_preserves_owned_state_and_rank_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake = _FakeDCP()
    monkeypatch.setattr(dcp_module, "_load_distributed_checkpoint_api", fake.api)
    _mock_distributed(monkeypatch)
    source = _binding(21)
    optimizer = _optimizer(source)
    sampler = _sampler()
    sampler.next_batch()
    identity = _identity(source, optimizer, sampler, initialization_seed=21)
    torch.manual_seed(901)
    random.seed(902)
    expected_state = {
        name: value.detach().clone()
        for name, value in source.adapter.artifact_state_dict().items()
    }
    path = tmp_path / "distributed"

    manifest = save_distributed_representation_checkpoint_atomic(
        path,
        binding=source,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    expected_batch = sampler.next_batch()

    assert path.is_dir()
    assert manifest.owned_state_names == tuple(sorted(expected_state))
    assert len(manifest.model_local_shard_sha256) == 2
    assert len(manifest.optimizer_local_shard_sha256) == 2
    assert all(len(value) == 64 for value in manifest.model_local_shard_sha256)
    assert all(len(value) == 64 for value in manifest.optimizer_local_shard_sha256)
    assert not any(
        name.startswith(("main_projection.", "d_deepstack_projections."))
        for name in manifest.owned_state_names
    )
    assert fake.calls[:3] == [
        "get_model_state_dict",
        "get_optimizer_state_dict",
        "dcp_save",
    ]
    assert fake.options[0].full_state_dict is False
    assert fake.options[0].ignore_frozen_params is True
    assert fake.options[0].strict is True

    target = _binding(22)
    target_optimizer = _optimizer(target)
    target_sampler = _sampler()
    result = restore_distributed_representation_checkpoint(
        path,
        binding=target,
        optimizer=target_optimizer,
        scheduler=None,
        sampler=target_sampler,
        expected_run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
    )

    assert result.exact and result.global_step == 0 and result.next_global_step == 1
    for name, expected in expected_state.items():
        torch.testing.assert_close(
            target.adapter.artifact_state_dict()[name], expected, rtol=0, atol=0
        )
    assert target_sampler.next_batch() == expected_batch
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    assert "dcp_load" in fake.calls
    assert "set_model_state_dict" in fake.calls
    assert "set_optimizer_state_dict" in fake.calls

    fake.leak_borrowed = True
    with pytest.raises(ReplayMismatchError, match="leaked borrowed Qwen"):
        save_distributed_representation_checkpoint_atomic(
            tmp_path / "borrowed-leak",
            binding=source,
            optimizer=optimizer,
            scheduler=None,
            sampler=sampler,
            run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
            global_step=0,
        )

    (path / "representation_metadata.sha256").write_text(
        f"{'0' * 64}\n", encoding="ascii"
    )
    rejected = _binding(23)
    rejected_optimizer = _optimizer(rejected)
    with pytest.raises(ReplayMismatchError, match="metadata digest mismatch"):
        restore_distributed_representation_checkpoint(
            path,
            binding=rejected,
            optimizer=rejected_optimizer,
            scheduler=None,
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )


@pytest.mark.parametrize(
    ("swapped_section", "message"),
    (
        ("adapter", "DCP Adapter local-shard content digest mismatch"),
        ("optimizer", "DCP optimizer local-shard content digest mismatch"),
    ),
)
def test_restore_rejects_swapped_dcp_payload_bound_to_another_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    swapped_section: str,
    message: str,
) -> None:
    fake = _FakeDCP()
    monkeypatch.setattr(dcp_module, "_load_distributed_checkpoint_api", fake.api)
    _mock_distributed(monkeypatch)
    source = _binding(41)
    optimizer = _optimizer(source)
    sampler = _sampler()
    identity = _identity(source, optimizer, sampler, initialization_seed=41)
    original_path = tmp_path / f"original-{swapped_section}"
    foreign_path = tmp_path / f"foreign-{swapped_section}"

    save_distributed_representation_checkpoint_atomic(
        original_path,
        binding=source,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )
    if swapped_section == "adapter":
        with torch.no_grad():
            next(iter(source.adapter.artifact_state_dict().values())).add_(1)
    else:
        parameter = optimizer.param_groups[0]["params"][0]
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.full_like(parameter, 2.0),
        }
    save_distributed_representation_checkpoint_atomic(
        foreign_path,
        binding=source,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=1,
    )
    shutil.rmtree(original_path / "dcp")
    shutil.copytree(foreign_path / "dcp", original_path / "dcp")

    target = _binding(42)
    target_optimizer = _optimizer(target)
    set_model_calls = fake.calls.count("set_model_state_dict")
    with pytest.raises(ReplayMismatchError, match=message):
        restore_distributed_representation_checkpoint(
            original_path,
            binding=target,
            optimizer=target_optimizer,
            scheduler=None,
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )
    assert fake.calls.count("set_model_state_dict") == set_model_calls


@pytest.mark.parametrize(
    ("failure_phase", "dcp_load_delta", "set_model_delta"),
    (
        ("distributed checkpoint restore metadata preflight", 0, 0),
        ("distributed checkpoint DCP load", 1, 0),
        ("distributed checkpoint restore apply", 1, 1),
    ),
)
def test_restore_agrees_on_peer_failure_before_entering_or_escaping_next_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    failure_phase: str,
    dcp_load_delta: int,
    set_model_delta: int,
) -> None:
    fake = _FakeDCP()
    monkeypatch.setattr(dcp_module, "_load_distributed_checkpoint_api", fake.api)
    _mock_distributed(monkeypatch)
    source = _binding(51)
    optimizer = _optimizer(source)
    sampler = _sampler()
    identity = _identity(source, optimizer, sampler, initialization_seed=51)
    path = tmp_path / failure_phase.rsplit(" ", 1)[-1]
    save_distributed_representation_checkpoint_atomic(
        path,
        binding=source,
        optimizer=optimizer,
        scheduler=None,
        sampler=sampler,
        run_identity=identity,
        accumulation=identity.accumulation,
        trainer_execution=identity.trainer_execution,
        global_step=0,
    )
    _mock_distributed(monkeypatch, peer_failure_phase=failure_phase)
    before_load = fake.calls.count("dcp_load")
    before_set = fake.calls.count("set_model_state_dict")

    target = _binding(52)
    target_optimizer = _optimizer(target)
    with pytest.raises(RuntimeError, match="injected peer failure"):
        restore_distributed_representation_checkpoint(
            path,
            binding=target,
            optimizer=target_optimizer,
            scheduler=None,
            sampler=_sampler(),
            expected_run_identity=identity,
            accumulation=identity.accumulation,
            trainer_execution=identity.trainer_execution,
        )
    assert fake.calls.count("dcp_load") == before_load + dcp_load_delta
    assert fake.calls.count("set_model_state_dict") == before_set + set_model_delta


def test_local_shard_digest_is_deterministic_for_dtensor_and_nested_optimizer_state(
    tmp_path,
) -> None:
    if torch.distributed.is_initialized():
        pytest.skip("digest fixture requires ownership of the process group")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.distributed.init_process_group(
        "gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{tmp_path / 'digest-rendezvous'}",
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        moment = distribute_tensor(
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
            mesh,
            (Shard(0),),
        )
        nested = {
            "state": {
                "adapter.weight": {
                    "step": torch.tensor(7.0),
                    "exp_avg": moment,
                    "exp_avg_sq": torch.ones(2, 3),
                }
            },
            "param_groups": [
                {
                    "params": ["adapter.weight"],
                    "betas": (0.9, 0.95),
                    "foreach": None,
                }
            ],
        }
        reordered = {
            "param_groups": nested["param_groups"],
            "state": nested["state"],
        }
        first = dcp_module._local_shard_state_digest(nested)
        assert first == dcp_module._local_shard_state_digest(nested)
        assert first == dcp_module._local_shard_state_digest(reordered)

        moment.to_local().add_(1)
        assert first != dcp_module._local_shard_state_digest(nested)
    finally:
        torch.distributed.destroy_process_group()


def test_rank_zero_full_export_uses_full_cpu_options_and_rejects_borrowed_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake = _FakeDCP()
    monkeypatch.setattr(dcp_module, "_load_distributed_checkpoint_api", fake.api)
    _mock_distributed(monkeypatch)
    binding = _binding(31)
    optimizer = _optimizer(binding)
    sampler = _sampler()
    identity = _identity(binding, optimizer, sampler, initialization_seed=31)

    exported = gather_rank_zero_full_adapter_owned_state(
        binding=binding,
        run_identity=identity,
        global_step=7,
    )
    assert exported.is_writer and exported.state is not None
    assert tuple(sorted(exported.state)) == exported.manifest.tensor_names
    options = fake.options[-1]
    assert options.full_state_dict is True
    assert options.cpu_offload is True
    assert options.ignore_frozen_params is True
    assert options.strict is True
    assert all(value.device.type == "cpu" for value in exported.state.values())

    artifact_path = tmp_path / "adapter-owned.pt"
    assert save_rank_zero_adapter_owned_state_export_atomic(artifact_path, exported)
    loaded = load_rank_zero_adapter_owned_state_export(
        artifact_path, expected_run_identity=identity
    )
    assert loaded.manifest == exported.manifest
    assert loaded.state is not None
    for name, expected in exported.state.items():
        torch.testing.assert_close(loaded.state[name], expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError, match="never overwrite"):
        save_rank_zero_adapter_owned_state_export_atomic(artifact_path, exported)

    non_writer_path = tmp_path / "must-not-exist.pt"
    non_writer = RankZeroAdapterOwnedStateExport(
        manifest=exported.manifest,
        state=None,
    )
    assert not save_rank_zero_adapter_owned_state_export_atomic(
        non_writer_path, non_writer
    )
    assert not non_writer_path.exists()

    assert loaded.state is not None
    next(iter(loaded.state.values())).add_(1)
    torch.save(loaded, artifact_path)
    with pytest.raises(ReplayMismatchError, match="tensor checksum mismatch"):
        load_rank_zero_adapter_owned_state_export(artifact_path)

    fake.leak_borrowed = True
    with pytest.raises(ReplayMismatchError, match="borrowed Qwen prefix"):
        gather_rank_zero_full_adapter_owned_state(
            binding=binding,
            run_identity=identity,
            global_step=7,
        )
