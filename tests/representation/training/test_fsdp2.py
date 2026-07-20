from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest
import torch
from torch import nn

from tgvf_rl.representation import FrozenProjectionPort, TGVFAdapter
from tgvf_rl.representation.training import fsdp2 as fsdp2_module
from tgvf_rl.representation.training.fsdp2 import (
    SUPPORTED_REPRESENTATION_TORCH_IDENTITIES,
    RepresentationFSDP2Config,
    apply_representation_fsdp2,
    build_representation_fsdp2_plan,
)


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(16, 6)
        self.register_buffer("identity_scale", torch.ones(()))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens.reshape(-1, 16)) * self.identity_scale


class _QwenMergerOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main_merger = _Merger()
        self.deepstack_mergers = nn.ModuleList(_Merger() for _ in range(3))


def _projection(module: nn.Module, identity: str) -> FrozenProjectionPort:
    return FrozenProjectionPort(
        module,
        identity=identity,
        input_dim=4,
        output_dim=6,
        spatial_merge_size=2,
    )


def _owned_pair(*, device: str = "cpu") -> tuple[_QwenMergerOwner, TGVFAdapter]:
    with torch.device(device):
        qwen = _QwenMergerOwner()
        adapter = TGVFAdapter(
            d_lm=6,
            d_v=4,
            attn_dim=5,
            main_projection=_projection(qwen.main_merger, "qwen.main"),
            deepstack_projections=tuple(
                _projection(module, f"qwen.deepstack.{index}")
                for index, module in enumerate(qwen.deepstack_mergers)
            ),
            branch_layers=(8, 16, 24),
        )
    qwen.requires_grad_(False)
    qwen.eval()
    adapter.train(True)
    return qwen, adapter


def test_cpu_and_meta_plans_inventory_owned_and_borrowed_state() -> None:
    for device in ("cpu", "meta"):
        qwen, adapter = _owned_pair(device=device)
        plan = build_representation_fsdp2_plan(adapter, qwen)

        assert plan.owned_parameter_names
        assert all(
            not name.startswith(("main_projection.", "d_deepstack_projections."))
            for name in plan.owned_parameter_names
        )
        assert plan.borrowed_parameter_names
        assert all(
            name.startswith(("main_projection.", "d_deepstack_projections."))
            for name in plan.borrowed_parameter_names
        )
        assert plan.borrowed_buffer_names == tuple(
            f"merger.{index}.identity_scale" for index in range(4)
        )
        assert len(plan.owned_group_module_names) == 52
        assert plan.owned_parameter_numel > plan.borrowed_parameter_numel


def test_plan_rejects_wrong_qwen_owner_and_parameter_freeze_drift() -> None:
    qwen, adapter = _owned_pair()
    other_qwen, _ = _owned_pair()
    with pytest.raises(ValueError, match="not owned by the selected Qwen"):
        build_representation_fsdp2_plan(adapter, other_qwen)

    qwen.train(True)
    with pytest.raises(ValueError, match="eval mode"):
        build_representation_fsdp2_plan(adapter, qwen)
    qwen.eval()

    qwen.main_merger.projection.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Qwen parameter"):
        build_representation_fsdp2_plan(adapter, qwen)
    qwen.main_merger.projection.weight.requires_grad_(False)

    owned = next(
        parameter
        for name, parameter in adapter.named_parameters()
        if not name.startswith(("main_projection.", "d_deepstack_projections."))
    )
    owned.requires_grad_(False)
    with pytest.raises(ValueError, match="unexpectedly frozen"):
        build_representation_fsdp2_plan(adapter, qwen)


def test_apply_fails_before_mutation_without_initialized_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

    qwen, adapter = _owned_pair()
    original_class = adapter.__class__
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="must be initialized"):
        apply_representation_fsdp2(
            adapter=adapter,
            qwen_model=qwen,
            mesh=object(),
            config=RepresentationFSDP2Config(world_size=2, reshard_after_forward=True),
            mixed_precision_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            ),
            offload_policy=OffloadPolicy(),
        )

    assert adapter.__class__ is original_class
    assert all(not parameter.requires_grad for parameter in qwen.parameters())


class _FakeMixedPrecisionPolicy:
    pass


class _FakeOffloadPolicy:
    pass


class _FakeDeviceMesh:
    pass


class _FakeDTensor:
    pass


class _FakeFSDPModule:
    def reshard(self) -> None:
        calls = getattr(self, "_fake_reshard_calls", 0)
        self._fake_reshard_calls = calls + 1


class _RecordingAccumulationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []

    def set_requires_gradient_sync(self, value: bool, *, recurse: bool) -> None:
        self.calls.append(("sync", value, recurse))

    def set_reshard_after_backward(self, value: bool, *, recurse: bool) -> None:
        self.calls.append(("reshard", value, recurse))

    def set_is_last_backward(self, value: bool) -> None:
        self.calls.append(("last", value))


@dataclass
class _FullyShardCall:
    modules: list[object]
    kwargs: list[dict[str, object]]


def test_fsdp2_accumulation_state_suppresses_nonfinal_communication() -> None:
    module = _RecordingAccumulationModule()

    fsdp2_module._set_fsdp2_accumulation_state(
        module,
        requires_gradient_sync=False,
        reshard_after_backward=False,
        is_last_backward=False,
    )
    fsdp2_module._set_fsdp2_accumulation_state(
        module,
        requires_gradient_sync=True,
        reshard_after_backward=True,
        is_last_backward=True,
    )

    assert module.calls == [
        ("sync", False, True),
        ("reshard", False, True),
        ("last", False),
        ("sync", True, True),
        ("reshard", True, True),
        ("last", True),
    ]


def test_apply_ignores_exact_borrowed_state_and_optimizer_owns_only_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen, adapter = _owned_pair()
    plan = build_representation_fsdp2_plan(adapter, qwen)
    borrowed_before = tuple(
        parameter
        for name, parameter in adapter.named_parameters()
        if name.startswith(("main_projection.", "d_deepstack_projections."))
    )
    buffers_before = tuple(
        buffer
        for merger in (qwen.main_merger, *tuple(qwen.deepstack_mergers))
        for buffer in merger.buffers()
    )
    call = _FullyShardCall(modules=[], kwargs=[])

    def fake_fully_shard(module, **kwargs):
        call.modules.append(module)
        call.kwargs.append(kwargs)
        targets = module if isinstance(module, list) else [module]
        for target in targets:
            fsdp_class = type(
                f"FakeFSDP{target.__class__.__name__}",
                (_FakeFSDPModule, target.__class__),
                {},
            )
            target.__class__ = fsdp_class
        return module

    api = fsdp2_module._FSDP2API(
        fully_shard=fake_fully_shard,
        fsdp_module_type=_FakeFSDPModule,
        mixed_precision_policy_type=_FakeMixedPrecisionPolicy,
        offload_policy_type=_FakeOffloadPolicy,
        device_mesh_type=_FakeDeviceMesh,
        dtensor_type=_FakeDTensor,
    )
    monkeypatch.setattr(fsdp2_module, "_load_fsdp2_api", lambda: api)
    monkeypatch.setattr(
        fsdp2_module,
        "_assert_distributed_prerequisites",
        lambda **_kwargs: None,
    )

    binding = apply_representation_fsdp2(
        adapter=adapter,
        qwen_model=qwen,
        mesh=_FakeDeviceMesh(),
        config=RepresentationFSDP2Config(world_size=2, reshard_after_forward=True),
        mixed_precision_policy=_FakeMixedPrecisionPolicy(),
        offload_policy=_FakeOffloadPolicy(),
    )

    assert len(call.modules) == 2
    assert isinstance(call.modules[0], list)
    assert tuple(call.modules[0]) == binding.owned_group_modules
    assert call.modules[1] is adapter
    for kwargs in call.kwargs:
        ignored = kwargs["ignored_params"]
        assert isinstance(ignored, set)
        assert {id(parameter) for parameter in ignored} == {
            id(parameter) for parameter in borrowed_before
        }
        assert kwargs["reshard_after_forward"] is True
    assert binding.plan == plan
    assert tuple(
        id(parameter) for parameter in binding.borrowed_qwen_merger_parameters
    ) == tuple(id(parameter) for parameter in borrowed_before)
    assert tuple(
        id(buffer) for buffer in binding.borrowed_qwen_merger_buffers
    ) == tuple(id(buffer) for buffer in buffers_before)

    optimizer = torch.optim.AdamW(binding.optimizer_parameters(), lr=1e-4)
    binding.assert_optimizer_ownership(optimizer)
    binding.reshard_owned_parameters()
    assert all(
        getattr(module, "_fake_reshard_calls", 0) == 1
        for module in binding.owned_group_modules
    )
    binding.assert_optimizer_ownership(optimizer)
    polluted = torch.optim.AdamW(
        (*binding.optimizer_parameters(), borrowed_before[0]), lr=1e-4
    )
    with pytest.raises(ValueError, match="every and only"):
        binding.assert_optimizer_ownership(polluted)


def test_fsdp2_config_and_supported_torch_api_identity_are_exact() -> None:
    with pytest.raises(ValueError, match="at least two ranks"):
        RepresentationFSDP2Config(world_size=1, reshard_after_forward=True)
    with pytest.raises(TypeError, match="explicit bool"):
        RepresentationFSDP2Config(
            world_size=2,
            reshard_after_forward=1,  # type: ignore[arg-type]
        )

    assert SUPPORTED_REPRESENTATION_TORCH_IDENTITIES == (
        ("2.9.0", "2.9.0+cu128"),
        ("2.11.0+cu129", "2.11.0+cu129"),
    )
    api = fsdp2_module._load_fsdp2_api()
    assert tuple(inspect.signature(api.fully_shard).parameters) == (
        "module",
        "mesh",
        "reshard_after_forward",
        "shard_placement_fn",
        "mp_policy",
        "offload_policy",
        "ignored_params",
    )
    assert api.fsdp_module_type.__module__ == "torch.distributed.fsdp"
    assert tuple(inspect.signature(api.fsdp_module_type.reshard).parameters) == (
        "self",
    )
    assert api.fsdp_module_type.__name__ == "FSDPModule"
    assert api.device_mesh_type.__module__ == "torch.distributed.device_mesh"
    assert api.device_mesh_type.__name__ == "DeviceMesh"
    assert api.dtensor_type.__module__ == "torch.distributed.tensor"
    assert api.dtensor_type.__name__ == "DTensor"


@pytest.mark.parametrize(
    ("distribution_version", "runtime_version"),
    (
        ("2.9.0", "2.9.0"),
        ("2.9.1", "2.9.1+cu128"),
        ("2.9.0", "2.9.0+cu129"),
        ("2.10.0+cu129", "2.10.0+cu129"),
        ("2.11.0+cu129", "2.11.0"),
        ("2.11.1+cu129", "2.11.1+cu129"),
        ("2.11.0+cu129", "2.11.0+cu128"),
        ("2.9.0", "2.11.0+cu129"),
        ("unparseable", "unparseable"),
    ),
)
def test_fsdp2_api_rejects_unaccepted_torch_identities(
    monkeypatch: pytest.MonkeyPatch,
    distribution_version: str,
    runtime_version: str,
) -> None:
    monkeypatch.setattr(
        fsdp2_module.metadata, "version", lambda _name: distribution_version
    )
    monkeypatch.setattr(torch, "__version__", runtime_version)
    with pytest.raises(RuntimeError, match="exact audited torch identity"):
        fsdp2_module._load_fsdp2_api()


def test_fsdp2_api_rejects_public_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.fsdp as torch_fsdp

    monkeypatch.setattr(torch_fsdp, "fully_shard", lambda module: module)
    with pytest.raises(RuntimeError, match="fully_shard public signature drifted"):
        fsdp2_module._load_fsdp2_api()
