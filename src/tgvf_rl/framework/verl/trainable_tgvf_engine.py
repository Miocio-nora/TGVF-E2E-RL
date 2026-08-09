"""FSDP2 engine for full-Qwen and jointly trainable RP66 replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ComponentRole
from tgvf_rl.observations.store import TrajectoryReplayBundle
from tgvf_rl.policy.trainable_tgvf_replay import (
    TRAINABLE_TGVF_ADAPTER_ATTRIBUTE,
    TrainableTGVFCurrentReplayPort,
)
from tgvf_rl.representation.adapter import TGVFAdapter, TGVFAdapterVariant
from tgvf_rl.representation.deepstack import (
    DDeepStackProjectionPorts,
    TrainableBorrowedProjectionPort,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)

from .exact_replay_engine import (
    _qwen3_forward_binding,
    _reshard_exact_replay_root,
    exact_replay_forward_step,
)
from .fused_exact_replay import fused_selected_next_token_logprobs
from .trainable_tgvf_weight_sync import (
    split_trainable_rp66_parameter_stream_for_snapshot,
)


TRAINABLE_TGVF_MODEL_TYPE = "tgvf_trainable_rp66_language_model"
TRAINABLE_TGVF_RUN_CONFIG_ENV = "TGVF_POLICY_RUN_CONFIG_PATH"


@dataclass(frozen=True, slots=True)
class TrainableTGVFReplayPortFactory:
    def __call__(
        self,
        *,
        engine: Any,
        model: nn.Module,
        role: ComponentRole,
        bundle: TrajectoryReplayBundle,
        model_training: bool,
    ) -> Any:
        _validate_worker(engine, model=model, role=role, bundle=bundle)
        materializer = (
            fused_selected_next_token_logprobs
            if bool(getattr(engine.model_config, "use_fused_kernels", False))
            else None
        )
        if role is ComponentRole.CURRENT:
            return TrainableTGVFCurrentReplayPort(
                engine=engine,
                model=model,
                selected_logprob_materializer=materializer,
            )

        from tgvf_rl.policy.qwen_replay import Qwen3RecordedPolicyForwardPort

        binding = _qwen3_forward_binding(
            engine=engine,
            model=model,
            role=role,
            bundle=bundle,
            model_training=model_training,
        )
        return Qwen3RecordedPolicyForwardPort(
            model=model,
            binding=binding,
            selected_logprob_materializer=materializer,
        )


_PORT_FACTORY = TrainableTGVFReplayPortFactory()
_ENGINE_CLASS: type[Any] | None = None


def make_trainable_tgvf_fsdp2_engine_class(
    upstream_engine_cls: type[Any],
) -> type[Any]:
    class TrainableTGVFFSDPEngineWithLMHead(upstream_engine_cls):
        def _build_module(self):
            if getattr(self.engine_config, "strategy", None) != "fsdp2":
                raise ValueError("trainable RP66 engine supports FSDP2 only")
            if getattr(self.model_config, "model_type", None) != TRAINABLE_TGVF_MODEL_TYPE:
                raise IdentityMismatchError("trainable RP66 model_type differs")
            _validate_trainable_execution_capabilities(self.model_config)
            self.model_config.model_type = "language_model"
            try:
                module = super()._build_module()
            finally:
                self.model_config.model_type = TRAINABLE_TGVF_MODEL_TYPE
            if bool(getattr(self.engine_config, "forward_only", False)):
                module.requires_grad_(False)
                module.eval()
                return module
            module.requires_grad_(True)
            adapter = build_trainable_rp66_adapter(
                module,
                run_config_path=_required_run_config_path(),
            )
            module.add_module(TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, adapter)
            module.train(True)
            return module

        def get_per_tensor_param(
            self,
            layered_summon=False,
            base_sync_done=False,
            **kwargs,
        ):
            result = super().get_per_tensor_param(
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
                **kwargs,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("FSDP2 weight stream must return (weights, peft)")
            stream, peft_config = result
            if bool(getattr(self.engine_config, "forward_only", False)):
                return stream, peft_config
            if peft_config is not None:
                raise ValueError("trainable RP66 pilot forbids policy LoRA")
            return (
                split_trainable_rp66_parameter_stream_for_snapshot(
                    stream,
                    base_sync_done=bool(base_sync_done),
                ),
                None,
            )

        def forward_step(self, micro_batch, loss_function, forward_only):
            return exact_replay_forward_step(
                engine=self,
                micro_batch=micro_batch,
                loss_function=loss_function,
                forward_only=forward_only,
                port_factory=_PORT_FACTORY,
            )

        def forward_backward_batch(self, *args, **kwargs):
            try:
                return super().forward_backward_batch(*args, **kwargs)
            finally:
                _reshard_exact_replay_root(self.module)

    TrainableTGVFFSDPEngineWithLMHead.__name__ = (
        "TrainableTGVFFSDPEngineWithLMHead"
    )
    TrainableTGVFFSDPEngineWithLMHead.__qualname__ = (
        "TrainableTGVFFSDPEngineWithLMHead"
    )
    TrainableTGVFFSDPEngineWithLMHead.__module__ = __name__
    return TrainableTGVFFSDPEngineWithLMHead


def register_trainable_tgvf_fsdp2_engine(
    *, registry: Any | None = None, upstream_engine_cls: type[Any] | None = None
) -> type[Any]:
    global _ENGINE_CLASS
    if _ENGINE_CLASS is not None and registry is None and upstream_engine_cls is None:
        return _ENGINE_CLASS
    if registry is None or upstream_engine_cls is None:
        from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead

        registry = registry or EngineRegistry
        upstream_engine_cls = upstream_engine_cls or FSDPEngineWithLMHead
    engine_cls = make_trainable_tgvf_fsdp2_engine_class(upstream_engine_cls)
    registered = registry.register(
        model_type=TRAINABLE_TGVF_MODEL_TYPE,
        backend="fsdp2",
        device=["cuda", "npu"],
    )(engine_cls)
    if registry.__name__ == "EngineRegistry":
        _ENGINE_CLASS = registered
    return registered


def build_trainable_rp66_adapter(
    model: nn.Module, *, run_config_path: str | Path
) -> TGVFAdapter:
    """Load RP66 owned tensors while borrowing canonical live Qwen mergers."""

    from tgvf_rl.policy.run_config import load_policy_e2e_smoke_run_config

    config = load_policy_e2e_smoke_run_config(run_config_path)
    export = _load_validated_trainable_rp66_export(config)

    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is None:
        raise TypeError("Qwen actor does not expose model.visual")
    mergers = (visual.merger, *tuple(visual.deepstack_merger_list))
    contract = export.manifest.run_identity.adapter_contract
    identities = (
        contract.main_projection_identity,
        *contract.deepstack_projection_identities,
    )
    ports = tuple(
        TrainableBorrowedProjectionPort(
            merger,
            identity=identity,
            input_dim=contract.d_v,
            output_dim=contract.d_lm,
            spatial_merge_size=contract.spatial_merge_size,
        )
        for merger, identity in zip(mergers, identities, strict=True)
    )
    variant = TGVFAdapterVariant(
        getattr(contract, "variant", TGVFAdapterVariant.FULL_D_DEEPSTACK.value)
    )
    adapter = TGVFAdapter(
        d_lm=contract.d_lm,
        d_v=contract.d_v,
        attn_dim=contract.attention_dim,
        main_projection=ports[0],
        deepstack_projections=DDeepStackProjectionPorts(
            branch_layers=contract.deepstack_branch_layers,
            projections=ports[1:],
        ),
        branch_layers=contract.deepstack_branch_layers,
        variant=variant,
    )
    owner = next(model.parameters())
    # FSDP2 builds non-loading ranks under a meta init context.  Adapter-owned
    # RP66 tensors must nevertheless be materialized before its full state is
    # captured and sharded; copying a real checkpoint into meta parameters is
    # a silent no-op.  The borrowed Qwen merger ports remain weak references to
    # the canonical meta modules and are materialized by FSDP2 itself.
    adapter_device = owner.device
    if adapter_device.type == "meta":
        adapter_device = torch.device("cpu")
    adapter.to(device=adapter_device, dtype=owner.dtype)
    adapter.load_artifact_state_dict(
        _adapter_state_for_runtime_dtype(export.state, dtype=owner.dtype)
    )
    adapter.requires_grad_(True)
    adapter.train(True)
    contract.assert_matches(adapter)
    if any(
        name.startswith(("main_projection.", "d_deepstack_projections."))
        for name, _ in adapter.named_parameters()
    ):
        raise RuntimeError("RP66 registered duplicate Qwen merger parameters")
    return adapter


def _load_validated_trainable_rp66_export(config: Any) -> Any:
    """Load and validate RP66 without requiring an initialized Qwen model."""

    export = load_rank_zero_adapter_owned_state_export(
        config.representation.artifact_path
    )
    if export.state is None:
        raise RuntimeError("RP66 artifact omitted Adapter-owned state")
    if state_digest(export.manifest) != config.representation.artifact.sha256:
        raise IdentityMismatchError("RP66 manifest identity differs from run config")
    run_identity = export.manifest.run_identity
    if run_identity.model != config.model:
        raise IdentityMismatchError("RP66 and actor Qwen identities differ")
    if run_identity.provider != config.representation.conditioning:
        raise IdentityMismatchError("RP66 conditioning identity differs")
    return export


def preflight_trainable_rp66_artifact(config: Any) -> None:
    """Fail before GPU allocation when the declared RP66 artifact is invalid."""

    _load_validated_trainable_rp66_export(config)


def _adapter_state_for_runtime_dtype(
    state: Mapping[str, torch.Tensor], *, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Cast floating artifact storage to the actor's master-parameter dtype.

    RP66's immutable Stage-1 artifact is stored in BF16, while the matched
    Crop-16 FSDP actor owns FP32 master parameters under BF16 mixed precision.
    Dtype is an execution property rather than artifact identity: the loader
    has already verified the source file and manifest hashes.  Keep the source
    tensors unchanged and preserve any non-floating state exactly.
    """

    if not isinstance(state, Mapping):
        raise TypeError("RP66 Adapter state must be a mapping")
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError("RP66 runtime parameter dtype must be floating point")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("RP66 Adapter state names must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"RP66 Adapter state {name!r} is not a tensor")
        normalized[name] = value.to(dtype=dtype) if value.is_floating_point() else value
    return normalized


def _required_run_config_path() -> Path:
    value = os.environ.get(TRAINABLE_TGVF_RUN_CONFIG_ENV)
    if not value:
        raise RuntimeError(f"{TRAINABLE_TGVF_RUN_CONFIG_ENV} is required")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("trainable RP66 run config path is invalid")
    return path


def _validate_worker(
    engine: Any,
    *,
    model: nn.Module,
    role: ComponentRole,
    bundle: TrajectoryReplayBundle,
) -> None:
    if getattr(engine.engine_config, "strategy", None) != "fsdp2":
        raise ValueError("trainable RP66 replay requires FSDP2")
    if getattr(engine.model_config, "model_type", None) != TRAINABLE_TGVF_MODEL_TYPE:
        raise IdentityMismatchError("trainable RP66 worker model_type differs")
    if getattr(engine.model_config, "path", None) != bundle.replay_record.model.revision_or_path:
        raise IdentityMismatchError("trainable RP66 worker model path differs")
    forward_only = bool(getattr(engine.engine_config, "forward_only", False))
    if forward_only is not (role is ComponentRole.REFERENCE):
        raise IdentityMismatchError("trainable RP66 worker role differs")
    attached = getattr(model, TRAINABLE_TGVF_ADAPTER_ATTRIBUTE, None)
    if role is ComponentRole.CURRENT and not isinstance(attached, TGVFAdapter):
        raise RuntimeError("trainable actor lost its RP66 Adapter")
    if role is ComponentRole.REFERENCE and attached is not None:
        raise RuntimeError("frozen reference must not own RP66")


def _validate_trainable_execution_capabilities(model_config: Any) -> None:
    """Validate optional kernels while the worker is building, before rollout."""

    if not bool(getattr(model_config, "use_fused_kernels", False)):
        return
    options = getattr(model_config, "fused_kernel_options", None)
    backend = options.get("impl_backend") if isinstance(options, Mapping) else None
    if backend != "torch":
        raise ValueError(
            "trainable exact replay currently implements the veRL torch fused backend"
        )
    # Import at worker startup so a missing/incompatible pinned veRL primitive
    # cannot surface only after rollout has already consumed GPU time.
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if not callable(getattr(FusedLinearForPPO, "forward", None)):
        raise RuntimeError("pinned veRL FusedLinearForPPO is unavailable")


__all__ = [
    "TRAINABLE_TGVF_MODEL_TYPE",
    "TRAINABLE_TGVF_RUN_CONFIG_ENV",
    "TrainableTGVFReplayPortFactory",
    "build_trainable_rp66_adapter",
    "make_trainable_tgvf_fsdp2_engine_class",
    "preflight_trainable_rp66_artifact",
    "register_trainable_tgvf_fsdp2_engine",
]
