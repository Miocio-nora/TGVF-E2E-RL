"""FSDP2 engine for matched full-Qwen plain-Crop exact replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.contracts.identity import ComponentRole
from tgvf_rl.observations.store import TrajectoryReplayBundle
from tgvf_rl.policy.trainable_crop_replay import TrainableCropCurrentReplayPort

from .exact_replay_engine import (
    _qwen3_forward_binding,
    _reshard_exact_replay_root,
    exact_replay_forward_step,
)
from .fused_exact_replay import FusedExactReplayMicrobatchMaterializer
from .trainable_tgvf_engine import _validate_trainable_execution_capabilities


TRAINABLE_CROP_MODEL_TYPE = "tgvf_trainable_crop_language_model"


@dataclass(frozen=True, slots=True)
class TrainableCropReplayPortFactory:
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
            FusedExactReplayMicrobatchMaterializer()
            if bool(getattr(engine.model_config, "use_fused_kernels", False))
            else None
        )
        if role is ComponentRole.CURRENT:
            return TrainableCropCurrentReplayPort(
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


_PORT_FACTORY = TrainableCropReplayPortFactory()
_ENGINE_CLASS: type[Any] | None = None


def make_trainable_crop_fsdp2_engine_class(
    upstream_engine_cls: type[Any],
) -> type[Any]:
    class TrainableCropFSDPEngineWithLMHead(upstream_engine_cls):
        def _build_module(self):
            if getattr(self.engine_config, "strategy", None) != "fsdp2":
                raise ValueError("trainable Crop engine supports FSDP2 only")
            if (
                getattr(self.model_config, "model_type", None)
                != TRAINABLE_CROP_MODEL_TYPE
            ):
                raise IdentityMismatchError("trainable Crop model_type differs")
            _validate_trainable_execution_capabilities(self.model_config)
            self.model_config.model_type = "language_model"
            try:
                module = super()._build_module()
            finally:
                self.model_config.model_type = TRAINABLE_CROP_MODEL_TYPE
            if getattr(module, "peft_config", None):
                raise ValueError("matched trainable Crop forbids policy LoRA")
            if bool(getattr(self.engine_config, "forward_only", False)):
                module.requires_grad_(False)
                module.eval()
            else:
                module.requires_grad_(True)
                module.train(True)
            return module

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

    TrainableCropFSDPEngineWithLMHead.__name__ = "TrainableCropFSDPEngineWithLMHead"
    TrainableCropFSDPEngineWithLMHead.__qualname__ = "TrainableCropFSDPEngineWithLMHead"
    TrainableCropFSDPEngineWithLMHead.__module__ = __name__
    return TrainableCropFSDPEngineWithLMHead


def register_trainable_crop_fsdp2_engine(
    *, registry: Any | None = None, upstream_engine_cls: type[Any] | None = None
) -> type[Any]:
    global _ENGINE_CLASS
    if _ENGINE_CLASS is not None and registry is None and upstream_engine_cls is None:
        return _ENGINE_CLASS
    if registry is None or upstream_engine_cls is None:
        from verl.workers.engine import EngineRegistry, FSDPEngineWithLMHead

        registry = registry or EngineRegistry
        upstream_engine_cls = upstream_engine_cls or FSDPEngineWithLMHead
    engine_cls = make_trainable_crop_fsdp2_engine_class(upstream_engine_cls)
    registered = registry.register(
        model_type=TRAINABLE_CROP_MODEL_TYPE,
        backend="fsdp2",
        device=["cuda", "npu"],
    )(engine_cls)
    if registry.__name__ == "EngineRegistry":
        _ENGINE_CLASS = registered
    return registered


def _validate_worker(
    engine: Any,
    *,
    model: nn.Module,
    role: ComponentRole,
    bundle: TrajectoryReplayBundle,
) -> None:
    if getattr(engine.engine_config, "strategy", None) != "fsdp2":
        raise ValueError("trainable Crop replay requires FSDP2")
    if getattr(engine.model_config, "model_type", None) != TRAINABLE_CROP_MODEL_TYPE:
        raise IdentityMismatchError("trainable Crop worker model_type differs")
    if (
        getattr(engine.model_config, "path", None)
        != bundle.replay_record.model.revision_or_path
    ):
        raise IdentityMismatchError("trainable Crop worker model path differs")
    forward_only = bool(getattr(engine.engine_config, "forward_only", False))
    if forward_only is not (role is ComponentRole.REFERENCE):
        raise IdentityMismatchError("trainable Crop worker role differs")
    trainable = any(parameter.requires_grad for parameter in model.parameters())
    if role is ComponentRole.CURRENT and not trainable:
        raise RuntimeError("trainable Crop actor lost its trainable state")
    if role is ComponentRole.REFERENCE and trainable:
        raise RuntimeError("frozen Crop reference owns trainable parameters")


__all__ = [
    "TRAINABLE_CROP_MODEL_TYPE",
    "TrainableCropReplayPortFactory",
    "make_trainable_crop_fsdp2_engine_class",
    "register_trainable_crop_fsdp2_engine",
]
