"""Leaf implementation for constructing the exact-replay veRL engine class.

The public factory remains in :mod:`exact_replay_engine`.  This module owns
only the dynamic upstream subclass and receives the replay callbacks from that
facade, keeping the dependency graph one-way and the public monkeypatch points
late-bound.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole


def build_exact_replay_fsdp2_engine_class(
    upstream_engine_cls: type[Any],
    *,
    port_factory: Callable[..., Any],
    model_type: str,
    forward_step: Callable[..., Any],
    engine_role: Callable[[Any], ComponentRole],
    reshard_root: Callable[[Any], None],
    snapshot_wrapper_provider: Callable[[], Callable[..., Any]],
    public_module: str,
) -> type[Any]:
    """Build the custom FSDP2 engine while preserving facade-owned hooks."""

    _validate_upstream_engine_surface(upstream_engine_cls)
    if not callable(port_factory):
        raise TypeError("exact replay port_factory must be callable")
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type == "language_model"
    ):
        raise ValueError("exact replay requires a distinct non-empty model_type")
    if not callable(forward_step):
        raise TypeError("exact replay forward_step must be callable")
    if not callable(engine_role):
        raise TypeError("exact replay engine_role must be callable")
    if not callable(reshard_root):
        raise TypeError("exact replay reshard_root must be callable")
    if not callable(snapshot_wrapper_provider):
        raise TypeError("exact replay snapshot_wrapper_provider must be callable")
    if not isinstance(public_module, str) or not public_module:
        raise ValueError("exact replay public_module must be a non-empty string")

    class ExactReplayFSDPEngineWithLMHead(upstream_engine_cls):
        def _build_module(self):
            engine_config = getattr(self, "engine_config", None)
            if getattr(engine_config, "strategy", None) != "fsdp2":
                raise ValueError("the exact replay engine supports FSDP2 only")
            model_config = getattr(self, "model_config", None)
            if getattr(model_config, "model_type", None) != model_type:
                raise IdentityMismatchError(
                    "TrainingWorker model_type differs from the exact replay registry key"
                )
            model_config.model_type = "language_model"
            try:
                module = super()._build_module()
            finally:
                model_config.model_type = model_type
            if getattr(engine_config, "forward_only", False):
                module.requires_grad_(False)
                module.eval()
            return module

        def _build_lora_module(self, module):
            if bool(getattr(self.engine_config, "forward_only", False)):
                module.requires_grad_(False)
                module.eval()
                return module
            return super()._build_lora_module(module)

        def get_per_tensor_param(
            self,
            layered_summon=False,
            base_sync_done=False,
            **kwargs,
        ):
            """Preserve veRL's stream while publishing the exact actor LoRA."""

            result = super().get_per_tensor_param(
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
                **kwargs,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "pinned FSDP2 get_per_tensor_param() must return "
                    "(parameter_stream, peft_config)"
                )
            parameter_stream, peft_config = result
            if not base_sync_done or engine_role(self) is ComponentRole.REFERENCE:
                return parameter_stream, peft_config
            if peft_config is None:
                raise ReplayMismatchError(
                    "current Policy Pilot adapter sync did not expose a LoRA-only "
                    "parameter stream"
                )
            snapshot_wrapper = snapshot_wrapper_provider()
            if not callable(snapshot_wrapper):
                raise TypeError("exact replay snapshot wrapper must be callable")
            return (
                snapshot_wrapper(
                    parameter_stream,
                    base_sync_done=True,
                ),
                peft_config,
            )

        def forward_step(self, micro_batch, loss_function, forward_only):
            return forward_step(
                engine=self,
                micro_batch=micro_batch,
                loss_function=loss_function,
                forward_only=forward_only,
                port_factory=port_factory,
            )

        def forward_backward_batch(self, *args, **kwargs):
            try:
                return super().forward_backward_batch(*args, **kwargs)
            finally:
                reshard_root(self.module)

    ExactReplayFSDPEngineWithLMHead.__name__ = "ExactReplayFSDPEngineWithLMHead"
    ExactReplayFSDPEngineWithLMHead.__qualname__ = "ExactReplayFSDPEngineWithLMHead"
    ExactReplayFSDPEngineWithLMHead.__module__ = public_module
    ExactReplayFSDPEngineWithLMHead.exact_replay_port_factory = port_factory
    ExactReplayFSDPEngineWithLMHead.exact_replay_model_type = model_type
    return ExactReplayFSDPEngineWithLMHead


def _validate_upstream_engine_surface(upstream_engine_cls: type[Any]) -> None:
    if not isinstance(upstream_engine_cls, type):
        raise TypeError("upstream FSDP engine must be a class")
    for name in ("_build_module", "forward_step", "get_per_tensor_param"):
        if not callable(getattr(upstream_engine_cls, name, None)):
            raise TypeError(f"upstream FSDP engine is missing {name}()")
    parameters = tuple(inspect.signature(upstream_engine_cls.forward_step).parameters)
    if parameters != ("self", "micro_batch", "loss_function", "forward_only"):
        raise RuntimeError("pinned FSDPEngineWithLMHead.forward_step signature changed")
    weight_parameters = tuple(
        inspect.signature(upstream_engine_cls.get_per_tensor_param).parameters
    )
    if weight_parameters != (
        "self",
        "layered_summon",
        "base_sync_done",
        "kwargs",
    ):
        raise RuntimeError(
            "pinned FSDPEngineWithLMHead.get_per_tensor_param signature changed"
        )


__all__ = ["build_exact_replay_fsdp2_engine_class"]
