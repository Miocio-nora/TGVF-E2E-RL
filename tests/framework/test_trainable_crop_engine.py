from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from tgvf_rl.framework.verl.trainable_crop_engine import (
    TRAINABLE_CROP_MODEL_TYPE,
    make_trainable_crop_fsdp2_engine_class,
    register_trainable_crop_fsdp2_engine,
)


class _ToyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(3, 4)
        self.model.language_model = nn.Linear(4, 5)


class _UpstreamEngine:
    def _build_module(self):
        assert self.model_config.model_type == "language_model"
        return _ToyQwen()


def _engine(*, forward_only: bool):
    engine_type = make_trainable_crop_fsdp2_engine_class(_UpstreamEngine)
    engine = engine_type()
    engine.engine_config = SimpleNamespace(
        strategy="fsdp2",
        forward_only=forward_only,
    )
    engine.model_config = SimpleNamespace(
        model_type=TRAINABLE_CROP_MODEL_TYPE,
        use_fused_kernels=False,
    )
    return engine


def test_current_engine_builds_full_trainable_qwen_without_adapter() -> None:
    engine = _engine(forward_only=False)

    module = engine._build_module()

    assert module.training
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert not hasattr(module, "tgvf_adapter")
    assert engine.model_config.model_type == TRAINABLE_CROP_MODEL_TYPE


def test_reference_engine_builds_frozen_eval_qwen() -> None:
    engine = _engine(forward_only=True)

    module = engine._build_module()

    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert engine.model_config.model_type == TRAINABLE_CROP_MODEL_TYPE


def test_registration_uses_distinct_crop_model_type() -> None:
    seen: dict[str, object] = {}

    class _Registry:
        __name__ = "FixtureRegistry"

        @classmethod
        def register(cls, **kwargs):
            seen.update(kwargs)

            def decorator(engine_type):
                return engine_type

            return decorator

    registered = register_trainable_crop_fsdp2_engine(
        registry=_Registry,
        upstream_engine_cls=_UpstreamEngine,
    )

    assert issubclass(registered, _UpstreamEngine)
    assert seen == {
        "model_type": TRAINABLE_CROP_MODEL_TYPE,
        "backend": "fsdp2",
        "device": ["cuda", "npu"],
    }
