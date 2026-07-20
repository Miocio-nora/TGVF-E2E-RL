from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.policy.test_exact_replay import _payload
from tests.policy.test_qwen_replay import _TinyQwen3
from tgvf_rl.contracts.identity import ComponentRole
from tgvf_rl.contracts.tokens import TokenOwnership
from tgvf_rl.framework.verl.data_bridge import (
    DATAPROTO_META_SCHEMA_FIELD,
    DATAPROTO_META_SCHEMA_VERSION,
    release_verl_data_proto_sidecars,
    to_verl_data_proto,
    validate_data_proto_integrity,
)
from tgvf_rl.framework.verl.exact_replay_engine import (
    TGVF_EXACT_REPLAY_MODEL_TYPE,
    Qwen3ConfigBoundReplayPortFactory,
    register_exact_replay_fsdp2_engine,
)
from tgvf_rl.framework.verl.rollout_bridge import (
    EXACT_PROMPT_IDS_FIELD,
    EXACT_RESPONSE_IDS_FIELD,
    TRAJECTORY_REPLAY_BUNDLE_FIELD,
)
from tgvf_rl.policy.config import QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN
from tgvf_rl.policy.qwen_replay import (
    Qwen3RecordedPolicyForwardPort,
    build_qwen3_decoder_lora_policy,
    freeze_qwen3_reference_model,
)


class _NoRawForwardModel(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        raise AssertionError("exact replay must never call raw model.forward")


@dataclass(frozen=True)
class _FakeRoleResult:
    role: ComponentRole
    bundle_sha256: str
    response_token_ids: tuple[int, ...]
    response_ownership: tuple[TokenOwnership, ...]
    policy_sampled_mask: torch.Tensor
    logprobs: torch.Tensor


class _FakeResponsePort:
    def __init__(self, model: _NoRawForwardModel, role: ComponentRole) -> None:
        self.model = model
        self.binding = SimpleNamespace(role=role)
        self.calls: list[str] = []

    def replay_response_logprobs(
        self,
        *,
        bundle,
        prompt_token_ids,
        response,
        sampling,
    ) -> _FakeRoleResult:
        del prompt_token_ids, sampling
        self.calls.append(bundle.bundle_sha256)
        mask = torch.tensor(
            tuple(
                owner is TokenOwnership.POLICY_SAMPLED
                for owner in response.ownership
            ),
            dtype=torch.bool,
        )
        basis = -torch.arange(
            1,
            len(response.token_ids) + 1,
            dtype=self.model.weight.dtype,
        )
        values = self.model.weight * basis * mask.to(dtype=basis.dtype)
        return _FakeRoleResult(
            role=self.binding.role,
            bundle_sha256=bundle.bundle_sha256,
            response_token_ids=response.token_ids,
            response_ownership=response.ownership,
            policy_sampled_mask=mask,
            logprobs=values,
        )


class _FakeUpstreamFSDPEngineWithLMHead:
    def __init__(self, *, model_config, engine_config, module) -> None:
        self.model_config = model_config
        self.engine_config = engine_config
        self._module_to_build = module
        self.module = module
        self.exact_replay_evidence = None

    def _build_module(self):
        assert self.model_config.model_type == "language_model"
        return self._module_to_build

    def forward_step(self, micro_batch, loss_function, forward_only):
        raise AssertionError((micro_batch, loss_function, forward_only))

    def get_data_parallel_group(self):
        return None


class _FakeEngineRegistry:
    registrations: dict[tuple[str, str, tuple[str, ...]], type] = {}

    @classmethod
    def register(cls, *, model_type, backend, device):
        key = (model_type, backend, tuple(device))

        def decorate(engine_cls):
            assert key not in cls.registrations
            cls.registrations[key] = engine_cls
            return engine_cls

        return decorate


def _live_tensordict():
    protocol = pytest.importorskip("verl.protocol")
    payload, _ = _payload()
    data = to_verl_data_proto(payload, data_proto_cls=protocol.DataProto)
    validate_data_proto_integrity(data)
    return payload, data, data.to_tensordict()


def _exact_rows(tensor_dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt = tuple(list(tensor_dict[EXACT_PROMPT_IDS_FIELD])[0])
    response = tuple(list(tensor_dict[EXACT_RESPONSE_IDS_FIELD])[0])
    return prompt, response


def _concrete_engine_config(bundle, *, reference: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model_type=TGVF_EXACT_REPLAY_MODEL_TYPE,
            path=bundle.replay_record.model.revision_or_path,
            tokenizer=None,
            hf_config=SimpleNamespace(
                model_type="qwen3_vl",
                _attn_implementation="fixture_eager",
            ),
            lora_rank=64,
            lora_alpha=64,
            target_modules=QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN,
            exclude_modules=None,
            lora_adapter_path=None,
        ),
        engine_config=SimpleNamespace(
            strategy="fsdp2",
            full_determinism=True,
            forward_only=reference,
        ),
        _autocast_dtype=torch.float32,
    )


def test_live_dataproto_meta_schema_reaches_worker_tensordict() -> None:
    payload, data, tensor_dict = _live_tensordict()
    try:
        assert DATAPROTO_META_SCHEMA_FIELD != "tgvf_bridge_schema_version"
        assert data.meta_info[DATAPROTO_META_SCHEMA_FIELD] == (
            DATAPROTO_META_SCHEMA_VERSION
        )
        assert tensor_dict[DATAPROTO_META_SCHEMA_FIELD] == (
            DATAPROTO_META_SCHEMA_VERSION
        )
        assert list(tensor_dict[EXACT_PROMPT_IDS_FIELD])[0]
    finally:
        payload.release_sidecars()


def test_meta_schema_tamper_is_rejected_by_validation_and_release() -> None:
    protocol = pytest.importorskip("verl.protocol")
    payload, _ = _payload()
    data = to_verl_data_proto(payload, data_proto_cls=protocol.DataProto)
    data.meta_info[DATAPROTO_META_SCHEMA_FIELD] = "changed"
    with pytest.raises(RuntimeError, match="meta transport schema"):
        validate_data_proto_integrity(data)
    with pytest.raises(RuntimeError, match="meta transport schema"):
        release_verl_data_proto_sidecars(data)
    data.meta_info[DATAPROTO_META_SCHEMA_FIELD] = DATAPROTO_META_SCHEMA_VERSION
    payload.release_sidecars()


def test_external_lib_import_registers_concrete_engine_in_real_registry() -> None:
    pytest.importorskip("verl")
    from verl.utils.import_utils import import_external_libs
    from verl.workers.engine import EngineRegistry

    import_external_libs("tgvf_rl.framework.verl.exact_bypass_loss")
    engine_cls = EngineRegistry.get_engine_cls(
        TGVF_EXACT_REPLAY_MODEL_TYPE,
        "fsdp2",
    )
    assert engine_cls.exact_replay_model_type == TGVF_EXACT_REPLAY_MODEL_TYPE
    assert isinstance(
        engine_cls.exact_replay_port_factory,
        Qwen3ConfigBoundReplayPortFactory,
    )
    assert EngineRegistry._engines[TGVF_EXACT_REPLAY_MODEL_TYPE]["fsdp2"] == {
        "cuda": engine_cls,
        "npu": engine_cls,
    }


def test_config_bound_factory_builds_existing_qwen_port_for_actor_and_ref() -> None:
    payload, _ = _payload()
    bundle = payload.non_tensor_batch[TRAJECTORY_REPLAY_BUNDLE_FIELD][0]
    factory = Qwen3ConfigBoundReplayPortFactory()

    actor_model = build_qwen3_decoder_lora_policy(_TinyQwen3()).model
    actor_port = factory(
        engine=_concrete_engine_config(bundle, reference=False),
        model=actor_model,
        role=ComponentRole.CURRENT,
        bundle=bundle,
        model_training=True,
    )
    assert isinstance(actor_port, Qwen3RecordedPolicyForwardPort)
    assert actor_port.binding.model == bundle.replay_record.model
    assert actor_port.binding.policy_version == bundle.replay_record.behavior_policy
    assert (
        actor_port.binding.lora_state_sha256
        == bundle.replay_record.behavior_policy.weights_sha256
    )

    reference_model = _TinyQwen3()
    freeze_qwen3_reference_model(reference_model)
    reference_port = factory(
        engine=_concrete_engine_config(bundle, reference=True),
        model=reference_model,
        role=ComponentRole.REFERENCE,
        bundle=bundle,
        model_training=False,
    )
    assert isinstance(reference_port, Qwen3RecordedPolicyForwardPort)
    assert reference_port.binding.model == actor_port.binding.model
    assert reference_port.binding.base_weights_sha256 == (
        actor_port.binding.base_weights_sha256
    )
    assert reference_port.binding.lora_state_sha256 is None
    assert all(not parameter.requires_grad for parameter in reference_model.parameters())
    payload.release_sidecars()


def test_same_registered_engine_restores_actor_and_ref_full_sequence_layout() -> None:
    _FakeEngineRegistry.registrations.clear()
    payload, _, micro_batch = _live_tensordict()
    ports: list[_FakeResponsePort] = []

    def port_factory(*, engine, model, role, bundle, model_training):
        del engine, bundle, model_training
        port = _FakeResponsePort(model, role)
        ports.append(port)
        return port

    engine_cls = register_exact_replay_fsdp2_engine(
        port_factory=port_factory,
        registry=_FakeEngineRegistry,
        upstream_engine_cls=_FakeUpstreamFSDPEngineWithLMHead,
        devices=("cuda",),
    )
    assert _FakeEngineRegistry.registrations[
        (TGVF_EXACT_REPLAY_MODEL_TYPE, "fsdp2", ("cuda",))
    ] is engine_cls

    actor_model = _NoRawForwardModel(0.5)
    actor = engine_cls(
        model_config=SimpleNamespace(model_type=TGVF_EXACT_REPLAY_MODEL_TYPE),
        engine_config=SimpleNamespace(strategy="fsdp2", forward_only=False),
        module=actor_model,
    )
    actor.module = actor._build_module()
    assert actor.model_config.model_type == TGVF_EXACT_REPLAY_MODEL_TYPE
    captured: dict[str, torch.Tensor] = {}

    def actor_loss(*, model_output, data, dp_group):
        del data
        assert dp_group is None
        prompt, response = _exact_rows(micro_batch)
        full = model_output["log_probs"].unbind()[0]
        response_values = full[len(prompt) - 1 : -1]
        assert response_values.shape == (len(response),)
        mask = micro_batch["response_mask"][0, : len(response)].bool()
        captured["actor_response"] = response_values.detach().clone()
        return -response_values[mask].sum(), {"selected": int(mask.sum().item())}

    actor_loss_value, actor_output = actor.forward_step(
        micro_batch,
        actor_loss,
        False,
    )
    actor_loss_value.backward()
    assert actor_model.weight.grad is not None
    assert actor_output["metrics"] == {"selected": 7}
    assert not actor_output["model_output"]["log_probs"].requires_grad
    assert actor.exact_replay_evidence.role is ComponentRole.CURRENT

    reference_model = _NoRawForwardModel(0.5)
    reference = engine_cls(
        model_config=SimpleNamespace(model_type=TGVF_EXACT_REPLAY_MODEL_TYPE),
        engine_config=SimpleNamespace(strategy="fsdp2", forward_only=True),
        module=reference_model,
    )
    reference.module = reference._build_module()
    reference.module.eval()
    assert all(not parameter.requires_grad for parameter in reference.module.parameters())
    _, reference_output = reference.forward_step(micro_batch, None, True)
    reference_full = reference_output["model_output"]["log_probs"].unbind()[0]
    prompt, response = _exact_rows(micro_batch)
    reference_response = reference_full[len(prompt) - 1 : -1]

    torch.testing.assert_close(reference_response, captured["actor_response"])
    assert reference_full[: len(prompt) - 1].count_nonzero().item() == 0
    assert reference_full[-1].item() == 0.0
    assert not reference_full.requires_grad
    assert reference.exact_replay_evidence.role is ComponentRole.REFERENCE
    assert ports[0].binding.role is ComponentRole.CURRENT
    assert ports[1].binding.role is ComponentRole.REFERENCE
    assert ports[0].calls == ports[1].calls
    payload.release_sidecars()
