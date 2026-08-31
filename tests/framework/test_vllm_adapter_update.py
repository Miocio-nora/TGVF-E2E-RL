from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.contracts.errors import IdentityMismatchError
from tgvf_rl.framework.verl.vllm_tool_runtime import (
    TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
    TGVFVLLMWorkerExtension,
    _adapter_owned_state_from_utility_wire,
    _adapter_owned_state_to_utility_wire,
    _runtime_classes,
    adapter_owned_state_sha256,
    bind_tgvf_adapter_state_update_manager,
)


class _FakeAdapter:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = {name: tensor.clone() for name, tensor in state.items()}
        self.load_count = 0
        self.requires_grad_value = True
        self.training = True

    def artifact_state_dict(self, *, keep_vars: bool = False):
        del keep_vars
        return self.state

    def load_artifact_state_dict(self, state):
        self.load_count += 1
        self.state = {name: tensor.clone() for name, tensor in state.items()}

    def requires_grad_(self, value: bool):
        self.requires_grad_value = value
        return self

    def eval(self):
        self.training = False
        return self


def _ack(step: int, digest: str, count: int) -> dict[str, object]:
    return {
        "schema_version": TGVF_ADAPTER_UPDATE_ACK_SCHEMA,
        "optimizer_step": step,
        "state_sha256": digest,
        "tensor_count": count,
        "applied": True,
        "cleared_source_count": 0,
        "cleared_trace_count": 0,
    }


def test_adapter_state_wire_digest_and_worker_update_are_exact() -> None:
    initial = {
        "query.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
        "value.bias": torch.zeros((2,), dtype=torch.bfloat16),
    }
    updated = {
        "query.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "value.bias": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
    }
    assert adapter_owned_state_sha256(updated) == adapter_owned_state_sha256(
        dict(reversed(tuple(updated.items())))
    )
    restored = _adapter_owned_state_from_utility_wire(
        _adapter_owned_state_to_utility_wire(updated)
    )
    for name in updated:
        torch.testing.assert_close(restored[name], updated[name])

    adapter = _FakeAdapter(initial)
    extension = object.__new__(TGVFVLLMWorkerExtension)
    extension._tgvf_adapter_module = adapter
    extension._tgvf_source_cache = {"source": object()}
    extension._tgvf_behavior_traces = {"trace": object()}
    digest = adapter_owned_state_sha256(updated)

    ack = extension.tgvf_update_adapter_owned_state(
        7, digest, _adapter_owned_state_to_utility_wire(updated)
    )

    assert ack == {
        **_ack(7, digest, 2),
        "cleared_source_count": 1,
        "cleared_trace_count": 1,
    }
    assert adapter.load_count == 1
    assert not adapter.requires_grad_value
    assert not adapter.training
    assert extension._tgvf_sources() == {}
    assert extension._tgvf_traces() == {}

    retry = extension.tgvf_update_adapter_owned_state(
        7, digest, _adapter_owned_state_to_utility_wire(updated)
    )
    assert retry["applied"] is False
    assert adapter.load_count == 1

    conflicting = {**updated, "value.bias": updated["value.bias"] + 1}
    with pytest.raises(IdentityMismatchError, match="same Adapter optimizer step"):
        extension.tgvf_update_adapter_owned_state(
            7,
            adapter_owned_state_sha256(conflicting),
            _adapter_owned_state_to_utility_wire(conflicting),
        )


def test_http_and_manager_fanout_validate_every_adapter_ack() -> None:
    pytest.importorskip("verl", reason="runtime classes require pinned veRL")
    state = {"query.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    digest = adapter_owned_state_sha256(state)
    calls: list[str] = []

    class Engine:
        async def collective_rpc(self, *, method: str, kwargs: dict[str, object]):
            calls.append(method)
            restored = _adapter_owned_state_from_utility_wire(kwargs["state_wire"])
            torch.testing.assert_close(restored["query.weight"], state["query.weight"])
            return [_ack(4, digest, 1)]

    class RemoteMethod:
        def __init__(self, name: str) -> None:
            self.name = name

        async def remote(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["state"] is state
            calls.append(self.name)
            return _ack(4, digest, 1)

    async def exercise():
        manager_cls, _client_cls, _replica_cls, server_cls = _runtime_classes()
        server = object.__new__(server_cls)
        server.global_steps = 4
        server.engine = Engine()
        server_ack = await server.tgvf_update_adapter_owned_state(
            optimizer_step=4,
            state_sha256=digest,
            state=state,
        )
        replicas = [
            SimpleNamespace(
                servers=tuple(
                    SimpleNamespace(tgvf_update_adapter_owned_state=RemoteMethod(name))
                    for name in ("server-0", "server-1")
                )
            )
        ]
        manager = object.__new__(manager_cls)
        manager.rollout_replicas = replicas
        manager_acks = await manager.update_adapter_owned_state(
            optimizer_step=4,
            state_sha256=digest,
            state=state,
        )
        bound = bind_tgvf_adapter_state_update_manager(replicas)
        return server_ack, manager_acks, bound, replicas

    server_ack, manager_acks, bound, replicas = asyncio.run(exercise())
    assert server_ack == _ack(4, digest, 1)
    assert manager_acks == (_ack(4, digest, 1), _ack(4, digest, 1))
    assert calls == ["tgvf_update_adapter_owned_state", "server-0", "server-1"]
    assert bound.rollout_replicas == replicas
    assert bound.rollout_replicas is not replicas
