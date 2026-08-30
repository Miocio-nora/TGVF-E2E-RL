from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.qwen.test_family_contract import _replay
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import ComponentRole, PolicyVersion
from tgvf_rl.contracts.tokens import (
    LogProbMeasurement,
    OwnedTokenSequence,
    SamplingIdentity,
    TokenOwnership,
)
from tgvf_rl.policy.exact_replay import (
    RecordedPolicyForwardBinding,
    ReplayParameterization,
)
from tgvf_rl.policy.qwen_replay import (
    Qwen3RecordedPolicyForwardPort,
    build_qwen3_decoder_lora_policy,
    freeze_qwen3_reference_model,
    replay_qwen3_current_reference,
)
from tgvf_rl.qwen.base import ReplayConsumer, resolve_replay_request


_BASE_SHA256 = "a" * 64
_LORA_SHA256 = "b" * 64
_FORWARD_SHA256 = "c" * 64


def _require_peft() -> None:
    pytest.importorskip(
        "peft",
        reason="Qwen3 decoder-LoRA integration requires optional PEFT",
    )


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        combined = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.o_proj(combined)


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.sigmoid(self.gate_proj(hidden)) * self.up_proj(hidden))


class _TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttention(hidden_size)
        self.mlp = _TinyMLP(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)


class _TinyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)
        self.layers = nn.ModuleList(
            _TinyDecoderLayer(hidden_size) for _ in range(36)
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
        **_: object,
    ) -> SimpleNamespace:
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        for branch in deepstack_visual_embeds:
            hidden = hidden.clone()
            hidden[visual_pos_masks] += branch
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _TinyVisual(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)])
        self.merger = nn.Linear(hidden_size, hidden_size, bias=False)
        self.deepstack_merger_list = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)
        )


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel(hidden_size)
        self.visual = _TinyVisual(hidden_size)


class _TinyQwen3(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.model = _TinyBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 32, bias=False)


def _binding(bundle, role: ComponentRole) -> RecordedPolicyForwardBinding:
    current = role is ComponentRole.CURRENT
    return RecordedPolicyForwardBinding(
        role=role,
        model=bundle.replay_record.model,
        policy_version=(
            bundle.replay_record.behavior_policy
            if current
            else PolicyVersion("reference", 0, _BASE_SHA256)
        ),
        parameterization=(
            ReplayParameterization.BASE_PLUS_LORA
            if current
            else ReplayParameterization.FROZEN_BASE
        ),
        base_weights_sha256=_BASE_SHA256,
        lora_state_sha256=_LORA_SHA256 if current else None,
        parameters_frozen=not current,
        deterministic_forward=True,
        lora_dropout=0.0,
        model_training=current,
        compute_dtype="float32",
        autocast_enabled=False,
        autocast_dtype=None,
        attention_backend="fixture_eager",
        forward_implementation_sha256=_FORWARD_SHA256,
    )


def _sampling(bundle) -> SamplingIdentity:
    return SamplingIdentity(
        policy_version=bundle.replay_record.behavior_policy,
        backend="vllm",
        backend_version="fixture",
        seed=7,
        rng_state_sha256="d" * 64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        logit_processors=(),
        measurement=LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
        asynchronous_staleness_steps=0,
    )


def _ports(bundle):
    base = _TinyQwen3()
    reference_model = copy.deepcopy(base)
    built = build_qwen3_decoder_lora_policy(base)
    reference_audit = freeze_qwen3_reference_model(reference_model)
    current = Qwen3RecordedPolicyForwardPort(
        model=built.model,
        binding=_binding(bundle, ComponentRole.CURRENT),
    )
    reference = Qwen3RecordedPolicyForwardPort(
        model=reference_model,
        binding=_binding(bundle, ComponentRole.REFERENCE),
    )
    return built, reference_audit, current, reference


def test_qwen3_lora_scope_and_exact_current_reference_replay() -> None:
    _require_peft()
    torch.manual_seed(19)
    store, handle = _replay(branches=3, calls=0)
    bundle = store.export_replay_bundle(handle)
    built, reference_audit, current, reference = _ports(bundle)
    response = OwnedTokenSequence(
        token_ids=(4, 5, 6, 7, 8),
        ownership=(
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.TOOL_OBSERVATION,
            TokenOwnership.POLICY_SAMPLED,
            TokenOwnership.TEMPLATE,
            TokenOwnership.POLICY_SAMPLED,
        ),
    )

    replay = replay_qwen3_current_reference(
        current=current,
        reference=reference,
        bundle=bundle,
        prompt_token_ids=(1, 2, 3),
        response=response,
        sampling=_sampling(bundle),
    )

    assert len(built.target_modules) == 36 * 7
    assert len(built.scope_audit.trainable_parameter_names) == 36 * 7 * 2
    assert all(
        ".lora_" in name for name in built.scope_audit.trainable_parameter_names
    )
    assert not reference_audit.trainable_parameter_names
    assert dict(built.scope_audit.frozen_category_parameter_counts) == {
        "vision_encoder": 1,
        "visual_merger": 1,
        "native_deepstack": 3,
        "input_embeddings": 1,
        "lm_head": 1,
    }
    assert replay.current.bundle_sha256 == bundle.bundle_sha256
    assert replay.reference.bundle_sha256 == bundle.bundle_sha256
    assert replay.current.logprobs.requires_grad
    assert not replay.reference.logprobs.requires_grad
    assert replay.current.policy_sampled_mask.tolist() == [True, False, True, False, True]
    assert torch.count_nonzero(replay.current.logprobs[~replay.current.policy_sampled_mask]) == 0

    replay.current.logprobs.sum().backward()
    current_parameters = dict(built.model.named_parameters())
    assert all(
        current_parameters[name].grad is not None
        for name in built.scope_audit.trainable_parameter_names
    )
    assert all(parameter.grad is None for parameter in reference.model.parameters())


def test_qwen3_lora_preserves_bfloat16_snapshot_dtype() -> None:
    _require_peft()
    get_peft_model_state_dict = pytest.importorskip(
        "peft.utils.save_and_load",
        reason="LoRA state export requires optional PEFT",
    ).get_peft_model_state_dict

    built = build_qwen3_decoder_lora_policy(
        _TinyQwen3().to(dtype=torch.bfloat16)
    )
    state = get_peft_model_state_dict(built.model, adapter_name="default")

    assert state
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16}


def test_qwen3_replay_rejects_wrong_role_tokens_and_mutated_bundle() -> None:
    _require_peft()
    store, handle = _replay(branches=3, calls=0)
    bundle = store.export_replay_bundle(handle)
    _, _, current, _ = _ports(bundle)
    reference_request = resolve_replay_request(
        store, handle, ReplayConsumer.REFERENCE
    )
    with pytest.raises(ReplayMismatchError, match="another role"):
        current.forward_recorded(reference_request)

    wrong_response = OwnedTokenSequence(
        token_ids=(4, 5, 6, 7, 9),
        ownership=(TokenOwnership.POLICY_SAMPLED,) * 5,
    )
    with pytest.raises(ReplayMismatchError, match="token IDs differ"):
        current.replay_response_logprobs(
            bundle=bundle,
            prompt_token_ids=(1, 2, 3),
            response=wrong_response,
            sampling=_sampling(bundle),
        )

    payload = next(
        item for item in bundle.tensor_payloads if item.tensor.dtype.is_floating_point
    )
    payload.tensor.add_(1.0)
    with pytest.raises(ReplayMismatchError, match="transport changed"):
        current.forward_replay_bundle(bundle)
