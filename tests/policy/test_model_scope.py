from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tgvf_rl.policy import (
    DecoderLoRAConfig,
    audit_policy_model_scope,
    audit_policy_pilot_model_scope,
    audit_reference_model_scope,
    expected_qwen3_decoder_lora_targets,
    resolve_qwen3_decoder_lora_targets,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.v_proj = nn.Linear(2, 2, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()
        self.input_layernorm = nn.LayerNorm(2)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList(_DecoderLayer() for _ in range(36))


class _Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.merger = nn.Linear(2, 2)
        self.deepstack_merger_list = nn.ModuleList(
            [nn.Linear(2, 2) for _ in range(3)]
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()
        self.visual = _Visual()


class _Qwen3Fixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(2, 8, bias=False)


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(base.in_features, rank, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, base.out_features, bias=False)}
        )


def _child(module: nn.Module, part: str) -> nn.Module:
    if part.isdigit():
        return module[int(part)]  # type: ignore[index]
    return getattr(module, part)


def _policy_fixture() -> _Qwen3Fixture:
    config = DecoderLoRAConfig()
    policy = _Qwen3Fixture()
    targets = resolve_qwen3_decoder_lora_targets(policy, config)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for target in targets:
        parts = target.split(".")
        parent: nn.Module = policy
        for part in parts[:-1]:
            parent = _child(parent, part)
        base = getattr(parent, parts[-1])
        setattr(parent, parts[-1], _LoRALinear(base, config.rank))
    policy.peft_config = {
        "default": SimpleNamespace(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules,
            exclude_modules=config.exclude_modules,
            bias="none",
        )
    }
    return policy


def _frozen_reference() -> _Qwen3Fixture:
    reference = _Qwen3Fixture()
    reference.requires_grad_(False)
    reference.eval()
    return reference


def _frozen_tgvf() -> nn.Module:
    adapter = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    adapter.requires_grad_(False)
    adapter.eval()
    return adapter


def test_positive_whitelist_resolves_every_decoder_projection_and_no_visuals() -> None:
    targets = resolve_qwen3_decoder_lora_targets(_Qwen3Fixture())
    assert targets == expected_qwen3_decoder_lora_targets()
    assert len(targets) == 252
    assert all(name.startswith("model.language_model.layers.") for name in targets)
    assert all("visual" not in name for name in targets)


def test_complete_startup_audit_proves_trainable_and_frozen_ownership() -> None:
    policy = _policy_fixture()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-5,
    )
    audit = audit_policy_pilot_model_scope(
        policy,
        _frozen_reference(),
        _frozen_tgvf(),
        optimizer,
    )

    assert len(audit.policy.decoder_target_modules) == 252
    assert len(audit.policy.trainable_parameter_names) == 504
    assert audit.policy.optimizer_parameter_names == tuple(
        sorted(audit.policy.trainable_parameter_names)
    )
    assert dict(audit.policy.frozen_category_parameter_counts).keys() == {
        "vision_encoder",
        "visual_merger",
        "native_deepstack",
        "input_embeddings",
        "lm_head",
    }
    assert not audit.reference.trainable_parameter_names
    assert audit.tgvf_parameter_names


def test_scope_audits_reject_vision_drift_wrong_dropout_and_reference_lora() -> None:
    policy = _policy_fixture()
    policy.model.visual.blocks[0].weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-LoRA base parameter"):
        audit_policy_model_scope(policy)

    policy = _policy_fixture()
    policy.peft_config["default"].lora_dropout = 0.1
    with pytest.raises(RuntimeError, match="built PEFT config"):
        audit_policy_model_scope(policy)

    policy = _policy_fixture()
    policy.model.language_model.layers[0].self_attn.q_proj.lora_A[
        "default"
    ] = nn.Linear(2, 8, bias=False)
    with pytest.raises(RuntimeError, match="rank-64"):
        audit_policy_model_scope(policy)

    reference = _frozen_reference()
    reference.peft_config = {"default": object()}
    with pytest.raises(RuntimeError, match="must not contain"):
        audit_reference_model_scope(reference)


def test_optimizer_must_own_exactly_the_decoder_lora_parameters() -> None:
    policy = _policy_fixture()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    base_parameter = policy.model.visual.blocks[0].weight
    optimizer = torch.optim.AdamW([*trainable, base_parameter], lr=1.0e-5)
    with pytest.raises(RuntimeError, match="optimizer ownership"):
        audit_policy_model_scope(policy, optimizer=optimizer)


def test_reference_and_tgvf_must_be_frozen_and_eval() -> None:
    reference = _frozen_reference()
    reference.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        audit_reference_model_scope(reference)

    policy = _policy_fixture()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-5,
    )
    tgvf = _frozen_tgvf()
    tgvf.train()
    with pytest.raises(RuntimeError, match="TGVF Adapter must be in eval"):
        audit_policy_pilot_model_scope(
            policy,
            _frozen_reference(),
            tgvf,
            optimizer,
        )
