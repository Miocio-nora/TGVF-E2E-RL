"""Positive-whitelist LoRA selection and startup freeze audits for Pilot v1."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import torch
from torch import nn

from .config import (
    QWEN3_DECODER_LORA_PROJECTIONS,
    QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN,
    DecoderLoRAConfig,
)


_QWEN3_DECODER_TARGET_RE = re.compile(QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN)
_QWEN3_DECODER_TARGET_IN_PARAMETER_RE = re.compile(
    r"(?:^|\.)(?P<target>model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)))(?:\.|$)"
)
_LORA_PARAMETER_TAIL_RE = re.compile(
    r"\.lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
)

_FROZEN_QWEN3_CATEGORY_FRAGMENTS = (
    ("native_deepstack", "model.visual.deepstack_merger_list."),
    ("visual_merger", "model.visual.merger."),
    ("vision_encoder", "model.visual."),
    ("input_embeddings", "model.language_model.embed_tokens."),
)


@dataclass(frozen=True, slots=True)
class ModelScopeAudit:
    """Immutable evidence from one policy or reference parameter walk."""

    role: str
    decoder_target_modules: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    optimizer_parameter_names: tuple[str, ...]
    frozen_category_parameter_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PolicyPilotScopeAudit:
    """Combined startup proof for policy, reference, and the TGVF Adapter."""

    policy: ModelScopeAudit
    reference: ModelScopeAudit
    tgvf_parameter_names: tuple[str, ...]


def expected_qwen3_decoder_lora_targets(
    config: DecoderLoRAConfig | None = None,
) -> tuple[str, ...]:
    """Return the complete 36-layer positive whitelist in stable order."""

    selected = config or DecoderLoRAConfig()
    if not isinstance(selected, DecoderLoRAConfig):
        raise TypeError("config must be DecoderLoRAConfig")
    return tuple(
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(selected.expected_decoder_layers)
        for projection in QWEN3_DECODER_LORA_PROJECTIONS
    )


def resolve_qwen3_decoder_lora_targets(
    model: nn.Module,
    config: DecoderLoRAConfig | None = None,
) -> tuple[str, ...]:
    """Resolve the pre-PEFT Qwen3 structure and reject any partial whitelist.

    This is intentionally a positive selection.  Vision linears, the visual
    merger, native DeepStack merger branches, embeddings, and ``lm_head`` can
    never match the anchored expression.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    selected = config or DecoderLoRAConfig()
    targets: list[str] = []
    for name, module in model.named_modules():
        if _QWEN3_DECODER_TARGET_RE.fullmatch(name) is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Qwen3 LoRA target {name!r} is not torch.nn.Linear")
        targets.append(name)
    expected = expected_qwen3_decoder_lora_targets(selected)
    if set(targets) != set(expected) or len(targets) != len(expected):
        missing = tuple(item for item in expected if item not in targets)
        extra = tuple(item for item in targets if item not in expected)
        raise RuntimeError(
            "Qwen3 decoder LoRA structure differs from the pinned 36-layer positive "
            f"whitelist: missing={missing!r} extra={extra!r}"
        )
    return expected


def _canonical_decoder_target(parameter_name: str) -> tuple[str, str] | None:
    match = _QWEN3_DECODER_TARGET_IN_PARAMETER_RE.search(parameter_name)
    if match is None:
        return None
    target = match.group("target")
    tail = parameter_name[match.end("target") :]
    lora_match = _LORA_PARAMETER_TAIL_RE.fullmatch(tail)
    if lora_match is None:
        return None
    return target, lora_match.group("side")


def _read_peft_config(model: nn.Module, expected: DecoderLoRAConfig) -> None:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or len(configs) != 1:
        raise RuntimeError("policy must expose exactly one built PEFT LoRA config")
    peft_config = next(iter(configs.values()))
    required = {
        "r": expected.rank,
        "lora_alpha": expected.alpha,
        "lora_dropout": expected.dropout,
        "target_modules": expected.target_modules,
        "exclude_modules": expected.exclude_modules,
        "bias": "none",
    }
    mismatches = {
        name: (getattr(peft_config, name, None), required_value)
        for name, required_value in required.items()
        if getattr(peft_config, name, None) != required_value
    }
    if mismatches:
        raise RuntimeError(
            f"built PEFT config differs from the Pilot LoRA contract: {mismatches!r}"
        )


def _frozen_category(parameter_name: str) -> str | None:
    for category, fragment in _FROZEN_QWEN3_CATEGORY_FRAGMENTS:
        if fragment in parameter_name:
            return category
    if parameter_name == "lm_head.weight" or ".lm_head." in parameter_name:
        return "lm_head"
    return None


def _audit_required_frozen_categories(
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
) -> tuple[tuple[str, int], ...]:
    counts = {
        "vision_encoder": 0,
        "visual_merger": 0,
        "native_deepstack": 0,
        "input_embeddings": 0,
        "lm_head": 0,
    }
    for name, parameter in named_parameters:
        category = _frozen_category(name)
        if category is None:
            continue
        if parameter.requires_grad:
            raise RuntimeError(f"required frozen {category} parameter is trainable: {name}")
        counts[category] += 1
    missing = tuple(name for name, count in counts.items() if count == 0)
    if missing:
        raise RuntimeError(
            f"Qwen3 frozen-component proof could not find categories: {missing!r}"
        )
    return tuple(counts.items())


def _optimizer_parameter_names(
    optimizer: torch.optim.Optimizer | None,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    trainable_names: tuple[str, ...],
) -> tuple[str, ...]:
    if optimizer is None:
        return ()
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    observed_ids: list[int] = []
    observed_names: list[str] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in observed_ids:
                raise RuntimeError("optimizer contains a duplicate parameter")
            if parameter_id not in names_by_id:
                raise RuntimeError("optimizer owns a parameter outside the policy model")
            observed_ids.append(parameter_id)
            observed_names.append(names_by_id[parameter_id])
    if set(observed_names) != set(trainable_names) or len(observed_names) != len(
        trainable_names
    ):
        missing = tuple(name for name in trainable_names if name not in observed_names)
        extra = tuple(name for name in observed_names if name not in trainable_names)
        raise RuntimeError(
            "optimizer ownership must equal the decoder-LoRA trainable set: "
            f"missing={missing!r} extra={extra!r}"
        )
    return tuple(sorted(observed_names))


def audit_policy_model_scope(
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    config: DecoderLoRAConfig | None = None,
) -> ModelScopeAudit:
    """Audit the post-PEFT policy before its first optimizer step."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    selected = config or DecoderLoRAConfig()
    _read_peft_config(model, selected)
    named_parameters = tuple(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("policy model has no parameters")

    expected_targets = expected_qwen3_decoder_lora_targets(selected)
    target_weights: dict[str, dict[str, nn.Parameter]] = {
        name: {} for name in expected_targets
    }
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in named_parameters:
        canonical = _canonical_decoder_target(name)
        is_lora_parameter = ".lora_" in name
        if canonical is None:
            if is_lora_parameter:
                raise RuntimeError(f"LoRA parameter is outside the decoder whitelist: {name}")
            if parameter.requires_grad:
                raise RuntimeError(f"non-LoRA base parameter is trainable: {name}")
            frozen_names.append(name)
            continue
        target, side = canonical
        if target not in target_weights:
            raise RuntimeError(f"LoRA parameter resolved to an unexpected target: {name}")
        if not parameter.requires_grad:
            raise RuntimeError(f"decoder LoRA parameter is frozen: {name}")
        if side in target_weights[target]:
            raise RuntimeError(
                f"decoder target exposes duplicate LoRA {side} weights: {target}"
            )
        target_weights[target][side] = parameter
        trainable_names.append(name)

    incomplete = tuple(
        target for target, sides in target_weights.items() if set(sides) != {"A", "B"}
    )
    if incomplete:
        raise RuntimeError(
            f"decoder targets do not each expose trainable LoRA A/B weights: {incomplete!r}"
        )
    for target, sides in target_weights.items():
        lora_a = sides["A"]
        lora_b = sides["B"]
        if (
            lora_a.ndim != 2
            or lora_b.ndim != 2
            or lora_a.shape[0] != selected.rank
            or lora_b.shape[1] != selected.rank
        ):
            raise RuntimeError(
                f"decoder target {target!r} does not materialize rank-{selected.rank} "
                "LoRA A/B matrices"
            )
    category_counts = _audit_required_frozen_categories(named_parameters)
    trainable = tuple(sorted(trainable_names))
    optimizer_names = _optimizer_parameter_names(
        optimizer, named_parameters, trainable
    )
    return ModelScopeAudit(
        role="policy",
        decoder_target_modules=expected_targets,
        trainable_parameter_names=trainable,
        frozen_parameter_names=tuple(sorted(frozen_names)),
        optimizer_parameter_names=optimizer_names,
        frozen_category_parameter_counts=category_counts,
    )


def audit_reference_model_scope(model: nn.Module) -> ModelScopeAudit:
    """Prove the reference is the frozen base model with no LoRA state."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if model.training:
        raise RuntimeError("frozen reference model must be in eval mode")
    peft_configs = getattr(model, "peft_config", None)
    if peft_configs:
        raise RuntimeError("frozen reference must not contain a PEFT/LoRA adapter")
    named_parameters = tuple(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("reference model has no parameters")
    for name, parameter in named_parameters:
        if ".lora_" in name:
            raise RuntimeError(f"frozen reference contains a LoRA parameter: {name}")
        if parameter.requires_grad:
            raise RuntimeError(f"frozen reference parameter is trainable: {name}")
    category_counts = _audit_required_frozen_categories(named_parameters)
    return ModelScopeAudit(
        role="reference",
        decoder_target_modules=(),
        trainable_parameter_names=(),
        frozen_parameter_names=tuple(sorted(name for name, _ in named_parameters)),
        optimizer_parameter_names=(),
        frozen_category_parameter_counts=category_counts,
    )


def _audit_tgvf_adapter_frozen(adapter: nn.Module) -> tuple[str, ...]:
    if not isinstance(adapter, nn.Module):
        raise TypeError("tgvf_adapter must be torch.nn.Module")
    if adapter.training:
        raise RuntimeError("frozen TGVF Adapter must be in eval mode")
    named_parameters = tuple(adapter.named_parameters())
    if not named_parameters:
        raise RuntimeError("TGVF Adapter has no auditable parameters")
    trainable = tuple(name for name, parameter in named_parameters if parameter.requires_grad)
    if trainable:
        raise RuntimeError(f"TGVF Adapter parameters are trainable: {trainable!r}")
    return tuple(sorted(name for name, _ in named_parameters))


def audit_policy_pilot_model_scope(
    policy: nn.Module,
    reference: nn.Module,
    tgvf_adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: DecoderLoRAConfig | None = None,
) -> PolicyPilotScopeAudit:
    """Run the complete launch-time ownership proof required by Pilot v1."""

    if optimizer is None:
        raise TypeError("Pilot startup audit requires the constructed optimizer")
    policy_audit = audit_policy_model_scope(
        policy, optimizer=optimizer, config=config
    )
    reference_audit = audit_reference_model_scope(reference)
    tgvf_names = _audit_tgvf_adapter_frozen(tgvf_adapter)
    return PolicyPilotScopeAudit(
        policy=policy_audit,
        reference=reference_audit,
        tgvf_parameter_names=tgvf_names,
    )


__all__ = [
    "ModelScopeAudit",
    "PolicyPilotScopeAudit",
    "audit_policy_model_scope",
    "audit_policy_pilot_model_scope",
    "audit_reference_model_scope",
    "expected_qwen3_decoder_lora_targets",
    "resolve_qwen3_decoder_lora_targets",
]
