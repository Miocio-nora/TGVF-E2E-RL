"""Canonical configuration hash and promotion-gate validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import Enum

from tgvf_rl.conditioning.base import TargetConditioningConfig
from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.protocol.schema import POLICY_RL_TOOL_NAMES

from .schema import RunConfig, RunGate


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def config_sha256(config: RunConfig) -> str:
    raw = json.dumps(
        _canonical(asdict(config)), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_run_config(config: RunConfig) -> None:
    if not config.run_id:
        raise ValueError("run_id must be non-empty")
    if not isinstance(config.target_conditioning, TargetConditioningConfig):
        raise TypeError(
            "target_conditioning must be an explicit TargetConditioningConfig"
        )
    if config.gate is RunGate.SKELETON:
        return
    if (
        config.prompt_identity is None
        or config.max_tool_calls is None
        or config.enabled_tool_names is None
    ):
        raise ContractUnsetError(
            "rollout requires explicit prompt identity, enabled tools, and tool-call cap"
        )
    if config.max_tool_calls <= 1:
        raise ValueError("multi-call safety cap must be greater than one")
    tool_names = tuple(config.enabled_tool_names)
    if not tool_names or len(set(tool_names)) != len(tool_names):
        raise ValueError("enabled rollout tool names must be non-empty and unique")
    unknown_tools = set(tool_names) - set(POLICY_RL_TOOL_NAMES)
    if unknown_tools:
        raise ValueError(f"unknown enabled rollout tools: {sorted(unknown_tools)!r}")
    if config.gate in {RunGate.GRPO_SMOKE, RunGate.SDPO_SMOKE, RunGate.PRODUCTION}:
        if config.objective_identity is None:
            raise ContractUnsetError(
                "optimizer execution requires an objective identity"
            )
    if config.gate is RunGate.PRODUCTION:
        missing = [
            name
            for name, value in (
                ("representation_artifact", config.representation_artifact),
                ("data_manifest", config.data_manifest),
                ("reward_identity", config.reward_identity),
            )
            if value is None
        ]
        if missing:
            raise ContractUnsetError(
                f"production configuration is unset: {', '.join(missing)}"
            )
