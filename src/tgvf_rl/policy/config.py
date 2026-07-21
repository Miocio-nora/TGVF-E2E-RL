"""Typed, fail-closed runtime envelope for Policy Pilot v1.

The dataclasses in this module contain the values accepted in
``PROJECT_TASK.md`` section 0.8.  Values that remain run identities do not
inherit vLLM or veRL defaults.  In particular, ``min_p`` starts unbound and
must be supplied explicitly before a live sampling request is constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

from tgvf_rl.contracts.errors import ContractUnsetError, IdentityMismatchError
from tgvf_rl.contracts.tokens import LogProbMeasurement, SamplingIdentity
from tgvf_rl.protocol import NativeToolCapabilityProfile


POLICY_PILOT_V1_CONFIG_SCHEMA = "policy-pilot-v1-20260720"
POLICY_PILOT_V1_MODEL_PATH = (
    "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
)
POLICY_PILOT_V1_MODEL_FAMILY = "qwen3_vl"
POLICY_PILOT_V1_MODEL_NAME = "Qwen3-VL-8B-Thinking"
POLICY_PILOT_V1_TOKENIZER_LENGTH = 151_669
POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256 = (
    "36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956"
)
POLICY_PILOT_V1_VLLM_VERSION = "0.12.0"
POLICY_PILOT_V1_POLICY_LOSS_NAME = "tgvf_policy_pilot_v1_grpo"
POLICY_PILOT_V1_VERL_EXECUTION_LOSS_MODE = "bypass_mode"
POLICY_PILOT_V1_VERL_ROLLOUT_LOSS_TYPE = "ppo_clip"
POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE = (
    "tgvf_rl.framework.verl.exact_bypass_loss"
)
POLICY_PILOT_V1_TOOL_PROFILE = NativeToolCapabilityProfile.TGVF_ONLY
POLICY_PILOT_V1_TOOL_NAMES = POLICY_PILOT_V1_TOOL_PROFILE.tool_names

QWEN3_DECODER_LAYER_COUNT = 36
QWEN3_DECODER_LORA_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN = (
    r"^model[.]language_model[.]layers[.][0-9]+[.]"
    r"(?:self_attn[.](?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp[.](?:gate_proj|up_proj|down_proj))$"
)


def _require_finite_real(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class PilotSamplingConfig:
    """One immutable probability-measure contract shared by every agent turn.

    ``trajectories_per_prompt`` is the GRPO group size.  It is deliberately not
    copied into :meth:`as_vllm_parameters`: the agent loop requests one
    continuation at a time after the initial group has been scheduled.
    ``max_response_length`` is the cumulative policy-token budget across that
    complete multi-turn trajectory, while ``max_tokens`` passed to the method
    is the remaining budget for the current turn.
    """

    trajectories_per_prompt: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_response_length: int = 8192
    asynchronous_staleness_steps: int = 0
    do_sample: bool = True
    stop_token_ids: tuple[int, ...] | None = None
    stop_strings: tuple[str, ...] | None = None
    include_stop_str_in_output: bool | None = None
    ignore_eos: bool | None = None
    backend: str = "vllm"
    backend_version: str = POLICY_PILOT_V1_VLLM_VERSION
    logit_processors: tuple[str, ...] = ()
    logprob_measurement: LogProbMeasurement = (
        LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS
    )

    def __post_init__(self) -> None:
        if self.stop_token_ids is not None:
            object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
            if any(
                type(token_id) is not int or token_id < 0
                for token_id in self.stop_token_ids
            ):
                raise ValueError("stop_token_ids must contain non-negative integers")
            if len(set(self.stop_token_ids)) != len(self.stop_token_ids):
                raise ValueError("stop_token_ids must be unique")
        if self.stop_strings is not None:
            object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
            if any(not isinstance(item, str) or not item for item in self.stop_strings):
                raise ValueError("stop_strings must contain non-empty strings")
        for name, value in (
            ("include_stop_str_in_output", self.include_stop_str_in_output),
            ("ignore_eos", self.ignore_eos),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool when bound")
        fixed_values = {
            "trajectories_per_prompt": (self.trajectories_per_prompt, 8),
            "temperature": (self.temperature, 1.0),
            "top_p": (self.top_p, 1.0),
            "top_k": (self.top_k, -1),
            "repetition_penalty": (self.repetition_penalty, 1.0),
            "presence_penalty": (self.presence_penalty, 0.0),
            "frequency_penalty": (self.frequency_penalty, 0.0),
            "max_response_length": (self.max_response_length, 8192),
            "asynchronous_staleness_steps": (
                self.asynchronous_staleness_steps,
                0,
            ),
            "do_sample": (self.do_sample, True),
            "backend": (self.backend, "vllm"),
            "backend_version": (
                self.backend_version,
                POLICY_PILOT_V1_VLLM_VERSION,
            ),
            "logit_processors": (tuple(self.logit_processors), ()),
            "logprob_measurement": (
                self.logprob_measurement,
                LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS,
            ),
        }
        for name, (actual, expected) in fixed_values.items():
            if actual != expected:
                raise ValueError(
                    f"Policy Pilot v1 requires {name}={expected!r}, got {actual!r}"
                )
        if self.min_p is not None:
            value = _require_finite_real(self.min_p, "min_p")
            if not 0.0 <= value <= 1.0:
                raise ValueError("min_p must lie in [0, 1]")

    @property
    def is_run_bound(self) -> bool:
        return self.min_p is not None and all(
            value is not None
            for value in (
                self.stop_token_ids,
                self.stop_strings,
                self.include_stop_str_in_output,
                self.ignore_eos,
            )
        )

    def bind_min_p(self, min_p: float) -> PilotSamplingConfig:
        """Return a new sampling identity with the explicit run input bound."""

        return replace(self, min_p=min_p)

    def bind_run_inputs(
        self,
        *,
        min_p: float,
        stop_token_ids: tuple[int, ...],
        stop_strings: tuple[str, ...],
        include_stop_str_in_output: bool,
        ignore_eos: bool,
    ) -> PilotSamplingConfig:
        """Bind all open probability/stopping inputs for one run identity."""

        return replace(
            self,
            min_p=min_p,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
        )

    def _require_run_bound(self) -> None:
        if not self.is_run_bound:
            raise ContractUnsetError(
                "Policy Pilot v1 min_p and all stop/EOS fields are explicit run "
                "inputs and remain unbound"
            )

    def remaining_response_tokens(self, consumed_policy_tokens: int) -> int:
        if type(consumed_policy_tokens) is not int or consumed_policy_tokens < 0:
            raise ValueError("consumed_policy_tokens must be a non-negative integer")
        remaining = self.max_response_length - consumed_policy_tokens
        if remaining <= 0:
            raise ValueError("the trajectory policy-token budget is exhausted")
        return remaining

    def as_vllm_parameters(self, *, max_tokens: int) -> Mapping[str, object]:
        """Build one single-continuation vLLM request from the remaining budget."""

        self._require_run_bound()
        if (
            type(max_tokens) is not int
            or max_tokens <= 0
            or max_tokens > self.max_response_length
        ):
            raise ValueError(
                "max_tokens must be a positive remaining trajectory budget no greater "
                "than max_response_length"
            )
        return MappingProxyType(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "stop_token_ids": list(self.stop_token_ids or ()),
                "stop": list(self.stop_strings or ()),
                "include_stop_str_in_output": self.include_stop_str_in_output,
                "ignore_eos": self.ignore_eos,
                "max_tokens": max_tokens,
                "logprobs": True,
            }
        )

    def validate_sampling_identity(
        self,
        identity: SamplingIdentity,
        *,
        expected_max_tokens: int,
    ) -> None:
        """Prove a recorded behavior trace used this exact probability measure."""

        if not isinstance(identity, SamplingIdentity):
            raise TypeError("identity must be a SamplingIdentity")
        self._require_run_bound()
        if (
            type(expected_max_tokens) is not int
            or expected_max_tokens <= 0
            or expected_max_tokens > self.max_response_length
        ):
            raise ValueError(
                "expected_max_tokens must be the positive remaining trajectory budget"
            )
        expected = {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": expected_max_tokens,
            "do_sample": self.do_sample,
            "stop_token_ids": self.stop_token_ids,
            "stop_strings": self.stop_strings,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "ignore_eos": self.ignore_eos,
            "logit_processors": self.logit_processors,
            "measurement": self.logprob_measurement,
            "asynchronous_staleness_steps": self.asynchronous_staleness_steps,
        }
        mismatches = {
            name: (getattr(identity, name), expected_value)
            for name, expected_value in expected.items()
            if getattr(identity, name) != expected_value
        }
        if mismatches:
            raise IdentityMismatchError(
                f"behavior SamplingIdentity differs from Policy Pilot v1: {mismatches!r}"
            )


@dataclass(frozen=True, slots=True)
class DecoderLoRAConfig:
    """Exact positive-whitelist LoRA scope accepted for Qwen3 Pilot v1."""

    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    initial_learning_rate: float = 1.0e-5
    target_modules: str = QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN
    exclude_modules: None = None
    expected_decoder_layers: int = QWEN3_DECODER_LAYER_COUNT
    reference_lora_enabled: bool = False

    def __post_init__(self) -> None:
        expected = {
            "rank": (self.rank, 64),
            "alpha": (self.alpha, 64),
            "dropout": (self.dropout, 0.0),
            "initial_learning_rate": (self.initial_learning_rate, 1.0e-5),
            "target_modules": (
                self.target_modules,
                QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN,
            ),
            "exclude_modules": (self.exclude_modules, None),
            "expected_decoder_layers": (
                self.expected_decoder_layers,
                QWEN3_DECODER_LAYER_COUNT,
            ),
            "reference_lora_enabled": (self.reference_lora_enabled, False),
        }
        for name, (actual, required) in expected.items():
            if actual != required:
                raise ValueError(
                    f"Policy Pilot v1 requires LoRA {name}={required!r}, got {actual!r}"
                )

    @property
    def expected_target_module_count(self) -> int:
        return self.expected_decoder_layers * len(QWEN3_DECODER_LORA_PROJECTIONS)


@dataclass(frozen=True, slots=True)
class PilotGRPOConfig:
    """Accepted optimizer-facing GRPO choices, separate from open batch details.

    ``policy_loss_name`` identifies the project-owned mathematical oracle.  It
    is deliberately distinct from ``verl_execution_loss_mode``: pinned veRL
    e003's public behavior-logprob bypass always dispatches its registered
    ``bypass_mode`` loss after assigning ``old_log_probs = rollout_log_probs``.
    The bridge therefore accepts that execution path only behind the pinned
    numerical/gradient parity gate.
    """

    total_training_epochs: int | None = None
    update_epochs: int = 1
    advantage_estimator: str = "grpo"
    sample_standard_deviation: bool = True
    group_std_epsilon: float = 1.0e-6
    zero_variance_advantage: float = 0.0
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    dual_clip: float = 3.0
    loss_aggregation: str = "token-mean"
    entropy_coefficient: float = 0.0
    kl_reward_coefficient: float = 0.0
    kl_loss_coefficient: float = 0.0
    maximum_gradient_norm: float = 1.0
    filter_groups: bool = False
    rollout_over_sample_rate: float = 0.0
    policy_loss_name: str = POLICY_PILOT_V1_POLICY_LOSS_NAME
    behavior_logprob_field: str = "rollout_log_probs"
    verl_execution_loss_mode: str = POLICY_PILOT_V1_VERL_EXECUTION_LOSS_MODE
    rollout_correction_bypass_mode: bool = True
    rollout_correction_loss_type: str = POLICY_PILOT_V1_VERL_ROLLOUT_LOSS_TYPE
    verl_external_loss_module: str = (
        POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE
    )
    rollout_importance_sampling: str | None = None
    rollout_rejection_sampling: str | None = None
    rollout_is_batch_normalize: bool = False

    def __post_init__(self) -> None:
        if self.total_training_epochs is not None and (
            type(self.total_training_epochs) is not int
            or self.total_training_epochs <= 0
        ):
            raise ValueError(
                "total_training_epochs is an unbound run input or a positive integer"
            )
        expected = {
            "update_epochs": (self.update_epochs, 1),
            "advantage_estimator": (self.advantage_estimator, "grpo"),
            "sample_standard_deviation": (
                self.sample_standard_deviation,
                True,
            ),
            "group_std_epsilon": (self.group_std_epsilon, 1.0e-6),
            "zero_variance_advantage": (self.zero_variance_advantage, 0.0),
            "clip_epsilon_low": (self.clip_epsilon_low, 0.2),
            "clip_epsilon_high": (self.clip_epsilon_high, 0.2),
            "dual_clip": (self.dual_clip, 3.0),
            "loss_aggregation": (self.loss_aggregation, "token-mean"),
            "entropy_coefficient": (self.entropy_coefficient, 0.0),
            "kl_reward_coefficient": (self.kl_reward_coefficient, 0.0),
            "kl_loss_coefficient": (self.kl_loss_coefficient, 0.0),
            "maximum_gradient_norm": (self.maximum_gradient_norm, 1.0),
            "filter_groups": (self.filter_groups, False),
            "rollout_over_sample_rate": (self.rollout_over_sample_rate, 0.0),
            "policy_loss_name": (
                self.policy_loss_name,
                POLICY_PILOT_V1_POLICY_LOSS_NAME,
            ),
            "behavior_logprob_field": (
                self.behavior_logprob_field,
                "rollout_log_probs",
            ),
            "verl_execution_loss_mode": (
                self.verl_execution_loss_mode,
                POLICY_PILOT_V1_VERL_EXECUTION_LOSS_MODE,
            ),
            "rollout_correction_bypass_mode": (
                self.rollout_correction_bypass_mode,
                True,
            ),
            "rollout_correction_loss_type": (
                self.rollout_correction_loss_type,
                POLICY_PILOT_V1_VERL_ROLLOUT_LOSS_TYPE,
            ),
            "verl_external_loss_module": (
                self.verl_external_loss_module,
                POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE,
            ),
            "rollout_importance_sampling": (
                self.rollout_importance_sampling,
                None,
            ),
            "rollout_rejection_sampling": (
                self.rollout_rejection_sampling,
                None,
            ),
            "rollout_is_batch_normalize": (
                self.rollout_is_batch_normalize,
                False,
            ),
        }
        for name, (actual, required) in expected.items():
            if actual != required:
                raise ValueError(
                    f"Policy Pilot v1 requires GRPO {name}={required!r}, got {actual!r}"
                )

    @property
    def ratio_bounds(self) -> tuple[float, float]:
        return (1.0 - self.clip_epsilon_low, 1.0 + self.clip_epsilon_high)


@dataclass(frozen=True, slots=True)
class PolicyPilotV1Config:
    """The fixed Pilot envelope plus explicitly unbound run-level sampling input."""

    schema_version: str = POLICY_PILOT_V1_CONFIG_SCHEMA
    model_family: str = POLICY_PILOT_V1_MODEL_FAMILY
    model_path: str = POLICY_PILOT_V1_MODEL_PATH
    native_deepstack_enabled: bool = True
    tool_profile: NativeToolCapabilityProfile = POLICY_PILOT_V1_TOOL_PROFILE
    enabled_tool_names: tuple[str, ...] = POLICY_PILOT_V1_TOOL_NAMES
    max_tgvf_call_attempts: int = 4
    image_max_pixels: int = 262144
    sampling: PilotSamplingConfig = field(default_factory=PilotSamplingConfig)
    lora: DecoderLoRAConfig = field(default_factory=DecoderLoRAConfig)
    grpo: PilotGRPOConfig = field(default_factory=PilotGRPOConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled_tool_names", tuple(self.enabled_tool_names))
        if not isinstance(self.tool_profile, NativeToolCapabilityProfile):
            raise TypeError("tool_profile must be NativeToolCapabilityProfile")
        expected = {
            "schema_version": (self.schema_version, POLICY_PILOT_V1_CONFIG_SCHEMA),
            "model_family": (self.model_family, POLICY_PILOT_V1_MODEL_FAMILY),
            "model_path": (self.model_path, POLICY_PILOT_V1_MODEL_PATH),
            "native_deepstack_enabled": (self.native_deepstack_enabled, True),
            "tool_profile": (
                self.tool_profile,
                POLICY_PILOT_V1_TOOL_PROFILE,
            ),
            "enabled_tool_names": (
                self.enabled_tool_names,
                POLICY_PILOT_V1_TOOL_NAMES,
            ),
            "max_tgvf_call_attempts": (self.max_tgvf_call_attempts, 4),
            "image_max_pixels": (self.image_max_pixels, 512 * 512),
        }
        for name, (actual, required) in expected.items():
            if actual != required:
                raise ValueError(
                    f"Policy Pilot v1 requires {name}={required!r}, got {actual!r}"
                )
        if not isinstance(self.sampling, PilotSamplingConfig):
            raise TypeError("sampling must be PilotSamplingConfig")
        if not isinstance(self.lora, DecoderLoRAConfig):
            raise TypeError("lora must be DecoderLoRAConfig")
        if not isinstance(self.grpo, PilotGRPOConfig):
            raise TypeError("grpo must be PilotGRPOConfig")

    @property
    def identity_sha256(self) -> str:
        def normalize(value: object) -> object:
            if isinstance(value, LogProbMeasurement):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in sorted(value.items())}
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            return value

        encoded = json.dumps(
            normalize(asdict(self)), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "POLICY_PILOT_V1_CONFIG_SCHEMA",
    "POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256",
    "POLICY_PILOT_V1_MODEL_FAMILY",
    "POLICY_PILOT_V1_MODEL_NAME",
    "POLICY_PILOT_V1_MODEL_PATH",
    "POLICY_PILOT_V1_POLICY_LOSS_NAME",
    "POLICY_PILOT_V1_TOOL_NAMES",
    "POLICY_PILOT_V1_TOOL_PROFILE",
    "POLICY_PILOT_V1_TOKENIZER_LENGTH",
    "POLICY_PILOT_V1_VERL_EXECUTION_LOSS_MODE",
    "POLICY_PILOT_V1_VERL_EXTERNAL_LOSS_MODULE",
    "POLICY_PILOT_V1_VERL_ROLLOUT_LOSS_TYPE",
    "POLICY_PILOT_V1_VLLM_VERSION",
    "QWEN3_DECODER_LAYER_COUNT",
    "QWEN3_DECODER_LORA_PROJECTIONS",
    "QWEN3_DECODER_LORA_TARGET_MODULE_PATTERN",
    "DecoderLoRAConfig",
    "PilotGRPOConfig",
    "PilotSamplingConfig",
    "PolicyPilotV1Config",
]
