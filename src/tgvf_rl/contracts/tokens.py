"""Token ownership and actual behavior-sampling identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math

from .identity import PolicyVersion, _validate_sha256


class TokenOwnership(str, Enum):
    TEMPLATE = "template"
    POLICY_SAMPLED = "policy_sampled"
    TOOL_OBSERVATION = "tool_observation"
    PADDING = "padding"

    @property
    def policy_loss_mask(self) -> int:
        """Only actual behavior-policy samples participate in policy loss."""

        return int(self is TokenOwnership.POLICY_SAMPLED)

    @property
    def requires_behavior_logprob(self) -> bool:
        """Padding and every environment/template token have no behavior logprob."""

        return self is TokenOwnership.POLICY_SAMPLED


class LogProbMeasurement(str, Enum):
    RAW_MODEL = "raw_model_logits"
    AFTER_SAMPLING_TRANSFORMS = "after_sampling_transforms"


@dataclass(frozen=True, slots=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token span must be non-empty and half-open")


@dataclass(frozen=True, slots=True)
class OwnedTokenSequence:
    token_ids: tuple[int, ...]
    ownership: tuple[TokenOwnership, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.ownership):
            raise ValueError("token_ids and ownership must have equal length")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be non-negative")

    @property
    def policy_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, owner in enumerate(self.ownership)
            if owner is TokenOwnership.POLICY_SAMPLED
        )


@dataclass(frozen=True, slots=True)
class SamplingIdentity:
    policy_version: PolicyVersion
    backend: str
    backend_version: str
    seed: int
    rng_state_sha256: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    logit_processors: tuple[str, ...]
    measurement: LogProbMeasurement
    asynchronous_staleness_steps: int
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int | None = None
    do_sample: bool | None = None
    stop_token_ids: tuple[int, ...] | None = None
    stop_strings: tuple[str, ...] | None = None
    include_stop_str_in_output: bool | None = None
    ignore_eos: bool | None = None

    def __post_init__(self) -> None:
        if not self.backend or not self.backend_version:
            raise ValueError("sampling backend identity must be explicit")
        if self.backend.lower() != "vllm":
            raise ValueError("vLLM is the only accepted rollout backend")
        numeric_values = (
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("sampling parameters must be finite")
        if type(self.seed) is not int:
            raise TypeError("sampling seed must be an integer")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature/top_p")
        if self.top_k < -1 or not 0 <= self.min_p <= 1:
            raise ValueError("invalid top_k/min_p")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.asynchronous_staleness_steps < 0:
            raise ValueError("staleness must be non-negative")
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer when recorded")
        if self.do_sample is not None and not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be bool when recorded")
        if self.stop_token_ids is not None:
            object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
            if any(
                type(token_id) is not int or token_id < 0
                for token_id in self.stop_token_ids
            ):
                raise ValueError("stop token IDs must be non-negative integers")
            if len(set(self.stop_token_ids)) != len(self.stop_token_ids):
                raise ValueError("stop token IDs must be unique")
        if self.stop_strings is not None:
            object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
            if any(not item for item in self.stop_strings):
                raise ValueError("stop strings must be non-empty")
        for name, value in (
            ("include_stop_str_in_output", self.include_stop_str_in_output),
            ("ignore_eos", self.ignore_eos),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool when recorded")
        if any(not item for item in self.logit_processors):
            raise ValueError("logit processor identities must be non-empty")
        _validate_sha256(self.rng_state_sha256)

    @property
    def transform_identity_sha256(self) -> str:
        """Identity of the probability measure, excluding policy/RNG identity."""

        payload = {
            "schema": "sampling-transform-v1",
            "backend": self.backend.lower(),
            "backend_version": self.backend_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_processors": self.logit_processors,
            "measurement": self.measurement.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def request_identity_sha256(self) -> str:
        """Identity of transforms plus stopping/length request semantics."""

        payload = {
            "schema": "sampling-request-v1",
            "transform_identity_sha256": self.transform_identity_sha256,
            "max_tokens": self.max_tokens,
            "do_sample": self.do_sample,
            "stop_token_ids": self.stop_token_ids,
            "stop_strings": self.stop_strings,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "ignore_eos": self.ignore_eos,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def has_identity_sampling_transforms(self) -> bool:
        return (
            self.temperature == 1.0
            and self.top_p == 1.0
            and self.top_k in {-1, 0}
            and self.min_p == 0.0
            and self.repetition_penalty == 1.0
            and self.presence_penalty == 0.0
            and self.frequency_penalty == 0.0
            and not self.logit_processors
        )


@dataclass(frozen=True, slots=True)
class BehaviorLogProbBlock:
    sampled_token_indices: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    sampling: SamplingIdentity

    def __post_init__(self) -> None:
        lengths = {
            len(self.sampled_token_indices),
            len(self.sampled_token_ids),
            len(self.logprobs),
        }
        if len(lengths) != 1:
            raise ValueError("behavior token indices, IDs, and logprobs must align")
        if tuple(sorted(self.sampled_token_indices)) != self.sampled_token_indices:
            raise ValueError("sampled token indices must be sorted")
        if len(set(self.sampled_token_indices)) != len(self.sampled_token_indices):
            raise ValueError("sampled token indices must be unique")
        if any(index < 0 for index in self.sampled_token_indices):
            raise ValueError("sampled token indices must be non-negative")
        if any(token_id < 0 for token_id in self.sampled_token_ids):
            raise ValueError("sampled token IDs must be non-negative")
        if any(not math.isfinite(value) or value > 1e-6 for value in self.logprobs):
            raise ValueError("behavior log probabilities must be finite and <= 0")
        if self.sampling.temperature < 1e-5 and any(
            abs(value) > 1e-6 for value in self.logprobs
        ):
            raise ValueError(
                "greedy behavior is a point mass; selected-token log probabilities must be zero"
            )
        if (
            self.sampling.measurement
            is not LogProbMeasurement.AFTER_SAMPLING_TRANSFORMS
        ):
            raise ValueError(
                "behavior log probabilities must be measured after all sampling transforms"
            )
