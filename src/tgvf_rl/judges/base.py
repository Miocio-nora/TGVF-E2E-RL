"""Judge roles are isolated from RL reference and SDPO teacher roles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ArtifactIdentity


@dataclass(frozen=True, slots=True)
class JudgeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"judge {name} must be a non-negative integer")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("judge total_tokens differs from prompt plus completion")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0.0:
            raise ValueError("judge cost_usd must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    request_id: str
    task_kind: str
    question: str
    candidate_answer: str
    reference_answer: str | None
    prompt_identity: ArtifactIdentity

    def __post_init__(self) -> None:
        if not self.request_id or not self.question or not self.candidate_answer:
            raise ValueError("judge request identity/question/answer must be non-empty")
        if self.task_kind not in {"math", "open_vqa"}:
            raise ValueError("judge task_kind must be math or open_vqa")


@dataclass(frozen=True, slots=True)
class JudgeResult:
    score: float
    rationale: str
    service_identity: ArtifactIdentity
    model_identity: ArtifactIdentity
    sampling_identity: ArtifactIdentity
    calibration_identity: ArtifactIdentity
    usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("judge score must be finite and lie in [0,1]")
        if not self.rationale:
            raise ValueError("judge rationale must be non-empty")
        if self.usage is not None and not isinstance(self.usage, JudgeUsage):
            raise TypeError("judge usage must be JudgeUsage or None")


class JudgeProvider(Protocol):
    def judge(self, request: JudgeRequest) -> JudgeResult: ...


class DisabledJudgeProvider:
    """Reserved 72B judge provider that cannot be activated accidentally."""

    def judge(self, request: JudgeRequest) -> JudgeResult:
        raise ContractUnsetError(
            "the Qwen2.5-72B judge service/prompt/sampling/calibration contract is unset"
        )
