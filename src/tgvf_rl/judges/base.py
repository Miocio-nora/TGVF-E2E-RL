"""Judge roles are isolated from RL reference and SDPO teacher roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tgvf_rl.contracts.errors import ContractUnsetError
from tgvf_rl.contracts.identity import ArtifactIdentity


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    request_id: str
    question: str
    candidate_answer: str
    reference_answer: str | None
    prompt_identity: ArtifactIdentity


@dataclass(frozen=True, slots=True)
class JudgeResult:
    score: float
    rationale: str
    service_identity: ArtifactIdentity
    model_identity: ArtifactIdentity
    sampling_identity: ArtifactIdentity
    calibration_identity: ArtifactIdentity


class JudgeProvider(Protocol):
    def judge(self, request: JudgeRequest) -> JudgeResult: ...


class DisabledJudgeProvider:
    """Reserved 72B judge provider that cannot be activated accidentally."""

    def judge(self, request: JudgeRequest) -> JudgeResult:
        raise ContractUnsetError(
            "the Qwen2.5-72B judge service/prompt/sampling/calibration contract is unset"
        )
