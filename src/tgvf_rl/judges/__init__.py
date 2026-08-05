"""Optional external judge boundary."""

from .base import (
    DisabledJudgeProvider,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
    JudgeUsage,
)
from .openai_compatible import (
    BoundOpenAICompatibleJudge,
    JUDGE_SAMPLE_FAILURE_ABORT,
    JUDGE_SAMPLE_FAILURE_ZERO,
    JudgeSampleFailureError,
    OpenAICompatibleJudgeConfig,
    OpenAICompatibleJudgeProvider,
    QWEN25_72B_RL_JUDGE_PROMPT_VERSION,
    QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT,
    load_openai_compatible_judge,
)

__all__ = [
    "DisabledJudgeProvider",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeResult",
    "JudgeUsage",
    "BoundOpenAICompatibleJudge",
    "JUDGE_SAMPLE_FAILURE_ABORT",
    "JUDGE_SAMPLE_FAILURE_ZERO",
    "JudgeSampleFailureError",
    "OpenAICompatibleJudgeConfig",
    "OpenAICompatibleJudgeProvider",
    "QWEN25_72B_RL_JUDGE_PROMPT_VERSION",
    "QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT",
    "load_openai_compatible_judge",
]
