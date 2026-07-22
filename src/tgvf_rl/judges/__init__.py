"""Optional external judge boundary."""

from .base import DisabledJudgeProvider, JudgeProvider, JudgeRequest, JudgeResult
from .openai_compatible import (
    OpenAICompatibleJudgeConfig,
    OpenAICompatibleJudgeProvider,
    QWEN25_72B_RL_JUDGE_PROMPT_VERSION,
    QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT,
)

__all__ = [
    "DisabledJudgeProvider",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeResult",
    "OpenAICompatibleJudgeConfig",
    "OpenAICompatibleJudgeProvider",
    "QWEN25_72B_RL_JUDGE_PROMPT_VERSION",
    "QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT",
]
