"""Exact, audited framework-stack identities.

Selectors are intentionally separate from package versions.  Callers choose a
named stack and receive the complete immutable identity; accepting independent
version strings would permit an unaudited cross-product of otherwise known
components.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


CONTROL_COMPATIBILITY_STACK = "control"
TORCH211_CU129_COMPATIBILITY_STACK = "torch211-cu129"


@dataclass(frozen=True, slots=True)
class AuditedCompatibilityStack:
    selector: str
    python_major_minor: tuple[int, int]
    torch_distribution_version: str
    torch_runtime_version: str
    transformers_distribution_version: str
    vllm_distribution_version: str
    vllm_archive_url: str | None
    vllm_archive_sha256: str | None
    verl_commit: str
    transfer_queue_distribution_version: str | None
    transfer_queue_archive_url: str | None
    transfer_queue_archive_sha256: str | None


_CONTROL = AuditedCompatibilityStack(
    selector=CONTROL_COMPATIBILITY_STACK,
    python_major_minor=(3, 12),
    torch_distribution_version="2.9.0",
    torch_runtime_version="2.9.0+cu128",
    transformers_distribution_version="4.57.6",
    vllm_distribution_version="0.12.0",
    vllm_archive_url=None,
    vllm_archive_sha256=None,
    verl_commit="e003163181731412595257a72ec173071efb125f",
    transfer_queue_distribution_version=None,
    transfer_queue_archive_url=None,
    transfer_queue_archive_sha256=None,
)
_TORCH211_CU129 = AuditedCompatibilityStack(
    selector=TORCH211_CU129_COMPATIBILITY_STACK,
    python_major_minor=(3, 12),
    torch_distribution_version="2.11.0+cu129",
    torch_runtime_version="2.11.0+cu129",
    transformers_distribution_version="4.57.6",
    vllm_distribution_version="0.23.0+cu129",
    vllm_archive_url=(
        "https://github.com/vllm-project/vllm/releases/download/v0.23.0/"
        "vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    ),
    vllm_archive_sha256=(
        "8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1"
    ),
    verl_commit="638b8ff84f279e054982f1f4633a546f3c6ced68",
    transfer_queue_distribution_version="0.1.8",
    transfer_queue_archive_url=(
        "https://files.pythonhosted.org/packages/84/7f/"
        "de3403bb53616cec07e0ec1dbdb6a88d5bcae121215bcdd4076f488cc4a7/"
        "transferqueue-0.1.8-py3-none-any.whl"
    ),
    transfer_queue_archive_sha256=(
        "078c4a63ba0c222fe684e96844c937dcd97f45ac94340a9c92eb03cfbc48cffd"
    ),
)

AUDITED_COMPATIBILITY_STACKS: Mapping[str, AuditedCompatibilityStack] = (
    MappingProxyType(
        {
            _CONTROL.selector: _CONTROL,
            _TORCH211_CU129.selector: _TORCH211_CU129,
        }
    )
)


def audited_compatibility_stack(selector: str) -> AuditedCompatibilityStack:
    """Resolve a named stack without accepting caller-supplied identities."""

    try:
        return AUDITED_COMPATIBILITY_STACKS[selector]
    except (KeyError, TypeError) as error:
        accepted = ", ".join(AUDITED_COMPATIBILITY_STACKS)
        raise ValueError(
            f"compatibility stack must be one of {accepted}; found {selector!r}"
        ) from error


def audited_stack_for_framework_pair(
    *, vllm_distribution_version: str, verl_commit: str
) -> AuditedCompatibilityStack:
    """Resolve an exact vLLM/veRL pair, rejecting every unaudited cross-product."""

    for stack in AUDITED_COMPATIBILITY_STACKS.values():
        if (
            stack.vllm_distribution_version == vllm_distribution_version
            and stack.verl_commit == verl_commit
        ):
            return stack
    raise ValueError(
        "vLLM/veRL pair is not an audited compatibility stack: "
        f"vllm={vllm_distribution_version!r} verl_commit={verl_commit!r}"
    )


__all__ = [
    "AUDITED_COMPATIBILITY_STACKS",
    "CONTROL_COMPATIBILITY_STACK",
    "TORCH211_CU129_COMPATIBILITY_STACK",
    "AuditedCompatibilityStack",
    "audited_compatibility_stack",
    "audited_stack_for_framework_pair",
]
