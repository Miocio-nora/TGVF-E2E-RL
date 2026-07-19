"""Fixed zero reward used only by the veRL actor-to-vLLM sync smoke."""

from __future__ import annotations

from typing import Any


DATA_SOURCE = "tgvf_verl_vllm_sync_gate"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Validate the fixture identity and return its exact non-RL zero reward."""

    del solution_str, extra_info
    if data_source != DATA_SOURCE:
        raise ValueError("unexpected data source for sync-gate reward")
    if float(ground_truth) != 0.0:
        raise ValueError("sync-gate ground truth must be exactly zero")
    return 0.0


__all__ = ["DATA_SOURCE", "compute_score"]
