"""Integrity-bound Adapter artifact contract and loader for evaluation inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)
from tgvf_rl.representation.training.post_training_evaluation import file_sha256

from ..runner import (
    ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
    _answer_utility_state_digest,
)
from .input_matching import _require_sha256


_PUBLIC_RUNNER_MODULE = (
    "tgvf_rl.representation.experiments.answer_utility.evaluation.runner"
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "run_identity_sha256",
        "global_step",
        "source_artifact_sha256",
        "experiment_config_sha256",
        "adapter_state_sha256",
        "adapter_state",
    }
)


@dataclass(frozen=True, slots=True)
class AnswerUtilityAdapterArtifact:
    path: Path
    file_sha256: str
    run_identity_sha256: str
    global_step: int
    source_artifact_sha256: str
    experiment_config_sha256: str
    adapter_state_sha256: str
    adapter_state: Mapping[str, torch.Tensor]


def load_answer_utility_adapter_artifact(
    path: str | Path,
) -> AnswerUtilityAdapterArtifact:
    """Load exactly the private seven-key final-artifact schema."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"answer-utility artifact is missing: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError("answer-utility artifact must be a mapping")
    if set(value) != _ARTIFACT_FIELDS:
        raise ValueError("answer-utility artifact must have exactly seven fields")
    if value["schema_version"] != ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("answer-utility artifact schema mismatch")
    state = value["adapter_state"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        )
    ):
        raise TypeError("answer-utility artifact Adapter state is invalid")
    observed_state_sha256 = _answer_utility_state_digest(state)
    if value["adapter_state_sha256"] != observed_state_sha256:
        raise ValueError("answer-utility artifact state digest mismatch")
    for name in (
        "run_identity_sha256",
        "source_artifact_sha256",
        "experiment_config_sha256",
        "adapter_state_sha256",
    ):
        _require_sha256(value[name], name=f"artifact {name}")
    global_step = value["global_step"]
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("answer-utility artifact global_step must be an integer")
    if global_step <= 0:
        raise ValueError("answer-utility artifact global_step must be positive")
    return AnswerUtilityAdapterArtifact(
        path=source,
        file_sha256=file_sha256(source),
        run_identity_sha256=value["run_identity_sha256"],
        global_step=global_step,
        source_artifact_sha256=value["source_artifact_sha256"],
        experiment_config_sha256=value["experiment_config_sha256"],
        adapter_state_sha256=value["adapter_state_sha256"],
        adapter_state=dict(state),
    )


freeze_public_class_annotations(
    AnswerUtilityAdapterArtifact,
    implementation_globals=globals(),
)
rebind_public_class(
    AnswerUtilityAdapterArtifact,
    implementation_module=__name__,
    public_module=_PUBLIC_RUNNER_MODULE,
)
rebind_public_function(
    load_answer_utility_adapter_artifact,
    implementation_module=__name__,
    public_module=_PUBLIC_RUNNER_MODULE,
)


__all__ = [
    "AnswerUtilityAdapterArtifact",
    "load_answer_utility_adapter_artifact",
]
