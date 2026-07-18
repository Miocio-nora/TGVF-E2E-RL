"""Atomic, strict coordination of policy-adjacent project state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Protocol

import torch

from tgvf_rl.contracts.errors import IdentityMismatchError, ReplayMismatchError
from tgvf_rl.contracts.identity import CodeIdentity
from tgvf_rl.observations.store import tensor_checksum

from .schema import (
    CheckpointBundle,
    CheckpointSection,
    ProjectCheckpointManifest,
    ResumeValidationResult,
)


class CheckpointContributor(Protocol):
    checkpoint_name: str
    checkpoint_version: str

    def checkpoint_state(self) -> object: ...

    def restore_checkpoint_state(self, state: object) -> None: ...


class CheckpointCoordinator:
    """Coordinates custom state; veRL remains responsible for distributed I/O."""

    def __init__(self) -> None:
        self._contributors: dict[str, CheckpointContributor] = {}

    def register(self, contributor: CheckpointContributor) -> None:
        name = contributor.checkpoint_name
        if not name or not contributor.checkpoint_version:
            raise ValueError("checkpoint contributor name/version must be non-empty")
        if name in self._contributors:
            raise ValueError(f"duplicate checkpoint contributor {name!r}")
        self._contributors[name] = contributor

    def collect(
        self,
        *,
        run_id: str,
        optimizer_step: int,
        code: CodeIdentity,
        rollout_policy_version: str,
    ) -> CheckpointBundle:
        state: dict[str, object] = {}
        sections: list[CheckpointSection] = []
        for name in sorted(self._contributors):
            contributor = self._contributors[name]
            value = contributor.checkpoint_state()
            digest = state_digest(value)
            state[name] = value
            sections.append(
                CheckpointSection(name, contributor.checkpoint_version, digest)
            )
        rng = capture_rng_state()
        state["__rng__"] = rng
        sections.append(CheckpointSection("__rng__", "rng-v1", state_digest(rng)))
        sections.sort(key=lambda item: item.name)
        manifest = ProjectCheckpointManifest(
            schema_version="project-checkpoint-v1",
            run_id=run_id,
            optimizer_step=optimizer_step,
            code=code,
            sections=tuple(sections),
            rollout_policy_version=rollout_policy_version,
            sampling_backend="vllm",
        )
        return CheckpointBundle(manifest, state)

    def validate_strict(self, bundle: CheckpointBundle) -> ResumeValidationResult:
        expected_names = set(self._contributors) | {"__rng__"}
        actual_names = set(bundle.state)
        manifest_names = {section.name for section in bundle.manifest.sections}
        if actual_names != expected_names or manifest_names != expected_names:
            raise ReplayMismatchError(
                f"checkpoint sections differ: expected={sorted(expected_names)} actual={sorted(actual_names)}"
            )
        sections = {section.name: section for section in bundle.manifest.sections}
        for name, value in bundle.state.items():
            section = sections[name]
            if state_digest(value) != section.state_sha256:
                raise ReplayMismatchError(f"checkpoint section digest mismatch: {name}")
            if name != "__rng__":
                expected_version = self._contributors[name].checkpoint_version
                if section.version != expected_version:
                    raise IdentityMismatchError(
                        f"checkpoint version mismatch for {name}: {section.version} != {expected_version}"
                    )
        return ResumeValidationResult(
            exact=True,
            validated_sections=tuple(sorted(expected_names)),
            next_optimizer_step=bundle.manifest.optimizer_step + 1,
        )

    def restore(self, bundle: CheckpointBundle) -> ResumeValidationResult:
        result = self.validate_strict(bundle)
        for name in sorted(self._contributors):
            self._contributors[name].restore_checkpoint_state(bundle.state[name])
        restore_rng_state(bundle.state["__rng__"])
        return result

    @staticmethod
    def save_atomic(bundle: CheckpointBundle, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def load(path: str | Path) -> CheckpointBundle:
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(value, CheckpointBundle):
            raise ReplayMismatchError("file is not a project checkpoint bundle")
        return value


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = tuple(
            item.cpu().clone() for item in torch.cuda.get_rng_state_all()
        )
    return state


def restore_rng_state(state: object) -> None:
    if (
        not isinstance(state, Mapping)
        or "python" not in state
        or "torch_cpu" not in state
    ):
        raise ReplayMismatchError("malformed RNG checkpoint state")
    random.setstate(state["python"])
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        raise ReplayMismatchError("torch CPU RNG state is not a tensor")
    torch.random.set_rng_state(cpu_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ReplayMismatchError(
                "checkpoint has CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(list(cuda_state))


def state_digest(value: object) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def _update_digest(hasher: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        hasher.update(b"tensor:")
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(tensor_checksum(value).encode())
    elif hasattr(value, "__dataclass_fields__"):
        hasher.update(type(value).__qualname__.encode())
        _update_digest(hasher, asdict(value))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping:")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(hasher, str(key))
            _update_digest(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        hasher.update(b"sequence:")
        for item in value:
            _update_digest(hasher, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        hasher.update(json.dumps(value, sort_keys=True).encode())
    else:
        raise TypeError(
            f"unsupported checkpoint state type: {type(value).__qualname__}"
        )
