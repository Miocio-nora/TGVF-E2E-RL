from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tgvf_rl.checkpoint.coordinator import CheckpointCoordinator
from tgvf_rl.checkpoint.schema import CheckpointBundle
from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.identity import CodeIdentity


@dataclass
class Contributor:
    checkpoint_name: str = "teacher"
    checkpoint_version: str = "teacher-v1"
    value: torch.Tensor = None

    def __post_init__(self):
        if self.value is None:
            self.value = torch.tensor([1.0])

    def checkpoint_state(self):
        return {"value": self.value.clone(), "step": 2}

    def restore_checkpoint_state(self, state):
        self.value = state["value"].clone()


def test_checkpoint_restores_contributor_and_rng(tmp_path) -> None:
    contributor = Contributor()
    coordinator = CheckpointCoordinator()
    coordinator.register(contributor)
    bundle = coordinator.collect(
        run_id="smoke",
        optimizer_step=2,
        code=CodeIdentity("repo", "abc"),
        rollout_policy_version="smoke:2",
    )
    path = tmp_path / "checkpoint.pt"
    coordinator.save_atomic(bundle, path)
    contributor.value.zero_()
    loaded = coordinator.load(path)
    result = coordinator.restore(loaded)
    assert result.exact
    assert result.next_optimizer_step == 3
    torch.testing.assert_close(contributor.value, torch.tensor([1.0]))


def test_checkpoint_rejects_missing_teacher_state() -> None:
    coordinator = CheckpointCoordinator()
    coordinator.register(Contributor())
    bundle = coordinator.collect(
        run_id="smoke",
        optimizer_step=2,
        code=CodeIdentity("repo", "abc"),
        rollout_policy_version="smoke:2",
    )
    tampered = CheckpointBundle(bundle.manifest, {"__rng__": bundle.state["__rng__"]})
    with pytest.raises(ReplayMismatchError, match="sections differ"):
        coordinator.validate_strict(tampered)
