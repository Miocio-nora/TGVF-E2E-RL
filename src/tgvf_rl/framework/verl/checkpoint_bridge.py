"""Project checkpoint contributors used alongside veRL's public FSDP2 I/O."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Protocol, runtime_checkable

from tgvf_rl.checkpoint import CheckpointCoordinator

from .compatibility import FSDP2BridgeConfig


@runtime_checkable
class StatefulTeacher(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class SDPOTeacherCheckpointContributor:
    """Make the independently stateful SDPO teacher a strict project section."""

    checkpoint_name = "sdpo_teacher_state"
    checkpoint_version = "sdpo-teacher-checkpoint-v1"

    def __init__(self, teacher: StatefulTeacher) -> None:
        if not isinstance(teacher, StatefulTeacher):
            raise TypeError("SDPO teacher must expose state_dict/load_state_dict")
        self.teacher = teacher

    def checkpoint_state(self) -> object:
        state = self.teacher.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("SDPO teacher state_dict must return a mapping")
        # A contributor owns a snapshot, never a live alias of teacher state.
        return deepcopy(dict(state))

    def restore_checkpoint_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("SDPO teacher checkpoint state must be a mapping")
        self.teacher.load_state_dict(deepcopy(dict(state)))

    # State-dict aliases make the contributor useful to other maintained
    # orchestration layers without changing CheckpointCoordinator's protocol.
    def state_dict(self) -> Mapping[str, object]:
        value = self.checkpoint_state()
        assert isinstance(value, Mapping)
        return value

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.restore_checkpoint_state(state)


def register_sdpo_teacher_checkpoint(
    coordinator: CheckpointCoordinator,
    teacher: StatefulTeacher,
) -> SDPOTeacherCheckpointContributor:
    """Register teacher state separately from actor/reference model shards."""

    if not isinstance(coordinator, CheckpointCoordinator):
        raise TypeError("coordinator must be CheckpointCoordinator")
    contributor = SDPOTeacherCheckpointContributor(teacher)
    coordinator.register(contributor)
    return contributor


def validate_fsdp2_checkpoint_config(config: FSDP2BridgeConfig) -> None:
    """Type-check hook used before delegating model shards to veRL."""

    if not isinstance(config, FSDP2BridgeConfig):
        raise TypeError("config must be a validated FSDP2BridgeConfig")
    # Construction performs all fail-closed semantic checks.  Access the
    # required fields here so an incompatible look-alike cannot pass by duck type.
    if config.checkpoint_async_save or not config.checkpoint_strict:
        raise ValueError("FSDP2 project checkpoints must be synchronous and strict")
