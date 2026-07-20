"""Explicit dispatch for mixed native visual-tool trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tgvf_rl.environment.agent_loop import ToolExecutionContext
from tgvf_rl.observations.store import ObservationHandle
from tgvf_rl.protocol.schema import NativeToolCall, POLICY_RL_TOOL_NAMES


ToolExecutor = Callable[[NativeToolCall, ToolExecutionContext], ObservationHandle]


@dataclass(frozen=True, slots=True)
class ToolRuntimeBinding:
    name: str
    execute: ToolExecutor

    def __post_init__(self) -> None:
        if self.name not in POLICY_RL_TOOL_NAMES:
            raise ValueError(f"unknown native tool binding {self.name!r}")
        if not callable(self.execute):
            raise TypeError("tool binding execute must be callable")


class NativeToolRuntimeRegistry:
    """Dispatch both tools while preserving one shared agent-loop call order."""

    def __init__(self, bindings: tuple[ToolRuntimeBinding, ...]) -> None:
        bindings = tuple(bindings)
        names = tuple(binding.name for binding in bindings)
        if not names or len(set(names)) != len(names):
            raise ValueError("tool runtime bindings must be non-empty and unique")
        self._bindings = {binding.name: binding for binding in bindings}

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    def execute(
        self, parsed_call: NativeToolCall, context: ToolExecutionContext
    ) -> ObservationHandle:
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("tool runtime context must be ToolExecutionContext")
        try:
            binding = self._bindings[parsed_call.name]
        except KeyError as error:
            raise ValueError(
                f"no runtime binding for native tool {parsed_call.name!r}"
            ) from error
        handle = binding.execute(parsed_call, context)
        if not isinstance(handle, ObservationHandle):
            raise TypeError("native tool runtime must return an ObservationHandle")
        return handle
