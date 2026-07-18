"""Registry for accepted, mathematically identified pure objectives."""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable

from .base import ObjectiveResult
from .grpo import compute_grpo_loss
from .sdpo.objective import compute_sdpo_loss


ObjectiveFunction = Callable[..., ObjectiveResult]


class ObjectiveRegistry:
    """Small fixed registry; hybrid objectives require a separate accepted task."""

    def __init__(self) -> None:
        self._objectives = MappingProxyType(
            {
                "grpo": compute_grpo_loss,
                "sdpo": compute_sdpo_loss,
            }
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._objectives))

    def get(self, name: str) -> ObjectiveFunction:
        try:
            return self._objectives[name]
        except KeyError as error:
            raise KeyError(
                f"unknown objective {name!r}; accepted objectives are {', '.join(self.names)}"
            ) from error

    def compute(self, name: str, *, spec: object, **inputs: object) -> ObjectiveResult:
        if spec is None:
            raise TypeError("an explicit immutable objective spec is required")
        return self.get(name)(spec=spec, **inputs)


OBJECTIVE_REGISTRY = ObjectiveRegistry()


def get_objective(name: str) -> ObjectiveFunction:
    return OBJECTIVE_REGISTRY.get(name)


def compute_named_objective(
    name: str,
    *,
    spec: object,
    **inputs: object,
) -> ObjectiveResult:
    """Evaluate an accepted named objective with an explicit spec identity."""

    return OBJECTIVE_REGISTRY.compute(name, spec=spec, **inputs)
