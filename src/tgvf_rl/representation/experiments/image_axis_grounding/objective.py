"""Acyclic objective contract for the image-axis grounding experiment.

The streaming VJP validates this immutable contract while the optimizer-boundary
trainer consumes it.  Keeping the contract in a neutral leaf preserves a
one-way ``trainer -> streaming`` dependency instead of making either execution
layer own a type needed by the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import FunctionType


IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION = "image_axis_grounding_objective_v1"


@dataclass(frozen=True, slots=True)
class ImageAxisGroundingObjectiveConfig:
    """Frozen first-arm image-axis loss identity.

    The first experiment intentionally exposes no silent tuning surface.  A
    changed weight, temperature, or negative count is a new named experiment
    and must receive a new schema instead of mutating this baseline.
    """

    image_axis_matrix_weight: float = 1.0
    image_axis_temperature: float = 1.0
    negative_count: int = 1
    schema_version: str = IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError("image-axis objective schema mismatch")
        if (
            isinstance(self.image_axis_matrix_weight, bool)
            or not isinstance(self.image_axis_matrix_weight, float)
            or not math.isfinite(self.image_axis_matrix_weight)
            or self.image_axis_matrix_weight != 1.0
        ):
            raise ValueError("v1 freezes image_axis_matrix_weight at 1.0")
        if (
            isinstance(self.image_axis_temperature, bool)
            or not isinstance(self.image_axis_temperature, float)
            or not math.isfinite(self.image_axis_temperature)
            or self.image_axis_temperature != 1.0
        ):
            raise ValueError("v1 freezes image_axis_temperature at 1.0")
        if (
            isinstance(self.negative_count, bool)
            or not isinstance(self.negative_count, int)
            or self.negative_count != 1
        ):
            raise ValueError("v1 requires exactly one wrong-image negative")

    @property
    def loss_weights(self) -> tuple[float]:
        return (self.image_axis_matrix_weight,)

    def validation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "image_axis_matrix_weight": self.image_axis_matrix_weight,
            "image_axis_temperature": self.image_axis_temperature,
            "negative_count": self.negative_count,
        }


# This class historically lived in ``trainer``.  Preserve its fully-qualified
# identity so old pickles and introspection continue to resolve through the
# trainer facade while both execution layers import this exact leaf-owned type.
_LEGACY_PUBLIC_MODULE = (
    "tgvf_rl.representation.experiments.image_axis_grounding.trainer"
)
ImageAxisGroundingObjectiveConfig.__module__ = _LEGACY_PUBLIC_MODULE
for _member in ImageAxisGroundingObjectiveConfig.__dict__.values():
    if isinstance(_member, FunctionType) and _member.__module__ == __name__:
        _member.__module__ = _LEGACY_PUBLIC_MODULE
    elif isinstance(_member, property):
        for _accessor in (_member.fget, _member.fset, _member.fdel):
            if isinstance(_accessor, FunctionType) and _accessor.__module__ == __name__:
                _accessor.__module__ = _LEGACY_PUBLIC_MODULE


__all__ = [
    "IMAGE_AXIS_GROUNDING_OBJECTIVE_SCHEMA_VERSION",
    "ImageAxisGroundingObjectiveConfig",
]
