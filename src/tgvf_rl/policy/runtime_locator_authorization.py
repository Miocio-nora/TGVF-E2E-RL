"""Deterministic Policy authorization copied from runtime-locator evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


POLICY_DRIVER_STARTUP_TARGET = "tgvf_rl.framework.verl.policy_main:main"
POLICY_RUNTIME_LOCATOR_AUTHORIZATION_KEYS = frozenset(
    {
        "runtime_locator_manifest_path",
        "runtime_locator_manifest_sha256",
        "runtime_locator_manifest_byte_length",
        "runtime_locator_manifest_identity_sha256",
        "runtime_locator_cache_tag",
        "runtime_locator_target_coordinates_json",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyRuntimeLocatorAuthorizationProof:
    """Exact serializable runtime-manifest authority for one Policy launch."""

    manifest_source_path: Path
    manifest_source_sha256: str
    manifest_source_byte_length: int
    manifest_identity_sha256: str
    cache_tag: str
    target_coordinates: tuple[str, ...]

    def __post_init__(self) -> None:
        path = Path(self.manifest_source_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime-locator proof manifest path must be absolute")
        object.__setattr__(self, "manifest_source_path", path)
        for field_name in (
            "manifest_source_sha256",
            "manifest_identity_sha256",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA256")
        if (
            type(self.manifest_source_byte_length) is not int
            or self.manifest_source_byte_length <= 0
        ):
            raise ValueError(
                "runtime-locator proof manifest byte length must be positive"
            )
        if type(self.cache_tag) is not str or not self.cache_tag:
            raise TypeError("runtime-locator proof cache tag must be exactly str")
        if self.target_coordinates != (POLICY_DRIVER_STARTUP_TARGET,):
            raise ValueError(
                "runtime-locator proof target coordinates differ from Policy"
            )

    def authorization_parameters(self) -> dict[str, str]:
        return {
            "runtime_locator_manifest_path": str(self.manifest_source_path),
            "runtime_locator_manifest_sha256": self.manifest_source_sha256,
            "runtime_locator_manifest_byte_length": str(
                self.manifest_source_byte_length
            ),
            "runtime_locator_manifest_identity_sha256": (
                self.manifest_identity_sha256
            ),
            "runtime_locator_cache_tag": self.cache_tag,
            "runtime_locator_target_coordinates_json": json.dumps(
                list(self.target_coordinates),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }


__all__ = [
    "POLICY_DRIVER_STARTUP_TARGET",
    "POLICY_RUNTIME_LOCATOR_AUTHORIZATION_KEYS",
    "PolicyRuntimeLocatorAuthorizationProof",
]
