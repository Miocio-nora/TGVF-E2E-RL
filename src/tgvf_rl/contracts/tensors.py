"""Content-addressed tensor descriptors with no framework container types."""

from __future__ import annotations

from dataclasses import dataclass

from .identity import _validate_sha256


@dataclass(frozen=True, slots=True)
class ContentAddress:
    algorithm: str
    digest: str
    nbytes: int

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError("only sha256 content addresses are supported")
        _validate_sha256(self.digest)
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: str
    layout: str
    stride: tuple[int, ...]
    checksum: str
    alias_id: str | None = None

    def __post_init__(self) -> None:
        if any(size < 0 for size in self.shape):
            raise ValueError("tensor shape must be non-negative")
        if not self.dtype or not self.layout:
            raise ValueError("tensor dtype and layout must be explicit")
        if len(self.stride) != len(self.shape):
            raise ValueError("tensor stride rank must equal shape rank")
        _validate_sha256(self.checksum)


@dataclass(frozen=True, slots=True)
class TensorArtifactRef:
    name: str
    address: ContentAddress
    descriptor: TensorDescriptor

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor artifact name must be non-empty")
        if self.address.digest != self.descriptor.checksum:
            raise ValueError("address and tensor descriptor checksum differ")


@dataclass(frozen=True, slots=True)
class TensorPayloadSet:
    main_d: TensorArtifactRef
    deepstack: tuple[TensorArtifactRef, ...]
    position_ids: TensorArtifactRef
    attention_mask: TensorArtifactRef
    token_type_ids: TensorArtifactRef | None = None
