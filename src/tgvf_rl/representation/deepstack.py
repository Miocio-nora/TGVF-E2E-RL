"""Typed D-DeepStack outputs and frozen Qwen projection ports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


D_DEEPSTACK_SCHEMA_VERSION = "tgvf-d-deepstack-v1"


def _validate_token_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    feature_dim: int | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape [N, D] or [B, N, D]")
    if tensor.shape[-2] == 0:
        raise ValueError(f"{name} must contain at least one token")
    if tensor.ndim == 3 and tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch item")
    if tensor.shape[-1] == 0:
        raise ValueError(f"{name} must have a non-empty feature dimension")
    if feature_dim is not None and tensor.shape[-1] != feature_dim:
        raise ValueError(
            f"{name} feature dimension is {tensor.shape[-1]}, expected {feature_dim}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


class FrozenProjectionPort(nn.Module):
    """Borrow a model merger through a shape-checked, permanently frozen port.

    The wrapped projection receives one unbatched ``[N, input_dim]`` item at a
    time.  This matches Qwen visual mergers while making batched behavior
    explicit instead of relying on an implementation-specific reshape.
    """

    def __init__(
        self,
        projection: nn.Module,
        *,
        identity: str,
        input_dim: int,
        output_dim: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        if not isinstance(projection, nn.Module):
            raise TypeError("projection must be an nn.Module")
        if not identity or not identity.strip():
            raise ValueError("projection identity must be non-empty")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")

        self.projection = projection
        self.identity = identity
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.spatial_merge_size = int(spatial_merge_size)
        self.projection.requires_grad_(False)
        self.train(False)

    @property
    def merge_group_size(self) -> int:
        return self.spatial_merge_size**2

    def train(self, mode: bool = True) -> FrozenProjectionPort:
        # A parent module's ``train()`` call must not reactivate merger dropout.
        super().train(False)
        return self

    def _project_one(self, tokens: torch.Tensor) -> torch.Tensor:
        parameter = next(self.projection.parameters(), None)
        buffer = next(self.projection.buffers(), None)
        owner = parameter if parameter is not None else buffer
        if owner is not None and tokens.device != owner.device:
            raise ValueError(
                "projection input and frozen projection must be on the same device"
            )
        if parameter is not None and parameter.dtype.is_floating_point:
            tokens = tokens.to(dtype=parameter.dtype)
        output = self.projection(tokens)
        if not isinstance(output, torch.Tensor):
            raise TypeError("frozen projection must return a torch.Tensor")
        return output

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_token_tensor(
            tokens, name="projection tokens", feature_dim=self.input_dim
        )
        token_count = int(tokens.shape[-2])
        if token_count % self.merge_group_size:
            raise ValueError(
                "projection token count must be divisible by spatial_merge_size**2: "
                f"{token_count} vs {self.merge_group_size}"
            )
        if any(parameter.requires_grad for parameter in self.projection.parameters()):
            raise RuntimeError("frozen projection parameters were made trainable")
        self.projection.eval()

        if tokens.ndim == 2:
            output = self._project_one(tokens)
        else:
            output = torch.stack(
                tuple(self._project_one(batch_tokens) for batch_tokens in tokens), dim=0
            )

        expected_shape = (
            *tokens.shape[:-2],
            token_count // self.merge_group_size,
            self.output_dim,
        )
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"projection output shape is {tuple(output.shape)}, expected {expected_shape}"
            )
        if not output.is_floating_point():
            raise TypeError("frozen projection output must use a floating-point dtype")
        return output


@dataclass(frozen=True, slots=True)
class DDeepStackPayload:
    """Ordered D branches ready for model-layer injection."""

    branch_layers: tuple[int, ...]
    branches: tuple[torch.Tensor, ...]
    projection_identities: tuple[str, ...]
    schema_version: str = D_DEEPSTACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != D_DEEPSTACK_SCHEMA_VERSION:
            raise ValueError(f"unsupported D-DeepStack schema {self.schema_version!r}")
        if not self.branch_layers:
            raise ValueError("D-DeepStack requires at least one branch")
        if len(self.branch_layers) != len(set(self.branch_layers)):
            raise ValueError("D-DeepStack branch layers must be unique")
        if tuple(sorted(self.branch_layers)) != self.branch_layers:
            raise ValueError("D-DeepStack branch layers must be strictly increasing")
        if any(layer < 0 for layer in self.branch_layers):
            raise ValueError("D-DeepStack branch layers must be non-negative")
        if not (
            len(self.branch_layers)
            == len(self.branches)
            == len(self.projection_identities)
        ):
            raise ValueError(
                "D-DeepStack layers, tensors, and projection identities must align"
            )
        if any(not identity for identity in self.projection_identities):
            raise ValueError("every D-DeepStack projection must have an identity")
        if len(set(self.projection_identities)) != len(self.projection_identities):
            raise ValueError("D-DeepStack projection identities must be unique")

        reference = self.branches[0]
        _validate_token_tensor(reference, name="D-DeepStack branch 0")
        for index, branch in enumerate(self.branches[1:], start=1):
            _validate_token_tensor(branch, name=f"D-DeepStack branch {index}")
            if branch.shape != reference.shape:
                raise ValueError("all D-DeepStack branches must have the same shape")
            if branch.device != reference.device or branch.dtype != reference.dtype:
                raise ValueError("all D-DeepStack branches must share device and dtype")

    @property
    def visual_embeds(self) -> tuple[torch.Tensor, ...]:
        return self.branches


class DDeepStackProjectionPorts(nn.Module):
    """Apply one explicit frozen projection to every required branch."""

    def __init__(
        self,
        *,
        branch_layers: Sequence[int],
        projections: Sequence[FrozenProjectionPort],
    ) -> None:
        super().__init__()
        layers = tuple(int(layer) for layer in branch_layers)
        ports = tuple(projections)
        if not layers:
            raise ValueError("D-DeepStack requires at least one branch layer")
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("branch_layers must be unique and strictly increasing")
        if any(layer < 0 for layer in layers):
            raise ValueError("branch_layers must be non-negative")
        if len(layers) != len(ports):
            raise ValueError("branch layers and frozen projections must align")
        if any(not isinstance(port, FrozenProjectionPort) for port in ports):
            raise TypeError(
                "every D-DeepStack projection must be a FrozenProjectionPort"
            )
        if len({id(port) for port in ports}) != len(ports):
            raise ValueError("D-DeepStack branches cannot share a projection port")
        identities = tuple(port.identity for port in ports)
        if len(set(identities)) != len(identities):
            raise ValueError("D-DeepStack projection identities must be unique")
        signature = {
            (port.input_dim, port.output_dim, port.spatial_merge_size) for port in ports
        }
        if len(signature) != 1:
            raise ValueError("all D-DeepStack projection ports must share dimensions")

        self.branch_layers = layers
        self.projections = nn.ModuleList(ports)

    def forward(self, branches: Sequence[torch.Tensor]) -> DDeepStackPayload:
        branch_tuple = tuple(branches)
        if len(branch_tuple) != len(self.branch_layers):
            raise ValueError(
                "D-DeepStack branch count mismatch: "
                f"features={len(branch_tuple)} layers={len(self.branch_layers)}"
            )
        projected = tuple(
            port(branch)
            for port, branch in zip(self.projections, branch_tuple, strict=True)
        )
        return DDeepStackPayload(
            branch_layers=self.branch_layers,
            branches=projected,
            projection_identities=tuple(port.identity for port in self.projections),
        )


def build_original_image_key_block_mask(
    *,
    attention_mask: torch.Tensor,
    original_image_token_indices: torch.Tensor,
    block_query_start: int,
    block_query_end: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a batched causal mask that hides original-image keys in a span."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, sequence_length]")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")
    batch_size, sequence_length = attention_mask.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError(
            "attention_mask batch and sequence dimensions must be non-empty"
        )
    start = int(block_query_start)
    end = sequence_length if block_query_end is None else int(block_query_end)
    if start < 0 or end < start or end > sequence_length:
        raise ValueError("blocked query span lies outside the sequence")
    if not isinstance(original_image_token_indices, torch.Tensor):
        raise TypeError("original_image_token_indices must be a torch.Tensor")
    if original_image_token_indices.ndim != 1:
        raise ValueError("original_image_token_indices must have shape [N]")
    if (
        original_image_token_indices.dtype == torch.bool
        or original_image_token_indices.is_floating_point()
    ):
        raise TypeError("original_image_token_indices must use an integer dtype")
    indices = original_image_token_indices.to(
        device=attention_mask.device, dtype=torch.long
    )
    if indices.numel():
        if int(indices.min()) < 0 or int(indices.max()) >= sequence_length:
            raise ValueError("original-image token index lies outside the sequence")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("original-image token indices must be unique")

    minimum = torch.finfo(dtype).min
    mask = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        device=attention_mask.device,
        dtype=dtype,
    )
    future = torch.ones(
        (sequence_length, sequence_length),
        device=attention_mask.device,
        dtype=torch.bool,
    ).triu(diagonal=1)
    mask.masked_fill_(future.view(1, 1, sequence_length, sequence_length), minimum)
    mask.masked_fill_(attention_mask[:, None, None, :] == 0, minimum)
    if indices.numel() and start < end:
        query_indices = torch.arange(start, end, device=attention_mask.device)
        mask[:, :, query_indices[:, None], indices[None, :]] = minimum
    return mask


__all__ = [
    "D_DEEPSTACK_SCHEMA_VERSION",
    "DDeepStackPayload",
    "DDeepStackProjectionPorts",
    "FrozenProjectionPort",
    "build_original_image_key_block_mask",
]
