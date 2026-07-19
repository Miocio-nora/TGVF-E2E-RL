"""Fail-closed Qwen3 runtime boundary for representation-phase training.

This module binds an explicitly supplied Qwen3 model and processor to one
``ModelIdentity``.  It borrows (without making trainable) the model-owned main
and DeepStack mergers, captures the exact pre-merger vision tensors, and hands
typed target conditioning plus vision state to the TGVF Adapter.

It intentionally does not provide a Qwen2.5-VL fallback.  That family needs a
separately accepted representation artifact and DeepStack contract.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningDependencies,
    TargetConditioningOutput,
    TargetConditioningProviderKind,
    TargetConditioningRequest,
    create_target_condition_provider,
)
from tgvf_rl.conditioning.base import BoundTargetConditionProvider
from tgvf_rl.contracts.identity import ModelIdentity
from tgvf_rl.protocol.native import NativeProtocolRenderer
from tgvf_rl.qwen.base import resolve_language_model
from tgvf_rl.representation.adapter import TGVFAdapter, TGVFAdapterInput
from tgvf_rl.representation.deepstack import (
    DDeepStackProjectionPorts,
    FrozenProjectionPort,
)


ACCEPTED_QWEN3_MODEL_PATH = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
ACCEPTED_QWEN3_TOKENIZER_LENGTH = 151669
ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256 = (
    "36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956"
)
QWEN3_REPRESENTATION_BRANCH_LAYERS = (8, 16, 24)
QWEN3_REPRESENTATION_SPATIAL_MERGE_SIZE = 2
QWEN3_REPRESENTATION_LANGUAGE_DIM = 4096
QWEN3_REPRESENTATION_VISION_DIM = 1152
QWEN3_PATCH_EMBED_LINEAR_FAST_PATH = "qwen3_patch_embed_reshape_linear_v1"
_QWEN3_PATCH_EMBED_KERNEL = (2, 16, 16)
_QWEN3_PATCH_EMBED_INPUT_WIDTH = 3 * 2 * 16 * 16


@dataclass(frozen=True, slots=True)
class Qwen3RepresentationComponentPaths:
    """Accepted Hugging Face Qwen3-VL component ownership paths."""

    vision_tower: str = "model.visual"
    main_merger: str = "model.visual.merger"
    deepstack_mergers: tuple[str, ...] = (
        "model.visual.deepstack_merger_list.0",
        "model.visual.deepstack_merger_list.1",
        "model.visual.deepstack_merger_list.2",
    )

    def __post_init__(self) -> None:
        paths = (self.vision_tower, self.main_merger, *self.deepstack_mergers)
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError("every Qwen3 component path must be non-empty")
        if len(self.deepstack_mergers) != 3:
            raise ValueError("Qwen3 representation runtime requires three mergers")
        if len(set(paths)) != len(paths):
            raise ValueError("Qwen3 component paths must be unique")


@dataclass(frozen=True, slots=True)
class Qwen3RepresentationArchitecture:
    """Dimensions read from, and subsequently pinned to, the bound model."""

    language_hidden_size: int
    vision_hidden_size: int
    spatial_merge_size: int
    branch_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.language_hidden_size <= 0 or self.vision_hidden_size <= 0:
            raise ValueError("Qwen3 representation dimensions must be positive")
        if self.spatial_merge_size <= 0:
            raise ValueError("Qwen3 spatial merge size must be positive")
        if self.branch_layers != QWEN3_REPRESENTATION_BRANCH_LAYERS:
            raise ValueError(
                "Qwen3 representation branches must be exactly (8, 16, 24)"
            )


@dataclass(frozen=True, slots=True)
class Qwen3ContextualHiddenStateStack:
    """Exact hidden-state tuple returned by one frozen-Qwen forward."""

    layers: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("contextual hidden-state stack cannot be empty")
        reference = self.layers[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("contextual hidden states must be tensors")
        if reference.ndim not in (2, 3) or not reference.is_floating_point():
            raise ValueError(
                "contextual hidden states must be floating [S,H] or [B,S,H] tensors"
            )
        for layer in self.layers[1:]:
            if not isinstance(layer, torch.Tensor):
                raise TypeError("contextual hidden states must be tensors")
            if layer.shape != reference.shape:
                raise ValueError("all contextual hidden-state layers must share shape")
            if layer.device != reference.device or layer.dtype != reference.dtype:
                raise ValueError(
                    "all contextual hidden-state layers must share device and dtype"
                )
        if any(layer.requires_grad for layer in self.layers):
            raise ValueError(
                "contextual Hq source must come from a deterministic frozen-Qwen forward"
            )

    def select(self, hidden_layer: int) -> torch.Tensor:
        if not isinstance(hidden_layer, int) or isinstance(hidden_layer, bool):
            raise TypeError("hidden_layer must be an integer")
        resolved = (
            hidden_layer if hidden_layer >= 0 else len(self.layers) + hidden_layer
        )
        if resolved < 0 or resolved >= len(self.layers):
            raise ValueError("configured contextual hidden layer is unavailable")
        return self.layers[resolved]


@dataclass(frozen=True, slots=True)
class Qwen3VisionPreMergeRequest:
    """One processor-materialized original image for the frozen vision tower."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.pixel_values, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor")
        if self.pixel_values.ndim != 2 or not self.pixel_values.is_floating_point():
            raise ValueError("pixel_values must be a floating [N,patch_dim] tensor")
        if not isinstance(self.image_grid_thw, torch.Tensor):
            raise TypeError("image_grid_thw must be a torch.Tensor")
        if self.image_grid_thw.shape != (1, 3):
            raise ValueError(
                "representation vision extraction currently requires exactly one image"
            )
        if self.image_grid_thw.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("image_grid_thw must use an integer dtype")
        if bool((self.image_grid_thw <= 0).any().item()):
            raise ValueError("image_grid_thw entries must be positive")
        expected_tokens = int(self.image_grid_thw.to(torch.int64).prod().item())
        if int(self.pixel_values.shape[0]) != expected_tokens:
            raise ValueError(
                "pixel_values row count must equal the image grid token count"
            )


@dataclass(frozen=True, slots=True)
class Qwen3VisionFeatures:
    """Frozen-Qwen source image tensors on both sides of every merger."""

    model_identity: ModelIdentity
    image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    branch_layers: tuple[int, ...]
    projection_identities: tuple[str, ...]
    pre_merge_main: torch.Tensor
    pre_merge_deepstack: tuple[torch.Tensor, ...]
    merged_main: torch.Tensor
    merged_deepstack: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_identity, ModelIdentity):
            raise TypeError("vision features require a ModelIdentity")
        if self.branch_layers != QWEN3_REPRESENTATION_BRANCH_LAYERS:
            raise ValueError("vision feature branch order must be (8, 16, 24)")
        if self.spatial_merge_size != QWEN3_REPRESENTATION_SPATIAL_MERGE_SIZE:
            raise ValueError("vision features require Qwen3 spatial merge size 2")
        if len(self.projection_identities) != 4:
            raise ValueError("vision features require main plus three projection IDs")
        if len(set(self.projection_identities)) != 4:
            raise ValueError("vision projection identities must be unique")
        if len(self.pre_merge_deepstack) != 3 or len(self.merged_deepstack) != 3:
            raise ValueError("vision features require all three DeepStack branches")
        if len(self.image_grid_thw) != 3 or any(x <= 0 for x in self.image_grid_thw):
            raise ValueError("image_grid_thw must contain three positive values")
        if any(value % self.spatial_merge_size for value in self.image_grid_thw[1:]):
            raise ValueError("image grid height/width must be spatial-merge divisible")

        pre = (self.pre_merge_main, *self.pre_merge_deepstack)
        merged = (self.merged_main, *self.merged_deepstack)
        _validate_feature_family(pre, name="pre-merge")
        _validate_feature_family(merged, name="merged")
        pre_tokens = (
            self.image_grid_thw[0] * self.image_grid_thw[1] * self.image_grid_thw[2]
        )
        merged_tokens = pre_tokens // (self.spatial_merge_size**2)
        if self.pre_merge_main.shape[-2] != pre_tokens:
            raise ValueError("pre-merge token count differs from image_grid_thw")
        if self.merged_main.shape[-2] != merged_tokens:
            raise ValueError("merged token count differs from image_grid_thw")
        if self.pre_merge_main.device != self.merged_main.device:
            raise ValueError("pre-merge and merged vision tensors must share a device")
        if any(tensor.requires_grad for tensor in (*pre, *merged)):
            raise ValueError(
                "frozen-Qwen vision features must not retain autograd graphs"
            )


class Qwen3RepresentationRuntime:
    """Bound runtime used by representation data/training code."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        renderer: NativeProtocolRenderer,
        model_identity: ModelIdentity,
        architecture: Qwen3RepresentationArchitecture,
        component_paths: Qwen3RepresentationComponentPaths,
        vision_tower: nn.Module,
        merger_modules: tuple[nn.Module, ...],
        projection_identities: tuple[str, ...],
        adapter: TGVFAdapter,
        conditioning_config: TargetConditioningConfig,
        conditioning_provider: BoundTargetConditionProvider,
        conditioning_embedding: nn.Module,
        patch_embed_fast_path: nn.Module | None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.renderer = renderer
        self.model_identity = model_identity
        self.architecture = architecture
        self.component_paths = component_paths
        self.vision_tower = vision_tower
        self.merger_modules = merger_modules
        self.projection_identities = projection_identities
        self.adapter = adapter
        self.conditioning_config = conditioning_config
        self.conditioning_provider = conditioning_provider
        self.patch_embed_fast_path = patch_embed_fast_path
        # The provider deliberately holds only a weak reference to a borrowed
        # embedding. Keep the tokenizer-bounded view alive with the runtime.
        self._conditioning_embedding = conditioning_embedding
        # A scope is local to one execution context and one runtime instance.
        # It is deliberately entered anew for every native same-image group;
        # no validation result survives a group or optimizer boundary.
        self._validated_group_scope: ContextVar[object | None] = ContextVar(
            f"qwen3_representation_group_scope_{id(self)}",
            default=None,
        )

    @property
    def tokenizer(self) -> Any:
        return self.renderer.tokenizer

    def assert_bound_invariants(self) -> None:
        """Reject tokenizer/model/component drift after factory construction."""

        self.renderer.assert_tokenizer_length()
        self.renderer.assert_chat_template_identity()
        _assert_processor_identity(self.renderer, self.model_identity)
        if (
            self.renderer.chat_template_sha256
            != self.model_identity.chat_template_sha256
        ):
            raise ValueError("runtime chat template differs from its ModelIdentity")
        _assert_model_config_identity(self.model, self.model_identity)
        if _read_architecture(self.model) != self.architecture:
            raise RuntimeError("bound Qwen architecture changed")
        if self.model.training or any(
            module.training for module in self.model.modules()
        ):
            raise RuntimeError("bound Qwen must remain entirely in eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("bound Qwen parameters must remain frozen")
        resolved = (
            _resolve_module(self.model, self.component_paths.main_merger),
            *tuple(
                _resolve_module(self.model, path)
                for path in self.component_paths.deepstack_mergers
            ),
        )
        if any(
            actual is not expected
            for actual, expected in zip(resolved, self.merger_modules, strict=True)
        ):
            raise RuntimeError("bound Qwen merger ownership changed")
        if (
            _resolve_module(self.model, self.component_paths.vision_tower)
            is not self.vision_tower
        ):
            raise RuntimeError("bound Qwen vision-tower ownership changed")
        if self.patch_embed_fast_path is not None:
            if getattr(self.vision_tower, "patch_embed", None) is not (
                self.patch_embed_fast_path
            ):
                raise RuntimeError("bound Qwen patch-embedding ownership changed")
            _assert_qwen3_patch_embed_linear_fast_path(self.patch_embed_fast_path)
        embedding = resolve_language_model(self.model).get_input_embeddings()
        if not isinstance(self._conditioning_embedding, _TokenizerBoundEmbedding):
            raise RuntimeError("runtime lost its tokenizer-bounded embedding view")
        if embedding is not self._conditioning_embedding.borrowed_embedding:
            raise RuntimeError("bound Qwen input-embedding ownership changed")
        _ = self._conditioning_embedding.weight

    @contextmanager
    def validated_group_execution(self) -> Iterator[None]:
        """Validate one native group at entry and unconditionally at exit.

        Public runtime operations remain independently fail-closed when called
        outside this scope.  Within one group, their repeated invariant scans
        are redundant: this scope proves the entry state and its ``finally``
        check rejects any tokenizer, template, model, freezing, or component
        mutation before the built group can escape to the trainer.
        """

        if self._validated_group_scope.get() is not None:
            raise RuntimeError("validated representation group scopes cannot nest")
        self.assert_bound_invariants()
        marker = object()
        reset_token = self._validated_group_scope.set(marker)
        try:
            yield
        finally:
            self._validated_group_scope.reset(reset_token)
            self.assert_bound_invariants()

    def _assert_public_runtime_boundary(self) -> None:
        """Run a full check unless this call is inside one validated group."""

        if self._validated_group_scope.get() is None:
            self.assert_bound_invariants()

    def build_target_condition(
        self,
        request: TargetConditioningRequest,
        *,
        contextual_hidden_states: Qwen3ContextualHiddenStateStack | None = None,
    ) -> TargetConditioningOutput:
        """Extract and validate target ``Hq`` through the selected provider."""

        if not isinstance(request, TargetConditioningRequest):
            raise TypeError("request must be a TargetConditioningRequest")
        if request.contextual_hidden_states is not None:
            raise ValueError(
                "runtime requests must not bypass the typed contextual-state stack"
            )
        self._assert_public_runtime_boundary()
        if (
            self.conditioning_config.provider
            is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
        ):
            if contextual_hidden_states is None:
                raise ValueError(
                    "contextual_hidden_state requires the exact frozen-Qwen state stack"
                )
            hidden_layer = self.conditioning_config.hidden_layer
            if hidden_layer is None:
                raise RuntimeError("contextual provider lost its hidden-layer identity")
            request = replace(
                request,
                contextual_hidden_states=contextual_hidden_states.select(hidden_layer),
            )
        elif contextual_hidden_states is not None:
            raise ValueError(
                "target_token_embedding cannot consume contextual hidden states"
            )

        condition = self.conditioning_provider.build(request)
        if condition.provenance.model != self.model_identity:
            raise RuntimeError("Hq provenance differs from the bound model")
        if condition.provenance.provider != self.conditioning_config.provider.value:
            raise RuntimeError("Hq provider provenance differs from runtime selection")
        if condition.values.shape[-1] != self.architecture.language_hidden_size:
            raise ValueError("Hq width differs from the bound Qwen language width")
        self._assert_public_runtime_boundary()
        return condition

    def extract_vision_features(
        self, request: Qwen3VisionPreMergeRequest
    ) -> Qwen3VisionFeatures:
        """Run frozen vision once and capture every model-owned merger boundary."""

        if not isinstance(request, Qwen3VisionPreMergeRequest):
            raise TypeError("request must be Qwen3VisionPreMergeRequest")
        self._assert_public_runtime_boundary()
        merge_size = self.architecture.spatial_merge_size
        grid = tuple(int(value) for value in request.image_grid_thw[0].tolist())
        if grid[1] % merge_size or grid[2] % merge_size:
            raise ValueError("image grid height/width must be spatial-merge divisible")
        owner = _first_module_tensor(self.vision_tower)
        if owner is not None and request.pixel_values.device != owner.device:
            raise ValueError(
                "pixel_values and the bound vision tower must share a device"
            )

        captures: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in self.merger_modules
        ]
        handles = []
        for index, merger in enumerate(self.merger_modules):
            handles.append(
                merger.register_forward_hook(
                    _capture_merger_call(captures[index]), with_kwargs=True
                )
            )
        try:
            with torch.no_grad():
                self.vision_tower(request.pixel_values, grid_thw=request.image_grid_thw)
        finally:
            for handle in handles:
                handle.remove()

        if any(len(rows) != 1 for rows in captures):
            counts = tuple(len(rows) for rows in captures)
            raise RuntimeError(
                f"each Qwen3 vision merger must execute exactly once; observed={counts}"
            )
        pre_merge = tuple(rows[0][0] for rows in captures)
        merged = tuple(rows[0][1] for rows in captures)
        features = Qwen3VisionFeatures(
            model_identity=self.model_identity,
            image_grid_thw=grid,
            spatial_merge_size=merge_size,
            branch_layers=self.architecture.branch_layers,
            projection_identities=self.projection_identities,
            pre_merge_main=pre_merge[0],
            pre_merge_deepstack=pre_merge[1:],
            merged_main=merged[0],
            merged_deepstack=merged[1:],
        )
        if features.pre_merge_main.shape[-1] != self.architecture.vision_hidden_size:
            raise ValueError("pre-merge features differ from the bound vision width")
        if features.merged_main.shape[-1] != self.architecture.language_hidden_size:
            raise ValueError("merged features differ from the bound language width")
        self._assert_public_runtime_boundary()
        return features

    def make_adapter_input(
        self,
        condition: TargetConditioningOutput,
        vision: Qwen3VisionFeatures,
    ) -> TGVFAdapterInput:
        """Bind exact same-model Hq and source vision tensors atomically."""

        if not isinstance(condition, TargetConditioningOutput):
            raise TypeError("condition must be a TargetConditioningOutput")
        if not isinstance(vision, Qwen3VisionFeatures):
            raise TypeError("vision must be Qwen3VisionFeatures")
        self._assert_public_runtime_boundary()
        if condition.provenance.model != self.model_identity:
            raise ValueError("conditioning model differs from the runtime binding")
        if condition.provenance.provider != self.conditioning_config.provider.value:
            raise ValueError("conditioning provider differs from the runtime selection")
        if vision.model_identity != self.model_identity:
            raise ValueError("vision model differs from the runtime binding")
        if vision.projection_identities != self.projection_identities:
            raise ValueError("vision merger identities differ from the runtime binding")
        return TGVFAdapterInput.from_conditioning(
            condition,
            pre_merge_visual_tokens=vision.pre_merge_main,
            deepstack_pre_merge_visual_tokens=vision.pre_merge_deepstack,
        )


def qwen3_input_embedding_identity(model_identity: ModelIdentity) -> str:
    """Canonical identity for the selected model's unresized embedding rows."""

    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")
    if model_identity.family != "qwen3_vl":
        raise ValueError("only qwen3_vl has an accepted representation runtime")
    return f"{model_identity.revision_or_path}::language_model.input_embeddings"


def install_qwen3_patch_embed_linear_fast_path(
    vision_tower: nn.Module,
) -> nn.Module:
    """Replace only Qwen3 patch-embedding arithmetic, preserving model state.

    The Hugging Face implementation applies a full-patch Conv3D to already
    flattened processor patches.  With kernel and stride equal to the complete
    patch, that operation is algebraically a Linear projection.  This installer
    keeps the original Conv3D module, Parameters, state-dict keys, and shapes;
    it binds only a repo-owned ``forward`` implementation to the existing patch
    module.  Patch count remains dynamic, so this does not impose a resolution
    cap.
    """

    if not isinstance(vision_tower, nn.Module):
        raise TypeError("vision_tower must be an nn.Module")
    patch_embed = getattr(vision_tower, "patch_embed", None)
    if not isinstance(patch_embed, nn.Module):
        raise TypeError("Qwen3 vision tower must expose an nn.Module patch_embed")
    _validate_qwen3_patch_embed_module(patch_embed)
    marker = getattr(patch_embed, "_tgvf_fast_path_identity", None)
    if marker is not None:
        if marker != QWEN3_PATCH_EMBED_LINEAR_FAST_PATH:
            raise RuntimeError("Qwen3 patch embed carries an unknown fast path")
        _assert_qwen3_patch_embed_linear_fast_path(patch_embed)
        return patch_embed

    parameter_inventory = tuple(
        (name, id(parameter)) for name, parameter in patch_embed.named_parameters()
    )
    state_inventory = tuple(
        (name, tuple(value.shape), value.dtype, value.device)
        for name, value in patch_embed.state_dict().items()
    )
    original_forward = patch_embed.forward
    try:
        patch_embed.forward = MethodType(  # type: ignore[method-assign]
            _qwen3_patch_embed_linear_forward,
            patch_embed,
        )
        patch_embed._tgvf_fast_path_identity = (  # type: ignore[attr-defined]
            QWEN3_PATCH_EMBED_LINEAR_FAST_PATH
        )
        _assert_qwen3_patch_embed_linear_fast_path(patch_embed)
        if parameter_inventory != tuple(
            (name, id(parameter)) for name, parameter in patch_embed.named_parameters()
        ):
            raise RuntimeError("patch-embed fast path changed Parameter identity")
        if state_inventory != tuple(
            (name, tuple(value.shape), value.dtype, value.device)
            for name, value in patch_embed.state_dict().items()
        ):
            raise RuntimeError("patch-embed fast path changed state-dict inventory")
    except Exception:
        patch_embed.forward = original_forward  # type: ignore[method-assign]
        if hasattr(patch_embed, "_tgvf_fast_path_identity"):
            delattr(patch_embed, "_tgvf_fast_path_identity")
        raise
    return patch_embed


def _qwen3_patch_embed_linear_forward(
    patch_embed: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    _validate_qwen3_patch_embed_module(patch_embed)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("Qwen3 patch input must be a torch.Tensor")
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[0] == 0
        or hidden_states.shape[1] != _QWEN3_PATCH_EMBED_INPUT_WIDTH
    ):
        raise ValueError("Qwen3 patch input must have shape [N,1536] with positive N")
    if not hidden_states.is_floating_point():
        raise TypeError("Qwen3 patch input must use a floating dtype")
    projection = patch_embed.proj
    if hidden_states.device != projection.weight.device:
        raise ValueError("Qwen3 patch input and projection must share a device")
    return F.linear(
        hidden_states.to(dtype=projection.weight.dtype),
        projection.weight.reshape(projection.out_channels, -1),
        projection.bias,
    )


def _assert_qwen3_patch_embed_linear_fast_path(patch_embed: nn.Module) -> None:
    _validate_qwen3_patch_embed_module(patch_embed)
    if (
        getattr(patch_embed, "_tgvf_fast_path_identity", None)
        != QWEN3_PATCH_EMBED_LINEAR_FAST_PATH
    ):
        raise RuntimeError("Qwen3 patch-embedding fast path identity changed")
    forward = getattr(patch_embed, "forward", None)
    if getattr(forward, "__func__", None) is not _qwen3_patch_embed_linear_forward:
        raise RuntimeError("Qwen3 patch-embedding forward implementation changed")


def _validate_qwen3_patch_embed_module(patch_embed: nn.Module) -> None:
    projection = getattr(patch_embed, "proj", None)
    if not isinstance(projection, nn.Conv3d):
        raise TypeError("Qwen3 patch_embed.proj must remain an nn.Conv3d")
    if (
        tuple(projection.kernel_size) != _QWEN3_PATCH_EMBED_KERNEL
        or tuple(projection.stride) != _QWEN3_PATCH_EMBED_KERNEL
        or tuple(projection.padding) != (0, 0, 0)
        or tuple(projection.dilation) != (1, 1, 1)
        or projection.groups != 1
        or projection.padding_mode != "zeros"
    ):
        raise ValueError("Qwen3 patch projection geometry is unsupported")
    if projection.in_channels != 3 or projection.out_channels != 1152:
        raise ValueError("Qwen3 patch projection channels are unsupported")
    if projection.bias is None:
        raise ValueError("Qwen3 patch projection requires its checkpoint bias")
    if not projection.weight.is_contiguous():
        raise ValueError("Qwen3 patch projection weight must remain contiguous")
    if (
        projection.weight.device != projection.bias.device
        or projection.weight.dtype != projection.bias.dtype
    ):
        raise ValueError("Qwen3 patch projection weight/bias must share device/dtype")
    expected_attributes = {
        "in_channels": 3,
        "temporal_patch_size": 2,
        "patch_size": 16,
        "embed_dim": 1152,
    }
    if any(
        getattr(patch_embed, name, None) != expected
        for name, expected in expected_attributes.items()
    ):
        raise ValueError("Qwen3 patch-embedding module attributes changed")


def create_qwen3_representation_runtime(
    *,
    model: nn.Module,
    processor: Any,
    model_identity: ModelIdentity,
    conditioning_config: TargetConditioningConfig,
    adapter_dtype: torch.dtype,
    attn_dim: int | None = None,
    component_paths: Qwen3RepresentationComponentPaths | None = None,
    fixture_mode: bool = False,
) -> Qwen3RepresentationRuntime:
    """Bind one frozen Qwen3 model to a new trainable TGVF Adapter.

    ``fixture_mode`` exists only for tiny CPU contract fixtures.  Without it,
    the accepted local Qwen3-VL-8B-Thinking identity and exact architecture are
    mandatory. ``adapter_dtype`` is deliberately explicit because representation
    training precision has no accepted hidden default.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")
    if model_identity.family != "qwen3_vl":
        raise ValueError(
            "representation runtime supports only qwen3_vl; Qwen2.5-VL needs "
            "its own accepted artifact and DeepStack path"
        )
    if not isinstance(conditioning_config, TargetConditioningConfig):
        raise TypeError("conditioning_config must be TargetConditioningConfig")
    if (
        not isinstance(adapter_dtype, torch.dtype)
        or not adapter_dtype.is_floating_point
    ):
        raise TypeError("adapter_dtype must be an explicit floating-point torch dtype")
    if not isinstance(fixture_mode, bool):
        raise TypeError("fixture_mode must be a bool")
    paths = component_paths or Qwen3RepresentationComponentPaths()
    if not isinstance(paths, Qwen3RepresentationComponentPaths):
        raise TypeError("component_paths must be Qwen3RepresentationComponentPaths")

    renderer = NativeProtocolRenderer(
        processor, expected_tokenizer_length=model_identity.tokenizer_length
    )
    if renderer.chat_template_sha256 != model_identity.chat_template_sha256:
        raise ValueError("processor chat template differs from ModelIdentity")
    _assert_processor_identity(renderer, model_identity)
    _assert_model_config_identity(model, model_identity)
    architecture = _read_architecture(model)
    if architecture.spatial_merge_size != QWEN3_REPRESENTATION_SPATIAL_MERGE_SIZE:
        raise ValueError("Qwen3 representation runtime requires spatial merge size 2")
    if not fixture_mode:
        _assert_accepted_production_identity(model_identity, architecture)
        if paths != Qwen3RepresentationComponentPaths():
            raise ValueError(
                "production runtime requires accepted Qwen3 component paths"
            )
    elif not (
        model_identity.model_name.startswith("tiny-")
        and model_identity.revision_or_path.startswith("/tiny-")
    ):
        raise ValueError(
            "fixture_mode is restricted to explicit tiny fixture identities"
        )

    vision_tower = _resolve_module(model, paths.vision_tower)
    mergers = (
        _resolve_module(model, paths.main_merger),
        *tuple(_resolve_module(model, path) for path in paths.deepstack_mergers),
    )
    if len({id(module) for module in mergers}) != 4:
        raise ValueError("main and DeepStack mergers must be distinct model modules")
    model_module_ids = {id(module) for module in model.modules()}
    if any(id(module) not in model_module_ids for module in (vision_tower, *mergers)):
        raise ValueError("all vision components must be owned by the selected model")

    model.requires_grad_(False)
    model.eval()
    patch_embed_fast_path = (
        None
        if fixture_mode
        else install_qwen3_patch_embed_linear_fast_path(vision_tower)
    )
    language_model = resolve_language_model(model)
    embedding = language_model.get_input_embeddings()
    if not isinstance(embedding, nn.Module):
        raise TypeError("Qwen input embedding must be an nn.Module")
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("Qwen input embedding must expose a rank-two weight")
    if int(weight.shape[0]) < model_identity.tokenizer_length:
        raise ValueError("Qwen embedding has fewer rows than the bound tokenizer")
    if int(weight.shape[1]) != architecture.language_hidden_size:
        raise ValueError("Qwen embedding width differs from text_config.hidden_size")
    vision_owner = _first_module_tensor(vision_tower)
    if vision_owner is None:
        raise ValueError("Qwen vision tower must own parameters or buffers")
    if vision_owner.device != weight.device:
        raise ValueError(
            "representation runtime requires Qwen vision and language state on one device"
        )
    for merger in mergers:
        merger_owner = _first_module_tensor(merger)
        if merger_owner is None or merger_owner.device != vision_owner.device:
            raise ValueError(
                "all Qwen mergers must be materialized on the vision device"
            )

    projection_ids = tuple(
        _projection_identity(model_identity, path)
        for path in (paths.main_merger, *paths.deepstack_mergers)
    )
    ports = tuple(
        FrozenProjectionPort(
            merger,
            identity=identity,
            input_dim=architecture.vision_hidden_size,
            output_dim=architecture.language_hidden_size,
            spatial_merge_size=architecture.spatial_merge_size,
        )
        for merger, identity in zip(mergers, projection_ids, strict=True)
    )
    with torch.device(weight.device):
        adapter = TGVFAdapter(
            d_lm=architecture.language_hidden_size,
            d_v=architecture.vision_hidden_size,
            attn_dim=attn_dim,
            main_projection=ports[0],
            deepstack_projections=DDeepStackProjectionPorts(
                branch_layers=architecture.branch_layers,
                projections=ports[1:],
            ),
            branch_layers=architecture.branch_layers,
        )
    _cast_adapter_owned_modules(adapter, device=weight.device, dtype=adapter_dtype)
    adapter.train(True)

    tokenizer_embedding = _TokenizerBoundEmbedding(
        embedding, tokenizer_length=model_identity.tokenizer_length
    )
    expected_embedding_identity = qwen3_input_embedding_identity(model_identity)
    if (
        conditioning_config.provider
        is TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
        and conditioning_config.embedding_identity != expected_embedding_identity
    ):
        raise ValueError(
            "target_token_embedding must use the canonical bound Qwen embedding identity"
        )
    provider = create_target_condition_provider(
        config=conditioning_config,
        model_identity=model_identity,
        dependencies=TargetConditioningDependencies(base_embedding=tokenizer_embedding),
    )
    runtime = Qwen3RepresentationRuntime(
        model=model,
        processor=processor,
        renderer=renderer,
        model_identity=model_identity,
        architecture=architecture,
        component_paths=paths,
        vision_tower=vision_tower,
        merger_modules=mergers,
        projection_identities=projection_ids,
        adapter=adapter,
        conditioning_config=conditioning_config,
        conditioning_provider=provider,
        conditioning_embedding=tokenizer_embedding,
        patch_embed_fast_path=patch_embed_fast_path,
    )
    runtime.assert_bound_invariants()
    return runtime


class _TokenizerBoundEmbedding(nn.Module):
    """Read-only valid-token row view over a possibly padded Qwen embedding."""

    def __init__(self, embedding: nn.Module, *, tokenizer_length: int) -> None:
        super().__init__()
        self._embedding_ref = weakref.ref(embedding)
        self.num_embeddings = int(tokenizer_length)
        weight = self.weight
        if weight.ndim != 2 or weight.shape[0] != tokenizer_length:
            raise ValueError("tokenizer-bounded embedding view has an invalid shape")

    @property
    def borrowed_embedding(self) -> nn.Module:
        embedding = self._embedding_ref()
        if embedding is None:
            raise RuntimeError("bound Qwen input embedding no longer exists")
        return embedding

    @property
    def weight(self) -> torch.Tensor:
        weight = getattr(self.borrowed_embedding, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("bound Qwen input embedding lost its weight")
        if weight.ndim != 2 or weight.shape[0] < self.num_embeddings:
            raise ValueError("bound Qwen input embedding rows changed")
        return weight[: self.num_embeddings]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("embedding input_ids must be a torch.Tensor")
        return self.borrowed_embedding(input_ids)


def _read_architecture(model: nn.Module) -> Qwen3RepresentationArchitecture:
    config = getattr(model, "config", None)
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", None)
    if config is None or vision is None or text is None:
        raise TypeError("Qwen3 model config must expose vision_config and text_config")
    vision_hidden = _required_positive_int(vision, "hidden_size")
    language_hidden = _required_positive_int(text, "hidden_size")
    out_hidden = _required_positive_int(vision, "out_hidden_size")
    if out_hidden != language_hidden:
        raise ValueError("vision out_hidden_size differs from text hidden_size")
    merge_size = _required_positive_int(vision, "spatial_merge_size")
    raw_layers = getattr(vision, "deepstack_visual_indexes", None)
    if isinstance(raw_layers, (str, bytes)) or not isinstance(raw_layers, Sequence):
        raise TypeError("vision config must expose DeepStack layer indexes")
    layers = tuple(int(layer) for layer in raw_layers)
    return Qwen3RepresentationArchitecture(
        language_hidden_size=language_hidden,
        vision_hidden_size=vision_hidden,
        spatial_merge_size=merge_size,
        branch_layers=layers,
    )


def _assert_accepted_production_identity(
    identity: ModelIdentity, architecture: Qwen3RepresentationArchitecture
) -> None:
    if identity.revision_or_path != ACCEPTED_QWEN3_MODEL_PATH:
        raise ValueError("production runtime requires the accepted local Qwen3 path")
    if Path(identity.model_name).name != "Qwen3-VL-8B-Thinking":
        raise ValueError("production runtime requires Qwen3-VL-8B-Thinking")
    if identity.tokenizer_length != ACCEPTED_QWEN3_TOKENIZER_LENGTH:
        raise ValueError("production runtime tokenizer identity differs from golden")
    if identity.chat_template_sha256 != ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256:
        raise ValueError(
            "production runtime chat-template identity differs from golden"
        )
    expected = (
        QWEN3_REPRESENTATION_LANGUAGE_DIM,
        QWEN3_REPRESENTATION_VISION_DIM,
        QWEN3_REPRESENTATION_SPATIAL_MERGE_SIZE,
        QWEN3_REPRESENTATION_BRANCH_LAYERS,
    )
    actual = (
        architecture.language_hidden_size,
        architecture.vision_hidden_size,
        architecture.spatial_merge_size,
        architecture.branch_layers,
    )
    if actual != expected:
        raise ValueError(
            f"production Qwen3 architecture differs: {actual} vs {expected}"
        )


def _assert_model_config_identity(model: nn.Module, identity: ModelIdentity) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("bound Qwen must expose config")
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise ValueError("bound model config is not qwen3_vl")
    actual_path = getattr(config, "_name_or_path", None)
    if not isinstance(actual_path, str) or actual_path != identity.revision_or_path:
        raise ValueError("bound model path differs from ModelIdentity")
    if Path(identity.model_name).name != Path(identity.revision_or_path).name:
        raise ValueError("ModelIdentity model name and path basename differ")


def _assert_processor_identity(
    renderer: NativeProtocolRenderer, identity: ModelIdentity
) -> None:
    tokenizer_path = getattr(renderer.tokenizer, "name_or_path", None)
    if (
        not isinstance(tokenizer_path, str)
        or tokenizer_path != identity.revision_or_path
    ):
        raise ValueError("bound tokenizer path differs from ModelIdentity")


def _cast_adapter_owned_modules(
    adapter: TGVFAdapter, *, device: torch.device, dtype: torch.dtype
) -> None:
    borrowed = {"main_projection", "d_deepstack_projections"}
    for name, module in adapter.named_children():
        if name not in borrowed:
            module.to(device=device, dtype=dtype)
    for name, parameter in adapter.named_parameters():
        is_borrowed = name.startswith(("main_projection.", "d_deepstack_projections."))
        if is_borrowed:
            if parameter.requires_grad:
                raise RuntimeError("borrowed Qwen merger became trainable")
            continue
        if not parameter.requires_grad:
            raise RuntimeError(
                f"Adapter-owned parameter is unexpectedly frozen: {name}"
            )
        if parameter.device != device or parameter.dtype != dtype:
            raise RuntimeError(f"Adapter-owned parameter placement differs: {name}")


def _required_positive_int(owner: Any, name: str) -> int:
    value = getattr(owner, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen3 config {name} must be a positive integer")
    return int(value)


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if part.isdecimal():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Qwen component path is unavailable: {path}"
                ) from error
        else:
            if not hasattr(current, part):
                raise ValueError(f"Qwen component path is unavailable: {path}")
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Qwen component path is not an nn.Module: {path}")
    return current


def _projection_identity(identity: ModelIdentity, path: str) -> str:
    material = (
        f"{identity.family}\0{identity.model_name}\0{identity.revision_or_path}\0{path}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{identity.family}:{path}:{digest}"


def _capture_merger_call(
    destination: list[tuple[torch.Tensor, torch.Tensor]],
):
    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        source = _extract_merger_input(args, kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Qwen3 merger must return a torch.Tensor")
        if source.ndim != 2 or output.ndim != 2:
            raise ValueError("Qwen3 merger boundaries must be rank-two tensors")
        destination.append((source.detach().clone(), output.detach().clone()))

    return hook


def _extract_merger_input(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.Tensor:
    positional = args[0] if args and isinstance(args[0], torch.Tensor) else None
    keyword = kwargs.get("hidden_states")
    if keyword is not None and not isinstance(keyword, torch.Tensor):
        raise TypeError("Qwen3 merger hidden_states must be a torch.Tensor")
    if positional is not None and keyword is not None and positional is not keyword:
        raise ValueError("Qwen3 merger received ambiguous hidden-state inputs")
    source = positional if positional is not None else keyword
    if not isinstance(source, torch.Tensor):
        raise TypeError("Qwen3 merger did not expose its pre-merge hidden states")
    return source


def _first_module_tensor(module: nn.Module) -> torch.Tensor | None:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter
    return next(module.buffers(), None)


def _validate_feature_family(tensors: tuple[torch.Tensor, ...], *, name: str) -> None:
    reference = tensors[0]
    if not isinstance(reference, torch.Tensor):
        raise TypeError(f"{name} features must be tensors")
    if reference.ndim != 2 or not reference.is_floating_point():
        raise ValueError(f"{name} features must be floating rank-two tensors")
    for tensor in tensors[1:]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} features must be tensors")
        if tensor.shape != reference.shape:
            raise ValueError(f"all {name} feature branches must share shape")
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError(f"all {name} feature branches must share device/dtype")


__all__ = [
    "ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256",
    "ACCEPTED_QWEN3_MODEL_PATH",
    "ACCEPTED_QWEN3_TOKENIZER_LENGTH",
    "QWEN3_REPRESENTATION_BRANCH_LAYERS",
    "QWEN3_REPRESENTATION_LANGUAGE_DIM",
    "QWEN3_REPRESENTATION_SPATIAL_MERGE_SIZE",
    "QWEN3_REPRESENTATION_VISION_DIM",
    "QWEN3_PATCH_EMBED_LINEAR_FAST_PATH",
    "Qwen3ContextualHiddenStateStack",
    "Qwen3RepresentationArchitecture",
    "Qwen3RepresentationComponentPaths",
    "Qwen3RepresentationRuntime",
    "Qwen3VisionFeatures",
    "Qwen3VisionPreMergeRequest",
    "create_qwen3_representation_runtime",
    "install_qwen3_patch_embed_linear_fast_path",
    "qwen3_input_embedding_identity",
]
