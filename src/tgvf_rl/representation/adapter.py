"""Batch-aware TGVF bidirectional adapter with required D-DeepStack output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from tgvf_rl.conditioning.base import (
    TargetConditioningOutput,
    TargetConditioningProvenance,
)

from .deepstack import (
    DDeepStackPayload,
    DDeepStackProjectionPorts,
    FrozenProjectionPort,
    _validate_token_tensor,
)


TGVF_ADAPTER_OUTPUT_SCHEMA_VERSION = "tgvf-adapter-output-v2"
_BORROWED_PROJECTION_STATE_PREFIXES = (
    "main_projection.",
    "d_deepstack_projections.",
)


@dataclass(frozen=True, slots=True)
class TGVFAdapterInput:
    """All target and original-image pre-merge state needed by the adapter."""

    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    deepstack_pre_merge_visual_tokens: tuple[torch.Tensor, ...]
    condition_provenance: TargetConditioningProvenance | None = None

    def __post_init__(self) -> None:
        _validate_token_tensor(self.target_hidden_states, name="target_hidden_states")
        _validate_token_tensor(
            self.pre_merge_visual_tokens, name="pre_merge_visual_tokens"
        )
        if not self.deepstack_pre_merge_visual_tokens:
            raise ValueError("all required D-DeepStack branches must be provided")
        if self.target_hidden_states.ndim != self.pre_merge_visual_tokens.ndim:
            raise ValueError(
                "target and visual inputs cannot mix batched and unbatched ranks"
            )
        if self.target_hidden_states.ndim == 3 and (
            self.target_hidden_states.shape[0] != self.pre_merge_visual_tokens.shape[0]
        ):
            raise ValueError("target and visual input batch sizes must match")
        if self.target_hidden_states.device != self.pre_merge_visual_tokens.device:
            raise ValueError("target and visual inputs must share a device")

        visual_prefix = self.pre_merge_visual_tokens.shape[:-1]
        for index, branch in enumerate(self.deepstack_pre_merge_visual_tokens):
            _validate_token_tensor(branch, name=f"deepstack branch {index}")
            if branch.shape[:-1] != visual_prefix:
                raise ValueError(
                    "every D-DeepStack pre-merge branch must match the main visual layout"
                )
            if branch.device != self.pre_merge_visual_tokens.device:
                raise ValueError("all visual inputs must share a device")

        provenance = self.condition_provenance
        if provenance is not None:
            target_is_batched = self.target_hidden_states.ndim == 3
            if provenance.batched != target_is_batched:
                raise ValueError("conditioning provenance batch mode differs from Hq")
            if self.target_hidden_states.shape[-2] != (
                provenance.target_span.end - provenance.target_span.start
            ):
                raise ValueError("Hq token count differs from conditioning provenance")
            if target_is_batched and (
                self.target_hidden_states.shape[0] != provenance.source_batch_size
            ):
                raise ValueError("Hq batch size differs from conditioning provenance")

    @classmethod
    def from_conditioning(
        cls,
        condition: TargetConditioningOutput,
        *,
        pre_merge_visual_tokens: torch.Tensor,
        deepstack_pre_merge_visual_tokens: Sequence[torch.Tensor],
    ) -> TGVFAdapterInput:
        if not isinstance(condition, TargetConditioningOutput):
            raise TypeError("condition must be a TargetConditioningOutput")
        return cls(
            target_hidden_states=condition.values,
            pre_merge_visual_tokens=pre_merge_visual_tokens,
            deepstack_pre_merge_visual_tokens=tuple(deepstack_pre_merge_visual_tokens),
            condition_provenance=condition.provenance,
        )


@dataclass(frozen=True, slots=True)
class BidirectionalAttentionOutput:
    conditioned_visual_tokens: torch.Tensor
    target_to_visual_attention: torch.Tensor
    visual_to_target_attention: torch.Tensor
    gate: torch.Tensor
    visual_salience: torch.Tensor


class TGVFBidirectionalAttention(nn.Module):
    """Two-way target/vision attention followed by a gated visual residual."""

    def __init__(self, *, d_lm: int, d_v: int, attn_dim: int | None = None) -> None:
        super().__init__()
        if d_lm <= 0 or d_v <= 0:
            raise ValueError("TGVF dimensions must be positive")
        if attn_dim is not None and attn_dim <= 0:
            raise ValueError("attn_dim must be positive when provided")
        self.d_lm = int(d_lm)
        self.d_v = int(d_v)
        self.attn_dim = self.d_v if attn_dim is None else int(attn_dim)

        self.target_norm = nn.LayerNorm(self.d_lm)
        self.target_proj = nn.Linear(self.d_lm, self.attn_dim)
        self.visual_norm = nn.LayerNorm(self.d_v)
        self.visual_proj = nn.Linear(self.d_v, self.attn_dim)
        self.target_q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.visual_k_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.visual_v_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.enriched_target_norm = nn.LayerNorm(self.attn_dim)
        self.visual_q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.target_k_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.target_v_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.context_to_delta = nn.Linear(self.attn_dim, self.d_v)
        self.gate_proj = nn.Linear(self.d_v + self.attn_dim, self.d_v)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
    ) -> BidirectionalAttentionOutput:
        _validate_token_tensor(
            target_hidden_states, name="target_hidden_states", feature_dim=self.d_lm
        )
        _validate_token_tensor(
            pre_merge_visual_tokens,
            name="pre_merge_visual_tokens",
            feature_dim=self.d_v,
        )
        if target_hidden_states.ndim != pre_merge_visual_tokens.ndim:
            raise ValueError(
                "target and visual tensors cannot mix batched/unbatched ranks"
            )
        if target_hidden_states.ndim == 3 and (
            target_hidden_states.shape[0] != pre_merge_visual_tokens.shape[0]
        ):
            raise ValueError("target and visual batch sizes must match")
        if target_hidden_states.device != pre_merge_visual_tokens.device:
            raise ValueError("target and visual tensors must share a device")

        target = target_hidden_states.to(dtype=self.target_norm.weight.dtype)
        visual_raw = pre_merge_visual_tokens.to(dtype=self.visual_norm.weight.dtype)
        target_tokens = self.target_proj(self.target_norm(target))
        visual_tokens = self.visual_norm(visual_raw)
        visual_projected = self.visual_proj(visual_tokens)

        target_context, target_to_visual = _cross_attention(
            self.target_q_proj(target_tokens),
            self.visual_k_proj(visual_projected),
            self.visual_v_proj(visual_projected),
        )
        enriched_target = self.enriched_target_norm(target_tokens + target_context)
        visual_context, visual_to_target = _cross_attention(
            self.visual_q_proj(visual_projected),
            self.target_k_proj(enriched_target),
            self.target_v_proj(enriched_target),
        )
        delta = self.context_to_delta(visual_context)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat((visual_tokens, visual_context), dim=-1))
        )
        gated_delta = gate * delta
        conditioned = visual_raw + gated_delta
        visual_salience = torch.softmax(
            torch.linalg.vector_norm(gated_delta.float(), dim=-1), dim=-1
        ).to(dtype=gated_delta.dtype)
        if gated_delta.ndim == 2:
            visual_salience = visual_salience.unsqueeze(0)
        return BidirectionalAttentionOutput(
            conditioned_visual_tokens=conditioned,
            target_to_visual_attention=target_to_visual,
            visual_to_target_attention=visual_to_target,
            gate=gate,
            visual_salience=visual_salience,
        )


@dataclass(frozen=True, slots=True)
class TGVFAdapterMetadata:
    branch_layers: tuple[int, ...]
    main_projection_identity: str
    deepstack_projection_identities: tuple[str, ...]
    batched: bool
    batch_size: int
    target_token_count: int
    pre_merge_visual_token_count: int
    d_token_count: int
    condition_provenance: TargetConditioningProvenance | None
    schema_version: str = TGVF_ADAPTER_OUTPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TGVF_ADAPTER_OUTPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported TGVF Adapter schema {self.schema_version!r}")
        if self.batch_size <= 0:
            raise ValueError("TGVF Adapter batch size must be positive")
        if (
            min(
                self.target_token_count,
                self.pre_merge_visual_token_count,
                self.d_token_count,
            )
            <= 0
        ):
            raise ValueError("TGVF Adapter token counts must be positive")
        if len(self.branch_layers) != len(self.deepstack_projection_identities):
            raise ValueError("branch layers and projection identities must align")


@dataclass(frozen=True, slots=True)
class TGVFAdapterOutput:
    main_d: torch.Tensor
    d_deepstack: DDeepStackPayload
    conditioned_pre_merge_visual_tokens: torch.Tensor
    conditioned_deepstack_pre_merge_visual_tokens: tuple[torch.Tensor, ...]
    main_attention: BidirectionalAttentionOutput
    deepstack_attention: tuple[BidirectionalAttentionOutput, ...]
    metadata: TGVFAdapterMetadata

    def __post_init__(self) -> None:
        _validate_token_tensor(self.main_d, name="main D")
        if len(self.conditioned_deepstack_pre_merge_visual_tokens) != len(
            self.d_deepstack.branches
        ):
            raise ValueError(
                "conditioned and projected D-DeepStack branches must align"
            )
        if len(self.deepstack_attention) != len(self.d_deepstack.branches):
            raise ValueError("D-DeepStack attention records must align with branches")
        for branch in self.d_deepstack.branches:
            if branch.shape != self.main_d.shape:
                raise ValueError("main D and all D-DeepStack outputs must share shape")
            if branch.device != self.main_d.device or branch.dtype != self.main_d.dtype:
                raise ValueError(
                    "main D and all D-DeepStack outputs must share device/dtype"
                )

    @property
    def deepstack_visual_embeds(self) -> tuple[torch.Tensor, ...]:
        return self.d_deepstack.branches


class TGVFAdapter(TGVFBidirectionalAttention):
    """Produce main D and every configured D-DeepStack branch."""

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        main_projection: FrozenProjectionPort,
        deepstack_projections: DDeepStackProjectionPorts
        | Sequence[FrozenProjectionPort],
        branch_layers: Sequence[int] = (8, 16, 24),
        attn_dim: int | None = None,
    ) -> None:
        super().__init__(d_lm=d_lm, d_v=d_v, attn_dim=attn_dim)
        if not isinstance(main_projection, FrozenProjectionPort):
            raise TypeError("main_projection must be a FrozenProjectionPort")
        if isinstance(deepstack_projections, DDeepStackProjectionPorts):
            projection_ports = deepstack_projections
            if (
                tuple(int(layer) for layer in branch_layers)
                != projection_ports.branch_layers
            ):
                raise ValueError(
                    "branch_layers differ from the D-DeepStack projection ports"
                )
        else:
            projection_ports = DDeepStackProjectionPorts(
                branch_layers=branch_layers, projections=deepstack_projections
            )
        expected_signature = (self.d_v, self.d_lm)
        if (
            main_projection.input_dim,
            main_projection.output_dim,
        ) != expected_signature:
            raise ValueError("main projection dimensions differ from TGVF dimensions")
        for port in projection_ports.projections:
            if (port.input_dim, port.output_dim) != expected_signature:
                raise ValueError(
                    "D-DeepStack projection dimensions differ from TGVF dimensions"
                )
            if port.spatial_merge_size != main_projection.spatial_merge_size:
                raise ValueError(
                    "main and D-DeepStack projections must share merge size"
                )
        all_ports = (main_projection, *tuple(projection_ports.projections))
        if len({id(port) for port in all_ports}) != len(all_ports):
            raise ValueError(
                "main and D-DeepStack branches cannot share projection ports"
            )
        if len({port.identity for port in all_ports}) != len(all_ports):
            raise ValueError("all frozen projection identities must be unique")

        self.spatial_merge_size = main_projection.spatial_merge_size
        self.main_projection = main_projection
        self.d_deepstack_projections = projection_ports
        self.d_deepstack_branch_layers = projection_ports.branch_layers
        self.d_deepstack_branch_adapters = nn.ModuleDict(
            {
                str(layer): TGVFBidirectionalAttention(
                    d_lm=self.d_lm, d_v=self.d_v, attn_dim=self.attn_dim
                )
                for layer in self.d_deepstack_branch_layers
            }
        )

    def artifact_state_dict(
        self, *, keep_vars: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return the Adapter-owned tensor subset, excluding Qwen mergers.

        The projection ports are registered modules so device placement and
        forward execution remain explicit, but their parameters belong to the
        frozen base model and are forbidden in the deployable Adapter artifact.
        This tensor subset is not a complete artifact by itself: the eventual
        checkpoint manifest must bind model, provider, projection, architecture,
        data, and training identities before loading it across runs.
        """

        full_state = super().state_dict(keep_vars=keep_vars)
        artifact = {
            name: value
            for name, value in full_state.items()
            if not name.startswith(_BORROWED_PROJECTION_STATE_PREFIXES)
        }
        if not artifact:
            raise RuntimeError("TGVF Adapter artifact state is unexpectedly empty")
        return artifact

    def load_artifact_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Strictly load an Adapter-only artifact without touching Qwen state."""

        if not isinstance(state, Mapping):
            raise TypeError("TGVF Adapter artifact state must be a mapping")
        expected = self.artifact_state_dict(keep_vars=True)
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise ValueError(
                "TGVF Adapter artifact keys mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        for name, expected_value in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TGVF Adapter artifact value {name!r} is not a tensor")
            if (
                value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
            ):
                raise ValueError(
                    f"TGVF Adapter artifact tensor {name!r} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}, expected "
                    f"{tuple(expected_value.shape)}/{expected_value.dtype}"
                )

        full_state_keys = set(super().state_dict())
        borrowed_keys = full_state_keys - set(expected)
        if not borrowed_keys or any(
            not name.startswith(_BORROWED_PROJECTION_STATE_PREFIXES)
            for name in borrowed_keys
        ):
            raise RuntimeError("borrowed Qwen merger state boundary is inconsistent")
        incompatible = super().load_state_dict(dict(state), strict=False)
        if (
            incompatible.unexpected_keys
            or set(incompatible.missing_keys) != borrowed_keys
        ):
            raise RuntimeError("strict TGVF Adapter tensor-subset load diverged")

    def forward(
        self,
        inputs: TGVFAdapterInput | None = None,
        *,
        target_hidden_states: torch.Tensor | None = None,
        pre_merge_visual_tokens: torch.Tensor | None = None,
        deepstack_pre_merge_visual_tokens: Sequence[torch.Tensor] | None = None,
        condition_provenance: TargetConditioningProvenance | None = None,
    ) -> TGVFAdapterOutput:
        if inputs is not None:
            if any(
                value is not None
                for value in (
                    target_hidden_states,
                    pre_merge_visual_tokens,
                    deepstack_pre_merge_visual_tokens,
                    condition_provenance,
                )
            ):
                raise ValueError(
                    "pass either TGVFAdapterInput or explicit tensor arguments"
                )
            if not isinstance(inputs, TGVFAdapterInput):
                raise TypeError("inputs must be a TGVFAdapterInput")
            adapter_input = inputs
        else:
            if target_hidden_states is None or pre_merge_visual_tokens is None:
                raise ValueError(
                    "target_hidden_states and pre_merge_visual_tokens are required"
                )
            if deepstack_pre_merge_visual_tokens is None:
                raise ValueError("all required D-DeepStack branch inputs are required")
            adapter_input = TGVFAdapterInput(
                target_hidden_states=target_hidden_states,
                pre_merge_visual_tokens=pre_merge_visual_tokens,
                deepstack_pre_merge_visual_tokens=tuple(
                    deepstack_pre_merge_visual_tokens
                ),
                condition_provenance=condition_provenance,
            )
        if len(adapter_input.deepstack_pre_merge_visual_tokens) != len(
            self.d_deepstack_branch_layers
        ):
            raise ValueError(
                "D-DeepStack branch count mismatch: "
                f"features={len(adapter_input.deepstack_pre_merge_visual_tokens)} "
                f"layers={len(self.d_deepstack_branch_layers)}"
            )

        main_attention = super().forward(
            target_hidden_states=adapter_input.target_hidden_states,
            pre_merge_visual_tokens=adapter_input.pre_merge_visual_tokens,
        )
        branch_attention = tuple(
            self.d_deepstack_branch_adapters[str(layer)](
                target_hidden_states=adapter_input.target_hidden_states,
                pre_merge_visual_tokens=branch,
            )
            for layer, branch in zip(
                self.d_deepstack_branch_layers,
                adapter_input.deepstack_pre_merge_visual_tokens,
                strict=True,
            )
        )
        conditioned_branches = tuple(
            output.conditioned_visual_tokens for output in branch_attention
        )
        main_d = self.main_projection(main_attention.conditioned_visual_tokens)
        d_deepstack = self.d_deepstack_projections(conditioned_branches)

        batched = main_d.ndim == 3
        batch_size = int(main_d.shape[0]) if batched else 1
        return TGVFAdapterOutput(
            main_d=main_d,
            d_deepstack=d_deepstack,
            conditioned_pre_merge_visual_tokens=main_attention.conditioned_visual_tokens,
            conditioned_deepstack_pre_merge_visual_tokens=conditioned_branches,
            main_attention=main_attention,
            deepstack_attention=branch_attention,
            metadata=TGVFAdapterMetadata(
                branch_layers=self.d_deepstack_branch_layers,
                main_projection_identity=self.main_projection.identity,
                deepstack_projection_identities=d_deepstack.projection_identities,
                batched=batched,
                batch_size=batch_size,
                target_token_count=int(adapter_input.target_hidden_states.shape[-2]),
                pre_merge_visual_token_count=int(
                    adapter_input.pre_merge_visual_tokens.shape[-2]
                ),
                d_token_count=int(main_d.shape[-2]),
                condition_provenance=adapter_input.condition_provenance,
            ),
        )


def _cross_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(queries.shape[-1])
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, values), attention


__all__ = [
    "TGVF_ADAPTER_OUTPUT_SCHEMA_VERSION",
    "BidirectionalAttentionOutput",
    "TGVFAdapter",
    "TGVFAdapterInput",
    "TGVFAdapterMetadata",
    "TGVFAdapterOutput",
    "TGVFBidirectionalAttention",
]
