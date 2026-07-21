"""Lossless Qwen3-VL multimodal packing for recorded TGVF observations.

The vLLM Qwen3-VL model represents one visual token as the concatenation of
the main visual embedding followed by its three DeepStack embeddings.  This
module is the sole bridge from project-owned, content-addressed replay state to
that transport representation.  It never reruns the vision tower or the TGVF
Adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.contracts.tensors import TensorArtifactRef
from tgvf_rl.observations.schema import (
    CropObservationRecord,
    CropTGVFObservationRecord,
    FocusedObservationRecord,
    SourceVisualState,
    TrajectorySourceVisual,
)
from tgvf_rl.observations.store import (
    ObservationHandle,
    ObservationStore,
    TrajectoryReplayBundle,
    TrajectoryReplayHandle,
    tensor_checksum,
)


QWEN3_DEEPSTACK_BRANCH_LAYERS = (8, 16, 24)
QWEN3_DEEPSTACK_BRANCH_COUNT = len(QWEN3_DEEPSTACK_BRANCH_LAYERS)


@dataclass(frozen=True, slots=True)
class PackedQwen3ImageItem:
    """One source-image or one call-specific D item accepted by vLLM."""

    kind: Literal[
        "source_image", "crop_image", "focused_d", "crop_focused_d"
    ]
    call_index: int | None
    positions: tuple[int, ...]
    image_embeds: torch.Tensor
    image_grid_thw: tuple[int, int, int]
    component_digests: tuple[str, ...]
    packed_tensor_sha256: str
    item_sha256: str

    def verify_integrity(self) -> None:
        if self.kind == "source_image" and self.call_index is not None:
            raise ReplayMismatchError("source image cannot have a tool call index")
        if self.kind in {"focused_d", "crop_image", "crop_focused_d"} and (
            self.call_index is None or self.call_index < 0
        ):
            raise ReplayMismatchError("focused D requires a non-negative call index")
        if self.image_embeds.ndim != 2:
            raise ReplayMismatchError("packed image item must have shape [N,4H]")
        if self.image_embeds.shape[0] != len(self.positions):
            raise ReplayMismatchError(
                "packed image rows and recorded visual positions differ"
            )
        if len(self.component_digests) != 1 + QWEN3_DEEPSTACK_BRANCH_COUNT:
            raise ReplayMismatchError(
                "packed Qwen3 item must identify main plus three DeepStack tensors"
            )
        actual = tensor_checksum(self.image_embeds)
        if actual != self.packed_tensor_sha256:
            raise ReplayMismatchError("packed image tensor checksum changed")
        if (
            _item_checksum(
                kind=self.kind,
                call_index=self.call_index,
                positions=self.positions,
                grid=self.image_grid_thw,
                component_digests=self.component_digests,
                packed_tensor_sha256=self.packed_tensor_sha256,
            )
            != self.item_sha256
        ):
            raise ReplayMismatchError("packed image item identity changed")


@dataclass(frozen=True, slots=True)
class PackedQwen3Replay:
    """Ordered vLLM input plus immutable replay identities.

    ``items`` is always source first, followed by one main-D item per tool call
    in contiguous call-index order.  Each item's feature columns are main,
    DeepStack layer 8, layer 16, then layer 24.
    """

    replay_handle: TrajectoryReplayHandle
    items: tuple[PackedQwen3ImageItem, ...]
    branch_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.branch_layers != QWEN3_DEEPSTACK_BRANCH_LAYERS:
            raise ReplayMismatchError(
                "vLLM Qwen3 packing requires DeepStack layers (8, 16, 24)"
            )
        if not self.items or self.items[0].kind != "source_image":
            raise ReplayMismatchError("packed replay must begin with the source image")
        calls = tuple(item.call_index for item in self.items[1:])
        if calls != tuple(range(len(calls))):
            raise ReplayMismatchError(
                "packed tool items must use contiguous call order after the source image"
            )
        for item in self.items:
            item.verify_integrity()

    @property
    def image_embeds(self) -> torch.Tensor:
        """Return a fresh concatenation in vLLM's flat multimodal field shape."""

        self.verify_integrity()
        return torch.cat(tuple(item.image_embeds for item in self.items), dim=0).clone()

    @property
    def image_grid_thw(self) -> torch.Tensor:
        self.verify_integrity()
        return torch.tensor(
            tuple(item.image_grid_thw for item in self.items), dtype=torch.long
        )

    @property
    def image_uuids(self) -> tuple[str, ...]:
        """Stable vLLM cache identities including layout and tensor content."""

        self.verify_integrity()
        return tuple(item.item_sha256 for item in self.items)

    def verify_integrity(self) -> None:
        for item in self.items:
            item.verify_integrity()

    def as_vllm_multi_modal_data(
        self,
    ) -> dict[str, list[dict[str, torch.Tensor]]]:
        """Build the public vLLM ``multi_modal_data`` value.

        Each image is a separate list item so vLLM's cache-disabled UUID
        builder observes the same item count as the custom parser. The parser
        coalesces the fields only after vLLM has established those identities.
        Callers may pass :attr:`image_uuids` as the corresponding image UUIDs
        when input caching is enabled.
        """

        self.verify_integrity()
        return {
            "image": [
                {
                    "image_embeds": item.image_embeds.clone(),
                    "image_grid_thw": torch.tensor(
                        (item.image_grid_thw,), dtype=torch.long
                    ),
                }
                for item in self.items
            ]
        }


def pack_qwen3_vllm_replay(
    store: ObservationStore,
    replay_handle: TrajectoryReplayHandle,
    *,
    expected_branch_layers: tuple[int, ...] = QWEN3_DEEPSTACK_BRANCH_LAYERS,
) -> PackedQwen3Replay:
    """Pack the exact recorded source and call-specific D tensors for vLLM.

    Reusing the source grid for D is exact because the TGVF Adapter preserves
    the source pre-merge spatial layout before applying the same frozen spatial
    merger.  A synthetic or malformed record whose D token count differs from
    that grid is rejected instead of inventing a grid.
    """

    if not isinstance(store, ObservationStore):
        raise TypeError("vLLM packing requires an ObservationStore")
    if not isinstance(replay_handle, TrajectoryReplayHandle):
        raise TypeError("vLLM packing requires a TrajectoryReplayHandle")
    expected_branch_layers = tuple(int(layer) for layer in expected_branch_layers)
    if expected_branch_layers != QWEN3_DEEPSTACK_BRANCH_LAYERS:
        raise ReplayMismatchError(
            "this Qwen3-vLLM bridge accepts exactly DeepStack layers (8, 16, 24)"
        )

    replay = store.resolve_replay(replay_handle)
    if replay.model.family != "qwen3_vl":
        raise ReplayMismatchError("Qwen3 vLLM packer received a different model family")
    records = tuple(
        store.resolve_record(handle) for handle in replay.observation_handles
    )
    source = replay.source_visual
    _validate_record_sequence(source, records, expected_branch_layers)

    grid = source.state.image_grid_thw
    merge_size = source.state.spatial_merge_size
    source_main = _resolve_features(
        store, source.state.merged_main, "source main"
    )
    source_branches = tuple(
        _resolve_features(store, ref, f"source DeepStack branch {index}")
        for index, ref in enumerate(source.state.merged_deepstack)
    )
    _validate_grid(grid, merge_size, source_main.shape[0])
    items = [
        _pack_item(
            kind="source_image",
            call_index=None,
            positions=source.positions,
            grid=grid,
            main=source_main,
            branches=source_branches,
            component_digests=(
                source.state.merged_main.address.digest,
                *(ref.address.digest for ref in source.state.merged_deepstack),
            ),
        )
    ]

    for record in records:
        if isinstance(record, FocusedObservationRecord):
            kind: Literal[
                "focused_d", "crop_image", "crop_focused_d"
            ] = "focused_d"
            item_grid = grid
            item_merge_size = merge_size
            positions = record.layout.d_positions
            main_ref = record.payload.main_d
            branch_refs = tuple(branch.d_tensor for branch in record.branches)
            label = "main D"
        elif isinstance(record, CropTGVFObservationRecord):
            kind = "crop_focused_d"
            item_grid = record.crop_visual.source.image_grid_thw
            item_merge_size = record.crop_visual.source.spatial_merge_size
            positions = record.layout.d_positions
            main_ref = record.payload.main_d
            branch_refs = tuple(branch.d_tensor for branch in record.branches)
            label = "crop-conditioned main D"
        elif isinstance(record, CropObservationRecord):
            kind = "crop_image"
            item_grid = record.crop_visual.image_grid_thw
            item_merge_size = record.crop_visual.spatial_merge_size
            positions = record.crop_visual.positions
            main_ref = record.crop_visual.merged_main
            branch_refs = record.crop_visual.merged_deepstack
            label = "crop image"
        else:  # pragma: no cover - ObservationStore owns the accepted union.
            raise TypeError("unknown Qwen3 observation type")
        main = _resolve_features(store, main_ref, f"call {record.call_index} {label}")
        branches = tuple(
            _resolve_features(store, ref, f"call {record.call_index} branch {index}")
            for index, ref in enumerate(branch_refs)
        )
        _validate_grid(item_grid, item_merge_size, main.shape[0])
        items.append(
            _pack_item(
                kind=kind,
                call_index=record.call_index,
                positions=positions,
                grid=item_grid,
                main=main,
                branches=branches,
                component_digests=(
                    main_ref.address.digest,
                    *(ref.address.digest for ref in branch_refs),
                ),
            )
        )

    return PackedQwen3Replay(
        replay_handle=replay_handle,
        items=tuple(items),
        branch_layers=expected_branch_layers,
    )


def pack_qwen3_vllm_replay_bundle(
    bundle: TrajectoryReplayBundle,
    *,
    expected_branch_layers: tuple[int, ...] = QWEN3_DEEPSTACK_BRANCH_LAYERS,
) -> PackedQwen3Replay:
    """Pack a transported worker-local bundle without observation recomputation."""

    store, replay_handle = ObservationStore.from_replay_bundle(bundle)
    return pack_qwen3_vllm_replay(
        store,
        replay_handle,
        expected_branch_layers=expected_branch_layers,
    )


def pack_qwen3_vllm_observation(
    store: ObservationStore,
    observation_handle: ObservationHandle,
    *,
    expected_branch_layers: tuple[int, ...] = QWEN3_DEEPSTACK_BRANCH_LAYERS,
) -> PackedQwen3ImageItem:
    """Pack one rollout-owned tool observation for the next live vLLM turn."""

    if not isinstance(store, ObservationStore):
        raise TypeError("vLLM observation packing requires an ObservationStore")
    if not isinstance(observation_handle, ObservationHandle):
        raise TypeError("vLLM observation packing requires an ObservationHandle")
    expected_branch_layers = tuple(expected_branch_layers)
    if expected_branch_layers != QWEN3_DEEPSTACK_BRANCH_LAYERS:
        raise ReplayMismatchError(
            "live Qwen3 observation packing requires DeepStack layers (8, 16, 24)"
        )
    record = store.resolve_record(observation_handle)
    if record.model.family != "qwen3_vl":
        raise ReplayMismatchError("Qwen3 observation packer received another family")
    if isinstance(record, FocusedObservationRecord):
        kind: Literal["crop_image", "focused_d", "crop_focused_d"] = "focused_d"
        grid = record.source_visual.image_grid_thw
        merge_size = record.source_visual.spatial_merge_size
        positions = record.layout.d_positions
        main_ref = record.payload.main_d
        branch_refs = tuple(branch.d_tensor for branch in record.branches)
        branch_layers = record.layout.deepstack_branch_layers
        label = "main D"
    elif isinstance(record, CropObservationRecord):
        kind = "crop_image"
        grid = record.crop_visual.image_grid_thw
        merge_size = record.crop_visual.spatial_merge_size
        positions = record.crop_visual.positions
        main_ref = record.crop_visual.merged_main
        branch_refs = record.crop_visual.merged_deepstack
        branch_layers = record.crop_visual.deepstack_branch_layers
        label = "crop image"
    elif isinstance(record, CropTGVFObservationRecord):
        kind = "crop_focused_d"
        grid = record.crop_visual.source.image_grid_thw
        merge_size = record.crop_visual.source.spatial_merge_size
        positions = record.layout.d_positions
        main_ref = record.payload.main_d
        branch_refs = tuple(branch.d_tensor for branch in record.branches)
        branch_layers = record.layout.deepstack_branch_layers
        label = "crop-conditioned main D"
    else:  # pragma: no cover - ObservationStore owns the accepted union.
        raise TypeError("unknown Qwen3 observation type")
    if branch_layers != expected_branch_layers:
        raise ReplayMismatchError(
            "live observation DeepStack branch order/layers differ from Qwen3"
        )
    main = _resolve_features(store, main_ref, f"call {record.call_index} {label}")
    branches = tuple(
        _resolve_features(store, ref, f"call {record.call_index} branch {index}")
        for index, ref in enumerate(branch_refs)
    )
    _validate_grid(grid, merge_size, main.shape[0])
    return _pack_item(
        kind=kind,
        call_index=record.call_index,
        positions=positions,
        grid=grid,
        main=main,
        branches=branches,
        component_digests=(
            main_ref.address.digest,
            *(ref.address.digest for ref in branch_refs),
        ),
    )


def _validate_record_sequence(
    source: TrajectorySourceVisual,
    records: tuple[
        FocusedObservationRecord | CropObservationRecord | CropTGVFObservationRecord,
        ...,
    ],
    expected_branch_layers: tuple[int, ...],
) -> None:
    if source.deepstack_branch_layers != expected_branch_layers:
        raise ReplayMismatchError(
            "recorded source DeepStack branch order/layers differ from Qwen3"
        )
    if len(source.state.merged_deepstack) != len(expected_branch_layers):
        raise ReplayMismatchError(
            "source image does not contain three DeepStack branches"
        )
    if any(
        injection_positions != source.positions
        for injection_positions in source.deepstack_injection_positions
    ):
        raise ReplayMismatchError(
            "vLLM source DeepStack positions must exactly equal the main visual positions"
        )
    source_identity = _source_state_identity(source.state)
    representation = None
    _validate_contiguous_positions(source.positions, "source image")
    previous_end = source.positions[-1]

    for expected_call, record in enumerate(records):
        if record.call_index != expected_call:
            raise ReplayMismatchError(
                "observation calls are not contiguous and ordered"
            )
        if isinstance(record, (FocusedObservationRecord, CropTGVFObservationRecord)):
            if representation is None:
                representation = record.representation
            elif record.representation != representation:
                raise ReplayMismatchError(
                    "representation identity changed within replay"
                )
        if _source_state_identity(record.source_visual) != source_identity or (
            _original_image_positions(record) != source.positions
        ):
            raise ReplayMismatchError(
                "recorded source visual state changed across calls"
            )
        if isinstance(record, (FocusedObservationRecord, CropTGVFObservationRecord)):
            branch_layers = record.layout.deepstack_branch_layers
            branch_record_layers = tuple(branch.layer for branch in record.branches)
            positions = record.layout.d_positions
            branch_positions = tuple(
                branch.injection_positions for branch in record.branches
            )
            label = "D"
        else:
            branch_layers = record.crop_visual.deepstack_branch_layers
            branch_record_layers = branch_layers
            positions = record.crop_visual.positions
            branch_positions = record.crop_visual.deepstack_injection_positions
            label = "crop"
        if branch_layers != expected_branch_layers:
            raise ReplayMismatchError(
                f"recorded {label} DeepStack branch order/layers differ from Qwen3"
            )
        if branch_record_layers != expected_branch_layers:
            raise ReplayMismatchError(
                f"{label} DeepStack branch records are out of order"
            )
        if any(
            injection_positions != positions for injection_positions in branch_positions
        ):
            raise ReplayMismatchError(
                "vLLM DeepStack positions must exactly equal the main visual positions"
            )
        _validate_contiguous_positions(positions, f"call {record.call_index} {label}")
        if positions[0] <= previous_end:
            raise ReplayMismatchError(
                "vLLM visual items must occur in source-then-call prompt order"
            )
        previous_end = positions[-1]


def _resolve_features(
    store: ObservationStore, ref: TensorArtifactRef, name: str
) -> torch.Tensor:
    # ``resolve_verified`` performs the address, descriptor, shape, and dtype
    # digest checks before this transport-specific validation.
    tensor = store.resolve_verified(ref)
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ReplayMismatchError(f"{name} must describe one vLLM request")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ReplayMismatchError(f"{name} must be a floating tensor [N,H]")
    return tensor.contiguous()


def _pack_item(
    *,
    kind: Literal[
        "source_image", "crop_image", "focused_d", "crop_focused_d"
    ],
    call_index: int | None,
    positions: tuple[int, ...],
    grid: tuple[int, int, int],
    main: torch.Tensor,
    branches: tuple[torch.Tensor, ...],
    component_digests: tuple[str, ...],
) -> PackedQwen3ImageItem:
    if len(branches) != QWEN3_DEEPSTACK_BRANCH_COUNT:
        raise ReplayMismatchError("Qwen3 packing requires exactly three branches")
    if any(
        branch.shape != main.shape
        or branch.dtype != main.dtype
        or branch.device != main.device
        for branch in branches
    ):
        raise ReplayMismatchError(
            "main and all DeepStack embeddings must share exact shape/dtype/device"
        )
    packed = torch.cat((main, *branches), dim=-1).contiguous().clone()
    packed_digest = tensor_checksum(packed)
    item_digest = _item_checksum(
        kind=kind,
        call_index=call_index,
        positions=positions,
        grid=grid,
        component_digests=component_digests,
        packed_tensor_sha256=packed_digest,
    )
    return PackedQwen3ImageItem(
        kind=kind,
        call_index=call_index,
        positions=positions,
        image_embeds=packed,
        image_grid_thw=grid,
        component_digests=component_digests,
        packed_tensor_sha256=packed_digest,
        item_sha256=item_digest,
    )


def _validate_grid(
    grid: tuple[int, int, int], spatial_merge_size: int, feature_count: int
) -> None:
    if len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid):
        raise ReplayMismatchError("image_grid_thw must contain three positive integers")
    temporal, height, width = grid
    if temporal != 1:
        raise ReplayMismatchError("precomputed source/D items must use an image grid")
    if spatial_merge_size <= 0 or (
        height % spatial_merge_size or width % spatial_merge_size
    ):
        raise ReplayMismatchError("image grid is not divisible by spatial merge size")
    expected = temporal * height * width // (spatial_merge_size**2)
    if expected != feature_count:
        raise ReplayMismatchError(
            "recorded feature count does not match the exact merged source grid"
        )


def _validate_contiguous_positions(positions: tuple[int, ...], name: str) -> None:
    if not positions:
        raise ReplayMismatchError(f"{name} positions must not be empty")
    if positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ReplayMismatchError(
            f"{name} positions must form one contiguous vLLM placeholder"
        )


def _item_checksum(
    *,
    kind: str,
    call_index: int | None,
    positions: tuple[int, ...],
    grid: tuple[int, int, int],
    component_digests: tuple[str, ...],
    packed_tensor_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "call_index": call_index,
            "component_digests": component_digests,
            "grid": grid,
            "kind": kind,
            "packed_tensor_sha256": packed_tensor_sha256,
            "positions": positions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_state_identity(source: SourceVisualState) -> tuple[object, ...]:
    return (
        source.image_sha256,
        source.decoded_rgb_sha256,
        source.premerge_main.address.digest,
        tuple(ref.address.digest for ref in source.premerge_deepstack),
        source.merged_main.address.digest,
        tuple(ref.address.digest for ref in source.merged_deepstack),
        source.image_grid_thw,
        source.spatial_merge_size,
    )


def _original_image_positions(
    record: FocusedObservationRecord | CropObservationRecord | CropTGVFObservationRecord,
) -> tuple[int, ...]:
    if isinstance(record, (FocusedObservationRecord, CropTGVFObservationRecord)):
        return record.layout.original_image_positions
    return record.original_image_positions


__all__ = [
    "QWEN3_DEEPSTACK_BRANCH_COUNT",
    "QWEN3_DEEPSTACK_BRANCH_LAYERS",
    "PackedQwen3ImageItem",
    "PackedQwen3Replay",
    "pack_qwen3_vllm_replay",
    "pack_qwen3_vllm_replay_bundle",
    "pack_qwen3_vllm_observation",
]
