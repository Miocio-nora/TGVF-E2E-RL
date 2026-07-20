#!/usr/bin/env python3
"""Materialize one immutable Adapter-only export from a recorded DCP checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.representation.training.distributed_checkpoint import (
    RankZeroAdapterOwnedStateExport,
    RankZeroAdapterOwnedStateManifest,
    _tensor_checksum,
    load_distributed_representation_checkpoint_metadata,
    load_rank_zero_adapter_owned_state_export,
    save_rank_zero_adapter_owned_state_export_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shape-template-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = load_distributed_representation_checkpoint_metadata(args.checkpoint)
    template = load_rank_zero_adapter_owned_state_export(args.shape_template_export)
    checkpoint_manifest = metadata.manifest
    if checkpoint_manifest.run_identity != template.manifest.run_identity:
        raise ValueError("checkpoint and shape template identify different runs")
    if checkpoint_manifest.run_identity_sha256 != template.manifest.run_identity_sha256:
        raise ValueError("checkpoint and shape template run digests differ")
    if checkpoint_manifest.owned_state_names != template.manifest.tensor_names:
        raise ValueError("checkpoint and shape template tensor names differ")
    assert template.state is not None
    state = {name: torch.empty_like(tensor) for name, tensor in template.state.items()}
    dcp.load({"adapter": state}, checkpoint_id=args.checkpoint / "dcp")
    names = tuple(sorted(state))
    manifest = RankZeroAdapterOwnedStateManifest(
        run_identity=checkpoint_manifest.run_identity,
        run_identity_sha256=checkpoint_manifest.run_identity_sha256,
        global_step=checkpoint_manifest.global_step,
        tensor_names=names,
        tensor_shapes=tuple(tuple(state[name].shape) for name in names),
        tensor_dtypes=tuple(str(state[name].dtype) for name in names),
        tensor_sha256=tuple(_tensor_checksum(state[name]) for name in names),
    )
    export = RankZeroAdapterOwnedStateExport(manifest=manifest, state=state)
    save_rank_zero_adapter_owned_state_export_atomic(args.output, export)
    print(f"global_step={manifest.global_step}")
    print(f"manifest_sha256={state_digest(manifest)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
