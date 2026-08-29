#!/usr/bin/env python3
"""Rebind the immutable PRL-26-B S32 HF tree without rerunning the FSDP merge."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.policy_full_model_snapshot import (  # noqa: E402
    FullModelCheckpointOwner,
    FullModelMaterializationMode,
    load_full_model_evaluation_snapshot,
    write_full_model_materialization_receipt,
    write_full_model_snapshot_manifest,
)


MAIN_ROOT = Path("/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl")
OWNER_ROOT = MAIN_ROOT / (
    "artifacts/policy/PRL-26-B-train512-s32-parity-crop-qwen3-instruct-"
    "bs16-n16-teacher25-ws8"
)
OLD_EVAL_ROOT = OWNER_ROOT / (
    "evaluation/PRL26-B-TRAIN512-S32-CROP-MATCHED-COREDEV2511-"
    "PIXEL512-BOUNDARYFIX-V1"
)
OLD_MANIFEST = OLD_EVAL_ROOT / "step32/runtime/full-model-snapshot.json"
OLD_RECEIPT = OLD_EVAL_ROOT / "step32/runtime/full-model-materialization.json"
OLD_MANIFEST_SHA256 = (
    "056d63519b845a42f68ead8b7f5986afc8e7b0895e3e9056e6735783e798ec32"
)
OLD_RECEIPT_SHA256 = (
    "78afcdb9822250ddb2ba8f5842f64cdc07ca31ed604a5ef968c7b801b16f8f6a"
)
EXPECTED_MODEL_TREE_SHA256 = (
    "c896acc40694af3c5b6518176a0275264c02f50e42852cdb0241f2533b5be13a"
)
EVALUATION_ID = (
    "PRL26-B-S32-OWNER-GENERIC86-TRAINING-RUN-COREDEV2511-PIXEL512-V1"
)
NEW_EVAL_ROOT = OWNER_ROOT / "evaluation" / EVALUATION_ID
OWNER_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_26_b_qwen3_instruct_full_crop_train512_parity_s32_bs16_n16_"
    "teacher25_ws8.toml"
)
MATERIALIZATION_CONFIG = REPOSITORY_ROOT / (
    "configs/policy/runs/"
    "prl_24_d_base_qwen3_instruct_full_crop_teacher25_native_prl13.toml"
)
COMPLETION = OWNER_ROOT / (
    "permanent-checkpoints/global_step_32/tgvf_permanent_checkpoint_receipt.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable materialization-reuse proof differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise RuntimeError(
                    f"immutable materialization-reuse proof differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def reuse(*, manifest_output: Path, receipt_output: Path, proof_output: Path) -> dict[str, object]:
    if (
        _sha256(OLD_MANIFEST) != OLD_MANIFEST_SHA256
        or _sha256(OLD_RECEIPT) != OLD_RECEIPT_SHA256
    ):
        raise RuntimeError("historical PRL-26-B materialization records differ")
    old = load_full_model_evaluation_snapshot(
        OLD_MANIFEST,
        OLD_RECEIPT,
        runtime_lightweight=True,
    )
    old_model_path = Path(old.receipt.model_path)
    if (
        old.policy_version.optimizer_step != 32
        or old.receipt.mode is not FullModelMaterializationMode.VERL_FSDP_MERGE
        or old.receipt.model_tree_sha256 != EXPECTED_MODEL_TREE_SHA256
        or old_model_path != (OLD_EVAL_ROOT / "step32/runtime/full-model-hf")
        or old_model_path.is_symlink()
        or not old_model_path.is_dir()
    ):
        raise RuntimeError("historical PRL-26-B HF materialization differs")

    owner = FullModelCheckpointOwner(
        run_id=old.policy_version.run_id,
        run_identity_sha256=old.run_identity_sha256,
        config_path=str(OWNER_CONFIG.resolve()),
        config_file_sha256=_sha256(OWNER_CONFIG),
        completion_path=str(COMPLETION.resolve()),
        completion_file_sha256=_sha256(COMPLETION),
    )
    manifest = replace(
        old.manifest,
        run_contract_path=str(MATERIALIZATION_CONFIG.resolve()),
        checkpoint_owner=owner,
    )
    receipt = replace(
        old.receipt,
        snapshot_identity_sha256=manifest.identity_sha256,
    )
    write_full_model_snapshot_manifest(manifest_output, manifest)
    write_full_model_materialization_receipt(receipt_output, receipt)
    rebound = load_full_model_evaluation_snapshot(
        manifest_output,
        receipt_output,
        runtime_lightweight=True,
    )
    if (
        rebound.manifest != manifest
        or rebound.receipt != receipt
        or Path(rebound.receipt.model_path) != old_model_path
        or rebound.receipt.model_tree_sha256 != EXPECTED_MODEL_TREE_SHA256
    ):
        raise RuntimeError("rebound PRL-26-B full-model closure differs")
    content: dict[str, object] = {
        "schema_version": "tgvf.prl26-b-full-model-materialization-reuse.v1",
        "status": "pass",
        "evaluation_id": EVALUATION_ID,
        "reuse_mode": "read_only_existing_hf_tree_no_merge",
        "source_manifest_path": str(OLD_MANIFEST),
        "source_manifest_sha256": _sha256(OLD_MANIFEST),
        "source_receipt_path": str(OLD_RECEIPT),
        "source_receipt_sha256": _sha256(OLD_RECEIPT),
        "source_model_path": str(old_model_path),
        "source_model_tree_sha256": old.receipt.model_tree_sha256,
        "rebound_manifest_path": str(manifest_output.resolve()),
        "rebound_manifest_sha256": _sha256(manifest_output),
        "rebound_manifest_identity_sha256": manifest.identity_sha256,
        "rebound_receipt_path": str(receipt_output.resolve()),
        "rebound_receipt_sha256": _sha256(receipt_output),
        "rebound_receipt_identity_sha256": receipt.identity_sha256,
        "model_bytes_copied": False,
        "merge_command_executed": False,
    }
    payload = {**content, "identity_sha256": _canonical_sha256(content)}
    _write_immutable_json(proof_output.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=NEW_EVAL_ROOT / "step32/runtime/full-model-snapshot.json",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=NEW_EVAL_ROOT / "step32/runtime/full-model-materialization.json",
    )
    parser.add_argument(
        "--proof-output",
        type=Path,
        default=NEW_EVAL_ROOT / "runtime/full-model-materialization-reuse.json",
    )
    args = parser.parse_args()
    payload = reuse(
        manifest_output=args.manifest_output.resolve(),
        receipt_output=args.receipt_output.resolve(),
        proof_output=args.proof_output.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
