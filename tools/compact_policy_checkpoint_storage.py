#!/usr/bin/env python3
"""Compact retired policy FSDP checkpoints into evaluation bundles.

This is an intentionally narrow migration for the 2026-08-24 policy-storage
audit.  The allowlists below are the destructive boundary.  In particular,
every PRL25-B/PRL25-C artifact is rejected even if it is passed explicitly.

Full-Qwen checkpoints become BF16 Hugging Face models.  A checkpoint that also
contains the project-owned ``tgvf_adapter.`` namespace keeps the matching,
content-addressed TGVF runtime snapshot beside the Qwen model.  Qwen-LoRA
checkpoints are detected and refused here: they must remain adapter bundles
bound to a shared immutable base model, not be expanded into full models.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_mode_quarantined,
)


WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACTS = WORKSPACE / "artifacts/policy"
BASE_MODEL = Path("/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct")
SCHEMA = "tgvf.policy-checkpoint-compaction.v1"
POST_VALIDATION_SCHEMA = "tgvf.policy-checkpoint-post-validation.v1"
DELETION_SCHEMA = "tgvf.non-scientific-checkpoint-deletion.v1"
EXCLUDED_FAMILIES = ("PRL-25-B", "PRL-25-C")
ADAPTER_PREFIX = "tgvf_adapter."


# Every entry is a retired formal/diagnostic full checkpoint that the storage
# audit classified as scientifically retainable.  Keep this explicit: adding a
# new experiment must be a reviewed source change, never an automatic glob.
FORMAL: tuple[tuple[str, tuple[int, ...]], ...] = (
    (
        "PRL-13-A-qwen3-instruct-grpo-bs256-n16-native-crop-t1-stratified-80step-gpu0123",
        (8,),
    ),
    (
        "PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-cleanfinal-16step-ws8",
        (5, 6, 8, 16),
    ),
    ("PRL-15-R1-qwen3-instruct-full-rp66-bs16-n16-crop16-math-equiv-ws4", (7, 8)),
    (
        "PRL-16-F0-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-matched-8step-ws8",
        (7, 8),
    ),
    (
        "PRL-16-F1-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-exact-matched-8step-ws8",
        (1, 2),
    ),
    (
        "PRL-16-F2-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-lifecycle-fix-8step-ws8",
        (4, 5, 6, 7, 8),
    ),
    (
        "PRL-17-R0-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-shaped-novisual-8step-ws8",
        (4, 5, 6, 7, 8),
    ),
    (
        "PRL-17-R1-qwen3-instruct-full-frozen-rp67-bs16-n16-t1-shaped-novisual-8step-ws8",
        (4, 5, 6, 7, 8),
    ),
    (
        "PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8",
        (4, 5, 6, 8, 15, 16),
    ),
    (
        "PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8",
        (4, 5, 6, 8, 15, 16),
    ),
    ("PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8", (2, 3, 8, 16)),
    (
        "PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-22-B-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-23-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher50-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-23-B-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher100-8step-ws8",
        (8, 15, 16),
    ),
    (
        "PRL-24-A-FMT2-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8",
        (4, 8, 12, 15, 16),
    ),
    (
        "PRL-24-A-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-16step-ws8",
        (2, 4, 8, 16, 24, 31, 32),
    ),
    (
        "PRL-24-B-FMT2-JOINT-qwen3-instruct-full-joint-rp67-bs64-n16-tfree-teacher25-8step-ws8",
        (4, 7, 8),
    ),
    (
        "PRL-24-C-FMT2-FG-qwen3-instruct-full-frozen-rp67-bs64-n16-tfree-teacher25-8step-ws8",
        (4, 8, 12, 15, 16),
    ),
    (
        "PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8-sp1",
        (1,),
    ),
)


# Only checkpoint payloads are deleted; logs and other experiment evidence in
# their parents stay in place.  PRL25-B/C candidates are deliberately absent.
NON_SCIENTIFIC_CHECKPOINTS: tuple[str, ...] = (
    "PRL-13-A-qwen3-instruct-grpo-bs256-n16-native-crop-t1-stratified-80step-gpu0123/smoke/checkpoints/global_step_1",
    "PRL-15-R0-qwen3-instruct-full-rp66-bs16-n16-t1-matched-8step-gpu0123/smoke/actor-rollout-only-v1/checkpoints/global_step_1",
    "PRL-15-R0-qwen3-instruct-full-rp66-bs16-n16-t1-matched-8step-gpu0123/smoke/checkpoints/global_step_1",
    "PRL-15-R1-qwen3-instruct-full-rp66-bs16-n16-crop16-math-equiv-ws4/smoke/matched-horizon-50d5ddb/checkpoints/global_step_1",
    "PRL-16-F1-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-exact-matched-8step-ws8/smoke/exact-offload-false-v1/checkpoints/global_step_1",
    "PRL-17-R0-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-shaped-novisual-8step-ws8/smoke/reward-switch-v1/checkpoints/global_step_1",
    "PRL-17-R0-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-shaped-novisual-8step-ws8/smoke/reward-switch-v4/checkpoints/global_step_1",
    "PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8/smoke/joint-rp67-fullstep-v1/checkpoints/global_step_1",
    "PRL-19-R0-C0-qwen3-instruct-full-frozen-rp67-bs4-n2-tfree-visual-api-canary-ws4/canary/checkpoints/global_step_1",
    "PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-8step-ws8/smoke/frozen-rp67-tfree-visual-api-fullstep-v1/checkpoints/global_step_1",
    "PRL-20-R0-C0-qwen3-instruct-full-frozen-rp67-bs4-n2-tfree-crop-tgvf-canary-ws4/canary/checkpoints/global_step_1",
    "PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8/smoke-integration/checkpoints/global_step_1",
    "PRL-24-A-C0-qwen3-instruct-full-frozen-rp67-bs4-n2-tfree-teacher25-1step-ws4/canary/checkpoints/global_step_1",
    "PRL-24-B-FMT2-JOINT-C0-qwen3-instruct-full-joint-rp67-bs4-n2-tfree-teacher25-1step-ws4/canary/checkpoints/global_step_1",
    "PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8/smoke-integration/checkpoints/global_step_1",
    "PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8-sp1/smoke-integration/checkpoints/global_step_1",
    "PRL-24-D-FMT2-qwen3-instruct-full-crop-bs64-n16-tfree-teacher25-16step-ws8-sp1-ca1/smoke-integration/checkpoints/global_step_1",
)

FAILED_TREE = (
    "PRL-20-R0-C0-qwen3-instruct-full-frozen-rp67-bs4-n2-tfree-crop-tgvf-canary-ws4-"
    "failed-fsdp-collective-mismatch-20260813T234356"
)


@dataclass(frozen=True)
class Candidate:
    run: Path
    step: int
    source: Path
    aliases: tuple[Path, ...]
    kind: str
    qwen_key_count: int
    tgvf_key_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_excluded(path: Path) -> None:
    text = str(path)
    if any(name in text for name in EXCLUDED_FAMILIES):
        raise ValueError(f"excluded PRL25 family path: {path}")


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _base_keys() -> frozenset[str]:
    index = _json(BASE_MODEL / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("base model safetensors index lacks weight_map")
    return frozenset(str(key) for key in weight_map)


def _rank_zero(checkpoint: Path) -> Path:
    files = sorted((checkpoint / "actor").glob("model_world_size_*_rank_0.pt"))
    if len(files) != 1:
        raise RuntimeError(f"expected one rank-zero model shard in {checkpoint}")
    return files[0]


def _source_dirs(run: Path, step: int) -> tuple[Path, ...]:
    matches: list[Path] = []
    for path in run.rglob(f"global_step_{step}"):
        if not path.is_dir() or path.is_symlink():
            continue
        parts = set(path.relative_to(run).parts)
        if parts.intersection(
            {
                "evaluation",
                "compact-checkpoints",
                "smoke",
                "smoke-integration",
                "canary",
            }
        ):
            continue
        try:
            _rank_zero(path)
        except RuntimeError:
            continue
        matches.append(path.resolve(strict=True))
    if not matches:
        raise FileNotFoundError(f"no formal full checkpoint for {run.name} S{step}")
    by_inode: dict[tuple[int, int], list[Path]] = {}
    for path in matches:
        stat = _rank_zero(path).stat()
        by_inode.setdefault((stat.st_dev, stat.st_ino), []).append(path)
    if len(by_inode) != 1:
        raise RuntimeError(
            f"multiple independent formal checkpoints for {run.name} S{step}"
        )
    return tuple(sorted(next(iter(by_inode.values())), key=str))


def _classify(checkpoint: Path) -> tuple[str, int, int]:
    import torch

    state = torch.load(
        _rank_zero(checkpoint), map_location="cpu", mmap=True, weights_only=False
    )
    if not isinstance(state, Mapping):
        raise TypeError(f"rank-zero state is not a mapping: {checkpoint}")
    keys = frozenset(str(key) for key in state)
    del state
    tgvf = frozenset(key for key in keys if key.startswith(ADAPTER_PREFIX))
    qwen = keys - tgvf
    if qwen == _base_keys():
        return ("full-qwen+tgvf" if tgvf else "full-qwen", len(qwen), len(tgvf))
    qwen_lora = frozenset(
        key for key in qwen if "lora_" in key.lower() or ".lora." in key.lower()
    )
    if qwen_lora:
        return ("qwen-lora", len(qwen), len(tgvf))
    return ("unknown", len(qwen), len(tgvf))


def candidates() -> tuple[Candidate, ...]:
    result: list[Candidate] = []
    for run_name, steps in FORMAL:
        run = (ARTIFACTS / run_name).resolve(strict=True)
        _reject_excluded(run)
        run_classification: tuple[str, int, int] | None = None
        for step in steps:
            compact = run / "compact-checkpoints" / f"global_step_{step}"
            if compact.is_dir() and (compact / "compaction-receipt.json").is_file():
                receipt = _json(compact / "compaction-receipt.json")
                source = Path(str(receipt["source_path"]))
                aliases = tuple(Path(str(path)) for path in receipt["source_aliases"])
                kind = str(receipt["checkpoint_kind"])
                qwen_count = int(receipt["qwen_parameter_key_count"])
                tgvf_count = int(receipt["tgvf_parameter_key_count"])
            else:
                aliases = _source_dirs(run, step)
                if run_classification is None:
                    run_classification = _classify(aliases[0])
                kind, qwen_count, tgvf_count = run_classification
                source = aliases[0]
            result.append(
                Candidate(run, step, source, aliases, kind, qwen_count, tgvf_count)
            )
    if len(result) != 81:
        raise RuntimeError(f"formal allowlist resolved to {len(result)}, expected 81")
    return tuple(result)


def _tree_stats(root: Path) -> dict[str, int]:
    files = [
        path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    ]
    return {
        "file_count": len(files),
        "apparent_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
    }


def _file_records(
    root: Path, *, hash_files: bool = True
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in retained bundle: {path}")
        if not path.is_file():
            continue
        record: dict[str, object] = {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if hash_files:
            record["sha256"] = _sha256(path)
        records.append(record)
    return tuple(records)


def _source_model_records(checkpoint: Path) -> tuple[dict[str, object], ...]:
    actor = checkpoint / "actor"
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no model shards: {checkpoint}")
    return tuple(
        {
            "relative_path": path.relative_to(checkpoint).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in shards
    )


def _validate_hf_model(model: Path) -> tuple[dict[str, object], ...]:
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor

    index = _json(model / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("materialized model lacks a safetensors weight_map")
    keys = frozenset(str(key) for key in weight_map)
    if keys != _base_keys():
        raise ValueError("materialized Qwen closure differs from bound base model")
    actual: set[str] = set()
    dtypes: set[str] = set()
    for shard_name in sorted(set(str(name) for name in weight_map.values())):
        shard = model / shard_name
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                actual.add(key)
                dtypes.add(str(handle.get_slice(key).get_dtype()))
    if actual != set(keys):
        raise ValueError("safetensors shard closure differs from its index")
    if dtypes != {"BF16"}:
        raise ValueError(f"compact model is not uniformly BF16: {sorted(dtypes)}")
    AutoConfig.from_pretrained(model, local_files_only=True)
    AutoProcessor.from_pretrained(model, local_files_only=True)
    return _file_records(model)


def _cpu_generation_smoke(model: Path) -> dict[str, object]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    policy = AutoModelForImageTextToText.from_pretrained(
        model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Reply with exactly OK"}]}
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], return_tensors="pt")
    generated: list[list[int]] = []
    decoded: list[str] = []
    with torch.inference_mode():
        for _ in range(2):
            output = policy.generate(**inputs, max_new_tokens=2, do_sample=False)
            new = output[:, inputs["input_ids"].shape[1] :]
            generated.append(new.tolist()[0])
            decoded.append(processor.batch_decode(new, skip_special_tokens=True)[0])
    if not generated[0] or generated[0] != generated[1]:
        raise RuntimeError(
            f"generation smoke is empty or non-deterministic: {generated!r}"
        )
    return {
        "device": "cpu",
        "prompt": "Reply with exactly OK",
        "generated_ids": generated[0],
        "decoded": decoded[0],
        "repeat_identical": True,
    }


def _copy_small_metadata(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(source)
        name = path.name
        if name.startswith(
            ("model_world_size_", "optim_world_size_", "extra_state_world_size_")
        ):
            continue
        if rel.parts[:2] == ("actor", "huggingface"):
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _adapter_snapshot(candidate: Candidate, target: Path) -> dict[str, object]:
    project_state_path = candidate.source / "actor/tgvf_policy_project_state.json"
    if not project_state_path.is_file():
        raise FileNotFoundError(
            f"TGVF checkpoint lacks project state: {candidate.source}"
        )
    project_state = _json(project_state_path)
    policy_version = project_state.get("policy_version")
    if not isinstance(policy_version, Mapping):
        raise TypeError("project state lacks policy_version")
    weights_sha = policy_version.get("weights_sha256")
    if not isinstance(weights_sha, str):
        raise TypeError("project state lacks policy weights identity")
    manifest_dir = candidate.run / "runtime-policy-state/lora-manifests"
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    for path in manifest_dir.glob(f"step-{candidate.step:08d}-*.json"):
        manifest = _json(path)
        if (
            manifest.get("optimizer_step") == candidate.step
            and manifest.get("weights_sha256") == weights_sha
        ):
            matches.append((path, manifest))
    if not matches:
        raise FileNotFoundError(
            f"no bound TGVF adapter manifest for {candidate.run.name} S{candidate.step}"
        )
    tensor_files = {str(manifest.get("tensor_file")) for _, manifest in matches}
    tensor_hashes = {str(manifest.get("tensor_file_sha256")) for _, manifest in matches}
    if len(tensor_files) != 1 or len(tensor_hashes) != 1:
        raise RuntimeError("matching adapter manifests disagree")
    manifest_path, manifest = sorted(matches, key=lambda item: str(item[0]))[0]
    tensor_path = candidate.run / "runtime-policy-state" / next(iter(tensor_files))
    if _sha256(tensor_path) != next(iter(tensor_hashes)):
        raise ValueError("TGVF adapter tensor digest differs from manifest")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, target / "adapter-manifest.json")
    os.link(tensor_path, target / "adapter.safetensors")
    return {
        "weights_sha256": weights_sha,
        "manifest_source": str(manifest_path),
        "manifest_sha256": _sha256(target / "adapter-manifest.json"),
        "tensor_source": str(tensor_path),
        "tensor_sha256": _sha256(target / "adapter.safetensors"),
        "tensor_bytes": (target / "adapter.safetensors").stat().st_size,
        "parameter_key_count": candidate.tgvf_key_count,
    }


def _find_existing_bundle(candidate: Candidate) -> Path | None:
    receipts = sorted(
        candidate.run.glob(
            "evaluation/**/qwen-only-bundle/materialization-receipt.json"
        )
    )
    valid: list[Path] = []
    for receipt_path in receipts:
        receipt = _json(receipt_path)
        if receipt.get("optimizer_step") != candidate.step:
            continue
        checkpoint_text = receipt.get("checkpoint_path")
        if (
            not isinstance(checkpoint_text, str)
            or candidate.run not in Path(checkpoint_text).parents
        ):
            continue
        valid.append(receipt_path.parent)
    return valid[0] if valid else None


def _merge_full_qwen(candidate: Candidate, target: Path) -> None:
    import torch
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    expected_qwen = _base_keys()

    class FullQwenMerger(FSDPModelMerger):
        def save_hf_model_and_tokenizer(
            self, state_dict: dict[str, torch.Tensor]
        ) -> None:
            adapter_keys = {key for key in state_dict if key.startswith(ADAPTER_PREFIX)}
            qwen_keys = set(state_dict) - adapter_keys
            if qwen_keys != set(expected_qwen):
                raise ValueError(
                    "merged Qwen state differs from base parameter closure"
                )
            if len(adapter_keys) != candidate.tgvf_key_count:
                raise ValueError(
                    "merged TGVF adapter closure differs from rank-zero audit"
                )
            for key in adapter_keys:
                del state_dict[key]
            super().save_hf_model_and_tokenizer(state_dict)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(target),
        local_dir=str(candidate.source / "actor"),
        hf_model_config_path=str(candidate.source / "actor/huggingface"),
        use_cpu_initialization=True,
    )
    merger = FullQwenMerger(config)
    try:
        merger.merge_and_save()
    finally:
        merger.cleanup()


def _promote_or_merge(candidate: Candidate, model: Path) -> tuple[str, str | None]:
    partial = (
        candidate.run
        / "compact-checkpoints"
        / f".global_step_{candidate.step}.full-hf.partial"
    )
    if partial.is_dir() and candidate.kind == "full-qwen":
        os.rename(partial, model)
        return "adopted-tested-upstream-merge", None
    existing = _find_existing_bundle(candidate)
    if existing is not None:
        shutil.copytree(existing / "model", model, copy_function=os.link)
        shutil.copy2(
            existing / "materialization-receipt.json",
            model.parent / "upstream-materialization-receipt.json",
        )
        return "hardlink-promoted-evaluation-bundle", str(existing)
    _merge_full_qwen(candidate, model)
    return "fresh-fsdp-merge", None


def compact_one(candidate: Candidate, *, prune_source: bool) -> dict[str, object]:
    if candidate.kind not in {"full-qwen", "full-qwen+tgvf"}:
        raise RuntimeError(
            f"refusing unsupported {candidate.kind}: {candidate.run.name} S{candidate.step}"
        )
    if prune_source and (owner := _process_is_using(candidate.aliases)) is not None:
        raise RuntimeError(f"refusing to prune an in-use checkpoint: {owner}")
    bundle = candidate.run / "compact-checkpoints" / f"global_step_{candidate.step}"
    _reject_excluded(bundle)
    if bundle.exists():
        receipt_path = bundle / "compaction-receipt.json"
        receipt = _json(receipt_path)
        model_files = _validate_hf_model(bundle / "model")
        model_tree_sha = _canonical_sha256(model_files)
        if model_tree_sha != receipt.get("model_tree_sha256"):
            raise RuntimeError("existing compact model tree differs from its receipt")
        smoke = receipt.get("generation_smoke")
        if not isinstance(smoke, Mapping) or smoke.get("repeat_identical") is not True:
            post_validation_path = bundle / "post-compaction-validation.json"
            expected = {
                "schema_version": POST_VALIDATION_SCHEMA,
                "compaction_receipt_sha256": _sha256(receipt_path),
                "model_tree_sha256": model_tree_sha,
            }
            if post_validation_path.exists():
                observed = _json(post_validation_path)
                if any(observed.get(key) != value for key, value in expected.items()):
                    raise RuntimeError("post-compaction validation identity differs")
                repeated = observed.get("generation_smoke")
                if (
                    not isinstance(repeated, Mapping)
                    or repeated.get("repeat_identical") is not True
                ):
                    raise RuntimeError("post-compaction deterministic smoke is absent")
            else:
                repeated = _cpu_generation_smoke(bundle / "model")
                _write_json(
                    post_validation_path,
                    {
                        **expected,
                        "validated_at_utc": _utc_now(),
                        "generation_smoke": repeated,
                    },
                )
        if prune_source:
            for alias in candidate.aliases:
                _reject_excluded(alias)
                if alias.exists():
                    shutil.rmtree(alias)
        return dict(receipt)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".global_step_{candidate.step}.",
            suffix=".compacting",
            dir=bundle.parent,
        )
    )
    try:
        source_stats = _tree_stats(candidate.source)
        source_model_files = _source_model_records(candidate.source)
        source_model_sha = _canonical_sha256(source_model_files)
        model = temporary / "model"
        method, promoted_from = _promote_or_merge(candidate, model)
        model_files = _validate_hf_model(model)
        model_sha = _canonical_sha256(model_files)
        protocol: dict[str, object] | None = None
        if candidate.tgvf_key_count:
            protocol = _adapter_snapshot(candidate, temporary / "protocol-state")
        _copy_small_metadata(candidate.source, temporary / "source-metadata")
        smoke = _cpu_generation_smoke(model)
        receipt: dict[str, object] = {
            "schema_version": SCHEMA,
            "created_at_utc": _utc_now(),
            "run": candidate.run.name,
            "optimizer_step": candidate.step,
            "checkpoint_kind": candidate.kind,
            "source_path": str(candidate.source),
            "source_aliases": [str(path) for path in candidate.aliases],
            "source_stats": source_stats,
            "source_model_files": list(source_model_files),
            "source_model_tree_sha256": source_model_sha,
            "qwen_parameter_key_count": candidate.qwen_key_count,
            "tgvf_parameter_key_count": candidate.tgvf_key_count,
            "materialization_method": method,
            "promoted_from": promoted_from,
            "model_files": list(model_files),
            "model_tree_sha256": model_sha,
            "model_bytes": sum(int(record["size_bytes"]) for record in model_files),
            "protocol_state": protocol,
            "generation_smoke": smoke,
            "source_prune_requested": prune_source,
        }
        _write_json(temporary / "compaction-receipt.json", receipt)
        os.rename(temporary, bundle)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if prune_source:
        for alias in candidate.aliases:
            _reject_excluded(alias)
            if alias.exists():
                shutil.rmtree(alias)
        for alias in candidate.aliases:
            if alias.exists():
                raise RuntimeError(f"source alias survived prune: {alias}")
    return receipt


def _process_is_using(paths: Sequence[Path]) -> str | None:
    resolved = tuple(path.resolve() for path in paths)
    needles = tuple(str(path) for path in resolved)
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    for line in output.splitlines():
        if "compact_policy_checkpoint_storage.py" in line:
            continue
        if any(needle in line for needle in needles):
            return line.strip()
    # A consumer need not mention the checkpoint in argv after opening it.
    # Check live file descriptors too; permission races and processes exiting
    # during the scan are benign and are retried by a later invocation.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / "cmdline").read_bytes()
        except OSError:
            continue
        if b"compact_policy_checkpoint_storage.py" in command:
            continue
        try:
            descriptors = tuple((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = descriptor.resolve(strict=True)
            except OSError:
                continue
            if any(target == root or root in target.parents for root in resolved):
                return f"pid={process.name} fd={descriptor.name} target={target}"
    return None


def delete_non_scientific() -> dict[str, object]:
    targets = tuple(
        (ARTIFACTS / rel).resolve(strict=True) for rel in NON_SCIENTIFIC_CHECKPOINTS
    )
    for target in targets:
        _reject_excluded(target)
        _rank_zero(target)
    failed = (ARTIFACTS / FAILED_TREE).resolve(strict=True)
    _reject_excluded(failed)
    in_use = _process_is_using((*targets, failed))
    if in_use:
        raise RuntimeError(f"refusing to delete an in-use artifact: {in_use}")
    execution = ARTIFACTS / "checkpoint-storage-compaction-20260824"
    execution.mkdir(parents=True, exist_ok=True)
    plan_path = execution / "non-scientific-deletion-plan.json"
    receipt_path = execution / "non-scientific-deletion-receipt.json"
    records = [
        {
            "path": str(path),
            **_tree_stats(path),
            "inventory": list(_file_records(path, hash_files=False)),
        }
        for path in (*targets, failed)
    ]
    plan = {
        "schema_version": DELETION_SCHEMA,
        "prepared_at_utc": _utc_now(),
        "checkpoint_payload_count": len(targets),
        "failed_tree_count": 1,
        "targets": records,
        "inventory_sha256": _canonical_sha256(records),
    }
    if not plan_path.exists():
        _write_json(plan_path, plan)
    for target in targets:
        shutil.rmtree(target)
    shutil.rmtree(failed)
    if any(path.exists() for path in (*targets, failed)):
        raise RuntimeError("one or more non-scientific targets survived deletion")
    receipt = {
        **plan,
        "completed_at_utc": _utc_now(),
        "deleted_apparent_bytes": sum(
            int(record["apparent_bytes"]) for record in records
        ),
        "deleted_allocated_bytes": sum(
            int(record["allocated_bytes"]) for record in records
        ),
        "recoverable": False,
    }
    _write_json(receipt_path, receipt)
    return receipt


def cmd_inventory() -> None:
    resolved = candidates()
    print(
        json.dumps(
            {
                "count": len(resolved),
                "kinds": {
                    kind: sum(item.kind == kind for item in resolved)
                    for kind in sorted({item.kind for item in resolved})
                },
                "items": [
                    {
                        "run": item.run.name,
                        "step": item.step,
                        "kind": item.kind,
                        "source": str(item.source),
                        "aliases": [str(path) for path in item.aliases],
                    }
                    for item in resolved
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_compact(args: argparse.Namespace) -> None:
    selected = candidates()
    if args.run:
        selected = tuple(item for item in selected if item.run.name == args.run)
    if args.step is not None:
        selected = tuple(item for item in selected if item.step == args.step)
    positioned = tuple(enumerate(selected, start=1))
    if args.from_position is not None:
        positioned = tuple(pair for pair in positioned if pair[0] >= args.from_position)
    if args.shard_count != 1:
        origin = args.from_position or 1
        positioned = tuple(
            pair
            for pair in positioned
            if (pair[0] - origin) % args.shard_count == args.shard_index
        )
    selected = tuple(item for _, item in positioned)
    if not selected:
        raise ValueError("selection is empty")
    for local_index, (position, item) in enumerate(positioned, start=1):
        print(
            f"[{local_index}/{len(selected)}; global={position}/81] "
            f"{item.run.name} S{item.step} {item.kind}",
            flush=True,
        )
        receipt = compact_one(item, prune_source=args.prune_source)
        print(
            json.dumps(
                {
                    "run": receipt["run"],
                    "step": receipt["optimizer_step"],
                    "model_bytes": receipt["model_bytes"],
                    "method": receipt["materialization_method"],
                    "pruned": args.prune_source,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    compact = subparsers.add_parser("compact")
    compact.add_argument("--run")
    compact.add_argument("--step", type=int)
    compact.add_argument("--prune-source", action="store_true")
    compact.add_argument("--from-position", type=int)
    compact.add_argument("--shard-count", type=int, default=1)
    compact.add_argument("--shard-index", type=int, default=0)
    subparsers.add_parser("delete-non-scientific")
    args = parser.parse_args()
    assert_legacy_standalone_mode_quarantined(
        "tools/compact_policy_checkpoint_storage.py",
        selected_mode=args.command,
        read_only_modes=("inventory",),
        blocked_modes=("compact", "delete-non-scientific"),
    )
    if getattr(args, "shard_count", 1) <= 0:
        parser.error("--shard-count must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "shard_count", 1):
        parser.error("--shard-index must be in [0, shard-count)")
    if args.command == "inventory":
        cmd_inventory()
    elif args.command == "compact":
        cmd_compact(args)
    elif args.command == "delete-non-scientific":
        print(json.dumps(delete_non_scientific(), indent=2, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
