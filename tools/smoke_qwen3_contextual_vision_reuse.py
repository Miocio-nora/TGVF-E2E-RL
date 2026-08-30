#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Compare native contextual conditioning with and without vision reuse."""

from __future__ import annotations

# Direct script execution is stopped before legacy path/environment mutation or
# heavyweight runtime imports. Importing the module for read-only compatibility
# tests remains possible; its public ``main`` retains a second fail-closed guard.
if __name__ == "__main__":
    import os as _early_quarantine_os

    _early_quarantine_root = _early_quarantine_os.path.realpath(__file__)
    for _early_quarantine_depth in range(2):
        _early_quarantine_root = _early_quarantine_os.path.dirname(
            _early_quarantine_root
        )
    _early_quarantine_os.execv(
        "/usr/bin/python3",
        (
            "/usr/bin/python3",
            "-I",
            _early_quarantine_os.path.join(
                _early_quarantine_root,
                "tools",
                "check_launch_gate.py",
            ),
            "quarantine-legacy",
            "--tool-id",
            "tools/smoke_qwen3_contextual_vision_reuse.py",
        ),
    )

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from tgvf_rl.checkpoint.coordinator import state_digest
from tgvf_rl.qwen.qwen3_vl import Qwen3VLAdapter
from tgvf_rl.representation.training.config import (
    load_representation_training_config,
)
from tgvf_rl.representation.training.data import (
    load_retained_representation_jsonl,
)
from tgvf_rl.representation.training.distributed_checkpoint import (
    load_rank_zero_adapter_owned_state_export,
)
from tgvf_rl.representation.training.evaluation_runner import (
    _enable_determinism,
    _load_qwen,
    _load_rgb_image,
    _require_file_sha256,
    _require_launch_environment,
    _seed_current_process,
    _torch_dtype,
    _validate_training_artifact_binding,
    _verify_live_code_identity,
    load_representation_internal_evaluation_run_config,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.qwen3_grounding import (
    Qwen3GroundingBuild,
    Qwen3GroundingDiagnosticBuilder,
    load_qwen3_grounding_manifest,
)
from tgvf_rl.representation.training.readout import (
    RepresentationVisualTensorBundle,
)
from tgvf_rl.representation.training.runtime import (
    create_qwen3_representation_runtime,
)
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--atol", type=float, default=0.015625)
    parser.add_argument("--rtol", type=float, default=0.015625)
    return parser


def _timed_build(callback) -> tuple[Qwen3GroundingBuild, float]:
    torch.cuda.synchronize()
    started = perf_counter()
    result = callback()
    torch.cuda.synchronize()
    return result, perf_counter() - started


def _observations(
    build: Qwen3GroundingBuild,
) -> dict[str, RepresentationVisualTensorBundle]:
    observations: dict[str, RepresentationVisualTensorBundle] = {}
    for case in build.cross_image.cases:
        observations[f"{case.case_id}:a"] = case.observation_a
        observations[f"{case.case_id}:b"] = case.observation_b
    for case in build.target_presence_cases:
        observations[f"{case.case_id}:positive"] = case.positive_observation
        observations[f"{case.case_id}:negative"] = case.negative_observation
    return observations


def _tensor_diagnostic(
    legacy: torch.Tensor,
    reused: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    if legacy.shape != reused.shape:
        raise ValueError("contextual vision-reuse parity changed a tensor shape")
    difference = (legacy.float() - reused.float()).abs()
    return {
        "shape": list(legacy.shape),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
        "within_tolerance": bool(torch.allclose(legacy, reused, atol=atol, rtol=rtol)),
    }


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/smoke_qwen3_contextual_vision_reuse.py"
    )
    args = _parser().parse_args()
    config = load_representation_internal_evaluation_run_config(args.config)
    _require_launch_environment(config)
    _verify_live_code_identity(config)
    training = load_representation_training_config(config.training_config_path)
    _require_file_sha256(
        config.training_config_path,
        config.training_config_sha256,
        name="training config",
    )
    _require_file_sha256(
        config.artifact_path,
        config.artifact_file_sha256,
        name="Adapter artifact",
    )
    evaluation = config.evaluation
    if (
        evaluation.grounding_manifest_path is None
        or evaluation.grounding_manifest_sha256 is None
    ):
        raise ValueError("contextual vision-reuse smoke requires grounding manifest")
    _require_file_sha256(
        evaluation.grounding_manifest_path,
        evaluation.grounding_manifest_sha256,
        name="grounding manifest",
    )
    report_path = evaluation.report_path
    if report_path is None:
        raise ValueError("contextual vision-reuse smoke requires report path")
    if report_path.exists():
        raise FileExistsError(f"smoke report already exists: {report_path}")
    if not report_path.parent.is_dir():
        raise FileNotFoundError(f"smoke report parent is missing: {report_path.parent}")

    export = load_rank_zero_adapter_owned_state_export(config.artifact_path)
    if export.state is None:
        raise RuntimeError("Adapter export has no tensor state")
    manifest = export.manifest
    if state_digest(manifest) != config.artifact_manifest_sha256:
        raise ValueError("Adapter artifact manifest SHA256 mismatch")
    if manifest.run_identity_sha256 != config.expected_run_identity_sha256:
        raise ValueError("Adapter artifact run identity mismatch")
    if manifest.global_step != config.expected_global_step:
        raise ValueError("Adapter artifact global step mismatch")
    _validate_training_artifact_binding(training, manifest.run_identity)
    if (
        config.evaluation_data_path is None
        or config.evaluation_data_source_sha256 is None
    ):
        raise ValueError("contextual vision-reuse smoke requires evaluation data")
    retained = load_retained_representation_jsonl(
        config.evaluation_data_path,
        expected_source_sha256=config.evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _enable_determinism()
    if evaluation.random_seed is None:
        raise ValueError("contextual vision-reuse smoke requires random seed")
    _seed_current_process(evaluation.random_seed)
    processor, model = _load_qwen(training, device=device)
    tokenizer_length_before = len(processor.tokenizer)
    runtime = create_qwen3_representation_runtime(
        model=model,
        processor=processor,
        model_identity=training.model_identity,
        conditioning_config=training.provider,
        adapter_dtype=_torch_dtype(training.model.dtype),
        adapter_variant=training.adapter_variant,
        fixture_mode=False,
    )
    manifest.run_identity.adapter_contract.assert_matches(runtime.adapter)
    runtime.adapter.load_artifact_state_dict(export.state)
    runtime.adapter.requires_grad_(False)
    runtime.adapter.eval()
    model.requires_grad_(False)
    model.eval()
    family = Qwen3VLAdapter()
    common = {
        "runtime": runtime,
        "family_adapter": family,
        "prompt": training.prompt,
        "image_loader": _load_rgb_image,
        "image_max_pixels": training.model.image_max_pixels,
    }
    legacy_builder = Qwen3NativeRepresentationGroupBuilder(**common)
    reused_builder = Qwen3NativeRepresentationGroupBuilder(
        **common,
        reuse_preencoded_vision_for_contextual_conditioning=True,
    )
    grounding_manifest = load_qwen3_grounding_manifest(
        evaluation.grounding_manifest_path
    )
    selected = tuple(
        probe
        for probe in grounding_manifest.target_presence_probes
        if probe.pair_id == args.pair_id
    )
    if len(selected) != 1:
        raise ValueError("pair-id must select exactly one target-presence probe")
    bounded_manifest = replace(
        grounding_manifest,
        cross_image_probes=(grounding_manifest.cross_image_probes[0],),
        target_presence_probes=selected,
    )

    def build(group_builder: Qwen3NativeRepresentationGroupBuilder):
        return Qwen3GroundingDiagnosticBuilder(
            runtime=runtime,
            group_builder=group_builder,
        ).build(
            manifest=bounded_manifest,
            data_manifest_sha256=retained.manifest.manifest_sha256,
            samples=retained.samples,
        )

    legacy, legacy_seconds = _timed_build(lambda: build(legacy_builder))
    reused, reused_seconds = _timed_build(lambda: build(reused_builder))
    legacy_observations = _observations(legacy)
    reused_observations = _observations(reused)
    if legacy_observations.keys() != reused_observations.keys():
        raise RuntimeError("contextual vision-reuse parity changed observation keys")
    diagnostics: dict[str, object] = {}
    all_within_tolerance = True
    for name in legacy_observations:
        legacy_bundle = legacy_observations[name]
        reused_bundle = reused_observations[name]
        tensor_diagnostics = {
            "main": _tensor_diagnostic(
                legacy_bundle.main,
                reused_bundle.main,
                atol=args.atol,
                rtol=args.rtol,
            )
        }
        for index, (legacy_branch, reused_branch) in enumerate(
            zip(legacy_bundle.deepstack, reused_bundle.deepstack, strict=True)
        ):
            tensor_diagnostics[f"deepstack_{index}"] = _tensor_diagnostic(
                legacy_branch,
                reused_branch,
                atol=args.atol,
                rtol=args.rtol,
            )
        diagnostics[name] = tensor_diagnostics
        all_within_tolerance = all_within_tolerance and all(
            bool(value["within_tolerance"]) for value in tensor_diagnostics.values()
        )
    if not all_within_tolerance:
        raise RuntimeError("contextual vision-reuse parity exceeded tolerance")
    if len(processor.tokenizer) != tokenizer_length_before:
        raise RuntimeError("contextual vision-reuse smoke changed tokenizer length")
    payload = {
        "schema_version": "qwen3-contextual-vision-reuse-smoke-v1",
        "run_id": config.run_id,
        "source_config_sha256": config.source_sha256,
        "code_commit": config.code_commit,
        "pair_id": args.pair_id,
        "artifact_manifest_sha256": config.artifact_manifest_sha256,
        "observation_count": len(diagnostics),
        "atol": args.atol,
        "rtol": args.rtol,
        "all_within_tolerance": all_within_tolerance,
        "tensor_diagnostics": diagnostics,
        "legacy_seconds": legacy_seconds,
        "reused_seconds": reused_seconds,
        "speedup": legacy_seconds / reused_seconds,
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": len(processor.tokenizer),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with report_path.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({**payload, "report_sha256": sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
