#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Run one real-Qwen cached/no-cache native-D continuation parity smoke."""

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
            "tools/smoke_qwen3_cached_continuation.py",
        ),
    )

import argparse
from dataclasses import asdict, replace
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
from tgvf_rl.representation.training.internal_evaluation import (
    NativeFreeContinuationRequest,
    create_injected_native_counterfactual_evaluator,
)
from tgvf_rl.representation.training.native_pipeline import (
    Qwen3NativeRepresentationGroupBuilder,
)
from tgvf_rl.representation.training.qwen3_grounding import (
    Qwen3GroundingDiagnosticBuilder,
    load_qwen3_grounding_manifest,
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
    parser.add_argument("--allow-logit-mismatch-for-diagnostics", action="store_true")
    return parser


def _synchronized_seconds(callback) -> tuple[object, float]:
    torch.cuda.synchronize()
    started = perf_counter()
    result = callback()
    torch.cuda.synchronize()
    return result, perf_counter() - started


def main() -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/smoke_qwen3_cached_continuation.py"
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
        raise ValueError("cached-continuation smoke requires grounding manifest")
    _require_file_sha256(
        evaluation.grounding_manifest_path,
        evaluation.grounding_manifest_sha256,
        name="grounding manifest",
    )
    report_path = evaluation.report_path
    if report_path is None:
        raise ValueError("cached-continuation smoke requires report path")
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
        raise ValueError("cached-continuation smoke requires explicit evaluation data")
    retained = load_retained_representation_jsonl(
        config.evaluation_data_path,
        expected_source_sha256=config.evaluation_data_source_sha256,
        warn_on_leakage=training.data.warn_on_target_leakage,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _enable_determinism()
    if evaluation.random_seed is None:
        raise ValueError("cached-continuation smoke requires random seed")
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
    group_builder = Qwen3NativeRepresentationGroupBuilder(
        runtime=runtime,
        family_adapter=family,
        prompt=training.prompt,
        image_loader=_load_rgb_image,
        image_max_pixels=training.model.image_max_pixels,
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
    grounding = Qwen3GroundingDiagnosticBuilder(
        runtime=runtime,
        group_builder=group_builder,
    ).build(
        manifest=bounded_manifest,
        data_manifest_sha256=retained.manifest.manifest_sha256,
        samples=retained.samples,
    )
    case = grounding.target_presence_cases[0]
    if evaluation.max_new_tokens is None or evaluation.eos_token_ids is None:
        raise ValueError("cached-continuation smoke generation contract is incomplete")
    evaluator = create_injected_native_counterfactual_evaluator(
        model=model,
        family_adapter=family,
        materializer=grounding.target_presence_materializer,
        eos_token_ids=evaluation.eos_token_ids,
        max_new_tokens=evaluation.max_new_tokens,
    )
    request = NativeFreeContinuationRequest(
        case_id=case.case_id,
        variant="value_a",
        expected_value=case.present_value,
        context=case.positive_context,
        observation_identity=case.positive_observation_identity,
        observation=case.positive_observation,
    )
    parity = evaluator.continuation_cache_parity(
        request,
        atol=args.atol,
        rtol=args.rtol,
        require_logits_within_tolerance=(not args.allow_logit_mismatch_for_diagnostics),
    )
    cached, cached_seconds = _synchronized_seconds(
        lambda: evaluator.free_continuation(request)
    )
    no_cache, no_cache_seconds = _synchronized_seconds(
        lambda: evaluator.free_continuation_no_cache(request)
    )
    if cached != no_cache or cached != parity.output:
        raise RuntimeError("timed cached/no-cache outputs differ from parity output")
    if len(processor.tokenizer) != tokenizer_length_before:
        raise RuntimeError("cached-continuation smoke changed tokenizer length")
    payload = {
        "schema_version": "qwen3-native-d-cached-continuation-smoke-v1",
        "run_id": config.run_id,
        "source_config_sha256": config.source_sha256,
        "code_commit": config.code_commit,
        "training_run_identity_sha256": manifest.run_identity_sha256,
        "artifact_manifest_sha256": config.artifact_manifest_sha256,
        "grounding_manifest_file_sha256": evaluation.grounding_manifest_sha256,
        "pair_id": args.pair_id,
        "case_id": case.case_id,
        "max_new_tokens": evaluation.max_new_tokens,
        "compared_logit_steps": parity.compared_logit_steps,
        "max_abs_logit_difference": parity.max_abs_logit_difference,
        "mean_abs_logit_difference": parity.mean_abs_logit_difference,
        "max_selected_token_logit_difference": (
            parity.max_selected_token_logit_difference
        ),
        "min_cached_top1_margin": parity.min_cached_top1_margin,
        "min_oracle_top1_margin": parity.min_oracle_top1_margin,
        "logits_within_tolerance": parity.logits_within_tolerance,
        "atol": parity.atol,
        "rtol": parity.rtol,
        "generated_token_ids": list(parity.output.generated_token_ids),
        "generated_text": parity.output.generated_text,
        "stop_reason": parity.output.stop_reason,
        "cached_seconds": cached_seconds,
        "no_cache_seconds": no_cache_seconds,
        "speedup": no_cache_seconds / cached_seconds,
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": len(processor.tokenizer),
        "model_identity": state_digest(asdict(manifest.run_identity.model)),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with report_path.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({**payload, "report_sha256": sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
