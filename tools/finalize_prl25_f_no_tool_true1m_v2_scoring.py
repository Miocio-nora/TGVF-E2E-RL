#!/usr/bin/env python3
"""Fail-closed PRL25-F no-tool TRUE1M V2 scoring closure.

The CoreDev summarizer intentionally emits a general seven-slice summary.  This
tool turns that raw summary into the frozen paper-facing result by extracting
Macro*, validating all headline populations, and binding the result to atomic
per-arm and four-arm completion receipts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from tgvf_rl.evaluation.coredev_results import (
    COREDEV_MACRO_STAR_COMPONENTS,
    COREDEV_MACRO_STAR_SCHEMA,
    extract_coredev_macro_star,
    write_json_atomic,
)
from tgvf_rl.evaluation.policy_coredev_scoring import (
    DATASETS,
    validate_vlmevalkit_eval_id,
)


STEPS = (0, 8, 16, 32)
TRUE1M_MAX_PIXELS = 1_003_520
SAMPLE_COUNT = 2_511
SLICE_COUNT = 7
MODEL = "Qwen3-VL-8B-Instruct"
EVALUATION_ID_PREFIX = "PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S"
EVALUATION_ID_SUFFIX = "-TRUE1M-V2"
TOTAL_EVALUATION_ID = "PRL25-F-NO-TOOL-RL-MATCHED-COREDEV2511-S0-S8-S16-S32-TRUE1M-V2"
STEP_COMPLETION_SCHEMA = "tgvf.prl25-f-no-tool-true1m-v2-step-scoring.v1"
AGGREGATE_COMPLETION_SCHEMA = "tgvf.prl25-f-no-tool-true1m-v2-scoring-completion.v2"
MATCHED_INFERENCE_COMPLETION_SCHEMA = (
    "tgvf.prl25-f-true1m-matched-inference-completion.v4"
)
INFERENCE_COMPLETION_SCHEMA = "tgvf.prl25-f-true1m-inference-completion.v4"
INFERENCE_STATUS_SCHEMA = "tgvf.prl25-f-true1m-inference-status.v2"
RANK_TREE_SCHEMA = "tgvf.prl25-f-true1m-rank-tree.v1"
INFERENCE_SOURCE_CLOSURE_SCHEMA = (
    "tgvf.prl25-f-no-tool-true1m-v2-inference-source-closure.v1"
)
MATHVERSE_VERSIONS = (
    "Text Dominant",
    "Vision Only",
    "Text Lite",
    "Vision Intensive",
    "Vision Dominant",
)
OCR_LANGUAGES = ("english", "chinese")
SINGLE_IMAGE_COUNTS = {"blink": 180, "mmmu": 269}
AGGREGATION = "unweighted_mean_of_seven_percent_components"
SOURCE_RUN_IDS = {
    0: "T20260826_G0",
    8: "T20260826_G8",
    16: "T20260826_G10",
    32: "T20260826_G20",
}
PINNED_RECEIPT_SCHEMA = "tgvf.vlmevalkit-pinned-reuse-receipt.v1"
SOURCE_CLOSURE_SCHEMA = "tgvf.prl25-f-no-tool-true1m-v2-scoring-source.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PINNED_RECEIPT_FIELDS = {
    "schema_version",
    "dataset",
    "model",
    "source_evaluation_id",
    "source_run_id",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_prediction_path",
    "source_prediction_sha256",
    "destination_run_id",
    "destination_status_path",
    "destination_status_sha256",
    "destination_prediction_path",
    "destination_prediction_sha256",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"accepted JSON artifact is not a regular file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read accepted JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"accepted JSON artifact is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validated_eval_root(eval_root: Path) -> Path:
    if eval_root.is_symlink() or not eval_root.is_dir():
        raise RuntimeError("TRUE1M V2 evaluation root is absent or is a symlink")
    resolved = eval_root.resolve(strict=True)
    if resolved.name != TOTAL_EVALUATION_ID:
        raise RuntimeError(
            "TRUE1M V2 evaluation-root basename differs from the frozen contract"
        )
    return resolved


def _contained_regular_file(raw_path: object, *, root: Path, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{name} path is absent")
    lexical = Path(raw_path)
    if not lexical.is_absolute():
        raise RuntimeError(f"{name} path is not absolute")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{name} path is absent") from error
    if (
        lexical != resolved
        or lexical.is_symlink()
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
    ):
        raise RuntimeError(f"{name} escapes its current step scoring root")
    return resolved


def _status_prediction_path(
    raw_path: object, *, run_dir: Path, scoring_root: Path, name: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{name} status prediction path is absent")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        if ".." in candidate.parts:
            raise RuntimeError(f"{name} status prediction path is unsafe")
        candidate = run_dir / candidate
    return _contained_regular_file(str(candidate), root=scoring_root, name=name)


def _require_digest(value: object, *, path: Path, name: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value != _sha256_file(path)
    ):
        raise RuntimeError(f"{name} digest differs from the referenced bytes")
    return value


def _keys(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} is not an object")
    if not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{name} has a non-string key")
    return tuple(value)


def _has_exact_keys(value: object, expected: Sequence[str], *, name: str) -> bool:
    observed = _keys(value, name=name)
    return len(observed) == len(expected) and set(observed) == set(expected)


def _percent(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise RuntimeError(f"{name} is not a finite percentage")
    return result


def _assert_close(observed: float, expected: float, *, name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{name} does not match its frozen components")


def validate_headline(headline: object) -> dict[str, Any]:
    """Validate the complete frozen Macro* payload, including its populations."""

    if not isinstance(headline, Mapping):
        raise RuntimeError("CoreDev Macro* headline is absent")
    if headline.get("schema_version") != COREDEV_MACRO_STAR_SCHEMA:
        raise RuntimeError("CoreDev Macro* schema differs")
    if headline.get("aggregation") != AGGREGATION:
        raise RuntimeError("CoreDev Macro* aggregation differs")

    components = headline.get("components_percent")
    if not _has_exact_keys(
        components, COREDEV_MACRO_STAR_COMPONENTS, name="Macro* components"
    ):
        raise RuntimeError("CoreDev Macro* component fields differ")
    assert isinstance(components, Mapping)
    component_values = {
        key: _percent(components[key], name=f"Macro* component {key}")
        for key in COREDEV_MACRO_STAR_COMPONENTS
    }

    ocr = headline.get("ocr_language_components_percent")
    if not _has_exact_keys(ocr, OCR_LANGUAGES, name="OCR headline components"):
        raise RuntimeError("OCR English/Chinese headline fields differ")
    assert isinstance(ocr, Mapping)
    ocr_values = {key: _percent(ocr[key], name=f"OCR {key}") for key in OCR_LANGUAGES}
    _assert_close(
        component_values["ocr_mean"],
        math.fsum(ocr_values.values()) / len(ocr_values),
        name="OCR mean",
    )

    mathverse = headline.get("mathverse_version_components_percent")
    if not _has_exact_keys(
        mathverse, MATHVERSE_VERSIONS, name="MathVerse headline components"
    ):
        raise RuntimeError("MathVerse five-version headline fields differ")
    assert isinstance(mathverse, Mapping)
    mathverse_values = {
        key: _percent(mathverse[key], name=f"MathVerse {key}")
        for key in MATHVERSE_VERSIONS
    }
    _assert_close(
        component_values["mathverse_five_version_macro"],
        math.fsum(mathverse_values.values()) / len(mathverse_values),
        name="MathVerse five-version macro",
    )

    populations = headline.get("single_image_counts")
    if not _has_exact_keys(
        populations,
        tuple(SINGLE_IMAGE_COUNTS),
        name="single-image headline populations",
    ):
        raise RuntimeError("BLINK/MMMU headline population fields differ")
    assert isinstance(populations, Mapping)
    for population, expected_count in SINGLE_IMAGE_COUNTS.items():
        record = populations[population]
        if not isinstance(record, Mapping) or set(record) != {"correct", "count"}:
            raise RuntimeError(f"{population} headline population schema differs")
        correct = record.get("correct")
        count = record.get("count")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != expected_count
            or not 0 <= correct <= count
        ):
            raise RuntimeError(f"{population} headline population differs")
        component = "blink_single_180" if population == "blink" else "mmmu_single_269"
        _assert_close(
            component_values[component],
            100.0 * correct / count,
            name=f"{population} single-image accuracy",
        )

    macro = _percent(headline.get("macro_star_percent"), name="Macro*")
    _assert_close(
        macro,
        math.fsum(component_values.values()) / len(component_values),
        name="Macro*",
    )
    return dict(headline)


def _evaluation_id(step: int) -> str:
    if step not in STEPS:
        raise RuntimeError(f"unexpected optimizer step: {step}")
    return f"{EVALUATION_ID_PREFIX}{step}{EVALUATION_ID_SUFFIX}"


def _summary_path(eval_root: Path, step: int) -> Path:
    return (
        eval_root / f"matched/step{step}/scoring/coredev-official-v1/"
        "coredev-2511-eval-summary.json"
    ).resolve()


def _step_completion_path(eval_root: Path, step: int) -> Path:
    return (
        eval_root / f"runtime/scoring-supervisor/s{step}-scoring-complete.json"
    ).resolve()


def _validated_identity_payload(
    path: Path, *, schema: str, root: Path, name: str
) -> dict[str, Any]:
    accepted_path = _contained_regular_file(str(path), root=root, name=name)
    payload = _load_json(accepted_path)
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if (
        payload.get("schema_version") != schema
        or not isinstance(identity, str)
        or _SHA256.fullmatch(identity) is None
        or identity != _canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{name} identity differs")
    return payload


def _validated_identity_binding(
    binding: object,
    *,
    expected_path: Path,
    schema: str,
    root: Path,
    name: str,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "file_sha256",
        "identity_sha256",
    }:
        raise RuntimeError(f"{name} binding is malformed")
    path = _contained_regular_file(binding.get("path"), root=root, name=name)
    if path != expected_path.resolve():
        raise RuntimeError(f"{name} path differs")
    payload = _validated_identity_payload(
        path,
        schema=schema,
        root=root,
        name=name,
    )
    if (
        binding.get("file_sha256") != _sha256_file(path)
        or binding.get("identity_sha256") != payload["identity_sha256"]
    ):
        raise RuntimeError(f"{name} bytes differ")
    return payload


def _validate_rank_tree(
    rank_tree: object, *, eval_root: Path, step: int
) -> dict[str, Any]:
    if not isinstance(rank_tree, Mapping):
        raise RuntimeError(f"S{step} rank tree is absent")
    identity = rank_tree.get("identity_sha256")
    unsigned = dict(rank_tree)
    unsigned.pop("identity_sha256", None)
    files = rank_tree.get("files")
    if (
        rank_tree.get("schema_version") != RANK_TREE_SCHEMA
        or rank_tree.get("evaluation_id") != _evaluation_id(step)
        or rank_tree.get("optimizer_step") != step
        or rank_tree.get("world_size") != 4
        or rank_tree.get("row_count") != 2240
        or not isinstance(identity, str)
        or identity != _canonical_sha256(unsigned)
        or not isinstance(files, list)
        or len(files) != 4
    ):
        raise RuntimeError(f"S{step} rank tree identity differs")

    arm_root = eval_root / f"matched/step{step}"
    global_identities: list[dict[str, object]] = []
    observed_ordinals: set[int] = set()
    expected_file_fields = {
        "path",
        "resolved_path",
        "sha256",
        "size_bytes",
        "rank",
        "line_count",
        "ordinal_sequence_sha256",
        "result_identity_sequence_sha256",
    }
    for rank, record in enumerate(files):
        if not isinstance(record, Mapping) or set(record) != expected_file_fields:
            raise RuntimeError(f"S{step} rank{rank} file record is malformed")
        expected_path = (arm_root / f"inference/rank-{rank}.jsonl").resolve()
        path = _contained_regular_file(
            record.get("path"),
            root=arm_root,
            name=f"S{step} rank{rank} JSONL",
        )
        if (
            record.get("rank") != rank
            or path != expected_path
            or record.get("resolved_path") != str(expected_path)
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError(f"S{step} rank{rank} byte binding differs")

        ordinals: list[int] = []
        identities: list[dict[str, object]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(f"S{step} rank{rank} has an empty row")
                try:
                    row = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"S{step} rank{rank} row {line_number} is malformed"
                    ) from error
                if not isinstance(row, dict):
                    raise RuntimeError(f"S{step} rank{rank} row is not an object")
                ordinal = row.get("ordinal")
                result_identity = row.get("result_identity_sha256")
                hash_payload = dict(row)
                hash_payload.pop("result_identity_sha256", None)
                # ``wall_seconds`` is durable runtime telemetry, not part of the
                # policy-result identity.  Keep this recomputation aligned with
                # validate_policy_benchmark_result(), while the enclosing rank
                # file SHA256 still binds the exact telemetry bytes.
                hash_payload.pop("wall_seconds", None)
                if (
                    type(ordinal) is not int
                    or ordinal % 4 != rank
                    or ordinal in observed_ordinals
                    or row.get("evaluation_id") != _evaluation_id(step)
                    or not isinstance(result_identity, str)
                    or _SHA256.fullmatch(result_identity) is None
                    or result_identity != _canonical_sha256(hash_payload)
                ):
                    raise RuntimeError(f"S{step} rank{rank} row identity differs")
                observed_ordinals.add(ordinal)
                ordinals.append(ordinal)
                identities.append(
                    {
                        "ordinal": ordinal,
                        "result_identity_sha256": result_identity,
                    }
                )
        ordinals.sort()
        identities.sort(key=lambda item: int(item["ordinal"]))
        if (
            record.get("line_count") != len(ordinals)
            or record.get("ordinal_sequence_sha256") != _canonical_sha256(ordinals)
            or record.get("result_identity_sequence_sha256")
            != _canonical_sha256(identities)
        ):
            raise RuntimeError(f"S{step} rank{rank} row sequence differs")
        global_identities.extend(identities)

    global_identities.sort(key=lambda item: int(item["ordinal"]))
    if len(observed_ordinals) != 2240 or rank_tree.get(
        "result_identity_sequence_sha256"
    ) != _canonical_sha256(global_identities):
        raise RuntimeError(f"S{step} global row identity sequence differs")
    return dict(rank_tree)


def _validate_processor_proof(*, eval_root: Path, step: int) -> dict[str, object]:
    arm_root = eval_root / f"matched/step{step}"
    config_path = _contained_regular_file(
        str(arm_root / "config.json"),
        root=arm_root,
        name=f"S{step} benchmark config",
    )
    proof_path = _contained_regular_file(
        str(arm_root / "runtime/true1m-processor-proof.json"),
        root=arm_root,
        name=f"S{step} processor proof",
    )
    config = _load_json(config_path)
    acceptance = _load_json(proof_path)
    proof = acceptance.get("proof")
    if (
        config.get("evaluation_id") != _evaluation_id(step)
        or config.get("evaluation_image_max_pixels") != TRUE1M_MAX_PIXELS
        or acceptance.get("schema_version")
        != "tgvf.prl25-f-true1m-processor-acceptance.v2"
        or acceptance.get("evaluation_id") != _evaluation_id(step)
        or acceptance.get("optimizer_step") != step
        or acceptance.get("gpu_or_api_used") is not False
        or acceptance.get("vllm_engine_constructed") is not False
        or not isinstance(proof, Mapping)
        or proof.get("configured_image_max_pixels") != TRUE1M_MAX_PIXELS
        or proof.get("synthetic_native_source_pixel_area") != 3_145_728
        or proof.get("synthetic_native_represented_pixel_area") != 995_328
        or proof.get("synthetic_native_visual_token_count") != 972
        or proof.get("runtime_mm_processor_kwargs")
        != {
            "size": {
                "shortest_edge": 65_536,
                "longest_edge": TRUE1M_MAX_PIXELS,
            }
        }
        or proof.get("runtime_override_path") != "mm_processor_kwargs.size.longest_edge"
        or proof.get("vllm_012_shallow_hashable") is not True
        or proof.get("nested_images_kwargs_present") is not False
        or proof.get("max_pixels_kwarg_present") is not False
        or proof.get("tool_schema_visible") is not False
        or proof.get("system_prompt_present") is not False
    ):
        raise RuntimeError(f"S{step} processor proof differs")
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "processor_proof_path": str(proof_path),
        "processor_proof_sha256": _sha256_file(proof_path),
    }


def validate_inference_source_closure(eval_root: Path) -> dict[str, Any]:
    """Bind scoring to the exact verified inference bytes and true1M proofs."""

    eval_root = _validated_eval_root(eval_root)
    control_root = eval_root / "runtime/supervisor"
    aggregate_path = control_root / "matched-inference-complete.json"
    aggregate = _validated_identity_payload(
        aggregate_path,
        schema=MATCHED_INFERENCE_COMPLETION_SCHEMA,
        root=eval_root,
        name="matched inference completion",
    )
    arms = aggregate.get("arms")
    if (
        aggregate.get("evaluation_id") != TOTAL_EVALUATION_ID
        or aggregate.get("optimizer_steps") != list(STEPS)
        or aggregate.get("completed_single_image_per_step") != 2240
        or aggregate.get("unsupported_multi_image_per_step") != 271
        or not isinstance(arms, Mapping)
        or set(arms) != {str(step) for step in STEPS}
    ):
        raise RuntimeError("matched inference aggregate identity differs")

    runtime_binding = aggregate.get("runtime_environment_contract")
    launch_binding = aggregate.get("worker_launch_contract")
    _validated_identity_binding(
        runtime_binding,
        expected_path=control_root / "runtime-environment-contract.json",
        schema="tgvf.prl25-f-runtime-environment-contract.v2",
        root=eval_root,
        name="runtime environment contract",
    )
    _validated_identity_binding(
        launch_binding,
        expected_path=control_root / "worker-launch-contract.json",
        schema="tgvf.prl25-f-worker-launch-contract.v2",
        root=eval_root,
        name="worker launch contract",
    )

    arm_records: list[dict[str, object]] = []
    for step in STEPS:
        arm_root = eval_root / f"matched/step{step}"
        completion_path = arm_root / "runtime/inference-complete.json"
        completion = _validated_identity_binding(
            arms[str(step)],
            expected_path=completion_path,
            schema=INFERENCE_COMPLETION_SCHEMA,
            root=eval_root,
            name=f"S{step} inference completion",
        )
        status = _validated_identity_binding(
            completion.get("status_receipt"),
            expected_path=arm_root / "runtime/inference-status.json",
            schema=INFERENCE_STATUS_SCHEMA,
            root=eval_root,
            name=f"S{step} inference status",
        )
        rank_tree = _validate_rank_tree(
            status.get("rank_tree"),
            eval_root=eval_root,
            step=step,
        )
        if (
            completion.get("evaluation_id") != _evaluation_id(step)
            or completion.get("optimizer_step") != step
            or completion.get("completed_single_image") != 2240
            or completion.get("unsupported_multi_image") != 271
            or completion.get("rank_tree_identity_sha256")
            != rank_tree["identity_sha256"]
            or completion.get("result_identity_sequence_sha256")
            != rank_tree["result_identity_sequence_sha256"]
        ):
            raise RuntimeError(f"S{step} inference completion closure differs")
        arm_records.append(
            {
                "optimizer_step": step,
                "evaluation_id": _evaluation_id(step),
                "inference_completion_path": str(completion_path.resolve()),
                "inference_completion_sha256": _sha256_file(completion_path),
                "inference_completion_identity_sha256": completion["identity_sha256"],
                "rank_tree_identity_sha256": rank_tree["identity_sha256"],
                "result_identity_sequence_sha256": rank_tree[
                    "result_identity_sequence_sha256"
                ],
                **_validate_processor_proof(eval_root=eval_root, step=step),
            }
        )
    content: dict[str, object] = {
        "schema_version": INFERENCE_SOURCE_CLOSURE_SCHEMA,
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "matched_inference_completion_path": str(aggregate_path.resolve()),
        "matched_inference_completion_sha256": _sha256_file(aggregate_path),
        "matched_inference_completion_identity_sha256": aggregate["identity_sha256"],
        "arms": arm_records,
    }
    return {**content, "identity_sha256": _canonical_sha256(content)}


def _receipt_source_closure(
    *,
    scoring_root: Path,
    dataset: str,
    summary_slice: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    dataset_root = scoring_root / dataset
    receipt_path = _contained_regular_file(
        str(dataset_root / "pinned-reuse-receipt.json"),
        root=scoring_root,
        name=f"S{step}/{dataset} pinned receipt",
    )
    receipt = _load_json(receipt_path)
    source_run_id = SOURCE_RUN_IDS[step]
    source_evaluation_id = _evaluation_id(step)
    if (
        set(receipt) != _PINNED_RECEIPT_FIELDS
        or receipt.get("schema_version") != PINNED_RECEIPT_SCHEMA
        or receipt.get("dataset") != dataset
        or receipt.get("model") != MODEL
        or receipt.get("source_evaluation_id") != source_evaluation_id
        or receipt.get("source_run_id") != source_run_id
    ):
        raise RuntimeError(f"S{step}/{dataset} pinned receipt identity differs")

    source_run_dir = dataset_root / MODEL / source_run_id
    source_manifest_expected = source_run_dir / "final-answer-view-manifest.json"
    source_prediction_expected = source_run_dir / f"{MODEL}_{dataset}.tsv"
    source_status_expected = source_run_dir / "status.json"
    source_manifest = _contained_regular_file(
        receipt.get("source_manifest_path"),
        root=scoring_root,
        name=f"S{step}/{dataset} source manifest",
    )
    source_prediction = _contained_regular_file(
        receipt.get("source_prediction_path"),
        root=scoring_root,
        name=f"S{step}/{dataset} source prediction",
    )
    source_status = _contained_regular_file(
        str(source_status_expected),
        root=scoring_root,
        name=f"S{step}/{dataset} source status",
    )
    if (
        source_manifest != source_manifest_expected
        or source_prediction != source_prediction_expected
    ):
        raise RuntimeError(f"S{step}/{dataset} pinned source paths differ")
    source_manifest_sha256 = _require_digest(
        receipt.get("source_manifest_sha256"),
        path=source_manifest,
        name=f"S{step}/{dataset} source manifest",
    )
    source_prediction_sha256 = _require_digest(
        receipt.get("source_prediction_sha256"),
        path=source_prediction,
        name=f"S{step}/{dataset} source prediction",
    )

    manifest = _load_json(source_manifest)
    derived = manifest.get("derived")
    if not isinstance(derived, Mapping):
        raise RuntimeError(f"S{step}/{dataset} source manifest lacks derived bytes")
    derived_path = _contained_regular_file(
        derived.get("path"),
        root=scoring_root,
        name=f"S{step}/{dataset} manifest-derived prediction",
    )
    if (
        manifest.get("contract") != "vlmevalkit-final-answer-view-v1"
        or derived_path != source_prediction
        or derived.get("sha256") != source_prediction_sha256
    ):
        raise RuntimeError(f"S{step}/{dataset} source manifest binding differs")

    source_status_payload = _load_json(source_status)
    source_entry = source_status_payload.get("datasets", {}).get(dataset)
    if not isinstance(source_entry, Mapping):
        raise RuntimeError(f"S{step}/{dataset} source status entry is absent")
    source_status_prediction = _status_prediction_path(
        source_entry.get("prediction_file"),
        run_dir=source_run_dir,
        scoring_root=scoring_root,
        name=f"S{step}/{dataset} source",
    )
    if (
        source_status_payload.get("eval_id") != source_run_id
        or source_status_payload.get("model_name") != MODEL
        or source_status_payload.get("mode") != "infer"
        or source_status_payload.get("reuse") is not False
        or source_status_payload.get("reuse_aux") != "infer"
        or source_entry.get("status") != "done"
        or source_entry.get("source_run") != source_evaluation_id
        or source_status_prediction != source_prediction
    ):
        raise RuntimeError(f"S{step}/{dataset} source status binding differs")

    destination_run_id = receipt.get("destination_run_id")
    try:
        validate_vlmevalkit_eval_id(destination_run_id)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"S{step}/{dataset} destination run identity differs"
        ) from error
    assert isinstance(destination_run_id, str)
    destination_run_dir = dataset_root / MODEL / destination_run_id
    destination_status_expected = destination_run_dir / "status.json"
    destination_prediction_expected = destination_run_dir / f"{MODEL}_{dataset}.tsv"
    destination_status = _contained_regular_file(
        receipt.get("destination_status_path"),
        root=scoring_root,
        name=f"S{step}/{dataset} destination status",
    )
    destination_prediction = _contained_regular_file(
        receipt.get("destination_prediction_path"),
        root=scoring_root,
        name=f"S{step}/{dataset} destination prediction",
    )
    if (
        destination_status != destination_status_expected
        or destination_prediction != destination_prediction_expected
    ):
        raise RuntimeError(f"S{step}/{dataset} destination paths differ")
    destination_status_sha256 = _require_digest(
        receipt.get("destination_status_sha256"),
        path=destination_status,
        name=f"S{step}/{dataset} destination status",
    )
    destination_prediction_sha256 = _require_digest(
        receipt.get("destination_prediction_sha256"),
        path=destination_prediction,
        name=f"S{step}/{dataset} destination prediction",
    )
    if destination_prediction_sha256 != source_prediction_sha256:
        raise RuntimeError(f"S{step}/{dataset} scorer changed prediction bytes")

    destination_status_payload = _load_json(destination_status)
    destination_entry = destination_status_payload.get("datasets", {}).get(dataset)
    if not isinstance(destination_entry, Mapping):
        raise RuntimeError(f"S{step}/{dataset} destination status entry is absent")
    destination_status_prediction = _status_prediction_path(
        destination_entry.get("prediction_file"),
        run_dir=destination_run_dir,
        scoring_root=scoring_root,
        name=f"S{step}/{dataset} destination",
    )
    if (
        destination_status_payload.get("eval_id") != destination_run_id
        or destination_status_payload.get("model_name") != MODEL
        or destination_status_payload.get("mode") != "eval"
        or destination_status_payload.get("reuse") is not True
        or destination_status_payload.get("reuse_aux") != "infer"
        or destination_entry.get("status") != "done"
        or destination_entry.get("source_run") != source_run_id
        or destination_status_prediction != destination_prediction
    ):
        raise RuntimeError(f"S{step}/{dataset} destination status binding differs")

    summary_status = _contained_regular_file(
        summary_slice.get("status_path"),
        root=scoring_root,
        name=f"S{step}/{dataset} summary status",
    )
    summary_prediction = _contained_regular_file(
        summary_slice.get("prediction_file"),
        root=scoring_root,
        name=f"S{step}/{dataset} summary prediction",
    )
    if (
        summary_slice.get("eval_id") != destination_run_id
        or summary_status != destination_status
        or summary_prediction != destination_prediction
        or summary_slice.get("prediction_sha256") != destination_prediction_sha256
        or summary_slice.get("metrics") != destination_entry.get("metrics")
        or summary_slice.get("primary_metric")
        != destination_entry.get("primary_metric")
    ):
        raise RuntimeError(f"S{step}/{dataset} summary destination binding differs")

    raw_judge_artifacts = summary_slice.get("judge_artifacts")
    if not isinstance(raw_judge_artifacts, list):
        raise RuntimeError(f"S{step}/{dataset} summary judge paths are malformed")
    if dataset == "OCRBench_v2":
        if raw_judge_artifacts:
            raise RuntimeError(f"S{step}/{dataset} unexpected judge artifacts")
    elif not raw_judge_artifacts:
        raise RuntimeError(f"S{step}/{dataset} judge artifacts are absent")
    judge_artifacts: list[dict[str, str]] = []
    observed_judge_paths: set[Path] = set()
    for ordinal, raw_path in enumerate(raw_judge_artifacts):
        artifact = _contained_regular_file(
            raw_path,
            root=scoring_root,
            name=f"S{step}/{dataset} judge artifact {ordinal}",
        )
        if (
            not artifact.is_relative_to(destination_run_dir)
            or artifact in observed_judge_paths
        ):
            raise RuntimeError(f"S{step}/{dataset} judge artifact binding differs")
        observed_judge_paths.add(artifact)
        judge_artifacts.append(
            {"path": str(artifact), "sha256": _sha256_file(artifact)}
        )

    return {
        "dataset": dataset,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "source_run_id": source_run_id,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "source_status_path": str(source_status),
        "source_status_sha256": _sha256_file(source_status),
        "source_prediction_path": str(source_prediction),
        "source_prediction_sha256": source_prediction_sha256,
        "destination_run_id": destination_run_id,
        "destination_status_path": str(destination_status),
        "destination_status_sha256": destination_status_sha256,
        "destination_prediction_path": str(destination_prediction),
        "destination_prediction_sha256": destination_prediction_sha256,
        "judge_artifacts": judge_artifacts,
    }


def validate_scoring_source_closure(
    summary: Mapping[str, Any], *, eval_root: Path, step: int
) -> dict[str, Any]:
    """Bind one headline to this exact TRUE1M-V2 arm and scorer byte tree."""

    eval_root = _validated_eval_root(eval_root)
    _evaluation_id(step)
    lexical_scoring_root = eval_root / f"matched/step{step}/scoring/coredev-official-v1"
    scoring_root = lexical_scoring_root.resolve()
    if (
        lexical_scoring_root.is_symlink()
        or not scoring_root.is_dir()
        or lexical_scoring_root != scoring_root
        or not scoring_root.is_relative_to(eval_root)
    ):
        raise RuntimeError(f"S{step} current scoring root is absent or unsafe")
    slices = summary.get("slices")
    if (
        not isinstance(slices, list)
        or len(slices) != len(DATASETS)
        or tuple(item.get("dataset") for item in slices if isinstance(item, Mapping))
        != DATASETS
    ):
        raise RuntimeError(f"S{step} scoring source slice order differs")
    for field, expected in (
        ("evaluation_id", _evaluation_id(step)),
        ("evaluation_contract", TOTAL_EVALUATION_ID),
        ("max_pixels", TRUE1M_MAX_PIXELS),
    ):
        if field in summary and summary.get(field) != expected:
            raise RuntimeError(f"S{step} pre-existing summary {field} differs")
    dataset_closures = [
        _receipt_source_closure(
            scoring_root=scoring_root,
            dataset=dataset,
            summary_slice=item,
            step=step,
        )
        for dataset, item in zip(DATASETS, slices, strict=True)
    ]
    return {
        "schema_version": SOURCE_CLOSURE_SCHEMA,
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "evaluation_id": _evaluation_id(step),
        "optimizer_step": step,
        "scoring_root": str(scoring_root),
        "datasets": dataset_closures,
    }


def _validate_summary_identity(summary: Mapping[str, Any], *, step: int) -> None:
    if summary.get("schema_version") != 1 or summary.get("phase") != "eval":
        raise RuntimeError(f"S{step} CoreDev summary schema/phase differs")
    if summary.get("model") != MODEL:
        raise RuntimeError(f"S{step} CoreDev summary model differs")
    slices = summary.get("slices")
    if (
        summary.get("sample_count") != SAMPLE_COUNT
        or summary.get("slice_count") != SLICE_COUNT
        or not isinstance(slices, Sequence)
        or isinstance(slices, (str, bytes))
        or len(slices) != SLICE_COUNT
    ):
        raise RuntimeError(f"S{step} CoreDev summary coverage differs")


def finalize_step(eval_root: Path, step: int) -> dict[str, Any]:
    """Extract, persist, and receipt one arm's complete frozen headline."""

    eval_root = _validated_eval_root(eval_root)
    summary_path = _summary_path(eval_root, step)
    summary = _load_json(summary_path)
    _validate_summary_identity(summary, step=step)
    source_closure = validate_scoring_source_closure(
        summary, eval_root=eval_root, step=step
    )
    headline = validate_headline(extract_coredev_macro_star(summary))
    summary.update(
        {
            "evaluation_id": _evaluation_id(step),
            "evaluation_contract": TOTAL_EVALUATION_ID,
            "max_pixels": TRUE1M_MAX_PIXELS,
            "scoring_source_closure": source_closure,
            "headline": headline,
        }
    )
    write_json_atomic(summary_path, summary)

    persisted = _load_json(summary_path)
    _validate_summary_identity(persisted, step=step)
    validate_headline(persisted.get("headline"))
    persisted_closure = validate_scoring_source_closure(
        persisted, eval_root=eval_root, step=step
    )
    if (
        persisted.get("evaluation_id") != _evaluation_id(step)
        or persisted.get("evaluation_contract") != TOTAL_EVALUATION_ID
        or persisted.get("max_pixels") != TRUE1M_MAX_PIXELS
        or persisted.get("scoring_source_closure") != persisted_closure
    ):
        raise RuntimeError(f"S{step} persisted headline identity differs")

    summary_sha256 = _sha256_file(summary_path)
    completion = {
        "schema_version": STEP_COMPLETION_SCHEMA,
        "status": "complete",
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "evaluation_id": _evaluation_id(step),
        "optimizer_step": step,
        "model": MODEL,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "sample_count": SAMPLE_COUNT,
        "slice_count": SLICE_COUNT,
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "headline_schema_version": COREDEV_MACRO_STAR_SCHEMA,
        "macro_star_percent": headline["macro_star_percent"],
        "blink_single_image_count": 180,
        "mmmu_single_image_count": 269,
        "mathverse_versions": list(MATHVERSE_VERSIONS),
        "scoring_source_closure_sha256": _canonical_sha256(persisted_closure),
    }
    completion_path = _step_completion_path(eval_root, step)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(completion_path, completion)
    return validate_step_completion(
        _load_json(completion_path), eval_root=eval_root, step=step
    )


def validate_step_completion(
    completion: object, *, eval_root: Path, step: int
) -> dict[str, Any]:
    eval_root = _validated_eval_root(eval_root)
    if not isinstance(completion, Mapping):
        raise RuntimeError(f"S{step} scoring completion is absent")
    expected = {
        "schema_version": STEP_COMPLETION_SCHEMA,
        "status": "complete",
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "evaluation_id": _evaluation_id(step),
        "optimizer_step": step,
        "model": MODEL,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "sample_count": SAMPLE_COUNT,
        "slice_count": SLICE_COUNT,
        "headline_schema_version": COREDEV_MACRO_STAR_SCHEMA,
        "blink_single_image_count": 180,
        "mmmu_single_image_count": 269,
        "mathverse_versions": list(MATHVERSE_VERSIONS),
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            raise RuntimeError(f"S{step} scoring completion {field} differs")
    summary_path = _summary_path(eval_root.resolve(), step)
    if completion.get("summary_path") != str(summary_path):
        raise RuntimeError(f"S{step} scoring completion summary path differs")
    if completion.get("summary_sha256") != _sha256_file(summary_path):
        raise RuntimeError(f"S{step} scoring completion summary digest differs")
    summary = _load_json(summary_path)
    _validate_summary_identity(summary, step=step)
    headline = validate_headline(summary.get("headline"))
    source_closure = validate_scoring_source_closure(
        summary, eval_root=eval_root, step=step
    )
    if summary.get("scoring_source_closure") != source_closure or completion.get(
        "scoring_source_closure_sha256"
    ) != _canonical_sha256(source_closure):
        raise RuntimeError(f"S{step} scoring source closure differs")
    macro = _percent(completion.get("macro_star_percent"), name=f"S{step} Macro*")
    _assert_close(macro, headline["macro_star_percent"], name=f"S{step} receipt Macro*")
    return dict(completion)


def finalize_all(eval_root: Path) -> dict[str, Any]:
    """Validate all four arms and write the aggregate formal completion receipt."""

    eval_root = _validated_eval_root(eval_root)
    arms: list[dict[str, Any]] = []
    for step in STEPS:
        completion_path = _step_completion_path(eval_root, step)
        completion = validate_step_completion(
            _load_json(completion_path), eval_root=eval_root, step=step
        )
        arms.append(
            {
                "optimizer_step": step,
                "evaluation_id": _evaluation_id(step),
                "step_completion_path": str(completion_path),
                "step_completion_sha256": _sha256_file(completion_path),
                "summary_path": completion["summary_path"],
                "summary_sha256": completion["summary_sha256"],
                "macro_star_percent": completion["macro_star_percent"],
                "scoring_source_closure_sha256": completion[
                    "scoring_source_closure_sha256"
                ],
            }
        )
    inference_source_closure = validate_inference_source_closure(eval_root)
    content = {
        "schema_version": AGGREGATE_COMPLETION_SCHEMA,
        "status": "complete",
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "model": MODEL,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "optimizer_steps": list(STEPS),
        "sample_count_per_step": SAMPLE_COUNT,
        "slice_count_per_step": SLICE_COUNT,
        "headline_contract": {
            "schema_version": COREDEV_MACRO_STAR_SCHEMA,
            "components": list(COREDEV_MACRO_STAR_COMPONENTS),
            "aggregation": AGGREGATION,
            "blink_single_image_count": 180,
            "mmmu_single_image_count": 269,
            "mathverse_versions": list(MATHVERSE_VERSIONS),
        },
        "inference_source_closure": inference_source_closure,
        "arms": arms,
    }
    result = {**content, "identity_sha256": _canonical_sha256(content)}
    output = (
        eval_root / "runtime/scoring-supervisor/matched-scoring-complete.json"
    ).resolve()
    write_json_atomic(output, result)
    return validate_aggregate_completion(_load_json(output), eval_root=eval_root)


def validate_aggregate_completion(
    completion: object, *, eval_root: Path
) -> dict[str, Any]:
    eval_root = _validated_eval_root(eval_root)
    if not isinstance(completion, Mapping):
        raise RuntimeError("aggregate scoring completion is absent")
    identity = completion.get("identity_sha256")
    unsigned = dict(completion)
    unsigned.pop("identity_sha256", None)
    if (
        not isinstance(identity, str)
        or _SHA256.fullmatch(identity) is None
        or identity != _canonical_sha256(unsigned)
    ):
        raise RuntimeError("aggregate scoring completion identity differs")
    expected = {
        "schema_version": AGGREGATE_COMPLETION_SCHEMA,
        "status": "complete",
        "evaluation_contract": TOTAL_EVALUATION_ID,
        "model": MODEL,
        "max_pixels": TRUE1M_MAX_PIXELS,
        "optimizer_steps": list(STEPS),
        "sample_count_per_step": SAMPLE_COUNT,
        "slice_count_per_step": SLICE_COUNT,
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            raise RuntimeError(f"aggregate scoring completion {field} differs")
    headline = completion.get("headline_contract")
    if not isinstance(headline, Mapping) or dict(headline) != {
        "schema_version": COREDEV_MACRO_STAR_SCHEMA,
        "components": list(COREDEV_MACRO_STAR_COMPONENTS),
        "aggregation": AGGREGATION,
        "blink_single_image_count": 180,
        "mmmu_single_image_count": 269,
        "mathverse_versions": list(MATHVERSE_VERSIONS),
    }:
        raise RuntimeError("aggregate scoring headline contract differs")
    inference_source_closure = validate_inference_source_closure(eval_root)
    if completion.get("inference_source_closure") != inference_source_closure:
        raise RuntimeError("aggregate scoring inference source closure differs")
    arms = completion.get("arms")
    if not isinstance(arms, list) or [
        arm.get("optimizer_step") for arm in arms
    ] != list(STEPS):
        raise RuntimeError("aggregate scoring arm order differs")
    for step, arm in zip(STEPS, arms, strict=True):
        if not isinstance(arm, Mapping):
            raise RuntimeError(f"S{step} aggregate scoring arm is malformed")
        completion_path = _step_completion_path(eval_root, step)
        if (
            arm.get("evaluation_id") != _evaluation_id(step)
            or arm.get("step_completion_path") != str(completion_path)
            or arm.get("step_completion_sha256") != _sha256_file(completion_path)
        ):
            raise RuntimeError(f"S{step} aggregate scoring receipt binding differs")
        step_completion = validate_step_completion(
            _load_json(completion_path), eval_root=eval_root, step=step
        )
        for field in (
            "summary_path",
            "summary_sha256",
            "macro_star_percent",
            "scoring_source_closure_sha256",
        ):
            if arm.get(field) != step_completion[field]:
                raise RuntimeError(f"S{step} aggregate scoring {field} differs")
    return dict(completion)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--step", type=int, choices=STEPS)
    mode.add_argument("--finalize-all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = (
        finalize_all(args.eval_root)
        if args.finalize_all
        else finalize_step(args.eval_root, args.step)
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
