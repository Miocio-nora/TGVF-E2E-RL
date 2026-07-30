#!/usr/bin/env python3
"""Summarize the pinned RP66 full-867 answer-utility checkpoint curve.

The inputs are four completed semantic-rescore directories.  This tool is
deliberately read-only with respect to those directories: it follows and
verifies the semantic-overlay -> generation-artifact hash chain, checks the
actual production checkpoint step, and then computes the scaling curve.

The step-2000 ``image_only`` records are the single shared baseline.  They are
never relabelled as predictions from another checkpoint; comparisons at steps
500/1000/1500 align that explicitly identified baseline by ``sample_id``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tomllib
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rp66-answer-utility-full867-scaling-v1"
SEMANTIC_SCHEMA_VERSION = "answer-utility-semantic-rescore-v1"
SEMANTIC_RECORD_SCHEMA_VERSION = "answer-utility-semantic-rescore-record-v1"
GENERATION_RECORD_SCHEMAS = {
    "answer-utility-instruct-evaluation-record-v1",
    "answer-utility-instruct-evaluation-record-v2",
}
GENERATION_IDENTITY_SCHEMA = "answer-utility-instruct-evaluation-v2"
EXPECTED_STEPS = (500, 1000, 1500, 2000)
CHECKPOINT_COMPARISON_PLAN = (
    ("step1000_vs_step0500", 1000, 500, ("vs_step0500", "adjacent")),
    ("step1500_vs_step0500", 1500, 500, ("vs_step0500",)),
    ("step2000_vs_step0500", 2000, 500, ("vs_step0500",)),
    ("step1500_vs_step1000", 1500, 1000, ("adjacent",)),
    ("step2000_vs_step1500", 2000, 1500, ("adjacent",)),
)
EXPECTED_SAMPLE_COUNT = 867
EXPECTED_RP66_RUN_IDENTITY_SHA256 = (
    "97ccfd849e1d66cdd57be805c27524fa97ca60973e5be45d6d060acd5bc54e53"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "534f5b1e648d0bca2b1ea2ff02f81e1fb7abbb456f16faacbb118ca94f7306b0"
)
EXPECTED_ORDERED_GROUP_MANIFEST_SHA256 = (
    "31cce579e919dccf3ba2702e09db3a7b2cfa65e412db079c7dcdcb15dcddbe78"
)
EXPECTED_CLAIM_SCOPE = "diagnostic_semantic_overlay_not_formal_pilot"
CORRECT_D_ARM = "image_correct_D"
BASELINE_ARM = "image_only"
EXPECTED_ARM_CONTRACTS = {
    CORRECT_D_ARM: {
        "d": "correct_target_stage1_main_and_all_deepstack",
        "oracle_target_transcript": True,
        "prompt": "image_question_plus_oracle_target_tool_transcript",
        "source_image": True,
    },
    BASELINE_ARM: {
        "d": "absent",
        "oracle_target_transcript": False,
        "prompt": "question_only_no_tool_schema",
        "source_image": True,
    },
}


@dataclass(frozen=True, slots=True)
class _Overlay:
    root: Path
    step: int
    candidate_id: str
    run_identity_sha256: str
    generation_identity_sha256: str
    semantic_manifest_sha256: str
    semantic_manifest_file_sha256: str
    semantic_summary_file_sha256: str
    semantic_records_file_sha256: str
    generation_identity_file_sha256: str
    generation_records_file_sha256: str
    source_evaluation_config_sha256: str
    judge_binding_sha256: str
    scoring_contract_version: str
    selected_samples: tuple[tuple[str, str], ...]
    by_arm: Mapping[str, Mapping[str, bool]]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _json_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    return _mapping(value, name=name)


def _jsonl(path: Path, *, name: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank line in {name}: {path}:{line_number}")
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"invalid JSON in {name}: {path}:{line_number}"
            ) from error
        rows.append(_mapping(value, name=f"{name} row {line_number}"))
    return tuple(rows)


def _require_sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _verify_bound_file(
    root: Path,
    binding_value: object,
    *,
    name: str,
    expected_rows: int | None = None,
) -> Path:
    binding = _mapping(binding_value, name=f"{name} binding")
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} binding path must be non-empty and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} binding escapes its root") from error
    expected_sha = _require_sha(binding.get("sha256"), name=f"{name} SHA256")
    observed_sha = _file_sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(f"{name} SHA256 differs: {path}")
    if expected_rows is not None and binding.get("rows") != expected_rows:
        raise ValueError(
            f"{name} row binding differs: expected {expected_rows}, "
            f"observed {binding.get('rows')!r}"
        )
    return path


def _metric_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(row["final_score"]["correct"] is True for row in records)
    total = len(records)
    return {
        "total": total,
        "diagnostic_final_correct": correct,
        "diagnostic_final_incorrect": total - correct,
        "diagnostic_semantic_accuracy": correct / total,
    }


def _verify_metric(
    value: object,
    records: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> None:
    observed = _mapping(value, name=name)
    expected = _metric_from_records(records)
    for key in (
        "total",
        "diagnostic_final_correct",
        "diagnostic_final_incorrect",
    ):
        if observed.get(key) != expected[key]:
            raise ValueError(
                f"{name}.{key} differs: expected {expected[key]!r}, "
                f"observed {observed.get(key)!r}"
            )
    accuracy = observed.get("diagnostic_semantic_accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise ValueError(f"{name}.diagnostic_semantic_accuracy must be numeric")
    if not math.isclose(float(accuracy), expected["diagnostic_semantic_accuracy"], abs_tol=1e-15):
        raise ValueError(f"{name}.diagnostic_semantic_accuracy differs")


def _load_overlay(
    root_value: str | Path,
    *,
    expected_step: int,
    expected_arms: tuple[str, ...],
    expected_samples: int,
    expected_training_run_identity_sha256: str,
    expected_data_manifest_sha256: str,
    expected_ordered_group_manifest_sha256: str,
) -> _Overlay:
    root = Path(root_value).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _json_mapping(manifest_path, name="semantic manifest")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "run_identity_sha256",
        "identity",
        "files",
        "unique_judge_requests",
        "judge_consumer_count",
        "manifest_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError(f"semantic manifest fields differ: {root}")
    if manifest.get("schema_version") != SEMANTIC_SCHEMA_VERSION:
        raise ValueError(f"semantic manifest schema differs: {root}")
    if manifest.get("status") != "complete":
        raise ValueError(f"semantic overlay is not complete: {root}")
    declared_manifest_sha = _require_sha(
        manifest.get("manifest_sha256"), name="semantic manifest identity SHA256"
    )
    manifest_without_sha = dict(manifest)
    del manifest_without_sha["manifest_sha256"]
    if _sha256_bytes(_canonical_json_bytes(manifest_without_sha)) != declared_manifest_sha:
        raise ValueError(f"semantic manifest identity SHA256 differs: {root}")

    run_identity_sha = _require_sha(
        manifest.get("run_identity_sha256"), name="semantic run identity SHA256"
    )
    identity = _mapping(manifest.get("identity"), name="semantic identity")
    if _sha256_bytes(_canonical_json_bytes(identity)) != run_identity_sha:
        raise ValueError(f"semantic run identity SHA256 differs: {root}")
    if identity.get("schema_version") != SEMANTIC_SCHEMA_VERSION:
        raise ValueError(f"semantic identity schema differs: {root}")
    if identity.get("evaluation_data_manifest_sha256") != expected_data_manifest_sha256:
        raise ValueError(f"semantic evaluation data manifest differs: {root}")
    if identity.get("deterministic_scoring_contract_version") is None:
        raise ValueError(f"semantic scoring contract is missing: {root}")

    source_config_path_value = identity.get("source_evaluation_config_path")
    if not isinstance(source_config_path_value, str) or not source_config_path_value:
        raise ValueError(f"semantic source evaluation config path is missing: {root}")
    source_config_path = Path(source_config_path_value).expanduser().resolve()
    source_config_sha = _require_sha(
        identity.get("source_evaluation_config_sha256"),
        name="source evaluation config SHA256",
    )
    if _file_sha256(source_config_path) != source_config_sha:
        raise ValueError(f"source evaluation config SHA256 differs: {root}")
    with source_config_path.open("rb") as handle:
        source_config = tomllib.load(handle)
    artifact_config = _mapping(source_config.get("artifact"), name="artifact config")
    if artifact_config.get("expected_global_step") != expected_step:
        raise ValueError(
            f"source config step differs for requested step {expected_step}: {root}"
        )
    if (
        artifact_config.get("expected_run_identity_sha256")
        != expected_training_run_identity_sha256
    ):
        raise ValueError(f"source config training run identity differs: {root}")
    evaluation_config = _mapping(
        source_config.get("evaluation"), name="evaluation config"
    )
    if (
        evaluation_config.get("ordered_group_manifest_sha256")
        != expected_ordered_group_manifest_sha256
    ):
        raise ValueError(f"source config ordered group manifest differs: {root}")

    files = _mapping(manifest.get("files"), name="semantic files")
    if set(files) != {"blind_requests", "judge_evidence", "overlay_records", "summary"}:
        raise ValueError(f"semantic file bindings differ: {root}")
    expected_record_count = expected_samples * len(expected_arms)
    records_path = _verify_bound_file(
        root,
        files["overlay_records"],
        name="semantic records",
        expected_rows=expected_record_count,
    )
    summary_path = _verify_bound_file(root, files["summary"], name="semantic summary")
    _verify_bound_file(root, files["blind_requests"], name="blind requests")
    _verify_bound_file(root, files["judge_evidence"], name="judge evidence")
    records = _jsonl(records_path, name="semantic records")
    if len(records) != expected_record_count:
        raise ValueError(
            f"semantic record count differs for step {expected_step}: "
            f"expected {expected_record_count}, observed {len(records)}"
        )

    generation_sources = identity.get("generation_sources")
    if not isinstance(generation_sources, list) or len(generation_sources) != 1:
        raise ValueError(
            f"step {expected_step} overlay must bind exactly one generation source"
        )
    source_binding = _mapping(generation_sources[0], name="generation source binding")
    generation_root_value = source_binding.get("root")
    if not isinstance(generation_root_value, str) or not generation_root_value:
        raise ValueError("generation source root is missing")
    generation_root = Path(generation_root_value).expanduser().resolve()
    generation_identity_path = generation_root / "identity.json"
    if (
        _file_sha256(generation_identity_path)
        != source_binding.get("identity_file_sha256")
    ):
        raise ValueError(f"generation identity file SHA256 differs: {generation_root}")
    generation_outer = _json_mapping(
        generation_identity_path, name="generation identity"
    )
    if set(generation_outer) != {"schema_version", "identity_sha256", "identity"}:
        raise ValueError(f"generation identity wrapper fields differ: {generation_root}")
    generation_identity = _mapping(
        generation_outer.get("identity"), name="generation identity payload"
    )
    generation_identity_sha = _require_sha(
        generation_outer.get("identity_sha256"), name="generation identity SHA256"
    )
    if _sha256_bytes(_canonical_json_bytes(generation_identity)) != generation_identity_sha:
        raise ValueError(f"generation identity SHA256 differs: {generation_root}")
    if source_binding.get("generation_identity_sha256") != generation_identity_sha:
        raise ValueError(f"semantic/generation identity binding differs: {root}")
    if generation_identity.get("schema_version") != GENERATION_IDENTITY_SCHEMA:
        raise ValueError(f"generation identity schema differs: {generation_root}")
    candidate_id = generation_identity.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"generation candidate ID is missing: {generation_root}")
    if source_binding.get("candidate_id") != candidate_id:
        raise ValueError(f"generation candidate ID binding differs: {root}")
    if generation_identity.get("candidate_kind") != "production_source":
        raise ValueError(f"candidate is not a production source: {generation_root}")
    for step_key in ("candidate_global_step", "production_source_global_step"):
        if generation_identity.get(step_key) != expected_step:
            raise ValueError(
                f"{step_key} differs: expected {expected_step}, "
                f"observed {generation_identity.get(step_key)!r}"
            )
    for run_key in (
        "candidate_training_run_identity_sha256",
        "production_source_run_identity_sha256",
    ):
        if generation_identity.get(run_key) != expected_training_run_identity_sha256:
            raise ValueError(f"{run_key} differs at step {expected_step}")
    if generation_identity.get("source_evaluation_config_sha256") != source_config_sha:
        raise ValueError(f"generation/source config SHA256 differs: {generation_root}")
    if generation_identity.get("data_manifest_sha256") != expected_data_manifest_sha256:
        raise ValueError(f"generation data manifest differs: {generation_root}")
    if (
        generation_identity.get("ordered_group_manifest_sha256")
        != expected_ordered_group_manifest_sha256
    ):
        raise ValueError(f"generation ordered manifest differs: {generation_root}")
    if generation_identity.get("arms") != list(expected_arms):
        raise ValueError(
            f"generation arms differ at step {expected_step}: "
            f"expected {list(expected_arms)!r}, "
            f"observed {generation_identity.get('arms')!r}"
        )
    arm_contracts = _mapping(
        generation_identity.get("arm_contracts"), name="generation arm contracts"
    )
    for arm in expected_arms:
        if arm_contracts.get(arm) != EXPECTED_ARM_CONTRACTS[arm]:
            raise ValueError(f"generation arm contract differs for {arm}")

    selected_raw = generation_identity.get("ordered_selected_samples")
    if not isinstance(selected_raw, list) or len(selected_raw) != expected_samples:
        raise ValueError(
            f"ordered selected sample count differs at step {expected_step}: "
            f"expected {expected_samples}"
        )
    selected: list[tuple[str, str]] = []
    for index, item_value in enumerate(selected_raw):
        item = _mapping(item_value, name=f"selected sample {index}")
        sample_id = item.get("sample_id")
        content_sha = item.get("sample_content_sha256")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"selected sample ID is invalid at index {index}")
        selected.append(
            (
                sample_id,
                _require_sha(content_sha, name=f"selected sample {index} content SHA256"),
            )
        )
    if len({sample_id for sample_id, _sha in selected}) != expected_samples:
        raise ValueError(f"selected sample IDs are not unique at step {expected_step}")

    generation_records_path = generation_root / "records.jsonl"
    generation_summary_path = generation_root / "summary.json"
    if _file_sha256(generation_records_path) != source_binding.get("records_file_sha256"):
        raise ValueError(f"generation records file SHA256 differs: {generation_root}")
    if _file_sha256(generation_summary_path) != source_binding.get("summary_file_sha256"):
        raise ValueError(f"generation summary file SHA256 differs: {generation_root}")
    generation_records = _jsonl(
        generation_records_path, name="generation records"
    )
    if len(generation_records) != expected_record_count:
        raise ValueError(f"generation record count differs at step {expected_step}")
    if source_binding.get("record_count") != expected_record_count:
        raise ValueError(f"generation source record binding differs at step {expected_step}")
    generation_summary = _json_mapping(
        generation_summary_path, name="generation summary"
    )
    if (
        generation_summary.get("status") != "complete"
        or generation_summary.get("run_identity_sha256") != generation_identity_sha
        or generation_summary.get("record_count") != expected_record_count
        or generation_summary.get("records_jsonl_sha256")
        != _file_sha256(generation_records_path)
    ):
        raise ValueError(f"generation summary does not bind completed records: {generation_root}")

    expected_keys = tuple(
        (sample_id, arm)
        for sample_id, _content_sha in selected
        for arm in expected_arms
    )
    generation_keys: list[tuple[str, str]] = []
    generation_record_sha_by_key: dict[tuple[str, str], str] = {}
    for index, row in enumerate(generation_records):
        if row.get("schema_version") not in GENERATION_RECORD_SCHEMAS:
            raise ValueError(f"generation record schema differs at row {index}")
        if row.get("run_identity_sha256") != generation_identity_sha:
            raise ValueError(f"generation record identity differs at row {index}")
        if row.get("candidate_id") != candidate_id:
            raise ValueError(f"generation record candidate differs at row {index}")
        key = (str(row.get("sample_id")), str(row.get("arm")))
        generation_keys.append(key)
        generation_record_sha_by_key[key] = _sha256_bytes(_canonical_json_bytes(row))
    if tuple(generation_keys) != expected_keys:
        raise ValueError(f"generation record ordering/selection differs at step {expected_step}")

    overlay_keys: list[tuple[str, str]] = []
    by_arm_rows: dict[str, list[Mapping[str, Any]]] = {
        arm: [] for arm in expected_arms
    }
    by_arm_results: dict[str, dict[str, bool]] = {
        arm: {} for arm in expected_arms
    }
    source_label = source_binding.get("label")
    for index, row in enumerate(records):
        if row.get("schema_version") != SEMANTIC_RECORD_SCHEMA_VERSION:
            raise ValueError(f"semantic record schema differs at row {index}")
        if row.get("run_identity_sha256") != run_identity_sha:
            raise ValueError(f"semantic record run identity differs at row {index}")
        if row.get("candidate_id") != candidate_id:
            raise ValueError(f"semantic record candidate differs at row {index}")
        if row.get("source_generation_identity_sha256") != generation_identity_sha:
            raise ValueError(f"semantic record generation identity differs at row {index}")
        if row.get("source_label") != source_label:
            raise ValueError(f"semantic record source label differs at row {index}")
        if Path(str(row.get("source_root"))).expanduser().resolve() != generation_root:
            raise ValueError(f"semantic record source root differs at row {index}")
        sample_id = row.get("sample_id")
        arm = row.get("arm")
        if not isinstance(sample_id, str) or arm not in expected_arms:
            raise ValueError(f"semantic record sample/arm differs at row {index}")
        key = (sample_id, str(arm))
        overlay_keys.append(key)
        if row.get("source_record_sha256") != generation_record_sha_by_key.get(key):
            raise ValueError(f"semantic/source record SHA256 differs at row {index}")
        overlay_sha = _require_sha(
            row.get("overlay_record_sha256"),
            name=f"semantic overlay record {index} SHA256",
        )
        overlay_identity = dict(row)
        del overlay_identity["overlay_record_sha256"]
        if _sha256_bytes(_canonical_json_bytes(overlay_identity)) != overlay_sha:
            raise ValueError(f"semantic overlay record SHA256 differs at row {index}")
        final_score = _mapping(row.get("final_score"), name=f"final score row {index}")
        if type(final_score.get("correct")) is not bool:
            raise ValueError(f"final correctness is not boolean at row {index}")
        if final_score.get("scope") != EXPECTED_CLAIM_SCOPE:
            raise ValueError(f"final score scope differs at row {index}")
        by_arm_rows[str(arm)].append(row)
        by_arm_results[str(arm)][sample_id] = bool(final_score["correct"])
    if tuple(overlay_keys) != expected_keys:
        raise ValueError(f"semantic record ordering/selection differs at step {expected_step}")

    summary = _json_mapping(summary_path, name="semantic summary")
    if (
        summary.get("schema_version") != SEMANTIC_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("run_identity_sha256") != run_identity_sha
        or summary.get("claim_scope") != EXPECTED_CLAIM_SCOPE
    ):
        raise ValueError(f"semantic summary identity/status differs: {root}")
    _verify_metric(summary.get("overall"), records, name="summary.overall")
    summary_by_arm = _mapping(summary.get("by_arm"), name="summary.by_arm")
    if set(summary_by_arm) != set(expected_arms):
        raise ValueError(f"semantic summary arms differ at step {expected_step}")
    for arm in expected_arms:
        _verify_metric(
            summary_by_arm.get(arm), by_arm_rows[arm], name=f"summary.by_arm.{arm}"
        )
    summary_by_candidate = _mapping(
        summary.get("by_candidate"), name="summary.by_candidate"
    )
    if set(summary_by_candidate) != {candidate_id}:
        raise ValueError(f"semantic summary candidate differs at step {expected_step}")
    candidate_summary = _mapping(
        summary_by_candidate[candidate_id], name="candidate summary"
    )
    _verify_metric(
        candidate_summary.get("overall"), records, name="candidate summary overall"
    )
    candidate_by_arm = _mapping(
        candidate_summary.get("by_arm"), name="candidate summary by_arm"
    )
    if set(candidate_by_arm) != set(expected_arms):
        raise ValueError(f"candidate summary arms differ at step {expected_step}")
    for arm in expected_arms:
        _verify_metric(
            candidate_by_arm.get(arm),
            by_arm_rows[arm],
            name=f"candidate summary by_arm.{arm}",
        )

    judge = _mapping(identity.get("judge"), name="semantic judge binding")
    return _Overlay(
        root=root,
        step=expected_step,
        candidate_id=candidate_id,
        run_identity_sha256=run_identity_sha,
        generation_identity_sha256=generation_identity_sha,
        semantic_manifest_sha256=declared_manifest_sha,
        semantic_manifest_file_sha256=_file_sha256(manifest_path),
        semantic_summary_file_sha256=_file_sha256(summary_path),
        semantic_records_file_sha256=_file_sha256(records_path),
        generation_identity_file_sha256=_file_sha256(generation_identity_path),
        generation_records_file_sha256=_file_sha256(generation_records_path),
        source_evaluation_config_sha256=source_config_sha,
        judge_binding_sha256=_sha256_bytes(_canonical_json_bytes(judge)),
        scoring_contract_version=str(
            identity["deterministic_scoring_contract_version"]
        ),
        selected_samples=tuple(selected),
        by_arm={arm: dict(values) for arm, values in by_arm_results.items()},
    )


def _wilson_interval(correct: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence": confidence,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _accuracy(values: Mapping[str, bool]) -> dict[str, Any]:
    total = len(values)
    correct = sum(value is True for value in values.values())
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "wilson_95": _wilson_interval(correct, total),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _paired_bootstrap_interval(
    *,
    wins: int,
    losses: int,
    ties: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    total = wins + losses + ties
    rng = random.Random(seed)
    win_boundary = wins / total
    loss_boundary = (wins + losses) / total
    estimates: list[float] = []
    for _replicate in range(replicates):
        delta_sum = 0
        for _sample in range(total):
            draw = rng.random()
            if draw < win_boundary:
                delta_sum += 1
            elif draw < loss_boundary:
                delta_sum -= 1
        estimates.append(delta_sum / total)
    estimates.sort()
    return {
        "method": "paired_nonparametric_percentile",
        "confidence": 0.95,
        "replicates": replicates,
        "seed": seed,
        "lower": _quantile(estimates, 0.025),
        "upper": _quantile(estimates, 0.975),
    }


def _mcnemar_exact_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * (numerator / (1 << discordant)))


def _paired_effect(
    treatment: Mapping[str, bool],
    baseline: Mapping[str, bool],
    *,
    step: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(treatment) != set(baseline):
        missing = sorted(set(baseline) - set(treatment))[:3]
        extra = sorted(set(treatment) - set(baseline))[:3]
        raise ValueError(
            f"paired sample IDs differ at step {step}; missing={missing}, extra={extra}"
        )
    ordered_ids = sorted(baseline)
    both_correct = sum(treatment[key] and baseline[key] for key in ordered_ids)
    wins = sum(treatment[key] and not baseline[key] for key in ordered_ids)
    losses = sum(not treatment[key] and baseline[key] for key in ordered_ids)
    both_incorrect = len(ordered_ids) - both_correct - wins - losses
    treatment_accuracy = (both_correct + wins) / len(ordered_ids)
    baseline_accuracy = (both_correct + losses) / len(ordered_ids)
    return {
        "baseline_arm": BASELINE_ARM,
        "baseline_provenance_global_step": 2000,
        "baseline_prediction_identity_relabelled": False,
        "baseline_reuse_scope": (
            "shared checkpoint-independent protocol baseline; step-2000 provenance "
            "retained and aligned by sample_id"
        ),
        "treatment_arm": CORRECT_D_ARM,
        "paired_samples": len(ordered_ids),
        "both_correct": both_correct,
        "treatment_only_correct_wins": wins,
        "baseline_only_correct_losses": losses,
        "both_incorrect": both_incorrect,
        "ties": both_correct + both_incorrect,
        "treatment_accuracy": treatment_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_delta": treatment_accuracy - baseline_accuracy,
        "paired_bootstrap_95": _paired_bootstrap_interval(
            wins=wins,
            losses=losses,
            ties=both_correct + both_incorrect,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + step,
        ),
        "mcnemar": {
            "method": "exact_two_sided_binomial",
            "discordant_pairs": wins + losses,
            "p_value": _mcnemar_exact_p_value(wins, losses),
        },
    }


def _checkpoint_paired_effect(
    treatment: Mapping[str, bool],
    control: Mapping[str, bool],
    *,
    comparison_id: str,
    comparison_roles: tuple[str, ...],
    treatment_step: int,
    control_step: int,
    treatment_candidate_id: str,
    control_candidate_id: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(treatment) != set(control):
        missing = sorted(set(control) - set(treatment))[:3]
        extra = sorted(set(treatment) - set(control))[:3]
        raise ValueError(
            f"checkpoint-paired sample IDs differ for {comparison_id}; "
            f"missing={missing}, extra={extra}"
        )
    ordered_ids = sorted(control)
    both_correct = sum(treatment[key] and control[key] for key in ordered_ids)
    wins = sum(treatment[key] and not control[key] for key in ordered_ids)
    losses = sum(not treatment[key] and control[key] for key in ordered_ids)
    both_incorrect = len(ordered_ids) - both_correct - wins - losses
    ties = both_correct + both_incorrect
    treatment_accuracy = (both_correct + wins) / len(ordered_ids)
    control_accuracy = (both_correct + losses) / len(ordered_ids)
    comparison_seed = (
        bootstrap_seed + 100_000_000 + treatment_step * 10_000 + control_step
    )
    return {
        "comparison_id": comparison_id,
        "comparison_roles": list(comparison_roles),
        "arm": CORRECT_D_ARM,
        "treatment_global_step": treatment_step,
        "control_global_step": control_step,
        "treatment_candidate_id": treatment_candidate_id,
        "control_candidate_id": control_candidate_id,
        "paired_samples": len(ordered_ids),
        "both_correct": both_correct,
        "treatment_only_correct_wins": wins,
        "control_only_correct_losses": losses,
        "both_incorrect": both_incorrect,
        "ties": ties,
        "treatment_accuracy": treatment_accuracy,
        "control_accuracy": control_accuracy,
        "accuracy_delta": treatment_accuracy - control_accuracy,
        "paired_bootstrap_95": _paired_bootstrap_interval(
            wins=wins,
            losses=losses,
            ties=ties,
            replicates=bootstrap_replicates,
            seed=comparison_seed,
        ),
        "mcnemar": {
            "method": "exact_two_sided_binomial",
            "discordant_pairs": wins + losses,
            "p_value": _mcnemar_exact_p_value(wins, losses),
        },
    }


def summarize_scaling(
    roots_by_step: Mapping[int, str | Path],
    *,
    expected_samples: int = EXPECTED_SAMPLE_COUNT,
    expected_training_run_identity_sha256: str = EXPECTED_RP66_RUN_IDENTITY_SHA256,
    expected_data_manifest_sha256: str = EXPECTED_DATA_MANIFEST_SHA256,
    expected_ordered_group_manifest_sha256: str = EXPECTED_ORDERED_GROUP_MANIFEST_SHA256,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_730,
) -> dict[str, Any]:
    if set(roots_by_step) != set(EXPECTED_STEPS):
        raise ValueError(
            f"exactly steps {list(EXPECTED_STEPS)} are required; "
            f"observed {sorted(roots_by_step)}"
        )
    overlays: dict[int, _Overlay] = {}
    for step in EXPECTED_STEPS:
        arms = (
            (BASELINE_ARM, CORRECT_D_ARM)
            if step == 2000
            else (CORRECT_D_ARM,)
        )
        overlays[step] = _load_overlay(
            roots_by_step[step],
            expected_step=step,
            expected_arms=arms,
            expected_samples=expected_samples,
            expected_training_run_identity_sha256=(
                expected_training_run_identity_sha256
            ),
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_ordered_group_manifest_sha256=(
                expected_ordered_group_manifest_sha256
            ),
        )

    reference_selection = overlays[500].selected_samples
    for step, overlay in overlays.items():
        if overlay.selected_samples != reference_selection:
            raise ValueError(f"ordered selected samples differ at step {step}")
    judge_bindings = {overlay.judge_binding_sha256 for overlay in overlays.values()}
    if len(judge_bindings) != 1:
        raise ValueError("semantic judge bindings differ across checkpoint overlays")
    scoring_contracts = {
        overlay.scoring_contract_version for overlay in overlays.values()
    }
    if len(scoring_contracts) != 1:
        raise ValueError("deterministic scoring contracts differ across overlays")

    baseline_values = overlays[2000].by_arm[BASELINE_ARM]
    baseline_metric = _accuracy(baseline_values)
    curve: list[dict[str, Any]] = []
    step500_accuracy: float | None = None
    previous_accuracy: float | None = None
    for step in EXPECTED_STEPS:
        values = overlays[step].by_arm[CORRECT_D_ARM]
        metric = _accuracy(values)
        accuracy = float(metric["accuracy"])
        if step500_accuracy is None:
            step500_accuracy = accuracy
        point = {
            "global_step": step,
            "candidate_id": overlays[step].candidate_id,
            "image_correct_D": metric,
            "accuracy_delta_from_step0500": accuracy - step500_accuracy,
            "accuracy_delta_from_previous_checkpoint": (
                None if previous_accuracy is None else accuracy - previous_accuracy
            ),
            "paired_vs_shared_image_only": _paired_effect(
                values,
                baseline_values,
                step=step,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        }
        curve.append(point)
        previous_accuracy = accuracy

    checkpoint_comparisons = [
        _checkpoint_paired_effect(
            overlays[treatment_step].by_arm[CORRECT_D_ARM],
            overlays[control_step].by_arm[CORRECT_D_ARM],
            comparison_id=comparison_id,
            comparison_roles=comparison_roles,
            treatment_step=treatment_step,
            control_step=control_step,
            treatment_candidate_id=overlays[treatment_step].candidate_id,
            control_candidate_id=overlays[control_step].candidate_id,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for (
            comparison_id,
            treatment_step,
            control_step,
            comparison_roles,
        ) in CHECKPOINT_COMPARISON_PLAN
    ]

    source_bindings = [
        {
            "global_step": step,
            "semantic_overlay_root": str(overlays[step].root),
            "semantic_run_identity_sha256": overlays[step].run_identity_sha256,
            "semantic_manifest_sha256": overlays[step].semantic_manifest_sha256,
            "semantic_manifest_file_sha256": overlays[
                step
            ].semantic_manifest_file_sha256,
            "semantic_summary_file_sha256": overlays[
                step
            ].semantic_summary_file_sha256,
            "semantic_records_file_sha256": overlays[
                step
            ].semantic_records_file_sha256,
            "generation_identity_sha256": overlays[
                step
            ].generation_identity_sha256,
            "generation_identity_file_sha256": overlays[
                step
            ].generation_identity_file_sha256,
            "generation_records_file_sha256": overlays[
                step
            ].generation_records_file_sha256,
            "source_evaluation_config_sha256": overlays[
                step
            ].source_evaluation_config_sha256,
        }
        for step in EXPECTED_STEPS
    ]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "summarizer_file_sha256": _file_sha256(Path(__file__).resolve()),
        "training_run_identity_sha256": expected_training_run_identity_sha256,
        "evaluation_data_manifest_sha256": expected_data_manifest_sha256,
        "ordered_group_manifest_sha256": expected_ordered_group_manifest_sha256,
        "ordered_sample_count": expected_samples,
        "ordered_sample_selection_sha256": _sha256_bytes(
            _canonical_json_bytes(reference_selection)
        ),
        "checkpoint_steps": list(EXPECTED_STEPS),
        "checkpoint_comparison_plan": [
            {
                "comparison_id": comparison_id,
                "treatment_global_step": treatment_step,
                "control_global_step": control_step,
                "comparison_roles": list(comparison_roles),
            }
            for (
                comparison_id,
                treatment_step,
                control_step,
                comparison_roles,
            ) in CHECKPOINT_COMPARISON_PLAN
        ],
        "correct_D_arm": CORRECT_D_ARM,
        "shared_baseline": {
            "arm": BASELINE_ARM,
            "generation_provenance_global_step": 2000,
            "prediction_identity_relabelled": False,
        },
        "judge_binding_sha256": next(iter(judge_bindings)),
        "deterministic_scoring_contract_version": next(iter(scoring_contracts)),
        "source_bindings": source_bindings,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
            "shared_baseline_seed_rule": "base_seed_plus_global_step",
            "checkpoint_comparison_seed_rule": (
                "base_seed_plus_100000000_plus_treatment_step_times_10000_"
                "plus_control_step"
            ),
        },
    }
    identity_sha = _sha256_bytes(_canonical_json_bytes(identity))
    payload_without_sha = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "identity_sha256": identity_sha,
        "identity": identity,
        "image_only_baseline": {
            **baseline_metric,
            "generation_provenance_global_step": 2000,
            "candidate_id": overlays[2000].candidate_id,
            "prediction_identity_relabelled": False,
        },
        "checkpoint_curve": curve,
        "checkpoint_to_checkpoint_paired_comparisons": checkpoint_comparisons,
    }
    return {
        **payload_without_sha,
        "summary_sha256": _sha256_bytes(_canonical_json_bytes(payload_without_sha)),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["image_only_baseline"]
    lines = [
        "# RP66 full-867 answer-utility checkpoint scaling",
        "",
        (
            f"Shared `image_only` baseline (step-2000 provenance): "
            f"{baseline['correct']}/{baseline['total']} = "
            f"{100.0 * baseline['accuracy']:.2f}%."
        ),
        "",
        (
            "| Step | `image_correct_D` | Accuracy | Wilson 95% CI | "
            "Paired delta vs image-only | Bootstrap 95% CI | Wins/Losses | "
            "McNemar p |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in summary["checkpoint_curve"]:
        metric = point["image_correct_D"]
        wilson = metric["wilson_95"]
        paired = point["paired_vs_shared_image_only"]
        bootstrap = paired["paired_bootstrap_95"]
        lines.append(
            "| {step} | {correct}/{total} | {accuracy:.2f}% | "
            "[{lower:.2f}%, {upper:.2f}%] | {delta:+.2f} pp | "
            "[{boot_lower:+.2f}, {boot_upper:+.2f}] pp | {wins}/{losses} | "
            "{p:.4g} |".format(
                step=point["global_step"],
                correct=metric["correct"],
                total=metric["total"],
                accuracy=100.0 * metric["accuracy"],
                lower=100.0 * wilson["lower"],
                upper=100.0 * wilson["upper"],
                delta=100.0 * paired["accuracy_delta"],
                boot_lower=100.0 * bootstrap["lower"],
                boot_upper=100.0 * bootstrap["upper"],
                wins=paired["treatment_only_correct_wins"],
                losses=paired["baseline_only_correct_losses"],
                p=paired["mcnemar"]["p_value"],
            )
        )
    lines.extend(
        [
            "",
            "## Checkpoint-to-checkpoint paired comparisons",
            "",
            (
                "| Comparison | Roles | Treatment/control accuracy | Delta | "
                "Bootstrap 95% CI | Wins/Losses/Ties | McNemar p |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for paired in summary["checkpoint_to_checkpoint_paired_comparisons"]:
        bootstrap = paired["paired_bootstrap_95"]
        lines.append(
            "| {treatment} vs {control} | {roles} | "
            "{treatment_accuracy:.2f}% / {control_accuracy:.2f}% | "
            "{delta:+.2f} pp | [{lower:+.2f}, {upper:+.2f}] pp | "
            "{wins}/{losses}/{ties} | {p:.4g} |".format(
                treatment=paired["treatment_global_step"],
                control=paired["control_global_step"],
                roles=", ".join(paired["comparison_roles"]),
                treatment_accuracy=100.0 * paired["treatment_accuracy"],
                control_accuracy=100.0 * paired["control_accuracy"],
                delta=100.0 * paired["accuracy_delta"],
                lower=100.0 * bootstrap["lower"],
                upper=100.0 * bootstrap["upper"],
                wins=paired["treatment_only_correct_wins"],
                losses=paired["control_only_correct_losses"],
                ties=paired["ties"],
                p=paired["mcnemar"]["p_value"],
            )
        )
    lines.extend(
        [
            "",
            (
                "The shared baseline keeps its step-2000 prediction identity; "
                "all paired comparisons use exact `sample_id` alignment. Results "
                "retain the semantic overlay's diagnostic (not formal-pilot) scope."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path_value: str | Path, payload: bytes) -> Path:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for step in EXPECTED_STEPS:
        parser.add_argument(
            f"--step-{step}",
            required=True,
            type=Path,
            help=f"completed semantic-rescore directory for RP66 step {step}",
        )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = {step: getattr(args, f"step_{step}") for step in EXPECTED_STEPS}
    result = summarize_scaling(
        roots,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    markdown = render_markdown(result)
    json_path = _write_new(args.output_json, _canonical_json_line(result))
    markdown_path = None
    if args.output_markdown is not None:
        markdown_path = _write_new(args.output_markdown, markdown.encode("utf-8"))
    print(markdown, end="")
    print(f"JSON: {json_path}", file=sys.stderr)
    if markdown_path is not None:
        print(f"Markdown: {markdown_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
