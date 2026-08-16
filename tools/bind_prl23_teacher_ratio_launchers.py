#!/usr/bin/env python3
"""Bind PRL23 launcher templates to accepted ratio artifacts exactly once."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ZERO_COMMIT = "0" * 40
ZERO_SHA256 = "0" * 64
HEX_RE = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class Arm:
    name: str
    percentage: int
    config: Path
    plan: Path
    artifact_root: Path


ARMS = (
    Arm(
        name="teacher50",
        percentage=50,
        config=Path(
            "configs/policy/runs/"
            "prl_23_a_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
            "teacher50_8step_ws8.toml"
        ),
        plan=Path(
            "configs/evaluation/"
            "prl23_a_frozen_rp67_tfree_teacher50_step8_step16_paired_seed_"
            "coredev2511_plan.json"
        ),
        artifact_root=Path(
            "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/"
            "policy_rl/PRL23-TEACHER50-MIXED-SCHEDULE-v1"
        ),
    ),
    Arm(
        name="teacher100",
        percentage=100,
        config=Path(
            "configs/policy/runs/"
            "prl_23_b_qwen3_instruct_full_frozen_rp67_bs16_n16_tfree_"
            "teacher100_8step_ws8.toml"
        ),
        plan=Path(
            "configs/evaluation/"
            "prl23_b_frozen_rp67_tfree_teacher100_step8_step16_paired_seed_"
            "coredev2511_plan.json"
        ),
        artifact_root=Path(
            "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/"
            "policy_rl/PRL23-TEACHER100-MIXED-SCHEDULE-v1"
        ),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_hex(value: str, length: int, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or HEX_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return normalized


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{field} must be a non-empty regular non-symlink file: {path}")
    return path


def _load_artifact(root: Path, expected_percentage: int) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact root must be a regular non-symlink directory: {root}")
    manifest_path = _regular_file(root / "manifest.json", "manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("ratio manifest must be a JSON object")
    if manifest_bytes != _canonical_json(manifest) + b"\n":
        raise ValueError("ratio manifest is not canonical JSONL")
    if manifest.get("dataset_kind") != "policy_t1_teacher_ratio_mix":
        raise ValueError("ratio manifest dataset_kind differs")
    if manifest.get("sample_count") != 20_480:
        raise ValueError("ratio manifest sample_count differs")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("ratio manifest schedule is missing")
    observed_percentage = schedule.get(
        "teacher_percentage", manifest.get("teacher_percentage")
    )
    if observed_percentage != expected_percentage:
        raise ValueError(
            f"ratio manifest teacher_percentage differs: {observed_percentage!r}"
        )
    if schedule.get("seed") != 42 or schedule.get("micro_size") != 16:
        raise ValueError("ratio manifest seed or micro_size differs")
    if schedule.get("teacher_per_micro") != expected_percentage * 16 // 100:
        raise ValueError("ratio manifest teacher_per_micro differs")
    source_counts = manifest.get("source_counts")
    if not isinstance(source_counts, dict):
        raise ValueError("ratio manifest source_counts is missing")
    if source_counts.get("teacher") != 20_480 * expected_percentage // 100:
        raise ValueError("ratio manifest teacher row count differs")

    content = dict(manifest)
    content_sha256 = _require_hex(
        str(content.pop("content_sha256", "")), 64, "manifest content_sha256"
    )
    if _sha256(_canonical_json(content)) != content_sha256:
        raise ValueError("ratio manifest content SHA256 differs")
    samples = manifest.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("ratio manifest samples descriptor is missing")
    if samples.get("path") != "samples.jsonl" or samples.get("rows") != 20_480:
        raise ValueError("ratio manifest samples descriptor differs")
    samples_sha256 = _require_hex(
        str(samples.get("sha256", "")), 64, "samples SHA256"
    )
    if _file_sha256(_regular_file(root / "samples.jsonl", "samples")) != samples_sha256:
        raise ValueError("ratio samples SHA256 differs")
    return {
        "manifest_file_sha256": _sha256(manifest_bytes),
        "content_sha256": content_sha256,
        "samples_sha256": samples_sha256,
    }


def _replace_toml_key(text: str, section: str, key: str, value: object) -> str:
    lines = text.splitlines(keepends=True)
    current = ""
    matches: list[int] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
        elif current == section and re.match(rf"^{re.escape(key)}\s*=", stripped):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one [{section}].{key}, found {len(matches)}")
    rendered = json.dumps(value) if isinstance(value, str) else str(value)
    newline = "\n" if lines[matches[0]].endswith("\n") else ""
    lines[matches[0]] = f"{key} = {rendered}{newline}"
    return "".join(lines)


def _bound_or_placeholder(current: object, desired: str, field: str) -> None:
    if current not in {ZERO_COMMIT, ZERO_SHA256, desired}:
        raise ValueError(f"{field} is already bound to a different identity")


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bind_arm(
    arm: Arm,
    *,
    repository_root: Path,
    shared_commit: str,
    iteration_identity_sha256: str,
    check_only: bool,
) -> dict[str, object]:
    artifact = _load_artifact(arm.artifact_root, arm.percentage)
    artifact["iteration_identity_sha256"] = iteration_identity_sha256
    config_path = repository_root / arm.config
    plan_path = repository_root / arm.plan
    config_text = _regular_file(config_path, "run config").read_text(encoding="utf-8")

    import tomllib

    current_config = tomllib.loads(config_text)
    if current_config["dataset"].get("teacher_percentage") != arm.percentage:
        raise ValueError(f"{arm.name} config teacher_percentage differs")
    _bound_or_placeholder(current_config["code"].get("commit"), shared_commit, "code.commit")
    for key, desired in artifact.items():
        _bound_or_placeholder(current_config["dataset"].get(key), desired, f"dataset.{key}")
    config_text = _replace_toml_key(config_text, "code", "commit", shared_commit)
    for key, desired in artifact.items():
        config_text = _replace_toml_key(config_text, "dataset", key, desired)
    config_bytes = config_text.encode("utf-8")
    config_sha256 = _sha256(config_bytes)

    plan = json.loads(_regular_file(plan_path, "evaluation plan").read_text(encoding="utf-8"))
    if plan.get("policy_config") != arm.config.as_posix():
        raise ValueError(f"{arm.name} plan policy_config differs")
    _bound_or_placeholder(
        plan.get("policy_config_sha256"), config_sha256, "plan.policy_config_sha256"
    )
    plan["policy_config_sha256"] = config_sha256
    plan_bytes = json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"

    if not check_only:
        _atomic_write(config_path, config_bytes)
        _atomic_write(plan_path, plan_bytes)
    return {
        "arm": arm.name,
        "teacher_percentage": arm.percentage,
        "artifact_root": str(arm.artifact_root),
        **artifact,
        "policy_config_sha256": config_sha256,
        "mode": "check" if check_only else "write",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-commit", required=True)
    parser.add_argument("--teacher50-iteration-identity-sha256", required=True)
    parser.add_argument("--teacher100-iteration-identity-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository_root = args.repository_root.resolve()
    shared_commit = _require_hex(args.shared_commit, 40, "shared commit")
    subprocess.run(
        ["git", "cat-file", "-e", f"{shared_commit}^{{commit}}"],
        cwd=repository_root,
        check=True,
    )
    iterations = {
        "teacher50": _require_hex(
            args.teacher50_iteration_identity_sha256,
            64,
            "Teacher50 iteration identity",
        ),
        "teacher100": _require_hex(
            args.teacher100_iteration_identity_sha256,
            64,
            "Teacher100 iteration identity",
        ),
    }
    records = [
        _bind_arm(
            arm,
            repository_root=repository_root,
            shared_commit=shared_commit,
            iteration_identity_sha256=iterations[arm.name],
            check_only=args.check_only,
        )
        for arm in ARMS
    ]
    print(json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
