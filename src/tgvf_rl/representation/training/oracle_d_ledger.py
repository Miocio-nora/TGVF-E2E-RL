"""Run identity, durable records, publication, and paired oracle-D summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterator

from tgvf_rl.public_api_compat import rebind_public_class, rebind_public_function

from .config import RepresentationTrainingConfig
from .oracle_d_schema import (
    ORACLE_D_UTILITY_SCHEMA_VERSION,
    ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION,
    OracleDUtilityArm,
    OracleDUtilityModelInput,
)


_PUBLIC_MODULE_PATH = Path(__file__).with_name("oracle_d_utility.py")


def _run_identity_payload(
    *,
    source_config: Any,
    training: RepresentationTrainingConfig,
    data_manifest_sha256: str,
    model_inputs: Sequence[OracleDUtilityModelInput],
    arms: tuple[OracleDUtilityArm, ...],
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    decode_mode: str,
    group_start: int,
    group_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    module_path = _PUBLIC_MODULE_PATH.resolve()
    root = module_path.parents[4]
    return {
        "schema_version": ORACLE_D_UTILITY_SCHEMA_VERSION,
        "claim_scope": "oracle_target_conditioned_D_utility_not_end_to_end_tool_selection",
        "source_config_path": str(source_config.source_path),
        "source_config_sha256": source_config.source_sha256,
        "training_config_sha256": source_config.training_config_sha256,
        "artifact_file_sha256": source_config.artifact_file_sha256,
        "artifact_manifest_sha256": source_config.artifact_manifest_sha256,
        "training_run_identity_sha256": source_config.expected_run_identity_sha256,
        "expected_global_step": source_config.expected_global_step,
        "model_name": training.model.model_name,
        "model_path": str(training.model.local_path),
        "data_manifest_sha256": data_manifest_sha256,
        "ordered_selected_samples": [
            {
                "sample_id": row.sample_id,
                "sample_content_sha256": row.sample_content_sha256,
                "image_group_key": row.image_group_key,
            }
            for row in model_inputs
        ],
        "arms": [arm.value for arm in arms],
        "arm_contracts": {arm.value: _arm_contract(arm) for arm in arms},
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": list(eos_token_ids),
        "legacy_source_config_eos_token_ids": list(
            source_config.evaluation.eos_token_ids
        ),
        "decode_mode": decode_mode,
        "greedy": True,
        "random_seed": source_config.evaluation.random_seed,
        "group_start": group_start,
        "group_limit": group_limit,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "image_only_tool_schema_exposed": False,
        "target_arm_tool_schema_exposed": True,
        "ground_truth_model_input": False,
        "post_focus_transcript_model_input": False,
        "scoring": "thinking_suffix_deterministic_mcq_exact_numeric_v1",
        "cached_decode_note": (
            "cached/no-cache token parity passed the bounded RP62 lane; strict full-logit "
            "parity was not established by RP61"
            if decode_mode == "cached"
            else "full-prefix no-cache oracle"
        ),
        "live_git_head": _git_head(root),
        "live_module_sha256": sha256(module_path.read_bytes()).hexdigest(),
    }


def _arm_contract(arm: OracleDUtilityArm) -> dict[str, Any]:
    return {
        OracleDUtilityArm.IMAGE_ONLY: {
            "prompt": "question_only_no_tool_schema",
            "source_image": True,
            "oracle_target_transcript": False,
            "d": "absent",
        },
        OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "all_zero_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "correct_target_stage1_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT: {
            "prompt": "image_question_no_tool_schema_original_image_slot",
            "source_image": False,
            "oracle_target_transcript": False,
            "d": "cyclic_next_target_same_image_stage1_main_and_all_deepstack_in_original_image_slot",
        },
        OracleDUtilityArm.TARGET_ZERO_D_ONLY: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "all_zero_main_and_all_deepstack",
        },
        OracleDUtilityArm.CORRECT_D_ONLY: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "correct_target_stage1_main_and_all_deepstack",
        },
        OracleDUtilityArm.IMAGE_TARGET_ZERO_D: {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "all_zero_main_and_all_deepstack",
        },
        OracleDUtilityArm.IMAGE_CORRECT_D: {
            "prompt": "image_question_plus_oracle_target_tool_transcript",
            "source_image": True,
            "oracle_target_transcript": True,
            "d": "correct_target_stage1_main_and_all_deepstack",
        },
        OracleDUtilityArm.MATCHED_WRONG_D: {
            "prompt": "question_plus_oracle_target_tool_transcript",
            "source_image": False,
            "oracle_target_transcript": True,
            "d": "cyclic_next_target_same_image_stage1_main_and_all_deepstack",
        },
    }[arm]


class _OracleRunLedger:
    """Atomic per-arm records with a reconstructable JSONL convenience view."""

    def __init__(
        self,
        root: Path,
        *,
        identity_payload: Mapping[str, Any],
        expected_keys: tuple[tuple[str, str], ...],
    ) -> None:
        self.root = root
        self.records_dir = root / "records"
        self.identity_path = root / "identity.json"
        self.jsonl_path = root / "records.jsonl"
        self.progress_path = root / "progress.json"
        self.summary_path = root / "summary.json"
        self.lock_path = root / "run.lock"
        self.identity_payload = dict(identity_payload)
        self.identity_sha256 = _canonical_sha256(self.identity_payload)
        self.expected_keys = expected_keys
        if len(set(expected_keys)) != len(expected_keys):
            raise ValueError("oracle ledger expected keys must be unique")
        self._completed: dict[tuple[str, str], dict[str, Any]] = {}

    @contextmanager
    def locked(self) -> Iterator[None]:
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"oracle output root is already active: {self.root}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def prepare(self) -> None:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        declared = {
            "schema_version": ORACLE_D_UTILITY_SCHEMA_VERSION,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity_payload,
        }
        if self.identity_path.exists():
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if observed != declared:
                raise ValueError(
                    "oracle output identity differs; choose another output root"
                )
        else:
            _atomic_write_json(self.identity_path, declared)
        expected = set(self.expected_keys)
        completed: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(self.records_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record.get("sample_id"), record.get("arm"))
            if record.get("run_identity_sha256") != self.identity_sha256:
                raise ValueError(f"oracle record has another run identity: {path}")
            if key not in expected or key in completed:
                raise ValueError(
                    f"oracle record key is unexpected or duplicate: {path}"
                )
            completed[key] = record
        self._completed = completed
        self._rebuild_jsonl()
        self._write_progress()

    @property
    def complete(self) -> bool:
        return len(self._completed) == len(self.expected_keys)

    def has(self, sample_id: str, arm: str) -> bool:
        return (sample_id, arm) in self._completed

    def commit(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        key = (payload.get("sample_id"), payload.get("arm"))
        if key not in set(self.expected_keys):
            raise ValueError("oracle record key was not declared")
        if key in self._completed:
            raise FileExistsError("oracle record was already committed")
        filename = sha256(f"{key[0]}\0{key[1]}".encode("utf-8")).hexdigest() + ".json"
        path = self.records_dir / filename
        _atomic_write_json(path, payload)
        self._completed[key] = payload
        with self.jsonl_path.open("ab") as handle:
            handle.write(_canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._write_progress()

    def summary(self) -> dict[str, object]:
        if not self.complete:
            raise RuntimeError("cannot summarize an incomplete oracle run")
        records = tuple(self._completed[key] for key in self.expected_keys)
        by_arm: dict[str, dict[str, object]] = {}
        for arm in self.identity_payload["arms"]:
            arm_records = tuple(record for record in records if record["arm"] == arm)
            correct = sum(record["score"]["correct"] is True for record in arm_records)
            incorrect = sum(
                record["score"]["correct"] is False for record in arm_records
            )
            unresolved = sum(
                record["score"]["correct"] is None for record in arm_records
            )
            routes: dict[str, int] = {}
            for record in arm_records:
                route = record["score"]["route"]
                routes[route] = routes.get(route, 0) + 1
            by_arm[arm] = {
                "correct": correct,
                "incorrect": incorrect,
                "unresolved": unresolved,
                "total": len(arm_records),
                "strict_lower_bound_accuracy": correct / len(arm_records),
                "formal_accuracy": (
                    correct / len(arm_records) if unresolved == 0 else None
                ),
                "score_routes": dict(sorted(routes.items())),
            }
        paired = {}
        for name, treatment, control in (
            (
                "direct_D_replacement_content_effect",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.DIRECT_ZERO_D_REPLACEMENT.value,
            ),
            (
                "direct_D_replacement_specificity",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.DIRECT_MATCHED_WRONG_D_REPLACEMENT.value,
            ),
            (
                "direct_D_replacement_vs_native_image",
                OracleDUtilityArm.DIRECT_CORRECT_D_REPLACEMENT.value,
                OracleDUtilityArm.IMAGE_ONLY.value,
            ),
            (
                "D_only_content_effect",
                OracleDUtilityArm.CORRECT_D_ONLY.value,
                OracleDUtilityArm.TARGET_ZERO_D_ONLY.value,
            ),
            (
                "image_plus_D_content_effect",
                OracleDUtilityArm.IMAGE_CORRECT_D.value,
                OracleDUtilityArm.IMAGE_TARGET_ZERO_D.value,
            ),
        ):
            if treatment in by_arm and control in by_arm:
                paired[name] = _paired_summary(records, treatment, control)
        payload: dict[str, object] = {
            "schema_version": ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "run_identity_sha256": self.identity_sha256,
            "sample_count": len({key[0] for key in self.expected_keys}),
            "record_count": len(records),
            "arms": by_arm,
            "paired_effects": paired,
            "records_jsonl": str(self.jsonl_path),
            "records_jsonl_sha256": sha256(self.jsonl_path.read_bytes()).hexdigest(),
            "interpretation": (
                "Correct-vs-zero paired effects isolate D content conditional on an "
                "oracle trajectory target. They do not measure autonomous target/tool selection."
            ),
        }
        parity_path = self.root / "image_only_native_parity.json"
        payload["image_only_native_parity"] = (
            json.loads(parity_path.read_text(encoding="utf-8"))
            if parity_path.exists()
            else None
        )
        _atomic_write_json(self.summary_path, payload)
        return payload

    def _rebuild_jsonl(self) -> None:
        payload = b"".join(
            _canonical_json_bytes(self._completed[key]) + b"\n"
            for key in self.expected_keys
            if key in self._completed
        )
        _atomic_write_bytes(self.jsonl_path, payload)

    def _write_progress(self) -> None:
        completed = len(self._completed)
        total = len(self.expected_keys)
        _atomic_write_json(
            self.progress_path,
            {
                "schema_version": "representation_oracle_d_utility_progress_v1",
                "run_identity_sha256": self.identity_sha256,
                "completed_records": completed,
                "total_records": total,
                "fraction": completed / total,
                "status": "complete" if completed == total else "running_or_resumable",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


def _paired_summary(
    records: Sequence[Mapping[str, Any]], treatment: str, control: str
) -> dict[str, object]:
    truth = {
        (record["sample_id"], record["arm"]): record["score"]["correct"]
        for record in records
    }
    sample_ids = sorted(
        sample_id
        for sample_id, arm in truth
        if arm == treatment and (sample_id, control) in truth
    )
    resolved = tuple(
        sample_id
        for sample_id in sample_ids
        if truth[(sample_id, treatment)] is not None
        and truth[(sample_id, control)] is not None
    )
    wins = sum(
        truth[(sample_id, treatment)] is True and truth[(sample_id, control)] is False
        for sample_id in resolved
    )
    losses = sum(
        truth[(sample_id, treatment)] is False and truth[(sample_id, control)] is True
        for sample_id in resolved
    )
    treatment_lower_bound = sum(
        truth[(sample_id, treatment)] is True for sample_id in sample_ids
    ) / len(sample_ids)
    control_lower_bound = sum(
        truth[(sample_id, control)] is True for sample_id in sample_ids
    ) / len(sample_ids)
    return {
        "treatment": treatment,
        "control": control,
        "paired_samples": len(sample_ids),
        "resolved_pairs": len(resolved),
        "unresolved_pairs": len(sample_ids) - len(resolved),
        "treatment_strict_lower_bound_accuracy": treatment_lower_bound,
        "control_strict_lower_bound_accuracy": control_lower_bound,
        "strict_lower_bound_accuracy_delta": (
            treatment_lower_bound - control_lower_bound
        ),
        "formal_accuracy_delta": (
            treatment_lower_bound - control_lower_bound
            if len(resolved) == len(sample_ids)
            else None
        ),
        "wins": wins,
        "losses": losses,
        "ties_among_resolved": len(resolved) - wins - losses,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


_IMPLEMENTATION_MODULE = __name__
_PUBLIC_MODULE = "tgvf_rl.representation.training.oracle_d_utility"
rebind_public_class(
    _OracleRunLedger,
    implementation_module=_IMPLEMENTATION_MODULE,
    public_module=_PUBLIC_MODULE,
)
for _facade_function in (
    _run_identity_payload,
    _arm_contract,
    _paired_summary,
    _canonical_json_bytes,
    _canonical_sha256,
    _atomic_write_json,
    _atomic_write_bytes,
    _git_head,
):
    rebind_public_function(
        _facade_function,
        implementation_module=_IMPLEMENTATION_MODULE,
        public_module=_PUBLIC_MODULE,
    )


__all__ = ["_OracleRunLedger", "_paired_summary", "_run_identity_payload"]
