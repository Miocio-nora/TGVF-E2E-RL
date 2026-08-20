"""Static experiment overlay for the native-Crop T-free control.

The PRL13 contract remains the source of truth for model, data, prompt, Crop
tool, optimiser, rollout and policy-loss semantics.  This overlay records the
one intentional scientific change (the T-free Stage3 reward) together with
the matched 16-step/world-8 execution shape and output identity.  Keeping the
overlay separate prevents the historical DeepEyes reward contract from being
silently reinterpreted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

from tgvf_rl.policy.deepeyes_native_contract import (
    DeepEyesNativeRunContract,
    load_deepeyes_native_run_contract,
)


CROP_TFREE_RUN_SCHEMA = "policy-e2e-native-crop-tfree-run-config-v1"
CROP_TFREE_BS64_FMT2_RUN_SCHEMA = (
    "policy-e2e-native-crop-tfree-bs64-fmt2-run-config-v1"
)
CROP_TFREE_BS64_FMT2_IMAGE_MAX_PIXELS = 1_003_520
CROP_TFREE_CODE_PLACEHOLDER = "CORE_COMMIT_REQUIRED"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "code",
    "base",
    "reward",
    "matched_training",
    "retention",
    "evaluation",
    "output",
}
_FIELDS: Mapping[str, set[str]] = {
    "code": {"repository", "commit", "dirty"},
    "base": {"contract_path", "contract_sha256"},
    "reward": {
        "profile",
        "manager_class",
        "answer_weight",
        "repeated_call_penalty",
        "protocol_error_penalty",
        "tool_utility_reward_enabled",
        "focus_reward_enabled",
        "grounding_reward_enabled",
        "positive_crop_bonus_enabled",
        "answer_verifier",
    },
    "matched_training": {
        "model_name",
        "training_mode",
        "vision_trainable",
        "projector_trainable",
        "language_trainable",
        "global_prompt_batch_size",
        "trajectories_per_prompt",
        "world_size",
        "actor_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "learning_rate",
        "temperature",
        "maximum_response_length",
        "maximum_optimizer_steps",
        "kl_coefficient",
        "shuffle_seed",
    },
    "retention": {
        "save_every_step",
        "permanent_checkpoint_steps",
        "maximum_rolling_checkpoints",
    },
    "evaluation": {
        "benchmark",
        "temperature",
        "seed",
        "checkpoint_steps",
        "reuse_step0",
    },
    "output": {"root"},
}

_COMMON_EXACT: Mapping[str, object] = {
    "code.repository": "Miocio-nora/TGVF-E2E-RL",
    "code.dirty": False,
    "reward.answer_weight": 2.0,
    "reward.repeated_call_penalty": 0.05,
    "reward.tool_utility_reward_enabled": False,
    "reward.focus_reward_enabled": False,
    "reward.grounding_reward_enabled": False,
    "reward.positive_crop_bonus_enabled": False,
    "reward.answer_verifier": "deepeyes_extraction_qwen25_72b_text",
    "matched_training.model_name": "Qwen3-VL-8B-Instruct",
    "matched_training.training_mode": "full",
    "matched_training.vision_trainable": True,
    "matched_training.projector_trainable": True,
    "matched_training.language_trainable": True,
    "matched_training.trajectories_per_prompt": 16,
    "matched_training.world_size": 8,
    "matched_training.actor_micro_batch_size_per_gpu": 32,
    "matched_training.learning_rate": 0.000001,
    "matched_training.temperature": 1.0,
    "matched_training.maximum_response_length": 20480,
    "matched_training.maximum_optimizer_steps": 16,
    "matched_training.kl_coefficient": 0.0,
    "matched_training.shuffle_seed": 42,
    "retention.save_every_step": True,
    "retention.maximum_rolling_checkpoints": 2,
    "evaluation.benchmark": "CoreDev-2511-unified-temp1-seed42",
    "evaluation.temperature": 1.0,
    "evaluation.seed": 42,
    "evaluation.reuse_step0": True,
}

_EXACT_BY_SCHEMA: Mapping[str, Mapping[str, object]] = {
    CROP_TFREE_RUN_SCHEMA: {
        **_COMMON_EXACT,
        "reward.profile": "stage3-shaped-v1-tfree",
        "reward.manager_class": (
            "tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward."
            "DeepEyesCropTFreeRewardManager"
        ),
        "reward.protocol_error_penalty": 1.0,
        "matched_training.global_prompt_batch_size": 16,
        "matched_training.gradient_accumulation_steps": 1,
        "retention.permanent_checkpoint_steps": [8, 16],
        "evaluation.checkpoint_steps": [8, 16],
    },
    CROP_TFREE_BS64_FMT2_RUN_SCHEMA: {
        **_COMMON_EXACT,
        "reward.profile": "stage3-shaped-v1-tfree-fmt2",
        "reward.manager_class": (
            "tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward."
            "DeepEyesCropTFreeFMT2RewardManager"
        ),
        "reward.protocol_error_penalty": 2.0,
        "matched_training.global_prompt_batch_size": 64,
        "matched_training.gradient_accumulation_steps": 4,
        "retention.permanent_checkpoint_steps": [2, 4, 8, 12, 16],
        "evaluation.checkpoint_steps": [2, 4, 8, 12, 16],
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Crop T-free config is missing {path}")
        value = value[part]
    return value


@dataclass(frozen=True, slots=True)
class CropTFreeRunContract:
    source_path: Path
    payload: Mapping[str, Any]
    source_sha256: str
    base_contract: DeepEyesNativeRunContract

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload).encode("ascii")).hexdigest()

    @property
    def output_root(self) -> Path:
        return Path(str(_nested(self.payload, "output.root")))

    @property
    def code_commit(self) -> str:
        return str(_nested(self.payload, "code.commit"))

    @property
    def reward_manager_class(self) -> str:
        return str(_nested(self.payload, "reward.manager_class"))

    @property
    def reward_manager_module_path(self) -> str:
        return "pkg://" + self.reward_manager_class.rsplit(".", 1)[0]

    @property
    def reward_profile(self) -> str:
        return str(_nested(self.payload, "reward.profile"))

    @property
    def protocol_error_penalty(self) -> float:
        return float(_nested(self.payload, "reward.protocol_error_penalty"))

    @property
    def global_prompt_batch_size(self) -> int:
        return int(_nested(self.payload, "matched_training.global_prompt_batch_size"))

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(_nested(self.payload, "matched_training.gradient_accumulation_steps"))

    @property
    def maximum_optimizer_steps(self) -> int:
        return int(_nested(self.payload, "matched_training.maximum_optimizer_steps"))

    @property
    def permanent_checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in _nested(self.payload, "retention.permanent_checkpoint_steps"))

    @property
    def maximum_rolling_checkpoints(self) -> int:
        return int(_nested(self.payload, "retention.maximum_rolling_checkpoints"))

    @property
    def image_max_pixels(self) -> int | None:
        """Return the current Teacher25 pixel cap without changing legacy runs."""

        if self.payload["schema_version"] == CROP_TFREE_BS64_FMT2_RUN_SCHEMA:
            return CROP_TFREE_BS64_FMT2_IMAGE_MAX_PIXELS
        return None

    def assert_launchable(self, repository_root: Path) -> None:
        if _COMMIT.fullmatch(self.code_commit) is None:
            raise RuntimeError("Crop T-free code.commit is not bound")
        root = repository_root.resolve(strict=True)
        subprocess.run(
            ["git", "cat-file", "-e", self.code_commit + "^{commit}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", self.code_commit, "HEAD"],
            cwd=root,
            check=False,
        ).returncode:
            raise RuntimeError("bound Crop T-free implementation is not in HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise RuntimeError("Crop T-free launch requires a clean worktree")


def load_crop_tfree_run_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    allow_placeholder: bool = True,
) -> CropTFreeRunContract:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Crop T-free config must be a regular file")
    raw = source.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Crop T-free config must be strict UTF-8 TOML") from error
    if set(payload) != _TOP_LEVEL:
        raise ValueError("Crop T-free top-level fields differ")
    schema = payload.get("schema_version")
    if schema not in _EXACT_BY_SCHEMA:
        raise ValueError("Crop T-free schema differs")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Crop T-free run_id is unsafe")
    for section, fields in _FIELDS.items():
        table = payload.get(section)
        if not isinstance(table, Mapping) or set(table) != fields:
            raise ValueError(f"Crop T-free [{section}] fields differ")
    for field, expected in _EXACT_BY_SCHEMA[str(schema)].items():
        if _nested(payload, field) != expected:
            raise ValueError(f"Crop T-free fixed field differs: {field}")

    commit = _nested(payload, "code.commit")
    if commit != CROP_TFREE_CODE_PLACEHOLDER and (
        not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None
    ):
        raise ValueError("Crop T-free code.commit differs")
    if not allow_placeholder and commit == CROP_TFREE_CODE_PLACEHOLDER:
        raise ValueError("launchable Crop T-free config requires bound code.commit")

    root = Path(repository_root).resolve(strict=True)
    base_relative = _nested(payload, "base.contract_path")
    if not isinstance(base_relative, str) or Path(base_relative).is_absolute():
        raise ValueError("base.contract_path must be repository-relative")
    base_path = (root / base_relative).resolve(strict=True)
    if not base_path.is_relative_to(root):
        raise ValueError("base.contract_path escapes the repository")
    base_sha = _nested(payload, "base.contract_sha256")
    if not isinstance(base_sha, str) or _SHA256.fullmatch(base_sha) is None:
        raise ValueError("base.contract_sha256 differs")
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_sha:
        raise ValueError("native Crop base contract file identity differs")
    base_contract = load_deepeyes_native_run_contract(base_path)

    output = _nested(payload, "output.root")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ValueError("Crop T-free output.root must be absolute")
    return CropTFreeRunContract(
        source_path=source.resolve(),
        payload=payload,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        base_contract=base_contract,
    )


__all__ = [
    "CROP_TFREE_BS64_FMT2_RUN_SCHEMA",
    "CROP_TFREE_BS64_FMT2_IMAGE_MAX_PIXELS",
    "CROP_TFREE_CODE_PLACEHOLDER",
    "CROP_TFREE_RUN_SCHEMA",
    "CropTFreeRunContract",
    "load_crop_tfree_run_contract",
]
