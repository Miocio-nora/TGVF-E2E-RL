"""Fail-closed static run contract for PRL13 native DeepEyes controls.

The checked-in TOMLs are intentionally non-launchable templates until the
native runtime implementation commit is merged.  ``assert_launchable`` also
requires a clean repository containing the bound implementation commit, so a
placeholder or an edited runtime can never start an expensive run.
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

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_BATCH_COUNTS,
    DEEPEYES_CANDIDATE_ROWS,
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_PROBE_NAME,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_PROMPTS_PER_STEP,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_TOTAL_STEPS,
)
from tgvf_rl.framework.verl.deepeyes_official_dataset import (
    DEEPEYES_OFFICIAL_DATASET_CLASS,
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    DEEPEYES_MAX_ACTIVE_PERCEPTION,
    DEEPEYES_THINKLITE_AGENT_NAME,
    DEEPEYES_TOOL_NAME,
    DEEPEYES_TOOL_PARSER,
    DEEPEYES_VISUAL_AGENT_NAME,
    THINKLITE_PROMPT_IDENTITY,
    VISUAL_PROMPT_IDENTITY,
)
from tgvf_rl.rewards.deepeyes_official import (
    DEEPEYES_BINARY_JUDGE_MAX_TOKENS,
    DEEPEYES_BINARY_JUDGE_MODEL,
    DEEPEYES_BINARY_JUDGE_TOP_P,
    DEEPEYES_THINKLITE_ANSWER_WEIGHT,
    DEEPEYES_THINKLITE_FORMAT_WEIGHT,
    DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256,
    DEEPEYES_THINKLITE_JUDGE_TEMPERATURE,
    DEEPEYES_VISUAL_ANSWER_LIMIT,
    DEEPEYES_VISUAL_ANSWER_WEIGHT,
    DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT,
    DEEPEYES_VISUAL_FORMAT_WEIGHT,
    DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256,
    DEEPEYES_VISUAL_JUDGE_TEMPERATURE,
)


DEEPEYES_NATIVE_RUN_SCHEMA = "policy-e2e-deepeyes-native-run-config-v1"
DEEPEYES_NATIVE_CODE_PLACEHOLDER = "CORE_COMMIT_REQUIRED"
DEEPEYES_NATIVE_EVALUATION_GATES = (0, 8, 20, 45, 80)
DEEPEYES_NATIVE_CHECKPOINT_GATES = (1, 8, 20, 45, 80)
DEEPEYES_NATIVE_GATES = DEEPEYES_NATIVE_EVALUATION_GATES
DEEPEYES_NATIVE_RUNTIME_ENTRYPOINT = "tgvf_rl.framework.verl.prl13_main"
DEEPEYES_NATIVE_AGENT_LOOP_TARGET = (
    "tgvf_rl.framework.verl.native_deepeyes_agent_loop.NativeDeepEyesAgentLoop"
)
DEEPEYES_JUDGE_TEMPLATE_FILE_SHA256 = (
    "bcd45d0e3ef996defd6c50fa227db341859acc86dac2f8d6dd36b8c425ba4a8c"
)
DEEPEYES_REWARD_MANAGER_MODULE_PATH = "pkg://tgvf_rl.rewards.deepeyes_verl_reward"
DEEPEYES_REWARD_MANAGER_CLASS_NAME = "DeepEyesOfficialRewardManager"
DEEPEYES_DATASET_MODULE_PATH = "pkg://tgvf_rl.framework.verl.deepeyes_official_dataset"

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOP_LEVEL = {
    "schema_version",
    "launch_enabled",
    "run_id",
    "arm",
    "code",
    "model",
    "dataset",
    "protocol",
    "reward",
    "algorithm",
    "optimization",
    "distributed",
    "rollout",
    "framework",
    "training",
    "evaluation",
    "audit",
    "metrics",
    "output",
}
_SECTION_FIELDS: Mapping[str, set[str]] = {
    "code": {"repository", "commit", "dirty"},
    "model": {
        "family",
        "name",
        "path",
        "training_mode",
        "lora_rank",
        "native_pixels",
        "vision_trainable",
        "projector_trainable",
        "language_trainable",
    },
    "dataset": {
        "root",
        "sample_count",
        "manifest_file_sha256",
        "content_sha256",
        "samples_sha256",
        "candidate_sidecar_path",
        "candidate_sidecar_rows",
        "candidate_sidecar_sha256",
        "selection_policy",
        "new_filtering",
        "dataloader_shuffle",
        "return_raw_chat",
        "filter_overlong_prompts",
        "length_filter_tool_schema",
        "schedule_mode",
        "schedule_seed",
        "batch_size",
        "vstar_per_batch",
        "arxivqa_per_batch",
        "thinklite_per_batch",
        "steps",
        "without_replacement",
        "probe_name",
        "probe_seed",
        "probe_size",
        "probe_excluded_from_training",
        "verl_dataset_module_path",
        "verl_dataset_class_name",
        "train_files",
        "probe_files",
    },
    "protocol": {
        "visual_prompt_bundle_sha256",
        "thinklite_prompt_bundle_sha256",
        "tool_name",
        "tool_parser",
        "multi_turn_format",
        "visual_agent_name",
        "thinklite_agent_name",
        "visual_sources",
        "thinklite_sources",
        "max_active_perception",
        "tool_observation_role",
        "tool_observation_policy_mask",
        "successful_crop_source",
        "coordinate_mapper",
        "unified_train_eval_coordinate_mapper",
    },
    "reward": {
        "visual_judge_model",
        "visual_judge_every_trajectory",
        "visual_judge_modality",
        "judge_service_config_path",
        "judge_service_config_sha256",
        "judge_prompt_sha256",
        "thinklite_judge_prompt_sha256",
        "judge_output",
        "judge_temperature",
        "thinklite_judge_temperature",
        "judge_top_p",
        "judge_max_tokens",
        "reward_manager",
        "reward_manager_source",
        "reward_manager_module_path",
        "reward_manager_class_name",
        "custom_reward_function",
        "judge_batch_max_concurrency",
        "judge_max_attempts",
        "judge_retry_backoff_seconds",
        "judge_retry_maximum_seconds",
        "judge_cache_max_entries",
        "judge_maximum_failure_fraction",
        "reward_num_workers",
        "visual_answer_weight",
        "visual_format_weight",
        "visual_conditional_tool_weight",
        "visual_answer_length_limit",
        "thinklite_verifier",
        "thinklite_fallback_model",
        "thinklite_answer_weight",
        "thinklite_format_weight",
        "format_valid_value",
        "format_invalid_value",
    },
    "algorithm": {
        "name",
        "advantage_estimator",
        "kl_coefficient",
        "actor_kl_loss",
        "norm_adv_by_std_in_grpo",
        "gamma",
        "lam",
        "entropy_coefficient",
    },
    "optimization": {
        "optimizer",
        "learning_rate",
        "actor_loss_reduction",
        "global_prompt_batch_size",
        "ppo_mini_batch_size",
        "actor_micro_batch_size_per_gpu",
        "gradient_clip_norm",
        "scheduler",
        "warmup_steps",
        "ppo_epochs",
        "dynamic_batch_size",
    },
    "distributed": {
        "world_size",
        "physical_gpu_ids",
        "fsdp_strategy",
        "parameter_offload",
        "optimizer_offload",
        "reference_parameter_offload",
        "complete_weight_sync",
    },
    "rollout": {
        "backend",
        "trajectories_per_prompt",
        "max_prompt_length",
        "max_response_length",
        "temperature",
        "top_p",
        "rollout_logprob_micro_batch_size_per_gpu",
        "reference_logprob_micro_batch_size_per_gpu",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "use_remove_padding",
        "max_user_turns",
        "max_assistant_turns",
        "max_parallel_calls",
        "tensor_parallel_size",
        "free_cache_engine",
        "single_response_max_tokens",
    },
    "framework": {
        "agent_loop_manager",
        "agent_loop_config_path",
        "tool_config_path",
        "native_visual_loop_target",
        "runtime_entrypoint",
        "standard_tool_agent_loop",
        "standard_multi_modal_inputs",
        "full_weight_sync",
        "custom_lossless_replay",
        "exact_replay",
    },
    "training": {
        "maximum_optimizer_steps",
        "first_update_gate_step",
        "require_vision_and_language_weight_change",
        "checkpoint_steps",
        "checkpoint_contents",
        "evaluation_steps",
        "checkpoint_before_evaluation",
        "validation_before_save",
        "resume_mode",
        "wandb_enabled",
        "logger",
        "project_name",
    },
    "evaluation": {
        "protocol",
        "datasets",
        "t1_probe_name",
        "t1_probe_source_metrics",
        "step0_required",
    },
    "audit": {
        "trajectory_retention",
        "trajectories_per_step",
        "retain_failed_and_zero_reward",
        "grounding_source",
    },
    "metrics": {
        "required",
        "vstar_grounding",
    },
    "output": {"root", "checkpoint_directory", "metrics_path"},
}

_EXACT_VALUES: Mapping[str, object] = {
    "code.repository": "Miocio-nora/TGVF-E2E-RL",
    "code.dirty": False,
    "model.family": "qwen3_vl",
    "model.name": "Qwen3-VL-8B-Instruct",
    "model.training_mode": "full",
    "model.lora_rank": 0,
    "model.native_pixels": True,
    "model.vision_trainable": True,
    "model.projector_trainable": True,
    "model.language_trainable": True,
    "dataset.sample_count": DEEPEYES_T1_SAMPLE_COUNT,
    "dataset.manifest_file_sha256": DEEPEYES_T1_MANIFEST_FILE_SHA256,
    "dataset.content_sha256": DEEPEYES_T1_CONTENT_SHA256,
    "dataset.samples_sha256": DEEPEYES_T1_SAMPLES_SHA256,
    "dataset.candidate_sidecar_rows": DEEPEYES_CANDIDATE_ROWS,
    "dataset.candidate_sidecar_sha256": DEEPEYES_CANDIDATE_SHA256,
    "dataset.selection_policy": "existing_final_t1_retain_t2_ignored",
    "dataset.new_filtering": False,
    "dataset.dataloader_shuffle": False,
    "dataset.return_raw_chat": True,
    "dataset.filter_overlong_prompts": True,
    "dataset.length_filter_tool_schema": "none",
    "dataset.schedule_seed": 42,
    "dataset.batch_size": DEEPEYES_PROMPTS_PER_STEP,
    "dataset.steps": DEEPEYES_TOTAL_STEPS,
    "dataset.without_replacement": True,
    "dataset.probe_name": DEEPEYES_PROBE_NAME,
    "dataset.probe_seed": DEEPEYES_PROBE_SEED,
    "dataset.probe_size": DEEPEYES_PROMPTS_PER_STEP,
    "dataset.probe_excluded_from_training": True,
    "dataset.verl_dataset_module_path": DEEPEYES_DATASET_MODULE_PATH,
    "dataset.verl_dataset_class_name": DEEPEYES_OFFICIAL_DATASET_CLASS.rsplit(".", 1)[
        -1
    ],
    "dataset.train_files": [str(DEEPEYES_TRAIN_SENTINEL)],
    "dataset.probe_files": [str(DEEPEYES_PROBE_SENTINEL)],
    "protocol.visual_prompt_bundle_sha256": VISUAL_PROMPT_IDENTITY.bundle_sha256,
    "protocol.thinklite_prompt_bundle_sha256": THINKLITE_PROMPT_IDENTITY.bundle_sha256,
    "protocol.tool_name": DEEPEYES_TOOL_NAME,
    "protocol.tool_parser": DEEPEYES_TOOL_PARSER,
    "protocol.multi_turn_format": DEEPEYES_TOOL_PARSER,
    "protocol.visual_agent_name": DEEPEYES_VISUAL_AGENT_NAME,
    "protocol.thinklite_agent_name": DEEPEYES_THINKLITE_AGENT_NAME,
    "protocol.visual_sources": ["vstar", "arxivqa"],
    "protocol.thinklite_sources": ["thinklite"],
    "protocol.max_active_perception": DEEPEYES_MAX_ACTIVE_PERCEPTION,
    "protocol.tool_observation_role": "user",
    "protocol.tool_observation_policy_mask": 0,
    "protocol.successful_crop_source": "runtime_execution_status",
    "protocol.coordinate_mapper": "qwen_0_1000_to_source_v1",
    "protocol.unified_train_eval_coordinate_mapper": True,
    "reward.visual_judge_model": DEEPEYES_BINARY_JUDGE_MODEL,
    "reward.visual_judge_every_trajectory": True,
    "reward.visual_judge_modality": "text_only",
    "reward.judge_prompt_sha256": DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256,
    "reward.thinklite_judge_prompt_sha256": (DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256),
    "reward.judge_output": "official_visual_judgement_and_math_true_false",
    "reward.judge_temperature": DEEPEYES_VISUAL_JUDGE_TEMPERATURE,
    "reward.thinklite_judge_temperature": DEEPEYES_THINKLITE_JUDGE_TEMPERATURE,
    "reward.judge_top_p": DEEPEYES_BINARY_JUDGE_TOP_P,
    "reward.judge_max_tokens": DEEPEYES_BINARY_JUDGE_MAX_TOKENS,
    "reward.reward_manager": "deepeyes_official_async",
    "reward.reward_manager_source": "importlib",
    "reward.reward_manager_module_path": DEEPEYES_REWARD_MANAGER_MODULE_PATH,
    "reward.reward_manager_class_name": DEEPEYES_REWARD_MANAGER_CLASS_NAME,
    "reward.custom_reward_function": "none",
    "reward.judge_batch_max_concurrency": 64,
    "reward.judge_max_attempts": 4,
    "reward.judge_retry_backoff_seconds": 0.25,
    "reward.judge_retry_maximum_seconds": 2.0,
    "reward.judge_cache_max_entries": 100000,
    "reward.judge_maximum_failure_fraction": 0.01,
    "reward.reward_num_workers": 1,
    "reward.visual_answer_weight": DEEPEYES_VISUAL_ANSWER_WEIGHT,
    "reward.visual_format_weight": DEEPEYES_VISUAL_FORMAT_WEIGHT,
    "reward.visual_conditional_tool_weight": DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT,
    "reward.visual_answer_length_limit": DEEPEYES_VISUAL_ANSWER_LIMIT,
    "reward.thinklite_verifier": "task_routed_math_verify_or_qwen25_72b",
    "reward.thinklite_fallback_model": DEEPEYES_BINARY_JUDGE_MODEL,
    "reward.thinklite_answer_weight": DEEPEYES_THINKLITE_ANSWER_WEIGHT,
    "reward.thinklite_format_weight": DEEPEYES_THINKLITE_FORMAT_WEIGHT,
    "reward.format_valid_value": 0,
    "reward.format_invalid_value": -1,
    "algorithm.name": "grpo",
    "algorithm.advantage_estimator": "grpo",
    "algorithm.kl_coefficient": 0.0,
    "algorithm.actor_kl_loss": False,
    "algorithm.norm_adv_by_std_in_grpo": True,
    "algorithm.gamma": 1.0,
    "algorithm.lam": 1.0,
    "algorithm.entropy_coefficient": 0.0,
    "optimization.optimizer": "adamw",
    "optimization.learning_rate": 0.000001,
    "optimization.actor_loss_reduction": ("deepeyes_official_micro_token_mean"),
    "optimization.global_prompt_batch_size": 256,
    "optimization.ppo_mini_batch_size": 256,
    "optimization.actor_micro_batch_size_per_gpu": 4,
    "optimization.gradient_clip_norm": 1.0,
    "optimization.scheduler": "constant",
    "optimization.warmup_steps": 0,
    "optimization.ppo_epochs": 1,
    "optimization.dynamic_batch_size": False,
    "distributed.world_size": 4,
    "distributed.physical_gpu_ids": [0, 1, 2, 3],
    "distributed.fsdp_strategy": "fsdp2",
    "distributed.parameter_offload": True,
    "distributed.optimizer_offload": True,
    "distributed.reference_parameter_offload": True,
    "distributed.complete_weight_sync": True,
    "rollout.backend": "vllm",
    "rollout.trajectories_per_prompt": 16,
    "rollout.max_prompt_length": 8192,
    "rollout.max_response_length": 20480,
    "rollout.temperature": 1.0,
    "rollout.top_p": 1.0,
    "rollout.rollout_logprob_micro_batch_size_per_gpu": 8,
    "rollout.reference_logprob_micro_batch_size_per_gpu": 8,
    "rollout.gpu_memory_utilization": 0.8,
    "rollout.max_num_batched_tokens": 32768,
    "rollout.enable_chunked_prefill": False,
    "rollout.use_remove_padding": True,
    "rollout.max_user_turns": 6,
    "rollout.max_assistant_turns": 7,
    "rollout.max_parallel_calls": 1,
    "rollout.tensor_parallel_size": 1,
    "rollout.free_cache_engine": True,
    "rollout.single_response_max_tokens": 10240,
    "framework.agent_loop_manager": (
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager"
    ),
    "framework.native_visual_loop_target": DEEPEYES_NATIVE_AGENT_LOOP_TARGET,
    "framework.runtime_entrypoint": DEEPEYES_NATIVE_RUNTIME_ENTRYPOINT,
    "framework.standard_tool_agent_loop": True,
    "framework.standard_multi_modal_inputs": True,
    "framework.full_weight_sync": True,
    "framework.custom_lossless_replay": False,
    "framework.exact_replay": False,
    "training.maximum_optimizer_steps": 80,
    "training.first_update_gate_step": 1,
    "training.require_vision_and_language_weight_change": True,
    "training.checkpoint_steps": list(DEEPEYES_NATIVE_CHECKPOINT_GATES),
    "training.checkpoint_contents": ["model", "hf_model", "optimizer", "extra"],
    "training.evaluation_steps": list(DEEPEYES_NATIVE_EVALUATION_GATES),
    "training.checkpoint_before_evaluation": True,
    "training.validation_before_save": False,
    "training.resume_mode": "auto_latest_saved_checkpoint",
    "training.wandb_enabled": True,
    "training.logger": ["console", "wandb"],
    "training.project_name": "tgvf-policy-rl",
    "evaluation.protocol": "official_visible_free_crop",
    "evaluation.datasets": [
        "T1-PROBE256",
        "DeepEyesDev591",
        "Grounding-200",
        "CoreDev2511",
    ],
    "evaluation.t1_probe_name": DEEPEYES_PROBE_NAME,
    "evaluation.t1_probe_source_metrics": ["accuracy", "tool_rate", "crop_rate"],
    "evaluation.step0_required": True,
    "audit.trajectory_retention": "all",
    "audit.trajectories_per_step": 4096,
    "audit.retain_failed_and_zero_reward": True,
    "audit.grounding_source": "candidate_sidecar_gt_regions",
    "metrics.required": [
        "source",
        "reward",
        "nonzero_grpo_group",
        "answer_length",
        "action_length",
        "response_length",
        "judge_requested",
        "visual_judge_requested",
        "thinklite_fallback_judge_requested",
        "judge_calls",
        "judge_cache_hits",
        "judge_retries",
        "judge_latency_seconds",
        "judge_failures",
    ],
    "metrics.vstar_grounding": [
        "first_iou",
        "best_iou",
        "gt_coverage",
        "crop_area",
        "action_count",
    ],
}

# Historical step-0/8 snapshots were trained under the wrapper-bearing prompt
# identities below.  Loading those immutable run contracts must remain
# possible for paired evaluation, while all newly rendered prompts use the
# clean identities in ``_EXACT_VALUES``.
_HISTORICAL_PROMPT_BUNDLE_SHA256S: Mapping[str, frozenset[str]] = {
    "protocol.visual_prompt_bundle_sha256": frozenset(
        {"b91cfa2e228f496f745a6e0b368cff836e6786ad104e312cd776a12e0784b2ef"}
    ),
    "protocol.thinklite_prompt_bundle_sha256": frozenset(
        {"bd3226c36dea66cc57a9b30fad5c846ce3d3a23a008781425c70214580aba545"}
    ),
}

_FORBIDDEN_KEY_FRAGMENTS = (
    "representation",
    "adapter",
    "image_embeds",
    "exact_replay_payload",
    "tgvf",
    "rp66",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"PRL13 config is missing {path}")
        value = value[part]
    return value


def _walk(value: object, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.append((path, nested))
            rows.extend(_walk(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            rows.extend(_walk(nested, f"{prefix}[{index}]"))
    return rows


@dataclass(frozen=True, slots=True)
class DeepEyesNativeRunContract:
    source_path: Path
    payload: Mapping[str, Any]
    source_sha256: str

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def arm(self) -> str:
        return str(self.payload["arm"])

    @property
    def code_commit(self) -> str:
        return str(_nested(self.payload, "code.commit"))

    @property
    def launch_enabled(self) -> bool:
        return bool(self.payload["launch_enabled"])

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload).encode("utf-8")).hexdigest()

    def assert_launchable(self, repository_root: Path) -> None:
        if not self.launch_enabled:
            raise RuntimeError("PRL13 template has launch_enabled=false")
        if _GIT_COMMIT.fullmatch(self.code_commit) is None:
            raise RuntimeError("PRL13 code.commit is still an unbound placeholder")
        root = repository_root.resolve(strict=True)
        subprocess.run(
            ["git", "cat-file", "-e", self.code_commit + "^{commit}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", self.code_commit, "HEAD"],
            cwd=root,
            check=False,
        )
        if ancestor.returncode != 0:
            raise RuntimeError("bound PRL13 core commit is not an ancestor of HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise RuntimeError("PRL13 launch requires a clean worktree")


def load_deepeyes_native_run_contract(
    path: str | Path,
    *,
    allow_template: bool = True,
    allow_historical_prompt_contract: bool = False,
) -> DeepEyesNativeRunContract:
    if type(allow_historical_prompt_contract) is not bool:
        raise TypeError("allow_historical_prompt_contract must be bool")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("PRL13 config must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("PRL13 config must be strict UTF-8 TOML") from error
    if set(payload) != _TOP_LEVEL:
        raise ValueError("PRL13 config top-level fields differ")
    if payload.get("schema_version") != DEEPEYES_NATIVE_RUN_SCHEMA:
        raise ValueError("PRL13 config schema differs")
    if type(payload.get("launch_enabled")) is not bool:
        raise TypeError("launch_enabled must be bool")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("PRL13 run_id is unsafe")
    arm = payload.get("arm")
    if arm not in {"stratified", "natural"}:
        raise ValueError("PRL13 arm must be stratified or natural")
    for section, fields in _SECTION_FIELDS.items():
        table = payload.get(section)
        if not isinstance(table, Mapping) or set(table) != fields:
            raise ValueError(f"PRL13 [{section}] fields differ")
    if _nested(payload, "dataset.schedule_mode") != arm:
        raise ValueError("arm and dataset.schedule_mode differ")
    observed_mix = {
        "vstar": _nested(payload, "dataset.vstar_per_batch"),
        "arxivqa": _nested(payload, "dataset.arxivqa_per_batch"),
        "thinklite": _nested(payload, "dataset.thinklite_per_batch"),
    }
    expected_mix = (
        dict(DEEPEYES_BATCH_COUNTS)
        if arm == "stratified"
        else {
            "vstar": 0,
            "arxivqa": 0,
            "thinklite": 0,
        }
    )
    if observed_mix != expected_mix:
        raise ValueError("PRL13 schedule arm source-count contract differs")
    for path_name, expected in _EXACT_VALUES.items():
        observed = _nested(payload, path_name)
        accepted_historical = (
            _HISTORICAL_PROMPT_BUNDLE_SHA256S.get(path_name, ())
            if allow_historical_prompt_contract and payload["launch_enabled"]
            else ()
        )
        if observed != expected and observed not in accepted_historical:
            raise ValueError(f"PRL13 fixed field differs: {path_name}")
    judge_service_sha256 = _nested(payload, "reward.judge_service_config_sha256")
    if (
        not isinstance(judge_service_sha256, str)
        or _SHA256.fullmatch(judge_service_sha256) is None
    ):
        raise ValueError("judge service config SHA-256 differs")
    if (
        not payload["launch_enabled"]
        and judge_service_sha256 != DEEPEYES_JUDGE_TEMPLATE_FILE_SHA256
    ):
        raise ValueError("template judge service config SHA-256 differs")
    code_commit = _nested(payload, "code.commit")
    if allow_template:
        if code_commit not in {DEEPEYES_NATIVE_CODE_PLACEHOLDER} and (
            not isinstance(code_commit, str)
            or _GIT_COMMIT.fullmatch(code_commit) is None
        ):
            raise ValueError("code.commit must be placeholder or full Git commit")
    elif not isinstance(code_commit, str) or _GIT_COMMIT.fullmatch(code_commit) is None:
        raise ValueError("launchable PRL13 config requires a full Git commit")
    if code_commit == DEEPEYES_NATIVE_CODE_PLACEHOLDER and payload["launch_enabled"]:
        raise ValueError("placeholder PRL13 config cannot enable launch")
    if not allow_template and payload["launch_enabled"] is not True:
        raise ValueError("launchable PRL13 config requires launch_enabled=true")
    for field_path, value in _walk(payload):
        lowered_path = field_path.casefold()
        if any(fragment in lowered_path for fragment in _FORBIDDEN_KEY_FRAGMENTS):
            raise ValueError(f"forbidden legacy PRL13 field: {field_path}")
        if isinstance(value, str) and any(
            fragment in value.casefold()
            for fragment in ("rp66", "image_embeds", "losslessreplay")
        ):
            raise ValueError(f"forbidden legacy PRL13 value: {field_path}")
    for path_name in (
        "model.path",
        "dataset.root",
        "dataset.candidate_sidecar_path",
        "reward.judge_service_config_path",
        "framework.agent_loop_config_path",
        "framework.tool_config_path",
        "output.root",
        "output.checkpoint_directory",
        "output.metrics_path",
    ):
        value = _nested(payload, path_name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"PRL13 path must be absolute: {path_name}")
    for path_name in ("dataset.train_files", "dataset.probe_files"):
        values = _nested(payload, path_name)
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], str)
            or not Path(values[0]).is_absolute()
        ):
            raise ValueError(
                f"PRL13 file list must contain one absolute path: {path_name}"
            )
    return DeepEyesNativeRunContract(
        source_path=source.resolve(),
        payload=payload,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "DEEPEYES_NATIVE_AGENT_LOOP_TARGET",
    "DEEPEYES_NATIVE_CODE_PLACEHOLDER",
    "DEEPEYES_NATIVE_GATES",
    "DEEPEYES_NATIVE_RUN_SCHEMA",
    "DEEPEYES_NATIVE_RUNTIME_ENTRYPOINT",
    "DeepEyesNativeRunContract",
    "load_deepeyes_native_run_contract",
]
