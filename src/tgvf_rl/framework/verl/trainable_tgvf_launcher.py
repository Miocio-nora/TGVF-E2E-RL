"""Pinned-veRL launcher for the matched full-Qwen plus trainable-RP66 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Literal
import warnings

from tgvf_rl.data.deepeyes_official_schedule import (
    DEEPEYES_CANDIDATE_SHA256,
    DEEPEYES_CANDIDATE_SIDECAR,
    DEEPEYES_PROBE_SEED,
    DEEPEYES_T1_CONTENT_SHA256,
    DEEPEYES_T1_MANIFEST_FILE_SHA256,
    DEEPEYES_T1_ROOT,
    DEEPEYES_T1_SAMPLE_COUNT,
    DEEPEYES_T1_SAMPLES_SHA256,
    DEEPEYES_TRAIN_SEED,
)
from tgvf_rl.data.policy_teacher_quarter_mix import (
    PolicyTeacherQuarterMixRuntimeBinding,
)
from tgvf_rl.data.policy_teacher_ratio_mix import (
    PolicyTeacherRatioMixRuntimeBinding,
)
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA,
    POLICY_E2E_TGVF_BACKED_MATCHED_RUN_CONFIG_SCHEMAS,
    POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.horizon_extension import (
    PolicyHorizonExtension,
    policy_horizon_extension_from_environment,
)
from tgvf_rl.policy.launch import assert_policy_execution_identity
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import visual_tool_prompt_identity
from tgvf_rl.protocol.native import native_assistant_dialect_for_model

from . import launcher as legacy_launcher
from .compatibility import (
    SPIKE_CANDIDATE_VERL_COMMIT,
    VerlDistributionIdentity,
    _local_git_state,
    installed_verl_distribution_identity,
    load_verl_public_api,
)
from .deepeyes_official_dataset import _verified_schedule_index
from .launcher import (
    UPSTREAM_VERL_CONFIG_NAME,
    UPSTREAM_VERL_MAIN_MODULE,
    UPSTREAM_VERL_V0_RUNNER_FQN,
)
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from .policy_main import compose_pinned_verl_config, run_pinned_verl_config
from .policy_checkpoint_lifecycle import POLICY_CHECKPOINT_LIFECYCLE_SCHEMA
from .policy_task_runner import (
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
    POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV,
)
from .prl14_crop16_reference import (
    PRL14_CROP16_COMMON_OVERRIDES,
    PRL14_CROP16_COMPLETION_SHA256,
    PRL14_CROP16_REMOVED_OVERRIDES,
    PRL14_CROP16_RUN_NAME,
    apply_prl14_crop16_common_controls,
)
from .tgvf_deepeyes_matched_dataset import (
    CROP_TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    DeepEyesCropTGVFMatchedDatasetBinding,
    DeepEyesTGVFMatchedDatasetBinding,
    TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from .policy_teacher_quarter_mix_dataset import (
    POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME,
    POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS,
    POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH,
    PolicyTeacherQuarterMixDatasetBinding,
)
from .policy_teacher_ratio_mix_dataset import (
    POLICY_TEACHER_RATIO_MIX_CONFIG_NAME,
    POLICY_TEACHER_RATIO_MIX_DATASET_CLASS,
    POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH,
    PolicyTeacherRatioMixDatasetBinding,
)
from .trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
    TGVF_CHECKPOINT_ENGINE_CONTROL_KEY,
)
from .trainable_tgvf_engine import (
    TRAINABLE_TGVF_MODEL_TYPE,
    preflight_trainable_rp66_artifact,
)


TRAINABLE_TGVF_LAUNCH_SCHEMA = "tgvf.trainable-rp66-verl-launch.v1"
TRAINABLE_TGVF_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_tgvf_external"
TRAINABLE_TGVF_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset"
)
TRAINABLE_TGVF_SMOKE_TARGET = 1
TRAINABLE_TGVF_CANARY_TARGET = 1
TRAINABLE_TGVF_CANARY_PROMPTS = 4
TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT = 2
TRAINABLE_TGVF_CANARY_POLICY_TOKEN_BUDGET = 512
# One focused observation contains at most 980 merged image tokens under the
# pinned 1,003,520-pixel Qwen3 geometry plus at most 61 wrapper/text tokens.
# Six admitted calls and one conservative cap-error slot therefore require no
# more than 512 + 7 * 1,041 = 7,799 response-side tokens.  Use the next binary
# capacity boundary without paying Crop-16's full 20,480-token transport cost.
TRAINABLE_TGVF_CANARY_MAX_ENVIRONMENT_TOKENS_PER_TURN = 1041
TRAINABLE_TGVF_CANARY_MAX_ENVIRONMENT_TURNS = 7
TRAINABLE_TGVF_CANARY_MIN_RESPONSE_TRANSPORT_LENGTH = (
    TRAINABLE_TGVF_CANARY_POLICY_TOKEN_BUDGET
    + TRAINABLE_TGVF_CANARY_MAX_ENVIRONMENT_TURNS
    * TRAINABLE_TGVF_CANARY_MAX_ENVIRONMENT_TOKENS_PER_TURN
)
TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH = 8192
TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES = frozenset({4, 8})
# Crop-16's matched schedule presents at most 2 prompt micros * n16 = 32
# trajectories to each TP1 rollout engine. vLLM otherwise derives capture
# sizes up to 512 from max_num_seqs=1024. Those unused CUDA-graph pools remain
# resident while the actor trains and can collide with the following rollout
# weight remap. Bound graph residency to the real per-engine concurrency; this
# changes execution scheduling only, never samples, losses, or gradients.
TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32)
# Legacy v1/v2 records predate an identity-bound optimizer residency control and
# therefore retain their effective True setting. New exact-control records bind
# residency explicitly in the run config; this constant is compatibility data,
# not a launcher-wide policy.
TRAINABLE_TGVF_ACTOR_OPTIMIZER_OFFLOAD = True
TRAINABLE_TGVF_SUPPORTED_RUN_CONFIG_SCHEMAS = (
    POLICY_E2E_TGVF_BACKED_MATCHED_RUN_CONFIG_SCHEMAS
)
TrainableTGVFLaunchMode = Literal["formal", "smoke", "canary"]

_OPTIONAL_PARENT_LAUNCH_ENV = frozenset(
    {
        "TGVF_POLICY_AGENT_LOOP_WORKER_INDEX",
        "TGVF_POLICY_HORIZON_EXTENSION_PATH",
        "TGVF_POLICY_HORIZON_EXTENSION_SHA256",
        POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV,
    }
)


# World4 keeps Crop-16's global BS16, n16 and fixed actor trajectory micro32.
# Each rank therefore accumulates two complete micros before the one optimizer
# update instead of world8's one micro per rank.  Only physical execution
# topology changes; the equal-micro DeepEyes loss still averages eight micros.
_WORLD4_TOPOLOGY_OVERRIDES = MappingProxyType(
    {
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 4,
        "actor_rollout_ref.rollout.agent.num_workers": 4,
        "trainer.n_gpus_per_node": 4,
    }
)


def _matched_batch_overrides(
    config: PolicyE2ESmokeRunConfig,
) -> dict[str, int]:
    """Map serialized prompt geometry to pinned-veRL batch fields.

    Scaling independent prompts changes the global mini-batch and local
    accumulation count, but does not silently enlarge the proven per-rank
    trajectory micro-batch.  That micro-batch remains ``prompt_micro * n``.
    """

    prompts = config.accumulation.global_prompt_batch_size
    prompt_micro = config.accumulation.prompt_micro_batch_size_per_rank
    rollout_prompt_micro = (
        config.accumulation.rollout_prompt_micro_batch_size_per_engine
    )
    rollouts = config.policy.sampling.trajectories_per_prompt
    return {
        "data.train_batch_size": prompts,
        "data.gen_batch_size": prompts,
        "actor_rollout_ref.actor.ppo_mini_batch_size": prompts,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": (
            prompt_micro * rollouts
        ),
        "actor_rollout_ref.rollout.n": rollouts,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": (
            rollout_prompt_micro * rollouts
        ),
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (
            prompt_micro * rollouts
        ),
    }


def _apply_matched_mathematical_controls(
    values: dict[str, object],
    *,
    config: PolicyE2ESmokeRunConfig,
    optimizer_horizon: int,
) -> None:
    world_size = config.distributed.world_size
    if world_size not in TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES:
        raise ValueError("trainable TGVF topology must use world4 or world8")
    if type(optimizer_horizon) is not int or optimizer_horizon <= 0:
        raise ValueError("trainable TGVF optimizer horizon must be positive")
    apply_prl14_crop16_common_controls(values)
    # Preserve every accepted Crop-16 control except the batch geometry that
    # is an explicit run-config variable in this scaling series.  For legacy
    # BS16 configs these values are identical to the PRL14 constants.
    values.update(_matched_batch_overrides(config))
    # PRL14's serialized completion contains -1 because its upstream trainer
    # filled the scheduler horizon later.  The project TaskRunner deliberately
    # preserves an explicit run-bound horizon before upstream construction.
    values["actor_rollout_ref.actor.optim.total_training_steps"] = optimizer_horizon
    values["actor_rollout_ref.rollout.cudagraph_capture_sizes"] = list(
        TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES
    )
    values["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] = (
        config.distributed.actor_optimizer_offload
    )
    if world_size == 4:
        values.update(_WORLD4_TOPOLOGY_OVERRIDES)


def _assert_matched_mathematical_controls(
    values: Mapping[str, object],
    *,
    config: PolicyE2ESmokeRunConfig,
    optimizer_horizon: int,
) -> None:
    world_size = config.distributed.world_size
    if world_size not in TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES:
        raise ValueError("trainable TGVF topology must use world4 or world8")
    if type(optimizer_horizon) is not int or optimizer_horizon <= 0:
        raise ValueError("trainable TGVF optimizer horizon must be positive")
    expected = dict(PRL14_CROP16_COMMON_OVERRIDES)
    expected.update(_matched_batch_overrides(config))
    expected["actor_rollout_ref.actor.optim.total_training_steps"] = optimizer_horizon
    expected["actor_rollout_ref.rollout.cudagraph_capture_sizes"] = list(
        TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES
    )
    expected["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] = (
        config.distributed.actor_optimizer_offload
    )
    if world_size == 4:
        expected.update(_WORLD4_TOPOLOGY_OVERRIDES)
    mismatches = {
        path: (values.get(path), value)
        for path, value in expected.items()
        if values.get(path) != value
    }
    unexpected = {
        path: values[path] for path in PRL14_CROP16_REMOVED_OVERRIDES if path in values
    }
    if mismatches or unexpected:
        raise ValueError(
            "trainable TGVF differs from the matched mathematical controls: "
            f"mismatches={mismatches!r}, unexpected={unexpected!r}"
        )


def _functional_canary_dataset(
    config: PolicyE2ESmokeRunConfig,
) -> tuple[Path, str]:
    """Select a canary split that is valid for the configured reward.

    Stage3-shaped reward requires an immutable utility label for every sampled
    row.  Its sidecar is intentionally scoped to the formal training prefix,
    while the historical smoke split is intentionally disjoint from formal
    data.  Use the first four labeled formal rows for that engineering gate;
    retain the source-covering smoke split for rewards without a sidecar.
    """

    # A canary must draw rows from the same immutable dataset bound by the run
    # config.  The teacher-mixture artifacts own a different ordered schedule
    # from the historical PRL13 train/smoke sentinels; selecting either legacy
    # sentinel here makes veRL emit sample ids that cannot exist in the bound
    # teacher-mixture index.  Use the mixture's own train file and let the
    # one-step trainer consume its prefix.
    if isinstance(
        config.dataset.runtime_binding,
        (
            PolicyTeacherQuarterMixRuntimeBinding,
            PolicyTeacherRatioMixRuntimeBinding,
        ),
    ):
        return config.dataset.root / "samples.jsonl", "bound_train_prefix"

    tool_utility = config.reward.tool_utility
    if tool_utility is None:
        return DEEPEYES_SMOKE_SENTINEL, "smoke"

    selected = _verified_schedule_index().train[:TRAINABLE_TGVF_CANARY_PROMPTS]
    if len(selected) != TRAINABLE_TGVF_CANARY_PROMPTS:
        raise ValueError("functional utility-labeled canary prefix is incomplete")
    for ordinal, sample in enumerate(selected):
        label = tool_utility.label_for_sample(sample.sample_id)
        if label.training_index != ordinal:
            raise ValueError(
                "functional utility-labeled canary order differs from sidecar"
            )
    return DEEPEYES_TRAIN_SENTINEL, "utility_labeled_train_prefix"


def _apply_functional_canary_controls(
    values: dict[str, object], config: PolicyE2ESmokeRunConfig
) -> None:
    """Shrink canary policy work while retaining room for tool observations.

    The serialized 512-token budget counts policy-sampled tokens only.  veRL's
    response tensors also carry environment-owned TGVF observation tokens, so
    their transport width must remain larger than that budget.  Reuse the
    a derived safe width instead of asking ``tokenizer.pad(max_length=512)``
    to truncate (it does not) or silently dropping observation tokens.
    """

    sentinel, _ = _functional_canary_dataset(config)
    values.update(
        {
            "data.train_files": [str(sentinel)],
            "data.val_files": [str(sentinel)],
            "data.train_batch_size": TRAINABLE_TGVF_CANARY_PROMPTS,
            "data.gen_batch_size": TRAINABLE_TGVF_CANARY_PROMPTS,
            "data.max_response_length": (
                TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH
            ),
            "actor_rollout_ref.actor.ppo_mini_batch_size": (
                TRAINABLE_TGVF_CANARY_PROMPTS
            ),
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.rollout.n": (TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT),
            "actor_rollout_ref.rollout.response_length": (
                TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH
            ),
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
        }
    )


def _assert_functional_canary_config(config: PolicyE2ESmokeRunConfig) -> None:
    """Require a serialized canary identity instead of mutating a formal run."""

    expected = {
        "distributed.world_size": (config.distributed.world_size, 4),
        "sampling.trajectories_per_prompt": (
            config.policy.sampling.trajectories_per_prompt,
            TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT,
        ),
        "sampling.max_response_length": (
            config.policy.sampling.max_response_length,
            TRAINABLE_TGVF_CANARY_POLICY_TOKEN_BUDGET,
        ),
        "accumulation.global_prompt_batch_size": (
            config.accumulation.global_prompt_batch_size,
            TRAINABLE_TGVF_CANARY_PROMPTS,
        ),
        "accumulation.prompt_micro_batch_size_per_rank": (
            config.accumulation.prompt_micro_batch_size_per_rank,
            1,
        ),
        "accumulation.rollout_prompt_micro_batch_size_per_engine": (
            config.accumulation.rollout_prompt_micro_batch_size_per_engine,
            1,
        ),
        "accumulation.gradient_accumulation_steps": (
            config.accumulation.gradient_accumulation_steps,
            1,
        ),
        "scheduler.total_steps": (config.scheduler.total_steps, 1),
        "training.maximum_optimizer_steps": (
            config.training.maximum_optimizer_steps,
            1,
        ),
        "training.checkpoint_steps": (config.training.checkpoint_steps, (0, 1)),
    }
    mismatches = {
        name: (actual, required)
        for name, (actual, required) in expected.items()
        if actual != required
    }
    if mismatches:
        raise ValueError(
            f"functional canary run config differs: mismatches={mismatches!r}"
        )
    _functional_canary_dataset(config)


def _plain_binding(
    binding: DeepEyesTGVFMatchedDatasetBinding
    | PolicyTeacherQuarterMixDatasetBinding
    | PolicyTeacherRatioMixDatasetBinding,
) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(binding).items()
    }


def _matched_dataset_binding(
    config: PolicyE2ESmokeRunConfig,
) -> (
    DeepEyesTGVFMatchedDatasetBinding
    | PolicyTeacherQuarterMixDatasetBinding
    | PolicyTeacherRatioMixDatasetBinding
):
    crop_tgvf = (
        config.schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA
    )
    binding_type = (
        DeepEyesCropTGVFMatchedDatasetBinding
        if crop_tgvf
        else DeepEyesTGVFMatchedDatasetBinding
    )
    prompt_identity = (
        CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
        if crop_tgvf
        else TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
    )
    if isinstance(config.dataset.runtime_binding, PolicyTeacherRatioMixRuntimeBinding):
        runtime_binding = config.dataset.runtime_binding
        return PolicyTeacherRatioMixDatasetBinding(
            root=config.dataset.root,
            manifest_file_sha256=runtime_binding.manifest_file_sha256,
            content_sha256=runtime_binding.content_sha256,
            samples_sha256=config.dataset.samples_sha256,
            iteration_identity_sha256=config.dataset.iteration_identity_sha256,
            schedule_seed=runtime_binding.schedule_seed,
            expected_sample_count=runtime_binding.expected_sample_count,
            teacher_percentage=runtime_binding.teacher_percentage,
            tool_profile=config.protocol.tool_profile,
            visual_prompt_bundle_sha256=prompt_identity.bundle_sha256,
            thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
            model_name=config.model.model_name,
            tokenizer_length=config.model.tokenizer_length,
            chat_template_sha256=config.model.chat_template_sha256,
        )
    if isinstance(
        config.dataset.runtime_binding, PolicyTeacherQuarterMixRuntimeBinding
    ):
        runtime_binding = config.dataset.runtime_binding
        return PolicyTeacherQuarterMixDatasetBinding(
            root=config.dataset.root,
            manifest_file_sha256=runtime_binding.manifest_file_sha256,
            content_sha256=runtime_binding.content_sha256,
            samples_sha256=config.dataset.samples_sha256,
            iteration_identity_sha256=config.dataset.iteration_identity_sha256,
            schedule_seed=runtime_binding.schedule_seed,
            expected_sample_count=runtime_binding.expected_sample_count,
            tool_profile=config.protocol.tool_profile,
            visual_prompt_bundle_sha256=prompt_identity.bundle_sha256,
            thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
            model_name=config.model.model_name,
            tokenizer_length=config.model.tokenizer_length,
            chat_template_sha256=config.model.chat_template_sha256,
        )
    return binding_type(
        root=DEEPEYES_T1_ROOT,
        candidate_sidecar_path=DEEPEYES_CANDIDATE_SIDECAR,
        manifest_file_sha256=DEEPEYES_T1_MANIFEST_FILE_SHA256,
        content_sha256=DEEPEYES_T1_CONTENT_SHA256,
        samples_sha256=DEEPEYES_T1_SAMPLES_SHA256,
        candidate_sidecar_sha256=DEEPEYES_CANDIDATE_SHA256,
        expected_sample_count=DEEPEYES_T1_SAMPLE_COUNT,
        schedule_mode="stratified",
        schedule_seed=DEEPEYES_TRAIN_SEED,
        probe_seed=DEEPEYES_PROBE_SEED,
        visual_prompt_bundle_sha256=prompt_identity.bundle_sha256,
        thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
        model_name=config.model.model_name,
        tokenizer_length=config.model.tokenizer_length,
        chat_template_sha256=config.model.chat_template_sha256,
    )


def _legacy_base_plan(config: PolicyE2ESmokeRunConfig):
    """Build e003's complete base map with its historical prompt binding only."""

    legacy_prompt = (
        config.protocol.prompt_sha256
        if isinstance(
            config.dataset.runtime_binding,
            (
                PolicyTeacherQuarterMixRuntimeBinding,
                PolicyTeacherRatioMixRuntimeBinding,
            ),
        )
        else visual_tool_prompt_identity(
            config.protocol.tool_profile,
            assistant_dialect=native_assistant_dialect_for_model(
                config.model.model_name
            ),
        ).bundle_sha256
    )
    compatible = replace(
        config,
        # The generic e003 launcher uses the historical v1 schema to select
        # Crop-16's expanded-trajectory batch contract. Project v2 configs
        # differ only in RP66 optimizer ownership, so project them onto that
        # same proven base path while retaining their canonical run identity.
        schema_version=POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
        protocol=replace(config.protocol, prompt_sha256=legacy_prompt),
    )
    return legacy_launcher.build_policy_e2e_smoke_verl_plan(compatible)


def _matched_dataset_runtime_identity(
    config: PolicyE2ESmokeRunConfig,
) -> tuple[str, str, str, str]:
    if isinstance(config.dataset.runtime_binding, PolicyTeacherRatioMixRuntimeBinding):
        visual_agent_name = (
            CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            if config.schema_version
            == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA
            else TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
        )
        return (
            POLICY_TEACHER_RATIO_MIX_DATASET_CLASS,
            POLICY_TEACHER_RATIO_MIX_DATASET_CLASS.rsplit(".", 1)[-1],
            POLICY_TEACHER_RATIO_MIX_CONFIG_NAME,
            visual_agent_name,
        )
    if isinstance(
        config.dataset.runtime_binding, PolicyTeacherQuarterMixRuntimeBinding
    ):
        visual_agent_name = (
            CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            if config.schema_version
            == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA
            else TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
        )
        return (
            POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS,
            POLICY_TEACHER_QUARTER_MIX_DATASET_CLASS.rsplit(".", 1)[-1],
            POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME,
            visual_agent_name,
        )
    if config.schema_version == POLICY_E2E_CROP_TGVF_TFREE_MATCHED_RUN_CONFIG_SCHEMA:
        return (
            CROP_TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
            "CropTGVFDeepEyesMatchedDataset",
            "deepeyes_crop_tgvf_matched",
            CROP_TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
        )
    return (
        TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
        "TGVFDeepEyesMatchedDataset",
        "deepeyes_tgvf_matched",
        TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
    )


def _matched_dataset_module_path(config: PolicyE2ESmokeRunConfig) -> str:
    if isinstance(config.dataset.runtime_binding, PolicyTeacherRatioMixRuntimeBinding):
        return POLICY_TEACHER_RATIO_MIX_DATASET_MODULE_PATH
    return (
        POLICY_TEACHER_QUARTER_MIX_DATASET_MODULE_PATH
        if isinstance(
            config.dataset.runtime_binding, PolicyTeacherQuarterMixRuntimeBinding
        )
        else TRAINABLE_TGVF_DATASET_MODULE_PATH
    )


def _matched_dataset_files(
    config: PolicyE2ESmokeRunConfig,
) -> tuple[Path, Path, int]:
    if isinstance(
        config.dataset.runtime_binding,
        (
            PolicyTeacherQuarterMixRuntimeBinding,
            PolicyTeacherRatioMixRuntimeBinding,
        ),
    ):
        return (
            config.dataset.root / "samples.jsonl",
            DEEPEYES_PROBE_SENTINEL,
            config.dataset.runtime_binding.schedule_seed,
        )
    return DEEPEYES_TRAIN_SENTINEL, DEEPEYES_PROBE_SENTINEL, DEEPEYES_TRAIN_SEED


def _adapter_weight_sync_payload(config: PolicyE2ESmokeRunConfig) -> str:
    return (
        "full_qwen_plus_trainable_rp66"
        if config.representation.adapter_trainable
        else "full_qwen_plus_frozen_rp66"
    )


def _permanent_checkpoint_steps(
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode,
    target_step: int,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> list[int]:
    if mode != "formal":
        return []
    if horizon_extension is not None:
        return [
            horizon_extension.source_optimizer_step,
            horizon_extension.target_optimizer_step,
        ]
    configured = list(config.training.permanent_checkpoint_steps)
    if configured:
        return [step for step in configured if step <= target_step]
    return [target_step]


def _replace_custom_record(
    values: dict[str, object],
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode,
    checkpoint_steps: tuple[int, ...],
    output_root: Path,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> None:
    custom = dict(values["actor_rollout_ref.rollout.custom"])  # type: ignore[arg-type]
    protocol = dict(custom["protocol"])  # type: ignore[arg-type]
    protocol.update(
        {
            "prompt_sha256": config.protocol.prompt_sha256,
            "maximum_tool_calls": 6,
        }
    )
    reward = dict(custom["reward"])  # type: ignore[arg-type]
    reward.update(
        {
            "schema_version": config.schema_version,
            "source_aware": True,
        }
    )
    every_completed_step = mode == "formal"
    permanent_steps = _permanent_checkpoint_steps(
        config,
        mode=mode,
        target_step=checkpoint_steps[-1],
        horizon_extension=horizon_extension,
    )
    custom.update(
        {
            "schema_version": TRAINABLE_TGVF_LAUNCH_SCHEMA,
            "protocol": protocol,
            "reward": reward,
            "trainable_tgvf": {
                "external_module": TRAINABLE_TGVF_EXTERNAL_MODULE,
                "model_type": TRAINABLE_TGVF_MODEL_TYPE,
                "checkpoint_manager_fqn": (
                    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
                ),
                "policy_lora": False,
                "vision_trainable": True,
                "adapter_update_mode": (
                    config.representation.adapter_update_mode.value
                ),
                "adapter_trainable": config.representation.adapter_trainable,
                "sync_every_optimizer_step": True,
            },
            "weight_sync": {
                "mode": config.distributed.weight_sync_mode,
                "interval_optimizer_steps": 1,
                "payload": _adapter_weight_sync_payload(config),
            },
            "reference_diagnostic": {
                "enabled": False,
                "coefficient": 0.0,
                "worker_route": "disabled_zero_kl_control",
                "observation_source": "not_computed",
            },
            "checkpoint_steps": list(checkpoint_steps),
            "checkpoint_lifecycle": {
                "schema_version": POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
                "checkpoint_steps": list(checkpoint_steps),
                "every_completed_step": every_completed_step,
                "rolling_retention_across_restarts": True,
                "rolling_max_checkpoints": 2,
                "permanent_steps": permanent_steps,
                "permanent_directory": (
                    str(output_root / "permanent-checkpoints")
                    if permanent_steps
                    else ""
                ),
            },
            "runtime_state_directory": str(output_root / "runtime-policy-state"),
            "metrics_path": str(output_root / "metrics.jsonl"),
            "launch_mode": mode,
        }
    )
    if mode == "canary":
        _, canary_split = _functional_canary_dataset(config)
        custom["functional_canary"] = {
            "minimum_successful_tgvf_observations": 1,
            "failure_boundary": "before_optimizer_mutation",
            "dataset_split": canary_split,
        }
    values["actor_rollout_ref.rollout.custom"] = custom


@dataclass(frozen=True, slots=True)
class TrainableTGVFVerlLaunchPlan:
    config: PolicyE2ESmokeRunConfig
    mode: TrainableTGVFLaunchMode
    target_step: int
    overrides: Mapping[str, object]
    environment: Mapping[str, str]
    external_components: Mapping[str, str]
    horizon_extension: PolicyHorizonExtension | None = None
    schema_version: str = TRAINABLE_TGVF_LAUNCH_SCHEMA
    main_module: str = UPSTREAM_VERL_MAIN_MODULE
    config_name: str = UPSTREAM_VERL_CONFIG_NAME
    runner_fqn: str = UPSTREAM_VERL_V0_RUNNER_FQN

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(
            self,
            "external_components",
            MappingProxyType(dict(self.external_components)),
        )
        expected = (
            self.horizon_extension.target_optimizer_step
            if self.mode == "formal" and self.horizon_extension is not None
            else self.target_step
            if self.mode == "formal"
            and 0
            < self.target_step
            <= self.config.training.maximum_optimizer_steps
            else TRAINABLE_TGVF_SMOKE_TARGET
            if self.mode == "smoke"
            else TRAINABLE_TGVF_CANARY_TARGET
            if self.mode == "canary"
            else None
        )
        if self.schema_version != TRAINABLE_TGVF_LAUNCH_SCHEMA or expected is None:
            raise ValueError("trainable TGVF launch identity differs")
        if self.target_step != expected:
            raise ValueError(f"{self.mode} target must be step {expected}")
        self._assert_trainable_path()

    def _assert_trainable_path(self) -> None:
        values = self.overrides
        (
            _dataset_class,
            dataset_class_name,
            dataset_config_name,
            visual_agent_name,
        ) = _matched_dataset_runtime_identity(self.config)
        dataset_module_path = _matched_dataset_module_path(self.config)
        if (
            self.config.schema_version
            not in TRAINABLE_TGVF_SUPPORTED_RUN_CONFIG_SCHEMAS
        ):
            raise ValueError("trainable TGVF launcher requires an RP66 run schema")
        if self.mode == "canary":
            _assert_functional_canary_config(self.config)
            canary_sentinel, canary_split = _functional_canary_dataset(self.config)
            if TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH < (
                TRAINABLE_TGVF_CANARY_MIN_RESPONSE_TRANSPORT_LENGTH
            ):
                raise ValueError(
                    "functional canary response transport cannot hold its "
                    "policy and environment token bounds"
                )
            canary_expected = {
                "data.train_files": [str(canary_sentinel)],
                "data.val_files": [str(canary_sentinel)],
                "data.train_batch_size": TRAINABLE_TGVF_CANARY_PROMPTS,
                "data.gen_batch_size": TRAINABLE_TGVF_CANARY_PROMPTS,
                "data.max_response_length": (
                    TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH
                ),
                "actor_rollout_ref.actor.ppo_mini_batch_size": (
                    TRAINABLE_TGVF_CANARY_PROMPTS
                ),
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
                "actor_rollout_ref.rollout.n": (
                    TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT
                ),
                "actor_rollout_ref.rollout.response_length": (
                    TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH
                ),
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
                "trainer.n_gpus_per_node": 4,
            }
            mismatches = {
                path: (values.get(path), expected)
                for path, expected in canary_expected.items()
                if values.get(path) != expected
            }
            if mismatches:
                raise ValueError(
                    f"functional canary controls differ: mismatches={mismatches!r}"
                )
        else:
            _assert_matched_mathematical_controls(
                values,
                config=self.config,
                optimizer_horizon=self.config.scheduler.total_steps,
            )
        required = {
            "actor_rollout_ref.model.lora_rank": 0,
            "actor_rollout_ref.model.lora.rank": 0,
            "actor_rollout_ref.model.lora.freeze_vision_model": False,
            "actor_rollout_ref.model.lora.freeze_vision_projection": False,
            "actor_rollout_ref.actor.freeze_vision_tower": False,
            "actor_rollout_ref.model.external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "actor_rollout_ref.model.model_type": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs": {
                TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {
                    "adapter_update_mode": (
                        self.config.representation.adapter_update_mode.value
                    )
                }
            },
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                visual_agent_name
            ),
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.actor.loss_agg_mode": NATIVE_DEEPEYES_LOSS_AGG_MODE,
            "data.custom_cls.path": dataset_module_path,
            "data.custom_cls.name": dataset_class_name,
            "data.train_batch_size": (
                TRAINABLE_TGVF_CANARY_PROMPTS
                if self.mode == "canary"
                else self.config.accumulation.global_prompt_batch_size
            ),
            "actor_rollout_ref.rollout.n": (
                TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT
                if self.mode == "canary"
                else self.config.policy.sampling.trajectories_per_prompt
            ),
            "trainer.total_training_steps": self.target_step,
        }
        for path, expected in required.items():
            if values.get(path) != expected:
                raise ValueError(f"trainable TGVF override differs at {path}")
        matched_binding_path = f"data.{dataset_config_name}"
        if matched_binding_path not in values:
            raise ValueError(
                f"trainable TGVF matched dataset binding is missing at {matched_binding_path}"
            )
        custom = values.get("actor_rollout_ref.rollout.custom")
        if not isinstance(custom, Mapping):
            raise ValueError("trainable TGVF custom record is missing")
        expected_trainable_tgvf = {
            "external_module": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "model_type": TRAINABLE_TGVF_MODEL_TYPE,
            "checkpoint_manager_fqn": (TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN),
            "policy_lora": False,
            "vision_trainable": True,
            "adapter_update_mode": (
                self.config.representation.adapter_update_mode.value
            ),
            "adapter_trainable": self.config.representation.adapter_trainable,
            "sync_every_optimizer_step": True,
        }
        if custom.get("trainable_tgvf") != expected_trainable_tgvf:
            raise ValueError("trainable TGVF adapter identity differs")
        if custom.get("weight_sync") != {
            "mode": self.config.distributed.weight_sync_mode,
            "interval_optimizer_steps": 1,
            "payload": _adapter_weight_sync_payload(self.config),
        }:
            raise ValueError("trainable TGVF sync identity differs")
        reward = custom.get("reward")
        if not isinstance(reward, Mapping) or reward.get("schema_version") != (
            self.config.schema_version
        ):
            raise ValueError("trainable TGVF reward schema differs")
        if custom.get("reference_diagnostic") != {
            "enabled": False,
            "coefficient": 0.0,
            "worker_route": "disabled_zero_kl_control",
            "observation_source": "not_computed",
        }:
            raise ValueError("trainable TGVF reference diagnostic must be disabled")
        expected_checkpoint_steps = (
            list(range(self.target_step + 1))
            if self.mode == "formal"
            else [0, 1]
        )
        expected_permanent_steps = _permanent_checkpoint_steps(
            self.config,
            mode=self.mode,
            target_step=self.target_step,
            horizon_extension=self.horizon_extension,
        )
        if custom.get("checkpoint_steps") != expected_checkpoint_steps:
            raise ValueError("trainable TGVF checkpoint schedule differs")
        if custom.get("checkpoint_lifecycle") != {
            "schema_version": POLICY_CHECKPOINT_LIFECYCLE_SCHEMA,
            "checkpoint_steps": expected_checkpoint_steps,
            "every_completed_step": self.mode == "formal",
            "rolling_retention_across_restarts": True,
            "rolling_max_checkpoints": 2,
            "permanent_steps": expected_permanent_steps,
            "permanent_directory": (
                str(self.config.output.root / "permanent-checkpoints")
                if expected_permanent_steps
                else ""
            ),
        }:
            raise ValueError("trainable TGVF checkpoint lifecycle differs")
        if self.environment.get(POLICY_REFERENCE_DIAGNOSTIC_ENV) != "0":
            raise ValueError("trainable TGVF reference diagnostic environment differs")
        requires_observation = self.environment.get(
            POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV
        )
        if self.mode == "canary":
            if requires_observation != "1":
                raise ValueError(
                    "functional canary must require a successful TGVF observation"
                )
            if custom.get("functional_canary") != {
                "minimum_successful_tgvf_observations": 1,
                "failure_boundary": "before_optimizer_mutation",
                "dataset_split": canary_split,
            }:
                raise ValueError("functional canary evidence contract differs")
        elif requires_observation is not None:
            raise ValueError(
                "matched formal/smoke launch inherited the canary observation gate"
            )

    def hydra_override_args(self) -> tuple[str, ...]:
        return tuple(
            f"++{path}={legacy_launcher._hydra_literal(value)}"
            for path, value in self.overrides.items()
        )


def build_trainable_tgvf_verl_launch_plan(
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode = "formal",
    target_step: int | None = None,
    smoke_id: str | None = None,
    horizon_extension: PolicyHorizonExtension | None = None,
) -> TrainableTGVFVerlLaunchPlan:
    if config.schema_version not in TRAINABLE_TGVF_SUPPORTED_RUN_CONFIG_SCHEMAS:
        raise ValueError("trainable TGVF launcher requires an RP66 run schema")
    if mode not in {"formal", "smoke", "canary"}:
        raise ValueError(f"unsupported trainable TGVF launch mode: {mode!r}")
    if smoke_id is not None:
        if mode != "smoke":
            raise ValueError("smoke_id is valid only in smoke mode")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", smoke_id) is None:
            raise ValueError("smoke_id must be a safe lowercase run label")
    if horizon_extension is not None:
        if mode != "formal":
            raise ValueError("Policy horizon extension is valid only in formal mode")
        horizon_extension.validate_for_config(config)
        if target_step is not None and target_step != horizon_extension.target_optimizer_step:
            raise ValueError("explicit target step differs from horizon extension")
    resolved_target = (
        horizon_extension.target_optimizer_step
        if horizon_extension is not None
        else
        target_step or config.training.maximum_optimizer_steps
        if mode == "formal"
        else TRAINABLE_TGVF_SMOKE_TARGET
        if mode == "smoke"
        else TRAINABLE_TGVF_CANARY_TARGET
        if mode == "canary"
        else -1
    )
    if mode == "formal" and horizon_extension is None and not (
        0 < resolved_target <= config.training.maximum_optimizer_steps
    ):
        raise ValueError(
            "formal target must be between 1 and the run-config maximum "
            f"({config.training.maximum_optimizer_steps})"
        )
    if mode != "formal" and target_step is not None and target_step != resolved_target:
        raise ValueError(f"{mode} target must be step {resolved_target}")
    base = _legacy_base_plan(config)
    values = dict(base.overrides)
    binding = _matched_dataset_binding(config)
    (
        dataset_class,
        dataset_class_name,
        dataset_config_name,
        visual_agent_name,
    ) = _matched_dataset_runtime_identity(config)
    dataset_module_path = _matched_dataset_module_path(config)
    train_file, probe_file, dataset_seed = _matched_dataset_files(config)
    values.pop("data.tgvf_policy_t1_mixed", None)
    values.pop(f"data.{POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME}", None)
    values.pop(f"data.{POLICY_TEACHER_RATIO_MIX_CONFIG_NAME}", None)
    values.update(
        {
            "data.train_files": [str(train_file)],
            "data.val_files": [str(probe_file)],
            "data.custom_cls.path": dataset_module_path,
            "data.custom_cls.name": dataset_class_name,
            f"data.{dataset_config_name}": _plain_binding(binding),
            "data.seed": dataset_seed,
            "actor_rollout_ref.model.external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "actor_rollout_ref.model.model_type": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_rollout_ref.model.lora_rank": 0,
            "actor_rollout_ref.model.lora.rank": 0,
            "actor_rollout_ref.model.lora.freeze_vision_model": False,
            "actor_rollout_ref.model.lora.freeze_vision_projection": False,
            "actor_rollout_ref.model.lora.freeze_language_model": False,
            "actor_rollout_ref.actor.freeze_vision_tower": False,
            "actor_rollout_ref.actor.loss_agg_mode": NATIVE_DEEPEYES_LOSS_AGG_MODE,
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                visual_agent_name
            ),
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs": {
                TGVF_CHECKPOINT_ENGINE_CONTROL_KEY: {
                    "adapter_update_mode": (
                        config.representation.adapter_update_mode.value
                    )
                }
            },
            "trainer.total_training_steps": resolved_target,
            "actor_rollout_ref.actor.optim.total_training_steps": (
                config.scheduler.total_steps
            ),
            "trainer.save_freq": 1,
        }
    )
    _apply_matched_mathematical_controls(
        values,
        config=config,
        optimizer_horizon=config.scheduler.total_steps,
    )
    if mode == "canary":
        _assert_functional_canary_config(config)
        _apply_functional_canary_controls(values, config)
    checkpoint_steps = tuple(range(resolved_target + 1)) if mode == "formal" else (0, 1)
    output_root = config.output.root
    environment = dict(base.environment)
    if horizon_extension is not None:
        environment.update(horizon_extension.environment)
    if mode == "smoke":
        output_root = output_root / "smoke"
        if smoke_id is not None:
            output_root = output_root / smoke_id
        values.update(
            {
                "trainer.experiment_name": config.run_id + "-SMOKE",
                # One-step engineering canaries belong in the local console
                # log only.  Keep the run config's W&B backend unchanged for
                # the formal experiment.
                "trainer.logger": ["console"],
                "trainer.default_local_dir": str(output_root / "checkpoints"),
                "trainer.max_actor_ckpt_to_keep": 2,
            }
        )
        environment["TGVF_POLICY_STATE_DIR"] = str(output_root / "runtime-policy-state")
    elif mode == "canary":
        output_root = output_root / "canary"
        values.update(
            {
                "trainer.experiment_name": config.run_id + "-CANARY",
                "trainer.logger": ["console"],
                "trainer.default_local_dir": str(output_root / "checkpoints"),
                "trainer.max_actor_ckpt_to_keep": 2,
            }
        )
        environment["TGVF_POLICY_STATE_DIR"] = str(output_root / "runtime-policy-state")
        environment[POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV] = "1"
    environment[POLICY_METRICS_PATH_ENV] = str(output_root / "metrics.jsonl")
    environment[POLICY_REFERENCE_DIAGNOSTIC_ENV] = "0"
    values["ray_kwargs.ray_init._temp_dir"] = str(
        legacy_launcher._ray_temp_dir(output_root)
    )
    _replace_custom_record(
        values,
        config,
        mode=mode,
        checkpoint_steps=checkpoint_steps,
        output_root=output_root,
        horizon_extension=horizon_extension,
    )
    external = dict(base.external_components)
    external.update(
        {
            "control_run": PRL14_CROP16_RUN_NAME,
            "control_completion_sha256": PRL14_CROP16_COMPLETION_SHA256,
            "dataset": dataset_class,
            "actor_engine": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "checkpoint_engine_manager": (TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN),
        }
    )
    return TrainableTGVFVerlLaunchPlan(
        config=config,
        mode=mode,
        target_step=resolved_target,
        overrides=values,
        environment=environment,
        external_components=external,
        horizon_extension=horizon_extension,
    )


def compose_trainable_tgvf_verl_config(plan: TrainableTGVFVerlLaunchPlan) -> object:
    if not isinstance(plan, TrainableTGVFVerlLaunchPlan):
        raise TypeError("plan must be TrainableTGVFVerlLaunchPlan")
    composed = compose_pinned_verl_config(plan.hydra_override_args())
    from omegaconf import OmegaConf

    missing = object()
    for path, expected in plan.overrides.items():
        selected = OmegaConf.select(composed, path, default=missing)
        if selected is missing:
            raise ValueError(f"pinned compose omitted trainable TGVF override {path}")
        actual = (
            OmegaConf.to_container(selected, resolve=True)
            if OmegaConf.is_config(selected)
            else selected
        )
        if actual != legacy_launcher._plain(expected):
            raise ValueError(f"pinned compose changed trainable TGVF override {path}")
    return composed


def apply_trainable_tgvf_launch_environment(
    plan: TrainableTGVFVerlLaunchPlan,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Install one run's environment without inheriting optional prior-run state."""

    if not isinstance(plan, TrainableTGVFVerlLaunchPlan):
        raise TypeError("plan must be TrainableTGVFVerlLaunchPlan")
    target = os.environ if environment is None else environment
    for name in _OPTIONAL_PARENT_LAUNCH_ENV:
        if name not in plan.environment:
            target.pop(name, None)
    target.update(plan.environment)
    return target


def preflight_trainable_tgvf_verl_runtime(
    plan: TrainableTGVFVerlLaunchPlan | None = None,
    composed: object | None = None,
) -> VerlDistributionIdentity:
    """Resolve all static dependencies before Ray or GPU workers are started."""

    if (plan is None) != (composed is None):
        raise TypeError("plan and composed config must be provided together")
    if plan is not None:
        assert_policy_execution_identity(
            plan.config,
            repository_root=Path(__file__).resolve().parents[4],
            horizon_extension=policy_horizon_extension_from_environment(plan.config),
        )

    load_verl_public_api(expected_commit=SPIKE_CANDIDATE_VERL_COMMIT)
    identity = installed_verl_distribution_identity()
    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if not isinstance(CheckpointEngineManager, type):
        raise RuntimeError("pinned veRL CheckpointEngineManager is unavailable")
    if not callable(getattr(FusedLinearForPPO, "forward", None)):
        raise RuntimeError("pinned veRL FusedLinearForPPO is unavailable")
    from tgvf_rl.framework.vllm import load_vllm_public_plugin_api

    load_vllm_public_plugin_api()
    if plan is None and composed is None:
        return identity

    from tgvf_rl.rewards.deepeyes_verl_reward import (
        DEEPEYES_MAXIMUM_ATTEMPTS_ENV,
        DEEPEYES_RETRY_BACKOFF_SECONDS_ENV,
        DEEPEYES_RETRY_MAXIMUM_SECONDS_ENV,
        DEEPEYES_RUN_GLOBAL_CONCURRENCY_CAP_ENV,
        DEEPEYES_TRANSIENT_FAILURE_FRACTION_ENV,
        effective_deepeyes_judge_transport_config,
        effective_run_global_judge_concurrency,
        load_deepeyes_judge_service_config,
    )

    from .deepeyes_official_dataset import _verified_schedule_index
    from .policy_runtime import _validate_trainer_runtime_identity
    from .policy_task_runner import (
        _actor_scheduler_horizon,
        _resolved_policy_metrics_path,
    )
    from .policy_weight_sync import PolicyWeightSyncState

    config = plan.config
    _actor_scheduler_horizon(composed)
    _validate_trainer_runtime_identity(composed, config)
    _resolved_policy_metrics_path(config, environment=os.environ)
    weight_state = PolicyWeightSyncState.from_environment(os.environ)
    if (
        weight_state.run_id != config.run_id
        or weight_state.run_identity_sha256 != config.identity_sha256
    ):
        raise RuntimeError("Policy preflight weight-sync identity differs")
    configured_path = os.environ.get("TGVF_POLICY_RUN_CONFIG_PATH")
    if configured_path is None or Path(configured_path).resolve() != config.source_path:
        raise RuntimeError("Policy preflight run-config path differs")
    preflight_trainable_rp66_artifact(config)
    _verified_schedule_index()
    if (
        config.reward.judge_config_path is None
        or config.reward.judge_config_sha256 is None
    ):
        raise RuntimeError("Policy preflight judge binding is missing")
    judge = load_deepeyes_judge_service_config(
        config.reward.judge_config_path,
        expected_file_sha256=config.reward.judge_config_sha256,
    )
    if not os.environ.get(judge.api_key_env, "").strip():
        raise RuntimeError(
            f"Policy preflight requires judge credential {judge.api_key_env}"
        )
    effective_judge_concurrency = effective_run_global_judge_concurrency(
        judge.maximum_concurrency,
        worker_count=config.distributed.world_size,
    )
    effective_judge = effective_deepeyes_judge_transport_config(judge)
    _append_launch_provenance(
        plan,
        identity,
        answer_judge_transport={
            "configured_run_global_maximum_concurrency": (
                judge.maximum_concurrency
            ),
            "effective_run_global_maximum_concurrency": (
                effective_judge_concurrency
            ),
            "worker_count": config.distributed.world_size,
            "runtime_cap_environment_name": (
                DEEPEYES_RUN_GLOBAL_CONCURRENCY_CAP_ENV
            ),
            "runtime_cap_environment_value": os.environ.get(
                DEEPEYES_RUN_GLOBAL_CONCURRENCY_CAP_ENV
            ),
            "configured_maximum_attempts": judge.maximum_attempts,
            "effective_maximum_attempts": effective_judge.maximum_attempts,
            "configured_retry_backoff_seconds": judge.retry_backoff_seconds,
            "effective_retry_backoff_seconds": (
                effective_judge.retry_backoff_seconds
            ),
            "configured_retry_maximum_seconds": judge.retry_maximum_seconds,
            "effective_retry_maximum_seconds": (
                effective_judge.retry_maximum_seconds
            ),
            "configured_maximum_transient_failure_fraction": (
                judge.maximum_transient_failure_fraction
            ),
            "effective_maximum_transient_failure_fraction": (
                effective_judge.maximum_transient_failure_fraction
            ),
            "retry_environment": {
                name: os.environ.get(name)
                for name in (
                    DEEPEYES_MAXIMUM_ATTEMPTS_ENV,
                    DEEPEYES_RETRY_BACKOFF_SECONDS_ENV,
                    DEEPEYES_RETRY_MAXIMUM_SECONDS_ENV,
                    DEEPEYES_TRANSIENT_FAILURE_FRACTION_ENV,
                )
            },
        },
    )
    return identity


def _append_launch_provenance(
    plan: TrainableTGVFVerlLaunchPlan,
    verl_identity: VerlDistributionIdentity,
    *,
    answer_judge_transport: Mapping[str, object] | None = None,
) -> None:
    """Best-effort durable provenance; bookkeeping must never kill training."""

    project_root = Path(__file__).resolve().parents[4]
    project: dict[str, object]
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        clean, state_sha256, changes = _local_git_state(project_root)
        project = {
            "root": str(project_root),
            "commit": commit,
            "clean": clean,
            "source_state_sha256": state_sha256,
            "changes": list(changes),
        }
    except Exception as error:  # provenance must not become an admission gate
        project = {
            "root": str(project_root),
            "unavailable": f"{type(error).__name__}: {error}",
        }
    record = {
        "schema_version": "tgvf.prl15-launch-provenance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "run_id": plan.config.run_id,
        "run_identity_sha256": plan.config.identity_sha256,
        "run_config_path": str(plan.config.source_path),
        "run_config_file_sha256": plan.config.source_sha256,
        "mode": plan.mode,
        "target_step": plan.target_step,
        "project": project,
        "verl": asdict(verl_identity),
    }
    if answer_judge_transport is not None:
        record["answer_judge_transport"] = dict(answer_judge_transport)
    metrics_path = Path(plan.environment[POLICY_METRICS_PATH_ENV])
    destination = metrics_path.parent / "launch-provenance.jsonl"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        warnings.warn(
            f"could not persist PRL15 launch provenance: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("formal", "smoke", "canary"), default="formal"
    )
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--smoke-id")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_policy_e2e_smoke_run_config(args.run_config.resolve())
    horizon_extension = policy_horizon_extension_from_environment(config)
    plan = build_trainable_tgvf_verl_launch_plan(
        config,
        mode=args.mode,
        target_step=args.target_step,
        smoke_id=args.smoke_id,
        horizon_extension=horizon_extension,
    )
    apply_trainable_tgvf_launch_environment(plan)
    composed = compose_trainable_tgvf_verl_config(plan)
    if args.compose_only:
        return
    preflight_trainable_tgvf_verl_runtime(plan, composed)
    run_pinned_verl_config(composed)


if __name__ == "__main__":
    main()


__all__ = [
    "TRAINABLE_TGVF_EXTERNAL_MODULE",
    "TRAINABLE_TGVF_CANARY_PROMPTS",
    "TRAINABLE_TGVF_CANARY_POLICY_TOKEN_BUDGET",
    "TRAINABLE_TGVF_CANARY_MIN_RESPONSE_TRANSPORT_LENGTH",
    "TRAINABLE_TGVF_CANARY_RESPONSE_TRANSPORT_LENGTH",
    "TRAINABLE_TGVF_CANARY_ROLLOUTS_PER_PROMPT",
    "TRAINABLE_TGVF_CANARY_TARGET",
    "TRAINABLE_TGVF_LAUNCH_SCHEMA",
    "TRAINABLE_TGVF_ACTOR_OPTIMIZER_OFFLOAD",
    "TRAINABLE_TGVF_ROLLOUT_CUDAGRAPH_CAPTURE_SIZES",
    "TRAINABLE_TGVF_SMOKE_TARGET",
    "TRAINABLE_TGVF_SUPPORTED_WORLD_SIZES",
    "POLICY_REQUIRE_SUCCESSFUL_TGVF_OBSERVATION_ENV",
    "TrainableTGVFVerlLaunchPlan",
    "apply_trainable_tgvf_launch_environment",
    "build_trainable_tgvf_verl_launch_plan",
    "compose_trainable_tgvf_verl_config",
    "main",
    "preflight_trainable_tgvf_verl_runtime",
]
