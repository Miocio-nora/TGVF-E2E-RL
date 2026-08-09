"""Pinned-veRL launcher for the matched full-Qwen plus trainable-RP66 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal

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
from tgvf_rl.policy.deepeyes_official_protocol import THINKLITE_PROMPT_IDENTITY
from tgvf_rl.policy.run_config import (
    POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
    PolicyE2ESmokeRunConfig,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.protocol import visual_tool_prompt_identity
from tgvf_rl.protocol.native import native_assistant_dialect_for_model

from . import launcher as legacy_launcher
from .launcher import (
    UPSTREAM_VERL_CONFIG_NAME,
    UPSTREAM_VERL_MAIN_MODULE,
    UPSTREAM_VERL_V0_RUNNER_FQN,
)
from .native_deepeyes_runtime import (
    NATIVE_DEEPEYES_LOSS_AGG_MODE,
    NATIVE_DEEPEYES_POLICY_LOSS_MODE,
)
from .policy_main import compose_pinned_verl_config
from .policy_task_runner import (
    POLICY_METRICS_PATH_ENV,
    POLICY_REFERENCE_DIAGNOSTIC_ENV,
)
from .tgvf_deepeyes_matched_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
    DeepEyesTGVFMatchedDatasetBinding,
    TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
    TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME,
)
from .trainable_tgvf_checkpoint_manager import (
    TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN,
)
from .trainable_tgvf_engine import TRAINABLE_TGVF_MODEL_TYPE


TRAINABLE_TGVF_LAUNCH_SCHEMA = "tgvf.trainable-rp66-verl-launch.v1"
TRAINABLE_TGVF_EXTERNAL_MODULE = "tgvf_rl.framework.verl.trainable_tgvf_external"
TRAINABLE_TGVF_DATASET_MODULE_PATH = (
    "pkg://tgvf_rl.framework.verl.tgvf_deepeyes_matched_dataset"
)
TRAINABLE_TGVF_FORMAL_TARGET = 8
TRAINABLE_TGVF_SMOKE_TARGET = 1
TrainableTGVFLaunchMode = Literal["formal", "smoke"]


def _plain_binding(binding: DeepEyesTGVFMatchedDatasetBinding) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(binding).items()
    }


def _matched_dataset_binding(
    config: PolicyE2ESmokeRunConfig,
) -> DeepEyesTGVFMatchedDatasetBinding:
    return DeepEyesTGVFMatchedDatasetBinding(
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
        visual_prompt_bundle_sha256=(
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256
        ),
        thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
        model_name=config.model.model_name,
        tokenizer_length=config.model.tokenizer_length,
        chat_template_sha256=config.model.chat_template_sha256,
    )


def _legacy_base_plan(config: PolicyE2ESmokeRunConfig):
    """Build e003's complete base map with its historical prompt binding only."""

    legacy_prompt = visual_tool_prompt_identity(
        config.protocol.tool_profile,
        assistant_dialect=native_assistant_dialect_for_model(config.model.model_name),
    ).bundle_sha256
    compatible = replace(
        config,
        protocol=replace(config.protocol, prompt_sha256=legacy_prompt),
    )
    # Git worktrees do not copy the ignored development-header bundle.  Reuse
    # the canonical project's exact bundle for the base plan's CPU preflight.
    old_root = legacy_launcher._PYTHON312_DEV_INCLUDE_ROOT
    old_include = legacy_launcher._PYTHON312_DEV_INCLUDE
    old_cpath = legacy_launcher._TRITON_CPATH
    local_python_h = old_include / "Python.h"
    if not local_python_h.is_file():
        canonical_root = (
            Path(__file__).resolve().parents[4].parent
            / "tgvf-e2e-rl/.deps/python312-dev/root/usr/include"
        )
        canonical_include = canonical_root / "python3.12"
        if not (canonical_include / "Python.h").is_file():
            raise ValueError("pinned Python 3.12 development headers are missing")
        legacy_launcher._PYTHON312_DEV_INCLUDE_ROOT = canonical_root
        legacy_launcher._PYTHON312_DEV_INCLUDE = canonical_include
        legacy_launcher._TRITON_CPATH = os.pathsep.join(
            (str(canonical_root), str(canonical_include))
        )
    try:
        return legacy_launcher.build_policy_e2e_smoke_verl_plan(compatible)
    finally:
        legacy_launcher._PYTHON312_DEV_INCLUDE_ROOT = old_root
        legacy_launcher._PYTHON312_DEV_INCLUDE = old_include
        legacy_launcher._TRITON_CPATH = old_cpath


def _replace_custom_record(
    values: dict[str, object],
    config: PolicyE2ESmokeRunConfig,
    *,
    mode: TrainableTGVFLaunchMode,
    checkpoint_steps: tuple[int, ...],
    output_root: Path,
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
            "schema_version": POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA,
            "source_aware": True,
        }
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
                "adapter_trainable": True,
                "sync_every_optimizer_step": True,
            },
            "weight_sync": {
                "mode": config.distributed.weight_sync_mode,
                "interval_optimizer_steps": 1,
                "payload": "full_qwen_plus_trainable_rp66",
            },
            "reference_diagnostic": {
                "enabled": False,
                "coefficient": 0.0,
                "worker_route": "disabled_zero_kl_control",
                "observation_source": "not_computed",
            },
            "checkpoint_steps": list(checkpoint_steps),
            "runtime_state_directory": str(output_root / "runtime-policy-state"),
            "metrics_path": str(output_root / "metrics.jsonl"),
            "launch_mode": mode,
        }
    )
    values["actor_rollout_ref.rollout.custom"] = custom


@dataclass(frozen=True, slots=True)
class TrainableTGVFVerlLaunchPlan:
    config: PolicyE2ESmokeRunConfig
    mode: TrainableTGVFLaunchMode
    target_step: int
    overrides: Mapping[str, object]
    environment: Mapping[str, str]
    external_components: Mapping[str, str]
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
            TRAINABLE_TGVF_FORMAL_TARGET
            if self.mode == "formal"
            else TRAINABLE_TGVF_SMOKE_TARGET
            if self.mode == "smoke"
            else None
        )
        if self.schema_version != TRAINABLE_TGVF_LAUNCH_SCHEMA or expected is None:
            raise ValueError("trainable TGVF launch identity differs")
        if self.target_step != expected:
            raise ValueError(f"{self.mode} target must be step {expected}")
        self._assert_trainable_path()

    def _assert_trainable_path(self) -> None:
        values = self.overrides
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
            "actor_rollout_ref.rollout.agent.default_agent_loop": (
                TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            ),
            "actor_rollout_ref.actor.policy_loss.loss_mode": (
                NATIVE_DEEPEYES_POLICY_LOSS_MODE
            ),
            "actor_rollout_ref.actor.loss_agg_mode": NATIVE_DEEPEYES_LOSS_AGG_MODE,
            "data.custom_cls.path": TRAINABLE_TGVF_DATASET_MODULE_PATH,
            "data.custom_cls.name": "TGVFDeepEyesMatchedDataset",
            "data.train_batch_size": 16,
            "actor_rollout_ref.rollout.n": 16,
            "trainer.total_training_steps": self.target_step,
        }
        for path, expected in required.items():
            if values.get(path) != expected:
                raise ValueError(f"trainable TGVF override differs at {path}")
        custom = values.get("actor_rollout_ref.rollout.custom")
        if not isinstance(custom, Mapping):
            raise ValueError("trainable TGVF custom record is missing")
        if custom.get("weight_sync") != {
            "mode": self.config.distributed.weight_sync_mode,
            "interval_optimizer_steps": 1,
            "payload": "full_qwen_plus_trainable_rp66",
        }:
            raise ValueError("trainable TGVF sync identity differs")
        if custom.get("reference_diagnostic") != {
            "enabled": False,
            "coefficient": 0.0,
            "worker_route": "disabled_zero_kl_control",
            "observation_source": "not_computed",
        }:
            raise ValueError("trainable TGVF reference diagnostic must be disabled")
        if self.environment.get(POLICY_REFERENCE_DIAGNOSTIC_ENV) != "0":
            raise ValueError("trainable TGVF reference diagnostic environment differs")

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
) -> TrainableTGVFVerlLaunchPlan:
    if config.schema_version != POLICY_E2E_TRAINABLE_RP66_RUN_CONFIG_SCHEMA:
        raise ValueError("trainable TGVF launcher requires the RP66 run schema")
    if mode not in {"formal", "smoke"}:
        raise ValueError(f"unsupported trainable TGVF launch mode: {mode!r}")
    if smoke_id is not None:
        if mode != "smoke":
            raise ValueError("smoke_id is valid only in smoke mode")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", smoke_id) is None:
            raise ValueError("smoke_id must be a safe lowercase run label")
    resolved_target = (
        TRAINABLE_TGVF_FORMAL_TARGET
        if mode == "formal"
        else TRAINABLE_TGVF_SMOKE_TARGET
        if mode == "smoke"
        else -1
    )
    if target_step is not None and target_step != resolved_target:
        raise ValueError(f"{mode} target must be step {resolved_target}")
    base = _legacy_base_plan(config)
    values = dict(base.overrides)
    binding = _matched_dataset_binding(config)
    values.pop("data.tgvf_policy_t1_mixed", None)
    values.update(
        {
            "data.train_files": [str(DEEPEYES_TRAIN_SENTINEL)],
            "data.val_files": [str(DEEPEYES_PROBE_SENTINEL)],
            "data.custom_cls.path": TRAINABLE_TGVF_DATASET_MODULE_PATH,
            "data.custom_cls.name": "TGVFDeepEyesMatchedDataset",
            "data.deepeyes_tgvf_matched": _plain_binding(binding),
            "data.seed": DEEPEYES_TRAIN_SEED,
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
                TGVF_DEEPEYES_MATCHED_VISUAL_AGENT_NAME
            ),
            "actor_rollout_ref.rollout.checkpoint_manager_class": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
            "trainer.total_training_steps": resolved_target,
            "actor_rollout_ref.actor.optim.total_training_steps": resolved_target,
            "trainer.save_freq": 1,
        }
    )
    checkpoint_steps = (0, 1, 4, 8) if mode == "formal" else (0, 1)
    output_root = config.output.root
    environment = dict(base.environment)
    if mode == "smoke":
        output_root = output_root / "smoke"
        if smoke_id is not None:
            output_root = output_root / smoke_id
        values.update(
            {
                "trainer.experiment_name": config.run_id + "-SMOKE",
                # One-step engineering canaries belong in the local console
                # log only.  Keep the run config's W&B backend unchanged for
                # the formal eight-step experiment.
                "trainer.logger": ["console"],
                "trainer.default_local_dir": str(output_root / "checkpoints"),
                "trainer.max_actor_ckpt_to_keep": 2,
                # CUDA graph capture is a pure serving optimization and adds
                # minutes to every one-step restart.  Keep the scientific
                # BS16 x n16 workload while making smoke iterations eager.
                "actor_rollout_ref.rollout.enforce_eager": True,
            }
        )
        environment["TGVF_POLICY_STATE_DIR"] = str(
            output_root / "runtime-policy-state"
        )
    environment[POLICY_METRICS_PATH_ENV] = str(output_root / "metrics.jsonl")
    environment[POLICY_REFERENCE_DIAGNOSTIC_ENV] = "0"
    _replace_custom_record(
        values,
        config,
        mode=mode,
        checkpoint_steps=checkpoint_steps,
        output_root=output_root,
    )
    external = dict(base.external_components)
    external.update(
        {
            "dataset": TGVF_DEEPEYES_MATCHED_DATASET_CLASS,
            "actor_engine": TRAINABLE_TGVF_MODEL_TYPE,
            "actor_external_lib": TRAINABLE_TGVF_EXTERNAL_MODULE,
            "checkpoint_engine_manager": (
                TRAINABLE_TGVF_CHECKPOINT_ENGINE_MANAGER_FQN
            ),
        }
    )
    return TrainableTGVFVerlLaunchPlan(
        config=config,
        mode=mode,
        target_step=resolved_target,
        overrides=values,
        environment=environment,
        external_components=external,
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--smoke-id")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_policy_e2e_smoke_run_config(args.run_config)
    plan = build_trainable_tgvf_verl_launch_plan(
        config,
        mode=args.mode,
        target_step=args.target_step,
        smoke_id=args.smoke_id,
    )
    os.environ.update(plan.environment)
    if args.compose_only:
        compose_trainable_tgvf_verl_config(plan)
        return
    from .policy_main import main as policy_main

    policy_main(plan.hydra_override_args())


if __name__ == "__main__":
    main()


__all__ = [
    "TRAINABLE_TGVF_EXTERNAL_MODULE",
    "TRAINABLE_TGVF_FORMAL_TARGET",
    "TRAINABLE_TGVF_LAUNCH_SCHEMA",
    "TRAINABLE_TGVF_SMOKE_TARGET",
    "TrainableTGVFVerlLaunchPlan",
    "build_trainable_tgvf_verl_launch_plan",
    "compose_trainable_tgvf_verl_config",
    "main",
]
