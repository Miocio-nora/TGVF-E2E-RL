#!/usr/bin/env python3
"""Launch a versioned native-Crop T-free control.

The rollout, full-model update, data schedule, prompt, Crop tool and policy
loss are inherited unchanged from the proven PRL14 native-Crop path.  The sole
scientific treatment is the versioned T-free Stage3 reward manager recorded
by the PRL21 overlay contract.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
VERL = Path(
    os.environ.get(
        "TGVF_VERL_CHECKOUT",
        "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/verl",
    )
)
CONFIG = (
    ROOT / "configs/policy/runs/"
    "prl_21_r0_qwen3_instruct_full_crop_bs16_n16_tfree_16step_ws8.toml"
)
sys.path[:0] = [str(SOURCE), str(VERL)]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SOURCE), str(VERL), os.environ.get("PYTHONPATH", "")]
)

from tgvf_rl.framework.verl.deepeyes_native_launcher import (  # noqa: E402
    apply_launch_environment,
    build_deepeyes_native_verl_launch_plan,
)
from tgvf_rl.framework.verl.prl13_main import (  # noqa: E402
    compose_pinned_deepeyes_config,
    preflight_pinned_deepeyes_config,
    run_pinned_deepeyes_config,
)
from tgvf_rl.policy.crop_tfree_contract import (  # noqa: E402
    CropTFreeRunContract,
    load_crop_tfree_run_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-config",
        type=_run_config_argument,
        default=CONFIG,
        help=(
            "Crop T-free overlay TOML. Relative paths are resolved from the "
            "repository root; the default remains the accepted PRL21 control."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "formal"),
        default="preflight",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start Ray/GPU/API work; preflight never starts GPU work.",
    )
    return parser


def _run_config_argument(value: str) -> Path:
    """Resolve an explicit experiment overlay without changing the default."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_complete(path: Path, *, world_size: int) -> bool:
    actor = path / "actor"
    model_shards = tuple(actor.glob(f"model_world_size_{world_size}_rank_*.pt"))
    return (
        path.is_dir()
        and actor.is_dir()
        and (path / "data.pt").is_file()
        and len(model_shards) == world_size
        and all(shard.stat().st_size > 0 for shard in model_shards)
        and (actor / "huggingface/config.json").is_file()
    )


def _retain_checkpoint(
    *,
    checkpoint_root: Path,
    output_root: Path,
    step: int,
    world_size: int,
) -> Path:
    source = checkpoint_root / f"global_step_{step}"
    destination = output_root / "permanent-checkpoints" / f"global_step_{step}"
    temporary = destination.parent / f".{destination.name}.partial"
    if _checkpoint_complete(destination, world_size=world_size):
        return destination
    if not _checkpoint_complete(source, world_size=world_size):
        raise RuntimeError(f"step-{step} source checkpoint is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, copy_function=os.link)
    os.replace(temporary, destination)
    if not _checkpoint_complete(destination, world_size=world_size):
        raise RuntimeError(f"retained step-{step} checkpoint is incomplete")
    return destination


def _retain_permanent_steps_when_ready(
    checkpoint_root: Path,
    output_root: Path,
    permanent_steps: tuple[int, ...],
    stop: threading.Event,
) -> None:
    pending = set(permanent_steps)
    while pending and not stop.wait(2.0):
        for step in tuple(sorted(pending)):
            source = checkpoint_root / f"global_step_{step}"
            if not _checkpoint_complete(source, world_size=8):
                continue
            _retain_checkpoint(
                checkpoint_root=checkpoint_root,
                output_root=output_root,
                step=step,
                world_size=8,
            )
            pending.remove(step)


def _common_reward_overrides(contract: CropTFreeRunContract) -> dict[str, object]:
    manager_name = contract.reward_manager_class.rsplit(".", 1)[-1]
    return {
        "reward.reward_manager.name": manager_name,
        "reward.reward_manager.module.path": contract.reward_manager_module_path,
        "reward.reward_manager.module.name": manager_name,
        "reward.deepeyes_official.judge_service_config_path": str(
            ROOT / "configs/policy/judges/prl13_qwen25_72b_binary_text_resilient.json"
        ),
        "reward.deepeyes_official.judge_service_config_sha256": (
            "fff705c59408f4863244ff28df3443176e85de83147344df6a2350859c233021"
        ),
    }


def _current_data_overrides(contract: CropTFreeRunContract) -> dict[str, object]:
    if contract.image_max_pixels is None:
        return {}
    return {"data.mm_processor_kwargs.max_pixels": contract.image_max_pixels}


def _build_plan(contract: CropTFreeRunContract, *, mode: str):
    if mode == "smoke":
        base = build_deepeyes_native_verl_launch_plan(
            contract.base_contract,
            mode="smoke",
            target_step=1,
        )
        output_root = contract.output_root / "smoke-integration"
        overrides = {
            **base.overrides,
            **_common_reward_overrides(contract),
            **_current_data_overrides(contract),
            "trainer.experiment_name": contract.run_id + "-SMOKE",
            "trainer.default_local_dir": str(output_root / "checkpoints"),
            "trainer.rollout_data_dir": str(output_root / "trajectories"),
            "trainer.validation_data_dir": str(output_root / "validation"),
            "trainer.logger": ["console"],
        }
        environment = {**base.environment, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        return replace(base, overrides=overrides, environment=environment)

    if mode != "formal":
        raise ValueError("plan mode must be smoke or formal")
    base = build_deepeyes_native_verl_launch_plan(
        contract.base_contract,
        mode="formal",
        # Eight is the proven PRL13/14 native-launch horizon gate. The
        # versioned Crop T-free overlay below owns the actual training horizon.
        target_step=8,
    )
    output_root = contract.output_root
    overrides = {
        **base.overrides,
        **_common_reward_overrides(contract),
        **_current_data_overrides(contract),
        "trainer.experiment_name": contract.run_id,
        "trainer.default_local_dir": str(output_root / "checkpoints"),
        "trainer.rollout_data_dir": str(output_root / "trajectories"),
        "trainer.validation_data_dir": str(output_root / "validation"),
        "trainer.total_training_steps": contract.maximum_optimizer_steps,
        "trainer.save_freq": 1,
        "trainer.max_actor_ckpt_to_keep": contract.maximum_rolling_checkpoints,
        "trainer.test_freq": 0,
        "trainer.resume_mode": "auto",
        "data.train_batch_size": contract.global_prompt_batch_size,
        "data.gen_batch_size": contract.global_prompt_batch_size,
        "actor_rollout_ref.actor.ppo_mini_batch_size": (
            contract.global_prompt_batch_size
        ),
        "trainer.n_gpus_per_node": 8,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 8,
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 8,
        "actor_rollout_ref.rollout.agent.num_workers": 8,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.65,
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.model.fused_kernel_options": {"impl_backend": "torch"},
        "actor_rollout_ref.model.override_config.text_config": {
            "_attn_implementation_internal": "flex_attention"
        },
        "actor_rollout_ref.model.override_config.vision_config": {
            "_attn_implementation_internal": "sdpa"
        },
    }
    environment = {
        **base.environment,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "TGVF_DEEPEYES_RUN_GLOBAL_JUDGE_CONCURRENCY_CAP": "8",
        "TGVF_DEEPEYES_JUDGE_MAXIMUM_ATTEMPTS": "8",
        "TGVF_DEEPEYES_JUDGE_RETRY_BACKOFF_SECONDS": "2",
        "TGVF_DEEPEYES_JUDGE_RETRY_MAXIMUM_SECONDS": "30",
        "TGVF_DEEPEYES_JUDGE_MAXIMUM_TRANSIENT_FAILURE_FRACTION": "0",
    }
    return replace(base, overrides=overrides, environment=environment)


def _record(
    contract: CropTFreeRunContract,
    plan: object,
    preflight: object,
    *,
    mode: str,
) -> dict[str, object]:
    return {
        **plan.as_record(),
        "schema_version": "tgvf.crop-tfree-launch-provenance.v2",
        "run_id": contract.run_id,
        "mode": mode,
        "overlay_config_path": str(contract.source_path),
        "overlay_config_sha256": contract.source_sha256,
        "overlay_identity_sha256": contract.identity_sha256,
        "base_contract_path": str(contract.base_contract.source_path),
        "base_contract_sha256": contract.base_contract.source_sha256,
        "reward_profile": contract.reward_profile,
        "reward_equation": (
            "2*A-0.05*max(0,N_attempt-1)-"
            f"{contract.protocol_error_penalty:g}[protocol_or_tool_error]"
        ),
        "global_prompt_batch_size": contract.global_prompt_batch_size,
        "gradient_accumulation_steps": contract.gradient_accumulation_steps,
        "permanent_checkpoint_steps": list(contract.permanent_checkpoint_steps),
        "compose_preflight": preflight,
        "created_at_unix_seconds": time.time(),
    }


def main() -> int:
    args = _parser().parse_args()
    contract = load_crop_tfree_run_contract(
        args.run_config,
        repository_root=ROOT,
        allow_placeholder=not args.launch,
    )
    plan_mode = "smoke" if args.mode == "smoke" else "formal"
    plan = _build_plan(contract, mode=plan_mode)
    config = compose_pinned_deepeyes_config(plan.hydra_override_args())
    preflight = preflight_pinned_deepeyes_config(config)
    expected_manager = contract.reward_manager_class
    if preflight.get("reward_manager_class") != expected_manager:
        raise RuntimeError("resolved Crop T-free reward manager identity differs")
    record = _record(contract, plan, preflight, mode=args.mode)
    if args.mode == "preflight" or not args.launch:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    contract.assert_launchable(ROOT)
    contract.base_contract.assert_launchable(ROOT)
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    output_root = (
        contract.output_root / "smoke-integration"
        if args.mode == "smoke"
        else contract.output_root
    )
    checkpoint_root = output_root / "checkpoints"
    if (
        checkpoint_root.exists()
        and not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file()
    ):
        raise RuntimeError("output root exists without a resumable checkpoint tracker")
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = output_root / "launch-provenance.jsonl"
    with provenance.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    stop = threading.Event()
    keeper: threading.Thread | None = None
    if args.mode == "formal":
        keeper = threading.Thread(
            target=_retain_permanent_steps_when_ready,
            args=(
                checkpoint_root,
                output_root,
                contract.permanent_checkpoint_steps,
                stop,
            ),
            name="retain-permanent-steps",
            daemon=True,
        )
        keeper.start()
    try:
        apply_launch_environment(plan)
        run_pinned_deepeyes_config(config)
    finally:
        stop.set()
        if keeper is not None:
            keeper.join(timeout=10.0)

    target = 1 if args.mode == "smoke" else contract.maximum_optimizer_steps
    world_size = 4 if args.mode == "smoke" else 8
    final_checkpoint = checkpoint_root / f"global_step_{target}"
    if not _checkpoint_complete(final_checkpoint, world_size=world_size):
        raise RuntimeError(f"step-{target} checkpoint is incomplete")
    if args.mode == "formal":
        retained = {
            step: _retain_checkpoint(
                checkpoint_root=checkpoint_root,
                output_root=output_root,
                step=step,
                world_size=8,
            )
            for step in contract.permanent_checkpoint_steps
        }
        penultimate = checkpoint_root / f"global_step_{target - 1}"
        if penultimate.is_dir():
            shutil.rmtree(penultimate)
        record.update(
            {
                "status": "target_checkpoint_complete",
                "final_checkpoint": str(final_checkpoint),
                "retained_checkpoints": {
                    str(step): str(path) for step, path in retained.items()
                },
            }
        )
    else:
        record.update(
            {
                "status": "smoke_checkpoint_complete",
                "final_checkpoint": str(final_checkpoint),
            }
        )
    record["completed_at_unix_seconds"] = time.time()
    (output_root / "completion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
