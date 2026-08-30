#!/usr/bin/python3 -I
"""Launch the 16-step clean-final DeepEyes crop pilot on eight GPUs.

The run intentionally preserves the proven PRL13 full-model shape (BS16,
n=16, micro=32) while changing only the answer protocol and horizon.  veRL
saves every step so an interrupted overnight run can resume.  A hard-linked
copy of step 8 is retained outside veRL's two-checkpoint rolling window.
"""

from __future__ import annotations
# ruff: noqa: E402

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
            "tools/launch_prl14_cleanfinal16.py",
        ),
    )

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
VERL = ROOT / ".deps/verl"
CONTRACT = (
    ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.toml"
)
RUN_NAME = "PRL-14-A-QWEN3-INSTRUCT-GRPO-BS16-N16-NATIVE-CROP-T1-CLEANFINAL-16STEP-WS8"
OUTPUT_ROOT = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/policy/"
    "PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-"
    "cleanfinal-16step-ws8"
)
TARGET_STEP = 16
PERMANENT_STEP = 8


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
from tgvf_rl.policy.deepeyes_native_contract import (  # noqa: E402
    load_deepeyes_native_run_contract,
)
from tgvf_rl.ops.launch_gate import (  # noqa: E402
    consume_launch_authorization,
    make_run_identity,
    materialize_ready_receipt,
    write_process_liveness_receipt,
)
from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start Ray/GPU/API work; omit for compose-only preflight.",
    )
    parser.add_argument("--authorization-token", type=Path)
    parser.add_argument("--freeze-override", type=Path)
    return parser


def _checkpoint_complete(path: Path) -> bool:
    actor = path / "actor"
    return (
        path.is_dir()
        and actor.is_dir()
        and (path / "data.pt").is_file()
        and any(actor.glob("model_world_size_8_rank_*.pt"))
        and (actor / "huggingface/config.json").is_file()
    )


def _retain_step8(checkpoint_root: Path, stop: threading.Event) -> None:
    source = checkpoint_root / f"global_step_{PERMANENT_STEP}"
    retained_root = OUTPUT_ROOT / "permanent-checkpoints"
    destination = retained_root / f"global_step_{PERMANENT_STEP}"
    temporary = retained_root / f".global_step_{PERMANENT_STEP}.partial"
    while not stop.wait(2.0):
        if _checkpoint_complete(destination):
            return
        if not _checkpoint_complete(source):
            continue
        retained_root.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary, copy_function=os.link)
        os.replace(temporary, destination)
        if not _checkpoint_complete(destination):
            raise RuntimeError("retained step-8 checkpoint is incomplete")
        return


def _build_plan():
    contract = load_deepeyes_native_run_contract(CONTRACT)
    # Step 8 is an existing formal gate.  The dedicated pilot then overrides
    # only the absolute horizon to 16; keeping the base plan avoids changing
    # PRL13's historical 1/8/20/45/80 contract.
    base = build_deepeyes_native_verl_launch_plan(
        contract, mode="formal", target_step=PERMANENT_STEP
    )
    overrides = dict(base.overrides)
    overrides.update(
        {
            "trainer.experiment_name": RUN_NAME,
            "trainer.default_local_dir": str(OUTPUT_ROOT / "checkpoints"),
            "trainer.rollout_data_dir": str(OUTPUT_ROOT / "trajectories"),
            "trainer.validation_data_dir": str(OUTPUT_ROOT / "validation"),
            "trainer.total_training_steps": TARGET_STEP,
            # Keep one rolling recovery checkpoint plus the latest.  Step 8
            # is separately hard-linked by _retain_step8 before veRL removes
            # its original directory.
            "trainer.save_freq": 1,
            "trainer.max_actor_ckpt_to_keep": 2,
            # External CoreDev evaluation owns the step0/8/16 comparison.
            "trainer.test_freq": 0,
            "trainer.resume_mode": "auto",
            "reward.deepeyes_official.judge_service_config_path": str(
                ROOT / "configs/policy/judges/"
                "prl13_qwen25_72b_binary_text_resilient.json"
            ),
            "reward.deepeyes_official.judge_service_config_sha256": (
                "fff705c59408f4863244ff28df3443176e85de83147344df6a2350859c233021"
            ),
            "data.train_batch_size": 16,
            "data.gen_batch_size": 16,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
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
    )
    environment = dict(base.environment)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    return contract, replace(base, overrides=overrides, environment=environment)


def main() -> int:
    assert_legacy_standalone_execution_quarantined("tools/launch_prl14_cleanfinal16.py")
    args = _parser().parse_args()
    contract, plan = _build_plan()
    config = compose_pinned_deepeyes_config(plan.hydra_override_args())
    preflight = preflight_pinned_deepeyes_config(config)
    record = {
        **plan.as_record(),
        "run_name": RUN_NAME,
        "actual_target_step": TARGET_STEP,
        "permanent_checkpoint_step": PERMANENT_STEP,
        "output_root": str(OUTPUT_ROOT),
        "compose_preflight": preflight,
    }
    if not args.launch:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    contract.assert_launchable(ROOT)
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    checkpoint_root = OUTPUT_ROOT / "checkpoints"
    retained = OUTPUT_ROOT / "permanent-checkpoints/global_step_8"
    if (
        checkpoint_root.exists()
        and not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file()
    ):
        raise RuntimeError("output root exists without a resumable checkpoint tracker")

    training_identity = make_run_identity(
        run_id=RUN_NAME,
        phase="training",
        command_id="tools/launch_prl14_cleanfinal16.py",
        parameters={
            "contract_path": str(CONTRACT.resolve()),
            "target_step": TARGET_STEP,
        },
    )
    training_gate = OUTPUT_ROOT / "runtime/training-launch-gate"
    ready = materialize_ready_receipt(
        training_gate,
        run_identity=training_identity,
        evidence_paths={"run_contract": CONTRACT},
    )
    record["launch_gate_ready"] = ready
    if args.authorization_token is None:
        record["status"] = "ready_launch_denied"
        record["launch_gate_directory"] = str(training_gate)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 3
    consume_launch_authorization(
        training_gate,
        args.authorization_token,
        ROOT / "configs/ops/experiment_execution_policy.json",
        expected_run_id=RUN_NAME,
        expected_phase="training",
        freeze_override_path=args.freeze_override,
    )
    write_process_liveness_receipt(
        OUTPUT_ROOT / "runtime/training-liveness.json",
        run_identity=training_identity,
    )

    stop = threading.Event()
    keeper = threading.Thread(
        target=_retain_step8,
        args=(checkpoint_root, stop),
        name="retain-step8",
        daemon=True,
    )
    keeper.start()
    try:
        apply_launch_environment(plan)
        run_pinned_deepeyes_config(config)
    finally:
        stop.set()
        keeper.join(timeout=10.0)

    final_checkpoint = checkpoint_root / f"global_step_{TARGET_STEP}"
    if not _checkpoint_complete(final_checkpoint):
        raise RuntimeError("step-16 checkpoint is incomplete")
    if not _checkpoint_complete(retained):
        raise RuntimeError("permanent step-8 checkpoint is incomplete")

    # max_keep=2 leaves step 15 and step 16.  Step 16 plus permanent step 8
    # are the requested final retention set.
    penultimate = checkpoint_root / f"global_step_{TARGET_STEP - 1}"
    if penultimate.is_dir():
        shutil.rmtree(penultimate)

    record.update(
        {
            "status": "target_checkpoint_complete",
            "final_checkpoint": str(final_checkpoint),
            "retained_step8_checkpoint": str(retained),
            "completed_at_unix_seconds": time.time(),
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "completion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
