#!/usr/bin/python3 -I
"""Compose, preflight, and explicitly launch one PRL13 horizon segment."""

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
            "tools/launch_prl13_native_deepeyes.py",
        ),
    )

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from tgvf_rl.ops.cli_authorization import (  # noqa: E402
    assert_legacy_standalone_execution_quarantined,
)


_DEEPEYES_NATIVE_VERL_COMMIT = "e003163181731412595257a72ec173071efb125f"


_DEFAULT_CONTRACT = (
    _ROOT / "configs/policy/runs/"
    "prl_13_a_qwen3_instruct_grpo_bs256_n16_native_crop_t1_stratified_"
    "80step_gpu0123.template.toml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=_DEFAULT_CONTRACT)
    parser.add_argument(
        "--mode",
        choices=(
            "formal",
            "smoke",
            "stress",
            "resident-stress",
            "resident-fast-stress",
            "resident-flex-stress",
            "resident-wide-flex-stress",
        ),
        default="formal",
        help=(
            "stress uses four fixed source-covering rows with formal n/token/"
            "micro-batch limits; resident-stress uses the same rows without "
            "CPU offload and with larger micro-batches; resident-fast-stress "
            "also removes activation recompute and FSDP forward resharding; "
            "resident-flex-stress keeps the safe resident shape and replaces "
            "SDPA with packed BlockMask attention; resident-wide-flex-stress "
            "also uses one checkpointed actor micro-batch per rank"
        ),
    )
    parser.add_argument("--target-step", type=int, default=1)
    parser.add_argument("--repository-root", type=Path, default=_ROOT)
    parser.add_argument("--verl-config-directory", type=Path)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start Ray/GPU/API work. Omit for compose-only CPU preflight.",
    )
    return parser


def _assert_pinned_checkout(repository_root: Path) -> Path:
    checkout = repository_root / ".deps/verl"
    if not checkout.is_dir():
        raise RuntimeError("pinned veRL checkout is missing")
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != _DEEPEYES_NATIVE_VERL_COMMIT:
        raise RuntimeError("veRL checkout is not pinned to e003")
    return checkout.resolve()


def _bind_pinned_verl_import(checkout: Path) -> None:
    """Bind this process and its future Ray workers to the checked sources."""

    loaded = tuple(
        name for name in sys.modules if name == "verl" or name.startswith("verl.")
    )
    if loaded:
        raise RuntimeError("veRL was imported before its pinned checkout was bound")
    sys.path.insert(0, str(checkout))
    from importlib.util import find_spec

    spec = find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pinned veRL package is not importable from its checkout")
    imported_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    expected_root = (checkout / "verl").resolve()
    if imported_root != expected_root:
        raise RuntimeError(
            f"veRL import identity differs: expected {expected_root}, got {imported_root}"
        )
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    worker_paths = [str(_SOURCE), str(checkout)]
    worker_paths.extend(
        value for value in inherited if value and value not in worker_paths
    )
    # Ray actors inherit process environment, not this driver's in-memory
    # sys.path mutations. Pin both the project Dataset/Reward modules and veRL.
    os.environ["PYTHONPATH"] = os.pathsep.join(worker_paths)


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/launch_prl13_native_deepeyes.py"
    )
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve(strict=True)
    checkout = _assert_pinned_checkout(repository_root)
    _bind_pinned_verl_import(checkout)

    # Import project bindings only after veRL is pinned: several of these
    # modules import ``verl`` transitively at module import time.
    from tgvf_rl.framework.verl.deepeyes_native_launcher import (
        apply_launch_environment,
        build_deepeyes_native_verl_launch_plan,
    )
    from tgvf_rl.framework.verl.prl13_main import (
        compose_pinned_deepeyes_config,
        preflight_pinned_deepeyes_config,
        run_pinned_deepeyes_config,
    )
    from tgvf_rl.policy.deepeyes_native_contract import (
        load_deepeyes_native_run_contract,
    )
    from tgvf_rl.rewards.deepeyes_verl_reward import (
        load_deepeyes_judge_service_config,
    )

    contract = load_deepeyes_native_run_contract(
        args.contract, allow_template=not args.launch
    )
    plan = build_deepeyes_native_verl_launch_plan(
        contract, mode=args.mode, target_step=args.target_step
    )
    config = compose_pinned_deepeyes_config(
        plan.hydra_override_args(),
        config_directory=args.verl_config_directory,
    )
    preflight = preflight_pinned_deepeyes_config(config)
    record = {**plan.as_record(), "compose_preflight": preflight}
    if not args.launch:
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    contract.assert_launchable(repository_root)
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY is required for a PRL13 launch")
    reward = contract.payload["reward"]
    load_deepeyes_judge_service_config(
        reward["judge_service_config_path"],
        expected_file_sha256=reward["judge_service_config_sha256"],
        require_launch_enabled=True,
    )
    if plan.horizon_already_satisfied():
        print(
            json.dumps(
                {**record, "status": "no_op_horizon_already_satisfied"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    apply_launch_environment(plan)
    run_pinned_deepeyes_config(config)
    checkpoint = plan.assert_target_checkpoint_complete()
    print(
        json.dumps(
            {
                **record,
                "status": "target_checkpoint_complete",
                "checkpoint": str(checkpoint),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
