#!/usr/bin/python3 -I
# ruff: noqa: E402
"""Profile one exact representation optimizer step under ``torchrun``.

This is a diagnostic launcher, not a training entry point.  It delegates the
run to the strict representation runner and only activates PyTorch's profiler
for one requested optimizer step.  Small high-level record-function scopes
make host/device gaps attributable without changing the training source path.
"""

from __future__ import annotations

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
            "tools/profile_representation_step.py",
        ),
    )

import argparse
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

import tgvf_rl.representation.training.native_pipeline as native_pipeline_module
import tgvf_rl.representation.training.streaming as streaming_module
import tgvf_rl.representation.training.trainer as trainer_module
from tgvf_rl.ops.cli_authorization import (
    assert_legacy_standalone_execution_quarantined,
)
from tgvf_rl.representation.training import runner


@dataclass(slots=True)
class _ScopeTiming:
    name: str
    wall_ns: int
    start: torch.cuda.Event
    end: torch.cuda.Event


class _StepProfiler:
    def __init__(self, *, profile_global_step: int, trace_dir: Path) -> None:
        if profile_global_step <= 0:
            raise ValueError("profile_global_step must be positive")
        self.profile_global_step = profile_global_step
        self.trace_dir = trace_dir
        self.active = False
        self.timings: list[_ScopeTiming] = []

    def scoped(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: object, **kwargs: object) -> Any:
            if not self.active:
                return function(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            started_ns = perf_counter_ns()
            with torch.profiler.record_function(name):
                start.record()
                result = function(*args, **kwargs)
                end.record()
            self.timings.append(
                _ScopeTiming(
                    name=name,
                    wall_ns=perf_counter_ns() - started_ns,
                    start=start,
                    end=end,
                )
            )
            return result

        return wrapped

    def install_static_scopes(self, stack: ExitStack) -> None:
        self._patch(
            stack,
            trainer_module,
            "score_streaming_same_image_group",
            "tgvf.score_group",
        )
        self._patch(
            stack,
            trainer_module,
            "backward_streaming_same_image_group",
            "tgvf.backward_group",
        )
        self._patch(
            stack,
            trainer_module,
            "score_streaming_same_image_groups",
            "tgvf.score_groups",
        )
        self._patch(
            stack,
            trainer_module,
            "backward_streaming_same_image_groups",
            "tgvf.backward_groups",
        )
        self._patch(
            stack,
            streaming_module,
            "_forward_cell_batch_losses",
            "tgvf.qwen_cell_batch",
        )
        for attribute, scope_name in (
            ("build_native_representation_messages", "tgvf.build_messages"),
            ("render_native_action_target", "tgvf.render_action"),
            ("render_native_evidence_labels", "tgvf.render_evidence"),
            ("_expand_native_visual_placeholders", "tgvf.expand_visual_tokens"),
            ("_qwen3_position_ids", "tgvf.mrope_positions"),
            ("_source_visual_identity", "tgvf.source_identity"),
        ):
            self._patch(
                stack,
                native_pipeline_module,
                attribute,
                scope_name,
            )

    def profile_runner_step(
        self,
        original: Callable[..., Any],
        trainer: trainer_module.RepresentationTrainer,
        **kwargs: object,
    ) -> Any:
        next_step = trainer.global_step + 1
        if next_step != self.profile_global_step:
            return original(trainer, **kwargs)
        rank = int(os.environ["RANK"])
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.timings.clear()
        with ExitStack() as instance_stack:
            group_builder = trainer.group_builder
            self._patch(
                instance_stack,
                trainer,
                "group_builder",
                "tgvf.group_builder",
            )
            self._patch(
                instance_stack,
                trainer.adapter,
                "forward",
                "tgvf.adapter_forward",
            )
            runtime = getattr(group_builder, "runtime", None)
            for attribute, scope_name in (
                ("_materialize_action_with_expansion", "tgvf.processor_action"),
                (
                    "_materialize_action_from_shared_visual",
                    "tgvf.shared_visual_action",
                ),
                ("_condition", "tgvf.target_condition"),
                ("_readout_row", "tgvf.readout_row"),
            ):
                if hasattr(group_builder, attribute):
                    self._patch(
                        instance_stack,
                        group_builder,
                        attribute,
                        scope_name,
                    )
            if runtime is not None:
                for attribute, scope_name in (
                    ("assert_bound_invariants", "tgvf.runtime_invariants"),
                    ("extract_vision_features", "tgvf.vision_features"),
                    ("make_adapter_input", "tgvf.make_adapter_input"),
                    ("build_target_condition", "tgvf.build_target_condition"),
                ):
                    if hasattr(runtime, attribute):
                        self._patch(
                            instance_stack,
                            runtime,
                            attribute,
                            scope_name,
                        )
            family_adapter = trainer.family_adapter
            if hasattr(family_adapter, "materialize_representation_supervision"):
                self._patch(
                    instance_stack,
                    family_adapter,
                    "materialize_representation_supervision",
                    "tgvf.materialize_supervision",
                )
            self.active = True
            try:
                with torch.profiler.profile(
                    activities=(
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as profile:
                    result = original(trainer, **kwargs)
            finally:
                self.active = False
        torch.cuda.synchronize()
        trace_path = self.trace_dir / f"rank-{rank}-step-{next_step}.json"
        profile.export_chrome_trace(str(trace_path))
        summary = {
            "rank": rank,
            "profile_global_step": next_step,
            "scopes": [
                {
                    "name": timing.name,
                    "wall_ms": timing.wall_ns / 1_000_000,
                    "default_stream_span_ms": timing.start.elapsed_time(timing.end),
                }
                for timing in self.timings
            ],
            "trace_path": str(trace_path.resolve()),
        }
        summary_path = self.trace_dir / f"rank-{rank}-step-{next_step}-summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        table_path = self.trace_dir / f"rank-{rank}-step-{next_step}-operators.txt"
        table_path.write_text(
            profile.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=80,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def _patch(
        self,
        stack: ExitStack,
        owner: object,
        attribute: str,
        scope_name: str,
    ) -> None:
        original = getattr(owner, attribute)
        setattr(owner, attribute, self.scoped(scope_name, original))
        stack.callback(setattr, owner, attribute, original)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile-global-step", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    assert_legacy_standalone_execution_quarantined(
        "tools/profile_representation_step.py"
    )
    args = _parser().parse_args(argv)
    profiler = _StepProfiler(
        profile_global_step=args.profile_global_step,
        trace_dir=args.trace_dir,
    )
    original_step = runner._run_train_step_with_periodic_telemetry
    with ExitStack() as stack:
        profiler.install_static_scopes(stack)

        def profiled_step(
            trainer: trainer_module.RepresentationTrainer,
            **kwargs: object,
        ) -> Any:
            return profiler.profile_runner_step(
                original_step,
                trainer,
                **kwargs,
            )

        runner._run_train_step_with_periodic_telemetry = profiled_step
        stack.callback(
            setattr,
            runner,
            "_run_train_step_with_periodic_telemetry",
            original_step,
        )
        result = runner.run_representation_training(args.config)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/python3 -I
