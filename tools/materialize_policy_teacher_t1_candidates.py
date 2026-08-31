#!/usr/bin/env python3
"""Materialize the independent RP66/RP67 teacher population for T1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgvf_rl.data.policy_teacher_t1 import materialize_teacher_t1_candidates


DEFAULT_TRAIN = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v4_teacher_50k_clean_imend/splits/"
    "tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl"
)
DEFAULT_TEST = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v4_teacher_50k_clean_imend/splits/"
    "tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
)
DEFAULT_COREDEV = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/evaluation/"
    "BE-04-qwen3-tgvf-step80-coredev2511-gpu0123/runtime/"
    "coredev-official-tasks.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--coredev-tasks", type=Path, default=DEFAULT_COREDEV)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_teacher_t1_candidates(
        args.train,
        args.test,
        args.coredev_tasks,
        args.output_root,
    )
    print(json.dumps(result.as_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
