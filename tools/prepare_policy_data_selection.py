"""Prepare and reduce CPU-only Policy RL data-selection records."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.data.policy_selection import (  # noqa: E402
    build_selection_requests,
    canonical_json_line,
    records_sha256,
    reduce_selection_attempts,
    summarize_selection_decisions,
)
from tgvf_rl.data.policy_selection_canary import (  # noqa: E402
    build_t1_canary_selection,
)
from tgvf_rl.data.policy_selection_full import (  # noqa: E402
    materialize_t1_full_selection,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: at least one record is required")
    return records


def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            observed = 0
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank lines are forbidden")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: record must be a JSON object"
                    )
                observed += 1
                yield value
            if observed == 0:
                raise ValueError(f"{path}: at least one record is required")


def _write_jsonl_new(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only Qwen3 Policy RL data-selection preparation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    requests_parser = subparsers.add_parser("build-requests")
    requests_parser.add_argument("--candidates", type=Path, required=True)
    requests_parser.add_argument("--output", type=Path, required=True)
    requests_parser.add_argument("--oracle-attempts", type=int, default=0)

    canary_parser = subparsers.add_parser("build-canary")
    canary_parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    canary_parser.add_argument("--output", type=Path, required=True)
    canary_parser.add_argument("--manifest-output", type=Path, required=True)

    full_parser = subparsers.add_parser("build-full")
    full_parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    full_parser.add_argument("--output", type=Path, required=True)
    full_parser.add_argument("--manifest-output", type=Path, required=True)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--candidates", type=Path, required=True)
    reduce_parser.add_argument("--attempts", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--summary-output", type=Path, required=True)
    reduce_parser.add_argument("--expected-oracle-attempts", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "build-full":
        result = materialize_t1_full_selection(
            args.candidates,
            output_path=args.output,
            manifest_path=args.manifest_output,
        )
        print(json.dumps({"command": args.command, **result}, indent=2, sort_keys=True))
        return 0
    if args.command == "build-canary":
        result = build_t1_canary_selection(_iter_jsonl(args.candidates))
        _write_jsonl_new(
            args.output, (dict(record) for record in result.selected_candidates)
        )
        _write_json_new(args.manifest_output, dict(result.manifest))
        print(
            json.dumps(
                {
                    "command": args.command,
                    "records": len(result.selected_candidates),
                    "manifest_sha256": result.manifest_sha256,
                    "output": str(args.output.resolve()),
                    "manifest_output": str(args.manifest_output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    candidates = _read_jsonl(args.candidates)
    if args.command == "build-requests":
        records = list(
            build_selection_requests(candidates, oracle_attempts=args.oracle_attempts)
        )
        _write_jsonl_new(args.output, records)
        result = {
            "command": args.command,
            "records": len(records),
            "records_sha256": records_sha256(records),
            "output": str(args.output.resolve()),
        }
    else:
        attempts = _read_jsonl(args.attempts)
        records = list(
            reduce_selection_attempts(
                candidates,
                attempts,
                expected_oracle_attempts=args.expected_oracle_attempts,
            )
        )
        summary = summarize_selection_decisions(records)
        summary["decision_records_sha256"] = records_sha256(records)
        _write_jsonl_new(args.output, records)
        _write_json_new(args.summary_output, summary)
        result = {
            "command": args.command,
            "records": len(records),
            "records_sha256": summary["decision_records_sha256"],
            "output": str(args.output.resolve()),
            "summary_output": str(args.summary_output.resolve()),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
