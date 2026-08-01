#!/usr/bin/env python3
"""Mirror representation JSONL telemetry to W&B without touching training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import tomllib
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not math.isfinite(args.poll_seconds):
        parser.error("--poll-seconds must be a finite positive number")
    if args.producer_pid is not None and args.producer_pid <= 0:
        parser.error("--producer-pid must be positive")
    return args


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise TypeError("representation config root must be a table")
    return value


def _wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    conditioning = config["conditioning"]
    data = config["data"]
    objective = config["objective"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    training = config["training"]
    fsdp2 = config["fsdp2"]
    initialization = config["initialization"]
    adapter_variant = config.get("adapter", {}).get("variant", "full_d_deepstack")
    global_batch = (
        data["train"]["batch_size"]
        * fsdp2["world_size"]
        * training["gradient_accumulation_steps"]
    )
    return {
        "phase": "representation",
        "run_id": config["run_id"],
        "schema_version": config["schema_version"],
        "code_commit": config["code"]["commit"],
        "model": model["model_name"],
        "model_family": model["family"],
        "dtype": model["dtype"],
        "attention_backend": model["attention_backend"],
        "image_max_pixels": model["image_max_pixels"],
        "conditioning_provider": conditioning["provider"],
        "conditioning_hidden_layer": conditioning.get("hidden_layer"),
        "conditioning_embedding_identity": conditioning.get("embedding_identity"),
        "adapter_variant": adapter_variant,
        "prompt_identity": config["prompt"]["identity"],
        "objective_identity": objective["identity"],
        "matrix_ce_mode": objective["matrix_ce_mode"],
        "matrix_ce_temperature": objective["matrix_ce_temperature"],
        "matrix_ce_weight": objective["matrix_ce_weight"],
        "l_gen_weight": objective["l_gen_weight"],
        "norm_weight": objective["norm_weight"],
        "manifold_weight": objective["manifold_weight"],
        "optimizer": optimizer["type"],
        "learning_rate": optimizer["learning_rate"],
        "weight_decay": optimizer["weight_decay"],
        "lr_scheduler": scheduler["kind"],
        "warmup_steps": scheduler["warmup_steps"],
        "target_optimizer_steps": training["target_optimizer_steps"],
        "validation_every_steps": training["validation_every_optimizer_steps"],
        "checkpoint_every_steps": config["checkpoint"]["save_every_optimizer_steps"],
        "micro_batch_size_per_rank": data["train"]["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "world_size": fsdp2["world_size"],
        "global_batch_size": global_batch,
        "reshard_after_forward": fsdp2["reshard_after_forward"],
        "seed": initialization["seed"],
        "train_source_sha256": data["train"]["source_sha256"],
        "validation_source_sha256": data["validation"]["source_sha256"],
    }


def _number(record: dict[str, Any], key: str) -> float | int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_payload(record: dict[str, Any]) -> dict[str, float | int]:
    event = record.get("event")
    step = _number(record, "global_step")
    if event not in {"train", "validation"} or step is None:
        return {}
    prefix = "train" if event == "train" else "validation"
    mapping = {
        "global_matrix_ce_loss": "matrix_ce_loss",
        "global_l_gen_loss": "l_gen_loss",
        "global_norm_loss": "norm_loss",
        "global_weighted_norm_loss": "weighted_norm_loss",
        "global_image_axis_loss": "image_axis_loss",
        "global_weighted_image_axis_loss": "weighted_image_axis_loss",
        "global_image_axis_correct_top1": "image_axis_correct_top1",
        "global_image_axis_score_gap": "image_axis_score_gap",
        "global_image_axis_row_count": "image_axis_row_count",
        "global_total_loss": "total_loss",
        "global_row_count": "row_count",
        "global_sample_count": "sample_count",
        "global_evidence_token_count": "evidence_token_count",
        "global_group_count": "group_count",
        "gradient_norm_before_clip": "gradient_norm_before_clip",
        "learning_rate": "learning_rate",
    }
    payload: dict[str, float | int] = {"global_step": int(step)}
    for source, destination in mapping.items():
        value = _number(record, source)
        if value is not None:
            payload[f"{prefix}/{destination}"] = value
    performance = record.get("performance")
    if event == "train" and isinstance(performance, dict):
        for source, destination in (
            ("max_rank_elapsed_seconds", "step_seconds"),
            ("global_rows_per_second", "rows_per_second"),
            ("global_matrices_per_second", "matrices_per_second"),
        ):
            value = _number(performance, source)
            if value is not None:
                payload[f"performance/{destination}"] = value
        ranks = performance.get("ranks")
        if isinstance(ranks, list) and ranks:
            for source, destination in (
                ("peak_allocated_bytes", "peak_allocated_gib"),
                ("peak_reserved_bytes", "peak_reserved_gib"),
            ):
                values = [
                    _number(rank, source) for rank in ranks if isinstance(rank, dict)
                ]
                finite = [float(value) for value in values if value is not None]
                if finite:
                    payload[f"performance/{destination}"] = max(finite) / 2**30
    return payload


def _records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        return []
    complete = raw if raw.endswith(b"\n") else raw.rsplit(b"\n", 1)[0] + b"\n"
    records: list[dict[str, Any]] = []
    for line in complete.decode("utf-8", errors="strict").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("metrics JSONL records must be objects")
        records.append(value)
    return records


def _load_next_line(path: Path) -> int:
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    next_line = value.get("next_line")
    if isinstance(next_line, bool) or not isinstance(next_line, int) or next_line < 0:
        raise ValueError("invalid W&B sidecar state")
    return next_line


def _save_next_line(path: Path, next_line: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"next_line": next_line}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _producer_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    args = _arguments()
    config = _read_toml(args.config)
    current_records = _records(args.metrics)
    if args.dry_run:
        payloads = [
            payload for row in current_records if (payload := metric_payload(row))
        ]
        print(json.dumps({"config": _wandb_config(config), "metrics": payloads}))
        return 0

    import wandb

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    provider_tag = str(config["conditioning"]["provider"]).replace("_", "-")
    matrix_ce_mode_tag = str(config["objective"]["matrix_ce_mode"]).replace("_", "-")
    matrix_ce_temperature_tag = (
        f"matrix-ce-temperature-{config['objective']['matrix_ce_temperature']}"
    )
    adapter_variant_tag = str(
        config.get("adapter", {}).get("variant", "full_d_deepstack")
    ).replace("_", "-")
    model_tag = (
        str(config["model"]["model_name"])
        .strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
    )
    method_tags: tuple[str, ...] = ()
    if "image-axis" in str(config["objective"]["identity"]).lower():
        method_tags = ("image-axis-grounded",)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.wandb_run_id,
        resume="allow",
        group="representation-phase",
        job_type="training-telemetry",
        tags=(
            "representation-phase",
            model_tag,
            provider_tag,
            f"adapter-{adapter_variant_tag}",
            f"matrix-ce-{matrix_ce_mode_tag}",
            matrix_ce_temperature_tag,
            *method_tags,
        ),
        config=_wandb_config(config),
        dir=str(args.wandb_dir),
        settings=wandb.Settings(init_timeout=60),
    )
    run.define_metric("global_step")
    for namespace in ("train/*", "validation/*", "performance/*"):
        run.define_metric(namespace, step_metric="global_step")
    print(json.dumps({"status": "started", "url": run.url}), flush=True)

    next_line = _load_next_line(args.state)
    exit_code = 0
    try:
        while True:
            records = _records(args.metrics)
            if next_line > len(records):
                raise RuntimeError("metrics JSONL was truncated while being mirrored")
            complete = False
            for line_index, record in enumerate(records[next_line:], start=next_line):
                payload = metric_payload(record)
                if payload:
                    run.log(payload)
                if record.get("event") == "start":
                    run.summary["run_identity_sha256"] = record.get(
                        "run_identity_sha256"
                    )
                    run.summary["canonical_config_sha256"] = record.get(
                        "canonical_config_sha256"
                    )
                if record.get("event") == "complete":
                    complete = True
                    run.summary["training_status"] = "complete"
                    run.summary["final_global_step"] = record.get("global_step")
                    run.summary["final_artifact_manifest_sha256"] = record.get(
                        "final_artifact_manifest_sha256"
                    )
                next_line = line_index + 1
                _save_next_line(args.state, next_line)
            if complete or args.once:
                break
            if not _producer_alive(args.producer_pid):
                run.summary["training_status"] = "producer_exited_without_complete"
                exit_code = 2
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
