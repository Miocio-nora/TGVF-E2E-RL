"""Two-rank FSDP2 checkpoint/resume smoke for the accepted veRL environment.

This is deliberately an infrastructure objective over a tiny deterministic
model.  It proves the executable FSDP2 and distributed-checkpoint lifecycle; it
does not select GRPO/SDPO mathematics or claim a Qwen training result.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tgvf_rl.cli import validate_smoke_config  # noqa: E402
from tgvf_rl.compatibility_stack import (  # noqa: E402
    AUDITED_COMPATIBILITY_STACKS,
    CONTROL_COMPATIBILITY_STACK,
    audited_compatibility_stack,
)
from tgvf_rl.framework.verl import (  # noqa: E402
    load_verl_public_api,
    verify_verl_distribution_identity,
)
from tgvf_rl.experiment_identity import validate_run_id  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(width, width * 2)
        self.linear_out = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = torch.nn.functional.gelu(self.linear_in(values), approximate="none")
        return self.norm(self.linear_out(values) + residual)


class TinyModel(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(width) for _ in range(layers))
        self.output = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return self.output(values)


def _bounded_path(raw: Path, *, expected_parent: str) -> Path:
    path = raw if raw.is_absolute() else REPOSITORY_ROOT / raw
    path = path.resolve()
    allowed = (REPOSITORY_ROOT / "artifacts" / expected_parent).resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"path must be a child of {allowed}")
    return path


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return local.detach().to(device="cpu").contiguous().numpy().tobytes()


def _model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def _parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        result[name] = local.detach().clone()
    return result


def _build_model(
    *, width: int, layers: int, seed: int, rank: int, world_size: int
) -> TinyModel:
    torch.manual_seed(seed)
    model = TinyModel(width, layers).to(torch.device("cuda", rank))
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.0, foreach=False, fused=False
    )


def _batch(*, width: int, seed: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(2, 3, width, generator=generator)
    target = torch.randn(2, 3, width, generator=generator)
    device = torch.device("cuda", rank)
    return values.to(device), target.to(device)


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    output = model(batch[0])
    loss = torch.mean((output - batch[1]) ** 2)
    loss.backward()
    optimizer.step()
    return output.detach(), loss.detach()


def _assert_exact_parameters(
    expected: dict[str, torch.Tensor], actual_model: nn.Module
) -> None:
    actual = _parameter_snapshot(actual_model)
    if expected.keys() != actual.keys():
        raise AssertionError("resumed FSDP2 parameter names changed")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            delta = torch.max(torch.abs(expected[name] - actual[name])).item()
            raise AssertionError(
                f"resumed parameter {name} differs from control; max_abs={delta}"
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stack",
        choices=tuple(AUDITED_COMPATIBILITY_STACKS),
        default=CONTROL_COMPATIBILITY_STACK,
        help="named audited compatibility stack (default: control)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_stack = audited_compatibility_stack(args.stack)
    config = validate_smoke_config(
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config,
        stack_selector=args.stack,
    )
    run_id = validate_run_id(config.get("run_id"))
    checkpoint_dir = _bounded_path(args.checkpoint_dir, expected_parent="compatibility")
    output_path = _bounded_path(args.output, expected_parent="compatibility")
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint directory already exists: {checkpoint_dir}")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly '2,3'")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be exactly ':4096:8'")

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this bounded smoke requires exactly two ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")

        identity = verify_verl_distribution_identity(
            expected_commit=selected_stack.verl_commit
        )
        api = load_verl_public_api(expected_commit=selected_stack.verl_commit)
        model_config = config["model"]
        width = int(model_config["hidden_size"])
        layers = int(model_config["layers"])
        seed = int(model_config["seed"])

        model = _build_model(
            width=width, layers=layers, seed=seed, rank=rank, world_size=world_size
        )
        optimizer = _optimizer(model)
        first_batch = _batch(width=width, seed=seed + 1, rank=rank)
        second_batch = _batch(width=width, seed=seed + 2, rank=rank)
        _step(model, optimizer, first_batch)

        options = StateDictOptions(strict=True)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        checkpoint = {
            "model": model_state,
            "optimizer": optimizer_state,
            "extra": {"completed_steps": torch.tensor(1, dtype=torch.int64)},
        }
        dcp_save(checkpoint, checkpoint_id=checkpoint_dir)
        dist.barrier()

        control_output, control_loss = _step(model, optimizer, second_batch)
        control_parameters = _parameter_snapshot(model)
        control_digest = _model_digest(model)

        resumed_model = _build_model(
            width=width, layers=layers, seed=seed, rank=rank, world_size=world_size
        )
        resumed_optimizer = _optimizer(resumed_model)
        resumed_model_state, resumed_optimizer_state = get_state_dict(
            resumed_model, resumed_optimizer, options=options
        )
        resumed_checkpoint = {
            "model": resumed_model_state,
            "optimizer": resumed_optimizer_state,
            "extra": {"completed_steps": torch.tensor(0, dtype=torch.int64)},
        }
        dcp_load(resumed_checkpoint, checkpoint_id=checkpoint_dir)
        set_state_dict(
            resumed_model,
            resumed_optimizer,
            model_state_dict=resumed_checkpoint["model"],
            optim_state_dict=resumed_checkpoint["optimizer"],
            options=options,
        )
        if int(resumed_checkpoint["extra"]["completed_steps"].item()) != 1:
            raise AssertionError("strict extra checkpoint state did not resume")

        resumed_output, resumed_loss = _step(
            resumed_model, resumed_optimizer, second_batch
        )
        if not torch.equal(control_output, resumed_output):
            delta = torch.max(torch.abs(control_output - resumed_output)).item()
            raise AssertionError(
                f"resumed forward differs from control; max_abs={delta}"
            )
        if not torch.equal(control_loss, resumed_loss):
            raise AssertionError("resumed loss differs from control")
        _assert_exact_parameters(control_parameters, resumed_model)
        resumed_digest = _model_digest(resumed_model)
        if control_digest != resumed_digest:
            raise AssertionError("resumed model digest differs from control")

        rank_result: dict[str, Any] = {
            "rank": rank,
            "logical_device": rank,
            "physical_device": (2, 3)[rank],
            "device_name": torch.cuda.get_device_name(rank),
            "control_loss": float(control_loss.item()),
            "model_shard_sha256": control_digest,
            "resume_exact": True,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_result)
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "tgvf-fsdp2-smoke-result-v1",
                "run_id": run_id,
                "result": "PASS",
                "scope": config["scope"],
                "objective": dict(config["objective"]),
                "checkpoint": dict(config["checkpoint"]),
                "versions": {
                    name: metadata.version(name)
                    for name in ("torch", "transformers", "vllm", "verl")
                },
                "verl": {
                    "commit": identity.commit,
                    "package_version": identity.package_version,
                    "source_kind": identity.source_kind,
                    "source_clean": identity.source_clean,
                },
                "public_api": {
                    "fsdp_engine_config": api.fsdp_engine_config.__name__,
                    "checkpoint_handler": api.checkpoint_handler.__name__,
                },
                "world_size": world_size,
                "ranks": gathered,
            }
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(payload, sort_keys=True))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
