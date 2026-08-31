"""Strict, read-only field decoders for policy run configurations.

This leaf owns value, path, and immutable legacy-judge validation.  It does
not parse TOML, load judge runtimes, launch work, or import the public
``policy.run_config`` facade.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re

from tgvf_rl.conditioning import (
    TargetConditioningConfig,
    TargetConditioningProviderKind,
)
from tgvf_rl.contracts.tokens import LogProbMeasurement

from .run_config_schema import (
    POLICY_E2E_DEEPEYES_RULE_FIRST_JUDGE_CONFIG_SHA256,
    POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_CONFIG_SHA256,
    SmokeDistributedBinding,
)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FQN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _conditioning(value: object) -> TargetConditioningConfig:
    if not isinstance(value, Mapping) or "provider" not in value:
        raise ValueError("representation.conditioning must bind a provider")
    try:
        provider = TargetConditioningProviderKind(value["provider"])
    except (TypeError, ValueError) as error:
        raise ValueError("representation conditioning provider is invalid") from error
    if provider is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE:
        if set(value) != {"provider", "hidden_layer"}:
            raise ValueError("contextual conditioning fields differ")
        return TargetConditioningConfig(
            provider=provider,
            hidden_layer=_integer(
                value["hidden_layer"], name="representation.conditioning.hidden_layer"
            ),
        )
    if set(value) != {"provider", "embedding_identity"}:
        raise ValueError("target-token embedding conditioning fields differ")
    return TargetConditioningConfig(
        provider=provider,
        embedding_identity=_text(
            value["embedding_identity"],
            name="representation.conditioning.embedding_identity",
        ),
    )


def _distributed(table: Mapping[str, object]) -> SmokeDistributedBinding:
    physical = _nonnegative_int_tuple(
        table["physical_gpu_ids"], name="distributed.physical_gpu_ids"
    )
    logical = _nonnegative_int_tuple(
        table["logical_gpu_ids"], name="distributed.logical_gpu_ids"
    )
    actor = _nonnegative_int_tuple(
        table["actor_logical_gpu_ids"], name="distributed.actor_logical_gpu_ids"
    )
    rollout = _nonnegative_int_tuple(
        table["rollout_logical_gpu_ids"], name="distributed.rollout_logical_gpu_ids"
    )
    for name, values in (
        ("physical_gpu_ids", physical),
        ("logical_gpu_ids", logical),
        ("actor_logical_gpu_ids", actor),
        ("rollout_logical_gpu_ids", rollout),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"distributed.{name} must be non-empty and unique")
    world_size = _positive_int(table["world_size"], name="distributed.world_size")
    expected_logical = tuple(range(world_size))
    if logical != expected_logical:
        raise ValueError(
            "distributed.logical_gpu_ids must be contiguous from zero and match "
            "distributed.world_size"
        )
    if len(physical) != world_size:
        raise ValueError(
            "distributed.physical_gpu_ids must match distributed.world_size"
        )
    if actor != logical:
        raise ValueError(
            "this smoke requires every logical GPU in the FSDP2 actor world"
        )
    placement = _text(table["placement"], name="distributed.placement")
    if placement != "colocated" or rollout != actor:
        raise ValueError("this smoke requires colocated actor/rollout placement")
    _require_exact(table["fsdp_strategy"], "fsdp2", "distributed.fsdp_strategy")
    _require_exact(table["rollout_backend"], "vllm", "distributed.rollout_backend")
    tp = _positive_int(
        table["vllm_tensor_parallel_size"], name="distributed.vllm_tensor_parallel_size"
    )
    if len(rollout) % tp != 0:
        raise ValueError("vLLM tensor parallel size must divide rollout GPUs")
    weight_sync_interval = _positive_int(
        table["weight_sync_interval_optimizer_steps"],
        name="distributed.weight_sync_interval_optimizer_steps",
    )
    if weight_sync_interval != 1:
        raise ValueError(
            "distributed.weight_sync_interval_optimizer_steps must be 1: "
            "the pinned synchronous veRL trainer publishes actor weights after "
            "every optimizer step and has no interval scheduler"
        )
    return SmokeDistributedBinding(
        physical_gpu_ids=physical,
        logical_gpu_ids=logical,
        world_size=world_size,
        actor_logical_gpu_ids=actor,
        rollout_logical_gpu_ids=rollout,
        fsdp_strategy=table["fsdp_strategy"],
        fsdp_reshard_after_forward=_boolean(
            table["fsdp_reshard_after_forward"],
            name="distributed.fsdp_reshard_after_forward",
        ),
        rollout_backend=table["rollout_backend"],
        vllm_tensor_parallel_size=tp,
        placement=placement,
        weight_sync_mode=_text(
            table["weight_sync_mode"], name="distributed.weight_sync_mode"
        ),
        weight_sync_interval_optimizer_steps=weight_sync_interval,
    )


def _table(
    payload: Mapping[str, object], name: str, fields: set[str]
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"policy E2E smoke [{name}] fields differ")
    return value


def _safe_run_id(value: object) -> str:
    text = _text(value, name="run_id")
    if _SAFE_RUN_ID.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError("run_id is not a safe path-independent identity")
    return text


def _safe_project_name(value: object) -> str:
    text = _text(value, name="training.project_name")
    if _SAFE_PROJECT_NAME.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError("training.project_name is not a safe logger identity")
    return text


def _fqn(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _FQN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a dotted Python symbol")
    return text


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _positive_int(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value: object, *, name: str) -> float:
    result = _real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: object, *, name: str) -> float:
    result = _real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval(value: object, *, name: str, inclusive: bool = False) -> float:
    result = _real(value, name=name)
    valid = 0.0 <= result <= 1.0 if inclusive else 0.0 < result < 1.0
    if not valid:
        raise ValueError(f"{name} lies outside its unit interval")
    return result


def _exact_real(value: object, expected: float, name: str) -> float:
    result = _real(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must equal {expected!r}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _text_tuple(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name=f"{name}[]") for item in _sequence(value, name=name))


def _nonnegative_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(
        _nonnegative_int(item, name=f"{name}[]") for item in _sequence(value, name=name)
    )


def _checkpoint_steps(value: object) -> tuple[int, ...]:
    steps = _nonnegative_int_tuple(value, name="training.checkpoint_steps")
    if (
        not steps
        or steps[0] != 0
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("training.checkpoint_steps must increase strictly from zero")
    return steps


def _logprob_measurement(value: object) -> LogProbMeasurement:
    try:
        return LogProbMeasurement(_text(value, name="sampling.logprob_measurement"))
    except ValueError as error:
        raise ValueError("sampling.logprob_measurement is invalid") from error


def _validate_deepeyes_strict_judge(
    path: Path,
    *,
    judge_config_sha256: str,
    visual_always: bool,
) -> None:
    """Validate the two immutable PRL12 judge identities without enabling them."""

    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("PRL12 judge config is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("PRL12 judge config schema differs")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("PRL12 judge scope differs")
    if scope.get("allows_mcq_judge_calls") is not visual_always:
        raise ValueError("PRL12 judge MCQ scope differs from strict-control arm")
    expected_scope = {
        "allows_policy_rl_reward": True,
        "allows_mcq_judge_calls": visual_always,
        "allows_reference_policy": False,
        "allows_sdpo_teacher": False,
        "allows_gpt_fallback": False,
        "formal_pilot_accepted": not visual_always,
    }
    if dict(scope) != expected_scope:
        raise ValueError("PRL12 judge scope differs from strict-control arm")
    _require_exact(
        payload.get("role"),
        "policy_rl_answer_judge_only",
        "PRL12 judge role",
    )
    expected_sha256 = (
        POLICY_E2E_DEEPEYES_VISUAL_ALWAYS_JUDGE_CONFIG_SHA256
        if visual_always
        else POLICY_E2E_DEEPEYES_RULE_FIRST_JUDGE_CONFIG_SHA256
    )
    _require_exact(
        judge_config_sha256,
        expected_sha256,
        "PRL12 judge config SHA256",
    )


def _existing_file(value: object, *, name: str) -> Path:
    unresolved = Path(value) if isinstance(value, (str, Path)) else None
    if unresolved is not None and unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = _absolute_path(value, name=name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not identify a file")
    return path


def _existing_directory(value: object, *, name: str) -> Path:
    unresolved = Path(_text(value, name=name))
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = _absolute_path(value, name=name)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not identify a directory")
    return path


def _absolute_path(value: object, *, name: str) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, name=name)
    repository_token = "${TGVF_REPOSITORY_ROOT}"
    if raw == repository_token or raw.startswith(repository_token + "/"):
        suffix = raw.removeprefix(repository_token).lstrip("/")
        raw = str(Path(__file__).resolve().parents[3] / suffix)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute normalized path")
    return path.resolve(strict=False)


def _optional_absolute_path(value: object, *, name: str) -> Path | None:
    if value == "":
        return None
    return _absolute_path(value, name=name)


def _require_within(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain under output.root") from error


def _require_exact(actual: object, expected: object, name: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{name} differs from required value {expected!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_json(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_json(item) for item in value]
    return value
