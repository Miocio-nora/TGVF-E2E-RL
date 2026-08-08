"""PRL13 manager for lossless heterogeneous AgentLoop worker batches.

Pinned veRL builds each worker's ``non_tensor_batch`` and
``reward_extra_keys`` schemas from only the rows assigned to that worker.  A
homogeneous visual worker therefore emits Crop/native audit columns that a
homogeneous ThinkLite worker may omit.  ``DataProto.concat`` requires both
column-compatible worker dictionaries and identical non-metric metadata; it
otherwise rejects the metadata or creates a short, row-misaligned column.

This manager keeps upstream scheduling, workers, metrics, and concatenation.
It only restores omitted columns as per-row ``None`` values before the
upstream concat.  In particular, it never fabricates multimodal tensors.
The actual sync-runtime ``global_steps`` column remains fail-closed and must
be present and exact on every trajectory.  Pinned veRL also creates
``min/max_global_steps`` compatibility columns with ``None`` values in this
mode; if a future runtime populates them, they must agree with ``global_steps``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import numpy as np


PRL13_AGENT_LOOP_MANAGER_FQN = (
    "tgvf_rl.framework.verl.native_deepeyes_manager."
    "PRL13HeterogeneousAgentLoopManager"
)
PRL13_REQUIRED_POLICY_COLUMNS = ("global_steps",)
PRL13_OPTIONAL_POLICY_RANGE_COLUMNS = (
    "min_global_steps",
    "max_global_steps",
)


def _restore_reward_extra_columns(
    output: object,
    *,
    worker: int,
    worker_size: int,
    reward_keys: Sequence[str],
) -> None:
    """Make the reward-manager payload authoritative for metric columns.

    Pinned veRL's ``AgentLoopWorker._postprocess`` first materializes the
    scalar, batch-safe ``reward_extra_info`` mapping returned by the reward
    manager.  It then expands every raw agent-loop ``extra_fields`` entry into
    the same ``non_tensor_batch`` and, for duplicate names such as
    ``crop_boxes`` or ``crop_best_call_iou``, silently overwrites the reward
    values.  Raw Crop audit uses variable-length lists and ``None`` for an
    unavailable IoU; those values cannot be concatenated or averaged by the
    upstream trainer.

    The nested reward mapping survives that overwrite.  Restore each declared
    reward column from it before heterogeneous worker schemas are normalized.
    This preserves the reward manager's canonical JSON encoding and NaN
    missing-value semantics without changing any reward score.
    """

    columns = output.non_tensor_batch  # type: ignore[attr-defined]
    nested = columns.get("reward_extra_info")
    if not isinstance(nested, np.ndarray) or nested.ndim == 0 or (
        nested.shape[0] != worker_size
    ):
        raise RuntimeError(
            f"PRL13 worker {worker} lacks row-aligned reward_extra_info"
        )

    mappings: list[Mapping[str, object]] = []
    for row, value in enumerate(nested):
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"PRL13 worker {worker} row {row} reward_extra_info "
                "is not a mapping"
            )
        missing = set(reward_keys).difference(value)
        if missing:
            raise RuntimeError(
                f"PRL13 worker {worker} row {row} reward_extra_info "
                f"omits declared keys: {sorted(missing)}"
            )
        mappings.append(value)

    for name in reward_keys:
        values = [mapping[name] for mapping in mappings]
        if any(value is None for value in values):
            raise RuntimeError(
                f"PRL13 worker {worker} reward column {name!r} contains null; "
                "optional numeric metrics must use NaN"
            )
        columns[name] = np.array(values)


def _python_step(value: object, *, column: str, worker: int, row: int) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is not int or value < 0:
        raise RuntimeError(
            f"PRL13 worker {worker} row {row} has invalid {column}: {value!r}"
        )
    return value


def normalize_prl13_worker_output_columns(
    outputs: Sequence[object],
) -> tuple[object, ...]:
    """Make worker-local optional columns rectangular before DataProto concat.

    Missing optional columns mean that no row in that worker produced the
    corresponding sidecar.  They are restored as object arrays of ``None``,
    exactly representing per-row absence.  Existing arrays are never changed.
    ``global_steps`` is not optional: filling it would hide stale or
    unidentified rollouts, so absence or disagreement fails.  Pinned sync veRL
    leaves its compatibility min/max columns null; populated values are checked.
    """

    normalized = tuple(outputs)
    if not normalized:
        raise ValueError("PRL13 rollout manager received no worker outputs")

    all_keys: set[str] = set()
    reward_extra_union: set[str] = set()
    rewarded_workers: list[bool] = []
    worker_sizes: list[int] = []
    for worker_index, output in enumerate(normalized):
        non_tensor_batch = getattr(output, "non_tensor_batch", None)
        if not isinstance(non_tensor_batch, dict):
            raise TypeError(
                f"PRL13 worker {worker_index} non_tensor_batch must be a dict"
            )
        worker_size = len(output)  # type: ignore[arg-type]
        if type(worker_size) is not int or worker_size <= 0:
            raise ValueError(f"PRL13 worker {worker_index} emitted an empty batch")
        worker_sizes.append(worker_size)
        all_keys.update(non_tensor_batch)
        for key, values in non_tensor_batch.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(
                    f"PRL13 worker {worker_index} column {key!r} is not ndarray"
                )
            if values.ndim == 0 or values.shape[0] != worker_size:
                raise ValueError(
                    f"PRL13 worker {worker_index} column {key!r} has "
                    f"length {values.shape[0] if values.ndim else 0}, "
                    f"expected {worker_size}"
                )

        batch = getattr(output, "batch", None)
        meta_info = getattr(output, "meta_info", None)
        if batch is None or not hasattr(batch, "keys"):
            raise TypeError(f"PRL13 worker {worker_index} lacks a tensor batch")
        if not isinstance(meta_info, dict):
            raise TypeError(f"PRL13 worker {worker_index} meta_info must be a dict")
        rewarded = "rm_scores" in batch.keys()
        rewarded_workers.append(rewarded)
        raw_reward_keys = meta_info.get("reward_extra_keys")
        if not rewarded:
            if raw_reward_keys is not None:
                raise RuntimeError(
                    f"PRL13 worker {worker_index} declares reward_extra_keys "
                    "without rm_scores"
                )
            continue
        if not isinstance(raw_reward_keys, list) or not raw_reward_keys:
            raise RuntimeError(
                f"PRL13 worker {worker_index} with rm_scores omitted "
                "reward_extra_keys"
            )
        if any(
            not isinstance(name, str) or not name
            for name in raw_reward_keys
        ) or len(set(raw_reward_keys)) != len(raw_reward_keys):
            raise RuntimeError(
                f"PRL13 worker {worker_index} has invalid reward_extra_keys"
            )
        missing_reward_columns = set(raw_reward_keys).difference(non_tensor_batch)
        if missing_reward_columns:
            raise RuntimeError(
                f"PRL13 worker {worker_index} reward metadata names missing "
                f"columns: {sorted(missing_reward_columns)}"
            )
        _restore_reward_extra_columns(
            output,
            worker=worker_index,
            worker_size=worker_size,
            reward_keys=raw_reward_keys,
        )
        reward_extra_union.update(raw_reward_keys)

    if any(rewarded_workers) and not all(rewarded_workers):
        raise RuntimeError("PRL13 workers disagree on presence of rm_scores")

    for worker_index, (output, worker_size) in enumerate(
        zip(normalized, worker_sizes, strict=True)
    ):
        columns = output.non_tensor_batch  # type: ignore[attr-defined]
        for name in PRL13_REQUIRED_POLICY_COLUMNS:
            if name not in columns:
                raise RuntimeError(
                    f"PRL13 worker {worker_index} omitted required policy "
                    f"column {name!r}"
                )
        for row in range(worker_size):
            step = _python_step(
                columns["global_steps"][row],
                column="global_steps",
                worker=worker_index,
                row=row,
            )
            range_values = tuple(
                columns[name][row]
                for name in PRL13_OPTIONAL_POLICY_RANGE_COLUMNS
                if name in columns
            )
            if range_values and all(value is None for value in range_values):
                continue
            if len(range_values) != 2 or any(
                value is None for value in range_values
            ):
                raise RuntimeError(
                    f"PRL13 worker {worker_index} row {row} policy-version "
                    "columns disagree"
                )
            minimum = _python_step(
                range_values[0],
                column="min_global_steps",
                worker=worker_index,
                row=row,
            )
            maximum = _python_step(
                range_values[1],
                column="max_global_steps",
                worker=worker_index,
                row=row,
            )
            if minimum != step or maximum != step:
                raise RuntimeError(
                    f"PRL13 worker {worker_index} row {row} policy-version "
                    "columns disagree"
                )

    # A training outer step must be generated by exactly one behavior-policy
    # version across all workers.
    observed_steps = {
        _python_step(
            output.non_tensor_batch["global_steps"][row],  # type: ignore[attr-defined]
            column="global_steps",
            worker=worker_index,
            row=row,
        )
        for worker_index, (output, worker_size) in enumerate(
            zip(normalized, worker_sizes, strict=True)
        )
        for row in range(worker_size)
    }
    if len(observed_steps) != 1:
        raise RuntimeError(
            "PRL13 worker outputs contain multiple behavior-policy versions: "
            f"{sorted(observed_steps)}"
        )

    canonical_reward_keys = sorted(reward_extra_union)
    for output, worker_size in zip(normalized, worker_sizes, strict=True):
        columns = output.non_tensor_batch  # type: ignore[attr-defined]
        for key in all_keys.difference(columns):
            missing = np.empty(worker_size, dtype=object)
            missing.fill(None)
            columns[key] = missing
        if canonical_reward_keys:
            # Source-specific reward fields are legitimate per-row absences,
            # already represented by the None columns above.  The metadata is
            # a schema declaration, so every worker must advertise the same
            # deterministic union before upstream DataProto.concat.
            output.meta_info["reward_extra_keys"] = canonical_reward_keys  # type: ignore[attr-defined]
        checker = getattr(output, "check_consistency", None)
        if not callable(checker):
            raise TypeError("PRL13 worker output lacks check_consistency()")
        checker()
    return normalized


try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    from verl.protocol import DataProto
    from verl.utils.ray_utils import auto_await
    from verl.utils.skip import SkipManager
except ModuleNotFoundError as error:  # pragma: no cover - CPU-only install
    _VERL_IMPORT_ERROR = error

    class PRL13HeterogeneousAgentLoopManager:  # type: ignore[no-redef]
        """Unavailable placeholder when the pinned veRL runtime is absent."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "PRL13HeterogeneousAgentLoopManager requires pinned veRL"
            ) from _VERL_IMPORT_ERROR

else:

    class PRL13HeterogeneousAgentLoopManager(AgentLoopManager):
        """Upstream manager with one pre-concat heterogeneous-schema repair."""

        @auto_await
        @SkipManager.annotate(role="rollout")
        async def generate_sequences(self, prompts: DataProto) -> DataProto:
            if "priority" not in prompts.non_tensor_batch:
                prompts.non_tensor_batch["priority"] = np.arange(
                    len(prompts), dtype=np.int64
                )

            chunks = prompts.chunk(len(self.agent_loop_workers))
            outputs = await asyncio.gather(
                *[
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(
                        self.agent_loop_workers, chunks, strict=True
                    )
                ]
            )
            normalize_prl13_worker_output_columns(outputs)
            output = DataProto.concat(outputs)

            metrics = [
                worker_output.meta_info.pop("metrics")
                for worker_output in outputs
            ]
            timing = self._performance_metrics(metrics, output)
            output.meta_info = {"timing": timing, **outputs[0].meta_info}
            return output

    PRL13HeterogeneousAgentLoopManager.__module__ = __name__


__all__ = [
    "PRL13_AGENT_LOOP_MANAGER_FQN",
    "PRL13_OPTIONAL_POLICY_RANGE_COLUMNS",
    "PRL13_REQUIRED_POLICY_COLUMNS",
    "PRL13HeterogeneousAgentLoopManager",
    "normalize_prl13_worker_output_columns",
]
