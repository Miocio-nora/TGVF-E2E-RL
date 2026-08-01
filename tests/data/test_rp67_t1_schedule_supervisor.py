from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "supervise_rp67_t1_schedule.py"
)
_SPEC = importlib.util.spec_from_file_location("rp67_t1_schedule_supervisor", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
supervisor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = supervisor
_SPEC.loader.exec_module(supervisor)


def _worker(rank: int, gpu: int, count: int, index: int) -> object:
    broad = count == 1
    tag = f"rank-{rank}" if broad else f"rank-{rank}-subshard-{index}-of-{count}"
    return supervisor.Worker(
        tag=tag,
        pgid_path=f"/runtime/{tag}.pgid",
        pgid=10_000 + rank * 100 + gpu,
        rank=rank,
        gpu=gpu,
        subshard_count=count,
        subshard_index=index,
        broad=broad,
        uid=1000,
        boot_id="00000000-0000-0000-0000-000000000000",
        starttime_ticks=123,
        argv_sha256="a" * 64,
    )


def _snapshot(
    *,
    complete: tuple[bool, bool, bool, bool] = (True, False, False, False),
    workers: tuple[object, ...] = (),
    cutoff: bool = False,
    acc: bool = False,
    busy: tuple[int, ...] = (),
) -> object:
    gpu_pids = {index: ((2000 + index,) if index in busy else ()) for index in range(8)}
    return supervisor.Snapshot(
        observed_at="2026-08-01T00:00:00+00:00",
        manifest_counts=(17046, 1, 1, 1),
        ranks_complete=complete,
        workers=workers,
        gpu_compute_pids=gpu_pids,
        acc_complete=acc,
        cutoff_reached=cutoff,
    )


def test_rank3_completion_reconciles_broad_to_four_way_in_two_passes() -> None:
    broad = _worker(2, 4, 1, 0)
    first = supervisor.build_plan(
        _snapshot(complete=(True, False, False, True), workers=(broad,)),
        post_cutoff_count=None,
    )
    assert first.stage == "four-way-gpu4567"
    assert first.stop_tags == ("rank-2",)
    assert first.launch_workers == ()

    second = supervisor.build_plan(
        _snapshot(complete=(True, False, False, True)), post_cutoff_count=None
    )
    assert [(item.gpu, item.subshard_count, item.subshard_index) for item in second.launch_workers] == [
        (4, 4, 0),
        (5, 4, 1),
        (6, 4, 2),
        (7, 4, 3),
    ]


def test_rank1_completion_reconciles_four_way_to_six_way() -> None:
    four = tuple(_worker(2, gpu, 4, index) for index, gpu in enumerate(range(4, 8)))
    first = supervisor.build_plan(
        _snapshot(complete=(True, True, False, True), workers=four),
        post_cutoff_count=None,
    )
    assert first.stop_tags == tuple(item.tag for item in four)
    assert first.launch_workers == ()

    second = supervisor.build_plan(
        _snapshot(complete=(True, True, False, True)), post_cutoff_count=None
    )
    assert [(item.gpu, item.subshard_count) for item in second.launch_workers] == [
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (6, 6),
        (7, 6),
    ]


def test_cutoff_keeps_only_existing_six_way_workers_on_gpu23() -> None:
    six = tuple(_worker(2, gpu, 6, index) for index, gpu in enumerate(range(2, 8)))
    plan = supervisor.build_plan(
        _snapshot(
            complete=(True, True, False, True),
            workers=six,
            cutoff=True,
        ),
        post_cutoff_count=6,
    )
    assert plan.stage == "post-cutoff-gpu23"
    assert plan.stop_tags == tuple(item.tag for item in six[2:])
    assert plan.launch_workers == ()

    stable = supervisor.build_plan(
        _snapshot(
            complete=(True, True, False, True),
            workers=six[:2],
            cutoff=True,
        ),
        post_cutoff_count=6,
    )
    assert stable.stop_tags == ()
    assert stable.launch_workers == ()


def test_cutoff_before_six_way_uses_complete_two_way_coverage() -> None:
    plan = supervisor.build_plan(
        _snapshot(complete=(True, True, False, True), cutoff=True),
        post_cutoff_count=2,
    )
    assert [(item.gpu, item.subshard_count, item.subshard_index) for item in plan.launch_workers] == [
        (2, 2, 0),
        (3, 2, 1),
    ]


def test_acc_marker_converges_directly_to_final_gpu0123() -> None:
    residual = (
        _worker(2, 2, 6, 0),
        _worker(2, 3, 6, 1),
    )
    first = supervisor.build_plan(
        _snapshot(
            complete=(True, True, False, True),
            workers=residual,
            cutoff=True,
            acc=True,
        ),
        post_cutoff_count=6,
    )
    assert first.stage == "final-gpu0123"
    assert first.stop_tags == tuple(item.tag for item in residual)

    second = supervisor.build_plan(
        _snapshot(
            complete=(True, True, False, True), cutoff=True, acc=True
        ),
        post_cutoff_count=6,
    )
    assert [item.gpu for item in second.launch_workers] == [0, 1, 2, 3]


def test_busy_target_gpu_waits_without_partial_launch() -> None:
    plan = supervisor.build_plan(
        _snapshot(
            complete=(True, True, False, True),
            cutoff=True,
            acc=True,
            busy=(1,),
        ),
        post_cutoff_count=6,
    )
    assert plan.launch_workers == ()
    assert "GPU1 has compute PIDs" in plan.wait_reasons[0]


def test_rank2_complete_is_never_relaunched() -> None:
    plan = supervisor.build_plan(
        _snapshot(complete=(True, True, True, True), cutoff=True, acc=True),
        post_cutoff_count=6,
    )
    assert plan.launch_workers == ()
    assert plan.stop_tags == ()
    assert plan.complete is True


def test_manifest_coverage_rejects_out_of_range_revision0_index(tmp_path: Path) -> None:
    for rank in range(4):
        for index in range(2):
            (tmp_path / f"rank-{rank:02d}-chunk-{index:06d}.json").write_text("{}")
    counts, complete = supervisor._manifest_coverage(tmp_path, (2, 2, 2, 2))
    assert counts == (2, 2, 2, 2)
    assert complete == (True, True, True, True)

    (tmp_path / "rank-02-chunk-000002.json").write_text("{}")
    with pytest.raises(supervisor.SupervisorBlockedError, match="unexpected revision-0"):
        supervisor._manifest_coverage(tmp_path, (2, 2, 2, 2))


def test_cutoff_requires_timezone() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        supervisor._aware_datetime("2026-08-01T07:40:00")


def test_supervisor_accepts_only_hash_bound_v2_semantic_publications(
    tmp_path: Path,
) -> None:
    def file_record(path: Path) -> dict[str, str]:
        return {
            "status": "complete",
            "path": str(path.resolve()),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }

    artifacts: dict[str, object] = {}
    int_diag = tmp_path / "int-diag.json"
    int_diag.write_text("{}\n")
    artifacts["int_diag"] = file_record(int_diag)
    for name in ("acc_first200", "acc_full867", "diag_first200_sixarm"):
        root = tmp_path / name
        root.mkdir()
        run_identity = "a" * 64
        summary = {
            "schema_version": supervisor.SEMANTIC_SCHEMA,
            "status": "complete",
            "run_identity_sha256": run_identity,
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary) + "\n")
        manifest = {
            "schema_version": supervisor.SEMANTIC_SCHEMA,
            "status": "complete",
            "run_identity_sha256": run_identity,
            "files": {
                "summary": {
                    "path": "summary.json",
                    "sha256": sha256(summary_path.read_bytes()).hexdigest(),
                }
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        artifacts[name] = {
            "status": "complete",
            "root": str(root.resolve()),
            "run_identity_sha256": run_identity,
            "summary": file_record(summary_path),
            "manifest": file_record(manifest_path),
        }
    marker = tmp_path / "marker-v2.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": supervisor.MARKER_SCHEMA,
                "status": "complete",
                "rp67_run_id": supervisor.EXPECTED_RP67_RUN_ID,
                "artifacts": artifacts,
            }
        )
        + "\n"
    )

    assert supervisor._validated_acc_marker(marker)
    full_summary = tmp_path / "acc_full867" / "summary.json"
    full_summary.write_text('{"drifted":true}\n')
    with pytest.raises(supervisor.SupervisorBlockedError, match="binding drifted"):
        supervisor._validated_acc_marker(marker)


def test_v2_schedule_uses_isolated_marker_and_runtime_paths() -> None:
    assert supervisor.DEFAULT_ACC_MARKER.name == (
        "rp67_step2000_all_validations_complete_v2.json"
    )
    assert supervisor.SUPERVISOR_RUNTIME_NAME == "supervisor-rp67-t1-v2-20260801"
