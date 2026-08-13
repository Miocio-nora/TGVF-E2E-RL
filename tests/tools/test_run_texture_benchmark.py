from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import ClassVar

from PIL import Image
import pytest

from tgvf_rl.evaluation.texture_bench.schema import (
    PipelineArm,
    PipelineBackend,
    PipelineKind,
    TextureBenchmarkMatrix,
    canonical_json_sha256,
)
from tgvf_rl.evaluation.texture_bench.stock_qwen import (
    STOCK_QWEN_RESULT_SCHEMA,
    STOCK_QWEN_VISION_IDENTITY_SCHEMA,
    stable_stock_qwen_seed,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/run_texture_benchmark.py"


def _tool() -> object:
    spec = importlib.util.spec_from_file_location("run_texture_benchmark", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix(
    tmp_path: Path,
    *,
    paired_kind: PipelineKind | None = None,
    task_count: int = 1,
) -> Path:
    tasks = (tmp_path / "tasks.jsonl").resolve()
    task_rows = []
    for ordinal in range(task_count):
        name = "image.png" if ordinal == 0 else f"image-{ordinal}.png"
        image = (tmp_path / name).resolve()
        Image.new("RGB", (8, 6), (ordinal, 2, 3)).save(image)
        sample_id = f"mmad-test-{ordinal}"
        task_rows.append(
            {
                "ordinal": ordinal,
                "dataset": "MMAD",
                "row_number": ordinal,
                "index": sample_id,
                "sample_id": sample_id,
                "question": "Question?",
                "image_paths": [str(image)],
                "answer": "A",
                "options": [["A", "yes"], ["B", "no"]],
                "metadata": [
                    ["score_dataset", "VisA"],
                    ["question_type_score", "Object Analysis"],
                ],
                "image_sha256s": [hashlib.sha256(image.read_bytes()).hexdigest()],
                "image_dimensions": [[8, 6]],
            }
        )
    tasks.write_text(
        "".join(
            json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n"
            for task in task_rows
        ),
        encoding="utf-8",
    )
    model = (tmp_path / "model").resolve()
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    pointer = (tmp_path / "pointer.json").resolve()
    pointer.write_text("{}\n", encoding="utf-8")
    tool_arms = []
    for kind in (
        PipelineKind.CROP,
        PipelineKind.TGVF,
        PipelineKind.TGVF_CROP,
    ):
        policy = (tmp_path / f"policy-{kind.value}.toml").resolve()
        policy.write_text(
            f"[protocol]\ntool_profile='{kind.policy_tool_profile}'\n",
            encoding="utf-8",
        )
        if kind is paired_kind:
            tool_arms.append(
                PipelineArm(
                    kind.value,
                    kind,
                    PipelineBackend.POLICY_BENCHMARK,
                    policy_config_path=policy,
                    paired_qwen_model_path=model,
                    paired_rp66_pointer_path=pointer,
                    paired_snapshot_receipt_path=(
                        tmp_path / f"{kind.value}-paired-receipt.json"
                    ).resolve(),
                    expected_optimizer_step=8,
                    evaluation_protocol="training_run",
                )
            )
        else:
            tool_arms.append(
                PipelineArm(
                    kind.value,
                    kind,
                    PipelineBackend.POLICY_BENCHMARK,
                    policy_config_path=policy,
                    lora_pointer_path=pointer,
                    expected_optimizer_step=8,
                    evaluation_protocol="training_run",
                )
            )
    arms = (
        PipelineArm(
            "original",
            PipelineKind.ORIGINAL,
            PipelineBackend.STOCK_QWEN_VLLM,
            model_path=model,
        ),
        *tool_arms,
    )
    matrix = TextureBenchmarkMatrix(
        matrix_id="texture-test",
        task_manifest_path=tasks,
        task_manifest_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        task_count=task_count,
        output_root=(tmp_path / "output").resolve(),
        arms=arms,
    )
    path = (tmp_path / "matrix.json").resolve()
    path.write_text(json.dumps(matrix.identity_payload()), encoding="utf-8")
    return path


class _DurableFakeRunner:
    calls: ClassVar[list[tuple[int, ...]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.vision = kwargs["vision"]
        assert kwargs["engine_kwargs"] == {"gpu_memory_utilization": 0.5}

    def run(self, tasks: object) -> list[dict[str, object]]:
        batch = tuple(tasks)  # type: ignore[arg-type]
        self.calls.append(tuple(task.ordinal for task in batch))
        rows = []
        for task in batch:
            vision_content = {
                "schema_version": STOCK_QWEN_VISION_IDENTITY_SCHEMA,
                "source_path": task.image_paths[0],
                "source_image_sha256": task.image_sha256s[0],
                "source_dimensions": list(task.image_dimensions[0]),
                "preprocess": asdict(self.vision),
                "preprocess_identity_sha256": self.vision.identity_sha256,
            }
            rows.append(
                {
                    "schema_version": STOCK_QWEN_RESULT_SCHEMA,
                    "ordinal": task.ordinal,
                    "sample_id": task.bound_sample_id,
                    "index": task.index,
                    "dataset": task.dataset,
                    "final_answer": "A",
                    "model_response": {"text": "A"},
                    "vision_identity": {
                        **vision_content,
                        "identity_sha256": canonical_json_sha256(vision_content),
                    },
                    "request_seed": stable_stock_qwen_seed(task),
                }
            )
        return rows


def test_validate_and_policy_commands_carry_shared_pixel_cap(tmp_path: Path) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)

    validated = tool.validate_matrix(matrix_path)
    commands = tool.policy_commands(matrix_path, arm_selector="tgvf_crop")

    assert validated["vision"]["max_pixels"] == 262_144
    assert validated["complete_four_arm_matrix"] is True
    assert validated["missing_pipeline_kinds"] == []
    materialize = commands["materialize"]
    cap_index = materialize.index("--image-max-pixels")
    assert materialize[cap_index + 1] == "262144"
    assert len(commands["workers_run_concurrently"]) == 4


def test_staged_three_arm_matrix_runs_before_crop_tgvf_snapshot_exists(
    tmp_path: Path,
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["arms"] = [arm for arm in payload["arms"] if arm["kind"] != "tgvf_crop"]
    payload["gpu_ids"] = [4, 5, 6, 7]
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    validated = tool.validate_matrix(matrix_path)
    assert validated["complete_four_arm_matrix"] is False
    assert validated["missing_pipeline_kinds"] == ["tgvf_crop"]
    commands = tool.policy_commands(matrix_path, arm_selector="tgvf")
    assert commands["arm"]["kind"] == "tgvf"
    materialize = commands["materialize"]
    first_gpu = materialize.index("--gpu-ids") + 1
    assert materialize[first_gpu : first_gpu + 4] == ["4", "5", "6", "7"]
    assert [
        worker["environment"]["CUDA_VISIBLE_DEVICES"]
        for worker in commands["workers_run_concurrently"]
    ] == ["4", "5", "6", "7"]
    with pytest.raises(ValueError, match="complete texture comparison"):
        tool.validate_matrix(matrix_path, require_complete_arms=True)


def test_original_runner_writes_model_and_result_identities(tmp_path: Path) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["batch_size"] == 2
            assert kwargs["vision"].max_pixels == 262_144

        def run(self, tasks: object) -> list[dict[str, object]]:
            task = tuple(tasks)[0]
            return [
                {
                    "ordinal": task.ordinal,
                    "sample_id": task.sample_id,
                    "final_answer": "A",
                }
            ]

    identity = tool.run_original(
        matrix_path,
        batch_size=2,
        max_tokens=17,
        runner_type=FakeRunner,
    )

    assert identity["model_tree"]["file_count"] == 1
    assert len(identity["model_tree"]["tree_sha256"]) == 64
    assert identity["results"]["line_count"] == 1
    assert len(identity["identity_sha256"]) == 64
    result = json.loads(
        (tmp_path / "output/original/results.jsonl").read_text(encoding="utf-8")
    )
    assert (
        result["task_manifest_sha256"]
        == hashlib.sha256(
            Path(json.loads(matrix_path.read_text())["task_manifest_path"]).read_bytes()
        ).hexdigest()
    )
    assert result["matrix_identity_sha256"] == identity["matrix_identity_sha256"]


def test_policy_commands_support_current_paired_qwen_rp66_snapshot(
    tmp_path: Path,
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path, paired_kind=PipelineKind.TGVF)

    commands = tool.policy_commands(matrix_path, arm_selector="tgvf")

    materialize = commands["materialize"]
    assert materialize[1].endswith(
        "tools/materialize_paired_tgvf_policy_benchmark_config.py"
    )
    assert materialize[materialize.index("--optimizer-step") + 1] == "8"
    assert materialize[materialize.index("--image-max-pixels") + 1] == "262144"
    assert materialize[materialize.index("--paired-seed-namespace") + 1] == (
        hashlib.sha256(
            Path(
                json.loads(matrix_path.read_text(encoding="utf-8"))[
                    "task_manifest_path"
                ]
            ).read_bytes()
        ).hexdigest()
    )


def test_original_workers_resume_by_ordinal_and_finalize_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path, task_count=5)
    options = {"gpu_memory_utilization": 0.5}
    _DurableFakeRunner.calls.clear()
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    first = tool.run_original_worker(
        matrix_path,
        rank=0,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        max_tasks=1,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )
    second = tool.run_original_worker(
        matrix_path,
        rank=0,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        max_tasks=1,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )

    assert _DurableFakeRunner.calls == [(0,), (2,)]
    assert first["completed_this_run"] == second["completed_this_run"] == 1
    assert second["completed"] == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        tool.finalize_original(
            matrix_path,
            world_size=2,
            gpu_ids=(4, 5),
            batch_size=2,
            max_tokens=17,
            engine_kwargs=options,
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    rank_one = tool.run_original_worker(
        matrix_path,
        rank=1,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    rank_zero = tool.run_original_worker(
        matrix_path,
        rank=0,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )

    assert rank_one["completed_this_run"] == 2
    assert rank_zero["completed_this_run"] == 1
    assert _DurableFakeRunner.calls == [(0,), (2,), (1, 3), (4,)]
    status = tool.original_status(
        matrix_path,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        engine_kwargs=options,
    )
    assert status["per_rank_completed"] == [3, 2]
    assert status["completed"] == status["total"] == 5
    assert status["remaining"] == 0
    assert status["complete"] is True

    identity = tool.finalize_original(
        matrix_path,
        world_size=2,
        gpu_ids=(4, 5),
        batch_size=2,
        max_tokens=17,
        engine_kwargs=options,
    )
    assert identity["results"]["line_count"] == 5
    assert identity["durable_execution"]["gpu_ids"] == [4, 5]
    merged = [
        json.loads(line)
        for line in (tmp_path / "output/original/results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["ordinal"] for row in merged] == list(range(5))
    assert [row["rank"] for row in merged] == [0, 1, 0, 1, 0]


def test_original_resume_rejects_tampered_identity_and_generation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)
    options = {"gpu_memory_utilization": 0.5}
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    tool.run_original_worker(
        matrix_path,
        rank=0,
        world_size=1,
        gpu_ids=(7,),
        batch_size=1,
        max_tokens=17,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )

    with pytest.raises(RuntimeError, match="execution identity differs"):
        tool.original_status(
            matrix_path,
            world_size=1,
            gpu_ids=(7,),
            batch_size=2,
            max_tokens=17,
            engine_kwargs=options,
        )

    result_path = tmp_path / "output/original/inference/rank-0.jsonl"
    row = json.loads(result_path.read_text(encoding="utf-8"))
    row["matrix_identity_sha256"] = "0" * 64
    digest_payload = dict(row)
    digest_payload.pop("result_identity_sha256")
    row["result_identity_sha256"] = canonical_json_sha256(digest_payload)
    result_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="matrix_identity_sha256 differs"):
        tool.original_status(
            matrix_path,
            world_size=1,
            gpu_ids=(7,),
            batch_size=1,
            max_tokens=17,
            engine_kwargs=options,
        )


def test_original_resume_rejects_unexpected_rank_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)
    options = {"gpu_memory_utilization": 0.5}
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    tool.run_original_worker(
        matrix_path,
        rank=0,
        world_size=1,
        gpu_ids=(7,),
        batch_size=1,
        max_tokens=17,
        engine_kwargs=options,
        runner_type=_DurableFakeRunner,
    )
    unexpected = tmp_path / "output/original/inference/rank-9.jsonl"
    unexpected.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected original rank"):
        tool.original_status(
            matrix_path,
            world_size=1,
            gpu_ids=(7,),
            batch_size=1,
            max_tokens=17,
            engine_kwargs=options,
        )


def test_original_worker_requires_exact_physical_gpu_and_rank_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool()
    matrix_path = _matrix(tmp_path)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=4"):
        tool.run_original_worker(
            matrix_path,
            rank=0,
            world_size=2,
            gpu_ids=(4, 5),
            engine_kwargs={"gpu_memory_utilization": 0.5},
            runner_type=_DurableFakeRunner,
        )

    output_root = tmp_path / "output/original"
    with tool._original_rank_lock(output_root, 0):
        with pytest.raises(RuntimeError, match="already active"):
            with tool._original_rank_lock(output_root, 0):
                pass


def test_cli_routes_original_rank_and_status_with_gpu_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _tool()
    observed: list[tuple[str, dict[str, object]]] = []

    def fake_worker(matrix: Path, **kwargs: object) -> dict[str, object]:
        observed.append((str(matrix), dict(kwargs)))
        return {"mode": "worker"}

    def fake_status(matrix: Path, **kwargs: object) -> dict[str, object]:
        observed.append((str(matrix), dict(kwargs)))
        return {"mode": "status"}

    monkeypatch.setattr(tool, "run_original_worker", fake_worker)
    monkeypatch.setattr(tool, "original_status", fake_status)
    matrix_path = tmp_path / "matrix.json"
    assert (
        tool.main(
            [
                "original",
                "--matrix",
                str(matrix_path),
                "--rank",
                "2",
                "--world-size",
                "4",
                "--gpu-ids",
                "4",
                "5",
                "6",
                "7",
                "--max-tasks",
                "9",
                "--no-verify-images",
            ]
        )
        == 0
    )
    assert observed[-1][1]["rank"] == 2
    assert observed[-1][1]["world_size"] == 4
    assert observed[-1][1]["gpu_ids"] == [4, 5, 6, 7]
    assert observed[-1][1]["max_tasks"] == 9
    assert observed[-1][1]["verify_images"] is False
    assert json.loads(capsys.readouterr().out)["mode"] == "worker"

    assert (
        tool.main(
            [
                "original-status",
                "--matrix",
                str(matrix_path),
                "--world-size",
                "4",
                "--gpu-ids",
                "4",
                "5",
                "6",
                "7",
            ]
        )
        == 0
    )
    assert observed[-1][1]["world_size"] == 4
    assert observed[-1][1]["gpu_ids"] == [4, 5, 6, 7]
    assert json.loads(capsys.readouterr().out)["mode"] == "status"
