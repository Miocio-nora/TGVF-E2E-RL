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
    / "run_rp68_post_training_evaluations.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "rp68_post_training_evaluations", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
evaluation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluation
_SPEC.loader.exec_module(evaluation)


def _receipt(tmp_path: Path) -> dict[str, object]:
    return {
        "run_identity_sha256": "a" * 64,
        "adapter_manifest_sha256": "b" * 64,
        "adapter_path": str(tmp_path / "adapter.pt"),
        "adapter_file_sha256": "c" * 64,
        "training_config_path": str(tmp_path / "training.toml"),
        "training_config_sha256": "d" * 64,
        "code_commit": "e" * 40,
        "evaluation_data_path": str(tmp_path / "test.jsonl"),
        "evaluation_data_source_sha256": "f" * 64,
        "conditioning_provider": "contextual_hidden_state",
        "prompt_identity": "qwen3-representation-image-question-v1",
        "prompt_sha256": "1" * 64,
    }


def _write_semantic(root: Path, *, samples: int) -> None:
    root.mkdir(parents=True)
    summary = {
        "schema_version": evaluation.SEMANTIC_SCHEMA,
        "status": "complete",
        "run_identity_sha256": "2" * 64,
        "overall": {"total": samples * len(evaluation.MAIN_ARMS)},
        "by_arm": {arm: {"total": samples} for arm in evaluation.MAIN_ARMS},
    }
    summary_payload = (json.dumps(summary, sort_keys=True) + "\n").encode()
    (root / "summary.json").write_bytes(summary_payload)
    manifest = {
        "schema_version": evaluation.SEMANTIC_SCHEMA,
        "status": "complete",
        "run_identity_sha256": "2" * 64,
        "files": {
            "summary": {
                "path": "summary.json",
                "sha256": sha256(summary_payload).hexdigest(),
            },
            "overlay_records": {
                "path": "records.jsonl",
                "rows": samples * len(evaluation.MAIN_ARMS),
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")


def test_rendered_configs_bind_dynamic_completed_artifact(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = evaluation._render_evaluation_config(
        receipt=receipt,
        evaluation_code_commit="4" * 40,
        run_id="RP68-EVAL",
        evaluation_id="rp68-eval-v1",
        manifest_path=tmp_path / "groups.json",
        manifest_sha256="3" * 64,
        report_path=tmp_path / "report.json",
        physical_gpu_id=4,
    ).decode()

    assert f'path = "{tmp_path / "adapter.pt"}"' in payload
    assert f'manifest_sha256 = "{"b" * 64}"' in payload
    assert f'expected_run_identity_sha256 = "{"a" * 64}"' in payload
    assert "expected_global_step = 2000" in payload
    assert "physical_gpu_id = 4" in payload
    assert f'commit = "{"4" * 40}"' in payload
    assert f'commit = "{"e" * 40}"' not in payload


def test_live_evaluation_commit_is_stable_across_non_code_commits(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git_output(*arguments: str) -> str:
        calls.append(arguments)
        if arguments[0] in {"diff", "ls-files"}:
            return ""
        assert arguments[:4] == ("log", "-1", "--format=%H", "HEAD")
        return "a" * 40

    monkeypatch.setattr(evaluation, "_git_output", fake_git_output)

    assert evaluation._live_evaluation_code_commit() == "a" * 40
    assert [call[0] for call in calls] == ["diff", "ls-files", "log"]


def test_live_evaluation_commit_rejects_dirty_code_paths(monkeypatch) -> None:
    def fake_git_output(*arguments: str) -> str:
        return "src/tgvf_rl/changed.py" if arguments[0] == "diff" else ""

    monkeypatch.setattr(evaluation, "_git_output", fake_git_output)

    with pytest.raises(
        evaluation.EvaluationBlockedError, match="requires clean live code paths"
    ):
        evaluation._live_evaluation_code_commit()


def test_int_report_and_marker_are_bound_and_tamper_evident(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    report = tmp_path / "int.json"
    config = tmp_path / "int.toml"
    marker = tmp_path / "marker.json"
    config.write_text("config\n")
    report.write_text(
        json.dumps(
            {
                "identity": {
                    "schema_version": "representation_internal_evaluation_v1",
                    "evaluation_id": "rp68-step2000-int-diag-v1",
                    "checkpoint_identity": receipt["adapter_manifest_sha256"],
                    "target_conditioning_provider": receipt["conditioning_provider"],
                    "random_seed": 42,
                    "prompt_identity": (
                        f"{receipt['prompt_identity']}:{receipt['prompt_sha256']}"
                    ),
                }
            }
        )
        + "\n"
    )
    marker.write_text(
        json.dumps(
            {
                "schema_version": evaluation.INT_MARKER_SCHEMA,
                "status": "complete",
                "run_id": evaluation.EXPECTED_RUN_ID,
                "run_identity_sha256": receipt["run_identity_sha256"],
                "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
                "evaluation_config_sha256": evaluation._file_sha256(config),
                "report": evaluation._artifact_record(report),
            }
        )
        + "\n"
    )

    assert evaluation._int_marker_is_current(
        marker, receipt=receipt, report_path=report, config_path=config
    )
    report.write_text("{}\n")
    with pytest.raises(evaluation.EvaluationBlockedError, match="marker drifted"):
        evaluation._int_marker_is_current(
            marker, receipt=receipt, report_path=report, config_path=config
        )


def test_semantic_publication_requires_both_exact_arms_and_sample_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "semantic"
    _write_semantic(root, samples=867)
    assert evaluation._semantic_complete(root, samples=867)

    summary = json.loads((root / "summary.json").read_text())
    summary["by_arm"].pop("image_correct_D")
    (root / "summary.json").write_text(json.dumps(summary) + "\n")
    with pytest.raises(evaluation.EvaluationBlockedError, match="publication differs"):
        evaluation._semantic_complete(root, samples=867)


@pytest.mark.parametrize("value", ["0", "0,0", "-1,2", "one,two", "0,1,2"])
def test_gpu_parser_rejects_non_tp2_bindings(value: str) -> None:
    with pytest.raises(Exception):
        evaluation._parse_gpu_ids(value)
