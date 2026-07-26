from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tgvf_rl.data.policy_selection_t1_judge import _strict_response


def test_strict_local_response_requires_model_stop_binary_and_usage() -> None:
    response = {
        "model": "Qwen2.5-72B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": '{"verdict":1,"rationale":"same"}'},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    verdict, rationale, usage = _strict_response(
        response, expected_model="Qwen2.5-72B-Instruct"
    )
    assert verdict == 1
    assert rationale == "same"
    assert usage["cost_usd"] == 0.0

    for changed, message in (
        ({**response, "model": "wrong"}, "model differs"),
        (
            {
                **response,
                "choices": [{**response["choices"][0], "finish_reason": "length"}],
            },
            "finish with stop",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            _strict_response(changed, expected_model="Qwen2.5-72B-Instruct")


def test_judge_cli_help_does_not_import_vllm_or_transformers() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "judge_policy_data_selection_t1.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finalize" in completed.stdout
    assert "run" in completed.stdout
