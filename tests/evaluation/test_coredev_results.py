import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from tgvf_rl.evaluation.coredev_materialize import (
    COREDEV_JUDGE_CONTRACTS,
    COREDEV_LLM_JUDGE_MODEL,
)
from tgvf_rl.evaluation.coredev_results import (
    COREDEV_BASELINE_MODEL,
    FailClosedJudge,
    check_qwen25_72b_judge,
    install_fail_closed_judge_builders,
    summarize_coredev_results,
)
from tgvf_rl.evaluation.vlmevalkit import COREDEV_2511


JUDGE_BASE_URL = "http://127.0.0.1:8012/v1"


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _StaticJudge:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, *_args: object, **_kwargs: object) -> str:
        return self.response


def _mathverse_score_prompt() -> str:
    return """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.

[Question]: What is 1 + 1?
[Standard Answer]: 2
[Model_answer] : 2
Judgement:"""


def test_judge_health_requires_exact_model_and_deterministic_completion() -> None:
    urls: list[str] = []

    def opener(req: object, *, timeout: float) -> _Response:
        assert timeout == 3
        url = req.full_url
        urls.append(urlparse(url).path)
        if url.endswith("/models"):
            return _Response({"data": [{"id": COREDEV_LLM_JUDGE_MODEL}]})
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": "TGVF_JUDGE_READY"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )

    result = check_qwen25_72b_judge(
        base_url=JUDGE_BASE_URL,
        timeout=3,
        opener=opener,
    )

    assert result["status"] == "pass"
    assert result["model"] == COREDEV_LLM_JUDGE_MODEL
    assert urls == ["/v1/models", "/v1/chat/completions"]


def test_judge_health_and_calls_fail_closed_instead_of_falling_back() -> None:
    unavailable = SimpleNamespace(working=lambda: False)
    with pytest.raises(RuntimeError, match="exact-matching fallback is forbidden"):
        FailClosedJudge(unavailable).working()

    exhausted = SimpleNamespace(
        working=lambda: True,
        generate=lambda *_args, **_kwargs: "Failed to obtain answer via API.",
    )
    judge = FailClosedJudge(exhausted)
    assert judge.working()
    with pytest.raises(RuntimeError, match="exhausted"):
        judge.generate("prompt")


def test_fail_closed_builder_installation_is_idempotent() -> None:
    delegate = SimpleNamespace(working=lambda: True, generate=lambda *_a, **_k: "A")
    module = SimpleNamespace(build_judge=lambda **_kwargs: delegate)

    install_fail_closed_judge_builders((module,))
    first = module.build_judge
    install_fail_closed_judge_builders((module,))

    assert module.build_judge is first
    assert isinstance(module.build_judge(model="judge"), FailClosedJudge)


def test_fail_closed_judge_is_multiprocessing_pickle_safe() -> None:
    delegate = SimpleNamespace(model="judge")
    restored = pickle.loads(pickle.dumps(FailClosedJudge(delegate)))
    assert restored.model == "judge"


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        ("1", "1"),
        ("\n  0\nThe answers differ.", "0"),
        ("10", "10"),
        ("1. The answers match.", "1. The answers match."),
        ("0/1", "0/1"),
        ("Judgement: 1", "Judgement: 1"),
    ),
)
def test_mathverse_score_response_only_accepts_a_leading_binary_token(
    response: str,
    expected: str,
) -> None:
    judge = FailClosedJudge(_StaticJudge(response))
    assert judge.generate(_mathverse_score_prompt()) == expected


def test_mathverse_response_adapter_does_not_change_extract_or_other_prompts() -> None:
    response = "1\nThe extracted answer is one."
    judge = FailClosedJudge(_StaticJudge(response))

    assert judge.generate("Model response: 'one'\nExtracted Answer:") == response
    assert judge.generate("Judge whether these answers match. Judgement:") == response


def test_mathverse_response_adapter_remains_active_after_pickle_round_trip() -> None:
    judge = FailClosedJudge(_StaticJudge("\n1\nThe answers are consistent."))
    restored = pickle.loads(pickle.dumps(judge))

    assert restored.generate(prompt=_mathverse_score_prompt()) == "1"


def _write_health(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "model": COREDEV_LLM_JUDGE_MODEL,
                "base_url": JUDGE_BASE_URL,
            }
        ),
        encoding="utf-8",
    )


def _materialize_eval_fixture(
    root: Path, *, evaluated_model: str = COREDEV_BASELINE_MODEL
) -> dict[str, Path]:
    artifacts = {}
    for spec in COREDEV_2511.slices:
        dataset_root = root / spec.vlmeval_dataset
        run_dir = dataset_root / evaluated_model / "T20260721-150000"
        run_dir.mkdir(parents=True)
        prediction = run_dir / f"{evaluated_model}_{spec.vlmeval_dataset}.tsv"
        rows = ["index\tprediction"]
        rows.extend(f"{index}\tanswer-{index}" for index in range(spec.sample_count))
        prediction.write_text("\n".join(rows) + "\n", encoding="utf-8")

        judge_contract = COREDEV_JUDGE_CONTRACTS[spec.vlmeval_dataset]
        judge_model = "" if judge_contract == "none_rule_based" else COREDEV_LLM_JUDGE_MODEL
        entry = {
            "status": "done",
            "prediction_file": str(prediction),
            "judge_model": judge_model,
            "metrics": {"Overall": 50.0},
            "primary_metric": "Overall",
        }
        status = {
            "schema_version": "1.0",
            "eval_id": "T20260721-150000",
            "model_name": evaluated_model,
            "commit": "7055d301",
            "mode": "eval",
            "datasets": {spec.vlmeval_dataset: entry},
        }
        (run_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
        if judge_contract != "none_rule_based":
            _write_health(dataset_root / "judge-health-pre.json")
            _write_health(dataset_root / "judge-health-post.json")
            artifact = run_dir / (
                f"{evaluated_model}_{spec.vlmeval_dataset}_"
                f"{COREDEV_LLM_JUDGE_MODEL}_result.tsv"
            )
            artifact.write_text("index\tlog\n0\tSucceed\n", encoding="utf-8")
            artifacts[spec.vlmeval_dataset] = artifact
    return artifacts


def test_seven_slice_aggregate_requires_complete_status_rows_and_judge_evidence(
    tmp_path: Path,
) -> None:
    _materialize_eval_fixture(tmp_path)

    result = summarize_coredev_results(
        work_dir=tmp_path,
        repository_root=tmp_path,
        phase="eval",
        expected_judge_base_url=JUDGE_BASE_URL,
    )

    assert result["status"] == "pass"
    assert result["sample_count"] == 2511
    assert result["slice_count"] == 7
    assert tuple(item["dataset"] for item in result["slices"]) == tuple(
        spec.vlmeval_dataset for spec in COREDEV_2511.slices
    )


def test_seven_slice_aggregate_accepts_explicit_instruct_model_identity(
    tmp_path: Path,
) -> None:
    evaluated_model = "Qwen3-VL-8B-Instruct"
    _materialize_eval_fixture(tmp_path, evaluated_model=evaluated_model)

    result = summarize_coredev_results(
        work_dir=tmp_path,
        repository_root=tmp_path,
        phase="eval",
        expected_judge_base_url=JUDGE_BASE_URL,
        expected_model=evaluated_model,
    )

    assert result["status"] == "pass"
    assert result["model"] == evaluated_model
    assert result["sample_count"] == 2511


def test_aggregate_rejects_silent_exact_or_random_judge_fallback(
    tmp_path: Path,
) -> None:
    artifacts = _materialize_eval_fixture(tmp_path)
    artifacts["VStarBench"].write_text(
        "index\tlog\n0\tFailed in Prefetch, no GPT-based answer matching\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fallback/failure marker"):
        summarize_coredev_results(
            work_dir=tmp_path,
            repository_root=tmp_path,
            phase="eval",
            expected_judge_base_url=JUDGE_BASE_URL,
        )


def test_aggregate_never_falls_back_to_an_older_success_after_new_failure(
    tmp_path: Path,
) -> None:
    _materialize_eval_fixture(tmp_path)
    failed_dir = (
        tmp_path
        / "VStarBench"
        / COREDEV_BASELINE_MODEL
        / "T20260721-160000"
    )
    failed_dir.mkdir()
    (failed_dir / "status.json").write_text(
        json.dumps(
            {
                "eval_id": "T20260721-160000",
                "model_name": COREDEV_BASELINE_MODEL,
                "commit": "7055d301",
                "mode": "eval",
                "datasets": {
                    "VStarBench": {
                        "status": "done",
                        "error_message": "judge unavailable",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="latest eval run failed"):
        summarize_coredev_results(
            work_dir=tmp_path,
            repository_root=tmp_path,
            phase="eval",
            expected_judge_base_url=JUDGE_BASE_URL,
        )
