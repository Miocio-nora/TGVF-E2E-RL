"""Exercise real MCQ/math/open reward routes against the RL judge service."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from tgvf_rl.contracts.identity import ArtifactIdentity
from tgvf_rl.judges import (
    OpenAICompatibleJudgeProvider,
    QWEN25_72B_RL_JUDGE_PROMPT_VERSION,
    QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT,
    load_openai_compatible_judge,
)
from tgvf_rl.rewards import AnswerTaskKind, PilotRewardPipeline, RewardContext
from tgvf_rl.rewards.schema import NormalizationSpec, PilotRewardSpec
from tgvf_rl.rewards.verifiers import RuleFirstAnswerVerifier


_REAL_CASES = {
    "mcq": {
        "sample_id": "deepeyes47k:4b74dbfbd36b31129eb811b919526a30448b05d2c05bb11736c7bc75c02e718e",
        "candidate_correct": "D",
        "candidate_wrong": "A",
    },
    "math": {
        "sample_id": "deepeyes47k:bdd3c553a12609c6b7082687d6d153d14febf9c60d8ffa90c5b39087b3acd925",
        "candidate_correct": "4.5 square units",
        "candidate_wrong": "9 square units",
    },
    "open": {
        "sample_id": "deepeyes47k:9f878be9ea753995964ee6cd74f4949097dd044671e34facb00df3cd5d26ca2c",
        "candidate_correct": "It commemorates Malaysia in 2013.",
        "candidate_wrong": "It is a birthday cake for a child.",
    },
}


class _CountingJudge:
    def __init__(self, delegate: OpenAICompatibleJudgeProvider) -> None:
        self.delegate = delegate
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def judge(self, request):
        self.calls += 1
        result = self.delegate.judge(request)
        if result.usage is not None:
            self.prompt_tokens += result.usage.prompt_tokens
            self.completion_tokens += result.usage.completion_tokens
            self.cost_usd += result.usage.cost_usd
        return result


def _canonical_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _artifact(name: str, version: str, payload: object) -> ArtifactIdentity:
    return ArtifactIdentity("policy-rl-answer-judge", name, version, _canonical_sha(payload))


def _load_rows(samples_path: Path) -> dict[str, dict[str, Any]]:
    wanted = {case["sample_id"] for case in _REAL_CASES.values()}
    rows: dict[str, dict[str, Any]] = {}
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") in wanted:
                rows[row["sample_id"]] = row
            if len(rows) == len(wanted):
                break
    if set(rows) != wanted:
        raise RuntimeError("real judge-route samples are missing")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(
            "artifacts/data/deepeyes47k/materialized-seed42-v1/samples.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["scope"]["allows_policy_rl_reward"] is not True:
        raise RuntimeError("judge config does not allow Policy RL reward")
    if config["scope"]["allows_mcq_judge_calls"] is not False:
        raise RuntimeError("MCQ judge calls must remain forbidden")
    prompt = config["prompt"]
    actual_prompt_sha = sha256(
        QWEN25_72B_RL_JUDGE_SYSTEM_PROMPT.encode()
    ).hexdigest()
    if (
        prompt["version"] != QWEN25_72B_RL_JUDGE_PROMPT_VERSION
        or prompt["sha256"] != actual_prompt_sha
    ):
        raise RuntimeError("RL judge prompt identity differs")

    bound = load_openai_compatible_judge(
        args.config,
        expected_file_sha256=sha256(args.config.read_bytes()).hexdigest(),
    )
    bound.provider.validate_credentials()
    judge = _CountingJudge(bound.provider)
    verifier = RuleFirstAnswerVerifier(
        rule_identity=_artifact("rule-first", "v1", {"routes": ["mcq", "math", "open"]}),
        normalization=NormalizationSpec(True, True, True),
        judge=judge,
        judge_prompt_identity=bound.prompt_identity,
        judge_model_identity=bound.model_identity,
        judge_service_identity=bound.service_identity,
        judge_sampling_identity=bound.sampling_identity,
        judge_calibration_identity=bound.calibration_identity,
    )
    reward_spec = PilotRewardSpec(
        pipeline_identity=_artifact("reward-pipeline", "v1", {"weights": [0.8, 0.2, 1.2]}),
        answer_verifier_identity=verifier.rule_identity,
        format_verifier_identity=_artifact("format", "v1", {"invalid": -1}),
        tool_verifier_identity=_artifact("conditional-tool", "v1", {"once": True}),
    )
    reward_pipeline = PilotRewardPipeline(reward_spec, verifier)
    rows = _load_rows(args.samples)
    kinds = {
        "mcq": AnswerTaskKind.MULTIPLE_CHOICE,
        "math": AnswerTaskKind.MATH,
        "open": AnswerTaskKind.OPEN_VQA,
    }
    results = []
    for route, case in _REAL_CASES.items():
        row = rows[case["sample_id"]]
        base = RewardContext(
            sample_id=row["sample_id"],
            question=row["extra_info"]["question"],
            candidate_answer=case["candidate_correct"],
            expected_answer=row["reward_model"]["ground_truth"],
            tool_call_count=0,
            task_kind=kinds[route],
            data_source=row["data_source"],
        )
        before = judge.calls
        correct_reward = reward_pipeline.score(base)
        wrong_reward = reward_pipeline.score(
            replace(base, candidate_answer=case["candidate_wrong"])
        )
        correct = correct_reward.answer_verification
        wrong = wrong_reward.answer_verification
        if correct is None or wrong is None:
            raise RuntimeError(f"{route} reward omitted answer verification")
        calls = judge.calls - before
        expected_calls = 0 if route == "mcq" else 2
        if (
            not correct.correct
            or wrong.correct
            or correct_reward.total != 0.8
            or wrong_reward.total != 0.0
            or calls != expected_calls
        ):
            raise RuntimeError(
                f"{route} route failed: correct={correct.correct}, "
                f"wrong={wrong.correct}, correct_reward={correct_reward.total}, "
                f"wrong_reward={wrong_reward.total}, judge_calls={calls}"
            )
        results.append(
            {
                "route": route,
                "sample_id": row["sample_id"],
                "correct_route": correct.route,
                "wrong_route": wrong.route,
                "judge_calls": calls,
                "correct_reward": correct_reward.total,
                "wrong_reward": wrong_reward.total,
            }
        )
    output = {
        "status": "pass",
        "judge_calls": judge.calls,
        "judge_prompt_tokens": judge.prompt_tokens,
        "judge_completion_tokens": judge.completion_tokens,
        "judge_cost_usd": judge.cost_usd,
        "cases": results,
        "prompt_sha256": bound.prompt_identity.sha256,
        "model_revision": config["model"]["revision"],
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
