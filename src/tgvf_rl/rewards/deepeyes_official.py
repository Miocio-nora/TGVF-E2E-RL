"""Pure DeepEyes-derived reward contract for the native Crop control.

No network client is constructed here.  Production supplies a text-only
Qwen2.5-72B binary judge implementing :class:`BinarySemanticJudge`; tests use
an in-memory fake.  This boundary makes it impossible for a CPU contract test
to spend API credit while still proving that every visual trajectory reaches
the semantic-judge route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol


DEEPEYES_OFFICIAL_REWARD_SCHEMA = "tgvf.deepeyes-native-reward.v1"
DEEPEYES_BINARY_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEEPEYES_VISUAL_JUDGE_TEMPERATURE = 0.3
DEEPEYES_THINKLITE_JUDGE_TEMPERATURE = 0.0
# Backward-compatible name for the visual route, which is the mandatory and
# dominant judge load in DeepEyes.
DEEPEYES_BINARY_JUDGE_TEMPERATURE = DEEPEYES_VISUAL_JUDGE_TEMPERATURE
DEEPEYES_BINARY_JUDGE_TOP_P = 1.0
DEEPEYES_BINARY_JUDGE_MAX_TOKENS = 16
DEEPEYES_VISUAL_ANSWER_LIMIT = 1000
DEEPEYES_VISUAL_ANSWER_WEIGHT = 0.8
DEEPEYES_VISUAL_FORMAT_WEIGHT = 0.2
DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT = 1.2
DEEPEYES_THINKLITE_ANSWER_WEIGHT = 1.2
DEEPEYES_THINKLITE_FORMAT_WEIGHT = 0.4

_ASSISTANT_TURN_BOUNDARY = re.compile(r"(?:^|\n)assistant\n")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_LEGACY_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

DEEPEYES_VISUAL_JUDGE_PROMPT_KIND = "vl_agent_get_prompt"
DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND = "vl_agent_math_verify_prompt"

# Exact observable templates from DeepEyes ``vl_agent.py``.  V*/Chart route
# uses ``get_prompt`` (not ``COMMON_VERIFY_PROMPT``); ThinkLite's failed
# math-verify route uses ``MATH_VERIFY_PROMPT``.
DEEPEYES_VISUAL_JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."
DEEPEYES_VISUAL_JUDGE_PREAMBLE = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
DEEPEYES_VISUAL_JUDGE_EXAMPLES = (
    """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
    """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""",
    """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""",
    """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""",
    """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""",
    """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""",
    """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""",
)
DEEPEYES_VISUAL_JUDGE_TEST_TEMPLATE = """
[Question]: {question}
[Standard Answer]: {reference_answer}
[Model_answer] : {candidate_answer}
Judgement:"""

DEEPEYES_MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{question}

**Reference Answer**
{reference_answer}

## Student Final Answer
{candidate_answer}"""


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256 = _sha256_json(
    {
        "schema": "deepeyes-vl-agent-get-prompt-v1",
        "system": DEEPEYES_VISUAL_JUDGE_SYSTEM_PROMPT,
        "preamble": DEEPEYES_VISUAL_JUDGE_PREAMBLE,
        "examples": DEEPEYES_VISUAL_JUDGE_EXAMPLES,
        "test_template": DEEPEYES_VISUAL_JUDGE_TEST_TEMPLATE,
        "output": "judgement_ascii_0_or_1",
    }
)
DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256 = _sha256_json(
    {
        "schema": "deepeyes-vl-agent-math-verify-prompt-v1",
        "user_template": DEEPEYES_MATH_VERIFY_PROMPT,
        "output": "equivalence_judgement_true_or_false",
    }
)
DEEPEYES_BINARY_JUDGE_PROMPT_SHA256 = DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256


def build_visual_judge_user_prompt(
    *, question: str, reference_answer: str, candidate_answer: str
) -> str:
    prompt = DEEPEYES_VISUAL_JUDGE_PREAMBLE
    for example in DEEPEYES_VISUAL_JUDGE_EXAMPLES:
        prompt += example + "\n\n"
    return prompt + DEEPEYES_VISUAL_JUDGE_TEST_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )


@dataclass(frozen=True, slots=True)
class DeepEyesBinaryJudgeRequest:
    trajectory_id: str
    sample_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    task_kind: str
    prompt_kind: str
    request_id: str

    @classmethod
    def build(
        cls,
        *,
        trajectory_id: str,
        sample_id: str,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        task_kind: str,
        prompt_kind: str,
    ) -> "DeepEyesBinaryJudgeRequest":
        for name, value in (
            ("trajectory_id", trajectory_id),
            ("sample_id", sample_id),
            ("question", question),
            ("reference_answer", reference_answer),
            ("candidate_answer", candidate_answer),
            ("task_kind", task_kind),
            ("prompt_kind", prompt_kind),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        prompt_hashes = {
            DEEPEYES_VISUAL_JUDGE_PROMPT_KIND: DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256,
            DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND: (
                DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256
            ),
        }
        if prompt_kind not in prompt_hashes:
            raise ValueError("DeepEyes judge prompt_kind is unsupported")
        identity = {
            "schema": "deepeyes-binary-judge-request-v1",
            "trajectory_id": trajectory_id,
            "sample_id": sample_id,
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
            "task_kind": task_kind,
            "prompt_kind": prompt_kind,
            "prompt_sha256": prompt_hashes[prompt_kind],
        }
        return cls(
            trajectory_id=trajectory_id,
            sample_id=sample_id,
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            task_kind=task_kind,
            prompt_kind=prompt_kind,
            request_id=_sha256_json(identity),
        )

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        if self.prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND:
            return (
                {"role": "system", "content": DEEPEYES_VISUAL_JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_visual_judge_user_prompt(
                        question=self.question,
                        reference_answer=self.reference_answer,
                        candidate_answer=self.candidate_answer,
                    ),
                },
            )
        return (
            {
                "role": "user",
                "content": DEEPEYES_MATH_VERIFY_PROMPT.format(
                    question=self.question,
                    reference_answer=self.reference_answer,
                    candidate_answer=self.candidate_answer,
                ),
            },
        )


class BinarySemanticJudge(Protocol):
    """Text-only Qwen2.5-72B service boundary; verdict is exactly bool."""

    def judge(self, request: DeepEyesBinaryJudgeRequest) -> bool: ...


def parse_binary_judge_output(text: object, *, prompt_kind: str) -> bool:
    """Parse only the two output shapes requested by official DeepEyes."""

    if not isinstance(text, str):
        raise TypeError("judge output must be text")
    value = text.strip()
    if prompt_kind == DEEPEYES_VISUAL_JUDGE_PROMPT_KIND:
        if value.startswith("Judgement:"):
            value = value.removeprefix("Judgement:").strip()
        if value == "1":
            return True
        if value == "0":
            return False
        raise ValueError("visual semantic judge must output Judgement: 0 or 1")
    if prompt_kind == DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND:
        judgement = value.split("## Equivalence Judgement")[-1].strip().casefold()
        if "true" in judgement and "false" not in judgement:
            return True
        if "false" in judgement and "true" not in judgement:
            return False
        raise ValueError("ThinkLite semantic judge must output unambiguous TRUE/FALSE")
    raise ValueError("DeepEyes judge prompt_kind is unsupported")


@dataclass(frozen=True, slots=True)
class DeepEyesAnswerExtraction:
    answer: str
    format_penalty: int
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        if self.format_penalty not in {0, -1}:
            raise ValueError("format_penalty must be 0 or -1")
        if self.valid != (self.format_penalty == 0):
            raise ValueError("format validity/penalty differ")


@dataclass(frozen=True, slots=True)
class DeepEyesRewardResult:
    total: float
    accuracy: int
    format_penalty: int
    conditional_tool: int
    answer: str
    verifier_route: str
    judge_called: bool

    def __post_init__(self) -> None:
        if self.accuracy not in {0, 1}:
            raise ValueError("accuracy must be binary")
        if self.format_penalty not in {0, -1}:
            raise ValueError("format_penalty must be 0 or -1")
        if self.conditional_tool not in {0, 1}:
            raise ValueError("conditional_tool must be binary")


def extract_visual_answer(response: str) -> DeepEyesAnswerExtraction:
    """Extract the direct final answer from the clean no-wrapper dialect.

    Native multi-turn dumps contain earlier assistant actions plus injected
    user observations.  The final assistant segment is therefore selected
    before reasoning/tool blocks are removed.  A legacy answer wrapper is
    unwrapped only so pre-fix checkpoints remain measurable; emitting it is a
    format violation and is never required for extraction.
    """

    if not isinstance(response, str):
        raise TypeError("response must be text")
    reasons: list[str] = []
    if response.count("<think>") != response.count("</think>"):
        reasons.append("think_count")
    if response.count("<|vision_start|><|image_pad|>") != response.count(
        "<|image_pad|><|vision_end|>"
    ):
        reasons.append("vision_count")
    if response.count("<tool_call>") != response.count("</tool_call>"):
        reasons.append("tool_call_count")
    if "<answer>" in response or "</answer>" in response:
        reasons.append("legacy_answer_wrapper")

    final_turn = _ASSISTANT_TURN_BOUNDARY.split(response)[-1]
    legacy_answers = _LEGACY_ANSWER_BLOCK.findall(final_turn)
    if legacy_answers:
        answer = legacy_answers[-1].strip()
    else:
        answer = _THINK_BLOCK.sub("", final_turn)
        answer = _TOOL_CALL_BLOCK.sub("", answer)
        for terminal in ("<|im_end|>", "<|endoftext|>"):
            answer = answer.replace(terminal, "")
        answer = answer.strip()
    if not answer:
        reasons.append("missing_direct_answer")
    valid = not reasons
    return DeepEyesAnswerExtraction(
        answer=answer,
        format_penalty=0 if valid else -1,
        valid=valid,
        reason="ok" if valid else ",".join(dict.fromkeys(reasons)),
    )


def extract_thinklite_answer(response: str) -> DeepEyesAnswerExtraction:
    """Copy ``compute_score_math``: last post-think boxed span wins."""

    if not isinstance(response, str):
        raise TypeError("response must be text")
    reasons: list[str] = []
    if response.count("<think>") != response.count("</think>"):
        reasons.append("think_count")
    after_reasoning = response.split("</think>")[-1]
    spans = re.findall(r"\\boxed{([^}]+)}", after_reasoning, flags=re.DOTALL)
    if not spans:
        reasons.append("missing_boxed")
    elif len(spans) > 1:
        reasons.append("multiple_boxed")
    answer = spans[-1] if spans else ""
    valid = not reasons
    return DeepEyesAnswerExtraction(
        answer=answer,
        format_penalty=0 if valid else -1,
        valid=valid,
        reason="ok" if valid else ",".join(dict.fromkeys(reasons)),
    )


def score_visual_trajectory(
    *,
    trajectory_id: str,
    sample_id: str,
    question: str,
    reference_answer: str,
    response: str,
    task_kind: str,
    successful_crop_count: int,
    judge: BinarySemanticJudge,
) -> DeepEyesRewardResult:
    """Apply ``0.8*acc + 0.2*format + 1.2*I(acc&crop_success)``.

    The judge is called exactly once even for malformed or overlong answers.
    As in official DeepEyes, format is an independent reward term; only the
    1000-character guard can force an otherwise positive judge verdict to 0.
    """

    if type(successful_crop_count) is not int or successful_crop_count < 0:
        raise ValueError("successful_crop_count must be a non-negative integer")
    extracted = extract_visual_answer(response)
    candidate_for_judge = extracted.answer or "[NO VALID FINAL ANSWER]"
    request = DeepEyesBinaryJudgeRequest.build(
        trajectory_id=trajectory_id,
        sample_id=sample_id,
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_for_judge,
        task_kind=task_kind,
        prompt_kind=DEEPEYES_VISUAL_JUDGE_PROMPT_KIND,
    )
    judged_correct = judge.judge(request)
    if type(judged_correct) is not bool:
        raise TypeError("DeepEyes binary judge must return bool")
    too_long = len(extracted.answer) >= DEEPEYES_VISUAL_ANSWER_LIMIT
    accuracy = int(judged_correct and not too_long)
    format_penalty = -1 if too_long else extracted.format_penalty
    conditional_tool = int(accuracy == 1 and successful_crop_count > 0)
    total = (
        DEEPEYES_VISUAL_ANSWER_WEIGHT * accuracy
        + DEEPEYES_VISUAL_FORMAT_WEIGHT * format_penalty
        + DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT * conditional_tool
    )
    return DeepEyesRewardResult(
        total=total,
        accuracy=accuracy,
        format_penalty=format_penalty,
        conditional_tool=conditional_tool,
        answer=extracted.answer,
        verifier_route="qwen2.5_72b_every_visual_trajectory",
        judge_called=True,
    )


class MathVerifier(Protocol):
    def __call__(self, reference_answer: str, candidate_answer: str) -> bool: ...


def score_thinklite_trajectory(
    *,
    trajectory_id: str,
    sample_id: str,
    question: str,
    reference_answer: str,
    response: str,
    task_kind: str,
    math_verify: MathVerifier,
    judge: BinarySemanticJudge,
) -> DeepEyesRewardResult:
    """Apply ThinkLite rule-first verification with the same 72B fallback."""

    extracted = extract_thinklite_answer(response)
    rule_correct = False
    if extracted.answer:
        try:
            rule_correct = math_verify(reference_answer, extracted.answer)
        except Exception:  # math-verify parse failures are semantic-fallback cases
            rule_correct = False
        if type(rule_correct) is not bool:
            raise TypeError("math_verify must return bool")
    judge_called = False
    if rule_correct:
        accuracy = 1
        route = "math_verify"
    elif extracted.answer:
        judge_called = True
        judged_correct = judge.judge(
            DeepEyesBinaryJudgeRequest.build(
                trajectory_id=trajectory_id,
                sample_id=sample_id,
                question=question,
                reference_answer=reference_answer,
                candidate_answer=extracted.answer,
                task_kind=task_kind,
                prompt_kind=DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND,
            )
        )
        if type(judged_correct) is not bool:
            raise TypeError("DeepEyes binary judge must return bool")
        accuracy = int(judged_correct)
        route = "math_verify_then_qwen2.5_72b"
    else:
        accuracy = 0
        route = "missing_boxed_answer"
    total = (
        DEEPEYES_THINKLITE_ANSWER_WEIGHT * accuracy
        + DEEPEYES_THINKLITE_FORMAT_WEIGHT * extracted.format_penalty
    )
    return DeepEyesRewardResult(
        total=total,
        accuracy=accuracy,
        format_penalty=extracted.format_penalty,
        conditional_tool=0,
        answer=extracted.answer,
        verifier_route=route,
        judge_called=judge_called,
    )


__all__ = [
    "BinarySemanticJudge",
    "DEEPEYES_BINARY_JUDGE_MAX_TOKENS",
    "DEEPEYES_BINARY_JUDGE_MODEL",
    "DEEPEYES_BINARY_JUDGE_PROMPT_SHA256",
    "DEEPEYES_BINARY_JUDGE_TEMPERATURE",
    "DEEPEYES_BINARY_JUDGE_TOP_P",
    "DEEPEYES_MATH_VERIFY_PROMPT",
    "DEEPEYES_OFFICIAL_REWARD_SCHEMA",
    "DEEPEYES_THINKLITE_ANSWER_WEIGHT",
    "DEEPEYES_THINKLITE_FORMAT_WEIGHT",
    "DEEPEYES_THINKLITE_JUDGE_PROMPT_KIND",
    "DEEPEYES_THINKLITE_JUDGE_PROMPT_SHA256",
    "DEEPEYES_THINKLITE_JUDGE_TEMPERATURE",
    "DEEPEYES_VISUAL_ANSWER_LIMIT",
    "DEEPEYES_VISUAL_ANSWER_WEIGHT",
    "DEEPEYES_VISUAL_CONDITIONAL_TOOL_WEIGHT",
    "DEEPEYES_VISUAL_FORMAT_WEIGHT",
    "DEEPEYES_VISUAL_JUDGE_PROMPT_KIND",
    "DEEPEYES_VISUAL_JUDGE_PROMPT_SHA256",
    "DEEPEYES_VISUAL_JUDGE_TEMPERATURE",
    "DeepEyesAnswerExtraction",
    "DeepEyesBinaryJudgeRequest",
    "DeepEyesRewardResult",
    "build_visual_judge_user_prompt",
    "extract_thinklite_answer",
    "extract_visual_answer",
    "parse_binary_judge_output",
    "score_thinklite_trajectory",
    "score_visual_trajectory",
]
