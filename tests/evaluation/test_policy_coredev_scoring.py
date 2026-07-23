from __future__ import annotations

import pytest

from tgvf_rl.evaluation.policy_coredev_scoring import normalize_policy_final_answer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A<|im_end|>", "A"),
        ("value 12  <|im_end|>  ", "value 12"),
        ("A<|im_end|><|endoftext|>", "A"),
        (None, None),
        ("  ", None),
    ],
)
def test_normalize_policy_final_answer_removes_only_terminal_markers(
    raw: object, expected: str | None
) -> None:
    assert normalize_policy_final_answer(raw) == expected


def test_normalize_policy_final_answer_rejects_non_text() -> None:
    with pytest.raises(TypeError):
        normalize_policy_final_answer(7)
