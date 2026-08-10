from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.evaluation.policy_coredev_scoring import (
    materialize_policy_coredev_scoring_views,
    normalize_policy_final_answer,
    validate_vlmevalkit_eval_id,
)


@pytest.mark.parametrize(
    "run_id",
    [
        "T20260810-123456",
        "T20260810_G0",
        "T20260810_GdeadBEEF0123456789",
    ],
)
def test_validate_vlmevalkit_eval_id_accepts_exact_scanner_grammars(
    run_id: str,
) -> None:
    assert validate_vlmevalkit_eval_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        None,
        "",
        "T20260810-PRL15-R1-RP66-STEP0",
        "T20260810-123456-STEP0",
        "T20260810_Gnot-hex",
        "prefix-T20260810-123456",
    ],
)
def test_validate_vlmevalkit_eval_id_rejects_non_discoverable_ids(
    run_id: object,
) -> None:
    with pytest.raises(ValueError, match="VLMEvalKit eval ID"):
        validate_vlmevalkit_eval_id(run_id)


def test_scoring_materializer_rejects_bad_run_id_before_reading_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="VLMEvalKit eval ID"):
        materialize_policy_coredev_scoring_views(
            inference_root=tmp_path / "missing-inference",
            tasks_path=tmp_path / "missing-tasks.jsonl",
            source_root=tmp_path / "missing-source",
            output_root=tmp_path / "output",
            evaluation_id="SEMANTIC-EVALUATION-ID",
            run_id="T20260810-PRL15-R1-RP66-STEP0",
            mathverse_source_json=tmp_path / "missing-mathverse.json",
        )


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
