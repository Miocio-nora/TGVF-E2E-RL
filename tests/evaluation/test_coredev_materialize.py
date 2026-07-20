from tgvf_rl.evaluation.coredev_materialize import (
    COREDEV_DATASET_CLASSES,
    COREDEV_JUDGE_CONTRACTS,
    COREDEV_LLM_JUDGE_MODEL,
    COREDEV_LLM_JUDGE_REPOSITORY,
    coredev_runtime_class_name,
    _vstar_fields,
)


def test_vstar_transform_uses_official_mcq_columns() -> None:
    fields = _vstar_fields(
        {
            "text": "Which?\n(A) one\n(B) two\n(C) three\n(D) four\nAnswer directly.",
            "label": "C",
            "category": "relative_position",
        }
    )
    assert fields == {
        "question": "Which?",
        "A": "one",
        "B": "two",
        "C": "three",
        "D": "four",
        "answer": "C",
        "category": "relative_position",
    }


def test_seven_slices_bind_official_classes_and_explicit_judge_contracts() -> None:
    assert set(COREDEV_DATASET_CLASSES) == set(COREDEV_JUDGE_CONTRACTS)
    assert len(COREDEV_DATASET_CLASSES) == 7
    assert COREDEV_JUDGE_CONTRACTS["OCRBench_v2"] == "none_rule_based"
    assert COREDEV_LLM_JUDGE_REPOSITORY == "Qwen/Qwen2.5-72B-Instruct"
    assert COREDEV_LLM_JUDGE_MODEL == "Qwen2.5-72B-Instruct"
    assert COREDEV_JUDGE_CONTRACTS["MathVista_MINI"] == "required_qwen2_5_72b_judge"
    assert COREDEV_JUDGE_CONTRACTS["MathVerse_MINI"] == "required_qwen2_5_72b_judge"
    assert coredev_runtime_class_name("MMMU_Pro_10c") == "CoreDev2511MMMUPro10cSlice"
