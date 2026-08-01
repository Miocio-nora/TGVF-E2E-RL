from __future__ import annotations

from pathlib import Path
import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test lane
    import tomli as tomllib

    sys.modules.setdefault("tomllib", tomllib)

import pytest
import torch
from PIL import Image

import tgvf_rl.representation.experiments.answer_utility.evaluation.runner as evaluation_runner_module
from tgvf_rl.representation.experiments.answer_utility.evaluation.runner import (
    DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS,
    AnswerUtilityEvaluationArm,
    _QwenImageGridContract,
    _evaluation_arm_contract,
    _generate_pending_answers,
    _implementation_file_manifest,
    _qwen_image_grid_thw,
    _reader_model_input,
    _same_target_wrong_image_model_inputs,
    build_answer_safe_wrong_mapping,
    build_same_target_wrong_image_mapping,
    load_answer_utility_adapter_artifact,
)
from tgvf_rl.representation.experiments.answer_utility.evaluation.scoring import (
    INSTRUCT_READER_INSTRUCTION,
    reader_question,
    score_instruct_generated_answer,
)
from tgvf_rl.representation.experiments.answer_utility.runner import (
    ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
    _answer_utility_state_digest,
)
from tgvf_rl.representation.training.oracle_d_utility import (
    OracleBatchCompatibilityError,
    OracleDUtilityGroundTruth,
    OracleGeneratedAnswer,
    split_oracle_d_utility_sample,
)
from tgvf_rl.representation.training.schema import (
    RepresentationChoice,
    RepresentationTrainingSample,
)


def _sample(
    index: int,
    *,
    answer: str,
    target: str | None = None,
    image: str = "/fixture/shared.png",
    image_id: str = "shared-image",
    source_dataset: str | None = None,
    source_profile: str | None = None,
    stable_image_uid: str | None = None,
) -> RepresentationTrainingSample:
    return RepresentationTrainingSample(
        sample_id=f"sample-{index}",
        image=image,
        image_id=image_id,
        question=f"question {index}",
        target=target or f"target {index}",
        evidence_description=f"evidence {index}",
        short_answer=answer,
        source_dataset=source_dataset,
        source_profile=source_profile,
        stable_image_uid=stable_image_uid,
    )


def _artifact_payload() -> dict[str, object]:
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    return {
        "schema_version": ANSWER_UTILITY_ARTIFACT_SCHEMA_VERSION,
        "run_identity_sha256": "1" * 64,
        "global_step": 500,
        "source_artifact_sha256": "2" * 64,
        "experiment_config_sha256": "3" * 64,
        "adapter_state_sha256": _answer_utility_state_digest(state),
        "adapter_state": state,
    }


def _truth(expected: str, choices: tuple[str, ...]) -> OracleDUtilityGroundTruth:
    return OracleDUtilityGroundTruth(
        sample_id="sample",
        short_answer=expected,
        choices=tuple(
            RepresentationChoice(label=chr(ord("A") + index), text=text)
            for index, text in enumerate(choices)
        ),
    )


def test_default_evaluation_declares_complete_primary_seven_arm_suite() -> None:
    assert DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS == (
        AnswerUtilityEvaluationArm.IMAGE_ONLY,
        AnswerUtilityEvaluationArm.D_ONLY_ZERO,
        AnswerUtilityEvaluationArm.D_ONLY_CORRECT,
        AnswerUtilityEvaluationArm.D_ONLY_WRONG,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_ZERO,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_CORRECT,
        AnswerUtilityEvaluationArm.IMAGE_PLUS_WRONG,
    )


def test_same_target_wrong_image_arm_declares_recomputed_contextual_contract() -> None:
    contract = _evaluation_arm_contract(
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
    )

    assert contract["source_image"] is True
    assert contract["oracle_target_transcript"] is True
    assert contract["contextual_target_hidden_state"] == "recomputed_on_wrong_image"
    assert "same_anchor_question_target" in contract["d"]
    assert (
        AnswerUtilityEvaluationArm.IMAGE_PLUS_SAME_TARGET_WRONG_IMAGE
        not in DEFAULT_ANSWER_UTILITY_EVALUATION_ARMS
    )


def _write_fixture_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color=color).save(path)


def _fixture_grid_contract() -> _QwenImageGridContract:
    return _QwenImageGridContract(
        patch_size=16,
        merge_size=2,
        min_pixels=32 * 32,
        max_pixels=64 * 64,
    )


def test_same_target_wrong_image_mapping_is_exact_grid_distinct_and_deterministic(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.png"
    other_anchor_path = tmp_path / "other-anchor.png"
    donor_path = tmp_path / "donor.png"
    _write_fixture_image(anchor_path, (10, 20, 30))
    _write_fixture_image(other_anchor_path, (70, 80, 90))
    _write_fixture_image(donor_path, (40, 50, 60))
    anchors = (
        _sample(
            0,
            answer="red",
            image=str(anchor_path),
            image_id="anchor-image",
            source_dataset="fixture-source",
            source_profile="fixture-profile",
            stable_image_uid="anchor-uid",
        ),
        _sample(
            1,
            answer="blue",
            image=str(anchor_path),
            image_id="anchor-image",
            source_dataset="fixture-source",
            source_profile="fixture-profile",
            stable_image_uid="anchor-uid",
        ),
    )
    donors = (
        _sample(
            2,
            answer="green",
            image=str(donor_path),
            image_id="donor-image",
            source_dataset="fixture-source",
            source_profile="fixture-profile",
            stable_image_uid="donor-uid",
        ),
        _sample(
            3,
            answer="yellow",
            image=str(donor_path),
            image_id="donor-image",
            source_dataset="fixture-source",
            source_profile="fixture-profile",
            stable_image_uid="donor-uid",
        ),
    )

    first = build_same_target_wrong_image_mapping(
        ((7, anchors),),
        donors,
        grid_contract=_fixture_grid_contract(),
        random_seed=42,
    )
    second = build_same_target_wrong_image_mapping(
        ((7, anchors),),
        donors,
        grid_contract=_fixture_grid_contract(),
        random_seed=42,
    )
    other_anchor = _sample(
        4,
        answer="purple",
        image=str(other_anchor_path),
        image_id="other-anchor-image",
        source_dataset="fixture-source",
        source_profile="fixture-profile",
        stable_image_uid="other-anchor-uid",
    )
    reordered_and_sharded = build_same_target_wrong_image_mapping(
        ((99, (other_anchor,)), (7, anchors)),
        tuple(reversed(donors)),
        grid_contract=_fixture_grid_contract(),
        random_seed=42,
    )

    assert first == second
    assert reordered_and_sharded["anchor-image"] == first["anchor-image"]
    donor = first["anchor-image"]
    assert donor.donor_image_group_key == "donor-image"
    assert donor.match_tier == "exact_grid_same_source_dataset"
    assert donor.image_grid_thw == (1, 4, 4)
    assert donor.anchor_image_sha256 != donor.donor_image_sha256

    anchor_model, _truth = split_oracle_d_utility_sample(anchors[0])
    changed = _same_target_wrong_image_model_inputs((anchor_model,), donor)[0]
    assert changed.image == str(donor_path)
    assert changed.image_group_key == "donor-image"
    assert changed.question == anchor_model.question
    assert changed.target == anchor_model.target
    assert changed.sample_id == anchor_model.sample_id
    assert changed.sample_content_sha256 == anchor_model.sample_content_sha256


def test_same_target_wrong_image_mapping_rejects_identical_image_bytes(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.png"
    duplicate_path = tmp_path / "duplicate.png"
    _write_fixture_image(anchor_path, (10, 20, 30))
    duplicate_path.write_bytes(anchor_path.read_bytes())
    anchor = _sample(
        0,
        answer="red",
        image=str(anchor_path),
        image_id="anchor-image",
        source_dataset="fixture-source",
        source_profile="fixture-profile",
    )
    duplicate = _sample(
        1,
        answer="green",
        image=str(duplicate_path),
        image_id="duplicate-image",
        source_dataset="fixture-source",
        source_profile="fixture-profile",
    )

    with pytest.raises(ValueError, match="byte-distinct wrong image"):
        build_same_target_wrong_image_mapping(
            ((0, (anchor,)),),
            (duplicate,),
            grid_contract=_fixture_grid_contract(),
            random_seed=42,
        )


def test_qwen_grid_prediction_uses_exact_patch_merge_geometry(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "square.png"
    _write_fixture_image(image_path, (1, 2, 3))

    assert _qwen_image_grid_thw(str(image_path), _fixture_grid_contract()) == (1, 4, 4)


def test_wrong_mapping_skips_cyclic_neighbor_with_same_normalized_answer() -> None:
    samples = (
        _sample(0, answer="500"),
        _sample(1, answer="  500  "),
        _sample(2, answer="600"),
    )

    mapping = build_answer_safe_wrong_mapping(((27, samples),))

    assert mapping == {
        "sample-0": "sample-2",
        "sample-1": "sample-2",
        "sample-2": "sample-0",
    }


def test_wrong_mapping_requires_both_different_answer_and_target() -> None:
    samples = (
        _sample(0, answer="A", target="same target"),
        _sample(1, answer="B", target=" SAME   TARGET "),
    )

    with pytest.raises(ValueError, match="different-target/different-answer"):
        build_answer_safe_wrong_mapping(((0, samples),))


def test_private_artifact_loader_accepts_only_bound_seven_key_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answer_utility_adapter.pt"
    torch.save(_artifact_payload(), path)

    artifact = load_answer_utility_adapter_artifact(path)

    assert artifact.global_step == 500
    assert artifact.run_identity_sha256 == "1" * 64
    assert artifact.source_artifact_sha256 == "2" * 64
    assert artifact.experiment_config_sha256 == "3" * 64
    assert set(artifact.adapter_state) == {"weight"}


def test_private_artifact_loader_rejects_extra_field(tmp_path: Path) -> None:
    path = tmp_path / "answer_utility_adapter.pt"
    payload = _artifact_payload()
    payload["unexpected"] = True
    torch.save(payload, path)

    with pytest.raises(ValueError, match="exactly seven fields"):
        load_answer_utility_adapter_artifact(path)


def test_private_artifact_loader_rejects_state_digest_drift(tmp_path: Path) -> None:
    path = tmp_path / "answer_utility_adapter.pt"
    payload = _artifact_payload()
    payload["adapter_state_sha256"] = "4" * 64
    torch.save(payload, path)

    with pytest.raises(ValueError, match="state digest mismatch"):
        load_answer_utility_adapter_artifact(path)


def test_reader_question_appends_only_the_versioned_short_answer_instruction() -> None:
    original = "What color are the pants?"

    result = reader_question(original)

    assert result == f"{original}\n\n{INSTRUCT_READER_INSTRUCTION}"
    assert original == "What color are the pants?"


def test_reader_copy_does_not_modify_the_d_materialization_input() -> None:
    sample = _sample(0, answer="white")
    original, _truth_row = split_oracle_d_utility_sample(sample)

    reader = _reader_model_input(original)

    assert original.question == "question 0"
    assert reader.question == f"question 0\n\n{INSTRUCT_READER_INSTRUCTION}"
    assert reader.target == original.target
    assert reader.sample_content_sha256 == original.sample_content_sha256


def test_implementation_manifest_binds_local_scorer_and_production_oracle() -> None:
    manifest = _implementation_file_manifest()

    assert (
        "tgvf_rl/representation/experiments/answer_utility/evaluation/scoring.py"
        in manifest
    )
    assert "tgvf_rl/representation/training/oracle_d_utility.py" in manifest


def test_instruct_scorer_defers_verbose_choice_text_to_semantic_judge() -> None:
    score = score_instruct_generated_answer(
        "The pants appear white across both legs.\n\nExtra explanation.<|im_end|>",
        _truth("white", ("white", "gray", "blue", "black")),
    )

    assert score.correct is None
    assert score.route == "semantic_unresolved"
    assert score.candidate_choice_label is None


def test_instruct_scorer_treats_a_glove_as_text_not_choice_a() -> None:
    score = score_instruct_generated_answer(
        "The player holds a glove.\n\nAnswer: A glove<|im_end|>",
        _truth("glove", ("bat", "glove", "helmet", "ball")),
    )

    assert score.correct is True
    assert score.route == "explicit_answer_vqa_exact"
    assert score.expected_choice_label == "B"
    assert score.candidate_choice_label is None


def test_instruct_scorer_has_no_label_route_when_choices_were_not_shown() -> None:
    score = score_instruct_generated_answer(
        "Answer: B<|im_end|>",
        _truth("glove", ("bat", "glove", "helmet", "ball")),
    )

    assert score.correct is None
    assert score.route == "semantic_unresolved"
    assert score.candidate_choice_label is None


def test_instruct_scorer_defers_unclosed_text_at_length_cap() -> None:
    score = score_instruct_generated_answer(
        "The pants appear white across both legs, with",
        _truth("white", ("white", "gray", "blue", "black")),
        generation_stop_reason="length_cap",
    )

    assert score.correct is None
    assert score.route == "length_cap_unresolved"
    assert score.candidate_choice_label is None


def test_instruct_scorer_keeps_unresolved_length_cap_out_of_incorrect_count() -> None:
    score = score_instruct_generated_answer(
        "Based on the image, the object appears",
        _truth("glove", ("bat", "glove", "helmet", "ball")),
        generation_stop_reason="length_cap",
    )

    assert score.correct is None
    assert score.route == "length_cap_unresolved"
    assert score.candidate_choice_label is None


def test_instruct_scorer_defers_unstructured_wrong_choice_text() -> None:
    score = score_instruct_generated_answer(
        "The pants appear black.",
        _truth("white", ("white", "gray", "blue", "black")),
    )

    assert score.correct is None
    assert score.route == "semantic_unresolved"
    assert score.candidate_choice_label is None


def _generated(token_id: int) -> OracleGeneratedAnswer:
    return OracleGeneratedAnswer(
        token_ids=(token_id,),
        text=str(token_id),
        stop_reason="natural_stop",
    )


def test_pending_compatible_arms_use_one_batched_decoder_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = (object(), object())
    batched_calls: list[tuple[object, ...]] = []
    scalar_calls: list[object] = []

    def fake_batched(**kwargs: object) -> tuple[OracleGeneratedAnswer, ...]:
        observed = tuple(kwargs["contexts"])  # type: ignore[arg-type]
        batched_calls.append(observed)
        return (_generated(1), _generated(2))

    def fake_scalar(**kwargs: object) -> OracleGeneratedAnswer:
        scalar_calls.append(kwargs["context"])
        return _generated(9)

    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answers_batched",
        fake_batched,
    )
    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answer",
        fake_scalar,
    )

    outputs = _generate_pending_answers(
        contexts=contexts,  # type: ignore[arg-type]
        runtime=object(),
        family_adapter=object(),
        eos_token_ids=(5,),
        max_new_tokens=4,
        decode_mode="cached",
        arm_batch_size=2,
    )

    assert tuple(output.token_ids for output in outputs) == ((1,), (2,))
    assert batched_calls == [contexts]
    assert scalar_calls == []


def test_pending_single_resume_arm_falls_back_to_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object()
    batched_calls: list[tuple[object, ...]] = []
    scalar_calls: list[object] = []
    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answers_batched",
        lambda **kwargs: batched_calls.append(tuple(kwargs["contexts"])),
    )

    def fake_scalar(**kwargs: object) -> OracleGeneratedAnswer:
        scalar_calls.append(kwargs["context"])
        return _generated(3)

    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answer",
        fake_scalar,
    )

    outputs = _generate_pending_answers(
        contexts=(context,),  # type: ignore[arg-type]
        runtime=object(),
        family_adapter=object(),
        eos_token_ids=(5,),
        max_new_tokens=4,
        decode_mode="cached",
        arm_batch_size=2,
    )

    assert tuple(output.token_ids for output in outputs) == ((3,),)
    assert batched_calls == []
    assert scalar_calls == [context]


@pytest.mark.parametrize("decode_mode", ("cached", "no_cache"))
def test_pending_arm_batch_falls_back_for_incompatibility_or_no_cache(
    monkeypatch: pytest.MonkeyPatch,
    decode_mode: str,
) -> None:
    contexts = (object(), object())
    batched_call_count = 0
    scalar_calls: list[object] = []

    def fake_batched(**_kwargs: object) -> tuple[OracleGeneratedAnswer, ...]:
        nonlocal batched_call_count
        batched_call_count += 1
        raise OracleBatchCompatibilityError("fixture incompatibility")

    def fake_scalar(**kwargs: object) -> OracleGeneratedAnswer:
        scalar_calls.append(kwargs["context"])
        return _generated(4)

    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answers_batched",
        fake_batched,
    )
    monkeypatch.setattr(
        evaluation_runner_module,
        "greedy_oracle_answer",
        fake_scalar,
    )

    outputs = _generate_pending_answers(
        contexts=contexts,  # type: ignore[arg-type]
        runtime=object(),
        family_adapter=object(),
        eos_token_ids=(5,),
        max_new_tokens=4,
        decode_mode=decode_mode,  # type: ignore[arg-type]
        arm_batch_size=2,
    )

    assert tuple(output.token_ids for output in outputs) == ((4,), (4,))
    assert batched_call_count == (1 if decode_mode == "cached" else 0)
    assert scalar_calls == list(contexts)
