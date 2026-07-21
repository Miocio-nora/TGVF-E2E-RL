from __future__ import annotations

from types import SimpleNamespace

import pytest

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)
from tgvf_rl.framework.vllm import (
    PREEXPANDED_PROMPT_CONTRACT_KWARG,
    VLLMPreExpandedPromptContract,
    VLLMVisualPlaceholderRange,
    bind_preexpanded_prompt_contract,
    rebind_preexpanded_prompt_contract,
    require_preexpanded_prompt_contract,
    split_preexpanded_prompt_contract,
)


IMAGE = 99


def test_n_greater_than_one_retains_n_instead_of_two_n_minus_one() -> None:
    prompt = (10, IMAGE, IMAGE, IMAGE, 11)
    kwargs = bind_preexpanded_prompt_contract(
        {"max_pixels": 262_144},
        prompt_token_ids=prompt,
        image_token_id=IMAGE,
        expected_image_items=1,
    )
    contract = require_preexpanded_prompt_contract(
        kwargs,
        prompt_token_ids=prompt,
        expected_image_items=1,
    )

    contract.validate_processed_prompt(
        prompt,
        {"image": [SimpleNamespace(offset=1, length=3)]},
    )
    assert contract.ordered_visual_placeholder_ranges == (
        VLLMVisualPlaceholderRange("image", 0, 1, 3),
    )

    # This is the exact stock token-replacement failure mode: replacing the
    # first token in an N=3 run with another three creates 2N-1=5 positions.
    twice_expanded = (10, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 11)
    with pytest.raises(ReplayMismatchError, match="length differs"):
        contract.validate_processed_prompt(
            twice_expanded,
            {"image": [SimpleNamespace(offset=1, length=5)]},
        )


def test_multiple_visual_items_preserve_order_and_distinct_lengths() -> None:
    prompt = (
        1,
        IMAGE,
        IMAGE,
        IMAGE,
        2,
        3,
        IMAGE,
        IMAGE,
        4,
    )
    contract = VLLMPreExpandedPromptContract.from_prompt(
        prompt,
        image_token_id=IMAGE,
        expected_image_items=2,
    )
    expected = (
        VLLMVisualPlaceholderRange("image", 0, 1, 3),
        VLLMVisualPlaceholderRange("image", 1, 6, 2),
    )
    assert contract.ordered_visual_placeholder_ranges == expected
    contract.validate_processed_prompt(
        prompt,
        {
            "image": [
                {"offset": 1, "length": 3},
                {"offset": 6, "length": 2},
            ]
        },
    )

    with pytest.raises(ReplayMismatchError, match="ordered visual"):
        contract.validate_processed_prompt(
            prompt,
            {
                "image": [
                    {"offset": 6, "length": 2},
                    {"offset": 1, "length": 3},
                ]
            },
        )


def test_contract_rejects_hash_range_item_count_and_modality_mismatch() -> None:
    prompt = (1, IMAGE, IMAGE, 2)
    kwargs = bind_preexpanded_prompt_contract(
        None,
        prompt_token_ids=prompt,
        image_token_id=IMAGE,
        expected_image_items=1,
    )
    contract, clean = split_preexpanded_prompt_contract(kwargs)
    assert clean == {}

    changed_same_length = (1, IMAGE, IMAGE, 3)
    with pytest.raises(IdentityMismatchError, match="hash differs"):
        contract.validate_submitted_prompt(
            changed_same_length,
            expected_image_items=1,
        )
    with pytest.raises(ReplayMismatchError, match="run count"):
        contract.validate_submitted_prompt(prompt, expected_image_items=2)
    with pytest.raises(ReplayMismatchError, match="exactly the image"):
        contract.validate_processed_prompt(
            prompt,
            {
                "image": [{"offset": 1, "length": 2}],
                "video": [],
            },
        )

    with pytest.raises(ContractUnsetError, match="missing"):
        split_preexpanded_prompt_contract({"max_pixels": 262_144})
    with pytest.raises(ReplayMismatchError, match="already bound"):
        bind_preexpanded_prompt_contract(
            kwargs,
            prompt_token_ids=prompt,
            image_token_id=IMAGE,
            expected_image_items=1,
        )


def test_rebind_advances_hash_ranges_and_preserves_non_contract_kwargs() -> None:
    first = (1, IMAGE, IMAGE, 2)
    first_kwargs = bind_preexpanded_prompt_contract(
        {"max_pixels": 262_144},
        prompt_token_ids=first,
        image_token_id=IMAGE,
        expected_image_items=1,
    )
    second = first + (3, IMAGE, IMAGE, IMAGE, 4)
    second_kwargs = rebind_preexpanded_prompt_contract(
        first_kwargs,
        prompt_token_ids=second,
        expected_image_items=2,
    )
    first_contract, _ = split_preexpanded_prompt_contract(first_kwargs)
    second_contract, clean = split_preexpanded_prompt_contract(second_kwargs)

    assert clean == {"max_pixels": 262_144}
    assert second_contract.image_token_id == IMAGE
    assert second_contract.prompt_token_ids_sha256 != (
        first_contract.prompt_token_ids_sha256
    )
    assert second_contract.ordered_visual_placeholder_ranges == (
        VLLMVisualPlaceholderRange("image", 0, 1, 2),
        VLLMVisualPlaceholderRange("image", 1, 5, 3),
    )
    assert PREEXPANDED_PROMPT_CONTRACT_KWARG in second_kwargs


def test_registered_qwen3_processor_bypasses_stock_token_replacement() -> None:
    pytest.importorskip("vllm")
    from vllm.multimodal.inputs import PlaceholderRange

    from tgvf_rl.framework.vllm.qwen3_plugin import (
        TGVFQwen3VLMultiModalProcessor,
    )

    class _Items:
        @staticmethod
        def get_all_counts():
            return {"image": 1}

    class _Placeholder:
        @staticmethod
        def to_range():
            return PlaceholderRange(offset=1, length=3)

    processor = object.__new__(TGVFQwen3VLMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **_: SimpleNamespace(image_token_id=IMAGE)
    )
    processor._to_mm_items = lambda _: _Items()
    processor._cached_apply_hf_processor = lambda prompt, _items, _kwargs, **_: (
        prompt,
        SimpleNamespace(
            kwargs={"image": "processed"},
            hashes={"image": ["hash"]},
            prompt_updates=object(),
        ),
        False,
    )
    processor._find_mm_placeholders = lambda _ids, _updates: {"image": [_Placeholder()]}
    processor._validate_mm_placeholders = lambda _ranges, _counts: None
    # If apply() regresses to the stock token-prompt path, this sentinel makes
    # the characteristic first-token replacement observable immediately.
    processor._maybe_apply_prompt_updates = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("stock prompt update must not run"))

    prompt = [10, IMAGE, IMAGE, IMAGE, 11]
    kwargs = bind_preexpanded_prompt_contract(
        {},
        prompt_token_ids=prompt,
        image_token_id=IMAGE,
        expected_image_items=1,
    )
    processed = processor.apply(prompt, {"image": [object()]}, kwargs)

    assert processed["prompt_token_ids"] == prompt
    assert len(processed["prompt_token_ids"]) == 5
    assert processed["mm_placeholders"]["image"] == [
        PlaceholderRange(offset=1, length=3)
    ]

    with pytest.raises(ContractUnsetError, match="missing"):
        processor.apply(prompt, {"image": [object()]}, {})

    class _TwiceExpandedPlaceholder:
        @staticmethod
        def to_range():
            return PlaceholderRange(offset=1, length=5)

    processor._find_mm_placeholders = lambda _ids, _updates: {
        "image": [_TwiceExpandedPlaceholder()]
    }
    with pytest.raises(ReplayMismatchError, match="ordered visual"):
        processor.apply(prompt, {"image": [object()]}, kwargs)
