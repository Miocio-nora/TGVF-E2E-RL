"""Fail-closed contract for Qwen3 prompts already expanded to visual tokens.

The policy pipeline uses one token coordinate system: every image placeholder
is already expanded to the exact number of merged visual features before the
prompt reaches vLLM.  Stock vLLM treats token-ID prompts as unexpanded and can
replace the first image token in an ``N``-token run with another ``N`` tokens,
silently producing ``2N-1`` positions.  This module binds the submitted token
IDs to their ordered visual ranges so the repo-owned processor can bypass that
replacement and prove that its processed evidence is identical.

The serialized contract travels in ``mm_processor_kwargs`` because that is the
only multimodal metadata surface preserved by the pinned veRL server manager.
The registered processor removes the reserved field before calling the HF
processor; it is never an image-processor option.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

from tgvf_rl.contracts.errors import (
    ContractUnsetError,
    IdentityMismatchError,
    ReplayMismatchError,
)


PREEXPANDED_PROMPT_SCHEMA = "tgvf-vllm-preexpanded-prompt-v1"
PREEXPANDED_PROMPT_CONTRACT_KWARG = "tgvf_preexpanded_prompt_contract"


def _prompt_token_ids(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a token-ID sequence")
    normalized = tuple(value)
    if not normalized or any(
        type(token_id) is not int or token_id < 0 for token_id in normalized
    ):
        raise ValueError(
            f"{field_name} must contain non-negative token IDs and be non-empty"
        )
    return normalized


def preexpanded_prompt_token_ids_sha256(prompt_token_ids: Sequence[int]) -> str:
    """Content-address one exact tokenized prompt."""

    prompt = _prompt_token_ids(prompt_token_ids, "prompt_token_ids")
    payload = json.dumps(
        {
            "schema": "tgvf-tokenized-prompt-v1",
            "prompt_token_ids": prompt,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class VLLMVisualPlaceholderRange:
    """One ordered, already-expanded visual item in token coordinates."""

    modality: str
    item_index: int
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.modality != "image":
            raise ValueError("Policy Pilot pre-expanded prompts support image only")
        if type(self.item_index) is not int or self.item_index < 0:
            raise ValueError("visual placeholder item_index must be non-negative")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("visual placeholder offset must be non-negative")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("visual placeholder length must be positive")

    def as_payload(self) -> dict[str, object]:
        return {
            "modality": self.modality,
            "item_index": self.item_index,
            "offset": self.offset,
            "length": self.length,
        }

    @classmethod
    def from_payload(cls, value: object) -> "VLLMVisualPlaceholderRange":
        if not isinstance(value, Mapping):
            raise TypeError("visual placeholder range payload must be a mapping")
        if set(value) != {"modality", "item_index", "offset", "length"}:
            raise ValueError("visual placeholder range payload fields differ")
        return cls(
            modality=value["modality"],  # type: ignore[arg-type]
            item_index=value["item_index"],  # type: ignore[arg-type]
            offset=value["offset"],  # type: ignore[arg-type]
            length=value["length"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VLLMPreExpandedPromptContract:
    """Submitted prompt identity plus ordered vLLM visual-item coordinates."""

    prompt_token_ids_sha256: str
    prompt_token_count: int
    image_token_id: int
    ordered_visual_placeholder_ranges: tuple[VLLMVisualPlaceholderRange, ...]
    schema: str = PREEXPANDED_PROMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREEXPANDED_PROMPT_SCHEMA:
            raise IdentityMismatchError("pre-expanded prompt schema differs")
        if len(self.prompt_token_ids_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.prompt_token_ids_sha256
        ):
            raise ValueError("pre-expanded prompt identity must be lowercase SHA256")
        if type(self.prompt_token_count) is not int or self.prompt_token_count <= 0:
            raise ValueError("pre-expanded prompt token count must be positive")
        if type(self.image_token_id) is not int or self.image_token_id < 0:
            raise ValueError("image_token_id must be a non-negative integer")
        ranges = self.ordered_visual_placeholder_ranges
        if not isinstance(ranges, tuple) or not ranges:
            raise ValueError("pre-expanded prompt requires at least one visual range")
        previous_end = -1
        for expected_index, placeholder in enumerate(ranges):
            if not isinstance(placeholder, VLLMVisualPlaceholderRange):
                raise TypeError("visual ranges must be VLLMVisualPlaceholderRange")
            if placeholder.item_index != expected_index:
                raise ValueError("visual placeholder item indices must be contiguous")
            if placeholder.offset < previous_end:
                raise ValueError(
                    "visual placeholder ranges must be ordered and disjoint"
                )
            if placeholder.offset + placeholder.length > self.prompt_token_count:
                raise ValueError("visual placeholder range exceeds the prompt")
            previous_end = placeholder.offset + placeholder.length

    @classmethod
    def from_prompt(
        cls,
        prompt_token_ids: Sequence[int],
        *,
        image_token_id: int,
        expected_image_items: int,
    ) -> "VLLMPreExpandedPromptContract":
        prompt = _prompt_token_ids(prompt_token_ids, "pre-expanded prompt_token_ids")
        if type(image_token_id) is not int or image_token_id < 0:
            raise ValueError("image_token_id must be a non-negative integer")
        if type(expected_image_items) is not int or expected_image_items <= 0:
            raise ValueError("expected_image_items must be a positive integer")
        runs = _maximal_token_runs(prompt, token_id=image_token_id)
        if len(runs) != expected_image_items:
            raise ReplayMismatchError(
                "pre-expanded prompt visual-run count differs from image-item count"
            )
        return cls(
            prompt_token_ids_sha256=preexpanded_prompt_token_ids_sha256(prompt),
            prompt_token_count=len(prompt),
            image_token_id=image_token_id,
            ordered_visual_placeholder_ranges=tuple(
                VLLMVisualPlaceholderRange("image", index, offset, length)
                for index, (offset, length) in enumerate(runs)
            ),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "prompt_token_count": self.prompt_token_count,
            "image_token_id": self.image_token_id,
            "ordered_visual_placeholder_ranges": [
                item.as_payload() for item in self.ordered_visual_placeholder_ranges
            ],
        }

    @classmethod
    def from_payload(cls, value: object) -> "VLLMPreExpandedPromptContract":
        if not isinstance(value, Mapping):
            raise TypeError("pre-expanded prompt contract must be a mapping")
        expected = {
            "schema",
            "prompt_token_ids_sha256",
            "prompt_token_count",
            "image_token_id",
            "ordered_visual_placeholder_ranges",
        }
        if set(value) != expected:
            raise ValueError("pre-expanded prompt contract fields differ")
        raw_ranges = value["ordered_visual_placeholder_ranges"]
        if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
            raise TypeError("ordered visual placeholder ranges must be a sequence")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],  # type: ignore[arg-type]
            prompt_token_count=value["prompt_token_count"],  # type: ignore[arg-type]
            image_token_id=value["image_token_id"],  # type: ignore[arg-type]
            ordered_visual_placeholder_ranges=tuple(
                VLLMVisualPlaceholderRange.from_payload(item) for item in raw_ranges
            ),
        )

    def validate_submitted_prompt(
        self,
        prompt_token_ids: Sequence[int],
        *,
        expected_image_items: int,
    ) -> None:
        prompt = _prompt_token_ids(prompt_token_ids, "submitted prompt_token_ids")
        if len(prompt) != self.prompt_token_count:
            raise ReplayMismatchError(
                "submitted pre-expanded prompt length differs from its contract"
            )
        if preexpanded_prompt_token_ids_sha256(prompt) != self.prompt_token_ids_sha256:
            raise IdentityMismatchError(
                "submitted pre-expanded prompt hash differs from its contract"
            )
        actual = type(self).from_prompt(
            prompt,
            image_token_id=self.image_token_id,
            expected_image_items=expected_image_items,
        )
        if (
            actual.ordered_visual_placeholder_ranges
            != self.ordered_visual_placeholder_ranges
        ):
            raise ReplayMismatchError(
                "submitted ordered visual placeholder ranges differ from contract"
            )

    def validate_processed_prompt(
        self,
        prompt_token_ids: Sequence[int],
        mm_placeholders: object,
    ) -> None:
        """Require processor output to retain the submitted coordinate exactly."""

        self.validate_submitted_prompt(
            prompt_token_ids,
            expected_image_items=len(self.ordered_visual_placeholder_ranges),
        )
        if not isinstance(mm_placeholders, Mapping):
            raise TypeError("processed multimodal placeholders must be a mapping")
        if set(mm_placeholders) != {"image"}:
            raise ReplayMismatchError(
                "processed placeholders must contain exactly the image modality"
            )
        raw_ranges = mm_placeholders["image"]
        if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
            raise TypeError("processed image placeholders must be a sequence")
        processed = tuple(
            _processed_placeholder_range(item, item_index=index)
            for index, item in enumerate(raw_ranges)
        )
        if processed != self.ordered_visual_placeholder_ranges:
            raise ReplayMismatchError(
                "processed ordered visual placeholder ranges differ from submitted prompt"
            )


def bind_preexpanded_prompt_contract(
    mm_processor_kwargs: Mapping[str, object] | None,
    *,
    prompt_token_ids: Sequence[int],
    image_token_id: int,
    expected_image_items: int,
) -> dict[str, object]:
    """Add the reserved contract once; silently replacing one is forbidden."""

    result = dict(mm_processor_kwargs or {})
    if PREEXPANDED_PROMPT_CONTRACT_KWARG in result:
        raise ReplayMismatchError("pre-expanded prompt contract is already bound")
    contract = VLLMPreExpandedPromptContract.from_prompt(
        prompt_token_ids,
        image_token_id=image_token_id,
        expected_image_items=expected_image_items,
    )
    result[PREEXPANDED_PROMPT_CONTRACT_KWARG] = contract.as_payload()
    return result


def rebind_preexpanded_prompt_contract(
    mm_processor_kwargs: Mapping[str, object] | None,
    *,
    prompt_token_ids: Sequence[int],
    expected_image_items: int,
) -> dict[str, object]:
    """Advance an already-bound contract to the next exact native turn."""

    previous, clean = split_preexpanded_prompt_contract(mm_processor_kwargs)
    return bind_preexpanded_prompt_contract(
        clean,
        prompt_token_ids=prompt_token_ids,
        image_token_id=previous.image_token_id,
        expected_image_items=expected_image_items,
    )


def require_preexpanded_prompt_contract(
    mm_processor_kwargs: Mapping[str, object] | None,
    *,
    prompt_token_ids: Sequence[int],
    expected_image_items: int,
) -> VLLMPreExpandedPromptContract:
    contract, _ = split_preexpanded_prompt_contract(mm_processor_kwargs)
    contract.validate_submitted_prompt(
        prompt_token_ids,
        expected_image_items=expected_image_items,
    )
    return contract


def split_preexpanded_prompt_contract(
    mm_processor_kwargs: Mapping[str, object] | None,
) -> tuple[VLLMPreExpandedPromptContract, dict[str, object]]:
    if not isinstance(mm_processor_kwargs, Mapping):
        raise ContractUnsetError(
            "vLLM request is missing the pre-expanded prompt contract"
        )
    if PREEXPANDED_PROMPT_CONTRACT_KWARG not in mm_processor_kwargs:
        raise ContractUnsetError(
            "vLLM request is missing the pre-expanded prompt contract"
        )
    clean = dict(mm_processor_kwargs)
    raw = clean.pop(PREEXPANDED_PROMPT_CONTRACT_KWARG)
    return VLLMPreExpandedPromptContract.from_payload(raw), clean


def _maximal_token_runs(
    prompt_token_ids: tuple[int, ...], *, token_id: int
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(prompt_token_ids):
        if prompt_token_ids[cursor] != token_id:
            cursor += 1
            continue
        start = cursor
        while cursor < len(prompt_token_ids) and prompt_token_ids[cursor] == token_id:
            cursor += 1
        runs.append((start, cursor - start))
    return tuple(runs)


def _processed_placeholder_range(
    value: object, *, item_index: int
) -> VLLMVisualPlaceholderRange:
    if isinstance(value, Mapping):
        offset = value.get("offset")
        length = value.get("length")
    else:
        offset = getattr(value, "offset", None)
        length = getattr(value, "length", None)
    if type(offset) is not int or type(length) is not int:
        raise TypeError("processed placeholder must expose integer offset/length")
    return VLLMVisualPlaceholderRange("image", item_index, offset, length)


__all__ = [
    "PREEXPANDED_PROMPT_CONTRACT_KWARG",
    "PREEXPANDED_PROMPT_SCHEMA",
    "VLLMPreExpandedPromptContract",
    "VLLMVisualPlaceholderRange",
    "bind_preexpanded_prompt_contract",
    "preexpanded_prompt_token_ids_sha256",
    "rebind_preexpanded_prompt_contract",
    "require_preexpanded_prompt_contract",
    "split_preexpanded_prompt_contract",
]
