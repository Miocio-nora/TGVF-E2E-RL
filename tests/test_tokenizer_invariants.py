from __future__ import annotations

from tgvf_rl.tokenizer_invariants import effective_tokenizer_length


class _Backend:
    def __init__(self, base_size: int, added_ids: tuple[int, ...]) -> None:
        self.base_size = base_size
        self.added_ids = added_ids

    def get_vocab_size(self, *, with_added_tokens: bool) -> int:
        if with_added_tokens:
            raise AssertionError("the merged vocabulary path must not be used")
        return self.base_size

    def get_added_tokens_decoder(self) -> dict[int, str]:
        return {token_id: f"token-{token_id}" for token_id in self.added_ids}


class _FastTokenizer:
    def __init__(self, base_size: int, added_ids: tuple[int, ...]) -> None:
        self.backend_tokenizer = _Backend(base_size, added_ids)

    def __len__(self) -> int:
        raise AssertionError("the slow tokenizer length path must not be used")


class _FallbackTokenizer:
    def __init__(self, length: int) -> None:
        self.length = length
        self.calls = 0

    def __len__(self) -> int:
        self.calls += 1
        return self.length


def test_contiguous_added_token_suffix_uses_exact_fast_length() -> None:
    tokenizer = _FastTokenizer(10, (10, 11, 12))

    assert effective_tokenizer_length(tokenizer) == 13

    tokenizer.backend_tokenizer.added_ids = (10, 11, 12, 13)
    assert effective_tokenizer_length(tokenizer) == 14


def test_ambiguous_added_token_layout_falls_back_to_native_length() -> None:
    tokenizer = _FallbackTokenizer(13)
    tokenizer.backend_tokenizer = _Backend(10, (10, 12))

    assert effective_tokenizer_length(tokenizer) == 13
    assert tokenizer.calls == 1


def test_tokenizer_without_fast_backend_uses_native_length() -> None:
    tokenizer = _FallbackTokenizer(17)

    assert effective_tokenizer_length(tokenizer) == 17
    assert tokenizer.calls == 1
