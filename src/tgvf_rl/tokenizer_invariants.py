"""Cheap, fail-closed tokenizer vocabulary invariants.

Hugging Face fast-tokenizer ``__len__`` asks the Rust backend to materialize
the complete vocabulary.  Qwen's 151k vocabulary makes that operation
surprisingly expensive even though representation training only needs to
prove that the tokenizer has not grown.  The fast path below derives the same
cardinality from the base-vocabulary count and a proven contiguous added-token
ID range.  Tokenizers that do not expose that proof retain their native
``len`` behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def effective_tokenizer_length(tokenizer: Any) -> int:
    """Return ``len(tokenizer)`` with an exact fast-tokenizer shortcut.

    The shortcut is used only when the backend reports a non-negative base
    vocabulary size and every added token occupies the exact contiguous suffix
    ``[base_size, base_size + added_count)``.  That condition proves the full
    cardinality without requesting the backend's expensive merged vocabulary.
    Any unfamiliar or ambiguous tokenizer falls back to its own ``__len__``.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    get_vocab_size = getattr(backend, "get_vocab_size", None)
    get_added_tokens_decoder = getattr(backend, "get_added_tokens_decoder", None)
    if callable(get_vocab_size) and callable(get_added_tokens_decoder):
        try:
            base_size = get_vocab_size(with_added_tokens=False)
            added_decoder = get_added_tokens_decoder()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        else:
            if (
                isinstance(base_size, int)
                and not isinstance(base_size, bool)
                and base_size >= 0
                and isinstance(added_decoder, Mapping)
            ):
                added_ids = tuple(added_decoder)
                if all(
                    isinstance(token_id, int) and not isinstance(token_id, bool)
                    for token_id in added_ids
                ) and set(added_ids) == set(
                    range(base_size, base_size + len(added_ids))
                ):
                    return base_size + len(added_ids)

    return len(tokenizer)
