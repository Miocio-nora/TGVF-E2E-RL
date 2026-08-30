"""Immutable schema types and identity constants for T1 run configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tgvf_rl.public_api_compat import rebind_public_class

from .policy_selection import SelectionSource
from .policy_selection_config_values import (
    _sha256_json,
    derive_t1_attempt_seed,
)

T1_RUN_CONFIG_SCHEMA = "tgvf.policy-selection.t1-run.v1"
T1_PROMPT_SCHEMA = "qwen-native-user-image-question-v1"
T1_SOURCE_RGB_SCHEMA = "tgvf.policy-selection.source-rgb-pixels.v1"
T1_PROMPT_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-native-prompt.v1"
T1_MODEL_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-model-identity.v1"
T1_PROCESSOR_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-processor-identity.v1"
T1_RUNTIME_IDENTITY_SCHEMA = "tgvf.policy-selection.t1-runtime-identity.v1"
T1_THINKING_ANSWER_PARSER = "last-think-suffix-v1"
T1_INSTRUCT_ANSWER_PARSER = "direct-completion-v1"
T1_MAX_PIXELS = 512 * 512
T1_MODEL_PATH_BY_REPOSITORY = MappingProxyType(
    {
        "Qwen/Qwen3-VL-8B-Thinking": (
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
        ),
        "Qwen/Qwen3-VL-8B-Instruct": (
            "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class T1ResponseBudget:
    revision: int
    max_model_len: int
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class T1DataSource:
    source: SelectionSource
    path: Path
    sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class T1RunConfig:
    """Validated immutable identity of one T1 generation run."""

    run_id: str
    manifest_sha256: str
    model: Mapping[str, Any]
    prompt: Mapping[str, Any]
    image: Mapping[str, Any]
    sampling: Mapping[str, Any]
    response_budgets: tuple[T1ResponseBudget, ...]
    runtime: Mapping[str, Any]
    data_sources: tuple[T1DataSource, ...]
    selection: Mapping[str, Any]
    verifier: Mapping[str, Any]
    output_root: Path
    _record_bytes: bytes

    def as_record(self) -> dict[str, Any]:
        return json.loads(self._record_bytes)

    @property
    def model_identity_sha256(self) -> str:
        return _sha256_json(
            {"schema": T1_MODEL_IDENTITY_SCHEMA, "model": dict(self.model)}
        )

    @property
    def processor_identity_sha256(self) -> str:
        return _sha256_json(
            {
                "schema": T1_PROCESSOR_IDENTITY_SCHEMA,
                "preprocessor_config_sha256": self.model["preprocessor_config_sha256"],
                "tokenizer_config_sha256": self.model["tokenizer_config_sha256"],
                "tokenizer_json_sha256": self.model["tokenizer_json_sha256"],
                "chat_template_sha256": self.model["chat_template_sha256"],
                "tokenizer_length": self.model["tokenizer_length"],
                "eos_token_id": self.model["eos_token_id"],
                "prompt": dict(self.prompt),
                "image": dict(self.image),
            }
        )

    @property
    def runtime_identity_sha256(self) -> str:
        return _sha256_json(
            {"schema": T1_RUNTIME_IDENTITY_SCHEMA, "runtime": dict(self.runtime)}
        )

    def attempt_seed(self, *, candidate_sha256: str, attempt_index: int) -> int:
        return derive_t1_attempt_seed(
            run_manifest_sha256=self.manifest_sha256,
            candidate_sha256=candidate_sha256,
            attempt_index=attempt_index,
            seed_root=self.sampling["seed_root"],
            seed_namespace=self.sampling["seed_namespace"],
        )

    def budget(self, revision: int) -> T1ResponseBudget:
        for budget in self.response_budgets:
            if budget.revision == revision:
                return budget
        raise ValueError(f"unknown response budget revision {revision}")


_PUBLIC_RUNTIME_MODULE = "tgvf_rl.data.policy_selection_runtime"
for _public_type in (T1ResponseBudget, T1DataSource, T1RunConfig):
    rebind_public_class(
        _public_type,
        implementation_module=__name__,
        public_module=_PUBLIC_RUNTIME_MODULE,
    )
del _public_type

__all__ = [
    "T1_INSTRUCT_ANSWER_PARSER",
    "T1_MAX_PIXELS",
    "T1_MODEL_PATH_BY_REPOSITORY",
    "T1_PROMPT_SCHEMA",
    "T1_RUN_CONFIG_SCHEMA",
    "T1_SOURCE_RGB_SCHEMA",
    "T1_THINKING_ANSWER_PARSER",
    "T1DataSource",
    "T1ResponseBudget",
    "T1RunConfig",
]
