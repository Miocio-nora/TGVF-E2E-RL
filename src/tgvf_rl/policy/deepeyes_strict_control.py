"""Fail-closed source controls for the strict DeepEyes comparison pilots.

The local mixed-v2 population deliberately keeps its original source names.
This module is the only place where those names are projected onto the three
public DeepEyes reward/tool families.  Keeping the projection executable and
content-addressed prevents an ``open`` ThinkLite row from silently falling
through to the visual/Crop path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from tgvf_rl.data.deepeyes47k import DeepEyesTaskKind
from tgvf_rl.protocol import NativeAssistantDialect, NativeToolCapabilityProfile
from tgvf_rl.protocol.tool_prompts import (
    direct_answer_prompt_identity,
    visual_tool_prompt_identity,
)
from tgvf_rl.rewards.schema import (
    PILOT_REWARD_EQUATION_DEEPEYES_MATH,
    PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
)


DEEPEYES_STRICT_CONTROL_SCHEMA = "tgvf.deepeyes-strict-control.v1"
DEEPEYES_STRICT_TRAJECTORY_AUDIT_RETENTION = "all"
DEEPEYES_STRICT_EXPECTED_TRAJECTORIES_PER_STEP = 4096

# These aliases are evidence-backed by the public DeepEyes training snapshot:
# V* keeps ``vstar``; local ArxivQA corresponds to released ``chart``; and the
# complete released no-tool math source is ``thinklite_eureka``.
DEEPEYES_STRICT_SOURCE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "vstar": "vstar",
        "arxivqa": "chart",
        "thinklite": "thinklite_eureka",
    }
)

_EXPECTED_TASK_KINDS: Mapping[str, frozenset[DeepEyesTaskKind]] = MappingProxyType(
    {
        "vstar": frozenset({DeepEyesTaskKind.OPEN}),
        "arxivqa": frozenset({DeepEyesTaskKind.MCQ}),
        # The corrected local pool is not the public ThinkLite pool: it has
        # open/MCQ rows as well as math rows.  All three are named explicitly so
        # none can inherit a visual/Crop default.
        "thinklite": frozenset(
            {DeepEyesTaskKind.MATH, DeepEyesTaskKind.OPEN, DeepEyesTaskKind.MCQ}
        ),
    }
)


class DeepEyesVisualAnswerVerifierMode(str, Enum):
    RULE_FIRST_QWEN25_72B = "rule_first_qwen25_72b"
    ALWAYS_QWEN25_72B = "always_qwen25_72b"


class DeepEyesSourceToolRoutingMode(str, Enum):
    UNIFORM_CROP = "uniform_crop"
    OFFICIAL_BY_SOURCE = "official_by_source"


@dataclass(frozen=True, slots=True)
class DeepEyesStrictSourceRoute:
    local_source: str
    official_source: str
    task_kind: DeepEyesTaskKind
    reward_equation: str
    tool_profile: NativeToolCapabilityProfile | None
    always_judge_answer: bool


@dataclass(frozen=True, slots=True)
class DeepEyesStrictControlBinding:
    """Two orthogonal DeepEyes controls used by PRL12-A and PRL12-B."""

    visual_answer_verifier: DeepEyesVisualAnswerVerifierMode
    source_tool_routing: DeepEyesSourceToolRoutingMode
    schema_version: str = DEEPEYES_STRICT_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(
            self.visual_answer_verifier, DeepEyesVisualAnswerVerifierMode
        ):
            raise TypeError("visual_answer_verifier mode is invalid")
        if not isinstance(self.source_tool_routing, DeepEyesSourceToolRoutingMode):
            raise TypeError("source_tool_routing mode is invalid")
        if self.schema_version != DEEPEYES_STRICT_CONTROL_SCHEMA:
            raise ValueError("DeepEyes strict-control schema differs")

    @classmethod
    def from_mapping(cls, value: object) -> "DeepEyesStrictControlBinding":
        if not isinstance(value, Mapping) or set(value) != {
            "visual_answer_verifier",
            "source_tool_routing",
            "trajectory_audit_retention",
            "expected_trajectories_per_step",
        }:
            raise ValueError("deepeyes_control fields differ")
        if (
            value["trajectory_audit_retention"]
            != DEEPEYES_STRICT_TRAJECTORY_AUDIT_RETENTION
            or value["expected_trajectories_per_step"]
            != DEEPEYES_STRICT_EXPECTED_TRAJECTORIES_PER_STEP
        ):
            raise ValueError("deepeyes_control trajectory-audit contract differs")
        try:
            verifier = DeepEyesVisualAnswerVerifierMode(value["visual_answer_verifier"])
            tool_routing = DeepEyesSourceToolRoutingMode(value["source_tool_routing"])
        except (TypeError, ValueError) as error:
            raise ValueError("deepeyes_control mode is unsupported") from error
        # These are intentionally separate single-axis pilots.  A combined AB
        # arm can be added only after the two causal controls are read.
        accepted = {
            (
                DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B,
                DeepEyesSourceToolRoutingMode.UNIFORM_CROP,
            ),
            (
                DeepEyesVisualAnswerVerifierMode.RULE_FIRST_QWEN25_72B,
                DeepEyesSourceToolRoutingMode.OFFICIAL_BY_SOURCE,
            ),
        }
        if (verifier, tool_routing) not in accepted:
            raise ValueError(
                "strict DeepEyes controls must be verifier-only PRL12-A or "
                "source/tool-routing-only PRL12-B"
            )
        return cls(verifier, tool_routing)

    def route(
        self, data_source: object, task_kind: object
    ) -> DeepEyesStrictSourceRoute:
        if not isinstance(data_source, str) or data_source not in (
            DEEPEYES_STRICT_SOURCE_ALIASES
        ):
            raise ValueError(
                "strict DeepEyes routing received an unsupported data_source: "
                f"{data_source!r}"
            )
        try:
            kind = (
                task_kind
                if isinstance(task_kind, DeepEyesTaskKind)
                else DeepEyesTaskKind(task_kind)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "strict DeepEyes routing received an unsupported task_kind: "
                f"{task_kind!r}"
            ) from error
        if kind not in _EXPECTED_TASK_KINDS[data_source]:
            raise ValueError(
                "strict DeepEyes source/task pairing is unsupported: "
                f"source={data_source!r} task_kind={kind.value!r}"
            )

        visual = data_source in {"vstar", "arxivqa"}
        if (
            self.source_tool_routing is DeepEyesSourceToolRoutingMode.OFFICIAL_BY_SOURCE
            and not visual
        ):
            tool_profile = None
        else:
            tool_profile = NativeToolCapabilityProfile.CROP_ONLY
        return DeepEyesStrictSourceRoute(
            local_source=data_source,
            official_source=DEEPEYES_STRICT_SOURCE_ALIASES[data_source],
            task_kind=kind,
            reward_equation=(
                PILOT_REWARD_EQUATION_DEEPEYES_VISUAL
                if visual
                else PILOT_REWARD_EQUATION_DEEPEYES_MATH
            ),
            tool_profile=tool_profile,
            always_judge_answer=(
                visual
                and self.visual_answer_verifier
                is DeepEyesVisualAnswerVerifierMode.ALWAYS_QWEN25_72B
            ),
        )

    def prompt_bundle_sha256(self, assistant_dialect: NativeAssistantDialect) -> str:
        if not isinstance(assistant_dialect, NativeAssistantDialect):
            raise TypeError("assistant_dialect is invalid")
        if self.source_tool_routing is DeepEyesSourceToolRoutingMode.UNIFORM_CROP:
            return visual_tool_prompt_identity(
                NativeToolCapabilityProfile.CROP_ONLY,
                assistant_dialect=assistant_dialect,
            ).bundle_sha256
        payload = {
            "schema": "tgvf.deepeyes-source-prompt-routing.v1",
            "routing_identity_sha256": self.identity_sha256,
            "visual_prompt_sha256": visual_tool_prompt_identity(
                NativeToolCapabilityProfile.CROP_ONLY,
                assistant_dialect=assistant_dialect,
            ).bundle_sha256,
            "thinklite_prompt_sha256": direct_answer_prompt_identity(
                assistant_dialect=assistant_dialect
            ).bundle_sha256,
        }
        return _sha256(payload)

    def prompt_sha256_for_source(
        self,
        data_source: object,
        task_kind: object,
        *,
        assistant_dialect: NativeAssistantDialect,
    ) -> str:
        route = self.route(data_source, task_kind)
        if route.tool_profile is None:
            return direct_answer_prompt_identity(
                assistant_dialect=assistant_dialect
            ).bundle_sha256
        return visual_tool_prompt_identity(
            route.tool_profile,
            assistant_dialect=assistant_dialect,
        ).bundle_sha256

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "visual_answer_verifier": self.visual_answer_verifier.value,
            "source_tool_routing": self.source_tool_routing.value,
            "source_aliases": dict(DEEPEYES_STRICT_SOURCE_ALIASES),
            "expected_task_kinds": {
                source: sorted(kind.value for kind in kinds)
                for source, kinds in _EXPECTED_TASK_KINDS.items()
            },
            "unknown_source_behavior": "fail_closed",
            "unknown_task_behavior": "fail_closed",
            "visual_reward_equation": PILOT_REWARD_EQUATION_DEEPEYES_VISUAL,
            "thinklite_reward_equation": PILOT_REWARD_EQUATION_DEEPEYES_MATH,
            "official_visual_tool_profile": "crop_only",
            "official_thinklite_tool_profile": "none",
            "trajectory_audit_retention": (DEEPEYES_STRICT_TRAJECTORY_AUDIT_RETENTION),
            "expected_trajectories_per_step": (
                DEEPEYES_STRICT_EXPECTED_TRAJECTORIES_PER_STEP
            ),
        }

    def as_config(self) -> dict[str, object]:
        return {
            "visual_answer_verifier": self.visual_answer_verifier.value,
            "source_tool_routing": self.source_tool_routing.value,
            "trajectory_audit_retention": (DEEPEYES_STRICT_TRAJECTORY_AUDIT_RETENTION),
            "expected_trajectories_per_step": (
                DEEPEYES_STRICT_EXPECTED_TRAJECTORIES_PER_STEP
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.as_record())


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DEEPEYES_STRICT_CONTROL_SCHEMA",
    "DEEPEYES_STRICT_EXPECTED_TRAJECTORIES_PER_STEP",
    "DEEPEYES_STRICT_SOURCE_ALIASES",
    "DEEPEYES_STRICT_TRAJECTORY_AUDIT_RETENTION",
    "DeepEyesSourceToolRoutingMode",
    "DeepEyesStrictControlBinding",
    "DeepEyesStrictSourceRoute",
    "DeepEyesVisualAnswerVerifierMode",
]
