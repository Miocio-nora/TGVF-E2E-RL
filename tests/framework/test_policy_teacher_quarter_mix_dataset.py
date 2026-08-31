from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgvf_rl.framework.verl import policy_teacher_quarter_mix_dataset as module
from tgvf_rl.framework.verl.policy_teacher_quarter_mix_dataset import (
    POLICY_TEACHER_QUARTER_MIX_AGENT_NAME,
    POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME,
    PolicyTeacherQuarterMixDataset,
    PolicyTeacherQuarterMixDatasetBinding,
)
from tgvf_rl.framework.verl.deepeyes_official_dataset import (
    DEEPEYES_PROBE_SENTINEL,
    DEEPEYES_SMOKE_SENTINEL,
    DEEPEYES_TRAIN_SENTINEL,
)
from tgvf_rl.policy.crop_tgvf_deepeyes_matched_protocol import (
    CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.deepeyes_official_protocol import (
    THINKLITE_PROMPT_IDENTITY,
    VISUAL_PROMPT_IDENTITY,
    tools_kwargs_for_source,
)
from tgvf_rl.policy.tgvf_deepeyes_matched_protocol import (
    TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY,
)
from tgvf_rl.policy.tgvf_target_guide_v2_protocol import (
    TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY,
    TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT,
)
from tgvf_rl.policy.no_tool_rl_protocol import NO_TOOL_RL_PROMPT_IDENTITY
from tgvf_rl.protocol import NativeToolCapabilityProfile


class _FakeTokenizer:
    image_token = "<|image_pad|>"
    image_token_id = 90_000

    def __len__(self) -> int:
        return module.POLICY_PILOT_V1_TOKENIZER_LENGTH

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.image_token, cursor):
                result.append(self.image_token_id)
                cursor += len(self.image_token)
            else:
                result.extend(byte + 1 for byte in text[cursor].encode("utf-8"))
                cursor += 1
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == self.image_token
        return self.image_token_id


class _FakeInstructProcessor:
    chat_template = "fake-qwen3-vl-instruct-native-template"

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 3136, "longest_edge": 16_777_216},
            merge_size=2,
        )

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tools == []
        assert tokenize is False
        assert add_generation_prompt is True
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<|im_start|>{message['role']}\n")
            content = message["content"]
            if isinstance(content, str):
                chunks.append(content)
            else:
                for item in content:
                    if item["type"] == "image":
                        chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
                    else:
                        chunks.append(item["text"])
            chunks.append("<|im_end|>\n")
        chunks.append("<|im_start|>assistant\n")
        return "".join(chunks)

    def __call__(
        self,
        *,
        text,
        images,
        padding,
        return_tensors,
        images_kwargs,
    ):
        assert len(text) == len(images) == 1
        assert images[0].mode == "RGB"
        assert padding is False
        assert return_tensors == "pt"
        canonical = self.tokenizer.encode(text[0], add_special_tokens=False)
        position = canonical.index(self.tokenizer.image_token_id)
        expanded = (
            canonical[:position]
            + [self.tokenizer.image_token_id] * 4
            + canonical[position + 1 :]
        )
        return {
            "input_ids": torch.tensor((expanded,), dtype=torch.long),
            "image_grid_thw": torch.tensor(((1, 4, 4),), dtype=torch.long),
        }


def _binding(
    root: Path,
    *,
    profile: NativeToolCapabilityProfile,
    prompt_sha256: str,
    template_sha256: str,
) -> PolicyTeacherQuarterMixDatasetBinding:
    return PolicyTeacherQuarterMixDatasetBinding(
        root=root,
        manifest_file_sha256="1" * 64,
        content_sha256="2" * 64,
        samples_sha256="3" * 64,
        iteration_identity_sha256="4" * 64,
        schedule_seed=42,
        expected_sample_count=20_480,
        tool_profile=profile,
        visual_prompt_bundle_sha256=prompt_sha256,
        thinklite_prompt_bundle_sha256=THINKLITE_PROMPT_IDENTITY.bundle_sha256,
        tokenizer_length=module.POLICY_PILOT_V1_TOKENIZER_LENGTH,
        model_name=module.POLICY_PILOT_V1_MODEL_NAME,
        chat_template_sha256=template_sha256,
    )


@pytest.mark.parametrize(
    ("profile", "prompt_sha256"),
    (
        (
            NativeToolCapabilityProfile.NO_TOOL,
            NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
        ),
        (
            NativeToolCapabilityProfile.CROP_ONLY,
            VISUAL_PROMPT_IDENTITY.bundle_sha256,
        ),
        (
            NativeToolCapabilityProfile.TGVF_ONLY,
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        ),
        (
            NativeToolCapabilityProfile.TGVF_ONLY,
            TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
        ),
        (
            NativeToolCapabilityProfile.CROP_TGVF,
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
        ),
    ),
)
def test_one_schedule_dispatches_without_changing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: NativeToolCapabilityProfile,
    prompt_sha256: str,
) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "source.png"
    image_module.new("RGB", (16, 8), color=(12, 34, 56)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = tuple(
        SimpleNamespace(
            sample_id=f"teacher-quarter:{index}",
            image_path=image_path,
            image_sha256=image_sha256,
            question=f"question-{source}",
            ground_truth=f"answer-{source}",
            data_source=source,
            source_dataset=("chartqa" if source == "teacher" else source),
            task_kind=("math" if source == "thinklite" else "mcq"),
            metadata={
                "candidate_sha256": f"{index + 1:x}" * 64,
                "gt_regions": ((1, 1, 8, 7),) if source == "vstar" else None,
                "tools_kwargs": tools_kwargs_for_source(
                    source,
                    ((1, 1, 8, 7),) if source == "vstar" else None,
                ),
                "mixture_role": "teacher" if source == "teacher" else "base",
                "parent": {
                    "dataset_kind": "fixture",
                    "row_index": index,
                    "row_sha256": f"{index + 5:x}" * 64,
                },
            },
        )
        for index, source in enumerate(("vstar", "arxivqa", "thinklite", "teacher"))
    )
    processor = _FakeInstructProcessor()
    template_sha256 = hashlib.sha256(
        processor.chat_template.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(module, "POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256", template_sha256)
    binding = _binding(
        tmp_path,
        profile=profile,
        prompt_sha256=prompt_sha256,
        template_sha256=template_sha256,
    )
    runtime = SimpleNamespace(
        samples=samples,
        samples_sha256=binding.samples_sha256,
        iteration_identity_sha256=binding.iteration_identity_sha256,
    )
    monkeypatch.setattr(
        module,
        "load_policy_teacher_quarter_mix_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    config = {
        POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME: binding.as_config(),
        "mm_processor_kwargs": {"max_pixels": 512 * 512},
    }
    dataset = PolicyTeacherQuarterMixDataset(
        str(binding.samples_path),
        tokenizer=processor.tokenizer,
        processor=processor,
        config=config,
    )

    rows = [dataset[index] for index in range(len(dataset))]
    assert [row["sample_id"] for row in rows] == [
        sample.sample_id for sample in samples
    ]
    assert all(
        row["dataset_iteration_identity_sha256"] == binding.iteration_identity_sha256
        for row in rows
    )
    assert all(
        row["agent_name"] == POLICY_TEACHER_QUARTER_MIX_AGENT_NAME for row in rows
    )
    assert rows[3]["extra_info"]["mixture_role"] == "teacher"
    assert rows[3]["extra_info"]["source_dataset"] == "chartqa"
    assert rows[3]["extra_info"]["source_route"] != "single_turn_no_tool"
    assert rows[2]["extra_info"]["source_route"] == "single_turn_no_tool"
    assert rows[3]["prompt_bundle_sha256"] == prompt_sha256
    assert rows[2]["prompt_bundle_sha256"] == (THINKLITE_PROMPT_IDENTITY.bundle_sha256)
    assert all("initial_prompt_token_ids" in row for row in rows)
    if profile is NativeToolCapabilityProfile.NO_TOOL:
        assert rows[0]["tools_kwargs"] == {}
        assert rows[1]["tools_kwargs"] == {}
        assert rows[3]["tools_kwargs"] == {}
        assert rows[0]["extra_info"]["need_tools_kwargs"] is False
        assert len(rows[0]["raw_prompt"]) == 1
        assert rows[0]["raw_prompt"][0]["role"] == "user"


def test_binding_rejects_a_renderer_identity_from_another_tool_profile(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="visual prompt identity differs"):
        _binding(
            tmp_path,
            profile=NativeToolCapabilityProfile.TGVF_ONLY,
            prompt_sha256=VISUAL_PROMPT_IDENTITY.bundle_sha256,
            template_sha256=module.POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
        )


def test_tgvf_renderer_route_is_explicitly_bound_by_prompt_hash() -> None:
    short_contract = module._visual_prompt_contract(
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
    )
    full_contract = module._visual_prompt_contract(
        NativeToolCapabilityProfile.TGVF_ONLY,
        TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY.bundle_sha256,
    )

    assert short_contract[0] is TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY
    assert full_contract[0] is TGVF_TARGET_GUIDE_V2_PROMPT_IDENTITY
    assert full_contract[2]("question")[0]["content"] == (
        TGVF_TARGET_GUIDE_V2_SYSTEM_PROMPT
    )
    assert short_contract[3] == full_contract[3]
    assert short_contract[4] == "matched_tgvf_visual_tool"
    assert full_contract[4] == short_contract[4]

    with pytest.raises(ValueError, match="visual prompt identity differs"):
        module._visual_prompt_contract(
            NativeToolCapabilityProfile.TGVF_ONLY,
            "f" * 64,
        )


@pytest.mark.parametrize(
    ("profile", "prompt_sha256", "vstar_expectation"),
    (
        (
            NativeToolCapabilityProfile.NO_TOOL,
            NO_TOOL_RL_PROMPT_IDENTITY.bundle_sha256,
            "direct_no_tool",
        ),
        (
            NativeToolCapabilityProfile.CROP_ONLY,
            VISUAL_PROMPT_IDENTITY.bundle_sha256,
            "crop_possible",
        ),
        (
            NativeToolCapabilityProfile.TGVF_ONLY,
            TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            "tgvf_possible",
        ),
        (
            NativeToolCapabilityProfile.CROP_TGVF,
            CROP_TGVF_DEEPEYES_MATCHED_PROMPT_IDENTITY.bundle_sha256,
            "crop_tgvf_possible",
        ),
    ),
)
def test_legacy_splits_preserve_schedule_identity_and_smoke_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: NativeToolCapabilityProfile,
    prompt_sha256: str,
    vstar_expectation: str,
) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "source.png"
    image_module.new("RGB", (16, 8), color=(12, 34, 56)).save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = tuple(
        SimpleNamespace(
            sample_id=f"legacy:{source}",
            candidate_sha256=f"{index + 1:x}" * 64,
            image_path=image_path,
            image_sha256=image_sha256,
            question=f"question-{source}",
            ground_truth=f"answer-{source}",
            data_source=source,
            task_kind="math" if source == "thinklite" else "mcq",
            tools_kwargs=tools_kwargs_for_source(
                source,
                ((1, 1, 8, 7),) if source == "vstar" else None,
            ),
        )
        for index, source in enumerate(("vstar", "arxivqa", "thinklite"))
    )
    schedule = SimpleNamespace(
        file_sha256="a" * 64,
        identity_sha256="b" * 64,
        schedule_identity_sha256="c" * 64,
        probe_manifest={"name": "fixture"},
        train=samples,
        probe=samples,
        smoke=samples,
    )
    monkeypatch.setattr(module, "_verified_schedule_index", lambda: schedule)

    processor = _FakeInstructProcessor()
    template_sha256 = hashlib.sha256(
        processor.chat_template.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(module, "POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256", template_sha256)
    binding = _binding(
        tmp_path,
        profile=profile,
        prompt_sha256=prompt_sha256,
        template_sha256=template_sha256,
    )
    config = {
        POLICY_TEACHER_QUARTER_MIX_CONFIG_NAME: binding.as_config(),
        "mm_processor_kwargs": {"max_pixels": 512 * 512},
    }

    smoke = PolicyTeacherQuarterMixDataset(
        str(DEEPEYES_SMOKE_SENTINEL),
        tokenizer=processor.tokenizer,
        processor=processor,
        config=config,
    )
    smoke_rows = [smoke[index] for index in range(len(smoke))]
    assert [row["extra_info"]["smoke_expectation"] for row in smoke_rows] == [
        vstar_expectation,
        "direct_no_call",
        "no_tool",
    ]
    assert all(row["extra_info"]["mixture_role"] == "smoke" for row in smoke_rows)
    assert smoke.probe_manifest == {"name": "fixture"}

    for sentinel, split in (
        (DEEPEYES_PROBE_SENTINEL, "probe"),
        (DEEPEYES_TRAIN_SENTINEL, "legacy_train"),
    ):
        dataset = PolicyTeacherQuarterMixDataset(
            str(sentinel),
            tokenizer=processor.tokenizer,
            processor=processor,
            config=config,
        )
        extra = dataset[0]["extra_info"]
        assert extra["mixture_role"] == split
        assert extra["schedule_index_file_sha256"] == "a" * 64
        assert extra["schedule_index_identity_sha256"] == "b" * 64
        assert extra["schedule_identity_sha256"] == "c" * 64
        assert "smoke_expectation" not in extra
