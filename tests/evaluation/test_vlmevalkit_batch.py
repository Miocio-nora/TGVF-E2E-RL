import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from tgvf_rl.evaluation.vlmevalkit_batch import (
    CoreDevBatchInferenceConfig,
    _capture_vllm_requests,
    _generate_batch,
    attach_coredev_batch_options_from_factory_kwargs,
    install_coredev_batched_inference,
    stable_coredev_request_seed,
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], list[object], bool]] = []

    def generate(
        self,
        prompts: list[dict[str, object]],
        *,
        sampling_params: object,
        use_tqdm: bool = True,
    ) -> list[object]:
        params = (
            sampling_params if isinstance(sampling_params, list) else [sampling_params]
        )
        self.calls.append((prompts, params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(text=f"{request['prompt']}|seed={param.seed}")
                ]
            )
            for request, param in zip(prompts, params, strict=True)
        ]


class FakeModel:
    use_vllm = True
    post_process = False
    is_api = False

    def __init__(self) -> None:
        self.llm = FakeLLM()
        self.dump_image_func = None

    def generate(self, *, message: object, dataset: str) -> str:
        params = SimpleNamespace(seed=None, temperature=1.0, max_tokens=40960)
        output = self.llm.generate(
            [{"prompt": f"{dataset}:{message}", "multi_modal_data": {"image": 1}}],
            sampling_params=params,
        )
        return output[0].outputs[0].text

    def set_dump_image(self, dump_image_func: object) -> None:
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset: str) -> bool:
        return False


def _config(batch_size: int = 8) -> CoreDevBatchInferenceConfig:
    return CoreDevBatchInferenceConfig(
        inference_batch_size=batch_size,
        request_seed_base=0,
        request_seed_namespace="coredev-2511-qwen3-direct-batched-v1",
    )


def _attached_model(batch_size: int = 8) -> FakeModel:
    wrapped = attach_coredev_batch_options_from_factory_kwargs(lambda **_: FakeModel())
    return wrapped(
        inference_batch_size=batch_size,
        request_seed_base=0,
        request_seed_namespace="coredev-2511-qwen3-direct-batched-v1",
    )


def test_stable_request_seed_is_canonical_and_low_31_bit() -> None:
    kwargs = {
        "seed_namespace": "coredev-2511-qwen3-direct-batched-v1",
        "seed_base": 0,
        "dataset_name": "VStarBench",
        "canonical_index": "row/7",
    }
    seed = stable_coredev_request_seed(**kwargs)
    assert seed == stable_coredev_request_seed(**kwargs)
    assert 0 <= seed <= (1 << 31) - 1
    assert seed != stable_coredev_request_seed(
        **{**kwargs, "canonical_index": "row/8"}
    )


def test_batch_reuses_scalar_request_builder_and_restores_constructor() -> None:
    model = FakeModel()
    original = model.llm.generate
    messages = ["a", "b", "c"]
    indices = [3, 5, 9]

    texts = _generate_batch(
        model=model,
        messages=messages,
        canonical_indices=indices,
        dataset_name="VStarBench",
        config=_config(),
    )

    assert model.llm.generate == original
    assert len(model.llm.calls) == 1
    requests, params, use_tqdm = model.llm.calls[0]
    assert [request["prompt"] for request in requests] == [
        "VStarBench:a",
        "VStarBench:b",
        "VStarBench:c",
    ]
    assert [param.seed for param in params] == [
        stable_coredev_request_seed(
            seed_namespace=_config().request_seed_namespace,
            seed_base=0,
            dataset_name="VStarBench",
            canonical_index=index,
        )
        for index in indices
    ]
    assert use_tqdm is False
    assert texts == [
        f"{request['prompt']}|seed={param.seed}"
        for request, param in zip(requests, params, strict=True)
    ]


def test_request_constructor_is_restored_when_scalar_preparation_fails() -> None:
    model = FakeModel()
    original = model.llm.generate

    def broken_generate(*, message: object, dataset: str) -> str:
        model.llm.generate(
            [{"prompt": str(message)}],
            sampling_params=SimpleNamespace(seed=None),
        )
        raise RuntimeError("prepare failed")

    model.generate = broken_generate  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="prepare failed"):
        _capture_vllm_requests(
            model=model,
            messages=["a"],
            dataset_name="VStarBench",
        )
    assert model.llm.generate == original


def _dump(data: object, path: str) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(data, handle)


def _load(path: str) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class FakeDataset:
    dataset_name = "VStarBench"
    force_use_dataset_prompt = False

    def __init__(self) -> None:
        self.data = pd.DataFrame({"index": ["r0", "r1", "r2", "r3", "r4"]})

    def __len__(self) -> int:
        return len(self.data)

    def build_prompt(self, row: pd.Series) -> str:
        return f"question-{row['index']}"

    def dump_image(self, row: object) -> list[str]:
        return []


def test_installed_batch_loop_reuses_checkpoint_and_is_batch_boundary_invariant(
    tmp_path: Path,
) -> None:
    original_calls: list[object] = []

    def original(**kwargs: object) -> object:
        original_calls.append(kwargs)
        return None

    module = SimpleNamespace(
        infer_data=original,
        get_rank_and_world_size=lambda: (0, 1),
        load=_load,
        dump=_dump,
        osp=os.path,
        tqdm=lambda **_: SimpleNamespace(update=lambda _: None, close=lambda: None),
    )
    install_coredev_batched_inference(module)
    dataset = FakeDataset()
    out_file = tmp_path / "0_1_VStarBench.pkl"
    existing = {"r0": "materialized-r0", "r1": "materialized-r1"}
    _dump(existing, str(out_file))

    model = _attached_model(batch_size=2)
    returned = module.infer_data(
        model=model,
        model_name="Qwen3-VL-8B-Thinking",
        work_dir=str(tmp_path),
        dataset=dataset,
        out_file=str(out_file),
    )

    assert returned is model
    assert original_calls == []
    result = _load(str(out_file))
    assert list(result) == ["r0", "r1", "r2", "r3", "r4"]
    assert result["r0"] == "materialized-r0"
    assert result["r1"] == "materialized-r1"
    assert [len(call[0]) for call in model.llm.calls] == [2, 1]

    regenerated = _attached_model(batch_size=1)
    regenerated_file = tmp_path / "regenerated.pkl"
    regenerated_dataset = FakeDataset()
    regenerated_dataset.data = regenerated_dataset.data.iloc[2:].reset_index(drop=True)
    module.infer_data(
        model=regenerated,
        model_name="Qwen3-VL-8B-Thinking",
        work_dir=str(tmp_path / "fresh"),
        dataset=regenerated_dataset,
        out_file=str(regenerated_file),
    )
    regenerated_result = _load(str(regenerated_file))
    assert [result[index] for index in ("r2", "r3", "r4")] == [
        regenerated_result[index] for index in ("r2", "r3", "r4")
    ]
