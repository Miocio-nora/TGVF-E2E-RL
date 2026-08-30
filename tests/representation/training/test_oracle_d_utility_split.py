from __future__ import annotations

import ast
import dataclasses
from hashlib import sha256
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import FunctionType, SimpleNamespace
from typing import get_type_hints

import torch

from tgvf_rl.representation.training import (
    oracle_d_execution,
    oracle_d_ledger,
    oracle_d_schema,
    oracle_d_utility,
)
from tgvf_rl.representation.training.readout import RepresentationVisualTensorBundle
from tgvf_rl.representation.training.schema import RepresentationChoice


_PUBLIC_MODULE = "tgvf_rl.representation.training.oracle_d_utility"
_TRAINING_ROOT = Path(__file__).parents[3] / "src/tgvf_rl/representation/training"
_IMPLEMENTATION_FILES = (
    "oracle_d_utility.py",
    "oracle_d_schema.py",
    "oracle_d_execution.py",
    "oracle_d_ledger.py",
)
_REBOUND_FUNCTIONS = (
    oracle_d_schema.split_oracle_d_utility_sample,
    oracle_d_schema.build_image_only_messages,
    oracle_d_schema.build_oracle_target_messages,
    oracle_d_schema.score_oracle_generated_answer,
    oracle_d_schema._thinking_final_answer,
    oracle_d_schema._strip_terminal_markers,
    oracle_d_schema._normalize_answer,
    oracle_d_schema._choice_label_for_expected,
    oracle_d_schema._multiple_choice_label,
    oracle_d_schema._parse_number,
    oracle_d_schema._require_model_input,
    oracle_d_execution.materialize_oracle_group_visuals,
    oracle_d_execution.greedy_oracle_answer,
    oracle_d_execution.greedy_oracle_answers_batched,
    oracle_d_execution.verify_image_only_injected_native_parity,
    oracle_d_execution._batched_next_logits,
    oracle_d_execution._oracle_target_condition,
    oracle_d_execution._render_direct_without_tools,
    oracle_d_execution._injected_block,
    oracle_d_execution._zero_bundle,
    oracle_d_execution._detached_bundle,
    oracle_d_execution._assert_visual_bundle_match,
    oracle_d_execution._greedy_token,
    oracle_d_execution._qwen3_multimodal_token_ids,
    oracle_d_execution._integer_sequence_sha256,
    oracle_d_ledger._run_identity_payload,
    oracle_d_ledger._arm_contract,
    oracle_d_ledger._paired_summary,
    oracle_d_ledger._canonical_json_bytes,
    oracle_d_ledger._canonical_sha256,
    oracle_d_ledger._atomic_write_json,
    oracle_d_ledger._atomic_write_bytes,
    oracle_d_ledger._git_head,
)
_ANNOTATION_FROZEN_TYPES = (
    oracle_d_schema.OracleDUtilityGroundTruth,
    oracle_d_schema.OracleGeneratedAnswer,
    oracle_d_schema.OracleGroupVisuals,
    oracle_d_schema.OracleArmContext,
)
_FROZEN_NON_PRIVATE_NAMESPACE = {
    "Any",
    "CachedTokenForwardRequest",
    "DEFAULT_ORACLE_D_UTILITY_ARMS",
    "DEFAULT_THINKING_EOS_TOKEN_IDS",
    "Enum",
    "Fraction",
    "InjectedForwardRequest",
    "InjectedVisualBlock",
    "Iterator",
    "Literal",
    "Mapping",
    "ModelActionTarget",
    "NATIVE_REPRESENTATION_PRE_REASONING",
    "NativeActionTarget",
    "NativeAssistantDialect",
    "ORACLE_D_UTILITY_RECORD_SCHEMA_VERSION",
    "ORACLE_D_UTILITY_SCHEMA_VERSION",
    "ORACLE_D_UTILITY_SUMMARY_SCHEMA_VERSION",
    "OracleAnswerScore",
    "OracleArmContext",
    "OracleBatchCompatibilityError",
    "OracleDUtilityArm",
    "OracleDUtilityGroundTruth",
    "OracleDUtilityModelInput",
    "OracleGeneratedAnswer",
    "OracleGroupVisuals",
    "OracleImageOnlyParity",
    "OrderedDict",
    "Path",
    "Qwen3ContextualHiddenStateStack",
    "Qwen3NativeRepresentationGroupBuilder",
    "Qwen3RepresentationRuntime",
    "Qwen3VLAdapter",
    "Qwen3VisionFeatures",
    "Qwen3VisionPreMergeRequest",
    "QwenVLMFamilyAdapter",
    "RepresentationChoice",
    "RepresentationTrainingConfig",
    "RepresentationTrainingSample",
    "RepresentationVisualTensorBundle",
    "Sequence",
    "TGVFAdapterOutput",
    "TGVF_FOCUS_TOOL_NAME",
    "TargetConditioningProviderKind",
    "TargetConditioningRequest",
    "annotations",
    "asdict",
    "batch_identical_injected_requests",
    "build_image_only_messages",
    "build_oracle_target_messages",
    "contextmanager",
    "create_qwen3_representation_runtime",
    "dataclass",
    "datetime",
    "greedy_oracle_answer",
    "greedy_oracle_answers_batched",
    "json",
    "load_rank_zero_adapter_owned_state_export",
    "load_representation_internal_evaluation_run_config",
    "load_representation_training_config",
    "load_retained_representation_jsonl",
    "materialize_oracle_group_visuals",
    "os",
    "prepare_oracle_arm_context",
    "re",
    "run_oracle_d_utility_evaluation",
    "score_oracle_generated_answer",
    "sha256",
    "split_oracle_d_utility_sample",
    "state_digest",
    "subprocess",
    "tempfile",
    "time",
    "timezone",
    "torch",
    "verify_image_only_injected_native_parity",
}


def _tree(filename: str) -> ast.Module:
    path = _TRAINING_ROOT / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(filename: str) -> set[str]:
    return {
        node.name
        for node in _tree(filename).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _local_imports(filename: str) -> set[str]:
    return {
        (node.module or "").split(".")[0]
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_oracle_d_facade_keeps_frozen_non_private_namespace() -> None:
    observed = {name for name in vars(oracle_d_utility) if not name.startswith("_")}
    assert observed == _FROZEN_NON_PRIVATE_NAMESPACE


def test_oracle_d_definitions_have_one_owner_and_one_way_imports() -> None:
    owners: dict[str, str] = {}
    for filename in _IMPLEMENTATION_FILES:
        for definition in _definitions(filename):
            assert definition not in owners, (
                f"{definition} is owned by {owners[definition]} and {filename}"
            )
            owners[definition] = filename

    assert len(owners) == 52
    assert _definitions("oracle_d_utility.py") == {
        "prepare_oracle_arm_context",
        "run_oracle_d_utility_evaluation",
        "_ordered_sample_groups",
        "_select_groups",
        "_normalize_arms",
        "_validate_selection",
        "_require_single_gpu_environment",
        "_normalize_eos_token_ids",
        "_model_eos_token_ids",
    }
    assert "oracle_d_utility" not in _local_imports("oracle_d_schema.py")
    assert "oracle_d_utility" not in _local_imports("oracle_d_execution.py")
    assert "oracle_d_utility" not in _local_imports("oracle_d_ledger.py")
    assert "oracle_d_execution" not in _local_imports("oracle_d_schema.py")
    assert "oracle_d_ledger" not in _local_imports("oracle_d_schema.py")
    assert "oracle_d_ledger" not in _local_imports("oracle_d_execution.py")


def test_oracle_d_facade_reexports_exact_leaf_objects() -> None:
    for leaf in (oracle_d_schema, oracle_d_execution, oracle_d_ledger):
        for name in leaf.__all__:
            assert getattr(oracle_d_utility, name) is getattr(leaf, name)

    for function in _REBOUND_FUNCTIONS:
        assert getattr(oracle_d_utility, function.__name__) is function


def test_moved_types_keep_module_methods_hints_and_class_pickle_coordinates() -> None:
    moved_types = (
        oracle_d_schema.OracleDUtilityArm,
        oracle_d_schema.OracleBatchCompatibilityError,
        oracle_d_schema.OracleDUtilityModelInput,
        oracle_d_schema.OracleDUtilityGroundTruth,
        oracle_d_schema.OracleAnswerScore,
        oracle_d_schema.OracleGeneratedAnswer,
        oracle_d_schema.OracleGroupVisuals,
        oracle_d_schema.OracleImageOnlyParity,
        oracle_d_schema.OracleArmContext,
        oracle_d_ledger._OracleRunLedger,
    )
    shared_dataclass_helpers = {
        dataclasses._dataclass_getstate,  # noqa: SLF001
        dataclasses._dataclass_setstate,  # noqa: SLF001
    }
    for moved_type in moved_types:
        assert moved_type.__module__ == _PUBLIC_MODULE
        assert pickle.loads(pickle.dumps(moved_type)) is moved_type
        resolved_hints = get_type_hints(moved_type)
        own_annotations = vars(moved_type).get("__annotations__", {})
        if moved_type in _ANNOTATION_FROZEN_TYPES:
            assert all(
                not isinstance(annotation, str)
                for annotation in own_annotations.values()
            )
            assert own_annotations == {
                name: resolved_hints[name] for name in own_annotations
            }
        if dataclasses.is_dataclass(moved_type):
            # Dataclass field metadata was captured at decoration time.  The
            # narrow compatibility fix resolves the class's raw annotations
            # without retroactively mutating that existing field metadata.
            assert all(
                isinstance(field.type, str) for field in dataclasses.fields(moved_type)
            )
        for name, member in vars(moved_type).items():
            if isinstance(member, property):
                functions = (member.fget, member.fset, member.fdel)
            elif isinstance(member, (classmethod, staticmethod)):
                assert (
                    vars(getattr(oracle_d_utility, moved_type.__name__))[name] is member
                )
                functions = (member.__func__,)
            else:
                functions = (member,)
            for function in functions:
                if not isinstance(function, FunctionType):
                    continue
                if function.__module__ == "enum":
                    continue
                expected = (
                    dataclasses.__name__
                    if function in shared_dataclass_helpers
                    else _PUBLIC_MODULE
                )
                assert function.__module__ == expected


def test_moved_functions_keep_module_qualname_and_type_hints() -> None:
    assert len(_REBOUND_FUNCTIONS) == 33
    assert len({function.__name__ for function in _REBOUND_FUNCTIONS}) == 33
    for function in _REBOUND_FUNCTIONS:
        assert function.__module__ == _PUBLIC_MODULE
        assert function.__qualname__ == function.__name__
        assert getattr(oracle_d_utility, function.__name__) is function
        assert pickle.loads(pickle.dumps(function)) is function
        get_type_hints(function)


def test_moved_instances_pickle_through_historical_facade() -> None:
    choice = RepresentationChoice(label="A", text="white")
    bundle = RepresentationVisualTensorBundle(
        main=torch.zeros((1, 1, 2)),
        deepstack=(torch.zeros((1, 1, 2)),),
        branch_layers=(1,),
        d_deepstack_active=False,
    )
    values = (
        oracle_d_schema.OracleDUtilityArm.IMAGE_ONLY,
        oracle_d_schema.OracleBatchCompatibilityError("fixture"),
        oracle_d_schema.OracleDUtilityModelInput(
            "sample", "image", "/tmp/image.png", "question", "target", "a" * 64
        ),
        oracle_d_schema.OracleDUtilityGroundTruth("sample", "white", (choice,)),
        oracle_d_schema.OracleAnswerScore(
            True, "exact", "white", "white", "white", "A", "A"
        ),
        oracle_d_schema.OracleGeneratedAnswer((1,), "white", "natural_stop"),
        oracle_d_schema.OracleGroupVisuals(bundle, {"sample": bundle}, (1, 1, 1)),
        oracle_d_schema.OracleImageOnlyParity(
            "sample", 1, 1, True, 0.0, 0.0, 2, (1, 1, 1)
        ),
        oracle_d_schema.OracleArmContext(
            oracle_d_schema.OracleDUtilityArm.IMAGE_ONLY,
            "a" * 64,
            "b" * 64,
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((3, 1, 2), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
            (),
            (0,),
            (),
            frozenset(),
        ),
        oracle_d_ledger._OracleRunLedger(
            Path("/tmp/oracle-pickle-fixture"),
            identity_payload={"arms": ["image_only"]},
            expected_keys=(("sample", "image_only"),),
        ),
    )
    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)


def test_ledger_identity_still_hashes_historical_facade_file(monkeypatch) -> None:
    facade = Path(oracle_d_utility.__file__).resolve()
    assert oracle_d_ledger._PUBLIC_MODULE_PATH.resolve() == facade
    assert (
        sha256(facade.read_bytes()).hexdigest()
        != sha256(Path(oracle_d_ledger.__file__).read_bytes()).hexdigest()
    )
    roots: list[Path] = []
    monkeypatch.setattr(
        oracle_d_ledger,
        "_git_head",
        lambda root: roots.append(root) or "f" * 40,
    )
    source = SimpleNamespace(
        source_path=Path("/tmp/eval.toml"),
        source_sha256="a" * 64,
        training_config_sha256="b" * 64,
        artifact_file_sha256="c" * 64,
        artifact_manifest_sha256="d" * 64,
        expected_run_identity_sha256="e" * 64,
        expected_global_step=1,
        evaluation=SimpleNamespace(eos_token_ids=(151645,), random_seed=42),
    )
    training = SimpleNamespace(
        model=SimpleNamespace(model_name="Qwen3-VL-8B-Thinking", local_path="/model")
    )
    model_input = oracle_d_schema.OracleDUtilityModelInput(
        "sample", "image", "/tmp/image.png", "question", "target", "a" * 64
    )

    payload = oracle_d_ledger._run_identity_payload(
        source_config=source,
        training=training,
        data_manifest_sha256="f" * 64,
        model_inputs=(model_input,),
        arms=(oracle_d_schema.OracleDUtilityArm.IMAGE_ONLY,),
        max_new_tokens=8,
        eos_token_ids=(151645,),
        decode_mode="cached",
        group_start=0,
        group_limit=1,
        shard_index=0,
        shard_count=1,
    )

    assert roots == [facade.parents[4]]
    assert payload["live_module_sha256"] == sha256(facade.read_bytes()).hexdigest()


def test_oracle_d_leaf_first_pickle_import_order() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    training_directory = repository_root / "src/tgvf_rl/representation/training"
    code = textwrap.dedent(
        """
        import importlib
        import pickle
        import sys
        import types
        from typing import get_type_hints

        import tgvf_rl.representation

        package = types.ModuleType("tgvf_rl.representation.training")
        package.__package__ = "tgvf_rl.representation.training"
        package.__path__ = [sys.argv[1]]
        sys.modules["tgvf_rl.representation.training"] = package

        schema = importlib.import_module(
            "tgvf_rl.representation.training.oracle_d_schema"
        )
        ledger = importlib.import_module(
            "tgvf_rl.representation.training.oracle_d_ledger"
        )
        facade_name = "tgvf_rl.representation.training.oracle_d_utility"
        assert facade_name not in sys.modules
        moved_types = (
            schema.OracleDUtilityArm,
            schema.OracleBatchCompatibilityError,
            schema.OracleDUtilityModelInput,
            schema.OracleDUtilityGroundTruth,
            schema.OracleAnswerScore,
            schema.OracleGeneratedAnswer,
            schema.OracleGroupVisuals,
            schema.OracleImageOnlyParity,
            schema.OracleArmContext,
            ledger._OracleRunLedger,
        )
        annotation_frozen_types = (
            schema.OracleDUtilityGroundTruth,
            schema.OracleGeneratedAnswer,
            schema.OracleGroupVisuals,
            schema.OracleArmContext,
        )
        for moved_type in moved_types:
            assert moved_type.__module__ == facade_name
            resolved_hints = get_type_hints(moved_type)
            own_annotations = vars(moved_type).get("__annotations__", {})
            if moved_type in annotation_frozen_types:
                assert all(
                    not isinstance(annotation, str)
                    for annotation in own_annotations.values()
                )
                assert own_annotations == {
                    name: resolved_hints[name] for name in own_annotations
                }
            assert facade_name not in sys.modules

        encoded = tuple(pickle.dumps(moved_type) for moved_type in moved_types)
        facade = importlib.import_module(facade_name)
        for moved_type, payload in zip(moved_types, encoded, strict=True):
            assert getattr(facade, moved_type.__name__) is moved_type
            assert pickle.loads(payload) is moved_type
        """
    )
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    environment.pop("OPENROUTER_API_KEY", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        (sys.executable, "-c", code, str(training_directory)),
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_oracle_d_modules_keep_size_headroom() -> None:
    for filename in _IMPLEMENTATION_FILES:
        line_count = len(
            (_TRAINING_ROOT / filename).read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= 850, f"{filename} grew to {line_count} lines"
