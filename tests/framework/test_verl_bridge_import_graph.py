from __future__ import annotations

import ast
from pathlib import Path
import pickle

from tgvf_rl.framework.verl import rollout_bridge, rollout_contract


_VERL_PACKAGE = Path(__file__).parents[2] / "src" / "tgvf_rl" / "framework" / "verl"
_PUBLIC_CONTRACT_EXPORTS = (
    "ACTUAL_RESPONSE_LOGPROBS_FIELD",
    "AGENT_LOOP_EXACT_SIDECAR_FIELDS",
    "BEHAVIOR_TRACE_HANDLES_FIELD",
    "BEHAVIOR_TRACE_RECORDS_FIELD",
    "BRIDGE_SCHEMA_FIELD",
    "BRIDGE_SCHEMA_VERSION",
    "DATAPROTO_META_SCHEMA_FIELD",
    "DATAPROTO_META_SCHEMA_VERSION",
    "EXACT_OBSERVATION_HANDLES_FIELD",
    "EXACT_PROMPT_IDS_FIELD",
    "EXACT_RESPONSE_IDS_FIELD",
    "OBJECTIVE_SENTINELS_FIELD",
    "ROLLOUT_PROVENANCE_SHA256_FIELD",
    "SIDECAR_RELEASE_FIELDS_FIELD",
    "SIDECAR_RELEASE_SCHEMA_FIELD",
    "SIDECAR_RELEASE_SCHEMA_VERSION",
    "TOKEN_OWNERSHIP_SHA256_FIELD",
    "TRAJECTORY_ID_FIELD",
    "TRAJECTORY_PAYLOAD_FIELD",
    "TRAJECTORY_REPLAY_BUNDLE_FIELD",
    "TRAJECTORY_REPLAY_HANDLE_FIELD",
    "TRAJECTORY_SHA256_FIELD",
    "RolloutBridgeRecord",
    "build_agent_loop_output",
    "parse_agent_loop_output",
    "rollout_provenance_checksum",
    "token_ownership_checksum",
    "trajectory_to_rollout_bridge",
)


def _relative_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_data_and_rollout_bridges_have_one_way_module_dependencies() -> None:
    data_imports = _relative_imports(_VERL_PACKAGE / "data_bridge.py")
    rollout_imports = _relative_imports(_VERL_PACKAGE / "rollout_bridge.py")
    contract_imports = _relative_imports(_VERL_PACKAGE / "rollout_contract.py")

    assert "rollout_contract" in data_imports
    assert "rollout_bridge" not in data_imports
    assert "data_bridge" in rollout_imports
    assert "rollout_contract" in rollout_imports
    assert {"data_bridge", "rollout_bridge"}.isdisjoint(contract_imports)


def test_rollout_bridge_preserves_the_established_contract_import_surface() -> None:
    for name in _PUBLIC_CONTRACT_EXPORTS:
        assert getattr(rollout_bridge, name) is getattr(rollout_contract, name)
    for name in (
        "RolloutBridgeRecord",
        "build_agent_loop_output",
        "parse_agent_loop_output",
        "rollout_provenance_checksum",
        "token_ownership_checksum",
        "trajectory_to_rollout_bridge",
    ):
        value = getattr(rollout_bridge, name)
        assert value.__module__ == rollout_bridge.__name__
        assert pickle.loads(pickle.dumps(value)) is value
