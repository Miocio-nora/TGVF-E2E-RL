from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_scoring_module_and_cli_help_are_cpu_only() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import tgvf_rl.data.policy_selection_t1_scoring
for name in ('torch', 'vllm', 'transformers', 'PIL'):
    assert name not in sys.modules, (name, sorted(sys.modules))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "score_policy_data_selection_t1.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--judge-config" in completed.stdout
    assert "--quality-exclusions" in completed.stdout
