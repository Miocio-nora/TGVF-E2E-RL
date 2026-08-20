from __future__ import annotations

from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).parents[2]


def test_fmt2_reward_manager_imports_in_a_fresh_reward_worker() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tgvf_rl.rewards.deepeyes_crop_tfree_verl_reward "
                "import DeepEyesCropTFreeFMT2RewardManager; "
                "print(DeepEyesCropTFreeFMT2RewardManager.__name__)"
            ),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "DeepEyesCropTFreeFMT2RewardManager"
