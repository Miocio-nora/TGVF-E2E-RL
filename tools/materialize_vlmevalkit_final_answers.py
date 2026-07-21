"""Create a final-answer-only scoring view from a completed VLMEvalKit TSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tgvf_rl.evaluation.final_answer_view import (  # noqa: E402
    materialize_final_answer_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an immutable final-answer VLMEvalKit scoring TSV."
    )
    parser.add_argument("source_tsv", type=Path)
    parser.add_argument("derived_tsv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mathverse-source-json", type=Path)
    args = parser.parse_args()
    result = materialize_final_answer_view(
        source_tsv=args.source_tsv,
        derived_tsv=args.derived_tsv,
        manifest_path=args.manifest,
        mathverse_source_json=args.mathverse_source_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
