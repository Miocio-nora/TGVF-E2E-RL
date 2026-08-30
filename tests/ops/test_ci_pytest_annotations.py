from __future__ import annotations

from pathlib import Path
import runpy


CI_ANNOTATION_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "emit_ci_pytest_annotations.py"
)
pytest_failure_annotations = runpy.run_path(CI_ANNOTATION_SCRIPT)[
    "pytest_failure_annotations"
]


def test_pytest_failure_annotations_expose_node_and_escaped_traceback(
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite failures="1" errors="1">
    <testcase classname="tests.test_contract" name="test_failure">
      <failure message="assert 1 == 2">line one
line two % value</failure>
    </testcase>
    <testcase classname="tests.test_contract" name="test_error">
      <error message="collection failed">traceback</error>
    </testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    annotations = pytest_failure_annotations(report)

    assert len(annotations) == 2
    assert "tests.test_contract::test_failure%0Aassert 1 == 2" in annotations[0]
    assert "line one%0Aline two %25 value" in annotations[0]
    assert "tests.test_contract::test_error%0Acollection failed" in annotations[1]


def test_pytest_failure_annotations_fail_observably_for_missing_report(
    tmp_path: Path,
) -> None:
    annotations = pytest_failure_annotations(tmp_path / "missing.xml")

    assert len(annotations) == 1
    assert "CPU pytest report unavailable" in annotations[0]
