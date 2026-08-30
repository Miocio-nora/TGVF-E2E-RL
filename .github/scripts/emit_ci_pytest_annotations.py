"""Emit compact GitHub workflow annotations from a pytest JUnit report."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


_MAX_ANNOTATION_CHARACTERS = 12_000


def _workflow_command_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def pytest_failure_annotations(report_path: Path) -> tuple[str, ...]:
    """Return one escaped workflow annotation for every failure or error."""

    try:
        root = ET.fromstring(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ET.ParseError) as error:
        detail = _workflow_command_data(
            f"pytest failed and its JUnit report could not be read: {error}"
        )
        return (f"::error title=CPU pytest report unavailable::{detail}",)

    annotations: list[str] = []
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        if failure is None:
            failure = testcase.find("error")
        if failure is None:
            continue
        node_id = "::".join(
            item
            for item in (testcase.get("classname", ""), testcase.get("name", ""))
            if item
        )
        details = "\n".join(
            item for item in (failure.get("message", ""), failure.text or "") if item
        ).strip()
        message = f"{node_id}\n{details}".strip()
        if len(message) > _MAX_ANNOTATION_CHARACTERS:
            message = message[:_MAX_ANNOTATION_CHARACTERS] + "\n...[truncated]"
        annotations.append(
            "::error title=Complete CPU pytest failure::"
            + _workflow_command_data(message)
        )
    if annotations:
        return tuple(annotations)
    return (
        "::error title=CPU pytest failure::"
        "pytest exited nonzero without a failure or error in its JUnit report",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    for annotation in pytest_failure_annotations(args.report):
        print(annotation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
