"""Deterministic, CPU-only selection of the 192-candidate T1 canary.

The selector is deliberately independent of generation outcomes.  It keeps a
bounded bottom-k reservoir per stratum, so selecting from the full candidate
population does not retain every canonical record in memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import heapq
import json
import re
from typing import Any

from .policy_selection import (
    POLICY_SELECTION_PRIMARY_SOURCES,
    SelectionCandidate,
    SelectionSource,
)


POLICY_SELECTION_CANARY_MANIFEST_SCHEMA = "tgvf.policy-selection.t1-canary-manifest.v1"
POLICY_SELECTION_CANARY_ALGORITHM_VERSION = "t1-canary-content-hash-v1"
T1_CANARY_PER_SOURCE = 64
T1_CANARY_TOTAL = 3 * T1_CANARY_PER_SOURCE

VSTAR_CANARY_SOURCE_FILES = (
    "GQA_data.json",
    "llava_focus_data.json",
    "spatial_relation_data.json",
    "vaw_attribute_data.json",
)
VSTAR_CANARY_PER_FILE = 16

# The pinned ThinkLite population contains only five numeric-percent answers.
# The remaining three slots from an otherwise even eight-per-family allocation
# are assigned to its two dominant answer forms.  This is part of the versioned
# selection algorithm, rather than a data-dependent fallback.
THINKLITE_ANSWER_FORM_QUOTAS: Mapping[str, int] = {
    "integer": 10,
    "decimal": 8,
    "fraction": 8,
    "percent": 5,
    "expression": 8,
    "yes-no": 8,
    "short-text": 9,
    "other": 8,
}

_SOURCE_ORDER = {
    SelectionSource.VSTAR.value: 0,
    SelectionSource.ARXIVQA.value: 1,
    SelectionSource.THINKLITE.value: 2,
}
_ARXIVQA_OPTION_TRANSFORM_VERSION = "arxivqa-canonical-options-v2"
_ARXIVQA_REMOVAL_REASONS = frozenset({"separator", "markdown_figure_heading"})

_INTEGER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)")
_DECIMAL_RE = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+|\.\d+)")
_FRACTION_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\s*/\s*"
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
)
_LATEX_FRACTION_RE = re.compile(r"\\(?:d|t)?frac\s*\{[^{}]+\}\s*\{[^{}]+\}")
_PERCENT_RE = re.compile(
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"\s*(?:%|\\%)"
)
_EXPRESSION_RE = re.compile(
    r"\\(?:sqrt|boxed|begin|sin|cos|tan|log|pi|times|div|pm|leq|geq|neq|cdot)"
    r"|[=×÷^<>√±]|\d\s*[+*/]\s*\d|\d\s+[−-]\s+\d"
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc
    return encoded.encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_canary_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a canary manifest with the project's canonical JSON rules."""

    return _canonical_json_bytes(manifest) + b"\n"


@dataclass(frozen=True, slots=True)
class T1CanarySelectionResult:
    """Selected canonical candidates and their content-bound audit manifest."""

    selected_candidates: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            canonical_canary_manifest_bytes(self.manifest)
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _CanaryItem:
    candidate: SelectionCandidate
    stratum: Mapping[str, Any]
    stratum_key: str
    selection_sha256: str


class _BottomK:
    """Order-independent reservoir retaining the k lowest selection hashes."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._heap: list[tuple[int, str, _CanaryItem]] = []

    def add(self, item: _CanaryItem) -> None:
        entry = (-int(item.selection_sha256, 16), item.selection_sha256, item)
        heapq.heappush(self._heap, entry)
        if len(self._heap) > self._limit:
            heapq.heappop(self._heap)

    def items(self) -> tuple[_CanaryItem, ...]:
        return tuple(sorted((entry[2] for entry in self._heap), key=_item_sort_key))


def _item_sort_key(item: _CanaryItem) -> tuple[str, str]:
    return item.stratum_key, item.selection_sha256


def _required_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _required_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence")
    return value


def _answer_length_bin(answer: Any) -> str:
    text = _required_string(answer, field_name="V* ground_truth")
    token_count = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    if token_count <= 2:
        return "tokens-1-2"
    if token_count <= 6:
        return "tokens-3-6"
    if token_count <= 15:
        return "tokens-7-15"
    return "tokens-16-plus"


def _bbox_relative_area_bin(candidate: SelectionCandidate) -> str:
    if not candidate.gt_regions:
        return "missing"
    image_area = int(candidate.image["width"]) * int(candidate.image["height"])
    smallest_relative_area = min(
        (right - left) * (bottom - top) / image_area
        for left, top, right, bottom in candidate.gt_regions
    )
    if smallest_relative_area < 0.001:
        return "lt-0.001"
    if smallest_relative_area < 0.01:
        return "0.001-to-0.01"
    if smallest_relative_area < 0.1:
        return "0.01-to-0.1"
    return "ge-0.1"


def classify_thinklite_answer_form(answer: Any) -> str:
    """Classify one ThinkLite ground truth into the frozen canary families."""

    text = " ".join(
        _required_string(answer, field_name="ThinkLite ground_truth").split()
    )
    folded = text.casefold().rstrip(".!")
    if folded in {"yes", "no", "true", "false"}:
        return "yes-no"
    if _PERCENT_RE.fullmatch(text):
        return "percent"
    if _INTEGER_RE.fullmatch(text):
        return "integer"
    if _DECIMAL_RE.fullmatch(text):
        return "decimal"
    if _FRACTION_RE.fullmatch(text) or _LATEX_FRACTION_RE.fullmatch(text):
        return "fraction"
    if _EXPRESSION_RE.search(text):
        return "expression"
    if len(text) <= 64 and len(text.split()) <= 4:
        return "short-text"
    return "other"


def _raw_label_form(raw_label: Any) -> str:
    text = _required_string(raw_label, field_name="ArxivQA raw_label")
    if re.match(r"^\[[A-Za-z]\](?:\s|$)", text):
        return "bracketed-letter"
    if re.match(r"^\([A-Za-z]\)(?:\s|$)", text):
        return "parenthesized-letter"
    if re.match(r"^[A-Za-z]\.(?:\s|$)", text):
        return "dot-suffix"
    if re.match(r"^[A-Za-z]\)(?:\s|$)", text):
        return "paren-suffix"
    if re.match(r"^[A-Za-z]:(?:\s|$)", text):
        return "colon-suffix"
    if re.fullmatch(r"[A-Za-z]", text):
        return "bare-letter"
    if re.match(r"^[A-Za-z]\s+", text):
        return "letter-plus-text"
    return "other"


def _arxivqa_stratum(candidate: SelectionCandidate) -> Mapping[str, Any]:
    metadata = _required_mapping(
        candidate.canonical_record.get("selection_metadata"),
        field_name="ArxivQA selection_metadata",
    )
    if metadata.get("option_transform_version") != _ARXIVQA_OPTION_TRANSFORM_VERSION:
        raise ValueError(
            "ArxivQA option_transform_version must be "
            f"{_ARXIVQA_OPTION_TRANSFORM_VERSION!r}"
        )

    options = _required_sequence(metadata.get("options"), field_name="ArxivQA options")
    raw_options = _required_sequence(
        metadata.get("raw_options"), field_name="ArxivQA raw_options"
    )
    source_indices = _required_sequence(
        metadata.get("source_option_indices"),
        field_name="ArxivQA source_option_indices",
    )
    removed_options = _required_sequence(
        metadata.get("removed_options"), field_name="ArxivQA removed_options"
    )
    option_count = metadata.get("option_count")
    if type(option_count) is not int or not 2 <= option_count <= 26:
        raise ValueError("ArxivQA option_count must be an integer in [2, 26]")
    if option_count != len(options) or option_count != len(source_indices):
        raise ValueError("ArxivQA option_count does not match retained options")
    if len(raw_options) != option_count + len(removed_options):
        raise ValueError("ArxivQA raw/retained/removed option counts do not reconcile")

    for index, option in enumerate(options):
        option = _required_string(option, field_name=f"ArxivQA options[{index}]")
        expected_prefix = f"{chr(ord('A') + index)}. "
        if not option.startswith(expected_prefix):
            raise ValueError("ArxivQA options must use canonical positional labels")
    for index, raw_option in enumerate(raw_options):
        _required_string(raw_option, field_name=f"ArxivQA raw_options[{index}]")

    retained_indices: list[int] = []
    for index, source_index in enumerate(source_indices):
        if type(source_index) is not int or not 0 <= source_index < len(raw_options):
            raise ValueError(f"ArxivQA source_option_indices[{index}] is invalid")
        retained_indices.append(source_index)
    if retained_indices != sorted(set(retained_indices)):
        raise ValueError("ArxivQA source_option_indices must be unique and sorted")

    removal_reasons: set[str] = set()
    removed_indices: list[int] = []
    for index, removed_value in enumerate(removed_options):
        removed = _required_mapping(
            removed_value, field_name=f"ArxivQA removed_options[{index}]"
        )
        source_index = removed.get("source_index")
        if type(source_index) is not int or not 0 <= source_index < len(raw_options):
            raise ValueError(
                f"ArxivQA removed_options[{index}].source_index is invalid"
            )
        raw_option = _required_string(
            removed.get("raw_option"),
            field_name=f"ArxivQA removed_options[{index}].raw_option",
        )
        if raw_option != raw_options[source_index].strip():
            raise ValueError(
                "ArxivQA removed option does not match its raw source entry"
            )
        reason = _required_string(
            removed.get("reason"),
            field_name=f"ArxivQA removed_options[{index}].reason",
        )
        if reason not in _ARXIVQA_REMOVAL_REASONS:
            raise ValueError(f"unknown ArxivQA removed-option reason: {reason!r}")
        removed_indices.append(source_index)
        removal_reasons.add(reason)
    if removed_indices != sorted(set(removed_indices)):
        raise ValueError("ArxivQA removed source indices must be unique and sorted")
    if set(retained_indices) | set(removed_indices) != set(range(len(raw_options))):
        raise ValueError(
            "ArxivQA retained and removed indices must partition raw options"
        )
    if set(retained_indices) & set(removed_indices):
        raise ValueError("ArxivQA retained and removed indices overlap")

    label_source_index = metadata.get("label_source_index")
    label_clean_index = metadata.get("label_clean_index")
    if type(label_clean_index) is not int or not 0 <= label_clean_index < option_count:
        raise ValueError("ArxivQA label_clean_index is invalid")
    if type(label_source_index) is not int or (
        label_source_index != retained_indices[label_clean_index]
    ):
        raise ValueError("ArxivQA label source/clean index mapping is invalid")
    expected_answer = chr(ord("A") + label_clean_index)
    if candidate.ground_truth != expected_answer:
        raise ValueError("ArxivQA ground_truth does not match label_clean_index")

    if not removal_reasons:
        removed_case = "none"
    elif removal_reasons == {"separator"}:
        removed_case = "separator"
    elif removal_reasons == {"markdown_figure_heading"}:
        removed_case = "markdown-figure-heading"
    else:
        removed_case = "mixed"
    return {
        "option_count": option_count,
        "removed_option_case": removed_case,
        "raw_label_form": _raw_label_form(metadata.get("raw_label")),
    }


def _selection_item(
    candidate: SelectionCandidate, stratum: Mapping[str, Any]
) -> _CanaryItem:
    normalized_stratum = json.loads(_canonical_json_bytes(stratum))
    stratum_key = _canonical_json_bytes(normalized_stratum).decode("utf-8")
    selection_sha256 = _sha256(
        {
            "algorithm_version": POLICY_SELECTION_CANARY_ALGORITHM_VERSION,
            "candidate_sha256": candidate.identity_sha256,
            "source": candidate.source.value,
            "stratum": normalized_stratum,
        }
    )
    return _CanaryItem(
        candidate=candidate,
        stratum=normalized_stratum,
        stratum_key=stratum_key,
        selection_sha256=selection_sha256,
    )


def _stratum_order_key(stratum_key: str) -> str:
    return _sha256(
        {
            "algorithm_version": POLICY_SELECTION_CANARY_ALGORITHM_VERSION,
            "stratum": json.loads(stratum_key),
        }
    )


def _coverage_then_stratified_select(
    items: Iterable[_CanaryItem],
    *,
    quota: int,
    coverage_axes: Sequence[str],
    preselected: Iterable[_CanaryItem] = (),
) -> tuple[_CanaryItem, ...]:
    available = {item.candidate.identity_sha256: item for item in items}
    if len(available) < quota:
        raise ValueError(
            f"stratified population has {len(available)} rows; needs {quota}"
        )
    chosen = {item.candidate.identity_sha256: item for item in preselected}
    if not set(chosen).issubset(available):
        raise ValueError("preselected row is absent from the stratified population")

    uncovered = {
        (axis, _canonical_json_bytes(item.stratum[axis]).decode("utf-8"))
        for item in available.values()
        for axis in coverage_axes
    }
    for item in chosen.values():
        for axis in coverage_axes:
            uncovered.discard(
                (axis, _canonical_json_bytes(item.stratum[axis]).decode("utf-8"))
            )
    while uncovered:
        candidates: list[tuple[int, str, _CanaryItem]] = []
        for identity, item in available.items():
            if identity in chosen:
                continue
            covered = {
                (axis, _canonical_json_bytes(item.stratum[axis]).decode("utf-8"))
                for axis in coverage_axes
            }
            gain = len(covered & uncovered)
            if gain:
                candidates.append((-gain, item.selection_sha256, item))
        if not candidates:
            raise ValueError("coverage axes cannot be satisfied")
        if len(chosen) >= quota:
            raise ValueError("coverage axes require more rows than the fixed quota")
        selected = min(candidates, key=lambda entry: (entry[0], entry[1]))[2]
        chosen[selected.candidate.identity_sha256] = selected
        for axis in coverage_axes:
            uncovered.discard(
                (
                    axis,
                    _canonical_json_bytes(selected.stratum[axis]).decode("utf-8"),
                )
            )

    remaining_by_stratum: dict[str, list[_CanaryItem]] = defaultdict(list)
    for identity, item in available.items():
        if identity not in chosen:
            remaining_by_stratum[item.stratum_key].append(item)
    for values in remaining_by_stratum.values():
        values.sort(key=lambda item: item.selection_sha256, reverse=True)

    ordered_strata = sorted(remaining_by_stratum, key=_stratum_order_key)
    while len(chosen) < quota:
        made_progress = False
        for stratum_key in ordered_strata:
            values = remaining_by_stratum[stratum_key]
            if not values:
                continue
            selected = values.pop()
            chosen[selected.candidate.identity_sha256] = selected
            made_progress = True
            if len(chosen) == quota:
                break
        if not made_progress:
            raise ValueError("stratified population was exhausted before quota")
    return tuple(sorted(chosen.values(), key=_item_sort_key))


def build_t1_canary_selection(
    candidates: Iterable[Mapping[str, Any]],
) -> T1CanarySelectionResult:
    """Select the outcome-independent, 64-per-source T1 canary population."""

    sample_ids: set[str] = set()
    candidate_hashes: set[str] = set()
    input_counts: Counter[str] = Counter()
    available_strata: Counter[tuple[str, str]] = Counter()

    vstar_buckets: dict[tuple[str, str], _BottomK] = {}
    arxivqa_buckets: dict[str, _BottomK] = {}
    thinklite_buckets = {
        family: _BottomK(quota)
        for family, quota in THINKLITE_ANSWER_FORM_QUOTAS.items()
    }
    arxivqa_j_rows = _BottomK(1)

    for record in candidates:
        candidate = SelectionCandidate.from_record(record)
        if candidate.source not in POLICY_SELECTION_PRIMARY_SOURCES:
            raise ValueError("T1 canary accepts only the frozen three-source pool")
        if candidate.sample_id in sample_ids:
            raise ValueError(f"duplicate candidate sample_id: {candidate.sample_id}")
        if candidate.identity_sha256 in candidate_hashes:
            raise ValueError(
                f"duplicate canonical candidate hash: {candidate.identity_sha256}"
            )
        sample_ids.add(candidate.sample_id)
        candidate_hashes.add(candidate.identity_sha256)
        source = candidate.source.value
        input_counts[source] += 1

        if candidate.source is SelectionSource.VSTAR:
            source_file = _required_string(
                candidate.provenance.get("source_file"),
                field_name="V* provenance.source_file",
            )
            if source_file not in VSTAR_CANARY_SOURCE_FILES:
                raise ValueError(f"unexpected V* source_file: {source_file!r}")
            stratum = {
                "source_file": source_file,
                "answer_length_bin": _answer_length_bin(candidate.ground_truth),
                "bbox_relative_area_bin": _bbox_relative_area_bin(candidate),
            }
            item = _selection_item(candidate, stratum)
            bucket_key = (source_file, item.stratum_key)
            bucket = vstar_buckets.setdefault(
                bucket_key, _BottomK(VSTAR_CANARY_PER_FILE)
            )
            bucket.add(item)
        elif candidate.source is SelectionSource.ARXIVQA:
            stratum = _arxivqa_stratum(candidate)
            item = _selection_item(candidate, stratum)
            arxivqa_buckets.setdefault(
                item.stratum_key, _BottomK(T1_CANARY_PER_SOURCE)
            ).add(item)
            if stratum["option_count"] == 10 and candidate.ground_truth == "J":
                arxivqa_j_rows.add(item)
        else:
            family = classify_thinklite_answer_form(candidate.ground_truth)
            stratum = {"answer_form": family}
            item = _selection_item(candidate, stratum)
            thinklite_buckets[family].add(item)
        available_strata[(source, item.stratum_key)] += 1

    missing_sources = [
        source.value
        for source in POLICY_SELECTION_PRIMARY_SOURCES
        if input_counts[source.value] < T1_CANARY_PER_SOURCE
    ]
    if missing_sources:
        raise ValueError(
            "each source needs at least 64 candidates; insufficient: "
            + ", ".join(missing_sources)
        )

    selected: list[_CanaryItem] = []
    for source_file in VSTAR_CANARY_SOURCE_FILES:
        file_items = [
            item
            for (candidate_source_file, _), bucket in vstar_buckets.items()
            if candidate_source_file == source_file
            for item in bucket.items()
        ]
        if len(file_items) < VSTAR_CANARY_PER_FILE:
            raise ValueError(
                f"V* source_file {source_file!r} needs at least "
                f"{VSTAR_CANARY_PER_FILE} candidates"
            )
        selected.extend(
            _coverage_then_stratified_select(
                file_items,
                quota=VSTAR_CANARY_PER_FILE,
                coverage_axes=("answer_length_bin", "bbox_relative_area_bin"),
            )
        )

    forced_j = arxivqa_j_rows.items()
    if not forced_j:
        raise ValueError("ArxivQA has no valid ten-option/J candidate")
    # The forced J row must survive even if its joint stratum contains more
    # than 64 rows with lower hashes than it does.
    arxivqa_item_by_hash = {
        item.candidate.identity_sha256: item
        for bucket in arxivqa_buckets.values()
        for item in bucket.items()
    }
    arxivqa_item_by_hash.update(
        {item.candidate.identity_sha256: item for item in forced_j}
    )
    selected.extend(
        _coverage_then_stratified_select(
            arxivqa_item_by_hash.values(),
            quota=T1_CANARY_PER_SOURCE,
            coverage_axes=("option_count", "removed_option_case", "raw_label_form"),
            preselected=forced_j,
        )
    )

    for family, quota in THINKLITE_ANSWER_FORM_QUOTAS.items():
        family_items = thinklite_buckets[family].items()
        if len(family_items) < quota:
            raise ValueError(
                f"ThinkLite answer form {family!r} has {len(family_items)} rows; "
                f"needs {quota}"
            )
        selected.extend(family_items)

    selected.sort(
        key=lambda item: (
            _SOURCE_ORDER[item.candidate.source.value],
            item.stratum_key,
            item.selection_sha256,
        )
    )
    if len(selected) != T1_CANARY_TOTAL:
        raise AssertionError("internal canary selection count mismatch")
    selected_hashes = [item.candidate.identity_sha256 for item in selected]
    if len(set(selected_hashes)) != T1_CANARY_TOTAL:
        raise AssertionError("internal duplicate canary candidate")

    selected_counts = Counter(item.candidate.source.value for item in selected)
    expected_counts = {
        source.value: T1_CANARY_PER_SOURCE
        for source in POLICY_SELECTION_PRIMARY_SOURCES
    }
    if dict(selected_counts) != expected_counts:
        raise AssertionError("internal per-source canary quota mismatch")

    selected_strata = Counter(
        (item.candidate.source.value, item.stratum_key) for item in selected
    )
    selected_records = [
        {
            "sample_id": item.candidate.sample_id,
            "sample_id_sha256": hashlib.sha256(
                item.candidate.sample_id.encode("utf-8")
            ).hexdigest(),
            "candidate_sha256": item.candidate.identity_sha256,
            "source": item.candidate.source.value,
            "stratum": item.stratum,
            "selection_sha256": item.selection_sha256,
        }
        for item in selected
    ]
    strata_records = [
        {
            "source": source,
            "stratum": json.loads(stratum_key),
            "available_count": available_count,
            "selected_count": selected_strata[(source, stratum_key)],
        }
        for (source, stratum_key), available_count in sorted(
            available_strata.items(),
            key=lambda entry: (
                _SOURCE_ORDER[entry[0][0]],
                _stratum_order_key(entry[0][1]),
            ),
        )
    ]
    descriptor: dict[str, Any] = {
        "schema_version": POLICY_SELECTION_CANARY_MANIFEST_SCHEMA,
        "selection_algorithm_version": POLICY_SELECTION_CANARY_ALGORITHM_VERSION,
        "selection_is_outcome_independent": True,
        "quotas": {
            "per_source": T1_CANARY_PER_SOURCE,
            "total": T1_CANARY_TOTAL,
            "vstar_per_source_file": VSTAR_CANARY_PER_FILE,
            "thinklite_answer_forms": dict(THINKLITE_ANSWER_FORM_QUOTAS),
        },
        "source_counts": {
            source.value: {
                "available": input_counts[source.value],
                "selected": selected_counts[source.value],
            }
            for source in POLICY_SELECTION_PRIMARY_SOURCES
        },
        "candidate_population_sha256": _sha256(sorted(candidate_hashes)),
        "strata": strata_records,
        "selected": selected_records,
        "selected_candidates_sha256": _sha256(selected_records),
    }
    manifest = {**descriptor, "content_sha256": _sha256(descriptor)}
    return T1CanarySelectionResult(
        selected_candidates=tuple(
            dict(item.candidate.canonical_record) for item in selected
        ),
        manifest=manifest,
    )
