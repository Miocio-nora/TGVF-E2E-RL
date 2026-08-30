"""Fail-closed boundaries between neutral code and historical run evidence."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .module_size_ratchet import (
    PRODUCTION_MODULE_LINE_LIMIT,
    ModuleSizeFinding,
    ModuleSizePolicyError,
    ProductionModuleSizeException,
    audit_production_module_sizes,
    compare_module_size_exception_ratchet,
    load_production_module_size_exceptions,
)

REPOSITORY_BOUNDARY_POLICY_SCHEMA = "tgvf-repository-boundary-policy-v3"
REPOSITORY_BOUNDARY_AUDIT_SCHEMA = "tgvf-repository-boundary-audit-v3"

SOURCE_ROOT = "src/tgvf_rl"
TOOLS_ROOT = "tools"
CANONICAL_CONFIG_ROOTS = (
    "configs/canonical/evaluation",
    "configs/canonical/policy",
    "configs/canonical/representation",
    "configs/ops",
)
EVIDENCE_ONLY_CONFIG_ROOTS = (
    "configs/evaluation",
    "configs/overnight",
    "configs/policy",
    "configs/representation",
    "configs/smoke",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_SPECIFIC_PATH_RE = re.compile(r"(?:^|[^a-z0-9])(?:prl|rp)[_-]?\d+", re.IGNORECASE)
_MACHINE_PATH_RE = re.compile(
    r"/(?:nvmesv|home/[A-Za-z0-9._-]+)"
    r"(?:/[A-Za-z0-9._~+@%=:,-]*)*"
)
_IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__"})
_IGNORED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
_CONTENT_TREE_HASH_DOMAIN = b"tgvf-evidence-config-content-tree-v1\0"


class RepositoryBoundaryError(RuntimeError):
    """Raised when the boundary policy itself cannot be trusted."""


@dataclass(frozen=True, slots=True)
class EvidenceConfigInventory:
    path: str
    file_count: int
    relative_paths_sha256: str
    content_tree_sha256: str

    def as_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "file_count": self.file_count,
            "relative_paths_sha256": self.relative_paths_sha256,
            "content_tree_sha256": self.content_tree_sha256,
        }


@dataclass(frozen=True, slots=True)
class MachinePathDebt:
    path: str
    literal_sha256: str
    count: int

    def as_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "literal_sha256": self.literal_sha256,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class RepositoryBoundaryPolicy:
    policy_id: str
    revision: int
    run_specific_code_allowlist: tuple[str, ...]
    machine_path_debt_allowlist: tuple[MachinePathDebt, ...]
    evidence_only_config_inventories: tuple[EvidenceConfigInventory, ...]
    production_module_size_exceptions: tuple[ProductionModuleSizeException, ...]
    schema_version: str = REPOSITORY_BOUNDARY_POLICY_SCHEMA


@dataclass(frozen=True, slots=True)
class BoundaryFinding:
    kind: str
    path: str
    message: str
    evidence: Mapping[str, object]

    def as_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class RepositoryBoundaryAudit:
    repository_root: Path
    policy_id: str
    policy_revision: int
    policy_sha256: str
    debts: tuple[BoundaryFinding, ...]
    violations: tuple[BoundaryFinding, ...]
    baseline_policy_sha256: str | None = None
    schema_version: str = REPOSITORY_BOUNDARY_AUDIT_SCHEMA

    @property
    def status(self) -> str:
        return "pass" if not self.violations else "blocked"

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "repository_root": str(self.repository_root),
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "policy_sha256": self.policy_sha256,
            "baseline_policy_sha256": self.baseline_policy_sha256,
            "summary": {
                "debt_count": len(self.debts),
                "violation_count": len(self.violations),
                "debt_kinds": _kind_counts(self.debts),
                "violation_kinds": _kind_counts(self.violations),
            },
            "debts": [finding.as_record() for finding in self.debts],
            "violations": [finding.as_record() for finding in self.violations],
        }


def relative_path_inventory_sha256(relative_paths: Sequence[str]) -> str:
    """Hash one sorted, duplicate-free relative-path inventory."""

    normalized = tuple(sorted(relative_paths))
    if len(set(normalized)) != len(normalized):
        raise RepositoryBoundaryError("relative-path inventory contains duplicates")
    for value in normalized:
        _repository_relative_path(value, field="inventory path")
    payload = "".join(f"{value}\n" for value in normalized).encode("utf-8")
    return sha256(payload).hexdigest()


def content_tree_inventory_sha256(entries: Sequence[tuple[str, bytes]]) -> str:
    """Hash sorted relative paths and exact file bytes with unambiguous framing."""

    normalized = tuple(sorted(entries, key=lambda entry: entry[0]))
    relative_paths = tuple(entry[0] for entry in normalized)
    if len(set(relative_paths)) != len(relative_paths):
        raise RepositoryBoundaryError("content-tree inventory contains duplicate paths")

    digest = sha256()
    digest.update(_CONTENT_TREE_HASH_DOMAIN)
    digest.update(len(normalized).to_bytes(8, byteorder="big", signed=False))
    for relative, content in normalized:
        _repository_relative_path(relative, field="content-tree inventory path")
        if not isinstance(content, bytes):
            raise RepositoryBoundaryError(
                "content-tree inventory contents must be exact bytes"
            )
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, byteorder="big", signed=False))
        digest.update(content)
    return digest.hexdigest()


def load_repository_boundary_policy(path: str | Path) -> RepositoryBoundaryPolicy:
    """Load the strict, portable debt baseline without auditing the repository."""

    policy_path = Path(path)
    if not policy_path.is_absolute():
        policy_path = policy_path.absolute()
    if policy_path.is_symlink() or not policy_path.is_file():
        raise RepositoryBoundaryError(
            "repository boundary policy must be an existing non-symlink file"
        )
    try:
        raw = policy_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepositoryBoundaryError(
            "repository boundary policy must be strict UTF-8 JSON"
        ) from error
    expected_fields = {
        "schema_version",
        "policy_id",
        "revision",
        "source_root",
        "tools_root",
        "canonical_config_roots",
        "evidence_only_config_inventories",
        "run_specific_code_allowlist",
        "machine_path_debt_allowlist",
        "production_module_line_limit",
        "production_module_size_exceptions",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RepositoryBoundaryError(
            "repository boundary policy has an unexpected field set"
        )
    if payload["schema_version"] != REPOSITORY_BOUNDARY_POLICY_SCHEMA:
        raise RepositoryBoundaryError("repository boundary policy schema differs")
    policy_id = payload["policy_id"]
    revision = payload["revision"]
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise RepositoryBoundaryError("repository boundary policy_id is invalid")
    if type(revision) is not int or revision <= 0:
        raise RepositoryBoundaryError("repository boundary revision must be positive")
    if payload["source_root"] != SOURCE_ROOT or payload["tools_root"] != TOOLS_ROOT:
        raise RepositoryBoundaryError("repository source/tools roots differ")
    if payload["production_module_line_limit"] != PRODUCTION_MODULE_LINE_LIMIT:
        raise RepositoryBoundaryError(
            "production module line limit must remain exactly "
            f"{PRODUCTION_MODULE_LINE_LIMIT}"
        )
    canonical_roots = _strict_string_list(
        payload["canonical_config_roots"], field="canonical_config_roots"
    )
    if canonical_roots != CANONICAL_CONFIG_ROOTS:
        raise RepositoryBoundaryError("canonical config roots differ")

    evidence_payload = payload["evidence_only_config_inventories"]
    if not isinstance(evidence_payload, list):
        raise RepositoryBoundaryError("evidence_only_config_inventories must be a list")
    inventories: list[EvidenceConfigInventory] = []
    for index, value in enumerate(evidence_payload):
        if not isinstance(value, dict) or set(value) != {
            "path",
            "file_count",
            "relative_paths_sha256",
            "content_tree_sha256",
        }:
            raise RepositoryBoundaryError(
                f"evidence inventory {index} has an unexpected field set"
            )
        inventory_path = _repository_relative_path(
            value["path"], field=f"evidence inventory {index} path"
        )
        file_count = value["file_count"]
        path_digest = value["relative_paths_sha256"]
        content_digest = value["content_tree_sha256"]
        if type(file_count) is not int or file_count < 0:
            raise RepositoryBoundaryError(
                f"evidence inventory {index} file_count is invalid"
            )
        _require_sha256(
            path_digest,
            field=f"evidence inventory {index} relative-path digest",
        )
        _require_sha256(
            content_digest,
            field=f"evidence inventory {index} content-tree digest",
        )
        inventories.append(
            EvidenceConfigInventory(
                inventory_path,
                file_count,
                path_digest,
                content_digest,
            )
        )
    inventories.sort(key=lambda item: item.path)
    if tuple(item.path for item in inventories) != EVIDENCE_ONLY_CONFIG_ROOTS:
        raise RepositoryBoundaryError("evidence-only config roots differ")

    run_allowlist = list(
        _strict_string_list(
            payload["run_specific_code_allowlist"],
            field="run_specific_code_allowlist",
        )
    )
    for index, value in enumerate(run_allowlist):
        relative = _repository_relative_path(
            value, field=f"run-specific allowlist path {index}"
        )
        if not _is_under(relative, SOURCE_ROOT, TOOLS_ROOT):
            raise RepositoryBoundaryError(
                "run-specific allowlist paths must remain under source or tools"
            )
        if not _RUN_SPECIFIC_PATH_RE.search(relative):
            raise RepositoryBoundaryError(
                f"run-specific allowlist path lacks a run identifier: {relative}"
            )
        run_allowlist[index] = relative
    if len(set(run_allowlist)) != len(run_allowlist):
        raise RepositoryBoundaryError("run-specific code allowlist contains duplicates")

    machine_payload = payload["machine_path_debt_allowlist"]
    if not isinstance(machine_payload, list):
        raise RepositoryBoundaryError("machine_path_debt_allowlist must be a list")
    machine_allowlist: list[MachinePathDebt] = []
    for index, value in enumerate(machine_payload):
        if not isinstance(value, dict) or set(value) != {
            "path",
            "literal_sha256",
            "count",
        }:
            raise RepositoryBoundaryError(
                f"machine-path debt {index} has an unexpected field set"
            )
        debt_path = _repository_relative_path(
            value["path"], field=f"machine-path debt {index} path"
        )
        digest = value["literal_sha256"]
        count = value["count"]
        if not _is_under(debt_path, SOURCE_ROOT, TOOLS_ROOT):
            raise RepositoryBoundaryError(
                "machine-path debt must remain under source or tools"
            )
        _require_sha256(digest, field=f"machine-path debt {index} digest")
        if type(count) is not int or count <= 0:
            raise RepositoryBoundaryError(
                f"machine-path debt {index} count must be positive"
            )
        machine_allowlist.append(MachinePathDebt(debt_path, digest, count))
    machine_keys = tuple((item.path, item.literal_sha256) for item in machine_allowlist)
    if len(set(machine_keys)) != len(machine_keys):
        raise RepositoryBoundaryError("machine-path debt allowlist contains duplicates")

    try:
        module_size_exceptions = load_production_module_size_exceptions(
            payload["production_module_size_exceptions"],
            source_root=SOURCE_ROOT,
        )
    except ModuleSizePolicyError as error:
        raise RepositoryBoundaryError(str(error)) from error

    return RepositoryBoundaryPolicy(
        policy_id=policy_id,
        revision=revision,
        run_specific_code_allowlist=tuple(sorted(run_allowlist)),
        machine_path_debt_allowlist=tuple(
            sorted(
                machine_allowlist,
                key=lambda item: (item.path, item.literal_sha256),
            )
        ),
        evidence_only_config_inventories=tuple(inventories),
        production_module_size_exceptions=module_size_exceptions,
    )


def compare_production_module_size_policies(
    baseline: RepositoryBoundaryPolicy,
    candidate: RepositoryBoundaryPolicy,
) -> tuple[BoundaryFinding, ...]:
    """Return monotonicity violations between parsed base and candidate policies."""

    return tuple(
        _boundary_finding(finding)
        for finding in compare_module_size_exception_ratchet(
            baseline.production_module_size_exceptions,
            candidate.production_module_size_exceptions,
        )
    )


def audit_repository_boundaries(
    repository_root: str | Path,
    policy_path: str | Path,
    *,
    baseline_policy_path: str | Path | None = None,
) -> RepositoryBoundaryAudit:
    """Audit current files against the explicit debt baseline."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RepositoryBoundaryError("repository_root must be a directory")
    policy_file = Path(policy_path)
    if not policy_file.is_absolute():
        policy_file = root / policy_file
    policy = load_repository_boundary_policy(policy_file)
    policy_sha256 = sha256(policy_file.read_bytes()).hexdigest()
    baseline_policy_sha256: str | None = None
    debts: list[BoundaryFinding] = []
    violations: list[BoundaryFinding] = []
    if baseline_policy_path is not None:
        baseline_file = Path(baseline_policy_path)
        if not baseline_file.is_absolute():
            baseline_file = root / baseline_file
        baseline_policy = load_repository_boundary_policy(baseline_file)
        baseline_policy_sha256 = sha256(baseline_file.read_bytes()).hexdigest()
        violations.extend(
            compare_production_module_size_policies(baseline_policy, policy)
        )

    source_files = _tree_files(
        root,
        SOURCE_ROOT,
        violations=violations,
        required=True,
    )
    tool_files = _tree_files(
        root,
        TOOLS_ROOT,
        violations=violations,
        required=True,
    )
    code_files = tuple(sorted((*source_files, *tool_files)))

    actual_run_specific = {
        path.relative_to(root).as_posix()
        for path in code_files
        if _RUN_SPECIFIC_PATH_RE.search(path.relative_to(root).as_posix())
    }
    expected_run_specific = set(policy.run_specific_code_allowlist)
    for relative in sorted(actual_run_specific & expected_run_specific):
        debts.append(
            BoundaryFinding(
                kind="run_specific_code",
                path=relative,
                message="existing run-specific code remains quarantined",
                evidence={},
            )
        )
    for relative in sorted(actual_run_specific - expected_run_specific):
        violations.append(
            BoundaryFinding(
                kind="new_run_specific_code",
                path=relative,
                message="new source/tool code must use a neutral capability name",
                evidence={},
            )
        )
    for relative in sorted(expected_run_specific - actual_run_specific):
        violations.append(
            BoundaryFinding(
                kind="stale_run_specific_allowlist",
                path=relative,
                message="removed run-specific debt must also be removed from policy",
                evidence={},
            )
        )

    decoded: dict[str, str] = {}
    actual_machine_paths: Counter[tuple[str, str]] = Counter()
    machine_literals: dict[tuple[str, str], str] = {}
    for path in code_files:
        relative = path.relative_to(root).as_posix()
        text = _read_canonical_text(path, relative=relative, violations=violations)
        if text is None:
            continue
        decoded[relative] = text
        for literal in _MACHINE_PATH_RE.findall(text):
            digest = sha256(literal.encode("utf-8")).hexdigest()
            key = (relative, digest)
            actual_machine_paths[key] += 1
            machine_literals[key] = literal

    observed_module_line_counts = {
        relative: len(text.splitlines())
        for relative, text in decoded.items()
        if _is_under(relative, SOURCE_ROOT) and relative.endswith(".py")
    }
    module_debts, module_violations = audit_production_module_sizes(
        observed_module_line_counts,
        policy.production_module_size_exceptions,
    )
    debts.extend(_boundary_finding(item) for item in module_debts)
    violations.extend(_boundary_finding(item) for item in module_violations)

    expected_machine_paths = {
        (item.path, item.literal_sha256): item.count
        for item in policy.machine_path_debt_allowlist
    }
    for key in sorted(actual_machine_paths.keys() & expected_machine_paths.keys()):
        observed = actual_machine_paths[key]
        expected = expected_machine_paths[key]
        relative, digest = key
        debts.append(
            BoundaryFinding(
                kind="machine_absolute_path",
                path=relative,
                message="existing machine-specific path remains registered debt",
                evidence={
                    "literal": machine_literals[key],
                    "literal_sha256": digest,
                    "registered_count": expected,
                    "observed_count": observed,
                },
            )
        )
        if observed != expected:
            violations.append(
                BoundaryFinding(
                    kind="machine_path_count_drift",
                    path=relative,
                    message="registered machine-path occurrence count changed",
                    evidence={
                        "literal_sha256": digest,
                        "registered_count": expected,
                        "observed_count": observed,
                    },
                )
            )
    for key in sorted(actual_machine_paths.keys() - expected_machine_paths.keys()):
        relative, digest = key
        violations.append(
            BoundaryFinding(
                kind="new_machine_absolute_path",
                path=relative,
                message="new source/tool machine-specific paths are forbidden",
                evidence={
                    "literal": machine_literals[key],
                    "literal_sha256": digest,
                    "observed_count": actual_machine_paths[key],
                },
            )
        )
    for key in sorted(expected_machine_paths.keys() - actual_machine_paths.keys()):
        relative, digest = key
        violations.append(
            BoundaryFinding(
                kind="stale_machine_path_allowlist",
                path=relative,
                message="removed machine-path debt must also be removed from policy",
                evidence={
                    "literal_sha256": digest,
                    "registered_count": expected_machine_paths[key],
                },
            )
        )

    run_specific_modules = {
        _source_module_name(relative)
        for relative in policy.run_specific_code_allowlist
        if _is_under(relative, SOURCE_ROOT) and relative.endswith(".py")
    }
    for path in source_files:
        relative = path.relative_to(root).as_posix()
        if not relative.endswith(".py") or relative in expected_run_specific:
            continue
        text = decoded.get(relative)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as error:
            violations.append(
                BoundaryFinding(
                    kind="source_parse_error",
                    path=relative,
                    message="neutral source must parse before imports can be audited",
                    evidence={"line": error.lineno, "offset": error.offset},
                )
            )
            continue
        importer_module = _source_module_name(relative)
        is_package = relative.endswith("/__init__.py")
        imported = _import_targets(
            tree,
            importer_module=importer_module,
            importer_is_package=is_package,
        )
        forbidden = sorted(
            target
            for target in imported
            if any(
                target == run_module or target.startswith(f"{run_module}.")
                for run_module in run_specific_modules
            )
        )
        if forbidden:
            violations.append(
                BoundaryFinding(
                    kind="neutral_imports_run_specific",
                    path=relative,
                    message="neutral source must not import run-specific modules",
                    evidence={"imports": forbidden},
                )
            )

    config_files = _tree_files(
        root,
        "configs",
        violations=violations,
        required=True,
    )
    for config_root in CANONICAL_CONFIG_ROOTS:
        directory = root / config_root
        if directory.is_symlink():
            violations.append(
                BoundaryFinding(
                    kind="symlink_boundary",
                    path=config_root,
                    message="canonical configuration roots must not be symlinks",
                    evidence={},
                )
            )
        elif not directory.is_dir():
            violations.append(
                BoundaryFinding(
                    kind="missing_boundary_root",
                    path=config_root,
                    message="required canonical configuration root is missing",
                    evidence={},
                )
            )
    for path in config_files:
        relative = path.relative_to(root).as_posix()
        if any(
            _is_under(relative, config_root) for config_root in CANONICAL_CONFIG_ROOTS
        ):
            if _RUN_SPECIFIC_PATH_RE.search(relative):
                violations.append(
                    BoundaryFinding(
                        kind="run_specific_canonical_config",
                        path=relative,
                        message="canonical configs must use neutral method names",
                        evidence={},
                    )
                )
            text = _read_canonical_text(
                path,
                relative=relative,
                violations=violations,
            )
            if text is not None:
                literals = sorted(set(_MACHINE_PATH_RE.findall(text)))
                if literals:
                    violations.append(
                        BoundaryFinding(
                            kind="machine_path_in_canonical_config",
                            path=relative,
                            message="canonical configs must use portable bindings",
                            evidence={"literals": literals},
                        )
                    )
        elif not any(
            _is_under(relative, config_root)
            for config_root in EVIDENCE_ONLY_CONFIG_ROOTS
        ):
            violations.append(
                BoundaryFinding(
                    kind="unclassified_config",
                    path=relative,
                    message="config is outside canonical and evidence-only roots",
                    evidence={},
                )
            )

    inventory_by_path = {
        item.path: item for item in policy.evidence_only_config_inventories
    }
    for config_root in EVIDENCE_ONLY_CONFIG_ROOTS:
        files = _tree_files(
            root,
            config_root,
            violations=violations,
            required=True,
        )
        base = root / config_root
        relative_paths = tuple(
            sorted(path.relative_to(base).as_posix() for path in files)
        )
        observed_path_digest = relative_path_inventory_sha256(relative_paths)
        content_entries: list[tuple[str, bytes]] = []
        content_read_failed = False
        for path, relative in zip(files, relative_paths, strict=True):
            try:
                content_entries.append((relative, path.read_bytes()))
            except OSError:
                content_read_failed = True
                violations.append(
                    BoundaryFinding(
                        kind="unreadable_evidence_config",
                        path=path.relative_to(root).as_posix(),
                        message=(
                            "evidence-only config bytes must be readable for hashing"
                        ),
                        evidence={},
                    )
                )
        observed_content_digest = (
            None
            if content_read_failed
            else content_tree_inventory_sha256(content_entries)
        )
        expected = inventory_by_path[config_root]
        debts.append(
            BoundaryFinding(
                kind="evidence_only_config_root",
                path=config_root,
                message="historical configs are evidence-only, not canonical inputs",
                evidence={
                    "registered_file_count": expected.file_count,
                    "observed_file_count": len(relative_paths),
                    "registered_relative_paths_sha256": (
                        expected.relative_paths_sha256
                    ),
                    "observed_relative_paths_sha256": observed_path_digest,
                    "registered_content_tree_sha256": expected.content_tree_sha256,
                    "observed_content_tree_sha256": observed_content_digest,
                },
            )
        )
        if (
            len(relative_paths) != expected.file_count
            or observed_path_digest != expected.relative_paths_sha256
            or observed_content_digest != expected.content_tree_sha256
        ):
            violations.append(
                BoundaryFinding(
                    kind="evidence_config_inventory_drift",
                    path=config_root,
                    message=(
                        "evidence-only config names and bytes are frozen; "
                        "use configs/canonical"
                    ),
                    evidence={
                        "registered_file_count": expected.file_count,
                        "observed_file_count": len(relative_paths),
                        "registered_relative_paths_sha256": (
                            expected.relative_paths_sha256
                        ),
                        "observed_relative_paths_sha256": observed_path_digest,
                        "registered_content_tree_sha256": (
                            expected.content_tree_sha256
                        ),
                        "observed_content_tree_sha256": observed_content_digest,
                    },
                )
            )

    return RepositoryBoundaryAudit(
        repository_root=root,
        policy_id=policy.policy_id,
        policy_revision=policy.revision,
        policy_sha256=policy_sha256,
        debts=tuple(sorted(debts, key=_finding_sort_key)),
        violations=tuple(sorted(violations, key=_finding_sort_key)),
        baseline_policy_sha256=baseline_policy_sha256,
    )


def _tree_files(
    root: Path,
    relative_root: str,
    *,
    violations: list[BoundaryFinding],
    required: bool,
) -> tuple[Path, ...]:
    directory = root / relative_root
    if directory.is_symlink():
        violations.append(
            BoundaryFinding(
                kind="symlink_boundary",
                path=relative_root,
                message="audited roots must not be symlinks",
                evidence={},
            )
        )
        return ()
    if not directory.is_dir():
        if required:
            violations.append(
                BoundaryFinding(
                    kind="missing_boundary_root",
                    path=relative_root,
                    message="required repository boundary root is missing",
                    evidence={},
                )
            )
        return ()
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            if name in _IGNORED_DIRECTORY_NAMES:
                continue
            if child.is_symlink():
                violations.append(
                    BoundaryFinding(
                        kind="symlink_boundary",
                        path=child.relative_to(root).as_posix(),
                        message="audited trees must not contain symlink directories",
                        evidence={},
                    )
                )
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            child = current_path / name
            if child.suffix in _IGNORED_FILE_SUFFIXES:
                continue
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                violations.append(
                    BoundaryFinding(
                        kind="symlink_boundary",
                        path=relative,
                        message="audited trees must not contain symlink files",
                        evidence={},
                    )
                )
            elif not child.is_file():
                violations.append(
                    BoundaryFinding(
                        kind="non_regular_boundary_file",
                        path=relative,
                        message="audited trees may contain only regular files",
                        evidence={},
                    )
                )
            else:
                files.append(child)
    return tuple(sorted(files))


def _read_canonical_text(
    path: Path,
    *,
    relative: str,
    violations: list[BoundaryFinding],
) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        violations.append(
            BoundaryFinding(
                kind="non_utf8_canonical_file",
                path=relative,
                message="canonical source/tools/config files must be UTF-8 text",
                evidence={},
            )
        )
        return None


def _source_module_name(relative: str) -> str:
    path = PurePosixPath(relative)
    if not _is_under(relative, SOURCE_ROOT) or path.suffix != ".py":
        raise RepositoryBoundaryError(f"not a Python source module: {relative}")
    module_path = path.relative_to("src").with_suffix("")
    parts = module_path.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_targets(
    tree: ast.AST,
    *,
    importer_module: str,
    importer_is_package: bool,
) -> set[str]:
    targets: set[str] = set()
    current_package = (
        importer_module if importer_is_package else importer_module.rpartition(".")[0]
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = current_package.split(".") if current_package else []
                remove = node.level - 1
                if remove > len(package_parts):
                    continue
                base_parts = package_parts[: len(package_parts) - remove]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                targets.add(base)
            for alias in node.names:
                if alias.name != "*":
                    targets.add(".".join(part for part in (base, alias.name) if part))
    return targets


def _repository_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RepositoryBoundaryError(f"{field} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RepositoryBoundaryError(f"{field} must be a canonical relative path")
    return value


def _strict_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RepositoryBoundaryError(f"{field} must be a list of strings")
    if len(set(value)) != len(value):
        raise RepositoryBoundaryError(f"{field} contains duplicates")
    return tuple(value)


def _is_under(relative: str, *roots: str) -> bool:
    path = PurePosixPath(relative)
    return any(
        path == PurePosixPath(root) or path.is_relative_to(root) for root in roots
    )


def _require_sha256(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RepositoryBoundaryError(f"{field} must be a lowercase SHA256")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RepositoryBoundaryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _kind_counts(findings: Sequence[BoundaryFinding]) -> dict[str, int]:
    return dict(sorted(Counter(item.kind for item in findings).items()))


def _finding_sort_key(finding: BoundaryFinding) -> tuple[str, str, str]:
    return finding.kind, finding.path, finding.message


def _boundary_finding(finding: ModuleSizeFinding) -> BoundaryFinding:
    return BoundaryFinding(
        kind=finding.kind,
        path=finding.path,
        message=finding.message,
        evidence=finding.evidence,
    )


__all__ = [
    "CANONICAL_CONFIG_ROOTS",
    "EVIDENCE_ONLY_CONFIG_ROOTS",
    "REPOSITORY_BOUNDARY_AUDIT_SCHEMA",
    "REPOSITORY_BOUNDARY_POLICY_SCHEMA",
    "BoundaryFinding",
    "EvidenceConfigInventory",
    "MachinePathDebt",
    "PRODUCTION_MODULE_LINE_LIMIT",
    "ProductionModuleSizeException",
    "RepositoryBoundaryAudit",
    "RepositoryBoundaryError",
    "RepositoryBoundaryPolicy",
    "audit_repository_boundaries",
    "compare_production_module_size_policies",
    "content_tree_inventory_sha256",
    "load_repository_boundary_policy",
    "relative_path_inventory_sha256",
]
