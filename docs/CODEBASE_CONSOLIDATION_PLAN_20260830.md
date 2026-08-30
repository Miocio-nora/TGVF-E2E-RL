# Codebase consolidation plan — 2026-08-30

Status: C0 is anchored by the remote annotated tag
`stabilization-c0-20260830`. C1's machine-disposition condition is satisfied,
the three C2 target facades are below 1,000 lines, and the four known C3 import
cycles are gone. All nine post-C2 priority decompositions are also complete,
reducing the current production-size inventory to 20 modules. C3 utility
migration remains partial. C4 produced a read-only inventory; it did not
authorize or execute a deletion. At verified code milestone `fd8da97`, the
latest complete hermetic CPU suite passes locally with 2,286 passed tests, five
explicit skips, and four non-failing warnings in 135.24 seconds. Remote CI
verification remains required.
Policy v3 revision 5 records the 20 remaining oversized modules and rejects
growth, slack, stale entries, or a relaxed baseline. This plan does not
authorize an experiment or rewrite historical evidence.

## Current macro milestone

The dated snapshots below remain historical anchors. The current local head
advances them as follows:

- production modules above 1,000 lines fell from 32 at C0, to 29 at the C2
  snapshot, and to 20 now;
- all nine priority decompositions after C2 are complete: exact replay, Policy
  weight sync, the native agent loop, representation configuration,
  representation checkpointing, Policy-selection runtime, answer-utility
  evaluation, Oracle-D utility, and distributed checkpointing;
- the four known multi-module strongly connected components fell to zero and
  remain guarded by a full-tree Tarjan regression test;
- execution/support surfaces fell from 82 at C0 to 79, and all 79 current
  surfaces have exactly one machine-checked classification;
- the production-size policy ratcheted from 29 exceptions to 20 at revision 5.

Verification of code milestone `fd8da97` is green: the repository-boundary
audit reports 64 visible debts (five evidence-only config roots, 25 machine
paths, 20 oversized modules, and 14 run-specific paths) and zero violations;
the control-plane audit reports 79 surfaces and zero violations. `tests/ops`
has 199 passing tests, the focused split and compatibility suites have 577
passing tests, and the complete hermetic CPU suite has 2,286 passed, five
skipped, and four non-failing warnings in 135.24 seconds. These are local
verification results; remote CI has not yet reproduced them.

This milestone does not close the runtime authority boundary. The experiment
policy still has `runtime_closure.launch_enabled=false` with all eight blocker
IDs present: `atomic_authority_transaction_missing`,
`child_environment_allowlist_missing`, `fd_bound_python_exec_missing`,
`immutable_runtime_code_package_missing`,
`policy_recursive_compile_closure_missing`,
`representation_eval_safe_artifact_missing`, `worker_member_claims_missing`,
and `worker_startup_envelope_missing`. C3 semantic-helper migration remains
partial. C4 remains an inventory only: its 72 worktrees, 62 branches, and seven
dirty worktrees must be preserved unless a later operator-approved action names
an exact path or ref.

## Why the repository feels fragmented

The problem is not only the number of branches. The C0 tag contained 245
Python modules below `src/tgvf_rl`, 82 execution/support surfaces, 32
production modules above 1,000 lines, and four multi-module import cycles. Its
three highest-risk files were `evaluation/policy_coredev.py` (3,420 lines),
`representation/training/internal_evaluation.py` (2,549 lines), and
`policy/run_config.py` (2,491 lines).

At the C2 completion snapshot `bd67d0a`, the corresponding measurements were
263 Python modules, 79 execution/support surfaces, 29 other modules above
1,000 lines, zero import cycles, and C2 facades of 902, 921, and 983 lines.
That snapshot's remaining structural inventory included:

- 29 production modules above 1,000 lines, then bound to their exact snapshot
  size, stable owner, rationale, and named next split by a fail-closed ratchet;
- repeated local implementations of hashing, canonical JSON, file validation,
  and atomic publication across more than 100 files;
- historical RP/PRL launchers, current reusable implementation, evidence
  materializers, and manuscript-facing summaries sharing the same `tools`
  namespace;
- 62 local branches and 72 worktrees at the C4 snapshot, including seven dirty
  worktrees that must be preserved.

These dimensions need different remedies. Deleting clean worktrees would not
fix module ownership, and moving large modules would not make historical score
contracts comparable.

## Canonical ownership map

| Concern | Canonical owner | Historical compatibility boundary |
|---|---|---|
| Public mutating commands | `tgvf_rl.cli` plus `tgvf_rl.ops` authorization | direct experiment launchers remain quarantined |
| Policy run contract | `tgvf_rl.policy.run_config` | legacy files under frozen config roots are read-only |
| Tool action/observation bytes | `tgvf_rl.protocol` and explicit environment contracts | generic86 and other old renderers require named legacy identities |
| Evaluation contract | canonical evaluation v2 schema and evaluator | old v1 manifests remain evidence, not launch templates |
| Result classification | result registry v2 | no registry-v2 row can be `golden` or produce a numeric delta |
| Repository controls | `configs/ops` and `tgvf_rl.ops` | RP/PRL scripts cannot mint their own launch authority |

New implementation must enter through the canonical owner. Compatibility
wrappers may delegate inward, but canonical modules must not import a run-named
RP/PRL script.

## Ordered cleanup

### C0 — establish a trustworthy baseline

The local stabilization baseline now has exact execution-surface and
legacy-config inventories, zero boundary violations, fail-closed launch and
result controls, zero test-collection errors, a passing 2,117-test hermetic CPU
collection (2,112 passed and five explicit dependency/environment skips),
an independently simulated clean runner (2,016 passed, 91 explicit optional
integration/environment skips, zero failures), Ruff, and no GPU process
started by the cleanup. The remote annotated tag
`stabilization-c0-20260830` preserves that baseline. Remote CI still has to
reproduce the later consolidation head rather than inheriting C0's result.

No module split or historical deletion belongs in C0. Combining security fixes
with broad moves would make review and provenance harder.

### C1 — collapse executable entry points

For every one of the 79 inventoried surfaces, choose exactly one disposition:

1. canonical CLI subcommand;
2. read-only inspection utility;
3. bounded, create-only artifact materializer;
4. historical compatibility wrapper that delegates to a canonical command;
5. statically constrained import-only support module;
6. permanent quarantine.

A wrapper can be removed only after all referenced configs, docs, tmux notes,
and artifact manifests have been migrated and a release/tag preserves its last
historical implementation. The execution-surface manifest must shrink in the
same commit as each removal; an allowlist expansion is not cleanup.

The first C1 batch removed three unreferenced quarantine shell wrappers in
`d6b5a1c` and shrank the manifest from 82 to 79 entries. Revision 3 assigns one
machine-checked disposition to every one of those 79 unique paths: two
canonical, two control-audit, 37 permanent quarantine, 13 mixed, 21 bounded
materializer, three read-only, and one import-only. This satisfies the C1
disposition condition; it does **not** mean all entry points are canonical CLI
commands or that runtime closure is enabled.

### C2 — split the three highest-risk modules

Split by responsibility while preserving public imports through temporary
re-exports and characterization tests:

- `evaluation/policy_coredev.py`: schema/loading, task materialization,
  inference, scoring, and receipt publication;
- `representation/training/internal_evaluation.py`: artifact loading,
  protocol adapters, metric computation, and reporting;
- `policy/run_config.py`: shared primitives, method-specific schemas, legacy
  readers, and canonical launch validation.

Each slice must leave the suite green and remove, rather than duplicate, the
old implementation. Target fewer than 1,000 lines per responsibility module.

C2 reached that target while retaining exact facade objects, historical pickle
coordinates, type hints, and characterization tests:

- the evaluation facade is 902 lines; extracted responsibility leaves are
  390, 634, 292, 813, and 884 lines;
- the representation facade is 921 lines; extracted artifact, contract, and
  native-runtime leaves are 102, 945, and 915 lines;
- the run-config facade is 983 lines; extracted schema, validation, canonical
  launch, and reward leaves are 511, 387, 838, and 259 lines.

The three named C2 targets are complete. At that snapshot, twenty-nine other
production modules still exceeded 1,000 lines, and policy v3 recorded each
exact ceiling, owner, rationale, and next split. The later nine-of-nine priority
decomposition series reduced that inventory to 20; policy revision 5 removes
the nine stale exceptions without adding an exception or raising a ceiling.
Candidate audit rejects new, growing, stale, or slack exceptions; optional
baseline comparison also rejects a newly added exception or raised ceiling.
The 20 surviving rows remain visible debt, not waivers.

### C3 — remove utility duplication and import cycles

Introduce small leaf modules for secure file reads, canonical JSON hashing,
and create-only/content-consistent publication. Migrate one subsystem at a
time, then delete its local helper. A shared helper is acceptable only when
its security and serialization semantics are identical; similarly named but
semantically different hashes must stay explicitly separate.

Break the four current import cycles by moving shared protocols toward leaf
modules, not by adding local imports as a permanent workaround:

- `framework.verl.data_bridge` ↔ `framework.verl.rollout_bridge`;
- `framework.verl.policy_live_runtime` ↔ `framework.verl.policy_runtime`;
- `evaluation.policy_coredev` ↔ `evaluation.policy_full_model_snapshot`;
- image-axis `streaming` ↔ `trainer`.

All four cycles are now removed and a full-tree Tarjan SCC regression test
prevents their silent return. Utility consolidation is intentionally partial:

- six byte-level canonical-JSON helpers and three digest helpers with exact
  semantics now delegate to `artifact_contracts.py`;
- create-only/content-consistent publication has one implementation and three
  production consumers (`policy_benchmark_config.py`,
  `policy_benchmark_scoring.py`, and `internal_evaluation_artifact.py`);
- `secure_file_read.py` defines separate leaf, absolute-chain, and
  descriptor-rooted contracts. Production migrations now include
  `evaluation/result_registry.py`, descriptor-bound Policy weight-snapshot
  reads, and representation-training configuration and external-file
  reads/probes. This does not claim that every repository reader or directory
  trust boundary has been migrated; a metadata-only probe also does not make a
  later runtime reopen an immutable descriptor binding.
- `public_api_compat.py` owns implementation-to-facade identity rebinding and
  changes only implementation-owned functions, preventing shared
  `dataclasses` or `typing` helpers from being mutated by import order. Its
  opt-in leaf-first annotation helper resolves only a moved class's own
  postponed annotations against explicit leaf globals before module rebinding;
  fake-package/import-order tests preserve type-hint resolution, exact object
  identity, historical module coordinates, and pickle compatibility.
- annotation freezing is deliberately not an automatic graph rewrite: future
  extracted annotated classes need an explicit opt-in and characterization
  test, and facade hooks that must remain late-bound still belong in the facade.

Names alone are not evidence of equivalent security semantics, so the
remaining readers and publishers must be migrated one reviewed group at a
time.

### C4 — retire historical repository state

Only after C0–C3, generate an explicit worktree/branch retention table with
commit, merge status, unique commits, dirty state, artifact references, and a
proposed recoverable action. Deletion requires operator approval for every
exact path/ref. Never infer that a clean worktree is disposable.

The frozen read-only inventory is
[`WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md`](WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md).
It records 72 worktrees and 62 branches at `05e32ea`, treats every artifact
reference as unknown, preserves all seven dirty worktrees, and identifies two
unanchored detached commits that must receive exact durable tags before any
later removal is considered. No action code in that document is deletion
approval.

## Definition of done

Consolidation is complete when:

- one documented public CLI owns all new mutating runs;
- every remaining direct entry point has one machine-checked disposition;
- canonical configs are portable, versioned, and runnable without a private
  worktree path;
- no production module exceeds 1,000 lines without a recorded exception;
- the four known import cycles are gone;
- common artifact primitives have one tested implementation per semantic
  contract;
- historical evidence remains byte-preserved and cannot be mistaken for a new
  comparable result;
- the full hermetic CPU suite and repository audits run in CI.

Current state: the entry-point disposition condition, the three C2 targets, all
nine post-C2 priority decompositions, the revision-5 production-size exception
ratchet, the known-cycle condition, and the read-only C4 inventory are
complete. The portable runtime/CLI closure remains disabled by its eight named
blockers; remaining semantic-helper migrations and remote CI are also still
open.
