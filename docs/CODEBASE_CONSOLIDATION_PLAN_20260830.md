# Codebase consolidation plan — 2026-08-30

Status: C0 is anchored by the remote annotated tag
`stabilization-c0-20260830`. C1's machine-disposition condition is satisfied,
the three C2 target facades are below 1,000 lines, and the four known C3 import
cycles are gone. C3 utility migration remains partial. C4 produced a read-only
inventory; it did not authorize or execute a deletion. The complete
post-consolidation hermetic CPU suite passes locally with 2,193 tests, five
explicit skips, and four non-failing warnings. Push and remote CI verification
remain required. A policy-v3 size ratchet now records all 29 remaining oversized
modules and rejects growth, slack, stale entries, or a relaxed baseline. This
plan does not authorize an experiment or rewrite historical evidence.

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
The remaining structural inventory still includes:

- 29 production modules above 1,000 lines, now bound to their exact current
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

The three named C2 targets are complete. Twenty-nine other production modules
still exceed 1,000 lines, but policy v3 now records each exact current ceiling,
owner, rationale, and next split. Candidate audit rejects new, growing, stale,
or slack exceptions; optional baseline comparison also rejects a newly added
exception or raised ceiling. These rows remain visible debt, not waivers.

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
  descriptor-rooted contracts, but the first production migration currently
  covers only `evaluation/result_registry.py`.
- `public_api_compat.py` owns implementation-to-facade identity rebinding for
  nine extracted modules and changes only implementation-owned functions,
  preventing shared `dataclasses` or `typing` helpers from being mutated by
  import order.

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
reference as unknown, and identifies two unanchored detached commits that must
receive exact durable tags before any later removal is considered. No action
code in that document is deletion approval.

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

Current state: the entry-point disposition condition, the three C2 targets,
the production-size exception ratchet, the known-cycle condition, and the
read-only C4 inventory are complete. The portable runtime/CLI closure,
remaining semantic-helper migrations, push, and remote CI are still open.
