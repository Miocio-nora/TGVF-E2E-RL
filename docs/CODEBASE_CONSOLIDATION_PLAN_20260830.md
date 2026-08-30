# Codebase consolidation plan — 2026-08-30

Status: C0 locally verified; C1-C4 remain a stabilization backlog. Remote CI
must still verify the pushed commit. This plan does not authorize an
experiment, delete a historical worktree, or rewrite historical evidence.

## Why the repository feels fragmented

The problem is not only the number of branches. A static inventory of the
stabilization line found:

- 245 Python modules below `src/tgvf_rl` and 79 inventoried execution/support
  surfaces across `src`, `tools`, and `spikes`;
- 31 production modules above 1,000 lines, led by
  `evaluation/policy_coredev.py` (3,414 lines at this audit point),
  `representation/training/internal_evaluation.py` (2,549 lines), and
  `policy/run_config.py` (2,491 lines);
- four multi-module import cycles;
- repeated local implementations of hashing, canonical JSON, file validation,
  and atomic publication across more than 100 files;
- historical RP/PRL launchers, current reusable implementation, evidence
  materializers, and manuscript-facing summaries sharing the same `tools`
  namespace;
- 62 local branches and 72 worktrees at the audit snapshot, including eight
  dirty worktrees that must be preserved.

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
started by the cleanup. C0 becomes the shared baseline only after the reviewed
commits are pushed and remote CI reproduces these checks.

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

### C4 — retire historical repository state

Only after C0–C3, generate an explicit worktree/branch retention table with
commit, merge status, unique commits, dirty state, artifact references, and a
proposed recoverable action. Deletion requires operator approval for every
exact path/ref. Never infer that a clean worktree is disposable.

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
