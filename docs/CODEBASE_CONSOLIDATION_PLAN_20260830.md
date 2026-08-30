# Codebase consolidation plan — 2026-08-30

Status: C0 is anchored by the remote annotated tag
`stabilization-c0-20260830`. C1's machine-disposition condition is satisfied,
the three C2 target facades are below 1,000 lines, and the four known C3 import
cycles are gone. All nine post-C2 priority decompositions are also complete,
and the later result-registry and CLI/authorization splits reduce the current
production-size inventory to 17 modules. C3 utility migration remains partial.
C4 produced a read-only inventory; it did not authorize or execute a deletion.
The final hermetic CPU rerun has 2,337 passed tests, five explicit skips, and
four non-failing warnings in 140.25 seconds, including the added size-tamper
case. Remote CI commit `a5dd0d1` is green for install, lint, repository
boundary, control plane, and the full CPU suite. The later fd-closure head
`ab508c4` is also green in remote CI run `33300849634` across install, lint,
both audits, and the complete CPU suite.
The subsequent strict child-environment milestone has a final local hermetic
CPU suite of 2,383 passed, five skipped, and four non-failing warnings in
143.29 seconds. Commit `5c058a1` is independently green in remote CI run
`33302879219` across install, lint, both audits, and the complete CPU suite. Its
expanded focused selection has 285 passing tests in 64.77 seconds, and the core
aggregate had already reached 126 passing tests.
The authorization-proof consumption follow-up is green with 2,386 passed, five
skipped, and four warnings in 141.13 seconds; its expanded focused selection has
289 passing tests in 64.83 seconds. Commit `311770f` passes the complete remote
CI workflow in run `33304029789`.
Worker-startup identity scaffold commit `6cb133d` is independently green across
the complete remote workflow in run `33304509263`.
The later compile-verifier import-firebreak follow-up has 2,427 passed, five
skipped, and four warnings in 143.16 seconds. Its first two remote attempts
exposed FIFO regression tests that charged dependency-heavy imports against the
short refusal timeout. Test-only commit `2488487` separates import readiness
from FIFO refusal and is green across the complete workflow in run
`33305858959`.
Atomic worker-startup envelope commit `cd1eb5e` and its UTF-8 hardening follow-up
`4bba7e9` are green across complete remote runs `33306088543` and `33306353388`,
respectively.
Runtime-locator scaffold commit `c60828d` has 50 focused tests, two isolated
import/firebreak tests, 201 repository/control tests, and all 404 `tests/ops`
tests passing locally. Its complete hermetic CPU suite has 2,508 passed, five
skipped, and four warnings in 139.30 seconds; complete remote run `33307853768`
is green.
Representation-startup authorization checkpoint `4dd8267` has 148 focused,
201 repository/control, and all 482 `tests/ops` tests passing. Its complete
hermetic CPU suite has 2,586 passed, five skipped, and four warnings in 141.50
seconds; complete remote run `33309273768` is green.
Policy v3 revision 7 records the 17 remaining oversized modules and rejects
growth, slack, stale entries, or a relaxed baseline. This plan does not
authorize an experiment or rewrite historical evidence.

## Verified macro milestone

The dated snapshots below remain historical anchors. Verified code checkpoint
`4dd8267` advances them as follows:

- production modules above 1,000 lines fell from 32 at C0, to 29 at the C2
  snapshot, to 20 at the `fd8da97` snapshot, and to 17 now;
- all nine priority decompositions after C2 are complete: exact replay, Policy
  weight sync, the native agent loop, representation configuration,
  representation checkpointing, Policy-selection runtime, answer-utility
  evaluation, Oracle-D utility, and distributed checkpointing;
- the four known multi-module strongly connected components fell to zero and
  remain guarded by a full-tree Tarjan regression test;
- execution/support surfaces fell from 82 at C0 to 79, and all 79 current
  surfaces have exactly one machine-checked classification;
- the production-size policy ratcheted from 29 exceptions to 20 at revision 5,
  then to 17 at revision 7.

The historical `fd8da97` checkpoint was green: the repository-boundary
audit reports 64 visible debts (five evidence-only config roots, 25 machine
paths, 20 oversized modules, and 14 run-specific paths) and zero violations;
the control-plane audit reports 79 surfaces and zero violations. `tests/ops`
has 199 passing tests, the focused split and compatibility suites have 577
passing tests, and the complete hermetic CPU suite has 2,286 passed, five
skipped, and four non-failing warnings in 135.24 seconds. These values are a
preserved historical snapshot, not the verified code checkpoint.

At verified code checkpoint `4dd8267`, repository-boundary policy v3 revision 7
passes
with 61 visible debts (five evidence-only roots, 25 machine paths, 17 oversized
modules, and 14 run-specific paths) and zero violations. Execution-surface
policy revision 7 binds 79 surfaces and the control-plane audit passes with
zero violations. Before the added size-tamper parameter, the focused
stabilization selection had 304
passing tests; the updated fd security file passes 10/10, and the fd-closure
full CPU rerun has 2,337 passed, five skipped, and four warnings in 140.25
seconds. The newer strict child-environment checkpoint has a final hermetic CPU
result of 2,383 passed, five skipped, and four warnings in 143.29 seconds; its
expanded focused selection has 285 passing tests in 64.77 seconds and its
earlier core aggregate had 126 passing tests. Remote run `33302879219` verifies
the same strict child-environment head across install, lint, both audits, and
the complete CPU suite.
The later authorization-proof consumption follow-up has 2,386 passed, five
skipped, and four warnings in 141.13 seconds, with 289 passing focused tests in
64.83 seconds. Commit `311770f` is green across install, lint, both audits, and
the complete CPU suite in remote run `33304029789`.
The newer compile-verifier import-firebreak head has 2,427 passed, five skipped,
and four warnings in 143.16 seconds. Execution-surface revision 7 and boundary
revision 7 pass locally at 79 surfaces/zero violations and 61 debts/zero
violations, respectively. Its lineage is remotely green at test-only FIFO timing
fix `2488487` in run `33305858959`. Envelope commit `cd1eb5e` and UTF-8
hardening commit `4bba7e9` are remotely green in runs `33306088543` and
`33306353388`. The later runtime-locator checkpoint `c60828d` has a complete local
result of 2,508 passed, five skipped, and four warnings in 139.30 seconds, plus
50 focused locator, two isolated import/firebreak, 201 repository/control, and
404 complete `tests/ops` passes. Complete remote run `33307853768` is green.
The Representation-startup authorization scaffold has 148 focused, 201
repository/control, and 482 complete `tests/ops` passes. Its full hermetic suite
has 2,586 passed, five skipped, and four warnings in 141.50 seconds; complete
remote run `33309273768` is green.
Predecessor commit `a5dd0d1` is green across install, lint, both audits, and the
full CPU job after five private-machine-path test fixtures were made hermetic.
The fd-closure head `ab508c4` is independently green in remote run
`33300849634` across the same stages.

This milestone does not close the runtime authority boundary. Experiment
policy revision 4 still has `runtime_closure.launch_enabled=false` with seven
IDs present: `atomic_authority_transaction_missing`,
`immutable_runtime_code_package_missing`,
`policy_recursive_compile_closure_missing`,
`representation_eval_safe_artifact_missing`,
`role_scoped_judge_secret_transport_missing`, `worker_member_claims_missing`,
and `worker_startup_envelope_missing`. The strict child-environment blocker is
narrowly closed, but immutable code packaging, Python startup/`.pth` and
import-before-authorization closure, an exact worker startup envelope,
role-scoped secret delivery, and exact Ray-descendant environments are not
thereby solved. C3 semantic-helper migration remains partial. C4 remains an
inventory only: its 72 worktrees, 62 branches, and seven dirty worktrees must be
preserved unless a later operator-approved action names an exact path or ref.

## 2026-08-30 fd-closure checkpoint

The first runtime blocker is closed without enabling launch. Representation and
Policy preflight retain an absolute no-follow regular-file descriptor, compare
its device/inode with `/proc/self/exe`, and carry it through the prepared plan.
Both final boundaries revalidate the same descriptor's bytes, digest, size,
stat, executable mode, and prepared identity before calling `os.execve` with
the integer fd. The declared Python path remains `argv[0]`; neither boundary
reopens that path or falls back through `/proc/self/fd`. Process-local binding
ownership uses explicit close plus `weakref.finalize` for abandoned plans.

Race, payload/size/mode tamper, descriptor reuse, real fd-exec,
pickle/import compatibility, and control-plane AST mutation tests pass. The
public CLI permits only one exact prepared-lifetime `try/finally`: preflight is
outside, authorization and dispatch are inside, and the sole finalizer is
`prepared.close_python_binding()`. A second independent security review found
no blocker, so experiment policy revision 3 removes
`fd_bound_python_exec_missing` and no other blocker.

The same split removes two stale size exceptions: `cli.py` is 963 lines and
`ops/cli_authorization.py` is 622, while their new `ops/cli_launch.py` and
`ops/cli_authorization_identity.py` leaves are 265 and 659 lines. The result
registry is also split into a 292-line facade, an 839-line schema leaf, and a
225-line support leaf. Together these changes reduce the oversized inventory
from the historical revision-5 count of 20 to the revision-7 count of 17.

At this fd-only checkpoint, `child_environment_allowlist_missing` was the next
runtime-closure priority. The later checkpoint below supersedes that specific
status without rewriting the fd checkpoint's evidence.

## 2026-08-30 strict child-environment checkpoint

Canonical representation and Policy launches now use two reviewed,
profile-specific child-environment bindings assembled from an empty mapping.
Host environment names are classified for the authorization audit, but their
values are never retrieved or copied. Repository-fixed baseline values,
profile-owned values, and exact late overlays for CLI worker authorization and
compile receipts are content-bound into the prepared launch identity; unknown
names, overwrites, or a changed field set fail closed. The older
sanitized-environment helpers remain available only for compatibility, and
neither canonical launch path uses them.

Representation additionally delegates only the exact 16 fields emitted by a
real two-worker `torchrun` under pinned PyTorch 2.9: `GROUP_RANK`,
`GROUP_WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT`, `RANK`, `ROLE_NAME`, `ROLE_RANK`, `ROLE_WORLD_SIZE`,
`TORCHELASTIC_ERROR_FILE`, `TORCHELASTIC_MAX_RESTARTS`,
`TORCHELASTIC_RESTART_COUNT`, `TORCHELASTIC_RUN_ID`,
`TORCHELASTIC_USE_AGENT_STORE`, and `WORLD_SIZE`. The smoke test verifies both
workers' exact added field set and verifies that `tgvf_rl`, its CLI, and the
child-environment module resolve from the stabilization repository. The fixed
repository `PYTHONPATH` prevents a wrong-worktree import in this launch shape;
it is not an immutable code package and does not close Python startup, `.pth`,
or import-before-authorization execution.

The expanded focused verification has 285 passing tests in 64.77 seconds; an
earlier core aggregate had 126 passing tests. The final hermetic CPU suite has
2,383 passed, five skipped, and four warnings in 143.29 seconds.
Repository-boundary policy v3 revision 7 remains at 61 debts and zero
violations, while execution-surface policy revision 5 binds all 79 surfaces
with zero control-plane violations.
Experiment policy revision 4 therefore replaces
`child_environment_allowlist_missing` with
`role_scoped_judge_secret_transport_missing`; launch remains false with seven
blockers. This narrow promotion does not claim an exact Ray-descendant
environment, role-scoped judge-secret transport, worker startup envelope,
worker member claims, or immutable runtime packaging.

## 2026-08-30 authorization-proof consumption follow-up

A CPU-only Ray 2.56.1 audit showed that the six one-use `TGVF_CLI_*` fields and
the two Policy compile-receipt fields otherwise propagate from the driver into
Ray infrastructure, TaskRunner, AgentLoopWorker, RewardLoopWorker, and a
synthetic GPU-role actor. The canonical entry paths now remove exactly those
validated fields before Representation dispatch and before Policy composes
Hydra or imports/starts Ray. Missing fields fail before partial mutation, and
torchrun rank fields plus persistent Policy runtime configuration remain
untouched.

This is environment-proof consumption, not receipt revocation: the receipt
files still exist under their guarded directories, and the verified identity
objects remain process-local until ordinary scope release. It also does not
define exact role-specific Ray envelopes or member claims. Experiment policy
revision 4 therefore remains frozen with the same seven blockers, including
`worker_startup_envelope_missing` and `worker_member_claims_missing`.

Execution-surface policy revision 6 rebinds the changed CLI, Policy worker, and
control utility. The expanded focused selection has 289 passing tests in 64.83
seconds; the final hermetic CPU suite has 2,386 passed, five skipped, and four
warnings in 141.13 seconds. Both repository audits remain green at 61 debts/zero
violations and 79 surfaces/zero violations.

## 2026-08-30 worker-startup identity scaffold

The dependency-light `tgvf_rl.ops.worker_startup` contract fixes the three
roles `policy-driver`, `representation-launcher`, and `representation-member`.
Its identity binds the exact argv, target, future runtime-package digest, and
future dependency-roots digest. Process-local verified evidence is PID-bound,
non-copyable, non-pickleable, and cannot be minted by ordinary construction.
Thirty-nine focused tests pass, and commit `6cb133d` is green in remote run
`33304509263`.

This is a scaffold, not a launch-path migration. It does not yet provide the
runtime/dependency locator, fixed role-to-target dispatch, Representation's
separate launcher/member envelopes, or member claims. No experiment blocker is
removed by this checkpoint.

## 2026-08-30 compile-verifier import-firebreak follow-up

The canonical Policy compile-prerequisite implementation now lives in the
dependency-light `tgvf_rl.ops.policy_compile_prerequisites` leaf. The former
`tgvf_rl.framework.verl.compile_prerequisites` path is an exact compatibility
facade: its four public dataclasses and five public functions are the same
objects and retain their historical import and pickle coordinates. Production
Policy launch and worker code imports the light leaf directly.

An isolated `python -B -P -S` regression imports that canonical leaf from an
exact `PYTHONPATH` and proves that Torch, NumPy, Hydra, Ray, veRL, and
`tgvf_rl.framework` remain unloaded. This removes one structural obstacle to a
future worker bootstrap; it does not provide the immutable runtime package,
dependency-root locator, startup envelope, or recursive compiler closure.
Experiment policy revision 4 therefore remains frozen with the same seven
blockers.

Execution-surface revision 7 binds the changed Policy worker and control
utility. Sixty-nine migration-focused tests pass; the full hermetic CPU suite
has 2,427 passed, five skipped, and four warnings in 143.16 seconds. Boundary
revision 7 remains at 61 debts/zero violations and the control audit remains at
79 surfaces/zero violations. Runs `33305253971` and `33305571095` failed only
because two FIFO regressions included the heavy import phase inside the short
refusal deadline. Commit `2488487` gives import readiness 30 seconds and then
requires FIFO refusal within five seconds; its complete remote workflow is green
in run `33305858959`.

## 2026-08-30 atomic worker-startup envelope follow-up

Commit `cd1eb5e` wraps worker identities in one strict canonical envelope. The
Policy role set is exactly `{policy-driver}`; the Representation role set is
exactly `{representation-launcher, representation-member}`, with only the
launcher admitted as entry. Nested duplicate or unknown fields, non-finite
values, non-canonical JSON, a changed identity digest, or a changed role set fail
closed. The envelope exposes exactly three authorization parameters: schema,
canonical JSON, and SHA-256. The legacy flat identity API is unchanged. The
complete remote workflow is green in run `33306088543`.

Commit `4bba7e9` also rejects every command argument that is not valid UTF-8
text, including a lone surrogate. Its complete remote workflow is green in run
`33306353388`. Neither commit wires the envelope into canonical dispatch or
provides Representation member claims, so experiment policy revision 4 remains
frozen with all seven blockers.

## 2026-08-30 explicit runtime-locator scaffold

Commit `c60828d` defines an externally SHA-256/byte-length-bound canonical
manifest for an exact executable, cache tag, authorized target coordinates,
the pure-Python import root above `tgvf_rl/`, and ordered dependency roots.
Descriptor-relative no-follow traversal verifies exact tree inventories and
retains root descriptors in PID-bound, non-copyable, non-pickleable evidence.
It never discovers paths from the interpreter or environment, and `.pth` files
remain inert. A fresh `python -B -P -S` regression imports the authorized
targets and a dependency through duplicated descriptor-backed roots.

Local validation has 50 runtime-locator tests, two isolated import/firebreak
tests, 201 repository/control tests, and all 404 `tests/ops` tests passing. The
complete hermetic CPU suite has 2,508 passed, five skipped, and four warnings in
139.30 seconds, and complete remote run `33307853768` is green. Evidence remains
explicitly incomplete:
`closure_complete=false`, with residual
`same-uid-mutable-executable-and-runtime-trees-have-no-atomic-observation-during-or-after-verification-v1`.
The sequential scan is not an atomic snapshot, the executable descriptor is not
retained through execution, and a formal stage-0 loader/module-origin proof is
still absent. This scaffold removes no experiment blocker; revision 4 remains
launch-disabled with the same seven identifiers.

## 2026-08-30 Representation startup authorization scaffold

Verified code checkpoint `4dd8267` reconstructs the complete protected
`WorkerStartupEnvelope` authorization group from canonical schema/JSON/SHA-256
parameters and checks the expected entry role. Exact-string keys are now
validated before scanning the `worker_startup_` namespace. This fixes the
review-discovered original ordering in which a `str` subclass could participate
in the namespace scan before rejection.

`RepresentationMemberClaim` and `RepresentationStartupPlan` are pure
authorization data. They admit exact world sizes two or four; require complete
global and local rank sets, equal single-node global/local ranks, and a unique
physical GPU mapping. The plan binds the complete envelope and mapping; each
claim binds their digests, the run/configuration/world identities, and its own
rank/GPU assignment. The plan reconstructs only
from its exact standalone authorization group. An individual claim is not
standalone authority.

Validation has 148 focused, 201 repository/control, and 482 complete
`tests/ops` passes. The full hermetic suite has 2,586 passed, five skipped, and
four warnings in 141.50 seconds; complete remote run `33309273768` is green.
There is no CLI/`exec` wiring or member-bootstrap verification. Formal stage-0,
immutable-runtime, and member-bootstrap blockers remain; revision 4 stays
launch-disabled with the same seven identifiers.

## 2026-08-31 Policy startup authorization binding scaffold

Implementation commit `e3aa054` and error-contract follow-up `48e3f07` require
every exact `PreparedPolicyLaunch` to receive live, exact-type, PID-bound
`VerifiedRuntimeLocatorScaffoldEvidence`. The evidence is borrowed rather than
retained or closed. Policy constructs its singleton `policy-driver` startup
envelope internally and binds the complete prepared command, fixed driver
target, runtime/dependency identities, and executable path, SHA-256, and byte
length to the fd-bound Python identity.

The prepared identity advances to schema v3. Its complete protected
`worker_startup_` namespace is reconstructed after authorization-parameter
aggregation and again at execution. Cross-group protected-name injection,
downgrade fields, changed envelopes, and prepared-plan subclasses fail closed.
The Policy/startup focused selection has 223 passing tests. Complete remote CI
run `33324381122` is green across install, lint, both audits, and the full CPU
suite.

This is an authorization-binding scaffold, not a completed startup chain. The
public Policy CLI does not yet construct or pass runtime-locator evidence, and
no stage-0 bootstrap verifies the envelope before runtime imports. The existing
compile and runtime-closure gates still precede this requirement. Experiment
policy revision 4 remains launch-disabled with the same seven exact blockers.

## 2026-08-31 Representation member selection-only scaffold

Checkpoint `1141df1` adds a dependency-light selector that reconstructs the
complete `RepresentationStartupPlan` from an exact caller-supplied outer CLI
identity and selects one rank only from an exact copied torchrun worker
environment. It binds the complete CLI gate identity, configuration and world
identities, full child-environment identity, CLI receipt/liveness strings,
torchrun topology, dynamic standalone fields, rank, and physical-GPU mapping.
A real pinned Torch 2.9 two-worker CPU subprocess regression characterizes the
emitted worker fields.

`RepresentationMemberSelection` is ordinary authorization data with
`authorization_scope="selection-only"` and `replay_protected=false`. It exposes
no standalone authorization parameters, establishes no authorization
consumption, verifies no runtime origin, imports no training target, and mints
no `VerifiedWorkerStartup`. Validation has 135 focused, 201 repository/control,
and 553 complete `tests/ops` passes. The committed full hermetic suite has 2,672
passed, five skipped, and four warnings in 146.52 seconds; complete remote run
`33324892712` is green.

Production member bootstrap remains unwired and the same seven launch blockers
remain. At code checkpoint `1141df1`, the canonical line is 124 commits ahead
of local `main` and 82 ahead of `origin/main`, with zero opposite-side commits
in both comparisons. Physical `main` still has 38 tracked and 55 untracked
status entries, so no promotion is authorized by this checkpoint.

## 2026-08-31 Representation member one-use receipt scaffold

Commit `0af4b41` adds a separate cooperative consumption boundary for the
selection-only record. A retained, PID-bound directory descriptor and
`openat(O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK|O_CREAT|O_EXCL)` create primitive burn
exactly one `representation-members/rank-<rank>.json` name beneath an already
prepared outer-token directory. Once the exclusive create succeeds, no error
path unlinks the leaf: a write, fsync, inode, path, or content failure leaves a
tombstone and requires a new outer launch token.

The receipt binds the verified outer token/liveness evidence, complete CLI
gate identity, startup plan and envelope, member claim, full raw environment,
TorchElastic identity, rank/GPU assignment, launcher and worker process
identities, and token/receipt directory inodes. The returned object is
immutable, non-copyable, non-serializable, and PID-bound; it reconstructs the
selection from canonical snapshots on every validation. Independent review
found a fork attack that could rewrite only the object's cached PID and start
ticks. The final implementation requires exact agreement among current
process, internal snapshot, and immutable receipt fields; the reproduced
attack now fails closed.

This remains explicitly `cooperative-same-uid-v1` and records
`hostile_same_uid_protected=false`. The public launcher does not yet pre-create
or wire the receipt directory, a same-UID hostile peer can still steal or
remove a slot, and there is no all-rank atomic release or
`VerifiedWorkerStartup`. Focused validation has 36 passes, the related fixed
Python/Torch boundary selection has 181 passes, and the complete `tests/ops`
set has 564 passes. Commit `6506a02` filters only the expected Python 3.12
warning from these intentional fork tests.

## 2026-08-31 Policy runtime-locator CLI authority binding

Commit `965acd9` closes the earlier public-CLI reachability gap without claiming
runtime immutability. `run-policy` now requires an explicit runtime-locator
manifest path, externally authorized source SHA-256, and positive source byte
length. Argparse validates their basic shape; compile/live/code blockers retain
their earlier precedence. Only after those checks does Policy load and verify
the exact manifest, current cache tag, fixed driver target, runtime trees, and
Python executable identity.

A deterministic `PolicyRuntimeLocatorAuthorizationProof` copies manifest
path/SHA/length, semantic identity, cache tag, and target coordinates into
Prepared schema v4 and the one-time gate identity. The process-local locator
evidence is closed on every success/failure path; the fd-bound Python
capability transfers only after successful preparation. `run-policy` and its
worker share command identity `tgvf-rl:run-policy:v4`; `plan-policy` remains
read-only and has no new authority arguments. Protected `runtime_locator_`
keys are exact both during preparation and at the consumed execution boundary.

Independent review found and closed two pre-commit defects: changed
execution-surface bytes initially retained revision 7, and consumed authority
initially admitted an extra `runtime_locator_legacy_*` key. Execution-surface
policy is now revision 8 with all changed surface hashes refreshed, and the
execution boundary rejects missing or extra locator keys. Policy-focused tests
have 96 passes; the independent focused/control rerun has 266 passes. The
combined hermetic CPU suite at `6506a02` has 2,710 passed, five skipped, and
four warnings in 141.88 seconds. Complete remote run `33327158411` is green
across install, lint, both audits, and the full CPU suite.

Neither slice removes a runtime-closure blocker. Locator verification remains
non-atomic and the Policy/Representation workers still lack a dependency-light
stage-0 that consumes their startup envelopes before importing the training
stack. Experiment policy revision 4 therefore remains
`runtime_closure.launch_enabled=false` with the same seven blocker IDs. At code
checkpoint `6506a02`, the canonical line is 128 commits ahead of local `main`
and 86 ahead of `origin/main`, with no opposite-side commits. Physical `main`
still has the same 38 tracked plus 55 untracked collapsed entries and remains
untouched.

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
`stabilization-c0-20260830` preserves that baseline. At that historical
checkpoint, the later consolidation head still awaited remote reproduction.
The later `a5dd0d1` predecessor and fd-closure head `ab508c4` have since passed
their respective remote CI runs.

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
That 20-row value is the historical `fd8da97` checkpoint. The subsequent
result-registry and CLI/authorization splits remove three more stale exceptions;
revision 7 now retains 17 visible debts, not waivers.

### C3 — remove utility duplication and import cycles

Introduce small leaf modules for secure file reads, canonical JSON hashing,
and create-only/content-consistent publication. Migrate one subsystem at a
time, then delete its local helper. A shared helper is acceptable only when
its security and serialization semantics are identical; similarly named but
semantically different hashes must stay explicitly separate.

Break the four then-current import cycles by moving shared protocols toward leaf
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
  `evaluation/result_registry_support.py`, descriptor-bound Policy weight-snapshot
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

At verified code checkpoint `4dd8267`, both local `main` and `origin/main` are
strict ancestors of the stabilization line. The line is ahead by 120 and 78 commits,
respectively; local `main` is itself 42 commits behind `origin/main`. Its delta
from local `main` is 299 paths: 140 additions, 156 modifications, and three
deletions. This linear graph permits a fast-forward but does not make the
physical worktree promotion-ready.

Physical `main` has 93 collapsed status entries: 38 unstaged tracked
modifications and 55 untracked entries, with no staged, deleted, renamed, or
copied entry. Expanded untracked enumeration has 132 entries total, including
94 untracked paths and 14 below `.worktrees/`. Nineteen dirty paths overlap the
stabilization delta. Eighteen modify/modify cases pass a read-only textual merge
probe. The remaining modify/delete case is
`tools/supervise_prl14_cleanfinal16_eval.sh`, modified on physical `main` but
deleted on the stabilization line. Fast-forward promotion cannot reconcile
that uncommitted state.

The two clean unanchored detached heads are
`2d61b07995b1d5b90c221fe1faf5090e8d985fef` at
`/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-rp70-20260802`
and `151701fdb13c5ecf6fac6c0f67760e51c427c277` at
`/tmp/prl26-gap-fix-NNVbZQ/worktree`. Neither has a containing local branch,
remote-tracking branch, or tag. Preserving physical-main state and anchoring
these commits require explicit operator-authorized actions; this plan does not
authorize promotion, tagging, archival, or deletion.

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

State at verified code checkpoint `4dd8267`: the entry-point disposition
condition, the three C2 targets, all
nine post-C2 priority decompositions, the revision-7 production-size exception
ratchet, the known-cycle condition, the fd-bound Python-exec blocker, and the
strict child-environment blocker are complete, and C4 remains a read-only
inventory. Validated one-use proof fields are also consumed before downstream
dispatch. The predecessor, fd-closure, strict child-environment, and
proof-consumption checkpoints all have green remote CPU CI. The newer
compile-verifier import-firebreak lineage is remotely green at FIFO timing fix
`2488487` in run `33305858959`; atomic envelope `cd1eb5e` and UTF-8 hardening
`4bba7e9` are green in runs `33306088543` and `33306353388`. Runtime checkpoint
`c60828d` has 2,508 local passes, five skips, and four warnings, but
deliberately reports incomplete closure; complete remote run `33307853768` is
green. Representation-startup checkpoint `4dd8267` has 2,586 local passes,
five skips, and four warnings; complete remote run `33309273768` is green.
Portable runtime/CLI closure remains disabled by seven named blockers, now
including `role_scoped_judge_secret_transport_missing`; immutable code/startup
closure, launch-integrated envelope/member bootstrap, Ray-descendant environment
closure, and the remaining semantic-helper migrations are still open.
