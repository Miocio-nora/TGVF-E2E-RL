# Repository stabilization audit — 2026-08-30

Status: active stabilization baseline. This document records the state that
must be preserved while the repository is consolidated. It is not an
authorization to launch training or evaluation.

## Canonical stabilization line

- Worktree: `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-stabilize-protocol-v1`
- Branch: `stabilize/protocol-contract-v1-20260830`
- Base: `origin/main` at `43f2295c15da2bbc14972af6b63f0038e7789c3f`
- Verified Representation-startup code checkpoint:
  `4dd826729ab42027e0204633af55edca96122a87`, synchronized with the remote
  stabilization line at verification time and green in complete remote CI run
  `33309273768`.
- Commit graph at that checkpoint: zero commits unique to local `main` and 120
  unique to the stabilization line; zero unique to `origin/main` and 78 unique
  to the stabilization line. Local `main` is 42 commits behind `origin/main`,
  and both are strict ancestors of the verified checkpoint.
- Experiment execution: frozen until the protocol, evidence, control-plane,
  and test gates in this document pass.
- Integration rule: selectively port reviewed changes. Do not merge an entire
  PRL experiment branch into this line.
- Runtime migration batches and their explicit non-migration set are defined
  in [`CANONICAL_RUNTIME_MIGRATION_PLAN_20260830.md`](CANONICAL_RUNTIME_MIGRATION_PLAN_20260830.md).

## Current audit closure

This table distinguishes an identified problem from a verified fix. A row is
not closed merely because the affected experiment has stopped running.

| Area | State | Required closure |
|---|---|---|
| Repository ownership | partial | Boundary policy v3 revision 7 passes with zero violations and 61 visible debts: five evidence-only roots, 25 machine paths, 17 oversized modules, and 14 run-specific paths. Every oversized row binds its current line count, owner, rationale, and next split; baseline comparison rejects a new exception or raised ceiling. Neutral-path semantic RP/PRL debt still needs a second policy layer. |
| Execution-surface inventory | verified | Execution-surface policy revision 7 binds 79 entries and their file bytes under the v2 recursive exact-set schema: every Python file below `tools`/`spikes`, every shell file there, and every `src` main/module-main, Python-shebang, executable-bit, or console-entry target. Missing, extra, duplicate, moved, reclassified, or content-drifted entries fail closed, and the control-plane audit passes with zero violations. |
| Public and historical launch surfaces | contained; fd execution and strict child environment narrowly closed | Legacy/spike surfaces remain inventoried and guarded. Descriptor-bound Python execution and exact empty-built canonical child environments have implementation, adversarial coverage, and policy promotion. Atomic worker-envelope, explicit runtime-locator, and Representation member-authorization scaffolds now exist, but none is wired into canonical CLI/exec authority and the locator explicitly reports incomplete closure. An individual member claim is not standalone authority. Experiment policy revision 4 keeps launch false with seven blockers. Immutable runtime packaging and formal stage-0/import closure, verified member bootstrap, role-scoped judge-secret transport, exact Ray-descendant environments, authorization transaction, and the artifact/compiler blockers remain unsolved. |
| Crop observation/action contracts | verified for the stabilization runtime | Matched60, legacy generic86 and strict/legacy action semantics are explicit identities with focused tests. No historical RP/PRL config binds the new strict loop, so this does not retroactively certify old artifacts. |
| TGVF/Atomic observation layout | verified | Live append and immutable replay/layout consume the same once-rendered protocol/dialect-bound bytes; the layout has no implicit Thinking renderer. |
| Historical TGVF/Atomic impact | verified | Implementation commits `b100d3d`, `ec0555b`, `8e6b3d`, `5baddc` and provenance checkouts `b87126a`, `017b507`, `001838b` explicitly pass one dialect-bound renderer to appender and layout. Training replay consumes the recorded token rows, so the unused fallback does not downgrade those rows. Their runtime is `training_run`/precomputed, not official-visible/native-pixel. |
| Result comparisons | contained, not closed | Registry v2 verifies score-file bytes/content and independent preregistration bytes, but those artifacts do not bind a score to evaluation identity, the exact trajectory set, weights and the full comparison contract. V2 rejects every `golden` status and every numeric delta. Its implementation is now split into a 292-line facade, 839-line schema leaf, and 225-line support leaf without changing historical imports or serialization coordinates. |
| Policy compile prerequisites | blocked | The hidden worktree-local default is removed and a strict content-bound v1 manifest now binds four minimum declared files. Launch remains blocked because recursive Python headers and the compiler system-toolchain are not yet closed by the manifest schema. |
| Snapshot filesystem closure | partial | LoRA closure reads use descriptor-relative traversal and immutable publications use no-replace semantics. Full-model freeze stores immutable manifest/receipt records rather than copying the external weights; official loading now hashes the complete bound checkpoint/model closure and repeats that verification immediately before vLLM construction, including same-size mutation tests. vLLM 0.12 exposes neither a loaded-adapter nor loaded-full-model digest, so same-UID mutation after the final verification remains a documented runtime residual. |
| Test discovery and behavior | verified Representation-startup checkpoint local and remote green; historical checkpoints preserved | The C0 tag remains a preserved 2,112-pass baseline, `ccef450` remains the 2,193-pass snapshot, and `fd8da97` remains the historical 2,286-pass snapshot. The fd-closure hermetic CPU rerun completed with 2,337 passed, five explicit skips, four non-failing warnings, and zero failures in 140.25 seconds; before its added size-tamper parameter, the focused selection had 304 passing tests and the updated fd security file passed 10/10. Predecessor commit `a5dd0d1` and fd-closure head `ab508c4` are remotely green, the latter in run `33300849634`. Strict child-environment head `5c058a1` has 2,383 local passes and is remotely green in run `33302879219`. The authorization-proof consumption follow-up has 2,386 passes and is remotely green in run `33304029789`; worker-startup scaffold commit `6cb133d` is remotely green in run `33304509263`. Compile-verifier lineage fix `2488487`, atomic envelope `cd1eb5e`, and UTF-8 hardening `4bba7e9` are green in remote runs `33305858959`, `33306088543`, and `33306353388`. Runtime-locator checkpoint `c60828d` has 50 focused, two isolated import/firebreak, 201 repository/control, and 404 complete `tests/ops` passes; its full hermetic suite has 2,508 passed, five skips, and four warnings in 139.30 seconds, and complete remote run `33307853768` is green. Representation-startup checkpoint `4dd8267` has 148 focused, 201 repository/control, and 482 complete `tests/ops` passes; its full hermetic suite has 2,586 passed, five skips, and four warnings in 141.50 seconds, and complete remote run `33309273768` is green. Current policies report boundary revision 7 at 61 debts/zero violations plus execution-surface revision 7 at 79 surfaces/zero violations. |

## 2026-08-30 fd-closure checkpoint

The executable pathname-rebinding and fd-selection chain is now closed from
bind through process replacement. Both canonical launch preflights open the
Python candidate through absolute component-by-component no-follow traversal,
retain the regular-file descriptor, and compare its device/inode with
`/proc/self/exe`. The prepared
plan carries a serializable fd-free identity and a process-local,
non-serializable descriptor owner. The two final execution boundaries compare
the binding identity with the prepared identity, reread the same fd, revalidate
digest, size, device, inode, mode, and executable bit, then call
`os.execve(integer_fd, argv, environment)`. They preserve the declared Python
path as `argv[0]` and have no path or `/proc/self/fd` fallback.

The retained owner uses explicit idempotent close plus `weakref.finalize`.
Preflight failure, authorization refusal, failed exec, abandoned ownership,
double close, and fd-number reuse are covered. Adversarial tests also cover a
candidate replacement between process-identity inspection and retained open,
payload/size/mode tampering on the same inode, unavailable `/proc` identity,
real fd execution after replacing the declared pathname, and
facade/pickle/type-hint/import-order compatibility. Control-plane negative
mutations reject caught dispatch, extra try/finally effects, a wrong finalizer,
wrong dispatch provenance, and rebinds. A second independent read-only security
review found no remaining blocker to the narrow fd-bound promotion; the known
same-inode write window after final verification belongs to
`immutable_runtime_code_package_missing`, not this closed blocker.

The control-plane now requires one exact prepared lifetime. Static preflight
occurs before a handler-free `try/finally`; authorization and dispatch are the
reviewed try body, and the only final statement is
`prepared.close_python_binding()`. The implementation split leaves `cli.py` at
963 lines and `ops/cli_authorization.py` at 622 lines; new
`ops/cli_launch.py` and `ops/cli_authorization_identity.py` leaves are 265 and
659 lines. Repository policy v3 revision 7 removes both stale facade
exceptions and retains 17 other oversized modules.

At this fd-only checkpoint, experiment execution policy revision 3 removed
`fd_bound_python_exec_missing`, but `runtime_closure.launch_enabled` remains
`false` with seven exact blockers. `child_environment_allowlist_missing` is
recorded here as the then-next priority. The later child-environment checkpoint
below supersedes that specific status without changing this fd evidence.

## 2026-08-30 strict child-environment checkpoint

Canonical representation and Policy launch environments are now constructed
from an empty mapping under separate fixed profiles. The builder iterates host
environment names only to record ignored/rejected-name audit identities; it
does not retrieve or copy any host value. The fixed baseline, explicit
profile-owned inputs, and exact CLI worker-authorization/compile-receipt late
overlays are bound into the prepared authorization identity. Unknown fields,
an overwrite, a changed base identity, or a non-exact late field set fails
closed. The old sanitized child-environment helpers remain as compatibility
APIs, but canonical representation and Policy launches no longer call them.

The representation profile admits the exact 16 worker fields observed from a
real two-worker launch under pinned PyTorch 2.9: `GROUP_RANK`,
`GROUP_WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT`, `RANK`, `ROLE_NAME`, `ROLE_RANK`, `ROLE_WORLD_SIZE`,
`TORCHELASTIC_ERROR_FILE`, `TORCHELASTIC_MAX_RESTARTS`,
`TORCHELASTIC_RESTART_COUNT`, `TORCHELASTIC_RUN_ID`,
`TORCHELASTIC_USE_AGENT_STORE`, and `WORLD_SIZE`. The smoke test checks that
these and only these fields are added to both worker environments, and that the
loaded `tgvf_rl`, CLI, and child-environment module origins are inside the
stabilization repository. A repository-fixed `PYTHONPATH` addresses the
observed wrong-worktree import only; it is not an immutable code package and
does not establish Python startup, `.pth`, or import-before-authorization
closure.

The expanded focused suite has 285 passing tests in 64.77 seconds; the earlier
core aggregate had 126 passing tests. The final hermetic CPU suite has 2,383
passed, five skipped, and four warnings in 143.29 seconds.
Repository-boundary revision 7 remains green with 61 debts and zero violations,
and execution-surface revision 5 binds 79 surfaces with zero control-plane
violations. Experiment execution policy revision 4 replaces
`child_environment_allowlist_missing` with
`role_scoped_judge_secret_transport_missing`; it still has seven exact
blockers and `runtime_closure.launch_enabled=false`.

This is deliberately a narrow closure. It does not solve immutable runtime
packaging, caller Python or `.pth` execution, imports that can occur before
authorization checks, the worker startup envelope, worker member claims,
role-scoped judge-secret transport, or exact environments for Ray descendants.

## 2026-08-30 authorization-proof consumption follow-up

A CPU-only Ray 2.56.1 audit confirmed that leaving the six `TGVF_CLI_*` fields
and two compile-receipt fields in the Policy driver environment propagates them
through Ray infrastructure and the TaskRunner, AgentLoopWorker,
RewardLoopWorker, and synthetic GPU-role actor. Canonical Representation now
removes its six validated CLI fields immediately before rank-local training
dispatch. Canonical Policy removes all eight validated one-use fields after
compile-receipt verification and before Hydra composition or Ray import/start.
The scrub is exact, rejects a missing field before any partial deletion, and
does not remove torchrun rank state or persistent Policy runtime settings.

The receipt files are not deleted and process-local identity objects are not a
revocation mechanism. Exact Ray role envelopes, member claims, Python
startup/import firebreaks, and secrets remain open. Experiment policy revision
4 is therefore unchanged and launch remains false with seven blockers.
Execution-surface revision 6 binds the changed surfaces. The expanded focused
selection has 289 passing tests in 64.83 seconds, the final hermetic CPU suite
has 2,386 passed, five skipped, and four warnings in 141.13 seconds, and both
audits pass with zero violations.

## 2026-08-30 worker-startup identity scaffold

The dependency-light `tgvf_rl.ops.worker_startup` contract defines exact
Policy driver, Representation launcher, and Representation member roles. Its
identity binds complete argv, target, and future runtime-package and
dependency-root digests. Verified evidence is PID-bound, non-copyable,
non-pickleable, and unavailable through ordinary construction. Thirty-nine
focused tests pass; commit `6cb133d` is green across the complete remote
workflow in run `33304509263`.

The scaffold is not wired into canonical launch dispatch and has no
runtime/dependency locator, fixed role-to-target mapping, dual Representation
envelopes, or member claims. Experiment policy revision 4 therefore remains
frozen with the same seven blockers.

## 2026-08-30 compile-verifier import-firebreak follow-up

The Policy compile-prerequisite implementation moved out of the eager
`tgvf_rl.framework.verl` package into the dependency-light
`tgvf_rl.ops.policy_compile_prerequisites` leaf. The old module is an exact
compatibility facade; four dataclasses and five public functions retain object
identity plus their historical import and pickle coordinates. Canonical Policy
launch and worker imports now use the light implementation path.

An exact-environment `python -B -P -S` regression proves that importing the
canonical verifier does not load Torch, NumPy, Hydra, Ray, veRL, or
`tgvf_rl.framework`. This removes a structural import-before-authorization
obstacle, but does not bind an immutable runtime package or dependency locator,
define the worker envelope/member claims, or close the recursive compiler
toolchain. Experiment policy revision 4 is unchanged and launch remains false
with seven blockers.

Execution-surface revision 7 binds the changed Policy worker and control
utility. Sixty-nine migration-focused tests pass; the final hermetic CPU suite
has 2,427 passed, five skipped, and four warnings in 143.16 seconds. Boundary
revision 7 reports 61 debts/zero violations and the control audit reports 79
surfaces/zero violations. Remote runs `33305253971` and `33305571095` exposed
two FIFO tests whose short deadlines also covered dependency-heavy imports.
Test-only commit `2488487` separates a 30-second readiness phase from the
five-second FIFO refusal phase and is green across the complete remote workflow
in run `33305858959`.

## 2026-08-30 atomic worker-startup envelope follow-up

Commit `cd1eb5e` adds an atomic `WorkerStartupEnvelope` without changing the
legacy flat identity API. Policy requires exactly `{policy-driver}`;
Representation requires exactly `{representation-launcher,
representation-member}`, and the member cannot be the entry role. Strict
canonical nested JSON rejects duplicate or unknown fields, non-finite values,
non-canonical spelling, an identity-digest mismatch, and a changed role set.
Exactly three authorization parameters bind the complete envelope: schema,
canonical JSON, and SHA-256. The complete remote workflow is green in run
`33306088543`.

Commit `4bba7e9` further rejects command arguments that cannot be encoded as
UTF-8, including a lone surrogate, before canonical serialization. It is green
across the complete remote workflow in run `33306353388`. These contracts are
not connected to canonical dispatch and do not provide member claims, so
experiment policy revision 4 remains frozen with all seven blockers.

## 2026-08-30 explicit runtime-locator scaffold

Commit `c60828d` adds a strict canonical manifest externally bound by source
SHA-256 and byte length. It declares the executable, cache tag, authorized
target coordinates, exact pure-Python import root above `tgvf_rl/`, and ordered
dependency roots. Descriptor-relative no-follow traversal verifies exact tree
inventories, rejects bytecode and native runtime-package candidates, and
retains the import/dependency root descriptors in PID-bound, non-copyable,
non-pickleable evidence. The implementation does not discover paths from the
interpreter or environment, and declared `.pth` files remain inert bytes. A
fresh `python -B -P -S` regression imports the authorized targets and a
dependency through duplicated descriptor-backed roots passed with `pass_fds`.

Local verification has 50 runtime-locator tests, two isolated import/firebreak
tests, 201 repository/control tests, and all 404 `tests/ops` tests passing. The
complete hermetic CPU suite has 2,508 passed, five skipped, and four warnings in
139.30 seconds. Commit `c60828d` is green across the complete remote workflow in
run `33307853768`; its `unit-and-control-plane` job `99247475104` completed
successfully.

The evidence deliberately reports `closure_complete=false` and the exact
residual
`same-uid-mutable-executable-and-runtime-trees-have-no-atomic-observation-during-or-after-verification-v1`.
A same-UID writer can change an earlier file while a later file is scanned or
change verified bytes afterward; the executable descriptor is not retained
through execution; and no formal stage-0 loader/module-origin proof exists.
This scaffold is not an immutable runtime package and removes no blocker.
Experiment policy revision 4 remains launch-disabled with the same seven exact
identifiers.

## 2026-08-30 Representation startup authorization scaffold

Verified code checkpoint `4dd8267` reconstructs the complete protected
`WorkerStartupEnvelope` authorization group, requiring exact-string keys before
the `worker_startup_` namespace scan and revalidating canonical JSON, schema,
SHA-256, and expected entry role. This ordering fixes the original
review-discovered blocker in which a `str` subclass could enter the namespace
scan before rejection.

`RepresentationMemberClaim` and `RepresentationStartupPlan` are pure
authorization data. Exact world sizes two or four require complete global/local
rank sets, equal single-node global/local ranks, and one-to-one physical GPU
mapping. The plan binds the complete envelope and mapping. Every claim binds
the envelope/member digests, run/configuration/world identities, and its own
rank/GPU assignment. Only the complete plan has an
exact standalone authorization group; an individual claim is not standalone
authority.

Validation has 148 focused, 201 repository/control, and 482 complete
`tests/ops` passes. The full suite has 2,586 passed, five skipped, and four
warnings in 141.50 seconds; complete remote run `33309273768` is green. There is
no CLI/`exec` wiring or member-bootstrap verification. Formal stage-0,
immutable-runtime, and member-bootstrap blockers remain. Revision 4 stays
launch-disabled with the same seven identifiers.

## 2026-08-31 Policy startup authorization binding audit

Policy startup-envelope binding landed in `e3aa054`; follow-up `48e3f07`
preserves the established public type-error contract. Exact live
runtime-locator evidence now produces an internally owned singleton
`policy-driver` envelope whose command, fixed target, runtime/dependency
digests, and executable path/SHA-256/length agree with the prepared fd-bound
Python identity. Prepared schema v3 includes the complete envelope.

Independent adversarial review found and closed two pre-commit defects:
cross-group injection into the protected `worker_startup_` namespace and
subclass dispatch around `PreparedPolicyLaunch.__post_init__`. Aggregate and
consumed authorization maps are now reconstructed as complete envelopes, and
execution accepts only the exact prepared type. The focused Policy/startup
selection has 223 passes; complete remote run `33324381122` is green.

The evidence remains borrowed and the runtime locator still reports incomplete
closure. The public CLI does not materialize or pass this evidence, and no
worker bootstrap consumes the envelope before runtime import. No blocker is
removed: revision 4 remains launch-disabled with the same seven identifiers.

## 2026-08-31 Representation member selection-only audit

At checkpoint `1141df1`, an exact `CLIExecutionAuthorizationIdentity` is
retained in the selection and its full gate identity digest is recorded. The
selector reconstructs the complete startup plan, verifies
configuration/world bindings and canonical identity JSON, validates the full
materialized child environment, and binds every raw environment field into
`full_environment_sha256`. Exact world sizes two and four, rank topology,
unique GPU mapping, pinned standalone torchrun fields, and a real two-worker CPU
Torch 2.9 subprocess are covered.

Independent review first found three pre-commit defects: two distinct outer CLI
authorities collapsed to the same selection, raw worker fields such as
`PYTHONPATH` were projected away, and the synthetic torchrun fixture accepted
an impossible agent-store/run-ID combination for the fixed `--standalone`
command. The final checkpoint binds the complete gate identity and full raw
environment, reuses the materialized child-environment verifier, requires
`TORCHELASTIC_USE_AGENT_STORE=True`, and requires a canonical lowercase UUID4
run ID. The original reviewer reproduced all three attacks against the fixed
version and confirmed that they fail closed or produce distinct records.

The resulting record is deliberately replayable:
`authorization_scope="selection-only"` and `replay_protected=false`. It neither
consumes receipt/liveness authority nor verifies runtime/module origin, imports
the training target, dispatches a member, or mints verified startup evidence.
Validation has 135 focused, 201 repository/control, and 553 complete
`tests/ops` passes. The full suite has 2,672 passed, five skipped, and four
warnings in 146.52 seconds; complete remote run `33324892712` is green. The
repository-boundary audit remains at 61 registered debts and zero violations;
the control-plane audit remains at 79 surfaces and zero violations.

Member bootstrap and the same seven closure blockers remain open. At code
checkpoint `1141df1`, the canonical line is 124 commits ahead of local `main`
and 82 ahead of `origin/main`, with no opposite-side commits. Physical `main`
remains unmodified by this work and still has 38 tracked plus 55 untracked
status entries; no promotion or cleanup is authorized here.

## 2026-08-31 Representation member one-use consumption audit

Commit `0af4b41` adds a deliberately unwired cooperative receipt consumer for
the selection-only checkpoint. It re-verifies outer CLI token consumption and
launcher liveness, retains exact token/receipt directory descriptors, and
burns one fixed rank leaf with no-follow `openat` and `O_EXCL`. The receipt
binds the complete gate, plan, envelope, claim, environment, TorchElastic,
rank/GPU, process, directory-inode, and outer-receipt identities. Partial
writes and all later failures leave a permanent tombstone for that token.

Independent adversarial review found one blocking defect after the initial 35
focused passes: a forked child could use `object.__setattr__` to replace the
consumed object's cached PID/start ticks because those cached values were not
cross-checked with the durable receipt. The repaired object stores no caller
selection reference, reconstructs its selection on every use, and requires
exact process identity agreement among current `/proc`, internal snapshot, and
receipt. The original exploit now fails with `receipt worker process identity
differs`. Final focused validation has 36 passes, related fixed-runtime tests
have 181, the complete `tests/ops` run has 564 passes, and success/failure FD
probes remain 4-to-4.

The scope remains honest: `cooperative-same-uid-v1`, not hostile-peer
protection, runtime provenance, cohort atomicity, or verified startup. The
launcher still has to pre-create and wire the private directory. Commit
`6506a02` suppresses only the precisely matched Python 3.12 warning caused by
the intentional fork regressions.

## 2026-08-31 Policy runtime-locator CLI authority audit

Commit `965acd9` gives public `run-policy` the previously missing explicit
locator authority triple: absolute manifest path, lowercase source SHA-256,
and positive byte length. After unchanged compile/live/code gates, production
preflight calls the strict manifest loader and scaffold verifier for the
current cache tag and exact Policy driver target. A deterministic typed proof
enters Prepared v4 and the one-time CLI gate identity; live evidence and the
Python FD follow separately tested success/failure ownership paths. The Policy
launcher and worker now share command ID v4, while `plan-policy` remains
unchanged.

Independent review found two blocking defects despite the first 95 targeted
passes. First, changed execution-surface hashes had not incremented the policy
revision. Second, execution compared expected locator values but did not reject
an additional consumed `runtime_locator_legacy_*` field. The final code raises
execution-surface revision 8, refreshes all affected content hashes, and
requires the exact locator namespace at both preparation and execution. The
final Policy-focused suite has 96 passes; the reviewer's post-fix hermetic
focused/control suite has 266 passes and zero violations.

The combined checkpoint `6506a02` passes Ruff, repository-boundary revision 7
at 61 debts/zero violations, execution-surface revision 8 at 79 surfaces/zero
violations, and the complete hermetic CPU suite at 2,710 passed, five skipped,
and four warnings in 141.88 seconds. Complete remote run `33327158411` is green
across install, lint, both audits, and the full CPU suite.

No experiment blocker changed. Runtime locator v1 is still a non-atomic
mutable-tree scaffold, and neither worker has a dependency-light stage-0 that
re-verifies runtime origin and consumes its envelope before heavy imports. The
same seven blockers and `launch_enabled=false` remain. The canonical line is
128 commits ahead of local `main` and 86 ahead of `origin/main`; physical
`main` is still untouched with 38 tracked and 55 untracked collapsed entries.

## Historical post-ratchet consolidation milestone (`fd8da97`)

This section preserves the earlier `fd8da97` snapshot. Its counts are not those
of the verified checkpoint and are not retroactively rewritten.

- The initial raw oversized-module scan found 32 production files above the
  limit. The first three decompositions established the 29-row policy-v3
  baseline; nine further reviewed decompositions now leave 20 exact
  exceptions. The production-size list in policy revision 5 drops exactly
  those nine satisfied rows, adds no size exception, and raises no ceiling.
  Revision 5 also relocates two unchanged machine-path debt records to the
  extracted Policy-selection config-schema leaf.
- The completed nine-module batch covers veRL exact-replay registration,
  weight-snapshot storage, the native policy client, representation config and
  checkpoint contracts, policy-selection runtime contracts, Answer Utility
  evaluation inputs, Oracle-D evaluation, and distributed checkpoints. Every
  facade targeted by these nine decompositions and every newly extracted leaf
  is below 1,000 lines. Six split families additionally enforce explicit
  per-file headroom ceilings no greater than 850 lines: representation config,
  representation checkpoint, Policy-selection runtime, Answer Utility,
  Oracle-D, and distributed checkpoint.
- The four known multi-module import SCCs remain at zero. The execution-surface
  inventory remains an exact 79-path set with zero control-plane violations.
- Independent compatibility review found and fixed missing facade exports for
  private helpers whose pickle coordinates had been rebound to the historical
  facade. A shared opt-in helper now resolves only a class's own necessary
  postponed annotations before module rebinding, using explicit leaf globals.
  Leaf-first imports therefore preserve type hints, pickle coordinates, and
  the one-way import DAG without a `sys.modules` facade backreference. The raw
  annotations of the selected classes become resolved type objects; existing
  dataclass `Field.type` metadata remains unchanged.
- Configuration payloads, bound manifests, and weight snapshots now use one
  no-follow descriptor payload for hashing plus parsing. Presence-only model
  and report checks use a metadata-only regular-file probe, including a sparse
  16-GiB regression that performs no payload read. The boundary is intentionally
  narrower than immutable-file semantics: hardlinks remain allowed, a
  same-inode concurrent rewrite can still yield a torn read, and an unopened
  real directory component can be replaced before its later descriptor open.
  The existing unsafe `torch.load(..., weights_only=False)` path remains a
  separate runtime-closure blocker.
- Formal verification at code milestone `fd8da97` is: repository boundary
  `pass` with 64 debts (`5` evidence-only roots, `25` machine paths, `20` oversized
  modules, `14` run-specific paths) and zero violations; control-plane `pass`
  with 79 surfaces and zero violations; `tests/ops` 199 passed; the combined
  representation/Answer Utility/import compatibility selection 577 passed;
  and the full hermetic suite 2,286 passed, five skipped, four warnings in
  135.24 seconds.
- Full-repository Ruff lint passes, and all 51 Python paths changed relative to
  the preceding published milestone (`0b5f42f`) pass the current formatter. A
  separate whole-tree
  formatter check still identifies 158 historical files that would be
  reformatted. They are pre-existing formatting debt and were not mechanically
  rewritten as part of this stabilization batch.
- Runtime closure remains non-overridably disabled with eight declared
  blockers. No GPU, training, evaluation, external judge, worktree deletion,
  or branch deletion was performed. These are local verification results;
  no remote-CI result is claimed.

## Preserved dirty worktrees

The following worktrees contained uncommitted state at audit time and must not
be removed, reset, stashed, or cleaned by the stabilization work:

| Status entries | Branch | Worktree |
|---:|---|---|
| 93 | `main` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl` |
| 19 | `prl25-c-tgvf-80step` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl25-crop-aligned` |
| 7 | `prl24-d-crop-bs64-fmt2-sp2` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-d-crop-bs64-fmt2-sp2` |
| 4 | `prl26b-generic86-historical-eval-20260830` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-historical` |
| 3 | `prl24-a-bs64` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl24-a-bs64` |
| 2 | `prl27-crop-replay-renderer-fix-20260830` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl27-replay-fix` |
| 1 | `prl26b-generic86-matched-eval-20260830` | `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-prl26b-generic86-eval` |

The physical `main` row's 93 collapsed entries are 38 unstaged tracked
modifications and 55 untracked entries, with no staged, deleted, renamed, or
copied entry. Expanding untracked directories yields 132 entries: the same 38
tracked modifications plus 94 untracked paths, 14 of them below `.worktrees/`.
At verified code checkpoint `4dd8267`, the stabilization delta from local
`main` spans 299 paths: 140 additions, 156 modifications, and three deletions.
Nineteen dirty
paths overlap that delta. Eighteen are modify/modify and pass a read-only
textual merge probe. The remaining modify/delete case is
`tools/supervise_prl14_cleanfinal16_eval.sh`, which is modified on physical
`main` and deleted on the stabilization line. A graph-only fast-forward cannot
merge or preserve this uncommitted state.

The graph itself is linear: the stabilization line is 120 commits ahead of
local `main` and 78 ahead of `origin/main`, with neither comparison having a
commit absent from the stabilization line. Local `main` is 42 commits behind
`origin/main`. Consequently, the canonical unified line exists, but physical
`main` is not promotion-ready.

At baseline there were 71 registered worktrees before this stabilization
worktree was added, 61 local branches, and 34 `origin/*` remote-tracking
branches. The new stabilization worktree/branch raises the first two counts by
one.

The frozen C4 read-only inventory reports 72 existing worktrees: 59 attached,
13 detached, 65 clean, and seven dirty, with none locked or marked prunable.
It also records 62 local branches. The complete per-path/per-ref table is
[`WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md`](WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md).
Every artifact reference remains unknown, so even a clean row is only a
review-only candidate. This audit does not remove a worktree or branch, and
every exact deletion still requires separate operator approval.

The two clean detached heads with no containing local branch, remote-tracking
branch, or tag are:

- `2d61b07995b1d5b90c221fe1faf5090e8d985fef` at
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-rp70-20260802`;
- `151701fdb13c5ecf6fac6c0f67760e51c427c277` at
  `/tmp/prl26-gap-fix-NNVbZQ/worktree`.

Physical-main preservation, durable tags for these heads, branch promotion,
archival, and deletion all require separate explicit operator authorization.

## Runtime preservation boundary

- All eight GPUs were idle at the start of stabilization.
- No training or evaluation process may be started by this work.
- Of 28 tmux panes, 23 were dead experiment panes. The five live panes were
  Codex/user sessions and are explicitly out of cleanup scope.
- Dead panes may be removed only after their final output, exit status, and
  artifact paths have been inventoried. Cleanup must target explicit session
  names; broad tmux deletion is forbidden.
- The stopped `prl26-e-atomic-train512-s32` pane contained only a waiter
  (`bash`/`sleep`), not a GPU or model process. Its exit status is 130. It must
  not be restarted implicitly.
- A residual `OPENROUTER_API_KEY` entry was removed from tmux's global
  environment during stabilization. Its value was not read or recorded.
  Existing process environments were not changed. The credential should also
  be rotated at the provider before any later external-judge run.

## Non-overridable runtime-closure gate

The frozen scheduling policy and the runtime-closure gate are intentionally
different controls. A reasoned freeze override can authorize a future scheduled
run only after runtime closure is complete; it cannot waive a code-execution or
artifact-safety blocker.

`configs/ops/experiment_execution_policy.json` revision 4 currently declares
`runtime_closure.launch_enabled=false`. Canonical Policy training,
representation training, their workers, and representation internal evaluation
fail before token consumption or unsafe artifact loading while any blocker
remains. The exact blocker identifiers cover:

- an immutable, content-bound runtime code package (the current `.venv312`
  editable installation points at the separate base worktree);
- an exact worker argv/environment startup envelope and per-member claims;
- role-scoped judge-secret transport that cannot fan out through Ray or another
  outer/global environment;
- a safe non-pickle representation-evaluation artifact;
- recursive compiler/header/system-toolchain closure;
- atomic publication of the combined one-time authorization transaction.

Descriptor-bound Python execution and the strict canonical child-environment
allowlist are no longer in that list. Their retained-fd and empty-built-profile
implementations, adversarial tests, and policy updates are recorded in the two
checkpoints above. The child-environment promotion adds the narrower
`role_scoped_judge_secret_transport_missing` blocker rather than treating
credentials as an environment value. Seven identifiers remain, and no launch
is enabled.

This is containment, not a claim that the dormant launch implementation is
already secure. Removing a blocker requires its implementation, adversarial
tests, policy revision, and review in the same migration batch.

## Evidence classification gate

Every reported result must be registered as exactly one of:

- `golden`: reserved for a future registry schema that mechanically verifies a
  producer receipt binding score content to evaluation identity, exact
  trajectory-set identity, weights and the full comparison contract; registry
  v2 rejects this status unconditionally;
- `standalone`: valid for its own declared contract but not a causal comparator;
- `confounded`: measured, but two or more interventions differ;
- `invalid`: a correctness invariant failed;
- `pending`: the required artifact does not yet exist or is incomplete.

The future score-provenance receipt must bind at least evaluation-identity SHA,
the exact ordered trajectory-set identity, task-manifest SHA, model/weight SHA,
image cap, prompt identity, observation/runtime/parser/action identities,
generation budget, scorer identity, seed policy, paired-RNG namespace, the
typed score-artifact SHA and the full comparison-contract SHA. Its producer
must derive these values from verified evaluator inputs rather than accept
caller labels as authority. Until that producer and verifier exist, registry
v2 is an evidence inventory only: no row can become `golden`, and the public
delta API always fails closed. Independent preregistration remains necessary
but is not score provenance.

## Protocol gate

- Crop observation text must be rendered once and the exact bytes must be used
  by both the live appender and immutable replay layout.
- Renderer/observation protocol selection must be explicit and fail closed.
  A silent generic fallback is forbidden.
- Historical `generic86` remains available only under an explicit legacy
  protocol identity; it must never be inferred from a missing renderer.
- Action precedence is a separate identity from observation bytes. Canonical
  runs require one terminal tool call per assistant turn; answer text after a
  tool action and multiple same-turn calls fail closed. Historical
  answer-over-action/last-call semantics remain available only through the
  explicit legacy action-boundary identity.
- The strict DeepEyes training loop classifies only the policy tokens added by
  the current assistant turn, even when the upstream Hermes parser terminates
  early with no parsed call. Missing tags, malformed JSON, multiple calls, and
  trailing text are recorded as invalid-action audit outcomes. Reward still
  performs the required judge call for every visual trajectory, but an invalid
  strict boundary replaces the candidate with a fixed no-answer sentinel and
  hard-gates answer/tool credit to zero regardless of the verdict.
- No checked-in historical configuration binds
  `StrictNativeDeepEyesAgentLoopV2`. PRL13 still names the legacy
  `NativeDeepEyesAgentLoop`, while the PRL26/PRL27 Crop/TGVF paths use the
  generic `VerlFrameworkNeutralAgentLoop`. Their artifacts also predate the
  `action_boundary_violation_*` telemetry. The strict fix is therefore a
  prospective canonical contract, not evidence that historical trajectories
  had zero malformed boundaries or that their rewards have been repaired.
- The retained PRL26-B official60 and generic86 evaluation outputs each contain
  2,240 trajectories. A post-hoc raw-text scan found zero incomplete calls,
  same-turn multiple calls, or nonterminal suffixes in either set. That is
  useful descriptive evidence, but the artifacts contain no producer-bound
  action-boundary protocol identity, violation count, or violation codes; it
  must not be promoted to strict-v2 telemetry.
- A read-only counterfactual replay below
  `artifacts/policy/PRL-13-A-qwen3-instruct-grpo-bs256-n16-native-crop-t1-stratified-80step-gpu0123`
  covered 13 retained trajectory and validation JSONL files (2,828 unique
  non-empty trajectories; 7,439 parsed
  assistant turns) classified each retained assistant turn with the current
  strict outer-boundary classifier. It found 790 trajectories with at least
  one strict violation, including 447 rows that historically had both
  `acc == 1` and a positive score. At turn level there were 1,369 violations:
  1,234 multiple-call turns, 128 terminal-suffix turns, and seven malformed-tag
  turns. This establishes material legacy/strict divergence for the retained
  PRL13 evidence. It is not a reward recomputation, does not inspect visual
  execution, and cannot be extrapolated to rollouts that were not retained;
  the 1,369 turn count is not a unique-trajectory count. PRL26/PRL27 use a
  different loop and are not classified by this scan.
- Native-visible evaluation is a cross-runtime robustness measurement, not a
  training-matched result for precomputed checkpoints.
- `image_max_pixels` must be bound in both the training and evaluation
  identities; an override creates a different comparison contract.
- The temporary `policy-e2e-explicit-observation-run-config-v1` bridge covers
  the older generic runtime only. It must not flatten or replace the newer
  method-specific NoTool/Crop/TGVF/Atomic @512 schemas; those constraints are
  ported and versioned separately before any four-arm launch is authorized.

## Control-plane gate

- A supervisor may emit a readiness receipt without launching the next stage.
- A mutating launch requires a bounded, run-identity-bound authorization token
  and must record token consumption.
- Waiters must have a bounded timeout and liveness checks.
- Credentials must be passed to the target session/process only. Global tmux
  environment mutation is forbidden.
- Direct entry-point coverage is recursive and exact, not marker-based. The
  v2-schema inventory is at revision 7 and contains 79 content-bound rows. It
  covers every Python file in `tools`/`spikes`, all shell files there, and `src`
  modules exposed by a main guard, `__main__.py`, a Python shebang, executable
  mode, or a `project.scripts` console entry. The one non-entry support module
  is admitted only under a strict import-only syntax class.
- Thirty-six permanent Python entries use only `os` before an exact top-level
  handoff to absolute `/usr/bin/python3 -I` and the quarantine controller;
  their helper guard and final uncaught dispatch are separately checked. The
  script-level guarantee starts after caller-interpreter startup: direct
  executable files have fixed isolated shebangs, while an explicit ordinary
  `python script.py` caller remains an external trust boundary because its
  `sitecustomize` can run before repository code. The remaining shell entry uses exact
  `/bin/bash -p` wrappers containing no PATH-resolved pre-guard command, reject
  a final script symlink, resolve ancestor symlinks physically, and execute
  `/usr/bin/python3 -I` directly. Hostile `PATH`, `BASH_ENV`, and `PYTHONPATH`,
  symlink attacks, caught guards, fake argparse receivers, comment/heredoc
  markers, and pre-guard runtime imports are covered by executable mutation
  tests.
- Every public mutating branch begins with an argument-free runtime-closure
  assertion and then matches an exact reviewed statement/call shape through
  preflight, authorization consume, and dispatch. Descriptor-owning branches
  place preflight outside one handler-free prepared-lifetime `try/finally`, with
  authorization and dispatch inside and only
  `prepared.close_python_binding()` in `finally`. Nested calls, extra effects,
  caught dispatch, a wrong finalizer, or import/rebind drift fail the audit.
  Guard, preflight, and dispatch names have one exact local definition or
  direct import provenance and cannot be rebound, shadowed, or deleted. The
  internal worker similarly requires its exact inherited-receipt verification
  followed by an argument-free closure assertion. Seven blockers keep runtime
  closure disabled, so this remains containment rather than launch authority.
- Only enumerated read-only modes may pass a mixed-mode guard;
  notably replay `plan` and checkpoint-storage `inventory` remain available,
  while GPU replay and compact/delete dispatches remain quarantined.
- Twenty-one reviewed write-once/content-consistent CPU materializers remain
  usable through the explicit `bounded_artifact_write` class. External
  judge/W&B entry points, arbitrary downstream exec, destructive maintenance,
  and direct GPU/training/evaluation entries are quarantined unless migrated
  to the canonical authorization path.
- The mixed-retained-pool CLI is additionally quarantined: the repository's T1
  final-scoring producer publishes schema v1, its consumer accepts only schema
  v2, and no content-bound bridge or native v2 producer is checked in. Existing
  synthetic v2 unit fixtures validate consumer internals but do not establish
  producer-to-consumer provenance. A future repair needs an end-to-end test over
  the exact producer output before the CLI can leave quarantine.

## Verification gate

- At the C0 annotated tag, the full CPU suite passed with 2,112 tests, five
  explicitly justified skips, zero failures, and four non-failing upstream or
  numeric warnings. This is a preserved baseline result, not the current
  consolidation-head count.
- The C0 independent clean-runner simulation masked local model directories and
  undeclared optional stacks while retaining declared test dependencies. It
  passed with 2,016 tests, 91 explicit optional-integration/local-model/CUDA
  skips, zero failures, and zero warnings. This closed the 13 cases that the
  first clean-runner audit exposed despite the richer local environment being
  green.
- At consolidation commit `ccef450`, the full hermetic CPU suite completed with
  2,193 passed, five explicit environment/dependency/history skips, four
  non-failing warnings, and zero failures in 118.46 seconds. The boundary audit
  passed with 73 visible debts and zero violations; the control-plane audit
  passed with 79 surfaces and zero violations. This is a historical snapshot;
  its local count is superseded by the later local suite. Each later remote
  status is stated separately below rather than inherited from this snapshot.
- At verified code checkpoint `4dd8267`, repository boundary policy v3 revision
  7 passes
  with 61 debts and zero violations, and execution-surface policy revision 7
  passes with 79 surfaces and zero violations. The compile-verifier
  import-firebreak follow-up has 2,427 passed, five skipped, four warnings, and
  zero failures in 143.16 seconds, with 69 migration-focused passes. The
  preceding authorization-proof consumption follow-up has 2,386 passed, five
  skipped, four warnings, and zero failures in 141.13 seconds, with 289 focused
  passes in 64.83 seconds. The
  preceding strict child-environment final hermetic CPU suite has 2,383 passed,
  five skipped, four warnings, and zero failures in 143.29 seconds. Its
  expanded focused selection has 285 passing tests in 64.77 seconds; its
  earlier core aggregate had 126 passing tests. The preceding fd checkpoint's
  focused selection had 304 passing tests before the added size-tamper
  parameter, and its updated fd security file passed 10/10. That fd checkpoint
  also has a preserved hermetic CPU result of 2,337 passed, five skipped, four
  warnings, and zero failures in 140.25 seconds. Predecessor commit `a5dd0d1`
  is green for install, lint, both audits, and full CPU after five private-path
  fixtures were made hermetic. Fd-closure head `ab508c4` independently passes
  install, lint, both audits, and the complete CPU suite in remote run
  `33300849634`. Strict child-environment head `5c058a1` independently passes
  the same stages in remote run `33302879219`. Authorization-proof head
  `311770f`, compile-line FIFO fix `2488487`, atomic envelope `cd1eb5e`, and
  UTF-8 hardening `4bba7e9` pass complete remote runs `33304029789`,
  `33305858959`, `33306088543`, and `33306353388`, respectively. Runtime checkpoint
  `c60828d` has 50 runtime-locator, two isolated import/firebreak, 201
  repository/control, and 404 complete `tests/ops` passes. Its full hermetic
  suite has 2,508 passed, five skipped, four warnings, and zero failures in
  139.30 seconds; complete remote run `33307853768` is green.
  Representation-startup checkpoint `4dd8267` has 148 focused, 201
  repository/control, and 482 complete `tests/ops` passes. Its full hermetic
  suite has 2,586 passed, five skipped, four warnings, and zero failures in
  141.50 seconds; complete remote run `33309273768` is green.
- Ruff passes across `src`, `tools`, `spikes`, and `tests`; `git diff --check`,
  shell syntax, the repository-boundary audit, and the control-plane audit also
  pass.
- Unit tests must not require private historical artifact directories.
- CI installs the explicit CPU test dependencies, unsets the external-judge
  credential, hides GPUs, runs both repository audits, and executes the full
  suite on every change. Optional veRL/vLLM tests skip only when their real
  integration dependency is unavailable.
- No fresh RL/evaluation launch is authorized until these gates pass and the
  resulting commit is reviewed.

## Structural debt after stabilization

The control work makes the present repository auditable; it does not pretend
that the source tree is already small or easy to extend. At C2 completion the
tree contained 263 Python modules and 29 registered production modules above
1,000 lines. That is a historical C2 snapshot. Policy revision 5 later retained
20 exact oversized-module exceptions after nine additional decompositions;
that is the historical `fd8da97` snapshot. Current revision 7 retains 17 after
the result-registry and CLI/authorization splits, and the new leaves create no
replacement oversized file.
The three first decomposition targets are complete:
`policy_coredev.py`, representation internal evaluation, and `run_config.py`
are now 902, 921, and 983 lines. Policy v3 now gives every remaining large
module an exact no-slack ceiling, stable owner, rationale, and named next split.
They remain visible debt and still require actual decomposition over time.

The four known multi-module import cycles are gone and a full-tree SCC test
guards that boundary. Exact-equivalent canonical JSON helpers and three
immutable-publication consumers now share tested implementations. Secure-file
reading remains a partial migration, but it now covers result-registry support
reads,
configuration payloads and external bindings, and weight-snapshot storage and
loading with explicitly tested descriptor/probe boundaries. The 79
execution/support surfaces all have one machine disposition, which is
containment rather than portable runtime closure.

The ordered reduction, compatibility/deletion rules, and measurable definition
of done are in
[`CODEBASE_CONSOLIDATION_PLAN_20260830.md`](CODEBASE_CONSOLIDATION_PLAN_20260830.md).
C4's separate inventory is read-only and did not execute or authorize
historical-state cleanup.
