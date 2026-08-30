# Codebase boundaries

This repository separates reusable implementation from historical experiment
evidence. The boundary is enforced by a CPU-only, fail-closed audit; it is not
only a naming convention.

The measured debt and ordered reduction plan are tracked in
[`CODEBASE_CONSOLIDATION_PLAN_20260830.md`](CODEBASE_CONSOLIDATION_PLAN_20260830.md).
The frozen worktree/ref review is recorded separately in
[`WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md`](WORKTREE_BRANCH_RETENTION_INVENTORY_20260830.md).
That inventory is read-only and does not authorize any deletion.

## Code ownership boundary

Neutral source is reusable code whose identity does not depend on a particular
RP or PRL run. New implementation belongs under neutral modules in
`src/tgvf_rl` or in reusable, neutrally named tools. Neutral Python source must
not import a run-specific RP/PRL module.

Some historical source and tool paths include RP or PRL identifiers. They are
quarantined compatibility or evidence code, not canonical extension points.
The boundary policy records every such path in an exact allowlist. A new or
renamed run-specific source/tool path is a violation; it must not be accepted
merely because it resembles an existing historical path.

## Configuration ownership boundary

All future canonical experiment configurations have exactly these homes:

- `configs/canonical/policy` for policy training and rollout contracts;
- `configs/canonical/representation` for representation training contracts;
- `configs/canonical/evaluation` for evaluation contracts.

`configs/ops` contains repository operations and control policies, including
the boundary policy itself. It is not a home for scientific run
configurations.

These five legacy roots are frozen evidence-only inventories:

- `configs/evaluation`;
- `configs/overnight`;
- `configs/policy`;
- `configs/representation`;
- `configs/smoke`.

Their many historical files are deliberately not promoted one by one. The
policy pins each root's file count, hash of its sorted relative-path inventory,
and a content-tree SHA-256 over every relative path and the exact bytes of the
corresponding file. The content-tree stream uses domain separation, then an
unsigned 64-bit big-endian file count and sorted path/content records whose
UTF-8 path and byte content are each prefixed by an unsigned 64-bit big-endian
length. Concatenation therefore cannot make two different trees equivalent.
Additions, removals, renames, or a one-byte change to an existing file fail the
audit. Existing files remain historical evidence and are not templates for new
work. Any intentional historical correction or migration requires explicit
review and a deliberate policy revision; ordinary development must leave these
roots unchanged.

Canonical configuration roots must exist as real directories in a clean
checkout and must not be symbolic links. Canonical configuration paths must be
portable and neutrally named. They must not contain RP/PRL run identifiers or
machine-specific absolute paths.

The historical policy-evaluation v1 schemas are readable evidence records but
are not launch contracts: they predate explicit pixel, success-observation,
action-boundary and complete snapshot bindings. Newly materialized policy
evaluations use the v2 constants owned by
`evaluation/policy_evaluation_config.py` and compatibility re-exported by
`policy_coredev.py`, plus the schema aid at
`configs/canonical/evaluation/policy_evaluation_v2.schema.json`. A v1 file is
never upgraded in place, and a new measurement must use a new evaluation ID
and output root.

## Evidence ownership boundary

Typed registries and future preregistration records live under `evidence/`;
they are neither executable run configurations nor raw experiment artifacts.
For example, policy-result classifications are owned by
`evidence/policy/result_registry_v2.json`. Registry artifact paths are
repository-relative bindings to immutable evidence, while the usually large
artifact trees themselves remain outside this source migration and are never
silently copied into `configs/`.

Generated manuscript tables must be materialized from a typed registry after
its artifact checks pass. A document is a consumer of evidence, not an
alternative score database.

The result-registry implementation is split along that ownership boundary:
`evaluation/result_registry.py` is a 292-line loading/facade surface,
`evaluation/result_registry_schema.py` owns 839 lines of enums, immutable
records, and comparison validation, and
`evaluation/result_registry_support.py` owns 225 lines of primitive validators
and secure-read support. Historical imports, object identity, pickle
coordinates, and type hints remain characterized; the facade is not a second
implementation.

## Artifact primitive ownership

Canonical JSON bytes and hashes with exact matching semantics belong to
`tgvf_rl.artifact_contracts`. Create-only and content-consistent immutable
publication belong to `tgvf_rl.immutable_publication`. Descriptor-bound
regular-file reads and metadata-only presence probes belong to
`tgvf_rl.secure_file_read`. Its path contracts are deliberately separate:

- the leaf reader refuses a symlink at the final component but retains normal
  operating-system semantics for ancestor symlinks;
- the absolute reader and probe open each POSIX path component relative to the
  already-open parent descriptor, refuse symlinks throughout the chain, open
  the leaf nonblocking, and require it to be a regular file;
- the beneath-root reader accepts only a normalized relative path and anchors
  every lookup below an already-bound directory descriptor.

The read contracts return bytes plus descriptor metadata from immediately
before and after the read. The probe contract returns one metadata snapshot,
reads no payload bytes, and closes the descriptor before returning. A probe is
therefore only a point-in-time presence/type observation, not durable authority
for a later open. Descriptor binding keeps each opened component or leaf
anchored despite a later pathname rebind, but it does not prevent in-place
writes to the same file during a read. The shared primitive verifies
regular-file type; it does not compare the before/after metadata or impose a
content digest.
Consumers that require immutable or content-bound input must add that check.

Migration is semantic and incremental. Similarly named helpers must not be
merged when they differ in symlink, stability, error, ownership, or publication
semantics. At this checkpoint the secure-reader production consumers are
`evaluation/result_registry_support.py`,
`framework/verl/policy_weight_snapshot_store.py`, and
`representation/training/config_values.py`; the existence of the shared leaf
does not claim that every historical reader has been migrated.

## Public API compatibility boundary

`tgvf_rl.public_api_compat` is the shared neutral leaf for preserving a moved
implementation object's historical facade and pickle coordinates. It changes
only implementation-owned functions and accessors, does not rewrite function
globals, and does not emulate arbitrary facade monkeypatching. Implementation
leaves must not reverse-import the facade to establish compatibility.

Postponed class annotations need an additional, opt-in leaf-first boundary.
Before rebinding a class to its public module, a leaf may call
`freeze_public_class_annotations` with the defining leaf's globals passed
explicitly. The helper resolves with those globals and class-local names, then
replaces only the keys in the class's own `__annotations__`; inherited
annotations are neither copied nor frozen. This permits `typing.get_type_hints`
to work when the leaf is imported before the facade, without importing the
facade in the reverse direction.

Freezing deliberately changes those raw annotation values from postponed
strings to resolved type objects. It is appropriate only where leaf-first type
hints and stable historical module coordinates outweigh preservation of the
raw strings. Existing dataclass field metadata is not retroactively rewritten,
so callers and tests must not assume that `dataclasses.fields(...).type` changes
with the class's own `__annotations__`.

## 2026-08-30 fd-closure checkpoint

Canonical representation and Policy launches now retain the exact regular-file
descriptor opened during preflight and pass that integer descriptor to their
respective `os.execve` boundaries. Binding uses absolute component-by-component
no-follow traversal, compares the retained candidate's device and inode with
`/proc/self/exe`, and fails closed when the platform cannot establish either
process identity or descriptor-based `execve`. Immediately before process
replacement, the same descriptor is reread and its digest, size, device,
inode, mode, executable bit, and prepared identity are revalidated. There is no
path reopen and no `/proc/self/fd` fallback at the execution boundary;
`argv[0]` remains the declared Python path.

Descriptor ownership is process-local and non-serializable. Explicit close and
the retained owner's `weakref.finalize` cover authorization failure, preflight
failure, failed exec, and abandoned prepared plans without closing a later fd
that reused the same number. Adversarial coverage includes candidate-replacement
races, payload/size/mode tampering, real fd execution after pathname
replacement, descriptor lifetime and reuse, and AST mutations for caught
dispatch, extra effects, wrong finalizers, wrong imports, and rebinding. A
second independent security review found no blocker to removing
`fd_bound_python_exec_missing`.

At this fd-only checkpoint, experiment execution policy revision 3 removed only
that blocker. `runtime_closure.launch_enabled` remained `false` with seven
blockers, and `child_environment_allowlist_missing` was the next priority. The
later child-environment checkpoint below supersedes that specific status.
Same-inode mutation between final verification and `execve` remains within
`immutable_runtime_code_package_missing`; fd closure does not claim to solve
immutable runtime packaging.

## 2026-08-30 strict child-environment checkpoint

The two canonical launch paths now own distinct fixed child-environment
profiles. Each environment is constructed from an empty mapping: host names are
classified into ignored/rejected audit identities, while host values are never
retrieved and no host entry is copied. Fixed baseline entries, explicitly
owned profile values, and exact late overlays are authorization-bound. The
final representation boundary admits only its six CLI worker-authorization
fields before `torchrun`; the final Policy boundary admits those six fields
plus its two compile-receipt fields. The corresponding internal entry points
revalidate the base identity and exact expected late field set. The historical
sanitized child-environment helpers remain for API compatibility only;
canonical launches no longer use them.

A real two-worker `torchrun` smoke under pinned PyTorch 2.9 establishes the
complete 16-field worker delta: `GROUP_RANK`, `GROUP_WORLD_SIZE`, `LOCAL_RANK`,
`LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `ROLE_NAME`,
`ROLE_RANK`, `ROLE_WORLD_SIZE`, `TORCHELASTIC_ERROR_FILE`,
`TORCHELASTIC_MAX_RESTARTS`, `TORCHELASTIC_RESTART_COUNT`,
`TORCHELASTIC_RUN_ID`, `TORCHELASTIC_USE_AGENT_STORE`, and `WORLD_SIZE`. Both
workers add exactly that set and load `tgvf_rl`, the CLI, and the
child-environment module from the stabilization repository. The fixed
repository `PYTHONPATH` prevents the observed wrong-worktree resolution; it
does not provide an immutable code package or close Python startup, `.pth`, or
import-before-authorization execution.

Experiment execution policy revision 4 replaces
`child_environment_allowlist_missing` with
`role_scoped_judge_secret_transport_missing`.
`runtime_closure.launch_enabled` remains `false` with seven blockers. The
expanded focused selection has 285 passing tests in 64.77 seconds, and the
earlier core aggregate had 126 passing tests. The final hermetic CPU suite has
2,383 passed, five skipped, and four warnings in 143.29 seconds.
Repository-boundary revision 7 passes at 61 debts and zero violations;
execution-surface revision 5 binds 79 surfaces and the control-plane audit has
zero violations.

This promotion is deliberately narrow. It does not establish an immutable
runtime code package, protection from caller Python/`.pth` or imports before
authorization, a complete worker startup envelope, per-member claims,
role-scoped judge-secret transport, or exact Ray-descendant environments.

## 2026-08-30 authorization-proof consumption follow-up

A CPU-only Ray 2.56.1 audit demonstrated that the six validated `TGVF_CLI_*`
fields and two Policy compile-receipt fields would otherwise be inherited by
Ray infrastructure and actor roles. Representation now removes its six fields
immediately before training dispatch; Policy removes all eight after compile
receipt verification and before Hydra composition or Ray startup. The helper
requires the complete exact field set before deleting any entry and preserves
torchrun rank state and persistent runtime settings.

This prevents environment inheritance only. It does not delete guarded receipt
files, revoke already verified process-local objects, define role-specific Ray
envelopes, or solve startup/member/secret authority. Experiment policy revision
4 and its seven blockers are unchanged. Execution-surface revision 6 binds the
changed files; 289 focused tests pass in 64.83 seconds and the full hermetic CPU
suite has 2,386 passed, five skipped, and four warnings in 141.13 seconds. Both
audits remain green with zero violations.

## 2026-08-30 worker-startup identity scaffold

The light `tgvf_rl.ops.worker_startup` leaf defines exact Policy driver,
Representation launcher, and Representation member roles. It content-binds
the complete argv, target, and future runtime-package and dependency-root
digests. Its verified evidence is PID-bound, non-copyable, non-pickleable, and
not publicly constructible. Thirty-nine focused tests pass; commit `6cb133d`
passes the complete remote workflow in run `33304509263`.

This contract is not yet the canonical executable entry. Runtime/dependency
locators, fixed role dispatch, dual Representation envelopes, and member claims
remain open, so experiment policy revision 4 and all seven blockers are
unchanged.

## 2026-08-30 compile-verifier import-firebreak follow-up

Policy compile-prerequisite verification is now owned by the dependency-light
`tgvf_rl.ops.policy_compile_prerequisites` leaf. The historical
`tgvf_rl.framework.verl.compile_prerequisites` module is an exact compatibility
facade and retains all established public and pickle coordinates. Canonical
Policy production imports use the light leaf.

An exact-environment `python -B -P -S` test proves the leaf imports without
Torch, NumPy, Hydra, Ray, veRL, or `tgvf_rl.framework`. This is a prerequisite
for authorization-before-runtime bootstrap work, not an immutable runtime
package or complete startup envelope. Experiment policy revision 4 remains
frozen with the same seven blockers. Execution-surface revision 7 binds the
changed Policy worker and control utility; the full hermetic CPU suite has
2,427 passed, five skipped, and four warnings in 143.16 seconds. Boundary and
control audits pass locally at 61 debts/zero violations and 79 surfaces/zero
violations. The first two remote attempts, runs `33305253971` at `2c2b17b` and
`33305571095` at `54fb8b9`, exposed FIFO regression tests that counted the
dependency-heavy import phase against the short refusal timeout. Test-only
commit `2488487` separates a 30-second import-readiness phase from the
five-second FIFO-refusal phase; the complete remote workflow is green in run
`33305858959`.

## 2026-08-30 atomic worker-startup envelope follow-up

Commit `cd1eb5e` adds one canonical `WorkerStartupEnvelope` around the existing
worker identities. A Policy envelope has exactly the `policy-driver` role. A
Representation envelope has exactly the `representation-launcher` and
`representation-member` roles, and a member cannot be the entry role. Nested
records use strict canonical JSON: duplicate or unknown fields, non-finite
values, non-canonical spelling, changed identity digests, and an incorrect role
set fail closed. One envelope contributes exactly three collision-free
authorization parameters: its schema, canonical JSON, and SHA-256. The legacy
flat identity API remains unchanged. This commit is green across the complete
remote workflow in run `33306088543`.

Commit `4bba7e9` additionally refuses a command argument that cannot be encoded
as UTF-8, including a lone surrogate, before canonical serialization. It is
green across the complete remote workflow in run `33306353388`.

These contracts are not yet connected to canonical launch dispatch. They do not
provide member claims, a formal stage-0 bootstrap, or role-scoped secret
transport. Experiment policy revision 4 therefore remains unchanged:
`runtime_closure.launch_enabled=false` with the same seven blockers.

## 2026-08-30 explicit runtime-locator scaffold

Commit `c60828d` adds a dependency-light, explicitly declared runtime locator.
Its strict canonical manifest is externally bound by source SHA-256 and byte
length and declares the executable, cache tag, authorized target coordinates,
the exact pure-Python import root above `tgvf_rl/`, and ordered dependency
roots. Descriptor-relative no-follow traversal verifies each exact directory
and regular-file inventory, rejects bytecode and native runtime-package
candidates, and retains the verified import/dependency root descriptors in
PID-bound, non-copyable, non-pickleable evidence. Declared `.pth` files remain
inert bytes. A fresh `python -B -P -S` test imports both authorized targets and
a dependency through duplicated `/proc/self/fd` roots passed with `pass_fds`.

Local validation has 50 runtime-locator tests, two isolated import/firebreak
tests, 201 repository/control tests, and 404 complete `tests/ops` tests passing.
The complete hermetic CPU suite has 2,508 passed, five skipped, and four warnings
in 139.30 seconds. Commit `c60828d` is green across the complete remote workflow
in run `33307853768`. The evidence deliberately reports `closure_complete=false`
and the exact residual
`same-uid-mutable-executable-and-runtime-trees-have-no-atomic-observation-during-or-after-verification-v1`.
Sequential verification is not one atomic filesystem snapshot; the executable
descriptor is not retained through execution; and the formal stage-0 loader and
module-origin proof is absent. This scaffold neither supplies an immutable
runtime package nor closes a launch blocker. Experiment policy revision 4 and
all seven blockers remain unchanged.

## 2026-08-30 Representation startup authorization scaffold

Verified code checkpoint `4dd8267` reconstructs a complete
`WorkerStartupEnvelope` authorization group before use. The reconstruction
requires an exact dictionary with exact-string keys. Within that broader CLI
map, the protected `worker_startup_` namespace must contain exactly its three
envelope names with exact-string values. Reconstruction then
revalidates the canonical JSON, schema, SHA-256, and expected entry role. Review
found that the original implementation scanned the protected namespace before
rejecting a `str` subclass key. The final implementation rejects every
non-exact-string key before that scan, closing the namespace blocker.

The dependency-light `RepresentationMemberClaim` and
`RepresentationStartupPlan` form pure authorization data for a single-node
Representation launch. Only world sizes two and four are admitted. Global and
local ranks must each cover exactly `0..world_size-1`, every member has equal
global/local rank, and physical GPU IDs form a one-to-one mapping. The plan
binds the complete envelope and full physical GPU mapping. Each claim binds the
envelope and Representation member digests, run/configuration identities,
world size, and its own rank/GPU assignment.
The plan's standalone authorization group is an exact schema/JSON/SHA-256
triple; an individual member claim is not standalone authority.

This checkpoint has 148 focused tests, 201 repository/control tests, and all
482 `tests/ops` tests passing. The complete hermetic CPU suite has 2,586 passed,
five skipped, and four warnings in 141.50 seconds; complete remote run
`33309273768` is green. There is no CLI or `exec`
wiring, no member-bootstrap verification, and no authority minted from an
individual claim. Formal stage-0 and immutable-runtime closure also remain
absent. Experiment policy revision 4 stays at
`runtime_closure.launch_enabled=false` with the same seven blockers.

## 2026-08-31 Policy startup authorization binding boundary

Commits `e3aa054` and `48e3f07` make runtime-locator evidence caller-owned,
borrowed process-local evidence and make the resulting singleton Policy startup
envelope `PreparedPolicyLaunch`-owned authorization data. The prepared object
cannot be subclassed; executable identity, full argv, fixed driver target,
runtime/dependency digests, and the exact three-field protected startup
namespace are revalidated before authorization use and again at execution.
Cross-group protected-name injection is rejected.

This does not transfer runtime-root descriptor ownership to the prepared plan
and does not establish worker-bootstrap authority. The canonical CLI currently
supplies no locator evidence, so it cannot reach this scaffold after the
existing closure gates are lifted without further wiring. The focused
Policy/startup selection has 223 passes, and complete remote run `33324381122`
is green. All seven runtime-closure blockers remain.

## 2026-08-31 Representation member selection-only boundary

Checkpoint `1141df1` gives the dependency-light selector ownership only of a
copied selection record. Selection requires the complete exact outer CLI gate
identity and complete materialized worker environment. Changes to any CLI
parameter or raw base, receipt/liveness, or torchrun field change the bound
record or fail verification. Rank selection is derived from the complete
startup plan and exact world/rank/GPU mapping.

The record explicitly states `authorization_scope="selection-only"` and
`replay_protected=false`. Receipt and liveness values are bound as input bytes
but are not consumed as one-use filesystem authority. The selector performs no
runtime-origin verification, target import, dispatch, or
`VerifiedWorkerStartup` minting. Validation has 135 focused, 201
repository/control, and 553 complete `tests/ops` passes. The full suite has
2,672 passed, five skipped, and four warnings in 146.52 seconds; complete remote
run `33324892712` is green.

A future member bootstrap must add one-use receipt consumption and
immutable-runtime verification. The seven launch blockers remain unchanged. At
code checkpoint `1141df1`, the canonical line is 124 commits ahead of local
`main` and 82 ahead of `origin/main`, while physical `main` remains dirty with
the same 93 collapsed status entries; this boundary does not authorize
promotion.

## 2026-08-30 canonical line versus physical `main`

At verified code checkpoint `4dd8267`, the graph is linear: the stabilization
line is 120 commits ahead of local `main` and 78 ahead of `origin/main`, with no
commits on either comparison
side that are absent from the stabilization line. Local `main` itself is 42
commits behind `origin/main`. The stabilization delta from local `main` spans
299 paths: 140 additions, 156 modifications, and three deletions.

That graph permits a fast-forward, but the physical `main` worktree is not
promotion-ready. It has 93 collapsed status entries: 38 unstaged tracked
modifications and 55 untracked entries, with no staged, deleted, renamed, or
copied entry. Expanding untracked directories yields 132 entries, including 94
untracked paths and 14 paths below `.worktrees/`. Nineteen dirty paths overlap
the stabilization delta. Eighteen are modify/modify and pass a read-only textual
merge probe; `tools/supervise_prl14_cleanfinal16_eval.sh` is modified on
physical `main` but deleted on the stabilization line and needs an explicit
decision. A fast-forward does not merge any of this uncommitted state.

Two clean detached worktrees also have no containing local branch,
remote-tracking branch, or tag:

- `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.eval-runtime-rp70-20260802`
  at `2d61b07995b1d5b90c221fe1faf5090e8d985fef`;
- `/tmp/prl26-gap-fix-NNVbZQ/worktree` at
  `151701fdb13c5ecf6fac6c0f67760e51c427c277`.

The canonical line is a real, tested integration line; it is not merely an
audit. Physical `main` promotion, preservation tags, archival, and deletion
remain separate operator-authorized actions.

## Absolute-path debt

Machine-specific absolute paths under source and tools are prohibited for new
code. The current policy preserves only the already-known debt using the exact
repository-relative file path, SHA-256 of the literal, and occurrence count.
The literal values themselves are not copied into the portable policy.

The audit reports two separate classes:

- `debts` are exact, known historical matches. They remain visible even when
  the audit passes and should be removed over time.
- `violations` are new debt, boundary crossings, policy drift, missing expected
  debt, or unreadable/untrusted inputs. Any violation blocks the audit.

The allowlists are a ratchet, not a suppression mechanism. Never change the
baseline simply to absorb a new machine path, run-specific module, or legacy
configuration. Fix the new debt at its source. A baseline change is acceptable
only for an intentional boundary migration with explicit review and evidence.

## Production module size debt

Repository-boundary policy `TGVF-REPOSITORY-BOUNDARIES-V3` is currently at
policy revision 7 and fixes the normal production-module limit at 1,000
physical lines. Each of the 17 current exceptions names one exact Python path
below `src/tgvf_rl`, its stable subsystem owner, concrete reason, next split
seam, and a ceiling equal to its current line count. The candidate audit fails
on an unregistered oversized module, growth above a ceiling, slack after a file
shrinks, a missing/moved file, or an exception retained after the module
reaches the limit.

The boundary CLI also accepts an explicit baseline policy. In that mode the
candidate exception paths must be a subset of the baseline and no ceiling may
increase, so changing code and its allowance together cannot relax the
ratchet. The initial v3 inventory is a reviewed bootstrap; its 29 rows are
historical structural debt, not the current exception count, and do not declare
those files well-factored.

## Running the audit

From the repository root, run:

```bash
.venv312/bin/python tools/audit_repository_boundaries.py \
  --repository-root . \
  --policy configs/ops/repository_boundary_policy.json
```

The command emits a structured report, exits successfully only when there are
no violations, does not import model runtimes, and requires no GPU. CI runs the
same audit with `CUDA_VISIBLE_DEVICES` empty before the test suite.

At this checkpoint policy revision 7 passes with 61 debts and zero violations:
five evidence-only configuration roots, 25 machine-specific absolute paths, 17
oversized production modules, and 14 quarantined run-specific code paths.
The former exceptions for `cli.py` and `ops/cli_authorization.py` are gone:
their facades are now 963 and 622 lines. Their extracted
`ops/cli_launch.py` and `ops/cli_authorization_identity.py` leaves are 265 and
659 lines and introduce no replacement size exception.

## Execution-surface ownership

`configs/ops/execution_surface_policy.json` is the machine-readable inventory
of every directly invocable repository entry point and reviewed script support
module. The control-plane audit includes every Python file below `tools` and
`spikes`, plus every `src` module selected by a module-main guard,
`__main__.py`, Python shebang, executable bit, or `project.scripts` console
entry. It also recursively includes every shell file below `tools` and
`spikes`. The discovered and manifested path sets must be exactly equal: an
omitted, extra, duplicate, moved, or newly added candidate blocks the audit.

Each manifest row binds the path, entry kind, classification, allowed and
blocked modes, authorization kind, conservative capability flags, workload
kinds, and SHA-256 of the exact file bytes. A one-byte implementation change
therefore requires review of both the code and its execution classification;
marker searches are not accepted as a completeness proof.

The `check_launch_gate.py` control row is bound to all eight real argparse
commands (`ready`, `authorize`, `override-freeze`, `consume`, `wait`, `status`,
`quarantine-legacy`, and `audit-control-plane`) and to its exact conservative
capability map. Parser-mode or policy drift blocks the audit.

The revision-7 stabilization inventory contains 79 unique paths, and each has
exactly one machine-checked disposition. The current control-plane audit covers
all 79 and reports zero violations. The inventory contains two canonical
entries, two control-audit utilities, 36 permanently quarantined Python
entries, one permanently quarantined shell entry, 13 mixed entries whose
explicitly read-only modes remain available, 21 bounded offline artifact
materializers, three read-only utilities, and one strictly import-only support
module. A materializer receives `bounded_artifact_write` only after its
content-bound implementation has been reviewed for write-once or
content-consistent idempotent publication; arbitrary overwrite and
cross-worktree frozen-config writers are quarantined. GPU,
training/evaluation, external-judge/network, arbitrary exec, and destructive
legacy or spike entries must be permanently quarantined or have a statically
verified mode guard before any business dispatch.

Permanent Python quarantine is structural rather than marker-based. Each of
the 36 entries has an exact top-level `os.execv` handoff through absolute
`/usr/bin/python3` with `-I` to the fixed `quarantine-legacy` controller. The
handoff uses only `os` and appears immediately after the optional module
docstring and `__future__` imports, before repository/runtime imports, path
injection, environment mutation, or other script-level side effects. Its
legacy `main` helper retains a second direct guard, and its final module
dispatch must be direct and uncaught. A guard hidden inside `try`/`except`, a
pre-guard import, or a post-catch dispatch fails the audit even if the file
hash is deliberately refreshed.

Python interpreter startup occurs before any script statement and is therefore
an explicit caller trust boundary. The executable legacy Python files use the
fixed `#!/usr/bin/python3 -I` shebang, and the isolated second hop is always
enforced. An explicit `python script.py` invocation is safe from caller-side
`sitecustomize` only when that caller also uses a trusted isolated interpreter,
for example `/usr/bin/python3 -I`; the repository does not claim to neutralize
hooks already executed by an ordinary hostile caller before its code starts.

Mixed read-only/quarantined Python entries must parse their mode through a
statically traced `argparse.ArgumentParser` receiver, call the mode guard as a
direct statement immediately after that parse, and end in a direct module
dispatch. Calls on an unrelated object named `parse_args`, caught guards, and
nested business dispatch do not satisfy this contract.

The remaining permanent shell entry is an exact byte-bound wrapper. It starts
with `/bin/bash -p`, derives its location using only Bash parameter expansion
and builtins, and `exec` absolute `/usr/bin/python3 -I` into the quarantine
controller. A symlink used as the final script path is rejected before path
derivation, while symlinked ancestor directories are resolved to their physical
directory with builtin `cd -P`. This makes the pre-guard path independent of
hostile `PATH`, `BASH_ENV`, and `PYTHONPATH`; comments and heredocs containing
the expected words cannot satisfy the audit.

Canonical public branches also use structural checks. Each branch begins with
an argument-free runtime-closure assertion and has an exact reviewed call
shape through preflight, one-time authorization consume, and dispatch. For the
two descriptor-owning commands, preflight completes before one strict
prepared-lifetime `try/finally`; authorization and dispatch are its only
reviewed body, it has no handler or `else`, and its `finally` contains only
`prepared.close_python_binding()`. Nested argument side effects, caught
dispatch, extra statements, a different finalizer, and control-symbol import or
rebind drift are rejected. The internal worker has an equivalent exact
inherited-receipt/closure prefix. These checks are containment while the seven
remaining runtime blockers keep launch disabled; fd closure alone does not
declare the public launch implementation complete.

The C1 disposition condition is therefore satisfied, but canonical launch and
runtime closure are not. A machine classification is containment and ownership
metadata, not evidence that every retained surface is a public CLI command.

Three conservative classifications deserve explicit rationale:

- `materialize_policy_selection_sources.py` can transitively download source
  data and invoke an archive subprocess, so it is not an offline-only
  materializer in its current form;
- `validate_vlmevalkit_deployment.py` accepts a caller-supplied deployment file
  whose Python executable is then invoked by subprocess, so it is not a closed
  read-only probe until that input is restricted to a content-bound authority.
- `materialize_policy_t1_mixed_retained_pool.py` is quarantined because the
  checked-in final-scoring producer emits manifest v1 while its consumer accepts
  only manifest v2. No checked-in v2 producer exists, and synthetic v2 unit-test
  fixtures are not an end-to-end provenance chain. Re-enable it only with an
  explicit content-bound v1-to-v2 migrator or a native v2 producer plus a
  producer-to-consumer integration test.

`audit_policy_data_selection_t1_replay.py plan`, the default no-launch plan in
`verl_fsdp2_vllm_sync_smoke.py`, and
`compact_policy_checkpoint_storage.py inventory` remain usable as verified
read-only modes. Their GPU replay and compact/delete modes remain blocked.

Run the combined control-plane audit with no GPU visibility:

```bash
CUDA_VISIBLE_DEVICES='' .venv312/bin/python tools/check_launch_gate.py \
  audit-control-plane --repository-root .
```

At this checkpoint execution-surface policy revision 7 binds all 79 surfaces,
and both the control-plane and repository-boundary audits pass locally. The
predecessor commit `a5dd0d1` is green in remote CI across dependency
installation, lint, boundary, control-plane, and the full CPU suite after five
tests with private machine dataset paths were made hermetic. The fd-closure
checkpoint head `ab508c4` is now independently green in remote CI run
`33300849634`; install, lint, both audits, and the complete CPU suite all
passed. The final local full suite,
including the added size-tamper case, is 2,337 passed, five skipped, and four
non-failing warnings in 140.25 seconds. Before that additional parameter, the
combined focused selection was 304 passed; the updated fd security file passes
10/10 and the final full suite includes the added case. These are preserved
fd-closure results. The subsequent strict child-environment expanded focused
selection has 285 passing tests in 64.77 seconds, with an earlier 126-pass core
aggregate. Its final hermetic CPU suite has 2,383 passed, five skipped, and four
warnings in 143.29 seconds. Commit `5c058a1` passes install, lint, both audits,
and the complete CPU suite in remote run `33302879219`.
The later authorization-proof consumption follow-up has 289 focused passes in
64.83 seconds and a full hermetic CPU result of 2,386 passed, five skipped, and
four warnings in 141.13 seconds. Commit `311770f` passes install, lint, both
audits, and the complete CPU suite in remote run `33304029789`.
The newer compile-verifier import-firebreak follow-up has 69 focused passes and
a full hermetic CPU result of 2,427 passed, five skipped, and four warnings in
143.16 seconds. After two FIFO test-harness timing failures, test-only commit
`2488487` passes the complete remote workflow in run `33305858959`. Atomic
worker-envelope commit `cd1eb5e` and UTF-8 hardening commit `4bba7e9` are green
in remote runs `33306088543` and `33306353388`, respectively. Runtime-locator
commit `c60828d` has 50 focused tests, two isolated import/firebreak tests, 201
repository/control tests, all 404 `tests/ops` tests, and a complete hermetic CPU
suite of 2,508 passed, five skipped, and four warnings in 139.30 seconds. Its
complete remote workflow is green in run `33307853768`. The non-atomic same-UID
mutation residual remains launch-blocking under the existing immutable-runtime
package blocker. Representation-startup authorization checkpoint `4dd8267` has
148 focused, 201 repository/control, and 482 complete `tests/ops` passes; its
complete hermetic CPU suite has 2,586 passed, five skipped, and four warnings in
141.50 seconds; complete remote run `33309273768` is green. It adds no
CLI/`exec` wiring and removes none of the seven
runtime-closure blockers.
