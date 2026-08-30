# Repository stabilization audit — 2026-08-30

Status: active stabilization baseline. This document records the state that
must be preserved while the repository is consolidated. It is not an
authorization to launch training or evaluation.

## Canonical stabilization line

- Worktree: `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl-stabilize-protocol-v1`
- Branch: `stabilize/protocol-contract-v1-20260830`
- Base: `origin/main` at `43f2295c15da2bbc14972af6b63f0038e7789c3f`
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
| Repository ownership | partial | The boundary audit passes with zero new violations and content-binds frozen legacy config roots; 45 exact historical debts remain registered after deleting five stale shell-path entries, and neutral-path semantic RP/PRL debt still needs a second policy layer. |
| Execution-surface inventory | verified | The v2 recursive exact-set manifest binds 82 entries and their file bytes: every Python file below `tools`/`spikes`, every shell file there, and every `src` main/module-main, Python-shebang, executable-bit, or console-entry target. Missing, extra, duplicate, moved, reclassified, or content-drifted entries fail closed. |
| Public and historical launch surfaces | contained, not closed | Legacy/spike surfaces are inventoried and guarded. Independent review found unresolved worker-envelope, executable-FD, environment, member-claim, authorization-transaction and runtime-package gaps. Policy v2 therefore sets the non-overridable canonical runtime closure to disabled. |
| Crop observation/action contracts | verified for the stabilization runtime | Matched60, legacy generic86 and strict/legacy action semantics are explicit identities with focused tests. No historical PRL/PRL config binds the new strict loop, so this does not retroactively certify old artifacts. |
| TGVF/Atomic observation layout | verified | Live append and immutable replay/layout consume the same once-rendered protocol/dialect-bound bytes; the layout has no implicit Thinking renderer. |
| Historical TGVF/Atomic impact | verified | Implementation commits `b100d3d`, `ec0555b`, `8e6b3d`, `5baddc` and provenance checkouts `b87126a`, `017b507`, `001838b` explicitly pass one dialect-bound renderer to appender and layout. Training replay consumes the recorded token rows, so the unused fallback does not downgrade those rows. Their runtime is `training_run`/precomputed, not official-visible/native-pixel. |
| Result comparisons | contained, not closed | Registry v2 verifies score-file bytes/content and independent preregistration bytes, but those artifacts do not bind a score to evaluation identity, the exact trajectory set, weights and the full comparison contract. V2 now rejects every `golden` status and every numeric delta. |
| Policy compile prerequisites | blocked | The hidden worktree-local default is removed and a strict content-bound v1 manifest now binds four minimum declared files. Launch remains blocked because recursive Python headers and the compiler system-toolchain are not yet closed by the manifest schema. |
| Snapshot filesystem closure | partial | LoRA closure reads use descriptor-relative traversal and immutable publications use no-replace semantics. Full-model freeze stores immutable manifest/receipt records rather than copying the external weights; official loading now hashes the complete bound checkpoint/model closure and repeats that verification immediately before vLLM construction, including same-size mutation tests. vLLM 0.12 exposes neither a loaded-adapter nor loaded-full-model digest, so same-UID mutation after the final verification remains a documented runtime residual. |
| Test discovery and behavior | verified | The final Python 3.12 collection discovered 2,117 tests with zero collection errors. The hermetic CPU run completed with 2,112 passed and five explicitly justified skips; no test failed. |

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

At baseline there were 71 registered worktrees before this stabilization
worktree was added, 61 local branches, and 34 `origin/*` remote-tracking
branches. The new stabilization worktree/branch raises the first two counts by
one.

A read-only inventory after adding the stabilization line reports 72 existing
worktrees: 64 clean and eight dirty, with none missing, locked or marked
prunable. Of 62 local branches, eight are merged into `origin/main` and 54 are
not. The 64 clean worktrees are cleanup candidates, not disposable evidence:
this audit does not remove them or their branches. A later cleanup must first
materialize an explicit keep/remove table, preserve every unmerged branch, and
obtain operator approval for the exact worktree paths.

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

`configs/ops/experiment_execution_policy.json` revision 2 currently declares
`runtime_closure.launch_enabled=false`. Canonical Policy training,
representation training, their workers, and representation internal evaluation
fail before token consumption or unsafe artifact loading while any blocker
remains. The exact blocker identifiers cover:

- an immutable, content-bound runtime code package (the current `.venv312`
  editable installation points at the separate base worktree);
- descriptor-bound Python execution rather than re-opening a mutable symlink;
- an exact worker argv/environment startup envelope and per-member claims;
- a strict child-environment allowlist;
- a safe non-pickle representation-evaluation artifact;
- recursive compiler/header/system-toolchain closure;
- atomic publication of the combined one-time authorization transaction.

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
  v2 inventory contains 82 content-bound rows. It covers every Python file in
  `tools`/`spikes`, all shell files there, and `src` modules exposed by a main
  guard, `__main__.py`, a Python shebang, executable mode, or a
  `project.scripts` console entry. The one non-entry support module is admitted
  only under a strict import-only syntax class.
- Thirty-six permanent Python entries use only `os` before an exact top-level
  handoff to absolute `/usr/bin/python3 -I` and the quarantine controller;
  their helper guard and final uncaught dispatch are separately checked. The
  script-level guarantee starts after caller-interpreter startup: direct
  executable files have fixed isolated shebangs, while an explicit ordinary
  `python script.py` caller remains an external trust boundary because its
  `sitecustomize` can run before repository code. Four shell entries use exact
  `/bin/bash -p` wrappers containing no PATH-resolved pre-guard command, reject
  a final script symlink, resolve ancestor symlinks physically, and execute
  `/usr/bin/python3 -I` directly. Hostile `PATH`, `BASH_ENV`, and `PYTHONPATH`,
  symlink attacks, caught guards, fake argparse receivers, comment/heredoc
  markers, and pre-guard runtime imports are covered by executable mutation
  tests.
- Every public mutating branch begins with an argument-free runtime-closure
  assertion and then matches an exact reviewed statement/call shape through
  preflight, authorization consume, and dispatch. Nested calls or extra side
  effects cannot hide in a guard/preflight argument or between those
  boundaries. Guard, preflight, and dispatch names have one exact local
  definition or direct import provenance and cannot be rebound, shadowed, or
  deleted. The internal worker similarly requires its exact inherited-receipt
  verification followed by an argument-free closure assertion. Runtime closure
  remains disabled, so this is containment rather than launch closure.
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

- The full CPU suite passes on the stabilization worktree: 2,112 passed, five
  explicitly justified skips, zero failures, and four non-failing upstream or
  numeric warnings.
- An independent clean-runner simulation masked local model directories and
  undeclared optional stacks while retaining declared test dependencies. It
  passed with 2,016 tests, 91 explicit optional-integration/local-model/CUDA
  skips, zero failures, and zero warnings. This closed the 13 cases that the
  first clean-runner audit exposed despite the richer local environment being
  green.
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
that the source tree is already small or easy to extend. The measured cleanup
backlog includes 31 production modules above 1,000 lines, four multi-module
import cycles, repeated local hashing/publication helpers across more than 100
files, 82 inventoried execution/support surfaces, and no portable runnable canonical model
config yet. `policy_coredev.py`, representation internal evaluation, and the
policy run-config union are the first decomposition targets.

The ordered reduction, compatibility/deletion rules, and measurable definition
of done are in
[`CODEBASE_CONSOLIDATION_PLAN_20260830.md`](CODEBASE_CONSOLIDATION_PLAN_20260830.md).
Those refactors happen after this fail-closed baseline; mixing broad module
moves into the current security/provenance commit would make both harder to
review.
