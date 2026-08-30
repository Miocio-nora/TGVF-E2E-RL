# Contributor and automation rules

These rules apply to the whole repository. More specific instructions may be
added in a subdirectory only when that subsystem needs a stricter contract.

## Authority and preservation

1. Treat committed source/configuration plus immutable, hash-bound artifacts as
   authoritative. A document or run label alone is not proof of runtime
   identity.
2. Preserve unrelated and uncommitted work. Never reset, clean, stash, remove,
   or rewrite a dirty worktree as part of another task.
3. Integrate experiment work selectively onto a clean branch from
   `origin/main`; do not merge a complete PRL experiment branch solely to obtain
   one fix.
4. Never infer permission to launch GPU work from a watcher, an available GPU,
   or the existence of a checkpoint.

## Experiment launch boundary

1. Training and evaluation are mutating operations. They require an explicit,
   run-identity-bound authorization token in addition to a complete ledger/run
   configuration.
2. Supervisors may validate prerequisites and emit a readiness receipt. They
   must not auto-launch a later stage without consuming that authorization.
3. Waiters require a bounded timeout and liveness checks. Failure, timeout, and
   cancellation must be distinct terminal states.
4. Pass credentials only to the target process/session. Do not place secrets in
   tmux's global environment, source-controlled files, logs, receipts, or
   artifact identities.

## Protocol and replay invariants

1. Every run explicitly binds its prompt, parser, tool schema, observation
   renderer, visual execution path, image cap, generation budget, RNG policy,
   scorer, and task manifest.
2. Missing protocol fields fail closed. A missing renderer must never select a
   generic fallback implicitly.
3. Render a successful tool observation once. The exact resulting bytes must be
   shared by the live continuation and immutable replay-layout calculation.
4. Historical behavior remains accessible only through an explicitly named
   legacy protocol identity.
5. Native-visible and precomputed-replay paths are different runtimes. Report a
   cross-runtime measurement as robustness evidence, not as training-matched
   evaluation.
6. Observation rendering and assistant action precedence are independent
   protocol axes. Canonical tool runs use the strict single-terminal-tool-call
   action boundary; historical answer-over-action/last-call behavior requires
   its explicit legacy identity.

## Result and documentation invariants

1. Register every result as `golden`, `standalone`, `confounded`, `invalid`, or
   `pending` before using it in a report table.
2. Result-registry v2 rejects every `golden` row and every numeric delta because
   it has no evaluator-owned score-provenance receipt. A future schema may
   calculate deltas only between two mechanically verified `golden` rows in one
   preregistered comparison group: hard invariants must match and the actual
   differences must equal the declared intervention axes exactly. `standalone`
   is never a causal comparator. A common task list, seed integer, or pixel cap
   is insufficient by itself.
3. Generate report tables from the result registry. Do not hand-copy a score
   into a canonical table without its artifact and contract identity.
4. Preserve failed and superseded evidence; mark it invalid/confounded rather
   than deleting or silently replacing it.

## Verification

1. Add or update tests for every changed contract. Tests must be hermetic and
   must not depend on private historical artifact directories.
2. Run the narrow affected tests first, then the repository CPU suite and Ruff
   before integration.
3. Do not claim a run, comparison, or cleanup is complete from absence of an
   obvious error. Verify the required artifacts, counts, hashes, and terminal
   receipts explicitly.
