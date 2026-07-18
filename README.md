TGVF End-to-End RL

This is the new, independently versioned TGVF project.

Status: **documentation scaffold only**. No implementation, dependency stack,
training run, or evaluation run has been authorized or created yet.

The project starts from four decisions:

1. Preserve the TGVF model structure and a representation-learning phase whose
   job is to make `D` target-specific and readable.
2. Do not carry forward Stage2 supervised fine-tuning. The policy starts from
   the original Qwen reasoning model and learns the complete tool trajectory by
   reinforcement learning.
3. Do not resize the tokenizer or introduce project-specific protocol tokens.
   Use Qwen's existing native tool-call, tool-response, thinking, and vision
   tokens through the native chat template.
4. Use upstream veRL for distributed rollout and optimization, with required
   FSDP2 support and an evidence-selected concrete parallel topology. This
   repository owns only the narrow TGVF model/runtime adapter, trajectory
   contract, rewards, and verification needed by veRL.

The authoritative task definition is [docs/PROJECT_TASK.md](docs/PROJECT_TASK.md).
Open implementation contracts are tracked in
[docs/OPEN_IMPLEMENTATION_CONTRACTS.md](docs/OPEN_IMPLEMENTATION_CONTRACTS.md).
The framework-skeleton reference is
[docs/TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md](docs/TGVF_E2E_RL_CODEX_IMPLEMENTATION_SPEC.md).
Rules for using the previous repository as a controlled reference are in
[docs/LEGACY_REFERENCE.md](docs/LEGACY_REFERENCE.md). Codex and contributor
working rules are in [AGENTS.md](AGENTS.md).

The previous `revisit_vlm` repository is a read-only reference, not a runtime
dependency of this project.
