# TGVF End-to-End RL

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
4. Use a mature RL framework for distributed rollout and optimization. This
   repository owns only the narrow TGVF model/runtime adapter, trajectory
   contract, rewards, and verification needed by that framework.

The authoritative task definition is [docs/PROJECT_TASK.md](docs/PROJECT_TASK.md).
Rules for using the previous repository as a controlled reference are in
[docs/LEGACY_REFERENCE.md](docs/LEGACY_REFERENCE.md). Codex and contributor
working rules are in [AGENTS.md](AGENTS.md).

The previous `revisit_vlm` repository is a read-only reference, not a runtime
dependency of this project.
