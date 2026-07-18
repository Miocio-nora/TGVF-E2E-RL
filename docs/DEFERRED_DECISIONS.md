# Deferred Decisions After the Framework Build

Status: **intentionally unset; implementation must fail closed**
Updated: **2026-07-19 JST**

This is the short operational checklist requested for decisions that are not
needed to build and smoke-test the framework. These values must not be inferred
from veRL, DeepEyes, SDPO, a legacy run, or a synthetic test fixture.

## Data and prompts

- real representation-phase dataset manifest, license, transforms, split, and
  native-tool target-span construction;
- policy-RL prompt population, sampling rule, group construction, and held-out
  evaluation manifest;
- final system/user prompt wording and native tool-call safety cap;
- the exact JSON target-to-`Hq` rule: hidden layer, absolute token-time
  alignment, quote/escape boundary policy, and contextual versus embedding
  provider selection per experiment.

## Reward and evaluation

- task verifiers and answer normalization per benchmark;
- reward components, weights, failure values, tool cost, malformed-call cost,
  and any length/format terms;
- target-quality or anti-leakage reward, if any;
- promotion thresholds for target specificity, causal readability, policy
  quality, and high-budget reasoning retention;
- whether and when the reserved Qwen2.5-72B judge is enabled, including its
  service/model revision, prompt, sampling, calibration, and failure policy.

## GRPO production contract

- population versus sample group standard deviation and epsilon;
- reward/advantage centering and scaling;
- behavior versus proximal-old policy identity and permitted staleness;
- lower/upper clipping, KL estimator and coefficient;
- token versus sequence normalization, global denominator, microbatch and
  gradient-accumulation semantics;
- trainable policy scope, learning rate, optimizer, and reference lifecycle.

The framework contains an explicitly named synthetic oracle contract for code
tests. It is not the production contract.

## SDPO production contract

- pure SDPO versus a separately named hybrid experiment;
- feedback source, success-demonstration selection, reprompt template, and
  truncation policy;
- KL/JSD direction and alpha;
- full-vocabulary, top-k-plus-tail, or sampled-token estimator;
- importance weighting/clipping;
- current, EMA, or trust-region teacher and its coefficient/update schedule;
- production normalization, optimizer, and loss composition.

Reference-style pure SDPO code, exact multi-call observation replay, teacher
state, and checkpoint/resume are framework deliverables. The choices above are
experiment identities, not reasons to leave the implementation as a stub.

## Model-family and scale decisions

- local/runtime identity for `Qwen/Qwen2.5-VL-7B-Instruct` and its own native
  representation artifact;
- accepted Qwen2.5 substitute for Qwen3 DeepStack, or an explicit declaration
  that its support level remains main-`D` only;
- policy/rollout placement and topology beyond the two-GPU smoke;
- long-run hyperparameters, checkpoint cadence, and production resource budget.

Only physical GPU indices 2 and 3 are authorized for the current smoke. This
file does not authorize real training.
