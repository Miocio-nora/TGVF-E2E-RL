# Deferred Decisions After the Framework Build

Status: **intentionally unset; implementation must fail closed**
Updated: **2026-07-20 JST**

This is the short operational checklist requested for decisions that are not
needed to build and smoke-test the framework. These values must not be inferred
from veRL, DeepEyes, SDPO, a legacy run, or a synthetic test fixture.

## Data and prompts

- dataset/image license record for the now-materialized selected pinned v4
  clean-imend train and v3 val-2k validation manifests;
- policy-RL prompt population, sampling rule, group construction, and held-out
  evaluation manifest;
- final policy-RL system/user prompt wording and native tool-call safety cap;

The representation user-message wording is not deferred: decision
`RPI-20260720-REPRESENTATION-NATIVE-TRAJECTORY` fixes it to the original image
plus unmodified dataset question, with no separately injected target field; the
teacher target is serialized as the native assistant tool-call argument. Its
fixed pre-tool reasoning, evidence reasoning, and final `short_answer`
placement are likewise the initial accepted v1 trajectory. The fixed identities
are `qwen3-representation-image-question-v1`,
`native_representation_prompt_v1`, `representation_sample_identity_v1`,
`retained_focus_rows_v1`, and `canonical_evidence_supervision_v1`; there is no
supported earlier answer-omitting or target-bearing compatibility schema.
The provider order is also not deferred: contextual hidden state at layer `-1`
runs first and target token embedding runs second, with identical seed `42`,
data/group order, fresh initialization, objective, batch plan, and cadence.

## Reward and evaluation

- task verifiers and answer normalization per benchmark;
- reward components, weights, failure values, tool cost, malformed-call cost,
  and any length/format terms;
- target-quality or anti-leakage reward, if any;
- policy-quality and high-budget-reasoning promotion thresholds;
- representation specificity/readability reuses the historical internal
  metrics and Golden report values as its comparison baseline rather than
  inventing new values;
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

## Repository distribution

- the public repository's software license. No license has been selected or
  inferred, so `pyproject.toml` intentionally has no license metadata and the
  repository currently grants no explicit reuse license.

Only physical GPU indices 2 and 3 were authorized for the completed bounded
contextual preflight. This file does not authorize continuation to the full
2,000-step run.
