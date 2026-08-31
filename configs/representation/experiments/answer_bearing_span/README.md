# RP70 answer-bearing-span supervision

Status: implementation and data preparation complete. The reviewed train and
validation sidecars are fully materialized and reload-verified. A formal
training TOML has not yet been published or launched.

RP70 retains the RP66 `full_d_deepstack` Adapter, the RP67 correct-image versus
donor-image matrix, the existing trajectory, prompt, sampler, target candidates,
and donor manifest. It changes only the causal labels used by L-gen and every
MatrixCE cell:

- evidence-description tokens inside explicitly annotated answer-bearing value
  spans;
- final-answer tokens;
- every other evidence, separator, suffix, and padding token is `-100`.

The pooled per-sample token mean is used for L-gen. Balanced MatrixCE uses its
negative as the cell score. The two row-wise MatrixCE reductions and the
historical norm term remain unchanged, giving weights `1 + 1 + 1 + 0.1` for
L-gen-span, target-matrix-span, image-matrix-span, and norm respectively.

## Data adjudication

The retained V4 source does not contain authoritative evidence character
spans. Its `value_span_text` was populated from the final `answer_text` during
conversion, and equals `short_answer` on all 39,998 train and 867 test rows.
Treating that field as a literal evidence span resolves only 20,144 train rows
(50.36%) and 406 test rows (46.83%). The remaining rows include capitalization,
number-word equivalence, paraphrases, derived answers, and inconsistent
evidence. Silent answer-only fallback would therefore define a mixed objective,
not the requested RP70.

RP70 instead requires one checksummed JSONL sidecar per split. Each record is
keyed by UID and binds the model-visible supervision semantics: question,
target, ordered choices, evidence, and final answer. It does not bind source
line, source-row bytes, image/image ID, visual provenance, or unrelated dataset
metadata. This lets RP67's explicitly constructed donor-image branch reuse the
same sparse labels without weakening the fields that determine prompts,
candidates, transcripts, or answer spans.

The loader accepts only explicit resolved Unicode character spans or an
explicit `verified_no_answer_bearing_evidence` adjudication. Missing, extra,
duplicate, unresolved, overlapping, out-of-bounds, or semantically drifted UIDs
fail before CUDA initialization. Dataset and sidecar row order may change; the
complete UID set and its order-independent semantic-population digest must
still match. No sample is filtered or modified.

The final reviewed artifacts are:

- train sidecar:
  `artifacts/representation_experiments/answer_bearing_span/sidecars/rp70_train_answer_bearing_span_component_reaudit_v4_reviewed_v1.jsonl`
  (`39,998` rows; SHA-256
  `6d715f4d5cf532617b06adfeec5ebb27e83bd630d570b6d4dc158fbfaf4df653`);
- validation sidecar:
  `artifacts/representation_experiments/answer_bearing_span/sidecars/rp70_validation_answer_bearing_span_component_reaudit_v4_reviewed_v1.jsonl`
  (`867` rows; SHA-256
  `e462748f621914dd88beb141188d0b7e93d7b5c1ab0df79a95380e5b62ab4660`);
- audit summary:
  `artifacts/representation_experiments/answer_bearing_span/rp70_component_reaudit_v4_reviewed_v1.audit.json`;
- source-conflict quarantine manifest:
  `artifacts/representation_experiments/answer_bearing_span/rp70_source_conflict_quarantine_v1.jsonl`.

The component re-audit covered all 638 API-repair rows. It produced 81 train
and one validation manual overrides. Fourteen ambiguous annotations were kept
unchanged instead of being over-corrected. Twelve upstream source conflicts are
recorded in the quarantine manifest but retained so RP70 remains the requested
single-variable objective comparison against RP67.

The next step is to publish a 500-step treatment TOML and outer experiment TOML
whose objective identity binds both sidecar SHA-256 values. Launch only through
`tools/run_representation_answer_bearing_span.py`; the ordinary core
representation launcher rejects RP70 identities before distributed or CUDA
startup.
