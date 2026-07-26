# Qwen3 Policy RL Data Selection

Status: **CPU candidate pools materialized and exact-leakage screened; paired
Instruct canary generated and deterministic scoring complete; semantic judge,
full T1, and T2 rule pending**

Decision: `QWEN3-RL-DATA-SELECTION-PREP-20260725` in
[`PROJECT_TASK.md`](PROJECT_TASK.md).

## Objective

Build a Qwen3-VL-8B-Instruct-native RL population with quality, learning properties, and
source mixture comparable to DeepEyes. DeepEyes is the hard methodological
reference; reproducing its exact released rows or an unpublished private
filter is not the goal.

The released 47K distribution is the comparison baseline:

| Source role | Rows | Share |
|---|---:|---:|
| V* fine-grained perception | 22,362 | 47.5% |
| ArxivQA chart/reasoning | 13,659 | 29.0% |
| ThinkLite-VL general reasoning | 11,031 | 23.4% |

The CPU reducer reports share deltas but does not impose an unaccepted
tolerance or rebalance the population.

The active selector is the official `Qwen/Qwen3-VL-8B-Instruct` revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, with 512-squared maximum image
area and its native generation prefill `<|im_start|>assistant\n`. Its answer
parser is `direct-completion-v1`: it removes terminal Qwen control tokens only
and never requires `</think>`. Thinking and Instruct attempts are distinct run
identities and are never combined.

## Hard-reference gates

`T1` uses exactly eight full-image Qwen3-VL-8B-Instruct attempts per candidate:

- `0/8`: exclude as too hard;
- `8/8`: exclude as too easy;
- `1/8` through `7/8`: retain;
- missing, duplicate, truncated, generation-failed, or verifier-failed result:
  unresolved, never silently incorrect.

`T2` adds the V*-only ground-truth-region branch. ArxivQA and ThinkLite rows
retain their `T1` status because DeepEyes applies perception utility only to
fine-grained perception data. The public paper does not fix the oracle attempt
count, region-composition rule, seed pairing, or exact gain threshold. The
implementation therefore records paired counts but refuses to decide V* `T2`
membership until those project choices are accepted.

`T3` and `T4` remain deferred; no current score is treated as `U_TGVF`.

## Candidate record

Each JSONL row uses schema `tgvf.policy-selection.candidate.v1` and contains:

```json
{
  "schema_version": "tgvf.policy-selection.candidate.v1",
  "sample_id": "stable-source-id",
  "source": "vstar",
  "question": "...",
  "ground_truth": "...",
  "image": {
    "path": "content-addressed/or/source-pinned/path.jpg",
    "sha256": "64 lowercase hex characters",
    "width": 1920,
    "height": 1080
  },
  "gt_regions": [[100, 120, 400, 520]],
  "provenance": {"dataset": "...", "revision": "...", "row_id": "..."}
}
```

Boxes are non-empty, half-open, original-image pixel boxes. `gt_regions` is
required only when V* oracle requests are generated. Ground truth is kept in a
separate verifier payload and never placed in `model_input`.

## CPU workflow

### Materialized candidate catalog

The authoritative CPU artifact is
`artifacts/data/policy_selection/catalog-v2.json`, SHA-256
`428e782a250414ad25dad2cc6333086739391d0c3a1b6d6ff898e78476da5bd9`.
It supersedes the pre-audit `catalog.json` for T1 inputs while retaining that
file as provenance.
Every row binds decoded image
bytes, original dimensions, source identity and ground truth; V* rows also bind
canonical source-pixel GT regions. The source artifacts remain immutable and a
separate screened view removes exact image-byte overlap with the current
CoreDev-2511 held-out population.

| Source | Source rows | Source rejects | Candidates | Held-out rejects | Eligible for T1 |
|---|---:|---:|---:|---:|---:|
| V* | 191,983 | 0 | 191,983 | 8 | 191,975 |
| ArxivQA | 100,000 | 107 | 99,893 | 0 | 99,893 |
| ThinkLite-VL | 69,997 | 3 | 69,994 | 152 | 69,842 |
| **Total** | **361,980** | **110** | **361,870** | **160** | **361,710** |

The 107 ArxivQA rejections are explicit invalid-option or ambiguous,
placeholder, or out-of-range label records. Before T1, the remaining rows were
rematerialized with positional `A`--`Z` labels and 18,812 explicit separator or
next-figure-heading entries removed while retaining their raw source indices.
This prevents option-format corruption from being measured as model difficulty.
The three ThinkLite rejections have
no usable ground truth. The leakage view is bound to CoreDev task SHA-256
`dc3ef2e25a20b490ecf775e42d1dc302f7b272e5346cc745a45438dd47abd6c4`
and 2,818 unique held-out image hashes. This is an exact-original-byte gate;
perceptual near-duplicate policy remains open and is not silently inferred.

As an auxiliary provenance audit only, the raw pools cover all 22,362 released
V* rows by question, all 13,659 released ArxivQA rows by base-question prefix,
and all 11,031 released ThinkLite rows by exact image-plus-question pair. This
does not claim identical images, option order, or selected membership and does
not substitute for Qwen3-native filtering.

### Request and reduction commands

Generate eight full-image requests and, when requested, V* oracle requests:

```bash
python tools/prepare_policy_data_selection.py build-requests \
  --candidates candidates.jsonl \
  --oracle-attempts 8 \
  --output requests.jsonl
```

GPU execution later produces attempt rows with schema
`tgvf.policy-selection.attempt.v1`. A scored row has Boolean `correct`; an
unscored row uses status `truncated`, `generation_error`, or `verifier_error`
and has no correctness value.

Reduce the attempts without model or Torch imports:

```bash
python tools/prepare_policy_data_selection.py reduce \
  --candidates candidates.jsonl \
  --attempts attempts.jsonl \
  --expected-oracle-attempts 8 \
  --output decisions.jsonl \
  --summary-output summary.json
```

Outputs are canonical, content-hashable, auditable, and created with exclusive
file semantics. Existing outputs are never overwritten.

## Instruct canary result and remaining gates

The completed T1-02 canary contains all 1,536 unique generations over 192
candidates, with zero sampled `<think>` tags and no generation error or length
finish. CPU rescoring under `direct-completion-v1` produced deterministic
manifest
`49828a06238cff17c37c6e3297ede06df1fc1fdebb6ae162bace300234ab0eb9`:
559 attempts were resolved deterministically, 969 attempt consumers require
936 content-deduplicated semantic-judge requests, and ArxivQA judge calls are
zero. Its provisional decisions are 51 retained, 2 too easy, 11 too hard, and
128 unresolved. These are a parser/correctness-canary result, not a final
source distribution; unresolved V* and ThinkLite candidates require the local
judge.

Still required before full T1 generation:

- a completed Instruct semantic-judge continuation and correctness audit;
- a fresh full-run identity, candidate count/mixture, and `PLANNED` ledger cell;
- accepted V* oracle composition and `T2` membership rule;
- accepted distribution tolerance or balancing rule.

Reverse matching against the released DeepEyes 47K is an optional coverage and
provenance audit, not a prerequisite for selecting a high-quality population.

## Source readiness

- V*-style: all 191,983 rows from the four GT-region-bearing annotation files
  resolve to decoded COCO/GQA image bytes and valid half-open GT boxes. The
  image materialization contains 107,798 source paths and the candidate view
  contains 100,916 unique image hashes. Of 282,650 boxes, 1,359 required
  boundary clipping and none became empty. The inherited image terms are
  recorded in `EXTERNAL_REFERENCES.md`; redistribution/production legal review
  remains separate. The 191-row V*Bench is evaluation data, not training data.
- ArxivQA: 99,893 verified candidates are materialized from the pinned JSONL
  and image archive under `arxivqa-canonical-options-v2`; ambiguous source
  labels are rejected rather than guessed. The screened candidate JSONL
  SHA-256 is
  `cda47ff2d9218f63e871598c898f18862c994ee0affb76294d0fe2a1dd991742`.
- ThinkLite-VL: 69,994 verified raw candidates are materialized with 65,027
  content-addressed images. The hard-11K set remains a quality/distribution
  reference and does not replace Qwen3 difficulty selection.

Exact revisions and license status are recorded in
[`EXTERNAL_REFERENCES.md`](EXTERNAL_REFERENCES.md).
