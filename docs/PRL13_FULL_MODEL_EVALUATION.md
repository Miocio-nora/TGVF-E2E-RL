# PRL13 full-model evaluation snapshot

PRL13 evaluation does not use a LoRA pointer or a `LoRARequest`. Step 0 binds
the configured base Hugging Face directory. A later step binds the complete
upstream veRL checkpoint directory (`global_step_N`, including actor model,
optimizer, RNG/extra shards, and `data.pt`) and exposes standalone Hugging Face
weights to stock vLLM.

Preflight a saved step before allocating an evaluation GPU:

```bash
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python \
  tools/materialize_prl13_full_model.py preflight \
  --run-config /absolute/path/to/bound-prl13.toml \
  --optimizer-step 8 \
  --source /absolute/path/to/checkpoints/global_step_8 \
  --target-dir /absolute/path/to/evaluation/step8-hf
```

The preflight validates the exact FSDP2 rank closure, rejects adapter/LoRA
artifacts, reports source size and conservative CPU-RAM/disk estimates, and
prints the exact pinned upstream merger command. The upstream FSDP merger is a
single-process CPU operation; it does not require a GPU or `torchrun`.

Materialize and write immutable snapshot/receipt records:

```bash
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python \
  tools/materialize_prl13_full_model.py materialize \
  --run-config /absolute/path/to/bound-prl13.toml \
  --optimizer-step 8 \
  --source /absolute/path/to/checkpoints/global_step_8 \
  --target-dir /absolute/path/to/evaluation/step8-hf \
  --snapshot-manifest /absolute/path/to/evaluation/step8-snapshot.json \
  --receipt /absolute/path/to/evaluation/step8-materialization.json
```

If the veRL checkpoint already contains complete weights under
`actor/huggingface`, the command binds that tree and skips the merge. Otherwise
it executes `python -m verl.model_merger merge --backend fsdp ...` and hashes
the resulting HF tree.

“Immutable” applies to the JSON identity records, not to a second private copy
of the model payload. The receipt continues to reference its external
`model_path`; the prepare/freeze path copies only the manifest and receipt.
Consequently, a frozen full-model record is valid only while both the source
checkpoint and materialized Hugging Face tree still match every recorded file
digest.

Before every evaluator launch, revalidate both the source checkpoint and the
materialized model:

```bash
/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python \
  tools/materialize_prl13_full_model.py validate \
  --snapshot-manifest /absolute/path/to/evaluation/step8-snapshot.json \
  --receipt /absolute/path/to/evaluation/step8-materialization.json
```

The full-model vLLM builder loads the receipt's `model_path`, sets
`enable_lora=False`, uses no custom recorded-feature architecture, and passes no
`lora_request` argument to generation. Official execution never uses the
metadata-only `runtime_lightweight` loader: it streams the exact checkpoint
and model closure when loading frozen records, then repeats that content check
as the first builder action immediately before importing/constructing vLLM.
Same-size mutation of a `.bin` or `.safetensors` file therefore fails even
when names and sizes are unchanged. vLLM does not expose an engine-loaded
full-model digest, so a same-UID mutation racing the later vLLM file reads
remains a documented residual; these records must not be described as a sealed
private weight copy.
