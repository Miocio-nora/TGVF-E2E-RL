# VLMEvalKit quick start

The official VLMEvalKit checkout is deployed outside this repository at commit
`7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`. Its lightweight runtime overlay is
also external, so this setup does not change the representation/RL Python
dependency set.

Validate the checkout, `run.py` hash, Python import, CLI, Qwen3 registration,
and the seven required benchmark aliases without downloading a model or data:

```bash
.venv312/bin/python tools/validate_vlmevalkit_deployment.py
```

The deployment identity is
`configs/evaluation/vlmevalkit_deployment_v1.json`. A direct-model baseline
example is `configs/evaluation/vlmevalkit_qwen3_vstar_example.json`; it points
to the accepted local Qwen3-VL-8B-Thinking directory and uses the official
VStarBench dataset/scorer.

After recording a real GPU run in `docs/EXPERIMENT_LEDGER.md`, launch that
example with:

```bash
export PYTHONPATH=/nvmesv/dredvpn009/tools/VLMEvalKit/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f:/nvmesv/dredvpn009/tools/VLMEvalKit/runtime-7055d301/site-packages
export LMUData=/nvmesv/dredvpn009/datasets/benchmarks
export PRED_FORMAT=tsv
export EVAL_FORMAT=json

CUDA_VISIBLE_DEVICES=<gpu_ids> .venv312/bin/python \
  /nvmesv/dredvpn009/tools/VLMEvalKit/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/run.py \
  --config configs/evaluation/vlmevalkit_qwen3_vstar_example.json \
  --work-dir artifacts/evaluation/qwen3_vstar_direct \
  --mode all
```

This JSON is an original-Qwen direct baseline, not the crop/TGVF policy-agent
adapter.

## CoreDev-2511

The seven ordered slices are already materialized at
`/nvmesv/dredvpn009/datasets/benchmarks/coredev_2511_vlmevalkit_7055d301_v1`.
Their pinned identity is
`configs/evaluation/coredev_2511_vlmevalkit_v1.json`. Verify every TSV/image
hash, row order, official class, prompt construction, and inherited scorer with:

```bash
.venv312/bin/python tools/validate_coredev_2511.py
```

They use the same pinned VLMEvalKit prompt and scorer implementation as each
source benchmark. Only the evaluated row population is sliced, so their metrics
are CoreDev subset metrics, not full-dataset leaderboard scores. Historical
CoreDev numbers from the legacy custom evaluator are comparison provenance, not
scorer parity evidence.

Judge routing is fixed as follows:

- OCRBench_v2 is rule based and uses no LLM judge.
- VStarBench, HRBench4K, BLINK, and MMMU_Pro_10c use Qwen2.5-72B only when the
  official MCQ extractor cannot resolve a choice deterministically.
- MathVista_MINI and MathVerse_MINI require Qwen2.5-72B.
- GPT fallback is forbidden.

Inference may be run without the judge. Evaluation fails closed unless the
exact Qwen model and service URL are supplied:

```bash
.venv312/bin/python tools/run_coredev_2511_vlmevalkit.py \
  --config configs/evaluation/coredev_2511_qwen3_direct_v1.json \
  --work-dir artifacts/evaluation/coredev_2511_qwen3_direct \
  --mode all \
  --judge Qwen2.5-72B-Instruct \
  --judge-base-url http://127.0.0.1:8012/v1 \
  --judge-api-nproc 8
```

The runner checks the pinned judge service before and after every scoring
invocation. It also replaces VLMEvalKit's unavailable-judge exact-matching
fallback with a hard failure and rejects exhausted API calls or random-choice
fallback records. A `done` status alone is therefore not sufficient for
acceptance.

After all seven independent inference or evaluation directories are present,
validate their status files, prediction row counts/hashes, judge evidence, and
slice metrics into one project-owned summary:

```bash
.venv312/bin/python tools/summarize_coredev_2511.py \
  --work-dir artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8 \
  --phase infer \
  --output artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8/infer-summary.json

.venv312/bin/python tools/summarize_coredev_2511.py \
  --work-dir artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8 \
  --phase eval \
  --judge-base-url http://127.0.0.1:8012/v1 \
  --output artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8/eval-summary.json
```

The seven official slice metrics remain separate in the JSON; no mixed-TSV
score or unweighted synthetic benchmark score is introduced.

## Qwen2.5-72B benchmark judge

The service identity is
`configs/evaluation/qwen25_72b_judge_service_v1.json`. It uses the exact
Hugging Face revision recorded there, vLLM 0.12.0, BF16, and tensor parallel 2
on physical GPUs 2 and 3. After recording the run in the experiment ledger:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 \
PATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
TOKENIZERS_PARALLELISM=false \
.venv312/bin/python -m \
  vllm.entrypoints.openai.api_server \
  --model /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --host 127.0.0.1 --port 8012 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 --seed 42 --generation-config vllm \
  --enable-prefix-caching
```

This deployment is a benchmark evaluator only. It is not an RL reward, frozen
reference policy, or SDPO teacher. The slash-free served name is intentional:
VLMEvalKit embeds the judge string in intermediate filenames; the underlying
model identity remains `Qwen/Qwen2.5-72B-Instruct` at the pinned revision.
