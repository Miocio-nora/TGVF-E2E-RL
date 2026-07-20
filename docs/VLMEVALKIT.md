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
adapter. CoreDev-2511 still requires seven separately materialized and hashed
TSV slices before it can be run or scored.
