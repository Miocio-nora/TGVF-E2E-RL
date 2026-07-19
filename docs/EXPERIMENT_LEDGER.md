# Experiment Ledger

I8H-20260719 authorizes the bounded compatibility cells below. It does not
authorize production training. Only physical GPUs 2 and 3 may be exposed.

Experiment namespaces are disjoint: `SC-*` is reserved for cells fixed by the
veRL compatibility matrix, while `RP-*` identifies bounded representation-
phase executions. A materialized run ID is never renamed after execution; an
identity collision is retained as `INVALID` and rerun under a new planned ID.

## Bounded cells and outcomes

### SC-20-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20`; mandatory Qwen3 vLLM
  latent execution smoke, narrower than the full L1..L3 promotion claim.
- Spike-plan git revision and VA0/VA1/VA2 approval references:
  `2ffa28e`; I8H-20260719 in `PROJECT_TASK.md` §0 and approved spike-plan §0.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: can real Qwen3-VL-8B-Thinking on vLLM 0.12 load the repo-owned
  public model/processor plugin and execute one native transcript containing a
  source item plus two precomputed main-D/three-branch DeepStack items while
  returning actual processed sampled-token log probabilities?
- Baseline and exact output path: no competing backend; result
  `artifacts/compatibility/SC-20-qwen3-vllm-latent-db3315a.json`, log
  `artifacts/compatibility/SC-20-qwen3-vllm-latent-db3315a.log`.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`; model type `qwen3_vl`,
  original architecture `Qwen3VLForConditionalGeneration`, plugin architecture
  `TGVFQwen3VLForConditionalGeneration`; tokenizer length `151669`; no weight-
  directory hash by user decision.
- Representation checkpoint identity: N/A—deterministic synthetic latent seed
  `20260719`; this cell does not claim a trained/native TGVF Adapter artifact.
- N/A fields and justification: reference policy, optimizer, gradient,
  checkpoint/resume, reward, scorer and dataset are N/A because this is one
  inference-only transport smoke.
- Policy/reference initialization: original local Qwen3 reasoning weights;
  reference not instantiated.
- Rollout policy version and allowed asynchronous staleness: immutable base
  model path for this no-update process; staleness `0`.
- Code commit and worktree state: runtime code commit `db3315a`; the launch
  worktree must be clean, with only this subsequently committed ledger entry
  differing from the runtime-code tree.
- Repository adapter/patch surface and hash:
  `packer.py@15aab9a71f02086f164b144cb28804ff31d658d4efb599766269fdffc5b05487`,
  `qwen3_plugin.py@1e4756d7c2dfd57edbb46bb8bc69cf52b9b4c924bf66218723d61a90e98202df`,
  `registration.py@fa2c3a48dc316833552a132047bd1993d680bc4d430b6477079cc34ab50d35dc`,
  smoke script
  `a97ed36b90d9da9eab92853ab38628a4d789dd045c5b7b8d38afe42ccfca693c`;
  no site-package patch.
- Dataset/manifest, hashes, sample rule, and n: synthetic one-request fixture,
  `n=1`, no dataset.
- Native prompt/tool schema hash: transcript text
  `c3f7e889eca24efa97ac605cb875ca88858b4c6b827bec4c59c411cbde7e091b`;
  tool schema
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: chat template
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  295 prompt-token IDs
  `d0dc12f15978045403a5a664e4e3eb5ce9f711a1638efad3a8886922f4525294`;
  assistant generation prefill asserted; this cell does not apply an RL loss
  mask.
- D/DeepStack/position/mask identity: three items (source, call 0, call 1), each
  grid `(1,2,2)`, one merged row, width `4096*(1+3)=16384`, branch layers
  `(8,16,24)`, BF16 latent SHA256
  `3f7711522f4bd3c10530b765a186b109003727934459ed388f35e60863d26cda`;
  vLLM processor owns native M-RoPE placeholder positions.
- Observation materialization/artifact identity used by all replays: this GPU
  cell has no policy/reference replay. CPU tests at `db3315a` separately prove
  `ObservationStore -> pack_qwen3_vllm_replay` content checks; the limitation is
  retained in the compatibility conclusion.
- RL framework/version/environment lock: veRL
  `e003163181731412595257a72ec173071efb125f` / `0.9.0.dev0`, vLLM `0.12.0`,
  Torch `2.9.0+cu128`, Transformers `4.57.6`, Python `3.12.3`; lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: N/A; no objective or update.
- Rollout/replay forward mode and adapter dropout/RNG contract: vLLM eager
  forward, prefix cache off, multimodal processor cache `0`, main+DeepStack
  precomputed embeddings, no TGVF recomputation; policy-adapter dropout N/A.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.12.0`, seed `20260719`,
  temperature `.7`, top-p `.9`, top-k `20`, min-p `.01`, repetition `1.05`,
  presence `.1`, frequency `.05`, no custom processor, two tokens,
  `processed_logprobs` after all represented transforms.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights/KV, no quantization,
  vLLM-selected attention, TP=2, no training mesh, max model length 512, eager.
- Logit/logprob/loss/gradient parity tolerances: structural engine PASS plus
  finite non-positive sampled processed logprobs; CPU vLLM 0.12 transform oracle
  requires exact elementwise equality. No loss/gradient in this cell.
- World size, microbatch, accumulation, and global batch: TP world `2`, one
  request, one completion, no accumulation.
- GPUs: physical `2` =
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`; physical `3` =
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; both NVIDIA B200 183359 MiB,
  compute capability 10.0, driver 570.195.03; logical mapping `2->0`, `3->1`.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:02:41+09:00` / `2026-07-19T07:03:32+09:00`, 51 seconds;
  engine PID 1360895, worker PIDs 1361456 and 1361457.
- Actual GPU-hours and peak scratch use: less than `0.029` two-device GPU-hours
  by wall-time upper bound; model weights were not loaded and observed memory
  remained 0 MiB at the pre-load sample; log 16,495 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-qwen3-vllm-latent-db3315a.json > artifacts/compatibility/SC-20-qwen3-vllm-latent-db3315a.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: smoke script SHA256 above; native renderer and plugin
  are code-commit identified; no task scorer.
- Metrics: public plugin resolved the requested architecture and initialized
  TP=2/NCCL, then both workers stopped before weight loading. Log SHA256:
  `c9f168332ba2f89a6bbde3e7709254b1e80fbd180e3aa882a19da54797282e9f`.
- Conclusion: `FAIL` as required by the hard gate. The subclass variadic
  constructor hid vLLM's new-style keyword-only `(vllm_config, prefix)`
  signature, so vLLM classified it as old-style and raised missing
  `vllm_config`. No latent forward or sampled-token result was produced.

### SC-20-R1-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R1`; mandatory bounded
  rerun after the isolated public-constructor fix from failed `SC-20`.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope corrective rerun.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: same as SC-20, with the added gate that vLLM must recognize the
  plugin as a new-style model and pass `vllm_config` and `prefix` by keyword.
- Baseline and exact output path: failed SC-20; result
  `artifacts/compatibility/SC-20-R1-qwen3-vllm-latent-22afd52.json`, log
  `artifacts/compatibility/SC-20-R1-qwen3-vllm-latent-22afd52.log`.
- Model and processor identity: identical to SC-20.
- Representation checkpoint identity: identical deterministic synthetic latent
  fixture to SC-20; trained representation checkpoint N/A.
- N/A fields and justification: identical to SC-20.
- Policy/reference initialization: identical to SC-20.
- Rollout policy version and allowed asynchronous staleness: identical to
  SC-20; staleness `0`.
- Code commit and worktree state: runtime code commit
  `22afd522bf444941e3d69741bc648f85d9be0afe`; launch worktree must be clean
  except for this subsequently committed ledger result/plan.
- Repository adapter/patch surface and hash: SC-20 surfaces unchanged except
  `qwen3_plugin.py@4480e9d6e28542ad3b5f3f99597349c3e6d6626c50652e2658cd4a71021f7bce`;
  smoke script remains
  `a97ed36b90d9da9eab92853ab38628a4d789dd045c5b7b8d38afe42ccfca693c`;
  no site-package patch.
- Dataset/manifest, hashes, sample rule, and n: identical to SC-20, `n=1`.
- Native prompt/tool schema hash: identical to SC-20.
- Chat-template/token-fixture hash and token-ownership masks: identical to
  SC-20.
- D/DeepStack/position/mask identity: identical to SC-20.
- Observation materialization/artifact identity used by all replays: identical
  limitation and CPU evidence to SC-20.
- RL framework/version/environment lock: identical to SC-20.
- Objective equations and normalization: N/A; no update.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical to
  SC-20.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: identical to SC-20.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: identical to SC-20.
- Logit/logprob/loss/gradient parity tolerances: identical to SC-20, plus the
  plugin constructor-signature CPU regression test must pass.
- World size, microbatch, accumulation, and global batch: identical to SC-20.
- GPUs: identical physical B200 UUIDs and mapping to SC-20.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:08:23+09:00` / `2026-07-19T07:09:49+09:00`, 86 seconds;
  engine PID 1373046, worker PIDs 1373596 and 1373597.
- Actual GPU-hours and peak scratch use: less than `0.048` two-device GPU-hours
  by wall-time upper bound; observed memory reached 4822 MiB/device before the
  load sample, and the worker reported 8.447 GiB for loaded model state; log
  45,627 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R1-qwen3-vllm-latent-22afd52.json > artifacts/compatibility/SC-20-R1-qwen3-vllm-latent-22afd52.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: same script/native renderer as SC-20, code commit and
  corrected plugin hash above.
- Metrics: the corrected public plugin resolved as a new-style model, TP=2/NCCL
  initialized, and Qwen weights loaded in 14.79 seconds. The run then stopped
  during vLLM's Triton rotary profiling. Log SHA256:
  `3825882000a40bc155d1619fac90fbdecbb2e952098a09a87a1b4e936a641d34`.
- Conclusion: `FAIL` as required by the hard gate. The launcher inherited
  `CC`/`CXX` from the legacy `revisit-vlm` Conda environment, and the host
  Python 3.12 installation lacks `/usr/include/python3.12/Python.h`; Triton's
  runtime CUDA helper compilation therefore failed before latent execution.
  This cell proves the constructor correction but produces no sampled-token
  result. No site package or source file was patched.

### SC-20-R2-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R2`; mandatory bounded
  rerun after isolating the host compiler/header failure in `SC-20-R1`.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope environment correction.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: same as SC-20-R1, with the additional gate that vLLM/Triton uses
  the system compiler and an exact locally extracted Ubuntu Python 3.12 header
  package rather than any legacy environment compiler.
- Baseline and exact output path: failed SC-20-R1; result
  `artifacts/compatibility/SC-20-R2-qwen3-vllm-latent-22afd52.json`, log
  `artifacts/compatibility/SC-20-R2-qwen3-vllm-latent-22afd52.log`.
- Model, processor, representation fixture, initialization, staleness,
  dataset, transcript, token fixture, D/DeepStack identity, replay limitation,
  RL lock, forward mode, sampling, dtypes, parallelism, tolerances, batch and
  GPU identities: identical to SC-20-R1.
- N/A fields and justification: identical to SC-20-R1.
- Code commit and worktree state: runtime code commit
  `22afd522bf444941e3d69741bc648f85d9be0afe`; launch worktree may differ only
  by the subsequently committed experiment-ledger record.
- Repository adapter/patch surface and hashes: identical to SC-20-R1; no source
  or site-package patch. Runtime-only Ubuntu packages are
  `libpython3.12-dev_3.12.3-1ubuntu0.15_amd64.deb` SHA256
  `ab00830dd4344f910acf410c671377841906f2cee8ebb4bf91044d64c77a50d0`
  and `python3.12-dev_3.12.3-1ubuntu0.15_amd64.deb` SHA256
  `0301b3a8dc5bc6706c0bf97a90496486f77279e5a3f6d5023b866bf63fef4e83`,
  extracted under ignored `.deps/python312-dev/root` without system install.
- Environment correction: `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, and `CPATH`
  points exactly to the extracted Python 3.12 include directory. No legacy
  repository code or Python package is imported.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:11:55+09:00` / `2026-07-19T07:13:11+09:00`, 76 seconds;
  engine PID 1380048, worker PIDs 1380538 and 1380539.
- Actual GPU-hours and peak scratch use: less than `0.043` two-device GPU-hours
  by wall-time upper bound; observed memory reached 4822 MiB/device before the
  load sample, and the worker reported 8.447 GiB for loaded model state; log
  74,762 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R2-qwen3-vllm-latent-22afd52.json > artifacts/compatibility/SC-20-R2-qwen3-vllm-latent-22afd52.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: same script/native renderer and corrected plugin as
  SC-20-R1.
- Metrics: `/usr/bin/gcc` was used and found the extracted `Python.h`; TP=2
  initialized and Qwen weights loaded in 9.01 seconds. Compilation stopped at
  Python's architecture-specific `pyconfig.h`. Log SHA256:
  `6270af290940cf9aa96adc69283142fe0d51a620959bb6951b97fb910201a955`.
- Conclusion: `FAIL` as required by the hard gate. The extracted `Python.h`
  includes `x86_64-linux-gnu/python3.12/pyconfig.h` relative to the parent
  include root, but R2's `CPATH` named only the Python subdirectory. No latent
  execution or sampled-token result was produced; no code/package was patched.

### SC-20-R3-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R3`; mandatory bounded
  rerun after correcting R2's incomplete extracted-header search path.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope environment correction.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question, baseline, model, processor, representation fixture,
  initialization, staleness, N/A fields, dataset, transcript, token fixture,
  D/DeepStack identity, replay limitation, RL lock, forward mode, sampling,
  dtypes, parallelism, tolerances, batch and GPU identities: identical to
  SC-20-R2, with failed R2 as the baseline.
- Exact output path: result
  `artifacts/compatibility/SC-20-R3-qwen3-vllm-latent-22afd52.json`, log
  `artifacts/compatibility/SC-20-R3-qwen3-vllm-latent-22afd52.log`.
- Code commit, repository surfaces and package identities: identical to
  SC-20-R2; no source or site-package patch.
- Environment correction: system `CC`/`CXX`; `CPATH` contains both the
  extracted `/usr/include` root and `/usr/include/python3.12` subdirectory.
  A CPU preprocessor preflight for `#include <Python.h>` passed; extracted
  architecture `pyconfig.h` SHA256 is
  `23931f53bc7ee512c6bd3162828747494eaa5e21f23dd63b31170ec1adfbe65e`.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:14:37+09:00` / `2026-07-19T07:15:55+09:00`, 78 seconds;
  engine PID 1385188, worker PIDs 1385479 and 1385480.
- Actual GPU-hours and peak scratch use: less than `0.044` two-device GPU-hours
  by wall-time upper bound; observed memory reached 4822 MiB/device before the
  load sample, and the worker reported 8.447 GiB for loaded model state; log
  58,401 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R3-qwen3-vllm-latent-22afd52.json > artifacts/compatibility/SC-20-R3-qwen3-vllm-latent-22afd52.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: same script/native renderer and corrected plugin as
  SC-20-R2.
- Metrics: the complete extracted-header path passed Triton runtime compilation;
  TP=2 initialized and Qwen weights loaded in 15.25 seconds. Profiling then
  entered the wheel-bundled visual FlashAttention kernel and raised
  `cudaErrorUnsupportedPtxVersion`. Log SHA256:
  `fbe98a6ac50731684dc256d810eb52beff7e35fd3225aa642a21ca1398099c03`.
- Conclusion: `FAIL` as required by the hard gate. vLLM 0.12's bundled visual
  FlashAttention PTX is not accepted by the host's NVIDIA 570.195.03 driver
  (reported CUDA capability 12.8). The Qwen3 vLLM implementation publicly
  supports a `TORCH_SDPA` multimodal-encoder override, so the next cell tests
  that explicit driver-portable path; no site package is patched.

### SC-20-R4-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R4`; mandatory bounded
  rerun using vLLM's public supported `TORCH_SDPA` visual-encoder path after
  the wheel/driver PTX mismatch in R3.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope public configuration correction.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question, model, processor, representation fixture, initialization,
  staleness, N/A fields, dataset, transcript, token fixture, D/DeepStack
  identity, replay limitation, RL lock, sampling, dtypes, parallelism,
  tolerances, batch and GPU identities: identical to SC-20-R3, with failed R3
  as the baseline.
- Exact output path: result
  `artifacts/compatibility/SC-20-R4-qwen3-vllm-latent-d39c8bb.json`, log
  `artifacts/compatibility/SC-20-R4-qwen3-vllm-latent-d39c8bb.log`.
- Code commit and worktree state: runtime code commit
  `d39c8bba780d259babb8b6c7c88ea9ba82fc58cc`; launch worktree may differ only
  by the subsequently committed experiment-ledger record.
- Repository adapter/patch surfaces and hashes:
  `qwen3_plugin.py@4480e9d6e28542ad3b5f3f99597349c3e6d6626c50652e2658cd4a71021f7bce`,
  `adapter.py@62c9c4e1561092f0d07cbd462aabf8dcabcda63a1835e5a50034540a09221e69`,
  `compatibility.py@335b6c36f2aae95c426c4626ba3fbdac5c419aee6c4beec8111a7dbbfadc3f5f`,
  `registration.py@f14315fa972324e3d4d7dc6554bbeba1adeebe5930ada5f3ff112dcb3a37a911`,
  smoke script
  `23b14c658ffc8ac9376d86d69ac6e7a62f8d53f848036b036156629027f30b22`;
  no site-package patch.
- Forward/attention/environment correction: vLLM eager main forward, cache
  settings identical to R3; `mm_encoder_attn_backend=TORCH_SDPA` is enforced
  by both the smoke and project veRL configuration validator. System compiler
  and complete extracted-header paths remain identical to R3.
- CPU gate before launch: complete suite `111 passed`; adapter tests prove the
  public override is emitted and fail closed on `FLASH_ATTN`.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:20:19+09:00` / `2026-07-19T07:21:35+09:00`, 76 seconds;
  engine PID 1397079, worker PIDs 1397504 and 1397505.
- Actual GPU-hours and peak scratch use: less than `0.043` two-device GPU-hours
  by wall-time upper bound; observed memory reached 4822 MiB/device before the
  load sample, and the worker reported 8.447 GiB for loaded model state; log
  139,245 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R4-qwen3-vllm-latent-d39c8bb.json > artifacts/compatibility/SC-20-R4-qwen3-vllm-latent-d39c8bb.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: script/native renderer and project validators at the
  hashes above.
- Metrics: visual `TORCH_SDPA` completed profiling, Qwen weights loaded in 9.45
  seconds, and the engine reported 114.11 GiB available KV memory. The language
  backend auto-selected FlashInfer, whose warmup JIT inherited the legacy
  Conda `nvcc` and failed on absent `cublasLt.h`/`nvrtc.h`. Log SHA256:
  `e55ad2b872f1bd7a12fef65c1a6222a6d245926262350f4d4ca5ec72d5b93eac`.
- Conclusion: `FAIL` as required by the hard gate. R4 proves the visual SDPA
  correction and reaches language-model KV initialization, but the implicit
  FlashInfer backend introduces an undeclared CUDA-toolkit JIT dependency. The
  next cell pins vLLM's public `TRITON_ATTN` backend; no site package is patched.

### SC-20-R5-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R5`; mandatory bounded
  rerun with both public driver-portable attention selections after R4.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope public configuration correction.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question, model, processor, representation fixture, initialization,
  staleness, N/A fields, dataset, transcript, token fixture, D/DeepStack
  identity, replay limitation, RL lock, sampling, dtypes, parallelism,
  tolerances, batch and GPU identities: identical to SC-20-R4, with failed R4
  as the baseline.
- Exact output path: result
  `artifacts/compatibility/SC-20-R5-qwen3-vllm-latent-515256c.json`, log
  `artifacts/compatibility/SC-20-R5-qwen3-vllm-latent-515256c.log`.
- Code commit and worktree state: runtime code commit
  `515256ca5cf7d931bbdcf3709af02eb24e59c867`; launch worktree may differ only
  by the subsequently committed experiment-ledger record.
- Repository adapter/patch surfaces and hashes: R4 surfaces unchanged except
  `adapter.py@7f039fa5f0e22daa325889a07c0b4e4c81e30faea8b0da9f7f7dec0f1475a206`,
  `registration.py@bc3c29f10962f45c7f99575622c54b1db127ec39de9e7921fe8da9053e1f5e77`,
  and smoke script
  `a2042c086b0e73894f9755504d78c46a3cc5d611e50217a18348041e64d85fe7`;
  no site-package patch.
- Forward/attention/environment correction: R4's `TORCH_SDPA` multimodal
  encoder remains; `VLLM_ATTENTION_BACKEND=TRITON_ATTN` selects the public
  Triton language attention backend. The project adapter emits and validates
  both required vLLM environment values before worker spawn. Compiler/header
  paths are identical to R4.
- CPU gate before launch: complete suite `111 passed`; runtime-environment tests
  prove `TRITON_ATTN` is required and reject `FLASHINFER`.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:23:54+09:00` / `2026-07-19T07:25:15+09:00`, 81 seconds;
  engine PID 1405122, worker PIDs 1405752 and 1405753.
- Actual GPU-hours and peak scratch use: less than `0.045` two-device GPU-hours
  by wall-time upper bound; observed memory reached 10,986 MiB/device, worker
  model-state report 8.447 GiB/device; log 12,321 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R5-qwen3-vllm-latent-515256c.json > artifacts/compatibility/SC-20-R5-qwen3-vllm-latent-515256c.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: script/native renderer and project validators at the
  hashes above.
- Metrics: both workers selected `TRITON_ATTN`; Qwen weights loaded in 3.01
  seconds; visual profiling, 114.11 GiB KV sizing and full engine warmup all
  completed (15.29 seconds). The first real request then failed in vLLM's
  cache-disabled UUID builder/hash path. Log SHA256:
  `38c3a91e6417e9968eba2fbe00d39b31f31e8f1b884ec029ad287233e9ac725b`.
- Conclusion: `FAIL` as required by the hard gate. The project passed three
  latent items in one field dictionary; vLLM's cache-disabled UUID builder
  counts that public value as one item, while the custom parser correctly
  expanded it to three, causing an out-of-range UUID lookup. R6 transports a
  public list of three per-item dictionaries and coalesces only inside the
  repo-owned parser. No site package is patched.

### SC-20-R6-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class: `SC-20-R6`; mandatory bounded
  rerun after aligning public latent item and UUID cardinalities in R5.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical to
  `SC-20`; I8H-20260719 covers this in-scope repo-owned interface correction.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question, model, processor, representation fixture, initialization,
  staleness, N/A fields, dataset, transcript, token fixture, D/DeepStack
  identity, replay limitation, RL lock, sampling, dtypes, attention,
  parallelism, tolerances, batch, GPU and environment identities: identical
  to SC-20-R5, with failed R5 as the baseline.
- Exact output path: result
  `artifacts/compatibility/SC-20-R6-qwen3-vllm-latent-f796c29.json`, log
  `artifacts/compatibility/SC-20-R6-qwen3-vllm-latent-f796c29.log`.
- Code commit and worktree state: runtime code commit
  `f796c29544410ac3514f218d81899554e78c4141`; launch worktree may differ only
  by the subsequently committed experiment-ledger record.
- Repository adapter/patch surfaces and hashes: R5 configuration surfaces
  unchanged; `packer.py@f1da42f47ee54680edb76243feacc4f7896f85a7ce563b5d77576f9c755cd908`,
  `qwen3_plugin.py@aecf42c0e21b4a9442f7b939fb3d7337895cad27d2111d26ec3c6b2fef1112dc`,
  smoke script
  `d841b5edd15db66742ccb96958e3e966d883cb6d6226120e5fc363ce401b6ba2`;
  no site-package patch.
- Public input correction: `multi_modal_data.image` is a list of exactly three
  dictionaries (source, call 0 D, call 1 D), each carrying its own merged
  main+DeepStack tensor and one-row grid. The custom public processor validates
  every item and coalesces fields only after UUID creation. Content, order and
  aggregate latent SHA are unchanged from SC-20.
- CPU gate before launch: complete suite `111 passed`; live vLLM parser test
  proves public list cardinality 3, parser cardinality 3, per-item shapes, and
  post-pack mutation rejection.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:29:40+09:00` / `2026-07-19T07:31:09+09:00`, 89 seconds;
  engine PID 1417608, worker PIDs 1418069 and 1418070.
- Actual GPU-hours and peak scratch use: less than `0.050` two-device GPU-hours
  by wall-time upper bound; observed peak memory 128,662 MiB/device including
  the deliberately large 114.11 GiB KV allocation; result 1,619 bytes, log
  11,215 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv312/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-R6-qwen3-vllm-latent-f796c29.json > artifacts/compatibility/SC-20-R6-qwen3-vllm-latent-f796c29.log 2>&1`.
- Outputs: result JSON and log paths above; no checkpoint.
- Scorer/parser identity: script/native renderer, packer and project processor
  at the hashes above.
- Metrics: `PASS`. Both workers selected `TRITON_ATTN`; weights loaded in 2.81
  seconds and engine initialization took 17.58 seconds. The exact 295-token
  native prompt contained two tool calls/responses; tokenizer length remained
  `151669`; three one-row, width-16384 latent items retained aggregate SHA256
  `3f7711522f4bd3c10530b765a186b109003727934459ed388f35e60863d26cda`.
  Sampled token IDs were `(58533,279)` and vLLM returned finite non-positive
  processed logprobs `(0.0,0.0)`. Result SHA256:
  `473a9dcb438a51336bdfa0ee82d3a46b49262869b36868a1c02064b2e55bc4c7`;
  log SHA256:
  `159b69c186271b002ee63d450e5792c3c4b71223a6847d1b5418047c1308bc87`.
- Conclusion: `PASS` for this bounded real-model transport smoke. It proves the
  public repo-owned vLLM plugin can execute the recorded-latent shape and native
  two-call transcript on Qwen3-VL-8B-Thinking TP=2 without tokenizer growth or
  site-package patches. It is not policy/reference replay parity, a trained
  TGVF Adapter result, or production reward/objective evidence.

### SC-30-FSDP2-INFRA

- Cell/matrix ID and mandatory/diagnostic class: `SC-30`; mandatory executable
  two-rank FSDP2 infrastructure/checkpoint smoke, not a Qwen L4 claim.
- Spike-plan git revision and VA0/VA1/VA2 approval references:
  `2ffa28e`; I8H-20260719 and approved spike-plan §0.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: does the accepted Python/Torch/veRL environment execute composable
  FSDP2 on two ranks and reproduce the next forward, scalar loss, optimizer
  update and local parameter shards exactly after strict distributed
  checkpoint teardown/reconstruction/resume?
- Baseline and exact output path: in-process uninterrupted step-2 control;
  result `artifacts/compatibility/SC-30-fsdp2-infra-db3315a.json`, checkpoint
  `artifacts/compatibility/SC-30-fsdp2-infra-db3315a-checkpoint`, log
  `artifacts/compatibility/SC-30-fsdp2-infra-db3315a.log`.
- Model and processor identity: `synthetic-tiny-fsdp2-model-v1`, width 16, two
  residual MLP blocks, FP32, seed `20260719`; processor N/A.
- Representation checkpoint identity: N/A—no TGVF or Qwen model in this
  infrastructure-only cell.
- N/A fields and justification: data, prompt, reward, behavior policy,
  reference policy, D/DeepStack and sampling are N/A because no RL rollout or
  objective is executed.
- Policy/reference initialization: N/A; tiny deterministic model is initialized
  identically for control/resume.
- Rollout policy version and allowed asynchronous staleness: N/A; no rollout or
  intervening update.
- Code commit and worktree state: runtime code commit `db3315a`; clean launch
  worktree, with only the committed ledger entry outside that runtime-code tree.
- Repository adapter/patch surface and hash: script
  `22380ea2f0ff0d44e847d5cc1065db30767cd9ca7ff73363c52aeac84d02ff8d`,
  config `f2a1f3f1abf3ed96a50ebdf6d4c6ccf0920d561457d7ec41d9b7c866f251660a`;
  public veRL `FSDPEngineConfig` and `CheckpointHandler` imports, public PyTorch
  `fully_shard` and distributed-checkpoint APIs; no patch.
- Dataset/manifest, hashes, sample rule, and n: deterministic random tensors for
  two steps, fixture seeds `20260720` and `20260721`; no dataset.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: same exact stack and lock SHA256 as
  SC-20.
- Objective equations and normalization: test-only identity
  `synthetic-fsdp2-mse-v1`; `mean((model(x)-target)**2)` over every output
  element; AdamW lr `.001`, weight decay `0`, no production RL meaning.
- Rollout/replay forward mode and adapter dropout/RNG contract: train forward;
  `torch.use_deterministic_algorithms(True)`, TF32 off, no dropout, fixed CPU
  batch generation, CUBLAS workspace `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: FP32, no KV/quantization/attention/TP;
  one-dimensional CUDA FSDP mesh of logical ranks `(0,1)`, world size 2.
- Logit/logprob/loss/gradient parity tolerances: resumed step-2 output, loss and
  local updated parameter shards must be bitwise equal (`atol=rtol=0`).
- World size, microbatch, accumulation, and global batch: world `2`; each rank
  sees shape `[2,3,16]`; one backward/update per step; accumulation `1`.
- GPUs: same physical B200 identities and logical mapping as SC-20.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T07:32:31+09:00` / `2026-07-19T07:32:57+09:00`, 26 seconds;
  torchrun launcher PID 1422747, two ranks identified in the result as 0 and 1.
- Actual GPU-hours and peak scratch use: less than `0.015` two-device GPU-hours
  by wall-time upper bound; the bounded run completed before a live peak-memory
  sample, returned both devices to 0 MiB, and wrote a 227,280-byte checkpoint,
  1,562-byte result and 4,027-byte log.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 NCCL_DEBUG=WARN timeout 600s .venv312/bin/torchrun --standalone --nproc-per-node=2 spikes/verl_compat/fsdp2_smoke.py --config configs/smoke/fsdp2.toml --checkpoint-dir artifacts/compatibility/SC-30-fsdp2-infra-db3315a-checkpoint --output artifacts/compatibility/SC-30-fsdp2-infra-db3315a.json > artifacts/compatibility/SC-30-fsdp2-infra-db3315a.log 2>&1`.
- Outputs: strict distributed checkpoint, result JSON and log above.
- Scorer/parser identity: smoke script/config SHA256 above.
- Metrics: `PASS`; both ranks reported control loss `1.5465291738510132` and
  `resume_exact=true`. Rank-local updated model-shard SHA256 values were
  `228dac25766a600def89284d138d47bfc9b3879862717aee00962cb3fcab3e4f`
  and `20ea8b816778e10b2e0e6a9b1bfd05e8c43a4962d98c2e7d45ba21569b2c30ed`.
  The installed veRL distribution was clean at the pinned commit and exposed
  `FSDPEngineConfig`/`CheckpointHandler`. Result SHA256:
  `b131beb78ce03ba68fb91ba37369959182527c1aa43b237d407520e068fc89b1`;
  log SHA256:
  `9f4dd8e801ee7fadde5e599f6a12c0caed673c6a2f59b2f5912727555aa7b38d`.
- Conclusion: `PASS` for the two-rank infrastructure/checkpoint question. The
  uninterrupted and teardown/reconstructed paths produced bitwise-identical
  step-2 output, scalar loss and post-update local shards with strict model,
  optimizer and extra-state restoration. This is not a Qwen FSDP2 memory or
  production-throughput claim.

### INVALID-RUN-ID-COLLISION-20260719-180827

- Cell/matrix ID and mandatory/diagnostic class: materialized run ID
  `SC-40-QWEN3-REPRESENTATION-FSDP2-EMBEDDING`; invalid diagnostic side result.
  The `SC-40` short namespace was already reserved for the compatibility
  matrix's SDPO CPU parity cell, so this run cannot receive representation-gate
  credit despite its technically successful execution.
- Spike-plan git revision and VA0/VA1/VA2 approval references: implementation
  baseline `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`; user-accepted
  `I8H-20260719` in `PROJECT_TASK.md` §0 and `AD-05G`/`AD-06`/`AD-07` in
  `OPEN_IMPLEMENTATION_CONTRACTS.md`.
- Lifecycle status: `COMPLETE`.
- Result: `INVALID`.
- Question: can the stable local Qwen3-VL-8B-Thinking model execute one
  deterministic native-format same-image Matrix-CE plus `L_gen` optimizer step
  with target-token-embedding conditioning, main `D` plus all three DeepStack
  branches, frozen Qwen, two-rank composable FSDP2 over every and only
  TGVF-Adapter-owned parameter, validation, synchronous distributed checkpoint,
  and rank-zero Adapter export without tokenizer growth?
- Baseline and exact output path: the `295 passed` CPU suite and `SC-30` generic
  FSDP2 resume result are pre-launch baselines. This cell writes under
  `artifacts/representation/SC-40-qwen3-representation-fsdp2-embedding/`:
  `adapter.pt`, `metrics.jsonl`, `checkpoints/`, and
  `run.log`. No pre-existing output is accepted and overwrite is disabled.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, family `qwen3_vl`,
  `Qwen3VLForConditionalGeneration`, Qwen3-VL processor plus fast Qwen2
  tokenizer, tokenizer length `151669`, BF16 weights, SDPA, local-files-only,
  no remote code, no tokenizer resize, and no full weight-directory hash by the
  accepted operational-identity decision.
- Representation checkpoint identity: initialization was fresh random TGVF
  Adapter seed `20260719`; historical checkpoints are forbidden. The planned
  outputs materialized as a 104-tensor/72,055,808-parameter Adapter-only export
  and `distributed-representation-checkpoint-v1` step-1 checkpoint; neither is
  a promoted artifact. Run identity SHA256 is
  `50fae1c344ec9286f9cfe9455490d09138eeeee4b732c9484ee86f87b8f5e71a`.
- N/A fields and justification: policy/reference models, rollout, behavior
  log probabilities, reward, GRPO, SDPO teacher, answer judge, vLLM sampling,
  policy staleness, KV cache, and exact-observation policy replay are N/A because
  this is supervised representation learning, not a policy-RL optimizer step.
- Policy/reference initialization: N/A. The frozen readout model is the original
  local Qwen3 reasoning model; only the freshly initialized TGVF Adapter is
  trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; no rollout or
  policy update. The representation step is fully synchronous.
- Code commit and worktree state: runtime implementation commit
  `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`; launch HEAD
  `bb407a53d67e6664f23162deb8b39a4be64e31b2` had a clean worktree. The latter
  commit contains only this config/ledger plan and changes no runtime-code
  identity path.
- Repository adapter/patch surface and hash: git tree
  `src/tgvf_rl/representation/training@b2ea66242bbd6e50660006327ae4ef612f5bf6f0`
  plus `src/tgvf_rl/qwen@6ae2676f7f08949f2425d736fe4d54751d53f69f`;
  config source SHA256
  `65b97f724e645fdbea93f458c6280252ad0ba693d8ccbb079e7bc8142deed582`
  and canonical SHA256
  `3e06509b2751fbcc0928385dd787e4c1f5188f93649c4d2cc557335de3498e24`.
  No external package or site-package patch is used.
- Dataset/manifest, hashes, sample rule, and n: repository-owned synthetic PPM
  fixtures only. Train JSONL SHA256
  `e939cd96278b6651349c18eea056ee52c1903a9fae1cb11e1a53e3061305bef3`,
  transformed manifest
  `e08a279db607ded1abaf725b92770468e85ea6cd68210144317b3e60ab721991`;
  validation JSONL SHA256
  `e483640c9799b02e4eaaccffe37f02cc9434198ed576c64a10d491251cc514f9`,
  transformed manifest
  `33a21a07686a2fb3f061519489a30224f34c01c788bbcb1ae38a05373efb30ad`.
  Image-byte hashes are train
  `7cbd3fc98b6c4867a4bcd32a59afcf816ed2d59bc6f5933aab7cc34735971023`,
  train2
  `8eb26f2e75b910112eff7ce7f6471ac05abb6e2c1905df9174b2e416cb55d6e3`,
  validation
  `c3e3e25031fe1d7d9f725622cba016947a11187681c01bc2f7acf9c5ed30ed06`,
  and validation2
  `e0e628249b7e89733a912544781a415062b8e23fd3487e5d2937140f2578e03c`.
  Each split has four accepted rows in two disjoint same-image K=2 groups;
  whole-group ownership gives one group to each rank. Sampler seeds are train
  `71` and validation `73`.
- Native prompt/tool schema hash: smoke-only prompt identity
  `qwen3-representation-smoke-only-v1`, SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native `tgvf_focus_tool` schema SHA256
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
  This prompt is explicitly forbidden as a production default.
- Chat-template/token-fixture hash and token-ownership masks: accepted chat
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  processor golden-file SHA256
  `1df319f994e31db398d008880b0678afb56e1b4390123957e56f25ba9c165a68`.
  Only exact processor-expanded `evidence_description` positions are labeled;
  prompt/tool JSON/tool result/wrappers/answer and every visual position are
  `-100`. The tokenizer length must remain exactly `151669` before and after.
- D/DeepStack/position/mask identity: each candidate is produced live by the
  TGVF Adapter from the original-image visual features and target-token input
  embedding, as an atomic main `D` plus branch layers `(8,16,24)`. Native
  processor positions/M-RoPE are preserved, and a 4-D additive causal mask
  blocks post-`D` evidence queries from original-image keys. Projection and
  Adapter contract identities are checkpoint-bound at runtime.
- Observation materialization/artifact identity used by all replays: no policy
  replay occurs. The detached K-by-K score pass and its differentiable
  cell-by-cell recomputation use the same in-memory candidate tensors within
  one synchronous step; equality is checked exactly before backward and no
  optimizer update intervenes.
- RL framework/version/environment lock: representation execution uses public
  PyTorch/Transformers rather than veRL trainer code; upstream veRL remains the
  project base at `e003163181731412595257a72ec173071efb125f` / `0.9.0.dev0`.
  Python `3.12.3`, Torch `2.9.0+cu128`, CUDA `12.8`, NCCL `2.27.5`,
  Transformers `4.57.6`, vLLM `0.12.0`, Pillow `12.3.0`; lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
  Bare Ubuntu `24.04.3` host/kernel `6.8.0-86-generic`, no container image;
  NVIDIA driver `570.195.03` from the same accepted host identity as `SC-20`.
- Objective equations and normalization: smoke-only identity
  `matrix-ce-plus-l-gen-smoke-only-v1`. For row `i`, candidate `j`,
  `s_ij` is the summed evidence-token log likelihood under frozen Qwen.
  `L_matrix = sum_i CE(s_i, i) / global_valid_rows` and
  `L_gen = sum_i(mean_evidence_token NLL_ii) / global_samples` across both
  ranks; `L = 1.0*L_matrix + 0.25*L_gen`. There is no Matrix-CE temperature.
  Manifold is disabled with weight `0`; norm loss is
  `unset_not_implemented`. AdamW is fully explicit: lr `3e-4`, betas
  `(0.9,0.999)`, eps `1e-8`, weight decay `.01`, no fused/foreach; constant
  one-step scheduler and global L2 gradient clip `1.0`.
- Rollout/replay forward mode and adapter dropout/RNG contract: Qwen eval and
  frozen; Adapter train with dropout exactly zero; `use_cache=false`;
  `torch.use_deterministic_algorithms(true)`, TF32 off, cuDNN benchmark off,
  current-device CUDA plus CPU/Python seed `20260719`, CUBLAS workspace
  `:4096:8`, synchronous update, and exact score-recompute equality.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A; same-image group sampling is
  deterministic without token generation, using the manifest-bound seeds
  above.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: Qwen/TGVF forward BF16, FSDP reduction
  FP32, no quantization, SDPA, no KV cache, no rollout TP; one-dimensional
  composable-FSDP2 mesh `fsdp=[2]`, `reshard_after_forward=true`, no offload,
  no activation checkpointing, no LoRA. Frozen Qwen is replicated once per
  rank; every and only the 52 Adapter-owned leaf modules are sharded; four
  borrowed frozen Qwen mergers remain excluded from Adapter ownership.
- Logit/logprob/loss/gradient parity tolerances: logprob parity is N/A. PASS
  requires a zero-exit `complete` result; finite Matrix-CE, `L_gen`, total loss,
  and pre-clip gradient norm; nonzero real Adapter gradient traversal; exact
  streaming recompute equality; all four global train rows; both validation
  groups; unchanged tokenizer length; a strict content-bound DCP sidecar; and a
  loadable Adapter-only export. This cell does not claim single-device/FSDP
  numerical parity or restored-next-step parity; `GPU-F02` through `GPU-F04`
  remain open for that broader claim.
- World size, microbatch, accumulation, and global batch: world `2`; one K=2
  same-image group per rank, local rows `2`, global rows/samples `4`, gradient
  accumulation `1`, exactly one optimizer step and one validation event.
- GPUs: physical `2` =
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`; physical `3` =
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; both NVIDIA B200 183359 MiB,
  compute capability 10.0, mapped `2->logical 0`, `3->logical 1`. No other GPU
  may be visible.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T18:08:27.648+09:00` / `2026-07-19T18:09:43.649+09:00`, about
  76 seconds; execution session `43069`, torchrun launcher PID `311941`, two
  ranks identified as logical 0 and 1 in the result.
- Actual GPU-hours and peak scratch use: less than `0.043` aggregate GPU-hour
  across both devices by wall-time upper bound. Output used about `537 MiB` of
  allocated disk (`577,925,311` apparent bytes), far below the `20 GiB` cap.
  A live peak-GPU-memory sample was not instrumented, so this cell makes no
  peak-memory/capacity claim; both Qwen copies and the complete step fit and the
  process exited normally.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/SC-40-qwen3-representation-fsdp2-embedding/run.log 2>&1`.
- Outputs: `adapter.pt` SHA256
  `aaada9eb5c55577fe4cddc7d901ee13a999af8c9d423c6a596056c340571cd3a`;
  `metrics.jsonl` SHA256
  `686663b5bebd4dd8ce7cb44022715ff09a8492ee30528e41c00fb73c25071217`;
  `run.log` SHA256
  `6025ea15eed8dcd9524a591fa0ccc594419200ad9971b59b5262285f6b1c359b`.
  The strict DCP directory contains both rank shards and its integrity-checked
  sidecar. Rank shard file SHA256 values are
  `6923d8a3e74690ef7b0c333602bbfa3b2e34c1335aff4c0ffa3d28ccecf33b32`
  and
  `811d38cf9f458995252b53b1d878cd67915083e70565b895694a5915298c6109`;
  sidecar payload SHA256 is
  `000a56537a9ae82704e6133aa85ecc0a8eaa0bbb732d094c291265a5b1fba85c`.
- Scorer/parser identity: no answer scorer. Strict config/parser, native
  processor/group builder, Matrix-CE/readability evaluator, runner, DCP loader,
  and export loader are identified by runtime commit and source-tree hashes
  above.
- Metrics: `SIDE_RESULT` technical execution pass. Train global rows/samples
  `4/4`; Matrix CE
  `0.6901558041572571`, `L_gen` `4.843873739242554`, weighted total
  `1.9011242389678955`, pre-clip gradient norm `13.23893928527832`, and used lr
  `0.0003`. Validation covers two groups/four rows/24 evidence tokens with
  Matrix CE `0.73828125`, `L_gen` `3.5234375`, and weighted total
  `1.619140625`. Both rank-owned sample lists are exact. Tokenizer length is
  `151669` before and after. Independent post-run loaders verified all 104
  exported tensor checksums, the sidecar digest, and the presence/structure of
  both ranks' recorded model/optimizer shard-content digests, plus run identity,
  global step 1, and world size 2. Recomputing those logical shard digests from
  restored DCP tensors remains the explicitly open restore gate. The export
  manifest SHA256 is
  `d1f63a7a6f1cb0b69890795d95430aaf9335dffb42f78b4bca7134fd430c7158`.
- Conclusion: `INVALID` because the run ID collides with the pre-existing
  `SC-40` SDPO matrix identity. Technically, real Qwen3,
  target-token-embedding conditioning, native main `D` plus all three
  DeepStack branches, same-image Matrix CE plus `L_gen`, nonzero Adapter
  backward, two-rank FSDP2 update, validation, DCP save, and rank-zero export
  completed without tokenizer growth. Those outputs are retained only for
  audit and are not promoted or used to close a gate. `RP-10` below reruns the
  same bounded question under a valid disjoint identity.

### RP-10-QWEN3-REPRESENTATION-FSDP2-EMBEDDING

- Cell/matrix ID and mandatory/diagnostic class: `RP-10`; mandatory bounded
  real-Qwen3 representation-phase backward/FSDP2 smoke for the
  target-token-embedding provider. `RP-*` is the representation execution
  namespace and does not overlap the veRL compatibility matrix.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime baseline
  `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`, corrected config commit
  `b65e812783ba8cda22c46dcf01329fa1d196e265`, user-accepted `I8H-20260719`,
  and `AD-05G`/`AD-06`/`AD-07`.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
- Question: identical to the invalid side result above: can stable local
  Qwen3-VL-8B-Thinking complete one deterministic native same-image Matrix-CE
  plus `L_gen` optimizer step with target-token-embedding conditioning, main
  `D` and all three branches, frozen Qwen, two-rank FSDP2, validation, DCP
  save, Adapter export, and no tokenizer growth?
- Baseline and exact output path: the `295 passed` CPU suite, `SC-30`, and the
  identity-invalid technical side result above. Corrected outputs must be new
  under `artifacts/representation/RP-10-qwen3-representation-fsdp2-embedding/`:
  `adapter.pt`, `metrics.jsonl`, `checkpoints/`, and `run.log`; no overwrite.
- Model and processor identity: identical to the fully resolved invalid side
  result above: stable local Qwen3-VL-8B-Thinking, tokenizer length `151669`,
  BF16, SDPA, local-only, no remote code or tokenizer resize.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  legacy checkpoint, no promoted input artifact. Planned outputs are a new
  Adapter-only export and step-1 distributed representation checkpoint.
- N/A fields and justification: identical to the invalid side result. There is
  no policy/reference rollout, behavior logprob, reward, GRPO, SDPO teacher,
  judge, vLLM sampling, KV cache, or policy replay in this representation run.
- Policy/reference initialization: N/A; original Qwen3 is frozen and only the
  freshly initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation step with no rollout or intervening update.
- Code commit and worktree state: runtime code
  `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`; corrected config commit
  `b65e812783ba8cda22c46dcf01329fa1d196e265`; launch requires a clean worktree
  after committing this `PLANNED` row, with no later runtime-code change.
- Repository adapter/patch surface and hash: training tree
  `b2ea66242bbd6e50660006327ae4ef612f5bf6f0`, Qwen tree
  `6ae2676f7f08949f2425d736fe4d54751d53f69f`; corrected config source SHA256
  `817cca317bb68adb9d68a9a3cb9a51a5d12fdb5d87194d6a7c4aee6ef3f07e76`
  and canonical SHA256
  `ce73fd382b853e535e841b03563ad729bc8ad48eaa243a05e2801e187e7276f6`.
  No package/site-package patch.
- Dataset/manifest, hashes, sample rule, and n: identical exact train/
  validation JSONL, transformed manifests, four PPM byte hashes, K=2 group
  ownership, `n=4` per split, and sampler seeds `71`/`73` recorded in the
  invalid side result; the corrected config changes only run/output identity.
- Native prompt/tool schema hash: identical smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`
  and native tool-schema SHA256
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: identical accepted
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  golden-file SHA256
  `1df319f994e31db398d008880b0678afb56e1b4390123957e56f25ba9c165a68`,
  exact evidence-only labels, visual `-100` labels, and tokenizer-length gate.
- D/DeepStack/position/mask identity: identical live main `D`, branch layers
  `(8,16,24)`, native positions/M-RoPE, 4-D original-image key block, and
  runtime-bound Adapter/projection identity contract.
- Observation materialization/artifact identity used by all replays: no policy
  replay; identical in-step candidate materialization and exact detached-score/
  differentiable-recompute equality, with no intervening update.
- RL framework/version/environment lock: identical lock and host identity:
  Python `3.12.3`, Torch `2.9.0+cu128`, CUDA `12.8`, NCCL `2.27.5`,
  Transformers `4.57.6`, vLLM `0.12.0`, Pillow `12.3.0`, upstream veRL
  `e003163181731412595257a72ec173071efb125f`, lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`,
  Ubuntu `24.04.3`, kernel `6.8.0-86-generic`, driver `570.195.03`, no container.
- Objective equations and normalization: identical explicit smoke equation:
  `s_ij` summed evidence log likelihood; global-row-mean Matrix CE plus global-
  sample-mean per-sample-token-mean `L_gen`; weights `1.0`/`.25`; manifold
  zero; norm unset; explicit AdamW lr `3e-4`; constant one-step scheduler;
  global L2 clip `1.0`.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical Qwen
  eval/frozen, Adapter dropout zero, `use_cache=false`, deterministic algorithms,
  TF32 off, cuDNN benchmark off, seed `20260719`, CUBLAS `:4096:8`, exact
  score-recompute equality.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: token sampling/logprobs N/A;
  deterministic same-image sampler seeds `71`/`73`.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 Qwen/TGVF, FP32 reduction, no
  quantization/KV/rollout TP, SDPA, `fsdp=[2]`, reshard after forward, no
  offload/activation checkpointing/LoRA; frozen Qwen replicated and exactly 52
  Adapter-owned leaves sharded.
- Logit/logprob/loss/gradient parity tolerances: identical structural PASS gate
  to the invalid side result: complete/zero-exit, finite component losses and
  nonzero gradient norm, exact recompute, four train rows/two validation groups,
  tokenizer unchanged, loadable 104-tensor export and integrity-checked DCP
  sidecar. No single-device numerical or restore/next-step parity claim.
- World size, microbatch, accumulation, and global batch: world 2; local K=2,
  global rows/samples 4; accumulation 1; one optimizer step and validation.
- GPUs: physical 2 and 3 only, with the exact B200 UUIDs and logical mapping
  recorded above; no other physical GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: `PENDING`.
- Actual GPU-hours and peak scratch use: hard timeout `1800s`, maximum `1.0`
  aggregate GPU-hour, new output cap `20 GiB`; retain failures for audit.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp10.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-10-qwen3-representation-fsdp2-embedding/run.log 2>&1`.
- Outputs: planned corrected `adapter.pt`, `metrics.jsonl`, one DCP directory,
  and `run.log` under the exact RP-10 path.
- Scorer/parser identity: no answer scorer; strict config/native pipeline/
  objective/evaluation/runner/DCP/export code at the tree identities above.
- Metrics: `PENDING`.
- Conclusion: `PENDING`; the same fail-fast rules as the invalid side result
  apply. RP-10 receives no gate credit until its own outputs pass independent
  integrity checks.

## Compatibility-spike status

CPU public-API, transport, objective and oracle tests passed before these rows
were entered. The completed cells are bounded evidence; they do not silently
close broader Qwen replay, Qwen2.5, production-objective or training gates.

## Required entry template

```text
### <ID>

- Cell/matrix ID and mandatory/diagnostic class:
- Spike-plan git revision and VA0/VA1/VA2 approval references:
- Lifecycle status: PLANNED | RUNNING | COMPLETE | CANCELLED
- Result: PENDING | PASS | FAIL | BLOCKED_NOT_RUN | INVALID | SIDE_RESULT
- Question:
- Baseline and exact output path:
- Model and processor identity:
- Representation checkpoint identity:
- N/A fields and justification:
- Policy/reference initialization:
- Rollout policy version and allowed asynchronous staleness:
- Code commit and worktree state:
- Repository adapter/patch surface and hash:
- Dataset/manifest, hashes, sample rule, and n:
- Native prompt/tool schema hash:
- Chat-template/token-fixture hash and token-ownership masks:
- D/DeepStack/position/mask identity:
- Observation materialization/artifact identity used by all replays:
- RL framework/version/environment lock:
- Objective equations and normalization:
- Rollout/replay forward mode and adapter dropout/RNG contract:
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention:
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh:
- Logit/logprob/loss/gradient parity tolerances:
- World size, microbatch, accumulation, and global batch:
- GPUs:
- Start/end timestamps, elapsed time, and session/process identity:
- Actual GPU-hours and peak scratch use:
- Command:
- Outputs:
- Scorer/parser identity:
- Metrics:
- Conclusion:
```

Before any launch, every applicable field must be resolved from files rather
than inferred from a script name or prior conversation.
