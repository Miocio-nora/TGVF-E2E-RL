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
  Adapter seed `20260719`; historical checkpoints are forbidden. The configured
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
  same bounded question under a valid disjoint identity. The invalid TOML was
  removed from the current tree after closeout to prevent accidental reuse; it
  remains recoverable at launch commit `bb407a5` with the source hash above.

### RP-10-QWEN3-REPRESENTATION-FSDP2-EMBEDDING

- Cell/matrix ID and mandatory/diagnostic class: `RP-10`; mandatory bounded
  real-Qwen3 representation-phase backward/FSDP2 smoke for the
  target-token-embedding provider. `RP-*` is the representation execution
  namespace and does not overlap the veRL compatibility matrix.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime baseline
  `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`, corrected config commit
  `b65e812783ba8cda22c46dcf01329fa1d196e265`, user-accepted `I8H-20260719`,
  and `AD-05G`/`AD-06`/`AD-07`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
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
  legacy checkpoint, no promoted input artifact. Outputs materialized as a new
  104-tensor/72,055,808-parameter Adapter-only export and step-1 distributed
  representation checkpoint. Run identity SHA256 is
  `da1d5c16a361e5b1cbb5b949b43af789fcd438402c38aebd86f7c89cbb1590e1`.
- N/A fields and justification: identical to the invalid side result. There is
  no policy/reference rollout, behavior logprob, reward, GRPO, SDPO teacher,
  judge, vLLM sampling, KV cache, or policy replay in this representation run.
- Policy/reference initialization: N/A; original Qwen3 is frozen and only the
  freshly initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation step with no rollout or intervening update.
- Code commit and worktree state: runtime code
  `ce6a15f5e7df6fabf57f0997cee279efb66a96e4`; corrected config commit
  `b65e812783ba8cda22c46dcf01329fa1d196e265`; launch HEAD
  `7dc016e843f811ab64ccfccf50da187be24a40c7` had a clean worktree and no later
  runtime-code change.
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
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T18:26:04.696+09:00` / `2026-07-19T18:26:38.962+09:00`, about
  34.3 seconds; execution session `32361`, torchrun launcher PID `410395`, and
  two logical ranks 0/1.
- Actual GPU-hours and peak scratch use: less than `0.020` aggregate GPU-hour
  by wall-time upper bound. Output used about `537 MiB` allocated disk
  (`577,924,967` apparent bytes), below the `20 GiB` cap. Peak GPU memory was
  not instrumented, so this remains a functional rather than capacity claim.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp10.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-10-qwen3-representation-fsdp2-embedding/run.log 2>&1`.
- Outputs: `adapter.pt` SHA256
  `1d6acbc6c2330410117c20420c7b40c11c1be29559c07ff16bbe9ed593b3e014`;
  `metrics.jsonl` SHA256
  `5b41c2107ad9be3a224e90c05f2f1f9b46408aa3ec48ec01551087ab61419fed`;
  `run.log` SHA256
  `21ae57da8db06b03be7939751ba0de02fa6bde80cfb4fa76a5a8752360f966df`.
  DCP rank shard file SHA256 values are
  `6923d8a3e74690ef7b0c333602bbfa3b2e34c1335aff4c0ffa3d28ccecf33b32`
  and
  `811d38cf9f458995252b53b1d878cd67915083e70565b895694a5915298c6109`;
  sidecar payload SHA256 is
  `2617f8bb7db41beae89b997dc8f454c7ecb0896f0c32fd649823731f6f845fd0`.
- Scorer/parser identity: no answer scorer; strict config/native pipeline/
  objective/evaluation/runner/DCP/export code at the tree identities above.
- Metrics: `PASS`. Train global rows/samples `4/4`; Matrix CE
  `0.6901558041572571`, `L_gen` `4.843873739242554`, total
  `1.9011242389678955`, pre-clip gradient norm `13.23893928527832`, lr
  `0.0003`. Validation covers two groups/four rows/24 evidence tokens with
  Matrix CE `0.73828125`, `L_gen` `3.5234375`, and total `1.619140625`.
  Tokenizer length remained `151669`. Independent loaders verified all 104
  export tensor checksums, sidecar integrity, recorded rank-local shard-digest
  fields, run identity, step, and world size. Adapter tensor checksums match the
  invalid side result exactly, providing a deterministic rerun cross-check;
  identity-bearing file/manifest hashes correctly differ. Export manifest
  SHA256 is
  `d05c8b04aa6f0c753afc9e2cc7e18adb5a530871e3f8cd1de82af4ef9cf2b761`.
- Conclusion: `PASS` for the valid `RP-10` bounded question. Real Qwen3,
  target-token-embedding conditioning, native main `D` plus all three
  DeepStack branches, Matrix CE plus `L_gen`, nonzero Adapter backward,
  two-rank FSDP2 update, validation, DCP save, and Adapter export completed
  without tokenizer growth. This does not claim production data/quality,
  promoted training, a contextual-provider comparison, single-device/FSDP
  numerical parity, or distributed restore/resumed-next-step parity.

### RP-11-QWEN3-REPRESENTATION-FSDP2-EMBEDDING-K4-RESUME

- Cell/matrix ID and mandatory/diagnostic class: `RP-11`; mandatory bounded
  real-Qwen3 representation-phase K=4/GA=4 continuous-versus-process-teardown
  resume smoke for the target-token-embedding provider.
- Spike-plan git revision and VA0/VA1/VA2 approval references: user-accepted
  `I8H-20260719`, `RPI-20260719-NORM-EVAL`, `AD-05G`, `AD-06`, and `AD-07`;
  runtime commit `41fb07d1c8bfdaf0388115de6062b1cc8732bcc0`; exact configs commit
  `69534f2440c58bb623c26d4811b76bb903d1c754`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: do an uninterrupted two-update run and an otherwise identical run
  stopped after update 1, fully torn down, reconstructed, restored, and advanced
  through update 2 produce exact Adapter tensors, recorded optimizer/scheduler/
  sampler/RNG/shard state, and scientific train/validation records?
- Baseline and exact output path: corrected `RP-10` plus `346 passed` CPU tests.
  New immutable lanes are
  `artifacts/representation/RP-11-qwen3-representation-fsdp2-embedding-k4-resume/continuous/`
  and `.../split/`; comparison output is `resume_comparison.json` in the RP-11
  root. No RP-10 artifact is an initialization.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, Qwen3-VL-8B-Thinking,
  tokenizer length `151669`, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  BF16, SDPA, local-only, no remote code and no tokenizer resize.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  source artifact or legacy checkpoint; DCP v2, planned optimizer target 2,
  historical scheduler horizon 2,000, checkpoint every update, keep-last 2.
- N/A fields and justification: no policy/reference rollout, behavior logprob,
  reward, GRPO, SDPO teacher, answer judge, vLLM sampling, KV replay, or policy
  replay exists in this representation-only smoke.
- Policy/reference initialization: N/A; original Qwen3 is frozen and only the
  freshly initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation updates with no intervening policy update.
- Code commit and worktree state: runtime code
  `41fb07d1c8bfdaf0388115de6062b1cc8732bcc0`, code-identity paths clean;
  config commit `69534f2440c58bb623c26d4811b76bb903d1c754`; launch HEAD
  `7ba4348378f699f9de9e821b3dabf23263e23680`. Runtime code paths were clean;
  the only launch-time difference from runtime commit was committed config and
  experiment-ledger state.
- Repository adapter/patch surface and hash: representation-training tree
  `5e2b227c9b2db7ec7c80ff57be68e10f6d243e03`; Qwen tree
  `6ae2676f7f08949f2425d736fe4d54751d53f69f`. Continuous/split-fresh/
  split-resume TOML source SHA256 values are respectively
  `7a5895c8c81173fecbe536fb5e791b1858a9ff75868a33ac872f023f469d4fc9`,
  `30024f5c0acc46352a902e0105b8eb8e18c0563fb3b543eec1eca70b8e1ded43`,
  and `ea2bbc9af5651de19c4fb2b8627fbf3979a4bbc4fdcd383e22445b3eadf819ea`;
  canonical SHA256 values are `3981efdabc5eb1f848b48c89987b94df7a877f2663d17fda7216840ab909d887`,
  `100427801783e61f68e2cdd42274898d62bf2f1d7c214c0e091228a85aeb383f`,
  and `5f78964d39b475f0dd8ddbfb4b2a7fc7d9cc211e4816368f751334acbad741ba`.
- Dataset/manifest, hashes, sample rule, and n: smoke train/validation JSONL
  source SHA256 `beb1b8a7c3f97811e8a8f9b0734d7484cc5de4d31861fe09b61342b3c88b61f2`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`;
  retained manifest `14b92fbfa061bcbfabd2790aaeec04dc429cb4d7e756f38d68be54d71da708f6`/
  `e18eb856f4c8b03c03a2f2a683a8267bb0581498de05fed403c4246e5cca3a13`;
  8 rows and two complete K=4 groups per split, rank 0/1 owns one group,
  sampler seeds 71/73. Empty split-overlap report SHA256 is
  `c23fe08408640a95d0fe9c482228908bc35e7816021e51b029cc6c9d6979c218`;
  validation-data identity is
  `fff7635f0e873bb45c8f6dc08e788e3bc1f33246d9751fb1842f69e1139591a0`.
- Native prompt/tool schema hash: explicit smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native `tgvf_focus_tool` schema SHA256
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: accepted Qwen3
  golden SHA256 `1df319f994e31db398d008880b0678afb56e1b4390123957e56f25ba9c165a68`;
  evidence-description-only labels, all visual/template positions `-100`, and
  no tokenizer growth.
- D/DeepStack/position/mask identity: native main `D`, ordered branch layers
  `(8,16,24)`, family-native M-RoPE, original-image post-D key blocking, and
  one atomic main-plus-all-branches candidate observation.
- Observation materialization/artifact identity used by all replays: no policy
  replay; Matrix scoring and differentiable recomputation consume the same
  in-update candidate observations. Resume binds exact validation image bytes,
  metrics prefix/cursor, sampler and RNG state.
- RL framework/version/environment lock: Python `3.12.3`, Torch `2.9.0+cu128`,
  CUDA `12.8`, NCCL `2.27.5`, Transformers `4.57.6`, vLLM `0.12.0`, veRL
  `0.9.0.dev0`, Pillow `12.3.0`; `requirements/compatibility.lock` SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: same-image diagonal-label Matrix CE
  is a global valid-row mean; `L_gen` is the global sample mean of per-sample
  causal evidence-token mean NLL; historical Norm is the global sample mean of
  `mean_t(log((||D_t||+1e-6)/clamp_min(mean_s||V_s||,1e-6))^2)`, with the
  three-branch mean averaged equally with main. Weights are `1.0/1.0/0.1`;
  manifold is exactly zero. AdamW is `1e-4`, `(0.9,0.999)`, epsilon `1e-8`,
  decay `.01`; clip `1.0`; cosine warmup 100, floor ratio `.1`, horizon 2,000.
- Rollout/replay forward mode and adapter dropout/RNG contract: Qwen frozen and
  eval, Adapter dropout zero, `use_cache=false`, deterministic algorithms,
  TF32 off, cuDNN benchmark off, seed `20260719`, CUBLAS `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: token sampling/logprobs N/A;
  deterministic same-image sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights/forward, FP32 reduction,
  no quantization/KV/rollout TP, SDPA, FSDP2 `fsdp=[2]`, reshard after forward,
  no offload.
- Logit/logprob/loss/gradient parity tolerances: Adapter manifests/tensors use
  exact equality; validated DCP model/optimizer/scheduler/sampler/RNG/shard
  digests and all train/validation scientific records must match exactly.
  Only timing and allocator evidence is excluded. Split step-1 DCP payload must
  be restored and content-digest checked; final DCP comparison is sidecar-state
  exact and does not claim a second independent payload restore.
- World size, microbatch, accumulation, and global batch: world 2, local K=4,
  four accumulation microsteps; 32 global rows/eight `4x4` matrices per update,
  64 rows/16 matrices over two updates; validation every update.
- GPUs: physical 2 `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, NVIDIA B200, logical 0/1.
  Preflight showed 0 MiB used and 182,642 MiB free on each; no other GPU may
  be visible.
- Start/end timestamps, elapsed time, and session/process identity: continuous
  launcher PID `535857`, `2026-07-19T20:12:18+09:00` to
  `20:13:37+09:00` (about 79 s); split-step-1 launcher PID `536992`,
  `20:14:01` to `20:14:54` (about 53 s); fresh resume launcher PID `538213`,
  `20:15:17` to `20:16:10` (about 53 s). Every invocation exited zero inside
  its 3,600-second bound.
- Actual GPU-hours and peak scratch use: less than `0.103` aggregate GPU-hour
  by the three-invocation wall-time upper bound; RP-11 artifacts use
  `2,023,388,296` bytes (`1.9 GiB`). Maximum measured train-step allocated/
  reserved CUDA memory was `21,217,806,848`/`22,299,017,216` bytes on one rank.
- Command: continuous:
  `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 3600s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp11_continuous.toml`;
  split step 1: same environment/torchrun with
  `representation_qwen3_embedding_rp11_split_fresh.toml --stop-after-global-step 1`;
  split resume: same environment/torchrun with
  `representation_qwen3_embedding_rp11_split_resume.toml`; then
  `.venv312/bin/python -m tgvf_rl.cli compare-representation-resume` with both
  exact artifacts, step-2 checkpoints, and metrics paths recorded above.
- Outputs: both Adapter files are byte-identical, SHA256
  `665d0189a43a242386110893ab111e27f57953d672588184b69ff9680d3d8d84`.
  Continuous/split metrics SHA256 are
  `19046cbcc53bdc6bb9d8d1a99bc110d694378a1bb14d3e45d2729a99f6d1da`/
  `bf31353f82c6199c90e99f98f84a4efdd2d7bd07db66ac7f0c4de80f30616fa6`;
  continuous/step1/resume log SHA256 are
  `2219f480b4f464871a23565f31018c0758e41ed3229bdb2526e8cd0309566524`/
  `ec94fe792e11add7525f07efba415c6fe6a5934db7cb38e5d68c8e846df30a74`/
  `c76bd86b6a84e64e063d360794b12e2b648eb76bf3782ff52f9c7c3a0188b7f8`.
  Comparator JSON SHA256 is
  `e077dedda1c374075bf282b1ccd3a838d356f2e5f06ef45797a6a859f1175577`.
- Scorer/parser identity: no answer scorer; strict config/native pipeline,
  objective/evaluation, DCP/history, and resume comparator at runtime commit.
- Metrics: `PASS`. Both lanes have identical scientific records. Step 1/2
  learning rates are `1.0000000000000002e-6`/`2.0000000000000003e-6`;
  total losses are `6.032827917486429`/`6.06851127743721`; Matrix CE
  `1.4035631567239761`/`1.4297739267349243`; `L_gen`
  `4.596969246864319`/`4.606441974639893`; Norm
  `.3229551389813423`/`.3229537606239319`; gradient norms
  `5.383439540863037`/`9.009767532348633`. Continuous step times are
  `21.574`/`20.366` s (`1.483`/`1.571` global rows/s). Validation totals at
  steps 1/2 are `5.7763159327209`/`5.756785213202238`. Tokenizer length stayed
  `151669`. The resume comparator reports `exact=true`, 104 Adapter tensors,
  two train and two validation records, world size 2, run identity SHA256
  `f56b5c64380e1a6c796c498a971a3accd1fc7b37f68d68d9b91c39d392a04299`,
  and exact recorded model/optimizer/rank-state digests.
- Conclusion: `PASS` for the bounded K=4/GA=4 real-Qwen3 teardown/resume
  question. This closes the representation executor's real distributed
  restore/next-update proof. It is not a production prompt/data-quality result,
  a semantic-quality threshold result, a formal native counterfactual run, or
  a promoted representation artifact.

### RP-12-QWEN3-REPRESENTATION-FSDP2-EMBEDDING-K4-GPR4-GA1-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-12`; diagnostic bounded
  real-Qwen3 representation-phase K=4/GPR=4/GA=1 direct-batch train-core
  throughput smoke for the target-token-embedding provider. This is not a
  promoted training run.
- Spike-plan git revision and approval references: user direction on
  `2026-07-19` to directly measure GA=1 after the RP-11 memory result;
  accepted task commit `30d0622`, `RPI-20260719-NORM-EVAL`, `AD-05G`, `AD-06`,
  and `AD-07`. Runtime commit is
  `9b47105e761ceac015094b46d3844f89df63bc10`; config/data commit is
  `6ebed9d02008f8ed0acf26b547b4e2d101875b0f`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: at the same mathematical global batch of 32 used by RP-11, what
  train-core optimizer-step time, rows/s, matrices/s, and peak CUDA memory are
  measured when four independent K=4 blocks per rank are executed directly at
  GA=1 instead of through four accumulation microsteps?
- Baseline and exact output path: RP-11 continuous step 2 measured `20.366` s,
  `32` rows, eight matrices, `1.571` rows/s, and at most
  `21,207,001,600` allocated bytes. RP-12 writes only under
  `artifacts/representation/RP-12-qwen3-representation-fsdp2-embedding-k4-gpr4-ga1-throughput/`.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, tokenizer length
  `151669`, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  BF16, SDPA, local-only, no remote code, and no tokenizer resize.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  source artifact and no legacy-checkpoint initialization; three optimizer
  updates under the historical 2,000-update scheduler horizon.
- N/A fields and justification: policy/reference rollout, behavior logprob,
  reward, GRPO, SDPO teacher, answer judge, vLLM sampling, KV replay, and policy
  replay are absent from this representation-only timing smoke.
- Policy/reference initialization: N/A; original Qwen3 is frozen and only the
  freshly initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; this is a
  synchronous representation update with no intervening policy update.
- Code commit and worktree state: runtime code at
  `9b47105e761ceac015094b46d3844f89df63bc10`; config source/canonical SHA256
  `95079be0c077314894962fe3ec273f5e3bfe5b6985ef7d5e77ab78e2a56c7721`/
  `440ef254e6c3aa2b6660fa54bb9733f596d71ca97c3859c65e8325ceecd8647c`.
  Launch HEAD was `c4f86d5ea2748109edd46a6c4b99156d0236d9bb`; the worktree was clean.
- Repository adapter/patch surface and hash: representation-training tree
  `e23a8cd63bdca430a6bf24f689b3707ff8fed777` and unchanged Qwen tree
  `6ae2676f7f08949f2425d736fe4d54751d53f69f`.
- Dataset/manifest, hashes, sample rule, and n: direct-batch smoke
  train/validation source SHA256
  `34053c694023461be9f0fe30fd1e525e27697894f705e5428fafad45cfeabc1c`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`;
  retained manifests `c95ec424a64d6303a90f245aa52fa1c24eba28963af0455ddcb0e7830efe53ba`/
  `e18eb856f4c8b03c03a2f2a683a8267bb0581498de05fed403c4246e5cca3a13`;
  32 unique rows/eight complete K=4 groups, with four distinct groups owned by
  each rank and sampler seeds 71/73. One update consumes one complete train
  epoch. Validation-data identity is
  `16c33970676fb41c197cb007a2a7ea0c5d51b0fbcd8f3268bef6284f2f579821`;
  train/validation image-byte manifests are
  `692847ec741898bd077d7831c5a73cfac4ce4ff3eac399603ce6170330d2ba26`/
  `f83d49e1290557428e807aca8624827a8b9a4a4892e523b88c281d010fa213d9`.
  The empty overlap-report SHA256 is
  `c23fe08408640a95d0fe9c482228908bc35e7816021e51b029cc6c9d6979c218`.
- Native prompt/tool schema hash: smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native `tgvf_focus_tool` schema SHA256
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: Qwen3 golden
  SHA256 `1df319f994e31db398d008880b0678afb56e1b4390123957e56f25ba9c165a68`;
  evidence-description-only labels and all visual/template positions `-100`.
- D/DeepStack/position/mask identity: native main `D`, ordered branch layers
  `(8,16,24)`, native M-RoPE, post-D original-image key blocking, and atomic
  main-plus-all-branches candidates, unchanged from RP-11.
- Observation materialization/artifact identity used by all replays: no policy
  replay; Matrix scoring and differentiable recompute use the same in-update
  candidate observations.
- RL framework/version/environment lock: Python `3.12.3`, Torch `2.9.0+cu128`,
  CUDA `12.8`, NCCL `2.27.5`, Transformers `4.57.6`, vLLM `0.12.0`, veRL
  `0.9.0.dev0`, Pillow `12.3.0`; compatibility lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: unchanged RP-11 global-row-mean
  Matrix CE, global sample mean `L_gen`, and fixed historical Norm; weights
  `1.0/1.0/0.1`, manifold zero, AdamW `1e-4`, gradient clip `1.0`, historical
  cosine warmup 100/floor ratio `.1`/horizon 2,000.
- Rollout/replay forward mode and adapter dropout/RNG contract: frozen Qwen in
  eval mode, Adapter dropout zero, no cache, deterministic algorithms, TF32
  off, seed `20260719`, and CUBLAS `:4096:8`.
- Sampling backend/version, parameters, and logprob convention: token sampling
  and logprobs N/A; deterministic same-image sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 forward, FP32 reduction, no
  quantization/KV/rollout TP, SDPA, FSDP2 `fsdp=[2]`, reshard after forward,
  and no offload.
- Logit/logprob/loss/gradient parity tolerances: the direct path is gated by a
  blockwise objective/gradient fixture and must keep four separate local
  `4x4` blocks rather than one `16x16` block. It must complete all updates,
  retain tokenizer length, report finite losses/gradients, and report the exact
  expected row/matrix counts; performance values are measurements.
- World size, direct batch, accumulation, and global batch: world 2, four
  distinct local K=4 groups per rank, GA=1; 16 local/32 global rows and eight
  independent `4x4` matrices per optimizer update, 96 row presentations/24
  matrix presentations over three updates.
- GPUs: only physical 2
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, both NVIDIA B200 and mapped to
  logical 0/1. Preflight: `0` MiB used and `182642` MiB free on each.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T22:40:57+09:00` to `22:42:17+09:00`, about 81 seconds;
  launcher PID `834140`, worker PIDs `834755`/`834756`; exit zero inside the
  1,800-second hard limit.
- Actual GPU-hours and peak scratch use: less than `0.045` aggregate GPU-hour;
  output root uses `577,934,377` bytes. Maximum train-step allocated/reserved
  CUDA memory was `31,016,656,896`/`33,632,026,624` bytes.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp12_ga1_throughput.toml`.
- Outputs: Adapter SHA256
  `b13a1c8e56221c5174af50ba489775e11824b6de932f53a92b9410f6b560fb72`;
  metrics/run-log SHA256
  `29dc8ca5f3be3acd09dd947f85dc51fac4135025cb8636cf11aff063e2a2e221`/
  `49b01386b99e0c4572324163bccb11b0df3aaf25b94f4f11ca5bdf7f6b69ad27`;
  final step-3 DCP checkpoint. Export-manifest SHA256 is
  `ad6e0670085ffa5046f99d486e425d0aaf9d073abc93fb3f62775d510fea60fb`.
- Scorer/parser identity: no answer scorer; unchanged strict native pipeline,
  objective, performance measurement, runner, DCP, and export implementation at
  the runtime commit.
- Metrics: `PASS`. Every step reported 32 rows/eight matrices and every rank
  reported 16 sample IDs from four distinct group keys. Step 1/2/3 took
  `17.662964305`/`16.225037827`/`16.200917722` seconds at
  `1.811700429`/`1.972260425`/`1.975196748` rows/s. The steady step-2/3 means
  are `16.212977775` seconds and `1.973728587` rows/s. Against RP-11 step 2,
  optimizer-step time fell `20.39%` and row throughput rose `25.62%`
  (`1.256x`). Peak allocated memory rose from `21,207,001,600` to
  `31,016,656,896` bytes but remains far below B200 capacity. Losses and
  gradients were finite; tokenizer length stayed `151669`; run identity is
  `7af1be27960842f086eb42a9851ed62c3e1f78c5dc32cec8ce6e8b8fb3489629`.
- Conclusion: `PASS` for the bounded same-global-batch throughput question.
  Direct GPR4/GA1 is materially faster and fits comfortably, so configured
  gradient accumulation is not justified by memory for this B200 geometry.
  This is throughput evidence, not a production prompt/data-quality result.

### RP-13-QWEN3-REPRESENTATION-FSDP2-EMBEDDING-K4-GPR4-GA1-SINGLEPASS-CELLB32-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-13`; diagnostic bounded
  real-Qwen3 representation-phase K=4/GPR=4/GA=1 single-pass physical-cell-B32
  train-core throughput smoke for the target-token-embedding provider. This is
  not a promoted training run. Launch is allowed only from the clean committed
  tree carrying the exact runtime and configuration identities below.
- Spike-plan git revision and approval references: user direction on
  `2026-07-19` fixes the direct CELLB32 comparison and the boundary in this
  planned entry. Accepted implementation/task revision is
  `2af89671f9628c7432a5431d1e6bd1e2b9843b74`, under decision
  `RPI-20260719-B200-BATCHED-READOUT`; its focused parity gate passed before
  launch.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: at the same mathematical global batch of 32 as RP-12, what
  train-core optimizer-step time, rows/s, matrices/s, physical Qwen call count,
  and peak CUDA memory are measured when every local Matrix-CE cell is executed
  once in exactly two differentiable physical B32 Qwen forwards per rank and
  the results remain four independent K=4 blocks rather than one cross-image
  matrix?
- Baseline and exact output path: RP-12 steady steps 2/3 measured
  `16.212977775` seconds, `1.973728587` global rows/s, approximately `0.493432`
  global matrices/s, and at most `31,016,656,896` allocated bytes. RP-13 may
  write only under
  `artifacts/representation/RP-13-qwen3-representation-fsdp2-embedding-k4-gpr4-ga1-singlepass-cellb32-throughput/`.
  No unverified historical wall time is a numerical baseline for this cell.
- Model and processor identity: unchanged RP-12 stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`; model family `qwen3_vl`,
  tokenizer length `151669`, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  BF16, SDPA, local-only, no remote code, and no tokenizer resize.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  source artifact and no legacy-checkpoint initialization. The initialized
  Adapter state must match RP-12 SHA256
  `715f385ffa0c506572020fbc2eaff8b33c0cdf1cdaad2324efaae11335b85df9`.
  The bounded run targets three optimizer updates under the unchanged
  historical 2,000-update scheduler horizon. The checkpoint-bound
  initialization identity confirmed the expected initial Adapter SHA256
  `715f385ffa0c506572020fbc2eaff8b33c0cdf1cdaad2324efaae11335b85df9`.
- N/A fields and justification: policy/reference rollout, behavior logprob,
  reward, GRPO, SDPO teacher, answer judge, vLLM sampling, KV replay, and policy
  replay are absent from this representation-only timing smoke.
- Policy/reference initialization: N/A; original Qwen3 is frozen and only the
  freshly initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; this is a
  synchronous representation update with no intervening policy update.
- Code commit and worktree state: runtime commit
  `2af89671f9628c7432a5431d1e6bd1e2b9843b74`; configuration source/canonical
  SHA256 `13c6586c5e68d5ea5db34dd7ae3192f83cfc653a574a61211fd65ff2eea20c0e`/
  `490e60b979bea1be2b6c3ef0fd190fd74b858ad582eb9b90e20f64534992aea2`.
  The launcher must verify that runtime commit through the strict code-identity
  check and start from a clean worktree. The exact clean launch HEAD is
  `62f722e91caa15a68512b3ba9af189cc5330fc70`.
- Repository adapter/patch surface and hash: representation-training tree
  `c1d2ad19e225458bd7a2dda70ea702eeea984623`; unchanged Qwen tree
  `6ae2676f7f08949f2425d736fe4d54751d53f69f`. Single-pass streaming,
  trainer, and physical-execution telemetry SHA256 are
  `e12545c6959b6848d0d58ee4a094d51fa29ca5e52bc969435712d4055361b6b0`,
  `08f8ab7f1f8b46d3c7c7b2d05e5cc2ba029901f7cfa5e5254ce8182e155b0209`,
  and `a46faebb4eb747e33998b614641b82890a2d9fdcd4d7ad237e1e03d825ad9b32`.
  No site-package patch is permitted.
- Dataset/manifest, hashes, sample rule, and n: unchanged RP-12 direct-batch
  smoke train/validation source SHA256
  `34053c694023461be9f0fe30fd1e525e27697894f705e5428fafad45cfeabc1c`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`;
  retained manifests
  `c95ec424a64d6303a90f245aa52fa1c24eba28963af0455ddcb0e7830efe53ba`/
  `e18eb856f4c8b03c03a2f2a683a8267bb0581498de05fed403c4246e5cca3a13`;
  32 unique rows/eight complete K=4 groups, four distinct groups per rank, and
  sampler seeds 71/73. The cadence-bound validation-data identity is
  `54a36c5b3f2b27d10d71a64d5be090f533b37780bb31b22027c3ca05a8f49900`;
  train/validation image-byte manifests are
  `692847ec741898bd077d7831c5a73cfac4ce4ff3eac399603ce6170330d2ba26`/
  `f83d49e1290557428e807aca8624827a8b9a4a4892e523b88c281d010fa213d9`;
  empty overlap-report SHA256 is
  `c23fe08408640a95d0fe9c482228908bc35e7816021e51b029cc6c9d6979c218`.
- Native prompt/tool schema hash: unchanged smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native `tgvf_focus_tool` schema SHA256
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: unchanged Qwen3
  golden SHA256
  `1df319f994e31db398d008880b0678afb56e1b4390123957e56f25ba9c165a68`;
  evidence-description-only labels and all visual/template positions `-100`.
- D/DeepStack/position/mask identity: unchanged native main `D`, ordered branch
  layers `(8,16,24)`, native M-RoPE, post-D original-image key blocking, and
  atomic main-plus-all-branches candidate observations.
- Observation materialization/artifact identity used by all replays: no policy
  replay. Each in-update candidate observation is materialized once and used by
  the live differentiable single-pass readout. Each physical B32 forward packs
  two complete row slots across all four local K=4 groups: eight cells from
  each group, partitioned back into eight group-local four-candidate CE rows.
  The second B32 forward completes the four independent score matrices.
  Cross-group logits never enter the same CE denominator, and detached
  score/recompute execution is forbidden in this cell.
- RL framework/version/environment lock: unchanged Python `3.12.3`, Torch
  `2.9.0+cu128`, CUDA `12.8`, NCCL `2.27.5`, Transformers `4.57.6`, vLLM
  `0.12.0`, veRL `0.9.0.dev0`, and Pillow `12.3.0`; compatibility lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: exactly unchanged RP-12 global-row-
  mean diagonal-label Matrix CE over eight separate K=4 blocks, global sample
  mean of per-sample evidence-token-mean `L_gen`, and fixed historical Norm;
  weights `1.0/1.0/0.1`, manifold zero, AdamW `1e-4`, gradient clip `1.0`, and
  historical cosine warmup 100/floor ratio `.1`/horizon 2,000. The physical
  Qwen packing changes execution only, never negatives or denominators.
- Rollout/replay forward mode and adapter dropout/RNG contract: frozen Qwen in
  eval mode, Adapter dropout zero, no cache, deterministic algorithms, TF32
  off, seed `20260719`, and CUBLAS `:4096:8`. Exactly two differentiable Qwen
  forward calls of physical size 32 occur per rank/update; every one of the 64
  local cells occurs once and no deterministic recompute pass occurs.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: token sampling and logprobs N/A;
  deterministic same-image sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: unchanged BF16 forward, FP32
  reduction, no quantization/KV/rollout TP, SDPA, FSDP2 `fsdp=[2]`, reshard
  after forward, and no offload.
- Logit/logprob/loss/gradient parity tolerances: focused fixture
  `tests/representation/training/test_streaming.py` at runtime commit
  `2af89671f9628c7432a5431d1e6bd1e2b9843b74`, SHA256
  `4667d397f629d9e39bec0f564f31f901b1cb4aad542b992d11321d1dcc54a5f4`,
  passed score/objective/Adapter-input-gradient parity at `atol=rtol=1e-6`,
  including mixed-length right padding. The focused trainer/telemetry suite
  passed `46` tests. Structural gates are exact: per rank/update schedule
  `(32,32)`, two calls, 64 cells, four K=4 matrices, no `16x16` cross-block CE,
  no second cell presentation, all losses/gradients finite, and tokenizer
  length unchanged. Performance values are measurements rather than a minimum
  PASS threshold.
- World size, microbatch, accumulation, and global batch: world 2; data batch
  size K=4 remains unchanged; four distinct same-image groups per rank and
  GA=1 produce 16 local/32 global rows and four local/eight global independent
  `4x4` matrices per update. Physical readout packing is two B32 Qwen batches
  per rank/update, or four calls/128 cells globally. Three timed updates contain
  96 global row presentations and 24 global matrix presentations.
- GPUs: only physical 2
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, both NVIDIA B200 and mapped to
  logical 0/1. Preflight immediately before the configuration commit found
  `0` MiB used and `182642` MiB free on each device; `/nvmesv` had about
  `24` TiB free. The launch repeats the clean-device check and exposes no other
  physical GPU.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-19T23:35:55+09:00` to `23:37:10+09:00`, about 76 seconds wall;
  torchrun launcher PID `1193865`. Worker PIDs were not persisted in the
  current runner log and remain a telemetry gap rather than an inferred value.
- Actual GPU-hours and peak scratch use: less than `0.043` aggregate GPU-hour;
  output root uses about `537` MiB. Maximum per-rank train-step allocated/
  reserved CUDA memory was `117,085,050,368`/`126,636,523,520` bytes.
- Command: planned command is
  `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp13_singlepass_cellb32_throughput.toml`.
  The materialized configuration source/canonical hashes are recorded above;
  it binds CELLB32 through its run ID, accepted task decision, and exact runtime
  commit. Target optimizer steps are `3`, and validation/checkpoint cadence are
  both `4`. Current checkpoint schema requires `save_final=true`, so one
  final DCP/export after the timed steps is allowed; no validation or periodic
  checkpoint may run during steps 1-3.
- Outputs: Adapter/metrics/run-log SHA256
  `339d3ad2f0378edea8c1b9db4685de64a83a449bc9bfcf391fb6b89e7d50a7f6`/
  `321acd2f9c453cab00f3f33c51998f1d733c46552cc2f76ea4c9bcd6e9bba776`/
  `0ddc1c1740985a1d80e892aa18dec178b05fccbb7115593ee2d3e8f1a0d32647`;
  final artifact-manifest SHA256
  `1ebbdb2b70050f9cef7ab5fcd2b2bbcb9c92aee2a4cebfedfd8d0161a83e4a70`;
  required final step-3 DCP metadata SHA256
  `90a5c228d115296c915b79b9bffa74e74801c58589901765e25d66c601ffdb3c`.
  Metrics carry nested schema
  `representation-qwen-physical-execution-v1` with
  `forward_batch_sizes_by_rank`, per-rank/global call and cell counts, and
  maximum physical batch.
- Scorer/parser identity: no answer scorer. The strict native pipeline,
  objective, DCP, and export are bound by the representation-training tree
  identity above; single-pass execution and telemetry have their separate
  SHA256 values above. Qwen3 family adapter SHA256 is
  `9993156852dd5d204e399dde00120aa102a3ca39dec541534313e27be9054c9f`.
- Metrics: `PASS`. Steps 1/2/3 took `15.969847741`/`15.678297025`/
  `16.195537078` seconds and produced `2.003776149`/`2.041037999`/
  `1.975852968` global rows/s. The step-2/3 steady means are
  `15.936917052` seconds and `2.008445484` rows/s. Every step reported 32
  rows/eight matrices, 16 sample IDs per rank, per-rank schedule `(32,32)`,
  64 cells/two calls per rank, and 128 cells/four calls globally. No validation
  event occurred. All losses/gradients were finite and tokenizer length stayed
  `151669`; run identity is
  `efd2a7dbe30fb0239f07ddf82302374213a1555d018052d75f17ef858d9576dd`.
- Conclusion: `PASS` for the bounded CELLB32 geometry and parity question, but
  not as a material throughput win. Against RP-12, steady step time improved
  only `1.70%` and row throughput `1.76%` (`1.017x`) while peak allocation rose
  from about 31.0 GB to 117.1 GB. Linear extrapolation at mathematical global
  batch 32 remains about `8.85` hours for 2,000 updates on two GPUs. The pinned
  historical global-batch-32 run used eight GPUs with one local K4 group per
  rank; RP-13 uses four local K4 groups per rank on two GPUs. Thus physical
  batching cannot erase the fourfold per-rank cell workload. This result does
  not promote an Adapter or close production prompt/data/quality gates.

### RP-14-QWEN3-REPRESENTATION-REAL-RESOLUTION-MAX-PIXELS-AB

- Cell and status: `RP-14`, `COMPLETE`; bounded diagnostic comparison accepted
  by `RPI-20260720-GOLDEN-IMAGE-CAP-AB`. It does not promote either output as a
  representation artifact.
- Question: does the Golden area cap `max_pixels=262144` materially change the
  real-resolution representation execution relative to the pinned processor's
  native `16777216` maximum?
- Runtime/config identity: implementation commit
  `81e4fd7a2b20f29a1620c2ef3f9df1121c45a69b`; launch commit
  `95b3b61600b1296e298bb7dd271e6913f85f1e9f`; cap/default TOML SHA256
  `536c8f7ad28d28ff7edb35ff4cbf305b9e574eee7fddfd8b2b8faf5a9584fd31`/
  `008a0e193ec4364dbf515ce39dc85e7a7ba73a72ec7ab8b358c414d27415d992`.
  Both lanes used target-token embedding, fresh seed `20260719`, K4/GPR1/GA1,
  two B200 ranks on physical GPUs 2 and 3, BF16/SDPA, and one requested update.
- Fixed diagnostic data: JSONL SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`.
  Rank inputs were a `1770x1138` image with SHA256
  `472bee888ff92c9f27a60759c2b422d46dc38d30781473f78b5c88bb77202edc`
  and a `1024x685` image with SHA256
  `49fbc0500a5f172a67c79ab09c998f849dabde9a90eb59ccf13b8617c2535bc1`.
- Processor geometry: native/capped grids were respectively
  `(1,72,110)/(1,24,38)` and `(1,42,64)/(1,26,38)`. Main-D token counts fell
  from `1980` to `228` and from `672` to `247`. Resize preserved aspect ratio;
  no square crop/resize was used.
- Capped result: `PASS`. The update completed in `9.057971587` seconds for
  eight rows, with physical schedules `(8,8)` on both ranks. Peak allocated
  CUDA memory was `61,723,877,888`/`63,822,406,656` bytes and peak reserved was
  `66,355,986,432`/`69,229,084,672` bytes. Metrics SHA256 is
  `e677be8c93fef323fee9accb502ff22d0018c38c376674f168c6e87913a9eca4`.
- Native-limit result: no optimizer-step metric was produced after about 106
  seconds beyond the durable start event, so the lane was terminated rather
  than spending more diagnostic GPU time. One-second telemetry observed up to
  `182274`/`122158` MiB framebuffer use. During the long imbalanced interval,
  one rank commonly reported `0%` SM while the other reported `100%`, showing
  rank straggling rather than balanced full-device use. This is a terminated
  lower-bound observation, not a completed timing number and not an OOM claim.
- Conclusion: the missing Golden cap is a material root cause for real-data
  memory and speed behavior. It does not explain the RP-12 versus RP-13
  31-GB/117-GB difference because their `8x4` fixture remains below both caps;
  that difference remains attributable to score/recompute versus single-pass
  B32 execution. Whether `262144` becomes the production cap remains open.

### SC-30-T211-FSDP2-INFRA-20260720

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-30-T211-FSDP2-INFRA-20260720`, exactly matching the materialized config
  `run_id`;
  mandatory Torch-2.11 two-rank composable-FSDP2 checkpoint/resume gate.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime commit
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; accepted re-spike in
  `PROJECT_TASK.md` §9.2 and I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: does the candidate stack execute a two-rank FSDP2 update and then
  reproduce the next output, loss, update, and local shards bitwise after a
  strict DCP teardown/reconstruct/resume?
- Baseline and exact output path: control `SC-30-FSDP2-INFRA`; result/log/DCP
  are `artifacts/compatibility/SC-30-T211-fsdp2-infra.json`,
  `artifacts/compatibility/SC-30-T211-fsdp2-infra.log`, and
  `artifacts/compatibility/SC-30-T211-fsdp2-infra-checkpoint/`.
- Model and processor identity: `synthetic-tiny-fsdp2-model-v1`, width 16, two
  residual MLP blocks, FP32, seed `20260719`; processor N/A.
- Representation checkpoint identity: N/A; this is synthetic infrastructure.
- N/A fields and justification: prompt, tokenizer, D/DeepStack, policy,
  reference, reward, sampling, behavior logprobs, GRPO, and SDPO are absent.
- Policy/reference initialization: N/A; deterministic tiny-model control and
  resumed lanes start from identical state.
- Rollout policy version and allowed asynchronous staleness: N/A; no rollout.
- Code commit and worktree state: runtime `2918c8913756e4bbac0e6aa171c102ceab4d409c`;
  clean launch HEAD `74551ba963dae47416fc6f07596789c1956e5f2e`.
- Repository adapter/patch surface and hash: `fsdp2_smoke.py` SHA256
  `a386e70f84417784ed2d3aa731578a488f873e2469d1dfec813d7a8f88065694`;
  config SHA256
  `526baee91aa2b19f47ea7c3f1342ff3c366b45e3fb90a24e55c6e744fa0e43f7`;
  public PyTorch composable FSDP2/DCP and veRL checkpoint imports, no patch.
- Dataset/manifest, hashes, sample rule, and n: generated two-step tensors with
  fixture seeds `20260720` and `20260721`; no external dataset.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: CPython `3.12.3`, Torch
  `2.11.0+cu129`, CUDA `12.9`, NCCL `2.28.9`, Transformers `4.57.6`, vLLM
  `0.23.0+cu129`, veRL `638b8ff84f279e054982f1f4633a546f3c6ced68`,
  TransferQueue `0.1.8`; candidate lock SHA256
  `b62e6f81e0f33ebc71468f81d94e427f5c7d954aab6223b151f48f339a5d60e6`.
- Objective equations and normalization: test-only global-element-mean
  `mean((model(x)-target)**2)`, AdamW lr `1e-3`, weight decay `0`,
  `foreach=false`, `fused=false`; no production RL interpretation.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  algorithms, TF32 off, no dropout, fixed seeds, CUBLAS `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: FP32, no KV/quantization/attention/TP;
  FSDP mesh `[2]`, reshard after forward, no offload.
- Logit/logprob/loss/gradient parity tolerances: output, scalar loss, and local
  updated shards after resume must be exact (`atol=rtol=0`).
- World size, microbatch, accumulation, and global batch: world 2; local tensor
  `[2,3,16]`; one update per step; accumulation 1.
- GPUs: physical 2 `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, B200 183359 MiB, driver
  `570.195.03`, mapped to logical 0/1. Preflight at
  `2026-07-20T03:05:40+09:00` found 0 MiB used on each.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:09:06.794+09:00` to `03:09:20.765+09:00`, about 14 seconds;
  torchrun launcher PID `1566529`, ranks 0/1 in the result.
- Actual GPU-hours and peak scratch use: less than `0.008` aggregate GPU-hour;
  DCP size `227277` bytes and devices returned to 0 MiB.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 600s .venv-torch211-cu129/bin/torchrun --standalone --nproc-per-node=2 spikes/verl_compat/fsdp2_smoke.py --stack torch211-cu129 --config configs/smoke/fsdp2_torch211.toml --checkpoint-dir artifacts/compatibility/SC-30-T211-fsdp2-infra-checkpoint --output artifacts/compatibility/SC-30-T211-fsdp2-infra.json > artifacts/compatibility/SC-30-T211-fsdp2-infra.log 2>&1`.
- Outputs: result/log SHA256
  `d37e9b968f0da629320d476b9caec864bac962413d67b40c44d4155221680d3d`/
  `20cc1bee642d445de51fc3441c68a0d7a5ee6a58365773dea57ad157efc4ab21`;
  strict DCP path above. The exact `run_id` resides in the validated TOML; the
  post-run script now additionally echoes it in future result payloads.
- Scorer/parser identity: exact script/config hashes above.
- Metrics: both ranks reported loss `1.5465292930603027`,
  `resume_exact=true`, and candidate identities Torch `2.11.0+cu129`, vLLM
  `0.23.0+cu129`, veRL commit `638b8ff...`; updated local-shard SHA256 values
  are `1a3bd2333543ba5ffaebb6670658167272803fcf66f0e17a449bdbe4ad457c95`
  and `154b8051d5d2b882039828e2a19725ce5da44445dfb5c1ebe96f27e1aad5fbd2`.
- Conclusion: `PASS` for candidate two-rank FSDP2 strict checkpoint/resume;
  this synthetic result alone does not promote the stack.

### SC-21-T211-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-VERL-VLLM-WEIGHT-SYNC`; mandatory upstream veRL FSDP2 actor to
  vLLM TP=2 generation/weight-sync gate.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: can upstream veRL V1 synchronously run TransferQueue generation,
  two nonzero FSDP2 actor updates, naive ZMQ/CUDA-IPC weight transfer to vLLM,
  and replay processed logprobs at staleness zero?
- Baseline and exact output path: no earlier combined gate; result, log,
  metrics, fixture, resolved config, and plan use the prefix
  `artifacts/compatibility/SC-21-T211-verl-vllm-weight-sync`.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, Qwen3-VL 8B Thinking,
  model config SHA256
  `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`,
  tokenizer length `151669`, BF16, local-only, no tokenizer resize.
- Representation checkpoint identity: N/A; repo TGVF plugin is deliberately
  disabled in this infrastructure gate.
- N/A fields and justification: production reward, reference-policy replay,
  main-D/D-DeepStack transport, GRPO, and SDPO are excluded; their mathematics
  are not authorized by this spike.
- Policy/reference initialization: actor from the original local Qwen3; no
  reference policy; vLLM uses dummy load followed by actor weight sync.
- Rollout policy version and allowed asynchronous staleness: version 0 then 1;
  synchronous trainer, required staleness 0 and no intervening update.
- Code commit and worktree state: runtime commit above; clean launch HEAD
  `74551ba963dae47416fc6f07596789c1956e5f2e`.
- Repository adapter/patch surface and hash: driver
  `c472232d0cba7b1a4e8a2f5808da1191797838fbcdfc188b14759a380aa0ab07`,
  manager hook `63cfacbc57b3cb352e49dafa187153703bec3858ac279ecf7dfb903eec335a81`,
  objective `f5f8bea96bb6226e8ae371b09ae30d0145d9c68f02bd8517875240e2de8d4fc3`,
  reward `f2da3ff00bf7d089bd8bb71ba642a4873f0a0638141c6c2f99e579c6386423e0`;
  no veRL/vLLM site-package patch.
- Dataset/manifest, hashes, sample rule, and n: generated two-row Parquet;
  logical fixture SHA256
  `6be68e092dc4576079fb5b3ae7182a4b22ebe9d7e9eb48f6b735a2a7acdfa22e`,
  deterministic order, train/validation n=2.
- Native prompt/tool schema hash: no tool invocation; fixed textual sync
  fixture from the driver hash above.
- Chat-template/token-fixture hash and token-ownership masks: Qwen native
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  the two 20-token prompt-ID SHA256 values are
  `e68c49613b090501650cfe227b8fd7b4b7dca657f96e0e645724e951bbb84888`
  and `27ffa098b0a0a0caba4d389f5901493d08137891d3e3f509b01299ccbc48e084`;
  generated response tokens alone form the update mask.
- D/DeepStack/position/mask identity: N/A by deliberate exclusion.
- Observation materialization/artifact identity used by all replays: each
  generated response and processed rollout logprob returns through
  TransferQueue to the same synchronous update; no observation recomputation.
- RL framework/version/environment lock: exact candidate stack and lock SHA256
  identical to `SC-30-T211-FSDP2-INFRA-20260720`.
- Objective equations and normalization: infrastructure-only registered
  zero-advantage estimator plus mean generated-token NLL; fixed zero reward;
  production GRPO/PPO/SDPO math is explicitly excluded.
- Rollout/replay forward mode and adapter dropout/RNG contract: full
  determinism, LoRA/dropout 0, sync V1 trainer, no cache sleep, seed `20260720`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.23.0+cu129`, seed
  `20260720`, materialized greedy sampling (`do_sample=false`, temperature 0,
  top-p 1, top-k -1, min-p 0, repetition 1, presence/frequency 0), 16-token
  cap, no custom processors, `processed_logprobs` after transforms.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, no quantization, Triton
  attention, vLLM TP=2, actor FSDP2 size 2, naive checkpoint backend,
  `free_cache_engine=false`, `enable_sleep_mode=false`.
- Logit/logprob/loss/gradient parity tolerances: both update gradients finite
  and nonzero; step two actor version 1/staleness 0; rollout/replay probability
  differences use maximum floor `.05`, mean floor `.01`, relative multiplier
  `5`, gross max `.2`, gross mean `.02`, Pearson minimum `.9`.
- World size, microbatch, accumulation, and global batch: two GPUs; train batch
  2, PPO mini-batch 2, micro-batch 1/GPU, one epoch/update, two updates.
- GPUs: same physical UUIDs/mapping/preflight as
  `SC-30-T211-FSDP2-INFRA-20260720`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:09:45.279+09:00` to `03:10:37.934+09:00`, about 53 seconds;
  TaskRunner PID `1580768`, worker PIDs `1582339/1582340`.
- Actual GPU-hours and peak scratch use: less than `0.030` aggregate GPU-hour;
  failure preceded model loading and devices returned to 0 MiB.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: result/log/plan/resolved-config/fixture SHA256 respectively
  `3dabdec8a576bae62255a6175dba7ef5b5cf30cebfa63ff406932b6470970901`,
  `096703f86b4b54a4793c403387966b7aa2b3e1f6ef807433fdffe45a07fd8fda`,
  `0fca490884c6f9eb5ac9700c7ebe3057b1c3ea0210d15c63d4255f3308f85a86`,
  `c0c7f8cd9ba6348ac5bcbd3b2edcf008c5351fcc6cb7574890217b48f7dc65a9`,
  and `0f87bf59669c566770d025c5b76e9f04371720c105cbf8d8e0bc2ed9f74ecdb1`;
  no metrics file was produced.
- Scorer/parser identity: driver and registered objective hashes above; public
  manager must be identical to upstream `AgentLoopManagerTQ`.
- Metrics: runtime, model, pip, Hydra and veRL config checks passed; no actor
  update or weight sync occurred.
- Conclusion: `FAIL` before model initialization. The driver incorrectly set
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`; veRL then treated physical
  accelerator IDs 2/3 as local ordinals inside a two-visible-device process
  and raised `CUDA error: invalid device ordinal`. R1 removes that setting,
  lets Ray assign logical ordinals, and adds explicit artifact `run_id` binding.

### SC-21-T211-R1-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-R1-VERL-VLLM-WEIGHT-SYNC`; mandatory corrective rerun of failed
  `SC-21-T211-VERL-VLLM-WEIGHT-SYNC`, with an explicit artifact-bound run ID.
- Spike-plan git revision and VA0/VA1/VA2 approval references: corrected
  runtime commit `a01c4b8caadef4c5d4afe72ec2a6477983a338eb`;
  `PROJECT_TASK.md` §9.2 and I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: unchanged from failed SC-21, now allowing Ray to assign logical
  per-actor CUDA ordinals while preserving exclusive physical-GPU 2/3 scope.
- Baseline and exact output path: failed SC-21; result/log/metrics/fixture/
  resolved-config/plan use the exact prefix
  `artifacts/compatibility/SC-21-T211-R1-verl-vllm-weight-sync`.
- Model and processor identity: identical stable local Qwen3 identity to SC-21.
- Representation checkpoint identity: N/A; TGVF plugin remains disabled.
- N/A fields and justification: identical production-math, reference replay,
  D/DeepStack, GRPO and SDPO exclusions to SC-21.
- Policy/reference initialization: original local Qwen3 actor, no reference;
  dummy vLLM load followed by actor weight sync.
- Rollout policy version and allowed asynchronous staleness: versions `[0,1]`,
  synchronous staleness exactly zero.
- Code commit and worktree state: corrected runtime commit above; clean launch
  HEAD `a5e75e3e7dcd335b756f67caf730cd892dd0c739`.
- Repository adapter/patch surface and hash: driver
  `03fd629449fd1b41c11a68956c3c7455e0b5f9ccd6bf3c1cfabdb4bb5e4b764d`,
  manager/objective/reward SHA256
  `63cfacbc57b3cb352e49dafa187153703bec3858ac279ecf7dfb903eec335a81`/
  `f5f8bea96bb6226e8ae371b09ae30d0145d9c68f02bd8517875240e2de8d4fc3`/
  `f2da3ff00bf7d089bd8bb71ba642a4873f0a0638141c6c2f99e579c6386423e0`;
  no external patch. The child environment explicitly removes
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
- Dataset/manifest, hashes, sample rule, and n: explicit-run-ID-bound two-row
  fixture logical SHA256
  `13efe47ba71789b5266087f34f486e75d2183f30e915193622bc56a6436ffa6a`,
  deterministic order, n=2 for train/validation.
- Native prompt/tool schema hash: no tool call; fixed sync-gate text.
- Chat-template/token-fixture hash and token-ownership masks: native template
  and exact two prompt-ID hashes are identical to corrected SC-21; generated
  tokens alone form the update mask.
- D/DeepStack/position/mask identity: N/A by deliberate exclusion.
- Observation materialization/artifact identity used by all replays: response,
  behavior processed logprobs, and top-level/row `run_id` travel through the
  same TransferQueue record to synchronous replay; no recomputation.
- RL framework/version/environment lock: exact candidate stack/lock identical
  to SC-21; all public probes and `pip check` passed in this environment.
- Objective equations and normalization: registered zero advantage plus
  generated-token-mean NLL with fixed zero reward; infrastructure only.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic,
  dropout zero, synchronous V1, seed `20260720`, no sleep/wake.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: actual greedy parameters and
  `processed_logprobs` exactly as corrected in SC-21.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, unquantized, Triton attention,
  TP2 vLLM and FSDP2 size 2; naive no-sleep weight sync.
- Logit/logprob/loss/gradient parity tolerances: identical explicit thresholds
  to SC-21; two nonzero actor updates, version 1 and staleness 0 are mandatory.
- World size, microbatch, accumulation, and global batch: two GPUs, batch 2,
  mini-batch 2, micro-batch 1/GPU, one epoch and two updates.
- GPUs: physical 2/3 and UUIDs/mapping exactly as SC-21; immediate preflight
  must show both free and no other GPU may be visible.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:27:24.963+09:00` to `03:28:17.875+09:00`, about 53 seconds;
  TaskRunner PID `1612327`, worker PIDs `1613628/1613629`.
- Actual GPU-hours and peak scratch use: less than `0.030` aggregate GPU-hour;
  failure preceded weight loading and devices returned to 0 MiB.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --run-id SC-21-T211-R1-VERL-VLLM-WEIGHT-SYNC --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-R1-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: result/log/plan/resolved-config/fixture SHA256 respectively
  `44044dc47408ca3c709dc10bf0deb8afc4fbbff44f898726e359c7dcda54e734`,
  `5fe480c63b09537a729709873259f5493bdd1b38900f083d5d5fee1cb1878587`,
  `b06d2b06586e3f2cb0097927dc292a924bccc905839735363eeae7efdafa4166`,
  `d0d6f79f00d35651c5700cb975e0c08a02238053a28cc1fdb3a224294897a155`,
  `83e865a32ad30cd1b886cbadf3931084f28741e3556e4b62c2b23828efce6b45`;
  all bind the explicit R1 run ID; no metrics file was produced.
- Scorer/parser identity: driver/manager/objective/reward hashes above.
- Metrics: runtime/model/pip/Hydra/veRL checks and corrected Ray logical GPU
  mapping passed; no update or sync occurred.
- Conclusion: `FAIL` during FSDP2 model construction because veRL defaulted the
  Hugging Face actor to `flash_attention_2`, which is intentionally absent from
  this candidate. R2 explicitly selects the already accepted SDPA actor path;
  it does not add or install FlashAttention.

### SC-21-T211-R2-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-R2-VERL-VLLM-WEIGHT-SYNC`; mandatory corrective rerun after R1
  reached actor construction and exposed veRL's unselected attention default.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime commit
  `dba7801fd0dae8b8d274ba24be65e7ac76b24cb2`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: unchanged combined FSDP2→vLLM sync question, now with the actor's
  Hugging Face attention implementation explicitly fixed to `sdpa`.
- Baseline and exact output path: failed R1; six outputs use exact prefix
  `artifacts/compatibility/SC-21-T211-R2-verl-vllm-weight-sync`.
- Model and processor identity: identical stable local Qwen3 identity to R1;
  actor uses SDPA and rollout remains vLLM Triton attention.
- Representation checkpoint identity: N/A; TGVF plugin remains disabled.
- N/A fields and justification: identical production GRPO/SDPO, reference and
  D/DeepStack exclusions to R1.
- Policy/reference initialization: original Qwen3 actor, no reference; dummy
  vLLM initial load then exact actor-weight synchronization.
- Rollout policy version and allowed asynchronous staleness: `[0,1]`, sync,
  staleness exactly zero.
- Code commit and worktree state: runtime commit above; clean launch HEAD
  `6cf55de991d6e06754f5d2e2aa30d3ce1e372550`.
- Repository adapter/patch surface and hash: driver
  `2b927b67f99a4c8011bba224ff2c6e84ded56deb3343bb8f2264b47beb27d6ee`;
  manager/objective/reward/lock hashes remain exactly those recorded for R1;
  no external or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: explicit-run-ID-bound two-row
  fixture logical SHA256
  `977b3de0d9c30d2a27b6abf33490443de98d006e00323fd666bb8fd163621f51`,
  deterministic n=2 train/validation.
- Native prompt/tool schema hash: no tool call; fixed sync fixture.
- Chat-template/token-fixture hash and token-ownership masks: exact template,
  two prompt hashes and generated-token-only mask from R1.
- D/DeepStack/position/mask identity: N/A by deliberate exclusion.
- Observation materialization/artifact identity used by all replays: exact
  response/behavior-logprob TransferQueue record with explicit R2 run ID; no
  recomputation.
- RL framework/version/environment lock: exact candidate stack and lock from
  R1; public probe, `pip check`, focused tests and Hydra resolution pass.
- Objective equations and normalization: zero advantage plus generated-token-
  mean NLL and fixed zero reward; infrastructure only.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic,
  actor SDPA, dropout zero, synchronous V1, seed `20260720`, no sleep/wake.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: exact R1 greedy processed-logprob
  contract.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, unquantized; actor SDPA, rollout
  Triton attention; TP2 vLLM, FSDP2 size 2, naive no-sleep sync.
- Logit/logprob/loss/gradient parity tolerances: identical explicit R1 gates,
  including nonzero gradients, version 1, staleness 0 and probability floors.
- World size, microbatch, accumulation, and global batch: identical R1 two-GPU
  batch 2 / mini-batch 2 / micro-batch 1 per GPU / two-update geometry.
- GPUs: only physical 2/3 with UUIDs and logical mapping already recorded;
  immediate preflight must be free.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:31:24.970+09:00` to `03:36:40.197+09:00`, about 315 seconds;
  TaskRunner PID `1630788`, workers `1632288/1632289`.
- Actual GPU-hours and peak scratch use: less than `0.176` aggregate GPU-hour;
  live memory reached about 11.3/9.9 GiB before failure and returned to zero.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --run-id SC-21-T211-R2-VERL-VLLM-WEIGHT-SYNC --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-R2-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: result/log/plan/resolved-config/fixture SHA256 respectively
  `fcd9534c495d58342a2c4e9364c5b73acbaf8edb689762223dfed60f399c28cc`,
  `2dceb372adecda2779f875ed4d64b02ad62cef1b68401a6a0b5db767f125b6bc`,
  `32673943cb48364ec75ebbd114b1493177b6aab303a63a2aa35ebc62333054ad`,
  `add3e4339f7cfce721862a95962109ea83ea24ea8d1248b7bcc1e00c7cae86ee`,
  `1ab80939d1fc0f586e2c7430aad7f21da78a302eea468531d04cc9e954f1b43d`;
  no metrics file was produced.
- Scorer/parser identity: driver/manager/objective/reward hashes above.
- Metrics: SDPA Qwen FSDP2 construction reached the 8.77B model and vLLM
  import; no optimizer update or sync occurred.
- Conclusion: `FAIL` when Triton compiled its CUDA helper: host
  `/usr/include/python3.12/Python.h` is absent. R3 provides the already pinned,
  locally extracted Ubuntu Python 3.12 headers used by the accepted control
  vLLM smoke; no system install or package change is made.

### SC-21-T211-R3-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-R3-VERL-VLLM-WEIGHT-SYNC`; mandatory corrective rerun after R2
  reached Triton import and confirmed the missing host header.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime commit
  `619c706d886fb711c714e2c697d36dfebe537387`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: unchanged combined FSDP2→vLLM gate, now with Triton's compiler
  pointed at the pinned local Python 3.12 headers.
- Baseline and exact output path: failed R2; six outputs use exact prefix
  `artifacts/compatibility/SC-21-T211-R3-verl-vllm-weight-sync`.
- Model and processor identity: identical Qwen3, actor SDPA and rollout Triton
  identities to R2.
- Representation checkpoint identity: N/A; no TGVF plugin.
- N/A fields and justification: unchanged R2 exclusions for production math,
  reference replay, D/DeepStack, GRPO and SDPO.
- Policy/reference initialization: original Qwen3 actor, no reference; dummy
  vLLM initialization then exact actor-weight sync.
- Rollout policy version and allowed asynchronous staleness: `[0,1]`, sync,
  staleness zero.
- Code commit and worktree state: runtime commit above; clean launch HEAD
  `cfe5bcd78f6d339f2d4c036b34ab739fdd4f6403`.
- Repository adapter/patch surface and hash: driver
  `ca2be71a4461b2eedada88a918f4efa5f2ba2dda85d32ecc9bc998c451abb2ea`;
  remaining code/lock hashes unchanged from R2. Compiler `/usr/bin/gcc`, C++
  compiler `/usr/bin/g++`, and `CPATH` are explicit. Extracted `Python.h`
  SHA256 is
  `729ef157f6026e6e1b3104593f87dddc597c3b83b60c7c2965878c62a56c6f7d`;
  the pinned Ubuntu dev-package provenance is recorded in
  `EXTERNAL_REFERENCES.md`; no system install or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: explicit-R3-ID two-row logical
  fixture SHA256
  `5d85c96008a38aae9fde4c97d9532582781a35145128e5b06b5bd7ccf257c41d`;
  deterministic n=2 train/validation.
- Native prompt/tool schema hash: no tool call; fixed sync text.
- Chat-template/token-fixture hash and token-ownership masks: unchanged exact
  native template/two prompt hashes/generated-token-only mask.
- D/DeepStack/position/mask identity: N/A by exclusion.
- Observation materialization/artifact identity used by all replays: same
  explicit-run-ID TransferQueue response/behavior-logprob record as R2.
- RL framework/version/environment lock: exact candidate stack and lock;
  focused tests, public probe, `pip check` and Hydra resolve pass.
- Objective equations and normalization: zero advantage plus token-mean NLL,
  fixed zero reward; infrastructure only.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic,
  actor SDPA, dropout zero, sync V1, seed `20260720`, no sleep/wake.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: unchanged greedy processed-logprob
  contract.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, no quantization, actor SDPA,
  rollout Triton, vLLM TP2 and actor FSDP2 size 2, naive no-sleep sync.
- Logit/logprob/loss/gradient parity tolerances: unchanged explicit R2 gates.
- World size, microbatch, accumulation, and global batch: unchanged two-GPU,
  batch-2, two-update R2 geometry.
- GPUs: only physical 2/3 and their recorded UUIDs; immediate free preflight.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:40:19.182+09:00` to `03:46:26.825+09:00`, about 368 seconds;
  TaskRunner `1659194`, actor workers `1660503/1660504`, vLLM server/core/TP
  workers `1670612/1671589/1671734/1671735`.
- Actual GPU-hours and peak scratch use: less than `0.205` aggregate GPU-hour;
  all devices returned to zero after failure.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --run-id SC-21-T211-R3-VERL-VLLM-WEIGHT-SYNC --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-R3-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: result/log/plan/resolved-config/fixture SHA256 respectively
  `3f066b8824bc07ab637f3de411914e4c62783c6e2509f3136874200fa672e3f5`,
  `2c31e44b1ab85e658bb41d447f20a5a0ec72cffa3c4cca7f93c79224f8091894`,
  `0cb71492f69b65364066c94b4ebea7afccf7d829bfdc4840a76275a9af9d38aa`,
  `d820543729f3aa39a491fc060df22befb099b6c6d9389d701120c71e6baf0f85`,
  `b079d5f62b11571744a759dcb85efb3c9bdf7cf8971ee12127abf95fafd54a1a`;
  no update metrics were produced.
- Scorer/parser identity: exact driver/manager/objective/reward hashes above.
- Metrics: FSDP2 actor, reward manager and vLLM server all initialized; failure
  occurred during vLLM multimodal profile before generation or weight sync.
- Conclusion: `FAIL`: vLLM chose its bundled FA2 for visual-encoder profiling,
  whose CUDA-12.9 PTX is unsupported by host driver 570.195.03 (reported CUDA
  12.8). R4 explicitly selects vLLM's public `TORCH_SDPA` multimodal encoder
  backend, matching the already accepted latent-smoke configuration; no driver
  or FlashAttention change is authorized.

### SC-21-T211-R4-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-R4-VERL-VLLM-WEIGHT-SYNC`; final mandatory corrective rerun after
  R3 exposed the unselected vLLM multimodal-attention default.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime commit
  `3fc90e9de54d91903f86d7a2b1eea95dfce5cf63`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: unchanged combined FSDP2→vLLM gate with both actor attention and
  vLLM visual-encoder profiling fixed to public Torch SDPA paths.
- Baseline and exact output path: failed R3; six outputs use prefix
  `artifacts/compatibility/SC-21-T211-R4-verl-vllm-weight-sync`.
- Model and processor identity: unchanged Qwen3; actor SDPA and vLLM
  multimodal encoder `TORCH_SDPA`; decoder Triton was requested but the runtime
  auto-selected bundled FlashAttention 2.
- Representation checkpoint identity: N/A; TGVF plugin remains disabled.
- N/A fields and justification: unchanged R3 exclusions.
- Policy/reference initialization: original Qwen3 actor, no reference; dummy
  rollout load then exact actor-weight sync.
- Rollout policy version and allowed asynchronous staleness: `[0,1]`, sync,
  staleness zero.
- Code commit and worktree state: clean launch commit
  `ab8e43576f2e2e79fd5cdaa76a0a712f5977ca87`; the runtime correction remains
  commit `3fc90e9de54d91903f86d7a2b1eea95dfce5cf63`.
- Repository adapter/patch surface and hash: driver
  `c56978dee7ab41a79adddece25bc2358d243efee912c4488e95591075c21ce5e`;
  other code/lock/header hashes unchanged from R3; no external patch.
- Dataset/manifest, hashes, sample rule, and n: explicit-R4-ID logical fixture
  SHA256
  `addeafd9e5d2f87cf39beb36c174c3c4563e84293599f149bbc23fbddbe4352e`,
  deterministic two rows.
- Native prompt/tool schema hash: no tool call; fixed sync fixture.
- Chat-template/token-fixture hash and token-ownership masks: unchanged R3
  exact template/prompt hashes and generated-token-only mask.
- D/DeepStack/position/mask identity: N/A by exclusion.
- Observation materialization/artifact identity used by all replays: exact R4
  TransferQueue response/behavior-logprob record, no recomputation.
- RL framework/version/environment lock: exact candidate stack/lock/header
  identity; focused tests, public probe, pip and Hydra checks pass.
- Objective equations and normalization: zero advantage plus generated-token-
  mean NLL, fixed zero reward; infrastructure only.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic,
  dropout zero, sync V1, seed `20260720`, no sleep/wake.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: unchanged greedy processed-logprob
  contract.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, no quantization; actor and
  multimodal encoder Torch SDPA, actual vLLM decoder bundled FA2 despite the
  requested Triton environment key; TP2/FSDP2 size 2, naive no-sleep sync.
- Logit/logprob/loss/gradient parity tolerances: unchanged explicit R3 gates.
- World size, microbatch, accumulation, and global batch: unchanged R3 two-GPU
  batch-2/two-update geometry.
- GPUs: physical 2/3 only, exact UUIDs above; immediate free preflight.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T03:48:52+09:00` / `2026-07-20T03:54:51+09:00`, 359 seconds;
  TaskRunner PID 1689171, TransferQueue controller/storage PIDs 1689930/1690079,
  FSDP workers 1690465/1690466, vLLM server/core PIDs 1699390/1699910, TP workers
  1700141/1700142, and agent-loop PID 1701506.
- Actual GPU-hours and peak scratch use: less than `0.200` two-device GPU-hours
  by wall-time upper bound; the six output artifacts total 328,342 bytes, no
  checkpoint was produced, and no trustworthy peak-memory sample was recorded.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --run-id SC-21-T211-R4-VERL-VLLM-WEIGHT-SYNC --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-R4-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: all six explicit-run-ID artifacts materialized. SHA256 values:
  result `abb82637f084b0491d315332749c0aa35d73c193a1742006e6f443518f214ea1`,
  log `5e10c57bd4522c8e020f338586d4c0ec93c58d270361140f32d6cca14c0024d4`,
  plan `4769b3ee74e8b28dc3fd2a6d358dd67a79ce571da1356266b6a236bf4825a14c`,
  resolved config
  `5b7c78f9da58e773e49db2867c5cc622eb3ea4d2f7f4c43bd13fbe91fe89230a`,
  parquet fixture
  `fcdde66a98ed8119d58ec7e837001483745893a9ab81b445e57cf4d2b1cb573c`,
  and empty metrics JSONL
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Scorer/parser identity: exact driver/manager/objective/reward hashes above.
- Metrics: the actor FSDP2 workers loaded the 8.77B model, the reward manager
  initialized, the vLLM TP2 server reached the first generation request, and
  the explicit multimodal `TORCH_SDPA` path avoided the earlier visual-kernel
  failure. vLLM `0.23.0+cu129` warned that `VLLM_ATTENTION_BACKEND` is unknown,
  selected `vllm/v1/attention/backends/flash_attn.py` for decoder attention,
  and its bundled `_vllm_fa2_C.varlen_fwd` raised
  `cudaErrorUnsupportedPtxVersion` on driver `570.195.03` (reported CUDA 12.8).
  No rollout item, weight update, behavior logprob, or replay metric was
  produced; the later zero-item partition assertion is downstream fallout.
- Conclusion: `FAIL` under the mandatory combined gate. The exact Torch
  `2.11.0+cu129` / vLLM `0.23.0+cu129` candidate is rejected on this host. No
  R5 or further attention-backend patch ladder is authorized; FlashAttention 2
  and FlashAttention 4 remain outside the production dependency set pending a
  separately approved compatibility spike.

### SC-20-T211-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-20-T211-QWEN3-VLLM-LATENT`; mandatory candidate-stack real-Qwen3 native
  repeated-tool/precomputed-latent vLLM smoke.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `a01c4b8caadef4c5d4afe72ec2a6477983a338eb`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `CANCELLED`.
- Result: `BLOCKED_NOT_RUN`.
- Question: does vLLM `0.23.0+cu129` load the public repo plugin and generate
  from the exact native two-call transcript with source plus two main-D/three-
  branch DeepStack items, without tokenizer growth?
- Baseline and exact output path: control `SC-20-R6`; result/log are
  `artifacts/compatibility/SC-20-T211-qwen3-vllm-latent.json` and `.log`.
- Model and processor identity: same stable local Qwen3 path, config/hash,
  tokenizer length and native template identity as `SC-21`; architecture
  override `TGVFQwen3VLForConditionalGeneration`.
- Representation checkpoint identity: deterministic synthetic latent seed
  `20260719`; no trained TGVF Adapter artifact claim.
- N/A fields and justification: optimizer, reference, reward, GRPO, SDPO,
  checkpoint/resume and task scorer are absent in this inference-only cell.
- Policy/reference initialization: original Qwen3 weights; no reference.
- Rollout policy version and allowed asynchronous staleness: immutable base
  model, version 0, staleness 0.
- Code commit and worktree state: runtime commit above; clean planned-launch
  descendant required.
- Repository adapter/patch surface and hash: plugin
  `964f551733d1ebdf96b53ed1cb1277c7d017bc755f1509438b99c2bde3ac3444`,
  registration
  `ed42f20520ffbd029139d393b7a96122399ce2144414d3b467018864a8d9ec23`,
  driver `e7c8dec94258ca62d174afaaaed10407a89cd35b3a6db0cd58787a66dc3696f8`;
  no site-package patch.
- Dataset/manifest, hashes, sample rule, and n: one synthetic request, no
  dataset; three deterministic BF16 latent rows from seed `20260719`.
- Native prompt/tool schema hash: native `tgvf_focus_tool` schema
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`;
  exactly two native calls and responses.
- Chat-template/token-fixture hash and token-ownership masks: template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  transcript text/token-ID SHA256
  `c3f7e889eca24efa97ac605cb875ca88858b4c6b827bec4c59c411cbde7e091b`/
  `d0dc12f15978045403a5a664e4e3eb5ce9f711a1638efad3a8886922f4525294`,
  exactly 295 prompt tokens; generation prefill asserted, no RL loss mask.
- D/DeepStack/position/mask identity: source/call0/call1, each grid `(1,2,2)`,
  one merged row, width `4096*(1+3)=16384`, branches `(8,16,24)`, native
  processor M-RoPE positions; `[3,16384]` aggregate BF16 latent SHA256
  `3f7711522f4bd3c10530b765a186b109003727934459ed388f35e60863d26cda`.
- Observation materialization/artifact identity used by all replays: one
  pre-materialized immutable public-input list; no policy/reference replay.
- RL framework/version/environment lock: exact candidate stack/lock identical
  to `SC-30-T211-FSDP2-INFRA-20260720`; the driver independently fail-closes Python,
  Torch distribution/runtime, Transformers, vLLM and veRL direct-URL commit.
- Objective equations and normalization: N/A; inference only.
- Rollout/replay forward mode and adapter dropout/RNG contract: eager vLLM,
  prefix cache off, multimodal processor cache zero, no TGVF recomputation.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.23.0+cu129`, seed
  `20260719`, temperature `.7`, top-p `.9`, top-k `20`, min-p `.01`,
  repetition `1.05`, presence `.1`, frequency `.05`, no custom processor, two
  tokens, sampled `processed_logprobs` after transforms.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights/KV, no quantization,
  `TRITON_ATTN`, TP=2, max length 512, eager; no training mesh.
- Logit/logprob/loss/gradient parity tolerances: structure exact; two sampled
  logprobs finite and non-positive; tokenizer stays `151669`.
- World size, microbatch, accumulation, and global batch: TP world 2, one
  request/completion, no accumulation.
- GPUs: same physical UUIDs/mapping/preflight as
  `SC-30-T211-FSDP2-INFRA-20260720`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: not
  launched; no process identity.
- Actual GPU-hours and peak scratch use: zero; no outputs materialized.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv-torch211-cu129/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --run-id SC-20-T211-QWEN3-VLLM-LATENT --stack torch211-cu129 --output artifacts/compatibility/SC-20-T211-qwen3-vllm-latent.json > artifacts/compatibility/SC-20-T211-qwen3-vllm-latent.log 2>&1`.
- Outputs: new result/log paths above; overwrite forbidden.
- Scorer/parser identity: driver, public plugin and native renderer at runtime
  commit and hashes above.
- Metrics: not collected.
- Conclusion: `BLOCKED_NOT_RUN`; the prerequisite combined veRL/FSDP2/vLLM
  R4 gate rejected this exact candidate before this cell was launched.

### RP-15P-T211-QWEN3-PATCH-EMBED

- Cell/matrix ID and mandatory/diagnostic class:
  `RP-15P-T211-QWEN3-PATCH-EMBED`; mandatory candidate patch-projection parity
  and bounded timing diagnostic on one authorized GPU.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `a01c4b8caadef4c5d4afe72ec2a6477983a338eb`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `CANCELLED`.
- Result: `BLOCKED_NOT_RUN`.
- Question: is the mathematically equivalent Linear projection numerically
  compatible with native Conv3D in FP32/BF16, and what is its bounded speedup
  on a fixed 512-by-512-equivalent patch tensor?
- Baseline and exact output path: native Conv3D in the same process; result
  `artifacts/compatibility/RP-15P-T211-qwen3-patch-embed.json`.
- Model and processor identity: accepted local Qwen3; config SHA256
  `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`,
  model index SHA256
  `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`;
  only patch weight/bias from `model-00004-of-00004.safetensors` are loaded;
  their storage SHA256 values are
  `f0b1dc6e1ebb10ba30eeea53044528dbe0cbbbc33bb3470d7d813b753350a3fa`/
  `a0602c5930df091e25f5951ebc53b94ef63c24aaacf22477d8df3520975d8b67`.
- Representation checkpoint identity: N/A; frozen base patch projection only.
- N/A fields and justification: data, prompt, tokenizer serialization,
  D/DeepStack, policy/reference, reward, sampling, optimizer, replay, GRPO and
  SDPO are absent.
- Policy/reference initialization: N/A.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: runtime commit above; clean planned-launch
  descendant required.
- Repository adapter/patch surface and hash: probe SHA256
  `204997e85b2cc3fc681b129e88ffa2be13e180324ad49ecf4df9312ce53572d9`;
  no model/package patch.
- Dataset/manifest, hashes, sample rule, and n: generated seed `20260720`,
  flattened shape `(1024,1536)` corresponding to grid `(1,32,32)`, FP32 input
  storage SHA256
  `d5f60ced7e11c0d9a31ca9ea7e53eb9fb0fc462243cfa4b75fb5d9a93b0b34b0`.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: candidate Torch/CUDA identity and
  lock SHA256 identical to `SC-30-T211-FSDP2-INFRA-20260720`.
- Objective equations and normalization: N/A; forward parity/timing only.
- Rollout/replay forward mode and adapter dropout/RNG contract: eval/no-grad,
  deterministic input; 3 warmups and 10 synchronized timed iterations/method.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: native FP32/BF16 weights/input, no
  quantization/KV/attention/TP/mesh.
- Logit/logprob/loss/gradient parity tolerances: FP32 `atol=rtol=1e-4`; BF16
  `atol=rtol=0.015625`; timing has no PASS floor.
- World size, microbatch, accumulation, and global batch: one process/GPU;
  1024 patches per call; no batch accumulation.
- GPUs: only physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, B200 183359 MiB, driver
  `570.195.03`; preflight found 0 MiB used.
- Start/end timestamps, elapsed time, and session/process identity: not
  launched; no process identity.
- Actual GPU-hours and peak scratch use: zero; no outputs materialized.
- Command: `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 timeout 600s .venv-torch211-cu129/bin/python spikes/verl_compat/qwen3_patch_embed_probe.py --run-id RP-15P-T211-QWEN3-PATCH-EMBED --runtime candidate --physical-gpu 3 --output artifacts/compatibility/RP-15P-T211-qwen3-patch-embed.json > artifacts/compatibility/RP-15P-T211-qwen3-patch-embed.log 2>&1`.
- Outputs: new explicit-run-ID JSON and log above; overwrite forbidden.
- Scorer/parser identity: exact probe hash above.
- Metrics: not collected.
- Conclusion: `BLOCKED_NOT_RUN`; the prerequisite combined veRL/FSDP2/vLLM
  R4 gate rejected this exact candidate. The Linear fast path remains an
  unpromoted option and was not tested under the rejected runtime.

### RP-15-QWEN3-REPRESENTATION-TORCH211-FSDP2-EMBEDDING-K4-GPR4-GA1-SINGLEPASS-CELLB32-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-15`; mandatory three-step
  real-Qwen representation FSDP2 forward/backward and throughput comparison.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `a01c4b8caadef4c5d4afe72ec2a6477983a338eb`; `PROJECT_TASK.md` §9.2,
  I8H-20260719 and the accepted RP-13 geometry.
- Lifecycle status: `CANCELLED`.
- Result: `BLOCKED_NOT_RUN`.
- Question: with mathematical global batch 32 and identical RP-13 math/data,
  how fast does the candidate stack execute three real Qwen3 TGVF-Adapter
  updates, and do all representation contracts remain valid?
- Baseline and exact output path: control `RP-13`; outputs under
  `artifacts/representation/RP-15-qwen3-representation-torch211-fsdp2-embedding-k4-gpr4-ga1-singlepass-cellb32-throughput/`.
- Model and processor identity: stable local Qwen3, config/template/tokenizer
  identities above, BF16/SDPA/local-only/no resize; `image_max_pixels=null` to
  keep exact RP-13 fixture identity.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  source artifact or legacy checkpoint; only Adapter trainable, Qwen frozen.
- N/A fields and justification: policy/reference rollout, behavior logprob,
  reward, GRPO, SDPO teacher, judge, vLLM and policy replay are absent.
- Policy/reference initialization: N/A; original local Qwen is frozen readout.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation updates only.
- Code commit and worktree state: runtime commit above; clean planned-launch
  descendant required; config binds the runtime commit with `dirty=false`.
- Repository adapter/patch surface and hash: representation-training tree
  `6eeacdb69e7b224caf57796d96eec02aca33453c`, Qwen tree
  `6ae2676f7f08949f2425d736fe4d54751d53f69f`; config source/canonical SHA256
  `7895f5bc7859e237588c79d4a4174eec685c3f9b5a2e7c769309e7a7b49120f7`/
  `df2c53457d81284f4822e2bf4b531cc7c514d7a0016246cbcacb1f7fd1a9a0c3`;
  no site-package patch.
- Dataset/manifest, hashes, sample rule, and n: train/validation JSONL SHA256
  `34053c694023461be9f0fe30fd1e525e27697894f705e5428fafad45cfeabc1c`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`;
  train has 32 rows/eight complete K4 same-image groups, four groups/rank,
  sampler seeds 71/73, required disjoint overlap hash
  `c23fe08408640a95d0fe9c482228908bc35e7816021e51b029cc6c9d6979c218`.
- Native prompt/tool schema hash: smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native tool schema
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`.
- Chat-template/token-fixture hash and token-ownership masks: template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  evidence-description-only labels; visual/template positions `-100`.
- D/DeepStack/position/mask identity: live TGVF Adapter main D plus ordered
  branches `(8,16,24)`, native M-RoPE, post-D original-image key blocking,
  target-token-embedding conditioning.
- Observation materialization/artifact identity used by all replays: each
  candidate observation is materialized once and used by the differentiable
  single pass; four independent K4 CE matrices/rank, no cross-group denominator
  and no detached recompute.
- RL framework/version/environment lock: exact candidate stack/lock identical
  to `SC-30-T211-FSDP2-INFRA-20260720`.
- Objective equations and normalization: unchanged RP-13 global-row-mean K4
  Matrix CE, global per-sample evidence-token-mean `L_gen`, historical Norm;
  weights `1/1/.1`, manifold `0`; AdamW lr `1e-4`, clip 1, historical cosine
  horizon 2000/warmup 100/min ratio .1.
- Rollout/replay forward mode and adapter dropout/RNG contract: frozen Qwen
  eval, Adapter dropout 0, no cache, deterministic algorithms, TF32 off, seed
  `20260719`, CUBLAS `:4096:8`; exactly two physical B32 forward calls/rank/update.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: token sampling N/A; deterministic
  same-image group sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 forward, FP32 reduction, no
  quantization/KV/TP, SDPA, FSDP2 `[2]`, reshard, no offload.
- Logit/logprob/loss/gradient parity tolerances: exact structural counts;
  finite losses/gradients; all Adapter gradients required; tokenizer `151669`;
  focused Matrix-CE single-pass parity tolerance `atol=rtol=1e-6`; performance
  has no minimum threshold.
- World size, microbatch, accumulation, and global batch: world 2; K4 data
  groups, four groups/rank/update, GA1; 16 local/32 global rows and eight global
  K4 matrices/update; three updates.
- GPUs: same physical UUIDs/mapping/preflight as
  `SC-30-T211-FSDP2-INFRA-20260720`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: not
  launched; no torchrun/rank identity.
- Actual GPU-hours and peak scratch use: zero; no output directory or
  checkpoint was materialized.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv-torch211-cu129/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp15_torch211_singlepass_cellb32_throughput.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-15-qwen3-representation-torch211-fsdp2-embedding-k4-gpr4-ga1-singlepass-cellb32-throughput/run.log 2>&1`.
- Outputs: Adapter, metrics, final strict DCP and log under the root above;
  overwrite forbidden; no validation/periodic save within steps 1-3.
- Scorer/parser identity: strict representation parser/runner/objective at
  runtime commit and tree/hash identities above.
- Metrics: not collected.
- Conclusion: `BLOCKED_NOT_RUN`; the mandatory combined veRL/FSDP2/vLLM R4
  gate rejected this exact candidate first. The accepted Torch 2.9 control
  remains authoritative for representation training.

### RP-16P-QWEN3-PATCH-EMBED-CONTROL

- Cell/matrix ID and mandatory/diagnostic class:
  `RP-16P-QWEN3-PATCH-EMBED-CONTROL`; diagnostic bounded control-stack
  patch-projection parity and timing cell.
- Spike-plan git revision and VA0/VA1/VA2 approval references: run-ID-aware
  probe commit `a01c4b8caadef4c5d4afe72ec2a6477983a338eb`; planned ledger
  commit `b14e2acd3a289fd78499dc3ced68aa88159d99fe`; user decision
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION` in `PROJECT_TASK.md` §2.1.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: on the accepted Torch 2.9 control runtime, is the flattened Linear
  expression numerically compatible with the native full-patch Conv3D in FP32
  and BF16, and what bounded synchronized speed ratio is observed?
- Baseline and exact output path: native checkpoint Conv3D in the same process;
  result/log are
  `artifacts/compatibility/RP-16P-control-qwen3-patch-embed.json` and `.log`.
- Model and processor identity: stable local Qwen3; config SHA256
  `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`,
  model index SHA256
  `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`;
  only `model.visual.patch_embed.proj.weight` and `.bias` are read from
  `model-00004-of-00004.safetensors`.
- Representation checkpoint identity: N/A; frozen base patch projection only.
- N/A fields and justification: dataset, prompt, tokenizer serialization,
  target conditioning, Adapter, D/DeepStack, policy/reference, reward,
  sampling, optimizer, replay, GRPO, SDPO and checkpoint/resume are absent.
- Policy/reference initialization: N/A.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: probe SHA256
  `204997e85b2cc3fc681b129e88ffa2be13e180324ad49ecf4df9312ce53572d9`;
  launch requires the clean commit containing this planned entry.
- Repository adapter/patch surface and hash: probe only; no project model code
  or installed package is patched.
- Dataset/manifest, hashes, sample rule, and n: deterministic seed `20260720`,
  one synthetic flattened `(1024,1536)` patch tensor representing grid
  `(1,32,32)` and input SHA256
  `d5f60ced7e11c0d9a31ca9ea7e53eb9fb0fc462243cfa4b75fb5d9a93b0b34b0`.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: accepted control Python `3.12.3`,
  Torch `2.9.0+cu128`/CUDA `12.8`; compatibility lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: native
  `conv3d(x.reshape(N,3,2,16,16), W, b).reshape(N,1152)` versus
  `linear(x, W.reshape(1152,1536), b)`; no training objective.
- Rollout/replay forward mode and adapter dropout/RNG contract: inference mode,
  deterministic algorithms, cuDNN benchmark off/deterministic on, seed
  `20260720`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: checkpoint weight/bias BF16; separate
  FP32 and BF16 projection comparisons; no KV, attention, TP or mesh.
- Logit/logprob/loss/gradient parity tolerances: output allclose FP32
  `atol=rtol=1e-4`, BF16 `atol=rtol=0.015625`; all finite. Timing has no PASS
  floor and cannot by itself authorize a code change.
- World size, microbatch, accumulation, and global batch: one process/GPU,
  1024 patches/call; three warmups plus ten synchronized iterations per method
  and dtype, 56 total projection calls including parity calls.
- GPUs: only physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, NVIDIA B200, driver
  `570.195.03`; the pre-plan check observed 0 MiB used.
- Start/end timestamps, elapsed time, and session/process identity: the launcher
  did not persist exact start/PID/elapsed telemetry; result/log completion mtimes
  are `2026-07-20 04:14:20.394/04:14:20.686 JST`, and the command was bounded by
  600 seconds.
- Actual GPU-hours and peak scratch use: at most `0.167` GPU-hours by the hard
  timeout; peak device/scratch memory was not captured. Result plus log occupy
  12,964 bytes.
- Command: `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 timeout 600s .venv312/bin/python spikes/verl_compat/qwen3_patch_embed_probe.py --run-id RP-16P-QWEN3-PATCH-EMBED-CONTROL --runtime control --physical-gpu 3 --output artifacts/compatibility/RP-16P-control-qwen3-patch-embed.json > artifacts/compatibility/RP-16P-control-qwen3-patch-embed.log 2>&1`.
- Outputs: result JSON SHA256
  `6cffb3f47e090ec767dbae9e61b2558bc4d44e1340c4d28f80a4dac704eaf150`
  (7,385 bytes) and log SHA256
  `beed078cc1cb8c3d4a07697fe2d5155c3791b1c231ac54d2dac7a62f24beca23`
  (5,579 bytes); overwrite was forbidden.
- Scorer/parser identity: exact probe hash above.
- Metrics: FP32 native Conv3D `0.3985923 ms`, Linear `0.150552 ms`,
  `2.647539x`, maximum absolute error `0.0`; BF16 native Conv3D
  `4711.1425299 ms`, Linear `0.08024 ms`, `58713.142197x`, maximum absolute
  error `0.015625`. Both passed their predeclared tolerances.
- Conclusion: projection-only parity and timing passed. Accepted-control BF16
  Conv3D is pathological on this B200. This result supplies the prerequisite
  evidence for a separately parity-tested repo-owned representation fast path;
  it does not by itself estimate end-to-end training throughput.

### RP-17-QWEN3-REPRESENTATION-FASTPATCH-REAL512-K4-GA4-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-17`; diagnostic bounded
  real-Qwen3 representation-phase train-core throughput measurement. It is not
  a promoted TGVF Adapter or a production prompt/data run.
- Spike-plan git revision and approval references: accepted decision
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION` in `PROJECT_TASK.md` section 2.1;
  production fast-path implementation commit
  `319c0375efd22e52a6b67b254519208b95dfa980` and config-bound GPU placement
  commit `1062e2db35a17376b33b9578be81ffb88c9c06e0`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: after replacing Qwen3's pathological BF16 full-patch Conv3D call
  with its parity-gated Linear expression, what steady two-B200 optimizer-step
  time is measured at the historical mathematical global batch 32 using
  local K=4, four accumulation microsteps, and the accepted `262144`-pixel
  cap; what 2,000-step train-core duration follows from that measurement?
- Baseline and exact output path: RP-11 GA4 steady tiny-image step was
  `20.366196423` seconds; RP-14A pre-fast-path capped real-image K4/GA1 step
  was `9.057971587` seconds. RP-16P measured the isolated BF16 patch projection
  at `4711.1425299` versus `0.08024` ms. RP-17 may write only under
  `artifacts/representation/RP-17-qwen3-fastpatch-real512-k4-ga4-throughput/`.
- Model and processor identity: stable local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, family `qwen3_vl`,
  tokenizer length `151669`, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  BF16, SDPA, local-only, no remote code, no tokenizer resize, and
  aspect-ratio-preserving `image_max_pixels=262144`.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`, no
  source artifact and no legacy-checkpoint initialization; three measured
  optimizer updates under the historical 2,000-update scheduler horizon.
- N/A fields and justification: rollout, behavior logprobs, policy/reference,
  reward, GRPO, SDPO teacher, answer judge, vLLM sampling, KV cache, and exact
  policy replay are absent from this representation-only timing diagnostic.
- Policy/reference initialization: N/A; Qwen3 is frozen and only the freshly
  initialized TGVF Adapter is trainable.
- Rollout policy version and allowed asynchronous staleness: N/A; execution is
  synchronous and no policy update exists.
- Code commit and worktree state: runtime commit
  `1062e2db35a17376b33b9578be81ffb88c9c06e0`; source/canonical config SHA256
  `adf4ffdced248a838aa1523a4fac9efa9e5f0e47e9cec22119b691b24d025f15`/
  `12f0a9982201cbf051357061df477e14b27670ecf44785c020354ce7405d3ad8`.
  Launch requires a clean committed tree and the runner's live-code check.
- Repository adapter/patch surface and hash: production patch is confined to
  `src/tgvf_rl/representation/training/runtime.py`, SHA256
  `0c22dc2f89753133fbbb54124beecd50837487f2a86c3cf03dfc03f8ad1dd901`;
  the existing Conv3D module, Parameters, state-dict keys, shapes, and values
  remain unchanged. No installed package or Qwen checkpoint is patched.
- Dataset/manifest, hashes, sample rule, and n: fixed RP-14A real-resolution
  train JSONL, source SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  eight rows/two complete K=4 groups over fixed DocVQA/TextVQA images; fixed
  validation source SHA256
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`.
  Each rank owns one group, so its four accumulation microsteps deliberately
  cross sampler epochs and repeat that fixed group; this is a throughput
  fixture, not a diversity or quality estimate.
- Native prompt/tool schema hash: smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`;
  native `tgvf_focus_tool` schema remains unchanged.
- Chat-template/token-fixture hash and token-ownership masks: accepted native
  Qwen3 chat-template identity above; only evidence-description tokens are
  labels, all visual/template positions are ignored, and tokenizer growth is
  forbidden.
- D/DeepStack/position/mask identity: unchanged native main `D`, ordered
  D-DeepStack layers `(8,16,24)`, native M-RoPE positions, and post-D
  original-image key blocking.
- Observation materialization/artifact identity used by all replays: no policy
  replay. Every Matrix-CE cell uses one complete main-D plus all-branch
  observation and the current single-pass differentiable readout path.
- RL framework/version/environment lock: accepted control Python `3.12.3`,
  Torch `2.9.0+cu128`, CUDA `12.8`, NCCL `2.27.5`, Transformers `4.57.6`,
  vLLM `0.12.0`, upstream veRL `0.9.0.dev0`, Pillow `12.3.0`; lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: legacy summed-evidence-NLL Matrix CE
  (`legacy_summed_nll`, inherited exactly from config schema v2), global
  valid-row mean, unchanged per-sample evidence-token-mean `L_gen`, historical
  Norm; weights `1.0/1.0/0.1`, manifold zero. AdamW `1e-4`, clip `1.0`, cosine
  warmup 100, floor ratio `.1`, horizon 2,000. Balanced Matrix CE is not used
  so this remains comparable to prior timing cells.
- Rollout/replay forward mode and adapter dropout/RNG contract: frozen Qwen in
  eval mode, TGVF Adapter dropout zero, no cache, deterministic algorithms,
  TF32 off, seed `20260719`, and CUBLAS workspace `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: token sampling/logprobs N/A;
  deterministic same-image sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout
  tensor parallelism, and training device mesh: BF16 forward, FP32 reduction,
  no quantization/KV/rollout TP, SDPA, FSDP2 `fsdp=[2]`, reshard after forward,
  and no offload.
- Logit/logprob/loss/gradient parity tolerances: RP-16P fast-path projection
  gate passed FP32 exact output and BF16 maximum absolute difference
  `0.015625`; production integration tests preserve exact state/Parameter
  identity, enforce geometry fail-closed, and cover FP32/BF16 parity. Timing
  has no numerical PASS floor; losses/gradients must be finite and tokenizer
  length must stay `151669`.
- World size, microbatch, accumulation, and global batch: world 2, one local
  K=4 same-image matrix per microstep, four gradient-accumulation microsteps,
  16 local/32 global rows and four local/eight global KxK matrices per update.
  The expected execution is two B8 Qwen calls per microstep/rank, eight calls
  and 64 cells per optimizer update/rank. Three updates are timed; steps 2/3
  form the predeclared steady mean.
- GPUs: only physical 0
  `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` and physical 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, both NVIDIA B200, mapped to
  logical 0/1. The user explicitly authorized this pair; immediate preflight
  found both at 0 MiB used and 0% utilization.
- Start/end timestamps, elapsed time, and session/process identity: PENDING.
- Actual GPU-hours and peak scratch use: PENDING; command hard limit is 1.0
  aggregate GPU-hour. The three train steps exclude validation and periodic
  checkpoints; the required final checkpoint/export is outside step timing.
- Command: after the clean/free-device preflight,
  `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp17_fastpatch_real512_ga4_throughput.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-17-qwen3-fastpatch-real512-k4-ga4-throughput/run.log 2>&1`.
- Outputs: PENDING; metrics, final Adapter export, final step-3 DCP, and run log
  are confined to the RP-17 output root and overwrite is forbidden.
- Scorer/parser identity: no answer scorer; strict native representation
  runner and config identity above.
- Metrics: PENDING. The reported 2,000-step estimate will be
  `mean(step 2, step 3) * 2000`, with initialization, validation, and periodic
  checkpoint overhead stated separately rather than hidden in train-core time.
- Conclusion: PENDING.

### RP-18-QWEN3-REPRESENTATION-NORESHARD-REAL512-K4-GA4-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-18`; diagnostic
  configuration-only A/B against RP-17.
- Spike-plan git revision and approval references: accepted
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION`; runtime commit
  `1062e2db35a17376b33b9578be81ffb88c9c06e0`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: does keeping the 72M-parameter TGVF Adapter unsharded between its
  forward and backward (`reshard_after_forward=false`) reduce RP-17's
  `12.0231005715`-second steady optimizer step without changing mathematics?
- Baseline and exact output path: RP-17 steps 2/3 `12.019430332`/
  `12.026770811` seconds; output only under
  `artifacts/representation/RP-18-qwen3-noreshard-real512-k4-ga4-throughput/`.
- Model and processor identity: exactly RP-17 stable Qwen3-VL-8B-Thinking,
  BF16/SDPA, tokenizer `151669`, native template, no resize, max pixels
  `262144`.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`; no
  source/legacy artifact; three updates under the 2,000-step scheduler.
- N/A fields and justification: policy/reference, rollout/logprobs, reward,
  GRPO, SDPO, judge, vLLM, KV cache and replay are absent.
- Policy/reference initialization: N/A; frozen Qwen and fresh Adapter only.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation training.
- Code commit and worktree state: runtime commit above; source/canonical TOML
  SHA256 `7bdc1bd0271d4a8377ac0d4adad0f099d51ad6a987f407738094dfd6a58a0b0f`/
  `1daf168b621d7ab6c4811df43145da25365270f6366786053522358f1ab3c0ed`;
  clean committed launch required.
- Repository adapter/patch surface and hash: no code/package/model patch; only
  the existing public FSDP2 boolean changes from RP-17 `true` to `false`.
- Dataset/manifest, hashes, sample rule, and n: identical RP-17 fixed two-image
  real-resolution K4 fixture and seeds; each rank repeats its one group across
  GA4, so this is only a matched execution A/B.
- Native prompt/tool schema hash: identical RP-17 smoke-only native prompt and
  `tgvf_focus_tool` schema.
- Chat-template/token-fixture hash and token-ownership masks: identical RP-17;
  evidence-only labels and no tokenizer growth.
- D/DeepStack/position/mask identity: identical main D plus branches `(8,16,24)`,
  native M-RoPE and post-D source-key blocking.
- Observation materialization/artifact identity used by all replays: no replay;
  identical single-pass complete candidate observations.
- RL framework/version/environment lock: identical accepted Torch 2.9 control
  lock SHA256 `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: identical RP-17 legacy summed-NLL
  Matrix CE, L_gen and Norm, weights `1/1/.1`, manifold zero.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical
  deterministic state, frozen Qwen, Adapter dropout zero and CUBLAS `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: sampling N/A; sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/FP32 reduction, SDPA, no
  quantization/KV/TP/offload, FSDP2 `[2]`; only reshard-after-forward is false.
- Logit/logprob/loss/gradient parity tolerances: RP-17 finite loss/gradient and
  structural contracts must remain; same-seed scientific metrics are compared,
  while timing and allocator values are excluded from exact parity.
- World size, microbatch, accumulation, and global batch: identical RP-17:
  world 2, K4, GA4/GPR1, global batch 32, eight B8 Qwen calls/rank/update.
- GPUs: only user-authorized physical GPUs 0 and 3, UUIDs recorded in RP-17,
  both observed at 0 MiB and 0% utilization immediately before planning.
- Start/end timestamps, elapsed time, and session/process identity: launched
  `2026-07-20 05:15:45 +09:00`; completed `05:16:50 +09:00`; torchrun parent
  PID `1799780`. The bounded invocation stayed below one aggregate GPU-hour.
- Actual GPU-hours and peak scratch use: about `0.036` aggregate GPU-hours by
  invocation wall time. Peak allocated/reserved bytes were
  `64,215,435,264`/`73,146,564,608` on the maximum rank.
- Command: `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp18_noreshard_real512_ga4_throughput.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-18-qwen3-noreshard-real512-k4-ga4-throughput/run.log 2>&1`.
- Outputs: complete metrics, final Adapter artifact, and final step-3 DCP under
  the exact RP-18 root; overwrite remained forbidden.
- Scorer/parser identity: no answer scorer; strict representation runner.
- Metrics: steps 1/2/3 were `13.207190879`, `11.702221723`, and
  `11.939163785` seconds. The predeclared steady mean was
  `11.820692754` seconds, implying `6.5671` train-core hours for 2,000 steps.
  Same-seed losses and gradient norms matched RP-17 exactly at every step.
- Conclusion: `reshard_after_forward=false` preserved the measured mathematics
  and improved the steady step by about `1.68%`; it is retained for the next
  timing cell but does not explain the discontinuous GPU utilization.

### RP-19-QWEN3-REPRESENTATION-CONTINUOUS-REAL512-K4-GA4-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-19`; bounded execution-
  continuity A/B against RP-18.
- Spike-plan git revision and approval references: accepted
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; runtime commit
  `fcd470c15e621fc2ac4849bb3be4d09cc008bc57`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: after reducing one K4 group's Qwen processor/image work from
  8 calls/12 image instances to 1 call/1 image instance and fusing normal-step
  host synchronizations, does the RP-18 GPU timeline become more continuous and
  does its `11.820692754`-second steady optimizer step improve?
- Baseline and exact output path: RP-18 steps 2/3 were `11.702221723` and
  `11.939163785` seconds. RP-19 writes only under
  `artifacts/representation/RP-19-qwen3-continuous-real512-k4-ga4-throughput/`.
- Model and processor identity: stable local Qwen3-VL-8B-Thinking, BF16/SDPA,
  tokenizer `151669`, exact native template, no resize, local-only, and
  aspect-ratio-preserving `image_max_pixels=262144`.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`; no
  source or legacy artifact; three updates under the unchanged 2,000-step
  scheduler horizon.
- N/A fields and justification: policy/reference, rollout/logprobs, reward,
  GRPO, SDPO, judge, vLLM sampling, KV cache and replay are absent from this
  representation-only timing diagnostic.
- Policy/reference initialization: N/A; frozen original Qwen and fresh TGVF
  Adapter only.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation training with no policy update.
- Code commit and worktree state: runtime commit above; source/canonical TOML
  SHA256 `0328ca66804ad185083d7e95ad5ad5b2f6fcaabc2957670c5686a4f1ee056fa6`/
  `e13d0fae6b4393ea1b104ecef51f2ece51bf4bd7a03493cafa7d1aa3f23b7445`;
  launch requires the committed config/ledger-only planning delta and no live
  implementation drift.
- Repository adapter/patch surface and hash: repo-owned changes are confined to
  the Qwen3 native representation group builder and trainer bookkeeping. The
  first processor result is the group-owned visual geometry; later transcripts
  derive exact native visual expansions, while Qwen, Adapter parameters,
  objectives, and installed packages remain unchanged.
- Dataset/manifest, hashes, sample rule, and n: identical RP-18 fixed two-image
  real-resolution K4 fixture, source SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  with the same validation fixture and seeds. Each rank repeats one group over
  GA4; this is a throughput fixture, not a quality estimate.
- Native prompt/tool schema hash: unchanged smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`
  and native `tgvf_focus_tool` schema.
- Chat-template/token-fixture hash and token-ownership masks: unchanged exact
  template SHA256 `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  evidence-only labels and no tokenizer growth.
- D/DeepStack/position/mask identity: unchanged main D, ordered branches
  `(8,16,24)`, native M-RoPE positions, and post-D source-key blocking.
- Observation materialization/artifact identity used by all replays: no policy
  replay; each Matrix-CE column still uses one complete differentiable main-D
  plus all D-DeepStack branches from that target.
- RL framework/version/environment lock: unchanged accepted Python 3.12 / Torch
  2.9 control lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: unchanged legacy summed-NLL Matrix CE,
  L_gen and Norm, weights `1/1/.1`, manifold zero, global reductions unchanged.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic,
  frozen Qwen eval state, Adapter dropout zero, no cache, TF32 off, and CUBLAS
  workspace `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: sampling/logprobs N/A; same-image
  sampler seeds remain 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 forward/FP32 reduction, SDPA, no
  quantization/KV/TP/offload, FSDP2 mesh `[2]`, and
  `reshard_after_forward=false`.
- Logit/logprob/loss/gradient parity tolerances: same-seed step losses, gradient
  norm, sample order, physical Qwen call sizes, and tokenizer length must equal
  RP-18; timing/allocator/utilization samples are measurement outputs. CPU gates
  passed 63 focused tests plus Ruff before planning.
- World size, microbatch, accumulation, and global batch: world 2, K4,
  GA4/GPR1, 16 local/32 global rows, eight B8 Qwen calls/rank/update, global
  batch 32 unchanged.
- GPUs: only user-authorized physical GPUs 0 and 3, mapped to logical 0/1; both
  must be idle at immediate preflight.
- Start/end timestamps, elapsed time, and session/process identity: torchrun
  parent PID `1820984` launched `2026-07-20 05:32:08 +09:00`; the complete
  utilization trace ended `05:33:11 +09:00`.
- Actual GPU-hours and peak scratch use: about `0.035` aggregate GPU-hours by
  invocation wall time. Maximum-rank peak allocated/reserved bytes were
  `64,216,335,872`/`73,125,593,088`. The 100 ms utilization trace contains
  340 in-training samples per GPU under the exact RP-19 root.
- Command: `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp19_continuous_real512_ga4_throughput.toml`;
  a separate read-only 100 ms `nvidia-smi` sampler observes physical 0 and 3.
- Outputs: complete metrics, final Adapter artifact, final step-3 DCP, run log,
  and 100 ms utilization trace under the RP-19 root; overwrite was forbidden.
- Scorer/parser identity: no answer scorer; strict native representation runner.
- Metrics: steps 1/2/3 were `11.886870550`, `11.032586041`, and
  `11.157873967` seconds. The predeclared steady mean was `11.095230004`
  seconds (`6.1640` train-core hours for 2,000 steps), a `6.14%` improvement
  over RP-18. Same-seed losses, gradient norms, sample order, and B8 physical
  calls matched RP-18 exactly. During the memory-resident training window,
  GPU0/GPU3 mean utilization was `27.3%`/`45.2%`; samples below 50% were
  `72.9%`/`54.4%`, zero-utilization samples were `46.8%`/`28.2%`, and the
  longest below-50% runs were about `2.9`/`1.8` seconds.
- Conclusion: processor/layout reuse and fused host reads are accepted: they
  preserve the exact measured objective and save `6.14%`. They do not solve
  the discontinuous GPU timeline. The next bounded cell therefore combines the
  same four local K4 matrices as two B16 direct-group accumulation windows
  (`GA2`) while preserving global batch 32.

### RP-20-QWEN3-REPRESENTATION-CONTINUOUS-REAL512-K4-GPR4-GA2-B16-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-20`; bounded B16/GA2
  utilization and throughput A/B against RP-19 B8/GA4.
- Spike-plan git revision and approval references: accepted
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; direct-group accumulation
  implementation commit `bd3d9ca2010767e3e14f0610efaa70bed7a2d5b6`.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
- Question: with the mathematical global batch fixed at 32, does partitioning
  four local K4 groups into two accumulation windows, each executed as B16,
  reduce RP-19's utilization gaps and `11.095230004`-second steady step?
- Baseline and exact output path: RP-19 steps 2/3 were `11.032586041` and
  `11.157873967` seconds. RP-20 writes only under
  `artifacts/representation/RP-20-qwen3-continuous-real512-k4-gpr4-ga2-b16-throughput/`.
- Model and processor identity: identical RP-19 stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, exact native template,
  no resize, local-only, and `image_max_pixels=262144`.
- Representation checkpoint identity: fresh TGVF Adapter seed `20260719`; no
  source/legacy artifact; three updates under the unchanged 2,000-step horizon.
- N/A fields and justification: policy/reference, rollout/logprobs, reward,
  GRPO, SDPO, judge, vLLM sampling, KV cache and replay are absent.
- Policy/reference initialization: N/A; frozen original Qwen and fresh TGVF
  Adapter only.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  representation update.
- Code commit and worktree state: runtime commit above; configuration
  source/canonical SHA256
  `17cc62baf2a43adfd68b2405f536cc25ef1385188e4b1ae348d2c5048a950810`/
  `f5fa2d519696ff5fa50e5ae5abf7c06774b98ae43ec460e938cffb3e88285c9e`;
  clean committed launch required.
- Repository adapter/patch surface and hash: RP-19 processor/layout and fused
  host-read changes remain; the new repo-owned change only partitions the
  versioned total direct-group identity across GA windows and fixes the
  performance matrix denominator. No Qwen/Adapter parameter or package patch.
- Dataset/manifest, hashes, sample rule, and n: 16-row/four-group matched
  real-resolution fixture, source SHA256
  `42bdeedf0d5375792ac7108142538434330e21c7b930768356ec693d780ef381`.
  Two group identities per rank reuse the exact RP-19 source image/content;
  deterministic cycling supplies four local matrices per update. Validation,
  overlap policy, and sampler seeds remain RP-19 identities.
- Native prompt/tool schema hash: unchanged smoke prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`
  and native `tgvf_focus_tool`.
- Chat-template/token-fixture hash and token-ownership masks: unchanged exact
  template SHA256 `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  evidence-only labels, and no tokenizer growth.
- D/DeepStack/position/mask identity: unchanged main D, ordered `(8,16,24)`
  branches, M-RoPE, and post-D source-key blocking.
- Observation materialization/artifact identity used by all replays: no replay;
  every K4 column remains one indivisible main-D/all-DeepStack observation, and
  the four CE matrices remain independent across the two B16 windows.
- RL framework/version/environment lock: unchanged accepted Python 3.12 /
  Torch 2.9 control lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: unchanged legacy summed-NLL Matrix CE,
  L_gen and Norm, weights `1/1/.1`, manifold zero. Both windows use the exact
  same full-step global row/sample denominators.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical
  deterministic RP-19 state, frozen Qwen eval, Adapter dropout zero, no cache,
  TF32 off, CUBLAS `:4096:8`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: sampling/logprobs N/A; same-image
  sampler seeds 71/73.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/FP32 reduction, SDPA, no
  quantization/KV/TP/offload, FSDP2 `[2]`, reshard false.
- Logit/logprob/loss/gradient parity tolerances: global row/sample counts must
  remain 32; Qwen calls must be four B16 calls/rank/update; tokenizer length and
  finite objectives/gradients are exact gates. Because B16 changes BF16 kernel
  grouping, compare objective/gradient values to RP-19 with the existing BF16
  execution tolerance rather than demanding bit identity. CPU gates passed
  93 focused tests plus Ruff.
- World size, microbatch, accumulation, and global batch: world 2, total GPR4,
  K4, GA2, two groups/eight rows per rank per accumulation window, global batch
  32. Expected physical readout is four B16 calls/rank/update for 64 cells.
- GPUs: only user-authorized physical 0 and 3 mapped to logical 0/1; both must
  be idle at immediate preflight.
- Start/end timestamps, elapsed time, and session/process identity: `PENDING`.
- Actual GPU-hours and peak scratch use: `PENDING`; hard limit one aggregate
  GPU-hour, plus one diagnostic 100 ms `nvidia-smi` trace.
- Command: `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp20_continuous_real512_b16_ga2_throughput.toml`;
  a read-only 100 ms utilization sampler observes physical 0/3.
- Outputs: `PENDING`; overwrite forbidden under the exact RP-20 root.
- Scorer/parser identity: no answer scorer; strict native representation runner.
- Metrics: `PENDING`; compare steady mean, peak allocation, and utilization
  distributions/longest below-50% runs directly with RP-19.
- Conclusion: `PENDING`.

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
