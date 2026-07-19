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

### SC-30-T211-FSDP2-INFRA

- Cell/matrix ID and mandatory/diagnostic class: `SC-30-T211-FSDP2-INFRA`;
  mandatory Torch-2.11 two-rank composable-FSDP2 checkpoint/resume gate.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime commit
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; accepted re-spike in
  `PROJECT_TASK.md` §9.2 and I8H-20260719.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
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
  launch must use a clean descendant containing only this config/ledger plan.
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
  `mean((model(x)-target)**2)`, AdamW; no production RL interpretation.
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
- Start/end timestamps, elapsed time, and session/process identity: pending;
  captured from the launched process and result.
- Actual GPU-hours and peak scratch use: pending; bounded by 600 seconds.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 600s .venv-torch211-cu129/bin/torchrun --standalone --nproc-per-node=2 spikes/verl_compat/fsdp2_smoke.py --stack torch211-cu129 --config configs/smoke/fsdp2_torch211.toml --checkpoint-dir artifacts/compatibility/SC-30-T211-fsdp2-infra-checkpoint --output artifacts/compatibility/SC-30-T211-fsdp2-infra.json > artifacts/compatibility/SC-30-T211-fsdp2-infra.log 2>&1`.
- Outputs: new result/log/DCP paths above; overwrite forbidden.
- Scorer/parser identity: exact script/config hashes above.
- Metrics: pending exact-resume and runtime identity assertions.
- Conclusion: pending; cannot promote the candidate by itself.

### SC-21-T211-VERL-VLLM-WEIGHT-SYNC

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-21-T211-VERL-VLLM-WEIGHT-SYNC`; mandatory upstream veRL FSDP2 actor to
  vLLM TP=2 generation/weight-sync gate.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
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
- Code commit and worktree state: runtime commit above; clean planned-launch
  descendant required.
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
  generated response tokens alone form the update mask.
- D/DeepStack/position/mask identity: N/A by deliberate exclusion.
- Observation materialization/artifact identity used by all replays: each
  generated response and processed rollout logprob returns through
  TransferQueue to the same synchronous update; no observation recomputation.
- RL framework/version/environment lock: exact candidate stack and lock SHA256
  identical to `SC-30-T211-FSDP2-INFRA`.
- Objective equations and normalization: infrastructure-only registered
  zero-advantage estimator plus mean generated-token NLL; fixed reward 1.0;
  production GRPO/PPO/SDPO math is explicitly excluded.
- Rollout/replay forward mode and adapter dropout/RNG contract: full
  determinism, LoRA/dropout 0, sync V1 trainer, no cache sleep, seed `20260720`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.23.0+cu129`, greedy
  (`do_sample=false`, temperature 1.0, top-p 1.0, top-k -1), 16-token cap, no
  custom processors, `processed_logprobs` after represented transforms.
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
  `SC-30-T211-FSDP2-INFRA`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: pending;
  driver and Ray/veRL process identities captured in outputs/log.
- Actual GPU-hours and peak scratch use: pending; 1800-second hard timeout.
- Command: `.venv-torch211-cu129/bin/python spikes/verl_compat/verl_fsdp2_vllm_sync_smoke.py --python .venv-torch211-cu129/bin/python --output artifacts/compatibility/SC-21-T211-verl-vllm-weight-sync.json --timeout-seconds 1800 --launch-gpu`.
- Outputs: prefix listed above; overwrite forbidden.
- Scorer/parser identity: driver and registered objective hashes above; public
  manager must be identical to upstream `AgentLoopManagerTQ`.
- Metrics: pending two-step sync, update, staleness, and logprob assertions.
- Conclusion: pending; passing proves the no-sleep/no-TGVF-plugin combined
  transport only, not production RL mathematics.

### SC-20-T211-QWEN3-VLLM-LATENT

- Cell/matrix ID and mandatory/diagnostic class:
  `SC-20-T211-QWEN3-VLLM-LATENT`; mandatory candidate-stack real-Qwen3 native
  repeated-tool/precomputed-latent vLLM smoke.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
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
  driver `d841b5edd15db66742ccb96958e3e966d883cb6d6226120e5fc363ce401b6ba2`;
  no site-package patch.
- Dataset/manifest, hashes, sample rule, and n: one synthetic request, no
  dataset; three deterministic BF16 latent rows from seed `20260719`.
- Native prompt/tool schema hash: native `tgvf_focus_tool` schema
  `1a0e7bc78b134c5f6d1258d894fa6cf130bb09226c59caffd6f1d0a871d2a361`;
  exactly two native calls and responses.
- Chat-template/token-fixture hash and token-ownership masks: template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`;
  generation prefill asserted; no RL loss mask.
- D/DeepStack/position/mask identity: source/call0/call1, each grid `(1,2,2)`,
  one merged row, width `4096*(1+3)=16384`, branches `(8,16,24)`, native
  processor M-RoPE positions; aggregate latent hash is asserted by output.
- Observation materialization/artifact identity used by all replays: one
  pre-materialized immutable public-input list; no policy/reference replay.
- RL framework/version/environment lock: exact candidate stack/lock identical
  to `SC-30-T211-FSDP2-INFRA`.
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
  `SC-30-T211-FSDP2-INFRA`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: pending.
- Actual GPU-hours and peak scratch use: pending; 1800-second timeout.
- Command: `CUDA_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CC=/usr/bin/gcc CXX=/usr/bin/g++ CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 1800s .venv-torch211-cu129/bin/python spikes/verl_compat/qwen3_vllm_latent_smoke.py --output artifacts/compatibility/SC-20-T211-qwen3-vllm-latent.json > artifacts/compatibility/SC-20-T211-qwen3-vllm-latent.log 2>&1`.
- Outputs: new result/log paths above; overwrite forbidden.
- Scorer/parser identity: driver, public plugin and native renderer at runtime
  commit and hashes above.
- Metrics: pending native transcript, latent, token and processed-logprob checks.
- Conclusion: pending; not a trained Adapter or replay-parity claim.

### RP-15P-T211-QWEN3-PATCH-EMBED

- Cell/matrix ID and mandatory/diagnostic class:
  `RP-15P-T211-QWEN3-PATCH-EMBED`; mandatory candidate patch-projection parity
  and bounded timing diagnostic on one authorized GPU.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; `PROJECT_TASK.md` §9.2 and
  I8H-20260719.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
- Question: is the mathematically equivalent Linear projection numerically
  compatible with native Conv3D in FP32/BF16, and what is its bounded speedup
  on a fixed 512-by-512-equivalent patch tensor?
- Baseline and exact output path: native Conv3D in the same process; result
  `artifacts/compatibility/RP-15P-T211-qwen3-patch-embed.json`.
- Model and processor identity: accepted local Qwen3; config SHA256
  `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`,
  model index SHA256
  `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`;
  only patch weight/bias from their declared safetensors shard are loaded.
- Representation checkpoint identity: N/A; frozen base patch projection only.
- N/A fields and justification: data, prompt, tokenizer serialization,
  D/DeepStack, policy/reference, reward, sampling, optimizer, replay, GRPO and
  SDPO are absent.
- Policy/reference initialization: N/A.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: runtime commit above; clean planned-launch
  descendant required.
- Repository adapter/patch surface and hash: probe SHA256
  `3e424250cc5bff0d76e8eee29cec3d53412349fbe79e59ce20953fc2a5a924f6`;
  no model/package patch.
- Dataset/manifest, hashes, sample rule, and n: generated seed `20260720`,
  flattened shape `(1024,1536)` corresponding to grid `(1,32,32)`.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: candidate Torch/CUDA identity and
  lock SHA256 identical to `SC-30-T211-FSDP2-INFRA`.
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
- Start/end timestamps, elapsed time, and session/process identity: pending.
- Actual GPU-hours and peak scratch use: pending; bounded 26 calls plus load.
- Command: `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 .venv-torch211-cu129/bin/python spikes/verl_compat/qwen3_patch_embed_probe.py --runtime candidate --physical-gpu 3 --output artifacts/compatibility/RP-15P-T211-qwen3-patch-embed.json`.
- Outputs: new JSON above; overwrite forbidden.
- Scorer/parser identity: exact probe hash above.
- Metrics: pending max error and synchronized latency/speed ratio per dtype.
- Conclusion: pending; a pass authorizes considering the Linear fast path but
  does not itself change production code.

### RP-15-QWEN3-REPRESENTATION-TORCH211-FSDP2-EMBEDDING-K4-GPR4-GA1-SINGLEPASS-CELLB32-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-15`; mandatory three-step
  real-Qwen representation FSDP2 forward/backward and throughput comparison.
- Spike-plan git revision and VA0/VA1/VA2 approval references: runtime
  `2918c8913756e4bbac0e6aa171c102ceab4d409c`; `PROJECT_TASK.md` §9.2,
  I8H-20260719 and the accepted RP-13 geometry.
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
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
  `aed59d79ebca37cc684259a79871e4ade3899362658721c002bd7a001b6e8ed0`/
  `ea4ab704b131ac405f491f24cf9f684d5c277e01ff58b6cc99b112f0ad4692bf`;
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
  to `SC-30-T211-FSDP2-INFRA`.
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
  `SC-30-T211-FSDP2-INFRA`; no other GPU visible.
- Start/end timestamps, elapsed time, and session/process identity: pending;
  torchrun/rank identities captured in the log/result.
- Actual GPU-hours and peak scratch use: pending; 1800-second timeout and one
  final checkpoint/export.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s .venv-torch211-cu129/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/smoke/representation_qwen3_embedding_rp15_torch211_singlepass_cellb32_throughput.toml > /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-15-qwen3-representation-torch211-fsdp2-embedding-k4-gpr4-ga1-singlepass-cellb32-throughput/run.log 2>&1`.
- Outputs: Adapter, metrics, final strict DCP and log under the root above;
  overwrite forbidden; no validation/periodic save within steps 1-3.
- Scorer/parser identity: strict representation parser/runner/objective at
  runtime commit and tree/hash identities above.
- Metrics: pending three step times/rows-per-second, peak CUDA, loss/gradient,
  token length, cell/call/matrix counts and final DCP/export checks.
- Conclusion: pending; bounded throughput evidence only, not a promoted
  representation artifact or production training result.

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
