# Experiment Ledger

I8H-20260719 authorizes the bounded compatibility cells below. It does not
authorize production training. Only physical GPUs 2 and 3 may be exposed.

## Planned bounded cells

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
- Lifecycle status: `PLANNED`.
- Result: `PENDING`.
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
- Start/end timestamps, elapsed time, and session/process identity: `PENDING`.
- Actual GPU-hours and peak scratch use: `PENDING`; hard timeout 600 seconds.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 NCCL_DEBUG=WARN timeout 600s .venv312/bin/torchrun --standalone --nproc-per-node=2 spikes/verl_compat/fsdp2_smoke.py --config configs/smoke/fsdp2.toml --checkpoint-dir artifacts/compatibility/SC-30-fsdp2-infra-db3315a-checkpoint --output artifacts/compatibility/SC-30-fsdp2-infra-db3315a.json > artifacts/compatibility/SC-30-fsdp2-infra-db3315a.log 2>&1`.
- Outputs: strict distributed checkpoint, result JSON and log above.
- Scorer/parser identity: smoke script/config SHA256 above.
- Metrics: `PENDING`—per-rank loss/shard digest, exact resume flag, versions and
  public veRL symbols.
- Conclusion: `PENDING`; any nondeterminism, missing state or rank failure stops
  the cell.

## Compatibility-spike status

CPU public-API, transport, objective and oracle tests passed before these rows
were entered. The two planned cells are bounded evidence; they do not silently
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
