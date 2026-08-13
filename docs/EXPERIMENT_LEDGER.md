# Experiment Ledger

I8H-20260719 authorizes the bounded compatibility cells below. It does not
authorize production training. Its physical-GPU 2/3 restriction remains fixed
for those cells. On 2026-07-21 the user separately authorized physical GPUs
0--3 for the original-policy benchmark baseline and its throughput work; that
authorization does not retroactively alter any earlier cell.

Experiment namespaces are disjoint: `SC-*` is reserved for cells fixed by the
veRL compatibility matrix, while `RP-*` identifies bounded representation-
phase executions, `BE-*` identifies benchmark evaluation runs, and `BJ-*`
identifies bounded benchmark-judge deployments. `T1-*` identifies inference-
only full-image policy-data-selection difficulty runs; it never denotes policy
training, a benchmark, or a TGVF/visual-tool experiment. A
materialized run ID is never renamed after execution; an identity collision is
retained as `INVALID` and rerun under a new planned ID.

Keep entries proportional to the run. Inference-only `BJ-*` rows record the
accepted identity/config hash, command, devices, timing/memory, result and root
cause; unrelated RL/training fields are covered by one concise N/A statement
instead of repeated bullets.

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
- Lifecycle status: `COMPLETE`.
- Result: `PASS` as a bounded execution/contract cell; `B16 x GA2` is rejected
  as the throughput default because it did not improve RP-19.
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
- Start/end timestamps, elapsed time, and session/process identity: launched
  `2026-07-20 05:45:52 +09:00` as torchrun parent PID `1833912`; final runner
  output completed at `05:46:55 +09:00` (about 63 seconds wall time).
- Actual GPU-hours and peak scratch use: the three measured train steps used
  `0.0190` aggregate train-core GPU-hours; the full launch was below `0.035`
  aggregate GPU-hours. The completed output tree occupies `578,004,317` bytes
  (`538 MiB` by `du`), below the one aggregate GPU-hour limit. A 100 ms
  `nvidia-smi` trace was captured.
- Command: `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp20_continuous_real512_b16_ga2_throughput.toml`;
  a read-only 100 ms utilization sampler observes physical 0/3.
- Outputs: completed metrics, final Adapter, step-3 checkpoint, run log, and
  utilization trace under the exact RP-20 root; final artifact manifest SHA256
  `c21f9697e332d0ba41272088bf0cbf2e0b8c9e15f8e8b5c57b88d08a19be4d9b`.
  Overwrite remains forbidden.
- Scorer/parser identity: no answer scorer; strict native representation runner.
- Metrics: steps 1/2/3 took `11.956811975`, `11.129316294`, and
  `11.143948025` seconds. The predeclared steady mean is `11.1366321595`
  seconds, equivalent to `6.1870` train-core hours for 2,000 steps and `0.37%`
  slower than RP-19. Every rank executed four B16 Qwen calls and 64 cells per
  step; global row/sample counts were exactly 32. The maximum rank peak
  allocated/reserved memory was `110,357,683,200`/`119,632,035,840` bytes.
  In the memory-resident trace, GPU0/GPU3 mean utilization was approximately
  `41.9%`/`32.7%`; below-50% samples were `55.9%`/`67.7%`, zero-utilization
  samples were `36.8%`/`45.2%`, and the longest below-50% runs were about
  `3.3`/`4.0` seconds.
- Conclusion: the direct-group accumulation implementation and B16 execution
  contract pass, but packing two groups per accumulation window neither
  improves throughput nor removes the utilization gaps. Keep RP-19's B8/GA4
  execution as the comparison/default path and next remove hot-path
  CUDA-to-host validation synchronizations without changing the objective.

### RP-21-QWEN3-REPRESENTATION-SYNCFUSED-REAL512-K4-GA4-THROUGHPUT

- Cell/matrix ID and mandatory/diagnostic class: `RP-21`; bounded hot-path
  synchronization/runtime-invariant throughput A/B against RP-19.
- Spike-plan git revision and approval references: accepted
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; Norm finite/replay fusion
  commit `281dc1b5ee12be925ea2edf8c04d4f771a82417a`; native conditioning/runtime
  boundary commit `66da1cffc71c2aa86e3403ad9caa110aab403185`.
- Lifecycle status: `ABORTED_BEFORE_LAUNCH`.
- Result: `NOT_RUN`; the mandatory idle-device preflight found physical GPU0
  occupied by PID `1858503` at about 121 GB and high utilization. No RP-21
  process was started and no output path was created.
- Question: with RP-19's exact B8/GA4 and global batch 32 retained, how much do
  fused Norm validation, bound CPU token authority, and per-group runtime
  invariant checks reduce the `11.095230004`-second steady step and the
  below-50%/zero-utilization intervals?
- Baseline and exact output path: RP-19 steps 2/3 were `11.032586041` and
  `11.157873967` seconds. RP-21 writes only under
  `artifacts/representation/RP-21-qwen3-syncfused-real512-k4-ga4-throughput/`.
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
- Code commit and worktree state: runtime commit
  `66da1cffc71c2aa86e3403ad9caa110aab403185`; configuration source/canonical
  SHA256 `c3bcc1639ac5ac6dc14630596411a6a0d39772632d6c398a939fcc414693ccd6`/
  `7125f18d096fdccdc7b21ec0a47e8bf3ebc0e2bdf0722737602efabea47b8278`;
  clean committed launch required.
- Repository adapter/patch surface and hash: processor/layout reuse and fused
  trainer host reads remain from RP-19. RP-21 additionally fuses all
  streaming Norm finite checks and live/stored comparisons, binds the
  processor-derived CPU token tuple to the exact input tensor identity/version,
  and validates frozen runtime invariants at each group entry/exit rather than
  every internal runtime call. Public checked fallbacks remain fail-closed.
- Dataset/manifest, hashes, sample rule, and n: identical RP-19 eight-row/two-
  group matched real-resolution fixture, source SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`;
  deterministic same-image K4 cycling, validation identity, overlap policy,
  and sampler seeds 71/73 are unchanged.
- Native prompt/tool schema hash: unchanged smoke prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`
  and native `tgvf_focus_tool`.
- Chat-template/token-fixture hash and token-ownership masks: unchanged exact
  template SHA256 `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  evidence-only labels, and no tokenizer growth.
- D/DeepStack/position/mask identity: unchanged main D, ordered `(8,16,24)`
  branches, M-RoPE, and post-D source-key blocking.
- Observation materialization/artifact identity used by all replays: no replay;
  every K4 column remains one indivisible main-D/all-DeepStack observation.
- RL framework/version/environment lock: unchanged accepted Python 3.12 /
  Torch 2.9 control lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: unchanged legacy summed-NLL Matrix CE,
  L_gen and Norm, weights `1/1/.1`, manifold zero; full-step global row/sample
  denominators are unchanged.
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
  remain 32; each rank must execute eight B8 Qwen calls and 64 cells per update;
  tokenizer length, sample order, objective values, and gradient norms must
  match RP-19 exactly. CPU gates passed 80 focused tests plus Ruff.
- World size, microbatch, accumulation, and global batch: world 2, K4, GA4,
  one group/four rows per rank per accumulation window, global batch 32.
- GPUs: only user-authorized physical 0 and 3 mapped to logical 0/1; both must
  be idle at immediate preflight.
- Start/end timestamps, elapsed time, and session/process identity: preflight
  rejected launch at approximately `2026-07-20 06:09 +09:00`; no session/PID.
- Actual GPU-hours and peak scratch use: zero; no output or utilization trace.
- Command: `CUDA_VISIBLE_DEVICES=0,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp21_syncfused_real512_ga4_throughput.toml`;
  a read-only 100 ms utilization sampler observes physical 0/3.
- Outputs: none; the exact RP-21 root was not created.
- Scorer/parser identity: no answer scorer; strict native representation runner.
- Metrics: none; preflight-only result.
- Conclusion: do not contend with the unrelated GPU0/1 training process. Rerun
  the unchanged question as RP-22 on currently idle, previously user-authorized
  physical GPUs 2 and 3 through the explicit configuration pair.

### RP-22-QWEN3-REPRESENTATION-SYNCFUSED-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and mandatory/diagnostic class: `RP-22`; exact RP-21 question
  rerun with only the explicit physical-device pair and output identity changed.
- Spike-plan git revision and approval references: accepted
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION` and
  `RPI-20260720-CONFIG-BOUND-GPU-PAIR`; implementation commits
  `281dc1b5ee12be925ea2edf8c04d4f771a82417a` and
  `66da1cffc71c2aa86e3403ad9caa110aab403185`.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question/baseline: identical RP-21 hot-path sync/utilization question against
  RP-19's exact B8/GA4 steps 2/3 (`11.032586041`/`11.157873967` seconds).
- Exact output path:
  `artifacts/representation/RP-22-qwen3-syncfused-real512-k4-ga4-throughput-gpu23/`;
  overwrite forbidden.
- Model/processor/representation initialization: stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, native template, no
  resize, `image_max_pixels=262144`; frozen original Qwen and fresh TGVF
  Adapter seed `20260719`; three updates under the 2,000-step horizon.
- N/A fields and justification: policy/reference, rollout/logprobs, reward,
  GRPO, SDPO, judge, vLLM sampling, KV cache and replay are absent; synchronous
  representation training has no asynchronous staleness.
- Code/worktree/config identity: runtime commit
  `66da1cffc71c2aa86e3403ad9caa110aab403185`; source/canonical config SHA256
  `6f325b9c8965c4dacef408214114af58672ea5b52708c4d537ff54c4bbc53ba4`/
  `2aee4f30e150e4aa7acfee534645062f702255a311ac9703acd0c458608ceaf9`;
  clean committed launch required.
- Repository patch surface: exactly RP-21's fused streaming Norm validation,
  bound processor-derived CPU token authority, per-group runtime invariant
  entry/exit, plus all accepted RP-19 processor/layout and trainer host-read
  changes. No package, Qwen parameter, Adapter parameter, or objective change.
- Dataset/sample identity: RP-19 eight-row/two-group real-resolution fixture,
  source SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  same-image K4 cycling, disjoint validation, sampler seeds 71/73.
- Prompt/tool/template/token ownership: prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`,
  native `tgvf_focus_tool`, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  evidence-only labels, no tokenizer growth.
- D/DeepStack/position/mask/observation identity: unchanged main D, ordered
  `(8,16,24)` branches, M-RoPE, source-key blocking, and atomic complete
  candidate observation; replay is N/A.
- Framework/objective/determinism identity: accepted Python 3.12/Torch 2.9
  lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  legacy summed-NLL Matrix CE + L_gen + Norm weights `1/1/.1`, manifold zero;
  frozen Qwen eval, Adapter dropout zero, no cache, TF32 off, CUBLAS `:4096:8`.
- Sampling/logprob identity: sampling and logprobs N/A; same-image seeds 71/73.
- Dtype/topology/batch identity: BF16 parameters/output, FP32 reduction, SDPA,
  no quantization/KV/TP/offload, FSDP2 mesh `[2]`, reshard false; world 2, K4,
  GA4, one group/four rows per rank per window, global batch 32.
- Parity gates: exactly 32 global rows/samples, eight B8 Qwen calls and 64 cells
  per rank/update, exact sample order/objectives/gradient norms versus RP-19,
  finite state, tokenizer `151669`; 80 focused CPU tests and Ruff passed.
- GPUs: physical `[2,3]` mapped in order to logical `[0,1]`; both reported zero
  memory use and zero utilization at the immediate launch preflight.
- Start/end/session, GPU-hours, scratch, outputs and metrics: torchrun parent
  PID `1866041` launched at `2026-07-20 06:12:16 +09:00` and completed at
  `06:13:23 +09:00` (about 67 seconds wall). The three measured steps used
  `0.0180` aggregate train-core GPU-hours and the full launch stayed below
  `0.0373` aggregate GPU-hours. The output tree occupies `578,014,408` bytes.
  Metrics, final Adapter, step-3 checkpoint, run log, and utilization trace are
  complete; final artifact manifest SHA256
  `e9ebccae60d701b4a3d25c967b609c3bfb49bb0949068baed145504d614e8c9b`.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp22_syncfused_real512_ga4_throughput_gpu23.toml`;
  a read-only 100 ms sampler observes physical 2/3.
- Scorer/parser identity: no answer scorer; strict native representation runner.
- Metrics: steps 1/2/3 took `11.557830680`, `10.449125817`, and
  `10.323950958` seconds. The steps-2/3 steady mean is `10.3865383875`
  seconds (`5.7703` train-core hours for 2,000 steps), `6.39%` faster than
  RP-19. All sample IDs, objective values, gradient norms, counts, and B8 call
  schedules match RP-19 exactly. Maximum rank peak allocated/reserved memory
  was `64,216,335,872`/`73,125,593,088` bytes. In the memory-resident trace,
  physical GPU2/GPU3 mean utilization was `29.5%`/`52.1%`; below-50% samples
  were `70.7%`/`46.7%`, zero samples were `46.3%`/`23.6%`, and longest
  below-50% runs were about `3.14`/`1.53` seconds.
- Conclusion: the fused checks and per-group invariant boundary are accepted;
  they preserve exact training semantics and save `6.39%`. Utilization remains
  discontinuous, so the next cell targets the remaining readout/mask content
  synchronizations and between-step telemetry rather than changing batch math.

### RP-23-QWEN3-REPRESENTATION-READOUTSYNC-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and class: `RP-23`; bounded native readout/mask synchronization
  A/B against RP-22.
- Approval/code: accepted `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`;
  implementation commit `75f219ae285f90cb325f72eb039e970f44af6259`.
- Lifecycle status: `COMPLETE`; result `PASS`.
- Question/baseline/output: do construction-bound native proofs for readout IDs,
  action masks, source positions, Qwen additive-mask/DeepStack layout, and causal
  labels beat RP-22's `10.3865383875`-second steady mean without changing any
  value or gradient? Output is only
  `artifacts/representation/RP-23-qwen3-readoutsync-real512-k4-ga4-throughput-gpu23/`.
- Model/processor/initialization: exact RP-22 stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, no resize,
  `image_max_pixels=262144`; frozen Qwen plus fresh TGVF Adapter seed
  `20260719`, three updates under the 2,000-step horizon.
- N/A: policy/reference, rollout/logprobs, reward, GRPO, SDPO, judge, vLLM
  sampling, KV cache, replay, and asynchronous staleness are absent.
- Code/worktree/config: code commit above; clean committed launch required;
  source/canonical config SHA256
  `3ceb09041b5d8aec6e1106e97420a9184667052acd04738ed03307a267c990fa`/
  `52bec3f3cb20c93c89aa0acdc1c12a67dd9b8a48948c3d4e847980d5fc9debf3`.
- Patch surface: public/generic checked APIs remain fail-closed. Only the native
  streaming path consumes sealed tensor identity/version proofs for tensors
  constructed from already-validated CPU tuples/layouts; mutation/replacement
  fails. Non-log steps also skip exact performance/object-gather telemetry, but
  this RP uses `log_every=1`, so its measurement path is unchanged.
- Data/sample/prompt/template: exact RP-22 K4 real-resolution fixture source
  SHA256 `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  disjoint validation, seeds 71/73; prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`,
  native `tgvf_focus_tool`, template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`.
- D/objective/determinism: exact main D plus `(8,16,24)` D-DeepStack atomic
  columns, M-RoPE/source-key blocking; legacy summed-NLL Matrix CE + L_gen +
  Norm weights `1/1/.1`, manifold zero; frozen eval Qwen, Adapter dropout zero,
  no cache, TF32 off, CUBLAS `:4096:8`.
- Framework/topology/batch: accepted Python3.12/Torch2.9 lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  BF16/FP32 reduction, no quantization/KV/TP/offload, FSDP2 `[2]`, reshard
  false; physical `[2,3]` to logical `[0,1]`; world2, K4, GA4, global batch32.
- Parity gates: exact RP-22 sample order, objectives, gradient norms, 32 rows,
  eight B8 calls/64 cells per rank/update, tokenizer `151669`; 339 integrated
  conditioning/Qwen/representation tests plus Ruff passed.
- Start/end/session, GPU-hours, scratch, outputs/metrics: torchrun parent PID
  `1889757` launched at `2026-07-20 06:29:07 +09:00` and completed at
  `06:30:10 +09:00` (about 63 seconds wall). The three measured steps used
  `0.0173` aggregate train-core GPU-hours and the full launch stayed below
  `0.0350` aggregate GPU-hours. The output tree occupies `577,992,792` bytes.
  Metrics, final Adapter, step-3 checkpoint, run log, and 100 ms utilization
  trace are complete; final artifact manifest SHA256
  `9fb31b1405ba1f806c5ae55221851f5629256c858094902808710c83b7dbe9fd`.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp23_readoutsync_real512_ga4_throughput_gpu23.toml`;
  read-only utilization sampling observes physical 2/3.
- Scorer/parser: no answer scorer; strict native representation runner.
- Metrics: steps 1/2/3 took `11.285435083`, `9.962707680`, and
  `9.927836759` seconds. The steps-2/3 steady mean is `9.9452722195` seconds
  (`5.5252` train-core hours for 2,000 steps), `4.25%` faster than RP-22 and
  `10.36%` faster than RP-19. Sample order, all objective values, gradient
  norms, counts, and eight B8 Qwen calls per rank/update match RP-22 exactly.
  Maximum rank peak allocated/reserved memory was `64,216,334,848`/
  `73,125,593,088` bytes. In the memory-resident trace, physical GPU2/GPU3
  mean utilization was `31.9%`/`35.1%`; below-50% samples were `67.3%`/
  `65.4%`, zero samples were `46.2%`/`45.5%`, and longest below-50% runs were
  about `3.20`/`1.58` seconds.
- Conclusion: the native sealed-proof optimization is accepted because it
  preserves exact training semantics and saves another `4.25%`. The trace is
  still strongly discontinuous: the next cell must target overlap of host
  preparation with device execution or another structural execution gap,
  rather than additional isolated validation checks.

### RP-24-QWEN3-REPRESENTATION-FSDPACCUM-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and class: `RP-24`; bounded FSDP2 gradient-accumulation
  communication A/B against RP-23.
- Approval/code: accepted `RPI-20260720-CONTROL-STACK-OPTIMIZATION` and
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; runtime commit
  `ac13f568061ca893400d343b9ebc9419127e2195` contains the Torch2.9
  `set_requires_gradient_sync`/`set_reshard_after_backward` implementation and
  intentionally excludes the later selective-lm-head patch.
- Lifecycle status: `COMPLETE`; result `SIDE_RESULT`.
- Question/baseline/output: does matching the historical GA4 `no_sync`
  schedule—three local accumulation backwards followed by one FP32
  reduce-scatter—remove the repeated FSDP communication/reshard bubbles and
  beat RP-23's `9.9452722195`-second steady mean? Output is only
  `artifacts/representation/RP-24-qwen3-fsdpaccum-real512-k4-ga4-throughput-gpu23/`.
- Model/processor/initialization: exact RP-23 stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, no resize,
  `image_max_pixels=262144`; frozen Qwen plus fresh TGVF Adapter seed
  `20260719`; three updates under the 2,000-step horizon.
- N/A: policy/reference, rollout/logprobs, reward, GRPO, SDPO, judge, vLLM
  sampling, KV cache, replay, asynchronous staleness, and answer scoring are
  absent from this representation-only diagnostic.
- Code/worktree/config: only committed code may launch; source/canonical TOML
  SHA256 `47df73b040f3caa9422c89b20de50064e5689b28663c1f04a312282a7432a56e`/
  `0629e27d877ca89f6b18873479829d5e2be389163f979b00b6fbe5b629c9239f`.
- Data/sample/prompt/template: exact RP-23 real-resolution K4 throughput
  fixture SHA256 `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  disjoint validation, seeds 71/73, native `tgvf_focus_tool`, prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`,
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`.
- D/objective/determinism: exact main D plus D-DeepStack `(8,16,24)`, native
  positions/masks, legacy summed-NLL Matrix CE + L_gen + Norm weights
  `1/1/.1`, manifold zero; frozen eval Qwen, Adapter dropout zero, no cache,
  TF32 off and CUBLAS `:4096:8`.
- Framework/topology/batch: accepted Python3.12/Torch2.9 lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  FSDP2 `[2]`, forward-to-backward reshard false, non-final backward reshard and
  gradient sync false, final backward both true; physical GPU2/3 UUIDs
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`/
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; world2, K4, GA4, global batch32,
  eight B8 Qwen calls/rank/update.
- Parity gates: exact RP-23 initial Adapter, sample order, step-1 objective
  values, counts, Qwen call schedule and tokenizer length. Gradient/update
  values are not declared bitwise against RP-23 because RP-23 reduced and cast
  every microstep while RP-24 accumulates unsharded FP32 gradients before one
  reduce-scatter; all values must be finite and the numerical delta is
  reported. Any window exception is process-fatal and the Trainer is fail-stop.
- Start/end/session, GPU-hours, scratch, outputs/metrics: torchrun parent PID
  `1920437` launched at `2026-07-20 06:58:24 +09:00` and completed at
  `06:59:29 +09:00` (about 65 seconds wall). The three measured steps used
  `0.0174` aggregate train-core GPU-hours and the full launch stayed below
  `0.0362` aggregate GPU-hours. The output tree occupies `577,981,782` bytes;
  metrics, final Adapter, step-3 checkpoint, run log and utilization trace are
  complete. Final artifact manifest SHA256 is
  `e1062dc24f9b74d91766228659dc43577835f56f9d22a95070f0833fa8a90415`.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_embedding_rp24_fsdpaccum_real512_ga4_throughput_gpu23.toml`;
  a read-only 100 ms sampler observes physical GPUs2/3.
- Metrics/parity: steps 1/2/3 took `11.152782980`, `10.038504998`, and
  `10.052605304` seconds. The steps-2/3 steady mean is `10.045555151` seconds
  (`5.5809` train-core hours for 2,000 steps), `1.01%` slower than RP-23.
  Step-1 sample order, all objective values, counts and eight B8 calls match
  RP-23 exactly. Step-1 gradient norm is `8.115238190` versus RP-23
  `8.115159988` (about `9.6e-6` relative); later losses follow the expected
  distinct FP32-accumulation update path. Maximum rank peak allocated/reserved
  memory is `64,426,703,360`/`73,052,192,768` bytes. In the memory-resident
  trace GPU2/GPU3 mean utilization is `31.8%`/`35.4%`, below-50% samples are
  `67.3%`/`63.6%`, zero samples are `48.7%`/`40.9%`, and longest below-50%
  intervals are about `3.34`/`1.95` seconds.
- Conclusion: retain the FSDP2 accumulation correction because it implements
  the intended GA4 no-sync mathematics, but reject communication/resharding as
  the explanation for the utilization gaps. It neither improves throughput
  nor makes the device timeline more continuous; the next cell isolates
  readout staging/full-vocabulary projection before CPU group prefetch.

### RP-25-QWEN3-REPRESENTATION-SELECTEDHEAD-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and class: `RP-25`; bounded selective language-model-head
  projection A/B against RP-24. Lifecycle status: `COMPLETE`; result `FAIL`
  under the predeclared gradient-parity gate. Output is a diagnostic side
  result and is not eligible for representation-artifact promotion.
- Approval/code/question: accepted
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION` and
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; runtime commit
  `4d2cdd4d0b209d41026200af50b600ec6d45f404`. Does gathering the exact
  evidence prediction hidden states before Qwen's position-wise linear
  `lm_head`, rather than materializing vocabulary logits at every sequence
  position, reduce RP-24's `10.045555151`-second steady step and its measured
  utilization gaps?
- Model/processor/initialization: stable local Qwen3-VL-8B-Thinking,
  BF16/SDPA, tokenizer `151669`, no resize, native template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`; frozen Qwen and fresh TGVF Adapter seed
  `20260719`; target-token-embedding provider.
- Data/prompt: exact RP-24 K4 fixture SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`,
  validation SHA256
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`,
  disjoint split identity, sampler seeds 71/73, native `tgvf_focus_tool`, and
  smoke prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`.
- D/objective/determinism: unchanged complete main D plus D-DeepStack
  `(8,16,24)`, native positions/masks, legacy summed-NLL Matrix CE + L_gen +
  Norm weights `1/1/.1`, manifold zero; frozen eval Qwen, Adapter dropout zero,
  no cache, TF32 off, and CUBLAS `:4096:8`. The selective path changes neither
  evidence labels nor causal position `p-1` and remains differentiable to every
  selected main-D/DeepStack input.
- Framework/topology/batch: accepted Python3.12/Torch2.9 control lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  FSDP2 mesh `[2]`, `reshard_after_forward=false`, corrected GA4 sync schedule,
  physical GPUs 2/3 mapped to logical 0/1, K4, world2, global batch32, eight B8
  Qwen calls/rank/update, three optimizer steps.
- Representation-only N/A contract: no rollout, sampling, policy/reference
  replay, behavior logprobs, reward, GRPO, SDPO, judge, vLLM sampling, KV cache,
  asynchronous staleness, or answer scoring is present in this cell.
- Parity gate: exact RP-24 sample order, global counts, Qwen call schedule,
  tokenizer length and step-1 objective identity are required. Because the
  BF16 GEMM and CE reduction geometry changes, each step-1 objective must have
  absolute delta at most `1e-4` and gradient-norm relative delta at most
  `1e-3`; exact deltas are reported. Failure rejects the fast path.
- Config/output: source/canonical TOML SHA256
  `4f7fa348e7e4382676dfb0bbf7b67d66d0d8ff9ba99c5f5172d7dc8db4dd03d6`/
  `0260057a997920c3ac84875b3e5cfdb1a12ea56c2e79a8335a7c02f17ded7d5d`;
  overwrite is forbidden under
  `artifacts/representation/RP-25-qwen3-selectedhead-real512-k4-ga4-throughput-gpu23/`.
  A read-only utilization sampler records both devices during the run.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/smoke/representation_qwen3_embedding_rp25_selectedhead_real512_ga4_throughput_gpu23.toml`.
- Execution/resources: both physical GPUs were idle at immediate preflight.
  Torchrun started at `2026-07-20 07:08:23 +09:00` and completed at about
  `07:09:30 +09:00`; the three measured steps used approximately `0.016881`
  aggregate train-core GPU-hours. The output tree occupies `577,982,664`
  bytes and contains metrics, a final diagnostic Adapter, step-3 checkpoint,
  run log, and utilization trace. Final artifact manifest SHA256 is
  `7ebd6556e585530a09bf029b074ebff21d26c2ddcdeb20f7a7ee6261b45c2381`.
- Metrics/identity: steps 1/2/3 took `10.805963864`, `9.800602207`, and
  `9.778940455` seconds. The steps-2/3 steady mean is `9.789771331` seconds,
  a `2.5462%` step-time reduction versus RP-24 (`5.4388` train-core hours for
  2,000 steps). Sample order, row/sample counts, tokenizer length, and eight
  B8 calls/rank/update match RP-24. Step-1 Matrix CE/L_gen/Norm deltas are
  approximately `-1.058e-6`/`-1.788e-7`/`0`, all inside the objective gate.
- Parity failure/conclusion: step-1 gradient norm is `8.156969070` versus
  RP-24's `8.115238190`, a `0.514229%` relative delta that exceeds the frozen
  `0.1%` gate. Maximum-rank peak allocated/reserved memory fell to
  `55,437,285,376`/`57,554,239,488` bytes, but the timeline remained bursty:
  GPU2/GPU3 mean utilization was `30.5%`/`37.7%`, below-50% samples were
  `69.3%`/`62.5%`, zero samples were `41.8%`/`37.5%`, and longest below-50%
  spans were about `3.23`/`1.54` seconds. Reject this selective-head execution
  path despite its modest speed/memory gain; full-vocabulary projection was
  not the primary utilization-gap cause. The next cell targets one-group-ahead
  CPU preparation while retaining the accepted full-logits mathematics.

### RP-26-QWEN3-REPRESENTATION-PREFETCH-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and class: `RP-26`; bounded one-group-ahead host-preparation
  A/B against the accepted full-logits RP-24 baseline. Lifecycle status:
  `COMPLETE`; result `SIDE_RESULT_REJECTED`. Output is diagnostic and is not
  eligible for representation-artifact promotion.
- Approval/code/question: accepted
  `RPI-20260720-CONTINUOUS-REPRESENTATION-EXECUTION`; runtime commit
  `857b3a96afeaab03342f83165ee0d5c864c204ba`. Does overlapping group
  `i+1` image decode/hash, native transcript/tokenization and Qwen processor
  work with group `i` device materialization/readout reduce RP-24's
  `10.045555151`-second steady step and long below-50% utilization spans?
- Execution boundary: the sampler fixes all four local K4 groups before the
  step. One step-scoped worker holds at most one CPU-only prepared group;
  ordering is `result_i -> submit(i+1) -> materialize_i`. H2D, frozen vision,
  target conditioning, TGVF Adapter, M-RoPE, Qwen readout and all autograd stay
  on the trainer thread. Every future is consumed or cancel/drained before the
  optimizer-step boundary; no prefetch state crosses checkpoint/resume.
- Model/data/prompt/initialization: exact RP-24 stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, no resize, native
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`, target-token-embedding provider, fresh Adapter
  seed `20260719`; train/validation SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`,
  seeds 71/73, disjoint split, native `tgvf_focus_tool`, prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`.
- D/objective/determinism: unchanged full-vocabulary logits, complete main D
  plus D-DeepStack `(8,16,24)`, native positions/masks, legacy summed-NLL
  Matrix CE + L_gen + Norm weights `1/1/.1`, manifold zero; frozen eval Qwen,
  Adapter dropout zero, no cache, TF32 off, CUBLAS `:4096:8`.
- Framework/topology/batch: accepted Python3.12/Torch2.9 lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  FSDP2 mesh `[2]`, `reshard_after_forward=false`, corrected GA4 sync schedule,
  physical GPUs2/3 mapped logical0/1, world2, K4, global batch32, eight B8 Qwen
  calls/rank/update, three optimizer steps.
- N/A: no rollout, sampling, policy/reference replay, behavior logprobs,
  reward, GRPO, SDPO, judge, vLLM sampling, KV cache, staleness, or answer
  scoring occurs in this representation-only cell.
- Parity/verification gate: 38 focused CPU tests plus Ruff passed before
  planning, including prepared-versus-synchronous exact tensor/score parity,
  true one-ahead overlap, thread ownership, ordering, fallback and fail-stop
  drain. RP-24 sample order, counts, Qwen schedule, tokenizer length, every
  step-1 objective and gradient norm must match exactly; any mismatch rejects
  the prefetch path.
- Config/output: source/canonical TOML SHA256
  `c470424402e02aacabd3f3cabee198f1a796d958870ab939e31a18fabc8136d3`/
  `d88678815f54eefb74b4e9dfa36b45af2004aaf4d62c5077077661972f33a85b`;
  overwrite is forbidden under
  `artifacts/representation/RP-26-qwen3-prefetch-real512-k4-ga4-throughput-gpu23/`.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/smoke/representation_qwen3_embedding_rp26_prefetch_real512_ga4_throughput_gpu23.toml`;
  a read-only utilization sampler records physical GPUs2/3.
- Execution/resources: both GPUs were idle at immediate preflight. Torchrun
  started at `2026-07-20 07:23:25 +09:00` and completed at about
  `07:24:29 +09:00`; measured train-core use was approximately `0.017669`
  aggregate GPU-hours. The output tree occupies `577,979,806` bytes and is
  complete; final diagnostic artifact manifest SHA256 is
  `bfaa43444052e9470b9f5f88d0c9da6814121567216f174f31c727b9abf3992c`.
- Metrics/parity: steps 1/2/3 took `11.231770953`, `10.239523763`, and
  `10.333306422` seconds. Sample order, all objective values, gradient norms,
  counts, tokenizer length, peak allocation/reservation, and eight B8 Qwen
  calls/rank/update match RP-24 exactly. The predeclared exact mathematics gate
  therefore passes.
- Throughput/utilization/conclusion: the steps-2/3 steady mean is
  `10.286415093` seconds (`5.7147` train-core hours for 2,000 steps), `2.3977%`
  slower than RP-24. GPU2/GPU3 mean utilization was `32.0%`/`35.9%`, below-50%
  samples were `67.1%`/`63.9%`, zero samples were `54.8%`/`46.5%`, and longest
  below-50% spans were about `3.34`/`2.72` seconds. Reject the prefetch path:
  real processor/template work is not the dominant utilization gap and worker
  scheduling/contention makes the update slower. The next diagnostic must use
  an operator timeline instead of another inferred bottleneck.

### RP-27-QWEN3-REPRESENTATION-OPERATOR-PROFILE-REAL512-K4-GA4-GPU23

- Cell/matrix ID and class: `RP-27`; bounded operator-timeline diagnostic on
  the accepted RP-24 full-logits path. Lifecycle status: `COMPLETE`; result
  `ROOT_CAUSE_FOUND`. No timing from the profiled step is used as a throughput
  estimate and its Adapter output is not promotion-eligible.
- Approval/code/question: accepted
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION` measured-attribution clause;
  diagnostic launcher/runtime commit
  `e642145eeab45dafb2a3ff51301e89a633fc0bfd`. Which high-level scope and CUDA
  operators contain the observed 1--3 second below-50% spans after RP-25 and
  RP-26 excluded full-vocabulary projection and host prefetch as primary causes?
- Profile contract: step 1 is an unprofiled warm update; step 2 alone records
  PyTorch CPU/CUDA activities and per-rank Chrome traces. Scoped CUDA events
  cover group builder, processor action, frozen vision, target conditioning,
  Adapter forward, readout-row/M-RoPE, each Qwen cell batch, group score and
  Adapter backward. The profiler changes timing only, not inputs, operations,
  objectives, RNG, optimizer ordering, or checkpoint state.
- Model/data/mathematics: exact RP-24 stable local Qwen3-VL-8B-Thinking,
  BF16/SDPA, target-token-embedding, tokenizer `151669`, native template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`, K4 fixture/validation SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`,
  prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`,
  fresh Adapter seed `20260719`, full main D + DeepStack `(8,16,24)`, legacy
  summed-NLL Matrix CE/L_gen/Norm weights `1/1/.1`, manifold zero.
- Determinism/topology: frozen eval Qwen, Adapter dropout zero, no cache, TF32
  off, CUBLAS `:4096:8`; accepted Python3.12/Torch2.9 lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`,
  FSDP2 mesh `[2]`, reshard false, corrected GA4 sync, physical GPUs2/3,
  world2, global batch32, eight B8 calls/rank/update, two optimizer steps.
- N/A: rollout, policy/reference replay, behavior logprobs, reward, GRPO, SDPO,
  judge, vLLM sampling, KV cache, asynchronous staleness, and answer scoring.
- Parity gate: unprofiled step-1 sample order, objectives, gradient, counts,
  Qwen schedule and tokenizer length must match RP-24 exactly. Step 2 must be
  finite and complete; profiler overhead is expected and not compared to the
  baseline. Ruff, format and syntax checks passed for the diagnostic launcher.
- Config/output: source/canonical TOML SHA256
  `0904bc81a27ae6d13016ae0bcdb4bf6747f2b98b03b1fae064ddcdacc65fe784`/
  `1f9975f302ecb33e9dac6d183c239fccc760b9538df4e4602a241b5e76b4b4c4`;
  overwrite forbidden under
  `artifacts/representation/RP-27-qwen3-operator-profile-real512-k4-ga4-gpu23/`.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2
  tools/profile_representation_step.py
  configs/smoke/representation_qwen3_embedding_rp27_operator_profile_real512_ga4_gpu23.toml
  --profile-global-step 2 --trace-dir
  artifacts/representation/RP-27-qwen3-operator-profile-real512-k4-ga4-gpu23/profile`.
- Execution/resources: both devices were idle at preflight; torchrun started
  `2026-07-20 07:30:37 +09:00` and completed about `07:31:59 +09:00`.
  Train-core use was approximately `0.012219` aggregate GPU-hours. The output
  tree is `1,026,602,351` bytes, including complete per-rank summaries,
  operator tables and 223.9/224.7 MB Chrome traces. Final diagnostic Adapter
  manifest SHA256 is
  `abbd1f418c5833fa5d6f1bfd76eacf0678cefcbcd37bd52801bf91fd4d62aade`.
- Parity/profile validity: unprofiled step 1 took `11.153583912` seconds and
  matches RP-24 exactly in sample order, every objective, gradient norm,
  counts, tokenizer length and eight B8 calls. Profiled step 2 completed in
  `10.840172723` seconds with exact RP-24 mathematical values; its timing is
  instrumentation-contaminated as planned.
- Root cause: across 16 readout rows/rank, `_readout_row` consumed
  `3,180.4` ms on rank0 and `3,900.8` ms on rank1, while profiler-attributed
  CUDA work inside that scope was only about `25.9`/`27.2` ms. Each call spent
  roughly 190--252 ms invoking Qwen3 `get_rope_index` on CUDA IDs. The upstream
  helper performs Python/list/scalar operations plus `argwhere`, `tolist` and
  repeated grid `item` reads; CUDA inputs therefore create many tiny kernels
  and device-to-host synchronizations although M-RoPE depends only on discrete
  IDs, mask and image grid.
- Secondary effect/conclusion: rank1's 720 ms larger cumulative readout delay,
  plus longer readout sequences, arrives late at the final synchronized
  backward; rank0 attributes about `1.328` seconds to the final NCCL
  reduce-scatter wait. Group builder spans were `5,979.1`/`7,031.1` ms,
  Qwen cell batches `1,218.3`/`1,296.3` ms, and score groups
  `3,222.5`/`3,496.3` ms. The next correction computes the unchanged native
  M-RoPE positions on CPU and transfers the single completed position tensor,
  then requires exact real-model position/objective/gradient parity.

### RP-28-QWEN3-REPRESENTATION-CPU-MROPE-REAL512-K4-GA4-THROUGHPUT-GPU23

- Cell/matrix ID and class: `RP-28`; bounded native M-RoPE placement A/B
  against RP-24 following the RP-27 root-cause profile. Lifecycle status:
  `COMPLETE`; result `PASS_RETAINED`. The code path is retained; this three-step
  output remains diagnostic rather than a promoted representation artifact.
- Approval/code/question: accepted
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION`; runtime commit
  `e461c66bd9916c97ff170679e51739922bb2fbab`. Does invoking the unchanged
  Qwen3 `get_rope_index` on CPU IDs/mask/grid and transferring its one finished
  position tensor eliminate the 3.2--3.9 seconds of readout-row synchronization
  spans identified by RP-27?
- Change boundary: only the device placement of discrete M-RoPE construction
  changes. The exact same model-owned helper, token IDs, all-one mask, two-image
  grid, output dtype/shape and final positions are used. Frozen vision, target
  conditioning, TGVF Adapter, main D/DeepStack, Qwen readout and all autograd
  remain on the original GPU path. No alternative position formula is ported.
- Model/data/prompt/initialization: exact RP-24 stable local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer `151669`, no resize, template
  SHA256 `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`, target-token-embedding, fresh Adapter seed
  `20260719`; train/validation SHA256
  `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d`/
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`,
  seeds 71/73, disjoint split, native `tgvf_focus_tool`, prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`.
- Mathematics/determinism: unchanged complete main D plus D-DeepStack
  `(8,16,24)`, native masks, full logits, legacy summed-NLL Matrix CE + L_gen +
  Norm `1/1/.1`, manifold zero; frozen eval Qwen, Adapter dropout zero, no
  cache, TF32 off, CUBLAS `:4096:8`.
- Framework/topology/batch: accepted Python3.12/Torch2.9 lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`;
  FSDP2 `[2]`, reshard false, corrected GA4 sync, physical GPUs2/3,
  logical0/1, world2, K4, global batch32, eight B8 calls/rank/update, three
  optimizer steps.
- N/A: no rollout, policy/reference replay, behavior logprobs, sampling,
  reward, GRPO, SDPO, judge, vLLM sampling, KV cache, staleness, or scoring.
- Parity/verification gate: 22 native/golden CPU tests plus Ruff passed. RP-24
  sample order, every objective, gradient norm, counts, Qwen schedule and
  tokenizer length must match exactly at step 1; later trajectory values must
  then remain exact. Any mismatch rejects CPU M-RoPE. Timing, memory and a
  100-ms physical-device utilization trace are recorded independently.
- Config/output: source/canonical TOML SHA256
  `8d991d7ccdc0e69c09b631f73f702d4130c773fde458a678f599f1141b54e2ba`/
  `686b21f79c7bbf61356b13f48efb1ad4cb611f7dbbc86cbadebda4db8db2742b`;
  overwrite forbidden under
  `artifacts/representation/RP-28-qwen3-cpu-mrope-real512-k4-ga4-throughput-gpu23/`.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/smoke/representation_qwen3_embedding_rp28_cpu_mrope_real512_ga4_throughput_gpu23.toml`.
- Execution/resources: both GPUs were idle at immediate preflight; torchrun
  started `2026-07-20 07:36:48 +09:00` and completed about
  `07:37:49 +09:00`. Train-core use was approximately `0.016381` aggregate
  GPU-hours. The complete output tree occupies `577,975,929` bytes; final
  diagnostic Adapter manifest SHA256 is
  `d23b4a3cebc8c49fd33cdb9cb511333739958a93670fa0d161960857bec3d754`.
- Parity: all three steps match RP-24 exactly in sample order, Matrix CE,
  L_gen, Norm, total objective, gradient norm, counts, tokenizer length and
  eight B8 Qwen calls/rank/update. This is end-to-end real-model evidence that
  CPU and CUDA invocation of the same native helper produced identical
  positions on every exercised transcript.
- Timing/memory: steps 1/2/3 took `10.614827115`, `9.439143186`, and
  `9.431757647` seconds. The steady mean is `9.435450417` seconds (`5.2419`
  train-core hours for 2,000 steps), a `6.0734%` step-time reduction and
  `6.4661%` throughput gain versus RP-24. Peak allocated/reserved memory and
  resident allocations remain unchanged.
- Utilization/conclusion: GPU2/GPU3 mean utilization rose to `33.8%`/`36.0%`;
  below-50% samples were `66.0%`/`63.3%`, zero samples fell to `38.1%`/`36.1%`,
  and longest below-50% spans were about `3.16`/`1.53` seconds. Retain CPU
  M-RoPE because it is exact and materially faster, but it does not fully
  eliminate the bursty timeline. Profile the retained path before selecting
  another optimization.

### RP-29-QWEN3-REPRESENTATION-POST-MROPE-PROFILE-REAL512-K4-GA4-GPU23

- Cell/matrix ID/class/status: `RP-29`; `COMPLETE` bounded operator profile of
  the retained RP-28 CPU-M-RoPE path. It is diagnostic only; profiled timing
  and Adapter output are not promotion-eligible.
- Approval/code/question: accepted control-stack measured attribution;
  diagnostic/runtime commit `026f44ea2a657037510b78381729f7b43324194e`.
  After the exact 6.07% RP-28 gain, which remaining group-builder sub-scope
  causes the longest utilization gap?
- Profile contract: exact RP-28 model/data/prompt/objective/seed/topology and
  global batch32. Step 1 is unprofiled and must match RP-28 exactly; step 2
  captures CPU/CUDA traces. Added scopes isolate runtime invariant scans,
  message/action/evidence rendering, processor/shared-visual action, source
  identity, supervision, CPU M-RoPE, vision, conditioning, Adapter, Qwen cell
  batches, score and backward. Instrumentation affects timing only.
- Fixed identities: Qwen3-VL-8B-Thinking BF16/SDPA, tokenizer `151669`, native
  template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`, target-token-embedding, main D + DeepStack
  `(8,16,24)`, legacy Matrix CE/L_gen/Norm `1/1/.1`, manifold zero, K4/GA4,
  FSDP2 world2 GPUs2/3, CUBLAS `:4096:8`, no cache/dropout/TF32.
- Representation-only N/A: rollout, sampling/logprobs, replay, reward, GRPO,
  SDPO, judge, vLLM, KV cache and staleness.
- Config/output: source/canonical SHA256
  `2ce677846ccd9536c36fc476e7c13ee81abb6b02c46b38496dd5cbe737c99358`/
  `cab0926fc6de54ac34ab66002e541285e72a928270889e34015a895376c27b93`;
  overwrite forbidden under
  `artifacts/representation/RP-29-qwen3-post-mrope-profile-real512-k4-ga4-gpu23/`.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2
  tools/profile_representation_step.py
  configs/smoke/representation_qwen3_embedding_rp29_post_mrope_profile_real512_ga4_gpu23.toml
  --profile-global-step 2 --trace-dir
  artifacts/representation/RP-29-qwen3-post-mrope-profile-real512-k4-ga4-gpu23/profile`.
- Result: `PASS` for attribution. The run completed two optimizer steps on
  physical GPUs 2/3; steps 1/2 took `10.992597855`/`10.404267961` seconds.
  Both steps preserved the RP-24/RP-28 sample order, losses, counts, gradient
  norms, tokenizer length and eight B8 Qwen calls per rank. The final Adapter
  manifest is
  `8ded17e9aeac3c8363abba369e65d9a28d8854dd60cdc7e7ef98f578c423d63c`.
- Attribution: the instrumented four-group step attributed `6737.5`/`6609.8`
  ms on rank 0/rank 1 to group construction. The profiler's nested CUDA events
  inflate fine-grained wall scopes, so those individual values are not treated
  as timing estimates. Independent CPU measurement and exact call accounting
  identify the blocking operation inside them: Qwen's fast-tokenizer
  `len(tokenizer)` took `17.58` ms and was invoked approximately 328 times per
  GA4 step, predicting roughly `5.77` seconds of host stalls. CPU M-RoPE,
  vision and Adapter are no longer plausible causes. RP-30 directly tests the
  exact cheaper tokenizer-invariant path without profiling instrumentation.
- Resource accounting: train-core usage was approximately `0.01189` GPU-hours;
  the complete diagnostic tree is `1023592362` bytes, including the two raw
  traces (`222306700` and `223261118` bytes). Profiled timing is diagnostic and
  is not a throughput estimate. Next action is exact batched transcript
  rendering/tokenization, followed by an unprofiled timing cell.

### RP-30-QWEN3-REPRESENTATION-FAST-TOKENIZER-INVARIANT-REAL512-K4-GA4-GPU23

- Cell/matrix ID and class: `RP-30`; `COMPLETE`, result `PASS_RETAINED`, bounded
  unprofiled throughput and utilization A/B against retained RP-28. The output
  is diagnostic rather than a promoted representation artifact.
- Approval/code/question: accepted
  `RPI-20260720-CONTROL-STACK-OPTIMIZATION`; runtime commit
  `1d3d37c88404446545b8a43014c8bbbfb2bd7716`. Does replacing repeated Qwen
  fast-tokenizer merged-vocabulary materialization with an exact contiguous-
  suffix cardinality proof remove the multi-second CPU gaps and make physical
  GPU utilization materially more continuous?
- Change boundary: the native tokenizer-length invariant still checks the base
  vocabulary and complete added-token suffix on every call. Only a proven
  contiguous added-token ID range uses the fast cardinality path; unfamiliar
  layouts fall back to native `len(tokenizer)`. Same-image K=4 action/evidence
  conversations are rendered/tokenized in ordered batches while preserving
  the scalar APIs and every transcript byte, token ID, offset, ownership span,
  label and identity hash. No model, Adapter, objective, optimizer, FSDP2 or
  CUDA operation changes.
- CPU attribution/parity gate: the accepted Qwen tokenizer reports base size
  `151643` plus contiguous added IDs `151643..151668`, exactly reproducing
  length `151669`. On the local processor, native `len(tokenizer)` measured
  `17.58` ms/call versus `0.0108` ms for the dynamic exact helper; real-fixture
  supervision fell from about `51.78` to `0.714` ms/call. Thirty-nine focused
  tests, including real-Qwen scalar/batch golden equality, passed with Ruff,
  format and diff checks. The GPU run must exactly match RP-28 step-by-step
  sample order, Matrix CE, L_gen, Norm, total objective, gradient norm, counts,
  tokenizer length, and eight B8 Qwen calls per rank/update.
- Model/data/prompt: exact RP-28 stable local Qwen3-VL-8B-Thinking BF16/SDPA,
  tokenizer/template identity, `image_max_pixels=262144`, target-token-
  embedding provider, train/validation fixture identities and sampler seeds
  71/73, native `tgvf_focus_tool`, and smoke-only prompt SHA256
  `ea2fb166448a2fb7af33017da635d85fe717265987e4c7073b588c443670ffd3`.
- Mathematics/determinism: unchanged complete main D plus D-DeepStack
  `(8,16,24)`, legacy summed-NLL Matrix CE + L_gen + Norm weights `1/1/.1`,
  manifold zero, frozen eval Qwen, fresh Adapter seed `20260719`, Adapter
  dropout zero, no cache, TF32 off, and CUBLAS `:4096:8`.
- Framework/topology/batch: accepted Python3.12/Torch2.9 control lock, FSDP2
  mesh `[2]`, `reshard_after_forward=false`, corrected GA4 synchronization,
  physical GPUs2/3 mapped to logical0/1, world2, K4, global batch32, three
  optimizer steps. A read-only utilization sampler records both physical GPUs.
- Representation-only N/A: rollout, sampling/logprobs, policy/reference replay,
  reward, GRPO, SDPO, judge, vLLM sampling, KV cache, asynchronous staleness,
  and answer scoring.
- Config/output: source/canonical TOML SHA256
  `0e2958e731b3b307cb8bab80f6cea10e9147e408fcd955967a46846ecb2c8ec4`/
  `634cc72dcf0a0ef2684fbe5a6b9ac3d8b5fc69fb3cadfc54daed476f145990e2`;
  overwrite is forbidden under
  `artifacts/representation/RP-30-qwen3-fast-tokenizer-invariant-real512-k4-ga4-gpu23/`.
- Planned command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/smoke/representation_qwen3_embedding_rp30_fast_tokenizer_invariant_real512_ga4_gpu23.toml`.
- Execution/resources: both physical GPUs were idle at preflight. Torchrun ran
  from `2026-07-20 07:55:26 +09:00` through `07:56:15 +09:00`. The three
  train-core steps used approximately `0.007254` aggregate GPU-hours; the
  complete output tree is `577958795` bytes. Final diagnostic Adapter manifest
  SHA256 is
  `43faca0148e8019bd4990645a51536825ea0179346a6263e853922983541b22c`.
- Exact parity: all three steps match RP-28 exactly in sample order, Matrix CE,
  L_gen, Norm, total objective, gradient norm, counts, tokenizer length and the
  eight B8 Qwen calls/rank/update. The final 104 Adapter tensors and all tensor
  SHA256 values are also exactly equal to RP-28; only run-bound artifact
  metadata changes.
- Timing: steps 1/2/3 took `4.943561714`, `4.062893931`, and `4.050138547`
  seconds. The steps-2/3 steady mean is `4.056516239` seconds: a `57.0077%`
  reduction and `2.3260x` throughput versus RP-28. At this measured train-core
  rate, 2,000 optimizer steps take approximately `2.2536` hours before periodic
  validation/checkpoint overhead.
- Utilization/conclusion: over the memory-resident train window, GPU2/GPU3 mean
  utilization rose to approximately `84.5%`/`83.8%`; below-50% samples fell to
  `14.8%`/`13.1%`, zero samples to `0%`/`1.6%`, and the longest below-50% spans
  to about `0.36`/`0.18` seconds. Retain both the exact fast tokenizer
  invariant and ordered batch transcript path. The repeated merged-vocabulary
  scans were the principal cause of the previously discontinuous GPU usage.

### REP-QWEN3-V4-CONTEXTUAL-V1-PREFLIGHT

- Cell/matrix ID and mandatory/diagnostic class: representation-phase formal
  configuration preflight; mandatory real-Qwen/provider wiring and gradient
  check, bounded to the exact run's first 10 optimizer steps.
- Spike-plan git revision and VA0/VA1/VA2 approval references: representation
  contract in `PROJECT_TASK.md`; user-approved implementation continuation on
  2026-07-20 JST. This is not a veRL compatibility-spike cell.
- Lifecycle status: COMPLETE
- Result: PASS
- Question: can the accepted contextual-hidden-state provider, selected formal
  data, native v1 prompt, Balanced Matrix CE objective and real Qwen3/TGVF
  Adapter execute a nonzero-gradient optimizer update under the production
  two-rank geometry?
- Baseline and exact output path:
  `artifacts/representation/REP-QWEN3-V4-CONTEXTUAL-V1/`; operational stop at
  optimizer step 10, with no final Adapter publication.
- Model and processor identity: local `Qwen3-VL-8B-Thinking`, tokenizer length
  151669, chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  BF16, SDPA, `image_max_pixels=262144`, no tokenizer resize.
- Representation checkpoint identity: fresh TGVF Adapter seed 42; contextual
  hidden state layer `-1`; no legacy or other source checkpoint; DCP v2 strict
  optimizer-boundary checkpoint. The bounded invocation is resumable but is
  not a promoted representation artifact.
- N/A fields and justification: rollout, policy/reference/teacher replay,
  reward, scorer, sampling and logprob fields are N/A because this is
  representation training, not policy RL or SDPO.
- Policy/reference initialization: N/A; frozen base Qwen, trainable TGVF
  Adapter only.
- Rollout policy version and allowed asynchronous staleness: N/A; synchronous
  teacher-forced representation update.
- Code commit and worktree state: code identity
  `f32470cc0ffca49ec2834757389ed8e2e065d90e`, clean across the runner's bound
  code paths; config source SHA256
  `2e7e1ea09174a0b435d1009a38c8d7d901e14787850698b347b3a165efe16faf`,
  canonical config SHA256
  `eb42fb3cb6873fe20b3c17d67306f569343096e4abc63f3088410892efdcef6e`.
- Repository adapter/patch surface and hash: repository-owned Qwen3 runtime,
  family adapter, TGVF Adapter, native representation builder, streaming loss,
  trainer and DCP runner at the code commit above; no runtime patch.
- Dataset/manifest, hashes, sample rule, and n: v4 clean-imend train source
  `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`,
  retained manifest
  `a089d13d46bb74d1870aee8dc9b056925c4524164233d11e81d0edb9a5b45a8c`,
  n=39998; v3 val-2k source
  `a228d28db76625d166dab874806c9034a244a683d41c7cecdc7f10f1aa754308`,
  retained manifest
  `f47bbff7c63ffa381ce2e2e263130c783057c6d408575ef4c4e3dd5b019c5a33`,
  n=1382; exact seven-image-path overlap report
  `e79a939ae8a757e42f49377b07d224338071396667e1a3330b91a35a3057987c`
  is recorded and accepted without row filtering.
- Native prompt/tool schema hash: `native_representation_prompt_v1`, prompt
  identity `qwen3-representation-image-question-v1`, exact `{question}` SHA256
  `bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9`;
  native `tgvf_focus_tool` action and native tool-response image.
- Chat-template/token-fixture hash and token-ownership masks: accepted local
  Qwen chat-template hash above; canonical action/target spans and evidence
  label masks from the bound native representation pipeline; choices remain
  non-rendered data identity metadata.
- D/DeepStack/position/mask identity: TGVF Adapter main D and three Qwen3
  D-DeepStack branches, native source-grid positions, two-visual-block causal
  evidence readout and repository-bound masks.
- Observation materialization/artifact identity used by all replays: N/A for RL
  replay; every Matrix CE column uses the one materialized complete candidate
  observation for that target in the current same-image group.
- RL framework/version/environment lock: N/A for veRL; repository `.venv312`
  control stack with Torch 2.9.0/CUDA 12.9 compatibility lock.
- Objective equations and normalization: Balanced Matrix CE score
  `-(evidence NLL sum / valid evidence-token count) / 1.0`, row-wise CE with
  diagonal targets and global valid-row mean; L_gen is the global sample mean
  of per-sample mean evidence NLL; historical norm formula at weight 0.1;
  weights Matrix CE/L_gen/Norm = 1.0/1.0/0.1; manifold contribution zero.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  teacher-forced forward, frozen Qwen, fresh Adapter seed 42, no asynchronous
  update and deterministic runner settings.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A; same-image sampler seed 42,
  no generation sampling in the training update.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights/outputs, FP32 FSDP reduce,
  no quantization or KV-cache rollout, SDPA; two-rank `fsdp=[2]`, no offload,
  `reshard_after_forward=false`.
- Logit/logprob/loss/gradient parity tolerances: finite scalar/gradient and
  nonzero required Adapter gradients; previously accepted synthetic functional
  parity remains the numerical oracle. RL logprob parity is N/A.
- World size, microbatch, accumulation, and global batch: world size 2, one K=4
  same-image matrix per rank/microstep, GA=4, eight matrices and 32 rows per
  optimizer step.
- GPUs: physical GPUs 2 and 3 only; logical CUDA devices 0 and 1.
- Start/end timestamps, elapsed time, and session/process identity: torchrun
  PID 2220272 started `2026-07-20 12:59:30 +09:00` and exited zero at
  `13:01:11 +09:00`; 101.1 seconds wall time.
- Actual GPU-hours and peak scratch use: approximately 0.0562 aggregate
  GPU-hours by two-device wall time. Rank peak allocated memory was
  42,000,555,520 / 63,945,762,304 bytes and peak reserved memory was
  93,935,632,384 / 80,717,283,328 bytes. The exact output tree is 403 MiB.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 1800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/representation/qwen3_v4_contextual_hidden_state_v1.toml
  --stop-after-global-step 10`.
- Outputs: exact step-10 DCP at
  `checkpoints/representation-qwen3-v4-contextual-v1-step-00000010`, metrics
  JSONL and `run-step10.log` under the output root above. DCP contains both
  rank shards, metadata and metadata SHA256. A CPU sidecar reload verified DCP
  v2, step 10, world size 2, the exact run identity and all 104 owned tensors;
  no final Adapter was published.
- Scorer/parser identity: native representation evidence supervision and
  strict tool-call parser at the code commit above; external scorer N/A.
- Metrics: status `paused_at_optimizer_boundary`, global step 10, run identity
  `4d0493ea3cfa052abf66f336f357ad015f7dbacee64ee54d7efe0c085e6969f3`;
  tokenizer stayed 151669. Step 10 processed 8 matrices / 32 rows in
  4.725605814 seconds (1.6929 matrices/s, 6.7716 rows/s); Matrix CE
  `1.3847746104`, L_gen `2.9906668067`, Norm `0.3271957580`, weighted Norm
  `0.0327195758`, total `4.4081609929`, and pre-clip gradient norm
  `1.5775129795`, all finite.
- Conclusion: `PASS` for formal contextual-provider wiring, exact selected-data
  admission, Balanced Matrix CE/L_gen/Norm backward, two-rank K4/GA4 update and
  durable step-10 checkpoint. This is a bounded resumable prefix, not the
  2,000-step result or formal internal-evaluation promotion.

### BJ-10-QWEN25-72B-VLLM-SERVICE

- Cell/matrix ID and mandatory/diagnostic class: `BJ-10`; bounded deployment
  smoke for the accepted CoreDev VLMEvalKit benchmark judge.
- Spike-plan git revision and VA0/VA1/VA2 approval references: user authorized
  Qwen2.5-72B deployment and continuation on 2026-07-20 JST; accepted evaluation
  architecture `EVAL-ARCH-20260720` plus Project Task §0.5.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: can the exact local `Qwen/Qwen2.5-72B-Instruct` snapshot load under
  the accepted vLLM/Torch stack with TP=2 on physical GPUs 2 and 3, expose the
  slash-free `Qwen2.5-72B-Instruct` OpenAI-compatible model identity, and return
  one deterministic nonempty chat completion with the exact requested text?
- Baseline and exact output path: no competing judge backend; output root
  `artifacts/evaluation/BJ-10-qwen25-72b-vllm-service-53b5c66`, server log
  `server.log`, snapshot validation `snapshot_validation.json`, service smoke
  `smoke.json`, and GPU sample `gpu.csv`.
- Model and processor identity: Hugging Face
  `Qwen/Qwen2.5-72B-Instruct` revision
  `495f39366efef23836d0cfae4fbe635880d2be31`, local path
  `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct`; 37 BF16 shards,
  145,412,519,312 weight bytes, tokenizer length 151665, chat-template SHA256
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Representation checkpoint identity: N/A; this is a text-only benchmark judge
  and does not load a TGVF Adapter.
- N/A fields and justification: policy/reference initialization, rollout policy
  version/staleness, behavior log probabilities, D/DeepStack observation,
  replay, reward, GRPO, SDPO, optimizer, gradient and checkpoint are N/A because
  this cell performs one inference-only service health request and no scoring.
- Policy/reference initialization: N/A as above; the judge is forbidden from
  acting as either role.
- Rollout policy version and allowed asynchronous staleness: N/A; no rollout or
  policy update.
- Code commit and worktree state: runtime code commit
  `53b5c665e937e5c5e67949e01e08510e2d97ea27`; launch worktree must be clean
  except for this subsequently committed ledger plan.
- Repository adapter/patch surface and hash: service config SHA256
  `d2437fd7165ed0ead92feb3191009873d017e80eb550640d995c68d1a25d328b`;
  smoke script SHA256
  `b7dea272a2eb2a4bc98257a5aec48d2004f0c10afe718a4e2c3655d0ec1c4ca7`;
  no external-checkout or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: synthetic one-request health
  fixture, `n=1`; no benchmark rows are evaluated.
- Native prompt/tool schema hash: no native tool schema; canonical request JSON
  SHA256 `30486f0066f51a616e4cf6d53403b067c6346abcbb017f414369ebe13c7e767e`.
- Chat-template/token-fixture hash and token-ownership masks: model chat-template
  hash above; token-ownership masks N/A because no training objective exists.
- D/DeepStack/position/mask identity: N/A; text-only judge.
- Observation materialization/artifact identity used by all replays: N/A; no
  replay.
- RL framework/version/environment lock: vLLM `0.12.0`, Torch `2.9.0+cu128`,
  Transformers `4.57.6`, Python `3.12.3`; compatibility lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: N/A; no loss or update.
- Rollout/replay forward mode and adapter dropout/RNG contract: vLLM V1 server,
  prefix caching enabled, model eval inference; no Adapter or replay.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.12.0`; request seed `42`,
  temperature `0`, maximum 32 output tokens; all other sampling controls use
  vLLM greedy defaults because `--generation-config vllm`; no custom logit
  processor and no log probabilities requested.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights, auto KV dtype, no
  quantization, vLLM-selected attention, TP=2, max model length 32768,
  GPU-memory utilization 0.85, maximum 64 sequences, no training mesh.
- Logit/logprob/loss/gradient parity tolerances: N/A; success requires exact
  served-model identity plus response text `TGVF_JUDGE_READY` and a normal
  finish reason.
- World size, microbatch, accumulation, and global batch: TP world `2`, one
  request/completion; batch and accumulation N/A.
- GPUs: physical `2` = `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and physical
  `3` = `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; both NVIDIA B200 183359
  MiB; logical mapping `2->0`, `3->1`.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T16:30:26+09:00` / `2026-07-20T16:31:32+09:00`, 66 seconds;
  API PID 2460166, engine PID 2461196, worker PIDs 2461730 and 2461731.
- Actual GPU-hours and peak scratch use: less than `0.037` two-device GPU-hours
  by wall-time upper bound; pre/post samples were 0 MiB and no 72B weights
  loaded, but no continuous peak sample was captured. Log 11,508 bytes; no
  checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CC=/usr/bin/gcc CXX=/usr/bin/g++
  CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12
  VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
  .venv312/bin/python -m vllm.entrypoints.openai.api_server
  /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct --served-model-name
  Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8012 --tensor-parallel-size 2
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85
  --max-num-seqs 64 --seed 42 --generation-config vllm
  --enable-prefix-caching`.
- Outputs: exact paths above; the service is stopped after the request so GPUs
  2 and 3 are released.
- Scorer/parser identity: `tools/smoke_qwen25_72b_judge.py` hash above; no
  benchmark scorer invoked.
- Metrics: no endpoint or chat response was produced. The server parsed the
  positional path only as `model_tag`, while its engine retained the vLLM 0.12
  default `model='Qwen/Qwen3-0.6B'`, resolved `Qwen3ForCausalLM`, and began
  loading that wrong model. The process was stopped immediately. Server-log
  SHA256 `f331acfd032e3bc90d43fabf08fd476ece02328a5b95b619d0b7fafd5d46c911`.
- Conclusion: `FAIL`; vLLM 0.12 requires the explicit `--model` option for this
  API-server path. The failure is identity-safe: no wrong-model completion was
  accepted and GPUs returned to 0 MiB. A separately planned `BJ-10-R1` uses the
  corrected command and is required for deployment availability; this cell
  makes no calibration or benchmark-quality claim.

### BJ-10-R1-QWEN25-72B-VLLM-SERVICE

- Cell/matrix ID and mandatory/diagnostic class: `BJ-10-R1`; bounded corrective
  deployment smoke after the fail-closed vLLM 0.12 positional-model result in
  `BJ-10`.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical user
  authorization and `EVAL-ARCH-20260720` scope to BJ-10.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: identical to BJ-10, with the added hard gate that both the API
  server and engine log the exact local 72B path supplied via explicit
  `--model`, never vLLM's default model.
- Baseline and exact output path: failed BJ-10; output root
  `artifacts/evaluation/BJ-10-R1-qwen25-72b-vllm-service-d67df17`, server log
  `server.log`, copied snapshot validation `snapshot_validation.json`, service
  smoke `smoke.json`, and GPU sample `gpu.csv`.
- Model and processor identity: identical snapshot, shard, tokenizer and chat-
  template identity to BJ-10.
- Representation checkpoint identity: N/A; identical rationale to BJ-10.
- N/A fields and justification: identical inference-only N/A fields to BJ-10.
- Policy/reference initialization: N/A; the judge is forbidden from either
  role.
- Rollout policy version and allowed asynchronous staleness: N/A; no rollout or
  update.
- Code commit and worktree state: runtime code commit
  `d67df1730527092fb96a103e2710f48bb85c2655`; launch worktree must be clean
  except for this subsequently committed ledger plan.
- Repository adapter/patch surface and hash: corrected service config SHA256
  `5af71667a951190f6fdb5d9c901aa28e4d347cec30389c37d926127530f21c7c`;
  unchanged smoke script SHA256
  `b7dea272a2eb2a4bc98257a5aec48d2004f0c10afe718a4e2c3655d0ec1c4ca7`;
  no external-checkout or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: identical synthetic `n=1`
  request and canonical request SHA256 to BJ-10; no benchmark rows.
- Native prompt/tool schema hash: no tool schema; request hash
  `30486f0066f51a616e4cf6d53403b067c6346abcbb017f414369ebe13c7e767e`.
- Chat-template/token-fixture hash and token-ownership masks: identical to
  BJ-10; masks N/A.
- D/DeepStack/position/mask identity: N/A; text-only judge.
- Observation materialization/artifact identity used by all replays: N/A; no
  replay.
- RL framework/version/environment lock: identical to BJ-10: vLLM `0.12.0`,
  Torch `2.9.0+cu128`, Transformers `4.57.6`, Python `3.12.3`, lock SHA256
  `df49237a21b66cd9009b55aee419a08715a3ad1d462cdb31bf842c16f5cd8058`.
- Objective equations and normalization: N/A; no loss/update.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical to
  BJ-10.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: identical to BJ-10; vLLM 0.12,
  seed 42, temperature 0, maximum 32 output tokens, greedy vLLM defaults, no
  custom processor and no log probabilities.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: identical to BJ-10; BF16, auto KV,
  unquantized, vLLM-selected attention, TP=2, max length 32768, utilization
  0.85, maximum 64 sequences, no training mesh.
- Logit/logprob/loss/gradient parity tolerances: success requires engine model
  path identity, served alias identity, exact response `TGVF_JUDGE_READY`, and
  normal finish; objective parity N/A.
- World size, microbatch, accumulation, and global batch: TP world `2`, one
  request/completion; training batch/accumulation N/A.
- GPUs: identical physical B200 indices, UUIDs and logical mapping to BJ-10.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T16:34:18+09:00` / `2026-07-20T16:39:30+09:00`, 312 seconds;
  API PID 2466090, engine PID 2466924, worker PIDs 2467284 and 2467285.
- Actual GPU-hours and peak scratch use: less than `0.174` two-device GPU-hours
  by wall-time upper bound; observed peak was 71,912 MiB per GPU while model
  weights occupied 67.8004 GiB/rank. Log 88,625 bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CC=/usr/bin/gcc CXX=/usr/bin/g++
  CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12
  VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
  .venv312/bin/python -m vllm.entrypoints.openai.api_server --model
  /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct --served-model-name
  Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8012 --tensor-parallel-size 2
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85
  --max-num-seqs 64 --seed 42 --generation-config vllm
  --enable-prefix-caching`.
- Outputs: exact paths above; the service will be stopped after the request and
  GPUs released.
- Scorer/parser identity: unchanged smoke script hash above; no benchmark
  scorer.
- Metrics: exact 72B identity and `Qwen2ForCausalLM` resolved; all 37 shards
  loaded in 221.08 seconds, total model load 223.35 seconds, Torch compile 22.03
  seconds, and 81.69 GiB/rank remained for a 535,376-token GPU KV cache. FlashInfer
  warm-up then failed because its subprocess could not resolve the existing
  `.venv312/bin/ninja`. Server-log SHA256
  `85bd9ae2df19d691a77768298dbffed7451fbb3ebcc406e6bcb43008789645ce`.
- Conclusion: `FAIL`; the model/vLLM/TP2 load is compatible, but the unactivated
  virtual environment omitted its Ninja executable from subprocess `PATH`.
  GPUs returned to 0 MiB and no endpoint response was accepted. Corrective
  `BJ-10-R2` explicitly fixes the existing executable path; no dependency is
  installed or changed, and no calibration or benchmark-quality claim follows.

### BJ-10-R2-QWEN25-72B-VLLM-SERVICE

- Cell/matrix ID and mandatory/diagnostic class: `BJ-10-R2`; bounded corrective
  deployment smoke after R1 proved the exact model load and exposed only the
  missing subprocess executable path.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical user
  authorization and `EVAL-ARCH-20260720` scope to BJ-10.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: identical to BJ-10-R1, with the additional gate that FlashInfer
  warm-up resolves the already installed `.venv312/bin/ninja` from an explicit
  deterministic `PATH`.
- Baseline and exact output path: failed BJ-10-R1; output root
  `artifacts/evaluation/BJ-10-R2-qwen25-72b-vllm-service-bda4a7f`, server log
  `server.log`, copied snapshot validation `snapshot_validation.json`, service
  smoke `smoke.json`, and GPU sample `gpu.csv`.
- Model and processor identity: identical fixed 72B snapshot, 37 shards,
  tokenizer and chat-template identity to BJ-10.
- Representation checkpoint identity: N/A; text-only judge.
- N/A fields and justification: identical inference-only N/A fields to BJ-10.
- Policy/reference initialization: N/A; forbidden judge roles remain unchanged.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: runtime code commit
  `bda4a7ff1612b987cebd3cc05809d8a119d14a7c`; launch worktree must be clean
  except for this subsequently committed ledger plan.
- Repository adapter/patch surface and hash: service config SHA256
  `80db006c3b75f5c40775aae8df16462bfb10ecf0724d6069b01ad58bc55e646d`;
  smoke script SHA256
  `b7dea272a2eb2a4bc98257a5aec48d2004f0c10afe718a4e2c3655d0ec1c4ca7`;
  Ninja executable SHA256
  `696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67`;
  no package, external-checkout or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: identical synthetic one-request
  fixture to BJ-10; `n=1`, no benchmark rows.
- Native prompt/tool schema hash: no tool schema; request SHA256
  `30486f0066f51a616e4cf6d53403b067c6346abcbb017f414369ebe13c7e767e`.
- Chat-template/token-fixture hash and token-ownership masks: identical to
  BJ-10; masks N/A.
- D/DeepStack/position/mask identity: N/A; text-only judge.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: identical vLLM/Torch/Transformers/
  Python and lock identity to BJ-10; only deterministic executable lookup is
  added.
- Objective equations and normalization: N/A; no loss/update.
- Rollout/replay forward mode and adapter dropout/RNG contract: identical to
  BJ-10.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: identical to BJ-10: vLLM 0.12,
  seed 42, temperature 0, maximum 32 tokens, greedy defaults, no custom
  processor or log probabilities.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/auto-KV, no quantization,
  FlashInfer selected by vLLM, TP=2, max length 32768, utilization 0.85,
  maximum 64 sequences, no training mesh.
- Logit/logprob/loss/gradient parity tolerances: exact engine/served identity,
  response `TGVF_JUDGE_READY`, and normal finish; objective parity N/A.
- World size, microbatch, accumulation, and global batch: TP world `2`, one
  request/completion; training batch/accumulation N/A.
- GPUs: identical physical B200 indices, UUIDs and logical mapping to BJ-10.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T16:41:28+09:00` / `2026-07-20T16:43:00+09:00`, 92 seconds;
  API PID 2473721, engine PID 2474245, worker PIDs 2474615 and 2474616.
- Actual GPU-hours and peak scratch use: less than `0.052` two-device GPU-hours
  by wall-time upper bound; observed peak was 71,912 MiB per GPU. Log 98,053
  bytes; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CC=/usr/bin/gcc CXX=/usr/bin/g++
  CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12
  PATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
  .venv312/bin/python -m vllm.entrypoints.openai.api_server --model
  /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct --served-model-name
  Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8012 --tensor-parallel-size 2
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85
  --max-num-seqs 64 --seed 42 --generation-config vllm
  --enable-prefix-caching`.
- Outputs: exact paths above; service stopped after smoke and GPUs released.
- Scorer/parser identity: unchanged smoke script; no benchmark scorer.
- Metrics: exact model loaded from OS cache in 23.40 seconds (25.66 seconds
  including setup), Torch compile took 13.28 seconds, 82.10 GiB/rank remained
  for a 538,064-token GPU KV cache, and Ninja was resolved. FlashInfer then
  invoked the stale `revisit-vlm` Conda `nvcc`; compilation failed on missing
  `cublasLt.h` and `nvrtc.h`. Server-log SHA256
  `ee89824ffa22feed9d068108ff51914dd43d1dd736769bcbc6211c236688afcc`.
- Conclusion: `FAIL`; explicit executable lookup closed the R1 failure, but the
  auto-selected FlashInfer backend inherited a legacy CUDA compiler. No endpoint
  response was accepted and GPUs returned to 0 MiB. `BJ-10-R3` uses the existing
  `TRITON_ATTN` path already proven by the repository's Qwen3 vLLM smoke; no
  dependency is installed or changed and no quality claim follows.

### BJ-10-R3-QWEN25-72B-VLLM-SERVICE

- Cell/matrix ID and mandatory/diagnostic class: `BJ-10-R3`; bounded corrective
  judge deployment smoke using the repository's already proven Triton attention
  path after R2 isolated FlashInfer's polluted CUDA JIT environment.
- Spike-plan git revision and VA0/VA1/VA2 approval references: identical user
  authorization and `EVAL-ARCH-20260720` scope to BJ-10.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: identical to R2, with the added gate that vLLM selects
  `TRITON_ATTN`, avoiding FlashInfer/NVCC while preserving exact 72B/TP2 service
  identity and the deterministic response contract.
- Baseline and exact output path: failed R2; output root
  `artifacts/evaluation/BJ-10-R3-qwen25-72b-vllm-service-5581164`, server log
  `server.log`, copied snapshot validation `snapshot_validation.json`, service
  smoke `smoke.json`, and GPU sample `gpu.csv`.
- Model and processor identity: identical fixed 72B snapshot, shards, tokenizer
  and chat-template identity to BJ-10.
- Representation checkpoint identity: N/A; text-only judge.
- N/A fields and justification: identical inference-only N/A fields to BJ-10.
- Policy/reference initialization: N/A; forbidden judge roles unchanged.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: runtime code commit
  `5581164249c32d936c6390c6b3c4643ee883b4f4`; launch worktree must be clean
  except for this subsequently committed ledger plan.
- Repository adapter/patch surface and hash: service config SHA256
  `85114f295ba57ec39c98785890e2041de7c756f73e04abee5424ca4eecc5f0e4`;
  smoke script SHA256
  `b7dea272a2eb2a4bc98257a5aec48d2004f0c10afe718a4e2c3655d0ec1c4ca7`;
  no dependency, external-checkout or site-package patch.
- Dataset/manifest, hashes, sample rule, and n: synthetic `n=1` request, no
  benchmark rows; request SHA256
  `30486f0066f51a616e4cf6d53403b067c6346abcbb017f414369ebe13c7e767e`.
- Native prompt/tool schema hash: no tool schema; request hash above.
- Chat-template/token-fixture hash and token-ownership masks: identical to
  BJ-10; masks N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: identical to BJ-10; vLLM 0.12, Torch
  2.9.0+cu128, Transformers 4.57.6, Python 3.12.3 and the exact lock hash.
- Objective equations and normalization: N/A; no loss/update.
- Rollout/replay forward mode and adapter dropout/RNG contract: vLLM V1 server,
  prefix caching, Triton attention, eval inference; no Adapter/replay.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: identical to BJ-10; seed 42,
  temperature 0, maximum 32 tokens, greedy vLLM defaults, no processors/logprobs.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16, auto KV, no quantization,
  `TRITON_ATTN`, TP=2, max length 32768, utilization 0.85, 64 maximum sequences,
  no training mesh.
- Logit/logprob/loss/gradient parity tolerances: exact engine/served identity,
  exact response `TGVF_JUDGE_READY`, normal finish; objective parity N/A.
- World size, microbatch, accumulation, and global batch: TP world `2`, one
  request/completion; training batch/accumulation N/A.
- GPUs: identical physical B200 indices, UUIDs and logical mapping to BJ-10.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-20T16:44:53+09:00` / `2026-07-20T16:47:26+09:00`, 153 seconds;
  API PID 2477747, engine PID 2478043, workers 2478691 and 2478692.
- Actual GPU-hours and peak scratch use: less than `0.085` two-device GPU-hours;
  observed peak 157,724 MiB/GPU, then 0 MiB after shutdown; no checkpoint.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CC=/usr/bin/gcc CXX=/usr/bin/g++
  CPATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include:/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.deps/python312-dev/root/usr/include/python3.12
  PATH=/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_ATTENTION_BACKEND=TRITON_ATTN TOKENIZERS_PARALLELISM=false
  .venv312/bin/python -m vllm.entrypoints.openai.api_server --model
  /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct --served-model-name
  Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8012 --tensor-parallel-size 2
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85
  --max-num-seqs 64 --seed 42 --generation-config vllm
  --enable-prefix-caching`.
- Outputs: exact paths above; service stopped after smoke and GPUs released.
- Scorer/parser identity: unchanged smoke script; no benchmark scorer.
- Metrics: 37 shards loaded in 24.87 seconds; engine warm-up 31.27 seconds.
  The API returned exact model alias and `TGVF_JUDGE_READY`, finish `stop`, 41
  prompt plus 7 completion tokens in 0.7323 seconds. Server log SHA256
  `559546edc248c8486310e80d5efcfc732cc331809fec319ea358a77dc3c10e94`;
  smoke JSON SHA256
  `ceca5ca5917166df7a87111d5d8083e10b7d44ec480fef8ebedd644559b48d2e`.
- Conclusion: `PASS` for local Qwen2.5-72B benchmark-judge deployment on the
  fixed Torch/vLLM stack. Service was stopped and GPUs released. This is not
  judge calibration, benchmark scoring, an RL reward, reference policy, or
  SDPO teacher claim.

### REP-QWEN3-V4-CONTEXTUAL-V2

- Lifecycle/result: `COMPLETE` / `FAIL`; user-authorized full representation-
  phase training, contextual-hidden-state provider first. The exact V1
  10-step preflight is retained unchanged as the baseline; V2 starts fresh
  because checkpoint code identity is strict.
- Identity: code `6496a4d135078f83430a63e59d9c14455fd85e69` with no bound-code
  dirt; config source/canonical SHA256 `c4888c33d72b85c7d7054082d0483b74d80c63b0e19eb0ba5b8cf8200e5b300d` /
  `43568e93ecb3e7a748fd8e8a6ec8efb60607084a2ee7786c35a3d18ec473ac25`.
  Output is exclusively
  `artifacts/representation/REP-QWEN3-V4-CONTEXTUAL-V2/`, overwrite forbidden.
- Model/data/prompt: frozen local Qwen3-VL-8B-Thinking, BF16 SDPA, tokenizer
  151669 with no resize, chat-template SHA256 `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  aspect-ratio-preserving `max_pixels=262144`; v4 clean-imend train
  SHA256 `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`
  (`n=39998`) and v3 validation SHA256
  `a228d28db76625d166dab874806c9034a244a683d41c7cecdc7f10f1aa754308`
  (`n=1382`), with the accepted recorded seven-image overlap. Native prompt v1
  is exactly `{question}`, SHA256
  `bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9`.
- Training identity: fresh TGVF Adapter seed 42; contextual layer `-1`; main D
  plus all three D-DeepStack branches; Balanced Matrix CE/L_gen/Norm weights
  `1/1/0.1`, temperature `1`, manifold `0`; AdamW LR `1e-4`, historical cosine,
  100 warmup, min ratio `.1`; 2000 optimizer steps, K4/B4 per rank, GA4,
  world2/global batch32, validation and DCP v2 every 500 steps, logging every
  10, internal evaluation disabled for this run.
- Execution: deterministic frozen-Qwen/Adapter-only FSDP2, `fsdp=[2]`,
  `reshard_after_forward=false`, BF16 parameters/outputs and FP32 reductions;
  physical GPU2 `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and GPU3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, both idle at planning
  `2026-07-20T17:11:35+09:00`. RL rollout/replay, reward, GRPO, SDPO, vLLM,
  sampling/logprobs, judge and answer scoring are N/A.
- Runtime: started `2026-07-20T17:12:57+09:00`; torchrun PID `2538519`, rank
  PIDs `2538642`/`2538643`. Step 10 reproduced the V1 preflight losses exactly;
  elapsed `4.764831926` seconds, all gradients finite, run identity
  `4c68acf45718fae12e959803826bf6c1fdba7f5701d3c54b3bbfccbd89640bf3`.
- W&B telemetry: read-only sidecar at code `997c476`, project
  `mio_mi0/tgvf-e2e-rl`, run `representation-qwen3-contextual-v2` / ID
  `rep-qwen3-v4-contextual-v2`; historical backfill and live upload verified
  through step 170. The runner-owned local JSONL remains authoritative.
- Command: `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 28800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/representation/qwen3_v4_contextual_hidden_state_v2.toml`, with stdout
  and stderr captured as `run.log` under the output root.
- Acceptance: reach step 2000; finite logged losses/gradients; strict step
  500/1000/1500/2000 checkpoints and validation events; tokenizer unchanged;
  publish the final Adapter artifact. Expected train-core time is about 2.25 h
  from RP-30, plus validation/checkpoint overhead.
- Failure: the last durable metric is step 210. At optimizer step 218, rank 1
  rejected source sample
  `tgvf_v4_teacher_50k:docvqa:ygjc0228_1:3::focus1` (source line 31416,
  SHA256 `d86e300f92a0266b6d421dc21303c500ecd6c9ef6318e02375d02ab335b6053f`)
  because Qwen token `1189` (`.\"`) overlaps the exact target-value end and
  closing JSON quote. Rank 0 subsequently timed out in FSDP2 reduce-scatter
  after 30 minutes. This is a deterministic target-span contract defect, not
  OOM or W&B failure. No checkpoint or Adapter artifact was produced; the
  output remains diagnostic-only and immutable.

### REP-QWEN3-V4-TARGET-EMBEDDING-V1

- Lifecycle/result: `CANCELLED` / `FAIL`; user-authorized paired formal
  representation run on physical GPUs 0 and 1 while contextual V2 continues on
  GPUs 2 and 3.
- Identity/output: code `6496a4d135078f83430a63e59d9c14455fd85e69`;
  config source/canonical SHA256
  `d8d5f58e8ce2aba5fdc8ff8ca521a756a7e548aa6238981b31dcd2be001d1c71` /
  `907ca84a8b8c5884f603eda648cbc399de41d7b75439be7d787897764a78b147`;
  fresh output
  `artifacts/representation/REP-QWEN3-V4-TARGET-EMBEDDING-V1/`, overwrite
  forbidden.
- Paired contract: identical to contextual V2 for frozen Qwen3-VL-8B-Thinking,
  v4/v3 data and overlap identity, native prompt v1, 512-square maximum pixel
  area, seed 42, Balanced Matrix CE/L_gen/Norm `1/1/0.1`, manifold `0`, AdamW
  and cosine schedule, K4/B4/GA4/world2/global-batch32, 2000 steps, validation
  and DCP v2 every 500, and disabled post-training internal evaluation. The
  selected provider is `target_token_embedding` using the frozen language-model
  input-embedding table; no contextual hidden state is used.
- Devices: physical GPU0 `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` and GPU1
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`, both B200 183359 MiB and idle at
  `2026-07-20T17:42:58+09:00`, exposed as logical 0/1. RL rollout/replay,
  reward, GRPO, SDPO, judge, sampling and behavior logprobs are N/A.
- Command: `CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 28800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/representation/qwen3_v4_target_token_embedding_v1.toml`, captured in
  the output root as `run.log`.
- W&B/acceptance: planned project `mio_mi0/tgvf-e2e-rl`, run name
  `representation-qwen3-target-embedding-v1`, stable ID
  `rep-qwen3-v4-target-embedding-v1`. Accept only finite losses/gradients,
  step-500 checkpoint/validation cadence, tokenizer invariance, step 2000 and a
  final Adapter artifact; local JSONL remains authoritative.
- Runtime: torchrun PID `2577005` started `2026-07-20T17:43:45+09:00`; rank
  PIDs `2577303`/`2577305`; run identity
  `f02221bff8ff38144fa1c455ed49d31b1057ee648d77804fa5756a5001896789`.
  Steps 10 and 20 completed with finite nonzero gradients; step 20 took
  `3.966164014` seconds. Live W&B upload is verified at the planned run URL.
- Failure: the last durable metric is step 210. The identical data order reached
  the same step-218 boundary-crossing sample as contextual V2; the run was
  terminated at `2026-07-20T18:06:01+09:00` instead of waiting for a second
  redundant 30-minute NCCL watchdog timeout. No checkpoint or Adapter artifact
  was produced; the output remains diagnostic-only and immutable.

### REP-QWEN3-V4-CONTEXTUAL-V3

- Lifecycle/result: `CANCELLED` / `FAIL`; fresh replacement for contextual V2
  after accepted repair `RPI-20260720-TARGET-TOKEN-COVER-V1`.
- Identity/output: code `2436861f027c319dadf1e775ffa9ef911f317dfb`;
  config source/canonical SHA256
  `156ffbfc83ef5209e85a6da745425baf3c20cdcc62bcf13a0ef03872de66e513` /
  `89e418f7702ff2d7bc7481cceb955b2bdf9779ccf79d421eff064197bb9cc236`;
  fresh output
  `artifacts/representation/REP-QWEN3-V4-CONTEXTUAL-V3/`, overwrite forbidden.
- Fixed contract/preflight: raw target offsets plus
  `minimal_overlapping_sampled_token_cover_v1`; rank-local exceptions abort the
  process group. Eighty-eight targeted tests passed. A CPU scan passed all
  sampler-reachable action targets: train `36117/39998` rows in 8209 usable
  image groups and validation `959/1382` rows in 226 usable image groups, with
  zero failures.
- Scientific identity: frozen local Qwen3-VL-8B-Thinking, BF16 SDPA, tokenizer
  length 151669/no resize, native prompt v1 `{question}`, max pixels 262144;
  v4 clean-imend train plus v3 validation and the recorded overlap; fresh seed
  42 contextual hidden state at layer `-1`; main D plus three D-DeepStack
  branches; Balanced Matrix CE/L_gen/Norm weights `1/1/0.1`, temperature `1`,
  manifold `0`; AdamW `1e-4`, historical cosine, warmup 100/min ratio `.1`.
- Execution: FSDP2 world 2, `fsdp=[2]`, `reshard_after_forward=false`, K4/B4
  per rank, GA4/global batch32, 2000 optimizer steps, log every 10 and
  validation/DCP v2 every 500; internal evaluation disabled. Physical GPUs 2
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` and 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b` were idle at
  `2026-07-20T18:28:32+09:00`. RL/reward/GRPO/SDPO/judge/rollout fields are N/A.
- W&B/command: project `mio_mi0/tgvf-e2e-rl`, run
  `representation-qwen3-contextual-v3`, ID `rep-qwen3-v4-contextual-v3`;
  `CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
  TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN
  timeout 28800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m
  tgvf_rl.cli run-representation
  configs/representation/qwen3_v4_contextual_hidden_state_v3.toml`.
- Acceptance: first prove step 219 or later with finite loss/gradients, then
  reach step 2000 with the configured validation/checkpoint cadence and final
  TGVF Adapter artifact.
- Runtime/result: started `2026-07-20T18:29:58+09:00`, reached last durable
  train metric step 470 with finite loss/gradients, and passed the prior
  step-218 target-span failure. It was terminated at
  `2026-07-20T19:09:24+09:00` before its first periodic boundary because the
  paired run had already proven the shared checkpoint defect. No checkpoint or
  Adapter artifact exists; the output and W&B run are diagnostic and immutable.

### REP-QWEN3-V4-TARGET-EMBEDDING-V2

- Lifecycle/result: `COMPLETE` / `FAIL`; paired fresh replacement for target-
  embedding V1 under the same accepted repair and preflight evidence as
  contextual V3.
- Identity/output: code `2436861f027c319dadf1e775ffa9ef911f317dfb`;
  config source/canonical SHA256
  `ff8e69ea09dbcdc6cebab7bbd1a2e2439b1849e4707dd2f3c589f2c55f02749d` /
  `a3b8ae806e751fd05d37e8141577fd63125113f8e5234196146772338ec6e55f`;
  fresh output
  `artifacts/representation/REP-QWEN3-V4-TARGET-EMBEDDING-V2/`, overwrite
  forbidden.
- Paired identity: every model/data/prompt/objective/optimizer/schedule/batch,
  cadence, determinism, span and preflight field is identical to contextual V3;
  only the provider is `target_token_embedding` from the frozen language-model
  input-embedding table and the device/output/run identities differ. Physical
  GPUs 0 `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` and 1
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc` were idle at
  `2026-07-20T18:28:32+09:00`. RL/reward/GRPO/SDPO/judge/rollout fields are N/A.
- W&B/command: project `mio_mi0/tgvf-e2e-rl`, run
  `representation-qwen3-target-embedding-v2`, ID
  `rep-qwen3-v4-target-embedding-v2`; `CUDA_VISIBLE_DEVICES=0,1
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 NCCL_DEBUG=WARN timeout 28800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation
  configs/representation/qwen3_v4_target_token_embedding_v2.toml`.
- Acceptance: first prove step 219 or later with finite loss/gradients, then
  reach step 2000 with the configured validation/checkpoint cadence and final
  TGVF Adapter artifact.
- Runtime/result: started `2026-07-20T18:29:58+09:00`; step 500 training and
  validation event 0 both completed with finite metrics. Immediately afterward,
  DCP save preflight failed on both ranks because no-grad validation under
  `reshard_after_forward=false` left module-visible full parameters resident
  while AdamW retained the sharded DTensor parameters. The strict ownership
  audit correctly rejected the mismatch; no checkpoint or Adapter artifact was
  written. This is fixed only by
  `RPI-20260720-PERIODIC-BOUNDARY-SMOKE-V1`; the old output and W&B run remain
  diagnostic and immutable.

### RP-31-QWEN3-REPRESENTATION-PERIODIC-BOUNDARY-CONTEXTUAL-GPU23

- Cell/status: mandatory periodic-boundary regression, `COMPLETE` / `INVALID`,
  authorized by `RPI-20260720-PERIODIC-BOUNDARY-SMOKE-V1`. It asks whether the
  real Torch 2.9 FSDP2 no-reshard path can execute train -> validation ->
  explicit reshard/ownership audit -> DCP v2 save, then restore and repeat.
- Code/config/output: code `705018b0d5bb1e02d0bae87c5e1503680db37eb9`;
  fresh config source/canonical SHA256
  `210da220d659cf2219c5b0daa7c33cffe8b8f6be5a3aebf0ac0b066481ebf073` /
  `e98351628541e0dc3db2902c0dbb55ea9c8ab10e3747a7a74dc6f68450c0ad81`;
  resume config source/canonical SHA256
  `d0b08899d24a0868ac71d0bc460b95ebf54524bf484424f1c040e0f1274045fe` /
  `5e14592d4580a908da46bb8ba2cb216c2c1a1cd6405fcc9529317365f3b1a85c`;
  immutable output
  `artifacts/representation/RP-31-qwen3-periodic-boundary-contextual-gpu23/`.
- Identity: frozen local Qwen3-VL-8B-Thinking, BF16 SDPA, tokenizer 151669/no
  resize, max pixels 262144; current native prompt v1 `{question}`; fixed K4
  train/validation fixtures SHA256 `7351cdcd81adf8861ed867144c27e2faa67587f3b31531fa54658cf54134800d` /
  `5a0ab5148d75d6b3df5c7c4e3ee61a5d824ddf2b82df6474769af730f6db4d12`;
  contextual hidden-state provider layer `-1`, fresh seed 42, Balanced Matrix
  CE/L_gen/Norm `1/1/.1`, temperature 1, manifold 0, AdamW `1e-4`, 2000-step
  historical-cosine horizon, K4/B4/GA4/world2/global batch32.
- Execution: FSDP2 `reshard_after_forward=false`; log, validation and DCP
  cadence all 1. Invocation A stops operationally after step 1; invocation B is
  a new torchrun restoring step 1 and stopping after step 2. RL, rollout,
  behavior logprobs, reward, GRPO, SDPO, vLLM and judge fields are N/A. GPUs 2
  and 3 were idle at `2026-07-20T19:19:40+09:00` with UUIDs already recorded in
  contextual V3.
- Commands: common deterministic environment from contextual V3, then
  `torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation
  configs/smoke/representation_qwen3_contextual_rp31_periodic_boundary_gpu23.toml
  --stop-after-global-step 1`; repeat in a fresh process with the `_resume.toml`
  config and `--stop-after-global-step 2`, each bounded by 1800 seconds.
- Acceptance: step-1 train and validation events, a complete DCP v2 directory,
  clean teardown, exact restore, step-2 train/validation and second DCP; finite
  metrics, exact cursor/history binding, and no ownership error.
- Result: rejected before model load at `2026-07-20T19:20:52+09:00`; the reused
  historical K4 fixture predates the current strict retained-focus transform
  and produces zero accepted rows. No training metric or checkpoint exists.
  Output is immutable; the replacement uses the exact formal data contract.

### RP-32-QWEN3-REPRESENTATION-PERIODIC-BOUNDARY-TARGET-EMBEDDING-GPU01

- Cell/status: paired mandatory regression, `COMPLETE` / `INVALID`, with every
  scientific/lifecycle field identical to RP-31 except the selected provider,
  GPUs and output identity. Provider is `target_token_embedding` from the
  frozen language-model input embedding.
- Code/config/output: code `705018b0d5bb1e02d0bae87c5e1503680db37eb9`;
  fresh config source/canonical SHA256
  `9e375da0ac616c67465789fa95eabbc0443704f45a3c9a1ffafe0245846c61fa` /
  `2f5fe6b432474d5d31d5301ade236b9d3306e9c96bde7e64a11b98f887e44e41`;
  resume config source/canonical SHA256
  `473ec1bb8ee16b31a36442b543756cf1a0eba9200cc0c401b413834409e304e2` /
  `a98a2558dd45c29c9b2c06e412782d052e03ba0dfb53ca875f6dbfbd3f4e9821`;
  immutable output
  `artifacts/representation/RP-32-qwen3-periodic-boundary-target-embedding-gpu01/`.
- Execution/commands: physical GPUs 0/1 were idle at
  `2026-07-20T19:19:40+09:00`; use the same two 1800-second deterministic
  torchrun invocations as RP-31 with the RP-32 fresh/resume config paths and
  operational stops 1/2. Acceptance is identical to RP-31; RL-only fields are
  N/A.
- Result: rejected before model load at `2026-07-20T19:20:54+09:00` for the
  same zero-accepted-row historical fixture as RP-31. No training metric or
  checkpoint exists; output is immutable.

### RP-33-QWEN3-REPRESENTATION-PERIODIC-BOUNDARY-FORMALDATA-CONTEXTUAL-GPU23

- Cell/status: `COMPLETE` / `PASS`; valid-data replacement for RP-31. The
  model, contextual provider, prompt, objective, optimizer, 2000-step scheduler
  horizon, K4/B4/GA4/world2 topology, no-reshard lifecycle, cadence-one
  train/validation/DCP and two-process stop-1/resume-to-2 acceptance are
  identical to RP-31.
- Data/code/output: exact formal v4 clean-imend train and v3 validation contract
  from contextual V3, including recorded overlap identity; code
  `705018b0d5bb1e02d0bae87c5e1503680db37eb9`. Fresh config source/canonical
  SHA256 `477d68574e20981fdeba1662ee42ddbaf52ade9c313b23a469483a443d837d56` /
  `290e32a4282382e50d72f31cf9d92aee1be99b3ac149818324fb3fa3b2594b2c`;
  resume config `3691dbc444cd1d342bb3e9b3175f1e281203e7fb62da46cfd87ead5a0b6e2879` /
  `0d00232d5a97272bd06741bbbe13fb51520de9e207fd5251e12fb95bbc35b05c`;
  immutable output
  `artifacts/representation/RP-33-qwen3-periodic-boundary-formaldata-contextual-gpu23/`.
- Command/resources: deterministic environment and two commands follow RP-31,
  substituting RP-33 fresh/resume configs; physical GPUs 2/3 were idle at
  `2026-07-20T19:22:03+09:00`. Each invocation is bounded by 1800 seconds. RL
  fields remain N/A.
- Result: fresh torchrun completed train step 1, validation event 0, DCP v2
  step 1 and clean paused teardown. A new process strictly restored it and
  completed train step 2, validation event 1, DCP v2 step 2 and clean teardown.
  Metrics history is exactly `start, train1, validation1, train2, validation2`;
  checkpoint cursors are `(step=1,next_validation=1)` and
  `(step=2,next_validation=2)`. Both checkpoints contain both rank shards and
  validated metadata; no ownership/restore error occurred.

### RP-34-QWEN3-REPRESENTATION-PERIODIC-BOUNDARY-FORMALDATA-TARGET-EMBEDDING-GPU01

- Cell/status: `COMPLETE` / `PASS`; target-token-embedding pair of RP-33 and
  valid-data replacement for RP-32. All fields and acceptance match RP-33
  except provider/device/output identity.
- Code/config/output: code `705018b0d5bb1e02d0bae87c5e1503680db37eb9`;
  fresh source/canonical SHA256
  `8a0b1cf6705d4cdc1c09473eb0031af55cc7229d6e2ada041729a98bcd59d7f6` /
  `53de05dc03294cedc57bc72c53c58d28a940eed3d189f295a3aa652c64d14a43`;
  resume source/canonical SHA256
  `bec760676fe160f589aa3acb15c3541d2cfd2cf09a740266827393306dba41ca` /
  `86ec0d24719b521d0525c203c1f8a7c04736b797c697e4a059c7c24d25a7c1e0`;
  immutable output
  `artifacts/representation/RP-34-qwen3-periodic-boundary-formaldata-target-embedding-gpu01/`.
- Command/resources: RP-34 fresh/resume configs with operational stops 1/2 on
  physical GPUs 0/1, idle at `2026-07-20T19:22:03+09:00`; 1800-second bounds;
  RL fields N/A.
- Result: identical PASS shape to RP-33: fresh train/validation/DCP step 1,
  strict new-process restore, resumed train/validation/DCP step 2, exact
  metrics-history and validation cursors 1/2, both rank shards present, clean
  teardowns and no ownership/restore error. GPUs 0--3 were released by
  `2026-07-20T19:25:56+09:00`.

### REP-QWEN3-V4-CONTEXTUAL-V4

- Lifecycle/result: `COMPLETE` / `PASS`; fresh formal replacement for V3
  after mandatory periodic-boundary smoke RP-33 passed.
- Identity/output: code `705018b0d5bb1e02d0bae87c5e1503680db37eb9`;
  config source/canonical SHA256
  `416bf6f4813407265f0eeaa00bfd75d65908a8db975b3c128cd9b49e07eb380b` /
  `8d04f5370252c61f9dd1c9b24dc8c70843d22c46d985b05c32318b94de32ce9d`;
  immutable fresh output `artifacts/representation/REP-QWEN3-V4-CONTEXTUAL-V4/`.
- Scientific/execution identity: exactly contextual V3 for model, formal
  train/validation data and recorded overlap, current native prompt v1, seed
  42, contextual layer `-1`, TGVF Adapter/main D/three DeepStack branches,
  Balanced Matrix CE/L_gen/Norm `1/1/.1`, temperature 1, manifold 0, AdamW and
  2000-step cosine, max pixels 262144, K4/B4/GA4/world2/global batch32,
  `reshard_after_forward=false`, logs every 10 and validation/DCP every 500.
  Only the accepted target-span and validation-reshard repairs, run/output and
  code identities differ. RL-only fields are N/A.
- GPU/W&B/command: physical GPUs 2/3 were idle at
  `2026-07-20T19:27:57+09:00`; W&B project `mio_mi0/tgvf-e2e-rl`, run
  `representation-qwen3-contextual-v4`, ID `rep-qwen3-v4-contextual-v4`.
  Deterministic environment from V3; `timeout 28800s .venv312/bin/torchrun
  --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation
  configs/representation/qwen3_v4_contextual_hidden_state_v4.toml`.
- Acceptance: finite train/validation metrics, DCP v2 at 500/1000/1500/2000,
  final step 2000 Adapter artifact, tokenizer invariance and clean teardown.
- Runtime: started `2026-07-20T19:28:50+09:00`; torchrun PID `2800191`, rank
  PIDs `2800348/2800349`, run identity
  `6c74885183eba8cfc6d7393a296b897b3caf939c88ffe016028c3a02b5f3d5c0`.
  Completed step 2000 with all four DCP boundaries and a clean teardown. Final
  validation Matrix CE/L_gen/Norm/total is
  `1.32421875/1.5166015625/0.4899977744/2.8898200899`; the loader-verified
  104-tensor artifact manifest is
  `dfa992fcea0cee8a0fb48f19892ff10ac42a6cf040d32afcae748c7724e80e10`.
  W&B finished and synced at
  `https://wandb.ai/mio_mi0/tgvf-e2e-rl/runs/rep-qwen3-v4-contextual-v4`.

### REP-QWEN3-V4-TARGET-EMBEDDING-V3

- Lifecycle/result: `COMPLETE` / `PASS`; paired fresh formal replacement for
  V2 after mandatory periodic-boundary smoke RP-34 passed.
- Identity/output: code `705018b0d5bb1e02d0bae87c5e1503680db37eb9`;
  config source/canonical SHA256
  `9662a163212a2de36f61b093f692c8262cba50daf548a9dfb95833c2bfc3b3eb` /
  `99c2058f2a67ce39366f2517dec9c7e2e1c73aa491fd5a77e6f0300c42c9a437`;
  immutable fresh output
  `artifacts/representation/REP-QWEN3-V4-TARGET-EMBEDDING-V3/`.
- Paired identity: every field and acceptance is identical to contextual V4
  except target-token-embedding provider, GPUs and run/output/W&B identities.
  Physical GPUs 0/1 were idle at `2026-07-20T19:27:57+09:00`; W&B run
  `representation-qwen3-target-embedding-v3`, ID
  `rep-qwen3-v4-target-embedding-v3`. Command uses the deterministic V2
  environment and `configs/representation/qwen3_v4_target_token_embedding_v3.toml`
  under the same 28800-second bound. RL-only fields are N/A.
- Runtime: started `2026-07-20T19:28:50+09:00`; torchrun PID `2800187`, rank
  PIDs `2800335/2800336`, run identity
  `5d23fafd91ec3af9a926e52af4fd6ed163349ae3734ef3c78057a329f55ee245`.
  Completed step 2000 with all four DCP boundaries and a clean teardown. Final
  validation Matrix CE/L_gen/Norm/total is
  `1.5625/1.8115234375/0.4689098001/3.4209144175`; the loader-verified
  104-tensor artifact manifest is
  `3cccf99efd21fc3ea2cd62d780ad2c7ffcaec7dbb8c6c4a49892f675c11a7469`.
  W&B finished and synced at
  `https://wandb.ai/mio_mi0/tgvf-e2e-rl/runs/rep-qwen3-v4-target-embedding-v3`.

### RP-35-QWEN3-REPRESENTATION-INTERNAL-EVAL-CONTEXTUAL-V4-GPU0

- Lifecycle/result: `COMPLETE` / `INVALID`; formal post-hoc internal evaluation
  of completed contextual artifact, without training resume or weight mutation.
- Identity: code `463108ed559a11210ef8703dfdbcc451038b2d7d`, clean;
  evaluation TOML SHA256
  `a3dd5f5f592637a6210233923dee7043ecb20fe3ea8c95689d323907265534f8`;
  original training TOML SHA256
  `416bf6f4813407265f0eeaa00bfd75d65908a8db975b3c128cd9b49e07eb380b`;
  training run identity
  `6c74885183eba8cfc6d7393a296b897b3caf939c88ffe016028c3a02b5f3d5c0`.
- Model/artifact: local Qwen3-VL-8B-Thinking, BF16/SDPA,
  max-pixels 262144, tokenizer 151669/no resize, native prompt v1 and chat
  template from the original run; contextual layer `-1`; step-2000 Adapter file
  SHA256 `50179c709c5788d83ffc58d13dcde9e15ed448b2cf3233a5db67cb7501106e75`,
  manifest `dfa992fcea0cee8a0fb48f19892ff10ac42a6cf040d32afcae748c7724e80e10`.
- Population: fixed retained validation manifest
  `f47bbff7c63ffa381ce2e2e263130c783057c6d408575ef4c4e3dd5b019c5a33`;
  31 exact-K4/grid-`(1,26,38)` groups, 124 unique rows, seed 42, ordered
  manifest `dc327967279cee32f014a3806a473b2442d3cd51df6cf4efa46a33b1f5806ab8`;
  one audited same-question/target native counterfactual `262` versus `38`,
  manifest `17238100d1fb807038befbffdc6e6d5808dbf9e6d63ceeb1904e03b596b474a0`.
- Execution: frozen Qwen and Adapter in eval mode; main D and all three
  D-DeepStack branches remain atomic; deterministic forward, seed 42, greedy
  native free continuation bounded at 64 tokens/EOS 151645. RL behavior,
  reference, rollout, objective, logprob, clipping, optimizer, batch and
  framework fields are N/A because no RL or optimization occurs.
- GPU/command/output: physical GPU 0
  `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2`, idle at preflight
  `2026-07-20`; `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 7200s .venv312/bin/python
  -m tgvf_rl.cli run-representation-internal-evaluation
  configs/representation/qwen3_v4_contextual_hidden_state_v4_internal_evaluation_v1.toml`.
  Immutable output directory
  `artifacts/representation/RP-35-qwen3-internal-eval-contextual-v4-gpu0/`.
- Result: model and Adapter loading succeeded, then the report reducer rejected
  the first BF16 row because it compared FP32-divided summed NLL with
  BF16-divided L_gen at `1e-6`. Exit 2; no report was published and no weights
  or training records changed. Superseded by RP-37.

### RP-36-QWEN3-REPRESENTATION-INTERNAL-EVAL-TARGET-EMBEDDING-V3-GPU3

- Lifecycle/result: `COMPLETE` / `INVALID`; exact paired evaluation of the
  target-token-embedding artifact. Population, model, prompt, deterministic
  evaluator, N/A fields and acceptance are identical to RP-35.
- Identity/artifact: code
  `463108ed559a11210ef8703dfdbcc451038b2d7d`, clean; evaluation TOML SHA256
  `2827799b1a258e058a736eedfb6a2bd03970a72b94540df034e68cdc795577fd`;
  original training TOML SHA256
  `9662a163212a2de36f61b093f692c8262cba50daf548a9dfb95833c2bfc3b3eb`;
  training run identity
  `5d23fafd91ec3af9a926e52af4fd6ed163349ae3734ef3c78057a329f55ee245`;
  step-2000 Adapter file SHA256
  `646a1b60523cdcc01ac2926d4cf5c1aa46cbfa3d43833be0c396aba730ce9e88`,
  manifest `3cccf99efd21fc3ea2cd62d780ad2c7ffcaec7dbb8c6c4a49892f675c11a7469`.
- GPU/command/output: physical GPU 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`, idle at preflight
  `2026-07-20`; the RP-35 deterministic command with `CUDA_VISIBLE_DEVICES=3`
  and target-embedding evaluation TOML. Immutable output directory
  `artifacts/representation/RP-36-qwen3-internal-eval-target-embedding-v3-gpu3/`.
- Result: same pre-report BF16 reduction-order rejection as RP-35. Exit 2; no
  report was published and no weights or training records changed. Superseded
  by RP-38.

### RP-37-QWEN3-REPRESENTATION-INTERNAL-EVAL-CONTEXTUAL-V4-GPU0

- Lifecycle/result: `COMPLETE` / `FAIL`; exact replacement for RP-35 after
  preserving the score tensor's BF16 divide-before-export reduction order.
  All model, Adapter, population, prompt, deterministic evaluator, N/A fields,
  GPU UUID, command environment and acceptance remain exactly RP-35.
- Identity/output: clean code
  `fb64769c1dc04bf1c7fab793be951e9cbd37b257`; evaluation TOML SHA256
  `9a051f2012d54b892874b8154e3be6eef01cd1788d995e4ed4c96d78b1bc4060`;
  command selects the contextual `internal_evaluation_v2.toml`; immutable
  output `artifacts/representation/RP-37-qwen3-internal-eval-contextual-v4-gpu0/`.
- Result: exit `0`; correct-D beats target-only/random at `0.9516/0.9597`, but
  same-image correct-D win rate is `0.5081` and K4 retrieval top-1 is `0.2984`.
  The artifact is readable but fails target-specificity promotion.

### RP-38-QWEN3-REPRESENTATION-INTERNAL-EVAL-TARGET-EMBEDDING-V3-GPU3

- Lifecycle/result: `COMPLETE` / `FAIL`; exact paired replacement for RP-36.
  All scientific/execution fields remain exactly RP-36/RP-37 except provider,
  bound Adapter and physical GPU 3.
- Identity/output: clean code
  `fb64769c1dc04bf1c7fab793be951e9cbd37b257`; evaluation TOML SHA256
  `80fed1529d1b79e2d5abf4e2ad06569c1ccef196eaf06531620f49dae4c80e05`;
  command selects the target-embedding `internal_evaluation_v2.toml`; immutable
  output
  `artifacts/representation/RP-38-qwen3-internal-eval-target-embedding-v3-gpu3/`.
- Result: exit `0`; correct-D beats target-only/random at `0.9677/0.9677`, but
  same-image correct-D win rate is `0.5726` and K4 retrieval top-1 is `0.3468`.
  This provider is directionally better but still fails target-specificity.

### RP-39-QWEN3-REP-MATRIXCE-LEGACY-CONTEXTUAL-500-GPU01

- Lifecycle/result: `COMPLETE` / `PASS`; execution-only positive-control diagnosis for the
  readable-but-weakly-specific contextual V4 result.
- Fixed identity: clean runtime code
  `fb64769c1dc04bf1c7fab793be951e9cbd37b257`; config source/canonical SHA256
  `b57369d50da21f8ff002eef1bfc82e86e388cf998a9e983a773dea5cbea35195` /
  `db585177f47f543720e35fcf83851c0bca2f5bd0b8f20a7416621f3be4672da4`;
  immutable output
  `artifacts/representation/RP-39-qwen3-matrix-ce-legacy-contextual-500-gpu01/`.
- Scientific identity: Qwen3-VL-8B-Thinking BF16/SDPA, tokenizer 151669/no
  resize, max-pixels 262144, contextual hidden state layer `-1`, native prompt
  v1, exact V4 clean-imend train and v3 validation data/overlap identities,
  same-image K4/B4/GA4/world2/global batch32, fresh seed42, AdamW `1e-4`,
  L_gen/Norm `1/.1`, manifold 0, and no-reshard FSDP2 exactly as contextual V4.
  The diagnostic objective changes Matrix-CE only to legacy summed NLL,
  temperature 1.0 and keeps weight 1.0.
- Horizon/acceptance: target 500 optimizer steps under the unchanged historical
  cosine horizon 2000/warmup100/min-ratio.1; log every10, one validation and
  DCP/final export at500. Training completion is not promotion. The artifact
  must subsequently run the same 31×K4/124-row internal manifest and audited
  counterfactual as RP-37/38.
- GPU/command: physical GPUs 0/1
  `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` /
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`; deterministic environment;
  `timeout 10800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m
  tgvf_rl.cli run-representation
  configs/representation/qwen3_matrix_ce_legacy_contextual_500step_gpu01.toml`.
  RL rollout/reference/logprob/reward fields are N/A.
- Result: step 500, validation Matrix CE `1.94921875`, exit `0`; artifact file/
  manifest SHA256 `76ae764d...2475` / `88dd6a86...3d51b`. Semantic result is
  supplied by RP-41 and does not promote this artifact.

### RP-40-QWEN3-REP-MATRIXCE-BALANCED-T01-CONTEXTUAL-500-GPU23

- Lifecycle/result: `COMPLETE` / `PASS`; execution-only paired Matrix-CE calibration lane.
- Fixed identity: clean runtime code
  `fb64769c1dc04bf1c7fab793be951e9cbd37b257`; config source/canonical SHA256
  `fc760ca089b816027af6024fe1aec61d8f91b08116f713bb3e38a35f8875baab` /
  `6e3ed4bcc7deba804e197b28595687dbd24a0bf02c9930971ba063063bf30333`;
  immutable output
  `artifacts/representation/RP-40-qwen3-matrix-ce-balanced-t01-contextual-500-gpu23/`.
- Paired identity: every scientific/horizon/acceptance field is RP-39 except
  Matrix-CE uses Balanced mean NLL at fixed temperature 0.1 and the unique
  objective/run/output identity. This pair isolates the accepted calibration
  bundle, not normalization and temperature independently.
- GPU/command: physical GPUs 2/3
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` /
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; deterministic RP-39 command with
  its balanced-T0.1 config. RL-only fields are N/A.
- Result: step 500, validation Matrix CE `1.92578125`, exit `0`; artifact file/
  manifest SHA256 `36166909...9ee` / `c071580b...06bc`. Semantic result is
  supplied by RP-42 and does not promote this artifact.

### RP-41/42-QWEN3-REP-INTERNAL-EVAL-MATRIXCE-STEP500

- Lifecycle/result: `COMPLETE` / `FAIL`; paired post-hoc evaluation of the
  completed RP-39 legacy and RP-40 Balanced-T0.1 step-500 artifacts.
- Fixed comparison: the same 31xK4/124-row ordered manifest, native
  counterfactual, seed 42, greedy 64-token readout, Qwen3 model, prompt,
  contextual provider, and max-pixels 262144 used by RP-37/38. No training or
  weight mutation occurs.
- Artifacts: RP-39 file/manifest
  `76ae764d3af41b0abcd74037bc110cf196de700cc7e8efc364e53a0c7a8a2475` /
  `88dd6a86e8b6e3ff26c1c3bf6fce64ddf474b27d6fbfe6f12f914c7e97a3d51b`;
  RP-40 file/manifest
  `361669094a57472d179030b6ac9a55d93a174ab877e6171d042155dd7e1f19ee` /
  `c071580bbb82c540a8493ce55aa90ac08cbdcd6f5aadfdf5e202a1b59ba370f7`.
- Execution: clean fixed evaluation runtime `fb64769`, physical GPUs 4/5,
  paired TOMLs `qwen3_matrix_ce_*_500step_internal_evaluation_gpu{4,5}.toml`;
  immutable outputs `RP-41-*` and `RP-42-*` under `artifacts/representation/`.
- Result: both exit `0`. RP-41 legacy versus RP-42 Balanced-T0.1 K4 top-1 is
  `0.3145` versus `0.2823`; wrong-same-image win rate is `0.4919` versus
  `0.5161`. Both remain near chance and fail promotion. Balanced-T0.1 improves
  wrong-different-image readability (`0.9355` versus `0.7581`) but not target
  specificity, so temperature calibration is not the isolated root cause.

### RP-43-QWEN3-REP-MATRIXCE-MASK-BOUNDARY-DIAGNOSTIC

- Lifecycle/result: `CANCELLED` / `INVALID`; both launches were stopped at user
  request before an accepted artifact. The proposed mask was rejected as the
  explanation for train/validation specificity divergence and the code was
  restored to the preceding evidence-boundary behavior.
- Fixed comparison: every RP-40 field remains unchanged except clean code
  commit `5ad8c10941633f7e0e2fa5449db9efa42d509a7c`, mask identity
  `causal_tool_response_onward_original_image_key_block_v2`, run/objective/
  output identity, and physical GPUs 0/1. Original-image keys are blocked from
  the exact native `<tool_response>` opener rather than only evidence queries.
- Config identity: source/canonical SHA256
  `1ef5dea8f3aeb33fd4fcb270e370095a338983008619906d90498262f5977ebd` /
  `c460df44a78772e650b893e1e90ef580b118cab4879e6481e9761a58d6fc7023`;
  `configs/representation/qwen3_matrix_ce_balanced_t01_contextual_maskfix_500step_gpu01.toml`.
- Model/data/math: local Qwen3-VL-8B-Thinking, v4 clean-imend train/v3 val,
  contextual layer -1, seed 42, max-pixels 262144, K4/B4/world2/GA4/global32,
  Balanced-T0.1 Matrix CE + L_gen + Norm weights `1/1/.1`, manifold zero,
  AdamW `1e-4`, unchanged 2,000-step scheduler horizon, target step 500.
- GPUs/command/output: the final attempted launch used B200 GPUs 0/1;
  deterministic clean-worktree
  `torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli run-representation`
  with the RP-43 config; immutable output
  `artifacts/representation/RP-43-qwen3-matrix-ce-balanced-t01-maskfix-contextual-500-gpu01/`.
- Acceptance: none; this cell must not be resumed or used as evidence.

### RP-44-QWEN3-REP-INTERNAL-EVAL-BALANCED-T01-V4-GOLDEN-GPU3

- Lifecycle/result: `COMPLETE` / `PASS`; post-hoc evaluation of the completed
  RP-40 Balanced-T0.1 step-500 Adapter on the historical Golden's actual
  same-distribution v4 clean-imend test population.
- Code/config: clean code commit `ffa41467613b24376374e53d99be5bf29e9d0a0b`;
  run config
  `configs/representation/qwen3_matrix_ce_balanced_t01_contextual_500step_internal_evaluation_v4_golden_gpu3.toml`.
- Fixed data identity: source SHA256 `de61c731...82d`, retained manifest
  `534f5b1e...b0`, exact first 200 retained rows in 46 ordered same-image groups;
  variable-K counts are K3/K4/K5/K6 = 6/19/20/1. Ordered manifest SHA256 is
  `55e2cde5...d8`; counterfactual manifest SHA256 is `4589d14f...cc4`.
- Artifact: RP-40 Adapter file/manifest SHA256 `36166909...9ee` /
  `c071580b...06bc`; no training or mutation. Query matrices use evidence-token
  mean NLL, matching the Golden score semantics. Shape-compatible
  wrong-different-image D remains an optional secondary control.
- GPU/command/output: physical GPU 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; deterministic single-GPU
  evaluation-only CLI; immutable report path
  `artifacts/representation/RP-44-qwen3-internal-eval-balanced-t01-v4-golden-gpu3/report.json`.
- Acceptance: technical completion requires 200 samples, 46 groups, exact
  manifest binding, tokenizer unchanged, and a complete immutable report.
  Scientific comparison is against Golden top-1 `0.77`; no v3 result is an
  active validation reference.
- Result: exit `0`; report SHA256
  `bfbac956a98bbcc6e1edce5fd23faf8b2a3cfadd4916e8981f17b5cc78c4f4a5`.
  Exact population checks passed (200 rows/46 groups), tokenizer remained
  `151669`, retrieval top-1/top-2/MRR are `0.735/0.87/0.84158`, versus Golden
  `0.77/0.915/0.86917`. Correct-D beats wrong-same-image D on `0.865` of rows;
  target-only and random-D controls are both `1.0`. This invalidates the v3
  fixed-grid report as a target-specificity diagnosis: RP-40 is strongly
  separable on the actual same-distribution population and lies 3.5 percentage
  points below Golden top-1 at step 500.

### RP-45/48-QWEN3-REP-500STEP-MATRIX-CE-THREE-WAY-V4-GOLDEN

- Lifecycle/result: `COMPLETE` / `PASS`; three-way step-500 contextual-hidden-
  state comparison on the exact RP-44 v4 Golden 200/46 population.
- Cells: RP-45 is legacy summed-NLL/T=1.0 from RP-39; RP-44 is the completed
  mean-NLL/T=0.1 RP-40 result; RP-48 is mean-NLL/T=1.0 from the original
  `REP-QWEN3-V4-CONTEXTUAL-V4` durable step-500 checkpoint.
- Step-500 materialization: the original DCP checkpoint is loaded without
  training into an evaluation-only Adapter export. File/manifest SHA256 are
  `4b54650b...ff6` / `2b676976...b7a`; original run identity remains
  `6c748851...5c0`. It is not a separately trained artifact.
- Execution: clean evaluator code `ffa4146`; RP-45/RP-48 use physical GPUs 0/3
  and immutable report roots under `artifacts/representation/RP-{45,48}-*`.
  All non-objective fields, manifests, prompt, provider, seed, and score
  semantics are fixed. This three-way matrix separates length normalization
  and temperature better than the earlier two-way bundled comparison.
- Result: all cells exited `0` on 200 rows/46 groups with tokenizer `151669`.
  Summed-NLL/T1.0, mean-NLL/T0.1, and mean-NLL/T1.0 respectively yield
  top-1 `0.72/0.735/0.59`, top-2 `0.905/0.87/0.805`, MRR
  `0.84017/0.84158/0.75442`, and mean diagonal gap
  `0.11473/0.13714/0.08649`. Report SHA256 values are
  `8fe6572e...f4a`, `bfbac956...4a5`, and `c6ace5a1...66a`. At step 500,
  mean-NLL/T0.1 improves top-1 by 1.5 points over legacy but does not improve
  top-2 or wrong-same win rate; mean-NLL/T1.0 is materially worse. The evidence
  therefore supports a small early top-1 benefit for the combined Balanced+
  low-temperature setting, not a broad claim that normalization alone wins.

### RP-46/47-QWEN3-REP-2000STEP-PROVIDER-PAIR-V4-GOLDEN

- Lifecycle/result: `COMPLETE` / `PASS`; exact provider comparison at 2000
  optimizer steps on the RP-44 v4 Golden 200/46 population.
- Fixed pair: both use Balanced mean-NLL Matrix CE at T=1.0, identical model,
  data, prompt, objective weights, seed, schedule, batch, resolution, and
  evaluation. RP-46 uses contextual hidden state; RP-47 uses target token
  embedding.
- Artifacts: contextual file/manifest SHA256 `50179c70...e75` /
  `dfa992fc...e10`; target-embedding `646a1b60...e88` / `3cccf99e...469`.
- Execution: clean evaluator code `ffa4146`; physical GPUs 1/2; immutable
  report roots under `artifacts/representation/RP-{46,47}-*`. Acceptance
  requires exact 200/46 binding, unchanged tokenizer, and complete reports for
  both cells before drawing a provider conclusion.
- Result: both cells exited `0`, retained tokenizer `151669`, and evaluated
  200 rows/46 groups. Contextual versus target embedding yields top-1
  `0.83/0.70`, top-2 `0.96/0.875`, MRR `0.9075/0.8275`, mean diagonal gap
  `2.4240/0.7743`, and wrong-same win rate `0.935/0.855`. Mean correct-D NLL
  is nearly identical (`1.28734/1.28916`), so the 13-point top-1 difference is
  target specificity rather than basic readability. Contextual also passes the
  teacher-forced direction flip and both free continuations; target embedding
  fails direction flip and one continuation. Report SHA256 values are
  `137e48e7...ca6` / `1570978b...d8b`. Contextual hidden state is the selected
  provider for the current Qwen3 representation artifact.
- Golden comparison: the historical Golden report on the exact same ordered
  200 rows/46 groups records top-1/top-2/MRR `0.77/0.915/0.86917`, mean/median
  diagonal gap `0.19563/0.15234`. The new contextual artifact records
  `0.83/0.96/0.9075` and `2.4240/1.7070`, improving top-1 by 6 points, top-2 by
  4.5 points, and MRR by 0.0383. This establishes that the new native artifact
  exceeds the Golden baseline on the accepted same-distribution specificity
  benchmark. It is not a strict training-method A/B: Golden used custom
  Protocol-C rows and tokenizer resize, its focus-force transcript, legacy
  Matrix CE and seed `20260525`; the new artifact uses the native transcript,
  no tokenizer resize, Balanced mean-NLL/T1.0 and seed 42. Evidence-type gains
  are strongest for OCR text, color, shape boundary and texture/material;
  chart, counting, object-part and pattern remain slightly below Golden and
  should not be hidden by the aggregate improvement.

### RP-49-QWEN3-REP-MATRIXCE-BALANCED-T01-CONTEXTUAL-2000-GPU01

- Cell/status: endpoint calibration cell / `COMPLETE` / `PASS`.
- Question: with contextual hidden-state conditioning fixed, does Balanced
  mean-NLL Matrix CE at temperature `0.1` improve the 2000-step target-
  specificity result over the completed Balanced/T=1.0 artifact?
- Code/config/output: clean runtime code
  `ffa41467613b24376374e53d99be5bf29e9d0a0b`; config source/canonical SHA256
  `16b02464586f5e9f8467727e0bfb1aa873c03e48483d0d3849cbd0fb49b09851` /
  `bafdc28102de70584ba532f82c1dc233e37d43eed9880dacc3d12c4e1a8a9728`;
  immutable output
  `artifacts/representation/RP-49-qwen3-matrix-ce-balanced-t01-contextual-2000-gpu01/`.
- Model/data: local Qwen3-VL-8B-Thinking, tokenizer 151669/no resize,
  BF16/SDPA, max-pixels 262144; v4 clean-imend train SHA256 `c94a38b...443c`
  and test SHA256 `de61c731...82d`, retained manifests `a089d13d...5a8c` /
  `534f5b1e...b0`. The exact two-record image-path-only overlap report is
  `3cad19a9...c27` and is accepted without row deletion.
- Mathematics/execution: native image-plus-question prompt v1, contextual
  layer `-1`, fresh seed 42, same-image K4/B4/world2/GA4/global batch 32;
  Balanced mean-NLL/T=0.1 Matrix CE + L_gen + Norm weights `1/1/.1`, manifold
  zero; AdamW `1e-4`, 2000-step cosine/warmup 100/min ratio .1, FSDP2
  `reshard_after_forward=false`, DCP/validation every 500. RL rollout,
  reference, behavior logprobs, KL and SDPO fields are N/A.
- GPU/command: physical GPUs 0/1 UUIDs
  `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` /
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`; deterministic environment and
  `timeout 28800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m
  tgvf_rl.cli run-representation
  configs/representation/qwen3_matrix_ce_balanced_t01_contextual_2000step_gpu01.toml`
  in tmux session `tgvf-rp49-bal-t01-2k`.
- Telemetry/acceptance: W&B `mio_mi0/tgvf-e2e-rl`, run/ID
  `representation-rp49-balanced-t01-contextual-2k` /
  `rep-rp49-balanced-t01-contextual-2k`. Require finite metrics, DCP at
  500/1000/1500/2000, final Adapter, tokenizer invariance and clean teardown.
  A separately content-bound post-hoc v4-Golden 200/46 evaluation is planned
  only after the final artifact hashes exist.
- Result: all 2000 optimizer steps completed with finite train/validation
  metrics, complete two-rank DCP checkpoints at 500/1000/1500/2000, and no
  tokenizer growth (`151669` before/after). The final artifact file/manifest/run
  SHA256 values are `fcda0b96...fc14` / `3ff14e66...f49e` /
  `980e4136...1bea`; metrics SHA256 is `a5a0bd0f...82b7`. W&B completed under
  `https://wandb.ai/mio_mi0/tgvf-e2e-rl/runs/rep-rp49-balanced-t01-contextual-2k`.

### RP-50-QWEN3-REP-MATRIXCE-LEGACY-T1-CONTEXTUAL-2000-GPU23

- Cell/status: summed-NLL endpoint control / `COMPLETE` / `PASS`.
- Question: what is the new native pipeline's contextual 2000-step endpoint
  under unchanged legacy summed-NLL Matrix CE, so Balanced ceiling claims can
  be made from a real paired endpoint rather than the historical Golden run?
- Code/config/output: clean runtime code
  `ffa41467613b24376374e53d99be5bf29e9d0a0b`; config source/canonical SHA256
  `f089ff038c44796e996fa71cc55b9b29184d57cb9d4923ff262469c52517396d` /
  `0e679f85224caf55f580ebaf35051e808afdb6d424146aab5b73cb868fe39ed0`;
  immutable output
  `artifacts/representation/RP-50-qwen3-matrix-ce-legacy-t1-contextual-2000-gpu23/`.
- Paired identity: every model, data, prompt, provider, seed, batch, optimizer,
  scheduler, resolution, objective weight, validation and checkpoint field is
  identical to RP-49. The only scientific delta is legacy summed NLL/T=1.0;
  run/output and physical-device identities necessarily differ. RL-only fields
  are N/A.
- GPU/command: physical GPUs 2/3 UUIDs
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` /
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; deterministic environment and
  `timeout 28800s .venv312/bin/torchrun --standalone --nproc-per-node=2 -m
  tgvf_rl.cli run-representation
  configs/representation/qwen3_matrix_ce_legacy_contextual_2000step_gpu23.toml`
  in tmux session `tgvf-rp50-legacy-t1-2k`.
- Telemetry/acceptance: W&B `mio_mi0/tgvf-e2e-rl`, run/ID
  `representation-rp50-legacy-t1-contextual-2k` /
  `rep-rp50-legacy-t1-contextual-2k`; the same four-checkpoint, final artifact,
  tokenizer and teardown gates as RP-49 apply. The final internal evaluation
  uses the same post-hoc v4-Golden 200/46 population after artifact hashing.
- Result: all 2000 optimizer steps completed with finite train/validation
  metrics, complete two-rank DCP checkpoints at 500/1000/1500/2000, and no
  tokenizer growth (`151669` before/after). The final artifact file/manifest/run
  SHA256 values are `dfd9c8cc...43d2` / `bc5b78c2...5b7f` /
  `684bf3d6...4846`; metrics SHA256 is `7542f334...c20`. W&B completed under
  `https://wandb.ai/mio_mi0/tgvf-e2e-rl/runs/rep-rp50-legacy-t1-contextual-2k`.

### RP-51/52-QWEN3-REP-2000STEP-MATRIXCE-ENDPOINT-V4-GOLDEN

- Cell/status: paired post-hoc internal evaluations / `COMPLETE` /
  `PASS`. RP-51 evaluates contextual Balanced mean-NLL/T=0.1 artifact
  RP-49 on physical GPU 0; RP-52 evaluates contextual legacy summed-NLL/T=1.0
  artifact RP-50 on physical GPU 2.
- Frozen evaluator/data: clean code
  `ffa41467613b24376374e53d99be5bf29e9d0a0b`; v4 clean-imend test source
  `de61c731...82d`; exact Golden first-200/46-group ordered manifest
  `55e2cde5...34d8`; counterfactual manifest `4589d14f...4cc4`; seed 42,
  max-new-tokens 64 and EOS 151645. Both use the same frozen Qwen3 readout,
  main-D plus all three learned D-DeepStack branches and native continuation
  contract; no training, policy rollout, reward, GRPO, SDPO or judge exists.
- RP-51 identity: config SHA256 `0017e309...cb2`; artifact file/manifest/run
  SHA256 `fcda0b96...fc14` / `3ff14e66...f49e` /
  `980e4136...1bea`; output
  `artifacts/representation/RP-51-qwen3-internal-eval-balanced-t01-contextual-2000-v4-golden-gpu0/report.json`;
  GPU 0 UUID `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2`.
- RP-52 identity: config SHA256 `3c27bc11...011e`; artifact
  file/manifest/run SHA256 `dfd9c8cc...43d2` /
  `bc5b78c2...5b7f` / `684bf3d6...4846`; output
  `artifacts/representation/RP-52-qwen3-internal-eval-legacy-contextual-2000-v4-golden-gpu2/report.json`;
  GPU 2 UUID `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`.
- Acceptance: each report must bind exactly 200 rows/46 groups, preserve
  tokenizer length 151669, contain finite readout/query/health metrics and both
  native counterfactual cases, and exit cleanly. Comparison is withheld until
  both cells pass.
- Result: both cells contain exactly 200 rows/46 groups, preserve tokenizer
  length 151669, bind the declared artifacts/data/transcript, and exit cleanly.
  RP-51 report SHA256 `7354f04f...2376` records top-1/top-2/MRR
  `0.835/0.965/0.91083`, mean/median diagonal gap `0.42064/0.30859`, mean
  correct-D NLL `1.25962`, and wrong-same-image win/advantage `0.95/0.75032`.
  RP-52 report SHA256 `8fb7bd54...a854` records
  `0.87/0.975/0.93017`, gap `0.30798/0.23047`, mean correct-D NLL `1.37649`,
  and wrong-same-image `0.945/0.57077`. RP-51 has one value-consistent native
  continuation but neither report produces the expected direction flip; RP-52
  has no value-consistent continuation. Against RP-46 Balanced/T=1.0
  (`0.83/0.96/0.9075`, mean gap `2.4240`, expected direction flip and both
  continuations), T=0.1 does not improve the 2000-step causal endpoint. Legacy
  has the best retrieval rank but substantially weaker separation/readability;
  these metrics therefore do not support replacing Balanced/T=1.0 with either
  endpoint.

### RP-53-QWEN3-REP-MAIN-D-ONLY-PERIODIC-BOUNDARY-GPU01

- Cell/status: mandatory main-D-only two-rank periodic checkpoint/resume
  smoke / `COMPLETE` / `PASS`.
- Code/config/output: runtime `673c0e4cdcf97b74feb5b0bae944d75f85988520`;
  fresh source/canonical config SHA256 `9a3397da...aec6` /
  `a578bd23...92d0`; resume `c105e9da...7ec1` /
  `e50aa040...963`; immutable root
  `artifacts/representation/RP-53-qwen3-main-d-only-periodic-boundary-gpu01/`.
- Fixed identity: local Qwen3-VL-8B-Thinking, BF16/SDPA, no tokenizer resize,
  max pixels 262144; v4 clean-imend train/test and recorded two-image-path
  overlap; native image-question prompt v1; contextual hidden state layer -1;
  main-D-only TGVF Adapter; seed 42; same-image K4/B4/world2/GA4/global batch
  32; Balanced mean-NLL Matrix CE T=1 + L_gen + main-D Norm weights
  1/1/.1, manifold 0; AdamW 1e-4 and 2000-step cosine schedule;
  FSDP2 no-reshard.
- Execution/acceptance: GPUs 0/1, fresh process stops after step 1 following
  train, validation and DCP; a new process strictly restores step 1 and stops
  after step 2 following the same boundary. Require finite metrics, 13
  Adapter-owned FSDP leaves, no learned D-DeepStack artifact tensors, both rank
  shards, exact cursor/history restore, and clean teardown. Policy/reference,
  behavior logprobs, GRPO, SDPO, reward and judge fields are N/A.
- Result: a fresh process completed train/validation/DCP at step 1 and a new
  process strictly restored it before completing the same boundary at step 2.
  Event history is exactly `start, train1, validation1, train2, validation2`;
  both checkpoints contain both rank shards. Loaded metadata binds run identity
  `2acc4883...1d5b`, structural variant `main_d_only`, 13 Adapter-owned FSDP
  leaves/26 state names, and zero branch-owned names. Both processes exited
  cleanly and released GPUs 0/1.

### RP-54-QWEN3-REP-MAIN-D-ONLY-BALANCED-T1-CONTEXTUAL-2000-GPU01

- Cell/status: D-DeepStack ablation paired to RP-46 / `COMPLETE` /
  `PASS`.
- Code/config/output: runtime `673c0e4cdcf97b74feb5b0bae944d75f85988520`;
  source/canonical config SHA256 `dba4455d...433c` /
  `2a7bd7f4...3462`; immutable root
  `artifacts/representation/RP-54-qwen3-main-d-only-balanced-t1-contextual-2000-gpu01/`.
- Optimizer-update pair contract: every RP-46 training field remains fixed—local
  Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer 151669/no resize, max pixels
  262144, v4 clean-imend train split, native image-question prompt v1,
  contextual layer -1, seed 42, K4/B4/world2/GA4/global batch 32, Balanced
  mean-NLL Matrix CE T=1 + L_gen + Norm weights 1/1/.1, manifold 0, AdamW
  1e-4, 2000-step cosine/warmup 100/min ratio .1, FSDP2 no-reshard and
  validation/DCP every 500. The sole method change is main-D-only: Qwen source
  DeepStack remains native, while TGVF D-DeepStack contributes no observation,
  objective, gradient, trainable parameter or artifact tensor; Norm is main-D
  only.
- Diagnostic-split caveat: RP-46's historical periodic validation used the v3
  accepted validation file, whereas RP-54 follows the later user decision to
  use the v4 clean-imend test split. Periodic validation has no optimizer,
  scheduler, checkpoint-selection or early-stopping feedback, so the 2000
  training updates remain paired. The promoted RP-46 and RP-55 internal
  evaluations are exactly matched on the same immutable v4-Golden 200/46
  population. This difference must not be described as literal all-field
  training-run identity.
- Acceptance: finite steps 1--2000, validation and complete DCPs at
  500/1000/1500/2000, final Adapter artifact with its v2 variant identity,
  tokenizer unchanged, W&B telemetry including `adapter_variant=main_d_only`,
  and later the exact v4 Golden first-200/46-group internal evaluation.
  Policy/reference, behavior logprobs, GRPO, SDPO, reward and judge fields are
  N/A.
- Result: all 2000 optimizer steps completed with finite metrics and complete
  two-rank DCP checkpoints at 500/1000/1500/2000. Tokenizer length remained
  151669. Final artifact file/manifest/run SHA256 values are
  `ab971b83...d723` / `c80c616c...fc8b` / `884e417b...54b`; metrics SHA256 is
  `31dd2d26...1705`. The artifact contains exactly 26 main-attention tensors,
  binds adapter-contract v2 variant `main_d_only`, and contains no learned
  D-DeepStack branch tensor. W&B synchronized all 206 metric rows and exited
  cleanly at
  `https://wandb.ai/mio_mi0/tgvf-e2e-rl/runs/rep-rp54-main-d-only-balanced-t1-contextual-2k`.

### RP-55-QWEN3-REP-MAIN-D-ONLY-2000STEP-V4-GOLDEN

- Cell/status: post-hoc main-D-only internal evaluation / `COMPLETE` /
  `PASS`.
- Frozen evaluator/config: clean code
  `673c0e4cdcf97b74feb5b0bae944d75f85988520`; config
  `configs/representation/qwen3_main_d_only_balanced_t1_contextual_2000step_internal_evaluation_v4_golden_gpu0.toml`
  SHA256 `5eaf3a4f...bcc2`.
- Artifact identity: RP-54 file/manifest/run SHA256
  `ab971b83...d723` / `c80c616c...fc8b` / `884e417b...54b`, expected step
  2000 and adapter variant `main_d_only`.
- Evaluation identity: v4 clean-imend test source `de61c731...82d`; exact
  Golden first-200/46-group ordered manifest `55e2cde5...34d8`;
  counterfactual manifest `4589d14f...4cc4`; native image-question prompt v1,
  contextual hidden-state provider, seed 42, max-new-tokens 64 and EOS 151645.
  The source-image Qwen native DeepStack remains active; focused-D readout is
  main D only and branch-health output must be empty rather than reporting the
  zero interface placeholders.
- GPU/output/acceptance: physical GPU 0 UUID
  `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2`; immutable output
  `artifacts/representation/RP-55-qwen3-internal-eval-main-d-only-balanced-t1-contextual-2000-v4-golden-gpu0/report.json`.
  Require exact artifact/config/data/transcript identities, 200 rows/46 groups,
  tokenizer 151669, finite query/readout/main-health metrics, empty learned
  branch health, both native counterfactual variants, and clean teardown.
  Training, policy/reference replay, behavior logprobs, reward, GRPO, SDPO and
  judge are N/A.
- Result: report SHA256 `4279e4b6...e68b` binds the declared artifact, source
  config SHA256 `5eaf3a4f...bcc2`, v4 data/prompt/provider identities, exactly
  200 rows/46 groups and tokenizer length 151669 before/after. Query
  top-1/top-2/MRR are `0.49/0.78/0.70083`; mean/median diagonal gap is
  `0.05113/0.0`. Mean correct-D NLL is `1.29352`; wrong-same-image D
  win/advantage is `0.76/0.23184`. Main D is finite and non-collapsed with mean
  D/source norm ratio `1.47824`; learned branch health is exactly empty. Both
  native counterfactual continuations terminate naturally with the expected
  values and the expected direction flip. The process exited cleanly and
  released GPU 0.

### RP-56--RP-60 QWEN3 representation audited-grounding matrix

- Cells/status/result: RP-56 Balanced/T=1 contextual full-D, RP-57
  Balanced/T=1 target-token-embedding full-D, RP-58 Balanced/T=0.1
  contextual full-D, RP-59 legacy summed-NLL/T=1 contextual full-D, and
  RP-60 Balanced/T=1 contextual main-D-only / RP-56, RP-57 and RP-59
  `RUNNING`; RP-58 `CANCELLED`; RP-60 `PLANNED` / RP-58 `INVALID`, all others
  `PENDING`.
  This is a post-hoc, read-only comparison of five completed 2000-step
  representation artifacts; it performs no training, checkpoint mutation or
  model selection.
- Accepted task and code: `RPI-20260721-AUDITED-D-GROUNDING-DIAGNOSTICS`;
  implementation commit `254f8b5a1a7b0e7575bb84492dc5a8f4daef321e`.
  Evaluation code paths are clean at launch; unrelated retained untracked
  runtime directories are outside the code-identity surface. The native
  grounding manifest file SHA256 is
  `a65aa6e6038ada1436302b60440136cc98b388552a7782b48ec95ed4324938c0`.
- Shared model/prompt/data: local Qwen3-VL-8B-Thinking, BF16/SDPA, tokenizer
  length 151669 with no resize, max-pixels 262144, native image-question prompt
  `qwen3-representation-image-question-v1` with SHA256
  `bf085a6e...23c9`; v4 clean-imend test JSONL SHA256
  `de61c731...82d`; exact Golden first-200/46-group manifest SHA256
  `55e2cde5...34d8`; historical native counterfactual manifest SHA256
  `4589d14f...4cc4`; seed 42, greedy continuation, max-new-tokens 64 and EOS
  151645.
- Diagnostic population and mathematics: nine cross-image value pairs (the
  original native pair plus eight audited/matched-geometry pairs) retain the
  historical summed teacher-forced value-logprob direction contract. Thirty-six
  audited same-image supported/unsupported target pairs score
  `mean_token_logp(PRESENT) - mean_token_logp(NOT_PRESENT)` for actual D and an
  exact-shape all-zero D baseline, plus actual-D free continuations. Every
  observation is materialized fresh from the immutable artifact and contains
  atomic main D plus every artifact-supported D-DeepStack branch; RP-60 remains
  explicitly main-D-only. Frozen Qwen/Adapter eval mode, deterministic forward,
  no cache inherited from the source image or pre-D transcript.
- Cell identities:
  - RP-56 / GPU 0 UUID `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` / config
    `qwen3_balanced_contextual_2000step_internal_evaluation_audited_grounding_gpu0.toml`
    SHA256 `c2f39a8a...0422` / artifact file, manifest and training-run hashes
    `50179c70...e75`, `dfa992fc...e10`, `6c748851...5c0` / output
    `artifacts/representation/RP-56-qwen3-audited-grounding-balanced-contextual-2000-gpu0/report.json`.
  - RP-57 / GPU 1 UUID `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc` / config
    `qwen3_balanced_target_embedding_2000step_internal_evaluation_audited_grounding_gpu1.toml`
    SHA256 `389a1eef...566` / artifact hashes `646a1b60...e88`,
    `3cccf99e...469`, `5d23fafd...245` / output
    `artifacts/representation/RP-57-qwen3-audited-grounding-balanced-target-embedding-2000-gpu1/report.json`.
  - RP-58 / GPU 2 UUID `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3` / config
    `qwen3_balanced_t01_contextual_2000step_internal_evaluation_audited_grounding_gpu2.toml`
    SHA256 `0814f277...73e0` / artifact hashes `fcda0b96...c14`,
    `3ff14e66...49e`, `980e4136...bea` / output
    `artifacts/representation/RP-58-qwen3-audited-grounding-balanced-t01-contextual-2000-gpu2/report.json`.
  - RP-59 / GPU 3 UUID `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b` / config
    `qwen3_legacy_contextual_2000step_internal_evaluation_audited_grounding_gpu3.toml`
    SHA256 `3518f5a4...944` / artifact hashes `dfd9c8cc...3d2`,
    `bc5b78c2...b7f`, `684bf3d6...846` / output
    `artifacts/representation/RP-59-qwen3-audited-grounding-legacy-contextual-2000-gpu3/report.json`.
  - RP-60 / GPU 0 after RP-56 releases it / config
    `qwen3_main_d_only_balanced_t1_contextual_2000step_internal_evaluation_audited_grounding_gpu0.toml`
    SHA256 `fe8bfc44...7cd6` / artifact hashes `ab971b83...d723`,
    `c80c616c...fc8b`, `884e417b...54b` / output
    `artifacts/representation/RP-60-qwen3-audited-grounding-main-d-only-balanced-t1-contextual-2000-gpu0/report.json`.
- Runtime/commands: single-GPU `.venv-torch211-cu129` deterministic evaluation;
  `CUDA_VISIBLE_DEVICES=<cell GPU> CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false timeout 7200s
  .venv-torch211-cu129/bin/python -m tgvf_rl.cli
  run-representation-internal-evaluation <cell TOML>`. RP-56--RP-59 launch in
  detached tmux sessions; RP-60 launches only after RP-56 exits and GPU 0 is
  observed free. Sampling backend, behavior/reference logprobs, policy/reference
  replay, KL, GRPO/SDPO, optimizer, gradient, world-size batching and judge are
  N/A because this is deterministic frozen representation evaluation.
- Launch: RP-56--RP-59 started `2026-07-21T10:01:14+09:00` in detached tmux
  sessions `rp56-rp60-grounding`, `rp57-grounding`, `rp58-grounding`, and
  `rp59-grounding`; RP-60 is the second command in `rp56-rp60-grounding` and
  remains gated on RP-56's successful exit.
- RP-58 interruption: stopped by user-authorized `SIGINT` at approximately
  `2026-07-21T10:10+09:00` after the evaluator was found to perform serial
  full-prefix decoding without KV cache. No report was published and GPU 2 was
  released. This cell is not evidence and will be replaced by a newly
  identified run only after `RPI-20260721-NATIVE-D-EVAL-THROUGHPUT` parity and
  throughput gates pass.
- Acceptance: each run must bind its exact config/artifact/data/prompt and
  manifest identities, preserve tokenizer length 151669, report exactly 200
  base rows/46 groups, nine cross-image cases and 36 target-presence cases with
  finite aggregate metrics, publish an immutable report, exit cleanly and
  release its GPU. Cross-cell conclusions are withheld until all five pass.

### RP-61-QWEN3-NATIVE-D-CACHED-CONTINUATION-PARITY-GPU2

- Cell/status/result: mandatory evaluator throughput gate / `COMPLETE` /
  `FAIL`.
- Question: does one exact main-D plus all-three-D-DeepStack Qwen3 native
  context produce token-identical greedy decoding with incremental KV cache,
  with BF16 per-step full-vocabulary logit parity and a measured speedup over
  the bounded full-prefix oracle?
- Identity: accepted task `RPI-20260721-NATIVE-D-EVAL-THROUGHPUT`; code commit
  `99b3ca6fb24d69f7c1ce5946186f4e91f15c238a`; config
  `configs/representation/qwen3_cached_continuation_parity_smoke_gpu2.toml`
  SHA256 `ff47e78a2ca0f4b465b2fd3c7f71570e9da6bc7f8c46ef7fab3e4fac20c1e40a`.
- Model/artifact/data: local Qwen3-VL-8B-Thinking, BF16/SDPA, max-pixels
  262144, tokenizer 151669/no resize; completed contextual Balanced/T=1
  step-2000 artifact file/manifest/run hashes `50179c70...e75` /
  `dfa992fc...e10` / `6c748851...5c0`; v4 clean-imend source SHA256
  `de61c731...82d`; audited grounding manifest file SHA256
  `a65aa6e...8c0`; pair `baseball-pants-not-locomotive`, positive target and
  actual atomic D observation.
- Forward/math contract: deterministic fresh no-source-image D-only transcript;
  identical M-RoPE positions and attention prefix; one injected-D prefill then
  single-token cached forwards with no visual reinjection. Compare against the
  existing full-prefix oracle at every one of at most eight greedy tokens.
  Require exact generated IDs/text/stop/extraction and BF16 full-vocabulary
  logits at `atol=rtol=0.015625`. Record cached/no-cache wall time and speedup;
  timing has no minimum PASS floor.
- Runtime: one process on physical GPU 2 UUID
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`; seed 42; no sampling, optimizer,
  policy/reference replay, GRPO, SDPO or judge. Command:
  `CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
  TOKENIZERS_PARALLELISM=false timeout 1800s .venv-torch211-cu129/bin/python
  tools/smoke_qwen3_cached_continuation.py
  configs/representation/qwen3_cached_continuation_parity_smoke_gpu2.toml
  --pair-id baseball-pants-not-locomotive --atol 0.015625 --rtol 0.015625`.
- Output: immutable
  `artifacts/representation/RP-61-qwen3-native-d-cached-continuation-parity-gpu2/report.json`;
  acceptance additionally requires tokenizer invariance, finite timing, clean
  exit and GPU release.
- Result: started `2026-07-21T10:30:47+09:00` and failed cleanly at
  `2026-07-21T10:31:17+09:00`; run-log SHA256 `40360f3e...1337`. Cached and
  no-cache paths produced the same eight greedy token IDs/text/stop state, but
  step-one full-vocabulary BF16 maximum absolute logit difference was `0.5`,
  exceeding the predeclared `0.015625` tolerance. No report was published and
  GPU 2 was released. This is not a cache-parity PASS.

### RP-62-QWEN3-NATIVE-D-CACHED-CONTINUATION-DIAGNOSTIC-GPU2

- Cell/status/result: diagnostic follow-up to RP-61 / `COMPLETE` /
  `SIDE_RESULT`.
- Identity: code `01678a6badbf2f92450ade0584868d0e10702ac1`; config
  `configs/representation/qwen3_cached_continuation_diagnostic_gpu2.toml`
  SHA256 `9b29f140aeb3e7fc3be200660303f0b4cb058d62952eb180aa098ddd739fdd43`;
  all model/artifact/data/pair/GPU/generation fields are exactly RP-61.
- Question/output: retain the strict `atol=rtol=0.015625` comparison but do not
  abort on full-vocabulary mismatch; report per-trace max/mean drift, selected-
  token drift, minimum cached/oracle top-1 margins, exact output identity and
  cached/no-cache wall time. This is diagnostic evidence and cannot itself
  convert RP-61 to PASS. Immutable report:
  `artifacts/representation/RP-62-qwen3-native-d-cached-continuation-diagnostic-gpu2/report.json`.
- Command: RP-61 command with the RP-62 config plus
  `--allow-logit-mismatch-for-diagnostics`; acceptance requires exact greedy
  output identity, finite diagnostics/timing, tokenizer invariance, clean exit
  and GPU release.
- Result: report/run-log SHA256 `8c354d12...9d3` / `86774f70...399`; exact
  eight-token output identity passed, but strict logits did not. Max/mean
  full-vocabulary drift was `0.5/0.04683`, selected-token drift was `0.375`,
  and minimum cached/oracle top-1 margins were `0.25/0.0`. Cached/no-cache
  timing was `0.35049/0.34967s`, only `0.9977x`; tokenizer remained 151669 and
  GPU 2 was released. The cache implementation removes redundant prefix
  computation but is not the observed end-to-end bottleneck for this short
  lane. Optimization proceeds to repeated vision/context materialization and
  serial scoring; RP-61 remains FAIL.

### RP-63-QWEN3-REP-AUDITED-GROUNDING-BALANCED-T01-CACHED-GPU2

- Cell/status/result: replacement for interrupted RP-58 / `COMPLETE` /
  `PASS`.
- Identity: code `01678a6badbf2f92450ade0584868d0e10702ac1`; config
  `configs/representation/qwen3_balanced_t01_contextual_2000step_internal_evaluation_audited_grounding_cached_gpu2.toml`
  SHA256 `ee8a3bdc...76c3`; physical GPU 2. Artifact, data, prompt, 200/46
  population, nine cross-image pairs, 36 target-presence pairs, seed and
  64-token/EOS generation contract are exactly RP-58. The only evaluator
  change is native incremental KV-cache continuation; teacher-forced scores and
  all metric definitions are unchanged.
- Command/output: the standard deterministic single-GPU internal-evaluation
  command for the RP-63 TOML under a 7200-second timeout; immutable report
  `artifacts/representation/RP-63-qwen3-audited-grounding-balanced-t01-contextual-2000-cached-gpu2/report.json`.
- Acceptance: exact identities, tokenizer 151669, 200 rows/46 groups, nine
  cross-image cases, 36 target-presence cases, finite aggregates, clean exit
  and GPU release. RP-61's logit-parity failure remains recorded and is not
  erased by this run.
- Result: launched `2026-07-21T10:46:03+09:00`; immutable report SHA256
  `02e299f3...baa` was published with tokenizer 151669, 200 rows/46 groups,
  nine cross-image and 36 target-presence pairs. Query top-1/MRR/mean gap is
  `0.855/0.91917/0.41913`; correct-D NLL is `1.25964`; wrong-same-image win
  rate/advantage is `0.945/0.74844`. Audited cross-image direction/continuation
  is `1/9` and `6/18`; target-presence continuation is `27/72`, mean actual
  separation is `-1.60047`, positive-D contribution is `+1.78937`, and
  negative-D false-positive amplification is `+3.63569`. T=0.1 therefore
  retains strong same-distribution ranking but fails the accepted grounding
  comparison against Balanced/T=1. Run-log SHA256 is `471369c2...5b3`; the
  process exited cleanly, tokenizer invariance passed, and GPU 2 was observed
  released at `2026-07-21T10:57:37+09:00`.

### RP-64-QWEN3-CONTEXTUAL-VISION-REUSE-PARITY-GPU3

- Cell/status/result: bounded evaluator optimization gate / `COMPLETE` /
  `FAIL`.
- Question: can contextual-hidden-state diagnostics reuse the exact source
  main/DeepStack visual features already materialized for the same group,
  remove redundant per-target frozen-vision forwards, preserve the complete
  TGVF Adapter observation within BF16 tolerance, and provide a measurable
  construction speedup?
- Identity: accepted task `RPI-20260721-NATIVE-D-EVAL-THROUGHPUT`; code commit
  `6c6a5e5d26ecf6d1c9a22081913b781e895146fc`; config
  `configs/representation/qwen3_contextual_vision_reuse_parity_smoke_gpu3.toml`
  SHA256 `47aa6ab88e05ce2069f70af9c28bf30ac667e60ca8b080d611d1b1bbe031dc84`.
- Model/artifact/data: local Qwen3-VL-8B-Thinking, BF16/SDPA, max-pixels
  262144, tokenizer 151669/no resize; immutable contextual Balanced/T=0.1
  step-2000 artifact file/manifest/run hashes `fcda0b96...c14` /
  `3ff14e66...f49e` / `980e4136...1bea`; v4 clean-imend source SHA256
  `de61c731...82d`; audited grounding manifest SHA256 `a65aa6e...8c0`.
- Bounded population/math: the first audited cross-image pair plus target-
  presence pair `baseball-pants-not-locomotive`, producing six complete main-D
  plus all-three-D-DeepStack observations. The baseline reruns native Qwen3
  multimodal conditioning per target; the candidate injects the exact already-
  materialized source tensors at the same image positions with identical
  M-RoPE and attention mask and selects final hidden layer `-1`. No training,
  objective, Adapter state, prompt, sample, reward or metric changes.
- Parity/performance: require every main-D and D-DeepStack tensor to satisfy
  `atol=rtol=0.015625`, exact tensor shapes and tokenizer length 151669. Record
  synchronized baseline/candidate build seconds and speedup; retain the path
  only if parity passes and the timing benefit is material.
- Runtime: one deterministic process on physical GPU 3 UUID
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; seed 42; no sampling,
  optimizer, policy/reference replay, GRPO, SDPO or judge. Command:
  `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
  TOKENIZERS_PARALLELISM=false timeout 1800s .venv-torch211-cu129/bin/python
  tools/smoke_qwen3_contextual_vision_reuse.py
  configs/representation/qwen3_contextual_vision_reuse_parity_smoke_gpu3.toml
  --pair-id baseball-pants-not-locomotive --atol 0.015625 --rtol 0.015625`.
- Output: immutable
  `artifacts/representation/RP-64-qwen3-contextual-vision-reuse-parity-gpu3/report.json`;
  acceptance also requires clean exit and GPU release. A failure leaves the
  existing evaluator and the concurrently running RP-63 unchanged.
- Result: started `2026-07-21T10:54:24+09:00` and failed closed at
  `2026-07-21T10:55:12+09:00`; run-log SHA256 `48ee93e...9d6`. At least one
  complete Adapter observation exceeded the declared BF16 parity tolerance,
  so no report was published, no speedup claim is made, the optimization is
  not enabled in the formal evaluator, and GPU 3 was released.

### Representation-phase endpoint evidence summary

- Provider: under the same Balanced/T=1.0 2000-step identity, contextual hidden
  state outperforms target token embedding on top-1 `0.83` versus `0.70`, mean
  diagonal gap `2.4240` versus `0.7743`, and wrong-same-image win rate `0.935`
  versus `0.855`, while correct-D NLL is essentially identical. Contextual also
  passes the accepted direction-flip and both continuations. This supports the
  selected contextual provider; target token embedding remains an implemented
  and validated alternative rather than the selected artifact.
- Matrix CE: contextual Balanced/T=0.1 reaches top-1 `0.835` and legacy summed
  NLL/T=1.0 reaches `0.87`, but their mean gaps are only `0.4206/0.3080` and
  neither passes the direction flip. Contextual Balanced/T=1.0 reaches top-1
  `0.83`, mean gap `2.4240`, wrong-same-image advantage `4.6433`, and passes the
  full native causal fixture. Thus retrieval rank alone exposes a real legacy
  counter-result, but the combined specificity/readability/causal evidence
  supports retaining Balanced/T=1.0 as the current default; lowering its
  training temperature to 0.1 is not an endpoint improvement.
- D-DeepStack: removing the learned focused-D branches while preserving Qwen's
  original-image native DeepStack lowers top-1 from `0.83` to `0.49`, MRR from
  `0.9075` to `0.70083`, mean gap from `2.4240` to `0.05113`, and
  wrong-same-image win rate from `0.935` to `0.76`. Correct-D NLL remains nearly
  unchanged (`1.28734` versus `1.29352`) and the one native counterfactual still
  passes. The learned D-DeepStack branches therefore supply broad target
  discrimination/separation rather than merely making an already-correct main
  D readable; the full-D-DeepStack variant remains the production default.
- Norm: the historical weight-0.1 objective is active but does not enforce
  strict D/source equality. Across full-D endpoints, main-D/source mean ratios
  span approximately `1.15--1.44`, with branch ratios reaching approximately
  `2.0--4.15`; main-D-only is `1.478`. These magnitudes do not monotonically
  explain specificity, so no stronger norm claim or new norm objective is
  accepted from this matrix. The measured ratios remain health diagnostics and
  any change to the norm mathematics requires a separate experiment.

### BE-01-QWEN3-DIRECT-COREDEV2511-GPU0123

- Cell/status/result: original-policy CoreDev-2511 inference baseline plus
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Accepted task/code: `EVAL-QWEN3-DIRECT-BASELINE-20260721`; code commit
  `c8a6133`; direct config SHA256 `782e2518...94f1`; launcher SHA256
  `2d718cd9...e128`. Output root is
  `artifacts/evaluation/BE-01-qwen3-direct-coredev2511-gpu0123` and must not
  contain predictions from another decoding identity.
- Model/arm: immutable local Qwen3-VL-8B-Thinking; config/generation config/
  tokenizer config/tokenizer JSON SHA256 `5cd45286...3661` /
  `fe72e865...e656` / `7b501e63...a7d5` / `a5d85b6d...3c7`; no TGVF Adapter,
  tool schema, tool call, crop, agent loop, policy update, reward, GRPO, SDPO,
  reference replay or judge during inference.
- Data/scorer: seven ordered CoreDev-2511 slices, 2,511 rows, identity
  `coredev-2511-vlmevalkit-7055d301-v1`, membership SHA256
  `a461d9b4...0579`; pinned config SHA256 `eb2a34b3...a267`. The deployment and
  full artifact/prompt/scorer validators must pass before launch.
- Prompt/decoding: upstream VLMEvalKit commit `7055d301...79f`, `run.py`
  SHA256 `efe24021...8653`, inherited official per-dataset prompts, no custom
  system prompt, max pixels `262144`; vLLM 0.12, V1, TRITON attention, BF16,
  TP1 per replica, engine seed 0, temperature 1.0, top-p 0.95, top-k 20,
  maximum 40960 generated tokens, repetition/presence penalties 1.0/0.0 and
  sampling enabled. No tokenizer growth.
- Topology: four torchrun ranks and four independent model replicas on physical
  GPUs 0/1/2/3 (UUID suffixes `dfb2`, `dcdc`, `abe3`, `1f2b`). Inference and
  evaluation are separate; the later evaluator reuses the exact prediction
  TSVs and may deploy the separately identified 72B judge on GPUs 2/3.
- Throughput/resume gate: inspect early VStarBench progress, generated-token
  counts, wall time, per-GPU utilization and peak memory. After at least one
  durable per-rank checkpoint, intentionally interrupt once and require an
  exact same-directory `--reuse --reuse-aux infer` restart without regenerating
  completed rows. If the one-request path is materially underutilized, stop
  before the full suite and optimize concurrency without changing this arm.
- Command: `CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1
  VLLM_ATTENTION_BACKEND=TRITON_ATTN TOKENIZERS_PARALLELISM=false
  PYTHONHASHSEED=42 .venv312/bin/torchrun --standalone --nproc-per-node=4
  tools/run_coredev_2511_vlmevalkit.py --config
  configs/evaluation/coredev_2511_qwen3_direct_v1.json --work-dir
  artifacts/evaluation/BE-01-qwen3-direct-coredev2511-gpu0123 --mode infer`.
- Result: launched in tmux at `2026-07-21T12:49:09+09:00` and failed before
  model load or prediction at `12:49:52`. Each outer rank correctly saw one
  B200, but the spawned vLLM EngineCore inherited `LOCAL_WORLD_SIZE=4`; when it
  re-imported upstream `run.py`, the one-GPU child tripped the upstream
  `NGPU >= LOCAL_WORLD_SIZE` assertion. No row was generated and all four GPUs
  were released. This output root is retained as failed evidence and is not
  reused.

### BE-01-R1-QWEN3-DIRECT-COREDEV2511-GPU0123

- Cell/status/result: corrected original-policy inference and throughput/resume
  gate / `COMPLETE` / `FAIL`.
- Fixed identity: all model, data, prompt, sampling, resolution, four-replica
  topology and later scorer fields are exactly BE-01. Code commit `a56bdb2`;
  launcher SHA256 `1f1d7e2f...990c`; output root
  `artifacts/evaluation/BE-01-R1-qwen3-direct-coredev2511-gpu0123`.
- Bounded repair: only while constructing a nested vLLM engine, the project
  wrapper removes the outer torchrun rank/rendezvous variables inherited by
  the spawned EngineCore and restores them before dataset inference. It keeps
  each rank's already isolated `CUDA_VISIBLE_DEVICES`, does not modify the
  pinned checkout, and does not change model inputs, outputs or decoding.
  CPU tests require inner absence and exact outer restoration.
- Acceptance/command: require four successful TP1 engine initializations,
  early VStarBench throughput and utilization evidence, then the same durable
  interruption/reuse proof and full-suite inference criteria as BE-01. Command
  is identical to BE-01 except for the R1 output root.
- Result: launched at `2026-07-21T12:53:00+09:00`. The repair passed BE-01's
  assertion: all four EngineCore processes reported vLLM 0.12, BF16, TP1,
  seed 0 and independent rank-0 worlds. The nested outer-NCCL/inner-vLLM
  topology then made no progress beyond EngineCore initialization: the log
  stopped at `12:53:49`, every card remained at 2,494 MiB and 0% utilization,
  and no process reached weight loading or generated a row. The run was
  terminated rather than left idle; four orphan EngineCore PIDs required
  explicit cleanup and GPUs 0--3 returned to 0 MiB. Log SHA256
  `9d25f333...e357`. The environment-isolation unit contract remains valid,
  but nested torchrun is rejected for this baseline.

### BE-01-R2-QWEN3-DIRECT-COREDEV2511-INDEPENDENT-GPU0123

- Cell/status/result: independent-process original-policy inference plus
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Fixed identity: model, tokenizer, max-pixels, prompt, sampling and all seven
  official slice/scorer contracts remain exactly BE-01. Code commit `7e61e54`;
  launcher SHA256 `e8dbbf51...7f66`; output root
  `artifacts/evaluation/BE-01-R2-qwen3-direct-coredev2511`.
- Execution repair: remove the outer torchrun/NCCL layer. Each process owns one
  physical B200, one vLLM V1 TP1 engine and one or more complete official
  dataset slices. `--coredev-data` materializes a content-addressed config from
  the pinned full config; it may select only known complete slices in canonical
  order and cannot change the model block. Seven per-slice prediction TSVs are
  later evaluated and summarized under the same 2,511-row suite identity.
- Initial wave/topology: GPU0=`VStarBench` (191), GPU1=`HRBench4K` (200),
  GPU2=`BLINK` (420), GPU3=`OCRBench_v2` (600). After observed completion,
  schedule `MMMU_Pro_10c`, `MathVista_MINI`, and `MathVerse_MINI` onto released
  GPUs. Every slice has its own tmux session, work directory and launcher log.
- Acceptance: one engine must first reach real generation; then compare early
  examples/s, output-token rate, utilization and memory across the four slice
  types. Intentionally interrupt VStarBench after a durable checkpoint and
  require same-directory `--reuse --reuse-aux infer` recovery without changing
  completed predictions. If single-request utilization is still materially
  poor, stop the wave and replace it with batched/concurrent submission under a
  new identity before consuming the full suite.
- Command template: `CUDA_VISIBLE_DEVICES=<gpu> VLLM_USE_V1=1
  VLLM_ATTENTION_BACKEND=TRITON_ATTN TOKENIZERS_PARALLELISM=false
  PYTHONHASHSEED=42 .venv312/bin/python tools/run_coredev_2511_vlmevalkit.py
  --config configs/evaluation/coredev_2511_qwen3_direct_v1.json --coredev-data
  <official_alias> --work-dir <R2_root>/<official_alias> --mode infer`.
- Result: the GPU0 VStarBench process launched at
  `2026-07-21T13:00:54+09:00`; the independent-process topology reached model
  construction normally, then failed closed before weight loading at
  `13:01:33`. vLLM 0.12's Qwen3-VL vision implementation supports
  `FLASH_ATTN`, `TORCH_SDPA`, or ROCm AITER FA and explicitly rejected the
  inherited judge-only `TRITON_ATTN` override. No prediction was generated and
  GPU0 released cleanly. Log SHA256 `0f3c9dff...7ecd`. The 72B judge retains
  TRITON attention; that backend is not shared with the Qwen3-VL policy.

### BE-01-R3-QWEN3-DIRECT-COREDEV2511-INDEPENDENT-GPU0123

- Cell/status/result: corrected independent-process inference and
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Identity/delta: exactly BE-01-R2, code `7e61e54`, except the Qwen3-VL policy
  process no longer sets the incompatible `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
  The pinned vLLM 0.12 runtime selects its supported backend; the actual backend
  must appear in the engine log and becomes part of this run identity. No new
  package is installed and no prompt, pixels, sampling or output field changes.
  New output root is
  `artifacts/evaluation/BE-01-R3-qwen3-direct-coredev2511`.
- Wave/acceptance: start VStarBench alone on GPU0 through real generation, then
  start HRBench4K/BLINK/OCRBench_v2 independently on GPUs 1/2/3 and apply the
  same early throughput plus durable VStar interruption/reuse gates from R2.
  The R2 command template applies with the R3 root and without an attention
  environment override.
- Result: launched on GPU0 at `2026-07-21T13:03:09+09:00`. The independent
  engine selected FlashInfer for decoder attention, loaded all weights in
  about 8 seconds (`16.97 GiB` model memory), and then failed during multimodal
  KV-profile vision execution. vLLM's bundled FlashAttention2 extension raised
  `CUDA error: the provided PTX was compiled with an unsupported toolchain` on
  B200. No prediction was generated and GPU0 released cleanly. The failure
  also exposed an unnecessary default `max_seq_len=262144`. Launcher log
  SHA256 `61fc87a8...61161`.

### BE-01-R4-QWEN3-DIRECT-COREDEV2511-B200-SDPA-GPU0123

- Cell/status/result: B200-safe independent original-policy inference plus
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Fixed identity: original local Qwen3-VL-8B-Thinking, tokenizer, native
  per-dataset prompts, max pixels `262144`, sampling fields, seven CoreDev-2511
  slice membership, and later official scorers remain BE-01. Code commit
  `742a63c`; direct config SHA256 `5e0099ad...33e0d`; launcher SHA256
  `4ac6f60d...ab70`. Output root is
  `artifacts/evaluation/BE-01-R4-qwen3-direct-coredev2511`.
- Runtime delta: keep vLLM 0.12 V1's selected FlashInfer decoder attention,
  explicitly set only `mm_encoder_attn_backend=TORCH_SDPA`, and cap
  `max_model_len=65536`. The project-owned factory bridge forwards only those
  two fields without changing the pinned VLMEvalKit checkout. The 65,536-token
  engine budget preserves at least 24,576 input tokens next to the accepted
  40,960 generation cap; prompts, sampling, output token budget, scores and
  model weights are unchanged. Unit/contract result: 12 passed; Ruff passed.
- Initial topology/gate: GPU0 runs the complete 191-row VStarBench slice as one
  independent TP1 process, engine seed 0, BF16 weights, `gpu_utils=0.9`. Require
  logs to confirm `max_seq_len=65536`, decoder FlashInfer and vision SDPA, then
  require real generated rows and inspect wall time, output-token rate,
  utilization and peak memory. Only after that evidence may GPUs 1/2/3 launch
  HRBench4K/BLINK/OCRBench_v2; later free GPUs take the remaining three slices.
  The durable interruption and exact same-directory reuse gate remains required.
- Command template: `CUDA_VISIBLE_DEVICES=<gpu> VLLM_USE_V1=1
  TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 .venv312/bin/python
  tools/run_coredev_2511_vlmevalkit.py --config
  configs/evaluation/coredev_2511_qwen3_direct_v1.json --coredev-data
  <official_alias> --work-dir <R4_root>/<official_alias> --mode infer`.
- Result: GPU0 VStarBench launched at `2026-07-21T13:10:10+09:00` and
  validated the repair: `max_seq_len=65536`, decoder FlashInfer, vision
  `TORCH_SDPA`, and 16.97-GiB weights loaded in 8.05 seconds. Initialization
  then failed before prediction because TorchInductor's Triton launcher compile
  required `Python.h`, absent from the host's Python 3.12 installation. It also
  spent about 45 seconds profiling a maximum-size video although this suite is
  image-only. GPU0 released cleanly. Launcher log SHA256
  `80de60f6...88c1`.

### BE-01-R5-QWEN3-DIRECT-COREDEV2511-IMAGEONLY-GPU0123

- Cell/status/result: image-only B200-safe original-policy inference plus
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Fixed identity: all model, tokenizer, prompt, resolution, sampling, output
  budget, CoreDev membership and scorer fields remain BE-01-R4. Code commit
  `9204501`; direct config SHA256 `64740b07...daa1d`; launcher SHA256
  `4ac6f60d...ab70`; output root
  `artifacts/evaluation/BE-01-R5-qwen3-direct-coredev2511`.
- Bounded runtime repair: vLLM retains the upstream `image=24` input limit and
  adds `video=0`; accepted inputs are unchanged, while maximum-video profiling
  is forbidden. Since passwordless system package installation is unavailable,
  Ubuntu `python3.12-dev` and `libpython3.12-dev` version
  `3.12.3-1ubuntu0.15` were downloaded and extracted without installation under
  `.eval-runtime-python312-dev`; package SHA256 values are
  `0301b3a8...f4e83` and `ab00830d...50d0`. `C_INCLUDE_PATH` exposes only
  those matching Python 3.12 headers to the run. No model/runtime Python wheel
  changed. Unit/contract result: 12 passed; Ruff passed.
- Initial topology/gate: complete 191-row VStarBench on GPU0, independent TP1,
  BF16, engine seed 0. Require real prediction progress, output-token rate,
  utilization and peak memory before launching the other three GPUs. The
  interruption/reuse gate and subsequent seven-slice scheduling remain R4.
- Command template: `CUDA_VISIBLE_DEVICES=<gpu> VLLM_USE_V1=1
  C_INCLUDE_PATH=<repo>/.eval-runtime-python312-dev/root/usr/include/python3.12:<repo>/.eval-runtime-python312-dev/root/usr/include
  TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 .venv312/bin/python
  tools/run_coredev_2511_vlmevalkit.py --config
  configs/evaluation/coredev_2511_qwen3_direct_v1.json --coredev-data
  <official_alias> --work-dir <R5_root>/<official_alias> --mode infer`.
- Result: GPU0 VStarBench launched at `2026-07-21T13:15:46+09:00`.
  Python/TorchInductor compilation and image-only profiling both passed; the
  encoder profile fell from about 45 seconds to about 10 seconds. FlashInfer's
  first B200-specific JIT then failed because its NVCC invocation did not see
  the already-installed venv CUDA development headers (`cublasLt.h` and
  `nvrtc.h`). No prediction was generated and GPU0 released cleanly. Launcher
  log SHA256 `c3852c3b...7cf7`.

### BE-01-R6-QWEN3-DIRECT-COREDEV2511-FLASHINFER-JIT-GPU0123

- Cell/status/result: complete B200 JIT environment and original-policy
  throughput/resume gate / `COMPLETE` / `FAIL`.
- Fixed identity: model, tokenizer, prompt, pixels, sampling, output budget,
  model/config/code, image-only modality restriction, CoreDev membership and
  scorer are exactly BE-01-R5 (`9204501`, config `64740b07...daa1d`). New
  output root is `artifacts/evaluation/BE-01-R6-qwen3-direct-coredev2511`.
- Bounded runtime delta: extend `CPATH` with the R5 Python 3.12 headers plus the
  headers already installed in this same venv by `nvidia-cublas-cu12=12.8.4.1`,
  `nvidia-cuda-nvrtc-cu12=12.8.93`, and
  `nvidia-cuda-runtime-cu12=12.8.90`. An NVCC preprocessing fixture including
  `Python.h`, `cublasLt.h`, and `nvrtc.h` passed. No wheel, model, prompt,
  kernel source, decoding field or scorer changes.
- Initial topology/gate: complete VStarBench on GPU0. Require the FlashInfer
  B200 JIT, engine startup, real prediction progress and throughput/GPU metrics
  before launching independent GPUs 1--3. R5's interruption/reuse and full-suite
  gates remain unchanged.
- Command delta: R5 command with `C_INCLUDE_PATH` replaced by `CPATH` containing
  the two extracted Python include roots and the venv cublas, cuda_nvrtc, and
  cuda_runtime include roots; output root is R6.
- Result: GPU0 VStarBench launched at `2026-07-21T13:19:30+09:00`. NVCC
  compiled both FlashInfer B200 objects successfully; the final shared-library
  link failed because the CUDA runtime wheel provides only versioned
  `libcudart.so.12`, while FlashInfer requests the conventional development
  name `libcudart.so`. No prediction was generated and GPU0 released cleanly.
  Launcher log SHA256 `00f17203...1074`.

### BE-01-R7-QWEN3-DIRECT-COREDEV2511-FLASHINFER-LINK-GPU0123

- Cell/status/result: complete B200 FlashInfer link and original-policy
  throughput/resume gate / `PLANNED` / `PENDING`.
- Fixed identity: exactly BE-01-R6; code/config/model/data/decoding are
  unchanged. Output root is
  `artifacts/evaluation/BE-01-R7-qwen3-direct-coredev2511`.
- Bounded runtime delta: a local unversioned linker alias under
  `.eval-runtime-python312-dev/lib/libcudart.so` points to this venv's immutable
  `nvidia-cuda-runtime-cu12=12.8.90` `libcudart.so.12`; `LIBRARY_PATH` exposes
  that directory only during JIT. A `g++ -shared -lcudart -lcuda` link fixture
  passed. No system package, wheel or CUDA binary changed.
- Gate/command: R6's GPU0 VStar gate and command, adding the local runtime-lib
  directory to `LIBRARY_PATH` and using the R7 output root. Launch GPUs 1--3
  only after real rows and throughput evidence.
- Throughput side result: the runtime repair passed and all four GPUs entered
  real generation. VStar exact reuse also passed: an intentional interruption
  after 40 materialized rows resumed with exactly 151 rows pending. A 20-second
  GPU0 sample showed 19/20 observations at 88% utilization and about 160--177
  output tokens/s. However, pinned VLMEvalKit submits one request at a time;
  observed Thinking rows reached the 40,960-token cap and blocked a card for
  three to four minutes. VStar was deliberately stopped with 130/191 durable
  scalar rows, HRBench4K completed 200/200 with no missing prediction, and
  BLINK/OCR scalar jobs remain diagnostic only. None may be mixed into the
  request-seeded batched baseline.

### BE-02-QWEN3-DIRECT-COREDEV2511-B8-GPU0123

- Cell/status/result: deterministic batched original-policy inference and
  throughput/resume gate / `COMPLETE` / `PASS`.
- Fixed identity: model, tokenizer, native prompts, max pixels `262144`, all
  decoding parameters including max generated tokens `40960`, image-only
  modality restriction, vLLM 0.12 V1, decoder FlashInfer, vision SDPA,
  `max_model_len=65536`, CoreDev-2511 membership and later scorer remain R7.
  Code commit `a0c2b95`; direct config SHA256 `4623ca19...67329`; launcher
  SHA256 `ece8ddd6...c364c`; output root
  `artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8`.
- Batch/sampling identity: physical inference batch `8`, engine
  `max_num_seqs=8`, engine seed `0`. Each row's request seed is the low 31 bits
  of SHA256 over canonical compact JSON `[namespace, 0, dataset_name,
  str(canonical_index)]`, namespace
  `coredev-2511-qwen3-direct-batched-v1`. The bridge invokes the pinned scalar
  wrapper to materialize each exact request and SamplingParams, submits the
  ordered eight-request list once, and maps outputs in input order. CPU result:
  16 tests passed; Ruff/compileall and pinned deployment validation passed.
- Durability: atomically dump the complete accumulated auxiliary dictionary
  after every finished batch. A restart may redo only an unfinished batch;
  per-row seeds make its regenerated content independent of batch position.
  Scalar R7 auxiliary files are forbidden as reuse sources.
- Initial gate/topology: run complete VStarBench on GPU0 in tmux. Confirm real
  B8 generation, aggregate output throughput, utilization, memory, exact eight
  rows in the first durable checkpoint, then intentionally interrupt/reuse.
  If B8 is healthy and materially faster on long-tail overlap, stop remaining
  scalar diagnostics and launch independent B8 replicas on GPUs 1--3 before
  continuing all seven slices.
- Command template: R7 command and runtime include/library environment, with
  code/config above, `<official_alias>`, and per-slice work directory under the
  BE-02 root. No `--reuse` on first launch; exact restart adds
  `--reuse --reuse-aux infer` in the same directory.
- Result: all seven immutable inference slices completed with 2,511/2,511
  predictions and no missing or duplicate row: VStarBench 191, HRBench4K 200,
  BLINK 420, OCRBench_v2 600, MMMU_Pro_10c 300, MathVista_MINI 300 and
  MathVerse_MINI 500. Formal scoring is recorded separately as BE-03.

### BE-02-J1-QWEN25-72B-COREDEV2511-GPU01

- Cell/status/result: fail-closed CoreDev-2511 scoring service / `COMPLETE` /
  `INVALID`; inference remains BE-02 and is not rerun or modified.
- Identity: local `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct`, served
  as `Qwen2.5-72B-Instruct`; vLLM 0.12 V1, Torch 2.9.0+cu128, BF16, TP2,
  `TRITON_ATTN`, max model length 32768, prefix caching, seed 42, port 8012.
  This is the already qualified BJ-10-R3 deployment moved from physical GPUs
  2/3 to the currently free physical GPUs 0/1; it is only an answer judge and
  is never an RL reference or SDPO teacher.
- Code/data/output: scorer commit `1866293`; pinned VLMEvalKit commit
  `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`; CoreDev identity
  `coredev-2511-vlmevalkit-7055d301-v1`, manifest SHA256
  `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`;
  BE-02 prediction root
  `artifacts/evaluation/BE-02-qwen3-direct-coredev2511-b8`; service evidence
  root `artifacts/evaluation/BE-02-J1-qwen25-72b-coredev2511-gpu01`.
- Gate: service health/model-identity smoke must pass before scoring. The
  repository-owned post-score acceptance rejects API failure, judge fallback,
  random choice, missing/duplicate rows, missing metrics, and stale older-run
  reuse. Score completed slices while BE-02 inference continues, then score
  OCR/MMMU when their exact final TSVs exist and aggregate exactly 2511 rows.
- Command: BJ-10-R3 command with `CUDA_VISIBLE_DEVICES=0,1` and the evidence
  root above. Scoring uses `tools/run_coredev_2511_vlmevalkit.py --mode eval`
  with explicit `--model`, `--data`, judge alias/base URL, `--judge-api-nproc
  8`, `--judge-timeout 600`, and `--reuse-aux infer`; it never instantiates a
  second Qwen3 inference model.
- N/A: representation artifacts, D/DeepStack, rollout log probabilities,
  policy/reference, reward, GRPO/SDPO, optimizer, gradients and training batch
  fields are absent from benchmark scoring.
- Result: the native 32,768-token J1 service passed health and remains the
  accepted judge deployment, but this first scoring attempt passed the full
  Thinking transcript rather than the required final answer and exceeded the
  context limit on three requests. Its metrics are invalid; the same service
  is used correctly on BE-03 final-answer views.

### BE-02-J2-QWEN25-72B-YARN131K-COREDEV2511-GPU23

- Cell/status/result: uniform long-context fail-closed CoreDev-2511 scoring
  service / `COMPLETE` / `INVALID`; BE-02 inference is immutable and was not
  rerun. J1 service and score artifacts remain immutable comparison evidence.
- Reason for the new identity: J1 accepted every short request but rejected
  exactly three official scorer requests after six retries each because their
  input lengths were 34,249, 41,228 and 41,483 tokens, above J1's native
  32,768-token service limit. Prediction truncation and a mixed J1/J2 result
  cache are forbidden because either would change the scoring contract.
- Model/runtime: the same accepted local
  `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct` snapshot and served name
  `Qwen2.5-72B-Instruct`; config/tokenizer-config SHA256
  `14ca2173...1a29` / `5b5d4f65...9583`; vLLM `0.12.0`, Torch
  `2.9.0+cu128`, Transformers `4.57.6`, Python `3.12.3`, BF16, TP2 and
  `TRITON_ATTN` on physical GPUs 2/3. The judge remains benchmark-only and is
  never an RL reference, reward model or SDPO teacher.
- Long-context contract: static YaRN factor `4.0`, original maximum position
  `32768`, effective maximum position and service `max_model_len=131072`, and
  `rope_theta=1000000.0`. The local model README explicitly prescribes YaRN
  for inputs above 32,768 and advertises 131,072-token context. Under vLLM
  0.12 the effective HF override is
  `{"max_position_embeddings":131072,"rope_parameters":{"factor":4.0,`
  `"original_max_position_embeddings":32768,"rope_theta":1000000.0,`
  `"rope_type":"yarn"}}`; a CPU construction check resolved
  `YaRNScalingRotaryEmbedding` with 131,072 cache rows. Static YaRN can affect
  short requests, so every judge-backed slice is rescored uniformly under J2.
- Service identity: host `127.0.0.1`, port `8012`, prefix caching, vLLM
  generation config, seed `42`, GPU memory utilization `0.85`, maximum 64
  sequences, service evidence root
  `artifacts/evaluation/BE-02-J2-qwen25-72b-yarn131k-coredev2511-gpu23`.
  J1 is stopped only after its independent OCR rule-scorer gate has completed;
  J2 then owns the same already-pinned endpoint so no fallback endpoint is
  introduced.
- Data/scorer: scorer commit `1866293`, runner SHA256
  `17cb19d4...1d8`, pinned VLMEvalKit commit
  `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`, CoreDev identity
  `coredev-2511-vlmevalkit-7055d301-v1`, manifest SHA256
  `a461d9b4...0579`. The isolated J2 scoring root receives byte-identical
  copies of only the seven completed BE-02 inference runs; J1 evaluation
  auxiliaries are never copied. VStarBench, HRBench4K, BLINK,
  MMMU_Pro_10c, MathVista_MINI and MathVerse_MINI are rescored under J2;
  OCRBench_v2 remains its native rule scorer and has no judge output.
- Service command: `CUDA_VISIBLE_DEVICES=2,3` with the accepted compiler,
  include, PATH and vLLM V1 environment; launch
  `.venv312/bin/python -m vllm.entrypoints.openai.api_server --model`
  `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct --served-model-name`
  `Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8012`
  `--tensor-parallel-size 2 --dtype bfloat16 --max-model-len 131072`
  `--hf-overrides <the exact JSON above> --gpu-memory-utilization 0.85`
  `--max-num-seqs 64 --seed 42 --generation-config vllm`
  `--enable-prefix-caching`.
- Acceptance: `/health`, `/v1/models`, the deterministic
  `TGVF_JUDGE_READY` completion, and an input beyond 32,768 tokens must pass;
  all six judge-backed slices must then have zero API/fallback/random-choice
  failures and native metrics. MathVerse aggregation additionally requires
  exact restoration of its source `problem_version` provenance before suite
  aggregation. Final suite acceptance remains exactly 2,511 unique rows.
- N/A: representation artifacts, D/DeepStack, rollout log probabilities,
  policy/reference, reward, GRPO/SDPO, optimizer, gradients and training batch
  fields are absent from benchmark scoring.
- Result: service health and a real 36,042-token request passed, proving that
  the 131,072-token static-YaRN service removed J1's context rejection. The
  uniform rerun nevertheless produced one official random-fallback marker in
  HRBench4K, one in BLINK, and one exhausted-retry marker in MathVerse_MINI;
  therefore its judge-backed results are rejected. Audit then showed that the
  scorer was incorrectly receiving the full Thinking transcript instead of
  the final answer required by `docs/PROJECT_TASK.md`: 2,492/2,511 responses
  have a non-empty suffix after the last `</think>`, while 19 are truncated or
  otherwise unclosed. J2 is retained only as long-context diagnostic evidence;
  static YaRN scores are not the formal baseline.

### BE-03-QWEN3-DIRECT-COREDEV2511-FINAL-ANSWER-J1-GPU01

- Cell/status/result: formal final-answer-only CoreDev-2511 baseline scoring /
  `COMPLETE` / `PASS`; no model inference was rerun and no BE-02 source TSV was
  modified.
- Inference identity: exactly the seven completed BE-02 B8 runs: VStarBench
  `T20260721-135348`, HRBench4K `T20260721-135434`, BLINK
  `T20260721-135433`, OCRBench_v2 `T20260721-135433`, MMMU_Pro_10c
  `T20260721-143452`, MathVista_MINI `T20260721-140904`, and MathVerse_MINI
  `T20260721-135749`. Their model, prompt, image resolution, decoding,
  request-seed and membership identities remain exactly BE-02.
- Scoring-view contract: for each immutable raw `prediction`, use the non-empty
  suffix after its last native `</think>` closer. A missing closer or empty
  suffix is an invalid trajectory and is scored deterministically wrong; it is
  never sent to an LLM judge and never enters VLMEvalKit's random-choice
  fallback. Invalid MCQ rows receive a row-unique sentinel under an otherwise
  unused uppercase option label; invalid non-MCQ rows receive a row-unique
  non-answer sentinel. Valid rows never receive content in the injected option
  column. All indices, order and source fields other than `prediction` remain
  byte-value identical, except the separately documented MathVerse metadata
  provenance join below.
- Implementation identity: repository HEAD `d9c356d`; materializer CLI SHA256
  `4299cf5b...18ad`; final-answer module SHA256 `ab7807b3...3ced`; six focused
  tests plus all 37 evaluation tests and Ruff pass before materialization. The
  output root is
  `artifacts/evaluation/BE-03-qwen3-direct-coredev2511-final-answer-j1`;
  every derived TSV has an immutable SHA256/count/field manifest linking it to
  its exact BE-02 source TSV.
- MathVerse provenance: enrich only `metadata.problem_version` by exact
  `source_row_index` lookup from the accepted local MathVerse `testmini.json`;
  require 500/500 joins and record the source JSON SHA256 in the derived
  manifest. No answer, question, index or prediction inference is changed by
  this join.
- Judge identity: the accepted native-context J1 service, local
  `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`, BF16, TP2 on physical GPUs 0/1, native maximum 32,768 tokens,
  seed 42, port 8012, exactly as BE-02-J1. Judge-backed auxiliaries are freshly
  generated in BE-03; neither J1 nor J2 evaluation caches are reused. The
  final-answer audit bounds the longest judge request below 2,100 tokens.
- MathVerse judge-output contract: its pinned VLMEvalKit scorer requires the
  judge result to be exactly `0` or `1`, while Qwen2.5-72B deterministically
  starts with that verdict and may append an explanation. The fail-closed
  wrapper (SHA256 `6657e93d...7878`) canonicalizes only the pinned MathVerse
  score prompt, and only a response matching `\A\s*([01])(?=\s|\Z)`, to that
  leading digit. Extraction prompts, every other dataset, embedded/non-leading
  digits and ambiguous outputs remain unchanged and therefore retry/fail under
  the official scorer. The rejected pre-fix run observed 255/500 format
  failures; all 255 first responses met this strict contract. The formal
  MathVerse result is rerun in a new eval run with no failed cache reuse.
- OCR scoring: OCRBench_v2 uses its native deterministic rule scorer and no
  answer judge. Its evaluation-local dependencies are pinned under the J1
  evidence root. The optional exact speed path replaces only OCRBench-v2's
  module-local `nltk.edit_distance` reference with `Levenshtein.distance`;
  patch SHA256 `bc0bbc7a...1370`, pinned scorer-source guard, 342,225 exhaustive
  mixed-Unicode pairs plus 39 real-row fixtures with zero distance or metric
  mismatches, and mandatory `OCR_FAST_EDIT_ACTIVE` startup evidence.
- Scorer/runtime: pinned VLMEvalKit commit
  `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`, CoreDev identity
  `coredev-2511-vlmevalkit-7055d301-v1`, manifest SHA256
  `a461d9b4...0579`, runner SHA256 `17cb19d4...1d8`; scoring processes do not
  reserve GPUs beyond the already running J1 service.
- Acceptance: exactly 2,511 unique rows and seven native metric artifacts;
  all 2,492 closed responses scored from their final answer, all 19 invalid
  responses deterministically wrong, and zero API failure, retry exhaustion,
  LLM fallback, random-choice fallback, missing/duplicate row or stale-run
  reuse. Any violation fails closed and prevents suite aggregation.
- N/A: representation artifacts, D/DeepStack, rollout log probabilities,
  policy/reference, reward, GRPO/SDPO, optimizer, gradients and training batch
  fields are absent from benchmark scoring.
- Result: repository acceptance passed exactly 7 slices and 2,511 unique rows
  with zero inference failure, judge failure, API/retry exhaustion, fallback,
  random choice or stale-run reuse. Materialization summary SHA256
  `4dbe5fa2...5db0`; final aggregate SHA256 `40d1b9bd...25bc`.
  Primary results are VStarBench `53.93%`, HRBench4K Average/all `54.50%`
  (the generic status reporter selects cycle-0/all `56.00%`), BLINK `62.38%`,
  OCRBench-v2 Chinese/English overall `39.38%`/`49.78%`, MMMU-Pro-10c
  `64.67%`, and MathVista-MINI `77.33%`. MathVerse-MINI reports Vision
  Dominant `68%`, Text Dominant `75%`, Text Lite `67%`, Vision Only `60%`, and
  Vision Intensive `67%` (five-version macro `67.4%`). The formal MathVerse
  eval run is `T20260721-165003`; all other exact eval IDs and metric artifacts
  are recorded in `coredev-2511-eval-summary.json`.

### TOOL-01-QWEN3-CROP-ATOMIC-LIVE-GPU2

- Cell/status/result: real-model visual-tool implementation smoke / `COMPLETE` /
  `INVALID`; this is a bounded execution check, not training or benchmark
  evidence.
- Question: do both newly accepted capabilities execute against the real
  Qwen3-VL processor/vision stack: (a) immutable-source plain crop returning
  native crop visual tensors and (b) one atomic `bbox + target` call that crops
  first and then produces main D plus all three D-DeepStack branches?
- Model/processor: local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, BF16/SDPA, tokenizer
  length 151669/no resize, native chat-template SHA256
  `36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956`,
  `image_max_pixels=262144`, frozen/eval vision and merger modules.
- Representation: completed contextual Balanced/T=0.1 RP-49 step-2000 Adapter
  at
  `artifacts/representation/RP-49-qwen3-matrix-ce-balanced-t01-contextual-2000-gpu01/adapter.pt`,
  file/manifest SHA256 `fcda0b96...fc14` / `3ff14e66...f49e`. Any use of the
  target-token-embedding provider in this lane is structural smoke only and is
  not a scientific claim about that contextual artifact.
- Protocol/data: atomic schema version `crop-tgvf-tool-v1`, schema SHA256
  `41f6f99f34b0d3e9fb5b7a4166af5c367cef78214285bc56f12c6ca45e02ceb9`;
  deterministic synthetic RGB source, fixed half-open source-pixel boxes, no
  external dataset and no tokenizer growth.
- Exact-state gate: source RGB, effective crop RGB, crop pre-merge/main and all
  three DeepStack visual tensors, main D/all three D-DeepStack tensors, native
  visual positions/masks and vLLM payload are materialized once and resolved
  from the same observation store. Reprocessing during replay is forbidden.
- Code/runtime: repository HEAD `1866293` plus the accepted uncommitted
  `ATOMIC-CROP-TGVF-20260721` implementation patch; `.venv312`, Torch
  2.9.0+cu128/Transformers local lock. Physical GPU 2, seed 42,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=42`,
  `TOKENIZERS_PARALLELISM=false`.
- Acceptance: both paths finish once without autograd state; model/tokenizer
  identities and exact tensor checks pass; fused record contains main D plus
  exactly branches `(8,16,24)`; Qwen replay/vLLM packing resolves the stored
  observation; clean exit and GPU release.
- N/A: sampling/logprobs, policy/reference replay comparison, reward, GRPO,
  SDPO, optimizer, gradients, update epochs, batches and benchmark scorer.
- Command/output: repository-owned smoke command under a 900-second timeout;
  immutable JSON/log output root
  `artifacts/tools/TOOL-01-qwen3-crop-atomic-live-gpu2/`.
- Result: Qwen and the Adapter loaded, then pre-execution contract construction
  rejected the smoke-only `SamplingIdentity.backend="implementation_smoke"`;
  the project correctly requires `vllm`. No tool call, observation, report or
  mutation occurred, and GPU 2 released. The implementation smoke is replaced
  by TOOL-01-R1 with the accepted backend identity.

### TOOL-01-R1-QWEN3-CROP-ATOMIC-LIVE-GPU2

- Cell/status/result: corrected real-model visual-tool implementation smoke /
  `COMPLETE` / `INVALID`.
- Fixed identity: model, processor, Adapter, synthetic RGB, boxes, target,
  protocol, exact-state gates, physical GPU 2, determinism and all N/A fields
  are exactly TOOL-01. The sole contract correction is the smoke trajectory's
  recorded backend/version `vllm` / `0.12.0`; no generation is performed.
- Command/output: TOOL-01 command with corrected repository script and a fresh
  immutable output root
  `artifacts/tools/TOOL-01-R1-qwen3-crop-atomic-live-gpu2/`.
- Result: plain crop produced grid `(1,18,16)`, 72 merged visual tokens and all
  three DeepStack branches; atomic crop+TGVF produced grid `(1,16,18)`, 72
  main-D tokens and D-DeepStack layers `(8,16,24)`. Both observations resolved
  through the exact stored-state vLLM payload path with SHA256 identities
  `bfe384c1...04f8a` and `a184f524...c956`. The Adapter file SHA256 was
  `fcda0b9...5fc14`; tokenizer length stayed 151669; the process exited cleanly
  and GPU 2 released. Report/run-log SHA256 are
  `22522a6bd56704ed63ff0af0c083f7de14a80090898fc96d8ff2fb18ec897b36` /
  `bd8acbabddb1b194a2e6a36884a404a8d9750c25270cb8e96c2b085505130324`;
  report path is
  `artifacts/tools/TOOL-01-R1-qwen3-crop-atomic-live-gpu2/report.json`.
- Invalidity discovered by post-run audit: the smoke loaded a contextual-hidden-
  state RP-49 artifact while constructing a target-token-embedding provider.
  The tensor execution remains diagnostic evidence only. The production
  runtime now rejects this combination through the loaded representation
  manifest; TOOL-01-R2 replaces it with the matching contextual provider.

### TOOL-01-R2-QWEN3-CROP-ATOMIC-REALMODEL-GPU0

- Cell/status/result: manifest-bound real-model visual-tool implementation
  smoke / `COMPLETE` / `PASS`; this is not a live vLLM generation claim.
- Question: after adding fail-closed decoded-RGB/source-feature binding,
  processor/layout provenance, plain/atomic source-pixel replay checks, and
  representation-manifest/provider/architecture binding, do plain crop and
  atomic crop+TGVF still execute once against the real Qwen3 vision stack and
  the exact contextual RP-49 Adapter?
- Fixed identity: local Qwen3-VL-8B-Thinking model/processor and native template
  identity from TOOL-01; contextual-hidden-state RP-49 step-2000 rank-zero
  export and its manifest; BF16/SDPA; max pixels 262144; source/crop boxes,
  target, seed 42 and deterministic no-grad/frozen state unchanged. The
  contextual hidden layer/provider and Adapter/DeepStack architecture are read
  from and checked against the export manifest rather than supplied as loose
  labels.
- Exact-state acceptance: dataset/file identity remains distinct from decoded
  RGB identity; source features must name the stored decoded-RGB digest; plain
  crop records processor/layout identities; both crop record kinds bind the
  same immutable source pixels; main D and all three D-DeepStack branches,
  native positions/masks, observation resolver and vLLM packing must pass; no
  tokenizer growth; clean GPU release.
- Scope boundary: no sampling, live vLLM engine ingestion, behavior logprobs,
  policy/reference comparison, reward, GRPO/SDPO, optimizer or benchmark
  scoring. A separate integration gate is required before claiming live
  next-turn vLLM generation.
- Runtime/GPU/output: current accepted implementation worktree, `.venv312`,
  Torch 2.9.0+cu128, Transformers 4.57.6; physical GPU 0; immutable root
  `artifacts/tools/TOOL-01-R2-qwen3-crop-atomic-realmodel-gpu0` (JSON file,
  SHA256 `f3dda01af199584976a63012f1822805b49ca2babf91949e7e5c6d4c94d9f52a`).
- Command: `CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false timeout 900 .venv312/bin/python tools/smoke_qwen3_crop_tools.py --adapter artifacts/representation/RP-49-qwen3-matrix-ce-balanced-t01-contextual-2000-gpu01/adapter.pt --output artifacts/tools/TOOL-01-R2-qwen3-crop-atomic-realmodel-gpu0 --image-max-pixels 262144`.
- Result: the manifest selected `contextual_hidden_state`; exact contextual
  source-prefix capture and both tool paths completed. Source grid was
  `(1,16,16)`/64 merged tokens; plain crop was `(1,18,16)`/72 tokens with three
  DeepStack branches; atomic crop+TGVF was `(1,16,18)`/72 main-D tokens with
  branches `(8,16,24)`. Observation-record hashes were `6471b6a1...8ebc` and
  `525a896b...d37b`; stored-state vLLM payload hashes were
  `bfe384c1...4f8a` and `db233884...3844`; tokenizer length remained 151669;
  GPU 0 returned to zero allocation.

### PRL-01-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory executable Policy
  Pilot vertical-slice smoke; one optimizer step followed by a clean-process
  latest-checkpoint resume check.
- Spike-plan git revision and approval references: accepted tasks
  `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721` in `PROJECT_TASK.md` sections 0.8.2 and
  0.8.3.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; veRL validated the composed config, then its raw file-path
  module loader executed `smoke_dataset.py` before registering the module in
  `sys.modules`. Python 3.12 dataclass annotation resolution rejected that
  invalid import state. No model worker or CUDA allocation started.
- Question: can one real DeepEyes MCQ flow through native Qwen tool prompting,
  8-way vLLM rollout, optional repeated `tgvf_focus_tool` turns, exact
  rollout-owned D replay, frozen-reference replay, one FSDP2 decoder-LoRA GRPO
  update, weight synchronization, checkpoint, and clean-process auto-resume?
- Baseline and exact output path: no prior output; immutable root
  `artifacts/policy/PRL-01-qwen3-grpo-1step-auto-resume-gpu0123`, absent at
  planning time. Initial/resume logs are `launch.log` and `resume.log` there.
- Model and processor identity: local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, tokenizer length 151669,
  native chat-template SHA256 `36e042fe...8956`, native DeepStack enabled,
  original-image `max_pixels=262144`; no tokenizer resize.
- Representation checkpoint identity: contextual hidden state at layer `-1`;
  RP-49 Balanced Matrix CE T=0.1 step-2000 Adapter, file/manifest SHA256
  `fcda0b96...fc14` / `3ff14e66...f49e`, run ID
  `RP-49-QWEN3-REP-MATRIXCE-BALANCED-T01-CONTEXTUAL-2000-GPU01`, run identity
  `980e4136...e1bea`.
- N/A fields and justification: non-formal MCQ smoke, so 72B judge is not
  applicable; Qwen2.5-VL, crop tools, SDPO, full fine-tuning, nonzero KL and
  formal-Pilot quality conclusions are out of scope.
- Policy/reference initialization: current/behavior start from the original
  Qwen3 base plus fresh decoder-only PEFT LoRA rank/alpha/dropout `64/64/0`;
  vision, merger, native DeepStack, TGVF Adapter, input embedding and lm_head
  are frozen. Reference is a distinct forward-only frozen base engine with no
  LoRA. Initial LR is `1e-5`.
- Rollout policy version and allowed asynchronous staleness: exact step-0 LoRA
  snapshot is content-addressed and installed into each local tool runtime and
  vLLM before generation; staleness exactly zero. Step-1 synchronization must
  precede paired checkpoint commit.
- Code commit and worktree state: executable code commit
  `b45e3761884a35364992a03b18c9fbee3b46cf01`; launch is allowed only from its
  clean descendant containing this exact config and this ledger entry.
- Repository adapter/patch surface and hash: strict run config
  `configs/policy/runs/prl_01_qwen3_grpo_1step_autoresume_gpu0123.toml`, source
  SHA256 `c62120e8...02086`, run identity `c5b4db5a...3d201`; native AgentLoop
  YAML SHA256 `388eba9b...351e`. CPU config composition and upstream
  `validate_config(use_reference_policy=True)` passed.
- Dataset/manifest, hashes, sample rule, and n: official DeepEyes 47K snapshot
  `5546681e...6ded3`, 47,052 rows, shuffle seed 42; manifest file/content/sample
  SHA256 `3483c317...6f477` / `2ddb3635...791a` / `e3937c67...fb51e`, iteration
  SHA256 `6ee358c9...3ad8c`. Cursor 7 is audited MCQ sample
  `deepeyes47k:2fabfc60...1acca7`, image SHA256 `2988a806...654e1`, answer C;
  four prompt rows per step and exactly 8 trajectories per row.
- Native prompt/tool schema hash: accepted TGVF-only v1 prompt text SHA256
  `390b334e...99f5`; `tgvf_focus_tool` schema SHA256 `f33f61d4...6aba5`;
  maximum four admitted attempts and fifth-attempt error SHA256
  `1f649e1d...ad200` with one final recovery turn.
- Chat-template/token-fixture hash and token-ownership masks: initial expanded
  prompt is 724 tokens with 234 native image-pad positions; token-ID SHA256
  `221b8952...d4d`. Only actually sampled assistant tokens have response/loss
  mask one; template, source/D image positions, tool/error response and padding
  have mask zero. The sampled `</tool_call>` closer remains in output.
- D/DeepStack/position/mask identity: every successful call records main D and
  exact D-DeepStack branches `(8,16,24)`, native positions/layout/masks and the
  source-visual binding. Missing/dummy branches, retokenization and observation
  recomputation fail closed.
- Observation materialization/artifact identity used by all replays: source
  visual and tool observations are materialized once in the rollout worker;
  current and frozen reference consume the same content-addressed replay bundle
  and release their worker-local sidecars only after synchronous consumers
  unwind.
- RL framework/version/environment lock: upstream veRL `0.9.0.dev` commit
  `e003163181731412595257a72ec173071efb125f` (clean), Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1 and Ray 2.56.1.
- Objective equations and normalization: per original prompt use sample std and
  `A=(r-mean)/(std+1e-6)`, constant-reward groups map to zero. Reward is
  `0.8*answer + 0.2*format + 1.2*conditional_tool`; no group filtering. For
  policy tokens, `rho=exp(logp_current-logp_behavior)`, clip `[0.8,1.2]`,
  dual clip `c=3`, one global policy-token mean, one update epoch, entropy/KL
  coefficients zero and max grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  eval forward, LoRA dropout zero, no update between behavior sampling and
  replay, actual processed behavior logprobs retained. Content-addressed turn
  RNG v1 has master seed 42 and derivation SHA256 `fe8d2da9...e867e`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12.0; seed above;
  temperature/top-p `1/1`, top-k `-1`, min-p `0`, repetition/presence/frequency
  `1/0/0`, no logit processors, post-transform processed logprobs, EOS 151645,
  stop string `</tool_call>`, hard aggregate response budget 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 actor/reference/local tool model,
  FP32 reductions/optimizer state, no quantization; HF replay and local tool
  model use SDPA, vLLM uses `TRITON_ATTN` with `TORCH_SDPA` multimodal encoder;
  prefix cache off, vLLM TP1, four-rank FSDP2 with
  `reshard_after_forward=false`, no activation checkpointing/remove-padding,
  no torch.compile, colocated placement.
- Logit/logprob/loss/gradient parity tolerances: token IDs, ownership, bundle
  hashes, current/reference observation identities and current/proximal-old
  replay are exact; all behavior/current/reference selected logprobs, loss and
  gradients must be finite. At the zero-staleness pre-update point,
  `actor/pg_clipfrac` must be zero and absolute mean `actor/ppo_kl` must not
  exceed the existing BF16 diagnostic scale `0.015625`; this is a smoke gate,
  not the still-unfrozen formal reference-KL estimator.
- World size, microbatch, accumulation, and global batch: world 4; global prompt
  batch 4; prompt/rollout microbatch one per rank/engine; 32 trajectories total;
  actor/ref logprob microbatch 8 trajectories per GPU; gradient accumulation 1.
- GPUs: physical/logical GPU IDs `0,1,2,3`, four NVIDIA B200 183359 MiB; all
  were idle at planning time.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T19:17:24+09:00` to `2026-07-21T19:18:06+09:00`, about 42 seconds;
  tmux `prl01_gpu0123`, driver PID 182212, Ray task PID 195904. The planned
  resume session was not started.
- Actual GPU-hours and peak scratch use: zero GPU-hours; GPUs 0-3 remained at
  zero allocation. No checkpoint/runtime-policy state was created.
- Command: initial and clean-process resume both use
  `PYTHONPATH=src .venv312/bin/python -m tgvf_rl.cli run-policy <absolute-config> --python /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python`;
  the second invocation starts only after the first exits and must load step 1
  then terminate without another update because total steps equals one.
- Outputs: failure log `launch.log`, 117,769 bytes, SHA256
  `596097a9...d048`; no checkpoint, LoRA snapshot, metrics file or W&B run.
- Scorer/parser identity: deterministic multiple-choice exact verifier SHA256
  `2a3d5fa4...2e1c`, strict native tool parser, no judge fallback.
- Metrics: zero optimizer steps, prompts, trajectories, generated tokens, tool
  calls and observations; all training metrics N/A because dataset class loading
  failed before worker creation.
- Conclusion: file-loader compatibility failure, not a model/training result.
  R1 replaces the absolute file loader with veRL's public normal-package
  `pkg://` import route and requires a direct loader regression gate before its
  separately planned launch.

### PRL-01-R1-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory executable Policy
  Pilot vertical-slice smoke; one optimizer step followed by a clean-process
  latest-checkpoint resume check. This is a fresh identity, not a retry written
  into PRL-01.
- Spike-plan git revision and approval references: accepted tasks
  `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721` in `PROJECT_TASK.md` sections 0.8.2 and
  0.8.3.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; the public package loader succeeded, but Hydra's CLI
  serialization changed the selected question's three real LF characters into
  three literal `\\n` pairs. The exact bound-content gate rejected the changed
  text before model workers or CUDA allocation.
- Question: after correcting only the veRL Dataset import route, can the exact
  PRL-01 native-tool trajectory complete rollout, exact-observation current and
  reference replay, one FSDP2 decoder-LoRA GRPO update, checkpoint, and
  clean-process auto-resume?
- Baseline and exact output path: PRL-01 failed before CUDA allocation;
  immutable new root
  `artifacts/policy/PRL-01-R1-qwen3-grpo-1step-auto-resume-gpu0123`, confirmed
  absent at planning time. Initial/resume logs are `launch.log` and
  `resume.log` there.
- Model and processor identity: local
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, tokenizer length 151669,
  native chat-template SHA256 `36e042fe...8956`, native DeepStack enabled,
  original-image `max_pixels=262144`; no tokenizer resize.
- Representation checkpoint identity: contextual hidden state layer `-1`;
  RP-49 Balanced Matrix CE T=0.1 step-2000 Adapter, file/manifest SHA256
  `fcda0b96...fc14` / `3ff14e66...f49e`, run identity
  `980e4136...e1bea`.
- N/A fields and justification: non-formal MCQ smoke, so 72B judge is not
  applicable; Qwen2.5-VL, crop tools, SDPO, full fine-tuning, nonzero KL and
  quality conclusions are out of scope.
- Policy/reference initialization: original Qwen3 base plus fresh decoder-only
  LoRA rank/alpha/dropout `64/64/0`; vision, merger, native DeepStack, TGVF
  Adapter, input embedding and lm_head frozen. Reference is a distinct frozen
  base forward engine without LoRA. LR `1e-5`.
- Rollout policy version and allowed asynchronous staleness: exact step-0 LoRA
  snapshot is installed in vLLM and each local tool runtime before generation;
  staleness zero, with no update between sampling and replay.
- Code commit and worktree state: executable code commit
  `5f5504ff3320793b01295ccbbb6c903467c57c69`; launch is allowed only from its
  clean descendant containing this config and ledger row.
- Repository adapter/patch surface and hash: strict config
  `configs/policy/runs/prl_01_r1_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  source SHA256 `3cc6d4db...8a931`, run identity `248ed669...eb74`; the sole
  runtime delta from PRL-01 is veRL's public
  `pkg://tgvf_rl.framework.verl.smoke_dataset` Dataset import route. Its direct
  `load_extern_object` regression plus focused suite passed (23 tests), and
  upstream config composition/validation passed.
- Dataset/manifest, hashes, sample rule, and n: DeepEyes 47K snapshot
  `5546681e...6ded3`, 47,052 rows, seed 42; manifest/content/sample/iteration
  SHA256 `3483c317...6f477` / `2ddb3635...791a` /
  `e3937c67...fb51e` / `6ee358c9...3ad8c`; audited MCQ cursor 7, answer C;
  global prompt batch 4 and eight trajectories per prompt.
- Native prompt/tool schema hash: TGVF-only v1 prompt SHA256
  `390b334e...99f5`; `tgvf_focus_tool` schema SHA256 `f33f61d4...6aba5`;
  four admitted calls, deterministic fifth-call error SHA256
  `1f649e1d...ad200`, and one recovery turn.
- Chat-template/token-fixture hash and token-ownership masks: 724-token initial
  expanded prompt, 234 image-pad positions, token-ID SHA256
  `221b8952...d4d`; only actual assistant samples participate in policy loss.
- D/DeepStack/position/mask identity: every successful call records main D,
  D-DeepStack branches `(8,16,24)`, positions, layout, masks and source binding;
  missing/dummy branches, retokenization and recomputation fail closed.
- Observation materialization/artifact identity used by all replays: each
  source/tool observation is materialized once by rollout; current and frozen
  reference replay consume that exact content-addressed bundle.
- RL framework/version/environment lock: upstream veRL `0.9.0.dev` commit
  `e003163181731412595257a72ec173071efb125f`, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1, Ray 2.56.1.
- Objective equations and normalization: sample-std GRPO
  `A=(r-group_mean)/(group_std+1e-6)` with constant groups zero; reward
  `0.8*answer + 0.2*format + 1.2*conditional_tool`; ratio against recorded
  behavior logprobs, clip `[0.8,1.2]`, dual clip 3, global policy-token mean,
  one update epoch, KL/entropy coefficients zero, max grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  eval replay, LoRA dropout zero, recorded post-transform behavior logprobs,
  content-addressed turn RNG v1, master seed 42, derivation SHA256
  `fe8d2da9...e867e`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12.0; temperature/top-p
  `1/1`, top-k `-1`, min-p 0, repetition/presence/frequency `1/0/0`, no logit
  processors, post-transform logprobs, EOS 151645, `</tool_call>` stop, total
  response budget 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 actor/reference/tool model, FP32
  reductions/optimizer, no quantization; HF SDPA, vLLM TRITON_ATTN with
  TORCH_SDPA multimodal encoder; TP1, four-rank FSDP2,
  `reshard_after_forward=false`, no activation checkpointing/remove-padding,
  no torch.compile, colocated placement.
- Logit/logprob/loss/gradient parity tolerances: exact token/mask/bundle/current
  and reference observation identities; finite selected logprobs/loss/gradient;
  at pre-update zero staleness, clip fraction zero and absolute mean PPO KL no
  greater than 0.015625.
- World size, microbatch, accumulation, and global batch: world 4; global
  prompt batch 4; prompt/rollout microbatch one per rank; 32 trajectories;
  actor/reference logprob microbatch 8 per GPU; gradient accumulation 1.
- GPUs: physical/logical IDs `0,1,2,3`, four B200 183359 MiB; availability must
  be rechecked immediately before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T19:23:35+09:00` to `2026-07-21T19:24:17+09:00`, about 42
  seconds; tmux `prl01_r1_gpu0123`, driver PID 201432, Ray task PID 215114.
  The planned resume session was not started.
- Actual GPU-hours and peak scratch use: zero GPU-hours; GPUs 0--3 remained at
  zero allocation. No checkpoint/runtime-policy state was created.
- Command: both processes use `PYTHONPATH=src .venv312/bin/python -m
  tgvf_rl.cli run-policy <absolute-R1-config> --python
  /nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/.venv312/bin/python`; resume
  starts only after the first exits and must load step 1 without a second
  update because total steps is one.
- Outputs: failure log `launch.log`, 115,979 bytes, SHA256
  `f1ba74b7...1d6e`; no checkpoint, LoRA snapshot, metrics file or W&B run.
- Scorer/parser identity: deterministic MCQ exact verifier SHA256
  `2a3d5fa4...2e1c`; strict native tool parser; no judge fallback.
- Metrics: zero optimizer steps, prompts, trajectories, generated tokens, tool
  calls and observations; all training metrics N/A because exact content
  binding failed before worker creation.
- Conclusion: Hydra free-text transport failure, not a data, model or training
  result. R2 carries question and ground truth as strict UTF-8 Base64 across
  the Hydra/Ray boundary; it must pass a real multiline compose regression and
  use a separately planned identity before launch.

### PRL-01-R2-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory executable Policy
  Pilot vertical-slice smoke plus clean-process latest-checkpoint resume.
- Spike-plan git revision and approval references: accepted tasks
  `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721`, `PROJECT_TASK.md` sections 0.8.2--0.8.3.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; exact Dataset binding and four-rank worker/model creation
  passed, but Hydra doubled all seven backslashes in the decoder-only LoRA
  regex. PEFT therefore matched zero modules and rejected adapter injection
  before rollout.
- Question: after byte-exact free-text transport, can the accepted native-tool
  trajectory complete 8-way rollout, exact-observation current/reference
  replay, one FSDP2 decoder-LoRA GRPO update, checkpoint and clean resume?
- Baseline and exact output path: PRL-01/R1 both failed before CUDA allocation;
  new immutable root
  `artifacts/policy/PRL-01-R2-qwen3-grpo-1step-auto-resume-gpu0123`, confirmed
  absent; logs `launch.log` and `resume.log`.
- Model and processor identity: local Qwen3-VL-8B-Thinking, tokenizer 151669,
  chat-template SHA256 `36e042fe...8956`, native DeepStack on,
  `max_pixels=262144`, no tokenizer resize.
- Representation checkpoint identity: contextual hidden state layer `-1`;
  RP-49 Balanced Matrix CE T=0.1 step 2000, Adapter file/manifest SHA256
  `fcda0b96...fc14` / `3ff14e66...f49e`, run identity
  `980e4136...e1bea`.
- N/A fields and justification: bounded non-formal MCQ smoke; 72B judge,
  Qwen2.5-VL, crop, SDPO, full tuning, nonzero KL and quality claims excluded.
- Policy/reference initialization: original Qwen3 plus fresh decoder-only LoRA
  `r=64`, alpha 64, dropout 0; vision, merger, native DeepStack, Adapter,
  embeddings and lm_head frozen; distinct frozen base reference; LR `1e-5`.
- Rollout policy version and allowed asynchronous staleness: exact step-0 LoRA
  synchronized before generation; staleness zero; no intervening update.
- Code commit and worktree state: executable commit
  `9f5b29ac8b6b4b9be7cd9cfabd3a2c2ed5d1f75c`; only this exact config and
  ledger descendant may be present as tracked launch changes.
- Repository adapter/patch surface and hash: config
  `configs/policy/runs/prl_01_r2_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `3c0a3fee...debae4`, run identity `0b2c087e...1c830`; Dataset binding
  schema `tgvf-verl-selected-sample-v2` transports question and ground truth as
  strict UTF-8 Base64. Ruff and the real Hydra compose suite passed, 22 tests.
- Dataset/manifest, hashes, sample rule, and n: DeepEyes 47K snapshot
  `5546681e...6ded3`, 47,052 rows, seed 42; manifest/content/sample/iteration
  hashes `3483c317...6f477` / `2ddb3635...791a` /
  `e3937c67...fb51e` / `6ee358c9...3ad8c`; audited cursor 7 MCQ answer C;
  batch 4, eight trajectories per prompt.
- Native prompt/tool schema hash: TGVF-only prompt `390b334e...99f5`, tool
  schema `f33f61d4...6aba5`, four calls plus deterministic fifth-call error
  `1f649e1d...ad200` and one recovery turn.
- Chat-template/token-fixture hash and token-ownership masks: 724-token prompt,
  234 image pads, token IDs `221b8952...d4d`; only sampled assistant tokens
  have policy/loss mask one.
- D/DeepStack/position/mask identity: exact main D, branches `(8,16,24)`,
  positions/layout/masks/source binding per successful call; no dummy branches,
  retokenization or recomputation.
- Observation materialization/artifact identity used by all replays: rollout
  materializes once; current and reference replay share the content-addressed
  recorded bundle.
- RL framework/version/environment lock: veRL `0.9.0.dev` commit
  `e003163181731412595257a72ec173071efb125f`, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1, Ray 2.56.1.
- Objective equations and normalization: sample-std advantage with epsilon
  `1e-6` and constant group zero; reward `0.8*answer + 0.2*format +
  1.2*conditional_tool`; recorded behavior ratio, clip `[0.8,1.2]`, dual clip
  3, global policy-token mean, one epoch, KL/entropy zero, max grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  eval replay, dropout zero, post-transform behavior logprobs, master seed 42,
  content-addressed RNG derivation `fe8d2da9...e867e`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12.0, T/p `1/1`, top-k
  `-1`, min-p 0, penalties `1/0/0`, no processors, post-transform logprobs,
  EOS 151645, stop `</tool_call>`, aggregate response cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 model/KV, FP32 reductions and
  optimizer, no quantization; HF SDPA, vLLM TRITON_ATTN/TORCH_SDPA; TP1,
  four-rank FSDP2, no reshard/checkpointing/remove-padding/compile, colocated.
- Logit/logprob/loss/gradient parity tolerances: exact identities and finite
  numerics; pre-update clip fraction zero and absolute mean PPO KL <= 0.015625.
- World size, microbatch, accumulation, and global batch: world 4; prompt batch
  4; prompt/rollout microbatch 1 per rank; actor/reference logprob microbatch 8
  trajectories per GPU; 32 trajectories; accumulation 1.
- GPUs: physical/logical IDs `0,1,2,3`, four B200 183359 MiB; recheck idle
  immediately before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T19:31:31+09:00` to `2026-07-21T19:33:01+09:00`, 90 seconds;
  tmux `prl01_r2_gpu0123`, driver PID 223066, Ray task PID 236504. Resume was
  not started.
- Actual GPU-hours and peak scratch use: less than 0.1 GPU-hours; transient
  model-init allocation was observed on all four GPUs, with rank-0 device use
  3.24 GiB after FSDP. All GPUs returned to zero; no checkpoint state exists.
- Command: `PYTHONPATH=src .venv312/bin/python -m tgvf_rl.cli run-policy
  <absolute-R2-config> --python <absolute-.venv312-python>` for both processes;
  resume starts only after successful step 1 and must not update again.
- Outputs: failure log `launch.log`, 146,296 bytes, SHA256
  `d806f61f...f212f`; no checkpoint, LoRA snapshot, metrics or W&B run.
- Scorer/parser identity: deterministic MCQ exact verifier
  `2a3d5fa4...2e1c`, strict native tool parser, no judge.
- Metrics: zero optimizer steps, rollouts, generated tokens and tool calls;
  model construction stopped at PEFT adapter injection.
- Conclusion: Hydra regex-transport failure, not a model topology or training
  result. The native model exposes exactly 36x7 decoder targets and no
  vision/embedding/lm-head targets. R3 uses a mathematically equivalent
  Hydra-safe `[.]`/`[0-9]+` regex, with compose/scope regression gates, under a
  separate identity.

### PRL-01-R3-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory 4-GPU executable
  Policy Pilot vertical slice plus clean-process checkpoint resume.
- Spike-plan git revision and approval references: accepted
  `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721`, task sections 0.8.2--0.8.3.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; exact data, decoder LoRA and FSDP2 model construction passed.
  vLLM then rejected upstream veRL's `VLLM_BATCH_INVARIANT=1` because the
  accepted rollout backend is `TRITON_ATTN`, whereas vLLM 0.12 permits that
  mode only with FlashAttention/FlashInfer backends.
- Question: with exact Dataset text and decoder-only LoRA regex transport now
  proven, can one step complete rollout, exact current/reference replay,
  FSDP2 GRPO update, synchronized checkpoint and clean resume?
- Baseline and exact output path: PRL-01/R1/R2 failed before rollout; new root
  `artifacts/policy/PRL-01-R3-qwen3-grpo-1step-auto-resume-gpu0123`, confirmed
  absent; logs `launch.log` and `resume.log`.
- Model and processor identity: local Qwen3-VL-8B-Thinking, tokenizer 151669,
  template `36e042fe...8956`, native DeepStack, max pixels 262144, no resize.
- Representation checkpoint identity: RP-49 contextual layer `-1`, Balanced
  Matrix CE T=0.1 step 2000; file/manifest `fcda0b96...fc14` /
  `3ff14e66...f49e`, run identity `980e4136...e1bea`.
- N/A fields and justification: bounded MCQ smoke; no judge, Qwen2.5-VL,
  crop, SDPO, full tuning, nonzero KL or quality conclusion.
- Policy/reference initialization: original base plus fresh decoder-only LoRA
  rank/alpha/dropout `64/64/0`, LR `1e-5`; vision, merger, DeepStack, Adapter,
  embeddings and head frozen; separate frozen base reference.
- Rollout policy version and allowed asynchronous staleness: exact synchronized
  step-0 snapshot, staleness zero, no update before replay.
- Code commit and worktree state: executable commit
  `4610afc343b3fde00ae4f37fea5495fd6f7a5ac7`; only config/ledger descendant
  tracked changes allowed at launch.
- Repository adapter/patch surface and hash: R3 config SHA256
  `ca918b92...f64f43`, run identity `907b8340...d7cdf`; Dataset binding v2 and
  Hydra-safe equivalent decoder regex
  `^model[.]language_model[.]layers[.][0-9]+[.]...$`. Ruff plus focused real
  compose/scope suites passed, 52 tests.
- Dataset/manifest, hashes, sample rule, and n: DeepEyes 47K snapshot
  `5546681e...6ded3`, 47,052 rows, seed 42; hashes
  `3483c317...6f477` / `2ddb3635...791a` / `e3937c67...fb51e` /
  `6ee358c9...3ad8c`; cursor 7, MCQ answer C, prompt batch 4, n=8.
- Native prompt/tool schema hash: TGVF-only v1 `390b334e...99f5`, schema
  `f33f61d4...6aba5`, four calls, fifth-call error `1f649e1d...ad200`.
- Chat-template/token-fixture hash and token-ownership masks: 724 initial
  tokens, 234 image pads, IDs `221b8952...d4d`; loss only on sampled assistant
  tokens.
- D/DeepStack/position/mask identity: recorded exact main D, branches
  `(8,16,24)`, layout/positions/masks/source binding; no recomputation.
- Observation materialization/artifact identity used by all replays: rollout
  materializes once and both replay roles consume the same addressed bundle.
- RL framework/version/environment lock: veRL e003163, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1, Ray 2.56.1.
- Objective equations and normalization: sample-std GRPO epsilon `1e-6`,
  constant groups zero; reward `0.8/0.2/1.2`; recorded behavior ratio, clip
  `[0.8,1.2]`, dual clip 3, global token mean, one epoch, KL/entropy zero,
  gradient norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: deterministic
  eval replay, dropout zero, post-transform behavior logprobs, seed 42 and RNG
  derivation `fe8d2da9...e867e`.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12.0; `1/1/-1/0`, penalties
  `1/0/0`, no processors, EOS 151645, `</tool_call>` stop, cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/FP32 optimizer, no quantization;
  HF SDPA, vLLM TRITON_ATTN/TORCH_SDPA; TP1, 4-rank FSDP2, no reshard,
  activation checkpoint, remove padding or compile; colocated.
- Logit/logprob/loss/gradient parity tolerances: exact identities, finite
  numerics, pre-update clipfrac zero and absolute mean PPO KL <=0.015625.
- World size, microbatch, accumulation, and global batch: world 4; prompt batch
  4; prompt/rollout microbatch 1/rank; logprob microbatch 8/GPU; 32
  trajectories; accumulation 1.
- GPUs: physical/logical 0--3, B200 183359 MiB; recheck idle before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T19:40:18+09:00` to `2026-07-21T19:42:31+09:00`, 133 seconds;
  tmux `prl01_r3_gpu0123`, driver PID 247482, Ray task PID 261025. Resume was
  not started.
- Actual GPU-hours and peak scratch use: less than 0.15 GPU-hours; observed
  peak device allocation was about 6.1 GiB/GPU during FSDP/vLLM startup. All
  GPUs returned to zero and no checkpoint state exists.
- Command: standard absolute R3 `run-policy` command under `.venv312`; resume
  only after step 1 and must perform no second update.
- Outputs: failure log `launch.log`, 145,560 bytes, SHA256
  `52246142...4cc19`; no checkpoint, metrics or W&B run.
- Scorer/parser identity: exact MCQ verifier `2a3d5fa4...2e1c`, strict native
  parser, no judge.
- Metrics: zero optimizer steps, rollouts, generated tokens and tool calls;
  vLLM core stopped during device initialization.
- Conclusion: attention/determinism compatibility failure, not a training
  result. R4 keeps TRITON_ATTN and actor/reference deterministic replay but
  truthfully identifies multi-turn rollout as content-addressed request-seeded
  and batch-sensitive: rollout full determinism false and explicit
  `VERL_FULL_DETERMINISM=0`, `VLLM_BATCH_INVARIANT=0`. Actual sampled behavior
  logprobs remain authoritative. Direct vLLM no-op and 75 focused tests passed.

### PRL-01-R4-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory 4-GPU Policy Pilot
  one-step vertical slice and clean-process resume.
- Spike-plan git revision and approval references: accepted task sections
  0.8.2--0.8.3, identities `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721`.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: does the exact R3 contract complete when the rollout truthfully
  uses request-seeded, batch-sensitive TRITON execution while replay remains
  fully deterministic?
- Baseline and exact output path: R3 reached vLLM init; R4 root
  `artifacts/policy/PRL-01-R4-qwen3-grpo-1step-auto-resume-gpu0123` contains
  the immutable failed `launch.log`; resume was not started.
- Model and processor identity: Qwen3-VL-8B-Thinking local path, tokenizer
  151669, template `36e042fe...8956`, DeepStack on, max pixels 262144.
- Representation checkpoint identity: RP-49 contextual layer -1, Balanced CE
  T=0.1 step 2000; file/manifest/run hashes `fcda0b96...fc14` /
  `3ff14e66...f49e` / `980e4136...e1bea`.
- N/A fields and justification: bounded exact-MCQ smoke; no judge, Qwen2.5,
  crop, SDPO, full tuning, nonzero KL or quality inference.
- Policy/reference initialization: base plus decoder-only LoRA `64/64/0`, LR
  `1e-5`; all non-decoder scopes frozen; separate unadapted frozen reference.
- Rollout policy version and allowed asynchronous staleness: synchronized
  step-0 behavior snapshot, zero staleness, no update before replay.
- Code commit and worktree state: executable commit
  `4b0fb90a1e036b31ecd7a0dde0872903ea141089`; clean descendant may add only
  this exact config and ledger row.
- Repository adapter/patch surface and hash: R4 config SHA256
  `3ceb74b4...414e1`, run identity `7b344259...ade804`; R3 inputs unchanged.
  Operational delta: rollout full determinism false, environment
  `VERL_FULL_DETERMINISM=0`/`VLLM_BATCH_INVARIANT=0`; actor/ref full
  determinism true. Direct vLLM gate and 75 focused tests passed.
- Dataset/manifest, hashes, sample rule, and n: unchanged exact R3 DeepEyes
  snapshot/hash tuple, seed 42, cursor 7 MCQ C; batch 4, n=8.
- Native prompt/tool schema hash: `390b334e...99f5` / `f33f61d4...6aba5`, four
  calls and fifth-call error `1f649e1d...ad200`.
- Chat-template/token-fixture hash and token-ownership masks: 724 tokens, 234
  image pads, IDs `221b8952...d4d`; sampled assistant tokens only.
- D/DeepStack/position/mask identity: exact recorded main D plus `(8,16,24)`
  branches/layout/positions/masks; no recomputation.
- Observation materialization/artifact identity used by all replays: one
  rollout-owned content-addressed bundle shared by current/reference replay.
- RL framework/version/environment lock: veRL e003163, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1, Ray 2.56.1.
- Objective equations and normalization: unchanged R3 sample-std GRPO,
  `0.8/0.2/1.2` reward, behavior ratio, `[0.8,1.2]`, dual 3, global token
  mean, one epoch, KL/entropy zero, grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: rollout
  identity `request_seeded_batch_sensitive_v1` with content-addressed per-turn
  seed and actual behavior logprobs; replay deterministic, dropout zero.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12, seed 42 derivation
  `fe8d2da9...e867e`, `1/1/-1/0`, penalties `1/0/0`, none, post-transform,
  EOS 151645, stop closer, cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/no quantization; HF SDPA, vLLM
  TRITON_ATTN/TORCH_SDPA; TP1, 4-rank FSDP2, no reshard/checkpointing/remove
  padding/compile; colocated.
- Logit/logprob/loss/gradient parity tolerances: exact identities, finite
  values, pre-update clipfrac zero and absolute mean PPO KL <=0.015625.
- World size, microbatch, accumulation, and global batch: world 4, prompt batch
  4, prompt/rollout microbatch 1/rank, logprob microbatch 8/GPU, 32
  trajectories, accumulation 1.
- GPUs: physical/logical 0--3, B200 183359 MiB; recheck before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T19:49:38+09:00` to the final logged output at
  `2026-07-21T19:53:48+09:00`, 250 seconds; tmux
  `prl01_r4_gpu0123`, Ray task PID 286796. The failed process tree was
  explicitly stopped after it stalled during teardown; resume was not started.
- Actual GPU-hours and peak scratch use: less than 0.28 GPU-hours; observed
  peak device allocation was about 94.7 GiB/GPU during vLLM CUDA-graph capture.
  All four GPUs returned to zero allocation.
- Command: standard absolute R4 `run-policy` command in `.venv312`; clean
  resume only after successful step 1 and must make no second update.
- Outputs: failed `launch.log`, 219,578 bytes, SHA256
  `66b5351e...3d15`; no checkpoint, metrics file, resume log, or W&B run.
- Scorer/parser identity: exact MCQ `2a3d5fa4...2e1c`, strict native parser.
- Metrics: zero optimizer steps, completed rollouts, generated trajectories,
  rewards, or tool observations. Dataset, decoder LoRA, four-rank FSDP2, and
  vLLM model initialization all passed before the first real LoRA forward.
- Conclusion: Triton's lazily compiled CUDA launcher inherited Python 3.12
  sysconfig's nonexistent `/usr/include/python3.12` and failed on `Python.h`.
  This is an environment-preflight failure, not an RL result. R5 binds
  `/usr/bin/gcc`, `/usr/bin/g++`, and the already accepted repo-local Python
  3.12 development headers; a matching compiler preflight passed. No
  `LIBRARY_PATH` change is needed because the same command resolves `libcuda`.

### PRL-01-R5-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory 4-GPU Policy Pilot
  one-step vertical slice and clean-process resume.
- Spike-plan git revision and approval references: accepted task sections
  0.8.2--0.8.3, identities `POLICY-PILOT-V1-VERTICAL-SLICE-20260721` and
  `POLICY-PILOT-V1-FOUR-GPU-20260721`.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`.
- Question: with the accepted Python 3.12 development headers explicitly
  inherited by every Ray/vLLM child, does the fixed R4 contract complete one
  real rollout/replay/update and a clean-process no-extra-update resume?
- Baseline and exact output path: R4 reached vLLM's first real LoRA forward;
  fresh root `artifacts/policy/PRL-01-R5-qwen3-grpo-1step-auto-resume-gpu0123`,
  absent at planning time; `launch.log`/`resume.log`.
- Model and processor identity: Qwen3-VL-8B-Thinking local path, tokenizer
  151669, template `36e042fe...8956`, DeepStack on, max pixels 262144.
- Representation checkpoint identity: RP-49 contextual layer -1, Balanced CE
  T=0.1 step 2000; file/manifest/run hashes `fcda0b96...fc14` /
  `3ff14e66...f49e` / `980e4136...e1bea`.
- N/A fields and justification: bounded exact-MCQ smoke; no judge, Qwen2.5,
  crop, SDPO, full tuning, nonzero KL or quality inference.
- Policy/reference initialization: base plus decoder-only LoRA `64/64/0`, LR
  `1e-5`; all non-decoder scopes frozen; separate unadapted frozen reference.
- Rollout policy version and allowed asynchronous staleness: synchronized
  step-0 behavior snapshot, zero staleness, no update before replay.
- Code commit and worktree state: executable commit
  `08a0a6875838f29c4142203d90c99fdf8ac76cea`; clean descendant may add only
  this exact config and ledger row.
- Repository adapter/patch surface and hash: R5 config SHA256
  `88b6d24f...d0c`, run identity `cff1864c...f9ea`; R4 inputs unchanged.
  Operational delta is only fixed `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, and
  `CPATH=.deps/python312-dev/root/usr/include:{same}/python3.12`. Header and
  Triton-shaped `-lcuda` compiler preflights passed; no `LIBRARY_PATH` delta.
- Dataset/manifest, hashes, sample rule, and n: unchanged exact R4 DeepEyes
  snapshot/hash tuple, seed 42, cursor 7 MCQ C; batch 4, n=8.
- Native prompt/tool schema hash: `390b334e...99f5` / `f33f61d4...6aba5`, four
  calls and fifth-call error `1f649e1d...ad200`.
- Chat-template/token-fixture hash and token-ownership masks: 724 tokens, 234
  image pads, IDs `221b8952...d4d`; sampled assistant tokens only.
- D/DeepStack/position/mask identity: exact recorded main D plus `(8,16,24)`
  branches/layout/positions/masks; no recomputation.
- Observation materialization/artifact identity used by all replays: one
  rollout-owned content-addressed bundle shared by current/reference replay.
- RL framework/version/environment lock: veRL e003163, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, PEFT 0.19.1, Ray 2.56.1.
- Objective equations and normalization: sample-std GRPO; reward
  `0.8/0.2/1.2`; behavior ratio, clip `[0.8,1.2]`, dual clip 3, global token
  mean, one epoch, KL/entropy zero, grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: rollout
  `request_seeded_batch_sensitive_v1` with content-addressed per-turn seeds and
  actual behavior logprobs; deterministic replay, dropout zero.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12, seed 42 derivation
  `fe8d2da9...e867e`, `1/1/-1/0`, penalties `1/0/0`, none, post-transform,
  EOS 151645, stop closer, cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/no quantization; HF SDPA, vLLM
  TRITON_ATTN/TORCH_SDPA; TP1, 4-rank FSDP2, no reshard/checkpointing/remove
  padding/compile; colocated.
- Logit/logprob/loss/gradient parity tolerances: exact identities, finite
  values, pre-update clipfrac zero and absolute mean PPO KL <=0.015625.
- World size, microbatch, accumulation, and global batch: world 4, prompt batch
  4, prompt/rollout microbatch 1/rank, logprob microbatch 8/GPU, 32
  trajectories, accumulation 1.
- GPUs: physical/logical 0--3, B200 183359 MiB; observed idle immediately before
  planning and must be rechecked before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T20:02:31+09:00` to `2026-07-21T20:07:52+09:00`, 321 seconds;
  tmux `prl01_r5_gpu0123`, driver PID 307883, Ray task PID 321625, exit 1.
  Resume was not started.
- Actual GPU-hours and peak scratch use: less than 0.36 GPU-hours; observed
  peak device allocation was about 94.7 GiB/GPU. All GPUs returned to zero.
- Command: absolute R5 `run-policy` command under `.venv312`; clean resume only
  after step 1 and must make no second update.
- Outputs: failed `launch.log`, 279,793 bytes, SHA256
  `45384ec5...d1ad`; no checkpoint, metrics file, resume log, or W&B run.
- Scorer/parser identity: exact MCQ `2a3d5fa4...2e1c`, strict native parser.
- Metrics: zero optimizer steps, completed rollouts, trajectories, rewards, or
  tool observations. The Python-header compiler boundary passed.
- Conclusion: R5 exposed the next independent upstream compatibility failure:
  during vLLM LoRA CUDA-graph warmup on B200/sm100,
  `torch.ops.vllm.lora_expand` reached Triton 3.5's automatic warp-specialized
  pipeline with PDL/GDC enabled and aborted with
  `tt.elementwise_inline_asm op pipeliner doesn't know how to predicate` and
  `LLVM ERROR: Fatal pipeliner error`. This is a rollout-kernel compiler
  failure, not an RL result.

### PRL-DIAG-01-VLLM-LORA-NO-PDL-GPU0

- Cell/matrix ID and mandatory/diagnostic class: diagnostic single-GPU
  synthetic vLLM LoRA expand-kernel gate.
- Spike-plan git revision and approval references: Policy Pilot vertical-slice
  authorization and R5's sealed sm100 LoRA compiler failure.
- Lifecycle status: `COMPLETE`.
- Result: `INVALID`.
- Question: is disabling optional PDL/GDC sufficient for vLLM 0.12's BF16 LoRA
  expand kernel to compile on B200 and match a Torch matmul oracle?
- Baseline and exact output path: R5 fatal pipeline failure; fresh diagnostic
  log `artifacts/policy/PRL-DIAG-01-vllm-lora-no-pdl-gpu0.log`.
- Model and processor identity: no model or processor; synthetic one-token,
  rank-64, hidden-4096 BF16 kernel fixture.
- Representation checkpoint identity: N/A; no TGVF Adapter is loaded.
- N/A fields and justification: no dataset, prompt, transcript, reward,
  rollout, replay, optimizer, checkpoint, reference policy, or judge.
- Policy/reference initialization: N/A; synthetic tensors only.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: commit
  `519a0c4aeb8e7ed99cbf9b8e25024c701177d988`; only this ledger row may be a
  tracked descendant at launch.
- Repository adapter/patch surface and hash: no repo patch; installed vLLM
  `lora_expand_op.py` SHA256 `d13f0c42...4a87f` and `utils.py` SHA256
  `b02898c1...a073` remain unchanged. The process-local diagnostic replaces
  only the imported `lora_expand_op.supports_pdl` callable with false.
- Dataset/manifest, hashes, sample rule, and n: N/A.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: Python 3.12.3, Torch 2.9.0+cu128,
  vLLM 0.12.0, Triton 3.5.0; the accepted R5 `CC`/`CXX`/`CPATH` values.
- Objective equations and normalization: numerical gate compares kernel output
  against BF16-input/weight Torch matmul using FP32 absolute error reporting;
  finite output and max absolute error <=0.25 are required.
- Rollout/replay forward mode and adapter dropout/RNG contract: N/A; synthetic
  tensors use seed 42 and no dropout.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 tensors; no cache, quantization,
  attention, parallelism, or training mesh.
- Logit/logprob/loss/gradient parity tolerances: N/A; output max absolute error
  <=0.25 and finite values.
- World size, microbatch, accumulation, and global batch: world 1, one token,
  no batch accumulation.
- GPUs: physical/logical GPU 0 only, B200 183359 MiB; recheck idle at launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T20:11:13+09:00` to `2026-07-21T20:11:21+09:00`, 8 seconds;
  foreground bounded diagnostic, exit 2.
- Actual GPU-hours and peak scratch use: less than 0.003 GPU-hours; GPU 0
  returned to zero allocation.
- Command: `.venv312/bin/python` inline synthetic LoRA expand gate with
  `CUDA_VISIBLE_DEVICES=0`, fixed compilers/headers, and PDL disabled only in
  the process-local imported module.
- Outputs: `artifacts/policy/PRL-DIAG-01-vllm-lora-no-pdl-gpu0.log`, 508
  bytes, SHA256 `f2ff75ee...2e79`.
- Scorer/parser identity: Torch matmul oracle in the same process.
- Metrics: PDL-disabled Triton compilation completed and output was finite;
  reported max absolute error 0.375288904 is invalid because the oracle used
  `weights[0,0].T`, incorrectly collapsing the hidden dimension and then
  broadcasting the scalar result across 4096 outputs.
- Conclusion: invalid numerical fixture, not a kernel failure. DIAG-02 changes
  only the oracle to `inputs[0].float() @ weights[0].float().T`.

### PRL-DIAG-02-VLLM-LORA-NO-PDL-GPU0

- Cell/matrix ID and mandatory/diagnostic class: diagnostic single-GPU
  synthetic vLLM LoRA expand-kernel parity gate.
- Spike-plan git revision and approval references: Policy Pilot vertical-slice
  authorization plus sealed R5 and DIAG-01 records.
- Lifecycle status: `COMPLETE`.
- Result: `PASS`.
- Question: with the corrected two-dimensional Torch oracle, does the
  PDL-disabled vLLM LoRA expand kernel compile on B200 and pass BF16 parity?
- Baseline and exact output path: DIAG-01 compiled successfully but had a
  malformed oracle; fresh log
  `artifacts/policy/PRL-DIAG-02-vllm-lora-no-pdl-gpu0.log`.
- Model and processor identity: no model or processor; exact DIAG-01 synthetic
  one-token, rank-64, hidden-4096 BF16 kernel fixture.
- Representation checkpoint identity: N/A; no TGVF Adapter is loaded.
- N/A fields and justification: no dataset, prompt, transcript, reward,
  rollout, replay, optimizer, checkpoint, reference policy, or judge.
- Policy/reference initialization: N/A; synthetic tensors only.
- Rollout policy version and allowed asynchronous staleness: N/A.
- Code commit and worktree state: executable code remains
  `519a0c4aeb8e7ed99cbf9b8e25024c701177d988`; later commits may change only
  experiment-ledger records.
- Repository adapter/patch surface and hash: no repo patch; same installed
  vLLM hashes as DIAG-01. Process-local `lora_expand_op.supports_pdl=false`.
- Dataset/manifest, hashes, sample rule, and n: N/A.
- Native prompt/tool schema hash: N/A.
- Chat-template/token-fixture hash and token-ownership masks: N/A.
- D/DeepStack/position/mask identity: N/A.
- Observation materialization/artifact identity used by all replays: N/A.
- RL framework/version/environment lock: exact DIAG-01 environment.
- Objective equations and normalization: corrected oracle is
  `inputs[0].float() @ weights[0].float().T`; finite output and max absolute
  error <=0.25 are required.
- Rollout/replay forward mode and adapter dropout/RNG contract: N/A; seed 42.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: N/A.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 tensors; all other fields N/A.
- Logit/logprob/loss/gradient parity tolerances: N/A; output max absolute error
  <=0.25 and finite values.
- World size, microbatch, accumulation, and global batch: world 1, one token,
  no accumulation.
- GPUs: physical/logical GPU 0, B200 183359 MiB; recheck idle before launch.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T20:12:39+09:00` to `2026-07-21T20:12:45+09:00`, 6 seconds;
  foreground bounded diagnostic, exit 0.
- Actual GPU-hours and peak scratch use: less than 0.002 GPU-hours; GPU 0
  returned to zero allocation.
- Command: exact DIAG-01 command with only the corrected oracle indexing.
- Outputs: `artifacts/policy/PRL-DIAG-02-vllm-lora-no-pdl-gpu0.log`, 128
  bytes, SHA256 `9c0d56fb...1a6d`.
- Scorer/parser identity: corrected Torch matmul oracle in the same process.
- Metrics: output finite; max absolute BF16-versus-FP32-oracle error
  `0.000686049`, below the accepted `0.25` bound.
- Conclusion: PASS. Disabling optional PDL/GDC removes R5's sm100 compiler
  failure without changing the LoRA expand computation. The production plugin
  may apply the same behavior only to the exact vLLM 0.12.0/Triton 3.5.0 stack.

### PRL-01-R6-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory 4-GPU Policy Pilot
  one-step vertical slice and clean-process resume.
- Spike-plan git revision and approval references: accepted task sections
  0.8.2--0.8.3 and DIAG-02 PASS.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; startup, four-rank model construction, LoRA synchronization
  and CUDA-graph capture passed, then the first rollout stopped before
  generation because the bridge read veRL's `DictConfigWrap` as a bare config.
- Question: does the fixed R5 contract complete one real rollout/replay/update
  and clean-process resume when only optional LoRA PDL/GDC is disabled on the
  exact failing vLLM/Triton stack?
- Baseline and exact output path: R5 reached LoRA CUDA-graph compilation;
  fresh root `artifacts/policy/PRL-01-R6-qwen3-grpo-1step-auto-resume-gpu0123`,
  absent at planning time; `launch.log`/`resume.log`.
- Model and processor identity: Qwen3-VL-8B-Thinking local path, tokenizer
  151669, template `36e042fe...8956`, DeepStack on, max pixels 262144.
- Representation checkpoint identity: RP-49 contextual layer -1, Balanced CE
  T=0.1 step 2000; file/manifest/run hashes `fcda0b96...fc14` /
  `3ff14e66...f49e` / `980e4136...e1bea`.
- N/A fields and justification: bounded exact-MCQ smoke; no judge, Qwen2.5,
  crop, SDPO, full tuning, nonzero KL or quality inference.
- Policy/reference initialization: base plus decoder-only LoRA `64/64/0`, LR
  `1e-5`; all non-decoder scopes frozen; separate unadapted frozen reference.
- Rollout policy version and allowed asynchronous staleness: synchronized
  step-0 behavior snapshot, zero staleness, no update before replay.
- Code commit and worktree state: executable commit
  `a033146338971eb33b59a18a6509a0f940205544`; clean descendant may add only
  this exact config and ledger row.
- Repository adapter/patch surface and hash: R6 config SHA256
  `a625466c...57e7`, run identity `9ccba1e6...b853`; R5 inputs unchanged.
  Repo-owned general plugin binds all four vLLM 0.12 LoRA `supports_pdl`
  aliases false only with Triton 3.5.0; mode
  `vllm-0.12-triton-3.5-lora-pdl-disabled-v1`. DIAG-02 error `0.000686049`.
- Dataset/manifest, hashes, sample rule, and n: unchanged DeepEyes snapshot,
  manifest and sample hashes; seed 42, cursor 7 MCQ C; batch 4, n=8.
- Native prompt/tool schema hash: `390b334e...99f5` / `f33f61d4...6aba5`, four
  calls and fifth-call error `1f649e1d...ad200`.
- Chat-template/token-fixture hash and token-ownership masks: 724 tokens, 234
  image pads, IDs `221b8952...d4d`; sampled assistant tokens only.
- D/DeepStack/position/mask identity: exact recorded main D plus `(8,16,24)`
  branches/layout/positions/masks; no recomputation.
- Observation materialization/artifact identity used by all replays: one
  rollout-owned content-addressed bundle shared by current/reference replay.
- RL framework/version/environment lock: veRL e003163, Python 3.12.3, Torch
  2.9.0+cu128, Transformers 4.57.6, vLLM 0.12.0, Triton 3.5.0, PEFT 0.19.1,
  Ray 2.56.1; accepted `CC`/`CXX`/`CPATH`.
- Objective equations and normalization: sample-std GRPO; reward
  `0.8/0.2/1.2`; behavior ratio, clip `[0.8,1.2]`, dual clip 3, global token
  mean, one epoch, KL/entropy zero, grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: rollout
  `request_seeded_batch_sensitive_v1`, actual behavior logprobs; deterministic
  replay, dropout zero.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12; seed 42 derivation
  `fe8d2da9...e867e`; `1/1/-1/0`, penalties `1/0/0`, none, post-transform,
  EOS 151645, stop closer, cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/no quantization; HF SDPA, vLLM
  TRITON_ATTN/TORCH_SDPA; TP1, 4-rank FSDP2, no reshard/checkpointing/remove
  padding/compile; colocated; CUDA graph remains enabled.
- Logit/logprob/loss/gradient parity tolerances: exact identities, finite
  values, pre-update clipfrac zero and absolute mean PPO KL <=0.015625.
- World size, microbatch, accumulation, and global batch: world 4, prompt batch
  4, prompt/rollout microbatch 1/rank, logprob microbatch 8/GPU, 32
  trajectories, accumulation 1.
- GPUs: physical/logical 0--3, B200 183359 MiB; idle at planning and recheck.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T20:17:42+09:00`--`2026-07-21T20:22:18+09:00`, 4m36s;
  tmux `prl01_r6_gpu0123`; no resume was attempted because step 1 did not run.
- Actual GPU-hours and peak scratch use: about 0.307 GPU-hours; 326 MiB output.
- Command: absolute R6 `run-policy` under `.venv312`; clean resume only after
  successful step 1 and must make no second update.
- Outputs: `launch.log` SHA256
  `ae38ebae3fe5b9ce38814e06f2df3aaad4bb2c772e44baa7e9990ac65d8df9b9`;
  step-0 LoRA snapshot and weight-sync request; W&B run `h91gdvrn`.
- Scorer/parser identity: exact MCQ `2a3d5fa4...2e1c`, strict native parser.
- Metrics: no trajectory, replay, optimizer step, or checkpoint was produced.
- Conclusion: fail closed before sampling; the trainer config contained the
  correct runtime identity, but upstream passed it through its Hydra-protection
  wrapper. The next run may change only the bridge unwrap plus identity/config.

### PRL-01-R7-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: mandatory 4-GPU policy RL
  one-step vertical slice and clean-process resume.
- Spike-plan git revision and approval references: accepted task sections
  0.8.2--0.8.3; DIAG-02 PASS; R6 wrapper root cause reproduced.
- Lifecycle status: `COMPLETE`.
- Result: `FAIL`; the wrapper fix passed and rollout workers were constructed,
  then exact step-0 LoRA installation rejected FP32 local tensors against the
  actor's BF16 snapshot before generation.
- Question: after matching veRL's `DictConfigWrap` constructor contract, can
  the fixed stack complete one rollout/replay/update and clean resume?
- Baseline and exact output path: R6 reached first rollout dispatch; fresh
  `artifacts/policy/PRL-01-R7-qwen3-grpo-1step-auto-resume-gpu0123`.
- Model and processor identity: local Qwen3-VL-8B-Thinking; tokenizer 151669;
  chat template `36e042fe...8956`; native DeepStack; max pixels 262144.
- Representation checkpoint identity: RP-49 contextual layer -1, Balanced CE
  T=0.1 step 2000; artifact/manifest/run hashes `fcda0b96...fc14` /
  `3ff14e66...f49e` / `980e4136...e1bea`.
- N/A fields and justification: bounded exact-MCQ smoke; no judge, Qwen2.5,
  crop, SDPO, full tuning, nonzero KL, or quality claim.
- Policy/reference initialization: base plus decoder-only LoRA `64/64/0`, LR
  `1e-5`; all specified frozen scopes retained; frozen unadapted reference.
- Rollout policy version and allowed asynchronous staleness: synchronized
  step-0 snapshot, zero staleness, no intervening update.
- Code commit and worktree state: `997ca081ba67fd0a7ea0dbf7cdf291e02e25b7f7`;
  clean descendant may add only this config and ledger row.
- Repository adapter/patch surface and hash: R7 config
  `051d858e...3638b`, identity `6eac8260...cfaf`; R6 PDL plugin unchanged;
  bridge now unwraps veRL trainer/data `DictConfigWrap` before factory binding.
- Dataset/manifest, hashes, sample rule, and n: fixed DeepEyes snapshot
  `5546681e...ded3`, manifest `3483c317...f477`, seed 42, cursor 7 MCQ C;
  prompt batch 4 and n=8.
- Native prompt/tool schema hash: `390b334e...99f5` / `f33f61d4...6aba5`;
  `tgvf_focus_tool`, four calls, standard fifth-call error.
- Chat-template/token-fixture hash and token-ownership masks: 724 tokens, 234
  image pads, fixture `221b8952...d4d`; sampled assistant tokens only.
- D/DeepStack/position/mask identity: rollout-recorded main D plus branches
  `(8,16,24)`, layouts, positions and masks; no recomputation.
- Observation materialization/artifact identity used by all replays: one
  rollout-owned content-addressed bundle for current/reference replay.
- RL framework/version/environment lock: veRL e003163; Python 3.12.3; Torch
  2.9.0+cu128; Transformers 4.57.6; vLLM 0.12.0; Triton 3.5.0; PEFT 0.19.1;
  Ray 2.56.1; accepted compiler/Python-header bindings.
- Objective equations and normalization: sample-std GRPO; reward
  `0.8/0.2/1.2`; ratio clip `[0.8,1.2]`, dual clip 3, global token mean,
  one epoch, KL/entropy 0, grad norm 1.
- Rollout/replay forward mode and adapter dropout/RNG contract: request-seeded
  rollout with actual behavior logprobs; deterministic replay; dropout 0.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM 0.12; seed 42/content-addressed;
  `1/1/-1/0`; penalties `1/0/0`; no processors; post-transform; cap 8192.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16/no quantization; HF SDPA and vLLM
  TRITON_ATTN/TORCH_SDPA; TP1; four-rank FSDP2; colocated; CUDA graphs on.
- Logit/logprob/loss/gradient parity tolerances: exact identities, finite
  values, pre-update clipfrac 0 and absolute mean PPO KL <=0.015625.
- World size, microbatch, accumulation, and global batch: 4; prompt/rollout
  microbatch 1/rank; logprob microbatch 8/GPU; 32 trajectories; accumulation 1.
- GPUs: physical/logical 0--3, B200 183359 MiB; 0 MiB and 0% at planning.
- Start/end timestamps, elapsed time, and session/process identity:
  `2026-07-21T20:29:41+09:00`--`2026-07-21T20:34:36+09:00`, about 4m55s;
  tmux `prl01_r7_gpu0123`, exit 1; no resume.
- Actual GPU-hours and peak scratch use: about 0.33 GPU-hours; GPUs released.
- Command: absolute R7 `run-policy` under `.venv312` with `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Outputs: `launch.log` SHA256
  `c148667a9d925f9fb774b6695c915c667501187964ccae4de6b1f69b09c4cb4f`;
  W&B `n9i4b5ig`; complete 504-tensor BF16 step-0 snapshot.
- Scorer/parser identity: exact MCQ `2a3d5fa4...2e1c`; strict native parser.
- Metrics: no generation, tool call, replay, optimizer step, or checkpoint.
- Conclusion: PEFT `autocast_adapter_dtype=True` promoted only the local
  rollout LoRA to FP32. All keys/shapes/values otherwise matched. R8 preserves
  the actor/base BF16 dtype and retains strict snapshot equality.

### PRL-01-R8-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step and
  clean-resume vertical slice. Exact BF16 LoRA installation passed and the
  first real generation reached vLLM, where veRL collapsed the already-expanded
  image-token run before the plugin's strict prompt check.
- Complete identity: config
  `configs/policy/runs/prl_01_r8_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `a3a2a7d3...bc275`, run identity `2b6fafa2...a52d9`; every model,
  data, checkpoint, prompt, replay, GRPO, sampling and topology field is frozen
  there rather than repeated here.
- Code/delta: `0541178830859c411c599200d04f60c2c8d9b589`; R7 plus only
  `autocast_adapter_dtype=False`, matching the BF16 actor snapshot while
  retaining strict 504-tensor equality. Targeted CPU gate: 1 passed in 4.77s.
- Question: can exact BF16 snapshot installation now proceed through one real
  rollout, exact current/reference replay, optimizer step, checkpoint and a
  clean no-extra-update resume?
- GPUs/output/session: physical 0--3, idle B200s at planning;
  `artifacts/policy/PRL-01-R8-qwen3-grpo-1step-auto-resume-gpu0123` (absent);
  tmux `prl01_r8_gpu0123`, then resume only after successful step 1.
- Start/end, outputs, metrics and conclusion: about
  `2026-07-21T20:41:25+09:00`--`20:46:05+09:00`; tmux exit 1, GPUs released;
  log SHA256 `a6c715fb...f9e02`. No generated/tool/replay/update result. R9
  disables only veRL's incompatible Qwen2.5 dedup alias inside the exact
  pre-expanded Qwen3 plugin; the plugin's per-request strict contract remains.

### PRL-01-R9-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean resume.
- Complete identity: R9 config SHA256 `57e419de...211dd`, run identity
  `683f7d65...88775`; the config is authoritative for all model/data/GRPO/
  replay/topology fields. Code `e5e268a611333b2ce148a19fe922ff3e9eb1d578`.
- Delta/question: R8 plus repo-owned preservation of the full pre-expanded
  Qwen3 prompt through pinned veRL; can rollout now proceed through generation,
  tool/replay/update/checkpoint/resume? Actual veRL CPU gate preserved 5/5 IDs;
  targeted tests 2 passed.
- GPUs/output/session: idle B200 0--3;
  `artifacts/policy/PRL-01-R9-qwen3-grpo-1step-auto-resume-gpu0123` (absent);
  tmux `prl01_r9_gpu0123`, exit 1. BF16 LoRA and pre-expanded vLLM generation
  passed; one valid focus call reached the real tool runtime. Contextual
  forward then rejected source visual `(234,4096)` where its single-sequence
  consumer requires `(1,234,4096)`; no replay/update/checkpoint was produced.

### PRL-DIAG-10-QWEN3-LIVE-TWO-CALL-CHAIN-GPU3

- Lifecycle/result: `COMPLETE` / `PASS`; diagnostic single-GPU real-model
  gate, not a training or quality experiment.
- Question: can the real selected sample execute two contextual TGVF calls so
  the second forward consumes the first recorded main D plus all three
  D-DeepStack branches, then pack source + two observations for the third vLLM
  turn without recomputation?
- Code/config identity: `cac41286c554cfdda67822c0cf29b9f21950e1d0`;
  `tools/smoke_policy_live_tool_chain.py` SHA256 `1c1bb633...6cac55`;
  R9 config SHA256 `57e419de...211dd` and run identity
  `683f7d65...88775` remain authoritative.
- Model/processor/precision: local Qwen3-VL-8B-Thinking, native DeepStack,
  BF16, SDPA, tokenizer 151669, chat template `36e042fe...8956`, max pixels
  262144; no vLLM server, Ray, optimizer, checkpoint, or reward execution.
- Data/representation: fixed R9 DeepEyes cursor-7 sample and RP-49 contextual
  layer -1 Balanced-T0.1 step-2000 artifact (`fcda0b96...fc14`); exact source
  image and native pre-expanded prompt from the selected-sample dataset.
- Tool/replay scope: two deterministic scripted native `tgvf_focus_tool`
  calls; maximum-call policy unchanged at four. Verify source/main-D/three
  branches at `[N,H]` in the store, `[1,N,H]` only at injected forward, and
  source + D1 + D2 as `[N,4H]` vLLM payloads with exact prompt-run binding.
- Runtime/GPU/output: `.venv312`, Torch 2.9.0+cu128; physical GPU 3 exposed as
  logical CUDA 0, presently idle; log
  `artifacts/policy/diagnostics/PRL-DIAG-10-qwen3-live-two-call-gpu3.log`.
- Command: `CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false timeout 900s
  .venv312/bin/python tools/smoke_policy_live_tool_chain.py --config
  configs/policy/runs/prl_01_r9_qwen3_grpo_1step_autoresume_gpu0123.toml
  --physical-gpu 3`.
- Result: `2026-07-21T21:20:51+09:00`--`21:20:59+09:00`, 7.36 s,
  exit 0. Two observations were materialized as main D `(234,4096)`; the third
  turn bound source + D1 + D2 as three `(234,16384)` vLLM items over 1330
  prompt tokens. Log SHA256 `ee53039b...74b6ac`; GPU 3 was released.

### PRL-01-R10-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r10_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `466654ca...01776`, run identity `1142dfaa...9bff3`; this config is
  authoritative for all model, data, prompt/tool, representation, sampling,
  reward, GRPO, optimizer, replay, precision, capacity, topology, checkpoint,
  and output fields.
- Code/delta: implementation commit `cac41286c554cfdda67822c0cf29b9f21950e1d0`;
  R9 plus only single-sequence `[1,N,H]` normalization at the live injected
  forward boundary and the explicit veRL variable-padding sidecar contract.
  The observation store remains `[N,H]`; no objective or sampling change.
- Preflight evidence: 95 related tests passed; the real-model two-call gate
  `PRL-DIAG-10` passed. It exercised source + main D + all three D-DeepStack
  branches across two calls and the third-turn vLLM payload.
- Question: does the composed path now complete generation, tool execution,
  exact current/reference replay, one optimizer step, step-1 checkpoint, and
  clean resume without an extra update?
- GPUs/output/session: physical/logical B200 0--3, required world size 4;
  output `artifacts/policy/PRL-01-R10-qwen3-grpo-1step-auto-resume-gpu0123`
  must be absent at launch; tmux `prl01_r10_gpu0123`.
- Exact launch: `CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=tgvf_qwen3_precomputed
  VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_V1=1
  VLLM_WORKER_MULTIPROC_METHOD=spawn CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false TORCH_DEVICE_BACKEND_AUTOLOAD=0
  timeout 3600s .venv312/bin/python -m tgvf_rl.cli run-policy <absolute-R10-config>
  --python <absolute-.venv312-python>` with the accepted compiler/Python-header
  environment used by R9.
- Result: `2026-07-21T21:26:29+09:00`--`21:31:33+09:00`, tmux
  `prl01_r10_gpu0123`, exit 1, about 0.34 GPU-hours; all GPUs released. W&B
  `o9st2sq4`; launch-log SHA256 `bdc84f11...ca42b4b`. First generation
  completed, but the token-byte decoder rejected a legal Qwen ByteLevel
  Unicode split because two sampled tokens shared one character offset. No
  parsed tool call, replay, optimizer step, checkpoint, or resume resulted.
- Conclusion: this was not a behavior/replay mismatch. R11 replaces the invalid
  re-tokenization/one-token-per-character assumption with sampled-ID-authority
  and exact per-token ByteLevel bytes, including non-canonical token
  segmentation and Unicode splits.

### PRL-01-R11-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r11_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `ed53d54b...5228b`, run identity `3c9b7312...a8bda`; the config is
  authoritative for every model/data/protocol/representation/GRPO/replay/
  optimizer/topology/checkpoint field.
- Code/delta: `ce7a2047f262b4ce5ab0c8238e0ce33a3c432ee2`; R10 plus only the exact
  Qwen ByteLevel sampled-token byte-span correction. Sampled IDs, rather than
  canonical re-tokenization, are authoritative; added-token literal bytes and
  ordinary ByteLevel bytes must reconstruct the decoded text exactly.
- Preflight: 130 related tests passed. Real local Qwen3 CPU checks passed for
  split Unicode, non-canonical `a`+`b`, leading whitespace, repeated newline,
  decomposed Unicode and native added tokens; all 151669 vocabulary rows use
  the accepted alphabet with no empty piece.
- Question: does rollout now pass exact text/token ownership and complete tool
  execution, current/reference replay, one update, checkpoint and clean resume?
- GPUs/output/session: idle physical/logical B200 0--3; output
  `artifacts/policy/PRL-01-R11-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r11_gpu0123`.
- Exact launch: same accepted R10 environment and command, substituting the
  absolute R11 config/output paths; timeout 3600 s.
- Result: `2026-07-21T21:48:15+09:00`--`21:53:26+09:00`, tmux
  `prl01_r11_gpu0123`, exit 1, about 0.35 GPU-hours; GPUs released. W&B
  `ej0peov6`; log SHA256 `a3225dd2...a7bc7d2`. The exact sampled-token fix
  passed and a native trajectory completed through generation/tool-loop
  execution. Its output builder then imported `AgentLoopMetrics` from veRL's
  package root although pinned veRL defines but does not export it there.
  Therefore no batched reward/replay/update/checkpoint/resume result exists.
- Conclusion: R12 uses the pinned definition module and directly exercises the
  default live metrics factory; no algorithm, sampling, or trajectory change.

### PRL-01-R12-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r12_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `a36df0c8...77ab89`, run identity `64233c8f...fd045`; the config remains
  authoritative for all model/data/protocol/objective/replay/topology fields.
- Code/delta: `b68e99deb6aeb834ee18d321a6b38f722f9c2dc8`; R11 plus the corrected
  pinned `AgentLoopMetrics` import, PyTorch-equivalent veRL unpadding for the
  accepted SDPA stack, and an already-complete resume guard. No model,
  rollout, reward, GRPO, or optimizer setting changed.
- Preflight: 136 related tests passed, including real pinned-veRL metrics,
  padding/no-padding round trip without `flash_attn`, exact-sidecar/replay
  contracts, and step-1 resume with zero extra optimizer mutation. Static
  current/reference replay audit found no deterministic next blocker.
- Question: can the composed run now finish native trajectories, batched
  reward, exact current/reference replay, one optimizer step, step-1 paired
  checkpoint, then cleanly load/sync it without step 2?
- GPUs/output/session: idle physical/logical B200 0--3; output
  `artifacts/policy/PRL-01-R12-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r12_gpu0123`, followed by a distinct resume session only
  after a successful step-1 checkpoint.
- Exact launch: same accepted R11 environment and command, substituting the
  absolute R12 config/output paths; timeout 3600 s.
- Result: `2026-07-21T22:04:22+09:00`--`22:10:07+09:00`, tmux
  `prl01_r12_gpu0123`, exit 1, about 0.38 GPU-hours; GPUs released. W&B
  `k8ecnc8i`; launch-log SHA256 `5f0a7e9f...624e9`. Native generation and tool
  execution reached the observation appender, which incorrectly re-tokenized
  policy-owned sampled text and rejected a legal non-canonical ByteLevel token
  segmentation. No batched replay, update, step-1 checkpoint, or resume
  occurred. R13 removes that re-tokenization: exact sampled IDs remain
  authoritative and only environment-owned response text is encoded.

### PRL-01-R13-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r13_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `b5d7077a...840cf`, run identity `1ccc6227...2025b`; the config is
  authoritative for all model/data/protocol/objective/replay/topology fields.
- Code/delta: `dc7129f130a2e341561bca85383355bfab6d1607`; R12 plus only
  sampled-ID-preserving native tool append. Policy-owned token IDs are never
  rebuilt from decoded text; environment-owned response text remains
  canonically encoded. No model, sampling, reward, GRPO, or optimizer change.
- Preflight: 503 relevant CPU tests passed (one optional `transfer_queue`
  smoke skipped). Static audit covered parser, target spans, all three tool
  runtimes, rollout bridge, trajectory validation, and exact replay; no other
  policy-text re-encoding path was found. The decoder and appender regression
  tests jointly cover legal non-canonical Qwen ByteLevel segmentation.
- Question: can the composed run now complete native trajectories, batched
  reward, exact current/reference replay, one optimizer step, step-1 paired
  checkpoint, then cleanly load/sync it without step 2?
- GPUs/output/session: physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R13-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent at launch; tmux `prl01_r13_gpu0123`, followed by a distinct resume
  session only after a successful step-1 checkpoint.
- Exact launch: same accepted R12 environment and command, substituting the
  absolute R13 config/output paths; timeout 3600 s.
- Result: `2026-07-21T22:18:26+09:00`--`22:23:55+09:00`, tmux
  `prl01_r13_gpu0123`, exit 1, about 0.37 GPU-hours; GPUs released. W&B
  `5wfe26o4`; launch-log SHA256 `6daf98c5...f9d5f`. R12's exact sampled-token
  fix passed. One ordinary sampled completion then lacked the accepted balanced
  think layout; the sampler incorrectly classified this model-format outcome
  as replay corruption and aborted the complete group before reward/replay.
  No optimizer step, checkpoint, or resume occurred. The next gate must retain
  such rows with exact behavior data and `format_reward=-1` rather than drop or
  abort them.

### PRL-01-R14-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  clean no-extra-update resume.
- Identity: config
  `configs/policy/runs/prl_01_r14_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `220544d7...d89f`, run identity `588dba06...3663`, code
  `1d610542de3bc7694ab4d3e64fe4556acd916bac`; all model/data/protocol,
  objective, replay, and topology fields remain authoritative in that config.
- Preflight: 523 relevant CPU tests passed (one optional `transfer_queue`
  smoke skipped). Invalid/truncated model output, cap recovery, deep JSON,
  exact behavior rows, complete n=8 reward groups, and lossless response-envelope
  compaction are covered through the composed veRL bridge.
- Question: can the run now complete rollout, reward, exact current/reference
  replay, one optimizer step, step-1 checkpoint, and a clean resume with no
  second update?
- GPUs/output/session: physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R14-qwen3-grpo-1step-auto-resume-gpu0123`; tmux
  `prl01_r14_gpu0123`, standard accepted `.venv312` launch, timeout 3600 s.
- Result: `2026-07-21T23:05:37+09:00`--`23:11:22+09:00`, exit 1, about
  0.38 GPU-hours; GPUs released. W&B `6xhgb5kd`; launch-log SHA256
  `741e0d1a...aa3ec`. R13's invalid-output retention passed and generation
  reached exact reward scoring. A missing-answer MCQ correctly produced the
  deterministic `missing_final_answer` route, but the veRL scorer still
  required `multiple_choice_rule` for every MCQ and aborted instead of
  retaining the row with answer 0 and format -1. No replay, update, checkpoint,
  or resume occurred.

### PRL-01-R15-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  immediate step-1 checkpoint and clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r15_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `5dfd8d3f...8901b`, run identity `19957776...55a79`, implementation
  commit `7a53bf506e27c8a227467253c7b5497c78441eed`; that config is authoritative
  for model, data, protocol, representation, sampling, reward, GRPO, replay,
  optimizer, precision, topology, capacity, and checkpoint fields.
- Delta/preflight: R14 plus only the deterministic missing-answer MCQ reward
  route. Such rows remain in the original n=8 group with answer 0, format -1,
  conditional-tool 0, and no judge call. Reward through replay/update static
  audit found no second deterministic blocker; 74 focused reward, DataProto,
  GRPO, exact-replay, actor lifecycle, and checkpoint tests passed.
- Question: can one composed mixed-length batch complete reward, exact
  current/reference replay, GRPO backward, one FSDP2 optimizer step, LoRA
  weight sync, step-1 paired checkpoint, and clean resume without step 2?
- GPUs/output/session: idle physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R15-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r15_gpu0123`, followed by a distinct resume session only
  after a successful checkpoint. Standard accepted `.venv312` four-GPU launch,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, vLLM 0.12.0 TP=1, BF16, SDPA actor/reference,
  Triton rollout attention, timeout 3600 s.
- Result: `2026-07-21T23:28+09:00`--`23:34:25+09:00`, exit 1; W&B
  `0j1f06ne`; launch-log SHA256 `bb1afbb...6d43`. Rollout, retained invalid
  reward rows, batch compaction, and group advantage completed. Reference exact
  replay then called the injected inner Qwen language model without the root
  FSDP2 pre-forward hook, leaving the root-owned final norm as a DTensor while
  hidden states were ordinary tensors. It failed at `aten.mul.Tensor`; no
  actor replay, backward, optimizer step, checkpoint, or resume occurred.

### PRL-01-R16-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL`; mandatory four-GPU one-step plus
  immediate step-1 checkpoint and clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r16_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `418faf13...300a`, run identity `ad6efd7e...790e`, implementation
  commit `5c143e36e70af50dd7ae238cab837612223c0af0`; all other identities remain
  authoritative in that config.
- Delta/preflight: R15 plus one exact-replay change: synchronously unshard only
  the non-recursive FSDP2 root parameter group before the injected inner
  language-model path. Child decoder, embedding, and lm-head FSDP2 hooks remain
  upstream-owned; the root remains unsharded as in pinned veRL/PyTorch FSDP2
  and through actor backward. Fourteen focused engine/Qwen tests passed.
- Question: does root materialization clear reference replay and allow exact
  current replay, GRPO backward, one optimizer step, paired step-1 checkpoint,
  and clean resume without step 2?
- GPUs/output/session: idle physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R16-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r16_gpu0123`, followed by a distinct resume session only
  after a successful checkpoint. Standard accepted `.venv312` launch,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, vLLM 0.12.0 TP=1, BF16, SDPA
  actor/reference, Triton rollout attention, timeout 3600 s.
- Result: `2026-07-21T23:43:30+09:00`--`23:50:27+09:00`, exit 1; W&B
  `l5swqsiw`; launch-log SHA256 `448ef7c7...4ad7`. Root unshard cleared exact
  reference replay and the run entered actor update. Actor FSDP had been given
  all eight expanded trajectories per rank in one autograd microbatch, retaining
  eight full graphs at about 140 GiB plus a roughly 35 GiB AgentLoop model and
  vLLM allocation; it OOMed before backward. No optimizer step, checkpoint, or
  resume occurred.

### PRL-01-R17-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `PLANNED` / `PENDING`; mandatory four-GPU one-step plus
  immediate step-1 checkpoint and clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r17_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `325e052e...5248`, run identity `1d014144...2d8d`, implementation
  commit `57c0772f78cbc89a63d816cc9b6e69080caad55b`; all other identities remain
  authoritative in that config.
- Delta/preflight: R16 partitions actor autograd into one expanded trajectory
  per rank per microbatch. The eight per-rank trajectories still form the same
  global n=8 GRPO batch and one optimizer step; reference replay remains the
  no-grad expanded batch of eight. Thirty-three focused config/composition and
  exact-replay tests passed.
- Question: can bounded actor graph retention complete current replay, GRPO
  backward, one optimizer step, paired step-1 checkpoint, and clean resume
  without step 2?
- GPUs/output/session: idle physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R17-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r17_gpu0123`, followed by a distinct resume session only
  after a successful checkpoint. Standard accepted `.venv312` launch,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, vLLM 0.12.0 TP=1, BF16, SDPA
  actor/reference, Triton rollout attention, timeout 3600 s.

### PRL-01-R18-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `COMPLETE` / `INVALID`; mandatory four-GPU one-step plus
  immediate step-1 checkpoint and clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r18_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `4d06d63b...b495`, run identity `4bb41c06...4779`, implementation
  commit `57c0772f78cbc89a63d816cc9b6e69080caad55b`.
- Delta/preflight: R17 with only vLLM `max_model_len` and
  `max_num_batched_tokens` reduced from 32,768 to 12,288, exactly covering the
  configured 4,096 prompt plus 8,192 response capacity. No objective, replay,
  model, or sampling changes.
- Question: does the reduced rollout KV reservation, combined with bounded
  actor autograd microbatches, leave enough headroom for the optimizer step?
- GPUs/output/session: idle physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R18-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r18_gpu0123`, followed by a distinct resume session only
  after a successful checkpoint. Standard accepted `.venv312` launch,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, vLLM 0.12.0 TP=1, BF16, SDPA
  actor/reference, Triton rollout attention, timeout 3600 s.
- Result: launcher rejected the config before Ray/model startup because
  `vllm_max_model_len=12,288` leaves no environment-owned tool-token reserve
  beyond the configured 8,192 policy response. No GPU work occurred.

### PRL-01-R19-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- Lifecycle/result: `PLANNED` / `PENDING`; mandatory four-GPU one-step plus
  immediate step-1 checkpoint and clean no-extra-update resume.
- Complete identity: config
  `configs/policy/runs/prl_01_r19_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `2bc80d41...d80f`, run identity `46a95cac...1473`, implementation
  commit `1e043b534537cb2c047d18615ffdf171ac2f9b80`.
- Delta/preflight: R17 with only vLLM `max_model_len` and
  `max_num_batched_tokens` reduced from 32,768 to 16,384. This preserves the
  required 4,096 prompt + 8,192 response and leaves a 4,096 environment-token
  reserve. No objective, replay, model, or sampling changes.
- Question: does the reduced rollout KV reservation, combined with bounded
  actor autograd microbatches, leave enough headroom for the optimizer step?
- GPUs/output/session: idle physical/logical B200 0--3, world size 4; output
  `artifacts/policy/PRL-01-R19-qwen3-grpo-1step-auto-resume-gpu0123` must be
  absent; tmux `prl01_r19_gpu0123`, followed by a distinct resume session only
  after a successful checkpoint. Standard accepted `.venv312` launch,
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, vLLM 0.12.0 TP=1, BF16, SDPA
  actor/reference, Triton rollout attention, timeout 3600 s.

## Compatibility-spike status

### PRL-01-R20-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `COMPLETE` / `FAIL`; config
  `configs/policy/runs/prl_01_r20_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `4166e843...52b5`, identity `ac43521d...2930`, code
  `6dacd1613cce4ecf55352f1ea3ce866b509f5842`, GPU 0--3, tmux
  `prl01_r20_gpu0123`. R19 plus actor-local PPO rollout-correction values
  matching the already frozen algorithm values; all other fields unchanged.

### PRL-01-R21-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `PLANNED` / `PENDING`; config
  `configs/policy/runs/prl_01_r21_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `44d95e39...b287`, identity `be8b28fe...b59c`, code
  `84edac101712d77b944b7b0e939153c66e7c13ed`, GPU 0--3, tmux
  `prl01_r21_gpu0123`. R20 plus root FSDP2 reshard after the complete
  forward/backward batch and before vLLM weight synchronization.

### PRL-01-R22-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `PLANNED` / `PENDING`; config
  `configs/policy/runs/prl_01_r22_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `4db8ac1b...45cc`, identity `4c88d47a...e5e5`, code
  `df718146052ef78dcce50c3eb06f27519b5b8d35`, GPU 0--3, tmux
  `prl01_r22_gpu0123`. R21 plus resident colocated vLLM weights
  (`free_cache_engine=false`, `enable_sleep_mode=false`) to remove wake-up
  duplication during LoRA synchronization.

### PRL-01-R23-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `PLANNED` / `PENDING`; config
  `configs/policy/runs/prl_01_r23_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `1e4244b5...d693`, identity `afe3f445...9e67a`, code
  `e5624990ad13fb9c8c6d95dbc57e3e8b94029e22`, GPU 0--3, tmux
  `prl01_r23_gpu0123`. R22 with only vLLM reserved GPU memory reduced from
  `0.50` to `0.25`; model length, sampling, exact replay, objective, batches,
  resident-weight synchronization, and checkpoint/resume contract unchanged.
  This tests whether releasing about 45 GiB of unused rollout KV reservation
  permits the already-reached actor update and subsequent LoRA sync.

### PRL-01-R24-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `COMPLETE` / `FAIL`; config
  `configs/policy/runs/prl_01_r24_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `e222526b...7243`, identity `6a3fd1f6...0642`, implementation
  `e7dea44a0e97c448c915d90157f9df856d2f839f`, GPU 0--3, tmux
  `prl01_r24_gpu0123`.
- R23 mathematics, data, sampling, replay, update, and capacity are unchanged.
  Runtime topology is corrected to the mature colocated lifecycle: one FSDP2
  actor role plus one vLLM rollout role, vLLM sleep/wake between rollout and
  update, a model-free AgentLoop, and the frozen TGVF Adapter mounted in the
  existing vLLM worker. Source vision runs once there; sampled Hq and D are
  materialized on the sticky rollout worker; exact main D/DeepStack tensors are
  transported into replay. No third Qwen model is loaded.
- Question: can this two-model lifecycle complete a native multi-turn rollout,
  exact current/reference replay, one GRPO optimizer step, LoRA synchronization,
  step-1 paired checkpoint, and clean resume without OOM or observation drift?
- Result: all four colocated actor/vLLM replicas initialized within roughly
  48 GiB per GPU, then the first AgentLoop row failed before GPU generation
  because the binding constructor rejected the runtime's async-only trajectory
  component builder. No optimizer step or checkpoint was produced.

### PRL-01-R25-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `PLANNED` / `PENDING`; config
  `configs/policy/runs/prl_01_r25_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `a485eb8c...cc9f`, identity `ca4fec2f...f752`, implementation
  `4eab66b3dc07a264a4e4f774c4cadca6dafc772c`, GPU 0--3, tmux
  `prl01_r25_gpu0123`.
- R24 mathematics, data, sampling, replay, capacity, two-model topology, and
  lifecycle are unchanged. The only implementation delta accepts an async-only
  trajectory component builder at the existing binding boundary; its focused
  CPU regression suite passed 19 tests.
- Question: after connecting the already-implemented async source-materialization
  path, can the two-model lifecycle complete rollout, exact replay, one GRPO
  update, LoRA synchronization, step-1 checkpoint, and clean resume?
- Result: `2026-07-22T02:20:54+09:00`--`02:24:57+09:00`, W&B
  `pq20pw0v`, launch-log SHA256 `35389c93...03c5b`. R25 passed the R24
  async-builder boundary, initialized all four actor/vLLM replicas, entered
  training, and invoked source-image materialization for the expanded rollout
  rows. vLLM's untyped utility transport decoded each tensor argument as a
  dtype/shape/`memoryview` structure; its following multiprocess queue could
  not pickle that structure. No generation, optimizer step, or checkpoint
  occurred; all four GPUs were released.

### PRL-01-R26-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `COMPLETE` / `FAIL`; config
  `configs/policy/runs/prl_01_r26_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `5668336f...d3fd1`, identity `d86a92dd...99c8d`, implementation
  `0926f146c7b10525d99684dcbb6f9bc81952c482`, GPU 0--3, tmux
  `prl01_r26_gpu0123`.
- R25 mathematics, data, sampling, replay, capacity, topology, and lifecycle
  are unchanged. The sole runtime delta serializes source processor tensors as
  explicit bytes/shape/dtype across vLLM's untyped utility boundary and
  reconstructs them inside the colocated worker. The same vLLM msgpack decode
  followed by Python pickle now passes, with exact tensor recovery; the focused
  suite passed 11 tests.
- Question: does source vision now materialize inside all four sticky rollout
  workers and allow the first real vLLM generation, then exact replay, one GRPO
  update, LoRA synchronization, step-1 checkpoint, and clean resume?
- Result: `2026-07-22T02:32:25+09:00`--`02:36:31+09:00`, W&B
  `pmeoo3xr`, launch-log SHA256 `f15e166b...ee6`. The input wire passed and
  source vision executed in every colocated vLLM worker. The return crossed
  vLLM's secure utility transport as an ordinary mapping, so the HTTP boundary
  did not recover `SourceVisualTensorBundle`; execution stopped before
  generation, optimizer update, or checkpoint, and GPUs 0--3 were released.

### PRL-01-R27-QWEN3-GRPO-1STEP-AUTORESUME-GPU0123

- `COMPLETE` / `PASS`; config
  `configs/policy/runs/prl_01_r27_qwen3_grpo_1step_autoresume_gpu0123.toml`,
  SHA256 `23ced063...51eb`, identity `85c35f8f...9fcc`, implementation
  `ff4ec3217bd02704320c2e36fd2bfe1c51b81686`, GPU 0--3.
- R26 mathematics, data, sampling, replay, capacity, topology, and lifecycle
  are unchanged. Source and focus results now cross vLLM's secure utility
  boundary through explicit primitive tensor wires, including BF16 main D and
  all D-DeepStack branches. The consolidated data-to-checkpoint contract suite
  passed 406 tests with insecure serialization disabled.
- Primary run: tmux `prl01_r27_gpu0123`, 2026-07-22 02:54:22--03:01:29 JST.
  The real 32-trajectory batch completed generation, 48 successful TGVF
  observations from 49 attempts, exact current/reference replay, one FSDP2
  GRPO update, step-1 LoRA publication, and a complete four-shard paired
  checkpoint. Step time was 159.403 s: generation 102.955 s, reference 11.677
  s, actor update 19.694 s, and synchronized publication/checkpoint 25.050 s.
  Peak actor allocation/reservation was 108.502/127.580 GiB per reported rank.
- The batch recorded four prompts, 32 trajectories, 187,250 policy tokens,
  172,776 reasoning tokens, 48 successful observations, one
  `tool_parse.missing_think_closer`, five format errors, and no correct answer
  on this bounded fixture. W&B run `n8r1174b` received the step metrics.
- Clean resume: tmux `prl01_r27_resume_gpu0123`, 2026-07-22
  03:02:47--03:07:11 JST. All four ranks loaded model, optimizer, RNG and LR
  scheduler at global step 1; the paired project state restored and republished
  the exact step-1 LoRA without another optimizer update.
- Primary/resume log SHA256:
  `a90481ce...7720` / `bc823c89...b90a`. Pair/project-state SHA256:
  `b436de09...04a5` / `cf66266e...e1b2`.
- Conclusion: the accepted two-model vertical slice and clean-process resume
  pass end to end. Ray teardown emitted post-success DataLoader/vLLM process
  cleanup noise; this did not corrupt the paired checkpoint. A subsequent
  project cleanup patch adds explicit StatefulDataLoader and EngineCore/HTTP
  shutdown and is covered by the CPU policy/framework regression suite.

### PRL-01-R28-QWEN3-GRPO-2STEP-DIAGNOSTIC-GPU0123

- `COMPLETE` / `FAIL (teardown gate)`; mandatory two-step continuation of the accepted
  vertical slice on physical GPU 0--3. Config
  `configs/policy/runs/prl_01_r28_qwen3_grpo_2step_diagnostic_gpu0123.toml`,
  file SHA256 `8f077d26...7e71`, run identity `fa008dc3...7c9a`, code
  `9942b05a08c078e86729230f561357f7453c7eb1`, clean tracked worktree.
- R27 model, Adapter, selected DeepEyes sample, prompt/tool bytes, reward,
  `n=8`, GRPO mathematics, global prompt batch 4, four-rank FSDP2/TP1
  colocated topology, capacity and exact-replay identities are unchanged.
  This cell has two optimizer steps, cosine total steps 2, epoch count 2, and
  checkpoints at 0/1/2 so the second rollout is generated by the synchronized
  step-1 policy.
- New diagnostics do not change the objective: behavior/current log-ratio
  absolute mean/P99/max and outside-clip fraction; exact Pilot step/cumulative
  W&B plus JSONL metrics; separate weight-sync, checkpoint and end-to-end step
  times; graceful DataLoader/vLLM shutdown.
- Output:
  `artifacts/policy/PRL-01-R28-qwen3-grpo-2step-diagnostic-gpu0123`; tmux
  `prl01_r28_gpu0123`. Command: standard absolute `run-policy` invocation in
  `.venv312` with `CUDA_VISIBLE_DEVICES=0,1,2,3` and insecure vLLM
  serialization unset.
- Question: does the already-passing update/checkpoint path also produce a
  real post-update rollout, publish the required diagnostics, finish with
  clean backend teardown, and expose steady-state step/GPU utilization without
  changing Pilot mathematics?
- Result: both real optimizer steps and the post-step-1 rollout passed. Step
  times were 158.18/141.84s; generation 102.65/89.34s, reference replay
  11.10/8.39s, actor update 20.33/18.42s, weight sync 8.78/9.04s, and
  checkpoint 15.31/16.62s. The run produced 64 trajectories, 394,689 policy
  tokens, 98 successful TGVF observations, and complete step-1/step-2 paired
  checkpoints. W&B run `c222x9a8` and the two-line `metrics.jsonl` contain the
  new metrics.
- Behavior/current diagnostics remained materially nonzero at both steps:
  absolute mean log-ratio 0.04859/0.04883, P99 0.4643/0.4580, and outside-clip
  fraction 0.06438/0.06418. This is a required parity investigation before a
  formal Pilot, not evidence of post-update staleness. Published LoRA snapshots
  contained exactly 504 decoder tensors and no visual/non-decoder tensors.
- During the two-step training window, per-second GPU samples showed about
  45.5--47.3% whole-window mean utilization including role transitions, with
  47.9--49.7% of samples at or above 80%; peak device memory was
  144.6--148.4GiB. Compute segments reached 86--100% utilization.
- A clean-process invocation loaded model, optimizer, RNG and scheduler for all
  four ranks from step 2, restored the epoch-boundary cursor, published the
  exact step-2 LoRA, performed no extra update, and exited 0. Both the training
  and resume invocations nevertheless reproduced vLLM C++ `pure virtual method
  called` teardown noise; the training invocation also left W&B to a failing
  atexit finish. Therefore the requested clean-teardown sub-gate failed and is
  repaired/tested under a subsequent code identity.

### PRL-01-R29-QWEN3-GRPO-1STEP-CLEAN-TEARDOWN-GPU0123

- `COMPLETE` / `FAIL`; bounded four-B200 revalidation of the R28 teardown
  repair. Config
  `configs/policy/runs/prl_01_r29_qwen3_grpo_1step_clean_teardown_gpu0123.toml`,
  file SHA256 `ea7f4625...eee4`, run identity `2f021f4d...fce7`, code
  `b530228e87416f24f29e4ab985c3a9eb138c81e4`, clean tracked implementation.
- Physical GPUs 0--3; four-rank FSDP2 actor/reference, four colocated TP1 vLLM
  workers, global prompt batch 4, `n=8`, per-rank prompt micro-batch 1,
  per-engine rollout prompt micro-batch 1, and gradient accumulation 1.
- R28 model, Adapter/provider, DeepEyes sample, prompt/tool/reward identities,
  sampling, exact-observation replay, LoRA scope, GRPO mathematics and capacity
  are unchanged. This cell performs one update and a step-1 paired checkpoint.
- Output:
  `artifacts/policy/PRL-01-R29-qwen3-grpo-1step-clean-teardown-gpu0123`;
  tmux `prl01_r29_gpu0123`. Acceptance requires exit 0, complete checkpoint and
  metrics, explicit W&B finish, and no `pure virtual method called`, EngineCore
  death, DataLoader-killed, traceback, or atexit error text.
- The real update and checkpoint completed: 32 trajectories, 196,302 generated
  policy tokens, 54 successful TGVF observations and 162.25s end-to-end. W&B
  run `xhp0jfj9` explicitly synchronized and printed its final run summary.
  Teardown then reproduced four vLLM `pure virtual method called` aborts and a
  W&B service atexit `BrokenPipeError`. GPU memory was fully released and the
  checkpoint/metrics are intact, but the clean-teardown acceptance gate failed.

### PRL-01-R30-QWEN3-GRPO-BS16-THROUGHPUT-GPU0123

- `COMPLETE` / `PASS`; one-update four-B200 throughput cell for the Pilot
  `global prompt batch=16`, selected to match the DeepEyes reference's four
  prompts per GPU while retaining Pilot `n=8`.
- Config
  `configs/policy/runs/prl_01_r30_qwen3_grpo_bs16_throughput_gpu0123.toml`,
  file SHA256 `938e6ba3265c36eed9a7b2d59e2dead3c56373132d5fedc907609657024ce2be`,
  run identity `6ad4607b14b6125f5aeba18ad975118ce3580dbb9bbd6070d98ca98597e85ee9`,
  code `6f4537c00bd6ca21fa910e1f7b632bf48f6b7340`, clean tracked implementation.
- Physical GPUs 0--3; four-rank FSDP2 actor/reference and four colocated TP1
  vLLM workers. Global prompt batch 16, `n=8` (128 trajectories), per-rank actor
  prompt micro-batch 1, per-engine rollout prompt micro-batch 1, and gradient
  accumulation 4. Actor/replay token caps remain 98,304 per micro-batch.
- R29 model, TGVF Adapter/provider, fixed DeepEyes sample, native prompt/tool,
  reward, sampling, exact-observation replay, LoRA scope and GRPO mathematics
  are unchanged. vLLM capacity is 0.45 memory utilization, 16,384 batched
  tokens and 32 sequences; this is the only runtime-capacity change.
- Output
  `artifacts/policy/PRL-01-R30-qwen3-grpo-bs16-throughput-gpu0123`; tmux
  `prl01_r30_gpu0123`. Acceptance requires one complete update/checkpoint with
  peak memory, generation/reference/actor/sync timing and trajectories/second.
  Exit-only teardown noise is recorded separately and does not invalidate the
  throughput measurement.
- The update completed in 301.33s with 128 trajectories and 799,469 generated
  policy tokens. Generation/reference/actor/sync/checkpoint times were
  178.14/34.62/64.53/8.73/15.23s. Aggregate generation throughput was
  4,487.95 policy tokens/s and end-to-end trajectory throughput was 0.4248/s,
  respectively 2.35x and 2.15x the R29 batch-4 measurement while processing
  about four times the tokens/trajectories. Actor peak allocated/reserved
  memory was 110.36/129.56GiB; sampled physical peak was 146,922MiB
  (about 143.5GiB).
- Across the measured step window, per-GPU average utilization was
  49.0--54.2%; 48.4--55.2% of samples were at or above 80%. Decode segments
  held all four GPUs around 92--95%. W&B run `en4rnqjj`, checkpoint and metrics
  completed; the already-known exit-only vLLM/W&B teardown noise recurred.

### PRL-01-R31-QWEN3-GRPO-BS32-THROUGHPUT-GPU0123

- `COMPLETE` / `PASS`; one-update four-B200 throughput cell for global prompt
  batch 32 and Pilot `n=8` (256 trajectories), matching the DeepEyes reference
  density of 64 trajectories per GPU without copying its 64-GPU global batch.
- Config
  `configs/policy/runs/prl_01_r31_qwen3_grpo_bs32_throughput_gpu0123.toml`,
  file SHA256 `59c4aab922a921b8b49c4b6673850fb8876fbb06e6bfb5bcfdb68d7c5f74e132`,
  run identity `7c39eeda78573e8c8785be9e60e022525fda388a7f0caa463828fe514bd6ddb0`,
  code `d3e2e688611d349247b5f640c87b91e475704d8e`, clean tracked implementation.
- Physical GPUs 0--3; four-rank FSDP2 actor/reference and four colocated TP1
  vLLM workers. Per-rank actor prompt micro-batch remains 1, gradient
  accumulation is 8, and actor/replay token caps remain 98,304. Thus actor
  activation memory is not enlarged by the global batch change.
- R30 model, TGVF Adapter/provider, selected DeepEyes sample, native protocol,
  reward, sampling, exact replay, LoRA scope and GRPO mathematics are
  unchanged. vLLM capacity is 0.65 memory utilization, 16,384 batched tokens
  and 64 sequences to accommodate the doubled rollout concurrency.
- Output
  `artifacts/policy/PRL-01-R31-qwen3-grpo-bs32-throughput-gpu0123`; tmux
  `prl01_r31_gpu0123`. Acceptance requires a complete update/checkpoint and the
  same timing, memory and utilization evidence as R30. Exit-only teardown noise
  remains a separately tracked failure.
- The update completed in 576.14s with 256 trajectories and 1,512,354 generated
  policy tokens. Generation/reference/actor/sync/checkpoint times were
  356.57/67.23/126.53/9.23/16.42s. Aggregate generation and end-to-end token
  throughput were 4,241.35 and 2,624.98 tokens/s, 0.945x and 0.989x the R30
  batch-16 rates. Trajectory throughput was 0.4443/s, 1.046x R30, but average
  generated length was lower; veRL total-token throughput was effectively flat
  at 786.32 versus 787.19 tokens/s. Batch 32 therefore increases samples per
  optimizer update but does not improve normalized compute throughput.
- Actor peak allocated/reserved memory was 112.12/131.24GiB. Sampled physical
  peak was 160,168MiB (about 156.4GiB), leaving materially less headroom than
  batch 16. Across the measured step, per-GPU mean utilization was 49.9--58.3%
  and 49.7--58.1% of samples were at or above 80%; main decode stayed near
  90%, followed by request-length/tool-turn tail imbalance. W&B run `bq4enenq`,
  checkpoint and metrics completed; known exit-only teardown noise recurred.

CPU public-API, transport, objective and oracle tests passed before these rows
were entered. The completed cells are bounded evidence; they do not silently
close broader Qwen replay, Qwen2.5, production-objective or training gates.

### PRL-01-R32-QWEN3-GRPO-BS16-CROP-ONLY-THROUGHPUT-GPU0123

- `COMPLETE` / `PASS`; one-update Crop-only comparison arm on physical GPUs 0--3. This
  is a separately identified visual-tool experiment, not a change to the
  TGVF-only formal Pilot v1.
- Config
  `configs/policy/runs/prl_01_r32_qwen3_grpo_bs16_crop_only_throughput_gpu0123.toml`,
  file SHA256 `fcbfeda7b3701040ae43bc35935eb7a14207c29a60cf8a1f4e1019f0b57225dd`,
  run identity `472cc02ed0d8eeab38e79efcca3363fd39ccdef3a33828c1e7d13d4a256fbf28`,
  code `d8592da50c3a0c444ce56f60e51b09d863beff25`, clean tracked implementation.
- Same model, selected DeepEyes sample, sampling (`n=8`), GRPO equation,
  LoRA scope, BS16/GA4 topology and capacity as R30. The only method arm is
  native `image_zoom_in_tool`; crop encoding reuses the colocated rollout
  replica's frozen vision tower and records crop RGB/main/DeepStack features
  for exact current/reference replay. No TGVF Adapter is loaded for this arm.
- Output
  `artifacts/policy/PRL-01-R32-qwen3-grpo-bs16-crop-only-throughput-gpu0123`;
  tmux `prl01_r32_gpu0123`. Acceptance is one complete update/checkpoint plus
  tool-call/result, replay, timing and memory evidence. Comparison baseline is
  the completed R30 BS16 TGVF-only cell.
- Result: 128 trajectories, 846,558 policy tokens, 193 tool attempts and 177
  successful Crop observations; repeated Crop, exact current/reference replay,
  actor update and the paired step-1 checkpoint all completed. The legacy
  metrics field calls these `successful_tgvf_observations`, but under this
  immutable Crop-only profile every such count is an `image_zoom_in_tool`
  observation.
- Generation/reference/actor/sync/checkpoint times were
  174.79/31.60/60.47/8.14/12.97s; end-to-end was 288.05s. Generation and
  end-to-end policy-token throughput were 4,843.36 and 2,938.95 tokens/s,
  1.079x and 1.108x R30. Step-window GPU utilization averaged 49.1--55.8%,
  with 49.2--54.6% of samples at or above 80%. Actor allocated/reserved peaks
  were 108.96/128.09GiB. W&B run `n1lc3ud5`; known exit-only vLLM/W&B teardown
  noise recurred after the valid checkpoint and status 0.

### PRL-01-R33-QWEN3-GRPO-BS16-TGVF-80STEP-GPU0123

- `FAILED` after optimizer step 27; continuous 80-step TGVF-only run on physical GPUs 0--3, BS16,
  `n=8`, GA4, checkpoints 0/10/20/45/80. Config SHA256
  `d6d502c4612a4ac5693700bd85a316bf9c31676dd22b2ce6f7ace1d736e99c7c`,
  run identity `f33ddf971d4f9cb833e5f921d968044a6e5ebc6b77a3713e67ef133661019e6c`,
  code `8f33abe2a36290aa99d68c8276f24b903b754e66`.
- Config `configs/policy/runs/prl_01_r33_qwen3_grpo_bs16_tgvf_80step_gpu0123.toml`;
  output `artifacts/policy/PRL-01-R33-qwen3-grpo-bs16-tgvf-80step-gpu0123`;
  tmux `prl01_r33_gpu0123`. R32 proved the selected BS16 capacity; acceptance
  requires the scheduled checkpoints, metrics and clean resumability.
- Result: 27 complete metric rows and valid step-10/20 checkpoints; step 28
  rollout failed closed because a legal truncated UTF-8 ByteLevel sequence was
  decoded to U+FFFD but the old span decoder compared raw and decoded byte
  lengths. W&B `zfqzrj0h`; R34 was not started.

### PRL-01-R34-QWEN3-GRPO-BS16-CROP-ONLY-80STEP-GPU0123

- `CANCELLED_NOT_RUN`; superseded because its code baseline predates the R33
  UTF-8 span correction. Bounded non-formal Crop-only comparison on physical GPUs
  0--3, BS16, `n=8`, GA4, checkpoints 0/10/20/45/80. It retains R33's frozen
  base/reference, decoder-LoRA GRPO mathematics, exact behavior log probabilities,
  zero staleness, 512-square pixel cap, seed 42, FSDP2/TP1 colocated topology,
  DeepEyes selected MCQ identity, and W&B metric surface; only the immutable tool
  profile/prompt/schema and Crop observation replay path differ.
- Config `configs/policy/runs/prl_01_r34_qwen3_grpo_bs16_crop_only_80step_gpu0123.toml`,
  SHA256 `115e5c2b3dbadb48800ab29b0d960e946162d21d6ab12e4d11caeb241bbd2ccc`;
  run identity `07174688aebc8761c2c7ffd76ae359b7c42dcd112b17dcaf01c5bd2b09f648df`;
  code baseline `50424dd9b114b15ecaa54949966891491e962b0e`; output
  `artifacts/policy/PRL-01-R34-qwen3-grpo-bs16-crop-only-80step-gpu0123`.
  It launches sequentially on tmux `prl01_r34_after_r33` only after R33 exits 0;
  a nonzero R33 status blocks launch rather than silently taking over the GPUs.

### PRL-01-R35-QWEN3-GRPO-BS16-TGVF-80STEP-UTF8FIX-GPU0123

- `CANCELLED` before any optimizer step; clean 80-step rerun of R33 on GPUs
  0--3 after the tokenizer
  replacement-character span correction. Every model, data, prompt, tool,
  sampling, GRPO, BS16/n8/GA4, FSDP2/TP1, reward and checkpoint field is held
  fixed; only code identity and output/run identity change. W&B receives the
  compact operator metric set; bounded raw trajectory audit samples are local
  artifacts and are not uploaded to W&B.
- Config `configs/policy/runs/prl_01_r35_qwen3_grpo_bs16_tgvf_80step_utf8fix_gpu0123.toml`,
  SHA256 `c1d6213cd321707f5fe31cb59e91777decb52978e3d4a4338223cc0e1f402499`;
  run identity `05d976dabe3403c5a88d6245a36eb5d9fde1730a427623f63f670fb5994adfd4`;
  code `36b2f0fc0ee67df47f78e2f336bdd045596bba00`; output
  `artifacts/policy/PRL-01-R35-qwen3-grpo-bs16-tgvf-80step-utf8fix-gpu0123`.
- Result: the local audit retained 19 completed rollout samples before manual
  cancellation. Fourteen semantically selected B and five selected A; none
  selected the bound ground-truth C. Direct image audit supports B because the
  -300 to -250 interval contains two red-box points while each other candidate
  interval contains at most one. The selected repeated smoke fixture is
  therefore unsuitable for reward training. The audit also exposed that the
  MCQ parser treats the first prose letter in `The...`/`Based...` as an answer
  option. W&B `1pqbs2yr`; no metric row or optimizer update was published.

### PRL-JG-02-OPENROUTER-QWEN25-72B-REAL-THREEROUTE

- Lifecycle/result: `COMPLETE` / `PASS`.
- Question: does the OpenRouter Qwen2.5-72B DeepInfra route provide the real
  DeepEyes MCQ/math/open reward contract before the mixed optimizer-step gate?
- Model/service: `qwen/qwen-2.5-72b-instruct`, DeepInfra FP8 hosted revision
  opaque, no provider fallback, no-data-collection/ZDR, temperature 0, top-p 1,
  seed 42, and bounded 429/503 retry. MCQ remains rule-only.
- Data/prompt: exact three DeepEyes-47K calibration IDs already pinned by the
  judge config; prompt SHA-256 `2fa039d7...86d2`.
- Training/GPU/TGVF/GRPO fields: N/A; this was a paid API reward-route check
  with no GPU allocation, rollout, observation, optimizer, or checkpoint.
- Output: `artifacts/policy/PRL-JG-02-openrouter-qwen25-72b-real-threeroute/deepinfra-calibration.json`,
  SHA-256 `cfe391c3...ec6c2`; judge config SHA-256 `34e94de4...06a6`.
- Metrics: four judge calls, 1,106 prompt tokens, 186 completion tokens,
  `$0.00047256`; every correct candidate scored `0.8`, every wrong candidate
  `0.0`, and MCQ made zero judge calls.
- Conclusion: real three-route calibration passed; admit the real mixed-task
  one-step optimizer/checkpoint gate.

### PRL-02-QWEN3-GRPO-BS16-TGVF-MIXED-JUDGE-1STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `PASS`; mandatory full-data mixed-task
  one-step integration gate before the formal 80-step Pilot.
- Complete identity: config
  `configs/policy/runs/prl_02_qwen3_grpo_bs16_tgvf_mixed_judge_1step_gpu0123.toml`,
  file SHA-256 `4393c7cb...d533f`, run identity `a79ea690...e36f4`, code
  `ceef47f5fca783b7f30afa72c5050d5302c377c2`; the config is authoritative
  for every model, data, protocol, objective, replay, topology, capacity,
  optimizer, checkpoint, and output field.
- Model/representation: frozen Qwen3-VL-8B-Thinking base/reference plus
  decoder-only LoRA r64/alpha64/dropout0 current policy; RP-49 contextual
  hidden-state TGVF Adapter SHA-256 `fcda0b96...fc14`, native DeepStack, and
  max pixels 262,144.
- Data/protocol/reward: full materialized DeepEyes-47K snapshot
  `5546681e...ded3`, manifest `3483c317...f477`, seed 42, 16 prompts and eight
  trajectories per prompt; TGVF-only native prompt/tool with four-call cap.
  MCQ is rule-only and unresolved math/open VQA uses the calibrated OpenRouter
  Qwen2.5-72B DeepInfra judge config SHA-256 `34e94de4...06a6`.
- Mathematics/replay: sample-std GRPO advantage with identical-reward groups
  set to zero, clip 0.2/0.2, dual clip 3, token mean, one update epoch, KL and
  entropy coefficients zero, max grad norm 1.0. Actual behavior logprobs and
  every policy-sampled assistant token are retained; current/reference replay
  consumes the exact immutable rollout-recorded main D/DeepStack/layout/masks
  with dropout zero, deterministic replay, and staleness zero.
- Runtime: physical/logical B200 GPUs 0--3, world size 4, FSDP2 no-reshard,
  colocated TP1 vLLM 0.12, BF16, BS16, per-rank prompt micro-batch 1, GA4,
  `n=8`, temperature/top-p 1, max response 8,192. GPUs were idle at planning.
- Output/session: `artifacts/policy/PRL-02-qwen3-grpo-bs16-tgvf-mixed-judge-1step-gpu0123`;
  tmux `prl02_mixed1_gpu0123`; checkpoint steps 0/1, compact W&B metrics and
  bounded local trajectory audits. Acceptance requires real mixed rewards,
  exact replay, one optimizer update, LoRA publication, paired checkpoint and
  a clean no-extra-update resume.
- Command: accepted `.venv312` absolute `run-policy` launch with
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, deterministic environment, and
  `OPENROUTER_API_KEY` inherited only from the tmux environment.
- Primary result: tmux `prl02_mixed1_gpu0123` exited 0 after 128 real
  trajectories, 66 DeepInfra judge calls, exact current/reference replay, one
  FSDP2 GRPO update, step-1 LoRA publication, and a complete paired checkpoint.
  W&B run `y3zwfhu7`; step time 264.72 s including 17.37 s checkpoint and 7.71 s
  weight sync; judge cost `$0.00692056`. Answer reward was 0.4375, conditional
  tool reward 0.421875, tool-call-attempt rate 0.921875, and format-error rate
  0.1015625. Post-success vLLM/W&B atexit noise did not change exit status or
  checkpoint completeness.
- Resume sub-run: `COMPLETE` / `PASS`, tmux `prl02_mixed1_resume_gpu0123` on
  the same physical GPUs 0--3. It reuses the exact committed run/config identity
  and checkpoint, changing only the upstream runtime `trainer.resume_mode`
  launch override from `disable` to `auto`. It exited 0 after all four ranks
  restored model, optimizer, RNG, and LR scheduler at step 1. The metrics file
  remained one line and the committed checkpoint was not rewritten, proving no
  additional rollout, judge call, or optimizer update; resume-log SHA-256
  `6ea539bf...e8ec`.

### PRL-02-QWEN3-GRPO-BS16-TGVF-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL (external judge rate limit)`; the PRL-02
  one-step clean-resume sub-run passed and tmux `prl02_formal80_gpu0123`
  launched on 2026-07-22 JST.
- Config
  `configs/policy/runs/prl_02_qwen3_grpo_bs16_tgvf_formal_pilot_80step_gpu0123.toml`,
  SHA-256 `62dda224...22f8`, identity `bd7c91df...9d21`, code
  `fc8a1878a8ec2d45b14fdb4c9dc0ae179e99fb4d`; this is the formal Pilot schema,
  not a smoke identity.
- The accepted one-step model, full DeepEyes-47K data, RP-49 Adapter, native
  TGVF tool, mixed reward, exact replay, BS16/n8/GA4, four-B200 FSDP2/TP1
  topology, and sampling/GRPO mathematics are unchanged. The calibrated
  DeepInfra judge has a separate formal scope identity.
- Training has 80 optimizer steps, cosine LR from `1e-5` with no warmup,
  checkpoints at 0/10/20/45/80, auto-resume, compact W&B metrics, and local
  bounded trajectory audits. Output is
  `artifacts/policy/PRL-02-qwen3-grpo-bs16-tgvf-formal-pilot-80step-gpu0123`;
  planned tmux session `prl02_formal80_gpu0123`, physical GPUs 0--3.
- Result: steps 1--6 completed and synchronized successfully; 96 prompts, 768
  trajectories, 1,540,578 policy tokens and 480 judge calls were retained.
  During the next rollout DeepInfra returned HTTP 429 for one judge request and
  all five retries exhausted. The fail-closed reward contract aborted before
  step 7 mutated actor state. No checkpoint existed before the planned step 10,
  so the published step-6 LoRA snapshot is diagnostic only and is not an exact
  optimizer/data/RNG resume point. W&B `rre1y7cy`; GPUs 0--3 were released.

### PRL-02-R1-QWEN3-GRPO-BS16-TGVF-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `CANCELLED` / `INVALID`; clean restart was launched at
  2026-07-22 15:14 JST but stopped before its first completed optimizer step.
  Its extra step-1/every-five-step checkpoint cadence was not user accepted;
  no metric or checkpoint is reusable. W&B `9h9vcp6b`; GPUs 0--3 released.
- The model, RP-49 representation artifact, full DeepEyes-47K snapshot,
  TGVF-only native protocol, BS16/n8/GA4 four-B200 topology, exact replay,
  GRPO equations, optimizer, scheduler, sampling, and reward composition are
  unchanged from PRL-02.
- Operational correction: code `5f3ec8f0f0cd95b48182b65f17ef00a9fb20639a`;
  DeepInfra judge v2 SHA-256 `1536a979...c74de`, ten 429/503 attempts with
  5-second exponential backoff capped at 120 seconds, and 2,400-second outer
  agent-loop timeout. Provider fallback remains disabled and reward remains
  fail-closed.
- Checkpoint correction: normal paired FSDP2/project checkpoints occur at step
  1 and every five steps. Any later training exception saves the last fully
  synchronized optimizer boundary after restoring its retained next-data
  cursor; a partial/unknown optimizer commit remains deliberately unsavable.
- Config
  `configs/policy/runs/prl_02_r1_qwen3_grpo_bs16_tgvf_formal_pilot_80step_gpu0123.toml`,
  SHA-256 `01617dab...11cc8`, run identity `3ff04939...eb67ca`.
  Output is isolated under
  `artifacts/policy/PRL-02-R1-qwen3-grpo-bs16-tgvf-formal-pilot-80step-gpu0123`;
  planned tmux `prl02_r1_formal80_gpu0123`, physical GPUs 0--3.

### PRL-02-R2-QWEN3-GRPO-BS16-TGVF-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL (judge response identity)`; this
  superseded cancelled R1 and
  restarts from the original Qwen policy. The first launch command at 15:24
  JST was rejected before Ray/GPU initialization by the exact-descendant
  preflight, so it produced no rollout, metric, checkpoint or optimizer step.
  The corrected launch began at 15:31 JST in tmux
  `prl02_r2_formal80_gpu0123` (pane PID 2387975) with `set -o pipefail`.
- All scientific settings and the DeepInfra v2 retry/2,400-second timeout are
  identical to R1. Normal checkpoints are restored to the accepted
  10/20/45/80 schedule. The exception path alone saves the last fully
  synchronized optimizer/data/RNG boundary before propagating the error.
- Config
  `configs/policy/runs/prl_02_r2_qwen3_grpo_bs16_tgvf_formal_pilot_80step_gpu0123.toml`;
  SHA-256 `aec8912f...e2e6d`, run identity `41619ee9...0fe7e`;
  output is isolated under
  `artifacts/policy/PRL-02-R2-qwen3-grpo-bs16-tgvf-formal-pilot-80step-gpu0123`;
  planned tmux `prl02_r2_formal80_gpu0123`, physical GPUs 0--3.
- Result: steps 1--2 completed. Cumulative answer reward was 0.62109375 over
  256 trajectories; the apparent step-2 jump was 97/128 versus 89/128 in the
  earlier formal run and does not reflect a hyperparameter change. During the
  next rollout one completed OpenRouter response named a model different from
  the pinned response identity and was correctly rejected before scoring.
  The new exception path then committed the full paired step-2 model,
  optimizer, actor RNG, metrics and next-data cursor checkpoint at
  `checkpoints/global_step_2`; tracker value is 2. W&B `flc3g7gq`.

### PRL-02-R3-QWEN3-GRPO-BS16-TGVF-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `FAIL (transient judge/runtime request)`;
  clean restart from the original
  Qwen policy because R2's exact run identity cannot adopt changed judge-client
  code while claiming an exact checkpoint resume. Launched in tmux
  `prl02_r3_formal80_gpu0123` (pane PID 2440870) with `set -o pipefail`.
- Scientific settings and the 10/20/45/80 checkpoint schedule are unchanged.
  Operational code `a5a314772832a00fa7f0ad2be66d73f88ab2cb5a` retries but
  never scores a completed response with a mismatched model identity; retry
  exhaustion remains fail-closed and invokes the proven emergency checkpoint.
- Config
  `configs/policy/runs/prl_02_r3_qwen3_grpo_bs16_tgvf_formal_pilot_80step_gpu0123.toml`,
  SHA-256 `7e8962a1...6996c`, identity `eeb09336...31a11`; output
  `artifacts/policy/PRL-02-R3-qwen3-grpo-bs16-tgvf-formal-pilot-80step-gpu0123`;
  planned tmux `prl02_r3_formal80_gpu0123`, physical GPUs 0--3.
- Result: optimizer steps 1--7 completed. The next rollout/reward batch stopped
  after 22 trajectory audits. GPU memory, W&B and Ray shutdown evidence exclude
  OOM and W&B failure, but the caught exception text was not persisted, so its
  exact transient request subtype is unknown. The recovery path successfully
  saved the full paired model, optimizer, project state, RNG and next-data
  cursor checkpoint at `checkpoints/global_step_7`; W&B `fc3jt6ql`.

### PRL-02-R4-QWEN3-GRPO-BS16-TGVF-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `CANCELLED` / `INVALID`; clean formal restart using accepted
  indefinite transient judge retry. R3 step 7 remains immutable provenance and
  is not relabeled as an exact R4 resume. Launched in tmux
  `prl02_r4_formal80_gpu0123` (pane PID 2552219) on 2026-07-22 JST.
- Scientific model/data/tool/reward/GRPO/BS16-n8-GA4/FSDP2-TP1 settings and
  normal checkpoints at 10/20/45/80 are unchanged. Judge v3 SHA-256
  `81943918...ce39` retries HTTP 408/425/429/5xx, transport failures and response
  model mismatch indefinitely with backoff capped at 120 seconds; permanent
  failures remain fail-closed.
- Config
  `configs/policy/runs/prl_02_r4_qwen3_grpo_bs16_tgvf_formal_pilot_80step_gpu0123.toml`,
  SHA-256 `07971df1...1c97b`, run identity `443d8513...dc3b`, code
  `b8610091c0ae41bec5dd312044be2f23db914445`; output
  `artifacts/policy/PRL-02-R4-qwen3-grpo-bs16-tgvf-formal-pilot-80step-gpu0123`;
  planned tmux `prl02_r4_formal80_gpu0123`, physical GPUs 0--3.
- Cancellation: after optimizer step 12, the representation binding was audited
  against the completed five-checkpoint endpoint decision and found to use
  RP-49 Balanced/T=0.1 rather than the selected contextual full-D-DeepStack
  Balanced/T=1.0 artifact. User authorized immediate stop. SIGINT at
  `2026-07-22T17:31+09:00` cleanly released GPUs 0--3; the complete step-10
  checkpoint plus step-1--12 metrics/trajectory audits remain immutable
  diagnostics. They are not eligible for resume into another representation
  identity or for the formal T=1.0 Pilot conclusion.

### PRL-02-R5-QWEN3-GRPO-BS16-TGVF-T1-FORMAL-PILOT-80STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `PASS`; mandatory clean replacement for the
  invalid RP-49-bound R4 run. It starts from the original Qwen3 policy with a
  fresh decoder-only LoRA and must not load any R3/R4 policy checkpoint.
- Question: does the accepted contextual-hidden-state, full-D-DeepStack,
  Balanced Matrix CE T=1.0 representation artifact support the fixed 80-step
  BS16/n8 GRPO Pilot under the already accepted resilient judge/runtime stack?
- Scientific settings: identical to R4 except the frozen representation
  artifact and identities. Qwen3-VL-8B-Thinking, DeepEyes-47K snapshot, native
  `tgvf_focus_tool` prompt/schema, max-pixels 262144, four calls, GRPO
  mathematics, reward `0.8/0.2/1.2`, seed 42, LoRA r64/alpha64/dropout0,
  BS16/n8/GA4, FSDP2 world-size 4, colocated vLLM TP1, and checkpoints
  10/20/45/80 remain fixed.
- Representation: `artifacts/representation/REP-QWEN3-V4-CONTEXTUAL-V4/adapter.pt`;
  contextual layer -1, Balanced mean-NLL Matrix CE T=1.0, step 2000, complete
  main D plus D-DeepStack layers 8/16/24. File/manifest/run SHA-256 are
  `50179c70...e75` / `dfa992fc...0e10` / `6c748851...d5c0`; expected run ID is
  `REP-QWEN3-V4-CONTEXTUAL-V4`.
- Config:
  `configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml`,
  pre-launch SHA-256 `f2d188f9...0d89`, run identity
  `d3c2ded3...e8d8`; code
  `7bc1c06999b60597f2853de0425a70d1bb74f2c7`; planned output
  `artifacts/policy/PRL-02-R5-qwen3-grpo-bs16-tgvf-t1-formal-pilot-80step-gpu0123`.
- GPUs/session: physical GPUs 0--3; planned tmux
  `prl02_r5_t1_formal80_gpu0123`. Launch only after config load confirms the
  exact T=1.0 artifact identities, the R5 output root has no resumable state,
  and all four GPUs are free.
- Operational note: the first launch attempt was rejected before worker/GPU
  startup because its stale code-baseline pointer included later tracked
  config/report commits. No training state was created; the pointer was moved
  to the immediate pre-R5 code baseline above before retrying the same run.
- Actual launch: 2026-07-22 17:40 JST from manifest commit `2f2cc70`; tmux
  `prl02_r5_t1_formal80_gpu0123`. Strict config validation passed, the runtime
  printed the exact T=1.0 artifact identity above, and four FSDP workers began
  loading on physical GPUs 0--3.
- Final result: all 80 optimizer steps and paired checkpoints 0/10/20/45/80
  completed. The run processed 1,280 prompts, 10,240 trajectories and
  16,939,557 generated policy tokens. Cumulative answer reward was 0.608398,
  conditional-tool reward 0.557129, tool-call-attempt rate 0.941895,
  format-error rate 0.085449 and mean reasoning length 1,341.02 tokens. The
  final checkpoint is `global_step_80`; W&B continuation run `f0zcm6tm`.

### PRL-03-R1-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123

- Lifecycle/result: `CANCELLED` / `INVALID`; formal Crop-only comparison arm for
  the completed PRL-02-R5 TGVF-only Pilot. It initializes a fresh decoder-only
  LoRA from the original Qwen3-VL-8B-Thinking base and does not load the R5
  policy checkpoint.
- Controlled comparison: the exact DeepEyes-47K snapshot/order, seed 42,
  max-pixels 262144, BS16/n8/GA4, sampling, mixed verifier and Qwen2.5-72B
  fallback judge, reward 0.8/0.2/1.2, GRPO mathematics, optimizer/scheduler,
  FSDP2 world-size 4, colocated vLLM TP1, capacity and checkpoints
  0/10/20/45/80 are held fixed from R5. The sole method change is the accepted
  Crop-only v2 prompt/schema with `image_zoom_in_tool` and a four-call cap.
- Observation/runtime: every crop is cut from the immutable original RGB image,
  encoded by the colocated frozen Qwen vision tower and materialized for exact
  current/reference replay. The TGVF Adapter is not invoked or loaded in this
  profile; the representation table remains only a strict shared config-schema
  binding and is not part of the Crop observation path.
- Config:
  `configs/policy/runs/prl_03_r1_qwen3_grpo_bs16_crop_only_formal_comparison_80step_gpu0123.toml`,
  SHA-256 `ad1825776a0fcea61cc264b827427e869ae5dae6de4d3e4407b07ad41034e88f`,
  run identity `cd1d7440dc117740e48ef2b702a6ccd751e1cf42504676236efdaf757589748c`,
  code baseline `ab3a6243033fab29808af73ad78c4b31384ef3d9`.
- GPUs/session/output: physical GPUs 0--3 were idle at planning on 2026-07-22
  23:17 JST. Planned tmux session is `prl03_r1_crop_formal80_gpu0123`; output is
  `artifacts/policy/PRL-03-R1-qwen3-grpo-bs16-crop-only-formal-comparison-80step-gpu0123`.
  Launch uses the repository `.venv312` `tgvf-rl run-policy` entry point with
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `PYTHONHASHSEED=42` and `TOKENIZERS_PARALLELISM=false`. Acceptance requires
  real repeated Crop observations, exact replay, all scheduled checkpoints,
  compact W&B metrics and bounded local trajectory audits.
- Invalidating result: the first optimizer step completed with 128 trajectories,
  79 Crop-call attempts and 63 successfully materialized Crop observations, but
  `conditional_tool_reward` was zero for the entire batch. The reward-context
  adapter counted only `ToolCallRecord` (TGVF-only) instances instead of the
  trajectory's materialized observations, so successful `CropToolCallRecord`
  calls were invisible to the conditional reward. The run was stopped after
  step 1 at 2026-07-22 23:43 JST; it is diagnostic only and must not be resumed.

### PRL-03-R2-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123

- Lifecycle/result: `COMPLETE` / `INVALID`; clean replacement for invalid R1.
  It starts from the original Qwen3-VL-8B-Thinking base with a fresh
  decoder-only LoRA and must not load an R1 or R5 policy checkpoint.
- Controlled comparison: all R5 model, DeepEyes-47K order, seed 42,
  max-pixels 262144, BS16/n8/GA4, sampling, mixed verifier/judge, GRPO,
  optimizer, FSDP2/vLLM placement and checkpoint settings remain fixed. The
  method arm uses only the accepted `image_zoom_in_tool` Crop profile and does
  not load or invoke the TGVF Adapter.
- Reward correction: a successful active-tool observation is now counted from
  the immutable trajectory observation records, not from the TGVF-only call
  subclass. Errors without a materialized observation remain ineligible; the
  reward weights `0.8/0.2/1.2` and one-bonus-per-trajectory rule are unchanged.
  Twenty-two focused rollout/reward tests passed before planning.
- Config:
  `configs/policy/runs/prl_03_r2_qwen3_grpo_bs16_crop_only_formal_comparison_80step_gpu0123.toml`,
  SHA-256 `1e554cfa75f229ba9fc679124de6e1d596abf74c7c19be202d621a86b011b3fb`,
  run identity `91251b9f74a30c40cfa34cd03ecdfe01d230f8789269236f63acc7d73cc426d8`,
  code baseline `e2288debec2a0cc63d9fded0d0c1a823c15c5bd2`.
- GPUs/session/output: physical GPUs 0--3; planned tmux
  `prl03_r2_crop_formal80_gpu0123`; output
  `artifacts/policy/PRL-03-R2-qwen3-grpo-bs16-crop-only-formal-comparison-80step-gpu0123`.
  Launch requires all four GPUs free, strict config validation, no resumable R2
  state and the real first-step conditional reward to agree with successful
  Crop observations before the run is allowed to continue unattended.
- Actual launch: 2026-07-22 23:37 JST from manifest commit `ef25ad1`; tmux
  `prl03_r2_crop_formal80_gpu0123`, W&B run `7468ybjm`. Step 1 completed in
  184.69 seconds with 128 trajectories, 49 successful Crop observations,
  answer reward 0.53125 and conditional-tool reward 0.1171875. Among 78 bounded
  trajectory audits, all 15 correct trajectories with a successful Crop
  observation received the conditional bonus and there were zero missing or
  spurious bonuses; every recorded call name was `image_zoom_in_tool`.
- Recovery at step 11: the next rollout exposed equal raw tensor bytes with
  different dtype/shape semantics in the observation store. The store now
  disambiguates those tensors with a private semantic key while preserving the
  public raw-byte SHA and exact-replay protocol. The complete step-11 FSDP2 and
  project checkpoint pair is the sole resume boundary; 58 focused observation,
  replay, lifecycle, checkpoint, and Crop tests passed before resumption.
  The launch provenance gate now permits this committed recovery descendant
  only because the same observed commit records the recovery here; dirty
  recovery code remains forbidden.
- Recovery at step 19: optimizer step 19 completed and its full checkpoint was
  saved, but the following rollout sampled a one-pixel-wide `330x1` Crop. The
  Qwen visual processor rejected its aspect ratio before an observation could
  be materialized, and the plain Crop runtime propagated that sampled-geometry
  error instead of returning the required standard recoverable tool error. The
  runtime now maps only the processor's explicit aspect-ratio rejection (plus
  the existing empty-after-clamp case) to `RecoverableToolExecutionError`;
  unrelated tensor/layout `ValueError`s still fail closed. All eight focused
  Crop runtime tests pass. Resume uses the complete step-19 boundary with all
  scientific and training settings unchanged.
- Recovery at step 72: optimizer step 72 completed and the exception path
  saved its complete FSDP2/project checkpoint pair, but Ray teardown did not
  forward the attempted step-73 root traceback into `train.log`; this was not
  OOM, data exhaustion, or the repaired aspect-ratio error. Before resuming the
  exact step-72 boundary, the trainer now writes the original exception to the
  TaskRunner's own stderr before quiescing services or saving recovery state.
  This changes diagnostics only; data order, policy state, optimizer state and
  all scientific settings remain unchanged.
- Completion: all 80 optimizer steps and the step-80 checkpoint completed.
  The run processed 1,280 prompts and 10,240 trajectories, with 7,305 Crop-call
  attempts, 4,167 materialized observations and 2,500 recorded
  `tool_execution_failed` events. W&B run `7468ybjm` and the local artifacts
  are retained as engineering evidence.
- Post-hoc invalidation, 2026-07-24: the Qwen3 model-facing coordinate contract
  was wrong. Qwen3-VL natively emits relative `0..1000` coordinates, but this
  run's runtime treated those values as immutable-original-image pixels. In
  4,504 parseable sampled Crop calls retained by the trajectory audits, 4,492
  boxes lay wholly in `0..1000`, 3,177 (70.5%) were changed by direct source
  clamping, and 1,760 (39.1%) became empty. The model-to-source coordinate
  conversion was absent even for boxes that happened to remain in bounds.
  Therefore the optimizer/checkpoint/replay mechanics completed, but all Crop
  localization, Crop-vs-TGVF, reward, and benchmark method conclusions from
  this run are invalid. The run must not be resumed or promoted.

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

### PRL-JG-01-R1-QWEN25-72B-REAL-THREEROUTE-GPU01

- Lifecycle/result: `COMPLETE` / `PASS`.
- Question: do real DeepEyes MCQ/math/open reward routes produce correct binary
  decisions, with zero MCQ judge calls and fail-closed 72B fallbacks?
- Code: clean commit `64a7b1e3f865695110a8af7dbbd96a9305567322`;
  judge config SHA-256 `bf1d4f84...e1e3bf`; checker SHA-256
  `e717686e...e8866`.
- Model/service: local Qwen2.5-72B-Instruct revision
  `495f39366efef23836d0cfae4fbe635880d2be31`, BF16, vLLM 0.12,
  TP2, physical GPUs 0/1, port 8013; temperature 0, top-p 1, seed 42,
  max judge tokens 256, strict JSON binary verdict.
- Data: materialized DeepEyes-47K seed-42 manifest
  `3483c317...6f477`; exact real sample IDs are pinned in the checker: one each
  from `chart`, `thinklite_eureka`, and `vstar`.
- Prompt/verifier: RL-judge prompt SHA-256 `2fa039d7...86d2`; rule-first MCQ,
  math/open semantic fallback; any request, timeout, JSON, or nonbinary error
  aborts the check. No GPT fallback.
- Training/replay/TGVF/GRPO fields: N/A; this is a reward-route integration
  check with no policy rollout, optimizer, model update, or observation.
- Command: serve the exact model on `127.0.0.1:8013`, then run
  `tools/check_policy_rl_judge_routes.py` and stop the service.
- Output: `artifacts/policy/PRL-JG-01-R1-qwen25-72b-real-threeroute-gpu01/`.
- Result: full reward composition returned `0.8` for every correct candidate and
  `0.0` for every wrong candidate. MCQ made zero judge calls; math and open VQA
  each made four calls across direct-verifier and composed-reward checks, with
  correct binary decisions throughout. Result SHA-256 `4a293359...b7504`;
  server-log SHA-256 `8ca76ccc...0a89`. Service stopped and GPUs 0/1 returned
  to 0 MiB.

### BE-04-QWEN3-TGVF-STEP80-COREDEV2511-GPU0123

- Cell/status/result: post-training TGVF-only policy benchmark / `RUNNING` /
  launched `2026-07-23 11:55 JST`; no optimizer, reward, reference replay, or
  weight update occurs.
- Question: how does the completed 80-step TGVF-only policy perform on the
  same official CoreDev benchmark content used by the original-policy baseline?
- Evaluated policy: base `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`
  plus the exact step-80 LoRA mapping identity
  `561132e49848fd43f8e7f352ef54782249aff59b2a5d331027a0e5e0f78be321`
  from run `PRL-02-R5-QWEN3-GRPO-BS16-TGVF-T1-FORMAL-PILOT-80STEP-GPU0123`;
  tensor-file SHA-256 `222fca29...9d54`, policy-run identity
  `d3c2ded3...ce8d8`.
- Representation/tool: immutable Adapter export
  `REP-QWEN3-V4-CONTEXTUAL-V4`, Balanced Matrix CE T=1 contextual hidden
  state, main D plus all three D-DeepStack branches; native prompt bundle
  `b44d8a46...11c8a`, tool `tgvf_focus_tool`, maximum four calls. Every
  observation is materialized by the same single-GPU vLLM replica that samples
  the target and is retained in the local trajectory audit.
- Code/config: clean implementation commit
  `b53ecac0807e164cb141d0715aa11027ab8608ee`; evaluator/config SHA-256
  `b30ad3f7...7817` / `dc7cf50c...b273`; bound policy config SHA-256
  `f2d188f9...0d89`.
- Data: immutable CoreDev-2511 identity
  `coredev-2511-vlmevalkit-7055d301-v1`, membership SHA-256
  `a461d9b4...0579`. The first executable tranche is all 2,240 single-image
  rows in canonical seven-slice order. The remaining 271 multi-image rows are
  explicitly pending because the accepted crop/TGVF schemas contain no
  `image_index`; taking only the first image or silently compositing images is
  forbidden. Task materialization SHA-256 `dc3ef2e2...6c4`.
- Sampling/runtime: vLLM `0.12.0`, one single-B200 replica per rank with eight
  concurrent request sequences, LoRA rank
  64, BF16, TRITON_ATTN language attention, TORCH_SDPA vision attention,
  max pixels `262144`, max model length `16384`; temperature `1`, top-p `1`,
  top-k disabled, all penalties disabled, cumulative max response length
  `8192`, content-addressed seed stream rooted at `42`, processed sampled-token
  logprobs retained. No quantization or prefix caching.
- GPUs/session/output: physical GPUs 0--3, DP4/TP1, one canonical-row shard per
  GPU; tmux `be04_be05_coredev_step80_gpu0123`; output
  `artifacts/evaluation/BE-04-qwen3-tgvf-step80-coredev2511-gpu0123`.
- Command: first run each rank with `tools/run_policy_coredev_2511.py --mode
  worker --rank <0..3> --world-size 4 --max-tasks 1`; after all four real rows
  pass, resume the same durable rank JSONL files without `--max-tasks`. Crop
  BE-05 follows automatically on the same GPU rank after BE-04 completes.
- Scorer: final-answer-only TSV materialization and the already accepted pinned
  VLMEvalKit/Qwen2.5-72B scorer follow only after inference is complete; GPT is
  forbidden. Metrics and elapsed/GPU-hours remain pending.
- Resume note (`2026-07-23`): rank 0 stopped after 238 durable rows on an
  unmapped padded LM-vocabulary row; rank 1 stopped after 108 rows because the
  standalone bridge had collapsed vLLM's exact EOS finish evidence. Ranks 2/3
  completed. Commit `aa5507791ea180ba9ed6fb1906fcd1efbb406a31`
  preserves zero-width sampled-token identity and exact standalone vLLM
  termination evidence. The first resume then exposed vLLM's native EOS pair
  `stop/None`; commit `9981f2aaf6a58e045dbb6847109c1f2c76f4a1cd`
  admits that exact vLLM representation without weakening length handling; 30
  focused CPU tests passed. Resume skips every durable row and uses the latter
  commit for only the missing trajectories.

### BE-05-QWEN3-CROP-STEP80-COREDEV2511-GPU0123

- Cell/status/result: post-training Crop-only policy comparison / `COMPLETE` /
  `INVALID`; no optimizer, reward, reference replay, or weight update occurred.
- Question: how does the completed 80-step Crop-only comparison perform on the
  identical CoreDev content and single-image tranche used by BE-04?
- Evaluated policy: the same frozen Qwen3-VL-8B-Thinking base plus exact
  step-80 LoRA mapping identity
  `eed4ffeaf5b77277a41dafeba428a20d5f3c8bce73049c02e63f63292d78b0b0`
  from run `PRL-03-R2-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123`;
  tensor-file SHA-256 `a10b3e32...b596`, policy-run identity
  `91251b9f...26d8`.
- Tool/config: native prompt bundle `4bc9d8e2...dab0`, tool
  `image_zoom_in_tool`, maximum four calls. Each accepted crop is re-encoded by
  the same vLLM replica's frozen Qwen vision tower and records main plus native
  DeepStack tensors. Config SHA-256 `b961103f...a46`; bound policy config
  SHA-256 `1e554cfa...b3fb`; implementation commit and evaluator are identical
  to BE-04.
- Data/sampling/runtime/GPU/scorer: exactly BE-04, including the 2,240-row
  single-image tranche, explicit 271-row multi-image hold, temperature/seed/
  8192-token contract, vLLM `0.12.0` BF16 DP4/TP1 runtime and final-answer-only
  pinned scorer. This lane changes only the trained LoRA and selected native
  visual tool/prompt.
- Session/output: automatically starts rank-for-rank after BE-04 in tmux
  `be04_be05_coredev_step80_gpu0123`; output
  `artifacts/evaluation/BE-05-qwen3-crop-step80-coredev2511-gpu0123`.
- Resume note (`2026-07-23`): rank 2 completed; rank 3 stopped after nine
  durable rows on the same collapsed EOS boundary; ranks 0/1 never started
  because their preceding BE-04 ranks stopped. Missing rows resume under the
  same exact-evidence/EOS fixes through
  `9981f2aaf6a58e045dbb6847109c1f2c76f4a1cd`.
- Completion/invalidation: all 2,240 supported single-image trajectories and
  the pinned scorer completed, but the 2026-07-24 audit established that this
  evaluator shared the training runtime's missing Qwen3 `0..1000`-to-source
  coordinate conversion. Its generated/scored artifacts are retained, but no
  Crop localization, benchmark, or comparison claim may use this cell. It must
  be rerun from a newly trained, coordinate-correct Crop policy rather than by
  reinterpreting the already sampled observations.
- CPU repair note (`2026-07-24`): decision `CROP-COORDINATES-20260724` adds the
  Qwen3 `0..1000` family conversion, Crop observation v2 provenance, prompt/tool
  schema v3 identities, and plain/atomic exact-RGB fixtures. This is an
  engineering repair only, not a new experiment or a revalidation of BE-05;
  no GPU work was launched.

### T1-01-QWEN3-512-FULLIMAGE-CANARY-GPU0123

- Cell/matrix ID and mandatory/diagnostic class: `T1-01`; mandatory inference-
  only policy-data-selection canary and promotion gate for later full T1
  scoring. It is not policy training, a benchmark, T2, or a visual-tool run.
- Spike-plan git revision and VA0/VA1/VA2 approval references: user decision
  `QWEN3-RL-DATA-T1-20260725`, accepted in `PROJECT_TASK.md` §0.8.9; GPU
  authorization is physical devices 0--3 and inference only.
- Lifecycle status: `RUNNING`; four-rank vertical smoke launched
  `2026-07-25 23:19 JST` and passed before full-canary resume. Revision 0
  completed `2026-07-25 23:37 JST` with all 50 immutable manifests and all
  1,536 logical attempts validated. The sole length-finish replay completed on
  physical GPU 2 at response-budget revision 1 on `2026-07-26 00:08 JST`; no
  revision-2 request remains eligible. Qwen3 generation is complete and all
  generation GPUs are released. Deterministic verification completed at
  `2026-07-26 00:33 JST`; the authorized local semantic judge is separately
  planned below, after which reduction and the promotion report remain.
- Result: `PENDING`.
- Question: under the original Qwen3-VL-8B-Thinking policy, its native
  processor with `max_pixels=262144`, and eight independently seeded responses,
  does the stratified 192-row canary produce complete and auditable full-image
  `0/8`, `8/8`, and `1--7/8` difficulty decisions without response length,
  image conversion, or verifier failure being counted as an incorrect answer?
- Baseline and exact output path: DeepEyes is the methodological reference; no
  competing model cell is part of this canary. Fresh output root
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_selection/t1/T1-01-QWEN3-512-FULLIMAGE-CANARY-GPU0123`;
  the vertical smoke is a resumable prefix of this same identity.
- Model and processor identity: frozen original
  `Qwen/Qwen3-VL-8B-Thinking` at
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, BF16, no quantization,
  tokenizer length `151669`; model config/generation config/tokenizer config/
  tokenizer JSON/preprocessor SHA-256 respectively `5cd45286...f3661`,
  `fe72e865...e656`, `7b501e63...a7d5`, `a5d85b6d...73c7`, and
  `27225450...e516`. Model, processor, and runtime identity SHA-256 are
  `56d26e9f...b659`, `12c1fdfb...044b`, and `86a9d7fc...a9e9`.
- Representation checkpoint identity: N/A; TGVF Adapter, main D, and
  D-DeepStack are not instantiated by full-image T1 scoring.
- N/A fields and justification: reference policy, LoRA, optimizer, gradient,
  checkpoint/resume of weights, behavior log probabilities, KL, GRPO/SDPO,
  reward composition, target conditioning, TGVF/crop tools, and policy/reference
  replay are N/A because this run samples a frozen base model and changes no
  weights. Chunk resume applies only to immutable inference evidence.
- Policy/reference initialization: one immutable original Qwen3 policy per
  worker; no reference-policy or judge model is colocated during generation.
- Rollout policy version and allowed asynchronous staleness: stable local base
  path above, no adapter and no updates; staleness `0`.
- Code commit and worktree state: repository `main` HEAD
  `45481498fbbedcd139112339956d87d533355ac4` with a dirty worktree containing
  pre-existing unrelated work plus the explicitly hashed T1 files below; no
  claim of a clean commit is made.
- Repository adapter/patch surface and hash: CPU evidence/runtime
  `src/tgvf_rl/data/policy_selection_runtime.py@38814960d2cb652c77ec726048e4e9ba5755fe421ae80745d9139724344fa07a`;
  vLLM boundary
  `src/tgvf_rl/data/policy_selection_vllm.py@6d96d3911d94e3f77b39a940190409353a29ed6017028499ad400dac9648672e`;
  CLI
  `tools/run_policy_data_selection_t1.py@cc5323b3531caaaa3ed0475e850685c0500cc8644a4e894a39d132179c15c827`;
  run config file
  `configs/policy/data_selection/qwen3_t1_512_canary_v1.json@a20e1e7a31532b6480ef29803277f8d95af1d62e3e029b7f386ee0494f8d5f90`.
  The canonical run-manifest SHA-256 is separately
  `077a271e1e3048222507ff402085f44455f7f937e1b60a65e04cb6a271e9db16`.
  No site-package patch is used.
- Post-revision-0 length-retry boundary: the hashes immediately above remain
  the exact revision-0 generation identities. Before any higher-budget launch,
  the primary worker was fail-closed against `budget_revision > 0`; its current
  boundary/CLI hashes are
  `src/tgvf_rl/data/policy_selection_vllm.py@68b5b522c0059cd0523046ff49c828fb8ee52f157b776b54d9f67a4c88f7d23e`
  and
  `tools/run_policy_data_selection_t1.py@7b99ddbfbf0fcf3aef52e2316d8a985cbb8c9ca084d0d8e3f11825b6bd858736`.
  The first dedicated length-only scheduler boundary was
  `src/tgvf_rl/data/policy_selection_vllm_retry.py@84bcff4ddc9fb1ff268159c46a438479638f714e4b56e3c317a982618bac1849`;
  it launched revision 1 on GPU 2 at `2026-07-26 00:01 JST`, observed a normal
  completion shorter than the revision-0 length prefix, failed closed under
  its extra exact-prefix gate, wrote no evidence/manifest, and released the GPU
  at `00:03 JST`. Exact-prefix equality across a changed vLLM max-context
  engine was not part of decision `QWEN3-RL-DATA-T1-20260725`; the corrected
  scheduler records common-prefix length and first divergence as immutable
  audit evidence while retaining strict logical-request, seed, prompt, image,
  backend, consecutive-budget, and larger-budget identity checks. Its hash is
  `src/tgvf_rl/data/policy_selection_vllm_retry.py@0bb929af80056d4f1c875253f3da1f1e178e2b0e1ced2f4bba76f93db7751390`.
  The dedicated CLI is
  `tools/run_policy_data_selection_t1_retry.py@40ecc4d74003ef127640043181b936b0422dda35e960dd2a4ca74ace490adcf1`.
  They reuse the unchanged `_generate_one` path, require an explicitly
  acknowledged logical-request set, and preserve the seed. CPU selection/
  runtime tests: `32 passed`.
- Revision-0 completion audit: `1,535 stop`, `1 length`, `0 generation error`;
  192 candidates have exactly eight unique attempts each, with 1,536 unique
  logical keys and request IDs. The single replay source is ArxivQA candidate
  `6a5acc05...ce76`, attempt `5`, seed `1951992457`, request
  `qwen3-selection:e6a8fff...b5b93d`, source-evidence SHA-256
  `b5ddd7cb...68c82`, rank `2`, local chunk `4`. Revision 1 must therefore
  publish at most one record in chunk namespace `1000004`.
- Revision-1 completion audit: the one-record immutable manifest is
  `rank-02-chunk-1000004.json@4b2f7a1b2e8d25696e21638d8fdfd3c648df339320504d71525957212b626dc4`;
  its evidence finished normally after `3,313` sampled tokens, so the effective
  1,536 logical attempts now all have a normal completion. The changed-context
  replay shared 71 initial tokens with revision 0 and first diverged at index
  71. Immutable audit sidecar
  `runtime/length-retry-audits/bb7e251ac240e229a824d385ba246841dcb24576a9b76800933b8e154099b65c.json`
  has the matching SHA-256
  `bb7e251ac240e229a824d385ba246841dcb24576a9b76800933b8e154099b65c`;
  the successful worker log SHA-256 is
  `09c09634bfeef4dc6f036c0beebc67204e219624b67a8e7831a32328e66e8c23`.
- Dataset/manifest, hashes, sample rule, and n: authoritative full-source
  catalog `catalog-v2.json@428e782a...bd9`, with screened V*/ArxivQA-v2/
  ThinkLite JSONL identities `5a0b974c...a322` (191,975),
  `cda47ff2...742` (99,893), and `438588d7...c4ba` (69,842). The outcome-
  independent canary is 64 rows per source, 192 rows and 1,536 revision-0
  logical attempts; candidates JSONL `1898129e...3fa`, manifest file
  `39a271a0...a67`, manifest content `34dc4f6d...77e`, and ordered selected-
  record identity `2aecc596...a8c9`. All referenced image bytes/dimensions were
  preflighted; 187 unique images and four legitimate reuse groups remain.
- Native prompt/tool schema hash: schema
  `qwen-native-user-image-question-v1`; exactly one user message containing the
  original image then canonical question, no system message and no tools,
  `add_generation_prompt=true`. Ground truth, GT regions, rationale, and
  provenance never enter the prompt. Candidate-specific semantic prompt and
  exact vLLM-expanded prompt-token hashes are retained per attempt.
- Chat-template/token-fixture hash and token-ownership masks: native template
  content SHA-256 `36e042fe...956`; template file SHA-256
  `7dc0b863...c1e5`; the prompt must end exactly with template-owned
  `<|im_start|>assistant\n<think>\n`. No policy loss mask applies; every
  completion token ID and raw text is retained as inference evidence.
- D/DeepStack/position/mask identity: N/A; original full-image Qwen visual
  tokens only. V* GT regions are preserved in the candidate record but never
  crop, mask, or condition T1 generation.
- Observation materialization/artifact identity used by all replays: encoded
  image SHA/dimensions are verified, decoded PIL source mode is recorded, then
  `Image.convert("RGB")` discards alpha before the original RGB pixels enter
  Qwen's fast processor. The runner does not pre-resize. Qwen owns its sole
  torch/torchvision Bicubic smart resize with factor `32`, minimum area `65536`,
  maximum area `262144`, and preserved aspect ratio. Source-RGB pixel hash,
  processed dimensions, processor identity, prompt tokens, and MM cache UUID
  are recorded; any replay must reproduce them exactly.
- RL framework/version/environment lock: no veRL. `.venv312`: Python `3.12.3`,
  vLLM `0.12.0`, Torch `2.9.0+cu128`, Transformers `4.57.6`, Pillow `12.3.0`,
  FlashInfer Python `0.5.3`, CUDA runtime `12.8`, driver `570.195.03`.
- Objective equations and normalization: no optimization objective. Once all
  valid attempts are verified, the fixed T1 reduction retains `1--7/8`, drops
  `0/8` and `8/8`, and leaves any incomplete attempt group unresolved.
- Rollout/replay forward mode and adapter dropout/RNG contract: inference mode,
  immutable base weights, no adapter/dropout, one deterministic request seed;
  no policy update can intervene. Audited replay uses the same recorded image,
  expanded prompt, seed, sampling configuration, and budget revision.
- Sampling backend/version, seed, temperature, top-p/top-k/min-p, penalties,
  logit processors, and logprob convention: vLLM `0.12.0`; temperature `1`,
  top-p `1`, top-k `-1` (disabled), min-p `0`, repetition penalty `1`, presence
  and frequency penalties `0`, no logit processor, no logprobs requested. Eight
  low-31-bit seeds derive from canonical run identity, candidate identity,
  attempt index, root `42`, and namespace
  `qwen3-policy-selection-t1-canary-v1`. Request stop ID is `151645`; with
  `generation_config=auto`, the recorded effective native EOS set is
  `[151645,151643]`; both terminal texts are parser-supported. EOS is enabled,
  stop strings are empty, and special tokens are retained in detokenization.
- Weight/KV-cache dtype, quantization, attention implementation, rollout tensor
  parallelism, and training device mesh: BF16 weights, KV dtype `auto` (must
  resolve compatibly at startup), no quantization; automatic FlashInfer decoder
  is required and must be confirmed in the worker log, while vision attention
  is explicitly `TORCH_SDPA`. Do not set global `VLLM_ATTENTION_BACKEND`.
  Independent DP4/TP1 replicas; no training mesh.
- Logit/logprob/loss/gradient parity tolerances: logit/logprob/loss/gradient are
  N/A. Exact equality is required for candidate/run identities, source RGB,
  expanded prompt-token IDs, attempt seed, response budget, sampled-token hash,
  finish/stop reason, and resumed chunk contents.
- World size, microbatch, accumulation, and global batch: four independent
  workers, one per physical GPU; `max_num_seqs=32`,
  `max_num_batched_tokens=65536`, four candidates/evidence chunk and eight
  attempts/candidate (up to 32 active requests per chunk). Training microbatch,
  accumulation, and global batch are N/A.
- GPUs: physical 0 `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2`; 1
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`; 2
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`; 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`.
- Start/end timestamps, elapsed time, and session/process identity: started
  `2026-07-25 23:19 JST`; end/total elapsed pending; tmux session
  `t1_01_qwen3_512_canary_gpu0123`, windows `gpu0`--`gpu3`.
- Actual GPU-hours and peak scratch use: pending.
- Command: first run `.venv312/bin/python tools/run_policy_data_selection_t1.py
  prepare --config configs/policy/data_selection/qwen3_t1_512_canary_v1.json`.
  Each rank then runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=<rank>`, `VLLM_USE_V1=1`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TOKENIZERS_PARALLELISM=false`,
  `PYTHONHASHSEED=42`, the recorded Python/CUDA include `CPATH` and
  `.eval-runtime-python312-dev/lib` `LIBRARY_PATH`, plus rank-local Triton and
  TorchInductor caches: `.venv312/bin/python
  tools/run_policy_data_selection_t1.py worker --config
  configs/policy/data_selection/qwen3_t1_512_canary_v1.json --rank <rank>
  --budget-revision 0 --max-chunks 1`. After all four smoke chunks validate,
  rerun the identical command without `--max-chunks 1` to resume the rest.
- Outputs: immutable `run-identity.json`, canonical config, content-addressed
  raw JSONL under `chunks/`, logical rank/chunk manifests under `manifests/`,
  rank logs and caches under `logs/` and `runtime/cache/`. The vertical smoke
  committed four rank-0 local chunks: 16 V* candidates, 128 unique logical
  attempts, 128 normal EOS completions, zero length/error, sampled-token range
  `49..5807` and total `115705`; exact chunk validation and per-candidate prompt/
  source-RGB consistency passed, and all four logs confirmed FlashInfer.
- Scorer/parser identity: last non-empty suffix after the final `</think>`;
  ArxivQA canonical row-bounded A--Z rule and zero judge calls; ThinkLite
  normalized exact/numeric-symbolic rule; V* normalized exact rule. Only
  unresolved ThinkLite/V* semantics may later use the separately frozen local
  Qwen2.5-72B judge config `37375048...573` and prompt `2fa039d7...86d2`, after
  Qwen3 releases GPUs. Length/error/verifier failures are never incorrect.
- Metrics: pending. Required: all 1,536 logical attempts accounted for and
  unique; token quantiles, length/error counts, per-source `0/8`, `8/8`, and
  `1--7/8`, verifier-route/judge-call rates, exact replay subset, four-GPU
  throughput, projected full-run GPU-hours, and zero ArxivQA judge calls.
- Conclusion: pending smoke, complete canary, verifier audit, and promotion
  decision. The full 361,710-row run is not authorized under this identity.

### T1-01-J1-QWEN25-72B-SEMANTIC-JUDGE-GPU01

- Cell/class and lifecycle/result: `T1-01-J1`; mandatory inference-only local
  semantic-verifier continuation of `T1-01`; `COMPLETE` / `PASS`. It performs
  no training, no Qwen3 generation, no crop/TGVF execution, and no paid or
  remote request.
- Authorization and source-quality boundary: user instruction on
  `2026-07-26 JST` to proceed without wasting GPU; accepted amendment
  `PROJECT_TASK.md` §0.8.9.1. The exact corrupt V* row
  `bd8222dad80f...334e1b4` and its eight consumers are verifier failures and are
  not dispatched. No replacement label is synthesized and no other completed
  canary evidence is rerun.
- Parent/run inputs: Qwen3 run manifest
  `077a271e1e3048222507ff402085f44455f7f937e1b60a65e04cb6a271e9db16`;
  deterministic scoring manifest
  `dfdade6bff679fa074225148c7fb3151bdf4fb3b1234a9cffe4496a7dbd94542`;
  effective attempts `1,536`; deterministic attempt artifact
  `49dff5414481544a75cdc985df89e04753283a1e4cb0b899151c328097e0fa39`.
  The content-deduplicated judge queue contains `939` requests serving `999`
  attempt consumers, SHA-256
  `f0f3f647c49ad2c3084c073c9fce63f6ea34d728128a7138b6573989eba49c79`.
  ArxivQA judge calls are exactly zero.
- Quality-exclusion identity: config
  `configs/policy/data_selection/t1_canary_quality_exclusions_v1.json@3ef0b0bb5f646eee0d1cb0a95e282486ac5e85782c47c0698029c1c04a9602a1`;
  the eight bad-row attempts remain `source_ground_truth_invalid` and
  unscored.
- Judge/model identity: local `Qwen/Qwen2.5-72B-Instruct` revision
  `495f39366efef23836d0cfae4fbe635880d2be31`, stable path
  `/nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct`, served model
  `Qwen2.5-72B-Instruct`, BF16, no quantization. Judge config
  `configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json@3737504858912a6392679d2c9720597cde58dd7d3218aa6f75b67ad00a769573`;
  prompt `2fa039d7...86d2`; temperature `0`, top-p `1`, max tokens `256`, seed
  `42`, strict JSON-object binary verdict.
- Runtime/topology: `.venv312` Python `3.12.3`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`; physical GPUs 0/1 with the UUIDs recorded in parent `T1-01`,
  TP2, BF16, `max_model_len=32768`, memory utilization `.85`,
  `max_num_seqs=64`, prefix caching, `TRITON_ATTN`, host `127.0.0.1`, port
  `8013`. Client concurrency is `32`; every request retries only transport,
  timeout, HTTP, or strict-response failure and aborts after five attempts.
- Code boundary: semantic writer/finalizer
  `src/tgvf_rl/data/policy_selection_t1_judge.py@485fc27cae4829cbaa870b42e12ae0f98fc42a942f5396fd8d6b11af70c3297d`;
  CLI `tools/judge_policy_data_selection_t1.py@6fe994536612db1a4527a3909671d24fd8fa5bdc47944bd137117ced45902fc6`;
  deterministic scorer
  `src/tgvf_rl/data/policy_selection_t1_scoring.py@ba4a159a88503a5a24ad80292e277c1ec2394ffd637c461255eac59d46bc438e`;
  generic provider
  `src/tgvf_rl/judges/openai_compatible.py@36e6755bd74ca0bf6a4a9f998d9be67754228103771baa9ee6c51f79e57fa66a`.
  Targeted verification: `33 passed`; Ruff passed.
- Evidence/output contract: exact response model, one choice,
  `finish_reason=stop`, strict binary JSON, and token usage are mandatory.
  Canonical request, raw response hash, response JSON, all judge identities,
  verdict, rationale and content hash are persisted before an atomic per-request
  resume index. Output roots are parent-run `scoring/judge-v1` and
  `scoring/final-v1`; logs are `logs/local-judge-server-gpu01.log` and
  `logs/local-judge-client.log`; tmux session `t1_01_judge_gpu01`.
- Planned server command: environment binds physical `CUDA_VISIBLE_DEVICES=0,1`,
  `VLLM_USE_V1=1`, spawn workers, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, the
  repository Python-3.12 headers and `.venv312/bin`; run
  `python -m vllm.entrypoints.openai.api_server --model <stable-local-path>
  --served-model-name Qwen2.5-72B-Instruct --host 127.0.0.1 --port 8013
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768
  --gpu-memory-utilization .85 --max-num-seqs 64 --seed 42
  --generation-config vllm --enable-prefix-caching`. The client runs
  `tools/judge_policy_data_selection_t1.py run` with the pinned T1 and judge
  configs and concurrency `32`, then `finalize` after all 939 indices validate.
- Runtime/result: server launched in the planned tmux session at
  `2026-07-26 00:41:24 JST`, became healthy with the exact served-model identity
  at `00:42:54`, and the client launched at `00:43:29`. All `939/939` unique
  requests committed by `00:44:22`; the service shut down cleanly by `00:44:35`
  and physical GPUs 0/1 returned to `20/1 MiB`. A no-service resume validation
  then loaded all `939` indices and wrote zero new records.
- Judge outcome/evidence: `760` unique verdict-1 and `179` unique verdict-0;
  `436,683` prompt, `42,731` completion, and `479,414` total judge tokens.
  Judge manifest identity
  `e90efe29591ba5fd1d3ffed29978611d5bddaa35322aac672754393ae89e4b05`;
  manifest file SHA-256 `efb0079c...c685`; server/client log SHA-256 values
  `1f3ef5fa...0d7` and `859bdca5...893`.
- Final reduction: all `1,528` valid attempts scored (`1,052` correct, `476`
  incorrect); the eight pinned corrupt-GT attempts are `verifier_error`, never
  incorrect. Across 191 valid candidates, `59` retain, `105` too easy, and
  `27` too hard; the corrupt candidate is the sole unresolved row. Per source:
  ArxivQA `34/13/17` retain/easy/hard, ThinkLite `12/48/4`, and valid V*
  `13/44/6`, plus one unresolved corrupt V* row. Final manifest identity
  `3f5f8e5b94456831f6578f4381f97ce4a819e83d6b6810ac8f6ae3e196cbaef3`;
  attempts/decisions/report SHA-256 values `ab8944bc...5a64`,
  `d2b4c443...599f`, and `3c6e5b90...4c02`.

### T1-02-QWEN3-INSTRUCT-512-FULLIMAGE-CANARY-GPU0123

- Cell/class and lifecycle/result: `T1-02`; diagnostic inference-only paired
  token-length and throughput canary; `COMPLETE` / `PASS`. It is not policy
  training, a difficulty-selection promotion, T2, a tool run, or an answer-
  judge deployment.
- Authorization and question: decision
  `QWEN3-INSTRUCT-TOKEN-CANARY-20260726`, accepted in `PROJECT_TASK.md`
  §0.8.9.2, authorizes physical GPUs 0--3. On the exact T1-01 192-candidate
  population and generation contract, how much shorter is the native
  Qwen3-VL-8B-Instruct completion than Qwen3-VL-8B-Thinking?
- Paired baseline: `T1-01-QWEN3-512-FULLIMAGE-CANARY-GPU0123`, whose effective
  1,536 generations contain `3,026,449` sampled tokens, mean `1,970.344401`;
  ArxivQA/ThinkLite/V* means are `3,092.218750`, `2,239.726563`, and
  `579.087891`.
- Model identity: official `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, stable local path
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct`, architecture
  `Qwen3VLForConditionalGeneration`, BF16, no quantization. All four exact
  shard sizes and SHA-256 identities plus the index identity are pinned in
  `EXTERNAL_REFERENCES.md` under “Qwen3-VL Instruct paired diagnostic model”.
- Model-facing files: config `5cd45286...f3661`, generation config
  `8469742d...c4d44`, tokenizer config `c2da7718...e2399`, tokenizer JSON
  `a5d85b6d...73c7`, preprocessor config `27225450...e516`, chat-template file
  `5c72a170...0acce`, and effective template content
  `3636d0f0...89e4`; tokenizer length `151669`, tokenizer EOS `151645`, native
  generation EOS `[151645,151643]`.
- Native prompt contract: exactly one user message containing original image
  then canonical question, no system message and no tools,
  `add_generation_prompt=true`. The Instruct template must end exactly with
  `<|im_start|>assistant\n` and contains no template-owned `<think>` opener.
  Raw sampled token IDs/text are retained. Correctness parsing and scoring are
  N/A; the config's direct-completion parser identity is reserved but is not
  invoked by this cell.
- Data identity: exact T1-01 outcome-independent canary, 64 rows per source,
  192 candidates and 1,536 attempts; candidates
  `1898129ed01060f0b21ddeff9f081661902efbd87c8c31221c633cb64753e3fa`,
  manifest file `39a271a02b9d8eee3bda272427e0e341d6662e161d45b4a299fd1e2530c92a67`.
  The fixed corrupt V* label is irrelevant to token length and remains one
  diagnostic prompt; no correctness claim is allowed.
- Observation/image identity: immutable source bytes are verified and decoded
  to RGB; Qwen owns the sole Bicubic resize with factor `32`, minimum area
  `65,536`, maximum area `262,144`, aspect ratio preserved, and one image per
  prompt. Source-RGB hash, processed dimensions, visual layout, expanded prompt
  token hash and multimodal cache UUID are recorded per attempt.
- Sampling/response budget: eight independently seeded attempts; temperature
  `1`, top-p `1`, top-k disabled, min-p `0`, repetition penalty `1`, presence
  and frequency penalties `0`, no logit processor, EOS enabled, no stop string,
  special tokens retained. Seed root `42`, namespace
  `qwen3-instruct-policy-selection-t1-canary-v1`; seeds include the new run
  manifest. Revision 0 is `40,960` new tokens / `65,536` context; only a length
  finish may enter separately authorized immutable retries at `98,304/131,072`
  then `196,608/262,144`.
- Runtime/topology: `.venv312`, Python `3.12.3`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`, Transformers `4.57.6`, Pillow `12.3.0`, FlashInfer `0.5.3`,
  driver `570.195.03`; four independent DP4/TP1 workers, BF16, no quantization,
  FlashInfer decoder and `TORCH_SDPA` vision attention, `max_num_seqs=32`,
  `max_num_batched_tokens=65536`, prefix caching and chunked prefill enabled,
  four candidates/evidence chunk.
- GPUs: physical 0 `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2`; 1
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`; 2
  `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`; 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`.
- Code/config identity: repository HEAD
  `45481498fbbedcd139112339956d87d533355ac4` with a pre-existing dirty
  worktree; accepted runtime
  `src/tgvf_rl/data/policy_selection_runtime.py@1c1326caecde39547ca6cf4215b3f3fc9b3b25e1bb6e4f534163662fef0e6804`,
  vLLM boundary
  `src/tgvf_rl/data/policy_selection_vllm.py@c1962244b90a553b678ae3c66db8eacadef14ce68c3fc82401db90f25ad0e5ad`,
  CLI
  `tools/run_policy_data_selection_t1.py@7b99ddbfbf0fcf3aef52e2316d8a985cbb8c9ca084d0d8e3f11825b6bd858736`,
  config
  `configs/policy/data_selection/qwen3_instruct_t1_512_canary_v1.json@e86ac1adecda4c91cf33cf622379cca7b7ca2b70256f72577bb20778e96615b4`.
  Focused CPU tests: `36 passed`; Ruff: pass.
- Run identity/output: run manifest
  `c67969a3fded1d07ac077711a6bccf5cb8b124a2a2e7d5e459835ff901b57f6a`,
  model/processor/runtime identities `527c2873...0f73f`,
  `357445bf...2a793`, and `86a9d7fc...ea9e9`; output root
  `artifacts/data/policy_selection/t1/T1-02-QWEN3-INSTRUCT-512-FULLIMAGE-CANARY-GPU0123`.
- Planned command: prepare with `.venv312/bin/python
  tools/run_policy_data_selection_t1.py prepare --config
  configs/policy/data_selection/qwen3_instruct_t1_512_canary_v1.json`; launch
  one rank per physical GPU with the same CLI `worker`, rank `0..3`, budget
  revision `0`, rank-local caches, `VLLM_USE_V1=1`, spawn workers, and the
  recorded Python-3.12 header/library environment. A one-chunk-per-rank smoke
  must validate before resumable full-canary continuation.
- Required result: all 1,536 unique logical attempts accounted for; finish-
  reason counts; overall and per-source mean/P50/P90/P95/P99/max sampled tokens;
  attempts/s, sampled tokens/s, elapsed time and GPU-hours; exact comparison
  with T1-01.
- Runtime/result: four-rank smoke ran from the first logged model-resolution at
  `2026-07-26 01:35:01 JST` through the last smoke log update at `01:37:19.903`.
  Its 128 V* attempts all stopped normally. The resumable continuation started
  at the first logged model-resolution at `01:38:37` and completed at
  `01:44:12.070`; all GPUs 0--3 then returned to zero MiB. The full-continuation
  window was `335.070` seconds for 1,408 new attempts and 926,368 sampled tokens,
  or `4.2021` attempts/s and `2,764.70` sampled tokens/s including startup.
  Logged per-rank occupancy sums to about `0.317` B200-GPU-hours for the
  continuation and `0.452` B200-GPU-hours including the required smoke.
- Integrity/result artifact: all `1,536/1,536` candidate-attempt keys and
  request IDs are unique; every finish reason is `stop`, with zero length
  finish, generation error, or output containing `<think>`/`</think>`. Report
  `token-length-report.json@1aa9cc9a6165542e5c13b869adf6e760a8c982234305ef383cb58f7bcbee4a4f`;
  ordered manifest-set and chunk-set audit digests are respectively
  `329455b7e634d4f4d68c7fae0b9fa6ae0372cb2bd6f3b8efaa6338eef55a1354`
  and `b0f6b7b38ed866bfd8ec1dba81a3f0538aee5023893db9873fb095194fa34855`.
  Smoke log SHA-256 values by rank 0--3 are `deef84d2...399b`,
  `0c685469...4924`, `9099f061...b220`, `777b632d...e71e`; continuation log
  identities are `d47a9e59...e012`, `7fe24f6f...c478`,
  `6ffa653a...ac63`, and `0887c4bc...6614`.
- Token metrics: total `950,835`, mean `619.033`, P50/P90/P95/P99/max
  `259/1,439/2,989/6,292/12,675`. ArxivQA mean and
  P50/P90/P95/P99/max are `806.160` and
  `404/1,988/3,390/4,963/6,371`; ThinkLite `911.361` and
  `294/2,703/4,678/9,297/12,675`; V* `139.578` and
  `108/287/357/583/3,160`.
- Paired conclusion: T1-01 Thinking produced `3,026,449` tokens at mean
  `1,970.344`; Instruct uses `31.42%` as many sampled tokens, a `3.183x`
  reduction. Thinking/Instruct mean ratios are `3.836x` ArxivQA, `2.458x`
  ThinkLite, and `4.149x` V*. For the proposed 170k V* + 32k ArxivQA + 69,842
  ThinkLite population, the source-weighted projection is about `905.4M`
  sampled tokens at mean `416.33`, versus `2.831B` for Thinking (`3.126x`
  fewer). Because 2.175M image/prompt prefills do not shrink with output length,
  the measured four-B200 T1 projection is `3.5--4.5` days, central estimate
  about four days; this supersedes the preliminary token-only 1--2 day guess.
  This cell proves length/throughput only and does not yet prove Instruct tool-
  trajectory quality or authorize Instruct-filtered difficulty as a substitute
  for the Thinking primary.

### RP-65-QWEN3-INSTRUCT-REPRESENTATION-PERIODIC-BOUNDARY-FORMALDATA-CONTEXTUAL-GPU01

- Cell/class and authorization: `RP-65`; diagnostic Instruct-primary
  representation-phase FSDP2 optimizer/checkpoint/resume smoke authorized by
  `QWEN3-INSTRUCT-CLOSURE-SMOKES-20260726` in `PROJECT_TASK.md` §0.8.9.4.
- Lifecycle/result: `COMPLETE` / `PASS`; ran `2026-07-26
  03:11:06--03:13:27 JST`.
- Question: can a fresh, non-Thinking-initialized Qwen3-VL-8B-Instruct TGVF
  Adapter complete optimizer step 1, validation, distributed checkpoint and
  clean teardown on GPUs 0--1, then strictly restore and complete step 2?
- Baseline/output: no promoted baseline artifact is consumed. Exact root is
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-65-qwen3-instruct-periodic-boundary-formaldata-contextual-gpu01`;
  it did not exist at planning time.
- Model/processor identity: official `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, stable local path
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct`, tokenizer length
  `151669`, effective chat-template SHA-256
  `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`,
  BF16, SDPA, no quantization, no tokenizer resize, and image maximum area
  `262144` pixels.
- Representation initialization/checkpoint identity: TGVF Adapter is fresh
  random initialization at seed `42`; Thinking Adapter and historical TGVF
  checkpoints are forbidden. Resume consumes only the step-1 DCP written by
  this exact cell at `checkpoints/representation-qwen3-instruct-rp65-periodic-boundary-step-00000001`.
- Policy/reference/rollout fields: policy RL, frozen reference, rollout,
  behavior log probabilities, KL, reward, GRPO, SDPO, tool execution and answer
  judge are N/A because this is supervised representation training only.
- Code identity: repository HEAD
  `45481498fbbedcd139112339956d87d533355ac4`, dirty worktree bound by the
  runner's complete live-code digest
  `94f604e49d8ec1b25b0871759c8424ba786ae0124b571d32e6dd98042c1c0f5e`.
  Fresh config file/source SHA-256 is
  `4819df716789025a3256963cc5d5fbd3bbde276630a80dc3e4a2ce8f232425ff`
  and canonical SHA-256 is
  `2aa6c89f37bd7c4395225d9560309554af2d9206b01e2acee8e6829e4bd2b645`;
  resume values are
  `5e3ff858f79b2bc9ba8a1c8c1bbe6d0eb1adb51ad98df3e8250fbda252500627`
  and `53e8a48ab34439834921e543ce7fdd214a87e355fcca696ddc78aea253911ff4`.
- Dataset identity: clean-imend v4 train/test JSONL SHA-256 values
  `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`
  and `de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d`;
  accepted overlap report
  `3cad19a9d0e359ea368071082092f431349859a7520bc03a39cce6e90dffdc27`.
- Prompt/transcript identity: native representation image-question prompt
  `{question}` at
  `bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9`.
  The Instruct evidence answer is serialized in assistant `content`; no
  DeepEyes policy-RL think trigger or template-owned think opener is added.
- D/DeepStack/conditioning: contextual hidden state provider, layer `-1`;
  preserved full main `D` plus all three D-DeepStack branches. Every branch is
  trained and validated under the native visual layout and masks.
- Objective: `L = 1.0 * balanced_mean_NLL_MatrixCE(T=1.0) + 1.0 * L_gen
  + 0.1 * L_norm`; manifold loss is disabled. AdamW uses LR `1e-4`, betas
  `(0.9,0.999)`, epsilon `1e-8`, weight decay `.01`; historical cosine horizon
  `2000`, warmup `100`, minimum LR ratio `.1`; gradient clipping norm `1.0`.
- Forward/checkpoint determinism: seed `42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  adapter dropout exactly zero, optimizer-boundary-only DCP, no policy update
  between the sampled training batch and its backward pass. Strict restore
  covers Adapter/model-owned state, optimizer, scheduler, sampler/RNG,
  metrics history and validation cursor.
- Runtime/topology: `.venv312` Python `3.12.3`, Torch `2.9.0+cu128`,
  Transformers `4.57.6`; native `torchrun` FSDP2 mesh `[2]`, no offload,
  `reshard_after_forward=false`, BF16 parameter/output and FP32 reduction.
  Per-rank batch `4`, one group/rank, gradient accumulation `4`, world size
  `2`, exact global batch `32`.
- GPUs: physical 0 `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` and physical 1
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`.
- Start/end/session, elapsed, GPU-hours and scratch: `2026-07-26
  03:11:06--03:13:27 JST`, about `141s` wall and below `0.08` B200-GPU-hours;
  tmux `rp65_instruct_repr_gpu01` exited cleanly. Peak allocated/reserved bytes
  were `62.665/77.915 GB` at step 1 and `62.841/79.608 GB` at step 2; artifact
  root uses `806 MiB`.
- Commands: fresh uses `CUDA_VISIBLE_DEVICES=0,1`, the determinism environment
  above, `PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=src`, and
  `.venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/smoke/representation_qwen3_instruct_rp65_periodic_boundary_formaldata_contextual_gpu01.toml
  --stop-after-global-step 1`. After strict resume-config validation, the same
  command uses the `_resume.toml` config and `--stop-after-global-step 2`.
- Outputs/acceptance: `fresh.log`, `resume.log`, metrics and DCP shards under
  the exact root. PASS requires events `start, train1, validation1, train2,
  validation2`, finite component/total losses and gradient norm, two rank
  shards in each step-1/step-2 DCP, exact strict restore, unchanged tokenizer,
  clean exits and released GPUs. This smoke does not promote an Adapter.
- Result evidence: exact events are `start`, train/validation step 1, then
  train/validation step 2 after strict restore. Step-1 train losses are Matrix
  CE `1.383356`, L-gen `4.202366`, weighted norm `.049564`, total `5.635286`,
  gradient norm `2.509434`; validation total `5.784420`. Step-2 train values are
  `1.385070`, `4.408077`, `.049477`, total `5.842623`, gradient norm `2.191001`;
  validation total `6.340272`. All are finite. Tokenizer remained `151669`.
  Each checkpoint contains two rank `.distcp` shards plus metadata. Run identity
  is `f954ea9a82fdd74068e04dcc41f80862401e5b7a379c499d076c5072af3d38dc`;
  metrics SHA-256 `8b9e08c8...1d07`, fresh/resume logs `ea5d850f...3c29` /
  `3974434a...f007`. GPUs 0--1 returned to idle. Conclusion: code-level Instruct
  representation support is now physically closed through optimizer and exact
  checkpoint/resume, but no 2000-step Adapter is promoted.

### PRL-DIAG-11-QWEN3-INSTRUCT-CROP-PROMPT-V4-GPU2

- Cell/class and authorization: `PRL-DIAG-11`; diagnostic inference-only
  policy-RL prompt-trigger canary authorized by
  `QWEN3-INSTRUCT-CLOSURE-SMOKES-20260726` in `PROJECT_TASK.md` §0.8.9.4.
- Lifecycle/result: `COMPLETE` / `FAIL`; ran `2026-07-26
  03:11:04--03:12:39 JST`.
- Question: with no template/environment think prefill, does the real Instruct
  model generate its own single `<think>...</think>` block on the initial
  native crop-tool prompt and again after a successful real crop observation?
- Baseline/output: DeepEyes' observable initial/post-tool instruction trigger is
  the method reference; no DeepEyes weights or outputs are consumed. Exact
  fresh output root is
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/canaries/qwen3-instruct-crop-prompt-v4-gpu2-20260726`.
- Model/processor identity: official `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, stable local path, tokenizer
  length `151669`, chat-template SHA-256
  `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`,
  BF16, SDPA, no quantization, maximum image area `262144`.
- Representation/policy/reference identity: no TGVF Adapter, LoRA, optimizer,
  reference policy, teacher, reward, KL, GRPO/SDPO or weight update. This is a
  frozen original-model crop-only prompt diagnostic, not a policy rollout used
  for loss; behavior log probabilities are therefore not requested.
- Code/config identity: HEAD
  `45481498fbbedcd139112339956d87d533355ac4` with dirty worktree. Prompt,
  appender and agent-loop SHA-256 values are respectively
  `a08a93e9285a2cc66cd6edf2d399d834628205ba95a8eeced42440bf58e4a83f`,
  `cc27c23096acea73951a66b1c8587be21304594ad2a3bde7345ca00c79c5493a`
  and `387797072691d5b8502c154803c124a9ed8b7929d849de84c0ea33d2bbd36bfa`.
  Driver `tools/canary_qwen3_instruct_crop_prompt.py` SHA-256 is
  `ce00c1d8e0bb3d10cc5bc23cc956eae427c5d0f755bee3290af6ce009f3fb193`;
  config SHA-256 is
  `be1052c624f5a13f7e03325b0f3ad8f2844d904e9028556a23bc1ff2c636ed57`.
- Dataset/sample identity: one pinned DeepEyes47K-derived local record
  `deepeyes47k:2fabfc60669372c10a147d0e4d84377d6092c981dd3212e066814676eb1acca7`;
  source-image byte SHA-256
  `2988a806a98f49472a37594afb332b7a37d901a35dba6975568a5423668654e1`,
  size `1480x2016`, with the exact question and answer `C` stored in config.
- Prompt/tool schema identity: crop-only Qwen native schema,
  `tgvf-visual-tool-prompts-v4-instruct` /
  `tgvf-visual-tool-responses-v2-instruct`, bundle
  `f42ddae1911714452e9a0fbf775f695c55f3796a76080c0c72d1c282319df0c4`.
  Initial prompt text/token hashes are `47af668b...ed19` / `f7d93bf4...4b1c`;
  post-crop values are `e7a9a1df...b467` / `806b0af5...521e`.
- Transcript/token ownership: both prompts end exactly with template-owned
  `<|im_start|>assistant\n` and no think opener. Initial user text says think
  first; the successful tool turn is exact observation text, image placeholder,
  repeated instruction, then assistant header. Every emitted token, including
  think/tool tags, is model sampled; raw text, exact token IDs and their hashes
  are retained. Environment repair is forbidden.
- Observation/crop identity: controlled native call uses original-relative
  Qwen coordinates `[0,0,1000,580]`, mapped to source box
  `[0,0,1480,1169]`; the real RGB crop is saved and byte-hashed. Initial arm
  receives the source image; follow-up receives source plus this crop.
- Runtime/sampling: `.venv312` Python `3.12.3`, Torch `2.9.0+cu128`,
  Transformers `4.57.6`; frozen eval/inference mode on one B200. Four seeds
  `2026072601..2026072604` per arm, temperature `1`, top-p `1`, top-k `0`,
  no logit processors, maximum `4096` new tokens/turn and stop string
  `</tool_call>`. There is no asynchronous staleness or KV/replay claim.
- Topology/tolerances: physical GPU 2, DP1/TP1, eight sequential generations.
  Numerical logit/loss/gradient parity is N/A; exact prompt hashes, crop bytes,
  raw completion/token IDs and configured seeds are required.
- GPU: physical 2 `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`.
- Start/end/session, elapsed, GPU-hours and scratch: `2026-07-26
  03:11:04--03:12:39 JST`, about `95s` wall / below `0.03` B200-GPU-hours;
  tmux `prl_instruct_prompt_v4_gpu2` exited cleanly, GPU 2 returned idle, and
  the output root uses `450 KiB`.
- Command: `CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
  .venv312/bin/python tools/canary_qwen3_instruct_crop_prompt.py --config
  configs/smoke/qwen3_instruct_crop_prompt_v4_gpu2_canary.json --physical-gpu 2`.
- Outputs/parser/metrics: immutable fresh `controlled_crop.png` and
  `result.json`; strict native parser validates the controlled call. Required
  per-arm metrics are opener/closer totals, starts-with-think and valid-single-
  think counts, native-tool envelope/action counts, termination reasons,
  max-token count and sampled-token total/mean/min/max. Missing or malformed
  think remains recorded evidence and fails compliance; it is never repaired.
- Result/conclusion: all four initial responses and all four post-crop responses
  emitted a syntactically enclosed native crop tool call, but opener/closer
  totals were `0/0` in both arms; starts-with-think and valid-single-think are
  `0/4` and no response hit the 4096-token budget. Initial sampled-token
  total/mean/range is `231/57.75/54--68`; post-crop is
  `205/51.25/47--59`. Thus the V4 wording reliably triggers tool use (`8/8`)
  but does not trigger policy-sampled think tags (`0/8`). Result JSON SHA-256 is
  `7e9a2fbf2527d714b61be87ee32de60c6b839f06c5943468546a7f9f81d4a1ed`;
  crop SHA-256 is `4f273b1f...809f`. A strict missing-think gate would reject
  every otherwise valid tool action, so V4 is not accepted for RL promotion.
- Post-canary CPU closure: the agent loop now follows DeepEyes' observable
  `0/0` envelope semantics. No-tag native calls/direct answers remain executable
  while think compliance remains zero and fully visible; any one-sided,
  duplicated, reversed or misplaced envelope still fails before tool execution.
  No tag is inserted. Current agent-loop SHA-256 is
  `9224870c73fe31c35cdebe7f6ad412ff68bdb8e9177b56ec1b76f69d53c8fb25`;
  a real-shape fixture proves no-tag Instruct tool call -> observation -> no-tag
  final answer retains both sampled turns and executes once. Combined Instruct
  run-config/prompt/appender/agent-loop/filter/representation tests: `71 passed`;
  Ruff passed. This repairs trajectory availability but does not change the
  measured conclusion that V4 itself triggers zero think tags.

### T1-03-QWEN3-INSTRUCT-512-FILTER-RESUME-SMOKE-GPU3

- Cell/class and authorization: `T1-03`; diagnostic inference-only T1 filter
  interruption/resume equivalence smoke authorized by
  `QWEN3-INSTRUCT-CLOSURE-SMOKES-20260726` in `PROJECT_TASK.md` §0.8.9.4.
- Lifecycle/result: `COMPLETE` / `FAIL`; launched `2026-07-26 03:11 JST` and
  failed closed during vLLM startup at `03:12:09`, before any request sampled.
- Question: can the Instruct T1 generation/filter input be stopped after one
  immutable chunk, resumed without duplicate attempts, and finish byte-
  identically to its declared uninterrupted logical request set?
- Baseline/output roots: active root
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_selection/t1/T1-03-QWEN3-INSTRUCT-512-FILTER-RESUME-SMOKE-GPU3`,
  continuous control suffix `-continuous-baseline`, audit suffix
  `-resume-audit`; all three were absent at planning time.
- Model/processor identity: official `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, stable local path, BF16, no
  quantization, tokenizer length `151669`, config/generation/tokenizer-config/
  tokenizer-JSON/preprocessor/chat-template SHA-256 values
  `5cd45286...f3661`, `8469742d...c4d44`, `c2da7718...e2399`,
  `a5d85b6d...73c7`, `27225450...e516`, `3636d0f0...89e4`.
- Representation/policy/reference fields: TGVF Adapter, D/DeepStack, LoRA,
  optimizer, reference policy, behavior log probabilities, KL, GRPO/SDPO,
  reward and tools are N/A. Frozen full-image inference changes no weights.
- Code/config identity: HEAD
  `45481498fbbedcd139112339956d87d533355ac4` with dirty worktree. Config
  `55686caaa4fc4ce503a23400acb909df8784777125d87528404b098d03bde221`;
  run manifest `d0c67a1be34772de2fe47e8119ad55524f16896efbab08a47d0fa84df489c34f`;
  resume plan `bfdd83e72417952e32fed3da8d97e79a43c801ed45c1582c63e6d19860c3571f`.
  Runtime/vLLM/smoke-driver SHA-256 values are
  `af09040a...0bbb`, `c1962244...e5ad`, `b3d9ad5c...c8bf`, and
  `e5b805a8...e5c3`; underlying worker CLI is `7b99ddb...58736`.
- Dataset/request identity: exact rank-3 first two chunks from the accepted
  192-row canary: eight V* candidates with identity hashes
  `af88d7d6...a4ef`, `410eecde...80f3`, `dbcba9cd...c88f`,
  `f9ff0b47...cc2f`, `1c865339...2747`, `70be874e...7647`,
  `8e46e9e...829f`, `fcfca043...3f3f`; eight attempts each, exactly 64
  unique logical requests. Parent candidates/manifest SHA-256 values are
  `1898129e...3fa` and `39a271a0...a67`.
- Prompt/transcript identity: exactly one user message, source image then
  canonical question, no system, no tools, `add_generation_prompt=true`; no
  policy-RL think trigger. Instruct template supplies only the assistant header.
  Ground truth, regions and provenance never enter generation.
- Observation identity: source image bytes/RGB pixels are verified; no
  pre-resize. Qwen fast Bicubic smart resize owns factor `32`, minimum area
  `65536`, maximum area `262144`, aspect ratio preservation and RGB conversion.
- Runtime/sampling: `.venv312` Python `3.12.3`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`, Transformers `4.57.6`, Pillow `12.3.0`, FlashInfer `0.5.3`;
  DP1/TP1, BF16, FlashInfer decoder, `TORCH_SDPA` vision attention,
  `max_num_seqs=32`, chunked prefill and prefix cache. Attempt seeds derive
  from root `42`, exact run manifest, candidate and attempt. Temperature `1`,
  top-p `1`, top-k disabled, min-p `0`, all penalties neutral, no logit
  processor, EOS enabled, revision-0 budget `40960/65536`; exact sampled token
  IDs/text are immutable evidence. No model update or asynchronous staleness.
- Objective/parser/tolerances: no optimization objective. Direct-completion-v1
  is reserved for later scoring but not used to alter this smoke's generation.
  Exact equality is required for run/config identities, seeds, request IDs,
  prompts, image hashes, two manifests and two evidence chunks.
- Topology/GPU: logical worker rank `3` of the fixed DP4 partition, executed
  alone on physical GPU 3 `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`;
  four candidates/chunk and 32 attempts/chunk. Training batch fields are N/A.
- Start/end/session, elapsed, GPU-hours and scratch: `2026-07-26 03:11--03:12
  JST`; tmux session `t1_03_instruct_resume_gpu3` exited nonzero. The worker
  reached model/profile loading only; peak model load was `18.0933 GiB` and
  usage was below `0.02` B200-GPU-hours.
- Commands: run, in order, `.venv312/bin/python
  tools/smoke_policy_data_selection_t1_resume.py {plan,baseline,interrupt,resume}
  --config configs/policy/data_selection/qwen3_instruct_t1_512_filter_resume_smoke_gpu3_v1.json`.
  The driver pins every worker subprocess to `CUDA_VISIBLE_DEVICES=3`, vLLM V1
  and spawn mode.
- Outputs/acceptance: continuous run must write `2` chunks/`64` records;
  SIGINT run must leave exact prefix `1` chunk/`32` records; resume must report
  `32` resumed plus `32` written; an immediate idempotent rerun must report
  `64` resumed, zero written and unchanged core digest. Run identity, canonical
  config, two manifests and two evidence chunks must be byte-identical between
  continuous and resumed roots; any drift, duplicate, extra/incomplete chunk or
  lingering process fails closed.
- Failure evidence/conclusion: Triton's startup compiler found the pinned
  Python `Python.h` but not its architecture-qualified
  `x86_64-linux-gnu/python3.12/pyconfig.h`; vLLM exited before KV-cache startup.
  The active root contains only the immutable run identity and canonical
  config, with zero manifests/chunks/attempts. The audit root contains only the
  startup log. No stop/resume conclusion is permitted from this attempt.

### T1-03-R1-QWEN3-INSTRUCT-512-FILTER-RESUME-SMOKE-GPU3

- Cell/class and authorization: `T1-03-R1`; infrastructure-only retry of the
  diagnostic T1 filter interruption/resume smoke, under the same accepted
  `QWEN3-INSTRUCT-CLOSURE-SMOKES-20260726` scope.
- Lifecycle/result: `COMPLETE` / `FAIL`; launched `2026-07-26 03:14:18 JST`;
  worker completed both baseline chunks at `03:15:51`, then the orchestration
  process failed while decoding the mixed vLLM log/JSON stream.
- Question/output: identical to `T1-03`. The zero-attempt parent roots will be
  preserved with suffix `-failed-attempt0`; R1 recreates the exact configured
  active, continuous-baseline and resume-audit roots from fresh paths.
- Immutable experiment identity: model, processor, dataset, 64 logical request
  identities, prompt, image preprocessing, sampling, vLLM version/topology,
  parser boundary, config SHA-256
  `55686caaa4fc4ce503a23400acb909df8784777125d87528404b098d03bde221`,
  run manifest `d0c67a1be34772de2fe47e8119ad55524f16896efbab08a47d0fa84df489c34f`,
  plan `bfdd83e72417952e32fed3da8d97e79a43c801ed45c1582c63e6d19860c3571f`,
  code hashes, GPU UUID, N/A fields and byte-equality acceptance are exactly the
  complete values recorded in parent `T1-03`; no algorithmic field changes.
- Sole runtime correction: `CPATH` is
  `.eval-runtime-python312-dev/root/usr/include:.eval-runtime-python312-dev/root/usr/include/python3.12`
  instead of only the second directory. A pre-launch GCC syntax check including
  `Python.h` passed and resolved the architecture-qualified `pyconfig.h`.
  `LIBRARY_PATH`, rank-local Triton/TorchInductor caches, CUDA visibility and
  all driver/worker commands remain otherwise identical.
- Policy/reference/objective/replay fields: unchanged N/A as frozen inference;
  no behavior log probabilities, rewards, optimizer or model update.
- GPU/session/timing/usage: physical GPU 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; planned tmux session
  `t1_03_r1_instruct_resume_gpu3`; `2026-07-26 03:14:18--03:15:51 JST`, under
  `0.03` B200-GPU-hours. Model load was `18.0933 GiB`; vLLM allocated a
  `134.24 GiB` KV-cache budget during the bounded worker.
- Command: the same ordered `{plan,baseline,interrupt,resume}` driver commands
  recorded in `T1-03`, under the corrected compile environment above.
- Output/result: the worker validly committed chunk 0
  `82a8c2d6...c158` and chunk 1 `f0dcb611...44a8`, exactly `64` records with
  no duplicate key. The driver then treated INFO/progress bytes preceding JSON
  as a JSON document and stopped before archiving the baseline. Therefore this
  attempt proves only uninterrupted two-chunk generation, not stop/resume.

### T1-03-R2-QWEN3-INSTRUCT-512-FILTER-RESUME-SMOKE-GPU3

- Cell/class and authorization: `T1-03-R2`; fail-closed orchestration repair and
  continuation of the accepted diagnostic T1 interruption/resume smoke.
- Lifecycle/result: `COMPLETE` / `PASS`; CPU baseline recovery/archive passed,
  then GPU interrupt/resume ran `2026-07-26 03:18:41--03:21:51 JST`.
- Question and immutable identity: same stop/resume byte-equivalence question,
  model/processor, prompt, image path, 64 logical requests, seeds, sampling,
  worker/vLLM code, config `55686caa...21`, run manifest `d0c67a1b...34f`, plan
  `bfdd83e7...571f`, GPU and all N/A objective/reference/replay fields recorded
  completely in `T1-03`. No sampled evidence is reinterpreted or altered.
- Input/baseline boundary: R1's two validated immutable chunks are the exact
  declared continuous baseline. The repaired CPU-only baseline phase validates
  its two manifests and final worker result, writes the baseline snapshot, then
  atomically archives the unchanged active directory. It does not regenerate
  or rewrite any attempt.
- Sole code repair: `tools/smoke_policy_data_selection_t1_resume.py` SHA-256
  `9af70934dfd6afbea9d3fac8a5e409f97282c78006fc6ede133632290385b00a`.
  Its JSON extractor now scans complete JSON objects inside mixed vLLM logs,
  and baseline recovery accepts only an already validated exact two-chunk set.
  Against the real R1 log it extracted four objects and the exact final
  `{chunks_written:2, records_written:64, records_resumed:0}` result. All
  worker/runtime/source/config hashes remain unchanged.
- Runtime correction/topology: the R1 corrected Python include path is retained;
  DP1/TP1, one physical B200, rank 3, FlashInfer/TORCH_SDPA and response budget
  remain exact. No model update, asynchronous staleness or behavior-logprob
  claim applies.
- GPU/session/timing/usage: physical GPU 3
  `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`; planned tmux session
  `t1_03_r2_instruct_resume_gpu3`; about `190s` wall and below `0.06`
  B200-GPU-hours for the two bounded model startups/workers. Observed peak GPU
  allocation was `158838 MiB`; all GPUs were released by `03:21:51`.
- Commands: first rerun the `baseline` driver command CPU-only to recover/archive
  R1. Then run the unchanged `interrupt` and `resume` driver commands; `resume`
  includes the final no-service idempotent worker invocation after completion.
- Outputs/acceptance: exact continuous `2/64`, deliberate SIGINT prefix `1/32`,
  resumed `32+32`, idempotent `64+0`, stable core digest and byte-identical run
  identity/config/two manifests/two chunks. Any drift or extra/incomplete chunk
  fails closed. Difficulty quality, T2 and training remain out of scope.
- Result evidence: the interrupted worker exited nonzero immediately after
  chunk 0 and left exactly `1` manifest / `32` records, snapshot
  `919450b96011d6322eb997520b8034b34f38721ecc37f27fc35aa439239e6eac`.
  Resume reported `records_resumed=32`, `records_written=32`; the immediate
  idempotent invocation reported `64/0`. Continuous and resumed snapshot hashes
  are both `8fd48126024899417e9eb403d48982954bce071fc177e3cc1789c213fba74268`;
  both core digests, including before/after idempotent rerun, are
  `f6930bd51c309393e1a53914a0ec1a1463bc19de8225b2536243d3aebcb3d012`.
  All six compared files are byte-identical. Canonical report identity is
  `863d8b924ccd80916b152dd52f708603b9efc3c87bdf119b99295648b124903d`
  and report-file SHA-256 is `892894ff...5d70`; interrupt/resume driver logs are
  `a62fd099...c02d` / `82bfda79...8cc2`. Conclusion: the T1 generation boundary
  now has a proven deliberate-stop, exact-resume and idempotent-restart path.

### RP-66-QWEN3-INSTRUCT-REP-BALANCED-T1-CONTEXTUAL-2000-GPU01

- Cell/class and authorization: `RP-66`; formal fresh Instruct representation-
  phase training authorized by `QWEN3-INSTRUCT-RP-T1-FORMAL-20260726` in
  `PROJECT_TASK.md` §0.8.9.5.
- Lifecycle/result: `RUNNING`; launched `2026-07-26 04:06:58 JST` after every
  identity below validated and physical GPUs 0--1 had no foreign compute
  process.
- Question/output: produce the first complete Qwen3-VL-8B-Instruct TGVF Adapter
  for the current line at
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/representation/RP-66-qwen3-instruct-balanced-t1-contextual-2000-gpu01/adapter.pt`.
- Model/processor identity: official `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, stable local path
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct`, tokenizer length
  `151669`, chat-template SHA-256
  `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`,
  BF16, SDPA, native fast image processing with maximum area `262144`, no
  tokenizer resize and no Thinking checkpoint reuse.
- TGVF/conditioning identity: fresh full main `D` plus D-DeepStack branches at
  Qwen visual layers `(8,16,24)`; contextual hidden state at layer `-1`;
  initialization seed `42`. RP-65 fixes the expected initial Adapter-state
  SHA-256 `fe4a55bb1ca8170aa9fbb881c917685002217e816b597d837ed2fde2ff08b7cb`.
- Data/prompt identity: v4 clean-imend train/test source SHA-256 values
  `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c` /
  `de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d`;
  retained manifests `a089d13d...5a8c` / `534f5b1e...06b0`; two exact image-
  path overlaps bound by report `3cad19a9...27`; validation-data identity
  `83c13af1379aeb9fc11f157e59ad5f78a834ebc4b4ec9a4fd5f4350449463760`.
  Prompt is native representation image + question only, SHA-256
  `bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9`;
  no policy-RL think/tool trigger is present.
- Objective/mathematics: Balanced mean-NLL Matrix CE at temperature `1.0`,
  L-gen mean per-sample NLL and Adapter norm, weights `1/1/.1`; global CE
  numerator over valid rows and global sum of per-sample mean NLL over sample
  count. AdamW LR `1e-4`, betas `.9/.999`, epsilon `1e-8`, weight decay `.01`;
  historical cosine 2,000 steps, 100 warmup, minimum LR ratio `.1`; gradient
  norm cap `1.0`. Policy/reference log probabilities, KL, GRPO/SDPO, reward,
  behavior replay, LoRA and a policy optimizer are N/A.
- Batch/topology: same-image K4, local group batch 4, one group/rank/microstep,
  gradient accumulation 4, FSDP2 world 2, global sample batch 32. Physical GPUs
  0--1 UUIDs `GPU-853e7816-9a2d-954e-ea14-8b62373bdfb2` and
  `GPU-32a298d3-ea53-7f70-7894-171fca21dcdc`; no-reshard, Adapter parameters
  sharded, frozen Qwen replicated.
- Schedule/checkpoint: 2,000 optimizer steps; metric every 10; full validation
  and synchronous optimizer-boundary DCP v2 at 500/1000/1500/2000; final
  rank-zero Adapter export. Ordinary checkpoints are retained, but no power-
  failure supervisor or automatic restart is part of this cell; nonzero exit
  is not PASS.
- Code/config identity: HEAD
  `45481498fbbedcd139112339956d87d533355ac4`, dirty-state SHA-256
  `67095b316963376cb1ee533ce5f7a898d1ab9728bb397f11662951785cd8f222`;
  config `configs/representation/qwen3_instruct_balanced_t1_contextual_2000step_gpu01.toml`,
  source TOML SHA-256 `37c06f4a7f2b538e668582471d4105f9c7dd93a86459d2e23d1f36e289c9d549`,
  canonical config SHA-256
  `bc73fb247bfd66f9578eea4a7c1fc48ad5767593fa97bf7a5bb7402bc25a7255`,
  expected run identity
  `97ccfd849e1d66cdd57be805c27524fa97ca60973e5be45d6d060acd5bc54e53`.
- Runtime/session/estimate: `.venv312` Python `3.12.3`, Torch `2.9.0+cu128`,
  Transformers `4.57.6`; planned tmux `rp66_instruct_2000_gpu01`, eight-hour
  process timeout, expected wall time 3--4 hours and 6--8 B200-GPU-hours.
- Launch confirmation: both ranks loaded the four model shards; the emitted
  run/config/data identities equal the planned values. Step 10 is finite:
  Matrix CE `1.386156`, L-gen `5.035878`, weighted norm `.046074`, total
  `6.468107`, pre-clip gradient norm `2.671467`; measured steady train-step
  ceiling was `4.655s`. GPUs 0--1 are active and GPUs 2--3 remain unused.
- Command: `CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=src timeout 28800s
  .venv312/bin/torchrun --standalone --nproc-per-node=2 -m tgvf_rl.cli
  run-representation configs/representation/qwen3_instruct_balanced_t1_contextual_2000step_gpu01.toml`.
- PASS gate: clean exit, exact expected run identity, finite train/validation
  components and gradient norms, terminal step 2000, four complete two-rank
  DCPs, tokenizer invariant, strictly loadable full-D/three-DeepStack final
  Adapter and manifest, no incomplete staging, and complete GPU teardown.

### T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123

- Cell/class and authorization: `T1-04`; formal inference-only `T1` generation
  authorized by `QWEN3-INSTRUCT-RP-T1-FORMAL-20260726` in `PROJECT_TASK.md`
  §0.8.9.5. It may launch only after RP-66 reaches its complete PASS gate and
  releases GPUs 0--1; RP-66 weights are not consumed by this tool-free filter.
- Lifecycle/result: `RUNNING_HYBRID_SUBSHARDED_REV0`; CPU preparation completed and the four
  revision-0 rank workers launched at `2026-07-26 10:29:32 JST`, then stopped
  at the user-requested resumable checkpoint on `2026-07-26 13:19:44 JST`.
  Output root
  `/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/artifacts/data/policy_selection/t1/T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123`
  retains the exact run identity/config records and all atomically committed
  chunk evidence.
- Model/prompt/image: same frozen local Qwen3-VL-8B-Instruct revision as RP-66,
  BF16/no quantization, tokenizer `151669`, template SHA-256 `3636d0f0...89e4`.
  Exactly one native user message contains source image then canonical question,
  no system, no tools, and `add_generation_prompt=true`; answer parser is
  `direct-completion-v1`. Qwen fast Bicubic smart resize preserves aspect ratio
  with factor 32, minimum area `65536`, maximum area `262144`, no pre-resize.
- Candidate identity: outcome-independent namespace
  `qwen3-instruct-t1-recommended-20260726-v1` selects candidate-identity hash
  bottom-k V* `170000` and ArxivQA `32000`, plus all ThinkLite `69842`:
  `271842` candidates and `2174736` logical attempts. Candidate JSONL SHA-256
  `51f4cfeeaa8278c2938f938a2992b30cf91ad9219d5c98d796c8137c60e8b3ec`;
  manifest SHA-256
  `4b99c90a02b031d871b560b2f9de1c1beba5e52730d3f29b83e74e268979326d`.
  Screened source counts/SHA-256 values are V* `191974` /
  `a8a2789c...9be4`, ArxivQA `99893` / `cda47ff2...1742`, and ThinkLite
  `69842` / `438588d7...2cba`.
- Sampling/budgets: eight independent stable candidate-attempt seeds under root
  42; temperature `1`, top-p `1`, top-k disabled, min-p `0`, all penalties
  neutral, no logit processor, EOS enabled. Primary revision is 40,960 new
  tokens in context 65,536; only length-finished attempts later replay at
  `98304/131072`, then `196608/262144`. Raw token IDs/text, prompt/image hashes,
  finish reason and seed are retained. No TGVF Adapter, crop tool, LoRA,
  optimizer, reward, policy/reference replay, behavior log probabilities,
  KL, GRPO or SDPO applies.
- Runtime/topology: `.venv312` Python `3.12.3`, vLLM `0.12.0`, Torch
  `2.9.0+cu128`, Transformers `4.57.6`, Pillow `12.3.0`, FlashInfer `0.5.3`;
  four independent DP workers, TP1, `max_num_seqs=32`, FlashInfer decoder,
  TORCH_SDPA vision, chunked prefill and prefix cache. Physical GPUs 0--3 are
  the four UUIDs recorded in RP-66 plus `GPU-11d59daa-e835-5f46-faaf-356bfebcabe3`
  and `GPU-a634a9e0-4e88-6f1f-764e-9a6c31581f2b`.
- Sharding/resume: stable rank candidate/chunk counts are r0 `68182/17046`, r1
  `67669/16918`, r2 `68177/17045`, r3 `67814/16954`; four candidates and 32
  attempts per full chunk. Evidence and manifests publish atomically. Restart
  strictly validates and skips a complete content-addressed chunk; an in-flight
  uncommitted chunk is recomputed. User stop sends SIGINT only to the four
  recorded worker process groups. Corrupt or identity-drifted chunks fail
  closed. `T2`, judge reduction and balancing remain out of scope.
- Code/config identity: config SHA-256
  `cd8c9a848ea8725beb74a03d762784323989f1a4e935e1f672f88832f2ca9182`,
  run manifest `bdc49eba27ff16aec58ac1116b7eda2a9148f62c334a6fbdd6385502fdf2141f`,
  model/processor/runtime identities `527c2873...73f` / `357445bf...793` /
  `86a9d7fc...9e9`. Recommended materializer/runtime/vLLM/worker SHA-256 values
  are `17b8133e...9f6`, `6c4320ed...f8e`, `c1962244...5ad`, and
  `7b99ddbf...736`. CPU preparation validation passed `42` focused tests plus
  Ruff; the earlier real T1-03-R2 smoke proves stop/resume byte equivalence.
- Estimate/commands: 4 B200s for 3.5--4.5 days, central estimate 4.0 days. Each
  rank starts in its own `setsid` process group with rank-matched
  `CUDA_VISIBLE_DEVICES`, `VLLM_USE_V1=1`, spawn workers, disabled tokenizer
  parallelism, the verified Python include/library paths and independent
  Triton/TorchInductor caches, invoking `tools/run_policy_data_selection_t1.py
  worker --config configs/policy/data_selection/qwen3_instruct_t1_512_vstar170k_arxiv32k_thinklite69842_v1.json
  --rank <0..3> --budget-revision 0`.
- Actual launch/progress: launched from committed HEAD
  `a2c925de76611848b033508b4dcb31bfa46fad93`. Rank 0--3 process-group leaders
  are recorded under `runtime/rank-<rank>.pgid`; launch logs are
  `logs/revision0-rank-<rank>.log`. A first tmux/`setsid` orchestration attempt
  left rank 2 running alone long enough to commit chunks 0--5. It was stopped
  with `SIGINT` while the launcher was corrected; those six immutable chunks
  were retained and the formal rank-2 worker strictly resumed from chunk 6.
  By `2026-07-26 10:32 JST`, all four workers had loaded the exact Instruct
  model with context 65,536, FlashInfer decoder and TORCH_SDPA vision path,
  occupied physical GPUs 0--3 at approximately 159 GiB each, and had committed
  120 manifests / 3,840 unique attempt records in total. The active PGIDs at
  launch were rank 0 `1313609`, rank 1 `1313611`, rank 2 `1313614`, and rank 3
  `1313616`. GPU 0 also had 0.3--0.5 GiB of non-compute G/EGL context owned by
  another user's jobs whose compute ran on GPUs 4--5; it did not block the
  rank-0 vLLM allocation or first committed chunks.
- Pause checkpoint: `SIGINT` was sent only after validating the four recorded
  process-group leaders at `13:19:29 JST`; all groups exited by `13:19:44` and
  physical GPUs 0--3 returned to zero compute memory. The immutable checkpoint
  contains rank 0--3 manifest counts `4342/4457/4574/4364`, respectively:
  `17737` complete chunks, `567584` revision-0 attempt records and `70948`
  complete candidates, or `26.098984%` of `2174736` logical attempts. Each
  rank's chunk indices are contiguous from zero through its last manifest, all
  four terminal evidence SHA-256 values match their manifests, and there are
  no empty manifest/chunk or temporary files. vLLM reports `EngineDeadError`
  in the four worker tails after the process-group interrupt; this is the
  expected in-flight shutdown path, and no uncommitted chunk is part of the
  checkpoint. Resume uses the identical command/config/rank/revision contract;
  committed chunks validate and skip, while at most one interrupted in-flight
  chunk per rank is recomputed.
- Remote continuation: at `2026-07-27 00:35:31 JST`, revision 0 resumed on
  host `s54` using its idle `NVIDIA RTX 6000 Ada Generation` 48 GiB device,
  UUID `GPU-869e331a-0afe-c618-7138-1efa17f3697e`, driver `560.35.03`.
  The logical four-rank/config identity remains unchanged; a single physical
  CUDA device runs logical ranks 0--3 sequentially. The mapping-only worker,
  CLI and launcher SHA-256 values are respectively
  `352b29996616b79d607b83ae5fed415ee7c7116e99977a8fe3e35a544c3cadad`,
  `5f210d24891c823f61e4c0af18aa9f0867f525c7105aaff4e9b43f9cb1f566a2`,
  and `66fc9990136f5d8bf7e8bc7a54be9baf62d07ab15fa90bc2a6067dc2341c9fec`;
  focused resume tests passed `3` tests and Ruff. The exact primary runtime
  versions above and the local veRL source at commit
  `e003163181731412595257a72ec173071efb125f` are installed remotely.
  A bounded rank-0 canary strictly resumed 138,944 records and atomically
  committed chunk 4342: 32 records, evidence SHA-256
  `e806cc9242ff5da1ed8b9277da1c1ddd0d29fd0b3fe18f145a590b5dd6f2a6ca`,
  manifest SHA-256
  `853972e62b2d3a2a99696436ea6fd79a4b9e4745145b665779412a61406ba38b`.
  Model weights occupy 18.0933 GiB and vLLM provisions 16.50 GiB / 120,144
  tokens of KV cache, sufficient for the 65,536-token revision-0 contract.
  The detached continuation process-group leader is `3280713`, recorded in
  `runtime/single-gpu.pgid`; its log is
  `logs/revision0-rtx6000ada-continuation.log`. Local B200 writers must remain
  stopped until remote output is stopped, incrementally synchronized back, and
  manifest/evidence hashes are validated. Cross-architecture sampled-token
  byte identity with the original B200 workers is not assumed; the GPU model
  boundary and canary provenance are therefore explicit here. The detached
  worker subsequently committed rank-0 chunks 4343 onward under approximately
  38.4 GiB device memory and 98% observed GPU utilization. A live four-chunk
  incremental backup to the paused local root took 5.56 seconds for manifests
  plus 5.52 seconds for evidence (786,428 new evidence bytes); all four copied
  manifest/evidence hashes validated. This read-only backup does not authorize
  a concurrent local writer.
- Local subshard acceleration: at `2026-07-30 23:15:11 JST`, the local broad
  rank-1/rank-3 process groups `3819902`/`3819905` were sent `SIGINT` after
  committed chunks 6599/5748 and fully exited within 15 seconds; rank 2 PGID
  `3819897` and the remote rank-0 writer were not interrupted.  This user-
  authorized disjoint-rank schedule supersedes the earlier blanket local-
  writer pause; the remote sequential launcher must stop after rank 0 before it
  can advance to a logical rank now owned locally.  The physical
  worker subdivision preserves the original logical rank and original
  `local_chunk_index`: ownership is
  `local_chunk_index % subshard_count == subshard_index`, so it changes neither
  run/config identity nor manifest names.  Rank 1 uses count 2 on GPUs 2/5;
  rank 3 uses count 3 on GPUs 4/6/7.  The broad and subshard launchers share a
  rank-scoped `flock`, reject live opposite topology, reject mixed counts or
  duplicate indices, fail closed on malformed PGID records, bind their output
  root to the canonical configured root, and close the lock FD in the child.
  The runtime/CLI/subshard-launcher/broad-launcher SHA-256 values are
  `5350022e...04f2`, `2fb381a8...cc9c`, `8feaba57...27cd`, and
  `b671a797...7a84`; focused resume/subshard/runtime/replay coverage passed 55
  tests plus Ruff and both launcher syntax/guard checks.  An independent
  read-only review returned GO before the switch.
- Subshard launch result: the first five cold-cache PGIDs failed before any
  chunk commit because this host lacks `/usr/include/python3.12`, causing a
  Triton `cuda_utils` helper rebuild to fail.  No manifest or evidence was
  published by those attempts.  Each isolated cache was then seeded byte-for-
  byte from its same-logical-rank cache that had already run this exact
  environment.  The successful PGIDs are rank-1 index 0/1
  `4041611`/`4041618` and rank-3 index 0/1/2
  `4041846`/`4041855`/`4041867`.  By `23:22:24 JST`, all five occupied their
  assigned B200 at approximately 159 GiB and had committed respectively chunks
  6604, 6613, 5772, 5785 and 5807; their residues are exactly 0/1 modulo 2 and
  0/1/2 modulo 3.  Concurrent rank-2 generation remained healthy.
- PASS gate: all four workers exit zero; exact expected revision-0 chunk set has
  no missing/extra/duplicate logical request; all `2174736` attempts are unique
  and accounted for; every manifest/evidence pair validates; no length/error is
  converted to incorrect; full image/prompt/seed/token provenance remains
  bound. Generation completion alone does not claim selected-dataset quality.

### RP-68-QWEN3-INSTRUCT-REP-BALANCED-T1-VISION-ROUTING-2000-GPU0123

- Cell/class and lifecycle: isolated representation-structure experiment;
  `COMPLETED_PASS` at `2026-08-02 03:34 JST`. The step-10 smoke and four-rank
  DCP passed, the strict resume continued from step 10 rather than step 0, and
  the run reached exactly step 2000 with finite metrics and a complete
  four-rank terminal DCP. It changes only the Adapter
  information route relative to RP-66 and does not change data, objective,
  optimizer, schedule, initialization, parameter shapes, or total Adapter
  capacity.
- Structural intervention: Adapter variant
  `full_d_deepstack_vision_routing`. The target hidden state may determine the
  second-stage attention query/key and therefore routing weights, but the value
  payload is the first-stage visual context rather than `target + visual`.
  This removes the direct target-to-D value path while preserving RP-66's main
  D, three DeepStack branches, 104 tensors, and approximately 72M parameters.
- Model/data/objective: Qwen3-VL-8B-Instruct; the exact RP-66 clean-imend train
  and test sources, image/question prompt, balanced Matrix CE + L-gen + norm
  objective with weights `1/1/.1`, AdamW LR `1e-4`, and 2,000-step historical
  cosine schedule are unchanged.
- Batch/topology: physical GPUs 0--3, FSDP2 world size 4, same-image K4,
  per-rank group batch 4, one group per rank per microstep, gradient
  accumulation 2. Thus each optimizer update still consumes eight image-axis
  matrices, matching RP-66 world2/GA4; the topology change does not change the
  scientific global-update batch.
- Fail-closed launch sequence: the full 2,000-step identity first runs with
  `--stop-after-global-step 10`; step 10 must emit finite metrics and a complete
  four-rank DCP. A distinct strict-resume config must load that exact DCP before
  the run may continue to step 2,000. Checkpoint publication precedes scheduled
  validation, so a validation failure cannot erase a completed optimizer
  boundary.
- Post-training gates: terminal Adapter/metrics identity is independently
  verified, then `INT-DIAG` runs the established first-200 internal diagnostic;
  `ACC-VAL` runs `image_only` and `image_correct_D` on first-200 and full-867
  with the pinned semantic rescore. Crop RL promotion is forbidden unless both
  hash-bound completion markers report `status=complete`.
- Code/config identity: implementation commit
  `06c8c47a8338921a17991551e40f71563da2002c`; fresh and strict-resume configs
  are `configs/representation/qwen3_instruct_balanced_t1_vision_routing_2000step_gpu0123.toml`
  and `configs/representation/qwen3_instruct_balanced_t1_vision_routing_2000step_gpu0123_resume_step10.toml`.
- Evaluation identity correction: the checkpoint receipt continues to bind the
  training commit above, while generated INT-DIAG/ACC-VAL configs bind the
  latest committed evaluator identity paths actually executed. This prevents
  unrelated post-training repository updates from falsely assigning the old
  training commit to a later evaluation program; dirty evaluator paths remain
  fail-closed.
- Controller recovery correction: the step-10 smoke and step-2000 resume append
  to the same metrics JSONL. Revalidation now binds the exact accepted byte
  prefix and its terminal record, permits only later appends, and still rejects
  shrinkage or any mutation within that prefix. This removes the false
  step-2000 transition refusal without weakening artifact identity checks.
- ACC output isolation correction: generation and semantic-rescore artifacts
  publish under `artifacts/evaluation`, outside the protected RP-68 training
  and Adapter tree. Control logs, generated source configs, and completion
  markers remain in the hash-bound post-training control directory.
- Output: `artifacts/representation/RP-68-qwen3-instruct-balanced-t1-vision-routing-2000-gpu0123`.
- Terminal identity: run identity SHA-256
  `27ad99c4dcc9f81ba0eaf8f60aa9180627ea047eafee6b9d2542904aa71378ab`;
  final `adapter.pt` SHA-256
  `051d8210ab62a5a6ac657f4df252d60c52dbd2efb37759e5e540705d04b67eee`;
  Adapter manifest SHA-256
  `5033200460ca521f375bdd184f98108d254e44f3dbaa8a96f1f1f0a13b723593`.
- Post-training gates: `INT-DIAG` completed at `03:51 JST` with report SHA-256
  `27ab7bf5d7b25be08404f63e9b9a2552b30c0fee7008840e615699c2b416d5f4`;
  `ACC-VAL` completed and was accepted at `04:21 JST`. On first-200,
  `image_only` was `141/200 = 70.5%` and `image_correct_D` was
  `170/200 = 85.0%`, a `+14.5 pp` gain. On full-867, `image_only` was
  `644/867 = 74.2791%` and `image_correct_D` was
  `748/867 = 86.2745%`, a `+11.9954 pp` gain.
- RP-66 comparison: the full-867 RP-68 correct-D score is nominally `+1.73 pp`
  over RP-66 step 2000, but the paired McNemar result is only `p ~= 0.096`, and
  identical RP-66/RP-68 image-only texts received approximately 2 pp different
  semantic-judge scores. The cleaner same-data INT-DIAG comparison regressed:
  query retrieval top-1 `87.0% -> 28.5%`, correct D beating same-image wrong D
  `95.5% -> 40.5%`, and correct D beating all controls `84.8% -> 25.0%`.
  Therefore RP-68 proves that vision-only value routing preserves strong
  answer utility, but it is not established as an improvement over RP-66 and
  its target specificity is substantially worse.

### PRL-04-R0-QWEN3-INSTRUCT-GRPO-BS16-CROP-T1CANARY-1STEP-GPU4567

- Cell/class and lifecycle: non-formal engineering smoke;
  `FAILED_PREFLIGHT_NO_GPU_ALLOCATION` at `2026-08-02 00:28:49 JST`. The
  fail-closed controller stopped this cell and did not start RP-68 or any
  downstream validation/RL stage.
- Model/tool/data: Qwen3-VL-8B-Instruct, native DeepStack enabled, fixed
  `image_zoom_in_tool` Crop-only protocol, and the verified 51-row retained
  ArxivQA subset from T1-02 Instruct canary. The data is explicitly
  `provisional/pilot`; it cannot be reported as the completed T1-04 selection.
- RL identity: GRPO, decoder LoRA rank/alpha 64, AdamW LR `1e-6`, global prompt
  batch 16, eight trajectories per prompt, physical GPUs 4--7, world size 4,
  gradient accumulation 4, and exactly one optimizer step. Reward retains the
  historical Crop pilot weights: answer `.8`, format `.2`, conditional tool
  use `1.2`.
- Acceptance: terminal metrics must show `optimizer_step=1`, 16 prompts and 128
  trajectories; `global_step_1` must be durable. The identical config is then
  launched with `resume_mode=auto`; a proof binds unchanged metrics and
  checkpoint-tree SHA-256 and rejects any `global_step_2`. No RP-68 training or
  formal Crop run may start if either check fails.
- Code/config identity: implementation commit
  `8e51818428585691635a5e415979e323f53dbd16`; config
  `configs/policy/runs/prl_04_r0_qwen3_instruct_grpo_bs16_crop_t1canary_1step_gpu4567.toml`.
- Output: `artifacts/policy/PRL-04-R0-qwen3-instruct-grpo-bs16-crop-t1canary-1step-gpu4567`.
- Failure evidence: veRL's Ray WorkerDict used physical resource indices as
  local CUDA ordinals while the parent process exposed only devices 4--7;
  workers 1/2 failed in `torch.cuda.set_device` with `invalid device ordinal`
  before model allocation, rollout, optimizer state, metrics, checkpoint, or
  output-root publication. The durable controller attempt/log is under
  `artifacts/overnight/RP68-CROP-RL-FAIL-CLOSED-20260802`.

### PRL-04-R0R1-QWEN3-INSTRUCT-GRPO-BS16-CROP-T1CANARY-1STEP-GPU0123

- Cell/class and lifecycle: corrected non-formal engineering smoke;
  `COMPLETED_PASS` at `2026-08-02 00:46 JST`. It preserves every R0 scientific
  field and changes
  only the physical device binding to the already validated veRL ordinal
  contract 0--3 plus a fresh run/output identity.
- Model/tool/data and RL identity: Qwen3-VL-8B-Instruct; Crop-only
  `image_zoom_in_tool`; verified 51-row provisional T1-02 ArxivQA canary;
  GRPO decoder LoRA rank/alpha 64; LR `1e-6`; BS16, n8, world4, GA4; exactly
  one optimizer step; answer/format/conditional-tool weights `.8/.2/1.2`.
- Acceptance remains unchanged: metrics terminal `optimizer_step=1`, 16
  prompts, 128 trajectories, durable `global_step_1`, followed by an exact
  auto-resume invocation proving unchanged metrics/checkpoint SHA-256 and no
  `global_step_2`.
- Code/config identity: implementation commit
  `06c8c47a8338921a17991551e40f71563da2002c`; config
  `configs/policy/runs/prl_04_r0r1_qwen3_instruct_grpo_bs16_crop_t1canary_1step_gpu0123.toml`.
- Output: `artifacts/policy/PRL-04-R0R1-qwen3-instruct-grpo-bs16-crop-t1canary-1step-gpu0123`.
- Fail-closed controller: `RP68-CROP-RL-FAIL-CLOSED-R1-20260802`, config
  SHA-256 `7e0e128a2bda7fdef7b52e327d04f1ce05b3995263c7a73090a67aad930ad712`;
  it has zero automatic retries and cannot promote past a failed artifact gate.
- Result: one real end-to-end optimizer step completed in 272.258 seconds with
  16 prompts, 128 trajectories and 36,909 generated policy tokens. The
  `global_step_1` checkpoint is durable. A second identical invocation loaded
  model, optimizer, RNG and scheduler state from that checkpoint and proved no
  step 2 was created; the checkpoint-tree SHA-256 is
  `7e14c9491f20b6cdb6526fb3d9f726a4c2b14d8811848409b6ec7452bb95f424`.

### T1-04-REV0-LENGTH-WAIVER-AND-ARXIVQA-MATERIALIZATION

- Cell/class and lifecycle: deterministic scoring policy adjustment;
  `COMPLETED_PASS` at `2026-08-02 02:51 JST`. The planned revision-1 GPU retry
  was cancelled before any GPU worker or revision-1 manifest launched. The 194
  revision-0 `finish_reason=length` attempts are 0.0089206% of 2,174,736
  logical attempts, so this exact T1-04 run treats them as terminal truncated,
  unscoreable evidence rather than regenerating them.
- Scope and effect: the waiver is bound only to run
  `T1-04-QWEN3-INSTRUCT-512-FULLIMAGE-271842-GPU0123`; unrelated future T1
  runs remain strict. The 194 attempts cover 161 candidates (0.0592256%): 31
  ArxivQA, 84 ThinkLite and 46 V*. Truncated attempts remain `correct=null`,
  never enter the semantic judge, and make only their affected candidate
  unresolved. Existing later-budget evidence, if present, still supersedes the
  revision-0 length result.
- Deterministic result: all 2,174,736 effective generations and attempts were
  validated and published. Status counts are scored `360569`, truncated `194`,
  and verifier-error/semantic-required `1813973`. Decision counts are retain
  `25393`, exclude-too-easy `5177`, exclude-too-hard `4367`, unresolved
  `236905`. ArxivQA requires zero semantic-judge calls; its 25,393 retained
  candidates are sufficient for the Crop pilot. The deterministic scoring
  manifest SHA-256 is
  `f307838c26e67fb5a162baf56d4e102f38c7e47856913c9dd414386e8917f309`.
- Materialized dataset: output
  `artifacts/data/policy_rl/T1-04-INSTRUCT-FULL-ARXIVQA-RETAINED-PROVISIONAL-PILOT-v1`
  contains exactly 25,393 rows with samples SHA-256
  `93b9886830903114df5bfadc8cc8911be627868a3a1b408bf5b7a77f1f4fd2e0`,
  content SHA-256
  `957a3de6e9f3e8e7bba39711e8c1fd0dcd08e9985390ffc7bbc46a6d0ab51147`,
  and manifest-file SHA-256
  `c8c217f16b8ddad58f06a7f71c3b97a2a1536a898b4237e983ffd746fea9e4e0`.
- Verification: focused scoring/reducer/runtime coverage passes 40 tests plus
  Ruff/format/diff checks. The full scorer and materializer chain exited zero;
  no revision-1/revision-2 writer remains live.

### PRL-04-R1-QWEN3-INSTRUCT-GRPO-BS16-CROP-T1FULL-80STEP-GPU0123

- Cell/class and lifecycle: formal Crop RL pilot;
  `INVALID_REWARD_FAIL_CLOSED_STEP5` at `2026-08-02 05:01 JST`. It was
  preflighted at `03:02 JST` and ran from `04:21 JST` after RP-68, INT-DIAG and
  ACC-VAL passed their artifact gates. The controller was intentionally sent
  SIGINT after optimizer step 5 when an Instruct-output MCQ reward false
  negative was proven. This was not an OOM, timeout, Crop execution failure or
  identity failure. No R1 optimizer state may be reused.
- Identity: Qwen3-VL-8B-Instruct, Crop-only, T1-04 retained 25,393-row ArxivQA
  dataset, BS16, eight trajectories per prompt, world size 4 on GPUs 0--3,
  gradient accumulation 4, AdamW LR `1e-6`, cosine 80-step schedule.
- Cadence: no validation before training; checkpoints at steps
  `0/10/20/45/80`, five retained actor checkpoints, console + W&B logging,
  `resume_mode=auto`. The OpenRouter Qwen2.5-72B formal-pilot judge config is
  hash-bound to
  `81943918f09a7f99ec96b5c79fbc830fdea8e8376350f8de32ed87619cd3ce39`.
- Config:
  `configs/policy/runs/prl_04_r1_qwen3_instruct_grpo_bs16_crop_t1full_80step_gpu0123.toml`.
  Run identity SHA-256 is
  `5832d3323ddb0decf95f071d9f75db0ea1da5ee4de4803c9bb746399acceb30e`;
  W&B run ID is `s98vkf1k`.
- Runtime proof: optimizer steps 1--5 completed in respectively
  `353.670/332.387/362.234/337.996/351.263 s` (mean `347.510 s`) under the
  exact BS16/n8 contract. Across the first four complete rollout batches,
  1,337 of 1,339 executable Crop calls succeeded (`99.85%`); all 828 audited
  executed boxes were legal after the corrected coordinate mapping. The two
  failures were recoverable model-generated out-of-range boxes, while the 97
  call-cap errors were expected fifth-call attempts. Thus the historical Crop
  coordinate bug is not the cause of this stop.
- Reward failure: Qwen3-VL-Instruct frequently emitted an explanation followed
  by a standalone option letter on the final non-empty line. The v1 mixed MCQ
  parser examined the start of the whole response, so those correct decisions
  became `letter_parse_failed`; this zeroed both the `.8` answer component and
  the `1.2` conditional-tool component even though format already has its own
  `.2` term. A bounded representative sample showed 16/80 (`20%`) clear false
  negatives. The independent retained-audit gate over all material available
  at stop found 69 answer false negatives and 68 conditional-tool false
  negatives among 446 records; report SHA-256 is
  `11b6d8236aeeb3cacaef5d5be4abbeb33994bf8c0d6dc405034c234ef9392bae`.
- Recovery note: the per-step LoRA snapshots are rollout weight-publication
  artifacts, not paired training checkpoints. Since upstream invokes the save
  hook only after positive optimizer steps, configured step 0 does not publish
  a full checkpoint; scheduled `global_step_10` is the first durable recovery
  boundary carrying actor/optimizer/scheduler/data/RNG/project state. R1 was
  stopped before that boundary and has no reusable full checkpoint. Its output,
  metrics, trajectory audits and W&B run `s98vkf1k` are retained for debugging
  but are scientifically invalid as a Crop pilot result.

### PRL-04-R2-QWEN3-INSTRUCT-GRPO-BS16-CROP-T1FULL-MCQ-REWARD-FIX

- Cell/class and lifecycle: isolated reward-recovery smoke plus formal Crop RL
  pilot; `RUNNING_FORMAL_STEP1_DURABLE` at `2026-08-02 06:15 JST`, after
  `READY_TO_LAUNCH` at `05:28 JST`. Both outputs and the controller runtime
  root are fresh. R2 starts from the initial policy state; it neither resumes
  nor copies R1 optimizer state.
- Controlled variables: Qwen3-VL-8B-Instruct, native DeepStack, Crop-only
  protocol, complete T1-04 retained ArxivQA selection (25,393 rows), RP-66
  representation binding, BS16, n8, world4 on GPUs 0--3, GA4, decoder LoRA
  rank/alpha 64, AdamW LR `1e-6`, and reward weights `.8/.2/1.2` are unchanged
  from R1. The only scientific change is deterministic MCQ decision parsing:
  a canonical A--H decision at the start of the final non-empty line takes
  precedence over earlier explicit markers. Arbitrary prose remains rejected;
  MCQ judge fallback remains disabled.
- Implementation identity: commit
  `d835dd33fc4ec41db3be879195a62c2b4bb049d9`; mixed answer-verifier contract
  SHA-256
  `2d7b4da3add10fd6ca8f45aa146731634163a1b39bb4e7952a74b8e649248b97`.
  Focused controller/reward/config/audit coverage passed. Offline replay of R1
  recovers the known terminal-letter false negatives, while an independent
  parser gate still rejects the original R1 artifacts.
- Exact fail-closed chain: `crop_rl_smoke_1step ->
  crop_rl_mcq_terminal_reward_gate -> crop_rl_auto_resume_proof ->
  crop_rl_80step`. The controller accepts only this four-stage tuple or the
  original exact seven-stage tuple; it cannot silently insert or reorder a
  stage. Smoke uses T1-04 full data and the same BS16/n8 batch contract as the
  formal run. Promotion requires terminal metric step 1, a durable step-1
  checkpoint, exactly 16 retained representative rollout-zero records from
  behavior step 0, at least one terminal-letter match to gold, zero clear
  answer/tool reward false negatives, and an immutable auto-resume proof that
  no step 2 was produced.
- Smoke config:
  `configs/policy/runs/prl_04_r2_smoke_qwen3_instruct_grpo_bs16_crop_t1full_1step_gpu0123.toml`;
  config-file SHA-256
  `f26a32046f96859304dddf350c62b01d5dd469f45fe91ee5dc900fa708ca0af5`;
  run identity
  `eb66520679a5f2c9f7154bb3c1a83c78ee8c0a9fcb44083c41263cdc4589ac3a`.
- Formal config:
  `configs/policy/runs/prl_04_r2_qwen3_instruct_grpo_bs16_crop_t1full_80step_gpu0123.toml`;
  config-file SHA-256
  `e6133d6294f4df5f93a32107c1cbeb0c7ff5a018876e7a69206e9841857441b3`;
  run identity
  `cfea4582fcb66abbf446d70d13c1870b7091decb5703806cb1a954007a317ddf`.
  It has no pre-training validation, logs to console and W&B, checkpoints at
  `1/5/10/20/30/40/50/60/70/80` (maximum five retained), and permits one
  bounded 60-second-delayed controller retry after the separately proven
  `resume_mode=auto` path.
- Controller config:
  `configs/overnight/prl04_r2_crop_rl_fail_closed_20260802.json`; SHA-256
  `90bf15cf20db72982f6d033687da3e5448960436b4da513c06514909123297f0`;
  runtime root
  `artifacts/overnight/PRL04-R2-CROP-RL-FAIL-CLOSED-20260802`. Static validation
  passed, and both policy plans report `launch_ready=true`, no blockers and no
  GPU work launched during planning.
- Promotion evidence: the one-step smoke completed at `05:47 JST` with one
  metric record (SHA-256
  `4ad6eb5f7a9b48761929e06f11500e9b8490b392d34a23b12d6fb1f5d541806b`)
  and a durable step-1 checkpoint. The independent MCQ gate passed on 99
  retained audit records: all 16 representative records were parseable, 85 of
  88 parseable terminal decisions matched gold, and both answer and
  conditional-tool clear-false-negative counts were zero. The gate report
  SHA-256 is
  `a3fdb84b569ac193959b8285d1ead233696d7007fb5fa000c1044c11ae930e2f`.
  The subsequent no-extra-step auto-resume proof records metric SHA-256
  `4ad6eb5f...410806b`, checkpoint-tree SHA-256
  `9fe6e0a49b131e2e885b8cb8e5a4fd465fa7e5b5c8382321b3d1f59962fdfd29`,
  and `proven=true`.
- Formal runtime: the controller accepted all three prerequisite stages and
  launched the 80-step run at `05:59:07 JST`; W&B run ID is `wh7t9wv9`. It
  completed optimizer step 1 in `381.334 s` (`338.934 s` before publication,
  `27.655 s` checkpoint, `14.727 s` weight sync) and continued into step 2.
  The first step used 16 prompts and 128 trajectories, produced 31,882 policy
  tokens and 326 successful observations from 344 Crop attempts, and reported
  mean answer reward `.7500`, mean conditional-tool reward `.7265625`, format
  error rate `.140625`, and zero judge calls. All 18 tool errors were expected
  call-limit events. `global_step_1` is a complete 28-file, 18,603,763,356-byte
  recovery boundary with four model, optimizer and extra-state rank shards and
  tracker value `1`. No traceback, OOM or NCCL error was present through this
  gate. The step-1 metrics-file SHA-256 is
  `bd44c2bd78e44a5fddae719adae9d093bc6391e323343e2397ce96954d7da9c9`;
  measured completion ETA is approximately `14:00--14:40 JST`.

### PRL-05-R0-QWEN3-INSTRUCT-CROP-ANSWER-PRIMARY-TOOLW0P2-20STEP

- Cell/class and lifecycle: isolated non-formal reward-weight ablation;
  `READY_TO_LAUNCH` at `2026-08-03 21:17 JST`. It starts fresh from the same
  Qwen3-VL-8B-Instruct policy state as PRL-04-R2 and does not reuse an R2
  optimizer checkpoint.
- Single scientific variable: `conditional_tool_weight` changes from `1.2` to
  `0.2`. Answer/format weights remain `.8/.2`; data, seed, Crop protocol,
  BS16, n8, world4/GA4, decoder LoRA r64, AdamW LR `1e-6`, zero KL, prompts,
  response budget and maximum four tool calls remain unchanged. The optimizer
  stops at step 20 while retaining the original 80-step cosine scheduler
  horizon, so the first 20 learning-rate values remain matched to R2.
- Motivation: R2 completed 80 steps with cumulative answer reward `.759473`,
  conditional-tool reward `.748828`, training tool-attempt rate `98.525%` and
  mean `2.684` calls per trajectory. Held-out Crop tool use was effectively
  flat at step 0/30/80 (`86.21/86.03/86.61%`), while the seven-slice CoreDev
  macro was approximately `51.27/50.15/51.18`; R2 did not improve over step 0.
  The old successful Stage3 made answer correctness nominally dominant and its
  positive tool-decision scale was approximately one quarter of answer, rather
  than the current 1.5x answer bonus.
- Offline counterfactual caveat: replaying all 10,240 R2 trajectories under
  `.8/.2/.2` changes more than `1e-3` advantage on only 316/1,280 groups;
  old/new advantage cosine is `.989832`. Correct-tool and correct-direct
  coexist in only 73 groups. This run is therefore a low-cost mechanism test,
  not an expectation of a large accuracy gain.
- Implementation: commit `14f987fde23f92688d48d9b71556142b5b8501fe`
  adds an exact named answer-primary profile, binds configured weights into the
  live reward pipeline identity, and validates the same equation through the
  trajectory and DataProto bridges. The legacy `.8/.2/1.2` digest and behavior
  remain accepted unchanged. Focused coverage passes 81 tests plus Ruff.
- Config:
  `configs/policy/runs/prl_05_r0_qwen3_instruct_grpo_bs16_crop_t1full_toolw0p2_20step_gpu0123.toml`.
  Checkpoints are planned at `1/5/10/20`, W&B remains enabled, and the output
  root is
  `artifacts/policy/PRL-05-R0-qwen3-instruct-grpo-bs16-crop-t1full-toolw0p2-20step-gpu0123`.
- Gate: step 20 is sufficient to compare answer reward, tool attempts,
  repeated calls, format health, update/KL diagnostics and a matched held-out
  checkpoint evaluation. A flat result is not a definitive long-horizon
  accuracy failure, but no change in tool behavior or answer trajectory is a
  stop signal; only a healthy result is eligible for a separately bound
  continuation to step 40 and the RP-66 TGVF arm.
- Pre-launch scheduler recovery: the first initialization attempt exposed
  pinned veRL v0 overwriting the configured actor optimizer horizon (`80`)
  with the bounded trainer gate (`20`) inside `RayPPOTrainer.__init__`. It was
  stopped before rollout step 0, metrics, checkpoint or W&B creation. Commit
  `f22a0688ee5b741413eaed3f9ea585d71215150d` now captures the configured
  actor horizon before the upstream constructor and restores it before worker
  initialization; the trainer still performs exactly 20 optimizer steps while
  workers construct the matched 80-step cosine schedule. The focused trainer
  suite passed 16 tests, and the combined policy runtime/config/reward suite
  passed 96 tests plus Ruff. A second launch attempt then failed closed before
  GPU allocation because this recovery had not yet been recorded in the
  observed ledger commit; it likewise produced no scientific artifact. The
  next launch must be accepted only after rank 0 prints `Total steps: 80`.

### PRL-06-R0-QWEN3-INSTRUCT-CROP-BS32-TOOLW0P2-20STEP

- Cell/class and lifecycle: isolated non-formal batch-size stability ablation;
  `READY_TO_LAUNCH` at `2026-08-04 01:35 JST`. It starts fresh from the same
  Qwen3-VL-8B-Instruct policy, T1-04 retained data, RP-66 adapter and Crop
  protocol as PRL-05.
- Single training variable: global prompt batch changes from 16 to 32 and the
  mechanically corresponding accumulation changes from GA4 to GA8. Rollout
  multiplicity remains n8, world size remains four, per-rank prompt and actor
  micro-batches remain one, PPO epochs remain one, and LR remains `1e-6`.
  Reward weights `.8/.2/.2`, seed, data order, LoRA, maximum response length,
  maximum tool calls, zero-KL setting and the 80-step cosine horizon are
  unchanged.
- Interpretation: step 10 consumes 320 prompts / 2,560 trajectories and is the
  sample-budget-matched comparison to PRL-05 step 20. Step 20 consumes 640 /
  5,120 and is the update-count-matched comparison; any difference there mixes
  batch size with twice the sample exposure. Both checkpoints must receive the
  same seven-slice CoreDev-2511 Crop ACC-VAL before promotion or rejection.
- Motivation and prior: PRL-05 held-out macro fell from the common step-0
  baseline `51.264974%` to `49.840874%` while tool use stayed near 87%. Its
  batches also contained many zero-advantage groups and sparse direct/tool
  contrasts, so BS32 may reduce gradient-sampling variance. This is a bounded
  stability test, not evidence that BS16 was inherently small: the successful
  legacy Stage3 used only eight prompt groups / 64 rollouts per update, versus
  PRL-05's 16 / 128.
- Config:
  `configs/policy/runs/prl_06_r0_qwen3_instruct_grpo_bs32_crop_t1full_toolw0p2_20step_gpu0123.toml`.
  It runs on GPU0-3, checkpoints at `1/5/10/20`, enables W&B, disables all
  in-training validation, and launches external evaluation only after the
  requested checkpoint is durably saved.

### PRL-07-QWEN3-INSTRUCT-CROP-BS16-CORRECT-FINAL-T1-MIXED

- Cell/class and lifecycle: correction rerun of the PRL-05 small-batch Crop
  pilot. A separate one-step smoke (`R0`) must pass before the fresh 20-step
  training run (`R1`) is launched on GPU0-3.
- Corrected data identity: the run consumes all `79,069` final T1 retains from
  the completed all-source pool, with VStar `39,205`, ArxivQA `25,393`, and
  ThinkLite `14,471`. It replaces the incomplete provisional ArxivQA-only
  `25,393` pool used by PRL-05/06. T2 and post-T1 balancing are not applied.
- Controlled training recipe: fresh Qwen3-VL-8B-Instruct, Crop only, BS16,
  n8, world4/GA4, decoder LoRA r64, AdamW LR `1e-6`, reward weights
  `.8/.2/.2`, zero KL, 20 updates, and the original 80-step cosine horizon are
  retained from PRL-05. Checkpoints are planned at `1/5/10/20`; W&B is on and
  in-training validation is off.
- Runtime gates: commit `2b5c258cdffa21439483a4e04109cfd2c94d6afc`
  adds a separate fail-closed mixed-data loader without changing the old
  ArxivQA loader. Focused and regression coverage passed 57 tests; the real
  artifact load verified all 79,069 rows, source/task routing, deterministic
  order, and 65,705 unique image byte hashes. The first 16 prompts contain
  VStar 8 / ArxivQA 5 / ThinkLite 3. The formal OpenRouter Qwen2.5-72B judge
  route passed a live MCQ/math/open health check before launch.
- Configs:
  `configs/policy/runs/prl_07_r0_smoke_qwen3_instruct_grpo_bs16_crop_t1mixed_final_1step_gpu0123.toml`
  and
  `configs/policy/runs/prl_07_r1_qwen3_instruct_grpo_bs16_crop_t1mixed_final_toolw0p2_20step_gpu0123.toml`.

### PRL-08-QWEN3-INSTRUCT-CROP-BS16-CORRECTED-T1-V2

- Cell/class and lifecycle: corrected-data rerun of the answer-primary Crop
  reward pilot. `R0` is a fresh one-step smoke on GPU0-3; only a successful
  smoke is eligible to start the separate fresh 20-step `R1` automatically.
- Data correction: both arms bind the finalized `mixed-v2` pool of `77,541`
  retained T1 samples: VStar `39,205`, ArxivQA `25,393`, and corrected
  ThinkLite `12,943`. The exact manifest/content/samples/iteration identities
  are pinned in each run config. This supersedes PRL-07's erroneous mixed-v1
  pool of `79,069` samples.
- Controlled recipe: Qwen3-VL-8B-Instruct, RP-66, Crop only, BS16, n8,
  world4/GA4, AdamW LR `1e-6`, reward weights `.8/.2/.2`, zero KL, and the
  original 80-step cosine horizon are unchanged. R1 stops at update 20 and
  retains checkpoints at `1/5/10/20`; W&B is enabled and in-training
  validation is disabled.
- Recovery boundary: implementation commit
  `ef0c495d82b0b6ce4dc1c666b27909f33f045378` adds the corrected ThinkLite
  task classifier and isolates only malformed/nonbinary completed judge
  responses to the current sample with answer reward zero and an auditable
  route. Credential, HTTP, transport, response-model, configuration and
  identity failures remain fail-closed. The historical v1-v3 judge bindings
  retain their abort behavior; PRL-08 alone binds judge-v4 SHA-256
  `1ec38f640f943702ad812dc367fc66edf843a663a1c1048ebb39a0d25fac18a9`.
- Configs:
  `configs/policy/runs/prl_08_r0_smoke_qwen3_instruct_grpo_bs16_crop_t1mixed_v2_1step_gpu0123.toml`
  and
  `configs/policy/runs/prl_08_r1_qwen3_instruct_grpo_bs16_crop_t1mixed_v2_toolw0p2_20step_gpu0123.toml`.
- Runtime result through step 20: `R0` and `R1` both completed and passed their
  artifact gates.  The R1 ledger contains exactly 20 contiguous optimizer
  records, 320 prompts and 2,560 trajectories.  Its cumulative answer reward
  is `.651953125`, conditional-tool reward is `.635546875`, tool-attempt rate
  is `.967578125`, and mean calls per trajectory is `2.12734375`; no isolated
  malformed/nonbinary judge response occurred.  The paired step-20 checkpoint
  is complete on all four ranks and records the next data cursor after exactly
  320 prompts.  W&B runs are `75sggp5u` (smoke) and `sz017s3n` (R1).
- Horizon-extension decision at `2026-08-05 JST`: the user authorized this
  same R1 lineage to continue from optimizer step 20 to step 80.  This is not a
  fresh arm and must never restart from step 0.  The original TOML, run ID,
  output root and checkpoint identity remain byte-for-byte unchanged; its
  actor scheduler was already constructed with the 80-step cosine horizon.
  A separate integrity-bound `policy-horizon-extension-v1` manifest is the
  only permitted operational override.  It may change only the trainer stop
  boundary to 80 and merge checkpoint boundaries `30/40/60/80` after the
  existing `1/5/10/20` plan.  It binds the original 20-row metrics prefix,
  source paired checkpoint/project state, latest LoRA identity and recovery
  weights, and validates the current exact checkpoint on every restart.
- Continuation acceptance: before GPU work, the tracker, metrics, paired
  checkpoint, four-rank shards and latest LoRA pointer must agree on the same
  completed step.  The first new metric must be step 21; any from-zero path is
  rejected.  At completion the metrics must be exactly steps 1--80 and the
  paired `global_step_80` checkpoint must be durable before external CoreDev
  ACC-VAL starts.  With five retained actor checkpoints, the intended final
  recovery set is `20/30/40/60/80`.
- The committed authorization manifest is
  `configs/policy/continuations/prl_08_r1_step20_to80.json`; it binds recovery
  implementation commit `055bf3a6f31cc9f53a0d2fef8f2b1b74e4314222` and
  the exact accepted step-20 artifact hashes.  Ordinary `run-policy` remains
  unchanged; this continuation is reachable only through the explicit
  `--horizon-extension` entry.

### PRL-09-QWEN3-INSTRUCT-TGVF-SHAPED-REWARD

- Cell/class and lifecycle: an isolated non-formal TGVF reward experiment.
  It borrows the five-part answer/tool/focus/grounding/protocol shaping idea
  from the previously successful project, but it is not a reproduction or
  continuation of that old Stage3 run.  `R1` is the required fresh one-step
  full-pipeline smoke on GPU4-7; only an accepted R1 is eligible for the
  separate fresh 80-step `R2` run on GPU0-3.
- Controlled model/data recipe: Qwen3-VL-8B-Instruct policy, frozen RP-66
  representation adapter, corrected mixed-v2 T1 retained pool, BS16, n8,
  world4/GA4, decoder LoRA r64, AdamW LR `1e-6`, zero KL and an 80-step cosine
  scheduler.  TGVF is limited to one successful call attempt.  In-training
  validation is disabled; the R2 step-80 checkpoint must trigger the external
  strict CoreDev-2511 ACC-VAL automatically.
- Reward equation: `2*A_gated + T + F + G + P`.  Tool necessity is bound to a
  deterministic, sample-local counterfactual utility sidecar.  Focus and
  grounding are scored by a gold-free local Qwen3-VL-32B-Thinking visual
  judge over the original image, question, tool target, post-tool reasoning
  and final answer.  The answer verifier remains rule-first with the pinned
  Qwen2.5-72B semantic fallback.  The internal profile spelling
  `stage3-shaped-v1` is retained only as a compatibility identifier; all run,
  artifact and W&B identities are independently named TGVF-shaped.
- Isolated implementation commit:
  `56bde7ea8d9865fc284913f1c564cfaacf96c86d`.  The focused runtime suite passed
  172 tests, and a clean closure audit passed 176 tests.  The launch plan binds
  the shaped reward FQN rather than the legacy Pilot reward FQN.
- Local visual judge gate: the exact production provider served
  Qwen3-VL-32B-Thinking locally on port 8013 and passed repeated real-image
  strict-JSON canaries.  It returned exactly `focus_score` and
  `grounding_score`, both `2`, with no parsing/thinking leakage.  The first
  request took about 10.8 seconds and a warm request about 0.36 seconds.
  Config-file SHA-256 is
  `d05a0e91554eaef7f700732d78d392b9ebec04be1012104af3a6d27d9e3be331`.
- R1 utility prefix: 16 sequential training samples, eight attempts each,
  with labels needed/optional/unnecessary = `3/10/3`.  Sidecar SHA-256 is
  `1f6870500d0bc5b9ad63e48b6aced1ff5f64acfd44ddc0f6445fdbe517f143ce`;
  manifest identity SHA-256 is
  `8a3ef0318c7dadeb74a259fc2c4054552019b53c49e6a169385635b27b2f6119`.
- R1 config:
  `configs/policy/runs/prl_09_r1_smoke_qwen3_instruct_grpo_bs16_tgvf_t1mixed_v2_shaped_1step_gpu4567.toml`.
  Acceptance requires one optimizer update over exactly 16 prompts / 128
  trajectories, at least one successful TGVF observation, positive visual
  judge applicability, complete visual coverage with zero visual failures,
  finite five-component metrics and a complete four-rank step-1 checkpoint.

### PRL-15-R0-QWEN3-INSTRUCT-FULL-RP66-MATCHED

- Lifecycle correction (2026-08-09): the first implementation incorrectly
  treated world4/micro1/GA4 as the control. That shape was derived from the
  TGVF plan rather than the completed Crop experiment and is scientifically
  invalid for the requested comparison. No formal run was started. The
  one-step `actor-rollout-only-v1` GPU0-3 closure is retained only as
  engineering evidence that joint Qwen/RP66 update and recovery execute.
- Correct control: the external source of truth is the completed PRL14 Crop-16
  record, SHA-256 `3907b310...f70f1`, and its permanent step-8 checkpoint.
  PRL15 now matches world8, actor/log-prob micro32, BS16, n16 (256
  trajectories), full Qwen vision/language training, one PPO epoch, constant
  LR `1e-6`, token/capacity settings, FlexAttention/SDPA, FSDP compile/reshard,
  DeepEyes token-mean loss, zero KL, and the official source-aware reward.
- Intended treatment: native Crop prompt/tool/observation is replaced by the
  matched TGVF prompt/tool and jointly trainable RP66 state. Dataset/AgentLoop,
  weight-publication and paired-checkpoint differences are permitted only as
  the plumbing needed to realize that treatment.
- Control direction: `configs/policy/controls/prl15_crop_rp66_matched.json`
  checks PRL15 against the hash-verified PRL14 record; it never copies TGVF
  settings into a synthesized Crop run. The audit entry point is
  `tools/audit_prl15_against_prl14_crop16.py`, and relaunching Crop from that
  tool is forbidden.
- Reward identity: both arms use `.8/.2/1.2` on visual samples and
  `1.2/.4/0` on ThinkLite, with the same Qwen2.5-72B judge config SHA
  `fff705c5...3021`. TGVF uses an asynchronous trajectory adapter around the
  same official answer extractors, judge prompts and failure semantics.
- Current gate: the corrected formal plan composes and the matched core fields
  pass CPU tests. A new world8/micro32 one-step GPU smoke is still required;
  until it passes, formal eight-step training is not authorized.
- Runtime isolation: smoke uses console-only logging and an isolated
  `output.root/smoke/<smoke-id>/` closure; formal metrics/W&B remain untouched.
- External ACC-VAL: step0 and step8 are paired states, respectively base Qwen
  plus stage1 RP66 and step8 Qwen plus step8 RP66. The dedicated
  `full_model_trainable_rp66` backend binds both members, requires the RP66
  vLLM update ACK, and writes their combined identity into every trajectory.
  `tools/run_prl15_paired_evaluation.py --wait-for-step8` can be started before
  training and will automatically prepare, resume, run and score both arms.
- Detailed protocol and commands:
  `docs/experiments/prl15_controlled_rl_preflight.md`.
- Historical engineering smoke evidence: 16 prompts produced 256 trajectories and 194
  successful TGVF observations, with answer reward `0.6328125`, conditional
  tool reward `0.453125`, tool-attempt rate `0.75`, and format-error rate
  `0.0078125`. The Qwen update had `actor/grad_norm=7.09375`; RP66 changed from
  step-0 state `05778a43...0318` to step-1 state `697d2a27...f25`; the four
  Qwen model shards, four optimizer shards, project state and Qwen/RP66 pair
  receipt were all saved. End-to-end update time was `689.29 s`, including
  `40.95 s` checkpointing and `6.46 s` RP66 publication. These measurements
  must not be used to estimate or authorize the corrected world8/micro32 run.
- Isolation evidence: this smoke wrote only below
  `output.root/smoke/actor-rollout-only-v1`, used console logging, created no
  W&B run, did not create formal `output.root/metrics.jsonl`, and did not
  modify the pre-existing legacy `output.root/smoke/metrics.jsonl`.
- Matched R1 gate (2026-08-10): the corrected world4 mathematical-equivalent
  execution of the Crop world8/micro32 control passed one complete optimizer
  step over BS16 x n16. It produced 256 trajectories and 192 successful TGVF
  observations, changed both the Qwen and RP66 states, and completed in
  `809.16 s` including a `65.16 s` paired checkpoint. This is the accepted
  scientific one-step gate, not the low-cost functionality canary.
- Formal recovery (2026-08-10): formal step 1 completed and was durably saved;
  step 2 then stopped before its optimizer update because pinned veRL had
  collapsed a normal vLLM native-EOS termination into `completed` while
  discarding the raw finish/stop pair. The project-owned vLLM server boundary
  now transports the exact pair and the live termination contract accepts
  vLLM's documented `stop + null` native-EOS outcome. It does not infer or
  forgive aborted, unknown, or malformed generations. Focused transport,
  recovery, and sampler tests cover native EOS, length, and tool-string stops.
- Checkpoint recovery correction (2026-08-10): the remaining formal horizon
  physically saves every completed optimizer step, retains the newest two
  exact paired checkpoints across restarts, and hard-links step 8 into a
  separately validated permanent checkpoint. The original run TOML and its
  run identity remain unchanged: its `0/1/4/8` list continues to identify the
  scientific milestone gates, while the formal runtime owns the denser
  recovery-save lifecycle. The corrected process therefore resumes from the
  already committed step-1 Qwen/RP66 pair rather than replaying step 1.
- Low-cost functionality gate (2026-08-10): before the formal resume, a
  console-only canary used 4 prompts x 2 trajectories and one real joint
  Qwen/RP66 optimizer update. It completed without termination/replay errors,
  produced 6 successful TGVF observations, and saved a complete paired
  checkpoint. The update itself took `164.28 s`; cold startup plus the update
  was about six minutes. This canary is engineering evidence only and is not
  a matched scientific result.
- Formal steps 2--3 and recovery (2026-08-10): the formal run resumed the
  exact step-1 model, optimizer, scheduler, RNG and data cursor under the same
  W&B run `ja23wygu`. Step 2 re-executed the batch that had previously exposed
  the hidden native-EOS bug and completed without any termination or tool
  error. Step 3 also completed. Their respective end-to-end times were
  `934.47 s` and `942.33 s`; answer rewards were `.6875` and `.71875`, and
  conditional-tool rewards were `.5234375` and `.546875`.
- Runtime checkpoint schedule correction (2026-08-10): step 2 revealed that
  the trainer save gate still read the identity-preserving TOML milestone list
  `0/1/4/8`, even though the formal lifecycle owned the recovery schedule
  `0..8`. The gate now consumes the same runtime lifecycle schedule as
  prepare/finalize/rotation. Failure recovery durably saved the last complete
  boundary as paired `global_step_3` with tracker `3`; no step-4 optimizer
  mutation occurred.
- Tool-attempt metrics correction (2026-08-10): the next step-4 batch exposed
  a metrics-only legacy bound of four admitted calls/fifth cap attempt. PRL15's
  locked live protocol permits six executed calls and records a seventh call
  only as the unexecuted cap error. Metrics validation now receives that bound
  from the run config, preserves all counts without clamping or dropping
  trajectories, and retains conservation, cap position/count, payload identity
  and replay-integrity checks. Legacy Pilot `M=4` and Stage3-shaped `M=1`
  remain separately covered. The v1 checkpoint mapping and the existing
  step-3 project-state bytes are unchanged.
- Formal step 4 (2026-08-10): exact recovery from step 3 completed the full
  matched batch in `1348.86 s`, including `58.82 s` checkpointing and `9.41 s`
  RP66 weight publication. The batch produced answer reward `.7734375`,
  conditional-tool reward `.578125`, format-error rate `.03515625`, 213
  successful observations, and exactly one legal unexecuted seventh cap
  attempt. The complete four-rank Qwen/RP66 pair is committed at step 4; the
  tracker is `4` and the rolling generations are steps 3 and 4.
- Policy visual-control containment (2026-08-10): step 5 stopped before its
  optimizer update because one policy response sampled Qwen's reserved
  `<|vision_start|>` token without a native image placeholder body. The old
  layout scanner misclassified every such policy-owned token as an
  environment image and killed the whole batch. The corrected agent loop
  retains the exact action IDs and behavior logprobs, marks only that
  trajectory invalid-format, and excludes it from further tool turns. Exact
  replay identifies visual content solely through recorded source/tool-image
  positions; its temporary M-RoPE view treats only the policy-owned opener as
  text, while actual replay IDs remain byte-for-byte unchanged. Recorded
  source and tool visual blocks remain strict, so genuine replay corruption is
  still fatal. The official reward extraction and weights are unchanged.
- Recovery gate for that correction: a low-cost targeted replay milestone ran
  before another matched batch. It reproduces the failure with Qwen's real
  visual token IDs and covers all three policy-owned visual controls through
  the current and frozen-reference injected-forward paths. The active tree
  passed 64 focused CPU/integration tests plus Ruff and diff checks. A formal
  preflight then re-read the exact step-3/step-4 committed pairs, their metrics
  and policy steps, the step-4 tracker, and the runtime `0..8` save schedule.
  No step-5 optimizer mutation occurred in the failed process; formal resume
  therefore starts at the exact step-4 boundary.
- Formal step 5 and native EOS correction (2026-08-10): the corrected visual
  containment path completed the next full matched batch and its paired
  checkpoint in `874.23 s`, including `63.21 s` checkpointing and `8.18 s`
  RP66 publication. Tracker `5`, project/metrics/policy step `5`, and rolling
  generations 4/5 agree. The following step-6 rollout then stopped before any
  optimizer update because the termination contract omitted Qwen3-VL's
  secondary native EOS token `151643` (`<|endoftext|>`). The request's explicit
  extra stop remains `151645`, but the pinned model generation config (SHA-256
  `8469742d...fc4d44`) declares effective native EOS IDs `151645/151643`, as
  did the preceding T1 selection contract.
- The training and CoreDev evaluator now share one exact Qwen3-VL final-turn
  outcome builder: both native EOS IDs, vLLM's hidden-native-EOS `stop/null`,
  and an exact length stop are accepted; an unrelated token such as `151644`
  remains rejected. Future mismatch errors include the exact finish/stop pair
  without response text. A CPU fixture reproduced the step-6 failure with
  `stop/151643` and verified the corrected behavior; 62 focused tests, Ruff,
  and diff checks pass. The run TOML and reward are unchanged, and preflight
  revalidated the exact step-4/step-5 recovery pair before resume.
- Formal step 6 (2026-08-10): exact recovery from step 5 replayed the batch
  that had exposed the secondary-EOS omission and completed it without a
  termination error. The update took `1100.39 s`, with answer reward
  `.5390625`, conditional-tool reward `.40625`, format-error rate `.03125`,
  and 192 successful TGVF observations. Its four-rank paired checkpoint is
  complete; tracker, project state and metrics all remain at the safe step-6
  boundary.
- Step-7 memory diagnosis (2026-08-10): the next batch stopped before its
  optimizer mutation in exact replay. This was not allocator fragmentation:
  a 32-row actor microbatch called `DTensor.full_tensor()` independently for
  every row, and every fused-logprob autograd context retained its own full
  BF16 LM-head copy until the single microbatch backward. Up to 32 copies
  (about `37.1 GiB`) accumulated on top of the row graphs; the long step-7
  batch reached 169--171 GiB allocated and then failed on the next 2.32-GiB
  FP32 gather. The unknown-commit guard correctly refused a step-7 checkpoint,
  so no partial optimizer state was admitted.
- Low-cost memory correction gate: a microbatch-scoped fused materializer now
  gathers/casts the LM head once and shares that differentiable tensor across
  all replay rows. A new materializer is created for every actor/reference
  microbatch and cannot cross an LM head, device or dtype boundary; it never
  survives an optimizer step. This preserves the intended real-valued loss
  and gradient sum and matches Crop's one-shared-weight-per-microbatch design.
  It intentionally does not claim bitwise equivalence with the buggy per-row
  BF16-cast accumulation: finite-precision LM-head gradients can differ in
  rounding order. CPU value/gradient tests pass, and a two-rank real-DTensor
  test completes in seconds with exactly one `full_tensor()` call per rank and
  local-shard gradient error below `2.4e-7`. The formal batch must not be used
  as the next debugger: a separate 4-prompt x 2-trajectory functional canary
  is required before resuming from step 6.
- LM-head-cache functional canary (2026-08-10): commit `8a6dd1b` completed an
  isolated console-only 4-prompt x 2-trajectory joint Qwen/RP66 step. It
  exercised rollout, exact replay, backward, optimizer, weight publication
  and all four checkpoint shards without the repeated-head OOM; peak actor
  allocation was `41.24 GiB` and end-to-end time was `175.69 s`. This sampled
  group had zero policy advantage (`pg_loss=0`, `grad_norm=0`), so it is only
  a code/lifecycle milestone; nonzero value and gradient equivalence is owned
  by the CPU and two-rank DTensor tests, and no learning claim is made from
  this canary.
- PRL16 F2 live-sync lifecycle correction (2026-08-11): F1 isolated its first
  same-process boundary as the failure: step 1 was healthy from a fresh
  process, while step 2 produced 90.23% format errors, zero observations and
  12,133.6 mean response tokens immediately after the first save/sync cycle.
  The project checkpoint bridge had synchronized and awakened the new Qwen +
  RP66 state, then performed another full-model vLLM level-2 sleep for the
  paired checkpoint and used a bare wake. Level-2 sleep discards the newly
  published full weights, so the custom TGVF runtime did not have a valid
  current-policy restore boundary. Commit `2dbdf85` makes only the trainable
  TGVF manager publish the same current-step Qwen + RP66 state again after the
  checkpoint sleep; the Crop lifecycle remains unchanged. F2 is a clean,
  separate output identity and will run a bounded two-continuous-step formal
  diagnostic before any longer scientific continuation.
- F2 result (2026-08-11): the exact world8, BS16 x n16 matched diagnostic
  completed two steps in one process and exited normally. Step 1/2 respectively
  produced answer reward `.48046875/.47265625`, format-error rate
  `.03125/.01953125`, `193/192` successful TGVF observations, and response
  length mean `575.15/736.59`. Step 2 therefore no longer reproduces F1's
  answer `0`, format error `.90234375`, zero observations, or response mean
  `12133.62`. The step-1 and step-2 state directories each contain two
  request-distinct RP66 manifests, proving that the post-checkpoint corrective
  publication ran at both boundaries. The paired step-2 checkpoint, all eight
  model/optimizer shards, project state, pair marker and tracker `2` are
  complete. Peak actor allocation at step 2 was `61.23 GiB`, not F1's
  `167.39 GiB`; all GPU processes shut down cleanly after completion. This
  confirms the checkpoint level-2 sleep/bare-wake boundary as the cause of the
  observed same-process live-sync collapse and validates commit `2dbdf85` as
  the scoped runtime fix.

### PRL-17-R1-QWEN3-INSTRUCT-FULL-FROZEN-RP67-SHAPED

- Planned 2026-08-12 as the representation-only follow-up to the successful
  PRL17-R0 frozen-RP66 shaped-reward pilot. The sole scientific treatment is
  RP66 step 2000 to RP67 step 2000; RP67 remains frozen during RL.
- RP67 artifact SHA-256 is `13332865...f0f68`, manifest SHA-256 is
  `2ea09896...33b1`, semantic state SHA-256 is `f223d1f0...0256`, and frozen
  runtime storage SHA-256 is `3f60f365...53a14`. RP66 and RP67 have identical
  104-tensor, 72,055,808-parameter BF16 Adapter structures.
- Model, T1 schedule, prompt, BS16 x n16, world8/micro2/GA1, constant `1e-6`
  learning rate, eight-step horizon, answer judge, checkpoint lifecycle, and
  executed reward `2*A_gated + T + P` are held fixed against R0. Focus and
  Grounding remain disabled.
- The RP66-derived 128-row utility sidecar is intentionally fixed. Recomputing
  RP67-dependent labels would introduce a second reward variable and belongs
  in a later experiment.
- Required sequence: console-only 4-prompt x 2-trajectory functional canary;
  then the tmux-owned eight-GPU formal run; then automatic paired step0/step8
  CoreDev-2511 evaluation. Detailed protocol:
  `docs/experiments/prl17_r1_rp67_shaped_reward_control.md`.
- Canary attempt 1 stopped before any optimizer mutation because the generic
  source-covering smoke split is intentionally disjoint from formal training,
  while the shaped reward sidecar intentionally covers only the 128 formal
  rows consumed by eight steps. The corrected shaped-reward canary uses the
  exact first four formal rows and validates all four immutable utility labels
  before Ray or GPU initialization. Missing labels remain fatal; no default or
  fallback label was introduced. Formal training data and reward are unchanged.
- Corrected canary result (2026-08-12): the exact four labeled prefix rows
  produced eight trajectories and six successful TGVF observations. The one
  optimizer step completed in `131.51 s` with total reward mean `1.375`,
  policy loss `0.01898`, gradient norm `3.7118`, and no judge transport error.
  All four model/optimizer/extra-state shards and the frozen-RP67 step-1 pair
  were saved, the tracker is `1`, all GPUs shut down cleanly, and no W&B run
  was created. This is an engineering gate only; formal R1 is now authorized.

### PRL-17-R2-QWEN3-INSTRUCT-FULL-FROZEN-RP67-TFREE

- Planned 2026-08-12 as the reward-only follow-up to PRL17-R1. RP67 remains
  frozen and model, retained T1 schedule, prompt, BS16 x n16, world8/micro2,
  constant `1e-6` learning rate, eight-step horizon, answer judge, seeds and
  checkpoint lifecycle remain matched to R1.
- The sole scientific treatment removes counterfactual tool-utility `T`: both
  its needed/optional/unnecessary decision score and its needed-without-tool
  answer gate are disabled. The executed no-visual equation is
  `2*A - 0.05*max(0, tool_calls-1) + P`, where `P=-1` for any protocol or
  tool-execution error. Focus and Grounding remain implemented but disabled.
- This is a real sidecar-free treatment: the RP66 or partial RP67 utility
  sidecars are neither loaded nor included in run/checkpoint identity. The
  historical T-enabled schema and runtime remain backward compatible.
- Required sequence: console-only 4-prompt x 2-trajectory functional canary;
  then the tmux-owned eight-GPU formal run; then automatic paired step0/step8
  CoreDev-2511 evaluation. Smoke outputs are isolated and never sent to W&B.

### PRL-18-R0-QWEN3-INSTRUCT-FULL-JOINT-RP67-TFREE

- Authorized 2026-08-13 as the trainable-Adapter control following the
  completed PRL17-R2 paired Step 0/8/16 result. It starts fresh from the same
  Qwen3-VL-8B-Instruct and RP67 Stage1 artifacts; it does not resume the
  frozen PRL17 model.
- The sole scientific treatment is
  `representation.adapter_update_mode=frozen_adapter -> joint`. Consequently
  the 72,055,808 RP67-owned parameters enter the same AdamW optimizer at
  `1e-6` and are republished to every rollout worker after each optimizer
  step. Full Qwen parameters, including the visual path, remain trainable as
  in PRL17-R2.
- Data, order and seed, TGVF prompt/schema, BS16 x n16, world8/micro2/GA1,
  `temperature=1`, constant LR, maximum response/tool-call limits and T-free
  reward are fixed. The executed reward remains answer correctness plus
  protocol and repeated-call penalties; tool utility `T`, focus and grounding
  remain disabled. Operational run/output/provenance paths and permanent-copy
  retention are separately named and are not scientific variables.
- Lifecycle: a console-only exact world8/BS16/n16 one-step smoke must prove a
  changed RP67 state and complete paired checkpoint. The formal run then stops
  at Step 8, binds its exact model/optimizer/scheduler/data/RNG/Adapter state
  into a runtime continuation manifest, and resumes in place to Step 16.
  Step 8 and Step 16 are permanent; rolling recovery checkpoints are saved
  every step. The outer supervisor is tmux-owned and retries only recoverable
  process/API interruptions.
- Evaluation: after Step 16 closes, evaluate only Step 8 and Step 16 in
  parallel on eight GPUs. Both arms use the frozen CoreDev-2511 headline
  contract and the same paired RNG namespace as PRL17-R2:
  `coredev2511-official-v1/rp67-tfree/step0-step8-step16/temp1/seed42/v1`.
  Step 0 is intentionally not rerun; the already measured PRL17-R2 paired
  Step 0 is the common initialization reference.
- Config:
  `configs/policy/runs/prl_18_r0_qwen3_instruct_full_joint_rp67_bs16_n16_tfree_novisual_8step_ws8.toml`.
  W&B is enabled only for formal training under run ID `prl18r0u`; smoke is
  console-only. Expected wall time is roughly 2 h 45 min--3 h for 16 training
  steps plus 50--65 min for the two-arm evaluation.

### PRL-19-R0-QWEN3-INSTRUCT-FULL-FROZEN-RP67-TFREE-VISUAL-API

- Authorized 2026-08-13 as the visual-reward treatment against the frozen
  PRL17-R2 RP67 T-free control. Qwen3-VL-8B-Instruct, the RP67 Step-2000
  Adapter, frozen-Adapter mode, retained T1 data/order, policy prompt and tool
  protocol, BS16 x n16, world8/micro2/GA1, temperature 1, constant `1e-6`
  learning rate, answer judge and all rollout limits remain fixed.
- The only scientific treatment enables Focus/Target and Grounding. Tool
  utility `T` remains disabled. The executed scalar is
  `2*A - 0.05*max(0, tool_calls-1) + F + G + P`, with the existing F mapping
  `2/1/0 -> 1/0.5/0`, G mapping `2/1/0 -> 1/0.5/-1`, and protocol/tool errors
  contributing `P=-1`.
- One gold-free OpenRouter request to pinned
  `qwen/qwen3-vl-32b-instruct` returns both F and G. The request contains the
  original image, question, all successful tool targets in call order,
  post-tool reasoning and final answer; it cannot contain any gold/reference
  answer. The run-global visual concurrency is 16 (two permits per AgentLoop
  worker), with four bounded attempts, exponential backoff, exact-request
  cache, explicit coverage/failure accounting and a 25% rolling provider
  failure circuit breaker.
- A real-image API canary passed exact model/schema/usage checks and separated
  a specific valid target and grounded statement `(F=2,G=2)` from an
  irrelevant target/hallucinated statement and an answer shortcut
  `(F=0,G=0)`. Observed API latency was about 1.1--2.1 seconds and cost about
  `$0.000085` per combined judgement.
- The first world4 BS4 x n2 functional training canary completed one optimizer
  step in 150.97 seconds with frozen RP67 SHA
  `3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14`,
  finite `grad_norm=6.98`, and both F/G components present. It exposed one
  malformed visual-judge completion among six applicable trajectories
  (5/6 covered). Runtime commit
  `1d193dce846d5f2eeb78499bec7bad1d873f6a96` therefore gives every malformed
  completion the same four-attempt bounded retry budget as transient transport
  failures; a persistent served-model identity mismatch remains a global
  failure. The exact matched smoke accepts sample-local degradation only when
  aggregate visual coverage is at least 99%, rather than aborting on one bad
  provider response.
- Runtime commit `9e3c6f74a44762b90e3a138f440246b1df5dff8c` also makes the
  multi-call judgement non-overlapping: F receives every ordered successful
  target, while G receives only assistant reasoning after the final successful
  tool call. Later tool-call JSON/target text is therefore not duplicated into
  the grounding channel.
- Launch descendant `f22650e2bf6b2d849756a2be9fbc2290a675248c`
  removes the now-dead local left by the malformed-output retry recovery. This
  is a behavior-preserving static cleanup; the configured executable identity
  remains the preceding `9e3c6f7` runtime commit.
- Lifecycle: first run a console-only BS4 x n2 world4 functional canary, then
  one exact world8/BS16/n16 matched smoke. After both gates pass, a tmux-owned
  fresh formal run proceeds Step 0 -> 8, binds an immutable horizon extension,
  and automatically continues Step 8 -> 16. The RP67 Adapter must remain
  byte-identical; Step 8 and Step 16 are permanent checkpoints.
- Evaluation starts automatically after Step 16 and runs only Step 8 and
  Step 16 under the same paired CoreDev-2511 seven-suite protocol and RNG
  namespace as PRL17-R2. The already measured paired PRL17-R2 Step 0 is the
  common initialization reference and is not rerun.
- Formal training completed all 16 optimizer steps on 2026-08-13. The first
  paired-evaluation attempt durably completed 1,953/4,480 single-image rows
  (Step 8: 1,015/2,240; Step 16: 938/2,240) before one Step-16 sample emitted
  text after a complete `</tool_call>`. The sampler correctly rejected that
  response, but the benchmark worker incorrectly promoted this model-output
  event to an eight-worker failure; seven orphan vLLM process groups then held
  the supervisor log pipe and GPUs open.
- Recovery runtime commit `22b2c28` introduces a narrow, structured
  `PolicyOutputContractError` only for this illegal suffix. The benchmark now
  persists that task as an identity-bound `sample_local_failure` with null
  answer and `invalid_format`, retains it in the scoring denominator as
  incorrect, and continues its siblings. Generic replay, identity, transport,
  artifact and unknown failures remain fail-closed. Evaluator ranks now launch
  in isolated process groups and a worker-level failure drains every vLLM
  descendant before the supervisor retries. Seventy-three focused CPU tests
  passed; all 1,953 prior rows passed strict resume validation unchanged. The
  paired evaluator resumed from those rows at 17:42 JST without retraining.
- The committed supervisor enforces API-key preflight, frozen-Adapter SHA at
  smoke, complete permanent checkpoint receipts at Steps 8/16, bounded resume
  for operational interruptions, one W&B identity, and automatic handoff to
  the two-arm paired evaluator. Smoke/canary runs remain console-only.
- Paired CoreDev-2511 evaluation completed on 2026-08-13 with `2,240/2,240`
  supported single-image rows per arm and an explicit 271-row multi-image hold.
  Canonical Macro* is `57.8849` at Step 8 and `57.5422` at Step 16. Against the
  same paired PRL17-R2 no-visual control (`56.1964`, `58.1996`), the visual
  treatment is `+1.6885 pp` at Step 8 and `-0.6573 pp` at Step 16. Against the
  common Step 0 (`57.0320`), visual Steps 8/16 are respectively `+0.8529` and
  `+0.5102 pp`. The accepted interpretation is an early visual-shaping benefit
  without sustained scaling; PRL19 Step 8 is the selected treatment checkpoint.
- Step-16 output health is worse in the extreme tail. OCR prediction mean/P99
  grows from `1,979.3/47,296.2` chars at Step 8 to `2,829.6/80,924.2` at Step
  16, while the no-visual Step-16 P99 is `31,473.8`. Max-token stops remain
  `26/2,240` in both PRL19 arms, so the finding is heavier stochastic
  repetition rather than a larger capped-response count. This is recorded as
  a health warning against scaling the current F/G scalar unchanged.
- The Step-16 official OCR metric completed successfully, but one
  `134,775`-character prediction exceeded Python `csv`'s historical 128-KiB
  per-field default during local result summarization. The reader now raises
  the limit only to the already materialized artifact bound and restores the
  global setting afterward in commit `c40030e`. Twenty-two focused tests, Ruff
  and the real artifact summary pass. This changes no prediction, denominator,
  scorer or metric.
- Full results, reward analysis, output-health audit and artifact hashes are in
  `docs/PRL19_RP67_FROZEN_TFREE_VISUAL_REWARD_PAIRED_RESULTS_20260813.md`.
