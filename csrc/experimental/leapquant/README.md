# Ascend LeapQuant GDN experiment

This directory contains the complete, opt-in native GDN state-compression
implementation for Qwen3.5-9B on Ascend910B3. Build, tests, synthetic benchmarks,
and serving no longer require an application repository or image dataset.
It is not included in the default custom-operator build and remains disabled
unless `additional_config.leapquant_config.enabled` is true.

## Implementation and ownership

- This repository owns the AscendC kernel, native Python binding, compressed GDN
  cache layout, prefill import/export, and Qwen3.5 integration.
- The matching vLLM branch owns immutable fixed-prefix retention and optional
  throughput admission. Neither feature contains Ascend-specific kernel code.
- Packet ABI remains LQ04: 27,744 bytes of payload per value head, padded to
  32,768 bytes for serving. A complete FP32 recurrent state is materialized
  inside UB for Fold, rather than allocated as persistent FP32 HBM state.
- Compression applies to GDN recurrent state. Model weights and full-attention
  KV remain BF16. TP/PP/PCP/DCP must all be one; MTP, weight/KV quantization, and
  cache transfer are rejected by the serving configuration validator.
- Fused, streaming, tiled, and experimental Cube/digit entry points remain
  available. Serving uses the ordinary fused padded-packet entry point.

The integrated kernel includes targeted initialization, live-window trimming,
paired Q/K readout, guarded FP16 residual contraction, fused reconstruction
scaling, in-place normalization, and range-checked two-step Newton reciprocals.
It also includes the reviewed tile16 work-buffer bound, tile128 FP32 mask,
legacy staging/Brcb alias, and HALF/staging combination fixes. Output DMA uses
the conservative wait by default. No per-token full-state CPU transfer is added.

## Build

Use the Python interpreter from the environment containing both vLLM and
vLLM-Ascend. Source the installed CANN environment before building and testing.
The tested configuration is CANN9.1.0, Ascend910B3, and 40 AIV cores.
The FP32 comparison and model serving also require this repository's normal
ACLNN custom-operator package, built/installed with vLLM-Ascend. Point CANN at
that installed vendor directory before importing PyTorch:

```bash
export ASCEND_CUSTOM_OPP_PATH=/absolute/path/to/installed/vendors/custom_transformer
export LD_LIBRARY_PATH="${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}"
```

This is separate from the LeapQuant library below. Without the custom package,
CANN may resolve its built-in recurrent operator, which rejects FP32 state;
do not silently switch the reference to BF16 when that happens.

```bash
cmake -S csrc/experimental/leapquant -B build/leapquant_native \
  -DASCEND_PYTHON_EXECUTABLE=/absolute/path/to/venv/bin/python \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DSOC_VERSION=Ascend910B3
cmake --build build/leapquant_native -j4
```

The library is `build/leapquant_native/lib/libleapquant_native.so`.
Keep different configurations in different build directories. Generated headers
are build artifacts; do not copy a pre-generated header beside the source.
Both host/device compilation use the same generated source snapshot and flags.

All boolean CMake switches accept `0` or `1`, rather than `ON`/`OFF`:

| Switch | Default | Purpose |
| --- | ---: | --- |
| `LEAPQUANT_POWER_ITERS` | 8 | Cold-start power iterations, integer 1 through 8 |
| `LEAPQUANT_STREAM_PREFETCH` | 1 | Prefetch only in explicit streaming mode |
| `LEAPQUANT_OPT_TARGETED_VEC_INIT` | 1 | Initialize required vector padding |
| `LEAPQUANT_OPT_TRIM_COEFF` | 1 | Compute coefficients only for live records |
| `LEAPQUANT_OPT_TRIM_TILED` | 1 | Load only live tiled records |
| `LEAPQUANT_OPT_OMIT_WEIGHT_ZERO` | 1 | Omit unused weight initialization |
| `LEAPQUANT_V4_FUSED_RECONSTRUCT_SCALE` | 1 | Fuse reconstruction scale with decay |
| `LEAPQUANT_V4_DIRECT_BASE_RESULTS` | 1 | Consume readout without legacy staging |
| `LEAPQUANT_HALF_BASE` | 1 | FP16 residual readout, FP32 state fitting/update |
| `LEAPQUANT_HALF_GUARD` | 1 | Fall back to FP32 when scaled operands exceed 2 |
| `LEAPQUANT_DIRECT_NORM` | 1 | Normalize the residual in place |
| `LEAPQUANT_USE_NEWTON_DIV` | 1 | Two refinements inside the safe denominator range |
| `LEAPQUANT_FMA_ROWS` / `LEAPQUANT_FMA_COLS` | 0 | Experimental contraction association |
| `LEAPQUANT_V4_RESIDUAL_PAIR` | 0 | Experimental tiled residual readout |
| `LEAPQUANT_DUAL_MATVEC` | 0 | Experimental FP32 paired-read interleaving |
| `LEAPQUANT_DELAY_OUTPUT_STORE` | 0 | Experimental delayed output DMA wait |

`LEAPQUANT_USE_NEWTON_DIV=1` always includes the range check: if any denominator
falls outside absolute range `[1e-30, 1e30]`, the operation falls back to CANN
`Div`. The switch is an arithmetic choice, not a promise of faster execution.
Disabling `LEAPQUANT_HALF_BASE` restores FP32 residual readout while retaining
compressed state; it does not restore the uncompressed FP32 GDN baseline.
Disable LeapQuant in the serving configuration to obtain that baseline.
Changing floating-point association can change quantization trajectories;
experimental switches require numerical and performance validation.

## Verification without images

```bash
export ASCEND_RT_VISIBLE_DEVICES=7
/absolute/path/to/venv/bin/python -m pytest -q \
  tests/ut/ops/test_leapquant_config.py \
  tests/ut/ops/test_leapquant_diagnostics.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_leapquant_native.py
```

Tests compile private temporary libraries. Coverage includes FP32 recurrence
comparison, sparse slots, mixed cursors, immutable window records, checkpoint
copy/resume, pending-record export, padded packets, graph replay, tile16/128
bounds, both legacy staging variants, half-readout fallback, matrix contractions,
and extreme-denominator behavior against native CANN `Div`.

```bash
/absolute/path/to/venv/bin/python benchmarks/ops/benchmark_leapquant.py \
  --library build/leapquant_native/lib/libleapquant_native.so \
  --batches 32 192 256 --output /tmp/leapquant_native.json
```

The benchmark compares the uncompressed FP32 GDN operator with each supplied
`--library`. Each graph contains 16 distinct token inputs and a real Fold.
It reports per-step NPU event samples and library SHA256. Timings include the
native invocation and Fold, rather than only ordinary Decode steps. They are
operator measurements and cannot substitute for model throughput or QA quality.

## Serving

Install the matching `cloud-drive-leapquant-ascend-exp` branches of
`shiverking/vllm` and `shiverking/vllm-ascend` in the same environment. Set
`PYTHONPATH` to their checkout roots when running directly from source. Replace
the model and native-library paths below with absolute paths on the target host.

```bash
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
vllm serve /absolute/path/to/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --port 1025 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 16384 --max-num-seqs 512 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.99 \
  --enable-chunked-prefill --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128,160,176,192,208,224,256,272,288,296,304,320,336,352,368,384,416,448,480,512]}' \
  --additional-config '{"fixed_prefix_cache":{"budget_fraction":0.02},"throughput_scheduler":{"enabled":true,"reservation_mode":"aggressive","near_tokens":1,"safety_fraction":0.005},"leapquant_config":{"enabled":true,"library":"/absolute/path/to/libleapquant_native.so","cores":40}}'
```

Fixed-prefix reuse requires registering a template and using the fixed-prefix
endpoint in the matching vLLM branch. Ordinary chat completion still works.
Request concurrency is a client setting; `max_num_seqs=512` is a server ceiling.
Memory fraction 0.99 reproduces the measured deployment and needs an otherwise
idle card. All three experimental features can be disabled independently.

For metadata verification, optionally add
`--worker-extension-cls vllm_ascend.worker.leapquant_diagnostics.LeapQuantWorkerExtension`.
Its `leapquant_cache_diagnostics` worker RPC reports actual GDN tensor types,
shapes, loaded library paths/SHA256, full-attention cache types, and graph sizes.
It installs no per-step hook by default. The optional
`leapquant_start_batch_observer` RPC records CPU scheduler metadata only; use
reset/export/stop RPCs explicitly. Export includes `dropped_samples` if the
bounded history is exhausted. No KV calibration or KV quantization is included.

## Performance evidence and limits

A previous matched, continuously replenished workload observed the following
single-card rates after 240 seconds of warmup over a fixed 600-second window.
Each configuration ran once; drain was excluded. This table records historical
measurements, not a new deployment certification or an included dataset.

| Configuration | Client concurrency | Images/s | Mean TTFT/s | Mean cache usage | Window preemptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 GDN, fixed prefix | 192 | 1.4567 | 2.130 | 98.24% | 58 |
| FP32 GDN, aggressive | 192 | 1.4583 | 2.123 | 98.22% | 4 |
| Previous compressed kernel | 256 | 1.4900 | 2.453 | 84.21% | 0 |
| Reviewed kernel | 192 | 1.4683 | 2.089 | 63.12% | 0 |
| Reviewed kernel | 256 | 1.5017 | 2.424 | 84.26% | 0 |

The reviewed kernel was only 0.78% above the previous kernel at C256; this is
insufficient to establish a stable speedup. Its clearer benefit is compressed
state capacity. Cache usage uses each configuration's own pool denominator;
it is not the same as physical HBM savings in a preallocated deployment.
No full-attention KV quantization was enabled. The earlier numerical audit
matched the previous kernel's tested trajectories, but weak-forgetting synthetic
stress still showed accumulated approximation error; arbitrary long-sequence
accuracy is not guaranteed. Do not interpret short Decode-only drain throughput
as a continuously replenished model baseline.

## Integration verification, 2026-11-08

- Native/config tests: 122 passed, including three temporary library builds.
- Diagnostics unit tests: 7 passed. Matching vLLM cache/admission tests: 22 passed.
- The standalone synthetic benchmark completed FP32 and native B32 paths.
  This smoke run is not used for a new performance claim.
- A Qwen3.5-9B server started with only the two source repositories and CANN SDK
  on `PYTHONPATH`. Its 24 compressed GDN layers and loaded library SHA256 were
  verified. Two text requests and observer start/export/stop passed. The server
  was stopped afterward; model graph replay is covered by the native tests.
- Manual pre-commit checks on all changed files passed. Full-repository
  `bash format.sh ci` reports an existing ShellCheck SC2328 error in
  `csrc/build.sh:524`; that build script was not changed by this integration.

The migrated kernel source matches the reviewed source byte for byte. Its SHA256
at integration is
`7f756859937a0fb032fee187d461f10dc39750589af123a6fa80c76d9c6bf4b8`.
