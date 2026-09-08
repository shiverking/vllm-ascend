# P3 ASR acceptance

Prerequisites: GVirt P3 build, offline `policy.json` with nonzero optimized shapes,
and the 129-token model gate. P2 is paused; Attention must be `legacy`.

Preserve the same MatMul synchronization environment as the successful reference.
The old 0.28/2.83 req/s client reports do not establish the effective flags. The
new server logs record the actual runtime policy and policy fingerprint. Do not
describe gains as P3-only if synchronization settings differ.

From vllm-ascend:

```bash
bash benchmarks/benchmark_qwen3_asr_310p_compare.sh \
  --configs xlite_full --decode-attention-backend legacy \
  --matmul-optimization p3_aclnn \
  --matmul-policy /home/y00899301/vllm_omni_debug/GVirt/xlite/p3_matmul_report/policy.json \
  --concurrencies "1 20" --num-prompts 100 --repetitions 1 \
  --output-len 64 --gpu-memory-utilization 0.8 \
  --result-dir "benchmark_results/p3_aclnn_$(date +%Y%m%d_%H%M%S)"
```

The script preserves Audio Graph gears `[26,52,78,128,256,384,512]`, held-out
warmup selection and proxy bypass. No Native modes are requested. The policy
path must be readable on the serving host. Startup reports whether it matched;
a mismatch uses default12288, not a silently reused policy from another build.

Serving additional-config fields are `matmul_optimization` (legacy/p3_aclnn)
and `matmul_policy` (optional JSON path). Rollback: choose legacy and omit policy;
no rebuild required. Do not combine P3 and paged_310p.

## Evidence and acceptance

Save summary.csv, detailed JSON, server logs with effective sync settings and
per-projection calls/chunks/copy bytes/workspace peaks. Diagnostics must be off.
P3 temporary-output planning adds 2 MiB to the pool estimate, not to the 512 MiB
ACLNN workspace limit; record the actual startup and peak memory.

Historical reference (rounded):

| Metric | c1 | c20 |
| --- | ---: | ---: |
| req/s | 0.28 | 2.83 |
| mean TPOT ms | 83.66 | 145.31 |
| generated tokens | 3985 | 3985 |

Target: `(c1>=0.322 AND c20>=2.6885)` OR `(c20>=3.2545 AND c1>=0.266)`.
Require 100 successes/0 failures for each concurrency, matching inputs/output
cap, identical per-request transcripts and no memory-budget violation. Matching
total tokens alone is insufficient; missing baseline texts means unverified,
not WER pass. Use the detailed JSON's `generated_texts` in original request order.

If targets fail, retain legacy default and inspect the short MatMul profile from
the same configuration; do not restart P2 or add a new kernel route in this stage.
