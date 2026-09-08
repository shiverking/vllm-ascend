# P2 310P performance acceptance

Use only after the GVirt new-path attention and 129-token model gates pass.
P2 remains opt-in. This implementation has not been benchmarked on NPU locally.

First inspect the **existing** successful P1 server log for the effective MatMul
policy, or its saved runtime stats. The history supplied contains both async and
forced-sync experiments; the 0.28/2.83 req/s output alone does not identify flags.
Do not invent a policy from throughput. Keep the verified setting unchanged.
The comparison script now records the existing flags; Xlite logs effective runtime
flags at startup, and P2 counters every 128 forwards.

From vllm-ascend, in the same installed environment (no Ascend extension rebuild):

```bash
bash benchmarks/benchmark_qwen3_asr_310p_compare.sh \
  --configs xlite_full --decode-attention-backend paged_310p \
  --concurrencies "1 20" --num-prompts 100 --repetitions 1 \
  --output-len 64 --gpu-memory-utilization 0.8 \
  --result-dir benchmark_results/p2_paged
```

Audio graph gears remain `[26,52,78,128,256,384,512]`. Both warmups use the
pre-existing held-out audio selection. Local HTTP requests bypass proxies.
Detailed result JSON, client/server logs and summary.csv remain under the result
directory. Use a fresh directory when changing the backend: old results are skipped.

Direct serving uses the same existing command, changing only additional-config:

```json
{"xlite_graph_config":{"enabled":true,"full_mode":true,"decode_attention_backend":"paged_310p"},"audio_encoder_aclgraph_sizes":[26,52,78,128,256,384,512]}
```

`legacy` is the explicit rollback. Selecting paged on an older extension fails at
startup. Do not change the default before hardware correctness/performance pass.

## Evidence to retain

- 100/100 successes at c1 and c20, output cap64 and identical input order.
- Per-request `generated_texts` equality against the previous detailed JSON;
  total token equality is insufficient. Missing texts means unverified, not pass.
- Effective MatMul flags, Audio gears, device budget and source commits.
- Paged Decode requests >0, `legacy_decode_requests=0`, `decode_kv_gather_bytes=0`.
  Paged request counters are layer invocations, not unique API requests.
- Compare 0.28/2.83 req/s and TPOT83.66/145.31ms. Target c20>=3.40,
  c1>=0.266. Historical rounded numbers give approximate relative changes.

`summarize_qwen3_asr_p2.py --result NEW.json --baseline OLD.json --concurrency 20
--output comparison.json` reports performance and text mismatches; it does not
claim WER, input identity or final acceptance from JSON metrics alone.
