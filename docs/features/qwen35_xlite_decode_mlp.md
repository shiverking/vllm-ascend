# Qwen3.5 xLite Decode MLP experiment

This opt-in integration replaces the dense MLP in pure Decode batches with
xLite kernels. Prefill, vision encoding, attention, GDN, KV allocation and
recurrent state indexing continue through the existing implementation. This
is a partial Decode adapter, not support for the full xLite hybrid backbone.

Enable it through the existing additional configuration:

```json
{
  "xlite_graph_config": {
    "enabled": true,
    "full_mode": false,
    "qwen35_decode_mlp_only": true
  }
}
```

The validated configuration is Qwen3.5 dense multimodal, BF16, TP=PP=DP=1,
without quantization, LoRA or speculative decoding. The bridge requires
`xlite==0.2.0rc4`, matching installed public headers, and a C++ toolchain.
It compiles a small binding once at worker startup, not xLite's kernel library.
Changing the package version fails explicitly rather than assuming ABI
compatibility. The C++ source must be included when packaging this experiment.

## Execution and memory

Weights remain owned by the original MLP and are passed by reference in their
existing ND/NZ format. One shared gate/up scratch and one shared activation
scratch serve all layers. Distinct per-layer output buffers prevent residual
aliasing. At max-num-seqs=512 with the 9B dimensions, these torch buffers use
164 MiB, plus a 32 MiB native runtime pool. Persistent torch buffers are included
in model memory accounting before sizing the KV cache.

There is no KV/GDN gather, scatter, host transfer or per-token weight conversion.
These buffers support the single model execution stream of the current worker;
concurrent forward streams require separate workspaces.

The bridge temporarily binds xLite to the current PyTorch NPU stream and calls
its exported asynchronous Matmul, SiLU-and-multiply and Matmul entrypoints.
It restores the original runtime stream on exit, including exception paths.
The standalone xLite Python SiLU binding synchronizes and must not be used on
the production graph path.

Routing stays inside an opaque custom operator: otherwise torch.compile can
specialize the initial Prefill flag and permanently bypass the native Decode
path. The startup diagnostic counts Python launches during warmups/captures;
it is not a count of graph replays. Encapsulating the Prefill MLP also changes
compiler visibility, so end-to-end performance must be measured independently
of the native MLP microbenchmark.

## Validation

CPU routing, buffer ownership and capacity checks:

```bash
python -m pytest tests/ut/xlite/test_qwen35_decode.py -q
```

NPU correctness and capture test (select an available card using the existing
Ascend visibility configuration):

```bash
python -m pytest tests/e2e/pull_request/one_card/test_xlite_qwen35_decode.py -q
```

The NPU test changes the input after capture, poisons the output, and verifies
that replay executes kernels and changes the result. Merely completing capture
is insufficient. BF16 kernels can change rounding and greedy generation near
close logits; neither bitwise equality nor QA quality is established by these
tests. Keep the feature opt-in and compare the actual image workload before
choosing a deployment default.

## Attribution controls

Two optional keys inside `xlite_graph_config` support controlled experiments:
`qwen35_decode_mlp_reference=true` keeps the adapter's buffers and custom-op
boundary but executes the original MLP, and
`qwen35_decode_mlp_diagnostics=true` counts host-side forward routes and Decode
tokens by attention state and padded graph size. Both default to false.
Diagnostics run outside the graph wrapper, count replays rather than just Python
kernel launches, exclude capture calls, and never synchronize device tensors.
Take counter differences around a workload to exclude startup warmups. The
reference control deliberately retains native-runtime memory allocations; use
an explicit equal KV budget to compare it with the unwrapped model.
