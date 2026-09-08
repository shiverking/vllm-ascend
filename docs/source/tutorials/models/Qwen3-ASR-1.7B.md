# Qwen3-ASR-1.7B

## Introduction

The released Qwen3-ASR-1.7B is a lightweight, high-performance automatic speech recognition (ASR) model developed by the Qwen Team. It delivers industry-leading recognition accuracy across Chinese/English multi-scene speech, Chinese dialects, multilingual and singing voice scenarios, with native support for long audio and streaming inference, and deep optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs). [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

`Qwen3-ASR-1.7B` is supported in `vllm-ascend`.

You can use our official docker image to run `Qwen3-ASR-1.7B` directly.

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

``` bash

vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json"
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url",
        "audio_url":
        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
}'
```

## Accuracy Evaluation

After all samples were processed, transcription quality was measured using:

- WER (Word Error Rate) for word-level recognition accuracy
- CER (Character Error Rate) for character-level recognition accuracy

The current evaluation results are:

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Accuracy | librispeech_asr / clean / test | Total Samples | 500 |
| Accuracy | librispeech_asr / clean / test | Success | 500 |
| Accuracy | librispeech_asr / clean / test | Failure | 0 |
| Accuracy | librispeech_asr / clean / test | WER | 0.035 |

## Performance

### Baseline Result

In the current evaluation, **Qwen3-ASR-1.7B** processed **100 samples** in approximately **57 seconds**, achieving an average throughput of **1.73 samples/s** under the current online serving setup.

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Performance | LibriSpeech test/clean (100 samples) | Total Samples | 100 |
| Performance | LibriSpeech test/clean (100 samples) | Total Runtime | 57 s |
| Performance | LibriSpeech test/clean (100 samples) | Average Throughput | 1.73 samples/s |

### Remarks

This result reflects end-to-end serving performance, including audio preprocessing, request construction, API communication, inference, and response parsing. Actual performance may vary depending on hardware, concurrency, audio length, and deployment configuration.

Further benchmarking is recommended for latency distribution, concurrent throughput, long-audio scenarios, and system resource utilization.

## Ascend 310P Xlite FP16 POC

The 310P POC runs the audio encoder with the native vLLM-Ascend path and
dispatches the complete language decoder and LM head to Xlite. It is limited to
one device, FP16, 128-token KV blocks, at most 20 sequences, and a 2048-token
context. Install an Xlite package whose build information reports
`soc=Ascend310P3`, `kernel_set=llm_fp16`, `abi=1`, and
`cache_layout=BSHD`; startup fails rather than silently falling back when this
contract is not met.

Start the correctness configuration with eager audio encoding:

```bash
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 2048 \
  --max-num-seqs 20 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --additional-config \
  '{"xlite_graph_config":{"enabled":true,"full_mode":true}}' \
  --port 8000
```

Verify a real WAV through the transcription API:

```bash
curl -sS http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "file=@/absolute/path/to/audio.wav" \
  -F "response_format=json"
```

For the performance configuration, add the dedicated audio encoder graph
gears. Decoder eager mode does not disable these encoder-only captures:

```bash
--additional-config \
'{"xlite_graph_config":{"enabled":true,"full_mode":true},"audio_encoder_aclgraph_sizes":[26,52,78,128,256,384,512]}'
```

For a local-corpus comparison, run
`benchmarks/benchmark_qwen3_asr_310p_compare.sh`. It restarts the server and
compares native eager Decoder, native `FULL_DECODE_ONLY`, and Xlite full mode.
All three configurations use the same audio encoder ACLGraphs. By default it
runs 100 requests once at client concurrency 1/2/4/8/16/20 and writes the
per-request data plus a `summary.csv`. The known model and local-corpus paths
are defaults, so the formal comparison is one command:

```bash
bash benchmarks/benchmark_qwen3_asr_310p_compare.sh
```

Use `--smoke` for a 40-request startup check at concurrency 1/8/20. The
script forces localhost health and benchmark requests to bypass inherited
HTTP/HTTPS proxies, checks health every 20 seconds, and mirrors server logs to
both the terminal and result directory. Prefix caching and the multimodal
processor cache are disabled so one server can execute every concurrency tier
without reusing earlier audio results. After each configuration starts, one
audio file excluded from the measured manifest is transcribed twice for
warmup; the individual concurrency runs do not issue additional warmups. The
server-side `max_num_seqs` and maximum tested client concurrency are both 20
in every configuration.

Current Xlite build metadata explicitly reports `aclnn_per_request` while
attention calls are serialized per request; this is the correctness fallback,
not a claim of batched PromptFlashAttention.
