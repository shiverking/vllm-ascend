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

### Supported Features

| Feature | Ascend 310P status |
|---------|--------------------|
| FP16, single NPU | Supported |
| Audio input | Supported |
| EAGLE3 speculative decoding | Experimental |
| FULL_DECODE_ONLY ACLGraph | Experimental |
| Expert parallel / FlashComm1 | Not applicable (dense, single NPU) |

The 310P EAGLE3 path described below uses `/home/models/Qwen3-ASR-1.7B`
as the target and `/home/y00899301/vllm_draft_ep5` as the draft model. The
draft checkpoint must contain the `d2t` tensor in `model.safetensors`; the
standalone `draft_to_target.pt` file is not read by vLLM.

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

### Baseline

``` bash

vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

### EAGLE3 on Ascend 310P

Run the server from `/workspace` so the installed editable vLLM and
vLLM Ascend packages are used. The practical 310P baseline is 4096 tokens;
the draft checkpoint is limited to 65536 tokens, so a 128K test is not
applicable.

```bash
cd /workspace
vllm serve /home/models/Qwen3-ASR-1.7B \
  --served-model-name qwen3-asr-eagle3 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method":"eagle3","model":"/home/y00899301/vllm_draft_ep5","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,16,32]}' \
  --port 8000
```

With `num_speculative_tokens=K`, each verifier decode graph contains
`batch_size * (K + 1)` tokens. The capture sizes above therefore cover request
batches 1, 4, and 8 when `K=3`; they are token counts rather than request batch
sizes.

To isolate graph failures, add `--enforce-eager` and add
`"enforce_eager":true` to `--speculative-config`. For a fast architecture
check, also add `--load-format dummy`; dummy weights do not validate `d2t`,
real weight loading, transcription quality, or EAGLE3 acceptance rate.

The existing 310P audio encoder graph can be enabled together with the
decoder graph by adding an observed set of audio sequence buckets, for example:

```bash
--additional-config '{"audio_encoder_aclgraph_sizes":[256,512,1024]}'
```

## Functional Verification

Once the server is started, use the transcription endpoint so vLLM builds the
Qwen3-ASR generation prompt. Do not use a bare audio-only chat request as an
ASR correctness check.

```shell
curl -sS http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/path/to/asr_en.wav" \
  -F "model=qwen3-asr-eagle3" \
  -F "language=en" \
  -F "temperature=0" \
  -F "max_completion_tokens=256"
```

For the baseline server, replace `qwen3-asr-eagle3` with the model ID returned
by `/v1/models`. Keep the endpoint, audio, language, temperature, and completion
limit identical when comparing baseline and EAGLE3. Verify all of the following:

1. `GET /v1/models` returns HTTP 200.
2. The first audio request returns a non-empty transcription and the server
   remains alive.
3. Greedy token IDs and transcription match the same FP16 server without
   speculative decoding.
4. Metrics report accepted draft tokens and logs show ACLGraph replay in graph
   mode.

Run batches of 1, 4, and 8 requests with mixed audio lengths to cover request
replacement, draft query-length buffers, and KV slot mapping.

## Accuracy Evaluation

For 310P EAGLE3, first compare 100 LibriSpeech test-clean samples against a
no-speculation FP16 baseline, then run the existing 500-sample evaluation.
Greedy outputs must match the baseline exactly. Record the measured 310P WER
and acceptance rate separately; the A2 result below is not a 310P result.

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
