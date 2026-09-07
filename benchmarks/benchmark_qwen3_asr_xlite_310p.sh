#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-benchmark_results/qwen3_asr_310p}"
CONFIG_LABEL="${CONFIG_LABEL:-xlite-eager-audio}"

mkdir -p "${RESULT_DIR}"

for concurrency in 1 2 8 20; do
  for repetition in 1 2 3; do
    vllm bench serve \
      --backend openai-audio \
      --endpoint /v1/audio/transcriptions \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --model "${MODEL}" \
      --dataset-name hf \
      --dataset-path openslr/librispeech_asr \
      --hf-subset clean \
      --hf-split test \
      --asr-min-audio-len-sec 0 \
      --asr-max-audio-len-sec 30 \
      --num-warmups 10 \
      --num-prompts 100 \
      --request-rate inf \
      --max-concurrency "${concurrency}" \
      --save-result \
      --save-detailed \
      --result-dir "${RESULT_DIR}" \
      --result-filename "${CONFIG_LABEL}-c${concurrency}-r${repetition}.json" \
      --metadata "configuration=${CONFIG_LABEL}" "concurrency=${concurrency}" "repetition=${repetition}"
  done
done
