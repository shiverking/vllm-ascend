#!/usr/bin/env bash
set -euo pipefail

# Compare native eager, native FULL_DECODE_ONLY, and Xlite full-mode decoders.
# Every configuration uses the same dedicated audio-encoder ACLGraph gears.

MODEL="${MODEL:-/home/models/Qwen3-ASR-1.7B}"
AUDIO_DIR="${AUDIO_DIR:-/home/y00899301/vllm_omni_debug/Qwen3-ASR-310P/test/asr/performance/corpus/en}"
PORT="${PORT:-1025}"
RESULT_DIR="${RESULT_DIR:-benchmark_results/qwen3_asr_310p_compare}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 20 32}"
CONFIGS="${CONFIGS:-native_eager native_full_decode_only xlite_full}"
REPETITIONS="${REPETITIONS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_WARMUPS="${NUM_WARMUPS:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-2048}"
AUDIO_GRAPH_SIZES="[26,52,78,128,256,384,512]"
DECODE_GRAPH_SIZES="[1,2,4,8,16,20]"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${RESULT_DIR}/audio_manifest.jsonl"
SERVER_PID=""

mkdir -p "${RESULT_DIR}/server_logs" "${RESULT_DIR}/client_logs"
python3 "${SCRIPT_DIR}/prepare_qwen3_asr_audio_dataset.py" \
  --audio-dir "${AUDIO_DIR}" --output "${DATASET}"

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      SERVER_PID=""
      return
    fi
    sleep 1
  done
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  until curl --silent --show-error --fail "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      echo "Server exited before becoming ready" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting ${SERVER_READY_TIMEOUT}s for the server" >&2
      return 1
    fi
    sleep 2
  done
}

start_server() {
  local config="$1"
  local log_file="${RESULT_DIR}/server_logs/${config}.log"
  local additional_config
  local -a extra_args=()

  additional_config="{\"audio_encoder_aclgraph_sizes\":${AUDIO_GRAPH_SIZES}}"
  case "${config}" in
    native_eager)
      extra_args+=(--enforce-eager)
      ;;
    native_full_decode_only)
      extra_args+=(
        --compilation-config
        "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":${DECODE_GRAPH_SIZES}}"
      )
      ;;
    xlite_full)
      additional_config="{\"audio_encoder_aclgraph_sizes\":${AUDIO_GRAPH_SIZES},\
\"xlite_graph_config\":{\"enabled\":true,\"full_mode\":true}}"
      ;;
    *)
      echo "Unknown configuration: ${config}" >&2
      return 2
      ;;
  esac

  echo "Starting ${config}; log=${log_file}"
  setsid vllm serve "${MODEL}" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --block-size 128 \
    --max-model-len "${MODEL_MAX_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --additional-config "${additional_config}" \
    --port "${PORT}" \
    "${extra_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

run_benchmark() {
  local config="$1"
  local concurrency="$2"
  local repetition="$3"
  local stem="${config}-c${concurrency}-r${repetition}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${RESULT_DIR}/${stem}.json" ]]; then
    echo "Skipping existing result ${stem}.json"
    return
  fi

  echo "Benchmark ${stem}"
  vllm bench serve \
    --backend openai-audio \
    --endpoint /v1/audio/transcriptions \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model "${MODEL}" \
    --dataset-name custom_audio \
    --dataset-path "${DATASET}" \
    --disable-shuffle \
    --output-len "${OUTPUT_LEN}" \
    --num-warmups "${NUM_WARMUPS}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate inf \
    --max-concurrency "${concurrency}" \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${stem}.json" \
    --metadata \
      "configuration=${config}" \
      "client_concurrency=${concurrency}" \
      "server_max_num_seqs=${MAX_NUM_SEQS}" \
      "repetition=${repetition}" \
    2>&1 | tee "${RESULT_DIR}/client_logs/${stem}.log"
}

for config in ${CONFIGS}; do
  start_server "${config}"
  for concurrency in ${CONCURRENCIES}; do
    for repetition in $(seq 1 "${REPETITIONS}"); do
      run_benchmark "${config}" "${concurrency}" "${repetition}"
    done
  done
  stop_server
done

python3 "${SCRIPT_DIR}/summarize_qwen3_asr_310p.py" \
  "${RESULT_DIR}" --expected-runs "${REPETITIONS}"
