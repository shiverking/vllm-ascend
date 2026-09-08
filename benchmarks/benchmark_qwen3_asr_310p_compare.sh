#!/usr/bin/env bash
set -euo pipefail

# Compare native eager, native FULL_DECODE_ONLY, and Xlite full-mode decoders.
# Every configuration uses the same dedicated audio-encoder ACLGraph gears.

MODEL="${MODEL:-/home/models/Qwen3-ASR-1.7B}"
AUDIO_DIR="${AUDIO_DIR:-/home/y00899301/vllm_omni_debug/Qwen3-ASR-310P/test/asr/performance/corpus/en}"
PORT="${PORT:-1025}"
RESULT_DIR="${RESULT_DIR:-}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 20 32}"
CONFIGS="${CONFIGS:-native_eager native_full_decode_only xlite_full}"
REPETITIONS="${REPETITIONS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_WARMUPS="${NUM_WARMUPS:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-20}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-2048}"
AUDIO_GRAPH_SIZES="[26,52,78,128,256,384,512]"
DECODE_GRAPH_SIZES="[1,2,4,8,16,20]"
RUN_KIND="formal"

usage() {
  cat <<'EOF'
Usage: benchmark_qwen3_asr_310p_compare.sh [options]

With no options, runs the formal 3-configuration comparison once at client
concurrency 1/2/4/8/16/20/32 using the known model and corpus paths.

Options:
  --smoke                 Run 40 requests at concurrency 1/20/32, 2 warmups
  --model PATH            Model path
  --audio-dir PATH        Directory containing local audio files
  --port PORT             Local serving port (default: 1025)
  --result-dir PATH       Result directory (default includes a timestamp)
  --configs "LIST"        Space-separated configurations
  --concurrencies "LIST"  Space-separated client concurrency values
  --num-prompts N         Measured requests per run
  --num-warmups N         Warmup requests per run
  --repetitions N         Repetitions per concurrency (default: 1)
  --force                 Overwrite an existing result rather than skip it
  -h, --help              Show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --smoke)
      RUN_KIND="smoke"
      CONCURRENCIES="1 20 32"
      NUM_PROMPTS=40
      NUM_WARMUPS=2
      REPETITIONS=1
      shift
      ;;
    --model) MODEL="$2"; shift 2 ;;
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --num-warmups) NUM_WARMUPS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --force) SKIP_EXISTING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="${RESULT_DIR:-benchmark_results/qwen3_asr_310p_${RUN_KIND}_${RUN_ID}}"
DATASET="${RESULT_DIR}/audio_manifest.jsonl"
SERVER_PID=""
RUN_NUMBER=0

# aiohttp honours NO_PROXY/no_proxy when trust_env=True. Keep the user's proxy
# for non-local traffic but force all benchmark and health traffic to bypass it.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

mkdir -p "${RESULT_DIR}/server_logs" "${RESULT_DIR}/client_logs"

log "Preparing deterministic local audio manifest"
python3 "${SCRIPT_DIR}/prepare_qwen3_asr_audio_dataset.py" \
  --audio-dir "${AUDIO_DIR}" --output "${DATASET}"

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  log "Stopping server process group ${SERVER_PID}"
  kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      SERVER_PID=""
      log "Server stopped"
      return
    fi
    sleep 1
  done
  log "Server did not stop gracefully; sending SIGKILL"
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local started_at=${SECONDS}
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  local http_code

  while true; do
    http_code="$(curl --noproxy '*' --silent --output /dev/null \
      --write-out '%{http_code}' --max-time 5 \
      "http://127.0.0.1:${PORT}/health" || true)"
    if [[ "${http_code}" == "200" ]]; then
      log "Server is ready after $((SECONDS - started_at))s"
      return
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      echo "Server exited before becoming ready; inspect the live output above." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting ${SERVER_READY_TIMEOUT}s for the server." >&2
      return 1
    fi
    log "Waiting for server: direct /health HTTP status=${http_code:-000}; next check in ${HEALTH_CHECK_INTERVAL}s"
    sleep "${HEALTH_CHECK_INTERVAL}"
  done
}

start_server() {
  local config="$1"
  local log_file="${RESULT_DIR}/server_logs/${config}.log"
  local additional_config
  local existing_code
  local -a extra_args=()

  existing_code="$(curl --noproxy '*' --silent --output /dev/null \
    --write-out '%{http_code}' --max-time 2 \
    "http://127.0.0.1:${PORT}/health" || true)"
  if [[ "${existing_code}" != "000" ]]; then
    echo "Port ${PORT} already has an HTTP service (status=${existing_code}); stop it before running this benchmark." >&2
    return 1
  fi

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

  log "Starting server configuration=${config}"
  log "Server log=${log_file}"
  log "Decoder slots=${MAX_NUM_SEQS}, batched tokens=${MAX_NUM_BATCHED_TOKENS}, audio graph sizes=${AUDIO_GRAPH_SIZES}"
  # Process substitution keeps SERVER_PID attached to the setsid process while
  # tee mirrors the complete server log to both the terminal and the log file.
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
    "${extra_args[@]}" > >(tee "${log_file}") 2>&1 &
  SERVER_PID=$!
  log "Server PID=${SERVER_PID}; health checks bypass HTTP/HTTPS proxies"
  wait_for_server
}

run_benchmark() {
  local config="$1"
  local concurrency="$2"
  local repetition="$3"
  local stem="${config}-c${concurrency}-r${repetition}"

  RUN_NUMBER=$((RUN_NUMBER + 1))
  if [[ "${SKIP_EXISTING}" == "1" && -s "${RESULT_DIR}/${stem}.json" ]]; then
    log "[run ${RUN_NUMBER}] Skipping existing ${stem}.json"
    return
  fi

  log "[run ${RUN_NUMBER}] Starting ${stem}: warmups=${NUM_WARMUPS}, measured requests=${NUM_PROMPTS}"
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
  log "[run ${RUN_NUMBER}] Finished ${stem}; result=${RESULT_DIR}/${stem}.json"
}

log "Benchmark plan: configs='${CONFIGS}', client concurrencies='${CONCURRENCIES}', repetitions=${REPETITIONS}"
log "Results will be written to ${RESULT_DIR}"
for config in ${CONFIGS}; do
  log "===== Configuration ${config} ====="
  start_server "${config}"
  for concurrency in ${CONCURRENCIES}; do
    log "--- Client concurrency ${concurrency} (server max_num_seqs=${MAX_NUM_SEQS}) ---"
    for repetition in $(seq 1 "${REPETITIONS}"); do
      run_benchmark "${config}" "${concurrency}" "${repetition}"
    done
  done
  stop_server
done

log "All benchmark runs completed; generating median summary"
python3 "${SCRIPT_DIR}/summarize_qwen3_asr_310p.py" \
  "${RESULT_DIR}" --expected-runs "${REPETITIONS}"
log "Benchmark complete: ${RESULT_DIR}/summary.csv"
