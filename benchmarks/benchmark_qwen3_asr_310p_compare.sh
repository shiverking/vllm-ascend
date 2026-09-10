#!/usr/bin/env bash
set -euo pipefail

# Compare native eager, native FULL_DECODE_ONLY, and Xlite full-mode decoders.
# Every configuration uses the same dedicated audio-encoder ACLGraph gears.

MODEL="${MODEL:-/home/models/Qwen3-ASR-1.7B}"
AUDIO_DIR="${AUDIO_DIR:-/home/y00899301/vllm_omni_debug/Qwen3-ASR-310P/test/asr/performance/corpus/en}"
PORT="${PORT:-1025}"
RESULT_DIR="${RESULT_DIR:-}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 20}"
CONFIGS="${CONFIGS:-native_eager native_full_decode_only xlite_full}"
REPETITIONS="${REPETITIONS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP_REQUESTS=2
OUTPUT_LEN="${OUTPUT_LEN:-256}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-20}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-2048}"
GPU_MEMORY_UTILIZATION=0.8
DECODE_ATTENTION_BACKEND=legacy
PREFILL_ATTENTION_BACKEND=legacy
MATMUL_BACKEND=m200_asr
MATMUL_OPTIMIZATION=legacy
MATMUL_POLICY=""
AUDIO_GRAPH_SIZES="[26,52,78,128,256,384,512]"
DECODE_GRAPH_SIZES="[1,2,4,8,16,20]"
RUN_KIND="formal"

usage() {
  cat <<'EOF'
Usage: benchmark_qwen3_asr_310p_compare.sh [options]

With no options, runs the formal 3-configuration comparison once at client
concurrency 1/2/4/8/16/20 using the known model and corpus paths.

Options:
  --smoke                 Run 40 requests at concurrency 1/8/20
  --model PATH            Model path
  --audio-dir PATH        Directory containing local audio files
  --port PORT             Local serving port (default: 1025)
  --result-dir PATH       Result directory (default includes a timestamp)
  --configs "LIST"        Space-separated configurations
  --concurrencies "LIST"  Space-separated client concurrency values
  --num-prompts N         Measured requests per run
  --output-len N          Maximum generated tokens, including startup warmups
  --decode-attention-backend BACKEND
                         Xlite Decode: legacy, direct_atb, native_atb, batched_aclnn or paged_310p (default: legacy)
  --prefill-attention-backend BACKEND
                         Xlite Prefill: legacy or batched_aclnn_probe (default: legacy)
  --matmul-backend BACKEND
                         Xlite MatMul: m200_asr or aclnn (default: m200_asr)
  --matmul-optimization MODE
                         legacy or p3_aclnn (default: legacy)
  --matmul-policy PATH    Offline P3 policy JSON on the serving host
  --gpu-memory-utilization FRACTION
                         Server device memory budget in (0, 1] (default: 0.8)
  --repetitions N         Repetitions per concurrency (default: 1)
  --force                 Overwrite an existing result rather than skip it
  -h, --help              Show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --smoke)
      RUN_KIND="smoke"
      CONCURRENCIES="1 8 20"
      NUM_PROMPTS=40
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
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --decode-attention-backend) DECODE_ATTENTION_BACKEND="${2:?backend required}"; shift 2 ;;
    --prefill-attention-backend) PREFILL_ATTENTION_BACKEND="${2:?backend required}"; shift 2 ;;
    --matmul-backend) MATMUL_BACKEND="${2:?backend required}"; shift 2 ;;
    --matmul-optimization) MATMUL_OPTIMIZATION="${2:?mode required}"; shift 2 ;;
    --matmul-policy) MATMUL_POLICY="${2:?policy required}"; shift 2 ;;
    --gpu-memory-utilization)
      if (( $# < 2 )); then
        echo "--gpu-memory-utilization requires a value in (0, 1]." >&2
        exit 2
      fi
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --force) SKIP_EXISTING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${DECODE_ATTENTION_BACKEND}" in
  legacy|direct_atb|native_atb|batched_aclnn|paged_310p) ;;
  *) echo "Invalid decode attention backend: ${DECODE_ATTENTION_BACKEND}" >&2; exit 2 ;;
esac
case "${PREFILL_ATTENTION_BACKEND}" in
  legacy|batched_aclnn_probe) ;;
  *) echo "Invalid prefill attention backend: ${PREFILL_ATTENTION_BACKEND}" >&2; exit 2 ;;
esac
if [[ "${PREFILL_ATTENTION_BACKEND}" != legacy && "${DECODE_ATTENTION_BACKEND}" != direct_atb ]]; then
  echo "batched_aclnn_probe requires --decode-attention-backend direct_atb" >&2; exit 2
fi
case "${MATMUL_BACKEND}" in
  m200_asr|aclnn) ;;
  *) echo "Invalid MatMul backend: ${MATMUL_BACKEND}" >&2; exit 2 ;;
esac
case "${MATMUL_OPTIMIZATION}" in
  legacy|p3_aclnn) ;;
  *) echo "Invalid MatMul optimization: ${MATMUL_OPTIMIZATION}" >&2; exit 2 ;;
esac
if [[ "${MATMUL_OPTIMIZATION}" == p3_aclnn && "${DECODE_ATTENTION_BACKEND}" != legacy ]]; then
  echo "P3 requires legacy Attention" >&2; exit 2
fi
if [[ -n "${MATMUL_POLICY}" && "${MATMUL_OPTIMIZATION}" != p3_aclnn ]]; then
  echo "--matmul-policy requires --matmul-optimization p3_aclnn" >&2; exit 2
fi

if ! [[ "${GPU_MEMORY_UTILIZATION}" =~ ^(0?\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  echo "Invalid --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}': expected a decimal in (0, 1]." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="${RESULT_DIR:-benchmark_results/qwen3_asr_310p_${RUN_KIND}_${RUN_ID}}"
DATASET="${RESULT_DIR}/audio_manifest.jsonl"
WARMUP_AUDIO_PATH_FILE="${RESULT_DIR}/warmup_audio.txt"
WARMUP_AUDIO=""
SERVER_PID=""
RUN_NUMBER=0

# aiohttp honours NO_PROXY/no_proxy when trust_env=True. Keep the user's proxy
# for non-local traffic but force all benchmark and health traffic to bypass it.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

if [[ "${DECODE_ATTENTION_BACKEND}" == "direct_atb" ]]; then
  # Direct ATB keeps one PagedAttention operation per Decoder layer and batch
  # size. Fixed output staging keeps those tensor signatures stable; compact
  # decode metadata is uploaded once per forward.
  export ATB_OPSRUNNER_SETUP_CACHE_ENABLE=1
  log "Direct ATB per-layer Setup cache enabled"
fi

mkdir -p "${RESULT_DIR}/server_logs" "${RESULT_DIR}/client_logs" \
  "${RESULT_DIR}/warmup_responses"

log "Preparing deterministic local audio manifest"
python3 "${SCRIPT_DIR}/prepare_qwen3_asr_audio_dataset.py" \
  --audio-dir "${AUDIO_DIR}" --output "${DATASET}" \
  --warmup-output "${WARMUP_AUDIO_PATH_FILE}"
WARMUP_AUDIO="$(<"${WARMUP_AUDIO_PATH_FILE}")"

for concurrency in ${CONCURRENCIES}; do
  if (( concurrency < 1 || concurrency > MAX_NUM_SEQS )); then
    echo "Client concurrency ${concurrency} is outside [1, ${MAX_NUM_SEQS}]." >&2
    exit 2
  fi
done

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

warm_up_server() {
  local config="$1"
  local warmup_index
  local response_file

  log "Running ${WARMUP_REQUESTS} startup warmups with held-out audio=${WARMUP_AUDIO}"
  log "Warmup generation limit=${OUTPUT_LEN} tokens"
  for warmup_index in $(seq 1 "${WARMUP_REQUESTS}"); do
    response_file="${RESULT_DIR}/warmup_responses/${config}-${warmup_index}.json"
    log "Warmup ${warmup_index}/${WARMUP_REQUESTS} for ${config}"
    curl --noproxy '*' --silent --show-error --fail \
      "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
      --form-string "model=${MODEL}" \
      --form "file=@${WARMUP_AUDIO}" \
      --form-string "response_format=json" \
      --form-string "max_completion_tokens=${OUTPUT_LEN}" \
      --output "${response_file}"
  done
  log "Startup warmups completed for ${config}"
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
\"xlite_graph_config\":{\"enabled\":true,\"full_mode\":true,\"decode_attention_backend\":\"${DECODE_ATTENTION_BACKEND}\",\"prefill_attention_backend\":\"${PREFILL_ATTENTION_BACKEND}\",\"matmul_backend\":\"${MATMUL_BACKEND}\"}}"
      additional_config="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["xlite_graph_config"].update(matmul_optimization=sys.argv[2],matmul_policy=sys.argv[3] or None); print(json.dumps(d))' "${additional_config}" "${MATMUL_OPTIMIZATION}" "${MATMUL_POLICY}")"
      ;;
    *)
      echo "Unknown configuration: ${config}" >&2
      return 2
      ;;
  esac

  log "Starting server configuration=${config}"
  log "Server log=${log_file}"
  log "GPU memory utilization=${GPU_MEMORY_UTILIZATION}"
  log "MatMul backend=${MATMUL_BACKEND}; optimization=${MATMUL_OPTIMIZATION}; policy=${MATMUL_POLICY:-default12288}"
  log "Xlite Decode backend=${DECODE_ATTENTION_BACKEND}; Prefill backend=${PREFILL_ATTENTION_BACKEND}; async_matmul=${XLITE_310P_ASYNC_MATMUL:-unset}; force_sync_matmul=${XLITE_310P_FORCE_SYNC_MATMUL:-unset}; force_sync_aclnn=${XLITE_310P_FORCE_SYNC_ACLNN:-unset}"
  log "Decoder slots=${MAX_NUM_SEQS}, batched tokens=${MAX_NUM_BATCHED_TOKENS}, audio graph sizes=${AUDIO_GRAPH_SIZES}"
  log "Cache policy: prefix cache disabled, multimodal processor cache disabled"
  # Process substitution keeps SERVER_PID attached to the setsid process while
  # tee mirrors the complete server log to both the terminal and the log file.
  setsid vllm serve "${MODEL}" \
    --dtype float16 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size 1 \
    --block-size 128 \
    --max-model-len "${MODEL_MAX_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --additional-config "${additional_config}" \
    --port "${PORT}" \
    "${extra_args[@]}" > >(tee "${log_file}") 2>&1 &
  SERVER_PID=$!
  log "Server PID=${SERVER_PID}; health checks bypass HTTP/HTTPS proxies"
  wait_for_server
  warm_up_server "${config}"
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

  log "[run ${RUN_NUMBER}] Starting ${stem}: measured requests=${NUM_PROMPTS}"
  # openai-audio is multipart/form-data and is not classified as a generic
  # OpenAI-compatible generation backend by vllm bench. Pass temperature in
  # its request body so TranscriptionRequest receives deterministic greedy
  # sampling without triggering serve.py's backend validation.
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
    --extra-body '{"temperature":0}' \
    --num-warmups 0 \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate inf \
    --max-concurrency "${concurrency}" \
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
      "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
      "decode_attention_backend=${DECODE_ATTENTION_BACKEND}" \
      "prefill_attention_backend=${PREFILL_ATTENTION_BACKEND}" \
      "temperature=0" \
      "matmul_backend=${MATMUL_BACKEND}" \
      "matmul_optimization=${MATMUL_OPTIMIZATION}" \
      "matmul_policy=${MATMUL_POLICY:-default12288}" \
      "async_matmul=${XLITE_310P_ASYNC_MATMUL:-unset}" \
      "force_sync_matmul=${XLITE_310P_FORCE_SYNC_MATMUL:-unset}" \
      "atb_setup_cache=${ATB_OPSRUNNER_SETUP_CACHE_ENABLE:-unset}" \
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
