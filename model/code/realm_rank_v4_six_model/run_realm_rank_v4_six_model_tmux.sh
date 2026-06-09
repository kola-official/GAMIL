#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAMIL_ROOT="${GAMIL_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${GAMIL_ROOT}/raw_data/local_sources}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${GAMIL_ROOT}/processed_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${GAMIL_ROOT}/checkpoint/local_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/realm_rank_v4_six_model}"
MODEL_ROOT="${OUTPUT_ROOT}/models"
INIT_ROOT="${OUTPUT_ROOT}/init"
LOG_ROOT="${OUTPUT_ROOT}/logs"
STATE_ROOT="${OUTPUT_ROOT}/state"
BENCHMARK_ROOT="${OUTPUT_ROOT}/benchmark"
DATA_PATH="${DATA_PATH:-${PROCESSED_DATA_ROOT}/realm_rank_v4/train_csv}"
TEST_FASTA="${TEST_FASTA:-${PROCESSED_DATA_ROOT}/realm_rank_v4/test.fasta.gz}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-${DATA_PATH}/tokenized_cache}"
TOKEN_CACHE_MODEL_REF="${TOKEN_CACHE_MODEL_REF:-${GAMIL_ROOT}/model/local_models/bert2_flash_attn2_patch}"

TEACHER_O="${TEACHER_O:-${GAMIL_ROOT}/model/local_models/staged_models/viralm-o}"
TEACHER_R="${TEACHER_R:-${GAMIL_ROOT}/model/local_models/staged_models/viralm-r}"
STUDENT_O_6L="${STUDENT_O_6L:-${INIT_ROOT}/viralm_o_6l}"
STUDENT_R_6L="${STUDENT_R_6L:-${INIT_ROOT}/viralm_r_v4_final_6l}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29610}"
TMUX_SESSION="${TMUX_SESSION:-realm_rank_v4_six_model}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
MEANPOOL_BATCH_SIZE="${MEANPOOL_BATCH_SIZE:-32}"
MEANPOOL_EVAL_BATCH_SIZE="${MEANPOOL_EVAL_BATCH_SIZE:-64}"
MIL_BATCH_SIZE="${MIL_BATCH_SIZE:-8}"
MIL_EVAL_BATCH_SIZE="${MIL_EVAL_BATCH_SIZE:-32}"
MIL_GRAD_ACCUM_STEPS="${MIL_GRAD_ACCUM_STEPS:-8}"
SCAN_CHUNK="${SCAN_CHUNK:-48}"
GRAD_CHUNK="${GRAD_CHUNK:-32}"
MIL_12L_SCAN_CHUNK="${MIL_12L_SCAN_CHUNK:-48}"
MIL_12L_GRAD_CHUNK="${MIL_12L_GRAD_CHUNK:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
MEANPOOL_DATALOADER_WORKERS="${MEANPOOL_DATALOADER_WORKERS:-16}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-4096}"
TOKENIZE_NUM_PROC="${TOKENIZE_NUM_PROC:-16}"
EVAL_FRAG_METRICS="${EVAL_FRAG_METRICS:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

mkdir -p "${MODEL_ROOT}" "${INIT_ROOT}" "${LOG_ROOT}" "${STATE_ROOT}" "${BENCHMARK_ROOT}"

if [[ "${RUN_IN_TMUX}" == "1" && "${REALM_RANK_V4_SIX_MODEL_WORKER:-0}" != "1" ]]; then
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${TMUX_SESSION}" >&2
    echo "Attach: tmux attach -t ${TMUX_SESSION}" >&2
    exit 1
  fi
  printf -v SCRIPT_DIR_Q '%q' "${SCRIPT_DIR}"
  printf -v CMD_Q '%q ' env \
    "REALM_RANK_V4_SIX_MODEL_WORKER=1" \
    "RUN_IN_TMUX=0" \
    "OUTPUT_ROOT=${OUTPUT_ROOT}" \
    "DATA_PATH=${DATA_PATH}" \
    "TEST_FASTA=${TEST_FASTA}" \
    "TOKENIZED_CACHE_DIR=${TOKENIZED_CACHE_DIR}" \
    "TOKEN_CACHE_MODEL_REF=${TOKEN_CACHE_MODEL_REF}" \
    "TEACHER_O=${TEACHER_O}" \
    "TEACHER_R=${TEACHER_R}" \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "TORCHRUN_BIN=${TORCHRUN_BIN}" \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    "NPROC_PER_NODE=${NPROC_PER_NODE}" \
    "BASE_MASTER_PORT=${BASE_MASTER_PORT}" \
    "FORCE_RERUN=${FORCE_RERUN}" \
    "MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH}" \
    "MEANPOOL_BATCH_SIZE=${MEANPOOL_BATCH_SIZE}" \
    "MEANPOOL_EVAL_BATCH_SIZE=${MEANPOOL_EVAL_BATCH_SIZE}" \
    "MIL_BATCH_SIZE=${MIL_BATCH_SIZE}" \
    "MIL_EVAL_BATCH_SIZE=${MIL_EVAL_BATCH_SIZE}" \
    "MIL_GRAD_ACCUM_STEPS=${MIL_GRAD_ACCUM_STEPS}" \
    "SCAN_CHUNK=${SCAN_CHUNK}" \
    "GRAD_CHUNK=${GRAD_CHUNK}" \
    "MIL_12L_SCAN_CHUNK=${MIL_12L_SCAN_CHUNK}" \
    "MIL_12L_GRAD_CHUNK=${MIL_12L_GRAD_CHUNK}" \
    "EVAL_FRAG_METRICS=${EVAL_FRAG_METRICS}" \
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
    "DATALOADER_WORKERS=${DATALOADER_WORKERS}" \
    "MEANPOOL_DATALOADER_WORKERS=${MEANPOOL_DATALOADER_WORKERS}" \
    "TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE}" \
    "TOKENIZE_NUM_PROC=${TOKENIZE_NUM_PROC}" \
    "SMOKE_TEST=${SMOKE_TEST}" \
    bash "${SCRIPT_DIR}/run_realm_rank_v4_six_model_tmux.sh"
  tmux new-session -d -s "${TMUX_SESSION}" "cd ${SCRIPT_DIR_Q} && ${CMD_Q}"
  echo "Started Realm-Rank v4 six-model pipeline in tmux."
  echo "  session: ${TMUX_SESSION}"
  echo "  pipeline log: ${LOG_ROOT}/pipeline.log"
  echo "Attach: tmux attach -t ${TMUX_SESSION}"
  echo "Tail: tail -f ${LOG_ROOT}/pipeline.log"
  exit 0
fi

PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export SCAN_CHUNK
export GRAD_CHUNK
export EVAL_FRAG_METRICS
export SMOKE_TEST

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE}" -gt 1 ]] && ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 && [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not executable: ${TORCHRUN_BIN}" >&2
  exit 1
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

quote_cmd() {
  printf '%q ' "$@"
}

record_repro() {
  echo "[$(timestamp)] Recording reproducibility context"
  nvidia-smi > "${LOG_ROOT}/nvidia-smi.txt" 2>&1 || true
  "${PYTHON_BIN}" -m pip freeze > "${LOG_ROOT}/pip_freeze.txt" 2>&1 || true
  "${CONDA_BIN:-conda}" list -n "${CONDA_ENV:-vl}" > "${LOG_ROOT}/conda_list_vl.txt" 2>&1 || true
  if git -C "${GAMIL_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${GAMIL_ROOT}" rev-parse HEAD > "${LOG_ROOT}/git_commit.txt" 2>&1 || true
    git -C "${GAMIL_ROOT}" status --short > "${LOG_ROOT}/git_status_short.txt" 2>&1 || true
    git -C "${GAMIL_ROOT}" diff --stat > "${LOG_ROOT}/git_diff_stat.txt" 2>&1 || true
  else
    echo "not a git worktree: ${GAMIL_ROOT}" > "${LOG_ROOT}/git_commit.txt"
  fi
}

launch_cmd() {
  local port="$1"
  shift
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_BIN}" \
      --nnodes 1 \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_port "${port}" \
      "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

run_logged() {
  local task_name="$1"
  shift
  local done_file="${STATE_ROOT}/${task_name}.done"
  local failed_file="${STATE_ROOT}/${task_name}.failed"
  local log_file="${LOG_ROOT}/${task_name}.log"

  if [[ "${FORCE_RERUN}" != "1" && -f "${done_file}" ]]; then
    echo "[$(timestamp)] SKIP ${task_name}: ${done_file} exists"
    return 0
  fi

  rm -f "${done_file}" "${failed_file}"
  echo "[$(timestamp)] START ${task_name}"
  echo "[$(timestamp)] LOG ${log_file}"
  set +e
  (
    echo "started_at=$(timestamp)"
    echo "task=${task_name}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "SMOKE_TEST=${SMOKE_TEST}"
    echo "command=$(quote_cmd "$@")"
    "$@"
    exit_code=$?
    if [[ "${exit_code}" -eq 0 ]]; then
      echo "finished_at=$(timestamp)"
    else
      echo "failed_at=$(timestamp)"
    fi
    exit "${exit_code}"
  ) > "${log_file}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    date '+%Y-%m-%d %H:%M:%S' > "${done_file}"
    echo "[$(timestamp)] DONE ${task_name}"
  else
    date '+%Y-%m-%d %H:%M:%S' > "${failed_file}"
    echo "[$(timestamp)] FAILED ${task_name} exit=${exit_code} log=${log_file}" >&2
    exit "${exit_code}"
  fi
}

run_python_logged() {
  local task_name="$1"
  shift
  run_logged "${task_name}" "${PYTHON_BIN}" "$@"
}

run_torch_logged() {
  local task_name="$1"
  local port="$2"
  shift 2
  run_logged "${task_name}" launch_cmd "${port}" "$@"
}

echo "[$(timestamp)] Realm-Rank v4 six-model pipeline"
echo "Output root: ${OUTPUT_ROOT}"
echo "Data path: ${DATA_PATH}"
echo "Test FASTA: ${TEST_FASTA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NPROC_PER_NODE=${NPROC_PER_NODE}, SMOKE_TEST=${SMOKE_TEST}"

record_repro

run_python_logged copy_euk_pro_references "${SCRIPT_DIR}/copy_euk_pro_references.py" \
  --output-dir "${OUTPUT_ROOT}/reference"

run_python_logged init_six_layer_students "${SCRIPT_DIR}/init_six_layer_students.py" \
  --output-root "${INIT_ROOT}"

COMMON_CACHE_ARGS=(
  --data-path "${DATA_PATH}"
  --model-max-length "${MODEL_MAX_LENGTH}"
  --tokenized-cache-dir "${TOKENIZED_CACHE_DIR}"
  --token-cache-model-ref "${TOKEN_CACHE_MODEL_REF}"
  --tokenizer-batch-size "${TOKENIZER_BATCH_SIZE}"
  --tokenize-num-proc "${TOKENIZE_NUM_PROC}"
  --rebuild-tokenized-cache False
  --smoke-test "${SMOKE_TEST}"
)

run_torch_logged viralm_o_6l_meanpool_kd "$((BASE_MASTER_PORT + 1))" "${SCRIPT_DIR}/train_meanpool_kd.py" \
  --student-model "${STUDENT_O_6L}" \
  --teacher-model "${TEACHER_O}" \
  --output-dir "${MODEL_ROOT}/viralm_o_6l_meanpool_kd" \
  --run-name viralm_o_6l_meanpool_kd \
  --per-device-train-batch-size "${MEANPOOL_BATCH_SIZE}" \
  --per-device-eval-batch-size "${MEANPOOL_EVAL_BATCH_SIZE}" \
  --dataloader-num-workers "${MEANPOOL_DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

run_torch_logged viralm_r_v4_final_6l_meanpool_kd "$((BASE_MASTER_PORT + 2))" "${SCRIPT_DIR}/train_meanpool_kd.py" \
  --student-model "${STUDENT_R_6L}" \
  --teacher-model "${TEACHER_R}" \
  --output-dir "${MODEL_ROOT}/viralm_r_v4_final_6l_meanpool_kd" \
  --run-name viralm_r_v4_final_6l_meanpool_kd \
  --per-device-train-batch-size "${MEANPOOL_BATCH_SIZE}" \
  --per-device-eval-batch-size "${MEANPOOL_EVAL_BATCH_SIZE}" \
  --dataloader-num-workers "${MEANPOOL_DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

run_torch_logged viralm_o_6l_gated_mil_kd "$((BASE_MASTER_PORT + 3))" "${SCRIPT_DIR}/train_gated_mil_kd.py" \
  --backbone-model "${MODEL_ROOT}/viralm_o_6l_meanpool_kd" \
  --teacher-model "${TEACHER_O}" \
  --resume-mil-state "${MODEL_ROOT}/viralm_o_6l_gated_mil_kd/best_mil_model.pt" \
  --output-dir "${MODEL_ROOT}/viralm_o_6l_gated_mil_kd" \
  --run-name viralm_o_6l_gated_mil_kd \
  --batch-size "${MIL_BATCH_SIZE}" \
  --eval-batch-size "${MIL_EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${MIL_GRAD_ACCUM_STEPS}" \
  --scan-chunk "${SCAN_CHUNK}" \
  --grad-chunk "${GRAD_CHUNK}" \
  --dataloader-workers "${DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

run_torch_logged viralm_r_v4_final_6l_gated_mil_kd "$((BASE_MASTER_PORT + 4))" "${SCRIPT_DIR}/train_gated_mil_kd.py" \
  --backbone-model "${MODEL_ROOT}/viralm_r_v4_final_6l_meanpool_kd" \
  --teacher-model "${TEACHER_R}" \
  --resume-mil-state "${MODEL_ROOT}/viralm_r_v4_final_6l_gated_mil_kd/best_mil_model.pt" \
  --output-dir "${MODEL_ROOT}/viralm_r_v4_final_6l_gated_mil_kd" \
  --run-name viralm_r_v4_final_6l_gated_mil_kd \
  --batch-size "${MIL_BATCH_SIZE}" \
  --eval-batch-size "${MIL_EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${MIL_GRAD_ACCUM_STEPS}" \
  --scan-chunk "${SCAN_CHUNK}" \
  --grad-chunk "${GRAD_CHUNK}" \
  --dataloader-workers "${DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

run_torch_logged viralm_o_12l_gated_mil "$((BASE_MASTER_PORT + 5))" "${SCRIPT_DIR}/train_gated_mil_supervised.py" \
  --backbone-model "${TEACHER_O}" \
  --output-dir "${MODEL_ROOT}/viralm_o_12l_gated_mil" \
  --run-name viralm_o_12l_gated_mil \
  --batch-size "${MIL_BATCH_SIZE}" \
  --eval-batch-size "${MIL_EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${MIL_GRAD_ACCUM_STEPS}" \
  --scan-chunk "${MIL_12L_SCAN_CHUNK}" \
  --grad-chunk "${MIL_12L_GRAD_CHUNK}" \
  --dataloader-workers "${DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

run_torch_logged viralm_r_v4_final_12l_gated_mil "$((BASE_MASTER_PORT + 6))" "${SCRIPT_DIR}/train_gated_mil_supervised.py" \
  --backbone-model "${TEACHER_R}" \
  --output-dir "${MODEL_ROOT}/viralm_r_v4_final_12l_gated_mil" \
  --run-name viralm_r_v4_final_12l_gated_mil \
  --batch-size "${MIL_BATCH_SIZE}" \
  --eval-batch-size "${MIL_EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${MIL_GRAD_ACCUM_STEPS}" \
  --scan-chunk "${MIL_12L_SCAN_CHUNK}" \
  --grad-chunk "${MIL_12L_GRAD_CHUNK}" \
  --dataloader-workers "${DATALOADER_WORKERS}" \
  "${COMMON_CACHE_ARGS[@]}"

BENCHMARK_ARGS=(
  "${SCRIPT_DIR}/benchmark_realm_rank_v4_test.py"
  --test-fasta "${TEST_FASTA}"
  --output-dir "${BENCHMARK_ROOT}"
  --model-root "${MODEL_ROOT}"
  --model-max-length "${MODEL_MAX_LENGTH}"
  --scan-chunk "${SCAN_CHUNK}"
  --dataloader-workers "${DATALOADER_WORKERS}"
)
if [[ "${SMOKE_TEST}" == "1" ]]; then
  BENCHMARK_ARGS+=(--max-records "${SMOKE_BENCHMARK_RECORDS:-128}" --allow-missing-models)
fi
run_python_logged benchmark "${BENCHMARK_ARGS[@]}"

echo "[$(timestamp)] All tasks complete."
