#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAMIL_ROOT="${GAMIL_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${GAMIL_ROOT}/raw_data/local_sources}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${GAMIL_ROOT}/processed_data}"
MODEL_ROOT="${MODEL_ROOT:-${GAMIL_ROOT}/model/local_models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${GAMIL_ROOT}/checkpoint/local_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAMIL_ROOT}/outputs}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SOURCE_FASTA_WAS_SET="${SOURCE_FASTA+x}"
SOURCE_FASTA="${SOURCE_FASTA:-${PROCESSED_DATA_ROOT}/realm_rank_v4/train.fasta.gz}"
SOURCE_DEV_FASTA="${SOURCE_DEV_FASTA:-}"
DATASET_DIR="${DATASET_DIR:-${PROCESSED_DATA_ROOT}/realm_rank_v4}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/bert2_flash_attn2_patch}"
SEED="${SEED:-42}"
DEV_FRACTION="${DEV_FRACTION:-0.1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-}"
TOKENIZE_NUM_PROC="${TOKENIZE_NUM_PROC:-}"
TOKENIZED_CACHE_DIR="${TOKENIZED_CACHE_DIR:-}"
TOKENIZED_CACHE_FORMAT="${TOKENIZED_CACHE_FORMAT:-arrow}"
REBUILD_TOKENIZED_CACHE="${REBUILD_TOKENIZED_CACHE:-False}"
SEQUENCE_EVAL_THRESHOLD="${SEQUENCE_EVAL_THRESHOLD:-0.5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_SAMPLES_PER_CLASS="${SMOKE_SAMPLES_PER_CLASS:-50}"
CSV_BUILD_WORKERS="${CSV_BUILD_WORKERS:-${BUILD_WORKERS:-}}"
CSV_BATCH_SIZE="${CSV_BATCH_SIZE:-4096}"
CSV_MAX_RECORDS="${CSV_MAX_RECORDS:-0}"
CSV_MAX_RECORDS_PER_CLASS="${CSV_MAX_RECORDS_PER_CLASS:-0}"
CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT="${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT:-0}"
REUSE_PREPARED_DATA="${REUSE_PREPARED_DATA:-1}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

FORCE_SINGLE_GPU="${FORCE_SINGLE_GPU:-0}"
TARGET_GPU="${TARGET_GPU:-0}"
USE_TORCHRUN="${USE_TORCHRUN:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"

USE_TRAIN_BATCH_PLAN="${USE_TRAIN_BATCH_PLAN:-0}"
TRAIN_BATCH_PLAN_PATH="${TRAIN_BATCH_PLAN_PATH:-}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-True}"

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-200}"
MEMORY_CHECK_INTERVAL_SEC="${MEMORY_CHECK_INTERVAL_SEC:-5}"

RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs/train}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${TS}.log}"
TMUX_SESSION="${TMUX_SESSION:-viralm_realm_rank_${TS}}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  DATA_OUT="${DATA_OUT:-${PROCESSED_DATA_ROOT}/realm_rank_v4/train_csv_smoke}"
  OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/viralm_realm_rank_smoke}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
  LOGGING_STEPS="${LOGGING_STEPS:-1}"
  EVAL_STEPS="${EVAL_STEPS:-20}"
  SAVE_STEPS="${SAVE_STEPS:-20}"
  if [[ -z "${DATALOADER_WORKERS}" ]]; then
    DATALOADER_WORKERS=0
  fi
  if [[ -z "${TOKENIZER_BATCH_SIZE}" ]]; then
    TOKENIZER_BATCH_SIZE=512
  fi
  if [[ -z "${TOKENIZE_NUM_PROC}" ]]; then
    TOKENIZE_NUM_PROC=2
  fi
  if [[ -z "${CSV_BUILD_WORKERS}" ]]; then
    CSV_BUILD_WORKERS=4
  fi
  if [[ "${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT}" == "0" && "${CSV_MAX_RECORDS_PER_CLASS}" == "0" && "${CSV_MAX_RECORDS}" == "0" ]]; then
    CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT="${SMOKE_SAMPLES_PER_CLASS}"
  fi
  RUN_NAME="${RUN_NAME:-viralm_realm_rank_smoke_seed_${SEED}}"
else
  DATA_OUT="${DATA_OUT:-${PROCESSED_DATA_ROOT}/realm_rank_v4/train_csv}"
  OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/viralm_realm_rank}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
  LOGGING_STEPS="${LOGGING_STEPS:-200}"
  EVAL_STEPS="${EVAL_STEPS:-3800}"
  SAVE_STEPS="${SAVE_STEPS:-3800}"
  if [[ -z "${DATALOADER_WORKERS}" ]]; then
    DATALOADER_WORKERS=16
  fi
  if [[ -z "${TOKENIZER_BATCH_SIZE}" ]]; then
    TOKENIZER_BATCH_SIZE=4096
  fi
  if [[ -z "${TOKENIZE_NUM_PROC}" ]]; then
    TOKENIZE_NUM_PROC=16
  fi
  if [[ -z "${CSV_BUILD_WORKERS}" ]]; then
    CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
    if [[ "${CPU_COUNT}" -gt 16 ]]; then
      CSV_BUILD_WORKERS=16
    else
      CSV_BUILD_WORKERS="${CPU_COUNT}"
    fi
  fi
  RUN_NAME="${RUN_NAME:-viralm_realm_rank_seed_${SEED}}"
fi

if [[ -z "${SOURCE_DEV_FASTA}" ]]; then
  DATA_OUT_ABS="${DATA_OUT}"
  if [[ "${DATA_OUT_ABS}" != /* ]]; then
    DATA_OUT_ABS="${GAMIL_ROOT}/${DATA_OUT_ABS}"
  fi
  if [[ -z "${DATASET_DIR}" && "$(basename "${DATA_OUT_ABS}")" == "train_csv" ]]; then
    DATASET_DIR="$(dirname "${DATA_OUT_ABS}")"
  fi
  if [[ -n "${DATASET_DIR}" ]]; then
    DATASET_DIR_ABS="${DATASET_DIR}"
    if [[ "${DATASET_DIR_ABS}" != /* ]]; then
      DATASET_DIR_ABS="${GAMIL_ROOT}/${DATASET_DIR_ABS}"
    fi
    if [[ -f "${DATASET_DIR_ABS}/train.fasta.gz" && -f "${DATASET_DIR_ABS}/dev.fasta.gz" ]]; then
      if [[ -z "${SOURCE_FASTA_WAS_SET}" ]]; then
        SOURCE_FASTA="${DATASET_DIR_ABS}/train.fasta.gz"
      fi
      SOURCE_DEV_FASTA="${DATASET_DIR_ABS}/dev.fasta.gz"
    fi
  fi
fi

if [[ -z "${TOKENIZED_CACHE_DIR}" ]]; then
  TOKENIZED_CACHE_DIR="${DATA_OUT}/tokenized_cache"
fi

to_bool_str() {
  local v
  v="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${v}" in
    1|true|yes|y|on)
      echo "True"
      ;;
    *)
      echo "False"
      ;;
  esac
}

REBUILD_TOKENIZED_CACHE_BOOL="$(to_bool_str "${REBUILD_TOKENIZED_CACHE}")"
DDP_FIND_UNUSED_PARAMETERS_BOOL="$(to_bool_str "${DDP_FIND_UNUSED_PARAMETERS}")"

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
    NPROC_PER_NODE=1
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE_DEVICES="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')"
    IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#VISIBLE_GPU_ARRAY[@]}"
  else
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
      GPU_COUNT=1
    fi
    if [[ "${GPU_COUNT}" -gt 2 ]]; then
      NPROC_PER_NODE=2
    else
      NPROC_PER_NODE="${GPU_COUNT}"
    fi
  fi
fi

if [[ "${USE_TORCHRUN}" == "auto" ]]; then
  if [[ "${FORCE_SINGLE_GPU}" == "0" && "${NPROC_PER_NODE}" -gt 1 ]]; then
    USE_TORCHRUN=1
  else
    USE_TORCHRUN=0
  fi
fi

collect_tree_rss_kb() {
  local root_pid="$1"
  ps -eo pid=,ppid=,rss= | awk -v root="${root_pid}" '
    {
      ppid[$1] = $2
      rss[$1] = $3
    }
    END {
      total = 0
      for (pid in ppid) {
        cur = pid
        while (cur in ppid) {
          if (cur == root) {
            total += rss[pid]
            break
          }
          cur = ppid[cur]
        }
      }
      print total + 0
    }
  '
}

run_with_memory_cap() {
  local stage_name="$1"
  shift

  local limit_kb=$(( MEMORY_LIMIT_GB * 1024 * 1024 ))
  local peak_rss_kb=0

  "$@" &
  local cmd_pid=$!

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    local tree_rss_kb
    tree_rss_kb="$(collect_tree_rss_kb "${cmd_pid}")"
    if [[ "${tree_rss_kb}" -gt "${peak_rss_kb}" ]]; then
      peak_rss_kb="${tree_rss_kb}"
    fi
    if [[ "${tree_rss_kb}" -gt "${limit_kb}" ]]; then
      local used_gib
      used_gib="$(awk -v kb="${tree_rss_kb}" 'BEGIN { printf "%.2f", kb/1024/1024 }')"
      echo "[MEMCAP] ${stage_name} exceeded ${MEMORY_LIMIT_GB}GiB (current=${used_gib}GiB). Killing process tree."
      local tree_pids
      tree_pids="$(
        ps -eo pid=,ppid= | awk -v root="${cmd_pid}" '
          {
            ppid[$1] = $2
          }
          END {
            for (pid in ppid) {
              cur = pid
              while (cur in ppid) {
                if (cur == root) {
                  print pid
                  break
                }
                cur = ppid[cur]
              }
            }
          }
        '
      )"
      if [[ -n "${tree_pids}" ]]; then
        kill -TERM ${tree_pids} 2>/dev/null || true
      fi
      kill -TERM "${cmd_pid}" 2>/dev/null || true
      wait "${cmd_pid}" 2>/dev/null || true
      return 137
    fi
    sleep "${MEMORY_CHECK_INTERVAL_SEC}"
  done

  wait "${cmd_pid}"
  local exit_code=$?
  local peak_gib
  peak_gib="$(awk -v kb="${peak_rss_kb}" 'BEGIN { printf "%.2f", kb/1024/1024 }')"
  echo "[MEMCAP] ${stage_name} peak_rss=${peak_gib}GiB"
  return "${exit_code}"
}

if [[ "${RUN_IN_BACKGROUND}" == "1" && "${VIRALM_BG_WORKER:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  ENV_VARS=(
    "RUN_IN_BACKGROUND=0"
    "VIRALM_BG_WORKER=1"
    "PYTHON_BIN=${PYTHON_BIN}"
    "TORCHRUN_BIN=${TORCHRUN_BIN}"
    "SOURCE_FASTA=${SOURCE_FASTA}"
    "SOURCE_DEV_FASTA=${SOURCE_DEV_FASTA}"
    "DATASET_DIR=${DATASET_DIR}"
    "MODEL_PATH=${MODEL_PATH}"
    "DATA_OUT=${DATA_OUT}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "SEED=${SEED}"
    "DEV_FRACTION=${DEV_FRACTION}"
    "MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
    "DATALOADER_WORKERS=${DATALOADER_WORKERS}"
    "TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE}"
    "TOKENIZE_NUM_PROC=${TOKENIZE_NUM_PROC}"
    "TOKENIZED_CACHE_DIR=${TOKENIZED_CACHE_DIR}"
    "TOKENIZED_CACHE_FORMAT=${TOKENIZED_CACHE_FORMAT}"
    "REBUILD_TOKENIZED_CACHE=${REBUILD_TOKENIZED_CACHE}"
    "SEQUENCE_EVAL_THRESHOLD=${SEQUENCE_EVAL_THRESHOLD}"
    "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
    "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
    "SMOKE_TEST=${SMOKE_TEST}"
    "SMOKE_SAMPLES_PER_CLASS=${SMOKE_SAMPLES_PER_CLASS}"
    "CSV_BUILD_WORKERS=${CSV_BUILD_WORKERS}"
    "CSV_BATCH_SIZE=${CSV_BATCH_SIZE}"
    "CSV_MAX_RECORDS=${CSV_MAX_RECORDS}"
    "CSV_MAX_RECORDS_PER_CLASS=${CSV_MAX_RECORDS_PER_CLASS}"
    "CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT=${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT}"
    "REUSE_PREPARED_DATA=${REUSE_PREPARED_DATA}"
    "FORCE_REBUILD_DATA=${FORCE_REBUILD_DATA}"
    "FORCE_SINGLE_GPU=${FORCE_SINGLE_GPU}"
    "TARGET_GPU=${TARGET_GPU}"
    "USE_TORCHRUN=${USE_TORCHRUN}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    "DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"
    "USE_TRAIN_BATCH_PLAN=${USE_TRAIN_BATCH_PLAN}"
    "TRAIN_BATCH_PLAN_PATH=${TRAIN_BATCH_PLAN_PATH}"
    "GROUP_BY_LENGTH=${GROUP_BY_LENGTH}"
    "MEMORY_LIMIT_GB=${MEMORY_LIMIT_GB}"
    "MEMORY_CHECK_INTERVAL_SEC=${MEMORY_CHECK_INTERVAL_SEC}"
    "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
    "LOGGING_STEPS=${LOGGING_STEPS}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "SAVE_STEPS=${SAVE_STEPS}"
    "RUN_NAME=${RUN_NAME}"
    "LOG_DIR=${LOG_DIR}"
    "LOG_FILE=${LOG_FILE}"
  )
  TMUX_CMD=(env "${ENV_VARS[@]}" bash "$0" "$@")
  printf -v TMUX_CMD_STR '%q ' "${TMUX_CMD[@]}"
  printf -v GAMIL_ROOT_Q '%q' "${GAMIL_ROOT}"
  printf -v LOG_FILE_Q '%q' "${LOG_FILE}"
  tmux new-session -d -s "${TMUX_SESSION}" "cd ${GAMIL_ROOT_Q} && ${TMUX_CMD_STR} > ${LOG_FILE_Q} 2>&1"
  echo "Background training started in tmux."
  echo "  session: ${TMUX_SESSION}"
  echo "  log: ${LOG_FILE}"
  echo "Attach: tmux attach -t ${TMUX_SESSION}"
  echo "Tail: tail -f ${LOG_FILE}"
  exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${USE_TORCHRUN}" == "1" ]] && ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 && [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not executable: ${TORCHRUN_BIN}" >&2
  exit 1
fi

if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"
fi
export TOKENIZERS_PARALLELISM="false"

if [[ "${USE_TORCHRUN}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
fi

mkdir -p "${DATA_OUT}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GAMIL root: ${GAMIL_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Model: ${MODEL_PATH}"
echo "Source FASTA: ${SOURCE_FASTA}"
if [[ -n "${SOURCE_DEV_FASTA}" ]]; then
  echo "Source dev FASTA: ${SOURCE_DEV_FASTA}"
  echo "CSV split mode: fixed train/dev FASTA inputs (no hash dev split)"
else
  echo "CSV split mode: legacy genome-hash dev split from Source FASTA"
fi
echo "CSV output: ${DATA_OUT}"
echo "Checkpoint output: ${OUTPUT_DIR}"
echo "CSV build: workers=${CSV_BUILD_WORKERS}, batch_size=${CSV_BATCH_SIZE}, dev_fraction=${DEV_FRACTION}, max_records=${CSV_MAX_RECORDS}, max_per_class=${CSV_MAX_RECORDS_PER_CLASS}, max_per_class_split=${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT}"
echo "Training knobs: batch_size=${BATCH_SIZE}, eval_batch_size=${EVAL_BATCH_SIZE}, model_max_length=${MODEL_MAX_LENGTH}, dataloader_workers=${DATALOADER_WORKERS}, tokenizer_batch=${TOKENIZER_BATCH_SIZE}, tokenize_num_proc=${TOKENIZE_NUM_PROC}, fp16=True, ddp_find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS_BOOL}"
echo "Token cache: dir=${TOKENIZED_CACHE_DIR}, format=${TOKENIZED_CACHE_FORMAT}, rebuild=${REBUILD_TOKENIZED_CACHE_BOOL}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
echo "GPU mode: force_single_gpu=${FORCE_SINGLE_GPU}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, use_torchrun=${USE_TORCHRUN}, nproc_per_node=${NPROC_PER_NODE}"
echo "Batch plan: use=${USE_TRAIN_BATCH_PLAN}, path=${TRAIN_BATCH_PLAN_PATH:-none}, group_by_length=${GROUP_BY_LENGTH}"

TRAIN_CSV="${DATA_OUT}/train.csv"
DEV_CSV="${DATA_OUT}/dev.csv"
SUMMARY_JSON="${DATA_OUT}/summary.json"

if [[ "${REUSE_PREPARED_DATA}" == "1" && "${FORCE_REBUILD_DATA}" != "1" && -f "${TRAIN_CSV}" && -f "${DEV_CSV}" ]]; then
  echo "[1/2] Reusing existing CSV files: ${TRAIN_CSV}, ${DEV_CSV}"
  if [[ -f "${SUMMARY_JSON}" ]]; then
    echo "[1/2] Existing summary: ${SUMMARY_JSON}"
  fi
else
  echo "[1/2] Preparing binary train/dev CSVs from FASTA..."
  if [[ -n "${SOURCE_DEV_FASTA}" ]]; then
    PREP_CMD=(
      "${PYTHON_BIN}"
      "${SCRIPT_DIR}/prepare_realm_rank_csv.py"
      --train-fasta "${SOURCE_FASTA}"
      --dev-fasta "${SOURCE_DEV_FASTA}"
      --output-dir "${DATA_OUT}"
      --num-workers "${CSV_BUILD_WORKERS}"
      --batch-size "${CSV_BATCH_SIZE}"
    )
  else
    PREP_CMD=(
      "${PYTHON_BIN}"
      "${SCRIPT_DIR}/prepare_realm_rank_csv.py"
      --input-fasta "${SOURCE_FASTA}"
      --output-dir "${DATA_OUT}"
      --seed "${SEED}"
      --dev-fraction "${DEV_FRACTION}"
      --num-workers "${CSV_BUILD_WORKERS}"
      --batch-size "${CSV_BATCH_SIZE}"
    )
  fi
  if [[ "${CSV_MAX_RECORDS}" -gt 0 ]]; then
    PREP_CMD+=(--max-records "${CSV_MAX_RECORDS}")
  fi
  if [[ "${CSV_MAX_RECORDS_PER_CLASS}" -gt 0 ]]; then
    PREP_CMD+=(--max-records-per-class "${CSV_MAX_RECORDS_PER_CLASS}")
  fi
  if [[ "${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT}" -gt 0 ]]; then
    PREP_CMD+=(--max-records-per-class-per-split "${CSV_MAX_RECORDS_PER_CLASS_PER_SPLIT}")
  fi
  run_with_memory_cap "prepare_csv" "${PREP_CMD[@]}"
fi

if [[ "${USE_TORCHRUN}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  TRAIN_LAUNCH_CMD=(
    "${TORCHRUN_BIN}"
    --standalone
    --nnodes 1
    --nproc_per_node "${NPROC_PER_NODE}"
    "${SCRIPT_DIR}/train.py"
  )
else
  TRAIN_LAUNCH_CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/train.py")
fi

echo "[2/2] Training binary virus/non-virus classifier with Adam + BCE (lr=1e-5, warmup=50)..."
TRAIN_CMD=(
  "${TRAIN_LAUNCH_CMD[@]}"
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_OUT}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --learning_rate 1e-5
  --warmup_steps 50
  --per_device_train_batch_size "${BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps 1
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --evaluation_strategy epoch
  --eval_strategy epoch
  --save_strategy epoch
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --load_best_model_at_end True
  --metric_for_best_model f1
  --greater_is_better True
  --fp16 True
  --dataloader_pin_memory True
  --dataloader_num_workers "${DATALOADER_WORKERS}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS_BOOL}"
  --tokenizer_batch_size "${TOKENIZER_BATCH_SIZE}"
  --tokenize_num_proc "${TOKENIZE_NUM_PROC}"
  --tokenized_cache_dir "${TOKENIZED_CACHE_DIR}"
  --tokenized_cache_format "${TOKENIZED_CACHE_FORMAT}"
  --rebuild_tokenized_cache "${REBUILD_TOKENIZED_CACHE_BOOL}"
  --sequence_eval_threshold "${SEQUENCE_EVAL_THRESHOLD}"
  --require_cuda True
  --seed "${SEED}"
  --report_to none
  --save_model True
  --eval_and_save_results True
)

if [[ "${USE_TRAIN_BATCH_PLAN}" == "1" ]]; then
  TRAIN_CMD+=(--group_by_length False --train_batch_plan_path "${TRAIN_BATCH_PLAN_PATH}")
else
  TRAIN_CMD+=(--group_by_length "$(to_bool_str "${GROUP_BY_LENGTH}")")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

run_with_memory_cap "train" "${TRAIN_CMD[@]}"

echo "Done. Output dir: ${OUTPUT_DIR}"
