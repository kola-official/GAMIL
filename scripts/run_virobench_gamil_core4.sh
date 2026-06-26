#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
MODEL_DIR="${MODEL_DIR:-}"
OUT="${OUT:-results/virobench_gamil}"
WINDOW_LEN="${WINDOW_LEN:-2048}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-2}"
EVAL_WINDOWS="${EVAL_WINDOWS:--1}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
SEEDS="${SEEDS:-42}"
TASKS="${TASKS:-ALL-host-genus ALL-host-times ALL-taxon-genus ALL-taxon-times}"
METHODS="${METHODS:-all}"
LOGREG_C_GRID="${LOGREG_C_GRID:-0.1,0.3,1.0,3.0}"
DEVICE="${DEVICE:-cuda:0}"
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-8}"
WINDOW_HEAD_BATCH_SIZE="${WINDOW_HEAD_BATCH_SIZE:-256}"
BAG_BATCH_SIZE="${BAG_BATCH_SIZE:-16}"
MAX_TRAIN_SEQUENCES="${MAX_TRAIN_SEQUENCES:-0}"
MAX_VAL_SEQUENCES="${MAX_VAL_SEQUENCES:-0}"
MAX_TEST_SEQUENCES="${MAX_TEST_SEQUENCES:-0}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR is required, e.g. MODEL_DIR=/path/to/DNABERT-2-117M $0" >&2
  exit 2
fi

for seed in ${SEEDS}; do
  for task in ${TASKS}; do
    "${PY}" scripts/run_virobench_gamil.py \
      --dataset-name "${task}" \
      --model-name DNABERT2-virobench \
      --model-dir "${MODEL_DIR}" \
      --window-len "${WINDOW_LEN}" \
      --train-num-windows "${TRAIN_WINDOWS}" \
      --eval-num-windows "${EVAL_WINDOWS}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${seed}" \
      --methods "${METHODS}" \
      --logreg-c-grid "${LOGREG_C_GRID}" \
      --device "${DEVICE}" \
      --emb-batch-size "${EMB_BATCH_SIZE}" \
      --window-head-batch-size "${WINDOW_HEAD_BATCH_SIZE}" \
      --bag-batch-size "${BAG_BATCH_SIZE}" \
      --max-train-sequences "${MAX_TRAIN_SEQUENCES}" \
      --max-val-sequences "${MAX_VAL_SEQUENCES}" \
      --max-test-sequences "${MAX_TEST_SEQUENCES}" \
      --output-dir "${OUT}"
  done
done
