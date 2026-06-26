#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
SEEDS="${SEEDS:-42 2025 3407}"
TASKS="${TASKS:-ALL-host-genus ALL-host-times ALL-taxon-genus ALL-taxon-times DNA-host-genus DNA-host-times DNA-taxon-genus DNA-taxon-times RNA-host-genus RNA-host-times RNA-taxon-genus RNA-taxon-times}"
MODEL="${MODEL:?Set MODEL to OmniReg-GPT, LucaVirus-default-step3.8M, or ViroHyena-253m}"
MODEL_DIR="${MODEL_DIR:-}"
GPU_INDEX="${GPU_INDEX:-0}"
OUT_ROOT="${OUT_ROOT:-results/virobench_models_234}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
METHODS="${METHODS:-all}"
LOGREG_C_GRID="${LOGREG_C_GRID:-0.1,0.3,1.0,3.0}"
WINDOW_HEAD_BATCH_SIZE="${WINDOW_HEAD_BATCH_SIZE:-256}"
BAG_BATCH_SIZE="${BAG_BATCH_SIZE:-16}"

case "${MODEL}" in
  OmniReg-GPT|OmniReg-large|omnireg-gpt)
    WINDOW_LEN="${WINDOW_LEN:-2048}"
    TRAIN_WINDOWS="${TRAIN_WINDOWS:-2}"
    EVAL_WINDOWS="${EVAL_WINDOWS:-16}"
    EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-2}"
    MODEL_TAG="OmniReg-GPT"
    ;;
  LucaVirus-default-step3.8M|LucaVirus)
    WINDOW_LEN="${WINDOW_LEN:-512}"
    TRAIN_WINDOWS="${TRAIN_WINDOWS:-8}"
    EVAL_WINDOWS="${EVAL_WINDOWS:-64}"
    EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-4}"
    MODEL_TAG="LucaVirus-default-step3.8M"
    ;;
  ViroHyena-253m|ViroHyena-253M)
    WINDOW_LEN="${WINDOW_LEN:-2048}"
    TRAIN_WINDOWS="${TRAIN_WINDOWS:-2}"
    EVAL_WINDOWS="${EVAL_WINDOWS:-16}"
    EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-4}"
    MODEL_TAG="ViroHyena-253m"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}" results/logs_234
echo "[config] model=${MODEL_TAG} seeds=${SEEDS} window=${WINDOW_LEN}/${TRAIN_WINDOWS}/${EVAL_WINDOWS} gpu=${GPU_INDEX}"
echo "[config] tasks=${TASKS}"
MODEL_DIR_ARGS=()
if [[ -n "${MODEL_DIR}" ]]; then
  MODEL_DIR_ARGS=(--model-dir "${MODEL_DIR}")
fi

for seed in ${SEEDS}; do
  for task in ${TASKS}; do
    echo "[run] model=${MODEL_TAG} task=${task} seed=${seed} time=$(date '+%F %T')"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PY}" scripts/run_virobench_gamil.py \
      --dataset-name "${task}" \
      --model-name "${MODEL_TAG}" \
      "${MODEL_DIR_ARGS[@]}" \
      --window-len "${WINDOW_LEN}" \
      --train-num-windows "${TRAIN_WINDOWS}" \
      --eval-num-windows "${EVAL_WINDOWS}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${seed}" \
      --methods "${METHODS}" \
      --logreg-c-grid "${LOGREG_C_GRID}" \
      --device cuda:0 \
      --emb-batch-size "${EMB_BATCH_SIZE}" \
      --window-head-batch-size "${WINDOW_HEAD_BATCH_SIZE}" \
      --bag-batch-size "${BAG_BATCH_SIZE}" \
      --output-dir "${OUT_ROOT}"
  done
done

echo "[done] model=${MODEL_TAG} time=$(date '+%F %T')"
