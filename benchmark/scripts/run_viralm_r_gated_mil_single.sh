#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAMIL_ROOT="${GAMIL_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${GAMIL_ROOT}/model/local_models}"
DATA_DIR="${BENCHMARK_DATA_ROOT:-${GAMIL_ROOT}/benchmark/local_data}"
RUN_ROOT="${OUTPUT_ROOT:-${GAMIL_ROOT}/outputs}/efficiency_runs"

CONDA_SH="${CONDA_SH:-}"
VL_ENV="${VL_ENV:-vl}"
GPU_ID="${GPU_ID:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1}"
SMOKE="${SMOKE:-0}"
SMOKE_RECORD_START="${SMOKE_RECORD_START:-0}"
SMOKE_RECORD_END="${SMOKE_RECORD_END:-2}"
MODELS="${MODELS:-all}"

VIRALM_SCRIPT="${VIRALM_SCRIPT:-${GAMIL_ROOT}/benchmark/scripts/run_viralm_flash_inference.py}"
INPUT_FASTA="${DATA_DIR}/efficiency_10000bp_5000.fasta"

RUN_PREFIX="${RUN_PREFIX:-viralm_o_r_gated_mil}"
if [[ "${SMOKE}" == "1" ]]; then
  RUN_PREFIX="${RUN_PREFIX}_smoke"
fi
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${RUN_ROOT}/${RUN_NAME}"
RESULT_DIR="${OUT_ROOT}/results"
LOG_DIR="${OUT_ROOT}/logs"
METRIC_DIR="${OUT_ROOT}/metrics"
STATUS_DIR="${OUT_ROOT}/status"

MODEL_NAMES=(
  "viralm-o"
  "viralm-r"
  "viralm-r-12l-gated-mil"
  "viralm-r-6l-gated-mil"
  "viralm-o-6l-gated-mil"
)
MODEL_DIRS=(
  "${MODEL_ROOT}/staged_models/viralm-o"
  "${MODEL_ROOT}/staged_models/viralm-r"
  "${MODEL_ROOT}/viralm_r_12l_gated_mil"
  "${MODEL_ROOT}/viralm_r_6l_gated_mil_kd"
  "${MODEL_ROOT}/viralm_o_6l_gated_mil_kd"
)
MODEL_IS_MIL=(0 0 1 1 1)

COMMON_ARGS=(
  --input "${INPUT_FASTA}"
  --filename "efficiency"
  --len 500
  --fragment-len 2000
  --min-tail-len 500
  --threshold 0.5
  --batch_size 16
  --dataloader_workers 4
  --prefetch_factor 4
  --model-max-length 512
  --infer-fp16 1
  --threads 4
  --warmup-batches 1
  --require-flash-attn
  --force
)

if [[ "${SMOKE}" == "1" ]]; then
  COMMON_ARGS+=(--record-start "${SMOKE_RECORD_START}" --record-end "${SMOKE_RECORD_END}")
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/run_viralm_r_gated_mil_single.log"
}

activate_env() {
  local env_name="$1"
  if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
    return 0
  fi
  set +u
  source "${CONDA_SH}"
  conda activate "${env_name}"
  set -u
}

validate_inputs() {
  [[ -f "${INPUT_FASTA}" ]] || { log "[ERROR] missing input FASTA: ${INPUT_FASTA}"; exit 2; }
  [[ -f "${VIRALM_SCRIPT}" ]] || { log "[ERROR] missing ViraLM-R script: ${VIRALM_SCRIPT}"; exit 2; }
  [[ -f "${SCRIPT_DIR}/monitor_run.py" ]] || { log "[ERROR] missing monitor_run.py"; exit 2; }
  local model_dir
  for model_dir in "${MODEL_DIRS[@]}"; do
    [[ -d "${model_dir}" ]] || { log "[ERROR] missing model directory: ${model_dir}"; exit 2; }
  done
}

model_selected() {
  local name="$1"
  if [[ "${MODELS}" == "all" ]]; then
    return 0
  fi
  local item
  for item in ${MODELS}; do
    if [[ "${item}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_outputs() {
  local name="$1"
  local out_dir="$2"
  local result_csv="${out_dir}/result_efficiency.csv"
  local fragment_csv="${out_dir}/fragment_result_efficiency.csv"
  local run_info_json="${out_dir}/run_info_efficiency.json"

  local missing=0
  for path in "${result_csv}" "${fragment_csv}" "${run_info_json}"; do
    if [[ ! -f "${path}" ]]; then
      log "[ERROR] ${name} missing output: ${path}"
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 4
  fi

  python - "${run_info_json}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
if int(info.get("sequence_results", 0)) <= 0:
    raise SystemExit("sequence_results is empty")
if int(info.get("fragment_results", 0)) <= 0:
    raise SystemExit("fragment_results is empty")
PY
}

run_model() {
  local idx="$1"
  local name="${MODEL_NAMES[${idx}]}"
  local model_dir="${MODEL_DIRS[${idx}]}"
  local is_mil="${MODEL_IS_MIL[${idx}]}"
  local out_dir="${RESULT_DIR}/${name}"
  local sample_csv="${METRIC_DIR}/samples/${name}.csv"
  local summary_json="${METRIC_DIR}/summary/${name}.json"
  local stdout_log="${LOG_DIR}/${name}.stdout.log"
  local stderr_log="${LOG_DIR}/${name}.stderr.log"
  local done_file="${STATUS_DIR}/${name}.done"
  local fail_file="${STATUS_DIR}/${name}.failed"

  rm -rf "${out_dir}"
  rm -f "${sample_csv}" "${summary_json}" "${stdout_log}" "${stderr_log}" "${done_file}" "${fail_file}"
  mkdir -p "${out_dir}"

  local cmd=(
    python "${VIRALM_SCRIPT}"
    "${COMMON_ARGS[@]}"
    --output "${out_dir}"
    --database "${model_dir}"
  )
  if [[ "${is_mil}" == "1" ]]; then
    cmd+=(--mil-sub-chunk-size 16)
  fi

  log "[RUN] ${name}"
  set +e
  (
    activate_env "${VL_ENV}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export TOKENIZERS_PARALLELISM=false
    python "${SCRIPT_DIR}/monitor_run.py" \
      --name "${name}" \
      --metrics-csv "${sample_csv}" \
      --summary-json "${summary_json}" \
      --stdout "${stdout_log}" \
      --stderr "${stderr_log}" \
      --interval "${MONITOR_INTERVAL}" \
      -- "${cmd[@]}"
  )
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "${rc}" > "${fail_file}"
    log "[FAILED] ${name} rc=${rc}"
    return "${rc}"
  fi

  if ! validate_outputs "${name}" "${out_dir}"; then
    echo "validation_failed" > "${fail_file}"
    log "[FAILED] ${name} output validation failed"
    return 4
  fi

  touch "${done_file}"
  log "[DONE] ${name}"
}

collect_summary() {
  python - "${OUT_ROOT}" "${SELECTED_MODEL_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
model_names = sys.argv[2:]
rows = []

for name in model_names:
    summary_path = run_root / "metrics" / "summary" / f"{name}.json"
    out_dir = run_root / "results" / name
    result_csv = out_dir / "result_efficiency.csv"
    fragment_csv = out_dir / "fragment_result_efficiency.csv"
    run_info_json = out_dir / "run_info_efficiency.json"
    row = {
        "model": name,
        "return_code": "",
        "elapsed_sec": "",
        "max_rss_mb": "",
        "max_gpu_mem_mb": "",
        "max_cpu_seconds": "",
        "metrics_csv": str((run_root / "metrics" / "samples" / f"{name}.csv").resolve()),
        "stdout": str((run_root / "logs" / f"{name}.stdout.log").resolve()),
        "stderr": str((run_root / "logs" / f"{name}.stderr.log").resolve()),
        "result_csv": str(result_csv.resolve()),
        "fragment_result_csv": str(fragment_csv.resolve()),
        "run_info_json": str(run_info_json.resolve()),
        "sequence_results": "",
        "fragment_results": "",
        "output_valid": all(path.is_file() for path in [result_csv, fragment_csv, run_info_json]),
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for key in [
            "return_code",
            "elapsed_sec",
            "max_rss_mb",
            "max_gpu_mem_mb",
            "max_cpu_seconds",
            "metrics_csv",
            "stdout",
            "stderr",
        ]:
            row[key] = summary.get(key, row[key])
    if run_info_json.is_file():
        info = json.loads(run_info_json.read_text())
        row["sequence_results"] = info.get("sequence_results", "")
        row["fragment_results"] = info.get("fragment_results", "")
    rows.append(row)

out_csv = run_root / "final_summary.csv"
out_json = run_root / "final_summary.json"
fieldnames = [
    "model",
    "return_code",
    "elapsed_sec",
    "max_rss_mb",
    "max_gpu_mem_mb",
    "max_cpu_seconds",
    "metrics_csv",
    "stdout",
    "stderr",
    "result_csv",
    "fragment_result_csv",
    "run_info_json",
    "sequence_results",
    "fragment_results",
    "output_valid",
]
with out_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
out_json.write_text(json.dumps(rows, indent=2) + "\n")
print(f"final_summary_csv={out_csv}")
print(f"final_summary_json={out_json}")
PY
}

if [[ -e "${OUT_ROOT}" ]]; then
  echo "Run directory already exists: ${OUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${METRIC_DIR}/samples" "${METRIC_DIR}/summary" "${STATUS_DIR}"

log "RUN_DIR=${OUT_ROOT}"
log "INPUT_FASTA=${INPUT_FASTA}"
log "GPU_ID=${GPU_ID} SMOKE=${SMOKE} MODELS=${MODELS}"

validate_inputs

SELECTED_MODEL_INDICES=()
SELECTED_MODEL_NAMES=()
for idx in "${!MODEL_NAMES[@]}"; do
  if model_selected "${MODEL_NAMES[${idx}]}"; then
    SELECTED_MODEL_INDICES+=("${idx}")
    SELECTED_MODEL_NAMES+=("${MODEL_NAMES[${idx}]}")
  fi
done
if [[ "${#SELECTED_MODEL_INDICES[@]}" -eq 0 ]]; then
  log "[ERROR] no models selected"
  exit 2
fi

failed=0
for idx in "${SELECTED_MODEL_INDICES[@]}"; do
  set +e
  run_model "${idx}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    failed="${rc}"
    log "[STOP] stopping after ${MODEL_NAMES[${idx}]} failure"
    break
  fi
done

collect_summary | tee -a "${LOG_DIR}/run_viralm_r_gated_mil_single.log"

if [[ "${failed}" -ne 0 ]]; then
  exit "${failed}"
fi

log "[DONE] completed ${#SELECTED_MODEL_NAMES[@]} model runs"
