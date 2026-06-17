#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/quick_start.sh --asset-dir DIR [--mode prepare|smoke|full] [--output-dir DIR]

Options:
  --asset-dir DIR     Directory containing the release archives and SHA256SUMS.
  --mode MODE         prepare, smoke, or full. Default: smoke.
  --output-dir DIR    Output directory. Default: outputs/quick_start.
  --python BIN        Python executable. Default: python.
  --device DEVICE     Benchmark device. Default: auto.
  --max-records N     Records per smoke benchmark. Default: 64.
  -h, --help          Show this help message.

Modes:
  prepare             Verify archives, extract assets, and run code checks.
  smoke               prepare + small benchmark subset.
  full                prepare + full benchmark suite.
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${GAMIL_ASSET_DIR:-}"
MODE="smoke"
OUTPUT_DIR="${ROOT_DIR}/outputs/quick_start"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
elif command -v conda >/dev/null 2>&1 && conda run -n vl python -c "import einops, datasets, transformers" >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n vl python)
else
  PYTHON_CMD=(python)
fi
DEVICE="${DEVICE:-auto}"
MAX_RECORDS="${MAX_RECORDS:-64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-dir)
      ASSET_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max-records)
      MAX_RECORDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  prepare|smoke|full)
    ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$ASSET_DIR" ]]; then
  echo "Missing --asset-dir. Provide the directory containing the release archives." >&2
  exit 2
fi

ASSET_DIR="$(cd "$ASSET_DIR" && pwd)"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

ARCHIVES=(
  "gamil_core_data_v1.tar.zst"
  "gamil_euk_pro_benchmark_v1.tar.zst"
  "gamil_model_weights_v1.tar.zst"
)

for archive in "${ARCHIVES[@]}"; do
  if [[ ! -f "${ASSET_DIR}/${archive}" ]]; then
    echo "Missing release archive: ${ASSET_DIR}/${archive}" >&2
    exit 1
  fi
done

if [[ -f "${ASSET_DIR}/SHA256SUMS" ]]; then
  (cd "$ASSET_DIR" && sha256sum -c SHA256SUMS)
else
  echo "Warning: ${ASSET_DIR}/SHA256SUMS not found; skipping checksum verification." >&2
fi

for archive in "${ARCHIVES[@]}"; do
  echo "Extracting ${archive}"
  tar --zstd -xf "${ASSET_DIR}/${archive}" -C "$ROOT_DIR"
done

export GAMIL_ROOT="$ROOT_DIR"
export PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${ROOT_DIR}/processed_data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoint/local_checkpoints}"

echo "Running Python syntax checks"
"${PYTHON_CMD[@]}" -c "import sys; print(sys.version)" >/dev/null
(cd "$ROOT_DIR" && "${PYTHON_CMD[@]}" -m py_compile $(find raw_data process_data train benchmark model/code -type f -name '*.py'))

"${PYTHON_CMD[@]}" "$ROOT_DIR/benchmark/scripts/run_viralm_flash_inference.py" --help >/dev/null

if [[ "$MODE" == "prepare" ]]; then
  echo "Prepare mode completed."
  exit 0
fi

DEVICE_ARGS=()
if [[ "$DEVICE" != "auto" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

CORE_BENCH_SCRIPT="${ROOT_DIR}/benchmark/scripts/run_realm_rank_benchmark.py"
HOST_BENCH_SCRIPT="${ROOT_DIR}/benchmark/scripts/run_euk_pro_benchmark.py"
MODEL_ROOT="$(find "$CHECKPOINT_ROOT" -maxdepth 3 -type d -path '*/models' | sort | head -n 1)"
CORE_TEST_FASTA="$(find "$PROCESSED_DATA_ROOT" -maxdepth 3 -type f -name 'test.fasta.gz' | sort | head -n 1)"
HOST_BENCH_ROOT="${PROCESSED_DATA_ROOT}/realm_rank_benchmark"
if [[ "$DEVICE" == "cpu" ]]; then
  SMOKE_MODEL="$(find "$MODEL_ROOT" -maxdepth 1 -type d -name '*meanpool_kd' | sort | tail -n 1 | xargs -r basename)"
else
  SMOKE_MODEL="$(find "$MODEL_ROOT" -maxdepth 1 -type d -name '*12l_gated_mil' | sort | tail -n 1 | xargs -r basename)"
fi

if [[ -z "$SMOKE_MODEL" ]]; then
  SMOKE_MODEL="$(find "$MODEL_ROOT" -maxdepth 1 -type d -name '*gated_mil*' | sort | tail -n 1 | xargs -r basename)"
fi

if [[ ! -e "$CORE_BENCH_SCRIPT" || ! -e "$HOST_BENCH_SCRIPT" || -z "$MODEL_ROOT" || -z "$CORE_TEST_FASTA" || ! -d "$HOST_BENCH_ROOT" || -z "$SMOKE_MODEL" ]]; then
  echo "Could not locate extracted data, model weights, or benchmark scripts." >&2
  echo "Run prepare mode with a complete release asset directory first." >&2
  exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
  echo "Running smoke benchmark"
  "${PYTHON_CMD[@]}" "$CORE_BENCH_SCRIPT" \
    --test-fasta "$CORE_TEST_FASTA" \
    --model-root "$MODEL_ROOT" \
    --output-dir "${OUTPUT_DIR}/core_smoke" \
    --models "$SMOKE_MODEL" \
    --max-records "$MAX_RECORDS" \
    --batch-size 8 \
    --mil-batch-size 2 \
    --dataloader-workers 0 \
    --fp16 False \
    --allow-missing-models \
    "${DEVICE_ARGS[@]}"

  "${PYTHON_CMD[@]}" "$HOST_BENCH_SCRIPT" \
    --data-root "$HOST_BENCH_ROOT" \
    --model-root "$MODEL_ROOT" \
    --output-dir "${OUTPUT_DIR}/host_smoke" \
    --models "$SMOKE_MODEL" \
    --benchmarks bench-pro \
    --lengths 500 \
    --max-records "$MAX_RECORDS" \
    --batch-size 8 \
    --mil-batch-size 2 \
    --dataloader-workers 0 \
    --fp16 False \
    --overwrite \
    --no-combined-predictions \
    "${DEVICE_ARGS[@]}"

  echo "Smoke benchmark completed. Outputs: ${OUTPUT_DIR}"
  exit 0
fi

echo "Running full benchmark"
"${PYTHON_CMD[@]}" "$CORE_BENCH_SCRIPT" \
  --test-fasta "$CORE_TEST_FASTA" \
  --model-root "$MODEL_ROOT" \
  --output-dir "${OUTPUT_DIR}/core_full" \
  "${DEVICE_ARGS[@]}"

"${PYTHON_CMD[@]}" "$HOST_BENCH_SCRIPT" \
  --data-root "$HOST_BENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-dir "${OUTPUT_DIR}/host_full" \
  --overwrite \
  "${DEVICE_ARGS[@]}"

echo "Full benchmark completed. Outputs: ${OUTPUT_DIR}"
