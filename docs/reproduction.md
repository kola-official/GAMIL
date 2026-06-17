# Reproduction Guide

This guide describes how to reproduce the released benchmarks from a fresh
checkout. Commands are run from the repository root.

## 1. Create The Environment

```bash
conda env create -f environment.yml
conda activate gamil
```

If Conda is not available, create an equivalent Python environment with the
packages listed in `environment.yml`.

## 2. Download The Release Assets

Place the release files in one directory:

```text
gamil_release_assets/
  gamil_core_data_v1.tar.zst
  gamil_euk_pro_benchmark_v1.tar.zst
  gamil_model_weights_v1.tar.zst
  SHA256SUMS
```

The quick-start script verifies the checksums and extracts the archives into the
repository. If the intended interpreter is not the active `python`, pass it with
`--python /path/to/python`.
When available, it will use the existing `vl` Conda environment automatically.

## 3. Run The Smoke Test

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

The smoke test runs on a small subset of records and is intended to confirm that
the environment, data, and weights are connected correctly.

## 4. Run The Full Benchmark

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode full
```

The full benchmark writes results under `outputs/quick_start/` unless
`--output-dir` is provided. Use a CUDA-capable machine for practical runtime.

## 5. Train Or Distill Models

The released weights are sufficient for benchmark reproduction. To run the
training pipeline from the prepared data:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 train/scripts/run_six_model_pipeline.sh
```

For a small validation run:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 RUN_IN_TMUX=0 SMOKE_TEST=1 \
  train/scripts/run_six_model_pipeline.sh
```

Main environment overrides:

```bash
export GAMIL_ROOT="$PWD"
export PROCESSED_DATA_ROOT="$GAMIL_ROOT/processed_data"
export CHECKPOINT_ROOT="$GAMIL_ROOT/checkpoint/local_checkpoints"
export OUTPUT_ROOT="$CHECKPOINT_ROOT/gamil_six_model"
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
```

## 6. Outputs

The benchmark scripts write predictions, metric tables, and summary JSON files
to the selected output directory. Checked-in files under `benchmark/results/`
are compact summaries for reference; large prediction tables are generated
locally.
