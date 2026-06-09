# Hardware And Runtime

The final runs were designed for CUDA execution but the scripts expose CPU/smoke paths for validation.

Recommended overrides:

```bash
export GAMIL_ROOT="$PWD"
export RAW_DATA_ROOT="$GAMIL_ROOT/raw_data/local_sources"
export PROCESSED_DATA_ROOT="$GAMIL_ROOT/processed_data"
export CHECKPOINT_ROOT="$GAMIL_ROOT/checkpoint/local_checkpoints"
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
```

For single-GPU or smoke validation:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 RUN_IN_TMUX=0 SMOKE_TEST=1 \
  train/scripts/run_six_model_pipeline.sh
```

External tools used by data construction and bio-annotation can be supplied with `BLASTN_BIN`, `MAKEBLASTDB_BIN`, `BIOANN_BIN`, and `BIOANN_PYTHON`.
