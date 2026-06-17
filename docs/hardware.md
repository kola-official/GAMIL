# Hardware Notes

GAMIL inference can run on CPU for small smoke tests, but CUDA is recommended
for full benchmark reproduction and training.

## Recommended Benchmark Setup

- GPU: one CUDA-capable GPU with enough memory for the selected batch size.
- CPU: 8 or more cores for data loading and metric computation.
- Memory: 32 GB RAM or more for full benchmark runs.
- Disk: at least 15 GB free space for extracted assets and generated outputs.

## Recommended Training Setup

- GPU: two CUDA-capable GPUs for the default training launcher.
- CPU: 16 or more cores.
- Memory: 64 GB RAM or more.
- Disk: additional space for checkpoints, logs, and prediction outputs.

## Useful Overrides

```bash
export GAMIL_ROOT="$PWD"
export PROCESSED_DATA_ROOT="$GAMIL_ROOT/processed_data"
export CHECKPOINT_ROOT="$GAMIL_ROOT/checkpoint/local_checkpoints"
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
```

For a single-GPU smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 RUN_IN_TMUX=0 SMOKE_TEST=1 \
  train/scripts/run_six_model_pipeline.sh
```

If CUDA is unavailable, use `scripts/quick_start.sh --mode smoke` first. CPU
execution is mainly for validation and will be slower.

## Verified Local Setup

The current release was smoke-tested on this machine:

- Linux 6.14.0-36-generic x86_64
- 2 x Intel Xeon Gold 6226R
- 2 x NVIDIA GeForce RTX 3090, 24 GB each
- NVIDIA driver 570.169
- Python 3.8.18
- PyTorch 2.0.1 with CUDA 11.8

Validated runs:

- fresh `conda env create -f environment.yml`
- `scripts/quick_start.sh --mode prepare`
- CPU smoke benchmark
- single-GPU smoke benchmark
