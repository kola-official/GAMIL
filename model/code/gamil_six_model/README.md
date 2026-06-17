# GAMIL Six-Model Pipeline

This directory contains the model training, distillation, and benchmark
implementations used by GAMIL. New users should start from the repository-level
quick-start script; this directory is mainly for advanced training and debugging.

## Full Training

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 train/scripts/run_six_model_pipeline.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 RUN_IN_TMUX=0 SMOKE_TEST=1 \
  train/scripts/run_six_model_pipeline.sh
```

## Outputs

Mean-pool models are saved as Hugging Face checkpoints. MIL models save
`best_mil_model.pt` plus tokenizer, config, and custom model code files.

Benchmark runs write prediction CSV files, metric tables, and summary JSON files
under the selected output directory.
