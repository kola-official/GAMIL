# Command Reference

This repository exposes a small set of user-facing commands. Most users only
need `scripts/quick_start.sh`.

## Setup And Reproduction

| Command | Purpose |
| --- | --- |
| `scripts/quick_start.sh` | Verify release assets, extract data and weights, run code checks, and launch benchmark smoke/full runs. |

## Data Preparation

| Command | Purpose |
| --- | --- |
| `raw_data/scripts/prepare_insect_data.py` | Prepare Insecta source summaries used by the dataset builder. |
| `process_data/scripts/prepare_realm_rank_csv.py` | Convert FASTA splits into CSV files for training and distillation. |

The released processed data are sufficient for benchmark reproduction. Rebuild
raw data only when regenerating the dataset from public source records.

## Training And Distillation

| Command | Purpose |
| --- | --- |
| `train/scripts/train.sh` | Train a teacher model with environment-variable overrides. |
| `train/scripts/run_six_model_pipeline.sh` | Run the full teacher/student training pipeline. |
| `distill/scripts/train_6l_meanpool_kd.py` | Train 6-layer mean-pool KD students. |
| `distill/scripts/train_6l_gated_mil_kd.py` | Train 6-layer gated-attention MIL KD students. |

## Benchmarking

| Command | Purpose |
| --- | --- |
| `benchmark/scripts/run_viralm_flash_inference.py` | Run FASTA inference with a released model directory. |
| `benchmark/scripts/metrics.py` | Compute sequence-level and fragment-level benchmark metrics. |
| `benchmark/scripts/collect_efficiency_summary.py` | Combine runtime-monitor outputs into compact CSV/JSON summaries. |

The quick-start script calls the benchmark entrypoints with the correct default
paths and is the recommended interface for new users.
