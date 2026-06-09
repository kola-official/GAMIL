# Script Inventory

This document describes the purpose of each top-level directory and each retained runnable script. Ignored local checkpoint/data indexes are not listed as repository scripts.

## Directory Overview

| Directory | Purpose |
| --- | --- |
| `raw_data/` | Raw-source preparation notes, manifests, and small verification summaries. Large raw files stay local-only. |
| `process_data/` | Realm-Rank v4 dataset construction, FASTA-to-CSV conversion, and v4 test-suite preparation. |
| `processed_data/realm_rank_v4/` | Final strict v4 dataset manifest, ignored FASTA links, and QC/result summaries. |
| `train/` | User-facing teacher training and six-model pipeline launchers. |
| `distill/` | User-facing KD and MIL distillation entrypoints. These point to canonical six-model code through symlinks. |
| `benchmark/` | Flash-attention inference, metrics, six-model benchmarks, bio-attention analysis, and efficiency evaluation. |
| `model/` | Canonical six-model source code, model staging helpers, and local model manifest. |
| `checkpoint/` | Manifest and ignored local indexes for selected best checkpoints. |
| `docs/` | Reproduction, hardware, deduplication, verification, and inventory documentation. |

## Raw Data Scripts

| Script | Purpose |
| --- | --- |
| `raw_data/scripts/prepare_insect_data.py` | Prepare strict Insecta raw-source records used by Realm-Rank v4 and write small verification summaries. |

## Process Data Scripts

| Script | Purpose |
| --- | --- |
| `process_data/scripts/build_realm_rank_dataset_v4.py` | Build the final strict Realm-Rank v4 dataset from approved raw source roots. |
| `process_data/scripts/realm_rank_dataset_common.py` | Shared helper module used by v4 builders; kept so v4 scripts do not depend on an old v3 entrypoint. |
| `process_data/scripts/prepare_realm_rank_csv.py` | Convert Realm-Rank v4 FASTA splits into CSV files used by training and distillation. |
| `process_data/scripts/build_realm_rank_test_v4.py` | Build the final Realm-Rank v4 test FASTA and metadata for benchmark evaluation. |

## Train Scripts

| Script | Purpose |
| --- | --- |
| `train/scripts/train.py` | Train the ViraLM-R teacher model on Realm-Rank v4 data. |
| `train/scripts/train.sh` | Shell launcher for teacher training with environment-variable overrides. |
| `train/scripts/run_six_model_pipeline.sh` | User-facing launcher for the final six-model Realm-Rank v4 pipeline; symlink to canonical six-model code. |

## Distill Scripts

| Script | Purpose |
| --- | --- |
| `distill/scripts/train_6l_meanpool_kd.py` | Train the 6-layer mean-pool KD student models; symlink to canonical six-model code. |
| `distill/scripts/train_6l_gated_mil_kd.py` | Train the 6-layer gated-attention MIL KD student models; symlink to canonical six-model code. |
| `distill/scripts/mil_training_common.py` | Shared MIL training utilities used by gated MIL scripts; symlink helper. |
| `distill/scripts/six_model_shared.py` | Shared data/model utilities used by six-model training and distillation; symlink helper. |

## Benchmark Scripts

| Script | Purpose |
| --- | --- |
| `benchmark/scripts/run_viralm_flash_inference.py` | Flash-attention adapted ViraLM/ViraLM-MIL inference runner. Supports sharded FASTA inference, MIL models, warmup, and `--require-flash-attn`. This supersedes older generic `viralm.py` copies. |
| `benchmark/scripts/metrics.py` | Compute euk/pro O-vs-R sequence and fragment metrics, including AUC summaries. |
| `benchmark/scripts/run_realm_rank_v4_six_model_benchmark.py` | Run the final Realm-Rank v4 six-model benchmark; symlink to canonical six-model code. |
| `benchmark/scripts/run_euk_pro_v4_six_model_benchmark.py` | Run the final euk/pro v4 six-model benchmark; symlink to canonical six-model code. |
| `benchmark/scripts/run_v4_bio_attention_analysis.py` | Run resumable biological attention analysis for Realm-Rank v4 test records. |
| `benchmark/scripts/build_efficiency_dataset.py` | Build the local FASTA input used by efficiency/runtime benchmarks. |
| `benchmark/scripts/collect_efficiency_summary.py` | Collect per-run efficiency outputs into compact summary CSV/JSON files. |
| `benchmark/scripts/monitor_run.py` | Monitor runtime, GPU, and process resource usage for efficiency experiments. |
| `benchmark/scripts/run_viralm_r_gated_mil_single.sh` | Launch the selected ViraLM-R/O and gated-MIL efficiency comparison using the flash inference runner. |

## Model Scripts And Code

| Script | Purpose |
| --- | --- |
| `model/scripts/stage_euk_pro_reference_models.py` | Stage euk/pro ViraLM-O and ViraLM-R teacher reference model files; symlink to canonical six-model helper. |
| `model/scripts/init_6l_student_models.py` | Initialize 6-layer student models from staged teacher references; symlink to canonical six-model helper. |
| `model/code/realm_rank_v4_six_model/experiment_config.py` | Central path, model-name, and required-file defaults for the six-model experiment. |
| `model/code/realm_rank_v4_six_model/shared.py` | Shared model loading, data loading, tokenization, and metric helpers. |
| `model/code/realm_rank_v4_six_model/mil_model.py` | Final gated-attention MIL model wrapper. |
| `model/code/realm_rank_v4_six_model/mil_train_common.py` | Shared MIL training routines. |
| `model/code/realm_rank_v4_six_model/train_meanpool_kd.py` | Canonical 6-layer mean-pool KD training implementation. |
| `model/code/realm_rank_v4_six_model/train_gated_mil_kd.py` | Canonical 6-layer gated-attention MIL KD training implementation. |
| `model/code/realm_rank_v4_six_model/train_gated_mil_supervised.py` | Canonical 12-layer supervised gated-attention MIL training implementation. |
| `model/code/realm_rank_v4_six_model/run_realm_rank_v4_six_model_tmux.sh` | Canonical full six-model pipeline launcher. |
| `model/code/realm_rank_v4_six_model/benchmark_realm_rank_v4_test.py` | Canonical Realm-Rank v4 six-model benchmark implementation. |
| `model/code/realm_rank_v4_six_model/benchmark_euk_pro_v4_six_models.py` | Canonical euk/pro v4 six-model benchmark implementation. |
| `model/code/realm_rank_v4_six_model/copy_euk_pro_references.py` | Canonical helper for staging euk/pro reference metric files. |
| `model/code/realm_rank_v4_six_model/init_six_layer_students.py` | Canonical helper for initializing 6-layer student checkpoints. |

## Naming And Deduplication

- Category directories (`train/`, `distill/`, `benchmark/`, `model/scripts/`) expose user-facing names that describe the operation.
- Canonical six-model implementations remain under `model/code/realm_rank_v4_six_model/`.
- Symlinks are used for category entrypoints that call canonical six-model code, so the repository does not carry duplicate script copies.
- The flash-attention inference script is intentionally named `run_viralm_flash_inference.py`; older `viralm.py` paths are excluded to avoid ambiguity.
