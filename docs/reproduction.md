# Reproduction

Run commands from the repository root unless noted otherwise.

## Environment

`environment.yml` is pinned from the current local `vl` Conda environment with package versions and pip pins, but without the machine-local `prefix`.

```bash
conda env create -f environment.yml
conda activate gamil
```

## Data Build

```bash
python process_data/scripts/build_realm_rank_dataset_v4.py --help
python process_data/scripts/build_realm_rank_dataset_v4.py --input-root "$RAW_DATA_ROOT" --output-dir "$PROCESSED_DATA_ROOT/realm_rank_v4"
python process_data/scripts/prepare_realm_rank_csv.py --help
python process_data/scripts/prepare_realm_rank_csv.py --input-fasta processed_data/realm_rank_v4/train.fasta.gz --dev-fasta processed_data/realm_rank_v4/dev.fasta.gz --output-dir processed_data/realm_rank_v4/train_csv
python process_data/scripts/build_realm_rank_test_v4.py --help
```

## Train And Distill

```bash
bash train/scripts/train.sh
bash train/scripts/run_six_model_pipeline.sh
python distill/scripts/train_6l_meanpool_kd.py --help
python distill/scripts/train_6l_gated_mil_kd.py --help
```

## Benchmark

```bash
python benchmark/scripts/run_viralm_flash_inference.py --help
python benchmark/scripts/metrics.py --help
python benchmark/scripts/run_realm_rank_v4_six_model_benchmark.py --help
python benchmark/scripts/run_euk_pro_v4_six_model_benchmark.py --help
python benchmark/scripts/run_v4_bio_attention_analysis.py --help
bash benchmark/scripts/run_viralm_r_gated_mil_single.sh
```

The checked-in result files under `processed_data/realm_rank_v4/results` and `benchmark/results` are summaries only. Full prediction tables, FASTA files, tokenized caches, and model weights are resolved through manifests and ignored symlinks.

The checkpoint manifest indexes five local final/staged models. When a source directory contains multiple checkpoints, `checkpoint/local_checkpoints/<entry>/` keeps only the selected best checkpoint or best MIL weight.
