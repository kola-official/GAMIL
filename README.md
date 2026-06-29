# GAMIL

GAMIL is a genome-level viral sequence identification framework built on
transformer sequence encoders and gated-attention multiple-instance learning.
This repository contains the code, lightweight benchmark summaries, and release
manifests needed to reproduce the reported experiments. Large sequence files
and trained weights are distributed separately as release assets.

Chinese documentation is available in `README.zh-CN.md`.

## Repository Layout

```text
GAMIL/
  raw_data/          raw-source preparation utilities and small summaries
  process_data/      dataset construction and FASTA-to-CSV utilities
  processed_data/    dataset manifests and checked-in QC summaries
  train/             training launchers
  distill/           knowledge-distillation and MIL training entrypoints
  benchmark/         inference, metrics, and benchmark scripts
  model/             model code and model manifests
  checkpoint/        selected-checkpoint manifest
  docs/              user documentation
  scripts/           one-command setup and reproduction helpers
```

## Quick Start

The quick-start script verifies the release assets, extracts them into the
expected folders, runs basic code checks, and launches a small benchmark smoke
test. It will use the existing `vl` Conda environment when available.

```bash
git clone https://github.com/kola-official/GAMIL.git
cd GAMIL

conda env create -f environment.yml
conda activate gamil

bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

Use `--mode full` to run the full benchmark suite after the smoke test succeeds.
The full run is GPU-oriented and may take substantially longer.
If you want to force a different interpreter, pass `--python /path/to/python`.

## Release Assets

Download all release files from the project release record before running the
quick start:

- Zenodo record: https://zenodo.org/records/20725522
- DOI: https://doi.org/10.5281/zenodo.20725522

| File | Purpose |
| --- | --- |
| `gamil_core_data_v1.tar.zst` | Training, validation, and test sequences plus tabular training data |
| `gamil_euk_pro_benchmark_v1.tar.zst` | Fixed-length eukaryotic and prokaryotic benchmark FASTA files |
| `gamil_model_weights_v1.tar.zst` | Base encoder, teacher models, and final GAMIL model weights |
| `SHA256SUMS` | Checksums used by the quick-start script |

Download with `wget`:

```bash
mkdir -p gamil_release_assets
cd gamil_release_assets

wget -O gamil_core_data_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_core_data_v1.tar.zst?download=1
wget -O gamil_euk_pro_benchmark_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_euk_pro_benchmark_v1.tar.zst?download=1
wget -O gamil_model_weights_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_model_weights_v1.tar.zst?download=1
wget -O SHA256SUMS https://zenodo.org/records/20725522/files/SHA256SUMS?download=1
```

Manual extraction is also supported:

```bash
tar --zstd -xf gamil_core_data_v1.tar.zst -C .
tar --zstd -xf gamil_euk_pro_benchmark_v1.tar.zst -C .
tar --zstd -xf gamil_model_weights_v1.tar.zst -C .
```

## Common Tasks

Run only asset extraction and code checks:

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode prepare
```

Run a smoke benchmark:

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

Run the full benchmark:

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode full
```

For training, distillation, hardware notes, and manual benchmark commands, see
`docs/reproduction.md` and `docs/hardware.md`.

## ViroBench Extension

This repository also includes a ViroBench classification extension that compares
ViroBench's default window-mean aggregation with post hoc aggregation probes,
and GAMIL gated-attention aggregation:

```bash
python scripts/run_virobench_gamil.py \
  --dataset-name ALL-host-genus \
  --model-name DNABERT2-virobench \
  --model-dir external/model_weight/DNABERT-2-117M \
  --window-len 2048 \
  --train-num-windows 2 \
  --eval-num-windows -1 \
  --epochs 80 \
  --patience 12 \
  --output-dir results/virobench_gamil
```

Reusable helpers are provided in:

- `scripts/run_virobench_gamil.py`
- `scripts/run_virobench_gamil_core4.sh`
- `scripts/run_virobench_models_234.sh`
- `scripts/summarize_virobench_gamil.py`
- `scripts/diagnose_virobench_gamil.py`

The ViroBench source, classification data, and compared backbone weights are
public upstream assets, so they are not bundled in this repository or uploaded
again to Zenodo. Prepare them locally with:

```bash
mkdir -p external
git clone https://github.com/SII-AGI4S/ViroBench external/ViroBench

python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download YDXX/ViroBench \
  --repo-type dataset \
  --local-dir external/ViroBench/hf_data \
  --local-dir-use-symlinks False

mkdir -p external/ViroBench/data/all_viral/cls_data
rsync -a external/ViroBench/hf_data/Classification/ \
  external/ViroBench/data/all_viral/cls_data/
```

Default model locations used by the runner:

| Model | Public source | Local path |
| --- | --- | --- |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` | `external/model_weight/DNABERT-2-117M` |
| LucaVirus | `LucaGroup/LucaVirus-default-step3.8M` | `external/model_weight/LucaVirus-default-step3.8M` |
| ViroHyena | `YDXX/ViroHyena-253m` | `external/ViroBench/pretrain/hyena-dna/ViroHyena-253m` |
| OmniReg-GPT | `wawpaopao/OmniReg-GPT` plus public weights/tokenizer | `external/official/OmniReg-GPT` and `external/model_weight/OmniReg-GPT` |

See `docs/virobench_gamil_extension.md` for the full data layout, model
download commands, smoke tests, and three-seed run examples.

## Environment

The recommended environment is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate gamil
```

Common environment variables:

```bash
export GAMIL_ROOT="$PWD"
export PROCESSED_DATA_ROOT="$GAMIL_ROOT/processed_data"
export CHECKPOINT_ROOT="$GAMIL_ROOT/checkpoint/local_checkpoints"
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
```

## Validation

After installing dependencies and extracting the release assets, run:

```bash
python -m py_compile $(find raw_data process_data train benchmark model/code -type f -name '*.py')
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

The smoke test writes outputs under `outputs/quick_start/` by default.

## Tested Environment

The repository was validated on the following local environment:

- OS: Linux 6.14.0-36-generic x86_64
- CPU: 2 x Intel Xeon Gold 6226R
- GPU: 2 x NVIDIA GeForce RTX 3090 (24 GB each)
- NVIDIA driver: 570.169
- Conda environment: `gamil-clean-test`
- Python: 3.8.18
- PyTorch: 2.0.1
- CUDA runtime used by PyTorch: 11.8

Validated workflows:

- Fresh `conda env create -f environment.yml`
- `quick_start.sh --mode prepare`
- CPU smoke benchmark
- GPU smoke benchmark on one RTX 3090

## Troubleshooting

- `ImportError: No module named 'einops'`: recreate the Conda environment from
  `environment.yml` and do not mix it with a partial system Python install.
- `conda env create` takes a long time: this file pins a full environment; the
  first solve/install can be slow.
- `CUDA initialization` driver warnings during `--help` or smoke runs: the
  command can still succeed on CPU, but full GPU training needs a newer driver
  that matches the installed PyTorch/CUDA build.
- `Missing release archive`: verify that all three `.tar.zst` files and
  `SHA256SUMS` are in the same directory passed to `--asset-dir`.

## Citation

If you use GAMIL in your work, cite the accompanying paper and the Zenodo
release DOI for the data/model assets:

- `10.5281/zenodo.20725522`
