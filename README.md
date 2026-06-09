# GAMIL

GAMIL is the trimmed Realm-Rank v4 workspace for the final ViraLM-R/GAMIL experiments. The repository body keeps lightweight source code, build instructions, result summaries, and manifests. Large local data, model weights, and checkpoints are indexed through manifest rows, with only file-level symlinks kept in-tree where they do not cause recursive scans.

## Scope

- Main dataset: `processed_data/realm_rank_v4` only. This is the strict final dataset.
- Excluded datasets and code paths: older Realm-Rank v2/v3 intermediates, ViraLM official text datasets, old ViraLM-MIL/disViralm scripts, and all IMGVR/Prodigal benchmark/data paths.
- Uploaded content: source scripts, documentation, manifests, and small metrics/QC summaries.
- Local-only content: FASTA/CSV training data, token caches, model weights, checkpoint directories, and large prediction files.

## Layout

```text
GAMIL/
  raw_data/          raw-source preparation scripts, summaries, and local source manifest
  process_data/      Realm-Rank v4 builders and CSV/test-suite preparation scripts
  processed_data/    Realm-Rank v4 summary results plus local file links
  train/             ViraLM-R training launcher and script
  distill/           symlinked final KD/MIL distillation scripts
  benchmark/         final inference, v4 benchmark, bio-attention, and efficiency scripts/results
  model/             final six-model code and local model symlink manifest
  checkpoint/        local checkpoint manifest
  docs/              hardware, deduplication, and reproduction notes
```

## Directory Roles

- `raw_data/`: local raw-source index and source-preparation scripts; no raw IMGVR/Prodigal data is included.
- `process_data/`: scripts that build the strict Realm-Rank v4 dataset and test suite.
- `processed_data/realm_rank_v4/`: v4 QC summaries, manifests, and ignored local FASTA links.
- `train/`: teacher training entrypoints and the user-facing six-model pipeline launcher.
- `distill/`: user-facing KD/MIL distillation entrypoints, symlinked to canonical six-model code.
- `benchmark/`: flash-attention inference, metrics, six-model benchmark, bio-attention, and efficiency evaluation.
- `model/`: canonical six-model model/training/benchmark source code plus model manifests.
- `checkpoint/`: manifest and ignored local indexes for the five selected final/staged checkpoints.
- `docs/`: hardware, reproduction, deduplication, verification, and script inventory notes.

For script-level details, see `docs/script_inventory.md`.

## Environment Variables

All runnable scripts are based on these roots and can be overridden:

- `GAMIL_ROOT`: defaults to this repository root.
- `RAW_DATA_ROOT`: defaults to `raw_data/local_sources`.
- `PROCESSED_DATA_ROOT`: defaults to `processed_data`.
- `MODEL_ROOT`: defaults to model/checkpoint roots depending on script context.
- `CHECKPOINT_ROOT`: defaults to `checkpoint/local_checkpoints`.
- `OUTPUT_ROOT`: defaults to a local output/checkpoint directory.
- `PYTHON_BIN`, `TORCHRUN_BIN`, `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`: launcher overrides.

## Quick Checks

```bash
python -m py_compile $(find raw_data process_data train benchmark model/code -type f -name '*.py')
python process_data/scripts/build_realm_rank_dataset_v4.py --help
python process_data/scripts/prepare_realm_rank_csv.py --help
python process_data/scripts/build_realm_rank_test_v4.py --help
python benchmark/scripts/run_viralm_flash_inference.py --help
python benchmark/scripts/run_realm_rank_v4_six_model_benchmark.py --help
```

See `docs/reproduction.md` for full command examples.
