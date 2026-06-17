# Release Assets

Large data files and model weights are distributed outside Git. Download all
files from the project release record before running the quick start.

## Files

| File | Contents |
| --- | --- |
| `gamil_core_data_v1.tar.zst` | Processed train/validation/test sequences and training CSV files |
| `gamil_euk_pro_benchmark_v1.tar.zst` | Fixed-length eukaryotic and prokaryotic benchmark FASTA files with metadata and QC summaries |
| `gamil_model_weights_v1.tar.zst` | Base encoder, teacher models, and final GAMIL model weights |
| `SHA256SUMS` | SHA256 checksums for the archives |

## Extraction

The recommended extraction path is:

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode prepare
```

Manual extraction:

```bash
tar --zstd -xf gamil_core_data_v1.tar.zst -C /path/to/GAMIL
tar --zstd -xf gamil_euk_pro_benchmark_v1.tar.zst -C /path/to/GAMIL
tar --zstd -xf gamil_model_weights_v1.tar.zst -C /path/to/GAMIL
```

## Excluded Files

The release assets omit intermediate work directories, optimizer states, runtime
logs, and full prediction tables. These files are regenerated locally when
running training or benchmark commands.

## Upload Helpers

Two helper scripts are included for publishing the prepared archives:

```bash
python scripts/upload_hf.py --repo-id <user_or_org>/<repo> --source-dir /path/to/archives
python scripts/upload_zenodo.py --source-dir /path/to/archives --metadata-json release/zenodo_metadata.example.json
```

Environment variables:

- `HF_TOKEN` for Hugging Face uploads.
- `ZENODO_TOKEN` for Zenodo uploads.

Zenodo publishing additionally requires metadata suitable for publication.
