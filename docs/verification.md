# Verification

Last verified on 2026-06-09.

## Passed

- `python -m py_compile` over retained Python sources.
- Exclusion grep only hits `README.md` and `docs/deduplication.md` exclusion notes.
- Source path grep over `.py`, `.sh`, and `README.md` has no machine-private absolute paths.
- Key/notification grep has no notification-key hits.
- Core `--help` checks pass for raw/process scripts, inference, metrics, v4 benchmarks, distillation scripts, and bio-attention.
- `train/scripts/train.py --help` passes in the existing `vl` Python environment.
- `environment.yml` is parseable YAML, has exact package pins exported from `vl`, and has no machine-local Conda `prefix`.
- All file-level symlinks in `checkpoint/local_checkpoints` resolve to existing local checkpoint files.
- `git status --ignored` shows local FASTA/data/cache paths ignored.
- `git ls-files --others --exclude-standard` exposes no model weights, FASTA data, Arrow cache, NumPy arrays, SQLite files, or checkpoint binaries.

## Environment Note

The system `python` is 3.13.9 and does not include `datasets`, so `python train/scripts/train.py --help` fails in that interpreter. Use `PYTHON_BIN` or a GAMIL environment with `datasets` installed.

On the current login node, PyTorch may emit a CUDA driver-version warning during `--help` imports. The CLI checks still exit successfully; training should be run on a node whose driver matches the selected PyTorch/CUDA build.
