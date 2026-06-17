# Validation

Use these checks after installing dependencies and extracting the release
assets.

## Code Checks

```bash
python -m py_compile $(find raw_data process_data train benchmark model/code -type f -name '*.py')
python benchmark/scripts/run_viralm_flash_inference.py --help
python benchmark/scripts/metrics.py --help
```

## Asset Checks

```bash
cd /path/to/gamil_release_assets
sha256sum -c SHA256SUMS
```

The quick-start script runs the checksum check automatically when
`--asset-dir` contains `SHA256SUMS`.

## Smoke Benchmark

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

Successful completion confirms that the environment can import the code, the
release assets are in place, and at least one model can run inference on a small
subset of the benchmark data.
