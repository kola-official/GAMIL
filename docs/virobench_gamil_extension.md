# ViroBench-GAMIL classification extension

This extension adds aggregation-focused classification experiments for ViroBench without modifying the upstream ViroBench runner.

## What It Tests

The experiment compares five sequence-level aggregation strategies on ViroBench classification tasks:

| Method | Input | Aggregation | Purpose |
|---|---|---|---|
| `VB-Default` | window logits | mean window logits | Matches ViroBench's native window-level MLP + sequence-level mean-logit aggregation. |
| `PostHoc-Quantile` | window probabilities | per-class high quantile | Diagnostic probe mirroring the original GAMIL quantile experiment. |
| `PostHoc-NoisyOR` | window probabilities | per-class noisy-OR | Diagnostic probe for strong localized evidence. |
| `PostHoc-LogReg` | window probability summaries | logistic regression | Diagnostic probe for whether window-score distributions contain information beyond the mean. |
| `GAMIL` | window embeddings | gated attention + sequence MLP | Main learnable aggregation method. |

Taxonomy tasks are handled as five parallel multiclass heads: `kingdom`, `phylum`, `class`, `order`, and `family`.
Host tasks use one multiclass head: `host_label`.

## Server Layout

The implementation runs from the GAMIL repository root:

```bash
cd /path/to/GAMIL
PY=python
```

Expected paths:

```text
external/ViroBench/                         # upstream ViroBench source
external/ViroBench/data/all_viral/cls_data/ # ViroBench classification data
external/model_weight/DNABERT-2-117M/       # default DNABERT-2 model path
results/virobench_gamil/                    # output root
```

If the model weights are elsewhere, pass `--model-dir`.

## Public Data and Model Assets

ViroBench source, data, and baseline backbones are public assets and are not
redistributed in this repository or in the GAMIL Zenodo release. Prepare them in
the local checkout before running the extension.

Install a downloader if needed:

```bash
python -m pip install -U "huggingface_hub[cli]"
```

Fetch the upstream ViroBench source:

```bash
mkdir -p external
git clone https://github.com/SII-AGI4S/ViroBench external/ViroBench
```

Download the ViroBench dataset from Hugging Face and copy the public
classification splits into the layout expected by the runner:

```bash
mkdir -p external/ViroBench/hf_data
huggingface-cli download YDXX/ViroBench \
  --repo-type dataset \
  --local-dir external/ViroBench/hf_data \
  --local-dir-use-symlinks False

mkdir -p external/ViroBench/data/all_viral/cls_data
rsync -a external/ViroBench/hf_data/Classification/ \
  external/ViroBench/data/all_viral/cls_data/

mkdir -p external/ViroBench/data/all_viral/cls_data_min_consistent
rsync -a external/ViroBench/hf_data/ViroBench-CLS-Lite/ \
  external/ViroBench/data/all_viral/cls_data_min_consistent/
```

The extension uses only the classification tasks. The ViroBench generation data
can stay in the downloaded Hugging Face cache unless you also run upstream
generation diagnostics.

Download model weights into repo-local ignored folders:

```bash
mkdir -p external/model_weight

huggingface-cli download zhihan1996/DNABERT-2-117M \
  --local-dir external/model_weight/DNABERT-2-117M \
  --local-dir-use-symlinks False

huggingface-cli download LucaGroup/LucaVirus-default-step3.8M \
  --local-dir external/model_weight/LucaVirus-default-step3.8M \
  --local-dir-use-symlinks False

huggingface-cli download YDXX/ViroHyena-253m \
  --local-dir external/ViroBench/pretrain/hyena-dna/ViroHyena-253m \
  --local-dir-use-symlinks False
```

OmniReg-GPT needs its public code plus checkpoint/tokenizer assets:

```bash
mkdir -p external/official external/model_weight
git clone https://github.com/wawpaopao/OmniReg-GPT external/official/OmniReg-GPT

# Place the public OmniReg-GPT checkpoint at:
#   external/model_weight/OmniReg-GPT/pytorch_model.bin
# and its tokenizer directory at:
#   external/model_weight/OmniReg-GPT/gena-lm-bert-large-t2t
```

If you keep OmniReg assets elsewhere, set:

```bash
export OMNIREG_REPO_DIR=/path/to/OmniReg-GPT
export OMNIREG_ASSET_DIR=/path/to/OmniReg-GPT-assets
```

## Smoke Test

Use the smallest feasible task first. This command checks data loading, embedding extraction, window MLP, post hoc probes, and GAMIL:

```bash
$PY scripts/run_virobench_gamil.py \
  --dataset-name ALL-host-genus \
  --model-name DNABERT2-virobench \
  --model-dir /path/to/DNABERT-2-117M \
  --window-len 2048 \
  --train-num-windows 2 \
  --eval-num-windows 8 \
  --epochs 3 \
  --patience 2 \
  --output-dir results/virobench_gamil_smoke
```

The smoke test intentionally uses few epochs and a small evaluation-window cap. Do not report smoke-test values in the paper.

## Pilot Runs

The first real pilot should use full validation/test window coverage:

```bash
$PY scripts/run_virobench_gamil.py \
  --dataset-name ALL-host-genus \
  --model-name DNABERT2-virobench \
  --model-dir /path/to/DNABERT-2-117M \
  --window-len 2048 \
  --train-num-windows 2 \
  --eval-num-windows -1 \
  --epochs 80 \
  --patience 12 \
  --output-dir results/virobench_gamil
```

Core four scenarios:

```bash
for task in ALL-host-genus ALL-host-times ALL-taxon-genus ALL-taxon-times; do
  $PY scripts/run_virobench_gamil.py \
    --dataset-name "$task" \
    --model-name DNABERT2-virobench \
    --model-dir /path/to/DNABERT-2-117M \
    --window-len 2048 \
    --train-num-windows 2 \
    --eval-num-windows -1 \
    --epochs 80 \
    --patience 12 \
    --output-dir results/virobench_gamil
done
```

For a formal three-seed run:

```bash
for seed in 42 2025 3407; do
  for task in ALL-host-genus ALL-host-times ALL-taxon-genus ALL-taxon-times; do
    $PY scripts/run_virobench_gamil.py \
      --dataset-name "$task" \
      --model-name DNABERT2-virobench \
      --model-dir /path/to/DNABERT-2-117M \
      --window-len 2048 \
      --train-num-windows 2 \
      --eval-num-windows -1 \
      --epochs 80 \
      --patience 12 \
      --seed "$seed" \
      --output-dir results/virobench_gamil
  done
done
```

Alternate public backbones can be launched with the wrapper script. Use
`MODEL_DIR` when the weights are not in the default repo-local path:

```bash
MODEL=LucaVirus-default-step3.8M \
MODEL_DIR=external/model_weight/LucaVirus-default-step3.8M \
GPU_INDEX=0 \
bash scripts/run_virobench_models_234.sh

MODEL=ViroHyena-253m \
MODEL_DIR=external/ViroBench/pretrain/hyena-dna/ViroHyena-253m \
GPU_INDEX=0 \
bash scripts/run_virobench_models_234.sh

MODEL=OmniReg-GPT \
OMNIREG_REPO_DIR=external/official/OmniReg-GPT \
OMNIREG_ASSET_DIR=external/model_weight/OmniReg-GPT \
GPU_INDEX=0 \
bash scripts/run_virobench_models_234.sh
```

## Outputs

Each run writes:

```text
summary.json                    # metrics for all reported methods
window_mlp_best.pt              # VB-default window-level MLP
window_logits_test.npz          # raw test window logits and sequence groups
gamil_best.pt                   # GAMIL sequence-level head
gamil_test_predictions.npz
gamil_test_attention.pt         # attention weights by sequence group
embeddings/*.pt                 # cached window embeddings
```

Post hoc methods are diagnostics. They should be reported as evidence that window-score distributions contain information beyond fixed mean aggregation, not as the final deployable model.

## Interpretation Guardrails

- `PostHoc-LogReg ~= GAMIL`: suggests most gain is already present in window-score distributions; frame GAMIL as deployable and diagnostic rather than uniquely stronger.
- No claim should be made about ViroBench generation tasks.
- Attention weights are aggregation diagnostics, not validated functional annotations.
