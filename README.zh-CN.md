# GAMIL

GAMIL 是一个面向基因组级病毒序列识别的模型框架，结合了
transformer 序列编码器和 gated-attention multiple-instance learning。
本仓库包含复现实验所需的代码、轻量级 benchmark 摘要和发布清单；
大型序列文件与训练好的模型权重通过独立发布资产分发。

英文说明见 `README.md`。

## 许可证

GAMIL 使用 GNU General Public License v3.0 发布。完整许可证文本见
`LICENSE`。

## 仓库结构

```text
GAMIL/
  raw_data/          原始来源数据准备工具和小型摘要
  process_data/      数据集构建与 FASTA-to-CSV 工具
  processed_data/    数据 manifest 与已提交的 QC 摘要
  train/             训练启动脚本
  distill/           知识蒸馏与 MIL 训练入口
  benchmark/         推理、指标计算和 benchmark 脚本
  model/             模型代码与模型 manifest
  checkpoint/        选中 checkpoint 的 manifest
  docs/              用户文档
  scripts/           一键安装、检查和复现脚本
```

## Quick Start

quick-start 脚本会校验发布资产、解压到预期目录、运行基础代码检查，并启动一个小规模 benchmark smoke test；如果本机有可用的 `vl` Conda 环境，会优先使用它。

```bash
git clone https://github.com/kola-official/GAMIL.git
cd GAMIL

conda env create -f environment.yml
conda activate gamil

bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

smoke test 通过后，可以使用 `--mode full` 运行完整 benchmark。完整流程更适合在 GPU 环境中运行，耗时会明显更长。
如果想强制使用其他解释器，可以额外传入 `--python /path/to/python`。

## 发布资产

运行 quick start 前，请从项目发布记录下载全部文件：

- Zenodo 记录页：https://zenodo.org/records/20725522
- DOI：https://doi.org/10.5281/zenodo.20725522

| 文件 | 用途 |
| --- | --- |
| `gamil_core_data_v1.tar.zst` | 训练、验证、测试序列以及训练表格数据 |
| `gamil_euk_pro_benchmark_v1.tar.zst` | 固定长度的真核和原核 benchmark FASTA 文件 |
| `gamil_model_weights_v1.tar.zst` | base encoder、teacher 模型和最终 GAMIL 模型权重 |
| `SHA256SUMS` | quick-start 脚本用于校验文件完整性 |

使用 `wget` 下载：

```bash
mkdir -p gamil_release_assets
cd gamil_release_assets

wget -O gamil_core_data_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_core_data_v1.tar.zst?download=1
wget -O gamil_euk_pro_benchmark_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_euk_pro_benchmark_v1.tar.zst?download=1
wget -O gamil_model_weights_v1.tar.zst https://zenodo.org/records/20725522/files/gamil_model_weights_v1.tar.zst?download=1
wget -O SHA256SUMS https://zenodo.org/records/20725522/files/SHA256SUMS?download=1
```

也可以手动解压：

```bash
tar --zstd -xf gamil_core_data_v1.tar.zst -C .
tar --zstd -xf gamil_euk_pro_benchmark_v1.tar.zst -C .
tar --zstd -xf gamil_model_weights_v1.tar.zst -C .
```

## 常用命令

只解压资产并运行代码检查：

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode prepare
```

运行 smoke benchmark：

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

运行完整 benchmark：

```bash
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode full
```

训练、蒸馏、硬件配置和手动 benchmark 命令见 `docs/reproduction.md` 与
`docs/hardware.md`。

## ViroBench 扩展

本仓库新增了 ViroBench 分类扩展，用于比较 ViroBench 默认的窗口均值
聚合、post hoc 聚合探针以及 GAMIL gated-attention 聚合：

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

相关脚本包括：

- `scripts/run_virobench_gamil.py`
- `scripts/run_virobench_gamil_core4.sh`
- `scripts/run_virobench_models_234.sh`
- `scripts/summarize_virobench_gamil.py`
- `scripts/diagnose_virobench_gamil.py`

ViroBench 源码、分类数据和对比 backbone 权重都是公开上游资产，因此
不随本仓库打包，也不重复上传到 Zenodo。请在本地准备：

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

runner 默认使用以下本地路径：

| 模型 | 公开来源 | 本地路径 |
| --- | --- | --- |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` | `external/model_weight/DNABERT-2-117M` |
| LucaVirus | `LucaGroup/LucaVirus-default-step3.8M` | `external/model_weight/LucaVirus-default-step3.8M` |
| ViroHyena | `YDXX/ViroHyena-253m` | `external/ViroBench/pretrain/hyena-dna/ViroHyena-253m` |
| OmniReg-GPT | `wawpaopao/OmniReg-GPT` 及其公开权重/tokenizer | `external/official/OmniReg-GPT` 与 `external/model_weight/OmniReg-GPT` |

完整数据布局、模型下载命令、smoke test 和三随机种子运行示例见
`docs/virobench_gamil_extension.md`。

## 环境

推荐使用 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate gamil
```

常用环境变量：

```bash
export GAMIL_ROOT="$PWD"
export PROCESSED_DATA_ROOT="$GAMIL_ROOT/processed_data"
export CHECKPOINT_ROOT="$GAMIL_ROOT/checkpoint/local_checkpoints"
export PYTHON_BIN=python
export TORCHRUN_BIN=torchrun
```

## 验证

安装依赖并解压发布资产后，运行：

```bash
python -m py_compile $(find raw_data process_data train benchmark model/code -type f -name '*.py')
bash scripts/quick_start.sh --asset-dir /path/to/gamil_release_assets --mode smoke
```

smoke test 默认把输出写入 `outputs/quick_start/`。

## 已验证环境

本仓库在以下本地环境中完成了实际验证：

- 操作系统：Linux 6.14.0-36-generic x86_64
- CPU：2 x Intel Xeon Gold 6226R
- GPU：2 x NVIDIA GeForce RTX 3090（每张 24 GB）
- NVIDIA 驱动：570.169
- Conda 环境：`gamil-clean-test`
- Python：3.8.18
- PyTorch：2.0.1
- PyTorch 使用的 CUDA runtime：11.8

已验证流程：

- 全新 `conda env create -f environment.yml`
- `quick_start.sh --mode prepare`
- CPU smoke benchmark
- 单张 RTX 3090 的 GPU smoke benchmark

## 常见问题

- `ImportError: No module named 'einops'`：重新用 `environment.yml` 创建 Conda 环境，不要混用系统 Python。
- `conda env create` 很慢：这是完整环境，首次求解和安装本来就会比较慢。
- 出现 `CUDA initialization` 驱动警告：`--help` 或 smoke 仍可能成功；但完整 GPU 训练需要和当前 PyTorch/CUDA 兼容的更新驱动。
- `Missing release archive`：确认三个 `.tar.zst` 和 `SHA256SUMS` 都在传给 `--asset-dir` 的同一目录下。

## 引用

如果在研究中使用 GAMIL，请引用配套论文，并同时引用数据和模型
资产 DOI：

- `10.5281/zenodo.20725522`
