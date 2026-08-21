# AdaWD-TS 预实验代码

本目录是 AdaWD-TS 论文的独立代码工作区。当前阶段对应
`../2027ICLR：动态宽深时序神经网络 (1).pdf` 第 1-4 页和第 12-17 页中的预实验。

预实验检验三个假设：

1. 局部状态更新需求 `U_i` 应当能够解释有效深度需求；
2. 局部模式多样性 `M_i` 应当能够解释有效宽度需求；
3. `U_i` 和 `M_i` 应分别解释不同的容量轴，而不是退化成同一个难度分数。

## 1. 当前实验范围

当前注册 9 个预测数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、ILI、ExchangeRate 和 Traffic。

当前可运行 5 个 Backbone：Crossformer、PatchTST、TimesNet、iTransformer 和 TimeMixer。

WPMMixer、TimeFilter 和 MultiPatchFormer 在 DropoutTS 中没有实现，目前只有缺失占位，不能启动实验。

> **Electricity 数据身份待确认**：当前代码按 DropoutTS 使用的 321 通道 LTSF Electricity
> 基准配置。论文草稿中的“家庭电参量及分表计量”对应另一套数据。正式下载和报告实验前必须选择其中一套，不能混用名称。

## 2. 目录结构

```text
AdaWD/
├── Baselines/                       # BasicTS 运行核心和 Backbone
│   ├── basicts/models/              # 5 个现有模型和 3 个缺失占位
│   └── registry.json                # 模型深度、宽度及 RAW 配置
├── datasets/
│   ├── catalog.json                 # 数据集名称、维度、频率和切分配置
│   ├── raw/                         # 需要手动放入的原始 CSV
│   └── processed/                   # prepare_dataset.py 生成的 NPY 文件
├── pre_experiments/
│   ├── config.json                  # 画像、候选容量、训练和效率配置
│   ├── results/                     # 实验输出
│   └── *.py                         # 各阶段入口脚本
├── src/adawd_preexp/                # 预实验核心实现
└── tests/                           # 自动测试
```

后续命令均假设当前目录为：

```bash
cd /home/devcontainers/ICLR/Exp/AdaWD
```

## 3. 安装运行环境

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training,test]"
```

如果机器需要指定 CUDA 版本，请先安装与本机 CUDA 匹配的 PyTorch，再执行最后一条安装命令。

检查环境：

```bash
python - <<'PY'
import numpy
import pandas
import scipy
import torch

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY

pytest -q
```

自动测试全部通过后再准备真实数据。

## 4. 下载和存放原始数据

### 4.1 一键下载（推荐）

项目提供了无需额外 Python 依赖的下载脚本。默认下载全部 9 个数据集，自动创建目录，
并在写入最终文件前校验 `date` 列、行数和特征列数：

```bash
./download_datasets.py
```

脚本支持断点续传；再次运行时会校验并跳过已有的完整文件。只下载部分数据集时使用：

```bash
./download_datasets.py --dataset ETTh1 Weather
```

默认依次尝试 THUML 官方 Hugging Face 仓库及其镜像，ETT 数据还会优先使用原始
ETDataset 仓库。网络中断后直接重新执行相同命令即可；使用 `--force` 可强制覆盖下载。

### 4.2 目录和文件名

原始数据必须按下面的结构存放：

```text
datasets/raw/
├── ETTh1/ETTh1.csv
├── ETTh2/ETTh2.csv
├── ETTm1/ETTm1.csv
├── ETTm2/ETTm2.csv
├── Weather/Weather.csv
├── Electricity/Electricity.csv
├── ILI/ILI.csv
├── ExchangeRate/ExchangeRate.csv
└── Traffic/Traffic.csv
```

以下备用文件名也可以识别：

- Weather：`weather.csv`；
- Electricity：`electricity.csv`；
- ILI：`Illness.csv` 或 `national_illness.csv`；
- ExchangeRate：`exchange_rate.csv`；
- Traffic：`traffic.csv`。

即使使用备用文件名，也应放在对应的标准目录下，例如 `datasets/raw/ILI/Illness.csv`。

### 4.3 CSV 格式要求

所有数据集均使用宽表 CSV：

```csv
date,feature_1,feature_2,feature_3
2016-07-01 00:00:00,1.2,3.4,5.6
2016-07-01 01:00:00,1.3,3.5,5.8
```

具体要求：

1. 第一行必须是列名；
2. 必须包含名为 `date` 的时间列；
3. 除 `date` 外的所有列必须能转换为数值；
4. 每一行是一个时间点，每一列是一个变量；
5. 数据必须按时间升序排列；
6. 不要额外保存 Pandas 行索引列，例如 `Unnamed: 0`；
7. 特征列数量应与下表一致。

| 数据集 | 原始文件 | 频率 | 预期时间点 | 特征列数 | 切分比例 |
| --- | --- | ---: | ---: | ---: | --- |
| ETTh1 | `ETTh1.csv` | 1 小时 | 17,420 | 7 | 6:2:2 |
| ETTh2 | `ETTh2.csv` | 1 小时 | 17,420 | 7 | 6:2:2 |
| ETTm1 | `ETTm1.csv` | 15 分钟 | 69,680 | 7 | 6:2:2 |
| ETTm2 | `ETTm2.csv` | 15 分钟 | 69,680 | 7 | 6:2:2 |
| Weather | `Weather.csv` | 10 分钟 | 52,696 | 21 | 7:1:2 |
| Electricity | `Electricity.csv` | 1 小时 | 26,304 | 321 | 7:1:2 |
| ILI | `ILI.csv` | 1 周 | 966 | 7 | 7:1:2 |
| ExchangeRate | `ExchangeRate.csv` | 1 天 | 7,588 | 8 | 7:1:2 |
| Traffic | `Traffic.csv` | 1 小时 | 17,544 | 862 | 7:1:2 |

“特征列数”不包含 `date` 列。权威配置位于 [`datasets/catalog.json`](datasets/catalog.json)。

### 4.4 下载后检查库存

```bash
python3 pre_experiments/check_inventory.py
```

数据存在时，对应状态应从 `missing_data` 变为 `raw_available`。

## 5. 数据预处理

处理单个数据集：

```bash
python3 pre_experiments/prepare_dataset.py \
  --dataset ETTh1 \
  --strict-shape
```

`--strict-shape` 会在时间点数量或特征数量不符合配置时直接报错。调试自定义数据时可以暂时去掉，正式实验建议保留。

批量处理 9 个数据集：

```bash
set -euo pipefail

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity ILI ExchangeRate Traffic; do
  python3 pre_experiments/prepare_dataset.py \
    --dataset "$dataset" \
    --strict-shape
done
```

每个数据集会生成：

```text
datasets/processed/ETTh1/
├── train_data.npy
├── val_data.npy
├── test_data.npy
├── train_timestamps.npy
├── val_timestamps.npy
├── test_timestamps.npy
└── meta.json
```

再次运行 `python3 pre_experiments/check_inventory.py`，状态应显示为 `processed_available`。

## 6. 数据特性预实验

该阶段计算论文定义的六个描述分量：

```text
U_i = mean(u_change, u_spectral, u_surprise)
M_i = mean(m_peak, m_band, m_channel)
```

### 6.1 画像单个数据集

默认优先读取完整原始 CSV：

```bash
python3 pre_experiments/profile_dataset.py --dataset ETTh1
```

如果希望严格按 train/validation/test 边界画像，读取处理后的 NPY：

```bash
python3 pre_experiments/profile_dataset.py \
  --dataset ETTh1 \
  --processed
```

限制窗口数的调试命令：

```bash
python3 pre_experiments/profile_dataset.py \
  --dataset ETTh1 \
  --window-size 96 \
  --stride 24 \
  --max-windows 100
```

Electricity 和 Traffic 默认均匀选择 64 个通道。正式敏感性实验可运行全通道版本：

```bash
python3 pre_experiments/profile_dataset.py \
  --dataset Electricity \
  --all-channels \
  --output-dir pre_experiments/results/profiles/Electricity_all_channels
```

### 6.2 批量画像

```bash
set -euo pipefail

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity ILI ExchangeRate Traffic; do
  python3 pre_experiments/profile_dataset.py --dataset "$dataset"
done
```

每个数据集生成：

```text
pre_experiments/results/profiles/ETTh1/
├── windows.csv      # 每个局部单元的六项分量、U、M 和分桶
└── summary.json     # 均值、标准差、P10/P50/P90、IQR 和 U/M 相关性
```

### 6.3 绘图

```bash
python3 pre_experiments/plot_profiles.py \
  --profiles pre_experiments/results/profiles/ETTh1/windows.csv
```

批量绘图：

```bash
set -euo pipefail

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity ILI ExchangeRate Traffic; do
  python3 pre_experiments/plot_profiles.py \
    --profiles "pre_experiments/results/profiles/${dataset}/windows.csv"
done
```

默认输出 `profile_diagnostics.pdf`。需要检查：

- `U` 或 `M` 是否集中在非常窄的区间；
- `U` 与 `M` 是否接近完全相关；
- low/mid/high 三个分桶是否都有足够样本；
- Q1-Q4 四类局部片段是否实际存在。

如果指标退化，应先调整画像窗口、步长或描述符超参数，不应直接开始全量模型训练。

## 7. 深度、宽度与 RAW 实验

### 7.1 当前默认配置

- 深度候选：`D={1,2,4,8}`；
- 宽度组候选：`W={1,2,4,8}`；
- 二维候选：`D={2,4,8} x W={2,4,8}`；
- 随机种子：`42,43,44`；
- 普通数据集：输入长度 96，预测长度 96/192/336/720；
- ILI：输入长度 24，预测长度 24/36/48/60；
- 训练指标：MAE、MSE、RMSE、MAPE、WAPE。

参数位于 [`pre_experiments/config.json`](pre_experiments/config.json)。四种轴的含义：

- `raw`：原始 Backbone 默认深度和宽度；
- `depth`：固定 RAW 宽度，只改变深度；
- `width`：固定 RAW 深度，只改变宽度；
- `joint`：运行二维深宽组合。

### 7.2 先检查计划，不训练

```bash
python3 pre_experiments/run_capacity_sweep.py \
  --dataset ETTh1 \
  --model PatchTST \
  --axis depth \
  --horizon 96 \
  --seeds 42 \
  --dry-run
```

`--dry-run` 会列出全部 `run_id`、深度、宽度和种子，不创建训练结果。

### 7.3 最小闭环实验

不要一开始运行全部组合。先使用 `ETTh1 + PatchTST + horizon=96 + seed=42`。

运行 RAW：

```bash
python3 pre_experiments/run_capacity_sweep.py \
  --dataset ETTh1 \
  --model PatchTST \
  --axis raw \
  --horizon 96 \
  --seeds 42 \
  --gpu 0 \
  --all
```

运行深度、宽度和二维实验：

```bash
set -euo pipefail

for axis in depth width joint; do
  python3 pre_experiments/run_capacity_sweep.py \
    --dataset ETTh1 \
    --model PatchTST \
    --axis "$axis" \
    --horizon 96 \
    --seeds 42 \
    --gpu 0 \
    --all
done
```

`--all` 表示顺序执行当前命令规划出的全部 run。也可以只运行 dry-run 列表中的某一项：

```bash
python3 pre_experiments/run_capacity_sweep.py \
  --dataset ETTh1 \
  --model PatchTST \
  --axis depth \
  --horizon 96 \
  --seeds 42 \
  --gpu 0 \
  --run-index 0
```

### 7.4 ILI 的命令

ILI 不使用 96 作为预测长度。例如：

```bash
python3 pre_experiments/run_capacity_sweep.py \
  --dataset ILI \
  --model PatchTST \
  --axis raw \
  --horizon 24 \
  --seeds 42 \
  --gpu 0 \
  --all
```

### 7.5 扩展到 5 个现有 Backbone

在最小闭环完全通过后，再运行：

```bash
set -euo pipefail

for model in Crossformer PatchTST TimesNet iTransformer TimeMixer; do
  for axis in raw depth width joint; do
    python3 pre_experiments/run_capacity_sweep.py \
      --dataset ETTh1 \
      --model "$model" \
      --axis "$axis" \
      --horizon 96 \
      --seeds 42 \
      --gpu 0 \
      --all
  done
done
```

确认 5 个模型均能完成后，才逐步扩展数据集、预测长度和三个随机种子。不要直接执行 9 数据集、5 模型、4 预测长度、4 种实验轴的全量笛卡尔积，它会产生数千次训练。

## 8. 每次训练保存什么

每个 run 保存在：

```text
pre_experiments/results/runs/<run_id>/
├── manifest.json
└── checkpoint/
    ├── test_metrics.json
    ├── test_results/
    │   ├── inputs.npy
    │   ├── prediction.npy
    │   └── targets.npy
    └── ...
```

`manifest.json` 包含：

- 数据集、模型、预测长度、深度、宽度和种子；
- MAE、MSE、RMSE、MAPE、WAPE；
- 总参数量和可训练参数量；
- FLOPs，若当前算子不支持则记录失败原因；
- 训练总时间、推理延迟中位数和 P90、吞吐量及 CUDA 峰值显存。

## 9. 生成逐局部单元损失

完成同一模型、数据集和预测长度下的候选容量实验后，运行：

```bash
python3 pre_experiments/build_local_losses.py
```

默认生成 `pre_experiments/results/local_losses.csv`。

调试时可以减少分析样本：

```bash
python3 pre_experiments/build_local_losses.py \
  --sample-stride 10 \
  --max-samples 1000 \
  --output pre_experiments/results/local_losses_debug.csv
```

正式结果不要使用调试采样参数，除非论文明确报告采样协议。

## 10. 计算饱和深度和宽度

```bash
python3 pre_experiments/analyze_saturation.py \
  --losses pre_experiments/results/local_losses.csv \
  --metric loss_mse \
  --epsilon 0.01
```

默认输出：

```text
pre_experiments/results/saturation/
├── depth_saturation.csv
├── depth_bucket_summary.csv
├── width_saturation.csv
├── width_bucket_summary.csv
├── joint_saturation.csv
├── joint_quadrant_summary.csv
└── diagnostics.json
```

重点验证：

- `U` 从 low 到 high 时，平均 `d_sat` 是否上升；
- 控制 `M` 后，回归 `d_sat ~ U + M` 中 `U` 的系数是否为正；
- `M` 从 low 到 high 时，平均 `w_sat` 是否上升；
- 控制 `U` 后，回归 `w_sat ~ U + M` 中 `M` 的系数是否为正；
- Q1/Q2/Q3/Q4 是否分别倾向浅窄、深窄、浅宽、深宽；
- 结论是否跨种子、数据集和 Backbone 稳定。

## 11. 推荐执行顺序

1. 确认 Electricity 使用哪套数据；
2. 放入 9 个原始 CSV；
3. 运行库存检查和严格数据预处理；
4. 完成 9 个数据集的 U/M 画像并检查指标是否退化；
5. 安装 PyTorch，运行自动测试；
6. 完成 `ETTh1 + PatchTST + 96 + seed 42` 的完整闭环；
7. 扩展到 5 个现有 Backbone；
8. 扩展到其他数据集和预测长度；
9. 增加种子 43、44；
10. 生成局部损失、饱和容量和 Q1-Q4 统计；
11. 最后再决定是否引入缺失的 3 个 2025 Backbone。

更详细的统计定义和公平性约束见 [`pre_experiments/README.md`](pre_experiments/README.md)。
