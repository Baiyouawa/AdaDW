# AdaWD-TS 预实验代码

本目录是 AdaWD-TS 论文的独立代码工作区。当前阶段对应
`../2027ICLR：动态宽深时序神经网络 (1).pdf` 第 1-4 页和第 12-17 页中的预实验。

预实验检验三个假设：

1. 局部状态更新需求 `U_i` 应当能够解释有效深度需求；
2. 局部模式多样性 `M_i` 应当能够解释有效宽度需求；
3. `U_i` 和 `M_i` 应分别解释不同的容量轴，而不是退化成同一个难度分数。

## 1. 当前实验范围

当前注册 9 个预测数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、ILI、ExchangeRate 和 Traffic。

当前可运行 8 个 Backbone：Crossformer、PatchTST、TimesNet、iTransformer、TimeMixer、WPMixer、TimeFilter 和 MultiPatchFormer。

WPMixer 基于有许可证的官方实现适配；TimeFilter 和 MultiPatchFormer 是依据论文结构编写的
独立 BasicTS 适配器。具体来源、commit 和边界见 `Baselines/README.md`。

完整的静态矩阵、Baseline 配置、数据泄露检查和剩余风险见
[`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md)。当前正式 RAW benchmark 仍为 `0/864`，
因此“静态链路完整”不能表述成“全部组合已经实际运行通过”。

> **Electricity 数据身份待确认**：当前代码按 DropoutTS 使用的 321 通道 LTSF Electricity
> 基准配置。论文草稿中的“家庭电参量及分表计量”对应另一套数据。正式下载和报告实验前必须选择其中一套，不能混用名称。

## 2. 目录结构

```text
AdaWD/
├── pixi.toml                        # Pixi 环境与一键任务
├── pixi.lock                        # 锁定的跨机器依赖版本
├── Baselines/                       # BasicTS 运行核心和 Backbone
│   ├── basicts/models/              # 8 个预测 Backbone
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
cd AdaDW
```

## 3. Pixi 运行环境与一键任务

本项目统一使用 Pixi 管理 Python、PyTorch、CUDA 运行库和实验依赖，不再需要手工创建
venv 或执行 `pip install`。首次运行：

```bash
pixi install
```

`pixi.lock` 会确保不同机器使用相同依赖版本。检查 PyTorch、CUDA 和自动测试：

```bash
pixi run install-check
pixi run test
```

三个主要一键任务是：

```bash
# 下载、严格预处理、画像并绘制全部 9 个数据集
pixi run preexp-dataset

# 深度预实验：默认 ETTh1 + PatchTST + horizon 96 + seed 42
pixi run preexp-depth

# 宽度预实验：默认 ETTh1 + PatchTST + horizon 96 + seed 42
pixi run preexp-width
```

完整 RAW 时序预测基准使用 8 个模型、9 个数据集、每个数据集注册的 4 个
horizon，以及随机种子 3407/3408/3409：

正式启动前必须用当前版本重新执行 `pixi run prepare-datasets`。旧版 `meta.json` 没有数据
指纹，runner 会拒绝复用，以免旧切分或旧预处理结果混入新协议。

```bash
# 仅生成并检查 864-run 计划，不训练
pixi run forecast-all-plan

# 一键顺序执行，自动跳过协议一致且已经完成的 run
pixi run forecast-all
```

正式全量运行前，先执行 72-run 的模型/数据集兼容性检查：

```bash
pixi run forecast-smoke-plan
pixi run forecast-smoke
```

Smoke 模式为每个模型/数据集选择最短 horizon、seed 3407 和 1 epoch；它只覆盖
训练、验证、预测、指标落盘和 checkpoint 清理链路，不用于比较模型精度。正式任务
仍使用 `benchmark_config.json` 中的标准 epoch。

八个模型的显式架构配置位于 `Baselines/registry.json`，模型训练轮数和 batch size 位于
`pre_experiments/benchmark_config.json`；Electricity 和 Traffic 另有显存安全上限。该配置是
本项目预注册的统一比较协议，不等于逐项复刻所有官方脚本。正式指标为逐通道
ZScore 空间的 MAE/MSE/RMSE，每组三个 seed 输出 mean 和 sample std。成功 run 仅保留
manifest 中的指标与效率记录、一个原始量纲预测切片 CSV 和对应 PNG。评估过程只选择性
捕获这个固定窗口，不保存完整测试集预测；生成图片后删除 checkpoint 和临时样本数组。

预测图不按误差挑选窗口：在能容纳该数据集最大 horizon 的公共测试区间内固定取相对位置
50% 的窗口，因此同一数据集四个 horizon 使用相同预测起点；再从通道 ID 中等距选取最多
4 个通道。每个 run 会生成单模型真实值/预测值图；每次调度结束还会按数据集和 horizon
生成 8 个 Baseline 的跨 seed 对比图。全量完成后共有 `9 x 4 = 36` 张汇总图。

长任务可按计划编号分段运行：

```bash
pixi run python pre_experiments/run_forecasting_benchmarks.py \
  --start-index 1 --stop-index 108
```

计划和最终结果位于：

```text
pre_experiments/results/forecasting_raw/plan.csv
pre_experiments/results/forecasting_raw/summary/per_seed.csv
pre_experiments/results/forecasting_raw/summary/summary.csv
pre_experiments/results/forecasting_raw/summary/coverage.csv
pre_experiments/results/forecasting_raw/summary/Result.md
```

正式训练前可只查看容量计划：

```bash
pixi run preexp-depth-plan
pixi run preexp-width-plan
```

任务支持追加参数。例如，在 ILI 上用 TimesNet、预测 24 步、三个种子运行深度实验：

```bash
pixi run preexp-depth \
  --dataset ILI \
  --model TimesNet \
  --horizon 24 \
  --seeds 42 43 44
```

可用 `pixi task list` 查看全部任务。自动测试通过后再开始正式实验。

## 4. 下载和存放原始数据

### 4.1 一键下载（推荐）

项目提供了无需额外 Python 依赖的下载脚本。默认下载全部 9 个数据集，自动创建目录，
并在写入最终文件前校验 `date` 列、行数和特征列数：

```bash
pixi run download-datasets
```

脚本支持断点续传；再次运行时会校验并跳过已有的完整文件。只下载部分数据集时使用：

```bash
pixi run download-datasets --dataset ETTh1 Weather
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
pixi run inventory
```

数据存在时，对应状态应从 `missing_data` 变为 `raw_available`。

## 5. 数据预处理

处理单个数据集：

```bash
pixi run python pre_experiments/prepare_dataset.py \
  --dataset ETTh1 \
  --strict-shape
```

`--strict-shape` 会在时间点数量或特征数量不符合配置时直接报错。调试自定义数据时可以暂时去掉，正式实验建议保留。

一键严格处理 9 个数据集：

```bash
pixi run prepare-datasets
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
├── train_time_index.npy
├── val_time_index.npy
├── test_time_index.npy
└── meta.json
```

再次运行 `pixi run inventory`，状态应显示为 `processed_available`。

## 6. 数据特性预实验

该阶段计算论文定义的六个描述分量：

```text
U_i = mean(u_change, u_spectral, u_surprise)
M_i = mean(m_peak, m_band, m_channel)
```

### 6.1 画像单个数据集

默认优先读取完整原始 CSV：

```bash
pixi run python pre_experiments/profile_dataset.py --dataset ETTh1
```

如果希望严格按 train/validation/test 边界画像，读取处理后的 NPY：

```bash
pixi run python pre_experiments/profile_dataset.py \
  --dataset ETTh1 \
  --processed
```

限制窗口数的调试命令：

```bash
pixi run python pre_experiments/profile_dataset.py \
  --dataset ETTh1 \
  --window-size 96 \
  --stride 24 \
  --max-windows 100
```

Electricity 和 Traffic 默认均匀选择 64 个通道。正式敏感性实验可运行全通道版本：

```bash
pixi run python pre_experiments/profile_dataset.py \
  --dataset Electricity \
  --all-channels \
  --output-dir pre_experiments/results/profiles/Electricity_all_channels
```

### 6.2 一键数据集预实验

```bash
pixi run preexp-dataset
```

该任务按 `datasets/catalog.json` 顺序对全部数据集执行严格预处理、U/M 画像和诊断图绘制。
只运行部分数据集时使用 `pixi run preexp-dataset --datasets ETTh1 Weather`。如需严格按
train/validation/test 边界画像，追加 `--processed-profiles`。

每个数据集生成：

```text
pre_experiments/results/profiles/ETTh1/
├── windows.csv      # 每个局部单元的六项分量、U、M 和分桶
└── summary.json     # 均值、标准差、P10/P50/P90、IQR 和 U/M 相关性
```

### 6.3 绘图

```bash
pixi run python pre_experiments/plot_profiles.py \
  --profiles pre_experiments/results/profiles/ETTh1/windows.csv
```

批量绘图：

```bash
set -euo pipefail

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity ILI ExchangeRate Traffic; do
  pixi run python pre_experiments/plot_profiles.py \
    --profiles "pre_experiments/results/profiles/${dataset}/windows.csv"
done
```

默认输出 `profile_diagnostics.pdf`。六个面板按时间窗口顺序展示 U/M 热图、跨通道窗口均值
轨迹、P10-P90 阴影、时间着色的 U-M 散点和四段时间箱线图。所有 U/M 坐标和色标固定为
`[0,1]`，需要检查：

- 窗口均值是否随时间发生具有实际幅度的变化，而不只是通道之间不同；
- `U` 或 `M` 是否集中在非常窄的区间；
- `U` 与 `M` 是否接近完全相关；
- low/mid/high 三个分桶是否都有足够样本；
- Q1-Q4 四类局部片段是否实际存在。

U/M 是多个有界描述分量的平均值，不是概率，`0.2-0.4` 的均值本身不表示指标失败。
判断时间多样性应以窗口轨迹、窗口级离散度和后续统计检验为准。如果指标退化，应先调整
画像窗口、步长或描述符超参数，不应直接开始全量模型训练。

## 7. 深度、宽度与 RAW 实验

### 7.1 当前默认配置

- 深度候选：`D={1,2,4,8}`；
- 宽度组候选：`W={1,2,4,8}`；
- 二维候选：`D={2,4,8} x W={2,4,8}`；
- 容量配置默认随机种子：`3407,3408,3409`；便捷 pilot 命令可显式使用 seed 42；
- 普通数据集：输入长度 96，预测长度 96/192/336/720；
- ILI：输入长度 24，预测长度 24/36/48/60；
- 训练指标：MAE、MSE、RMSE、MAPE、WAPE。

参数位于 [`pre_experiments/config.json`](pre_experiments/config.json)。四种轴的含义：

- `raw`：`registry.json` 中该 Backbone 的显式 `benchmark_config`；
- `depth`：固定 RAW 宽度，只改变深度；
- `width`：固定 RAW 深度，只改变宽度；
- `joint`：运行二维深宽组合。

### 7.2 先检查计划，不训练

```bash
pixi run preexp-depth-plan
pixi run preexp-width-plan
```

`--dry-run` 会列出全部 `run_id`、深度、宽度和种子，不创建训练结果。

### 7.3 最小闭环实验

不要一开始运行全部组合。先使用 `ETTh1 + PatchTST + horizon=96 + seed=42`。

运行 RAW：

```bash
pixi run python pre_experiments/run_capacity_sweep.py \
  --dataset ETTh1 \
  --model PatchTST \
  --axis raw \
  --horizon 96 \
  --seeds 42 \
  --gpu 0 \
  --all
```

一键运行深度和宽度预实验：

```bash
pixi run preexp-depth
pixi run preexp-width
```

两项任务默认采用 `ETTh1 + PatchTST + horizon=96 + seed=42 + GPU 0`，分别顺序执行
`D={1,2,4,8}` 和 `W={1,2,4,8}`。编排器会先进行严格数据预处理。

运行 RAW 和二维实验时仍可使用底层入口：

```bash
set -euo pipefail

for axis in raw joint; do
  pixi run python pre_experiments/run_capacity_sweep.py \
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
pixi run python pre_experiments/run_capacity_sweep.py \
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
pixi run preexp-depth \
  --dataset ILI \
  --model PatchTST \
  --horizon 24 \
  --seeds 42
```

### 7.5 扩展到 8 个 Backbone

在最小闭环完全通过后，再运行：

```bash
set -euo pipefail

for model in Crossformer PatchTST TimesNet iTransformer TimeMixer WPMixer TimeFilter MultiPatchFormer; do
  for axis in raw depth width joint; do
    pixi run python pre_experiments/run_capacity_sweep.py \
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

确认 8 个模型均能完成后，才逐步扩展数据集、预测长度和三个随机种子。不要直接执行 9 数据集、8 模型、4 预测长度、4 种实验轴的全量笛卡尔积，它会产生数千次训练。

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
- 配置要求的指标；正式 normalized RAW benchmark 为 MAE、MSE、RMSE；
- 总参数量和可训练参数量；
- FLOPs，若当前算子不支持则记录失败原因；
- 训练总时间、推理延迟中位数和 P90、吞吐量及 CUDA 峰值显存。

正式 RAW run 的目录还会保留：

```text
forecast_slice.csv       # 固定测试窗口，原始量纲的历史/真实未来/预测
forecast_vs_target.png   # 单模型、单 seed 的预测图
```

汇总目录另外生成：

```text
summary/visualization_index.csv
summary/visualizations/<dataset>__h<horizon>__baseline_forecasts.png
```

## 9. 生成逐局部单元损失

完成同一模型、数据集和预测长度下的候选容量实验后，运行：

```bash
pixi run python pre_experiments/build_local_losses.py
```

默认生成 `pre_experiments/results/local_losses.csv`。

调试时可以减少分析样本：

```bash
pixi run python pre_experiments/build_local_losses.py \
  --sample-stride 10 \
  --max-samples 1000 \
  --output pre_experiments/results/local_losses_debug.csv
```

正式结果不要使用调试采样参数，除非论文明确报告采样协议。

## 10. 计算饱和深度和宽度

```bash
pixi run python pre_experiments/analyze_saturation.py \
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
2. 运行 `pixi install` 和 `pixi run test`；
3. 运行 `pixi run preexp-dataset` 下载、预处理并画像 9 个数据集；
4. 检查 U/M 指标是否退化；
5. 运行深度和宽度任务的 plan；
6. 完成 `ETTh1 + PatchTST + 96 + seed 42` 的完整闭环；
7. 扩展到 8 个 Backbone；
8. 扩展到其他数据集和预测长度；
9. 增加种子 43、44；
10. 生成局部损失、饱和容量和 Q1-Q4 统计；
11. 在全数据集扩展前，分别复核 3 个 2025 Backbone 的论文协议与官方超参数。

更详细的统计定义和公平性约束见 [`pre_experiments/README.md`](pre_experiments/README.md)。
