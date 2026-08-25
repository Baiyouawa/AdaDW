# 模型预实验说明

## 1. 实验目的

本预实验在 ETTh1 上检验以下两个局部容量假设：

1. 局部状态更新需求 `U` 越大，该时间窗口需要的最佳深度越大；
2. 局部模式多样性 `M` 越大，该时间窗口需要的最佳宽度越大。

实验不是用整个数据集的平均损失选择一个全局模型，而是对测试集中的每个时间窗口分别计算候选模型损失，再得到随时间变化的最佳深度和最佳宽度轨迹。U/M 也从完全相同的模型输入窗口重新计算，从而与容量轨迹逐窗口对齐。

## 2. 模型选择

三个 Backbone 分别代表 2023、2024 和 2025 年：

| 年份 | 模型 | RAW 深度 (`e_layers`) | RAW 主宽度 (`d_model`) | 配套 `d_ff` |
|---|---|---:|---:|---:|
| 2023 | PatchTST | 3 | 16 | 128 |
| 2024 | TimeMixer | 2 | 16 | 32 |
| 2025 | MultiPatchFormer | 1 | 256 | 256 |

上述值来自三个模型原始运行配置，而不是本地适配器 dataclass 曾经使用的默认值。原始 `e_layers` 映射为本项目的 `num_layers`。宽度轴使用模型主表示维度 `d_model`；`d_ff` 按原始扩展比例随 `d_model` 联动，避免只改变 FFN 中间层或破坏原始宽度比例。

除深度和宽度外，构造模型时固定以下原始结构参数：

| 模型 | 固定结构参数 |
|---|---|
| PatchTST | `n_heads=4`、`dropout=0.3`、`fc_dropout=0.3`、`head_dropout=0`、`patch_len=16`、`stride=8`；`d_ff=8*d_model` |
| TimeMixer | `down_sampling_layers=3`、`down_sampling_window=2`、`down_sampling_method=avg`；`d_ff=2*d_model` |
| MultiPatchFormer | `n_heads=8`；`d_ff=d_model` |

## 3. 默认配置

主配置位于 `pre_experiments/config.json`，模型轨迹部分为：

```json
"model_trajectory": {
  "dataset": "ETTh1",
  "models": ["PatchTST", "TimeMixer", "MultiPatchFormer"],
  "horizon": 96,
  "metric": "loss_mse",
  "sample_stride": 24,
  "selection_tolerance": 0.0,
  "seeds": [3407]
}
```

公共训练配置为：

| 配置项 | 默认值 |
|---|---:|
| 输入长度 | 96 |
| 预测长度 | 96 |
| epoch | 100 |
| batch size | 64 |
| learning rate | 0.0002 |
| weight decay | 0.0005 |
| early stopping patience | 10 |
| seed | 3407 |
| 局部损失 | MSE |
| 时间窗口采样步长 | 24 |

ETTh1 是小时级数据，因此 `sample_stride=24` 表示每隔一天选取一个长度为 96 的输入窗口。容量轨迹只覆盖测试时间段；原有 `UM_temporal_trajectory.pdf` 覆盖完整原始序列，二者不能直接把 0%-100% 横轴视为同一个绝对日期。本流程在测试输入上重新计算 U/M，生成的是严格对齐的测试段曲线。

训练和逐窗口损失默认使用训练集统计量标准化后的尺度。U/M 描述符内部采用稳健标准化，因而不依赖各通道的原始量纲；所有候选容量必须保持相同的数据标准化和损失尺度。

## 4. 容量搜索空间

深度和宽度分别进行单轴扫描，以避免联合网格中深度与宽度相互补偿而干扰相关性判断。

因此，本文的“最佳深度”是固定 RAW 宽度条件下的最佳深度，“最佳宽度”是固定 RAW 深度条件下的最佳宽度，并不是在深度 x 宽度联合网格中选出的全局最佳参数对。这一口径更适合分别检验 `U -> depth` 和 `M -> width`。

### 深度实验

- 每个模型使用 5 个正整数候选；
- 宽度固定为该模型的 RAW 宽度；
- 每个模型共 `5 个深度 x 1 个种子 = 5` 次训练。

| 模型 | 五个深度候选 | RAW 位置 |
|---|---|---:|
| PatchTST | `1, 2, 3, 4, 5` | 3 |
| TimeMixer | `1, 2, 3, 4, 5` | 2 |
| MultiPatchFormer | `1, 2, 3, 4, 5` | 1 |

如果 RAW 深度为 6，理想的对称五档可取 `2,4,6,8,10`。本实验三个 RAW 深度只有 3、2、1，受“网络至少包含一个 block”的正整数下界限制，不可能都同时提供两个更浅候选。因此使用共同的 `1..5` 搜索集合，并在结果中把 TimeMixer 和 MultiPatchFormer 的左侧搜索空间不足作为边界限制报告，不能构造 0 层或负层来追求形式对称。

### 宽度实验

- 以原始 `d_model` 为中心，向下缩小到 `RAW/2`、`RAW/4`，向上扩大到 `RAW*2`、`RAW*4`；
- `d_ff` 保持各模型的原始比例同步缩放；
- 深度固定为该模型的 RAW 深度；
- 每个模型共 `5 个宽度 x 1 个种子 = 5` 次训练。

三个模型的实际主宽度及配套 `d_ff` 如下：

| 模型 | 五档 `d_model` | 对应五档 `d_ff` |
|---|---|---|
| PatchTST | `4, 8, 16, 32, 64` | `32, 64, 128, 256, 512` |
| TimeMixer | `4, 8, 16, 32, 64` | `8, 16, 32, 64, 128` |
| MultiPatchFormer | `64, 128, 256, 512, 1024` | `64, 128, 256, 512, 1024` |

总训练数为 `3 个模型 x 2 个容量轴 x 5 个候选值 x 1 个种子 = 30`。注册表用模型专属 `width_unit` 和宽度组编码上述 `d_model`；报告中以实际 `d_model` 为主，并同时给出联动的 `d_ff`。

## 5. 实验步骤

### 5.1 检查实验计划

```bash
pixi run model-preexp-plan
```

该命令只打印 30 个 run 的计划，不训练模型。也可以直接调用：

```bash
pixi run python pre_experiments/run_model_preexperiments.py --dry-run
```

### 5.2 执行容量实验

```bash
pixi run model-preexp
```

默认输出目录为：

```text
pre_experiments/results/model_trajectories/runs/
```

每个 run 保存 `manifest.json`、测试预测、测试目标和测试输入。批量入口会跳过状态为 complete 且三个预测数组均存在的 run，因此中断后可以使用同一命令继续。

需要覆盖训练参数时，例如：

```bash
pixi run python pre_experiments/run_model_preexperiments.py \
  --all \
  --gpu 0 \
  --seeds 3407 \
  --epochs 100 \
  --batch-size 64
```

### 5.3 生成逐窗口局部损失

```bash
pixi run model-preexp-losses
```

对每个保存的测试样本，脚本执行以下操作：

1. 把长度 96 的输入视为当前时间窗口；
2. 在该窗口的每个 ETTh1 通道上计算 U 和 M；
3. 计算该通道未来 96 步预测的 MAE 和 MSE；
4. 保存模型、种子、深度、宽度组、实际宽度和时间窗口索引。

不要在正式结果中设置 `--max-samples`。三个模型和所有容量候选必须使用相同的 seed 与 `--sample-stride`。

### 5.4 选择最佳容量并画时序曲线

```bash
pixi run model-preexp-analyze
```

等价的完整命令是：

```bash
pixi run python pre_experiments/analyze_model_trajectories.py \
  --losses pre_experiments/results/model_trajectories/local_losses.csv \
  --dataset ETTh1 \
  --models PatchTST TimeMixer MultiPatchFormer \
  --horizon 96 \
  --metric loss_mse \
  --epsilon 0.0
```

对每个“模型 x 时间窗口 x 容量候选”，在 seed 3407 下先跨通道平均 MSE，然后：

- 深度轴从该模型的 5 个深度候选中选择平均 MSE 最小者；
- 宽度轴从该模型的 5 个 `d_model` 候选中选择平均 MSE 最小者；
- 若损失相同，选择较小容量；
- `epsilon=0.0` 表示严格最小损失，符合主实验定义；
- 可另外使用 `epsilon=0.01` 做敏感性实验，此时选择距最优损失 1% 以内的最小容量。

分析前会检查每个模型的 5 个候选容量是否齐全、是否包含指定 seed 3407 和 ETTh1 的全部 7 个通道、深度与宽度实验是否覆盖同一批窗口，以及 U/M 是否在模型和容量之间严格对齐；发现部分 run 缺失或错位时直接报错，不会从残缺候选中选择“最佳值”。

## 6. 输出文件

默认生成：

```text
pre_experiments/results/model_trajectories/
├── local_losses.csv
├── model_capacity_trajectories.csv
├── model_capacity_correlations.json
└── model_capacity_temporal_trajectories.pdf
```

`model_capacity_trajectories.csv` 每行对应一个模型、一个时间窗口和一个容量轴。关键字段包括：

| 字段 | 含义 |
|---|---|
| `relative_time` | 测试段内的相对时间顺序 |
| `U_mean`、`U_p10`、`U_p90` | 窗口内跨通道的 U 均值和区间 |
| `M_mean`、`M_p10`、`M_p90` | 窗口内跨通道的 M 均值和区间 |
| `best_depth` | 该窗口的最佳离散深度 |
| `best_width_group` | 该窗口的最佳相对宽度组 |
| `best_d_model` | 该窗口的最佳主表示宽度 |
| `best_d_ff` | 与最佳 `d_model` 联动的 FFN 中间维度 |
| `selected_loss` | seed 3407 下所选容量的跨通道平均损失 |
| `optimal_loss` | 候选集合中的严格最小损失 |

PDF 每个模型占一行：左图比较 `U` 与最佳深度，右图比较 `M` 与最佳 `d_model`。容量使用阶梯线，因为候选深度和宽度是离散值。

训练 manifest 和 `local_losses.csv` 中，`width` 字段保存实际 `d_model`，`coupled_width` 字段保存按原始比例联动的 `d_ff`；`width_group` 仅用于表示相对容量档位。

## 7. 结果判断

不能仅凭两条曲线局部同涨同跌就确认假设。主要查看 `model_capacity_correlations.json` 中每个模型分别计算的：

- `U_mean` 与 `best_depth` 的 Spearman 相关系数；
- `M_mean` 与 `best_width_group` 的 Spearman 相关系数；
- 普通 Spearman p 值；
- 保留时间序列顺序结构的 circular-shift 检验 p 值。

支持假设至少需要：三个模型中的相关方向大体一致，相关系数为正，并且结果对 `epsilon=0` 与 `epsilon=0.01`、MSE 与 MAE 两种口径不过度敏感。若最佳容量长期停留在某个模型搜索集合的最小值或最大值，应先扩展搜索范围，不能把边界截断结果解释为真实最优容量。

此外，时间窗口之间高度重叠，普通 Spearman p 值会低估时间自相关带来的不确定性。因此结论应优先参考逐模型效应方向、circular-shift p 值和跨模型一致性；合并三个模型的 pooled 相关仅作描述，不作为主要显著性证据。

本实验得到的是使用测试目标事后计算的 oracle 最佳容量，只用于验证 U/M 是否具有容量解释力，不能直接作为在线动态选层或选宽策略的性能结果。若后续要报告动态推理收益，容量映射必须只在训练集或验证集拟合，再在未参与容量选择的测试区间评估。
