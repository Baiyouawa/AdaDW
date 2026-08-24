# AdaWD 预实验、正式 Baseline 与绘图代码详解

本文按当前仓库中的真实代码路径解释整个实验系统，重点回答四个问题：

1. 数据如何从 CSV 变成模型可读取的训练、验证、测试样本；
2. 预实验如何计算局部状态更新需求 `U`、模式多样性 `M`，以及如何得到饱和深度和饱和宽度；
3. 八个 forecasting baseline 如何被统一到 BasicTS 训练接口并完成正式实验；
4. 当前代码实际画了哪些图，每张图的横轴、纵轴、点、柱、颜色和虚线分别表示什么。

需要先明确：仓库中目前只有 [`pre_experiments/plot_profiles.py`](pre_experiments/plot_profiles.py)
直接使用 Matplotlib 生成静态 PDF。BasicTS 训练器还会把 loss 和指标写入 TensorBoard。
容量饱和分析和正式 RAW benchmark 当前只输出 CSV、JSON、Markdown，不会自动生成论文中的
容量趋势图或 baseline 对比图。本文后面会分别说明“已经实现的图”和“建议根据结果表绘制的图”。

## 1. 两条实验主线

代码分成两条相互关联、但目的不同的实验主线。

### 1.1 预实验主线

预实验要验证：

- 局部状态更新需求 `U` 是否对应更大的有效深度；
- 局部模式多样性 `M` 是否对应更大的有效宽度；
- `U` 和 `M` 是否具有相对独立的解释力，而不是同一个“难度分数”的重复表达。

完整数据流为：

```text
原始 CSV
  -> prepare_dataset.py：严格检查、切分并保存 NPY
  -> profile_dataset.py：滑窗计算每个“窗口 x 通道”的 U/M
  -> plot_profiles.py：绘制 U/M 时间热图、时间轨迹、时间着色散点和时间分段箱线图

不同深度/宽度的模型训练
  -> run_capacity_sweep.py：RAW/depth/width/joint 容量扫描
  -> build_local_losses.py：计算每个“测试样本 x 通道”的局部 MAE/MSE
  -> analyze_saturation.py：选出每个局部单元的 d_sat/w_sat
  -> CSV/JSON：分桶趋势、四象限统计和 U/M 偏回归
```

### 1.2 正式 baseline 主线

正式实验比较八个模型在九个预测数据集、四个预测长度、三个随机种子下的 RAW 性能：

```text
benchmark_config.json
  -> run_forecasting_benchmarks.py：生成 864-run 计划并逐项调度
  -> run_capacity_sweep.py --axis raw：执行一项真实训练
  -> BasicTSRunner：训练、验证、加载最佳 checkpoint、测试
  -> manifest.json：保存预测指标和效率指标
  -> summarize_forecasting_benchmarks.py：按三个种子汇总
  -> per_seed.csv / summary.csv / coverage.csv / Result.md
```

这里的 `RAW` 表示使用 [`Baselines/registry.json`](Baselines/registry.json) 中该模型登记的
原始深度和宽度，不是原始量纲误差的意思。正式配置当前采用 `metric_scale=normalized`，
所以 MAE、MSE、RMSE 是逐通道 ZScore 空间中的指标。

## 2. 配置文件是如何分工的

### 2.1 数据集目录 `datasets/catalog.json`

[`datasets/catalog.json`](datasets/catalog.json) 是数据协议的唯一来源，记录：

- 原始文件名和别名；
- 时间列名、采样频率、预期行数和通道数；
- train/validation/test 切分比例；
- 输入长度和四个预测长度；
- U/M 画像的窗口长度和步长；
- 高维数据画像时最多选择多少个通道。

下面先介绍九个数据集的背景，再给出基于当前九份 CSV 实际检查得到的 dim、size、split、
frequency、prediction length 和 domain。背景来源主要包括
[ETDataset 官方仓库](https://github.com/zhouhaoyi/ETDataset)、
[Max Planck Jena 气象站](https://www.bgc-jena.mpg.de/wetter/)、
[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)、
[CDC FluView/ILINet](https://www.cdc.gov/fluview/overview/index.html)、
[Caltrans PeMS](https://pems.dot.ca.gov/) 和
[LSTNet 数据预处理仓库](https://github.com/laiguokun/multivariate-time-series-data)。
本项目真正下载的是 THUML Time-Series-Library 中的 benchmark CSV；下载映射见
[`download_datasets.py`](download_datasets.py)。

#### 2.1.1 ETTh1

ETT 是 Electricity Transformer Temperature 数据集，最初用于研究长序列预测。官方说明称，
它来自中国某省两个区域、两个变电站的电力变压器，包含外部负荷与油温。油温能够反映
变压器运行状态，长期预测可以辅助负荷调度和设备安全评估。

`ETTh1` 中的 `h` 表示 hourly，`1` 表示第一台变压器/第一个站点。当前 CSV 的七个信号列为：

| 列 | 含义 |
| --- | --- |
| `HUFL` | High Useful Load，高有用负荷 |
| `HULL` | High Useless Load，高无用负荷 |
| `MUFL` | Middle Useful Load，中有用负荷 |
| `MULL` | Middle Useless Load，中无用负荷 |
| `LUFL` | Low Useful Load，低有用负荷 |
| `LULL` | Low Useless Load，低无用负荷 |
| `OT` | Oil Temperature，油温，也是 ETT 原始定义中的目标变量 |

本项目按 multivariate-to-multivariate 方式预测全部七个通道，不是只预测 `OT`。CSV 实际覆盖
`2016-07-01 00:00:00` 到 `2018-06-26 19:00:00`，共 17,420 个严格连续的小时点，
无重复时间、无 `NaN/Inf`。

#### 2.1.2 ETTh2

`ETTh2` 是 ETT 的第二个站点/第二台变压器的小时级序列。它与 ETTh1 具有完全相同的列结构、
采样频率和时间范围，但观测值来自另一台设备，因此负荷幅度、油温范围与局部状态变化并不
相同。把 ETTh1 和 ETTh2 同时纳入实验，可以检查模型结论是否只适用于一个变压器。

当前 CSV 同样是 17,420 行、7 个信号、每小时一行，时间从
`2016-07-01 00:00:00` 到 `2018-06-26 19:00:00`，时间戳严格递增且没有缺失数值。

#### 2.1.3 ETTm1

`ETTm1` 对应第一台变压器的 15 分钟级版本；`m` 表示 minute-level benchmark。它保留与
ETTh1 相同的六类负荷和油温 `OT`，但一天有 96 个观测点，可以展示更细粒度的日内波动。

当前 CSV 从 `2016-07-01 00:00:00` 到 `2018-06-26 19:45:00`，共 69,680 行、7 个信号。
所有相邻时间差均为 15 分钟，无重复、无 `NaN/Inf`。虽然 ETT 官方背景常用“大约两年、
约 70,080 点”描述 minute 版本，当前实验必须以实际文件的 69,680 行作为 size。

#### 2.1.4 ETTm2

`ETTm2` 是第二台变压器的 15 分钟级序列，与 ETTh2 对应。它兼具第二站点的不同负荷分布和
minute-level 的高时间分辨率，适合检验模型能否同时处理局部快速变化与长期周期。

当前 CSV 的时间、行数和列结构与 ETTm1 相同：69,680 行、7 个信号、严格 15 分钟间隔，
覆盖 `2016-07-01 00:00:00` 到 `2018-06-26 19:45:00`，没有重复或 `NaN/Inf`。

#### 2.1.5 Weather

Weather benchmark 来自德国耶拿 Max Planck Institute for Biogeochemistry 的气象站数据。
当前文件选择 2020 日历年附近的一年，并按 10 分钟粒度组织，用于同时预测压力、温度、湿度、
风、降雨、辐射等多种气象观测。

当前 21 个信号可以分为：

- 压力：`p (mbar)`；
- 温度：`T (degC)`、`Tpot (K)`、`Tdew (degC)`、`Tlog (degC)`；
- 湿度和水汽：`rh (%)`、`VPmax`、`VPact`、`VPdef`、`sh`、`H2OC`；
- 空气密度：`rho (g/m**3)`；
- 风：`wv (m/s)`、`max. wv (m/s)`、`wd (deg)`；
- 降水：`rain (mm)`、`raining (s)`；
- 辐射与光合有效辐射：`SWDR`、`PAR`、`max. PAR`；
- `OT`：当前 benchmark 保留的第 21 个输出通道。当前 CSV 已把其原始语义列名改成 `OT`，
  因此代码层面只能安全地把它解释为最后一个被预测变量，不能仅凭 `OT` 推断成 ETT 油温。

CSV 实际包含 52,696 行，时间从 `2020-01-01 00:10:00` 到
`2021-01-01 00:00:00`。名义 frequency 是 10 分钟，但实际检查发现：

- `2020-05-12 06:00:00` 出现两次；
- `2020-05-29 09:30:00` 后直接跳到 `11:10:00`，间隔 1 小时 40 分钟；
- 52,695 个相邻时间差中，52,693 个是正常的 10 分钟；
- CSV 没有 Pandas `NaN`，但含 81 个 `-9999` 哨兵值：`wv` 1 个、`max. PAR` 30 个、
  `OT` 50 个；
- `SWDR`、`PAR`、`max. PAR` 的单位字符在当前 UTF-8 表头中已显示为 `�`，属于元数据编码损坏。

当前预处理器只把 `NaN` 视为 null，`-9999` 会参与 ZScore 拟合和训练。正式实验前应明确清洗
或掩码协议；否则极端哨兵值会扭曲均值、标准差和预测误差。

#### 2.1.6 Electricity

Electricity 来源可追溯到 UCI `ElectricityLoadDiagrams20112014`：原始数据记录葡萄牙 370 个
客户的用电量，每 15 分钟一个值。LSTNet 的数据说明称，由于部分维度在 2011 年为零，预处理
去除了 2011 年记录，最终得到 321 个客户并转换为小时级序列。当前项目使用的正是这个标准
LTSF 321 通道版本，不是“家庭单户电参量及分表计量”数据集。

当前 CSV 的信号列名为 `0..319, OT`，即 321 个匿名客户序列。`OT` 在这里是最后一个客户
通道的统一 benchmark 命名，不是油温。文件有 26,304 个严格连续的小时点，时间字段从
`2016-07-01 02:00:00` 到 `2019-07-02 01:00:00`，没有 `NaN/Inf`。

需要区分“原始数据时间”和“当前 benchmark 时间字段”：UCI 原始数据来自 2011-2014，
而当前 CSV 写入的是 2016-2019 时间戳。模型实际使用后者构造 timestamp 特征，因此报告本
仓库实验时应列当前 CSV 时间范围，同时在背景中说明其原始客户负荷来源。

#### 2.1.7 ILI

ILI 是 Influenza-Like Illness 流感样病例数据，来自美国 CDC 流感监测体系 FluView/ILINet。
ILINet 汇总门诊机构每周报告的总就诊与符合 ILI 症状定义的就诊，用于反映流感活动强度。
它的样本量明显少于其他数据集，但周期单位是一周，适合测试长季节周期和突发流行峰。

当前 CSV 的七个信号为：

| 列 | 当前含义 |
| --- | --- |
| `% WEIGHTED ILI` | 经地区/人口等权重汇总的 ILI 就诊比例 |
| `%UNWEIGHTED ILI` | 未加权 ILI 就诊比例 |
| `AGE 0-4` | 0-4 岁 ILI 病例计数 |
| `AGE 5-24` | 5-24 岁 ILI 病例计数 |
| `ILITOTAL` | ILI 病例总数 |
| `NUM. OF PROVIDERS` | 参与报告的医疗服务提供者数量 |
| `OT` | benchmark 重命名后的最后一个预测通道；其数值量级与 ILINet 的总就诊人数一致，但当前 CSV 没保留原字段名 |

当前文件有 966 个严格连续的周点，从 `2002-01-01` 到 `2020-06-30`，无重复、无
`NaN/Inf`。输入长度只有 24 周，预测长度为 24/36/48/60 周，不能套用其他数据集的
96/192/336/720 协议。

#### 2.1.8 ExchangeRate

ExchangeRate 是 LSTNet 使用的多变量汇率 benchmark。原始说明把它描述为八个国家/地区的
每日汇率，包括澳大利亚、英国、加拿大、瑞士、中国、日本、新西兰和新加坡。汇率序列同时
具有长期趋势、制度变化、市场冲击和跨货币相关性。

当前 CSV 已把货币名称匿名化为 `0..6, OT`。因为文件本身没有保存“编号到货币”的映射，
分析当前文件时不应擅自声称某一编号对应某个国家。当前 size 为 7,588 天、dim 为 8，
从 `1990-01-01` 到 `2010-10-10`，相邻时间严格为 1 天，没有重复或 `NaN/Inf`。

原始 LSTNet 背景常写 1990-2016，但当前标准 forecasting CSV 截止 2010-10-10；本项目的 size、
split 和时间特征必须以当前 CSV 为准。

#### 2.1.9 Traffic

Traffic 来源于 California Department of Transportation 的 PeMS 传感器系统。LSTNet 数据说明
将其描述为 San Francisco Bay Area 高速公路传感器的小时级 road occupancy rate；每个通道
对应一个传感器，数值通常在 0 到 1 之间。当前 CSV 实际最小值为 0、最大值为 0.724，
与占有率比例的解释一致。

当前文件包含 `0..860, OT` 共 862 个匿名传感器通道、17,544 个严格连续的小时点，时间字段
从 `2016-07-01 02:00:00` 到 `2018-07-02 01:00:00`，没有重复或 `NaN/Inf`。该时间范围与
上游 LSTNet README 对原始数据的文字描述并不完全一致，现有文件无法证明时间戳经过了何种
转换；因此背景中的 PeMS 来源说明和当前 CSV 时间范围必须分开报告。

#### 2.1.10 CSV 实检后的统一协议表

下表中的定义为：

- `dim`：数值信号列数，不包含 `date`；
- `size`：CSV 数据行数，不包含表头；
- `split`：代码实际用于 train/validation/test 目标区间的点数；
- `frequency`：目录声明并由 CSV 主时间间隔核验的采样频率；
- `prediction length`：预测步数，括号中是按 frequency 换算的物理时长；
- `domain`：任务所属现实领域。

九个任务当前都按 multivariate-to-multivariate 协议运行，即 dim 个信号全部作为历史输入，
模型也同时预测未来的 dim 个信号；`date` 只用于生成时间特征，不属于预测维度。

| 数据集 | dim | size | split 比例 | split 点数 T/V/Test | frequency | prediction length | domain |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| ETTh1 | 7 | 17,420 | 6:2:2 | 10,452 / 3,484 / 3,484 | 1 小时 | 96/192/336/720（4/8/14/30 天） | 电力变压器负荷与油温 |
| ETTh2 | 7 | 17,420 | 6:2:2 | 10,452 / 3,484 / 3,484 | 1 小时 | 96/192/336/720（4/8/14/30 天） | 电力变压器负荷与油温 |
| ETTm1 | 7 | 69,680 | 6:2:2 | 41,808 / 13,936 / 13,936 | 15 分钟 | 96/192/336/720（1/2/3.5/7.5 天） | 电力变压器负荷与油温 |
| ETTm2 | 7 | 69,680 | 6:2:2 | 41,808 / 13,936 / 13,936 | 15 分钟 | 96/192/336/720（1/2/3.5/7.5 天） | 电力变压器负荷与油温 |
| Weather | 21 | 52,696 | 7:1:2 | 36,887 / 5,269 / 10,540 | 名义 10 分钟；有 2 个异常间隔 | 96/192/336/720（16/32/56/120 小时） | 气象与大气观测 |
| Electricity | 321 | 26,304 | 7:1:2 | 18,412 / 2,630 / 5,262 | 1 小时 | 96/192/336/720（4/8/14/30 天） | 多客户用电负荷 |
| ILI | 7 | 966 | 7:1:2 | 676 / 96 / 194 | 1 周 | 24/36/48/60（24/36/48/60 周） | 公共卫生与流感样病例监测 |
| ExchangeRate | 8 | 7,588 | 7:1:2 | 5,311 / 758 / 1,519 | 1 天 | 96/192/336/720（96/192/336/720 天） | 外汇市场 |
| Traffic | 862 | 17,544 | 7:1:2 | 12,280 / 1,754 / 3,510 | 1 小时 | 96/192/336/720（4/8/14/30 天） | 高速公路占有率/交通流 |

Weather 的 120 小时即 5 天；ILI 的 24/36/48/60 周分别约为 5.5/8.3/11.0/13.8 个月，
但实验索引始终使用“步数”，不会按自然月重新采样。

#### 2.1.11 Split 与处理后 NPY 长度的区别

上表列的是不重复的目标时间区间。`prepare_dataset()` 为了让 validation/test 的第一个样本
拥有历史上下文，会把左侧 `input_length` 个点复制进对应 NPY。对当前已生成的 NPY 做实际
shape 检查后，文件长度如下：

| 数据集 | train_data.npy | val_data.npy | test_data.npy | 左侧上下文 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1/ETTh2 | 10,452 | 3,580 | 3,580 | 96 |
| ETTm1/ETTm2 | 41,808 | 14,032 | 14,032 | 96 |
| Weather | 36,887 | 5,365 | 10,636 | 96 |
| Electricity | 18,412 | 2,726 | 5,358 | 96 |
| ILI | 676 | 120 | 218 | 24 |
| ExchangeRate | 5,311 | 854 | 1,615 | 96 |
| Traffic | 12,280 | 1,850 | 3,606 | 96 |

这些上下文点只用于构造输入，不应再次计入 validation/test 目标样本量。

#### 2.1.12 CSV 一致性与质量结论

实际使用 Pandas 对九份 CSV 做了逐文件检查，结论如下：

1. 九份文件的 size、dim 均与 `catalog.json` 完全一致；
2. 九份文件的数值列都能转成数值，没有 Pandas `NaN` 或正负无穷；
3. 除 Weather 外，其余八份文件时间戳严格递增、无重复，所有相邻间隔一致；
4. Weather 有一个重复时间戳、一个 100 分钟间隔、81 个 `-9999` 哨兵字段和三个乱码单位列；
5. Electricity/Traffic 的零值可能是真实零负荷/零占有率，不能不加判断地当成缺失值；
6. Electricity 与论文草稿中“家庭电参量”身份仍有冲突，当前代码和 CSV 明确对应 321 客户
   LTSF Electricity，正式报告必须使用这个准确名称。

高维画像还有一项与训练不同的设置：Electricity 和 Traffic 训练使用全部 321/862 通道，
但初始 U/M 画像分别均匀选择 64 个通道。`--all-channels` 可以关闭画像采样限制。

### 2.2 预实验配置 `pre_experiments/config.json`

[`pre_experiments/config.json`](pre_experiments/config.json) 控制三类参数：

- `profiler`：AR 阶数、频带数量、频谱峰阈值、最大窗口数等；
- `capacity`：深度候选、宽度组候选、二维网格、饱和容差和默认种子；
- `training/efficiency`：训练超参数、指标、效率预热和计时次数。

当前容量候选为：

```text
depth:      D = {1, 2, 4, 8}
width:      W = {1, 2, 4, 8}
joint:      D = {2, 4, 8}, W = {2, 4, 8}
epsilon:    0.01
```

`W` 是“宽度组数”，真实中间层宽度为：

```text
actual_width = width_group * model.width_unit
```

因此不同模型的 `W=4` 不代表相同的神经元数量，更不代表相同 FLOPs。

### 2.3 模型注册表 `Baselines/registry.json`

[`Baselines/registry.json`](Baselines/registry.json) 负责把统一的 `depth`、`width` 映射到
每个模型自己的配置类。当前映射如下：

| 模型 | RAW 深度 | RAW 宽度 | width_unit | RAW width_group |
| --- | ---: | ---: | ---: | ---: |
| Crossformer | 2 | 2048 | 256 | 8 |
| PatchTST | 1 | 1024 | 128 | 8 |
| TimesNet | 1 | 1024 | 128 | 8 |
| iTransformer | 1 | 1024 | 128 | 8 |
| TimeMixer | 1 | 1024 | 128 | 8 |
| WPMixer | 2 | 1024 | 128 | 8 |
| TimeFilter | 2 | 256 | 64 | 4 |
| MultiPatchFormer | 1 | 512 | 64 | 8 |

例如 TimeFilter 宽度扫描 `W={1,2,4,8}` 对应真实 `intermediate_size={64,128,256,512}`，
其 RAW 位于 `W=4`；Crossformer 则对应 `{256,512,1024,2048}`，RAW 位于 `W=8`。

### 2.4 正式实验配置 `pre_experiments/benchmark_config.json`

[`pre_experiments/benchmark_config.json`](pre_experiments/benchmark_config.json) 指定正式 RAW
实验的种子、epoch 和 batch size：

| 模型 | Epoch | 默认 Batch |
| --- | ---: | ---: |
| Crossformer | 20 | 32 |
| PatchTST | 100 | 64 |
| TimesNet | 10 | 32 |
| iTransformer | 10 | 32 |
| TimeMixer | 10 | 32 |
| WPMixer | 10 | 64 |
| TimeFilter | 10 | 32 |
| MultiPatchFormer | 20 | 32 |

Electricity 的 batch size 最多为 16，Traffic 最多为 8。三个正式种子为
`3407/3408/3409`。

要注意一个实验解释边界：正式配置按模型设置不同的 epoch 和 batch，但
[`run_capacity_sweep.py`](pre_experiments/run_capacity_sweep.py) 仍统一从
`pre_experiments/config.json` 读取 Adam 学习率 `2e-4`、weight decay `5e-4` 和早停耐心
`10`。因此当前代码并不是逐模型完整复制所有官方优化器协议，而是“模型相关训练预算 +
统一 BasicTS 优化器设置”。报告公平性时应准确描述这一点。

## 3. 数据预处理代码

入口是 [`pre_experiments/prepare_dataset.py`](pre_experiments/prepare_dataset.py)，真正逻辑在
[`src/adawd_preexp/data.py`](src/adawd_preexp/data.py)。

### 3.1 CSV 读取与检查

`_read_forecasting_csv()` 执行以下操作：

1. 用 Pandas 读取 CSV；
2. 取出 `date` 列并严格转换为时间；
3. 其余列严格转换为数值；
4. 输出形状为 `[time, channel]` 的 `float64` 数组。

`--strict-shape` 会使行数或通道数不匹配直接报错。正式数据准备通过
[`pre_experiments/run_preexperiments.py`](pre_experiments/run_preexperiments.py) 调用时会自动
加上该选项。

### 3.2 时间特征

`_timestamp_features()` 为每个时间点生成四维特征：

```text
time_of_day, day_of_week, day_of_month, day_of_year
```

四项都被缩放到大致 `[0,1]`。当前八个模型中只有 TimesNet 在
`capacity.build_model()` 中被设置为实际读取 `inputs_timestamps`；其他模型虽然数据文件中
存在 timestamp 数组，但当前适配器不会把它作为模型输入。

### 3.3 时间顺序切分

数据不随机打乱后再切分，而是按时间顺序切成 train、validation、test。ETT 使用
`0.6/0.2/0.2`，其他当前数据集使用 `0.7/0.1/0.2`。

validation 和 test 的 NPY 会在左侧额外保留一个 `input_length` 的上下文：

```text
train = [0, train_end)
val   = [train_end - input_length, val_end)
test  = [val_end - input_length, end)
```

这段重叠用于构造验证集和测试集的第一个历史输入窗口，不是把验证/测试目标泄漏到训练集。

输出包括：

```text
datasets/processed/<dataset>/
  train_data.npy / val_data.npy / test_data.npy
  train_timestamps.npy / val_timestamps.npy / test_timestamps.npy
  meta.json
```

## 4. U/M 数据特性预实验

### 4.1 什么是“局部单元”

[`src/adawd_preexp/profiler.py`](src/adawd_preexp/profiler.py) 先沿时间轴滑窗，然后对每个被
选中的通道分别产生一行记录。因此一行 `windows.csv` 表示：

```text
局部单元 = 一个时间窗口 + 一个通道
unit_id   = segment:window_start:channel
```

例如 ETTh1 默认最多取 512 个窗口、每个窗口 7 个通道，所以有 `512 * 7 = 3584`
个局部单元。绘图中的“样本数”或“Local units”指这些行，不是独立时间窗口数。

默认 `profile_dataset.py` 优先读取完整原始 CSV，此时 `segment=raw`。传入 `--processed`
后才分别读取 train/val/test，并保证滑窗不跨越三个 split 的边界。

### 4.2 预处理：稳健缩放与去趋势

对每个窗口内的单通道序列，代码先处理非有限值，再使用中位数和 MAD 做稳健缩放：

```text
z_t = (x_t - median(x)) / (1.4826 * MAD(x) + epsilon)
```

频谱相关指标还会用最小二乘直线去除线性趋势。这样做减少了不同通道原始量纲对 U/M 的
影响。常量窗口最终会得到 `U=0, M=0`，对应行为已有自动测试。

### 4.3 状态更新需求 U

每个局部单元计算三个 `[0,1]` 分量：

1. `u_change`：比较窗口前半段和后半段的中位数、稳健尺度是否改变；变化越大越接近 1。
2. `u_spectral`：分别计算前后半段归一化功率谱，再用 Jensen-Shannon divergence 衡量频谱漂移。
3. `u_surprise`：在前半段拟合带 ridge 的 AR 模型，预测后半段前几个点；AR 误差相对“用前半段中位数预测”的基线越大，surprise 越高。

最终：

```text
U = (u_change + u_spectral + u_surprise) / 3
```

`U` 高表示这个局部窗口发生了更强的状态改变、频谱改变或短期不可预测性。论文假设是这类
局部单元可能需要更深的变换链路，但 `U` 本身不是预测误差，也不能直接证明需要更深网络。

### 4.4 模式多样性 M

`M` 由以下分量构成：

1. `m_peak`：去趋势频谱中显著峰的数量，达到配置的饱和峰数后记为 1。
2. `m_band`：把频谱按几何间隔划分为六个频带，计算归一化频带能量熵；能量分布越多样越接近 1。
3. `m_channel`：对同一窗口的多通道残差做 SVD，用协方差特征值熵得到有效秩，再归一化到 `[0,1]`。

`m_channel` 描述的是整个多通道窗口的联合结构，所以同一窗口内各个被选通道会共享相同的
`channel_effective_rank` 和 `m_channel`；`m_peak`、`m_band` 则是各通道分别计算。

多变量数据：

```text
M = (m_peak + m_band + m_channel) / 3
```

单变量画像无法定义跨通道有效秩，因此：

```text
M = (m_peak + m_band) / 2
```

`M` 高表示该局部单元包含更多频率结构、频带能量分散或跨通道独立结构。论文假设是这类
局部单元可能需要更宽的中间表示，同样需要后续容量实验验证。

### 4.5 分桶与汇总

`profile_segments()` 在每个数据集内部用 1/3、2/3 分位数分别对 U 和 M 分桶：

```text
score <= 1/3 quantile  -> low
score >= 2/3 quantile  -> high
otherwise              -> mid
```

[`pre_experiments/profile_dataset.py`](pre_experiments/profile_dataset.py) 输出：

- `windows.csv`：每个局部单元的六个分量、U/M、通道、窗口位置和分桶；
- `summary.json`：各分量的 mean、总体标准差、P10/P50/P90、IQR，以及 U/M Spearman 相关。

分桶是数据集内部的相对等级。一个数据集的 `U=high` 不一定比另一个数据集的 `U=mid`
绝对值更高。

## 5. 静态 PDF：面向时间异质性的六面板图

绘图代码是 [`pre_experiments/plot_profiles.py`](pre_experiments/plot_profiles.py)。默认输出
`windows.csv` 同目录下的 `profile_diagnostics.pdf`。当前图不再使用每个数据集自己的最小/最大
值自动铺满横轴，而是所有 U/M 热图、轨迹、散点和箱线图固定使用 `[0,1]` 坐标。

这项修改对应预实验的真实目标：证明同一个数据集的不同时间片段有不同局部状态更新程度和
模式多样性，而不是证明分数看起来覆盖很大的画布。

### 5.1 U/M 时间-通道热图

- 横轴：时间窗口的相对顺序，从 0% 到 100%；
- 纵轴：原始通道 ID；
- 每个格子：一个“时间窗口 x 通道”局部单元；
- 颜色：U 或 M，固定 0-1 色标。

竖向色带说明许多通道在相同时间段共同变化，横向色带说明某些通道持续具有较高或较低需求，
局部亮斑说明特定时间/通道的局部复杂性。高维 Electricity 和 Traffic 的当前热图仍只画
画像抽取的 64 个通道，不能替代全通道敏感性图。

### 5.2 按时间着色的 U-M 平面

- 横轴：U，0-1；
- 纵轴：M，0-1；
- 每个点：一个局部单元；
- 点颜色：相对时间，早期为 0，晚期为 1；
- 黑色虚线：U/M 的数据集内中位数。

相比按 `U_bucket/M_bucket` 着色，时间颜色能直接检查点云是否随时间迁移。颜色不表示模型容量，
也不表示完整四象限；它只表示该局部单元来自时间轴的哪个位置。

### 5.3 U/M 时间轨迹

- 横轴：相对时间窗口顺序 0-1；
- 纵轴：U 或 M，0-1；
- 实线：该窗口跨通道的均值；
- 阴影：该窗口跨通道的 P10-P90 区间。

实线随时间起伏，才是“不同时间片段平均需求不同”的直接证据。阴影宽度则反映同一个时间
片段内部通道之间的差异，不能把通道差异误读成时间差异。

### 5.4 早中晚四段箱线图

- 横轴：0-25%、25-50%、50-75%、75-100% 四个时间区间；
- 纵轴：U/M，0-1；
- 蓝色：U；橙色：M；
- 箱体、中位线和须：对应时间区间内所有局部单元的分布摘要。

四分位区间只是可视化分组，不是显著性检验。正式结论还需要窗口级效应量、bootstrap 或
混合效应模型，以及后续 U→`d_sat`、M→`w_sat` 的容量关联。

### 5.5 “数值偏小”应该如何解读

U/M 是多个 `[0,1]` 描述分量的平均值，不是概率，也没有“0.5 才算复杂”的阈值。当前
`0.2-0.4` 的均值可能对应稳定但非恒定的局部结构；绝对均值不能单独证明或否定多样性。

更重要的是窗口级统计。当前九份 profile 的窗口均值 P10-P90 和“窗口间方差占局部总方差比例”
如下：

| 数据集 | U 窗口 P10-P90 | U 时间方差占比 | M 窗口 P10-P90 | M 时间方差占比 |
| --- | --- | ---: | --- | ---: |
| ETTh1 | 0.256-0.436 | 0.433 | 0.451-0.592 | 0.251 |
| ETTh2 | 0.288-0.461 | 0.258 | 0.381-0.618 | 0.294 |
| ETTm1 | 0.300-0.416 | 0.418 | 0.314-0.551 | 0.447 |
| ETTm2 | 0.289-0.422 | 0.285 | 0.337-0.568 | 0.333 |
| Weather | 0.261-0.377 | 0.107 | 0.188-0.301 | 0.099 |
| Electricity | 0.247-0.288 | 0.065 | 0.274-0.309 | 0.036 |
| ILI | 0.288-0.514 | 0.609 | 0.167-0.389 | 0.725 |
| ExchangeRate | 0.327-0.448 | 0.185 | 0.314-0.525 | 0.395 |
| Traffic | 0.221-0.260 | 0.079 | 0.322-0.374 | 0.076 |

这里的“时间方差占比”是窗口均值差异贡献的描述性比例，不是显著性 p 值。它表明当前证据
并不支持“九个数据集的时间异质性同样强”：ILI、ETT 和部分 ExchangeRate 的时间变化更清楚；
Electricity、Traffic 和 Weather 的窗口均值变化较弱，主要差异可能来自通道或窗口内部结构。

因此当前预实验可以支持的严谨结论是：

> U/M 在部分数据集和部分时间片段上显示出稳定的局部异质性，且这种异质性可被时间热图和
> 窗口轨迹定位；是否对所有数据集成立、以及是否对应动态深度/宽度收益，仍需显著性检验和
> 容量实验验证。

不能仅依据 U/M 均值较大、分位桶数量接近三等分，或直方图视觉宽度来宣称“已证明多样性”。

## 6. 深度、宽度和联合容量扫描

核心规划函数位于 [`src/adawd_preexp/capacity.py`](src/adawd_preexp/capacity.py)，执行入口是
[`pre_experiments/run_capacity_sweep.py`](pre_experiments/run_capacity_sweep.py)。

### 6.1 四种 axis

- `raw`：只产生一个 RAW 深宽组合；
- `depth`：宽度固定为 RAW，只扫描 `D={1,2,4,8}`；
- `width`：深度固定为 RAW，只扫描 `W={1,2,4,8}`；
- `joint`：扫描 `D={2,4,8} x W={2,4,8}` 共 9 个组合。

每个 run 还会与 horizon、seed 做笛卡尔积，并形成唯一 ID：

```text
<model>__<dataset>__h<horizon>__<axis>__d<depth>__wg<width_group>__s<seed>
```

上层一键入口 [`pre_experiments/run_preexperiments.py`](pre_experiments/run_preexperiments.py)
默认是 `ETTh1 + PatchTST + 最短 horizon + seed 42`。但如果直接调用底层
`run_capacity_sweep.py` 且不传 `--seeds`，`plan_sweep()` 会读取 `config.json` 中的
`3407/3408/3409`。这两个默认值不同，复现实验时应显式传入种子，避免混淆。

### 6.2 从统一容量值构造模型

`build_model()` 根据注册表动态导入模型和配置类，并注入：

```text
input_len, output_len, num_features,
num_layers=run.depth,
intermediate_size=run.width
```

TimesNet 额外启用 timestamp；TimeFilter 根据数据集设置 patch length。所有模型最终都必须
满足统一接口：

```text
输入  [batch, input_time, channel]
输出  [batch, output_time, channel]
```

### 6.3 单次执行发生了什么

`execute()` 的顺序为：

1. 建立 `manifest.json`，先标记 `status=running`；
2. 构造一个未训练模型，测参数量、推理延迟、吞吐量、峰值显存和可支持时的 FLOPs；
3. 建立 `BasicTSForecastingConfig`；
4. 使用 BasicTS 训练，并在每个 epoch 后验证；
5. 依据验证集 MAE 保存最佳 checkpoint，早停耐心为 10；
6. 训练结束后重新加载最佳 checkpoint，在 test split 上评估；
7. 读取 `test_metrics.json`，把指标与效率写回 manifest；
8. 成功则记为 `complete`，异常则记为 `failed`。

训练 loss 默认是 masked MAE。评价指标由 `metric_scale` 控制：`normalized` 只计算
MAE/MSE/RMSE，`original` 计算配置中的 MAE/MSE/RMSE/MAPE/WAPE。`artifact_policy` 不控制
指标种类，只控制是否保留预测数组和 checkpoint。当前容量一键任务和正式 benchmark 都未
覆盖默认的 `metric_scale=normalized`，因此实际记录 MAE/MSE/RMSE。

效率字段包括：

- `total_parameters`、`trainable_parameters`；
- `flops_per_batch`，算子不支持时允许为 `null`；
- batch=1 的推理 latency median/P90 和 throughput；
- benchmark 与训练 CUDA peak memory；
- 总训练 wall time。

不同深度或宽度标签不等价，因此比较容量收益时要同时报告这些真实效率指标。

## 7. 从局部误差到饱和深度/宽度

### 7.1 为什么需要保存完整预测

[`pre_experiments/build_local_losses.py`](pre_experiments/build_local_losses.py) 需要读取每个 run 的：

```text
inputs.npy, prediction.npy, targets.npy
```

因此容量预实验必须使用 `artifact_policy=full`。正式 benchmark 默认
`artifact_policy=metrics`，成功后会删除整个 checkpoint 目录，所以正式 RAW manifest 不能
再直接用于局部损失分析。

### 7.2 局部误差定义

在这一步，局部单元改为：

```text
局部单元 = 一个测试滑窗样本 + 一个通道
unit_id   = test:sample_index:channel
```

对同一个测试样本和通道，沿完整预测 horizon 求平均：

```text
loss_mae = mean_h |prediction_h - target_h|
loss_mse = mean_h (prediction_h - target_h)^2
```

同时对该样本的历史输入窗口重新计算 U/M。这样每个容量候选都能与完全相同的局部单元对齐。
不能用数据集级平均误差替代这一步，否则无法判断某个局部区域的饱和容量。

### 7.3 单轴饱和容量

[`src/adawd_preexp/saturation.py`](src/adawd_preexp/saturation.py) 对每个固定的
`dataset/model/seed/horizon/unit_id` 执行：

```text
best_loss = 所有候选容量中的最小 loss
eligible  = loss <= (1 + epsilon) * best_loss
saturation = eligible 中最小的容量
```

默认 `epsilon=0.01`，即允许比最优局部误差最多高 1%。

- 深度实验输出 `d_sat`：满足近最优条件的最小 depth；
- 宽度实验输出 `w_sat`：满足近最优条件的最小 width_group。

选择“最小近最优容量”而不是绝对最小误差对应容量，是为了避免把极小的随机误差改善误判为
必须增加容量。

### 7.4 联合饱和容量

联合扫描先找近最优组合，再计算：

```text
capacity_cost = depth * width_group
```

优先选 cost 最小者；cost 相同时依次选 depth、width_group 更小者。这里的 cost 只是用于
网格内排序的容量代理，不是参数量或 FLOPs，论文结果仍应报告实测计算成本。

### 7.5 分桶、四象限与回归

`analyze_saturation.py` 输出：

- `depth_saturation.csv` 与 `width_saturation.csv`：逐局部单元的饱和容量；
- `depth_bucket_summary.csv`：每个 dataset/model 下 U low/mid/high 的 d_sat 统计；
- `width_bucket_summary.csv`：每个 dataset/model 下 M low/mid/high 的 w_sat 统计；
- `joint_quadrant_summary.csv`：Q1-Q4 的 d_sat、w_sat、cost 统计；
- `diagnostics.json`：`d_sat ~ U + M`、`w_sat ~ U + M` 的 OLS 结果。

四象限使用每个数据集自己的 U/M 中位数：

```text
Q1 = U低 M低
Q2 = U高 M低
Q3 = U低 M高
Q4 = U高 M高
```

等于中位数的值被划为“高”。当前 `diagnostics.json` 中的 OLS 是把输入文件中的可用行合并
后做普通最小二乘，只含截距、U、M，没有自动加入 dataset/model/seed 固定效应，也没有
cluster-robust 标准误。正式论文统计若需要控制这些层级，应在此基础上扩展，不能把当前
pooled OLS 写成已经完成的多层模型。

## 8. 容量结果应该怎样画图

本节是根据现有 CSV 的正确作图方式，当前仓库尚未实现这些静态图。

### 8.1 U 与饱和深度趋势图

数据源：`depth_bucket_summary.csv`。

- 横轴：`U_bucket`，固定顺序 `low -> mid -> high`；
- 纵轴：平均 `d_sat`；
- 误差线：建议使用跨局部单元或跨 seed 的置信区间，但必须在图注说明计算层级；
- 分面/曲线：dataset 和 model 不应无标识地混在一起，可按数据集分面、模型着色。

验证目标是横轴从 low 到 high 时，纵轴是否总体上升。横轴不是 U 的原始连续数值。

### 8.2 M 与饱和宽度趋势图

数据源：`width_bucket_summary.csv`。

- 横轴：`M_bucket`，固定顺序 `low -> mid -> high`；
- 纵轴：平均 `w_sat`，单位是 width_group；
- 如果要展示真实宽度，应另算 `w_sat * width_unit`，并明确单位；
- 不同模型的 width_group 相同不表示真实宽度或计算量相同。

### 8.3 连续散点图

数据源：`depth_saturation.csv` 或 `width_saturation.csv`。

建议分别绘制：

```text
图 A：横轴 U，纵轴 d_sat，颜色 M
图 B：横轴 M，纵轴 w_sat，颜色 U
```

由于 d_sat/w_sat 只取少数离散候选值，点会重叠，适合加入轻微 jitter、透明度或箱线/小提琴
辅助，但 jitter 后的坐标不能被解释为真实容量值。

### 8.4 Q1-Q4 联合容量图

数据源：`joint_quadrant_summary.csv`。

- 横轴：`Q1/Q2/Q3/Q4`；
- 纵轴：可分别画 mean d_sat、mean w_sat 或 mean capacity_cost；
- 建议把深度和宽度做成两个子图，不要在同一纵轴强行叠加不同单位；
- 预期模式：Q2 相对 Q1 更深，Q3 相对 Q1 更宽，Q4 两者都高。

### 8.5 容量-误差曲线

如果直接从 `local_losses.csv` 画模型容量曲线：

- 深度曲线横轴：`depth`；纵轴：固定局部单元集合上的 mean loss_mae 或 mean loss_mse；
- 宽度曲线横轴：`width_group` 或真实 `width`，二者必须在轴标题中区分；
- 联合实验适合热力图：横轴 width_group，纵轴 depth，颜色为 loss。

不要把 horizon 放在纵轴，也不要把 run index 当作容量轴。run index 只是调度顺序。

## 9. BasicTS 训练、归一化与指标

### 9.1 样本构造

[`Baselines/basicts/data/tsf_dataset.py`](Baselines/basicts/data/tsf_dataset.py) 用滑动窗口构造：

```text
inputs  = data[index : index + input_len]
targets = data[index + input_len : index + input_len + output_len]
```

一个样本的形状为：

```text
inputs:  [input_len, channels]
targets: [output_len, channels]
```

DataLoader 加入 batch 后成为 `[batch, time, channel]`。

### 9.2 ZScore 归一化

[`Baselines/basicts/scaler/z_score_scaler.py`](Baselines/basicts/scaler/z_score_scaler.py) 只用训练数据
拟合每个通道的 mean/std，然后对 train/val/test 的 inputs 和 targets 应用：

```text
z = (x - train_mean[channel]) / train_std[channel]
```

`norm_each_channel=True`，所以每个变量独立归一化。常数通道的 std 会被置为 1，避免除零。

当 `metric_scale=normalized` 时，`rescale=False`，评估前不会 inverse transform，最终指标位于
ZScore 空间。当 `metric_scale=original` 时，postprocess 才会把 inputs、prediction、targets
恢复到原始量纲后计算指标。

PatchTST、TimesNet、iTransformer、TimeMixer、WPMixer、TimeFilter、MultiPatchFormer 内部还可能
使用 RevIN。这是模型内部按样本做的可逆归一化，不替代外部训练集 ZScore；当前代码实际上是
“数据级 ZScore + 模型级 RevIN”的组合。

### 9.3 Loss 与评价指标

默认训练 loss 是 masked MAE：

```text
MAE  = mean(|prediction - target|)
MSE  = mean((prediction - target)^2)
RMSE = sqrt(MSE)
```

mask 会排除 `NaN` 等无效目标，并按有效元素数量归一化。正式汇总从每个 run 的
`metrics.overall` 读取 MAE/MSE/RMSE。

三个种子的汇总使用：

- `mean()`：种子均值；
- `std(ddof=1)`：样本标准差；
- 只有一个种子时 std 记为空，而不是错误地记为 0。

## 10. 八个 baseline 的代码结构

八个模型均接收 `[B,L,C]`，输出 `[B,H,C]`，其中 `B` 是 batch，`L` 是历史长度，
`H` 是预测长度，`C` 是通道数。容量扫描统一把 `num_layers` 当深度，把
`intermediate_size` 当宽度代理，但它们在不同架构中扮演的计算角色并不完全相同。

### 10.1 Crossformer

代码：[`crossformer_arch.py`](Baselines/basicts/models/Crossformer/arch/crossformer_arch.py)。

- 把每个通道的时间序列切成 patch；
- encoder 先做时间维注意力，再通过可学习 router 做跨通道两阶段注意力；
- 更深层通过 patch merging 构造更粗时间尺度；
- decoder 的多个尺度分别预测，最后把各尺度输出相加。

`num_layers` 同时影响多尺度 encoder 和 decoder 层数，`intermediate_size` 是两阶段注意力后
FFN 的中间维度。默认 hidden size 512、8 heads、patch length 16。

### 10.2 PatchTST

代码：[`patchtst_arch.py`](Baselines/basicts/models/PatchTST/arch/patchtst_arch.py)。

- 每个变量独立切成重叠 patch；
- patch token 经过 Transformer encoder；
- 展平全部 patch 表示，再用预测头映射到 horizon；
- 默认启用 RevIN，不启用 trend/seasonal 双分支 decomposition。

`num_layers` 是 encoder 层数，`intermediate_size` 是 Transformer FFN 宽度。默认 patch length
16、stride 8、hidden size 256、1 head。

### 10.3 TimesNet

代码：[`timesnet_arch.py`](Baselines/basicts/models/TimesNet/arch/timesnet_arch.py) 和
[`times_block.py`](Baselines/basicts/models/TimesNet/arch/times_block.py)。

- FFT 找出幅值最大的 top-k 周期；
- 对每个周期把一维时间序列重排成二维周期结构；
- 使用多核 Inception 2D convolution 提取周期内和周期间变化；
- 按频率幅值自适应加权多个周期输出。

`num_layers` 是 TimesBlock 数量；`intermediate_size` 是 Inception 卷积的中间通道数，而不是
Transformer FFN。TimesNet 是当前唯一显式使用四维 timestamp 特征的 baseline。

### 10.4 iTransformer

代码：[`itransformer_arch.py`](Baselines/basicts/models/iTransformer/arch/itransformer_arch.py)。

- 与普通时间 token Transformer 不同，它把每个变量的完整历史序列嵌入为一个 token；
- self-attention 在变量 token 之间建模跨变量关系；
- 每个变量 token 经线性头输出完整 horizon。

`num_layers` 是 encoder 层数，`intermediate_size` 是 FFN 宽度。默认 hidden size 256、1 head，
启用 RevIN。

### 10.5 TimeMixer

代码：[`timemixer_arch.py`](Baselines/basicts/models/TimeMixer/arch/timemixer_arch.py) 和
[`mixing_layers.py`](Baselines/basicts/models/TimeMixer/arch/mixing_layers.py)。

- 对输入连续下采样，形成多个时间尺度；
- 每个尺度做 seasonal/trend 分解；
- seasonal 从高分辨率向低分辨率 bottom-up 混合；
- trend 从低分辨率向高分辨率 top-down 混合；
- 每个尺度单独预测，最后求和。

`num_layers` 是 PastDecomposableMixing block 数量；`intermediate_size` 是 block 内 MLP 宽度。
默认下采样窗口 2、下采样 3 层、channel independence 开启。

### 10.6 WPMixer

代码：[`wpmixer_arch.py`](Baselines/basicts/models/WPMMixer/arch/wpmixer_arch.py)。

- 用 Haar wavelet 逐级分解 approximation 与 detail；
- 每个分辨率分支独立切 patch；
- MixerBlock 先沿 patch/token 维混合，再沿 hidden/channel 维混合；
- 预测各级 approximation/detail 后做逆 Haar 重建。

`num_layers` 是每个分辨率分支重复的 MixerBlock 数；`intermediate_size` 同时控制 token mixer
和 channel mixer 的隐藏宽度。默认 wavelet level 2、patch length 4、stride 2。

该适配器来源与 MIT 许可记录在 [`Baselines/SOURCE_LICENSE.md`](Baselines/SOURCE_LICENSE.md)。

### 10.7 TimeFilter

代码：[`timefilter_arch.py`](Baselines/basicts/models/TimeFilter/arch/timefilter_arch.py)。

- 每个通道切成不重叠 patch，并把“通道 x patch”展平成图节点；
- query/key 相似度形成动态图边；
- 每个节点只保留 top `keep_ratio` 的边，当前默认 0.5；
- 过滤后的邻接权重聚合 value，再经过 FFN；
- 最后按通道恢复 patch 表示并预测 horizon。

`num_layers` 是 FilteredGraphBlock 数量，`intermediate_size` 是图 block 的 FFN 宽度。默认
hidden size 128、4 heads。patch length 会由 `capacity.py` 按数据集设置，而不是全部固定为 4。

官方仓库在审查 commit 未声明软件许可，因此这里是按论文设计写的独立 BasicTS 实现，而不是
直接复制官方源码，详见 [`Baselines/SOURCE_LICENSE.md`](Baselines/SOURCE_LICENSE.md)。

### 10.8 MultiPatchFormer

代码：[`multipatchformer_arch.py`](Baselines/basicts/models/MultiPatchFormer/arch/multipatchformer_arch.py)。

- 用多组 patch length/stride 并行提取时间 token；
- 把不同分支投影到同样 patch 数后融合；
- causal temporal attention 在每个通道内部建模；
- channel attention 再建模变量之间的关系；
- horizon 被分成多个 segment，后续 segment 的预测头读取前面已生成 segment。

`num_layers` 会同时增加 temporal block 和 channel block，实际 block 数成对增长；
`intermediate_size` 是两个 encoder block 的 FFN 宽度。默认四个 patch 分支、hidden size 256、
8 heads、8 个预测 segment。

该模型同样是独立实现，许可边界见 [`Baselines/SOURCE_LICENSE.md`](Baselines/SOURCE_LICENSE.md)。

## 11. 正式 RAW benchmark 调度与恢复

[`pre_experiments/run_forecasting_benchmarks.py`](pre_experiments/run_forecasting_benchmarks.py)
按以下顺序生成计划：

```text
model -> dataset -> horizon -> seed
```

正式计划数：

```text
8 models * 9 datasets * 4 horizons * 3 seeds = 864 runs
```

Smoke 计划只取每个数据集最短 horizon、第一个 seed、1 epoch：

```text
8 models * 9 datasets = 72 runs
```

`--start-index/--stop-index` 只筛选计划表中的连续区间，适合分机器或分时段运行，不改变实验
定义。`--dry-run` 只写 `plan.csv`，不会启动训练。

恢复逻辑不只是看目录是否存在。只有 manifest 同时满足以下条件才跳过：

- `status=complete`；
- epochs 和 batch size 与当前计划相同；
- metric scale 和 artifact policy 与当前配置相同。

因此改变训练协议后，旧结果不会被错误当成已完成的新实验。

每个正式 run 实际仍调用 `run_capacity_sweep.py --axis raw --run-index 0`，所以预实验和正式实验
共享完全相同的模型构造、BasicTS 训练、指标与效率记录代码。

## 12. 正式结果表及其含义

[`pre_experiments/summarize_forecasting_benchmarks.py`](pre_experiments/summarize_forecasting_benchmarks.py)
生成四类文件：

### 12.1 `per_seed.csv`

一行表示一个 `model + dataset + horizon + seed`，包含该 seed 的 MAE/MSE/RMSE、epoch、batch
和 run_id。它是做显著性分析或配对种子比较的基础表。

### 12.2 `summary.csv`

一行表示一个 `model + dataset + horizon`，包含：

- 三个种子的数量和列表；
- 是否三个预期种子全部完成；
- MAE/MSE/RMSE 的 mean 和 sample std。

### 12.3 `coverage.csv`

它把原始 864-run 计划与已完成 manifest 左连接。`status=complete` 表示存在匹配结果，
否则是 `pending`。这张表用于检查覆盖率，不用于比较精度。

### 12.4 `Result.md`

它是 `summary.csv` 的 Markdown 展示版，每行显示模型、数据集、horizon、训练预算、种子数和
`mean +/- sample std`。

## 13. 正式 baseline 结果应该怎样画图

当前正式汇总代码没有生成静态图。若基于 `summary.csv` 画论文图，建议采用以下轴定义。

### 13.1 Horizon-误差曲线

- 横轴：forecast horizon，即 `output_length`；
- 纵轴：`MAE_mean`、`MSE_mean` 或 `RMSE_mean`，一次只用一个主指标；
- 曲线颜色：model；
- 分面：dataset；
- 误差线：对应指标的跨 seed sample std。

不要把不同采样频率的数据集直接放到同一“物理时间”横轴后仍只标 96/192/336/720。例如：

- ETTh1 的 horizon 96 是 96 小时；
- ETTm1 的 horizon 96 是 96 个 15 分钟点，即 24 小时；
- Weather 的 horizon 96 是 16 小时；
- ILI 的 horizon 24 是 24 周。

跨数据集比较时，横轴应标“预测步数”，或者先根据 `frequency_minutes` 转成明确的物理时长。

### 13.2 模型柱状图

- 横轴：model；
- 纵轴：固定 dataset 和 horizon 下的 MAE/MSE/RMSE mean；
- 误差线：跨三个 seed 的 sample std；
- 每张图或每个 panel 必须固定 dataset、horizon 和 metric scale。

不能把不同数据集的 normalized MSE 直接平均后当作唯一总排名，除非论文预先定义了聚合规则。

### 13.3 精度-效率 Pareto 图

效率字段来自各 run 的 manifest，适合画：

```text
横轴：参数量 / FLOPs / latency / peak memory 中的一项
纵轴：MAE 或 MSE
点颜色：model
点形状或分面：dataset/horizon
```

误差越低越好，成本通常也越低越好，所以较理想点位于左下。FLOPs 为 `null` 的模型不能用
参数量冒充 FLOPs，应在图中缺失或单独注明。

## 14. TensorBoard 训练曲线：横轴和纵轴

BasicTS 的 [`MeterPool`](Baselines/basicts/utils/meter_pool.py) 使用
`SummaryWriter.add_scalar(name, value, global_step=step)` 写曲线。

### 14.1 当前 epoch 训练模式

当前实验使用 `num_epochs`，所以：

- train 曲线横轴：epoch 编号；
- val 曲线横轴：验证记录编号 `epoch // val_interval`；当前 `val_interval=1`，数值等同 epoch；
- 纵轴：对应 tag 的 epoch 加权平均值，如 `train/loss`、`train/MAE`、`val/MAE`；
- `train/lr` 的纵轴是 learning rate。

loss/metric 按有效 target 元素数加权，不是简单对 batch 均值再做不加权平均。

### 14.2 test 曲线

一般 BasicTS 可以按 `test_interval` 在训练中写 test 曲线，横轴是 test 发生时的 epoch 记录。
但本仓库在 `run_capacity_sweep.py` 中把 `test_interval` 设为 `run_epochs + 1`，目的是避免训练
过程中反复看 test set。最终只在训练完成后加载最佳模型评估；此时不会按 epoch 写一条正式
test TensorBoard 曲线，最终 test 指标以 `test_metrics.json/manifest.json` 为准。

### 14.3 TensorBoard 文件是否保留

- 容量预实验默认 `artifact_policy=full`：checkpoint 目录和 TensorBoard event 文件保留；
- 正式 benchmark 默认 `artifact_policy=metrics`：读取指标后删除 checkpoint 目录，TensorBoard
  event 也随之删除，只在 manifest 中保留最终指标和效率。

因此正式实验结束后想画训练曲线，必须事先修改 artifact policy 或单独备份 event 文件；不能
期望从 `summary.csv` 恢复逐 epoch 曲线。

## 15. 常用命令与输出对应关系

### 15.1 数据画像及三联图

```bash
pixi run preexp-dataset
```

依次下载、严格预处理、画像并绘图。单数据集绘图：

```bash
pixi run python pre_experiments/plot_profiles.py \
  --profiles pre_experiments/results/profiles/ETTh1/windows.csv
```

### 15.2 容量计划和训练

```bash
pixi run preexp-depth-plan
pixi run preexp-width-plan
pixi run preexp-depth
pixi run preexp-width
```

二维联合实验需直接调用 `run_capacity_sweep.py --axis joint`。

### 15.3 局部损失和饱和容量

```bash
pixi run python pre_experiments/build_local_losses.py
pixi run python pre_experiments/analyze_saturation.py \
  --losses pre_experiments/results/local_losses.csv \
  --metric loss_mse \
  --epsilon 0.01
```

### 15.4 正式 baseline

```bash
# 只生成 864-run 计划
pixi run forecast-all-plan

# 72-run 兼容性检查
pixi run forecast-smoke

# 正式顺序运行并自动汇总
pixi run forecast-all
```

## 16. 结果解读时最容易出现的错误

1. 把 U/M 当成预测误差。U/M 是输入结构描述符，误差必须来自模型预测。
2. 把自动缩放后的图幅当成绝对离散度。当前诊断图固定 U/M 为 0-1，时间异质性应看窗口轨迹和窗口级统计。
3. 把一个点当成一个完整窗口。散点中的一个点是“窗口 x 通道”。
4. 把虚线当成 high 桶边界。虚线是中位数，high 桶从 2/3 分位开始。
5. 把颜色当成四象限。当前红/蓝/灰只是带优先级的高分桶提示。
6. 把 width_group 当成真实神经元数。真实宽度还要乘各模型的 width_unit。
7. 把 `depth * width_group` 当 FLOPs。它只是联合网格内的排序代理。
8. 用 dataset 平均误差定义局部饱和容量。饱和容量必须对同一 unit 比较全部候选。
9. 把 normalized MAE 与原始量纲 MAE 混用。当前正式结果是逐通道 ZScore 空间。
10. 把 smoke 精度当正式结果。Smoke 只有 1 epoch、1 seed、最短 horizon，只验证链路。
11. 在不同采样频率数据集间把相同 horizon 当成相同物理时长。
12. 认为正式 metrics 策略保留 checkpoint/TensorBoard；成功后这些目录会被删除。

## 17. 自动测试覆盖了什么

[`tests/`](tests) 中与本文最相关的检查包括：

- U/M 始终在 `[0,1]`，状态突变会提高 U，多频信号会提高 M；
- 常量信号得到 `U=0, M=0`；
- 画像窗口不跨 split，通道采样保留原通道 ID；
- 深度扫描确实为 `{1,2,4,8}` 且固定 RAW 宽度；
- 三个 2025 模型能按统一配置构造并输出正确形状；
- 正式计划严格包含 864 runs，smoke 计划严格包含 72 runs；
- 三种 seed 的均值、样本标准差和 coverage 计算正确；
- 训练协议不同的旧 manifest 不会混入新汇总；
- 最后一个不足 batch 的测试结果仍按连续 offset 正确写入 NPY。

这些测试能保证规划、形状和统计管道的基本一致性，但不能替代完整 GPU 训练、跨机器复现、
论文级显著性检验以及 Electricity/Traffic 全通道敏感性实验。
