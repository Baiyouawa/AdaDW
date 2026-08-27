# 深度/宽度模型预实验与八模型对比实验方案

本文只整理实验逻辑、当前代码、正式配置、执行步骤和预期结果，不包含任何新训练结果，也不要求现在启动训练。

文中所说的 `RAW` 不是“原始量纲误差”。正式 baseline 的结构参数由 [`Baselines/registry.json`](Baselines/registry.json) 中每个模型独立的 `benchmark_config` 明确给出，epoch 和 batch size 按 [`pre_experiments/benchmark_config.json`](pre_experiments/benchmark_config.json) 设置，而优化器、学习率、权重衰减、早停和评估流程由 BasicTS 统一。深宽容量预实验另有参考深度/宽度，不能与正式 RAW 配置混用。本地标准协议并不等于逐模型、逐数据集、逐 horizon 完整复刻每个官方仓库的所有超参数。

## 1. 先回答两个核心问题

### 1.1 深度/宽度预实验应该怎么做

应该对相同的模型、数据集、预测长度、随机种子和局部测试单元，分别构造完整的深度曲线和宽度曲线：

- 深度实验只改变 `depth`，宽度固定为该模型的 RAW 宽度；
- 宽度实验只改变 `width_group`，深度固定为该模型的 RAW 深度；
- 对每个“测试输入窗口 + 通道”计算输入的 `U/M` 和各容量模型的局部预测误差；
- 从误差曲线中找出最小近最优深度 `d_sat` 和最小近最优宽度 `w_sat`；
- 检验 `U` 越大时 `d_sat` 是否越大，`M` 越大时 `w_sat` 是否越大；
- 同时控制另一个描述符，检验这种对应是否具有轴向特异性，而不是总难度同时推高深度和宽度。

不能先看到某个数据集的平均 `U` 高，就只给它运行深模型；也不能因为平均 `M` 低，就只给它运行窄模型。那样没有同一单元上的容量反事实，无法证明 `U -> depth` 或 `M -> width`。所有纳入正式验证的数据集都应使用同一组预注册候选，然后再观察饱和容量是否随 `U/M` 改变。

这里能“确保”的是对应关系被正确检验，而不是确保实验一定得到正相关。`U/M` 不能在看过测试误差后被重新定义，候选范围、分桶和统计模型也不能为了得到预期方向而事后调整。

### 1.2 是否应该覆盖 8 个模型和 9 个数据集

如果论文要声称结论跨数据集、跨架构成立，正式的一维深度扫描和一维宽度扫描应覆盖全部 `8 models x 9 datasets`。但不建议未经分层就把 4 个 horizon、两个单轴和完整二维网格一次性全部展开，因为 run 数会迅速膨胀。

推荐的范围是：

1. 全部 8 模型、9 数据集、最短注册 horizon、3 个种子完成深度和宽度单轴主实验；
2. 二维联合网格用于验证 `U/M` 四象限，可先做全部数据集上的代表模型，再根据论文是否强调“联合动态宽深”决定是否扩展到全部 8 个模型；
3. 正常 baseline 对比独立使用 RAW 配置，在全部 8 模型、9 数据集、4 个 horizon、3 个种子上运行；
4. 容量实验与 RAW baseline 可以在同一批计算任务中调度，但必须使用独立计划、独立结果目录和明确的训练协议，不能因为某组参数数值相同就直接混用结果。

按照当前候选集，最短 horizon 的正式单轴实验规模为：

```text
depth: 8 x 9 x 1 horizon x 3 seeds x 4 depths = 864 runs
width: 8 x 9 x 1 horizon x 3 seeds x 4 widths = 864 runs
两条单轴合计                                      = 1,728 runs

当前 joint: 8 x 9 x 1 horizon x 3 seeds x 9 pairs = 1,944 runs
正式 RAW:    8 x 9 x 4 horizons x 3 seeds          =   864 runs
```

若深度、宽度和当前 joint 都直接扩展到 4 个 horizon，容量部分会变成 `3,456 + 3,456 + 7,776 = 14,688` runs，再加 RAW 共 15,552 runs。这也是推荐把“最短 horizon 的全覆盖主检验”和“代表性 horizon 稳健性检验”分开的原因。

深度和宽度计划中各自都包含一次数值上等于 RAW 的组合，但当前 run 的 `axis` 不同、run ID 不同，代码仍会把它们当成独立训练。是否复用必须由新的统一计划显式保证协议一致，不能人工拼接 manifest。

## 2. 数据画像与容量假设怎样对应

### 2.1 已完成的数据画像定义

画像核心实现在 [`src/adawd_preexp/profiler.py`](src/adawd_preexp/profiler.py)。对每个局部“时间窗口 + 通道”计算：

```text
U = mean(u_change, u_spectral, u_surprise)
M = mean(m_peak, m_band, m_channel)     # 多变量画像
M = mean(m_peak, m_band)                # 单变量画像
```

- `u_change`：窗口前后半段的位置和尺度变化；
- `u_spectral`：前后半段归一化频谱的漂移；
- `u_surprise`：前半段 AR 模型对后半段的预测意外程度；
- `m_peak`：去趋势频谱中的显著峰数量；
- `m_band`：多个频带之间的能量熵；
- `m_channel`：同一窗口所选通道的有效秩贡献。

因此假设不是“`U` 就等于层数、`M` 就等于神经元数”，而是：更高的局部状态更新需求应更常需要更深的变换链，更高的局部模式多样性应更常需要更宽的中间表征。

### 2.2 前文九数据集结果给出的先验排序

当前 [`pre_experiments/results/Result.md`](pre_experiments/results/Result.md) 中的描述性均值如下。这些值可用于预注册跨数据集的方向性预期，但不能代替局部容量验证。

| 数据集 | U 均值 | M 均值 | 对正式容量实验的先验角色 |
| --- | ---: | ---: | --- |
| ETTh1 | 0.335 | 0.515 | 高 M，重点观察宽度饱和是否右移 |
| ETTh2 | 0.372 | 0.499 | U/M 都较高，重点观察联合高容量 |
| ETTm1 | 0.352 | 0.432 | U/M 相关较高，重点检验偏回归特异性 |
| ETTm2 | 0.351 | 0.444 | M 离散度大，适合宽度和联合验证 |
| Weather | 0.312 | 0.240 | M 较低，作为低 M 对照，但时间异质性较弱 |
| Electricity | 0.269 | 0.292 | U/M 较低的高维对照，当前只画像 64 通道 |
| ILI | 0.397 | 0.274 | 最高 U、较低 M，最适合检验“深而不必宽” |
| ExchangeRate | 0.384 | 0.421 | 高 U、较高 M 且二者相关低，适合轴向解耦 |
| Traffic | 0.308 | 0.345 | U 居中、M 居中且 U 时间变化较明显的高维对照，当前只画像 64 通道 |

跨数据集最直接的预期是：

- ILI、ExchangeRate、ETTh2 的平均 `d_sat` 倾向更高；
- ETTh1、ETTh2、ETTm2 的平均 `w_sat` 倾向更高；
- ILI 应比 ETTh1 更偏向“深度需求”，ETTh1 应比 ILI 更偏向“宽度需求”；
- ETTh2 可能同时需要较深和较宽；
- Electricity、Weather 不应预设一定获得显著动态容量收益，因为当前画像显示它们的窗口级时间变化较弱；Traffic 的 U 时间变化在加密窗口后较明显，但 M 的窗口级变化仍弱。

这些只是预期，不是必须得到的结果。九个数据集的样本频率、维数、画像窗口和通道采样不同，`U/M` 没有通用的 `0.5` 阈值。正式主检验仍应在每个数据集内部使用连续分数或 low/mid/high 分位桶，再跨模型和数据集汇总效应。

### 2.3 “与前文对应”目前不是完全同一画像协议

这里必须区分两种画像：

1. 前文数据画像：`profile_dataset.py` 优先在完整原始序列上按 `catalog.json` 的 `profile_window/profile_stride` 计算；
2. 容量画像：`build_local_losses.py` 从每个测试 run 保存的输入中重新计算 `U/M`，窗口长度等于模型实际输入长度。

两者使用同一套 `U/M` 公式，但窗口尺度并不总相同：

| 数据集 | 前文 profile window | 预测输入长度 | 是否相同 |
| --- | ---: | ---: | --- |
| ETTh1/ETTh2/ETTm1/ETTm2 | 96 | 96 | 是 |
| ExchangeRate | 96 | 96 | 是 |
| Weather | 144 | 96 | 否 |
| Electricity | 168 | 96 | 否 |
| Traffic | 168 | 96 | 否 |
| ILI | 48 | 24 | 否 |

当前容量代码的做法有一个重要优点：每个 `U/M` 与同一个模型实际看到的测试输入严格对齐，适合回答“这个输入是否需要更多容量”。但它不能直接声称使用了前文完全相同的画像窗口。

正式报告建议采用两层验证：

- 主分析使用当前容量代码的 forecast-context `U/M`，因为它与预测误差一一对齐；
- 一致性分析比较 forecast-context 分数与前文 profile-scale 分数在相同时间位置上的排序，报告 Spearman 相关和数据集排序是否稳定；
- 若两种尺度结论不一致，应报告尺度敏感性，不能只保留有利结果。

Traffic 当前已经按 `profile_window=168/profile_stride=24` 重新生成 512 个窗口，catalog、`summary.json` 和 `Result.md` 已一致。旧的 104 窗口、步长 168 结果不能再与当前结果混入同一统计表；正式冻结后应记录配置版本或 hash，避免再次出现协议漂移。

## 3. 当前深度/宽度代码实际上怎样工作

### 3.1 入口和候选集

容量规划在 [`src/adawd_preexp/capacity.py`](src/adawd_preexp/capacity.py)，单次训练入口是 [`pre_experiments/run_capacity_sweep.py`](pre_experiments/run_capacity_sweep.py)，便捷入口是 [`pre_experiments/run_preexperiments.py`](pre_experiments/run_preexperiments.py)。

[`pre_experiments/config.json`](pre_experiments/config.json) 当前登记：

```text
depth candidates       D = {1, 2, 4, 8}
width group candidates W = {1, 2, 4, 8}
joint depth            D = {2, 4, 8}
joint width group      W = {2, 4, 8}
saturation tolerance       = 0.01
formal seeds               = {3407, 3408, 3409}
```

四种 axis 的含义是：

| axis | 变化项 | 固定项 | 每个 horizon/seed 的组合数 |
| --- | --- | --- | ---: |
| `raw` | 不变化 | RAW 深度和 RAW 宽度 | 1 |
| `depth` | `D={1,2,4,8}` | RAW 宽度 | 4 |
| `width` | `W={1,2,4,8}` | RAW 深度 | 4 |
| `joint` | `D={2,4,8}` 和 `W={2,4,8}` | 无 | 9 |

便捷命令当前默认只运行 `ETTh1 + PatchTST + 最短 horizon + seed 42`。底层 `plan_sweep()` 在没有显式传 seeds 时却使用 `3407/3408/3409`。因此便捷入口只是 pilot，不是 8 x 9 正式调度器，正式任务必须显式固定 seeds。

### 3.2 八个模型的深度和宽度映射

所有模型都把统一深度映射到 `num_layers`，把统一宽度映射到 `intermediate_size`。实际宽度为：

```text
actual_width = width_group * width_unit
```

| 模型 | 深度语义 | 宽度语义 | RAW D | width unit | RAW width | RAW W |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Crossformer | 多尺度 encoder/decoder 层级 | FFN 中间维 | 2 | 256 | 2048 | 8 |
| PatchTST | Transformer encoder 层 | FFN 中间维 | 1 | 128 | 1024 | 8 |
| TimesNet | TimesBlock 层 | Inception 中间通道 | 1 | 128 | 1024 | 8 |
| iTransformer | inverted-token encoder 层 | FFN 中间维 | 1 | 128 | 1024 | 8 |
| TimeMixer | past-decomposable mixing block | mixing FFN 中间维 | 1 | 128 | 1024 | 8 |
| WPMixer | 多分辨率 mixer block | mixer FFN 中间维 | 2 | 128 | 1024 | 8 |
| TimeFilter | filtered graph block | graph-block FFN 中间维 | 2 | 64 | 256 | 4 |
| MultiPatchFormer | temporal/channel block 对 | encoder FFN 中间维 | 1 | 64 | 512 | 8 |

因此相同的 `D=4` 或 `W=4` 只表示相同的容量档位，不表示相同参数量、FLOPs 或实际 block 数。比如：

```text
Crossformer W={1,2,4,8} -> width={256,512,1024,2048}
TimeFilter W={1,2,4,8}   -> width={64,128,256,512}
```

当前 joint 网格还有两个解释限制：

- 它不包含 `D=1` 或 `W=1`，不利于观察 Q1 的最小容量；
- RAW 深度为 1 的五个模型，其 RAW 组合根本不在 joint 网格内。

如果二维联合实验是论文核心，推荐正式网格改为 `D={1,2,4,8} x W={1,2,4,8}`，或者至少确保每个模型的 RAW 组合和单轴最小组合都在网格中。若最高档仍频繁被选为饱和容量，应把该样本标记为右删失，并在 pilot 后扩展到 16，而不能直接宣称 8 就是充分容量。

### 3.3 模型构造与其他默认结构参数

`build_model()` 为每个 run 注入 `input_len/output_len/num_features/num_layers/intermediate_size`。其他参数来自本地模型配置类：

| 模型 | 主要固定结构配置 |
| --- | --- |
| Crossformer | hidden 512，8 heads，patch 16，win size 2，router factor 10，dropout 0.05 |
| PatchTST | hidden 256，1 head，patch 16/stride 8，RevIN，attention/FC dropout 0.1 |
| TimesNet | hidden 256，3 kernels，FFT top-k 5，dropout 0.1，当前唯一显式启用 timestamp 的模型 |
| iTransformer | hidden 256，1 head，RevIN，dropout 0.1 |
| TimeMixer | hidden 256，3 层 2 倍平均下采样，channel independence，RevIN，moving average 25 |
| WPMixer | hidden 256，wavelet level 2，patch 4/stride 2，dropout 0.1 |
| TimeFilter | hidden 128，4 heads，keep ratio 0.5，dropout 0.1，patch length 按数据集注册 |
| MultiPatchFormer | hidden 256，8 heads，patch `(8,16,24,32)`，strides `(8,8,7,6)`，8 个预测段 |

TimesNet 的 timestamp sizes 由数据频率构造；TimeFilter 的 patch length 分别为 ETTh1 2、ETTh2 4、ETTm1 8、ETTm2 16、Weather 48、Electricity 32、ILI 4、ExchangeRate 4、Traffic 96。

### 3.4 一次容量 run 的训练和验证流程

`run_capacity_sweep.py::execute()` 当前执行：

1. 创建 run 目录并写入 `status=running` 的 `manifest.json`；
2. 构造未训练模型，先测静态和推理效率；
3. 读取处理后的 train/val/test 数组；
4. 仅用训练集拟合逐通道 ZScore scaler；
5. 用滑动窗口构造 `[batch,input_time,channel] -> [batch,output_time,channel]` 样本；
6. 使用 masked MAE 训练，默认 Adam、学习率 `2e-4`、weight decay `5e-4`；
7. 每个 epoch 验证，以验证 MAE 最小保存最佳 checkpoint，早停 patience 为 10；
8. 训练结束加载最佳 checkpoint，在 test split 上计算 MAE/MSE/RMSE；
9. 正式 RAW 流式评估时只捕获固定测试窗口并导出原始量纲 CSV/PNG，再删除临时数组和 checkpoint；
10. 将最终指标、效率和 `status=complete/failed` 写回 manifest。

容量便捷入口没有读取 `benchmark_config.json` 的模型相关 epoch/batch，未覆盖时使用 `config.json` 的 `100 epochs + batch 64`。这与正式 RAW baseline 的模型相关预算不同，是当前代码中必须明确的协议差异。

### 3.5 从预测数组构造局部容量标签

[`pre_experiments/build_local_losses.py`](pre_experiments/build_local_losses.py) 要求每个容量 run 保留：

```text
inputs.npy
prediction.npy
targets.npy
```

对每个测试样本和所选通道建立：

```text
unit_id  = test:<sample_index>:<channel>
loss_mae = mean_h |prediction - target|
loss_mse = mean_h (prediction - target)^2
```

随后在同一个输入窗口上重新计算 `U/M`，并把 `model/seed/axis/depth/width_group/width/horizon` 合并到一行。默认遍历所有测试滑窗；Electricity 和 Traffic 按 catalog 均匀抽取 64 个通道，其余数据集使用全部通道。

在 `metric_scale=normalized` 时，保存的 inputs/prediction/targets 位于逐通道 ZScore 空间。`U/M` 内部还会对每个通道做 robust scale，因此对平移和正比例缩放基本不敏感，但正式报告仍应注明前文画像读取原始序列、容量画像读取模型实际使用的标准化输入。

容量 run 必须使用 `artifact_policy=full`。正式 RAW 默认 `artifact_policy=metrics`，评估时不写完整测试集预测，只选择性捕获固定窗口，成功后仅保留紧凑预测切片和 PNG。两类 run 不能放在同一 `runs-root` 后直接执行 `build_local_losses.py`，因为脚本会尝试读取目录下每个 complete manifest 的完整预测数组，遇到 metrics-only RAW run 会报缺文件。

### 3.6 饱和深度、饱和宽度和统计诊断

[`src/adawd_preexp/saturation.py`](src/adawd_preexp/saturation.py) 对每个相同的 `dataset/model/seed/horizon/unit_id` 计算：

```text
best_loss = 所有候选容量中的最小局部 loss
eligible  = loss <= 1.01 * best_loss
saturation = eligible 中的最小容量
```

- 深度轴输出 `d_sat`；
- 宽度轴输出 `w_sat`，这里仍是 width group 而不是实际中间层维数；
- joint 先找近最优组合，再按 `depth * width_group` 最小选择，平局时先选更小 depth、再选更小 width group。

`depth * width_group` 只是网格内选择代理，不能当作真实计算量。真实成本必须使用参数、FLOPs、延迟和显存记录。

[`pre_experiments/analyze_saturation.py`](pre_experiments/analyze_saturation.py) 当前输出：

```text
depth_saturation.csv
width_saturation.csv
depth_bucket_summary.csv
width_bucket_summary.csv
joint_saturation.csv
joint_quadrant_summary.csv
diagnostics.json
```

其中 low/mid/high 是每个数据集内部的 1/3、2/3 分位桶；Q1-Q4 使用每个数据集自己的 U/M 中位数：

```text
Q1: U低 M低    -> 预期浅、窄
Q2: U高 M低    -> 预期深、窄
Q3: U低 M高    -> 预期浅、宽
Q4: U高 M高    -> 预期深、宽
```

当前 `diagnostics.json` 只做 pooled OLS：`d_sat ~ U + M` 和 `w_sat ~ U + M`。它没有 dataset/model/horizon 固定效应，没有处理同一时间窗跨 seed、相邻滑窗和多通道的相关性，也没有 cluster-robust 标准误。它适合流水线诊断，不足以作为论文最终显著性模型。

还要注意：当前 `d_sat/w_sat` 使用测试集的未来 targets 事后计算，是用于机制分析的 oracle 容量标签。它能回答“这个局部单元实际上需要多大容量”，但在线预测时无法提前知道。若后续要报告 AdaWD 动态选择带来的最终 test 精度/效率，必须在 train/validation 上学习 `U/M -> depth/width` 映射和所有阈值，冻结后只用测试历史输入的 U/M 选择容量，再在未参与选择的 test targets 上评价。不能在同一 test 上先估计 oracle 饱和容量，再把它当成可部署策略的测试结果。

## 4. 推荐的正式模型预实验方案

### 4.1 实验目标和主假设

预先写清以下假设，避免根据结果修改口径：

```text
H1: beta_U in d_sat ~ U + M > 0
H2: beta_M in w_sat ~ U + M > 0
H3a: U 对 d_sat 的解释强于 M 对 d_sat 的解释
H3b: M 对 w_sat 的解释强于 U 对 w_sat 的解释
H4: Q2 相对 Q3 更深但不更宽，Q3 相对 Q2 更宽但不更深，Q4 两者都高
```

辅助假设可以检验高 `U/M` 单元从增加对应容量轴获得的误差下降是否更大，但不要把“容量更大”和“预测一定更准”写成同一命题。

### 4.2 阶段 A：协议冻结和极端配置兼容性检查

正式训练前先冻结一个 capacity plan，至少包含：

- 8 个准确模型名和代码 commit；
- 9 个数据集的文件校验、split、input length、horizon；
- depth/width/joint 候选；
- seeds、epoch、batch、学习率、早停；
- metric scale 和 artifact policy；
- PyTorch/CUDA/GPU 型号；
- profile 公式版本、窗口尺度、通道采样；
- 结果目录和协议 hash。

兼容性检查不用于精度比较。至少检查每个模型/数据集在最小、RAW、最大容量上能完成 forward/backward，尤其关注 Crossformer 的大宽度、Traffic/Electricity 的高通道数和 `D/W=8` 的显存。

### 4.3 阶段 B：全部 8 x 9 的单轴主实验

建议主实验固定每个数据集最短 horizon：

```text
ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity/ExchangeRate/Traffic: horizon 96
ILI: horizon 24
seeds: 3407, 3408, 3409
depth: 1,2,4,8; width fixed RAW
width group: 1,2,4,8; depth fixed RAW
metric scale: normalized
artifact policy: full
```

选择最短 horizon 不是因为长 horizon 不重要，而是容量假设首先针对局部输入结构。用单一、预注册 horizon 可以把正式单轴规模控制在 1,728 runs。之后再在代表性短/长 horizon 上做稳健性验证。

容量主实验推荐继承 `benchmark_config.json` 的模型相关最大 epoch 和默认 batch，而不是使用便捷入口统一的 100/64；若高维数据或最大容量需要更小 batch，应对同一 model/dataset 的整条容量曲线统一采用较小值。这样容量候选之间保持公平，RAW baseline 与容量实验的训练口径也更容易比较。

每个模型/数据集内部，所有容量候选必须使用相同：

- 数据 split、input/horizon、seed；
- 最大 epoch、早停规则和优化器；
- 有效 batch size；
- normalization 和 loss；
- checkpoint 选择指标。

若最大容量 OOM，不应只给最大模型降低 batch。应为该模型/数据集整条容量曲线统一采用能容纳最大候选的 batch，或使用梯度累积保持相同有效 batch。当前代码没有容量计划级梯度累积和 OOM 自适应，这需要在正式执行器中补齐或在冻结配置里人工统一。

### 4.4 阶段 C：二维联合和 horizon 稳健性

推荐优先选择画像角色清楚的数据集：

- ILI：高 U、低 M；
- ETTh1：中 U、高 M；
- ETTh2：U/M 都高；
- Electricity：较低 U/M 的高维对照；Traffic：U 时间变化较明显、M 时间变化较弱的高维对照；
- ExchangeRate：U/M 相关最低的解耦样本。

代表模型至少覆盖不同结构族：attention/patch、frequency/convolution、mixer、graph/filter。若论文核心贡献是动态宽深机制而不是单个 backbone，最终最好扩展至全部 8 模型。

二维正式网格建议把最小容量档 1 纳入：

```text
D={1,2,4,8} x W={1,2,4,8}
```

长 horizon 稳健性不必对全部组合穷举。可以预注册每个数据集最短和最长 horizon，对单轴结论复验，或者在每个画像角色中选择一个数据集做完整 horizon 曲线。选择规则必须在看到容量结果前确定。

### 4.5 正式统计和绘图

当前 CSV 可支持下列主图：

1. `U bucket -> d_sat`：横轴 low/mid/high，纵轴平均或中位 `d_sat`，按 dataset 分面、model 着色；
2. `M bucket -> w_sat`：相同设计；
3. 连续 `U-d_sat` 和 `M-w_sat`：显示离散容量的条件概率或抖动散点，不用普通连续回归线掩盖等级结构；
4. Q1-Q4 容量分配图：分别报告 d_sat、w_sat，而不是只画二者乘积；
5. 容量-误差-效率 Pareto 图：横轴 latency/FLOPs/parameters，纵轴 MSE，颜色表示 U/M 桶；
6. 边界命中率：各 dataset/model 中 `d_sat=8`、`w_sat=8` 的比例，用于判断网格是否截断。

推荐先在相同 `unit_id` 上跨三个 seed 汇总或配对，再做时间块 bootstrap，避免把高度重叠的滑窗和三个 seed 当成完全独立样本。最终模型至少应包含 dataset/model/horizon 效应；由于 `d_sat/w_sat` 是有序离散变量，ordinal mixed model 或分层 bootstrap 比 pooled OLS 更合适。可以保留 OLS 作为容易解释的辅助结果，但不能只报告其普通 p 值。

### 4.6 容量实验命令模板（仅说明，不在本文执行）

检查一个计划：

```bash
pixi run python pre_experiments/run_capacity_sweep.py \
  --dataset ILI \
  --model TimesNet \
  --axis depth \
  --horizon 24 \
  --seeds 3407 3408 3409 \
  --dry-run
```

正式单项模板应显式写出协议：

```bash
pixi run python pre_experiments/run_capacity_sweep.py \
  --dataset ILI \
  --model TimesNet \
  --axis depth \
  --horizon 24 \
  --seeds 3407 3408 3409 \
  --epochs 10 \
  --batch-size 32 \
  --metric-scale normalized \
  --artifact-policy full \
  --output-root pre_experiments/results/capacity_main \
  --gpu 0 \
  --all
```

现有代码没有生成全部 8 x 9 容量计划的总调度器。正式运行前应增加类似 RAW benchmark 的 capacity plan CSV 和可恢复调度器，不建议依赖手工复制 72 组命令。

局部损失与饱和分析模板：

```bash
pixi run python pre_experiments/build_local_losses.py \
  --runs-root pre_experiments/results/capacity_main \
  --output pre_experiments/results/capacity_main/local_losses.csv

pixi run python pre_experiments/analyze_saturation.py \
  --losses pre_experiments/results/capacity_main/local_losses.csv \
  --metric loss_mse \
  --epsilon 0.01 \
  --output-dir pre_experiments/results/capacity_main/saturation
```

## 5. RAW baseline 正常对比实验

### 5.1 当前代码已经有完整的 864-run 计划

[`pre_experiments/run_forecasting_benchmarks.py`](pre_experiments/run_forecasting_benchmarks.py) 已构造：

```text
8 models x 9 datasets x 4 horizons x 3 seeds = 864 runs
```

计划顺序为 `model -> dataset -> horizon -> seed`。每项实际调用同一个 `run_capacity_sweep.py`，但指定 `axis=raw`，因此模型构造、训练、验证、测试和效率测量与容量实验共享底层实现。

9 个数据集协议为：

| 数据集 | input | horizons | split | channels |
| --- | ---: | --- | --- | ---: |
| ETTh1/ETTh2/ETTm1/ETTm2 | 96 | 96/192/336/720 | 6:2:2 | 7 |
| Weather | 96 | 96/192/336/720 | 7:1:2 | 21 |
| Electricity | 96 | 96/192/336/720 | 7:1:2 | 321 |
| ILI | 24 | 24/36/48/60 | 7:1:2 | 7 |
| ExchangeRate | 96 | 96/192/336/720 | 7:1:2 | 8 |
| Traffic | 96 | 96/192/336/720 | 7:1:2 | 862 |

### 5.2 当前 baseline 标准训练配置

| 模型 | RAW D | RAW width | 最大 epoch | 默认 batch | Electricity batch | Traffic batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossformer | 2 | 2048 | 20 | 32 | 16 | 8 |
| PatchTST | 1 | 1024 | 100 | 64 | 16 | 8 |
| TimesNet | 1 | 1024 | 10 | 32 | 16 | 8 |
| iTransformer | 1 | 1024 | 10 | 32 | 16 | 8 |
| TimeMixer | 1 | 1024 | 10 | 32 | 16 | 8 |
| WPMixer | 2 | 1024 | 10 | 64 | 16 | 8 |
| TimeFilter | 2 | 256 | 10 | 32 | 16 | 8 |
| MultiPatchFormer | 1 | 512 | 20 | 32 | 16 | 8 |

共同协议是：

```text
seeds: 3407/3408/3409
optimizer: Adam
learning rate: 2e-4
weight decay: 5e-4
loss: masked MAE
checkpoint selection: minimum validation MAE
early stopping patience: 10
normalization: per-channel ZScore fitted on train only
reported metrics: normalized MAE/MSE/RMSE
artifact policy: metrics
deterministic: true
```

模型使用不同 epoch/batch 是本仓库对其公开训练预算的折中，但优化器协议仍统一。这一口径应写入论文；不要简称为“完全官方配置”。若要做完整官方复现，需要另外建立 dataset/horizon-specific 配置并与当前统一 BasicTS 对比区分。

### 5.3 能否在容量实验时顺带跑 RAW baseline

可以在调度层面一起安排，因为两者共享执行器和效率测量；但科学和存储协议应分开：

| 项目 | 容量预实验 | RAW baseline |
| --- | --- | --- |
| 目的 | 估计局部 d_sat/w_sat 与 U/M 的关系 | 比较正常预测精度和效率 |
| axis | depth/width/joint | raw |
| horizon | 主实验先用最短 horizon | 全部 4 个 horizon |
| artifacts | `full`，必须保存预测数组 | `metrics`，保留紧凑切片/图，删除 checkpoint/完整数组 |
| epoch/batch | 同一容量曲线严格固定 | 按 benchmark_config 的模型配置 |
| 输出目录 | `results/capacity_main` | `results/forecasting_raw` |

只有在 model、dataset、horizon、seed、D/W、epoch、batch、优化器、metric scale、数据版本和确定性配置全部一致时，一个 RAW 数值组合才有资格被复用。当前代码没有跨 axis 复用和协议 hash，因此最稳妥的现状是分开训练和记录。

### 5.4 当前 RAW 调度和汇总步骤

只生成正式计划：

```bash
pixi run forecast-all-plan
```

72-run smoke 只取最短 horizon、第一个 seed 和 1 epoch，用于兼容性检查，不用于精度比较：

```bash
pixi run forecast-smoke-plan
pixi run forecast-smoke
```

正式入口：

```bash
pixi run forecast-all
```

可以用 `--start-index/--stop-index` 按 `plan.csv` 分片。恢复时要求 manifest 状态、有限的 MAE/MSE/RMSE、数据指纹、完整协议签名以及预测切片/PNG 全部匹配；缺少任何一项都不会跳过。summarizer 同样按 `run_id + protocol_signature + data_fingerprint` 与当前 plan 做严格内连接，不会把旧配置结果混入新汇总。

[`pre_experiments/summarize_forecasting_benchmarks.py`](pre_experiments/summarize_forecasting_benchmarks.py) 当前生成：

```text
per_seed.csv: model/dataset/horizon/seed 的 MAE/MSE/RMSE
summary.csv:  三种 seed 的 mean 和 sample std
coverage.csv: 864 项 complete/pending 覆盖率
visualization_index.csv: 36 组 dataset/horizon 对比图的模型与 seed 覆盖率
visualizations/*.png: 原始量纲的真实值与八模型预测对比
Result.md:    summary.csv 的 Markdown 表
```

当前正式 `forecasting_raw` 汇总仍为空表/全 pending；已有 smoke 只能证明部分链路可运行，不能作为正式比较结果。

## 6. 效率记录：已经做到什么，还缺什么

### 6.1 每个 run 已自动记录的字段

[`src/adawd_preexp/efficiency.py`](src/adawd_preexp/efficiency.py) 会在训练前对同一架构做 batch=1 forward benchmark，并将以下字段写入 manifest：

```text
total_parameters
trainable_parameters
flops_per_batch
flops_error
benchmark_batch_size
latency_median_ms
latency_p90_ms
throughput_samples_per_second
benchmark_peak_cuda_bytes
device
warmup_iterations
timed_iterations
training_wall_seconds
training_peak_cuda_bytes
```

当前默认 warmup 10 次、计时 50 次、benchmark batch size 1。CUDA 测量前后会同步；不支持 profiler FLOPs 的算子允许 `flops_per_batch=null`。

因此“正常 baseline 预测时顺带记录效率”底层已经实现，不需要再写另一套 profiler。但字段解释要准确：

- latency 是随机初始化模型的纯 forward 架构延迟，不含数据加载和指标计算；
- throughput 是 batch=1 延迟换算值，不是训练吞吐；
- `training_wall_seconds` 包围整个 launcher，实际包含训练期间验证和训练后的最佳 checkpoint 测试，更适合称为端到端 train/eval wall time；
- training peak memory 也可能包含验证/最终测试；
- profiler FLOPs 可能因算子支持不足而低估或为空。

### 6.2 当前正式汇总遗漏了效率

RAW manifest 中有效率，但当前 `summarize_forecasting_benchmarks.py` 只读取 `metrics.overall`，没有把 efficiency 展开到 `per_seed.csv` 或 `summary.csv`。所以现状是“单 run 已记录，论文表尚未汇总”。

正式实验前建议增加：

```text
efficiency_per_seed.csv
efficiency_summary.csv
accuracy_efficiency.csv
```

汇总规则建议：

- 参数量：同一 model/dataset/horizon 配置应一致，直接报告并做一致性检查；
- FLOPs：报告支持率和非空值，不能把 null 当 0；
- latency/throughput：同一硬件上取各 seed run 的中位数及 IQR；
- wall time：报告 mean +/- sample std，同时注明是否早停；
- peak memory：报告中位数和最大值；
- 所有效率表加入 GPU 型号、CUDA、PyTorch、精度模式和并发状态。

当前 manifest 只保存了类似 `cuda:0` 的 device 字符串，没有完整硬件/软件指纹。不同机器结果不能直接汇总，正式调度器应补充环境元数据。

## 7. 预期结果和判定标准

### 7.1 支持深度假设的结果

至少应同时看到：

- 每个数据集内部 `U: low -> mid -> high` 时 `d_sat` 分布总体右移；
- 控制 `M` 后，`U` 对 `d_sat` 的系数或等级效应为正；
- 高 U 单元从增加深度得到的局部误差下降大于低 U 单元；
- 该方向在多数模型/数据集上同向，而不是只由 ILI 或单个模型驱动；
- `M` 对 d_sat 的效应不应系统性强于 U。

### 7.2 支持宽度假设的结果

对应地应看到：

- `M: low -> mid -> high` 时 `w_sat` 分布总体右移；
- 控制 `U` 后，`M` 对 `w_sat` 为正；
- 高 M 单元从增加宽度得到更明显的误差下降；
- 该方向跨模型/数据集具有一致性；
- `U` 对 w_sat 不应系统性强于 M。

### 7.3 支持宽深解耦的结果

- Q2（高 U、低 M）相对 Q3 应表现为更深但不更宽；
- Q3（低 U、高 M）相对 Q2 应表现为更宽但不更深；
- Q4 同时高，Q1 同时低；
- `d_sat ~ U + M` 中 U 主效应更清楚，`w_sat ~ U + M` 中 M 主效应更清楚；
- ETTm1/ETTm2 即使 U/M 相关较高，偏效应仍能区分，才说明不是同一个难度分数。

### 7.4 RAW baseline 和效率的预期产物

正式完成后应有 8 x 9 x 4 = 288 个 model/dataset/horizon 汇总组，每组 3 个 seed，报告 normalized MAE/MSE/RMSE 的 mean +/- sample std。效率表应能够回答：

- 哪些模型精度最好；
- 哪些模型参数最少、延迟最低、显存最低；
- 精度领先是否以明显效率代价换取；
- 同一模型的成本如何随通道数和 horizon 改变；
- 动态选择深宽后，是否能靠近 RAW 精度同时减少平均成本。

不应预先承诺所有模型的误差都会随深度或宽度单调下降，也不应预先承诺 horizon 越长所有指标必然严格变差。真实训练可能非单调，正因如此代码采用“1% 近最优的最小容量”而不是绝对最小误差点。

### 7.5 哪些结果意味着假设没有被支持

- high U 与 low U 的 d_sat 没有稳定差异；
- high M 与 low M 的 w_sat 没有稳定差异；
- U 同样或更强地预测宽度，M 同样或更强地预测深度；
- 效应只存在于单个数据集、单个 backbone 或单个 seed；
- 大量单元命中最大 D/W，说明网格截断而不是得到明确饱和；
- 容量增加带来的改善小于跨 seed 波动；
- 更大容量只是增加训练时间/显存，没有稳定误差收益。

这些都应作为有效实验结论报告，而不是通过事后更换窗口、候选或数据集筛选掉。

## 8. 当前实现与目标实验之间的差距

| 项目 | 当前状态 | 正式实验需要 |
| --- | --- | --- |
| 数据画像 | 9 数据集已有 U/M 文件和描述性结果 | 冻结 Traffic stride；处理画像尺度一致性 |
| 8 x 9 容量计划 | 单项 CLI 可运行，便捷入口只跑一个 model/dataset | 新增总 plan、coverage、resume 和协议 hash |
| 深度轴 | `{1,2,4,8}`，固定 RAW width | 保留；统计最大档命中率，必要时扩到 16 |
| 宽度轴 | `{1,2,4,8}`，固定 RAW depth | 保留；报告 actual width 和真实效率 |
| joint | `{2,4,8} x {2,4,8}` | 建议包含 1 和每个模型 RAW 组合 |
| U/M 对齐 | 测试输入上重新画像，公式一致 | 增加与前文 profile-scale 的一致性分析 |
| 容量训练预算 | 默认 100 epoch/batch 64 或手工覆盖 | 从冻结 capacity 配置按 model/dataset 统一生成 |
| RAW baseline | 已有 864-run 可恢复调度器 | 正式执行前完成数据和环境检查 |
| 局部预测数组 | capacity full 可保留 | 与 metrics-only RAW 分目录 |
| oracle 容量标签 | 当前从 test targets 事后计算 | 只用于机制分析；动态策略在 train/val 学习后到独立 test 评价 |
| 饱和统计 | 分桶、四象限、pooled OLS | 增加分层/固定效应、时间块 bootstrap |
| 效率 | 每个 manifest 已记录 | 增加 per-seed/summary/Pareto 汇总和硬件指纹 |
| 正式图 | 容量和 baseline 当前不自动画 | 从冻结 CSV 生成论文图并记录统计层级 |

## 9. 正式启动前检查清单

### 数据和画像

- 确认 Electricity 使用的是 321 通道 LTSF Electricity，还是论文草稿所述的另一数据集，名称不能混用；
- 明确 Weather 中 `-9999` 哨兵值的清洗或掩码协议，当前预处理只识别 NaN；
- 冻结 Traffic 的 window 168/stride 24 协议，并排除旧 stride 168 结果；
- 对 Electricity/Traffic 做 64 通道与全通道画像敏感性检查；
- 固定所有原始文件校验值、split 和 processed meta。

### 模型和训练

- 固定 8 个模型的代码 commit、RAW D/W、width unit 和其他结构默认值；
- 确认每个模型/数据集最大 D/W 可训练；
- 为整条容量曲线固定相同有效 batch，不按候选临时改变；
- 显式使用 seeds 3407/3408/3409，不使用便捷入口默认 seed 42；
- 记录 early stopping 的实际 epoch，而不只记录最大 epoch；
- capacity 使用 `artifact_policy=full`，RAW 使用独立目录的 `metrics`；
- 同一协议改变后使用新输出目录或协议 hash，避免同 run ID 覆写旧结果。
- 把 oracle 容量相关性与可部署策略分开；任何 U/M 阈值或路由器只在 train/validation 上确定。

### 统计和报告

- 预注册主 metric（建议 normalized MSE，MAE 作为稳健性结果）；
- 预注册 saturation tolerance 0.01，并补充 0/0.02/0.05 敏感性分析；
- 报告最大容量边界命中率；
- 对相邻窗口、通道和 seed 使用配对/分层统计；
- 分开报告局部关系、数据集平均关系和跨模型汇总；
- 同时报告实际参数、FLOPs 可用率、延迟、显存和端到端时间；
- smoke 结果只用于链路验证，不能进入精度排名。

## 10. 最终建议

本项目下一步不应按“高 U 数据集只跑深模型、高 M 数据集只跑宽模型”的方式分配实验，而应在全部 8 个模型和 9 个数据集上给出相同的一维容量候选，再从局部误差曲线估计 `d_sat/w_sat`。这样才有直接证据验证前文的 `U -> depth` 和 `M -> width`。

正常 baseline 对比可以与容量任务在同一阶段调度，当前也已经有 864-run RAW 计划和单 run 效率记录。但它应保持独立的 axis、训练预算、artifact policy、结果目录和汇总表。正式开跑前最优先补齐的是：8 x 9 capacity 总调度器、画像尺度一致性分析、joint 网格、效率汇总、协议 hash 和分层统计。完成这些后，模型预实验回答“为什么某些局部片段需要更深或更宽”，RAW baseline 回答“八个模型在标准配置下谁更准、代价多少”，两组实验才能共同支撑动态宽深方法的动机与比较结论。
