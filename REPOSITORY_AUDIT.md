# AdaDW 代码库静态审计

审计日期：2026-08-27。按照本轮要求，本次只做代码与配置的静态审计，没有启动训练、
测试、dry-run、smoke test 或结果重算。

## 1. 结论

当前正式 RAW forecasting 调度器在逻辑上覆盖：

```text
8 Backbones x 9 datasets x 4 horizons x 3 seeds = 864 runs
```

计划的四个 horizon 是每个数据集注册的四种预测长度，不是四种输入窗口。除 ILI 使用
`24/36/48/60` 外，其余八个数据集使用 `96/192/336/720`。所有数据集均使用一个固定的
历史输入长度：ILI 为 24，其余为 96。

代码中已有八个模型的构造入口、九个数据集的目录协议、三种随机种子、四种预测长度、
训练/验证/测试链路、结果 manifest、汇总器和断点续跑逻辑。八个模型现在分别拥有显式且
互不相同的 `benchmark_config`，正式 RAW benchmark 不再误用深宽容量预实验的参考配置。

这只能证明静态闭环完整，不能替代实际兼容性测试。仓库中的正式结果仍为 `0/864`；历史
smoke 只证明八个模型曾在 ILI 上完成过一次最小训练链路，并未覆盖九个数据集。

## 2. 实验矩阵

| 数据集 | 通道 | 输入长度 | 四个 horizon | 时间切分 |
| --- | ---: | ---: | --- | --- |
| ETTh1 | 7 | 96 | 96/192/336/720 | 60%/20%/20% |
| ETTh2 | 7 | 96 | 96/192/336/720 | 60%/20%/20% |
| ETTm1 | 7 | 96 | 96/192/336/720 | 60%/20%/20% |
| ETTm2 | 7 | 96 | 96/192/336/720 | 60%/20%/20% |
| Weather | 21 | 96 | 96/192/336/720 | 70%/10%/20% |
| Electricity | 321 | 96 | 96/192/336/720 | 70%/10%/20% |
| ILI | 7 | 24 | 24/36/48/60 | 70%/10%/20% |
| ExchangeRate | 8 | 96 | 96/192/336/720 | 70%/10%/20% |
| Traffic | 862 | 96 | 96/192/336/720 | 70%/10%/20% |

调度器同时校验：恰好 8 个 available Backbone、9 个数据集、每个数据集 4 个 horizon、
恰好 3 个唯一 seed，以及 864 个唯一 `run_id` 和协议签名。

## 3. Baseline 配置

| Backbone | 年份 | 主要 RAW 架构配置 | Epoch / Batch |
| --- | ---: | --- | ---: |
| Crossformer | 2023 | layers=2, hidden=512, d_ff=2048, heads=8, patch=16 | 80 / 32 |
| PatchTST | 2023 | layers=1, d_model=256, d_ff=1024, head=1, patch/stride=16/8 | 100 / 64 |
| TimesNet | 2023 | layers=1, hidden=256, conv=1024, kernels=3, top-k=5, timestamp | 80 / 32 |
| iTransformer | 2024 | layers=1, hidden=256, d_ff=1024, head=1, RevIN | 80 / 32 |
| TimeMixer | 2024 | layers=1, hidden=256, d_ff=1024, 3-level average downsampling | 80 / 32 |
| WPMixer | 2025 | layers=2, hidden=256, d_ff=1024, wavelet-level=2 | 80 / 64 |
| TimeFilter | 2025 | layers=2, hidden=128, d_ff=256, heads=4, keep=0.5 | 80 / 32 |
| MultiPatchFormer | 2025 | layers=1, hidden=256, d_ff=512, heads=8, four patch scales | 80 / 32 |

Electricity 的有效 batch 上限为 16，Traffic 为 8。模型结构不同，但正式对比共享 MSE
训练目标、MSE 验证选模、Adam `lr=2e-4`、`weight_decay=5e-4` 和 patience=10。这是本项目
的统一公平协议，不应表述为逐模型、逐数据集完整复刻所有官方训练脚本。

PatchTST、TimeMixer、MultiPatchFormer 的深宽预实验另有 `raw_depth/raw_width` 和
`fixed_config`。它们只服务于容量曲线，不能当作正式 RAW benchmark 架构配置。

## 4. 数据泄露审计

当前主链没有发现直接的测试标签泄露：

1. 原始 CSV 必须时间单调、无重复时间戳、信号为有限数值，并按目录配置做连续时间切分。
2. validation/test 文件只向前附加 `input_length` 个历史点。第一个预测目标分别从验证边界和
   测试边界开始，历史上下文不会让目标跨回前一 split，也不会把未来目标放进输入。
3. ZScore scaler 仅在 `train_data.npy` 上拟合；validation/test 只应用训练统计量。
4. checkpoint 只依据 validation MSE 保存和 early stop。
5. test loader 不在训练 epoch 中创建或评估；训练结束后才加载 validation 最优 checkpoint，
   在 test split 上进行最终评估。
6. runner 在非训练阶段不会把 target 传给声明了 target 参数的模型。
7. processed metadata 保存原始数据 SHA256、切分边界和数据指纹；每个计划项保存完整协议
   签名。架构、训练参数、数据目录、关键训练代码或 seed 变化后，旧 manifest 不会被续跑器
   静默复用，汇总器也不会将其混入新计划。
8. 预测图固定使用四个 horizon 公共测试区间的 50% 位置和等距通道 ID，同一数据集保持相同
   预测起点，不根据测试误差或目标形态挑选“好看”窗口。

需要区别两种测试集使用方式：正式 RAW benchmark 只做最终评估；现有 ETTh1 模型容量预实验
使用测试目标构造逐窗口 oracle。后者是事后上界分析，不是可部署选择器，不能用来训练控制器
后再在同一测试集报告泛化性能。

## 5. 仍需确认的风险

- 正式矩阵尚未运行，当前不能声称 864 个组合均无 OOM、无算子兼容问题。
- ETT 当前采用本项目的 60/20/20 连续比例切分；部分官方实现采用固定月份边界。与论文表格
  横向比较前必须统一协议，不能混用结果。
- Electricity 当前指向 321 通道 LTSF Electricity。论文草稿描述的 household power/
  sub-metering 是另一数据集，必须在正式报告前确认身份。
- WPMixer 是基于有许可证官方实现的 BasicTS 适配；TimeFilter 和 MultiPatchFormer 是根据
  论文结构编写的独立适配器，不等同于复制官方仓库实现。
- Traffic/Electricity 的高通道模型可能具有较高显存峰值。已有 batch cap 只是静态保护，
  仍需未来实际 smoke 后才能定稿。
- 仓库里的旧 `forecasting_raw/plan.csv` 是历史的未执行计划。正式启动入口会重建带协议签名
  的计划；不要把旧 CSV 当作新协议已经运行的证据。
- 当前本地 processed metadata 是旧格式。正式启动前需重新执行 `prepare-datasets` 生成数据
  指纹；这是有意的协议迁移保护，不应绕过后直接复用旧 NPY。

## 6. 代码库内容

| 路径 | 内容 |
| --- | --- |
| `Baselines/basicts/` | BasicTS 训练、验证、测试、指标、scaler 与 8 个模型适配器 |
| `Baselines/registry.json` | 模型入口、容量参数和独立 RAW benchmark 架构配置 |
| `datasets/catalog.json` | 9 个数据集的形状、频率、split、输入长度和 horizons |
| `datasets/raw/` | 原始 CSV，使用 Git LFS；不把 processed NPY 提交到 Git |
| `src/adawd_preexp/` | 数据准备、U/M profiler、容量计划、效率与饱和分析核心代码 |
| `pre_experiments/` | 数据画像、容量实验、RAW benchmark 与汇总命令入口 |
| `tests/` | 计划、模型形状、容量映射、结果保存和分析逻辑的测试代码 |
| `README.md` | 使用入口；`code.md`、`expand.md`、`new.md` 是设计和实验说明 |

## 7. 已有结果与图片

- `pre_experiments/results/Result.md`：九数据集 U/M 数据画像报告。
- `pre_experiments/results/profiles/<dataset>/`：每个数据集的 U heatmap、M heatmap、U/M
  temporal trajectory，共 27 张 PDF，以及 `windows.csv`、`summary.json`。
- `pre_experiments/results/model_trajectories/`：ETTh1、3 个模型、30/30 run 的容量预实验
  报告、相关统计、逐窗口表以及 PDF/PNG。结论支持“最优容量随片段变化”，但不支持当前
  U->depth、M->width 的单调映射。
- `image.png`：现有消融图参考；`image copy.png`：现有 Traffic 轨迹截图。两者按原文件保留，
  但来源和论文用途应在正式发布材料中补充说明。
- `pre_experiments/results/forecasting_raw/summary/Result.md`：正式 RAW benchmark 当前为
  `0/864`，没有精度结论。

正式实验完成后，每个 run 还会保留一个原始量纲 `forecast_slice.csv` 和单模型 PNG；汇总器
将生成 36 张 `dataset x horizon` 的八模型对比图及 `visualization_index.csv`。完整预测数组
不会落盘；test loader 流式评估时只选择性捕获预注册的一个窗口，生成紧凑产物后删除临时
样本数组。

Git 只发布可复核的代码、配置、报告、图和统计表。checkpoint、完整预测数组、TensorBoard、
训练日志、缓存和 processed NPY 继续忽略，避免仓库膨胀及把临时运行状态误当正式结果。
