# Pre-experiment protocol

## 1. Dataset-characteristic validation

For each fixed rolling window and profiled channel, the code records all six
paper descriptors instead of only the aggregates:

```text
U_i = mean(u_change, u_spectral, u_surprise)
M_i = mean(m_peak, m_band, m_channel)          # multivariate
M_i = mean(m_peak, m_band)                     # univariate
```

`profile_dataset.py` writes `windows.csv` and `summary.json`. The required
diagnostics are the U/M distributions, P10/P50/P90, a U-by-M scatter plot,
within-dataset low/mid/high buckets, and the U/M Spearman correlation. A narrow
IQR or nearly perfect U/M correlation is a failed descriptor diagnostic, not a
positive result.

Electricity and Traffic are extremely wide. Their catalog entries use a
deterministic, evenly spaced 64-channel subset for the initial profiler pass.
The selected channel IDs remain in `windows.csv`; the full-channel sensitivity
run should be reported before publication.

## 2. Capacity validation

Each model uses its architecture's native block count as depth and its native
intermediate transformation dimension as width. Width labels `{1,2,4,8}` are
nested group counts; concrete dimensions are `group_count * width_unit` and are
stored in every run manifest. This avoids falsely treating an integer group
label as the same number of channels across architectures.

- depth axis: `D={1,2,4,8}`, fixed RAW width;
- width axis: `W={1,2,4,8}`, fixed RAW depth;
- 2-D axis: `D={2,4,8} x W={2,4,8}`;
- RAW: the untouched model defaults in `Baselines/registry.json`.

For each local unit, `analyze_saturation.py` selects the smallest candidate
whose loss is within `(1 + epsilon)` of that unit's best loss. For the 2-D
experiment it selects the lowest `depth * width_group` cost among near-optimal
combinations. The resulting `d_sat` and `w_sat` are merged with U/M profiles.

The confirmatory tests are:

- ordinal trend of mean `d_sat` across U low/mid/high buckets;
- ordinal trend of mean `w_sat` across M low/mid/high buckets;
- partial regressions `d_sat ~ U + M` and `w_sat ~ U + M`;
- Q1-Q4 allocation tables using within-dataset U/M medians;
- consistency across seeds, datasets and Backbones.

Do not claim H1-H3 from a dataset-level average. Saturation is defined per
local unit and requires all capacity candidates for that same unit.

## 3. RAW and efficiency records

Every executed run stores its manifest, five forecasting metrics, predictions,
targets, per-window MAE/MSE, total/trainable parameters, profiler FLOPs, training
wall time, inference latency (median and P90), throughput, and CUDA peak memory
where available. FLOPs may be null when an operator is unsupported; that is
reported rather than replaced by a parameter-count proxy.

The current environment has no raw data and no PyTorch installation, so only
planning and data-analysis code can run here. `--dry-run` validates experiment
coverage without launching training.

## Fairness notes

- Use identical splits, input/output lengths, optimizer budget and seeds for all
  capacity candidates of one Backbone/dataset pair.
- Report actual compute; depth 4 in Crossformer is not compute-equivalent to
  depth 4 in PatchTST.
- Three missing 2025 Backbones remain disabled until verified implementations
  and architecture-specific depth/width adapters are added.
