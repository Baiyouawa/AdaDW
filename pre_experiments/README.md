# Pre-experiment protocol

## 1. Dataset-characteristic validation

For each fixed rolling window and profiled channel, the code records all six
paper descriptors instead of only the aggregates:

```text
U_i = mean(u_change, u_spectral, u_surprise)
M_i = mean(m_peak, m_band, m_channel)          # multivariate
M_i = mean(m_peak, m_band)                     # univariate
```

`profile_dataset.py` writes `windows.csv` and `summary.json`. The visual
diagnostics are exported as three publication-ready figures per dataset: a U
time-channel heatmap, an M time-channel heatmap, and a two-panel U/M temporal
trajectory figure. Heatmaps use dataset-specific 2nd-98th percentile color
limits so temporal contrast remains visible while the color bars still report
the original score values. The trajectory panels report the window mean and
P10-P90 across channels on aligned relative-time ticks. Traffic uses a
one-day profile stride (24 hourly samples) and the shared 512-window cap so its
trajectory has the same real-window density as the other long datasets. A
narrow window-level spread is not a positive result. Absolute U/M means are
not probabilities and have no universal 0.5 threshold.

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

Capacity runs with `artifact_policy=full` store a manifest, configured metrics,
predictions, targets, per-window MAE/MSE and efficiency records. Formal RAW runs
use `artifact_policy=metrics`: they keep normalized MAE/MSE/RMSE and efficiency
metadata plus a compact original-scale forecast slice and PNG, then remove
checkpoints and temporary selected-sample arrays. Formal evaluation selectively
captures only the registered visualization window rather than writing full-test
prediction arrays. FLOPs may be null when an operator is
unsupported; that is reported rather than replaced by a parameter-count proxy.

## 4. Full forecasting benchmark

`run_forecasting_benchmarks.py` builds the Cartesian product of all eight
registered Backbones, all nine forecasting datasets, each dataset's four
catalog horizons, and seeds 3407/3408/3409. Model-specific epoch and batch
settings are recorded in `benchmark_config.json`; distinct architecture settings
are explicit in `Baselines/registry.json`. The runner is resumable and
supports `--start-index`/`--stop-index` for scheduling subsets of `plan.csv`.

The benchmark uses normalized MAE/MSE/RMSE and `artifact_policy=metrics`.
Checkpoints exist only while fitting and evaluating the best epoch, then are
deleted after their metrics are copied into the run manifest. The summary tool
produces per-seed values, mean/sample-standard-deviation tables, coverage, and
a Markdown report.

Each run visualizes the fixed 50% position in the test interval shared by all
four horizons and at most four evenly spaced channel IDs. Thus all horizons of
one dataset have the same forecast start; neither targets nor errors influence
selection. The final summary
builds one plot per dataset/horizon (36 for the full matrix), overlaying ground
truth and all eight Backbones. Backbone curves are means across available seeds,
with one-sample-standard-deviation bands. Plots use original data units and real
timestamps, while accuracy metrics remain in normalized space.

Every planned run carries a data fingerprint and protocol signature. Resume and
summary operations require an exact signature match, so results from changed
data, architecture or training code are not silently reused. The formal matrix
is still 0/864 complete; static coverage is not evidence that every combination
has passed a runtime smoke test.

## Fairness notes

- Use identical splits, input/output lengths, optimizer budget and seeds for all
  capacity candidates of one Backbone/dataset pair.
- Report actual compute; depth 4 in Crossformer is not compute-equivalent to
  depth 4 in PatchTST.
- The three 2025 Backbones use architecture-specific BasicTS adapters. Their
  source repositories, reviewed commits and adaptation notes are recorded in
  `Baselines/README.md`.
