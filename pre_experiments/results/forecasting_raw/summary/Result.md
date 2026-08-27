# Nine-Dataset RAW Forecasting Benchmark

Coverage: 0/864 runs complete.
Results are mean +/- sample standard deviation across seeds 3407, 3408 and 3409.
MAE, MSE and RMSE are computed in the per-channel ZScore-normalized space.
Successful runs retain metrics, efficiency metadata and compact forecast visualizations; evaluation captures only the registered sample, and checkpoints plus temporary sample arrays are deleted.

| Model | Dataset | Horizon | Epochs | Batch | Seeds | MSE | MAE | RMSE | Complete |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Forecast Visualizations

No forecast visualization is available yet. Completed runs will use the fixed
50% position in the test interval shared by all four horizons and evenly spaced
channel IDs, without target/error-based selection.
