# Dataset inventory and preparation

The nine experiment entries are ETTh1, ETTh2, ETTm1, ETTm2, Weather,
Electricity, ILI, ExchangeRate and Traffic. `catalog.json` is the
single source of truth for names, dimensions, frequencies and split policies.

Download all nine datasets from the project root with:

```bash
pixi run download-datasets
```

The downloader resumes partial transfers, skips existing valid files, falls
back to a Hugging Face mirror and validates the CSV dimensions before replacing
the destination. Use `pixi run download-datasets --help` to select individual
datasets or override the repository endpoint. All project commands run in the
locked Pixi environment; use `pixi run preexp-dataset` to download, strictly
prepare, profile and plot every registered dataset in one task.

There is one unresolved identity conflict: DropoutTS and the standard LTSF
protocol use a 321-channel Electricity client-consumption series, while the
paper draft describes the distinct household power/sub-metering dataset. The
catalog currently follows the reusable 321-channel code and marks the identity
for confirmation. Do not rename one dataset as the other in a reported table.

DropoutTS contains preparation scripts for these forecasting entries, but it
contains no raw or processed arrays. The AdaWD preprocessor was
therefore rewritten to use explicit paths, retain the complete series and emit
metadata with every processed dataset.

Expected raw layout:

```text
datasets/raw/
  ETTh1/ETTh1.csv
  ETTh2/ETTh2.csv
  ETTm1/ETTm1.csv
  ETTm2/ETTm2.csv
  Weather/Weather.csv
  Electricity/Electricity.csv
  ILI/ILI.csv                 # Illness.csv is also accepted
  ExchangeRate/ExchangeRate.csv
  Traffic/Traffic.csv
```

CSV forecasting files must have one timestamp column (`date` by default) and
numeric signal columns. The generated `datasets/processed/<name>/` directory
contains `train_data.npy`, `val_data.npy`, `test_data.npy`, matching timestamp
feature arrays, `train/val/test_time_index.npy` with the real `datetime64[ns]`
axis, and `meta.json` with signal-column names. BasicTS consumes the data and
timestamp-feature arrays; forecast visualization uses the real time-index arrays.
