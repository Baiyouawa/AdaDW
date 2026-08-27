"""Deterministic forecast slices and cross-Backbone visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {
    "Crossformer": "#2F6B9A",
    "PatchTST": "#D1495B",
    "TimesNet": "#2A9D8F",
    "iTransformer": "#7A5195",
    "TimeMixer": "#E07A2D",
    "WPMixer": "#4F772D",
    "TimeFilter": "#8C564B",
    "MultiPatchFormer": "#5C677D",
}


def select_sample_index(sample_count: int, relative_position: float) -> int:
    """Select a deterministic test sample without consulting targets or errors."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= relative_position <= 1.0:
        raise ValueError("relative_position must be in [0, 1]")
    return min(sample_count - 1, int(round(relative_position * (sample_count - 1))))


def select_channel_ids(channel_count: int, max_channels: int) -> list[int]:
    """Select evenly spaced channel IDs independently of observed values."""

    if channel_count < 1 or max_channels < 1:
        raise ValueError("channel_count and max_channels must be positive")
    count = min(channel_count, max_channels)
    return np.unique(
        np.rint(np.linspace(0, channel_count - 1, count)).astype(int)
    ).tolist()


def _inverse_zscore(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * std + mean


def _load_original_scale_window(
    result_dir: Path,
    processed_dir: Path,
    sample_index: int,
    channels: Sequence[int],
    metric_scale: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        name: result_dir / f"{name}.npy"
        for name in ("inputs", "prediction", "targets")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing forecast arrays: {missing}")

    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    if arrays["prediction"].shape != arrays["targets"].shape:
        raise ValueError("Prediction and target arrays have different shapes")
    if arrays["inputs"].shape[0] != arrays["targets"].shape[0]:
        raise ValueError("Input and target arrays have different sample counts")

    selected = np.asarray(channels, dtype=int)
    inputs = np.take(arrays["inputs"][sample_index], selected, axis=-1).T
    prediction = np.take(arrays["prediction"][sample_index], selected, axis=-1).T
    targets = np.take(arrays["targets"][sample_index], selected, axis=-1).T

    if metric_scale == "normalized":
        train = np.load(processed_dir / "train_data.npy", mmap_mode="r")
        mean = np.mean(train, axis=0, dtype=np.float64)[selected, None]
        std = np.std(train, axis=0, dtype=np.float64)[selected, None]
        std[std == 0.0] = 1.0
        inputs = _inverse_zscore(inputs, mean, std)
        prediction = _inverse_zscore(prediction, mean, std)
        targets = _inverse_zscore(targets, mean, std)
    elif metric_scale != "original":
        raise ValueError(f"Unsupported metric scale: {metric_scale}")
    return inputs, prediction, targets


def _load_window_timestamps(
    processed_dir: Path,
    sample_index: int,
    input_length: int,
    output_length: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    path = processed_dir / "test_time_index.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; re-run dataset preparation with the current code."
        )
    time_index = np.load(path, mmap_mode="r")
    stop = sample_index + input_length + output_length
    if stop > len(time_index):
        raise ValueError("Selected forecast window exceeds the stored test time index")
    history = pd.to_datetime(np.asarray(time_index[sample_index : sample_index + input_length]))
    future = pd.to_datetime(np.asarray(time_index[sample_index + input_length : stop]))
    return pd.DatetimeIndex(history), pd.DatetimeIndex(future)


def _build_slice_frame(
    *,
    dataset: str,
    model: str,
    horizon: int,
    seed: int,
    sample_index: int,
    sample_position: float,
    channels: Sequence[int],
    channel_names: Sequence[str],
    history_time: pd.DatetimeIndex,
    future_time: pd.DatetimeIndex,
    inputs: np.ndarray,
    prediction: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for local_index, channel in enumerate(channels):
        common = {
            "dataset": dataset,
            "model": model,
            "horizon": horizon,
            "seed": seed,
            "sample_index": sample_index,
            "sample_position": sample_position,
            "channel": int(channel),
            "channel_name": channel_names[channel],
            "scale": "original",
        }
        for offset, (timestamp, observed) in enumerate(
            zip(history_time, inputs[local_index]), start=-len(history_time)
        ):
            rows.append(
                {
                    **common,
                    "phase": "history",
                    "step": offset,
                    "timestamp": timestamp.isoformat(),
                    "observed": float(observed),
                    "prediction": np.nan,
                }
            )
        for step, (timestamp, observed, predicted) in enumerate(
            zip(future_time, targets[local_index], prediction[local_index]), start=1
        ):
            rows.append(
                {
                    **common,
                    "phase": "forecast",
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "observed": float(observed),
                    "prediction": float(predicted),
                }
            )
    return pd.DataFrame(rows)


def _plot_run_slice(frame: pd.DataFrame, output: Path) -> None:
    channels = frame[["channel", "channel_name"]].drop_duplicates().itertuples(index=False)
    channel_items = list(channels)
    figure, axes = plt.subplots(
        len(channel_items),
        1,
        figsize=(12.0, max(3.0, 2.45 * len(channel_items))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, item in enumerate(channel_items):
        axis = axes[row, 0]
        subset = frame[frame["channel"] == item.channel].copy()
        subset["timestamp"] = pd.to_datetime(subset["timestamp"])
        history = subset[subset["phase"] == "history"]
        forecast = subset[subset["phase"] == "forecast"]
        axis.plot(history["timestamp"], history["observed"], color="#6B7280", linewidth=1.4)
        axis.plot(
            forecast["timestamp"], forecast["observed"], color="#111827",
            linewidth=1.8, label="Ground truth",
        )
        axis.plot(
            forecast["timestamp"], forecast["prediction"], color="#C94F2D",
            linewidth=1.5, label="Prediction",
        )
        axis.axvline(forecast["timestamp"].iloc[0], color="#9CA3AF", linestyle="--", linewidth=1.0)
        axis.set_ylabel(str(item.channel_name))
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if row == 0:
            axis.legend(loc="upper left", ncol=2, frameon=False)
    first = frame.iloc[0]
    figure.suptitle(
        f"{first['model']} | {first['dataset']} | horizon {int(first['horizon'])} | "
        f"seed {int(first['seed'])} | test window {int(first['sample_index'])}",
        fontsize=12,
    )
    axes[-1, 0].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_run_forecast_visualization(
    *,
    result_dir: Path,
    processed_dir: Path,
    run_dir: Path,
    dataset: str,
    model: str,
    horizon: int,
    seed: int,
    input_length: int,
    output_length: int,
    metric_scale: str,
    sample_position: float,
    sample_index: int | None,
    max_channels: int,
) -> dict[str, Any]:
    """Persist one target-independent forecast slice from evaluation outputs."""

    prediction = np.load(result_dir / "prediction.npy", mmap_mode="r")
    stored_indices_path = result_dir / "sample_indices.npy"
    if stored_indices_path.is_file():
        stored_indices = np.load(stored_indices_path)
        if sample_index is None:
            raise ValueError("sample_index is required for selectively stored results")
        matches = np.flatnonzero(stored_indices == sample_index)
        if len(matches) != 1:
            raise ValueError(
                f"Selected sample {sample_index} is absent from stored indices "
                f"{stored_indices.tolist()}"
            )
        result_index = int(matches[0])
    else:
        if sample_index is None:
            sample_index = select_sample_index(len(prediction), sample_position)
        result_index = sample_index
    channel_count = int(prediction.shape[-1])
    channels = select_channel_ids(channel_count, max_channels)
    inputs, predicted, targets = _load_original_scale_window(
        result_dir, processed_dir, result_index, channels, metric_scale
    )
    history_time, future_time = _load_window_timestamps(
        processed_dir, sample_index, input_length, output_length
    )
    metadata = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
    channel_names = metadata.get("signal_columns") or [f"channel_{i}" for i in range(channel_count)]
    if len(channel_names) != channel_count:
        raise ValueError("Processed metadata signal columns do not match prediction channels")

    frame = _build_slice_frame(
        dataset=dataset,
        model=model,
        horizon=horizon,
        seed=seed,
        sample_index=sample_index,
        sample_position=sample_position,
        channels=channels,
        channel_names=channel_names,
        history_time=history_time,
        future_time=future_time,
        inputs=inputs,
        prediction=predicted,
        targets=targets,
    )
    csv_path = run_dir / "forecast_slice.csv"
    plot_path = run_dir / "forecast_vs_target.png"
    frame.to_csv(csv_path, index=False)
    _plot_run_slice(frame, plot_path)
    return {
        "selection_rule": "fixed_common_horizon_test_position_and_evenly_spaced_channels",
        "sample_position": sample_position,
        "sample_index": sample_index,
        "channels": channels,
        "channel_names": [channel_names[channel] for channel in channels],
        "scale": "original",
        "slice_csv": csv_path.name,
        "plot_png": plot_path.name,
        "forecast_start": future_time[0].isoformat(),
        "forecast_end": future_time[-1].isoformat(),
    }


def _assert_consistent_observations(frames: Sequence[pd.DataFrame]) -> None:
    if not frames:
        return
    keys = ["channel", "phase", "step", "timestamp"]
    reference = frames[0][keys + ["observed"]].sort_values(keys).reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame[keys + ["observed"]].sort_values(keys).reset_index(drop=True)
        if not reference[keys].equals(candidate[keys]) or not np.allclose(
            reference["observed"], candidate["observed"], rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Forecast slices do not share identical observations and timestamps")


def _plot_baseline_comparison(
    frames: Sequence[pd.DataFrame],
    model_order: Sequence[str],
    output: Path,
) -> None:
    _assert_consistent_observations(frames)
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    channel_items = combined[["channel", "channel_name"]].drop_duplicates().sort_values("channel")
    figure, axes = plt.subplots(
        len(channel_items),
        1,
        figsize=(13.0, max(3.2, 2.65 * len(channel_items))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, item in enumerate(channel_items.itertuples(index=False)):
        axis = axes[row, 0]
        subset = combined[combined["channel"] == item.channel]
        reference = subset[
            (subset["model"] == subset["model"].iloc[0])
            & (subset["seed"] == subset["seed"].iloc[0])
        ]
        history = reference[reference["phase"] == "history"]
        truth = reference[reference["phase"] == "forecast"]
        axis.plot(history["timestamp"], history["observed"], color="#6B7280", linewidth=1.35)
        axis.plot(
            truth["timestamp"], truth["observed"], color="#111827",
            linewidth=2.0, label="Ground truth", zorder=10,
        )
        for index, model in enumerate(model_order):
            model_frame = subset[(subset["model"] == model) & (subset["phase"] == "forecast")]
            if model_frame.empty:
                continue
            grouped = model_frame.groupby(["step", "timestamp"], sort=True)["prediction"]
            mean = grouped.mean()
            std = grouped.std(ddof=1).fillna(0.0)
            times = pd.DatetimeIndex(mean.index.get_level_values("timestamp"))
            color = MODEL_COLORS.get(model, f"C{index}")
            axis.plot(times, mean.to_numpy(), color=color, linewidth=1.25, label=model)
            if bool((std > 0).any()):
                axis.fill_between(
                    times,
                    (mean - std).to_numpy(),
                    (mean + std).to_numpy(),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
        axis.axvline(truth["timestamp"].iloc[0], color="#9CA3AF", linestyle="--", linewidth=1.0)
        axis.set_ylabel(str(item.channel_name))
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=5, frameon=False,
    )
    first = combined.iloc[0]
    figure.suptitle(
        f"Baseline forecasts | {first['dataset']} | horizon {int(first['horizon'])} | "
        "mean across available seeds",
        fontsize=13,
        y=1.035,
    )
    axes[-1, 0].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_baseline_comparison_visualizations(
    *,
    runs: pd.DataFrame,
    runs_root: Path,
    output_dir: Path,
    model_order: Sequence[str],
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Build one seed-aggregated comparison plot per dataset and horizon."""

    columns = [
        "dataset", "horizon", "model_count", "run_count", "complete",
        "sample_position", "sample_index", "channels", "forecast_start",
        "forecast_end", "plot_png",
    ]
    visual_dir = output_dir / "visualizations"
    if visual_dir.is_dir():
        for stale in visual_dir.glob("*__baseline_forecasts.png"):
            stale.unlink()
    if runs.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for (dataset, horizon), group in runs.groupby(["dataset", "output_length"], sort=True):
        frames = []
        for row in group.itertuples(index=False):
            slice_name = getattr(row, "visualization_slice", None)
            if not isinstance(slice_name, str) or not slice_name:
                continue
            path = runs_root / row.run_id / Path(slice_name).name
            if path.is_file():
                frames.append(pd.read_csv(path))
        if not frames:
            continue
        plot_name = f"{dataset}__h{int(horizon)}__baseline_forecasts.png"
        plot_path = visual_dir / plot_name
        _plot_baseline_comparison(frames, model_order, plot_path)
        combined = pd.concat(frames, ignore_index=True)
        observed_pairs = set(zip(combined["model"], combined["seed"].astype(int)))
        expected_pairs = {
            (model, int(seed)) for model in model_order for seed in expected_seeds
        }
        records.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "model_count": int(combined["model"].nunique()),
                "run_count": len(observed_pairs),
                "complete": observed_pairs == expected_pairs,
                "sample_position": float(combined["sample_position"].iloc[0]),
                "sample_index": int(combined["sample_index"].iloc[0]),
                "channels": ",".join(str(value) for value in sorted(combined["channel"].unique())),
                "forecast_start": combined.loc[combined["phase"] == "forecast", "timestamp"].min(),
                "forecast_end": combined.loc[combined["phase"] == "forecast", "timestamp"].max(),
                "plot_png": f"visualizations/{plot_name}",
            }
        )
    return pd.DataFrame(records, columns=columns)
