#!/usr/bin/env python3
"""Plot separate temporal U/M profile figures for one dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


SCORE_STYLES = {
    "U": {"color": "#167D8D", "cmap": "viridis"},
    "M": {"color": "#C94F2D", "cmap": "magma"},
}


def _even_index_ticks(length: int, maximum: int) -> np.ndarray:
    """Return unique integer tick positions spanning all indexed values."""
    if length <= 1:
        return np.array([0], dtype=int)
    return np.unique(
        np.rint(np.linspace(0, length - 1, min(maximum, length))).astype(int)
    )


def _relative_labels(positions: np.ndarray, length: int) -> list[str]:
    denominator = max(length - 1, 1)
    return [f"{position / denominator:.0%}" for position in positions]


def _robust_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    lower, upper = np.quantile(finite, [0.02, 0.98])
    if upper - lower < 0.02:
        midpoint = (lower + upper) / 2.0
        lower, upper = midpoint - 0.01, midpoint + 0.01
    return max(0.0, float(lower)), min(1.0, float(upper))


def _trajectory_limits(lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    low = float(np.nanmin(lower))
    high = float(np.nanmax(upper))
    span = max(high - low, 0.05)
    padding = 0.08 * span
    return max(0.0, low - padding), min(1.0, high + padding)


def _format_channel(channel: object) -> str:
    if isinstance(channel, (float, np.floating)) and float(channel).is_integer():
        return str(int(channel))
    return str(channel)


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_heatmap(
    frame: pd.DataFrame,
    dataset: str,
    score: str,
    window_count: int,
    output: Path,
) -> None:
    table = frame.pivot_table(
        index="channel", columns="window_index", values=score, aggfunc="mean"
    ).sort_index()
    values = table.to_numpy(dtype=float)
    vmin, vmax = _robust_color_limits(values)
    color_map = plt.get_cmap(SCORE_STYLES[score]["cmap"]).copy()
    color_map.set_bad("#D9D9D9")

    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    image = axis.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
    )

    x_ticks = _even_index_ticks(window_count, maximum=6)
    y_ticks = _even_index_ticks(len(table.index), maximum=9)
    axis.set_xticks(x_ticks, labels=_relative_labels(x_ticks, window_count))
    axis.set_yticks(
        y_ticks,
        labels=[_format_channel(table.index[position]) for position in y_ticks],
    )
    axis.set_xlim(-0.5, window_count - 0.5)
    axis.set_ylim(len(table.index) - 0.5, -0.5)
    axis.set(
        xlabel="Relative time-window order",
        ylabel="Channel",
        title=f"{dataset}: {score} across channels and time windows",
    )
    axis.tick_params(direction="out", length=4, width=0.8)

    color_bar = figure.colorbar(image, ax=axis, pad=0.025, extend="both")
    color_bar.set_label(f"{score} score")
    color_bar.locator = ticker.LinearLocator(5)
    color_bar.formatter = ticker.FormatStrFormatter("%.2f")
    color_bar.update_ticks()
    _save_figure(figure, output)


def _plot_temporal_trajectories(
    window_summary: pd.DataFrame,
    dataset: str,
    output: Path,
) -> None:
    window_count = len(window_summary)
    relative_time = np.linspace(0.0, 1.0, max(window_count, 1))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.4),
        sharex=True,
        constrained_layout=True,
    )

    for axis, score in zip(axes, ("U", "M")):
        style = SCORE_STYLES[score]
        mean = window_summary[f"{score}_mean"].to_numpy(dtype=float)
        p10 = window_summary[f"{score}_p10"].to_numpy(dtype=float)
        p90 = window_summary[f"{score}_p90"].to_numpy(dtype=float)
        axis.fill_between(
            relative_time,
            p10,
            p90,
            color=style["color"],
            alpha=0.18,
            linewidth=0,
            label="P10-P90 across channels",
        )
        axis.plot(
            relative_time,
            mean,
            color=style["color"],
            linewidth=1.6,
            label="Window mean",
        )
        axis.set_ylim(*_trajectory_limits(p10, p90))
        axis.set_ylabel(f"{score} score")
        axis.set_title(f"{score} temporal trajectory", loc="left", fontsize=11)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=4))
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.45)
        axis.tick_params(direction="out", length=4, width=0.8)
        axis.legend(loc="best", fontsize=8, frameon=False)

    x_ticks = np.linspace(0.0, 1.0, 6)
    axes[-1].set_xticks(x_ticks, labels=[f"{value:.0%}" for value in x_ticks])
    axes[-1].set_xlim(0.0, 1.0)
    axes[-1].set_xlabel("Relative time-window order")
    figure.suptitle(f"{dataset}: temporal U/M trajectories", fontsize=13)
    _save_figure(figure, output)


def plot_profiles(profiles: Path, output_dir: Path | None = None) -> list[Path]:
    frame = pd.read_csv(profiles)
    required = {"U", "M", "channel", "segment", "window_start"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Profile table is missing: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Profile table is empty")

    dataset = str(frame["dataset"].iloc[0]) if "dataset" in frame else profiles.stem
    windows = frame[["segment", "window_start"]].drop_duplicates().reset_index(drop=True)
    windows["window_index"] = np.arange(len(windows), dtype=int)
    frame = frame.merge(
        windows,
        on=["segment", "window_start"],
        how="left",
        validate="many_to_one",
    )
    window_count = len(windows)
    window_summary = (
        frame.groupby("window_index", sort=True)
        .agg(
            U_mean=("U", "mean"),
            U_p10=("U", lambda values: values.quantile(0.1)),
            U_p90=("U", lambda values: values.quantile(0.9)),
            M_mean=("M", "mean"),
            M_p10=("M", lambda values: values.quantile(0.1)),
            M_p90=("M", lambda values: values.quantile(0.9)),
        )
        .reindex(np.arange(window_count))
    )

    destination = output_dir or profiles.parent
    outputs = [
        destination / "U_heatmap.pdf",
        destination / "M_heatmap.pdf",
        destination / "UM_temporal_trajectory.pdf",
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    ):
        _plot_heatmap(frame, dataset, "U", window_count, outputs[0])
        _plot_heatmap(frame, dataset, "M", window_count, outputs[1])
        _plot_temporal_trajectories(window_summary, dataset, outputs[2])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    for output in plot_profiles(args.profiles, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
