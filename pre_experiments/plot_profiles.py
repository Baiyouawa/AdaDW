#!/usr/bin/env python3
"""Plot temporal U/M heterogeneity for one profile table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=10000)
    args = parser.parse_args()
    frame = pd.read_csv(args.profiles)
    required = {"U", "M", "U_bucket", "M_bucket"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Profile table is missing: {sorted(required - set(frame.columns))}")
    dataset = frame["dataset"].iloc[0] if "dataset" in frame else args.profiles.stem

    # A profile row is one channel inside one rolling window. Preserve the
    # generated segment/window order and attach a stable temporal index.
    window_columns = ["segment", "window_start"]
    windows = frame[window_columns].drop_duplicates().reset_index(drop=True)
    windows["window_index"] = np.arange(len(windows), dtype=int)
    frame = frame.merge(windows, on=window_columns, how="left", validate="many_to_one")
    window_count = len(windows)
    relative_time = np.linspace(0.0, 1.0, max(window_count, 1))

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)

    heatmaps = []
    for axis, score, title, cmap in (
        (axes[0, 0], "U", "U by channel and time window", "viridis"),
        (axes[0, 1], "M", "M by channel and time window", "magma"),
    ):
        table = frame.pivot_table(
            index="channel", columns="window_index", values=score, aggfunc="mean"
        ).sort_index()
        image = axis.imshow(
            table.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        heatmaps.append(image)
        axis.set(
            xlabel="Relative time-window order",
            ylabel="Channel",
            title=title,
            xlim=(-0.5, max(window_count - 0.5, 0.5)),
        )
        tick_count = min(6, max(window_count, 1))
        tick_positions = np.linspace(0, max(window_count - 1, 0), tick_count)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([f"{value:.0%}" for value in np.linspace(0, 1, tick_count)])
        channels = table.index.to_numpy()
        channel_ticks = np.linspace(0, max(len(channels) - 1, 0), min(8, len(channels)))
        axis.set_yticks(channel_ticks)
        axis.set_yticklabels([str(channels[int(round(pos))]) for pos in channel_ticks])

    # Color the joint plane by temporal position instead of by a high-score
    # bucket. This makes the time dependence visible in the U-M relationship.
    sample = frame.sample(min(args.max_points, len(frame)), random_state=42)
    colors = sample["window_index"].to_numpy() / max(window_count - 1, 1)
    scatter = axes[0, 2].scatter(
        sample["U"], sample["M"], s=8, c=colors, cmap="plasma", vmin=0.0, vmax=1.0,
        alpha=0.45, linewidths=0,
    )
    axes[0, 2].axvline(frame["U"].median(), color="black", linewidth=0.8, linestyle="--")
    axes[0, 2].axhline(frame["M"].median(), color="black", linewidth=0.8, linestyle="--")
    axes[0, 2].set(
        xlabel="U: state-update demand",
        ylabel="M: pattern diversity",
        title="U-M plane colored by relative time",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )

    window_summary = frame.groupby("window_index", sort=True).agg(
        U_mean=("U", "mean"),
        U_p10=("U", lambda values: values.quantile(0.1)),
        U_p90=("U", lambda values: values.quantile(0.9)),
        M_mean=("M", "mean"),
        M_p10=("M", lambda values: values.quantile(0.1)),
        M_p90=("M", lambda values: values.quantile(0.9)),
    )
    for axis, score, color, title in (
        (axes[1, 0], "U", "#167D8D", "Temporal U trajectory"),
        (axes[1, 1], "M", "#D15C32", "Temporal M trajectory"),
    ):
        axis.fill_between(
            relative_time,
            window_summary[f"{score}_p10"],
            window_summary[f"{score}_p90"],
            color=color,
            alpha=0.16,
            label="P10-P90 across channels",
        )
        axis.plot(
            relative_time,
            window_summary[f"{score}_mean"],
            color=color,
            linewidth=1.5,
            label="window mean",
        )
        axis.set(
            xlabel="Relative time-window order",
            ylabel=f"{score} score (0-1)",
            title=title,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
        axis.legend(loc="best", fontsize=8, frameon=False)

    # Compare early, middle and late portions on one fixed score scale.
    frame["time_quartile"] = np.minimum(
        (frame["window_index"].to_numpy() * 4) // max(window_count, 1), 3
    )
    positions = np.arange(4)
    u_values = [frame.loc[frame["time_quartile"] == q, "U"] for q in range(4)]
    m_values = [frame.loc[frame["time_quartile"] == q, "M"] for q in range(4)]
    first = axes[1, 2].boxplot(
        u_values, positions=positions - 0.17, widths=0.28, patch_artist=True,
        showfliers=False, boxprops={"facecolor": "#167D8D", "alpha": 0.55},
        medianprops={"color": "black"},
    )
    second = axes[1, 2].boxplot(
        m_values, positions=positions + 0.17, widths=0.28, patch_artist=True,
        showfliers=False, boxprops={"facecolor": "#D15C32", "alpha": 0.55},
        medianprops={"color": "black"},
    )
    axes[1, 2].plot([], [], color="#167D8D", linewidth=7, alpha=0.55, label="U")
    axes[1, 2].plot([], [], color="#D15C32", linewidth=7, alpha=0.55, label="M")
    axes[1, 2].set(
        xlabel="Temporal quartile (early -> late)",
        ylabel="Score (0-1)",
        title="Score shift across time",
        xticks=positions,
        xticklabels=["0-25%", "25-50%", "50-75%", "75-100%"],
        xlim=(-0.6, 3.6),
        ylim=(0.0, 1.0),
    )
    axes[1, 2].legend(loc="best", fontsize=8, frameon=False)

    figure.colorbar(
        heatmaps[0], ax=[axes[0, 0], axes[0, 1]], shrink=0.82,
        label="U/M score (fixed 0-1 scale)",
    )
    figure.colorbar(scatter, ax=axes[0, 2], shrink=0.82, label="Relative time")
    figure.suptitle(f"{dataset}: temporal heterogeneity of local descriptors", fontsize=13)

    output = args.output or args.profiles.with_name("profile_diagnostics.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
