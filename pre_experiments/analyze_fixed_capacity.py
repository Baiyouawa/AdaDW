#!/usr/bin/env python3
"""Diagnose whether one fixed model capacity can be optimal in every window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_preexperiment_config
from adawd_preexp.model_trajectory import build_model_trajectories


MODEL_COLORS = {
    "PatchTST": "#167D8D",
    "TimeMixer": "#D97706",
    "MultiPatchFormer": "#7C3AED",
}
AXES = ("depth", "width")


def _capacity_column(axis: str) -> str:
    if axis == "depth":
        return "depth"
    if axis == "width":
        return "width"
    raise ValueError("axis must be depth or width")


def _window_loss_matrix(frame: pd.DataFrame, axis: str, metric: str) -> pd.DataFrame:
    capacity = _capacity_column(axis)
    matrix = (
        frame[frame["axis"] == axis]
        .groupby(["segment", "window_start", capacity], sort=True)[metric]
        .mean()
        .unstack(capacity)
        .sort_index()
        .sort_index(axis=1)
    )
    if matrix.empty or matrix.isna().any().any():
        raise ValueError(f"Incomplete window/capacity loss matrix for axis={axis}")
    return matrix


def _is_strictly_larger(left: pd.Series, right: pd.Series) -> pd.Series:
    tied = np.isclose(left.to_numpy(), right.to_numpy(), rtol=1e-12, atol=1e-12)
    return pd.Series((left.to_numpy() > right.to_numpy()) & ~tied, index=left.index)


def summarize_fixed_capacity(
    losses: pd.DataFrame,
    *,
    dataset: str,
    models: list[str],
    horizon: int,
    metric: str = "loss_mse",
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Compare the best fixed capacity with a per-window test-target oracle."""

    # This call applies the experiment's full candidate/seed/channel coverage checks.
    build_model_trajectories(
        losses,
        dataset=dataset,
        models=models,
        horizon=horizon,
        metric=metric,
        epsilon=0.0,
    )
    filtered = losses[
        (losses["dataset"] == dataset)
        & (losses["model"].isin(models))
        & (losses["horizon"] == horizon)
    ]
    rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for model in models:
        model_frame = filtered[filtered["model"] == model]
        for axis in AXES:
            matrix = _window_loss_matrix(model_frame, axis, metric)
            matrices[(model, axis)] = matrix
            mean_by_capacity = matrix.mean(axis=0)
            best_fixed = mean_by_capacity.idxmin()
            oracle = matrix.min(axis=1)
            winners = matrix.idxmin(axis=1)
            fixed_losses = matrix[best_fixed]
            fixed_suboptimal = _is_strictly_larger(fixed_losses, oracle)
            optimal_mask = pd.DataFrame(
                np.isclose(
                    matrix.to_numpy(),
                    oracle.to_numpy()[:, np.newaxis],
                    rtol=1e-12,
                    atol=1e-12,
                ),
                index=matrix.index,
                columns=matrix.columns,
            )
            optimal_counts = optimal_mask.sum(axis=0)

            starts = matrix.index.get_level_values("window_start").to_numpy(dtype=int)
            positive_deltas = np.diff(starts)
            positive_deltas = positive_deltas[positive_deltas > 0]
            stride = int(np.median(positive_deltas)) if len(positive_deltas) else horizon
            # Sampling at least one horizon apart gives a simple disjoint-target
            # robustness check for the single test segment used here.
            disjoint_step = max(1, int(np.ceil(horizon / stride)))
            disjoint = matrix.iloc[::disjoint_step]
            disjoint_fixed = disjoint.mean(axis=0).idxmin()
            disjoint_oracle = disjoint.min(axis=1)
            disjoint_suboptimal = _is_strictly_larger(
                disjoint[disjoint_fixed], disjoint_oracle
            )

            fixed_mean = float(fixed_losses.mean())
            oracle_mean = float(oracle.mean())
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "axis": axis,
                    "capacity_parameter": "num_layers" if axis == "depth" else "d_model",
                    "candidates": ",".join(str(int(value)) for value in matrix.columns),
                    "n_windows": int(len(matrix)),
                    "best_fixed_capacity": int(best_fixed),
                    "best_fixed_mse": fixed_mean,
                    "per_window_oracle_mse": oracle_mean,
                    "oracle_gain_pct": 100.0 * (fixed_mean - oracle_mean) / fixed_mean,
                    "best_fixed_optimal_windows": int(optimal_mask[best_fixed].sum()),
                    "best_fixed_optimal_pct": 100.0 * float(optimal_mask[best_fixed].mean()),
                    "best_fixed_suboptimal_windows": int(fixed_suboptimal.sum()),
                    "best_fixed_suboptimal_pct": 100.0 * float(fixed_suboptimal.mean()),
                    "max_any_fixed_optimal_windows": int(optimal_counts.max()),
                    "max_any_fixed_optimal_pct": 100.0 * float(optimal_counts.max() / len(matrix)),
                    "distinct_window_optima": int(winners.nunique()),
                    "capacity_switches": int(winners.ne(winners.shift()).sum() - 1),
                    "disjoint_n_windows": int(len(disjoint)),
                    "disjoint_best_fixed_capacity": int(disjoint_fixed),
                    "disjoint_oracle_gain_pct": 100.0
                    * float((disjoint[disjoint_fixed].mean() - disjoint_oracle.mean())
                            / disjoint[disjoint_fixed].mean()),
                    "disjoint_fixed_suboptimal_pct": 100.0
                    * float(disjoint_suboptimal.mean()),
                    "disjoint_distinct_window_optima": int(
                        disjoint.idxmin(axis=1).nunique()
                    ),
                }
            )
    return pd.DataFrame(rows), matrices


def _plot_fixed_capacity_gap(
    summary: pd.DataFrame,
    matrices: dict[tuple[str, str], pd.DataFrame],
    output: Path,
    *,
    dataset: str,
    horizon: int,
) -> None:
    combinations = [(model, axis) for model in summary["model"].unique() for axis in AXES]
    winner_ranks: list[np.ndarray] = []
    row_labels: list[str] = []
    for model, axis in combinations:
        matrix = matrices[(model, axis)]
        winners = matrix.idxmin(axis=1)
        rank = {capacity: index for index, capacity in enumerate(matrix.columns)}
        winner_ranks.append(winners.map(rank).to_numpy(dtype=int))
        candidates = ", ".join(str(int(value)) for value in matrix.columns)
        row_labels.append(f"{model}  {axis}\n[{candidates}]")

    figure = plt.figure(figsize=(13.6, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    timeline_axis = figure.add_subplot(grid[0, :])
    gain_axis = figure.add_subplot(grid[1, 0])
    suboptimal_axis = figure.add_subplot(grid[1, 1])

    capacity_colors = ["#287271", "#7FAE9D", "#E9C46A", "#F4A261", "#C94F2D"]
    color_map = ListedColormap(capacity_colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), color_map.N)
    image = timeline_axis.imshow(
        np.vstack(winner_ranks),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=norm,
    )
    timeline_axis.set_yticks(range(len(row_labels)), row_labels)
    timeline_axis.set_xticks(
        np.linspace(0, len(winner_ranks[0]) - 1, 5),
        ["0%", "25%", "50%", "75%", "100%"],
    )
    timeline_axis.set_xlabel("Relative test-window order")
    timeline_axis.set_title(
        "A. Strict per-window optimum changes across time",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    timeline_axis.tick_params(axis="y", length=0, pad=9)
    timeline_axis.tick_params(axis="x", length=0, pad=5)
    for position in (1.5, 3.5):
        timeline_axis.axhline(position, color="white", linewidth=3.0)
    for spine in timeline_axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(
        image,
        ax=timeline_axis,
        location="right",
        shrink=0.88,
        pad=0.015,
        ticks=range(5),
    )
    colorbar.set_ticklabels(["smallest", "2nd", "3rd", "4th", "largest"])
    colorbar.set_label("Winning capacity rank")
    colorbar.outline.set_visible(False)

    short_names = {
        "PatchTST": "PatchTST",
        "TimeMixer": "TimeMixer",
        "MultiPatchFormer": "MPFormer",
    }
    labels = [
        f"{short_names.get(model, model)}\n{'D' if axis == 'depth' else 'W'}"
        for model, axis in combinations
    ]
    x = np.arange(len(combinations))
    colors = [MODEL_COLORS.get(model, "#555555") for model, _ in combinations]
    hatches = ["" if axis == "depth" else "///" for _, axis in combinations]
    ordered = summary.set_index(["model", "axis"]).loc[combinations]

    gain_bars = gain_axis.bar(
        x,
        ordered["oracle_gain_pct"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.55,
    )
    for bar, hatch, value in zip(gain_bars, hatches, ordered["oracle_gain_pct"]):
        bar.set_hatch(hatch)
        gain_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    gain_axis.set_ylim(0, max(6.3, float(ordered["oracle_gain_pct"].max()) + 0.8))
    gain_axis.set_ylabel("MSE reduction (%)")
    gain_axis.set_title(
        "B. Per-window oracle improves on the best fixed model",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    suboptimal_bars = suboptimal_axis.bar(
        x,
        ordered["best_fixed_suboptimal_pct"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.55,
    )
    for bar, hatch, value in zip(
        suboptimal_bars, hatches, ordered["best_fixed_suboptimal_pct"]
    ):
        bar.set_hatch(hatch)
        suboptimal_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    suboptimal_axis.set_ylim(0, 88)
    suboptimal_axis.set_ylabel("Strictly suboptimal windows (%)")
    suboptimal_axis.set_title(
        "C. The best fixed model misses most windows",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    for axis in (gain_axis, suboptimal_axis):
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", length=0, labelsize=8.5)
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"{dataset}: no fixed depth or width is optimal across all time windows",
        fontsize=15,
        fontweight="bold",
        y=1.015,
    )
    figure.text(
        0.5,
        -0.015,
        f"3 backbones | {int(summary['n_windows'].iloc[0])} sampled windows | horizon "
        f"{horizon} | MSE | seed 3407. D = depth, W = width. Test-target oracle "
        "measures headroom; it is not a deployable selector.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def analyze(
    losses_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    models: list[str],
    horizon: int,
    metric: str = "loss_mse",
) -> list[Path]:
    if not losses_path.is_file():
        raise FileNotFoundError(f"Loss table not found: {losses_path}")
    losses = pd.read_csv(losses_path)
    summary, matrices = summarize_fixed_capacity(
        losses,
        dataset=dataset,
        models=models,
        horizon=horizon,
        metric=metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fixed_capacity_gap.csv"
    pdf_path = output_dir / "fixed_capacity_vs_oracle.pdf"
    png_path = output_dir / "fixed_capacity_vs_oracle.png"
    summary.to_csv(summary_path, index=False)
    _plot_fixed_capacity_gap(summary, matrices, pdf_path, dataset=dataset, horizon=horizon)
    _plot_fixed_capacity_gap(summary, matrices, png_path, dataset=dataset, horizon=horizon)
    return [summary_path, pdf_path, png_path]


def main() -> None:
    experiment = load_preexperiment_config()["model_trajectory"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--losses",
        type=Path,
        default=PROJECT_ROOT
        / "pre_experiments"
        / "results"
        / "model_trajectories"
        / "local_losses.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pre_experiments" / "results" / "model_trajectories",
    )
    parser.add_argument("--dataset", default=experiment["dataset"])
    parser.add_argument("--models", nargs="+", default=experiment["models"])
    parser.add_argument("--horizon", type=int, default=experiment["horizon"])
    parser.add_argument("--metric", choices=["loss_mse", "loss_mae"], default="loss_mse")
    args = parser.parse_args()
    for path in analyze(
        args.losses,
        args.output_dir,
        dataset=args.dataset,
        models=args.models,
        horizon=args.horizon,
        metric=args.metric,
    ):
        print(path)


if __name__ == "__main__":
    main()
