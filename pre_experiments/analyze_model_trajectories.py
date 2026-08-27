#!/usr/bin/env python3
"""Select and plot temporal best-depth/best-width trajectories for Backbones."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_preexperiment_config
from adawd_preexp.model_trajectory import build_model_trajectories


MODEL_COLORS = {
    "PatchTST": "#167D8D",
    "iTransformer": "#C94F2D",
    "TimeFilter": "#6A4C93",
    "Crossformer": "#3A6EA5",
    "TimeMixer": "#2A9D8F",
    "WPMixer": "#E9C46A",
    "MultiPatchFormer": "#8A5A44",
}


def _spearman_label(left: pd.Series, right: pd.Series) -> str:
    if left.nunique() < 2 or right.nunique() < 2:
        return "n/a"
    return f"{float(spearmanr(left, right).statistic):.2f}"


def _plot_trajectories(
    trajectory: pd.DataFrame,
    output: Path,
    dataset: str,
    model_order: list[str] | None,
) -> None:
    available = set(trajectory["model"].unique())
    models = [model for model in (model_order or []) if model in available]
    models.extend(sorted(available - set(models)))
    figure, axes = plt.subplots(
        len(models), 2, figsize=(11.0, max(3.3, 2.8 * len(models))),
        squeeze=False, sharex=True, constrained_layout=True,
    )
    for row, model in enumerate(models):
        color = MODEL_COLORS.get(model, f"C{row}")
        for column, (axis_name, score, capacity, capacity_label) in enumerate(
            (("depth", "U_mean", "best_depth", "best depth"),
             ("width", "M_mean", "best_d_model", "best d_model"))
        ):
            axis = axes[row][column]
            frame = trajectory[(trajectory["model"] == model) & (trajectory["axis"] == axis_name)]
            if frame.empty:
                axis.set_visible(False)
                continue
            x = frame["relative_time"].to_numpy(dtype=float)
            score_values = frame[score].to_numpy(dtype=float)
            score_low = frame[f"{score[:1]}_p10"].to_numpy(dtype=float)
            score_high = frame[f"{score[:1]}_p90"].to_numpy(dtype=float)
            capacity_values = frame[capacity].to_numpy(dtype=float)
            capacity_axis = axis.twinx()
            axis.fill_between(x, score_low, score_high, color=color, alpha=0.14, linewidth=0)
            axis.plot(x, score_values, color=color, linewidth=1.35, label=score.split("_")[0])
            capacity_axis.step(
                x,
                capacity_values,
                where="mid",
                color="#333333",
                linewidth=1.2,
                linestyle="--",
                label=capacity_label,
            )
            axis.set_ylabel(score.split("_")[0], color=color)
            capacity_axis.set_ylabel(capacity_label, color="#333333")
            axis.set_ylim(0.0, 1.0)
            capacity_axis.set_ylim(bottom=0.0)
            axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.45)
            rho = _spearman_label(frame[score], frame[capacity])
            axis.set_title(
                f"{model}: {axis_name} capacity (Spearman rho={rho})",
                loc="left",
                fontsize=10,
            )
            axis.tick_params(axis="y", colors=color, direction="out")
            capacity_axis.tick_params(axis="y", colors="#333333", direction="out")
            lines, labels = axis.get_legend_handles_labels()
            lines2, labels2 = capacity_axis.get_legend_handles_labels()
            axis.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Relative time-window order")
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    figure.suptitle(f"{dataset}: U/M and temporal best-capacity trajectories", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze(
    losses: Path,
    output_dir: Path,
    *,
    dataset: str = "ETTh1",
    models: list[str] | None = None,
    horizon: int | None = None,
    metric: str = "loss_mse",
    epsilon: float | None = None,
) -> list[Path]:
    losses = Path(losses)
    if not losses.is_file():
        raise FileNotFoundError(
            f"Loss table not found: {losses}. "
            "Run `pixi run model-preexp-losses` first."
        )
    config = load_preexperiment_config()
    tolerance = (
        config["model_trajectory"]["selection_tolerance"]
        if epsilon is None
        else epsilon
    )
    trajectory, diagnostics = build_model_trajectories(
        pd.read_csv(losses), dataset=dataset, models=models, horizon=horizon,
        metric=metric, epsilon=float(tolerance),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "model_capacity_trajectories.csv"
    diagnostics_path = output_dir / "model_capacity_correlations.json"
    figure_path = output_dir / "model_capacity_temporal_trajectories.pdf"
    trajectory.to_csv(trajectory_path, index=False)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_trajectories(trajectory, figure_path, dataset, models)
    return [trajectory_path, diagnostics_path, figure_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "pre_experiments" / "results" / "model_trajectories")
    experiment = load_preexperiment_config()["model_trajectory"]
    parser.add_argument("--dataset", default=experiment["dataset"])
    parser.add_argument("--models", nargs="+", default=experiment["models"])
    parser.add_argument("--horizon", type=int, default=experiment["horizon"])
    parser.add_argument(
        "--metric",
        choices=["loss_mse", "loss_mae"],
        default=experiment["metric"],
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="near-optimal tolerance; default is model_trajectory.selection_tolerance",
    )
    args = parser.parse_args()
    for path in analyze(args.losses, args.output_dir, dataset=args.dataset, models=args.models,
                        horizon=args.horizon, metric=args.metric, epsilon=args.epsilon):
        print(path)


if __name__ == "__main__":
    main()
