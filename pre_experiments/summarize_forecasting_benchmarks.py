#!/usr/bin/env python3
"""Aggregate normalized forecasting metrics across seeds and report coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IDENTIFIERS = ["model", "dataset", "output_length"]


def collect_runs(runs_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("axis") != "raw":
            continue
        overall = (manifest.get("metrics") or {}).get("overall") or {}
        if not overall:
            continue
        row = {
            "model": manifest["model"],
            "dataset": manifest["dataset"],
            "output_length": int(manifest["output_length"]),
            "seed": int(manifest["seed"]),
            "epochs": int(manifest["epochs"]),
            "batch_size": int(manifest["batch_size"]),
            "metric_scale": manifest.get("metric_scale", "unknown"),
            "run_id": manifest["run_id"],
        }
        row.update({name: float(value) for name, value in overall.items()})
        rows.append(row)
    columns = [
        "model", "dataset", "output_length", "seed", "epochs", "batch_size",
        "metric_scale", "run_id", "MAE", "MSE", "RMSE",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_runs(frame: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    records = []
    if frame.empty:
        return pd.DataFrame()
    for key, group in frame.groupby(IDENTIFIERS, sort=True):
        observed = sorted(group["seed"].unique().tolist())
        record = dict(zip(IDENTIFIERS, key))
        record.update(
            seed_count=len(observed),
            seeds=",".join(str(seed) for seed in observed),
            complete=observed == sorted(expected_seeds),
            epochs=int(group["epochs"].iloc[0]),
            batch_size=int(group["batch_size"].iloc[0]),
        )
        for metric in ("MAE", "MSE", "RMSE"):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else None
            )
        records.append(record)
    return pd.DataFrame(records)


def coverage(plan: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "dataset", "horizon", "seed"]
    completed = runs.rename(columns={"output_length": "horizon"})[keys].copy()
    completed["status"] = "complete"
    output = plan.merge(completed, on=keys, how="left")
    output["status"] = output["status"].fillna("pending")
    return output


def filter_runs_to_plan(runs: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    planned = plan.rename(columns={"horizon": "output_length"})[
        ["model", "dataset", "output_length", "seed", "epochs", "batch_size"]
    ]
    return runs.merge(
        planned,
        on=["model", "dataset", "output_length", "seed", "epochs", "batch_size"],
        how="inner",
        validate="one_to_one",
    )


def format_metric(row, name: str) -> str:
    mean = getattr(row, f"{name}_mean")
    std = getattr(row, f"{name}_std")
    std_text = "n/a" if std is None or pd.isna(std) else f"{std:.6f}"
    return f"{mean:.6f} +/- {std_text}"


def write_markdown(
    summary: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    expected_seeds: list[int],
    path: Path,
) -> None:
    complete_runs = int((coverage_frame["status"] == "complete").sum())
    lines = [
        "# Nine-Dataset RAW Forecasting Benchmark",
        "",
        f"Coverage: {complete_runs}/{len(coverage_frame)} runs complete.",
        "Results are mean +/- sample standard deviation across seeds "
        + ", ".join(str(seed) for seed in expected_seeds)
        + ".",
        "MAE, MSE and RMSE are computed in the per-channel ZScore-normalized space.",
        "Successful runs retain metrics and efficiency metadata only; checkpoints are deleted.",
        "",
        "| Model | Dataset | Horizon | Epochs | Batch | Seeds | MSE | MAE | RMSE | Complete |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not summary.empty:
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.dataset} | {row.output_length} | {row.epochs} "
                f"| {row.batch_size} | {row.seed_count} | {format_metric(row, 'MSE')} "
                f"| {format_metric(row, 'MAE')} | {format_metric(row, 'RMSE')} "
                f"| {'yes' if row.complete else 'no'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408, 3409])
    args = parser.parse_args()
    plan = pd.read_csv(args.plan)
    runs = filter_runs_to_plan(collect_runs(args.runs_root), plan)
    summary = aggregate_runs(runs, args.seeds)
    coverage_frame = coverage(plan, runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    coverage_frame.to_csv(args.output_dir / "coverage.csv", index=False)
    write_markdown(summary, coverage_frame, args.seeds, args.output_dir / "Result.md")
    print(f"completed_runs={len(runs)}/{len(plan)}")
    print(
        f"complete_seed_groups="
        f"{int(summary['complete'].sum()) if not summary.empty else 0}/{len(summary)}"
    )


if __name__ == "__main__":
    main()
