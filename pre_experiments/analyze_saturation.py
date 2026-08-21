#!/usr/bin/env python3
"""Compute saturation capacities and H1-H3 diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_preexperiment_config
from adawd_preexp.saturation import (
    add_score_buckets,
    axis_saturation,
    bucket_summary,
    joint_saturation,
    ols_diagnostics,
)


def _attach_scores(saturation: pd.DataFrame, losses: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        column
        for column in ["dataset", "unit_id", "U", "M", "u_change", "u_spectral", "u_surprise", "m_peak", "m_band", "m_channel"]
        if column in losses.columns
    ]
    scores = losses[score_columns].drop_duplicates(["dataset", "unit_id"])
    merged = saturation.merge(scores, on=["dataset", "unit_id"], how="left", validate="many_to_one")
    return add_score_buckets(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--metric", choices=["loss_mae", "loss_mse"], default="loss_mse")
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "pre_experiments" / "results" / "saturation")
    args = parser.parse_args()
    losses = pd.read_csv(args.losses)
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = load_preexperiment_config()["capacity"]["saturation_tolerance"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {"metric": args.metric, "epsilon": epsilon, "axes": {}}

    specifications = [
        ("depth", "depth", "d_sat", "U"),
        ("width", "width_group", "w_sat", "M"),
    ]
    for axis, capacity, response, score in specifications:
        subset = losses[losses["axis"] == axis]
        if subset.empty:
            diagnostics["axes"][axis] = {"status": "missing"}
            continue
        saturation = axis_saturation(subset, capacity, response, args.metric, epsilon)
        saturation = _attach_scores(saturation, subset)
        saturation.to_csv(args.output_dir / f"{axis}_saturation.csv", index=False)
        buckets = bucket_summary(saturation, response, score)
        buckets.to_csv(args.output_dir / f"{axis}_bucket_summary.csv", index=False)
        diagnostics["axes"][axis] = {
            "status": "complete",
            "num_units": int(len(saturation)),
            "regression": ols_diagnostics(saturation, response),
        }

    joint = losses[losses["axis"] == "joint"]
    if joint.empty:
        diagnostics["axes"]["joint"] = {"status": "missing"}
    else:
        saturation = joint_saturation(joint, args.metric, epsilon)
        saturation = _attach_scores(saturation, joint)
        saturation.to_csv(args.output_dir / "joint_saturation.csv", index=False)
        quadrant = (
            saturation.groupby(["dataset", "model", "quadrant"])[["d_sat", "w_sat", "capacity_cost"]]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        quadrant.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in quadrant.columns]
        quadrant.to_csv(args.output_dir / "joint_quadrant_summary.csv", index=False)
        diagnostics["axes"]["joint"] = {"status": "complete", "num_units": int(len(saturation))}

    with (args.output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
    print(args.output_dir)


if __name__ == "__main__":
    main()

