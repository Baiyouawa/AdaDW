"""Per-unit saturation and hypothesis diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


IDENTIFIER_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "horizon",
    "unit_id",
    "segment",
    "window_start",
    "channel",
]


def _existing(frame: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in frame.columns]


def axis_saturation(
    losses: pd.DataFrame,
    capacity_column: str,
    output_column: str,
    loss_column: str = "loss_mse",
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Select the smallest per-unit near-optimal capacity."""

    required = {"unit_id", capacity_column, loss_column}
    missing = required - set(losses.columns)
    if missing:
        raise ValueError(f"Loss table is missing columns: {sorted(missing)}")
    identifiers = _existing(losses, IDENTIFIER_COLUMNS)
    output = []
    for key, group in losses.groupby(identifiers, dropna=False, sort=False):
        key = (key,) if not isinstance(key, tuple) else key
        averaged = group.groupby(capacity_column, as_index=False)[loss_column].mean()
        optimum = averaged[loss_column].min()
        eligible = averaged[averaged[loss_column] <= (1.0 + epsilon) * optimum]
        selected = eligible.sort_values(capacity_column).iloc[0]
        row = dict(zip(identifiers, key))
        row.update(
            {
                output_column: float(selected[capacity_column]),
                "selected_loss": float(selected[loss_column]),
                "optimal_loss": float(optimum),
                "relative_tolerance": epsilon,
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def joint_saturation(
    losses: pd.DataFrame,
    loss_column: str = "loss_mse",
    epsilon: float = 0.01,
) -> pd.DataFrame:
    required = {"unit_id", "depth", "width_group", loss_column}
    missing = required - set(losses.columns)
    if missing:
        raise ValueError(f"Joint loss table is missing columns: {sorted(missing)}")
    identifiers = _existing(losses, IDENTIFIER_COLUMNS)
    output = []
    for key, group in losses.groupby(identifiers, dropna=False, sort=False):
        key = (key,) if not isinstance(key, tuple) else key
        averaged = group.groupby(["depth", "width_group"], as_index=False)[loss_column].mean()
        optimum = averaged[loss_column].min()
        eligible = averaged[averaged[loss_column] <= (1.0 + epsilon) * optimum].copy()
        eligible["cost"] = eligible["depth"] * eligible["width_group"]
        selected = eligible.sort_values(["cost", "depth", "width_group"]).iloc[0]
        row = dict(zip(identifiers, key))
        row.update(
            {
                "d_sat": float(selected["depth"]),
                "w_sat": float(selected["width_group"]),
                "capacity_cost": float(selected["cost"]),
                "selected_loss": float(selected[loss_column]),
                "optimal_loss": float(optimum),
                "relative_tolerance": epsilon,
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def add_score_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if not {"dataset", "U", "M"}.issubset(output.columns):
        raise ValueError("Profiles must contain dataset, U and M")
    for score in ("U", "M"):
        output[f"{score}_bucket"] = "mid"
        for _, indices in output.groupby("dataset").groups.items():
            values = output.loc[indices, score]
            lower, upper = values.quantile([1.0 / 3.0, 2.0 / 3.0])
            output.loc[indices[values <= lower], f"{score}_bucket"] = "low"
            output.loc[indices[values >= upper], f"{score}_bucket"] = "high"
    output["quadrant"] = "middle"
    for _, indices in output.groupby("dataset").groups.items():
        u_median = output.loc[indices, "U"].median()
        m_median = output.loc[indices, "M"].median()
        u_high = output.loc[indices, "U"] >= u_median
        m_high = output.loc[indices, "M"] >= m_median
        output.loc[indices, "quadrant"] = np.select(
            [~u_high & ~m_high, u_high & ~m_high, ~u_high & m_high, u_high & m_high],
            ["Q1", "Q2", "Q3", "Q4"],
            default="middle",
        )
    return output


def ols_diagnostics(frame: pd.DataFrame, response: str) -> Dict[str, object]:
    clean = frame[[response, "U", "M"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 3:
        return {"n": int(len(clean)), "error": "insufficient observations"}
    design = np.column_stack([np.ones(len(clean)), clean["U"], clean["M"]])
    target = clean[response].to_numpy(dtype=float)
    coefficients = np.linalg.pinv(design) @ target
    residuals = target - design @ coefficients
    degrees_freedom = len(target) - design.shape[1]
    variance = float(residuals @ residuals / degrees_freedom)
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * student_t.sf(np.abs(t_values), degrees_freedom)
    names = ["intercept", "U", "M"]
    return {
        "n": int(len(clean)),
        "degrees_freedom": int(degrees_freedom),
        "r_squared": float(1.0 - (residuals @ residuals) / np.sum((target - target.mean()) ** 2))
        if np.var(target) > 0
        else None,
        "terms": {
            name: {
                "coefficient": float(coefficients[index]),
                "standard_error": float(standard_errors[index]),
                "t": float(t_values[index]),
                "p": float(p_values[index]),
            }
            for index, name in enumerate(names)
        },
    }


def bucket_summary(frame: pd.DataFrame, response: str, score: str) -> pd.DataFrame:
    bucket_column = f"{score}_bucket"
    order = pd.CategoricalDtype(["low", "mid", "high"], ordered=True)
    output = (
        frame.assign(**{bucket_column: frame[bucket_column].astype(order)})
        .groupby(_existing(frame, ["dataset", "model"]) + [bucket_column], observed=True)[response]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    return output
