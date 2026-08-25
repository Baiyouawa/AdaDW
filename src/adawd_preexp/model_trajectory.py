"""Temporal best-capacity analysis for Backbone pre-experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .catalog import load_backbone_registry, load_preexperiment_config, resolve_dataset
from .capacity import capacity_candidates


SCORE_COLUMNS = (
    "U",
    "M",
    "u_change",
    "u_spectral",
    "u_surprise",
    "m_peak",
    "m_band",
    "m_channel",
)
WINDOW_KEYS = ("dataset", "model", "horizon", "segment", "window_start")


def _finite_spearman(left: pd.Series, right: pd.Series) -> tuple[float | None, float | None]:
    values = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(values) < 3 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return None, None
    result = spearmanr(values["left"], values["right"])
    return float(result.statistic), float(result.pvalue)


def _circular_shift_pvalue(
    left: pd.Series,
    right: pd.Series,
) -> float | None:
    """Test temporal association while retaining each trajectory's ordering."""

    values = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(values) < 4 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return None
    left_values = values["left"].to_numpy(dtype=float)
    right_values = values["right"].to_numpy(dtype=float)
    observed = abs(float(spearmanr(left_values, right_values).statistic))
    exceedances = 0
    for shift in range(1, len(values)):
        statistic = abs(
            float(spearmanr(left_values, np.roll(right_values, shift)).statistic)
        )
        exceedances += statistic >= observed
    return float((exceedances + 1) / len(values))


def _normalise_models(models: Iterable[str] | None) -> list[str] | None:
    if models is None:
        return None
    values = list(dict.fromkeys(str(model) for model in models))
    if not values:
        raise ValueError("At least one model is required")
    registry = load_backbone_registry()["models"]
    unknown = sorted(set(values) - set(registry))
    if unknown:
        raise KeyError(f"Unknown Backbone(s): {', '.join(unknown)}")
    unavailable = [model for model in values if registry[model].get("status") != "available"]
    if unavailable:
        raise RuntimeError(f"Unavailable Backbone(s): {', '.join(unavailable)}")
    return values


def _filter_losses(
    losses: pd.DataFrame,
    dataset: str | None,
    models: Iterable[str] | None,
    horizon: int | None,
    metric: str,
) -> pd.DataFrame:
    required = {
        *WINDOW_KEYS,
        "seed",
        "unit_id",
        "channel",
        "axis",
        "depth",
        "width_group",
        "width",
        "coupled_width",
        metric,
        "U",
        "M",
    }
    missing = sorted(required - set(losses.columns))
    if missing:
        raise ValueError(f"Loss table is missing columns: {missing}")
    output = losses.copy()
    if dataset is not None:
        output = output[output["dataset"] == dataset]
    selected_models = _normalise_models(models)
    if selected_models is not None:
        output = output[output["model"].isin(selected_models)]
    if horizon is not None:
        output = output[output["horizon"] == horizon]
    if output.empty:
        raise ValueError("No loss rows remain after dataset/model/horizon filtering")
    if selected_models is not None:
        missing_models = sorted(set(selected_models) - set(output["model"].unique()))
        if missing_models:
            raise ValueError(
                f"Loss table has no rows for requested model(s): {', '.join(missing_models)}"
            )
    if output["dataset"].nunique() != 1:
        raise ValueError("Trajectory analysis requires exactly one dataset")
    if horizon is None and output["horizon"].nunique() != 1:
        values = ", ".join(str(value) for value in sorted(output["horizon"].unique()))
        raise ValueError(f"Multiple horizons found ({values}); pass --horizon explicitly")
    numeric = output[[metric, "U", "M"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Loss table contains non-finite {metric}, U or M values")
    if (output[metric] < 0).any():
        raise ValueError(f"Loss table contains negative {metric} values")
    return output


def _score_summary(losses: pd.DataFrame) -> pd.DataFrame:
    """Summarize scores without counting repeated seeds/capacities as new units."""

    score_columns = [column for column in SCORE_COLUMNS if column in losses.columns]
    profiles = losses[[*WINDOW_KEYS, "channel", *score_columns]].drop_duplicates(
        [*WINDOW_KEYS, "channel"]
    )
    named_aggregations: dict[str, tuple[str, object]] = {
        "profiled_channels": ("channel", "nunique")
    }
    for score in ("U", "M"):
        named_aggregations[f"{score}_mean"] = (score, "mean")
        named_aggregations[f"{score}_p10"] = (
            score,
            lambda values: values.quantile(0.10),
        )
        named_aggregations[f"{score}_p90"] = (
            score,
            lambda values: values.quantile(0.90),
        )
    return (
        profiles.groupby(list(WINDOW_KEYS), sort=True)
        .agg(**named_aggregations)
        .reset_index()
    )


def _validate_coverage(losses: pd.DataFrame) -> None:
    """Reject partial sweeps instead of selecting from an incomplete candidate set."""

    experiment = load_preexperiment_config()
    expected_seeds = tuple(
        sorted(int(seed) for seed in experiment["model_trajectory"]["seeds"])
    )
    registry = load_backbone_registry()["models"]
    problems: list[str] = []
    temporal_channel_keys = (
        "dataset",
        "horizon",
        "segment",
        "window_start",
        "channel",
    )
    score_variants = losses.groupby(list(temporal_channel_keys))[["U", "M"]].nunique()
    inconsistent_scores = int((score_variants > 1).any(axis=1).sum())
    if inconsistent_scores:
        problems.append(
            f"{inconsistent_scores} window/channel units have inconsistent U/M scores "
            "across models, axes, capacities or seeds"
        )
    for model in sorted(losses["model"].unique()):
        specifications = (
            ("depth", "depth", set(capacity_candidates(model, "depth"))),
            ("width", "width_group", set(capacity_candidates(model, "width"))),
        )
        for axis, capacity_column, expected in specifications:
            subset = losses[(losses["model"] == model) & (losses["axis"] == axis)]
            if subset.empty:
                problems.append(f"{model}/{axis}: no rows")
                continue
            model_entry = registry[model]
            raw_width_group = int(
                model_entry["raw_width"] // model_entry["width_unit"]
            )
            if axis == "depth" and set(subset["width_group"].astype(int)) != {
                raw_width_group
            }:
                problems.append(
                    f"{model}/{axis}: width must stay at RAW group {raw_width_group}"
                )
            if axis == "width" and set(subset["depth"].astype(int)) != {
                int(model_entry["raw_depth"])
            }:
                problems.append(
                    f"{model}/{axis}: depth must stay at RAW value "
                    f"{model_entry['raw_depth']}"
                )
            expected_width = (
                subset["width_group"].astype(int) * int(model_entry["width_unit"])
            )
            if not np.array_equal(
                subset["width"].astype(int).to_numpy(), expected_width.to_numpy()
            ):
                problems.append(
                    f"{model}/{axis}: concrete width does not equal "
                    "width_group * width_unit"
                )
            if "coupled_width_parameter" in model_entry:
                expected_coupled_width = (
                    subset["width"].astype(int)
                    * int(model_entry["coupled_width_ratio"])
                )
                if not np.array_equal(
                    subset["coupled_width"].astype(int).to_numpy(),
                    expected_coupled_width.to_numpy(),
                ):
                    problems.append(
                        f"{model}/{axis}: coupled width does not preserve the "
                        "registered d_ff/d_model ratio"
                    )
            actual = set(subset[capacity_column].astype(int).unique())
            if actual != expected:
                problems.append(
                    f"{model}/{axis}: candidates={sorted(actual)}, expected={sorted(expected)}"
                )
                continue
            coverage = subset[
                [*WINDOW_KEYS, capacity_column, "seed", "channel"]
            ].drop_duplicates()
            window_counts = coverage.groupby(list(WINDOW_KEYS))[capacity_column].nunique()
            incomplete = int((window_counts != len(expected)).sum())
            if incomplete:
                problems.append(f"{model}/{axis}: {incomplete} windows miss candidates")
            seed_coverage = coverage.groupby(list(WINDOW_KEYS))["seed"].agg(
                lambda values: tuple(sorted(set(int(value) for value in values)))
            )
            missing_seeds = int(
                seed_coverage.map(lambda values: values != expected_seeds).sum()
            )
            if missing_seeds:
                problems.append(
                    f"{model}/{axis}: {missing_seeds} windows do not contain seeds "
                    f"{list(expected_seeds)}"
                )
            _, dataset_entry = resolve_dataset(str(subset["dataset"].iloc[0]))
            channel_limit = dataset_entry.get("max_profile_channels")
            expected_channel_count = min(
                int(dataset_entry["expected_channels"]),
                int(channel_limit)
                if channel_limit is not None
                else int(dataset_entry["expected_channels"]),
            )
            channel_coverage = coverage.groupby(list(WINDOW_KEYS))["channel"].nunique()
            missing_channels = int((channel_coverage != expected_channel_count).sum())
            if missing_channels:
                problems.append(
                    f"{model}/{axis}: {missing_channels} windows do not contain "
                    f"{expected_channel_count} profiled channels"
                )
            coverage = coverage.assign(
                sample_key=coverage["seed"].astype(str)
                + ":"
                + coverage["channel"].astype(str)
            )
            signatures = (
                coverage.groupby([*WINDOW_KEYS, capacity_column], sort=False)[
                    "sample_key"
                ]
                .agg(lambda values: tuple(sorted(values)))
                .rename("signature")
                .reset_index()
            )
            inconsistent = int(
                (
                    signatures.groupby(list(WINDOW_KEYS))["signature"].nunique()
                    != 1
                ).sum()
            )
            if inconsistent:
                problems.append(
                    f"{model}/{axis}: {inconsistent} windows have inconsistent seed/channel coverage"
                )
    reference_windows: set[tuple[object, ...]] | None = None
    reference_model: str | None = None
    temporal_keys = ("dataset", "horizon", "segment", "window_start")
    for model in sorted(losses["model"].unique()):
        depth_windows = set(
            losses.loc[
                (losses["model"] == model) & (losses["axis"] == "depth"),
                list(temporal_keys),
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        width_windows = set(
            losses.loc[
                (losses["model"] == model) & (losses["axis"] == "width"),
                list(temporal_keys),
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if depth_windows != width_windows:
            problems.append(
                f"{model}: depth/width temporal windows differ "
                f"(depth_only={len(depth_windows - width_windows)}, "
                f"width_only={len(width_windows - depth_windows)})"
            )
        model_windows = depth_windows | width_windows
        if reference_windows is None:
            reference_windows = model_windows
            reference_model = model
        elif model_windows != reference_windows:
            problems.append(
                f"{model}: temporal windows differ from {reference_model} "
                f"(missing={len(reference_windows - model_windows)}, "
                f"extra={len(model_windows - reference_windows)})"
            )
    if problems:
        raise ValueError("Incomplete capacity sweep: " + "; ".join(problems))


def _select_temporal_capacity(
    losses: pd.DataFrame,
    axis: str,
    metric: str,
    epsilon: float,
) -> pd.DataFrame:
    """Select one near-optimal capacity after pooling channels and seeds per window."""

    if axis == "depth":
        capacity_column, output_column = "depth", "best_depth"
    elif axis == "width":
        capacity_column, output_column = "width_group", "best_width_group"
    else:
        raise ValueError("axis must be depth or width")
    subset = losses[losses["axis"] == axis]
    if subset.empty:
        raise ValueError(f"No '{axis}' rows found in loss table")
    averaged = (
        subset.groupby([*WINDOW_KEYS, capacity_column], as_index=False, sort=True)[metric]
        .mean()
        .rename(columns={metric: "mean_loss"})
    )
    rows: list[dict[str, object]] = []
    for key, group in averaged.groupby(list(WINDOW_KEYS), sort=True, dropna=False):
        optimum = float(group["mean_loss"].min())
        eligible = group[group["mean_loss"] <= (1.0 + epsilon) * optimum]
        selected = eligible.sort_values(capacity_column).iloc[0]
        row = dict(zip(WINDOW_KEYS, key))
        row.update(
            {
                output_column: int(selected[capacity_column]),
                "selected_loss": float(selected["mean_loss"]),
                "optimal_loss": optimum,
                "relative_tolerance": float(epsilon),
            }
        )
        rows.append(row)
    selected = pd.DataFrame(rows)
    selected = selected.merge(
        _score_summary(subset),
        on=list(WINDOW_KEYS),
        how="left",
        validate="one_to_one",
    )
    counts = (
        subset.groupby(list(WINDOW_KEYS), sort=True)
        .agg(seeds=("seed", "nunique"), local_units=("unit_id", "nunique"))
        .reset_index()
    )
    selected = selected.merge(
        counts, on=list(WINDOW_KEYS), how="left", validate="one_to_one"
    )
    if axis == "width":
        registry = load_backbone_registry()["models"]
        selected["best_width"] = selected.apply(
            lambda row: int(
                row["best_width_group"] * registry[row["model"]]["width_unit"]
            ),
            axis=1,
        )
        selected["best_d_model"] = selected["best_width"]
        selected["best_d_ff"] = selected.apply(
            lambda row: int(
                row["best_width"]
                * registry[row["model"]].get("coupled_width_ratio", 1)
            ),
            axis=1,
        )
    selected.insert(0, "axis", axis)
    selected = selected.sort_values(
        ["model", "segment", "window_start"]
    ).reset_index(drop=True)
    selected["window_index"] = selected.groupby("model", sort=False).cumcount()
    maxima = selected.groupby("model")["window_index"].transform("max")
    selected["relative_time"] = np.divide(
        selected["window_index"],
        maxima,
        out=np.zeros(len(selected), dtype=float),
        where=maxima.to_numpy() > 0,
    )
    return selected


def build_model_trajectories(
    losses: pd.DataFrame,
    *,
    dataset: str | None = "ETTh1",
    models: Iterable[str] | None = None,
    horizon: int | None = None,
    metric: str = "loss_mse",
    epsilon: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select discrete best depth/width per time window and compute correlations.

    Candidate loss is averaged over profiled channels and random seeds inside
    each window. ``epsilon=0`` selects the strict loss minimum; a positive
    epsilon selects the smallest capacity within that relative distance of the
    minimum, reducing noisy preference for unnecessarily large models.
    """

    if metric not in {"loss_mse", "loss_mae"}:
        raise ValueError("metric must be loss_mse or loss_mae")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    filtered = _filter_losses(losses, dataset, models, horizon, metric)
    _validate_coverage(filtered)
    trajectory = pd.concat(
        [
            _select_temporal_capacity(filtered, axis, metric, epsilon)
            for axis in ("depth", "width")
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics: dict[str, object] = {
        "dataset": str(filtered["dataset"].iloc[0]),
        "horizon": int(filtered["horizon"].iloc[0]),
        "metric": metric,
        "epsilon": float(epsilon),
        "selection_unit": "time_window",
        "loss_aggregation": "mean_across_channels_and_seeds",
        "models": sorted(filtered["model"].unique().tolist()),
        "axes": {},
        "per_model": {},
    }
    for axis, score, capacity in (
        ("depth", "U_mean", "best_depth"),
        ("width", "M_mean", "best_width_group"),
    ):
        frame = trajectory[trajectory["axis"] == axis]
        rho, pvalue = _finite_spearman(frame[score], frame[capacity])
        diagnostics["axes"][axis] = {
            "n_windows": int(len(frame)),
            "scope": "pooled_descriptive_only",
            "spearman_score_capacity": rho,
            "spearman_pvalue": pvalue,
        }
    for model in diagnostics["models"]:
        diagnostics["per_model"][model] = {}
        for axis, score, capacity in (
            ("depth", "U_mean", "best_depth"),
            ("width", "M_mean", "best_width_group"),
        ):
            frame = trajectory[
                (trajectory["model"] == model) & (trajectory["axis"] == axis)
            ]
            rho, pvalue = _finite_spearman(frame[score], frame[capacity])
            diagnostics["per_model"][model][axis] = {
                "n_windows": int(len(frame)),
                "spearman_score_capacity": rho,
                "spearman_pvalue": pvalue,
                "circular_shift_pvalue": _circular_shift_pvalue(
                    frame[score], frame[capacity]
                ),
            }
    return trajectory, diagnostics
