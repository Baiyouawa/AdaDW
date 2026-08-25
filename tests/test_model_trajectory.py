import pandas as pd
import pytest

from adawd_preexp.model_trajectory import build_model_trajectories


def _loss_table() -> pd.DataFrame:
    rows = []
    targets = {
        "depth": [1, 3, 5],
        "width": [1, 4, 16],
    }
    scores = [(0.1, 0.2), (0.5, 0.6), (0.9, 0.9)]
    for axis, capacities in (
        ("depth", [1, 2, 3, 4, 5]),
        ("width", [1, 2, 4, 8, 16]),
    ):
        for window_start, (u_score, m_score) in enumerate(scores):
            target = targets[axis][window_start]
            for seed in (3407,):
                for channel in range(7):
                    for capacity in capacities:
                        rows.append(
                            {
                                "dataset": "ETTh1",
                                "model": "PatchTST",
                                "seed": seed,
                                "horizon": 96,
                                "unit_id": f"test:{window_start}:{channel}",
                                "segment": "test",
                                "window_start": window_start,
                                "channel": channel,
                                "axis": axis,
                                "depth": capacity if axis == "depth" else 3,
                                "width_group": capacity if axis == "width" else 4,
                                "width": (capacity if axis == "width" else 4) * 4,
                                "coupled_width": (
                                    (capacity if axis == "width" else 4) * 4 * 8
                                ),
                                "loss_mse": 1.0 + abs(capacity - target),
                                "loss_mae": 1.0 + abs(capacity - target),
                                "U": u_score,
                                "M": m_score,
                            }
                        )
    return pd.DataFrame(rows)


def test_trajectory_selects_one_discrete_capacity_per_window():
    trajectory, diagnostics = build_model_trajectories(
        _loss_table(), models=["PatchTST"], horizon=96, epsilon=0.0
    )

    depth = trajectory[trajectory["axis"] == "depth"]
    width = trajectory[trajectory["axis"] == "width"]
    assert depth["best_depth"].tolist() == [1, 3, 5]
    assert width["best_width_group"].tolist() == [1, 4, 16]
    assert width["best_d_model"].tolist() == [4, 16, 64]
    assert width["best_d_ff"].tolist() == [32, 128, 512]
    assert depth["seeds"].tolist() == [1, 1, 1]
    assert depth["local_units"].tolist() == [7, 7, 7]
    assert diagnostics["per_model"]["PatchTST"]["depth"][
        "spearman_score_capacity"
    ] == pytest.approx(1.0)


def test_trajectory_rejects_incomplete_capacity_coverage():
    losses = _loss_table()
    losses = losses[~((losses["axis"] == "depth") & (losses["depth"] == 5))]

    with pytest.raises(ValueError, match="Incomplete capacity sweep"):
        build_model_trajectories(losses, models=["PatchTST"], horizon=96)


def test_near_optimal_tolerance_prefers_the_smaller_capacity():
    losses = _loss_table()
    mask = (
        (losses["axis"] == "depth")
        & (losses["window_start"] == 1)
        & (losses["depth"] == 1)
    )
    losses.loc[mask, ["loss_mse", "loss_mae"]] = 1.005

    trajectory, _ = build_model_trajectories(
        losses, models=["PatchTST"], horizon=96, epsilon=0.01
    )

    depth = trajectory[trajectory["axis"] == "depth"]
    assert depth["best_depth"].tolist() == [1, 1, 5]
