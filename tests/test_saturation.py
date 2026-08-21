import pandas as pd

from adawd_preexp.saturation import axis_saturation, joint_saturation


def test_axis_saturation_selects_smallest_near_optimal_capacity():
    losses = pd.DataFrame(
        {
            "dataset": ["D"] * 4,
            "model": ["M"] * 4,
            "unit_id": ["u0"] * 4,
            "depth": [1, 2, 4, 8],
            "loss_mse": [1.2, 1.005, 1.0, 1.1],
        }
    )
    result = axis_saturation(losses, "depth", "d_sat", epsilon=0.01)
    assert result.loc[0, "d_sat"] == 2


def test_joint_saturation_minimizes_depth_times_width():
    losses = pd.DataFrame(
        {
            "dataset": ["D"] * 4,
            "model": ["M"] * 4,
            "unit_id": ["u0"] * 4,
            "depth": [2, 2, 4, 8],
            "width_group": [4, 8, 2, 2],
            "loss_mse": [1.0, 0.995, 1.0, 0.995],
        }
    )
    result = joint_saturation(losses, epsilon=0.01)
    assert result.loc[0, "d_sat"] == 2
    assert result.loc[0, "w_sat"] == 4
    assert result.loc[0, "capacity_cost"] == 8
