import pandas as pd

from pre_experiments.run_forecasting_benchmarks import (
    build_plan,
    load_config,
    training_command,
)
from pre_experiments.summarize_forecasting_benchmarks import (
    aggregate_runs,
    coverage,
    filter_runs_to_plan,
)


def test_default_forecasting_plan_contains_864_runs():
    config = load_config(
        __import__("pathlib").Path("pre_experiments/benchmark_config.json")
    )
    plan = build_plan(config)

    assert len(plan) == 8 * 9 * 4 * 3 == 864
    assert len({run.protocol_signature for run in plan}) == 864
    assert len({run.run_id for run in plan}) == 864
    assert {run.visualize_forecast for run in plan} == {True}
    assert {run.visualization_sample_position for run in plan} == {0.5}
    assert {run.visualization_max_channels for run in plan} == {4}
    assert config["seeds"] == [3407, 3408, 3409]
    ili_horizons = {run.horizon for run in plan if run.dataset == "ILI"}
    ett_horizons = {run.horizon for run in plan if run.dataset == "ETTh1"}
    assert ili_horizons == {24, 36, 48, 60}
    assert ett_horizons == {96, 192, 336, 720}
    command = training_command(plan[0], config, __import__("pathlib").Path("runs"), "0")
    assert "--visualize-forecast" in command
    assert command[command.index("--visualization-sample-position") + 1] == "0.5"


def test_smoke_plan_contains_one_epoch_per_model_and_dataset():
    config = load_config(
        __import__("pathlib").Path("pre_experiments/benchmark_config.json")
    )
    plan = build_plan(config, smoke=True)

    assert len(plan) == 8 * 9 == 72
    assert {run.seed for run in plan} == {3407}
    assert {run.epochs for run in plan} == {1}
    assert {run.horizon for run in plan if run.dataset == "ILI"} == {24}
    assert {run.horizon for run in plan if run.dataset == "ETTh1"} == {96}


def test_seed_aggregation_and_coverage():
    runs = pd.DataFrame(
        {
            "model": ["M"] * 3,
            "dataset": ["D"] * 3,
            "output_length": [96] * 3,
            "seed": [3407, 3408, 3409],
            "epochs": [10] * 3,
            "batch_size": [32] * 3,
            "metric_scale": ["normalized"] * 3,
            "protocol_signature": ["p"] * 3,
            "data_fingerprint": ["d"] * 3,
            "run_id": ["a", "b", "c"],
            "MSE": [0.2, 0.3, 0.4],
            "MAE": [0.1, 0.2, 0.3],
            "RMSE": [0.4, 0.5, 0.6],
        }
    )
    plan = pd.DataFrame(
        {
            "index": [1, 2, 3, 4],
            "model": ["M"] * 4,
            "dataset": ["D"] * 4,
            "horizon": [96] * 4,
            "seed": [3407, 3408, 3409, 3410],
            "epochs": [10] * 4,
            "batch_size": [32] * 4,
            "run_id": ["a", "b", "c", "d"],
            "protocol_signature": ["p"] * 4,
            "data_fingerprint": ["d"] * 4,
        }
    )

    summary = aggregate_runs(runs, [3407, 3408, 3409])
    coverage_frame = coverage(plan, runs)

    assert summary.loc[0, "complete"]
    assert summary.loc[0, "MSE_mean"] == 0.3
    assert summary.loc[0, "MSE_std"] == 0.1
    assert coverage_frame["status"].tolist() == ["complete"] * 3 + ["pending"]


def test_runs_with_a_different_training_protocol_are_excluded():
    runs = pd.DataFrame(
        {
            "model": ["M"],
            "dataset": ["D"],
            "output_length": [96],
            "seed": [3407],
            "epochs": [100],
            "batch_size": [64],
            "metric_scale": ["normalized"],
            "protocol_signature": ["old-protocol"],
            "data_fingerprint": ["d"],
            "run_id": ["old"],
            "MAE": [0.1],
            "MSE": [0.2],
            "RMSE": [0.3],
        }
    )
    plan = pd.DataFrame(
        {
            "model": ["M"],
            "dataset": ["D"],
            "horizon": [96],
            "seed": [3407],
            "epochs": [20],
            "batch_size": [32],
            "run_id": ["new"],
            "protocol_signature": ["new-protocol"],
            "data_fingerprint": ["d"],
        }
    )

    assert filter_runs_to_plan(runs, plan).empty
