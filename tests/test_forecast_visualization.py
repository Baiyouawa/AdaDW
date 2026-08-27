import json

import numpy as np
import pandas as pd

from adawd_preexp.forecast_visualization import (
    build_baseline_comparison_visualizations,
    export_run_forecast_visualization,
    select_channel_ids,
    select_sample_index,
)


def test_visualization_selection_is_deterministic_and_value_independent():
    assert select_sample_index(11, 0.5) == 5
    assert select_sample_index(10, 0.5) == 4
    assert select_channel_ids(7, 4) == [0, 2, 4, 6]
    assert select_channel_ids(2, 4) == [0, 1]


def test_export_retains_original_scale_slice_and_png(tmp_path):
    result_dir = tmp_path / "checkpoint" / "test_results"
    processed_dir = tmp_path / "processed"
    run_dir = tmp_path / "run"
    result_dir.mkdir(parents=True)
    processed_dir.mkdir()
    run_dir.mkdir()

    train = np.array([[10.0, 100.0], [12.0, 104.0], [14.0, 108.0]], dtype=np.float32)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    inputs_original = np.array(
        [
            [[10.0, 100.0], [12.0, 104.0]],
            [[12.0, 104.0], [14.0, 108.0]],
            [[14.0, 108.0], [16.0, 112.0]],
        ],
        dtype=np.float32,
    )
    targets_original = inputs_original + np.array([[[4.0, 8.0]]], dtype=np.float32)
    prediction_original = targets_original + 1.0
    for name, values in {
        "inputs": (inputs_original - mean) / std,
        "targets": (targets_original - mean) / std,
        "prediction": (prediction_original - mean) / std,
    }.items():
        np.save(result_dir / f"{name}.npy", values)
    np.save(processed_dir / "train_data.npy", train)
    np.save(
        processed_dir / "test_time_index.npy",
        pd.date_range("2026-01-01", periods=6, freq="h").to_numpy(),
    )
    (processed_dir / "meta.json").write_text(
        json.dumps({"signal_columns": ["load", "temperature"]}), encoding="utf-8"
    )

    metadata = export_run_forecast_visualization(
        result_dir=result_dir,
        processed_dir=processed_dir,
        run_dir=run_dir,
        dataset="Tiny",
        model="Model",
        horizon=2,
        seed=3407,
        input_length=2,
        output_length=2,
        metric_scale="normalized",
        sample_position=0.5,
        sample_index=None,
        max_channels=2,
    )

    frame = pd.read_csv(run_dir / metadata["slice_csv"])
    forecast = frame[(frame["phase"] == "forecast") & (frame["channel"] == 0)]
    assert metadata["sample_index"] == 1
    assert np.allclose(forecast["observed"], targets_original[1, :, 0])
    assert np.allclose(forecast["prediction"], prediction_original[1, :, 0])
    assert (run_dir / metadata["plot_png"]).is_file()


def test_baseline_comparison_builds_seed_aggregated_index(tmp_path):
    runs_root = tmp_path / "runs"
    rows = []
    for model, shift in (("A", 0.0), ("B", 1.0)):
        run_id = f"{model}-run"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "dataset": ["Tiny"] * 4,
                "model": [model] * 4,
                "horizon": [2] * 4,
                "seed": [1] * 4,
                "sample_index": [3] * 4,
                "sample_position": [0.5] * 4,
                "channel": [0] * 4,
                "channel_name": ["load"] * 4,
                "scale": ["original"] * 4,
                "phase": ["history", "history", "forecast", "forecast"],
                "step": [-2, -1, 1, 2],
                "timestamp": pd.date_range("2026-01-01", periods=4, freq="h").astype(str),
                "observed": [1.0, 2.0, 3.0, 4.0],
                "prediction": [np.nan, np.nan, 3.0 + shift, 4.0 + shift],
            }
        )
        frame.to_csv(run_dir / "forecast_slice.csv", index=False)
        rows.append(
            {
                "dataset": "Tiny",
                "output_length": 2,
                "model": model,
                "seed": 1,
                "run_id": run_id,
                "visualization_slice": "forecast_slice.csv",
            }
        )

    index = build_baseline_comparison_visualizations(
        runs=pd.DataFrame(rows),
        runs_root=runs_root,
        output_dir=tmp_path / "summary",
        model_order=["A", "B"],
        expected_seeds=[1],
    )

    assert index.loc[0, "complete"]
    assert index.loc[0, "model_count"] == 2
    assert (tmp_path / "summary" / index.loc[0, "plot_png"]).is_file()
