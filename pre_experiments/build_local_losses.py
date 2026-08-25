#!/usr/bin/env python3
"""Convert saved predictions into aligned per-local-unit loss tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_preexperiment_config, resolve_dataset
from adawd_preexp.profiler import ProfilerConfig, profile_window


def _channel_ids(count: int, limit: int | None) -> np.ndarray:
    if limit is None or count <= limit:
        return np.arange(count, dtype=int)
    return np.unique(np.rint(np.linspace(0, count - 1, limit)).astype(int))


def _local_profiles(
    inputs: np.ndarray,
    dataset_name: str,
    sample_stride: int,
    max_samples: int | None,
) -> pd.DataFrame:
    _, entry = resolve_dataset(dataset_name)
    base = load_preexperiment_config()["profiler"]
    config = ProfilerConfig(
        window_size=inputs.shape[1],
        stride=inputs.shape[1],
        ar_order=base["ar_order"],
        ar_forecast_points=base["ar_forecast_points"],
        ar_ridge=base["ar_ridge"],
        peak_saturation_count=base["peak_saturation_count"],
        peak_prominence_ratio=base["peak_prominence_ratio"],
        peak_min_distance=base["peak_min_distance"],
        spectral_median_kernel=base["spectral_median_kernel"],
        frequency_bands=base["frequency_bands"],
        epsilon=base["epsilon"],
        max_windows_per_segment=None,
        max_profile_channels=entry.get("max_profile_channels"),
    )
    sample_indices = np.arange(0, len(inputs), sample_stride)
    if max_samples is not None:
        sample_indices = sample_indices[:max_samples]
    channels = _channel_ids(inputs.shape[2], config.max_profile_channels)
    rows = []
    for sample_index in sample_indices:
        for record in profile_window(np.asarray(inputs[sample_index]), config, channels):
            channel = int(record["channel"])
            record.update(
                dataset=dataset_name,
                segment="test",
                sample_index=int(sample_index),
                window_start=int(sample_index),
                unit_id=f"test:{int(sample_index)}:{channel}",
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _load_results(
    run_dir: Path, manifest: Dict[str, object]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_dir = Path(str(manifest.get("checkpoint_dir", run_dir / "checkpoint")))
    result_dir = checkpoint_dir / "test_results"
    paths = [result_dir / name for name in ("inputs.npy", "prediction.npy", "targets.npy")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved arrays for {run_dir.name}: {missing}")
    return tuple(np.load(path, mmap_mode="r") for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "pre_experiments" / "results" / "runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "pre_experiments" / "results" / "local_losses.csv")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.sample_stride < 1:
        raise ValueError("sample-stride must be positive")

    cached_profiles: Dict[Tuple[str, int, int, int | None], pd.DataFrame] = {}
    all_rows = []
    manifests = sorted(args.runs_root.rglob("manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        run_dir = manifest_path.parent
        inputs, predictions, targets = _load_results(run_dir, manifest)
        if predictions.shape != targets.shape:
            raise ValueError(f"Prediction/target shape mismatch in {run_dir.name}")
        key = (manifest["dataset"], inputs.shape[1], args.sample_stride, args.max_samples)
        if key not in cached_profiles:
            cached_profiles[key] = _local_profiles(
                inputs,
                manifest["dataset"],
                args.sample_stride,
                args.max_samples,
            )
        profiles = cached_profiles[key].copy()
        sample_index = profiles["sample_index"].to_numpy(dtype=int)
        channel = profiles["channel"].to_numpy(dtype=int)
        error = predictions[sample_index, :, channel] - targets[sample_index, :, channel]
        profiles["loss_mae"] = np.mean(np.abs(error), axis=1)
        profiles["loss_mse"] = np.mean(error**2, axis=1)
        for field in (
            "model",
            "seed",
            "axis",
            "depth",
            "width_group",
            "width",
            "coupled_width",
        ):
            profiles[field] = manifest.get(field)
        profiles["horizon"] = manifest["output_length"]
        profiles["run_id"] = manifest["run_id"]
        all_rows.append(profiles)
    if not all_rows:
        raise RuntimeError(f"No completed runs with saved predictions under {args.runs_root}")
    output = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
