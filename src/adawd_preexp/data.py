"""Dataset loading and BasicTS-compatible preprocessing."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .catalog import PROJECT_ROOT, find_raw_files, resolve_dataset


Segment = Tuple[str, np.ndarray]


def _read_forecasting_csv(path: Path, date_column: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"{path} has no required timestamp column '{date_column}'")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop(date_column), errors="raise"))
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{path} contains no signal columns")
    return values, timestamps


def _validate_shape(name: str, values: np.ndarray, entry: Dict[str, Any], strict: bool) -> None:
    issues = []
    expected_channels = entry.get("expected_channels")
    expected_steps = entry.get("expected_time_steps")
    if expected_channels is not None and values.shape[-1] != expected_channels:
        issues.append(f"channels={values.shape[-1]} (expected {expected_channels})")
    if values.ndim == 2 and expected_steps is not None and values.shape[0] != expected_steps:
        issues.append(f"time_steps={values.shape[0]} (expected {expected_steps})")
    if issues:
        message = f"{name} shape differs from the catalog: " + ", ".join(issues)
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)


def _timestamp_features(index: pd.DatetimeIndex) -> np.ndarray:
    midnight = index.normalize()
    time_of_day = (index - midnight).total_seconds().to_numpy() / 86400.0
    return np.column_stack(
        [
            time_of_day,
            index.dayofweek.to_numpy() / 7.0,
            (index.day.to_numpy() - 1) / 31.0,
            (index.dayofyear.to_numpy() - 1) / 366.0,
        ]
    ).astype(np.float32)


def _split_bounds(length: int, ratios: Sequence[float]) -> Tuple[int, int]:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Invalid train/validation/test ratios: {ratios}")
    train_end = int(length * ratios[0])
    val_end = train_end + int(length * ratios[1])
    return train_end, val_end


def prepare_dataset(
    dataset_name: str,
    output_root: Path | None = None,
    strict_shape: bool = False,
) -> Path:
    """Prepare one registered dataset without downloading or truncating it."""

    canonical, entry = resolve_dataset(dataset_name)
    output_root = output_root or PROJECT_ROOT / "datasets" / "processed"
    output_dir = output_root / canonical
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = find_raw_files(canonical)

    if entry["task"] == "classification":
        arrays = [np.loadtxt(path, dtype=np.float32) for path in raw_files]
        joined = np.concatenate(arrays, axis=0)
        labels = joined[:, 0]
        samples = joined[:, 1:, None]
        if samples.shape[0] != entry.get("expected_samples"):
            message = f"{canonical} has {samples.shape[0]} samples; expected {entry.get('expected_samples')}"
            if strict_shape:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        np.save(output_dir / "samples.npy", samples)
        np.save(output_dir / "labels.npy", labels)
        metadata = {
            "name": canonical,
            "task": entry["task"],
            "shape": list(samples.shape),
            "source_files": [str(path) for path in raw_files],
            "forecasting_status": entry["forecasting_status"],
        }
    else:
        values, index = _read_forecasting_csv(raw_files[0], entry["date_column"])
        _validate_shape(canonical, values, entry, strict_shape)
        timestamp_values = _timestamp_features(index)
        train_end, val_end = _split_bounds(len(values), entry["split"])
        context = int(entry["forecast_input_length"])
        slices = {
            "train": slice(0, train_end),
            "val": slice(train_end - context, val_end),
            "test": slice(val_end - context, len(values)),
        }
        for split_name, split_slice in slices.items():
            np.save(output_dir / f"{split_name}_data.npy", values[split_slice].astype(np.float32))
            np.save(
                output_dir / f"{split_name}_timestamps.npy",
                timestamp_values[split_slice],
            )
        metadata = {
            "name": canonical,
            "task": entry["task"],
            "domain": entry["domain"],
            "frequency_minutes": entry["frequency_minutes"],
            "shape": list(values.shape),
            "split": entry["split"],
            "split_lengths": {name: split_slice.stop - split_slice.start for name, split_slice in slices.items()},
            "validation_test_context": context,
            "timestamp_features": ["time_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "source_files": [str(raw_files[0])],
            "retains_full_series": True,
        }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return output_dir


def load_profile_segments(dataset_name: str, prefer_raw: bool = True) -> List[Segment]:
    """Load independent segments; windows never cross dataset split boundaries."""

    canonical, entry = resolve_dataset(dataset_name)
    processed_dir = PROJECT_ROOT / "datasets" / "processed" / canonical

    if prefer_raw:
        try:
            raw_files = find_raw_files(canonical)
        except FileNotFoundError:
            raw_files = ()
        if raw_files and entry["task"] == "forecasting":
            values, _ = _read_forecasting_csv(raw_files[0], entry["date_column"])
            return [("raw", values)]
        if raw_files and entry["task"] == "classification":
            arrays = [np.loadtxt(path, dtype=np.float64) for path in raw_files]
            joined = np.concatenate(arrays, axis=0)
            return [(f"sample_{idx}", row[1:, None]) for idx, row in enumerate(joined)]

    if entry["task"] == "classification":
        samples_path = processed_dir / "samples.npy"
        if not samples_path.is_file():
            raise FileNotFoundError(f"Prepare {canonical} first: missing {samples_path}")
        samples = np.load(samples_path, mmap_mode="r")
        return [(f"sample_{idx}", np.asarray(sample)) for idx, sample in enumerate(samples)]

    segments: List[Segment] = []
    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}_data.npy"
        if path.is_file():
            segments.append((split, np.load(path, mmap_mode="r")))
    if not segments:
        raise FileNotFoundError(
            f"No raw or processed data found for {canonical}. See datasets/README.md."
        )
    return segments
