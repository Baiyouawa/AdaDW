"""Dataset loading and BasicTS-compatible preprocessing."""

from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .catalog import PROJECT_ROOT, find_raw_files, resolve_dataset


Segment = Tuple[str, np.ndarray]


def _read_forecasting_csv(
    path: Path, date_column: str
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"{path} has no required timestamp column '{date_column}'")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop(date_column), errors="raise"))
    signal_columns = [str(column) for column in frame.columns]
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{path} contains no signal columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or infinite signal values")
    return values, timestamps, signal_columns


def _validate_time_axis(
    name: str,
    index: pd.DatetimeIndex,
    frequency_minutes: int,
    strict: bool,
) -> None:
    if index.hasnans:
        raise ValueError(f"{name} contains missing timestamps")
    if index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps are not strictly chronological")
    if len(index) < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    expected = pd.Timedelta(minutes=int(frequency_minutes))
    deltas = index[1:] - index[:-1]
    if not bool((deltas == expected).all()):
        message = f"{name} timestamp spacing differs from the catalog frequency {expected}"
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        values, index, signal_columns = _read_forecasting_csv(
            raw_files[0], entry["date_column"]
        )
        _validate_shape(canonical, values, entry, strict_shape)
        _validate_time_axis(canonical, index, entry["frequency_minutes"], strict_shape)
        timestamp_values = _timestamp_features(index)
        train_end, val_end = _split_bounds(len(values), entry["split"])
        context = int(entry["forecast_input_length"])
        max_horizon = max(int(value) for value in entry["forecast_horizons"])
        target_bounds = {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, len(values)],
        }
        for split_name, (start, stop) in target_bounds.items():
            if stop - start < max_horizon:
                raise ValueError(
                    f"{canonical} {split_name} target interval is shorter than max horizon "
                    f"{max_horizon}: {stop - start}"
                )
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
            np.save(
                output_dir / f"{split_name}_time_index.npy",
                index[split_slice].to_numpy(dtype="datetime64[ns]"),
            )
        fingerprint_payload = {
            "schema_version": 2,
            "name": canonical,
            "source_sha256": _sha256(raw_files[0]),
            "shape": list(values.shape),
            "frequency_minutes": int(entry["frequency_minutes"]),
            "input_length": context,
            "horizons": [int(value) for value in entry["forecast_horizons"]],
            "target_bounds": target_bounds,
            "storage_bounds": {
                name: [int(split_slice.start), int(split_slice.stop)]
                for name, split_slice in slices.items()
            },
        }
        metadata = {
            "name": canonical,
            "task": entry["task"],
            "domain": entry["domain"],
            "frequency_minutes": entry["frequency_minutes"],
            "shape": list(values.shape),
            "split": entry["split"],
            "target_bounds": target_bounds,
            "storage_bounds": fingerprint_payload["storage_bounds"],
            "split_lengths": {name: split_slice.stop - split_slice.start for name, split_slice in slices.items()},
            "validation_test_context": context,
            "timestamp_features": ["time_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "time_index_encoding": "datetime64[ns]",
            "signal_columns": signal_columns,
            "source_files": [str(raw_files[0])],
            "source_sha256": fingerprint_payload["source_sha256"],
            "data_fingerprint": _data_fingerprint(fingerprint_payload),
            "metadata_schema_version": 2,
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
            values, _, _ = _read_forecasting_csv(raw_files[0], entry["date_column"])
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
