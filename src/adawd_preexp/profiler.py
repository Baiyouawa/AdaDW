"""Bounded Structural Profiler used by the AdaWD pre-experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt

from .data import Segment


@dataclass(frozen=True)
class ProfilerConfig:
    window_size: int = 96
    stride: int = 24
    ar_order: int = 4
    ar_forecast_points: int = 8
    ar_ridge: float = 1e-3
    peak_saturation_count: int = 8
    peak_prominence_ratio: float = 0.05
    peak_min_distance: int = 2
    spectral_median_kernel: int = 5
    frequency_bands: int = 6
    epsilon: float = 1e-8
    max_windows_per_segment: int | None = 512
    max_profile_channels: int | None = None

    def validate(self) -> None:
        if self.window_size < 16 or self.window_size % 2:
            raise ValueError("window_size must be an even integer >= 16")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.ar_order < 1 or self.ar_order >= self.window_size // 2:
            raise ValueError("ar_order must be smaller than a half-window")
        if self.frequency_bands < 2:
            raise ValueError("frequency_bands must be >= 2")


def _interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    x = np.arange(len(output))
    finite = np.isfinite(output)
    if finite.all():
        return output
    if not finite.any():
        return np.zeros_like(output)
    output[~finite] = np.interp(x[~finite], x[finite], output[finite])
    return output


def _mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _robust_scale(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = _interpolate_nonfinite(values)
    median = np.median(values)
    scale = 1.4826 * _mad(values)
    return (values - median) / (scale + epsilon)


def _detrend(values: np.ndarray) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, len(values))
    design = np.column_stack([np.ones(len(values)), x])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _normalized_spectrum(values: np.ndarray, epsilon: float) -> np.ndarray:
    power = np.abs(np.fft.rfft(values)) ** 2
    power = power[1:] if len(power) > 1 else power
    total = power.sum()
    if total <= epsilon:
        return np.zeros(max(len(power), 1), dtype=np.float64)
    return power / total


def _js_divergence(left: np.ndarray, right: np.ndarray, epsilon: float) -> float:
    size = min(len(left), len(right))
    left = left[:size]
    right = right[:size]
    left = left / (left.sum() + epsilon)
    right = right / (right.sum() + epsilon)
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left * np.log2((left + epsilon) / (midpoint + epsilon)))
    kl_right = np.sum(right * np.log2((right + epsilon) / (midpoint + epsilon)))
    return float(np.clip(0.5 * (kl_left + kl_right), 0.0, 1.0))


def _change_score(values: np.ndarray, epsilon: float) -> float:
    half = len(values) // 2
    left = _interpolate_nonfinite(values[:half])
    right = _interpolate_nonfinite(values[half:])
    mu_left, mu_right = np.median(left), np.median(right)
    scale_left = 1.4826 * _mad(left)
    scale_right = 1.4826 * _mad(right)
    location_delta = abs(mu_left - mu_right) / (scale_left + scale_right + epsilon)
    scale_delta = abs(np.log((scale_left + epsilon) / (scale_right + epsilon)))
    return float(1.0 - np.exp(-(location_delta + scale_delta)))


def _spectral_drift(values: np.ndarray, epsilon: float) -> float:
    residual = _detrend(_robust_scale(values, epsilon))
    half = len(residual) // 2
    return _js_divergence(
        _normalized_spectrum(residual[:half], epsilon),
        _normalized_spectrum(residual[half:], epsilon),
        epsilon,
    )


def _ar_surprise(values: np.ndarray, config: ProfilerConfig) -> float:
    series = _robust_scale(values, config.epsilon)
    half = len(series) // 2
    left = series[:half]
    right = series[half:]
    order = min(config.ar_order, half - 2)
    x_rows, targets = [], []
    for index in range(order, len(left)):
        x_rows.append([left[index - lag] for lag in range(1, order + 1)])
        targets.append(left[index])
    design = np.asarray(x_rows)
    targets_array = np.asarray(targets)
    regularizer = config.ar_ridge * np.eye(order)
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ targets_array)
    history = list(left)
    forecast_count = min(config.ar_forecast_points, len(right))
    predictions = []
    for _ in range(forecast_count):
        features = np.asarray([history[-lag] for lag in range(1, order + 1)])
        prediction = float(features @ coefficients)
        predictions.append(prediction)
        history.append(prediction)
    truth = right[:forecast_count]
    ar_error = float(np.sum((truth - predictions) ** 2))
    baseline_error = float(np.sum((truth - np.median(left)) ** 2))
    ratio = ar_error / (baseline_error + config.epsilon)
    return float(ratio / (1.0 + ratio))


def _peak_score(residual: np.ndarray, config: ProfilerConfig) -> float:
    spectrum = _normalized_spectrum(residual, config.epsilon)
    if len(spectrum) < 3 or np.max(spectrum) <= config.epsilon:
        return 0.0
    kernel = min(config.spectral_median_kernel, len(spectrum) if len(spectrum) % 2 else len(spectrum) - 1)
    kernel = max(kernel, 1)
    baseline = medfilt(spectrum, kernel_size=kernel)
    excess = np.maximum(spectrum - baseline, 0.0)
    threshold = config.peak_prominence_ratio * np.max(spectrum)
    peaks, _ = find_peaks(
        excess,
        prominence=threshold,
        distance=config.peak_min_distance,
    )
    return float(min(1.0, len(peaks) / config.peak_saturation_count))


def _band_entropy(residual: np.ndarray, config: ProfilerConfig) -> float:
    spectrum = _normalized_spectrum(residual, config.epsilon)
    if len(spectrum) <= 1:
        return 0.0
    raw_edges = np.geomspace(1, len(spectrum) + 1, config.frequency_bands + 1)
    edges = np.rint(raw_edges).astype(int) - 1
    edges[0], edges[-1] = 0, len(spectrum)
    edges = np.maximum.accumulate(edges)
    energies = np.asarray([spectrum[edges[i] : edges[i + 1]].sum() for i in range(config.frequency_bands)])
    energies = energies / (energies.sum() + config.epsilon)
    entropy = -np.sum(energies * np.log(energies + config.epsilon))
    return float(np.clip(entropy / np.log(config.frequency_bands), 0.0, 1.0))


def _channel_effective_rank(window: np.ndarray, epsilon: float) -> Tuple[float, float]:
    channels = window.shape[1]
    if channels == 1:
        return 1.0, 0.0
    residuals = np.column_stack(
        [_detrend(_robust_scale(window[:, channel], epsilon)) for channel in range(channels)]
    )
    singular_values = np.linalg.svd(residuals, compute_uv=False, full_matrices=False)
    covariance_eigenvalues = singular_values**2 / max(len(window) - 1, 1)
    probabilities = covariance_eigenvalues / (covariance_eigenvalues.sum() + epsilon)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + epsilon))))
    score = (effective_rank - 1.0) / (channels - 1.0 + epsilon)
    return effective_rank, float(np.clip(score, 0.0, 1.0))


def profile_window(
    window: np.ndarray,
    config: ProfilerConfig,
    channels: Sequence[int] | None = None,
) -> List[Dict[str, float]]:
    """Profile one [time, channel] support and return one record per selected channel."""

    config.validate()
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window[:, None]
    if window.ndim != 2 or len(window) != config.window_size:
        raise ValueError(f"Expected window shape [{config.window_size}, C], got {window.shape}")
    channel_ids = list(range(window.shape[1])) if channels is None else list(channels)
    if not channel_ids or min(channel_ids) < 0 or max(channel_ids) >= window.shape[1]:
        raise ValueError("channels must contain valid indices for the supplied window")
    analysis_window = window[:, channel_ids]
    effective_rank, channel_score = _channel_effective_rank(analysis_window, config.epsilon)
    records: List[Dict[str, float]] = []
    for local_channel, channel in enumerate(channel_ids):
        values = analysis_window[:, local_channel]
        residual = _detrend(_robust_scale(values, config.epsilon))
        u_change = _change_score(values, config.epsilon)
        u_spectral = _spectral_drift(values, config.epsilon)
        u_surprise = _ar_surprise(values, config)
        m_peak = _peak_score(residual, config)
        m_band = _band_entropy(residual, config)
        if analysis_window.shape[1] == 1:
            pattern_diversity = 0.5 * (m_peak + m_band)
        else:
            pattern_diversity = (m_peak + m_band + channel_score) / 3.0
        records.append(
            {
                "channel": int(channel),
                "u_change": u_change,
                "u_spectral": u_spectral,
                "u_surprise": u_surprise,
                "m_peak": m_peak,
                "m_band": m_band,
                "channel_effective_rank": effective_rank,
                "m_channel": channel_score,
                "U": (u_change + u_spectral + u_surprise) / 3.0,
                "M": pattern_diversity,
            }
        )
    return records


def _limited_indices(count: int, limit: int | None) -> np.ndarray:
    if limit is None or count <= limit:
        return np.arange(count, dtype=int)
    return np.unique(np.rint(np.linspace(0, count - 1, limit)).astype(int))


def profile_segments(
    dataset_name: str,
    segments: Iterable[Segment],
    config: ProfilerConfig,
) -> pd.DataFrame:
    """Compute rolling profiles without crossing independent segment boundaries."""

    config.validate()
    output: List[Dict[str, object]] = []
    for segment_id, values in segments:
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        if len(values) < config.window_size:
            continue
        starts = np.arange(0, len(values) - config.window_size + 1, config.stride)
        starts = starts[_limited_indices(len(starts), config.max_windows_per_segment)]
        channels = _limited_indices(values.shape[1], config.max_profile_channels)
        for start in starts:
            stop = int(start + config.window_size)
            records = profile_window(values[int(start) : stop], config, channels=channels)
            for record in records:
                channel = int(record["channel"])
                record.update(
                    {
                        "dataset": dataset_name,
                        "segment": segment_id,
                        "window_start": int(start),
                        "window_stop": stop,
                        "unit_id": f"{segment_id}:{int(start)}:{channel}",
                    }
                )
                output.append(record)
    if not output:
        raise ValueError(f"No valid length-{config.window_size} windows for {dataset_name}")
    frame = pd.DataFrame(output)
    for score in ("U", "M"):
        lower = frame[score].quantile(1.0 / 3.0)
        upper = frame[score].quantile(2.0 / 3.0)
        frame[f"{score}_bucket"] = np.select(
            [frame[score] <= lower, frame[score] >= upper],
            ["low", "high"],
            default="mid",
        )
    return frame


def summarize_profiles(frame: pd.DataFrame, config: ProfilerConfig) -> Dict[str, object]:
    descriptors = [
        "u_change",
        "u_spectral",
        "u_surprise",
        "m_peak",
        "m_band",
        "m_channel",
        "U",
        "M",
    ]
    summary: Dict[str, object] = {
        "dataset": str(frame["dataset"].iloc[0]),
        "num_profiled_units": int(len(frame)),
        "num_windows": int(frame[["segment", "window_start"]].drop_duplicates().shape[0]),
        "num_profiled_channels": int(frame["channel"].nunique()),
        "config": asdict(config),
        "scores": {},
        "spearman_U_M": float(frame[["U", "M"]].corr(method="spearman").iloc[0, 1]),
    }
    for descriptor in descriptors:
        values = frame[descriptor]
        summary["scores"][descriptor] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p10": float(values.quantile(0.1)),
            "p50": float(values.quantile(0.5)),
            "p90": float(values.quantile(0.9)),
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        }
    return summary
