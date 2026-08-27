"""Capacity sweep planning and adapters for the migrated Backbones."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from .catalog import load_backbone_registry, load_preexperiment_config, resolve_dataset


TIMEFILTER_PATCH_LENGTHS = {
    "ETTh1": 2,
    "ETTh2": 4,
    "ETTm1": 8,
    "ETTm2": 16,
    "Weather": 48,
    "Electricity": 32,
    "ILI": 4,
    "ExchangeRate": 4,
    "Traffic": 96,
}


@dataclass(frozen=True)
class SweepRun:
    axis: str
    model: str
    dataset: str
    input_length: int
    output_length: int
    depth: int
    width_group: int
    width: int
    coupled_width: int | None
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model}__{self.dataset}__h{self.output_length}__{self.axis}"
            f"__d{self.depth}__wg{self.width_group}__s{self.seed}"
        )

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["run_id"] = self.run_id
        return output


def _model_entry(model_name: str) -> Dict[str, Any]:
    models = load_backbone_registry()["models"]
    if model_name not in models:
        raise KeyError(f"Unknown Backbone '{model_name}'. Available: {', '.join(models)}")
    entry = models[model_name]
    if entry["status"] != "available":
        raise RuntimeError(f"{model_name} is unavailable: {entry['reason']}")
    return entry


def capacity_candidates(model_name: str, axis: str) -> List[int]:
    """Return the model-specific candidates for one capacity axis."""

    entry = _model_entry(model_name)
    experiment = load_preexperiment_config()["capacity"]
    if axis == "depth":
        values = entry.get("depth_candidates", experiment["depth_candidates"])
    elif axis == "width":
        values = entry.get(
            "width_group_candidates", experiment["width_group_candidates"]
        )
    else:
        raise ValueError("axis must be depth or width")
    candidates = [int(value) for value in values]
    if len(candidates) != len(set(candidates)) or any(value < 1 for value in candidates):
        raise ValueError(
            f"{model_name} has invalid {axis} candidates: {candidates}"
        )
    return candidates


def plan_sweep(
    model_name: str,
    dataset_name: str,
    axis: str,
    output_length: int | None = None,
    seeds: List[int] | None = None,
) -> List[SweepRun]:
    entry = _model_entry(model_name)
    canonical, dataset = resolve_dataset(dataset_name)
    if dataset["task"] != "forecasting":
        raise RuntimeError(
            f"{canonical} is a {dataset['task']} dataset and has no approved forecasting protocol"
        )
    experiment = load_preexperiment_config()["capacity"]
    input_length = int(dataset["forecast_input_length"])
    horizons = dataset["forecast_horizons"]
    if output_length is not None:
        if output_length not in horizons:
            raise ValueError(f"Horizon {output_length} is not registered for {canonical}: {horizons}")
        horizons = [output_length]
    seeds = experiment["seeds"] if seeds is None else seeds
    raw_group = int(entry["raw_width"] // entry["width_unit"])

    if axis == "raw":
        benchmark = entry.get("benchmark_config")
        if not isinstance(benchmark, dict):
            raise ValueError(f"{model_name} has no explicit benchmark_config")
        raw_depth = int(benchmark[entry["depth_parameter"]])
        raw_width = int(benchmark[entry["width_parameter"]])
        if raw_width % int(entry["width_unit"]):
            raise ValueError(
                f"{model_name} benchmark width {raw_width} is not divisible by "
                f"width_unit {entry['width_unit']}"
            )
        raw_group = raw_width // int(entry["width_unit"])
        pairs = [(raw_depth, raw_group)]
    elif axis == "depth":
        pairs = [(depth, raw_group) for depth in capacity_candidates(model_name, "depth")]
    elif axis == "width":
        pairs = [
            (entry["raw_depth"], group)
            for group in capacity_candidates(model_name, "width")
        ]
    elif axis == "joint":
        pairs = [
            (depth, group)
            for depth in experiment["grid_depth_candidates"]
            for group in experiment["grid_width_group_candidates"]
        ]
    else:
        raise ValueError("axis must be one of raw, depth, width or joint")

    return [
        SweepRun(
            axis=axis,
            model=model_name,
            dataset=canonical,
            input_length=input_length,
            output_length=int(horizon),
            depth=int(depth),
            width_group=int(group),
            width=int(group * entry["width_unit"]),
            coupled_width=(
                int(entry["benchmark_config"][entry["coupled_width_parameter"]])
                if axis == "raw" and "coupled_width_parameter" in entry
                else int(group * entry["width_unit"] * entry["coupled_width_ratio"])
                if "coupled_width_parameter" in entry
                else None
            ),
            seed=int(seed),
        )
        for horizon in horizons
        for seed in seeds
        for depth, group in pairs
    ]


def timestamp_sizes(dataset_name: str) -> List[int]:
    _, entry = resolve_dataset(dataset_name)
    steps_per_day = max(1, round(1440 / entry["frequency_minutes"]))
    return [steps_per_day, 7, 31, 366]


def build_model(run: SweepRun) -> Tuple[type, Any, bool]:
    """Build an architecture adapter; importing torch is deferred to this call."""

    entry = _model_entry(run.model)
    module = importlib.import_module(entry["module"])
    model_class = getattr(module, entry["model_class"])
    config_class = getattr(module, entry["config_class"])
    _, dataset = resolve_dataset(run.dataset)
    kwargs: Dict[str, Any] = {
        "input_len": run.input_length,
        "output_len": run.output_length,
        "num_features": dataset["expected_channels"],
    }
    if run.axis == "raw":
        kwargs.update(entry["benchmark_config"])
    else:
        kwargs.update(entry.get("fixed_config", {}))
        kwargs[entry["depth_parameter"]] = run.depth
        kwargs[entry["width_parameter"]] = run.width
        if "coupled_width_parameter" in entry:
            kwargs[entry["coupled_width_parameter"]] = run.coupled_width
    use_timestamps = run.model == "TimesNet"
    if run.model == "TimesNet":
        kwargs.update(use_timestamps=True, timestamp_sizes=timestamp_sizes(run.dataset))
    if run.model == "TimeFilter":
        kwargs["patch_len"] = min(TIMEFILTER_PATCH_LENGTHS[run.dataset], run.input_length)
    return model_class, config_class(**kwargs), use_timestamps
