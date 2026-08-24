#!/usr/bin/env python3
"""Run all eight Backbones on all nine forecasting datasets, resumably."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_backbone_registry, load_dataset_catalog


DEFAULT_CONFIG = PROJECT_ROOT / "pre_experiments" / "benchmark_config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "pre_experiments" / "results" / "forecasting_raw"


@dataclass(frozen=True)
class BenchmarkRun:
    index: int
    model: str
    dataset: str
    horizon: int
    seed: int
    epochs: int
    batch_size: int


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required_models = [
        name
        for name, entry in load_backbone_registry()["models"].items()
        if entry["status"] == "available"
    ]
    if list(config["models"]) != required_models:
        raise ValueError(
            "benchmark model order must match available registry models: "
            f"expected={required_models}, configured={list(config['models'])}"
        )
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("benchmark seeds must be unique")
    return config


def build_plan(config: dict, smoke: bool = False) -> list[BenchmarkRun]:
    catalog = load_dataset_catalog()
    caps = config.get("dataset_batch_caps", {})
    plan = []
    for model, model_config in config["models"].items():
        for dataset, dataset_config in catalog.items():
            batch_size = min(
                int(model_config["batch_size"]), int(caps.get(dataset, model_config["batch_size"]))
            )
            horizons = dataset_config["forecast_horizons"][:1] if smoke else dataset_config["forecast_horizons"]
            seeds = config["seeds"][:1] if smoke else config["seeds"]
            epochs = 1 if smoke else int(model_config["epochs"])
            for horizon in horizons:
                for seed in seeds:
                    plan.append(
                        BenchmarkRun(
                            index=len(plan) + 1,
                            model=model,
                            dataset=dataset,
                            horizon=int(horizon),
                            seed=int(seed),
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )
    return plan


def manifest_path(output_root: Path, run: BenchmarkRun) -> Path | None:
    prefix = f"{run.model}__{run.dataset}__h{run.horizon}__raw"
    matches = list(output_root.glob(f"{prefix}__d*__wg*__s{run.seed}/manifest.json"))
    return matches[0] if len(matches) == 1 else None


def is_complete(output_root: Path, run: BenchmarkRun, config: dict) -> bool:
    path = manifest_path(output_root, run)
    if path is None:
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "complete"
        and manifest.get("epochs") == run.epochs
        and manifest.get("batch_size") == run.batch_size
        and manifest.get("metric_scale") == config["metric_scale"]
        and manifest.get("artifact_policy") == config["artifact_policy"]
    )


def write_plan(plan: list[BenchmarkRun], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "plan.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(run) for run in plan)
    return path


def training_command(run: BenchmarkRun, config: dict, output_root: Path, gpu: str) -> list[str]:
    return [
        sys.executable,
        "pre_experiments/run_capacity_sweep.py",
        "--dataset", run.dataset,
        "--model", run.model,
        "--axis", "raw",
        "--horizon", str(run.horizon),
        "--seeds", str(run.seed),
        "--gpu", gpu,
        "--run-index", "0",
        "--epochs", str(run.epochs),
        "--batch-size", str(run.batch_size),
        "--metric-scale", config["metric_scale"],
        "--artifact-policy", config["artifact_policy"],
        "--output-root", str(output_root),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one epoch at the shortest horizon and first seed for each model/dataset",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failed runs and continue checking the remaining plan",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--stop-index", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    plan = build_plan(config, smoke=args.smoke)
    write_plan(plan, args.output_root)
    stop_index = args.stop_index or len(plan)
    selected = [run for run in plan if args.start_index <= run.index <= stop_index]
    print(f"planned_runs={len(plan)} selected_runs={len(selected)}")
    print(f"plan={args.output_root / 'plan.csv'}")
    if args.dry_run:
        return

    for position, run in enumerate(selected, start=1):
        label = (
            f"{run.model} {run.dataset} h{run.horizon} s{run.seed} "
            f"epochs={run.epochs} batch={run.batch_size}"
        )
        if not args.no_resume and is_complete(args.output_root, run, config):
            print(f"[{position}/{len(selected)}] skip complete: {label}", flush=True)
            continue
        print(f"[{position}/{len(selected)}] run: {label}", flush=True)
        try:
            subprocess.run(
                training_command(run, config, args.output_root, args.gpu),
                cwd=PROJECT_ROOT,
                check=True,
            )
        except subprocess.CalledProcessError:
            if not args.continue_on_error:
                raise
            print(f"[{position}/{len(selected)}] FAILED: {label}", flush=True)

    subprocess.run(
        [
            sys.executable,
            "pre_experiments/summarize_forecasting_benchmarks.py",
            "--runs-root", str(args.output_root),
            "--plan", str(args.output_root / "plan.csv"),
            "--output-dir", str(args.output_root / "summary"),
            "--seeds", *(
                str(seed) for seed in (config["seeds"][:1] if args.smoke else config["seeds"])
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
