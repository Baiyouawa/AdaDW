#!/usr/bin/env python3
"""Plan or execute the three-year Backbone capacity pre-experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_backbone_registry, load_preexperiment_config, resolve_dataset
from adawd_preexp.capacity import plan_sweep


DEFAULT_MODELS = ("PatchTST", "TimeMixer", "MultiPatchFormer")


def _default_models() -> list[str]:
    configured = load_preexperiment_config().get("model_trajectory", {}).get("models")
    return list(configured or DEFAULT_MODELS)


def _validate_models(models: list[str]) -> None:
    registry = load_backbone_registry()["models"]
    if len(models) != 3:
        raise ValueError("Exactly three models are required: one each from 2023, 2024 and 2025")
    years = []
    for model in models:
        if model not in registry:
            raise KeyError(f"Unknown Backbone '{model}'")
        if registry[model].get("status") != "available":
            raise RuntimeError(f"Backbone '{model}' is not available")
        years.append(int(registry[model]["year"]))
    if sorted(years) != [2023, 2024, 2025]:
        raise ValueError(f"Models must cover years 2023, 2024 and 2025; got {years}")


def build_plan(models: list[str], dataset: str, horizon: int, seeds: list[int]) -> list[dict[str, object]]:
    _validate_models(models)
    plan = []
    for model in models:
        for axis in ("depth", "width"):
            plan.extend(run.to_dict() for run in plan_sweep(model, dataset, axis, horizon, seeds))
    return plan


def _is_complete(output_root: Path, run_id: str) -> bool:
    manifest_path = output_root / run_id / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete":
        return False
    checkpoint_value = manifest.get("checkpoint_dir")
    if not checkpoint_value:
        return False
    checkpoint_dir = Path(str(checkpoint_value))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    result_dir = checkpoint_dir / "test_results"
    return all(
        (result_dir / name).is_file()
        for name in ("inputs.npy", "prediction.npy", "targets.npy")
    )


def run_capacity(
    model: str,
    axis: str,
    dataset: str,
    horizon: int,
    seeds: list[int],
    args: argparse.Namespace,
) -> None:
    runs = plan_sweep(model, dataset, axis, horizon, seeds)
    for run_index, run in enumerate(runs):
        if _is_complete(args.output_root, run.run_id):
            print(f"skip complete {run.run_id}", flush=True)
            continue
        command = [
            sys.executable,
            "pre_experiments/run_capacity_sweep.py",
            "--dataset",
            dataset,
            "--model",
            model,
            "--axis",
            axis,
            "--horizon",
            str(horizon),
            "--seeds",
            *map(str, seeds),
            "--run-index",
            str(run_index),
            "--output-root",
            str(args.output_root),
        ]
        if args.gpu is not None:
            command.extend(["--gpu", args.gpu])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        print(f"+ {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    config = load_preexperiment_config()
    trajectory_config = config["model_trajectory"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=3, default=_default_models(), metavar=("MODEL_2023", "MODEL_2024", "MODEL_2025"))
    parser.add_argument("--dataset", default=trajectory_config["dataset"])
    parser.add_argument("--horizon", type=int, default=trajectory_config["horizon"])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=trajectory_config["seeds"]
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "pre_experiments" / "results" / "model_trajectories" / "runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true", help="execute every depth and width run")
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    canonical, dataset_entry = resolve_dataset(args.dataset)
    if args.horizon not in dataset_entry["forecast_horizons"]:
        raise ValueError(f"Horizon {args.horizon} is not registered for {canonical}: {dataset_entry['forecast_horizons']}")
    plan = build_plan(args.models, canonical, args.horizon, args.seeds)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if not args.all:
        raise SystemExit("Training is opt-in; pass --all, or use --dry-run to inspect the plan")
    for model in args.models:
        for axis in ("depth", "width"):
            run_capacity(model, axis, canonical, args.horizon, args.seeds, args)


if __name__ == "__main__":
    main()
