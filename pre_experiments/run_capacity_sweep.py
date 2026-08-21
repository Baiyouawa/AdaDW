#!/usr/bin/env python3
"""Plan or execute RAW/depth/width/joint Backbone runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Baselines"))

from adawd_preexp.capacity import build_model, plan_sweep, timestamp_sizes
from adawd_preexp.catalog import load_preexperiment_config, resolve_dataset
from adawd_preexp.efficiency import measure_efficiency


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def execute(run, gpu: str | None) -> None:
    try:
        import torch
        from basicts import BasicTSLauncher
        from basicts.configs import BasicTSForecastingConfig
        from basicts.runners.callback import EarlyStopping
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires the dependencies in `pip install -e .[training]`") from exc

    experiment = load_preexperiment_config()
    training = experiment["training"]
    efficiency_config = experiment["efficiency"]
    _, dataset = resolve_dataset(run.dataset)
    data_path = PROJECT_ROOT / "datasets" / "processed" / run.dataset
    if not (data_path / "train_data.npy").is_file():
        raise FileNotFoundError(f"Prepare {run.dataset} first; missing {data_path / 'train_data.npy'}")

    run_dir = PROJECT_ROOT / "pre_experiments" / "results" / "runs" / run.run_id
    checkpoint_dir = run_dir / "checkpoint"
    manifest = run.to_dict()
    manifest.update(status="running", checkpoint_dir=str(checkpoint_dir))
    _write_json(run_dir / "manifest.json", manifest)

    model_class, model_config, use_timestamps = build_model(run)
    benchmark_device = "cuda:0" if gpu is not None and torch.cuda.is_available() else "cpu"
    benchmark_model = model_class(model_config)
    static_efficiency = measure_efficiency(
        benchmark_model,
        input_length=run.input_length,
        num_features=dataset["expected_channels"],
        batch_size=efficiency_config["benchmark_batch_size"],
        device=benchmark_device,
        warmup_iterations=efficiency_config["warmup_iterations"],
        timed_iterations=efficiency_config["timed_iterations"],
        use_timestamps=use_timestamps,
        timestamp_sizes=timestamp_sizes(run.dataset) if use_timestamps else None,
    )
    del benchmark_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = BasicTSForecastingConfig(
        model=model_class,
        model_config=model_config,
        dataset_name=run.dataset,
        data_file_path=str(data_path),
        input_len=run.input_length,
        output_len=run.output_length,
        use_timestamps=use_timestamps,
        gpus=gpu,
        num_epochs=training["epochs"],
        batch_size=training["batch_size"],
        optimizer_params={"lr": training["learning_rate"], "weight_decay": training["weight_decay"]},
        metrics=training["metrics"],
        callbacks=[EarlyStopping(patience=training["early_stopping_patience"])],
        seed=run.seed,
        save_results=training["save_predictions"],
        eval_after_train=True,
        test_interval=training["epochs"] + 1,
        ckpt_save_dir=str(checkpoint_dir),
        train_data_num_workers=4,
        val_data_num_workers=4,
        test_data_num_workers=4,
    )
    started = time.perf_counter()
    try:
        BasicTSLauncher.launch_training(config)
    except BaseException:
        manifest.update(status="failed", efficiency=static_efficiency)
        _write_json(run_dir / "manifest.json", manifest)
        raise
    training_seconds = time.perf_counter() - started
    static_efficiency["training_wall_seconds"] = training_seconds
    static_efficiency["training_peak_cuda_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() and gpu is not None else None
    )
    metrics_path = checkpoint_dir / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    manifest.update(status="complete", metrics=metrics, efficiency=static_efficiency)
    _write_json(run_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--axis", choices=["raw", "depth", "width", "joint"], required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--all", action="store_true", help="Execute every planned run sequentially")
    args = parser.parse_args()
    runs = plan_sweep(args.model, args.dataset, args.axis, args.horizon, args.seeds)
    if args.dry_run:
        print(json.dumps([run.to_dict() for run in runs], indent=2))
        return
    if args.all == (args.run_index is not None):
        raise SystemExit("Choose exactly one of --run-index or --all (use --dry-run to inspect the plan)")
    selected = runs if args.all else [runs[args.run_index]]
    for run in selected:
        print(f"running {run.run_id}")
        execute(run, args.gpu)


if __name__ == "__main__":
    main()

