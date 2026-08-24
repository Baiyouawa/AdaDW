#!/usr/bin/env python3
"""Plan or execute RAW/depth/width/joint Backbone runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def execute(
    run,
    gpu: str | None,
    output_root: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    metric_scale: str = "normalized",
    artifact_policy: str = "full",
) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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

    output_root = output_root or PROJECT_ROOT / "pre_experiments" / "results" / "runs"
    run_dir = output_root / run.run_id
    checkpoint_dir = run_dir / "checkpoint"
    manifest = run.to_dict()
    manifest.update(status="running", checkpoint_dir=str(checkpoint_dir))
    _write_json(run_dir / "manifest.json", manifest)

    static_efficiency = None
    resolved_checkpoint_dir = None
    try:
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

        run_epochs = int(epochs if epochs is not None else training["epochs"])
        run_batch_size = int(batch_size if batch_size is not None else training["batch_size"])
        if metric_scale == "normalized":
            run_metrics = [
                metric for metric in training["metrics"] if metric in {"MAE", "MSE", "RMSE"}
            ]
        else:
            run_metrics = training["metrics"]
        save_predictions = training["save_predictions"] and artifact_policy == "full"
        config = BasicTSForecastingConfig(
            model=model_class,
            model_config=model_config,
            dataset_name=run.dataset,
            data_file_path=str(data_path),
            input_len=run.input_length,
            output_len=run.output_length,
            use_timestamps=use_timestamps,
            gpus=gpu,
            num_epochs=run_epochs,
            batch_size=run_batch_size,
            optimizer_params={"lr": training["learning_rate"], "weight_decay": training["weight_decay"]},
            metrics=run_metrics,
            callbacks=[EarlyStopping(patience=training["early_stopping_patience"])],
            seed=run.seed,
            save_results=save_predictions,
            eval_after_train=True,
            rescale=metric_scale == "original",
            deterministic=True,
            cudnn_benchmark=False,
            cudnn_determinstic=True,
            test_interval=run_epochs + 1,
            ckpt_save_dir=str(checkpoint_dir),
            train_data_num_workers=4,
            val_data_num_workers=4,
            test_data_num_workers=4,
        )
        resolved_checkpoint_dir = checkpoint_dir / config.md5
        manifest.update(
            checkpoint_dir=str(resolved_checkpoint_dir),
            epochs=run_epochs,
            batch_size=run_batch_size,
            metric_scale=metric_scale,
            artifact_policy=artifact_policy,
        )
        _write_json(run_dir / "manifest.json", manifest)

        started = time.perf_counter()
        BasicTSLauncher.launch_training(config)
        training_seconds = time.perf_counter() - started
        static_efficiency["training_wall_seconds"] = training_seconds
        static_efficiency["training_peak_cuda_bytes"] = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and gpu is not None
            else None
        )
        metrics_path = resolved_checkpoint_dir / "test_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Training finished without test metrics: {metrics_path}")
        if save_predictions:
            result_dir = resolved_checkpoint_dir / "test_results"
            missing_results = [
                str(result_dir / name)
                for name in ("inputs.npy", "prediction.npy", "targets.npy")
                if not (result_dir / name).is_file()
            ]
            if missing_results:
                raise FileNotFoundError(f"Training finished without saved results: {missing_results}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if artifact_policy == "metrics":
            shutil.rmtree(resolved_checkpoint_dir)
            manifest.update(checkpoint_dir=None, artifacts_removed=True)
    except BaseException:
        manifest.update(status="failed")
        if static_efficiency is not None:
            manifest["efficiency"] = static_efficiency
        if artifact_policy == "metrics" and resolved_checkpoint_dir is not None:
            shutil.rmtree(resolved_checkpoint_dir, ignore_errors=True)
            manifest.update(checkpoint_dir=None, artifacts_removed=True)
        _write_json(run_dir / "manifest.json", manifest)
        raise
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
    parser.add_argument("--epochs", type=int, help="override configured epochs")
    parser.add_argument("--batch-size", type=int, help="override configured batch size")
    parser.add_argument("--output-root", type=Path, help="override run output directory")
    parser.add_argument(
        "--metric-scale", choices=["normalized", "original"], default="normalized"
    )
    parser.add_argument(
        "--artifact-policy", choices=["full", "metrics"], default="full"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--all", action="store_true", help="Execute every planned run sequentially")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    runs = plan_sweep(args.model, args.dataset, args.axis, args.horizon, args.seeds)
    if args.dry_run:
        print(json.dumps([run.to_dict() for run in runs], indent=2))
        return
    if args.all == (args.run_index is not None):
        raise SystemExit("Choose exactly one of --run-index or --all (use --dry-run to inspect the plan)")
    selected = runs if args.all else [runs[args.run_index]]
    for run in selected:
        print(f"running {run.run_id}")
        execute(
            run,
            args.gpu,
            args.output_root,
            args.epochs,
            args.batch_size,
            args.metric_scale,
            args.artifact_policy,
        )


if __name__ == "__main__":
    main()
