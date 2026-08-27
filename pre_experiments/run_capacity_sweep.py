#!/usr/bin/env python3
"""Plan or execute RAW/depth/width/joint Backbone runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Baselines"))

from adawd_preexp.capacity import build_model, plan_sweep, timestamp_sizes
from adawd_preexp.catalog import load_preexperiment_config, resolve_dataset
from adawd_preexp.efficiency import measure_efficiency
from adawd_preexp.forecast_visualization import (
    export_run_forecast_visualization,
    select_sample_index,
)


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
    loss: str | None = None,
    target_metric: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    data_fingerprint: str | None = None,
    protocol_signature: str | None = None,
    visualize_forecast: bool = False,
    visualization_sample_position: float = 0.5,
    visualization_max_channels: int = 4,
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
    if visualize_forecast:
        required_visualization_files = [
            data_path / "meta.json",
            data_path / "test_time_index.npy",
        ]
        missing_visualization_files = [
            str(path) for path in required_visualization_files if not path.is_file()
        ]
        if missing_visualization_files:
            raise FileNotFoundError(
                "Re-prepare the dataset before forecast visualization; missing "
                f"{missing_visualization_files}"
            )
    if data_fingerprint is not None:
        meta_path = data_path / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Re-prepare {run.dataset}; missing {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_fingerprint = metadata.get("data_fingerprint")
        if actual_fingerprint is None:
            raise RuntimeError(
                f"Re-prepare {run.dataset}; {meta_path} uses legacy metadata without "
                "a data fingerprint."
            )
        if actual_fingerprint != data_fingerprint:
            raise RuntimeError(
                f"Processed data changed after planning {run.dataset}: "
                f"planned={data_fingerprint}, actual={actual_fingerprint}. Rebuild the plan."
            )

    output_root = output_root or PROJECT_ROOT / "pre_experiments" / "results" / "runs"
    run_dir = output_root / run.run_id
    checkpoint_dir = run_dir / "checkpoint"
    manifest = run.to_dict()
    manifest.update(
        status="running",
        checkpoint_dir=str(checkpoint_dir),
        protocol_signature=protocol_signature,
        data_fingerprint=data_fingerprint,
        visualization_config={
            "enabled": visualize_forecast,
            "sample_position": visualization_sample_position,
            "max_channels": visualization_max_channels,
        },
    )
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
        run_loss = loss or "MAE"
        run_target_metric = target_metric or run_loss
        run_learning_rate = float(
            learning_rate if learning_rate is not None else training["learning_rate"]
        )
        run_weight_decay = float(
            weight_decay if weight_decay is not None else training["weight_decay"]
        )
        run_patience = int(
            early_stopping_patience
            if early_stopping_patience is not None
            else training["early_stopping_patience"]
        )
        if metric_scale == "normalized":
            run_metrics = [
                metric for metric in training["metrics"] if metric in {"MAE", "MSE", "RMSE"}
            ]
        else:
            run_metrics = training["metrics"]
        save_predictions = visualize_forecast or (
            training["save_predictions"] and artifact_policy == "full"
        )
        visualization_sample_index = None
        result_sample_indices = None
        if visualize_forecast:
            test_data = np.load(data_path / "test_data.npy", mmap_mode="r")
            common_sample_count = (
                len(test_data)
                - run.input_length
                - max(int(horizon) for horizon in dataset["forecast_horizons"])
                + 1
            )
            visualization_sample_index = select_sample_index(
                common_sample_count, visualization_sample_position
            )
            if artifact_policy == "metrics":
                result_sample_indices = [visualization_sample_index]
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
            loss=run_loss,
            target_metric=run_target_metric,
            optimizer_params={"lr": run_learning_rate, "weight_decay": run_weight_decay},
            metrics=run_metrics,
            callbacks=[EarlyStopping(patience=run_patience)],
            seed=run.seed,
            save_results=save_predictions,
            result_sample_indices=result_sample_indices,
            eval_after_train=True,
            rescale=metric_scale == "original",
            deterministic=True,
            cudnn_benchmark=False,
            cudnn_determinstic=True,
            test_interval=None,
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
            model_config=asdict(model_config) if is_dataclass(model_config) else dict(model_config),
            training_config={
                "loss": run_loss,
                "target_metric": run_target_metric,
                "learning_rate": run_learning_rate,
                "weight_decay": run_weight_decay,
                "early_stopping_patience": run_patience,
            },
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
        if visualize_forecast:
            visualization = export_run_forecast_visualization(
                result_dir=resolved_checkpoint_dir / "test_results",
                processed_dir=data_path,
                run_dir=run_dir,
                dataset=run.dataset,
                model=run.model,
                horizon=run.output_length,
                seed=run.seed,
                input_length=run.input_length,
                output_length=run.output_length,
                metric_scale=metric_scale,
                sample_position=visualization_sample_position,
                sample_index=visualization_sample_index,
                max_channels=visualization_max_channels,
            )
            manifest["visualization"] = visualization
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
    parser.add_argument("--loss", choices=["MAE", "MSE"])
    parser.add_argument("--target-metric", choices=["MAE", "MSE", "RMSE"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--data-fingerprint")
    parser.add_argument("--protocol-signature")
    parser.add_argument("--visualize-forecast", action="store_true")
    parser.add_argument("--visualization-sample-position", type=float, default=0.5)
    parser.add_argument("--visualization-max-channels", type=int, default=4)
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
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive")
    if not 0.0 <= args.visualization_sample_position <= 1.0:
        parser.error("--visualization-sample-position must be in [0, 1]")
    if args.visualization_max_channels < 1:
        parser.error("--visualization-max-channels must be positive")
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
            args.loss,
            args.target_metric,
            args.learning_rate,
            args.weight_decay,
            args.early_stopping_patience,
            args.data_fingerprint,
            args.protocol_signature,
            args.visualize_forecast,
            args.visualization_sample_position,
            args.visualization_max_channels,
        )


if __name__ == "__main__":
    main()
