#!/usr/bin/env python3
"""Run all eight Backbones on all nine forecasting datasets, resumably."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import PROJECT_ROOT as CATALOG_PROJECT_ROOT
from adawd_preexp.catalog import load_backbone_registry, load_dataset_catalog


DEFAULT_CONFIG = PROJECT_ROOT / "pre_experiments" / "benchmark_config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "pre_experiments" / "results" / "forecasting_raw"
PROTOCOL_SCHEMA_VERSION = 2
SHARED_PROTOCOL_FILES = (
    "src/adawd_preexp/capacity.py",
    "src/adawd_preexp/data.py",
    "pre_experiments/run_capacity_sweep.py",
)
SHARED_BASELINE_DIRS = (
    "Baselines/basicts/configs",
    "Baselines/basicts/data",
    "Baselines/basicts/metrics",
    "Baselines/basicts/modules",
    "Baselines/basicts/runners",
    "Baselines/basicts/scaler",
    "Baselines/basicts/utils",
)


@dataclass(frozen=True)
class BenchmarkRun:
    index: int
    model: str
    dataset: str
    horizon: int
    seed: int
    run_id: str
    epochs: int
    batch_size: int
    loss: str
    target_metric: str
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    data_fingerprint: str
    protocol_signature: str


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_digest(model: str) -> str:
    paths = [PROJECT_ROOT / relative for relative in SHARED_PROTOCOL_FILES]
    for relative in SHARED_BASELINE_DIRS:
        paths.extend(sorted((PROJECT_ROOT / relative).rglob("*.py")))
    paths.extend(sorted((PROJECT_ROOT / "Baselines" / "basicts" / "models" / model).rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _data_fingerprint(dataset: str) -> str:
    meta_path = CATALOG_PROJECT_ROOT / "datasets" / "processed" / dataset / "meta.json"
    if not meta_path.is_file():
        return "unprepared"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(metadata.get("data_fingerprint", "legacy-metadata"))


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
    if len(config["seeds"]) != 3:
        raise ValueError("the full benchmark requires exactly three seeds")
    defaults = config.get("training_defaults")
    required_defaults = {
        "loss", "target_metric", "learning_rate", "weight_decay",
        "early_stopping_patience",
    }
    if not isinstance(defaults, dict) or set(defaults) != required_defaults:
        raise ValueError(
            "training_defaults must contain exactly: "
            + ", ".join(sorted(required_defaults))
        )
    if defaults["loss"] not in {"MAE", "MSE"}:
        raise ValueError("benchmark loss must be MAE or MSE")
    if defaults["target_metric"] not in {"MAE", "MSE", "RMSE"}:
        raise ValueError("benchmark target_metric must be MAE, MSE or RMSE")
    if float(defaults["learning_rate"]) <= 0 or float(defaults["weight_decay"]) < 0:
        raise ValueError("invalid optimizer settings")
    if int(defaults["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive")
    catalog = load_dataset_catalog()
    if len(catalog) != 9 or any(len(entry["forecast_horizons"]) != 4 for entry in catalog.values()):
        raise ValueError("the full benchmark requires nine datasets with four horizons each")
    registry = load_backbone_registry()["models"]
    benchmark_configs = []
    for model in required_models:
        model_config = config["models"][model]
        if int(model_config["epochs"]) < 1 or int(model_config["batch_size"]) < 1:
            raise ValueError(f"{model} has invalid epochs or batch_size")
        architecture = registry[model].get("benchmark_config")
        if not isinstance(architecture, dict):
            raise ValueError(f"{model} has no explicit benchmark_config in the registry")
        for parameter in (registry[model]["depth_parameter"], registry[model]["width_parameter"]):
            if parameter not in architecture:
                raise ValueError(f"{model} benchmark_config is missing {parameter}")
        benchmark_configs.append(json.dumps(architecture, sort_keys=True))
    if len(set(benchmark_configs)) != len(benchmark_configs):
        raise ValueError("each Backbone must have a distinct explicit benchmark_config")
    return config


def build_plan(config: dict, smoke: bool = False) -> list[BenchmarkRun]:
    catalog = load_dataset_catalog()
    registry = load_backbone_registry()["models"]
    caps = config.get("dataset_batch_caps", {})
    defaults = config["training_defaults"]
    source_digests = {model: _source_digest(model) for model in config["models"]}
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
                    architecture = registry[model]["benchmark_config"]
                    depth = int(architecture[registry[model]["depth_parameter"]])
                    width = int(architecture[registry[model]["width_parameter"]])
                    width_unit = int(registry[model]["width_unit"])
                    if width % width_unit:
                        raise ValueError(
                            f"{model} benchmark width {width} is not divisible by "
                            f"width_unit {width_unit}"
                        )
                    width_group = width // width_unit
                    run_id = (
                        f"{model}__{dataset}__h{int(horizon)}__raw"
                        f"__d{depth}__wg{width_group}__s{int(seed)}"
                    )
                    data_fingerprint = _data_fingerprint(dataset)
                    protocol = {
                        "schema_version": PROTOCOL_SCHEMA_VERSION,
                        "model": model,
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "run_id": run_id,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "training": defaults,
                        "metric_scale": config["metric_scale"],
                        "artifact_policy": config["artifact_policy"],
                        "dataset_catalog": dataset_config,
                        "data_fingerprint": data_fingerprint,
                        "backbone": registry[model],
                        "source_digest": source_digests[model],
                    }
                    plan.append(
                        BenchmarkRun(
                            index=len(plan) + 1,
                            model=model,
                            dataset=dataset,
                            horizon=int(horizon),
                            seed=int(seed),
                            run_id=run_id,
                            epochs=epochs,
                            batch_size=batch_size,
                            loss=str(defaults["loss"]),
                            target_metric=str(defaults["target_metric"]),
                            learning_rate=float(defaults["learning_rate"]),
                            weight_decay=float(defaults["weight_decay"]),
                            early_stopping_patience=int(defaults["early_stopping_patience"]),
                            data_fingerprint=data_fingerprint,
                            protocol_signature=_canonical_digest(protocol),
                        )
                    )
    expected = sum(
        (1 if smoke else len(dataset_config["forecast_horizons"]))
        * (1 if smoke else len(config["seeds"]))
        for dataset_config in catalog.values()
    ) * len(config["models"])
    if len(plan) != expected:
        raise RuntimeError(f"incomplete benchmark matrix: expected={expected}, actual={len(plan)}")
    run_ids = [run.run_id for run in plan]
    signatures = [run.protocol_signature for run in plan]
    if len(set(run_ids)) != len(run_ids) or len(set(signatures)) != len(signatures):
        raise RuntimeError("benchmark plan contains duplicate run IDs or protocol signatures")
    return plan


def manifest_path(output_root: Path, run: BenchmarkRun) -> Path | None:
    path = output_root / run.run_id / "manifest.json"
    return path if path.is_file() else None


def is_complete(output_root: Path, run: BenchmarkRun, config: dict) -> bool:
    path = manifest_path(output_root, run)
    if path is None:
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    overall = (manifest.get("metrics") or {}).get("overall") or {}
    try:
        metrics_complete = all(
            name in overall and math.isfinite(float(overall[name]))
            for name in ("MAE", "MSE", "RMSE")
        )
    except (TypeError, ValueError):
        metrics_complete = False
    return (
        manifest.get("status") == "complete"
        and manifest.get("run_id") == run.run_id
        and manifest.get("protocol_signature") == run.protocol_signature
        and manifest.get("data_fingerprint") == run.data_fingerprint
        and metrics_complete
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
        "--loss", run.loss,
        "--target-metric", run.target_metric,
        "--learning-rate", str(run.learning_rate),
        "--weight-decay", str(run.weight_decay),
        "--early-stopping-patience", str(run.early_stopping_patience),
        "--data-fingerprint", run.data_fingerprint,
        "--protocol-signature", run.protocol_signature,
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
