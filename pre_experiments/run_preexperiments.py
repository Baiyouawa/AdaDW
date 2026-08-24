#!/usr/bin/env python3
"""Run dataset, depth or width pre-experiment stages with consistent defaults."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_dataset_catalog, resolve_dataset


def run_command(command: Sequence[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def prepare_command(dataset: str) -> list[str]:
    return [
        sys.executable,
        "pre_experiments/prepare_dataset.py",
        "--dataset",
        dataset,
        "--strict-shape",
    ]


def capacity_command(
    axis: str,
    dataset: str,
    model: str,
    horizon: int,
    seeds: Sequence[int],
    gpu: str | None,
    dry_run: bool,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> list[str]:
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
        *(str(seed) for seed in seeds),
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        if epochs is not None:
            command.extend(["--epochs", str(epochs)])
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
        if gpu is not None:
            command.extend(["--gpu", gpu])
        command.append("--all")
    return command


def run_dataset_stage(
    datasets: Sequence[str],
    download: bool,
    prepare_only: bool,
    processed_profiles: bool,
) -> None:
    if download:
        run_command(
            [
                sys.executable,
                "download_datasets.py",
                "--dataset",
                *datasets,
            ]
        )

    for dataset in datasets:
        canonical, _ = resolve_dataset(dataset)
        run_command(prepare_command(canonical))

    if prepare_only:
        return

    for dataset in datasets:
        canonical, _ = resolve_dataset(dataset)
        profile_command = [
            sys.executable,
            "pre_experiments/profile_dataset.py",
            "--dataset",
            canonical,
        ]
        if processed_profiles:
            profile_command.append("--processed")
        run_command(profile_command)
        run_command(
            [
                sys.executable,
                "pre_experiments/plot_profiles.py",
                "--profiles",
                f"pre_experiments/results/profiles/{canonical}/windows.csv",
            ]
        )


def run_capacity_stage(args: argparse.Namespace) -> None:
    canonical, entry = resolve_dataset(args.dataset)
    horizon = args.horizon or int(entry["forecast_horizons"][0])
    if horizon not in entry["forecast_horizons"]:
        choices = ", ".join(str(value) for value in entry["forecast_horizons"])
        raise ValueError(f"{canonical} horizon must be one of: {choices}")

    if not args.skip_prepare and not args.dry_run:
        run_command(prepare_command(canonical))
    run_command(
        capacity_command(
            axis=args.stage,
            dataset=canonical,
            model=args.model,
            horizon=horizon,
            seeds=args.seeds,
            gpu=None if args.cpu else args.gpu,
            dry_run=args.dry_run,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    dataset_parser = subparsers.add_parser(
        "dataset", help="strictly prepare, profile and plot datasets"
    )
    dataset_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(load_dataset_catalog()),
        help="datasets to process (default: all registered datasets)",
    )
    dataset_parser.add_argument(
        "--download",
        action="store_true",
        help="download and validate the selected raw datasets before preprocessing",
    )
    dataset_parser.add_argument(
        "--prepare-only", action="store_true", help="stop after strict preprocessing"
    )
    dataset_parser.add_argument(
        "--processed-profiles",
        action="store_true",
        help="profile prepared split arrays instead of the complete raw CSV",
    )

    for stage in ("depth", "width"):
        capacity_parser = subparsers.add_parser(stage, help=f"run a {stage} capacity sweep")
        capacity_parser.add_argument("--dataset", default="ETTh1")
        capacity_parser.add_argument("--model", default="PatchTST")
        capacity_parser.add_argument(
            "--horizon",
            type=int,
            help="forecast horizon (default: the dataset's shortest registered horizon)",
        )
        capacity_parser.add_argument("--seeds", type=int, nargs="+", default=[42])
        capacity_parser.add_argument("--gpu", default="0")
        capacity_parser.add_argument("--cpu", action="store_true")
        capacity_parser.add_argument("--dry-run", action="store_true")
        capacity_parser.add_argument("--skip-prepare", action="store_true")
        capacity_parser.add_argument("--epochs", type=int)
        capacity_parser.add_argument("--batch-size", type=int)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "dataset":
        datasets = args.datasets or list(load_dataset_catalog())
        run_dataset_stage(datasets, args.download, args.prepare_only, args.processed_profiles)
    else:
        run_capacity_stage(args)


if __name__ == "__main__":
    main()
