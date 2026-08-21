#!/usr/bin/env python3
"""Compute paper-defined U/M descriptors for one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import load_preexperiment_config, resolve_dataset
from adawd_preexp.data import load_profile_segments
from adawd_preexp.profiler import ProfilerConfig, profile_segments, summarize_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--all-channels", action="store_true")
    parser.add_argument("--processed", action="store_true", help="Prefer prepared split arrays over a raw CSV")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    canonical, dataset_entry = resolve_dataset(args.dataset)
    profiler_values = dict(load_preexperiment_config()["profiler"])
    profiler_values.update(
        window_size=args.window_size or dataset_entry["profile_window"],
        stride=args.stride or dataset_entry["profile_stride"],
        max_profile_channels=None
        if args.all_channels
        else dataset_entry.get("max_profile_channels"),
    )
    if args.max_windows is not None:
        profiler_values["max_windows_per_segment"] = args.max_windows
    config = ProfilerConfig(**profiler_values)
    segments = load_profile_segments(canonical, prefer_raw=not args.processed)
    profiles = profile_segments(canonical, segments, config)
    summary = summarize_profiles(profiles, config)

    output_dir = args.output_dir or PROJECT_ROOT / "pre_experiments" / "results" / "profiles" / canonical
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_dir / "windows.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"profiles={output_dir / 'windows.csv'}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

