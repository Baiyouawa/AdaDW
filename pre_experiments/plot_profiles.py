#!/usr/bin/env python3
"""Plot U/M distributions and the joint descriptor plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=10000)
    args = parser.parse_args()
    frame = pd.read_csv(args.profiles)
    required = {"U", "M", "U_bucket", "M_bucket"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Profile table is missing: {sorted(required - set(frame.columns))}")
    dataset = frame["dataset"].iloc[0] if "dataset" in frame else args.profiles.stem
    sample = frame.sample(min(args.max_points, len(frame)), random_state=42)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].hist(frame["U"], bins=40, color="#167D8D", alpha=0.9)
    axes[0].set(xlabel="State-update demand U", ylabel="Local units", title="U distribution")
    axes[1].hist(frame["M"], bins=40, color="#D15C32", alpha=0.9)
    axes[1].set(xlabel="Pattern diversity M", ylabel="Local units", title="M distribution")
    colors = np.where(sample["U_bucket"] == "high", "#B33A3A", np.where(sample["M_bucket"] == "high", "#2864A6", "#777777"))
    axes[2].scatter(sample["U"], sample["M"], s=8, c=colors, alpha=0.45, linewidths=0)
    axes[2].axvline(frame["U"].median(), color="black", linewidth=0.8, linestyle="--")
    axes[2].axhline(frame["M"].median(), color="black", linewidth=0.8, linestyle="--")
    axes[2].set(xlabel="U", ylabel="M", title="Joint descriptor plane")
    figure.suptitle(str(dataset), fontsize=12)

    output = args.output or args.profiles.with_name("profile_diagnostics.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()

