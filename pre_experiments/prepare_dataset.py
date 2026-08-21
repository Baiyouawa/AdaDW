#!/usr/bin/env python3
"""Prepare one registered dataset for profiling and BasicTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--strict-shape", action="store_true")
    args = parser.parse_args()
    output = prepare_dataset(args.dataset, args.output_root, args.strict_shape)
    print(output)


if __name__ == "__main__":
    main()

