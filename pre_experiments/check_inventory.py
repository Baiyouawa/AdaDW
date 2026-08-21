#!/usr/bin/env python3
"""Report which registered datasets and Backbones are actually available."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adawd_preexp.catalog import find_raw_files, load_backbone_registry, load_dataset_catalog


def main() -> None:
    datasets = []
    for name, entry in load_dataset_catalog().items():
        try:
            raw_files = [str(path) for path in find_raw_files(name)]
            status = "raw_available"
        except FileNotFoundError:
            raw_files = []
            processed = PROJECT_ROOT / "datasets" / "processed" / name
            status = "processed_available" if (processed / "train_data.npy").is_file() else "missing_data"
        datasets.append({"name": name, "task": entry["task"], "status": status, "files": raw_files})
    backbones = []
    for name, entry in load_backbone_registry()["models"].items():
        backbones.append(
            {
                "name": name,
                "year": entry["year"],
                "status": entry["status"],
                "reason": entry.get("reason"),
            }
        )
    print(json.dumps({"datasets": datasets, "backbones": backbones}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
