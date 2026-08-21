"""Repository paths and structured experiment catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CATALOG_PATH = PROJECT_ROOT / "datasets" / "catalog.json"
BACKBONE_REGISTRY_PATH = PROJECT_ROOT / "Baselines" / "registry.json"
PREEXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "pre_experiments" / "config.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_catalog() -> Dict[str, Dict[str, Any]]:
    return load_json(DATASET_CATALOG_PATH)


def load_backbone_registry() -> Dict[str, Any]:
    return load_json(BACKBONE_REGISTRY_PATH)


def load_preexperiment_config() -> Dict[str, Any]:
    return load_json(PREEXPERIMENT_CONFIG_PATH)


def resolve_dataset(name: str) -> Tuple[str, Dict[str, Any]]:
    catalog = load_dataset_catalog()
    if name in catalog:
        return name, catalog[name]
    lowered = name.lower()
    for canonical, entry in catalog.items():
        aliases = entry.get("aliases", [])
        if canonical.lower() == lowered or any(alias.lower() == lowered for alias in aliases):
            return canonical, entry
    choices = ", ".join(catalog)
    raise KeyError(f"Unknown dataset '{name}'. Available datasets: {choices}")


def find_raw_files(dataset_name: str) -> Tuple[Path, ...]:
    canonical, entry = resolve_dataset(dataset_name)
    candidate_names = [canonical] + list(entry.get("aliases", []))
    candidate_dirs = [PROJECT_ROOT / "datasets" / "raw" / name for name in candidate_names]
    for raw_dir in candidate_dirs:
        found = tuple(raw_dir / name for name in entry["raw_files"] if (raw_dir / name).is_file())
        if entry["task"] == "classification":
            expected = tuple(raw_dir / name for name in entry["raw_files"])
            if all(path.is_file() for path in expected):
                return expected
        if found:
            return found
    raw_dir = candidate_dirs[0]
    expected_names = ", ".join(entry["raw_files"])
    raise FileNotFoundError(f"No raw file for {canonical} in {raw_dir}; expected {expected_names}")
