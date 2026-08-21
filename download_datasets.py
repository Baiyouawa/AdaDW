#!/usr/bin/env python3
"""Download and validate all nine AdaWD forecasting datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw"

# THUML maintains the benchmark bundle used by Time-Series-Library. The mirror
# has the same repository layout and is useful where Hugging Face is blocked.
THUML_BASE_URLS = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main",
    "https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main",
)


@dataclass(frozen=True)
class DatasetSpec:
    remote_path: str
    filename: str
    rows: int
    channels: int
    extra_urls: tuple[str, ...] = ()


REMOTE_PATHS = {
    "ETTh1": "ETT-small/ETTh1.csv",
    "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv",
    "ETTm2": "ETT-small/ETTm2.csv",
    "Weather": "weather/weather.csv",
    "Electricity": "electricity/electricity.csv",
    "ILI": "illness/national_illness.csv",
    "ExchangeRate": "exchange_rate/exchange_rate.csv",
    "Traffic": "traffic/traffic.csv",
}

ETT_ORIGINAL_URLS = {
    name: f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
}


def load_dataset_specs() -> dict[str, DatasetSpec]:
    catalog_path = PROJECT_ROOT / "datasets" / "catalog.json"
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    if set(catalog) != set(REMOTE_PATHS):
        missing_sources = sorted(set(catalog) - set(REMOTE_PATHS))
        unknown_sources = sorted(set(REMOTE_PATHS) - set(catalog))
        raise RuntimeError(
            "download source map and dataset catalog differ: "
            f"missing_sources={missing_sources}, unknown_sources={unknown_sources}"
        )

    return {
        name: DatasetSpec(
            remote_path=REMOTE_PATHS[name],
            filename=entry["raw_files"][0],
            rows=int(entry["expected_time_steps"]),
            channels=int(entry["expected_channels"]),
            extra_urls=(ETT_ORIGINAL_URLS[name],) if name in ETT_ORIGINAL_URLS else (),
        )
        for name, entry in catalog.items()
    }


DATASETS = load_dataset_specs()


class ValidationError(RuntimeError):
    """Raised when a downloaded file is not the expected benchmark CSV."""


def human_size(size: Optional[int]) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def validate_csv(path: Path, spec: DatasetSpec) -> None:
    """Check the schema and exact benchmark dimensions without pandas."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise ValidationError("file is empty")
            if "date" not in header:
                raise ValidationError("missing required 'date' column")
            expected_columns = spec.channels + 1
            if len(header) != expected_columns:
                raise ValidationError(
                    f"columns={len(header)}; expected {expected_columns} "
                    f"(date + {spec.channels} signals)"
                )

            rows = 0
            for line_number, row in enumerate(reader, start=2):
                if len(row) != expected_columns:
                    raise ValidationError(
                        f"row {line_number} has {len(row)} columns; expected {expected_columns}"
                    )
                rows += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(str(exc)) from exc

    if rows != spec.rows:
        raise ValidationError(f"rows={rows}; expected {spec.rows}")


def dataset_urls(spec: DatasetSpec, base_urls: Iterable[str]) -> list[str]:
    # ETT's original repository is preferred; the THUML bundle is the fallback.
    return [
        *spec.extra_urls,
        *(f"{base.rstrip('/')}/{spec.remote_path}" for base in base_urls),
    ]


def report_progress(
    name: str, downloaded: int, total: Optional[int], final: bool = False
) -> None:
    if total:
        percent = min(downloaded / total * 100, 100.0)
        message = (
            f"  {name}: {human_size(downloaded)} / {human_size(total)} "
            f"({percent:5.1f}%)"
        )
    else:
        message = f"  {name}: {human_size(downloaded)}"

    if sys.stdout.isatty():
        print(f"\r{message}", end="\n" if final else "", flush=True)
    elif final:
        print(message)


def download_once(url: str, part_path: Path, timeout: float, name: str) -> None:
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "AdaWD-dataset-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        # A previous run can leave a complete .part file. Some servers answer a
        # range request at EOF with 416; validation below decides if it is usable.
        if exc.code == 416 and existing:
            return
        raise

    with response:
        status = getattr(response, "status", response.getcode())
        resumed = bool(existing and status == 206)
        mode = "ab" if resumed else "wb"
        downloaded = existing if resumed else 0
        content_length = response.headers.get("Content-Length")
        total = downloaded + int(content_length) if content_length else None
        last_report = time.monotonic()

        with part_path.open(mode) as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if sys.stdout.isatty() and now - last_report >= 0.5:
                    report_progress(name, downloaded, total)
                    last_report = now
            handle.flush()
            os.fsync(handle.fileno())
        report_progress(name, downloaded, total, final=True)


def download_dataset(
    name: str,
    spec: DatasetSpec,
    base_urls: tuple[str, ...],
    timeout: float,
    retries: int,
    force: bool,
    unavailable_origins: set[str],
) -> str:
    destination = RAW_ROOT / name / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and not force:
        try:
            validate_csv(destination, spec)
        except ValidationError as exc:
            print(f"[{name}] Existing file is invalid ({exc}); downloading a replacement.")
        else:
            print(f"[{name}] OK, already present: {destination.relative_to(PROJECT_ROOT)}")
            return "skipped"

    part_path = destination.with_name(destination.name + ".part")
    if force:
        part_path.unlink(missing_ok=True)

    failures: list[str] = []
    for url in dataset_urls(spec, base_urls):
        origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if origin in unavailable_origins:
            failures.append(f"{url}: host was unreachable earlier")
            continue

        print(f"[{name}] Source: {url}")
        host_unreachable = False
        for attempt in range(1, retries + 1):
            try:
                download_once(url, part_path, timeout, name)
                validate_csv(part_path, spec)
                os.replace(part_path, destination)
                print(f"[{name}] Saved: {destination.relative_to(PROJECT_ROOT)}")
                return "downloaded"
            except ValidationError as exc:
                failures.append(f"{url}: validation failed: {exc}")
                part_path.unlink(missing_ok=True)
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                failures.append(f"{url} (attempt {attempt}/{retries}): {reason}")
                host_unreachable = isinstance(exc, (URLError, TimeoutError)) and not isinstance(
                    exc, HTTPError
                )
                if attempt < retries:
                    print(f"  retry {attempt + 1}/{retries}: {reason}")
                    time.sleep(min(attempt * 2, 5))
        if host_unreachable:
            unavailable_origins.add(origin)

    details = "\n    ".join(failures)
    raise RuntimeError(f"all sources failed for {name}:\n    {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the nine AdaWD datasets into datasets/raw/<name>/ and "
            "validate their CSV dimensions."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="dataset(s) to download (default: all nine)",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        help=(
            "alternate Time-Series-Library repository base URL; may be repeated. "
            "It replaces the default Hugging Face endpoints"
        ),
    )
    parser.add_argument("--force", action="store_true", help="download again even if valid files exist")
    parser.add_argument("--timeout", type=float, default=30, help="network timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="attempts per source")
    parser.add_argument("--list", action="store_true", help="list targets without downloading")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries <= 0:
        parser.error("--retries must be positive")
    return args


def main() -> int:
    args = parse_args()
    selected = list(DATASETS) if "all" in args.dataset else list(dict.fromkeys(args.dataset))
    base_urls = tuple(args.base_url) if args.base_url else THUML_BASE_URLS

    if args.list:
        for name in selected:
            spec = DATASETS[name]
            print(f"{name:12} -> datasets/raw/{name}/{spec.filename}")
        return 0

    print(f"Downloading {len(selected)} dataset(s) into {RAW_ROOT}")
    unavailable_origins: set[str] = set()
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    errors: list[str] = []
    for name in selected:
        try:
            status = download_dataset(
                name,
                DATASETS[name],
                base_urls,
                args.timeout,
                args.retries,
                args.force,
                unavailable_origins,
            )
            counts[status] += 1
        except RuntimeError as exc:
            counts["failed"] += 1
            errors.append(str(exc))
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)

    print(
        "\nSummary: "
        f"downloaded={counts['downloaded']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}"
    )
    if errors:
        print("Re-run the same command to resume incomplete downloads.", file=sys.stderr)
        return 1
    print("All requested datasets are present and passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
