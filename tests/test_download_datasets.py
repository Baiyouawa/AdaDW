import csv
import io

import pytest

import download_datasets
from download_datasets import DatasetSpec, ValidationError, dataset_urls, validate_csv


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_validate_csv_accepts_expected_shape(tmp_path):
    path = tmp_path / "tiny.csv"
    spec = DatasetSpec("tiny.csv", "tiny.csv", rows=2, channels=2)
    write_csv(
        path,
        [
            ["date", "a", "b"],
            ["2026-01-01", "1", "2"],
            ["2026-01-02", "3", "4"],
        ],
    )

    validate_csv(path, spec)


def test_validate_csv_rejects_wrong_dimensions(tmp_path):
    path = tmp_path / "tiny.csv"
    spec = DatasetSpec("tiny.csv", "tiny.csv", rows=2, channels=2)
    write_csv(path, [["date", "a"], ["2026-01-01", "1"]])

    with pytest.raises(ValidationError, match="columns=2; expected 3"):
        validate_csv(path, spec)


def test_validate_csv_rejects_unresolved_lfs_pointer(tmp_path):
    path = tmp_path / "tiny.csv"
    spec = DatasetSpec("tiny.csv", "tiny.csv", rows=2, channels=2)
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 42\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unresolved Git LFS pointer"):
        validate_csv(path, spec)


def test_download_once_rejects_truncated_response(tmp_path, monkeypatch):
    response = io.BytesIO(b"short")
    response.status = 200
    response.headers = {"Content-Length": "10"}
    monkeypatch.setattr(download_datasets.URL_OPENER, "open", lambda *args, **kwargs: response)
    part_path = tmp_path / "tiny.csv.part"

    with pytest.raises(OSError, match="received 5 of 10 bytes"):
        download_datasets.download_once(
            "https://example.test/tiny.csv", part_path, timeout=1, name="tiny"
        )

    assert part_path.read_bytes() == b"short"


def test_dataset_urls_prefer_original_source():
    spec = DatasetSpec(
        "group/data.csv",
        "data.csv",
        rows=1,
        channels=1,
        extra_urls=("https://original.example/data.csv",),
    )

    assert dataset_urls(spec, ["https://bundle.example/root/"]) == [
        "https://original.example/data.csv",
        "https://bundle.example/root/group/data.csv",
    ]
