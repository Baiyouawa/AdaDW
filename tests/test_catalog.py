import pytest

from adawd_preexp.catalog import is_git_lfs_pointer, load_json


def test_load_json_explains_unresolved_lfs_pointer(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 42\n",
        encoding="utf-8",
    )

    assert is_git_lfs_pointer(path)
    with pytest.raises(RuntimeError, match="unresolved Git LFS pointer"):
        load_json(path)


def test_load_json_reports_source_location(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"dataset": }\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1, column 13"):
        load_json(path)
