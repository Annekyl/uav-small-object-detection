import zipfile
from pathlib import Path

import pytest

from scripts.download_visdrone import safe_extract


def test_safe_extract(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    output = tmp_path / "output"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("split/images/example.jpg", b"image")
    safe_extract(archive, output)
    assert (output / "split/images/example.jpg").read_bytes() == b"image"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="不安全路径"):
        safe_extract(archive, tmp_path / "output")
