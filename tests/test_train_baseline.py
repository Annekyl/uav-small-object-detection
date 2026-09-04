from pathlib import Path

import pytest
import yaml

from scripts.train_baseline import build_runtime_dataset_yaml, parse_override, validate_dataset


def make_dataset(root: Path) -> None:
    for folder in (
        "VisDrone2019-DET-train",
        "VisDrone2019-DET-val",
        "VisDrone2019-DET-test-dev",
    ):
        images, labels = root / folder / "images", root / folder / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        (images / "sample.jpg").write_bytes(b"not-needed-for-counting")
        (labels / "sample.txt").write_text("", encoding="utf-8")


def test_validate_dataset(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    assert validate_dataset(tmp_path) == {"train": 1, "val": 1, "test": 1}


def test_validate_dataset_rejects_count_mismatch(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "VisDrone2019-DET-val/labels/sample.txt").unlink()
    with pytest.raises(ValueError, match="数量不一致"):
        validate_dataset(tmp_path)


def test_runtime_dataset_uses_absolute_root(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    output = tmp_path / "runtime/data.yaml"
    template.write_text("path: old\nnames:\n  0: pedestrian\n", encoding="utf-8")
    build_runtime_dataset_yaml(template, tmp_path / "dataset", output)
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert data["train"] == "VisDrone2019-DET-train/images"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("batch=8", ("batch", 8)), ("device=cpu", ("device", "cpu")), ("amp=false", ("amp", False))],
)
def test_parse_override(text: str, expected: tuple[str, object]) -> None:
    assert parse_override(text) == expected
