from pathlib import Path

import pytest
from PIL import Image

from scripts.visualize_visdrone_labels import (
    YoloObject,
    choose_images,
    draw_labels,
    parse_yolo_label,
)


def test_parse_yolo_label(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("3 0.50000000 0.50000000 0.20000000 0.40000000\n", encoding="utf-8")
    assert parse_yolo_label(label) == [YoloObject(3, 0.5, 0.5, 0.2, 0.4)]


@pytest.mark.parametrize(
    "line",
    ["10 0.5 0.5 0.1 0.1", "0 1.2 0.5 0.1 0.1", "0 0.5 0.5 0 0.1", "0 0.99 0.5 0.1 0.1"],
)
def test_parse_yolo_label_rejects_invalid_values(tmp_path: Path, line: str) -> None:
    label = tmp_path / "invalid.txt"
    label.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_yolo_label(label)


def test_choose_images_is_deterministic(tmp_path: Path) -> None:
    for index in range(5):
        Image.new("RGB", (10, 10)).save(tmp_path / f"{index}.jpg")
    assert choose_images(tmp_path, 3, 42) == choose_images(tmp_path, 3, 42)


def test_draw_labels_creates_output(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    output = tmp_path / "output.jpg"
    Image.new("RGB", (100, 80), "white").save(image)
    draw_labels(image, [YoloObject(3, 0.5, 0.5, 0.2, 0.4)], output)
    assert output.is_file()
    assert Image.open(output).size == (100, 80)
