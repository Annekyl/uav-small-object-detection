from pathlib import Path

import pytest
import yaml

from scripts.train_baseline import (
    PROJECT_ROOT,
    build_runtime_dataset_yaml,
    load_yaml,
    parse_override,
    validate_dataset,
)


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


def test_a1_changes_only_model_initialization_and_output_identity() -> None:
    baseline = load_yaml(PROJECT_ROOT / "configs/train/yolov8n_baseline_640.yaml")
    p2 = load_yaml(PROJECT_ROOT / "configs/train/yolov8n_p2_640.yaml")

    allowed_differences = {"model", "pretrained_weights", "project", "name"}
    shared_keys = (set(baseline) | set(p2)) - allowed_differences
    assert {key: baseline.get(key) for key in shared_keys} == {
        key: p2.get(key) for key in shared_keys
    }
    assert p2["pretrained_weights"] == "yolov8n.pt"


def test_p2_model_has_four_detection_scales() -> None:
    model = load_yaml(PROJECT_ROOT / "configs/models/yolov8n-p2.yaml")
    detect = model["head"][-1]

    assert model["nc"] == 10
    assert model["scale"] == "n"
    assert detect[0] == [18, 21, 24, 27]
    assert detect[2] == "Detect"
