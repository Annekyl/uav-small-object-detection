from pathlib import Path

from PIL import Image

from uav_detection.data.visdrone import convert_split, parse_annotation_line, summarize_annotations


def test_parse_annotation_line() -> None:
    obj = parse_annotation_line("10,20,30,40,1,4,0,2", 1, Path("sample.txt"))
    assert (obj.left, obj.top, obj.width, obj.height, obj.category) == (10, 20, 30, 40, 4)


def test_convert_filters_non_training_classes(tmp_path: Path) -> None:
    images, annotations, labels = (tmp_path / name for name in ("images", "annotations", "labels"))
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (100, 80)).save(images / "0001.jpg")
    (annotations / "0001.txt").write_text(
        "10,20,30,40,1,4,0,0\n0,0,10,10,0,0,0,0\n20,20,10,10,1,11,0,0\n",
        encoding="utf-8",
    )
    stats = convert_split(images, annotations, labels)
    assert (stats.converted_objects, stats.ignored_regions, stats.other_objects) == (1, 1, 1)
    assert (labels / "0001.txt").read_text(encoding="utf-8").startswith("3 ")


def test_summarize_coco_sizes(tmp_path: Path) -> None:
    (tmp_path / "0001.txt").write_text(
        "0,0,10,10,1,1,0,0\n0,0,40,40,1,4,0,0\n0,0,100,100,1,9,0,0\n", encoding="utf-8"
    )
    report = summarize_annotations(tmp_path)
    assert report["coco_area_counts"] == {"small": 1, "medium": 1, "large": 1}
