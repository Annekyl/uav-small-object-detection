"""VisDrone2019-DET 标注统计及 YOLO 格式转换工具。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class VisDroneObject:
    left: float
    top: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int


@dataclass
class ConversionStatistics:
    images: int = 0
    annotations: int = 0
    converted_objects: int = 0
    ignored_regions: int = 0
    other_objects: int = 0
    invalid_rows: int = 0
    invalid_boxes: int = 0
    clipped_boxes: int = 0
    missing_annotations: int = 0


def parse_annotation_line(line: str, line_number: int, source: Path) -> VisDroneObject:
    fields = [field.strip() for field in line.strip().rstrip(",").split(",")]
    if len(fields) != 8:
        raise ValueError(f"{source}:{line_number}: 期望 8 个字段，实际为 {len(fields)}")
    try:
        left, top, width, height = (float(value) for value in fields[:4])
        score, category, truncation, occlusion = (int(float(value)) for value in fields[4:])
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: 包含非数值字段") from exc
    return VisDroneObject(left, top, width, height, score, category, truncation, occlusion)


def read_annotations(path: Path) -> tuple[list[VisDroneObject], int]:
    objects: list[VisDroneObject] = []
    invalid_rows = 0
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                objects.append(parse_annotation_line(line, line_number, path))
            except ValueError:
                invalid_rows += 1
    return objects, invalid_rows


def clip_box(
    obj: VisDroneObject, image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(image_width), obj.left))
    y1 = max(0.0, min(float(image_height), obj.top))
    x2 = max(0.0, min(float(image_width), obj.left + obj.width))
    y2 = max(0.0, min(float(image_height), obj.top + obj.height))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def to_yolo_line(
    category: int, box: tuple[float, float, float, float], width: int, height: int
) -> str:
    x1, y1, x2, y2 = box
    values = (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    return f"{category - 1} " + " ".join(f"{value:.8f}" for value in values)


def convert_split(
    images_dir: Path, annotations_dir: Path, labels_dir: Path
) -> ConversionStatistics:
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError("图像目录或标注目录不存在")
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = ConversionStatistics()
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in images:
        stats.images += 1
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        output_path = labels_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            stats.missing_annotations += 1
            output_path.write_text("", encoding="utf-8")
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        objects, invalid_rows = read_annotations(annotation_path)
        stats.invalid_rows += invalid_rows
        stats.annotations += len(objects)
        lines: list[str] = []
        for obj in objects:
            if obj.category == 0:
                stats.ignored_regions += 1
                continue
            if obj.category == 11:
                stats.other_objects += 1
                continue
            if not 1 <= obj.category <= 10:
                stats.invalid_rows += 1
                continue
            if obj.width <= 0 or obj.height <= 0:
                stats.invalid_boxes += 1
                continue
            box = clip_box(obj, image_width, image_height)
            if box is None:
                stats.invalid_boxes += 1
                continue
            if box != (obj.left, obj.top, obj.left + obj.width, obj.top + obj.height):
                stats.clipped_boxes += 1
            lines.append(to_yolo_line(obj.category, box, image_width, image_height))
        stats.converted_objects += len(lines)
        output_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return stats


def summarize_annotations(annotations_dir: Path) -> dict[str, object]:
    class_counts: Counter[int] = Counter()
    area_counts = {"small": 0, "medium": 0, "large": 0}
    total_rows = invalid_rows = 0
    files = sorted(annotations_dir.glob("*.txt"))
    for path in files:
        objects, count = read_annotations(path)
        invalid_rows += count
        for obj in objects:
            total_rows += 1
            class_counts[obj.category] += 1
            if 1 <= obj.category <= 10 and obj.width > 0 and obj.height > 0:
                area = obj.width * obj.height
                area_counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    named_counts = {
        (
            "ignored-regions"
            if category == 0
            else "others"
            if category == 11
            else VISDRONE_CLASSES[category - 1]
            if 1 <= category <= 10
            else f"invalid-{category}"
        ): count
        for category, count in sorted(class_counts.items())
    }
    return {
        "annotation_files": len(files),
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "class_counts": named_counts,
        "coco_area_counts": area_counts,
    }


def write_report(report: dict[str, object], output: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="转换为 YOLO 标签")
    convert.add_argument("--images", type=Path, required=True)
    convert.add_argument("--annotations", type=Path, required=True)
    convert.add_argument("--labels", type=Path, required=True)
    convert.add_argument("--report", type=Path)
    stats = subparsers.add_parser("stats", help="统计原始标注")
    stats.add_argument("--annotations", type=Path, required=True)
    stats.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        write_report(asdict(convert_split(args.images, args.annotations, args.labels)), args.report)
    else:
        write_report(summarize_annotations(args.annotations), args.report)


if __name__ == "__main__":
    main()
