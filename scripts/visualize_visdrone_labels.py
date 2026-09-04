"""抽样检查 VisDrone 的 YOLO 标签并生成标注图和联系图。"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from uav_detection.data.visdrone import IMAGE_SUFFIXES, VISDRONE_CLASSES

COLORS = (
    "#ff3b30",
    "#ff9500",
    "#ffcc00",
    "#34c759",
    "#00c7be",
    "#32ade6",
    "#007aff",
    "#5856d6",
    "#af52de",
    "#ff2d55",
)


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class SampleRecord:
    split: str
    image: str
    label: str
    objects: int
    output: str


def parse_yolo_label(path: Path) -> list[YoloObject]:
    objects: list[YoloObject] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: YOLO 标签必须有 5 个字段")
            try:
                class_id = int(fields[0])
                center_x, center_y, width, height = map(float, fields[1:])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: 标签包含非数值字段") from exc
            if not 0 <= class_id < len(VISDRONE_CLASSES):
                raise ValueError(f"{path}:{line_number}: 类别 {class_id} 超出 0-9")
            if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
                raise ValueError(f"{path}:{line_number}: 坐标不在 [0, 1] 范围")
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"{path}:{line_number}: 框宽高必须大于 0")
            x1, y1 = center_x - width / 2, center_y - height / 2
            x2, y2 = center_x + width / 2, center_y + height / 2
            tolerance = 1e-6
            if x1 < -tolerance or y1 < -tolerance or x2 > 1 + tolerance or y2 > 1 + tolerance:
                raise ValueError(f"{path}:{line_number}: 框边界超出图像")
            objects.append(YoloObject(class_id, center_x, center_y, width, height))
    return objects


def draw_labels(image_path: Path, objects: list[YoloObject], output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(1, round(min(image.size) / 500))
    for obj in objects:
        image_width, image_height = image.size
        x1 = (obj.center_x - obj.width / 2) * image_width
        y1 = (obj.center_y - obj.height / 2) * image_height
        x2 = (obj.center_x + obj.width / 2) * image_width
        y2 = (obj.center_y + obj.height / 2) * image_height
        color = COLORS[obj.class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{obj.class_id}:{VISDRONE_CLASSES[obj.class_id]}"
        text_box = draw.textbbox((x1, y1), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 2)
        draw.rectangle((x1, text_y, x1 + text_width + 4, text_y + text_height + 2), fill=color)
        draw.text(
            (x1 + 2, text_y), label, fill="white", font=font, stroke_width=1, stroke_fill="black"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def choose_images(images_dir: Path, count: int, seed: int) -> list[Path]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if count > len(images):
        raise ValueError(f"请求抽取 {count} 张，但 {images_dir} 仅有 {len(images)} 张")
    return sorted(random.Random(seed).sample(images, count))


def create_contact_sheet(
    images: list[Path], output_path: Path, columns: int = 5, tile_size: tuple[int, int] = (360, 240)
) -> None:
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#202124")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(images):
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
        thumbnail.thumbnail((tile_size[0], tile_size[1] - 20), Image.Resampling.LANCZOS)
        column, row = index % columns, index // columns
        left = column * tile_size[0] + (tile_size[0] - thumbnail.width) // 2
        top = row * tile_size[1] + 20 + (tile_size[1] - 20 - thumbnail.height) // 2
        sheet.paste(thumbnail, (left, top))
        draw.text(
            (column * tile_size[0] + 4, row * tile_size[1] + 3), path.stem, fill="white", font=font
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def visualize_split(
    dataset_root: Path, split_folder: str, split_name: str, count: int, seed: int, output_root: Path
) -> list[SampleRecord]:
    split_root = dataset_root / split_folder
    images_dir, labels_dir = split_root / "images", split_root / "labels"
    selected = choose_images(images_dir, count, seed)
    records: list[SampleRecord] = []
    rendered: list[Path] = []
    for image_path in selected:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"标签不存在：{label_path}")
        objects = parse_yolo_label(label_path)
        output_path = output_root / split_name / f"{image_path.stem}.jpg"
        draw_labels(image_path, objects, output_path)
        rendered.append(output_path)
        records.append(
            SampleRecord(
                split_name, str(image_path), str(label_path), len(objects), str(output_path)
            )
        )
    create_contact_sheet(rendered, output_root / f"{split_name}-联系图.jpg")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/VisDrone2019-DET"))
    parser.add_argument("--output", type=Path, default=Path("results/data/label-audit"))
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    records = visualize_split(
        args.dataset, "VisDrone2019-DET-train", "train", args.train_count, args.seed, args.output
    )
    records.extend(
        visualize_split(
            args.dataset,
            "VisDrone2019-DET-val",
            "val",
            args.val_count,
            args.seed + 1,
            args.output,
        )
    )
    manifest = {"seed": args.seed, "samples": [asdict(record) for record in records]}
    manifest_path = args.output / "抽样清单.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已生成 {len(records)} 张标注图：{args.output}")


if __name__ == "__main__":
    main()
