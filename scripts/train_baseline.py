"""加载受版本管理的配置，检查环境并启动 YOLOv8n Baseline 训练。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"配置不是 YAML 映射：{path}")
    return data


def validate_dataset(dataset_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, folder in REQUIRED_SPLITS.items():
        split_root = dataset_root / folder
        images_dir, labels_dir = split_root / "images", split_root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"{split} 缺少 images 或 labels：{split_root}")
        image_count = sum(1 for path in images_dir.iterdir() if path.suffix.lower() == ".jpg")
        label_count = sum(1 for path in labels_dir.glob("*.txt"))
        if image_count != label_count:
            raise ValueError(f"{split} 图像 {image_count} 与标签 {label_count} 数量不一致")
        counts[split] = image_count
    return counts


def build_runtime_dataset_yaml(template: Path, dataset_root: Path, output: Path) -> Path:
    data = load_yaml(template)
    data["path"] = str(dataset_root.resolve())
    data["train"] = "VisDrone2019-DET-train/images"
    data["val"] = "VisDrone2019-DET-val/images"
    data["test"] = "VisDrone2019-DET-test-dev/images"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("ultralytics", "torch", "torchvision", "pyyaml")
        },
    }
    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        info["cuda"] = {"available": False, "reason": "torch 未安装"}
    return info


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"覆盖参数必须为 key=value：{value}")
    key, raw_value = value.split("=", 1)
    return key, yaml.safe_load(raw_value)


def prepare_run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    for override in args.override:
        key, value = parse_override(override)
        config[key] = value

    dataset_root = args.dataset.resolve()
    counts = validate_dataset(dataset_root)
    data_template = (PROJECT_ROOT / str(config["data"])).resolve()
    runtime_data = PROJECT_ROOT / "results/runtime/visdrone-resolved.yaml"
    build_runtime_dataset_yaml(data_template, dataset_root, runtime_data)
    config["data"] = str(runtime_data)

    metadata = {
        "config_source": str(config_path),
        "dataset_root": str(dataset_root),
        "dataset_counts": counts,
        "resolved_config": config,
        "environment": environment_info(),
    }
    return config, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/train/yolov8n_baseline_640.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/VisDrone2019-DET",
    )
    parser.add_argument("--override", action="append", default=[], help="覆盖训练参数：key=value")
    parser.add_argument("--dry-run", action="store_true", help="只检查并输出配置，不导入 PyTorch")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, metadata = prepare_run(args)
    metadata_path = PROJECT_ROOT / "results/experiments" / f"{config['name']}-launch.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(f"启动信息：{metadata_path}")
    if args.dry_run:
        print("配置检查通过；dry-run 不启动训练。")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装训练依赖，请先执行 uv sync --extra train") from exc

    model_path = config.pop("model")
    YOLO(model_path).train(**config)


if __name__ == "__main__":
    main()
