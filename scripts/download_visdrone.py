"""下载并安全解压 VisDrone2019-DET 数据集。"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = {
    "train": "VisDrone2019-DET-train.zip",
    "val": "VisDrone2019-DET-val.zip",
    "test": "VisDrone2019-DET-test-dev.zip",
}
BUFFER_SIZE = 1024 * 1024


def download_with_resume(url: str, destination: Path) -> None:
    """下载文件；若存在 .part 文件则尝试通过 HTTP Range 续传。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "uav-small-object-detection/0.1"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下载失败：HTTP {exc.code}，{url}") from exc

    status = getattr(response, "status", 200)
    if downloaded and status != 206:
        downloaded = 0
        partial.unlink(missing_ok=True)

    content_length = int(response.headers.get("Content-Length", "0"))
    total = downloaded + content_length if content_length else 0
    mode = "ab" if downloaded else "wb"
    last_percent = -1
    with response, partial.open(mode) as output:
        while chunk := response.read(BUFFER_SIZE):
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = int(downloaded * 100 / total)
                if percent != last_percent:
                    print(f"\r{destination.name}: {percent:3d}%", end="", flush=True)
                    last_percent = percent
    print()
    partial.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    """拒绝包含目录穿越路径的 ZIP，并解压到目标目录。"""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"ZIP 包含不安全路径：{member.filename}")
        zip_file.extractall(destination)


def validate_split(dataset_root: Path, split: str) -> None:
    folder_name = ARCHIVES[split].removesuffix(".zip")
    split_root = dataset_root / folder_name
    images = split_root / "images"
    annotations = split_root / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        raise RuntimeError(f"解压后目录结构不完整：{split_root}")
    image_count = sum(1 for path in images.iterdir() if path.suffix.lower() == ".jpg")
    annotation_count = sum(1 for path in annotations.glob("*.txt"))
    if image_count == 0 or annotation_count == 0:
        raise RuntimeError(f"解压后没有找到图像或标注：{split_root}")
    print(f"{split}: {image_count} 张图像，{annotation_count} 个标注文件")


def prepare_split(dataset_root: Path, archive_root: Path, split: str, keep_archive: bool) -> None:
    archive_name = ARCHIVES[split]
    archive = archive_root / archive_name
    extracted = dataset_root / archive_name.removesuffix(".zip")
    if extracted.exists():
        print(f"已存在，跳过下载：{extracted}")
        validate_split(dataset_root, split)
        return
    if not archive.exists():
        download_with_resume(f"{BASE_URL}/{archive_name}", archive)
    else:
        print(f"使用已有压缩包：{archive}")
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"文件不是有效 ZIP：{archive}")
    print(f"正在解压：{archive.name}")
    safe_extract(archive, dataset_root)
    validate_split(dataset_root, split)
    if not keep_archive:
        archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/VisDrone2019-DET"))
    parser.add_argument("--split", choices=["all", *ARCHIVES], default="all")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    archive_root = args.output / "archives"
    splits = ARCHIVES if args.split == "all" else (args.split,)
    for split in splits:
        prepare_split(args.output, archive_root, split, args.keep_archives)
    if archive_root.exists() and not any(archive_root.iterdir()):
        shutil.rmtree(archive_root)


if __name__ == "__main__":
    main()
