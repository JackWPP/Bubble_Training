"""Create a small YOLO debug dataset from yolo_dataset_grouped."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != ROOT.resolve() or not resolved.name.startswith("yolo_dataset_debug"):
        raise ValueError(f"Refusing to clean unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def collect_positive_images(source: Path, split: str) -> list[Path]:
    images = sorted((source / split / "images").glob("*.jpg"))
    positives = []
    for image in images:
        label = source / split / "labels" / f"{image.stem}.txt"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            positives.append(image)
    return positives


def copy_split(source: Path, output: Path, split: str, count: int, rng: random.Random) -> None:
    (output / split / "images").mkdir(parents=True, exist_ok=True)
    (output / split / "labels").mkdir(parents=True, exist_ok=True)
    images = collect_positive_images(source, split)
    selected = rng.sample(images, min(count, len(images)))
    for image in selected:
        label = source / split / "labels" / f"{image.stem}.txt"
        shutil.copy2(image, output / split / "images" / image.name)
        shutil.copy2(label, output / split / "labels" / label.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "yolo_dataset_grouped")
    parser.add_argument("--output", type=Path, default=ROOT / "yolo_dataset_debug")
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.clean:
        safe_clean(output)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    copy_split(source, output, "train", args.train, rng)
    copy_split(source, output, "val", args.val, rng)
    copy_split(source, output, "test", args.val, rng)
    yaml_text = f"""path: {output.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ["bubble"]
"""
    (output / "bubble_debug.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"debug dataset written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
