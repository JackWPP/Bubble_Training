"""Validate Paper V4 curve-friendly dataset properties."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCES = {
    "20+40",
    "60+80",
    "big_fengchao",
    "bubble_1",
    "bubble_fc",
    "bubble_pad",
    "job_13_dataset_2026_04_30_19_34_23_coco 1.0",
}


def is_base_transform(transform: str) -> bool:
    return transform in {"raw", "letterbox"} or transform.startswith("tile_")


def label_distribution(dataset: Path, split: str) -> dict:
    widths: list[float] = []
    heights: list[float] = []
    for label_path in (dataset / split / "labels").glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            _cls, _cx, _cy, bw, bh = [float(part) for part in parts]
            widths.append(bw * 640)
            heights.append(bh * 640)
    total = len(widths)
    small = sum(1 for width, height in zip(widths, heights) if width < 16 or height < 16)
    return {
        "boxes": total,
        "small_lt16": small,
        "small_lt16_percent": round(small / total * 100, 2) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "yolo_dataset_paper_v4")
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    errors: list[str] = []
    data_yaml = dataset / "bubble.yaml"
    manifest_path = dataset / "manifest.json"
    stats_path = dataset / "build_stats.json"
    for path in (data_yaml, manifest_path, stats_path):
        if not path.exists():
            errors.append(f"missing {path}")
    if errors:
        print("\n".join(errors))
        return 1

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("split_mode") != "paper-v4":
        errors.append(f"unexpected split_mode: {stats.get('split_mode')}")

    by_source_split: dict[str, Counter] = defaultdict(Counter)
    by_source_key: dict[str, set[str]] = defaultdict(set)
    augmented_val_test = []
    transform_counts: dict[str, Counter] = defaultdict(Counter)
    for item in manifest:
        split = item["split"]
        transform = item.get("transform", "")
        by_source_split[item["source"]][split] += 1
        by_source_key[item["source_key"]].add(split)
        transform_counts[split][transform] += 1
        if split in {"val", "test"} and not is_base_transform(transform):
            augmented_val_test.append(item["image"])

    source_key_leaks = {key: splits for key, splits in by_source_key.items() if len(splits) > 1}
    if source_key_leaks:
        errors.append(f"source_key leakage: {len(source_key_leaks)}")
    if augmented_val_test:
        errors.append(f"augmented val/test samples: {len(augmented_val_test)}")
    missing_sources = REQUIRED_SOURCES - set(by_source_split)
    if missing_sources:
        errors.append(f"missing sources: {sorted(missing_sources)}")
    for source in sorted(REQUIRED_SOURCES & set(by_source_split)):
        if by_source_split[source]["train"] <= 0:
            errors.append(f"{source} has no train samples")

    unexpected_train_transforms = {
        transform
        for transform in transform_counts["train"]
        if not is_base_transform(transform) and transform != "hflip"
    }
    if unexpected_train_transforms:
        errors.append(f"unexpected train transforms: {sorted(unexpected_train_transforms)}")

    for split in ("train", "val", "test"):
        image_dir = dataset / data[split]
        label_dir = dataset / split / "labels"
        if not image_dir.exists():
            errors.append(f"missing split image dir: {image_dir}")
        if not label_dir.exists():
            errors.append(f"missing split label dir: {label_dir}")
        if len(list(image_dir.glob("*.jpg"))) != len(list(label_dir.glob("*.txt"))):
            errors.append(f"{split} image/label count mismatch")

    if errors:
        print("paper-v4 validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("paper-v4 validation ok")
    print("benchmark=source-key stratified random split / in-distribution")
    for split in ("train", "val", "test"):
        items = [item for item in manifest if item["split"] == split]
        boxes = sum(int(item.get("boxes", 0)) for item in items)
        sources = sorted({item["source"] for item in items})
        dist = label_distribution(dataset, split)
        base = sum(1 for item in items if is_base_transform(item.get("transform", "")))
        print(
            f"{split}: images={len(items)} base={base} aug={len(items) - base} "
            f"boxes={boxes} avg={boxes / len(items):.2f} small_lt16={dist['small_lt16_percent']:.2f}% sources={sources}"
        )
    print("train transforms:", dict(sorted(transform_counts["train"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
