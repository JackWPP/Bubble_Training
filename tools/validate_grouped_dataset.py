"""Validate the grouped YOLO dataset leakage constraints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "yolo_dataset_grouped")
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    data_yaml = dataset / "bubble.yaml"
    manifest_path = dataset / "manifest.json"
    errors: list[str] = []

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        image_dir = dataset / data[split]
        if not image_dir.exists():
            errors.append(f"Missing image directory: {image_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_group: dict[str, set[str]] = defaultdict(set)
    augmented_val_test = []
    for item in manifest:
        by_group[item["group_key"]].add(item["split"])
        transform = item.get("transform", "")
        is_base_transform = transform in {"raw", "letterbox"} or transform.startswith("tile_")
        if item["split"] in {"val", "test"} and not is_base_transform:
            augmented_val_test.append(item.get("image_name") or item.get("image") or "<unknown>")
    leaks = {group: sorted(splits) for group, splits in by_group.items() if len(splits) > 1}
    if leaks:
        errors.append(f"group_key leakage across splits: {len(leaks)}")
    if augmented_val_test:
        errors.append(f"val/test contain augmented samples: {len(augmented_val_test)}")

    if errors:
        print("dataset validation failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("dataset validation ok")
    print(f"groups={len(by_group)} images={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
