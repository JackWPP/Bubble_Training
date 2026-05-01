"""Validate Balanced V2 dataset properties."""

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
    return transform in {"letterbox", "raw"} or transform.startswith("tile_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "yolo_dataset_balanced_v2")
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
    if stats.get("split_mode") != "balanced-v2":
        errors.append(f"unexpected split_mode: {stats.get('split_mode')}")

    by_source_split: dict[str, Counter] = defaultdict(Counter)
    by_source_key: dict[str, set[str]] = defaultdict(set)
    augmented_val_test = []
    for item in manifest:
        by_source_split[item["source"]][item["split"]] += 1
        by_source_key[item["source_key"]].add(item["split"])
        if item["split"] in {"val", "test"} and not is_base_transform(item.get("transform", "")):
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
        if by_source_split[source]["val"] <= 0:
            errors.append(f"{source} has no val samples")
        if by_source_split[source]["test"] <= 0:
            errors.append(f"{source} has no test samples")

    for split in ("train", "val", "test"):
        image_dir = dataset / data[split]
        if not image_dir.exists():
            errors.append(f"missing split dir: {image_dir}")

    if errors:
        print("balanced-v2 validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("balanced-v2 validation ok")
    for split in ("train", "val", "test"):
        items = [item for item in manifest if item["split"] == split]
        boxes = sum(int(item.get("boxes", 0)) for item in items)
        sources = sorted({item["source"] for item in items})
        print(f"{split}: images={len(items)} boxes={boxes} avg={boxes / len(items):.2f} sources={sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
