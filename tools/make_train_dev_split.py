"""Build train/dev file-list YAML from yolo_dataset_grouped train split.

The dev split is selected by source_key from non-augmented train samples. All
samples with a selected source_key are excluded from the train list so augmented
derivatives cannot leak into checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
BASE_TRANSFORMS = ("raw", "letterbox", "tile_")


def is_base_sample(item: dict) -> bool:
    transform = item.get("transform", "")
    return transform in {"raw", "letterbox"} or transform.startswith("tile_")


def image_path(dataset: Path, item: dict) -> Path:
    return dataset / item["split"] / "images" / item["image"]


def summarize(items: Iterable[dict]) -> dict:
    items = list(items)
    source_counts = Counter(item["source"] for item in items)
    group_counts = Counter(item["group_key"] for item in items)
    boxes = sum(int(item.get("boxes", 0)) for item in items)
    return {
        "images": len(items),
        "boxes": boxes,
        "avg_boxes_per_image": round(boxes / len(items), 3) if items else 0.0,
        "sources": dict(sorted(source_counts.items())),
        "groups": dict(sorted(group_counts.items())),
    }


def select_dev_keys(base_train: list[dict], ratio: float, min_images: int, seed: int) -> set[str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in base_train:
        by_source[item["source"]].append(item)

    rng = random.Random(seed)
    dev_keys: set[str] = set()
    target = max(min_images, int(round(len(base_train) * ratio)))
    for source, items in sorted(by_source.items()):
        source_target = max(1, int(round(len(items) * ratio)))
        keys = sorted({item["source_key"] for item in items})
        rng.shuffle(keys)
        selected_count = 0
        for key in keys:
            key_count = sum(1 for item in items if item["source_key"] == key)
            dev_keys.add(key)
            selected_count += key_count
            if selected_count >= source_target:
                break

    if sum(1 for item in base_train if item["source_key"] in dev_keys) < target:
        remaining = sorted({item["source_key"] for item in base_train} - dev_keys)
        rng.shuffle(remaining)
        for key in remaining:
            dev_keys.add(key)
            if sum(1 for item in base_train if item["source_key"] in dev_keys) >= target:
                break
    return dev_keys


def write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(p.as_posix() for p in images) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "yolo_dataset_grouped")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "configs" / "data")
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--min-images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    train_items = [item for item in manifest if item["split"] == "train"]
    base_train = [item for item in train_items if is_base_sample(item)]
    if not base_train:
        raise RuntimeError("No base train samples found in manifest")

    dev_keys = select_dev_keys(base_train, args.ratio, args.min_images, args.seed)
    dev_items = [item for item in base_train if item["source_key"] in dev_keys]
    train_selected = [item for item in train_items if item["source_key"] not in dev_keys]
    leaked = sorted({item["source_key"] for item in train_selected} & dev_keys)
    if leaked:
        raise RuntimeError(f"source_key leakage detected: {leaked[:5]}")

    train_list = out_dir / "bubble_train_dev_train.txt"
    dev_list = out_dir / "bubble_train_dev_val.txt"
    yaml_path = out_dir / "bubble_train_dev.yaml"
    stats_path = out_dir / "bubble_train_dev_stats.json"

    write_list(train_list, [image_path(dataset, item).resolve() for item in train_selected])
    write_list(dev_list, [image_path(dataset, item).resolve() for item in dev_items])
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {ROOT.as_posix()}",
                f"train: {train_list.resolve().as_posix()}",
                f"val: {dev_list.resolve().as_posix()}",
                f"test: {(dataset / 'test' / 'images').resolve().as_posix()}",
                "",
                "nc: 1",
                'names: ["bubble"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    official_val = [item for item in manifest if item["split"] == "val"]
    official_test = [item for item in manifest if item["split"] == "test"]
    stats = {
        "dataset": str(dataset),
        "seed": args.seed,
        "ratio": args.ratio,
        "dev_source_keys": sorted(dev_keys),
        "train_after_split": summarize(train_selected),
        "selection_dev_val": summarize(dev_items),
        "official_grouped_val": summarize(official_val),
        "official_grouped_test": summarize(official_test),
        "leakage_source_key_count": len(leaked),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {yaml_path}")
    print(f"wrote {train_list}")
    print(f"wrote {dev_list}")
    print(f"wrote {stats_path}")
    print(json.dumps({k: stats[k] for k in ("train_after_split", "selection_dev_val")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
