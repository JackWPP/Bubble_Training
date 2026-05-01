"""Validate train/dev file-list split generated from grouped train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "data" / "bubble_train_dev_stats.json")
    parser.add_argument("--train-list", type=Path, default=ROOT / "configs" / "data" / "bubble_train_dev_train.txt")
    parser.add_argument("--dev-list", type=Path, default=ROOT / "configs" / "data" / "bubble_train_dev_val.txt")
    args = parser.parse_args()

    stats_path = args.stats if args.stats.is_absolute() else ROOT / args.stats
    train_list = args.train_list if args.train_list.is_absolute() else ROOT / args.train_list
    dev_list = args.dev_list if args.dev_list.is_absolute() else ROOT / args.dev_list
    errors: list[str] = []
    for path in (stats_path, train_list, dev_list):
        if not path.exists():
            errors.append(f"missing {path}")
    if errors:
        for error in errors:
            print(error)
        return 1

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_images = {line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    dev_images = {line.strip() for line in dev_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    overlap = train_images & dev_images
    if overlap:
        errors.append(f"image list overlap: {len(overlap)}")
    if stats.get("leakage_source_key_count", 1) != 0:
        errors.append(f"source_key leakage: {stats.get('leakage_source_key_count')}")
    if not train_images or not dev_images:
        errors.append("empty train or dev list")

    if errors:
        print("train/dev split validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("train/dev split validation ok")
    print(
        "train={train} dev={dev} official_val={oval} official_test={otest}".format(
            train=stats["train_after_split"]["images"],
            dev=stats["selection_dev_val"]["images"],
            oval=stats["official_grouped_val"]["images"],
            otest=stats["official_grouped_test"]["images"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
