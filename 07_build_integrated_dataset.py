"""
Build an integrated YOLO dataset for bubble detection from local COCO exports.

Inputs:
  Dataset/*/annotations/instances_default.json
  Dataset/*/images/default/*

Outputs:
  yolo_dataset_integrated/{train,val,test}/{images,labels}
  yolo_dataset_integrated/bubble.yaml
  yolo_dataset_integrated/build_stats.json
  yolo_dataset_integrated/manifest.json
  DATASET_BUILD_REPORT.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TILE_SIZE = 640
TILE_STRIDE = 480
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05
GROUP_TRAIN_RATIO = 0.75
GROUP_VAL_RATIO = 0.10
GROUP_TEST_RATIO = 0.15
BALANCED_TRAIN_RATIO = 0.70
BALANCED_VAL_RATIO = 0.15
BALANCED_TEST_RATIO = 0.15
BALANCED_V3_TRAIN_RATIO = 0.60
BALANCED_V3_VAL_RATIO = 0.20
BALANCED_V3_TEST_RATIO = 0.20
PAPER_V4_TRAIN_RATIO = 0.70
PAPER_V4_VAL_RATIO = 0.15
PAPER_V4_TEST_RATIO = 0.15
SEED = 42
CLASS_ID = 0
CLASS_NAME = "bubble"
BASE_TRANSFORM_RE = re.compile(r"^(letterbox|tile_x\d+_y\d+)$")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class SourceImage:
    source: str
    image_id: str
    source_key: str
    group_key: str
    file_name: str
    path: Path
    width: int
    height: int
    boxes: list[Box]


@dataclass
class OutputSample:
    split: str
    image_name: str
    label_name: str
    source: str
    source_key: str
    group_key: str
    transform: str
    labels: list[tuple[int, float, float, float, float]]
    width: int = TILE_SIZE
    height: int = TILE_SIZE


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "dataset"


def ensure_clean_output(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    cwd = Path.cwd().resolve()
    is_allowed_dataset_dir = output_dir.parent == cwd and output_dir.name.startswith("yolo_dataset_")
    if not is_allowed_dataset_dir:
        raise ValueError(f"Refusing to clean unexpected output path: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_preview").mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_group_key(source_name: str, file_name: str) -> str:
    """Return the coarser physical source group used for leakage-safe splits."""
    stem = Path(file_name).stem
    if source_name.startswith("job_13"):
        match = re.match(r"(.+)_\d+$", stem)
        return f"{source_name}|{match.group(1) if match else stem}"
    if source_name == "bubble_fc":
        video_name = re.sub(r"_202\d.*$", "", stem, flags=re.IGNORECASE)
        return f"{source_name}|{video_name}"
    if source_name in {"big_fengchao", "bubble_pad", "bubble_1"}:
        return f"{source_name}|video_or_scene"
    if source_name in {"20+40", "60+80"}:
        return f"{source_name}|capture_series"
    return f"{source_name}|{stem}"


def find_image_path(image_dir: Path, file_name: str) -> Path | None:
    candidates = [
        image_dir / file_name,
        image_dir / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(image_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def clip_coco_box(bbox: list[float], width: int, height: int) -> Box | None:
    if len(bbox) < 4:
        return None
    x, y, w, h = map(float, bbox[:4])
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def load_coco_sources(dataset_root: Path) -> tuple[list[SourceImage], dict]:
    source_images: list[SourceImage] = []
    stats = {
        "datasets": [],
        "invalid_boxes": 0,
        "missing_images": 0,
        "empty_images": 0,
        "category_names": Counter(),
    }

    annotation_files = sorted(dataset_root.glob("*/annotations/instances_default.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No COCO annotation files found under {dataset_root}")

    for ann_path in annotation_files:
        dataset_dir = ann_path.parents[1]
        source_name = dataset_dir.name
        image_dir = dataset_dir / "images" / "default"
        data = read_json(ann_path)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = {item.get("id"): item.get("name", str(item.get("id"))) for item in data.get("categories", [])}

        anns_by_image: dict[object, list[dict]] = defaultdict(list)
        for ann in annotations:
            anns_by_image[ann.get("image_id")].append(ann)

        dataset_invalid = 0
        dataset_missing = 0
        dataset_empty = 0
        dataset_boxes = 0
        common_dims = Counter()

        for image in images:
            image_id = image.get("id")
            width = int(image.get("width", 0))
            height = int(image.get("height", 0))
            file_name = str(image.get("file_name", ""))
            if width <= 0 or height <= 0 or not file_name:
                dataset_invalid += len(anns_by_image.get(image_id, []))
                continue

            image_path = find_image_path(image_dir, file_name)
            if image_path is None:
                dataset_missing += 1
                continue

            boxes: list[Box] = []
            for ann in anns_by_image.get(image_id, []):
                category = categories.get(ann.get("category_id"), str(ann.get("category_id")))
                stats["category_names"][category] += 1
                box = clip_coco_box(ann.get("bbox", []), width, height)
                if box is None:
                    dataset_invalid += 1
                    continue
                boxes.append(box)

            if not boxes:
                dataset_empty += 1
            dataset_boxes += len(boxes)
            common_dims[(width, height)] += 1
            source_key = f"{source_name}::{image_id}::{Path(file_name).name}"
            group_key = derive_group_key(source_name, Path(file_name).name)
            source_images.append(
                SourceImage(
                    source=source_name,
                    image_id=str(image_id),
                    source_key=source_key,
                    group_key=group_key,
                    file_name=Path(file_name).name,
                    path=image_path,
                    width=width,
                    height=height,
                    boxes=boxes,
                )
            )

        stats["invalid_boxes"] += dataset_invalid
        stats["missing_images"] += dataset_missing
        stats["empty_images"] += dataset_empty
        stats["datasets"].append(
            {
                "name": source_name,
                "images": len(images),
                "loaded_images": len([item for item in source_images if item.source == source_name]),
                "annotations": len(annotations),
                "valid_boxes": dataset_boxes,
                "invalid_boxes": dataset_invalid,
                "missing_images": dataset_missing,
                "empty_images": dataset_empty,
                "common_dimensions": [
                    {"width": w, "height": h, "count": c}
                    for (w, h), c in common_dims.most_common()
                ],
            }
        )

    stats["category_names"] = dict(stats["category_names"])
    return source_images, stats


def split_sources(items: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_sources_by_group(items: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        groups[item.group_key].append(item)

    group_items = sorted(groups.items())
    if len(group_items) <= 15:
        assignments = choose_balanced_group_assignment(group_items)
    else:
        assignments = choose_greedy_group_assignment(group_items, seed)

    splits: dict[str, list[SourceImage]] = {"train": [], "val": [], "test": []}
    for group_key, group_sources in group_items:
        splits[assignments[group_key]].extend(group_sources)
    return splits


def split_sources_balanced_v2(items: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    """Split each source/group by source_key so every major source reaches train/val/test.

    This is the main paper-experiment split. It avoids the strict grouped mode's
    whole-domain holdout for sources that only have one coarse group.
    """
    rng = random.Random(seed)
    splits: dict[str, list[SourceImage]] = {"train": [], "val": [], "test": []}

    by_source: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    for source in sorted(by_source):
        by_group: dict[str, list[SourceImage]] = defaultdict(list)
        for item in by_source[source]:
            by_group[item.group_key].append(item)
        for group_key in sorted(by_group):
            group_items = sorted(by_group[group_key], key=lambda item: item.source_key)
            assigned = split_source_key_bucket(group_items, rng)
            for split, split_items in assigned.items():
                splits[split].extend(split_items)

    for split in splits:
        splits[split].sort(key=lambda item: (item.source, item.group_key, item.source_key))
    return splits


def split_sources_balanced_v3(items: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    """Split each source/group by source_key with a larger val/test share."""
    rng = random.Random(seed)
    splits: dict[str, list[SourceImage]] = {"train": [], "val": [], "test": []}
    two_key_toggle = 0

    by_source: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    for source in sorted(by_source):
        by_group: dict[str, list[SourceImage]] = defaultdict(list)
        for item in by_source[source]:
            by_group[item.group_key].append(item)
        for group_key in sorted(by_group):
            group_items = sorted(by_group[group_key], key=lambda item: item.source_key)
            assigned, two_key_toggle = split_source_key_bucket_v3(group_items, rng, two_key_toggle)
            for split, split_items in assigned.items():
                splits[split].extend(split_items)

    for split in splits:
        splits[split].sort(key=lambda item: (item.source, item.group_key, item.source_key))
    return splits


def split_sources_paper_v4(items: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    """Source-key stratified random split for in-distribution paper curves.

    This mode keeps every source_key in exactly one split while allowing
    source/group overlap across train/val/test. It is intentionally easier than
    OOD grouped evaluation and is meant for the main training curve benchmark.
    """
    rng = random.Random(seed)
    splits: dict[str, list[SourceImage]] = {"train": [], "val": [], "test": []}

    by_source: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)

    for source in sorted(by_source):
        by_key: dict[str, list[SourceImage]] = defaultdict(list)
        for item in by_source[source]:
            by_key[item.source_key].append(item)
        keys = sorted(by_key)
        rng.shuffle(keys)
        n = len(keys)
        if n == 1:
            counts = {"train": 1, "val": 0, "test": 0}
        elif n == 2:
            counts = {"train": 1, "val": 1, "test": 0}
        elif n == 3:
            counts = {"train": 1, "val": 1, "test": 1}
        else:
            n_val = max(1, round(n * PAPER_V4_VAL_RATIO))
            n_test = max(1, round(n * PAPER_V4_TEST_RATIO))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            counts = {"train": n - n_val - n_test, "val": n_val, "test": n_test}

        split_keys = {
            "train": keys[: counts["train"]],
            "val": keys[counts["train"] : counts["train"] + counts["val"]],
            "test": keys[counts["train"] + counts["val"] :],
        }
        for split in ("train", "val", "test"):
            for key in split_keys[split]:
                splits[split].extend(by_key[key])

    for split in splits:
        splits[split].sort(key=lambda item: (item.source, item.group_key, item.source_key))
    return splits


def split_source_key_bucket(items: list[SourceImage], rng: random.Random) -> dict[str, list[SourceImage]]:
    """Split a same-source/same-group bucket without source_key leakage."""
    by_key: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        by_key[item.source_key].append(item)

    keys = sorted(by_key)
    rng.shuffle(keys)
    n = len(keys)
    if n == 1:
        counts = {"train": 1, "val": 0, "test": 0}
    elif n == 2:
        counts = {"train": 1, "val": 1, "test": 0}
    elif n == 3:
        counts = {"train": 1, "val": 1, "test": 1}
    else:
        n_val = max(1, round(n * BALANCED_VAL_RATIO))
        n_test = max(1, round(n * BALANCED_TEST_RATIO))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        counts = {"train": n - n_val - n_test, "val": n_val, "test": n_test}

    split_keys = {
        "train": keys[: counts["train"]],
        "val": keys[counts["train"] : counts["train"] + counts["val"]],
        "test": keys[counts["train"] + counts["val"] :],
    }
    return {
        split: [item for key in split_keys[split] for item in by_key[key]]
        for split in ("train", "val", "test")
    }


def split_source_key_bucket_v3(
    items: list[SourceImage],
    rng: random.Random,
    two_key_toggle: int,
) -> tuple[dict[str, list[SourceImage]], int]:
    """Split a same-source/same-group bucket for balanced-v3 without source_key leakage."""
    by_key: dict[str, list[SourceImage]] = defaultdict(list)
    for item in items:
        by_key[item.source_key].append(item)

    keys = sorted(by_key)
    rng.shuffle(keys)
    n = len(keys)
    if n == 1:
        counts = {"train": 1, "val": 0, "test": 0}
    elif n == 2:
        holdout = "val" if two_key_toggle % 2 == 0 else "test"
        counts = {"train": 1, "val": 1 if holdout == "val" else 0, "test": 1 if holdout == "test" else 0}
        two_key_toggle += 1
    elif n == 3:
        counts = {"train": 1, "val": 1, "test": 1}
    else:
        n_val = max(1, round(n * BALANCED_V3_VAL_RATIO))
        n_test = max(1, round(n * BALANCED_V3_TEST_RATIO))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        counts = {"train": n - n_val - n_test, "val": n_val, "test": n_test}

    split_keys = {
        "train": keys[: counts["train"]],
        "val": keys[counts["train"] : counts["train"] + counts["val"]],
        "test": keys[counts["train"] + counts["val"] :],
    }
    return (
        {
            split: [item for key in split_keys[split] for item in by_key[key]]
            for split in ("train", "val", "test")
        },
        two_key_toggle,
    )


def choose_balanced_group_assignment(group_items: list[tuple[str, list[SourceImage]]]) -> dict[str, str]:
    split_names = ("train", "val", "test")
    total = sum(len(group_sources) for _, group_sources in group_items)
    targets = {
        "train": total * GROUP_TRAIN_RATIO,
        "val": total * GROUP_VAL_RATIO,
        "test": total * GROUP_TEST_RATIO,
    }
    best_assignment: tuple[str, ...] | None = None
    best_score: tuple[float, int, tuple[str, ...]] | None = None

    for assignment in itertools.product(split_names, repeat=len(group_items)):
        counts = Counter()
        for split, (_, group_sources) in zip(assignment, group_items):
            counts[split] += len(group_sources)
        if any(counts[split] == 0 for split in split_names):
            continue
        if counts["train"] < counts["val"] or counts["train"] < counts["test"]:
            continue

        ratio_error = sum(((counts[split] - targets[split]) / total) ** 2 for split in split_names)
        outside_band_penalty = 0.0
        val_ratio = counts["val"] / total
        test_ratio = counts["test"] / total
        if not 0.08 <= val_ratio <= 0.18:
            outside_band_penalty += min(abs(val_ratio - 0.08), abs(val_ratio - 0.18))
        if not 0.10 <= test_ratio <= 0.20:
            outside_band_penalty += min(abs(test_ratio - 0.10), abs(test_ratio - 0.20))
        group_count_imbalance = abs(assignment.count("val") - assignment.count("test"))
        score = (ratio_error + outside_band_penalty, group_count_imbalance, assignment)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise ValueError("Unable to create non-empty group-aware train/val/test split")

    return {
        group_key: split
        for split, (group_key, _) in zip(best_assignment, group_items)
    }


def choose_greedy_group_assignment(group_items: list[tuple[str, list[SourceImage]]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(group_items)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: len(item[1]), reverse=True)

    total = sum(len(group_sources) for _, group_sources in group_items)
    targets = {
        "train": total * GROUP_TRAIN_RATIO,
        "val": total * GROUP_VAL_RATIO,
        "test": total * GROUP_TEST_RATIO,
    }
    counts = Counter()
    assignments: dict[str, str] = {}
    for group_key, group_sources in shuffled:
        split = min(
            ("train", "val", "test"),
            key=lambda name: ((counts[name] + len(group_sources) - targets[name]) / total) ** 2,
        )
        assignments[group_key] = split
        counts[split] += len(group_sources)
    return assignments


def window_starts(length: int, tile_size: int = TILE_SIZE, stride: int = TILE_STRIDE) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def to_yolo(box: Box, size: int = TILE_SIZE) -> tuple[int, float, float, float, float]:
    cx = ((box.x1 + box.x2) / 2.0) / size
    cy = ((box.y1 + box.y2) / 2.0) / size
    bw = (box.x2 - box.x1) / size
    bh = (box.y2 - box.y1) / size
    return CLASS_ID, cx, cy, bw, bh


def labels_to_abs_boxes(labels: Iterable[tuple[int, float, float, float, float]], size: int = TILE_SIZE) -> list[Box]:
    boxes: list[Box] = []
    for _, cx, cy, bw, bh in labels:
        abs_cx = cx * size
        abs_cy = cy * size
        abs_w = bw * size
        abs_h = bh * size
        boxes.append(Box(abs_cx - abs_w / 2, abs_cy - abs_h / 2, abs_cx + abs_w / 2, abs_cy + abs_h / 2))
    return boxes


def process_small_image(image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    h, w = image.shape[:2]
    scale = min(TILE_SIZE / w, TILE_SIZE / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (TILE_SIZE - new_w) // 2
    pad_y = (TILE_SIZE - new_h) // 2

    canvas = np.full((TILE_SIZE, TILE_SIZE, 3), 114, dtype=np.uint8)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    labels: list[tuple[int, float, float, float, float]] = []
    for box in boxes:
        transformed = Box(
            box.x1 * scale + pad_x,
            box.y1 * scale + pad_y,
            box.x2 * scale + pad_x,
            box.y2 * scale + pad_y,
        )
        if transformed.width >= 4 and transformed.height >= 4:
            labels.append(to_yolo(transformed))
    return canvas, labels


def process_tile(
    image: np.ndarray,
    boxes: list[Box],
    start_x: int,
    start_y: int,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    h, w = image.shape[:2]
    end_x = min(start_x + TILE_SIZE, w)
    end_y = min(start_y + TILE_SIZE, h)
    canvas = np.full((TILE_SIZE, TILE_SIZE, 3), 114, dtype=np.uint8)
    crop = image[start_y:end_y, start_x:end_x]
    canvas[: crop.shape[0], : crop.shape[1]] = crop

    labels: list[tuple[int, float, float, float, float]] = []
    for box in boxes:
        center_inside = start_x <= box.cx < start_x + TILE_SIZE and start_y <= box.cy < start_y + TILE_SIZE
        if not center_inside:
            continue
        clipped = Box(
            max(box.x1, start_x) - start_x,
            max(box.y1, start_y) - start_y,
            min(box.x2, end_x) - start_x,
            min(box.y2, end_y) - start_y,
        )
        retained_ratio = clipped.area / box.area if box.area > 0 else 0.0
        if clipped.width >= 4 and clipped.height >= 4 and retained_ratio >= 0.40:
            labels.append(to_yolo(clipped))
    return canvas, labels


def write_image_and_label(
    output_dir: Path,
    split: str,
    image_name: str,
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
) -> None:
    image_path = output_dir / split / "images" / image_name
    label_path = output_dir / split / "labels" / f"{Path(image_name).stem}.txt"
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    with label_path.open("w", encoding="utf-8") as handle:
        for cls, cx, cy, bw, bh in labels:
            handle.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used_names:
        stem = Path(base).stem
        candidate = f"{stem}_{index:02d}.jpg"
        index += 1
    used_names.add(candidate)
    return candidate


def transform_hflip(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]):
    return cv2.flip(image, 1), [(cls, 1.0 - cx, cy, bw, bh) for cls, cx, cy, bw, bh in labels]


def transform_vflip(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]):
    return cv2.flip(image, 0), [(cls, cx, 1.0 - cy, bw, bh) for cls, cx, cy, bw, bh in labels]


def transform_rot90(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), [(cls, 1.0 - cy, cx, bh, bw) for cls, cx, cy, bw, bh in labels]


def transform_rot180(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]):
    return cv2.rotate(image, cv2.ROTATE_180), [(cls, 1.0 - cx, 1.0 - cy, bw, bh) for cls, cx, cy, bw, bh in labels]


def transform_rot270(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), [(cls, cy, 1.0 - cx, bh, bw) for cls, cx, cy, bw, bh in labels]


def transform_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def transform_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = image.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def transform_hsv(image: np.ndarray, h_shift: float, s_factor: float, v_factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def transform_noise(image: np.ndarray, rng: np.random.Generator, std: float = 6.0) -> np.ndarray:
    noise = rng.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def transform_photometric_low(image: np.ndarray) -> np.ndarray:
    return transform_hsv(transform_brightness(image, 0.92), -2, 0.96, 0.98)


def transform_photometric_high(image: np.ndarray) -> np.ndarray:
    return transform_hsv(transform_contrast(image, 1.08), 2, 1.04, 1.04)


def extract_crops(train_images: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]]]]) -> list[dict]:
    crops: list[dict] = []
    for image, labels in train_images:
        for cls, box in zip([label[0] for label in labels], labels_to_abs_boxes(labels)):
            x1 = max(0, int(math.floor(box.x1)))
            y1 = max(0, int(math.floor(box.y1)))
            x2 = min(TILE_SIZE, int(math.ceil(box.x2)))
            y2 = min(TILE_SIZE, int(math.ceil(box.y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            crops.append({"crop": crop, "cls": cls, "w": x2 - x1, "h": y2 - y1})
    return crops


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def transform_copy_paste(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    crops: list[dict],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    if not crops:
        return image.copy(), list(labels)

    result = image.copy()
    new_labels = list(labels)
    existing = labels_to_abs_boxes(labels)
    paste_count = rng.randint(2, 6)

    for _ in range(paste_count):
        crop_info = rng.choice(crops)
        crop = crop_info["crop"]
        scale = rng.uniform(0.85, 1.15)
        target_w = max(4, int(round(crop_info["w"] * scale)))
        target_h = max(4, int(round(crop_info["h"] * scale)))
        if target_w >= TILE_SIZE or target_h >= TILE_SIZE:
            continue
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        for _attempt in range(30):
            x = rng.randint(0, TILE_SIZE - target_w)
            y = rng.randint(0, TILE_SIZE - target_h)
            new_box = Box(x, y, x + target_w, y + target_h)
            if all(box_iou(new_box, old) <= 0.15 for old in existing):
                result[y : y + target_h, x : x + target_w] = resized
                existing.append(new_box)
                new_labels.append(to_yolo(new_box))
                break

    return result, new_labels


def clamp_labels(labels: list[tuple[int, float, float, float, float]]) -> list[tuple[int, float, float, float, float]]:
    cleaned = []
    seen = set()
    for cls, cx, cy, bw, bh in labels:
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if cls == CLASS_ID and bw > 0 and bh > 0:
            key = (cls, round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6))
            if key not in seen:
                cleaned.append((cls, cx, cy, bw, bh))
                seen.add(key)
    return cleaned


def generate_base_samples(
    splits: dict[str, list[SourceImage]],
    output_dir: Path,
    seed: int,
) -> tuple[list[OutputSample], list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], OutputSample]], dict]:
    rng = random.Random(seed)
    used_names: set[str] = set()
    manifest: list[OutputSample] = []
    train_base_images: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], OutputSample]] = []
    generation_stats = {
        "positive_windows": Counter(),
        "empty_window_candidates": Counter(),
        "kept_empty_windows": Counter(),
        "large_images_tiled": Counter(),
        "small_images_letterboxed": Counter(),
        "dropped_boxes_after_transform": 0,
    }

    for split, items in splits.items():
        empty_candidates: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], OutputSample]] = []
        positive_outputs: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], OutputSample]] = []

        for item in items:
            image = cv2.imread(str(item.path))
            if image is None:
                continue
            h, w = image.shape[:2]
            base_stem = f"{slugify(item.source)}_{slugify(Path(item.file_name).stem)}"

            if max(w, h) > TILE_SIZE:
                generation_stats["large_images_tiled"][split] += 1
                for y in window_starts(h):
                    for x in window_starts(w):
                        tile, labels = process_tile(image, item.boxes, x, y)
                        labels = clamp_labels(labels)
                        name = unique_name(f"{base_stem}_tile_x{x}_y{y}.jpg", used_names)
                        sample = OutputSample(split, name, f"{Path(name).stem}.txt", item.source, item.source_key, item.group_key, f"tile_x{x}_y{y}", labels)
                        if labels:
                            positive_outputs.append((tile, labels, sample))
                        else:
                            empty_candidates.append((tile, labels, sample))
            else:
                generation_stats["small_images_letterboxed"][split] += 1
                canvas, labels = process_small_image(image, item.boxes)
                labels = clamp_labels(labels)
                generation_stats["dropped_boxes_after_transform"] += max(0, len(item.boxes) - len(labels))
                name = unique_name(f"{base_stem}_letterbox.jpg", used_names)
                sample = OutputSample(split, name, f"{Path(name).stem}.txt", item.source, item.source_key, item.group_key, "letterbox", labels)
                if labels:
                    positive_outputs.append((canvas, labels, sample))
                else:
                    empty_candidates.append((canvas, labels, sample))

        max_empty = int(len(positive_outputs) * 0.10)
        kept_empty = rng.sample(empty_candidates, min(len(empty_candidates), max_empty)) if max_empty > 0 else []
        generation_stats["positive_windows"][split] = len(positive_outputs)
        generation_stats["empty_window_candidates"][split] = len(empty_candidates)
        generation_stats["kept_empty_windows"][split] = len(kept_empty)

        for image, labels, sample in positive_outputs + kept_empty:
            write_image_and_label(output_dir, split, sample.image_name, image, labels)
            manifest.append(sample)
            if split == "train":
                train_base_images.append((image, labels, sample))

    generation_stats = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in generation_stats.items()
    }
    return manifest, train_base_images, generation_stats


def augment_train(
    train_base_images: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], OutputSample]],
    output_dir: Path,
    seed: int,
    profile: str = "full",
    v3_profile: str = "uniform",
) -> list[OutputSample]:
    rng = random.Random(seed + 1)
    np_rng = np.random.default_rng(seed + 1)
    used_names = {path.name for path in (output_dir / "train" / "images").glob("*.jpg")}
    augmented_manifest: list[OutputSample] = []
    crops = extract_crops([(image, labels) for image, labels, _ in train_base_images])

    if profile == "balanced-v2":
        geometric_transforms = [("hflip", transform_hflip)]
        copy_paste_count = 0
    elif profile == "balanced-v3":
        geometric_transforms = [("hflip", transform_hflip)]
        copy_paste_count = 0
    elif profile == "paper-v4":
        geometric_transforms = [("hflip", transform_hflip)]
        copy_paste_count = 0
    else:
        geometric_transforms = [
            ("hflip", transform_hflip),
            ("vflip", transform_vflip),
            ("rot90", transform_rot90),
            ("rot180", transform_rot180),
            ("rot270", transform_rot270),
        ]
        copy_paste_count = 2

    for image, labels, base_sample in train_base_images:
        if not labels:
            continue
        stem = Path(base_sample.image_name).stem

        for suffix, fn in geometric_transforms:
            aug_image, aug_labels = fn(image, labels)
            aug_labels = clamp_labels(aug_labels)
            name = unique_name(f"{stem}_{suffix}.jpg", used_names)
            write_image_and_label(output_dir, "train", name, aug_image, aug_labels)
            augmented_manifest.append(
                OutputSample("train", name, f"{Path(name).stem}.txt", base_sample.source, base_sample.source_key, base_sample.group_key, suffix, aug_labels)
            )

        if profile == "paper-v4":
            photometric = []
        elif profile == "balanced-v3":
            photometric = [
                ("photometric_low", transform_photometric_low(image)),
                ("photometric_high", transform_photometric_high(image)),
                ("noise_std4", transform_noise(image, np_rng, std=4.0)),
            ]
            if v3_profile == "source-balanced":
                if base_sample.source.startswith("job_13"):
                    photometric = [("photometric_low", transform_photometric_low(image))]
                elif base_sample.source in {"bubble_1", "bubble_fc", "bubble_pad"}:
                    photometric.append(("contrast110", transform_contrast(image, 1.10)))
        else:
            photometric = [
                ("bright085", transform_brightness(image, 0.85)),
                ("bright115", transform_brightness(image, 1.15)),
                ("contrast090", transform_contrast(image, 0.90)),
                ("contrast110", transform_contrast(image, 1.10)),
                ("hsv_warm", transform_hsv(image, 4, 1.08, 1.04)),
                ("hsv_cool", transform_hsv(image, -4, 0.92, 0.96)),
                ("noise", transform_noise(image, np_rng)),
            ]
        for suffix, aug_image in photometric:
            name = unique_name(f"{stem}_{suffix}.jpg", used_names)
            write_image_and_label(output_dir, "train", name, aug_image, labels)
            augmented_manifest.append(
                OutputSample("train", name, f"{Path(name).stem}.txt", base_sample.source, base_sample.source_key, base_sample.group_key, suffix, labels)
            )

        for index in range(copy_paste_count):
            aug_image, aug_labels = transform_copy_paste(image, labels, crops, rng)
            aug_labels = clamp_labels(aug_labels)
            name = unique_name(f"{stem}_cpaste{index}.jpg", used_names)
            write_image_and_label(output_dir, "train", name, aug_image, aug_labels)
            augmented_manifest.append(
                OutputSample("train", name, f"{Path(name).stem}.txt", base_sample.source, base_sample.source_key, base_sample.group_key, f"copy_paste_{index}", aug_labels)
            )

    return augmented_manifest


def validate_dataset(output_dir: Path, manifest: list[OutputSample], enforce_group_isolation: bool = True) -> dict:
    errors: list[str] = []
    split_by_source: dict[str, set[str]] = defaultdict(set)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    label_counts = Counter()
    box_widths: list[float] = []
    box_heights: list[float] = []
    augmented_val_test: list[str] = []

    for sample in manifest:
        image_path = output_dir / sample.split / "images" / sample.image_name
        label_path = output_dir / sample.split / "labels" / sample.label_name
        if not image_path.exists():
            errors.append(f"Missing image: {image_path}")
        if not label_path.exists():
            errors.append(f"Missing label: {label_path}")
            continue
        split_by_source[sample.source_key].add(sample.split)
        split_by_group[sample.group_key].add(sample.split)
        if sample.split in {"val", "test"} and not BASE_TRANSFORM_RE.match(sample.transform):
            augmented_val_test.append(f"{sample.split}/{sample.image_name}:{sample.transform}")
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        label_counts[sample.split] += len(lines)
        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{line_no} expected 5 columns")
                continue
            try:
                cls, cx, cy, bw, bh = [float(part) for part in parts]
            except ValueError:
                errors.append(f"{label_path}:{line_no} has non-numeric values")
                continue
            if int(cls) != CLASS_ID:
                errors.append(f"{label_path}:{line_no} invalid class {cls}")
            if not all(0.0 <= value <= 1.0 for value in [cx, cy, bw, bh]):
                errors.append(f"{label_path}:{line_no} has value outside [0,1]")
            if bw <= 0 or bh <= 0:
                errors.append(f"{label_path}:{line_no} has non-positive box")
            box_widths.append(bw * TILE_SIZE)
            box_heights.append(bh * TILE_SIZE)
        if len(lines) != len(set(lines)):
            errors.append(f"{label_path} contains duplicate labels")

    leakage = {source: sorted(splits) for source, splits in split_by_source.items() if len(splits) > 1}
    if leakage:
        errors.append(f"Source images crossed splits: {len(leakage)}")
    group_leakage = {group: sorted(splits) for group, splits in split_by_group.items() if len(splits) > 1}
    if enforce_group_isolation and group_leakage:
        errors.append(f"Source groups crossed splits: {len(group_leakage)}")
    if augmented_val_test:
        errors.append(f"Validation/test contain augmented samples: {len(augmented_val_test)}")

    images_by_split = {
        split: len(list((output_dir / split / "images").glob("*.jpg")))
        for split in ["train", "val", "test"]
    }
    labels_by_split = {
        split: len(list((output_dir / split / "labels").glob("*.txt")))
        for split in ["train", "val", "test"]
    }
    for split in ["train", "val", "test"]:
        if images_by_split[split] != labels_by_split[split]:
            errors.append(f"{split} image/label count mismatch: {images_by_split[split]} vs {labels_by_split[split]}")

    return {
        "ok": not errors,
        "errors": errors[:50],
        "error_count": len(errors),
        "images_by_split": images_by_split,
        "labels_by_split": labels_by_split,
        "boxes_by_split": dict(label_counts),
        "box_width_px": summarize_numbers(box_widths),
        "box_height_px": summarize_numbers(box_heights),
        "source_split_leakage_count": len(leakage),
        "source_split_leakage_examples": dict(list(leakage.items())[:10]),
        "group_split_leakage_count": len(group_leakage),
        "group_split_leakage_examples": dict(list(group_leakage.items())[:10]),
        "augmented_val_test_count": len(augmented_val_test),
        "augmented_val_test_examples": augmented_val_test[:10],
    }


def summarize_numbers(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
    }


def summarize_label_dimensions(output_dir: Path) -> dict:
    by_split: dict[str, dict] = {}
    for split in ["train", "val", "test"]:
        widths: list[float] = []
        heights: list[float] = []
        for label_path in (output_dir / split / "labels").glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                _cls, _cx, _cy, bw, bh = [float(part) for part in parts]
                widths.append(bw * TILE_SIZE)
                heights.append(bh * TILE_SIZE)
        total = len(widths)
        small = sum(1 for width, height in zip(widths, heights) if width < 16 or height < 16)
        by_split[split] = {
            "boxes": total,
            "small_lt16": small,
            "small_lt16_percent": round(small / total * 100, 3) if total else 0.0,
            "width_px": summarize_numbers(widths),
            "height_px": summarize_numbers(heights),
        }
    return by_split


def manifest_to_json(manifest: list[OutputSample]) -> list[dict]:
    return [
        {
            "split": item.split,
            "image": item.image_name,
            "label": item.label_name,
            "source": item.source,
            "source_key": item.source_key,
            "group_key": item.group_key,
            "transform": item.transform,
            "boxes": len(item.labels),
            "width": item.width,
            "height": item.height,
        }
        for item in manifest
    ]


def write_yaml(output_dir: Path) -> None:
    yaml_text = f"""train: train/images
val: val/images
test: test/images

nc: 1
names: ["{CLASS_NAME}"]
"""
    (output_dir / "bubble.yaml").write_text(yaml_text, encoding="utf-8")


def draw_previews(output_dir: Path, manifest: list[OutputSample], seed: int) -> None:
    rng = random.Random(seed)
    for split in ["train", "val", "test"]:
        samples = [item for item in manifest if item.split == split and item.labels]
        for index, sample in enumerate(rng.sample(samples, min(3, len(samples)))):
            image_path = output_dir / split / "images" / sample.image_name
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for _cls, cx, cy, bw, bh in sample.labels:
                x1 = int((cx - bw / 2) * TILE_SIZE)
                y1 = int((cy - bh / 2) * TILE_SIZE)
                x2 = int((cx + bw / 2) * TILE_SIZE)
                y2 = int((cy + bh / 2) * TILE_SIZE)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            preview_name = f"{split}_{index}_{Path(sample.image_name).stem}_overlay.jpg"
            cv2.imwrite(str(output_dir / "quality_preview" / preview_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_report(output_dir: Path, source_stats: dict, build_stats: dict, validation: dict) -> None:
    dataset_rows = []
    for item in source_stats["datasets"]:
        common_dims = ", ".join(
            f"{dim['width']}x{dim['height']}({dim['count']})"
            for dim in item["common_dimensions"][:3]
        )
        dataset_rows.append(
            f"| {item['name']} | {item['loaded_images']} | {item['valid_boxes']} | {item['missing_images']} | {item['invalid_boxes']} | {common_dims} |"
        )

    image_counts = validation["images_by_split"]
    box_counts = validation["boxes_by_split"]
    report = f"""# 综合气泡检测数据集构建说明

## 构建概览

本数据集由 `Dataset` 目录下 7 个 COCO 导出子数据集合并生成，所有标注类别统一映射为 `bubble`，YOLO 类别编号为 `0`。输出目录为 `yolo_dataset_integrated`，训练配置文件为 `yolo_dataset_integrated/bubble.yaml`。

采用固定随机种子 `42` 按原图随机拆分：`train/val/test = 80/15/5`。所有由同一原图派生出的切片和增强样本只保留在同一个 split 中，避免同一原图跨训练、验证和测试集合。

## 数据来源统计

| 来源 | 有效图片数 | 有效标注框数 | 缺失图片数 | 无效框数 | 主要分辨率 |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(dataset_rows)}

原始类别统计：`{source_stats['category_names']}`。

## 窗口与标注处理

- 任一边大于 `640` 的图片使用 `640x640` 滑动窗口切片，stride 为 `480`，边缘自动补最后一窗以覆盖全图。
- 任一边不大于 `640` 的图片保持比例缩放，并居中 padding 到 `640x640`。
- 切片中的 bbox 需要满足：中心点落入 tile；裁剪后宽高均不少于 `4px`；保留面积不少于原框面积的 `40%`。
- 空切片作为负样本最多保留为正样本切片数的 `10%`，本次实际保留统计见 `build_stats.json`。

## 数据增强

只对 `train` 集执行离线增强，`val` 和 `test` 保持未增强。增强包括水平翻转、垂直翻转、90/180/270 度旋转、轻度亮度/对比度/HSV 调整、轻噪声和 copy-paste。copy-paste 的气泡裁剪仅来自 train split，避免验证/测试泄漏。

## 输出数据统计

| Split | 图片数 | 标注文件数 | 标注框数 |
| --- | ---: | ---: | ---: |
| train | {image_counts.get('train', 0)} | {validation['labels_by_split'].get('train', 0)} | {box_counts.get('train', 0)} |
| val | {image_counts.get('val', 0)} | {validation['labels_by_split'].get('val', 0)} | {box_counts.get('val', 0)} |
| test | {image_counts.get('test', 0)} | {validation['labels_by_split'].get('test', 0)} | {box_counts.get('test', 0)} |

构建后 bbox 宽度统计：`{validation['box_width_px']}`。

构建后 bbox 高度统计：`{validation['box_height_px']}`。

## 质量检查

- 每张输出图片都有同名 YOLO 标签文件。
- 所有标签均为 5 列，类别为 `0`，归一化坐标位于 `[0, 1]`。
- bbox 宽高均为正值。
- 同一原图派生样本没有跨 split。
- 抽检 overlay 图输出在 `yolo_dataset_integrated/quality_preview`。

校验结果：`{"PASS" if validation["ok"] else "FAIL"}`；错误数：`{validation["error_count"]}`。

## 已知限制

本次采用用户指定的随机按图拆分。由于部分图片来自视频帧或相近采样帧，随机拆分可能让相近帧分别进入 train 和 val/test，验证指标可能略高于真实跨场景泛化表现。如果后续要评估更严格的泛化能力，建议改为按来源或视频分组拆分。
"""
    report += build_distribution_balance_section(build_stats, validation)
    Path("DATASET_BUILD_REPORT.md").write_text(report, encoding="utf-8")


def build_distribution_balance_section(build_stats: dict, validation: dict) -> str:
    output_balance = build_stats.get("output_balance", {})
    source_rows = []
    for source, item in sorted(output_balance.get("source_summary", {}).items()):
        source_rows.append(
            f"| {source} | {item.get('train_images', 0)} | {item.get('val_images', 0)} | "
            f"{item.get('test_images', 0)} | {item.get('total_images', 0)} | "
            f"{item.get('total_boxes', 0)} | {item.get('image_percent', 0):.1f}% |"
        )

    split_density_rows = []
    for split in ["train", "val", "test"]:
        item = output_balance.get("split_density", {}).get(split, {})
        split_density_rows.append(
            f"| {split} | {item.get('images', 0)} | {item.get('boxes', 0)} | "
            f"{item.get('boxes_per_image', 0):.2f} |"
        )

    return f"""

## 样本分布与均衡性说明

从类别维度看，本数据集是单类别气泡检测任务，所有有效标注均统一映射为 `bubble`，因此不存在多类别检测中常见的类别长尾问题。当前更需要关注的是来源场景、目标密度和目标尺度是否覆盖充分。

从来源维度看，构建后的样本并非简单按原图数量平均分配，而是经过滑窗切片后自然提高了高分辨率、密集气泡场景的训练占比。这种分布是符合任务目标的：大图和高密度画面更容易出现小目标、遮挡、边缘截断和局部密集排列，增加这些样本的训练权重，有助于模型学习气泡检测中更困难也更有价值的场景。同时，`20+40`、`60+80`、`bubble_fc` 等不同采集条件仍被纳入训练集，能够提供背景、尺度和亮度变化上的补充。

| 来源 | train 图像数 | val 图像数 | test 图像数 | 总图像数 | 总标注框数 | 图像占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(source_rows)}

从 split 维度看，原始图片按 `80/15/5` 随机拆分后再进行派生处理，所有由同一原图生成的切片和增强样本都保留在同一 split 内。训练集数量显著大于验证集和测试集，是因为离线增强只作用于 train；val/test 保持未增强，用于更稳定地反映真实数据分布。

| Split | 图像数 | 标注框数 | 平均每图标注框数 |
| --- | ---: | ---: | ---: |
{chr(10).join(split_density_rows)}

从目标尺度看，构建后 bbox 宽度中位数约为 `{validation['box_width_px'].get('p50', 0)}` px，高度中位数约为 `{validation['box_height_px'].get('p50', 0)}` px，说明 640 滑窗策略有效避免了把 1920x1080 大图直接缩放到 640 时造成的小气泡过度压缩。整体上，该数据集更偏向“气泡密集、小目标友好、复杂场景充分”的训练集，而不是追求每个来源绝对等量的统计均匀。这个选择更贴近后续模型训练目标。

需要注意的是，由于当前按原图随机拆分，val/test 中小样本来源的覆盖不如 train 完整，尤其 `bubble_pad`、`bubble_1` 等来源没有进入验证或测试 split。因此，本数据集适合作为综合训练集和常规验证基线；若后续要严谨评估跨采集条件泛化能力，建议额外建立按来源或按视频分组的独立测试集。
"""


def write_grouped_report(output_dir: Path, source_stats: dict, build_stats: dict, validation: dict) -> None:
    dataset_rows = []
    for item in source_stats["datasets"]:
        common_dims = ", ".join(
            f"{dim['width']}x{dim['height']}({dim['count']})"
            for dim in item["common_dimensions"][:3]
        )
        dataset_rows.append(
            f"| {item['name']} | {item['loaded_images']} | {item['valid_boxes']} | "
            f"{item['missing_images']} | {item['invalid_boxes']} | {common_dims} |"
        )

    image_counts = validation["images_by_split"]
    label_counts = validation["labels_by_split"]
    box_counts = validation["boxes_by_split"]
    density = build_stats["output_balance"]["split_density"]
    split_rows = []
    for split in ["train", "val", "test"]:
        item = density.get(split, {})
        split_rows.append(
            f"| {split} | {image_counts.get(split, 0)} | {label_counts.get(split, 0)} | "
            f"{box_counts.get(split, 0)} | {item.get('base_images', 0)} | "
            f"{item.get('augmented_images', 0)} | {item.get('boxes_per_image', 0):.2f} |"
        )

    source_rows = []
    for source, item in sorted(build_stats["output_balance"]["source_summary"].items()):
        source_rows.append(
            f"| {source} | {item.get('train_images', 0)} | {item.get('val_images', 0)} | "
            f"{item.get('test_images', 0)} | {item.get('total_images', 0)} | "
            f"{item.get('total_boxes', 0)} |"
        )

    group_rows = []
    for group_key, item in sorted(build_stats["output_balance"]["group_summary"].items()):
        group_rows.append(
            f"| {group_key} | {item.get('train_images', 0)} | {item.get('val_images', 0)} | "
            f"{item.get('test_images', 0)} | {item.get('total_images', 0)} | "
            f"{item.get('total_boxes', 0)} |"
        )

    split_mode = build_stats.get("split_mode", "source")
    ratio = build_stats.get("split_ratio", {})
    report = f"""# 气泡检测 YOLO 数据集构建报告

## 构建概览

本数据集由 `Dataset` 目录中的 COCO 导出合并生成，所有标注统一映射为单类 `bubble`，YOLO 类别编号为 `0`。

- 输出目录：`{output_dir.as_posix()}`
- 训练配置：`{(output_dir / "bubble.yaml").as_posix()}`
- split 模式：`{split_mode}`
- 随机种子：`{build_stats.get("seed")}`
- 目标比例：train/val/test = {ratio.get("train")}/{ratio.get("val")}/{ratio.get("test")}

`group` 模式会先按物理源头或采集条件生成 `group_key`，再做 train/val/test 划分。这样同一视频、同一实验条件或同一采集序列不会同时进入训练集和验证/测试集，可作为论文最终泛化测试集。

## 原始来源统计

| 来源 | 有效图像数 | 有效标注框数 | 缺失图像数 | 无效框数 | 主要分辨率 |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(dataset_rows)}

原始类别统计：`{source_stats['category_names']}`

## 处理策略

- 大图使用 `640x640` 滑动窗口切片，stride 为 `480`，边缘自动补最后一窗。
- 小图保持比例缩放，并居中 padding 到 `640x640`。
- 切片内 bbox 需要满足：中心点落入 tile；裁剪后宽高不少于 `4px`；保留面积不少于原框面积的 `40%`。
- 离线增强只作用于 `train`，`val` 和 `test` 保持未增强。
- `manifest.json` 同时记录 `source_key` 和 `group_key`，用于复查精确原图泄漏与场景级泄漏。

## 输出统计

| Split | 图像数 | 标签文件数 | 标注框数 | 基础样本数 | 增强样本数 | 平均框/图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(split_rows)}

bbox 宽度统计：`{validation['box_width_px']}`

bbox 高度统计：`{validation['box_height_px']}`

## 泄漏与质量检查

- 精确原图 source_key 跨 split 数：`{validation['source_split_leakage_count']}`
- 物理源头 group_key 跨 split 数：`{validation['group_split_leakage_count']}`
- val/test 离线增强样本数：`{validation['augmented_val_test_count']}`
- 校验结果：`{"PASS" if validation["ok"] else "FAIL"}`，错误数：`{validation["error_count"]}`

## 来源覆盖

| 来源 | train 图像数 | val 图像数 | test 图像数 | 总图像数 | 总标注框数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(source_rows)}

## Group 覆盖

| group_key | train 图像数 | val 图像数 | test 图像数 | 总图像数 | 总标注框数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(group_rows)}

## 论文使用建议

`yolo_dataset_integrated` 可作为历史随机原图划分与快速开发集；`yolo_dataset_grouped` 应作为正式泛化测试集。论文中建议表述为：本文按物理源头和采集条件隔离训练、验证与测试数据，避免相邻帧、同实验条件和同源切片导致的指标虚高。
"""
    local_report = output_dir / "DATASET_BUILD_REPORT.md"
    top_level_report = Path("DATASET_GROUPED_BUILD_REPORT.md") if output_dir.name == "yolo_dataset_grouped" else Path("DATASET_BUILD_REPORT.md")
    local_report.write_text(report, encoding="utf-8")
    top_level_report.write_text(report, encoding="utf-8")


def write_balanced_v2_report(output_dir: Path, source_stats: dict, build_stats: dict, validation: dict) -> None:
    write_grouped_report(output_dir, source_stats, build_stats, validation)
    report_path = output_dir / "DATASET_BUILD_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "# 气泡检测 YOLO 数据集构建报告",
        "# 气泡检测 YOLO Balanced V2 主数据集构建报告",
        1,
    )
    report = report.replace(
        "`group` 模式会先按物理源头或采集条件生成 `group_key`，再做 train/val/test 划分。这样同一视频、同一实验条件或同一采集序列不会同时进入训练集和验证/测试集，可作为论文最终泛化测试集。",
        "`balanced-v2` 模式按 source -> group_key -> source_key 分层切分，使所有主要数据来源都覆盖 train/val/test；同一原图或同一高分辨率图切片不会跨 split。该数据集作为后续 baseline 与消融实验的主数据集；旧 `yolo_dataset_grouped` 保留为 OOD 泛化压力测试集。",
        1,
    )
    report = report.replace(
        "`yolo_dataset_integrated` 可作为历史随机原图划分与快速开发集；`yolo_dataset_grouped` 应作为正式泛化测试集。论文中建议表述为：本文按物理源头和采集条件隔离训练、验证与测试数据，避免相邻帧、同实验条件和同源切片导致的指标虚高。",
        "`yolo_dataset_balanced_v2` 作为主实验数据集，用于 baseline、SSB、GLRB、NWD 等消融；`yolo_dataset_grouped` 作为额外 OOD 泛化测试集，不用于 checkpoint 选择或主 baseline 定义。",
        1,
    )
    report_path.write_text(report, encoding="utf-8")
    Path("DATASET_BUILD_REPORT.md").write_text(report, encoding="utf-8")
    Path("DATASET_BALANCED_V2_BUILD_REPORT.md").write_text(report, encoding="utf-8")


def write_balanced_v3_report(output_dir: Path, source_stats: dict, build_stats: dict, validation: dict) -> None:
    write_grouped_report(output_dir, source_stats, build_stats, validation)
    report_path = output_dir / "DATASET_BUILD_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    report += (
        "\n## Balanced V3 Notes\n\n"
        "- split target: train/val/test = 0.60/0.20/0.20\n"
        f"- augmentation profile: `{build_stats.get('augmentation', {}).get('v3_profile', 'uniform')}`\n"
        "- source_key leakage is forbidden; group leakage is expected for this main-training split.\n"
        "- Primary detector metrics for this phase are mAP@50, precision, and recall; mAP@50-95 is kept as a localization diagnostic.\n"
    )
    report_path.write_text(report, encoding="utf-8")
    Path("DATASET_BUILD_REPORT.md").write_text(report, encoding="utf-8")
    Path("DATASET_BALANCED_V3_BUILD_REPORT.md").write_text(report, encoding="utf-8")


def write_paper_v4_report(output_dir: Path, source_stats: dict, build_stats: dict, validation: dict) -> None:
    dataset_rows = []
    for item in source_stats["datasets"]:
        common_dims = ", ".join(
            f"{dim['width']}x{dim['height']}({dim['count']})"
            for dim in item["common_dimensions"][:3]
        )
        dataset_rows.append(
            f"| {item['name']} | {item['loaded_images']} | {item['valid_boxes']} | "
            f"{item['missing_images']} | {item['invalid_boxes']} | {common_dims} |"
        )

    image_counts = validation["images_by_split"]
    label_counts = validation["labels_by_split"]
    box_counts = validation["boxes_by_split"]
    density = build_stats["output_balance"]["split_density"]
    split_rows = []
    for split in ["train", "val", "test"]:
        item = density.get(split, {})
        split_rows.append(
            f"| {split} | {image_counts.get(split, 0)} | {label_counts.get(split, 0)} | "
            f"{box_counts.get(split, 0)} | {item.get('base_images', 0)} | "
            f"{item.get('augmented_images', 0)} | {item.get('boxes_per_image', 0):.2f} |"
        )

    source_rows = []
    for source, item in sorted(build_stats["output_balance"]["source_summary"].items()):
        source_rows.append(
            f"| {source} | {item.get('train_images', 0)} | {item.get('val_images', 0)} | "
            f"{item.get('test_images', 0)} | {item.get('total_images', 0)} | "
            f"{item.get('total_boxes', 0)} |"
        )

    group_rows = []
    for group_key, item in sorted(build_stats["output_balance"]["group_summary"].items()):
        group_rows.append(
            f"| {group_key} | {item.get('train_images', 0)} | {item.get('val_images', 0)} | "
            f"{item.get('test_images', 0)} | {item.get('total_images', 0)} | "
            f"{item.get('total_boxes', 0)} |"
        )

    ratio = build_stats.get("split_ratio", {})
    report = f"""# Bubble YOLO Paper V4 Dataset Build Report

## Benchmark Definition

Paper V4 is a source-key stratified random split / in-distribution benchmark.
It is designed for the main paper training curves and baseline selection. It is
not a strict cross-domain or grouped OOD benchmark.

- output directory: `{output_dir.as_posix()}`
- data config: `{(output_dir / "bubble.yaml").as_posix()}`
- split mode: `paper-v4`
- seed: `{build_stats.get("seed")}`
- target split: train/val/test = {ratio.get("train")}/{ratio.get("val")}/{ratio.get("test")}
- train augmentation: base samples + horizontal flip only
- validation/test augmentation: none
- leakage rule: `source_key` cannot cross split
- intentional relaxation: `source` and `group_key` may cross split
- external stress test: report `yolo_dataset_grouped` separately as grouped OOD

## Raw Source Summary

| Source | Valid images | Valid boxes | Missing images | Invalid boxes | Common dimensions |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(dataset_rows)}

Raw categories: `{source_stats['category_names']}`

## Processing Strategy

- Large images are tiled with 640x640 sliding windows, stride 480.
- Small images are letterboxed into 640x640 while preserving aspect ratio.
- A retained tile box must keep its center in the tile, have width/height at
  least 4 px after clipping, and retain at least 40% of the original area.
- Offline augmentation is train-only and limited to horizontal flip.
- `manifest.json` records both `source_key` and `group_key` for leakage review.

## Output Summary

| Split | Images | Label files | Boxes | Base samples | Augmented samples | Boxes/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(split_rows)}

Box width statistics: `{validation['box_width_px']}`

Box height statistics: `{validation['box_height_px']}`

## Quality Checks

- source_key split leakage count: `{validation['source_split_leakage_count']}`
- group_key split overlap count: `{validation['group_split_leakage_count']}` (intentional)
- augmented val/test samples: `{validation['augmented_val_test_count']}`
- validation result: `{"PASS" if validation["ok"] else "FAIL"}` with `{validation["error_count"]}` errors

## Source Coverage

| Source | Train images | Val images | Test images | Total images | Total boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(source_rows)}

## Group Coverage

| group_key | Train images | Val images | Test images | Total images | Total boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(group_rows)}

## Paper Wording

Use Paper V4 as the source-key stratified in-distribution benchmark for the
main baseline curves. Do not describe it as a strict OOD split. Use
`yolo_dataset_grouped` as the external grouped OOD stress test and report it in
a separate generalization section.
"""
    report_path = output_dir / "DATASET_BUILD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    Path("DATASET_BUILD_REPORT.md").write_text(report, encoding="utf-8")
    Path("DATASET_PAPER_V4_BUILD_REPORT.md").write_text(report, encoding="utf-8")


def summarize_output_balance(manifest: list[OutputSample]) -> dict:
    source_names = sorted({item.source for item in manifest})
    source_summary = {}
    for source in source_names:
        source_items = [item for item in manifest if item.source == source]
        source_summary[source] = {
            "train_images": sum(1 for item in source_items if item.split == "train"),
            "val_images": sum(1 for item in source_items if item.split == "val"),
            "test_images": sum(1 for item in source_items if item.split == "test"),
            "total_images": len(source_items),
            "total_boxes": sum(len(item.labels) for item in source_items),
            "image_percent": len(source_items) / len(manifest) * 100 if manifest else 0.0,
        }

    group_names = sorted({item.group_key for item in manifest})
    group_summary = {}
    for group_key in group_names:
        group_items = [item for item in manifest if item.group_key == group_key]
        group_summary[group_key] = {
            "source": group_items[0].source if group_items else "",
            "train_images": sum(1 for item in group_items if item.split == "train"),
            "val_images": sum(1 for item in group_items if item.split == "val"),
            "test_images": sum(1 for item in group_items if item.split == "test"),
            "total_images": len(group_items),
            "total_boxes": sum(len(item.labels) for item in group_items),
        }

    split_density = {}
    for split in ["train", "val", "test"]:
        split_items = [item for item in manifest if item.split == split]
        boxes = sum(len(item.labels) for item in split_items)
        base_count = sum(1 for item in split_items if BASE_TRANSFORM_RE.match(item.transform))
        split_density[split] = {
            "images": len(split_items),
            "base_images": base_count,
            "augmented_images": len(split_items) - base_count,
            "boxes": boxes,
            "boxes_per_image": boxes / len(split_items) if split_items else 0.0,
        }

    return {
        "source_summary": source_summary,
        "group_summary": group_summary,
        "split_density": split_density,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build integrated bubble YOLO dataset.")
    parser.add_argument("--input", type=Path, default=Path("Dataset"))
    parser.add_argument("--output", type=Path, default=Path("yolo_dataset_grouped"))
    parser.add_argument(
        "--split-mode",
        choices=["group", "source", "balanced-v2", "balanced_v2", "balanced-v3", "balanced_v3", "paper-v4", "paper_v4"],
        default="group",
        help="group isolates coarse physical sources; source keeps legacy per-image random split; balanced-v2/v3 are main experiment splits; paper-v4 is an in-distribution curve-friendly split.",
    )
    parser.add_argument(
        "--v3-profile",
        choices=["uniform", "source-balanced"],
        default="uniform",
        help="balanced-v3 train augmentation profile.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    ensure_clean_output(output_dir)

    split_aliases = {"balanced_v2": "balanced-v2", "balanced_v3": "balanced-v3", "paper_v4": "paper-v4"}
    split_mode = split_aliases.get(args.split_mode, args.split_mode)
    sources, source_stats = load_coco_sources(input_dir)
    if split_mode == "group":
        splits = split_sources_by_group(sources, args.seed)
    elif split_mode == "balanced-v2":
        splits = split_sources_balanced_v2(sources, args.seed)
    elif split_mode == "balanced-v3":
        splits = split_sources_balanced_v3(sources, args.seed)
    elif split_mode == "paper-v4":
        splits = split_sources_paper_v4(sources, args.seed)
    else:
        splits = split_sources(sources, args.seed)
    base_manifest, train_base_images, generation_stats = generate_base_samples(splits, output_dir, args.seed)
    augmented_manifest = augment_train(train_base_images, output_dir, args.seed, profile=split_mode, v3_profile=args.v3_profile)
    manifest = base_manifest + augmented_manifest

    write_yaml(output_dir)
    validation = validate_dataset(output_dir, manifest, enforce_group_isolation=split_mode == "group")
    draw_previews(output_dir, manifest, args.seed)

    split_source_counts = {
        split: Counter(item.source for item in items)
        for split, items in splits.items()
    }
    split_group_counts = {
        split: Counter(item.group_key for item in items)
        for split, items in splits.items()
    }
    if split_mode == "group":
        split_ratio = {"train": GROUP_TRAIN_RATIO, "val": GROUP_VAL_RATIO, "test": GROUP_TEST_RATIO}
    elif split_mode == "balanced-v2":
        split_ratio = {"train": BALANCED_TRAIN_RATIO, "val": BALANCED_VAL_RATIO, "test": BALANCED_TEST_RATIO}
    elif split_mode == "balanced-v3":
        split_ratio = {"train": BALANCED_V3_TRAIN_RATIO, "val": BALANCED_V3_VAL_RATIO, "test": BALANCED_V3_TEST_RATIO}
    elif split_mode == "paper-v4":
        split_ratio = {"train": PAPER_V4_TRAIN_RATIO, "val": PAPER_V4_VAL_RATIO, "test": PAPER_V4_TEST_RATIO}
    else:
        split_ratio = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
    build_stats = {
        "seed": args.seed,
        "split_mode": split_mode,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "split_ratio": split_ratio,
        "source_images_total": len(sources),
        "source_images_by_split": {split: len(items) for split, items in splits.items()},
        "source_counts_by_split": {split: dict(counter) for split, counter in split_source_counts.items()},
        "group_counts_by_split": {split: dict(counter) for split, counter in split_group_counts.items()},
        "generation": generation_stats,
        "augmentation": {
            "train_base_samples": len(train_base_images),
            "augmented_train_samples": len(augmented_manifest),
            "v3_profile": args.v3_profile if split_mode == "balanced-v3" else "",
            "profile": "base+hflip" if split_mode == "paper-v4" else split_mode,
        },
        "box_distribution_by_split": summarize_label_dimensions(output_dir),
        "output_balance": summarize_output_balance(manifest),
        "validation": validation,
    }
    (output_dir / "build_stats.json").write_text(json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest_to_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    if split_mode == "balanced-v2":
        write_balanced_v2_report(output_dir, source_stats, build_stats, validation)
    elif split_mode == "balanced-v3":
        write_balanced_v3_report(output_dir, source_stats, build_stats, validation)
    elif split_mode == "paper-v4":
        write_paper_v4_report(output_dir, source_stats, build_stats, validation)
    else:
        write_grouped_report(output_dir, source_stats, build_stats, validation)

    print(json.dumps(build_stats, ensure_ascii=False, indent=2))
    if not validation["ok"]:
        raise SystemExit(f"Validation failed with {validation['error_count']} errors")


if __name__ == "__main__":
    main()
