"""Build YOLO-seg dataset from SAM3-enhanced COCO JSON annotations.

Input:  Dataset/*/annotations/instances_default_segmented.json (bbox + polygon)
Output: segmentation/datasets/paper_v4_seg/ (YOLO-seg format with polygon labels)

Strategies match 07_build_integrated_dataset.py paper-v4:
  - Source-key stratified split (70/15/15)
  - Only horizontal flip augmentation
  - 640x640 tiles for large images, letterbox for small
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely import box as shapely_box
from shapely.geometry import Polygon, box

TILE_SIZE = 640
TILE_STRIDE = 480
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CLASS_ID = 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "paper_v4_seg"

SOURCES = [
    "20+40",
    "60+80",
    "big_fengchao",
    "bubble_1",
    "bubble_fc",
    "bubble_pad",
    "job_13_dataset_2026_04_30_19_34_23_coco 1.0",
]


@dataclass
class SourceImage:
    path: Path
    source: str
    source_key: str
    group_key: str
    file_name: str
    boxes: list = field(default_factory=list)       # list of (x1,y1,x2,y2) absolute
    polygons: list = field(default_factory=list)    # list of [[(x1,y1),...]] absolute


@dataclass
class OutputSample:
    split: str
    image_name: str
    label_name: str
    source: str
    source_key: str
    group_key: str
    transform: str
    labels: list = field(default_factory=list)   # list of (class_id, [(x1,y1),...]) normalized


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


def window_starts(size: int) -> list[int]:
    if size <= TILE_SIZE:
        return [0]
    starts = list(range(0, size - TILE_SIZE + 1, TILE_STRIDE))
    if starts[-1] + TILE_SIZE < size:
        starts.append(size - TILE_SIZE)
    return starts


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    return Polygon(poly).area


def clip_polygon_to_tile(
    poly_abs: list[tuple[float, float]],
    tile_x: int,
    tile_y: int,
) -> list[tuple[float, float]] | None:
    """Clip a polygon (absolute image coords) to a tile, return normalized coords."""
    if len(poly_abs) < 3:
        return None
    try:
        polygon = Polygon(poly_abs)
        tile = box(tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE)
        clipped = polygon.intersection(tile)
        if clipped.is_empty or clipped.area < 1.0:
            return None
        if clipped.geom_type == "Polygon":
            coords = [(pt[0] - tile_x, pt[1] - tile_y) for pt in clipped.exterior.coords[:-1]]
        elif clipped.geom_type == "MultiPolygon":
            # Take the largest sub-polygon
            biggest = max(clipped.geoms, key=lambda g: g.area)
            coords = [(pt[0] - tile_x, pt[1] - tile_y) for pt in biggest.exterior.coords[:-1]]
        else:
            return None

        if len(coords) < 3:
            return None
        # Normalize to [0, 1]
        return [(x / TILE_SIZE, y / TILE_SIZE) for x, y in coords]
    except Exception:
        return None


def bbox_center_in_tile(bbox_xyxy: tuple[float, float, float, float], tile_x: int, tile_y: int) -> bool:
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    return tile_x <= cx < tile_x + TILE_SIZE and tile_y <= cy < tile_y + TILE_SIZE


def polygon_center_in_tile(poly_abs: list[tuple[float, float]], tile_x: int, tile_y: int) -> bool:
    """Check if the polygon centroid falls within the tile."""
    if len(poly_abs) < 3:
        return False
    centroid = Polygon(poly_abs).centroid
    return tile_x <= centroid.x < tile_x + TILE_SIZE and tile_y <= centroid.y < tile_y + TILE_SIZE


def process_tile_seg(
    image: np.ndarray,
    annotations: list[dict],  # list of {"bbox_xyxy": ..., "polygon": [[x1,y1],...]}
    tile_x: int,
    tile_y: int,
) -> tuple[np.ndarray, list[tuple[int, list[tuple[float, float]]]]]:
    """Extract tile and clip polygons to tile boundaries."""
    h, w = image.shape[:2]
    end_x = min(tile_x + TILE_SIZE, w)
    end_y = min(tile_y + TILE_SIZE, h)
    canvas = np.full((TILE_SIZE, TILE_SIZE, 3), 114, dtype=np.uint8)
    crop = image[tile_y:end_y, tile_x:end_x]
    canvas[: crop.shape[0], : crop.shape[1]] = crop

    labels: list[tuple[int, list[tuple[float, float]]]] = []
    for ann in annotations:
        bbox_xyxy = ann["bbox_xyxy"]
        polygon = ann["polygon"]

        if not bbox_center_in_tile(bbox_xyxy, tile_x, tile_y):
            continue

        # Clip polygon to tile
        clipped = clip_polygon_to_tile(polygon, tile_x, tile_y)
        if clipped is None:
            continue

        # Check retained area ratio (vs original polygon area in image coords)
        orig_area = polygon_area(polygon)
        clipped_area = polygon_area([(x * TILE_SIZE, y * TILE_SIZE) for x, y in clipped])
        if orig_area > 0 and clipped_area / orig_area < 0.40:
            continue

        labels.append((CLASS_ID, clipped))

    return canvas, labels


def process_small_image_seg(
    image: np.ndarray,
    annotations: list[dict],
) -> tuple[np.ndarray, list[tuple[int, list[tuple[float, float]]]]]:
    """Letterbox small image and scale/translate polygons."""
    h, w = image.shape[:2]
    scale = TILE_SIZE / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((TILE_SIZE, TILE_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (TILE_SIZE - new_w) // 2
    pad_y = (TILE_SIZE - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    labels: list[tuple[int, list[tuple[float, float]]]] = []
    for ann in annotations:
        polygon = ann["polygon"]
        transformed = [
            (x * scale + pad_x, y * scale + pad_y)
            for x, y in polygon
        ]
        # Check transformed polygon is valid
        if len(transformed) < 3:
            continue
        poly = Polygon(transformed)
        if poly.area < 4:
            continue
        # Normalize
        normalized = [(x / TILE_SIZE, y / TILE_SIZE) for x, y in transformed]
        labels.append((CLASS_ID, normalized))

    return canvas, labels


def write_seg_label(
    output_dir: Path,
    split: str,
    image_name: str,
    image: np.ndarray,
    labels: list[tuple[int, list[tuple[float, float]]]],
) -> None:
    """Write image + YOLO-seg polygon label file."""
    image_path = output_dir / split / "images" / image_name
    label_path = output_dir / split / "labels" / f"{Path(image_name).stem}.txt"
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, points in labels:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            f.write(f"{cls_id} {coords}\n")


def unique_name(base: str, used: set[str]) -> str:
    candidate = base
    idx = 1
    while candidate in used:
        stem = Path(base).stem
        candidate = f"{stem}_{idx:02d}.jpg"
        idx += 1
    used.add(candidate)
    return candidate


def transform_hflip_seg(image: np.ndarray, labels: list[tuple[int, list[tuple[float, float]]]]) -> tuple:
    """Horizontal flip: image + polygon labels."""
    flipped = cv2.flip(image, 1)
    new_labels = []
    for cls_id, points in labels:
        new_points = [(1.0 - x, y) for x, y in points]
        new_labels.append((cls_id, new_points))
    return flipped, new_labels


def load_enhanced_coco(source_name: str) -> tuple[list[SourceImage], int]:
    """Load SAM3-enhanced COCO JSON with segmentation polygons."""
    coco_path = DATASET_DIR / source_name / "annotations" / "instances_default_segmented.json"
    if not coco_path.exists():
        # Fallback to original if segmented doesn't exist
        coco_path = DATASET_DIR / source_name / "annotations" / "instances_default.json"
        if not coco_path.exists():
            return [], 0

    with open(coco_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    source_images: list[SourceImage] = []
    poly_count = 0

    for img_info in images:
        img_path = DATASET_DIR / source_name / "images" / "default" / img_info["file_name"]
        if not img_path.exists():
            continue

        boxes_xyxy = []
        polygons_abs = []
        for ann in anns_by_image.get(img_info["id"], []):
            box = coco_bbox_to_xyxy(ann["bbox"])
            boxes_xyxy.append(box)

            # Extract polygon from segmentation
            seg = ann.get("segmentation", [])
            if seg and isinstance(seg, list) and len(seg) > 0:
                seg_data = seg[0]
                if isinstance(seg_data, list) and len(seg_data) >= 6:
                    poly = [(seg_data[i], seg_data[i + 1]) for i in range(0, len(seg_data), 2)]
                    polygons_abs.append(poly)
                    poly_count += 1
                    continue
            # No valid polygon — use bbox as polygon
            x, y = ann["bbox"][0], ann["bbox"][1]
            w, h = ann["bbox"][2], ann["bbox"][3]
            rect_poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            polygons_abs.append(rect_poly)

        source_key = f"{source_name}_{img_info['file_name']}"
        group_key = source_name

        source_images.append(SourceImage(
            path=img_path,
            source=source_name,
            source_key=source_key,
            group_key=group_key,
            file_name=img_info["file_name"],
            boxes=boxes_xyxy,
            polygons=polygons_abs,
        ))

    return source_images, poly_count


def split_by_source_key(images: list[SourceImage]) -> dict[str, list[SourceImage]]:
    """Stratified split by source_key (paper-v4 style)."""
    groups = defaultdict(list)
    for img in images:
        groups[img.source_key].append(img)

    keys = sorted(groups.keys())
    rng = random.Random(44)
    rng.shuffle(keys)

    n_train = int(len(keys) * TRAIN_RATIO)
    n_val = int(len(keys) * VAL_RATIO)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }
    for i, key in enumerate(keys):
        if i < n_train:
            splits["train"].extend(groups[key])
        elif i < n_train + n_val:
            splits["val"].extend(groups[key])
        else:
            splits["test"].extend(groups[key])

    return splits


def build_dataset(seed: int = 44):
    """Full dataset build pipeline."""
    print("=" * 60)
    print("YOLO-seg Dataset Builder")
    print("=" * 60)

    # Load all sources
    all_images: list[SourceImage] = []
    total_poly = 0
    for source in SOURCES:
        images, count = load_enhanced_coco(source)
        all_images.extend(images)
        total_poly += count
        has_seg = 0
        for img in images:
            has_seg += sum(1 for p in img.polygons if len(p) > 4 and len(p) != 4)
        print(f"  {source}: {len(images)} images, {sum(len(img.boxes) for img in images)} annots, {has_seg} with poly")

    print(f"\nTotal: {len(all_images)} images, {sum(len(img.boxes) for img in all_images)} annotations, {total_poly} with polygons")

    # Split
    splits = split_by_source_key(all_images)
    for split, items in splits.items():
        print(f"  {split}: {len(items)} images")

    # Generate tiles + letterboxed samples
    rng = random.Random(seed)
    used_names: set[str] = set()
    rng2 = random.Random(seed + 1)  # For empty tile selection
    stats = {"positive_tiles": Counter(), "empty_tiles": Counter(), "tiled": 0, "letterboxed": 0}

    for split, items in splits.items():
        # Create output dirs
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

        positive_outputs: list = []
        empty_outputs: list = []

        for item in items:
            image = cv2.imread(str(item.path))
            if image is None:
                continue
            h, w = image.shape[:2]
            base_stem = f"{slugify(item.source)}_{slugify(Path(item.file_name).stem)}"

            # Build annotation dicts for tile processing
            annotations = [{"bbox_xyxy": box, "polygon": poly}
                          for box, poly in zip(item.boxes, item.polygons)]

            if max(w, h) > TILE_SIZE:
                stats["tiled"] += 1
                for y in window_starts(h):
                    for x in window_starts(w):
                        tile, labels = process_tile_seg(image, annotations, x, y)
                        name = unique_name(f"{base_stem}_tile_x{x}_y{y}.jpg", used_names)
                        if labels:
                            positive_outputs.append((tile, labels, split, name))
                        else:
                            empty_outputs.append((tile, labels, split, name))
            else:
                stats["letterboxed"] += 1
                canvas, labels = process_small_image_seg(image, annotations)
                name = unique_name(f"{base_stem}_letterbox.jpg", used_names)
                if labels:
                    positive_outputs.append((canvas, labels, split, name))
                else:
                    empty_outputs.append((canvas, labels, split, name))

        # Keep empty tiles up to 10% of positive
        max_empty = max(1, int(len(positive_outputs) * 0.10))
        kept_empty = rng2.sample(empty_outputs, min(len(empty_outputs), max_empty)) if max_empty > 0 and empty_outputs else []

        for tile, labels, s, name in positive_outputs + kept_empty:
            write_seg_label(OUTPUT_DIR, s, name, tile, labels)

        stats["positive_tiles"][split] = len(positive_outputs)
        stats["empty_tiles"][split] = len(kept_empty)
        print(f"  {split}: {len(positive_outputs)} positive + {len(kept_empty)} empty tiles")

    # Augmentation: hflip for train only (paper-v4 style)
    train_images_dir = OUTPUT_DIR / "train" / "images"
    train_labels_dir = OUTPUT_DIR / "train" / "labels"
    aug_count = 0
    for label_path in sorted(train_labels_dir.glob("*.txt")):
        img_path = train_images_dir / f"{label_path.stem}.jpg"
        if not img_path.exists():
            continue
        with open(label_path) as f:
            lines = f.read().strip().split("\n")
        labels = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
            labels.append((cls_id, coords))

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        flipped_img, flipped_labels = transform_hflip_seg(image, labels)
        new_name = f"{label_path.stem}_hflip.jpg"
        write_seg_label(OUTPUT_DIR, "train", new_name, flipped_img, flipped_labels)
        aug_count += 1

    print(f"\n  Augmentation: {aug_count} hflip copies added to train")

    # Write dataset YAML
    yaml_path = OUTPUT_DIR / "bubble_seg.yaml"
    yaml_path.write_text(
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: [\"bubble\"]\n"
    )
    print(f"  Dataset YAML: {yaml_path}")

    # Stats
    for split in ["train", "val", "test"]:
        n_imgs = len(list((OUTPUT_DIR / split / "images").glob("*.jpg")))
        n_labels = len(list((OUTPUT_DIR / split / "labels").glob("*.txt")))
        print(f"  {split} final: {n_imgs} images, {n_labels} labels")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = Path(args.output)

    build_dataset(seed=args.seed)


if __name__ == "__main__":
    main()
